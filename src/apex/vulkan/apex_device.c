/* SPDX-License-Identifier: MIT */
#include "apex_device.h"
#include "apex_native_uapi.h"
#include "apex_graphics.h"
#include "apex_draw.h"
#include "drm-uapi/apex_drm.h"
#include "vk_alloc.h"
#include "vk_buffer.h"
#include "vk_buffer_view.h"
#include "vk_cmd_enqueue_entrypoints.h"
#include "vk_command_buffer.h"
#include "vk_command_pool.h"
#include "vk_common_entrypoints.h"
#include "vk_descriptor_set_layout.h"
#include "vk_device_memory.h"
#include "vk_drm_syncobj.h"
#include "vk_format.h"
#include "vk_image.h"
#include "vk_query_pool.h"
#include "vk_render_pass.h"
#include "vk_log.h"
#include "vk_physical_device.h"
#include "vk_sampler.h"
#include "wsi_common.h"
#include "util/format/u_format.h"
#include "util/log.h"
#include "util/u_debug.h"
#include "util/os_time.h"
#include <errno.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

struct apex_memory_storage {
   struct list_head link;
   struct apex_bo bo;
   unsigned refs;
};
struct apex_memory {
   struct vk_device_memory vk;
   void *data;
   struct apex_memory_storage *storage;
};
struct apex_buffer {
   struct vk_buffer vk;
   struct apex_memory *memory;
   VkDeviceSize offset;
   VkExternalMemoryHandleTypeFlags external_types;
};
struct apex_image {
   struct vk_image vk;
   struct apex_memory *memory;
   VkDeviceSize offset, size;
   struct {
      VkDeviceSize offset;
      uint32_t row_stride, slice_stride;
   } levels[13];
};
struct apex_descriptor_pool {
   struct vk_object_base base;
   struct list_head sets;
   uint32_t capacity, allocated;
   uint64_t descriptors[VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT + 1];
   uint64_t used[VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT + 1];
};
struct apex_descriptor_set {
   struct vk_object_base base;
   struct list_head link;
   struct apex_descriptor_pool *pool;
   struct apex_set_layout *layout;
   union {
      VkDescriptorBufferInfo buffer;
      VkDescriptorImageInfo image;
      VkBufferView texel;
   } descriptors[];
};
/* Immutable command binding; dispatches retain the offsets supplied at bind. */
struct apex_bound_set {
   struct apex_descriptor_set *set;
   uint32_t refs;
   uint32_t offsets[];
};
struct apex_dispatch {
   struct list_head link;
   struct apex_program *program;
   struct apex_bound_set *sets[MESA_VK_MAX_DESCRIPTOR_SETS];
   uint32_t groups;
   struct apex_dispatch_parameters parameters;
   uint8_t push[APEX_MAX_PUSH_CONSTANTS];
   /* Graphics jobs append the draw block after the dispatch parameters. */
   bool graphics;
   uint32_t draw[APEX_DRAW_WORDS];
};
struct apex_pending_dispatch {
   struct list_head link;
   struct apex_bo table;
   struct drm_apex_vm_submit args;
   uint64_t point;
};
/* Recorded device memory created at first submission: snapshots of host
 * bytes (read-only) or zero-filled draw scratch (read-write). */
struct apex_upload {
   struct list_head link;
   struct apex_bo bo;
   uint64_t reserved_va, size;
   bool scratch;
   uint8_t data[];
};
struct apex_attachment {
   struct apex_image *image;
   uint32_t level, layer;
};
struct apex_command_buffer {
   struct vk_command_buffer vk;
   struct list_head dispatches, uploads;
   struct apex_pipeline *pipeline;
   struct apex_bound_set *sets[MESA_VK_MAX_DESCRIPTOR_SETS];
   uint8_t push[APEX_MAX_PUSH_CONSTANTS];
   /* Graphics bind point. */
   struct apex_bound_set *graphics_sets[MESA_VK_MAX_DESCRIPTOR_SETS];
   uint8_t graphics_push[APEX_MAX_PUSH_CONSTANTS];
   struct apex_shader *vertex, *fragment;
   struct vk_vertex_input_state vertex_input;
   struct vk_sample_locations_state sample_locations;
   struct { uint64_t va, size; } bindings[APEX_DRAW_MAX_BINDINGS];
   struct { uint64_t va, size; uint32_t bytes; } index;
   uint64_t occlusion; /* active occlusion query slot VA */
   struct {
      VkRect2D area;
      uint32_t color_count;
      struct apex_attachment color[APEX_DRAW_MAX_COLOR], depth;
      bool has_depth, has_stencil;
   } rendering;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(apex_memory, vk.base, VkDeviceMemory, VK_OBJECT_TYPE_DEVICE_MEMORY);
VK_DEFINE_NONDISP_HANDLE_CASTS(apex_buffer, vk.base, VkBuffer, VK_OBJECT_TYPE_BUFFER);
VK_DEFINE_NONDISP_HANDLE_CASTS(apex_image, vk.base, VkImage, VK_OBJECT_TYPE_IMAGE);
VK_DEFINE_NONDISP_HANDLE_CASTS(apex_descriptor_pool, base, VkDescriptorPool, VK_OBJECT_TYPE_DESCRIPTOR_POOL);
VK_DEFINE_NONDISP_HANDLE_CASTS(apex_descriptor_set, base, VkDescriptorSet, VK_OBJECT_TYPE_DESCRIPTOR_SET);
VK_DEFINE_HANDLE_CASTS(apex_command_buffer, vk.base, VkCommandBuffer, VK_OBJECT_TYPE_COMMAND_BUFFER);
VK_DEFINE_NONDISP_HANDLE_CASTS(apex_sampler, vk.base, VkSampler, VK_OBJECT_TYPE_SAMPLER);

static int
gem_close(struct apex_device *device, uint32_t handle)
{
   struct drm_gem_close args = {.handle = handle};
   return ioctl(device->fd, DRM_IOCTL_GEM_CLOSE, &args);
}

static VkResult
check_status(struct vk_device *vk)
{
   struct apex_device *device = (void *)vk;
   struct drm_apex_vm_status status = {0};
   int ret = ioctl(device->fd, DRM_IOCTL_APEX_VM_STATUS, &status);
   if (ret || status.error)
      return vk_device_set_lost(vk, "Apex asynchronous terminal failure (ioctl errno=%d, VM error=%d)",
                                ret ? errno : 0, status.error);
   return VK_SUCCESS;
}

static VkResult
gem_create(struct apex_device *device, uint64_t size, uint32_t flags,
           uint32_t *handle, void **data)
{
   struct drm_apex_gem_create create = {.size = size, .flags = flags};
   if (ioctl(device->fd, DRM_IOCTL_APEX_GEM_CREATE, &create))
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
   struct drm_apex_gem_mmap map = {.handle = create.handle};
   if (!ioctl(device->fd, DRM_IOCTL_APEX_GEM_MMAP, &map)) {
      void *ptr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, device->fd, map.offset);
      if (ptr != MAP_FAILED) {
         *handle = create.handle;
         *data = ptr;
         return VK_SUCCESS;
      }
   }
   gem_close(device, create.handle);
   return VK_ERROR_MEMORY_MAP_FAILED;
}

void
apex_bo_finish(struct apex_device *device, struct apex_bo *bo)
{
   /* After async device loss, UNMAP could wait an unsignaled job and prevent file
    * close from cancelling it. Keep the mapping and VA reserved until close;
    * the kernel VM retains backing independently of the GEM handle. */
   if (bo->va && !(device->completion && vk_device_is_lost(&device->vk))) {
      struct drm_apex_vm_bind bind = {
         .operation = APEX_DRM_VM_BIND_UNMAP, .va = bo->va, .bytes = bo->size,
      };
      if (ioctl(device->fd, DRM_IOCTL_APEX_VM_BIND, &bind)) {
         /* Keep this address reserved until file close if unbind failed. */
         vk_device_set_lost(&device->vk, "Apex VM unbind failed");
      } else {
         mtx_lock(&device->va_mutex);
         util_vma_heap_free(&device->va_heap, bo->va, bo->size);
         mtx_unlock(&device->va_mutex);
      }
   }
   if (bo->map)
      munmap(bo->map, bo->size);
   if (bo->handle && gem_close(device, bo->handle))
      vk_device_set_lost(&device->vk, "Apex GEM close failed");
   *bo = (struct apex_bo){0};
}

/* Consumes reserved_va on both success and failure; zero allocates a new VA. */
static VkResult
bo_create(struct apex_device *device, uint64_t size, uint32_t flags,
          uint32_t gem_flags, uint64_t reserved_va, struct apex_bo *bo)
{
   bo->size = align64(size, 4096);
   uint64_t va = reserved_va;
   VkResult result = gem_create(device, bo->size, gem_flags, &bo->handle, &bo->map);
   if (result != VK_SUCCESS)
      goto fail;
   if (!va) {
      mtx_lock(&device->va_mutex);
      va = util_vma_heap_alloc(&device->va_heap, bo->size, 4096);
      mtx_unlock(&device->va_mutex);
   }
   result = VK_ERROR_OUT_OF_DEVICE_MEMORY;
   if (va) {
      struct drm_apex_vm_bind bind = {
         .operation = APEX_DRM_VM_BIND_MAP, .flags = flags, .handle = bo->handle,
         .va = va, .bytes = bo->size,
      };
      if (!ioctl(device->fd, DRM_IOCTL_APEX_VM_BIND, &bind)) {
         bo->va = va;
         return VK_SUCCESS;
      }
   }
fail:
   if (va) {
      mtx_lock(&device->va_mutex);
      util_vma_heap_free(&device->va_heap, va, bo->size);
      mtx_unlock(&device->va_mutex);
   }
   apex_bo_finish(device, bo);
   return result;
}

static VkResult
bo_transfer(struct apex_device *device, struct apex_bo *bo, uint32_t direction,
            uint64_t offset, uint64_t bytes)
{
   struct drm_apex_gem_transfer transfer = {
      .handle = bo->handle, .direction = direction, .offset = offset, .bytes = bytes,
   };
   return ioctl(device->fd, DRM_IOCTL_APEX_GEM_TRANSFER, &transfer) ?
      vk_device_set_lost(&device->vk, "Apex memory transfer failed") : VK_SUCCESS;
}

/* memory_mutex spans FD_TO_HANDLE and lookup through final GEM_CLOSE. PRIME
 * returns an existing handle without acquiring another handle reference. */
static struct apex_memory_storage *
find_memory(struct apex_device *device, uint32_t handle)
{
   list_for_each_entry(struct apex_memory_storage, storage, &device->memories, link)
      if (storage->bo.handle == handle)
         return storage;
   return NULL;
}

static uint64_t
dma_buf_size(int fd)
{
   off_t size = lseek(fd, 0, SEEK_END);
   if (size <= 0 || size > 64 * 1024 * 1024 || size % 4096 ||
       lseek(fd, 0, SEEK_SET) < 0)
      return 0;
   return size;
}

static VkResult
import_memory(struct apex_device *device, int fd, uint64_t size,
              struct apex_memory_storage **out)
{
   uint64_t extent = dma_buf_size(fd);
   if (!extent || size > extent)
      return VK_ERROR_INVALID_EXTERNAL_HANDLE;
   mtx_lock(&device->memory_mutex);
   struct drm_prime_handle prime = {.fd = fd};
   VkResult result = VK_ERROR_INVALID_EXTERNAL_HANDLE;
   if (ioctl(device->fd, DRM_IOCTL_PRIME_FD_TO_HANDLE, &prime))
      goto unlock;
   struct apex_memory_storage *storage = find_memory(device, prime.handle);
   if (storage) {
      storage->refs++;
      *out = storage;
      result = VK_SUCCESS;
      goto unlock;
   }
   storage = vk_zalloc(&device->vk.alloc, sizeof(*storage), 8, VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!storage) {
      gem_close(device, prime.handle);
      result = VK_ERROR_OUT_OF_HOST_MEMORY;
      goto unlock;
   }
   storage->bo = (struct apex_bo){.handle = prime.handle, .size = extent};
   mtx_lock(&device->va_mutex);
   uint64_t va = util_vma_heap_alloc(&device->va_heap, extent, 4096);
   mtx_unlock(&device->va_mutex);
   struct drm_apex_vm_bind bind = {
      .operation = APEX_DRM_VM_BIND_MAP, .flags = APEX_DRM_VM_READ | APEX_DRM_VM_WRITE,
      .handle = prime.handle, .va = va, .bytes = extent,
   };
   if (!va || ioctl(device->fd, DRM_IOCTL_APEX_VM_BIND, &bind)) {
      if (va) {
         mtx_lock(&device->va_mutex);
         util_vma_heap_free(&device->va_heap, va, extent);
         mtx_unlock(&device->va_mutex);
      }
      apex_bo_finish(device, &storage->bo);
      vk_free(&device->vk.alloc, storage);
      result = VK_ERROR_OUT_OF_DEVICE_MEMORY;
      goto unlock;
   }
   storage->bo.va = va;
   storage->refs = 1;
   list_addtail(&storage->link, &device->memories);
   *out = storage;
   result = VK_SUCCESS;
unlock:
   mtx_unlock(&device->memory_mutex);
   return result;
}

static VKAPI_ATTR VkResult VKAPI_CALL
apex_GetMemoryFdKHR(VkDevice dev, const VkMemoryGetFdInfoKHR *info, int *fd)
{
   VK_FROM_HANDLE(apex_device, device, dev);
   VK_FROM_HANDLE(apex_memory, memory, info->memory);
   *fd = -1;
   if (!device->prime_coherent || !(info->handleType & APEX_EXTERNAL_MEMORY_TYPES) ||
       util_bitcount(info->handleType) != 1 || !(memory->vk.export_handle_types & info->handleType))
      return VK_ERROR_INVALID_EXTERNAL_HANDLE;
   struct drm_prime_handle prime = {
      .handle = memory->storage->bo.handle, .flags = DRM_CLOEXEC | DRM_RDWR,
   };
   if (ioctl(device->fd, DRM_IOCTL_PRIME_HANDLE_TO_FD, &prime))
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   *fd = prime.fd;
   return VK_SUCCESS;
}

static VKAPI_ATTR VkResult VKAPI_CALL
apex_GetMemoryFdPropertiesKHR(VkDevice dev, VkExternalMemoryHandleTypeFlagBits type,
                            int fd, VkMemoryFdPropertiesKHR *properties)
{
   VK_FROM_HANDLE(apex_device, device, dev);
   properties->memoryTypeBits = 0;
   if (!device->prime_coherent || type != VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT ||
       !dma_buf_size(fd))
      return VK_ERROR_INVALID_EXTERNAL_HANDLE;
   mtx_lock(&device->memory_mutex);
   struct drm_prime_handle prime = {.fd = fd};
   int ret = ioctl(device->fd, DRM_IOCTL_PRIME_FD_TO_HANDLE, &prime);
   if (!ret && !find_memory(device, prime.handle))
      ret = gem_close(device, prime.handle);
   mtx_unlock(&device->memory_mutex);
   if (ret)
      return VK_ERROR_INVALID_EXTERNAL_HANDLE;
   properties->memoryTypeBits = 2; /* Foreign backing has no Vulkan CPU access. */
   return VK_SUCCESS;
}

static uint32_t
host_memory_types(const struct apex_device *device)
{
   return 1 | (device->host_coherent ? 1u << (1 + device->prime_coherent) : 0);
}

static VKAPI_ATTR VkResult VKAPI_CALL
apex_AllocateMemory(VkDevice dev, const VkMemoryAllocateInfo *info,
                    const VkAllocationCallbacks *alloc, VkDeviceMemory *out)
{
   VK_FROM_HANDLE(apex_device, device, dev);
   *out = VK_NULL_HANDLE;
   if (info->memoryTypeIndex > (unsigned)(device->prime_coherent + device->host_coherent))
      return VK_ERROR_FEATURE_NOT_PRESENT;
   const VkImportMemoryFdInfoKHR *import = vk_find_struct_const(info->pNext, IMPORT_MEMORY_FD_INFO_KHR);
   if (import && !import->handleType)
      import = NULL;
   const VkExportMemoryAllocateInfo *export = vk_find_struct_const(info->pNext, EXPORT_MEMORY_ALLOCATE_INFO);
   VkExternalMemoryHandleTypeFlags types = export ? export->handleTypes : 0;
   if ((import && (import->handleType != VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT &&
                   import->handleType != VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT)) ||
       (types & ~APEX_EXTERNAL_MEMORY_TYPES) ||
       /* Imports use the device-only type. Export also admits explicit-transfer
        * type 0: after first export shmem is canonical and transfers only wait. */
       ((import || types) && (!device->prime_coherent ||
                              (info->memoryTypeIndex != 1 && (import || info->memoryTypeIndex != 0)))))
      return VK_ERROR_INVALID_EXTERNAL_HANDLE;
   vk_foreach_struct_const(sType, ext, info->pNext) {
      if (sType != VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO &&
          sType != VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR &&
          sType != VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO &&
          sType != VK_STRUCTURE_TYPE_WSI_MEMORY_ALLOCATE_INFO_MESA)
         return VK_ERROR_FEATURE_NOT_PRESENT;
      /* Dedicated resources use ordinary storage with a zero binding offset. */
   }
   if ((!info->allocationSize && (!import || types)) || info->allocationSize > 64 * 1024 * 1024)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
   struct apex_memory *mem = vk_device_memory_create(&device->vk, info, alloc, sizeof(*mem));
   if (!mem)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   VkResult result;
   if (import) {
      result = import_memory(device, import->fd, info->allocationSize, &mem->storage);
   } else if (device->transport == APEX_TRANSPORT_DRM) {
      mem->storage = vk_zalloc(&device->vk.alloc, sizeof(*mem->storage), 8,
                               VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
      if (!mem->storage) {
         vk_device_memory_destroy(&device->vk, alloc, &mem->vk);
         return VK_ERROR_OUT_OF_HOST_MEMORY;
      }
      result = bo_create(device, info->allocationSize,
                         APEX_DRM_VM_READ | APEX_DRM_VM_WRITE,
                         device->host_coherent && info->memoryTypeIndex == 1u + device->prime_coherent ?
                            APEX_DRM_GEM_HOST_COHERENT : 0, 0, &mem->storage->bo);
      if (result == VK_SUCCESS) {
         mem->data = mem->storage->bo.map;
         mem->storage->refs = 1;
         mtx_lock(&device->memory_mutex);
         list_addtail(&mem->storage->link, &device->memories);
         mtx_unlock(&device->memory_mutex);
      } else {
         vk_free(&device->vk.alloc, mem->storage);
      }
   } else {
      mem->data = vk_zalloc2(&device->vk.alloc, alloc, info->allocationSize, 8,
                             VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
      result = mem->data ? VK_SUCCESS : VK_ERROR_OUT_OF_HOST_MEMORY;
   }
   if (result != VK_SUCCESS) {
      vk_device_memory_destroy(&device->vk, alloc, &mem->vk);
      return result;
   }
   if (import)
      close(import->fd); /* Ownership transfers only on successful allocation. */
   *out = apex_memory_to_handle(mem);
   return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL
apex_FreeMemory(VkDevice dev, VkDeviceMemory handle, const VkAllocationCallbacks *alloc)
{
   VK_FROM_HANDLE(apex_device, device, dev);
   VK_FROM_HANDLE(apex_memory, memory, handle);
   if (!memory)
      return;
   if (device->transport == APEX_TRANSPORT_DRM) {
      mtx_lock(&device->memory_mutex);
      if (!--memory->storage->refs) {
         list_del(&memory->storage->link);
         apex_bo_finish(device, &memory->storage->bo);
         vk_free(&device->vk.alloc, memory->storage);
      }
      mtx_unlock(&device->memory_mutex);
   } else {
      vk_free2(&device->vk.alloc, alloc, memory->data);
   }
   vk_device_memory_destroy(&device->vk, alloc, &memory->vk);
}

static VKAPI_ATTR VkResult VKAPI_CALL
apex_MapMemory2(VkDevice dev, const VkMemoryMapInfo *info, void **out)
{
   VK_FROM_HANDLE(apex_device, device, dev);
   VK_FROM_HANDLE(apex_memory, memory, info->memory);
   *out = NULL;
   if (!(host_memory_types(device) & (1u << memory->vk.memory_type_index)) ||
       info->flags || info->offset >= memory->vk.size ||
       (info->size != VK_WHOLE_SIZE && info->size > memory->vk.size - info->offset))
      return VK_ERROR_MEMORY_MAP_FAILED;
   *out = (uint8_t *)memory->data + info->offset;
   return VK_SUCCESS;
}

static VKAPI_ATTR VkResult VKAPI_CALL
apex_UnmapMemory2(VkDevice dev, const VkMemoryUnmapInfo *info)
{
   return info->flags ? VK_ERROR_FEATURE_NOT_PRESENT : VK_SUCCESS;
}

static VkResult
mapped_memory_ranges(VkDevice dev, uint32_t count, const VkMappedMemoryRange *ranges,
                     uint32_t direction)
{
   VK_FROM_HANDLE(apex_device, device, dev);
   for (unsigned i = 0; i < count; i++) {
      VK_FROM_HANDLE(apex_memory, memory, ranges[i].memory);
      if (!(host_memory_types(device) & (1u << memory->vk.memory_type_index)) ||
          ranges[i].offset >= memory->vk.size)
         return VK_ERROR_MEMORY_MAP_FAILED;
      uint64_t bytes = ranges[i].size == VK_WHOLE_SIZE ?
         memory->vk.size - ranges[i].offset : ranges[i].size;
      if (!bytes || bytes > memory->vk.size - ranges[i].offset)
         return VK_ERROR_MEMORY_MAP_FAILED;
      if (device->transport == APEX_TRANSPORT_DRM && !memory->vk.memory_type_index) {
         VkResult result = bo_transfer(device, &memory->storage->bo, direction, ranges[i].offset, bytes);
         if (result != VK_SUCCESS)
            return result;
      }
   }
   return VK_SUCCESS;
}

static VKAPI_ATTR VkResult VKAPI_CALL
apex_FlushMappedMemoryRanges(VkDevice dev, uint32_t count, const VkMappedMemoryRange *ranges)
{
   return mapped_memory_ranges(dev, count, ranges, APEX_DRM_TRANSFER_TO_LOCAL);
}

static VKAPI_ATTR VkResult VKAPI_CALL
apex_InvalidateMappedMemoryRanges(VkDevice dev, uint32_t count, const VkMappedMemoryRange *ranges)
{
   return mapped_memory_ranges(dev, count, ranges, APEX_DRM_TRANSFER_FROM_LOCAL);
}

static VKAPI_ATTR VkResult VKAPI_CALL
apex_CreateBuffer(VkDevice dev, const VkBufferCreateInfo *info,
                  const VkAllocationCallbacks *alloc, VkBuffer *out)
{
   VK_FROM_HANDLE(apex_device, device, dev);
   *out = VK_NULL_HANDLE;
   const VkExternalMemoryBufferCreateInfo *external =
      vk_find_struct_const(info->pNext, EXTERNAL_MEMORY_BUFFER_CREATE_INFO);
   if (external && external->handleTypes &&
       (!device->prime_coherent || (external->handleTypes & ~APEX_EXTERNAL_MEMORY_TYPES)))
      return VK_ERROR_INVALID_EXTERNAL_HANDLE;
   if (info->flags || info->sharingMode != VK_SHARING_MODE_EXCLUSIVE ||
       (vk_buffer_usage_flags(info) & ~(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                                       VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                       VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                       VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT |
                                       VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT |
                                       VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT)))
      return VK_ERROR_FEATURE_NOT_PRESENT;
   struct apex_buffer *buffer = vk_buffer_create(&device->vk, info, alloc, sizeof(*buffer));
   if (!buffer)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   buffer->external_types = external ? external->handleTypes : 0;
   *out = apex_buffer_to_handle(buffer);
   return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL
apex_DestroyBuffer(VkDevice dev, VkBuffer handle, const VkAllocationCallbacks *alloc)
{
   VK_FROM_HANDLE(apex_device, device, dev);
   VK_FROM_HANDLE(apex_buffer, buffer, handle);
   if (buffer)
      vk_buffer_destroy(&device->vk, alloc, &buffer->vk);
}

static VKAPI_ATTR void VKAPI_CALL
apex_DestroyBufferView(VkDevice dev, VkBufferView handle, const VkAllocationCallbacks *alloc)
{
   VK_FROM_HANDLE(apex_device, device, dev);
   VK_FROM_HANDLE(vk_buffer_view, view, handle);
   if (view)
      vk_buffer_view_destroy(&device->vk, alloc, view);
}

static VKAPI_ATTR void VKAPI_CALL
apex_GetDeviceBufferMemoryRequirements(VkDevice dev,
   const VkDeviceBufferMemoryRequirements *info, VkMemoryRequirements2 *out)
{
   VK_FROM_HANDLE(apex_device, device, dev);
   const VkExternalMemoryBufferCreateInfo *external =
      vk_find_struct_const(info->pCreateInfo->pNext, EXTERNAL_MEMORY_BUFFER_CREATE_INFO);
   out->memoryRequirements = (VkMemoryRequirements) {
      .size = align64(info->pCreateInfo->size, 64), .alignment = 64,
      .memoryTypeBits = external && external->handleTypes ? (device->prime_coherent ? 2 : 0) :
                         host_memory_types(device) | (device->prime_coherent ? 2 : 0),
   };
   VkMemoryDedicatedRequirements *dedicated =
      vk_find_struct(out->pNext, MEMORY_DEDICATED_REQUIREMENTS);
   if (dedicated) {
      dedicated->prefersDedicatedAllocation = VK_FALSE;
      dedicated->requiresDedicatedAllocation = VK_FALSE;
   }
}

static VKAPI_ATTR void VKAPI_CALL
apex_GetBufferMemoryRequirements2(VkDevice dev, const VkBufferMemoryRequirementsInfo2 *info,
                                 VkMemoryRequirements2 *out)
{
   VK_FROM_HANDLE(apex_buffer, buffer, info->buffer);
   vk_common_GetBufferMemoryRequirements2(dev, info, out);
   if (buffer->external_types)
      out->memoryRequirements.memoryTypeBits = 2;
}

static VKAPI_ATTR VkResult VKAPI_CALL
apex_BindBufferMemory2(VkDevice dev, uint32_t count, const VkBindBufferMemoryInfo *infos)
{
   for (unsigned i = 0; i < count; i++) {
      VK_FROM_HANDLE(apex_memory, mem, infos[i].memory);
      VK_FROM_HANDLE(apex_buffer, buffer, infos[i].buffer);
      if (buffer->external_types && mem->vk.memory_type_index != 1)
         return VK_ERROR_INVALID_EXTERNAL_HANDLE;
      if (infos[i].memoryOffset % 64 || infos[i].memoryOffset > mem->vk.size ||
          buffer->vk.size > mem->vk.size - infos[i].memoryOffset)
         return VK_ERROR_OUT_OF_DEVICE_MEMORY;
      buffer->memory = mem;
      buffer->offset = infos[i].memoryOffset;
      buffer->vk.device_address = (mem->storage ? mem->storage->bo.va : 0) + buffer->offset;
   }
   return VK_SUCCESS;
}

/* Tiling features for images; vertex-buffer features for buffers. */
VkFormatFeatureFlags2
apex_format_features(VkFormat format, bool buffer)
{
   const VkFormatFeatureFlags2 transfer = VK_FORMAT_FEATURE_2_TRANSFER_SRC_BIT |
                                          VK_FORMAT_FEATURE_2_TRANSFER_DST_BIT;
   const VkFormatFeatureFlags2 color = transfer | VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BIT;
   if (buffer) {
      const struct util_format_description *desc =
         util_format_description(vk_format_to_pipe_format(format));
      if (!desc || desc->layout != UTIL_FORMAT_LAYOUT_PLAIN || desc->block.bits > 128 ||
          util_format_is_srgb(vk_format_to_pipe_format(format)))
         return 0;
      for (unsigned c = 0; c < desc->nr_channels; c++) {
         const struct util_format_channel_description *ch = &desc->channel[c];
         if (ch->size % 8 || ch->size > 32 || ch->type == UTIL_FORMAT_TYPE_FIXED ||
             (ch->type == UTIL_FORMAT_TYPE_FLOAT && ch->size != 32))
            return 0;
      }
      uint32_t encoded[3];
      return VK_FORMAT_FEATURE_2_VERTEX_BUFFER_BIT |
             (apex_format_encode(format, VK_IMAGE_ASPECT_COLOR_BIT, NULL, encoded) ?
              VK_FORMAT_FEATURE_2_UNIFORM_TEXEL_BUFFER_BIT : 0);
   }
   VkFormatFeatureFlags2 features = 0;
   uint32_t encoded[3];
   bool depth = vk_format_has_depth(format);
   if (apex_format_encode(format, depth ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT,
                          NULL, encoded)) {
      features |= transfer | VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT;
      if (!(encoded[0] & (1u << 20)))
         features |= VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_FILTER_LINEAR_BIT |
                     VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_FILTER_MINMAX_BIT;
      if (depth)
         features |= VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_DEPTH_COMPARISON_BIT;
   }
   enum pipe_format pformat = vk_format_to_pipe_format(format);
   if (vk_format_is_depth_or_stencil(format)) {
      switch (format) {
      case VK_FORMAT_D16_UNORM: case VK_FORMAT_X8_D24_UNORM_PACK32: case VK_FORMAT_D32_SFLOAT:
      case VK_FORMAT_S8_UINT: case VK_FORMAT_D24_UNORM_S8_UINT: case VK_FORMAT_D32_SFLOAT_S8_UINT:
         features |= transfer | VK_FORMAT_FEATURE_2_DEPTH_STENCIL_ATTACHMENT_BIT;
         break;
      default:
         break;
      }
   } else if (apex_attachment_format_supported(pformat)) {
      features |= color;
      if (!util_format_is_pure_integer(pformat))
         features |= VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BLEND_BIT;
   }
   if (format == VK_FORMAT_R32_UINT)
      features |= VK_FORMAT_FEATURE_2_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_2_STORAGE_IMAGE_ATOMIC_BIT;
   if (format == VK_FORMAT_R8G8B8A8_UNORM)
      features |= VK_FORMAT_FEATURE_2_STORAGE_IMAGE_BIT;
   return features;
}

VkResult
apex_image_format_properties(const VkPhysicalDeviceImageFormatInfo2 *info,
                              VkImageFormatProperties2 *properties)
{
   properties->imageFormatProperties = (VkImageFormatProperties){0};
   VkFormatFeatureFlags2 features = apex_format_features(info->format, false);
   VkImageUsageFlags usage = 0;
   if (features & VK_FORMAT_FEATURE_2_TRANSFER_SRC_BIT) usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
   if (features & VK_FORMAT_FEATURE_2_TRANSFER_DST_BIT) usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
   if (features & VK_FORMAT_FEATURE_2_STORAGE_IMAGE_BIT) usage |= VK_IMAGE_USAGE_STORAGE_BIT;
   if (features & VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BIT) usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
   if (features & VK_FORMAT_FEATURE_2_DEPTH_STENCIL_ATTACHMENT_BIT)
      usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
   const VkImageCreateFlags flags = VK_IMAGE_CREATE_ALIAS_BIT | VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT |
      VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT | VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT |
      VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;
   if (features & VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT)
      usage |= VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;
   /* Storage and attachment paths address 2D images only. */
   if (info->type != VK_IMAGE_TYPE_2D)
      usage &= ~(VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                 VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT);
   /* Aliased or mutable images share one linear layout for identical parameters. */
   if (!features || info->type > VK_IMAGE_TYPE_3D ||
       (info->tiling != VK_IMAGE_TILING_LINEAR && info->tiling != VK_IMAGE_TILING_OPTIMAL) ||
       (info->flags & ~flags) || (info->usage & ~usage) ||
       ((info->flags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT) && info->type != VK_IMAGE_TYPE_2D))
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
   const VkPhysicalDeviceExternalImageFormatInfo *external =
      vk_find_struct_const(info->pNext, PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO);
   if (external && external->handleType)
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
   bool three_d = info->type == VK_IMAGE_TYPE_3D;
   properties->imageFormatProperties = (VkImageFormatProperties) {
      .maxExtent = {4096, info->type == VK_IMAGE_TYPE_1D ? 1 : 4096, three_d ? 2048 : 1},
      .maxMipLevels = 13, .maxArrayLayers = three_d ? 1 : 256,
      .sampleCounts = VK_SAMPLE_COUNT_1_BIT, .maxResourceSize = 64 * 1024 * 1024,
   };
   return VK_SUCCESS;
}

/* Both API tilings currently use this linear, mip-major allocation. Optimal
 * tiling keeps its layout private; linear subresources expose the same strides. */
static VkDeviceSize
image_layout(const VkImageCreateInfo *info, struct apex_image *image)
{
   VkDeviceSize size = 0;
   for (unsigned l = 0; l < info->mipLevels; l++) {
      uint32_t row = align(u_minify(info->extent.width, l) * vk_format_get_blocksize(info->format), 64);
      uint32_t slice = row * u_minify(info->extent.height, l);
      if (image) {
         image->levels[l].offset = size;
         image->levels[l].row_stride = row;
         image->levels[l].slice_stride = slice;
      }
      /* Each level holds all layers, or all depth slices of a 3D image. */
      size += (uint64_t)slice * (info->imageType == VK_IMAGE_TYPE_3D ?
                                 u_minify(info->extent.depth, l) : info->arrayLayers);
   }
   return size;
}

static VKAPI_ATTR VkResult VKAPI_CALL
apex_CreateImage(VkDevice dev, const VkImageCreateInfo *info,
                  const VkAllocationCallbacks *alloc, VkImage *out)
{
   VK_FROM_HANDLE(apex_device, device, dev);
   *out = VK_NULL_HANDLE;
   VkPhysicalDeviceImageFormatInfo2 format = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
      .format = info->format, .type = info->imageType, .tiling = info->tiling,
      .usage = info->usage, .flags = info->flags,
   };
   VkImageFormatProperties2 props = {.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2};
   if (device->transport != APEX_TRANSPORT_DRM ||
       apex_image_format_properties(&format, &props) != VK_SUCCESS ||
       info->samples != VK_SAMPLE_COUNT_1_BIT || !info->extent.width || !info->extent.height ||
       !info->extent.depth || info->extent.width > 4096 || info->extent.height > 4096 ||
       info->extent.depth > (info->imageType == VK_IMAGE_TYPE_3D ? 2048u : 1u) ||
       !info->arrayLayers || info->arrayLayers > 256 || !info->mipLevels ||
       info->mipLevels > util_logbase2(MAX3(info->extent.width, info->extent.height,
                                            info->extent.depth)) + 1)
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
   VkDeviceSize size = image_layout(info, NULL);
   if (size > props.imageFormatProperties.maxResourceSize)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
   struct apex_image *image = vk_image_create(&device->vk, info, alloc, sizeof(*image));
   if (!image)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   image->size = image_layout(info, image);
   *out = apex_image_to_handle(image);
   return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL
apex_DestroyImage(VkDevice dev, VkImage handle, const VkAllocationCallbacks *alloc)
{
   VK_FROM_HANDLE(apex_device, device, dev);
   VK_FROM_HANDLE(apex_image, image, handle);
   if (image)
      vk_image_destroy(&device->vk, alloc, &image->vk);
}

static void
image_memory_requirements(struct apex_device *device, VkDeviceSize size, VkMemoryRequirements2 *out)
{
   /* Images may also live in device-only PRIME-capable storage. */
   out->memoryRequirements = (VkMemoryRequirements) {
      .size = size, .alignment = 64,
      .memoryTypeBits = host_memory_types(device) | (device->prime_coherent ? 2 : 0),
   };
   VkMemoryDedicatedRequirements *dedicated = vk_find_struct(out->pNext, MEMORY_DEDICATED_REQUIREMENTS);
   if (dedicated) {
      dedicated->prefersDedicatedAllocation = VK_FALSE;
      dedicated->requiresDedicatedAllocation = VK_FALSE;
   }
}

static VKAPI_ATTR void VKAPI_CALL
apex_GetDeviceImageMemoryRequirements(VkDevice dev,
   const VkDeviceImageMemoryRequirements *info, VkMemoryRequirements2 *out)
{
   VK_FROM_HANDLE(apex_device, device, dev);
   image_memory_requirements(device, image_layout(info->pCreateInfo, NULL), out);
}

static VKAPI_ATTR void VKAPI_CALL
apex_GetImageMemoryRequirements2(VkDevice dev,
   const VkImageMemoryRequirementsInfo2 *info, VkMemoryRequirements2 *out)
{
   VK_FROM_HANDLE(apex_device, device, dev);
   VK_FROM_HANDLE(apex_image, image, info->image);
   image_memory_requirements(device, image->size, out);
}

static VKAPI_ATTR VkResult VKAPI_CALL
apex_BindImageMemory2(VkDevice dev, uint32_t count, const VkBindImageMemoryInfo *infos)
{
   for (unsigned i = 0; i < count; i++) {
      VK_FROM_HANDLE(apex_memory, mem, infos[i].memory);
      VK_FROM_HANDLE(apex_image, image, infos[i].image);
      if (infos[i].memoryOffset % 64 || infos[i].memoryOffset > mem->vk.size ||
          image->size > mem->vk.size - infos[i].memoryOffset)
         return VK_ERROR_OUT_OF_DEVICE_MEMORY;
      image->memory = mem;
      image->offset = infos[i].memoryOffset;
   }
   return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL
apex_GetImageSubresourceLayout(VkDevice dev, VkImage handle,
   const VkImageSubresource *subresource, VkSubresourceLayout *out)
{
   VK_FROM_HANDLE(apex_image, image, handle);
   unsigned l = subresource->mipLevel;
   *out = (VkSubresourceLayout) {
      .offset = image->levels[l].offset + (uint64_t)image->levels[l].slice_stride * subresource->arrayLayer,
      .size = image->levels[l].slice_stride,
      .rowPitch = image->levels[l].row_stride,
      .arrayPitch = image->levels[l].slice_stride,
      .depthPitch = image->levels[l].slice_stride,
   };
}

static VKAPI_ATTR VkResult VKAPI_CALL
apex_CreateImageView(VkDevice dev, const VkImageViewCreateInfo *info,
                      const VkAllocationCallbacks *alloc, VkImageView *out)
{
   VK_FROM_HANDLE(apex_device, device, dev);
   VK_FROM_HANDLE(apex_image, image, info->image);
   *out = VK_NULL_HANDLE;
   /* Views may reinterpret texels of the same size (mutable-format images). */
   if (info->flags || !image ||
       vk_format_get_blocksize(info->format) != vk_format_get_blocksize(image->vk.format))
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
   struct vk_image_view *view = vk_image_view_create(&device->vk, info, alloc, sizeof(*view));
   if (!view)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   *out = vk_image_view_to_handle(view);
   return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL
apex_DestroyImageView(VkDevice dev, VkImageView handle, const VkAllocationCallbacks *alloc)
{
   VK_FROM_HANDLE(apex_device, device, dev);
   VK_FROM_HANDLE(vk_image_view, view, handle);
   if (view)
      vk_image_view_destroy(&device->vk, alloc, view);
}

static VKAPI_ATTR VkResult VKAPI_CALL
apex_CreateSampler(VkDevice dev, const VkSamplerCreateInfo *info,
                   const VkAllocationCallbacks *alloc, VkSampler *out)
{
   VK_FROM_HANDLE(apex_device, device, dev);
   *out = VK_NULL_HANDLE;
   struct apex_sampler *sampler = vk_sampler_create(&device->vk, info, alloc, sizeof(*sampler));
   if (!sampler)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   apex_sampler_encode(info, &sampler->vk, sampler->row);
   *out = apex_sampler_to_handle(sampler);
   return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL
apex_DestroySampler(VkDevice dev, VkSampler handle, const VkAllocationCallbacks *alloc)
{
   VK_FROM_HANDLE(apex_device, device, dev);
   VK_FROM_HANDLE(vk_sampler, sampler, handle);
   if (sampler)
      vk_sampler_destroy(&device->vk, alloc, sampler);
}

static VKAPI_ATTR VkResult VKAPI_CALL
apex_CreateDescriptorSetLayout(VkDevice dev, const VkDescriptorSetLayoutCreateInfo *info,
                               const VkAllocationCallbacks *alloc, VkDescriptorSetLayout *out)
{
   VK_FROM_HANDLE(apex_device, device, dev);
   *out = VK_NULL_HANDLE;
   const VkDescriptorSetLayoutBindingFlagsCreateInfo *binding_flags =
      vk_find_struct_const(info->pNext, DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO);
   /* Buffer meta kernels have an empty push-descriptor layout. */
   VkDescriptorSetLayoutCreateFlags supported = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
   if (!info->bindingCount)
      supported |= VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT;
   if (info->flags & ~supported)
      return VK_ERROR_FEATURE_NOT_PRESENT;
   unsigned count = 0, descriptors = 0, slots = 0, immutable = 0;
   bool dynamic = false, update_after_bind = false;
   for (unsigned i = 0; i < info->bindingCount; i++) {
      const VkDescriptorSetLayoutBinding *b = &info->pBindings[i];
      VkDescriptorBindingFlags flags = binding_flags && binding_flags->bindingCount ?
         binding_flags->pBindingFlags[i] : 0;
      bool type_ok = b->descriptorType <= VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
      if (b->binding >= APEX_MAX_BINDINGS || !type_ok ||
          (uint64_t)b->descriptorCount * apex_descriptor_slots(b->descriptorType) >
             APEX_MAX_DESCRIPTORS - slots ||
          (flags & ~VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT) ||
          (device->transport == APEX_TRANSPORT_NATIVE && b->descriptorType != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER))
         return VK_ERROR_FEATURE_NOT_PRESENT;
      descriptors += b->descriptorCount;
      slots += b->descriptorCount * apex_descriptor_slots(b->descriptorType);
      if (b->pImmutableSamplers && (b->descriptorType == VK_DESCRIPTOR_TYPE_SAMPLER ||
                                    b->descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER))
         immutable += b->descriptorCount;
      count = MAX2(count, b->binding + 1);
      dynamic |= vk_descriptor_type_is_dynamic(b->descriptorType);
      update_after_bind |= flags & VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
   }
   if (dynamic && update_after_bind)
      return VK_ERROR_FEATURE_NOT_PRESENT;
   if (device->transport == APEX_TRANSPORT_NATIVE &&
       (count != 1 || descriptors != 1))
      return VK_ERROR_FEATURE_NOT_PRESENT;
   size_t size = sizeof(struct apex_set_layout) + count * sizeof(struct apex_binding_layout);
   struct apex_set_layout *layout = vk_descriptor_set_layout_zalloc(&device->vk,
      size + immutable * 8 * sizeof(uint32_t), info);
   if (!layout)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   layout->samplers = (void *)((uint8_t *)layout + size);
   layout->binding_count = count;
   layout->descriptor_count = descriptors;
   layout->slot_count = slots;
   for (unsigned b = 0; b < count; b++)
      layout->bindings[b].immutable = ~0u;
   immutable = 0;
   for (unsigned i = 0; i < info->bindingCount; i++) {
      const VkDescriptorSetLayoutBinding *binding = &info->pBindings[i];
      unsigned b = binding->binding;
      layout->bindings[b].count = binding->descriptorCount;
      layout->bindings[b].type = binding->descriptorType;
      layout->bindings[b].stages = binding->stageFlags;
      layout->counts[layout->bindings[b].type] +=
         layout->bindings[b].count;
      layout->bindings[b].flags = binding_flags && binding_flags->bindingCount ?
         binding_flags->pBindingFlags[i] : 0;
      /* Immutable samplers are copied: the layout outlives the handles. */
      if (binding->pImmutableSamplers && (binding->descriptorType == VK_DESCRIPTOR_TYPE_SAMPLER ||
                                          binding->descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER)) {
         layout->bindings[b].immutable = immutable;
         for (unsigned e = 0; e < binding->descriptorCount; e++)
            memcpy(layout->samplers[immutable++],
                   apex_sampler_from_handle(binding->pImmutableSamplers[e])->row, 8 * sizeof(uint32_t));
      }
   }
   unsigned offset = 0, slot = 0;
   for (unsigned b = 0; b < count; b++) {
      layout->bindings[b].offset = offset;
      layout->bindings[b].slot = slot;
      offset += layout->bindings[b].count;
      slot += layout->bindings[b].count * apex_descriptor_slots(layout->bindings[b].type);
      if (vk_descriptor_type_is_dynamic(layout->bindings[b].type))
         layout->vk.dynamic_descriptor_count += layout->bindings[b].count;
   }
   struct mesa_blake3 hash;
   _mesa_blake3_init(&hash);
   _mesa_blake3_update(&hash, &info->flags, sizeof(info->flags));
   _mesa_blake3_update(&hash, layout->bindings, count * sizeof(layout->bindings[0]));
   _mesa_blake3_update(&hash, layout->samplers, immutable * 8 * sizeof(uint32_t));
   _mesa_blake3_final(&hash, layout->vk.blake3);
   *out = vk_descriptor_set_layout_to_handle(&layout->vk);
   return VK_SUCCESS;
}

static VKAPI_ATTR VkResult VKAPI_CALL
apex_CreateDescriptorPool(VkDevice dev, const VkDescriptorPoolCreateInfo *info,
                          const VkAllocationCallbacks *alloc, VkDescriptorPool *out)
{
   VK_FROM_HANDLE(apex_device, device, dev);
   *out = VK_NULL_HANDLE;
   if (info->flags & ~(VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT |
                        VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT))
      return VK_ERROR_FEATURE_NOT_PRESENT;
   uint64_t descriptors[VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT + 1] = {0};
   for (unsigned i = 0; i < info->poolSizeCount; i++) {
      VkDescriptorType type = info->pPoolSizes[i].type;
      if ((unsigned)type >= ARRAY_SIZE(descriptors))
         return VK_ERROR_FEATURE_NOT_PRESENT;
      descriptors[type] += info->pPoolSizes[i].descriptorCount;
   }
   struct apex_descriptor_pool *pool = vk_object_zalloc(&device->vk, alloc,
      sizeof(*pool), VK_OBJECT_TYPE_DESCRIPTOR_POOL);
   if (!pool)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   list_inithead(&pool->sets);
   pool->capacity = info->maxSets;
   memcpy(pool->descriptors, descriptors, sizeof(descriptors));
   *out = apex_descriptor_pool_to_handle(pool);
   return VK_SUCCESS;
}

static void
free_set(struct vk_device *device, struct apex_descriptor_set *set)
{
   list_del(&set->link);
   set->pool->allocated--;
   for (unsigned type = 0; type < ARRAY_SIZE(set->layout->counts); type++)
      set->pool->used[type] -= set->layout->counts[type];
   vk_descriptor_set_layout_unref(device, &set->layout->vk);
   vk_object_free(device, NULL, set);
}

static VKAPI_ATTR VkResult VKAPI_CALL
apex_ResetDescriptorPool(VkDevice dev, VkDescriptorPool handle, VkDescriptorPoolResetFlags flags)
{
   VK_FROM_HANDLE(apex_device, device, dev);
   VK_FROM_HANDLE(apex_descriptor_pool, pool, handle);
   list_for_each_entry_safe(struct apex_descriptor_set, set, &pool->sets, link)
      free_set(&device->vk, set);
   return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL
apex_DestroyDescriptorPool(VkDevice dev, VkDescriptorPool handle, const VkAllocationCallbacks *alloc)
{
   VK_FROM_HANDLE(apex_device, device, dev);
   VK_FROM_HANDLE(apex_descriptor_pool, pool, handle);
   if (pool) {
      apex_ResetDescriptorPool(dev, handle, 0);
      vk_object_free(&device->vk, alloc, pool);
   }
}

static VKAPI_ATTR VkResult VKAPI_CALL
apex_FreeDescriptorSets(VkDevice dev, VkDescriptorPool pool, uint32_t count, const VkDescriptorSet *sets)
{
   VK_FROM_HANDLE(apex_device, device, dev);
   for (unsigned i = 0; i < count; i++) {
      VK_FROM_HANDLE(apex_descriptor_set, set, sets[i]);
      if (set)
         free_set(&device->vk, set);
   }
   return VK_SUCCESS;
}

static VKAPI_ATTR VkResult VKAPI_CALL
apex_AllocateDescriptorSets(VkDevice dev, const VkDescriptorSetAllocateInfo *info, VkDescriptorSet *out)
{
   VK_FROM_HANDLE(apex_device, device, dev);
   VK_FROM_HANDLE(apex_descriptor_pool, pool, info->descriptorPool);
   for (unsigned i = 0; i < info->descriptorSetCount; i++)
      out[i] = VK_NULL_HANDLE;
   uint64_t descriptors[VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT + 1] = {0};
   for (unsigned i = 0; i < info->descriptorSetCount; i++) {
      const struct apex_set_layout *layout =
         (const void *)vk_descriptor_set_layout_from_handle(info->pSetLayouts[i]);
      for (unsigned type = 0; type < ARRAY_SIZE(descriptors); type++)
         descriptors[type] += layout->counts[type];
   }
   for (unsigned type = 0; type < ARRAY_SIZE(descriptors); type++)
      if (descriptors[type] > pool->descriptors[type] - pool->used[type])
         return VK_ERROR_OUT_OF_POOL_MEMORY;
   if (info->descriptorSetCount > pool->capacity - pool->allocated)
      return VK_ERROR_OUT_OF_POOL_MEMORY;
   for (unsigned i = 0; i < info->descriptorSetCount; i++) {
      struct apex_set_layout *layout =
         (void *)vk_descriptor_set_layout_from_handle(info->pSetLayouts[i]);
      struct apex_descriptor_set *set = vk_object_zalloc(&device->vk, NULL,
         sizeof(*set) + layout->descriptor_count * sizeof(set->descriptors[0]), VK_OBJECT_TYPE_DESCRIPTOR_SET);
      if (!set) {
         apex_FreeDescriptorSets(dev, info->descriptorPool, i, out);
         memset(out, 0, info->descriptorSetCount * sizeof(*out));
         return VK_ERROR_OUT_OF_HOST_MEMORY;
      }
      set->pool = pool;
      set->layout = (void *)vk_descriptor_set_layout_ref(&layout->vk);
      list_addtail(&set->link, &pool->sets);
      pool->allocated++;
      for (unsigned type = 0; type < ARRAY_SIZE(descriptors); type++)
         pool->used[type] += layout->counts[type];
      out[i] = apex_descriptor_set_to_handle(set);
   }
   return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL
apex_UpdateDescriptorSets(VkDevice dev, uint32_t write_count, const VkWriteDescriptorSet *writes,
                          uint32_t copy_count, const VkCopyDescriptorSet *copies)
{
   for (unsigned i = 0; i < write_count; i++) {
      VK_FROM_HANDLE(apex_descriptor_set, set, writes[i].dstSet);
      assert(writes[i].dstBinding < set->layout->binding_count &&
             writes[i].descriptorType == set->layout->bindings[writes[i].dstBinding].type);
      unsigned start = set->layout->bindings[writes[i].dstBinding].offset + writes[i].dstArrayElement;
      assert(start + writes[i].descriptorCount <= set->layout->descriptor_count);
      for (unsigned d = 0; d < writes[i].descriptorCount; d++) {
         switch (writes[i].descriptorType) {
         case VK_DESCRIPTOR_TYPE_SAMPLER:
         case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
         case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
         case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
         case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
            set->descriptors[start + d].image = writes[i].pImageInfo[d];
            break;
         case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
         case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
            set->descriptors[start + d].texel = writes[i].pTexelBufferView[d];
            break;
         default:
            set->descriptors[start + d].buffer = writes[i].pBufferInfo[d];
            break;
         }
      }
   }
   for (unsigned i = 0; i < copy_count; i++) {
      VK_FROM_HANDLE(apex_descriptor_set, src, copies[i].srcSet);
      VK_FROM_HANDLE(apex_descriptor_set, dst, copies[i].dstSet);
      unsigned from = src->layout->bindings[copies[i].srcBinding].offset + copies[i].srcArrayElement;
      unsigned to = dst->layout->bindings[copies[i].dstBinding].offset + copies[i].dstArrayElement;
      assert(from + copies[i].descriptorCount <= src->layout->descriptor_count &&
             to + copies[i].descriptorCount <= dst->layout->descriptor_count);
      memmove(&dst->descriptors[to], &src->descriptors[from],
              copies[i].descriptorCount * sizeof(src->descriptors[0]));
   }
}

static void
bound_set_unref(struct apex_command_buffer *cmd, struct apex_bound_set *bound)
{
   if (bound && !--bound->refs)
      vk_free(&cmd->vk.pool->alloc, bound);
}

static void
clear_commands(struct apex_command_buffer *cmd)
{
   struct apex_device *device = (void *)cmd->vk.base.device;
   list_for_each_entry_safe(struct apex_upload, upload, &cmd->uploads, link) {
      list_del(&upload->link);
      if (upload->reserved_va) {
         mtx_lock(&device->va_mutex);
         util_vma_heap_free(&device->va_heap, upload->reserved_va, align64(upload->size, 4096));
         mtx_unlock(&device->va_mutex);
      }
      apex_bo_finish(device, &upload->bo);
      vk_free(&cmd->vk.pool->alloc, upload);
   }
   list_for_each_entry_safe(struct apex_dispatch, dispatch, &cmd->dispatches, link) {
      list_del(&dispatch->link);
      for (unsigned s = 0; s < ARRAY_SIZE(dispatch->sets); s++)
         bound_set_unref(cmd, dispatch->sets[s]);
      vk_free(&cmd->vk.pool->alloc, dispatch);
   }
   cmd->pipeline = NULL;
   for (unsigned s = 0; s < ARRAY_SIZE(cmd->sets); s++) {
      bound_set_unref(cmd, cmd->sets[s]);
      bound_set_unref(cmd, cmd->graphics_sets[s]);
   }
   memset(cmd->sets, 0, sizeof(cmd->sets));
   memset(cmd->graphics_sets, 0, sizeof(cmd->graphics_sets));
   cmd->vertex = cmd->fragment = NULL;
   memset(cmd->bindings, 0, sizeof(cmd->bindings));
   memset(&cmd->index, 0, sizeof(cmd->index));
   cmd->occlusion = 0;
   memset(&cmd->rendering, 0, sizeof(cmd->rendering));
   memset(cmd->push, 0, sizeof(cmd->push));
   memset(cmd->graphics_push, 0, sizeof(cmd->graphics_push));
}

static void
reset_command_buffer(struct vk_command_buffer *vk, VkCommandBufferResetFlags flags)
{
   clear_commands((struct apex_command_buffer *)vk);
   vk_command_buffer_reset(vk);
}

static void
destroy_command_buffer(struct vk_command_buffer *vk)
{
   struct vk_command_pool *pool = vk->pool;
   clear_commands((struct apex_command_buffer *)vk);
   vk_command_buffer_finish(vk);
   vk_free(&pool->alloc, vk);
}

static VkResult create_command_buffer(struct vk_command_pool *, VkCommandBufferLevel,
                                      struct vk_command_buffer **);
static const struct vk_command_buffer_ops command_ops = {
   .create = create_command_buffer, .reset = reset_command_buffer, .destroy = destroy_command_buffer,
};

static VkResult
create_command_buffer(struct vk_command_pool *pool, VkCommandBufferLevel level, struct vk_command_buffer **out)
{
   struct apex_command_buffer *cmd = vk_zalloc(&pool->alloc, sizeof(*cmd), 8,
                                              VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!cmd)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   VkResult result = vk_command_buffer_init_with_params(&cmd->vk,
      &(struct vk_command_buffer_init_params) {
         .pool = pool, .ops = &command_ops, .level = level,
         .needs_cmd_queue = level == VK_COMMAND_BUFFER_LEVEL_SECONDARY,
      });
   if (result != VK_SUCCESS) {
      vk_free(&pool->alloc, cmd);
      return result;
   }
   list_inithead(&cmd->dispatches);
   list_inithead(&cmd->uploads);
   cmd->vk.dynamic_graphics_state.vi = &cmd->vertex_input;
   cmd->vk.dynamic_graphics_state.ms.sample_locations = &cmd->sample_locations;
   *out = &cmd->vk;
   return VK_SUCCESS;
}

static VKAPI_ATTR VkResult VKAPI_CALL
apex_BeginCommandBuffer(VkCommandBuffer handle, const VkCommandBufferBeginInfo *info)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   vk_command_buffer_begin(&cmd->vk, info);
   return VK_SUCCESS;
}

static VKAPI_ATTR VkResult VKAPI_CALL
apex_EndCommandBuffer(VkCommandBuffer handle)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   return vk_command_buffer_end(&cmd->vk);
}

static VKAPI_ATTR void VKAPI_CALL
apex_CmdBindPipeline(VkCommandBuffer handle, VkPipelineBindPoint point, VkPipeline pipeline)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   if (point == VK_PIPELINE_BIND_POINT_GRAPHICS) {
      vk_common_CmdBindPipeline(handle, point, pipeline);
      return;
   }
   if (point != VK_PIPELINE_BIND_POINT_COMPUTE) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   cmd->pipeline = apex_pipeline_from_handle(pipeline);
}

static VKAPI_ATTR void VKAPI_CALL
apex_CmdBindDescriptorSets2(VkCommandBuffer handle, const VkBindDescriptorSetsInfo *info)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   uint32_t first = info->firstSet, count = info->descriptorSetCount;
   const VkDescriptorSet *sets = info->pDescriptorSets;
   const uint32_t *offsets = info->pDynamicOffsets;
   if (!(info->stageFlags & (VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_ALL_GRAPHICS)) ||
       first > MESA_VK_MAX_DESCRIPTOR_SETS || count > MESA_VK_MAX_DESCRIPTOR_SETS - first) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   unsigned required = 0;
   for (unsigned s = 0; s < count; s++)
      required += apex_descriptor_set_from_handle(sets[s])->layout->vk.dynamic_descriptor_count;
   if (required != info->dynamicOffsetCount) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   unsigned consumed = 0;
   for (unsigned s = 0; s < count; s++) {
      struct apex_descriptor_set *set = apex_descriptor_set_from_handle(sets[s]);
      struct apex_bound_set *bound = vk_zalloc(&cmd->vk.pool->alloc,
         sizeof(*bound) + set->layout->descriptor_count * sizeof(bound->offsets[0]), 8,
         VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
      if (!bound) {
         vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
         return;
      }
      bound->set = set;
      bound->refs = 1;
      /* Consume in set, binding-number, array-element order, including unused bindings. */
      for (unsigned b = 0; b < set->layout->binding_count; b++) {
         const struct apex_binding_layout *binding = &set->layout->bindings[b];
         if (!vk_descriptor_type_is_dynamic(binding->type))
            continue;
         for (unsigned d = 0; d < binding->count; d++)
            bound->offsets[binding->offset + d] = offsets[consumed++];
      }
      /* Compute and graphics bind points keep separate set bindings. */
      if (info->stageFlags & VK_SHADER_STAGE_COMPUTE_BIT) {
         bound_set_unref(cmd, cmd->sets[first + s]);
         cmd->sets[first + s] = bound;
      }
      if (info->stageFlags & VK_SHADER_STAGE_ALL_GRAPHICS) {
         if (info->stageFlags & VK_SHADER_STAGE_COMPUTE_BIT)
            bound->refs++;
         bound_set_unref(cmd, cmd->graphics_sets[first + s]);
         cmd->graphics_sets[first + s] = bound;
      }
   }
}

static VKAPI_ATTR void VKAPI_CALL
apex_CmdPushConstants2(VkCommandBuffer handle, const VkPushConstantsInfo *info)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   uint32_t offset = info->offset, size = info->size;
   if (offset % 4 || size % 4 ||
       !size || offset >= APEX_MAX_PUSH_CONSTANTS || size > APEX_MAX_PUSH_CONSTANTS - offset) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   /* Compute and graphics bind points keep separate push images. */
   if (info->stageFlags & VK_SHADER_STAGE_COMPUTE_BIT)
      memcpy(cmd->push + offset, info->pValues, size);
   if (info->stageFlags & VK_SHADER_STAGE_ALL_GRAPHICS)
      memcpy(cmd->graphics_push + offset, info->pValues, size);
}

static void
push_compute(struct apex_command_buffer *cmd, struct apex_dispatch_parameters parameters,
             uint32_t groups)
{
   struct apex_dispatch *dispatch = vk_alloc(&cmd->vk.pool->alloc, sizeof(*dispatch), 8,
                                             VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!dispatch) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
      return;
   }
   *dispatch = (struct apex_dispatch) {
      .program = &cmd->pipeline->program, .groups = groups, .parameters = parameters,
   };
   memcpy(dispatch->sets, cmd->sets, sizeof(dispatch->sets));
   memcpy(dispatch->push, cmd->push, sizeof(dispatch->push));
   for (unsigned s = 0; s < ARRAY_SIZE(dispatch->sets); s++)
      if (dispatch->sets[s])
         dispatch->sets[s]->refs++;
   list_addtail(&dispatch->link, &cmd->dispatches);
}

static VKAPI_ATTR void VKAPI_CALL
apex_CmdDispatch(VkCommandBuffer handle, uint32_t x, uint32_t y, uint32_t z)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   if (!x || !y || !z)
      return;
   if (x > 65535 || y > 65535 || z > 65535 || !cmd->pipeline ||
       (!cmd->pipeline->program.table && (!cmd->sets[0] || x > 1024 || y != 1 || z != 1))) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   /* Jobs cover linear workgroup ranges bounded by the queue count and the
    * padded private-storage capacity; each workgroup runs once. */
   uint64_t total = (uint64_t)x * y * z;
   if (total > UINT32_MAX) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   uint32_t limit = cmd->pipeline->program.max_workgroups;
   for (uint32_t first = 0; first < total; first += limit) {
      uint32_t count = MIN2(total - first, limit);
      push_compute(cmd, (struct apex_dispatch_parameters) {
         .base = {first, first + count, count}, .groups = {x, y, z}}, count);
   }
}

/* The grid comes from the buffer when the job runs: a bounded launch strides
 * through every workgroup. */
static VKAPI_ATTR void VKAPI_CALL
apex_CmdDispatchIndirect(VkCommandBuffer handle, VkBuffer buffer, VkDeviceSize offset)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   VK_FROM_HANDLE(apex_buffer, args, buffer);
   if (!cmd->pipeline || !cmd->pipeline->program.table) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   uint64_t va = args->vk.device_address + offset;
   uint32_t count = MIN2(cmd->pipeline->program.max_workgroups, 1024);
   push_compute(cmd, (struct apex_dispatch_parameters) {
      .base = {0, 0, count}, .indirect = {va, va >> 32}}, count);
}

void
apex_cmd_bind_shaders(struct vk_command_buffer *vk, uint32_t count,
                      const mesa_shader_stage *stages, struct vk_shader **const shaders)
{
   struct apex_command_buffer *cmd = (void *)vk;
   for (uint32_t i = 0; i < count; i++) {
      struct apex_shader *shader = shaders[i] ? container_of(shaders[i], struct apex_shader, vk) : NULL;
      if (stages[i] == MESA_SHADER_VERTEX)
         cmd->vertex = shader;
      else if (stages[i] == MESA_SHADER_FRAGMENT)
         cmd->fragment = shader;
   }
}

static uint64_t image_address(const struct apex_image *image, unsigned level, unsigned layer,
                              VkOffset3D offset);

/* Reserves GPUVA for zero-filled read-write memory created at submission. */
static uint64_t
record_scratch(struct apex_command_buffer *cmd, uint64_t size)
{
   struct apex_device *device = (void *)cmd->vk.base.device;
   struct apex_upload *upload = vk_zalloc(&cmd->vk.pool->alloc, sizeof(*upload), 8,
                                         VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!upload) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
      return 0;
   }
   upload->size = MAX2(size, 4);
   upload->scratch = true;
   mtx_lock(&device->va_mutex);
   upload->reserved_va = util_vma_heap_alloc(&device->va_heap, align64(upload->size, 4096), 4096);
   mtx_unlock(&device->va_mutex);
   if (!upload->reserved_va) {
      vk_free(&cmd->vk.pool->alloc, upload);
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return 0;
   }
   list_addtail(&upload->link, &cmd->uploads);
   return upload->reserved_va;
}

static struct apex_attachment
attachment(const VkRenderingAttachmentInfo *info)
{
   VK_FROM_HANDLE(vk_image_view, view, info ? info->imageView : VK_NULL_HANDLE);
   if (!view)
      return (struct apex_attachment){0};
   return (struct apex_attachment) {
      .image = (struct apex_image *)view->image,
      .level = view->base_mip_level, .layer = view->base_array_layer,
   };
}

static uint64_t
attachment_address(const struct apex_attachment *a)
{
   return image_address(a->image, a->level, a->layer, (VkOffset3D){0});
}

/* Clear texels: pattern words 0-3, write mask words 4-7. */
static void record_clear(struct apex_command_buffer *cmd, const struct apex_image *image,
                         unsigned level, unsigned layer, unsigned layers, VkRect2D rect,
                         const uint32_t texel[8]);
static void pack_color(VkFormat format, const VkClearColorValue *color, uint32_t texel[8]);
static void pack_depth_stencil(VkFormat format, VkImageAspectFlags aspects,
                               const VkClearDepthStencilValue *value, uint32_t texel[8]);

static VKAPI_ATTR void VKAPI_CALL
apex_CmdBeginRendering(VkCommandBuffer handle, const VkRenderingInfo *info)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   struct apex_attachment depth = attachment(info->pDepthAttachment);
   struct apex_attachment stencil = attachment(info->pStencilAttachment);
   if (info->layerCount > 1 || info->viewMask || info->colorAttachmentCount > APEX_DRAW_MAX_COLOR ||
       (depth.image && stencil.image && memcmp(&depth, &stencil, sizeof(depth)))) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   memset(&cmd->rendering, 0, sizeof(cmd->rendering));
   cmd->rendering.area = info->renderArea;
   cmd->rendering.color_count = info->colorAttachmentCount;
   /* RESUMING continues an earlier suspended pass without load operations. */
   bool resuming = info->flags & VK_RENDERING_RESUMING_BIT;
   for (unsigned k = 0; k < info->colorAttachmentCount; k++) {
      const VkRenderingAttachmentInfo *a = &info->pColorAttachments[k];
      cmd->rendering.color[k] = attachment(a);
      if (!cmd->rendering.color[k].image || resuming || a->loadOp != VK_ATTACHMENT_LOAD_OP_CLEAR)
         continue;
      uint32_t texel[8];
      pack_color(cmd->rendering.color[k].image->vk.format, &a->clearValue.color, texel);
      record_clear(cmd, cmd->rendering.color[k].image, cmd->rendering.color[k].level,
                   cmd->rendering.color[k].layer, 1, info->renderArea, texel);
   }
   /* One depth/stencil target: the depth view, else the stencil view. */
   cmd->rendering.depth = depth.image ? depth : stencil;
   cmd->rendering.has_depth = depth.image;
   cmd->rendering.has_stencil = stencil.image;
   VkImageAspectFlags clear = 0;
   VkClearDepthStencilValue value = {0};
   if (depth.image && info->pDepthAttachment->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) {
      clear |= VK_IMAGE_ASPECT_DEPTH_BIT;
      value.depth = info->pDepthAttachment->clearValue.depthStencil.depth;
   }
   if (stencil.image && info->pStencilAttachment->loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) {
      clear |= VK_IMAGE_ASPECT_STENCIL_BIT;
      value.stencil = info->pStencilAttachment->clearValue.depthStencil.stencil;
   }
   if (clear && !resuming) {
      uint32_t texel[8];
      pack_depth_stencil(cmd->rendering.depth.image->vk.format, clear, &value, texel);
      record_clear(cmd, cmd->rendering.depth.image, cmd->rendering.depth.level,
                   cmd->rendering.depth.layer, 1, info->renderArea, texel);
   }
}

static VKAPI_ATTR void VKAPI_CALL
apex_CmdClearAttachments(VkCommandBuffer handle, uint32_t count, const VkClearAttachment *attachments,
                         uint32_t rect_count, const VkClearRect *rects)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   for (unsigned i = 0; i < count; i++) {
      const VkClearAttachment *a = &attachments[i];
      const struct apex_attachment *target = a->aspectMask & VK_IMAGE_ASPECT_COLOR_BIT ?
         (a->colorAttachment < APEX_DRAW_MAX_COLOR ? &cmd->rendering.color[a->colorAttachment] : NULL) :
         &cmd->rendering.depth;
      if (!target || !target->image)
         continue;
      uint32_t texel[8];
      if (a->aspectMask & VK_IMAGE_ASPECT_COLOR_BIT)
         pack_color(target->image->vk.format, &a->clearValue.color, texel);
      else
         pack_depth_stencil(target->image->vk.format, a->aspectMask, &a->clearValue.depthStencil, texel);
      for (unsigned r = 0; r < rect_count; r++)
         record_clear(cmd, target->image, target->level, target->layer + rects[r].baseArrayLayer,
                      rects[r].layerCount, rects[r].rect, texel);
   }
}

static VKAPI_ATTR void VKAPI_CALL
apex_CmdEndRendering(VkCommandBuffer handle)
{
   /* Attachments are written in place; stores need no resolve. */
}

static VKAPI_ATTR void VKAPI_CALL
apex_CmdBindVertexBuffers2(VkCommandBuffer handle, uint32_t first, uint32_t count,
                           const VkBuffer *buffers, const VkDeviceSize *offsets,
                           const VkDeviceSize *sizes, const VkDeviceSize *strides)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   for (uint32_t i = 0; i < count && first + i < APEX_DRAW_MAX_BINDINGS; i++) {
      VK_FROM_HANDLE(apex_buffer, buffer, buffers[i]);
      /* A null buffer binds zero bytes: fetches read zero. */
      cmd->bindings[first + i].va = buffer ? buffer->vk.device_address + offsets[i] : 0;
      cmd->bindings[first + i].size = !buffer ? 0 :
         sizes && sizes[i] != VK_WHOLE_SIZE ? sizes[i] : buffer->vk.size - offsets[i];
   }
   if (strides)
      vk_cmd_set_vertex_binding_strides(&cmd->vk, first, count, strides);
}

static VKAPI_ATTR void VKAPI_CALL
apex_CmdBindIndexBuffer2(VkCommandBuffer handle, VkBuffer buffer, VkDeviceSize offset,
                         VkDeviceSize size, VkIndexType type)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   VK_FROM_HANDLE(apex_buffer, index, buffer);
   cmd->index.bytes = type == VK_INDEX_TYPE_UINT32 ? 4 : type == VK_INDEX_TYPE_UINT16 ? 2 :
                      type == VK_INDEX_TYPE_UINT8 ? 1 : 0;
   cmd->index.va = index ? index->vk.device_address + offset : 0;
   cmd->index.size = !index ? 0 : size == VK_WHOLE_SIZE ? index->vk.size - offset : size;
   if (!cmd->index.bytes)
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
}

static void
push_job(struct apex_command_buffer *cmd, struct apex_program *program,
         struct apex_bound_set *const *sets, uint32_t groups, uint32_t base,
         uint32_t end, const uint32_t *draw)
{
   struct apex_dispatch *job = vk_zalloc(&cmd->vk.pool->alloc, sizeof(*job), 8,
                                         VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!job) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
      return;
   }
   job->program = program;
   job->groups = groups;
   job->parameters = (struct apex_dispatch_parameters){.base = {base}, .groups = {groups, end, 1}};
   memcpy(job->push, cmd->graphics_push, sizeof(job->push));
   job->graphics = true;
   memcpy(job->draw, draw, sizeof(job->draw));
   if (sets) {
      memcpy(job->sets, sets, sizeof(job->sets));
      for (unsigned s = 0; s < ARRAY_SIZE(job->sets); s++)
         if (job->sets[s])
            job->sets[s]->refs++;
   }
   list_addtail(&job->link, &cmd->dispatches);
}

static uint32_t
float_bits(float value)
{
   uint32_t bits;
   memcpy(&bits, &value, sizeof(bits));
   return bits;
}

/* Records vertex, setup and fragment jobs for one direct draw. */
static void
record_draw(struct apex_command_buffer *cmd, uint32_t vertex_count, uint32_t instance_count,
            uint32_t first_vertex, uint32_t first_instance, bool indexed, int32_t vertex_offset)
{
   struct apex_device *device = (void *)cmd->vk.base.device;
   const struct vk_dynamic_graphics_state *dyn = &cmd->vk.dynamic_graphics_state;
   if (!vertex_count || !instance_count)
      return;
   uint32_t topology = dyn->ia.primitive_topology;
   uint32_t prims;
   switch (topology) {
   case VK_PRIMITIVE_TOPOLOGY_POINT_LIST: prims = vertex_count; break;
   case VK_PRIMITIVE_TOPOLOGY_LINE_LIST: prims = vertex_count / 2; break;
   case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP: prims = vertex_count >= 2 ? vertex_count - 1 : 0; break;
   case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST: prims = vertex_count / 3; break;
   default: prims = vertex_count >= 3 ? vertex_count - 2 : 0; break;
   }
   if (topology > VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN ||
       !cmd->vertex || !cmd->fragment ||
       dyn->vp.viewport_count != 1 || dyn->rs.polygon_mode != VK_POLYGON_MODE_FILL ||
       (indexed && !cmd->index.bytes)) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   mesa_logd("Apex draw: %u vertices x %u instances, %u primitives, topology %u", vertex_count,
             instance_count, prims, topology);
   struct apex_program *setup;
   VkResult result = apex_internal_program(device, APEX_INTERNAL_SETUP, &setup);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd->vk, result);
      return;
   }
   uint64_t vertices = (uint64_t)vertex_count * instance_count;
   uint64_t vertex_bytes = vertices * cmd->vertex->vertex.stride * 4;
   uint64_t prim_bytes = (uint64_t)prims * instance_count * APEX_SUBPRIMS * APEX_PRIM_WORDS * 4;
   if (vertex_bytes > 64 * 1024 * 1024 || prim_bytes > 64 * 1024 * 1024) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_DEVICE_MEMORY);
      return;
   }
   uint32_t draw[APEX_DRAW_WORDS] = {0};
   uint64_t vertex_va = record_scratch(cmd, vertex_bytes);
   uint64_t prim_va = record_scratch(cmd, prim_bytes);
   if (!vertex_va || !prim_va)
      return;
   draw[APEX_DRAW_VERTEX_LO] = vertex_va;
   draw[APEX_DRAW_VERTEX_HI] = vertex_va >> 32;
   draw[APEX_DRAW_VERTEX_STRIDE] = cmd->vertex->vertex.stride;
   draw[APEX_DRAW_VERTEX_COUNT] = vertex_count;
   draw[APEX_DRAW_INSTANCE_COUNT] = instance_count;
   draw[APEX_DRAW_FIRST_VERTEX] = first_vertex;
   draw[APEX_DRAW_FIRST_INSTANCE] = first_instance;
   draw[APEX_DRAW_PRIM_LO] = prim_va;
   draw[APEX_DRAW_PRIM_HI] = prim_va >> 32;
   draw[APEX_DRAW_PRIM_COUNT] = prims;
   draw[APEX_DRAW_TOPOLOGY] = topology;
   /* The framebuffer is the smallest bound attachment. */
   uint32_t width = 4096, height = 4096;
   for (unsigned k = 0; k <= APEX_DRAW_MAX_COLOR; k++) {
      const struct apex_attachment *a = k < APEX_DRAW_MAX_COLOR ? &cmd->rendering.color[k] :
                                                                  &cmd->rendering.depth;
      if (!a->image)
         continue;
      width = MIN2(width, u_minify(a->image->vk.extent.width, a->level));
      height = MIN2(height, u_minify(a->image->vk.extent.height, a->level));
      unsigned slot = k < APEX_DRAW_MAX_COLOR ? APEX_DRAW_COLOR + k * APEX_DRAW_COLOR_WORDS :
                                                APEX_DRAW_DEPTH_TARGET;
      uint64_t va = attachment_address(a);
      draw[slot] = va;
      draw[slot + 1] = va >> 32;
      draw[slot + 2] = a->image->levels[a->level].row_stride;
   }
   draw[APEX_DRAW_WIDTH] = width;
   draw[APEX_DRAW_HEIGHT] = height;
   const VkViewport *vp = &dyn->vp.viewports[0];
   const float viewport[6] = {vp->x, vp->y, vp->width, vp->height, vp->minDepth, vp->maxDepth};
   for (unsigned i = 0; i < 6; i++)
      draw[APEX_DRAW_VIEWPORT + i] = float_bits(viewport[i]);
   const VkRect2D *scissor = &dyn->vp.scissors[0];
   const VkRect2D *area = &cmd->rendering.area;
   int64_t x0 = MAX3(scissor->offset.x, area->offset.x, 0);
   int64_t y0 = MAX3(scissor->offset.y, area->offset.y, 0);
   int64_t x1 = MIN3((int64_t)scissor->offset.x + scissor->extent.width,
                     (int64_t)area->offset.x + area->extent.width, width);
   int64_t y1 = MIN3((int64_t)scissor->offset.y + scissor->extent.height,
                     (int64_t)area->offset.y + area->extent.height, height);
   draw[APEX_DRAW_SCISSOR] = x0;
   draw[APEX_DRAW_SCISSOR + 1] = y0;
   draw[APEX_DRAW_SCISSOR + 2] = MAX2(x1, x0);
   draw[APEX_DRAW_SCISSOR + 3] = MAX2(y1, y0);
   draw[APEX_DRAW_CULL] = dyn->rs.cull_mode;
   draw[APEX_DRAW_FRONT_FACE] = dyn->rs.front_face;
   const struct apex_image *ds = cmd->rendering.depth.image;
   bool depth = ds && cmd->rendering.has_depth, stencil = ds && cmd->rendering.has_stencil;
   draw[APEX_DRAW_DEPTH] = (depth && dyn->ds.depth.test_enable) |
                           (depth && dyn->ds.depth.test_enable && dyn->ds.depth.write_enable) << 1 |
                           (stencil && dyn->ds.stencil.test_enable) << 2 |
                           (depth && dyn->rs.depth_bias.enable) << 3 |
                           dyn->ds.depth.compare_op << 4;
   for (unsigned f = 0; f < 2; f++) {
      const struct vk_stencil_test_face_state *face = f ? &dyn->ds.stencil.back : &dyn->ds.stencil.front;
      draw[APEX_DRAW_STENCIL + f * 2] = face->op.fail | face->op.pass << 3 | face->op.depth_fail << 6 |
                                        face->op.compare << 9;
      draw[APEX_DRAW_STENCIL + f * 2 + 1] = face->compare_mask | face->write_mask << 8 |
                                            (face->reference & 0xff) << 16;
   }
   draw[APEX_DRAW_DEPTH_BIAS] = float_bits(dyn->rs.depth_bias.constant_factor);
   draw[APEX_DRAW_DEPTH_BIAS + 1] = float_bits(dyn->rs.depth_bias.slope_factor);
   draw[APEX_DRAW_DEPTH_BIAS + 2] = float_bits(dyn->rs.depth_bias.clamp);
   if (ds) {
      enum pipe_format zs = vk_format_to_pipe_format(ds->vk.format);
      const struct util_format_description *desc = util_format_description(zs);
      unsigned z = desc->swizzle[0];
      if (z <= PIPE_SWIZZLE_W && desc->channel[z].type == UTIL_FORMAT_TYPE_UNSIGNED)
         draw[APEX_DRAW_DEPTH_BIAS + 3] = desc->channel[z].size;
   }
   for (unsigned l = 0; l < 32; l++) {
      int slot = cmd->vertex->vertex.slot[VARYING_SLOT_VAR0 + l];
      draw[APEX_DRAW_SLOTS + l] = slot < 0 ? ~0u : slot;
   }
   int point_size = cmd->vertex->vertex.slot[VARYING_SLOT_PSIZ];
   draw[APEX_DRAW_POINT_SIZE] = point_size < 0 ? ~0u : point_size;
   for (unsigned b = 0; b < APEX_DRAW_MAX_BINDINGS; b++) {
      unsigned slot = APEX_DRAW_BINDINGS + b * APEX_DRAW_BINDING_WORDS;
      draw[slot] = cmd->bindings[b].va;
      draw[slot + 1] = cmd->bindings[b].va >> 32;
      draw[slot + 2] = MIN2(cmd->bindings[b].size, UINT32_MAX);
      draw[slot + 3] = dyn->vi_binding_strides[b];
   }
   if (indexed) {
      draw[APEX_DRAW_INDEX] = cmd->index.va;
      draw[APEX_DRAW_INDEX + 1] = cmd->index.va >> 32;
      draw[APEX_DRAW_INDEX + 2] = cmd->index.bytes;
      draw[APEX_DRAW_INDEX + 3] = vertex_offset;
      draw[APEX_DRAW_RESTART] = dyn->ia.primitive_restart_enable;
   }
   for (unsigned k = 0; k < APEX_DRAW_MAX_COLOR; k++) {
      const struct vk_color_blend_attachment_state *a = &dyn->cb.attachments[k];
      bool enabled = dyn->cb.color_write_enables & BITFIELD_BIT(k);
      draw[APEX_DRAW_BLEND + k * 2] = a->src_color_blend_factor | a->dst_color_blend_factor << 8 |
         a->src_alpha_blend_factor << 16 | a->dst_alpha_blend_factor << 24;
      draw[APEX_DRAW_BLEND + k * 2 + 1] = a->color_blend_op | a->alpha_blend_op << 8 |
         (enabled ? a->write_mask : 0) << 16 | (uint32_t)a->blend_enable << 24;
   }
   for (unsigned c = 0; c < 4; c++)
      draw[APEX_DRAW_BLEND_CONSTANTS + c] = float_bits(dyn->cb.blend_constants[c]);
   draw[APEX_DRAW_OCCLUSION] = cmd->occlusion;
   draw[APEX_DRAW_OCCLUSION + 1] = cmd->occlusion >> 32;

   struct apex_program *vs = &cmd->vertex->program;
   uint32_t limit = MIN2(vs->max_workgroups, 1024);
   uint32_t vertex_groups = DIV_ROUND_UP(vertices, 64);
   for (uint32_t base = 0; base < vertex_groups; base += limit)
      push_job(cmd, vs, cmd->graphics_sets, MIN2(vertex_groups - base, limit), base, 0, draw);
   if (dyn->rs.rasterizer_discard_enable || !prims)
      return;
   uint32_t setup_groups = DIV_ROUND_UP((uint64_t)prims * instance_count, 64);
   limit = MIN2(setup->max_workgroups, 1024);
   for (uint32_t base = 0; base < setup_groups; base += limit)
      push_job(cmd, setup, NULL, MIN2(setup_groups - base, limit), base, 0, draw);
   uint32_t tiles = (DIV_ROUND_UP(draw[APEX_DRAW_SCISSOR + 2], 4) - draw[APEX_DRAW_SCISSOR] / 4) *
                    (DIV_ROUND_UP(draw[APEX_DRAW_SCISSOR + 3], 4) - draw[APEX_DRAW_SCISSOR + 1] / 4);
   if (!tiles)
      return;
   /* Smallest bins (one 4-pixel tile up) whose ordered lists fit in 32 MiB. */
   uint64_t total = (uint64_t)prims * instance_count;
   uint32_t chunks = DIV_ROUND_UP(total, APEX_BIN_CHUNK), shift = 2, columns, rows, bin_x0, bin_y0;
   for (;; shift++) {
      bin_x0 = draw[APEX_DRAW_SCISSOR] >> shift;
      bin_y0 = draw[APEX_DRAW_SCISSOR + 1] >> shift;
      columns = ((draw[APEX_DRAW_SCISSOR + 2] + (1u << shift) - 1) >> shift) - bin_x0;
      rows = ((draw[APEX_DRAW_SCISSOR + 3] + (1u << shift) - 1) >> shift) - bin_y0;
      if ((uint64_t)columns * rows * total * 4 <= 32 * 1024 * 1024 || shift == 13)
         break;
   }
   uint64_t bins = (uint64_t)columns * rows;
   uint64_t lists = record_scratch(cmd, bins * total * 4);
   uint64_t counts = record_scratch(cmd, bins * chunks * 4);
   if (!lists || !counts)
      return;
   draw[APEX_DRAW_BIN_LISTS] = lists;
   draw[APEX_DRAW_BIN_LISTS + 1] = lists >> 32;
   draw[APEX_DRAW_BIN_COUNTS] = counts;
   draw[APEX_DRAW_BIN_COUNTS + 1] = counts >> 32;
   draw[APEX_DRAW_BIN_SHIFT] = shift;
   draw[APEX_DRAW_BIN_COLUMNS] = columns;
   draw[APEX_DRAW_BIN_ROWS] = rows;
   draw[APEX_DRAW_BIN_CHUNKS] = chunks;
   draw[APEX_DRAW_BIN_X0] = bin_x0;
   draw[APEX_DRAW_BIN_Y0] = bin_y0;
   struct apex_program *binner;
   result = apex_internal_program(device, APEX_INTERNAL_BIN, &binner);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd->vk, result);
      return;
   }
   uint32_t bin_groups = DIV_ROUND_UP(align64(bins, 16) * chunks, 16);
   limit = MIN2(binner->max_workgroups, 1024);
   for (uint32_t base = 0; base < bin_groups; base += limit)
      push_job(cmd, binner, NULL, MIN2(bin_groups - base, limit), base, 0, draw);
   /* The launch watchdog bounds each workgroup: a fragment job walks at most
    * APEX_TILES_PER_WORKGROUP tiles per workgroup. */
   struct apex_program *fs = &cmd->fragment->program;
   uint32_t groups = MIN3(tiles, fs->max_workgroups, 1024);
   for (uint32_t base = 0; base < tiles; base += groups * APEX_TILES_PER_WORKGROUP)
      push_job(cmd, fs, cmd->graphics_sets, MIN2(groups, tiles - base), base,
               MIN2(tiles, base + groups * APEX_TILES_PER_WORKGROUP), draw);
}

static VKAPI_ATTR void VKAPI_CALL
apex_CmdDraw(VkCommandBuffer handle, uint32_t vertex_count, uint32_t instance_count,
             uint32_t first_vertex, uint32_t first_instance)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   record_draw(cmd, vertex_count, instance_count, first_vertex, first_instance, false, 0);
}

static VKAPI_ATTR void VKAPI_CALL
apex_CmdDrawIndexed(VkCommandBuffer handle, uint32_t index_count, uint32_t instance_count,
                    uint32_t first_index, int32_t vertex_offset, uint32_t first_instance)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   record_draw(cmd, index_count, instance_count, first_index, first_instance, true, vertex_offset);
}

/* ---- Events and queries --------------------------------------------------- */

/* Small host-readable device words: host-coherent SYSTEM memory when the
 * device has it, otherwise explicit-transfer storage. */
static VkResult
host_words_create(struct apex_device *device, uint64_t size, struct apex_bo *bo)
{
   return bo_create(device, size, APEX_DRM_VM_READ | APEX_DRM_VM_WRITE,
                    device->host_coherent ? APEX_DRM_GEM_HOST_COHERENT : 0, 0, bo);
}

static VkResult
host_words_sync(struct apex_device *device, struct apex_bo *bo, uint32_t direction,
                uint64_t offset, uint64_t size)
{
   return device->host_coherent ? VK_SUCCESS : bo_transfer(device, bo, direction, offset, size);
}

struct apex_event {
   struct vk_object_base base;
   struct apex_bo bo;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(apex_event, base, VkEvent, VK_OBJECT_TYPE_EVENT);

static VKAPI_ATTR VkResult VKAPI_CALL
apex_CreateEvent(VkDevice dev, const VkEventCreateInfo *info,
                 const VkAllocationCallbacks *alloc, VkEvent *out)
{
   VK_FROM_HANDLE(apex_device, device, dev);
   struct apex_event *event = vk_object_zalloc(&device->vk, alloc, sizeof(*event), VK_OBJECT_TYPE_EVENT);
   if (!event)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   VkResult result = host_words_create(device, 4096, &event->bo);
   if (result != VK_SUCCESS) {
      vk_object_free(&device->vk, alloc, event);
      return result;
   }
   *out = apex_event_to_handle(event);
   return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL
apex_DestroyEvent(VkDevice dev, VkEvent handle, const VkAllocationCallbacks *alloc)
{
   VK_FROM_HANDLE(apex_device, device, dev);
   VK_FROM_HANDLE(apex_event, event, handle);
   if (!event)
      return;
   apex_bo_finish(device, &event->bo);
   vk_object_free(&device->vk, alloc, event);
}

static VkResult
set_event(struct apex_device *device, struct apex_event *event, uint32_t value)
{
   *(volatile uint32_t *)event->bo.map = value;
   return host_words_sync(device, &event->bo, APEX_DRM_TRANSFER_TO_LOCAL, 0, 4);
}

static VKAPI_ATTR VkResult VKAPI_CALL
apex_GetEventStatus(VkDevice dev, VkEvent handle)
{
   VK_FROM_HANDLE(apex_device, device, dev);
   VK_FROM_HANDLE(apex_event, event, handle);
   VkResult result = vk_device_check_status(&device->vk);
   if (result == VK_SUCCESS)
      result = host_words_sync(device, &event->bo, APEX_DRM_TRANSFER_FROM_LOCAL, 0, 4);
   if (result != VK_SUCCESS)
      return result;
   return *(volatile uint32_t *)event->bo.map ? VK_EVENT_SET : VK_EVENT_RESET;
}

static VKAPI_ATTR VkResult VKAPI_CALL
apex_SetEvent(VkDevice dev, VkEvent handle)
{
   return set_event(apex_device_from_handle(dev), apex_event_from_handle(handle), 1);
}

static VKAPI_ATTR VkResult VKAPI_CALL
apex_ResetEvent(VkDevice dev, VkEvent handle)
{
   return set_event(apex_device_from_handle(dev), apex_event_from_handle(handle), 0);
}

/* Fills one word in recording order; the queue executes jobs in order. */
static void
record_word(struct apex_command_buffer *cmd, uint64_t va, uint64_t size, uint32_t value)
{
   struct apex_device *device = (void *)cmd->vk.base.device;
   struct apex_pipeline *pipeline = cmd->pipeline;
   uint8_t push[APEX_MAX_PUSH_CONSTANTS];
   memcpy(push, cmd->push, sizeof(push));
   VkDeviceAddressRangeKHR range = {.address = va, .size = size};
   vk_meta_fill_memory(&cmd->vk, &device->meta, &range, 0, value);
   cmd->pipeline = pipeline;
   memcpy(cmd->push, push, sizeof(push));
}

static VKAPI_ATTR void VKAPI_CALL
apex_CmdSetEvent2(VkCommandBuffer handle, VkEvent event, const VkDependencyInfo *info)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   record_word(cmd, apex_event_from_handle(event)->bo.va, 4, 1);
}

static VKAPI_ATTR void VKAPI_CALL
apex_CmdResetEvent2(VkCommandBuffer handle, VkEvent event, VkPipelineStageFlags2 stage)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   record_word(cmd, apex_event_from_handle(event)->bo.va, 4, 0);
}

/* Queue work is in order: device-side signals precede every later wait.
 * Host signals must precede submission of the waiting batch. */
static VKAPI_ATTR void VKAPI_CALL
apex_CmdWaitEvents2(VkCommandBuffer handle, uint32_t count, const VkEvent *events,
                    const VkDependencyInfo *infos)
{
}

struct apex_query_pool {
   struct vk_query_pool vk;
   struct apex_bo bo;
};
VK_DEFINE_NONDISP_HANDLE_CASTS(apex_query_pool, vk.base, VkQueryPool, VK_OBJECT_TYPE_QUERY_POOL);

static VKAPI_ATTR VkResult VKAPI_CALL
apex_CreateQueryPool(VkDevice dev, const VkQueryPoolCreateInfo *info,
                     const VkAllocationCallbacks *alloc, VkQueryPool *out)
{
   VK_FROM_HANDLE(apex_device, device, dev);
   *out = VK_NULL_HANDLE;
   if (info->queryType != VK_QUERY_TYPE_OCCLUSION && info->queryType != VK_QUERY_TYPE_TIMESTAMP)
      return VK_ERROR_FEATURE_NOT_PRESENT;
   struct apex_query_pool *pool = vk_query_pool_create(&device->vk, info, alloc, sizeof(*pool));
   if (!pool)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   VkResult result = host_words_create(device, (uint64_t)MAX2(info->queryCount, 1) * APEX_QUERY_STRIDE,
                                       &pool->bo);
   if (result == VK_SUCCESS)
      result = host_words_sync(device, &pool->bo, APEX_DRM_TRANSFER_TO_LOCAL, 0, pool->bo.size);
   if (result != VK_SUCCESS) {
      apex_bo_finish(device, &pool->bo);
      vk_query_pool_destroy(&device->vk, alloc, &pool->vk);
      return result;
   }
   *out = apex_query_pool_to_handle(pool);
   return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL
apex_DestroyQueryPool(VkDevice dev, VkQueryPool handle, const VkAllocationCallbacks *alloc)
{
   VK_FROM_HANDLE(apex_device, device, dev);
   VK_FROM_HANDLE(apex_query_pool, pool, handle);
   if (!pool)
      return;
   apex_bo_finish(device, &pool->bo);
   vk_query_pool_destroy(&device->vk, alloc, &pool->vk);
}

static VKAPI_ATTR void VKAPI_CALL
apex_ResetQueryPool(VkDevice dev, VkQueryPool handle, uint32_t first, uint32_t count)
{
   VK_FROM_HANDLE(apex_device, device, dev);
   VK_FROM_HANDLE(apex_query_pool, pool, handle);
   memset((uint8_t *)pool->bo.map + (uint64_t)first * APEX_QUERY_STRIDE, 0,
          (uint64_t)count * APEX_QUERY_STRIDE);
   host_words_sync(device, &pool->bo, APEX_DRM_TRANSFER_TO_LOCAL,
                   (uint64_t)first * APEX_QUERY_STRIDE, (uint64_t)count * APEX_QUERY_STRIDE);
}

static VKAPI_ATTR VkResult VKAPI_CALL
apex_GetQueryPoolResults(VkDevice dev, VkQueryPool handle, uint32_t first, uint32_t count,
                         size_t size, void *data, VkDeviceSize stride, VkQueryResultFlags flags)
{
   VK_FROM_HANDLE(apex_device, device, dev);
   VK_FROM_HANDLE(apex_query_pool, pool, handle);
   const uint8_t *slots = (const uint8_t *)pool->bo.map + (uint64_t)first * APEX_QUERY_STRIDE;
   VkResult status = VK_SUCCESS;
   for (uint32_t q = 0; q < count; q++) {
      const volatile uint32_t *slot = (const void *)(slots + (uint64_t)q * APEX_QUERY_STRIDE);
      uint64_t offset = (uint64_t)(first + q) * APEX_QUERY_STRIDE;
      for (;;) {
         VkResult result = vk_device_check_status(&device->vk);
         if (result == VK_SUCCESS)
            result = host_words_sync(device, &pool->bo, APEX_DRM_TRANSFER_FROM_LOCAL, offset,
                                     APEX_QUERY_STRIDE);
         if (result != VK_SUCCESS)
            return result;
         if (slot[2] || !(flags & VK_QUERY_RESULT_WAIT_BIT))
            break;
         /* Results arrive from the queue; poll without holding locks. */
         nanosleep(&(struct timespec){.tv_nsec = 100000}, NULL);
      }
      bool available = slot[2];
      uint64_t value = slot[0] | (uint64_t)slot[1] << 32;
      uint8_t *out = (uint8_t *)data + q * stride;
      if (available || (flags & VK_QUERY_RESULT_PARTIAL_BIT)) {
         if (flags & VK_QUERY_RESULT_64_BIT)
            memcpy(out, &value, 8);
         else
            *(uint32_t *)out = value > UINT32_MAX ? UINT32_MAX : value;
      }
      if (!available)
         status = VK_NOT_READY;
      if (flags & VK_QUERY_RESULT_WITH_AVAILABILITY_BIT) {
         if (flags & VK_QUERY_RESULT_64_BIT)
            *(uint64_t *)(out + 8) = available;
         else
            *(uint32_t *)(out + 4) = available;
      }
   }
   return status;
}

static VKAPI_ATTR void VKAPI_CALL
apex_CmdResetQueryPool(VkCommandBuffer handle, VkQueryPool pool, uint32_t first, uint32_t count)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   if (count)
      record_word(cmd, apex_query_pool_from_handle(pool)->bo.va + (uint64_t)first * APEX_QUERY_STRIDE,
                  (uint64_t)count * APEX_QUERY_STRIDE, 0);
}

static VKAPI_ATTR void VKAPI_CALL
apex_CmdBeginQueryIndexedEXT(VkCommandBuffer handle, VkQueryPool pool, uint32_t query,
                             VkQueryControlFlags flags, uint32_t index)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   uint64_t slot = apex_query_pool_from_handle(pool)->bo.va + (uint64_t)query * APEX_QUERY_STRIDE;
   record_word(cmd, slot, 8, 0);
   cmd->occlusion = slot;
}

static VKAPI_ATTR void VKAPI_CALL
apex_CmdEndQueryIndexedEXT(VkCommandBuffer handle, VkQueryPool pool, uint32_t query, uint32_t index)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   cmd->occlusion = 0;
   record_word(cmd, apex_query_pool_from_handle(pool)->bo.va + (uint64_t)query * APEX_QUERY_STRIDE + 8,
               4, 1);
}

static void push_job(struct apex_command_buffer *cmd, struct apex_program *program,
                     struct apex_bound_set *const *sets, uint32_t groups, uint32_t base,
                     uint32_t end, const uint32_t *draw);

static VKAPI_ATTR void VKAPI_CALL
apex_CmdWriteTimestamp2(VkCommandBuffer handle, VkPipelineStageFlags2 stage, VkQueryPool pool,
                        uint32_t query)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   struct apex_device *device = (void *)cmd->vk.base.device;
   struct apex_program *program;
   VkResult result = apex_internal_program(device, APEX_INTERNAL_TIMESTAMP, &program);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd->vk, result);
      return;
   }
   uint64_t slot = apex_query_pool_from_handle(pool)->bo.va + (uint64_t)query * APEX_QUERY_STRIDE;
   uint32_t words[APEX_DRAW_WORDS] = {[APEX_QUERY_SLOT] = slot, [APEX_QUERY_SLOT + 1] = slot >> 32};
   push_job(cmd, program, NULL, 1, 0, 0, words);
}

static VKAPI_ATTR void VKAPI_CALL
apex_CmdCopyQueryPoolResults(VkCommandBuffer handle, VkQueryPool pool, uint32_t first,
                             uint32_t count, VkBuffer buffer, VkDeviceSize offset,
                             VkDeviceSize stride, VkQueryResultFlags flags)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   VK_FROM_HANDLE(apex_buffer, dst, buffer);
   struct apex_device *device = (void *)cmd->vk.base.device;
   struct apex_program *program;
   VkResult result = apex_internal_program(device, APEX_INTERNAL_QUERY_COPY, &program);
   if (result != VK_SUCCESS || !count) {
      if (result != VK_SUCCESS)
         vk_command_buffer_set_error(&cmd->vk, result);
      return;
   }
   uint64_t slot = apex_query_pool_from_handle(pool)->bo.va + (uint64_t)first * APEX_QUERY_STRIDE;
   uint64_t va = dst->vk.device_address + offset;
   uint32_t words[APEX_DRAW_WORDS] = {
      [APEX_QUERY_SLOT] = slot, [APEX_QUERY_SLOT + 1] = slot >> 32,
      [APEX_QUERY_DST] = va, [APEX_QUERY_DST + 1] = va >> 32,
      [APEX_QUERY_DST_STRIDE] = stride, [APEX_QUERY_COUNT] = count, [APEX_QUERY_FLAGS] = flags,
   };
   push_job(cmd, program, NULL, DIV_ROUND_UP(count, 16), 0, 0, words);
}

static VKAPI_ATTR VkResult VKAPI_CALL
apex_CreateBufferView(VkDevice dev, const VkBufferViewCreateInfo *info,
                      const VkAllocationCallbacks *alloc, VkBufferView *out)
{
   VK_FROM_HANDLE(apex_device, device, dev);
   struct vk_buffer_view *view = vk_buffer_view_create(&device->vk, info, alloc, sizeof(*view));
   if (!view)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   *out = vk_buffer_view_to_handle(view);
   return VK_SUCCESS;
}

/* No memory type is lazily allocated. */
static VKAPI_ATTR void VKAPI_CALL
apex_GetDeviceMemoryCommitment(VkDevice dev, VkDeviceMemory memory, VkDeviceSize *committed)
{
   *committed = 0;
}

static VKAPI_ATTR void VKAPI_CALL
apex_GetImageSparseMemoryRequirements2(VkDevice dev, const VkImageSparseMemoryRequirementsInfo2 *info,
   uint32_t *count, VkSparseImageMemoryRequirements2 *requirements)
{
   *count = 0;
}

static VKAPI_ATTR void VKAPI_CALL
apex_GetDeviceImageSparseMemoryRequirements(VkDevice dev, const VkDeviceImageMemoryRequirements *info,
   uint32_t *count, VkSparseImageMemoryRequirements2 *requirements)
{
   *count = 0;
}

static VKAPI_ATTR void VKAPI_CALL
apex_CmdFillBuffer(VkCommandBuffer handle, VkBuffer buffer, VkDeviceSize offset,
                   VkDeviceSize size, uint32_t data)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   VK_FROM_HANDLE(apex_buffer, dst, buffer);
   struct apex_device *device = (void *)cmd->vk.base.device;
   if (device->transport != APEX_TRANSPORT_DRM) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   VkDeviceAddressRangeKHR range = vk_device_address_range(&dst->vk, offset, size);
   /* VK_WHOLE_SIZE leaves the final incomplete word untouched. */
   range.size &= ~3ull;
   if (!range.size)
      return;
   struct apex_pipeline *pipeline = cmd->pipeline;
   uint8_t push[APEX_MAX_PUSH_CONSTANTS];
   memcpy(push, cmd->push, sizeof(push));
   vk_meta_fill_memory(&cmd->vk, &device->meta, &range, dst->vk.address_flags, data);
   cmd->pipeline = pipeline;
   memcpy(cmd->push, push, sizeof(push));
}

static VKAPI_ATTR void VKAPI_CALL
apex_CmdCopyBuffer2(VkCommandBuffer handle, const VkCopyBufferInfo2 *info)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   struct apex_device *device = (void *)cmd->vk.base.device;
   if (device->transport != APEX_TRANSPORT_DRM) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   struct apex_pipeline *pipeline = cmd->pipeline;
   uint8_t push[APEX_MAX_PUSH_CONSTANTS];
   memcpy(push, cmd->push, sizeof(push));
   vk_meta_copy_buffer(&cmd->vk, &device->meta, info);
   cmd->pipeline = pipeline;
   memcpy(cmd->push, push, sizeof(push));
}

static VKAPI_ATTR void VKAPI_CALL
apex_CmdUpdateBuffer(VkCommandBuffer handle, VkBuffer buffer, VkDeviceSize offset,
                     VkDeviceSize size, const void *data)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   struct apex_device *device = (void *)cmd->vk.base.device;
   if (device->transport != APEX_TRANSPORT_DRM) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   struct apex_pipeline *pipeline = cmd->pipeline;
   uint8_t push[APEX_MAX_PUSH_CONSTANTS];
   memcpy(push, cmd->push, sizeof(push));
   vk_meta_update_buffer(&cmd->vk, &device->meta, buffer, offset, size, data);
   cmd->pipeline = pipeline;
   memcpy(cmd->push, push, sizeof(push));
}

static uint64_t
image_address(const struct apex_image *image, unsigned level, unsigned layer, VkOffset3D offset)
{
   return image->memory->storage->bo.va + image->offset + image->levels[level].offset +
          (uint64_t)layer * image->levels[level].slice_stride +
          (uint64_t)offset.y * image->levels[level].row_stride +
          (uint64_t)offset.x * vk_format_get_blocksize(image->vk.format);
}

/* Linear image rows use the ordinary cached buffer meta kernels. Neither path
 * binds descriptor sets; preserve the application's pipeline and push image. */
/* One invocation per 64-word or 64-element chunk of a row. */
static void
push_copy(struct apex_command_buffer *cmd, struct apex_program *copy, const uint32_t *words,
          uint64_t chunks)
{
   uint32_t groups = DIV_ROUND_UP(chunks, 16);
   uint32_t limit = MIN2(copy->max_workgroups, 1024);
   for (uint32_t base = 0; base < groups; base += limit)
      push_job(cmd, copy, NULL, MIN2(groups - base, limit), base, 0, words);
}

/* Copies one aspect of combined depth/stencil texels as strided elements,
 * writing only `mask` bits of each destination element. */
static void
copy_elements(struct apex_command_buffer *cmd, uint64_t src, uint32_t src_row, uint64_t src_slice,
              uint32_t src_stride, uint64_t dst, uint32_t dst_row, uint64_t dst_slice,
              uint32_t dst_stride, uint32_t element, uint32_t mask, VkExtent3D extent, uint32_t layers)
{
   struct apex_device *device = (void *)cmd->vk.base.device;
   struct apex_program *copy;
   VkResult result = src_slice > UINT32_MAX || dst_slice > UINT32_MAX ? VK_ERROR_FEATURE_NOT_PRESENT :
                     apex_internal_program(device, APEX_INTERNAL_COPY, &copy);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd->vk, result);
      return;
   }
   uint32_t words[APEX_DRAW_WORDS] = {
      [APEX_COPY_SRC] = src, [APEX_COPY_SRC + 1] = src >> 32,
      [APEX_COPY_DST] = dst, [APEX_COPY_DST + 1] = dst >> 32,
      [APEX_COPY_SRC_ROW] = src_row, [APEX_COPY_DST_ROW] = dst_row,
      [APEX_COPY_SRC_SLICE] = src_slice, [APEX_COPY_DST_SLICE] = dst_slice,
      [APEX_COPY_WORDS] = extent.width, [APEX_COPY_ROWS] = extent.height, [APEX_COPY_LAYERS] = layers,
      [APEX_COPY_ELEMENT] = element, [APEX_COPY_SRC_STRIDE] = src_stride,
      [APEX_COPY_DST_STRIDE] = dst_stride, [APEX_COPY_DST_MASK] = mask,
   };
   push_copy(cmd, copy, words, DIV_ROUND_UP(extent.width, 64) * extent.height * layers);
}

static void
copy_image_rows(struct apex_command_buffer *cmd, uint64_t src, uint32_t src_row, uint64_t src_slice,
                uint64_t dst, uint32_t dst_row, uint64_t dst_slice, VkExtent3D extent, uint32_t layers,
                uint32_t pixel)
{
   struct apex_device *device = (void *)cmd->vk.base.device;
   uint64_t row_bytes = (uint64_t)extent.width * pixel;
   struct apex_program *copy;
   /* Word-aligned regions copy in few jobs: one invocation per 64-word chunk. */
   if (!((src | dst | src_row | dst_row | src_slice | dst_slice | row_bytes) & 3) &&
       src_slice <= UINT32_MAX && dst_slice <= UINT32_MAX &&
       apex_internal_program(device, APEX_INTERNAL_COPY, &copy) == VK_SUCCESS) {
      uint32_t words[APEX_DRAW_WORDS] = {0};
      words[APEX_COPY_SRC] = src;
      words[APEX_COPY_SRC + 1] = src >> 32;
      words[APEX_COPY_DST] = dst;
      words[APEX_COPY_DST + 1] = dst >> 32;
      words[APEX_COPY_SRC_ROW] = src_row;
      words[APEX_COPY_DST_ROW] = dst_row;
      words[APEX_COPY_SRC_SLICE] = src_slice;
      words[APEX_COPY_DST_SLICE] = dst_slice;
      words[APEX_COPY_WORDS] = row_bytes / 4;
      words[APEX_COPY_ROWS] = extent.height;
      words[APEX_COPY_LAYERS] = layers;
      push_copy(cmd, copy, words, DIV_ROUND_UP(row_bytes / 4, 64) * extent.height * layers);
      return;
   }
   struct apex_pipeline *pipeline = cmd->pipeline;
   uint8_t push[APEX_MAX_PUSH_CONSTANTS];
   memcpy(push, cmd->push, sizeof(push));
   for (unsigned z = 0; z < layers; z++) {
      for (unsigned y = 0; y < extent.height; y++) {
         VkDeviceMemoryCopyKHR region = {
            .sType = VK_STRUCTURE_TYPE_DEVICE_MEMORY_COPY_KHR,
            .srcRange = {.address = src + z * src_slice + (uint64_t)y * src_row, .size = row_bytes},
            .dstRange = {.address = dst + z * dst_slice + (uint64_t)y * dst_row, .size = row_bytes},
         };
         VkCopyDeviceMemoryInfoKHR info = {
            .sType = VK_STRUCTURE_TYPE_COPY_DEVICE_MEMORY_INFO_KHR, .regionCount = 1, .pRegions = &region,
         };
         vk_meta_copy_memory(&cmd->vk, &device->meta, &info);
      }
   }
   cmd->pipeline = pipeline;
   memcpy(cmd->push, push, sizeof(push));
}

static void
copy_buffer_image(struct apex_command_buffer *cmd, struct apex_buffer *buffer,
                   struct apex_image *image, unsigned count, const VkBufferImageCopy2 *regions,
                   bool to_image)
{
   for (unsigned i = 0; i < count; i++) {
      const VkBufferImageCopy2 *r = &regions[i];
      struct vk_image_buffer_layout layout = vk_image_buffer_copy_layout(&image->vk, r);
      unsigned l = r->imageSubresource.mipLevel;
      uint64_t image_va = image_address(image, l, r->imageSubresource.baseArrayLayer, r->imageOffset);
      uint64_t buffer_va = buffer->vk.device_address + r->bufferOffset;
      VkFormat format = image->vk.format;
      if (vk_format_has_depth(format) && vk_format_has_stencil(format)) {
         /* Buffers hold one aspect: 4-byte depth or 1-byte stencil elements. */
         const struct util_format_description *desc =
            util_format_description(vk_format_to_pipe_format(format));
         bool stencil = r->imageSubresource.aspectMask == VK_IMAGE_ASPECT_STENCIL_BIT;
         const struct util_format_channel_description *ch = &desc->channel[desc->swizzle[stencil]];
         unsigned element = stencil ? 1 : 4, texel = vk_format_get_blocksize(format);
         uint64_t texel_va = image_va + ch->shift / 8;
         if (to_image)
            copy_elements(cmd, buffer_va, layout.row_stride_B, layout.image_stride_B, element,
                          texel_va, image->levels[l].row_stride, image->levels[l].slice_stride, texel,
                          element, ch->size == 32 ? ~0u : BITFIELD_MASK(ch->size), r->imageExtent,
                          r->imageSubresource.layerCount);
         else
            copy_elements(cmd, texel_va, image->levels[l].row_stride, image->levels[l].slice_stride,
                          texel, buffer_va, layout.row_stride_B, layout.image_stride_B, element,
                          element, ~0u, r->imageExtent, r->imageSubresource.layerCount);
         continue;
      }
      if (to_image)
         copy_image_rows(cmd, buffer_va, layout.row_stride_B, layout.image_stride_B,
                         image_va, image->levels[l].row_stride, image->levels[l].slice_stride,
                         r->imageExtent, r->imageSubresource.layerCount,
                         vk_format_get_blocksize(image->vk.format));
      else
         copy_image_rows(cmd, image_va, image->levels[l].row_stride, image->levels[l].slice_stride,
                         buffer_va, layout.row_stride_B, layout.image_stride_B,
                         r->imageExtent, r->imageSubresource.layerCount,
                         vk_format_get_blocksize(image->vk.format));
   }
}

static VKAPI_ATTR void VKAPI_CALL
apex_CmdCopyBufferToImage2(VkCommandBuffer handle, const VkCopyBufferToImageInfo2 *info)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   VK_FROM_HANDLE(apex_buffer, buffer, info->srcBuffer);
   VK_FROM_HANDLE(apex_image, image, info->dstImage);
   copy_buffer_image(cmd, buffer, image, info->regionCount, info->pRegions, true);
}

static VKAPI_ATTR void VKAPI_CALL
apex_CmdCopyImageToBuffer2(VkCommandBuffer handle, const VkCopyImageToBufferInfo2 *info)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   VK_FROM_HANDLE(apex_image, image, info->srcImage);
   VK_FROM_HANDLE(apex_buffer, buffer, info->dstBuffer);
   copy_buffer_image(cmd, buffer, image, info->regionCount, info->pRegions, false);
}

static VKAPI_ATTR void VKAPI_CALL
apex_CmdCopyImage2(VkCommandBuffer handle, const VkCopyImageInfo2 *info)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   VK_FROM_HANDLE(apex_image, src, info->srcImage);
   VK_FROM_HANDLE(apex_image, dst, info->dstImage);
   for (unsigned i = 0; i < info->regionCount; i++) {
      const VkImageCopy2 *r = &info->pRegions[i];
      unsigned sl = r->srcSubresource.mipLevel, dl = r->dstSubresource.mipLevel;
      copy_image_rows(cmd, image_address(src, sl, r->srcSubresource.baseArrayLayer, r->srcOffset),
                      src->levels[sl].row_stride, src->levels[sl].slice_stride,
                      image_address(dst, dl, r->dstSubresource.baseArrayLayer, r->dstOffset),
                      dst->levels[dl].row_stride, dst->levels[dl].slice_stride,
                      r->extent, r->srcSubresource.layerCount,
                      vk_format_get_blocksize(src->vk.format));
   }
}

/* Records the internal clear kernel over layers x rect of one level. */
static void
record_clear(struct apex_command_buffer *cmd, const struct apex_image *image, unsigned level,
             unsigned layer, unsigned layers, VkRect2D rect, const uint32_t texel[8])
{
   struct apex_device *device = (void *)cmd->vk.base.device;
   unsigned width = u_minify(image->vk.extent.width, level);
   unsigned height = u_minify(image->vk.extent.height, level);
   if (image->vk.image_type == VK_IMAGE_TYPE_3D) {
      /* Clear every depth slice of a 3D level as layers. */
      layer = 0;
      layers = u_minify(image->vk.extent.depth, level);
   }
   int64_t x0 = MAX2(rect.offset.x, 0), y0 = MAX2(rect.offset.y, 0);
   int64_t x1 = MIN2((int64_t)rect.offset.x + rect.extent.width, width);
   int64_t y1 = MIN2((int64_t)rect.offset.y + rect.extent.height, height);
   if (x0 >= x1 || y0 >= y1 || !layers)
      return;
   struct apex_program *program;
   VkResult result = apex_internal_program(device, APEX_INTERNAL_CLEAR, &program);
   if (result != VK_SUCCESS) {
      vk_command_buffer_set_error(&cmd->vk, result);
      return;
   }
   uint64_t va = image_address(image, level, layer, (VkOffset3D){x0, y0, 0});
   uint32_t words[APEX_DRAW_WORDS] = {
      [APEX_CLEAR_DST] = va, [APEX_CLEAR_DST + 1] = va >> 32,
      [APEX_CLEAR_ROW] = image->levels[level].row_stride,
      [APEX_CLEAR_SLICE] = image->levels[level].slice_stride,
      [APEX_CLEAR_WIDTH] = x1 - x0, [APEX_CLEAR_ROWS] = y1 - y0, [APEX_CLEAR_LAYERS] = layers,
      [APEX_CLEAR_BYTES] = vk_format_get_blocksize(image->vk.format),
   };
   memcpy(&words[APEX_CLEAR_PATTERN], texel, 4 * sizeof(uint32_t));
   memcpy(&words[APEX_CLEAR_MASK], texel + 4, 4 * sizeof(uint32_t));
   uint64_t items = DIV_ROUND_UP(x1 - x0, 64) * (y1 - y0) * layers;
   uint32_t groups = DIV_ROUND_UP(items, 16), limit = MIN2(program->max_workgroups, 1024);
   for (uint32_t base = 0; base < groups; base += limit)
      push_job(cmd, program, NULL, MIN2(groups - base, limit), base, 0, words);
}

static void
pack_color(VkFormat format, const VkClearColorValue *color, uint32_t texel[8])
{
   memset(texel, 0, 4 * sizeof(uint32_t));
   memset(texel + 4, 0xff, 4 * sizeof(uint32_t));
   util_format_pack_rgba(vk_format_to_pipe_format(format), texel, color, 1);
}

static void
pack_depth_stencil(VkFormat format, VkImageAspectFlags aspects, const VkClearDepthStencilValue *value,
                   uint32_t texel[8])
{
   memset(texel, 0, 8 * sizeof(uint32_t));
   enum pipe_format pformat = vk_format_to_pipe_format(format);
   const struct util_format_description *desc = util_format_description(pformat);
   /* Z is swizzle 0 and S swizzle 1 of a depth/stencil description; each
    * aspect packs separately and contributes its bits under its mask. */
   uint8_t stencil = value->stencil;
   for (unsigned i = 0; i < 2; i++) {
      unsigned channel = desc->swizzle[i];
      if (!(aspects & (i ? VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_DEPTH_BIT)) ||
          channel > PIPE_SWIZZLE_W)
         continue;
      const struct util_format_channel_description *ch = &desc->channel[channel];
      uint32_t packed[4] = {0}, mask = BITFIELD_MASK(ch->size) << (ch->shift % 32);
      if (i)
         util_format_pack_s_8uint(pformat, packed, &stencil, 1);
      else
         util_format_pack_z_float(pformat, packed, &value->depth, 1);
      texel[ch->shift / 32] |= packed[ch->shift / 32] & mask;
      texel[4 + ch->shift / 32] |= mask;
   }
}

static void
clear_ranges(struct apex_command_buffer *cmd, struct apex_image *image, uint32_t count,
             const VkImageSubresourceRange *ranges, const uint32_t texel[8])
{
   for (unsigned i = 0; i < count; i++) {
      unsigned levels = vk_image_subresource_level_count(&image->vk, &ranges[i]);
      unsigned layers = vk_image_subresource_layer_count(&image->vk, &ranges[i]);
      for (unsigned l = ranges[i].baseMipLevel; l < ranges[i].baseMipLevel + levels; l++)
         record_clear(cmd, image, l, ranges[i].baseArrayLayer, layers,
                      (VkRect2D){{0, 0}, {image->vk.extent.width, image->vk.extent.height}}, texel);
   }
}

static VKAPI_ATTR void VKAPI_CALL
apex_CmdClearColorImage(VkCommandBuffer handle, VkImage img, VkImageLayout layout,
                       const VkClearColorValue *color, uint32_t count, const VkImageSubresourceRange *ranges)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   VK_FROM_HANDLE(apex_image, image, img);
   uint32_t texel[8];
   pack_color(image->vk.format, color, texel);
   clear_ranges(cmd, image, count, ranges, texel);
}

static VKAPI_ATTR void VKAPI_CALL
apex_CmdClearDepthStencilImage(VkCommandBuffer handle, VkImage img, VkImageLayout layout,
                               const VkClearDepthStencilValue *value, uint32_t count,
                               const VkImageSubresourceRange *ranges)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   VK_FROM_HANDLE(apex_image, image, img);
   for (unsigned i = 0; i < count; i++) {
      uint32_t texel[8];
      pack_depth_stencil(image->vk.format, ranges[i].aspectMask, value, texel);
      clear_ranges(cmd, image, 1, &ranges[i], texel);
   }
}

/* Recording snapshots host data and reserves an address without kernel waits.
 * Submission materializes this immutable source before publishing dependencies. */
static VkResult
bind_map_upload(struct vk_command_buffer *vk, struct vk_meta_device *meta,
                 VkBuffer handle, void **map_out)
{
   struct apex_command_buffer *cmd = (void *)vk;
   struct apex_device *device = (void *)vk->base.device;
   VK_FROM_HANDLE(vk_buffer, buffer, handle);
   struct apex_upload *upload = vk_zalloc(&vk->pool->alloc, sizeof(*upload) + buffer->size,
                                         8, VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!upload)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   upload->size = buffer->size;
   mtx_lock(&device->va_mutex);
   upload->reserved_va = util_vma_heap_alloc(&device->va_heap, align64(upload->size, 4096), 4096);
   mtx_unlock(&device->va_mutex);
   if (!upload->reserved_va) {
      vk_free(&vk->pool->alloc, upload);
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
   }
   buffer->device_address = upload->reserved_va;
   *map_out = upload->data;
   list_addtail(&upload->link, &cmd->uploads);
   return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL
apex_CmdPipelineBarrier2(VkCommandBuffer handle, const VkDependencyInfo *info)
{
   /* Every native dispatch below completes allocation release before the next
    * dispatch acquires its input. Images retain the same linear layout across
    * layout transitions; buffer and image barriers need no extra operation. */
}

static int
native_command(struct apex_device *device, uint32_t operation)
{
   struct apex_ioctl_native r = {.operation = operation};
   return ioctl(device->fd, APEX_IOCTL_NATIVE, &r);
}

/* Rows for one descriptor element. Unwritten descriptors keep the zero rows;
 * no nullDescriptor feature is advertised and unused bindings are skipped. */
static bool
write_descriptor(union apex_descriptor *rows, const struct apex_binding_layout *binding,
                 unsigned element, const struct apex_set_layout *layout,
                 const struct apex_descriptor_set *set, uint32_t dynamic,
                 const void *value)
{
   switch (binding->type) {
   case VK_DESCRIPTOR_TYPE_SAMPLER:
   case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
   case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
   case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT: {
      const VkDescriptorImageInfo *info = value;
      bool sampler = binding->type == VK_DESCRIPTOR_TYPE_SAMPLER;
      bool combined = binding->type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      if (!sampler) {
         VK_FROM_HANDLE(vk_image_view, view, info->imageView);
         if (view) {
            struct apex_image *image = (void *)view->image;
            if (!image->memory)
               return false;
            uint64_t va = image->memory->storage->bo.va + image->offset;
            struct apex_sampled_descriptor d = {
               .low = va, .high = va >> 32,
               .width = image->vk.extent.width, .height = image->vk.extent.height,
               .depth = image->vk.extent.depth, .layers = image->vk.array_layers,
               .base_level = view->base_mip_level, .levels = view->level_count,
               .base_layer = view->base_array_layer, .layer_count = view->layer_count,
               .view_type = view->view_type,
            };
            if (!apex_format_encode(view->format, view->aspects, &view->swizzle, d.format))
               return false;
            const uint32_t *words = (const uint32_t *)&d;
            for (unsigned w = 0; w < 16; w++)
               ((uint32_t *)rows)[w] = util_cpu_to_le32(words[w]);
         }
      }
      if (sampler || combined) {
         const uint32_t *row = NULL;
         if (binding->immutable != ~0u)
            row = layout->samplers[binding->immutable + element];
         else if (info->sampler)
            row = apex_sampler_from_handle(info->sampler)->row;
         uint32_t *out = (uint32_t *)&rows[combined ? 2 : 0];
         for (unsigned w = 0; row && w < 8; w++)
            out[w] = util_cpu_to_le32(row[w]);
      }
      return true;
   }
   case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE: {
      const VkDescriptorImageInfo *info = value;
      VK_FROM_HANDLE(vk_image_view, view, info->imageView);
      if (!view)
         return true;
      struct apex_image *image = (void *)view->image;
      if (!image->memory)
         return false;
      unsigned l = view->base_mip_level;
      uint64_t va = image->memory->storage->bo.va + image->offset + image->levels[l].offset +
                    (uint64_t)view->base_array_layer * image->levels[l].slice_stride;
      rows->image = (struct apex_image_descriptor) {
         util_cpu_to_le32(va), util_cpu_to_le32(va >> 32),
         util_cpu_to_le32(u_minify(image->vk.extent.width, l)),
         util_cpu_to_le32(u_minify(image->vk.extent.height, l)),
         util_cpu_to_le32(view->layer_count),
         util_cpu_to_le32(image->levels[l].row_stride),
         util_cpu_to_le32(image->levels[l].slice_stride), 0,
      };
      return true;
   }
   case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
   case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER: {
      VK_FROM_HANDLE(vk_buffer_view, view, *(const VkBufferView *)value);
      if (!view)
         return true;
      struct apex_buffer *buffer = (void *)view->buffer;
      if (!buffer->memory)
         return false;
      uint64_t va = buffer->memory->storage->bo.va + buffer->offset + view->offset;
      uint32_t format[3];
      if (!apex_format_encode(view->format, VK_IMAGE_ASPECT_COLOR_BIT, NULL, format))
         return false;
      uint32_t words[8] = {va, va >> 32, view->range, view->elements, format[0], format[1], format[2]};
      for (unsigned w = 0; w < 8; w++)
         ((uint32_t *)rows)[w] = util_cpu_to_le32(words[w]);
      return true;
   }
   default: {
      const VkDescriptorBufferInfo *info = value;
      VK_FROM_HANDLE(apex_buffer, buffer, info->buffer);
      if (!buffer)
         return true;
      if (!buffer->memory || info->offset >= buffer->vk.size)
         return false;
      uint64_t range = info->range == VK_WHOLE_SIZE ? buffer->vk.size - info->offset : info->range;
      if ((info->range == VK_WHOLE_SIZE && dynamic) || dynamic % 4 ||
          dynamic > buffer->vk.size - info->offset || !range || range > UINT32_MAX ||
          range > buffer->vk.size - info->offset - dynamic)
         return false;
      uint64_t va = buffer->memory->storage->bo.va + buffer->offset + info->offset + dynamic;
      rows->buffer = (struct apex_buffer_descriptor) {
         util_cpu_to_le32(va), util_cpu_to_le32(va >> 32), util_cpu_to_le32(range), 0,
      };
      return true;
   }
   }
}

static VkResult
drm_prepare(struct apex_device *device, const struct apex_dispatch *dispatch,
            struct apex_bo *table)
{
   struct apex_program *program = dispatch->program;
   if (!program->table)
      return VK_ERROR_FEATURE_NOT_PRESENT;
   size_t push_offset = (program->descriptor_count + 1) * sizeof(union apex_descriptor);
   size_t parameters_offset = apex_program_trailer(program);
   size_t bytes = parameters_offset + sizeof(struct apex_dispatch_parameters) +
                  (dispatch->graphics ? sizeof(dispatch->draw) : 0);
   union apex_descriptor *rows = calloc(1, bytes);
   if (!rows)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   memcpy((uint8_t *)rows + push_offset, dispatch->push, program->push_size);
   struct apex_dispatch_parameters *parameters = (void *)((uint8_t *)rows + parameters_offset);
   for (unsigned axis = 0; axis < 3; axis++) {
      parameters->base[axis] = util_cpu_to_le32(dispatch->parameters.base[axis]);
      parameters->groups[axis] = util_cpu_to_le32(dispatch->parameters.groups[axis]);
   }
   for (unsigned w = 0; w < 2; w++)
      parameters->indirect[w] = util_cpu_to_le32(dispatch->parameters.indirect[w]);
   if (dispatch->graphics) {
      uint32_t *draw = (void *)(parameters + 1);
      for (unsigned i = 0; i < APEX_DRAW_WORDS; i++)
         draw[i] = util_cpu_to_le32(dispatch->draw[i]);
   }
   VkResult result = VK_ERROR_DEVICE_LOST;
   for (unsigned s = 0; s < program->set_count; s++) {
      const struct apex_set_layout *layout = program->set_layouts[s];
      if (!layout || !layout->descriptor_count)
         continue;
      const struct apex_bound_set *bound = dispatch->sets[s];
      const struct apex_descriptor_set *set = bound ? bound->set : NULL;
      for (unsigned b = 0; b < layout->binding_count; b++) {
         const struct apex_binding_layout *binding = &layout->bindings[b];
         unsigned slots = apex_descriptor_slots(binding->type);
         for (unsigned e = 0; e < binding->count; e++) {
            unsigned row = program->set_offsets[s] + binding->slot + e * slots;
            unsigned d = binding->offset + e;
            if (!BITSET_TEST(program->used_descriptors, row))
               continue;
            if (!set || memcmp(set->layout->vk.blake3, layout->vk.blake3, BLAKE3_OUT_LEN))
               goto out;
            if (!write_descriptor(&rows[row], binding, e, layout, set, bound->offsets[d],
                                  &set->descriptors[d]))
               goto out;
         }
      }
   }
   if (!program->bo.handle) {
      result = bo_create(device, program->code.size,
                         APEX_DRM_VM_READ | APEX_DRM_VM_EXEC, 0, 0, &program->bo);
      if (result != VK_SUCCESS)
         goto out;
      memcpy(program->bo.map, program->code.data, program->code.size);
      result = bo_transfer(device, &program->bo, APEX_DRM_TRANSFER_TO_LOCAL,
                           0, program->code.size);
      if (result != VK_SUCCESS) {
         apex_bo_finish(device, &program->bo);
         goto out;
      }
   }
   result = bo_create(device, bytes, APEX_DRM_VM_READ | APEX_DRM_VM_WRITE, 0, 0, table);
   if (result != VK_SUCCESS)
      goto out;
   memcpy(table->map, rows, bytes);
   result = bo_transfer(device, table, APEX_DRM_TRANSFER_TO_LOCAL, 0, bytes);
out:
   free(rows);
   return result;
}

static VkResult
drm_dispatch(struct apex_device *device, const struct apex_dispatch *dispatch)
{
   struct apex_bo table = {0};
   VkResult result = drm_prepare(device, dispatch, &table);
   if (result != VK_SUCCESS)
      goto out;
   struct drm_apex_vm_exec args = {
      .program_va = dispatch->program->bo.va,
      .program_bytes = dispatch->program->code.size,
      .data_va = table.va,
      .workgroups = dispatch->groups,
   };
   /* No implicit data transfer. Host visibility requires flush/invalidate.
    * Do not retry EXEC on EINTR: the kernel cancels/drains the accepted job. */
   result = !ioctl(device->fd, DRM_IOCTL_APEX_VM_EXEC, &args) && args.status == 1 ?
      VK_SUCCESS : VK_ERROR_DEVICE_LOST;
out:
   apex_bo_finish(device, &table);
   return result;
}

static VkResult
dispatch_compute(struct apex_device *device, const struct apex_dispatch *dispatch)
{
   if (device->transport == APEX_TRANSPORT_DRM)
      return drm_dispatch(device, dispatch);
   const VkDescriptorBufferInfo *binding = &dispatch->sets[0]->set->descriptors[0].buffer;
   VK_FROM_HANDLE(apex_buffer, buffer, binding->buffer);
   if (!buffer || !buffer->memory || binding->offset >= buffer->vk.size)
      return VK_ERROR_DEVICE_LOST;
   uint64_t bytes = binding->range == VK_WHOLE_SIZE ? buffer->vk.size - binding->offset : binding->range;
   if (!bytes || bytes > buffer->vk.size - binding->offset)
      return VK_ERROR_DEVICE_LOST;
   void *data = (uint8_t *)buffer->memory->data + buffer->offset + binding->offset;
   if (native_command(device, APEX_NATIVE_CREATE))
      return VK_ERROR_DEVICE_LOST;
   VkResult result = VK_ERROR_DEVICE_LOST;
   struct apex_ioctl_native program = {
      .operation = APEX_NATIVE_ALLOC, .kind = APEX_NATIVE_PROGRAM,
      .user_ptr = (uintptr_t)dispatch->program->code.data, .bytes = dispatch->program->code.size,
   };
   struct apex_ioctl_native allocation = {.operation = APEX_NATIVE_ALLOC, .bytes = bytes};
   if (ioctl(device->fd, APEX_IOCTL_NATIVE, &program) ||
       ioctl(device->fd, APEX_IOCTL_NATIVE, &allocation))
      goto out;
   struct apex_ioctl_native transfer = {
      .operation = APEX_NATIVE_UPLOAD, .handle = allocation.handle,
      .user_ptr = (uintptr_t)data, .bytes = bytes,
   };
   if (ioctl(device->fd, APEX_IOCTL_NATIVE, &transfer) || native_command(device, APEX_NATIVE_START))
      goto out;
   struct apex_ioctl_native submit = {
      .operation = APEX_NATIVE_SUBMIT, .kind = APEX_NATIVE_COMPUTE,
      .handle = program.handle, .data_handle = allocation.handle,
      .workgroups = dispatch->groups == 1 ? 0 : dispatch->groups,
   };
   if (ioctl(device->fd, APEX_IOCTL_NATIVE, &submit))
      goto out;
   uint64_t deadline = os_time_get_nano() + 5000000000ull;
   for (;;) {
      struct apex_ioctl_native poll = {.operation = APEX_NATIVE_POLL, .identity = submit.identity};
      if (ioctl(device->fd, APEX_IOCTL_NATIVE, &poll))
         goto out;
      if (poll.status) {
         if (poll.status != 1)
            goto out;
         break;
      }
      if (os_time_get_nano() >= deadline)
         goto out;
      struct timespec pause = {.tv_nsec = 1000000};
      nanosleep(&pause, NULL);
   }
   if (native_command(device, APEX_NATIVE_STOP))
      goto out;
   transfer.operation = APEX_NATIVE_DOWNLOAD;
   if (ioctl(device->fd, APEX_IOCTL_NATIVE, &transfer))
      goto out;
   result = VK_SUCCESS;
out:
   if (native_command(device, APEX_NATIVE_CLOSE))
      result = VK_ERROR_DEVICE_LOST;
   return result;
}

static void
free_pending(struct apex_device *device, struct apex_pending_dispatch *pending)
{
   list_del(&pending->link);
   apex_bo_finish(device, &pending->table);
   free(pending);
}

static VkResult
reap_descriptors(struct apex_device *device)
{
   uint64_t completed = 0;
   struct drm_syncobj_timeline_array query = {
      .handles = (uintptr_t)&device->completion, .points = (uintptr_t)&completed,
      .count_handles = 1,
   };
   if (ioctl(device->fd, DRM_IOCTL_SYNCOBJ_QUERY, &query))
      return vk_device_set_lost(&device->vk, "Apex completion query failed");
   list_for_each_entry_safe(struct apex_pending_dispatch, pending, &device->retired, link) {
      if (pending->point > completed)
         break;
      free_pending(device, pending);
   }
   return vk_device_check_status(&device->vk);
}

static VkResult
enqueue(struct apex_device *device, struct drm_apex_vm_submit *args,
        const struct drm_apex_sync *inputs, unsigned input_count,
        const struct drm_apex_sync *outputs, unsigned output_count)
{
   args->inputs = input_count ? (uintptr_t)inputs : 0;
   args->input_count = input_count;
   args->outputs = (uintptr_t)outputs;
   args->output_count = output_count;
   if (vk_device_check_status(&device->vk) != VK_SUCCESS)
      return VK_ERROR_DEVICE_LOST;
   /* Never spin on resource pressure: pending work may depend on this very
    * submission for progress. An interrupted enqueue is not retried either. */
   return ioctl(device->fd, DRM_IOCTL_APEX_VM_SUBMIT, args) ?
      vk_device_set_lost(&device->vk, "Apex asynchronous enqueue failed") : VK_SUCCESS;
}

static VkResult
submit_async(struct apex_device *device, struct vk_queue_submit *submit)
{
   struct list_head prepared;
   list_inithead(&prepared);
   VkResult result = reap_descriptors(device);
   if (result != VK_SUCCESS)
      return result;
   /* MAP/TRANSFER may wait prior jobs. Prepare every immutable table before
    * publishing this submission's potentially unsignaled dependencies. */
   for (unsigned i = 0; i < submit->command_buffer_count; i++) {
      struct apex_command_buffer *cmd = (void *)submit->command_buffers[i];
      list_for_each_entry(struct apex_dispatch, dispatch, &cmd->dispatches, link) {
         struct apex_pending_dispatch *pending = calloc(1, sizeof(*pending));
         if (!pending) {
            result = VK_ERROR_OUT_OF_HOST_MEMORY;
            goto out;
         }
         list_addtail(&pending->link, &prepared);
         result = drm_prepare(device, dispatch, &pending->table);
         if (result != VK_SUCCESS)
            goto out;
         pending->args = (struct drm_apex_vm_submit) {
            .program_va = dispatch->program->bo.va,
            .program_bytes = dispatch->program->code.size,
            .data_va = pending->table.va, .workgroups = dispatch->groups,
         };
      }
   }
   /* The common submit thread already waited for fence publication. Timeline
    * zero is a no-op, not the binary point-zero interpretation of this UAPI. */
   for (unsigned i = 0; i < submit->wait_count;) {
      struct drm_apex_sync inputs[APEX_DRM_MAX_SYNCS];
      unsigned count = 0;
      while (i < submit->wait_count && count < ARRAY_SIZE(inputs)) {
         const struct vk_sync_wait *wait = &submit->waits[i++];
         if ((wait->sync->flags & VK_SYNC_IS_TIMELINE) && !wait->wait_value)
            continue;
         struct vk_drm_syncobj *sync = vk_sync_as_drm_syncobj(wait->sync);
         if (!sync) { result = VK_ERROR_FEATURE_NOT_PRESENT; goto out; }
         inputs[count++] = (struct drm_apex_sync) {
            .handle = sync->syncobj, .point = wait->wait_value,
         };
      }
      if (!count)
         continue;
      struct drm_apex_sync done = {.handle = device->completion, .point = device->point + 1};
      struct drm_apex_vm_submit args = {.flags = APEX_DRM_SUBMIT_SYNC_ONLY};
      result = enqueue(device, &args, inputs, count, &done, 1);
      if (result != VK_SUCCESS)
         goto out;
      device->point = done.point;
   }
   /* APEX_DEBUG_JOBS=1 waits for each job and names the first that fails. */
   static int debug_jobs = -1;
   if (debug_jobs < 0)
      debug_jobs = debug_get_bool_option("APEX_DEBUG_JOBS", false);
   list_for_each_entry_safe(struct apex_pending_dispatch, pending, &prepared, link) {
      struct drm_apex_sync done = {.handle = device->completion, .point = device->point + 1};
      result = enqueue(device, &pending->args, NULL, 0, &done, 1);
      if (result != VK_SUCCESS)
         goto out;
      if (debug_jobs) {
         uint64_t point = done.point;
         struct drm_syncobj_timeline_wait wait = {
            .handles = (uintptr_t)&device->completion, .points = (uintptr_t)&point,
            .count_handles = 1, .timeout_nsec = INT64_MAX,
            .flags = DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT,
         };
         ioctl(device->fd, DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT, &wait);
         struct drm_apex_vm_status status = {0};
         if (ioctl(device->fd, DRM_IOCTL_APEX_VM_STATUS, &status) || status.error)
            mesa_loge("Apex job failed: program %u bytes, %u workgroups, error %d",
                      (unsigned)pending->args.program_bytes, pending->args.workgroups, status.error);
      }
      pending->point = device->point = done.point;
      list_del(&pending->link);
      list_addtail(&pending->link, &device->retired);
   }
   /* Ordered sync-only jobs join waits and publish all user signals, including
    * zero-command submissions. Reserve one output for our retirement timeline. */
   unsigned i = 0;
   do {
      struct drm_apex_sync outputs[APEX_DRM_MAX_SYNCS] = {
         {.handle = device->completion, .point = device->point + 1},
      };
      unsigned count = 1;
      while (i < submit->signal_count && count < ARRAY_SIZE(outputs)) {
         const struct vk_sync_signal *signal = &submit->signals[i++];
         struct vk_drm_syncobj *sync = vk_sync_as_drm_syncobj(signal->sync);
         if (!sync) { result = VK_ERROR_FEATURE_NOT_PRESENT; goto out; }
         outputs[count++] = (struct drm_apex_sync) {
            .handle = sync->syncobj, .point = signal->signal_value,
         };
      }
      struct drm_apex_vm_submit args = {.flags = APEX_DRM_SUBMIT_SYNC_ONLY};
      result = enqueue(device, &args, NULL, 0, outputs, count);
      if (result != VK_SUCCESS)
         goto out;
      device->point = outputs[0].point;
   } while (i < submit->signal_count);
out:
   list_for_each_entry_safe(struct apex_pending_dispatch, pending, &prepared, link)
      free_pending(device, pending);
   return result;
}

static VkResult
submit_queue(struct vk_queue *queue, struct vk_queue_submit *submit)
{
   struct apex_device *device = (struct apex_device *)queue->base.device;
   if (vk_queue_submit_has_bind(submit) || submit->is_protected)
      return vk_queue_set_lost(queue, "unsupported Apex submission");
   for (unsigned i = 0; i < submit->command_buffer_count; i++) {
      struct apex_command_buffer *cmd = (void *)submit->command_buffers[i];
      list_for_each_entry(struct apex_upload, upload, &cmd->uploads, link) {
         if (upload->bo.handle)
            continue;
         uint64_t va = upload->reserved_va;
         upload->reserved_va = 0;
         uint32_t access = APEX_DRM_VM_READ | (upload->scratch ? APEX_DRM_VM_WRITE : 0);
         if (bo_create(device, upload->size, access, 0, va, &upload->bo) != VK_SUCCESS)
            return vk_queue_set_lost(queue, "Apex update allocation failed");
         if (upload->scratch)
            continue;
         memcpy(upload->bo.map, upload->data, upload->size);
         if (bo_transfer(device, &upload->bo, APEX_DRM_TRANSFER_TO_LOCAL, 0, upload->size) != VK_SUCCESS)
            return vk_queue_set_lost(queue, "Apex update upload failed");
      }
   }
   if (device->completion) {
      if (submit_async(device, submit) != VK_SUCCESS)
         return vk_queue_set_lost(queue, "Apex asynchronous submission failed");
      return VK_SUCCESS;
   }
   if (vk_sync_wait_many(&device->vk, submit->wait_count, submit->waits,
                        VK_SYNC_WAIT_COMPLETE, UINT64_MAX) != VK_SUCCESS)
      return vk_queue_set_lost(queue, "Apex dependency wait failed");
   for (unsigned i = 0; i < submit->command_buffer_count; i++) {
      struct apex_command_buffer *cmd = (struct apex_command_buffer *)submit->command_buffers[i];
      list_for_each_entry(struct apex_dispatch, dispatch, &cmd->dispatches, link) {
         if (dispatch_compute(device, dispatch) != VK_SUCCESS)
            return vk_queue_set_lost(queue, "Apex dispatch or allocation release failed");
      }
   }
   if (vk_sync_signal_many(&device->vk, submit->signal_count, submit->signals) != VK_SUCCESS)
      return vk_queue_set_lost(queue, "Apex completion signal failed");
   return VK_SUCCESS;
}

static VKAPI_ATTR VkResult VKAPI_CALL
apex_QueueWaitIdle(VkQueue handle)
{
   VK_FROM_HANDLE(vk_queue, queue, handle);
   if (queue->base.device->physical->supported_sync_types)
      return vk_common_QueueWaitIdle(handle);
   return vk_device_is_lost(queue->base.device) ? VK_ERROR_DEVICE_LOST : VK_SUCCESS;
}

static VKAPI_ATTR VkResult VKAPI_CALL
apex_GetFenceStatus(VkDevice dev, VkFence fence)
{
   VK_FROM_HANDLE(apex_device, device, dev);
   VkResult result = vk_common_GetFenceStatus(dev, fence);
   VkResult status = vk_device_check_status(&device->vk);
   return status == VK_SUCCESS ? result : status;
}

static VKAPI_ATTR VkResult VKAPI_CALL
apex_GetSemaphoreCounterValue(VkDevice dev, VkSemaphore semaphore, uint64_t *value)
{
   VK_FROM_HANDLE(apex_device, device, dev);
   VkResult result = vk_common_GetSemaphoreCounterValue(dev, semaphore, value);
   VkResult status = vk_device_check_status(&device->vk);
   return status == VK_SUCCESS ? result : status;
}

VkResult
apex_device_init(struct apex_device *device, struct vk_physical_device *physical,
                  const VkDeviceCreateInfo *info, const VkAllocationCallbacks *alloc, int fd,
                  enum apex_transport transport)
{
   if (info->queueCreateInfoCount != 1)
      return vk_errorf(physical, VK_ERROR_FEATURE_NOT_PRESENT,
                       "Apex requires one queue create info, received %u", info->queueCreateInfoCount);
   if (info->pQueueCreateInfos[0].queueFamilyIndex ||
       info->pQueueCreateInfos[0].queueCount != 1 || info->pQueueCreateInfos[0].flags)
      return vk_errorf(physical, VK_ERROR_FEATURE_NOT_PRESENT,
                       "Apex requires one unflagged queue from family 0");
   bool async = false, prime_coherent = false, host_coherent = false;
   if (transport == APEX_TRANSPORT_DRM) {
      struct drm_apex_info caps = {0};
      if (ioctl(fd, DRM_IOCTL_APEX_INFO, &caps) || caps.version != 2 ||
          (caps.capabilities & (APEX_DRM_CAP_SHMEM | APEX_DRM_CAP_GPUVM)) !=
          (APEX_DRM_CAP_SHMEM | APEX_DRM_CAP_GPUVM))
         return VK_ERROR_INCOMPATIBLE_DRIVER;
      async = (caps.capabilities & APEX_DRM_CAP_ASYNC) && physical->supported_sync_types;
      prime_coherent = caps.capabilities & APEX_DRM_CAP_PRIME_COHERENT;
      host_coherent = caps.capabilities & APEX_DRM_CAP_HOST_COHERENT;
   }
   const struct vk_device_entrypoint_table entrypoints = {
      .CreateComputePipelines = apex_CreateComputePipelines,
      .AllocateMemory = apex_AllocateMemory, .FreeMemory = apex_FreeMemory,
      .GetMemoryFdKHR = apex_GetMemoryFdKHR, .GetMemoryFdPropertiesKHR = apex_GetMemoryFdPropertiesKHR,
      .MapMemory2 = apex_MapMemory2, .UnmapMemory2 = apex_UnmapMemory2,
      .FlushMappedMemoryRanges = apex_FlushMappedMemoryRanges,
      .InvalidateMappedMemoryRanges = apex_InvalidateMappedMemoryRanges,
      .CreateBuffer = apex_CreateBuffer, .DestroyBuffer = apex_DestroyBuffer,
      .DestroyBufferView = apex_DestroyBufferView,
      .GetDeviceBufferMemoryRequirements = apex_GetDeviceBufferMemoryRequirements,
      .GetBufferMemoryRequirements2 = apex_GetBufferMemoryRequirements2,
      .BindBufferMemory2 = apex_BindBufferMemory2,
      .CreateImage = apex_CreateImage, .DestroyImage = apex_DestroyImage,
      .GetDeviceImageMemoryRequirements = apex_GetDeviceImageMemoryRequirements,
      .GetImageMemoryRequirements2 = apex_GetImageMemoryRequirements2,
      .BindImageMemory2 = apex_BindImageMemory2,
      .GetImageSubresourceLayout = apex_GetImageSubresourceLayout,
      .CreateImageView = apex_CreateImageView, .DestroyImageView = apex_DestroyImageView,
      .CreateSampler = apex_CreateSampler, .DestroySampler = apex_DestroySampler,
      .CreateQueryPool = apex_CreateQueryPool, .DestroyQueryPool = apex_DestroyQueryPool,
      .ResetQueryPool = apex_ResetQueryPool, .GetQueryPoolResults = apex_GetQueryPoolResults,
      .CmdResetQueryPool = apex_CmdResetQueryPool,
      .CmdBeginQueryIndexedEXT = apex_CmdBeginQueryIndexedEXT,
      .CmdEndQueryIndexedEXT = apex_CmdEndQueryIndexedEXT,
      .CmdWriteTimestamp2 = apex_CmdWriteTimestamp2,
      .CmdCopyQueryPoolResults = apex_CmdCopyQueryPoolResults,
      .CreateEvent = apex_CreateEvent, .DestroyEvent = apex_DestroyEvent,
      .GetEventStatus = apex_GetEventStatus, .SetEvent = apex_SetEvent, .ResetEvent = apex_ResetEvent,
      .CmdSetEvent2 = apex_CmdSetEvent2, .CmdResetEvent2 = apex_CmdResetEvent2,
      .CmdWaitEvents2 = apex_CmdWaitEvents2,
      .CreateBufferView = apex_CreateBufferView,
      .GetDeviceMemoryCommitment = apex_GetDeviceMemoryCommitment,
      .GetImageSparseMemoryRequirements2 = apex_GetImageSparseMemoryRequirements2,
      .GetDeviceImageSparseMemoryRequirements = apex_GetDeviceImageSparseMemoryRequirements,
      .CreateDescriptorSetLayout = apex_CreateDescriptorSetLayout,
      .CreateDescriptorPool = apex_CreateDescriptorPool, .DestroyDescriptorPool = apex_DestroyDescriptorPool,
      .ResetDescriptorPool = apex_ResetDescriptorPool, .AllocateDescriptorSets = apex_AllocateDescriptorSets,
      .FreeDescriptorSets = apex_FreeDescriptorSets, .UpdateDescriptorSets = apex_UpdateDescriptorSets,
      .BeginCommandBuffer = apex_BeginCommandBuffer, .EndCommandBuffer = apex_EndCommandBuffer,
      .CmdBindPipeline = apex_CmdBindPipeline, .CmdBindDescriptorSets2 = apex_CmdBindDescriptorSets2,
      .CmdPushConstants2 = apex_CmdPushConstants2,
      .CmdDispatch = apex_CmdDispatch, .CmdDispatchIndirect = apex_CmdDispatchIndirect,
      .CmdPipelineBarrier2 = apex_CmdPipelineBarrier2,
      .CmdFillBuffer = apex_CmdFillBuffer, .CmdCopyBuffer2 = apex_CmdCopyBuffer2,
      .CmdUpdateBuffer = apex_CmdUpdateBuffer,
      .CmdCopyBufferToImage2 = apex_CmdCopyBufferToImage2,
      .CmdCopyImageToBuffer2 = apex_CmdCopyImageToBuffer2,
      .CmdCopyImage2 = apex_CmdCopyImage2, .CmdClearColorImage = apex_CmdClearColorImage,
      .CmdBeginRendering = apex_CmdBeginRendering, .CmdEndRendering = apex_CmdEndRendering,
      .CmdClearAttachments = apex_CmdClearAttachments,
      .CmdClearDepthStencilImage = apex_CmdClearDepthStencilImage,
      .CmdBindVertexBuffers2 = apex_CmdBindVertexBuffers2,
      .CmdBindIndexBuffer2 = apex_CmdBindIndexBuffer2,
      .CmdDraw = apex_CmdDraw, .CmdDrawIndexed = apex_CmdDrawIndexed,
      .QueueWaitIdle = apex_QueueWaitIdle,
      .GetFenceStatus = apex_GetFenceStatus,
      .GetSemaphoreCounterValue = apex_GetSemaphoreCounterValue,
   };
   /* Secondary commands own Mesa's copied argument queue. ExecuteCommands
    * replays it through the real table into primary dispatch/upload snapshots. */
   vk_device_dispatch_table_from_entrypoints(&device->cmd_dispatch, &entrypoints, true);
   vk_device_dispatch_table_from_entrypoints(&device->cmd_dispatch,
                                             &vk_common_device_entrypoints, false);
   struct vk_device_dispatch_table dispatch;
   vk_device_dispatch_table_from_entrypoints(&dispatch,
      &vk_cmd_enqueue_unless_primary_device_entrypoints, true);
   vk_device_dispatch_table_from_entrypoints(&dispatch, &entrypoints, false);
   vk_device_dispatch_table_from_entrypoints(&dispatch, &wsi_device_entrypoints, false);
   VkResult result = vk_device_init(&device->vk, physical, &dispatch, info, alloc);
   if (result != VK_SUCCESS)
      return result;
   device->vk.command_dispatch_table = &device->cmd_dispatch;
   device->vk.shader_ops = &apex_device_shader_ops;
   device->vk.command_buffer_ops = &command_ops;
   device->fd = fd;
   device->transport = transport;
   device->prime_coherent = prime_coherent;
   device->host_coherent = host_coherent;
   device->completion = 0;
   device->point = 0;
   memset(device->internal, 0, sizeof(device->internal));
   list_inithead(&device->retired);
   list_inithead(&device->memories);
   if (mtx_init(&device->va_mutex, mtx_plain) != thrd_success) {
      vk_device_finish(&device->vk);
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }
   if (mtx_init(&device->memory_mutex, mtx_plain) != thrd_success) {
      mtx_destroy(&device->va_mutex);
      vk_device_finish(&device->vk);
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }
   util_vma_heap_init(&device->va_heap, 2 * 1024 * 1024, (1ull << 39) - 2 * 1024 * 1024);
   if (transport == APEX_TRANSPORT_DRM)
      vk_device_set_drm_fd(&device->vk, fd);
   if (physical->supported_sync_types) {
      device->vk.copy_sync_payloads = vk_drm_syncobj_copy_payloads;
      vk_device_enable_threaded_submit(&device->vk);
   }
   result = vk_queue_init(&device->queue, &device->vk, &info->pQueueCreateInfos[0], 0);
   if (result != VK_SUCCESS) {
      util_vma_heap_finish(&device->va_heap);
      mtx_destroy(&device->memory_mutex);
      mtx_destroy(&device->va_mutex);
      vk_device_finish(&device->vk);
      return result;
   }
   result = vk_meta_device_init(&device->vk, &device->meta);
   if (result != VK_SUCCESS) {
      vk_queue_finish(&device->queue);
      util_vma_heap_finish(&device->va_heap);
      mtx_destroy(&device->memory_mutex);
      mtx_destroy(&device->va_mutex);
      vk_device_finish(&device->vk);
      return result;
   }
   for (unsigned i = 0; i < VK_META_BUFFER_CHUNK_SIZE_COUNT; i++)
      device->meta.buffer_access.optimal_wg_size[i] = 16;
   device->meta.cmd_bind_map_buffer = bind_map_upload;
   device->queue.driver_submit = submit_queue;
   if (async) {
      struct drm_syncobj_create create = {0};
      if (ioctl(fd, DRM_IOCTL_SYNCOBJ_CREATE, &create)) {
         apex_device_finish(device);
         return VK_ERROR_OUT_OF_HOST_MEMORY;
      }
      device->completion = create.handle;
      device->vk.check_status = check_status;
   }
   if (physical->supported_sync_types) {
      result = vk_queue_enable_submit_thread(&device->queue);
      if (result != VK_SUCCESS) {
         apex_device_finish(device);
         return result;
      }
   }
   return VK_SUCCESS;
}

void
apex_device_finish(struct apex_device *device)
{
   vk_queue_finish(&device->queue);
   /* UNMAP waits queued jobs and retains unsafe backing on a failed drain. */
   list_for_each_entry_safe(struct apex_pending_dispatch, pending, &device->retired, link)
      free_pending(device, pending);
   if (device->completion) {
      struct drm_syncobj_destroy destroy = {.handle = device->completion};
      ioctl(device->fd, DRM_IOCTL_SYNCOBJ_DESTROY, &destroy);
   }
   apex_graphics_finish(device);
   vk_meta_device_finish(&device->vk, &device->meta);
   util_vma_heap_finish(&device->va_heap);
   mtx_destroy(&device->memory_mutex);
   mtx_destroy(&device->va_mutex);
   vk_device_finish(&device->vk);
}
