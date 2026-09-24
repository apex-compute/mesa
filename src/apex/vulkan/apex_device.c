/* SPDX-License-Identifier: MIT */
#include "apex_device.h"
#include "apex_native_uapi.h"
#include "apex_pipeline.h"
#include "drm-uapi/apex_drm.h"
#include "vk_alloc.h"
#include "vk_buffer.h"
#include "vk_cmd_enqueue_entrypoints.h"
#include "vk_command_buffer.h"
#include "vk_command_pool.h"
#include "vk_common_entrypoints.h"
#include "vk_descriptor_set_layout.h"
#include "vk_device_memory.h"
#include "vk_drm_syncobj.h"
#include "vk_image.h"
#include "vk_log.h"
#include "vk_physical_device.h"
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
   uint64_t descriptors[VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC + 1];
   uint64_t used[VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC + 1];
};
struct apex_descriptor_set {
   struct vk_object_base base;
   struct list_head link;
   struct apex_descriptor_pool *pool;
   struct apex_set_layout *layout;
   union {
      VkDescriptorBufferInfo buffer;
      VkDescriptorImageInfo image;
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
   struct apex_pipeline *pipeline;
   struct apex_bound_set *sets[MESA_VK_MAX_DESCRIPTOR_SETS];
   uint32_t groups;
   struct apex_dispatch_parameters parameters;
   uint8_t push[APEX_MAX_PUSH_CONSTANTS];
};
struct apex_pending_dispatch {
   struct list_head link;
   struct apex_bo table;
   struct drm_apex_vm_submit args;
   uint64_t point;
};
struct apex_upload {
   struct list_head link;
   struct apex_bo bo;
   uint64_t reserved_va, size;
   uint8_t data[];
};
struct apex_command_buffer {
   struct vk_command_buffer vk;
   struct list_head dispatches, uploads;
   struct apex_pipeline *pipeline;
   struct apex_bound_set *sets[MESA_VK_MAX_DESCRIPTOR_SETS];
   uint8_t push[APEX_MAX_PUSH_CONSTANTS];
};
VK_DEFINE_NONDISP_HANDLE_CASTS(apex_memory, vk.base, VkDeviceMemory, VK_OBJECT_TYPE_DEVICE_MEMORY);
VK_DEFINE_NONDISP_HANDLE_CASTS(apex_buffer, vk.base, VkBuffer, VK_OBJECT_TYPE_BUFFER);
VK_DEFINE_NONDISP_HANDLE_CASTS(apex_image, vk.base, VkImage, VK_OBJECT_TYPE_IMAGE);
VK_DEFINE_NONDISP_HANDLE_CASTS(apex_descriptor_pool, base, VkDescriptorPool, VK_OBJECT_TYPE_DESCRIPTOR_POOL);
VK_DEFINE_NONDISP_HANDLE_CASTS(apex_descriptor_set, base, VkDescriptorSet, VK_OBJECT_TYPE_DESCRIPTOR_SET);
VK_DEFINE_HANDLE_CASTS(apex_command_buffer, vk.base, VkCommandBuffer, VK_OBJECT_TYPE_COMMAND_BUFFER);

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
gem_create(struct apex_device *device, uint64_t size, uint32_t *handle, void **data)
{
   struct drm_apex_gem_create create = {.size = size};
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
          uint64_t reserved_va, struct apex_bo *bo)
{
   bo->size = align64(size, 4096);
   uint64_t va = reserved_va;
   VkResult result = gem_create(device, bo->size, &bo->handle, &bo->map);
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

static VKAPI_ATTR VkResult VKAPI_CALL
apex_AllocateMemory(VkDevice dev, const VkMemoryAllocateInfo *info,
                    const VkAllocationCallbacks *alloc, VkDeviceMemory *out)
{
   VK_FROM_HANDLE(apex_device, device, dev);
   *out = VK_NULL_HANDLE;
   if (info->memoryTypeIndex > (device->prime_coherent ? 1u : 0u))
      return VK_ERROR_FEATURE_NOT_PRESENT;
   const VkImportMemoryFdInfoKHR *import = vk_find_struct_const(info->pNext, IMPORT_MEMORY_FD_INFO_KHR);
   if (import && !import->handleType)
      import = NULL;
   const VkExportMemoryAllocateInfo *export = vk_find_struct_const(info->pNext, EXPORT_MEMORY_ALLOCATE_INFO);
   VkExternalMemoryHandleTypeFlags types = export ? export->handleTypes : 0;
   if ((import && (import->handleType != VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT &&
                   import->handleType != VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT)) ||
       (types & ~APEX_EXTERNAL_MEMORY_TYPES) ||
       ((import || types) && (!device->prime_coherent || info->memoryTypeIndex != 1)))
      return VK_ERROR_INVALID_EXTERNAL_HANDLE;
   vk_foreach_struct_const(sType, ext, info->pNext) {
      if (sType != VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO &&
          sType != VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR &&
          sType != VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO)
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
                         APEX_DRM_VM_READ | APEX_DRM_VM_WRITE, 0, &mem->storage->bo);
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
   VK_FROM_HANDLE(apex_memory, memory, info->memory);
   *out = NULL;
   if (memory->vk.memory_type_index || info->flags || info->offset >= memory->vk.size ||
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
      if (memory->vk.memory_type_index || ranges[i].offset >= memory->vk.size)
         return VK_ERROR_MEMORY_MAP_FAILED;
      uint64_t bytes = ranges[i].size == VK_WHOLE_SIZE ?
         memory->vk.size - ranges[i].offset : ranges[i].size;
      if (!bytes || bytes > memory->vk.size - ranges[i].offset)
         return VK_ERROR_MEMORY_MAP_FAILED;
      if (device->transport == APEX_TRANSPORT_DRM) {
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
                                       VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT)))
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
apex_GetDeviceBufferMemoryRequirements(VkDevice dev,
   const VkDeviceBufferMemoryRequirements *info, VkMemoryRequirements2 *out)
{
   VK_FROM_HANDLE(apex_device, device, dev);
   const VkExternalMemoryBufferCreateInfo *external =
      vk_find_struct_const(info->pCreateInfo->pNext, EXTERNAL_MEMORY_BUFFER_CREATE_INFO);
   out->memoryRequirements = (VkMemoryRequirements) {
      .size = align64(info->pCreateInfo->size, 64), .alignment = 64,
      .memoryTypeBits = external && external->handleTypes ? (device->prime_coherent ? 2 : 0) :
                         (device->prime_coherent ? 3 : 1),
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

VkResult
apex_image_format_properties(const VkPhysicalDeviceImageFormatInfo2 *info,
                              VkImageFormatProperties2 *properties)
{
   properties->imageFormatProperties = (VkImageFormatProperties){0};
   if (info->format != VK_FORMAT_R32_UINT || info->type != VK_IMAGE_TYPE_2D ||
       (info->tiling != VK_IMAGE_TILING_LINEAR && info->tiling != VK_IMAGE_TILING_OPTIMAL) ||
       info->flags || (info->usage & ~(VK_IMAGE_USAGE_STORAGE_BIT |
                                     VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT)))
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
   const VkPhysicalDeviceExternalImageFormatInfo *external =
      vk_find_struct_const(info->pNext, PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO);
   if (external && external->handleType)
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
   properties->imageFormatProperties = (VkImageFormatProperties) {
      .maxExtent = {4096, 4096, 1}, .maxMipLevels = 13, .maxArrayLayers = 256,
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
      uint32_t row = align(u_minify(info->extent.width, l) * 4, 64);
      uint32_t slice = row * u_minify(info->extent.height, l);
      if (image) {
         image->levels[l].offset = size;
         image->levels[l].row_stride = row;
         image->levels[l].slice_stride = slice;
      }
      size += (uint64_t)slice * info->arrayLayers;
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
       info->sharingMode != VK_SHARING_MODE_EXCLUSIVE ||
       info->samples != VK_SAMPLE_COUNT_1_BIT || !info->extent.width || !info->extent.height ||
       info->extent.width > 4096 || info->extent.height > 4096 || info->extent.depth != 1 ||
       !info->arrayLayers || info->arrayLayers > 256 || !info->mipLevels ||
       info->mipLevels > util_logbase2(MAX2(info->extent.width, info->extent.height)) + 1)
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
image_memory_requirements(VkDeviceSize size, VkMemoryRequirements2 *out)
{
   out->memoryRequirements = (VkMemoryRequirements) {
      .size = size, .alignment = 64, .memoryTypeBits = 1,
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
   image_memory_requirements(image_layout(info->pCreateInfo, NULL), out);
}

static VKAPI_ATTR void VKAPI_CALL
apex_GetImageMemoryRequirements2(VkDevice dev,
   const VkImageMemoryRequirementsInfo2 *info, VkMemoryRequirements2 *out)
{
   VK_FROM_HANDLE(apex_image, image, info->image);
   image_memory_requirements(image->size, out);
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
   *out = VK_NULL_HANDLE;
   if (info->flags || info->format != VK_FORMAT_R32_UINT ||
       (info->viewType != VK_IMAGE_VIEW_TYPE_2D && info->viewType != VK_IMAGE_VIEW_TYPE_2D_ARRAY))
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
   unsigned count = 0, descriptors = 0;
   bool dynamic = false, update_after_bind = false;
   for (unsigned i = 0; i < info->bindingCount; i++) {
      const VkDescriptorSetLayoutBinding *b = &info->pBindings[i];
      VkDescriptorBindingFlags flags = binding_flags && binding_flags->bindingCount ?
         binding_flags->pBindingFlags[i] : 0;
      if (b->binding >= APEX_MAX_BINDINGS || b->descriptorCount > APEX_MAX_DESCRIPTORS - descriptors ||
          (flags & ~VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT) ||
          (b->descriptorType != VK_DESCRIPTOR_TYPE_STORAGE_IMAGE &&
           (b->descriptorType < VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER ||
            b->descriptorType > VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC)) ||
          (device->transport == APEX_TRANSPORT_NATIVE && b->descriptorType != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) ||
          b->stageFlags != VK_SHADER_STAGE_COMPUTE_BIT)
         return VK_ERROR_FEATURE_NOT_PRESENT;
      descriptors += b->descriptorCount;
      count = MAX2(count, b->binding + 1);
      dynamic |= vk_descriptor_type_is_dynamic(b->descriptorType);
      update_after_bind |= flags & VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
   }
   if (dynamic && update_after_bind)
      return VK_ERROR_FEATURE_NOT_PRESENT;
   if (device->transport == APEX_TRANSPORT_NATIVE &&
       (count != 1 || descriptors != 1))
      return VK_ERROR_FEATURE_NOT_PRESENT;
   struct apex_set_layout *layout = vk_descriptor_set_layout_zalloc(&device->vk,
      sizeof(*layout) + count * sizeof(layout->bindings[0]), info);
   if (!layout)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   layout->binding_count = count;
   layout->descriptor_count = descriptors;
   for (unsigned i = 0; i < info->bindingCount; i++) {
      unsigned b = info->pBindings[i].binding;
      layout->bindings[b].count = info->pBindings[i].descriptorCount;
      layout->bindings[b].type = info->pBindings[i].descriptorType;
      layout->counts[layout->bindings[b].type] +=
         layout->bindings[b].count;
      layout->bindings[b].flags = binding_flags && binding_flags->bindingCount ?
         binding_flags->pBindingFlags[i] : 0;
   }
   unsigned offset = 0;
   for (unsigned b = 0; b < count; b++) {
      layout->bindings[b].offset = offset;
      offset += layout->bindings[b].count;
      if (vk_descriptor_type_is_dynamic(layout->bindings[b].type))
         layout->vk.dynamic_descriptor_count += layout->bindings[b].count;
   }
   struct mesa_blake3 hash;
   _mesa_blake3_init(&hash);
   _mesa_blake3_update(&hash, &info->flags, sizeof(info->flags));
   _mesa_blake3_update(&hash, layout->bindings, count * sizeof(layout->bindings[0]));
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
   uint64_t descriptors[VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC + 1] = {0};
   for (unsigned i = 0; i < info->poolSizeCount; i++) {
      VkDescriptorType type = info->pPoolSizes[i].type;
      if (type != VK_DESCRIPTOR_TYPE_STORAGE_IMAGE &&
          (type < VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER || type > VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC))
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
   uint64_t descriptors[VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC + 1] = {0};
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
         if (writes[i].descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
            set->descriptors[start + d].image = writes[i].pImageInfo[d];
         else
            set->descriptors[start + d].buffer = writes[i].pBufferInfo[d];
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
   for (unsigned s = 0; s < ARRAY_SIZE(cmd->sets); s++)
      bound_set_unref(cmd, cmd->sets[s]);
   memset(cmd->sets, 0, sizeof(cmd->sets));
   memset(cmd->push, 0, sizeof(cmd->push));
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
   if (info->stageFlags != VK_SHADER_STAGE_COMPUTE_BIT || first > MESA_VK_MAX_DESCRIPTOR_SETS ||
       count > MESA_VK_MAX_DESCRIPTOR_SETS - first) {
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
      bound_set_unref(cmd, cmd->sets[first + s]);
      cmd->sets[first + s] = bound;
   }
}

static VKAPI_ATTR void VKAPI_CALL
apex_CmdPushConstants2(VkCommandBuffer handle, const VkPushConstantsInfo *info)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   /* Push ranges may name stages unsupported by this compute-only queue. */
   if (!(info->stageFlags & VK_SHADER_STAGE_COMPUTE_BIT))
      return;
   uint32_t offset = info->offset, size = info->size;
   if (offset % 4 || size % 4 ||
       !size || offset >= APEX_MAX_PUSH_CONSTANTS || size > APEX_MAX_PUSH_CONSTANTS - offset) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   memcpy(cmd->push + offset, info->pValues, size);
}

static VKAPI_ATTR void VKAPI_CALL
apex_CmdDispatch(VkCommandBuffer handle, uint32_t x, uint32_t y, uint32_t z)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   if (!x || !y || !z)
      return;
   if (x > 65535 || y > 65535 || z > 65535 || !cmd->pipeline ||
       (!cmd->pipeline->layout && (!cmd->sets[0] || x > 1024 || y != 1 || z != 1))) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   /* Bound each row/chunk by queue count and padded private-storage capacity.
    * Each retains the API grid and its origin. */
   uint32_t limit = cmd->pipeline->max_workgroups;
   for (uint32_t gz = 0; gz < z; gz++)
      for (uint32_t gy = 0; gy < y; gy++)
         for (uint32_t gx = 0; gx < x; gx += limit) {
            struct apex_dispatch *dispatch = vk_alloc(&cmd->vk.pool->alloc, sizeof(*dispatch), 8,
                                                      VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
            if (!dispatch) {
               vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
               return;
            }
            *dispatch = (struct apex_dispatch) {
               .pipeline = cmd->pipeline, .groups = MIN2(x - gx, limit),
               .parameters = {.base = {gx, gy, gz}, .groups = {x, y, z}},
            };
            memcpy(dispatch->sets, cmd->sets, sizeof(dispatch->sets));
            memcpy(dispatch->push, cmd->push, sizeof(dispatch->push));
            for (unsigned s = 0; s < ARRAY_SIZE(dispatch->sets); s++)
               if (dispatch->sets[s])
                  dispatch->sets[s]->refs++;
            list_addtail(&dispatch->link, &cmd->dispatches);
         }
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
          (uint64_t)offset.y * image->levels[level].row_stride + (uint64_t)offset.x * 4;
}

/* Linear image rows use the ordinary cached buffer meta kernels. Neither path
 * binds descriptor sets; preserve the application's pipeline and push image. */
static void
copy_image_rows(struct apex_command_buffer *cmd, uint64_t src, uint32_t src_row, uint64_t src_slice,
                uint64_t dst, uint32_t dst_row, uint64_t dst_slice, VkExtent3D extent, uint32_t layers)
{
   struct apex_device *device = (void *)cmd->vk.base.device;
   struct apex_pipeline *pipeline = cmd->pipeline;
   uint8_t push[APEX_MAX_PUSH_CONSTANTS];
   memcpy(push, cmd->push, sizeof(push));
   for (unsigned z = 0; z < layers; z++) {
      for (unsigned y = 0; y < extent.height; y++) {
         VkDeviceMemoryCopyKHR region = {
            .sType = VK_STRUCTURE_TYPE_DEVICE_MEMORY_COPY_KHR,
            .srcRange = {.address = src + z * src_slice + (uint64_t)y * src_row, .size = extent.width * 4ull},
            .dstRange = {.address = dst + z * dst_slice + (uint64_t)y * dst_row, .size = extent.width * 4ull},
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
      if (to_image)
         copy_image_rows(cmd, buffer_va, layout.row_stride_B, layout.image_stride_B,
                         image_va, image->levels[l].row_stride, image->levels[l].slice_stride,
                         r->imageExtent, r->imageSubresource.layerCount);
      else
         copy_image_rows(cmd, image_va, image->levels[l].row_stride, image->levels[l].slice_stride,
                         buffer_va, layout.row_stride_B, layout.image_stride_B,
                         r->imageExtent, r->imageSubresource.layerCount);
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
                      r->extent, r->srcSubresource.layerCount);
   }
}

static VKAPI_ATTR void VKAPI_CALL
apex_CmdClearColorImage(VkCommandBuffer handle, VkImage img, VkImageLayout layout,
                       const VkClearColorValue *color, uint32_t count, const VkImageSubresourceRange *ranges)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   VK_FROM_HANDLE(apex_image, image, img);
   struct apex_device *device = (void *)cmd->vk.base.device;
   struct apex_pipeline *pipeline = cmd->pipeline;
   uint8_t push[APEX_MAX_PUSH_CONSTANTS];
   memcpy(push, cmd->push, sizeof(push));
   for (unsigned i = 0; i < count; i++) {
      unsigned levels = vk_image_subresource_level_count(&image->vk, &ranges[i]);
      unsigned layers = vk_image_subresource_layer_count(&image->vk, &ranges[i]);
      for (unsigned l = ranges[i].baseMipLevel; l < ranges[i].baseMipLevel + levels; l++) {
         /* Padding belongs to this subresource and may be cleared with it. */
         VkDeviceAddressRangeKHR range = {
            .address = image_address(image, l, ranges[i].baseArrayLayer, (VkOffset3D){0}),
            .size = (uint64_t)layers * image->levels[l].slice_stride,
         };
         vk_meta_fill_memory(&cmd->vk, &device->meta, &range, 0, color->uint32[0]);
      }
   }
   cmd->pipeline = pipeline;
   memcpy(cmd->push, push, sizeof(push));
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

static VkResult
drm_prepare(struct apex_device *device, const struct apex_dispatch *dispatch,
            struct apex_bo *table)
{
   struct apex_pipeline *pipeline = dispatch->pipeline;
   if (!pipeline->layout)
      return VK_ERROR_FEATURE_NOT_PRESENT;
   size_t push_offset = (pipeline->descriptor_count + 1) * sizeof(union apex_descriptor);
   size_t parameters_offset = push_offset + pipeline->push_size;
   size_t bytes = parameters_offset + sizeof(struct apex_dispatch_parameters);
   union apex_descriptor *rows = calloc(1, bytes);
   if (!rows)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   memcpy((uint8_t *)rows + push_offset, dispatch->push, pipeline->push_size);
   struct apex_dispatch_parameters *parameters = (void *)((uint8_t *)rows + parameters_offset);
   for (unsigned axis = 0; axis < 3; axis++) {
      parameters->base[axis] = util_cpu_to_le32(dispatch->parameters.base[axis]);
      parameters->groups[axis] = util_cpu_to_le32(dispatch->parameters.groups[axis]);
   }
   VkResult result = VK_ERROR_DEVICE_LOST;
   for (unsigned s = 0; s < pipeline->layout->set_count; s++) {
      const struct apex_set_layout *layout = (const void *)pipeline->layout->set_layouts[s];
      if (!layout || !layout->descriptor_count)
         continue;
      const struct apex_bound_set *bound = dispatch->sets[s];
      const struct apex_descriptor_set *set = bound ? bound->set : NULL;
      for (unsigned b = 0, d = 0; d < layout->descriptor_count; d++) {
         while (d >= layout->bindings[b].offset + layout->bindings[b].count)
            b++;
         if (!BITSET_TEST(pipeline->used_descriptors, pipeline->set_offsets[s] + d))
            continue;
         if (!set || memcmp(set->layout->vk.blake3, layout->vk.blake3, BLAKE3_OUT_LEN))
            goto out;
         if (layout->bindings[b].type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) {
            VK_FROM_HANDLE(vk_image_view, view, set->descriptors[d].image.imageView);
            if (!view)
               continue;
            struct apex_image *image = (void *)view->image;
            if (!image->memory)
               goto out;
            unsigned l = view->base_mip_level;
            uint64_t va = image->memory->storage->bo.va + image->offset + image->levels[l].offset +
                          (uint64_t)view->base_array_layer * image->levels[l].slice_stride;
            rows[pipeline->set_offsets[s] + d].image = (struct apex_image_descriptor) {
               util_cpu_to_le32(va), util_cpu_to_le32(va >> 32),
               util_cpu_to_le32(u_minify(image->vk.extent.width, l)),
               util_cpu_to_le32(u_minify(image->vk.extent.height, l)),
               util_cpu_to_le32(view->layer_count),
               util_cpu_to_le32(image->levels[l].row_stride),
               util_cpu_to_le32(image->levels[l].slice_stride), 0,
            };
            continue;
         }
         const VkDescriptorBufferInfo *binding = &set->descriptors[d].buffer;
         VK_FROM_HANDLE(apex_buffer, buffer, binding->buffer);
         /* Unwritten descriptors use the zero row. No nullDescriptor feature
          * is advertised. Unused bindings are never dereferenced above. */
         if (!buffer)
            continue;
         if (!buffer->memory || binding->offset >= buffer->vk.size)
            goto out;
         uint64_t range = binding->range == VK_WHOLE_SIZE ?
            buffer->vk.size - binding->offset : binding->range;
         uint64_t dynamic = bound->offsets[d];
         if ((binding->range == VK_WHOLE_SIZE && dynamic) || dynamic % 4 ||
             dynamic > buffer->vk.size - binding->offset || !range || range > UINT32_MAX ||
             range > buffer->vk.size - binding->offset - dynamic)
            goto out;
         uint64_t va = buffer->memory->storage->bo.va + buffer->offset + binding->offset + dynamic;
         rows[pipeline->set_offsets[s] + d].buffer = (struct apex_buffer_descriptor) {
            util_cpu_to_le32(va), util_cpu_to_le32(va >> 32), util_cpu_to_le32(range), 0,
         };
      }
   }
   if (!pipeline->program.handle) {
      result = bo_create(device, pipeline->code.size,
                         APEX_DRM_VM_READ | APEX_DRM_VM_EXEC, 0, &pipeline->program);
      if (result != VK_SUCCESS)
         goto out;
      memcpy(pipeline->program.map, pipeline->code.data, pipeline->code.size);
      result = bo_transfer(device, &pipeline->program, APEX_DRM_TRANSFER_TO_LOCAL,
                           0, pipeline->code.size);
      if (result != VK_SUCCESS) {
         apex_bo_finish(device, &pipeline->program);
         goto out;
      }
   }
   result = bo_create(device, bytes, APEX_DRM_VM_READ | APEX_DRM_VM_WRITE, 0, table);
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
      .program_va = dispatch->pipeline->program.va,
      .program_bytes = dispatch->pipeline->code.size,
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
      .user_ptr = (uintptr_t)dispatch->pipeline->code.data, .bytes = dispatch->pipeline->code.size,
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
            .program_va = dispatch->pipeline->program.va,
            .program_bytes = dispatch->pipeline->code.size,
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
   list_for_each_entry_safe(struct apex_pending_dispatch, pending, &prepared, link) {
      struct drm_apex_sync done = {.handle = device->completion, .point = device->point + 1};
      result = enqueue(device, &pending->args, NULL, 0, &done, 1);
      if (result != VK_SUCCESS)
         goto out;
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
         if (bo_create(device, upload->size, APEX_DRM_VM_READ, va, &upload->bo) != VK_SUCCESS)
            return vk_queue_set_lost(queue, "Apex update allocation failed");
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
   bool async = false, prime_coherent = false;
   if (transport == APEX_TRANSPORT_DRM) {
      struct drm_apex_info caps = {0};
      if (ioctl(fd, DRM_IOCTL_APEX_INFO, &caps) || caps.version != 2 ||
          (caps.capabilities & (APEX_DRM_CAP_SHMEM | APEX_DRM_CAP_GPUVM)) !=
          (APEX_DRM_CAP_SHMEM | APEX_DRM_CAP_GPUVM))
         return VK_ERROR_INCOMPATIBLE_DRIVER;
      async = (caps.capabilities & APEX_DRM_CAP_ASYNC) && physical->supported_sync_types;
      prime_coherent = caps.capabilities & APEX_DRM_CAP_PRIME_COHERENT;
   }
   const struct vk_device_entrypoint_table entrypoints = {
      .CreateComputePipelines = apex_CreateComputePipelines,
      .AllocateMemory = apex_AllocateMemory, .FreeMemory = apex_FreeMemory,
      .GetMemoryFdKHR = apex_GetMemoryFdKHR, .GetMemoryFdPropertiesKHR = apex_GetMemoryFdPropertiesKHR,
      .MapMemory2 = apex_MapMemory2, .UnmapMemory2 = apex_UnmapMemory2,
      .FlushMappedMemoryRanges = apex_FlushMappedMemoryRanges,
      .InvalidateMappedMemoryRanges = apex_InvalidateMappedMemoryRanges,
      .CreateBuffer = apex_CreateBuffer, .DestroyBuffer = apex_DestroyBuffer,
      .GetDeviceBufferMemoryRequirements = apex_GetDeviceBufferMemoryRequirements,
      .GetBufferMemoryRequirements2 = apex_GetBufferMemoryRequirements2,
      .BindBufferMemory2 = apex_BindBufferMemory2,
      .CreateImage = apex_CreateImage, .DestroyImage = apex_DestroyImage,
      .GetDeviceImageMemoryRequirements = apex_GetDeviceImageMemoryRequirements,
      .GetImageMemoryRequirements2 = apex_GetImageMemoryRequirements2,
      .BindImageMemory2 = apex_BindImageMemory2,
      .GetImageSubresourceLayout = apex_GetImageSubresourceLayout,
      .CreateImageView = apex_CreateImageView, .DestroyImageView = apex_DestroyImageView,
      .CreateDescriptorSetLayout = apex_CreateDescriptorSetLayout,
      .CreateDescriptorPool = apex_CreateDescriptorPool, .DestroyDescriptorPool = apex_DestroyDescriptorPool,
      .ResetDescriptorPool = apex_ResetDescriptorPool, .AllocateDescriptorSets = apex_AllocateDescriptorSets,
      .FreeDescriptorSets = apex_FreeDescriptorSets, .UpdateDescriptorSets = apex_UpdateDescriptorSets,
      .BeginCommandBuffer = apex_BeginCommandBuffer, .EndCommandBuffer = apex_EndCommandBuffer,
      .CmdBindPipeline = apex_CmdBindPipeline, .CmdBindDescriptorSets2 = apex_CmdBindDescriptorSets2,
      .CmdPushConstants2 = apex_CmdPushConstants2,
      .CmdDispatch = apex_CmdDispatch, .CmdPipelineBarrier2 = apex_CmdPipelineBarrier2,
      .CmdFillBuffer = apex_CmdFillBuffer, .CmdCopyBuffer2 = apex_CmdCopyBuffer2,
      .CmdUpdateBuffer = apex_CmdUpdateBuffer,
      .CmdCopyBufferToImage2 = apex_CmdCopyBufferToImage2,
      .CmdCopyImageToBuffer2 = apex_CmdCopyImageToBuffer2,
      .CmdCopyImage2 = apex_CmdCopyImage2, .CmdClearColorImage = apex_CmdClearColorImage,
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
   VkResult result = vk_device_init(&device->vk, physical, &dispatch, info, alloc);
   if (result != VK_SUCCESS)
      return result;
   device->vk.command_dispatch_table = &device->cmd_dispatch;
   device->vk.command_buffer_ops = &command_ops;
   device->fd = fd;
   device->transport = transport;
   device->prime_coherent = prime_coherent;
   device->completion = 0;
   device->point = 0;
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
   vk_meta_device_finish(&device->vk, &device->meta);
   util_vma_heap_finish(&device->va_heap);
   mtx_destroy(&device->memory_mutex);
   mtx_destroy(&device->va_mutex);
   vk_device_finish(&device->vk);
}
