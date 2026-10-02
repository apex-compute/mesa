/* SPDX-License-Identifier: MIT */
#include "apex_private.h"
#include "apex_format.h"
#include "apex_pipeline.h"
#include "drm-uapi/apex_drm.h"
#include "drm-uapi/drm_fourcc.h"
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
#include "vk_descriptor_update_template.h"
#include "wsi_common.h"
#include "util/format/u_format.h"
#include "util/log.h"
#include "util/u_debug.h"
#include <errno.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

static int
gem_close(struct apex_device *device, uint32_t handle)
{
   struct drm_gem_close args = {.handle = handle};
   return ioctl(device->fd, DRM_IOCTL_GEM_CLOSE, &args);
}

/* The status page reports queue faults without an ioctl; QUEUE_STATUS names
 * the fault and the file's sticky error. */
static VkResult
check_status(struct vk_device *vk)
{
   struct apex_device *device = (void *)vk;
   const volatile uint32_t *status = device->status_map;
   if (!(status[APEX_STATUS_STATE / 4] & (APEX_STATUS_FAULTED | APEX_STATUS_RESET)))
      return VK_SUCCESS;
   struct drm_apex_queue_status query = {.queue_id = device->queue_id};
   int ret = ioctl(device->fd, DRM_IOCTL_APEX_QUEUE_STATUS, &query);
   return vk_device_set_lost(vk, "Apex queue fault (state %#x, status %u, unit %u, address %#" PRIx64
                             ", errno %d)", ret ? 0 : query.state, query.fault_status,
                             query.fault_unit, (uint64_t)query.fault_address,
                             ret ? errno : query.error);
}

static int
vm_bind(struct apex_device *device, uint32_t op, uint32_t flags, uint32_t handle,
        uint64_t va, uint64_t bytes)
{
   struct drm_apex_vm_bind_op bind_op = {
      .op = op, .flags = flags, .handle = handle, .va = va, .bytes = bytes,
   };
   struct drm_apex_vm_bind bind = {.ops = (uintptr_t)&bind_op, .op_count = 1};
   return ioctl(device->fd, DRM_IOCTL_APEX_VM_BIND, &bind);
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
   /* After device loss, UNMAP could wait on work of the lost queue and delay
    * file close. Keep the mapping and VA reserved until close; the kernel VM
    * retains backing independently of the GEM handle. */
   if (bo->va && !vk_device_is_lost(&device->vk)) {
      if (vm_bind(device, APEX_VM_UNMAP, 0, 0, bo->va, bo->size)) {
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
VkResult
apex_bo_create(struct apex_device *device, uint64_t size, uint32_t flags,
          uint32_t gem_flags, uint64_t reserved_va, struct apex_bo *bo)
{
   bo->size = align64(size, 4096);
   bo->system = gem_flags & APEX_GEM_SYSTEM;
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
   if (va && !vm_bind(device, APEX_VM_MAP, flags, bo->handle, va, bo->size)) {
      bo->va = va;
      return VK_SUCCESS;
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
   if (size <= 0 || size > APEX_MAX_ALLOCATION || size % 4096 ||
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
   /* Foreign dma-bufs import as SYSTEM mappings without CPU access. */
   storage->bo = (struct apex_bo){.handle = prime.handle, .size = extent, .system = true};
   mtx_lock(&device->va_mutex);
   uint64_t va = util_vma_heap_alloc(&device->va_heap, extent, 4096);
   mtx_unlock(&device->va_mutex);
   if (!va || vm_bind(device, APEX_VM_MAP, APEX_VM_READ | APEX_VM_WRITE, prime.handle, va, extent)) {
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

/* Memory types: 0 explicit-transfer host, 1 device-only PRIME, the optional
 * host-coherent SYSTEM type, then LOCAL-resident storage with PRIME. */
static uint32_t
local_memory_types(const struct apex_device *device)
{
   return device->prime_coherent ? 1u << (2 + device->host_coherent) : 0;
}

static VKAPI_ATTR VkResult VKAPI_CALL
apex_GetMemoryFdKHR(VkDevice dev, const VkMemoryGetFdInfoKHR *info, int *fd)
{
   VK_FROM_HANDLE(apex_device, device, dev);
   VK_FROM_HANDLE(apex_memory, memory, info->memory);
   *fd = -1;
   if (!(info->handleType & APEX_EXTERNAL_MEMORY_TYPES) ||
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

/* Type 0 is LOCAL (BAR2 write-combined), type 1 SYSTEM (cached shmem); both
 * are host coherent. */
#define APEX_MEMORY_TYPES 3u

static VKAPI_ATTR VkResult VKAPI_CALL
apex_GetMemoryFdPropertiesKHR(VkDevice dev, VkExternalMemoryHandleTypeFlagBits type,
                            int fd, VkMemoryFdPropertiesKHR *properties)
{
   VK_FROM_HANDLE(apex_device, device, dev);
   properties->memoryTypeBits = 0;
   if (type != VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT || !dma_buf_size(fd))
      return VK_ERROR_INVALID_EXTERNAL_HANDLE;
   mtx_lock(&device->memory_mutex);
   struct drm_prime_handle prime = {.fd = fd};
   int ret = ioctl(device->fd, DRM_IOCTL_PRIME_FD_TO_HANDLE, &prime);
   if (!ret && !find_memory(device, prime.handle))
      ret = gem_close(device, prime.handle);
   mtx_unlock(&device->memory_mutex);
   if (ret)
      return VK_ERROR_INVALID_EXTERNAL_HANDLE;
   properties->memoryTypeBits = APEX_MEMORY_TYPES;
   return VK_SUCCESS;
}

static VKAPI_ATTR VkResult VKAPI_CALL
apex_AllocateMemory(VkDevice dev, const VkMemoryAllocateInfo *info,
                    const VkAllocationCallbacks *alloc, VkDeviceMemory *out)
{
   VK_FROM_HANDLE(apex_device, device, dev);
   *out = VK_NULL_HANDLE;
   if (info->memoryTypeIndex > 1)
      return VK_ERROR_FEATURE_NOT_PRESENT;
   bool local = local_memory_types(device) & (1u << info->memoryTypeIndex);
   const VkImportMemoryFdInfoKHR *import = vk_find_struct_const(info->pNext, IMPORT_MEMORY_FD_INFO_KHR);
   if (import && !import->handleType)
      import = NULL;
   const VkExportMemoryAllocateInfo *export = vk_find_struct_const(info->pNext, EXPORT_MEMORY_ALLOCATE_INFO);
   VkExternalMemoryHandleTypeFlags types = export ? export->handleTypes : 0;
   if ((import && (import->handleType != VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT &&
                   import->handleType != VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT)) ||
       (types & ~APEX_EXTERNAL_MEMORY_TYPES) ||
       /* PRIME exports LOCAL objects. Imports take either type (Zink binds
        * them from its device-local heap): the exporter owns the placement
        * and the import is mapped by GPU VA only, routed as SYSTEM. */
       (types && !import && info->memoryTypeIndex != 0))
      return VK_ERROR_INVALID_EXTERNAL_HANDLE;
   /* Every allocation has a device address; capture/replay is unsupported. */
   const VkMemoryAllocateFlagsInfo *flags = vk_find_struct_const(info->pNext, MEMORY_ALLOCATE_FLAGS_INFO);
   if (flags && (flags->flags & VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT))
      return VK_ERROR_FEATURE_NOT_PRESENT;
   vk_foreach_struct_const(sType, ext, info->pNext) {
      if (sType != VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO &&
          sType != VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO &&
          sType != VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR &&
          sType != VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO &&
          sType != VK_STRUCTURE_TYPE_WSI_MEMORY_ALLOCATE_INFO_MESA)
         return VK_ERROR_FEATURE_NOT_PRESENT;
      /* Dedicated resources use ordinary storage with a zero binding offset. */
   }
   if ((!info->allocationSize && (!import || types)) || info->allocationSize > APEX_MAX_ALLOCATION) {
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
   }
   struct apex_memory *mem = vk_device_memory_create(&device->vk, info, alloc, sizeof(*mem));
   if (!mem)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   VkResult result;
   if (import) {
      result = import_memory(device, import->fd, info->allocationSize, &mem->storage);
   } else {
      mem->storage = vk_zalloc(&device->vk.alloc, sizeof(*mem->storage), 8,
                               VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
      if (!mem->storage) {
         vk_device_memory_destroy(&device->vk, alloc, &mem->vk);
         return VK_ERROR_OUT_OF_HOST_MEMORY;
      }
      result = apex_bo_create(device, info->allocationSize, APEX_VM_READ | APEX_VM_WRITE,
                         info->memoryTypeIndex == 1 ? APEX_GEM_SYSTEM : 0, 0, &mem->storage->bo);
      if (result == VK_SUCCESS) {
         mem->data = mem->storage->bo.map;
         mem->storage->refs = 1;
         mtx_lock(&device->memory_mutex);
         list_addtail(&mem->storage->link, &device->memories);
         mtx_unlock(&device->memory_mutex);
      } else {
         vk_free(&device->vk.alloc, mem->storage);
      }
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
   mtx_lock(&device->memory_mutex);
   if (!--memory->storage->refs) {
      list_del(&memory->storage->link);
      apex_bo_finish(device, &memory->storage->bo);
      vk_free(&device->vk.alloc, memory->storage);
   }
   mtx_unlock(&device->memory_mutex);
   vk_device_memory_destroy(&device->vk, alloc, &memory->vk);
}

static VKAPI_ATTR VkResult VKAPI_CALL
apex_MapMemory2(VkDevice dev, const VkMemoryMapInfo *info, void **out)
{
   VK_FROM_HANDLE(apex_memory, memory, info->memory);
   *out = NULL;
   if (!memory->data || info->flags || info->offset >= memory->vk.size ||
       (info->size != VK_WHOLE_SIZE && info->size > memory->vk.size - info->offset))
      return VK_ERROR_MEMORY_MAP_FAILED;
   if (!memory->data) {
      VkResult result = map_storage(device, memory);
      if (result != VK_SUCCESS)
         return result;
   }
   *out = (uint8_t *)memory->data + info->offset;
   return VK_SUCCESS;
}

static VKAPI_ATTR VkResult VKAPI_CALL
apex_UnmapMemory2(VkDevice dev, const VkMemoryUnmapInfo *info)
{
   return info->flags ? VK_ERROR_FEATURE_NOT_PRESENT : VK_SUCCESS;
}

/* Both memory types are host coherent: ranges are only validated. */
static VkResult
mapped_memory_ranges(uint32_t count, const VkMappedMemoryRange *ranges)
{
   for (unsigned i = 0; i < count; i++) {
      VK_FROM_HANDLE(apex_memory, memory, ranges[i].memory);
      if (!memory->data || ranges[i].offset >= memory->vk.size)
         return VK_ERROR_MEMORY_MAP_FAILED;
      uint64_t bytes = ranges[i].size == VK_WHOLE_SIZE ?
         memory->vk.size - ranges[i].offset : ranges[i].size;
      if (!bytes || bytes > memory->vk.size - ranges[i].offset)
         return VK_ERROR_MEMORY_MAP_FAILED;
   }
   return VK_SUCCESS;
}

static VKAPI_ATTR VkResult VKAPI_CALL
apex_FlushMappedMemoryRanges(VkDevice dev, uint32_t count, const VkMappedMemoryRange *ranges)
{
   return mapped_memory_ranges(count, ranges);
}

static VKAPI_ATTR VkResult VKAPI_CALL
apex_InvalidateMappedMemoryRanges(VkDevice dev, uint32_t count, const VkMappedMemoryRange *ranges)
{
   return mapped_memory_ranges(count, ranges);
}

static VKAPI_ATTR VkResult VKAPI_CALL
apex_CreateBuffer(VkDevice dev, const VkBufferCreateInfo *info,
                  const VkAllocationCallbacks *alloc, VkBuffer *out)
{
   VK_FROM_HANDLE(apex_device, device, dev);
   *out = VK_NULL_HANDLE;
   const VkExternalMemoryBufferCreateInfo *external =
      vk_find_struct_const(info->pNext, EXTERNAL_MEMORY_BUFFER_CREATE_INFO);
   if (external && (external->handleTypes & ~APEX_EXTERNAL_MEMORY_TYPES))
      return VK_ERROR_INVALID_EXTERNAL_HANDLE;
   if (info->flags || info->sharingMode != VK_SHARING_MODE_EXCLUSIVE ||
       (vk_buffer_usage_flags(info) & ~(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                                       VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                       VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                       VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT |
                                       VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT |
                                       VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                                       VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                       VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT |
                                       VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_COUNTER_BUFFER_BIT_EXT |
                                       VK_BUFFER_USAGE_CONDITIONAL_RENDERING_BIT_EXT)))
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
   out->memoryRequirements = (VkMemoryRequirements) {
      .size = align64(info->pCreateInfo->size, 64), .alignment = 64,
      .memoryTypeBits = APEX_MEMORY_TYPES,
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
   vk_common_GetBufferMemoryRequirements2(dev, info, out);
}

static VKAPI_ATTR VkResult VKAPI_CALL
apex_BindBufferMemory2(VkDevice dev, uint32_t count, const VkBindBufferMemoryInfo *infos)
{
   for (unsigned i = 0; i < count; i++) {
      VK_FROM_HANDLE(apex_memory, mem, infos[i].memory);
      VK_FROM_HANDLE(apex_buffer, buffer, infos[i].buffer);
      if (infos[i].memoryOffset % 64 || infos[i].memoryOffset > mem->vk.size ||
          buffer->vk.size > mem->vk.size - infos[i].memoryOffset)
         return VK_ERROR_OUT_OF_DEVICE_MEMORY;
      buffer->memory = mem;
      buffer->offset = infos[i].memoryOffset;
      buffer->vk.device_address = (mem->storage ? mem->storage->bo.va : 0) + buffer->offset;
   }
   return VK_SUCCESS;
}

/* Image features follow the texture unit's and ROP's formats; texel and
 * storage access converts through format words; vertex fetch reads the
 * vertex input kinds. Linear images are 2D, single-level and single-sampled. */
static VkFormatFeatureFlags2
image_features(VkFormat format, bool linear)
{
   const VkFormatFeatureFlags2 transfer = VK_FORMAT_FEATURE_2_TRANSFER_SRC_BIT |
                                          VK_FORMAT_FEATURE_2_TRANSFER_DST_BIT;
   /* ETC2/EAC images sample their decoded plane. */
   if (apex_decoded_format(format, NULL))
      return linear ? 0 : transfer | VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT |
                          VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_FILTER_LINEAR_BIT |
                          VK_FORMAT_FEATURE_2_BLIT_SRC_BIT;
   enum pipe_format pformat = vk_format_to_pipe_format(format);
   const struct util_format_description *desc = util_format_description(pformat);
   if (!desc || desc->block.width != 1 || desc->block.height != 1)
      return 0;
   VkFormatFeatureFlags2 features = transfer;
   bool depth = vk_format_has_depth(format), stencil = vk_format_has_stencil(format);
   uint8_t code, channel[4];
   if (apex_hw_texture_format(format, depth ? VK_IMAGE_ASPECT_DEPTH_BIT : stencil ?
                              VK_IMAGE_ASPECT_STENCIL_BIT : VK_IMAGE_ASPECT_COLOR_BIT, &code, channel) &&
       !(linear && (depth || stencil))) {
      features |= VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT;
      unsigned class = code & 15, type = code >> 4 & 7;
      /* The texture unit filters every normalized and float format but
       * four-channel binary32. */
      bool filter = type != APEX_HW_UINT && type != APEX_HW_SINT && !stencil &&
                    !(class == APEX_HW_CLASS_RGBA32 && type == APEX_HW_FLOAT);
      if (filter)
         features |= VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_FILTER_LINEAR_BIT |
                     VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_FILTER_MINMAX_BIT;
      /* Comparison applies to D16, D24 and R32F images. */
      if (depth && format != VK_FORMAT_D32_SFLOAT_S8_UINT)
         features |= VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_DEPTH_COMPARISON_BIT;
      if (!vk_format_is_scaled(format))
         features |= VK_FORMAT_FEATURE_2_BLIT_SRC_BIT;
   }
   uint32_t words[3];
   bool storage = !depth && !stencil && !util_format_is_srgb(pformat) &&
                  util_is_power_of_two_nonzero(desc->block.bits / 8) && desc->block.bits >= 8 &&
                  apex_format_encode(format, VK_IMAGE_ASPECT_COLOR_BIT, NULL, words);
   if (storage)
      features |= VK_FORMAT_FEATURE_2_STORAGE_IMAGE_BIT |
                  VK_FORMAT_FEATURE_2_STORAGE_READ_WITHOUT_FORMAT_BIT |
                  VK_FORMAT_FEATURE_2_STORAGE_WRITE_WITHOUT_FORMAT_BIT;
   if (storage && (format == VK_FORMAT_R32_UINT || format == VK_FORMAT_R32_SINT))
      features |= VK_FORMAT_FEATURE_2_STORAGE_IMAGE_ATOMIC_BIT;
   if (apex_hw_depth_format(format) && !linear)
      features |= VK_FORMAT_FEATURE_2_DEPTH_STENCIL_ATTACHMENT_BIT | VK_FORMAT_FEATURE_2_BLIT_DST_BIT;
   if (apex_hw_color_format(format)) {
      features |= VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_2_BLIT_DST_BIT;
      if (!util_format_is_pure_integer(pformat))
         features |= VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BLEND_BIT;
   }
   return features;
}

/* Tiling features for images; texel and vertex-buffer features for buffers. */
VkFormatFeatureFlags2
apex_format_features(VkFormat format, bool buffer)
{
   /* Subsampled and multi-planar YCbCr formats are unsupported, as are
    * values without a Mesa format (maintenance5 allows any enumerant) and
    * D16_UNORM_S8_UINT. */
   if (format == VK_FORMAT_UNDEFINED || format == VK_FORMAT_D16_UNORM_S8_UINT ||
       vk_format_get_ycbcr_info(format) || vk_format_to_pipe_format(format) == PIPE_FORMAT_NONE ||
       (format >= VK_FORMAT_G8B8G8R8_422_UNORM && format <= VK_FORMAT_G16_B16_R16_3PLANE_444_UNORM))
      return 0;
   if (!buffer)
      return image_features(format, false);
   enum pipe_format pformat = vk_format_to_pipe_format(format);
   const struct util_format_description *desc = util_format_description(pformat);
   VkFormatFeatureFlags2 features = 0;
   uint32_t words[3];
   if (util_format_is_srgb(pformat) || !desc || desc->layout != UTIL_FORMAT_LAYOUT_PLAIN ||
       !apex_format_encode(format, VK_IMAGE_ASPECT_COLOR_BIT, NULL, words))
      return 0;
   features |= VK_FORMAT_FEATURE_2_UNIFORM_TEXEL_BUFFER_BIT;
   if (!vk_format_is_depth_or_stencil(format) && desc->block.bits <= 128) {
      features |= VK_FORMAT_FEATURE_2_STORAGE_TEXEL_BUFFER_BIT |
                  VK_FORMAT_FEATURE_2_STORAGE_READ_WITHOUT_FORMAT_BIT |
                  VK_FORMAT_FEATURE_2_STORAGE_WRITE_WITHOUT_FORMAT_BIT;
      if (format == VK_FORMAT_R32_UINT || format == VK_FORMAT_R32_SINT)
         features |= VK_FORMAT_FEATURE_2_STORAGE_TEXEL_BUFFER_ATOMIC_BIT;
   }
   bool swap;
   if (apex_hw_vertex_format(format, &swap))
      features |= VK_FORMAT_FEATURE_2_VERTEX_BUFFER_BIT;
   return features;
}

/* Linear tiling features. */
static VkFormatFeatureFlags2
linear_features(VkFormat format)
{
   return apex_format_features(format, false) ? image_features(format, true) : 0;
}

/* DRM_FORMAT_MOD_LINEAR is the one modifier: the linear allocation of a
 * single-level, single-layer 2D color image, which other devices can import. */
bool
apex_format_modifier_supported(VkFormat format)
{
   return linear_features(format) && !vk_format_is_depth_or_stencil(format);
}

VkFormatFeatureFlags2
apex_linear_format_features(VkFormat format)
{
   return linear_features(format);
}

#define APEX_MAX_IMAGE_2D 8192u
#define APEX_MAX_IMAGE_3D 512u
#define APEX_MAX_LAYERS 2048u

VkResult
apex_image_format_properties(const VkPhysicalDeviceImageFormatInfo2 *info, bool prime,
                             VkImageFormatProperties2 *properties)
{
   properties->imageFormatProperties = (VkImageFormatProperties){0};
   VkExternalImageFormatProperties *external_props =
      vk_find_struct(properties->pNext, EXTERNAL_IMAGE_FORMAT_PROPERTIES);
   if (external_props)
      external_props->externalMemoryProperties = (VkExternalMemoryProperties){0};
   bool drm = info->tiling == VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT;
   bool linear = info->tiling == VK_IMAGE_TILING_LINEAR || drm;
   VkFormatFeatureFlags2 features = linear ? linear_features(info->format) :
                                    apex_format_features(info->format, false);
   /* Extended usage needs each usage from one of the view formats. */
   const VkImageFormatListCreateInfo *view_formats =
      vk_find_struct_const(info->pNext, IMAGE_FORMAT_LIST_CREATE_INFO);
   VkFormatFeatureFlags2 view_features = features;
   if ((info->flags & VK_IMAGE_CREATE_EXTENDED_USAGE_BIT) && view_formats)
      for (unsigned f = 0; f < view_formats->viewFormatCount; f++)
         if (vk_format_get_blocksize(view_formats->pViewFormats[f]) == vk_format_get_blocksize(info->format))
            view_features |= linear ? linear_features(view_formats->pViewFormats[f]) :
                             apex_format_features(view_formats->pViewFormats[f], false);
   if (drm) {
      const VkPhysicalDeviceImageDrmFormatModifierInfoEXT *modifier =
         vk_find_struct_const(info->pNext, PHYSICAL_DEVICE_IMAGE_DRM_FORMAT_MODIFIER_INFO_EXT);
      if (!modifier || modifier->drmFormatModifier != DRM_FORMAT_MOD_LINEAR ||
          !apex_format_modifier_supported(info->format))
         return VK_ERROR_FORMAT_NOT_SUPPORTED;
   }
   VkImageUsageFlags usage = 0;
   if (view_features & VK_FORMAT_FEATURE_2_TRANSFER_SRC_BIT) usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
   if (view_features & VK_FORMAT_FEATURE_2_TRANSFER_DST_BIT) usage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
   if (view_features & VK_FORMAT_FEATURE_2_STORAGE_IMAGE_BIT) usage |= VK_IMAGE_USAGE_STORAGE_BIT;
   if (view_features & VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BIT) usage |= VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
   if (view_features & VK_FORMAT_FEATURE_2_DEPTH_STENCIL_ATTACHMENT_BIT)
      usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
   if (view_features & VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_BIT)
      usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
   if (view_features & (VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_2_DEPTH_STENCIL_ATTACHMENT_BIT))
      usage |= VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT;
   /* Input attachments fetch through the texture unit. */
   if (usage & VK_IMAGE_USAGE_SAMPLED_BIT &&
       view_features & (VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_2_DEPTH_STENCIL_ATTACHMENT_BIT))
      usage |= VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT;
   const VkImageCreateFlags flags = VK_IMAGE_CREATE_ALIAS_BIT | VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT |
      VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT | VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT |
      VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;
   if (!features || info->type > VK_IMAGE_TYPE_3D ||
       (info->tiling != VK_IMAGE_TILING_LINEAR && info->tiling != VK_IMAGE_TILING_OPTIMAL && !drm) ||
       (info->flags & ~flags) || (info->usage & ~usage) ||
       (linear && (info->type != VK_IMAGE_TYPE_2D ||
                   (info->flags & (VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT | VK_IMAGE_CREATE_2D_ARRAY_COMPATIBLE_BIT)))) ||
       ((info->flags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT) && info->type != VK_IMAGE_TYPE_2D))
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
   /* External images are PRIME-shared bytes of the same allocation,
    * importable and exportable as opaque fds or dma-bufs. ETC2/EAC images
    * keep a private decoded plane and stay internal. */
   const VkPhysicalDeviceExternalImageFormatInfo *external =
      vk_find_struct_const(info->pNext, PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO);
   if (external && external->handleType) {
      if (!prime || apex_decoded_format(info->format, NULL) ||
          (external->handleType != VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT &&
           external->handleType != VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT))
         return VK_ERROR_FORMAT_NOT_SUPPORTED;
      if (external_props)
         external_props->externalMemoryProperties = (VkExternalMemoryProperties) {
            .externalMemoryFeatures = VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT |
                                      VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT,
            .exportFromImportedHandleTypes = APEX_EXTERNAL_MEMORY_TYPES,
            .compatibleHandleTypes = APEX_EXTERNAL_MEMORY_TYPES,
         };
   }
   bool three_d = info->type == VK_IMAGE_TYPE_3D;
   unsigned extent = three_d ? APEX_MAX_IMAGE_3D : APEX_MAX_IMAGE_2D;
   properties->imageFormatProperties = (VkImageFormatProperties) {
      .maxExtent = {extent, info->type == VK_IMAGE_TYPE_1D ? 1 : extent, three_d ? APEX_MAX_IMAGE_3D : 1},
      .maxMipLevels = linear ? 1 : util_logbase2(extent) + 1,
      .maxArrayLayers = three_d || linear ? 1 : APEX_MAX_LAYERS,
      .sampleCounts = VK_SAMPLE_COUNT_1_BIT, .maxResourceSize = 1ull << 31,
   };
   /* Multisampling: 2D optimal attachments of 2, 4 or 8 samples. */
   if (info->type == VK_IMAGE_TYPE_2D && !linear && !(info->flags & VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT) &&
       (features & (VK_FORMAT_FEATURE_2_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_2_DEPTH_STENCIL_ATTACHMENT_BIT)) &&
       !(info->usage & VK_IMAGE_USAGE_STORAGE_BIT))
      properties->imageFormatProperties.sampleCounts |= VK_SAMPLE_COUNT_2_BIT | VK_SAMPLE_COUNT_4_BIT |
                                                        VK_SAMPLE_COUNT_8_BIT;
   return VK_SUCCESS;
}

/* The texture unit's tiled layout for optimal images of 1-, 2-, 4-, 8- or
 * 16-byte texels; a linear layout of rows of texels or compressed blocks
 * otherwise. */
static VkDeviceSize
image_layout(const VkImageCreateInfo *info, struct apex_image *image)
{
   VkFormat format = info->format;
   unsigned bytes = vk_format_get_blocksize(format);
   unsigned bw = vk_format_get_blockwidth(format), bh = vk_format_get_blockheight(format);
   bool three_d = info->imageType == VK_IMAGE_TYPE_3D;
   bool tiled = info->tiling == VK_IMAGE_TILING_OPTIMAL && bw == 1 && bh == 1 &&
                util_is_power_of_two_nonzero(bytes) && bytes <= 16;
   const VkExtent3D blocks = {DIV_ROUND_UP(info->extent.width, bw), DIV_ROUND_UP(info->extent.height, bh),
                              info->extent.depth};
   apex_hw_layout_init(&image->layout, bytes, three_d, blocks, info->mipLevels, info->arrayLayers,
                       info->samples, tiled);
   VkDeviceSize size = image->layout.size;
   image->decoded = 0;
   VkFormat decoded = apex_decoded_format(format, NULL);
   if (decoded) {
      apex_hw_layout_init(&image->decoded_layout, vk_format_get_blocksize(decoded), three_d, info->extent,
                          info->mipLevels, info->arrayLayers, 1, true);
      image->decoded = align64(size, APEX_HW_TILE);
      size = image->decoded + image->decoded_layout.size;
   }
   return size;
}

const struct apex_hw_layout *
apex_image_layout(const struct apex_image *image, VkFormat view_format, uint64_t *va)
{
   *va = apex_image_va(image);
   if (apex_decoded_format(view_format, NULL)) {
      *va += image->decoded;
      return &image->decoded_layout;
   }
   return &image->layout;
}

static VKAPI_ATTR VkResult VKAPI_CALL
apex_CreateImage(VkDevice dev, const VkImageCreateInfo *info,
                  const VkAllocationCallbacks *alloc, VkImage *out)
{
   VK_FROM_HANDLE(apex_device, device, dev);
   *out = VK_NULL_HANDLE;
   const VkExternalMemoryImageCreateInfo *external =
      vk_find_struct_const(info->pNext, EXTERNAL_MEMORY_IMAGE_CREATE_INFO);
   VkExternalMemoryHandleTypeFlags external_types = external ? external->handleTypes : 0;
   if (external_types & ~APEX_EXTERNAL_MEMORY_TYPES)
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
   /* DRM tiling selects LINEAR from a list, or takes an explicit layout whose
    * pitch is the linear layout's own. */
   const VkImageDrmFormatModifierListCreateInfoEXT *modifiers =
      vk_find_struct_const(info->pNext, IMAGE_DRM_FORMAT_MODIFIER_LIST_CREATE_INFO_EXT);
   const VkImageDrmFormatModifierExplicitCreateInfoEXT *explicit_layout =
      vk_find_struct_const(info->pNext, IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT);
   VkDeviceSize plane_offset = 0;
   if (info->tiling == VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT) {
      bool linear = false;
      for (unsigned m = 0; modifiers && m < modifiers->drmFormatModifierCount; m++)
         linear |= modifiers->pDrmFormatModifiers[m] == DRM_FORMAT_MOD_LINEAR;
      if (explicit_layout) {
         if (explicit_layout->drmFormatModifier != DRM_FORMAT_MOD_LINEAR)
            return VK_ERROR_FORMAT_NOT_SUPPORTED;
         struct apex_image layout;
         image_layout(info, &layout);
         const VkSubresourceLayout *plane = explicit_layout->pPlaneLayouts;
         if (explicit_layout->drmFormatModifierPlaneCount != 1 || plane->offset % 64 ||
             plane->rowPitch != layout.layout.level[0].pitch)
            return VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT;
         plane_offset = plane->offset;
      } else if (!linear) {
         return VK_ERROR_FORMAT_NOT_SUPPORTED;
      }
   }
   const VkImageFormatListCreateInfo *view_formats =
      vk_find_struct_const(info->pNext, IMAGE_FORMAT_LIST_CREATE_INFO);
   VkImageFormatListCreateInfo view_list = view_formats ? *view_formats : (VkImageFormatListCreateInfo){0};
   view_list.pNext = NULL;
   VkPhysicalDeviceImageDrmFormatModifierInfoEXT modifier_info = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_DRM_FORMAT_MODIFIER_INFO_EXT,
      .pNext = view_formats ? &view_list : NULL,
      .drmFormatModifier = DRM_FORMAT_MOD_LINEAR,
   };
   VkPhysicalDeviceExternalImageFormatInfo external_info = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO,
      .pNext = &modifier_info,
   };
   VkPhysicalDeviceImageFormatInfo2 format = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2, .pNext = &external_info,
      .format = info->format, .type = info->imageType, .tiling = info->tiling,
      .usage = info->usage, .flags = info->flags,
   };
   VkImageFormatProperties2 props = {.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2};
   /* Each requested handle type must be supported on its own. */
   bool supported = true;
   for (unsigned t = 0; supported && t < 2; t++) {
      external_info.handleType = t ? external_types & VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT :
                                     external_types & VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
      supported = apex_image_format_properties(&format, &props) == VK_SUCCESS;
   }
   const VkExtent3D *max = &props.imageFormatProperties.maxExtent;
   if (!supported ||
       !(props.imageFormatProperties.sampleCounts & info->samples) ||
       info->mipLevels > props.imageFormatProperties.maxMipLevels ||
       info->arrayLayers > props.imageFormatProperties.maxArrayLayers ||
       !info->extent.width || !info->extent.height || !info->extent.depth ||
       info->extent.width > max->width || info->extent.height > max->height ||
       info->extent.depth > max->depth || !info->arrayLayers || !info->mipLevels ||
       info->mipLevels > util_logbase2(MAX3(info->extent.width, info->extent.height,
                                            info->extent.depth)) + 1)
      return VK_ERROR_FORMAT_NOT_SUPPORTED;
   struct apex_image probe;
   VkDeviceSize size = image_layout(info, &probe);
   if (size > props.imageFormatProperties.maxResourceSize)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
   struct apex_image *image = vk_image_create(&device->vk, info, alloc, sizeof(*image));
   if (!image)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   image->size = image_layout(info, image);
   image->plane_offset = plane_offset;
   if (info->tiling == VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT)
      image->vk.drm_format_mod = DRM_FORMAT_MOD_LINEAR;
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

/* Tiled images start on a 4 KiB tile; linear ones on 64 bytes. */
static void
image_memory_requirements(VkDeviceSize size, bool tiled, VkMemoryRequirements2 *out)
{
   out->memoryRequirements = (VkMemoryRequirements) {
      .size = size, .alignment = tiled ? APEX_HW_TILE : 64, .memoryTypeBits = APEX_MEMORY_TYPES,
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
   const VkExternalMemoryImageCreateInfo *external =
      vk_find_struct_const(info->pCreateInfo->pNext, EXTERNAL_MEMORY_IMAGE_CREATE_INFO);
   const VkImageDrmFormatModifierExplicitCreateInfoEXT *explicit_layout =
      vk_find_struct_const(info->pCreateInfo->pNext, IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT);
   VkDeviceSize plane_offset = info->pCreateInfo->tiling == VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT &&
      explicit_layout ? explicit_layout->pPlaneLayouts[0].offset : 0;
   struct apex_image image;
   VkDeviceSize size = image_layout(info->pCreateInfo, &image);
   image_memory_requirements(plane_offset + size, image.layout.tiled || image.decoded, out);
}

static VKAPI_ATTR void VKAPI_CALL
apex_GetImageMemoryRequirements2(VkDevice dev,
   const VkImageMemoryRequirementsInfo2 *info, VkMemoryRequirements2 *out)
{
   VK_FROM_HANDLE(apex_device, device, dev);
   VK_FROM_HANDLE(apex_image, image, info->image);
   image_memory_requirements(image->plane_offset + image->size, image->layout.tiled || image->decoded, out);
}

static VKAPI_ATTR VkResult VKAPI_CALL
apex_BindImageMemory2(VkDevice dev, uint32_t count, const VkBindImageMemoryInfo *infos)
{
   VK_FROM_HANDLE(apex_device, device, dev);
   for (unsigned i = 0; i < count; i++) {
      VK_FROM_HANDLE(apex_memory, mem, infos[i].memory);
      VK_FROM_HANDLE(apex_image, image, infos[i].image);
      if (image->vk.external_handle_types &&
          !(local_memory_types(device) & (1u << mem->vk.memory_type_index)))
         return VK_ERROR_INVALID_EXTERNAL_HANDLE;
      /* The image starts at its explicit DRM plane offset. */
      VkDeviceSize offset = infos[i].memoryOffset + image->plane_offset;
      unsigned alignment = image->layout.tiled || image->decoded ? APEX_HW_TILE : 64;
      if (infos[i].memoryOffset % alignment || offset > mem->vk.size || image->size > mem->vk.size - offset)
         return VK_ERROR_OUT_OF_DEVICE_MEMORY;
      image->memory = mem;
      image->offset = offset;
   }
   return VK_SUCCESS;
}

/* A subresource is one layer of a level, or every depth slice of a 3D level;
 * DRM memory plane 0 is level 0, layer 0. Offsets include an explicit plane
 * offset. Tiled levels report no row pitch. */
static void
subresource_layout(const struct apex_image *image, const VkImageSubresource *subresource,
                   VkSubresourceLayout2 *out)
{
   const struct apex_hw_layout *layout = &image->layout;
   const struct apex_hw_level *l = &layout->level[subresource->mipLevel];
   unsigned planes = image->vk.image_type == VK_IMAGE_TYPE_3D ? l->depth : layout->samples;
   out->subresourceLayout = (VkSubresourceLayout) {
      .offset = image->plane_offset + l->offset + layout->layer_stride * subresource->arrayLayer,
      .size = l->plane * planes,
      .rowPitch = layout->tiled ? 0 : l->pitch,
      .arrayPitch = layout->layer_stride,
      .depthPitch = l->plane,
   };
}

static VKAPI_ATTR void VKAPI_CALL
apex_GetImageSubresourceLayout2(VkDevice dev, VkImage handle,
   const VkImageSubresource2 *subresource, VkSubresourceLayout2 *out)
{
   subresource_layout(apex_image_from_handle(handle), &subresource->imageSubresource, out);
}

/* The layout an image of these parameters would have. */
static VKAPI_ATTR void VKAPI_CALL
apex_GetDeviceImageSubresourceLayout(VkDevice dev, const VkDeviceImageSubresourceInfo *info,
                                     VkSubresourceLayout2 *out)
{
   struct apex_image image = {.plane_offset = 0};
   image.vk.image_type = info->pCreateInfo->imageType;
   image.vk.extent = info->pCreateInfo->extent;
   image_layout(info->pCreateInfo, &image);
   subresource_layout(&image, &info->pSubresource->imageSubresource, out);
}

static VKAPI_ATTR VkResult VKAPI_CALL
apex_CreateImageView(VkDevice dev, const VkImageViewCreateInfo *info,
                      const VkAllocationCallbacks *alloc, VkImageView *out)
{
   VK_FROM_HANDLE(apex_device, device, dev);
   VK_FROM_HANDLE(apex_image, image, info->image);
   *out = VK_NULL_HANDLE;
   /* Views may reinterpret texels of the same size (mutable-format images). */
   if ((info->flags & ~VK_IMAGE_VIEW_CREATE_DRIVER_INTERNAL_BIT_MESA) || !image ||
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
   apex_hw_sampler_descriptor(info, &sampler->vk, sampler->row);
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

/* Set storage elements and table rows of one binding; `count` is the API
 * descriptorCount (bytes for inline uniform blocks). */
static uint64_t
binding_elements(VkDescriptorType type, uint32_t count)
{
   return type == VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK ?
      DIV_ROUND_UP(count, sizeof(((struct apex_descriptor_set *)0)->descriptors[0])) : count;
}

static uint64_t
binding_rows(VkDescriptorType type, uint32_t count)
{
   return type == VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK ?
      (count ? 1 + DIV_ROUND_UP(count, sizeof(union apex_descriptor)) : 0) :
      (uint64_t)count * apex_descriptor_slots(type);
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
   VkDescriptorSetLayoutCreateFlags supported = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT |
                                                VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT;
   if (info->flags & ~supported)
      return VK_ERROR_FEATURE_NOT_PRESENT;
   unsigned count = 0, descriptors = 0, slots = 0, immutable = 0;
   bool dynamic = false, update_after_bind = false;
   for (unsigned i = 0; i < info->bindingCount; i++) {
      const VkDescriptorSetLayoutBinding *b = &info->pBindings[i];
      VkDescriptorBindingFlags flags = binding_flags && binding_flags->bindingCount ?
         binding_flags->pBindingFlags[i] : 0;
      bool type_ok = apex_descriptor_bucket(b->descriptorType) < APEX_DESCRIPTOR_BUCKETS;
      bool inline_block = b->descriptorType == VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK;
      if (b->binding >= APEX_MAX_BINDINGS || !type_ok ||
          (inline_block && b->descriptorCount > APEX_MAX_INLINE_BYTES) ||
          binding_rows(b->descriptorType, b->descriptorCount) > APEX_MAX_DESCRIPTORS - slots ||
          (flags & ~VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT))
         return VK_ERROR_FEATURE_NOT_PRESENT;
      descriptors += binding_elements(b->descriptorType, b->descriptorCount);
      slots += binding_rows(b->descriptorType, b->descriptorCount);
      if (b->pImmutableSamplers && (b->descriptorType == VK_DESCRIPTOR_TYPE_SAMPLER ||
                                    b->descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER))
         immutable += b->descriptorCount;
      count = MAX2(count, b->binding + 1);
      dynamic |= vk_descriptor_type_is_dynamic(b->descriptorType);
      update_after_bind |= flags & VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
   }
   if (dynamic && update_after_bind)
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
      bool inline_block = binding->descriptorType == VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK;
      layout->bindings[b].count = inline_block ? !!binding->descriptorCount : binding->descriptorCount;
      layout->bindings[b].bytes = inline_block ? binding->descriptorCount : 0;
      layout->bindings[b].type = binding->descriptorType;
      layout->bindings[b].stages = binding->stageFlags;
      layout->counts[apex_descriptor_bucket(binding->descriptorType)] += binding->descriptorCount;
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
      unsigned api_count = layout->bindings[b].bytes ? layout->bindings[b].bytes : layout->bindings[b].count;
      offset += binding_elements(layout->bindings[b].type, api_count);
      slot += binding_rows(layout->bindings[b].type, api_count);
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
   uint64_t descriptors[APEX_DESCRIPTOR_BUCKETS] = {0};
   for (unsigned i = 0; i < info->poolSizeCount; i++) {
      unsigned bucket = apex_descriptor_bucket(info->pPoolSizes[i].type);
      if (bucket >= ARRAY_SIZE(descriptors))
         return VK_ERROR_FEATURE_NOT_PRESENT;
      descriptors[bucket] += info->pPoolSizes[i].descriptorCount;
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

void
apex_descriptor_set_free(struct vk_device *device, struct apex_descriptor_set *set)
{
   list_del(&set->link);
   if (set->pool) {
      set->pool->allocated--;
      for (unsigned type = 0; type < ARRAY_SIZE(set->layout->counts); type++)
         set->pool->used[type] -= set->layout->counts[type];
   }
   vk_descriptor_set_layout_unref(device, &set->layout->vk);
   vk_object_free(device, NULL, set);
}

static VKAPI_ATTR VkResult VKAPI_CALL
apex_ResetDescriptorPool(VkDevice dev, VkDescriptorPool handle, VkDescriptorPoolResetFlags flags)
{
   VK_FROM_HANDLE(apex_device, device, dev);
   VK_FROM_HANDLE(apex_descriptor_pool, pool, handle);
   list_for_each_entry_safe(struct apex_descriptor_set, set, &pool->sets, link)
      apex_descriptor_set_free(&device->vk, set);
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
         apex_descriptor_set_free(&device->vk, set);
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
   uint64_t descriptors[APEX_DESCRIPTOR_BUCKETS] = {0};
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

/* Stores descriptor `count` values of one binding starting at `element`;
 * value i is at data + i * stride. */
static void
write_set(struct apex_descriptor_set *set, uint32_t binding, uint32_t element, uint32_t count,
          VkDescriptorType type, const void *data, size_t stride)
{
   assert(binding < set->layout->binding_count && type == set->layout->bindings[binding].type);
   if (type == VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK) {
      /* `element` and `count` are byte offset and size. */
      assert(element + count <= set->layout->bindings[binding].bytes);
      memcpy((uint8_t *)&set->descriptors[set->layout->bindings[binding].offset] + element, data, count);
      return;
   }
   unsigned start = set->layout->bindings[binding].offset + element;
   assert(start + count <= set->layout->descriptor_count);
   for (unsigned d = 0; d < count; d++) {
      const void *value = (const uint8_t *)data + d * stride;
      switch (type) {
      case VK_DESCRIPTOR_TYPE_SAMPLER:
      case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
      case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
      case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
      case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
         set->descriptors[start + d].image = *(const VkDescriptorImageInfo *)value;
         break;
      case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
      case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
         set->descriptors[start + d].texel = *(const VkBufferView *)value;
         break;
      default:
         set->descriptors[start + d].buffer = *(const VkDescriptorBufferInfo *)value;
         break;
      }
   }
}

void
apex_descriptor_set_write(struct apex_descriptor_set *set, const VkWriteDescriptorSet *w)
{
   switch (w->descriptorType) {
   case VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK: {
      const VkWriteDescriptorSetInlineUniformBlock *block =
         vk_find_struct_const(w->pNext, WRITE_DESCRIPTOR_SET_INLINE_UNIFORM_BLOCK);
      write_set(set, w->dstBinding, w->dstArrayElement, block->dataSize, w->descriptorType,
                block->pData, 0);
      break;
   }
   case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
   case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
      write_set(set, w->dstBinding, w->dstArrayElement, w->descriptorCount, w->descriptorType,
                w->pTexelBufferView, sizeof(VkBufferView));
      break;
   case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER: case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
   case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC: case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
      write_set(set, w->dstBinding, w->dstArrayElement, w->descriptorCount, w->descriptorType,
                w->pBufferInfo, sizeof(VkDescriptorBufferInfo));
      break;
   default:
      write_set(set, w->dstBinding, w->dstArrayElement, w->descriptorCount, w->descriptorType,
                w->pImageInfo, sizeof(VkDescriptorImageInfo));
      break;
   }
}

void
apex_descriptor_set_write_template(struct apex_descriptor_set *set, const struct vk_descriptor_update_template *templ,
               const void *data)
{
   for (uint32_t e = 0; e < templ->entry_count; e++) {
      const struct vk_descriptor_template_entry *entry = &templ->entries[e];
      write_set(set, entry->binding, entry->array_element, entry->array_count, entry->type,
                (const uint8_t *)data + entry->offset, entry->stride);
   }
}

/* Capture/replay addresses are unsupported: bufferDeviceAddressCaptureReplay
 * is not advertised, so opaque addresses are zero. */
static VKAPI_ATTR uint64_t VKAPI_CALL
apex_GetBufferOpaqueCaptureAddress(VkDevice dev, const VkBufferDeviceAddressInfo *info)
{
   return 0;
}

static VKAPI_ATTR uint64_t VKAPI_CALL
apex_GetDeviceMemoryOpaqueCaptureAddress(VkDevice dev, const VkDeviceMemoryOpaqueCaptureAddressInfo *info)
{
   return 0;
}

/* Support is exactly successful layout creation. */
static VKAPI_ATTR void VKAPI_CALL
apex_GetDescriptorSetLayoutSupport(VkDevice dev, const VkDescriptorSetLayoutCreateInfo *info,
                                   VkDescriptorSetLayoutSupport *support)
{
   VK_FROM_HANDLE(vk_device, device, dev);
   VkDescriptorSetLayout layout;
   support->supported = apex_CreateDescriptorSetLayout(dev, info, NULL, &layout) == VK_SUCCESS;
   if (support->supported)
      device->dispatch_table.DestroyDescriptorSetLayout(dev, layout, NULL);
}

static VKAPI_ATTR void VKAPI_CALL
apex_UpdateDescriptorSetWithTemplate(VkDevice dev, VkDescriptorSet handle,
                                     VkDescriptorUpdateTemplate templ, const void *data)
{
   apex_descriptor_set_write_template(apex_descriptor_set_from_handle(handle),
                  vk_descriptor_update_template_from_handle(templ), data);
}

static VKAPI_ATTR void VKAPI_CALL
apex_UpdateDescriptorSets(VkDevice dev, uint32_t write_count, const VkWriteDescriptorSet *writes,
                          uint32_t copy_count, const VkCopyDescriptorSet *copies)
{
   for (unsigned i = 0; i < write_count; i++)
      apex_descriptor_set_write(apex_descriptor_set_from_handle(writes[i].dstSet), &writes[i]);
   for (unsigned i = 0; i < copy_count; i++) {
      VK_FROM_HANDLE(apex_descriptor_set, src, copies[i].srcSet);
      VK_FROM_HANDLE(apex_descriptor_set, dst, copies[i].dstSet);
      const struct apex_binding_layout *source = &src->layout->bindings[copies[i].srcBinding];
      if (source->type == VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK) {
         write_set(dst, copies[i].dstBinding, copies[i].dstArrayElement, copies[i].descriptorCount,
                   source->type, (uint8_t *)&src->descriptors[source->offset] + copies[i].srcArrayElement, 0);
         continue;
      }
      unsigned from = src->layout->bindings[copies[i].srcBinding].offset + copies[i].srcArrayElement;
      unsigned to = dst->layout->bindings[copies[i].dstBinding].offset + copies[i].dstArrayElement;
      assert(from + copies[i].descriptorCount <= src->layout->descriptor_count &&
             to + copies[i].descriptorCount <= dst->layout->descriptor_count);
      memmove(&dst->descriptors[to], &src->descriptors[from],
              copies[i].descriptorCount * sizeof(src->descriptors[0]));
   }
}

/* ---- Events and queries --------------------------------------------------- */

/* Small host-read device words in cached SYSTEM memory. */
static VkResult
host_words_create(struct apex_device *device, uint64_t size, struct apex_bo *bo)
{
   return apex_bo_create(device, size, APEX_VM_READ | APEX_VM_WRITE, APEX_GEM_SYSTEM, 0, bo);
}


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
   return VK_SUCCESS;
}

static VKAPI_ATTR VkResult VKAPI_CALL
apex_GetEventStatus(VkDevice dev, VkEvent handle)
{
   VK_FROM_HANDLE(apex_device, device, dev);
   VK_FROM_HANDLE(apex_event, event, handle);
   VkResult result = vk_device_check_status(&device->vk);
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


static VKAPI_ATTR VkResult VKAPI_CALL
apex_CreateQueryPool(VkDevice dev, const VkQueryPoolCreateInfo *info,
                     const VkAllocationCallbacks *alloc, VkQueryPool *out)
{
   VK_FROM_HANDLE(apex_device, device, dev);
   *out = VK_NULL_HANDLE;
   if (info->queryType != VK_QUERY_TYPE_OCCLUSION && info->queryType != VK_QUERY_TYPE_TIMESTAMP &&
       info->queryType != VK_QUERY_TYPE_TRANSFORM_FEEDBACK_STREAM_EXT)
      return VK_ERROR_FEATURE_NOT_PRESENT;
   struct apex_query_pool *pool = vk_query_pool_create(&device->vk, info, alloc, sizeof(*pool));
   if (!pool)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   VkResult result = host_words_create(device, (uint64_t)MAX2(info->queryCount, 1) * APEX_QUERY_STRIDE,
                                       &pool->bo);
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
   VK_FROM_HANDLE(apex_query_pool, pool, handle);
   memset((uint8_t *)pool->bo.map + (uint64_t)first * APEX_QUERY_STRIDE, 0,
          (uint64_t)count * APEX_QUERY_STRIDE);
}

static VKAPI_ATTR VkResult VKAPI_CALL
apex_GetQueryPoolResults(VkDevice dev, VkQueryPool handle, uint32_t first, uint32_t count,
                         size_t size, void *data, VkDeviceSize stride, VkQueryResultFlags flags)
{
   VK_FROM_HANDLE(apex_device, device, dev);
   VK_FROM_HANDLE(apex_query_pool, pool, handle);
   const uint8_t *slots = (const uint8_t *)pool->bo.map + (uint64_t)first * APEX_QUERY_STRIDE;
   unsigned values = pool->vk.query_type == VK_QUERY_TYPE_TRANSFORM_FEEDBACK_STREAM_EXT ? 2 : 1;
   unsigned width = flags & VK_QUERY_RESULT_64_BIT ? 8 : 4;
   VkResult status = VK_SUCCESS;
   for (uint32_t q = 0; q < count; q++) {
      const volatile uint32_t *slot = (const void *)(slots + (uint64_t)q * APEX_QUERY_STRIDE);
      for (;;) {
         VkResult result = vk_device_check_status(&device->vk);
         if (result != VK_SUCCESS)
            return result;
         if (slot[APEX_QUERY_AVAILABLE / 4] || !(flags & VK_QUERY_RESULT_WAIT_BIT))
            break;
         /* Results arrive from the queue; poll without holding locks. */
         nanosleep(&(struct timespec){.tv_nsec = 100000}, NULL);
      }
      bool available = slot[APEX_QUERY_AVAILABLE / 4];
      uint8_t *out = (uint8_t *)data + q * stride;
      for (unsigned v = 0; v < values && (available || (flags & VK_QUERY_RESULT_PARTIAL_BIT)); v++) {
         uint64_t value = slot[2 * v] | (uint64_t)slot[2 * v + 1] << 32;
         if (width == 8)
            memcpy(out + 8 * v, &value, 8);
         else
            *(uint32_t *)(out + 4 * v) = value > UINT32_MAX ? UINT32_MAX : value;
      }
      if (!available)
         status = VK_NOT_READY;
      if (flags & VK_QUERY_RESULT_WITH_AVAILABILITY_BIT) {
         if (width == 8)
            *(uint64_t *)(out + 8 * values) = available;
         else
            *(uint32_t *)(out + 4 * values) = available;
      }
   }
   return status;
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

/* The program uploads once, on first submission, into LOCAL through BAR2;
 * the next batch invalidates instruction caches. */
VkResult
apex_program_upload(struct apex_device *device, struct apex_program *program)
{
   if (program->bo.handle)
      return VK_SUCCESS;
   VkResult result = apex_bo_create(device, program->code.size, APEX_VM_READ | APEX_VM_EXEC, 0, 0,
                               &program->bo);
   if (result != VK_SUCCESS)
      return result;
   memcpy(program->bo.map, program->code.data, program->code.size);
   device->programs_uploaded = true;
   return VK_SUCCESS;
}

/* Batches retire at the SIGNAL of their sequence into `retire`. */
static uint64_t
retired_sequence(const struct apex_device *device)
{
   return *(const volatile uint64_t *)device->retire.map;
}

static void
retire_arenas(struct apex_device *device)
{
   uint64_t retired = retired_sequence(device);
   list_for_each_entry_safe(struct apex_arena, arena, &device->busy_arenas, link) {
      if (arena->sequence > retired)
         break;
      list_del(&arena->link);
      list_addtail(&arena->link, &device->free_arenas);
   }
}

/* Waits for the batch of `sequence` to retire; false on loss or timeout. */
static bool
wait_retired(struct apex_device *device, uint64_t sequence, uint64_t timeout_ns)
{
   uint64_t deadline = os_time_get_nano() + timeout_ns;
   while (retired_sequence(device) < sequence) {
      if (vk_device_check_status(&device->vk) != VK_SUCCESS || os_time_get_nano() >= deadline)
         return false;
      nanosleep(&(struct timespec){.tv_nsec = 20000}, NULL);
   }
   return true;
}

static void
destroy_arena(struct apex_device *device, struct apex_arena *arena)
{
   list_del(&arena->link);
   apex_bo_finish(device, &arena->bo);
   free(arena);
}

/* Reuses the smallest retired arena that fits, else creates a power-of-two
 * LOCAL arena; steady-state submission makes no allocation ioctl. */
struct apex_arena *
apex_arena_get(struct apex_device *device, uint64_t bytes)
{
   retire_arenas(device);
   struct apex_arena *best = NULL;
   list_for_each_entry(struct apex_arena, arena, &device->free_arenas, link)
      if (arena->bo.size >= bytes && (!best || arena->bo.size < best->bo.size))
         best = arena;
   if (best) {
      list_del(&best->link);
      return best;
   }
   struct apex_arena *arena = calloc(1, sizeof(*arena));
   if (!arena)
      return NULL;
   if (apex_bo_create(device, MAX2(util_next_power_of_two64(bytes), 65536), APEX_VM_READ, 0, 0,
                 &arena->bo) != VK_SUCCESS) {
      free(arena);
      return NULL;
   }
   return arena;
}

/* One batch on the ring: acquire barrier, a WAIT on kwait for the batch's
 * syncobj waits, one INDIRECT per command buffer, the retirement SIGNAL and,
 * for syncobj signals, a KFENCE with its kernel fences. The doorbell store
 * publishes it; no ioctl runs without syncobj waits or signals. */
static VkResult
submit_batch(struct apex_device *device, struct vk_queue_submit *submit)
{
   VkResult result = vk_device_check_status(&device->vk);
   if (result != VK_SUCCESS)
      return result;
   uint32_t count = submit->command_buffer_count;
   uint64_t *ib_va = calloc(MAX2(count, 1), sizeof(*ib_va));
   uint32_t *ib_dwords = calloc(MAX2(count, 1), sizeof(*ib_dwords));
   struct apex_arena **arenas = calloc(MAX2(count, 1), sizeof(*arenas));
   struct apex_ib ring_words;
   apex_ib_init(&ring_words);
   result = VK_ERROR_OUT_OF_HOST_MEMORY;
   if (!ib_va || !ib_dwords || !arenas)
      goto out;
   for (uint32_t i = 0; i < count; i++) {
      struct apex_command_buffer *cmd = (void *)submit->command_buffers[i];
      result = apex_cmd_prepare(device, cmd, &arenas[i], &ib_va[i]);
      if (result != VK_SUCCESS)
         goto out;
      ib_dwords[i] = cmd->ib.count;
   }
   uint64_t sequence = device->sequence + 1;
   /* Every syncobj wait of the batch shares one kwait value. */
   uint64_t kwait = 0;
   for (uint32_t i = 0; i < submit->wait_count; i++) {
      const struct vk_sync_wait *wait = &submit->waits[i];
      if ((wait->sync->flags & VK_SYNC_IS_TIMELINE) && !wait->wait_value)
         continue;
      struct vk_drm_syncobj *sync = vk_sync_as_drm_syncobj(wait->sync);
      if (!sync) {
         result = VK_ERROR_FEATURE_NOT_PRESENT;
         goto out;
      }
      if (!kwait)
         kwait = device->kwait + 1;
      struct drm_apex_queue_wait args = {
         .queue_id = device->queue_id, .syncobj = sync->syncobj,
         .point = wait->wait_value, .value = kwait,
      };
      if (ioctl(device->fd, DRM_IOCTL_APEX_QUEUE_WAIT, &args)) {
         result = vk_device_set_lost(&device->vk, "Apex queue wait attach failed");
         goto out;
      }
   }
   if (kwait)
      device->kwait = kwait;
   const struct apex_cp_batch batch = {
      .acquire_cache = device->programs_uploaded ?
         APEX_CP_CACHE_INSTRUCTION | APEX_CP_CACHE_L1 | APEX_CP_CACHE_TEXTURE : 0,
      .kwait_va = device->status_va + APEX_STATUS_KWAIT, .kwait_value = kwait,
      .ib_count = count, .ib_va = ib_va, .ib_dwords = ib_dwords,
      .retire_va = device->retire.va, .sequence = sequence,
      .kfence = submit->signal_count != 0,
   };
   apex_cp_batch(&ring_words, &batch);
   if (ring_words.failed) {
      result = VK_ERROR_OUT_OF_HOST_MEMORY;
      goto out;
   }
   if (!apex_ring_reserve(&device->ring, ring_words.count, 5000000000ull)) {
      result = vk_device_set_lost(&device->vk, "Apex ring did not drain");
      goto out;
   }
   apex_ring_write(&device->ring, ring_words.words, ring_words.count);
   apex_ring_publish(&device->ring);
   device->sequence = sequence;
   device->programs_uploaded = false;
   for (uint32_t i = 0; i < count; i++)
      arenas[i]->sequence = sequence;
   result = VK_SUCCESS;
   for (uint32_t i = 0; i < submit->signal_count; i++) {
      const struct vk_sync_signal *signal = &submit->signals[i];
      struct vk_drm_syncobj *sync = vk_sync_as_drm_syncobj(signal->sync);
      struct drm_apex_queue_fence args = {
         .queue_id = device->queue_id, .syncobj = sync ? sync->syncobj : 0,
         .seqno = sequence, .point = signal->signal_value,
      };
      if (!sync || ioctl(device->fd, DRM_IOCTL_APEX_QUEUE_FENCE, &args)) {
         result = vk_device_set_lost(&device->vk, "Apex queue fence attach failed");
         break;
      }
   }
out:
   /* Arenas of an unpublished batch return to the free list. */
   for (uint32_t i = 0; arenas && i < count; i++) {
      if (arenas[i] && arenas[i]->sequence == UINT64_MAX) {
         list_del(&arenas[i]->link);
         list_addtail(&arenas[i]->link, &device->free_arenas);
      }
   }
   apex_ib_finish(&ring_words);
   free(arenas);
   free(ib_dwords);
   free(ib_va);
   return result;
}

static VkResult
submit_queue(struct vk_queue *queue, struct vk_queue_submit *submit)
{
   struct apex_device *device = (struct apex_device *)queue->base.device;
   if (vk_queue_submit_has_bind(submit) || submit->is_protected || !device->ring.map)
      return vk_queue_set_lost(queue, "unsupported Apex submission");
   for (unsigned i = 0; i < submit->command_buffer_count; i++) {
      struct apex_command_buffer *cmd = (void *)submit->command_buffers[i];
      list_for_each_entry(struct apex_upload, upload, &cmd->uploads, link) {
         if (upload->bo.handle)
            continue;
         uint64_t va = upload->reserved_va;
         upload->reserved_va = 0;
         if (apex_bo_create(device, upload->size, APEX_VM_READ, 0, va, &upload->bo) != VK_SUCCESS)
            return vk_queue_set_lost(queue, "Apex update allocation failed");
         memcpy(upload->bo.map, upload->data, upload->size);
      }
   }
   if (submit_batch(device, submit) != VK_SUCCESS)
      return vk_queue_set_lost(queue, "Apex submission failed");
   return VK_SUCCESS;
}

static VKAPI_ATTR VkResult VKAPI_CALL
apex_QueueWaitIdle(VkQueue handle)
{
   VK_FROM_HANDLE(vk_queue, queue, handle);
   struct apex_device *device = (void *)queue->base.device;
   if (device->vk.physical->supported_sync_types)
      return vk_common_QueueWaitIdle(handle);
   if (device->ring.map && !wait_retired(device, device->sequence, UINT64_MAX))
      return VK_ERROR_DEVICE_LOST;
   return vk_device_is_lost(&device->vk) ? VK_ERROR_DEVICE_LOST : VK_SUCCESS;
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

#define APEX_RING_BYTES (64 * 1024)

/* The ring, the private arena and the bin pool live in LOCAL, the
 * retirement word in SYSTEM; the kernel supplies the status page and
 * doorbell mappings. */
static VkResult
create_queue(struct apex_device *device)
{
   VkResult result = apex_bo_create(device, APEX_RING_BYTES, APEX_VM_READ, 0, 0, &device->ring_bo);
   if (result == VK_SUCCESS)
      result = host_words_create(device, 4096, &device->retire);
   if (result == VK_SUCCESS) {
      mtx_lock(&device->va_mutex);
      uint64_t va = util_vma_heap_alloc(&device->va_heap, APEX_PRIVATE_BYTES, APEX_PRIVATE_BYTES);
      mtx_unlock(&device->va_mutex);
      result = va ? apex_bo_create(device, APEX_PRIVATE_BYTES, APEX_VM_READ | APEX_VM_WRITE, 0, va,
                              &device->private_arena) : VK_ERROR_OUT_OF_DEVICE_MEMORY;
   }
   if (result == VK_SUCCESS)
      result = apex_bo_create(device, APEX_BIN_POOL_BYTES, APEX_VM_READ | APEX_VM_WRITE, 0, 0,
                              &device->bin_pool);
   if (result != VK_SUCCESS)
      return result;
   memset(device->retire.map, 0, 8);
   struct drm_apex_queue_create create = {
      .ring_va = device->ring_bo.va, .ring_bytes = APEX_RING_BYTES,
   };
   if (ioctl(device->fd, DRM_IOCTL_APEX_QUEUE_CREATE, &create))
      return errno == ENODEV ? VK_ERROR_INCOMPATIBLE_DRIVER : VK_ERROR_INITIALIZATION_FAILED;
   device->queue_id = create.queue_id;
   device->status_va = create.status_va;
   void *status = mmap(NULL, 4096, PROT_READ, MAP_SHARED, device->fd, create.status_offset);
   void *doorbell = mmap(NULL, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, device->fd,
                         create.doorbell_offset);
   device->status_map = status == MAP_FAILED ? NULL : status;
   device->doorbell_map = doorbell == MAP_FAILED ? NULL : doorbell;
   if (!device->status_map || !device->doorbell_map)
      return VK_ERROR_INITIALIZATION_FAILED;
   device->ring = (struct apex_ring) {
      .map = device->ring_bo.map, .dwords = APEX_RING_BYTES / 4,
      .status = device->status_map, .doorbell = device->doorbell_map,
   };
   return VK_SUCCESS;
}

static void
destroy_queue(struct apex_device *device)
{
   if (device->status_va) {
      struct drm_apex_queue_destroy destroy = {.queue_id = device->queue_id};
      ioctl(device->fd, DRM_IOCTL_APEX_QUEUE_DESTROY, &destroy);
   }
   if (device->status_map)
      munmap(device->status_map, 4096);
   if (device->doorbell_map)
      munmap(device->doorbell_map, 4096);
   device->status_map = device->doorbell_map = NULL;
   device->status_va = 0;
   device->ring = (struct apex_ring){0};
   apex_bo_finish(device, &device->bin_pool);
   apex_bo_finish(device, &device->private_arena);
   apex_bo_finish(device, &device->retire);
   apex_bo_finish(device, &device->ring_bo);
}

VkResult
apex_device_init(struct apex_device *device, struct vk_physical_device *physical,
                  const VkDeviceCreateInfo *info, const VkAllocationCallbacks *alloc, int fd)
{
   if (info->queueCreateInfoCount != 1)
      return vk_errorf(physical, VK_ERROR_FEATURE_NOT_PRESENT,
                       "Apex requires one queue create info, received %u", info->queueCreateInfoCount);
   if (info->pQueueCreateInfos[0].queueFamilyIndex ||
       info->pQueueCreateInfos[0].queueCount != 1 || info->pQueueCreateInfos[0].flags)
      return vk_errorf(physical, VK_ERROR_FEATURE_NOT_PRESENT,
                       "Apex requires one unflagged queue from family 0");
   struct drm_apex_info caps = {0};
   if (fd >= 0 && ioctl(fd, DRM_IOCTL_APEX_INFO, &caps))
      return VK_ERROR_INCOMPATIBLE_DRIVER;
   struct vk_device_entrypoint_table entrypoints = {
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
      .GetImageSubresourceLayout2 = apex_GetImageSubresourceLayout2,
      .GetDeviceImageSubresourceLayout = apex_GetDeviceImageSubresourceLayout,
      .CreateImageView = apex_CreateImageView, .DestroyImageView = apex_DestroyImageView,
      .CreateSampler = apex_CreateSampler, .DestroySampler = apex_DestroySampler,
      .CreateQueryPool = apex_CreateQueryPool, .DestroyQueryPool = apex_DestroyQueryPool,
      .ResetQueryPool = apex_ResetQueryPool, .GetQueryPoolResults = apex_GetQueryPoolResults,
      .CreateEvent = apex_CreateEvent, .DestroyEvent = apex_DestroyEvent,
      .GetEventStatus = apex_GetEventStatus, .SetEvent = apex_SetEvent, .ResetEvent = apex_ResetEvent,
      .CreateBufferView = apex_CreateBufferView,
      .GetDeviceMemoryCommitment = apex_GetDeviceMemoryCommitment,
      .GetImageSparseMemoryRequirements2 = apex_GetImageSparseMemoryRequirements2,
      .GetDeviceImageSparseMemoryRequirements = apex_GetDeviceImageSparseMemoryRequirements,
      .CreateDescriptorSetLayout = apex_CreateDescriptorSetLayout,
      .CreateDescriptorPool = apex_CreateDescriptorPool, .DestroyDescriptorPool = apex_DestroyDescriptorPool,
      .ResetDescriptorPool = apex_ResetDescriptorPool, .AllocateDescriptorSets = apex_AllocateDescriptorSets,
      .FreeDescriptorSets = apex_FreeDescriptorSets, .UpdateDescriptorSets = apex_UpdateDescriptorSets,
      .UpdateDescriptorSetWithTemplate = apex_UpdateDescriptorSetWithTemplate,
      .GetDescriptorSetLayoutSupport = apex_GetDescriptorSetLayoutSupport,
      .GetBufferOpaqueCaptureAddress = apex_GetBufferOpaqueCaptureAddress,
      .GetDeviceMemoryOpaqueCaptureAddress = apex_GetDeviceMemoryOpaqueCaptureAddress,
      .QueueWaitIdle = apex_QueueWaitIdle,
      .GetFenceStatus = apex_GetFenceStatus,
      .GetSemaphoreCounterValue = apex_GetSemaphoreCounterValue,
   };
   apex_cmd_entrypoints(&entrypoints);
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
   device->vk.command_buffer_ops = &apex_command_buffer_ops;
   device->fd = fd;
   device->queue_id = 0;
   device->status_va = 0;
   device->status_map = device->doorbell_map = NULL;
   device->ring_bo = device->retire = device->private_arena = (struct apex_bo){0};
   device->ring = (struct apex_ring){0};
   device->sequence = device->kwait = 0;
   device->programs_uploaded = false;
   memset(device->internal, 0, sizeof(device->internal));
   device->empty_fragment = NULL;
   device->bin_pool = (struct apex_bo){0};
   list_inithead(&device->busy_arenas);
   list_inithead(&device->free_arenas);
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
   if (fd >= 0)
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
   apex_cmd_meta_init(&device->meta);
   /* Meta draws rectangles with its own vertex shader writing gl_Layer. */
   device->meta.use_rect_list_pipeline = true;
   device->meta.use_gs_for_layer = false;
   device->queue.driver_submit = submit_queue;
   if (fd >= 0) {
      result = create_queue(device);
      if (result != VK_SUCCESS) {
         apex_device_finish(device);
         return result;
      }
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
   /* A lost queue keeps its arenas mapped and reserved until file close. */
   if (device->ring.map && wait_retired(device, device->sequence, 5000000000ull))
      retire_arenas(device);
   list_for_each_entry_safe(struct apex_arena, arena, &device->free_arenas, link)
      destroy_arena(device, arena);
   list_for_each_entry_safe(struct apex_arena, arena, &device->busy_arenas, link)
      destroy_arena(device, arena);
   destroy_queue(device);
   apex_graphics_finish(device);

   vk_meta_device_finish(&device->vk, &device->meta);
   util_vma_heap_finish(&device->va_heap);
   mtx_destroy(&device->memory_mutex);
   mtx_destroy(&device->va_mutex);
   vk_device_finish(&device->vk);
}
