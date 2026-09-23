/* SPDX-License-Identifier: MIT */
#include "apex_device.h"
#include "apex_native_uapi.h"
#include "apex_pipeline.h"
#include "drm-uapi/apex_drm.h"
#include "vk_alloc.h"
#include "vk_buffer.h"
#include "vk_command_buffer.h"
#include "vk_command_pool.h"
#include "vk_common_entrypoints.h"
#include "vk_descriptor_set_layout.h"
#include "vk_device_memory.h"
#include "vk_drm_syncobj.h"
#include "vk_log.h"
#include "vk_physical_device.h"
#include "util/os_time.h"
#include <errno.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <time.h>

struct apex_memory {
   struct vk_device_memory vk;
   void *data;
   struct apex_bo bo;
};
struct apex_buffer {
   struct vk_buffer vk;
   struct apex_memory *memory;
   VkDeviceSize offset;
};
struct apex_descriptor_pool {
   struct vk_object_base base;
   struct list_head sets;
   uint32_t capacity, allocated;
   uint64_t descriptors[4], used[4]; /* VkDescriptorType minus UNIFORM_BUFFER */
};
struct apex_descriptor_set {
   struct vk_object_base base;
   struct list_head link;
   struct apex_descriptor_pool *pool;
   struct apex_set_layout *layout;
   VkDescriptorBufferInfo buffers[];
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
};
struct apex_command_buffer {
   struct vk_command_buffer vk;
   struct list_head dispatches;
   struct apex_pipeline *pipeline;
   struct apex_bound_set *sets[MESA_VK_MAX_DESCRIPTOR_SETS];
};
VK_DEFINE_NONDISP_HANDLE_CASTS(apex_memory, vk.base, VkDeviceMemory, VK_OBJECT_TYPE_DEVICE_MEMORY);
VK_DEFINE_NONDISP_HANDLE_CASTS(apex_buffer, vk.base, VkBuffer, VK_OBJECT_TYPE_BUFFER);
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
   if (bo->va) {
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

static VkResult
bo_create(struct apex_device *device, uint64_t size, uint32_t flags, struct apex_bo *bo)
{
   bo->size = align64(size, 4096);
   VkResult result = gem_create(device, bo->size, &bo->handle, &bo->map);
   if (result != VK_SUCCESS)
      return result;
   mtx_lock(&device->va_mutex);
   uint64_t va = util_vma_heap_alloc(&device->va_heap, bo->size, 4096);
   mtx_unlock(&device->va_mutex);
   if (va) {
      struct drm_apex_vm_bind bind = {
         .operation = APEX_DRM_VM_BIND_MAP, .flags = flags, .handle = bo->handle,
         .va = va, .bytes = bo->size,
      };
      if (!ioctl(device->fd, DRM_IOCTL_APEX_VM_BIND, &bind)) {
         bo->va = va;
         return VK_SUCCESS;
      }
      mtx_lock(&device->va_mutex);
      util_vma_heap_free(&device->va_heap, va, bo->size);
      mtx_unlock(&device->va_mutex);
   }
   apex_bo_finish(device, bo);
   return VK_ERROR_OUT_OF_DEVICE_MEMORY;
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

static VKAPI_ATTR VkResult VKAPI_CALL
apex_AllocateMemory(VkDevice dev, const VkMemoryAllocateInfo *info,
                    const VkAllocationCallbacks *alloc, VkDeviceMemory *out)
{
   VK_FROM_HANDLE(apex_device, device, dev);
   *out = VK_NULL_HANDLE;
   if (info->memoryTypeIndex || info->pNext)
      return VK_ERROR_FEATURE_NOT_PRESENT;
   if (!info->allocationSize || info->allocationSize > 64 * 1024 * 1024)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;
   struct apex_memory *mem = vk_device_memory_create(&device->vk, info, alloc, sizeof(*mem));
   if (!mem)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   VkResult result;
   if (device->transport == APEX_TRANSPORT_DRM) {
      result = bo_create(device, info->allocationSize,
                         APEX_DRM_VM_READ | APEX_DRM_VM_WRITE, &mem->bo);
      mem->data = mem->bo.map;
   } else {
      mem->data = vk_zalloc2(&device->vk.alloc, alloc, info->allocationSize, 8,
                             VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
      result = mem->data ? VK_SUCCESS : VK_ERROR_OUT_OF_HOST_MEMORY;
   }
   if (result != VK_SUCCESS) {
      vk_device_memory_destroy(&device->vk, alloc, &mem->vk);
      return result;
   }
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
      apex_bo_finish(device, &memory->bo);
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
   if (info->flags || info->offset >= memory->vk.size ||
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
      if (ranges[i].offset >= memory->vk.size)
         return VK_ERROR_MEMORY_MAP_FAILED;
      uint64_t bytes = ranges[i].size == VK_WHOLE_SIZE ?
         memory->vk.size - ranges[i].offset : ranges[i].size;
      if (!bytes || bytes > memory->vk.size - ranges[i].offset)
         return VK_ERROR_MEMORY_MAP_FAILED;
      if (device->transport == APEX_TRANSPORT_DRM) {
         VkResult result = bo_transfer(device, &memory->bo, direction, ranges[i].offset, bytes);
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
   if (info->flags || info->sharingMode != VK_SHARING_MODE_EXCLUSIVE ||
       (vk_buffer_usage_flags(info) & ~(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT)))
      return VK_ERROR_FEATURE_NOT_PRESENT;
   struct apex_buffer *buffer = vk_buffer_create(&device->vk, info, alloc, sizeof(*buffer));
   if (!buffer)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
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
   out->memoryRequirements = (VkMemoryRequirements) {
      .size = align64(info->pCreateInfo->size, 64), .alignment = 64, .memoryTypeBits = 1,
   };
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
   }
   return VK_SUCCESS;
}

static VKAPI_ATTR VkResult VKAPI_CALL
apex_CreateDescriptorSetLayout(VkDevice dev, const VkDescriptorSetLayoutCreateInfo *info,
                               const VkAllocationCallbacks *alloc, VkDescriptorSetLayout *out)
{
   VK_FROM_HANDLE(apex_device, device, dev);
   *out = VK_NULL_HANDLE;
   const VkDescriptorSetLayoutBindingFlagsCreateInfo *binding_flags =
      vk_find_struct_const(info->pNext, DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO);
   if (info->flags & ~VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT)
      return VK_ERROR_FEATURE_NOT_PRESENT;
   unsigned count = 0, descriptors = 0;
   bool dynamic = false, update_after_bind = false;
   for (unsigned i = 0; i < info->bindingCount; i++) {
      const VkDescriptorSetLayoutBinding *b = &info->pBindings[i];
      VkDescriptorBindingFlags flags = binding_flags && binding_flags->bindingCount ?
         binding_flags->pBindingFlags[i] : 0;
      if (b->binding >= APEX_MAX_BINDINGS || b->descriptorCount > APEX_MAX_DESCRIPTORS - descriptors ||
          (flags & ~VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT) ||
          b->descriptorType < VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER ||
          b->descriptorType > VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC ||
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
      layout->counts[layout->bindings[b].type - VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER] +=
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
   uint64_t descriptors[4] = {0};
   for (unsigned i = 0; i < info->poolSizeCount; i++) {
      VkDescriptorType type = info->pPoolSizes[i].type;
      if (type < VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER || type > VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC)
         return VK_ERROR_FEATURE_NOT_PRESENT;
      descriptors[type - VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER] += info->pPoolSizes[i].descriptorCount;
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
   uint64_t descriptors[4] = {0};
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
         sizeof(*set) + layout->descriptor_count * sizeof(set->buffers[0]), VK_OBJECT_TYPE_DESCRIPTOR_SET);
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
      memcpy(&set->buffers[start], writes[i].pBufferInfo,
             writes[i].descriptorCount * sizeof(set->buffers[0]));
   }
   for (unsigned i = 0; i < copy_count; i++) {
      VK_FROM_HANDLE(apex_descriptor_set, src, copies[i].srcSet);
      VK_FROM_HANDLE(apex_descriptor_set, dst, copies[i].dstSet);
      unsigned from = src->layout->bindings[copies[i].srcBinding].offset + copies[i].srcArrayElement;
      unsigned to = dst->layout->bindings[copies[i].dstBinding].offset + copies[i].dstArrayElement;
      assert(from + copies[i].descriptorCount <= src->layout->descriptor_count &&
             to + copies[i].descriptorCount <= dst->layout->descriptor_count);
      memmove(&dst->buffers[to], &src->buffers[from], copies[i].descriptorCount * sizeof(src->buffers[0]));
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
   if (level != VK_COMMAND_BUFFER_LEVEL_PRIMARY)
      return VK_ERROR_FEATURE_NOT_PRESENT;
   struct apex_command_buffer *cmd = vk_zalloc(&pool->alloc, sizeof(*cmd), 8,
                                              VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!cmd)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   VkResult result = vk_command_buffer_init(pool, &cmd->vk, &command_ops, level);
   if (result != VK_SUCCESS) {
      vk_free(&pool->alloc, cmd);
      return result;
   }
   list_inithead(&cmd->dispatches);
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
apex_CmdBindDescriptorSets(VkCommandBuffer handle, VkPipelineBindPoint point,
                           VkPipelineLayout layout, uint32_t first, uint32_t count,
                           const VkDescriptorSet *sets, uint32_t dynamic_count, const uint32_t *offsets)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   if (point != VK_PIPELINE_BIND_POINT_COMPUTE || first > MESA_VK_MAX_DESCRIPTOR_SETS ||
       count > MESA_VK_MAX_DESCRIPTOR_SETS - first) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   unsigned required = 0;
   for (unsigned s = 0; s < count; s++)
      required += apex_descriptor_set_from_handle(sets[s])->layout->vk.dynamic_descriptor_count;
   if (required != dynamic_count) {
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
apex_CmdDispatch(VkCommandBuffer handle, uint32_t x, uint32_t y, uint32_t z)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   if (!x || !y || !z)
      return;
   if (x > 1024 || y != 1 || z != 1 || !cmd->pipeline ||
       (!cmd->pipeline->layout && !cmd->sets[0])) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
      return;
   }
   struct apex_dispatch *dispatch = vk_alloc(&cmd->vk.pool->alloc, sizeof(*dispatch), 8,
                                             VK_SYSTEM_ALLOCATION_SCOPE_OBJECT);
   if (!dispatch) {
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_OUT_OF_HOST_MEMORY);
      return;
   }
   *dispatch = (struct apex_dispatch) {.pipeline = cmd->pipeline, .groups = x};
   memcpy(dispatch->sets, cmd->sets, sizeof(dispatch->sets));
   for (unsigned s = 0; s < ARRAY_SIZE(dispatch->sets); s++)
      if (dispatch->sets[s])
         dispatch->sets[s]->refs++;
   list_addtail(&dispatch->link, &cmd->dispatches);
}

static VKAPI_ATTR void VKAPI_CALL
apex_CmdPipelineBarrier2(VkCommandBuffer handle, const VkDependencyInfo *info)
{
   VK_FROM_HANDLE(apex_command_buffer, cmd, handle);
   if (info->imageMemoryBarrierCount)
      vk_command_buffer_set_error(&cmd->vk, VK_ERROR_FEATURE_NOT_PRESENT);
   /* Every native dispatch below completes allocation release before the next
    * dispatch acquires its input. Buffer barriers need no additional operation. */
}

static int
native_command(struct apex_device *device, uint32_t operation)
{
   struct apex_ioctl_native r = {.operation = operation};
   return ioctl(device->fd, APEX_IOCTL_NATIVE, &r);
}

static VkResult
drm_dispatch(struct apex_device *device, const struct apex_dispatch *dispatch)
{
   struct apex_pipeline *pipeline = dispatch->pipeline;
   if (!pipeline->layout)
      return VK_ERROR_FEATURE_NOT_PRESENT;
   size_t bytes = (pipeline->descriptor_count + 1) * sizeof(struct apex_buffer_descriptor);
   struct apex_buffer_descriptor *rows = calloc(1, bytes);
   if (!rows)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   VkResult result = VK_ERROR_DEVICE_LOST;
   struct apex_bo table = {0};
   for (unsigned s = 0; s < pipeline->layout->set_count; s++) {
      const struct apex_set_layout *layout = (const void *)pipeline->layout->set_layouts[s];
      if (!layout || !layout->descriptor_count)
         continue;
      const struct apex_bound_set *bound = dispatch->sets[s];
      const struct apex_descriptor_set *set = bound ? bound->set : NULL;
      for (unsigned d = 0; d < layout->descriptor_count; d++) {
         if (!BITSET_TEST(pipeline->used_descriptors, pipeline->set_offsets[s] + d))
            continue;
         if (!set || memcmp(set->layout->vk.blake3, layout->vk.blake3, BLAKE3_OUT_LEN))
            goto out;
         const VkDescriptorBufferInfo *binding = &set->buffers[d];
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
         uint64_t va = buffer->memory->bo.va + buffer->offset + binding->offset + dynamic;
         rows[pipeline->set_offsets[s] + d] = (struct apex_buffer_descriptor) {
            util_cpu_to_le32(va), util_cpu_to_le32(va >> 32), util_cpu_to_le32(range), 0,
         };
      }
   }
   if (!pipeline->program.handle) {
      result = bo_create(device, pipeline->code.size,
                         APEX_DRM_VM_READ | APEX_DRM_VM_EXEC, &pipeline->program);
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
   result = bo_create(device, bytes, APEX_DRM_VM_READ | APEX_DRM_VM_WRITE, &table);
   if (result != VK_SUCCESS)
      goto out;
   memcpy(table.map, rows, bytes);
   result = bo_transfer(device, &table, APEX_DRM_TRANSFER_TO_LOCAL, 0, bytes);
   if (result != VK_SUCCESS)
      goto out;
   struct drm_apex_vm_exec args = {
      .program_va = pipeline->program.va, .program_bytes = pipeline->code.size,
      .data_va = table.va,
      .workgroups = dispatch->groups,
   };
   /* No implicit data transfer. Host visibility requires flush/invalidate.
    * Do not retry EXEC on EINTR: the kernel cancels/drains the accepted job. */
   result = !ioctl(device->fd, DRM_IOCTL_APEX_VM_EXEC, &args) && args.status == 1 ?
      VK_SUCCESS : VK_ERROR_DEVICE_LOST;
out:
   apex_bo_finish(device, &table);
   free(rows);
   return result;
}

static VkResult
dispatch_compute(struct apex_device *device, const struct apex_dispatch *dispatch)
{
   if (device->transport == APEX_TRANSPORT_DRM)
      return drm_dispatch(device, dispatch);
   const VkDescriptorBufferInfo *binding = &dispatch->sets[0]->set->buffers[0];
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

static VkResult
submit_queue(struct vk_queue *queue, struct vk_queue_submit *submit)
{
   struct apex_device *device = (struct apex_device *)queue->base.device;
   if (vk_queue_submit_has_bind(submit) || submit->is_protected)
      return vk_queue_set_lost(queue, "unsupported Apex submission");
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

VkResult
apex_device_init(struct apex_device *device, struct vk_physical_device *physical,
                  const VkDeviceCreateInfo *info, const VkAllocationCallbacks *alloc, int fd,
                  enum apex_transport transport)
{
   if (info->queueCreateInfoCount != 1 || info->pQueueCreateInfos[0].queueFamilyIndex ||
       info->pQueueCreateInfos[0].queueCount != 1 || info->pQueueCreateInfos[0].flags)
      return VK_ERROR_FEATURE_NOT_PRESENT;
   if (transport == APEX_TRANSPORT_DRM) {
      struct drm_apex_info caps = {0};
      if (ioctl(fd, DRM_IOCTL_APEX_INFO, &caps) || caps.version != 2 ||
          (caps.capabilities & (APEX_DRM_CAP_SHMEM | APEX_DRM_CAP_GPUVM)) !=
          (APEX_DRM_CAP_SHMEM | APEX_DRM_CAP_GPUVM))
         return VK_ERROR_INCOMPATIBLE_DRIVER;
   }
   const struct vk_device_dispatch_table dispatch = {
      .CreateComputePipelines = apex_CreateComputePipelines,
      .AllocateMemory = apex_AllocateMemory, .FreeMemory = apex_FreeMemory,
      .MapMemory2 = apex_MapMemory2, .UnmapMemory2 = apex_UnmapMemory2,
      .FlushMappedMemoryRanges = apex_FlushMappedMemoryRanges,
      .InvalidateMappedMemoryRanges = apex_InvalidateMappedMemoryRanges,
      .CreateBuffer = apex_CreateBuffer, .DestroyBuffer = apex_DestroyBuffer,
      .GetDeviceBufferMemoryRequirements = apex_GetDeviceBufferMemoryRequirements,
      .BindBufferMemory2 = apex_BindBufferMemory2,
      .CreateDescriptorSetLayout = apex_CreateDescriptorSetLayout,
      .CreateDescriptorPool = apex_CreateDescriptorPool, .DestroyDescriptorPool = apex_DestroyDescriptorPool,
      .ResetDescriptorPool = apex_ResetDescriptorPool, .AllocateDescriptorSets = apex_AllocateDescriptorSets,
      .FreeDescriptorSets = apex_FreeDescriptorSets, .UpdateDescriptorSets = apex_UpdateDescriptorSets,
      .BeginCommandBuffer = apex_BeginCommandBuffer, .EndCommandBuffer = apex_EndCommandBuffer,
      .CmdBindPipeline = apex_CmdBindPipeline, .CmdBindDescriptorSets = apex_CmdBindDescriptorSets,
      .CmdDispatch = apex_CmdDispatch, .CmdPipelineBarrier2 = apex_CmdPipelineBarrier2,
      .QueueWaitIdle = apex_QueueWaitIdle,
   };
   VkResult result = vk_device_init(&device->vk, physical, &dispatch, info, alloc);
   if (result != VK_SUCCESS)
      return result;
   device->vk.command_buffer_ops = &command_ops;
   device->fd = fd;
   device->transport = transport;
   if (mtx_init(&device->va_mutex, mtx_plain) != thrd_success) {
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
      mtx_destroy(&device->va_mutex);
      vk_device_finish(&device->vk);
      return result;
   }
   device->queue.driver_submit = submit_queue;
   if (physical->supported_sync_types) {
      result = vk_queue_enable_submit_thread(&device->queue);
      if (result != VK_SUCCESS) {
         vk_queue_finish(&device->queue);
         util_vma_heap_finish(&device->va_heap);
         mtx_destroy(&device->va_mutex);
         vk_device_finish(&device->vk);
         return result;
      }
   }
   return VK_SUCCESS;
}

void
apex_device_finish(struct apex_device *device)
{
   vk_queue_finish(&device->queue);
   util_vma_heap_finish(&device->va_heap);
   mtx_destroy(&device->va_mutex);
   vk_device_finish(&device->vk);
}
