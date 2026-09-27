/* SPDX-License-Identifier: MIT */
#ifndef APEX_DEVICE_H
#define APEX_DEVICE_H
#include "vk_device.h"
#include "vk_queue.h"
#include "vk_descriptor_set_layout.h"
#include "vk_meta.h"
#include "util/vma.h"

enum apex_transport { APEX_TRANSPORT_NATIVE, APEX_TRANSPORT_DRM };

/* Host table limits; descriptors are ordinary LOCAL memory. */
#define APEX_MAX_DESCRIPTORS 4096
#define APEX_MAX_BINDINGS 1024
#define APEX_MAX_PUSH_CONSTANTS 256
#define APEX_EXTERNAL_MEMORY_TYPES (VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT | \
                                    VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT)
struct apex_binding_layout {
   uint32_t offset, count, flags;
   VkDescriptorType type;
   VkShaderStageFlags stages;
};
struct apex_set_layout {
   struct vk_descriptor_set_layout vk;
   uint32_t binding_count, descriptor_count;
   uint32_t counts[VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT + 1];
   struct apex_binding_layout bindings[];
};
struct apex_buffer_descriptor {
   uint32_t low, high, bytes, reserved;
};
struct apex_image_descriptor {
   uint32_t low, high, width, height, depth, row_stride, slice_stride, reserved;
};
union apex_descriptor {
   struct apex_buffer_descriptor buffer;
   struct apex_image_descriptor image;
};

/* Immutable trailer after descriptors and push constants, in little endian. */
struct apex_dispatch_parameters {
   uint32_t base[3];
   uint32_t groups[3];
};

struct apex_bo {
   void *map;
   uint64_t va, size;
   uint32_t handle;
};

/* Internal single-queue device. The caller owns fd through device teardown.
 * DRM callers supply physical->supported_sync_types from vk_drm_syncobj_get_type.
 * The native qualification path may omit sync types and submit synchronously.
 */
struct apex_device {
   struct vk_device vk;
   struct vk_device_dispatch_table cmd_dispatch;
   struct vk_queue queue;
   struct vk_meta_device meta;
   int fd;
   enum apex_transport transport;
   bool prime_coherent;
   bool host_coherent;
   struct util_vma_heap va_heap;
   mtx_t va_mutex;
   /* Serializes PRIME handle lookup and final backing destruction. */
   mtx_t memory_mutex;
   struct list_head memories;
   /* Submit-thread-owned descriptor retirement and private completion timeline. */
   uint32_t completion;
   uint64_t point;
   struct list_head retired;
   /* Internal triangle setup program, compiled on first draw. */
   struct apex_program *setup;
};
VK_DEFINE_HANDLE_CASTS(apex_device, vk.base, VkDevice, VK_OBJECT_TYPE_DEVICE);

void apex_bo_finish(struct apex_device *device, struct apex_bo *bo);

VkFormatFeatureFlags2 apex_format_features(VkFormat format, bool buffer);
VkResult apex_image_format_properties(const VkPhysicalDeviceImageFormatInfo2 *info,
                                      VkImageFormatProperties2 *properties);

VkResult apex_device_init(struct apex_device *device,
                          struct vk_physical_device *physical,
                          const VkDeviceCreateInfo *info,
                          const VkAllocationCallbacks *alloc, int fd,
                          enum apex_transport transport);
void apex_device_finish(struct apex_device *device);
#endif
