/* SPDX-License-Identifier: MIT */
#ifndef APEX_DEVICE_H
#define APEX_DEVICE_H
#include "vk_device.h"
#include "vk_queue.h"

enum apex_transport { APEX_TRANSPORT_NATIVE, APEX_TRANSPORT_DRM };

/* Internal single-queue device. The caller owns fd through device teardown.
 * DRM callers supply physical->supported_sync_types from vk_drm_syncobj_get_type.
 * The native qualification path may omit sync types and submit synchronously.
 */
struct apex_device {
   struct vk_device vk;
   struct vk_queue queue;
   int fd;
   enum apex_transport transport;
};
VK_DEFINE_HANDLE_CASTS(apex_device, vk.base, VkDevice, VK_OBJECT_TYPE_DEVICE);

VkResult apex_device_init(struct apex_device *device,
                          struct vk_physical_device *physical,
                          const VkDeviceCreateInfo *info,
                          const VkAllocationCallbacks *alloc, int fd,
                          enum apex_transport transport);
void apex_device_finish(struct apex_device *device);
#endif
