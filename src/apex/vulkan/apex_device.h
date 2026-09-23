/* SPDX-License-Identifier: MIT */
#ifndef APEX_DEVICE_H
#define APEX_DEVICE_H
#include "vk_device.h"
#include "vk_queue.h"

/* Internal single-queue qualification device. The caller owns native_fd. */
struct apex_device {
   struct vk_device vk;
   struct vk_queue queue;
   int native_fd;
};
VK_DEFINE_HANDLE_CASTS(apex_device, vk.base, VkDevice, VK_OBJECT_TYPE_DEVICE);

VkResult apex_device_init(struct apex_device *device,
                          struct vk_physical_device *physical,
                          const VkDeviceCreateInfo *info,
                          const VkAllocationCallbacks *alloc, int native_fd);
void apex_device_finish(struct apex_device *device);
#endif
