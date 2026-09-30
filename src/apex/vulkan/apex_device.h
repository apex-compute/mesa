/* SPDX-License-Identifier: MIT */
#ifndef APEX_DEVICE_H
#define APEX_DEVICE_H
#include "vk_device.h"
#include "vk_queue.h"
#include "vk_descriptor_set_layout.h"
#include "vk_meta.h"
#include "vk_sampler.h"
#include "apex_format.h"
#include "apex_cp.h"
#include "util/vma.h"

enum apex_internal {
   APEX_INTERNAL_SETUP, APEX_INTERNAL_BIN, APEX_INTERNAL_COPY, APEX_INTERNAL_CLEAR, APEX_INTERNAL_TIMESTAMP, APEX_INTERNAL_QUERY_COPY,
   APEX_INTERNAL_RESOLVE, APEX_INTERNAL_ETC2, APEX_INTERNAL_XFB,
   APEX_INTERNAL_COUNT
};

/* Host table limits; descriptors are ordinary LOCAL memory. */
#define APEX_MAX_DESCRIPTORS 4096
/* Largest allocation: the per-client page tables map about 1 GiB. */
#define APEX_MAX_ALLOCATION (1024ull * 1024 * 1024)
#define APEX_MAX_BINDINGS 1024
#define APEX_MAX_INLINE_BYTES 4096
/* Pool accounting buckets: core types, then inline uniform block bytes. */
#define APEX_DESCRIPTOR_BUCKETS (VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT + 2)
static inline unsigned
apex_descriptor_bucket(VkDescriptorType type)
{
   return type == VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK ? VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT + 1 :
          (unsigned)type <= VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT ? type : APEX_DESCRIPTOR_BUCKETS;
}
#define APEX_MAX_PUSH_CONSTANTS 256
#define APEX_EXTERNAL_MEMORY_TYPES (VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT | \
                                    VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT)
/* `offset` indexes set storage by array element; `slot` indexes table rows,
 * apex_descriptor_slots() per element. Immutable sampler rows start at
 * `immutable` in the layout's sampler array (~0 when absent). An inline
 * uniform block is one element of `bytes` bytes: set storage holds the bytes
 * and the table holds a buffer row followed by the data rows. */
struct apex_binding_layout {
   uint32_t offset, slot, count, flags, immutable, bytes;
   VkDescriptorType type;
   VkShaderStageFlags stages;
};
struct apex_set_layout {
   struct vk_descriptor_set_layout vk;
   uint32_t binding_count, descriptor_count, slot_count;
   uint32_t counts[APEX_DESCRIPTOR_BUCKETS];
   uint32_t (*samplers)[8];
   struct apex_binding_layout bindings[];
};
struct apex_sampler {
   struct vk_sampler vk;
   uint32_t row[8];
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
/* Compute jobs: base = first and end linear workgroup and the launch stride,
 * groups = API grid, indirect = VkDispatchIndirectCommand address supplying
 * the grid (zero for direct jobs), origin = vkCmdDispatchBase offset. Graphics
 * jobs use base[0] and groups[0..1] as their launch range. predicate =
 * conditional rendering's 32-bit value address (zero when unconditional):
 * the job does nothing when the value is zero, or nonzero if inverted. */
struct apex_dispatch_parameters {
   uint32_t base[3];
   uint32_t groups[3];
   uint32_t indirect[2];
   uint32_t origin[3];
   uint32_t predicate[2];
   uint32_t inverted;
};
_Static_assert(sizeof(struct apex_dispatch_parameters) == APEX_DISPATCH_WORDS * 4,
              "trailer words shared with the internal kernels");

struct apex_bo {
   void *map;
   uint64_t va, size;
   uint32_t handle;
   bool system;
};

/* Internal single-queue device on one user-mode ring. A negative fd builds an
 * offline device for compiler tests: no memory, queue or submission. */
#define APEX_PRIVATE_BYTES (2u * 1024 * 1024)

struct apex_device {
   struct vk_device vk;
   struct vk_device_dispatch_table cmd_dispatch;
   struct vk_queue queue;
   struct vk_meta_device meta;
   int fd;
   struct util_vma_heap va_heap;
   mtx_t va_mutex;
   /* Serializes PRIME handle lookup and final backing destruction. */
   mtx_t memory_mutex;
   struct list_head memories;
   /* Queue: ring, read-only status page and doorbell page. The submit thread
    * owns sequence, kwait, the arenas and ring.wptr. */
   uint32_t queue_id;
   uint64_t status_va;
   void *status_map, *doorbell_map;
   struct apex_bo ring_bo;
   struct apex_ring ring;
   /* SIGNAL target retiring batch arenas without an interrupt. */
   struct apex_bo retire;
   /* Private data of the one dispatch in flight: APEX_PRIVATE_BYTES of
    * LOCAL at a 2 MiB-aligned GPUVA, the compute private base. */
   struct apex_bo private_arena;
   uint64_t sequence, kwait;
   struct list_head busy_arenas, free_arenas;
   /* A program upload since the last batch invalidates instruction caches. */
   bool programs_uploaded;
   /* Internal programs, compiled on first use. */
   struct apex_program *internal[APEX_INTERNAL_COUNT];
   /* Indirect draw scratch and parameter block, created on first use under
    * memory_mutex (see APEX_ARENA_* in apex_draw.h). */
   struct apex_bo arena;
   /* 4x resolve programs by format, compiled on first use. */
   struct apex_program *resolve[VK_FORMAT_ASTC_12x12_SRGB_BLOCK + 1];
};
VK_DEFINE_HANDLE_CASTS(apex_device, vk.base, VkDevice, VK_OBJECT_TYPE_DEVICE);

void apex_bo_finish(struct apex_device *device, struct apex_bo *bo);

VkFormatFeatureFlags2 apex_format_features(VkFormat format, bool buffer);
/* `prime` admits external images (APEX_DRM_CAP_PRIME_COHERENT). */
VkResult apex_image_format_properties(const VkPhysicalDeviceImageFormatInfo2 *info, bool prime,
                                      VkImageFormatProperties2 *properties);
bool apex_format_modifier_supported(VkFormat format);

VkResult apex_device_init(struct apex_device *device,
                          struct vk_physical_device *physical,
                          const VkDeviceCreateInfo *info,
                          const VkAllocationCallbacks *alloc, int fd);
void apex_device_finish(struct apex_device *device);
#endif
