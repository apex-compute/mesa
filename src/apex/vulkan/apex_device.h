/* SPDX-License-Identifier: MIT */
#ifndef APEX_DEVICE_H
#define APEX_DEVICE_H
#include "vk_device.h"
#include "vk_queue.h"
#include "vk_descriptor_set_layout.h"
#include "vk_meta.h"
#include "vk_sampler.h"
#include "vk_ycbcr_conversion.h"
#include "apex_format.h"
#include "apex_job.h"
#include "apex_cp.h"
#include "util/vma.h"

/* Internal compute kernels (apex_job.h). */
enum apex_internal {
   APEX_INTERNAL_COPY, APEX_INTERNAL_CLEAR, APEX_INTERNAL_QUERY_COPY, APEX_INTERNAL_ETC2, APEX_INTERNAL_BC,
   APEX_INTERNAL_XFB,
   APEX_INTERNAL_COUNT
};

/* Table rows of a pipeline. The descriptor limits keep every pipeline
 * layout within them: a combined image sampler counts as a sampler and a
 * sampled image, a storage image takes two rows and inline uniform blocks at
 * most 4 + 4096 / 32, so the limits sum to 3,852 rows. */
#define APEX_MAX_DESCRIPTORS 4096
#define APEX_SET_SAMPLERS 640
#define APEX_SET_SAMPLED_IMAGES 2048
#define APEX_SET_STORAGE_IMAGES 256
#define APEX_SET_UNIFORM_BUFFERS 128
#define APEX_SET_STORAGE_BUFFERS 256
#define APEX_SET_DYNAMIC_BUFFERS 64
#define APEX_SET_INPUT_ATTACHMENTS 8
#define APEX_SET_DESCRIPTORS 1024
/* Largest allocation: the per-client page tables map about 1 GiB. */
#define APEX_MAX_ALLOCATION (1024ull * 1024 * 1024)
/* Binding numbers below this; layouts index bindings by number. */
#define APEX_MAX_BINDINGS (1u << 16)
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
/* Set storage holds encoded table rows: `row` is the binding's first, with
 * `slots` rows per element (apex_descriptor_slots(), and a combined image
 * sampler with YCbCr conversions an image row per plane before its sampler
 * row). `slot` is its first row in a
 * pipeline's table; an update-after-bind binding takes one table row, a
 * buffer row over its rows in the set's pool memory, which the GPU reads at
 * execution. `dynamic` numbers a dynamic buffer binding's first dynamic
 * offset. Immutable sampler rows start at `immutable` in the layout's sampler
 * array (~0 when absent). An inline uniform block is one element of `bytes`
 * bytes: a buffer row followed by the data rows. */
struct apex_binding_layout {
   uint32_t row, slot, slots, dynamic, count, flags, immutable, bytes;
   VkDescriptorType type;
   VkShaderStageFlags stages;
};
/* Rows of set storage and of a table, and the elements of an
 * update-after-bind binding. */
#define APEX_MAX_SET_ROWS (1u << 21)
#define APEX_MAX_BINDING_ELEMENTS (1u << 20)
struct apex_set_layout {
   struct vk_descriptor_set_layout vk;
   uint32_t binding_count, row_count, slot_count;
   /* Whether a binding is in pool memory: sets of the layout live there. */
   bool pool_memory;
   uint32_t counts[APEX_DESCRIPTOR_BUCKETS];
   uint32_t (*samplers)[8];
   /* Each immutable sampler's YCbCr conversion; format 0 without one. */
   struct vk_ycbcr_conversion_state *ycbcr;
   struct apex_binding_layout bindings[];
};
/* Update-after-bind bindings, and every binding of an update-after-bind
 * layout except inline uniform blocks, carry UPDATE_AFTER_BIND. */
static inline bool
apex_binding_in_memory(const struct apex_binding_layout *binding)
{
   return binding->flags & VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
}
/* `row` is the texture-unit sampler descriptor. */
struct apex_sampler {
   struct vk_sampler vk;
   uint32_t row[8];
};
/* Buffer instructions check each access against bytes; flag bit 0 makes
 * out-of-range loads read zero and drops out-of-range stores. */
#define APEX_BUFFER_ROBUST 1u
struct apex_buffer_descriptor {
   uint32_t low, high, bytes, flags;
};
/* A 32-byte table row: a buffer descriptor, a texture-unit image or sampler
 * descriptor (apex_hw.h), a texel-buffer row or format words. */
union apex_descriptor {
   struct apex_buffer_descriptor buffer;
   uint32_t words[8];
};

struct apex_bo {
   void *map;
   uint64_t va, size;
   uint32_t handle;
   bool system;
};

/* Internal single-queue device on one user-mode ring. A negative fd builds an
 * offline device for compiler tests: no memory, queue or submission. */
#define APEX_PRIVATE_BYTES (2u * 1024 * 1024)
/* The queue's bin pool (Docs/architecture.md, Pipeline and render passes):
 * the pass record, the draw slots, the bin heads, the vertex-output and
 * primitive regions and the bin chunks to the end. */
#define APEX_BIN_POOL_BYTES (32u * 1024 * 1024)
#define APEX_POOL_DRAW_BYTES (2u * 1024 * 1024)
#define APEX_POOL_VERTEX_BYTES (8u * 1024 * 1024)
#define APEX_POOL_PRIMITIVE_BYTES (6u * 1024 * 1024)

/* Queue family 0 runs graphics, compute and transfer, family 1 transfer
 * only; each family has one queue on its own kernel queue. */
#define APEX_QUEUE_FAMILIES 2

/* One kernel queue: ring, read-only status page and doorbell page. The
 * queue's submitter owns sequence, kwait, the arenas and ring.wptr. */
struct apex_queue {
   struct vk_queue vk;
   uint32_t id;
   uint32_t priority; /* the kernel's effective drm_apex_queue_create priority */
   /* The kernel's status page, and its kwait word in LOCAL. */
   uint64_t status_va, kwait_va;
   void *status_map, *doorbell_map;
   struct apex_bo ring_bo;
   struct apex_ring ring;
   /* SIGNAL target retiring batch arenas without an interrupt. */
   struct apex_bo retire;
   /* Private data of the dispatches using private memory, one at a time:
    * APEX_PRIVATE_BYTES of LOCAL at a 2 MiB-aligned GPUVA. */
   struct apex_bo private_arena;
   /* The queue's bin pool, written by the geometry front end; family 0 only. */
   struct apex_bo bin_pool;
   uint64_t sequence, kwait;
   struct list_head busy_arenas, free_arenas;
   /* A program upload since the queue's last batch invalidates its
    * instruction caches. */
   bool programs_uploaded;
};

/* vk_meta's host data (clear and blit vertices) in chunks of LOCAL that
 * command buffers sub-allocate while recording and free at reset, so a
 * clear needs no kernel call and no address is unmapped under the GPU. */
#define APEX_UPLOAD_CHUNK_BYTES (1u << 20)
struct apex_upload_chunk {
   struct list_head link;
   struct apex_bo bo;
   struct util_vma_heap heap;
};
struct apex_upload;

struct apex_device {
   struct vk_device vk;
   struct vk_device_dispatch_table cmd_dispatch;
   struct apex_queue queues[APEX_QUEUE_FAMILIES];
   struct vk_meta_device meta;
   int fd;
   struct util_vma_heap va_heap;
   mtx_t va_mutex;
   /* Serializes PRIME handle lookup and final backing destruction. */
   mtx_t memory_mutex;
   struct list_head memories;
   mtx_t upload_mutex;
   struct list_head upload_chunks;
   /* Internal programs and the empty fragment program of depth-only draws,
    * compiled on first use under memory_mutex. */
   struct apex_program *internal[APEX_INTERNAL_COUNT];
   struct apex_program *empty_fragment;
};
VK_DEFINE_HANDLE_CASTS(apex_device, vk.base, VkDevice, VK_OBJECT_TYPE_DEVICE);

void apex_bo_finish(struct apex_device *device, struct apex_bo *bo);
/* Takes `size` bytes of the upload heap: their address and CPU mapping. */
VkResult apex_upload_alloc(struct apex_device *device, uint64_t size, struct apex_upload *upload, void **map);
void apex_upload_free(struct apex_device *device, struct apex_upload *upload);

VkFormatFeatureFlags2 apex_format_features(VkFormat format, bool buffer);
VkResult apex_image_format_properties(const VkPhysicalDeviceImageFormatInfo2 *info,
                                      VkImageFormatProperties2 *properties);
bool apex_format_modifier_supported(VkFormat format);
VkFormatFeatureFlags2 apex_linear_format_features(VkFormat format);

VkResult apex_bo_create(struct apex_device *device, uint64_t size, uint32_t flags,
                        uint32_t gem_flags, uint64_t reserved_va, struct apex_bo *bo);
struct apex_program;
/* Uploads a program on first use; the next batch invalidates instruction caches. */
VkResult apex_program_upload(struct apex_device *device, struct apex_program *program);

VkResult apex_device_init(struct apex_device *device,
                          struct vk_physical_device *physical,
                          const VkDeviceCreateInfo *info,
                          const VkAllocationCallbacks *alloc, int fd);
void apex_device_finish(struct apex_device *device);
/* A zeroed set layout of `binding_count` bindings and `samplers` immutable
 * sampler rows. */
struct apex_set_layout *apex_set_layout_alloc(struct apex_device *device, VkDescriptorSetLayoutCreateFlags flags,
                                              unsigned binding_count, unsigned samplers);

#endif
