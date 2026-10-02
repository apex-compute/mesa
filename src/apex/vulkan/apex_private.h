/* SPDX-License-Identifier: MIT */
/* Driver objects shared by the device (apex_device.c) and command recording
 * (apex_cmd.c). */
#ifndef APEX_PRIVATE_H
#define APEX_PRIVATE_H
#include "apex_device.h"
#include "apex_graphics.h"
#include "apex_hw.h"
#include "apex_job.h"
#include "vk_buffer.h"
#include "vk_command_buffer.h"
#include "vk_device_memory.h"
#include "vk_image.h"
#include "vk_query_pool.h"
#include "util/u_dynarray.h"

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
};
/* Images keep the texture unit's tiled layout (apex_hw.h) when their texels
 * are 1, 2, 4, 8 or 16 bytes and their tiling optimal, else a linear one.
 * ETC2/EAC images append a decoded plane at `decoded` bytes: a tiled image
 * of the decoded format that sampling reads. */
struct apex_image {
   struct vk_image vk;
   struct apex_memory *memory;
   VkDeviceSize offset, size, decoded;
   /* Explicit DRM plane offset from the memory binding. */
   VkDeviceSize plane_offset;
   struct apex_hw_layout layout, decoded_layout;
};
static inline uint64_t
apex_image_va(const struct apex_image *image)
{
   return image->memory->storage->bo.va + image->offset;
}

struct apex_descriptor_pool {
   struct vk_object_base base;
   struct list_head sets;
   uint32_t capacity, allocated;
   uint64_t descriptors[APEX_DESCRIPTOR_BUCKETS];
   uint64_t used[APEX_DESCRIPTOR_BUCKETS];
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
/* Immutable command binding; tables retain the offsets supplied at bind. */
struct apex_bound_set {
   struct apex_descriptor_set *set;
   uint32_t refs;
   uint32_t offsets[];
};
struct apex_event {
   struct vk_object_base base;
   struct apex_bo bo;
};
struct apex_query_pool {
   struct vk_query_pool vk;
   struct apex_bo bo;
};
/* Submission memory: the command buffer's data, tables and IB copy, retired
 * when the batch's SIGNAL reaches `sequence`. */
struct apex_arena {
   struct list_head link;
   struct apex_bo bo;
   uint64_t sequence;
};
/* Recorded device memory created at first submission: a snapshot of host
 * bytes for vk_meta. */
struct apex_upload {
   struct list_head link;
   struct apex_bo bo;
   uint64_t reserved_va, size;
   uint8_t data[];
};

/* A descriptor table written at submission: the rows its programs use, the
 * push image, then the trailer (grid and internal-kernel job words). */
struct apex_table {
   struct list_head link;
   struct apex_program *programs[2];
   struct apex_bound_set *sets[MESA_VK_MAX_DESCRIPTOR_SETS];
   uint8_t push[APEX_MAX_PUSH_CONSTANTS];
   uint32_t trailer[APEX_TRAILER_WORDS + APEX_JOB_WORDS];
   /* Trailer words w (bit w) whose pair {w, w + 1} holds a data offset,
    * written as its GPUVA. */
   uint64_t relocs;
   uint64_t offset;      /* in the submission arena */
};
/* IB dwords (low, high) patched with a GPUVA at submission. */
enum apex_patch_kind {
   APEX_PATCH_TABLE,     /* a table's start */
   APEX_PATCH_GRID,      /* a table's trailer */
   APEX_PATCH_PROGRAM,   /* an uploaded program */
   APEX_PATCH_DATA,      /* an offset in the command buffer's data */
};
struct apex_patch {
   uint32_t dword;
   enum apex_patch_kind kind;
   union {
      struct apex_table *table;
      struct apex_program *program;
      uint64_t offset;
   };
};

/* One attachment of the current render pass. */
struct apex_attachment {
   struct apex_image *image;
   VkFormat format;
   uint32_t level, layer;
   VkImageAspectFlags aspects;
};

struct apex_command_buffer {
   struct vk_command_buffer vk;
   /* Packets in recording order, beginning with a full barrier. `pending`
    * holds the BARRIER classes of work issued since the last barrier. */
   struct apex_ib ib;
   uint32_t pending;
   /* Submission data: state blocks (64-byte aligned), tables and patches. */
   struct util_dynarray data, patches;
   struct list_head tables, uploads, push_sets;
   /* push_sets owns push descriptor sets; pushed[] is the latest per set. */
   struct apex_descriptor_set *pushed[MESA_VK_MAX_DESCRIPTOR_SETS];
   /* Compute bind point. */
   struct apex_pipeline *pipeline;
   struct apex_bound_set *sets[MESA_VK_MAX_DESCRIPTOR_SETS];
   uint8_t push[APEX_MAX_PUSH_CONSTANTS];
   /* A dispatch using the private arena may still run. */
   bool private_pending;
   /* Graphics bind point. */
   struct apex_bound_set *graphics_sets[MESA_VK_MAX_DESCRIPTOR_SETS];
   uint8_t graphics_push[APEX_MAX_PUSH_CONSTANTS];
   struct apex_shader *vertex, *fragment;
   struct vk_vertex_input_state vertex_input;
   struct vk_sample_locations_state sample_locations;
   struct apex_hw_binding bindings[APEX_HW_MAX_BINDINGS];
   struct { uint64_t va, size; uint32_t type; } index;
   /* The graphics table of the last draw, reused until bindings change. */
   struct apex_table *graphics_table;
   /* State blocks and registers last emitted: blocks by contents, register
    * groups by value; `emitted` false forces every group. */
   struct {
      bool emitted;
      uint32_t vertex[11], fragment[10], dynamic[APEX_HW_DYNAMIC_DWORDS];
      struct apex_program *vertex_program, *fragment_program;
      struct apex_table *table;
      uint32_t view;
      uint64_t blocks[5];
      uint32_t vertex_input[APEX_HW_VERTEX_INPUT_DWORDS], raster[APEX_HW_RASTER_DWORDS],
               depth_stencil[APEX_HW_DEPTH_STENCIL_DWORDS], blend[APEX_HW_BLEND_DWORDS],
               viewport[APEX_HW_VIEWPORT_DWORDS];
   } state;
   /* Transform feedback: the bound buffers and, while active, the state
    * block (APEX_XFB_STATE_*) at data offset `state`; the active stream
    * query's slot VA. */
   struct {
      struct { uint64_t va, size; } buffers[APEX_XFB_BUFFERS];
      bool active;
      uint64_t state, query;
   } xfb;
   /* Occlusion query slots 0-7 in use, and availability writes waiting
    * for the end of the render pass. */
   uint8_t query_slots;
   struct { uint64_t va; uint8_t slot; } queries[8];
   struct util_dynarray pass_availability;
   unsigned meta;
   /* Conditional rendering: the predicate while active, whether PREDICATE
    * is on in the queue, and whether vk_meta work counts as the
    * application's (vkCmdClearAttachments). */
   struct {
      uint64_t va;
      bool active, inverted, on, meta;
   } predicate;
   struct {
      bool active, suspending;
      VkRect2D area;
      uint32_t layers, view_mask, samples, color_count;
      struct apex_attachment color[APEX_HW_MAX_COLOR], depth;
      bool has_depth, has_stencil;
      struct apex_hw_pass pass;
   } rendering;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(apex_memory, vk.base, VkDeviceMemory, VK_OBJECT_TYPE_DEVICE_MEMORY);
VK_DEFINE_NONDISP_HANDLE_CASTS(apex_buffer, vk.base, VkBuffer, VK_OBJECT_TYPE_BUFFER);
VK_DEFINE_NONDISP_HANDLE_CASTS(apex_image, vk.base, VkImage, VK_OBJECT_TYPE_IMAGE);
VK_DEFINE_NONDISP_HANDLE_CASTS(apex_descriptor_pool, base, VkDescriptorPool, VK_OBJECT_TYPE_DESCRIPTOR_POOL);
VK_DEFINE_NONDISP_HANDLE_CASTS(apex_descriptor_set, base, VkDescriptorSet, VK_OBJECT_TYPE_DESCRIPTOR_SET);
VK_DEFINE_HANDLE_CASTS(apex_command_buffer, vk.base, VkCommandBuffer, VK_OBJECT_TYPE_COMMAND_BUFFER);
VK_DEFINE_NONDISP_HANDLE_CASTS(apex_sampler, vk.base, VkSampler, VK_OBJECT_TYPE_SAMPLER);
VK_DEFINE_NONDISP_HANDLE_CASTS(apex_event, base, VkEvent, VK_OBJECT_TYPE_EVENT);
VK_DEFINE_NONDISP_HANDLE_CASTS(apex_query_pool, vk.base, VkQueryPool, VK_OBJECT_TYPE_QUERY_POOL);

/* apex_device.c */
struct vk_descriptor_update_template;
void apex_descriptor_set_free(struct vk_device *device, struct apex_descriptor_set *set);
void apex_descriptor_set_write(struct apex_descriptor_set *set, const VkWriteDescriptorSet *write);
void apex_descriptor_set_write_template(struct apex_descriptor_set *set,
                                        const struct vk_descriptor_update_template *templ,
                                        const void *data);
/* Reuses the smallest retired arena that fits, else creates one. */
struct apex_arena *apex_arena_get(struct apex_device *device, uint64_t bytes);
/* The layout of a view's level: the decoded plane of ETC2/EAC images. */
const struct apex_hw_layout *apex_image_layout(const struct apex_image *image, VkFormat view_format,
                                               uint64_t *va);

/* apex_cmd.c */
extern const struct vk_command_buffer_ops apex_command_buffer_ops;
struct vk_meta_device;
void apex_cmd_meta_init(struct vk_meta_device *meta);
void apex_cmd_entrypoints(struct vk_device_entrypoint_table *table);
/* Writes the command buffer's data, tables and patched IB into one arena. */
VkResult apex_cmd_prepare(struct apex_device *device, struct apex_command_buffer *cmd,
                          struct apex_arena **out, uint64_t *ib_va);
#endif
