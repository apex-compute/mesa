/* SPDX-License-Identifier: MIT */
#ifndef APEX_FORMAT_H
#define APEX_FORMAT_H
#include <stdbool.h>
#include <stdint.h>
#include <vulkan/vulkan_core.h>
#include "util/format/u_formats.h"

struct vk_sampler;

/* Format word 0: texel bytes [4:0], sRGB [5], output swizzle [19:8] (3 bits
 * per RGBA output: stored channel 0-3, 4 zero, 5 one), pure integer [20],
 * packed RGB9E5 [21], packed R11G11B10 unsigned floats [22]; the packed
 * formats have no channel fields.
 * Words 1-2: stored channels 0-3 as 16-bit fields: bit offset [6:0], bit size
 * [12:7], channel type [15:13]. */
enum apex_channel_type {
   APEX_CHANNEL_VOID, APEX_CHANNEL_UNORM, APEX_CHANNEL_SNORM, APEX_CHANNEL_UINT,
   APEX_CHANNEL_SINT, APEX_CHANNEL_FLOAT, APEX_CHANNEL_USCALED, APEX_CHANNEL_SSCALED,
};
#define APEX_SWIZZLE_0 4
#define APEX_SWIZZLE_1 5
#define APEX_FORMAT_RGB9E5 (1u << 21)
#define APEX_FORMAT_R11G11B10 (1u << 22)

/* Sampled image descriptor: two 32-byte table rows. The view selects levels
 * and layers of the image's mip-major linear allocation; the shader derives
 * each level's pitch and offset from the level-0 extent. */
struct apex_sampled_descriptor {
   uint32_t low, high;              /* image level 0, layer 0 */
   uint32_t width, height, depth;   /* image level 0 */
   uint32_t layers;                 /* image array layers */
   uint32_t base_level, levels;     /* view */
   uint32_t base_layer, layer_count;
   uint32_t format[3];
   uint32_t view_type;              /* VkImageViewType */
   uint32_t samples;                /* 1 or 4; samples of a texel are consecutive */
   uint32_t reserved;
};

/* Sampler row: word 0 = mag linear [0], min linear [1], mip linear [2],
 * address U [6:4], V [10:8], W [14:12], compare enable [16], compare op
 * [19:17], unnormalized [20], reduction [22:21] (0 average, 1 min, 2 max);
 * words 1-3 = LOD bias, min LOD, max LOD (FP32); words 4-7 border color bits. */
struct apex_sampler_descriptor {
   uint32_t flags;
   float bias, min_lod, max_lod;
   uint32_t border[4];
};

bool apex_format_encode(VkFormat format, VkImageAspectFlags aspect, const VkComponentMapping *mapping,
                        uint32_t out[3]);
bool apex_attachment_format_supported(enum pipe_format format);
void apex_sampler_encode(const VkSamplerCreateInfo *info, const struct vk_sampler *sampler,
                         uint32_t out[8]);

/* Table rows occupied by one element of a descriptor binding. */
static inline unsigned
apex_descriptor_slots(VkDescriptorType type)
{
   switch (type) {
   case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER: return 3;
   case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
   case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
   case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT: return 2;
   default: return 1;
   }
}
#endif
