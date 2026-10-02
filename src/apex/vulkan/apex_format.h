/* SPDX-License-Identifier: MIT */
#ifndef APEX_FORMAT_H
#define APEX_FORMAT_H
#include <stdbool.h>
#include <stdint.h>
#include <vulkan/vulkan_core.h>
#include "util/format/u_formats.h"

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

bool apex_format_encode(VkFormat format, VkImageAspectFlags aspect, const VkComponentMapping *mapping,
                        uint32_t out[3]);
/* ETC2/EAC images keep a decoded plane: its format and the decode kind
 * (APEX_ETC2_* / APEX_EAC_*), or VK_FORMAT_UNDEFINED for other formats. */
VkFormat apex_decoded_format(VkFormat format, uint32_t *kind);

/* Table rows occupied by one element of a descriptor binding: combined
 * image samplers hold the image then the sampler descriptor, storage images
 * the image descriptor then the view's format words. */
static inline unsigned
apex_descriptor_slots(VkDescriptorType type)
{
   switch (type) {
   case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
   case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE: return 2;
   default: return 1;
   }
}
#endif
