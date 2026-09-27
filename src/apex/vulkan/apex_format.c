/* SPDX-License-Identifier: MIT */
/* Data-driven texel formats and sampler rows consumed by the software sampler
 * (apex_texture.c). A format word triple describes up to four stored channels
 * as bit fields, so one decoder serves every plain format. */
#include "apex_device.h"
#include "apex_format.h"
#include "util/format/u_format.h"
#include "vk_format.h"
#include "vk_sampler.h"

static uint32_t
channel_type(const struct util_format_channel_description *ch)
{
   switch (ch->type) {
   case UTIL_FORMAT_TYPE_UNSIGNED:
      return ch->normalized ? APEX_CHANNEL_UNORM : ch->pure_integer ? APEX_CHANNEL_UINT :
                                                                      APEX_CHANNEL_USCALED;
   case UTIL_FORMAT_TYPE_SIGNED:
      return ch->normalized ? APEX_CHANNEL_SNORM : ch->pure_integer ? APEX_CHANNEL_SINT :
                                                                      APEX_CHANNEL_SSCALED;
   case UTIL_FORMAT_TYPE_FLOAT:
      return ch->size == 32 || ch->size == 16 ? APEX_CHANNEL_FLOAT : APEX_CHANNEL_VOID;
   default:
      return APEX_CHANNEL_VOID;
   }
}

bool
apex_format_encode(VkFormat format, VkImageAspectFlags aspect, const VkComponentMapping *mapping,
                   uint32_t out[3])
{
   enum pipe_format pformat = vk_format_to_pipe_format(format);
   if (aspect == VK_IMAGE_ASPECT_STENCIL_BIT)
      pformat = util_format_stencil_only(pformat);
   else if (aspect == VK_IMAGE_ASPECT_DEPTH_BIT && util_format_is_depth_and_stencil(pformat))
      pformat = util_format_get_depth_only(pformat);
   const struct util_format_description *desc = util_format_description(pformat);
   if (!desc || desc->layout != UTIL_FORMAT_LAYOUT_PLAIN || desc->block.width != 1 ||
       desc->block.height != 1 || desc->block.bits % 8 || desc->block.bits > 128)
      return false;
   /* Depth/stencil planes keep the combined texel size in memory. */
   unsigned bytes = util_format_get_blocksize(vk_format_to_pipe_format(format));
   out[0] = bytes | (desc->colorspace == UTIL_FORMAT_COLORSPACE_SRGB) << 5;
   out[1] = out[2] = 0;
   for (unsigned c = 0; c < desc->nr_channels; c++) {
      const struct util_format_channel_description *ch = &desc->channel[c];
      uint32_t type = channel_type(ch);
      if (ch->type != UTIL_FORMAT_TYPE_VOID && (type == APEX_CHANNEL_VOID || ch->size > 32))
         return false;
      uint32_t field = ch->shift | ch->size << 7 | type << 13;
      out[1 + c / 2] |= field << (16 * (c % 2));
   }
   /* Output component c reads format component view[c], stored as swizzle[f]. */
   const VkComponentSwizzle view[4] = {
      mapping ? mapping->r : VK_COMPONENT_SWIZZLE_IDENTITY,
      mapping ? mapping->g : VK_COMPONENT_SWIZZLE_IDENTITY,
      mapping ? mapping->b : VK_COMPONENT_SWIZZLE_IDENTITY,
      mapping ? mapping->a : VK_COMPONENT_SWIZZLE_IDENTITY,
   };
   for (unsigned c = 0; c < 4; c++) {
      VkComponentSwizzle s = view[c] == VK_COMPONENT_SWIZZLE_IDENTITY ?
         VK_COMPONENT_SWIZZLE_R + c : view[c];
      uint32_t stored;
      if (s == VK_COMPONENT_SWIZZLE_ZERO) {
         stored = APEX_SWIZZLE_0;
      } else if (s == VK_COMPONENT_SWIZZLE_ONE) {
         stored = APEX_SWIZZLE_1;
      } else {
         unsigned f = desc->swizzle[s - VK_COMPONENT_SWIZZLE_R];
         stored = f <= PIPE_SWIZZLE_W ? f : f == PIPE_SWIZZLE_1 ? APEX_SWIZZLE_1 : APEX_SWIZZLE_0;
      }
      out[0] |= stored << (8 + 3 * c);
   }
   bool integer = util_format_is_pure_integer(pformat);
   out[0] |= integer << 20;
   return true;
}

void
apex_sampler_encode(const VkSamplerCreateInfo *info, const struct vk_sampler *sampler,
                    uint32_t out[8])
{
   const VkSamplerReductionModeCreateInfo *reduction =
      vk_find_struct_const(info->pNext, SAMPLER_REDUCTION_MODE_CREATE_INFO);
   uint32_t mode = reduction ? reduction->reductionMode : VK_SAMPLER_REDUCTION_MODE_WEIGHTED_AVERAGE;
   float values[3] = {info->mipLodBias, info->minLod, info->maxLod};
   out[0] = (info->magFilter == VK_FILTER_LINEAR) |
            (info->minFilter == VK_FILTER_LINEAR) << 1 |
            (info->mipmapMode == VK_SAMPLER_MIPMAP_MODE_LINEAR) << 2 |
            info->addressModeU << 4 | info->addressModeV << 8 | info->addressModeW << 12 |
            (uint32_t)info->compareEnable << 16 | (info->compareEnable ? info->compareOp : 0) << 17 |
            (uint32_t)info->unnormalizedCoordinates << 20 |
            (mode == VK_SAMPLER_REDUCTION_MODE_MIN ? 1u : mode == VK_SAMPLER_REDUCTION_MODE_MAX ? 2u : 0u) << 21;
   memcpy(&out[1], values, sizeof(values));
   memcpy(&out[4], &sampler->border_color_value, 4 * sizeof(uint32_t));
}

/* Color attachments: plain formats of 8 to 128 bits whose channels are
 * normalized or pure integers of up to 32 bits, FP16 or FP32, each within
 * one word (see the fragment kernel in apex_graphics.c). */
bool
apex_attachment_format_supported(enum pipe_format format)
{
   const struct util_format_description *desc = util_format_description(format);
   if (!desc || desc->layout != UTIL_FORMAT_LAYOUT_PLAIN || desc->block.width != 1 ||
       desc->block.height != 1 || desc->block.bits < 8 || desc->block.bits > 128 ||
       !util_is_power_of_two_nonzero(desc->block.bits))
      return false;
   for (unsigned c = 0; c < desc->nr_channels; c++) {
      const struct util_format_channel_description *ch = &desc->channel[c];
      bool integer = (ch->type == UTIL_FORMAT_TYPE_UNSIGNED || ch->type == UTIL_FORMAT_TYPE_SIGNED) &&
                     (ch->normalized || ch->pure_integer);
      bool fp = ch->type == UTIL_FORMAT_TYPE_FLOAT && (ch->size == 16 || ch->size == 32);
      if ((ch->type != UTIL_FORMAT_TYPE_VOID && !integer && !fp) || ch->shift % 32 + ch->size > 32)
         return false;
   }
   return true;
}
