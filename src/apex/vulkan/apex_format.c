/* SPDX-License-Identifier: MIT */
/* Data-driven texel formats of texel buffers and storage images
 * (apex_texture.c). A format word triple describes up to four stored channels
 * as bit fields, so one decoder serves every plain format. */
#include "apex_device.h"
#include "apex_format.h"
#include "util/format/u_format.h"
#include "vk_format.h"

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
   /* Shared-exponent and small-float RGB formats decode as whole words. */
   if (pformat == PIPE_FORMAT_R9G9B9E5_FLOAT || pformat == PIPE_FORMAT_R11G11B10_FLOAT) {
      out[0] = 4 | (pformat == PIPE_FORMAT_R9G9B9E5_FLOAT ? APEX_FORMAT_RGB9E5 : APEX_FORMAT_R11G11B10);
      const VkComponentSwizzle view[4] = {
         mapping ? mapping->r : VK_COMPONENT_SWIZZLE_IDENTITY,
         mapping ? mapping->g : VK_COMPONENT_SWIZZLE_IDENTITY,
         mapping ? mapping->b : VK_COMPONENT_SWIZZLE_IDENTITY,
         mapping ? mapping->a : VK_COMPONENT_SWIZZLE_IDENTITY,
      };
      for (unsigned c = 0; c < 4; c++) {
         VkComponentSwizzle s = view[c] == VK_COMPONENT_SWIZZLE_IDENTITY ? VK_COMPONENT_SWIZZLE_R + c : view[c];
         uint32_t stored = s == VK_COMPONENT_SWIZZLE_ZERO ? APEX_SWIZZLE_0 :
                           s == VK_COMPONENT_SWIZZLE_ONE || s == VK_COMPONENT_SWIZZLE_A ? APEX_SWIZZLE_1 :
                           s - VK_COMPONENT_SWIZZLE_R;
         out[0] |= stored << (8 + 3 * c);
      }
      out[1] = out[2] = 0;
      return true;
   }
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
   /* Format component f is stored channel rgba[f]. Depth and stencil
    * convert to RGBA as (D or S, 0, 0, 1). */
   unsigned rgba[4] = {desc->swizzle[0], desc->swizzle[1], desc->swizzle[2], desc->swizzle[3]};
   if (desc->colorspace == UTIL_FORMAT_COLORSPACE_ZS) {
      rgba[0] = desc->swizzle[aspect == VK_IMAGE_ASPECT_STENCIL_BIT];
      rgba[1] = rgba[2] = PIPE_SWIZZLE_0;
      rgba[3] = PIPE_SWIZZLE_1;
   }
   /* Output component c reads format component view[c], stored as rgba[f]. */
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
         unsigned f = rgba[s - VK_COMPONENT_SWIZZLE_R];
         stored = f <= PIPE_SWIZZLE_W ? f : f == PIPE_SWIZZLE_1 ? APEX_SWIZZLE_1 : APEX_SWIZZLE_0;
      }
      out[0] |= stored << (8 + 3 * c);
   }
   bool integer = util_format_is_pure_integer(pformat);
   out[0] |= integer << 20;
   return true;
}

VkFormat
apex_decoded_format(VkFormat format, uint32_t *kind)
{
   static const struct { VkFormat format, decoded; uint32_t kind; } table[] = {
      {VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK, VK_FORMAT_R8G8B8A8_UNORM, APEX_ETC2_RGB8},
      {VK_FORMAT_ETC2_R8G8B8_SRGB_BLOCK, VK_FORMAT_R8G8B8A8_SRGB, APEX_ETC2_RGB8},
      {VK_FORMAT_ETC2_R8G8B8A1_UNORM_BLOCK, VK_FORMAT_R8G8B8A8_UNORM, APEX_ETC2_RGBA1},
      {VK_FORMAT_ETC2_R8G8B8A1_SRGB_BLOCK, VK_FORMAT_R8G8B8A8_SRGB, APEX_ETC2_RGBA1},
      {VK_FORMAT_ETC2_R8G8B8A8_UNORM_BLOCK, VK_FORMAT_R8G8B8A8_UNORM, APEX_ETC2_RGBA8},
      {VK_FORMAT_ETC2_R8G8B8A8_SRGB_BLOCK, VK_FORMAT_R8G8B8A8_SRGB, APEX_ETC2_RGBA8},
      {VK_FORMAT_EAC_R11_UNORM_BLOCK, VK_FORMAT_R32_SFLOAT, APEX_EAC_R11},
      {VK_FORMAT_EAC_R11_SNORM_BLOCK, VK_FORMAT_R32_SFLOAT, APEX_EAC_R11_SNORM},
      {VK_FORMAT_EAC_R11G11_UNORM_BLOCK, VK_FORMAT_R32G32_SFLOAT, APEX_EAC_RG11},
      {VK_FORMAT_EAC_R11G11_SNORM_BLOCK, VK_FORMAT_R32G32_SFLOAT, APEX_EAC_RG11_SNORM},
   };
   for (unsigned i = 0; i < ARRAY_SIZE(table); i++) {
      if (table[i].format == format) {
         if (kind)
            *kind = table[i].kind;
         return table[i].decoded;
      }
   }
   return VK_FORMAT_UNDEFINED;
}
