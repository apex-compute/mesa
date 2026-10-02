/* SPDX-License-Identifier: MIT */
#include "apex_hw.h"
#include "vk_graphics_state.h"
#include "vk_sampler.h"
#include "util/bitscan.h"
#include "util/macros.h"
#include "util/u_math.h"
#include <math.h>
#include <string.h>

static void
put(uint32_t *w, unsigned lo, unsigned width, uint64_t v)
{
   for (unsigned i = 0; i < width; i++) {
      uint32_t bit = 1u << ((lo + i) & 31);
      if ((v >> i) & 1)
         w[(lo + i) >> 5] |= bit;
      else
         w[(lo + i) >> 5] &= ~bit;
   }
}

static uint32_t
float_bits(float value)
{
   uint32_t bits;
   memcpy(&bits, &value, sizeof(bits));
   return bits;
}

/* ---- Formats ------------------------------------------------------------ */

struct format_entry {
   VkFormat format;
   uint8_t class, type, bgr;
};

/* Every color format the texture unit and the ROP store natively. */
static const struct format_entry formats[] = {
#define T5(fmt, cls) \
   {VK_FORMAT_##fmt##_UNORM, APEX_HW_CLASS_##cls, APEX_HW_UNORM, 0}, \
   {VK_FORMAT_##fmt##_SNORM, APEX_HW_CLASS_##cls, APEX_HW_SNORM, 0}, \
   {VK_FORMAT_##fmt##_UINT, APEX_HW_CLASS_##cls, APEX_HW_UINT, 0}, \
   {VK_FORMAT_##fmt##_SINT, APEX_HW_CLASS_##cls, APEX_HW_SINT, 0}
   T5(R8, R8), {VK_FORMAT_R8_SRGB, APEX_HW_CLASS_R8, APEX_HW_SRGB, 0},
   T5(R8G8, RG8), {VK_FORMAT_R8G8_SRGB, APEX_HW_CLASS_RG8, APEX_HW_SRGB, 0},
   T5(R8G8B8A8, RGBA8), {VK_FORMAT_R8G8B8A8_SRGB, APEX_HW_CLASS_RGBA8, APEX_HW_SRGB, 0},
   {VK_FORMAT_B8G8R8A8_UNORM, APEX_HW_CLASS_RGBA8, APEX_HW_UNORM, 1},
   {VK_FORMAT_B8G8R8A8_SNORM, APEX_HW_CLASS_RGBA8, APEX_HW_SNORM, 1},
   {VK_FORMAT_B8G8R8A8_UINT, APEX_HW_CLASS_RGBA8, APEX_HW_UINT, 1},
   {VK_FORMAT_B8G8R8A8_SINT, APEX_HW_CLASS_RGBA8, APEX_HW_SINT, 1},
   {VK_FORMAT_B8G8R8A8_SRGB, APEX_HW_CLASS_RGBA8, APEX_HW_SRGB, 1},
   {VK_FORMAT_A8B8G8R8_UNORM_PACK32, APEX_HW_CLASS_RGBA8, APEX_HW_UNORM, 0},
   {VK_FORMAT_A8B8G8R8_SNORM_PACK32, APEX_HW_CLASS_RGBA8, APEX_HW_SNORM, 0},
   {VK_FORMAT_A8B8G8R8_UINT_PACK32, APEX_HW_CLASS_RGBA8, APEX_HW_UINT, 0},
   {VK_FORMAT_A8B8G8R8_SINT_PACK32, APEX_HW_CLASS_RGBA8, APEX_HW_SINT, 0},
   {VK_FORMAT_A8B8G8R8_SRGB_PACK32, APEX_HW_CLASS_RGBA8, APEX_HW_SRGB, 0},
   T5(R16, R16), {VK_FORMAT_R16_SFLOAT, APEX_HW_CLASS_R16, APEX_HW_FLOAT, 0},
   T5(R16G16, RG16), {VK_FORMAT_R16G16_SFLOAT, APEX_HW_CLASS_RG16, APEX_HW_FLOAT, 0},
   T5(R16G16B16A16, RGBA16), {VK_FORMAT_R16G16B16A16_SFLOAT, APEX_HW_CLASS_RGBA16, APEX_HW_FLOAT, 0},
#undef T5
#define T3(fmt, cls) \
   {VK_FORMAT_##fmt##_UINT, APEX_HW_CLASS_##cls, APEX_HW_UINT, 0}, \
   {VK_FORMAT_##fmt##_SINT, APEX_HW_CLASS_##cls, APEX_HW_SINT, 0}, \
   {VK_FORMAT_##fmt##_SFLOAT, APEX_HW_CLASS_##cls, APEX_HW_FLOAT, 0}
   T3(R32, R32), T3(R32G32, RG32), T3(R32G32B32A32, RGBA32),
#undef T3
   {VK_FORMAT_R5G6B5_UNORM_PACK16, APEX_HW_CLASS_565, APEX_HW_UNORM, 0},
   {VK_FORMAT_B5G6R5_UNORM_PACK16, APEX_HW_CLASS_565, APEX_HW_UNORM, 1},
   {VK_FORMAT_A2B10G10R10_UNORM_PACK32, APEX_HW_CLASS_RGB10A2, APEX_HW_UNORM, 0},
   {VK_FORMAT_A2B10G10R10_UINT_PACK32, APEX_HW_CLASS_RGB10A2, APEX_HW_UINT, 0},
   {VK_FORMAT_A2R10G10B10_UNORM_PACK32, APEX_HW_CLASS_RGB10A2, APEX_HW_UNORM, 1},
   {VK_FORMAT_A2R10G10B10_UINT_PACK32, APEX_HW_CLASS_RGB10A2, APEX_HW_UINT, 1},
};

static const unsigned class_channels[] = {0, 1, 2, 4, 1, 2, 4, 1, 2, 4, 3, 4, 1};

static const struct format_entry *
find_format(VkFormat format)
{
   for (unsigned i = 0; i < ARRAY_SIZE(formats); i++)
      if (formats[i].format == format)
         return &formats[i];
   return NULL;
}

bool
apex_hw_texture_format(VkFormat format, VkImageAspectFlags aspect, uint8_t *code, uint8_t channel[4])
{
   uint8_t cls, type, bgr = 0;
   /* Depth reads (D, 0, 0, 1) and stencil (S, 0, 0, 1). */
   int read = -1;
   bool stencil = aspect == VK_IMAGE_ASPECT_STENCIL_BIT;
   switch (format) {
   case VK_FORMAT_D16_UNORM: cls = APEX_HW_CLASS_R16, type = APEX_HW_UNORM, read = 0; break;
   case VK_FORMAT_X8_D24_UNORM_PACK32: cls = APEX_HW_CLASS_D24, type = APEX_HW_UNORM, read = 0; break;
   case VK_FORMAT_D24_UNORM_S8_UINT:
      cls = stencil ? APEX_HW_CLASS_RGBA8 : APEX_HW_CLASS_D24;
      type = stencil ? APEX_HW_UINT : APEX_HW_UNORM;
      read = stencil ? 3 : 0;
      break;
   case VK_FORMAT_D32_SFLOAT: cls = APEX_HW_CLASS_R32, type = APEX_HW_FLOAT, read = 0; break;
   /* The stencil dword holds S in bits 7:0 and zeros above. */
   case VK_FORMAT_D32_SFLOAT_S8_UINT:
      cls = APEX_HW_CLASS_RG32, type = stencil ? APEX_HW_UINT : APEX_HW_FLOAT, read = stencil;
      break;
   case VK_FORMAT_S8_UINT: cls = APEX_HW_CLASS_R8, type = APEX_HW_UINT, read = 0; break;
   default: {
      const struct format_entry *e = find_format(format);
      if (!e)
         return false;
      cls = e->class, type = e->type, bgr = e->bgr;
      break;
   }
   }
   *code = APEX_HW_FORMAT(cls, type, bgr);
   for (unsigned c = 0; c < 4; c++) {
      if (read >= 0)
         channel[c] = c == 0 ? read : c == 3 ? APEX_HW_SWIZZLE_1 : APEX_HW_SWIZZLE_0;
      else if (c < class_channels[cls])
         channel[c] = c;
      else
         channel[c] = c == 3 ? APEX_HW_SWIZZLE_1 : APEX_HW_SWIZZLE_0;
   }
   return true;
}

uint8_t
apex_hw_color_format(VkFormat format)
{
   const struct format_entry *e = find_format(format);
   return e ? APEX_HW_FORMAT(e->class, e->type, e->bgr) : 0;
}

enum apex_hw_depth
apex_hw_depth_format(VkFormat format)
{
   switch (format) {
   case VK_FORMAT_D16_UNORM: return APEX_HW_D16;
   case VK_FORMAT_X8_D24_UNORM_PACK32: return APEX_HW_X8_D24;
   case VK_FORMAT_D24_UNORM_S8_UINT: return APEX_HW_D24S8;
   case VK_FORMAT_D32_SFLOAT: return APEX_HW_D32F;
   case VK_FORMAT_S8_UINT: return APEX_HW_S8;
   case VK_FORMAT_D32_SFLOAT_S8_UINT: return APEX_HW_D32F_S8;
   default: return APEX_HW_DEPTH_NONE;
   }
}

/* ---- Layout ------------------------------------------------------------- */

/* Block width and height logs by log2 texel bytes. */
static const unsigned block_w[5] = {3, 3, 2, 2, 2}, block_h[5] = {3, 2, 2, 1, 0};

unsigned
apex_hw_tile_width(unsigned bytes)
{
   return 8u << block_w[util_logbase2(bytes)];
}

unsigned
apex_hw_tile_height(unsigned bytes)
{
   return 8u << block_h[util_logbase2(bytes)];
}

void
apex_hw_layout_init(struct apex_hw_layout *layout, unsigned bytes, bool three_d, VkExtent3D extent,
                    unsigned levels, unsigned layers, unsigned samples, bool tiled)
{
   memset(layout, 0, sizeof(*layout));
   layout->tiled = tiled;
   layout->bytes = bytes;
   layout->levels = levels;
   layout->samples = samples;
   layout->layers = layers;
   uint64_t offset = 0;
   for (unsigned l = 0; l < levels; l++) {
      struct apex_hw_level *level = &layout->level[l];
      level->width = u_minify(extent.width, l);
      level->height = u_minify(extent.height, l);
      level->depth = three_d ? u_minify(extent.depth, l) : 1;
      unsigned planes = three_d ? level->depth : samples;
      if (tiled) {
         unsigned tw = apex_hw_tile_width(bytes), th = apex_hw_tile_height(bytes);
         level->pitch = DIV_ROUND_UP(level->width, tw);
         level->plane = (uint64_t)level->pitch * DIV_ROUND_UP(level->height, th) * APEX_HW_TILE;
      } else {
         level->pitch = align(level->width * bytes, 64);
         level->plane = (uint64_t)level->pitch * level->height;
      }
      level->offset = offset;
      offset += level->plane * planes;
   }
   /* Layers start on 4 KiB boundaries. */
   layout->layer_stride = align64(offset, APEX_HW_TILE);
   layout->size = layers > 1 ? layout->layer_stride * layers : offset;
}

uint64_t
apex_hw_texel_offset(const struct apex_hw_layout *layout, unsigned l, unsigned layer, unsigned plane,
                     unsigned x, unsigned y)
{
   const struct apex_hw_level *level = &layout->level[l];
   uint64_t base = layer * layout->layer_stride + level->offset + plane * level->plane;
   if (!layout->tiled)
      return base + (uint64_t)y * level->pitch + (uint64_t)x * layout->bytes;
   unsigned lb = util_logbase2(layout->bytes), bw = block_w[lb], bh = block_h[lb];
   uint64_t page = (uint64_t)(y >> (bh + 3)) * level->pitch + (x >> (bw + 3));
   unsigned bx = (x >> bw) & 7, by = (y >> bh) & 7, block = 0;
   for (unsigned i = 0; i < 3; i++)
      block |= ((bx >> i) & 1) << (2 * i) | ((by >> i) & 1) << (2 * i + 1);
   unsigned sh = MIN2(bh, 2), tx = x & ((1u << bw) - 1), ty = y & ((1u << bh) - 1);
   unsigned within = (((ty >> 2) << (bw - 2)) + (tx >> 2)) * (4u << sh) + (ty & ((1u << sh) - 1)) * 4 +
                     (tx & 3);
   return base + page * APEX_HW_TILE + block * 64 + ((uint64_t)within << lb);
}

/* ---- Descriptors -------------------------------------------------------- */

void
apex_hw_image_descriptor(const struct apex_hw_image *image, uint32_t d[8])
{
   memset(d, 0, 32);
   put(d, 0, 40, image->va);
   put(d, APEX_HW_IMAGE_FORMAT, 8, image->format);
   put(d, APEX_HW_IMAGE_TILED, 1, image->tiled);
   put(d, APEX_HW_IMAGE_SAMPLES, 3, image->samples_log2);
   for (unsigned c = 0; c < 4; c++)
      put(d, APEX_HW_IMAGE_SWIZZLE + 3 * c, 3, image->swizzle[c]);
   put(d, APEX_HW_IMAGE_WIDTH, 14, image->width - 1);
   put(d, APEX_HW_IMAGE_HEIGHT, 14, image->height - 1);
   put(d, APEX_HW_IMAGE_DEPTH, 11, image->depth - 1);
   put(d, APEX_HW_IMAGE_LEVELS, 5, image->levels);
   put(d, APEX_HW_IMAGE_PITCH, 16, image->pitch64);
   put(d, APEX_HW_IMAGE_LAYER_STRIDE, 34, image->stride64);
   put(d, APEX_HW_IMAGE_MIN_LOD, 12, image->min_lod);
}

/* Q(n).8 fixed point, clamped to the field. */
static int32_t
q8(float value, int32_t low, int32_t high)
{
   float scaled = roundf(value * 256.0f);
   return scaled < low ? low : scaled > high ? high : (int32_t)scaled;
}

void
apex_hw_sampler_descriptor(const VkSamplerCreateInfo *info, const struct vk_sampler *sampler,
                           uint32_t d[8])
{
   memset(d, 0, 32);
   put(d, 0, 1, info->magFilter == VK_FILTER_LINEAR);
   put(d, 1, 1, info->minFilter == VK_FILTER_LINEAR);
   put(d, 2, 1, info->mipmapMode == VK_SAMPLER_MIPMAP_MODE_LINEAR);
   const VkSamplerAddressMode modes[3] = {info->addressModeU, info->addressModeV, info->addressModeW};
   for (unsigned a = 0; a < 3; a++)
      put(d, APEX_HW_SAMPLER_ADDRESS + 3 * a, 3, modes[a]);
   put(d, APEX_HW_SAMPLER_COMPARE, 1, info->compareEnable);
   /* VkCompareOp values are the less, equal and greater bit set. */
   put(d, APEX_HW_SAMPLER_COMPARE_OP, 3, info->compareEnable ? info->compareOp : 0);
   put(d, APEX_HW_SAMPLER_REDUCTION, 2, sampler->reduction_mode == VK_SAMPLER_REDUCTION_MODE_MIN ? 1 :
                                        sampler->reduction_mode == VK_SAMPLER_REDUCTION_MODE_MAX ? 2 : 0);
   put(d, APEX_HW_SAMPLER_UNNORMALIZED, 1, info->unnormalizedCoordinates);
   put(d, APEX_HW_SAMPLER_BIAS, 14, (uint32_t)q8(info->mipLodBias, -8192, 8191) & 0x3fff);
   put(d, APEX_HW_SAMPLER_MIN_LOD, 12, q8(info->minLod, 0, 4095));
   put(d, APEX_HW_SAMPLER_MAX_LOD, 12, q8(info->maxLod, 0, 4095));
   /* Every border is the custom RGBA of the resolved Vulkan border color:
    * binary32 for float borders, integers for integer ones. */
   put(d, APEX_HW_SAMPLER_BORDER, 3, 3);
   memcpy(&d[4], sampler->border_color_value.uint32, 16);
}

/* ---- State blocks ------------------------------------------------------- */

enum vertex_kind {
   KIND_NONE, KIND_FLOAT32, KIND_UINT32, KIND_SINT32, KIND_UNORM8, KIND_SNORM8, KIND_UINT8,
   KIND_SINT8, KIND_UNORM16, KIND_SNORM16, KIND_UINT16, KIND_SINT16, KIND_FLOAT16,
   KIND_UNORM10_2, KIND_UINT10_2, KIND_USCALED8, KIND_SSCALED8, KIND_USCALED16, KIND_SSCALED16,
};

uint8_t
apex_hw_vertex_format(VkFormat format, bool *swap_rb)
{
   static const struct { VkFormat base; unsigned kind; } families[] = {
      {VK_FORMAT_R8_UNORM, KIND_UNORM8}, {VK_FORMAT_R8_SNORM, KIND_SNORM8},
      {VK_FORMAT_R8_USCALED, KIND_USCALED8}, {VK_FORMAT_R8_SSCALED, KIND_SSCALED8},
      {VK_FORMAT_R8_UINT, KIND_UINT8}, {VK_FORMAT_R8_SINT, KIND_SINT8},
      {VK_FORMAT_R16_UNORM, KIND_UNORM16}, {VK_FORMAT_R16_SNORM, KIND_SNORM16},
      {VK_FORMAT_R16_USCALED, KIND_USCALED16}, {VK_FORMAT_R16_SSCALED, KIND_SSCALED16},
      {VK_FORMAT_R16_UINT, KIND_UINT16}, {VK_FORMAT_R16_SINT, KIND_SINT16},
      {VK_FORMAT_R16_SFLOAT, KIND_FLOAT16},
   };
   *swap_rb = false;
   /* R, RG, RGB and RGBA families of 8 and 16 bits: seven enumerants per
    * family (UNORM, SNORM, USCALED, SSCALED, UINT, SINT, SRGB or SFLOAT);
    * B8G8R8 sits between R8G8B8 and R8G8B8A8. */
   static const unsigned eight[4] = {0, 7, 14, 28}, sixteen[4] = {0, 7, 14, 21};
   for (unsigned f = 0; f < ARRAY_SIZE(families); f++) {
      const unsigned *offsets = families[f].base < VK_FORMAT_R16_UNORM ? eight : sixteen;
      for (unsigned n = 0; n < 4; n++)
         if (families[f].base + offsets[n] == format)
            return families[f].kind << 2 | n;
   }
   switch (format) {
   case VK_FORMAT_R32_UINT: return KIND_UINT32 << 2;
   case VK_FORMAT_R32_SINT: return KIND_SINT32 << 2;
   case VK_FORMAT_R32_SFLOAT: return KIND_FLOAT32 << 2;
   case VK_FORMAT_R32G32_UINT: return KIND_UINT32 << 2 | 1;
   case VK_FORMAT_R32G32_SINT: return KIND_SINT32 << 2 | 1;
   case VK_FORMAT_R32G32_SFLOAT: return KIND_FLOAT32 << 2 | 1;
   case VK_FORMAT_R32G32B32_UINT: return KIND_UINT32 << 2 | 2;
   case VK_FORMAT_R32G32B32_SINT: return KIND_SINT32 << 2 | 2;
   case VK_FORMAT_R32G32B32_SFLOAT: return KIND_FLOAT32 << 2 | 2;
   case VK_FORMAT_R32G32B32A32_UINT: return KIND_UINT32 << 2 | 3;
   case VK_FORMAT_R32G32B32A32_SINT: return KIND_SINT32 << 2 | 3;
   case VK_FORMAT_R32G32B32A32_SFLOAT: return KIND_FLOAT32 << 2 | 3;
   case VK_FORMAT_A8B8G8R8_UNORM_PACK32: return KIND_UNORM8 << 2 | 3;
   case VK_FORMAT_A8B8G8R8_SNORM_PACK32: return KIND_SNORM8 << 2 | 3;
   case VK_FORMAT_A8B8G8R8_USCALED_PACK32: return KIND_USCALED8 << 2 | 3;
   case VK_FORMAT_A8B8G8R8_SSCALED_PACK32: return KIND_SSCALED8 << 2 | 3;
   case VK_FORMAT_A8B8G8R8_UINT_PACK32: return KIND_UINT8 << 2 | 3;
   case VK_FORMAT_A8B8G8R8_SINT_PACK32: return KIND_SINT8 << 2 | 3;
   case VK_FORMAT_B8G8R8A8_UNORM: *swap_rb = true; return KIND_UNORM8 << 2 | 3;
   case VK_FORMAT_B8G8R8A8_SNORM: *swap_rb = true; return KIND_SNORM8 << 2 | 3;
   case VK_FORMAT_B8G8R8A8_UINT: *swap_rb = true; return KIND_UINT8 << 2 | 3;
   case VK_FORMAT_B8G8R8A8_SINT: *swap_rb = true; return KIND_SINT8 << 2 | 3;
   case VK_FORMAT_A2B10G10R10_UNORM_PACK32: return KIND_UNORM10_2 << 2 | 3;
   case VK_FORMAT_A2B10G10R10_UINT_PACK32: return KIND_UINT10_2 << 2 | 3;
   case VK_FORMAT_A2R10G10B10_UNORM_PACK32: *swap_rb = true; return KIND_UNORM10_2 << 2 | 3;
   case VK_FORMAT_A2R10G10B10_UINT_PACK32: *swap_rb = true; return KIND_UINT10_2 << 2 | 3;
   default: return 0;
   }
}

void
apex_hw_vertex_input_block(const struct vk_dynamic_graphics_state *dyn,
                           const struct apex_hw_binding bindings[APEX_HW_MAX_BINDINGS],
                           unsigned clip_distances, uint32_t out[APEX_HW_VERTEX_INPUT_DWORDS])
{
   memset(out, 0, APEX_HW_VERTEX_INPUT_DWORDS * 4);
   const struct vk_vertex_input_state *vi = dyn->vi;
   for (unsigned b = 0; b < APEX_HW_MAX_BINDINGS; b++) {
      bool instance = vi && (vi->bindings_valid & BITFIELD_BIT(b)) &&
                      vi->bindings[b].input_rate == VK_VERTEX_INPUT_RATE_INSTANCE;
      uint32_t *w = &out[4 * b];
      w[0] = (uint32_t)bindings[b].va;
      w[1] = ((uint32_t)(bindings[b].va >> 32) & 0xff) | (uint32_t)instance << 8 |
             (uint32_t)MIN2(dyn->vi_binding_strides[b], 0xffff) << 16;
      w[2] = (uint32_t)MIN2(bindings[b].size, UINT32_MAX);
      w[3] = instance ? vi->bindings[b].divisor : 0;
   }
   u_foreach_bit(a, vi ? vi->attributes_valid : 0) {
      if (a >= APEX_HW_MAX_ATTRIBUTES)
         break;
      bool swap;
      uint32_t *w = &out[64 + 2 * a];
      w[0] = vi->attributes[a].binding | apex_hw_vertex_format(vi->attributes[a].format, &swap) << 8 |
             (vi->attributes[a].offset & 0xffff) << 16;
      w[1] = APEX_HW_ATTRIBUTE_REGISTER(a);
   }
   out[96] = clip_distances & 15;
}

void
apex_hw_raster_block(const struct vk_dynamic_graphics_state *dyn, bool sample_shading, bool late_depth,
                     uint32_t out[APEX_HW_RASTER_DWORDS])
{
   memset(out, 0, APEX_HW_RASTER_DWORDS * 4);
   const struct vk_rasterization_state *rs = &dyn->rs;
   /* Line modes: rectangular 0, Bresenham 1, smooth 2. */
   uint32_t line = rs->line.mode == VK_LINE_RASTERIZATION_MODE_BRESENHAM_KHR ? 1 :
                   rs->line.mode == VK_LINE_RASTERIZATION_MODE_RECTANGULAR_SMOOTH_KHR ? 2 : 0;
   uint32_t samples = MAX2(dyn->ms.rasterization_samples, 1);
   out[0] = (rs->cull_mode & 3) |
            (uint32_t)(rs->front_face == VK_FRONT_FACE_COUNTER_CLOCKWISE) << 2 |
            (MIN2(rs->polygon_mode, 2) & 3) << 3 | line << 5 |
            (uint32_t)(rs->provoking_vertex == VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT) << 7 |
            (uint32_t)rs->depth_bias.enable << 8 |
            (uint32_t)vk_rasterization_state_depth_clip_enable(rs) << 9 |
            (uint32_t)rs->rasterizer_discard_enable << 10 | util_logbase2(samples) << 11 |
            (uint32_t)dyn->ms.alpha_to_coverage_enable << 13 |
            (uint32_t)dyn->ms.alpha_to_one_enable << 14 | (uint32_t)sample_shading << 15 |
            (uint32_t)late_depth << 16;
   out[1] = dyn->ms.sample_mask & BITFIELD_MASK(samples);
}

void
apex_hw_depth_stencil_block(const struct vk_dynamic_graphics_state *dyn, VkFormat depth_format,
                            bool has_depth, bool has_stencil, uint32_t out[APEX_HW_DEPTH_STENCIL_DWORDS])
{
   memset(out, 0, APEX_HW_DEPTH_STENCIL_DWORDS * 4);
   const struct vk_depth_stencil_state *ds = &dyn->ds;
   bool test = has_depth && ds->depth.test_enable;
   bool stencil = has_stencil && ds->stencil.test_enable;
   const struct vk_stencil_test_face_state *faces[2] = {&ds->stencil.front, &ds->stencil.back};
   bool keep = true;
   for (unsigned f = 0; f < 2; f++)
      keep &= faces[f]->op.fail == VK_STENCIL_OP_KEEP && faces[f]->op.depth_fail == VK_STENCIL_OP_KEEP;
   bool fixed = depth_format == VK_FORMAT_D16_UNORM || depth_format == VK_FORMAT_X8_D24_UNORM_PACK32 ||
                depth_format == VK_FORMAT_D24_UNORM_S8_UINT;
   out[0] = (uint32_t)test | (uint32_t)(test && ds->depth.write_enable) << 1 |
            (test ? ds->depth.compare_op & 7 : 0) << 2 |
            (uint32_t)(has_depth && ds->depth.bounds_test.enable) << 5 |
            (uint32_t)dyn->rs.depth_clamp_enable << 6 | (uint32_t)stencil << 7 |
            (uint32_t)(!stencil || keep) << 8 | (uint32_t)(has_depth && !fixed) << 9;
   for (unsigned f = 0; f < 2; f++) {
      const struct vk_stencil_test_face_state *face = faces[f];
      uint32_t ops = (face->op.fail & 7) | (face->op.pass & 7) << 3 | (face->op.depth_fail & 7) << 6 |
                     (face->op.compare & 7) << 9;
      out[1] |= ops << (16 * f);
   }
   /* r: the minimum resolvable difference of a fixed-point depth format. */
   unsigned bits = depth_format == VK_FORMAT_D16_UNORM ? 16 : 24;
   out[2] = has_depth && fixed ? float_bits(ldexpf(1.0f, -(int)bits)) : 0;
}

void
apex_hw_blend_block(const struct vk_dynamic_graphics_state *dyn, unsigned attachments,
                    uint32_t out[APEX_HW_BLEND_DWORDS])
{
   memset(out, 0, APEX_HW_BLEND_DWORDS * 4);
   const struct vk_color_blend_state *cb = &dyn->cb;
   out[0] = (uint32_t)cb->logic_op_enable | (cb->logic_op & 15) << 1;
   for (unsigned a = 0; a < MIN2(attachments, APEX_HW_MAX_COLOR); a++) {
      const struct vk_color_blend_attachment_state *s = &cb->attachments[a];
      uint32_t mask = cb->color_write_enables & BITFIELD_BIT(a) ? s->write_mask & 15 : 0;
      /* A logic op disables blending of every attachment. */
      bool enable = s->blend_enable && !cb->logic_op_enable;
      out[1 + a] = mask | (s->dst_alpha_blend_factor & 31) << 4 | (s->src_alpha_blend_factor & 31) << 9 |
                   (s->alpha_blend_op & 7) << 14 | (s->dst_color_blend_factor & 31) << 17 |
                   (s->src_color_blend_factor & 31) << 22 | (s->color_blend_op & 7) << 27 |
                   (uint32_t)enable << 30;
   }
}

unsigned
apex_hw_guard_band(float origin, float extent)
{
   /* screen = ndc * extent / 2 + origin + extent / 2 for ndc in [-g, g]. */
   double half = fabs(extent) / 2, centre = fabs((double)origin + extent / 2.0);
   if (half == 0)
      return 30;
   double g = (65536.0 - centre) / half;
   if (g < 2)
      return 0;
   int k = (int)floor(log2(g));
   return MIN2(k, 30);
}

void
apex_hw_viewport_block(const struct vk_dynamic_graphics_state *dyn, uint32_t out[APEX_HW_VIEWPORT_DWORDS])
{
   memset(out, 0, APEX_HW_VIEWPORT_DWORDS * 4);
   const struct vk_viewport_state *vp = &dyn->vp;
   for (unsigned v = 0; v < MIN2(vp->viewport_count, APEX_HW_MAX_VIEWPORTS); v++) {
      const VkViewport *view = &vp->viewports[v];
      const float values[6] = {view->x, view->y, view->width, view->height, view->minDepth, view->maxDepth};
      for (unsigned i = 0; i < 6; i++)
         out[8 * v + i] = float_bits(values[i]);
      out[8 * v + 6] = apex_hw_guard_band(view->x, view->width);
      out[8 * v + 7] = apex_hw_guard_band(view->y, view->height);
   }
   for (unsigned v = 0; v < MIN2(vp->scissor_count, APEX_HW_MAX_VIEWPORTS); v++) {
      const VkRect2D *s = &vp->scissors[v];
      int64_t x0 = CLAMP(s->offset.x, 0, 0xffff), y0 = CLAMP(s->offset.y, 0, 0xffff);
      int64_t x1 = CLAMP((int64_t)s->offset.x + s->extent.width, x0, 0xffff);
      int64_t y1 = CLAMP((int64_t)s->offset.y + s->extent.height, y0, 0xffff);
      out[128 + 2 * v] = (uint32_t)x0 | (uint32_t)y0 << 16;
      out[129 + 2 * v] = (uint32_t)x1 | (uint32_t)y1 << 16;
   }
}

void
apex_hw_dynamic_registers(const struct vk_dynamic_graphics_state *dyn, uint32_t out[APEX_HW_DYNAMIC_DWORDS])
{
   const struct vk_stencil_test_face_state *faces[2] = {&dyn->ds.stencil.front, &dyn->ds.stencil.back};
   for (unsigned f = 0; f < 2; f++)
      out[f] = (faces[f]->reference & 0xff) | (faces[f]->compare_mask & 0xff) << 8 |
               (faces[f]->write_mask & 0xff) << 16;
   for (unsigned c = 0; c < 4; c++)
      out[2 + c] = float_bits(dyn->cb.blend_constants[c]);
   out[6] = float_bits(dyn->rs.depth_bias.constant_factor);
   out[7] = float_bits(dyn->rs.depth_bias.slope_factor);
   out[8] = float_bits(dyn->rs.depth_bias.clamp);
   out[9] = float_bits(dyn->ds.depth.bounds_test.min);
   out[10] = float_bits(dyn->ds.depth.bounds_test.max);
   out[11] = float_bits(dyn->rs.line.width);
   out[12] = MIN2(dyn->vp.viewport_count, APEX_HW_MAX_VIEWPORTS);
}

/* ---- Pass record -------------------------------------------------------- */

static const unsigned selector_w[5] = {64, 64, 32, 32, 16}, selector_h[5] = {64, 32, 32, 16, 16};

unsigned
apex_hw_selector_width(unsigned selector)
{
   return selector_w[MIN2(selector, 4)];
}

unsigned
apex_hw_selector_height(unsigned selector)
{
   return selector_h[MIN2(selector, 4)];
}

int
apex_hw_tile_selector(unsigned samples, unsigned planes)
{
   if (planes > APEX_HW_TILE_PLANES)
      return -1;
   for (unsigned s = 0; s < 5; s++)
      if (selector_w[s] * selector_h[s] * samples * MAX2(planes, 1) <= 65536)
         return s;
   return -1;
}

static void
surface_words(const struct apex_hw_surface *s, uint32_t w[3])
{
   w[0] = (uint32_t)s->va;
   w[1] = ((uint32_t)(s->va >> 32) & 0xff) | (s->pitch & 0xffff) << 16;
   w[2] = s->layer_stride;
}

bool
apex_hw_pass_record(struct apex_hw_pass *pass, uint32_t out[APEX_HW_PASS_DWORDS])
{
   memset(out, 0, APEX_HW_PASS_DWORDS * 4);
   /* Planes: color attachments in order, 16-byte texels on even planes,
    * then the depth-stencil attachment. */
   unsigned plane = 0;
   for (unsigned k = 0; k <= APEX_HW_MAX_COLOR; k++) {
      struct apex_hw_attachment *a = &pass->attachment[k];
      if (!a->present)
         continue;
      if (a->planes == 2)
         plane = align(plane, 2);
      uint32_t *w = &out[APEX_HW_PASS_ATTACHMENT(k)];
      w[0] = 1 | (uint32_t)a->load << 1 | (uint32_t)!a->store << 3 | (uint32_t)a->resolve << 4 |
             (uint32_t)a->surface.tiled << 5 | (uint32_t)(a->resolve && a->resolve_surface.tiled) << 6 |
             (plane & 15) << 8 | (uint32_t)a->format << 16;
      surface_words(&a->surface, &w[1]);
      if (k == APEX_HW_DEPTH_ATTACHMENT) {
         w[4] = float_bits(a->clear_depth);
         w[6] = a->clear[0];
         w[7] = a->clear[1];
      } else {
         memcpy(&w[4], a->clear, 16);
      }
      w[8] = a->surface.sample_stride;
      if (a->resolve)
         surface_words(&a->resolve_surface, &w[9]);
      plane += a->planes;
   }
   int selector = apex_hw_tile_selector(pass->samples, plane);
   if (selector < 0)
      return false;
   out[0] = (uint32_t)pass->pool;
   out[1] = (uint32_t)(pass->pool >> 32);
   out[2] = (uint32_t)pass->pool_bytes;
   out[3] = pass->x0 | pass->y0 << 16;
   out[4] = pass->x1 | pass->y1 << 16;
   out[5] = pass->layers;
   out[7] = selector | util_logbase2(MAX2(pass->samples, 1)) << 4;
   out[8] = pass->draw_bytes;
   out[9] = pass->vertex_bytes;
   out[10] = pass->primitive_bytes;
   return apex_hw_pass_valid(out);
}

bool
apex_hw_color_renderable(uint8_t format)
{
   unsigned cls = format & 15, type = format >> 4 & 7;
   if (cls == APEX_HW_CLASS_1555)
      return type == APEX_HW_UNORM || type == APEX_HW_ALPHA_FIRST;
   if (cls == APEX_HW_CLASS_NONE || cls == APEX_HW_CLASS_D24 || cls == APEX_HW_CLASS_4444 ||
       type >= APEX_HW_ALPHA_FIRST)
      return false;
   if (type == APEX_HW_SRGB)
      return cls <= APEX_HW_CLASS_RGBA8 || cls == APEX_HW_CLASS_RGBX8;
   if (type == APEX_HW_FLOAT)
      return cls >= APEX_HW_CLASS_R16 && cls <= APEX_HW_CLASS_RGBA32;
   if (type == APEX_HW_SNORM)
      return cls < APEX_HW_CLASS_R32 || cls > APEX_HW_CLASS_RGB10A2;
   return true;
}

bool
apex_hw_pass_valid(const uint32_t r[APEX_HW_PASS_DWORDS])
{
   unsigned selector = r[7] & 7, samples_log2 = r[7] >> 4 & 3;
   uint32_t x0 = r[3] & 0xffff, y0 = r[3] >> 16, x1 = r[4] & 0xffff, y1 = r[4] >> 16;
   if (selector > 4 || x1 <= x0 || y1 <= y0 || !r[5] || r[5] > 2048)
      return false;
   unsigned tw = selector_w[selector], th = selector_h[selector];
   uint64_t bins = (uint64_t)((x1 - 1) / tw - x0 / tw + 1) * ((y1 - 1) / th - y0 / th + 1) * r[5];
   if (bins > APEX_HW_MAX_BINS)
      return false;
   /* Attachment formats and planes: a 16-byte target takes an even plane
    * pair; the planes end within the tile buffer at the tile size and
    * samples. */
   unsigned planes = 0;
   for (unsigned k = 0; k <= APEX_HW_MAX_COLOR; k++) {
      uint32_t flags = r[APEX_HW_PASS_ATTACHMENT(k)];
      if (!(flags & 1))
         continue;
      unsigned plane = flags >> 8 & 15, format = flags >> 16 & 0xff, end = plane + 1;
      if (k == APEX_HW_DEPTH_ATTACHMENT) {
         if ((format & 7) == APEX_HW_DEPTH_NONE || (format & 7) > APEX_HW_D32F_S8)
            return false;
      } else {
         if (!apex_hw_color_renderable(format))
            return false;
         if ((format & 15) == APEX_HW_CLASS_RGBA32) {
            if (plane & 1)
               return false;
            end = plane + 2;
         }
      }
      planes = MAX2(planes, end);
   }
   if (((uint64_t)planes << (util_logbase2(tw) + util_logbase2(th) + samples_log2)) > 65536)
      return false;
   /* The regions in pool order, each inside the pool with room for one bin
    * chunk: the 512-byte header, the draw slots, the bin heads, the vertex
    * outputs and the primitive records. */
   uint64_t chunks = 512 + (uint64_t)r[8] + align64(bins * 4, 64) + r[9] + r[10];
   return r[8] >= 1024 && r[10] >= 128 && chunks + 64 <= r[2];
}
