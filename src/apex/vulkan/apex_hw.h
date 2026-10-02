/* SPDX-License-Identifier: MIT */
/* The device memory formats the driver writes for the M4 machine
 * (Docs/architecture.md, Fixed-function graphics): texture-unit image and
 * sampler descriptors, the tiled layout the ROP and texture unit share,
 * the immutable state blocks, the dynamic registers and the pass record
 * with its attachment descriptors. Every word is little-endian. */
#ifndef APEX_HW_H
#define APEX_HW_H
#include <stdbool.h>
#include <stdint.h>
#include <vulkan/vulkan_core.h>

struct vk_dynamic_graphics_state;
struct vk_sampler;

/* Texture-unit format byte: class 3:0, number type 6:4, BGR order 7. Color
 * attachments add class 13, 4-byte RGBX8 without alpha. */
enum apex_hw_class {
   APEX_HW_CLASS_NONE, APEX_HW_CLASS_R8, APEX_HW_CLASS_RG8, APEX_HW_CLASS_RGBA8,
   APEX_HW_CLASS_R16, APEX_HW_CLASS_RG16, APEX_HW_CLASS_RGBA16, APEX_HW_CLASS_R32,
   APEX_HW_CLASS_RG32, APEX_HW_CLASS_RGBA32, APEX_HW_CLASS_565, APEX_HW_CLASS_RGB10A2,
   APEX_HW_CLASS_D24, APEX_HW_CLASS_RGBX8, APEX_HW_CLASS_4444, APEX_HW_CLASS_1555,
};
/* The packed 16-bit classes are UNORM with alpha in the low field (type 0)
 * or the high field (APEX_HW_ALPHA_FIRST). */
enum apex_hw_type {
   APEX_HW_UNORM, APEX_HW_SNORM, APEX_HW_UINT, APEX_HW_SINT, APEX_HW_FLOAT, APEX_HW_SRGB,
   APEX_HW_ALPHA_FIRST, APEX_HW_SHARED_EXPONENT,
};
#define APEX_HW_FORMAT(class, type, bgr) ((class) | (type) << 4 | (bgr) << 7)
#define APEX_HW_SWIZZLE_0 4
#define APEX_HW_SWIZZLE_1 5

/* Texture view of `format` reading `aspect`: the format byte and, per API
 * component, the stored channel or APEX_HW_SWIZZLE_0/1 before the view's
 * component mapping. False when the texture unit cannot read it. */
bool apex_hw_texture_format(VkFormat format, VkImageAspectFlags aspect, uint8_t *code,
                            uint8_t channel[4]);
/* Color-attachment format byte, 0 when the ROP cannot render the format. */
uint8_t apex_hw_color_format(VkFormat format);
/* Whether the ROP renders a color format byte: classes 1-11 and 13 with a
 * number type the class takes, and 1555 (15) UNORM in either alpha order. */
bool apex_hw_color_renderable(uint8_t format);
/* DepthFormat of a depth/stencil attachment, 0 when unsupported. */
enum apex_hw_depth {
   APEX_HW_DEPTH_NONE, APEX_HW_D16, APEX_HW_X8_D24, APEX_HW_D24S8, APEX_HW_D32F, APEX_HW_S8,
   APEX_HW_D32F_S8,
};
enum apex_hw_depth apex_hw_depth_format(VkFormat format);

/* Image layout. Tiled levels are row-major grids of 4 KiB tiles of 8 x 8
 * Morton-ordered 64-byte blocks of 4 x 4 texels at 32 bits, 8 x 4 at 16,
 * 8 x 8 at 8, 4 x 2 at 64 and 4 x 1 at 128, texels in 4 x 4 sub-blocks left
 * to right then top to bottom, row-major within each. Levels follow each
 * other in whole tiles, each array layer holds its whole mip chain and the
 * samples or 3D slices of a level are consecutive planes. Linear images are
 * 2D, single-level and single-sampled with 64-byte aligned rows. */
#define APEX_HW_TILE 4096u
#define APEX_HW_MAX_LEVELS 15
struct apex_hw_level {
   uint64_t offset;      /* bytes from the layer's start */
   uint64_t plane;       /* bytes per sample plane or 3D slice */
   uint32_t pitch;       /* row bytes (linear) or tiles per row (tiled) */
   uint32_t width, height, depth;
};
struct apex_hw_layout {
   bool tiled;
   uint32_t bytes;       /* texel bytes, or block bytes of compressed formats */
   uint32_t levels, samples, layers;
   uint64_t layer_stride, size;
   struct apex_hw_level level[APEX_HW_MAX_LEVELS];
};
/* Tile width and height in texels of a texel size (1, 2, 4, 8, 16 bytes). */
unsigned apex_hw_tile_width(unsigned bytes);
unsigned apex_hw_tile_height(unsigned bytes);
/* `extent` in texels (blocks for compressed formats); 3D images keep their
 * slices as planes. */
void apex_hw_layout_init(struct apex_hw_layout *layout, unsigned bytes, bool three_d, VkExtent3D extent,
                         unsigned levels, unsigned layers, unsigned samples, bool tiled);
/* Byte offset of texel (x, y) of `plane` (sample or 3D slice) of `level` in
 * `layer`. */
uint64_t apex_hw_texel_offset(const struct apex_hw_layout *layout, unsigned level, unsigned layer,
                              unsigned plane, unsigned x, unsigned y);

/* Texture-unit image descriptor (32 bytes) bit offsets. */
#define APEX_HW_IMAGE_FORMAT 40
#define APEX_HW_IMAGE_TILED 48
#define APEX_HW_IMAGE_SAMPLES 52         /* log2 */
#define APEX_HW_IMAGE_SWIZZLE 55         /* four 3-bit selects */
#define APEX_HW_IMAGE_WIDTH 67           /* minus one, 14 bits */
#define APEX_HW_IMAGE_HEIGHT 81          /* minus one, 14 bits */
#define APEX_HW_IMAGE_DEPTH 95           /* depth or layers minus one, 11 bits */
#define APEX_HW_IMAGE_LEVELS 106         /* level count, 5 bits; zero for null descriptors */
#define APEX_HW_IMAGE_PITCH 111          /* linear row pitch, 64-byte units, 16 bits */
#define APEX_HW_IMAGE_LAYER_STRIDE 127   /* 64-byte units, 34 bits */
#define APEX_HW_IMAGE_MIN_LOD 161        /* Q4.8, 12 bits */
/* Sampler descriptor (32 bytes) bit offsets. */
#define APEX_HW_SAMPLER_ADDRESS 3        /* three 3-bit VkSamplerAddressMode */
#define APEX_HW_SAMPLER_COMPARE 12
#define APEX_HW_SAMPLER_COMPARE_OP 13    /* VkCompareOp as less, equal, greater bits */
#define APEX_HW_SAMPLER_REDUCTION 16
#define APEX_HW_SAMPLER_UNNORMALIZED 18
#define APEX_HW_SAMPLER_BIAS 23          /* Q5.8 signed, 14 bits */
#define APEX_HW_SAMPLER_MIN_LOD 37       /* Q4.8, 12 bits */
#define APEX_HW_SAMPLER_MAX_LOD 49       /* Q4.8, 12 bits */
#define APEX_HW_SAMPLER_BORDER 61        /* 3 = custom RGBA at bits 255:128 */

/* One image view as the texture unit reads it: the view's base level and
 * layer are folded into `va`, so level 0 of the descriptor is the view's
 * base level. */
struct apex_hw_image {
   uint64_t va;
   uint8_t format;
   bool tiled;
   uint8_t samples_log2;
   uint8_t swizzle[4];
   uint32_t width, height, depth, levels;
   uint32_t pitch64;
   uint64_t stride64;
   uint32_t min_lod;     /* Q4.8 */
};
void apex_hw_image_descriptor(const struct apex_hw_image *image, uint32_t d[8]);
void apex_hw_sampler_descriptor(const VkSamplerCreateInfo *info, const struct vk_sampler *sampler,
                                uint32_t d[8]);

/* Vertex attribute format byte {kind 6:2, components - 1 1:0}, 0 when vertex
 * fetch cannot read the format. `swap_rb` reports BGR component order, which
 * the vertex program swizzles. */
uint8_t apex_hw_vertex_format(VkFormat format, bool *swap_rb);

/* State blocks. */
#define APEX_HW_VERTEX_INPUT_DWORDS 128
#define APEX_HW_RASTER_DWORDS 16
#define APEX_HW_DEPTH_STENCIL_DWORDS 16
#define APEX_HW_BLEND_DWORDS 64
#define APEX_HW_VIEWPORT_DWORDS 256
#define APEX_HW_DYNAMIC_DWORDS 13
#define APEX_HW_MAX_BINDINGS 16
#define APEX_HW_MAX_ATTRIBUTES 16
#define APEX_HW_MAX_VIEWPORTS 16
#define APEX_HW_MAX_COLOR 8
/* Vertex attribute location L lands at vector registers 2 + 4L .. 5 + 4L. */
#define APEX_HW_ATTRIBUTE_REGISTER(location) (2 + 4 * (location))

struct apex_hw_binding {
   uint64_t va, size;
};
/* Vertex input: bindings and strides, then the attributes of the dynamic
 * vertex input state. */
void apex_hw_vertex_input_block(const struct vk_dynamic_graphics_state *dyn,
                                const struct apex_hw_binding bindings[APEX_HW_MAX_BINDINGS],
                                unsigned clip_distances, uint32_t out[APEX_HW_VERTEX_INPUT_DWORDS]);
/* `late_depth` when the fragment program tests depth at commit. */
void apex_hw_raster_block(const struct vk_dynamic_graphics_state *dyn, bool sample_shading,
                          bool late_depth, uint32_t out[APEX_HW_RASTER_DWORDS]);
void apex_hw_depth_stencil_block(const struct vk_dynamic_graphics_state *dyn, VkFormat depth_format,
                                 bool has_depth, bool has_stencil,
                                 uint32_t out[APEX_HW_DEPTH_STENCIL_DWORDS]);
void apex_hw_blend_block(const struct vk_dynamic_graphics_state *dyn, unsigned attachments,
                         uint32_t out[APEX_HW_BLEND_DWORDS]);
/* Guard-band exponent k: positions within 2^k of the viewport's clip
 * volume map within the snapped +-2^16 pixel range. */
unsigned apex_hw_guard_band(float origin, float extent);
void apex_hw_viewport_block(const struct vk_dynamic_graphics_state *dyn,
                            uint32_t out[APEX_HW_VIEWPORT_DWORDS]);
void apex_hw_dynamic_registers(const struct vk_dynamic_graphics_state *dyn,
                               uint32_t out[APEX_HW_DYNAMIC_DWORDS]);

/* Pass record: 128 dwords, the pass registers 0x080-0x0ff. */
#define APEX_HW_PASS_DWORDS 128
#define APEX_HW_PASS_ATTACHMENT(k) (16 + 12 * (k))
#define APEX_HW_DEPTH_ATTACHMENT 8
#define APEX_HW_TILE_PLANES 16
#define APEX_HW_MAX_BINS 4096u
enum apex_hw_load { APEX_HW_LOAD, APEX_HW_CLEAR, APEX_HW_LOAD_DONT_CARE };
/* One attachment surface: a level of an image at its first layer. */
struct apex_hw_surface {
   uint64_t va;
   bool tiled;
   uint32_t pitch;       /* 64-byte units (linear) or tiles per row (tiled) */
   uint32_t layer_stride, sample_stride; /* 64-byte units */
};
struct apex_hw_attachment {
   bool present, store, resolve;
   enum apex_hw_load load;
   uint8_t format;       /* color format byte or apex_hw_depth */
   unsigned planes;      /* tile-buffer planes */
   uint32_t clear[4];    /* texel in memory layout */
   float clear_depth;    /* depth-stencil attachment */
   struct apex_hw_surface surface, resolve_surface;
};
struct apex_hw_pass {
   uint64_t pool, pool_bytes;
   uint32_t x0, y0, x1, y1;   /* render area, exclusive end */
   uint32_t layers, samples;
   uint32_t draw_bytes, vertex_bytes, primitive_bytes;
   struct apex_hw_attachment attachment[APEX_HW_MAX_COLOR + 1];
};
/* Tile size selector of a pass of `planes` tile-buffer planes, or -1 when
 * no tile fits. */
int apex_hw_tile_selector(unsigned samples, unsigned planes);
unsigned apex_hw_selector_width(unsigned selector);
unsigned apex_hw_selector_height(unsigned selector);
/* Writes the pass record; false when the raster back end would reject it
 * (apex_hw_pass_valid). */
bool apex_hw_pass_record(struct apex_hw_pass *pass, uint32_t out[APEX_HW_PASS_DWORDS]);
/* The raster back end's pass validation, which reports INVALID_PASS: a
 * tile-size selector of at most 4, a non-empty render area, 1 to 2,048
 * layers and at most APEX_HW_MAX_BINS bins, colour formats the ROP renders,
 * depth formats 1-6, planes that fit the tile buffer at the tile size and
 * samples, a draw region of at least 1 KiB, a primitive region of at least
 * 128 bytes and every region with one bin chunk inside the pool. */
bool apex_hw_pass_valid(const uint32_t record[APEX_HW_PASS_DWORDS]);
#endif
