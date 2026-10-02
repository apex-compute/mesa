/* SPDX-License-Identifier: MIT */
/* Offline gate of the M4 memory formats the driver writes (apex_hw.h):
 * every state block, the dynamic registers, the pass record and its
 * attachment descriptors, texture-unit image and sampler descriptors and
 * the tiled layout, each against the bit positions of
 * Docs/architecture.md (Fixed-function graphics) written out here, and the
 * internal copy kernel run on the ISA model (Tooling/apex_isa) through the
 * tiled layout. */
#include "apex_device.h"
#include "apex_graphics.h"
#include "apex_hw.h"
#include "apex_job.h"
#include "vk_alloc.h"
#include "vk_device.h"
#include "vk_graphics_state.h"
#include "vk_instance.h"
#include "vk_physical_device.h"
#include "vk_sampler.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { \
   fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); abort(); \
} } while (0)

static uint32_t
f2u(float f)
{
   uint32_t u;
   memcpy(&u, &f, 4);
   return u;
}

/* Bits [lo, lo + width) of a little-endian word array. */
static uint64_t
field(const uint32_t *w, unsigned lo, unsigned width)
{
   uint64_t v = 0;
   for (unsigned i = 0; i < width; i++)
      v |= (uint64_t)((w[(lo + i) / 32] >> ((lo + i) % 32)) & 1) << i;
   return v;
}

static void
test_vertex_input(void)
{
   struct vk_vertex_input_state vi = {
      .bindings_valid = 0x5, .attributes_valid = 0x8003,
      .bindings[0] = {.input_rate = VK_VERTEX_INPUT_RATE_VERTEX},
      .bindings[2] = {.input_rate = VK_VERTEX_INPUT_RATE_INSTANCE, .divisor = 7},
      .attributes[0] = {.binding = 0, .format = VK_FORMAT_R32G32B32_SFLOAT, .offset = 12},
      .attributes[1] = {.binding = 2, .format = VK_FORMAT_R8G8B8A8_UNORM, .offset = 0x1234},
      .attributes[15] = {.binding = 0, .format = VK_FORMAT_R16G16_SINT, .offset = 4},
   };
   struct vk_dynamic_graphics_state dyn = {.vi = &vi};
   dyn.vi_binding_strides[0] = 28;
   dyn.vi_binding_strides[2] = 0x1000;
   struct apex_hw_binding bindings[APEX_HW_MAX_BINDINGS] = {
      [0] = {0x12345678c0ull, 4096}, [2] = {0xab00001000ull, 0x10000},
   };
   uint32_t w[APEX_HW_VERTEX_INPUT_DWORDS];
   apex_hw_vertex_input_block(&dyn, bindings, 5, w);
   /* Binding b at dword 4b: GPUVA 31:0; GPUVA 39:32 in 7:0, per-instance 8,
    * stride 31:16; bytes; divisor. */
   CHECK(w[0] == 0x345678c0 && w[1] == (0x12 | 28u << 16) && w[2] == 4096 && w[3] == 0);
   CHECK(w[8] == 0x00001000 && w[9] == (0xab | 1u << 8 | 0x1000u << 16) && w[10] == 0x10000 && w[11] == 7);
   /* Attribute a at dword 64 + 2a: binding 3:0, format 15:8 (kind 6:2,
    * components - 1 in 1:0), offset 31:16; first register 2 + 4a. */
   CHECK(w[64] == (0 | (1u << 2 | 2) << 8 | 12u << 16) && w[65] == 2);   /* FLOAT32 x3 */
   CHECK(w[66] == (2 | (4u << 2 | 3) << 8 | 0x1234u << 16) && w[67] == 6); /* UNORM8 x4 */
   CHECK(w[94] == ((11u << 2 | 1) << 8 | 4u << 16) && w[95] == 62);        /* SINT16 x2 */
   for (unsigned a = 2; a < 15; a++)
      CHECK(!w[64 + 2 * a] && !w[65 + 2 * a]);
   /* Dword 96: clip-distance count. */
   CHECK(w[96] == 5);
   bool swap;
   CHECK(apex_hw_vertex_format(VK_FORMAT_B8G8R8A8_UNORM, &swap) == (4u << 2 | 3) && swap);
   CHECK(apex_hw_vertex_format(VK_FORMAT_R16_SFLOAT, &swap) == 12u << 2 && !swap);
   CHECK(apex_hw_vertex_format(VK_FORMAT_A2B10G10R10_UNORM_PACK32, &swap) == (13u << 2 | 3));
   CHECK(apex_hw_vertex_format(VK_FORMAT_R8G8_USCALED, &swap) == (15u << 2 | 1));
   CHECK(apex_hw_vertex_format(VK_FORMAT_R16G16B16_SSCALED, &swap) == (18u << 2 | 2));
   CHECK(apex_hw_vertex_format(VK_FORMAT_R8G8B8_SINT, &swap) == (7u << 2 | 2));
   CHECK(!apex_hw_vertex_format(VK_FORMAT_R64_SFLOAT, &swap));
   puts("PASS vertex input block: bindings, attributes, registers 2 + 4L, clip count, kinds");
}

static void
test_raster_depth_blend(void)
{
   struct vk_dynamic_graphics_state dyn = {0};
   dyn.rs.cull_mode = VK_CULL_MODE_BACK_BIT;
   dyn.rs.front_face = VK_FRONT_FACE_COUNTER_CLOCKWISE;
   dyn.rs.polygon_mode = VK_POLYGON_MODE_LINE;
   dyn.rs.line.mode = VK_LINE_RASTERIZATION_MODE_BRESENHAM_KHR;
   dyn.rs.provoking_vertex = VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT;
   dyn.rs.depth_bias.enable = true;
   dyn.rs.depth_clip_enable = VK_MESA_DEPTH_CLIP_ENABLE_TRUE;
   dyn.ms.rasterization_samples = 4;
   dyn.ms.sample_mask = 0xfffa;
   dyn.ms.alpha_to_coverage_enable = true;
   uint32_t r[APEX_HW_RASTER_DWORDS];
   apex_hw_raster_block(&dyn, true, true, r);
   /* cull 1:0, front CCW 2, polygon 4:3, line 6:5, provoking last 7, bias 8,
    * depth clip 9, discard 10, log2 samples 12:11, alpha to coverage 13,
    * alpha to one 14, sample shading 15, late depth 16; dword 1 mask. */
   CHECK(r[0] == (2 | 1u << 2 | 1u << 3 | 1u << 5 | 1u << 7 | 1u << 8 | 1u << 9 | 2u << 11 | 1u << 13 |
                  1u << 15 | 1u << 16));
   CHECK(r[1] == 0xa);
   dyn.rs.rasterizer_discard_enable = true;
   dyn.rs.depth_clip_enable = VK_MESA_DEPTH_CLIP_ENABLE_FALSE;
   dyn.rs.line.mode = VK_LINE_RASTERIZATION_MODE_RECTANGULAR_KHR;
   dyn.ms.alpha_to_one_enable = true;
   dyn.ms.rasterization_samples = 1;
   apex_hw_raster_block(&dyn, false, false, r);
   CHECK(r[0] == (2 | 1u << 2 | 1u << 3 | 1u << 7 | 1u << 8 | 1u << 10 | 1u << 13 | 1u << 14) && r[1] == 0);

   dyn.ds.depth.test_enable = dyn.ds.depth.write_enable = true;
   dyn.ds.depth.compare_op = VK_COMPARE_OP_GREATER_OR_EQUAL;
   dyn.ds.stencil.test_enable = true;
   dyn.ds.stencil.front.op.fail = VK_STENCIL_OP_KEEP;
   dyn.ds.stencil.front.op.pass = VK_STENCIL_OP_INCREMENT_AND_WRAP;
   dyn.ds.stencil.front.op.depth_fail = VK_STENCIL_OP_KEEP;
   dyn.ds.stencil.front.op.compare = VK_COMPARE_OP_EQUAL;
   dyn.ds.stencil.back.op.fail = VK_STENCIL_OP_ZERO;
   dyn.ds.stencil.back.op.pass = VK_STENCIL_OP_REPLACE;
   dyn.ds.stencil.back.op.depth_fail = VK_STENCIL_OP_INVERT;
   dyn.ds.stencil.back.op.compare = VK_COMPARE_OP_ALWAYS;
   dyn.rs.depth_clamp_enable = true;
   uint32_t d[APEX_HW_DEPTH_STENCIL_DWORDS];
   apex_hw_depth_stencil_block(&dyn, VK_FORMAT_D24_UNORM_S8_UINT, true, true, d);
   /* test 0, write 1, compare 4:2, bounds 5, clamp 6, stencil 7, fail and
    * depth-fail keep 8, r from depths 9; ops {fail 2:0, pass 5:3, depth
    * fail 8:6, compare 11:9} front 11:0, back 27:16; dword 2 r. */
   CHECK(d[0] == (1 | 2 | 6u << 2 | 1u << 6 | 1u << 7));
   CHECK(d[1] == (0 | 6u << 3 | 0u << 6 | 2u << 9 | (1u | 2u << 3 | 5u << 6 | 7u << 9) << 16));
   CHECK(d[2] == f2u(1.0f / 16777216.0f));
   dyn.ds.stencil.back.op.fail = dyn.ds.stencil.back.op.depth_fail = VK_STENCIL_OP_KEEP;
   apex_hw_depth_stencil_block(&dyn, VK_FORMAT_D32_SFLOAT, true, false, d);
   CHECK(d[0] == (1 | 2 | 6u << 2 | 1u << 6 | 1u << 8 | 1u << 9) && d[2] == 0);
   apex_hw_depth_stencil_block(&dyn, VK_FORMAT_D16_UNORM, true, false, d);
   CHECK(d[2] == f2u(1.0f / 65536.0f));
   /* A stencil-only attachment tests no depth. */
   apex_hw_depth_stencil_block(&dyn, VK_FORMAT_S8_UINT, false, true, d);
   CHECK((d[0] & 0x7f) == (1u << 6) && (d[0] & (1u << 7)));

   dyn.cb.logic_op_enable = false;
   dyn.cb.logic_op = VK_LOGIC_OP_XOR;
   dyn.cb.color_write_enables = 0xfd;
   dyn.cb.attachments[0] = (struct vk_color_blend_attachment_state){
      .blend_enable = true, .src_color_blend_factor = VK_BLEND_FACTOR_SRC_ALPHA,
      .dst_color_blend_factor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, .color_blend_op = VK_BLEND_OP_ADD,
      .src_alpha_blend_factor = VK_BLEND_FACTOR_ONE, .dst_alpha_blend_factor = VK_BLEND_FACTOR_SRC1_ALPHA,
      .alpha_blend_op = VK_BLEND_OP_MAX, .write_mask = 0xb};
   dyn.cb.attachments[1] = (struct vk_color_blend_attachment_state){.write_mask = 0xf};
   uint32_t b[APEX_HW_BLEND_DWORDS];
   apex_hw_blend_block(&dyn, 2, b);
   /* logic enable 0, op 4:1; attachment a at 1 + a: mask 3:0, alpha dst
    * 8:4, alpha src 13:9, alpha op 16:14, color dst 21:17, color src
    * 26:22, color op 29:27, enable 30 in Vulkan numbering. */
   CHECK(b[0] == (6u << 1));
   CHECK(b[1] == (0xb | 17u << 4 | 1u << 9 | 4u << 14 | 7u << 17 | 6u << 22 | 0u << 27 | 1u << 30));
   CHECK(b[2] == 0);  /* color write disabled */
   for (unsigned i = 3; i < APEX_HW_BLEND_DWORDS; i++)
      CHECK(!b[i]);
   dyn.cb.logic_op_enable = true;
   apex_hw_blend_block(&dyn, 2, b);
   CHECK(b[0] == (1 | 6u << 1) && !(b[1] & (1u << 30)));
   puts("PASS raster, depth-stencil and blend blocks: every field at its documented bits");
}

static void
test_viewport_dynamic(void)
{
   struct vk_dynamic_graphics_state dyn = {0};
   dyn.vp.viewport_count = 2;
   dyn.vp.scissor_count = 2;
   dyn.vp.viewports[0] = (VkViewport){0, 0, 1920, 1080, 0, 1};
   dyn.vp.viewports[1] = (VkViewport){100, 600, 256, -128, 1, 0.25f};
   dyn.vp.scissors[0] = (VkRect2D){{0, 0}, {1920, 1080}};
   dyn.vp.scissors[1] = (VkRect2D){{-5, 7}, {100, 70000}};
   uint32_t v[APEX_HW_VIEWPORT_DWORDS];
   apex_hw_viewport_block(&dyn, v);
   /* Viewport v at 8v: x, y, width, height, min, max (binary32), guard-band
    * exponents x and y; scissor v at 128 + 2v: x0 | y0 << 16, exclusive
    * x1 | y1 << 16. */
   CHECK(v[0] == 0 && v[1] == 0 && v[2] == f2u(1920) && v[3] == f2u(1080) && v[4] == 0 && v[5] == f2u(1));
   CHECK(v[8] == f2u(100) && v[9] == f2u(600) && v[10] == f2u(256) && v[11] == f2u(-128) &&
         v[12] == f2u(1) && v[13] == f2u(0.25f));
   /* 2^k * w/2 + |x + w/2| stays within 2^16 pixels. */
   for (unsigned i = 0; i < 2; i++) {
      for (unsigned axis = 0; axis < 2; axis++) {
         float o = axis ? dyn.vp.viewports[i].y : dyn.vp.viewports[i].x;
         float e = axis ? dyn.vp.viewports[i].height : dyn.vp.viewports[i].width;
         unsigned k = v[8 * i + 6 + axis];
         CHECK(k == apex_hw_guard_band(o, e));
         CHECK(ldexp(fabs(e) / 2, k) + fabs(o + e / 2) <= 65536.0);
         CHECK(ldexp(fabs(e) / 2, k + 1) + fabs(o + e / 2) > 65536.0);
      }
   }
   CHECK(v[128] == 0 && v[129] == (1920 | 1080u << 16));
   CHECK(v[130] == (0 | 7u << 16) && v[131] == (95 | 0xffffu << 16));
   CHECK(!v[16] && !v[132]);

   dyn.ds.stencil.front.reference = 0xab;
   dyn.ds.stencil.front.compare_mask = 0xf0;
   dyn.ds.stencil.front.write_mask = 0x0f;
   dyn.ds.stencil.back.reference = 3;
   dyn.ds.stencil.back.compare_mask = 0xff;
   dyn.ds.stencil.back.write_mask = 0x80;
   dyn.cb.blend_constants[0] = 0.25f;
   dyn.cb.blend_constants[3] = 1;
   dyn.rs.depth_bias.constant_factor = 2;
   dyn.rs.depth_bias.slope_factor = -1.5f;
   dyn.rs.depth_bias.clamp = 0.125f;
   dyn.ds.depth.bounds_test.min = 0.1f;
   dyn.ds.depth.bounds_test.max = 0.9f;
   dyn.rs.line.width = 2.5f;
   uint32_t r[APEX_HW_DYNAMIC_DWORDS];
   apex_hw_dynamic_registers(&dyn, r);
   /* Registers from 0x060: stencil {reference 7:0, compare mask 15:8,
    * write mask 23:16} front then back; blend constants; depth bias
    * constant, slope, clamp; bounds; line width; viewport count. */
   CHECK(r[0] == (0xab | 0xf0u << 8 | 0x0fu << 16) && r[1] == (3 | 0xffu << 8 | 0x80u << 16));
   CHECK(r[2] == f2u(0.25f) && r[3] == 0 && r[4] == 0 && r[5] == f2u(1));
   CHECK(r[6] == f2u(2) && r[7] == f2u(-1.5f) && r[8] == f2u(0.125f));
   CHECK(r[9] == f2u(0.1f) && r[10] == f2u(0.9f) && r[11] == f2u(2.5f) && r[12] == 2);
   puts("PASS viewport block and dynamic registers: viewports, guard bands, scissors, stencil, bias");
}

static void
test_pass(void)
{
   struct apex_hw_pass pass = {
      .pool = 0x4200000000ull, .pool_bytes = 32 << 20, .x0 = 10, .y0 = 20, .x1 = 1930, .y1 = 1100,
      .layers = 2, .samples = 4, .draw_bytes = 2 << 20, .vertex_bytes = 8 << 20, .primitive_bytes = 6 << 20,
   };
   pass.attachment[0] = (struct apex_hw_attachment){
      .present = true, .load = APEX_HW_CLEAR, .store = false, .resolve = true, .format = 0x83, .planes = 1,
      .clear = {0x11223344}, .surface = {0x9876543000ull, true, 61, 0x1000, 0x400},
      .resolve_surface = {0x1122334000ull, false, 120, 0x2000, 0},
   };
   pass.attachment[1] = (struct apex_hw_attachment){
      .present = true, .load = APEX_HW_LOAD, .store = true, .format = 0x49, .planes = 2,
      .clear = {1, 2, 3, 4}, .surface = {0x5500000000ull, true, 8, 0x80, 0x40},
   };
   pass.attachment[8] = (struct apex_hw_attachment){
      .present = true, .load = APEX_HW_LOAD_DONT_CARE, .store = true, .format = 6, .planes = 1,
      .clear = {f2u(0.5f), 0x7f}, .clear_depth = 0.5f, .surface = {0x6600000000ull, true, 61, 0x900, 0x240},
   };
   uint32_t w[APEX_HW_PASS_DWORDS];
   CHECK(apex_hw_pass_record(&pass, w));
   /* Planes 0, then 2-3 for the 16-byte target, then 4: five planes at
    * 4 samples fit 64 x 32 tiles (selector 1). */
   CHECK(w[0] == 0 && w[1] == 0x42 && w[2] == 32u << 20 && w[3] == (10 | 20u << 16) &&
         w[4] == (1930 | 1100u << 16) && w[5] == 2 && w[6] == 0 && w[7] == (1 | 2u << 4));
   CHECK(w[8] == 2u << 20 && w[9] == 8u << 20 && w[10] == 6u << 20);
   for (unsigned i = 11; i < 16; i++)
      CHECK(!w[i]);
   /* Attachment k at dword 16 + 12k: {present 0, load 2:1, store 3
    * (1 don't care), resolve 4, tiled 5, resolve tiled 6, first plane 11:8,
    * format 23:16}; GPUVA; GPUVA 39:32 | pitch << 16; layer stride; clear
    * value 4-7; sample stride 8; resolve GPUVA, GPUVA 39:32 | pitch,
    * layer stride. */
   const uint32_t *a = &w[16];
   CHECK(a[0] == (1 | 1u << 1 | 1u << 3 | 1u << 4 | 1u << 5 | 0u << 8 | 0x83u << 16));
   CHECK(a[1] == 0x76543000 && a[2] == (0x98 | 61u << 16) && a[3] == 0x1000 && a[4] == 0x11223344 &&
         a[8] == 0x400 && a[9] == 0x22334000 && a[10] == (0x11 | 120u << 16) && a[11] == 0x2000);
   a = &w[28];
   CHECK(a[0] == (1 | 0u << 1 | 1u << 5 | 2u << 8 | 0x49u << 16) && a[4] == 1 && a[7] == 4 && a[8] == 0x40);
   a = &w[16 + 12 * 8];
   CHECK(a[0] == (1 | 2u << 1 | 1u << 5 | 4u << 8 | 6u << 16));
   CHECK(a[4] == f2u(0.5f) && a[5] == 0 && a[6] == f2u(0.5f) && a[7] == 0x7f && a[8] == 0x240);
   /* Tile selector: the largest tile with pixels x samples x planes <= 65536. */
   CHECK(apex_hw_tile_selector(1, 16) == 0 && apex_hw_tile_selector(1, 17) == -1);
   CHECK(apex_hw_tile_selector(2, 9) == 1 && apex_hw_tile_selector(8, 2) == 0 &&
         apex_hw_tile_selector(8, 3) == 1);
   CHECK(apex_hw_tile_selector(8, 16) == 3 && apex_hw_tile_selector(4, 4) == 0 &&
         apex_hw_tile_selector(4, 5) == 1);
   CHECK(apex_hw_selector_width(3) == 32 && apex_hw_selector_height(3) == 16);
   /* At most 4096 bins: 4096 x 4096 at 64 x 64 tiles is the limit. */
   struct apex_hw_pass big = {.pool = 0x4200000000ull, .pool_bytes = 32 << 20, .x1 = 4096, .y1 = 4096,
      .layers = 1, .samples = 1, .draw_bytes = 2 << 20, .vertex_bytes = 8 << 20, .primitive_bytes = 6 << 20};
   CHECK(apex_hw_pass_record(&big, w));
   big.layers = 2;
   CHECK(!apex_hw_pass_record(&big, w));
   puts("PASS pass record: header line, attachment descriptors, planes, tile selector, bin bound");
}

/* The raster back end's pass validation (raster_bins.veryl), written out
 * here against records that each break one rule. */
static void
test_pass_validation(void)
{
   uint32_t r[APEX_HW_PASS_DWORDS], w[APEX_HW_PASS_DWORDS];
   /* One 64 x 64 bin of a single-sampled RGBA8 pass in a 64 KiB pool:
    * header 512, draws 1 KiB, heads 64, vertex outputs 2 KiB, primitives
    * 128 and one chunk end at 3,840 bytes. */
   const uint32_t base[17] = {0, 0x42, 3840, 0, 64 | 64u << 16, 1, 0, 0, 1024, 2048, 128,
                              0, 0, 0, 0, 0, 1 | 3u << 16};
   memset(r, 0, sizeof(r));
   memcpy(r, base, sizeof(base));
   CHECK(apex_hw_pass_valid(r));
#define REJECTS(dword, value) do { \
   memcpy(w, r, sizeof(r)); w[dword] = (value); CHECK(!apex_hw_pass_valid(w)); } while (0)
#define ACCEPTS(dword, value) do { \
   memcpy(w, r, sizeof(r)); w[dword] = (value); CHECK(apex_hw_pass_valid(w)); } while (0)
   /* Header: selector, render area, layers, bins. */
   REJECTS(7, 5);
   ACCEPTS(7, 4);
   REJECTS(4, 0 | 64u << 16);
   REJECTS(3, 64 | 0u << 16);
   REJECTS(5, 0);
   REJECTS(5, 2049);
   memcpy(w, r, sizeof(r));
   w[2] = 32 << 20, w[4] = 128 | 64u << 16, w[5] = 2048;
   CHECK(apex_hw_pass_valid(w));
   w[4] = 129 | 64u << 16;
   CHECK(!apex_hw_pass_valid(w));
   /* Regions: draws at least 1 KiB, primitives at least 128 bytes, the
    * last chunk inside the pool. */
   REJECTS(8, 1023);
   REJECTS(10, 127);
   REJECTS(2, 3839);
   ACCEPTS(2, 3840);
   memcpy(w, r, sizeof(r));
   w[4] = 128 | 64u << 16; /* two bins still take one 64-byte head line */
   CHECK(apex_hw_pass_valid(w));
   w[4] = 64 | 1088u << 16; /* 17 bins take two */
   CHECK(!apex_hw_pass_valid(w));
   /* Color formats: classes 1-11 and 13 with the types they take, 1555
    * with either alpha order; no 4444, D24 or packed float. */
   const struct { uint8_t format; bool ok; } formats[] = {
      {0x00, false}, {0x01, true}, {0x13, true}, {0x53, true}, {0xd3, true}, {0x5d, true},
      {0x54, false}, {0x44, true}, {0x49, true}, {0x4a, false}, {0x4b, false}, {0x0a, true},
      {0x8a, true}, {0x2b, true}, {0x14, true}, {0x16, true}, {0x17, false}, {0x1b, false},
      {0x1d, true}, {0x0c, false}, {0x0e, false}, {0x6e, false}, {0x0f, true}, {0x8f, true},
      {0x6f, true}, {0xef, true}, {0x1f, false}, {0x4f, false}, {0x63, false}, {0x7b, false},
      {0x07, true}, {0x37, true},
   };
   for (unsigned i = 0; i < ARRAY_SIZE(formats); i++) {
      CHECK(apex_hw_color_renderable(formats[i].format) == formats[i].ok);
      memcpy(w, r, sizeof(r));
      w[16] = 1 | (uint32_t)formats[i].format << 16;
      CHECK(apex_hw_pass_valid(w) == formats[i].ok);
   }
   /* Depth formats 1-6; an absent attachment's byte is not checked. */
   const unsigned depth = APEX_HW_PASS_ATTACHMENT(APEX_HW_DEPTH_ATTACHMENT);
   for (unsigned f = 0; f < 8; f++) {
      memcpy(w, r, sizeof(r));
      w[depth] = 1 | 1u << 8 | f << 16;
      CHECK(apex_hw_pass_valid(w) == (f >= 1 && f <= 6));
   }
   ACCEPTS(28, 0x7fu << 16);
   /* Planes: an RGBA32 target on an even pair, ending within 16 planes at
    * 64 x 64 single-sampled; 4 samples fit four planes at 64 x 64. */
   ACCEPTS(28, 1 | 2u << 8 | 0x49u << 16);
   REJECTS(28, 1 | 3u << 8 | 0x49u << 16);
   ACCEPTS(28, 1 | 14u << 8 | 0x49u << 16);
   REJECTS(28, 1 | 15u << 8 | 0x49u << 16);
   ACCEPTS(28, 1 | 15u << 8 | 0x03u << 16);
   memcpy(w, r, sizeof(r));
   w[7] = 2 << 4;
   w[depth] = 1 | 3u << 8 | 3u << 16;
   CHECK(apex_hw_pass_valid(w));
   w[depth] = 1 | 4u << 8 | 3u << 16;
   CHECK(!apex_hw_pass_valid(w));
   w[7] = 1 | 2u << 4; /* 64 x 32 tiles hold eight planes at 4 samples */
   CHECK(apex_hw_pass_valid(w));
#undef REJECTS
#undef ACCEPTS
   /* The driver's pool regions and every color attachment format it
    * renders pass. */
   CHECK(APEX_POOL_DRAW_BYTES >= 1024 && APEX_POOL_PRIMITIVE_BYTES >= 128);
   for (VkFormat f = VK_FORMAT_UNDEFINED; f <= VK_FORMAT_ASTC_12x12_SRGB_BLOCK; f++) {
      uint8_t code = apex_hw_color_format(f);
      CHECK(!code || apex_hw_color_renderable(code));
   }
   struct apex_hw_pass pass = {.pool = 0x4200000000ull, .pool_bytes = APEX_BIN_POOL_BYTES, .x1 = 4096,
      .y1 = 4096, .layers = 1, .samples = 1, .draw_bytes = APEX_POOL_DRAW_BYTES,
      .vertex_bytes = APEX_POOL_VERTEX_BYTES, .primitive_bytes = APEX_POOL_PRIMITIVE_BYTES};
   CHECK(apex_hw_pass_record(&pass, w));
   pass.primitive_bytes = 64;
   CHECK(!apex_hw_pass_record(&pass, w));
   pass.primitive_bytes = APEX_POOL_PRIMITIVE_BYTES;
   pass.pool_bytes = APEX_POOL_DRAW_BYTES + APEX_POOL_VERTEX_BYTES + APEX_POOL_PRIMITIVE_BYTES;
   CHECK(!apex_hw_pass_record(&pass, w));
   puts("PASS pass validation: selector, area, layers, bins, regions, color and depth formats, planes");
}

/* Texel (x, y) of a tiled level, written independently from the
 * architecture's prose: 4 KiB tiles of 8 x 8 Morton blocks (x in the even
 * bits) of 4 x 4, 8 x 4, 8 x 8, 4 x 2 or 4 x 1 texels, texels in 4 x 4
 * sub-blocks left to right then top to bottom, row-major within each. */
static uint64_t
reference_offset(unsigned bytes, unsigned width, unsigned x, unsigned y)
{
   unsigned bw = bytes == 4 ? 4 : bytes == 2 ? 8 : bytes == 1 ? 8 : 4;
   unsigned bh = bytes == 4 ? 4 : bytes == 2 ? 4 : bytes == 1 ? 8 : bytes == 8 ? 2 : 1;
   unsigned tw = 8 * bw, th = 8 * bh, tiles = (width + tw - 1) / tw;
   uint64_t tile = (y / th) * tiles + x / tw;
   unsigned bx = (x % tw) / bw, by = (y % th) / bh, morton = 0;
   for (unsigned i = 0; i < 3; i++)
      morton |= ((bx >> i) & 1) << (2 * i) | ((by >> i) & 1) << (2 * i + 1);
   unsigned ix = x % bw, iy = y % bh, sub_h = bh < 4 ? bh : 4;
   unsigned sub = (iy / 4) * (bw / 4) + ix / 4;
   unsigned within = sub * 4 * sub_h + (iy % 4) * 4 + ix % 4;
   return tile * 4096 + morton * 64 + within * bytes;
}

static void
test_layout(void)
{
   static const unsigned sizes[5] = {1, 2, 4, 8, 16};
   for (unsigned s = 0; s < 5; s++) {
      struct apex_hw_layout layout;
      apex_hw_layout_init(&layout, sizes[s], false, (VkExtent3D){100, 70, 1}, 3, 5, 1, true);
      CHECK(apex_hw_tile_width(sizes[s]) * apex_hw_tile_height(sizes[s]) * sizes[s] == 4096);
      for (unsigned y = 0; y < 70; y += 3)
         for (unsigned x = 0; x < 100; x++)
            CHECK(apex_hw_texel_offset(&layout, 0, 0, 0, x, y) == reference_offset(sizes[s], 100, x, y));
      /* Levels follow in whole tiles; a layer holds the mip chain. */
      uint64_t level1 = (uint64_t)DIV_ROUND_UP(100, apex_hw_tile_width(sizes[s])) *
                        DIV_ROUND_UP(70, apex_hw_tile_height(sizes[s])) * 4096;
      CHECK(layout.level[1].offset == level1 && layout.level[1].width == 50 && layout.level[2].height == 17);
      CHECK(layout.layer_stride % 4096 == 0 && layout.size == layout.layer_stride * 5);
      CHECK(apex_hw_texel_offset(&layout, 1, 2, 0, 3, 4) ==
            2 * layout.layer_stride + level1 + reference_offset(sizes[s], 50, 3, 4));
   }
   /* Samples and 3D slices are consecutive planes of a level. */
   struct apex_hw_layout ms;
   apex_hw_layout_init(&ms, 4, false, (VkExtent3D){64, 64, 1}, 1, 1, 4, true);
   CHECK(ms.level[0].pitch == 2 && ms.level[0].plane == 4 * 4096 && ms.size == 4 * 4 * 4096);
   CHECK(apex_hw_texel_offset(&ms, 0, 0, 3, 0, 0) == 3 * 4 * 4096);
   struct apex_hw_layout volume;
   apex_hw_layout_init(&volume, 2, true, (VkExtent3D){64, 32, 8}, 2, 1, 1, true);
   CHECK(volume.level[0].plane == 4096 && volume.level[1].offset == 8 * 4096 && volume.level[1].depth == 4);
   /* Linear levels: 64-byte aligned rows. */
   struct apex_hw_layout linear;
   apex_hw_layout_init(&linear, 3, false, (VkExtent3D){30, 10, 1}, 1, 1, 1, false);
   CHECK(linear.level[0].pitch == 128 && apex_hw_texel_offset(&linear, 0, 0, 0, 5, 2) == 2 * 128 + 15);
   puts("PASS tiled layout: Morton blocks and sub-blocks at every texel size, levels, layers, planes");
}

static void
test_descriptors(void)
{
   /* Image: [39:0] GPUVA, [43:40] class, [46:44] type, [47] BGR, [48] tiled,
    * [54:52] samples log2, [66:55] swizzle, [80:67] width - 1, [94:81]
    * height - 1, [105:95] depth or layers - 1, [110:106] levels, [126:111]
    * row pitch, [160:127] layer stride, [172:161] minimum LOD. */
   const struct apex_hw_image image = {
      .va = 0xfedcba9876ull, .format = 0x83 | 0x50, .tiled = true, .samples_log2 = 2,
      .swizzle = {2, 1, 0, APEX_HW_SWIZZLE_1}, .width = 1920, .height = 1080, .depth = 6, .levels = 11,
      .pitch64 = 0xbeef, .stride64 = 0x2abcdef12ull, .min_lod = 0x345,
   };
   uint32_t d[8];
   apex_hw_image_descriptor(&image, d);
   CHECK(field(d, 0, 40) == 0xfedcba9876ull && field(d, 40, 4) == 3 && field(d, 44, 3) == 5 &&
         field(d, 47, 1) == 1 && field(d, 48, 1) == 1 && field(d, 49, 3) == 0 && field(d, 52, 3) == 2);
   CHECK(field(d, 55, 3) == 2 && field(d, 58, 3) == 1 && field(d, 61, 3) == 0 && field(d, 64, 3) == 5);
   CHECK(field(d, 67, 14) == 1919 && field(d, 81, 14) == 1079 && field(d, 95, 11) == 5 &&
         field(d, 106, 5) == 11 && field(d, 111, 16) == 0xbeef && field(d, 127, 34) == 0x2abcdef12ull &&
         field(d, 161, 12) == 0x345 && field(d, 173, 83) == 0);
   /* Formats: class, type, BGR and the stored channel of each component. */
   uint8_t code, ch[4];
   CHECK(apex_hw_texture_format(VK_FORMAT_B8G8R8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT, &code, ch) &&
         code == (3 | 5u << 4 | 0x80) && ch[0] == 0 && ch[3] == 3);
   CHECK(apex_hw_texture_format(VK_FORMAT_R8G8_SNORM, VK_IMAGE_ASPECT_COLOR_BIT, &code, ch) &&
         code == (2 | 1u << 4) && ch[1] == 1 && ch[2] == APEX_HW_SWIZZLE_0 && ch[3] == APEX_HW_SWIZZLE_1);
   CHECK(apex_hw_texture_format(VK_FORMAT_R16G16B16A16_SFLOAT, VK_IMAGE_ASPECT_COLOR_BIT, &code, ch) &&
         code == (6 | 4u << 4));
   CHECK(apex_hw_texture_format(VK_FORMAT_R32_UINT, VK_IMAGE_ASPECT_COLOR_BIT, &code, ch) && code == (7 | 2u << 4));
   CHECK(apex_hw_texture_format(VK_FORMAT_B5G6R5_UNORM_PACK16, VK_IMAGE_ASPECT_COLOR_BIT, &code, ch) &&
         code == (10 | 0x80) && ch[3] == APEX_HW_SWIZZLE_1);
   CHECK(apex_hw_texture_format(VK_FORMAT_A2B10G10R10_UINT_PACK32, VK_IMAGE_ASPECT_COLOR_BIT, &code, ch) &&
         code == (11 | 2u << 4));
   CHECK(apex_hw_texture_format(VK_FORMAT_D24_UNORM_S8_UINT, VK_IMAGE_ASPECT_DEPTH_BIT, &code, ch) &&
         code == 12 && ch[0] == 0 && ch[1] == APEX_HW_SWIZZLE_0 && ch[3] == APEX_HW_SWIZZLE_1);
   CHECK(apex_hw_texture_format(VK_FORMAT_D24_UNORM_S8_UINT, VK_IMAGE_ASPECT_STENCIL_BIT, &code, ch) &&
         code == (3 | 2u << 4) && ch[0] == 3);
   CHECK(apex_hw_texture_format(VK_FORMAT_D32_SFLOAT_S8_UINT, VK_IMAGE_ASPECT_STENCIL_BIT, &code, ch) &&
         code == (8 | 2u << 4) && ch[0] == 1);
   CHECK(apex_hw_texture_format(VK_FORMAT_D16_UNORM, VK_IMAGE_ASPECT_DEPTH_BIT, &code, ch) && code == 4);
   CHECK(!apex_hw_texture_format(VK_FORMAT_R8G8B8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, &code, ch));
   /* The packed 16-bit classes, 4444 (14) and 1555 (15): type 0 with alpha
    * in the low field, type 6 in the high field, BGR swapping R and B. */
   const struct { VkFormat format; uint8_t code; bool render; } packed[] = {
      {VK_FORMAT_R4G4B4A4_UNORM_PACK16, 14, false}, {VK_FORMAT_B4G4R4A4_UNORM_PACK16, 14 | 0x80, false},
      {VK_FORMAT_A4R4G4B4_UNORM_PACK16, 14 | 6u << 4, false},
      {VK_FORMAT_A4B4G4R4_UNORM_PACK16, 14 | 6u << 4 | 0x80, false},
      {VK_FORMAT_R5G5B5A1_UNORM_PACK16, 15, true}, {VK_FORMAT_B5G5R5A1_UNORM_PACK16, 15 | 0x80, true},
      {VK_FORMAT_A1R5G5B5_UNORM_PACK16, 15 | 6u << 4, true},
      {VK_FORMAT_A1B5G5R5_UNORM_PACK16_KHR, 15 | 6u << 4 | 0x80, true},
   };
   for (unsigned i = 0; i < ARRAY_SIZE(packed); i++) {
      CHECK(apex_hw_texture_format(packed[i].format, VK_IMAGE_ASPECT_COLOR_BIT, &code, ch) &&
            code == packed[i].code && ch[0] == 0 && ch[1] == 1 && ch[2] == 2 && ch[3] == 3);
      CHECK(apex_hw_color_format(packed[i].format) == (packed[i].render ? packed[i].code : 0));
   }
   /* Class 11 with FLOAT is B10G11R11, with type 7 E5B9G9R9; alpha reads
    * one and neither renders. */
   CHECK(apex_hw_texture_format(VK_FORMAT_B10G11R11_UFLOAT_PACK32, VK_IMAGE_ASPECT_COLOR_BIT, &code, ch) &&
         code == (11 | 4u << 4) && ch[0] == 0 && ch[2] == 2 && ch[3] == APEX_HW_SWIZZLE_1);
   CHECK(apex_hw_texture_format(VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, VK_IMAGE_ASPECT_COLOR_BIT, &code, ch) &&
         code == (11 | 7u << 4) && ch[2] == 2 && ch[3] == APEX_HW_SWIZZLE_1);
   CHECK(!apex_hw_color_format(VK_FORMAT_B10G11R11_UFLOAT_PACK32) &&
         !apex_hw_color_format(VK_FORMAT_E5B9G9R9_UFLOAT_PACK32));
   CHECK(apex_hw_color_format(VK_FORMAT_A8B8G8R8_SRGB_PACK32) == (3 | 5u << 4) &&
         !apex_hw_color_format(VK_FORMAT_D16_UNORM));
   /* The descriptor carries the byte at [47:40]. */
   const struct apex_hw_image e5 = {.format = 11 | 7u << 4 | 0x80, .width = 1, .height = 1, .depth = 1, .levels = 1};
   apex_hw_image_descriptor(&e5, d);
   CHECK(field(d, 40, 4) == 11 && field(d, 44, 3) == 7 && field(d, 47, 1) == 1);
   CHECK(apex_hw_depth_format(VK_FORMAT_D16_UNORM) == 1 && apex_hw_depth_format(VK_FORMAT_X8_D24_UNORM_PACK32) == 2 &&
         apex_hw_depth_format(VK_FORMAT_D24_UNORM_S8_UINT) == 3 && apex_hw_depth_format(VK_FORMAT_D32_SFLOAT) == 4 &&
         apex_hw_depth_format(VK_FORMAT_S8_UINT) == 5 && apex_hw_depth_format(VK_FORMAT_D32_SFLOAT_S8_UINT) == 6);

   /* Sampler: [0] mag, [1] min linear, [2] linear mip, [11:3] address
    * modes, [12] compare, [15:13] op, [17:16] reduction, [18]
    * unnormalized, [36:23] bias Q5.8, [48:37] min and [60:49] max LOD Q4.8,
    * [63:61] border type, [255:128] custom RGBA. */
   const VkSamplerCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .magFilter = VK_FILTER_LINEAR,
      .minFilter = VK_FILTER_NEAREST, .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
      .addressModeU = VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER, .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
      .mipLodBias = -2.5f, .compareEnable = true, .compareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
      .minLod = 1.25f, .maxLod = VK_LOD_CLAMP_NONE, .unnormalizedCoordinates = false,
   };
   struct vk_sampler sampler = {.reduction_mode = VK_SAMPLER_REDUCTION_MODE_MAX};
   sampler.border_color_value.float32[0] = 0.5f;
   sampler.border_color_value.float32[3] = 1.0f;
   apex_hw_sampler_descriptor(&info, &sampler, d);
   CHECK(field(d, 0, 1) == 1 && field(d, 1, 1) == 0 && field(d, 2, 1) == 1 && field(d, 3, 3) == 4 &&
         field(d, 6, 3) == 3 && field(d, 9, 3) == 0 && field(d, 12, 1) == 1 && field(d, 13, 3) == 3 &&
         field(d, 16, 2) == 2 && field(d, 18, 1) == 0 && field(d, 19, 4) == 0);
   CHECK(field(d, 23, 14) == ((uint32_t)-640 & 0x3fff) && field(d, 37, 12) == 320 && field(d, 49, 12) == 4095 &&
         field(d, 61, 3) == 3 && field(d, 64, 64) == 0 && d[4] == f2u(0.5f) && d[7] == f2u(1.0f));
   puts("PASS texture-unit descriptors: image and sampler bits, format classes with 4444, 1555 and "
        "packed floats, depth and stencil views");
}

/* The internal copy kernel, compiled by p7-isa and run on the ISA model,
 * copies a linear buffer into a tiled image surface at the offsets of
 * apex_hw_texel_offset. */
static void
test_copy_kernel(struct apex_device *device)
{
   struct apex_program *copy;
   CHECK(apex_internal_program(device, APEX_INTERNAL_COPY, &copy) == VK_SUCCESS);
   const unsigned width = 40, height = 9, bytes = 4;
   struct apex_hw_layout layout;
   apex_hw_layout_init(&layout, bytes, false, (VkExtent3D){width, height, 1}, 1, 1, 1, true);
   const uint64_t table_va = 0x100000, src_va = 0x200000, dst_va = 0x400000;
   uint32_t *src = malloc(width * height * bytes), *dst = calloc(1, layout.size);
   for (unsigned i = 0; i < width * height; i++)
      src[i] = 0x9e3779b9u * (i + 1);
   uint32_t table[8 + APEX_TRAILER_WORDS + APEX_JOB_WORDS] = {0};
   table[3] = 1; /* robust sentinel row */
   uint32_t *grid = &table[8], *job = &table[APEX_JOB_BASE];
   grid[0] = DIV_ROUND_UP(width, 16), grid[1] = height, grid[2] = 1;
   uint32_t *s = &job[APEX_COPY_SRC], *t = &job[APEX_COPY_DST];
   s[APEX_SURFACE_VA] = src_va, s[APEX_SURFACE_FLAGS] = 1 << 4, s[APEX_SURFACE_PITCH] = width * bytes;
   s[APEX_SURFACE_TEXEL] = bytes;
   t[APEX_SURFACE_VA] = dst_va, t[APEX_SURFACE_FLAGS] = APEX_SURFACE_TILED | 2 << 1 | 1 << 4;
   t[APEX_SURFACE_PITCH] = layout.level[0].pitch, t[APEX_SURFACE_TEXEL] = bytes;
   job[APEX_COPY_EXTENT] = width, job[APEX_COPY_EXTENT + 1] = height, job[APEX_COPY_EXTENT + 2] = 1;
   job[APEX_COPY_BYTES] = bytes, job[APEX_COPY_MASK] = ~0u;
   struct apex_sim_region regions[] = {
      {table_va, table, sizeof(table)}, {src_va, src, width * height * bytes}, {dst_va, dst, layout.size},
   };
   uint32_t user[16] = {table_va, 0, table_va + 32, 0};
   char diagnostic[256] = "";
   if (apex_simulate(copy->code.data, copy->code.size, user, grid, 0, NULL, regions, 3, diagnostic)) {
      fprintf(stderr, "simulation: %s\n", diagnostic);
      abort();
   }
   for (unsigned y = 0; y < height; y++)
      for (unsigned x = 0; x < width; x++)
         CHECK(dst[apex_hw_texel_offset(&layout, 0, 0, 0, x, y) / 4] == src[y * width + x]);
   free(src);
   free(dst);
   puts("PASS copy kernel on the ISA model: a linear buffer into the tiled layout");
}

int
main(void)
{
   test_vertex_input();
   test_raster_depth_blend();
   test_viewport_dynamic();
   test_pass();
   test_pass_validation();
   test_layout();
   test_descriptors();
   struct vk_instance instance;
   const struct vk_instance_extension_table extensions = {0};
   const struct vk_instance_dispatch_table instance_dispatch = {0};
   const VkInstanceCreateInfo ii = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
   CHECK(vk_instance_init(&instance, &extensions, &instance_dispatch, &ii, vk_default_allocator()) == VK_SUCCESS);
   (void)vk_instance_to_handle(&instance);
   struct vk_physical_device physical;
   const struct vk_physical_device_dispatch_table physical_dispatch = {0};
   const struct vk_properties properties = {.subgroupSize = 16};
   CHECK(vk_physical_device_init(&physical, &instance, NULL, NULL, &properties, &physical_dispatch) == VK_SUCCESS);
   (void)vk_physical_device_to_handle(&physical);
   struct apex_device device;
   const float priority = 1;
   const VkDeviceQueueCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueCount = 1, .pQueuePriorities = &priority};
   const VkDeviceCreateInfo di = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qi};
   CHECK(apex_device_init(&device, &physical, &di, NULL, -1) == VK_SUCCESS);
   test_copy_kernel(&device);
   apex_device_finish(&device);
   vk_physical_device_finish(&physical);
   vk_instance_finish(&instance);
   return 0;
}
