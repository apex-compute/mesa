/* SPDX-License-Identifier: MIT */
/* Software sampler: lowers NIR texture instructions to descriptor-row loads,
 * texel address arithmetic, a data-driven format decoder and filtering on the
 * ordinary vector machine. Rows follow apex_format.h. Cube maps select a face
 * per sample and clamp at face edges. */
#include "apex_pipeline.h"
#include "apex_format.h"
#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include <math.h>

struct texture_lowering {
   struct apex_program *program;
   bool invalid;
};

static nir_def *
row_word(nir_builder *b, nir_def *row, unsigned word)
{
   return nir_load_ssbo(b, 1, 32, nir_imm_int(b, 0),
                        nir_iadd_imm(b, nir_imul_imm(b, row, sizeof(union apex_descriptor)), word * 4),
                        .align_mul = 4, .access = ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER);
}

/* Resolves a texture or sampler deref to its first table row; out-of-range
 * array indices select the zero sentinel row. */
static nir_def *
deref_row(nir_builder *b, struct texture_lowering *ctx, nir_deref_instr *deref, bool sampler_part)
{
   nir_def *index = nir_imm_int(b, 0);
   if (deref->deref_type == nir_deref_type_array) {
      index = deref->arr.index.ssa;
      deref = nir_deref_instr_parent(deref);
   }
   nir_variable *var = deref->deref_type == nir_deref_type_var ? deref->var : NULL;
   struct apex_program *program = ctx->program;
   if (!var || var->data.descriptor_set >= program->set_count) {
      ctx->invalid = true;
      return nir_imm_int(b, 0);
   }
   const struct apex_set_layout *layout = program->set_layouts[var->data.descriptor_set];
   unsigned binding = var->data.binding;
   if (!layout || binding >= layout->binding_count || !layout->bindings[binding].count) {
      ctx->invalid = true;
      return nir_imm_int(b, 0);
   }
   const struct apex_binding_layout *bl = &layout->bindings[binding];
   unsigned slots = apex_descriptor_slots(bl->type);
   unsigned base = program->set_offsets[var->data.descriptor_set] + bl->slot;
   BITSET_SET_COUNT(program->used_descriptors, base, bl->count * slots);
   /* A combined descriptor keeps its sampler in the third row. */
   unsigned part = sampler_part && bl->type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ? 2 : 0;
   return nir_bcsel(b, nir_ult_imm(b, index, bl->count),
                    nir_iadd_imm(b, nir_imul_imm(b, index, slots), base + part),
                    nir_imm_int(b, program->descriptor_count));
}

struct image_info {
   nir_def *address;          /* 2x32 */
   nir_def *width, *height, *depth, *layers;
   nir_def *base_level, *levels, *base_layer, *layer_count;
   nir_def *format[3];
   nir_def *bytes, *samples;
   bool three_d;
};

static struct image_info
load_image(nir_builder *b, nir_def *row, bool three_d)
{
   struct image_info i = {.three_d = three_d};
   nir_def *w[16];
   for (unsigned k = 0; k < 15; k++)
      w[k] = row_word(b, nir_iadd_imm(b, row, k / 8), k % 8);
   i.address = nir_vec2(b, w[0], w[1]);
   i.width = w[2];
   i.height = w[3];
   i.depth = w[4];
   i.layers = w[5];
   i.base_level = w[6];
   i.levels = w[7];
   i.base_layer = w[8];
   i.layer_count = w[9];
   i.format[0] = w[10];
   i.format[1] = w[11];
   i.format[2] = w[12];
   i.bytes = nir_iand_imm(b, w[10], 31);
   i.samples = w[14];
   return i;
}

static nir_def *
minify(nir_builder *b, nir_def *size, nir_def *level)
{
   return nir_umax(b, nir_ushr(b, size, level), nir_imm_int(b, 1));
}

/* Row pitch and texel layout of absolute level `level`. */
static nir_def *
row_pitch(nir_builder *b, const struct image_info *i, nir_def *level)
{
   /* Multisampled rows hold every sample and align to 128 bytes. */
   nir_def *bytes = nir_imul(b, nir_imul(b, minify(b, i->width, level), i->bytes), i->samples);
   nir_def *align = nir_bcsel(b, nir_ugt_imm(b, i->samples, 1), nir_imm_int(b, 127), nir_imm_int(b, 63));
   return nir_iand(b, nir_iadd(b, bytes, align), nir_inot(b, align));
}

static nir_def *
plane_bytes(nir_builder *b, const struct image_info *i, nir_def *level)
{
   return nir_imul(b, row_pitch(b, i, level), minify(b, i->height, level));
}

/* Byte offset of absolute level `level`: mip-major levels, each holding all
 * layers (or all depth slices for 3D images). */
static nir_def *
level_offset(nir_builder *b, const struct image_info *i, nir_def *level)
{
   nir_variable *offset = nir_local_variable_create(b->impl, glsl_uint_type(), "level_offset");
   nir_variable *l = nir_local_variable_create(b->impl, glsl_uint_type(), "level");
   nir_store_var(b, offset, nir_imm_int(b, 0), 1);
   nir_store_var(b, l, nir_imm_int(b, 0), 1);
   nir_push_loop(b);
   nir_def *current = nir_load_var(b, l);
   nir_push_if(b, nir_uge(b, current, level));
   nir_jump(b, nir_jump_break);
   nir_pop_if(b, NULL);
   nir_def *count = i->three_d ? minify(b, i->depth, current) : i->layers;
   nir_store_var(b, offset, nir_iadd(b, nir_load_var(b, offset),
                                    nir_imul(b, plane_bytes(b, i, current), count)), 1);
   nir_store_var(b, l, nir_iadd_imm(b, current, 1), 1);
   nir_pop_loop(b, NULL);
   return nir_load_var(b, offset);
}

/* Unsigned float with a 5-bit exponent (bias 15) and `mantissa` bits. */
nir_def *
apex_small_float_to_float(nir_builder *b, nir_def *v, unsigned mantissa)
{
   nir_def *e = nir_ushr_imm(b, v, mantissa);
   nir_def *m = nir_iand_imm(b, v, BITFIELD_MASK(mantissa));
   nir_def *normal = nir_ior(b, nir_ishl_imm(b, nir_iadd_imm(b, e, 112), 23), nir_ishl_imm(b, m, 23 - mantissa));
   nir_def *special = nir_ior(b, nir_imm_int(b, 0x7f800000), nir_ishl_imm(b, m, 23 - mantissa));
   nir_def *denormal = nir_fmul_imm(b, nir_u2f32(b, m), 1.0 / (1 << 14) / (1 << mantissa));
   return nir_bcsel(b, nir_ieq_imm(b, e, 0), denormal, nir_bcsel(b, nir_ieq_imm(b, e, 31), special, normal));
}

nir_def *
apex_half_to_float(nir_builder *b, nir_def *h)
{
   nir_def *sign = nir_ishl_imm(b, nir_iand_imm(b, h, 0x8000), 16);
   nir_def *e = nir_iand_imm(b, nir_ushr_imm(b, h, 10), 31);
   nir_def *m = nir_iand_imm(b, h, 0x3ff);
   nir_def *normal = nir_ior(b, nir_ishl_imm(b, nir_iadd_imm(b, e, 112), 23), nir_ishl_imm(b, m, 13));
   nir_def *special = nir_ior(b, nir_imm_int(b, 0x7f800000), nir_ishl_imm(b, m, 13));
   nir_def *denormal = nir_fmul_imm(b, nir_u2f32(b, m), 1.0 / (1 << 24));
   nir_def *value = nir_bcsel(b, nir_ieq_imm(b, e, 0), denormal,
                              nir_bcsel(b, nir_ieq_imm(b, e, 31), special, normal));
   return nir_ior(b, value, sign);
}

/* FP32 to FP16 bits with round-to-nearest-even in the low 16 bits. */
nir_def *
apex_float_to_half(nir_builder *b, nir_def *f)
{
   nir_def *sign = nir_iand_imm(b, nir_ushr_imm(b, f, 16), 0x8000);
   nir_def *abs = nir_iand_imm(b, f, 0x7fffffff);
   /* Normal results: rebias the exponent, then round 13 mantissa bits away. */
   nir_def *rebased = nir_iadd_imm(b, abs, -(112 << 23));
   nir_def *normal = nir_ushr_imm(b, nir_iadd(b, nir_iadd_imm(b, rebased, 0xfff),
                                              nir_iand_imm(b, nir_ushr_imm(b, rebased, 13), 1)), 13);
   /* Subnormal results are exact multiples of 2^-24 after rounding. */
   nir_def *subnormal = nir_f2u32(b, nir_fround_even(b, nir_fmul_imm(b, abs, 16777216.0)));
   nir_def *nan = nir_ior_imm(b, nir_iand_imm(b, nir_ushr_imm(b, abs, 13), 0x3ff), 0x7e00);
   nir_def *result = nir_bcsel(b, nir_ult_imm(b, abs, 0x38800000), subnormal, normal);
   result = nir_bcsel(b, nir_uge_imm(b, abs, 0x477ff000), nir_imm_int(b, 0x7c00), result);
   result = nir_bcsel(b, nir_uge_imm(b, abs, 0x7f800001), nan, result);
   return nir_ior(b, result, sign);
}

static nir_def *
srgb_to_linear(nir_builder *b, nir_def *c)
{
   nir_def *low = nir_fmul_imm(b, c, 1.0 / 12.92);
   nir_def *high = nir_fpow(b, nir_fmul_imm(b, nir_fadd_imm(b, c, 0.055), 1.0 / 1.055),
                            nir_imm_float(b, 2.4));
   return nir_bcsel(b, nir_fge(b, nir_imm_float(b, 0.04045f), c), low, high);
}

/* Decodes one texel at byte address `address` into RGBA words: FP32 bits
 * for normalized/float formats, integers for pure-integer formats. */
static nir_def *
decode_texel(nir_builder *b, const struct image_info *i, nir_def *address)
{
   /* Texels up to 16 bytes are naturally aligned for 4..16-byte formats;
    * 1/2-byte texels sit inside one aligned word. */
   nir_def *byte = nir_channel(b, address, 0);
   nir_def *misalign = nir_iand_imm(b, byte, 3);
   nir_def *aligned = nir_vec2(b, nir_iand_imm(b, byte, ~3u), nir_channel(b, address, 1));
   nir_def *words[4], *zero = nir_imm_int(b, 0);
   for (unsigned k = 0; k < 4; k++) {
      nir_def *present = nir_ult(b, nir_imm_int(b, k * 4), i->bytes);
      nir_push_if(b, present);
      nir_def *loaded = nir_load_global_2x32(b, 1, 32,
         nir_build_addr_iadd_imm(b, aligned, nir_address_format_2x32bit_global, nir_var_mem_global,
                                 k * 4), .align_mul = 4);
      nir_pop_if(b, NULL);
      words[k] = nir_if_phi(b, loaded, zero);
   }
   words[0] = nir_ushr(b, words[0], nir_ishl_imm(b, misalign, 3));
   nir_def *channels[4];
   for (unsigned k = 0; k < 4; k++) {
      nir_def *field = nir_iand_imm(b, nir_ushr_imm(b, i->format[1 + k / 2], 16 * (k % 2)), 0xffff);
      nir_def *shift = nir_iand_imm(b, field, 127);
      nir_def *size = nir_iand_imm(b, nir_ushr_imm(b, field, 7), 63);
      nir_def *type = nir_ushr_imm(b, field, 13);
      nir_def *word = nir_bcsel(b, nir_ult_imm(b, shift, 32), words[0],
                      nir_bcsel(b, nir_ult_imm(b, shift, 64), words[1],
                      nir_bcsel(b, nir_ult_imm(b, shift, 96), words[2], words[3])));
      nir_def *bits = nir_ushr(b, word, nir_iand_imm(b, shift, 31));
      nir_def *full = nir_ieq_imm(b, size, 32);
      nir_def *mask = nir_bcsel(b, full, nir_imm_int(b, ~0u),
                                nir_isub(b, nir_ishl(b, nir_imm_int(b, 1), size), nir_imm_int(b, 1)));
      nir_def *raw = nir_iand(b, bits, mask);
      nir_def *pad = nir_isub(b, nir_imm_int(b, 32), size);
      nir_def *extended = nir_bcsel(b, full, raw, nir_ishr(b, nir_ishl(b, raw, pad), pad));
      nir_def *unorm = nir_fdiv(b, nir_u2f32(b, raw), nir_u2f32(b, mask));
      nir_def *snorm = nir_fmax(b, nir_fdiv(b, nir_i2f32(b, extended),
                                           nir_u2f32(b, nir_ushr_imm(b, mask, 1))),
                                nir_imm_float(b, -1.0f));
      nir_def *fp = nir_bcsel(b, nir_ieq_imm(b, size, 16), apex_half_to_float(b, raw), raw);
      nir_def *value = raw;
      value = nir_bcsel(b, nir_ieq_imm(b, type, APEX_CHANNEL_UNORM), unorm, value);
      value = nir_bcsel(b, nir_ieq_imm(b, type, APEX_CHANNEL_SNORM), snorm, value);
      value = nir_bcsel(b, nir_ieq_imm(b, type, APEX_CHANNEL_SINT), extended, value);
      value = nir_bcsel(b, nir_ieq_imm(b, type, APEX_CHANNEL_FLOAT), fp, value);
      value = nir_bcsel(b, nir_ieq_imm(b, type, APEX_CHANNEL_USCALED), nir_u2f32(b, raw), value);
      value = nir_bcsel(b, nir_ieq_imm(b, type, APEX_CHANNEL_SSCALED), nir_i2f32(b, extended), value);
      channels[k] = value;
   }
   /* Packed RGB formats replace channels 0..2. */
   nir_def *e5 = nir_ushr_imm(b, words[0], 27);
   nir_def *scale = nir_ishl_imm(b, nir_iadd_imm(b, e5, 127 - 24), 23); /* 2^(E - 15 - 9) */
   nir_def *packed9 = nir_ine_imm(b, nir_iand_imm(b, i->format[0], APEX_FORMAT_RGB9E5), 0);
   nir_def *packed11 = nir_ine_imm(b, nir_iand_imm(b, i->format[0], APEX_FORMAT_R11G11B10), 0);
   const unsigned small_shift[3] = {0, 11, 22}, small_bits[3] = {11, 11, 10};
   for (unsigned k = 0; k < 3; k++) {
      nir_def *m9 = nir_iand_imm(b, nir_ushr_imm(b, words[0], 9 * k), 0x1ff);
      nir_def *rgb9 = nir_fmul(b, nir_u2f32(b, m9), scale);
      nir_def *small = apex_small_float_to_float(b, nir_iand_imm(b,
         nir_ushr_imm(b, words[0], small_shift[k]), BITFIELD_MASK(small_bits[k])), small_bits[k] - 5);
      channels[k] = nir_bcsel(b, packed9, rgb9, nir_bcsel(b, packed11, small, channels[k]));
   }
   nir_def *integer = nir_ine_imm(b, nir_iand_imm(b, i->format[0], 1u << 20), 0);
   nir_def *srgb = nir_ine_imm(b, nir_iand_imm(b, i->format[0], 1u << 5), 0);
   nir_def *one = nir_bcsel(b, integer, nir_imm_int(b, 1), nir_imm_float(b, 1.0f));
   nir_def *out[4];
   for (unsigned c = 0; c < 4; c++) {
      nir_def *select = nir_iand_imm(b, nir_ushr_imm(b, i->format[0], 8 + 3 * c), 7);
      nir_def *value = nir_imm_int(b, 0);
      for (unsigned k = 0; k < 4; k++)
         value = nir_bcsel(b, nir_ieq_imm(b, select, k), channels[k], value);
      value = nir_bcsel(b, nir_ieq_imm(b, select, APEX_SWIZZLE_1), one, value);
      /* sRGB applies to color channels, never to alpha. */
      if (c < 3)
         value = nir_bcsel(b, nir_iand(b, srgb, nir_ult_imm(b, select, 3)),
                           srgb_to_linear(b, value), value);
      out[c] = value;
   }
   return nir_vec(b, out, 4);
}

struct sampler_info {
   nir_def *flags, *bias, *min_lod, *max_lod, *border[4];
};

static struct sampler_info
load_sampler(nir_builder *b, nir_def *row)
{
   struct sampler_info s;
   s.flags = row_word(b, row, 0);
   s.bias = row_word(b, row, 1);
   s.min_lod = row_word(b, row, 2);
   s.max_lod = row_word(b, row, 3);
   for (unsigned c = 0; c < 4; c++)
      s.border[c] = row_word(b, row, 4 + c);
   return s;
}

static nir_def *
flag(nir_builder *b, nir_def *flags, unsigned bit)
{
   return nir_ine_imm(b, nir_iand_imm(b, flags, 1u << bit), 0);
}

/* Applies a VkSamplerAddressMode to integer texel index `i` in [0, size),
 * given period = i mod 2 * size. Returns the wrapped index; `outside` is set
 * for border texels. */
static nir_def *
wrap(nir_builder *b, nir_def *mode, nir_def *i, nir_def *period, nir_def *size, nir_def **outside)
{
   nir_def *repeat = nir_bcsel(b, nir_ilt(b, period, size), period, nir_isub(b, period, size));
   nir_def *mirror = nir_bcsel(b, nir_ilt(b, period, size), period,
                               nir_isub(b, nir_isub(b, nir_ishl_imm(b, size, 1), period), nir_imm_int(b, 1)));
   nir_def *clamp = nir_imin(b, nir_imax(b, i, nir_imm_int(b, 0)), nir_iadd_imm(b, size, -1));
   nir_def *once = nir_bcsel(b, nir_ilt_imm(b, i, 0), nir_isub(b, nir_imm_int(b, -1), i), i);
   nir_def *mirror_clamp = nir_imin(b, once, nir_iadd_imm(b, size, -1));
   nir_def *result = repeat;
   result = nir_bcsel(b, nir_ieq_imm(b, mode, VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT), mirror, result);
   result = nir_bcsel(b, nir_ieq_imm(b, mode, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE), clamp, result);
   result = nir_bcsel(b, nir_ieq_imm(b, mode, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER), clamp, result);
   result = nir_bcsel(b, nir_ieq_imm(b, mode, VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE), mirror_clamp, result);
   nir_def *out_of_range = nir_ior(b, nir_ilt_imm(b, i, 0), nir_ige(b, i, size));
   *outside = nir_ior(b, *outside, nir_iand(b, out_of_range,
      nir_ieq_imm(b, mode, VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER)));
   return result;
}

/* One texel of level `level` (absolute) at integer x, y, z/layer after
 * addressing; border texels return the sampler border color. */
static nir_def *
tap(nir_builder *b, const struct image_info *img, const struct sampler_info *s, nir_def *level,
    nir_def *offset, nir_def *x, nir_def *y, nir_def *z, nir_def *border)
{
   nir_def *pitch = row_pitch(b, img, level);
   nir_def *plane = plane_bytes(b, img, level);
   nir_def *bytes = nir_iadd(b, offset, nir_iadd(b, nir_imul(b, z, plane),
      nir_iadd(b, nir_imul(b, y, pitch), nir_imul(b, x, img->bytes))));
   nir_push_if(b, border);
   nir_def *border_value = nir_vec4(b, s->border[0], s->border[1], s->border[2], s->border[3]);
   nir_push_else(b, NULL);
   nir_def *value = decode_texel(b, img, nir_build_addr_iadd(b, img->address,
      nir_address_format_2x32bit_global, nir_var_mem_global, bytes));
   nir_pop_if(b, NULL);
   return nir_if_phi(b, border_value, value);
}

static nir_def *
compare(nir_builder *b, nir_def *op, nir_def *reference, nir_def *texel)
{
   /* VkCompareOp bit 0 less, bit 1 equal, bit 2 greater. */
   nir_def *lt = nir_flt(b, reference, texel), *eq = nir_feq(b, reference, texel);
   nir_def *gt = nir_flt(b, texel, reference);
   nir_def *pass = nir_ior(b, nir_iand(b, nir_ine_imm(b, nir_iand_imm(b, op, 1), 0), lt),
                           nir_ior(b, nir_iand(b, nir_ine_imm(b, nir_iand_imm(b, op, 2), 0), eq),
                                   nir_iand(b, nir_ine_imm(b, nir_iand_imm(b, op, 4), 0), gt)));
   return nir_b2f32(b, pass);
}

struct sample_request {
   nir_def *coord[3];         /* normalized (or unnormalized) texel-space inputs */
   nir_def *layer;            /* integer array layer (absolute within view) */
   nir_def *offset[3];        /* integer texel offsets */
   nir_def *reference;        /* depth reference or NULL */
   unsigned dims;             /* 1, 2 or 3 */
   bool integer;
};

/* Samples one or two mip levels (relative to the view) with nearest or
 * linear taps. Taps and levels are loops so the decoder is emitted once.
 * Returns RGBA FP32 bits, integers for integer formats, or with `gather`
 * set, component `component` of the four bilinear taps in gather order. */
static nir_def *
filter(nir_builder *b, const struct image_info *img, const struct sampler_info *s,
       const struct sample_request *r, nir_def *level0, nir_def *two_levels, nir_def *level_weight,
       nir_def *linear, bool gather, unsigned component)
{
   nir_function_impl *impl = b->impl;
   const struct glsl_type *vec4 = glsl_vec4_type();
   nir_variable *sum = nir_local_variable_create(impl, vec4, "sum");
   nir_variable *low = nir_local_variable_create(impl, vec4, "minimum");
   nir_variable *high = nir_local_variable_create(impl, vec4, "maximum");
   nir_variable *gathered = nir_local_variable_create(impl, vec4, "gathered");
   nir_variable *level_var = nir_local_variable_create(impl, glsl_uint_type(), "level_iteration");
   nir_variable *tap_var = nir_local_variable_create(impl, glsl_uint_type(), "tap");
   nir_def *zero4 = nir_imm_zero(b, 4, 32);
   nir_store_var(b, sum, zero4, 0xf);
   nir_store_var(b, low, nir_imm_vec4(b, INFINITY, INFINITY, INFINITY, INFINITY), 0xf);
   nir_store_var(b, high, nir_imm_vec4(b, -INFINITY, -INFINITY, -INFINITY, -INFINITY), 0xf);
   nir_store_var(b, gathered, zero4, 0xf);
   nir_store_var(b, level_var, nir_imm_int(b, 0), 1);
   nir_def *unnormalized = flag(b, s->flags, 20);
   nir_def *layer = r->layer ? nir_iadd(b, img->base_layer, r->layer) : img->base_layer;

   nir_push_loop(b);
   nir_def *iteration = nir_load_var(b, level_var);
   nir_push_if(b, nir_uge(b, iteration, nir_bcsel(b, two_levels, nir_imm_int(b, 2), nir_imm_int(b, 1))));
   nir_jump(b, nir_jump_break);
   nir_pop_if(b, NULL);
   nir_def *level = nir_umin(b, nir_iadd(b, level0, iteration), nir_iadd_imm(b, img->levels, -1));
   nir_def *lw = nir_bcsel(b, two_levels,
      nir_bcsel(b, nir_ieq_imm(b, iteration, 0), nir_fsub(b, nir_imm_float(b, 1.0f), level_weight),
                level_weight), nir_imm_float(b, 1.0f));
   nir_def *absolute = nir_iadd(b, img->base_level, level);
   nir_def *offset = level_offset(b, img, absolute);
   nir_def *size[3] = {minify(b, img->width, absolute), minify(b, img->height, absolute),
                       minify(b, img->depth, absolute)};
   nir_def *base[3], *frac[3], *period[3];
   for (unsigned a = 0; a < r->dims; a++) {
      nir_def *t = nir_bcsel(b, unnormalized, r->coord[a], nir_fmul(b, r->coord[a], nir_u2f32(b, size[a])));
      nir_def *shifted = nir_bcsel(b, linear, nir_fadd_imm(b, t, -0.5f), t);
      nir_def *f = nir_ffloor(b, shifted);
      base[a] = nir_iadd(b, nir_f2i32(b, f), r->offset[a] ? r->offset[a] : nir_imm_int(b, 0));
      frac[a] = nir_fsub(b, shifted, f);
      /* One division per level and axis; each tap steps it by at most one. */
      period[a] = nir_imod(b, base[a], nir_ishl_imm(b, size[a], 1));
   }
   nir_def *taps = nir_bcsel(b, linear, nir_imm_int(b, 1 << r->dims), nir_imm_int(b, 1));
   nir_store_var(b, tap_var, nir_imm_int(b, 0), 1);
   nir_push_loop(b);
   nir_def *t = nir_load_var(b, tap_var);
   nir_push_if(b, nir_uge(b, t, taps));
   nir_jump(b, nir_jump_break);
   nir_pop_if(b, NULL);
   nir_def *index[3] = {nir_imm_int(b, 0), nir_imm_int(b, 0), nir_imm_int(b, 0)};
   nir_def *weight = lw, *border = nir_imm_false(b);
   for (unsigned a = 0; a < r->dims; a++) {
      nir_def *upper = nir_ine_imm(b, nir_iand_imm(b, nir_ushr_imm(b, t, a), 1), 0);
      nir_def *mode = nir_iand_imm(b, nir_ushr_imm(b, s->flags, 4 + 4 * a), 7);
      nir_def *next = nir_iadd_imm(b, period[a], 1);
      next = nir_bcsel(b, nir_ieq(b, next, nir_ishl_imm(b, size[a], 1)), nir_imm_int(b, 0), next);
      index[a] = wrap(b, mode, nir_iadd(b, base[a], nir_b2i32(b, upper)),
                      nir_bcsel(b, upper, next, period[a]), size[a], &border);
      nir_def *w = nir_bcsel(b, upper, frac[a], nir_fsub(b, nir_imm_float(b, 1.0f), frac[a]));
      weight = nir_bcsel(b, linear, nir_fmul(b, weight, w), weight);
   }
   nir_def *z = img->three_d ? index[2] : layer;
   nir_def *texel = tap(b, img, s, absolute, offset, index[0], index[1], z, border);
   if (r->reference) {
      nir_def *op = nir_iand_imm(b, nir_ushr_imm(b, s->flags, 17), 7);
      texel = nir_vec4(b, compare(b, op, r->reference, nir_channel(b, texel, 0)),
                       nir_imm_float(b, 0.0f), nir_imm_float(b, 0.0f), nir_imm_float(b, 1.0f));
   }
   if (gather) {
      /* Gather order: (i0,j1), (i1,j1), (i1,j0), (i0,j0); tap bit 0 = i1, bit 1 = j1. */
      nir_def *slot = nir_bcsel(b, nir_ieq_imm(b, t, 0), nir_imm_int(b, 3),
                      nir_bcsel(b, nir_ieq_imm(b, t, 1), nir_imm_int(b, 2),
                      nir_bcsel(b, nir_ieq_imm(b, t, 2), nir_imm_int(b, 0), nir_imm_int(b, 1))));
      nir_def *old = nir_load_var(b, gathered), *parts[4];
      for (unsigned k = 0; k < 4; k++)
         parts[k] = nir_bcsel(b, nir_ieq_imm(b, slot, k), nir_channel(b, texel, component),
                              nir_channel(b, old, k));
      nir_store_var(b, gathered, nir_vec(b, parts, 4), 0xf);
   } else if (r->integer) {
      nir_store_var(b, sum, texel, 0xf);
   } else {
      /* Taps with zero weight do not join min/max reduction. */
      nir_def *active = nir_fneu(b, weight, nir_imm_float(b, 0.0f));
      nir_store_var(b, sum, nir_ffma(b, texel, weight, nir_load_var(b, sum)), 0xf);
      nir_store_var(b, low, nir_bcsel(b, active, nir_fmin(b, nir_load_var(b, low), texel),
                                      nir_load_var(b, low)), 0xf);
      nir_store_var(b, high, nir_bcsel(b, active, nir_fmax(b, nir_load_var(b, high), texel),
                                       nir_load_var(b, high)), 0xf);
   }
   nir_store_var(b, tap_var, nir_iadd_imm(b, t, 1), 1);
   nir_pop_loop(b, NULL);
   nir_store_var(b, level_var, nir_iadd_imm(b, iteration, 1), 1);
   nir_pop_loop(b, NULL);
   if (gather)
      return nir_load_var(b, gathered);
   if (r->integer)
      return nir_load_var(b, sum);
   nir_def *reduction = nir_iand_imm(b, nir_ushr_imm(b, s->flags, 21), 3);
   return nir_bcsel(b, nir_ieq_imm(b, reduction, 1), nir_load_var(b, low),
                    nir_bcsel(b, nir_ieq_imm(b, reduction, 2), nir_load_var(b, high),
                              nir_load_var(b, sum)));
}

/* Cube direction to face coordinates in [0, 1] and face index. */
static void
cube_face(nir_builder *b, nir_def *dir, nir_def **s, nir_def **t, nir_def **face)
{
   nir_def *x = nir_channel(b, dir, 0), *y = nir_channel(b, dir, 1), *z = nir_channel(b, dir, 2);
   nir_def *ax = nir_fabs(b, x), *ay = nir_fabs(b, y), *az = nir_fabs(b, z);
   nir_def *is_x = nir_iand(b, nir_fge(b, ax, ay), nir_fge(b, ax, az));
   nir_def *is_y = nir_iand(b, nir_inot(b, is_x), nir_fge(b, ay, az));
   nir_def *neg_x = nir_flt_imm(b, x, 0.0f), *neg_y = nir_flt_imm(b, y, 0.0f), *neg_z = nir_flt_imm(b, z, 0.0f);
   /* Vulkan table: +X (-z,-y), -X (z,-y), +Y (x,z), -Y (x,-z), +Z (x,-y), -Z (-x,-y). */
   nir_def *sc = nir_bcsel(b, is_x, nir_bcsel(b, neg_x, z, nir_fneg(b, z)),
                           nir_bcsel(b, is_y, x, nir_bcsel(b, neg_z, nir_fneg(b, x), x)));
   nir_def *tc = nir_bcsel(b, is_x, nir_fneg(b, y),
                           nir_bcsel(b, is_y, nir_bcsel(b, neg_y, nir_fneg(b, z), z), nir_fneg(b, y)));
   nir_def *ma = nir_bcsel(b, is_x, ax, nir_bcsel(b, is_y, ay, az));
   nir_def *half = nir_fdiv(b, nir_imm_float(b, 0.5f), ma);
   *s = nir_fadd_imm(b, nir_fmul(b, sc, half), 0.5f);
   *t = nir_fadd_imm(b, nir_fmul(b, tc, half), 0.5f);
   *face = nir_bcsel(b, is_x, nir_b2i32(b, neg_x),
                     nir_bcsel(b, is_y, nir_iadd_imm(b, nir_b2i32(b, neg_y), 2),
                               nir_iadd_imm(b, nir_b2i32(b, neg_z), 4)));
}

static nir_def *
tex_src(nir_tex_instr *tex, nir_tex_src_type type)
{
   int index = nir_tex_instr_src_index(tex, type);
   return index < 0 ? NULL : tex->src[index].src.ssa;
}

static bool
lower_tex(nir_builder *b, nir_tex_instr *tex, struct texture_lowering *ctx)
{
   b->cursor = nir_before_instr(&tex->instr);
   nir_def *texture_deref = tex_src(tex, nir_tex_src_texture_deref);
   nir_def *sampler_deref = tex_src(tex, nir_tex_src_sampler_deref);
   if (!texture_deref || tex->is_sparse || tex->def.bit_size != 32) {
      ctx->invalid = true;
      return false;
   }
   nir_def *image_row = deref_row(b, ctx, nir_instr_as_deref(nir_def_instr(texture_deref)), false);
   bool buffer = tex->sampler_dim == GLSL_SAMPLER_DIM_BUF;
   if (buffer) {
      /* Texel buffers: address, byte range, elements, format words. */
      nir_def *coord = tex_src(tex, nir_tex_src_coord);
      if (tex->op == nir_texop_txs) {
         nir_def_replace(&tex->def, row_word(b, image_row, 3));
         return true;
      }
      struct image_info img = {0};
      img.address = nir_vec2(b, row_word(b, image_row, 0), row_word(b, image_row, 1));
      for (unsigned k = 0; k < 3; k++)
         img.format[k] = row_word(b, image_row, 4 + k);
      img.bytes = nir_iand_imm(b, img.format[0], 31);
      nir_def *inside = nir_ult(b, coord, row_word(b, image_row, 3));
      nir_def *zero = nir_imm_zero(b, 4, 32);
      nir_push_if(b, inside);
      nir_def *value = decode_texel(b, &img, nir_build_addr_iadd(b, img.address,
         nir_address_format_2x32bit_global, nir_var_mem_global, nir_imul(b, coord, img.bytes)));
      nir_pop_if(b, NULL);
      nir_def_replace(&tex->def, nir_trim_vector(b, nir_if_phi(b, value, zero),
                                                 tex->def.num_components));
      return true;
   }
   bool cube = tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE;
   bool three_d = tex->sampler_dim == GLSL_SAMPLER_DIM_3D;
   unsigned dims = tex->sampler_dim == GLSL_SAMPLER_DIM_1D ? 1 : three_d ? 3 : 2;
   struct image_info img = load_image(b, image_row, three_d);
   /* Null descriptors have zero levels: queries return zero, reads zero. */
   nir_def *null = nir_ieq_imm(b, img.levels, 0);
   nir_def *lod_src = tex_src(tex, nir_tex_src_lod);
   switch (tex->op) {
   case nir_texop_txs: {
      nir_def *level = nir_iadd(b, img.base_level, lod_src ? lod_src : nir_imm_int(b, 0));
      nir_def *size[4] = {minify(b, img.width, level), minify(b, img.height, level),
                          three_d ? minify(b, img.depth, level) : img.layer_count, NULL};
      unsigned n = tex->def.num_components;
      if (tex->is_array)
         size[n - 1] = cube ? nir_udiv_imm(b, img.layer_count, 6) : img.layer_count;
      nir_def_replace(&tex->def, nir_bcsel(b, null, nir_imm_zero(b, n, 32), nir_vec(b, size, n)));
      return true;
   }
   case nir_texop_query_levels:
      nir_def_replace(&tex->def, img.levels);
      return true;
   case nir_texop_texture_samples:
      nir_def_replace(&tex->def, img.samples);
      return true;
   case nir_texop_samples_identical:
      nir_def_replace(&tex->def, nir_imm_true(b));
      return true;
   default:
      break;
   }
   /* Level arithmetic of null descriptors stays within one level. */
   img.levels = nir_umax(b, img.levels, nir_imm_int(b, 1));
   nir_def *coord = tex_src(tex, nir_tex_src_coord);
   bool integer = nir_alu_type_get_base_type(tex->dest_type) != nir_type_float;
   if (tex->op == nir_texop_txf || tex->op == nir_texop_txf_ms) {
      /* Fetches ignore samplers; out-of-range texels read zero. */
      nir_def *level = lod_src ? lod_src : nir_imm_int(b, 0);
      nir_def *absolute = nir_iadd(b, img.base_level, level);
      nir_def *offset_src = tex_src(tex, nir_tex_src_offset);
      nir_def *index[3], *inside = nir_iand(b, nir_inot(b, null), nir_ult(b, level, img.levels));
      nir_def *size[3] = {minify(b, img.width, absolute), minify(b, img.height, absolute),
                          minify(b, img.depth, absolute)};
      for (unsigned a = 0; a < dims; a++) {
         index[a] = nir_channel(b, coord, a);
         if (offset_src)
            index[a] = nir_iadd(b, index[a], nir_channel(b, offset_src, a));
         inside = nir_iand(b, inside, nir_ult(b, index[a], size[a]));
      }
      for (unsigned a = dims; a < 3; a++)
         index[a] = nir_imm_int(b, 0);
      /* Sample s of texel x is element x * samples + s. */
      nir_def *ms_index = tex_src(tex, nir_tex_src_ms_index);
      nir_def *sample = ms_index ? ms_index : nir_imm_int(b, 0);
      inside = nir_iand(b, inside, nir_ult(b, sample, img.samples));
      index[0] = nir_iadd(b, nir_imul(b, index[0], img.samples), sample);
      nir_def *z = three_d ? index[2] : img.base_layer;
      if (tex->is_array) {
         nir_def *layer = nir_channel(b, coord, dims);
         inside = nir_iand(b, inside, nir_ult(b, layer, img.layer_count));
         z = nir_iadd(b, z, layer);
      }
      nir_def *zero = nir_imm_zero(b, 4, 32);
      nir_push_if(b, inside);
      nir_def *offset = level_offset(b, &img, absolute);
      nir_def *pitch = row_pitch(b, &img, absolute), *plane = plane_bytes(b, &img, absolute);
      nir_def *bytes = nir_iadd(b, offset, nir_iadd(b, nir_imul(b, z, plane),
         nir_iadd(b, nir_imul(b, index[1], pitch), nir_imul(b, index[0], img.bytes))));
      nir_def *value = decode_texel(b, &img, nir_build_addr_iadd(b, img.address,
         nir_address_format_2x32bit_global, nir_var_mem_global, bytes));
      nir_pop_if(b, NULL);
      nir_def_replace(&tex->def, nir_trim_vector(b, nir_if_phi(b, value, zero),
                                                 tex->def.num_components));
      return true;
   }
   if (!sampler_deref) {
      ctx->invalid = true;
      return false;
   }
   struct sampler_info s = load_sampler(b, deref_row(b, ctx, nir_instr_as_deref(nir_def_instr(sampler_deref)), true));
   struct sample_request r = {.dims = dims, .integer = integer};
   nir_def *layer = NULL;
   if (cube) {
      nir_def *face_s, *face_t, *face;
      cube_face(b, coord, &face_s, &face_t, &face);
      r.coord[0] = face_s;
      r.coord[1] = face_t;
      layer = face;
      if (tex->is_array)
         layer = nir_iadd(b, face, nir_imul_imm(b, nir_f2i32(b, nir_fround_even(b,
            nir_channel(b, coord, 3))), 6));
   } else {
      for (unsigned a = 0; a < dims; a++)
         r.coord[a] = nir_channel(b, coord, a);
      if (tex->is_array) {
         nir_def *l = nir_f2i32(b, nir_fround_even(b, nir_channel(b, coord, dims)));
         layer = nir_imin(b, nir_imax(b, l, nir_imm_int(b, 0)), nir_iadd_imm(b, img.layer_count, -1));
      }
   }
   r.layer = layer;
   nir_def *offset_src = tex_src(tex, nir_tex_src_offset);
   for (unsigned a = 0; a < 3; a++)
      r.offset[a] = offset_src && a < dims && !cube ? nir_channel(b, offset_src, a) : NULL;
   r.reference = tex_src(tex, nir_tex_src_comparator);

   /* Level of detail: explicit, gradient or implicit through quad derivatives
    * (fragment kernels only; other stages use level 0). */
   nir_def *lambda;
   nir_def *ddx = tex_src(tex, nir_tex_src_ddx), *ddy = tex_src(tex, nir_tex_src_ddy);
   if (tex->op == nir_texop_txl) {
      lambda = lod_src;
   } else if (tex->op == nir_texop_txd || ((tex->op == nir_texop_tex || tex->op == nir_texop_txb ||
               tex->op == nir_texop_lod || tex->op == nir_texop_tg4) &&
              b->shader->info.stage == MESA_SHADER_FRAGMENT)) {
      nir_def *base_size[3] = {nir_u2f32(b, minify(b, img.width, img.base_level)),
                               nir_u2f32(b, minify(b, img.height, img.base_level)),
                               nir_u2f32(b, minify(b, img.depth, img.base_level))};
      nir_def *rho = nir_imm_float(b, 0.0f);
      for (unsigned axis = 0; axis < 2; axis++) {
         nir_def *length = nir_imm_float(b, 0.0f);
         for (unsigned a = 0; a < dims; a++) {
            nir_def *c = cube ? r.coord[a] : nir_channel(b, coord, a);
            nir_def *d = tex->op == nir_texop_txd ? nir_channel(b, axis ? ddy : ddx, a) :
                         axis ? nir_ddy(b, c) : nir_ddx(b, c);
            nir_def *scaled = nir_bcsel(b, flag(b, s.flags, 20), d, nir_fmul(b, d, base_size[a]));
            length = nir_ffma(b, scaled, scaled, length);
         }
         rho = nir_fmax(b, rho, nir_fsqrt(b, length));
      }
      lambda = nir_flog2(b, rho);
   } else {
      lambda = nir_imm_float(b, 0.0f);
   }
   nir_def *bias = tex_src(tex, nir_tex_src_bias);
   if (tex->op != nir_texop_txl && tex->op != nir_texop_txf)
      lambda = nir_fadd(b, lambda, s.bias);
   if (bias)
      lambda = nir_fadd(b, lambda, bias);
   nir_def *min_lod = tex_src(tex, nir_tex_src_min_lod);
   nir_def *low = min_lod ? nir_fmax(b, s.min_lod, min_lod) : s.min_lod;
   lambda = nir_fmin(b, nir_fmax(b, lambda, low), s.max_lod);
   nir_def *max_level = nir_u2f32(b, nir_iadd_imm(b, img.levels, -1));
   if (tex->op == nir_texop_lod) {
      nir_def *level = nir_fmin(b, nir_fmax(b, lambda, nir_imm_float(b, 0.0f)), max_level);
      nir_def_replace(&tex->def, nir_bcsel(b, null, nir_imm_zero(b, 2, 32), nir_vec2(b, level, lambda)));
      return true;
   }
   /* Magnification at lambda <= 0 uses the mag filter; minification the min
    * filter, then nearest or linear between mip levels. */
   nir_def *minifying = nir_flt(b, nir_imm_float(b, 0.0f), lambda);
   nir_def *linear = nir_bcsel(b, minifying, flag(b, s.flags, 1), flag(b, s.flags, 0));
   if (integer)
      linear = nir_imm_false(b);
   nir_def *level_f = nir_fmin(b, nir_fmax(b, lambda, nir_imm_float(b, 0.0f)), max_level);
   nir_def *mip_linear = nir_iand(b, flag(b, s.flags, 2), minifying);
   if (integer)
      mip_linear = nir_imm_false(b);
   nir_def *nearest_level = nir_f2u32(b, nir_bcsel(b, minifying,
      nir_fmin(b, nir_ffloor(b, nir_fadd_imm(b, level_f, 0.5f)), max_level), nir_imm_float(b, 0.0f)));
   nir_def *level0 = nir_bcsel(b, mip_linear, nir_f2u32(b, nir_ffloor(b, level_f)), nearest_level);
   if (tex->op == nir_texop_tg4) {
      /* Gather reads the four bilinear taps of level 0. */
      nir_def *parts = filter(b, &img, &s, &r, nir_imm_int(b, 0), nir_imm_false(b),
                              nir_imm_float(b, 0.0f), nir_imm_true(b), true, tex->component);
      nir_def_replace(&tex->def, nir_bcsel(b, null, nir_imm_zero(b, 4, 32), parts));
      return true;
   }
   nir_def *weight = nir_fsub(b, level_f, nir_ffloor(b, level_f));
   nir_def *result = filter(b, &img, &s, &r, level0, mip_linear, weight, linear, false, 0);
   result = nir_bcsel(b, null, nir_imm_zero(b, 4, 32), result);
   if (r.reference)
      result = nir_channel(b, result, 0);
   nir_def_replace(&tex->def, nir_trim_vector(b, result, tex->def.num_components));
   return true;
}

/* Stores RGBA components (FP32 bits or integers) as one texel of the format
 * words: the inverse of decode_texel. Sub-word texels merge into their word
 * with atomics so neighbouring invocations never overwrite each other. */
static void
encode_texel(nir_builder *b, const struct image_info *i, nir_def *address, nir_def *value)
{
   nir_def *words[4];
   for (unsigned w = 0; w < 4; w++)
      words[w] = nir_imm_int(b, 0);
   for (unsigned k = 0; k < 4; k++) {
      nir_def *field = nir_iand_imm(b, nir_ushr_imm(b, i->format[1 + k / 2], 16 * (k % 2)), 0xffff);
      nir_def *shift = nir_iand_imm(b, field, 127);
      nir_def *size = nir_iand_imm(b, nir_ushr_imm(b, field, 7), 63);
      nir_def *type = nir_ushr_imm(b, field, 13);
      /* The API component whose swizzle selects stored channel k. */
      nir_def *v = nir_imm_int(b, 0);
      for (unsigned c = 0; c < 4; c++) {
         nir_def *select = nir_iand_imm(b, nir_ushr_imm(b, i->format[0], 8 + 3 * c), 7);
         v = nir_bcsel(b, nir_ieq_imm(b, select, k), nir_channel(b, value, c), v);
      }
      nir_def *full = nir_ieq_imm(b, size, 32);
      nir_def *mask = nir_bcsel(b, full, nir_imm_int(b, ~0u),
                                nir_isub(b, nir_ishl(b, nir_imm_int(b, 1), size), nir_imm_int(b, 1)));
      nir_def *half = nir_ushr_imm(b, mask, 1);
      nir_def *unorm = nir_f2u32(b, nir_fround_even(b, nir_fmul(b, nir_fsat(b, v), nir_u2f32(b, mask))));
      nir_def *snorm = nir_f2i32(b, nir_fround_even(b, nir_fmul(b, nir_fmin(b, nir_fmax(b, v,
         nir_imm_float(b, -1.0f)), nir_imm_float(b, 1.0f)), nir_u2f32(b, half))));
      nir_def *uint = nir_umin(b, v, mask);
      nir_def *sint = nir_imin(b, nir_imax(b, v, nir_ineg(b, nir_iadd_imm(b, half, 1))), half);
      nir_def *fp = nir_bcsel(b, nir_ieq_imm(b, size, 16), apex_float_to_half(b, v), v);
      nir_def *bits = nir_imm_int(b, 0);
      bits = nir_bcsel(b, nir_ieq_imm(b, type, APEX_CHANNEL_UNORM), unorm, bits);
      bits = nir_bcsel(b, nir_ieq_imm(b, type, APEX_CHANNEL_SNORM), snorm, bits);
      bits = nir_bcsel(b, nir_ieq_imm(b, type, APEX_CHANNEL_UINT), uint, bits);
      bits = nir_bcsel(b, nir_ieq_imm(b, type, APEX_CHANNEL_SINT), sint, bits);
      bits = nir_bcsel(b, nir_ieq_imm(b, type, APEX_CHANNEL_FLOAT), fp, bits);
      bits = nir_ishl(b, nir_iand(b, bits, mask), nir_iand_imm(b, shift, 31));
      for (unsigned w = 0; w < 4; w++)
         words[w] = nir_ior(b, words[w], nir_bcsel(b, nir_ieq_imm(b, nir_ushr_imm(b, shift, 5), w),
                                                    bits, nir_imm_int(b, 0)));
   }
   nir_def *byte = nir_channel(b, address, 0);
   nir_push_if(b, nir_ult_imm(b, i->bytes, 4));
   {
      nir_def *aligned = nir_vec2(b, nir_iand_imm(b, byte, ~3u), nir_channel(b, address, 1));
      nir_def *shift = nir_ishl_imm(b, nir_iand_imm(b, byte, 3), 3);
      nir_def *mask = nir_ishl(b, nir_isub(b, nir_ishl(b, nir_imm_int(b, 1), nir_ishl_imm(b, i->bytes, 3)),
                                           nir_imm_int(b, 1)), shift);
      nir_global_atomic_2x32(b, 32, aligned, nir_inot(b, mask), .atomic_op = nir_atomic_op_iand);
      nir_global_atomic_2x32(b, 32, aligned, nir_iand(b, nir_ishl(b, words[0], shift), mask),
                             .atomic_op = nir_atomic_op_ior);
   }
   nir_push_else(b, NULL);
   for (unsigned w = 0; w < 4; w++) {
      nir_push_if(b, nir_ult(b, nir_imm_int(b, w * 4), i->bytes));
      nir_store_global_2x32(b, words[w], nir_build_addr_iadd_imm(b, address,
         nir_address_format_2x32bit_global, nir_var_mem_global, w * 4), .align_mul = 4);
      nir_pop_if(b, NULL);
   }
   nir_pop_if(b, NULL);
}

/* Storage images use the sampled-image descriptor (level base_level only);
 * storage texel buffers use the texel-buffer row. Loads decode and stores
 * encode through the view's format words, so shader formats are optional. */
static bool
lower_storage(nir_builder *b, nir_intrinsic_instr *i, struct texture_lowering *ctx)
{
   bool load = i->intrinsic == nir_intrinsic_image_deref_load;
   bool store = i->intrinsic == nir_intrinsic_image_deref_store;
   bool swap = i->intrinsic == nir_intrinsic_image_deref_atomic_swap;
   bool atomic = swap || i->intrinsic == nir_intrinsic_image_deref_atomic;
   bool size = i->intrinsic == nir_intrinsic_image_deref_size;
   bool samples = i->intrinsic == nir_intrinsic_image_deref_samples;
   if (!load && !store && !atomic && !size && !samples)
      return false;
   b->cursor = nir_before_instr(&i->instr);
   if ((!store && i->def.bit_size != 32) ||
       (store && i->src[3].ssa->bit_size != 32)) {
      ctx->invalid = true;
      return false;
   }
   /* The binding must hold storage images, or texel buffers for buffer images. */
   enum glsl_sampler_dim dim = nir_intrinsic_image_dim(i);
   nir_variable *var = nir_deref_instr_get_variable(nir_src_as_deref(i->src[0]));
   const struct apex_set_layout *layout = var && var->data.descriptor_set < ctx->program->set_count ?
      ctx->program->set_layouts[var->data.descriptor_set] : NULL;
   VkDescriptorType type = layout && var->data.binding < layout->binding_count ?
      layout->bindings[var->data.binding].type : VK_DESCRIPTOR_TYPE_MAX_ENUM;
   if (dim == GLSL_SAMPLER_DIM_BUF ? type != VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER &&
                                     type != VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER :
                                     type != VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) {
      ctx->invalid = true;
      return false;
   }
   nir_def *row = deref_row(b, ctx, nir_src_as_deref(i->src[0]), false);
   bool array = nir_intrinsic_image_array(i);
   struct image_info img;
   nir_def *inside, *address;
   nir_def *coord = size || samples ? NULL : i->src[1].ssa;
   if (dim == GLSL_SAMPLER_DIM_BUF) {
      /* Texel buffers: address, byte range, elements, format words. */
      img = (struct image_info){0};
      img.address = nir_vec2(b, row_word(b, row, 0), row_word(b, row, 1));
      for (unsigned k = 0; k < 3; k++)
         img.format[k] = row_word(b, row, 4 + k);
      img.bytes = nir_iand_imm(b, img.format[0], 31);
      if (size) {
         nir_def_replace(&i->def, nir_trim_vector(b, row_word(b, row, 3), i->def.num_components));
         return true;
      }
      nir_def *x = nir_channel(b, coord, 0);
      inside = nir_ult(b, x, row_word(b, row, 3));
      address = nir_build_addr_iadd(b, img.address, nir_address_format_2x32bit_global,
                                    nir_var_mem_global, nir_imul(b, x, img.bytes));
   } else {
      bool three_d = dim == GLSL_SAMPLER_DIM_3D, cube = dim == GLSL_SAMPLER_DIM_CUBE;
      unsigned dims = dim == GLSL_SAMPLER_DIM_1D ? 1 : three_d ? 3 : 2;
      img = load_image(b, row, three_d);
      /* Storage images have one sample; null descriptors zero levels. */
      nir_def *null = nir_ieq_imm(b, img.levels, 0);
      if (samples) {
         nir_def_replace(&i->def, nir_b2i32(b, nir_inot(b, null)));
         return true;
      }
      nir_def *level = img.base_level;
      nir_def *extent[3] = {minify(b, img.width, level), minify(b, img.height, level),
                            minify(b, img.depth, level)};
      if (size) {
         nir_def *out[4];
         for (unsigned a = 0; a < dims; a++)
            out[a] = extent[a];
         unsigned n = dims;
         if (array || cube) {
            nir_def *layers = cube ? nir_udiv_imm(b, img.layer_count, 6) : img.layer_count;
            if (array)
               out[n++] = layers;
         }
         for (unsigned a = n; a < 4; a++)
            out[a] = nir_imm_int(b, 1);
         nir_def_replace(&i->def, nir_bcsel(b, null, nir_imm_zero(b, i->def.num_components, 32),
                                            nir_vec(b, out, i->def.num_components)));
         return true;
      }
      nir_def *index[3];
      inside = nir_inot(b, null);
      for (unsigned a = 0; a < 3; a++) {
         index[a] = a < dims ? nir_channel(b, coord, a) : nir_imm_int(b, 0);
         if (a < dims)
            inside = nir_iand(b, inside, nir_ult(b, index[a], extent[a]));
      }
      nir_def *z = index[2];
      if (!three_d) {
         /* Cube images address a face (or face of a layer) as a layer. */
         nir_def *layer = array || cube ? nir_channel(b, coord, dims) : nir_imm_int(b, 0);
         inside = nir_iand(b, inside, nir_ult(b, layer, img.layer_count));
         z = nir_iadd(b, img.base_layer, layer);
      }
      nir_def *offset = level_offset(b, &img, level);
      nir_def *bytes = nir_iadd(b, offset, nir_iadd(b, nir_imul(b, z, plane_bytes(b, &img, level)),
         nir_iadd(b, nir_imul(b, index[1], row_pitch(b, &img, level)), nir_imul(b, index[0], img.bytes))));
      address = nir_build_addr_iadd(b, img.address, nir_address_format_2x32bit_global,
                                    nir_var_mem_global, bytes);
   }
   nir_def *zero = nir_imm_zero(b, load ? 4 : 1, 32);
   nir_push_if(b, inside);
   nir_def *result = NULL;
   if (load) {
      result = decode_texel(b, &img, address);
   } else if (store) {
      nir_def *value = i->src[3].ssa;
      nir_def *channels[4];
      for (unsigned c = 0; c < 4; c++)
         channels[c] = c < value->num_components ? nir_channel(b, value, c) : nir_imm_int(b, 0);
      encode_texel(b, &img, address, nir_vec(b, channels, 4));
   } else if (swap) {
      result = nir_global_atomic_swap_2x32(b, 32, address, i->src[3].ssa, i->src[4].ssa,
                                           .atomic_op = nir_intrinsic_atomic_op(i));
   } else {
      result = nir_global_atomic_2x32(b, 32, address, i->src[3].ssa,
                                      .atomic_op = nir_intrinsic_atomic_op(i));
   }
   nir_pop_if(b, NULL);
   if (store) {
      nir_instr_remove(&i->instr);
      return true;
   }
   nir_def *phi = nir_if_phi(b, result, zero);
   nir_def_replace(&i->def, nir_trim_vector(b, phi, i->def.num_components));
   return true;
}

bool
apex_lower_textures(struct apex_program *program, nir_shader *nir)
{
   struct texture_lowering ctx = {.program = program};
   struct util_dynarray list;
   util_dynarray_init(&list, NULL);
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_tex || instr->type == nir_instr_type_intrinsic)
               util_dynarray_append(&list, instr);
         }
      }
   }
   bool progress = false;
   nir_builder b = nir_builder_create(nir_shader_get_entrypoint(nir));
   util_dynarray_foreach(&list, nir_instr *, instr)
      progress |= (*instr)->type == nir_instr_type_tex ? lower_tex(&b, nir_instr_as_tex(*instr), &ctx) :
                  lower_storage(&b, nir_instr_as_intrinsic(*instr), &ctx);
   util_dynarray_fini(&list);
   nir_progress(progress, nir_shader_get_entrypoint(nir), nir_metadata_none);
   return !ctx.invalid;
}
