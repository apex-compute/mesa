/* SPDX-License-Identifier: MIT */
/* Texture and image instructions. Sampled images go to the texture unit with
 * the image and sampler descriptors of their table rows (apex_hw.h); texel
 * buffers and storage images address memory, storage images in the texture
 * unit's tiled layout, and convert texels through the view's format words
 * (apex_format.h). */
#include "apex_pipeline.h"
#include "apex_format.h"
#include "apex_hw.h"
#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"

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

/* The eight dwords of a descriptor row. */
static nir_def *
row_words(nir_builder *b, nir_def *row)
{
   return nir_load_ssbo(b, 8, 32, nir_imm_int(b, 0), nir_imul_imm(b, row, sizeof(union apex_descriptor)),
                        .align_mul = 32,
                        .access = ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER | ACCESS_CAN_SPECULATE);
}

/* Bits [lo, lo + width) of a descriptor, width at most 32. */
static nir_def *
bits(nir_builder *b, nir_def *words, unsigned lo, unsigned width)
{
   unsigned dword = lo / 32, shift = lo % 32;
   nir_def *x = nir_ushr_imm(b, nir_channel(b, words, dword), shift);
   if (shift + width > 32)
      x = nir_ior(b, x, nir_ishl_imm(b, nir_channel(b, words, dword + 1), 32 - shift));
   return width == 32 ? x : nir_iand_imm(b, x, BITFIELD_MASK(width));
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
   /* A combined descriptor keeps its sampler in the second row. */
   unsigned part = sampler_part && bl->type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER ? 1 : 0;
   return nir_bcsel(b, nir_ult_imm(b, index, bl->count),
                    nir_iadd_imm(b, nir_imul_imm(b, index, slots), base + part),
                    nir_imm_int(b, program->descriptor_count));
}

/* A texel format: the format words and texel bytes. */
struct image_info {
   nir_def *format[3];
   nir_def *bytes;
};

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

/* FP32 to a float with a 5-bit exponent (bias 15) and `mantissa` bits,
 * round-to-nearest-even: FP16 with a sign bit above, or the unsigned 11-
 * and 10-bit floats, where negative values become zero. */
nir_def *
apex_float_to_small(nir_builder *b, nir_def *f, unsigned mantissa, bool sign)
{
   unsigned shift = 23 - mantissa;
   nir_def *abs = nir_iand_imm(b, f, 0x7fffffff);
   /* Normal results: rebias the exponent, then round the low mantissa bits away. */
   nir_def *rebased = nir_iadd_imm(b, abs, -(112 << 23));
   nir_def *normal = nir_ushr_imm(b, nir_iadd(b, nir_iadd_imm(b, rebased, (1u << (shift - 1)) - 1),
                                              nir_iand_imm(b, nir_ushr_imm(b, rebased, shift), 1)), shift);
   /* Subnormal results are exact multiples of 2^(-14 - mantissa) after rounding. */
   nir_def *subnormal = nir_f2u32(b, nir_fround_even(b, nir_fmul_imm(b, abs, (double)(1u << (14 + mantissa)))));
   nir_def *nan = nir_ior_imm(b, nir_iand_imm(b, nir_ushr_imm(b, abs, shift), BITFIELD_MASK(mantissa)),
                              0x1fu << mantissa | 1u << (mantissa - 1));
   nir_def *result = nir_bcsel(b, nir_ult_imm(b, abs, 0x38800000), subnormal, normal);
   /* Finite overflow: FP16 rounds to infinity; the unsigned formats clamp
    * to their largest finite value, as GL and Mesa's packers define. */
   nir_def *overflow = sign ? nir_imm_int(b, 0x1fu << mantissa) :
                              nir_bcsel(b, nir_ieq_imm(b, abs, 0x7f800000), nir_imm_int(b, 0x1fu << mantissa),
                                        nir_imm_int(b, (0x1fu << mantissa) - 1));
   result = nir_bcsel(b, nir_uge_imm(b, abs, 0x47800000 - (1u << (22 - mantissa))), overflow, result);
   nir_def *is_nan = nir_uge_imm(b, abs, 0x7f800001);
   result = nir_bcsel(b, is_nan, nan, result);
   if (sign)
      return nir_ior(b, result, nir_iand_imm(b, nir_ushr_imm(b, f, 31 - (4 + mantissa + 1)),
                                            1u << (5 + mantissa)));
   return nir_bcsel(b, nir_iand(b, nir_ilt_imm(b, f, 0), nir_inot(b, is_nan)), nir_imm_int(b, 0), result);
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
   /* Power-of-two texels are naturally aligned (1/2-byte texels sit inside
    * one word); 3- and 6-byte texels may start mid-word and span words, so
    * the covering words are loaded and funnel-shifted to the texel start. */
   nir_def *byte = nir_channel(b, address, 0);
   nir_def *misalign = nir_iand_imm(b, byte, 3);
   nir_def *aligned = nir_vec2(b, nir_iand_imm(b, byte, ~3u), nir_channel(b, address, 1));
   nir_def *span = nir_iadd(b, i->bytes, misalign);
   nir_def *raw[5], *words[4], *zero = nir_imm_int(b, 0);
   for (unsigned k = 0; k < 5; k++) {
      nir_def *present = nir_ult(b, nir_imm_int(b, k * 4), span);
      nir_push_if(b, present);
      nir_def *loaded = nir_load_global_2x32(b, 1, 32,
         nir_build_addr_iadd_imm(b, aligned, nir_address_format_2x32bit_global, nir_var_mem_global,
                                 k * 4), .align_mul = 4);
      nir_pop_if(b, NULL);
      raw[k] = nir_if_phi(b, loaded, zero);
   }
   nir_def *shift = nir_ishl_imm(b, misalign, 3);
   nir_def *carry = nir_isub(b, nir_imm_int(b, 32), shift);
   for (unsigned k = 0; k < 4; k++)
      words[k] = nir_bcsel(b, nir_ieq_imm(b, misalign, 0), raw[k],
                           nir_ior(b, nir_ushr(b, raw[k], shift), nir_ishl(b, raw[k + 1], carry)));
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
      nir_def *fp = nir_bcsel(b, nir_ieq_imm(b, size, 16), apex_float_to_small(b, v, 10, true), v);
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

static nir_def *
tex_src(nir_tex_instr *tex, nir_tex_src_type type)
{
   int index = nir_tex_instr_src_index(tex, type);
   return index < 0 ? NULL : tex->src[index].src.ssa;
}

/* textureQueryLod: the implicit level of detail from quad derivatives
 * scaled by the base level's size, before (y) and after (x) the view's
 * level clamp. */
static nir_def *
query_lod(nir_builder *b, nir_tex_instr *tex, nir_def *image, nir_def *sampler)
{
   nir_def *coord = tex_src(tex, nir_tex_src_coord);
   bool cube = tex->sampler_dim == GLSL_SAMPLER_DIM_CUBE;
   unsigned dims = tex->sampler_dim == GLSL_SAMPLER_DIM_1D ? 1 :
                   tex->sampler_dim == GLSL_SAMPLER_DIM_3D || cube ? 3 : 2;
   nir_def *size[3] = {
      nir_u2f32(b, nir_iadd_imm(b, bits(b, image, APEX_HW_IMAGE_WIDTH, 14), 1)),
      nir_u2f32(b, nir_iadd_imm(b, bits(b, image, APEX_HW_IMAGE_HEIGHT, 14), 1)),
      nir_u2f32(b, nir_iadd_imm(b, bits(b, image, APEX_HW_IMAGE_DEPTH, 11), 1)),
   };
   nir_def *rho = nir_imm_float(b, 0.0f);
   for (unsigned axis = 0; axis < 2; axis++) {
      nir_def *length = nir_imm_float(b, 0.0f);
      for (unsigned a = 0; a < dims; a++) {
         nir_def *c = nir_channel(b, coord, a);
         nir_def *d = axis ? nir_ddy(b, c) : nir_ddx(b, c);
         nir_def *scaled = nir_fmul(b, d, cube ? size[0] : size[a]);
         length = nir_ffma(b, scaled, scaled, length);
      }
      rho = nir_fmax(b, rho, nir_fsqrt(b, length));
   }
   nir_def *bias = nir_fmul_imm(b, nir_i2f32(b, nir_ishr_imm(b, nir_ishl_imm(b,
      bits(b, sampler, APEX_HW_SAMPLER_BIAS, 14), 18), 18)), 1.0 / 256);
   nir_def *min_lod = nir_fmul_imm(b, nir_u2f32(b, bits(b, sampler, APEX_HW_SAMPLER_MIN_LOD, 12)), 1.0 / 256);
   nir_def *max_lod = nir_fmul_imm(b, nir_u2f32(b, bits(b, sampler, APEX_HW_SAMPLER_MAX_LOD, 12)), 1.0 / 256);
   nir_def *lambda = nir_fmin(b, nir_fmax(b, nir_fadd(b, nir_flog2(b, rho), bias), min_lod), max_lod);
   nir_def *levels = bits(b, image, APEX_HW_IMAGE_LEVELS, 5);
   nir_def *max_level = nir_u2f32(b, nir_iadd_imm(b, nir_umax(b, levels, nir_imm_int(b, 1)), -1));
   nir_def *level = nir_fmin(b, nir_fmax(b, lambda, nir_imm_float(b, 0.0f)), max_level);
   return nir_bcsel(b, nir_ieq_imm(b, levels, 0), nir_imm_zero(b, 2, 32), nir_vec2(b, level, lambda));
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
   if (tex->sampler_dim == GLSL_SAMPLER_DIM_BUF) {
      /* Texel buffers: address, byte range, elements, format words. */
      nir_def *coord = tex_src(tex, nir_tex_src_coord);
      if (tex->op == nir_texop_txs) {
         nir_def_replace(&tex->def, row_word(b, image_row, 3));
         return true;
      }
      struct image_info img;
      nir_def *address = nir_vec2(b, row_word(b, image_row, 0), row_word(b, image_row, 1));
      for (unsigned k = 0; k < 3; k++)
         img.format[k] = row_word(b, image_row, 4 + k);
      img.bytes = nir_iand_imm(b, img.format[0], 31);
      nir_def *inside = nir_ult(b, coord, row_word(b, image_row, 3));
      nir_def *zero = nir_imm_zero(b, 4, 32);
      nir_push_if(b, inside);
      nir_def *value = decode_texel(b, &img, nir_build_addr_iadd(b, address,
         nir_address_format_2x32bit_global, nir_var_mem_global, nir_imul(b, coord, img.bytes)));
      nir_pop_if(b, NULL);
      nir_def_replace(&tex->def, nir_trim_vector(b, nir_if_phi(b, value, zero),
                                                 tex->def.num_components));
      return true;
   }
   nir_def *image = row_words(b, image_row);
   switch (tex->op) {
   case nir_texop_texture_samples:
      nir_def_replace(&tex->def, nir_ishl(b, nir_imm_int(b, 1), bits(b, image, APEX_HW_IMAGE_SAMPLES, 3)));
      return true;
   case nir_texop_samples_identical:
      nir_def_replace(&tex->def, nir_imm_false(b));
      return true;
   default:
      break;
   }
   bool sampled = tex->op != nir_texop_txf && tex->op != nir_texop_txf_ms &&
                  tex->op != nir_texop_txs && tex->op != nir_texop_query_levels;
   nir_def *sampler = image;
   if (sampled) {
      if (!sampler_deref) {
         ctx->invalid = true;
         return false;
      }
      sampler = row_words(b, deref_row(b, ctx, nir_instr_as_deref(nir_def_instr(sampler_deref)), true));
   }
   if (tex->op == nir_texop_lod) {
      nir_def_replace(&tex->def, query_lod(b, tex, image, sampler));
      return true;
   }
   /* The compiler encodes the instruction around both descriptors. */
   nir_tex_instr_remove_src(tex, nir_tex_instr_src_index(tex, nir_tex_src_texture_deref));
   int s = nir_tex_instr_src_index(tex, nir_tex_src_sampler_deref);
   if (s >= 0)
      nir_tex_instr_remove_src(tex, s);
   nir_tex_instr_add_src(tex, nir_tex_src_backend1, image);
   nir_tex_instr_add_src(tex, nir_tex_src_backend2, sampler);
   return true;
}

/* Byte offset of texel (x, y) in a tiled level of `tiles` tiles per row and
 * log2 texel bytes `lb` (apex_hw.h). */
static nir_def *
tiled_offset(nir_builder *b, nir_def *x, nir_def *y, nir_def *lb, nir_def *tiles)
{
   /* Block width and height logs by lb: 3 3 2 2 2 and 3 2 2 1 0. */
   nir_def *bw = nir_bcsel(b, nir_ult_imm(b, lb, 2), nir_imm_int(b, 3), nir_imm_int(b, 2));
   nir_def *bh = nir_bcsel(b, nir_ieq_imm(b, lb, 0), nir_imm_int(b, 3),
                 nir_bcsel(b, nir_ult_imm(b, lb, 3), nir_imm_int(b, 2), nir_isub(b, nir_imm_int(b, 4), lb)));
   nir_def *page = nir_iadd(b, nir_imul(b, nir_ushr(b, y, nir_iadd_imm(b, bh, 3)), tiles),
                            nir_ushr(b, x, nir_iadd_imm(b, bw, 3)));
   nir_def *bx = nir_iand_imm(b, nir_ushr(b, x, bw), 7), *by = nir_iand_imm(b, nir_ushr(b, y, bh), 7);
   nir_def *block = nir_imm_int(b, 0);
   for (unsigned i = 0; i < 3; i++) {
      block = nir_ior(b, block, nir_ishl_imm(b, nir_iand_imm(b, nir_ushr_imm(b, bx, i), 1), 2 * i));
      block = nir_ior(b, block, nir_ishl_imm(b, nir_iand_imm(b, nir_ushr_imm(b, by, i), 1), 2 * i + 1));
   }
   nir_def *one = nir_imm_int(b, 1);
   nir_def *sh = nir_umin(b, bh, nir_imm_int(b, 2));
   nir_def *tx = nir_iand(b, x, nir_iadd_imm(b, nir_ishl(b, one, bw), -1));
   nir_def *ty = nir_iand(b, y, nir_iadd_imm(b, nir_ishl(b, one, bh), -1));
   nir_def *sub = nir_iadd(b, nir_ishl(b, nir_ushr_imm(b, ty, 2), nir_iadd_imm(b, bw, -2)), nir_ushr_imm(b, tx, 2));
   nir_def *row = nir_iand(b, ty, nir_iadd_imm(b, nir_ishl(b, one, sh), -1));
   nir_def *within = nir_iadd(b, nir_ishl(b, sub, nir_iadd_imm(b, sh, 2)),
                              nir_iadd(b, nir_imul_imm(b, row, 4), nir_iand_imm(b, tx, 3)));
   return nir_iadd(b, nir_iadd(b, nir_imul_imm(b, page, APEX_HW_TILE), nir_imul_imm(b, block, 64)),
                   nir_ishl(b, within, lb));
}

/* Storage images take the image descriptor of the view's level and base
 * layer, then its format words; storage texel buffers the texel-buffer
 * row. Loads decode and stores encode through the view's format words, so
 * shader formats are optional. */
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
      for (unsigned k = 0; k < 3; k++)
         img.format[k] = row_word(b, row, 4 + k);
      img.bytes = nir_iand_imm(b, img.format[0], 31);
      if (size) {
         nir_def_replace(&i->def, nir_trim_vector(b, row_word(b, row, 3), i->def.num_components));
         return true;
      }
      nir_def *x = nir_channel(b, coord, 0);
      inside = nir_ult(b, x, row_word(b, row, 3));
      address = nir_build_addr_iadd(b, nir_vec2(b, row_word(b, row, 0), row_word(b, row, 1)),
                                    nir_address_format_2x32bit_global, nir_var_mem_global,
                                    nir_imul(b, x, img.bytes));
   } else {
      bool three_d = dim == GLSL_SAMPLER_DIM_3D, cube = dim == GLSL_SAMPLER_DIM_CUBE;
      unsigned dims = dim == GLSL_SAMPLER_DIM_1D ? 1 : three_d ? 3 : 2;
      nir_def *d = row_words(b, row);
      for (unsigned k = 0; k < 3; k++)
         img.format[k] = row_word(b, nir_iadd_imm(b, row, 1), k);
      img.bytes = nir_iand_imm(b, img.format[0], 31);
      /* Storage images have one sample; null descriptors zero levels. */
      nir_def *null = nir_ieq_imm(b, bits(b, d, APEX_HW_IMAGE_LEVELS, 5), 0);
      if (samples) {
         nir_def_replace(&i->def, nir_b2i32(b, nir_inot(b, null)));
         return true;
      }
      nir_def *extent[3] = {
         nir_iadd_imm(b, bits(b, d, APEX_HW_IMAGE_WIDTH, 14), 1),
         nir_iadd_imm(b, bits(b, d, APEX_HW_IMAGE_HEIGHT, 14), 1),
         nir_iadd_imm(b, bits(b, d, APEX_HW_IMAGE_DEPTH, 11), 1),
      };
      if (size) {
         nir_def *out[4];
         for (unsigned a = 0; a < dims; a++)
            out[a] = extent[a];
         unsigned n = dims;
         if (array)
            out[n++] = cube ? nir_udiv_imm(b, extent[2], 6) : extent[2];
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
      /* Cube images address a face (or a face of a cube) as a layer. */
      nir_def *z = three_d ? index[2] : array || cube ? nir_channel(b, coord, dims) : nir_imm_int(b, 0);
      if (!three_d)
         inside = nir_iand(b, inside, nir_ult(b, z, extent[2]));
      nir_def *tiled = nir_ine_imm(b, bits(b, d, APEX_HW_IMAGE_TILED, 1), 0);
      nir_def *lb = nir_find_lsb(b, img.bytes);
      /* Tiles are 64 x 64, 64 x 32, 32 x 32, 32 x 16 or 32 x 8 texels by lb;
       * 3D slices are planes of the level, layers a layer stride apart. */
      nir_def *tw = nir_bcsel(b, nir_ult_imm(b, lb, 2), nir_imm_int(b, 64), nir_imm_int(b, 32));
      nir_def *th = nir_ishl(b, nir_imm_int(b, 8), nir_iadd_imm(b, nir_bcsel(b, nir_ieq_imm(b, lb, 0),
         nir_imm_int(b, 3), nir_bcsel(b, nir_ult_imm(b, lb, 3), nir_imm_int(b, 2), nir_isub(b,
         nir_imm_int(b, 4), lb))), 0));
      nir_def *tiles = nir_udiv(b, nir_iadd(b, extent[0], nir_iadd_imm(b, tw, -1)), tw);
      nir_def *plane = nir_imul_imm(b, nir_imul(b, tiles, nir_udiv(b, nir_iadd(b, extent[1],
                                    nir_iadd_imm(b, th, -1)), th)), APEX_HW_TILE);
      nir_def *layer_stride = nir_imul_imm(b, bits(b, d, APEX_HW_IMAGE_LAYER_STRIDE, 32), 64);
      nir_def *linear = nir_iadd(b, nir_imul(b, index[1], nir_imul_imm(b, bits(b, d, APEX_HW_IMAGE_PITCH, 16), 64)),
                                 nir_imul(b, index[0], img.bytes));
      nir_def *offset = nir_bcsel(b, tiled, tiled_offset(b, index[0], index[1], lb, tiles), linear);
      offset = nir_iadd(b, offset, nir_imul(b, z, three_d ? plane : layer_stride));
      nir_def *base = nir_vec2(b, nir_channel(b, d, 0), nir_iand_imm(b, nir_channel(b, d, 1), 0xff));
      address = nir_build_addr_iadd(b, base, nir_address_format_2x32bit_global, nir_var_mem_global, offset);
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
