/* SPDX-License-Identifier: MIT */
/* FP32 library lowering onto the native FADD/FMUL/FFMA, U2F/F2U and integer
 * operations. Rounding functions are exact. Reciprocal, division, square root
 * and reciprocal square root use Newton-Raphson on FFMA; exp2/log2 use range
 * reduction plus polynomials; sin/cos use quarter-turn reduction. Accuracy
 * targets the Vulkan SPIR-V environment precision table. */
#include "apex.h"
#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"

static nir_def *
fimm(nir_builder *b, float value)
{
   return nir_imm_float(b, value);
}

static nir_def *
bits_float(nir_builder *b, uint32_t bits)
{
   return nir_imm_int(b, bits);
}

/* Magnitude bits and sign bit of an FP32 value held as a 32-bit word. */
static nir_def *
magnitude(nir_builder *b, nir_def *x)
{
   return nir_iand_imm(b, x, 0x7fffffffu);
}

static nir_def *
sign_bit(nir_builder *b, nir_def *x)
{
   return nir_iand_imm(b, x, 0x80000000u);
}

/* 2^k for k in [-126, 127] as an FP32 word. */
static nir_def *
exp2_int(nir_builder *b, nir_def *k)
{
   return nir_ishl_imm(b, nir_iadd_imm(b, k, 127), 23);
}

/* x * 2^k for integer k in [-252, 254] with one rounding. Each half-scale
 * factor is normal, so the first multiply is exact and the second rounds. */
static nir_def *
scale(nir_builder *b, nir_def *x, nir_def *k)
{
   nir_def *half = nir_ishr_imm(b, k, 1);
   return nir_fmul(b, nir_fmul(b, x, exp2_int(b, half)),
                   exp2_int(b, nir_isub(b, k, half)));
}

static nir_def *
trunc32(nir_builder *b, nir_def *x)
{
   nir_def *ax = magnitude(b, x);
   /* Unbiased exponent e in [0, 22]: clear the 23 - e fraction bits. */
   nir_def *e = nir_iadd_imm(b, nir_ushr_imm(b, ax, 23), -127);
   nir_def *mask = nir_ishl(b, nir_imm_int(b, -1), nir_isub(b, nir_imm_int(b, 23), e));
   nir_def *result = nir_bcsel(b, nir_ult_imm(b, ax, 0x3f800000u), sign_bit(b, x),
                               nir_iand(b, x, mask));
   return nir_bcsel(b, nir_uge_imm(b, ax, 0x4b000000u), x, result);
}

/* Integer-valued t and x differ only for finite inputs with fraction bits. */
static nir_def *
floor32(nir_builder *b, nir_def *x)
{
   nir_def *t = trunc32(b, x);
   nir_def *adjust = nir_iand(b, nir_ine(b, t, x), nir_ine_imm(b, sign_bit(b, x), 0));
   return nir_bcsel(b, adjust, nir_fadd(b, t, fimm(b, -1.0f)), t);
}

static nir_def *
ceil32(nir_builder *b, nir_def *x)
{
   nir_def *t = trunc32(b, x);
   nir_def *adjust = nir_iand(b, nir_ine(b, t, x), nir_ieq_imm(b, sign_bit(b, x), 0));
   return nir_bcsel(b, adjust, nir_fadd(b, t, fimm(b, 1.0f)), t);
}

/* Adding and removing 2^23 rounds the magnitude to nearest even. */
static nir_def *
round_even32(nir_builder *b, nir_def *x)
{
   nir_def *ax = magnitude(b, x);
   nir_def *big = fimm(b, 8388608.0f);
   nir_def *r = nir_fadd(b, nir_fadd(b, ax, big), nir_fneg(b, big));
   return nir_bcsel(b, nir_uge_imm(b, ax, 0x4b000000u), x, nir_ior(b, r, sign_bit(b, x)));
}

/* Newton-Raphson reciprocal of a normal FP32 m in [1, 2). */
static nir_def *
rcp_mantissa(nir_builder *b, nir_def *m)
{
   /* Minimax linear estimate on [1, 2): relative error at most 1/17. */
   nir_def *y = nir_ffma(b, m, fimm(b, -8.0f / 17.0f), fimm(b, 24.0f / 17.0f));
   for (unsigned i = 0; i < 3; i++) {
      nir_def *e = nir_ffma(b, nir_fneg(b, m), y, fimm(b, 1.0f));
      y = nir_ffma(b, y, e, y);
   }
   return y;
}

/* Reciprocal with IEEE special cases. Subnormal inputs are scaled first. */
static nir_def *
rcp32(nir_builder *b, nir_def *x)
{
   nir_def *tiny = nir_ult_imm(b, magnitude(b, x), 0x00800000u);
   nir_def *xs = nir_bcsel(b, tiny, nir_fmul(b, x, bits_float(b, 0x5f800000u)), x); /* 2^64 */
   nir_def *ax = magnitude(b, xs);
   nir_def *e = nir_iadd_imm(b, nir_ushr_imm(b, ax, 23), -127);
   nir_def *m = nir_ior_imm(b, nir_iand_imm(b, ax, 0x007fffffu), 0x3f800000u);
   nir_def *r = scale(b, rcp_mantissa(b, m), nir_ineg(b, e));
   r = nir_bcsel(b, tiny, nir_fmul(b, r, bits_float(b, 0x5f800000u)), r);
   r = nir_ior(b, r, sign_bit(b, x));
   nir_def *ax0 = magnitude(b, x);
   r = nir_bcsel(b, nir_ieq_imm(b, ax0, 0), nir_ior_imm(b, sign_bit(b, x), 0x7f800000u), r);
   r = nir_bcsel(b, nir_ieq_imm(b, ax0, 0x7f800000u), sign_bit(b, x), r);
   return nir_bcsel(b, nir_ult(b, nir_imm_int(b, 0x7f800000u), ax0), nir_imm_int(b, 0x7fc00000u), r);
}

/* Quotient refined by one residual step; zero, infinite and NaN estimates
 * already carry the IEEE result for the operands' special cases. */
static nir_def *
div32(nir_builder *b, nir_def *a, nir_def *d)
{
   nir_def *r = rcp32(b, d);
   nir_def *q = nir_fmul(b, a, r);
   nir_def *residual = nir_ffma(b, nir_fneg(b, d), q, a);
   nir_def *refined = nir_ffma(b, residual, r, q);
   nir_def *exponent = nir_iand_imm(b, q, 0x7f800000u);
   nir_def *special = nir_ior(b, nir_ieq_imm(b, exponent, 0), nir_ieq_imm(b, exponent, 0x7f800000u));
   return nir_bcsel(b, special, q, refined);
}

/* Reciprocal square root of a positive normal x. The initial estimate has
 * 0.18% relative error; three Newton steps exceed FP32 precision. */
static nir_def *
rsq_normal(nir_builder *b, nir_def *x)
{
   nir_def *y = nir_isub(b, nir_imm_int(b, 0x5f375a86u), nir_ushr_imm(b, x, 1));
   nir_def *half = nir_fmul(b, x, fimm(b, 0.5f));
   for (unsigned i = 0; i < 3; i++) {
      nir_def *r = nir_ffma(b, nir_fneg(b, nir_fmul(b, half, y)), y, fimm(b, 0.5f));
      y = nir_ffma(b, y, r, y);
   }
   return y;
}

static nir_def *
rsq32(nir_builder *b, nir_def *x)
{
   nir_def *ax = magnitude(b, x);
   nir_def *tiny = nir_ult_imm(b, ax, 0x00800000u);
   nir_def *xs = nir_bcsel(b, tiny, nir_fmul(b, x, bits_float(b, 0x5f800000u)), x); /* 2^64 */
   nir_def *r = rsq_normal(b, xs);
   r = nir_bcsel(b, tiny, nir_fmul(b, r, bits_float(b, 0x4f800000u)), r); /* 2^32 */
   r = nir_bcsel(b, nir_ieq_imm(b, ax, 0), nir_ior_imm(b, sign_bit(b, x), 0x7f800000u), r);
   r = nir_bcsel(b, nir_ieq_imm(b, x, 0x7f800000u), nir_imm_int(b, 0), r);
   nir_def *invalid = nir_ior(b, nir_ult(b, nir_imm_int(b, 0x7f800000u), ax),
                              nir_iand(b, nir_ine_imm(b, sign_bit(b, x), 0), nir_ine_imm(b, ax, 0)));
   return nir_bcsel(b, invalid, nir_imm_int(b, 0x7fc00000u), r);
}

static nir_def *
sqrt32(nir_builder *b, nir_def *x)
{
   nir_def *ax = magnitude(b, x);
   nir_def *tiny = nir_ult_imm(b, ax, 0x00800000u);
   nir_def *xs = nir_bcsel(b, tiny, nir_fmul(b, x, bits_float(b, 0x5f800000u)), x); /* 2^64 */
   nir_def *y = rsq_normal(b, xs);
   nir_def *s = nir_fmul(b, xs, y);
   nir_def *residual = nir_ffma(b, nir_fneg(b, s), s, xs);
   s = nir_ffma(b, residual, nir_fmul(b, y, fimm(b, 0.5f)), s);
   s = nir_bcsel(b, tiny, nir_fmul(b, s, bits_float(b, 0x2f800000u)), s); /* 2^-32 */
   /* Zeros keep their sign; +inf is exact; negative and NaN inputs are invalid. */
   s = nir_bcsel(b, nir_ior(b, nir_ieq_imm(b, ax, 0), nir_ieq_imm(b, x, 0x7f800000u)), x, s);
   nir_def *invalid = nir_ior(b, nir_ult(b, nir_imm_int(b, 0x7f800000u), ax),
                              nir_iand(b, nir_ine_imm(b, sign_bit(b, x), 0), nir_ine_imm(b, ax, 0)));
   return nir_bcsel(b, invalid, nir_imm_int(b, 0x7fc00000u), s);
}

/* 2^x: x = n + f with integer n and |f| <= 1/2, 2^f by its degree-7 Taylor
 * series in f*ln2 (truncation below 6e-9), then an exact two-step scale. */
static nir_def *
exp2_32(nir_builder *b, nir_def *x)
{
   nir_def *n = round_even32(b, x);
   nir_def *f = nir_fadd(b, x, nir_fneg(b, n));
   static const double c[] = {
      1.0, 0.6931471805599453, 0.2402265069591007, 0.05550410866482158,
      0.009618129107628477, 0.0013333558146428443, 1.5403530393381606e-4,
      1.525273380405984e-5,
   };
   nir_def *p = fimm(b, c[7]);
   for (int i = 6; i >= 0; i--)
      p = nir_ffma(b, p, f, fimm(b, c[i]));
   /* |n| <= 160 keeps both half-scale factors normal; results beyond the
    * FP32 range overflow or underflow in the final multiply. */
   nir_def *clamped = nir_bcsel(b, nir_flt(b, n, fimm(b, -160.0f)), fimm(b, -160.0f),
                                nir_bcsel(b, nir_flt(b, fimm(b, 160.0f), n), fimm(b, 160.0f), n));
   nir_def *r = scale(b, p, nir_f2i32(b, clamped));
   r = nir_bcsel(b, nir_ieq_imm(b, x, 0x7f800000u), x, r);
   r = nir_bcsel(b, nir_ieq_imm(b, x, 0xff800000u), nir_imm_int(b, 0), r);
   return nir_bcsel(b, nir_ult(b, nir_imm_int(b, 0x7f800000u), magnitude(b, x)),
                    nir_imm_int(b, 0x7fc00000u), r);
}

/* log2(x) = e + ln(m)/ln2 with m in [sqrt(1/2), sqrt(2)); ln(m) = 2 atanh(s)
 * for s = (m-1)/(m+1), |s| <= 0.1716, through s^11 (truncation below 1e-9). */
static nir_def *
log2_32(nir_builder *b, nir_def *x)
{
   nir_def *ax = magnitude(b, x);
   nir_def *tiny = nir_ult_imm(b, ax, 0x00800000u);
   nir_def *xs = nir_bcsel(b, tiny, nir_fmul(b, x, bits_float(b, 0x5f800000u)), x); /* 2^64 */
   nir_def *bits = magnitude(b, xs);
   /* Rebias mantissas above sqrt(2) into [sqrt(1/2), 1). */
   nir_def *high = nir_uge_imm(b, nir_iand_imm(b, bits, 0x007fffffu), 0x003504f3u);
   nir_def *e = nir_iadd(b, nir_iadd_imm(b, nir_ushr_imm(b, bits, 23), -127), nir_b2i32(b, high));
   e = nir_bcsel(b, tiny, nir_iadd_imm(b, e, -64), e);
   nir_def *m = nir_ior(b, nir_iand_imm(b, bits, 0x007fffffu),
                        nir_bcsel(b, high, nir_imm_int(b, 0x3f000000u), nir_imm_int(b, 0x3f800000u)));
   nir_def *s = div32(b, nir_fadd(b, m, fimm(b, -1.0f)), nir_fadd(b, m, fimm(b, 1.0f)));
   nir_def *s2 = nir_fmul(b, s, s);
   nir_def *p = fimm(b, 2.0f / 11.0f);
   for (int k = 9; k >= 1; k -= 2)
      p = nir_ffma(b, p, s2, fimm(b, 2.0f / k));
   nir_def *ln = nir_fmul(b, p, s);
   nir_def *r = nir_ffma(b, ln, fimm(b, 1.4426950408889634f), nir_i2f32(b, e));
   r = nir_bcsel(b, nir_ieq_imm(b, ax, 0), nir_imm_int(b, 0xff800000u), r);
   r = nir_bcsel(b, nir_ieq_imm(b, x, 0x7f800000u), x, r);
   nir_def *invalid = nir_ior(b, nir_ult(b, nir_imm_int(b, 0x7f800000u), ax),
                              nir_iand(b, nir_ine_imm(b, sign_bit(b, x), 0), nir_ine_imm(b, ax, 0)));
   return nir_bcsel(b, invalid, nir_imm_int(b, 0x7fc00000u), r);
}

/* Reduce to turns t in [-1/2, 1/2], then quadrant q = round(4t) and
 * r = (t - q/4) * 2pi in [-pi/4, pi/4]; Taylor terms through r^9 / r^8. */
static nir_def *
sincos32(nir_builder *b, nir_def *x, bool cosine)
{
   nir_def *t = nir_fmul(b, x, fimm(b, 0.15915494309189535f));
   t = nir_fadd(b, t, nir_fneg(b, round_even32(b, t)));
   nir_def *q4 = round_even32(b, nir_fmul(b, t, fimm(b, 4.0f)));
   nir_def *r = nir_fmul(b, nir_ffma(b, q4, fimm(b, -0.25f), t), fimm(b, 6.283185307179586f));
   nir_def *q = nir_iand_imm(b, nir_iadd_imm(b, nir_f2i32(b, q4), cosine ? 1 : 0), 3);
   nir_def *r2 = nir_fmul(b, r, r);
   nir_def *sin_p = fimm(b, 1.0f / 362880.0f);
   sin_p = nir_ffma(b, sin_p, r2, fimm(b, -1.0f / 5040.0f));
   sin_p = nir_ffma(b, sin_p, r2, fimm(b, 1.0f / 120.0f));
   sin_p = nir_ffma(b, sin_p, r2, fimm(b, -1.0f / 6.0f));
   nir_def *sin_r = nir_ffma(b, nir_fmul(b, sin_p, r2), r, r);
   nir_def *cos_p = fimm(b, 1.0f / 40320.0f);
   cos_p = nir_ffma(b, cos_p, r2, fimm(b, -1.0f / 720.0f));
   cos_p = nir_ffma(b, cos_p, r2, fimm(b, 1.0f / 24.0f));
   cos_p = nir_ffma(b, cos_p, r2, fimm(b, -0.5f));
   nir_def *cos_r = nir_ffma(b, cos_p, r2, fimm(b, 1.0f));
   /* sin(x + q*pi/2): q=0 sin, 1 cos, 2 -sin, 3 -cos; cosine adds one quadrant. */
   nir_def *odd = nir_ine_imm(b, nir_iand_imm(b, q, 1), 0);
   nir_def *value = nir_bcsel(b, odd, cos_r, sin_r);
   nir_def *negate = nir_ine_imm(b, nir_iand_imm(b, q, 2), 0);
   value = nir_bcsel(b, negate, nir_fneg(b, value), value);
   return nir_bcsel(b, nir_uge_imm(b, magnitude(b, x), 0x7f800000u), nir_imm_int(b, 0x7fc00000u), value);
}

static bool
float_library(const nir_instr *instr, const void *data)
{
   if (instr->type != nir_instr_type_alu)
      return false;
   const nir_alu_instr *a = nir_instr_as_alu(instr);
   if (a->def.bit_size != 32)
      return false;
   switch (a->op) {
   case nir_op_ftrunc: case nir_op_ffloor: case nir_op_fceil:
   case nir_op_fround_even:
   case nir_op_frcp: case nir_op_fdiv: case nir_op_frsq: case nir_op_fsqrt:
   case nir_op_fexp2: case nir_op_flog2: case nir_op_fsin: case nir_op_fcos:
      return true;
   default:
      return false;
   }
}

static nir_def *
lower_float_library(nir_builder *b, nir_instr *instr, void *data)
{
   nir_alu_instr *a = nir_instr_as_alu(instr);
   nir_def *x = nir_ssa_for_alu_src(b, a, 0);
   /* Rounding tricks and Newton residuals rely on IEEE results; forbid
    * algebraic rewrites such as (a + b) - b -> a on the expansion. */
   b->fp_math_ctrl = nir_fp_no_fast_math;
   switch (a->op) {
   case nir_op_ftrunc: return trunc32(b, x);
   case nir_op_ffloor: return floor32(b, x);
   case nir_op_fceil: return ceil32(b, x);
   case nir_op_fround_even: return round_even32(b, x);
   case nir_op_frcp: return rcp32(b, x);
   case nir_op_fdiv: return div32(b, x, nir_ssa_for_alu_src(b, a, 1));
   case nir_op_frsq: return rsq32(b, x);
   case nir_op_fsqrt: return sqrt32(b, x);
   case nir_op_fexp2: return exp2_32(b, x);
   case nir_op_flog2: return log2_32(b, x);
   case nir_op_fsin: return sincos32(b, x, false);
   case nir_op_fcos: return sincos32(b, x, true);
   default: UNREACHABLE("filtered FP32 library op");
   }
}

bool
apex_lower_float_library(nir_shader *nir)
{
   return nir_shader_lower_instructions(nir, float_library, lower_float_library, NULL);
}
