/* SPDX-License-Identifier: MIT */
#include "apex.h"
#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "util/u_dynarray.h"
#include <stdio.h>

const struct nir_shader_compiler_options apex_nir_options = {
   .lower_fdiv = true, .lower_flrp32 = true,
   .lower_bit_count = true, .lower_bitfield_reverse = true, .lower_mul_high = true,
   .lower_mul_2x32_64 = true,
   .lower_extract_byte = true, .lower_extract_word = true,
   .lower_bitfield_extract = true, .lower_bitfield_insert = true,
   .lower_ifind_msb = true, .lower_find_lsb = true,
};

static uint32_t value(nir_def *def, unsigned component)
{
   return def->index * 4 + component;
}

static void emit(struct util_dynarray *ops, uint32_t op, uint32_t d,
                 uint32_t a, uint32_t b, uint32_t c, uint32_t imm)
{
   struct apex_op i = {op, d, a, b, c, imm};
   util_dynarray_append(ops, i);
}

static unsigned alu_op(nir_op op)
{
   switch (op) {
   case nir_op_mov: return 0x21;
   case nir_op_iadd: return 0x22;
   case nir_op_isub: return 0x23;
   case nir_op_imul: return 0x24;
   case nir_op_iand: return 0x25;
   case nir_op_ior: return 0x26;
   case nir_op_ixor: return 0x27;
   case nir_op_ishl: return 0x28;
   case nir_op_ushr: return 0x29;
   case nir_op_ult32: return 0x2a;
   case nir_op_ieq32: return 0x2b;
   case nir_op_b32csel: return 0x2e;
   case nir_op_fadd: return 0x30;
   case nir_op_fmul: return 0x31;
   case nir_op_ffma:
   case nir_op_ffma_weak: return 0x32;
   case nir_op_u2f32: return 0x33;
   case nir_op_f2u32: return 0x34;
   default: return 0;
   }
}

static bool lower_launch(nir_builder *b, nir_intrinsic_instr *i, void *data)
{
   bool *invalid = data;
   nir_def *replacement = NULL;
   b->cursor = nir_before_instr(&i->instr);
   switch (i->intrinsic) {
   case nir_intrinsic_load_base_global_invocation_id:
      replacement = nir_imm_ivec3(b, 0, 0, 0); break;
   case nir_intrinsic_load_local_invocation_id:
      /* The admitted workgroup size is exactly 16x1x1. */
      replacement = nir_vec3(b, nir_load_local_invocation_index(b),
                            nir_imm_int(b, 0), nir_imm_int(b, 0)); break;
   case nir_intrinsic_load_workgroup_id:
      /* One native launch is one 16x1x1 workgroup. The queue supplies its
       * coarse base; there is no second in-launch workgroup index. */
      replacement = nir_imm_ivec3(b, 0, 0, 0); break;
   case nir_intrinsic_vulkan_resource_index:
      if (nir_intrinsic_desc_set(i) || nir_intrinsic_binding(i) ||
          !nir_src_is_const(i->src[0]) || nir_src_as_uint(i->src[0])) {
         *invalid = true; return false;
      }
      replacement = nir_imm_ivec2(b, 0, 0); break;
   case nir_intrinsic_load_vulkan_descriptor:
      replacement = i->src[0].ssa; break;
   default: return false;
   }
   nir_def_rewrite_uses(&i->def, replacement);
   nir_instr_remove(&i->instr);
   return true;
}

static bool fp32_sign_conversion(const nir_instr *instr, const void *data)
{
   if (instr->type != nir_instr_type_alu)
      return false;
   const nir_alu_instr *a = nir_instr_as_alu(instr);
   return a->def.bit_size == 32 && a->src[0].src.ssa->bit_size == 32 &&
          (a->op == nir_op_fneg || a->op == nir_op_fabs ||
           a->op == nir_op_i2f32 || a->op == nir_op_f2i32);
}

static nir_def *lower_fp32_sign_conversion(nir_builder *b, nir_instr *instr, void *data)
{
   nir_alu_instr *a = nir_instr_as_alu(instr);
   nir_def *x = nir_ssa_for_alu_src(b, a, 0);
   if (a->op == nir_op_fneg)
      return nir_ixor(b, x, nir_imm_int(b, 0x80000000u));
   if (a->op == nir_op_fabs)
      return nir_iand_imm(b, x, 0x7fffffffu);
   if (a->op == nir_op_i2f32) {
      /* INT_MIN's magnitude is representable as an unsigned word. */
      nir_def *magnitude = nir_u2f32(b, nir_iabs(b, x));
      return nir_ixor(b, magnitude, nir_iand_imm(b, x, 0x80000000u));
   }
   nir_def *magnitude = nir_f2u32(b, nir_iand_imm(b, x, 0x7fffffffu));
   return nir_bcsel(b, nir_ilt_imm(b, x, 0), nir_ineg(b, magnitude), magnitude);
}

static bool fp32_minmax_sign(const nir_instr *instr, const void *data)
{
   if (instr->type != nir_instr_type_alu)
      return false;
   const nir_alu_instr *a = nir_instr_as_alu(instr);
   return a->def.bit_size == 32 &&
          (a->op == nir_op_fmin || a->op == nir_op_fmax || a->op == nir_op_fsign);
}

static nir_def *lower_fp32_minmax_sign(nir_builder *b, nir_instr *instr, void *data)
{
   nir_alu_instr *a = nir_instr_as_alu(instr);
   nir_def *x = nir_ssa_for_alu_src(b, a, 0);
   nir_def *abs_x = nir_iand_imm(b, x, 0x7fffffffu);
   nir_def *nan_x = nir_ult(b, nir_imm_int(b, 0x7f800000u), abs_x);
   if (a->op == nir_op_fsign) {
      nir_def *one = nir_ior_imm(b, nir_iand_imm(b, x, 0x80000000u), 0x3f800000u);
      return nir_bcsel(b, nan_x, nir_imm_int(b, 0),
                      nir_bcsel(b, nir_ieq_imm(b, abs_x, 0), x, one));
   }
   nir_def *y = nir_ssa_for_alu_src(b, a, 1);
   nir_def *abs_y = nir_iand_imm(b, y, 0x7fffffffu);
   nir_def *nan_y = nir_ult(b, nir_imm_int(b, 0x7f800000u), abs_y);
   bool minimum = a->op == nir_op_fmin;
   nir_def *ordered = minimum ? nir_bcsel(b, nir_flt(b, x, y), x, y) :
                                nir_bcsel(b, nir_flt(b, x, y), y, x);
   /* NIR requires minimumNumber/maximumNumber: -0 < +0 and a single
    * NaN yields the numeric operand. Quiet a pair of NaNs canonically. */
   nir_def *zeros = minimum ? nir_ior(b, x, y) : nir_iand(b, x, y);
   ordered = nir_bcsel(b, nir_ieq_imm(b, nir_ior(b, abs_x, abs_y), 0), zeros, ordered);
   return nir_bcsel(b, nan_x,
                   nir_bcsel(b, nan_y, nir_imm_int(b, 0x7fc00000u), y),
                   nir_bcsel(b, nan_y, x, ordered));
}

static bool fp32_comparison(const nir_instr *instr, const void *data)
{
   if (instr->type != nir_instr_type_alu)
      return false;
   const nir_alu_instr *a = nir_instr_as_alu(instr);
   return (a->op == nir_op_flt || a->op == nir_op_fge ||
           a->op == nir_op_feq || a->op == nir_op_fneu) &&
          a->src[0].src.ssa->bit_size == 32 && a->src[1].src.ssa->bit_size == 32;
}

static nir_def *lower_fp32_comparison(nir_builder *b, nir_instr *instr, void *data)
{
   nir_alu_instr *a = nir_instr_as_alu(instr);
   nir_def *x = nir_ssa_for_alu_src(b, a, 0);
   nir_def *y = nir_ssa_for_alu_src(b, a, 1);
   nir_def *abs_x = nir_iand_imm(b, x, 0x7fffffffu);
   nir_def *abs_y = nir_iand_imm(b, y, 0x7fffffffu);
   nir_def *inf = nir_imm_int(b, 0x7f800000u);
   nir_def *unordered = nir_ior(b, nir_ult(b, inf, abs_x), nir_ult(b, inf, abs_y));
   nir_def *ordered = nir_inot(b, unordered);
   nir_def *both_zero = nir_ieq_imm(b, nir_ior(b, abs_x, abs_y), 0);
   if (a->op == nir_op_feq || a->op == nir_op_fneu) {
      nir_def *equal = nir_iand(b, ordered, nir_ior(b, nir_ieq(b, x, y), both_zero));
      return a->op == nir_op_feq ? equal : nir_inot(b, equal);
   }
   /* IEEE magnitude bits increase with magnitude. Invert negative encodings
    * and flip the sign bit on nonnegative encodings to obtain unsigned order.
    * NaNs are unordered; the two zero encodings compare equal. */
   nir_def *negative_mask = nir_imm_int(b, UINT32_MAX);
   nir_def *positive_mask = nir_imm_int(b, 0x80000000u);
   nir_def *key_x = nir_ixor(b, x, nir_bcsel(b, nir_ilt_imm(b, x, 0), negative_mask, positive_mask));
   nir_def *key_y = nir_ixor(b, y, nir_bcsel(b, nir_ilt_imm(b, y, 0), negative_mask, positive_mask));
   nir_def *less = nir_iand(b, nir_ult(b, key_x, key_y), nir_inot(b, both_zero));
   return nir_iand(b, ordered, a->op == nir_op_flt ? less : nir_inot(b, less));
}

static bool int32_division(const nir_instr *instr, const void *data)
{
   if (instr->type != nir_instr_type_alu)
      return false;
   const nir_alu_instr *a = nir_instr_as_alu(instr);
   return a->def.bit_size == 32 &&
          (a->op == nir_op_udiv || a->op == nir_op_umod || a->op == nir_op_idiv ||
           a->op == nir_op_irem || a->op == nir_op_imod);
}

static nir_def *lower_int32_division(nir_builder *b, nir_instr *instr, void *data)
{
   nir_alu_instr *a = nir_instr_as_alu(instr);
   nir_def *x = nir_ssa_for_alu_src(b, a, 0);
   nir_def *y = nir_ssa_for_alu_src(b, a, 1);
   bool is_signed = a->op != nir_op_udiv && a->op != nir_op_umod;
   bool quotient = a->op == nir_op_udiv || a->op == nir_op_idiv;
   nir_def *remainder = is_signed ? nir_iabs(b, x) : x;
   nir_def *divisor = is_signed ? nir_iabs(b, y) : y;
   nir_def *result = nir_imm_int(b, 0);
   /* Test before shifting the divisor: a successful subtraction then fits
    * in 32 bits even when the divisor or dividend has its high bit set. */
   for (int bit = 31; bit >= 0; bit--) {
      nir_def *fits = nir_uge(b, nir_ushr_imm(b, remainder, bit), divisor);
      remainder = nir_bcsel(b, fits,
         nir_isub(b, remainder, nir_ishl_imm(b, divisor, bit)), remainder);
      if (quotient)
         result = nir_ior(b, result, nir_bcsel(b, fits, nir_imm_int(b, 1u << bit), nir_imm_int(b, 0)));
   }
   if (!quotient)
      result = remainder;
   if (is_signed) {
      nir_def *negative = nir_ilt_imm(b, quotient ? nir_ixor(b, x, y) : x, 0);
      result = nir_bcsel(b, negative, nir_ineg(b, result), result);
      if (a->op == nir_op_imod) {
         nir_def *adjust = nir_iand(b, nir_ine_imm(b, result, 0),
                                    nir_ilt_imm(b, nir_ixor(b, result, y), 0));
         result = nir_bcsel(b, adjust, nir_iadd(b, result, y), result);
      }
   }
   return result;
}

static bool uint32_msb(const nir_instr *instr, const void *data)
{
   return instr->type == nir_instr_type_alu &&
          nir_instr_as_alu(instr)->op == nir_op_ufind_msb &&
          nir_instr_as_alu(instr)->src[0].src.ssa->bit_size == 32;
}

static nir_def *lower_uint32_msb(nir_builder *b, nir_instr *instr, void *data)
{
   nir_def *x = nir_ssa_for_alu_src(b, nir_instr_as_alu(instr), 0);
   nir_def *remaining = x;
   nir_def *index = nir_imm_int(b, 0);
   for (unsigned shift = 16; shift; shift >>= 1) {
      nir_def *high = nir_ushr_imm(b, remaining, shift);
      nir_def *nonzero = nir_ine_imm(b, high, 0);
      remaining = nir_bcsel(b, nonzero, high, remaining);
      index = nir_ior(b, index, nir_bcsel(b, nonzero, nir_imm_int(b, shift), nir_imm_int(b, 0)));
   }
   return nir_bcsel(b, nir_ieq_imm(b, x, 0), nir_imm_int(b, -1), index);
}

static int atomic_op(nir_atomic_op op)
{
   switch (op) {
   case nir_atomic_op_iadd: return 0;
   case nir_atomic_op_xchg: return 1;
   case nir_atomic_op_cmpxchg: return 2;
   case nir_atomic_op_iand: return 3;
   case nir_atomic_op_ior: return 4;
   case nir_atomic_op_ixor: return 5;
   case nir_atomic_op_imin: return 6;
   case nir_atomic_op_imax: return 7;
   case nir_atomic_op_umin: return 8;
   case nir_atomic_op_umax: return 9;
   default: return -1;
   }
}

struct loop_masks { uint32_t live, iteration; struct loop_masks *parent; };
struct control_state {
   struct util_dynarray *ops;
   uint32_t *temporary;
   uint32_t zero, discard;
   struct loop_masks *loop;
   struct apex_compile_result *output;
};
static bool emit_block(struct util_dynarray *ops, nir_block *block, uint32_t *temporary,
                       struct control_state *control);
static bool emit_cf(struct control_state *c, struct exec_list *list);

static int fail(struct apex_compile_result *output, const char *message)
{
   snprintf(output->diagnostic, sizeof(output->diagnostic), "%s", message);
   return 1;
}

int apex_from_nir(nir_shader *nir, struct apex_compile_result *output)
{
   *output = (struct apex_compile_result){0};
   if (nir->info.stage != MESA_SHADER_COMPUTE)
      return fail(output, "only compute shaders are supported");
   nir_validate_shader(nir, "Apex SPIR-V import");
   bool zero_shared = nir->info.zero_initialize_shared_memory;
   nir_foreach_variable_with_modes(var, nir, nir_var_mem_shared) {
      if (var->constant_initializer) {
         if (!var->constant_initializer->is_null_constant) {
            return fail(output, "shared initializer must be zero");
         }
         zero_shared = true;
      }
   }
   if (nir->info.workgroup_size_variable || nir->info.workgroup_size[0]!=16 ||
       nir->info.workgroup_size[1]!=1 || nir->info.workgroup_size[2]!=1) {
      return fail(output, "native launch requires local size 16x1x1");
   }
   NIR_PASS(_, nir, nir_lower_variable_initializers, nir_var_function_temp);
   NIR_PASS(_, nir, nir_lower_returns);
   NIR_PASS(_, nir, nir_inline_functions);
   NIR_PASS(_, nir, nir_opt_deref);
   NIR_PASS(_, nir, nir_lower_vars_to_ssa);
   NIR_PASS(_, nir, nir_remove_dead_variables, nir_var_function_temp, NULL);
   NIR_PASS(_, nir, nir_lower_vars_to_explicit_types, nir_var_function_temp, glsl_get_natural_size_align_bytes);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_function_temp, nir_address_format_32bit_offset);
   NIR_PASS(_, nir, nir_lower_vars_to_explicit_types, nir_var_mem_shared, glsl_get_natural_size_align_bytes);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_shared, nir_address_format_32bit_offset);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_ssbo, nir_address_format_32bit_index_offset);
   NIR_PASS(_, nir, nir_lower_system_values);
   bool invalid = false;
   nir_shader_intrinsics_pass(nir, lower_launch, nir_metadata_control_flow, &invalid);
   if (invalid) return fail(output, "only SSBO set 0 binding 0 is supported");
   NIR_PASS(_, nir, nir_lower_alu_to_scalar, NULL, NULL);
   NIR_PASS(_, nir, nir_shader_lower_instructions, fp32_minmax_sign, lower_fp32_minmax_sign, NULL);
   NIR_PASS(_, nir, nir_shader_lower_instructions, fp32_comparison, lower_fp32_comparison, NULL);
   NIR_PASS(_, nir, nir_shader_lower_instructions, int32_division, lower_int32_division, NULL);
   const nir_lower_subgroups_options subgroups = {
      .subgroup_size = 16, .ballot_bit_size = 32, .ballot_components = 1,
      .lower_to_scalar = true, .lower_vote = true, .lower_vote_ieq = true,
      .lower_vote_bool_eq = true, .lower_elect = true,
      .lower_first_invocation_to_ballot = true, .lower_read_first_invocation = true,
      .lower_subgroup_masks = true, .lower_inverse_ballot = true,
      .lower_relative_shuffle = true, .lower_quad = true,
      .lower_reduce = true,
   };
   bool progress;
   do {
      progress = false;
      NIR_PASS(progress, nir, nir_lower_subgroups, &subgroups);
      NIR_PASS(progress, nir, nir_lower_alu_to_scalar, NULL, NULL);
      NIR_PASS(progress, nir, nir_opt_algebraic);
      NIR_PASS(progress, nir, nir_shader_lower_instructions, uint32_msb, lower_uint32_msb, NULL);
      NIR_PASS(progress, nir, nir_lower_alu);
      NIR_PASS(progress, nir, nir_opt_copy_prop);
      NIR_PASS(progress, nir, nir_opt_dce);
      NIR_PASS(progress, nir, nir_opt_constant_folding);
      NIR_PASS(progress, nir, nir_opt_cse);
   } while (progress);
   NIR_PASS(_, nir, nir_shader_lower_instructions, fp32_sign_conversion, lower_fp32_sign_conversion, NULL);
   NIR_PASS(_, nir, nir_lower_phis_to_scalar, NULL, NULL);
   NIR_PASS(_, nir, nir_lower_continue_constructs);
   NIR_PASS(_, nir, nir_lower_bool_to_int32);
   NIR_PASS(_, nir, nir_convert_from_ssa, true, false);
   nir_validate_shader(nir, "Apex normalized NIR");
   if (getenv("APEX_DUMP_NIR")) nir_print_shader(nir, stderr);
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_index_ssa_defs(impl);
   uint32_t temporary = impl->ssa_alloc * 4;
   struct util_dynarray ops;
   util_dynarray_init(&ops, NULL);
   int result = 1;
   if (zero_shared && nir->info.shared_size) {
      /* Each lane clears one word per 64-byte chunk. Padding belongs to this
       * workgroup allocation, so the final partial chunk cannot touch a neighbor. */
      if (nir->info.shared_size > 32768) {
         fail(output, "shared memory exceeds 32768 bytes");
         goto done;
      }
      nir->info.shared_size = (nir->info.shared_size + 63) & ~63u;
      uint32_t lane = temporary++, shift = temporary++, offset = temporary++;
      uint32_t zero = temporary++;
      emit(&ops, 0x40, lane, 0, 0, 0, 1);
      emit(&ops, 0x20, shift, 0, 0, 0, 2);
      emit(&ops, 0x28, offset, lane, shift, 0, 0);
      emit(&ops, 0x20, zero, 0, 0, 0, 0);
      for (unsigned base = 0; base < nir->info.shared_size; base += 64) {
         uint32_t chunk = temporary++, address = temporary++;
         emit(&ops, 0x20, chunk, 0, 0, 0, base);
         emit(&ops, 0x22, address, offset, chunk, 0, 0);
         emit(&ops, 0x54, 0, address, zero, 0, 0);
      }
      emit(&ops, 7, 0, 0, 0, 0, 0);
   }
   struct control_state control = { .ops=&ops, .temporary=&temporary,
                                    .zero=temporary++, .discard=temporary++, .output=output };
   if (exec_list_length(&impl->body)>1)
      emit(&ops, 0x10, control.zero, 0, 0, 0, 0);
   if (!emit_cf(&control, &impl->body)) goto done;
   nir_validate_shader(nir, "Apex backend boundary");
   result = apex_emit(util_dynarray_begin(&ops), util_dynarray_num_elements(&ops, struct apex_op), nir->info.shared_size, nir->scratch_size, output);
done:
   util_dynarray_fini(&ops);
   return result;
}

static bool emit_block(struct util_dynarray *output, nir_block *block,
                       uint32_t *next_temporary, struct control_state *control)
{
   struct util_dynarray ops = *output;
   uint32_t temporary = *next_temporary;
   bool result = true;
   {
      nir_foreach_instr(instr, block) {
         nir_def *def = nir_instr_def(instr);
         if (def && (def->bit_size != 32 || def->num_components > 4)) goto unsupported;
         if (instr->type == nir_instr_type_load_const) {
            nir_load_const_instr *c = nir_instr_as_load_const(instr);
            for (unsigned j = 0; j < c->def.num_components; j++)
               emit(&ops, 0x20, value(&c->def,j), 0, 0, 0, c->value[j].u32);
         } else if (instr->type == nir_instr_type_alu) {
            nir_alu_instr *a = nir_instr_as_alu(instr);
            unsigned op = alu_op(a->op);
            if (a->op == nir_op_b2i32) {
               uint32_t one = temporary++;
               emit(&ops, 0x20, one, 0, 0, 0, 1);
               emit(&ops, 0x25, value(&a->def, 0),
                    value(a->src[0].src.ssa, a->src[0].swizzle[0]), one, 0, 0);
               continue;
            }
            if (a->op == nir_op_ineg) {
               uint32_t zero = temporary++;
               emit(&ops, 0x20, zero, 0, 0, 0, 0);
               emit(&ops, 0x23, value(&a->def, 0), zero,
                    value(a->src[0].src.ssa, a->src[0].swizzle[0]), 0, 0);
               continue;
            }
            if (a->op == nir_op_ishr || a->op == nir_op_iabs || a->op == nir_op_isign) {
               uint32_t operand = value(a->src[0].src.ssa, a->src[0].swizzle[0]);
               uint32_t top = temporary++, sign = temporary++, zero = temporary++;
               uint32_t mask = temporary++;
               emit(&ops, 0x20, top, 0, 0, 0, 31);
               emit(&ops, 0x29, sign, operand, top, 0, 0);
               emit(&ops, 0x20, zero, 0, 0, 0, 0);
               emit(&ops, 0x23, mask, zero, sign, 0, 0);
               if (a->op == nir_op_isign) {
                  uint32_t nonzero = temporary++, positive = temporary++;
                  /* Comparisons produce all-one masks. Negate for 0/1,
                   * then OR with the negative-operand mask for -1. */
                  emit(&ops, 0x2a, nonzero, zero, operand, 0, 0);
                  emit(&ops, 0x23, positive, zero, nonzero, 0, 0);
                  emit(&ops, 0x26, value(&a->def, 0), positive, mask, 0, 0);
               } else {
                  uint32_t biased = temporary++;
                  emit(&ops, 0x27, biased, operand, mask, 0, 0);
                  if (a->op == nir_op_iabs) {
                     emit(&ops, 0x23, value(&a->def, 0), biased, mask, 0, 0);
                  } else {
                     /* Complement negative operands around a logical shift
                      * so vacated bits receive their original sign. */
                     uint32_t count = value(a->src[1].src.ssa, a->src[1].swizzle[0]);
                     uint32_t shifted = temporary++;
                     emit(&ops, 0x29, shifted, biased, count, 0, 0);
                     emit(&ops, 0x27, value(&a->def, 0), shifted, mask, 0, 0);
                  }
               }
               continue;
            }
            bool signed_order = a->op == nir_op_ilt32 || a->op == nir_op_ige32 ||
                                a->op == nir_op_imin || a->op == nir_op_imax;
            bool minimum = a->op == nir_op_imin || a->op == nir_op_umin;
            bool maximum = a->op == nir_op_imax || a->op == nir_op_umax;
            if (signed_order || minimum || maximum) {
               uint32_t left = value(a->src[0].src.ssa, a->src[0].swizzle[0]);
               uint32_t right = value(a->src[1].src.ssa, a->src[1].swizzle[0]);
               uint32_t lhs = left, rhs = right;
               if (signed_order) {
                  /* Flipping the sign bit maps signed order onto unsigned order. */
                  uint32_t sign = temporary++;
                  lhs = temporary++;
                  rhs = temporary++;
                  emit(&ops, 0x20, sign, 0, 0, 0, 0x80000000u);
                  emit(&ops, 0x27, lhs, left, sign, 0, 0);
                  emit(&ops, 0x27, rhs, right, sign, 0, 0);
               }
               uint32_t comparison = a->op == nir_op_ilt32 ? value(&a->def, 0) : temporary++;
               emit(&ops, 0x2a, comparison, lhs, rhs, 0, 0);
               if (minimum || maximum) {
                  emit(&ops, 0x2e, value(&a->def, 0), comparison,
                       minimum ? left : right, minimum ? right : left, 0);
               } else if (a->op == nir_op_ige32) {
                  uint32_t all = temporary++;
                  emit(&ops, 0x20, all, 0, 0, 0, UINT32_MAX);
                  emit(&ops, 0x27, value(&a->def, 0), comparison, all, 0, 0);
               }
               continue;
            }
            if (a->op==nir_op_inot || a->op==nir_op_ine32 || a->op==nir_op_uge32) {
               uint32_t all=temporary++, operand=value(a->src[0].src.ssa,a->src[0].swizzle[0]);
               emit(&ops,0x20,all,0,0,0,UINT32_MAX);
               if (a->op!=nir_op_inot) {
                  uint32_t comparison=temporary++;
                  emit(&ops,a->op==nir_op_ine32?0x2b:0x2a,comparison,operand,
                       value(a->src[1].src.ssa,a->src[1].swizzle[0]),0,0);
                  operand=comparison;
               }
               emit(&ops,0x27,value(&a->def,0),operand,all,0,0);
               continue;
            }
            if (nir_op_is_vec(a->op)) {
               for (unsigned j = 0; j < a->def.num_components; j++)
                  emit(&ops, 0x21, value(&a->def,j), value(a->src[j].src.ssa,a->src[j].swizzle[0]),0,0,0);
               continue;
            }
            if (!op || a->def.num_components != 1) goto unsupported;
            uint32_t src[3] = {0};
            for (unsigned j = 0; j < nir_op_infos[a->op].num_inputs; j++)
               src[j] = value(a->src[j].src.ssa, a->src[j].swizzle[0]);
            emit(&ops, op, value(&a->def,0), src[0], src[1], src[2], 0);
         } else if (instr->type == nir_instr_type_intrinsic) {
            nir_intrinsic_instr *i = nir_instr_as_intrinsic(instr);
            switch (i->intrinsic) {
            case nir_intrinsic_decl_reg:
               if (nir_intrinsic_bit_size(i)!=32 || nir_intrinsic_num_components(i)!=1 ||
                   nir_intrinsic_num_array_elems(i)) goto unsupported;
               emit(&ops,0x20,value(&i->def,0),0,0,0,0); break;
            case nir_intrinsic_load_reg:
               if (i->def.num_components!=1) goto unsupported;
               emit(&ops,0x21,value(&i->def,0),value(i->src[0].ssa,0),0,0,0); break;
            case nir_intrinsic_store_reg:
               if (i->num_components!=1) goto unsupported;
               emit(&ops,0x21,value(i->src[1].ssa,0),value(i->src[0].ssa,0),0,0,0); break;
            case nir_intrinsic_load_subgroup_invocation:
            case nir_intrinsic_load_local_invocation_index:
               emit(&ops,0x40,value(&i->def,0),0,0,0,i->intrinsic==nir_intrinsic_load_subgroup_invocation?0:1); break;
            case nir_intrinsic_load_global_invocation_id:
               for (unsigned j=0;j<3;j++) emit(&ops,0x40,value(&i->def,j),0,0,0,2+j);
               break;
            case nir_intrinsic_load_base_workgroup_id:
               for (unsigned j=0;j<3;j++) {
                  uint32_t scalar=temporary++;
                  emit(&ops,0x41,scalar,0,0,0,j);
                  emit(&ops,0x2d,value(&i->def,j),scalar,0,0,0);
               } break;
            case nir_intrinsic_shader_clock: {
               unsigned scope=nir_intrinsic_memory_scope(i);
               if (scope!=SCOPE_SUBGROUP && scope!=SCOPE_DEVICE) goto unsupported;
               uint32_t scalar=temporary++;
               emit(&ops,0x42,scalar,0,0,0,scope==SCOPE_DEVICE);
               for (unsigned j=0;j<2;j++) emit(&ops,0xf2,value(&i->def,j),scalar,0,0,j);
               break;
            }
            case nir_intrinsic_ballot: {
               uint32_t scalar=temporary++;
               emit(&ops,0x43,scalar,value(i->src[0].ssa,0),0,0,0);
               emit(&ops,0x2d,value(&i->def,0),scalar,0,0,0);
               for (unsigned j=1;j<i->def.num_components;j++) emit(&ops,0x20,value(&i->def,j),0,0,0,0);
               break;
            }
            case nir_intrinsic_read_invocation:
            case nir_intrinsic_shuffle:
               if (i->num_components!=1) goto unsupported;
               emit(&ops,0x44,value(&i->def,0),value(i->src[0].ssa,0),value(i->src[1].ssa,0),0,0); break;
            case nir_intrinsic_barrier:
               if (nir_intrinsic_execution_scope(i) == SCOPE_NONE) {
                  /* Shared storage is workgroup-local even when SPIR-V uses
                   * Device scope, as GLSL memoryBarrierShared does. */
                  if ((nir_intrinsic_memory_modes(i) &
                       ~(nir_var_mem_shared | nir_var_mem_ssbo | nir_var_mem_global)) ||
                      (nir_intrinsic_memory_semantics(i) & ~NIR_MEMORY_ACQ_REL)) goto unsupported;
               } else if (nir_intrinsic_execution_scope(i) != SCOPE_WORKGROUP ||
                          nir_intrinsic_memory_scope(i) > SCOPE_DEVICE) goto unsupported;
               /* The scheduler drains all memory tokens before this boundary.
                * One admitted 16-lane workgroup occupies one physical wave. */
               emit(&ops,7,0,0,0,0,0); break;
            case nir_intrinsic_load_scratch:
            case nir_intrinsic_store_scratch: {
               bool store=i->intrinsic==nir_intrinsic_store_scratch;
               if (i->num_components!=1 || nir_intrinsic_align_mul(i)<4 || nir_intrinsic_align_offset(i)%4) goto unsupported;
               emit(&ops,store?0x59:0x58,store?0:value(&i->def,0),value(i->src[store?1:0].ssa,0),store?value(i->src[0].ssa,0):0,0,0);
               break;
            }
            case nir_intrinsic_load_shared:
            case nir_intrinsic_store_shared: {
               bool store=i->intrinsic==nir_intrinsic_store_shared;
               if (i->num_components!=1 || nir_intrinsic_base(i)!=0 || nir_intrinsic_align_mul(i)<4 || nir_intrinsic_align_offset(i)%4) goto unsupported;
               emit(&ops,store?0x54:0x53,store?0:value(&i->def,0),value(i->src[store?1:0].ssa,0),store?value(i->src[0].ssa,0):0,0,0);
               break;
            }
            case nir_intrinsic_ssbo_atomic:
            case nir_intrinsic_ssbo_atomic_swap:
            case nir_intrinsic_global_atomic_2x32:
            case nir_intrinsic_global_atomic_swap_2x32:
            case nir_intrinsic_shared_atomic:
            case nir_intrinsic_shared_atomic_swap: {
               bool shared=i->intrinsic==nir_intrinsic_shared_atomic || i->intrinsic==nir_intrinsic_shared_atomic_swap;
               bool paired=i->intrinsic==nir_intrinsic_global_atomic_2x32 || i->intrinsic==nir_intrinsic_global_atomic_swap_2x32;
               int operation=atomic_op(nir_intrinsic_atomic_op(i));
               if (operation<0 || i->num_components!=1 || i->def.bit_size!=32) goto unsupported;
               uint32_t addr=value(i->src[shared || paired?0:1].ssa,0);
               unsigned data=shared || paired?1:2;
               if (shared) {if(nir_intrinsic_base(i)) goto unsupported;}
               else if (paired) {
                  nir_def *address=i->src[0].ssa;
                  if(address->bit_size!=32 || address->num_components!=2) goto unsupported;
                  addr=temporary++;
                  emit(&ops,0xf1,addr,value(address,0),value(address,1),0,0);
               }
               else {
                  if(!nir_src_is_const(i->src[0]) || nir_src_as_uint(i->src[0]) || nir_intrinsic_offset_shift(i)) goto unsupported;
                  uint32_t global=temporary++;emit(&ops,0xf0,global,addr,0,0,0);addr=global;
               }
               uint32_t operand=value(i->src[data].ssa,0);
               if (operation==2) {
                  uint32_t pair=temporary++;
                  emit(&ops,0xf1,pair,operand,value(i->src[data+1].ssa,0),0,0);operand=pair;
               }
               emit(&ops,shared?0x55:0x52,value(&i->def,0),addr,operand,0,operation);
               break;
            }
            case nir_intrinsic_vulkan_resource_index:
               if (nir_intrinsic_desc_set(i)!=0 || nir_intrinsic_binding(i)!=0 || !nir_src_is_const(i->src[0]) || nir_src_as_uint(i->src[0])!=0) goto unsupported;
               for (unsigned j=0;j<i->def.num_components;j++) emit(&ops,0x20,value(&i->def,j),0,0,0,0);
               break;
            case nir_intrinsic_load_vulkan_descriptor:
               for (unsigned j=0;j<i->def.num_components;j++) emit(&ops,0x21,value(&i->def,j),value(i->src[0].ssa,j),0,0,0);
               break;
            case nir_intrinsic_load_global_2x32:
            case nir_intrinsic_store_global_2x32: {
               bool store = i->intrinsic == nir_intrinsic_store_global_2x32;
               nir_def *address = i->src[store ? 1 : 0].ssa;
               if (i->num_components != 1 || address->bit_size != 32 || address->num_components != 2 ||
                   nir_intrinsic_align_mul(i) < 4 || nir_intrinsic_align_offset(i) % 4 ||
                   (store && nir_intrinsic_write_mask(i) != 1)) goto unsupported;
               uint32_t pair = temporary++;
               emit(&ops, 0xf1, pair, value(address, 0), value(address, 1), 0, 0);
               emit(&ops, store ? 0x51 : 0x50, store ? 0 : value(&i->def, 0), pair,
                    store ? value(i->src[0].ssa, 0) : 0, 0, 0);
               break;
            }
            case nir_intrinsic_load_ssbo:
            case nir_intrinsic_store_ssbo: {
               bool store=i->intrinsic==nir_intrinsic_store_ssbo;
               unsigned index=store?1:0, offset=store?2:1;
               if (!nir_src_is_const(i->src[index]) || nir_src_as_uint(i->src[index])!=0 || i->num_components!=1 || nir_intrinsic_align_mul(i)<4 || nir_intrinsic_align_offset(i)%4 || nir_intrinsic_offset_shift(i)) goto unsupported;
               uint32_t addr=temporary++;
               emit(&ops,0xf0,addr,value(i->src[offset].ssa,0),0,0,0);
               emit(&ops,store?0x51:0x50,store?0:value(&i->def,0),addr,store?value(i->src[0].ssa,0):0,0,0); break;
            }
            default: goto unsupported;
            }
         } else if (instr->type == nir_instr_type_jump) {
            nir_jump_instr *jump = nir_instr_as_jump(instr);
            if (!control->loop || (jump->type!=nir_jump_break && jump->type!=nir_jump_continue)) goto unsupported;
            uint32_t current=temporary++;
            emit(&ops,6,current,control->zero,0,0,0);
            emit(&ops,0x17,control->loop->iteration,control->loop->iteration,current,0,0);
            if (jump->type==nir_jump_break)
               emit(&ops,0x17,control->loop->live,control->loop->live,current,0,0);
         } else { goto unsupported; }
         continue;
unsupported:
         if (instr->type == nir_instr_type_intrinsic)
            snprintf(control->output->diagnostic, sizeof(control->output->diagnostic),
                     "unsupported normalized NIR intrinsic: %s",
                     nir_intrinsic_infos[nir_instr_as_intrinsic(instr)->intrinsic].name);
         else if (instr->type == nir_instr_type_alu)
            snprintf(control->output->diagnostic, sizeof(control->output->diagnostic),
                     "unsupported normalized NIR ALU: %s",
                     nir_op_infos[nir_instr_as_alu(instr)->op].name);
         else
            snprintf(control->output->diagnostic, sizeof(control->output->diagnostic),
                     "unsupported normalized NIR instruction type: %u", instr->type);
         result=false; goto done;
      }
   }
done:
   *output=ops;
   *next_temporary=temporary;
   return result;
}

static void restore_mask(struct control_state *c, uint32_t mask)
{
   if (c->loop) {
      uint32_t filtered=(*c->temporary)++;
      emit(c->ops,0x15,filtered,mask,c->loop->iteration,0,0);
      mask=filtered;
   }
   emit(c->ops,6,c->discard,mask,0,0,0);
}

static bool emit_cf(struct control_state *c, struct exec_list *list)
{
   foreach_list_typed(nir_cf_node,node,node,list) {
      if (node->type==nir_cf_node_block) {
         if (!emit_block(c->ops,nir_cf_node_as_block(node),c->temporary,c)) return false;
      } else if (node->type==nir_cf_node_if) {
         nir_if *nif=nir_cf_node_as_if(node);
         uint32_t condition=(*c->temporary)++, saved=(*c->temporary)++, otherwise=(*c->temporary)++;
         emit(c->ops,0x43,condition,value(nif->condition.ssa,0),0,0,0);
         emit(c->ops,6,saved,condition,0,0,0);
         if (!emit_cf(c,&nif->then_list)) return false;
         emit(c->ops,0x17,otherwise,saved,condition,0,0);
         restore_mask(c,otherwise);
         if (!emit_cf(c,&nif->else_list)) return false;
         restore_mask(c,saved);
      } else if (node->type==nir_cf_node_loop) {
         nir_loop *loop=nir_cf_node_as_loop(node);
         if (!exec_list_is_empty(&loop->continue_list)) {
            fail(c->output, "loop continue construct must be lowered"); return false;
         }
         uint32_t saved=(*c->temporary)++;
         struct loop_masks masks={.live=(*c->temporary)++, .iteration=(*c->temporary)++, .parent=c->loop};
         emit(c->ops,6,saved,c->zero,0,0,0);
         emit(c->ops,0x11,masks.live,saved,0,0,0);
         uint32_t header=util_dynarray_num_elements(c->ops,struct apex_op);
         emit(c->ops,0x11,masks.iteration,masks.live,0,0,0);
         emit(c->ops,6,c->discard,masks.iteration,0,0,0);
         c->loop=&masks;
         if (!emit_cf(c,&loop->body)) return false;
         emit(c->ops,5,0,masks.live,0,0,header);
         c->loop=masks.parent;
         restore_mask(c,saved);
      } else {
         fail(c->output, "unsupported control-flow node"); return false;
      }
   }
   return true;
}
