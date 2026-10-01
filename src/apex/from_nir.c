/* SPDX-License-Identifier: MIT */
/* NIR to Apex machine IR. Divergence analysis places uniform values in scalar
 * registers; divergent booleans are lane masks and uniform ones 0/1 flags. */
#include "apex.h"
#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "util/u_dynarray.h"
#include "util/u_math.h"
#include <stdio.h>

const struct nir_shader_compiler_options apex_nir_options = {
   .lower_fdiv = true, .lower_flrp32 = true, .lower_fpow = true, .lower_fmod = true,
   .lower_scmp = true, .lower_fdph = true, .lower_fisnormal = true, .lower_fsign = true,
   .lower_isign = true, .lower_usub_sat = true, .lower_uadd_sat = true, .lower_iadd_sat = true,
   .lower_hadd = true, .lower_mul_2x32_64 = true, .lower_bitfield_insert = true,
   .lower_insert_byte = true, .lower_insert_word = true, .lower_uadd_carry = true,
   .lower_usub_borrow = true, .lower_fquantize2f16 = true,
   .lower_pack_unorm_2x16 = true, .lower_pack_snorm_2x16 = true, .lower_pack_unorm_4x8 = true,
   .lower_pack_snorm_4x8 = true, .lower_unpack_unorm_2x16 = true,
   .lower_unpack_snorm_2x16 = true, .lower_unpack_unorm_4x8 = true,
   .lower_unpack_snorm_4x8 = true, .lower_pack_half_2x16 = true,
   .lower_pack_32_2x16_split = true, .lower_unpack_32_2x16_split = true,
   .lower_pack_64_2x32 = true, .lower_pack_split = true, .lower_extract_byte = false,
   .has_bitfield_select = true, .has_ldexp = true, .lower_device_index_to_zero = true,
   .float_mul_add32 = nir_float_muladd_support_has_ffma | nir_float_muladd_support_fuse,
   .lower_int64_options = ~0, .lower_doubles_options = ~0,
   .max_unroll_iterations = 32,
};

/* Standalone resource layout (apex-compile and compiler tests): user data
 * s0-s1 hold the root GPUVA; set s < 8 binding b < 256 element e has a
 * 64-byte slot at root + 64 (1 + 256 s + b + e) (buffer descriptor, or image
 * then sampler descriptor); push constants start at root + 64 * 2049; s2-s4
 * hold the grid size. */
#define STANDALONE_PUSH (64u * 2049u)
#define SLOT_MARK 0x80000000u

struct def_map { uint32_t id; uint8_t off; };
struct loop_state {
   bool divergent;
   uint32_t live, iter, head, latch, end;
   struct loop_state *parent;
};
struct ctx {
   nir_shader *nir;
   struct util_dynarray ops, values;
   struct def_map *defs;
   bool *vector, *saturated;
   struct hash_table *range;
   uint32_t labels, restore, end;
   struct loop_state *loop;
   unsigned depth;
   /* Fragment masks: lanes not terminated, lanes not demoted, launch helpers. */
   uint32_t alive, covered, helper, root;
   uint32_t out[48];
   uint8_t out_written[48];
   struct apex_header header;
   bool failed;
   char *diagnostic;
};

static void failf(struct ctx *c, const char *message, const char *detail)
{
   if (!c->failed)
      snprintf(c->diagnostic, 1024, "%s%s", message, detail ? detail : "");
   c->failed = true;
}

static uint32_t new_value(struct ctx *c, bool vector, unsigned width)
{
   struct apex_value v = {vector, width};
   util_dynarray_append(&c->values, v);
   return util_dynarray_num_elements(&c->values, struct apex_value) - 1;
}
static bool is_vec(struct ctx *c, uint32_t operand)
{
   if (!(operand >> 31))
      return (operand >> 30) & 1 && (operand & 0xff) < APEX_SCALAR;
   return util_dynarray_element(&c->values, struct apex_value, apex_val_id(operand))->cls;
}
static unsigned width_of(uint32_t operand) { return operand >> 31 ? apex_val_n(operand) : 1; }
static uint32_t sub(uint32_t operand, unsigned off, unsigned n)
{
   return apex_val(apex_val_id(operand), apex_val_off(operand) + off, n);
}

static unsigned emit(struct ctx *c, unsigned op, uint32_t d, uint32_t a, uint32_t b, uint32_t cc,
                     uint32_t hi)
{
   struct apex_op o = {op, {d, a, b, cc}, hi, 0, 0};
   util_dynarray_append(&c->ops, o);
   return util_dynarray_num_elements(&c->ops, struct apex_op) - 1;
}
static struct apex_op *last(struct ctx *c)
{
   return util_dynarray_top_ptr(&c->ops, struct apex_op);
}
static uint32_t value(struct ctx *c, bool vector, unsigned width)
{
   return apex_val(new_value(c, vector, width), 0, width);
}
static uint32_t constant(struct ctx *c, uint32_t v)
{
   uint32_t d = value(c, false, 1);
   emit(c, APEX_CONST, d, 0, 0, 0, 0);
   last(c)->imm = v;
   return d;
}
static uint32_t copy(struct ctx *c, bool vector, uint32_t src)
{
   uint32_t d = value(c, vector, width_of(src));
   emit(c, APEX_COPY, d, src, 0, 0, 0);
   return d;
}
/* A scalar view of a uniform operand. */
static uint32_t scalar(struct ctx *c, uint32_t x)
{
   return is_vec(c, x) ? copy(c, false, x) : x;
}
static uint32_t vector(struct ctx *c, uint32_t x)
{
   return is_vec(c, x) ? x : copy(c, true, x);
}
static uint32_t op1(struct ctx *c, unsigned op, bool vec, uint32_t a)
{
   uint32_t d = value(c, vec, 1);
   emit(c, op, d, a, 0, 0, 0);
   return d;
}
static uint32_t op2(struct ctx *c, unsigned op, bool vec, uint32_t a, uint32_t b)
{
   uint32_t d = value(c, vec, 1);
   emit(c, op, d, a, b, 0, 0);
   return d;
}
static uint32_t label(struct ctx *c) { return ++c->labels; }
static void place(struct ctx *c, uint32_t l)
{
   emit(c, APEX_LABEL, 0, 0, 0, 0, 0);
   last(c)->imm = l;
}
static void branch(struct ctx *c, unsigned op, uint32_t cond, uint32_t l)
{
   emit(c, op, 0, cond, 0, 0, 0);
   last(c)->imm = l;
}
static void set_exec(struct ctx *c, uint32_t x)
{
   emit(c, APEX_S_SETEXEC, 0, x, 0, 0, 0);
}
static uint32_t exec_copy(struct ctx *c)
{
   return copy(c, false, apex_phys(APEX_EXEC));
}

/* NIR values. */
static unsigned words(nir_def *d) { return d->bit_size == 64 ? 2 : 1; }
static uint32_t def_operand(struct ctx *c, nir_def *d, unsigned comp)
{
   struct def_map m = c->defs[d->index];
   if (!m.id) {
      failf(c, "internal: use of an unemitted value", NULL);
      return apex_val(1, 0, 1);
   }
   return apex_val(m.id, m.off + comp * words(d), words(d));
}
static uint32_t src(struct ctx *c, nir_src s, unsigned comp) { return def_operand(c, s.ssa, comp); }
static uint32_t alu_src(struct ctx *c, nir_alu_instr *a, unsigned i)
{
   return src(c, a->src[i].src, a->src[i].swizzle[0]);
}
/* A trivial store_reg (nir_trivialize_registers) lets the producer write the
 * register directly. */
static nir_intrinsic_instr *register_store(nir_def *d)
{
   if (d->bit_size == 1 || !list_is_singular(&d->uses))
      return NULL;
   nir_src *use = list_first_entry(&d->uses, nir_src, use_link);
   if (nir_src_is_if(use) || nir_src_use_instr(use)->type != nir_instr_type_intrinsic)
      return NULL;
   nir_intrinsic_instr *store = nir_instr_as_intrinsic(nir_src_use_instr(use));
   if (store->intrinsic != nir_intrinsic_store_reg || use != &store->src[0] ||
       nir_intrinsic_write_mask(store) != BITFIELD_MASK(d->num_components))
      return NULL;
   nir_intrinsic_instr *decl = nir_def_as_intrinsic(store->src[1].ssa);
   if (nir_intrinsic_num_components(decl) != d->num_components ||
       nir_intrinsic_bit_size(decl) != d->bit_size)
      return NULL;
   return store;
}
static uint32_t define(struct ctx *c, nir_def *d, bool vec)
{
   nir_intrinsic_instr *store = register_store(d);
   if (store) {
      struct def_map reg = c->defs[store->src[1].ssa->index];
      if (reg.id && util_dynarray_element(&c->values, struct apex_value, reg.id)->cls == (vec && d->bit_size != 1)) {
         c->defs[d->index] = reg;
         return apex_val(reg.id, 0, words(d) * d->num_components);
      }
   }
   unsigned w = words(d) * d->num_components;
   uint32_t id = new_value(c, vec && d->bit_size != 1, w);
   c->defs[d->index] = (struct def_map){id, 0};
   return apex_val(id, 0, w);
}
static void alias(struct ctx *c, nir_def *d, uint32_t operand)
{
   c->defs[d->index] = (struct def_map){apex_val_id(operand), apex_val_off(operand)};
}
static bool vec_def(struct ctx *c, nir_def *d) { return c->vector[d->index]; }

/* Booleans: a divergent bool is a lane mask, a uniform one a 0/1 flag. */
static uint32_t mask_of(struct ctx *c, nir_src s, unsigned comp)
{
   uint32_t x = src(c, s, comp);
   if (s.ssa->divergent)
      return x;
   /* All lanes when set, no lanes when clear. */
   return op2(c, APEX_S_SUB, false, constant(c, 0), x);
}
static uint32_t flag_of(struct ctx *c, nir_src s, unsigned comp)
{
   if (s.ssa->divergent)
      failf(c, "internal: divergent boolean used as uniform", NULL);
   return src(c, s, comp);
}
static uint32_t mask_to_flag(struct ctx *c, uint32_t m)
{
   return op2(c, APEX_S_CMP_NE, false, m, constant(c, 0));
}

/* Class selection. A uniform value lives in scalar registers when a scalar
 * form computes it from scalar sources; FP add, multiply, FMA, reciprocal and
 * its root have scalar forms with the vector forms' bits. */
static bool scalar_alu(nir_op op)
{
   switch (op) {
   case nir_op_iadd: case nir_op_isub: case nir_op_imul: case nir_op_umul_high:
   case nir_op_iand: case nir_op_ior: case nir_op_ixor: case nir_op_inot: case nir_op_ishl:
   case nir_op_ushr: case nir_op_ishr: case nir_op_imin: case nir_op_imax: case nir_op_umin:
   case nir_op_umax: case nir_op_ineg: case nir_op_iabs: case nir_op_ubitfield_extract:
   case nir_op_ibitfield_extract: case nir_op_extract_u8: case nir_op_extract_i8:
   case nir_op_extract_u16: case nir_op_extract_i16: case nir_op_bit_count: case nir_op_find_lsb:
   case nir_op_mov: case nir_op_vec2: case nir_op_vec3: case nir_op_vec4: case nir_op_vec5:
   case nir_op_vec8: case nir_op_vec16: case nir_op_bcsel: case nir_op_b2i32: case nir_op_b2f32:
   case nir_op_fneg: case nir_op_fabs: case nir_op_pack_64_2x32_split:
   case nir_op_unpack_64_2x32_split_x: case nir_op_unpack_64_2x32_split_y:
   case nir_op_u2u32: case nir_op_i2i32: case nir_op_fadd: case nir_op_fsub: case nir_op_fmul:
   case nir_op_ffma: case nir_op_ffma_weak: case nir_op_frcp: case nir_op_frsq: case nir_op_fsat:
      return true;
   default:
      return false;
   }
}

static bool root_load_scalar(nir_intrinsic_instr *i, unsigned address)
{
   return !i->src[address].ssa->divergent &&
          ((nir_intrinsic_access(i) & ACCESS_CAN_REORDER) || i->intrinsic == nir_intrinsic_load_ubo);
}

static bool intrinsic_scalar(struct ctx *c, nir_intrinsic_instr *i)
{
   switch (i->intrinsic) {
   case nir_intrinsic_load_workgroup_id: case nir_intrinsic_load_subgroup_id:
   case nir_intrinsic_load_num_subgroups: case nir_intrinsic_load_subgroup_size:
   case nir_intrinsic_load_view_index: case nir_intrinsic_shader_clock:
   case nir_intrinsic_ballot: case nir_intrinsic_read_invocation:
   case nir_intrinsic_read_first_invocation: case nir_intrinsic_first_invocation:
   case nir_intrinsic_load_kernel_input: case nir_intrinsic_load_base_vertex:
   case nir_intrinsic_load_first_vertex: case nir_intrinsic_load_base_instance:
   case nir_intrinsic_load_draw_id: case nir_intrinsic_load_push_constant:
   case nir_intrinsic_load_base_workgroup_id: case nir_intrinsic_load_workgroup_size:
      return true;
   case nir_intrinsic_load_ssbo: case nir_intrinsic_load_ubo:
      return root_load_scalar(i, 1) && !i->src[0].ssa->divergent;
   case nir_intrinsic_get_ssbo_size:
      return !i->src[0].ssa->divergent;
   case nir_intrinsic_load_global_2x32:
      return root_load_scalar(i, 0);
   case nir_intrinsic_load_reg:
      return !c->vector[i->src[0].ssa->index];
   default:
      return false;
   }
}

static bool all_scalar(struct ctx *c, nir_alu_instr *a)
{
   for (unsigned s = 0; s < nir_op_infos[a->op].num_inputs; s++)
      if (a->src[s].src.ssa->bit_size != 1 && c->vector[a->src[s].src.ssa->index])
         return false;
   return true;
}

/* Decides the class of every value; registers iterate to a fixed point. */
static void classify(struct ctx *c, nir_function_impl *impl)
{
   bool changed;
   do {
      changed = false;
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            nir_def *d = nir_instr_def(instr);
            bool vec = false;
            if (instr->type == nir_instr_type_alu) {
               nir_alu_instr *a = nir_instr_as_alu(instr);
               vec = d->divergent || !scalar_alu(a->op) || !all_scalar(c, a);
            } else if (instr->type == nir_instr_type_intrinsic) {
               nir_intrinsic_instr *i = nir_instr_as_intrinsic(instr);
               if (i->intrinsic == nir_intrinsic_store_reg) {
                  nir_def *reg = i->src[1].ssa;
                  if (i->src[0].ssa->bit_size != 1 && c->vector[i->src[0].ssa->index] &&
                      !c->vector[reg->index]) {
                     c->vector[reg->index] = true;
                     changed = true;
                  }
                  continue;
               }
               if (!d)
                  continue;
               if (i->intrinsic == nir_intrinsic_decl_reg) {
                  if (nir_intrinsic_divergent(i) && !c->vector[d->index]) {
                     c->vector[d->index] = true;
                     changed = true;
                  }
                  continue;
               }
               vec = d->divergent || !intrinsic_scalar(c, i);
            } else if (instr->type == nir_instr_type_load_const ||
                       instr->type == nir_instr_type_undef) {
               vec = false;
            } else {
               vec = d != NULL;
            }
            nir_intrinsic_instr *store = d ? register_store(d) : NULL;
            if (store && c->vector[store->src[1].ssa->index] &&
                (instr->type == nir_instr_type_alu || instr->type == nir_instr_type_intrinsic ||
                 instr->type == nir_instr_type_tex))
               vec = true;
            if (d && d->bit_size == 1)
               vec = false;
            if (d && vec != c->vector[d->index]) {
               c->vector[d->index] = vec;
               changed = true;
            }
         }
      }
   } while (changed);
}

/* Floating-point source with folded negate/absolute modifiers. */
static uint32_t fp_src(struct ctx *c, nir_alu_instr *a, unsigned i, unsigned field, uint32_t *hi)
{
   nir_src s = a->src[i].src;
   unsigned comp = a->src[i].swizzle[0];
   bool neg = false, abs = false;
   for (;;) {
      nir_alu_instr *p = nir_def_as_alu_or_null(s.ssa);
      if (!p || (p->op != nir_op_fneg && p->op != nir_op_fabs) || p->def.bit_size != 32)
         break;
      if (p->op == nir_op_fneg && !abs)
         neg = !neg;
      if (p->op == nir_op_fabs)
         abs = true;
      comp = p->src[0].swizzle[comp];
      s = p->src[0].src;
   }
   *hi |= (unsigned)neg << (7 + field) | (unsigned)abs << (10 + field);
   return src(c, s, comp);
}
static bool single_fsat_use(nir_def *d)
{
   if (!list_is_singular(&d->uses))
      return false;
   nir_src *use = list_first_entry(&d->uses, nir_src, use_link);
   if (nir_src_is_if(use))
      return false;
   nir_instr *user = nir_src_use_instr(use);
   return user->type == nir_instr_type_alu && nir_instr_as_alu(user)->op == nir_op_fsat;
}

/* The scalar form of a vector FP operation; subtraction negates b. */
static unsigned scalar_fp(unsigned op)
{
   switch (op) {
   case APEX_V_ADD_F: case APEX_V_SUB_F: return APEX_S_ADD_F;
   case APEX_V_MUL_F: return APEX_S_MUL_F;
   case APEX_V_FMA_F: return APEX_S_FMA_F;
   case APEX_V_RCP: return APEX_S_RCP;
   case APEX_V_RSQ: return APEX_S_RSQ;
   default: return 0;
   }
}

static void fp_op(struct ctx *c, nir_alu_instr *a, unsigned op)
{
   unsigned n = nir_op_infos[a->op].num_inputs;
   uint32_t hi = 0, s[3] = {0};
   for (unsigned i = 0; i < n; i++)
      s[i] = fp_src(c, a, i, i + 1, &hi);
   bool vec = vec_def(c, &a->def);
   if (!vec) {
      if (op == APEX_V_SUB_F)
         hi ^= 1u << 9;
      op = scalar_fp(op);
   }
   if (op == APEX_V_LDEXP) {
      hi &= ~((1u << 9) | (1u << 12));
      s[1] = alu_src(c, a, 1);
   }
   if (single_fsat_use(&a->def)) {
      hi |= 1u << 14;
      c->saturated[a->def.index] = true;
   }
   emit(c, op, define(c, &a->def, vec), s[0], s[1], n > 2 ? s[2] : 0, hi);
}

static void int_op(struct ctx *c, nir_alu_instr *a, unsigned vop, unsigned sop)
{
   bool vec = vec_def(c, &a->def);
   if (!vec && !sop)
      vec = true;
   unsigned n = nir_op_infos[a->op].num_inputs;
   uint32_t s[3] = {0};
   for (unsigned i = 0; i < n; i++)
      s[i] = alu_src(c, a, i);
   emit(c, vec ? vop : sop, define(c, &a->def, vec), s[0], s[1], n > 2 ? s[2] : 0, 0);
}

static void compare(struct ctx *c, nir_alu_instr *a, unsigned vop, unsigned cond, bool swap,
                    unsigned sop)
{
   uint32_t x, y, hi = 0;
   if (vop == APEX_V_CMP_F) {
      x = fp_src(c, a, 0, 1, &hi);
      y = fp_src(c, a, 1, 2, &hi);
   } else {
      x = alu_src(c, a, 0);
      y = alu_src(c, a, 1);
   }
   bool divergent = a->def.divergent;
   if (!divergent && sop && !is_vec(c, x) && !is_vec(c, y)) {
      emit(c, sop, define(c, &a->def, false), swap ? y : x, swap ? x : y, 0, 0);
      return;
   }
   uint32_t m = divergent ? define(c, &a->def, false) : value(c, false, 1);
   emit(c, vop, m, x, y, apex_raw(cond), hi);
   if (!divergent)
      alias(c, &a->def, mask_to_flag(c, m));
}

static void bool_op(struct ctx *c, nir_alu_instr *a)
{
   bool divergent = a->def.divergent;
   unsigned n = nir_op_infos[a->op].num_inputs;
   uint32_t s[3];
   for (unsigned i = 0; i < n; i++)
      s[i] = divergent ? mask_of(c, a->src[i].src, a->src[i].swizzle[0])
                       : flag_of(c, a->src[i].src, a->src[i].swizzle[0]);
   uint32_t d = define(c, &a->def, false);
   switch (a->op) {
   case nir_op_iand: emit(c, APEX_S_AND, d, s[0], s[1], 0, 0); break;
   case nir_op_ior: emit(c, APEX_S_OR, d, s[0], s[1], 0, 0); break;
   case nir_op_ixor: case nir_op_ine: emit(c, APEX_S_XOR, d, s[0], s[1], 0, 0); break;
   case nir_op_ieq:
      if (divergent)
         emit(c, APEX_S_XOR, d, op2(c, APEX_S_XOR, false, s[0], s[1]), constant(c, ~0u), 0, 0);
      else
         emit(c, APEX_S_CMP_EQ, d, s[0], s[1], 0, 0);
      break;
   case nir_op_inot:
      if (divergent)
         emit(c, APEX_S_NOT, d, s[0], 0, 0, 0);
      else
         emit(c, APEX_S_XOR, d, s[0], constant(c, 1), 0, 0);
      break;
   case nir_op_bcsel:
      if (divergent)
         emit(c, APEX_S_OR, d, op2(c, APEX_S_AND, false, s[0], s[1]),
              op2(c, APEX_S_ANDN2, false, s[2], s[0]), 0, 0);
      else
         emit(c, APEX_S_CSELECT, d, s[1], s[2], s[0], 0);
      break;
   case nir_op_mov:
      emit(c, APEX_COPY, d, s[0], 0, 0, 0);
      break;
   default:
      failf(c, "unsupported boolean ALU: ", nir_op_infos[a->op].name);
   }
}

static uint32_t quadperm(struct ctx *c, uint32_t x, unsigned pattern)
{
   return op2(c, APEX_V_QUADPERM, true, vector(c, x), constant(c, pattern));
}

static void emit_alu(struct ctx *c, nir_alu_instr *a)
{
   nir_def *d = &a->def;
   bool vec = vec_def(c, d);
   if (d->bit_size == 1 || (a->op == nir_op_bcsel && d->bit_size == 1)) {
      switch (a->op) {
      case nir_op_ieq: case nir_op_ine:
         if (a->src[0].src.ssa->bit_size == 1) { bool_op(c, a); return; }
         break;
      case nir_op_iand: case nir_op_ior: case nir_op_ixor: case nir_op_inot:
      case nir_op_bcsel: case nir_op_mov:
         bool_op(c, a);
         return;
      default:
         break;
      }
   }
   switch (a->op) {
   case nir_op_mov:
      alias(c, d, alu_src(c, a, 0));
      return;
   case nir_op_u2u32: case nir_op_i2i32: case nir_op_f2f32:
      if (a->src[0].src.ssa->bit_size == 32) { alias(c, d, alu_src(c, a, 0)); return; }
      break;
   case nir_op_vec2: case nir_op_vec3: case nir_op_vec4: case nir_op_vec5: case nir_op_vec8:
   case nir_op_vec16: {
      unsigned n = d->num_components, w = words(d);
      uint32_t first = alu_src(c, a, 0);
      bool consecutive = true;
      for (unsigned i = 1; i < n; i++)
         consecutive &= alu_src(c, a, i) == sub(first, i * w, w);
      if (consecutive && (!is_vec(c, first) || vec) && (is_vec(c, first) || !vec)) {
         alias(c, d, first);
         return;
      }
      uint32_t g = define(c, d, vec);
      for (unsigned i = 0; i < n; i++)
         emit(c, APEX_COPY, sub(g, i * w, w), alu_src(c, a, i), 0, 0, 0);
      return;
   }
   case nir_op_pack_64_2x32_split: {
      uint32_t lo = alu_src(c, a, 0), hi = alu_src(c, a, 1);
      if (hi == sub(lo, 1, 1) && is_vec(c, lo) == vec) {
         alias(c, d, sub(lo, 0, 2));
         return;
      }
      uint32_t g = define(c, d, vec);
      emit(c, APEX_COPY, sub(g, 0, 1), lo, 0, 0, 0);
      emit(c, APEX_COPY, sub(g, 1, 1), hi, 0, 0, 0);
      return;
   }
   case nir_op_unpack_64_2x32_split_x: case nir_op_unpack_64_2x32_split_y:
      alias(c, d, sub(alu_src(c, a, 0), a->op == nir_op_unpack_64_2x32_split_y, 1));
      return;
   case nir_op_unpack_half_2x16: {
      uint32_t x = alu_src(c, a, 0), g = define(c, d, true);
      emit(c, APEX_V_CVT_F16_LO, sub(g, 0, 1), x, 0, 0, 0);
      emit(c, APEX_V_CVT_F16_HI, sub(g, 1, 1), x, 0, 0, 0);
      return;
   }
   case nir_op_fsat:
      if (c->saturated[a->src[0].src.ssa->index]) {
         alias(c, d, alu_src(c, a, 0));
         return;
      } else {
         /* max(x, 0) clamped, or x × 1 clamped: NaN and -0 give +0 in both. */
         uint32_t hi = 1u << 14;
         uint32_t x = fp_src(c, a, 0, 1, &hi);
         emit(c, vec ? APEX_V_MAX_F : APEX_S_MUL_F, define(c, d, vec), x,
              constant(c, vec ? 0 : 0x3f800000u), 0, hi);
         return;
      }
   default:
      break;
   }
   if (d->num_components != 1 || (d->bit_size != 32 && d->bit_size != 1)) {
      failf(c, "unsupported normalized NIR ALU: ", nir_op_infos[a->op].name);
      return;
   }
   switch (a->op) {
   case nir_op_iadd: int_op(c, a, APEX_V_ADD, APEX_S_ADD); return;
   case nir_op_isub: int_op(c, a, APEX_V_SUB, APEX_S_SUB); return;
   case nir_op_imul: int_op(c, a, APEX_V_MUL_LO, APEX_S_MUL); return;
   case nir_op_umul_high: int_op(c, a, APEX_V_MUL_HI_U, APEX_S_MUL_HI_U); return;
   case nir_op_imul_high: int_op(c, a, APEX_V_MUL_HI_I, 0); return;
   case nir_op_iand: int_op(c, a, APEX_V_AND, APEX_S_AND); return;
   case nir_op_ior: int_op(c, a, APEX_V_OR, APEX_S_OR); return;
   case nir_op_ixor: int_op(c, a, APEX_V_XOR, APEX_S_XOR); return;
   case nir_op_inot: int_op(c, a, APEX_V_NOT, APEX_S_NOT); return;
   case nir_op_ishl: int_op(c, a, APEX_V_SHL, APEX_S_SHL); return;
   case nir_op_ushr: int_op(c, a, APEX_V_SHR, APEX_S_SHR); return;
   case nir_op_ishr: int_op(c, a, APEX_V_ASHR, APEX_S_ASHR); return;
   case nir_op_imin: int_op(c, a, APEX_V_MIN_I, APEX_S_MIN_I); return;
   case nir_op_imax: int_op(c, a, APEX_V_MAX_I, APEX_S_MAX_I); return;
   case nir_op_umin: int_op(c, a, APEX_V_MIN_U, APEX_S_MIN_U); return;
   case nir_op_umax: int_op(c, a, APEX_V_MAX_U, APEX_S_MAX_U); return;
   case nir_op_bit_count: int_op(c, a, APEX_V_POPCNT, APEX_S_POPCNT); return;
   case nir_op_find_lsb: int_op(c, a, APEX_V_FFBL, APEX_S_FF1); return;
   case nir_op_ufind_msb: int_op(c, a, APEX_V_FFBH_U, 0); return;
   case nir_op_ifind_msb: int_op(c, a, APEX_V_FFBH_I, 0); return;
   case nir_op_bitfield_reverse: int_op(c, a, APEX_V_BFREV, 0); return;
   case nir_op_bitfield_select: int_op(c, a, APEX_V_BFI, 0); return;
   case nir_op_ineg:
      emit(c, vec ? APEX_V_SUB : APEX_S_SUB, define(c, d, vec), constant(c, 0), alu_src(c, a, 0), 0, 0);
      return;
   case nir_op_iabs: {
      uint32_t x = alu_src(c, a, 0);
      uint32_t n = op2(c, vec ? APEX_V_SUB : APEX_S_SUB, vec, constant(c, 0), x);
      emit(c, vec ? APEX_V_MAX_I : APEX_S_MAX_I, define(c, d, vec), x, n, 0, 0);
      return;
   }
   case nir_op_fneg: case nir_op_fabs:
      emit(c, a->op == nir_op_fneg ? (vec ? APEX_V_XOR : APEX_S_XOR) : (vec ? APEX_V_AND : APEX_S_AND),
           define(c, d, vec), alu_src(c, a, 0),
           constant(c, a->op == nir_op_fneg ? 0x80000000u : 0x7fffffffu), 0, 0);
      return;
   case nir_op_extract_u8: case nir_op_extract_i8: case nir_op_extract_u16:
   case nir_op_extract_i16: {
      unsigned bits = a->op == nir_op_extract_u8 || a->op == nir_op_extract_i8 ? 8 : 16;
      bool sign = a->op == nir_op_extract_i8 || a->op == nir_op_extract_i16;
      unsigned k = nir_src_comp_as_uint(a->src[1].src, a->src[1].swizzle[0]);
      emit(c, vec ? (sign ? APEX_V_BFE_I : APEX_V_BFE_U) : (sign ? APEX_S_BFE_I : APEX_S_BFE_U),
           define(c, d, vec), alu_src(c, a, 0), constant(c, (k * bits) | bits << 8), 0, 0);
      return;
   }
   case nir_op_ubitfield_extract: case nir_op_ibitfield_extract: {
      bool sign = a->op == nir_op_ibitfield_extract;
      uint32_t field;
      if (nir_src_is_const(a->src[1].src) && nir_src_is_const(a->src[2].src)) {
         field = constant(c, (nir_src_comp_as_uint(a->src[1].src, a->src[1].swizzle[0]) & 31) |
                          (MIN2(nir_src_comp_as_uint(a->src[2].src, a->src[2].swizzle[0]), 32) << 8));
      } else {
         uint32_t off = alu_src(c, a, 1), bits = alu_src(c, a, 2);
         bool v = is_vec(c, off) || is_vec(c, bits);
         uint32_t shifted = op2(c, v ? APEX_V_SHL : APEX_S_SHL, v, bits, constant(c, 8));
         field = op2(c, v ? APEX_V_OR : APEX_S_OR, v,
                     op2(c, v ? APEX_V_AND : APEX_S_AND, v, off, constant(c, 31)), shifted);
         vec |= v;
      }
      emit(c, vec ? (sign ? APEX_V_BFE_I : APEX_V_BFE_U) : (sign ? APEX_S_BFE_I : APEX_S_BFE_U),
           define(c, d, vec), alu_src(c, a, 0), field, 0, 0);
      return;
   }
   case nir_op_bcsel: {
      nir_src cond = a->src[0].src;
      uint32_t t = alu_src(c, a, 1), f = alu_src(c, a, 2);
      if (!vec && !cond.ssa->divergent && !is_vec(c, t) && !is_vec(c, f)) {
         emit(c, APEX_S_CSELECT, define(c, d, false), t, f, flag_of(c, cond, a->src[0].swizzle[0]), 0);
         return;
      }
      emit(c, APEX_V_CNDMASK, define(c, d, true), f, t, mask_of(c, cond, a->src[0].swizzle[0]), 0);
      return;
   }
   case nir_op_b2i32: case nir_op_b2f32: {
      nir_src b = a->src[0].src;
      uint32_t one = constant(c, a->op == nir_op_b2i32 ? 1 : 0x3f800000u);
      if (!b.ssa->divergent) {
         uint32_t fl = flag_of(c, b, a->src[0].swizzle[0]);
         if (a->op == nir_op_b2i32)
            alias(c, d, fl);
         else
            emit(c, APEX_S_CSELECT, define(c, d, false), one, constant(c, 0), fl, 0);
         return;
      }
      emit(c, APEX_V_CNDMASK, define(c, d, true), constant(c, 0), one,
           mask_of(c, b, a->src[0].swizzle[0]), 0);
      return;
   }
   case nir_op_ieq: compare(c, a, APEX_V_CMP_I, 0, false, APEX_S_CMP_EQ); return;
   case nir_op_ine: compare(c, a, APEX_V_CMP_I, 1, false, APEX_S_CMP_NE); return;
   case nir_op_ilt: compare(c, a, APEX_V_CMP_I, 2, false, APEX_S_CMP_LT_I); return;
   case nir_op_ige: compare(c, a, APEX_V_CMP_I, 5, true, APEX_S_CMP_LE_I); return;
   case nir_op_ult: compare(c, a, APEX_V_CMP_U, 2, false, APEX_S_CMP_LT_U); return;
   case nir_op_uge: compare(c, a, APEX_V_CMP_U, 5, true, APEX_S_CMP_LE_U); return;
   case nir_op_feq: compare(c, a, APEX_V_CMP_F, 0, false, 0); return;
   case nir_op_fneu: compare(c, a, APEX_V_CMP_F, 1, false, 0); return;
   case nir_op_flt: compare(c, a, APEX_V_CMP_F, 2, false, 0); return;
   case nir_op_fge: compare(c, a, APEX_V_CMP_F, 5, false, 0); return;
   case nir_op_fadd: fp_op(c, a, APEX_V_ADD_F); return;
   case nir_op_fsub: fp_op(c, a, APEX_V_SUB_F); return;
   case nir_op_fmul: fp_op(c, a, APEX_V_MUL_F); return;
   case nir_op_ffma: case nir_op_ffma_weak: fp_op(c, a, APEX_V_FMA_F); return;
   case nir_op_fmin: fp_op(c, a, APEX_V_MIN_F); return;
   case nir_op_fmax: fp_op(c, a, APEX_V_MAX_F); return;
   case nir_op_ffloor: fp_op(c, a, APEX_V_FLOOR); return;
   case nir_op_fceil: fp_op(c, a, APEX_V_CEIL); return;
   case nir_op_ftrunc: fp_op(c, a, APEX_V_TRUNC); return;
   case nir_op_fround_even: fp_op(c, a, APEX_V_RNDNE); return;
   case nir_op_ffract: fp_op(c, a, APEX_V_FRACT); return;
   case nir_op_frcp: fp_op(c, a, APEX_V_RCP); return;
   case nir_op_frsq: fp_op(c, a, APEX_V_RSQ); return;
   case nir_op_fsqrt: fp_op(c, a, APEX_V_SQRT); return;
   case nir_op_fexp2: fp_op(c, a, APEX_V_EXP2); return;
   case nir_op_flog2: fp_op(c, a, APEX_V_LOG2); return;
   case nir_op_frexp_sig: fp_op(c, a, APEX_V_FREXP_MANT); return;
   case nir_op_frexp_exp: fp_op(c, a, APEX_V_FREXP_EXP); return;
   case nir_op_ldexp: fp_op(c, a, APEX_V_LDEXP); return;
   case nir_op_f2u32: fp_op(c, a, APEX_V_CVT_U_F); return;
   case nir_op_f2i32: fp_op(c, a, APEX_V_CVT_I_F); return;
   case nir_op_u2f32: int_op(c, a, APEX_V_CVT_F_U, 0); return;
   case nir_op_i2f32: int_op(c, a, APEX_V_CVT_F_I, 0); return;
   case nir_op_pack_half_2x16_split: fp_op(c, a, APEX_V_CVT_PK_F16); return;
   case nir_op_fsin: case nir_op_fcos: {
      uint32_t hi = 0;
      uint32_t x = fp_src(c, a, 0, 1, &hi);
      uint32_t turns = value(c, true, 1);
      emit(c, APEX_V_MUL_F, turns, x, constant(c, 0x3e22f983u /* 1/(2 pi) */), 0, hi);
      emit(c, a->op == nir_op_fsin ? APEX_V_SIN : APEX_V_COS, define(c, d, true), turns, 0, 0, 0);
      return;
   }
   default:
      failf(c, "unsupported normalized NIR ALU: ", nir_op_infos[a->op].name);
   }
}

/* Memory. */
/* User data s0-s1: the root GPUVA, read into one pair for the program. */
static uint32_t root(struct ctx *c)
{
   return c->root;
}
static uint32_t mem_hi(int32_t offset, unsigned size_code)
{
   return ((uint32_t)offset & 0xfffff) | size_code << 20;
}
static bool fits20(int64_t v) { return v >= -(1 << 19) && v < (1 << 19); }
/* Splits a byte offset into a register part and a 20-bit immediate. */
/* Splits a constant term off an offset into the immediate. Only buffer
 * offsets add in 32 bits; elsewhere the register and the sign-extended
 * immediate add without wrapping, so a positive term folds only when the
 * rest cannot exceed 2^32 - 1 minus it (a negative one leaves a valid,
 * non-negative offset unchanged). */
static uint32_t split_offset(struct ctx *c, nir_src s, int64_t *imm, bool wraps)
{
   *imm = 0;
   if (nir_src_is_const(s) && fits20(nir_src_as_int(s)))
      return (*imm = nir_src_as_int(s)), 0;
   nir_alu_instr *add = nir_def_as_alu_or_null(s.ssa);
   if (add && add->op == nir_op_iadd) {
      for (unsigned k = 0; k < 2; k++) {
         if (!nir_src_is_const(add->src[k].src))
            continue;
         int64_t v = nir_src_comp_as_int(add->src[k].src, add->src[k].swizzle[0]);
         nir_scalar rest = nir_get_scalar(add->src[1 - k].src.ssa, add->src[1 - k].swizzle[0]);
         if (fits20(v) && (wraps || v <= 0 || !nir_addition_might_overflow(c->nir, c->range, rest, v))) {
            *imm = v;
            return alu_src(c, add, 1 - k);
         }
      }
   }
   return src(c, s, 0);
}

/* Access alignment: the address is align_offset modulo align_mul bytes. */
struct alignment { unsigned mul, offset; };
static struct alignment access_alignment(nir_intrinsic_instr *i)
{
   if (!nir_intrinsic_has_align_mul(i))
      return (struct alignment){4, 0};
   return (struct alignment){nir_intrinsic_align_mul(i), nir_intrinsic_align_offset(i)};
}
/* Dwords of the piece at byte `at` of an access with `left` dwords to go.
 * Every lane address rounds down to 4n bytes (16 for three dwords, 32 for
 * eight), so a piece is as wide as its alignment allows: scalar loads take
 * 1, 2, 4 or 8 dwords, vector accesses 1-4. */
static unsigned piece_dwords(struct alignment a, unsigned at, unsigned left, bool scalar_load)
{
   unsigned align = nir_combined_align(a.mul, (a.offset + at) % a.mul);
   for (unsigned p = MIN2(left, scalar_load ? 8 : 4); p > 1; p--) {
      if (scalar_load && (p & (p - 1)))
         continue;
      if (align >= 4 * util_next_power_of_two(p))
         return p;
   }
   return 1;
}

/* A scalar load of `n` dwords into a new scalar group, in aligned pieces. */
static uint32_t s_load(struct ctx *c, unsigned op, uint32_t base, uint32_t offset, int64_t imm,
                       unsigned n, struct alignment a)
{
   uint32_t g = value(c, false, n);
   for (unsigned k = 0; k < n;) {
      unsigned piece = piece_dwords(a, 4 * k, n - k, true);
      unsigned code = piece == 8 ? 3 : piece == 4 ? 2 : piece == 2 ? 1 : 0;
      emit(c, op, sub(g, k, piece), offset ? scalar(c, offset) : 0, base,
           0, mem_hi(imm + 4 * k, code));
      last(c)->flags = APEX_REORDER;
      k += piece;
   }
   return g;
}

/* A buffer descriptor {GPUVA, bytes, flags} source. */
static uint32_t buffer_descriptor(struct ctx *c, nir_src s)
{
   return sub(src(c, s, 0), 0, 4);
}
static bool raw_root(nir_src s)
{
   return s.ssa->num_components == 1 && nir_src_is_const(s) && nir_src_as_uint(s) == 0;
}

/* Runs `body` once per distinct descriptor among active lanes. */
struct waterfall { uint32_t saved, remaining, start; };
static uint32_t waterfall_begin(struct ctx *c, uint32_t desc, struct waterfall *w)
{
   w->saved = exec_copy(c);
   w->start = label(c);
   place(c, w->start);
   uint32_t s = value(c, false, 4);
   for (unsigned k = 0; k < 4; k++)
      emit(c, APEX_COPY, sub(s, k, 1), sub(desc, k, 1), 0, 0, 0);
   uint32_t m = 0;
   for (unsigned k = 0; k < 4; k++) {
      uint32_t e = value(c, false, 1);
      emit(c, APEX_V_CMP_U, e, sub(desc, k, 1), sub(s, k, 1), apex_raw(0), 0);
      m = m ? op2(c, APEX_S_AND, false, m, e) : e;
   }
   w->remaining = value(c, false, 1);
   emit(c, APEX_S_AND_SAVEEXEC, w->remaining, m, 0, 0, 0);
   emit(c, APEX_S_ANDN2, w->remaining, w->remaining, m, 0, 0);
   return s;
}
static void waterfall_end(struct ctx *c, struct waterfall *w)
{
   set_exec(c, w->remaining);
   branch(c, APEX_S_CBRANCH_EXECNZ, 0, w->start);
   set_exec(c, w->saved);
}

static uint32_t vector_data(struct ctx *c, nir_src s, unsigned n)
{
   uint32_t first = src(c, s, 0);
   uint32_t g = sub(first, 0, n);
   if (is_vec(c, first) && s.ssa->num_components == n)
      return g;
   return copy(c, true, g);
}

/* An even-aligned pair: a whole two-dword value, else a copy. */
static uint32_t pair_value(struct ctx *c, uint32_t x)
{
   uint32_t id = apex_val_id(x);
   uint32_t p = sub(x, 0, 2);
   if (!apex_val_off(x) && util_dynarray_element(&c->values, struct apex_value, id)->width == 2)
      return p;
   return copy(c, is_vec(c, x), p);
}

static int atomic_code(nir_atomic_op op)
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

/* Fragment helper lanes and demoted lanes never write memory. */
static uint32_t guard_begin(struct ctx *c)
{
   if (c->nir->info.stage != MESA_SHADER_FRAGMENT)
      return 0;
   uint32_t writable = c->covered ? op2(c, APEX_S_ANDN2, false, c->covered, c->helper) :
                       op2(c, APEX_S_NOT, false, c->helper, 0);
   uint32_t saved = value(c, false, 1);
   emit(c, APEX_S_AND_SAVEEXEC, saved, writable, 0, 0, 0);
   return saved;
}
static void guard_end(struct ctx *c, uint32_t saved)
{
   if (saved)
      set_exec(c, saved);
}

/* One memory intrinsic: kind selects the address space. */
enum space { ROOT, BUFFER, GLOBAL, SHARED, SCRATCH };

static void memory(struct ctx *c, nir_intrinsic_instr *i, enum space space, bool store,
                   bool atomic)
{
   unsigned n = store ? i->src[0].ssa->num_components : atomic ? 1 : i->def.num_components;
   unsigned bits = store ? i->src[0].ssa->bit_size : i->def.bit_size;
   /* Loads up to 16 dwords (scalar loads, or four-dword vector pieces). */
   if (bits != 32 || n > (store || atomic ? 4 : 16) || (atomic && n != 1)) {
      failf(c, "unsupported memory access width: ", nir_intrinsic_infos[i->intrinsic].name);
      return;
   }
   if (store && nir_intrinsic_has_write_mask(i) && nir_intrinsic_write_mask(i) != BITFIELD_MASK(n)) {
      failf(c, "partial store write mask: ", nir_intrinsic_infos[i->intrinsic].name);
      return;
   }
   /* Source positions: [data], [descriptor], address. */
   unsigned data = 0, address;
   nir_src *desc_src = NULL;
   switch (i->intrinsic) {
   case nir_intrinsic_load_ssbo: case nir_intrinsic_load_ubo:
      desc_src = &i->src[0]; address = 1; break;
   case nir_intrinsic_store_ssbo:
      desc_src = &i->src[1]; address = 2; break;
   case nir_intrinsic_ssbo_atomic: case nir_intrinsic_ssbo_atomic_swap:
      desc_src = &i->src[0]; address = 1; data = 2; break;
   case nir_intrinsic_store_global_2x32: case nir_intrinsic_store_shared:
   case nir_intrinsic_store_scratch:
      address = 1; break;
   case nir_intrinsic_global_atomic_2x32: case nir_intrinsic_global_atomic_swap_2x32:
   case nir_intrinsic_shared_atomic: case nir_intrinsic_shared_atomic_swap:
      address = 0; data = 1; break;
   default:
      address = 0; break;
   }
   int64_t imm = 0;
   uint32_t base = 0, offset = 0, a = 0, b = 0;
   bool uniform = !i->src[address].ssa->divergent;
   bool reorder = !store && !atomic &&
                  ((nir_intrinsic_has_access(i) && (nir_intrinsic_access(i) & ACCESS_CAN_REORDER)) ||
                   i->intrinsic == nir_intrinsic_load_ubo);
   bool returns = atomic && !nir_def_is_unused(&i->def);
   unsigned op;
   struct waterfall wf;
   bool fall = false;
   uint32_t result = 0;
   if (space == ROOT || space == BUFFER) {
      offset = split_offset(c, i->src[address], &imm, space == BUFFER);
      if (space == ROOT)
         base = root(c);
      else if (desc_src->ssa->num_components != 4) {
         failf(c, "unsupported buffer index: ", nir_intrinsic_infos[i->intrinsic].name);
         return;
      } else {
         base = buffer_descriptor(c, *desc_src);
         if (is_vec(c, base)) {
            base = waterfall_begin(c, base, &wf);
            fall = true;
         }
      }
      if (reorder && uniform && !fall) {
         result = s_load(c, space == ROOT ? APEX_S_LOAD : APEX_S_BUFFER_LOAD, base, offset, imm, n,
                         access_alignment(i));
         alias(c, &i->def, result);
         return;
      }
      if (space == ROOT) {
         op = store ? APEX_GLOBAL_STORE : atomic ? APEX_GLOBAL_ATOMIC : APEX_GLOBAL_LOAD;
      } else {
         op = store ? APEX_BUFFER_STORE : atomic ? APEX_BUFFER_ATOMIC : APEX_BUFFER_LOAD;
      }
      a = offset ? vector(c, offset) : 0;
      b = base;
   } else if (space == GLOBAL) {
      if (!atomic && nir_combined_align(nir_intrinsic_align_mul(i), nir_intrinsic_align_offset(i)) < 4) {
         failf(c, "unsupported memory alignment: ", nir_intrinsic_infos[i->intrinsic].name);
         return;
      }
      if (i->src[address].ssa->num_components != 2) {
         failf(c, "unsupported global address", NULL);
         return;
      }
      uint32_t pair = pair_value(c, src(c, i->src[address], 0));
      if (reorder && uniform) {
         alias(c, &i->def, s_load(c, APEX_S_LOAD, pair_value(c, scalar(c, pair)), 0, 0, n,
                                  access_alignment(i)));
         return;
      }
      op = store ? APEX_GLOBAL_STORE : atomic ? APEX_GLOBAL_ATOMIC : APEX_GLOBAL_LOAD;
      if (is_vec(c, pair))
         a = pair;
      else
         b = pair;
   } else if (space == SHARED) {
      offset = split_offset(c, i->src[address], &imm, false);
      imm += nir_intrinsic_has_base(i) ? nir_intrinsic_base(i) : 0;
      if (!fits20(imm)) {
         failf(c, "shared offset out of range", NULL);
         return;
      }
      op = store ? APEX_SHARED_STORE : atomic ? APEX_SHARED_ATOMIC : APEX_SHARED_LOAD;
      a = offset ? vector(c, offset) : 0;
   } else {
      if (c->nir->info.stage != MESA_SHADER_COMPUTE) {
         failf(c, "private memory needs a compute launch", NULL);
         return;
      }
      offset = split_offset(c, i->src[address], &imm, false);
      imm += nir_intrinsic_has_base(i) ? nir_intrinsic_base(i) : 0;
      op = store ? APEX_SCRATCH_STORE : APEX_SCRATCH_LOAD;
      a = offset ? vector(c, offset) : 0;
   }
   uint32_t guard = store || atomic ? guard_begin(c) : 0;
   uint32_t hi = mem_hi(imm, n - 1);
   if (atomic) {
      int code = atomic_code(nir_intrinsic_atomic_op(i));
      if (code < 0) {
         failf(c, "unsupported atomic operation", NULL);
         return;
      }
      hi |= (uint32_t)code << 24 | (uint32_t)returns << 28;
      uint32_t operand = value(c, true, code == 2 ? 2 : 1);
      if (code == 2) {
         /* d = new value, d + 1 = comparison value. */
         emit(c, APEX_COPY, sub(operand, 0, 1), src(c, i->src[data + 1], 0), 0, 0, 0);
         emit(c, APEX_COPY, sub(operand, 1, 1), src(c, i->src[data], 0), 0, 0, 0);
      } else {
         emit(c, APEX_COPY, operand, src(c, i->src[data], 0), 0, 0, 0);
      }
      emit(c, op, operand, a, b, 0, hi);
      if (returns) {
         uint32_t r = define(c, &i->def, true);
         emit(c, APEX_COPY, r, sub(operand, 0, 1), 0, 0, 0);
      }
   } else if (store) {
      uint32_t v = vector_data(c, i->src[0], n);
      for (unsigned k = 0, piece; k < n; k += piece) {
         piece = space == SCRATCH ? MIN2(n - k, 4) : piece_dwords(access_alignment(i), 4 * k, n - k, false);
         emit(c, op, piece == n ? v : sub(v, k, piece), a, b, 0, mem_hi(imm + 4 * k, piece - 1));
      }
   } else {
      uint32_t d = fall ? value(c, true, n) : define(c, &i->def, true);
      for (unsigned k = 0, piece; k < n; k += piece) {
         piece = space == SCRATCH ? MIN2(n - k, 4) : piece_dwords(access_alignment(i), 4 * k, n - k, false);
         emit(c, op, sub(d, k, piece), a, b, 0, mem_hi(imm + 4 * k, piece - 1));
         if (reorder)
            last(c)->flags = APEX_REORDER;
      }
      if (fall) {
         uint32_t r = define(c, &i->def, true);
         emit(c, APEX_COPY, r, d, 0, 0, 0);
      }
   }
   guard_end(c, guard);
   if (fall)
      waterfall_end(c, &wf);
}

/* Control flow. */
static void restore(struct ctx *c, uint32_t m)
{
   if (c->loop && c->loop->divergent)
      m = op2(c, APEX_S_AND, false, m, c->loop->iter);
   if (c->alive)
      m = op2(c, APEX_S_AND, false, m, c->alive);
   set_exec(c, m);
}

static void emit_cf(struct ctx *c, struct exec_list *list);

static bool only_break(struct exec_list *list)
{
   if (exec_list_length(list) != 1)
      return false;
   nir_block *b = nir_cf_node_as_block(exec_node_data(nir_cf_node, exec_list_get_head(list), node));
   nir_instr *last = nir_block_last_instr(b);
   return last && last == nir_block_first_instr(b) && last->type == nir_instr_type_jump &&
          nir_instr_as_jump(last)->type == nir_jump_break;
}

static void emit_if(struct ctx *c, nir_if *nif)
{
   bool has_else = !nir_cf_list_is_empty_block(&nif->else_list);
   if (!nif->condition.ssa->divergent) {
      uint32_t otherwise = label(c), end = label(c);
      /* if (x == 0) skips the then side when x != 0. */
      nir_alu_instr *cmp = nir_def_as_alu_or_null(nif->condition.ssa);
      unsigned op = APEX_S_CBRANCH_Z;
      uint32_t cond = 0;
      if (cmp && (cmp->op == nir_op_ieq || cmp->op == nir_op_ine) &&
          cmp->src[0].src.ssa->bit_size == 32) {
         for (unsigned k = 0; k < 2 && !cond; k++) {
            if (nir_src_is_const(cmp->src[k].src) &&
                !nir_src_comp_as_uint(cmp->src[k].src, cmp->src[k].swizzle[0]) &&
                !is_vec(c, alu_src(c, cmp, 1 - k))) {
               cond = alu_src(c, cmp, 1 - k);
               op = cmp->op == nir_op_ieq ? APEX_S_CBRANCH_NZ : APEX_S_CBRANCH_Z;
            }
         }
      }
      branch(c, op, cond ? cond : flag_of(c, nif->condition, 0), otherwise);
      emit_cf(c, &nif->then_list);
      if (has_else)
         branch(c, APEX_S_BRANCH, 0, end);
      place(c, otherwise);
      emit_cf(c, &nif->else_list);
      place(c, end);
      return;
   }
   uint32_t m = mask_of(c, nif->condition, 0);
   /* A side that only breaks removes its lanes; the other side runs in place. */
   struct loop_state *l = c->loop;
   bool then_break = only_break(&nif->then_list), else_break = only_break(&nif->else_list);
   if (l && l->divergent && (then_break || else_break) && !(then_break && else_break)) {
      uint32_t leaving = op2(c, then_break ? APEX_S_AND : APEX_S_ANDN2, false,
                             then_break ? m : apex_phys(APEX_EXEC), then_break ? apex_phys(APEX_EXEC) : m);
      emit(c, APEX_S_ANDN2, l->iter, l->iter, leaving, 0, 0);
      emit(c, APEX_S_ANDN2, l->live, l->live, leaving, 0, 0);
      set_exec(c, op2(c, APEX_S_ANDN2, false, apex_phys(APEX_EXEC), leaving));
      branch(c, APEX_S_CBRANCH_EXECZ, 0, c->restore);
      emit_cf(c, then_break ? &nif->else_list : &nif->then_list);
      return;
   }
   uint32_t saved = value(c, false, 1);
   emit(c, APEX_S_AND_SAVEEXEC, saved, m, 0, 0, 0);
   uint32_t else_mask = has_else ? op2(c, APEX_S_ANDN2, false, saved, m) : 0;
   uint32_t otherwise = label(c), end = label(c), outer = c->restore;
   branch(c, APEX_S_CBRANCH_EXECZ, 0, otherwise);
   c->restore = otherwise;
   c->depth++;
   emit_cf(c, &nif->then_list);
   place(c, otherwise);
   if (has_else) {
      restore(c, else_mask);
      branch(c, APEX_S_CBRANCH_EXECZ, 0, end);
      c->restore = end;
      emit_cf(c, &nif->else_list);
      place(c, end);
   }
   c->depth--;
   c->restore = outer;
   restore(c, saved);
}

static void emit_loop(struct ctx *c, nir_loop *loop)
{
   struct loop_state s = {.divergent = nir_loop_is_divergent(loop), .parent = c->loop,
                          .head = label(c), .latch = label(c), .end = label(c)};
   uint32_t saved = 0, outer = c->restore;
   if (s.divergent) {
      saved = exec_copy(c);
      s.live = copy(c, false, saved);
      s.iter = value(c, false, 1);
   }
   place(c, s.head);
   if (s.divergent) {
      set_exec(c, c->alive ? op2(c, APEX_S_AND, false, s.live, c->alive) : s.live);
      emit(c, APEX_COPY, s.iter, s.live, 0, 0, 0);
      c->restore = s.latch;
      c->depth++;
   }
   c->loop = &s;
   emit_cf(c, &loop->body);
   c->loop = s.parent;
   place(c, s.latch);
   if (s.divergent) {
      branch(c, APEX_S_CBRANCH_NZ, s.live, s.head);
      place(c, s.end);
      c->depth--;
      c->restore = outer;
      restore(c, saved);
   } else {
      branch(c, APEX_S_BRANCH, 0, s.head);
      place(c, s.end);
   }
}

static void emit_jump(struct ctx *c, nir_jump_instr *j)
{
   struct loop_state *l = c->loop;
   if (!l || (j->type != nir_jump_break && j->type != nir_jump_continue)) {
      failf(c, "unsupported jump", NULL);
      return;
   }
   if (!l->divergent) {
      branch(c, APEX_S_BRANCH, 0, j->type == nir_jump_break ? l->end : l->head);
      return;
   }
   /* Every active lane leaves; the wave resumes at the next exec restore. */
   emit(c, APEX_S_ANDN2, l->iter, l->iter, apex_phys(APEX_EXEC), 0, 0);
   if (j->type == nir_jump_break)
      emit(c, APEX_S_ANDN2, l->live, l->live, apex_phys(APEX_EXEC), 0, 0);
   branch(c, APEX_S_BRANCH, 0, c->restore);
}

/* Stage inputs and outputs. */
static uint32_t launch_vector(struct ctx *c, unsigned reg)
{
   return copy(c, true, apex_phys(reg));
}
static uint32_t launch_bits(struct ctx *c, unsigned shift, unsigned bits)
{
   return op2(c, APEX_V_BFE_U, true, apex_phys(3), constant(c, shift | bits << 8));
}
/* Fragment input k is vertex output varying k (Docs/isa.md, Vertex outputs
 * and fragment inputs); its header byte carries the flat bit. */
static unsigned fragment_input(struct ctx *c, unsigned location, bool flat)
{
   if (location < VARYING_SLOT_VAR0 || location >= VARYING_SLOT_VAR0 + 32) {
      failf(c, "unsupported fragment input: ", gl_varying_slot_name_for_stage(location, MESA_SHADER_FRAGMENT));
      return 0;
   }
   unsigned k = location - VARYING_SLOT_VAR0;
   c->header.input_count = MAX2(c->header.input_count, k + 1);
   if (flat)
      c->header.inputs[k] = 0x80;
   return k;
}
static unsigned record_slot(struct ctx *c, unsigned location, unsigned *component)
{
   switch (location) {
   case VARYING_SLOT_POS: return 0;
   case VARYING_SLOT_PSIZ: *component = 0; return 1;
   case VARYING_SLOT_LAYER: *component = 1; return 1;
   case VARYING_SLOT_VIEWPORT: *component = 2; return 1;
   case VARYING_SLOT_CLIP_DIST0: return 2;
   case VARYING_SLOT_CLIP_DIST1: return 3;
   default:
      if (location >= VARYING_SLOT_VAR0 && location < VARYING_SLOT_VAR0 + 32)
         return 4 + location - VARYING_SLOT_VAR0;
      failf(c, "unsupported vertex output: ", gl_varying_slot_name_for_stage(location, MESA_SHADER_VERTEX));
      return 0;
   }
}
static void store_output(struct ctx *c, nir_intrinsic_instr *i)
{
   nir_io_semantics io = nir_intrinsic_io_semantics(i);
   unsigned component = nir_intrinsic_component(i), slot;
   if (c->nir->info.stage == MESA_SHADER_VERTEX) {
      slot = record_slot(c, io.location, &component);
   } else if (io.location == FRAG_RESULT_DEPTH) {
      slot = 8, component = 0;
   } else if (io.location == FRAG_RESULT_SAMPLE_MASK) {
      slot = 9, component = 0;
   } else if (io.location >= FRAG_RESULT_DATA0) {
      slot = io.location - FRAG_RESULT_DATA0 + io.dual_source_blend_index;
      if (io.dual_source_blend_index)
         c->header.output |= 1u << 8;
      c->header.output |= 1u << slot;
   } else {
      failf(c, "unsupported fragment output", NULL);
      return;
   }
   if (slot >= ARRAY_SIZE(c->out))
      return;
   if (!c->out[slot])
      c->out[slot] = value(c, true, 4);
   unsigned mask = nir_intrinsic_write_mask(i);
   for (unsigned k = 0; k < i->src[0].ssa->num_components; k++) {
      if (!(mask & (1u << k)) || component + k >= 4)
         continue;
      emit(c, APEX_COPY, sub(c->out[slot], component + k, 1), src(c, i->src[0], k), 0, 0, 0);
      c->out_written[slot] |= 1u << (component + k);
   }
}
static void finish_outputs(struct ctx *c)
{
   if (c->nir->info.stage == MESA_SHADER_VERTEX) {
      /* One 64-byte record line per four vec4s. */
      unsigned records = 0;
      for (unsigned s = 0; s < ARRAY_SIZE(c->out); s++)
         if (c->out[s])
            records = s + 1;
      unsigned stride = 64 * DIV_ROUND_UP(MAX2(records, 1), 4);
      c->header.output = stride;
      uint32_t base = value(c, false, 2);
      emit(c, APEX_COPY, sub(base, 0, 1), apex_phys(APEX_SCALAR + 16), 0, 0, 0);
      emit(c, APEX_COPY, sub(base, 1, 1), apex_phys(APEX_SCALAR + 17), 0, 0, 0);
      uint32_t lane = op2(c, APEX_V_MUL_LO, true, apex_phys(APEX_LANE), constant(c, stride));
      for (unsigned s = 0; s < records; s++)
         if (c->out[s])
            emit(c, APEX_GLOBAL_STORE, c->out[s], lane, base, 0, mem_hi(s * 16, 3));
      return;
   }
   if (c->nir->info.stage != MESA_SHADER_FRAGMENT)
      return;
   /* Lanes absent from exec at the done export are discarded. */
   uint32_t m = c->alive ? c->alive : 0;
   if (c->covered)
      m = m ? op2(c, APEX_S_AND, false, m, c->covered) : c->covered;
   if (m)
      set_exec(c, m);
   int final = -1;
   for (int s = 9; s >= 0 && final < 0; s--)
      if (c->out[s])
         final = s;
   for (int s = 0; s < 10; s++) {
      if (!c->out[s] && !(s == 0 && final < 0))
         continue;
      uint32_t data = c->out[s] ? c->out[s] : value(c, true, 4);
      emit(c, APEX_EXP, apex_raw(s), data, apex_raw(s == final || final < 0 ? 1 : 0), 0, 0);
   }
}

/* Textures (standalone descriptors; Mesa's Vulkan path lowers its own). */
static unsigned tex_dim(nir_tex_instr *t)
{
   switch (t->sampler_dim) {
   case GLSL_SAMPLER_DIM_1D: return t->is_array ? 4 : 0;
   case GLSL_SAMPLER_DIM_2D: case GLSL_SAMPLER_DIM_RECT: case GLSL_SAMPLER_DIM_EXTERNAL:
   case GLSL_SAMPLER_DIM_MS:
      return t->is_array ? 5 : 1;
   case GLSL_SAMPLER_DIM_3D: return 2;
   case GLSL_SAMPLER_DIM_CUBE: return t->is_array ? 6 : 3;
   default: return 7;
   }
}
/* Image (backend1) and sampler (backend2) descriptors: eight scalar dwords. */
static uint32_t texture_descriptor(struct ctx *c, nir_tex_instr *t, bool sampler)
{
   int k = nir_tex_instr_src_index(t, sampler ? nir_tex_src_backend2 : nir_tex_src_backend1);
   if (k < 0 || t->src[k].src.ssa->num_components != 8) {
      failf(c, "texture without descriptors", NULL);
      return 0;
   }
   if (t->src[k].src.ssa->divergent) {
      failf(c, "non-uniform texture index", NULL);
      return 0;
   }
   return scalar(c, sub(src(c, t->src[k].src, 0), 0, 8));
}
/* Bits lo .. lo + width - 1 of an image descriptor (architecture, Texture
 * unit: width - 1 at 67, height - 1 at 81, depth or layers - 1 at 95,
 * levels at 106). */
static uint32_t descriptor_bits(struct ctx *c, uint32_t image, unsigned lo, unsigned width)
{
   unsigned dword = lo / 32, shift = lo % 32;
   uint32_t x = op2(c, APEX_S_BFE_U, false, sub(image, dword, 1), constant(c, shift | MIN2(width, 32 - shift) << 8));
   if (shift + width > 32) {
      uint32_t high = op2(c, APEX_S_BFE_U, false, sub(image, dword + 1, 1), constant(c, (shift + width - 32) << 8));
      x = op2(c, APEX_S_OR, false, x, op2(c, APEX_S_SHL, false, high, constant(c, 32 - shift)));
   }
   return x;
}
static void texture_size(struct ctx *c, nir_tex_instr *t, uint32_t image)
{
   uint32_t g = define(c, &t->def, false);
   if (t->op == nir_texop_query_levels) {
      emit(c, APEX_COPY, g, descriptor_bits(c, image, 106, 5), 0, 0, 0);
      return;
   }
   int lod = nir_tex_instr_src_index(t, nir_tex_src_lod);
   for (unsigned k = 0; k < t->def.num_components; k++) {
      bool layer = t->is_array && k == t->def.num_components - 1;
      static const unsigned lo[3] = {67, 81, 95};
      unsigned field = layer ? 2 : k;
      uint32_t size = op2(c, APEX_S_ADD, false, descriptor_bits(c, image, lo[field], field == 2 ? 11 : 14),
                          constant(c, 1));
      if (layer && t->sampler_dim == GLSL_SAMPLER_DIM_CUBE) {
         /* Cube arrays count cubes: layers / 6. */
         size = op2(c, APEX_S_SHR, false, op2(c, APEX_S_MUL_HI_U, false, size, constant(c, 0xaaaaaaabu)),
                    constant(c, 2));
      } else if (lod >= 0 && !layer) {
         size = op2(c, APEX_S_MAX_U, false, op2(c, APEX_S_SHR, false, size, scalar(c, src(c, t->src[lod].src, 0))),
                    constant(c, 1));
      }
      emit(c, APEX_COPY, sub(g, k, 1), size, 0, 0, 0);
   }
}
/* Coordinate registers in the texture unit's layout (Docs/isa.md, Texture
 * and export): u, v, w or layer, then the level, bias or reference; the
 * second four hold gradients, or the level or bias beside a reference. */
static void emit_tex(struct ctx *c, nir_tex_instr *t)
{
   unsigned dim = tex_dim(t);
   if (dim == 7) {
      failf(c, "unsupported texture dimensionality", NULL);
      return;
   }
   uint32_t image = texture_descriptor(c, t, false);
   if (t->op == nir_texop_txs || t->op == nir_texop_query_levels) {
      texture_size(c, t, image);
      return;
   }
   unsigned variant;
   switch (t->op) {
   case nir_texop_tex:
      variant = c->nir->info.stage == MESA_SHADER_FRAGMENT ? 0 : 1; break;
   case nir_texop_txb: variant = 2; break;
   case nir_texop_txl: variant = 1; break;
   case nir_texop_txd: variant = 3; break;
   case nir_texop_txf: case nir_texop_txf_ms: variant = 0; break;
   case nir_texop_tg4: variant = 5; break;
   default:
      failf(c, "unsupported texture op", NULL);
      return;
   }
   bool fetch = t->op == nir_texop_txf || t->op == nir_texop_txf_ms;
   bool cube = dim == 3 || dim == 6;
   int coord = nir_tex_instr_src_index(t, nir_tex_src_coord);
   int lod = nir_tex_instr_src_index(t, nir_tex_src_lod);
   int bias = nir_tex_instr_src_index(t, nir_tex_src_bias);
   int cmp = nir_tex_instr_src_index(t, nir_tex_src_comparator);
   int off = nir_tex_instr_src_index(t, nir_tex_src_offset);
   int ddx = nir_tex_instr_src_index(t, nir_tex_src_ddx);
   int ddy = nir_tex_instr_src_index(t, nir_tex_src_ddy);
   int ms = nir_tex_instr_src_index(t, nir_tex_src_ms_index);
   if (variant == 3 && (dim == 2 || cube)) {
      failf(c, "explicit gradients take two axes", NULL);
      return;
   }
   unsigned mask = t->is_shadow ? 1 : t->op == nir_texop_tg4 ? 0xf : nir_def_components_read(&t->def);
   if (!mask)
      mask = 1;
   bool compare = cmp >= 0 && !fetch;
   /* Texture hi (isa.rs texture_hi): the sampler base joins at encoding. */
   uint32_t hi = mask << 5 | (t->op == nir_texop_tg4 ? t->component << 9 : 0) | variant << 11 |
                 (unsigned)compare << 14 | dim << 15;
   if (off >= 0 && !fetch) {
      for (unsigned k = 0; k < t->src[off].src.ssa->num_components; k++) {
         if (!nir_src_is_const(t->src[off].src)) {
            failf(c, "dynamic texel offsets", NULL);
            return;
         }
         int32_t o = nir_src_comp_as_int(t->src[off].src, k);
         if (o < -8 || o > 7) {
            failf(c, "texel offset out of range", NULL);
            return;
         }
         hi |= ((uint32_t)o & 15) << (18 + 4 * k);
      }
   }
   uint32_t parts[8] = {0};
   if (cube) {
      /* The compiler selects the face: face coordinates in [0, 1] and the
       * integer face | cube index << 3. */
      uint32_t x = src(c, t->src[coord].src, 0), y = src(c, t->src[coord].src, 1);
      uint32_t z = src(c, t->src[coord].src, 2);
      uint32_t ma = value(c, true, 1), sc = value(c, true, 1), tc = value(c, true, 1), face = value(c, true, 1);
      emit(c, APEX_V_CUBEMA, ma, x, y, z, 0);
      emit(c, APEX_V_CUBESC, sc, x, y, z, 0);
      emit(c, APEX_V_CUBETC, tc, x, y, z, 0);
      emit(c, APEX_V_CUBEID, face, x, y, z, 0);
      uint32_t r = value(c, true, 1), half = constant(c, 0x3f000000u);
      emit(c, APEX_V_RCP, r, ma, 0, 0, 1u << 11);
      for (unsigned k = 0; k < 2; k++) {
         parts[k] = value(c, true, 1);
         emit(c, APEX_V_FMA_F, parts[k], k ? tc : sc, r, half, 0);
      }
      uint32_t index = op1(c, APEX_V_CVT_U_F, true, face);
      if (dim == 6) {
         uint32_t layer = op1(c, APEX_V_CVT_U_F, true, op1(c, APEX_V_RNDNE, true, src(c, t->src[coord].src, 3)));
         index = op2(c, APEX_V_OR, true, index, op2(c, APEX_V_SHL, true, layer, constant(c, 3)));
      }
      parts[2] = index;
   } else {
      for (unsigned k = 0; k < t->coord_components; k++)
         parts[k] = src(c, t->src[coord].src, k);
      if (fetch && off >= 0)
         for (unsigned k = 0; k < t->src[off].src.ssa->num_components; k++)
            parts[k] = op2(c, APEX_V_ADD, true, parts[k], src(c, t->src[off].src, k));
   }
   uint32_t level = 0;
   if (fetch)
      level = ms >= 0 ? src(c, t->src[ms].src, 0) : lod >= 0 ? src(c, t->src[lod].src, 0) : constant(c, 0);
   else if (variant == 1)
      level = lod >= 0 ? src(c, t->src[lod].src, 0) : constant(c, 0);
   else if (variant == 2)
      level = src(c, t->src[bias].src, 0);
   if (compare) {
      parts[3] = src(c, t->src[cmp].src, 0);
      if (level)
         parts[4] = level;
   } else if (level) {
      parts[3] = level;
   }
   if (variant == 3) {
      /* d/dx and d/dy of u and v; the unit reads each quad's lowest lane. */
      for (unsigned k = 0; k < 2 && k < t->src[ddx].src.ssa->num_components; k++) {
         parts[4 + k] = src(c, t->src[ddx].src, k);
         parts[6 + k] = src(c, t->src[ddy].src, k);
      }
   }
   unsigned count = variant == 3 || (compare && level) ? 8 : 4;
   uint32_t coord_group = value(c, true, count);
   for (unsigned k = 0; k < count; k++)
      if (parts[k])
         emit(c, APEX_COPY, sub(coord_group, k, 1), parts[k], 0, 0, 0);
   uint32_t sampler = fetch ? image : texture_descriptor(c, t, true);
   unsigned results = util_bitcount(mask);
   uint32_t out = value(c, true, results);
   emit(c, fetch ? APEX_IMAGE_FETCH : APEX_IMAGE_SAMPLE, out, coord_group, image, sampler, hi);
   uint32_t g = define(c, &t->def, true);
   unsigned r = 0;
   for (unsigned k = 0; k < t->def.num_components; k++)
      if (mask & (1u << k))
         emit(c, APEX_COPY, sub(g, k, 1), sub(out, r++, 1), 0, 0, 0);
}

static void barrier(struct ctx *c, nir_intrinsic_instr *i)
{
   mesa_scope exec = nir_intrinsic_execution_scope(i), mem = nir_intrinsic_memory_scope(i);
   nir_variable_mode modes = nir_intrinsic_memory_modes(i);
   nir_memory_semantics sem = nir_intrinsic_memory_semantics(i);
   if ((modes & ~(nir_var_mem_shared | nir_var_mem_ssbo | nir_var_mem_global |
                  nir_var_mem_ubo | nir_var_image)) ||
       (sem & ~(NIR_MEMORY_ACQ_REL)) || exec > SCOPE_WORKGROUP) {
      failf(c, "unsupported barrier", NULL);
      return;
   }
   if (sem && mem > SCOPE_INVOCATION) {
      /* Shared memory stays workgroup-local at any scope. */
      unsigned scope = !(modes & ~nir_var_mem_shared) || mem <= SCOPE_WORKGROUP ? 1 : 2;
      unsigned hi = scope | (sem & NIR_MEMORY_ACQUIRE ? 4 : 0) | (sem & NIR_MEMORY_RELEASE ? 8 : 0);
      emit(c, APEX_S_FENCE, 0, 0, 0, 0, hi);
   }
   if (exec == SCOPE_WORKGROUP && c->nir->info.stage == MESA_SHADER_COMPUTE)
      emit(c, APEX_S_BARRIER, 0, 0, 0, 0, 0);
}

static void emit_intrinsic(struct ctx *c, nir_intrinsic_instr *i)
{
   nir_def *d = nir_intrinsic_infos[i->intrinsic].has_dest ? &i->def : NULL;
   mesa_shader_stage stage = c->nir->info.stage;
   switch (i->intrinsic) {
   case nir_intrinsic_decl_reg: {
      unsigned n = nir_intrinsic_num_components(i) * (nir_intrinsic_bit_size(i) == 64 ? 2 : 1);
      if (nir_intrinsic_num_array_elems(i)) {
         failf(c, "register arrays are unsupported", NULL);
         return;
      }
      c->defs[d->index] = (struct def_map){new_value(c, c->vector[d->index] && nir_intrinsic_bit_size(i) != 1, n), 0};
      return;
   }
   case nir_intrinsic_load_reg: {
      nir_def *reg = i->src[0].ssa;
      uint32_t id = c->defs[reg->index].id;
      nir_intrinsic_instr *store = register_store(d);
      if (store && store->src[1].ssa != reg) {
         uint32_t r = apex_val(id, 0, util_dynarray_element(&c->values, struct apex_value, id)->width);
         emit(c, APEX_COPY, define(c, d, c->vector[reg->index]), r, 0, 0, 0);
      } else {
         c->defs[d->index] = (struct def_map){id, 0};
      }
      return;
   }
   case nir_intrinsic_store_reg: {
      nir_def *reg = i->src[1].ssa;
      uint32_t id = c->defs[reg->index].id;
      unsigned w = words(i->src[0].ssa);
      unsigned mask = nir_intrinsic_write_mask(i);
      bool divergent_bool = i->src[0].ssa->bit_size == 1 &&
                            util_dynarray_element(&c->values, struct apex_value, id)->cls == 0 &&
                            nir_intrinsic_divergent(nir_def_as_intrinsic(reg));
      for (unsigned k = 0; k < i->src[0].ssa->num_components; k++) {
         if (!(mask & (1u << k)))
            continue;
         uint32_t r = apex_val(id, k * w, w);
         if (i->src[0].ssa->bit_size == 1) {
            if (divergent_bool) {
               /* Scalar writes are unmasked: merge the active lanes. */
               uint32_t m = mask_of(c, i->src[0], k);
               if (c->depth) {
                  uint32_t keep = op2(c, APEX_S_ANDN2, false, r, apex_phys(APEX_EXEC));
                  uint32_t take = op2(c, APEX_S_AND, false, m, apex_phys(APEX_EXEC));
                  emit(c, APEX_S_OR, r, keep, take, 0, 0);
               } else {
                  emit(c, APEX_COPY, r, m, 0, 0, 0);
               }
            } else {
               emit(c, APEX_COPY, r, flag_of(c, i->src[0], k), 0, 0, 0);
            }
         } else if (src(c, i->src[0], k) != r) {
            emit(c, APEX_COPY, r, src(c, i->src[0], k), 0, 0, 0);
         }
      }
      return;
   }
   case nir_intrinsic_load_local_invocation_index:
      alias(c, d, launch_vector(c, 0));
      return;
   case nir_intrinsic_ddx: case nir_intrinsic_ddx_coarse: case nir_intrinsic_ddx_fine:
   case nir_intrinsic_ddy: case nir_intrinsic_ddy_coarse: case nir_intrinsic_ddy_fine: {
      nir_intrinsic_op o = i->intrinsic;
      bool y = o == nir_intrinsic_ddy || o == nir_intrinsic_ddy_coarse || o == nir_intrinsic_ddy_fine;
      bool fine = o == nir_intrinsic_ddx_fine || o == nir_intrinsic_ddy_fine;
      if (d->num_components != 1 || d->bit_size != 32) {
         failf(c, "unsupported derivative width", NULL);
         return;
      }
      uint32_t x = src(c, i->src[0], 0);
      /* Quad lanes: 0 (0,0), 1 (1,0), 2 (0,1), 3 (1,1). */
      unsigned hi_pattern = fine ? (y ? 0xee : 0xf5) : (y ? 0xaa : 0x55);
      unsigned lo_pattern = fine ? (y ? 0x44 : 0xa0) : 0x00;
      emit(c, APEX_V_SUB_F, define(c, d, true), quadperm(c, x, hi_pattern),
           quadperm(c, x, lo_pattern), 0, 0);
      return;
   }
   case nir_intrinsic_load_subgroup_invocation:
      alias(c, d, copy(c, true, apex_phys(APEX_LANE)));
      return;
   case nir_intrinsic_load_workgroup_id: {
      uint32_t g = define(c, d, false);
      for (unsigned k = 0; k < 3; k++)
         emit(c, APEX_COPY, sub(g, k, 1), apex_phys(APEX_SCALAR + 16 + k), 0, 0, 0);
      return;
   }
   case nir_intrinsic_load_base_workgroup_id: {
      uint32_t g = define(c, d, false);
      for (unsigned k = 0; k < 3; k++)
         emit(c, APEX_COPY, sub(g, k, 1), constant(c, 0), 0, 0, 0);
      return;
   }
   case nir_intrinsic_load_subgroup_id:
      alias(c, d, copy(c, false, apex_phys(APEX_SCALAR + 19)));
      return;
   case nir_intrinsic_load_num_subgroups:
      alias(c, d, constant(c, DIV_ROUND_UP(c->header.local[0] * c->header.local[1] * c->header.local[2], 16)));
      return;
   case nir_intrinsic_load_subgroup_size:
      alias(c, d, constant(c, 16));
      return;
   case nir_intrinsic_load_kernel_input: {
      if (!nir_src_is_const(i->src[0])) {
         failf(c, "dynamic user data index", NULL);
         return;
      }
      unsigned base = (nir_intrinsic_base(i) + nir_src_as_uint(i->src[0])) / 4;
      uint32_t g = define(c, d, false);
      for (unsigned k = 0; k < d->num_components; k++)
         emit(c, APEX_COPY, sub(g, k, 1), apex_phys(APEX_SCALAR + base + k), 0, 0, 0);
      return;
   }
   case nir_intrinsic_shader_clock:
      emit(c, APEX_S_MEMTIME, define(c, d, false), 0, 0, 0, 0);
      return;
   case nir_intrinsic_barrier:
      barrier(c, i);
      return;
   case nir_intrinsic_load_ssbo: case nir_intrinsic_load_ubo:
      memory(c, i, raw_root(i->src[0]) ? ROOT : BUFFER, false, false);
      return;
   case nir_intrinsic_store_ssbo:
      memory(c, i, raw_root(i->src[1]) ? ROOT : BUFFER, true, false);
      return;
   case nir_intrinsic_ssbo_atomic: case nir_intrinsic_ssbo_atomic_swap:
      memory(c, i, raw_root(i->src[0]) ? ROOT : BUFFER, false, true);
      return;
   case nir_intrinsic_get_ssbo_size: {
      if (i->src[0].ssa->num_components != 4) {
         failf(c, "unsupported buffer index: get_ssbo_size", NULL);
         return;
      }
      uint32_t desc = buffer_descriptor(c, i->src[0]);
      alias(c, d, sub(desc, 2, 1));
      return;
   }
   case nir_intrinsic_load_global_2x32:
      memory(c, i, GLOBAL, false, false);
      return;
   case nir_intrinsic_store_global_2x32:
      memory(c, i, GLOBAL, true, false);
      return;
   case nir_intrinsic_global_atomic_2x32: case nir_intrinsic_global_atomic_swap_2x32:
      memory(c, i, GLOBAL, false, true);
      return;
   case nir_intrinsic_load_shared:
      memory(c, i, SHARED, false, false);
      return;
   case nir_intrinsic_store_shared:
      memory(c, i, SHARED, true, false);
      return;
   case nir_intrinsic_shared_atomic: case nir_intrinsic_shared_atomic_swap:
      memory(c, i, SHARED, false, true);
      return;
   case nir_intrinsic_load_scratch:
      memory(c, i, SCRATCH, false, false);
      return;
   case nir_intrinsic_store_scratch:
      memory(c, i, SCRATCH, true, false);
      return;
   case nir_intrinsic_load_push_constant: {
      int64_t imm;
      uint32_t offset = split_offset(c, i->src[0], &imm, false);
      imm += STANDALONE_PUSH + nir_intrinsic_base(i);
      if (i->src[0].ssa->divergent) {
         uint32_t g = define(c, d, true), n = d->num_components;
         for (unsigned k = 0, piece; k < n; k += piece) {
            piece = piece_dwords(access_alignment(i), 4 * k, n - k, false);
            emit(c, APEX_GLOBAL_LOAD, sub(g, k, piece), vector(c, offset), root(c), 0,
                 mem_hi(imm + 4 * k, piece - 1));
            last(c)->flags = APEX_REORDER;
         }
      } else {
         alias(c, d, s_load(c, APEX_S_LOAD, root(c), offset, imm, d->num_components, access_alignment(i)));
      }
      return;
   }
   /* Subgroups. */
   case nir_intrinsic_ballot: {
      nir_src b = i->src[0];
      if (!b.ssa->divergent)
         emit(c, APEX_S_CSELECT, define(c, d, false), apex_phys(APEX_EXEC), constant(c, 0), flag_of(c, b, 0), 0);
      else
         emit(c, APEX_S_AND, define(c, d, false), src(c, b, 0), apex_phys(APEX_EXEC), 0, 0);
      return;
   }
   case nir_intrinsic_inverse_ballot:
      alias(c, d, i->src[0].ssa->divergent ? src(c, i->src[0], 0) : scalar(c, src(c, i->src[0], 0)));
      return;
   case nir_intrinsic_vote_any: case nir_intrinsic_vote_all: {
      nir_src b = i->src[0];
      if (!b.ssa->divergent) {
         alias(c, d, flag_of(c, b, 0));
         return;
      }
      uint32_t m = op2(c, APEX_S_AND, false, src(c, b, 0), apex_phys(APEX_EXEC));
      if (i->intrinsic == nir_intrinsic_vote_any)
         emit(c, APEX_S_CMP_NE, define(c, d, false), m, constant(c, 0), 0, 0);
      else
         emit(c, APEX_S_CMP_EQ, define(c, d, false), m, apex_phys(APEX_EXEC), 0, 0);
      return;
   }
   case nir_intrinsic_elect: {
      uint32_t first = op1(c, APEX_S_FF1, false, apex_phys(APEX_EXEC));
      emit(c, APEX_S_SHL, define(c, d, false), constant(c, 1), first, 0, 0);
      return;
   }
   case nir_intrinsic_first_invocation:
      emit(c, APEX_S_FF1, define(c, d, false), apex_phys(APEX_EXEC), 0, 0, 0);
      return;
   case nir_intrinsic_read_first_invocation:
   case nir_intrinsic_read_invocation: {
      if (d->bit_size == 1) {
         failf(c, "boolean subgroup reads must be lowered", NULL);
         return;
      }
      uint32_t g = define(c, d, false);
      for (unsigned k = 0; k < d->num_components * words(d); k++) {
         uint32_t x = src(c, i->src[0], 0);
         x = sub(x, k, 1);
         if (!is_vec(c, x))
            emit(c, APEX_COPY, sub(g, k, 1), x, 0, 0, 0);
         else if (i->intrinsic == nir_intrinsic_read_first_invocation)
            emit(c, APEX_V_READFIRSTLANE, sub(g, k, 1), x, 0, 0, 0);
         else
            emit(c, APEX_V_READLANE, sub(g, k, 1), x, scalar(c, src(c, i->src[1], 0)), 0, 0);
      }
      return;
   }
   case nir_intrinsic_quad_broadcast: case nir_intrinsic_quad_swap_horizontal:
   case nir_intrinsic_quad_swap_vertical: case nir_intrinsic_quad_swap_diagonal: {
      /* Two bits per quad lane select its source lane. */
      unsigned pattern = i->intrinsic == nir_intrinsic_quad_swap_horizontal ? 0xb1 :
                         i->intrinsic == nir_intrinsic_quad_swap_vertical ? 0x4e :
                         i->intrinsic == nir_intrinsic_quad_swap_diagonal ? 0x1b :
                         (nir_src_as_uint(i->src[1]) & 3) * 0x55;
      uint32_t g = define(c, d, true);
      for (unsigned k = 0; k < words(d); k++)
         emit(c, APEX_V_QUADPERM, sub(g, k, 1), vector(c, sub(src(c, i->src[0], 0), k, 1)),
              constant(c, pattern), 0, 0);
      return;
   }
   case nir_intrinsic_mbcnt_amd: {
      uint32_t m = scalar(c, src(c, i->src[0], 0));
      uint32_t count = op1(c, APEX_V_MBCNT, true, m);
      emit(c, APEX_V_ADD, define(c, d, true), count, src(c, i->src[1], 0), 0, 0);
      return;
   }
   case nir_intrinsic_shuffle: {
      uint32_t g = define(c, d, true);
      for (unsigned k = 0; k < words(d); k++)
         emit(c, APEX_V_PERMLANE, sub(g, k, 1), vector(c, sub(src(c, i->src[0], 0), k, 1)),
              src(c, i->src[1], 0), 0, 0);
      return;
   }
   /* Fragment. */
   case nir_intrinsic_terminate: case nir_intrinsic_terminate_if:
   case nir_intrinsic_demote: case nir_intrinsic_demote_if: {
      bool terminate = i->intrinsic == nir_intrinsic_terminate || i->intrinsic == nir_intrinsic_terminate_if;
      bool conditional = i->intrinsic == nir_intrinsic_terminate_if || i->intrinsic == nir_intrinsic_demote_if;
      uint32_t m = conditional ? op2(c, APEX_S_AND, false, mask_of(c, i->src[0], 0), apex_phys(APEX_EXEC))
                               : exec_copy(c);
      uint32_t *state = terminate ? &c->alive : &c->covered;
      emit(c, APEX_S_ANDN2, *state, *state, m, 0, 0);
      if (terminate) {
         set_exec(c, op2(c, APEX_S_ANDN2, false, apex_phys(APEX_EXEC), m));
         if (c->loop && c->loop->divergent) {
            emit(c, APEX_S_ANDN2, c->loop->iter, c->loop->iter, m, 0, 0);
            emit(c, APEX_S_ANDN2, c->loop->live, c->loop->live, m, 0, 0);
         }
         branch(c, APEX_S_CBRANCH_EXECZ, 0, c->restore);
      }
      return;
   }
   case nir_intrinsic_load_helper_invocation: case nir_intrinsic_is_helper_invocation: {
      uint32_t h = c->helper;
      if (c->covered)
         h = op2(c, APEX_S_ORN2, false, h, c->covered);
      alias(c, d, h);
      return;
   }
   case nir_intrinsic_load_barycentric_pixel: case nir_intrinsic_load_barycentric_centroid:
   case nir_intrinsic_load_barycentric_sample: {
      bool centroid = i->intrinsic == nir_intrinsic_load_barycentric_centroid;
      bool linear = nir_intrinsic_interp_mode(i) == INTERP_MODE_NOPERSPECTIVE;
      if (linear && centroid) {
         failf(c, "noperspective centroid interpolation is unsupported", NULL);
         return;
      }
      if (centroid)
         c->header.flags |= 1u << 5;
      if (i->intrinsic == nir_intrinsic_load_barycentric_sample)
         c->header.flags |= 1u << 7;
      uint32_t g = define(c, d, true);
      if (linear) {
         /* Screen-space weights from the perspective ones: i Q / q1 and
          * j Q / q2 with the pixel's 1/w (v4) and the hidden 1/w1, 1/w2. */
         c->header.flags |= 1u << 6;
         for (unsigned k = 0; k < 2; k++) {
            uint32_t q = value(c, true, 1);
            emit(c, APEX_V_INTERP_FLAT, q, 0, 0, apex_raw(129 + k), 0);
            uint32_t scale = op2(c, APEX_V_MUL_F, true, apex_phys(4), op1(c, APEX_V_RCP, true, q));
            emit(c, APEX_V_MUL_F, sub(g, k, 1), apex_phys(k), scale, 0, 0);
         }
         return;
      }
      emit(c, APEX_COPY, sub(g, 0, 1), apex_phys(centroid ? 5 : 0), 0, 0, 0);
      emit(c, APEX_COPY, sub(g, 1, 1), apex_phys(centroid ? 6 : 1), 0, 0, 0);
      return;
   }
   case nir_intrinsic_load_interpolated_input: case nir_intrinsic_load_input: {
      nir_io_semantics io = nir_intrinsic_io_semantics(i);
      unsigned component = nir_intrinsic_component(i);
      if (stage == MESA_SHADER_VERTEX) {
         unsigned location = io.location - VERT_ATTRIB_GENERIC0;
         if (io.location < VERT_ATTRIB_GENERIC0 || location >= 16 || d->bit_size != 32) {
            failf(c, "unsupported vertex input", NULL);
            return;
         }
         uint32_t g = define(c, d, true);
         for (unsigned k = 0; k < d->num_components; k++)
            emit(c, APEX_COPY, sub(g, k, 1), apex_phys(2 + 4 * location + component + k), 0, 0, 0);
         return;
      }
      if (stage != MESA_SHADER_FRAGMENT || d->bit_size != 32) {
         failf(c, "unsupported stage input", NULL);
         return;
      }
      bool flat = i->intrinsic == nir_intrinsic_load_input;
      unsigned input = fragment_input(c, io.location, flat);
      uint32_t g = define(c, d, true);
      for (unsigned k = 0; k < d->num_components; k++) {
         uint32_t attribute = apex_raw(4 * input + component + k);
         if (flat) {
            emit(c, APEX_V_INTERP_FLAT, sub(g, k, 1), 0, 0, attribute, 0);
         } else {
            uint32_t bary = src(c, i->src[0], 0);
            emit(c, APEX_V_INTERP, sub(g, k, 1), sub(bary, 0, 1), sub(bary, 1, 1), attribute, 0);
         }
      }
      return;
   }
   case nir_intrinsic_load_frag_coord: {
      /* Pixel centers from v2 (x | y << 16); w is the interpolated 1/w in v4. */
      unsigned read = nir_def_components_read(d);
      uint32_t g = define(c, d, true);
      if (read & 1)
         emit(c, APEX_V_ADD_F, sub(g, 0, 1),
              op1(c, APEX_V_CVT_F_U, true, op2(c, APEX_V_AND, true, apex_phys(2), constant(c, 0xffff))),
              constant(c, 0x3f000000u), 0, 0);
      if (read & 2)
         emit(c, APEX_V_ADD_F, sub(g, 1, 1),
              op1(c, APEX_V_CVT_F_U, true, op2(c, APEX_V_SHR, true, apex_phys(2), constant(c, 16))),
              constant(c, 0x3f000000u), 0, 0);
      if (read & 4) {
         /* The depth plane (attribute code 132) at the pixel centre relative
          * to its origin x0 | y0 << 16 (code 135). */
         uint32_t origin = value(c, true, 1);
         emit(c, APEX_V_INTERP_FLAT, origin, 0, 0, apex_raw(135), 0);
         uint32_t x = op2(c, APEX_V_SUB, true, op2(c, APEX_V_AND, true, apex_phys(2), constant(c, 0xffff)),
                          op2(c, APEX_V_AND, true, origin, constant(c, 0xffff)));
         uint32_t y = op2(c, APEX_V_SUB, true, op2(c, APEX_V_SHR, true, apex_phys(2), constant(c, 16)),
                          op2(c, APEX_V_SHR, true, origin, constant(c, 16)));
         uint32_t fx = op2(c, APEX_V_ADD_F, true, op1(c, APEX_V_CVT_F_I, true, x), constant(c, 0x3f000000u));
         uint32_t fy = op2(c, APEX_V_ADD_F, true, op1(c, APEX_V_CVT_F_I, true, y), constant(c, 0x3f000000u));
         emit(c, APEX_V_INTERP, sub(g, 2, 1), fx, fy, apex_raw(132), 0);
      }
      if (read & 8) {
         c->header.flags |= 1u << 6;
         emit(c, APEX_COPY, sub(g, 3, 1), apex_phys(4), 0, 0, 0);
      }
      return;
   }
   case nir_intrinsic_load_front_face:
      alias(c, d, op2(c, APEX_V_CMP_U, false, launch_bits(c, 10, 1), constant(c, 0)));
      last(c)->f[3] = apex_raw(1);
      return;
   case nir_intrinsic_load_sample_id:
      alias(c, d, launch_bits(c, 12, 3));
      return;
   case nir_intrinsic_load_sample_mask_in:
      alias(c, d, launch_bits(c, 0, 8));
      return;
   case nir_intrinsic_load_view_index: case nir_intrinsic_load_layer_id:
      alias(c, d, copy(c, false, apex_phys(APEX_SCALAR + 16)));
      return;
   case nir_intrinsic_store_output:
      store_output(c, i);
      return;
   /* Vertex. */
   case nir_intrinsic_load_vertex_id:
      alias(c, d, launch_vector(c, 0));
      return;
   case nir_intrinsic_load_instance_id:
      alias(c, d, op2(c, APEX_V_SUB, true, apex_phys(1), apex_phys(APEX_SCALAR + 20)));
      return;
   case nir_intrinsic_load_base_vertex: case nir_intrinsic_load_first_vertex:
      alias(c, d, copy(c, false, apex_phys(APEX_SCALAR + 19)));
      return;
   case nir_intrinsic_load_base_instance:
      alias(c, d, copy(c, false, apex_phys(APEX_SCALAR + 20)));
      return;
   case nir_intrinsic_load_draw_id:
      alias(c, d, copy(c, false, apex_phys(APEX_SCALAR + 18)));
      return;
   default:
      failf(c, "unsupported normalized NIR intrinsic: ", nir_intrinsic_infos[i->intrinsic].name);
   }
}

static void emit_block(struct ctx *c, nir_block *block)
{
   nir_foreach_instr(instr, block) {
      if (c->failed)
         return;
      switch (instr->type) {
      case nir_instr_type_load_const: {
         nir_load_const_instr *k = nir_instr_as_load_const(instr);
         uint32_t g = define(c, &k->def, false);
         for (unsigned j = 0; j < k->def.num_components; j++) {
            if (k->def.bit_size == 64) {
               emit(c, APEX_CONST, sub(g, 2 * j, 1), 0, 0, 0, 0);
               last(c)->imm = (uint32_t)k->value[j].u64;
               emit(c, APEX_CONST, sub(g, 2 * j + 1, 1), 0, 0, 0, 0);
               last(c)->imm = k->value[j].u64 >> 32;
            } else {
               emit(c, APEX_CONST, sub(g, j, 1), 0, 0, 0, 0);
               last(c)->imm = k->def.bit_size == 1 ? k->value[j].b : k->value[j].u32;
            }
         }
         break;
      }
      case nir_instr_type_undef: {
         nir_undef_instr *u = nir_instr_as_undef(instr);
         uint32_t g = define(c, &u->def, false);
         for (unsigned j = 0; j < width_of(g); j++) {
            emit(c, APEX_CONST, sub(g, j, 1), 0, 0, 0, 0);
         }
         break;
      }
      case nir_instr_type_alu:
         emit_alu(c, nir_instr_as_alu(instr));
         break;
      case nir_instr_type_intrinsic:
         emit_intrinsic(c, nir_instr_as_intrinsic(instr));
         break;
      case nir_instr_type_tex:
         emit_tex(c, nir_instr_as_tex(instr));
         break;
      case nir_instr_type_jump:
         emit_jump(c, nir_instr_as_jump(instr));
         break;
      case nir_instr_type_deref:
         break;
      default:
         failf(c, "unsupported normalized NIR instruction type", NULL);
      }
   }
}

static void emit_cf(struct ctx *c, struct exec_list *list)
{
   foreach_list_typed(nir_cf_node, node, node, list) {
      if (c->failed)
         return;
      switch (node->type) {
      case nir_cf_node_block: emit_block(c, nir_cf_node_as_block(node)); break;
      case nir_cf_node_if: emit_if(c, nir_cf_node_as_if(node)); break;
      case nir_cf_node_loop: emit_loop(c, nir_cf_node_as_loop(node)); break;
      default: failf(c, "unsupported control-flow node", NULL);
      }
   }
}

/* Standalone resources and launch values. */
static bool lower_launch(nir_builder *b, nir_intrinsic_instr *i, void *data)
{
   b->cursor = nir_before_instr(&i->instr);
   nir_def *r;
   switch (i->intrinsic) {
   case nir_intrinsic_vulkan_resource_index: {
      unsigned slot = 64 * (1 + 256 * nir_intrinsic_desc_set(i) + nir_intrinsic_binding(i));
      nir_def *index = nir_iadd_imm(b, nir_imul_imm(b, i->src[0].ssa, 64), slot | SLOT_MARK);
      r = nir_vec2(b, index, nir_imm_int(b, 0));
      break;
   }
   case nir_intrinsic_vulkan_resource_reindex:
      r = nir_vec2(b, nir_iadd(b, nir_channel(b, i->src[0].ssa, 0), nir_imul_imm(b, i->src[1].ssa, 64)),
                   nir_channel(b, i->src[0].ssa, 1));
      break;
   case nir_intrinsic_load_vulkan_descriptor:
      r = i->src[0].ssa;
      break;
   case nir_intrinsic_load_num_workgroups:
      r = nir_load_kernel_input(b, 3, 32, nir_imm_int(b, 8), .base = 0, .range = 12);
      break;
   default:
      return false;
   }
   nir_def_rewrite_uses(&i->def, r);
   nir_instr_remove(&i->instr);
   return true;
}

/* Standalone buffer accesses read their descriptor from the slot in NIR, so
 * CSE and LICM share and hoist it. */
static bool lower_slots(nir_builder *b, nir_intrinsic_instr *i, void *data)
{
   unsigned index;
   switch (i->intrinsic) {
   case nir_intrinsic_load_ssbo: case nir_intrinsic_load_ubo: case nir_intrinsic_ssbo_atomic:
   case nir_intrinsic_ssbo_atomic_swap: case nir_intrinsic_get_ssbo_size:
      index = 0; break;
   case nir_intrinsic_store_ssbo:
      index = 1; break;
   default:
      return false;
   }
   nir_src *s = &i->src[index];
   if (s->ssa->num_components != 1 || (nir_src_is_const(*s) && !(nir_src_as_uint(*s) & SLOT_MARK)))
      return false;
   b->cursor = nir_before_instr(&i->instr);
   nir_def *desc = nir_load_ssbo(b, 4, 32, nir_imm_int(b, 0), nir_iand_imm(b, s->ssa, ~SLOT_MARK),
      .align_mul = 16, .access = ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER | ACCESS_CAN_SPECULATE);
   if (i->intrinsic == nir_intrinsic_get_ssbo_size) {
      nir_def_replace(&i->def, nir_channel(b, desc, 2));
      return true;
   }
   nir_src_rewrite(s, desc);
   return true;
}

/* Standalone textures read their image and sampler descriptors from the
 * slot in NIR, so CSE and LICM share and hoist them. */
static bool lower_texture_slots(nir_builder *b, nir_instr *instr, void *data)
{
   if (instr->type != nir_instr_type_tex)
      return false;
   nir_tex_instr *t = nir_instr_as_tex(instr);
   int k = nir_tex_instr_src_index(t, nir_tex_src_texture_deref);
   if (k < 0)
      return false;
   nir_deref_instr *deref = nir_src_as_deref(t->src[k].src);
   nir_variable *var = nir_deref_instr_get_variable(deref);
   b->cursor = nir_before_instr(instr);
   nir_def *slot = nir_imm_int(b, 64 * (1 + 256 * var->data.descriptor_set + var->data.binding));
   if (deref->deref_type == nir_deref_type_array)
      slot = nir_iadd(b, slot, nir_imul_imm(b, deref->arr.index.ssa, 64));
   nir_def *words[2];
   for (unsigned d = 0; d < 2; d++)
      words[d] = nir_load_ssbo(b, 8, 32, nir_imm_int(b, 0), nir_iadd_imm(b, slot, 32 * d),
         .align_mul = 32, .access = ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER | ACCESS_CAN_SPECULATE);
   nir_tex_instr_remove_src(t, k);
   int s = nir_tex_instr_src_index(t, nir_tex_src_sampler_deref);
   if (s >= 0)
      nir_tex_instr_remove_src(t, s);
   nir_tex_instr_add_src(t, nir_tex_src_backend1, words[0]);
   nir_tex_instr_add_src(t, nir_tex_src_backend2, words[1]);
   return true;
}

static bool subword_roundtrip(const nir_instr *instr, const void *data)
{
   if (instr->type != nir_instr_type_alu)
      return false;
   const nir_alu_instr *a = nir_instr_as_alu(instr);
   if (a->op != nir_op_u2u32 || a->def.num_components != 1)
      return false;
   nir_alu_instr *narrow = nir_def_as_alu_or_null(a->src[0].src.ssa);
   return narrow && (narrow->op == nir_op_u2u8 || narrow->op == nir_op_u2u16) &&
      narrow->src[0].src.ssa->bit_size == 32;
}
static nir_def *lower_subword_roundtrip(nir_builder *b, nir_instr *instr, void *data)
{
   nir_alu_instr *a = nir_instr_as_alu(instr);
   nir_alu_instr *narrow = nir_def_as_alu_or_null(a->src[0].src.ssa);
   nir_def *source = nir_channel(b, nir_ssa_for_alu_src(b, narrow, 0), a->src[0].swizzle[0]);
   return nir_iand_imm(b, source, BITFIELD_MASK(narrow->def.bit_size));
}

/* 64-bit global addresses become paired words of up to four dwords. */
static bool lower_global(nir_builder *b, nir_intrinsic_instr *i, void *data)
{
   bool store = i->intrinsic == nir_intrinsic_store_global;
   bool atomic = i->intrinsic == nir_intrinsic_global_atomic;
   bool swap = i->intrinsic == nir_intrinsic_global_atomic_swap;
   bool load = i->intrinsic == nir_intrinsic_load_global || i->intrinsic == nir_intrinsic_load_global_constant;
   if (!store && !atomic && !swap && !load)
      return false;
   b->cursor = nir_before_instr(&i->instr);
   nir_def *address = nir_unpack_64_2x32(b, i->src[store ? 1 : 0].ssa);
   nir_def *r = NULL;
   enum gl_access_qualifier access = nir_intrinsic_access(i);
   if (i->intrinsic == nir_intrinsic_load_global_constant)
      access |= ACCESS_CAN_REORDER | ACCESS_NON_WRITEABLE;
   if (store)
      nir_store_global_2x32(b, i->src[0].ssa, address, .align_mul = nir_intrinsic_align_mul(i),
                            .align_offset = nir_intrinsic_align_offset(i),
                            .write_mask = nir_intrinsic_write_mask(i), .access = access);
   else if (swap)
      r = nir_global_atomic_swap_2x32(b, i->def.bit_size, address, i->src[1].ssa, i->src[2].ssa,
                                      .atomic_op = nir_intrinsic_atomic_op(i), .access = access);
   else if (atomic)
      r = nir_global_atomic_2x32(b, i->def.bit_size, address, i->src[1].ssa,
                                 .atomic_op = nir_intrinsic_atomic_op(i), .access = access);
   else
      r = nir_load_global_2x32(b, i->def.num_components, i->def.bit_size, address,
                               .align_mul = nir_intrinsic_align_mul(i),
                               .align_offset = nir_intrinsic_align_offset(i), .access = access);
   if (r)
      nir_def_rewrite_uses(&i->def, r);
   nir_instr_remove(&i->instr);
   return true;
}

static nir_mem_access_size_align
access_size(nir_intrinsic_op op, uint8_t bytes, uint8_t bits, uint32_t align_mul,
            uint32_t align_offset, bool constant, enum gl_access_qualifier access, const void *data)
{
   uint32_t align = nir_combined_align(align_mul, align_offset);
   if (align < 4)
      return (nir_mem_access_size_align){.num_components = 1, .bit_size = 32, .align = 4,
                                         .shift = nir_mem_access_shift_method_scalar};
   unsigned limit = (access & ACCESS_CAN_REORDER) || op == nir_intrinsic_load_ubo ? 16 : 4;
   return (nir_mem_access_size_align){.num_components = MAX2(MIN2(bytes / 4, limit), 1),
                                      .bit_size = 32, .align = 4,
                                      .shift = nir_mem_access_shift_method_scalar};
}

/* Adjacent dword accesses merge into 64-128-bit lane (or scalar) accesses. */
static bool vectorize(unsigned align_mul, unsigned align_offset, unsigned bit_size,
                      unsigned num_components, int64_t hole_size, nir_intrinsic_instr *low,
                      nir_intrinsic_instr *high, void *data)
{
   /* Read-only loads may become wide scalar loads. */
   bool constant = nir_intrinsic_infos[low->intrinsic].has_dest &&
                   (low->intrinsic == nir_intrinsic_load_ubo ||
                    (nir_intrinsic_has_access(low) && (nir_intrinsic_access(low) & ACCESS_CAN_REORDER)));
   /* Vector accesses take 1-4 dwords at their natural alignment (16 bytes
    * for three); scalar loads split into aligned pieces. */
   unsigned align = nir_combined_align(align_mul, align_offset);
   return bit_size == 32 && num_components <= (constant ? 16 : 4) && hole_size <= 0 && align >= 4 &&
          (constant || align >= 4 * util_next_power_of_two(num_components));
}

/* A value an if's branch computes and the code after the if computes again
 * moves before the if, where CSE shares it; the work never runs more often
 * than before. */
static bool available_before(nir_src *src, void *before)
{
   return nir_block_dominates(nir_def_block(src->ssa), before);
}
static bool hoist_repeated(nir_shader *nir)
{
   bool progress = false;
   nir_foreach_function_impl(impl, nir) {
      nir_metadata_require(impl, nir_metadata_dominance);
      bool moved = false;
      nir_foreach_block(before, impl) {
         nir_cf_node *next = nir_cf_node_next(&before->cf_node);
         if (!next || next->type != nir_cf_node_if)
            continue;
         nir_block *after = nir_cf_node_as_block(nir_cf_node_next(next));
         nir_foreach_block_in_cf_node(block, next) {
            nir_foreach_instr_safe(instr, block) {
               if ((instr->type != nir_instr_type_alu && instr->type != nir_instr_type_intrinsic) ||
                   !nir_instr_can_speculate(instr) || !nir_foreach_src(instr, available_before, before))
                  continue;
               nir_foreach_instr(later, after) {
                  if (nir_instrs_equal(instr, later)) {
                     nir_instr_move(nir_after_block(before), instr);
                     moved = true;
                     break;
                  }
               }
            }
         }
      }
      progress |= nir_progress(moved, impl, nir_metadata_control_flow);
   }
   return progress;
}

/* Position keeps its association, so pipelines computing the same position
 * expression rasterize the same depth (equal-depth multipass). */
static void keep_position_order(nir_shader *nir)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   struct util_dynarray stack;
   util_dynarray_init(&stack, NULL);
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         instr->pass_flags = 0;
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *i = nir_instr_as_intrinsic(instr);
         if (i->intrinsic == nir_intrinsic_store_output &&
             nir_intrinsic_io_semantics(i).location == VARYING_SLOT_POS)
            util_dynarray_append(&stack, nir_def_instr(i->src[0].ssa));
      }
   }
   while (util_dynarray_num_elements(&stack, nir_instr *)) {
      nir_instr *instr = util_dynarray_pop(&stack, nir_instr *);
      if (instr->pass_flags || (instr->type != nir_instr_type_alu && instr->type != nir_instr_type_phi))
         continue;
      instr->pass_flags = 1;
      if (instr->type == nir_instr_type_phi) {
         nir_foreach_phi_src(p, nir_instr_as_phi(instr))
            util_dynarray_append(&stack, nir_def_instr(p->src.ssa));
         continue;
      }
      nir_alu_instr *alu = nir_instr_as_alu(instr);
      alu->fp_math_ctrl |= nir_op_valid_fp_math_ctrl(alu->op, nir_fp_no_reassoc | nir_fp_no_transform);
      for (unsigned k = 0; k < nir_op_infos[alu->op].num_inputs; k++)
         util_dynarray_append(&stack, nir_def_instr(alu->src[k].src.ssa));
   }
   util_dynarray_fini(&stack);
}

static int fail(struct apex_compile_result *output, const char *message)
{
   snprintf(output->diagnostic, sizeof(output->diagnostic), "%s", message);
   return 1;
}

static bool licm_filter(nir_instr *instr, nir_loop *loop, bool dominates_exit)
{
   return dominates_exit || nir_instr_can_speculate(instr);
}

static unsigned type_size(const struct glsl_type *type, bool bindless)
{
   return glsl_count_attribute_slots(type, false);
}

int apex_from_nir(nir_shader *nir, struct apex_compile_result *output)
{
   *output = (struct apex_compile_result){0};
   mesa_shader_stage stage = nir->info.stage;
   if (stage != MESA_SHADER_COMPUTE && stage != MESA_SHADER_VERTEX && stage != MESA_SHADER_FRAGMENT)
      return fail(output, "only compute, vertex and fragment shaders are supported");
   nir_validate_shader(nir, "Apex SPIR-V import");
   bool zero_shared = nir->info.zero_initialize_shared_memory;
   nir_foreach_variable_with_modes(var, nir, nir_var_mem_shared) {
      if (var->constant_initializer) {
         if (!var->constant_initializer->is_null_constant)
            return fail(output, "shared initializer must be zero");
         zero_shared = true;
      }
   }
   unsigned local[3] = {16, 1, 1};
   if (stage == MESA_SHADER_COMPUTE) {
      unsigned invocations = 1;
      static const unsigned axis_limits[3] = {256, 256, 64};
      for (unsigned axis = 0; axis < 3; axis++) {
         unsigned size = nir->info.workgroup_size[axis];
         if (!size || size > axis_limits[axis] || invocations > 256 / size)
            return fail(output, "native launch requires constant local dimensions within 256x256x64 and at most 256 invocations");
         invocations *= size;
         local[axis] = size;
      }
      if (nir->info.workgroup_size_variable)
         return fail(output, "native launch requires constant local dimensions within 256x256x64 and at most 256 invocations");
   }
   NIR_PASS(_, nir, nir_lower_variable_initializers, nir_var_function_temp);
   NIR_PASS(_, nir, nir_lower_returns);
   NIR_PASS(_, nir, nir_inline_functions);
   NIR_PASS(_, nir, nir_opt_deref);
   nir_remove_non_entrypoints(nir);
   NIR_PASS(_, nir, nir_lower_variable_initializers, nir_var_shader_temp);
   if (stage != MESA_SHADER_COMPUTE) {
      NIR_PASS(_, nir, nir_split_per_member_structs);
      NIR_PASS(_, nir, nir_lower_io_vars_to_temporaries, nir_shader_get_entrypoint(nir),
               nir_var_shader_out);
      NIR_PASS(_, nir, nir_lower_global_vars_to_local);
      NIR_PASS(_, nir, nir_split_var_copies);
      NIR_PASS(_, nir, nir_lower_var_copies);
   }
   NIR_PASS(_, nir, nir_lower_global_vars_to_local);
   /* Graphics launches have no private base: dynamically indexed locals
    * become selects over registers. */
   if (stage != MESA_SHADER_COMPUTE)
      NIR_PASS(_, nir, nir_lower_indirect_derefs_to_if_else_trees, nir_var_function_temp, UINT32_MAX);
   NIR_PASS(_, nir, nir_lower_vars_to_ssa);
   NIR_PASS(_, nir, nir_opt_undef);
   /* Unroll before private arrays lower to memory, so constant indices
    * leave them in registers. */
   bool unrolled;
   do {
      unrolled = false;
      NIR_PASS(unrolled, nir, nir_lower_vars_to_ssa);
      NIR_PASS(unrolled, nir, nir_opt_copy_prop);
      NIR_PASS(unrolled, nir, nir_opt_remove_phis);
      NIR_PASS(unrolled, nir, nir_opt_dce);
      NIR_PASS(unrolled, nir, nir_opt_constant_folding);
      NIR_PASS(unrolled, nir, nir_opt_algebraic);
      NIR_PASS(unrolled, nir, nir_opt_dead_cf);
      NIR_PASS(unrolled, nir, nir_opt_deref);
      NIR_PASS(unrolled, nir, nir_opt_loop_unroll);
   } while (unrolled);
   NIR_PASS(_, nir, nir_remove_dead_variables, nir_var_function_temp | nir_var_shader_in | nir_var_shader_out, NULL);
   if (stage != MESA_SHADER_COMPUTE) {
      nir_assign_io_var_locations(nir, nir_var_shader_in);
      nir_assign_io_var_locations(nir, nir_var_shader_out);
      NIR_PASS(_, nir, nir_lower_io, nir_var_shader_in | nir_var_shader_out, type_size,
               nir_lower_io_use_interpolated_input_intrinsics);
      NIR_PASS(_, nir, nir_lower_io_to_scalar, nir_var_shader_out, NULL, NULL);
   }
   if (zero_shared && nir->info.shared_size) {
      if (nir->info.shared_size > 32768)
         return fail(output, "shared memory exceeds 32768 bytes");
      NIR_PASS(_, nir, nir_zero_initialize_shared_memory, nir->info.shared_size, 16);
   }
   NIR_PASS(_, nir, nir_lower_vars_to_explicit_types, nir_var_function_temp, glsl_get_natural_size_align_bytes);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_function_temp, nir_address_format_32bit_offset);
   NIR_PASS(_, nir, nir_lower_vars_to_explicit_types, nir_var_mem_shared, glsl_get_natural_size_align_bytes);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_shared, nir_address_format_32bit_offset);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_ssbo | nir_var_mem_ubo, nir_address_format_32bit_index_offset);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_push_const, nir_address_format_32bit_offset);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_global, nir_address_format_64bit_global);
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_launch, nir_metadata_control_flow, NULL);
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_slots, nir_metadata_control_flow, NULL);
   NIR_PASS(_, nir, nir_shader_instructions_pass, lower_texture_slots, nir_metadata_control_flow, NULL);
   const nir_lower_mem_access_bit_sizes_options mem = {
      .callback = access_size,
      .modes = nir_var_mem_global | nir_var_function_temp | nir_var_mem_ssbo | nir_var_mem_ubo |
               nir_var_mem_shared | nir_var_mem_push_const,
      .may_lower_unaligned_stores_to_atomics = true,
   };
   NIR_PASS(_, nir, nir_lower_mem_access_bit_sizes, &mem);
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_global, nir_metadata_none, NULL);
   NIR_PASS(_, nir, nir_lower_int64);
   NIR_PASS(_, nir, nir_lower_64bit_phis);
   NIR_PASS(_, nir, nir_lower_system_values);
   NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_launch, nir_metadata_control_flow, NULL);
   if (stage == MESA_SHADER_COMPUTE) {
      const nir_lower_compute_system_values_options geometry = {.lower_cs_local_id_to_index = true};
      NIR_PASS(_, nir, nir_lower_compute_system_values, &geometry);
   }
   NIR_PASS(_, nir, nir_lower_flrp, 32, false);
   NIR_PASS(_, nir, nir_lower_alu_to_scalar, NULL, NULL);
   NIR_PASS(_, nir, nir_opt_constant_folding);
   NIR_PASS(_, nir, nir_opt_algebraic);
   NIR_PASS(_, nir, nir_opt_idiv_const, 32);
   NIR_PASS(_, nir, nir_lower_idiv, &(nir_lower_idiv_options){0});
   const nir_lower_subgroups_options subgroups = {
      .subgroup_size = 16, .ballot_bit_size = 32, .ballot_components = 1,
      .lower_to_scalar = true, .lower_vote_ieq = true, .lower_vote_feq = true,
      .lower_vote_bool_eq = true, .lower_subgroup_masks = true, .lower_relative_shuffle = true,
      .lower_quad_broadcast_dynamic = true, .lower_quad_vote = true,
      .lower_ballot_bit_count_to_mbcnt_amd = true,
      .lower_reduce = true, .lower_rotate_to_shuffle = true,
      .lower_shuffle_to_32bit = true,
   };
   bool progress;
   do {
      progress = false;
      NIR_PASS(progress, nir, nir_lower_subgroups, &subgroups);
      NIR_PASS(progress, nir, nir_lower_alu_to_scalar, NULL, NULL);
      NIR_PASS(progress, nir, nir_lower_phis_to_scalar, NULL, NULL);
      NIR_PASS(progress, nir, nir_opt_algebraic);
      NIR_PASS(progress, nir, nir_lower_pack);
      NIR_PASS(progress, nir, nir_lower_alu);
      NIR_PASS(progress, nir, nir_shader_lower_instructions, subword_roundtrip, lower_subword_roundtrip, NULL);
      NIR_PASS(progress, nir, nir_opt_copy_prop);
      NIR_PASS(progress, nir, nir_opt_remove_phis);
      NIR_PASS(progress, nir, nir_opt_dce);
      NIR_PASS(progress, nir, nir_opt_constant_folding);
      NIR_PASS(progress, nir, hoist_repeated);
      NIR_PASS(progress, nir, nir_opt_cse);
      NIR_PASS(progress, nir, nir_opt_licm, licm_filter);
      NIR_PASS(progress, nir, nir_opt_dead_cf);
      NIR_PASS(progress, nir, nir_opt_peephole_select, &(nir_opt_peephole_select_options){.limit = 8});
      NIR_PASS(progress, nir, nir_opt_loop_unroll);
   } while (progress);
   const nir_load_store_vectorize_options vector_access = {
      .callback = vectorize,
      .modes = nir_var_mem_ssbo | nir_var_mem_ubo | nir_var_mem_shared | nir_var_mem_push_const,
      .bounds_checked_modes = nir_var_mem_ssbo | nir_var_mem_ubo,
   };
   bool vectorized = false;
   NIR_PASS(vectorized, nir, nir_opt_load_store_vectorize, &vector_access);
   if (vectorized) {
      NIR_PASS(_, nir, nir_opt_copy_prop);
      NIR_PASS(_, nir, nir_opt_cse);
      NIR_PASS(_, nir, nir_opt_dce);
   }
   NIR_PASS(_, nir, nir_opt_algebraic_late);
   /* Fusion leaves the fused products behind until DCE; is_used_once
    * patterns need them gone. */
   NIR_PASS(_, nir, nir_opt_dce);
   if (stage == MESA_SHADER_VERTEX)
      keep_position_order(nir);
   NIR_PASS(_, nir, apex_nir_opt_late);
   NIR_PASS(_, nir, nir_opt_copy_prop);
   NIR_PASS(_, nir, nir_opt_dce);
   NIR_PASS(_, nir, nir_lower_continue_constructs);
   NIR_PASS(_, nir, nir_opt_dead_cf);
   /* Vectors of loop values build after the loop, not every iteration. */
   NIR_PASS(_, nir, nir_opt_sink, nir_move_copies);
   /* Values leaving a loop pass through exit phis, so a value that is
    * uniform inside a loop with a divergent exit becomes divergent there. */
   NIR_PASS(_, nir, nir_convert_to_lcssa, true, true);
   nir_divergence_analysis(nir);
   NIR_PASS(_, nir, nir_convert_from_ssa, true, true);
   NIR_PASS(_, nir, nir_trivialize_registers);
   nir_validate_shader(nir, "Apex normalized NIR");
   if (getenv("APEX_DUMP_NIR"))
      nir_print_shader(nir, stderr);
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_index_ssa_defs(impl);
   struct ctx c = {.nir = nir, .diagnostic = output->diagnostic};
   util_dynarray_init(&c.ops, NULL);
   util_dynarray_init(&c.values, NULL);
   struct apex_value reserved = {0, 1};
   util_dynarray_append(&c.values, reserved);
   c.defs = calloc(impl->ssa_alloc + 1, sizeof(*c.defs));
   c.vector = calloc(impl->ssa_alloc + 1, sizeof(bool));
   c.saturated = calloc(impl->ssa_alloc + 1, sizeof(bool));
   c.range = _mesa_pointer_hash_table_create(NULL);
   c.header.stage = stage == MESA_SHADER_COMPUTE ? APEX_COMPUTE :
                    stage == MESA_SHADER_VERTEX ? APEX_VERTEX : APEX_FRAGMENT;
   memcpy(c.header.local, local, sizeof(local));
   c.header.shared = stage == MESA_SHADER_COMPUTE ? nir->info.shared_size : 0;
   c.header.private_bytes = stage == MESA_SHADER_COMPUTE ? align(nir->scratch_size, 4) : 0;
   classify(&c, impl);
   c.root = value(&c, false, 2);
   emit(&c, APEX_COPY, sub(c.root, 0, 1), apex_phys(APEX_SCALAR + 0), 0, 0, 0);
   emit(&c, APEX_COPY, sub(c.root, 1, 1), apex_phys(APEX_SCALAR + 1), 0, 0, 0);
   int result = 1;
   if (stage == MESA_SHADER_FRAGMENT) {
      bool terminates = false, demotes = false, side = false;
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_op op = nir_instr_as_intrinsic(instr)->intrinsic;
            terminates |= op == nir_intrinsic_terminate || op == nir_intrinsic_terminate_if;
            demotes |= op == nir_intrinsic_demote || op == nir_intrinsic_demote_if;
            side |= op == nir_intrinsic_store_ssbo || op == nir_intrinsic_ssbo_atomic ||
                    op == nir_intrinsic_ssbo_atomic_swap || op == nir_intrinsic_store_global_2x32 ||
                    op == nir_intrinsic_global_atomic_2x32 || op == nir_intrinsic_global_atomic_swap_2x32;
         }
      }
      c.helper = value(&c, false, 1);
      emit(&c, APEX_V_CMP_U, c.helper, launch_bits(&c, 11, 1), constant(&c, 0), apex_raw(1), 0);
      if (terminates) {
         c.alive = exec_copy(&c);
         c.header.flags |= 1u << 1;
      }
      if (demotes) {
         c.covered = exec_copy(&c);
         c.header.flags |= 1u << 1;
      }
      if (side)
         c.header.flags |= 1u << 4;
      if (nir->info.outputs_written & BITFIELD64_BIT(FRAG_RESULT_DEPTH))
         c.header.flags |= 1u << 2;
      if (nir->info.outputs_written & BITFIELD64_BIT(FRAG_RESULT_SAMPLE_MASK))
         c.header.flags |= 1u << 3;
      if (nir->info.fs.early_fragment_tests || !(c.header.flags & 0x1e))
         c.header.flags |= 1u;
   }
   c.end = label(&c);
   c.restore = c.end;
   emit_cf(&c, &impl->body);
   if (c.failed)
      goto done;
   place(&c, c.end);
   finish_outputs(&c);
   emit(&c, APEX_S_ENDPGM, 0, 0, 0, 0, 0);
   if (c.failed)
      goto done;
   nir_validate_shader(nir, "Apex backend boundary");
   result = apex_emit(util_dynarray_begin(&c.ops), util_dynarray_num_elements(&c.ops, struct apex_op),
                      util_dynarray_begin(&c.values),
                      util_dynarray_num_elements(&c.values, struct apex_value), &c.header, output);
done:
   free(c.defs);
   free(c.vector);
   free(c.saturated);
   _mesa_hash_table_destroy(c.range, NULL);
   util_dynarray_fini(&c.ops);
   util_dynarray_fini(&c.values);
   return result;
}
