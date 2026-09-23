/* SPDX-License-Identifier: MIT */
#include "apex.h"
#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "util/u_dynarray.h"
#include <stdio.h>

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
   bool progress;
   do {
      progress = false;
      NIR_PASS(progress, nir, nir_opt_copy_prop);
      NIR_PASS(progress, nir, nir_opt_dce);
      NIR_PASS(progress, nir, nir_opt_constant_folding);
      NIR_PASS(progress, nir, nir_opt_cse);
   } while (progress);
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
            if (a->op == nir_op_ishr) {
               /* Complement negative operands before and after the logical
                * shift so vacated bits receive their original sign. */
               uint32_t operand = value(a->src[0].src.ssa, a->src[0].swizzle[0]);
               uint32_t count = value(a->src[1].src.ssa, a->src[1].swizzle[0]);
               uint32_t top = temporary++, sign = temporary++, zero = temporary++;
               uint32_t mask = temporary++, biased = temporary++, shifted = temporary++;
               emit(&ops, 0x20, top, 0, 0, 0, 31);
               emit(&ops, 0x29, sign, operand, top, 0, 0);
               emit(&ops, 0x20, zero, 0, 0, 0, 0);
               emit(&ops, 0x23, mask, zero, sign, 0, 0);
               emit(&ops, 0x27, biased, operand, mask, 0, 0);
               emit(&ops, 0x29, shifted, biased, count, 0, 0);
               emit(&ops, 0x27, value(&a->def, 0), shifted, mask, 0, 0);
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
            case nir_intrinsic_shuffle:
               if (i->num_components!=1) goto unsupported;
               emit(&ops,0x44,value(&i->def,0),value(i->src[0].ssa,0),value(i->src[1].ssa,0),0,0); break;
            case nir_intrinsic_barrier:
               if (nir_intrinsic_execution_scope(i)!=SCOPE_WORKGROUP || nir_intrinsic_memory_scope(i)>SCOPE_DEVICE) goto unsupported;
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
            case nir_intrinsic_shared_atomic:
            case nir_intrinsic_shared_atomic_swap: {
               bool shared=i->intrinsic==nir_intrinsic_shared_atomic || i->intrinsic==nir_intrinsic_shared_atomic_swap;
               int operation=atomic_op(nir_intrinsic_atomic_op(i));
               if (operation<0 || i->num_components!=1) goto unsupported;
               uint32_t addr=value(i->src[shared?0:1].ssa,0);
               unsigned data=shared?1:2;
               if (shared) {if(nir_intrinsic_base(i)) goto unsupported;}
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
