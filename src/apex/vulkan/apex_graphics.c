/* SPDX-License-Identifier: MIT */
/* Graphics shaders as native programs. A draw executes three jobs on the
 * ordinary queue: the vertex kernel writes one output record per vertex, the
 * internal setup kernel clips, snaps and writes primitive records, and the
 * fragment kernel walks screen tiles, rasterizes every primitive in order,
 * runs the fragment shader in 2x2 quads and merges into the attachments.
 * Draw state is data in the job trailer (apex_draw.h); attachment formats
 * specialize the fragment kernel. */
#include "apex_graphics.h"
#include "apex_draw.h"
#include "apex_setup_spv.h"
#include "apex_bin_spv.h"
#include "apex_copy_spv.h"
#include "apex_clear_spv.h"
#include "apex_timestamp_spv.h"
#include "apex_querycopy_spv.h"
#include "apex_resolve_spv.h"
#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_format_convert.h"
#include "compiler/spirv/nir_spirv.h"
#include "util/format/u_format.h"
#include "compiler/spirv/spirv_info.h"
#include "vk_alloc.h"
#include "vk_format.h"
#include "vk_graphics_state.h"
#include "vk_log.h"
#include "vk_nir.h"
#include "vk_pipeline.h"
#include "vk_shader.h"

/* Byte offset of draw word `word` in a graphics job's data root. */
static unsigned
draw_offset(const struct apex_program *program, unsigned word)
{
   return apex_program_trailer(program) + sizeof(struct apex_dispatch_parameters) + word * 4;
}

/* Draw words are reloaded at each use: uniform values have no scalar
 * register home here, and hoisting them keeps vector registers live. */
static nir_def *
root_word(nir_builder *b, unsigned offset)
{
   return nir_load_ssbo(b, 1, 32, nir_imm_int(b, 0), nir_imm_int(b, offset), .align_mul = 4,
                        .access = ACCESS_NON_WRITEABLE);
}

static nir_def *
draw_address(nir_builder *b, const struct apex_program *program, unsigned word)
{
   return nir_vec2(b, root_word(b, draw_offset(program, word)),
                   root_word(b, draw_offset(program, word + 1)));
}

/* Dynamic words of indirect draws come from the resolved parameter block. */
static nir_def *
draw_word(nir_builder *b, const struct apex_program *program, unsigned word)
{
   if (!APEX_DRAW_DYNAMIC(word))
      return root_word(b, draw_offset(program, word));
   nir_def *params = draw_address(b, program, APEX_DRAW_PARAMS);
   nir_push_if(b, nir_ine_imm(b, nir_ior(b, nir_channel(b, params, 0), nir_channel(b, params, 1)), 0));
   nir_def *resolved = nir_load_global_2x32(b, 1, 32,
      nir_build_addr_iadd_imm(b, params, nir_address_format_2x32bit_global, nir_var_mem_global, word * 4),
      .align_mul = 4);
   nir_push_else(b, NULL);
   nir_def *direct = root_word(b, draw_offset(program, word));
   nir_pop_if(b, NULL);
   return nir_if_phi(b, resolved, direct);
}

static nir_def *
address_add(nir_builder *b, nir_def *address, nir_def *bytes)
{
   return nir_build_addr_iadd(b, address, nir_address_format_2x32bit_global,
                              nir_var_mem_global, bytes);
}

static nir_def *
load_word(nir_builder *b, nir_def *address)
{
   return nir_load_global_2x32(b, 1, 32, address, .align_mul = 4);
}

static void
store_word(nir_builder *b, nir_def *value, nir_def *address)
{
   nir_store_global_2x32(b, value, address, .align_mul = 4);
}

/* Native launch geometry of the current job: coarse workgroup index. */
static nir_def *
native_workgroup(nir_builder *b)
{
   return nir_channel(b, nir_load_base_workgroup_id(b, 32), 0);
}

static unsigned
attribute_slots(const struct glsl_type *type, bool bindless)
{
   return glsl_count_attribute_slots(type, false);
}

/* Converts in/out variables to IO intrinsics indexed by API location. */
static void
lower_stage_io(nir_shader *nir)
{
   NIR_PASS(_, nir, nir_lower_io_vars_to_temporaries, nir_shader_get_entrypoint(nir),
            nir_var_shader_in | nir_var_shader_out);
   NIR_PASS(_, nir, nir_lower_global_vars_to_local);
   NIR_PASS(_, nir, nir_split_var_copies);
   NIR_PASS(_, nir, nir_lower_var_copies);
   NIR_PASS(_, nir, nir_lower_vars_to_ssa);
   NIR_PASS(_, nir, nir_lower_indirect_derefs_to_if_else_trees, nir_var_shader_in | nir_var_shader_out,
            UINT32_MAX);
   nir_foreach_variable_with_modes(var, nir, nir_var_shader_in | nir_var_shader_out) {
      var->data.driver_location = var->data.location;
      if (nir->info.stage == MESA_SHADER_VERTEX && var->data.mode == nir_var_shader_in)
         var->data.driver_location -= VERT_ATTRIB_GENERIC0;
   }
   NIR_PASS(_, nir, nir_lower_io, nir_var_shader_in | nir_var_shader_out, attribute_slots,
            nir_lower_io_use_interpolated_input_intrinsics);
   NIR_PASS(_, nir, nir_opt_constant_folding);
   NIR_PASS(_, nir, nir_lower_system_values);
   nir_shader_gather_info(nir, nir_shader_get_entrypoint(nir));
}

/* Makes a lowered stage an ordinary native launch of `invocations` lanes. */
static void
convert_to_launch(nir_shader *nir, unsigned invocations)
{
   nir_remove_dead_variables(nir, nir_var_shader_in | nir_var_shader_out, NULL);
   nir->info.stage = MESA_SHADER_COMPUTE;
   memset(&nir->info.cs, 0, sizeof(nir->info.cs));
   nir->info.workgroup_size[0] = invocations;
   nir->info.workgroup_size[1] = 1;
   nir->info.workgroup_size[2] = 1;
   nir->info.workgroup_size_variable = false;
}

/* ---- Vertex stage -------------------------------------------------------- */

struct vertex_lowering {
   struct apex_shader *shader;
   const struct vk_vertex_input_state *vi;
   nir_def *vertex, *instance, *index, *record;
   bool invalid;
};

/* One vertex channel of a plain format, as FP32 or integer bits. */
static nir_def *
fetch_channel(nir_builder *b, const struct util_format_description *desc, unsigned c,
              nir_def *address, nir_def *byte)
{
   const struct util_format_channel_description *ch = &desc->channel[c];
   unsigned bits = ch->size, shift = ch->shift;
   /* Load the aligned word(s) covering the channel. */
   nir_def *first = nir_iadd_imm(b, byte, shift / 8);
   nir_def *aligned = nir_iand_imm(b, first, ~3u);
   nir_def *offset = nir_ishl_imm(b, nir_iand_imm(b, first, 3), 3);
   nir_def *lo = load_word(b, address_add(b, address, aligned));
   nir_def *value;
   if (bits == 32) {
      /* 32-bit channels are word aligned. */
      value = lo;
   } else {
      nir_def *hi = load_word(b, address_add(b, address, nir_iadd_imm(b, aligned, 4)));
      nir_def *joined = nir_bcsel(b, nir_ieq_imm(b, offset, 0), lo,
         nir_ior(b, nir_ushr(b, lo, offset), nir_ishl(b, hi, nir_isub(b, nir_imm_int(b, 32), offset))));
      value = nir_iand_imm(b, nir_ushr_imm(b, joined, shift % 8), BITFIELD_MASK(bits));
   }
   switch (ch->type) {
   case UTIL_FORMAT_TYPE_FLOAT:
      /* The vertex format table admits only 32-bit float channels. */
      return value;
   case UTIL_FORMAT_TYPE_UNSIGNED:
      if (ch->normalized)
         return nir_fmul_imm(b, nir_u2f32(b, value), 1.0 / BITFIELD_MASK(bits));
      return desc->channel[c].pure_integer ? value : nir_u2f32(b, value);
   case UTIL_FORMAT_TYPE_SIGNED: {
      nir_def *extended = bits == 32 ? value :
         nir_ishr_imm(b, nir_ishl_imm(b, value, 32 - bits), 32 - bits);
      if (ch->normalized)
         return nir_fmax(b, nir_fmul_imm(b, nir_i2f32(b, extended), 1.0 / BITFIELD_MASK(bits - 1)),
                         nir_imm_float(b, -1.0f));
      return desc->channel[c].pure_integer ? extended : nir_i2f32(b, extended);
   }
   default:
      return nir_imm_int(b, 0);
   }
}

static nir_def *
fetch_attribute(nir_builder *b, struct vertex_lowering *ctx, unsigned location,
                unsigned component, unsigned count, bool integer)
{
   const struct vk_vertex_attribute_state *attr = &ctx->vi->attributes[location];
   const struct vk_vertex_binding_state *binding = &ctx->vi->bindings[attr->binding];
   const struct apex_program *program = &ctx->shader->program;
   enum pipe_format pformat = vk_format_to_pipe_format(attr->format);
   const struct util_format_description *desc = util_format_description(pformat);
   if (!(ctx->vi->attributes_valid & BITFIELD_BIT(location)) || !desc ||
       desc->layout != UTIL_FORMAT_LAYOUT_PLAIN || desc->block.bits > 128) {
      ctx->invalid = true;
      return nir_imm_zero(b, count, 32);
   }
   unsigned slot = APEX_DRAW_BINDINGS + attr->binding * APEX_DRAW_BINDING_WORDS;
   /* Instance-rate elements: firstInstance + (InstanceIndex - firstInstance) / divisor;
    * a zero divisor repeats firstInstance. */
   nir_def *first_instance = draw_word(b, program, APEX_DRAW_FIRST_INSTANCE);
   nir_def *element = ctx->vertex;
   if (binding->input_rate == VK_VERTEX_INPUT_RATE_INSTANCE) {
      element = binding->divisor == 1 ? ctx->instance :
                binding->divisor == 0 ? first_instance :
                nir_iadd(b, first_instance,
                         nir_udiv_imm(b, nir_isub(b, ctx->instance, first_instance), binding->divisor));
   }
   nir_def *byte = nir_iadd_imm(b, nir_imul(b, element, draw_word(b, program, slot + 3)), attr->offset);
   /* Robust fetch: an attribute beyond the bound range reads zero. */
   nir_def *inside = nir_uge(b, draw_word(b, program, slot + 2),
                             nir_iadd_imm(b, byte, desc->block.bits / 8));
   nir_def *address = draw_address(b, program, slot);
   nir_def *values[4], *fallback[4];
   /* Phi sources must exist before the if: phis lead their block. */
   for (unsigned c = 0; c < count; c++)
      fallback[c] = component + c == 3 ? (integer ? nir_imm_int(b, 1) : nir_imm_float(b, 1.0f)) :
                                         nir_imm_int(b, 0);
   nir_push_if(b, inside);
   nir_def *fetched[4];
   for (unsigned c = 0; c < 4; c++) {
      unsigned swizzle = desc->swizzle[c];
      if (swizzle <= PIPE_SWIZZLE_W)
         fetched[c] = fetch_channel(b, desc, swizzle, address, byte);
      else if (swizzle == PIPE_SWIZZLE_1)
         fetched[c] = integer ? nir_imm_int(b, 1) : nir_imm_float(b, 1.0f);
      else
         fetched[c] = nir_imm_int(b, 0);
   }
   nir_pop_if(b, NULL);
   for (unsigned c = 0; c < count; c++)
      values[c] = nir_if_phi(b, fetched[component + c], fallback[c]);
   return nir_vec(b, values, count);
}

static bool
lower_vertex_intrinsic(nir_builder *b, nir_intrinsic_instr *i, void *data)
{
   struct vertex_lowering *ctx = data;
   const struct apex_program *program = &ctx->shader->program;
   b->cursor = nir_before_instr(&i->instr);
   nir_def *replacement = NULL;
   switch (i->intrinsic) {
   case nir_intrinsic_load_input: {
      if (i->def.bit_size != 32) {
         ctx->invalid = true;
         return false;
      }
      unsigned location = nir_intrinsic_io_semantics(i).location - VERT_ATTRIB_GENERIC0;
      nir_alu_type type = nir_intrinsic_dest_type(i);
      replacement = fetch_attribute(b, ctx, location, nir_intrinsic_component(i),
                                    i->def.num_components,
                                    nir_alu_type_get_base_type(type) != nir_type_float);
      break;
   }
   case nir_intrinsic_store_output: {
      unsigned location = nir_intrinsic_io_semantics(i).location;
      int slot = ctx->shader->vertex.slot[location];
      if (slot < 0) {
         /* Outputs without a consumer-visible slot are discarded. */
         nir_instr_remove(&i->instr);
         return true;
      }
      nir_def *value = i->src[0].ssa;
      unsigned component = nir_intrinsic_component(i);
      for (unsigned c = 0; c < value->num_components; c++) {
         if (!(nir_intrinsic_write_mask(i) & BITFIELD_BIT(c)))
            continue;
         nir_def *word = nir_channel(b, value, c);
         if (word->bit_size != 32) {
            ctx->invalid = true;
            return false;
         }
         store_word(b, word, address_add(b, ctx->record,
                                         nir_imm_int(b, (slot + component + c) * 4)));
      }
      nir_instr_remove(&i->instr);
      return true;
   }
   case nir_intrinsic_load_vertex_id:
      replacement = ctx->index;
      break;
   case nir_intrinsic_load_vertex_id_zero_base:
      replacement = nir_isub(b, ctx->index, draw_word(b, program, APEX_DRAW_FIRST_VERTEX));
      break;
   case nir_intrinsic_load_first_vertex:
   case nir_intrinsic_load_base_vertex:
      replacement = draw_word(b, program, i->intrinsic == nir_intrinsic_load_first_vertex ?
                              APEX_DRAW_FIRST_VERTEX : APEX_DRAW_INDEX + 3);
      break;
   case nir_intrinsic_load_instance_id:
      replacement = nir_isub(b, ctx->instance, draw_word(b, program, APEX_DRAW_FIRST_INSTANCE));
      break;
   case nir_intrinsic_load_base_instance:
      replacement = draw_word(b, program, APEX_DRAW_FIRST_INSTANCE);
      break;
   case nir_intrinsic_load_draw_id:
   case nir_intrinsic_load_view_index:
      replacement = nir_imm_int(b, 0);
      break;
   case nir_intrinsic_load_is_indexed_draw:
      replacement = nir_ine_imm(b, draw_word(b, program, APEX_DRAW_INDEX + 2), 0);
      break;
   default:
      return false;
   }
   nir_def_rewrite_uses(&i->def, replacement);
   nir_instr_remove(&i->instr);
   return true;
}

/* Record layout: position first, then written generic locations in order. */
static void
assign_vertex_slots(struct apex_shader *shader, const nir_shader *nir)
{
   memset(shader->vertex.slot, 0xff, sizeof(shader->vertex.slot));
   shader->vertex.slot[VARYING_SLOT_POS] = 0;
   unsigned words = 4;
   for (unsigned l = 0; l < 32; l++) {
      if (nir->info.outputs_written & BITFIELD64_BIT(VARYING_SLOT_VAR0 + l)) {
         shader->vertex.slot[VARYING_SLOT_VAR0 + l] = words;
         words += 4;
      }
   }
   if (nir->info.outputs_written & BITFIELD64_BIT(VARYING_SLOT_PSIZ))
      shader->vertex.slot[VARYING_SLOT_PSIZ] = words++;
   shader->vertex.stride = words;
}

/* Runs `lower` on each collected intrinsic; lowering may insert control flow. */
static bool
lower_collected(nir_shader *nir, bool (*lower)(nir_builder *, nir_intrinsic_instr *, void *),
                void *data)
{
   struct util_dynarray list;
   util_dynarray_init(&list, NULL);
   nir_foreach_function_impl(impl, nir) {
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type == nir_instr_type_intrinsic)
               util_dynarray_append(&list, nir_instr_as_intrinsic(instr));
         }
      }
   }
   bool progress = false;
   nir_builder b = nir_builder_create(nir_shader_get_entrypoint(nir));
   util_dynarray_foreach(&list, nir_intrinsic_instr *, i)
      progress |= lower(&b, *i, data);
   util_dynarray_fini(&list);
   nir_progress(progress, nir_shader_get_entrypoint(nir), nir_metadata_none);
   return progress;
}

static bool
build_vertex_kernel(struct apex_shader *shader, nir_shader *nir,
                    const struct vk_vertex_input_state *vi)
{
   assign_vertex_slots(shader, nir);
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_cf_list body;
   nir_cf_extract(&body, nir_before_impl(impl), nir_after_impl(impl));
   nir_builder builder = nir_builder_at(nir_before_impl(impl));
   nir_builder *b = &builder;
   const struct apex_program *program = &shader->program;
   struct vertex_lowering ctx = {.shader = shader, .vi = vi};
   /* Linear vertex over V x I: workgroups base + native launch, stepping by
    * the launch count below the job's end, so indirect jobs cover the draw. */
   unsigned trailer = apex_program_trailer(program);
   nir_def *per_instance = draw_word(b, program, APEX_DRAW_VERTEX_COUNT);
   nir_def *total = nir_imul(b, per_instance, draw_word(b, program, APEX_DRAW_INSTANCE_COUNT));
   nir_def *first_instance = draw_word(b, program, APEX_DRAW_FIRST_INSTANCE);
   nir_def *end = nir_umin(b, root_word(b, trailer + offsetof(struct apex_dispatch_parameters, groups) + 4),
                           nir_ushr_imm(b, nir_iadd_imm(b, total, 63), 6));
   nir_variable *w_var = nir_local_variable_create(impl, glsl_uint_type(), "vertex_group");
   nir_store_var(b, w_var, nir_iadd(b, root_word(b, trailer), native_workgroup(b)), 1);
   nir_loop *loop = nir_push_loop(b);
   nir_def *w = nir_load_var(b, w_var);
   nir_push_if(b, nir_uge(b, w, end));
   nir_jump(b, nir_jump_break);
   nir_pop_if(b, NULL);
   nir_def *linear = nir_iadd(b, nir_imul_imm(b, w, 64), nir_load_local_invocation_index(b));
   nir_def *instance = nir_udiv(b, linear, nir_umax(b, per_instance, nir_imm_int(b, 1)));
   nir_def *position = nir_isub(b, linear, nir_imul(b, instance, per_instance));
   ctx.instance = nir_iadd(b, instance, first_instance);
   nir_if *guard = nir_push_if(b, nir_ult(b, linear, total));
   /* Indexed draws read 8/16/32-bit indices; others count from firstVertex.
    * A restart index shades vertex 0 for a record no primitive reads. */
   nir_def *index_bytes = draw_word(b, program, APEX_DRAW_INDEX + 2);
   nir_push_if(b, nir_ine_imm(b, index_bytes, 0));
   nir_def *byte = nir_imul(b, nir_iadd(b, position, draw_word(b, program, APEX_DRAW_FIRST_VERTEX)),
                            index_bytes);
   nir_def *word = load_word(b, address_add(b, draw_address(b, program, APEX_DRAW_INDEX),
                                            nir_iand_imm(b, byte, ~3u)));
   nir_def *mask = nir_bcsel(b, nir_ieq_imm(b, index_bytes, 4), nir_imm_int(b, ~0),
                             nir_bcsel(b, nir_ieq_imm(b, index_bytes, 2), nir_imm_int(b, 0xffff),
                                       nir_imm_int(b, 0xff)));
   nir_def *index = nir_iand(b, nir_ushr(b, word, nir_ishl_imm(b, nir_iand_imm(b, byte, 3), 3)), mask);
   nir_def *restart = nir_iand(b, nir_ieq(b, index, mask),
                               nir_ine_imm(b, draw_word(b, program, APEX_DRAW_RESTART), 0));
   index = nir_iadd(b, nir_bcsel(b, restart, nir_imm_int(b, 0), index),
                    draw_word(b, program, APEX_DRAW_INDEX + 3));
   nir_push_else(b, NULL);
   nir_def *direct = nir_iadd(b, position, draw_word(b, program, APEX_DRAW_FIRST_VERTEX));
   nir_pop_if(b, NULL);
   ctx.index = nir_if_phi(b, index, direct);
   ctx.vertex = ctx.index;
   ctx.record = address_add(b, draw_address(b, program, APEX_DRAW_VERTEX_LO),
                            nir_imul_imm(b, linear, shader->vertex.stride * 4));
   nir_cf_reinsert(&body, b->cursor);
   b->cursor = nir_after_cf_list(&guard->then_list);
   nir_pop_if(b, guard);
   nir_store_var(b, w_var, nir_iadd(b, w, root_word(b, trailer + offsetof(struct apex_dispatch_parameters, groups))), 1);
   nir_pop_loop(b, loop);
   lower_collected(nir, lower_vertex_intrinsic, &ctx);
   return !ctx.invalid;
}

/* ---- Fragment stage ------------------------------------------------------ */

/* Lane l of a SIMD16 wave shades pixel (2 (l>>2 & 1) + (l & 1),
 * 2 (l >> 3) + (l>>1 & 1)) of a 4x4 tile: lanes 4q..4q+3 form 2x2 quad q. */
struct fragment_lowering {
   struct apex_shader *shader;
   nir_variable *perspective, *linear, *coord, *front, *covered, *killed, *sources, *primitive;
   nir_variable *color[APEX_DRAW_MAX_COLOR], *depth;
   bool invalid;
};

static nir_def *
attribute(nir_builder *b, struct fragment_lowering *ctx, nir_def *source, unsigned location,
          unsigned component)
{
   const struct apex_program *program = &ctx->shader->program;
   nir_def *slot = draw_word(b, program, APEX_DRAW_SLOTS + location - VARYING_SLOT_VAR0);
   nir_def *word = nir_iadd(b, nir_imul(b, source, draw_word(b, program, APEX_DRAW_VERTEX_STRIDE)),
                            nir_iadd_imm(b, slot, component));
   return load_word(b, address_add(b, draw_address(b, program, APEX_DRAW_VERTEX_LO),
                                   nir_imul_imm(b, word, 4)));
}

static nir_def *
lane_shuffle(nir_builder *b, nir_def *value, nir_def *lane)
{
   nir_def *channels[4];
   for (unsigned c = 0; c < value->num_components; c++)
      channels[c] = nir_shuffle(b, nir_channel(b, value, c), lane);
   return nir_vec(b, channels, value->num_components);
}

static bool
lower_fragment_intrinsic(nir_builder *b, nir_intrinsic_instr *i, void *data)
{
   struct fragment_lowering *ctx = data;
   b->cursor = nir_before_instr(&i->instr);
   nir_def *replacement = NULL;
   switch (i->intrinsic) {
   case nir_intrinsic_load_point_coord:
      replacement = nir_channels(b, nir_load_var(b, ctx->linear), 0x6);
      break;
   case nir_intrinsic_load_interpolated_input:
   case nir_intrinsic_load_input: {
      unsigned location = nir_intrinsic_io_semantics(i).location;
      unsigned component = nir_intrinsic_component(i);
      if (i->def.bit_size != 32) {
         ctx->invalid = true;
         return false;
      }
      nir_def *sources = nir_load_var(b, ctx->sources);
      if (location == VARYING_SLOT_PRIMITIVE_ID) {
         replacement = nir_load_var(b, ctx->primitive);
         break;
      }
      if (location == VARYING_SLOT_PNTC) {
         /* Point setup stores gl_PointCoord in the second and third weights. */
         nir_def *weights = nir_load_var(b, ctx->linear);
         nir_def *coord = nir_vec4(b, nir_channel(b, weights, 1), nir_channel(b, weights, 2),
                                   nir_imm_float(b, 0.0f), nir_imm_float(b, 1.0f));
         replacement = nir_channels(b, coord, BITFIELD_RANGE(component, i->def.num_components));
         break;
      }
      if (location < VARYING_SLOT_VAR0 || location >= VARYING_SLOT_VAR0 + 32) {
         replacement = nir_imm_zero(b, i->def.num_components, 32);
         break;
      }
      nir_def *values[4];
      if (i->intrinsic == nir_intrinsic_load_input) {
         /* Flat inputs take the provoking vertex, source 0. */
         for (unsigned c = 0; c < i->def.num_components; c++)
            values[c] = attribute(b, ctx, nir_channel(b, sources, 0), location, component + c);
      } else {
         nir_intrinsic_instr *bary = nir_src_as_intrinsic(i->src[0]);
         bool linear = bary && nir_intrinsic_interp_mode(bary) == INTERP_MODE_NOPERSPECTIVE;
         nir_def *weights = nir_load_var(b, linear ? ctx->linear : ctx->perspective);
         for (unsigned c = 0; c < i->def.num_components; c++) {
            nir_def *sum = NULL;
            for (unsigned j = 0; j < 3; j++) {
               nir_def *term = nir_fmul(b, nir_channel(b, weights, j),
                  attribute(b, ctx, nir_channel(b, sources, j), location, component + c));
               sum = sum ? nir_fadd(b, sum, term) : term;
            }
            values[c] = sum;
         }
      }
      replacement = nir_vec(b, values, i->def.num_components);
      break;
   }
   case nir_intrinsic_load_frag_coord:
      replacement = nir_load_var(b, ctx->coord);
      break;
   case nir_intrinsic_load_front_face:
      replacement = nir_load_var(b, ctx->front);
      if (i->def.bit_size == 32)
         replacement = nir_b2b32(b, replacement);
      break;
   case nir_intrinsic_load_helper_invocation:
   case nir_intrinsic_is_helper_invocation:
      replacement = nir_ior(b, nir_inot(b, nir_load_var(b, ctx->covered)), nir_load_var(b, ctx->killed));
      if (i->def.bit_size == 32)
         replacement = nir_b2b32(b, replacement);
      break;
   case nir_intrinsic_load_sample_id:
   case nir_intrinsic_load_layer_id:
   case nir_intrinsic_load_view_index:
      replacement = nir_imm_int(b, 0);
      break;
   case nir_intrinsic_load_sample_mask_in:
      replacement = nir_b2i32(b, nir_load_var(b, ctx->covered));
      break;
   case nir_intrinsic_load_sample_pos:
   case nir_intrinsic_load_sample_pos_or_center:
      replacement = nir_imm_vec2(b, 0.5f, 0.5f);
      break;
   case nir_intrinsic_load_barycentric_pixel:
   case nir_intrinsic_load_barycentric_centroid:
   case nir_intrinsic_load_barycentric_sample:
   case nir_intrinsic_load_barycentric_at_sample:
   case nir_intrinsic_load_barycentric_at_offset:
      /* Consumed by load_interpolated_input through its interpolation mode. */
      return false;
   case nir_intrinsic_store_output: {
      unsigned location = nir_intrinsic_io_semantics(i).location;
      nir_def *value = i->src[0].ssa;
      unsigned component = nir_intrinsic_component(i);
      nir_variable *var = location == FRAG_RESULT_DEPTH ? ctx->depth :
         location >= FRAG_RESULT_DATA0 && location < FRAG_RESULT_DATA0 + APEX_DRAW_MAX_COLOR ?
         ctx->color[location - FRAG_RESULT_DATA0] : NULL;
      if (var && value->bit_size == 32) {
         nir_def *old = nir_load_var(b, var);
         nir_def *channels[4];
         for (unsigned c = 0; c < old->num_components; c++) {
            unsigned source = c - component;
            channels[c] = c >= component && source < value->num_components &&
               (nir_intrinsic_write_mask(i) & BITFIELD_BIT(source)) ?
               nir_channel(b, value, source) : nir_channel(b, old, c);
         }
         nir_store_var(b, var, nir_vec(b, channels, old->num_components), BITFIELD_MASK(4));
      } else if (var) {
         ctx->invalid = true;
         return false;
      }
      nir_instr_remove(&i->instr);
      return true;
   }
   case nir_intrinsic_terminate:
   case nir_intrinsic_demote:
      nir_store_var(b, ctx->killed, nir_imm_true(b), 1);
      nir_instr_remove(&i->instr);
      return true;
   case nir_intrinsic_terminate_if:
   case nir_intrinsic_demote_if:
      nir_store_var(b, ctx->killed, nir_ior(b, nir_load_var(b, ctx->killed), i->src[0].ssa), 1);
      nir_instr_remove(&i->instr);
      return true;
   case nir_intrinsic_ddx: case nir_intrinsic_ddx_fine: case nir_intrinsic_ddx_coarse:
   case nir_intrinsic_ddy: case nir_intrinsic_ddy_fine: case nir_intrinsic_ddy_coarse: {
      bool x = i->intrinsic == nir_intrinsic_ddx || i->intrinsic == nir_intrinsic_ddx_fine ||
               i->intrinsic == nir_intrinsic_ddx_coarse;
      bool coarse = i->intrinsic == nir_intrinsic_ddx_coarse || i->intrinsic == nir_intrinsic_ddy_coarse;
      unsigned axis = x ? 1 : 2;
      nir_def *lane = nir_load_subgroup_invocation(b);
      /* Coarse derivatives use the quad's first row/column pair. */
      nir_def *base = nir_iand_imm(b, lane, coarse ? ~3u : ~axis);
      nir_def *value = i->src[0].ssa;
      replacement = nir_fsub(b, lane_shuffle(b, value, nir_ior_imm(b, base, axis)),
                             lane_shuffle(b, value, base));
      break;
   }
   default:
      return false;
   }
   nir_def_rewrite_uses(&i->def, replacement);
   nir_instr_remove(&i->instr);
   return true;
}

/* Texel values are 4 words; sub-word texels occupy the low bits. */
static unsigned
texel_words(enum pipe_format format)
{
   return MAX2(util_format_get_blocksizebits(format) / 32, 1);
}

static nir_def *
channel_bits(nir_builder *b, nir_def *words, const struct util_format_channel_description *ch)
{
   nir_def *word = nir_channel(b, words, ch->shift / 32);
   return ch->size == 32 ? word : nir_iand_imm(b, nir_ushr_imm(b, word, ch->shift % 32),
                                               BITFIELD_MASK(ch->size));
}

/* Texel words to API components as FP32 (normalized and float formats). */
static nir_def *
unpack_color(nir_builder *b, enum pipe_format format, nir_def *words)
{
   const struct util_format_description *desc = util_format_description(format);
   nir_def *channels[4];
   for (unsigned s = 0; s < 4; s++) {
      unsigned swizzle = desc->swizzle[s];
      if (swizzle > PIPE_SWIZZLE_W) {
         channels[s] = nir_imm_float(b, swizzle == PIPE_SWIZZLE_1 ? 1.0f : 0.0f);
         continue;
      }
      const struct util_format_channel_description *ch = &desc->channel[swizzle];
      nir_def *raw = channel_bits(b, words, ch);
      if (ch->type == UTIL_FORMAT_TYPE_FLOAT) {
         channels[s] = ch->size == 16 ? apex_half_to_float(b, raw) : raw;
      } else if (ch->type == UTIL_FORMAT_TYPE_UNSIGNED) {
         channels[s] = nir_fmul_imm(b, nir_u2f32(b, raw), 1.0 / u_uintN_max(ch->size));
      } else {
         nir_def *value = nir_ishr_imm(b, nir_ishl_imm(b, raw, 32 - ch->size), 32 - ch->size);
         channels[s] = nir_fmax(b, nir_fmul_imm(b, nir_i2f32(b, value), 1.0 / u_intN_max(ch->size)),
                                nir_imm_float(b, -1.0f));
      }
   }
   nir_def *color = nir_vec(b, channels, 4);
   if (desc->colorspace == UTIL_FORMAT_COLORSPACE_SRGB)
      color = nir_vector_insert_imm(b, nir_format_srgb_to_linear(b, color), channels[3], 3);
   return color;
}

/* API components to texel words: normalized channels round to nearest even,
 * integers clamp, FP16 rounds to nearest even. */
static nir_def *
pack_color(nir_builder *b, enum pipe_format format, nir_def *color)
{
   const struct util_format_description *desc = util_format_description(format);
   if (desc->colorspace == UTIL_FORMAT_COLORSPACE_SRGB)
      color = nir_vector_insert_imm(b, nir_format_linear_to_srgb(b, color), nir_channel(b, color, 3), 3);
   nir_def *words[4];
   for (unsigned w = 0; w < 4; w++)
      words[w] = nir_imm_int(b, 0);
   for (unsigned c = 0; c < desc->nr_channels; c++) {
      const struct util_format_channel_description *ch = &desc->channel[c];
      unsigned s = 0;
      while (s < 4 && desc->swizzle[s] != c)
         s++;
      if (s == 4 || ch->type == UTIL_FORMAT_TYPE_VOID)
         continue;
      nir_def *v = nir_channel(b, color, s);
      const unsigned bits[1] = {ch->size};
      if (ch->type == UTIL_FORMAT_TYPE_FLOAT)
         v = ch->size == 16 ? apex_float_to_half(b, v) : v;
      else if (ch->normalized)
         v = ch->type == UTIL_FORMAT_TYPE_UNSIGNED ? nir_format_float_to_unorm(b, v, bits) :
                                                     nir_format_float_to_snorm(b, v, bits);
      else
         v = ch->type == UTIL_FORMAT_TYPE_UNSIGNED ? nir_format_clamp_uint(b, v, bits) :
                                                     nir_format_clamp_sint(b, v, bits);
      if (ch->size < 32)
         v = nir_iand_imm(b, v, BITFIELD_MASK(ch->size));
      unsigned w = ch->shift / 32;
      words[w] = nir_ior(b, words[w], nir_ishl_imm(b, v, ch->shift % 32));
   }
   return nir_vec(b, words, 4);
}

/* Lane l shades pixel x = (l & 1) | (l >> 1 & 2) of its quad row, so the
 * pixels sharing a word with l's pixel are lanes l ^ 1 (16-bit texels) and
 * also l ^ 4 and l ^ 5 (8-bit texels). */
struct texel_target {
   unsigned bits;
   nir_def *address;          /* word-aligned address of the texel's first word */
   nir_def *shift;            /* bit offset of a sub-word texel */
};

static struct texel_target
texel_target(nir_builder *b, nir_def *base, nir_def *pitch, nir_def *px, nir_def *py,
             enum pipe_format format)
{
   struct texel_target t = {.bits = util_format_get_blocksizebits(format)};
   nir_def *byte = nir_iadd(b, nir_imul(b, py, pitch), nir_imul_imm(b, px, t.bits / 8));
   if (t.bits < 32) {
      t.shift = nir_ishl_imm(b, nir_iand_imm(b, byte, 3), 3);
      byte = nir_iand_imm(b, byte, ~3u);
   }
   t.address = address_add(b, base, byte);
   return t;
}

static nir_def *
load_texel(nir_builder *b, const struct texel_target *t)
{
   nir_def *w[4];
   for (unsigned i = 0; i < 4; i++)
      w[i] = i < MAX2(t->bits / 32, 1) ? load_word(b, address_add(b, t->address, nir_imm_int(b, i * 4))) :
                                         nir_imm_int(b, 0);
   if (t->bits < 32)
      w[0] = nir_iand_imm(b, nir_ushr(b, w[0], t->shift), BITFIELD_MASK(t->bits));
   return nir_vec(b, w, 4);
}

/* Runs with the whole wave active. A sub-word texel's word gathers every
 * pixel sharing it, unchanged ones included, and its first pixel's lane
 * stores it when any of them is in `store`. */
static void
store_texel(nir_builder *b, const struct texel_target *t, nir_def *texel, nir_def *lane,
            nir_def *store)
{
   if (t->bits >= 32) {
      nir_push_if(b, store);
      for (unsigned i = 0; i < t->bits / 32; i++)
         store_word(b, nir_channel(b, texel, i), address_add(b, t->address, nir_imm_int(b, i * 4)));
      nir_pop_if(b, NULL);
      return;
   }
   static const unsigned partners[] = {1, 4, 5};
   nir_def *word = nir_ishl(b, nir_channel(b, texel, 0), t->shift);
   nir_def *any = nir_b2i32(b, store), *combined = word;
   for (unsigned p = 0; p < (t->bits == 16 ? 1 : 3); p++) {
      nir_def *other = nir_ixor(b, lane, nir_imm_int(b, partners[p]));
      combined = nir_ior(b, combined, nir_shuffle(b, word, other));
      any = nir_ior(b, any, nir_shuffle(b, nir_b2i32(b, store), other));
   }
   nir_push_if(b, nir_iand(b, nir_ieq_imm(b, t->shift, 0), nir_ine_imm(b, any, 0)));
   store_word(b, combined, t->address);
   nir_pop_if(b, NULL);
}

/* VkCompareOp as a bit set: bit 0 less, bit 1 equal, bit 2 greater. */
static nir_def *
compare_op(nir_builder *b, nir_def *op, nir_def *less, nir_def *equal, nir_def *greater)
{
   return nir_ior(b, nir_iand(b, nir_ine_imm(b, nir_iand_imm(b, op, 1), 0), less),
                  nir_ior(b, nir_iand(b, nir_ine_imm(b, nir_iand_imm(b, op, 2), 0), equal),
                             nir_iand(b, nir_ine_imm(b, nir_iand_imm(b, op, 4), 0), greater)));
}

/* VkStencilOp on an 8-bit stencil value. */
static nir_def *
stencil_op(nir_builder *b, nir_def *op, nir_def *s, nir_def *reference)
{
   nir_def *up = nir_iadd_imm(b, s, 1), *down = nir_iadd_imm(b, s, -1);
   nir_def *table[] = {
      s, nir_imm_int(b, 0), reference, nir_umin(b, up, nir_imm_int(b, 255)),
      nir_bcsel(b, nir_ieq_imm(b, s, 0), s, down), nir_ixor(b, s, nir_imm_int(b, 0xff)),
      nir_iand_imm(b, up, 0xff), nir_iand_imm(b, down, 0xff),
   };
   nir_def *result = table[0];
   for (unsigned i = 1; i < ARRAY_SIZE(table); i++)
      result = nir_bcsel(b, nir_ieq_imm(b, op, i), table[i], result);
   return result;
}

/* One VkBlendFactor for channel c. Dual-source factors read as zero. */
static nir_def *
blend_factor(nir_builder *b, nir_def *factor, nir_def *src, nir_def *dst, nir_def *constant,
             unsigned c)
{
   nir_def *one = nir_imm_float(b, 1.0f);
   nir_def *src_a = nir_channel(b, src, 3), *dst_a = nir_channel(b, dst, 3);
   nir_def *table[] = {
      nir_imm_float(b, 0.0f), one,
      nir_channel(b, src, c), nir_fsub(b, one, nir_channel(b, src, c)),
      nir_channel(b, dst, c), nir_fsub(b, one, nir_channel(b, dst, c)),
      src_a, nir_fsub(b, one, src_a), dst_a, nir_fsub(b, one, dst_a),
      nir_channel(b, constant, c), nir_fsub(b, one, nir_channel(b, constant, c)),
      nir_channel(b, constant, 3), nir_fsub(b, one, nir_channel(b, constant, 3)),
      c == 3 ? one : nir_fmin(b, src_a, nir_fsub(b, one, dst_a)),
   };
   nir_def *result = table[0];
   for (unsigned f = 1; f < ARRAY_SIZE(table); f++)
      result = nir_bcsel(b, nir_ieq_imm(b, factor, f), table[f], result);
   return result;
}

/* VkBlendOp ADD, SUBTRACT, REVERSE_SUBTRACT, MIN, MAX. */
static nir_def *
blend_op(nir_builder *b, nir_def *op, nir_def *s, nir_def *d, nir_def *sf, nir_def *df)
{
   nir_def *ss = nir_fmul(b, s, sf), *dd = nir_fmul(b, d, df);
   nir_def *result = nir_fadd(b, ss, dd);
   result = nir_bcsel(b, nir_ieq_imm(b, op, 1), nir_fsub(b, ss, dd), result);
   result = nir_bcsel(b, nir_ieq_imm(b, op, 2), nir_fsub(b, dd, ss), result);
   result = nir_bcsel(b, nir_ieq_imm(b, op, 3), nir_fmin(b, s, d), result);
   return nir_bcsel(b, nir_ieq_imm(b, op, 4), nir_fmax(b, s, d), result);
}

/* Bits of packed color storage written by API channel mask `mask`. */
static nir_def *
write_bits(nir_builder *b, enum pipe_format format, nir_def *mask, unsigned word)
{
   const struct util_format_description *desc = util_format_description(format);
   nir_def *bits = nir_imm_int(b, 0);
   for (unsigned s = 0; s < 4; s++) {
      unsigned channel = desc->swizzle[s];
      if (channel > PIPE_SWIZZLE_W)
         continue;
      const struct util_format_channel_description *ch = &desc->channel[channel];
      unsigned first = ch->shift, last = ch->shift + ch->size;
      if (last <= word * 32 || first >= word * 32 + 32)
         continue;
      uint32_t field = ch->size == 32 ? ~0u : BITFIELD_MASK(ch->size) << (first - word * 32);
      bits = nir_bcsel(b, nir_ine_imm(b, nir_iand_imm(b, mask, 1u << s), 0),
                       nir_ior_imm(b, bits, field), bits);
   }
   return bits;
}

static nir_def *
loop_counter(nir_builder *b, nir_variable *var, nir_def *limit)
{
   nir_def *value = nir_load_var(b, var);
   nir_push_if(b, nir_uge(b, value, limit));
   nir_jump(b, nir_jump_break);
   nir_pop_if(b, NULL);
   return value;
}

static bool
build_fragment_kernel(struct apex_shader *shader, nir_shader *nir,
                      const struct vk_graphics_pipeline_state *state)
{
   const struct vk_render_pass_state *rp = state ? state->rp : NULL;
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   const struct apex_program *program = &shader->program;
   struct fragment_lowering ctx = {.shader = shader};
   const struct glsl_type *vec4 = glsl_vec4_type(), *vec3 = glsl_vec_type(3);
   ctx.perspective = nir_local_variable_create(impl, vec3, "perspective");
   ctx.linear = nir_local_variable_create(impl, vec3, "linear");
   ctx.coord = nir_local_variable_create(impl, vec4, "coord");
   ctx.front = nir_local_variable_create(impl, glsl_bool_type(), "front");
   ctx.covered = nir_local_variable_create(impl, glsl_bool_type(), "covered");
   ctx.killed = nir_local_variable_create(impl, glsl_bool_type(), "killed");
   ctx.sources = nir_local_variable_create(impl, glsl_uvec_type(3), "sources");
   ctx.primitive = nir_local_variable_create(impl, glsl_uint_type(), "primitive");
   ctx.depth = nir_local_variable_create(impl, glsl_float_type(), "depth");
   for (unsigned k = 0; k < APEX_DRAW_MAX_COLOR; k++)
      ctx.color[k] = nir_local_variable_create(impl, vec4, "color");
   lower_collected(nir, lower_fragment_intrinsic, &ctx);
   if (ctx.invalid)
      return false;

   nir_cf_list body;
   nir_cf_extract(&body, nir_before_impl(impl), nir_after_impl(impl));
   nir_builder builder = nir_builder_at(nir_before_impl(impl));
   nir_builder *b = &builder;

   /* Attachments the fragment shader writes, with their storage formats. */
   enum pipe_format formats[APEX_DRAW_MAX_COLOR] = {0};
   nir_variable *stored[APEX_DRAW_MAX_COLOR] = {0};
   for (unsigned k = 0; rp && k < rp->color_attachment_count && k < APEX_DRAW_MAX_COLOR; k++) {
      if (!(nir->info.outputs_written & BITFIELD64_BIT(FRAG_RESULT_DATA0 + k)) ||
          rp->color_attachment_formats[k] == VK_FORMAT_UNDEFINED)
         continue;
      formats[k] = vk_format_to_pipe_format(rp->color_attachment_formats[k]);
      if (!apex_attachment_format_supported(formats[k]))
         return false;
      stored[k] = nir_local_variable_create(impl, glsl_uvec4_type(), "stored");
   }
   /* One depth/stencil texel: Z is swizzle 0 and S swizzle 1 of its format. */
   VkFormat ds_vk = !rp ? VK_FORMAT_UNDEFINED : rp->depth_attachment_format != VK_FORMAT_UNDEFINED ?
                    rp->depth_attachment_format : rp->stencil_attachment_format;
   enum pipe_format ds_format = vk_format_to_pipe_format(ds_vk);
   const struct util_format_channel_description *z_channel = NULL, *s_channel = NULL;
   if (ds_vk != VK_FORMAT_UNDEFINED) {
      const struct util_format_description *desc = util_format_description(ds_format);
      if (rp->depth_attachment_format != VK_FORMAT_UNDEFINED && desc->swizzle[0] <= PIPE_SWIZZLE_W)
         z_channel = &desc->channel[desc->swizzle[0]];
      if (rp->stencil_attachment_format != VK_FORMAT_UNDEFINED && desc->swizzle[1] <= PIPE_SWIZZLE_W)
         s_channel = &desc->channel[desc->swizzle[1]];
      if ((z_channel && !(z_channel->type == UTIL_FORMAT_TYPE_FLOAT ? z_channel->size == 32 :
                          z_channel->normalized && z_channel->size <= 24)) ||
          (s_channel && s_channel->size != 8))
         return false;
   }
   bool ds_target = z_channel || s_channel;
   nir_variable *stored_ds = nir_local_variable_create(impl, glsl_uvec4_type(), "stored_ds");

   nir_def *lane = nir_load_subgroup_invocation(b);
   nir_def *lx = nir_ior(b, nir_iand_imm(b, lane, 1), nir_iand_imm(b, nir_ushr_imm(b, lane, 1), 2));
   nir_def *ly = nir_ior(b, nir_iand_imm(b, nir_ushr_imm(b, lane, 1), 1),
                         nir_iand_imm(b, nir_ushr_imm(b, lane, 2), 2));
   unsigned trailer = apex_program_trailer(program);
   nir_variable *t_var = nir_local_variable_create(impl, glsl_uint_type(), "tile");
   nir_store_var(b, t_var, nir_iadd(b, root_word(b, trailer), native_workgroup(b)), 1);
   nir_loop *tile_loop = nir_push_loop(b);
   nir_def *x0 = draw_word(b, program, APEX_DRAW_SCISSOR);
   nir_def *y0 = draw_word(b, program, APEX_DRAW_SCISSOR + 1);
   nir_def *x1 = draw_word(b, program, APEX_DRAW_SCISSOR + 2);
   nir_def *y1 = draw_word(b, program, APEX_DRAW_SCISSOR + 3);
   nir_def *tx0 = nir_ushr_imm(b, x0, 2), *ty0 = nir_ushr_imm(b, y0, 2);
   nir_def *tw = nir_isub(b, nir_ushr_imm(b, nir_iadd_imm(b, x1, 3), 2), tx0);
   nir_def *th = nir_isub(b, nir_ushr_imm(b, nir_iadd_imm(b, y1, 3), 2), ty0);
   /* A job whose chunk range starts past the draw's chunks has no work. */
   nir_def *empty = nir_ior(b, nir_ior(b, nir_uge(b, x0, x1), nir_uge(b, y0, y1)),
                            nir_uge(b, draw_word(b, program, APEX_DRAW_CHUNK_RANGE),
                                    draw_word(b, program, APEX_DRAW_BIN_CHUNKS)));
   /* A job covers tiles [base, end) with a stride of its workgroup count. */
   nir_def *end = root_word(b, trailer + offsetof(struct apex_dispatch_parameters, groups) + 4);
   nir_def *tiles = nir_bcsel(b, empty, nir_imm_int(b, 0), nir_umin(b, nir_imul(b, tw, th), end));
   nir_def *t = loop_counter(b, t_var, tiles);
   nir_def *ty = nir_udiv(b, t, tw);
   nir_def *tx = nir_isub(b, t, nir_imul(b, ty, tw));
   nir_def *tile_x = nir_ishl_imm(b, nir_iadd(b, tx0, tx), 2);
   nir_def *tile_y = nir_ishl_imm(b, nir_iadd(b, ty0, ty), 2);
   nir_def *px = nir_iadd(b, tile_x, lx), *py = nir_iadd(b, tile_y, ly);
   /* Rows of the scissor stay inside every attachment; each row is padded to
    * whole tiles, so a tile's texels there may be read and rewritten. */
   nir_def *row = nir_iand(b, nir_uge(b, py, y0), nir_ult(b, py, y1));
   nir_def *inside = nir_iand(b, row, nir_iand(b, nir_uge(b, px, x0), nir_ult(b, px, x1)));

   /* Current attachment contents of this lane's pixel. */
   struct texel_target color_target[APEX_DRAW_MAX_COLOR], ds;
   for (unsigned k = 0; k < APEX_DRAW_MAX_COLOR; k++) {
      if (!stored[k])
         continue;
      unsigned slot = APEX_DRAW_COLOR + k * APEX_DRAW_COLOR_WORDS;
      color_target[k] = texel_target(b, draw_address(b, program, slot), draw_word(b, program, slot + 2),
                                     px, py, formats[k]);
   }
   if (ds_target)
      ds = texel_target(b, draw_address(b, program, APEX_DRAW_DEPTH_TARGET),
                        draw_word(b, program, APEX_DRAW_DEPTH_TARGET + 2), px, py, ds_format);
   nir_def *zero4 = nir_imm_zero(b, 4, 32);
   nir_push_if(b, row);
   nir_def *loaded[APEX_DRAW_MAX_COLOR] = {0};
   for (unsigned k = 0; k < APEX_DRAW_MAX_COLOR; k++)
      if (stored[k])
         loaded[k] = load_texel(b, &color_target[k]);
   nir_def *loaded_ds = ds_target ? load_texel(b, &ds) : NULL;
   nir_pop_if(b, NULL);
   /* Phis first, then the stores that consume them. */
   for (unsigned k = 0; k < APEX_DRAW_MAX_COLOR; k++)
      if (stored[k])
         loaded[k] = nir_if_phi(b, loaded[k], zero4);
   if (ds_target)
      loaded_ds = nir_if_phi(b, loaded_ds, zero4);
   for (unsigned k = 0; k < APEX_DRAW_MAX_COLOR; k++)
      if (stored[k])
         nir_store_var(b, stored[k], loaded[k], 0xf);
   if (ds_target)
      nir_store_var(b, stored_ds, loaded_ds, 0xf);

   /* The tile lies in one bin; walk that bin's ordered list segments. */
   nir_def *shift = draw_word(b, program, APEX_DRAW_BIN_SHIFT);
   nir_def *columns = draw_word(b, program, APEX_DRAW_BIN_COLUMNS);
   nir_def *bin = nir_iadd(b, nir_imul(b, nir_isub(b, nir_ushr(b, tile_y, shift),
                                                    draw_word(b, program, APEX_DRAW_BIN_Y0)), columns),
                           nir_isub(b, nir_ushr(b, tile_x, shift), draw_word(b, program, APEX_DRAW_BIN_X0)));
   nir_variable *c_var = nir_local_variable_create(impl, glsl_uint_type(), "chunk");
   nir_store_var(b, c_var, draw_word(b, program, APEX_DRAW_CHUNK_RANGE), 1);
   nir_loop *chunk_loop = nir_push_loop(b);
   nir_def *chunks = draw_word(b, program, APEX_DRAW_BIN_CHUNKS);
   nir_def *chunk = loop_counter(b, c_var, nir_umin(b, chunks,
                                                    draw_word(b, program, APEX_DRAW_CHUNK_RANGE + 1)));
   nir_def *entries = load_word(b, address_add(b, draw_address(b, program, APEX_DRAW_BIN_COUNTS),
      nir_ishl_imm(b, nir_iadd(b, nir_imul(b, bin, chunks), chunk), 2)));
   nir_def *segment = nir_iadd(b, nir_imul(b, bin, nir_imul(b, draw_word(b, program, APEX_DRAW_PRIM_COUNT),
                                                           draw_word(b, program, APEX_DRAW_INSTANCE_COUNT))),
                               nir_imul_imm(b, chunk, APEX_BIN_CHUNK));
   nir_variable *p_var = nir_local_variable_create(impl, glsl_uint_type(), "entry");
   nir_store_var(b, p_var, nir_imm_int(b, 0), 1);
   nir_loop *prim_loop = nir_push_loop(b);
   nir_def *entry = loop_counter(b, p_var, entries);
   nir_def *p = load_word(b, address_add(b, draw_address(b, program, APEX_DRAW_BIN_LISTS),
                                         nir_ishl_imm(b, nir_iadd(b, segment, entry), 2)));
   nir_def *first = address_add(b, draw_address(b, program, APEX_DRAW_PRIM_LO),
                                nir_imul(b, p, nir_bcsel(b, nir_ule_imm(b, draw_word(b, program, APEX_DRAW_TOPOLOGY), 2),
                                                         nir_imm_int(b, APEX_SUBPRIMS_FOR(0) * APEX_PRIM_WORDS * 4),
                                                         nir_imm_int(b, APEX_SUBPRIMS_FOR(3) * APEX_PRIM_WORDS * 4))));
   nir_def *count = load_word(b, address_add(b, first, nir_imm_int(b, APEX_PRIM_COUNT * 4)));
   nir_variable *s_var = nir_local_variable_create(impl, glsl_uint_type(), "subprimitive");
   nir_store_var(b, s_var, nir_imm_int(b, 0), 1);
   nir_loop *sub_loop = nir_push_loop(b);
   nir_def *s = loop_counter(b, s_var, count);
   nir_def *record = address_add(b, first, nir_imul_imm(b, s, APEX_PRIM_WORDS * 4));
#define FIELD(word) load_word(b, address_add(b, record, nir_imm_int(b, (word) * 4)))
   nir_def *overlap = nir_iand(b,
      nir_iand(b, nir_ult(b, FIELD(APEX_PRIM_BOX), nir_iadd_imm(b, tile_x, 4)),
                  nir_ult(b, tile_x, FIELD(APEX_PRIM_BOX + 2))),
      nir_iand(b, nir_ult(b, FIELD(APEX_PRIM_BOX + 1), nir_iadd_imm(b, tile_y, 4)),
                  nir_ult(b, tile_y, FIELD(APEX_PRIM_BOX + 3))));
   nir_push_if(b, overlap);
   nir_def *qx = nir_i2i64(b, nir_iadd_imm(b, nir_ishl_imm(b, px, 8), 128));
   nir_def *qy = nir_i2i64(b, nir_iadd_imm(b, nir_ishl_imm(b, py, 8), 128));
   nir_def *edge[3];
   nir_def *covered = inside;
   for (unsigned e = 0; e < 3; e++) {
      unsigned w = APEX_PRIM_EDGE + e * 4;
      nir_def *c = nir_pack_64_2x32_split(b, FIELD(w + 2), FIELD(w + 3));
      edge[e] = nir_iadd(b, nir_iadd(b, nir_imul(b, nir_i2i64(b, FIELD(w)), qx),
                                         nir_imul(b, nir_i2i64(b, FIELD(w + 1)), qy)), c);
      covered = nir_iand(b, covered, nir_ige_imm(b, edge[e], 0));
   }
   nir_push_if(b, nir_vote_any(b, 1, covered));
   nir_def *reciprocal = FIELD(APEX_PRIM_RECIPROCAL_AREA);
   nir_def *lambda[3], *weighted[3];
   /* z = z0 + l1 (z1 - z0) + l2 (z2 - z0) is exact for constant depth. */
   nir_def *z = FIELD(APEX_PRIM_Z), *sum = NULL;
   for (unsigned i = 0; i < 3; i++) {
      lambda[i] = nir_fmul(b, nir_i2f32(b, edge[i]), reciprocal);
      if (i)
         z = nir_ffma(b, lambda[i], nir_fsub(b, FIELD(APEX_PRIM_Z + i), FIELD(APEX_PRIM_Z)), z);
      weighted[i] = nir_fmul(b, lambda[i], FIELD(APEX_PRIM_INV_W + i));
      sum = sum ? nir_fadd(b, sum, weighted[i]) : weighted[i];
   }
   z = nir_fadd(b, z, FIELD(APEX_PRIM_DEPTH_OFFSET));
   nir_def *normalize = nir_frcp(b, sum);
   nir_def *perspective[3], *linear[3];
   for (unsigned j = 0; j < 3; j++) {
      perspective[j] = linear[j] = NULL;
      for (unsigned i = 0; i < 3; i++) {
         nir_def *m = FIELD(APEX_PRIM_WEIGHTS + i * 3 + j);
         nir_def *pw = nir_fmul(b, nir_fmul(b, weighted[i], normalize), m);
         nir_def *lw = nir_fmul(b, lambda[i], m);
         perspective[j] = perspective[j] ? nir_fadd(b, perspective[j], pw) : pw;
         linear[j] = linear[j] ? nir_fadd(b, linear[j], lw) : lw;
      }
   }
   nir_store_var(b, ctx.perspective, nir_vec(b, perspective, 3), 0x7);
   nir_store_var(b, ctx.linear, nir_vec(b, linear, 3), 0x7);
   nir_store_var(b, ctx.coord, nir_vec4(b, nir_fadd_imm(b, nir_u2f32(b, px), 0.5f),
                                        nir_fadd_imm(b, nir_u2f32(b, py), 0.5f), z, sum), 0xf);
   nir_store_var(b, ctx.front, nir_ine_imm(b, nir_iand_imm(b, FIELD(APEX_PRIM_FLAGS), 1), 0), 1);
   nir_store_var(b, ctx.sources, nir_vec3(b, FIELD(APEX_PRIM_SOURCE), FIELD(APEX_PRIM_SOURCE + 1),
                                          FIELD(APEX_PRIM_SOURCE + 2)), 0x7);
   nir_store_var(b, ctx.primitive, FIELD(APEX_PRIM_ID), 1);
   nir_store_var(b, ctx.covered, covered, 1);
   nir_store_var(b, ctx.killed, nir_imm_false(b), 1);
   nir_store_var(b, ctx.depth, z, 1);
   for (unsigned k = 0; k < APEX_DRAW_MAX_COLOR; k++)
      nir_store_var(b, ctx.color[k], nir_imm_zero(b, 4, 32), 0xf);
   /* Whole quads execute the shader so helpers supply derivatives. */
   nir_def *quad = nir_b2i32(b, covered);
   for (unsigned d = 1; d < 4; d++)
      quad = nir_ior(b, quad, nir_shuffle(b, nir_b2i32(b, covered), nir_ixor(b, lane, nir_imm_int(b, d))));
   nir_if *shade = nir_push_if(b, nir_ine_imm(b, quad, 0));
   nir_cf_reinsert(&body, b->cursor);
   b->cursor = nir_after_cf_list(&shade->then_list);
   nir_pop_if(b, shade);

   nir_def *live = nir_iand(b, covered, nir_inot(b, nir_load_var(b, ctx.killed)));
   if (ds_target) {
      /* Stencil test, then depth test; stencil ops follow their outcome. */
      nir_def *control = draw_word(b, program, APEX_DRAW_DEPTH);
      nir_def *texel = nir_load_var(b, stored_ds);
      nir_def *words[4];
      for (unsigned w = 0; w < 4; w++)
         words[w] = nir_channel(b, texel, w);
      nir_def *depth_pass = nir_imm_true(b), *depth_write = nir_imm_false(b), *fragment = NULL;
      if (z_channel) {
         nir_def *z = nir_load_var(b, ctx.depth);
         nir_def *old = channel_bits(b, texel, z_channel);
         nir_def *op = nir_iand_imm(b, nir_ushr_imm(b, control, 4), 7);
         nir_def *pass;
         if (z_channel->type == UTIL_FORMAT_TYPE_FLOAT) {
            fragment = z;
            pass = compare_op(b, op, nir_flt(b, z, old), nir_feq(b, z, old), nir_flt(b, old, z));
         } else {
            const unsigned bits[1] = {z_channel->size};
            fragment = nir_format_float_to_unorm(b, z, bits);
            pass = compare_op(b, op, nir_ult(b, fragment, old), nir_ieq(b, fragment, old),
                              nir_ult(b, old, fragment));
         }
         nir_def *test = nir_ine_imm(b, nir_iand_imm(b, control, 1), 0);
         depth_pass = nir_ior(b, nir_inot(b, test), pass);
         depth_write = nir_iand(b, test, nir_ine_imm(b, nir_iand_imm(b, control, 2), 0));
      }
      nir_def *stencil_pass = nir_imm_true(b);
      if (s_channel) {
         nir_def *front = nir_load_var(b, ctx.front);
         nir_def *ops = nir_bcsel(b, front, draw_word(b, program, APEX_DRAW_STENCIL),
                                  draw_word(b, program, APEX_DRAW_STENCIL + 2));
         nir_def *masks = nir_bcsel(b, front, draw_word(b, program, APEX_DRAW_STENCIL + 1),
                                    draw_word(b, program, APEX_DRAW_STENCIL + 3));
         nir_def *compare_mask = nir_iand_imm(b, masks, 0xff);
         nir_def *write_mask = nir_iand_imm(b, nir_ushr_imm(b, masks, 8), 0xff);
         nir_def *reference = nir_iand_imm(b, nir_ushr_imm(b, masks, 16), 0xff);
         nir_def *old = channel_bits(b, texel, s_channel);
         nir_def *r = nir_iand(b, reference, compare_mask), *v = nir_iand(b, old, compare_mask);
         nir_def *test = nir_ine_imm(b, nir_iand_imm(b, control, 4), 0);
         nir_def *pass = compare_op(b, nir_iand_imm(b, nir_ushr_imm(b, ops, 9), 7),
                                    nir_ult(b, r, v), nir_ieq(b, r, v), nir_ult(b, v, r));
         stencil_pass = nir_ior(b, nir_inot(b, test), pass);
         nir_def *shift = nir_bcsel(b, nir_inot(b, stencil_pass), nir_imm_int(b, 0),
                                    nir_bcsel(b, depth_pass, nir_imm_int(b, 3), nir_imm_int(b, 6)));
         nir_def *result = stencil_op(b, nir_iand_imm(b, nir_ushr(b, ops, shift), 7), old, reference);
         result = nir_ior(b, nir_iand(b, result, write_mask), nir_iand(b, old, nir_inot(b, write_mask)));
         unsigned w = s_channel->shift / 32, at = s_channel->shift % 32;
         nir_def *updated = nir_ior(b, nir_iand_imm(b, words[w], ~(0xffu << at)), nir_ishl_imm(b, result, at));
         words[w] = nir_bcsel(b, nir_iand(b, live, test), updated, words[w]);
      }
      live = nir_iand(b, live, nir_iand(b, depth_pass, stencil_pass));
      if (z_channel) {
         unsigned w = z_channel->shift / 32, at = z_channel->shift % 32;
         uint32_t field = z_channel->size == 32 ? ~0u : BITFIELD_MASK(z_channel->size) << at;
         nir_def *updated = nir_ior(b, nir_iand_imm(b, words[w], ~field), nir_ishl_imm(b, fragment, at));
         words[w] = nir_bcsel(b, nir_iand(b, live, depth_write), updated, words[w]);
      }
      nir_store_var(b, stored_ds, nir_vec(b, words, 4), 0xf);
   }
   for (unsigned k = 0; k < APEX_DRAW_MAX_COLOR; k++) {
      if (!stored[k])
         continue;
      nir_def *old = nir_load_var(b, stored[k]);
      nir_def *src = nir_load_var(b, ctx.color[k]);
      nir_def *state_word = draw_word(b, program, APEX_DRAW_BLEND + k * 2);
      nir_def *ops = draw_word(b, program, APEX_DRAW_BLEND + k * 2 + 1);
      bool integer = util_format_is_pure_integer(formats[k]);
      nir_def *result = src;
      /* Statically disabled blending emits nothing; otherwise the enable
       * bit selects a uniform branch. */
      bool never = !state || !state->cb || (!BITSET_TEST(state->dynamic, MESA_VK_DYNAMIC_CB_BLEND_ENABLES) &&
                   (k >= state->cb->attachment_count || !state->cb->attachments[k].blend_enable));
      if (!integer && !never) {
         nir_push_if(b, nir_ine_imm(b, nir_iand_imm(b, ops, 1u << 24), 0));
         nir_def *dst = unpack_color(b, formats[k], old);
         nir_def *constant = nir_vec4(b, draw_word(b, program, APEX_DRAW_BLEND_CONSTANTS),
            draw_word(b, program, APEX_DRAW_BLEND_CONSTANTS + 1),
            draw_word(b, program, APEX_DRAW_BLEND_CONSTANTS + 2),
            draw_word(b, program, APEX_DRAW_BLEND_CONSTANTS + 3));
         /* Fixed-point attachments clamp blend inputs to their range. */
         nir_def *blend_src = src;
         if (util_format_is_unorm(formats[k])) {
            blend_src = nir_fsat(b, src);
            constant = nir_fsat(b, constant);
         } else if (util_format_is_snorm(formats[k])) {
            blend_src = nir_fclamp(b, src, nir_imm_float(b, -1.0f), nir_imm_float(b, 1.0f));
            constant = nir_fclamp(b, constant, nir_imm_float(b, -1.0f), nir_imm_float(b, 1.0f));
         }
         nir_def *channels[4];
         for (unsigned c = 0; c < 4; c++) {
            bool alpha = c == 3;
            nir_def *sf = blend_factor(b, nir_iand_imm(b, nir_ushr_imm(b, state_word, alpha ? 16 : 0), 0x1f),
                                       blend_src, dst, constant, c);
            nir_def *df = blend_factor(b, nir_iand_imm(b, nir_ushr_imm(b, state_word, alpha ? 24 : 8), 0x1f),
                                       blend_src, dst, constant, c);
            nir_def *op = nir_iand_imm(b, nir_ushr_imm(b, ops, alpha ? 8 : 0), 0xff);
            channels[c] = blend_op(b, op, nir_channel(b, blend_src, c), nir_channel(b, dst, c), sf, df);
         }
         nir_def *blended = nir_vec(b, channels, 4);
         nir_pop_if(b, NULL);
         result = nir_if_phi(b, blended, src);
      }
      nir_def *packed = pack_color(b, formats[k], result);
      nir_def *mask = nir_iand_imm(b, nir_ushr_imm(b, ops, 16), 0xf);
      nir_def *merged[4];
      for (unsigned w = 0; w < 4; w++) {
         if (w >= texel_words(formats[k])) {
            merged[w] = nir_imm_int(b, 0);
            continue;
         }
         nir_def *bits = write_bits(b, formats[k], mask, w);
         merged[w] = nir_ior(b, nir_iand(b, nir_channel(b, packed, w), bits),
                             nir_iand(b, nir_channel(b, old, w), nir_inot(b, bits)));
      }
      nir_store_var(b, stored[k], nir_bcsel(b, live, nir_vec(b, merged, 4), old), 0xf);
   }
   /* Occlusion queries count live samples: one atomic per wave. */
   nir_def *occlusion = draw_address(b, program, APEX_DRAW_OCCLUSION);
   nir_push_if(b, nir_ine_imm(b, nir_ior(b, nir_channel(b, occlusion, 0), nir_channel(b, occlusion, 1)), 0));
   nir_def *passed = nir_bit_count(b, nir_ballot(b, 1, 32, live));
   nir_push_if(b, nir_elect(b, 1));
   nir_global_atomic_2x32(b, 32, occlusion, passed, .atomic_op = nir_atomic_op_iadd);
   nir_pop_if(b, NULL);
   nir_pop_if(b, NULL);
   nir_pop_if(b, NULL); /* any covered */
   nir_pop_if(b, NULL); /* box overlap */
#undef FIELD
   nir_store_var(b, s_var, nir_iadd_imm(b, s, 1), 1);
   nir_pop_loop(b, sub_loop);
   nir_store_var(b, p_var, nir_iadd_imm(b, entry, 1), 1);
   nir_pop_loop(b, prim_loop);
   nir_store_var(b, c_var, nir_iadd_imm(b, chunk, 1), 1);
   nir_pop_loop(b, chunk_loop);

   for (unsigned k = 0; k < APEX_DRAW_MAX_COLOR; k++)
      if (stored[k])
         store_texel(b, &color_target[k], nir_load_var(b, stored[k]), lane, inside);
   if (ds_target)
      store_texel(b, &ds, nir_load_var(b, stored_ds), lane, inside);
   nir_def *groups = root_word(b, trailer + offsetof(struct apex_dispatch_parameters, groups));
   nir_store_var(b, t_var, nir_iadd(b, t, groups), 1);
   nir_pop_loop(b, tile_loop);
   return true;
}

/* ---- Shader objects ------------------------------------------------------ */

static void
apex_shader_destroy(struct vk_device *device, struct vk_shader *vk,
                    const VkAllocationCallbacks *alloc)
{
   struct apex_shader *shader = container_of(vk, struct apex_shader, vk);
   apex_program_finish((struct apex_device *)device, &shader->program);
   for (unsigned s = 0; s < ARRAY_SIZE(shader->set_layouts); s++)
      if (shader->set_layouts[s])
         vk_descriptor_set_layout_unref(device, shader->set_layouts[s]);
   vk_shader_free(device, alloc, vk);
}

/* Binaries are not yet portable across layouts; caches recompile. */
static bool
apex_shader_serialize(struct vk_device *device, const struct vk_shader *shader, struct blob *blob)
{
   return false;
}

static VkResult
apex_shader_executable_properties(struct vk_device *device, const struct vk_shader *shader,
                                  uint32_t *count, VkPipelineExecutablePropertiesKHR *properties)
{
   *count = 0;
   return VK_SUCCESS;
}

static VkResult
apex_shader_executable_statistics(struct vk_device *device, const struct vk_shader *shader,
                                  uint32_t index, uint32_t *count,
                                  VkPipelineExecutableStatisticKHR *statistics)
{
   *count = 0;
   return VK_SUCCESS;
}

static VkResult
apex_shader_executable_representations(struct vk_device *device, const struct vk_shader *shader,
   uint32_t index, uint32_t *count, VkPipelineExecutableInternalRepresentationKHR *representations)
{
   *count = 0;
   return VK_SUCCESS;
}

static const struct vk_shader_ops apex_shader_ops = {
   .destroy = apex_shader_destroy,
   .serialize = apex_shader_serialize,
   .get_executable_properties = apex_shader_executable_properties,
   .get_executable_statistics = apex_shader_executable_statistics,
   .get_executable_internal_representations = apex_shader_executable_representations,
};

static VkResult
compile_stage(struct vk_device *device, struct vk_shader_compile_info *info,
              const struct vk_graphics_pipeline_state *state,
              const VkAllocationCallbacks *alloc, struct vk_shader **out)
{
   nir_shader *nir = info->nir;
   *out = NULL;
   if (info->stage != MESA_SHADER_VERTEX && info->stage != MESA_SHADER_FRAGMENT) {
      ralloc_free(nir);
      return VK_ERROR_FEATURE_NOT_PRESENT;
   }
   struct apex_shader *shader = vk_shader_zalloc(device, &apex_shader_ops, info->stage,
                                                 alloc, sizeof(*shader));
   if (!shader) {
      ralloc_free(nir);
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }
   for (unsigned s = 0; s < info->set_layout_count; s++)
      if (info->set_layouts[s])
         shader->set_layouts[s] = vk_descriptor_set_layout_ref(info->set_layouts[s]);
   VkResult result = VK_ERROR_FEATURE_NOT_PRESENT;
   /* All graphics stages share one push image covering every stage's range. */
   if (!apex_program_layout(&shader->program, info->set_layout_count, info->set_layouts,
                            info->push_constant_range_count, info->push_constant_ranges,
                            VK_SHADER_STAGE_ALL_GRAPHICS))
      goto fail;
   lower_stage_io(nir);
   if (!apex_program_lower_resources(&shader->program, nir))
      goto fail;
   bool built = info->stage == MESA_SHADER_VERTEX ?
      state && state->vi && build_vertex_kernel(shader, nir, state->vi) :
      build_fragment_kernel(shader, nir, state);
   if (!built)
      goto fail;
   convert_to_launch(nir, info->stage == MESA_SHADER_VERTEX ? 64 : 16);
   result = apex_program_compile(device, &shader->program, nir);
   if (result != VK_SUCCESS)
      goto fail;
   ralloc_free(nir);
   *out = &shader->vk;
   return VK_SUCCESS;
fail:
   ralloc_free(nir);
   apex_shader_destroy(device, &shader->vk, alloc);
   return result;
}

static VkResult
apex_compile(struct vk_device *device, uint32_t count, struct vk_shader_compile_info *infos,
             const struct vk_graphics_pipeline_state *state, const struct vk_features *features,
             const VkAllocationCallbacks *alloc, struct vk_shader **shaders)
{
   VkResult result = VK_SUCCESS;
   for (uint32_t i = 0; i < count; i++) {
      if (result != VK_SUCCESS) {
         ralloc_free(infos[i].nir);
         shaders[i] = NULL;
         continue;
      }
      result = compile_stage(device, &infos[i], state, alloc, &shaders[i]);
   }
   if (result != VK_SUCCESS) {
      for (uint32_t i = 0; i < count; i++)
         if (shaders[i])
            apex_shader_destroy(device, shaders[i], alloc);
   }
   return result;
}

static VkResult
apex_deserialize(struct vk_device *device, struct blob_reader *blob, uint32_t version,
                 const VkAllocationCallbacks *alloc, struct vk_shader **out)
{
   return vk_error(device, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);
}

static const struct nir_shader_compiler_options *
apex_get_nir_options(struct vk_physical_device *physical, mesa_shader_stage stage,
                     const struct vk_pipeline_robustness_state *rs)
{
   return &apex_nir_options;
}

static struct spirv_to_nir_options
apex_get_spirv_options(struct vk_physical_device *physical, mesa_shader_stage stage,
                       const struct vk_pipeline_robustness_state *rs)
{
   return (struct spirv_to_nir_options) {
      .ssbo_addr_format = nir_address_format_32bit_index_offset,
      .ubo_addr_format = nir_address_format_32bit_index_offset,
      .shared_addr_format = nir_address_format_32bit_offset,
      .phys_ssbo_addr_format = nir_address_format_64bit_global,
      .push_const_addr_format = nir_address_format_32bit_offset,
   };
}

/* Everything that specializes a stage: vertex input formats and attachment formats. */
static void
apex_hash_state(struct vk_physical_device *physical, const struct vk_graphics_pipeline_state *state,
                const struct vk_features *features, VkShaderStageFlags stages, blake3_hash out)
{
   struct mesa_blake3 hash;
   _mesa_blake3_init(&hash);
   if (state && (stages & VK_SHADER_STAGE_VERTEX_BIT) && state->vi)
      _mesa_blake3_update(&hash, state->vi, sizeof(*state->vi));
   if (state && (stages & VK_SHADER_STAGE_FRAGMENT_BIT) && state->rp) {
      _mesa_blake3_update(&hash, state->rp->color_attachment_formats,
                          sizeof(state->rp->color_attachment_formats));
      _mesa_blake3_update(&hash, &state->rp->color_attachment_count,
                          sizeof(state->rp->color_attachment_count));
      _mesa_blake3_update(&hash, &state->rp->depth_attachment_format,
                          sizeof(state->rp->depth_attachment_format));
      _mesa_blake3_update(&hash, &state->rp->stencil_attachment_format,
                          sizeof(state->rp->stencil_attachment_format));
   }
   _mesa_blake3_final(&hash, out);
}

const struct vk_device_shader_ops apex_device_shader_ops = {
   .get_nir_options = apex_get_nir_options,
   .get_spirv_options = apex_get_spirv_options,
   .hash_state = apex_hash_state,
   .compile = apex_compile,
   .deserialize = apex_deserialize,
   .cmd_bind_shaders = apex_cmd_bind_shaders,
   .cmd_set_dynamic_graphics_state = vk_cmd_set_dynamic_graphics_state,
};

/* ---- Internal programs ---------------------------------------------------- */

VkResult
apex_internal_program(struct apex_device *device, enum apex_internal which,
                      struct apex_program **out)
{
   if (device->internal[which]) {
      *out = device->internal[which];
      return VK_SUCCESS;
   }
   static const struct { const uint32_t *code; size_t words; } sources[] = {
      [APEX_INTERNAL_SETUP] = {apex_setup_spv, ARRAY_SIZE(apex_setup_spv)},
      [APEX_INTERNAL_BIN] = {apex_bin_spv, ARRAY_SIZE(apex_bin_spv)},
      [APEX_INTERNAL_COPY] = {apex_copy_spv, ARRAY_SIZE(apex_copy_spv)},
      [APEX_INTERNAL_CLEAR] = {apex_clear_spv, ARRAY_SIZE(apex_clear_spv)},
      [APEX_INTERNAL_TIMESTAMP] = {apex_timestamp_spv, ARRAY_SIZE(apex_timestamp_spv)},
      [APEX_INTERNAL_QUERY_COPY] = {apex_querycopy_spv, ARRAY_SIZE(apex_querycopy_spv)},
      [APEX_INTERNAL_RESOLVE] = {apex_resolve_spv, ARRAY_SIZE(apex_resolve_spv)},
   };
   struct apex_program *program = vk_zalloc(&device->vk.alloc, sizeof(*program), 8,
                                            VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!program)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   /* Internal kernels address the data root directly: no descriptor sets. */
   program->table = true;
   struct spirv_capabilities caps = {
      .Shader = true, .Int64 = true, .PhysicalStorageBufferAddresses = true,
      .ShaderClockKHR = true,
   };
   struct spirv_to_nir_options options = apex_get_spirv_options(device->vk.physical,
                                                                MESA_SHADER_COMPUTE, NULL);
   options.environment = NIR_SPIRV_VULKAN;
   options.capabilities = &caps;
   nir_shader *nir = spirv_to_nir(sources[which].code, sources[which].words, NULL,
                                  MESA_SHADER_COMPUTE, "main", &options, &apex_nir_options);
   VkResult result = nir ? apex_program_compile(&device->vk, program, nir) :
                           VK_ERROR_INITIALIZATION_FAILED;
   ralloc_free(nir);
   if (result != VK_SUCCESS) {
      apex_program_finish(device, program);
      vk_free(&device->vk.alloc, program);
      return result;
   }
   device->internal[which] = program;
   *out = program;
   return VK_SUCCESS;
}

void
apex_graphics_finish(struct apex_device *device)
{
   apex_bo_finish(device, &device->arena);
   for (unsigned i = 0; i < APEX_INTERNAL_COUNT; i++) {
      if (device->internal[i]) {
         apex_program_finish(device, device->internal[i]);
         vk_free(&device->vk.alloc, device->internal[i]);
         device->internal[i] = NULL;
      }
   }
}
