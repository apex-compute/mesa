/* SPDX-License-Identifier: MIT */
/* Graphics shaders as P7 vertex and fragment programs (Docs/isa.md) and the
 * internal compute kernels. Vertex attribute location L reads vector
 * registers 2 + 4L in the memory order of its components; vertex programs
 * store the output record, fragment programs export their targets. Fragment
 * programs are specialized on the attachment formats: R8, RG8, RGBA8 and
 * BGRA8 UNORM and sRGB targets take UNORM16 codes (Docs/architecture.md,
 * ROP and tile buffer). Clip then cull distances also pass as varyings 30
 * and 31: the fragment program discards where a clip distance is negative
 * and reads its gl_ClipDistance and gl_CullDistance there. */
#include "apex_graphics.h"
#include "apex_hw.h"
#include "apex_job.h"
#include "apex_copy_spv.h"
#include "apex_clear_spv.h"
#include "apex_querycopy_spv.h"
#include "apex_etc2_spv.h"
#include "apex_xfb_spv.h"
#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/nir/nir_xfb_info.h"
#include "compiler/spirv/nir_spirv.h"
#include "compiler/spirv/spirv_info.h"
#include "vk_alloc.h"
#include "vk_format.h"
#include "vk_graphics_state.h"
#include "vk_log.h"
#include "vk_nir.h"
#include "vk_pipeline.h"
#include "vk_shader.h"
#include "util/format/u_format.h"
#include "util/u_dynarray.h"

#define CLIP_VARYING (VARYING_SLOT_VAR0 + 30)

/* Clip and cull distances of the vertex stage. */
struct distances {
   unsigned clip, cull;
};

static nir_variable *
stage_variable(nir_shader *nir, nir_variable_mode mode, unsigned location, const struct glsl_type *type,
               const char *name)
{
   nir_variable *var = nir_find_variable_with_location(nir, mode, location);
   if (!var) {
      var = nir_variable_create(nir, mode, type, name);
      var->data.location = location;
   }
   return var;
}

/* The merged clip and cull distance arrays (nir_merge_clip_cull_distance_vars):
 * the compact array starting `offset` distances into the clip slots. */
static nir_variable *
distance_variable(nir_shader *nir, nir_variable_mode mode, unsigned offset)
{
   nir_foreach_variable_with_modes(var, nir, mode) {
      if (var->data.compact && (var->data.location == VARYING_SLOT_CLIP_DIST0 ||
                                var->data.location == VARYING_SLOT_CLIP_DIST1) &&
          4 * (var->data.location - VARYING_SLOT_CLIP_DIST0) + var->data.location_frac == offset)
         return var;
   }
   return NULL;
}

/* ---- Vertex stage -------------------------------------------------------- */

/* BGR-ordered attribute formats arrive in memory order; swap red and blue. */
static bool
swap_attributes(nir_builder *b, nir_intrinsic_instr *i, void *data)
{
   const struct vk_vertex_input_state *vi = data;
   if (i->intrinsic != nir_intrinsic_load_deref || i->def.num_components < 3)
      return false;
   nir_deref_instr *deref = nir_src_as_deref(i->src[0]);
   if (deref->deref_type != nir_deref_type_var || deref->var->data.mode != nir_var_shader_in ||
       deref->var->data.location < VERT_ATTRIB_GENERIC0)
      return false;
   unsigned location = deref->var->data.location - VERT_ATTRIB_GENERIC0;
   bool swap = false;
   if (!vi || location >= APEX_HW_MAX_ATTRIBUTES || !(vi->attributes_valid & BITFIELD_BIT(location)) ||
       !apex_hw_vertex_format(vi->attributes[location].format, &swap) || !swap)
      return false;
   b->cursor = nir_after_instr(&i->instr);
   unsigned swizzle[4] = {2, 1, 0, 3};
   nir_def *swapped = nir_swizzle(b, &i->def, swizzle, i->def.num_components);
   nir_def_rewrite_uses_after(&i->def, swapped);
   return true;
}

/* Multiview: the view index is vertex user data s2 and the layer. */
static bool
lower_view_index(nir_builder *b, nir_intrinsic_instr *i, void *data)
{
   if (i->intrinsic != nir_intrinsic_load_view_index)
      return false;
   b->cursor = nir_before_instr(&i->instr);
   nir_def_replace(&i->def, nir_load_kernel_input(b, 1, 32, nir_imm_int(b, 0), .base = 8, .range = 4));
   return true;
}

static bool
lower_vertex(nir_shader *nir, const struct vk_graphics_pipeline_state *state, struct distances d)
{
   unsigned varyings = d.clip + d.cull;
   NIR_PASS(_, nir, nir_split_per_member_structs);
   NIR_PASS(_, nir, nir_lower_returns);
   if (state && state->vi)
      NIR_PASS(_, nir, nir_shader_intrinsics_pass, swap_attributes, nir_metadata_control_flow,
               (void *)state->vi);
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder builder = nir_builder_at(nir_after_impl(impl));
   nir_builder *b = &builder;
   if (state && state->mv && state->mv->view_mask) {
      nir_variable *layer = stage_variable(nir, nir_var_shader_out, VARYING_SLOT_LAYER, glsl_int_type(),
                                           "apex_layer");
      nir_store_var(b, layer, nir_load_view_index(b), 1);
      NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_view_index, nir_metadata_control_flow, NULL);
   }
   if (varyings) {
      nir_variable *clip = distance_variable(nir, nir_var_shader_out, 0);
      nir_variable *cull = distance_variable(nir, nir_var_shader_out, d.clip);
      for (unsigned v = 0; v < DIV_ROUND_UP(varyings, 4); v++) {
         if (nir_find_variable_with_location(nir, nir_var_shader_out, CLIP_VARYING + v))
            return false;
         nir_variable *out = nir_variable_create(nir, nir_var_shader_out, glsl_vec4_type(), "apex_clip");
         out->data.location = CLIP_VARYING + v;
         nir_def *values[4];
         for (unsigned c = 0; c < 4; c++) {
            unsigned k = 4 * v + c;
            values[c] = k < d.clip ? nir_load_array_var_imm(b, clip, k) :
                        k < varyings ? nir_load_array_var_imm(b, cull, k - d.clip) : nir_imm_float(b, 0.0f);
         }
         nir_store_var(b, out, nir_vec(b, values, 4), 0xf);
      }
   }
   nir_progress(true, impl, nir_metadata_none);
   return true;
}

/* ---- Transform feedback capture ------------------------------------------ */

/* The capture program: the vertex program built as a compute kernel over a
 * draw's output vertices (apex_job.h, APEX_XFB_*). Invocations stride by
 * the dispatch over output vertex o = (instance * primitives + p) * n + k
 * of n-vertex primitives in Vulkan's topology order, up to the primitives
 * that fit every written buffer. Each fetches its attributes in software,
 * runs the vertex shader and stores the captured outputs at the buffer's
 * offset plus o times its stride. Strips and fans with primitive restart
 * find primitive p by scanning the indices. */
struct capture {
   const struct apex_program *program;
   const struct vk_vertex_input_state *vi;
   const nir_xfb_info *xfb;
   nir_def *flags, *first, *first_instance, *vertex_offset, *draw_index, *indexed;
   nir_def *vertex, *instance, *output, *state;
   bool invalid;
};

static nir_def *
root_word(nir_builder *b, nir_def *offset)
{
   return nir_load_ssbo(b, 1, 32, nir_imm_int(b, 0), offset, .align_mul = 4,
                        .access = ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER | ACCESS_CAN_SPECULATE);
}

static nir_def *
job_word(nir_builder *b, const struct apex_program *program, nir_def *word)
{
   return root_word(b, nir_iadd_imm(b, nir_ishl_imm(b, word, 2),
                                    apex_program_trailer(program) + APEX_TRAILER_WORDS * 4));
}

static nir_def *
job_imm(nir_builder *b, const struct apex_program *program, unsigned word)
{
   return job_word(b, program, nir_imm_int(b, word));
}

static nir_def *
job_address(nir_builder *b, const struct apex_program *program, unsigned word)
{
   return nir_vec2(b, job_imm(b, program, word), job_imm(b, program, word + 1));
}

static nir_def *
address_add(nir_builder *b, nir_def *address, nir_def *bytes)
{
   return nir_build_addr_iadd(b, address, nir_address_format_2x32bit_global, nir_var_mem_global, bytes);
}

static nir_def *
load_word(nir_builder *b, nir_def *address)
{
   return nir_load_global_2x32(b, 1, 32, address, .align_mul = 4);
}

static nir_def *
address_nonzero(nir_builder *b, nir_def *address)
{
   return nir_ine_imm(b, nir_ior(b, nir_channel(b, address, 0), nir_channel(b, address, 1)), 0);
}

/* Draw command word `word`, from the indirect command when there is one. */
static nir_def *
draw_word(nir_builder *b, const struct apex_program *program, nir_def *word)
{
   nir_def *params = job_address(b, program, APEX_XFB_PARAMS);
   nir_push_if(b, address_nonzero(b, params));
   nir_def *loaded = load_word(b, address_add(b, params, nir_ishl_imm(b, word, 2)));
   nir_push_else(b, NULL);
   nir_def *direct = job_word(b, program, nir_iadd_imm(b, word, APEX_XFB_DRAW));
   nir_pop_if(b, NULL);
   return nir_if_phi(b, loaded, direct);
}

/* `bytes` (1-4) bytes at byte offset `byte` as the low bits of a word. */
static nir_def *
load_bytes(nir_builder *b, nir_def *address, nir_def *byte, unsigned bytes)
{
   nir_def *aligned = address_add(b, address, nir_iand_imm(b, byte, ~3u));
   nir_def *shift = nir_ishl_imm(b, nir_iand_imm(b, byte, 3), 3);
   nir_def *lo = nir_ushr(b, load_word(b, aligned), shift);
   if (bytes == 1)
      return lo;
   nir_push_if(b, nir_ugt_imm(b, nir_iadd_imm(b, nir_iand_imm(b, byte, 3), bytes), 4));
   nir_def *hi = load_word(b, address_add(b, aligned, nir_imm_int(b, 4)));
   nir_def *joined = nir_ior(b, lo, nir_ishl(b, hi, nir_isub_imm(b, 32, shift)));
   nir_pop_if(b, NULL);
   return nir_if_phi(b, joined, lo);
}

/* The index at draw position `position`, zero beyond the bound bytes;
 * `mask` receives the index type's restart value. Indices are aligned. */
static nir_def *
fetch_index(nir_builder *b, struct capture *ctx, nir_def *position, nir_def **mask)
{
   nir_def *shift = nir_iand_imm(b, ctx->flags, 3);
   nir_def *byte = nir_ishl(b, nir_iadd(b, ctx->first, position), shift);
   nir_def *end = nir_iadd(b, byte, nir_ishl(b, nir_imm_int(b, 1), shift));
   nir_def *zero = nir_imm_int(b, 0);
   nir_push_if(b, nir_uge(b, job_imm(b, ctx->program, APEX_XFB_INDEX_BYTES), end));
   nir_def *word = load_bytes(b, job_address(b, ctx->program, APEX_XFB_INDEX), byte, 1);
   nir_pop_if(b, NULL);
   word = nir_if_phi(b, word, zero);
   *mask = nir_bcsel(b, nir_ieq_imm(b, shift, 2), nir_imm_int(b, ~0),
                     nir_bcsel(b, nir_ieq_imm(b, shift, 1), nir_imm_int(b, 0xffff), nir_imm_int(b, 0xff)));
   return nir_iand(b, word, *mask);
}

/* Restart scan over `count` positions: the primitive count, and with
 * `target` the first position of primitive target's run and its ordinal
 * in the run. */
static void
restart_scan(nir_builder *b, struct capture *ctx, nir_def *count, nir_def *verts, nir_def *target,
             nir_def **primitives, nir_def **start, nir_def **ordinal)
{
   nir_function_impl *impl = b->impl;
   nir_variable *i = nir_local_variable_create(impl, glsl_uint_type(), "scan_position");
   nir_variable *run = nir_local_variable_create(impl, glsl_uint_type(), "scan_run");
   nir_variable *n = nir_local_variable_create(impl, glsl_uint_type(), "scan_primitives");
   nir_variable *first = nir_local_variable_create(impl, glsl_uint_type(), "scan_start");
   nir_variable *ord = nir_local_variable_create(impl, glsl_uint_type(), "scan_ordinal");
   nir_variable *vars[5] = {i, run, n, first, ord};
   for (unsigned v = 0; v < ARRAY_SIZE(vars); v++)
      nir_store_var(b, vars[v], nir_imm_int(b, 0), 1);
   nir_push_loop(b);
   nir_def *position = nir_load_var(b, i);
   nir_break_if(b, nir_uge(b, position, count));
   nir_def *mask;
   nir_def *index = fetch_index(b, ctx, position, &mask);
   nir_def *length = nir_bcsel(b, nir_ieq(b, index, mask), nir_imm_int(b, 0),
                               nir_iadd_imm(b, nir_load_var(b, run), 1));
   nir_store_var(b, run, length, 1);
   nir_store_var(b, first, nir_bcsel(b, nir_ieq_imm(b, length, 1), position, nir_load_var(b, first)), 1);
   nir_store_var(b, i, nir_iadd_imm(b, position, 1), 1);
   nir_push_if(b, nir_uge(b, length, verts));
   nir_def *found = nir_load_var(b, n);
   if (target) {
      nir_store_var(b, ord, nir_isub(b, length, verts), 1);
      nir_break_if(b, nir_ieq(b, found, target));
   }
   nir_store_var(b, n, nir_iadd_imm(b, found, 1), 1);
   nir_pop_if(b, NULL);
   nir_pop_loop(b, NULL);
   *primitives = nir_load_var(b, n);
   *start = nir_load_var(b, first);
   *ordinal = nir_load_var(b, ord);
}

/* Vertex k of primitive j of a run as a position from the run's start, in
 * Vulkan's topology order for the first or last provoking vertex. */
static nir_def *
topology_vertex(nir_builder *b, nir_def *topology, nir_def *last, nir_def *j, nir_def *k)
{
   nir_def *odd = nir_iand_imm(b, j, 1), *even = nir_ixor(b, odd, nir_imm_int(b, 1));
   nir_def *k0 = nir_ieq_imm(b, k, 0), *k1 = nir_ieq_imm(b, k, 1);
   nir_def *next = nir_iadd_imm(b, j, 1);
   nir_def *strip_first = nir_bcsel(b, k0, j, nir_bcsel(b, k1, nir_iadd(b, next, odd), nir_iadd(b, next, even)));
   nir_def *strip_last = nir_bcsel(b, k0, nir_iadd(b, j, odd),
                                   nir_bcsel(b, k1, nir_iadd(b, j, even), nir_iadd_imm(b, j, 2)));
   nir_def *fan_first = nir_bcsel(b, k0, next, nir_bcsel(b, k1, nir_iadd_imm(b, j, 2), nir_imm_int(b, 0)));
   nir_def *fan_last = nir_bcsel(b, k0, nir_imm_int(b, 0), nir_iadd(b, j, k));
   nir_def *values[6] = {
      j, nir_iadd(b, nir_imul_imm(b, j, 2), k), nir_iadd(b, j, k), nir_iadd(b, nir_imul_imm(b, j, 3), k),
      nir_bcsel(b, last, strip_last, strip_first), nir_bcsel(b, last, fan_last, fan_first),
   };
   nir_def *result = values[5];
   for (int t = 4; t >= 0; t--)
      result = nir_bcsel(b, nir_ieq_imm(b, topology, t), values[t], result);
   return result;
}

/* One channel of a plain vertex format as FP32 or integer bits. */
static nir_def *
fetch_channel(nir_builder *b, const struct util_format_description *desc, unsigned c,
              nir_def *address, nir_def *byte)
{
   const struct util_format_channel_description *ch = &desc->channel[c];
   unsigned bits = ch->size, shift = ch->shift;
   nir_def *value = load_bytes(b, address, nir_iadd_imm(b, byte, shift / 8), DIV_ROUND_UP(shift % 8 + bits, 8));
   if (bits < 32)
      value = nir_iand_imm(b, nir_ushr_imm(b, value, shift % 8), BITFIELD_MASK(bits));
   switch (ch->type) {
   case UTIL_FORMAT_TYPE_FLOAT:
      return bits == 16 ? apex_half_to_float(b, value) : value;
   case UTIL_FORMAT_TYPE_UNSIGNED:
      if (ch->normalized)
         return nir_fmul_imm(b, nir_u2f32(b, value), 1.0 / BITFIELD_MASK(bits));
      return ch->pure_integer ? value : nir_u2f32(b, value);
   case UTIL_FORMAT_TYPE_SIGNED: {
      nir_def *extended = bits == 32 ? value : nir_ishr_imm(b, nir_ishl_imm(b, value, 32 - bits), 32 - bits);
      if (ch->normalized)
         return nir_fmax(b, nir_fmul_imm(b, nir_i2f32(b, extended), 1.0 / BITFIELD_MASK(bits - 1)),
                         nir_imm_float(b, -1.0f));
      return ch->pure_integer ? extended : nir_i2f32(b, extended);
   }
   default:
      return nir_imm_int(b, 0);
   }
}

/* Attribute `location` through its binding's words in the vertex-input
 * block; outside the bound bytes it reads (0, 0, 0, 1). */
static nir_def *
fetch_attribute(nir_builder *b, struct capture *ctx, unsigned location, unsigned component,
                unsigned count, bool integer)
{
   const struct vk_vertex_input_state *vi = ctx->vi;
   const struct util_format_description *desc = NULL;
   if (vi && location < APEX_HW_MAX_ATTRIBUTES && (vi->attributes_valid & BITFIELD_BIT(location)))
      desc = util_format_description(vk_format_to_pipe_format(vi->attributes[location].format));
   if (!desc || desc->layout != UTIL_FORMAT_LAYOUT_PLAIN || desc->block.bits > 128 ||
       desc->channel[0].size > 32 || component + count > 4) {
      ctx->invalid = true;
      return nir_imm_zero(b, count, 32);
   }
   const struct vk_vertex_attribute_state *attr = &vi->attributes[location];
   nir_def *block = job_address(b, ctx->program, APEX_XFB_VERTEX_INPUT);
   nir_def *words = nir_load_global_2x32(b, 4, 32, address_add(b, block, nir_imm_int(b, 16 * attr->binding)),
                                         .align_mul = 16);
   nir_def *w1 = nir_channel(b, words, 1), *divisor = nir_channel(b, words, 3);
   nir_def *address = nir_vec2(b, nir_channel(b, words, 0), nir_iand_imm(b, w1, 0xff));
   /* Instance rate: firstInstance + (InstanceIndex - firstInstance) / divisor;
    * a zero divisor repeats firstInstance. */
   nir_def *stepped = nir_iadd(b, ctx->first_instance,
                               nir_udiv(b, nir_isub(b, ctx->instance, ctx->first_instance),
                                        nir_umax(b, divisor, nir_imm_int(b, 1))));
   nir_def *instance = nir_bcsel(b, nir_ieq_imm(b, divisor, 0), ctx->first_instance, stepped);
   nir_def *element = nir_bcsel(b, nir_ine_imm(b, nir_iand_imm(b, w1, 0x100), 0), instance, ctx->vertex);
   nir_def *byte = nir_iadd_imm(b, nir_imul(b, element, nir_ushr_imm(b, w1, 16)), attr->offset);
   nir_def *inside = nir_uge(b, nir_channel(b, words, 2), nir_iadd_imm(b, byte, desc->block.bits / 8));
   nir_def *fallback[4], *fetched[4], *values[4];
   for (unsigned c = 0; c < count; c++)
      fallback[c] = component + c == 3 ? (integer ? nir_imm_int(b, 1) : nir_imm_float(b, 1.0f)) :
                    nir_imm_int(b, 0);
   nir_push_if(b, inside);
   for (unsigned c = 0; c < count; c++) {
      unsigned swizzle = desc->swizzle[component + c];
      if (swizzle <= PIPE_SWIZZLE_W)
         fetched[c] = fetch_channel(b, desc, swizzle, address, byte);
      else if (swizzle == PIPE_SWIZZLE_1)
         fetched[c] = integer ? nir_imm_int(b, 1) : nir_imm_float(b, 1.0f);
      else
         fetched[c] = nir_imm_int(b, 0);
   }
   nir_pop_if(b, NULL);
   for (unsigned c = 0; c < count; c++)
      values[c] = nir_if_phi(b, fetched[c], fallback[c]);
   return nir_vec(b, values, count);
}

/* Stores the captured components of one output store. */
static void
capture_output(nir_builder *b, struct capture *ctx, nir_intrinsic_instr *i)
{
   nir_src *offset = nir_get_io_offset_src(i);
   if (!nir_src_is_const(*offset)) {
      ctx->invalid = true;
      return;
   }
   unsigned location = nir_intrinsic_io_semantics(i).location + nir_src_as_uint(*offset);
   nir_def *value = i->src[0].ssa;
   for (unsigned o = 0; o < ctx->xfb->output_count; o++) {
      const nir_xfb_output_info *out = &ctx->xfb->outputs[o];
      if (out->location != location)
         continue;
      if (out->data_is_16bit || ctx->xfb->buffer_to_stream[out->buffer] || value->bit_size != 32) {
         ctx->invalid = true;
         return;
      }
      unsigned stride = ctx->xfb->buffers[out->buffer].stride;
      nir_def *buffer = nir_load_global_2x32(b, 2, 32,
         address_add(b, ctx->state, nir_imm_int(b, 4 * (APEX_XFB_STATE_BUFFERS + 3 * out->buffer))),
         .align_mul = 4);
      nir_def *base = nir_iadd(b, load_word(b, address_add(b, ctx->state, nir_imm_int(b, 4 * out->buffer))),
                               nir_imul_imm(b, ctx->output, stride));
      for (unsigned c = 0; c < value->num_components; c++) {
         unsigned component = nir_intrinsic_component(i) + c;
         if (!(nir_intrinsic_write_mask(i) & BITFIELD_BIT(c)) || !(out->component_mask & BITFIELD_BIT(component)))
            continue;
         unsigned byte = out->offset + 4 * util_bitcount(out->component_mask & BITFIELD_MASK(component));
         nir_store_global_2x32(b, nir_channel(b, value, c), address_add(b, buffer, nir_iadd_imm(b, base, byte)),
                               .align_mul = 4);
      }
   }
}

static bool
lower_capture(nir_builder *b, nir_intrinsic_instr *i, void *data)
{
   struct capture *ctx = data;
   b->cursor = nir_before_instr(&i->instr);
   nir_def *replacement;
   switch (i->intrinsic) {
   case nir_intrinsic_load_input: {
      if (i->def.bit_size != 32) {
         ctx->invalid = true;
         return false;
      }
      nir_alu_type type = nir_intrinsic_dest_type(i);
      replacement = fetch_attribute(b, ctx, nir_intrinsic_io_semantics(i).location - VERT_ATTRIB_GENERIC0,
                                    nir_intrinsic_component(i), i->def.num_components,
                                    nir_alu_type_get_base_type(type) != nir_type_float);
      break;
   }
   case nir_intrinsic_store_output:
      capture_output(b, ctx, i);
      nir_instr_remove(&i->instr);
      return true;
   case nir_intrinsic_load_vertex_id:
      replacement = ctx->vertex;
      break;
   case nir_intrinsic_load_vertex_id_zero_base:
      replacement = nir_isub(b, ctx->vertex, nir_bcsel(b, ctx->indexed, ctx->vertex_offset, ctx->first));
      break;
   case nir_intrinsic_load_first_vertex:
   case nir_intrinsic_load_base_vertex:
      replacement = nir_bcsel(b, ctx->indexed, ctx->vertex_offset, ctx->first);
      break;
   case nir_intrinsic_load_instance_id:
      replacement = nir_isub(b, ctx->instance, ctx->first_instance);
      break;
   case nir_intrinsic_load_base_instance:
      replacement = ctx->first_instance;
      break;
   case nir_intrinsic_load_draw_id:
      replacement = ctx->draw_index;
      break;
   case nir_intrinsic_load_view_index:
      replacement = nir_imm_int(b, 0);
      break;
   case nir_intrinsic_load_is_indexed_draw:
      replacement = nir_b2i32(b, ctx->indexed);
      break;
   default:
      return false;
   }
   nir_def_replace(&i->def, replacement);
   return true;
}

static unsigned
attribute_slots(const struct glsl_type *type, bool bindless)
{
   return glsl_count_attribute_slots(type, false);
}

/* Builds the capture program of a vertex shader from its NIR; `program`
 * holds the vertex program's descriptor layout. */
static VkResult
build_capture(struct vk_device *device, nir_shader *nir, const struct vk_vertex_input_state *vi,
              struct apex_program *program)
{
   NIR_PASS(_, nir, nir_split_per_member_structs);
   NIR_PASS(_, nir, nir_lower_returns);
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   NIR_PASS(_, nir, nir_lower_io_vars_to_temporaries, impl, nir_var_shader_in | nir_var_shader_out);
   NIR_PASS(_, nir, nir_lower_global_vars_to_local);
   NIR_PASS(_, nir, nir_split_var_copies);
   NIR_PASS(_, nir, nir_lower_var_copies);
   NIR_PASS(_, nir, nir_lower_vars_to_ssa);
   NIR_PASS(_, nir, nir_lower_indirect_derefs_to_if_else_trees, nir_var_shader_in | nir_var_shader_out,
            UINT32_MAX);
   nir_foreach_variable_with_modes(var, nir, nir_var_shader_in | nir_var_shader_out) {
      var->data.driver_location = var->data.location;
      if (var->data.mode == nir_var_shader_in)
         var->data.driver_location -= VERT_ATTRIB_GENERIC0;
   }
   NIR_PASS(_, nir, nir_lower_io, nir_var_shader_in | nir_var_shader_out, attribute_slots, 0);
   if (!apex_program_lower_resources(program, nir))
      return VK_ERROR_FEATURE_NOT_PRESENT;
   const nir_xfb_info *xfb = nir->xfb_info;
   nir->xfb_info = NULL;
   nir->info.stage = MESA_SHADER_COMPUTE;
   memset(&nir->info.cs, 0, sizeof(nir->info.cs));
   nir->info.workgroup_size[0] = APEX_XFB_LOCAL;
   nir->info.workgroup_size[1] = nir->info.workgroup_size[2] = 1;
   nir->info.workgroup_size_variable = false;

   impl = nir_shader_get_entrypoint(nir);
   nir_cf_list body;
   nir_cf_extract(&body, nir_before_impl(impl), nir_after_impl(impl));
   nir_builder builder = nir_builder_at(nir_before_impl(impl));
   nir_builder *b = &builder;
   struct capture ctx = {.program = program, .vi = vi, .xfb = xfb};
   nir_def *zero = nir_imm_int(b, 0);
   ctx.flags = job_imm(b, program, APEX_XFB_FLAGS);
   ctx.indexed = nir_ine_imm(b, nir_iand_imm(b, ctx.flags, APEX_XFB_FLAG_INDEXED), 0);
   nir_def *count = draw_word(b, program, nir_imm_int(b, 0));
   nir_def *instances = draw_word(b, program, nir_imm_int(b, 1));
   ctx.first = draw_word(b, program, nir_imm_int(b, 2));
   ctx.vertex_offset = draw_word(b, program, nir_imm_int(b, 3));
   ctx.first_instance = draw_word(b, program, nir_bcsel(b, ctx.indexed, nir_imm_int(b, 4), nir_imm_int(b, 3)));
   ctx.draw_index = job_imm(b, program, APEX_XFB_DRAW_INDEX);
   ctx.state = job_address(b, program, APEX_XFB_STATE);
   nir_def *topology = nir_iand_imm(b, nir_ushr_imm(b, ctx.flags, 8), 7);
   nir_def *last = nir_ine_imm(b, nir_iand_imm(b, ctx.flags, APEX_XFB_FLAG_LAST), 0);
   nir_def *verts = nir_bcsel(b, nir_ieq_imm(b, topology, 0), nir_imm_int(b, 1),
                              nir_bcsel(b, nir_ult_imm(b, topology, 3), nir_imm_int(b, 2), nir_imm_int(b, 3)));
   nir_def *strip = nir_ior(b, nir_ieq_imm(b, topology, 2), nir_uge_imm(b, topology, 4));
   nir_def *restart = nir_iand(b, nir_iand(b, ctx.indexed, strip),
                               nir_ine_imm(b, nir_iand_imm(b, ctx.flags, APEX_XFB_FLAG_RESTART), 0));
   /* Primitives per instance; none at or past the draw count. */
   nir_def *runs = nir_bcsel(b, strip, nir_isub(b, nir_iadd_imm(b, count, 1), verts), nir_udiv(b, count, verts));
   runs = nir_bcsel(b, nir_ult(b, count, verts), zero, runs);
   nir_push_if(b, restart);
   nir_def *scanned, *unused_start, *unused_ordinal;
   restart_scan(b, &ctx, count, verts, NULL, &scanned, &unused_start, &unused_ordinal);
   nir_pop_if(b, NULL);
   nir_def *primitives = nir_if_phi(b, scanned, runs);
   nir_def *count_va = job_address(b, program, APEX_XFB_COUNT), *unbounded = nir_imm_int(b, ~0);
   nir_push_if(b, address_nonzero(b, count_va));
   nir_def *draws = load_word(b, count_va);
   nir_pop_if(b, NULL);
   draws = nir_if_phi(b, draws, unbounded);
   primitives = nir_bcsel(b, nir_ult(b, ctx.draw_index, draws), primitives, zero);
   /* Primitives that fit every written buffer. */
   nir_def *fit = nir_imul(b, primitives, instances);
   for (unsigned buffer = 0; buffer < APEX_XFB_BUFFERS; buffer++) {
      unsigned stride = xfb && (xfb->buffers_written & BITFIELD_BIT(buffer)) ? xfb->buffers[buffer].stride : 0;
      if (!stride)
         continue;
      nir_def *offset = load_word(b, address_add(b, ctx.state, nir_imm_int(b, 4 * buffer)));
      nir_def *size = load_word(b, address_add(b, ctx.state,
                                               nir_imm_int(b, 4 * (APEX_XFB_STATE_BUFFERS + 3 * buffer + 2))));
      nir_def *room = nir_udiv(b, nir_isub(b, size, offset), nir_imul_imm(b, verts, stride));
      fit = nir_umin(b, fit, nir_bcsel(b, nir_ult(b, offset, size), room, zero));
   }
   nir_def *outputs = nir_imul(b, fit, verts);
   nir_def *invocations = nir_imul_imm(b, root_word(b, nir_imm_int(b, apex_program_trailer(program))),
                                       APEX_XFB_LOCAL);
   nir_variable *o_var = nir_local_variable_create(impl, glsl_uint_type(), "capture_vertex");
   nir_store_var(b, o_var, nir_iadd(b, nir_imul_imm(b, nir_channel(b, nir_load_workgroup_id(b), 0), APEX_XFB_LOCAL),
                                    nir_load_local_invocation_index(b)), 1);
   nir_loop *loop = nir_push_loop(b);
   nir_def *o = nir_load_var(b, o_var);
   nir_break_if(b, nir_uge(b, o, outputs));
   nir_store_var(b, o_var, nir_iadd(b, o, invocations), 1);
   nir_def *global = nir_udiv(b, o, verts), *k = nir_isub(b, o, nir_imul(b, global, verts));
   nir_def *instance = nir_udiv(b, global, primitives);
   nir_def *p = nir_isub(b, global, nir_imul(b, instance, primitives));
   nir_push_if(b, restart);
   nir_def *found, *start, *ordinal;
   restart_scan(b, &ctx, count, verts, p, &found, &start, &ordinal);
   nir_pop_if(b, NULL);
   start = nir_if_phi(b, start, zero);
   nir_def *j = nir_if_phi(b, ordinal, p);
   nir_def *position = nir_iadd(b, start, topology_vertex(b, topology, last, j, k));
   nir_def *direct = nir_iadd(b, ctx.first, position), *mask;
   nir_push_if(b, ctx.indexed);
   nir_def *index = nir_iadd(b, fetch_index(b, &ctx, position, &mask), ctx.vertex_offset);
   nir_pop_if(b, NULL);
   ctx.vertex = nir_if_phi(b, index, direct);
   ctx.instance = nir_iadd(b, ctx.first_instance, instance);
   ctx.output = o;
   nir_cf_reinsert(&body, b->cursor);
   b->cursor = nir_after_cf_list(&loop->body);
   nir_pop_loop(b, loop);
   nir_progress(true, impl, nir_metadata_none);

   /* Lowering inserts control flow: collect the intrinsics first. */
   struct util_dynarray list;
   util_dynarray_init(&list, NULL);
   nir_foreach_block(block, impl) {
      nir_foreach_instr(instr, block) {
         if (instr->type == nir_instr_type_intrinsic)
            util_dynarray_append(&list, nir_instr_as_intrinsic(instr));
      }
   }
   nir_builder lower = nir_builder_create(impl);
   util_dynarray_foreach(&list, nir_intrinsic_instr *, i)
      lower_capture(&lower, *i, &ctx);
   util_dynarray_fini(&list);
   nir_progress(true, impl, nir_metadata_none);
   NIR_PASS(_, nir, nir_lower_vars_to_ssa);
   if (ctx.invalid)
      return VK_ERROR_FEATURE_NOT_PRESENT;
   return apex_program_compile(device, program, nir);
}

/* ---- Fragment stage ------------------------------------------------------ */

/* Fixed-point ROP targets: R8, RG8 and RGBA8 classes, UNORM or sRGB. */
static bool
unorm16_target(VkFormat format)
{
   uint8_t code = apex_hw_color_format(format);
   unsigned class = code & 15, type = code >> 4 & 7;
   return (class == APEX_HW_CLASS_R8 || class == APEX_HW_CLASS_RG8 || class == APEX_HW_CLASS_RGBA8) &&
          (type == APEX_HW_UNORM || type == APEX_HW_SRGB);
}

struct output_conversion {
   uint32_t targets;     /* color locations taking UNORM16 codes */
};

/* v = round(clamp(c, 0, 1) * 65535) in bits 15:0 of each component. */
static bool
convert_output(nir_builder *b, nir_intrinsic_instr *i, void *data)
{
   const struct output_conversion *conversion = data;
   if (i->intrinsic != nir_intrinsic_store_deref)
      return false;
   nir_deref_instr *deref = nir_src_as_deref(i->src[0]);
   nir_variable *var = nir_deref_instr_get_variable(deref);
   if (!var || var->data.mode != nir_var_shader_out || var->data.location < FRAG_RESULT_DATA0 ||
       deref->deref_type != nir_deref_type_var || i->src[1].ssa->bit_size != 32 ||
       !(conversion->targets & BITFIELD_BIT(var->data.location - FRAG_RESULT_DATA0)) ||
       glsl_get_base_type(glsl_without_array(var->type)) != GLSL_TYPE_FLOAT)
      return false;
   b->cursor = nir_before_instr(&i->instr);
   nir_def *codes = nir_f2u32(b, nir_fround_even(b, nir_fmul_imm(b, nir_fsat(b, i->src[1].ssa), 65535.0)));
   nir_src_rewrite(&i->src[1], codes);
   return true;
}

struct distance_inputs {
   nir_def *values;      /* clip then cull distances */
   nir_variable *clip, *cull;
   unsigned clip_count;
};

/* A fragment program's reads of gl_ClipDistance and gl_CullDistance come
 * from the distance varyings. */
static bool
lower_distance_input(nir_builder *b, nir_intrinsic_instr *i, void *data)
{
   const struct distance_inputs *inputs = data;
   if (i->intrinsic != nir_intrinsic_load_deref)
      return false;
   nir_deref_instr *deref = nir_src_as_deref(i->src[0]);
   nir_variable *var = nir_deref_instr_get_variable(deref);
   if (!var || deref->deref_type != nir_deref_type_array || (var != inputs->clip && var != inputs->cull))
      return false;
   b->cursor = nir_before_instr(&i->instr);
   nir_def *index = deref->arr.index.ssa;
   if (var == inputs->cull)
      index = nir_iadd_imm(b, index, inputs->clip_count);
   nir_def_replace(&i->def, nir_vector_extract(b, inputs->values, index));
   return true;
}

static bool
lower_fragment(nir_shader *nir, const struct vk_graphics_pipeline_state *state, struct distances d)
{
   unsigned varyings = d.clip + d.cull;
   NIR_PASS(_, nir, nir_split_per_member_structs);
   NIR_PASS(_, nir, nir_lower_io_vars_to_temporaries, nir_shader_get_entrypoint(nir), nir_var_shader_out);
   struct output_conversion conversion = {0};
   for (unsigned k = 0; state && state->rp && k < state->rp->color_attachment_count; k++)
      if (unorm16_target(state->rp->color_attachment_formats[k]))
         conversion.targets |= BITFIELD_BIT(k);
   if (conversion.targets)
      NIR_PASS(_, nir, nir_shader_intrinsics_pass, convert_output, nir_metadata_control_flow, &conversion);
   if (state && state->ms && state->ms->sample_shading_enable &&
       state->ms->min_sample_shading * state->ms->rasterization_samples > 1.0f)
      nir->info.fs.uses_sample_shading = true;
   if (varyings) {
      nir_function_impl *impl = nir_shader_get_entrypoint(nir);
      nir_builder builder = nir_builder_at(nir_before_impl(impl));
      nir_builder *b = &builder;
      nir_def *values[8];
      for (unsigned v = 0; v < DIV_ROUND_UP(varyings, 4); v++) {
         if (nir_find_variable_with_location(nir, nir_var_shader_in, CLIP_VARYING + v))
            return false;
         nir_variable *in = nir_variable_create(nir, nir_var_shader_in, glsl_vec4_type(), "apex_clip");
         in->data.location = CLIP_VARYING + v;
         nir_def *value = nir_load_var(b, in);
         for (unsigned c = 0; c < 4; c++)
            values[4 * v + c] = nir_channel(b, value, c);
      }
      for (unsigned c = varyings; c < 8; c++)
         values[c] = nir_imm_float(b, 0.0f);
      /* Inputs of gl_ClipDistance and gl_CullDistance, merged like the
       * vertex stage's when present. */
      unsigned clip_inputs = nir->info.clip_distance_array_size;
      const struct distance_inputs inputs = {
         nir_vec(b, values, 8), clip_inputs ? distance_variable(nir, nir_var_shader_in, 0) : NULL,
         nir->info.cull_distance_array_size ? distance_variable(nir, nir_var_shader_in, clip_inputs) : NULL,
         d.clip,
      };
      nir_def *clipped = nir_imm_false(b);
      for (unsigned c = 0; c < d.clip; c++)
         clipped = nir_ior(b, clipped, nir_flt_imm(b, values[c], 0.0f));
      if (d.clip)
         nir_terminate_if(b, clipped);
      nir_progress(true, impl, nir_metadata_none);
      NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_distance_input, nir_metadata_control_flow,
               (void *)&inputs);
   }
   return true;
}

/* ---- Shader objects ------------------------------------------------------ */

static void
apex_shader_destroy(struct vk_device *device, struct vk_shader *vk,
                    const VkAllocationCallbacks *alloc)
{
   struct apex_shader *shader = container_of(vk, struct apex_shader, vk);
   apex_program_finish((struct apex_device *)device, &shader->program);
   apex_program_finish((struct apex_device *)device, &shader->vertex.capture);
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
              const struct vk_graphics_pipeline_state *state, struct distances distances,
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
   nir_shader *capture = NULL;
   if (info->stage == MESA_SHADER_VERTEX) {
      shader->vertex.clip_distances = nir->info.clip_distance_array_size;
      const nir_xfb_info *xfb = nir->xfb_info;
      if (xfb && xfb->output_count) {
         capture = nir_shader_clone(NULL, nir);
         for (unsigned b = 0; b < APEX_XFB_BUFFERS; b++)
            if (xfb->buffers_written & BITFIELD_BIT(b))
               shader->vertex.xfb_strides[b] = xfb->buffers[b].stride;
      }
      if (!lower_vertex(nir, state, distances))
         goto fail;
   } else {
      /* Input attachments fetch at the fragment's pixel and layer, the view
       * under multiview. */
      const nir_input_attachment_options options = {
         .use_view_id_for_layer = state && state->mv && state->mv->view_mask,
      };
      NIR_PASS(_, nir, nir_lower_input_attachments, &options);
      if (!lower_fragment(nir, state, distances))
         goto fail;
   }
   if (!apex_program_lower_resources(&shader->program, nir))
      goto fail;
   result = apex_program_compile(device, &shader->program, nir);
   if (result != VK_SUCCESS)
      goto fail;
   if (info->stage == MESA_SHADER_VERTEX) {
      const uint8_t *header = shader->program.code.data;
      shader->vertex.stride = header[28] | header[29] << 8;
   }
   if (capture) {
      result = VK_ERROR_FEATURE_NOT_PRESENT;
      if (!apex_program_layout(&shader->vertex.capture, info->set_layout_count, info->set_layouts,
                               info->push_constant_range_count, info->push_constant_ranges,
                               VK_SHADER_STAGE_ALL_GRAPHICS))
         goto fail;
      result = build_capture(device, capture, state ? state->vi : NULL, &shader->vertex.capture);
      if (result != VK_SUCCESS)
         goto fail;
      ralloc_free(capture);
   }
   ralloc_free(nir);
   *out = &shader->vk;
   return VK_SUCCESS;
fail:
   ralloc_free(capture);
   ralloc_free(nir);
   apex_shader_destroy(device, &shader->vk, alloc);
   return result;
}

static VkResult
apex_compile(struct vk_device *device, uint32_t count, struct vk_shader_compile_info *infos,
             const struct vk_graphics_pipeline_state *state, const struct vk_features *features,
             const VkAllocationCallbacks *alloc, struct vk_shader **shaders)
{
   /* Clip and cull distances of the vertex stage reach a fragment stage
    * compiled with it as varyings. */
   struct distances distances = {0};
   bool fragment = false;
   for (uint32_t i = 0; i < count; i++) {
      fragment |= infos[i].stage == MESA_SHADER_FRAGMENT;
      if (infos[i].stage == MESA_SHADER_VERTEX) {
         distances.clip = infos[i].nir->info.clip_distance_array_size;
         distances.cull = infos[i].nir->info.cull_distance_array_size;
      }
   }
   if (!fragment || distances.clip + distances.cull > 8)
      distances = (struct distances){0};
   VkResult result = VK_SUCCESS;
   for (uint32_t i = 0; i < count; i++) {
      if (result != VK_SUCCESS) {
         ralloc_free(infos[i].nir);
         shaders[i] = NULL;
         continue;
      }
      result = compile_stage(device, &infos[i], state, distances, alloc, &shaders[i]);
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

/* Everything that specializes a stage: vertex input formats, attachment
 * formats, the view mask and sample shading. */
static void
apex_hash_state(struct vk_physical_device *physical, const struct vk_graphics_pipeline_state *state,
                const struct vk_features *features, VkShaderStageFlags stages, blake3_hash out)
{
   struct mesa_blake3 hash;
   _mesa_blake3_init(&hash);
   if (state && (stages & VK_SHADER_STAGE_VERTEX_BIT) && state->vi)
      _mesa_blake3_update(&hash, state->vi, sizeof(*state->vi));
   if (state && state->mv)
      _mesa_blake3_update(&hash, &state->mv->view_mask, sizeof(state->mv->view_mask));
   if (state && state->rp) {
      if (stages & VK_SHADER_STAGE_FRAGMENT_BIT) {
         _mesa_blake3_update(&hash, state->rp->color_attachment_formats,
                             sizeof(state->rp->color_attachment_formats));
         _mesa_blake3_update(&hash, &state->rp->color_attachment_count,
                             sizeof(state->rp->color_attachment_count));
      }
   }
   if (state && state->ms && (stages & VK_SHADER_STAGE_FRAGMENT_BIT)) {
      _mesa_blake3_update(&hash, &state->ms->rasterization_samples, sizeof(state->ms->rasterization_samples));
      _mesa_blake3_update(&hash, &state->ms->sample_shading_enable, sizeof(state->ms->sample_shading_enable));
      _mesa_blake3_update(&hash, &state->ms->min_sample_shading, sizeof(state->ms->min_sample_shading));
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

bool
apex_program_sample_shading(const struct apex_program *program)
{
   return program->code.data[5] & (1u << 7);
}

bool
apex_program_late_depth(const struct apex_program *program)
{
   return !(program->code.data[5] & 1u);
}

/* ---- Internal programs ---------------------------------------------------- */

/* Internal kernels address their table directly as binding 0 of set 0. */
static bool
lower_root(nir_builder *b, nir_intrinsic_instr *i, void *data)
{
   b->cursor = nir_before_instr(&i->instr);
   switch (i->intrinsic) {
   case nir_intrinsic_vulkan_resource_index:
      nir_def_replace(&i->def, nir_imm_ivec2(b, 0, 0));
      return true;
   case nir_intrinsic_load_vulkan_descriptor:
      nir_def_replace(&i->def, i->src[0].ssa);
      return true;
   default:
      return false;
   }
}

static VkResult
compile_internal(struct apex_device *device, nir_shader *nir, struct apex_program **out)
{
   struct apex_program *program = vk_zalloc(&device->vk.alloc, sizeof(*program), 8,
                                            VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!program) {
      ralloc_free(nir);
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }
   program->table = true;
   VkResult result = VK_ERROR_INITIALIZATION_FAILED;
   if (nir) {
      NIR_PASS(_, nir, nir_shader_intrinsics_pass, lower_root, nir_metadata_control_flow, NULL);
      NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_ssbo, nir_address_format_32bit_index_offset);
      result = apex_program_compile(&device->vk, program, nir);
   }
   ralloc_free(nir);
   if (result != VK_SUCCESS) {
      apex_program_finish(device, program);
      vk_free(&device->vk.alloc, program);
      return result;
   }
   *out = program;
   return VK_SUCCESS;
}

VkResult
apex_internal_program(struct apex_device *device, enum apex_internal which, struct apex_program **out)
{
   mtx_lock(&device->memory_mutex);
   VkResult result = VK_SUCCESS;
   if (!device->internal[which]) {
      static const struct { const uint32_t *code; size_t words; } sources[] = {
         [APEX_INTERNAL_COPY] = {apex_copy_spv, ARRAY_SIZE(apex_copy_spv)},
         [APEX_INTERNAL_CLEAR] = {apex_clear_spv, ARRAY_SIZE(apex_clear_spv)},
         [APEX_INTERNAL_QUERY_COPY] = {apex_querycopy_spv, ARRAY_SIZE(apex_querycopy_spv)},
         [APEX_INTERNAL_ETC2] = {apex_etc2_spv, ARRAY_SIZE(apex_etc2_spv)},
         [APEX_INTERNAL_XFB] = {apex_xfb_spv, ARRAY_SIZE(apex_xfb_spv)},
      };
      struct spirv_capabilities caps = {
         .Shader = true, .Int64 = true, .PhysicalStorageBufferAddresses = true,
      };
      struct spirv_to_nir_options options = apex_get_spirv_options(device->vk.physical,
                                                                   MESA_SHADER_COMPUTE, NULL);
      options.environment = NIR_SPIRV_VULKAN;
      options.capabilities = &caps;
      nir_shader *nir = spirv_to_nir(sources[which].code, sources[which].words, NULL,
                                     MESA_SHADER_COMPUTE, "main", &options, &apex_nir_options);
      result = compile_internal(device, nir, &device->internal[which]);
   }
   *out = device->internal[which];
   mtx_unlock(&device->memory_mutex);
   return result;
}

VkResult
apex_empty_fragment_program(struct apex_device *device, struct apex_program **out)
{
   mtx_lock(&device->memory_mutex);
   VkResult result = VK_SUCCESS;
   if (!device->empty_fragment) {
      nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_FRAGMENT, &apex_nir_options,
                                                     "apex_empty_fragment");
      result = compile_internal(device, b.shader, &device->empty_fragment);
   }
   *out = device->empty_fragment;
   mtx_unlock(&device->memory_mutex);
   return result;
}

void
apex_graphics_finish(struct apex_device *device)
{
   for (unsigned i = 0; i < APEX_INTERNAL_COUNT; i++) {
      if (device->internal[i]) {
         apex_program_finish(device, device->internal[i]);
         vk_free(&device->vk.alloc, device->internal[i]);
         device->internal[i] = NULL;
      }
   }
   if (device->empty_fragment) {
      apex_program_finish(device, device->empty_fragment);
      vk_free(&device->vk.alloc, device->empty_fragment);
      device->empty_fragment = NULL;
   }
}
