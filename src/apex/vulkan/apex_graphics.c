/* SPDX-License-Identifier: MIT */
/* Graphics shaders as P7 vertex and fragment programs (Docs/isa.md) and the
 * internal compute kernels. Vertex programs start with the compiler's
 * vertex prolog for the pipeline's vertex input state and store the output
 * record; fragment programs export their targets. Fragment
 * programs are specialized on the attachment formats: R8, RG8, RGBA8 and
 * BGRA8 UNORM and sRGB targets take UNORM16 codes (Docs/architecture.md,
 * ROP and tile buffer). Clip then cull distances also pass as varyings 30
 * and 31: the fragment program discards where a clip distance is negative
 * and reads its gl_ClipDistance and gl_CullDistance there. The fragment
 * launch carries no layer, so a fragment program that reads the layer or view
 * index takes it from a flat varying in the lowest slot neither stage uses. */
#include "apex_graphics.h"
#include "apex_hw.h"
#include "apex_job.h"
#include "apex_copy_spv.h"
#include "apex_clear_spv.h"
#include "apex_querycopy_spv.h"
#include "apex_etc2_spv.h"
#include "apex_bc_spv.h"
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

/* What the last stage before rasterization passes the fragment stage beside
 * its varyings: the clip and cull distance counts and the layer's varying
 * slot (0 when the fragment stage reads no layer); the input record slots a
 * geometry shader reads. */
struct link {
   unsigned clip, cull, layer;
   uint64_t records;
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
lower_vertex(nir_shader *nir, const struct vk_graphics_pipeline_state *state, struct link d)
{
   unsigned varyings = d.clip + d.cull;
   NIR_PASS(_, nir, nir_split_per_member_structs);
   NIR_PASS(_, nir, nir_lower_returns);
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_builder builder = nir_builder_at(nir_after_impl(impl));
   nir_builder *b = &builder;
   nir_variable *link = NULL;
   if (d.layer) {
      link = nir_variable_create(nir, nir_var_shader_out, glsl_int_type(), "apex_link_layer");
      link->data.location = d.layer;
      link->data.interpolation = INTERP_MODE_FLAT;
      /* Every store of gl_Layer also stores the link. */
      nir_foreach_block(block, impl) {
         nir_foreach_instr_safe(instr, block) {
            if (instr->type != nir_instr_type_intrinsic)
               continue;
            nir_intrinsic_instr *store = nir_instr_as_intrinsic(instr);
            nir_variable *var = store->intrinsic == nir_intrinsic_store_deref ?
               nir_intrinsic_get_var(store, 0) : NULL;
            if (!var || var->data.mode != nir_var_shader_out || var->data.location != VARYING_SLOT_LAYER)
               continue;
            nir_builder after = nir_builder_at(nir_after_instr(instr));
            nir_store_var(&after, link, store->src[1].ssa, 1);
         }
      }
   }
   if (state && state->mv && state->mv->view_mask) {
      /* gl_ViewIndex becomes load_view_index first. */
      NIR_PASS(_, nir, nir_lower_system_values);
      nir_variable *layer = stage_variable(nir, nir_var_shader_out, VARYING_SLOT_LAYER, glsl_int_type(),
                                           "apex_layer");
      nir_def *view = nir_load_view_index(b);
      nir_store_var(b, layer, view, 1);
      if (link)
         nir_store_var(b, link, view, 1);
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
 * that fit every written buffer. Each runs the vertex prolog and the vertex
 * shader and stores the captured outputs at the buffer's
 * offset plus o times its stride. Strips and fans with primitive restart
 * find primitive p by scanning the indices. */
struct capture {
   const struct apex_program *program;
   const nir_xfb_info *xfb;
   nir_def *flags, *first, *first_instance, *vertex_offset, *draw_index, *indexed;
   nir_def *vertex, *instance, *output, *state;
   /* Input records: the varying slots kept and this vertex's record. */
   uint64_t records;
   nir_def *record;
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
   nir_def *params = job_address(b, program, APEX_DRAW_INDIRECT);
   nir_push_if(b, address_nonzero(b, params));
   nir_def *loaded = load_word(b, address_add(b, params, nir_ishl_imm(b, word, 2)));
   nir_push_else(b, NULL);
   nir_def *direct = job_word(b, program, nir_iadd_imm(b, word, APEX_DRAW_COMMAND));
   nir_pop_if(b, NULL);
   return nir_if_phi(b, loaded, direct);
}

/* The word holding byte offset `byte`, shifted so that byte is its low bits. */
static nir_def *
load_bytes(nir_builder *b, nir_def *address, nir_def *byte)
{
   nir_def *aligned = address_add(b, address, nir_iand_imm(b, byte, ~3u));
   return nir_ushr(b, load_word(b, aligned), nir_ishl_imm(b, nir_iand_imm(b, byte, 3), 3));
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
   nir_push_if(b, nir_uge(b, job_imm(b, ctx->program, APEX_DRAW_INDEX_BYTES), end));
   nir_def *word = load_bytes(b, job_address(b, ctx->program, APEX_DRAW_INDICES), byte);
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

/* Record slot of varying `location` among the slots `mask` keeps: one vec4
 * each, in location order. */
static unsigned
record_slot(uint64_t mask, unsigned location)
{
   return util_bitcount64(mask & BITFIELD64_MASK(location));
}

/* Stores a vertex output the geometry shader reads to its input record. */
static void
record_output(nir_builder *b, struct capture *ctx, nir_intrinsic_instr *i)
{
   nir_src *offset = nir_get_io_offset_src(i);
   if (!nir_src_is_const(*offset) || i->src[0].ssa->bit_size != 32) {
      ctx->invalid = true;
      return;
   }
   unsigned location = nir_intrinsic_io_semantics(i).location + nir_src_as_uint(*offset);
   if (location >= 64 || !(ctx->records & BITFIELD64_BIT(location)))
      return;
   unsigned base = 16 * record_slot(ctx->records, location) + 4 * nir_intrinsic_component(i);
   for (unsigned c = 0; c < i->src[0].ssa->num_components; c++)
      if (nir_intrinsic_write_mask(i) & BITFIELD_BIT(c))
         nir_store_global_2x32(b, nir_channel(b, i->src[0].ssa, c),
                               address_add(b, ctx->record, nir_imm_int(b, base + 4 * c)), .align_mul = 4);
}

static bool
lower_capture(nir_builder *b, nir_intrinsic_instr *i, void *data)
{
   struct capture *ctx = data;
   b->cursor = nir_before_instr(&i->instr);
   nir_def *replacement;
   switch (i->intrinsic) {
   case nir_intrinsic_load_kernel_input:
      /* The prolog's vertex input block address (s22-s23). */
      if (nir_intrinsic_base(i) != 88)
         return false;
      replacement = job_address(b, ctx->program, APEX_DRAW_VERTEX_INPUT);
      break;
   case nir_intrinsic_store_output:
      if (ctx->records)
         record_output(b, ctx, i);
      else
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

/* A draw kernel is a compute kernel of APEX_DRAW_LOCAL invocations. */
static void
compute_kernel(nir_shader *nir)
{
   nir->info.stage = MESA_SHADER_COMPUTE;
   memset(&nir->info.cs, 0, sizeof(nir->info.cs));
   nir->info.workgroup_size[0] = APEX_DRAW_LOCAL;
   nir->info.workgroup_size[1] = nir->info.workgroup_size[2] = 1;
   nir->info.workgroup_size_variable = false;
}

static unsigned
attribute_slots(const struct glsl_type *type, bool bindless)
{
   return glsl_count_attribute_slots(type, false);
}

/* Lowers a vertex shader's I/O and resources and makes it a compute kernel
 * of APEX_DRAW_LOCAL invocations, returning its transform feedback outputs;
 * `program` holds the vertex program's descriptor layout. */
static bool
draw_kernel(nir_shader *nir, const struct apex_vertex_input *vi, struct apex_program *program,
            const nir_xfb_info **xfb)
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
   if (!apex_program_lower_resources(program, nir) || !apex_nir_lower_vertex_input(nir, vi))
      return false;
   compute_kernel(nir);
   *xfb = nir->xfb_info;
   nir->xfb_info = NULL;
   return true;
}

/* The draw's parameters (APEX_DRAW_*) with its vertex or index count and
 * instance count; no instances for a draw the count buffer skips. */
static void
draw_parameters(nir_builder *b, struct capture *ctx, nir_def **count, nir_def **instances)
{
   const struct apex_program *program = ctx->program;
   ctx->flags = job_imm(b, program, APEX_DRAW_FLAGS);
   ctx->indexed = nir_ine_imm(b, nir_iand_imm(b, ctx->flags, APEX_DRAW_FLAG_INDEXED), 0);
   *count = draw_word(b, program, nir_imm_int(b, 0));
   ctx->first = draw_word(b, program, nir_imm_int(b, 2));
   ctx->vertex_offset = draw_word(b, program, nir_imm_int(b, 3));
   ctx->first_instance = draw_word(b, program, nir_bcsel(b, ctx->indexed, nir_imm_int(b, 4), nir_imm_int(b, 3)));
   ctx->draw_index = job_imm(b, program, APEX_DRAW_NUMBER);
   nir_def *count_va = job_address(b, program, APEX_DRAW_COUNT), *unbounded = nir_imm_int(b, ~0);
   nir_push_if(b, address_nonzero(b, count_va));
   nir_def *draws = load_word(b, count_va);
   nir_pop_if(b, NULL);
   draws = nir_if_phi(b, draws, unbounded);
   *instances = nir_bcsel(b, nir_ult(b, ctx->draw_index, draws), draw_word(b, program, nir_imm_int(b, 1)),
                          nir_imm_int(b, 0));
}

/* The vertex for draw position `position`, or false for a restart index. */
static nir_def *
draw_vertex(nir_builder *b, struct capture *ctx, nir_def *position, nir_def **restarted)
{
   nir_def *direct = nir_iadd(b, ctx->first, position), *mask, *no = nir_imm_false(b);
   nir_push_if(b, ctx->indexed);
   nir_def *index = fetch_index(b, ctx, position, &mask);
   nir_def *restart = nir_iand(b, nir_ieq(b, index, mask),
                               nir_ine_imm(b, nir_iand_imm(b, ctx->flags, APEX_DRAW_FLAG_RESTART), 0));
   index = nir_iadd(b, index, ctx->vertex_offset);
   nir_pop_if(b, NULL);
   if (restarted)
      *restarted = nir_if_phi(b, restart, no);
   return nir_if_phi(b, index, direct);
}

/* Loops `body` over invocations o of the dispatch below `total`, striding
 * by the dispatch's invocations; the loop starts at the cursor. */
static nir_loop *
invocation_loop(nir_builder *b, const struct apex_program *program, nir_def *total, nir_def **o)
{
   nir_def *invocations = nir_imul_imm(b, root_word(b, nir_imm_int(b, apex_program_trailer(program))),
                                       APEX_DRAW_LOCAL);
   nir_variable *o_var = nir_local_variable_create(b->impl, glsl_uint_type(), "draw_invocation");
   nir_store_var(b, o_var, nir_iadd(b, nir_imul_imm(b, nir_channel(b, nir_load_workgroup_id(b), 0),
                                                    APEX_DRAW_LOCAL),
                                    nir_load_local_invocation_index(b)), 1);
   nir_loop *loop = nir_push_loop(b);
   *o = nir_load_var(b, o_var);
   nir_break_if(b, nir_uge(b, *o, total));
   nir_store_var(b, o_var, nir_iadd(b, *o, invocations), 1);
   return loop;
}

/* Lowers the kernel's intrinsics; lowering inserts control flow, so they
 * are collected first. */
static void
lower_draw_kernel(nir_shader *nir, struct capture *ctx)
{
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
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
      lower_capture(&lower, *i, ctx);
   util_dynarray_fini(&list);
   nir_progress(true, impl, nir_metadata_none);
   NIR_PASS(_, nir, nir_lower_vars_to_ssa);
}

/* Builds the capture program of a vertex shader from its NIR; `program`
 * holds the vertex program's descriptor layout. */
static VkResult
build_capture(struct vk_device *device, nir_shader *nir, const struct apex_vertex_input *vi,
              struct apex_program *program)
{
   const nir_xfb_info *xfb;
   if (!draw_kernel(nir, vi, program, &xfb))
      return VK_ERROR_FEATURE_NOT_PRESENT;
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_cf_list body;
   nir_cf_extract(&body, nir_before_impl(impl), nir_after_impl(impl));
   nir_builder builder = nir_builder_at(nir_before_impl(impl));
   nir_builder *b = &builder;
   struct capture ctx = {.program = program, .xfb = xfb};
   nir_def *zero = nir_imm_int(b, 0);
   nir_def *count, *instances;
   draw_parameters(b, &ctx, &count, &instances);
   ctx.state = job_address(b, program, APEX_XFB_STATE);
   nir_def *topology = nir_iand_imm(b, nir_ushr_imm(b, ctx.flags, 8), 15);
   nir_def *last = nir_ine_imm(b, nir_iand_imm(b, ctx.flags, APEX_DRAW_FLAG_LAST), 0);
   nir_def *verts = nir_bcsel(b, nir_ieq_imm(b, topology, 0), nir_imm_int(b, 1),
                              nir_bcsel(b, nir_ult_imm(b, topology, 3), nir_imm_int(b, 2), nir_imm_int(b, 3)));
   nir_def *strip = nir_ior(b, nir_ieq_imm(b, topology, 2), nir_uge_imm(b, topology, 4));
   nir_def *restart = nir_iand(b, nir_iand(b, ctx.indexed, strip),
                               nir_ine_imm(b, nir_iand_imm(b, ctx.flags, APEX_DRAW_FLAG_RESTART), 0));
   /* Primitives per instance; none at or past the draw count. */
   nir_def *runs = nir_bcsel(b, strip, nir_isub(b, nir_iadd_imm(b, count, 1), verts), nir_udiv(b, count, verts));
   runs = nir_bcsel(b, nir_ult(b, count, verts), zero, runs);
   nir_push_if(b, restart);
   nir_def *scanned, *unused_start, *unused_ordinal;
   restart_scan(b, &ctx, count, verts, NULL, &scanned, &unused_start, &unused_ordinal);
   nir_pop_if(b, NULL);
   nir_def *primitives = nir_if_phi(b, scanned, runs);
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
   nir_def *outputs = nir_imul(b, fit, verts), *o;
   nir_loop *loop = invocation_loop(b, program, outputs, &o);
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
   ctx.vertex = draw_vertex(b, &ctx, position, NULL);
   ctx.instance = nir_iadd(b, ctx.first_instance, instance);
   ctx.output = o;
   nir_cf_reinsert(&body, b->cursor);
   b->cursor = nir_after_cf_list(&loop->body);
   nir_pop_loop(b, loop);
   nir_progress(true, impl, nir_metadata_none);
   lower_draw_kernel(nir, &ctx);
   if (ctx.invalid)
      return VK_ERROR_FEATURE_NOT_PRESENT;
   return apex_program_compile(device, program, nir, NULL);
}

/* ---- Geometry shading ---------------------------------------------------- */

/* A geometry shader runs as two compute passes ahead of its draw (apex_job.h,
 * APEX_GS_*). The records program, the vertex shader as a kernel, writes the
 * outputs the geometry shader reads for every instance and draw position to
 * the input records, one vec4 per slot in location order. The geometry
 * program runs once per instance, input primitive and invocation, in that
 * order: it reads its primitive's input records, writes each emitted vertex
 * to output record o * max_vertices + n and each emitted primitive, as a list
 * primitive, to the index list; unused list primitives index the culled
 * vertex past the last output record. The draw rasterizes the index list
 * with the pass-through program, which copies an output record to its
 * outputs. */

/* Builds the records program of a vertex shader that feeds a geometry
 * shader reading `records` varying slots. */
static VkResult
build_records(struct vk_device *device, nir_shader *nir, const struct apex_vertex_input *vi,
              uint64_t records, struct apex_program *program)
{
   const nir_xfb_info *xfb;
   if (!draw_kernel(nir, vi, program, &xfb))
      return VK_ERROR_FEATURE_NOT_PRESENT;
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   nir_cf_list body;
   nir_cf_extract(&body, nir_before_impl(impl), nir_after_impl(impl));
   nir_builder builder = nir_builder_at(nir_before_impl(impl));
   nir_builder *b = &builder;
   struct capture ctx = {.program = program, .records = records};
   nir_def *count, *instances, *o;
   draw_parameters(b, &ctx, &count, &instances);
   nir_def *total = nir_umin(b, nir_imul(b, count, instances), job_imm(b, program, APEX_GS_RECORD_CAPACITY));
   nir_def *base = job_address(b, program, APEX_GS_RECORDS);
   nir_loop *loop = invocation_loop(b, program, total, &o);
   nir_def *instance = nir_udiv(b, o, count), *position = nir_isub(b, o, nir_imul(b, instance, count));
   nir_def *restarted;
   ctx.vertex = draw_vertex(b, &ctx, position, &restarted);
   ctx.instance = nir_iadd(b, ctx.first_instance, instance);
   ctx.record = address_add(b, base, nir_imul_imm(b, o, 16 * util_bitcount64(records)));
   nir_push_if(b, nir_inot(b, restarted));
   nir_cf_reinsert(&body, b->cursor);
   nir_pop_if(b, NULL);
   b->cursor = nir_after_cf_list(&loop->body);
   nir_pop_loop(b, loop);
   nir_progress(true, impl, nir_metadata_none);
   lower_draw_kernel(nir, &ctx);
   if (ctx.invalid)
      return VK_ERROR_FEATURE_NOT_PRESENT;
   return apex_program_compile(device, program, nir, NULL);
}

static unsigned
input_vertices(enum mesa_prim prim)
{
   switch (prim) {
   case MESA_PRIM_POINTS: return 1;
   case MESA_PRIM_LINES: return 2;
   case MESA_PRIM_LINES_ADJACENCY: return 4;
   case MESA_PRIM_TRIANGLES_ADJACENCY: return 6;
   default: return 3;
   }
}

static nir_def *
select_vertex(nir_builder *b, nir_def *k, const int table[6], nir_def *base)
{
   nir_def *value = nir_iadd_imm(b, base, table[5]);
   for (int v = 4; v >= 0; v--)
      value = nir_bcsel(b, nir_ieq_imm(b, k, v), nir_iadd_imm(b, base, table[v]), value);
   return value;
}

/* Position from the run's start of input vertex k of primitive j among n
 * primitives, for Vulkan topology `topology` (0-9) in Vulkan's input
 * order. */
static nir_def *
input_position(nir_builder *b, nir_def *topology, nir_def *last, nir_def *j, nir_def *n, nir_def *k)
{
   /* Triangle strips with adjacency: v0, v2 and v4 make the triangle. */
   static const int only[6] = {0, 1, 2, 5, 4, 3}, first[6] = {0, 1, 2, 6, 4, 3};
   static const int middle_odd[6] = {2, -2, 0, 3, 4, 6}, middle_even[6] = {0, -2, 2, 6, 4, 3};
   static const int last_odd[6] = {2, -2, 0, 3, 4, 5}, last_even[6] = {0, -2, 2, 5, 4, 3};
   nir_def *zero = nir_imm_int(b, 0), *two_j = nir_imul_imm(b, j, 2);
   nir_def *odd = nir_ine_imm(b, nir_iand_imm(b, j, 1), 0);
   nir_def *final = nir_ieq(b, j, nir_iadd_imm(b, n, -1));
   nir_def *strip = nir_bcsel(b, final,
      nir_bcsel(b, odd, select_vertex(b, k, last_odd, two_j), select_vertex(b, k, last_even, two_j)),
      nir_bcsel(b, odd, select_vertex(b, k, middle_odd, two_j), select_vertex(b, k, middle_even, two_j)));
   strip = nir_bcsel(b, nir_ieq_imm(b, j, 0),
                     nir_bcsel(b, final, select_vertex(b, k, only, zero), select_vertex(b, k, first, zero)),
                     strip);
   nir_def *value = nir_bcsel(b, nir_ieq_imm(b, topology, 8), nir_iadd(b, nir_imul_imm(b, j, 6), k), strip);
   value = nir_bcsel(b, nir_ieq_imm(b, topology, 7), nir_iadd(b, j, k), value);
   value = nir_bcsel(b, nir_ieq_imm(b, topology, 6), nir_iadd(b, nir_imul_imm(b, j, 4), k), value);
   return nir_bcsel(b, nir_ult_imm(b, topology, 6), topology_vertex(b, topology, last, j, k), value);
}

/* Input primitives of `count` positions of Vulkan topology `topology`. */
static nir_def *
input_primitives(nir_builder *b, nir_def *topology, nir_def *count)
{
   /* Per-topology vertices per list primitive and vertices a strip shares. */
   static const unsigned size[10] = {1, 2, 2, 3, 3, 3, 4, 4, 6, 6};
   static const unsigned shared[10] = {0, 0, 1, 0, 2, 2, 0, 3, 0, 4};
   nir_def *result = nir_imm_int(b, 0);
   for (unsigned t = 0; t < 10; t++) {
      nir_def *n = shared[t] ? nir_bcsel(b, nir_ult_imm(b, count, size[t]), nir_imm_int(b, 0),
                                         nir_udiv_imm(b, nir_iadd_imm(b, count, -(int)shared[t]),
                                                      size[t] - shared[t]))
                             : nir_udiv_imm(b, count, size[t]);
      result = nir_bcsel(b, nir_ieq_imm(b, topology, t), n, result);
   }
   return result;
}

struct geometry {
   struct capture draw;
   uint64_t inputs, outputs;
   unsigned max_vertices, max_primitives, primitive_vertices, invocations;
   nir_def *topology, *last, *start, *j, *primitives, *count, *instance;
   nir_def *records, *output, *list, *vertex_base, *null_index, *primitive_id, *invocation_id;
   nir_variable *current[64], *written;
};

static void
store_list(nir_builder *b, struct geometry *ctx, nir_def *slot, nir_def *index)
{
   nir_store_global_2x32(b, index, address_add(b, ctx->list, nir_ishl_imm(b, slot, 2)), .align_mul = 4);
}

/* The list primitive closing at vertex `v`, the `s`th of its strip. */
static void
emit_primitive(nir_builder *b, struct geometry *ctx, nir_def *v, nir_def *s)
{
   unsigned n = ctx->primitive_vertices;
   nir_push_if(b, nir_uge_imm(b, s, n - 1));
   nir_def *q = nir_load_var(b, ctx->written);
   nir_def *strip_start = nir_iadd(b, ctx->vertex_base, nir_isub(b, v, s));
   nir_def *slot = nir_imul_imm(b, q, n);
   for (unsigned k = 0; k < n; k++) {
      nir_def *offset = n == 3 ? topology_vertex(b, nir_imm_int(b, 4), ctx->last, nir_iadd_imm(b, s, -2),
                                                 nir_imm_int(b, k))
                               : nir_iadd_imm(b, s, k + 1 - n);
      store_list(b, ctx, nir_iadd_imm(b, slot, k), nir_iadd(b, strip_start, offset));
   }
   nir_store_var(b, ctx->written, nir_iadd_imm(b, q, 1), 1);
   nir_pop_if(b, NULL);
}

static bool
lower_geometry(nir_builder *b, nir_intrinsic_instr *i, struct geometry *ctx)
{
   b->cursor = nir_before_instr(&i->instr);
   switch (i->intrinsic) {
   case nir_intrinsic_load_per_vertex_input: {
      nir_src *offset = nir_get_io_offset_src(i);
      unsigned location = nir_intrinsic_io_semantics(i).location;
      if (!nir_src_is_const(*offset) || i->def.bit_size != 32 ||
          location + nir_src_as_uint(*offset) >= 64) {
         ctx->draw.invalid = true;
         return false;
      }
      location += nir_src_as_uint(*offset);
      nir_def *value = nir_imm_zero(b, i->def.num_components, 32);
      if (ctx->inputs & BITFIELD64_BIT(location)) {
         nir_def *position = nir_iadd(b, ctx->start, input_position(b, ctx->topology, ctx->last, ctx->j,
                                                                    ctx->primitives, i->src[0].ssa));
         nir_def *record = nir_iadd(b, nir_imul(b, ctx->instance, ctx->count), position);
         unsigned byte = 16 * record_slot(ctx->inputs, location) + 4 * nir_intrinsic_component(i);
         nir_def *address = address_add(b, ctx->records,
            nir_iadd_imm(b, nir_imul_imm(b, record, 16 * util_bitcount64(ctx->inputs)), byte));
         value = nir_load_global_2x32(b, i->def.num_components, 32, address, .align_mul = 4);
      }
      nir_def_replace(&i->def, value);
      return true;
   }
   case nir_intrinsic_store_output: {
      nir_src *offset = nir_get_io_offset_src(i);
      unsigned location = nir_intrinsic_io_semantics(i).location;
      if (!nir_src_is_const(*offset) || i->src[0].ssa->bit_size != 32 ||
          location + nir_src_as_uint(*offset) >= 64) {
         ctx->draw.invalid = true;
         return false;
      }
      location += nir_src_as_uint(*offset);
      if (ctx->outputs & BITFIELD64_BIT(location)) {
         /* Components land at their place in the slot's vec4. */
         unsigned component = nir_intrinsic_component(i), n = i->src[0].ssa->num_components;
         nir_def *channels[4];
         for (unsigned x = 0; x < 4; x++)
            channels[x] = x >= component && x < component + n ? nir_channel(b, i->src[0].ssa, x - component)
                                                              : nir_imm_int(b, 0);
         nir_store_var(b, ctx->current[location], nir_vec(b, channels, 4),
                       (nir_intrinsic_write_mask(i) << component) & 0xf);
      }
      nir_instr_remove(&i->instr);
      return true;
   }
   case nir_intrinsic_emit_vertex_with_counter: {
      if (nir_intrinsic_stream_id(i)) {
         nir_instr_remove(&i->instr);
         return true;
      }
      nir_def *v = i->src[0].ssa, *s = i->src[1].ssa;
      nir_def *vertex = nir_iadd(b, ctx->vertex_base, v);
      nir_def *record = address_add(b, ctx->output, nir_imul_imm(b, vertex, 16 * util_bitcount64(ctx->outputs)));
      u_foreach_bit64(location, ctx->outputs)
         nir_store_global_2x32(b, nir_load_var(b, ctx->current[location]),
                               address_add(b, record, nir_imm_int(b, 16 * record_slot(ctx->outputs, location))),
                               .align_mul = 4);
      emit_primitive(b, ctx, v, s);
      nir_instr_remove(&i->instr);
      return true;
   }
   case nir_intrinsic_end_primitive_with_counter:
   case nir_intrinsic_set_vertex_and_primitive_count:
      nir_instr_remove(&i->instr);
      return true;
   case nir_intrinsic_load_primitive_id:
      nir_def_replace(&i->def, ctx->primitive_id);
      return true;
   case nir_intrinsic_load_invocation_id:
      nir_def_replace(&i->def, ctx->invocation_id);
      return true;
   default:
      return false;
   }
}

/* The output record slots of a geometry or vertex shader: its outputs but
 * the ones the fixed function reads from the record's first line. */
static uint64_t
output_slots(const nir_shader *nir)
{
   uint64_t slots = 0;
   nir_foreach_variable_with_modes(var, nir, nir_var_shader_out) {
      if (var->data.location >= 64)
         continue;
      unsigned n = var->data.compact ? DIV_ROUND_UP(glsl_get_length(var->type) + var->data.location_frac, 4) :
                   glsl_count_attribute_slots(var->type, false);
      slots |= BITFIELD64_RANGE(var->data.location, MIN2(n, 64 - var->data.location));
   }
   return slots;
}

/* Builds the geometry program of a geometry shader whose inputs come from
 * `inputs` record slots. */
static VkResult
build_geometry(struct vk_device *device, nir_shader *nir, uint64_t inputs, struct apex_program *program,
               struct apex_shader *shader)
{
   struct geometry ctx = {
      .draw.program = program, .inputs = inputs, .outputs = output_slots(nir),
      .max_vertices = MAX2(nir->info.gs.vertices_out, 1), .invocations = MAX2(nir->info.gs.invocations, 1),
   };
   ctx.primitive_vertices = nir->info.gs.output_primitive == MESA_PRIM_POINTS ? 1 :
                            nir->info.gs.output_primitive == MESA_PRIM_LINE_STRIP ? 2 : 3;
   ctx.max_primitives = MAX2(ctx.max_vertices + 1 - ctx.primitive_vertices, 1);
   unsigned vertices = input_vertices(nir->info.gs.input_primitive);
   /* One stream: geometryStreams is not reported. */
   NIR_PASS(_, nir, nir_lower_gs_intrinsics, nir_lower_gs_intrinsics_count_vertices_per_primitive);
   NIR_PASS(_, nir, nir_split_per_member_structs);
   NIR_PASS(_, nir, nir_lower_returns);
   nir_function_impl *impl = nir_shader_get_entrypoint(nir);
   NIR_PASS(_, nir, nir_lower_io_vars_to_temporaries, impl, nir_var_shader_out);
   NIR_PASS(_, nir, nir_lower_global_vars_to_local);
   NIR_PASS(_, nir, nir_split_var_copies);
   NIR_PASS(_, nir, nir_lower_var_copies);
   NIR_PASS(_, nir, nir_lower_vars_to_ssa);
   NIR_PASS(_, nir, nir_lower_indirect_derefs_to_if_else_trees, nir_var_shader_out, UINT32_MAX);
   NIR_PASS(_, nir, nir_lower_system_values);
   nir_foreach_variable_with_modes(var, nir, nir_var_shader_in | nir_var_shader_out)
      var->data.driver_location = var->data.location;
   NIR_PASS(_, nir, nir_lower_io, nir_var_shader_in | nir_var_shader_out, attribute_slots, 0);
   if (!apex_program_lower_resources(program, nir))
      return VK_ERROR_FEATURE_NOT_PRESENT;
   shader->geometry.inputs = inputs;
   shader->geometry.outputs = ctx.outputs;
   shader->geometry.max_vertices = ctx.max_vertices;
   shader->geometry.max_primitives = ctx.max_primitives;
   shader->geometry.primitive_vertices = ctx.primitive_vertices;
   shader->geometry.invocations = ctx.invocations;
   shader->geometry.input_vertices = vertices;
   compute_kernel(nir);

   impl = nir_shader_get_entrypoint(nir);
   nir_cf_list body;
   nir_cf_extract(&body, nir_before_impl(impl), nir_after_impl(impl));
   nir_builder builder = nir_builder_at(nir_before_impl(impl));
   nir_builder *b = &builder;
   nir_def *instances, *o;
   draw_parameters(b, &ctx.draw, &ctx.count, &instances);
   ctx.topology = nir_iand_imm(b, nir_ushr_imm(b, ctx.draw.flags, 8), 15);
   ctx.last = nir_ine_imm(b, nir_iand_imm(b, ctx.draw.flags, APEX_DRAW_FLAG_LAST), 0);
   nir_def *strip = nir_ior(b, nir_ieq_imm(b, ctx.topology, 2), nir_ieq_imm(b, ctx.topology, 4));
   strip = nir_ior(b, strip, nir_ieq_imm(b, ctx.topology, 5));
   nir_def *restart = nir_iand(b, nir_iand(b, ctx.draw.indexed, strip),
                               nir_ine_imm(b, nir_iand_imm(b, ctx.draw.flags, APEX_DRAW_FLAG_RESTART), 0));
   nir_def *zero = nir_imm_int(b, 0), *verts = nir_imm_int(b, vertices);
   nir_def *listed = input_primitives(b, ctx.topology, ctx.count);
   nir_push_if(b, restart);
   nir_def *scanned, *unused_start, *unused_ordinal;
   restart_scan(b, &ctx.draw, ctx.count, verts, NULL, &scanned, &unused_start, &unused_ordinal);
   nir_pop_if(b, NULL);
   nir_def *primitives = nir_if_phi(b, scanned, listed);
   nir_def *per_instance = nir_imul_imm(b, primitives, ctx.invocations);
   nir_def *total = nir_umin(b, nir_imul(b, per_instance, instances), job_imm(b, program, APEX_GS_CAPACITY));
   ctx.records = job_address(b, program, APEX_GS_RECORDS);
   ctx.output = job_address(b, program, APEX_GS_OUTPUT);
   ctx.list = job_address(b, program, APEX_GS_LIST);
   ctx.written = nir_local_variable_create(impl, glsl_uint_type(), "geometry_written");
   u_foreach_bit64(location, ctx.outputs)
      ctx.current[location] = nir_local_variable_create(impl, glsl_vec4_type(), "geometry_output");
   /* An indirect draw's arguments: the list of every invocation that fits. */
   nir_def *arguments = job_address(b, program, APEX_GS_ARGUMENTS);
   nir_push_if(b, nir_iand(b, address_nonzero(b, arguments),
                           nir_ieq_imm(b, nir_load_local_invocation_index(b), 0)));
   nir_push_if(b, nir_ieq_imm(b, nir_channel(b, nir_load_workgroup_id(b), 0), 0));
   nir_store_global_2x32(b, nir_vec4(b, nir_imul_imm(b, total, ctx.max_primitives * ctx.primitive_vertices),
                                     nir_imm_int(b, 1), zero, zero),
                         arguments, .align_mul = 4);
   nir_store_global_2x32(b, zero, address_add(b, arguments, nir_imm_int(b, 16)), .align_mul = 4);
   nir_pop_if(b, NULL);
   nir_pop_if(b, NULL);
   nir_loop *loop = invocation_loop(b, program, total, &o);
   ctx.instance = nir_udiv(b, o, per_instance);
   nir_def *rest = nir_isub(b, o, nir_imul(b, ctx.instance, per_instance));
   nir_def *p = nir_udiv_imm(b, rest, ctx.invocations);
   ctx.primitive_id = p;
   ctx.invocation_id = nir_isub(b, rest, nir_imul_imm(b, p, ctx.invocations));
   ctx.primitives = primitives;
   nir_push_if(b, restart);
   nir_def *found, *start, *ordinal;
   restart_scan(b, &ctx.draw, ctx.count, verts, p, &found, &start, &ordinal);
   nir_pop_if(b, NULL);
   ctx.start = nir_if_phi(b, start, zero);
   ctx.j = nir_if_phi(b, ordinal, p);
   ctx.vertex_base = nir_imul_imm(b, o, ctx.max_vertices);
   ctx.list = address_add(b, ctx.list, nir_imul_imm(b, o, 4 * ctx.max_primitives * ctx.primitive_vertices));
   nir_store_var(b, ctx.written, zero, 1);
   nir_cf_reinsert(&body, b->cursor);
   /* Unused list primitives index the culled vertex. */
   nir_def *null_index = nir_imul_imm(b, job_imm(b, program, APEX_GS_CAPACITY), ctx.max_vertices);
   nir_variable *q = nir_local_variable_create(impl, glsl_uint_type(), "geometry_unused");
   nir_store_var(b, q, nir_imul_imm(b, nir_load_var(b, ctx.written), ctx.primitive_vertices), 1);
   nir_push_loop(b);
   nir_def *slot = nir_load_var(b, q);
   nir_break_if(b, nir_uge_imm(b, slot, ctx.max_primitives * ctx.primitive_vertices));
   store_list(b, &ctx, slot, null_index);
   nir_store_var(b, q, nir_iadd_imm(b, slot, 1), 1);
   nir_pop_loop(b, NULL);
   b->cursor = nir_after_cf_list(&loop->body);
   nir_pop_loop(b, loop);
   nir_progress(true, impl, nir_metadata_none);

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
      lower_geometry(&lower, *i, &ctx);
   util_dynarray_fini(&list);
   nir_progress(true, impl, nir_metadata_none);
   NIR_PASS(_, nir, nir_lower_vars_to_ssa);
   if (ctx.draw.invalid)
      return VK_ERROR_FEATURE_NOT_PRESENT;
   return apex_program_compile(device, program, nir, NULL);
}

/* Loads the components of `var` (or its element) from output record slot
 * `location` on, through `deref`. */
static void
copy_record(nir_builder *b, nir_def *record, uint64_t outputs, nir_deref_instr *deref,
            const struct glsl_type *type, unsigned location, unsigned component, bool compact)
{
   if (compact) {
      for (unsigned e = 0; e < glsl_get_length(type); e++) {
         unsigned c = component + e;
         unsigned slot = location + c / 4;
         if (!(outputs & BITFIELD64_BIT(slot)))
            continue;
         nir_def *value = nir_load_global_2x32(b, 1, 32, address_add(b, record,
            nir_imm_int(b, 16 * record_slot(outputs, slot) + 4 * (c % 4))), .align_mul = 4);
         nir_store_deref(b, nir_build_deref_array_imm(b, deref, e), value, 1);
      }
      return;
   }
   if (glsl_type_is_array(type) || glsl_type_is_matrix(type)) {
      const struct glsl_type *element = glsl_get_array_element(type);
      unsigned slots = glsl_count_attribute_slots(element, false);
      for (unsigned e = 0; e < glsl_get_length(type); e++)
         copy_record(b, record, outputs, nir_build_deref_array_imm(b, deref, e), element,
                     location + e * slots, component, false);
      return;
   }
   if (!(outputs & BITFIELD64_BIT(location)) || glsl_get_bit_size(type) != 32)
      return;
   unsigned n = glsl_get_vector_elements(type);
   nir_def *value = nir_load_global_2x32(b, n, 32, address_add(b, record,
      nir_imm_int(b, 16 * record_slot(outputs, location) + 4 * component)), .align_mul = 4);
   nir_store_deref(b, deref, value, BITFIELD_MASK(n));
}

/* The pass-through vertex shader of a geometry shader: it copies output
 * record gl_VertexIndex to outputs declared as the geometry shader's, and
 * culls the vertex past the last record. Job word APEX_GS_OUTPUT and
 * APEX_GS_CAPACITY of its table locate the records. */
static nir_shader *
build_passthrough(const nir_shader *geometry, const struct apex_program *program, uint64_t outputs)
{
   nir_builder builder = nir_builder_init_simple_shader(MESA_SHADER_VERTEX, &apex_nir_options,
                                                        "apex_passthrough");
   nir_builder *b = &builder;
   nir_shader *nir = b->shader;
   nir->info.clip_distance_array_size = geometry->info.clip_distance_array_size;
   nir->info.cull_distance_array_size = geometry->info.cull_distance_array_size;
   nir_def *vertex = nir_load_vertex_id(b);
   nir_def *culled = nir_uge(b, vertex, nir_imul_imm(b, job_imm(b, program, APEX_GS_CAPACITY),
                                                     MAX2(geometry->info.gs.vertices_out, 1)));
   nir_def *record = address_add(b, job_address(b, program, APEX_GS_OUTPUT),
                                 nir_imul_imm(b, nir_bcsel(b, culled, nir_imm_int(b, 0), vertex),
                                              16 * util_bitcount64(outputs)));
   nir_foreach_variable_with_modes(var, geometry, nir_var_shader_out) {
      if (var->data.location == VARYING_SLOT_PRIMITIVE_ID || var->data.location >= 64)
         continue;
      nir_variable *out = nir_variable_clone(var, nir);
      nir_shader_add_variable(nir, out);
      copy_record(b, record, outputs, nir_build_deref_var(b, out), out->type, out->data.location,
                  out->data.location_frac, out->data.compact);
   }
   nir_variable *cull = nir_variable_create(nir, nir_var_shader_out, glsl_int_type(), "apex_cull_primitive");
   cull->data.location = VARYING_SLOT_CULL_PRIMITIVE;
   nir_store_var(b, cull, nir_b2i32(b, culled), 1);
   return nir;
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
lower_fragment(nir_shader *nir, const struct vk_graphics_pipeline_state *state, struct link d)
{
   unsigned varyings = d.clip + d.cull;
   NIR_PASS(_, nir, nir_split_per_member_structs);
   NIR_PASS(_, nir, nir_lower_io_vars_to_temporaries, nir_shader_get_entrypoint(nir), nir_var_shader_out);
   NIR_PASS(_, nir, nir_split_var_copies);
   NIR_PASS(_, nir, nir_lower_var_copies);
   struct output_conversion conversion = {0};
   for (unsigned k = 0; state && state->rp && k < state->rp->color_attachment_count; k++)
      if (unorm16_target(state->rp->color_attachment_formats[k]))
         conversion.targets |= BITFIELD_BIT(k);
   if (conversion.targets)
      NIR_PASS(_, nir, nir_shader_intrinsics_pass, convert_output, nir_metadata_control_flow, &conversion);
   if (state && state->ms && state->ms->sample_shading_enable &&
       state->ms->min_sample_shading * state->ms->rasterization_samples > 1.0f)
      nir->info.fs.uses_sample_shading = true;
   /* The layer and view index come from the link, or are zero. */
   NIR_PASS(_, nir, nir_lower_system_values);
   bool multiview = state && state->mv && state->mv->view_mask;
   nir_def *layer = NULL;
   nir_function_impl *entry = nir_shader_get_entrypoint(nir);
   nir_builder start = nir_builder_at(nir_before_impl(entry));
   if (d.layer) {
      nir_variable *in = nir_variable_create(nir, nir_var_shader_in, glsl_int_type(), "apex_link_layer");
      in->data.location = d.layer;
      in->data.interpolation = INTERP_MODE_FLAT;
      layer = nir_load_var(&start, in);
   } else {
      layer = nir_imm_int(&start, 0);
   }
   nir_def *view = multiview ? layer : nir_imm_int(&start, 0);
   nir_foreach_block(block, entry) {
      nir_foreach_instr_safe(instr, block) {
         if (instr->type != nir_instr_type_intrinsic)
            continue;
         nir_intrinsic_instr *i = nir_instr_as_intrinsic(instr);
         if (i->intrinsic == nir_intrinsic_load_layer_id)
            nir_def_replace(&i->def, layer);
         else if (i->intrinsic == nir_intrinsic_load_view_index)
            nir_def_replace(&i->def, view);
      }
   }
   nir_progress(true, entry, nir_metadata_control_flow);
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
   apex_program_finish((struct apex_device *)device, &shader->geometry.draw);
   for (unsigned s = 0; s < ARRAY_SIZE(shader->set_layouts); s++)
      if (shader->set_layouts[s])
         vk_descriptor_set_layout_unref(device, shader->set_layouts[s]);
   vk_shader_free(device, alloc, vk);
}

/* ---- Serialization -------------------------------------------------------
 * A cached shader holds its programs and the set layouts they read: the
 * program code and table metadata, then each layout's bindings and immutable
 * samplers, with the hash that matches it to bound sets. */

static unsigned
layout_samplers(const struct apex_set_layout *layout)
{
   unsigned samplers = 0;
   for (unsigned b = 0; b < layout->binding_count; b++)
      if (layout->bindings[b].immutable != ~0u)
         samplers = MAX2(samplers, layout->bindings[b].immutable + layout->bindings[b].count);
   return samplers;
}

static void
write_layout(struct blob *blob, const struct apex_set_layout *layout)
{
   unsigned samplers = layout_samplers(layout);
   blob_write_uint32(blob, layout->vk.flags);
   blob_write_bytes(blob, layout->vk.blake3, sizeof(layout->vk.blake3));
   blob_write_uint32(blob, layout->vk.dynamic_descriptor_count);
   blob_write_uint32(blob, layout->binding_count);
   blob_write_uint32(blob, samplers);
   blob_write_uint32(blob, layout->row_count);
   blob_write_uint32(blob, layout->slot_count);
   blob_write_uint8(blob, layout->pool_memory);
   blob_write_bytes(blob, layout->counts, sizeof(layout->counts));
   blob_write_bytes(blob, layout->bindings, layout->binding_count * sizeof(layout->bindings[0]));
   blob_write_bytes(blob, layout->samplers, samplers * sizeof(layout->samplers[0]));
}

static struct apex_set_layout *
read_layout(struct vk_device *device, struct blob_reader *blob)
{
   VkDescriptorSetLayoutCreateFlags flags = blob_read_uint32(blob);
   blake3_hash hash;
   blob_copy_bytes(blob, hash, sizeof(hash));
   uint32_t dynamic = blob_read_uint32(blob);
   uint32_t bindings = blob_read_uint32(blob), samplers = blob_read_uint32(blob);
   if (blob->overrun || bindings > APEX_MAX_BINDINGS || samplers > APEX_MAX_SET_ROWS)
      return NULL;
   struct apex_set_layout *layout = apex_set_layout_alloc((struct apex_device *)device, flags,
                                                          bindings, samplers);
   if (!layout)
      return NULL;
   memcpy(layout->vk.blake3, hash, sizeof(hash));
   layout->vk.dynamic_descriptor_count = dynamic;
   layout->row_count = blob_read_uint32(blob);
   layout->slot_count = blob_read_uint32(blob);
   layout->pool_memory = blob_read_uint8(blob);
   blob_copy_bytes(blob, layout->counts, sizeof(layout->counts));
   blob_copy_bytes(blob, layout->bindings, bindings * sizeof(layout->bindings[0]));
   blob_copy_bytes(blob, layout->samplers, samplers * sizeof(layout->samplers[0]));
   return layout;
}

/* The program's layouts are the shader's; it records only their count. */
static void
write_program(struct blob *blob, const struct apex_program *program)
{
   blob_write_uint64(blob, program->code.size);
   blob_write_bytes(blob, program->code.data, program->code.size);
   blob_write_uint32(blob, program->code.instructions);
   blob_write_uint32(blob, program->code.vector);
   blob_write_uint32(blob, program->code.scalar);
   blob_write_uint32(blob, program->code.spills);
   blob_write_uint8(blob, program->table);
   blob_write_uint32(blob, program->set_count);
   blob_write_bytes(blob, program->set_offsets, sizeof(program->set_offsets));
   blob_write_uint32(blob, program->descriptor_count);
   blob_write_uint32(blob, program->push_size);
   blob_write_uint32(blob, program->max_workgroups);
   blob_write_bytes(blob, program->used_descriptors, sizeof(program->used_descriptors));
}

static bool
read_program(struct blob_reader *blob, struct apex_program *program,
             struct vk_descriptor_set_layout *const *layouts)
{
   uint64_t size = blob_read_uint64(blob);
   const void *code = blob_read_bytes(blob, size);
   if (blob->overrun)
      return false;
   apex_compile_result_load(&program->code, code, size);
   program->code.instructions = blob_read_uint32(blob);
   program->code.vector = blob_read_uint32(blob);
   program->code.scalar = blob_read_uint32(blob);
   program->code.spills = blob_read_uint32(blob);
   program->table = blob_read_uint8(blob);
   program->set_count = blob_read_uint32(blob);
   blob_copy_bytes(blob, program->set_offsets, sizeof(program->set_offsets));
   program->descriptor_count = blob_read_uint32(blob);
   program->push_size = blob_read_uint32(blob);
   program->max_workgroups = blob_read_uint32(blob);
   blob_copy_bytes(blob, program->used_descriptors, sizeof(program->used_descriptors));
   if (blob->overrun || program->set_count > MESA_VK_MAX_DESCRIPTOR_SETS)
      return false;
   for (unsigned s = 0; s < program->set_count; s++)
      program->set_layouts[s] = (const void *)layouts[s];
   return true;
}

static bool
apex_shader_serialize(struct vk_device *device, const struct vk_shader *vk, struct blob *blob)
{
   const struct apex_shader *shader = container_of(vk, struct apex_shader, vk);
   blob_write_uint32(blob, vk->stage);
   for (unsigned s = 0; s < ARRAY_SIZE(shader->set_layouts); s++) {
      blob_write_uint8(blob, shader->set_layouts[s] != NULL);
      if (shader->set_layouts[s])
         write_layout(blob, (const void *)shader->set_layouts[s]);
   }
   write_program(blob, &shader->program);
   blob_write_uint32(blob, shader->vertex.stride);
   blob_write_uint8(blob, shader->vertex.clip_distances);
   blob_write_uint8(blob, shader->vertex.robust);
   blob_write_bytes(blob, shader->vertex.xfb_strides, sizeof(shader->vertex.xfb_strides));
   write_program(blob, &shader->vertex.capture);
   blob_write_uint64(blob, shader->vertex.records);
   write_program(blob, &shader->geometry.draw);
   blob_write_uint32(blob, shader->geometry.stride);
   blob_write_uint8(blob, shader->geometry.clip_distances);
   blob_write_uint64(blob, shader->geometry.inputs);
   blob_write_uint64(blob, shader->geometry.outputs);
   blob_write_uint16(blob, shader->geometry.max_vertices);
   blob_write_uint16(blob, shader->geometry.max_primitives);
   blob_write_uint16(blob, shader->geometry.invocations);
   blob_write_uint8(blob, shader->geometry.primitive_vertices);
   blob_write_uint8(blob, shader->geometry.input_vertices);
   return !blob->out_of_memory;
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

/* The vertex prolog's input state; false when the pipeline's vertex input
 * state is dynamic. */
static bool
vertex_input(const struct vk_graphics_pipeline_state *state, const struct vk_pipeline_robustness_state *rs,
             struct apex_vertex_input *out)
{
   memset(out, 0, sizeof(*out));
   if (!state || !state->vi)
      return false;
   const struct vk_vertex_input_state *vi = state->vi;
   u_foreach_bit(a, vi->attributes_valid & BITFIELD_MASK(APEX_HW_MAX_ATTRIBUTES))
      out->attributes[a] = (struct apex_vertex_attribute){
         vk_format_to_pipe_format(vi->attributes[a].format), vi->attributes[a].binding, vi->attributes[a].offset,
      };
   u_foreach_bit(b, vi->bindings_valid & BITFIELD_MASK(APEX_HW_MAX_BINDINGS))
      out->bindings[b] = (struct apex_vertex_binding){
         vi->bindings[b].input_rate == VK_VERTEX_INPUT_RATE_INSTANCE, vi->bindings[b].divisor,
      };
   out->robust = rs && rs->vertex_inputs != VK_PIPELINE_ROBUSTNESS_BUFFER_BEHAVIOR_DISABLED;
   return true;
}

static VkResult
compile_stage(struct vk_device *device, struct vk_shader_compile_info *info,
              const struct vk_graphics_pipeline_state *state, struct link link,
              const VkAllocationCallbacks *alloc, struct vk_shader **out)
{
   nir_shader *nir = info->nir;
   *out = NULL;
   if (info->stage != MESA_SHADER_VERTEX && info->stage != MESA_SHADER_GEOMETRY &&
       info->stage != MESA_SHADER_FRAGMENT) {
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
   nir_shader *capture = NULL;
   /* All graphics stages share one push image covering every stage's range. */
   if (!apex_program_layout(&shader->program, info->set_layout_count, info->set_layouts,
                            info->push_constant_range_count, info->push_constant_ranges,
                            VK_SHADER_STAGE_ALL_GRAPHICS))
      goto fail;
   struct apex_vertex_input vi;
   if (info->stage == MESA_SHADER_VERTEX && link.records) {
      if (!vertex_input(state, info->robustness, &vi))
         goto fail;
      shader->vertex.robust = vi.robust;
      shader->vertex.records = link.records;
      result = build_records(device, nir, &vi, link.records, &shader->program);
      if (result != VK_SUCCESS)
         goto fail;
      ralloc_free(nir);
      *out = &shader->vk;
      return VK_SUCCESS;
   }
   if (info->stage == MESA_SHADER_GEOMETRY) {
      /* Transform feedback captures vertex shader outputs only. */
      if ((nir->xfb_info && nir->xfb_info->output_count) ||
          !apex_program_layout(&shader->geometry.draw, info->set_layout_count, info->set_layouts,
                               info->push_constant_range_count, info->push_constant_ranges,
                               VK_SHADER_STAGE_ALL_GRAPHICS))
         goto fail;
      nir_shader *draw = build_passthrough(nir, &shader->geometry.draw, output_slots(nir));
      result = build_geometry(device, nir, link.records, &shader->program, shader);
      if (result == VK_SUCCESS) {
         const struct apex_vertex_input none = {0};
         shader->geometry.clip_distances = draw->info.clip_distance_array_size;
         result = lower_vertex(draw, state, link) && apex_program_lower_resources(&shader->geometry.draw, draw) ?
                  apex_program_compile(device, &shader->geometry.draw, draw, &none) :
                  VK_ERROR_FEATURE_NOT_PRESENT;
      }
      ralloc_free(draw);
      if (result != VK_SUCCESS)
         goto fail;
      const uint8_t *header = shader->geometry.draw.code.data;
      shader->geometry.stride = header[28] | header[29] << 8;
      ralloc_free(nir);
      *out = &shader->vk;
      return VK_SUCCESS;
   }
   if (info->stage == MESA_SHADER_VERTEX) {
      if (!vertex_input(state, info->robustness, &vi))
         goto fail;
      shader->vertex.robust = vi.robust;
      shader->vertex.clip_distances = nir->info.clip_distance_array_size;
      const nir_xfb_info *xfb = nir->xfb_info;
      if (xfb && xfb->output_count) {
         capture = nir_shader_clone(NULL, nir);
         for (unsigned b = 0; b < APEX_XFB_BUFFERS; b++)
            if (xfb->buffers_written & BITFIELD_BIT(b))
               shader->vertex.xfb_strides[b] = xfb->buffers[b].stride;
      }
      if (!lower_vertex(nir, state, link))
         goto fail;
   } else {
      /* Input attachments fetch at the fragment's pixel and layer, the view
       * under multiview. */
      const nir_input_attachment_options options = {
         .use_view_id_for_layer = state && state->mv && state->mv->view_mask,
      };
      NIR_PASS(_, nir, nir_lower_input_attachments, &options);
      if (!lower_fragment(nir, state, link))
         goto fail;
   }
   if (!apex_program_lower_resources(&shader->program, nir))
      goto fail;
   result = apex_program_compile(device, &shader->program, nir,
                                 info->stage == MESA_SHADER_VERTEX ? &vi : NULL);
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
      result = build_capture(device, capture, &vi, &shader->vertex.capture);
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

/* The varying slot linking the layer to a fragment stage that reads it, the
 * view index or an input attachment: the lowest of varyings 0-29 that
 * neither stage uses. Zero when the layer is always zero or no slot is free. */
static unsigned
layer_slot(nir_shader *vertex, nir_shader *fragment, bool multiview)
{
   /* Reads are system value variables (SPIR-V) or intrinsics (vk_meta). */
   nir_shader_gather_info(fragment, nir_shader_get_entrypoint(fragment));
   bool reads = BITSET_TEST(fragment->info.system_values_read, SYSTEM_VALUE_LAYER_ID) ||
                BITSET_TEST(fragment->info.system_values_read, SYSTEM_VALUE_VIEW_INDEX);
   nir_foreach_variable_with_modes(var, fragment, nir_var_system_value)
      reads |= var->data.location == SYSTEM_VALUE_LAYER_ID || var->data.location == SYSTEM_VALUE_VIEW_INDEX;
   nir_foreach_variable_with_modes(var, fragment, nir_var_uniform | nir_var_image) {
      enum glsl_sampler_dim dim = glsl_get_sampler_dim(glsl_without_array(var->type));
      reads |= glsl_type_is_image(glsl_without_array(var->type)) &&
               (dim == GLSL_SAMPLER_DIM_SUBPASS || dim == GLSL_SAMPLER_DIM_SUBPASS_MS);
   }
   if (!reads || (!multiview && !nir_find_variable_with_location(vertex, nir_var_shader_out, VARYING_SLOT_LAYER)))
      return 0;
   uint32_t used = 0;
   nir_foreach_variable_with_modes(var, vertex, nir_var_shader_out)
      if (var->data.location >= VARYING_SLOT_VAR0)
         used |= BITFIELD_RANGE(var->data.location - VARYING_SLOT_VAR0,
                                glsl_count_attribute_slots(var->type, false)) & BITFIELD_MASK(30);
   nir_foreach_variable_with_modes(var, fragment, nir_var_shader_in)
      if (var->data.location >= VARYING_SLOT_VAR0)
         used |= BITFIELD_RANGE(var->data.location - VARYING_SLOT_VAR0,
                                glsl_count_attribute_slots(var->type, false)) & BITFIELD_MASK(30);
   unsigned free = ffs(~used & BITFIELD_MASK(30));
   return free ? VARYING_SLOT_VAR0 + free - 1 : 0;
}

static VkResult
apex_compile(struct vk_device *device, uint32_t count, struct vk_shader_compile_info *infos,
             const struct vk_graphics_pipeline_state *state, const struct vk_features *features,
             const VkAllocationCallbacks *alloc, struct vk_shader **shaders)
{
   /* Clip and cull distances of the vertex stage reach a fragment stage
    * compiled with it as varyings. */
   struct link link = {0};
   nir_shader *vertex = NULL, *geometry = NULL, *fragment = NULL;
   for (uint32_t i = 0; i < count; i++) {
      if (infos[i].stage == MESA_SHADER_VERTEX)
         vertex = infos[i].nir;
      if (infos[i].stage == MESA_SHADER_GEOMETRY)
         geometry = infos[i].nir;
      if (infos[i].stage == MESA_SHADER_FRAGMENT)
         fragment = infos[i].nir;
   }
   /* The geometry shader's inputs are its records; its outputs, through the
    * pass-through program, reach the fragment stage. */
   if (geometry) {
      nir_foreach_variable_with_modes(var, geometry, nir_var_shader_in) {
         if (var->data.location >= 64 || !glsl_type_is_array(var->type))
            continue;
         const struct glsl_type *type = glsl_get_array_element(var->type);
         unsigned n = var->data.compact ? DIV_ROUND_UP(glsl_get_length(type) + var->data.location_frac, 4) :
                      glsl_count_attribute_slots(type, false);
         link.records |= BITFIELD64_RANGE(var->data.location, MIN2(n, 64 - var->data.location));
      }
      /* Every record holds a slot. */
      link.records |= BITFIELD64_BIT(VARYING_SLOT_POS);
   }
   nir_shader *last = geometry ? geometry : vertex;
   if (last && fragment) {
      link.clip = last->info.clip_distance_array_size;
      link.cull = last->info.cull_distance_array_size;
      if (link.clip + link.cull > 8)
         link.clip = link.cull = 0;
      link.layer = layer_slot(last, fragment, state && state->mv && state->mv->view_mask);
   }
   VkResult result = VK_SUCCESS;
   for (uint32_t i = 0; i < count; i++) {
      if (result != VK_SUCCESS) {
         ralloc_free(infos[i].nir);
         shaders[i] = NULL;
         continue;
      }
      result = compile_stage(device, &infos[i], state, link, alloc, &shaders[i]);
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
   mesa_shader_stage stage = blob_read_uint32(blob);
   if (blob->overrun || (stage != MESA_SHADER_VERTEX && stage != MESA_SHADER_GEOMETRY &&
                         stage != MESA_SHADER_FRAGMENT))
      return vk_error(device, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);
   struct apex_shader *shader = vk_shader_zalloc(device, &apex_shader_ops, stage, alloc, sizeof(*shader));
   if (!shader)
      return vk_error(device, VK_ERROR_OUT_OF_HOST_MEMORY);
   bool valid = true;
   for (unsigned s = 0; valid && s < ARRAY_SIZE(shader->set_layouts); s++) {
      if (!blob_read_uint8(blob))
         continue;
      struct apex_set_layout *layout = read_layout(device, blob);
      shader->set_layouts[s] = layout ? &layout->vk : NULL;
      valid = layout != NULL;
   }
   valid = valid && read_program(blob, &shader->program, shader->set_layouts);
   shader->vertex.stride = blob_read_uint32(blob);
   shader->vertex.clip_distances = blob_read_uint8(blob);
   shader->vertex.robust = blob_read_uint8(blob);
   blob_copy_bytes(blob, shader->vertex.xfb_strides, sizeof(shader->vertex.xfb_strides));
   valid = valid && read_program(blob, &shader->vertex.capture, shader->set_layouts);
   shader->vertex.records = blob_read_uint64(blob);
   valid = valid && read_program(blob, &shader->geometry.draw, shader->set_layouts);
   shader->geometry.stride = blob_read_uint32(blob);
   shader->geometry.clip_distances = blob_read_uint8(blob);
   shader->geometry.inputs = blob_read_uint64(blob);
   shader->geometry.outputs = blob_read_uint64(blob);
   shader->geometry.max_vertices = blob_read_uint16(blob);
   shader->geometry.max_primitives = blob_read_uint16(blob);
   shader->geometry.invocations = blob_read_uint16(blob);
   shader->geometry.primitive_vertices = blob_read_uint8(blob);
   shader->geometry.input_vertices = blob_read_uint8(blob);
   valid = valid && !blob->overrun;
   if (!valid) {
      apex_shader_destroy(device, &shader->vk, alloc);
      return vk_error(device, VK_ERROR_INCOMPATIBLE_SHADER_BINARY_EXT);
   }
   *out = &shader->vk;
   return VK_SUCCESS;
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
      .ssbo_addr_format = nir_address_format_vec2_index_32bit_offset,
      .ubo_addr_format = nir_address_format_vec2_index_32bit_offset,
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
      result = apex_program_compile(&device->vk, program, nir, NULL);
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
         [APEX_INTERNAL_BC] = {apex_bc_spv, ARRAY_SIZE(apex_bc_spv)},
         [APEX_INTERNAL_XFB] = {apex_xfb_spv, ARRAY_SIZE(apex_xfb_spv)},
      };
      struct spirv_capabilities caps = {
         .Shader = true, .Int64 = true, .PhysicalStorageBufferAddresses = true,
      };
      struct spirv_to_nir_options options = apex_get_spirv_options(device->vk.physical,
                                                                   MESA_SHADER_COMPUTE, NULL);
      /* Internal kernels address their table with one index. */
      options.ssbo_addr_format = nir_address_format_32bit_index_offset;
      options.ubo_addr_format = nir_address_format_32bit_index_offset;
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
