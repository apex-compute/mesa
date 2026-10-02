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
#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/spirv/nir_spirv.h"
#include "compiler/spirv/spirv_info.h"
#include "vk_alloc.h"
#include "vk_format.h"
#include "vk_graphics_state.h"
#include "vk_log.h"
#include "vk_nir.h"
#include "vk_pipeline.h"
#include "vk_shader.h"

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
   if (info->stage == MESA_SHADER_VERTEX) {
      shader->vertex.clip_distances = nir->info.clip_distance_array_size;
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
