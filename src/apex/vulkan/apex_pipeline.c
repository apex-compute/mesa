/* SPDX-License-Identifier: MIT */
#include "apex_pipeline.h"
#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/spirv/nir_spirv.h"
#include "vk_device.h"
#include "vk_log.h"
#include "util/log.h"
#include "util/os_misc.h"
#include "util/u_atomic.h"
#include <stdio.h>

struct descriptor_lowering {
   struct apex_program *program;
   bool invalid;
};

static bool
lower_resource(nir_builder *b, nir_intrinsic_instr *i, void *data)
{
   struct descriptor_lowering *ctx = data;
   nir_def *replacement;
   b->cursor = nir_before_instr(&i->instr);
   switch (i->intrinsic) {
   case nir_intrinsic_vulkan_resource_index: {
      unsigned set = nir_intrinsic_desc_set(i), binding = nir_intrinsic_binding(i);
      if (set >= ctx->program->set_count ||
          (nir_intrinsic_desc_type(i) != nir_descriptor_type_storage_buffer &&
           nir_intrinsic_desc_type(i) != nir_descriptor_type_uniform_buffer)) {
         ctx->invalid = true;
         return false;
      }
      const struct apex_set_layout *layout =
         ctx->program->set_layouts[set];
      if (!layout || binding >= layout->binding_count || !layout->bindings[binding].count) {
         ctx->invalid = true;
         return false;
      }
      VkDescriptorType type = nir_intrinsic_desc_type(i) == nir_descriptor_type_uniform_buffer ?
         VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      VkDescriptorType dynamic_type = nir_intrinsic_desc_type(i) == nir_descriptor_type_uniform_buffer ?
         VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
      if (layout->bindings[binding].type != type && layout->bindings[binding].type != dynamic_type &&
          (type != VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER ||
           layout->bindings[binding].type != VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK)) {
         ctx->invalid = true;
         return false;
      }
      nir_def *index = i->src[0].ssa;
      const struct apex_binding_layout *bl = &layout->bindings[binding];
      /* Opaque resource references retain the binding bounds through
       * reindex: a table row and count below 2^15 each, or the row of an
       * update-after-bind binding with bit 31. */
      unsigned base = ctx->program->set_offsets[set] + bl->slot;
      unsigned packed;
      if (apex_binding_in_memory(bl)) {
         BITSET_SET(ctx->program->used_descriptors, base);
         packed = base | 1u << 31;
      } else {
         BITSET_SET_COUNT(ctx->program->used_descriptors, base, bl->count);
         packed = base | bl->count << 16;
      }
      replacement = nir_vec3(b, nir_imm_int(b, packed), index, nir_imm_int(b, 0));
      break;
   }
   case nir_intrinsic_vulkan_resource_reindex:
      replacement = nir_vec3(b, nir_channel(b, i->src[0].ssa, 0),
         nir_iadd(b, nir_channel(b, i->src[0].ssa, 1), i->src[1].ssa), nir_imm_int(b, 0));
      break;
   case nir_intrinsic_load_vulkan_descriptor: {
      /* The descriptor load changes the opaque reference into {table row,
       * element, offset}: element ~0 is the row's own buffer, otherwise the
       * row addresses the element's row in pool memory. */
      nir_def *packed = nir_channel(b, i->src[0].ssa, 0);
      nir_def *index = nir_channel(b, i->src[0].ssa, 1);
      nir_def *row = nir_iand_imm(b, packed, 0x7fff);
      nir_def *direct = nir_vec3(b,
         nir_bcsel(b, nir_ult(b, index, nir_ubfe_imm(b, packed, 16, 15)), nir_iadd(b, row, index),
                   nir_imm_int(b, ctx->program->descriptor_count)),
         nir_imm_int(b, ~0), nir_imm_int(b, 0));
      nir_def *memory = nir_vec3(b, row, nir_umin(b, index, nir_imm_int(b, APEX_MAX_BINDING_ELEMENTS)),
                                 nir_imm_int(b, 0));
      nir_scalar s = nir_scalar_resolved(packed, 0);
      if (nir_scalar_is_const(s))
         replacement = nir_scalar_as_uint(s) >> 31 ? memory : direct;
      else
         replacement = nir_bcsel(b, nir_ige_imm(b, packed, 0), direct, memory);
      break;
   }
   default:
      return false;
   }
   nir_def_rewrite_uses(&i->def, replacement);
   nir_instr_remove(&i->instr);
   return true;
}

static bool
lower_buffer(nir_builder *b, nir_intrinsic_instr *i, void *data)
{
   struct descriptor_lowering *ctx = data;
   bool store = i->intrinsic == nir_intrinsic_store_ssbo;
   bool size = i->intrinsic == nir_intrinsic_get_ssbo_size;
   bool swap = i->intrinsic == nir_intrinsic_ssbo_atomic_swap;
   bool atomic = swap || i->intrinsic == nir_intrinsic_ssbo_atomic;
   if (atomic && nir_intrinsic_offset_shift(i)) {
      ctx->invalid = true;
      return false;
   }
   if (!store && !size && !atomic && i->intrinsic != nir_intrinsic_load_ssbo &&
       i->intrinsic != nir_intrinsic_load_ubo)
      return false;
   if (!size && ((store ? i->src[0].ssa->bit_size : i->def.bit_size) != 32 ||
                 i->num_components > 4 ||
                 (atomic ? i->num_components != 1 :
                  nir_intrinsic_align_mul(i) < 4 || nir_intrinsic_align_offset(i) % 4))) {
      ctx->invalid = true;
      return false;
   }
   nir_src *index = &i->src[store ? 1 : 0];
   /* Driver loads of the table itself (job words) index it directly. */
   if (index->ssa->num_components == 1)
      return false;
   b->cursor = nir_before_instr(&i->instr);
   /* The table row holds the buffer descriptor {GPUVA, bytes, flags}; buffer
    * instructions check each access against it. Every slot, including the
    * null sentinel, is backed independently of the application's range. An
    * update-after-bind element's descriptor is read through the row. */
   const enum gl_access_qualifier access = ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER | ACCESS_CAN_SPECULATE;
   nir_def *row = nir_imul_imm(b, nir_channel(b, index->ssa, 0), sizeof(union apex_descriptor));
   nir_def *descriptor = nir_load_ssbo(b, 4, 32, nir_imm_int(b, 0), row, .align_mul = 16, .access = access);
   nir_def *element = nir_channel(b, index->ssa, 1);
   nir_scalar e = nir_scalar_resolved(element, 0);
   if (!nir_scalar_is_const(e) || nir_scalar_as_uint(e) != ~0u) {
      nir_def *memory = nir_load_ssbo(b, 4, 32, descriptor,
                                      nir_imul_imm(b, element, sizeof(union apex_descriptor)),
                                      .align_mul = 16, .access = access);
      memory = nir_vector_insert_imm(b, memory, nir_imm_int(b, APEX_BUFFER_ROBUST), 3);
      descriptor = nir_scalar_is_const(e) ? memory :
         nir_bcsel(b, nir_ieq_imm(b, element, ~0), descriptor, memory);
   }
   if (size)
      nir_def_replace(&i->def, nir_channel(b, descriptor, 2));
   else
      nir_src_rewrite(index, descriptor);
   return true;
}

/* Image memory is global memory: storage images lower to global access in
 * apex_lower_textures. */
static bool
lower_image(nir_builder *b, nir_intrinsic_instr *i, void *data)
{
   if (i->intrinsic != nir_intrinsic_barrier)
      return false;
   nir_variable_mode modes = nir_intrinsic_memory_modes(i);
   if (!(modes & nir_var_image))
      return false;
   nir_intrinsic_set_memory_modes(i, (modes & ~nir_var_image) | nir_var_mem_global);
   return true;
}

void
apex_program_finish(struct apex_device *device, struct apex_program *program)
{
   if (program->bo.handle)
      apex_bo_finish(device, &program->bo);
   apex_compile_result_finish(&program->code);
}

static void
apex_pipeline_destroy(struct vk_device *device, struct vk_pipeline *vk,
                      const VkAllocationCallbacks *alloc)
{
   struct apex_pipeline *pipeline = (struct apex_pipeline *)vk;
   apex_program_finish((struct apex_device *)device, &pipeline->program);
   if (pipeline->layout)
      vk_pipeline_layout_unref(device, pipeline->layout);
   vk_pipeline_free(device, alloc, vk);
}

/* User data s2:s3 hold the address of the dispatch grid: the table's
 * grid words, or the VkDispatchIndirectCommand of an indirect dispatch. The
 * launch registers carry the workgroup, base included. */
static bool
lower_grid(nir_builder *b, nir_intrinsic_instr *i, void *data)
{
   if (i->intrinsic != nir_intrinsic_load_num_workgroups)
      return false;
   b->cursor = nir_before_instr(&i->instr);
   nir_def *address = nir_load_kernel_input(b, 2, 32, nir_imm_int(b, 0),
                                            .base = APEX_USER_GRID * 4, .range = 8);
   nir_def_replace(&i->def, nir_load_global_2x32(b, 3, 32, address, .align_mul = 4,
      .access = ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER | ACCESS_CAN_SPECULATE));
   return true;
}

static bool
lower_push_constant(nir_builder *b, nir_intrinsic_instr *i, void *data)
{
   struct descriptor_lowering *ctx = data;
   if (i->intrinsic != nir_intrinsic_load_push_constant)
      return false;
   unsigned base = nir_intrinsic_base(i), size = ctx->program->push_size;
   bool aligned = (nir_intrinsic_align_mul(i) >= 4 && !(nir_intrinsic_align_offset(i) % 4)) ||
      (nir_src_is_const(i->src[0]) && !(nir_src_as_uint(i->src[0]) % 4));
   unsigned words = i->def.bit_size / 32;
   if ((i->def.bit_size != 32 && i->def.bit_size != 64) || i->num_components > 4 || !aligned ||
       base % 4 || base > size) {
      ctx->invalid = true;
      return false;
   }
   unsigned table_bytes = (ctx->program->descriptor_count + 1) * sizeof(union apex_descriptor);
   b->cursor = nir_before_instr(&i->instr);
   nir_def *values[8];
   for (unsigned c = 0; c < i->num_components * words; c++) {
      unsigned end = base + c * 4 + 4;
      if (end > size) {
         values[c] = nir_imm_int(b, 0);
         continue;
      }
      /* Clamping keeps the immutable load inside the push range, so it
       * speculates and hoists; words beyond the range read as zero. */
      nir_def *offset = i->src[0].ssa;
      nir_def *value = nir_load_ssbo(b, 1, 32, nir_imm_int(b, 0),
         nir_iadd_imm(b, nir_umin(b, offset, nir_imm_int(b, size - end)), table_bytes + base + c * 4),
         .align_mul = 4, .access = ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER | ACCESS_CAN_SPECULATE);
      values[c] = nir_bcsel(b, nir_ule_imm(b, offset, size - end), value, nir_imm_int(b, 0));
   }
   if (words == 2)
      for (unsigned c = 0; c < i->num_components; c++)
         values[c] = nir_pack_64_2x32(b, nir_vec2(b, values[c * 2], values[c * 2 + 1]));
   nir_def_rewrite_uses(&i->def, nir_vec(b, values, i->num_components));
   nir_instr_remove(&i->instr);
   return true;
}

bool
apex_program_layout(struct apex_program *program, uint32_t set_count,
                    struct vk_descriptor_set_layout *const *sets,
                    uint32_t push_range_count, const VkPushConstantRange *ranges,
                    VkShaderStageFlags stages)
{
   program->table = true;
   program->set_count = set_count;
   for (unsigned s = 0; s < set_count; s++) {
      const struct apex_set_layout *set = (const void *)sets[s];
      program->set_layouts[s] = set;
      program->set_offsets[s] = program->descriptor_count;
      if (set) program->descriptor_count += set->slot_count;
   }
   if (program->descriptor_count > APEX_MAX_DESCRIPTORS)
      return false;
   for (unsigned r = 0; r < push_range_count; r++) {
      const VkPushConstantRange *range = &ranges[r];
      if (!(range->stageFlags & stages))
         continue;
      if (!range->size ||
          range->offset % 4 || range->size % 4 || range->offset >= APEX_MAX_PUSH_CONSTANTS ||
          range->size > APEX_MAX_PUSH_CONSTANTS - range->offset)
         return false;
      program->push_size = MAX2(program->push_size, range->offset + range->size);
   }
   /* A graphics stage sees only its own push range, while the vertex and
    * fragment programs read one table: each reserves the whole push image so
    * both place push data and the trailer alike. */
   if (stages != VK_SHADER_STAGE_COMPUTE_BIT)
      program->push_size = APEX_MAX_PUSH_CONSTANTS;
   return true;
}

/* Buffer accesses become 32-bit components before descriptor lowering, which
 * addresses buffers in dwords: 64-bit values split into pairs. */
static nir_mem_access_size_align
buffer_access_size(nir_intrinsic_op op, uint8_t bytes, uint8_t bit_size, uint32_t align_mul,
                   uint32_t align_offset, bool offset_is_const, enum gl_access_qualifier access,
                   const void *data)
{
   return (nir_mem_access_size_align){
      .num_components = MAX2(MIN2(bytes / 4, 4), 1), .bit_size = 32, .align = 4,
      .shift = nir_mem_access_shift_method_scalar,
   };
}

static bool
sampled_texture(const nir_tex_instr *tex, unsigned src, void *data)
{
   return tex->sampler_dim != GLSL_SAMPLER_DIM_BUF;
}

bool
apex_program_lower_resources(struct apex_program *program, nir_shader *nir)
{
   const nir_lower_tex_options tex = {.lower_txp = ~0u, .lower_tg4_offsets = true};
   NIR_PASS(_, nir, nir_lower_tex, &tex);
   /* The texture unit takes one image and sampler per instruction, so a
    * non-uniform sampled image loops over its distinct descriptors. Texel
    * buffers, storage images and buffers address memory per lane. */
   const nir_lower_non_uniform_access_options non_uniform = {
      .types = nir_lower_non_uniform_texture_access | nir_lower_non_uniform_texture_query,
      .tex_src_callback = sampled_texture,
   };
   NIR_PASS(_, nir, nir_lower_non_uniform_access, &non_uniform);
   /* Readonly buffer reads may reorder, so uniform ones become scalar loads. */
   NIR_PASS(_, nir, nir_opt_access, &(nir_opt_access_options){.is_vulkan = true});
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_ssbo | nir_var_mem_ubo,
            nir_address_format_vec2_index_32bit_offset);
   NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_push_const, nir_address_format_32bit_offset);
   /* Memory is dword-granular: narrower stores become masked atomic updates
    * that leave the neighbouring bytes intact. */
   const nir_lower_mem_access_bit_sizes_options buffer_sizes = {
      .callback = buffer_access_size, .modes = nir_var_mem_ssbo | nir_var_mem_ubo,
      .may_lower_unaligned_stores_to_atomics = true,
   };
   NIR_PASS(_, nir, nir_lower_mem_access_bit_sizes, &buffer_sizes);
   struct descriptor_lowering ctx = {.program = program};
   nir_shader_intrinsics_pass(nir, lower_resource, nir_metadata_control_flow, &ctx);
   nir_shader_intrinsics_pass(nir, lower_buffer, nir_metadata_none, &ctx);
   nir_shader_intrinsics_pass(nir, lower_image, nir_metadata_none, &ctx);
   nir_shader_intrinsics_pass(nir, lower_push_constant, nir_metadata_none, &ctx);
   /* Texture lowering reads table rows directly, so it follows buffer lowering. */
   if (ctx.invalid || !apex_lower_textures(program, nir))
      return false;
   NIR_PASS(_, nir, nir_lower_system_values);
   return true;
}

VkResult
apex_program_compile(struct vk_device *device, struct apex_program *program, nir_shader *nir,
                     const struct apex_vertex_input *vi)
{
   uint64_t invocations = nir->info.workgroup_size[0] * nir->info.workgroup_size[1] *
      nir->info.workgroup_size[2];
   if (apex_from_nir(nir, vi, &program->code))
      return vk_errorf(device, VK_ERROR_FEATURE_NOT_PRESENT,
                       "Apex compile: %s", program->code.diagnostic);
   uint32_t private_bytes;
   memcpy(&private_bytes, program->code.data + 24, sizeof(private_bytes));
   mesa_logd("Apex program: %u instructions, s%u v%u, %u spills, %u private bytes/lane, %u invocations",
             program->code.instructions, program->code.scalar, program->code.vector,
             program->code.spills, util_le32_to_cpu(private_bytes), (unsigned)invocations);
   const char *dump = os_get_option("APEX_DUMP_PROGRAMS");
   if (dump) {
      static unsigned serial;
      char path[512];
      snprintf(path, sizeof(path), "%s/program-%u.apx", dump, p_atomic_inc_return(&serial));
      FILE *f = fopen(path, "wb");
      if (f) {
         fwrite(program->code.data, 1, program->code.size, f);
         fclose(f);
      }
   }
   /* Wave k of a DISPATCH takes private base + 32 k private bytes: the
    * device's private arena bounds the workgroups of one DISPATCH. */
   uint64_t extent = util_le32_to_cpu(private_bytes) * align64(invocations, 32);
   program->max_workgroups = extent ? MIN2(APEX_PRIVATE_BYTES / extent, UINT32_MAX) : UINT32_MAX;
   return program->max_workgroups ? VK_SUCCESS : VK_ERROR_FEATURE_NOT_PRESENT;
}

static const struct vk_pipeline_ops pipeline_ops = {
   .destroy = apex_pipeline_destroy,
};

static VkResult
create_compute_pipeline(struct vk_device *device,
                        const VkComputePipelineCreateInfo *info,
                        const VkAllocationCallbacks *alloc, VkPipeline *out)
{
   VkPipelineCreateFlags2KHR flags = vk_compute_pipeline_create_flags(info);
   const VkPipelineCreateFlags2KHR supported =
      VK_PIPELINE_CREATE_2_DISABLE_OPTIMIZATION_BIT |
      VK_PIPELINE_CREATE_2_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT |
      VK_PIPELINE_CREATE_2_EARLY_RETURN_ON_FAILURE_BIT |
      VK_PIPELINE_CREATE_2_ALLOW_DERIVATIVES_BIT | VK_PIPELINE_CREATE_2_DERIVATIVE_BIT |
      VK_PIPELINE_CREATE_2_DISPATCH_BASE_BIT;
   /* Subgroups are always full and 32 wide. */
   const VkPipelineShaderStageCreateFlags stage_flags =
      VK_PIPELINE_SHADER_STAGE_CREATE_ALLOW_VARYING_SUBGROUP_SIZE_BIT |
      VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
   if ((flags & ~supported) || info->stage.stage != VK_SHADER_STAGE_COMPUTE_BIT ||
       (info->stage.flags & ~stage_flags))
      return VK_ERROR_FEATURE_NOT_PRESENT;

   const VkPipelineShaderStageRequiredSubgroupSizeCreateInfo *subgroup =
      vk_find_struct_const(info->stage.pNext,
                           PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO);
   if (subgroup && subgroup->requiredSubgroupSize != 32)
      return VK_ERROR_FEATURE_NOT_PRESENT;

   /* No pipeline cache exists yet, so every accepted shader needs compilation. */
   if (flags & VK_PIPELINE_CREATE_2_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT)
      return VK_PIPELINE_COMPILE_REQUIRED;

   /* Programs without a layout address buffers as standalone programs do. */
   const nir_address_format buffers = info->layout ? nir_address_format_vec2_index_32bit_offset :
                                                     nir_address_format_32bit_index_offset;
   const struct spirv_to_nir_options spirv_options = {
      .environment = NIR_SPIRV_VULKAN,
      .ssbo_addr_format = buffers,
      .ubo_addr_format = buffers,
      .shared_addr_format = nir_address_format_32bit_offset,
      .skip_os_break_in_debug_build = true,
   };
   nir_shader *nir = NULL;
   VkResult result = vk_pipeline_shader_stage_to_nir(
      device, flags, &info->stage, &spirv_options, &apex_nir_options, NULL, &nir);
   if (result != VK_SUCCESS)
      return result;

   struct apex_pipeline *pipeline = vk_pipeline_zalloc(
      device, &pipeline_ops, VK_PIPELINE_BIND_POINT_COMPUTE, flags,
      alloc, sizeof(*pipeline));
   if (pipeline == NULL) {
      ralloc_free(nir);
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }
   pipeline->vk.stages = VK_SHADER_STAGE_COMPUTE_BIT;
   struct vk_pipeline_layout *layout = vk_pipeline_layout_from_handle(info->layout);
   /* Without a layout the program keeps binding 0 as its raw storage buffer:
    * the RTL benches' fixtures. Submission requires a descriptor table. */
   if (layout) {
      pipeline->layout = vk_pipeline_layout_ref(layout);
      if (!apex_program_layout(&pipeline->program, layout->set_count, layout->set_layouts,
                               layout->push_range_count, layout->push_ranges,
                               VK_SHADER_STAGE_COMPUTE_BIT))
         goto unsupported_layout;
      if (!apex_program_lower_resources(&pipeline->program, nir))
         goto unsupported_layout;
   }
   nir_shader_intrinsics_pass(nir, lower_grid, nir_metadata_control_flow, NULL);
   result = apex_program_compile(device, &pipeline->program, nir, NULL);
   ralloc_free(nir);
   if (result != VK_SUCCESS) {
      apex_pipeline_destroy(device, &pipeline->vk, alloc);
      return result;
   }
   *out = apex_pipeline_to_handle(pipeline);
   return VK_SUCCESS;

unsupported_layout:
   ralloc_free(nir);
   apex_pipeline_destroy(device, &pipeline->vk, alloc);
   return VK_ERROR_FEATURE_NOT_PRESENT;
}

VKAPI_ATTR VkResult VKAPI_CALL
apex_CreateComputePipelines(VkDevice _device, VkPipelineCache cache,
                           uint32_t count,
                           const VkComputePipelineCreateInfo *infos,
                           const VkAllocationCallbacks *alloc,
                           VkPipeline *pipelines)
{
   VK_FROM_HANDLE(vk_device, device, _device);
   VkResult result = VK_SUCCESS;
   for (uint32_t i = 0; i < count; i++)
      pipelines[i] = VK_NULL_HANDLE;
   for (uint32_t i = 0; i < count; i++) {
      VkResult r = create_compute_pipeline(device, &infos[i], alloc, &pipelines[i]);
      if (r == VK_SUCCESS)
         continue;
      /* A later compile-required status must not hide an earlier error. */
      if (result == VK_SUCCESS || r < 0)
         result = r;
      if (vk_compute_pipeline_create_flags(&infos[i]) &
          VK_PIPELINE_CREATE_2_EARLY_RETURN_ON_FAILURE_BIT)
         break;
   }
   return result;
}
