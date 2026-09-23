/* SPDX-License-Identifier: MIT */
#include "apex_pipeline.h"
#include "compiler/nir/nir.h"
#include "compiler/nir/nir_builder.h"
#include "compiler/spirv/nir_spirv.h"
#include "vk_device.h"
#include "vk_log.h"

struct descriptor_lowering {
   struct apex_pipeline *pipeline;
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
      if (set >= ctx->pipeline->layout->set_count ||
          (nir_intrinsic_desc_type(i) != nir_descriptor_type_storage_buffer &&
           nir_intrinsic_desc_type(i) != nir_descriptor_type_uniform_buffer)) {
         ctx->invalid = true;
         return false;
      }
      const struct apex_set_layout *layout =
         (const void *)ctx->pipeline->layout->set_layouts[set];
      if (!layout || binding >= layout->binding_count || !layout->bindings[binding].count) {
         ctx->invalid = true;
         return false;
      }
      VkDescriptorType type = nir_intrinsic_desc_type(i) == nir_descriptor_type_uniform_buffer ?
         VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      VkDescriptorType dynamic_type = nir_intrinsic_desc_type(i) == nir_descriptor_type_uniform_buffer ?
         VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
      if (layout->bindings[binding].type != type && layout->bindings[binding].type != dynamic_type) {
         ctx->invalid = true;
         return false;
      }
      nir_def *index = i->src[0].ssa;
      BITSET_SET_COUNT(ctx->pipeline->used_descriptors,
         ctx->pipeline->set_offsets[set] + layout->bindings[binding].offset,
         layout->bindings[binding].count);
      /* Invalid descriptor indices select the zero-filled sentinel row. */
      nir_def *slot = nir_bcsel(b, nir_ult_imm(b, index, layout->bindings[binding].count),
         nir_iadd_imm(b, index, ctx->pipeline->set_offsets[set] + layout->bindings[binding].offset),
         nir_imm_int(b, ctx->pipeline->descriptor_count));
      replacement = nir_vec2(b, slot, nir_imm_int(b, 0));
      break;
   }
   case nir_intrinsic_load_vulkan_descriptor:
      replacement = i->src[0].ssa;
      break;
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
   if (i->intrinsic == nir_intrinsic_vulkan_resource_reindex ||
       (atomic && nir_intrinsic_offset_shift(i))) {
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
   b->cursor = nir_before_instr(&i->instr);
   nir_def *row = nir_imul_imm(b, i->src[store ? 1 : 0].ssa, sizeof(struct apex_buffer_descriptor));
   nir_def *words[3];
   for (unsigned c = 0; c < 3; c++)
      words[c] = nir_load_ssbo(b, 1, 32, nir_imm_int(b, 0), nir_iadd_imm(b, row, c * 4), .align_mul = 4);
   if (size) {
      nir_def_rewrite_uses(&i->def, words[2]);
   } else {
      nir_def *base = nir_vec2(b, words[0], words[1]);
      nir_def *offset = i->src[store ? 2 : 1].ssa;
      nir_def *values[4];
      for (unsigned c = 0; c < i->num_components; c++) {
         if (store && !(nir_intrinsic_write_mask(i) & (1u << c)))
            continue;
         /* Subtract from the range before comparing: offset + width may wrap. */
         nir_def *inside = nir_iand(b, nir_uge_imm(b, words[2], c * 4 + 4),
            nir_uge(b, nir_iadd_imm(b, words[2], -(int)(c * 4 + 4)), offset));
         nir_push_if(b, inside);
         nir_def *address = nir_build_addr_iadd(b, base, nir_address_format_2x32bit_global,
            nir_var_mem_global, nir_iadd_imm(b, offset, c * 4));
         nir_def *loaded = NULL;
         if (store)
            nir_store_global_2x32(b, nir_channel(b, i->src[0].ssa, c), address,
                                 .align_mul = 4, .access = nir_intrinsic_access(i));
         else if (swap)
            loaded = nir_global_atomic_swap_2x32(b, 32, address, i->src[2].ssa, i->src[3].ssa,
               .atomic_op = nir_intrinsic_atomic_op(i), .access = nir_intrinsic_access(i));
         else if (atomic)
            loaded = nir_global_atomic_2x32(b, 32, address, i->src[2].ssa,
               .atomic_op = nir_intrinsic_atomic_op(i), .access = nir_intrinsic_access(i));
         else
            loaded = nir_load_global_2x32(b, 1, 32, address,
                                        .align_mul = 4, .access = nir_intrinsic_access(i));
         nir_push_else(b, NULL);
         nir_def *zero = nir_imm_int(b, 0);
         nir_pop_if(b, NULL);
         if (!store)
            values[c] = nir_if_phi(b, loaded, zero);
      }
      if (!store)
         nir_def_rewrite_uses(&i->def, nir_vec(b, values, i->num_components));
   }
   nir_instr_remove(&i->instr);
   return true;
}

static void
apex_pipeline_destroy(struct vk_device *device, struct vk_pipeline *vk,
                      const VkAllocationCallbacks *alloc)
{
   struct apex_pipeline *pipeline = (struct apex_pipeline *)vk;
   if (pipeline->program.handle)
      apex_bo_finish((struct apex_device *)device, &pipeline->program);
   if (pipeline->layout)
      vk_pipeline_layout_unref(device, pipeline->layout);
   apex_compile_result_finish(&pipeline->code);
   vk_pipeline_free(device, alloc, vk);
}

static bool
lower_push_constant(nir_builder *b, nir_intrinsic_instr *i, void *data)
{
   struct descriptor_lowering *ctx = data;
   if (i->intrinsic != nir_intrinsic_load_push_constant)
      return false;
   unsigned base = nir_intrinsic_base(i), size = ctx->pipeline->push_size;
   if (i->def.bit_size != 32 || i->num_components > 4 ||
       nir_intrinsic_align_mul(i) < 4 || nir_intrinsic_align_offset(i) % 4 ||
       base % 4 || base > size) {
      ctx->invalid = true;
      return false;
   }
   unsigned table_bytes = (ctx->pipeline->descriptor_count + 1) * sizeof(struct apex_buffer_descriptor);
   b->cursor = nir_before_instr(&i->instr);
   nir_def *values[4];
   for (unsigned c = 0; c < i->num_components; c++) {
      unsigned end = base + c * 4 + 4;
      nir_def *inside = end <= size ? nir_ule_imm(b, i->src[0].ssa, size - end) : nir_imm_false(b);
      nir_push_if(b, inside);
      nir_def *value = nir_load_ssbo(b, 1, 32, nir_imm_int(b, 0),
         nir_iadd_imm(b, i->src[0].ssa, table_bytes + base + c * 4), .align_mul = 4);
      nir_push_else(b, NULL);
      nir_def *zero = nir_imm_int(b, 0);
      nir_pop_if(b, NULL);
      values[c] = nir_if_phi(b, value, zero);
   }
   nir_def_rewrite_uses(&i->def, nir_vec(b, values, i->num_components));
   nir_instr_remove(&i->instr);
   return true;
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
      VK_PIPELINE_CREATE_2_EARLY_RETURN_ON_FAILURE_BIT;
   if ((flags & ~supported) || info->stage.stage != VK_SHADER_STAGE_COMPUTE_BIT ||
       info->stage.flags)
      return VK_ERROR_FEATURE_NOT_PRESENT;

   const VkPipelineShaderStageRequiredSubgroupSizeCreateInfo *subgroup =
      vk_find_struct_const(info->stage.pNext,
                           PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO);
   if (subgroup && subgroup->requiredSubgroupSize != 16)
      return VK_ERROR_FEATURE_NOT_PRESENT;

   /* No pipeline cache exists yet, so every accepted shader needs compilation. */
   if (flags & VK_PIPELINE_CREATE_2_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT)
      return VK_PIPELINE_COMPILE_REQUIRED;

   const struct spirv_to_nir_options spirv_options = {
      .environment = NIR_SPIRV_VULKAN,
      .ssbo_addr_format = nir_address_format_32bit_index_offset,
      .ubo_addr_format = nir_address_format_32bit_index_offset,
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
   if (((struct apex_device *)device)->transport == APEX_TRANSPORT_DRM) {
      if (!layout)
         goto unsupported_layout;
      pipeline->layout = vk_pipeline_layout_ref(layout);
      for (unsigned s = 0; s < layout->set_count; s++) {
         const struct apex_set_layout *set = (const void *)layout->set_layouts[s];
         pipeline->set_offsets[s] = pipeline->descriptor_count;
         if (set) pipeline->descriptor_count += set->descriptor_count;
      }
      if (pipeline->descriptor_count > APEX_MAX_DESCRIPTORS)
         goto unsupported_layout;
      for (unsigned r = 0; r < layout->push_range_count; r++) {
         const VkPushConstantRange *range = &layout->push_ranges[r];
         if (!(range->stageFlags & VK_SHADER_STAGE_COMPUTE_BIT))
            continue;
         if (!range->size ||
             range->offset % 4 || range->size % 4 || range->offset >= APEX_MAX_PUSH_CONSTANTS ||
             range->size > APEX_MAX_PUSH_CONSTANTS - range->offset)
            goto unsupported_layout;
         pipeline->push_size = MAX2(pipeline->push_size, range->offset + range->size);
      }
      NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_ssbo | nir_var_mem_ubo,
               nir_address_format_32bit_index_offset);
      NIR_PASS(_, nir, nir_lower_explicit_io, nir_var_mem_push_const, nir_address_format_32bit_offset);
      struct descriptor_lowering ctx = {.pipeline = pipeline};
      nir_shader_intrinsics_pass(nir, lower_resource, nir_metadata_control_flow, &ctx);
      nir_shader_intrinsics_pass(nir, lower_buffer, nir_metadata_none, &ctx);
      nir_shader_intrinsics_pass(nir, lower_push_constant, nir_metadata_none, &ctx);
      if (ctx.invalid)
         goto unsupported_layout;
   }
   int failed = apex_from_nir(nir, &pipeline->code);
   ralloc_free(nir);
   if (failed) {
      result = vk_errorf(device, VK_ERROR_FEATURE_NOT_PRESENT,
                        "Apex compute: %s", pipeline->code.diagnostic);
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
