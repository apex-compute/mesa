/* SPDX-License-Identifier: MIT */
#include "apex_pipeline.h"
#include "compiler/nir/nir.h"
#include "compiler/spirv/nir_spirv.h"
#include "vk_device.h"
#include "vk_log.h"

static void
apex_pipeline_destroy(struct vk_device *device, struct vk_pipeline *vk,
                      const VkAllocationCallbacks *alloc)
{
   struct apex_pipeline *pipeline = (struct apex_pipeline *)vk;
   if (pipeline->program.handle)
      apex_bo_finish((struct apex_device *)device, &pipeline->program);
   apex_compile_result_finish(&pipeline->code);
   vk_pipeline_free(device, alloc, vk);
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
