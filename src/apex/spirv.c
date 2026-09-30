/* SPDX-License-Identifier: MIT */
/* Standalone SPIR-V entry for apex-compile and the shader tests; Mesa's
 * Vulkan runtime owns this step for the driver. */
#include "apex.h"
#include "compiler/nir/nir.h"
#include "compiler/spirv/nir_spirv.h"
#include "compiler/spirv/spirv.h"
#include "compiler/spirv/spirv_info.h"
#include <spirv-tools/libspirv.h>
#include <stdio.h>

int apex_compile_spirv(const uint32_t *words, size_t count, struct apex_compile_result *out)
{
   *out = (struct apex_compile_result){0};
   if (count < 5 || words[0] != SpvMagicNumber) {
      snprintf(out->diagnostic, sizeof(out->diagnostic), "not a SPIR-V module");
      return 1;
   }
   spv_context context = spvContextCreate(SPV_ENV_VULKAN_1_3);
   spv_diagnostic diagnostic = NULL;
   /* Vulkan 1.3 layouts: standard uniform and scalar block layouts. */
   spv_validator_options layout = spvValidatorOptionsCreate();
   spvValidatorOptionsSetUniformBufferStandardLayout(layout, true);
   spvValidatorOptionsSetScalarBlockLayout(layout, true);
   spv_const_binary_t binary = {words, count};
   spv_result_t valid = spvValidateWithOptions(context, layout, &binary, &diagnostic);
   spvValidatorOptionsDestroy(layout);
   if (valid != SPV_SUCCESS)
      snprintf(out->diagnostic, sizeof(out->diagnostic), "invalid SPIR-V: %s",
               diagnostic ? diagnostic->error : "");
   spvDiagnosticDestroy(diagnostic);
   spvContextDestroy(context);
   if (valid != SPV_SUCCESS)
      return 1;
   /* The first entry point's execution model selects the stage. */
   mesa_shader_stage stage = MESA_SHADER_COMPUTE;
   for (size_t w = 5; w < count;) {
      unsigned op = words[w] & 0xffff, n = words[w] >> 16;
      if (!n)
         break;
      if (op == SpvOpEntryPoint && w + 1 < count) {
         stage = words[w + 1] == SpvExecutionModelVertex ? MESA_SHADER_VERTEX :
                 words[w + 1] == SpvExecutionModelFragment ? MESA_SHADER_FRAGMENT :
                 MESA_SHADER_COMPUTE;
         break;
      }
      w += n;
   }
   glsl_type_singleton_init_or_ref();
   const struct spirv_capabilities caps = {
      .Shader = true, .GroupNonUniform = true, .GroupNonUniformBallot = true,
      .GroupNonUniformShuffle = true, .ShaderClockKHR = true, .Int64 = true,
      .PhysicalStorageBufferAddresses = true, .DemoteToHelperInvocation = true,
      .GroupNonUniformVote = true, .GroupNonUniformArithmetic = true,
      .GroupNonUniformShuffleRelative = true, .GroupNonUniformClustered = true,
      .GroupNonUniformQuad = true, .ImageQuery = true, .SampledCubeArray = true,
      .DerivativeControl = true, .ClipDistance = true, .CullDistance = true,
      .VulkanMemoryModel = true, .VulkanMemoryModelDeviceScope = true,
      .StorageImageWriteWithoutFormat = true, .StorageImageReadWithoutFormat = true,
      .SampledImageArrayDynamicIndexing = true, .StorageBufferArrayDynamicIndexing = true,
      .UniformBufferArrayDynamicIndexing = true, .ShaderNonUniform = true,
      .RuntimeDescriptorArray = true, .ImageGatherExtended = true, .MinLod = true,
      .Sampled1D = true, .Image1D = true, .MultiView = true,
      .SampleRateShading = true, .InputAttachment = true, .DrawParameters = true,
   };
   const struct spirv_to_nir_options spv = {
      .environment = NIR_SPIRV_VULKAN, .capabilities = &caps,
      .ssbo_addr_format = nir_address_format_32bit_index_offset,
      .ubo_addr_format = nir_address_format_32bit_index_offset,
      .shared_addr_format = nir_address_format_32bit_offset,
      .phys_ssbo_addr_format = nir_address_format_64bit_global,
      .push_const_addr_format = nir_address_format_32bit_offset,
      .skip_os_break_in_debug_build = true,
   };
   nir_shader *nir = spirv_to_nir(words, count, NULL, stage, "main", &spv, &apex_nir_options);
   int result = 1;
   if (nir)
      result = apex_from_nir(nir, out);
   else
      snprintf(out->diagnostic, sizeof(out->diagnostic), "SPIR-V translation failed");
   ralloc_free(nir);
   glsl_type_singleton_decref();
   return result;
}
