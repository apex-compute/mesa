/* SPDX-License-Identifier: MIT */
#include "apex.h"
#include "compiler/nir/nir.h"
#include "compiler/spirv/nir_spirv.h"
#include "compiler/spirv/spirv.h"
#include "compiler/spirv/spirv_info.h"
#include "util/u_math.h"
#include <spirv-tools/libspirv.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
   if (argc == 4 && argv[1][0] == '-')
      return apex_tool(argv[1], argv[2], argv[3]);
   if (argc != 3) {
      fprintf(stderr, "usage: apex-compile input.spv output.apx\n"
              "       apex-compile --mir|--assemble|--disassemble|--validate input output\n");
      return 1;
   }
   FILE *f = fopen(argv[1], "rb");
   if (!f) return 1;
   if (fseek(f, 0, SEEK_END)) { fclose(f); return 1; }
   long size = ftell(f);
   if (size < 20 || size % 4 || size > 64 * 1024 * 1024) { fclose(f); return 1; }
   rewind(f);
   uint32_t *words = malloc(size);
   if (!words) { fclose(f); return 1; }
   size_t got = fread(words, 1, size, f);
   fclose(f);
   if (got != (size_t)size || words[0] != SpvMagicNumber) { free(words); return 1; }
   spv_context context = spvContextCreate(SPV_ENV_VULKAN_1_1);
   spv_diagnostic diagnostic = NULL;
   spv_result_t valid = spvValidateBinary(context, words, size / 4, &diagnostic);
   if (valid != SPV_SUCCESS) {
      if (diagnostic) fprintf(stderr, "apex: invalid SPIR-V: %s\n", diagnostic->error);
   }
   spvDiagnosticDestroy(diagnostic);
   spvContextDestroy(context);
   if (valid != SPV_SUCCESS) { free(words); return 1; }
   glsl_type_singleton_init_or_ref();
   struct spirv_capabilities caps = {
      .Shader = true, .GroupNonUniform = true, .GroupNonUniformBallot = true,
      .GroupNonUniformShuffle = true, .ShaderClockKHR = true,
   };
   struct nir_shader_compiler_options opts = { .lower_fdiv = true, .lower_flrp32 = true };
   struct spirv_to_nir_options spv = {
      .environment = NIR_SPIRV_VULKAN, .capabilities = &caps,
      .ssbo_addr_format = nir_address_format_32bit_index_offset,
      .ubo_addr_format = nir_address_format_32bit_index_offset,
      .shared_addr_format = nir_address_format_32bit_offset,
      .skip_os_break_in_debug_build = true,
   };
   nir_shader *nir = spirv_to_nir(words, size / 4, NULL, MESA_SHADER_COMPUTE,
                                "main", &spv, &opts);
   free(words);
   struct apex_compile_result compiled = {0};
   int result = nir ? apex_from_nir(nir, &compiled) : 1;
   ralloc_free(nir);
   glsl_type_singleton_decref();
   if (result) {
      if (compiled.diagnostic[0])
         fprintf(stderr, "apex: %s\n", compiled.diagnostic);
   } else {
      uint32_t h[10];
      memcpy(h, compiled.data, sizeof(h));
      fprintf(stderr, "apex: %u instructions, s%u v%u, shared %u, private %u/lane\n",
              util_le32_to_cpu(h[2]), util_le32_to_cpu(h[4]), util_le32_to_cpu(h[5]),
              util_le32_to_cpu(h[6]), util_le32_to_cpu(h[7]));
      f = fopen(argv[2], "wb");
      if (!f) {
         perror(argv[2]);
         result = 1;
      } else {
         bool written = fwrite(compiled.data, 1, compiled.size, f) == compiled.size;
         int closed = fclose(f);
         if (!written || closed) {
            fprintf(stderr, "apex: could not write %s\n", argv[2]);
            result = 1;
         }
      }
   }
   apex_compile_result_finish(&compiled);
   return result;
}
