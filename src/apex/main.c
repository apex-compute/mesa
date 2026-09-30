/* SPDX-License-Identifier: MIT */
#include "apex.h"
#include "util/u_math.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
   if (argc == 4 && argv[1][0] == '-')
      return apex_tool(argv[1], argv[2], argv[3]);
   if (argc != 3) {
      fprintf(stderr, "usage: apex-compile input.spv output.apx\n"
              "       apex-compile --assemble|--disassemble|--validate input output\n");
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
   if (got != (size_t)size) { free(words); return 1; }
   struct apex_compile_result compiled;
   int result = apex_compile_spirv(words, size / 4, &compiled);
   free(words);
   if (result) {
      fprintf(stderr, "apex: %s\n", compiled.diagnostic);
   } else {
      uint32_t h[16];
      memcpy(h, compiled.data, sizeof(h));
      fprintf(stderr, "apex: %u instructions, s%u v%u, %u spills, shared %u, private %u/lane\n",
              compiled.instructions, compiled.scalar, compiled.vector, compiled.spills,
              util_le32_to_cpu(h[5]), util_le32_to_cpu(h[6]));
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
