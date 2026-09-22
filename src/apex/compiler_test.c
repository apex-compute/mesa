/* SPDX-License-Identifier: MIT */
#include "apex.h"
#include "compiler/nir/nir_builder.h"
#include "util/u_math.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { \
   fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); abort(); \
} } while (0)

static nir_shader *shader(uint32_t constant)
{
   static const nir_shader_compiler_options options = {0};
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &options,
                                                "Apex in-process compiler test");
   b.shader->info.workgroup_size[0] = 16;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   nir_store_ssbo(&b, nir_imm_int(&b, constant), nir_imm_int(&b, 0),
                  nir_imm_int(&b, 12), .align_mul = 4);
   return b.shader;
}

static uint32_t word(const uint8_t *bytes)
{
   uint32_t value;
   memcpy(&value, bytes, sizeof(value));
   return util_le32_to_cpu(value);
}

static void check_binary(const struct apex_compile_result *r, uint32_t constant)
{
   CHECK(r->data && r->size >= 40 && !r->diagnostic[0]);
   CHECK(word(r->data) == 0x31585041 && word(r->data + 4) == 1);
   CHECK(r->size == 40 + 8 * (size_t)word(r->data + 8));
   CHECK(word(r->data + 32) == 16 && word(r->data + 36) == 4);
   bool immediate = false, store = false;
   for (size_t i = 40; i < r->size; i += 8) {
      immediate |= r->data[i] == 0x20 && word(r->data + i + 4) == constant;
      store |= r->data[i] == 0x51;
   }
   CHECK(immediate && store);
}

static void reject(nir_shader *nir, struct apex_compile_result *result,
                   const char *diagnostic)
{
   CHECK(apex_from_nir(nir, result) != 0);
   ralloc_free(nir);
   CHECK(!result->data && result->size == 0);
   CHECK(strstr(result->diagnostic, diagnostic));
   apex_compile_result_finish(result);
   CHECK(!result->data && !result->size && !result->diagnostic[0]);
}

int main(void)
{
   glsl_type_singleton_init_or_ref();
   struct apex_compile_result a = {0}, b = {0};
   nir_shader *nir = shader(0x13579bdf);
   CHECK(apex_from_nir(nir, &a) == 0);
   ralloc_free(nir);
   nir = shader(0x2468ace0);
   CHECK(apex_from_nir(nir, &b) == 0);
   ralloc_free(nir);
   CHECK(a.data != b.data);
   check_binary(&a, 0x13579bdf);
   check_binary(&b, 0x2468ace0);
   apex_compile_result_finish(&a);
   apex_compile_result_finish(&a);
   CHECK(!a.data && !a.size && !a.diagnostic[0]);
   check_binary(&b, 0x2468ace0);

   nir = shader(17);
   nir->info.workgroup_size[0] = 8;
   reject(nir, &a, "16x1x1");
   nir = shader(17);
   nir->info.stage = MESA_SHADER_FRAGMENT;
   reject(nir, &a, "only compute");
   nir = shader(17);
   nir_builder builder = nir_builder_at(nir_after_impl(nir_shader_get_entrypoint(nir)));
   nir_store_ssbo(&builder, nir_imm_int(&builder, 3), nir_imm_int(&builder, 1),
                  nir_imm_int(&builder, 0), .align_mul = 4);
   reject(nir, &a, "store_ssbo");

   const struct apex_op invalid = {.op = 256};
   CHECK(apex_emit(&invalid, 1, 0, 0, &a) != 0);
   CHECK(!a.data && !a.size && !strcmp(a.diagnostic, "opcode overflow"));
   apex_compile_result_finish(&a);
   CHECK(apex_emit(NULL, 0, 128, 32, &a) == 0);
   CHECK(a.data && a.size >= 40 && !a.diagnostic[0]);
   CHECK(word(a.data + 24) == 128 && word(a.data + 28) == 32);
   apex_compile_result_finish(&a);
   apex_compile_result_finish(&b);
   glsl_type_singleton_decref();
   puts("PASS Apex in-process compiler ownership, diagnostics and metadata");
   return 0;
}
