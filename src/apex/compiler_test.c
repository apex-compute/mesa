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

static nir_shader *global_shader(unsigned load_alignment, unsigned store_alignment)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &apex_nir_options,
                                                "Apex paired global addresses");
   b.shader->info.workgroup_size[0] = 16;
   b.shader->info.workgroup_size[1] = 1;
   b.shader->info.workgroup_size[2] = 1;
   nir_def *descriptor[4];
   for (unsigned i = 0; i < 4; i++)
      descriptor[i] = nir_load_ssbo(&b, 1, 32, nir_imm_int(&b, 0),
                                    nir_imm_int(&b, 4 * i), .align_mul = 4);
   nir_def *lane = nir_load_local_invocation_index(&b);
   nir_def *offset = nir_ishl_imm(&b, lane, 2);
   nir_def *src = nir_build_addr_iadd(&b, nir_vec2(&b, descriptor[0], descriptor[1]),
                                     nir_address_format_2x32bit_global, nir_var_mem_global,
                                     nir_iadd_imm(&b, offset, 0x20090));
   nir_def *dst = nir_build_addr_iadd(&b, nir_vec2(&b, descriptor[2], descriptor[3]),
                                     nir_address_format_2x32bit_global, nir_var_mem_global,
                                     nir_iadd_imm(&b, offset, 132));
   nir_def *input = nir_load_global_2x32(&b, 1, 32, src, .align_mul = load_alignment);
   nir_def *result = nir_iadd(&b, nir_imul_imm(&b, input, 7), lane);
   nir_store_global_2x32(&b, result, dst, .align_mul = store_alignment);
   return b.shader;
}

static nir_shader *geometry_shader(unsigned x, unsigned y, unsigned z)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &apex_nir_options,
                                                  "Apex multi-wave geometry");
   b.shader->info.workgroup_size[0] = x;
   b.shader->info.workgroup_size[1] = y;
   b.shader->info.workgroup_size[2] = z;
   nir_def *subgroup = nir_load_subgroup_id(&b);
   nir_def *count = nir_load_num_subgroups(&b);
   nir_def *local = nir_load_local_invocation_index(&b);
   nir_store_ssbo(&b, nir_iadd(&b, local, nir_iadd(&b, subgroup, count)),
                  nir_imm_int(&b, 0), nir_imm_int(&b, 0), .align_mul = 4);
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

int main(int argc, char **argv)
{
   CHECK(argc >= 1 && argc <= 3);
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

   const unsigned valid_wide[][3] = {{17, 1, 1}, {16, 2, 1}, {4, 4, 4}, {4, 4, 16}};
   for (unsigned i = 0; i < ARRAY_SIZE(valid_wide); i++) {
      nir = geometry_shader(valid_wide[i][0], valid_wide[i][1], valid_wide[i][2]);
      CHECK(apex_from_nir(nir, &a) == 0);
      ralloc_free(nir);
      unsigned invocations = valid_wide[i][0] * valid_wide[i][1] * valid_wide[i][2];
      CHECK(word(a.data) == 0x32585041 && word(a.data + 4) == 2);
      CHECK(word(a.data + 40) == invocations && !word(a.data + 44));
      CHECK(a.size == 48 + 8 * (size_t)word(a.data + 8));
      bool wave_slot = false;
      for (size_t offset = 48; offset < a.size; offset += 8)
         wave_slot |= a.data[offset] == 0x41 && word(a.data + offset + 4) == 7;
      CHECK(wave_slot);
      apex_compile_result_finish(&a);
   }

   const unsigned invalid_sizes[][3] = {
      {257, 1, 1}, {1, 257, 1}, {1, 1, 65}, {256, 2, 1}, {0, 1, 16},
      {UINT16_MAX, UINT16_MAX, 16},
   };
   for (unsigned i = 0; i < ARRAY_SIZE(invalid_sizes); i++) {
      nir = shader(17);
      for (unsigned axis = 0; axis < 3; axis++)
         nir->info.workgroup_size[axis] = invalid_sizes[i][axis];
      reject(nir, &a, "at most 256 invocations");
   }
   nir = shader(17);
   nir->info.workgroup_size_variable = true;
   reject(nir, &a, "constant local dimensions");
   nir = shader(17);
   nir->info.stage = MESA_SHADER_FRAGMENT;
   reject(nir, &a, "only compute");
   nir = shader(17);
   nir_builder builder = nir_builder_at(nir_after_impl(nir_shader_get_entrypoint(nir)));
   nir_store_ssbo(&builder, nir_imm_int(&builder, 3), nir_imm_int(&builder, 1),
                  nir_imm_int(&builder, 0), .align_mul = 4);
   reject(nir, &a, "store_ssbo");

   const struct apex_op invalid = {.op = 256};
   CHECK(apex_emit(&invalid, 1, 0, 0, 16, &a) != 0);
   CHECK(!a.data && !a.size && !strcmp(a.diagnostic, "opcode overflow"));
   apex_compile_result_finish(&a);
   CHECK(apex_emit(NULL, 0, 128, 32, 16, &a) == 0);
   CHECK(a.data && a.size >= 40 && !a.diagnostic[0]);
   CHECK(word(a.data + 24) == 128 && word(a.data + 28) == 32);
   apex_compile_result_finish(&a);
   apex_compile_result_finish(&b);

   const struct {
      mesa_scope scope;
      nir_variable_mode modes;
      nir_memory_semantics semantics;
      bool supported;
   } barriers[] = {
      {SCOPE_WORKGROUP, nir_var_mem_shared, NIR_MEMORY_ACQUIRE, true},
      {SCOPE_WORKGROUP, nir_var_mem_shared, NIR_MEMORY_RELEASE, true},
      {SCOPE_WORKGROUP, nir_var_mem_shared, NIR_MEMORY_ACQ_REL, true},
      {SCOPE_DEVICE, nir_var_mem_shared, NIR_MEMORY_ACQ_REL, true},
      {SCOPE_WORKGROUP, nir_var_mem_ssbo, NIR_MEMORY_ACQ_REL, true},
      {SCOPE_DEVICE, nir_var_mem_ssbo, NIR_MEMORY_ACQ_REL, true},
      {SCOPE_DEVICE, nir_var_mem_global, NIR_MEMORY_ACQ_REL, true},
      {SCOPE_DEVICE, nir_var_image, NIR_MEMORY_ACQ_REL, false},
      {SCOPE_WORKGROUP, nir_var_mem_shared, NIR_MEMORY_MAKE_AVAILABLE | NIR_MEMORY_RELEASE, false},
      {SCOPE_WORKGROUP, nir_var_mem_shared, NIR_MEMORY_MAKE_VISIBLE | NIR_MEMORY_ACQUIRE, false},
   };
   for (unsigned n = 0; n < ARRAY_SIZE(barriers); n++) {
      nir = shader(17);
      builder = nir_builder_at(nir_after_impl(nir_shader_get_entrypoint(nir)));
      nir_barrier(&builder, .execution_scope = SCOPE_NONE, .memory_scope = barriers[n].scope,
                   .memory_modes = barriers[n].modes, .memory_semantics = barriers[n].semantics);
      if (!barriers[n].supported) {
         reject(nir, &a, "barrier");
         continue;
      }
      CHECK(apex_from_nir(nir, &a) == 0);
      ralloc_free(nir);
      check_binary(&a, 17);
      bool drain = false;
      for (size_t i = 40; i < a.size; i += 8) {
         CHECK(a.data[i] != 7);
         drain |= a.data[i] == 8;
      }
      CHECK(drain);
      apex_compile_result_finish(&a);
   }

   nir = shader(17);
   builder = nir_builder_at(nir_after_impl(nir_shader_get_entrypoint(nir)));
   nir_barrier(&builder, .execution_scope = SCOPE_WORKGROUP, .memory_scope = SCOPE_WORKGROUP,
               .memory_modes = nir_var_mem_shared, .memory_semantics = NIR_MEMORY_ACQ_REL);
   CHECK(apex_from_nir(nir, &a) == 0);
   ralloc_free(nir);
   bool rendezvous = false;
   for (size_t i = 40; i < a.size; i += 8) {
      CHECK(a.data[i] != 8);
      rendezvous |= a.data[i] == 7;
   }
   CHECK(rendezvous);
   apex_compile_result_finish(&a);

   nir = global_shader(4, 4);
   CHECK(apex_from_nir(nir, &a) == 0);
   ralloc_free(nir);
   check_binary(&a, 7);
   if (argc >= 2) {
      FILE *f = fopen(argv[1], "wb");
      CHECK(f && fwrite(a.data, 1, a.size, f) == a.size);
      CHECK(fclose(f) == 0);
   }
   apex_compile_result_finish(&a);
   reject(global_shader(2, 4), &a, "load_global_2x32");
   reject(global_shader(4, 2), &a, "store_global_2x32");

   builder = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &apex_nir_options,
                                             "Apex unsigned saturating subtraction");
   builder.shader->info.workgroup_size[0] = 16;
   builder.shader->info.workgroup_size[1] = builder.shader->info.workgroup_size[2] = 1;
   nir_def *offset = nir_ishl_imm(&builder, nir_load_local_invocation_index(&builder), 2);
   nir_def *left = nir_load_ssbo(&builder, 1, 32, nir_imm_int(&builder, 0), offset, .align_mul = 4);
   nir_def *right = nir_load_ssbo(&builder, 1, 32, nir_imm_int(&builder, 0),
                                 nir_iadd_imm(&builder, offset, 64), .align_mul = 4);
   nir_store_ssbo(&builder, nir_usub_sat(&builder, left, right), nir_imm_int(&builder, 0),
                  nir_iadd_imm(&builder, offset, 128), .align_mul = 4);
   CHECK(apex_from_nir(builder.shader, &a) == 0);
   ralloc_free(builder.shader);
   if (argc == 3) {
      FILE *f = fopen(argv[2], "wb");
      CHECK(f && fwrite(a.data, 1, a.size, f) == a.size);
      CHECK(fclose(f) == 0);
   }
   apex_compile_result_finish(&a);
   glsl_type_singleton_decref();
   puts("PASS Apex in-process compiler ownership, diagnostics, metadata and paired global addresses");
   return 0;
}
