/* SPDX-License-Identifier: MIT */
/* In-process NIR entry: ownership, diagnostics, header metadata, barriers and
 * paired global addresses, with every compiled program run on the ISA model. */
#include "apex.h"
#include "compiler/nir/nir_builder.h"
#include "util/format/u_formats.h"
#include "util/half_float.h"
#include "util/u_math.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { \
   fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); abort(); \
} } while (0)

#define ROOT 0x100000ull

static uint32_t word(const uint8_t *bytes)
{
   uint32_t value;
   memcpy(&value, bytes, sizeof(value));
   return util_le32_to_cpu(value);
}

static nir_shader *compute(const char *name, unsigned x, unsigned y, unsigned z)
{
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &apex_nir_options, "%s", name);
   b.shader->info.workgroup_size[0] = x;
   b.shader->info.workgroup_size[1] = y;
   b.shader->info.workgroup_size[2] = z;
   return b.shader;
}
static nir_builder builder(nir_shader *s)
{
   return nir_builder_at(nir_after_impl(nir_shader_get_entrypoint(s)));
}

/* Stores a constant through the raw root (SSBO index 0). */
static nir_shader *shader(uint32_t constant)
{
   nir_shader *s = compute("Apex in-process compiler test", 32, 1, 1);
   nir_builder b = builder(s);
   nir_store_ssbo(&b, nir_imm_int(&b, constant), nir_imm_int(&b, 0), nir_imm_int(&b, 12),
                  .align_mul = 4);
   return s;
}

static nir_shader *global_shader(unsigned load_alignment, unsigned store_alignment)
{
   nir_shader *s = compute("Apex paired global addresses", 32, 1, 1);
   nir_builder bb = builder(s), *b = &bb;
   nir_def *descriptor[4];
   for (unsigned i = 0; i < 4; i++)
      descriptor[i] = nir_load_ssbo(b, 1, 32, nir_imm_int(b, 0), nir_imm_int(b, 4 * i), .align_mul = 4);
   nir_def *lane = nir_load_local_invocation_index(b);
   nir_def *offset = nir_ishl_imm(b, lane, 2);
   nir_def *src = nir_build_addr_iadd(b, nir_vec2(b, descriptor[0], descriptor[1]),
                                     nir_address_format_2x32bit_global, nir_var_mem_global,
                                     nir_iadd_imm(b, offset, 0x20090));
   nir_def *dst = nir_build_addr_iadd(b, nir_vec2(b, descriptor[2], descriptor[3]),
                                     nir_address_format_2x32bit_global, nir_var_mem_global,
                                     nir_iadd_imm(b, offset, 132));
   nir_def *input = nir_load_global_2x32(b, 1, 32, src, .align_mul = load_alignment);
   nir_def *result = nir_iadd(b, nir_imul_imm(b, input, 7), lane);
   nir_store_global_2x32(b, result, dst, .align_mul = store_alignment);
   return s;
}

static nir_shader *geometry_shader(unsigned x, unsigned y, unsigned z)
{
   nir_shader *s = compute("Apex multi-wave geometry", x, y, z);
   nir_builder b = builder(s);
   nir_def *subgroup = nir_load_subgroup_id(&b);
   nir_def *count = nir_load_num_subgroups(&b);
   nir_def *local = nir_load_local_invocation_index(&b);
   nir_store_ssbo(&b, nir_iadd(&b, local, nir_iadd(&b, subgroup, nir_ishl_imm(&b, count, 16))),
                  nir_imm_int(&b, 0), nir_ishl_imm(&b, local, 2), .align_mul = 4);
   return s;
}

/* Runs one workgroup with the raw root at ROOT. */
static void run(const struct apex_compile_result *r, uint8_t *root, size_t bytes,
                uint8_t *extra, uint64_t extra_va, size_t extra_bytes)
{
   struct apex_sim_region regions[2] = {{ROOT, root, bytes}, {extra_va, extra, extra_bytes}};
   uint32_t user[16] = {(uint32_t)ROOT}, grid[3] = {1, 1, 1};
   char diagnostic[256] = "";
   if (apex_simulate(r->data, r->size, user, grid, 0, NULL, regions, extra ? 2 : 1, diagnostic)) {
      fprintf(stderr, "simulation: %s\n", diagnostic);
      abort();
   }
}

static void check_binary(const struct apex_compile_result *r, uint32_t constant)
{
   CHECK(r->data && r->size >= 64 && !r->diagnostic[0]);
   CHECK(word(r->data) == 0x50585041 && (word(r->data + 4) & 0xff) == APEX_COMPUTE);
   CHECK(r->size == 64 + (size_t)word(r->data + 8));
   CHECK(r->instructions == word(r->data + 8) / 8 && !r->spills);
   uint8_t root[64] = {0};
   run(r, root, sizeof(root), NULL, 0, 0);
   CHECK(word(root + 12) == constant);
}

static void reject(nir_shader *nir, struct apex_compile_result *result, const char *diagnostic)
{
   CHECK(apex_from_nir(nir, NULL, result) != 0);
   ralloc_free(nir);
   CHECK(!result->data && result->size == 0);
   CHECK(strstr(result->diagnostic, diagnostic));
   apex_compile_result_finish(result);
   CHECK(!result->data && !result->size && !result->diagnostic[0]);
}

static bool has_opcode(const struct apex_compile_result *r, uint8_t op)
{
   for (size_t i = 64; i < r->size; i += 8)
      if (r->data[i] == op)
         return true;
   return false;
}

/* The vertex prolog on the ISA model: unaligned BGRA8, FP16 and SNORM16
 * through dword windows, a packed 10:10:10:2 per-instance attribute with
 * a divisor, a first-instance attribute and a scaled byte; with `robust`,
 * vertices past binding 0's bytes read zero. */
static void test_vertex_prolog(bool robust)
{
   static const struct { uint32_t format, binding, offset; bool integer; } attributes[6] = {
      {PIPE_FORMAT_B8G8R8A8_UNORM, 0, 1}, {PIPE_FORMAT_R16G16_FLOAT, 1, 2},
      {PIPE_FORMAT_R16G16B16_SNORM, 1, 4}, {PIPE_FORMAT_R10G10B10A2_UINT, 2, 0, true},
      {PIPE_FORMAT_R32G32_SINT, 3, 0, true}, {PIPE_FORMAT_R8_SSCALED, 0, 5},
   };
   static const uint32_t strides[4] = {7, 10, 4, 8};
   struct apex_vertex_input vi = {.robust = robust};
   vi.bindings[2] = (struct apex_vertex_binding){true, 3};
   vi.bindings[3] = (struct apex_vertex_binding){true, 0};
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_VERTEX, &apex_nir_options, "Apex vertex prolog");
   nir_variable *pos = nir_variable_create(b.shader, nir_var_shader_out, glsl_vec4_type(), "pos");
   pos->data.location = VARYING_SLOT_POS;
   nir_store_var(&b, pos, nir_imm_vec4(&b, 0, 0, 0, 1), 0xf);
   for (unsigned k = 0; k < 6; k++) {
      vi.attributes[k] = (struct apex_vertex_attribute){attributes[k].format, attributes[k].binding,
                                                        attributes[k].offset};
      const struct glsl_type *type = attributes[k].integer ? glsl_ivec4_type() : glsl_vec4_type();
      nir_variable *in = nir_variable_create(b.shader, nir_var_shader_in, type, "in");
      in->data.location = VERT_ATTRIB_GENERIC0 + k;
      nir_variable *out = nir_variable_create(b.shader, nir_var_shader_out, type, "out");
      out->data.location = VARYING_SLOT_VAR0 + k;
      nir_store_var(&b, out, nir_load_var(&b, in), 0xf);
   }
   struct apex_compile_result r = {0};
   if (apex_from_nir(b.shader, &vi, &r))
      fprintf(stderr, "%s\n", r.diagnostic);
   ralloc_free(b.shader);
   CHECK(r.data && !r.spills);
   const unsigned stride = r.data[28] | r.data[29] << 8, first = 3, base_instance = 5, instance = 12;
   CHECK(stride == 192);
   /* Buffers: bytes, and FP16 words that are normal or zero. */
   uint8_t *data[4], *out = calloc(APEX_LANES, stride), block[512] = {0};
   unsigned bytes[4] = {7 * 40, 10 * 40, 64, 64};
   if (robust)
      bytes[0] = 7 * 20;
   struct apex_sim_region regions[6];
   for (unsigned n = 0; n < 4; n++) {
      data[n] = calloc(1, 512);
      for (unsigned k = 0; k < 512; k++)
         data[n][k] = (uint8_t)(k * 151 + 17 * n + 3);
      if (n == 1)
         for (unsigned k = 0; k < 256; k++) {
            uint16_t h = _mesa_float_to_half(((int)(k % 61) - 30) * 0.375f);
            memcpy(data[n] + 2 * k, &h, 2);
         }
      uint64_t va = 0x400000 + 0x10000 * n;
      regions[n] = (struct apex_sim_region){va, data[n], 512};
      uint32_t d[4] = {(uint32_t)va, va >> 32, align(bytes[n], 4), robust};
      memcpy(block + 16 * n, d, 16);
      memcpy(block + 256 + 4 * n, &strides[n], 4);
   }
   regions[4] = (struct apex_sim_region){0x800000, out, APEX_LANES * stride};
   regions[5] = (struct apex_sim_region){0x700000, block, sizeof(block)};
   struct apex_sim_wave *wave = calloc(1, sizeof(*wave));
   wave->exec = ~0u;
   wave->scalar[16] = 0x800000;
   wave->scalar[19] = first;
   wave->scalar[20] = base_instance;
   wave->scalar[21] = instance;
   wave->scalar[22] = 0x700000;
   for (unsigned l = 0; l < APEX_LANES; l++)
      wave->vector[0][l] = first + l;
   uint32_t user[16] = {0}, grid[3] = {1, 1, 1};
   char diagnostic[256] = "";
   if (apex_simulate(r.data, r.size, user, grid, 0, wave, regions, 6, diagnostic)) {
      fprintf(stderr, "simulation: %s\n", diagnostic);
      abort();
   }
   for (unsigned l = 0; l < APEX_LANES; l++) {
      unsigned v = first + l;
      union { float f; uint32_t u; } want[6][4];
      const uint8_t *a0 = data[0] + 7 * v + 1;
      bool inside = !robust || 7 * v + 5 <= bytes[0];
      for (unsigned c = 0; c < 4; c++)
         want[0][c].f = inside ? (float)a0[(unsigned[]){2, 1, 0, 3}[c]] * (float)(1.0 / 255) : 0.0f;
      for (unsigned c = 0; c < 4; c++) {
         uint16_t h;
         memcpy(&h, data[1] + 10 * v + 2 + 2 * c, 2);
         want[1][c].f = c < 2 ? _mesa_half_to_float(h) : c == 3 ? 1.0f : 0.0f;
         int16_t x;
         memcpy(&x, data[1] + 10 * v + 4 + 2 * c, 2);
         want[2][c].f = c < 3 ? fmaxf((float)x * (float)(1.0 / 32767), -1.0f) : 1.0f;
      }
      uint32_t w;
      memcpy(&w, data[2] + 4 * (base_instance + (instance - base_instance) / 3), 4);
      for (unsigned c = 0; c < 4; c++)
         want[3][c].u = c < 3 ? w >> (10 * c) & 1023 : w >> 30;
      for (unsigned c = 0; c < 4; c++) {
         if (c < 2)
            memcpy(&want[4][c].u, data[3] + 8 * base_instance + 4 * c, 4);
         else
            want[4][c].u = c == 3;
      }
      want[5][0].f = inside ? (float)(int8_t)data[0][7 * v + 5] : 0.0f;
      want[5][1].f = want[5][2].f = 0.0f;
      want[5][3].f = 1.0f;
      for (unsigned k = 0; k < 6; k++)
         for (unsigned c = 0; c < 4; c++) {
            uint32_t got = word(out + l * stride + 4 * (16 + 4 * k + c));
            if (got != want[k][c].u)
               fprintf(stderr, "lane %u attribute %u component %u: 0x%08x, expected 0x%08x\n", l, k, c, got,
                       want[k][c].u);
            CHECK(got == want[k][c].u);
         }
   }
   for (unsigned n = 0; n < 4; n++)
      free(data[n]);
   free(out);
   free(wave);
   apex_compile_result_finish(&r);
}

int main(int argc, char **argv)
{
   CHECK(argc >= 1 && argc <= 3);
   glsl_type_singleton_init_or_ref();
   struct apex_compile_result a = {0}, b = {0};
   nir_shader *nir = shader(0x13579bdf);
   CHECK(apex_from_nir(nir, NULL, &a) == 0);
   ralloc_free(nir);
   nir = shader(0x2468ace0);
   CHECK(apex_from_nir(nir, NULL, &b) == 0);
   ralloc_free(nir);
   CHECK(a.data != b.data);
   check_binary(&a, 0x13579bdf);
   check_binary(&b, 0x2468ace0);
   apex_compile_result_finish(&a);
   apex_compile_result_finish(&a);
   CHECK(!a.data && !a.size && !a.diagnostic[0]);
   check_binary(&b, 0x2468ace0);

   {
      nir_shader *init = compute("Apex private global initializer", 32, 1, 1);
      nir_builder ib = builder(init);
      nir_variable *var = nir_variable_create(init, nir_var_shader_temp, glsl_uint_type(), "initialized");
      var->constant_initializer = rzalloc(init, nir_constant);
      var->constant_initializer->values[0].u32 = 0x2468ace0;
      nir_store_ssbo(&ib, nir_load_var(&ib, var), nir_imm_int(&ib, 0), nir_imm_int(&ib, 12), .align_mul = 4);
      CHECK(apex_from_nir(init, NULL, &a) == 0);
      nir_intrinsic_instr *store = nir_instr_as_intrinsic(nir_block_last_instr(
         nir_start_block(nir_shader_get_entrypoint(init))));
      CHECK(store->intrinsic == nir_intrinsic_store_ssbo);
      CHECK(nir_src_is_const(store->src[0]) && nir_src_as_uint(store->src[0]) == 0x2468ace0);
      CHECK(nir_src_as_uint(store->src[2]) == 12);
      ralloc_free(init);
      check_binary(&a, 0x2468ace0);
      apex_compile_result_finish(&a);
   }

   /* A conditional invariant load hoists only when speculatable; the loop
    * stores loaded + iteration for lanes below the iteration. */
   for (unsigned speculate = 0; speculate < 2; speculate++) {
      nir_shader *s = compute("Apex conditional invariant load", 32, 1, 1);
      nir_builder lb = builder(s), *l = &lb;
      nir_function_impl *impl = nir_shader_get_entrypoint(s);
      nir_variable *index = nir_local_variable_create(impl, glsl_uint_type(), "index");
      nir_def *zero = nir_imm_int(l, 0);
      nir_def *lane = nir_load_local_invocation_index(l);
      /* A loaded trip count keeps the loop from unrolling. */
      nir_def *trips = nir_load_ssbo(l, 1, 32, zero, nir_imm_int(l, 4), .align_mul = 4,
         .access = ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER | ACCESS_CAN_SPECULATE);
      nir_store_var(l, index, zero, 1);
      nir_push_loop(l);
      nir_def *iteration = nir_load_var(l, index);
      nir_push_if(l, nir_uge(l, iteration, trips));
      nir_jump(l, nir_jump_break);
      nir_pop_if(l, NULL);
      nir_push_if(l, nir_ult(l, lane, iteration));
      nir_def *loaded = nir_load_ssbo(l, 1, 32, zero, zero, .align_mul = 4,
         .access = ACCESS_NON_WRITEABLE | ACCESS_CAN_REORDER | (speculate ? ACCESS_CAN_SPECULATE : 0));
      nir_store_ssbo(l, nir_iadd(l, loaded, iteration), zero,
                     nir_iadd_imm(l, nir_ishl_imm(l, lane, 2), 64), .align_mul = 4);
      nir_pop_if(l, NULL);
      nir_store_var(l, index, nir_iadd_imm(l, iteration, 1), 1);
      nir_pop_loop(l, NULL);
      if (apex_from_nir(s, NULL, &a))
         fprintf(stderr, "%s\n", a.diagnostic);
      CHECK(a.data);
      unsigned loads = 0;
      nir_foreach_block(block, impl) {
         nir_foreach_instr(instr, block) {
            if (instr->type != nir_instr_type_intrinsic ||
                nir_instr_as_intrinsic(instr)->intrinsic != nir_intrinsic_load_ssbo ||
                !nir_src_is_const(nir_instr_as_intrinsic(instr)->src[1]) ||
                nir_src_as_uint(nir_instr_as_intrinsic(instr)->src[1]))
               continue;
            loads++;
            CHECK((block == nir_start_block(impl)) == (speculate != 0));
         }
      }
      CHECK(loads == 1);
      ralloc_free(s);
      uint8_t root[192] = {0};
      root[0] = 40;
      root[4] = 3;
      run(&a, root, sizeof(root), NULL, 0, 0);
      for (unsigned k = 0; k < 32; k++)
         CHECK(word(root + 64 + 4 * k) == (k < 2 ? 42 : 0));
      apex_compile_result_finish(&a);
   }

   /* Workgroup shapes: header local size and per-lane launch values. */
   const unsigned valid_wide[][3] = {
      {33, 1, 1}, {16, 2, 1}, {4, 4, 4}, {4, 4, 16}, {256, 1, 1}, {1, 256, 1}, {1, 1, 64}, {3, 1, 1},
   };
   for (unsigned i = 0; i < ARRAY_SIZE(valid_wide); i++) {
      nir = geometry_shader(valid_wide[i][0], valid_wide[i][1], valid_wide[i][2]);
      CHECK(apex_from_nir(nir, NULL, &a) == 0);
      ralloc_free(nir);
      unsigned invocations = valid_wide[i][0] * valid_wide[i][1] * valid_wide[i][2];
      CHECK(word(a.data + 16) == (valid_wide[i][0] | valid_wide[i][1] << 16) &&
            (word(a.data + 20) & 0xffff) == valid_wide[i][2]);
      uint8_t *root = calloc(1, 1024 + 64);
      run(&a, root, 1024 + 64, NULL, 0, 0);
      for (unsigned k = 0; k < 256; k++) {
         unsigned waves = DIV_ROUND_UP(invocations, 32);
         CHECK(word(root + 4 * k) == (k < invocations ? k + k / 32 + (waves << 16) : 0));
      }
      free(root);
      apex_compile_result_finish(&a);
   }

   const unsigned invalid_sizes[][3] = {
      {257, 1, 1}, {1, 257, 1}, {1, 1, 65}, {256, 2, 1}, {0, 1, 16}, {UINT16_MAX, UINT16_MAX, 16},
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
   nir->info.stage = MESA_SHADER_GEOMETRY;
   reject(nir, &a, "only compute, vertex and fragment");
   nir = shader(17);
   nir_builder nb = builder(nir);
   nir_store_ssbo(&nb, nir_imm_int(&nb, 3), nir_imm_int(&nb, 1), nir_imm_int(&nb, 0), .align_mul = 4);
   reject(nir, &a, "store_ssbo");

   const struct apex_op invalid = {.op = 0x200};
   const struct apex_value values[1] = {{0, 1}};
   struct apex_header header = {.stage = APEX_COMPUTE, .local = {32, 1, 1}, .shared = 128,
                                .private_bytes = 32};
   CHECK(apex_emit(&invalid, 1, values, 1, &header, &a) != 0);
   CHECK(!a.data && !a.size && !strcmp(a.diagnostic, "opcode overflow"));
   apex_compile_result_finish(&a);
   CHECK(apex_emit(NULL, 0, values, 1, &header, &a) == 0);
   CHECK(a.data && a.size == 72 && !a.diagnostic[0]);
   CHECK(word(a.data + 20) == (1 | 128 << 16) && word(a.data + 24) == 32);
   CHECK(a.data[64] == APEX_S_ENDPGM);
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
      {SCOPE_WORKGROUP, nir_var_mem_shared, NIR_MEMORY_MAKE_AVAILABLE | NIR_MEMORY_RELEASE, false},
      {SCOPE_WORKGROUP, nir_var_mem_shared, NIR_MEMORY_MAKE_VISIBLE | NIR_MEMORY_ACQUIRE, false},
   };
   for (unsigned n = 0; n < ARRAY_SIZE(barriers); n++) {
      nir = shader(17);
      nb = builder(nir);
      nir_barrier(&nb, .execution_scope = SCOPE_NONE, .memory_scope = barriers[n].scope,
                  .memory_modes = barriers[n].modes, .memory_semantics = barriers[n].semantics);
      if (!barriers[n].supported) {
         reject(nir, &a, "barrier");
         continue;
      }
      CHECK(apex_from_nir(nir, NULL, &a) == 0);
      ralloc_free(nir);
      check_binary(&a, 17);
      CHECK(has_opcode(&a, APEX_S_FENCE) && !has_opcode(&a, APEX_S_BARRIER));
      apex_compile_result_finish(&a);
   }

   nir = shader(17);
   nb = builder(nir);
   nir_barrier(&nb, .execution_scope = SCOPE_WORKGROUP, .memory_scope = SCOPE_WORKGROUP,
               .memory_modes = nir_var_mem_shared, .memory_semantics = NIR_MEMORY_ACQ_REL);
   CHECK(apex_from_nir(nir, NULL, &a) == 0);
   ralloc_free(nir);
   CHECK(has_opcode(&a, APEX_S_BARRIER));
   check_binary(&a, 17);
   apex_compile_result_finish(&a);

   /* Paired global addresses: descriptors in the root, data elsewhere. */
   nir = global_shader(4, 4);
   CHECK(apex_from_nir(nir, NULL, &a) == 0);
   ralloc_free(nir);
   {
      const uint64_t data_va = 0xfeffff0000ull; /* the low word carries */
      uint8_t *data = calloc(1, 0x21000);
      uint8_t root[64] = {0};
      uint32_t words[4] = {(uint32_t)data_va, data_va >> 32, (uint32_t)data_va, data_va >> 32};
      memcpy(root, words, sizeof(words));
      for (unsigned k = 0; k < 32; k++) {
         uint32_t v = 1000 + 3 * k;
         memcpy(data + 0x20090 + 4 * k, &v, 4);
      }
      run(&a, root, sizeof(root), data, data_va, 0x21000);
      for (unsigned k = 0; k < 32; k++)
         CHECK(word(data + 132 + 4 * k) == (1000 + 3 * k) * 7 + k);
      free(data);
   }
   if (argc >= 2) {
      FILE *f = fopen(argv[1], "wb");
      CHECK(f && fwrite(a.data, 1, a.size, f) == a.size);
      CHECK(fclose(f) == 0);
   }
   apex_compile_result_finish(&a);
   reject(global_shader(2, 4), &a, "load_global_2x32");
   reject(global_shader(4, 2), &a, "store_global_2x32");

   nir = compute("Apex unsigned saturating subtraction", 32, 1, 1);
   nb = builder(nir);
   nir_def *offset = nir_ishl_imm(&nb, nir_load_local_invocation_index(&nb), 2);
   nir_def *left = nir_load_ssbo(&nb, 1, 32, nir_imm_int(&nb, 0), offset, .align_mul = 4);
   nir_def *right = nir_load_ssbo(&nb, 1, 32, nir_imm_int(&nb, 0), nir_iadd_imm(&nb, offset, 128),
                                  .align_mul = 4);
   nir_store_ssbo(&nb, nir_usub_sat(&nb, left, right), nir_imm_int(&nb, 0),
                  nir_iadd_imm(&nb, offset, 256), .align_mul = 4);
   CHECK(apex_from_nir(nir, NULL, &a) == 0);
   ralloc_free(nir);
   {
      uint8_t root[384] = {0};
      for (unsigned k = 0; k < 32; k++) {
         uint32_t l = k * 0x11111111u, r = 0x80000000u - k;
         memcpy(root + 4 * k, &l, 4);
         memcpy(root + 128 + 4 * k, &r, 4);
      }
      run(&a, root, sizeof(root), NULL, 0, 0);
      for (unsigned k = 0; k < 32; k++) {
         uint32_t l = k * 0x11111111u, r = 0x80000000u - k;
         CHECK(word(root + 256 + 4 * k) == (l > r ? l - r : 0));
      }
   }
   if (argc == 3) {
      FILE *f = fopen(argv[2], "wb");
      CHECK(f && fwrite(a.data, 1, a.size, f) == a.size);
      CHECK(fclose(f) == 0);
   }
   apex_compile_result_finish(&a);
   test_vertex_prolog(false);
   test_vertex_prolog(true);
   glsl_type_singleton_decref();
   puts("PASS Apex in-process compiler: ownership, diagnostics, metadata, barriers, paired global addresses and the vertex prolog on the ISA model");
   return 0;
}
