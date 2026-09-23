/* SPDX-License-Identifier: MIT */
#ifndef APEX_H
#define APEX_H
#include <stddef.h>
#include <stdint.h>
struct nir_shader;
struct nir_shader_compiler_options;
extern const struct nir_shader_compiler_options apex_nir_options;
/* Initialize to zero before compilation. Success returns owned APX bytes;
 * failure returns no bytes and a NUL-terminated diagnostic. Finish before reuse.
 * No compiler call writes files or requires a process-global error sink. */
struct apex_compile_result {
   uint8_t *data;
   size_t size;
   char diagnostic[1024];
};
void apex_compile_result_finish(struct apex_compile_result *);
/* NIR adapter sends source-neutral opcodes and SSA value IDs to Rust MIR. */
struct apex_op { uint32_t op, d, a, b, c, imm; };
int apex_emit(const struct apex_op *, size_t, uint32_t shared, uint32_t private_bytes,
              uint32_t local_invocations,
              struct apex_compile_result *);
int apex_tool(const char *mode, const char *input, const char *output);
/* Mutates caller-owned NIR; the caller retains its lifetime and GLSL type ref. */
int apex_from_nir(struct nir_shader *, struct apex_compile_result *);
#endif
