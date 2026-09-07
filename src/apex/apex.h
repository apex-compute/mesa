/* SPDX-License-Identifier: MIT */
#ifndef APEX_H
#define APEX_H
#include <stddef.h>
#include <stdint.h>
struct nir_shader;
/* NIR adapter sends source-neutral opcodes and SSA value IDs to Rust MIR. */
struct apex_op { uint32_t op, d, a, b, c, imm; };
int apex_emit(const struct apex_op *, size_t, uint32_t shared, uint32_t private_bytes, const char *path);
int apex_tool(const char *mode, const char *input, const char *output);
int apex_from_nir(struct nir_shader *, const char *path);
#endif
