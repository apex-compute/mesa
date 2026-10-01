/* SPDX-License-Identifier: MIT */
#ifndef APEX_H
#define APEX_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
struct nir_shader;
struct nir_shader_compiler_options;
extern const struct nir_shader_compiler_options apex_nir_options;
/* Initialize to zero before compilation. Success returns owned program bytes
 * and their register, instruction and spill counts; failure returns no bytes
 * and a NUL-terminated diagnostic. Finish before reuse. */
struct apex_compile_result {
   uint8_t *data;
   size_t size;
   uint32_t instructions, vector, scalar, spills;
   char diagnostic[1024];
};
void apex_compile_result_finish(struct apex_compile_result *);

/* P7 ISA opcodes (isa.rs owns the encoding). */
enum apex_opcode {
   APEX_S_NOP = 0x00, APEX_S_ENDPGM = 0x01, APEX_S_TRAP = 0x02, APEX_S_BRANCH = 0x03,
   APEX_S_CBRANCH_Z = 0x04, APEX_S_CBRANCH_NZ = 0x05, APEX_S_CBRANCH_EXECZ = 0x06,
   APEX_S_CBRANCH_EXECNZ = 0x07, APEX_S_BARRIER = 0x08, APEX_S_FENCE = 0x09,
   APEX_S_SLEEP = 0x0a, APEX_S_MOV = 0x20, APEX_S_ADD = 0x21, APEX_S_SUB = 0x22,
   APEX_S_MUL = 0x23, APEX_S_MUL_HI_U = 0x24, APEX_S_AND = 0x25, APEX_S_OR = 0x26,
   APEX_S_XOR = 0x27, APEX_S_ANDN2 = 0x28, APEX_S_ORN2 = 0x29, APEX_S_NOT = 0x2a,
   APEX_S_SHL = 0x2b, APEX_S_SHR = 0x2c, APEX_S_ASHR = 0x2d, APEX_S_BFE_U = 0x2e,
   APEX_S_BFE_I = 0x2f, APEX_S_MIN_I = 0x30, APEX_S_MIN_U = 0x31, APEX_S_MAX_I = 0x32,
   APEX_S_MAX_U = 0x33, APEX_S_CMP_EQ = 0x34, APEX_S_CMP_NE = 0x35,
   APEX_S_CMP_LT_I = 0x36, APEX_S_CMP_LT_U = 0x37, APEX_S_CMP_LE_I = 0x38,
   APEX_S_CMP_LE_U = 0x39, APEX_S_CSELECT = 0x3a, APEX_S_ADD64 = 0x3b,
   APEX_S_SUB64 = 0x3c, APEX_S_SHL64 = 0x3d, APEX_S_FF1 = 0x3e, APEX_S_POPCNT = 0x3f,
   APEX_S_AND_SAVEEXEC = 0x40, APEX_S_OR_SAVEEXEC = 0x41, APEX_S_ANDN2_SAVEEXEC = 0x42,
   APEX_S_SETEXEC = 0x43, APEX_S_MEMTIME = 0x44, APEX_S_LAUNCH = 0x45, APEX_V_MOV = 0x60,
   APEX_V_ADD = 0x61, APEX_V_SUB = 0x62, APEX_V_MUL_LO = 0x63, APEX_V_MUL_HI_U = 0x64,
   APEX_V_MUL_HI_I = 0x65, APEX_V_AND = 0x66, APEX_V_OR = 0x67, APEX_V_XOR = 0x68,
   APEX_V_NOT = 0x69, APEX_V_SHL = 0x6a, APEX_V_SHR = 0x6b, APEX_V_ASHR = 0x6c,
   APEX_V_BFE_U = 0x6d, APEX_V_BFE_I = 0x6e, APEX_V_BFI = 0x6f, APEX_V_MIN_I = 0x70,
   APEX_V_MIN_U = 0x71, APEX_V_MAX_I = 0x72, APEX_V_MAX_U = 0x73, APEX_V_ADD_CO = 0x74,
   APEX_V_SUB_CO = 0x75, APEX_V_ADDC = 0x76, APEX_V_SUBB = 0x77, APEX_V_POPCNT = 0x78,
   APEX_V_FFBL = 0x79, APEX_V_FFBH_U = 0x7a, APEX_V_FFBH_I = 0x7b, APEX_V_BFREV = 0x7c,
   APEX_V_PERM = 0x7d, APEX_V_MORTON = 0x7e, APEX_V_CNDMASK = 0x7f, APEX_V_CMP_I = 0x80,
   APEX_V_CMP_U = 0x81, APEX_V_CMP_F = 0x82, APEX_V_CMP_CLASS = 0x83, APEX_V_ADD_F = 0x84,
   APEX_V_SUB_F = 0x85, APEX_V_MUL_F = 0x86, APEX_V_FMA_F = 0x87, APEX_V_MIN_F = 0x88,
   APEX_V_MAX_F = 0x89, APEX_V_FLOOR = 0x8a, APEX_V_CEIL = 0x8b, APEX_V_TRUNC = 0x8c,
   APEX_V_RNDNE = 0x8d, APEX_V_FRACT = 0x8e, APEX_V_FREXP_MANT = 0x8f,
   APEX_V_FREXP_EXP = 0x90, APEX_V_LDEXP = 0x91, APEX_V_RCP = 0x92, APEX_V_RSQ = 0x93,
   APEX_V_SQRT = 0x94, APEX_V_EXP2 = 0x95, APEX_V_LOG2 = 0x96, APEX_V_SIN = 0x97,
   APEX_V_COS = 0x98, APEX_V_CVT_F_U = 0x99, APEX_V_CVT_F_I = 0x9a, APEX_V_CVT_U_F = 0x9b,
   APEX_V_CVT_I_F = 0x9c, APEX_V_CVT_PK_F16 = 0x9d, APEX_V_CVT_F16_LO = 0x9e,
   APEX_V_CVT_F16_HI = 0x9f, APEX_V_CUBEID = 0xa0, APEX_V_CUBESC = 0xa1,
   APEX_V_CUBETC = 0xa2, APEX_V_CUBEMA = 0xa3, APEX_V_READLANE = 0xa4,
   APEX_V_READFIRSTLANE = 0xa5, APEX_V_WRITELANE = 0xa6, APEX_V_PERMLANE = 0xa7,
   APEX_V_QUADPERM = 0xa8, APEX_V_MBCNT = 0xa9, APEX_V_INTERP = 0xaa,
   APEX_V_INTERP_FLAT = 0xab, APEX_S_LOAD = 0xc0, APEX_S_BUFFER_LOAD = 0xc1,
   APEX_GLOBAL_LOAD = 0xc2, APEX_GLOBAL_STORE = 0xc3, APEX_GLOBAL_ATOMIC = 0xc4,
   APEX_BUFFER_LOAD = 0xc5, APEX_BUFFER_STORE = 0xc6, APEX_BUFFER_ATOMIC = 0xc7,
   APEX_SCRATCH_LOAD = 0xc8, APEX_SCRATCH_STORE = 0xc9, APEX_SHARED_LOAD = 0xca,
   APEX_SHARED_STORE = 0xcb, APEX_SHARED_ATOMIC = 0xcc, APEX_IMAGE_SAMPLE = 0xe0,
   APEX_IMAGE_FETCH = 0xe1, APEX_EXP = 0xe2,
};

/* Machine IR: ISA opcodes over typed values. Fields f[0..3] map onto the
 * instruction's d, a, b and c. */
enum apex_pseudo { APEX_CONST = 0x100, APEX_COPY = 0x101, APEX_LABEL = 0x102 };
#define APEX_REORDER 1u
struct apex_op { uint32_t op, f[4], hi, imm, flags; };
/* cls: 0 scalar, 1 vector; width in dwords. Value 0 is reserved. */
struct apex_value { uint8_t cls, width; };
/* Operands: a value (bit 31; dwords off..off+n-1 of value id, n <= 16), a
 * physical operand code (bit 30) or a raw field (bit 29). */
static inline uint32_t apex_val(uint32_t id, unsigned off, unsigned n)
{
   return 1u << 31 | (n - 1) << 27 | off << 23 | id;
}
static inline uint32_t apex_val_id(uint32_t x) { return x & 0x7fffff; }
static inline unsigned apex_val_off(uint32_t x) { return (x >> 23) & 15; }
static inline unsigned apex_val_n(uint32_t x) { return ((x >> 27) & 15) + 1; }
static inline uint32_t apex_phys(unsigned code) { return 1u << 30 | code; }
static inline uint32_t apex_raw(unsigned v) { return 1u << 29 | v; }
#define APEX_SCALAR 128u
#define APEX_EXEC 224u
#define APEX_LANE 225u
enum apex_stage { APEX_COMPUTE, APEX_VERTEX, APEX_FRAGMENT };
struct apex_header {
   uint32_t stage, flags, local[3], shared, private_bytes, output, input_count;
   uint8_t inputs[32]; /* bit 7 flat; input k is vertex output varying k */
};
int apex_emit(const struct apex_op *, size_t, const struct apex_value *, size_t,
              const struct apex_header *, struct apex_compile_result *);

/* Runs a program on the ISA reference model (Tooling/apex_isa, located by
 * the APEX_ISA environment variable): a compute grid, or one vertex or
 * fragment wave. Fragment waves take quad origins from v2 and coverage,
 * primitive slot and facing from v3; primitive p's attribute block holds
 * APEX_ATTRIBUTE_BLOCK dwords in the ring layout of Docs/isa.md. */
#define APEX_ATTRIBUTE_BLOCK (8 + 12 * 32)
struct apex_sim_region { uint64_t gpuva; void *data; uint64_t size; };
struct apex_sim_wave {
   uint32_t exec, scalar[32], vector[72][16];
   const uint32_t *attributes;
   uint32_t primitives;
   uint32_t exports[10][16][4], exported[10];
};
int apex_simulate(const uint8_t *program, size_t size, const uint32_t user[16],
                  const uint32_t groups[3], uint64_t private_base, struct apex_sim_wave *,
                  struct apex_sim_region *, size_t regions, char diagnostic[256]);

int apex_tool(const char *mode, const char *input, const char *output);
/* Mutates caller-owned NIR; the caller retains its lifetime and GLSL type ref.
 * Compute, vertex and fragment stages. */
int apex_from_nir(struct nir_shader *, struct apex_compile_result *);
/* Late algebraic rules (apex_nir_algebraic.py). */
bool apex_nir_opt_late(struct nir_shader *);
/* Validates Vulkan 1.3 SPIR-V and compiles its first entry point (standalone). */
int apex_compile_spirv(const uint32_t *words, size_t count, struct apex_compile_result *);
#endif
