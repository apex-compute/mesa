# P7 ISA: resolved ambiguities

The backend (`isa.rs`, `from_nir.c`) and the test model (`sim.rs`) implement
`Docs/architecture.md` "Shader core and ISA". Where that contract leaves a
choice open, they make the choice below. Reconcile each item with the RTL
reference simulator and write the result into the architecture section when
the core integrates.

## Encoding

- Unused operand fields are zero and ignored (code 0 is also `v0`). Absent
  memory `a`/`b` operands are 255.
- A literal (`a` or `b` = 255) is allowed only in ALU forms without a `c`
  operand. Compares (condition in `c`), `v_cndmask`, `v_fma_f`, `v_bfi`,
  `v_perm`, the cube helpers, `v_addc`/`v_subb`, `v_add_co`/`v_sub_co` and
  `s_cselect` take no literal. The assembler rejects modifiers or clamp with
  a literal.
- The inline constants are exactly the fifteen listed; −4.0 is a literal.
- Negate/absolute bits are valid only on FP sources (FP arithmetic, rounding,
  transcendental, `v_cvt_u_f`/`v_cvt_i_f`, `v_cvt_pk_f16`, `v_cmp_f`,
  `v_cmp_class`, `v_ldexp` `a`, cube helpers); elsewhere they trap. Clamp is
  valid on FP results only and maps NaN to 0.
- `d` = `exec` (224) is a valid scalar-ALU and compare destination.
- `s_launch` takes its index as the literal: 0–2 workgroup x/y/z, 3 wave
  index, 4–6 local size.
- Branch offsets are signed instruction counts from the next instruction.
- Vector registers of any two-dword field (pairs and two-dword memory data)
  start at an even index. Scalar groups of four or eight dwords are
  four-aligned by the allocator; decode enforces only pair alignment.
- `image_fetch` carries variant 4 (fetch) with sampler, gather, compare and
  offset fields zero; `image_sample` never uses variant 4.
- Export targets 8 (depth) and 9 (sample mask) read four registers and use
  the first.

## Program header (64 bytes)

Word 0 magic `0x50585041`; 1 stage (0 compute, 1 vertex, 2 fragment) with
fragment flags in bits 15:8 (early tests, discards, exports depth, exports
sample mask, side effects, centroid, `1/w`, per-sample); 2 code bytes; 3 entry
instruction; 4 local size `x | y << 9 | z << 18`; 5 shared bytes; 6 private
bytes per lane; 7 vertex output stride, or fragment color-target mask with
dual source in bit 8; 8–15 fragment inputs, one byte each: bit 7 valid, bit 6
flat, bits 5:0 vertex output record vec4.

## Semantics

- `v_min_f`/`v_max_f` order −0 below +0.
- `v_ffbh_u` returns the index of the highest set bit counted from bit 0,
  `v_ffbh_i` that of the highest bit differing from the sign; `v_ffbl`,
  `v_ffbh_*` and `s_ff1` return −1 when there is none.
- `v_bfe_*`/`s_bfe_*`: `b = offset[4:0] | width[13:8]`, width capped at 32,
  bits past bit 31 read zero.
- `v_perm` selector byte *s* < 8 picks byte *s* of `{a (high), b (low)}`,
  otherwise zero. `v_mbcnt` has no addend. `v_quadperm` takes its pattern from
  `b` (usually the literal), two bits per quad lane; inactive source lanes read
  zero as for `v_permlane`. `v_readfirstlane` with an empty `exec` reads lane 0.
- `v_interp` component index `c` = input · 4 + component; each lane's
  primitive slot is state the core keeps from the launch, not `v3`, which the
  program may overwrite. The result is `fma(j, P20, fma(i, P10, P0))`.
- Hidden per-primitive attributes (the Z plane and per-vertex `1/w`) have no
  index yet; the compiler rejects `gl_FragCoord.z` and `noperspective`
  inputs until they do.
- Fragment `exp` writes the lanes in `exec`; a lane absent from `exec` at the
  done export is discarded. Terminate removes lanes from `exec`; demote keeps
  the lane running and leaves it out of `exec` at the exports. A program
  without color writes exports target 0 with done; the header's color mask
  tells the ROP to ignore it. Helper lanes (launch `v3` bit 11) and demoted
  lanes run stores and atomics with `exec` masked by the compiler.
- Scalar loads read memory directly in the model. The compiler issues them
  only for read-only data (uniform buffers, push constants, descriptors,
  `CAN_REORDER` accesses), since constant-cache coherence with a wave's own
  vector stores is unspecified.

## Memory addressing

- 32-bit register offsets and the sign-extended immediate add modulo 2^32
  before they meet a 64-bit base.
- `global_*`: with a scalar pair `b`, `a` is a 32-bit unsigned offset; with
  `b` absent, `a` is a vector address pair.
- `s_load`: `s[b]:s[b+1] + (s[a] + imm)`. `s_buffer_load`, `buffer_*`: each
  dword is checked against the descriptor's bytes; `s_buffer_load` reads zero
  past the range whatever the flags.
- `scratch_*`: `b` is the private base pair (launch `s20`–`s21`), the word is
  `(v[a] + imm) / 4`, *wave* is the wave's index in its workgroup and
  multi-dword accesses use consecutive words. Vertex and fragment launches
  carry no private base, so the compiler lowers dynamically indexed locals to
  selects there and spills only in compute programs.
- `shared_*`: byte address `v[a] + s[b] + imm` against the header's shared
  bytes.
- Atomics: size code 0 is 32-bit, 1 is 64-bit; the operand sits at `d`; for
  compare-exchange `d` holds the new value and `d + size` the comparison
  value; with the return bit the old value replaces `d`.

## Stage interfaces

- Vertex inputs: attribute location *L* arrives in `v(2 + 4L)`–`v(5 + 4L)`;
  `v0` is the vertex index including the base vertex and `v1` the instance
  index including the base instance (`s20`).
- Vertex output record: vec4 0 position, vec4 1 {point size, layer,
  viewport, cull mask}, varying location *L* at vec4 2 + *L*, clip and cull
  distances at vec4 34–35. Varyings precede the clip distances so a fragment
  header names its inputs without the vertex program; the stride is 16 bytes
  times the highest written vec4 plus one.
- Texture coordinate registers, in order: coordinates with the array layer,
  the level or bias, the comparison value, the packed offset
  (`x | y << 8 | z << 16`, six-bit signed), then d/dx and d/dy. Cube images
  take `(s, t, face [+ 6 · layer])` selected by the compiler; `image_fetch`
  takes integer coordinates and the level.
- Image and sampler descriptor bit layouts are the model's (see
  `sim.rs` `texel`/`texture`); the texture unit's layout replaces them.

## Mesa ABI

- User data `s0`–`s1` hold the root GPUVA in every stage. Vulkan rows are 32
  bytes; a buffer row is the descriptor `{GPUVA low, high, bytes, flags}` with
  flag bit 0 (robust) set on every row including the null sentinel; push
  constants follow the sentinel and dispatch metadata the push image.
- Compute workgroup IDs arrive in `s16`–`s18` with the base included; the
  wave index in `s19`.
- The standalone layout (`apex-compile`, tests) puts set *s* binding *b*
  element *e* at root + 64 (1 + 256 *s* + *b* + *e*) (buffer descriptor, or
  image descriptor then sampler descriptor), push constants at
  root + 64 · 2049 and the grid size in `s2`–`s4`.
