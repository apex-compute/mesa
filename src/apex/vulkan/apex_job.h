/* SPDX-License-Identifier: MIT */
/* Job words of the internal compute kernels, shared by the driver and the
 * kernels' GLSL. All values are little-endian 32-bit words. A descriptor
 * table's trailer, after its rows and push constants, holds the dispatch
 * grid (x, y, z, pad) and then the job words; internal kernels have no
 * descriptors or push constants, so their trailer starts at word 8. */
#ifndef APEX_JOB_H
#define APEX_JOB_H

#define APEX_TRAILER_WORDS 4          /* grid x, y, z, pad */
#define APEX_JOB_WORDS 32
#define APEX_JOB_BASE 12              /* word of job word 0 in an internal kernel's table */

/* A surface of one image level (or a buffer region) as the kernels address
 * it: texel (x, y) of plane z = layer * samples + sample. Tiled surfaces use
 * the texture unit's layout (apex_hw.h); linear ones a byte row pitch. */
#define APEX_SURFACE_VA 0             /* level, layer 0, sample 0: lo, hi */
#define APEX_SURFACE_FLAGS 2          /* tiled [0], log2 texel bytes [3:1], samples [11:4] */
#define APEX_SURFACE_PITCH 3          /* bytes per row (linear) or 4 KiB tiles per row (tiled) */
#define APEX_SURFACE_LAYER 4          /* bytes between layers */
#define APEX_SURFACE_SAMPLE 5         /* bytes between sample planes */
#define APEX_SURFACE_ORIGIN 6         /* x | y << 16 of the region's first texel */
#define APEX_SURFACE_TEXEL 7          /* texel bytes (linear surfaces of 3, 6 or 12 bytes) */
#define APEX_SURFACE_WORDS 8
#define APEX_SURFACE_TILED 1u

/* Copy: one invocation per element of a width x height x planes region.
 * An element is `bytes` (1, 2, 4, 8 or 16) bytes at a byte offset within
 * each side's texel; the destination keeps bits outside `mask` (elements of
 * at most 4 bytes; ~0 otherwise). */
#define APEX_COPY_SRC 0
#define APEX_COPY_DST 8
#define APEX_COPY_EXTENT 16           /* width, height, planes */
#define APEX_COPY_BYTES 19
#define APEX_COPY_SRC_OFFSET 20
#define APEX_COPY_DST_OFFSET 21
#define APEX_COPY_MASK 22

/* Clear: one invocation per texel of a width x height x planes region; the
 * pattern's bits under the mask replace the texel's. */
#define APEX_CLEAR_DST 0
#define APEX_CLEAR_EXTENT 8           /* width, height, planes */
#define APEX_CLEAR_BYTES 11           /* texel bytes: 1, 2, 3, 4, 6, 8, 12 or 16 */
#define APEX_CLEAR_PATTERN 12         /* packed texel, 4 words */
#define APEX_CLEAR_MASK 16            /* texel bits to write, 4 words */

/* ETC2/EAC decode: one invocation per texel of a width x height x layers
 * region, from its first block to the decoded plane. */
#define APEX_DECODE_DST 0             /* decoded plane surface */
#define APEX_DECODE_SRC 8             /* first block VA lo, hi */
#define APEX_DECODE_SRC_ROW 10        /* bytes per block row */
#define APEX_DECODE_SRC_SLICE 11
#define APEX_DECODE_EXTENT 12         /* texel width, height, layers */
#define APEX_DECODE_KIND 15
/* Block kinds; 16-byte blocks are RGBA8 and both RG11 kinds. */
#define APEX_ETC2_RGB8 0
#define APEX_ETC2_RGBA1 1
#define APEX_ETC2_RGBA8 2
#define APEX_EAC_R11 3
#define APEX_EAC_R11_SNORM 4
#define APEX_EAC_RG11 5
#define APEX_EAC_RG11_SNORM 6

/* Query slots: one 64-bit value (two for transform feedback streams:
 * primitives written, then needed), the 32-bit availability word at byte
 * 16, padding. */
#define APEX_QUERY_STRIDE 32
#define APEX_QUERY_AVAILABLE 16
/* Query copy: one invocation per query. */
#define APEX_QUERY_SLOT 0             /* first slot VA lo, hi */
#define APEX_QUERY_DST 2              /* copy destination VA lo, hi */
#define APEX_QUERY_DST_STRIDE 4
#define APEX_QUERY_COUNT 5
#define APEX_QUERY_FLAGS 6            /* VkQueryResultFlags */
#define APEX_QUERY_VALUES 7           /* values per query: 1 or 2 */

/* Transform feedback. A draw's capture program (the vertex program built as
 * a compute kernel, apex_graphics.c) and the bookkeeping kernel share these
 * job words. The draw's parameters are its VkDraw(Indexed)IndirectCommand
 * words, read from APEX_XFB_PARAMS when nonzero; a nonzero count address
 * skips draw APEX_XFB_DRAW_INDEX at or beyond the u32 it holds. */
#define APEX_XFB_DRAW 0               /* 4 or 5 command words */
#define APEX_XFB_PARAMS 5             /* indirect command VA lo, hi */
#define APEX_XFB_COUNT 7              /* draw count VA lo, hi */
#define APEX_XFB_DRAW_INDEX 9
#define APEX_XFB_INDEX 10             /* index buffer VA lo, hi */
#define APEX_XFB_INDEX_BYTES 12
#define APEX_XFB_FLAGS 13             /* index type 1:0, indexed 2, restart 3, last provoking 4, topology 10:8 */
#define APEX_XFB_VERTEX_INPUT 14      /* vertex-input block VA lo, hi */
#define APEX_XFB_STATE 16             /* state VA lo, hi */
#define APEX_XFB_QUERY 18             /* stream query slot VA lo, hi; 0 without one */
#define APEX_XFB_STRIDES 20           /* buffer strides, two per word, 16 bits each */
#define APEX_XFB_MODE 22
/* Bookkeeping modes: advance the offsets and query by a draw, load the
 * offsets from counter buffers, store them to counter buffers. */
#define APEX_XFB_ADVANCE 0
#define APEX_XFB_LOAD 1
#define APEX_XFB_STORE 2
#define APEX_XFB_COUNTERS 0           /* load and store: 4 counter VAs lo, hi; 0 skips */
#define APEX_XFB_FLAG_INDEXED (1u << 2)
#define APEX_XFB_FLAG_RESTART (1u << 3)
#define APEX_XFB_FLAG_LAST (1u << 4)
/* State, 16 words from vkCmdBeginTransformFeedbackEXT: each buffer's byte
 * offset, then each buffer's {VA lo, VA hi, bytes}. */
#define APEX_XFB_BUFFERS 4
#define APEX_XFB_STATE_OFFSETS 0
#define APEX_XFB_STATE_BUFFERS 4
#define APEX_XFB_STATE_WORDS 16
/* Capture workgroup width. */
#define APEX_XFB_LOCAL 64

#endif
