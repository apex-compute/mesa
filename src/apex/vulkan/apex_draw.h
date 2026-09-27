/* SPDX-License-Identifier: MIT */
/* Draw block shared by the driver, the setup kernel (GLSL) and generated
 * vertex/fragment kernels. All values are little-endian 32-bit words. The
 * block follows the dispatch parameters in a job's immutable trailer. */
#ifndef APEX_DRAW_H
#define APEX_DRAW_H

/* Vertex output records: stride words per vertex, slot 0 is the clip position. */
#define APEX_DRAW_VERTEX_LO 0
#define APEX_DRAW_VERTEX_HI 1
#define APEX_DRAW_VERTEX_STRIDE 2
#define APEX_DRAW_VERTEX_COUNT 3      /* vertices per instance */
#define APEX_DRAW_INSTANCE_COUNT 4
#define APEX_DRAW_FIRST_VERTEX 5
#define APEX_DRAW_FIRST_INSTANCE 6
/* Primitive records: APEX_SUBPRIMS records of APEX_PRIM_WORDS per input primitive. */
#define APEX_DRAW_PRIM_LO 7
#define APEX_DRAW_PRIM_HI 8
#define APEX_DRAW_PRIM_COUNT 9        /* input primitives per instance */
#define APEX_DRAW_TOPOLOGY 10
#define APEX_DRAW_WIDTH 11            /* framebuffer extent in pixels */
#define APEX_DRAW_HEIGHT 12
#define APEX_DRAW_VIEWPORT 13         /* x, y, width, height, minDepth, maxDepth (FP32) */
#define APEX_DRAW_SCISSOR 19          /* x0, y0, x1, y1 exclusive, intersected with the render area */
#define APEX_DRAW_CULL 23             /* VkCullModeFlags */
#define APEX_DRAW_FRONT_FACE 24       /* VkFrontFace */
#define APEX_DRAW_DEPTH 25            /* bit 0 test, bit 1 write, bits 4..6 VkCompareOp */
#define APEX_DRAW_COLOR 26            /* per attachment: VA lo, VA hi, row stride */
#define APEX_DRAW_COLOR_WORDS 3
#define APEX_DRAW_MAX_COLOR 8
#define APEX_DRAW_DEPTH_TARGET 50     /* VA lo, VA hi, row stride */
#define APEX_DRAW_SLOTS 53            /* 32 generic locations: record word offset or ~0 */
#define APEX_DRAW_BINDINGS 85         /* per vertex binding: VA lo, VA hi, bytes, stride */
#define APEX_DRAW_BINDING_WORDS 4
#define APEX_DRAW_MAX_BINDINGS 16
#define APEX_DRAW_INDEX 149           /* VA lo, VA hi, bytes/index (0 non-indexed), vertex offset */
/* Per attachment: word 0 = src color | dst color << 8 | src alpha << 16 |
 * dst alpha << 24 (VkBlendFactor); word 1 = color op | alpha op << 8 |
 * write mask << 16 | enable << 24. */
#define APEX_DRAW_BLEND 153
#define APEX_DRAW_BLEND_CONSTANTS 169 /* FP32 RGBA */
#define APEX_DRAW_OCCLUSION 173     /* query slot VA lo, hi; zero when inactive */
/* Primitive bins: square bins of 1 << shift pixels over the scissor, one
 * ordered list segment per bin and 1024-primitive chunk. */
#define APEX_DRAW_BIN_LISTS 175       /* VA lo, hi: bin * primitives + chunk * 256 + k */
#define APEX_DRAW_BIN_COUNTS 177      /* VA lo, hi: bin * chunks + chunk */
#define APEX_DRAW_BIN_SHIFT 179
#define APEX_DRAW_BIN_COLUMNS 180
#define APEX_DRAW_BIN_ROWS 181
#define APEX_DRAW_BIN_CHUNKS 182
#define APEX_DRAW_BIN_X0 183          /* first bin column and row */
#define APEX_DRAW_BIN_Y0 184
#define APEX_DRAW_WORDS 185
#define APEX_BIN_CHUNK 256

/* Primitive record fields. Edge i is opposite vertex i; E(p) = a x + b y + c
 * in Q16.8 fixed point with the top-left bias folded into c: covered iff all
 * three E >= 0. The bounding box is in pixels, maxima exclusive. */
#define APEX_PRIM_EDGE 0              /* 3 x {a, b, c lo, c hi} */
#define APEX_PRIM_BOX 12              /* x0, y0, x1, y1 */
#define APEX_PRIM_RECIPROCAL_AREA 16  /* FP32 1 / sum(E) at any point */
#define APEX_PRIM_Z 17                /* window z per vertex (FP32) */
#define APEX_PRIM_INV_W 20            /* 1/w per vertex (FP32) */
#define APEX_PRIM_WEIGHTS 23          /* 3 x 3 FP32: sub-vertex i weight of source vertex j */
#define APEX_PRIM_SOURCE 32           /* 3 source vertex record indices */
#define APEX_PRIM_FLAGS 35            /* bit 0 front-facing; bits 8..9 provoking source */
#define APEX_PRIM_ID 36
#define APEX_PRIM_COUNT 37            /* first record only: records written for the input primitive */
#define APEX_PRIM_UNION_BOX 40        /* first record only: box of all records, pixels */
#define APEX_PRIM_WORDS 44
#define APEX_TILES_PER_WORKGROUP 1 /* per fragment job: bounded launch runtime */
#define APEX_SUBPRIMS 8               /* a clipped triangle fans into at most 7 */

/* Internal pitched copy job words (in place of the draw block). */
#define APEX_COPY_SRC 0
#define APEX_COPY_DST 2
#define APEX_COPY_SRC_ROW 4
#define APEX_COPY_DST_ROW 5
#define APEX_COPY_SRC_SLICE 6
#define APEX_COPY_DST_SLICE 7
#define APEX_COPY_WORDS 8             /* 32-bit words per row */
#define APEX_COPY_ROWS 9
#define APEX_COPY_LAYERS 10

/* Query slots: 64-bit value, 32-bit availability, padding. */
#define APEX_QUERY_STRIDE 16
/* Internal query job words (in place of the draw block). */
#define APEX_QUERY_SLOT 0             /* first slot VA lo, hi */
#define APEX_QUERY_DST 2              /* copy destination VA lo, hi */
#define APEX_QUERY_DST_STRIDE 4
#define APEX_QUERY_COUNT 5
#define APEX_QUERY_FLAGS 6            /* VkQueryResultFlags */

/* Internal clear job words (in place of the draw block). */
#define APEX_CLEAR_DST 0              /* first texel VA lo, hi */
#define APEX_CLEAR_ROW 2              /* row pitch */
#define APEX_CLEAR_SLICE 3            /* layer pitch */
#define APEX_CLEAR_WIDTH 4            /* texels per row */
#define APEX_CLEAR_ROWS 5
#define APEX_CLEAR_LAYERS 6
#define APEX_CLEAR_BYTES 7            /* texel bytes: 1, 2, 4, 8 or 16 */
#define APEX_CLEAR_PATTERN 8          /* packed texel, 4 words */

#endif
