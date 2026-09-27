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
#define APEX_DRAW_WORDS 153

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
#define APEX_PRIM_WORDS 40
#define APEX_SUBPRIMS 8               /* a clipped triangle fans into at most 7 */

#endif
