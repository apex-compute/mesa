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
/* Primitive records: APEX_SUBPRIMS_FOR(topology) records of APEX_PRIM_WORDS per
 * input primitive. */
#define APEX_DRAW_PRIM_LO 7
#define APEX_DRAW_PRIM_HI 8
#define APEX_DRAW_PRIM_COUNT 9        /* input primitives per instance */
#define APEX_DRAW_TOPOLOGY 10
#define APEX_DRAW_WIDTH 11            /* framebuffer extent in pixels */
#define APEX_DRAW_HEIGHT 12
#define APEX_DRAW_PROVOKING 13        /* VkProvokingVertexModeEXT */
#define APEX_DRAW_INDEX_SIZE 14       /* bound index bytes: indices beyond read zero */
#define APEX_DRAW_LOGIC_OP 15         /* VkLogicOp | enable << 4 */
/* VkPolygonMode | line quad << 4 (1 parallelogram, 2 rectangle, 3 smooth
 * rectangle) | stipple enable << 8. */
#define APEX_DRAW_RASTER 16
#define APEX_DRAW_LINE_STIPPLE 17     /* factor | pattern << 16 */
#define APEX_DRAW_LINE_WIDTH 18       /* FP32 */
#define APEX_DRAW_SCISSOR 19          /* union of the scissors below: x0, y0, x1, y1 exclusive */
#define APEX_DRAW_CULL 23             /* VkCullModeFlags */
#define APEX_DRAW_FRONT_FACE 24       /* VkFrontFace */
/* Bit 0 depth test, bit 1 depth write, bit 2 stencil test, bit 3 depth bias,
 * bits 4..6 depth VkCompareOp, bit 8 depth clamp. */
#define APEX_DRAW_DEPTH 25
#define APEX_DRAW_COLOR 26            /* per attachment: VA lo, VA hi, row stride */
#define APEX_DRAW_COLOR_WORDS 3
#define APEX_DRAW_MAX_COLOR 8
#define APEX_DRAW_DEPTH_TARGET 50     /* VA lo, VA hi, row stride */
#define APEX_DRAW_SLOTS 53            /* 32 generic locations: record word offset or ~0 */
#define APEX_DRAW_BINDINGS 85         /* per vertex binding: VA lo, VA hi, bytes, stride */
#define APEX_DRAW_BINDING_WORDS 4
#define APEX_DRAW_MAX_BINDINGS 16
#define APEX_DRAW_INDEX 149           /* VA lo, VA hi, bytes/index 1, 2, 4 (0 non-indexed), vertex offset */
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
#define APEX_DRAW_POINT_SIZE 185     /* vertex-record word of gl_PointSize, ~0 for 1.0 */
/* Front then back face: ops word = fail | pass << 3 | depth fail << 6 |
 * compare << 9 (VkStencilOp, VkCompareOp); masks word = compare mask |
 * write mask << 8 | reference << 16. */
#define APEX_DRAW_STENCIL 186
/* Constant factor, slope factor, clamp (FP32) and depth unorm bits (0 float). */
#define APEX_DRAW_DEPTH_BIAS 190
#define APEX_DRAW_RESTART 194        /* indexed primitive restart enabled */
/* Indirect draws: the resolve job writes the dynamic words (counts, first
 * vertex/instance, vertex offset and bin geometry) at their draw-block
 * indices into a parameter block; kernels read those words from it when
 * APEX_DRAW_PARAMS is nonzero. Resolve reads the command at APEX_DRAW_INDIRECT
 * and zeroes draws at or beyond the count word at APEX_DRAW_INDIRECT_COUNT. */
#define APEX_DRAW_PARAMS 195          /* parameter block VA lo, hi; zero when direct */
#define APEX_DRAW_INDIRECT 197        /* VkDraw[Indexed]IndirectCommand VA lo, hi */
#define APEX_DRAW_INDIRECT_COUNT 199  /* count VA lo, hi; zero without a count buffer */
#define APEX_DRAW_INDIRECT_INDEX 201  /* draw index for the count comparison */
/* Fragment jobs: bin chunks [first, end) of this job. Consecutive jobs keep
 * API order per pixel while each workgroup walks a bounded primitive count. */
#define APEX_DRAW_CHUNK_RANGE 202
#define APEX_DRAW_MAX_VIEWPORTS 16
#define APEX_DRAW_VIEWPORTS 204       /* per viewport: x, y, width, height, minDepth, maxDepth (FP32) */
#define APEX_DRAW_SCISSORS 300        /* per viewport: x0, y0, x1, y1 exclusive, within the render area */
#define APEX_DRAW_VIEWPORT_SLOT 364   /* vertex-record word of gl_ViewportIndex, ~0 for viewport 0 */
#define APEX_DRAW_VIEWPORT_COUNT 365
/* Layered rendering: per color attachment then depth, the byte stride between
 * layers; the fragment job's layer; the vertex-record word of gl_Layer (~0
 * for layer 0); and the multiview view index (~0 without multiview), which
 * also selects the layer. */
#define APEX_DRAW_LAYER_STRIDES 366
#define APEX_DRAW_LAYER 375
#define APEX_DRAW_LAYER_SLOT 376
#define APEX_DRAW_VIEW 377
/* Sample mask bits 0..15, alpha to coverage bit 16, alpha to one bit 17. */
#define APEX_DRAW_MULTISAMPLE 378
/* Clip distances: vertex-record word of the first | clip count << 16 |
 * cull count << 20; cull distances follow the clip distances. */
#define APEX_DRAW_CLIP 379
#define APEX_DRAW_WORDS 380
#define APEX_FRAGMENT_CHUNKS 8        /* bin chunks per fragment job */
#define APEX_DRAW_DYNAMIC(w) (((w) >= APEX_DRAW_VERTEX_COUNT && (w) <= APEX_DRAW_FIRST_INSTANCE) || \
                              (w) == APEX_DRAW_PRIM_COUNT || (w) == APEX_DRAW_INDEX + 3 || \
                              ((w) >= APEX_DRAW_BIN_SHIFT && (w) <= APEX_DRAW_BIN_Y0))
/* Indirect draw scratch: one device arena reused by every indirect draw,
 * since a device's jobs execute serially. Every job maps the whole VM, so
 * the arena stays small; larger indirect draws resolve to zero vertices. */
#define APEX_ARENA_VERTEX_BYTES (16u << 20)
#define APEX_ARENA_PRIM_BYTES (32u << 20)
#define APEX_ARENA_LIST_BYTES (16u << 20)
#define APEX_ARENA_COUNT_BYTES (256u << 10)
/* Direct draws size scratch per draw up to these limits. */
#define APEX_DRAW_MAX_SCRATCH (64u << 20)
#define APEX_DRAW_MAX_LISTS (32u << 20)
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
/* Bit 0 front-facing, bit 1 stippled line, bit 2 smooth line; bits 16..19
 * viewport index; bits 20..31 layer. */
#define APEX_PRIM_FLAGS 35
#define APEX_PRIM_ID 36
#define APEX_PRIM_COUNT 37            /* first record only: records written for the input primitive */
#define APEX_PRIM_DEPTH_OFFSET 38     /* depth bias of this triangle (FP32) */
#define APEX_PRIM_PROVOKING 39        /* provoking vertex record index: flat inputs */
#define APEX_PRIM_UNION_BOX 40        /* first record only: box of all records, pixels */
/* Lines (FP32): window origin x, y; unit direction x, y (the major axis for
 * parallelograms); stipple counter at the origin; length; half width. */
#define APEX_PRIM_LINE 44
#define APEX_PRIM_WORDS 52
#define APEX_TILES_PER_WORKGROUP 1 /* per fragment job: bounded launch runtime */
/* A clipped triangle fans into at most 7 records; point and line quads into 2. */
#define APEX_SUBPRIMS_FOR(topology) ((topology) <= 2u ? 2u : 8u)

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
/* Element mode (depth/stencil aspects): APEX_COPY_WORDS counts elements of
 * 1 or 4 bytes spaced by the strides; the mask selects destination bits. */
#define APEX_COPY_ELEMENT 11          /* element bytes, 0 for word rows */
#define APEX_COPY_SRC_STRIDE 12
#define APEX_COPY_DST_STRIDE 13
#define APEX_COPY_DST_MASK 14

/* Internal 4x resolve job words (in place of the draw block): one invocation
 * per destination pixel of a width x rows x layers region. */
#define APEX_RESOLVE_SRC 0            /* first source pixel VA lo, hi */
#define APEX_RESOLVE_SRC_ROW 2
#define APEX_RESOLVE_SRC_SLICE 3
#define APEX_RESOLVE_DST 4            /* first destination pixel VA lo, hi */
#define APEX_RESOLVE_DST_ROW 6
#define APEX_RESOLVE_DST_SLICE 7
#define APEX_RESOLVE_WIDTH 8
#define APEX_RESOLVE_ROWS 9
#define APEX_RESOLVE_LAYERS 10

/* Internal ETC2/EAC decode job words (in place of the draw block): one
 * invocation per texel of a width x height x layers region, from its first
 * block to its first texel in the image's decoded plane. */
#define APEX_DECODE_SRC 0             /* first block VA lo, hi */
#define APEX_DECODE_DST 2             /* first decoded texel VA lo, hi */
#define APEX_DECODE_SRC_ROW 4         /* bytes per block row */
#define APEX_DECODE_SRC_SLICE 5
#define APEX_DECODE_DST_ROW 6
#define APEX_DECODE_DST_SLICE 7
#define APEX_DECODE_WIDTH 8           /* texels */
#define APEX_DECODE_HEIGHT 9
#define APEX_DECODE_LAYERS 10
#define APEX_DECODE_KIND 11
/* Block kinds; 16-byte blocks are RGBA8 and both RG11 kinds. */
#define APEX_ETC2_RGB8 0
#define APEX_ETC2_RGBA1 1
#define APEX_ETC2_RGBA8 2
#define APEX_EAC_R11 3
#define APEX_EAC_R11_SNORM 4
#define APEX_EAC_RG11 5
#define APEX_EAC_RG11_SNORM 6

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
#define APEX_CLEAR_BYTES 7            /* texel bytes: 1, 2, 3, 4, 6, 8, 12 or 16 */
#define APEX_CLEAR_PATTERN 8          /* packed texel, 4 words */
#define APEX_CLEAR_MASK 12            /* texel bits to write, 4 words */

#endif
