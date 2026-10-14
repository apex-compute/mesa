/* SPDX-License-Identifier: MIT */
#ifndef APEX_GRAPHICS_H
#define APEX_GRAPHICS_H
#include "apex_pipeline.h"
#include "vk_shader.h"
#include "compiler/shader_enums.h"

/* A graphics stage compiled to a P7 vertex or fragment program; a vertex
 * shader feeding a geometry shader compiles to the records program and a
 * geometry shader to the geometry program (apex_graphics.c, Geometry
 * shading). */
struct apex_shader {
   struct vk_shader vk;
   struct apex_program program;
   struct vk_descriptor_set_layout *set_layouts[MESA_VK_MAX_DESCRIPTOR_SETS];
   /* The last pre-rasterization stage with transform feedback outputs: the
    * capture program (code.size nonzero) and each buffer's stride. */
   struct apex_program capture;
   uint16_t xfb_strides[4];
   struct {
      /* Output record bytes per vertex and clip distances written; robust
       * vertex input bindings. */
      uint32_t stride;
      uint8_t clip_distances;
      bool robust;
      /* Feeding a geometry shader: the input record slots written. */
      uint64_t records;
      /* Stores or atomics to memory, which a pass replayed per region would
       * repeat. */
      bool stores;
   } vertex;
   struct {
      /* The pass-through vertex program, its output record bytes and clip
       * distances. */
      struct apex_program draw;
      uint32_t stride;
      uint8_t clip_distances;
      /* Input and output record slots; per invocation, the vertices and list
       * primitives of `primitive_vertices` vertices; invocations per input
       * primitive of `input_vertices`. */
      uint64_t inputs, outputs;
      uint16_t max_vertices, max_primitives, invocations;
      uint8_t primitive_vertices, input_vertices;
   } geometry;
};

extern const struct vk_device_shader_ops apex_device_shader_ops;

/* Implemented by the command buffer. */
void apex_cmd_bind_shaders(struct vk_command_buffer *cmd, uint32_t count,
                           const mesa_shader_stage *stages, struct vk_shader **const shaders);

/* Device-owned internal programs, compiled on first use. */
VkResult apex_internal_program(struct apex_device *device, enum apex_internal which,
                               struct apex_program **out);
/* The fragment program of draws without a fragment shader: it exports
 * coverage only, for depth, stencil and occlusion. */
VkResult apex_empty_fragment_program(struct apex_device *device, struct apex_program **out);
/* Fragment program header flags (Docs/isa.md, Program header). */
bool apex_program_sample_shading(const struct apex_program *program);
bool apex_program_late_depth(const struct apex_program *program);
void apex_graphics_finish(struct apex_device *device);
#endif
