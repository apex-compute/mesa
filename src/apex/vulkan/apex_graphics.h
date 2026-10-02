/* SPDX-License-Identifier: MIT */
#ifndef APEX_GRAPHICS_H
#define APEX_GRAPHICS_H
#include "apex_pipeline.h"
#include "vk_shader.h"
#include "compiler/shader_enums.h"

/* A graphics stage compiled to one P7 vertex or fragment program. */
struct apex_shader {
   struct vk_shader vk;
   struct apex_program program;
   struct vk_descriptor_set_layout *set_layouts[MESA_VK_MAX_DESCRIPTOR_SETS];
   struct {
      /* Output record bytes per vertex and clip distances written. */
      uint32_t stride;
      uint8_t clip_distances;
      /* With transform feedback outputs: the capture program (code.size
       * nonzero) and each buffer's stride. */
      struct apex_program capture;
      uint16_t xfb_strides[4];
   } vertex;
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
