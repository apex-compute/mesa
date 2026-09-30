/* SPDX-License-Identifier: MIT */
#ifndef APEX_GRAPHICS_H
#define APEX_GRAPHICS_H
#include "apex_pipeline.h"
#include "vk_shader.h"
#include "compiler/shader_enums.h"
#include "apex_draw.h"

/* A graphics stage compiled to one native program. Vertex records hold the
 * clip position then each written generic location; `slot` maps varying
 * locations to record words (negative when absent). */
struct apex_shader {
   struct vk_shader vk;
   struct apex_program program;
   struct vk_descriptor_set_layout *set_layouts[MESA_VK_MAX_DESCRIPTOR_SETS];
   /* Vertex shaders of pipelines without a fragment shader: the empty
    * fragment kernel that rasterizes for depth, stencil and occlusion. */
   struct apex_shader *depth_only;
   struct {
      int16_t slot[VARYING_SLOT_MAX];
      uint32_t stride;
      uint8_t clip_distances, cull_distances;
      /* Transform feedback: APEX_DRAW_XFB_OUTPUTS entries and buffer strides. */
      uint32_t xfb[APEX_DRAW_MAX_XFB_OUTPUTS];
      uint16_t xfb_strides[APEX_DRAW_MAX_XFB_BUFFERS];
      uint8_t xfb_count;
   } vertex;
};

extern const struct vk_device_shader_ops apex_device_shader_ops;

/* Implemented by the command buffer. */
void apex_cmd_bind_shaders(struct vk_command_buffer *cmd, uint32_t count,
                           const mesa_shader_stage *stages, struct vk_shader **const shaders);

/* Device-owned internal programs, compiled on first use. */
VkResult apex_internal_program(struct apex_device *device, enum apex_internal which,
                               struct apex_program **out);
VkResult apex_resolve_program(struct apex_device *device, VkFormat format,
                              struct apex_program **out);
void apex_graphics_finish(struct apex_device *device);
#endif
