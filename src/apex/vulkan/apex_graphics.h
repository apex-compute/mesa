/* SPDX-License-Identifier: MIT */
#ifndef APEX_GRAPHICS_H
#define APEX_GRAPHICS_H
#include "apex_pipeline.h"
#include "vk_shader.h"
#include "compiler/shader_enums.h"

/* A graphics stage compiled to one native program. Vertex records hold the
 * clip position then each written generic location; `slot` maps varying
 * locations to record words (negative when absent). */
struct apex_shader {
   struct vk_shader vk;
   struct apex_program program;
   struct vk_descriptor_set_layout *set_layouts[MESA_VK_MAX_DESCRIPTOR_SETS];
   struct {
      int8_t slot[VARYING_SLOT_MAX];
      uint32_t stride;
   } vertex;
};

extern const struct vk_device_shader_ops apex_device_shader_ops;

/* Implemented by the command buffer. */
void apex_cmd_bind_shaders(struct vk_command_buffer *cmd, uint32_t count,
                           const mesa_shader_stage *stages, struct vk_shader **const shaders);

/* Device-owned internal triangle setup program, compiled on first use. */
VkResult apex_setup_program(struct apex_device *device, struct apex_program **out);
void apex_graphics_finish(struct apex_device *device);
#endif
