/* SPDX-License-Identifier: MIT */
#ifndef APEX_PIPELINE_H
#define APEX_PIPELINE_H

#include "apex/apex.h"
#include "apex_device.h"
#include "vk_pipeline.h"
#include "vk_pipeline_layout.h"
#include "util/bitset.h"
#include "nir_builder.h"

/* One P7 program and its descriptor-table layout: rows for every set
 * binding, the zero sentinel row, push constants and the trailer
 * (apex_job.h). The owner holds references to the set layouts. The binary
 * uploads lazily on the submit thread. `table` is false only for
 * layout-less fixtures. User data s0:s1 hold the table's GPUVA; compute
 * user data s2:s3 the address of the dispatch grid, vertex user data s2 the
 * view index. `max_workgroups` bounds a DISPATCH by the private arena. */
#define APEX_USER_GRID 2
struct apex_program {
   struct apex_compile_result code;
   struct apex_bo bo;
   bool table;
   uint32_t set_count;
   const struct apex_set_layout *set_layouts[MESA_VK_MAX_DESCRIPTOR_SETS];
   uint32_t set_offsets[MESA_VK_MAX_DESCRIPTOR_SETS];
   uint32_t descriptor_count;
   uint32_t push_size;
   uint32_t max_workgroups;
   BITSET_DECLARE(used_descriptors, APEX_MAX_DESCRIPTORS);
};

/* Byte offset of the trailer after descriptors and push data. */
static inline uint32_t
apex_program_trailer(const struct apex_program *program)
{
   return (program->descriptor_count + 1) * sizeof(union apex_descriptor) + program->push_size;
}

/* Fixed local shapes with 1-256 invocations. */
struct apex_pipeline {
   struct vk_pipeline vk;
   struct apex_program program;
   struct vk_pipeline_layout *layout;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(apex_pipeline, vk.base, VkPipeline,
                              VK_OBJECT_TYPE_PIPELINE);

/* Assigns descriptor rows for every set and the push extent for `stages`. */
bool apex_program_layout(struct apex_program *program, uint32_t set_count,
                         struct vk_descriptor_set_layout *const *sets,
                         uint32_t push_range_count, const VkPushConstantRange *ranges,
                         VkShaderStageFlags stages);
/* Lowers descriptors, push constants, global access and Int64 in place. */
bool apex_program_lower_resources(struct apex_program *program, struct nir_shader *nir);
/* Compiles lowered NIR and derives the private-arena workgroup limit. */
VkResult apex_program_compile(struct vk_device *device, struct apex_program *program,
                              struct nir_shader *nir);
void apex_program_finish(struct apex_device *device, struct apex_program *program);
/* Lowers texture and image instructions (apex_texture.c). */
bool apex_lower_textures(struct apex_program *program, struct nir_shader *nir);
/* FP16 bits in the low half of a 32-bit value to FP32 bits, and back. */
nir_def *apex_half_to_float(nir_builder *b, nir_def *h);
nir_def *apex_small_float_to_float(nir_builder *b, nir_def *v, unsigned mantissa);
/* FP32 to FP16 (mantissa 10, signed) or unsigned 11/10-bit floats. */
nir_def *apex_float_to_small(nir_builder *b, nir_def *f, unsigned mantissa, bool sign);

VKAPI_ATTR VkResult VKAPI_CALL
apex_CreateComputePipelines(VkDevice device, VkPipelineCache cache,
                           uint32_t count,
                           const VkComputePipelineCreateInfo *infos,
                           const VkAllocationCallbacks *alloc,
                           VkPipeline *pipelines);

#endif
