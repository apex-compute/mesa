/* SPDX-License-Identifier: MIT */
#ifndef APEX_PIPELINE_H
#define APEX_PIPELINE_H

#include "apex/apex.h"
#include "apex_device.h"
#include "vk_pipeline.h"
#include "vk_pipeline_layout.h"
#include "util/bitset.h"

/* Fixed local shapes totaling 16 invocations. No API capability advertisement. */
struct apex_pipeline {
   struct vk_pipeline vk;
   struct apex_compile_result code;
   struct apex_bo program;
   struct vk_pipeline_layout *layout;
   uint32_t set_offsets[MESA_VK_MAX_DESCRIPTOR_SETS];
   uint32_t descriptor_count;
   uint32_t push_size;
   BITSET_DECLARE(used_descriptors, APEX_MAX_DESCRIPTORS);
};

VK_DEFINE_NONDISP_HANDLE_CASTS(apex_pipeline, vk.base, VkPipeline,
                              VK_OBJECT_TYPE_PIPELINE);

VKAPI_ATTR VkResult VKAPI_CALL
apex_CreateComputePipelines(VkDevice device, VkPipelineCache cache,
                           uint32_t count,
                           const VkComputePipelineCreateInfo *infos,
                           const VkAllocationCallbacks *alloc,
                           VkPipeline *pipelines);

#endif
