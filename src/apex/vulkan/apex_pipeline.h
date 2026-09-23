/* SPDX-License-Identifier: MIT */
#ifndef APEX_PIPELINE_H
#define APEX_PIPELINE_H

#include "apex/apex.h"
#include "apex_device.h"
#include "vk_pipeline.h"

/* Fixed compute launch ABI: one SSBO at set 0/binding 0, local size 16x1x1.
 * No device discovery or API capability advertisement is provided here. */
struct apex_pipeline {
   struct vk_pipeline vk;
   struct apex_compile_result code;
   struct apex_bo program;
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
