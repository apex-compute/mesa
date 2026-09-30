/* SPDX-License-Identifier: MIT */
#include "apex_graphics.h"
#include "apex_pipeline.h"
#include "vk_alloc.h"
#include "vk_buffer.h"
#include "vk_command_buffer.h"
#include "vk_common_entrypoints.h"
#include "vk_device.h"
#include "vk_instance.h"
#include "vk_physical_device.h"
#include "vk_sampler.h"
#include "util/u_math.h"
#include "util/format_r11g11b10f.h"
#include "util/half_float.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(x) do { if (!(x)) { \
   fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); abort(); \
} } while (0)

static uint32_t
word(const uint8_t *bytes)
{
   uint32_t value;
   memcpy(&value, bytes, sizeof(value));
   return util_le32_to_cpu(value);
}

static void
write_fixture(const char *directory, const char *name, VkPipeline handle,
              uint32_t bias, uint32_t scale, unsigned invocations)
{
   struct apex_pipeline *p = apex_pipeline_from_handle(handle);
   char path[4096];
   CHECK(snprintf(path, sizeof(path), "%s/%s.apx", directory, name) < sizeof(path));
   FILE *f = fopen(path, "wb");
   CHECK(f);
   CHECK(fwrite(p->program.code.data, 1, p->program.code.size, f) == p->program.code.size);
   CHECK(fclose(f) == 0);
   uint32_t input[800], expected[800];
   for (unsigned i = 0; i < 800; i++) {
      input[i] = util_cpu_to_le32(0xd00d0000 + i);
      expected[i] = i < invocations ? util_cpu_to_le32(bias + i * scale) : input[i];
   }
   CHECK(snprintf(path, sizeof(path), "%s/%s.input.bin", directory, name) < sizeof(path));
   f = fopen(path, "wb");
   CHECK(f && fwrite(input, 1, sizeof(input), f) == sizeof(input));
   CHECK(fclose(f) == 0);
   CHECK(snprintf(path, sizeof(path), "%s/%s.expected.bin", directory, name) < sizeof(path));
   f = fopen(path, "wb");
   CHECK(f && fwrite(expected, 1, sizeof(expected), f) == sizeof(expected));
   CHECK(fclose(f) == 0);
}

static void *VKAPI_PTR
fail_alloc(void *data, size_t size, size_t alignment, VkSystemAllocationScope scope)
{
   return NULL;
}

static void
test_descriptors(struct vk_physical_device *physical, const char *path,
                 const char *fixture, const char *output, bool dynamic, bool pointers)
{
   bool images = !strcmp(fixture, "mesa-images");
   struct apex_device device;
   const float priority = 1;
   const VkDeviceQueueCreateInfo queue_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueCount = 1, .pQueuePriorities = &priority,
   };
   const VkDeviceCreateInfo device_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &queue_info,
   };
   CHECK(apex_device_init(&device, physical, &device_info, NULL, -1) == VK_SUCCESS);
   /* Compiler/layout test only: an offline device without a fd.
    * No device-memory allocation, submission or ioctl occurs in this test. */
   VkDevice dev = apex_device_to_handle(&device);
   const struct vk_device_dispatch_table *v = &device.vk.dispatch_table;
   if (images) {
      CHECK(v->CreateSampler && v->DestroySampler);
      VkSamplerCreateInfo sampler_info = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
         .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
         .addressModeV = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT,
         .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
         .borderColor = VK_BORDER_COLOR_INT_OPAQUE_WHITE};
      VkSampler samplers[2];
      for (unsigned i = 0; i < 2; i++) {
         CHECK(v->CreateSampler(dev, &sampler_info, NULL, &samplers[i]) == VK_SUCCESS);
         struct vk_sampler *s = vk_sampler_from_handle(samplers[i]);
         CHECK(s && s->address_mode_u == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER &&
               s->address_mode_v == VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT &&
               s->address_mode_w == VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE);
         CHECK(s->border_color == VK_BORDER_COLOR_INT_OPAQUE_WHITE &&
               s->border_color_value.uint32[0] == 1 && s->border_color_value.uint32[3] == 1);
      }
      CHECK(samplers[0] != samplers[1]);
      VkAllocationCallbacks allocation = *vk_default_allocator();
      allocation.pfnAllocation = fail_alloc;
      VkSampler failed_sampler = samplers[0];
      CHECK(v->CreateSampler(dev, &sampler_info, &allocation, &failed_sampler) ==
            VK_ERROR_OUT_OF_HOST_MEMORY && !failed_sampler);
      for (unsigned i = 0; i < 2; i++)
         v->DestroySampler(dev, samplers[i], NULL);
      v->DestroySampler(dev, VK_NULL_HANDLE, NULL);
   }
   FILE *f = fopen(path, "rb");
   CHECK(f && fseek(f, 0, SEEK_END) == 0);
   long size = ftell(f);
   CHECK(size > 0 && size % 4 == 0);
   rewind(f);
   uint32_t *spirv = malloc(size);
   CHECK(spirv && fread(spirv, 1, size, f) == size && !fclose(f));
   VkShaderModule module;
   const VkShaderModuleCreateInfo module_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = size, .pCode = spirv,
   };
   CHECK(v->CreateShaderModule(dev, &module_info, NULL, &module) == VK_SUCCESS);
   free(spirv);
   VkDescriptorSetLayout sets[2];
   VkDescriptorType storage = dynamic ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
   VkDescriptorType uniform = dynamic ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
   for (unsigned s = 0; s < 2; s++) {
      VkDescriptorSetLayoutBinding bindings[] = {
         {.binding = s ? 5 : 3, .descriptorCount = s ? 2 : 1,
          .descriptorType = s && images ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : storage,
          .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
         {.binding = 2, .descriptorCount = 2, .descriptorType = pointers ? storage : uniform,
          .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
      };
      VkDescriptorSetLayoutCreateInfo info = {
         .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
         .bindingCount = s ? 2 : 1, .pBindings = bindings,
      };
      CHECK(v->CreateDescriptorSetLayout(dev, &info, NULL, &sets[s]) == VK_SUCCESS);
   }
   const VkDescriptorPoolSize sizes[] = {
      {storage, 7}, {uniform, images ? 6 : 2},
      /* Static descriptors cannot satisfy a dynamic descriptor budget. */
      {images ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
       images ? 2 : dynamic ? 100 : 0},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, dynamic ? 100 : 0},
   };
   const VkDescriptorPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
      .maxSets = 4, .poolSizeCount = pointers ? 1 : images ? 3 : dynamic ? 4 : 2, .pPoolSizes = sizes,
   };
   VkDescriptorPool pool;
   CHECK(v->CreateDescriptorPool(dev, &pool_info, NULL, &pool) == VK_SUCCESS);
   VkDescriptorSetAllocateInfo alloc = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = pool, .descriptorSetCount = 2, .pSetLayouts = sets,
   };
   VkDescriptorSet allocated[2], extra;
   CHECK(v->AllocateDescriptorSets(dev, &alloc, allocated) == VK_SUCCESS);
   alloc.descriptorSetCount = 1;
   alloc.pSetLayouts = &sets[1];
   CHECK(v->AllocateDescriptorSets(dev, &alloc, &extra) == VK_ERROR_OUT_OF_POOL_MEMORY && !extra);
   alloc.pSetLayouts = &sets[0];
   CHECK(v->AllocateDescriptorSets(dev, &alloc, &extra) == VK_SUCCESS);
   CHECK(v->FreeDescriptorSets(dev, pool, 1, &allocated[1]) == VK_SUCCESS);
   alloc.pSetLayouts = &sets[1];
   CHECK(v->AllocateDescriptorSets(dev, &alloc, &allocated[1]) == VK_SUCCESS);
   v->DestroyDescriptorPool(dev, pool, NULL);
   const VkPushConstantRange push_ranges[] = {
      {VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 16, 48},
      {VK_SHADER_STAGE_VERTEX_BIT, 128, 128},
   };
   const VkPipelineLayoutCreateInfo layout_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 2, .pSetLayouts = sets,
      .pushConstantRangeCount = 2, .pPushConstantRanges = push_ranges,
   };
   VkPipelineLayout layout;
   CHECK(v->CreatePipelineLayout(dev, &layout_info, NULL, &layout) == VK_SUCCESS);
   const VkComputePipelineCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, .layout = layout,
      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = module, .pName = "main"},
   };
   VkPipeline pipeline;
   CHECK(v->CreateComputePipelines(dev, VK_NULL_HANDLE, 1, &info, NULL, &pipeline) == VK_SUCCESS);
   struct apex_pipeline *p = apex_pipeline_from_handle(pipeline);
   CHECK(p->program.descriptor_count == (images ? 7 : 5) && p->program.set_offsets[0] == 0 &&
         p->program.set_offsets[1] == 1);
   CHECK(p->program.push_size == 64);
   CHECK(p->layout->dynamic_descriptor_offset[1] == (dynamic ? 1 : 0));
   VkDescriptorSetLayout swapped[] = {sets[1], sets[0]};
   VkPipelineLayoutCreateInfo wrong_layout_info = layout_info;
   wrong_layout_info.pSetLayouts = swapped;
   VkPipelineLayout wrong_layout;
   CHECK(v->CreatePipelineLayout(dev, &wrong_layout_info, NULL, &wrong_layout) == VK_SUCCESS);
   VkComputePipelineCreateInfo wrong_info = info;
   wrong_info.layout = wrong_layout;
   VkPipeline rejected;
   CHECK(v->CreateComputePipelines(dev, VK_NULL_HANDLE, 1, &wrong_info, NULL, &rejected) ==
         VK_ERROR_FEATURE_NOT_PRESENT && !rejected);
   v->DestroyPipelineLayout(dev, wrong_layout, NULL);
   const VkDescriptorSetLayoutBinding wrong_types[] = {
      {.binding = 2, .descriptorType = pointers ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
       .descriptorCount = 2, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
      {.binding = 5, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
       .descriptorCount = 2, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
   };
   const VkDescriptorSetLayoutCreateInfo wrong_set_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 2, .pBindings = wrong_types,
   };
   swapped[0] = sets[0];
   CHECK(v->CreateDescriptorSetLayout(dev, &wrong_set_info, NULL, &swapped[1]) == VK_SUCCESS);
   CHECK(v->CreatePipelineLayout(dev, &wrong_layout_info, NULL, &wrong_layout) == VK_SUCCESS);
   wrong_info.layout = wrong_layout;
   CHECK(v->CreateComputePipelines(dev, VK_NULL_HANDLE, 1, &wrong_info, NULL, &rejected) ==
         VK_ERROR_FEATURE_NOT_PRESENT && !rejected);
   v->DestroyPipelineLayout(dev, wrong_layout, NULL);
   v->DestroyDescriptorSetLayout(dev, swapped[1], NULL);
   const VkPushConstantRange rejected_ranges[] = {
      {VK_SHADER_STAGE_COMPUTE_BIT, 16, 244},
      {VK_SHADER_STAGE_COMPUTE_BIT, 18, 48},
      {VK_SHADER_STAGE_COMPUTE_BIT, 16, 49},
      {VK_SHADER_STAGE_COMPUTE_BIT, 256, 4},
   };
   wrong_layout_info = layout_info;
   wrong_layout_info.pushConstantRangeCount = 1;
   for (unsigned r = 0; r < ARRAY_SIZE(rejected_ranges); r++) {
      wrong_layout_info.pPushConstantRanges = &rejected_ranges[r];
      CHECK(v->CreatePipelineLayout(dev, &wrong_layout_info, NULL, &wrong_layout) == VK_SUCCESS);
      wrong_info.layout = wrong_layout;
      CHECK(v->CreateComputePipelines(dev, VK_NULL_HANDLE, 1, &wrong_info, NULL, &rejected) ==
            VK_ERROR_FEATURE_NOT_PRESENT && !rejected);
      v->DestroyPipelineLayout(dev, wrong_layout, NULL);
   }
   v->DestroyPipelineLayout(dev, layout, NULL);
   for (unsigned s = 0; s < 2; s++) v->DestroyDescriptorSetLayout(dev, sets[s], NULL);
   v->DestroyShaderModule(dev, module, NULL);
   CHECK(p->layout->set_count == 2); /* The pipeline retains layout metadata. */
   if (output) {
      char name[4096];
      CHECK(snprintf(name, sizeof(name), "%s/%s.apx", output, fixture) < sizeof(name));
      f = fopen(name, "wb");
      CHECK(f && fwrite(p->program.code.data, 1, p->program.code.size, f) == p->program.code.size && !fclose(f));
   }
   v->DestroyPipeline(dev, pipeline, NULL);
   apex_device_finish(&device);
}

static void
test_rgba(struct vk_physical_device *physical, const char *path)
{
   struct apex_device device;
   const float priority = 1;
   const VkDeviceQueueCreateInfo qi = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueCount = 1,
      .pQueuePriorities = &priority,
   };
   const VkDeviceCreateInfo di = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qi,
   };
   CHECK(apex_device_init(&device, physical, &di, NULL, -1) == VK_SUCCESS);
   VkDevice dev = apex_device_to_handle(&device);
   const struct vk_device_dispatch_table *v = &device.vk.dispatch_table;
   FILE *f = fopen(path, "rb");
   CHECK(f && !fseek(f, 0, SEEK_END));
   long bytes = ftell(f);
   CHECK(bytes > 0 && bytes % 4 == 0);
   rewind(f);
   uint32_t *spirv = malloc(bytes);
   CHECK(spirv && fread(spirv, 1, bytes, f) == bytes && !fclose(f));
   VkShaderModule module;
   VkShaderModuleCreateInfo mi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                  .codeSize = bytes, .pCode = spirv};
   CHECK(v->CreateShaderModule(dev, &mi, NULL, &module) == VK_SUCCESS);
   free(spirv);
   VkDescriptorSetLayoutBinding bindings[] = {
      {.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
       .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
      {.binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
       .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
   };
   VkDescriptorSetLayoutCreateInfo si = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                         .bindingCount = 2, .pBindings = bindings};
   VkDescriptorSetLayout set;
   CHECK(v->CreateDescriptorSetLayout(dev, &si, NULL, &set) == VK_SUCCESS);
   VkPipelineLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                    .setLayoutCount = 1, .pSetLayouts = &set};
   VkPipelineLayout layout;
   CHECK(v->CreatePipelineLayout(dev, &li, NULL, &layout) == VK_SUCCESS);
   VkComputePipelineCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .layout = layout, .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = module, .pName = "main"}};
   VkPipeline pipeline;
   CHECK(v->CreateComputePipelines(dev, VK_NULL_HANDLE, 1, &pi, NULL, &pipeline) == VK_SUCCESS);
   CHECK(apex_pipeline_from_handle(pipeline)->program.descriptor_count == 3); /* image: 2 rows */
   v->DestroyPipeline(dev, pipeline, NULL);
   v->DestroyPipelineLayout(dev, layout, NULL);
   v->DestroyDescriptorSetLayout(dev, set, NULL);
   v->DestroyShaderModule(dev, module, NULL);
   apex_device_finish(&device);
}

static void
test_dispatch(struct vk_physical_device *physical, const char *path, const char *output,
              bool grid, bool multiple)
{
   struct apex_device device;
   const float priority = 1;
   const VkDeviceQueueCreateInfo qi = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueCount = 1, .pQueuePriorities = &priority,
   };
   const VkDeviceCreateInfo di = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qi,
   };
   CHECK(apex_device_init(&device, physical, &di, NULL, -1) == VK_SUCCESS);
   VkDevice dev = apex_device_to_handle(&device);
   const struct vk_device_dispatch_table *v = &device.vk.dispatch_table;
   FILE *f = fopen(path, "rb");
   CHECK(f && !fseek(f, 0, SEEK_END));
   long bytes = ftell(f);
   CHECK(bytes > 0 && bytes % 4 == 0);
   rewind(f);
   uint32_t *spirv = malloc(bytes);
   CHECK(spirv && fread(spirv, 1, bytes, f) == bytes && !fclose(f));
   VkShaderModule module;
   const VkShaderModuleCreateInfo mi = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = bytes, .pCode = spirv,
   };
   CHECK(v->CreateShaderModule(dev, &mi, NULL, &module) == VK_SUCCESS);
   free(spirv);
   const VkDescriptorSetLayoutBinding bindings[] = {
      {.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1,
       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
      {.binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1,
       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
   };
   const VkDescriptorSetLayoutCreateInfo si = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = multiple ? 2 : 1, .pBindings = bindings,
   };
   VkDescriptorSetLayout set;
   CHECK(v->CreateDescriptorSetLayout(dev, &si, NULL, &set) == VK_SUCCESS);
   /* Nonzero push extent catches trailer-offset mistakes. */
   const VkPushConstantRange push = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 20};
   const VkPipelineLayoutCreateInfo li = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &set,
      .pushConstantRangeCount = multiple ? 0 : 1, .pPushConstantRanges = &push,
   };
   VkPipelineLayout layout;
   CHECK(v->CreatePipelineLayout(dev, &li, NULL, &layout) == VK_SUCCESS);
   const uint32_t value = 1;
   const VkSpecializationMapEntry entry = {0, 0, 4};
   const VkSpecializationInfo specialization = {
      .mapEntryCount = 1, .pMapEntries = &entry, .dataSize = 4, .pData = &value,
   };
   const VkComputePipelineCreateInfo pi = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, .layout = layout,
      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = module, .pName = "main",
         .pSpecializationInfo = grid ? &specialization : NULL},
   };
   VkPipeline pipeline;
   CHECK(v->CreateComputePipelines(dev, VK_NULL_HANDLE, 1, &pi, NULL, &pipeline) == VK_SUCCESS);
   struct apex_pipeline *p = apex_pipeline_from_handle(pipeline);
   CHECK(p->program.descriptor_count == (multiple ? 2 : 1) && p->program.push_size == (multiple ? 0 : 20));
   uint64_t padded_private = (uint64_t)word(p->program.code.data + 24) * 16;
   CHECK(p->program.max_workgroups && p->program.max_workgroups <= 1024);
   CHECK(padded_private * p->program.max_workgroups <= 2097152);
   CHECK(p->program.max_workgroups == 1024 || padded_private * (p->program.max_workgroups + 1) > 2097152);
   if (output) {
      char filename[4096];
      const char *name = multiple ? "mesa-multiple" : grid ? "mesa-dispatch-grid" : "mesa-dispatch";
      CHECK(snprintf(filename, sizeof(filename), "%s/%s.apx", output, name) < sizeof(filename));
      f = fopen(filename, "wb");
      CHECK(f && fwrite(p->program.code.data, 1, p->program.code.size, f) == p->program.code.size && !fclose(f));
   }
   v->DestroyPipeline(dev, pipeline, NULL);
   v->DestroyPipelineLayout(dev, layout, NULL);
   v->DestroyDescriptorSetLayout(dev, set, NULL);
   v->DestroyShaderModule(dev, module, NULL);
   apex_device_finish(&device);
}

static void
test_fill(struct vk_physical_device *physical, const char *output)
{
   struct apex_device device;
   const float priority = 1;
   VkDeviceQueueCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueCount = 1, .pQueuePriorities = &priority};
   VkDeviceCreateInfo di = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qi};
   CHECK(apex_device_init(&device, physical, &di, NULL, -1) == VK_SUCCESS);
   /* Compile/record only. Supply an address without creating a kernel BO. */
   VkDevice dev = apex_device_to_handle(&device);
   const struct vk_device_dispatch_table *v = &device.vk.dispatch_table;
   VkBuffer buffer;
   VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = 259, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
         VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
   CHECK(v->CreateBuffer(dev, &bi, NULL, &buffer) == VK_SUCCESS);
   vk_buffer_from_handle(buffer)->device_address = 0x1000020200ull;
   VkCommandPool pool;
   VkCommandPoolCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
   CHECK(v->CreateCommandPool(dev, &pi, NULL, &pool) == VK_SUCCESS);
   VkCommandBuffer cmd;
   VkCommandBufferAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = pool, .commandBufferCount = 1};
   CHECK(v->AllocateCommandBuffers(dev, &ai, &cmd) == VK_SUCCESS);
   VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
   CHECK(v->BeginCommandBuffer(cmd, &begin) == VK_SUCCESS);
   /* The driver fills and copies through CP packets; Mesa's compute fill
    * and copy kernels remain the unaligned-SYSTEM path and RTL fixtures. */
   struct vk_command_buffer *vk_cmd = vk_command_buffer_from_handle(cmd);
   struct vk_buffer *vk_buffer = vk_buffer_from_handle(buffer);
   VkDeviceAddressRangeKHR range = vk_device_address_range(vk_buffer, 4, 196);
   vk_meta_fill_memory(vk_cmd, &device.meta, &range, 0, 0xa5c31e79);
   range = vk_device_address_range(vk_buffer, 256, VK_WHOLE_SIZE);
   range.size &= ~3ull;
   vk_meta_fill_memory(vk_cmd, &device.meta, &range, 0, 0xbaadf00d);
   enum vk_meta_object_key_type key = VK_META_OBJECT_KEY_FILL_BUFFER;
   VkPipeline handle = vk_meta_lookup_pipeline(&device.meta, &key, sizeof(key));
   CHECK(handle);
   struct apex_pipeline *pipeline = apex_pipeline_from_handle(handle);
   CHECK(!pipeline->program.descriptor_count && pipeline->program.push_size == 16);
   if (output) {
      char filename[4096];
      CHECK(snprintf(filename, sizeof(filename), "%s/mesa-fill.apx", output) < sizeof(filename));
      FILE *f = fopen(filename, "wb");
      CHECK(f && fwrite(pipeline->program.code.data, 1, pipeline->program.code.size, f) == pipeline->program.code.size && !fclose(f));
   }
   for (unsigned chunk = 1; chunk <= 16; chunk *= 2) {
      const VkBufferCopy2 region = {.sType = VK_STRUCTURE_TYPE_BUFFER_COPY_2,
         .srcOffset = chunk, .dstOffset = 128 + chunk, .size = chunk * 3};
      const VkCopyBufferInfo2 copy = {.sType = VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
         .srcBuffer = buffer, .dstBuffer = buffer, .regionCount = 1, .pRegions = &region};
      vk_meta_copy_buffer(vk_cmd, &device.meta, &copy);
      struct { enum vk_meta_object_key_type type; uint32_t chunk; } copy_key = {
         VK_META_OBJECT_KEY_COPY_BUFFER, chunk,
      };
      handle = vk_meta_lookup_pipeline(&device.meta, &copy_key, sizeof(copy_key));
      CHECK(handle);
      pipeline = apex_pipeline_from_handle(handle);
      CHECK(!pipeline->program.descriptor_count && pipeline->program.push_size == 24);
      if (output) {
         char filename[4096];
         CHECK(snprintf(filename, sizeof(filename), "%s/mesa-copy%u.apx", output, chunk) < sizeof(filename));
         FILE *f = fopen(filename, "wb");
         CHECK(f && fwrite(pipeline->program.code.data, 1, pipeline->program.code.size, f) == pipeline->program.code.size && !fclose(f));
      }
   }
   CHECK(v->EndCommandBuffer(cmd) == VK_SUCCESS);
   v->DestroyCommandPool(dev, pool, NULL);
   v->DestroyBuffer(dev, buffer, NULL);
   apex_device_finish(&device);
}

static VkShaderModule
load_module(const struct vk_device_dispatch_table *v, VkDevice dev, const char *path)
{
   FILE *f = fopen(path, "rb");
   CHECK(f && !fseek(f, 0, SEEK_END));
   long bytes = ftell(f);
   CHECK(bytes > 0 && bytes % 4 == 0);
   rewind(f);
   uint32_t *spirv = malloc(bytes);
   CHECK(spirv && fread(spirv, 1, bytes, f) == bytes && !fclose(f));
   VkShaderModule module;
   VkShaderModuleCreateInfo mi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                  .codeSize = bytes, .pCode = spirv};
   CHECK(v->CreateShaderModule(dev, &mi, NULL, &module) == VK_SUCCESS);
   free(spirv);
   return module;
}

/* A dynamic-rendering pipeline builds the vertex kernel (vertex fetch,
 * indexed and direct paths) and the fragment kernel; NIR validation runs
 * on every pass in debug builds. */
/* `gles` adds the GLES 2 rasterization state: line polygon mode, wide
 * stippled smooth lines, a logic op, alpha to one and a depth/stencil
 * attachment; without `fragment_path` the vertex shader owns the depth-only
 * fragment kernel. */
static void
test_draw_pipeline(struct apex_device *device, const char *vertex_path, const char *fragment_path,
                   bool gles)
{
   VkDevice dev = apex_device_to_handle(device);
   const struct vk_device_dispatch_table *v = &device->vk.dispatch_table;
   VkShaderModule modules[2] = {load_module(v, dev, vertex_path),
                                fragment_path ? load_module(v, dev, fragment_path) : VK_NULL_HANDLE};
   const VkPipelineShaderStageCreateInfo stages[2] = {
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = modules[0], .pName = "main"},
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = modules[1], .pName = "main"},
   };
   const VkVertexInputBindingDescription binding = {0, 32, VK_VERTEX_INPUT_RATE_VERTEX};
   /* RGBA32F position and an RGB8 color whose missing alpha fetch fills. */
   const VkVertexInputAttributeDescription attributes[2] = {
      {0, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 0}, {1, 0, VK_FORMAT_R8G8B8_UNORM, 16},
   };
   const VkPipelineVertexInputStateCreateInfo vi = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      .vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = &binding,
      .vertexAttributeDescriptionCount = 2, .pVertexAttributeDescriptions = attributes,
   };
   const VkPipelineInputAssemblyStateCreateInfo ia = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
   };
   const VkPipelineViewportStateCreateInfo vp = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1, .scissorCount = 1,
   };
   const VkPipelineRasterizationProvokingVertexStateCreateInfoEXT provoking = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_PROVOKING_VERTEX_STATE_CREATE_INFO_EXT,
      .provokingVertexMode = VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT,
   };
   const VkPipelineRasterizationLineStateCreateInfoKHR line = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_LINE_STATE_CREATE_INFO_KHR, .pNext = &provoking,
      .lineRasterizationMode = VK_LINE_RASTERIZATION_MODE_RECTANGULAR_SMOOTH_KHR,
      .stippledLineEnable = VK_TRUE, .lineStippleFactor = 3, .lineStipplePattern = 0xf0f0,
   };
   const VkPipelineRasterizationStateCreateInfo rs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .pNext = gles ? (const void *)&line : &provoking,
      .polygonMode = gles ? VK_POLYGON_MODE_LINE : VK_POLYGON_MODE_FILL, .lineWidth = gles ? 3.0f : 1.0f,
   };
   const VkPipelineMultisampleStateCreateInfo ms = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT, .alphaToOneEnable = gles,
   };
   const VkPipelineDepthStencilStateCreateInfo dss = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .depthTestEnable = VK_TRUE, .depthWriteEnable = VK_TRUE, .depthCompareOp = VK_COMPARE_OP_LESS,
   };
   const VkPipelineColorBlendAttachmentState blend = {.colorWriteMask = 0xf};
   const VkPipelineColorBlendStateCreateInfo cb = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .logicOpEnable = gles, .logicOp = VK_LOGIC_OP_XOR,
      .attachmentCount = 1, .pAttachments = &blend,
   };
   const VkDynamicState dynamic[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
   const VkPipelineDynamicStateCreateInfo ds = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
      .dynamicStateCount = 2, .pDynamicStates = dynamic,
   };
   const VkFormat color = VK_FORMAT_R8G8B8A8_UNORM;
   const VkPipelineRenderingCreateInfo rendering = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .colorAttachmentCount = 1, .pColorAttachmentFormats = &color,
      .depthAttachmentFormat = gles ? VK_FORMAT_D24_UNORM_S8_UINT : VK_FORMAT_UNDEFINED,
      .stencilAttachmentFormat = gles ? VK_FORMAT_D24_UNORM_S8_UINT : VK_FORMAT_UNDEFINED,
   };
   const VkPipelineLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
   VkPipelineLayout layout;
   CHECK(v->CreatePipelineLayout(dev, &li, NULL, &layout) == VK_SUCCESS);
   const VkGraphicsPipelineCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &rendering,
      .stageCount = fragment_path ? 2 : 1, .pStages = stages, .pVertexInputState = &vi,
      .pInputAssemblyState = &ia, .pViewportState = &vp, .pRasterizationState = &rs,
      .pMultisampleState = &ms, .pDepthStencilState = gles ? &dss : NULL,
      .pColorBlendState = &cb, .pDynamicState = &ds, .layout = layout,
   };
   VkPipeline pipeline;
   CHECK(v->CreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &info, NULL, &pipeline) == VK_SUCCESS);
   v->DestroyPipeline(dev, pipeline, NULL);
   v->DestroyPipelineLayout(dev, layout, NULL);
   for (unsigned i = 0; i < 2; i++)
      if (modules[i])
         v->DestroyShaderModule(dev, modules[i], NULL);
}

/* apex_float_to_small against Mesa's reference conversions, through NIR
 * constant folding. */
static void
test_small_floats(void)
{
   static const float values[] = {0.0f, 1.0f, 0.5f, 3.14159f, 65000.0f, 1e10f, 6.1e-5f, 3e-6f, 1e-9f,
                                  -2.0f, INFINITY, 1.0f + 1.0f / 64, 1.0f + 3.0f / 128};
   const nir_shader_compiler_options options = {0};
   nir_builder b = nir_builder_init_simple_shader(MESA_SHADER_COMPUTE, &options, "small_floats");
   for (unsigned i = 0; i < ARRAY_SIZE(values); i++) {
      nir_def *f = nir_imm_float(&b, values[i]);
      nir_store_global(&b, apex_float_to_small(&b, f, 10, true), nir_imm_int64(&b, i * 12), .align_mul = 4);
      nir_store_global(&b, apex_float_to_small(&b, f, 6, false), nir_imm_int64(&b, i * 12 + 4), .align_mul = 4);
      nir_store_global(&b, apex_float_to_small(&b, f, 5, false), nir_imm_int64(&b, i * 12 + 8), .align_mul = 4);
   }
   bool progress;
   do {
      progress = false;
      NIR_PASS(progress, b.shader, nir_opt_copy_prop);
      NIR_PASS(progress, b.shader, nir_opt_constant_folding);
      NIR_PASS(progress, b.shader, nir_opt_dce);
   } while (progress);
   unsigned n = 0;
   nir_foreach_block(block, nir_shader_get_entrypoint(b.shader)) {
      nir_foreach_instr(instr, block) {
         if (instr->type != nir_instr_type_intrinsic ||
             nir_instr_as_intrinsic(instr)->intrinsic != nir_intrinsic_store_global)
            continue;
         nir_intrinsic_instr *store = nir_instr_as_intrinsic(instr);
         CHECK(nir_src_is_const(store->src[0]) && nir_src_is_const(store->src[1]));
         uint32_t got = nir_src_as_uint(store->src[0]);
         unsigned at = nir_src_as_uint(store->src[1]), i = at / 12, which = at % 12 / 4;
         uint32_t rgb = float3_to_r11g11b10f((float[3]){values[i], values[i], values[i]});
         uint32_t want = which == 0 ? _mesa_float_to_half(values[i]) :
                         which == 1 ? rgb & 0x7ff : rgb >> 22;
         if (got != want)
            fprintf(stderr, "%g mantissa %u: 0x%x, expected 0x%x\n", values[i],
                    which == 0 ? 10 : which == 1 ? 6 : 5, got, want);
         CHECK(got == want);
         n++;
      }
   }
   CHECK(n == 3 * ARRAY_SIZE(values));
   /* The fragment kernel packs R11G11B10 through its channel fields. */
   const struct util_format_description *desc = util_format_description(PIPE_FORMAT_R11G11B10_FLOAT);
   CHECK(desc->nr_channels == 3 && desc->channel[0].shift == 0 && desc->channel[0].size == 11 &&
         desc->channel[1].shift == 11 && desc->channel[2].shift == 22 && desc->channel[2].size == 10 &&
         desc->channel[2].type == UTIL_FORMAT_TYPE_FLOAT);
   ralloc_free(b.shader);
}

/* Offline backend coverage of the software graphics and sampling paths:
 * the internal kernels, a draw pipeline and a compute shader sampling
 * two combined image samplers (fetch, nearest and linear with border
 * replacement). */
static void
test_graphics_programs(struct vk_physical_device *physical, const char *texture_path,
                       const char *vertex_path, const char *fragment_path, const char *clip_path,
                       const char *clip_fragment_path, const char *xfb_path)
{
   struct apex_device device;
   const float priority = 1;
   const VkDeviceQueueCreateInfo qi = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueCount = 1,
      .pQueuePriorities = &priority,
   };
   const VkDeviceCreateInfo di = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qi,
   };
   CHECK(apex_device_init(&device, physical, &di, NULL, -1, APEX_TRANSPORT_NATIVE) == VK_SUCCESS);
   device.transport = APEX_TRANSPORT_DRM;
   test_small_floats();
   /* Border swizzles: format components, conversion to RGBA, view mapping. */
#define SW(r, g, b, a) ((r) | (g) << 3 | (b) << 6 | (a) << 9)
   const unsigned Z = APEX_SWIZZLE_0, O = APEX_SWIZZLE_1;
   const VkComponentMapping identity = {0}, reversed = {
      VK_COMPONENT_SWIZZLE_A, VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_R};
   CHECK(apex_border_swizzle(VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, &identity) == SW(0, 1, 2, 3));
   CHECK(apex_border_swizzle(VK_FORMAT_B8G8R8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, &reversed) == SW(3, 2, 1, 0));
   CHECK(apex_border_swizzle(VK_FORMAT_R8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, &identity) == SW(0, Z, Z, O));
   CHECK(apex_border_swizzle(VK_FORMAT_R8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT, &reversed) == SW(O, Z, Z, 0));
   CHECK(apex_border_swizzle(VK_FORMAT_A8_UNORM_KHR, VK_IMAGE_ASPECT_COLOR_BIT, &identity) == SW(Z, Z, Z, 3));
   CHECK(apex_border_swizzle(VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK, VK_IMAGE_ASPECT_COLOR_BIT, &identity) ==
         SW(0, 1, 2, O));
   const VkComponentMapping rrr1 = {VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_R,
                                    VK_COMPONENT_SWIZZLE_ONE};
   CHECK(apex_border_swizzle(VK_FORMAT_D24_UNORM_S8_UINT, VK_IMAGE_ASPECT_STENCIL_BIT, &rrr1) == SW(0, 0, 0, O));
   CHECK(apex_border_swizzle(VK_FORMAT_D32_SFLOAT, VK_IMAGE_ASPECT_DEPTH_BIT, &identity) == SW(0, Z, Z, O));
   /* Depth and stencil texels read (D or S, 0, 0, 1): word 0 bits 8..19. */
   uint32_t words[3];
   CHECK(apex_format_encode(VK_FORMAT_D24_UNORM_S8_UINT, VK_IMAGE_ASPECT_DEPTH_BIT, &identity, words));
   CHECK(((words[0] >> 8) & 0xfff) == SW(0, Z, Z, O));
   CHECK(apex_format_encode(VK_FORMAT_D24_UNORM_S8_UINT, VK_IMAGE_ASPECT_STENCIL_BIT, &identity, words));
   CHECK(((words[0] >> 8) & 0xfff) == SW(1, Z, Z, O) && (words[0] & (1u << 20)));
   CHECK(apex_format_encode(VK_FORMAT_D32_SFLOAT_S8_UINT, VK_IMAGE_ASPECT_DEPTH_BIT, &rrr1, words));
   CHECK(((words[0] >> 8) & 0xfff) == SW(0, 0, 0, O));
#undef SW
   /* Custom border colors reach the sampler row without a format. */
   VkDevice handle = apex_device_to_handle(&device);
   const VkSamplerCustomBorderColorCreateInfoEXT custom = {
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CUSTOM_BORDER_COLOR_CREATE_INFO_EXT,
      .customBorderColor.int32 = {-3, 7, 1 << 30, -1},
   };
   const VkSamplerCreateInfo sampler_info = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
      .pNext = &custom, .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
      .borderColor = VK_BORDER_COLOR_INT_CUSTOM_EXT};
   VkSampler sampler;
   CHECK(device.vk.dispatch_table.CreateSampler(handle, &sampler_info, NULL, &sampler) == VK_SUCCESS);
   CHECK(!memcmp(&((struct apex_sampler *)vk_sampler_from_handle(sampler))->row[4],
                 custom.customBorderColor.int32, 16));
   device.vk.dispatch_table.DestroySampler(handle, sampler, NULL);
   /* A 3D level's subresource spans its depth slices. */
   const VkImageCreateInfo volume = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_3D, .format = VK_FORMAT_R16_UNORM, .extent = {32, 48, 56},
      .mipLevels = 2, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_LINEAR, .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT};
   for (unsigned level = 0; level < 2; level++) {
      const VkImageSubresource2 sub = {.sType = VK_STRUCTURE_TYPE_IMAGE_SUBRESOURCE_2,
         .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, 0}};
      const VkDeviceImageSubresourceInfo query = {
         .sType = VK_STRUCTURE_TYPE_DEVICE_IMAGE_SUBRESOURCE_INFO,
         .pCreateInfo = &volume, .pSubresource = &sub};
      VkSubresourceLayout2 layout = {.sType = VK_STRUCTURE_TYPE_SUBRESOURCE_LAYOUT_2};
      device.vk.dispatch_table.GetDeviceImageSubresourceLayout(handle, &query, &layout);
      /* Rows pad to 64 bytes: level 0 is 64 x 48 x 56, level 1 64 x 24 x 28 at 172032. */
      const VkSubresourceLayout *l = &layout.subresourceLayout;
      CHECK(l->offset == (level ? 64 * 48 * 56 : 0) && l->rowPitch == 64 &&
            l->depthPitch == 64u * (48 >> level) && l->size == l->depthPitch * (56 >> level));
   }
   /* Every internal kernel compiles through the backend. */
   for (unsigned k = 0; k < APEX_INTERNAL_COUNT; k++) {
      struct apex_program *internal;
      CHECK(apex_internal_program(&device, k, &internal) == VK_SUCCESS);
      CHECK(internal->code.size > 48);
   }
   test_draw_pipeline(&device, vertex_path, fragment_path, false);
   test_draw_pipeline(&device, clip_path, clip_fragment_path, true);
   test_draw_pipeline(&device, clip_path, NULL, true);
   test_draw_pipeline(&device, xfb_path, fragment_path, false);
   VkDevice dev = apex_device_to_handle(&device);
   const struct vk_device_dispatch_table *v = &device.vk.dispatch_table;
   VkShaderModule module = load_module(v, dev, texture_path);
   VkDescriptorSetLayoutBinding bindings[] = {
      {.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
       .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
      {.binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
       .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
      {.binding = 2, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
       .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
   };
   VkDescriptorSetLayoutCreateInfo si = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                         .bindingCount = 3, .pBindings = bindings};
   VkDescriptorSetLayout set;
   CHECK(v->CreateDescriptorSetLayout(dev, &si, NULL, &set) == VK_SUCCESS);
   VkPipelineLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                    .setLayoutCount = 1, .pSetLayouts = &set};
   VkPipelineLayout layout;
   CHECK(v->CreatePipelineLayout(dev, &li, NULL, &layout) == VK_SUCCESS);
   VkComputePipelineCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .layout = layout, .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = module, .pName = "main"}};
   VkPipeline pipeline;
   CHECK(v->CreateComputePipelines(dev, VK_NULL_HANDLE, 1, &pi, NULL, &pipeline) == VK_SUCCESS);
   /* Buffer row, two three-row combined descriptors. */
   CHECK(apex_pipeline_from_handle(pipeline)->program.descriptor_count == 7);
   v->DestroyPipeline(dev, pipeline, NULL);
   v->DestroyPipelineLayout(dev, layout, NULL);
   v->DestroyDescriptorSetLayout(dev, set, NULL);
   v->DestroyShaderModule(dev, module, NULL);
   apex_device_finish(&device);
}

/* bench.comp through the Vulkan lowering, run on the ISA model with a table
 * laid out as drm_prepare writes it: descriptor rows, the robust sentinel,
 * the push image and the dispatch trailer. */
static void
test_bench(struct vk_physical_device *physical, const char *path)
{
   struct apex_device device;
   const float priority = 1;
   const VkDeviceQueueCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueCount = 1, .pQueuePriorities = &priority};
   const VkDeviceCreateInfo di = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qi};
   CHECK(apex_device_init(&device, physical, &di, NULL, -1, APEX_TRANSPORT_NATIVE) == VK_SUCCESS);
   device.transport = APEX_TRANSPORT_DRM;
   VkDevice dev = apex_device_to_handle(&device);
   const struct vk_device_dispatch_table *v = &device.vk.dispatch_table;
   FILE *f = fopen(path, "rb");
   CHECK(f && !fseek(f, 0, SEEK_END));
   long bytes = ftell(f);
   rewind(f);
   uint32_t *spirv = malloc(bytes);
   CHECK(spirv && fread(spirv, 1, bytes, f) == bytes && !fclose(f));
   VkShaderModule module;
   const VkShaderModuleCreateInfo mi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = bytes, .pCode = spirv};
   CHECK(v->CreateShaderModule(dev, &mi, NULL, &module) == VK_SUCCESS);
   free(spirv);
   const VkDescriptorSetLayoutBinding bindings[] = {
      {.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1,
       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
      {.binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1,
       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
   };
   const VkDescriptorSetLayoutCreateInfo si = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .bindingCount = 2, .pBindings = bindings};
   VkDescriptorSetLayout set;
   CHECK(v->CreateDescriptorSetLayout(dev, &si, NULL, &set) == VK_SUCCESS);
   const VkPushConstantRange push = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 8};
   const VkPipelineLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &set, .pushConstantRangeCount = 1, .pPushConstantRanges = &push};
   VkPipelineLayout layout;
   CHECK(v->CreatePipelineLayout(dev, &li, NULL, &layout) == VK_SUCCESS);
   const VkComputePipelineCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .layout = layout, .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = module, .pName = "main"}};
   VkPipeline pipeline;
   CHECK(v->CreateComputePipelines(dev, VK_NULL_HANDLE, 1, &pi, NULL, &pipeline) == VK_SUCCESS);
   struct apex_program *program = &apex_pipeline_from_handle(pipeline)->program;
   CHECK(program->descriptor_count == 2 && program->push_size == 8 && !program->code.spills);
   const unsigned groups = 3, count = 2000, stride = groups * 256;
   const uint64_t table_va = 0x10000, src_va = 0x200000, dst_va = 0x400000;
   unsigned trailer = apex_program_trailer(program);
   uint8_t *table = calloc(1, trailer + sizeof(struct apex_dispatch_parameters));
   uint32_t *src = malloc(count * 16), *dst = calloc(1, count * 16);
   for (unsigned i = 0; i < count * 4; i++)
      src[i] = util_cpu_to_le32(0x9e3779b9u * (i + 1));
   union apex_descriptor *rows = (void *)table;
   for (unsigned r = 0; r <= program->descriptor_count; r++)
      rows[r].buffer.flags = util_cpu_to_le32(APEX_BUFFER_ROBUST);
   rows[0].buffer = (struct apex_buffer_descriptor){src_va, 0, count * 16, APEX_BUFFER_ROBUST};
   rows[1].buffer = (struct apex_buffer_descriptor){dst_va, 0, count * 16, APEX_BUFFER_ROBUST};
   uint32_t *push_image = (void *)(table + (program->descriptor_count + 1) * sizeof(union apex_descriptor));
   push_image[0] = 2; /* copy */
   push_image[1] = count;
   struct apex_dispatch_parameters *parameters = (void *)(table + trailer);
   *parameters = (struct apex_dispatch_parameters){.base = {0, groups, groups}, .groups = {groups, 1, 1}};
   struct apex_sim_region regions[] = {
      {table_va, table, trailer + sizeof(*parameters)}, {src_va, src, count * 16}, {dst_va, dst, count * 16},
   };
   uint32_t user[16] = {table_va}, grid[3] = {groups, 1, 1};
   uint64_t executed;
   char diagnostic[256] = "";
   if (apex_simulate(program->code.data, program->code.size, user, grid, 0, NULL, regions, 3,
                     &executed, diagnostic)) {
      fprintf(stderr, "simulation: %s\n", diagnostic);
      abort();
   }
   for (unsigned i = 0; i < count * 4; i++)
      CHECK(dst[i] == src[i]);
   (void)stride;
   free(table);
   free(src);
   free(dst);
   v->DestroyPipeline(dev, pipeline, NULL);
   v->DestroyPipelineLayout(dev, layout, NULL);
   v->DestroyDescriptorSetLayout(dev, set, NULL);
   v->DestroyShaderModule(dev, module, NULL);
   apex_device_finish(&device);
}

/* The apex-bench scenes' graphics pipelines (software raster kernels until
 * the fixed-function path lands) compile without spills. */
static void
test_graphics(struct vk_physical_device *physical, const char *vs, const char *fs, bool game)
{
   struct apex_device device;
   const float priority = 1;
   const VkDeviceQueueCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueCount = 1, .pQueuePriorities = &priority};
   const VkDeviceCreateInfo di = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qi};
   CHECK(apex_device_init(&device, physical, &di, NULL, -1, APEX_TRANSPORT_NATIVE) == VK_SUCCESS);
   device.transport = APEX_TRANSPORT_DRM;
   VkDevice dev = apex_device_to_handle(&device);
   const struct vk_device_dispatch_table *v = &device.vk.dispatch_table;
   VkShaderModule modules[2];
   const char *paths[2] = {vs, fs};
   for (unsigned m = 0; m < 2; m++) {
      FILE *f = fopen(paths[m], "rb");
      CHECK(f && !fseek(f, 0, SEEK_END));
      long bytes = ftell(f);
      rewind(f);
      uint32_t *spirv = malloc(bytes);
      CHECK(spirv && fread(spirv, 1, bytes, f) == bytes && !fclose(f));
      CHECK(v->CreateShaderModule(dev, &(VkShaderModuleCreateInfo){
         .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = bytes, .pCode = spirv},
         NULL, &modules[m]) == VK_SUCCESS);
      free(spirv);
   }
   VkDescriptorSetLayout set_layout;
   CHECK(v->CreateDescriptorSetLayout(dev, &(VkDescriptorSetLayoutCreateInfo){
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 1,
      .pBindings = &(VkDescriptorSetLayoutBinding){0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                                   VK_SHADER_STAGE_FRAGMENT_BIT}},
      NULL, &set_layout) == VK_SUCCESS);
   VkPipelineLayout layout;
   CHECK(v->CreatePipelineLayout(dev, &(VkPipelineLayoutCreateInfo){
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
      .pSetLayouts = &set_layout}, NULL, &layout) == VK_SUCCESS);
   const VkFormat color = VK_FORMAT_R8G8B8A8_UNORM, depth = VK_FORMAT_D32_SFLOAT;
   VkPipeline pipeline;
   CHECK(v->CreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &(VkGraphicsPipelineCreateInfo){
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
      .pNext = &(VkPipelineRenderingCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, .colorAttachmentCount = 1,
         .pColorAttachmentFormats = &color, .depthAttachmentFormat = game ? depth : VK_FORMAT_UNDEFINED},
      .stageCount = 2,
      .pStages = (VkPipelineShaderStageCreateInfo[]){
         {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = modules[0], .pName = "main"},
         {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
          .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = modules[1], .pName = "main"}},
      .pVertexInputState = &(VkPipelineVertexInputStateCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
         .vertexBindingDescriptionCount = 1,
         .pVertexBindingDescriptions = &(VkVertexInputBindingDescription){0, 32, VK_VERTEX_INPUT_RATE_VERTEX},
         .vertexAttributeDescriptionCount = 2,
         .pVertexAttributeDescriptions = (VkVertexInputAttributeDescription[]){
            {0, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 0}, {1, 0, VK_FORMAT_R32G32_SFLOAT, 16}}},
      .pInputAssemblyState = &(VkPipelineInputAssemblyStateCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
         .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST},
      .pViewportState = &(VkPipelineViewportStateCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1,
         .pViewports = &(VkViewport){0, 0, 1920, 1080, 0, 1}, .scissorCount = 1,
         .pScissors = &(VkRect2D){{0, 0}, {1920, 1080}}},
      .pRasterizationState = &(VkPipelineRasterizationStateCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
         .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE, .lineWidth = 1},
      .pMultisampleState = &(VkPipelineMultisampleStateCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
         .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT},
      .pDepthStencilState = &(VkPipelineDepthStencilStateCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
         .depthTestEnable = game, .depthWriteEnable = game, .depthCompareOp = VK_COMPARE_OP_LESS},
      .pColorBlendState = &(VkPipelineColorBlendStateCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, .attachmentCount = 1,
         .pAttachments = &(VkPipelineColorBlendAttachmentState){
            .blendEnable = !game, .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
            .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, .colorBlendOp = VK_BLEND_OP_ADD,
            .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE, .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
            .alphaBlendOp = VK_BLEND_OP_ADD, .colorWriteMask = 0xf}},
      .layout = layout}, NULL, &pipeline) == VK_SUCCESS);
   v->DestroyPipeline(dev, pipeline, NULL);
   v->DestroyPipelineLayout(dev, layout, NULL);
   v->DestroyDescriptorSetLayout(dev, set_layout, NULL);
   for (unsigned m = 0; m < 2; m++)
      v->DestroyShaderModule(dev, modules[m], NULL);
   apex_device_finish(&device);
}

int main(int argc, char **argv)
{
   bool graphics = argc == 6 && !strcmp(argv[1], "--graphics");
   CHECK(graphics || argc == 13 || argc == 14);
   const char *output = argc == 14 ? argv[13] : NULL;
   FILE *f = fopen(graphics ? argv[2] : argv[1], "rb");
   CHECK(f && fseek(f, 0, SEEK_END) == 0);
   long size = ftell(f);
   CHECK(size > 0 && size % 4 == 0);
   rewind(f);
   uint32_t *spirv = malloc(size);
   CHECK(spirv && fread(spirv, 1, size, f) == size);
   CHECK(fclose(f) == 0);

   /* Internal fixture only: no ICD or physical-device advertisement. */
   struct vk_instance instance;
   const struct vk_instance_extension_table extensions = {0};
   const struct vk_instance_dispatch_table instance_dispatch = {0};
   const VkInstanceCreateInfo instance_info = {
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
   };
   CHECK(vk_instance_init(&instance, &extensions, &instance_dispatch,
                          &instance_info, vk_default_allocator()) == VK_SUCCESS);
   (void)vk_instance_to_handle(&instance);
   struct vk_physical_device physical;
   const struct vk_physical_device_dispatch_table physical_dispatch = {0};
   const struct vk_properties properties = {
      .subgroupSize = 16, .minSubgroupSize = 16, .maxSubgroupSize = 16,
      .maxComputeWorkGroupCount = {1024, 1, 1}, .maxComputeWorkGroupSize = {16, 1, 1},
   };
   CHECK(vk_physical_device_init(&physical, &instance, NULL, NULL,
                                 &properties, &physical_dispatch) == VK_SUCCESS);
   (void)vk_physical_device_to_handle(&physical);
   struct apex_device device = {0};
   const struct vk_device_dispatch_table dispatch = {
      .CreateComputePipelines = apex_CreateComputePipelines,
   };
   const VkDeviceCreateInfo device_info = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
   CHECK(vk_device_init(&device.vk, &physical, &dispatch, &device_info, NULL) == VK_SUCCESS);
   VkDevice dev = apex_device_to_handle(&device);
   VkShaderModule module;
   const VkShaderModuleCreateInfo module_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = size, .pCode = spirv,
   };
   CHECK(device.vk.dispatch_table.CreateShaderModule(dev, &module_info, NULL, &module) == VK_SUCCESS);
   free(spirv);

   VkComputePipelineCreateInfo infos[3] = {{
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {
         .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_COMPUTE_BIT,
         .module = module, .pName = "main",
      },
   }};
   /* Map order differs from data order; unused padding must not become bias. */
   uint32_t values[] = {0xbad, 7, 101};
   VkSpecializationMapEntry entries[] = {{7, 8, 4}, {29, 4, 4}};
   VkSpecializationInfo specialization = {
      .mapEntryCount = 2, .pMapEntries = entries,
      .dataSize = sizeof(values), .pData = values,
   };
   infos[1] = infos[2] = infos[0];
   infos[1].stage.pSpecializationInfo = &specialization;
   infos[2].stage.pSpecializationInfo = &specialization;
   infos[2].stage.pName = "alternate";
   VkPipeline pipelines[3];
   CHECK(device.vk.dispatch_table.CreateComputePipelines(dev, VK_NULL_HANDLE, 3,
                                                       infos, NULL, pipelines) == VK_SUCCESS);
   for (unsigned i = 0; i < 3; i++) {
      struct apex_pipeline *p = apex_pipeline_from_handle(pipelines[i]);
      CHECK(p && p->program.code.size > 40 && p->program.code.data);
      CHECK(p->vk.bind_point == VK_PIPELINE_BIND_POINT_COMPUTE);
      for (unsigned j = 0; j < i; j++) {
         struct apex_pipeline *q = apex_pipeline_from_handle(pipelines[j]);
         CHECK(p->program.code.data != q->program.code.data);
         CHECK(p->program.code.size != q->program.code.size ||
               memcmp(p->program.code.data, q->program.code.data, p->program.code.size) != 0);
      }
   }

   /* A failed middle entry must not discard successes or stop later entries. */
   VkComputePipelineCreateInfo batch[3] = {infos[0], infos[0], infos[0]};
   batch[1].stage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
   VkPipeline mixed[3];
   CHECK(apex_CreateComputePipelines(dev, VK_NULL_HANDLE, 3, batch, NULL, mixed) ==
         VK_ERROR_FEATURE_NOT_PRESENT);
   CHECK(mixed[0] && !mixed[1] && mixed[2]);
   for (unsigned i = 0; i < 3; i++)
      device.vk.dispatch_table.DestroyPipeline(dev, mixed[i], NULL);
   batch[1].flags = VK_PIPELINE_CREATE_EARLY_RETURN_ON_FAILURE_BIT;
   CHECK(apex_CreateComputePipelines(dev, VK_NULL_HANDLE, 3, batch, NULL, mixed) ==
         VK_ERROR_FEATURE_NOT_PRESENT);
   CHECK(mixed[0] && !mixed[1] && !mixed[2]);
   device.vk.dispatch_table.DestroyPipeline(dev, mixed[0], NULL);

   VkPipelineCreateFlags2CreateInfo flags2 = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO,
      .flags = VK_PIPELINE_CREATE_2_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT,
   };
   batch[0] = infos[0];
   batch[0].pNext = &flags2;
   CHECK(apex_CreateComputePipelines(dev, VK_NULL_HANDLE, 1, batch, NULL, mixed) ==
         VK_PIPELINE_COMPILE_REQUIRED);
   CHECK(!mixed[0]);
   batch[0] = infos[0];
   batch[1].flags = 0;
   batch[2].pNext = &flags2;
   CHECK(apex_CreateComputePipelines(dev, VK_NULL_HANDLE, 3, batch, NULL, mixed) ==
         VK_ERROR_FEATURE_NOT_PRESENT);
   CHECK(mixed[0] && !mixed[1] && !mixed[2]);
   device.vk.dispatch_table.DestroyPipeline(dev, mixed[0], NULL);
   VkAllocationCallbacks allocation = *vk_default_allocator();
   allocation.pfnAllocation = fail_alloc;
   CHECK(apex_CreateComputePipelines(dev, VK_NULL_HANDLE, 1, infos, &allocation, mixed) ==
         VK_ERROR_OUT_OF_HOST_MEMORY);
   CHECK(!mixed[0]);
   batch[0] = infos[0];
   batch[0].stage.pName = "missing";
   CHECK(apex_CreateComputePipelines(dev, VK_NULL_HANDLE, 1, batch, NULL, mixed) != VK_SUCCESS);
   CHECK(!mixed[0]);
   batch[0].stage.pName = "wide";
   CHECK(apex_CreateComputePipelines(dev, VK_NULL_HANDLE, 1, batch, NULL, mixed) == VK_SUCCESS);
   CHECK(mixed[0]);
   struct apex_pipeline *wide = apex_pipeline_from_handle(mixed[0]);
   /* Header: magic, compute stage, local size 32x1x1. */
   CHECK(wide->program.code.size >= 64 && word(wide->program.code.data) == 0x50585041 &&
         word(wide->program.code.data + 4) == 0 &&
         word(wide->program.code.data + 16) == (32 | 1 << 9 | 1 << 18));
   if (output) {
      char path[4096];
      CHECK(snprintf(path, sizeof(path), "%s/mesa-wide.apx", output) < sizeof(path));
      FILE *wide_file = fopen(path, "wb");
      CHECK(wide_file && fwrite(wide->program.code.data, 1, wide->program.code.size, wide_file) ==
            wide->program.code.size && !fclose(wide_file));
   }
   device.vk.dispatch_table.DestroyPipeline(dev, mixed[0], NULL);
   VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO,
      .requiredSubgroupSize = 8,
   };
   batch[0] = infos[0];
   batch[0].stage.pNext = &subgroup;
   CHECK(apex_CreateComputePipelines(dev, VK_NULL_HANDLE, 1, batch, NULL, mixed) ==
         VK_ERROR_FEATURE_NOT_PRESENT);
   CHECK(!mixed[0]);
   subgroup.requiredSubgroupSize = 16;
   CHECK(apex_CreateComputePipelines(dev, VK_NULL_HANDLE, 1, batch, NULL, mixed) == VK_SUCCESS);
   device.vk.dispatch_table.DestroyPipeline(dev, mixed[0], NULL);

   /* Compiled pipelines survive both the module and specialization storage. */
   device.vk.dispatch_table.DestroyShaderModule(dev, module, NULL);
   memset(values, 0, sizeof(values));
   if (output) {
      write_fixture(output, "mesa-default", pipelines[0], 37, 3, 16);
      write_fixture(output, "mesa-specialized", pipelines[1], 101, 7, 16);
      write_fixture(output, "mesa-entrypoint", pipelines[2], 112, 7, 12);
   }
   for (unsigned i = 0; i < 3; i++)
      device.vk.dispatch_table.DestroyPipeline(dev, pipelines[i], NULL);
   vk_device_finish(&device.vk);
   test_descriptors(&physical, argv[2], "mesa-descriptors", output, false, false);
   test_descriptors(&physical, argv[3], "mesa-atomics", output, false, false);
   test_descriptors(&physical, argv[4], "mesa-reindex", output, false, true);
   test_descriptors(&physical, argv[2], "mesa-descriptors", NULL, true, false);
   test_descriptors(&physical, argv[3], "mesa-atomics", NULL, true, false);
   test_descriptors(&physical, argv[4], "mesa-reindex", NULL, true, true);
   test_descriptors(&physical, argv[7], "mesa-scalar", output, false, false);
   test_descriptors(&physical, argv[7], "mesa-scalar", NULL, true, false);
   test_descriptors(&physical, argv[8], "mesa-images", output, false, false);
   test_rgba(&physical, argv[9]);
   test_dispatch(&physical, argv[5], output, false, false);
   test_dispatch(&physical, argv[5], output, true, false);
   test_dispatch(&physical, argv[6], output, false, true);
   test_fill(&physical, output);
   test_bench(&physical, argv[10]);
   test_graphics(&physical, argv[11], argv[12], false);
   test_graphics(&physical, argv[11], argv[12], true);
   vk_physical_device_finish(&physical);
   vk_instance_finish(&instance);
   puts("PASS Apex Mesa compute pipelines: specialization, entrypoints, lifetime, failures");
   return 0;
}
