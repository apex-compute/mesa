/* SPDX-License-Identifier: MIT */
#include "apex_pipeline.h"
#include "vk_alloc.h"
#include "vk_buffer.h"
#include "vk_common_entrypoints.h"
#include "vk_device.h"
#include "vk_instance.h"
#include "vk_physical_device.h"
#include "util/u_math.h"
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
   CHECK(fwrite(p->code.data, 1, p->code.size, f) == p->code.size);
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
   CHECK(apex_device_init(&device, physical, &device_info, NULL, -1,
                          APEX_TRANSPORT_NATIVE) == VK_SUCCESS);
   /* Compiler/layout test only: choose the DRM descriptor ABI without a fd.
    * No memory allocation, submission or ioctl occurs in this test. */
   device.transport = APEX_TRANSPORT_DRM;
   VkDevice dev = apex_device_to_handle(&device);
   const struct vk_device_dispatch_table *v = &device.vk.dispatch_table;
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
          .descriptorType = storage, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
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
      {storage, 7}, {uniform, 2},
      /* Static descriptors cannot satisfy a dynamic descriptor budget. */
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, dynamic ? 100 : 0},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, dynamic ? 100 : 0},
   };
   const VkDescriptorPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
      .maxSets = 4, .poolSizeCount = pointers ? 1 : dynamic ? 4 : 2, .pPoolSizes = sizes,
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
   CHECK(p->descriptor_count == 5 && p->set_offsets[0] == 0 && p->set_offsets[1] == 1);
   CHECK(p->push_size == 64);
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
      CHECK(f && fwrite(p->code.data, 1, p->code.size, f) == p->code.size && !fclose(f));
   }
   v->DestroyPipeline(dev, pipeline, NULL);
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
   CHECK(apex_device_init(&device, physical, &di, NULL, -1, APEX_TRANSPORT_NATIVE) == VK_SUCCESS);
   device.transport = APEX_TRANSPORT_DRM;
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
   CHECK(p->descriptor_count == (multiple ? 2 : 1) && p->push_size == (multiple ? 0 : 20));
   uint64_t padded_private = (uint64_t)word(p->code.data + 28) * 16;
   CHECK(p->max_workgroups && p->max_workgroups <= 1024);
   CHECK(padded_private * p->max_workgroups <= 2097152);
   CHECK(p->max_workgroups == 1024 || padded_private * (p->max_workgroups + 1) > 2097152);
   if (output) {
      char filename[4096];
      const char *name = multiple ? "mesa-multiple" : grid ? "mesa-dispatch-grid" : "mesa-dispatch";
      CHECK(snprintf(filename, sizeof(filename), "%s/%s.apx", output, name) < sizeof(filename));
      f = fopen(filename, "wb");
      CHECK(f && fwrite(p->code.data, 1, p->code.size, f) == p->code.size && !fclose(f));
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
   CHECK(apex_device_init(&device, physical, &di, NULL, -1, APEX_TRANSPORT_NATIVE) == VK_SUCCESS);
   /* Compile/record only. Supply an address without creating a kernel BO. */
   device.transport = APEX_TRANSPORT_DRM;
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
   v->CmdFillBuffer(cmd, buffer, 4, 196, 0xa5c31e79);
   v->CmdFillBuffer(cmd, buffer, 256, VK_WHOLE_SIZE, 0xbaadf00d);
   enum vk_meta_object_key_type key = VK_META_OBJECT_KEY_FILL_BUFFER;
   VkPipeline handle = vk_meta_lookup_pipeline(&device.meta, &key, sizeof(key));
   CHECK(handle);
   struct apex_pipeline *pipeline = apex_pipeline_from_handle(handle);
   CHECK(!pipeline->descriptor_count && pipeline->push_size == 16);
   if (output) {
      char filename[4096];
      CHECK(snprintf(filename, sizeof(filename), "%s/mesa-fill.apx", output) < sizeof(filename));
      FILE *f = fopen(filename, "wb");
      CHECK(f && fwrite(pipeline->code.data, 1, pipeline->code.size, f) == pipeline->code.size && !fclose(f));
   }
   for (unsigned chunk = 1; chunk <= 16; chunk *= 2) {
      VkBufferCopy region = {.srcOffset = chunk, .dstOffset = 128 + chunk, .size = chunk * 3};
      v->CmdCopyBuffer(cmd, buffer, buffer, 1, &region);
      struct { enum vk_meta_object_key_type type; uint32_t chunk; } copy_key = {
         VK_META_OBJECT_KEY_COPY_BUFFER, chunk,
      };
      handle = vk_meta_lookup_pipeline(&device.meta, &copy_key, sizeof(copy_key));
      CHECK(handle);
      pipeline = apex_pipeline_from_handle(handle);
      CHECK(!pipeline->descriptor_count && pipeline->push_size == 24);
      if (output) {
         char filename[4096];
         CHECK(snprintf(filename, sizeof(filename), "%s/mesa-copy%u.apx", output, chunk) < sizeof(filename));
         FILE *f = fopen(filename, "wb");
         CHECK(f && fwrite(pipeline->code.data, 1, pipeline->code.size, f) == pipeline->code.size && !fclose(f));
      }
   }
   CHECK(v->EndCommandBuffer(cmd) == VK_SUCCESS);
   v->DestroyCommandPool(dev, pool, NULL);
   v->DestroyBuffer(dev, buffer, NULL);
   apex_device_finish(&device);
}

int main(int argc, char **argv)
{
   CHECK(argc == 8 || argc == 9);
   const char *output = argc == 9 ? argv[8] : NULL;
   FILE *f = fopen(argv[1], "rb");
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
   struct apex_device device = {.transport = APEX_TRANSPORT_NATIVE};
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
      CHECK(p && p->code.size > 40 && p->code.data);
      CHECK(p->vk.bind_point == VK_PIPELINE_BIND_POINT_COMPUTE);
      for (unsigned j = 0; j < i; j++) {
         struct apex_pipeline *q = apex_pipeline_from_handle(pipelines[j]);
         CHECK(p->code.data != q->code.data);
         CHECK(p->code.size != q->code.size ||
               memcmp(p->code.data, q->code.data, p->code.size) != 0);
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
   CHECK(wide->code.size >= 48 && word(wide->code.data) == 0x32585041 &&
         word(wide->code.data + 4) == 2 && word(wide->code.data + 40) == 32 &&
         !word(wide->code.data + 44));
   if (output) {
      char path[4096];
      CHECK(snprintf(path, sizeof(path), "%s/mesa-wide.apx", output) < sizeof(path));
      FILE *wide_file = fopen(path, "wb");
      CHECK(wide_file && fwrite(wide->code.data, 1, wide->code.size, wide_file) ==
            wide->code.size && !fclose(wide_file));
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
   test_dispatch(&physical, argv[5], output, false, false);
   test_dispatch(&physical, argv[5], output, true, false);
   test_dispatch(&physical, argv[6], output, false, true);
   test_fill(&physical, output);
   vk_physical_device_finish(&physical);
   vk_instance_finish(&instance);
   puts("PASS Apex Mesa compute pipelines: specialization, entrypoints, lifetime, failures");
   return 0;
}
