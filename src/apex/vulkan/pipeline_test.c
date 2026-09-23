/* SPDX-License-Identifier: MIT */
#include "apex_pipeline.h"
#include "vk_alloc.h"
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

static void
write_fixture(const char *directory, const char *name, VkPipeline handle,
              uint32_t bias, uint32_t scale)
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
      expected[i] = i < 16 ? util_cpu_to_le32(bias + i * scale) : input[i];
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

int main(int argc, char **argv)
{
   CHECK(argc == 2 || argc == 3);
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
   };
   CHECK(vk_physical_device_init(&physical, &instance, NULL, NULL,
                                 &properties, &physical_dispatch) == VK_SUCCESS);
   (void)vk_physical_device_to_handle(&physical);
   struct vk_device device;
   const struct vk_device_dispatch_table dispatch = {
      .CreateComputePipelines = apex_CreateComputePipelines,
   };
   const VkDeviceCreateInfo device_info = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
   CHECK(vk_device_init(&device, &physical, &dispatch, &device_info, NULL) == VK_SUCCESS);
   VkDevice dev = vk_device_to_handle(&device);
   VkShaderModule module;
   const VkShaderModuleCreateInfo module_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = size, .pCode = spirv,
   };
   CHECK(device.dispatch_table.CreateShaderModule(dev, &module_info, NULL, &module) == VK_SUCCESS);
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
   CHECK(device.dispatch_table.CreateComputePipelines(dev, VK_NULL_HANDLE, 3,
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
      device.dispatch_table.DestroyPipeline(dev, mixed[i], NULL);
   batch[1].flags = VK_PIPELINE_CREATE_EARLY_RETURN_ON_FAILURE_BIT;
   CHECK(apex_CreateComputePipelines(dev, VK_NULL_HANDLE, 3, batch, NULL, mixed) ==
         VK_ERROR_FEATURE_NOT_PRESENT);
   CHECK(mixed[0] && !mixed[1] && !mixed[2]);
   device.dispatch_table.DestroyPipeline(dev, mixed[0], NULL);

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
   device.dispatch_table.DestroyPipeline(dev, mixed[0], NULL);
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
   CHECK(apex_CreateComputePipelines(dev, VK_NULL_HANDLE, 1, batch, NULL, mixed) ==
         VK_ERROR_FEATURE_NOT_PRESENT);
   CHECK(!mixed[0]);
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
   device.dispatch_table.DestroyPipeline(dev, mixed[0], NULL);

   /* Compiled pipelines survive both the module and specialization storage. */
   device.dispatch_table.DestroyShaderModule(dev, module, NULL);
   memset(values, 0, sizeof(values));
   if (argc == 3) {
      write_fixture(argv[2], "mesa-default", pipelines[0], 37, 3);
      write_fixture(argv[2], "mesa-specialized", pipelines[1], 101, 7);
      write_fixture(argv[2], "mesa-entrypoint", pipelines[2], 112, 7);
   }
   for (unsigned i = 0; i < 3; i++)
      device.dispatch_table.DestroyPipeline(dev, pipelines[i], NULL);
   vk_device_finish(&device);
   vk_physical_device_finish(&physical);
   vk_instance_finish(&instance);
   puts("PASS Apex Mesa compute pipelines: specialization, entrypoints, lifetime, failures");
   return 0;
}
