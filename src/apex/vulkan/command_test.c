/* SPDX-License-Identifier: MIT */
#include "apex_device.h"
#include "apex_pipeline.h"
#include "apex_native_uapi.h"
#include "vk_alloc.h"
#include "vk_instance.h"
#include "vk_physical_device.h"
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { \
   fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); abort(); \
} } while (0)

/* Link-time interception tests the transport, not shader execution. A real
 * fd bypasses it and the same Vulkan commands/results run on the device. */
static struct {
   int fault;
   unsigned calls[11], step, dispatch;
   struct apex_pipeline *pipelines[3];
   uint32_t *mapped;
   uint32_t expected[2048], payload[800];
} mock;
static const unsigned starts[] = {48, 1088, 48};
static const unsigned biases[] = {37, 101, 112};
static const unsigned scales[] = {3, 7, 7};

int __real_ioctl(int fd, unsigned long request, ...);
int __wrap_ioctl(int fd, unsigned long request, ...);

int
__wrap_ioctl(int fd, unsigned long request, ...)
{
   va_list args;
   va_start(args, request);
   struct apex_ioctl_native *r = va_arg(args, void *);
   va_end(args);
   if (fd >= 0)
      return __real_ioctl(fd, request, r);
   CHECK(request == APEX_IOCTL_NATIVE && r->operation <= 10);
   mock.calls[r->operation]++;
   if ((int)r->operation == mock.fault) {
      errno = EIO;
      return -1;
   }
   /* Failures must close the session without attempting a readback. */
   if (mock.fault >= 0 && r->operation == APEX_NATIVE_CLOSE)
      return 0;
   const unsigned operations[] = {0, 1, 1, 2, 4, 5, 6, 6, 8, 3, 10};
   CHECK(mock.step < ARRAY_SIZE(operations) && r->operation == operations[mock.step]);
   CHECK(!r->queue && !r->dependency && !r->signal && !r->offset);
   unsigned d = mock.dispatch;
   switch (mock.step++) {
   case 0:
      CHECK(!r->bytes && !r->user_ptr);
      break;
   case 1:
      CHECK(r->kind == APEX_NATIVE_PROGRAM && r->bytes == mock.pipelines[d]->code.size);
      CHECK(!memcmp((void *)(uintptr_t)r->user_ptr, mock.pipelines[d]->code.data, r->bytes));
      r->handle = 13;
      break;
   case 2:
      CHECK(r->kind == APEX_NATIVE_DATA && r->bytes == sizeof(mock.payload));
      r->handle = 27;
      break;
   case 3:
      CHECK(r->handle == 27 && r->bytes == sizeof(mock.payload));
      CHECK((void *)(uintptr_t)r->user_ptr == mock.mapped + starts[d]);
      CHECK(!memcmp((void *)(uintptr_t)r->user_ptr, mock.expected + starts[d], r->bytes));
      memcpy(mock.payload, (void *)(uintptr_t)r->user_ptr, r->bytes);
      break;
   case 5:
      CHECK(r->handle == 13 && r->data_handle == 27 && r->kind == APEX_NATIVE_COMPUTE);
      CHECK(!r->workgroups);
      r->identity = 0x123456789ull;
      break;
   case 6:
      CHECK(r->identity == 0x123456789ull);
      r->status = 0;
      break;
   case 7:
      CHECK(r->identity == 0x123456789ull);
      r->status = mock.fault == 99 ? 2 : 1;
      break;
   case 9:
      CHECK(r->handle == 27 && r->bytes == sizeof(mock.payload));
      CHECK((void *)(uintptr_t)r->user_ptr == mock.mapped + starts[d]);
      for (unsigned i = 0; i < 16; i++)
         mock.payload[i] = biases[d] + scales[d] * i;
      memcpy((void *)(uintptr_t)r->user_ptr, mock.payload, r->bytes);
      memcpy(mock.expected + starts[d], mock.payload, r->bytes);
      break;
   case 10:
      mock.step = 0;
      mock.dispatch++;
      break;
   }
   return 0;
}

static void
run(struct vk_physical_device *physical, const uint32_t *spirv, size_t size, int fd, int fault)
{
   memset(&mock, 0, sizeof(mock));
   mock.fault = fault;
   struct apex_device device;
   const float priority = 1.0f;
   const VkDeviceQueueCreateInfo queue_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueCount = 1, .pQueuePriorities = &priority,
   };
   const VkPhysicalDeviceDescriptorIndexingFeatures features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES,
      .descriptorBindingStorageBufferUpdateAfterBind = VK_TRUE,
   };
   const VkDeviceCreateInfo device_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &features,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &queue_info,
   };
   CHECK(apex_device_init(&device, physical, &device_info, NULL, fd) == VK_SUCCESS);
   VkDevice dev = apex_device_to_handle(&device);
   const struct vk_device_dispatch_table *v = &device.vk.dispatch_table;
   VkQueue queue;
   v->GetDeviceQueue(dev, 0, 0, &queue);
   CHECK(queue == vk_queue_to_handle(&device.queue));
   VkShaderModule module;
   const VkShaderModuleCreateInfo module_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = size, .pCode = spirv,
   };
   CHECK(v->CreateShaderModule(dev, &module_info, NULL, &module) == VK_SUCCESS);
   const VkDescriptorSetLayoutBinding binding = {
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
   };
   VkDescriptorSetLayout set_layout;
   const VkDescriptorBindingFlags binding_flags = VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT;
   const VkDescriptorSetLayoutBindingFlagsCreateInfo binding_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
      .bindingCount = 1, .pBindingFlags = &binding_flags,
   };
   const VkDescriptorSetLayoutCreateInfo set_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .pNext = &binding_info, .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
      .bindingCount = 1, .pBindings = &binding,
   };
   CHECK(v->CreateDescriptorSetLayout(dev, &set_info, NULL, &set_layout) == VK_SUCCESS);
   VkPipelineLayout layout;
   const VkPipelineLayoutCreateInfo layout_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &set_layout,
   };
   CHECK(v->CreatePipelineLayout(dev, &layout_info, NULL, &layout) == VK_SUCCESS);
   const uint32_t values[] = {7, 101};
   const VkSpecializationMapEntry entries[] = {{7, 4, 4}, {29, 0, 4}};
   const VkSpecializationInfo spec = {
      .mapEntryCount = 2, .pMapEntries = entries, .dataSize = sizeof(values), .pData = values,
   };
   VkComputePipelineCreateInfo pipeline_info = {
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, .layout = layout,
      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = module, .pName = "main"},
   };
   VkPipeline pipelines[3];
   for (unsigned i = 0; i < 3; i++) {
      if (i) pipeline_info.stage.pSpecializationInfo = &spec;
      if (i == 2) pipeline_info.stage.pName = "alternate";
      CHECK(v->CreateComputePipelines(dev, VK_NULL_HANDLE, 1, &pipeline_info, NULL, &pipelines[i]) == VK_SUCCESS);
      mock.pipelines[i] = apex_pipeline_from_handle(pipelines[i]);
   }
   v->DestroyShaderModule(dev, module, NULL);
   const VkMemoryAllocateInfo mem_info = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = sizeof(mock.expected),
   };
   VkDeviceMemory memory;
   CHECK(v->AllocateMemory(dev, &mem_info, NULL, &memory) == VK_SUCCESS);
   CHECK(v->MapMemory(dev, memory, 0, VK_WHOLE_SIZE, 0, (void **)&mock.mapped) == VK_SUCCESS);
   for (unsigned i = 0; i < ARRAY_SIZE(mock.expected); i++)
      mock.mapped[i] = mock.expected[i] = 0xd00d0000 + i;
   VkBuffer buffers[2];
   const VkBufferCreateInfo buffer_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 3456,
      .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
   };
   for (unsigned i = 0; i < 2; i++) {
      CHECK(v->CreateBuffer(dev, &buffer_info, NULL, &buffers[i]) == VK_SUCCESS);
      VkMemoryRequirements req;
      v->GetBufferMemoryRequirements(dev, buffers[i], &req);
      CHECK(req.size == 3456 && req.alignment == 64 && req.memoryTypeBits == 1);
      CHECK(v->BindBufferMemory(dev, buffers[i], memory, i ? 4096 : 64) == VK_SUCCESS);
   }
   const VkDescriptorPoolSize pool_size = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
   const VkDescriptorPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT | VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
      .maxSets = 3, .poolSizeCount = 1, .pPoolSizes = &pool_size,
   };
   VkDescriptorPool pool;
   CHECK(v->CreateDescriptorPool(dev, &pool_info, NULL, &pool) == VK_SUCCESS);
   VkDescriptorSetLayout layouts[2] = {set_layout, set_layout};
   VkDescriptorSetAllocateInfo set_alloc = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = pool, .descriptorSetCount = 2, .pSetLayouts = layouts,
   };
   VkDescriptorSet sets[2], extra;
   CHECK(v->AllocateDescriptorSets(dev, &set_alloc, sets) == VK_SUCCESS);
   set_alloc.descriptorSetCount = 1;
   CHECK(v->AllocateDescriptorSets(dev, &set_alloc, &extra) == VK_ERROR_OUT_OF_POOL_MEMORY && !extra);
   VkDescriptorBufferInfo bindings[] = {
      {buffers[0], 128, 3200}, {buffers[1], 256, VK_WHOLE_SIZE},
   };
   if (fault == 100) bindings[0].range = 3329;
   if (fault == 101) bindings[0].offset = 3456;
   VkWriteDescriptorSet writes[2] = {{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = sets[0],
      .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &bindings[1], /* Changed after recording, before submission. */
   }};
   writes[1] = writes[0];
   writes[1].dstSet = sets[1];
   v->UpdateDescriptorSets(dev, 2, writes, 0, NULL);
   VkCommandPool command_pool;
   const VkCommandPoolCreateInfo command_pool_info = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
   };
   CHECK(v->CreateCommandPool(dev, &command_pool_info, NULL, &command_pool) == VK_SUCCESS);
   VkCommandBuffer cmd;
   const VkCommandBufferAllocateInfo command_alloc = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = command_pool, .commandBufferCount = 1,
   };
   CHECK(v->AllocateCommandBuffers(dev, &command_alloc, &cmd) == VK_SUCCESS);
   const VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
   CHECK(v->BeginCommandBuffer(cmd, &begin) == VK_SUCCESS);
   for (unsigned i = 0; i < 2; i++) {
      v->CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines[i]);
      v->CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &sets[i], 0, NULL);
      v->CmdDispatch(cmd, 1, 1, 1);
      const VkMemoryBarrier2 barrier = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
         .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
         .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
         .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
         .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
      };
      const VkDependencyInfo dependency = {
         .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
         .memoryBarrierCount = 1, .pMemoryBarriers = &barrier,
      };
      v->CmdPipelineBarrier2(cmd, &dependency);
   }
   v->CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines[2]);
   v->CmdDispatch(cmd, 0, 1, 1);
   CHECK(v->EndCommandBuffer(cmd) == VK_SUCCESS);
   writes[0].pBufferInfo = &bindings[0];
   v->UpdateDescriptorSets(dev, 1, writes, 0, NULL);
   const VkCommandBufferSubmitInfo cb_submit = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = cmd,
   };
   const VkSubmitInfo2 submit = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
      .commandBufferInfoCount = 1, .pCommandBufferInfos = &cb_submit,
   };
   VkResult result = v->QueueSubmit2(queue, 1, &submit, VK_NULL_HANDLE);
   if (fault >= 0) {
      CHECK(result == VK_ERROR_DEVICE_LOST);
      CHECK(v->QueueWaitIdle(queue) == VK_ERROR_DEVICE_LOST);
      CHECK(mock.calls[APEX_NATIVE_CREATE] == (fault >= 100 ? 0 : 1));
      CHECK(mock.calls[APEX_NATIVE_CLOSE] == (fault == APEX_NATIVE_CREATE || fault >= 100 ? 0 : 1));
      if (fault != APEX_NATIVE_DOWNLOAD && fault != APEX_NATIVE_CLOSE)
         CHECK(!mock.calls[APEX_NATIVE_DOWNLOAD]);
      unsigned calls = mock.calls[APEX_NATIVE_CREATE];
      CHECK(v->QueueSubmit2(queue, 1, &submit, VK_NULL_HANDLE) == VK_ERROR_DEVICE_LOST);
      CHECK(mock.calls[APEX_NATIVE_CREATE] == calls);
   } else {
      CHECK(result == VK_SUCCESS && v->QueueWaitIdle(queue) == VK_SUCCESS);
      for (unsigned i = 0; i < ARRAY_SIZE(mock.expected); i++) {
         uint32_t expected = 0xd00d0000 + i;
         if (i >= 48 && i < 64) expected = 37 + 3 * (i - 48);
         if (i >= 1088 && i < 1104) expected = 101 + 7 * (i - 1088);
         CHECK(mock.mapped[i] == expected);
      }
      /* Reset must discard both old dispatches and bound pipeline/set state. */
      CHECK(v->ResetCommandBuffer(cmd, 0) == VK_SUCCESS);
      CHECK(v->BeginCommandBuffer(cmd, &begin) == VK_SUCCESS);
      v->CmdDispatch(cmd, 1, 1, 1);
      CHECK(v->EndCommandBuffer(cmd) == VK_ERROR_FEATURE_NOT_PRESENT);
      CHECK(v->ResetCommandBuffer(cmd, 0) == VK_SUCCESS);
      CHECK(v->BeginCommandBuffer(cmd, &begin) == VK_SUCCESS);
      v->CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines[2]);
      v->CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, sets, 0, NULL);
      v->CmdDispatch(cmd, 1, 1, 1);
      CHECK(v->EndCommandBuffer(cmd) == VK_SUCCESS);
      CHECK(v->QueueSubmit2(queue, 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS);
      CHECK(v->DeviceWaitIdle(dev) == VK_SUCCESS);
      for (unsigned i = 0; i < ARRAY_SIZE(mock.expected); i++) {
         uint32_t expected = 0xd00d0000 + i;
         if (i >= 48 && i < 64) expected = 112 + 7 * (i - 48);
         if (i >= 1088 && i < 1104) expected = 101 + 7 * (i - 1088);
         CHECK(mock.mapped[i] == expected);
      }
      if (fd < 0) CHECK(mock.dispatch == 3 && mock.step == 0);
   }
   v->DestroyCommandPool(dev, command_pool, NULL);
   CHECK(v->FreeDescriptorSets(dev, pool, 2, sets) == VK_SUCCESS);
   CHECK(v->AllocateDescriptorSets(dev, &set_alloc, &extra) == VK_SUCCESS);
   CHECK(v->ResetDescriptorPool(dev, pool, 0) == VK_SUCCESS);
   v->DestroyDescriptorPool(dev, pool, NULL);
   for (unsigned i = 0; i < 3; i++) v->DestroyPipeline(dev, pipelines[i], NULL);
   v->DestroyPipelineLayout(dev, layout, NULL);
   v->DestroyDescriptorSetLayout(dev, set_layout, NULL);
   for (unsigned i = 0; i < 2; i++) v->DestroyBuffer(dev, buffers[i], NULL);
   v->UnmapMemory(dev, memory);
   v->FreeMemory(dev, memory, NULL);
   apex_device_finish(&device);
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
   struct vk_instance instance;
   const struct vk_instance_extension_table extensions = {0};
   const struct vk_instance_dispatch_table instance_dispatch = {0};
   const VkInstanceCreateInfo instance_info = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
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
   int fd = argc == 3 ? open(argv[2], O_RDWR | O_CLOEXEC) : -1;
   CHECK(argc != 3 || fd >= 0);
   run(&physical, spirv, size, fd, -1);
   if (fd >= 0) {
      CHECK(close(fd) == 0);
      puts("PASS Apex Mesa native command submission: 3 dispatches, 2 buffers, 2048 words/guards");
   } else {
      const int faults[] = {0, 1, 2, 3, 4, 5, 6, 8, 10, 99, 100, 101};
      for (unsigned i = 0; i < ARRAY_SIZE(faults); i++)
         run(&physical, spirv, size, -1, faults[i]);
      puts("PASS Apex Mesa command transport: binding, offsets, reset, release, device loss (mock ioctl)");
   }
   free(spirv);
   vk_physical_device_finish(&physical);
   vk_instance_finish(&instance);
   return 0;
}
