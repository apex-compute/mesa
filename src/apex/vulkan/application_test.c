/* SPDX-License-Identifier: MIT */
/* Real loader/application gate. No Mesa private headers or mocked GPU results. */
#include <vulkan/vulkan.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { \
   fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); abort(); \
} } while (0)
#define INSTANCE(name) PFN_vk##name name = (PFN_vk##name)gipa(instance, "vk" #name); CHECK(name)
#define DEVICE(name) PFN_vk##name name = (PFN_vk##name)gdpa(device, "vk" #name); CHECK(name)

int main(int argc, char **argv)
{
   CHECK((argc == 3 || (argc == 4 && (!strcmp(argv[3], "--dispatch") || !strcmp(argv[3], "--fill")))) && geteuid() != 0);
   const int grid = argc == 4 && !strcmp(argv[3], "--dispatch");
   const int fill = argc == 4 && !strcmp(argv[3], "--fill");
   const unsigned word_count = grid ? 262144 : fill ? 32784 : 1024;
   const unsigned bind_words = fill ? 16 : 0;
   CHECK(!setenv("VK_DRIVER_FILES", argv[1], 1));
   CHECK(!setenv("APEX_DEVELOPMENT", "1", 1));
   void *loader = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
   CHECK(loader);
   PFN_vkGetInstanceProcAddr gipa = (PFN_vkGetInstanceProcAddr)dlsym(loader, "vkGetInstanceProcAddr");
   CHECK(gipa);
   VkInstance instance = VK_NULL_HANDLE;
   INSTANCE(CreateInstance);
   VkInstanceCreateInfo instance_info = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
   CHECK(CreateInstance(&instance_info, NULL, &instance) == VK_SUCCESS);
   INSTANCE(DestroyInstance);
   INSTANCE(EnumeratePhysicalDevices);
   uint32_t count = 0;
   VkPhysicalDevice physical;
   VkResult result = EnumeratePhysicalDevices(instance, &count, NULL);
   if ((result == VK_ERROR_INITIALIZATION_FAILED || result == VK_SUCCESS) && !count) {
      DestroyInstance(instance, NULL);
      dlclose(loader);
      puts("SKIP Apex application: no compatible render device (not GPU execution)");
      return 77;
   }
   CHECK(result == VK_SUCCESS && count == 1);
   CHECK(EnumeratePhysicalDevices(instance, &count, &physical) == VK_SUCCESS && count == 1);
   INSTANCE(CreateDevice);
   INSTANCE(GetDeviceProcAddr);
   PFN_vkGetDeviceProcAddr gdpa = GetDeviceProcAddr;
   const float priority = 1;
   VkDeviceQueueCreateInfo queue_info = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                        .queueCount = 1, .pQueuePriorities = &priority};
   VkDeviceCreateInfo device_info = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                                     .queueCreateInfoCount = 1, .pQueueCreateInfos = &queue_info};
   VkDevice device;
   CHECK(CreateDevice(physical, &device_info, NULL, &device) == VK_SUCCESS);
   DEVICE(DestroyDevice);
   DEVICE(GetDeviceQueue);
   DEVICE(CreateBuffer);
   DEVICE(DestroyBuffer);
   DEVICE(GetBufferMemoryRequirements);
   DEVICE(AllocateMemory);
   DEVICE(FreeMemory);
   DEVICE(BindBufferMemory);
   DEVICE(MapMemory);
   DEVICE(UnmapMemory);
   DEVICE(FlushMappedMemoryRanges);
   DEVICE(InvalidateMappedMemoryRanges);
   DEVICE(CreateDescriptorSetLayout);
   DEVICE(DestroyDescriptorSetLayout);
   DEVICE(CreatePipelineLayout);
   DEVICE(DestroyPipelineLayout);
   DEVICE(CreateDescriptorPool);
   DEVICE(DestroyDescriptorPool);
   DEVICE(AllocateDescriptorSets);
   DEVICE(UpdateDescriptorSets);
   DEVICE(CreateShaderModule);
   DEVICE(DestroyShaderModule);
   DEVICE(CreateComputePipelines);
   DEVICE(DestroyPipeline);
   DEVICE(CreateCommandPool);
   DEVICE(DestroyCommandPool);
   DEVICE(AllocateCommandBuffers);
   DEVICE(BeginCommandBuffer);
   DEVICE(EndCommandBuffer);
   DEVICE(CmdBindPipeline);
   DEVICE(CmdBindDescriptorSets);
   DEVICE(CmdDispatch);
   DEVICE(CmdFillBuffer);
   DEVICE(CmdPipelineBarrier);
   DEVICE(CreateFence);
   DEVICE(DestroyFence);
   DEVICE(ResetFences);
   DEVICE(WaitForFences);
   DEVICE(QueueSubmit);
   DEVICE(QueueWaitIdle);
   VkQueue queue;
   GetDeviceQueue(device, 0, 0, &queue);
   VkBufferCreateInfo buffer_info = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = (word_count - bind_words) * 4 - (fill ? 3 : 0),
      .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
         (fill ? VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT : 0)};
   VkBuffer buffer;
   CHECK(CreateBuffer(device, &buffer_info, NULL, &buffer) == VK_SUCCESS);
   VkMemoryRequirements requirements;
   GetBufferMemoryRequirements(device, buffer, &requirements);
   CHECK(requirements.memoryTypeBits & 1);
   VkMemoryAllocateInfo allocate = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                    .allocationSize = requirements.size + bind_words * 4};
   VkDeviceMemory memory;
   CHECK(AllocateMemory(device, &allocate, NULL, &memory) == VK_SUCCESS);
   CHECK(BindBufferMemory(device, buffer, memory, bind_words * 4) == VK_SUCCESS);
   uint32_t *words;
   CHECK(MapMemory(device, memory, 0, VK_WHOLE_SIZE, 0, (void **)&words) == VK_SUCCESS);
   for (unsigned i = 0; i < word_count; i++) words[i] = 0xca000000 + i * 37;
   VkMappedMemoryRange range = {.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
                                .memory = memory, .size = VK_WHOLE_SIZE};
   CHECK(FlushMappedMemoryRanges(device, 1, &range) == VK_SUCCESS);
   VkDescriptorSetLayoutBinding binding = {.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};
   VkDescriptorSetLayoutCreateInfo set_info = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                              .bindingCount = 1, .pBindings = &binding};
   VkDescriptorSetLayout set_layout;
   CHECK(CreateDescriptorSetLayout(device, &set_info, NULL, &set_layout) == VK_SUCCESS);
   VkPipelineLayoutCreateInfo layout_info = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                             .setLayoutCount = 1, .pSetLayouts = &set_layout};
   VkPipelineLayout layout;
   CHECK(CreatePipelineLayout(device, &layout_info, NULL, &layout) == VK_SUCCESS);
   VkDescriptorPoolSize pool_size = {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1};
   VkDescriptorPoolCreateInfo pool_info = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &pool_size};
   VkDescriptorPool pool;
   CHECK(CreateDescriptorPool(device, &pool_info, NULL, &pool) == VK_SUCCESS);
   VkDescriptorSetAllocateInfo set_allocate = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = pool, .descriptorSetCount = 1, .pSetLayouts = &set_layout};
   VkDescriptorSet set;
   CHECK(AllocateDescriptorSets(device, &set_allocate, &set) == VK_SUCCESS);
   VkDescriptorBufferInfo descriptor = {.buffer = buffer, .offset = 256,
      .range = grid ? (word_count - 64) * 4 : 128};
   VkWriteDescriptorSet write = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = set, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &descriptor};
   UpdateDescriptorSets(device, 1, &write, 0, NULL);
   FILE *file = fopen(argv[2], "rb");
   CHECK(file && !fseek(file, 0, SEEK_END));
   long bytes = ftell(file);
   CHECK(bytes > 0 && bytes % 4 == 0);
   rewind(file);
   uint32_t *spirv = malloc(bytes);
   CHECK(spirv && fread(spirv, 1, bytes, file) == (size_t)bytes && !fclose(file));
   VkShaderModuleCreateInfo shader_info = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                          .codeSize = bytes, .pCode = spirv};
   VkShaderModule shader;
   CHECK(CreateShaderModule(device, &shader_info, NULL, &shader) == VK_SUCCESS);
   free(spirv);
   VkSpecializationMapEntry entries[] = {{7, 0, 4}, {29, 4, 4}, {0, 8, 4}};
   const uint32_t constants[] = {101, 7, 1};
   VkSpecializationInfo specialization = {.mapEntryCount = grid ? 3 : 2, .pMapEntries = entries,
      .dataSize = sizeof(constants), .pData = constants};
   VkComputePipelineCreateInfo pipeline_info = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = shader, .pName = "main",
                .pSpecializationInfo = grid ? &specialization : NULL},
      .layout = layout};
   VkPipeline pipelines[2];
   CHECK(CreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, NULL, &pipelines[0]) == VK_SUCCESS);
   pipeline_info.stage.pName = grid ? "main" : "alternate";
   pipeline_info.stage.pSpecializationInfo = &specialization;
   CHECK(CreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, NULL, &pipelines[1]) == VK_SUCCESS);
   DestroyShaderModule(device, shader, NULL);
   VkCommandPoolCreateInfo commands_info = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
   VkCommandPool commands;
   CHECK(CreateCommandPool(device, &commands_info, NULL, &commands) == VK_SUCCESS);
   VkCommandBufferAllocateInfo command_allocate = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = commands, .commandBufferCount = 2};
   VkCommandBuffer buffers[2];
   CHECK(AllocateCommandBuffers(device, &command_allocate, buffers) == VK_SUCCESS);
   VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
   VkFenceCreateInfo fence_info = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
   VkFence fence;
   CHECK(CreateFence(device, &fence_info, NULL, &fence) == VK_SUCCESS);
   for (unsigned pass = 0; pass < 2; pass++) {
      const unsigned x = grid ? (pass ? 1025 : 2) : 1;
      const unsigned y = grid && !pass ? 3 : 1;
      const unsigned z = grid && !pass ? 4 : 1;
      CHECK(BeginCommandBuffer(buffers[pass], &begin) == VK_SUCCESS);
      CmdBindPipeline(buffers[pass], VK_PIPELINE_BIND_POINT_COMPUTE, pipelines[pass]);
      CmdBindDescriptorSets(buffers[pass], VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, NULL);
      if (fill) {
         CmdFillBuffer(buffers[pass], buffer, 4, 65540, 0xa5c31e79);
         CmdFillBuffer(buffers[pass], buffer, (word_count - bind_words - 3) * 4, VK_WHOLE_SIZE, 0x7900beef);
         VkMemoryBarrier barrier = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT};
         CmdPipelineBarrier(buffers[pass], VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &barrier, 0, NULL, 0, NULL);
      }
      CmdDispatch(buffers[pass], x, y, z);
      CHECK(EndCommandBuffer(buffers[pass]) == VK_SUCCESS);
      VkSubmitInfo submit = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                             .commandBufferCount = 1, .pCommandBuffers = &buffers[pass]};
      CHECK(QueueSubmit(queue, 1, &submit, fence) == VK_SUCCESS);
      CHECK(WaitForFences(device, 1, &fence, VK_TRUE, grid || fill ? 120000000000ull : 10000000000ull) == VK_SUCCESS);
      CHECK(InvalidateMappedMemoryRanges(device, 1, &range) == VK_SUCCESS);
      for (unsigned i = 0; i < word_count; i++) {
         uint32_t expected = 0xca000000 + i * 37;
         if (fill && i >= bind_words + 1 && i < bind_words + 1 + 16385) expected = 0xa5c31e79;
         if (fill && i >= word_count - 3 && i < word_count - 1) expected = 0x7900beef;
         if (grid && i >= 64 && i < 64 + x * y * z * 192) {
            unsigned group = (i - 64) / 192, lane = (i - 64) / 16 % 12;
            unsigned gx = group % x, gy = group / x % y, gz = group / (x * y);
            unsigned lx = lane % 3, ly = lane / 3 % 2, lz = lane / 6;
            const uint32_t values[] = {gx, gy, gz, lane, gx * 3 + lx, gy * 2 + ly,
               gz * 2 + lz, 0x12345678, x, y, z, 0xabcdef01, lx, ly, lz, 12};
            expected = values[(i - 64) % 16];
         } else if (!grid) {
            if ((!fill || !pass) && i >= bind_words + 64 && i < bind_words + 80)
               expected = 37 + (i - bind_words - 64) * 3;
            if (pass && i >= bind_words + 64 && i < bind_words + 76)
               expected = 112 + (i - bind_words - 64) * 7;
         }
         if (words[i] != expected)
            fprintf(stderr, "pass %u word %u: 0x%08x != 0x%08x\n", pass, i, words[i], expected);
         CHECK(words[i] == expected);
      }
      /* Without a flush these writes must not replace retained LOCAL results. */
      if (!grid && !fill) for (unsigned i = 76; i < 80; i++) words[i] = 0xdead0000 + i;
      if (grid) printf("PASS Apex loader dispatch: %ux%ux%u groups, local 3x2x2, %u words/guards\n",
                       x, y, z, word_count);
      if (fill) printf("PASS Apex loader fill: pass %u, 65540-byte range, whole-size tail, suballocation, compute state, %u words/guards\n",
                       pass, word_count);
      CHECK(ResetFences(device, 1, &fence) == VK_SUCCESS);
   }
   CHECK(QueueWaitIdle(queue) == VK_SUCCESS);
   DestroyFence(device, fence, NULL);
   DestroyCommandPool(device, commands, NULL);
   DestroyPipeline(device, pipelines[1], NULL);
   DestroyPipeline(device, pipelines[0], NULL);
   DestroyDescriptorPool(device, pool, NULL);
   DestroyPipelineLayout(device, layout, NULL);
   DestroyDescriptorSetLayout(device, set_layout, NULL);
   UnmapMemory(device, memory);
   DestroyBuffer(device, buffer, NULL);
   FreeMemory(device, memory, NULL);
   DestroyDevice(device, NULL);
   DestroyInstance(instance, NULL);
   CHECK(!dlclose(loader));
   if (!grid && !fill) puts("PASS Apex loader compute: 2 dispatches, main/partial specialization, retained LOCAL, 1024 words/guards each");
   return 0;
}
