/* SPDX-License-Identifier: MIT */
/* Public-loader sampler gate: a 4x4 RGBA8 texture read by texelFetch,
 * nearest and bilinear textureLod in a compute shader, checked against
 * host references. */
#include <vulkan/vulkan.h>
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { \
   fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); \
} } while (0)
#define VK(x) CHECK((x) == VK_SUCCESS)

static void *
read_file(const char *path, size_t *size)
{
   FILE *f = fopen(path, "rb");
   CHECK(f && !fseek(f, 0, SEEK_END));
   long n = ftell(f);
   CHECK(n > 0 && !fseek(f, 0, SEEK_SET));
   void *data = malloc(n);
   CHECK(data && fread(data, 1, n, f) == (size_t)n && !fclose(f));
   *size = n;
   return data;
}

static uint32_t
texel(unsigned x, unsigned y)
{
   return (x * 60 + 10) | (y * 60 + 20) << 8 | ((x + y) * 30 + 5) << 16 | (255 - x * y * 10) << 24;
}

static double
channel(unsigned x, unsigned y, unsigned c)
{
   return ((texel(x, y) >> (8 * c)) & 255) / 255.0;
}

int
main(int argc, char **argv)
{
   CHECK(argc == 3);
   setenv("APEX_DEVELOPMENT", "1", 1);
   setenv("VK_DRIVER_FILES", argv[1], 1);
   size_t code_size;
   uint32_t *code = read_file(argv[2], &code_size);
   void *loader = dlopen("libvulkan.so.1", RTLD_NOW);
   CHECK(loader);
   PFN_vkGetInstanceProcAddr gipa = (PFN_vkGetInstanceProcAddr)dlsym(loader, "vkGetInstanceProcAddr");
#define GI(name) PFN_vk##name name = (PFN_vk##name)gipa(instance, "vk" #name); CHECK(name)
   VkInstance instance = VK_NULL_HANDLE;
   GI(CreateInstance);
   VK(CreateInstance(&(VkInstanceCreateInfo){
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &(VkApplicationInfo){
         .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_0},
   }, NULL, &instance));
   GI(EnumeratePhysicalDevices); GI(GetPhysicalDeviceMemoryProperties); GI(CreateDevice);
   GI(GetDeviceProcAddr);
   uint32_t count = 1;
   VkPhysicalDevice physical;
   CHECK(EnumeratePhysicalDevices(instance, &count, &physical) >= 0 && count == 1);
   VkPhysicalDeviceMemoryProperties memory;
   GetPhysicalDeviceMemoryProperties(physical, &memory);
   VkDevice device;
   VK(CreateDevice(physical, &(VkDeviceCreateInfo){
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &(VkDeviceQueueCreateInfo){
         .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueCount = 1,
         .pQueuePriorities = &(float){1.0f}},
   }, NULL, &device));
#define GD(name) PFN_vk##name name = (PFN_vk##name)GetDeviceProcAddr(device, "vk" #name); CHECK(name)
   GD(GetDeviceQueue); GD(CreateBuffer); GD(AllocateMemory); GD(BindBufferMemory); GD(MapMemory);
   GD(FlushMappedMemoryRanges); GD(InvalidateMappedMemoryRanges); GD(CreateImage);
   GD(GetImageMemoryRequirements); GD(BindImageMemory); GD(CreateImageView); GD(CreateSampler);
   GD(CreateShaderModule); GD(CreateDescriptorSetLayout); GD(CreatePipelineLayout);
   GD(CreateComputePipelines); GD(CreateDescriptorPool); GD(AllocateDescriptorSets);
   GD(UpdateDescriptorSets); GD(CreateCommandPool); GD(AllocateCommandBuffers);
   GD(BeginCommandBuffer); GD(CmdCopyBufferToImage); GD(CmdPipelineBarrier); GD(CmdBindPipeline);
   GD(CmdBindDescriptorSets); GD(CmdDispatch); GD(EndCommandBuffer); GD(QueueSubmit);
   GD(QueueWaitIdle); GD(DestroyDevice);
   VkQueue queue;
   GetDeviceQueue(device, 0, 0, &queue);
   uint32_t host_type = UINT32_MAX;
   for (uint32_t i = 0; i < memory.memoryTypeCount && host_type == UINT32_MAX; i++)
      if (memory.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
         host_type = i;
   /* One allocation: upload at 0, results at 4 KiB, image at 64 KiB. */
   VkDeviceMemory allocation;
   VK(AllocateMemory(device, &(VkMemoryAllocateInfo){
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = 128 * 1024,
      .memoryTypeIndex = host_type}, NULL, &allocation));
   uint8_t *mapped;
   VK(MapMemory(device, allocation, 0, VK_WHOLE_SIZE, 0, (void **)&mapped));
   memset(mapped, 0, 128 * 1024);
   for (unsigned y = 0; y < 4; y++)
      for (unsigned x = 0; x < 4; x++)
         ((uint32_t *)mapped)[y * 4 + x] = texel(x, y);
   VkMappedMemoryRange whole = {
      .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, .memory = allocation, .size = VK_WHOLE_SIZE};
   VK(FlushMappedMemoryRanges(device, 1, &whole));
   VkBuffer upload, results;
   VK(CreateBuffer(device, &(VkBufferCreateInfo){.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = 64, .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT}, NULL, &upload));
   VK(BindBufferMemory(device, upload, allocation, 0));
   VK(CreateBuffer(device, &(VkBufferCreateInfo){.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
      .size = 48 * 16, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}, NULL, &results));
   VK(BindBufferMemory(device, results, allocation, 4096));
   VkImage image;
   VK(CreateImage(device, &(VkImageCreateInfo){
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
      .format = VK_FORMAT_R8G8B8A8_UNORM, .extent = {4, 4, 1}, .mipLevels = 1, .arrayLayers = 1,
      .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT}, NULL, &image));
   VK(BindImageMemory(device, image, allocation, 65536));
   VkImageView view;
   VK(CreateImageView(device, &(VkImageViewCreateInfo){
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = image,
      .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = VK_FORMAT_R8G8B8A8_UNORM,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}}, NULL, &view));
   VkSampler samplers[2];
   for (unsigned i = 0; i < 2; i++)
      VK(CreateSampler(device, &(VkSamplerCreateInfo){
         .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
         .magFilter = i ? VK_FILTER_LINEAR : VK_FILTER_NEAREST,
         .minFilter = i ? VK_FILTER_LINEAR : VK_FILTER_NEAREST,
         .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
         .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
         .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .maxLod = 0.0f}, NULL, &samplers[i]));
   VkShaderModule module;
   VK(CreateShaderModule(device, &(VkShaderModuleCreateInfo){
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = code_size, .pCode = code},
      NULL, &module));
   VkDescriptorSetLayout set_layout;
   VK(CreateDescriptorSetLayout(device, &(VkDescriptorSetLayoutCreateInfo){
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 3,
      .pBindings = (VkDescriptorSetLayoutBinding[]){
         {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
         {1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT},
         {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT}}},
      NULL, &set_layout));
   VkPipelineLayout layout;
   VK(CreatePipelineLayout(device, &(VkPipelineLayoutCreateInfo){
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
      .pSetLayouts = &set_layout}, NULL, &layout));
   VkPipeline pipeline;
   VK(CreateComputePipelines(device, VK_NULL_HANDLE, 1, &(VkComputePipelineCreateInfo){
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, .layout = layout,
      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = module, .pName = "main"}},
      NULL, &pipeline));
   VkDescriptorPool pool;
   VK(CreateDescriptorPool(device, &(VkDescriptorPoolCreateInfo){
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1, .poolSizeCount = 2,
      .pPoolSizes = (VkDescriptorPoolSize[]){{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
                                             {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2}}},
      NULL, &pool));
   VkDescriptorSet set;
   VK(AllocateDescriptorSets(device, &(VkDescriptorSetAllocateInfo){
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = pool,
      .descriptorSetCount = 1, .pSetLayouts = &set_layout}, &set));
   VkDescriptorImageInfo images[2] = {
      {samplers[0], view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
      {samplers[1], view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}};
   UpdateDescriptorSets(device, 3, (VkWriteDescriptorSet[]){
      {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = 0,
       .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
       .pBufferInfo = &(VkDescriptorBufferInfo){results, 0, VK_WHOLE_SIZE}},
      {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = 1,
       .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
       .pImageInfo = &images[0]},
      {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = 2,
       .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
       .pImageInfo = &images[1]}}, 0, NULL);
   VkCommandPool command_pool;
   VK(CreateCommandPool(device, &(VkCommandPoolCreateInfo){
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}, NULL, &command_pool));
   VkCommandBuffer cmd;
   VK(AllocateCommandBuffers(device, &(VkCommandBufferAllocateInfo){
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = command_pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1}, &cmd));
   VK(BeginCommandBuffer(cmd, &(VkCommandBufferBeginInfo){
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}));
   CmdCopyBufferToImage(cmd, upload, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
      &(VkBufferImageCopy){.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                           .imageExtent = {4, 4, 1}});
   CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
   CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, NULL);
   CmdDispatch(cmd, 1, 1, 1);
   VK(EndCommandBuffer(cmd));
   VK(QueueSubmit(queue, 1, &(VkSubmitInfo){
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cmd},
      VK_NULL_HANDLE));
   VK(QueueWaitIdle(queue));
   VK(InvalidateMappedMemoryRanges(device, 1, &whole));
   const float *out = (const float *)(mapped + 4096);
   unsigned failures = 0;
   for (unsigned i = 0; i < 48; i++) {
      unsigned x = i % 4, y = (i / 4) % 4, block = i / 16;
      for (unsigned c = 0; c < 4; c++) {
         double want = channel(x, y, c);
         if (block == 2) {
            /* Bilinear at the shared corner of texels x..x+1, y..y+1, clamped. */
            unsigned x1 = x + 1 > 3 ? 3 : x + 1, y1 = y + 1 > 3 ? 3 : y + 1;
            want = (channel(x, y, c) + channel(x1, y, c) + channel(x, y1, c) + channel(x1, y1, c)) / 4;
         }
         float got = out[i * 4 + c];
         if (fabs(got - want) > 1.5 / 255 && failures++ < 8)
            fprintf(stderr, "block %u texel (%u,%u) channel %u: got %f want %f\n", block, x, y, c,
                    got, want);
      }
   }
   if (failures) {
      fprintf(stderr, "FAIL Apex texture: %u mismatches\n", failures);
      return 1;
   }
   printf("PASS Apex texture: RGBA8 texelFetch, nearest and bilinear textureLod, 48 texels\n");
   DestroyDevice(device, NULL);
   return 0;
}
