/* SPDX-License-Identifier: MIT */
/* Public-loader compute runner for numeric gates: binds one storage buffer
 * holding the input file at set 0 binding 0, dispatches the given workgroup
 * count and writes the released buffer to the output file. */
#include <vulkan/vulkan.h>
#include <dlfcn.h>
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
   CHECK(f);
   CHECK(!fseek(f, 0, SEEK_END));
   long n = ftell(f);
   CHECK(n > 0 && !fseek(f, 0, SEEK_SET));
   void *data = malloc(n);
   CHECK(data && fread(data, 1, n, f) == (size_t)n && !fclose(f));
   *size = n;
   return data;
}

int
main(int argc, char **argv)
{
   CHECK(argc == 5 || argc == 6);
   size_t code_size, bytes;
   uint32_t *code = read_file(argv[1], &code_size);
   uint8_t *input = read_file(argv[2], &bytes);
   uint32_t groups = argc == 6 ? strtoul(argv[5], NULL, 0) : 1;
   CHECK(bytes % 4 == 0 && groups);
   setenv("APEX_DEVELOPMENT", "1", 1);
   setenv("VK_DRIVER_FILES", argv[4], 1);
   void *loader = dlopen("libvulkan.so.1", RTLD_NOW);
   CHECK(loader);
   PFN_vkGetInstanceProcAddr gipa = (PFN_vkGetInstanceProcAddr)dlsym(loader, "vkGetInstanceProcAddr");
   CHECK(gipa);
#define GI(name) PFN_vk##name name = (PFN_vk##name)gipa(instance, "vk" #name); CHECK(name)
   VkInstance instance = VK_NULL_HANDLE;
   GI(CreateInstance);
   VK(CreateInstance(&(VkInstanceCreateInfo){
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &(VkApplicationInfo){
         .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_0},
   }, NULL, &instance));
   GI(EnumeratePhysicalDevices); GI(GetPhysicalDeviceMemoryProperties);
   GI(CreateDevice); GI(GetDeviceProcAddr); GI(DestroyInstance);
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
   GD(GetDeviceQueue); GD(CreateBuffer); GD(GetBufferMemoryRequirements); GD(AllocateMemory);
   GD(BindBufferMemory); GD(MapMemory); GD(FlushMappedMemoryRanges);
   GD(InvalidateMappedMemoryRanges); GD(CreateShaderModule); GD(CreateDescriptorSetLayout);
   GD(CreatePipelineLayout); GD(CreateComputePipelines); GD(CreateDescriptorPool);
   GD(AllocateDescriptorSets); GD(UpdateDescriptorSets); GD(CreateCommandPool);
   GD(AllocateCommandBuffers); GD(BeginCommandBuffer); GD(CmdBindPipeline);
   GD(CmdBindDescriptorSets); GD(CmdDispatch); GD(EndCommandBuffer); GD(QueueSubmit);
   GD(QueueWaitIdle); GD(DeviceWaitIdle); GD(DestroyDevice);
   VkQueue queue;
   GetDeviceQueue(device, 0, 0, &queue);
   VkBuffer buffer;
   VK(CreateBuffer(device, &(VkBufferCreateInfo){
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = bytes,
      .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT}, NULL, &buffer));
   VkMemoryRequirements requirements;
   GetBufferMemoryRequirements(device, buffer, &requirements);
   uint32_t type = UINT32_MAX;
   for (uint32_t i = 0; i < memory.memoryTypeCount && type == UINT32_MAX; i++)
      if ((requirements.memoryTypeBits & (1u << i)) &&
          (memory.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
         type = i;
   CHECK(type != UINT32_MAX);
   VkDeviceMemory allocation;
   VK(AllocateMemory(device, &(VkMemoryAllocateInfo){
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = requirements.size,
      .memoryTypeIndex = type}, NULL, &allocation));
   VK(BindBufferMemory(device, buffer, allocation, 0));
   uint8_t *mapped;
   VK(MapMemory(device, allocation, 0, VK_WHOLE_SIZE, 0, (void **)&mapped));
   memcpy(mapped, input, bytes);
   VkMappedMemoryRange range = {
      .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, .memory = allocation, .size = VK_WHOLE_SIZE};
   VK(FlushMappedMemoryRanges(device, 1, &range));
   VkShaderModule module;
   VK(CreateShaderModule(device, &(VkShaderModuleCreateInfo){
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = code_size, .pCode = code},
      NULL, &module));
   VkDescriptorSetLayout set_layout;
   VK(CreateDescriptorSetLayout(device, &(VkDescriptorSetLayoutCreateInfo){
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 1,
      .pBindings = &(VkDescriptorSetLayoutBinding){
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT}}, NULL, &set_layout));
   VkPipelineLayout layout;
   VK(CreatePipelineLayout(device, &(VkPipelineLayoutCreateInfo){
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
      .pSetLayouts = &set_layout}, NULL, &layout));
   VkPipeline pipeline;
   VK(CreateComputePipelines(device, VK_NULL_HANDLE, 1, &(VkComputePipelineCreateInfo){
      .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, .layout = layout,
      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = module, .pName = "main"},
   }, NULL, &pipeline));
   VkDescriptorPool pool;
   VK(CreateDescriptorPool(device, &(VkDescriptorPoolCreateInfo){
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1, .poolSizeCount = 1,
      .pPoolSizes = &(VkDescriptorPoolSize){VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1}}, NULL, &pool));
   VkDescriptorSet set;
   VK(AllocateDescriptorSets(device, &(VkDescriptorSetAllocateInfo){
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = pool,
      .descriptorSetCount = 1, .pSetLayouts = &set_layout}, &set));
   UpdateDescriptorSets(device, 1, &(VkWriteDescriptorSet){
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &(VkDescriptorBufferInfo){buffer, 0, VK_WHOLE_SIZE}}, 0, NULL);
   VkCommandPool command_pool;
   VK(CreateCommandPool(device, &(VkCommandPoolCreateInfo){
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}, NULL, &command_pool));
   VkCommandBuffer cmd;
   VK(AllocateCommandBuffers(device, &(VkCommandBufferAllocateInfo){
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = command_pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1}, &cmd));
   VK(BeginCommandBuffer(cmd, &(VkCommandBufferBeginInfo){
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}));
   CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
   CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, NULL);
   CmdDispatch(cmd, groups, 1, 1);
   VK(EndCommandBuffer(cmd));
   VK(QueueSubmit(queue, 1, &(VkSubmitInfo){
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cmd},
      VK_NULL_HANDLE));
   VK(QueueWaitIdle(queue));
   VK(InvalidateMappedMemoryRanges(device, 1, &range));
   FILE *out = fopen(argv[3], "wb");
   CHECK(out && fwrite(mapped, 1, bytes, out) == bytes && !fclose(out));
   VK(DeviceWaitIdle(device));
   DestroyDevice(device, NULL);
   DestroyInstance(instance, NULL);
   return 0;
}
