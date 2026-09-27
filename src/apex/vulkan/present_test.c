/* SPDX-License-Identifier: MIT */
/* Public-loader presentation gate: VK_KHR_display surface on the Apex
 * connector, a FIFO swapchain, and one rotating triangle per frame over a
 * cycling clear color. Prints the chosen mode and presented frame count. */
#include <vulkan/vulkan.h>
#include <dlfcn.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

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

int
main(int argc, char **argv)
{
   CHECK(argc == 5 || argc == 6);
   const unsigned frames = strtoul(argv[4], NULL, 0);
   /* Optional seconds to keep the last frame on screen for capture. */
   const unsigned hold = argc == 6 ? strtoul(argv[5], NULL, 0) : 0;
   setenv("APEX_DEVELOPMENT", "1", 1);
   setenv("VK_DRIVER_FILES", argv[1], 1);
   size_t vs_size, fs_size;
   uint32_t *vs_code = read_file(argv[2], &vs_size), *fs_code = read_file(argv[3], &fs_size);
   void *loader = dlopen("libvulkan.so.1", RTLD_NOW);
   CHECK(loader);
   PFN_vkGetInstanceProcAddr gipa = (PFN_vkGetInstanceProcAddr)dlsym(loader, "vkGetInstanceProcAddr");
#define GI(name) PFN_vk##name name = (PFN_vk##name)gipa(instance, "vk" #name); CHECK(name)
   VkInstance instance = VK_NULL_HANDLE;
   GI(CreateInstance);
   const char *instance_extensions[] = {VK_KHR_SURFACE_EXTENSION_NAME, VK_KHR_DISPLAY_EXTENSION_NAME};
   VK(CreateInstance(&(VkInstanceCreateInfo){
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &(VkApplicationInfo){
         .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_0},
      .enabledExtensionCount = 2, .ppEnabledExtensionNames = instance_extensions,
   }, NULL, &instance));
   GI(EnumeratePhysicalDevices); GI(GetPhysicalDeviceMemoryProperties); GI(CreateDevice);
   GI(GetDeviceProcAddr); GI(DestroyInstance); GI(GetPhysicalDeviceDisplayPropertiesKHR);
   GI(GetDisplayModePropertiesKHR); GI(GetPhysicalDeviceDisplayPlanePropertiesKHR);
   GI(CreateDisplayPlaneSurfaceKHR); GI(GetPhysicalDeviceSurfaceSupportKHR);
   GI(GetPhysicalDeviceSurfaceCapabilitiesKHR); GI(GetPhysicalDeviceSurfaceFormatsKHR);
   GI(DestroySurfaceKHR);
   uint32_t count = 1;
   VkPhysicalDevice physical;
   CHECK(EnumeratePhysicalDevices(instance, &count, &physical) >= 0 && count == 1);

   count = 1;
   VkDisplayPropertiesKHR display;
   CHECK(GetPhysicalDeviceDisplayPropertiesKHR(physical, &count, &display) >= 0 && count == 1);
   uint32_t mode_count = 0;
   VK(GetDisplayModePropertiesKHR(physical, display.display, &mode_count, NULL));
   CHECK(mode_count);
   VkDisplayModePropertiesKHR *modes = calloc(mode_count, sizeof(*modes));
   VK(GetDisplayModePropertiesKHR(physical, display.display, &mode_count, modes));
   unsigned chosen = 0;
   for (unsigned i = 0; i < mode_count; i++) {
      const VkDisplayModeParametersKHR *m = &modes[i].parameters;
      if (m->visibleRegion.width == 1920 && m->visibleRegion.height == 1080 &&
          m->refreshRate >= 59000 && m->refreshRate <= 61000)
         chosen = i;
   }
   VkExtent2D extent = modes[chosen].parameters.visibleRegion;
   printf("display %s mode %ux%u@%.3f\n", display.displayName ? display.displayName : "?",
          extent.width, extent.height, modes[chosen].parameters.refreshRate / 1000.0);
   count = 1;
   VkDisplayPlanePropertiesKHR plane;
   CHECK(GetPhysicalDeviceDisplayPlanePropertiesKHR(physical, &count, &plane) >= 0 && count >= 1);
   VkSurfaceKHR surface;
   VK(CreateDisplayPlaneSurfaceKHR(instance, &(VkDisplaySurfaceCreateInfoKHR){
      .sType = VK_STRUCTURE_TYPE_DISPLAY_SURFACE_CREATE_INFO_KHR,
      .displayMode = modes[chosen].displayMode, .planeIndex = 0,
      .transform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
      .alphaMode = VK_DISPLAY_PLANE_ALPHA_OPAQUE_BIT_KHR, .imageExtent = extent}, NULL, &surface));
   VkBool32 supported = VK_FALSE;
   VK(GetPhysicalDeviceSurfaceSupportKHR(physical, 0, surface, &supported));
   CHECK(supported);
   VkSurfaceCapabilitiesKHR caps;
   VK(GetPhysicalDeviceSurfaceCapabilitiesKHR(physical, surface, &caps));
   uint32_t format_count = 0;
   VK(GetPhysicalDeviceSurfaceFormatsKHR(physical, surface, &format_count, NULL));
   VkSurfaceFormatKHR *surface_formats = calloc(format_count, sizeof(*surface_formats));
   VK(GetPhysicalDeviceSurfaceFormatsKHR(physical, surface, &format_count, surface_formats));
   bool bgra = false;
   for (unsigned i = 0; i < format_count; i++)
      bgra |= surface_formats[i].format == VK_FORMAT_B8G8R8A8_UNORM;
   CHECK(bgra);

   VkDevice device;
   const char *device_extensions[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
   VK(CreateDevice(physical, &(VkDeviceCreateInfo){
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &(VkDeviceQueueCreateInfo){
         .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueCount = 1,
         .pQueuePriorities = &(float){1.0f}},
      .enabledExtensionCount = 1, .ppEnabledExtensionNames = device_extensions,
   }, NULL, &device));
#define GD(name) PFN_vk##name name = (PFN_vk##name)GetDeviceProcAddr(device, "vk" #name); CHECK(name)
   GD(GetDeviceQueue); GD(CreateBuffer); GD(GetBufferMemoryRequirements); GD(AllocateMemory);
   GD(BindBufferMemory); GD(MapMemory); GD(FlushMappedMemoryRanges); GD(CreateImageView);
   GD(CreateShaderModule); GD(CreatePipelineLayout); GD(CreateGraphicsPipelines);
   GD(CreateRenderPass); GD(CreateFramebuffer); GD(CreateCommandPool); GD(AllocateCommandBuffers);
   GD(BeginCommandBuffer); GD(CmdBeginRenderPass); GD(CmdBindPipeline); GD(CmdBindVertexBuffers);
   GD(CmdDraw); GD(CmdEndRenderPass); GD(EndCommandBuffer); GD(QueueSubmit); GD(DeviceWaitIdle);
   GD(DestroyDevice); GD(CreateSwapchainKHR); GD(GetSwapchainImagesKHR); GD(AcquireNextImageKHR);
   GD(QueuePresentKHR); GD(DestroySwapchainKHR); GD(CreateSemaphore); GD(CreateFence);
   GD(WaitForFences); GD(ResetFences); GD(ResetCommandBuffer);
   VkQueue queue;
   GetDeviceQueue(device, 0, 0, &queue);

   VkSwapchainKHR swapchain;
   VK(CreateSwapchainKHR(device, &(VkSwapchainCreateInfoKHR){
      .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR, .surface = surface,
      .minImageCount = caps.minImageCount < 2 ? 2 : caps.minImageCount,
      .imageFormat = VK_FORMAT_B8G8R8A8_UNORM, .imageColorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
      .imageExtent = extent, .imageArrayLayers = 1,
      .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
      .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE, .preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
      .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR, .presentMode = VK_PRESENT_MODE_FIFO_KHR,
      .clipped = VK_TRUE}, NULL, &swapchain));
   uint32_t image_count = 0;
   VK(GetSwapchainImagesKHR(device, swapchain, &image_count, NULL));
   VkImage *images = calloc(image_count, sizeof(*images));
   VK(GetSwapchainImagesKHR(device, swapchain, &image_count, images));

   VkRenderPass pass;
   VK(CreateRenderPass(device, &(VkRenderPassCreateInfo){
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO, .attachmentCount = 1,
      .pAttachments = &(VkAttachmentDescription){
         .format = VK_FORMAT_B8G8R8A8_UNORM, .samples = VK_SAMPLE_COUNT_1_BIT,
         .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
         .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE, .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
         .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED, .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR},
      .subpassCount = 1,
      .pSubpasses = &(VkSubpassDescription){
         .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS, .colorAttachmentCount = 1,
         .pColorAttachments = &(VkAttachmentReference){0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL}},
   }, NULL, &pass));
   VkFramebuffer *framebuffers = calloc(image_count, sizeof(*framebuffers));
   for (unsigned i = 0; i < image_count; i++) {
      VkImageView view;
      VK(CreateImageView(device, &(VkImageViewCreateInfo){
         .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = images[i],
         .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = VK_FORMAT_B8G8R8A8_UNORM,
         .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}}, NULL, &view));
      VK(CreateFramebuffer(device, &(VkFramebufferCreateInfo){
         .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO, .renderPass = pass, .attachmentCount = 1,
         .pAttachments = &view, .width = extent.width, .height = extent.height, .layers = 1},
         NULL, &framebuffers[i]));
   }

   VkShaderModule modules[2];
   VK(CreateShaderModule(device, &(VkShaderModuleCreateInfo){
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = vs_size, .pCode = vs_code},
      NULL, &modules[0]));
   VK(CreateShaderModule(device, &(VkShaderModuleCreateInfo){
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = fs_size, .pCode = fs_code},
      NULL, &modules[1]));
   VkPipelineLayout layout;
   VK(CreatePipelineLayout(device, &(VkPipelineLayoutCreateInfo){
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO}, NULL, &layout));
   VkPipeline pipeline;
   VK(CreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &(VkGraphicsPipelineCreateInfo){
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .stageCount = 2,
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
            {0, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 0}, {1, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 16}}},
      .pInputAssemblyState = &(VkPipelineInputAssemblyStateCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
         .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST},
      .pViewportState = &(VkPipelineViewportStateCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1,
         .pViewports = &(VkViewport){0, 0, extent.width, extent.height, 0, 1}, .scissorCount = 1,
         .pScissors = &(VkRect2D){{0, 0}, extent}},
      .pRasterizationState = &(VkPipelineRasterizationStateCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
         .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE, .lineWidth = 1},
      .pMultisampleState = &(VkPipelineMultisampleStateCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
         .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT},
      .pColorBlendState = &(VkPipelineColorBlendStateCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, .attachmentCount = 1,
         .pAttachments = &(VkPipelineColorBlendAttachmentState){.colorWriteMask = 0xf}},
      .layout = layout, .renderPass = pass,
   }, NULL, &pipeline));

   /* One small vertex buffer per swapchain image, rewritten before each frame. */
   VkPhysicalDeviceMemoryProperties memory;
   GetPhysicalDeviceMemoryProperties(physical, &memory);
   uint32_t host_type = UINT32_MAX;
   for (uint32_t i = 0; i < memory.memoryTypeCount && host_type == UINT32_MAX; i++)
      if (memory.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
         host_type = i;
   VkDeviceMemory vertex_memory;
   VK(AllocateMemory(device, &(VkMemoryAllocateInfo){
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = 4096 * image_count,
      .memoryTypeIndex = host_type}, NULL, &vertex_memory));
   uint8_t *mapped;
   VK(MapMemory(device, vertex_memory, 0, VK_WHOLE_SIZE, 0, (void **)&mapped));
   VkBuffer *vertex_buffers = calloc(image_count, sizeof(*vertex_buffers));
   for (unsigned i = 0; i < image_count; i++) {
      VK(CreateBuffer(device, &(VkBufferCreateInfo){
         .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 96,
         .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT}, NULL, &vertex_buffers[i]));
      VK(BindBufferMemory(device, vertex_buffers[i], vertex_memory, 4096 * i));
   }
   VkCommandPool command_pool;
   VK(CreateCommandPool(device, &(VkCommandPoolCreateInfo){
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
      .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT}, NULL, &command_pool));
   VkCommandBuffer *cmds = calloc(image_count, sizeof(*cmds));
   VK(AllocateCommandBuffers(device, &(VkCommandBufferAllocateInfo){
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = command_pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = image_count}, cmds));
   VkSemaphore acquired, rendered;
   VK(CreateSemaphore(device, &(VkSemaphoreCreateInfo){.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO},
                      NULL, &acquired));
   VK(CreateSemaphore(device, &(VkSemaphoreCreateInfo){.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO},
                      NULL, &rendered));
   VkFence *fences = calloc(image_count, sizeof(*fences));
   for (unsigned i = 0; i < image_count; i++)
      VK(CreateFence(device, &(VkFenceCreateInfo){.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                                                  .flags = VK_FENCE_CREATE_SIGNALED_BIT}, NULL, &fences[i]));

   struct timespec start, end;
   clock_gettime(CLOCK_MONOTONIC, &start);
   for (unsigned frame = 0; frame < frames; frame++) {
      uint32_t index;
      VK(AcquireNextImageKHR(device, swapchain, UINT64_MAX, acquired, VK_NULL_HANDLE, &index));
      VK(WaitForFences(device, 1, &fences[index], VK_TRUE, UINT64_MAX));
      VK(ResetFences(device, 1, &fences[index]));
      float angle = frame * 0.05f;
      float *v = (float *)(mapped + 4096 * index);
      for (unsigned k = 0; k < 3; k++) {
         float a = angle + k * 2.0943951f;
         const float color[3][4] = {{1, 0.2f, 0.2f, 1}, {0.2f, 1, 0.2f, 1}, {0.2f, 0.4f, 1, 1}};
         float x = 0.7f * cosf(a) * extent.height / extent.width, y = 0.7f * sinf(a);
         memcpy(v + k * 8, (float[4]){x, y, 0.5f, 1}, 16);
         memcpy(v + k * 8 + 4, color[k], 16);
      }
      VkMappedMemoryRange range = {
         .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, .memory = vertex_memory,
         .offset = 4096 * index, .size = 4096};
      VK(FlushMappedMemoryRanges(device, 1, &range));
      VkCommandBuffer cmd = cmds[index];
      VK(ResetCommandBuffer(cmd, 0));
      VK(BeginCommandBuffer(cmd, &(VkCommandBufferBeginInfo){
         .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}));
      float phase = frame * 0.02f;
      CmdBeginRenderPass(cmd, &(VkRenderPassBeginInfo){
         .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, .renderPass = pass,
         .framebuffer = framebuffers[index], .renderArea = {{0, 0}, extent}, .clearValueCount = 1,
         .pClearValues = &(VkClearValue){.color = {.float32 = {
            0.15f + 0.1f * sinf(phase), 0.15f, 0.25f + 0.1f * cosf(phase), 1}}}},
         VK_SUBPASS_CONTENTS_INLINE);
      CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
      CmdBindVertexBuffers(cmd, 0, 1, &vertex_buffers[index], &(VkDeviceSize){0});
      CmdDraw(cmd, 3, 1, 0, 0);
      CmdEndRenderPass(cmd);
      VK(EndCommandBuffer(cmd));
      VkPipelineStageFlags stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
      VK(QueueSubmit(queue, 1, &(VkSubmitInfo){
         .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .waitSemaphoreCount = 1, .pWaitSemaphores = &acquired,
         .pWaitDstStageMask = &stage, .commandBufferCount = 1, .pCommandBuffers = &cmd,
         .signalSemaphoreCount = 1, .pSignalSemaphores = &rendered}, fences[index]));
      VK(QueuePresentKHR(queue, &(VkPresentInfoKHR){
         .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR, .waitSemaphoreCount = 1,
         .pWaitSemaphores = &rendered, .swapchainCount = 1, .pSwapchains = &swapchain,
         .pImageIndices = &index}));
   }
   VK(DeviceWaitIdle(device));
   clock_gettime(CLOCK_MONOTONIC, &end);
   double seconds = end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec) * 1e-9;
   printf("PASS Apex present: %u frames on %u swapchain images in %.2f s (%.2f frames/s)\n",
          frames, image_count, seconds, frames / seconds);
   fflush(stdout);
   if (hold)
      nanosleep(&(struct timespec){.tv_sec = hold}, NULL);
   DestroySwapchainKHR(device, swapchain, NULL);
   DestroyDevice(device, NULL);
   DestroySurfaceKHR(instance, surface, NULL);
   DestroyInstance(instance, NULL);
   return 0;
}
