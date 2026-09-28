/* SPDX-License-Identifier: MIT */
/*
 * Frame-time benchmark through the public loader: renders offscreen frames of
 * a representative scene and reports per-frame wall time from submit to idle.
 *
 *   apex-bench ICD desktop|game [frames]
 *
 * desktop: 1920x1080 RGBA8, a full-screen textured background plus eight
 *          blended 640x480 textured windows (compositor-like).
 * game:    1280x720 RGBA8 + D32, a textured, depth-tested grid of 64k
 *          triangles in 64 draws.
 */
#include <dlfcn.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vulkan/vulkan.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)
#define VK(x) CHECK((x) == VK_SUCCESS)

static uint32_t *
read_file(const char *path, size_t *size)
{
   FILE *f = fopen(path, "rb");
   CHECK(f && !fseek(f, 0, SEEK_END));
   long bytes = ftell(f);
   CHECK(bytes > 0 && !fseek(f, 0, SEEK_SET));
   uint32_t *data = malloc(bytes);
   CHECK(data && fread(data, 1, bytes, f) == (size_t)bytes);
   fclose(f);
   *size = bytes;
   return data;
}

struct vertex { float position[4], uv[2], pad[2]; };

static double
now(void)
{
   struct timespec t;
   clock_gettime(CLOCK_MONOTONIC, &t);
   return t.tv_sec + t.tv_nsec * 1e-9;
}

int
main(int argc, char **argv)
{
   CHECK(argc >= 3 && (!strcmp(argv[2], "desktop") || !strcmp(argv[2], "game")));
   const bool game = !strcmp(argv[2], "game");
   const unsigned frames = argc > 3 ? atoi(argv[3]) : 10;
   const uint32_t width = game ? 1280 : 1920, height = game ? 720 : 1080;
   char dir[512];
   snprintf(dir, sizeof(dir), "%s", argv[0]);
   char *slash = strrchr(dir, '/');
   if (slash)
      *slash = 0;
   else
      strcpy(dir, ".");
   char path[600];
   size_t vs_size, fs_size;
   snprintf(path, sizeof(path), "%s/bench.vert.spv", dir);
   uint32_t *vs_code = read_file(path, &vs_size);
   snprintf(path, sizeof(path), "%s/bench.frag.spv", dir);
   uint32_t *fs_code = read_file(path, &fs_size);

   setenv("APEX_DEVELOPMENT", "1", 1);
   setenv("VK_DRIVER_FILES", argv[1], 1);
   void *loader = dlopen("libvulkan.so.1", RTLD_NOW);
   CHECK(loader);
   PFN_vkGetInstanceProcAddr gipa = (PFN_vkGetInstanceProcAddr)dlsym(loader, "vkGetInstanceProcAddr");
#define GI(name) PFN_vk##name name = (PFN_vk##name)gipa(instance, "vk" #name); CHECK(name)
   VkInstance instance = VK_NULL_HANDLE;
   GI(CreateInstance);
   VK(CreateInstance(&(VkInstanceCreateInfo){
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &(VkApplicationInfo){
         .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_1}}, NULL, &instance));
   GI(EnumeratePhysicalDevices); GI(GetPhysicalDeviceMemoryProperties); GI(CreateDevice); GI(GetDeviceProcAddr);
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
         .pQueuePriorities = &(float){1.0f}}}, NULL, &device));
#define GD(name) PFN_vk##name name = (PFN_vk##name)GetDeviceProcAddr(device, "vk" #name); CHECK(name)
   GD(GetDeviceQueue); GD(CreateBuffer); GD(AllocateMemory); GD(BindBufferMemory); GD(MapMemory);
   GD(FlushMappedMemoryRanges); GD(CreateImage); GD(GetImageMemoryRequirements); GD(BindImageMemory);
   GD(CreateImageView); GD(CreateSampler); GD(CreateDescriptorSetLayout); GD(CreateDescriptorPool);
   GD(AllocateDescriptorSets); GD(UpdateDescriptorSets); GD(CreateShaderModule); GD(CreatePipelineLayout);
   GD(CreateGraphicsPipelines); GD(CreateRenderPass); GD(CreateFramebuffer); GD(CreateCommandPool);
   GD(AllocateCommandBuffers); GD(BeginCommandBuffer); GD(CmdBeginRenderPass); GD(CmdBindPipeline);
   GD(CmdBindVertexBuffers); GD(CmdBindDescriptorSets); GD(CmdDraw); GD(CmdEndRenderPass);
   GD(CmdCopyBufferToImage); GD(CmdPipelineBarrier); GD(EndCommandBuffer); GD(QueueSubmit); GD(QueueWaitIdle);
   VkQueue queue;
   GetDeviceQueue(device, 0, 0, &queue);

   uint32_t host_type = UINT32_MAX;
   for (uint32_t i = 0; i < memory.memoryTypeCount && host_type == UINT32_MAX; i++)
      if (memory.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
         host_type = i;
   CHECK(host_type != UINT32_MAX);

   /* Geometry: game grid (256 x 128 quads) or desktop quads. */
   const unsigned texture_size = 256;
   unsigned vertex_count;
   struct vertex *vertices;
   if (game) {
      const unsigned gx = 256, gy = 128;
      vertex_count = gx * gy * 6;
      vertices = calloc(vertex_count, sizeof(*vertices));
      unsigned v = 0;
      for (unsigned y = 0; y < gy; y++) {
         for (unsigned x = 0; x < gx; x++) {
            static const unsigned corner[6][2] = {{0, 0}, {1, 0}, {0, 1}, {1, 0}, {1, 1}, {0, 1}};
            for (unsigned c = 0; c < 6; c++) {
               float u = (float)(x + corner[c][0]) / gx, w = (float)(y + corner[c][1]) / gy;
               /* A tilted, wavy terrain: depth varies across the screen. */
               float z = 0.2f + 0.6f * w + 0.05f * sinf(u * 20.0f);
               vertices[v++] = (struct vertex){{u * 2 - 1, w * 2 - 1, z, 1}, {u * 8, w * 4}};
            }
         }
      }
   } else {
      vertex_count = 9 * 6;
      vertices = calloc(vertex_count, sizeof(*vertices));
      for (unsigned q = 0; q < 9; q++) {
         float x0 = -1, y0 = -1, x1 = 1, y1 = 1;
         if (q) {
            float w = 640.0f / width * 2, h = 480.0f / height * 2;
            x0 = -0.9f + (q - 1) % 4 * 0.45f;
            y0 = -0.8f + (q - 1) / 4 * 0.8f;
            x1 = x0 + w;
            y1 = y0 + h;
         }
         const float corner[6][2] = {{x0, y0}, {x1, y0}, {x0, y1}, {x1, y0}, {x1, y1}, {x0, y1}};
         for (unsigned c = 0; c < 6; c++)
            vertices[q * 6 + c] = (struct vertex){{corner[c][0], corner[c][1], 0.5f, 1},
                                                  {corner[c][0] == x0 ? 0.0f : 1.0f,
                                                   corner[c][1] == y0 ? 0.0f : 1.0f}};
      }
   }
   const VkDeviceSize vertex_bytes = (VkDeviceSize)vertex_count * sizeof(struct vertex);
   const VkDeviceSize texel_offset = (vertex_bytes + 4095) & ~4095ull;
   const VkDeviceSize texel_bytes = texture_size * texture_size * 4;
   const VkDeviceSize image_offset = (texel_offset + texel_bytes + 65535) & ~65535ull;
   const VkDeviceSize total = image_offset + 3 * (VkDeviceSize)width * height * 4 + (8 << 20);
   VkDeviceMemory allocation;
   VK(AllocateMemory(device, &(VkMemoryAllocateInfo){
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = total,
      .memoryTypeIndex = host_type}, NULL, &allocation));
   uint8_t *mapped;
   VK(MapMemory(device, allocation, 0, VK_WHOLE_SIZE, 0, (void **)&mapped));
   memcpy(mapped, vertices, vertex_bytes);
   uint32_t *texels = (uint32_t *)(mapped + texel_offset);
   for (unsigned y = 0; y < texture_size; y++)
      for (unsigned x = 0; x < texture_size; x++)
         texels[y * texture_size + x] = ((x ^ y) & 16 ? 0xc0ffffff : 0xc0402010) | ((x * 7) & 0xff) << 8;
   VK(FlushMappedMemoryRanges(device, 1, &(VkMappedMemoryRange){
      .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, .memory = allocation, .size = VK_WHOLE_SIZE}));

   VkBuffer vertex_buffer, staging;
   VK(CreateBuffer(device, &(VkBufferCreateInfo){
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = vertex_bytes,
      .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT}, NULL, &vertex_buffer));
   VK(BindBufferMemory(device, vertex_buffer, allocation, 0));
   VK(CreateBuffer(device, &(VkBufferCreateInfo){
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = texel_bytes,
      .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT}, NULL, &staging));
   VK(BindBufferMemory(device, staging, allocation, texel_offset));

   /* Texture, color target and (game) depth target. */
   const VkFormat formats[3] = {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_D32_SFLOAT};
   const VkExtent3D extents[3] = {{texture_size, texture_size, 1}, {width, height, 1}, {width, height, 1}};
   const VkImageUsageFlags usages[3] = {
      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
      VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT};
   VkImage images[3];
   VkImageView views[3];
   VkDeviceSize offset = image_offset;
   for (unsigned i = 0; i < 3; i++) {
      VK(CreateImage(device, &(VkImageCreateInfo){
         .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = formats[i],
         .extent = extents[i], .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
         .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = usages[i]}, NULL, &images[i]));
      VkMemoryRequirements requirements;
      GetImageMemoryRequirements(device, images[i], &requirements);
      offset = (offset + requirements.alignment - 1) & ~(requirements.alignment - 1);
      VK(BindImageMemory(device, images[i], allocation, offset));
      offset += requirements.size;
      VK(CreateImageView(device, &(VkImageViewCreateInfo){
         .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = images[i],
         .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = formats[i],
         .subresourceRange = {i == 2 ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}},
         NULL, &views[i]));
   }
   CHECK(offset <= total);

   VkSampler sampler;
   VK(CreateSampler(device, &(VkSamplerCreateInfo){
      .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .magFilter = VK_FILTER_LINEAR,
      .minFilter = VK_FILTER_LINEAR, .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT,
      .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT, .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT,
      .maxLod = 0}, NULL, &sampler));
   VkDescriptorSetLayout set_layout;
   VK(CreateDescriptorSetLayout(device, &(VkDescriptorSetLayoutCreateInfo){
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 1,
      .pBindings = &(VkDescriptorSetLayoutBinding){0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                                   VK_SHADER_STAGE_FRAGMENT_BIT}}, NULL, &set_layout));
   VkDescriptorPool pool;
   VK(CreateDescriptorPool(device, &(VkDescriptorPoolCreateInfo){
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1, .poolSizeCount = 1,
      .pPoolSizes = &(VkDescriptorPoolSize){VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1}}, NULL, &pool));
   VkDescriptorSet set;
   VK(AllocateDescriptorSets(device, &(VkDescriptorSetAllocateInfo){
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = pool,
      .descriptorSetCount = 1, .pSetLayouts = &set_layout}, &set));
   UpdateDescriptorSets(device, 1, &(VkWriteDescriptorSet){
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .descriptorCount = 1,
      .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
      .pImageInfo = &(VkDescriptorImageInfo){sampler, views[0], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}},
      0, NULL);

   const unsigned attachments = game ? 2 : 1;
   VkRenderPass pass;
   VK(CreateRenderPass(device, &(VkRenderPassCreateInfo){
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO, .attachmentCount = attachments,
      .pAttachments = (VkAttachmentDescription[]){
         {.format = formats[1], .samples = VK_SAMPLE_COUNT_1_BIT, .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
          .storeOp = VK_ATTACHMENT_STORE_OP_STORE, .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
          .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
          .finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
         {.format = formats[2], .samples = VK_SAMPLE_COUNT_1_BIT, .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
          .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE, .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
          .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
          .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL}},
      .subpassCount = 1,
      .pSubpasses = &(VkSubpassDescription){
         .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS, .colorAttachmentCount = 1,
         .pColorAttachments = &(VkAttachmentReference){0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
         .pDepthStencilAttachment = game ?
            &(VkAttachmentReference){1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL} : NULL}},
      NULL, &pass));
   VkFramebuffer framebuffer;
   VK(CreateFramebuffer(device, &(VkFramebufferCreateInfo){
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO, .renderPass = pass, .attachmentCount = attachments,
      .pAttachments = &views[1], .width = width, .height = height, .layers = 1}, NULL, &framebuffer));

   VkShaderModule modules[2];
   VK(CreateShaderModule(device, &(VkShaderModuleCreateInfo){
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = vs_size, .pCode = vs_code},
      NULL, &modules[0]));
   VK(CreateShaderModule(device, &(VkShaderModuleCreateInfo){
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = fs_size, .pCode = fs_code},
      NULL, &modules[1]));
   VkPipelineLayout layout;
   VK(CreatePipelineLayout(device, &(VkPipelineLayoutCreateInfo){
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
      .pSetLayouts = &set_layout}, NULL, &layout));
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
         .pVertexBindingDescriptions = &(VkVertexInputBindingDescription){
            0, sizeof(struct vertex), VK_VERTEX_INPUT_RATE_VERTEX},
         .vertexAttributeDescriptionCount = 2,
         .pVertexAttributeDescriptions = (VkVertexInputAttributeDescription[]){
            {0, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 0}, {1, 0, VK_FORMAT_R32G32_SFLOAT, 16}}},
      .pInputAssemblyState = &(VkPipelineInputAssemblyStateCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
         .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST},
      .pViewportState = &(VkPipelineViewportStateCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1,
         .pViewports = &(VkViewport){0, 0, width, height, 0, 1}, .scissorCount = 1,
         .pScissors = &(VkRect2D){{0, 0}, {width, height}}},
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
      .layout = layout, .renderPass = pass}, NULL, &pipeline));

   VkCommandPool command_pool;
   VK(CreateCommandPool(device, &(VkCommandPoolCreateInfo){
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}, NULL, &command_pool));
   VkCommandBuffer upload, frame;
   VK(AllocateCommandBuffers(device, &(VkCommandBufferAllocateInfo){
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = command_pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1}, &upload));
   VK(AllocateCommandBuffers(device, &(VkCommandBufferAllocateInfo){
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = command_pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1}, &frame));
   VK(BeginCommandBuffer(upload, &(VkCommandBufferBeginInfo){
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}));
   CmdCopyBufferToImage(upload, staging, images[0], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
      &(VkBufferImageCopy){.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                           .imageExtent = extents[0]});
   VK(EndCommandBuffer(upload));
   VK(QueueSubmit(queue, 1, &(VkSubmitInfo){
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &upload},
      VK_NULL_HANDLE));
   VK(QueueWaitIdle(queue));

   /* One frame: the game draws its grid in 64 slices, the desktop 9 quads. */
   const unsigned draws = game ? 64 : 9;
   VK(BeginCommandBuffer(frame, &(VkCommandBufferBeginInfo){
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}));
   CmdBeginRenderPass(frame, &(VkRenderPassBeginInfo){
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, .renderPass = pass, .framebuffer = framebuffer,
      .renderArea = {{0, 0}, {width, height}}, .clearValueCount = attachments,
      .pClearValues = (VkClearValue[]){{.color = {.float32 = {0.1f, 0.1f, 0.1f, 1}}},
                                       {.depthStencil = {1.0f, 0}}}}, VK_SUBPASS_CONTENTS_INLINE);
   CmdBindPipeline(frame, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
   CmdBindDescriptorSets(frame, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &set, 0, NULL);
   CmdBindVertexBuffers(frame, 0, 1, &vertex_buffer, &(VkDeviceSize){0});
   for (unsigned d = 0; d < draws; d++)
      CmdDraw(frame, vertex_count / draws, 1, d * (vertex_count / draws), 0);
   CmdEndRenderPass(frame);
   VK(EndCommandBuffer(frame));

   double total_time = 0, best = 1e9;
   for (unsigned f = 0; f < frames; f++) {
      double start = now();
      VK(QueueSubmit(queue, 1, &(VkSubmitInfo){
         .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &frame},
         VK_NULL_HANDLE));
      VK(QueueWaitIdle(queue));
      double t = now() - start;
      total_time += t;
      best = fmin(best, t);
      printf("frame %u: %.1f ms\n", f, t * 1e3);
   }
   printf("%s %ux%u: %u triangles, %u draws, mean %.1f ms (%.2f fps), best %.1f ms\n", argv[2], width,
          height, vertex_count / 3, draws, total_time / frames * 1e3, frames / total_time, best * 1e3);
   return 0;
}
