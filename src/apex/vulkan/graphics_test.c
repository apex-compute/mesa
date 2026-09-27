/* SPDX-License-Identifier: MIT */
/* Public-loader graphics gate: three depth-tested triangles with per-vertex
 * colors, one crossing the near plane, drawn through a render pass into
 * RGBA8 color and D32 depth. An independent reference repeats Vulkan's
 * viewport transform, Q16.8 snapping, top-left coverage, depth test and
 * perspective-correct interpolation for every pixel. */
#include <vulkan/vulkan.h>
#include <dlfcn.h>
#include <linux/dma-buf.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { \
   fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); \
} } while (0)
#define VK(x) CHECK((x) == VK_SUCCESS)

enum { WIDTH = 64, HEIGHT = 48, TRIANGLES = 3 };

/* Clip-space position then RGBA color per vertex. Triangle 0 crosses z = 0. */
static const float vertices[TRIANGLES * 3][8] = {
   {-0.9f, 0.5f, -0.5f, 1.0f, 1, 1, 0, 1}, {-0.5f, 0.95f, 0.6f, 1.0f, 0, 1, 1, 1},
   {0.0f, 0.3f, 0.6f, 1.0f, 1, 0, 1, 1},
   {-0.8f, -0.9f, 0.5f, 1.0f, 1, 0, 0, 1}, {0.7f, -0.5f, 0.5f, 1.0f, 0, 1, 0, 1},
   {-0.3f, 0.85f, 0.5f, 1.0f, 0, 0, 1, 1},
   {-0.4f, -1.2f, 0.5f, 2.0f, 1, 1, 1, 1}, {0.9f, 0.2f, 0.25f, 1.0f, 0, 1, 1, 1},
   {0.05f, 0.45f, 0.125f, 0.5f, 1, 0, 1, 1},
};

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

static int64_t
snap(double v)
{
   return (int64_t)nearbyint(v * 256.0);
}

struct reference {
   bool known;           /* false near clipped edges: either outcome is accepted */
   bool covered;
   /* Clip-generated vertices snap independently: the depth plane may shift. */
   double depth, depth_tolerance, color[4];
};

/* Independent per-pixel reference in double precision. */
static void
reference(struct reference *out)
{
   for (unsigned p = 0; p < WIDTH * HEIGHT; p++)
      out[p] = (struct reference){.known = true, .depth = 1.0, .depth_tolerance = 1e-5};
   for (unsigned t = 0; t < TRIANGLES; t++) {
      const float (*v)[8] = &vertices[t * 3];
      double sx[3], sy[3], sz[3], rw[3];
      int64_t qx[3], qy[3];
      for (unsigned i = 0; i < 3; i++) {
         rw[i] = 1.0 / v[i][3];
         sx[i] = v[i][0] * rw[i] * WIDTH / 2 + WIDTH / 2.0;
         sy[i] = v[i][1] * rw[i] * HEIGHT / 2 + HEIGHT / 2.0;
         sz[i] = v[i][2] * rw[i];
         qx[i] = snap(sx[i]);
         qy[i] = snap(sy[i]);
      }
      int64_t area = (qx[1] - qx[0]) * (qy[2] - qy[0]) - (qx[2] - qx[0]) * (qy[1] - qy[0]);
      bool clipped = sz[0] < 0 || sz[1] < 0 || sz[2] < 0;
      for (unsigned y = 0; y < HEIGHT; y++) {
         for (unsigned x = 0; x < WIDTH; x++) {
            int64_t px = x * 256 + 128, py = y * 256 + 128;
            int64_t e[3];
            bool inside = area != 0;
            for (unsigned i = 0; i < 3 && inside; i++) {
               unsigned a = area > 0 ? (i + 1) % 3 : (i + 2) % 3;
               unsigned b = area > 0 ? (i + 2) % 3 : (i + 1) % 3;
               int64_t dx = qx[b] - qx[a], dy = qy[b] - qy[a];
               e[i] = dx * (py - qy[a]) - dy * (px - qx[a]);
               bool top_left = dy < 0 || (dy == 0 && dx > 0);
               inside = e[i] > 0 || (e[i] == 0 && top_left);
            }
            if (!inside)
               continue;
            double sum = (double)(e[0] + e[1] + e[2]);
            double l[3] = {e[0] / sum, e[1] / sum, e[2] / sum};
            double z = l[0] * sz[0] + l[1] * sz[1] + l[2] * sz[2];
            struct reference *r = &out[y * WIDTH + x];
            if (clipped && fabs(z) < 0.02) {
               r->known = false;
               continue;
            }
            if (z < 0 || z > 1 || !(z < r->depth))
               continue;
            double w[3], s = 0;
            for (unsigned i = 0; i < 3; i++)
               s += w[i] = l[i] * rw[i];
            r->covered = true;
            r->depth = z;
            r->depth_tolerance = clipped ? 1e-3 : 1e-5;
            for (unsigned c = 0; c < 4; c++)
               r->color[c] = (w[0] * v[0][4 + c] + w[1] * v[1][4 + c] + w[2] * v[2][4 + c]) / s;
         }
      }
   }
}

int
main(int argc, char **argv)
{
   /* --export reads results through an exported dma-buf, as KMS scanout does. */
   CHECK(argc == 4 || (argc == 5 && !strcmp(argv[4], "--export")));
   const bool export = argc == 5;
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
   VK(CreateInstance(&(VkInstanceCreateInfo){
      .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .pApplicationInfo = &(VkApplicationInfo){
         .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_API_VERSION_1_0},
   }, NULL, &instance));
   GI(EnumeratePhysicalDevices); GI(GetPhysicalDeviceMemoryProperties); GI(CreateDevice);
   GI(GetDeviceProcAddr); GI(DestroyInstance);
   uint32_t count = 1;
   VkPhysicalDevice physical;
   CHECK(EnumeratePhysicalDevices(instance, &count, &physical) >= 0 && count == 1);
   VkPhysicalDeviceMemoryProperties memory;
   GetPhysicalDeviceMemoryProperties(physical, &memory);
   VkDevice device;
   const char *extensions[] = {VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
      VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME, VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME};
   VK(CreateDevice(physical, &(VkDeviceCreateInfo){
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &(VkDeviceQueueCreateInfo){
         .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueCount = 1,
         .pQueuePriorities = &(float){1.0f}},
      .enabledExtensionCount = export ? 3 : 0, .ppEnabledExtensionNames = extensions,
   }, NULL, &device));
#define GD(name) PFN_vk##name name = (PFN_vk##name)GetDeviceProcAddr(device, "vk" #name); CHECK(name)
   GD(GetDeviceQueue); GD(CreateBuffer); GD(GetBufferMemoryRequirements); GD(AllocateMemory);
   GD(BindBufferMemory); GD(MapMemory); GD(FlushMappedMemoryRanges); GD(InvalidateMappedMemoryRanges);
   GD(CreateImage); GD(GetImageMemoryRequirements); GD(BindImageMemory); GD(CreateImageView);
   GD(CreateShaderModule); GD(CreatePipelineLayout); GD(CreateGraphicsPipelines);
   GD(CreateRenderPass); GD(CreateFramebuffer); GD(CreateCommandPool); GD(AllocateCommandBuffers);
   GD(BeginCommandBuffer); GD(CmdBeginRenderPass); GD(CmdBindPipeline); GD(CmdBindVertexBuffers);
   GD(CmdDraw); GD(CmdEndRenderPass); GD(CmdPipelineBarrier); GD(CmdCopyImageToBuffer);
   GD(EndCommandBuffer); GD(QueueSubmit); GD(QueueWaitIdle); GD(DeviceWaitIdle); GD(DestroyDevice);
   VkQueue queue;
   GetDeviceQueue(device, 0, 0, &queue);

   uint32_t host_type = UINT32_MAX;
   for (uint32_t i = 0; i < memory.memoryTypeCount && host_type == UINT32_MAX; i++)
      if (memory.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
         host_type = i;
   CHECK(host_type != UINT32_MAX);
   /* One host-visible allocation: vertices at 0, color readback at 4 KiB,
    * depth readback after it; images follow at 64 KiB. */
   const VkDeviceSize color_offset = 4096, depth_offset = color_offset + WIDTH * HEIGHT * 4;
   const VkDeviceSize image_offset = 65536, total = 1024 * 1024;
   VkDeviceMemory allocation;
   VK(AllocateMemory(device, &(VkMemoryAllocateInfo){
      .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = total,
      .memoryTypeIndex = host_type}, NULL, &allocation));
   uint8_t *mapped;
   VK(MapMemory(device, allocation, 0, VK_WHOLE_SIZE, 0, (void **)&mapped));
   memset(mapped, 0xa5, total);
   memcpy(mapped, vertices, sizeof(vertices));
   VkMappedMemoryRange whole = {
      .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, .memory = allocation, .size = VK_WHOLE_SIZE};
   VK(FlushMappedMemoryRanges(device, 1, &whole));

   VkBuffer vertex_buffer, readback;
   VK(CreateBuffer(device, &(VkBufferCreateInfo){
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = sizeof(vertices),
      .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT}, NULL, &vertex_buffer));
   VK(BindBufferMemory(device, vertex_buffer, allocation, 0));
   VK(CreateBuffer(device, &(VkBufferCreateInfo){
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = WIDTH * HEIGHT * 8,
      .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT}, NULL, &readback));
   uint8_t *exported = NULL;
   int export_fd = -1;
   if (export) {
      GD(GetMemoryFdKHR);
      uint32_t device_type = UINT32_MAX;
      for (uint32_t i = 0; i < memory.memoryTypeCount && device_type == UINT32_MAX; i++)
         if (memory.memoryTypes[i].propertyFlags == VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
            device_type = i;
      CHECK(device_type != UINT32_MAX);
      VkDeviceMemory shared;
      VK(AllocateMemory(device, &(VkMemoryAllocateInfo){
         .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = WIDTH * HEIGHT * 8,
         .memoryTypeIndex = device_type,
         .pNext = &(VkExportMemoryAllocateInfo){.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
            .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT}}, NULL, &shared));
      VK(GetMemoryFdKHR(device, &(VkMemoryGetFdInfoKHR){
         .sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR, .memory = shared,
         .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT}, &export_fd));
      exported = mmap(NULL, WIDTH * HEIGHT * 8, PROT_READ, MAP_SHARED, export_fd, 0);
      CHECK(exported != MAP_FAILED);
      VkBuffer external;
      VK(CreateBuffer(device, &(VkBufferCreateInfo){
         .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = WIDTH * HEIGHT * 8,
         .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
         .pNext = &(VkExternalMemoryBufferCreateInfo){
            .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
            .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT}}, NULL, &external));
      VK(BindBufferMemory(device, external, shared, 0));
      readback = external;
   } else {
      VK(BindBufferMemory(device, readback, allocation, color_offset));
   }

   const VkFormat formats[2] = {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_D32_SFLOAT};
   VkImage images[2];
   VkImageView views[2];
   VkDeviceSize offset = image_offset;
   for (unsigned i = 0; i < 2; i++) {
      VK(CreateImage(device, &(VkImageCreateInfo){
         .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
         .format = formats[i], .extent = {WIDTH, HEIGHT, 1}, .mipLevels = 1, .arrayLayers = 1,
         .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
         .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | (i ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT :
                                                        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)},
         NULL, &images[i]));
      VkMemoryRequirements requirements;
      GetImageMemoryRequirements(device, images[i], &requirements);
      CHECK(requirements.memoryTypeBits & (1u << host_type));
      offset = (offset + requirements.alignment - 1) & ~(requirements.alignment - 1);
      VK(BindImageMemory(device, images[i], allocation, offset));
      offset += requirements.size;
      VK(CreateImageView(device, &(VkImageViewCreateInfo){
         .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = images[i],
         .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = formats[i],
         .subresourceRange = {i ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}},
         NULL, &views[i]));
   }
   CHECK(offset <= total);

   VkRenderPass pass;
   VK(CreateRenderPass(device, &(VkRenderPassCreateInfo){
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO, .attachmentCount = 2,
      .pAttachments = (VkAttachmentDescription[]){
         {.format = formats[0], .samples = VK_SAMPLE_COUNT_1_BIT,
          .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
          .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE, .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
          .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED, .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL},
         {.format = formats[1], .samples = VK_SAMPLE_COUNT_1_BIT,
          .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
          .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE, .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
          .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED, .finalLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL}},
      .subpassCount = 1,
      .pSubpasses = &(VkSubpassDescription){
         .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS, .colorAttachmentCount = 1,
         .pColorAttachments = &(VkAttachmentReference){0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL},
         .pDepthStencilAttachment = &(VkAttachmentReference){
            1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL}},
   }, NULL, &pass));
   VkFramebuffer framebuffer;
   VK(CreateFramebuffer(device, &(VkFramebufferCreateInfo){
      .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO, .renderPass = pass, .attachmentCount = 2,
      .pAttachments = views, .width = WIDTH, .height = HEIGHT, .layers = 1}, NULL, &framebuffer));

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
         .pVertexBindingDescriptions = &(VkVertexInputBindingDescription){
            0, sizeof(vertices[0]), VK_VERTEX_INPUT_RATE_VERTEX},
         .vertexAttributeDescriptionCount = 2,
         .pVertexAttributeDescriptions = (VkVertexInputAttributeDescription[]){
            {0, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 0}, {1, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 16}}},
      .pInputAssemblyState = &(VkPipelineInputAssemblyStateCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
         .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST},
      .pViewportState = &(VkPipelineViewportStateCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1,
         .pViewports = &(VkViewport){0, 0, WIDTH, HEIGHT, 0, 1}, .scissorCount = 1,
         .pScissors = &(VkRect2D){{0, 0}, {WIDTH, HEIGHT}}},
      .pRasterizationState = &(VkPipelineRasterizationStateCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
         .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE, .lineWidth = 1},
      .pMultisampleState = &(VkPipelineMultisampleStateCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
         .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT},
      .pDepthStencilState = &(VkPipelineDepthStencilStateCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
         .depthTestEnable = VK_TRUE, .depthWriteEnable = VK_TRUE, .depthCompareOp = VK_COMPARE_OP_LESS},
      .pColorBlendState = &(VkPipelineColorBlendStateCreateInfo){
         .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO, .attachmentCount = 1,
         .pAttachments = &(VkPipelineColorBlendAttachmentState){.colorWriteMask = 0xf}},
      .layout = layout, .renderPass = pass,
   }, NULL, &pipeline));

   VkCommandPool command_pool;
   VK(CreateCommandPool(device, &(VkCommandPoolCreateInfo){
      .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}, NULL, &command_pool));
   VkCommandBuffer cmd;
   VK(AllocateCommandBuffers(device, &(VkCommandBufferAllocateInfo){
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = command_pool,
      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1}, &cmd));
   VK(BeginCommandBuffer(cmd, &(VkCommandBufferBeginInfo){
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO}));
   CmdBeginRenderPass(cmd, &(VkRenderPassBeginInfo){
      .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO, .renderPass = pass, .framebuffer = framebuffer,
      .renderArea = {{0, 0}, {WIDTH, HEIGHT}}, .clearValueCount = 2,
      .pClearValues = (VkClearValue[]){{.color = {.float32 = {0, 0, 0.5f, 1}}},
                                       {.depthStencil = {1.0f, 0}}}}, VK_SUBPASS_CONTENTS_INLINE);
   CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
   CmdBindVertexBuffers(cmd, 0, 1, &vertex_buffer, &(VkDeviceSize){0});
   CmdDraw(cmd, TRIANGLES * 3, 1, 0, 0);
   CmdEndRenderPass(cmd);
   for (unsigned i = 0; i < 2; i++)
      CmdCopyImageToBuffer(cmd, images[i], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback, 1,
         &(VkBufferImageCopy){.bufferOffset = i ? depth_offset - color_offset : 0,
            .imageSubresource = {i ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .imageExtent = {WIDTH, HEIGHT, 1}});
   VK(EndCommandBuffer(cmd));
   VK(QueueSubmit(queue, 1, &(VkSubmitInfo){
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cmd},
      VK_NULL_HANDLE));
   VK(QueueWaitIdle(queue));
   VK(InvalidateMappedMemoryRanges(device, 1, &whole));
   if (export) {
      struct dma_buf_sync sync = {.flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ};
      CHECK(!ioctl(export_fd, DMA_BUF_IOCTL_SYNC, &sync));
      memcpy(mapped + color_offset, exported, WIDTH * HEIGHT * 8);
   }

   struct reference *expected = calloc(WIDTH * HEIGHT, sizeof(*expected));
   CHECK(expected);
   reference(expected);
   const uint32_t *color = (const void *)(mapped + color_offset);
   const float *depth = (const void *)(mapped + depth_offset);
   unsigned covered = 0, unknown = 0, failures = 0;
   for (unsigned p = 0; p < WIDTH * HEIGHT; p++) {
      const struct reference *r = &expected[p];
      if (!r->known) {
         unknown++;
         continue;
      }
      double want[4] = {0, 0, 0.5, 1};
      if (r->covered) {
         covered++;
         memcpy(want, r->color, sizeof(want));
      }
      bool ok = fabs(depth[p] - r->depth) <= r->depth_tolerance;
      for (unsigned c = 0; c < 4; c++)
         ok &= abs((int)((color[p] >> (8 * c)) & 0xff) - (int)lround(want[c] * 255)) <= 2;
      if (!ok && failures++ < 8)
         fprintf(stderr, "pixel (%u,%u): color 0x%08x depth %.6f, want (%.3f %.3f %.3f %.3f) depth %.6f%s\n",
                 p % WIDTH, p / WIDTH, color[p], depth[p], want[0], want[1], want[2], want[3],
                 r->depth, r->covered ? "" : " (clear)");
   }
   for (VkDeviceSize i = depth_offset + WIDTH * HEIGHT * 4; i < image_offset; i++)
      if (mapped[i] != 0xa5 && failures++ < 8)
         fprintf(stderr, "guard byte %lu changed\n", (unsigned long)i);
   if (failures) {
      fprintf(stderr, "FAIL Apex graphics: %u mismatches\n", failures);
      return 1;
   }
   printf("PASS Apex graphics: %ux%u RGBA8/D32 render pass, 3 depth-tested triangles, "
          "%u covered and %u clear pixels exact, %u near-clip edge pixels unconstrained\n",
          WIDTH, HEIGHT, covered, WIDTH * HEIGHT - covered - unknown, unknown);
   VK(DeviceWaitIdle(device));
   DestroyDevice(device, NULL);
   DestroyInstance(instance, NULL);
   return 0;
}
