/* SPDX-License-Identifier: MIT */
#include "apex_device.h"
#include "apex_pipeline.h"
#include "apex_native_uapi.h"
#include "compiler/spirv/spirv.h"
#include "drm-uapi/apex_drm.h"
#include "vk_alloc.h"
#include "vk_drm_syncobj.h"
#include "vk_instance.h"
#include "vk_physical_device.h"
#include <errno.h>
#include <fcntl.h>
#include <grp.h>
#include <stdarg.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { \
   fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); abort(); \
} } while (0)

/* Link-time interception tests the transport, not shader execution. A real
 * fd bypasses it and the same Vulkan commands/results run on the device. */
static struct {
   int fault;
   unsigned calls[11], step, dispatch;
   int drm_fd;
   unsigned objects, live, exec_calls;
   bool images;
   uint64_t next_offset;
   struct {
      uint64_t offset, size, va;
      uint32_t flags, uploads;
      uint8_t *local;
      bool live;
   } gems[128];
   struct apex_pipeline *pipelines[3];
   uint32_t *mapped;
   uint32_t expected[2048], payload[800];
} mock;
static const unsigned starts[] = {48, 1088, 48, 48};
static const unsigned biases[] = {37, 101, 112, 112};
static const unsigned scales[] = {3, 7, 7, 7};
static unsigned invocations[] = {16, 16, 12, 12};
static int sync_fd = -1;

int __real_ioctl(int fd, unsigned long request, ...);
int __wrap_ioctl(int fd, unsigned long request, ...);

static int
image_exec(struct drm_apex_vm_exec *r)
{
   unsigned table, program;
   for (table = 1; table <= mock.objects && mock.gems[table].va != r->data_va; table++);
   for (program = 1; program <= mock.objects && mock.gems[program].va != r->program_va; program++);
   CHECK(table <= mock.objects && program <= mock.objects);
   CHECK(mock.gems[program].flags == (APEX_DRM_VM_READ | APEX_DRM_VM_EXEC));
   CHECK(mock.gems[table].uploads == 1 && mock.gems[program].uploads == 1);
   const uint32_t *words = (void *)mock.gems[table].local;
   unsigned d = mock.dispatch++;
   uint64_t base = mock.gems[1].va;
   if (d == 0 || d == 6) {
      CHECK(r->program_bytes == mock.pipelines[0]->program.code.size && r->workgroups == 1);
      /* The descriptor update after recording swaps views: mip0/layer2 first,
       * mip1/layer1 second. Offsets include the nonzero image memory binding. */
      const uint64_t addresses[] = {base + 4224, base + 1152, base + 2240};
      const uint32_t tails[][6] = {{768, 0, 0, 0, 0, 0}, {11, 7, 2, 64, 448, 0}, {5, 3, 2, 64, 192, 0}};
      for (unsigned row = 0; row < 3; row++) {
         CHECK(((uint64_t)util_le32_to_cpu(words[row * 8 + 1]) << 32 |
                util_le32_to_cpu(words[row * 8])) == addresses[row]);
         for (unsigned i = 0; i < 6; i++) CHECK(util_le32_to_cpu(words[row * 8 + 2 + i]) == tails[row][i]);
      }
      for (unsigned i = 24; i < 32; i++) CHECK(!words[i]);
      for (unsigned i = 0; i < 4; i++) CHECK(words[32 + i] == 0xc0010000 + i);
   } else if (d == 1 || d == 5) {
      /* Internal clear job: texels only, row padding untouched. */
      const uint32_t *clear = words + 8 + 8;
      unsigned offset = d == 1 ? 9280 : 14592;
      uint32_t color = d == 1 ? 0x5a17c0de : 0xff80ff00;
      CHECK(((uint64_t)clear[1] << 32 | clear[0]) == base + offset);
      CHECK(clear[7] == 4 && clear[8] == color);
      CHECK(d == 1 ? clear[4] == 4 && clear[5] == 2 && clear[6] == 2 && clear[2] == 64 && clear[3] == 128 :
                     clear[4] == 2 && clear[5] == 2 && clear[6] == 1 && clear[2] == 64);
      for (unsigned layer = 0; layer < clear[6]; layer++)
         for (unsigned y = 0; y < clear[5]; y++)
            for (unsigned x = 0; x < clear[4]; x++)
               ((uint32_t *)mock.gems[1].local)[(offset + layer * clear[3] + y * clear[2]) / 4 + x] = color;
   } else {
      /* One internal pitched-copy job per region covers both layers and rows. */
      CHECK(d >= 2 && d <= 4);
      unsigned phase = d - 2;
      const unsigned from[] = {12316, 2308, 8648}, to[] = {2308, 8648, 12800};
      const unsigned src_row[] = {20, 64, 64}, dst_row[] = {64, 64, 28};
      const unsigned src_slice[] = {60, 192, 320}, dst_slice[] = {192, 320, 112};
      const uint32_t *copy = words + 8 + 8;
      CHECK(((uint64_t)copy[1] << 32 | copy[0]) == base + from[phase]);
      CHECK(((uint64_t)copy[3] << 32 | copy[2]) == base + to[phase]);
      CHECK(copy[4] == src_row[phase] && copy[5] == dst_row[phase]);
      CHECK(copy[6] == src_slice[phase] && copy[7] == dst_slice[phase]);
      CHECK(copy[8] == 3 && copy[9] == 2 && copy[10] == 2);
      /* Model the copy kernel only; this is transport evidence. */
      for (unsigned layer = 0; layer < 2; layer++)
         for (unsigned y = 0; y < 2; y++)
            memcpy(mock.gems[1].local + to[phase] + layer * dst_slice[phase] + y * dst_row[phase],
                   mock.gems[1].local + from[phase] + layer * src_slice[phase] + y * src_row[phase], 12);
   }
   r->status = 1;
   return 0;
}

static int
drm_ioctl(unsigned long request, void *arg)
{
   if (request == DRM_IOCTL_APEX_INFO) {
      *(struct drm_apex_info *)arg = (struct drm_apex_info) {
         .version = mock.fault == 208 ? 1 : 2,
         .capabilities = APEX_DRM_CAP_SHMEM | APEX_DRM_CAP_EXEC |
                         (mock.fault == 207 ? 0 : APEX_DRM_CAP_GPUVM),
         .max_buffer_bytes = 64 * 1024 * 1024,
      };
   } else if (request == DRM_IOCTL_APEX_GEM_CREATE) {
      struct drm_apex_gem_create *r = arg;
      CHECK(!r->flags && !r->handle && r->size);
      if (mock.objects && mock.fault == 201) { errno = ENOMEM; return -1; }
      if (mock.objects == 2 && mock.fault == 213) { errno = ENOMEM; return -1; }
      unsigned h = ++mock.objects;
      CHECK(h < ARRAY_SIZE(mock.gems));
      r->size = align64(r->size, 4096);
      r->handle = h;
      mock.gems[h].size = r->size;
      mock.gems[h].offset = mock.next_offset;
      mock.gems[h].live = true;
      mock.gems[h].local = calloc(1, r->size);
      CHECK(mock.gems[h].local);
      mock.live++;
      mock.next_offset += r->size + 4096;
      CHECK(!ftruncate(mock.drm_fd, mock.next_offset));
   } else if (request == DRM_IOCTL_APEX_GEM_MMAP) {
      struct drm_apex_gem_mmap *r = arg;
      CHECK(!r->flags && r->handle <= mock.objects && mock.gems[r->handle].live);
      if (r->handle > 1 && mock.fault == 202) { errno = EIO; return -1; }
      r->offset = r->handle > 1 && mock.fault == 203 ? 1 : mock.gems[r->handle].offset;
   } else if (request == DRM_IOCTL_APEX_VM_BIND) {
      struct drm_apex_vm_bind *r = arg;
      CHECK(!r->pad && !r->offset && r->bytes && !(r->bytes % 4096));
      CHECK(r->va >= 2 * 1024 * 1024 && !(r->va % 4096) &&
            r->va + r->bytes <= (1ull << 39));
      if (r->operation == APEX_DRM_VM_BIND_MAP) {
         unsigned h = r->handle;
         CHECK(h && h <= mock.objects && mock.gems[h].live && !mock.gems[h].va);
         CHECK(r->bytes == mock.gems[h].size);
         if (!mock.images)
            CHECK(r->flags == (APEX_DRM_VM_READ | (h == 2 || h == 4 || h == 6 ? APEX_DRM_VM_EXEC : APEX_DRM_VM_WRITE)));
         if (h > 1 && mock.fault == 209) { errno = ENOMEM; return -1; }
         if (h == 3 && mock.fault == 211) { errno = ENOMEM; return -1; }
         for (unsigned i = 1; i <= mock.objects; i++)
            CHECK(!mock.gems[i].va || r->va + r->bytes <= mock.gems[i].va ||
                  r->va >= mock.gems[i].va + mock.gems[i].size);
         mock.gems[h].va = r->va;
         mock.gems[h].flags = r->flags;
      } else {
         CHECK(r->operation == APEX_DRM_VM_BIND_UNMAP && !r->flags && !r->handle);
         unsigned h;
         for (h = 1; h <= mock.objects && mock.gems[h].va != r->va; h++);
         CHECK(h <= mock.objects && mock.gems[h].size == r->bytes);
         mock.gems[h].va = 0;
      }
   } else if (request == DRM_IOCTL_APEX_GEM_TRANSFER) {
      struct drm_apex_gem_transfer *r = arg;
      unsigned h = r->handle;
      CHECK(h && h <= mock.objects && mock.gems[h].live && !r->flags && !r->pad);
      CHECK(r->bytes && r->offset <= mock.gems[h].size &&
            r->bytes <= mock.gems[h].size - r->offset);
      if (h > 1 && mock.fault == 210) { errno = EIO; return -1; }
      if (h == 3 && mock.fault == 212) { errno = EIO; return -1; }
      void *local = mock.gems[h].local + r->offset;
      off_t offset = mock.gems[h].offset + r->offset;
      if (r->direction == APEX_DRM_TRANSFER_TO_LOCAL) {
         mock.gems[h].uploads++;
         CHECK(pread(mock.drm_fd, local, r->bytes, offset) == r->bytes);
      } else {
         CHECK(r->direction == APEX_DRM_TRANSFER_FROM_LOCAL && h == 1);
         CHECK(pwrite(mock.drm_fd, local, r->bytes, offset) == r->bytes);
      }
   } else if (request == DRM_IOCTL_GEM_CLOSE) {
      struct drm_gem_close *r = arg;
      CHECK(r->handle <= mock.objects && mock.gems[r->handle].live);
      CHECK(!mock.gems[r->handle].va);
      free(mock.gems[r->handle].local);
      mock.gems[r->handle].live = false;
      mock.live--;
   } else {
      CHECK(request == DRM_IOCTL_APEX_VM_EXEC);
      struct drm_apex_vm_exec *r = arg;
      if (mock.images)
         return image_exec(r);
      unsigned d = mock.dispatch;
      unsigned p = MIN2(d, 2), h = 2 * p + 2;
      mock.exec_calls++;
      CHECK(d < 4 && r->program_va == mock.gems[h].va);
      CHECK(mock.gems[h].live && mock.gems[1].live);
      CHECK(mock.gems[h].uploads == 1 && mock.gems[1].uploads == 2);
      CHECK(!r->flags && !r->status && !r->reason && !r->timestamp);
      CHECK(r->program_bytes == mock.pipelines[p]->program.code.size);
      unsigned table = d < 3 ? h + 1 : 8;
      CHECK(r->data_va == mock.gems[table].va && mock.gems[table].live);
      CHECK(mock.gems[table].uploads == 1);
      const uint32_t *rows = (const void *)mock.gems[table].local;
      for (unsigned i = 0; i < 8; i++) CHECK(!rows[i]); /* unused binding 0 */
      const uint32_t *row = rows + 8; /* shader binding 7, element 0 */
      uint64_t va = (uint64_t)util_le32_to_cpu(row[1]) << 32 | util_le32_to_cpu(row[0]);
      CHECK(va == mock.gems[1].va + starts[d] * 4);
      CHECK(util_le32_to_cpu(row[2]) == sizeof(mock.payload));
      for (unsigned i = 3; i < 8; i++) CHECK(!row[i]);
      const uint32_t *extra = rows + 16;
      uint64_t addr = (uint64_t)util_le32_to_cpu(extra[1]) << 32 | util_le32_to_cpu(extra[0]);
      CHECK(addr == mock.gems[1].va + (d == 1 ? 48 : 1088) * 4);
      CHECK(util_le32_to_cpu(extra[2]) == sizeof(mock.payload));
      for (unsigned i = 3; i < 8; i++) CHECK(!extra[i]);
      for (unsigned i = 24; i < 56; i++) CHECK(!rows[i]); /* unused set + sentinel */
      CHECK(mock.pipelines[p]->program.push_size == 256);
      for (unsigned i = 0; i < 64; i++) {
         uint32_t expected = i < 4 ? 0 : 0xa5100000 + i * 37;
         if (d == 1 && i == 6) expected = 0xc0ffee00;
         if (d == 1 && i == 7) expected = 0xabad1dea;
         if (d >= 2) expected = i == 63 ? 0xdecafbad : 0;
         CHECK(rows[56 + i] == expected);
      }
      CHECK(r->workgroups == 1);
      CHECK(!memcmp(mock.gems[h].local, mock.pipelines[p]->program.code.data, r->program_bytes));
      void *local = mock.gems[1].local + starts[d] * 4;
      memcpy(mock.payload, local, sizeof(mock.payload));
      CHECK(!memcmp(mock.payload, mock.expected + starts[d], sizeof(mock.payload)));
      if (mock.fault == 204) { errno = EINTR; return -1; }
      if (mock.fault == 205 || mock.fault == 206) {
         r->status = mock.fault == 205 ? 4 : 5;
         return 0;
      }
      for (unsigned i = 0; i < invocations[d]; i++)
         mock.payload[i] = biases[d] + scales[d] * i;
      memcpy(local, mock.payload, sizeof(mock.payload));
      memcpy(mock.expected + starts[d], mock.payload, sizeof(mock.payload));
      r->status = 1;
      mock.dispatch++;
   }
   return 0;
}

int
__wrap_ioctl(int fd, unsigned long request, ...)
{
   va_list args;
   va_start(args, request);
   struct apex_ioctl_native *r = va_arg(args, void *);
   va_end(args);
   if (fd >= 0 && fd == mock.drm_fd)
      return drm_ioctl(request, r);
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
      CHECK(r->kind == APEX_NATIVE_PROGRAM && r->bytes == mock.pipelines[d]->program.code.size);
      CHECK(!memcmp((void *)(uintptr_t)r->user_ptr, mock.pipelines[d]->program.code.data, r->bytes));
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
      for (unsigned i = 0; i < invocations[d]; i++)
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
run(struct vk_physical_device *physical, const uint32_t *spirv, size_t size, int fd, int fault,
    enum apex_transport transport, bool dynamic, bool secondary)
{
   memset(&mock, 0, sizeof(mock));
   mock.fault = fault;
   mock.drm_fd = -1;
   bool mocked = fd < 0;
   if (mocked && transport == APEX_TRANSPORT_DRM) {
      mock.drm_fd = fd = memfd_create("apex-drm-test", MFD_CLOEXEC);
      CHECK(fd >= 0);
      mock.next_offset = 4096;
   }
   struct apex_device device;
   const float priority = 1.0f;
   const VkDeviceQueueCreateInfo queue_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueCount = 1, .pQueuePriorities = &priority,
   };
   VkPhysicalDeviceTimelineSemaphoreFeatures timeline_feature = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES,
      .timelineSemaphore = sync_fd >= 0,
   };
   const VkPhysicalDeviceDescriptorIndexingFeatures features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES,
      .pNext = &timeline_feature,
      .descriptorBindingStorageBufferUpdateAfterBind = VK_TRUE,
   };
   const VkDeviceCreateInfo device_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .pNext = &features,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &queue_info,
   };
   VkResult initialized = apex_device_init(&device, physical, &device_info, NULL, fd, transport);
   if (fault == 207 || fault == 208) {
      CHECK(initialized == VK_ERROR_INCOMPATIBLE_DRIVER && !mock.objects && !mock.exec_calls);
      CHECK(!close(fd));
      return;
   }
   CHECK(initialized == VK_SUCCESS);
   if (sync_fd >= 0 && (mocked || transport == APEX_TRANSPORT_NATIVE)) {
      if (device.vk.sync) device.vk.sync->finalize(device.vk.sync);
      vk_device_set_drm_fd(&device.vk, sync_fd);
   }
   VkDevice dev = apex_device_to_handle(&device);
   const struct vk_device_dispatch_table *v = &device.vk.dispatch_table;
   VkQueue queue;
   v->GetDeviceQueue(dev, 0, 0, &queue);
   CHECK(queue == vk_queue_to_handle(&device.queue));
   VkShaderModule module;
   bool tables = transport == APEX_TRANSPORT_DRM;
   uint32_t *code = malloc(size);
   CHECK(code);
   memcpy(code, spirv, size);
   if (invocations[0] != 16) {
      unsigned changed = 0;
      for (unsigned pos = 5; pos < size / 4; pos += code[pos] >> 16) {
         CHECK(code[pos] >> 16);
         if ((code[pos] & 0xffff) == SpvOpExecutionMode &&
             code[pos+2] == SpvExecutionModeLocalSize && code[pos+3] == 16) {
            CHECK(code[pos+4] == 1 && code[pos+5] == 1);
            code[pos+3] = invocations[0];
            changed++;
         }
      }
      CHECK(changed == 1);
   }
   if (tables) {
      unsigned changed = 0;
      for (unsigned pos = 5; pos < size / 4; pos += code[pos] >> 16) {
         CHECK(code[pos] >> 16);
         if ((code[pos] & 0xffff) == SpvOpDecorate && code[pos+2] == SpvDecorationBinding) {
            CHECK(code[pos+3] == 0);
            code[pos+3] = 7;
            changed++;
         }
      }
      CHECK(changed == 1);
   }
   const VkShaderModuleCreateInfo module_info = {
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = size, .pCode = code,
   };
   CHECK(v->CreateShaderModule(dev, &module_info, NULL, &module) == VK_SUCCESS);
   free(code);
   VkDescriptorSetLayoutBinding binding[] = {
      {.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1,
       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
      {.binding = 7, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 2,
       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
   };
   if (dynamic) {
      binding[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
      binding[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
      VkDescriptorSetLayoutBinding tmp = binding[0];
      binding[0] = binding[1];
      binding[1] = tmp; /* Input order must not determine dynamic-offset order. */
   }
   VkDescriptorSetLayout set_layout;
   const VkDescriptorBindingFlags binding_flags[] = {
      VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT, VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT,
   };
   const VkDescriptorSetLayoutBindingFlagsCreateInfo binding_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
      .bindingCount = tables ? 2 : 1, .pBindingFlags = binding_flags,
   };
   VkDescriptorSetLayoutCreateInfo set_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
      .pNext = &binding_info, .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
      .bindingCount = tables ? 2 : 1, .pBindings = binding,
   };
   if (dynamic) {
      CHECK(v->CreateDescriptorSetLayout(dev, &set_info, NULL, &set_layout) == VK_ERROR_FEATURE_NOT_PRESENT);
      set_info.pNext = NULL;
      set_info.flags = 0;
   }
   CHECK(v->CreateDescriptorSetLayout(dev, &set_info, NULL, &set_layout) == VK_SUCCESS);
   VkDescriptorSetLayout layouts[2] = {set_layout, set_layout};
   VkPipelineLayout layout;
   const VkPushConstantRange push_range = {
      VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 16, 240,
   };
   const VkPipelineLayoutCreateInfo layout_info = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = tables ? 2 : 1, .pSetLayouts = layouts,
      .pushConstantRangeCount = tables ? 1 : 0, .pPushConstantRanges = &push_range,
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
      if (i < 2 && invocations[0] != 16)
         CHECK(!memcmp(mock.pipelines[i]->program.code.data, "APX2", 4));
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
   const VkMappedMemoryRange whole = {
      .sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, .memory = memory, .size = VK_WHOLE_SIZE,
   };
   CHECK(v->FlushMappedMemoryRanges(dev, 1, &whole) == VK_SUCCESS);
   /* Nonzero-offset WHOLE_SIZE must transfer only the remaining tail. */
   VkMappedMemoryRange range = whole;
   range.offset = sizeof(mock.expected) - 64;
   mock.mapped[2047] = mock.expected[2047] = 0x12345678;
   mock.mapped[100] = 0xbad; /* Outside this flush must remain unchanged in LOCAL. */
   CHECK(v->FlushMappedMemoryRanges(dev, 1, &range) == VK_SUCCESS);
   mock.mapped[100] = mock.expected[100];
   range.offset = sizeof(mock.expected);
   CHECK(v->FlushMappedMemoryRanges(dev, 1, &range) == VK_ERROR_MEMORY_MAP_FAILED);
   range.offset = 0;
   range.size = 0;
   CHECK(v->InvalidateMappedMemoryRanges(dev, 1, &range) == VK_ERROR_MEMORY_MAP_FAILED);
   range.size = sizeof(mock.expected) + 64;
   CHECK(v->FlushMappedMemoryRanges(dev, 1, &range) == VK_ERROR_MEMORY_MAP_FAILED);
   VkBuffer buffers[2];
   const VkBufferCreateInfo buffer_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 3456,
      .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
   };
   for (unsigned i = 0; i < 2; i++) {
      CHECK(v->CreateBuffer(dev, &buffer_info, NULL, &buffers[i]) == VK_SUCCESS);
      VkMemoryRequirements req;
      v->GetBufferMemoryRequirements(dev, buffers[i], &req);
      CHECK(req.size == 3456 && req.alignment == 64 && (req.memoryTypeBits & 1));
      CHECK(v->BindBufferMemory(dev, buffers[i], memory, i ? 4096 : 64) == VK_SUCCESS);
   }
   const VkDescriptorPoolSize pool_sizes[] = {
      {dynamic ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
       dynamic ? 4 : tables ? 6 : 2},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 2},
   };
   const VkDescriptorPoolCreateInfo pool_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT | VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT,
      .maxSets = 3, .poolSizeCount = dynamic ? 2 : 1, .pPoolSizes = pool_sizes,
   };
   VkDescriptorPool pool;
   CHECK(v->CreateDescriptorPool(dev, &pool_info, NULL, &pool) == VK_SUCCESS);
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
   if (dynamic) {
      bindings[0].offset = 64;
      /* The second descriptor retains WHOLE_SIZE and requires offset zero. */
      if (fault == 214) bindings[0].range = VK_WHOLE_SIZE;
   }
   if (fault == 100) bindings[0].range = 3329;
   if (fault == 101) bindings[0].offset = 3456;
   const VkDescriptorBufferInfo unused_uniform = {buffers[0], 0, 64};
   VkWriteDescriptorSet writes[2] = {{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = sets[0],
      .descriptorCount = 1,
      .descriptorType = dynamic ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = dynamic ? &unused_uniform : &bindings[1],
   }};
   writes[1] = writes[0];
   writes[1].dstSet = sets[1];
   v->UpdateDescriptorSets(dev, 2, writes, 0, NULL);
   if (tables) {
      VkDescriptorBufferInfo array[] = {bindings[1], bindings[0]};
      VkWriteDescriptorSet extra = writes[0];
      extra.dstBinding = 7;
      extra.descriptorType = dynamic ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      extra.descriptorCount = 2;
      extra.pBufferInfo = array;
      for (unsigned s = 0; s < 2; s++) {
         extra.dstSet = sets[s];
         v->UpdateDescriptorSets(dev, 1, &extra, 0, NULL);
      }
      /* Array element writes and copies must preserve their neighbors. */
      extra.dstSet = sets[0];
      extra.dstArrayElement = 1;
      extra.descriptorCount = 1;
      extra.pBufferInfo = &bindings[0];
      v->UpdateDescriptorSets(dev, 1, &extra, 0, NULL);
      VkCopyDescriptorSet copy = {
         .sType = VK_STRUCTURE_TYPE_COPY_DESCRIPTOR_SET,
         .srcSet = sets[1], .srcBinding = 7,
         .dstSet = sets[0], .dstBinding = 7, .dstArrayElement = 1, .descriptorCount = 1,
      };
      v->UpdateDescriptorSets(dev, 0, NULL, 1, &copy);
   }
   if (dynamic) {
      writes[0].dstBinding = 7;
      writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC;
      writes[0].pBufferInfo = &bindings[0];
      v->UpdateDescriptorSets(dev, 1, writes, 0, NULL);
   }
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
   if (tables) {
      uint32_t values[60];
      for (unsigned i = 0; i < ARRAY_SIZE(values); i++) values[i] = 0xa5100000 + (i + 4) * 37;
      const uint32_t bad_ranges[][2] = {{252, 8}, {UINT32_MAX - 3, 4}, {17, 4}, {16, 5}};
      for (unsigned i = 0; i < ARRAY_SIZE(bad_ranges); i++) {
         v->CmdPushConstants(cmd, layout, push_range.stageFlags,
                              bad_ranges[i][0], bad_ranges[i][1], values);
         CHECK(v->EndCommandBuffer(cmd) == VK_ERROR_FEATURE_NOT_PRESENT);
         CHECK(v->ResetCommandBuffer(cmd, 0) == VK_SUCCESS);
         CHECK(v->BeginCommandBuffer(cmd, &begin) == VK_SUCCESS);
      }
   }
   if (dynamic) {
      uint32_t offsets[] = {4, 64, 0, 12, 0, 64};
      v->CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 2, sets, 5, offsets);
      CHECK(v->EndCommandBuffer(cmd) == VK_ERROR_FEATURE_NOT_PRESENT);
      CHECK(v->ResetCommandBuffer(cmd, 0) == VK_SUCCESS);
      CHECK(v->BeginCommandBuffer(cmd, &begin) == VK_SUCCESS);
   }
   VkCommandBuffer primary = cmd, secondary_cmd = VK_NULL_HANDLE;
   const VkCommandBufferInheritanceInfo inheritance = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO,
   };
   const VkCommandBufferBeginInfo secondary_begin = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .pInheritanceInfo = &inheritance,
   };
   if (secondary) {
      VkCommandBufferAllocateInfo alloc = command_alloc;
      alloc.level = VK_COMMAND_BUFFER_LEVEL_SECONDARY;
      CHECK(v->AllocateCommandBuffers(dev, &alloc, &secondary_cmd) == VK_SUCCESS);
      cmd = secondary_cmd;
      CHECK(v->BeginCommandBuffer(cmd, &secondary_begin) == VK_SUCCESS);
   }
   if (dynamic) {
      uint32_t offsets[] = {4, 64, 0, 12, 0, 64};
      if (fault == 215) offsets[1] = 196; /* Range ends four bytes beyond the buffer. */
      if (fault == 216) offsets[1] = UINT32_MAX - 3;
      if (fault == 217) offsets[1] = 65;
      const VkBindDescriptorSetsInfo bind = {
         .sType = VK_STRUCTURE_TYPE_BIND_DESCRIPTOR_SETS_INFO, .layout = layout,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .firstSet = 0,
         .descriptorSetCount = 2, .pDescriptorSets = sets,
         .dynamicOffsetCount = 6, .pDynamicOffsets = offsets,
      };
      v->CmdBindDescriptorSets2(cmd, &bind);
      memset(offsets, 0xff, sizeof(offsets)); /* Consumed at bind, before dispatch. */
   } else if (tables)
      v->CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 1, 1, &sets[1], 0, NULL);
   if (tables) {
      uint32_t values[60];
      for (unsigned i = 0; i < ARRAY_SIZE(values); i++) values[i] = 0xa5100000 + (i + 4) * 37;
      v->CmdPushConstants(cmd, layout, push_range.stageFlags, 16, sizeof(values), values);
      memset(values, 0xff, sizeof(values)); /* Consume before dispatch, not submission. */
      VkPushConstantRange ignored_range = {VK_SHADER_STAGE_FRAGMENT_BIT, 16, 240};
      VkPipelineLayoutCreateInfo ignored_info = layout_info;
      ignored_info.pPushConstantRanges = &ignored_range;
      VkPipelineLayout ignored;
      CHECK(v->CreatePipelineLayout(dev, &ignored_info, NULL, &ignored) == VK_SUCCESS);
      v->CmdPushConstants(cmd, ignored, VK_SHADER_STAGE_FRAGMENT_BIT, 16, sizeof(values), values);
      v->DestroyPipelineLayout(dev, ignored, NULL);
   }
   for (unsigned i = 0; i < 2; i++) {
      if (tables && i) {
         uint32_t values[] = {0xc0ffee00, 0xabad1dea};
         const VkPushConstantsInfo push = {
            .sType = VK_STRUCTURE_TYPE_PUSH_CONSTANTS_INFO, .layout = layout,
            .stageFlags = push_range.stageFlags, .offset = 24,
            .size = sizeof(values), .pValues = values,
         };
         v->CmdPushConstants2(cmd, &push);
         memset(values, 0xff, sizeof(values));
      }
      v->CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines[i]);
      if (dynamic && i) {
         uint32_t offsets[] = {20, 0, 64};
         v->CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &sets[i], 3, offsets);
         memset(offsets, 0xff, sizeof(offsets));
      } else if (!dynamic)
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
   if (tables) {
      const uint32_t values[60] = {0};
      v->CmdPushConstants(cmd, layout, push_range.stageFlags, 16, sizeof(values), values);
   }
   if (dynamic) {
      /* Rebinding the same set must not mutate either recorded dispatch. */
      const uint32_t offsets[] = {36, 128, 0};
      v->CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, sets, 3, offsets);
   }
   v->CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines[2]);
   v->CmdDispatch(cmd, 0, 1, 1);
   CHECK(v->EndCommandBuffer(cmd) == VK_SUCCESS);
   if (secondary) {
      v->CmdExecuteCommands(primary, 1, &secondary_cmd);
      CHECK(v->EndCommandBuffer(primary) == VK_SUCCESS);
      cmd = primary;
   }
   writes[0].pBufferInfo = &bindings[0];
   writes[0].dstBinding = tables ? 7 : 0;
   if (!dynamic) v->UpdateDescriptorSets(dev, 1, writes, 0, NULL);
   const VkCommandBufferSubmitInfo cb_submit = {
      .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = cmd,
   };
   VkFence fence = VK_NULL_HANDLE;
   VkSemaphore timeline = VK_NULL_HANDLE, binary = VK_NULL_HANDLE;
   VkSemaphoreSubmitInfo wait = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
      .value = 7, .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT};
   VkSemaphoreSubmitInfo signals[2] = {
      {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
       .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT},
      {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO,
       .value = 11, .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT},
   };
   VkSubmitInfo2 submit = {
      .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
      .commandBufferInfoCount = 1, .pCommandBufferInfos = &cb_submit,
   };
   if (sync_fd >= 0) {
      const VkFenceCreateInfo fence_info = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
      CHECK(v->CreateFence(dev, &fence_info, NULL, &fence) == VK_SUCCESS);
      CHECK(v->GetFenceStatus(dev, fence) == VK_NOT_READY);
      const VkSemaphoreTypeCreateInfo type = {
         .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
         .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE, .initialValue = 5,
      };
      VkSemaphoreCreateInfo sem_info = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &type};
      CHECK(v->CreateSemaphore(dev, &sem_info, NULL, &timeline) == VK_SUCCESS);
      sem_info.pNext = NULL;
      CHECK(v->CreateSemaphore(dev, &sem_info, NULL, &binary) == VK_SUCCESS);
      wait.semaphore = timeline;
      signals[0].semaphore = binary;
      signals[1].semaphore = timeline;
      submit.waitSemaphoreInfoCount = 1;
      submit.pWaitSemaphoreInfos = &wait;
      submit.signalSemaphoreInfoCount = 2;
      submit.pSignalSemaphoreInfos = signals;
   }
   VkResult result = v->QueueSubmit2(queue, 1, &submit, fence);
   if (sync_fd >= 0) {
      CHECK(result == VK_SUCCESS);
      CHECK(v->WaitForFences(dev, 1, &fence, VK_TRUE, 1000000) == VK_TIMEOUT);
      CHECK(!memcmp(mock.mapped, mock.expected, sizeof(mock.expected)));
      VkSemaphoreSignalInfo signal = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
         .semaphore = timeline, .value = 6};
      CHECK(v->SignalSemaphore(dev, &signal) == VK_SUCCESS);
      CHECK(v->WaitForFences(dev, 1, &fence, VK_TRUE, 1000000) == VK_TIMEOUT);
      signal.value = 7;
      CHECK(v->SignalSemaphore(dev, &signal) == VK_SUCCESS);
      result = v->WaitForFences(dev, 1, &fence, VK_TRUE, 5000000000ull);
   }
   if (fault >= 0) {
      CHECK(result == VK_ERROR_DEVICE_LOST);
      CHECK(v->QueueWaitIdle(queue) == VK_ERROR_DEVICE_LOST);
      if (transport == APEX_TRANSPORT_DRM) {
         CHECK(mock.live == ((fault >= 204 && fault <= 206) || (fault >= 211 && fault <= 213) ? 2 : 1) && !mock.calls[APEX_NATIVE_CREATE]);
         CHECK(mock.exec_calls == (fault >= 204 && fault <= 206 ? 1 : 0));
         CHECK(!memcmp(mock.mapped, mock.expected, sizeof(mock.expected)));
      } else {
         CHECK(mock.calls[APEX_NATIVE_CREATE] == (fault >= 100 ? 0 : 1));
         CHECK(mock.calls[APEX_NATIVE_CLOSE] == (fault == APEX_NATIVE_CREATE || fault >= 100 ? 0 : 1));
         if (fault != APEX_NATIVE_DOWNLOAD && fault != APEX_NATIVE_CLOSE)
            CHECK(!mock.calls[APEX_NATIVE_DOWNLOAD]);
      }
      unsigned calls = mock.calls[APEX_NATIVE_CREATE];
      unsigned exec_calls = mock.exec_calls;
      CHECK(v->QueueSubmit2(queue, 1, &submit, VK_NULL_HANDLE) == VK_ERROR_DEVICE_LOST);
      CHECK(mock.calls[APEX_NATIVE_CREATE] == calls);
      CHECK(mock.exec_calls == exec_calls);
   } else {
      CHECK(result == VK_SUCCESS && v->QueueWaitIdle(queue) == VK_SUCCESS);
      if (transport == APEX_TRANSPORT_DRM) {
         CHECK(mock.mapped[48] == 0xd00d0030 && mock.mapped[1088] == 0xd00d0440);
         range.offset = 1088 * 4;
         range.size = 64;
         CHECK(v->InvalidateMappedMemoryRanges(dev, 1, &range) == VK_SUCCESS);
         CHECK(mock.mapped[48] == 0xd00d0030 && mock.mapped[1088] == 101);
      }
      CHECK(v->InvalidateMappedMemoryRanges(dev, 1, &whole) == VK_SUCCESS);
      if (sync_fd >= 0) {
         uint64_t value;
         CHECK(v->GetSemaphoreCounterValue(dev, timeline, &value) == VK_SUCCESS && value == 11);
         CHECK(v->ResetFences(dev, 1, &fence) == VK_SUCCESS);
         CHECK(v->GetFenceStatus(dev, fence) == VK_NOT_READY);
         wait.semaphore = binary;
         wait.value = 0;
         signals[1].value = 23;
         submit.signalSemaphoreInfoCount = 1;
         submit.pSignalSemaphoreInfos = &signals[1];
      }
      for (unsigned i = 0; i < ARRAY_SIZE(mock.expected); i++) {
         uint32_t expected = 0xd00d0000 + i;
         if (i == 2047) expected = 0x12345678;
         if (i >= 48 && i < 48 + invocations[0]) expected = 37 + 3 * (i - 48);
         if (i >= 1088 && i < 1088 + invocations[1]) expected = 101 + 7 * (i - 1088);
         CHECK(mock.mapped[i] == expected);
      }
      /* An unflushed shadow write must not overwrite retained GPU data. */
      if (transport == APEX_TRANSPORT_DRM)
         mock.mapped[100] = 0xbad;
      /* Reset must discard both old dispatches and bound pipeline/set state. */
      CHECK(v->ResetCommandBuffer(cmd, 0) == VK_SUCCESS);
      CHECK(v->BeginCommandBuffer(cmd, &begin) == VK_SUCCESS);
      v->CmdDispatch(cmd, 1, 1, 1);
      CHECK(v->EndCommandBuffer(cmd) == VK_ERROR_FEATURE_NOT_PRESENT);
      CHECK(v->ResetCommandBuffer(cmd, 0) == VK_SUCCESS);
      CHECK(v->BeginCommandBuffer(cmd, &begin) == VK_SUCCESS);
      if (secondary) {
         CHECK(v->ResetCommandBuffer(secondary_cmd, 0) == VK_SUCCESS);
         cmd = secondary_cmd;
         CHECK(v->BeginCommandBuffer(cmd, &secondary_begin) == VK_SUCCESS);
      }
      v->CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines[2]);
      if (dynamic) {
         const uint32_t offsets[] = {28, 64, 0};
         v->CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, sets, 3, offsets);
      } else
         v->CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, sets, 0, NULL);
      if (tables && !dynamic)
         v->CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 1, 1, &sets[1], 0, NULL);
      if (tables) {
         const uint32_t value = 0xdecafbad;
         v->CmdPushConstants(cmd, layout, push_range.stageFlags, 252, sizeof(value), &value);
      }
      v->CmdDispatch(cmd, 1, 1, 1);
      CHECK(v->EndCommandBuffer(cmd) == VK_SUCCESS);
      if (secondary) {
         v->CmdExecuteCommands(primary, 1, &secondary_cmd);
         CHECK(v->EndCommandBuffer(primary) == VK_SUCCESS);
         cmd = primary;
      }
      CHECK(v->QueueSubmit2(queue, 1, &submit, fence) == VK_SUCCESS);
      CHECK(v->DeviceWaitIdle(dev) == VK_SUCCESS);
      if (sync_fd >= 0) {
         uint64_t value;
         CHECK(v->GetFenceStatus(dev, fence) == VK_SUCCESS);
         CHECK(v->GetSemaphoreCounterValue(dev, timeline, &value) == VK_SUCCESS && value == 23);
      }
      if (transport == APEX_TRANSPORT_DRM) {
         VkSubmitInfo2 repeat = submit;
         repeat.waitSemaphoreInfoCount = repeat.signalSemaphoreInfoCount = 0;
         if (secondary) {
            /* Replay must leave the secondary queue reusable after the
             * primary's dispatch snapshots have been discarded. */
            CHECK(v->ResetCommandBuffer(primary, 0) == VK_SUCCESS);
            CHECK(v->BeginCommandBuffer(primary, &begin) == VK_SUCCESS);
            v->CmdExecuteCommands(primary, 1, &secondary_cmd);
            CHECK(v->EndCommandBuffer(primary) == VK_SUCCESS);
         }
         CHECK(v->QueueSubmit2(queue, 1, &repeat, VK_NULL_HANDLE) == VK_SUCCESS);
         CHECK(v->DeviceWaitIdle(dev) == VK_SUCCESS);
      }
      CHECK(v->InvalidateMappedMemoryRanges(dev, 1, &whole) == VK_SUCCESS);
      for (unsigned i = 0; i < ARRAY_SIZE(mock.expected); i++) {
         uint32_t expected = 0xd00d0000 + i;
         if (i == 2047) expected = 0x12345678;
         if (i >= 48 && i < 60) expected = 112 + 7 * (i - 48);
         if (i >= 60 && i < 48 + invocations[0]) expected = 37 + 3 * (i - 48);
         if (i >= 1088 && i < 1088 + invocations[1]) expected = 101 + 7 * (i - 1088);
         CHECK(mock.mapped[i] == expected);
      }
      if (mocked) CHECK(mock.dispatch == (transport == APEX_TRANSPORT_DRM ? 4 : 3) && mock.step == 0);
   }
   if (sync_fd >= 0) {
      v->DestroyFence(dev, fence, NULL);
      v->DestroySemaphore(dev, binary, NULL);
      v->DestroySemaphore(dev, timeline, NULL);
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
   if (fault < 0 && transport == APEX_TRANSPORT_DRM) {
      /* Free all objects, then reuse the address for a dedicated odd-sized
       * buffer. Query outputs start true to catch an untouched pNext chain. */
      VkBufferCreateInfo odd_buffer = buffer_info;
      odd_buffer.size = 4097;
      VkBuffer buffer;
      CHECK(v->CreateBuffer(dev, &odd_buffer, NULL, &buffer) == VK_SUCCESS);
      VkMemoryDedicatedRequirements dedicated_req = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS,
         .prefersDedicatedAllocation = VK_TRUE, .requiresDedicatedAllocation = VK_TRUE,
      };
      VkMemoryRequirements2 req = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2, .pNext = &dedicated_req,
      };
      const VkBufferMemoryRequirementsInfo2 req_info = {
         .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2, .buffer = buffer,
      };
      v->GetBufferMemoryRequirements2(dev, &req_info, &req);
      CHECK(req.memoryRequirements.size == 4160 && req.memoryRequirements.alignment == 64 &&
            (req.memoryRequirements.memoryTypeBits & 1));
      CHECK(!dedicated_req.prefersDedicatedAllocation && !dedicated_req.requiresDedicatedAllocation);
      VkMemoryRequirements legacy_req;
      v->GetBufferMemoryRequirements(dev, buffer, &legacy_req);
      CHECK(legacy_req.size == 4160 && legacy_req.alignment == 64 && (legacy_req.memoryTypeBits & 1));
      const VkMemoryDedicatedAllocateInfo dedicated = {
         .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO, .buffer = buffer,
      };
      VkMemoryAllocateInfo odd = mem_info;
      odd.pNext = &dedicated;
      odd.allocationSize = 4160;
      CHECK(v->AllocateMemory(dev, &odd, NULL, &memory) == VK_SUCCESS);
      CHECK(v->BindBufferMemory(dev, buffer, memory, 0) == VK_SUCCESS);
      CHECK(v->MapMemory(dev, memory, 0, VK_WHOLE_SIZE, 0, (void **)&mock.mapped) == VK_SUCCESS);
      for (unsigned i = 0; i < odd.allocationSize; i++)
         CHECK(((uint8_t *)mock.mapped)[i] == 0);
      if (mocked) {
         CHECK(mock.objects == 9 && mock.live == 1);
         CHECK(mock.gems[9].va == (1ull << 39) - 8192);
      }
      v->UnmapMemory(dev, memory);
      v->DestroyBuffer(dev, buffer, NULL);
      v->FreeMemory(dev, memory, NULL);
   }
   apex_device_finish(&device);
   if (mock.drm_fd >= 0) {
      CHECK(!mock.live && !close(mock.drm_fd));
      mock.drm_fd = -1;
   }
}

static void
run_images(struct vk_physical_device *physical, const uint32_t *spirv, size_t size)
{
   mock.images = true;
   mock.fault = -1;
   mock.next_offset = 4096;
   mock.drm_fd = memfd_create("apex-image-test", MFD_CLOEXEC);
   CHECK(mock.drm_fd >= 0);
   physical->properties.maxComputeWorkGroupCount[0] = 1024;
   physical->properties.maxComputeWorkGroupSize[0] = 16;
   const float priority = 1;
   const VkDeviceQueueCreateInfo qi = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueCount = 1, .pQueuePriorities = &priority,
   };
   const VkDeviceCreateInfo di = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .queueCreateInfoCount = 1, .pQueueCreateInfos = &qi,
   };
   struct apex_device device;
   CHECK(apex_device_init(&device, physical, &di, NULL, mock.drm_fd, APEX_TRANSPORT_DRM) == VK_SUCCESS);
   VkDevice dev = apex_device_to_handle(&device);
   const struct vk_device_dispatch_table *v = &device.vk.dispatch_table;
   VkDeviceMemory memory;
   VkMemoryAllocateInfo mi = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = 16384};
   CHECK(v->AllocateMemory(dev, &mi, NULL, &memory) == VK_SUCCESS);
   uint32_t *mapped, expected[4096];
   CHECK(v->MapMemory(dev, memory, 0, VK_WHOLE_SIZE, 0, (void **)&mapped) == VK_SUCCESS);
   for (unsigned i = 0; i < ARRAY_SIZE(expected); i++) mapped[i] = expected[i] = 0x81230000 + 13 * i;
   VkMappedMemoryRange range = {.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE, .memory = memory, .size = VK_WHOLE_SIZE};
   CHECK(v->FlushMappedMemoryRanges(dev, 1, &range) == VK_SUCCESS);
   VkImage images[2];
   VkImageCreateInfo ii = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
      .format = VK_FORMAT_R32_UINT, .extent = {11, 7, 1}, .mipLevels = 3, .arrayLayers = 4,
      .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_LINEAR,
      .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
   };
   for (unsigned i = 0; i < 2; i++) {
      if (i) { ii.extent = (VkExtent3D){9, 5, 1}; ii.mipLevels = 2; ii.arrayLayers = 3; ii.tiling = VK_IMAGE_TILING_OPTIMAL; }
      CHECK(v->CreateImage(dev, &ii, NULL, &images[i]) == VK_SUCCESS);
      VkMemoryRequirements req;
      v->GetImageMemoryRequirements(dev, images[i], &req);
      CHECK(req.size == (i ? 1344 : 2816) && req.alignment == 64 && (req.memoryTypeBits & 1));
      VkMemoryDedicatedRequirements dedicated = {.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS,
         .prefersDedicatedAllocation = VK_TRUE, .requiresDedicatedAllocation = VK_TRUE};
      VkMemoryRequirements2 req2 = {.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2, .pNext = &dedicated};
      VkDeviceImageMemoryRequirements dri = {.sType = VK_STRUCTURE_TYPE_DEVICE_IMAGE_MEMORY_REQUIREMENTS, .pCreateInfo = &ii};
      v->GetDeviceImageMemoryRequirements(dev, &dri, &req2);
      CHECK(req.size == req2.memoryRequirements.size && req.alignment == req2.memoryRequirements.alignment &&
            req.memoryTypeBits == req2.memoryRequirements.memoryTypeBits);
      CHECK(!dedicated.prefersDedicatedAllocation && !dedicated.requiresDedicatedAllocation);
      CHECK(v->BindImageMemory(dev, images[i], memory, 1) == VK_ERROR_OUT_OF_DEVICE_MEMORY);
      CHECK(v->BindImageMemory(dev, images[i], memory, 16320) == VK_ERROR_OUT_OF_DEVICE_MEMORY);
      CHECK(v->BindImageMemory(dev, images[i], memory, i ? 8192 : 256) == VK_SUCCESS);
   }
   VkImage rejected;
   ii.format = VK_FORMAT_BC1_RGB_UNORM_BLOCK;
   CHECK(v->CreateImage(dev, &ii, NULL, &rejected) == VK_ERROR_FORMAT_NOT_SUPPORTED && !rejected);
   ii.format = VK_FORMAT_R8G8B8A8_UNORM;
   CHECK(v->CreateImage(dev, &ii, NULL, &rejected) == VK_SUCCESS);
   v->DestroyImage(dev, rejected, NULL);
   ii.extent = (VkExtent3D){2, 2, 1}; ii.mipLevels = 1; ii.arrayLayers = 1;
   ii.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
   VkImage rgba;
   CHECK(v->CreateImage(dev, &ii, NULL, &rgba) == VK_SUCCESS);
   CHECK(v->BindImageMemory(dev, rgba, memory, 14592) == VK_SUCCESS);
   VkPhysicalDeviceImageFormatInfo2 format = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2,
      .format = ii.format, .type = ii.imageType, .tiling = ii.tiling, .usage = ii.usage};
   VkImageFormatProperties2 props = {.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2};
   CHECK(apex_image_format_properties(&format, &props) == VK_SUCCESS &&
         props.imageFormatProperties.sampleCounts == VK_SAMPLE_COUNT_1_BIT);
   format.usage |= VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
   CHECK(apex_image_format_properties(&format, &props) == VK_ERROR_FORMAT_NOT_SUPPORTED);
   VkImageSubresource sub = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .mipLevel = 1, .arrayLayer = 1};
   VkSubresourceLayout sublayout;
   v->GetImageSubresourceLayout(dev, images[0], &sub, &sublayout);
   CHECK(sublayout.offset == 1984 && sublayout.rowPitch == 64 && sublayout.arrayPitch == 192 && sublayout.size == 192);
   VkImageView views[2];
   VkImageViewCreateInfo vi = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = images[0],
      .viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY, .format = VK_FORMAT_R32_UINT,
      .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 1, 1, 1, 2},
   };
   CHECK(v->CreateImageView(dev, &vi, NULL, &views[0]) == VK_SUCCESS);
   vi.subresourceRange = (VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 2, VK_REMAINING_ARRAY_LAYERS};
   CHECK(v->CreateImageView(dev, &vi, NULL, &views[1]) == VK_SUCCESS);
   VkBuffer buffers[2];
   VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 2048,
      .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
   for (unsigned i = 0; i < 2; i++) {
      CHECK(v->CreateBuffer(dev, &bi, NULL, &buffers[i]) == VK_SUCCESS);
      CHECK(v->BindBufferMemory(dev, buffers[i], memory, i ? 12288 : 4096) == VK_SUCCESS);
   }
   VkDescriptorSetLayout layouts[2];
   for (unsigned i = 0; i < 2; i++) {
      VkDescriptorSetLayoutBinding binding = {.binding = i ? 5 : 3, .descriptorCount = i ? 2 : 1,
         .descriptorType = i ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
         .stageFlags = VK_SHADER_STAGE_ALL};
      VkDescriptorSetLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
         .bindingCount = 1, .pBindings = &binding};
      CHECK(v->CreateDescriptorSetLayout(dev, &li, NULL, &layouts[i]) == VK_SUCCESS);
      const VkShaderStageFlags stages[] = {VK_SHADER_STAGE_COMPUTE_BIT,
         VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT, VK_SHADER_STAGE_FRAGMENT_BIT};
      for (unsigned s = 0; s < ARRAY_SIZE(stages); s++) {
         binding.stageFlags = stages[s];
         VkDescriptorSetLayout other;
         CHECK(v->CreateDescriptorSetLayout(dev, &li, NULL, &other) == VK_SUCCESS);
         CHECK(memcmp(vk_descriptor_set_layout_from_handle(layouts[i])->blake3,
                      vk_descriptor_set_layout_from_handle(other)->blake3, BLAKE3_OUT_LEN));
         v->DestroyDescriptorSetLayout(dev, other, NULL);
      }
      binding.descriptorType = VK_DESCRIPTOR_TYPE_INLINE_UNIFORM_BLOCK;
      VkDescriptorSetLayout unsupported;
      CHECK(v->CreateDescriptorSetLayout(dev, &li, NULL, &unsupported) == VK_ERROR_FEATURE_NOT_PRESENT && !unsupported);
   }
   VkDescriptorPoolSize sizes[VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT + 1];
   for (unsigned type = 0; type < ARRAY_SIZE(sizes); type++)
      sizes[type] = (VkDescriptorPoolSize){type, type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE ? 2 : 1};
   VkDescriptorPoolCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 3, .poolSizeCount = ARRAY_SIZE(sizes), .pPoolSizes = sizes};
   VkDescriptorPool pool;
   CHECK(v->CreateDescriptorPool(dev, &pi, NULL, &pool) == VK_SUCCESS);
   VkDescriptorSetAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = pool, .descriptorSetCount = 2, .pSetLayouts = layouts};
   VkDescriptorSet sets[2];
   CHECK(v->AllocateDescriptorSets(dev, &ai, sets) == VK_SUCCESS);
   ai.descriptorSetCount = 1;
   VkDescriptorSet extra;
   CHECK(v->AllocateDescriptorSets(dev, &ai, &extra) == VK_ERROR_OUT_OF_POOL_MEMORY && !extra);
   VkDescriptorBufferInfo db = {.buffer = buffers[0], .offset = 128, .range = 768};
   VkDescriptorImageInfo images_info[] = {{.imageView = views[0], .imageLayout = VK_IMAGE_LAYOUT_GENERAL},
                                        {.imageView = views[1], .imageLayout = VK_IMAGE_LAYOUT_GENERAL}};
   VkWriteDescriptorSet writes[] = {
      {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = sets[0], .dstBinding = 3,
       .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &db},
      {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = sets[1], .dstBinding = 5,
       .descriptorCount = 2, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = images_info},
   };
   v->UpdateDescriptorSets(dev, 2, writes, 0, NULL);
   VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 16};
   VkPipelineLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 2, .pSetLayouts = layouts, .pushConstantRangeCount = 1, .pPushConstantRanges = &pr};
   VkPipelineLayout layout;
   CHECK(v->CreatePipelineLayout(dev, &li, NULL, &layout) == VK_SUCCESS);
   VkShaderModuleCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = size, .pCode = spirv};
   VkShaderModule module;
   CHECK(v->CreateShaderModule(dev, &si, NULL, &module) == VK_SUCCESS);
   VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, .layout = layout,
      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
         .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = module, .pName = "main"}};
   VkPipeline pipeline;
   CHECK(v->CreateComputePipelines(dev, VK_NULL_HANDLE, 1, &ci, NULL, &pipeline) == VK_SUCCESS);
   mock.pipelines[0] = apex_pipeline_from_handle(pipeline);
   VkCommandPool cp;
   VkCommandPoolCreateInfo cpi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
   CHECK(v->CreateCommandPool(dev, &cpi, NULL, &cp) == VK_SUCCESS);
   VkCommandBuffer cmd;
   VkCommandBufferAllocateInfo cai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = cp, .commandBufferCount = 1};
   CHECK(v->AllocateCommandBuffers(dev, &cai, &cmd) == VK_SUCCESS);
   VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
   CHECK(v->BeginCommandBuffer(cmd, &begin) == VK_SUCCESS);
   v->CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
   v->CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 2, sets, 0, NULL);
   const uint32_t push[] = {0xc0010000, 0xc0010001, 0xc0010002, 0xc0010003};
   v->CmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), push);
   v->CmdDispatch(cmd, 1, 1, 1);
   VkClearColorValue color = {.uint32 = {0x5a17c0de}};
   VkImageSubresourceRange clear = {VK_IMAGE_ASPECT_COLOR_BIT, 1, VK_REMAINING_MIP_LEVELS, 1, VK_REMAINING_ARRAY_LAYERS};
   v->CmdClearColorImage(cmd, images[1], VK_IMAGE_LAYOUT_GENERAL, &color, 1, &clear);
   VkBufferImageCopy2 copy = {.sType = VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2,
      .bufferOffset = 28, .bufferRowLength = 5, .bufferImageHeight = 3,
      .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 1, 1, 2}, .imageOffset = {1, 1, 0}, .imageExtent = {3, 2, 1}};
   VkCopyBufferToImageInfo2 to = {.sType = VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2,
      .srcBuffer = buffers[1], .dstImage = images[0], .dstImageLayout = VK_IMAGE_LAYOUT_GENERAL,
      .regionCount = 1, .pRegions = &copy};
   v->CmdCopyBufferToImage2(cmd, &to);
   VkImageMemoryBarrier barrier = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
      .oldLayout = VK_IMAGE_LAYOUT_GENERAL, .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
      .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT, .dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .image = images[0], .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 1, 1, 1, 2}};
   v->CmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                         0, NULL, 0, NULL, 1, &barrier);
   VkImageCopy region = {.srcSubresource = copy.imageSubresource, .srcOffset = copy.imageOffset,
      .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 2}, .dstOffset = {2, 2, 0}, .extent = copy.imageExtent};
   v->CmdCopyImage(cmd, images[0], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, images[1], VK_IMAGE_LAYOUT_GENERAL, 1, &region);
   copy.bufferOffset = 512; copy.bufferRowLength = 7; copy.bufferImageHeight = 4;
   copy.imageSubresource = region.dstSubresource; copy.imageOffset = region.dstOffset;
   VkCopyImageToBufferInfo2 from = {.sType = VK_STRUCTURE_TYPE_COPY_IMAGE_TO_BUFFER_INFO_2,
      .srcImage = images[1], .srcImageLayout = VK_IMAGE_LAYOUT_GENERAL, .dstBuffer = buffers[1],
      .regionCount = 1, .pRegions = &copy};
   v->CmdCopyImageToBuffer2(cmd, &from);
   VkClearColorValue rgba_color = {.float32 = {0.0f, 1.0f, 0.5f, 1.0f}};
   VkImageSubresourceRange rgba_range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
   v->CmdClearColorImage(cmd, rgba, VK_IMAGE_LAYOUT_GENERAL, &rgba_color, 1, &rgba_range);
   v->CmdDispatch(cmd, 1, 1, 1); /* Meta commands must restore pipeline/push/descriptors. */
   CHECK(v->EndCommandBuffer(cmd) == VK_SUCCESS);
   images_info[0].imageView = views[1]; images_info[1].imageView = views[0];
   v->UpdateDescriptorSets(dev, 1, &writes[1], 0, NULL);
   VkCommandBufferSubmitInfo cb = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = cmd};
   VkSubmitInfo2 submit = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2, .commandBufferInfoCount = 1, .pCommandBufferInfos = &cb};
   CHECK(v->QueueSubmit2(vk_queue_to_handle(&device.queue), 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS);
   CHECK(v->QueueWaitIdle(vk_queue_to_handle(&device.queue)) == VK_SUCCESS && mock.dispatch == 7);
   CHECK(v->InvalidateMappedMemoryRanges(dev, 1, &range) == VK_SUCCESS);
   for (unsigned layer = 0; layer < 2; layer++)
      for (unsigned y = 0; y < 2; y++)
         for (unsigned x = 0; x < 4; x++) expected[2320 + layer * 32 + y * 16 + x] = 0x5a17c0de;
   for (unsigned y = 0; y < 2; y++)
      for (unsigned x = 0; x < 2; x++) expected[3648 + y * 16 + x] = 0xff80ff00;
   for (unsigned z = 0; z < 2; z++) for (unsigned y = 0; y < 2; y++) for (unsigned x = 0; x < 3; x++) {
      uint32_t value = 0x81230000 + 13 * (3079 + z * 15 + y * 5 + x);
      expected[577 + z * 48 + y * 16 + x] = value;
      expected[2162 + z * 80 + y * 16 + x] = value;
      expected[3200 + z * 28 + y * 7 + x] = value;
   }
   CHECK(!memcmp(mapped, expected, sizeof(expected)));
   v->DestroyCommandPool(dev, cp, NULL);
   v->DestroyPipeline(dev, pipeline, NULL); v->DestroyPipelineLayout(dev, layout, NULL);
   v->DestroyShaderModule(dev, module, NULL); v->DestroyDescriptorPool(dev, pool, NULL);
   v->DestroyImage(dev, rgba, NULL);
   for (unsigned i = 0; i < 2; i++) {
      v->DestroyDescriptorSetLayout(dev, layouts[i], NULL);
      v->DestroyImageView(dev, views[i], NULL); v->DestroyImage(dev, images[i], NULL);
      v->DestroyBuffer(dev, buffers[i], NULL);
   }
   v->UnmapMemory(dev, memory); v->FreeMemory(dev, memory, NULL);
   apex_device_finish(&device);
   CHECK(!mock.live && !close(mock.drm_fd));
   puts("PASS Apex image resources: mip/layer views, descriptor updates, transfer pitches, clear ranges, meta state, 4096 words/guards (mock transport)");
}

int main(int argc, char **argv)
{
   CHECK(argc == 2 || argc == 3 ||
         (argc == 4 && (!strcmp(argv[2], "--drm") || !strcmp(argv[2], "--syncobj") ||
                       !strcmp(argv[2], "--drm-multiwave") || !strcmp(argv[2], "--drm-async"))));
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
   if (argc == 3 && !strcmp(argv[2], "--images")) {
      run_images(&physical, spirv, size);
      free(spirv);
      vk_physical_device_finish(&physical);
      vk_instance_finish(&instance);
      return 0;
   }
   int fd = argc > 2 ? open(argv[argc - 1], O_RDWR | O_CLOEXEC) : -1;
   CHECK(argc == 2 || fd >= 0);
   bool multiwave = argc == 4 && !strcmp(argv[2], "--drm-multiwave");
   bool require_async = argc == 4 && !strcmp(argv[2], "--drm-async");
   bool async = false;
   enum apex_transport transport = argc == 4 && (!strcmp(argv[2], "--drm") || multiwave || require_async) ?
      APEX_TRANSPORT_DRM : APEX_TRANSPORT_NATIVE;
   mock.drm_fd = -1;
   struct vk_sync_type sync_type;
   const struct vk_sync_type *sync_types[] = {&sync_type, NULL};
   if (argc == 4) {
      if (!geteuid()) {
         CHECK(!setgroups(0, NULL) && !setgid(65534) && !setuid(65534));
      }
      CHECK(geteuid() != 0);
      sync_fd = fd;
      sync_type = vk_drm_syncobj_get_type(fd);
      CHECK(sync_type.features & VK_SYNC_FEATURE_TIMELINE);
      physical.supported_sync_types = sync_types;
      if (transport == APEX_TRANSPORT_NATIVE) fd = -1;
   }
   if (transport == APEX_TRANSPORT_DRM) {
      struct drm_apex_info caps = {0};
      CHECK(!ioctl(fd, DRM_IOCTL_APEX_INFO, &caps));
      async = caps.capabilities & APEX_DRM_CAP_ASYNC;
      if (caps.version != 2 || !(caps.capabilities & APEX_DRM_CAP_GPUVM) || (require_async && !async)) {
         fprintf(stderr, "SKIP Apex Mesa DRM execution: missing GPUVM v2 or requested async capability\n");
         CHECK(!close(fd));
         free(spirv);
         vk_physical_device_finish(&physical);
         vk_instance_finish(&instance);
         return 77;
      }
   }
   run(&physical, spirv, size, fd, -1, transport, false, false);
   if (multiwave || argc == 2) {
      const unsigned widths[] = {17, 32, 256};
      for (unsigned i = 0; i < ARRAY_SIZE(widths); i++) {
         invocations[0] = invocations[1] = widths[i];
         run(&physical, spirv, size, fd, -1, APEX_TRANSPORT_DRM, false, false);
      }
      invocations[0] = invocations[1] = 16;
      printf("PASS Apex Mesa APX2 %s: 17/32/256 invocations, 12 dispatches, later-wave values and retained tails, 2048 words/guards\n",
             multiwave ? "DRM GPUVM execution" : "mock transport (not GPU execution)");
   }
   if (fd >= 0) {
      CHECK(close(fd) == 0);
      printf("PASS Apex Mesa %s command submission: %u dispatches, 2 buffers, 2048 words/guards\n",
             transport == APEX_TRANSPORT_DRM ? (async ? "DRM async GPUVM" : "DRM GPUVM") : "native",
             transport == APEX_TRANSPORT_DRM ? 4 : 3);
   } else {
      const int faults[] = {0, 1, 2, 3, 4, 5, 6, 8, 10, 99, 100, 101};
      for (unsigned i = 0; i < ARRAY_SIZE(faults); i++)
         run(&physical, spirv, size, -1, faults[i], APEX_TRANSPORT_NATIVE, false, false);
      const int drm_faults[] = {-1, 100, 101, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213};
      for (unsigned i = 0; i < ARRAY_SIZE(drm_faults); i++)
         run(&physical, spirv, size, -1, drm_faults[i], APEX_TRANSPORT_DRM, false, false);
      const int dynamic_faults[] = {-1, 214, 215, 216, 217};
      for (unsigned i = 0; i < ARRAY_SIZE(dynamic_faults); i++)
         run(&physical, spirv, size, -1, dynamic_faults[i], APEX_TRANSPORT_DRM, true, false);
      run(&physical, spirv, size, -1, -1, APEX_TRANSPORT_DRM, false, true);
      run(&physical, spirv, size, -1, -1, APEX_TRANSPORT_DRM, true, true);
      puts("PASS Apex secondary replay: pipeline, descriptor/dynamic offsets, push snapshots, reset/reuse, 2048 words/guards (mock transport)");
      puts("PASS Apex Mesa native/DRM transport: persistent GPUVA, explicit ranges, retained data/programs, cleanup, device loss (mock ioctl)");
      if (sync_fd >= 0) {
         CHECK(!close(sync_fd));
         puts("PASS Apex Mesa DRM syncobjs: timeline wait-before-signal, binary dependency, fence reuse, fault wakeup (mock execution)");
      }
   }
   free(spirv);
   vk_physical_device_finish(&physical);
   vk_instance_finish(&instance);
   return 0;
}
