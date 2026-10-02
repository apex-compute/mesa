/* SPDX-License-Identifier: MIT */
/* Offline command-processor gate: packet encodings for every opcode, IB
 * growth, ring wrap and space accounting, the vkQueueSubmit batch sequence,
 * and a device recorded through Vulkan whose ring the command-processor model
 * in mock_kernel.h executes. Steady-state submission must issue no ioctl. */
#include "apex_device.h"
#include "apex_job.h"
#include "apex_pipeline.h"
#include "mock_kernel.h"
#include "vk_alloc.h"
#include "vk_buffer.h"
#include "vk_instance.h"
#include "vk_physical_device.h"
#include <stdarg.h>
#include <sys/ioctl.h>

#define CHECK MOCK_CHECK

static struct mock_kernel k;
static struct {
   unsigned count;
   struct mock_dispatch last;
} seen;

int __real_ioctl(int fd, unsigned long request, ...);
int __wrap_ioctl(int fd, unsigned long request, ...);
int
__wrap_ioctl(int fd, unsigned long request, ...)
{
   va_list ap;
   va_start(ap, request);
   void *arg = va_arg(ap, void *);
   va_end(ap);
   if (fd == k.fd)
      return mock_kernel_ioctl(&k, request, arg);
   return __real_ioctl(fd, request, arg);
}

static void
expect(struct apex_ib *ib, const uint32_t *want, unsigned count)
{
   CHECK(!ib->failed && ib->count == count);
   for (unsigned i = 0; i < count; i++) {
      if (ib->words[i] != want[i]) {
         fprintf(stderr, "word %u: %#x, expected %#x\n", i, ib->words[i], want[i]);
         abort();
      }
   }
   apex_ib_reset(ib);
}
#define EXPECT(ib, ...) do { const uint32_t w[] = {__VA_ARGS__}; expect(ib, w, ARRAY_SIZE(w)); } while (0)

static void
test_encoder(void)
{
   struct apex_ib ib;
   apex_ib_init(&ib);
   apex_cp_packet(&ib, APEX_CP_NOP, 2, (uint32_t[]){9, 8});
   EXPECT(&ib, 0x00000200, 9, 8);
   apex_cp_indirect(&ib, 0x123456789abcull, 77);
   EXPECT(&ib, 0x00000301, 0x56789abc, 0x1234, 77);
   apex_cp_set_state(&ib, APEX_STATE_COMPUTE_USER, 3, (uint32_t[]){1, 2, 3});
   EXPECT(&ib, 0x00000402, 0x10, 1, 2, 3);
   apex_cp_dispatch(&ib, 4, 5, 6);
   EXPECT(&ib, 0x00000310, 4, 5, 6);
   apex_cp_dispatch_indirect(&ib, 0xab00000040ull);
   EXPECT(&ib, 0x00000211, 0x40, 0xab);
   apex_cp_draw(&ib, 3, 1, 2, 7);
   EXPECT(&ib, 0x00000420, 3, 1, 2, 7);
   apex_cp_draw_indexed(&ib, 6, 2, 1, -4, 7);
   EXPECT(&ib, 0x00000521, 6, 2, 1, 0xfffffffc, 7);
   apex_cp_draw_indirect(&ib, false, 0x100000200ull, 10, 16, 0x300ull);
   EXPECT(&ib, 0x00000622, 0x200, 1, 10, 16, 0x300, 0);
   apex_cp_draw_indirect(&ib, true, 0x400ull, 3, 20, 0);
   EXPECT(&ib, 0x00000623, 0x400, 0, 3, 20, 0, 0);
   apex_cp_begin_pass(&ib);
   apex_cp_end_pass(&ib);
   EXPECT(&ib, 0x00000028, 0x00000029);
   apex_cp_copy(&ib, 0x1000, 0x2000000003ull, 0x100000005ull);
   EXPECT(&ib, 0x00000630, 0x1000, 0, 3, 0x20, 5, 1);
   apex_cp_fill(&ib, 0x10000004ull, 256, 0xdeadbeef);
   EXPECT(&ib, 0x00000531, 0x10000004, 0, 256, 0, 0xdeadbeef);
   apex_cp_copy_rect(&ib, &(struct apex_cp_rect){
      .src = 0x500000001ull, .dst = 0x600000002ull, .src_pitch = 64, .dst_pitch = 128,
      .row_bytes = 48, .rows = 3, .slices = 2, .src_slice_pitch = 192, .dst_slice_pitch = 384});
   EXPECT(&ib, 0x00000b32, 1, 5, 64, 2, 6, 128, 48, 3, 2, 192, 384);
   apex_cp_barrier(&ib, APEX_CP_CLASS_ALL, 0x1f);
   EXPECT(&ib, 0x00000240, 7, 0x1f);
   apex_cp_wait(&ib, 0x1010, 0x100000002ull);
   EXPECT(&ib, 0x00000450, 0x1010, 0, 2, 1);
   apex_cp_signal(&ib, 0x70000008ull, 9);
   EXPECT(&ib, 0x00000451, 0x70000008, 0, 9, 0);
   apex_cp_kfence(&ib, 0x500000006ull);
   EXPECT(&ib, 0x00000252, 6, 5);
   apex_cp_write(&ib, 0x2000000004ull, APEX_CP_AFTER_PRIOR_WORK, 2, (uint32_t[]){0xa, 0xb});
   EXPECT(&ib, 0x00000553, 4, 0x20, 1, 0xa, 0xb);
   apex_cp_timestamp(&ib, 0x800, APEX_CP_AFTER_PRIOR_WORK);
   EXPECT(&ib, 0x00000360, 0x800, 0, 1);
   apex_cp_query_begin(&ib, 3);
   apex_cp_query_end(&ib, 3, 0x900);
   EXPECT(&ib, 0x00000161, 3, 0x00000362, 3, 0x900, 0);
   apex_cp_packet(&ib, APEX_CP_TLB_INVALIDATE, 1, (uint32_t[]){9});
   apex_cp_packet(&ib, APEX_CP_VM_DRAIN, 1, (uint32_t[]){9});
   EXPECT(&ib, 0x00000180, 9, 0x00000181, 9);

   /* Growth keeps earlier words; reserve returns contiguous space. */
   for (uint32_t i = 0; i < 100000; i++)
      apex_cp_packet(&ib, APEX_CP_NOP, 1, &i);
   CHECK(!ib.failed && ib.count == 200000 && ib.capacity >= 200000);
   for (uint32_t i = 0; i < 100000; i++)
      CHECK(ib.words[2 * i] == 0x100 && ib.words[2 * i + 1] == i);
   apex_ib_finish(&ib);
   puts("PASS Apex CP encoder: every opcode's header and payload, IB growth");
}

static void
test_ring(void)
{
   uint32_t map[16] = {0}, status[16] = {0}, doorbell[4] = {0};
   struct apex_ring ring = {.map = map, .dwords = 16, .status = status, .doorbell = doorbell};
   uint32_t words[20];
   for (unsigned i = 0; i < 20; i++)
      words[i] = i + 1;
   CHECK(apex_ring_reserve(&ring, 16, 0));
   apex_ring_write(&ring, words, 12);
   apex_ring_publish(&ring);
   CHECK(doorbell[0] == 12 && ring.wptr == 12);
   /* The status page lags; the doorbell read returns the live rptr. */
   status[0] = 0;
   doorbell[0] = 8;
   CHECK(apex_ring_reserve(&ring, 8, 0));
   apex_ring_write(&ring, words + 12, 8);
   for (unsigned i = 0; i < 4; i++)
      CHECK(map[12 + i] == 13 + i && map[i] == 17 + i);
   CHECK(map[4] == 5 && ring.wptr == 20);
   apex_ring_publish(&ring);
   CHECK(doorbell[0] == 20);
   /* Full: a faulted queue fails at once, a stalled one after the timeout. */
   doorbell[0] = 4;
   status[1] = APEX_STATUS_ENABLED | APEX_STATUS_FAULTED;
   CHECK(!apex_ring_reserve(&ring, 1, UINT64_MAX));
   status[1] = APEX_STATUS_ENABLED;
   CHECK(!apex_ring_reserve(&ring, 1, 1000000));
   status[0] = 20;
   CHECK(apex_ring_reserve(&ring, 16, 0));

   struct apex_ib ib;
   apex_ib_init(&ib);
   const uint64_t va[2] = {0x200000, 0x300040};
   const uint32_t dwords[2] = {10, 20};
   apex_cp_batch(&ib, &(struct apex_cp_batch) {
      .acquire_cache = 7, .kwait_va = 0x1010, .kwait_value = 3, .ib_count = 2,
      .ib_va = va, .ib_dwords = dwords, .retire_va = 0x400000, .sequence = 9, .kfence = true});
   EXPECT(&ib, 0x240, 0, 7, 0x450, 0x1010, 0, 3, 0, 0x301, 0x200000, 0, 10,
          0x301, 0x300040, 0, 20, 0x451, 0x400000, 0, 9, 0, 0x252, 9, 0);
   apex_cp_batch(&ib, &(struct apex_cp_batch) {
      .ib_count = 1, .ib_va = va, .ib_dwords = dwords, .retire_va = 0x400000, .sequence = 10});
   EXPECT(&ib, 0x301, 0x200000, 0, 10, 0x451, 0x400000, 0, 10, 0);
   apex_ib_finish(&ib);
   puts("PASS Apex user ring: wrap, lagging status rptr, live doorbell rptr, fault/timeout, batch sequence");
}

static void
on_dispatch(struct mock_kernel *kernel, const struct mock_dispatch *d)
{
   seen.count++;
   seen.last = *d;
}

static uint32_t *
load_spirv(const char *path, size_t *size)
{
   FILE *f = fopen(path, "rb");
   CHECK(f && !fseek(f, 0, SEEK_END));
   long bytes = ftell(f);
   CHECK(bytes > 0 && bytes % 4 == 0);
   rewind(f);
   uint32_t *code = malloc(bytes);
   CHECK(code && fread(code, 1, bytes, f) == (size_t)bytes && !fclose(f));
   *size = bytes;
   return code;
}

static void
test_device(const char *spirv_path)
{
   struct vk_instance instance;
   const struct vk_instance_extension_table extensions = {0};
   const struct vk_instance_dispatch_table instance_dispatch = {0};
   const VkInstanceCreateInfo ii = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
   CHECK(vk_instance_init(&instance, &extensions, &instance_dispatch, &ii,
                          vk_default_allocator()) == VK_SUCCESS);
   (void)vk_instance_to_handle(&instance);
   struct vk_physical_device physical;
   const struct vk_physical_device_dispatch_table physical_dispatch = {0};
   const struct vk_properties properties = {
      .subgroupSize = 16, .minSubgroupSize = 16, .maxSubgroupSize = 16,
      .maxComputeWorkGroupCount = {65535, 65535, 65535}, .maxComputeWorkGroupSize = {256, 256, 64},
   };
   CHECK(vk_physical_device_init(&physical, &instance, NULL, NULL, &properties,
                                 &physical_dispatch) == VK_SUCCESS);
   (void)vk_physical_device_to_handle(&physical);

   mock_kernel_init(&k);
   k.on_dispatch = on_dispatch;
   struct apex_device device;
   const float priority = 1;
   const VkDeviceQueueCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueCount = 1, .pQueuePriorities = &priority};
   const VkDeviceCreateInfo di = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qi};
   CHECK(apex_device_init(&device, &physical, &di, NULL, k.fd) == VK_SUCCESS);
   /* Queue creation: a readable LOCAL ring and a SYSTEM retirement word. */
   CHECK(k.queue && k.ring_bytes == 65536 && device.queue_id == MOCK_QUEUE_ID &&
         device.status_va == MOCK_STATUS_VA);
   CHECK(k.gems[1].va == k.ring_va && k.gems[1].vm_flags == APEX_VM_READ && !k.gems[1].flags);
   CHECK(k.gems[2].flags == APEX_GEM_SYSTEM && k.gems[2].vm_flags == (APEX_VM_READ | APEX_VM_WRITE));
   /* The private arena: 2 MiB of LOCAL, read-write at a 2 MiB-aligned GPUVA. */
   CHECK(device.private_arena.handle == 3 && k.gems[3].va == device.private_arena.va &&
         k.gems[3].va && !(k.gems[3].va % APEX_PRIVATE_BYTES) && k.gems[3].size == APEX_PRIVATE_BYTES &&
         !k.gems[3].flags && k.gems[3].vm_flags == (APEX_VM_READ | APEX_VM_WRITE));
   /* The bin pool: 32 MiB of LOCAL the geometry front end writes. */
   CHECK(device.bin_pool.handle == 4 && k.gems[4].size == APEX_BIN_POOL_BYTES && !k.gems[4].flags &&
         k.gems[4].vm_flags == (APEX_VM_READ | APEX_VM_WRITE));
   VkDevice dev = apex_device_to_handle(&device);
   const struct vk_device_dispatch_table *v = &device.vk.dispatch_table;
   VkQueue queue;
   v->GetDeviceQueue(dev, 0, 0, &queue);

   /* LOCAL and SYSTEM memory, both host visible and coherent. */
   VkDeviceMemory memory[2];
   uint8_t *map[2];
   for (unsigned t = 0; t < 2; t++) {
      const VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
         .allocationSize = 65536, .memoryTypeIndex = t};
      CHECK(v->AllocateMemory(dev, &ai, NULL, &memory[t]) == VK_SUCCESS);
      CHECK(v->MapMemory(dev, memory[t], 0, VK_WHOLE_SIZE, 0, (void **)&map[t]) == VK_SUCCESS);
      memset(map[t], 0, 65536);
   }
   CHECK(k.gems[5].flags == 0 && k.gems[6].flags == APEX_GEM_SYSTEM);
   VkBuffer local, system;
   const VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 16384,
      .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
               VK_BUFFER_USAGE_TRANSFER_DST_BIT};
   CHECK(v->CreateBuffer(dev, &bi, NULL, &local) == VK_SUCCESS);
   CHECK(v->CreateBuffer(dev, &bi, NULL, &system) == VK_SUCCESS);
   VkMemoryRequirements req;
   v->GetBufferMemoryRequirements(dev, local, &req);
   CHECK(req.memoryTypeBits == 3 && req.alignment == 64);
   CHECK(v->BindBufferMemory(dev, local, memory[0], 0) == VK_SUCCESS);
   CHECK(v->BindBufferMemory(dev, system, memory[1], 0) == VK_SUCCESS);
   const uint64_t local_va = vk_buffer_from_handle(local)->device_address;
   const uint64_t system_va = vk_buffer_from_handle(system)->device_address;
   CHECK(local_va == k.gems[5].va && system_va == k.gems[6].va);

   VkEvent event;
   const VkEventCreateInfo ei = {.sType = VK_STRUCTURE_TYPE_EVENT_CREATE_INFO};
   CHECK(v->CreateEvent(dev, &ei, NULL, &event) == VK_SUCCESS);
   VkQueryPool pool;
   const VkQueryPoolCreateInfo qpi = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
      .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = 2};
   CHECK(v->CreateQueryPool(dev, &qpi, NULL, &pool) == VK_SUCCESS);

   /* A compute pipeline over one storage buffer. */
   size_t size;
   uint32_t *code = load_spirv(spirv_path, &size);
   VkShaderModule module;
   const VkShaderModuleCreateInfo mi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = size, .pCode = code};
   CHECK(v->CreateShaderModule(dev, &mi, NULL, &module) == VK_SUCCESS);
   free(code);
   const VkDescriptorSetLayoutBinding binding = {.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};
   const VkDescriptorSetLayoutCreateInfo sli = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 1,
      .pBindings = &binding};
   VkDescriptorSetLayout set_layout;
   CHECK(v->CreateDescriptorSetLayout(dev, &sli, NULL, &set_layout) == VK_SUCCESS);
   const VkPipelineLayoutCreateInfo pli = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
      .setLayoutCount = 1, .pSetLayouts = &set_layout};
   VkPipelineLayout layout;
   CHECK(v->CreatePipelineLayout(dev, &pli, NULL, &layout) == VK_SUCCESS);
   const VkComputePipelineCreateInfo pci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
      .layout = layout, .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                  .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = module,
                                  .pName = "main"}};
   VkPipeline pipeline;
   CHECK(v->CreateComputePipelines(dev, VK_NULL_HANDLE, 1, &pci, NULL, &pipeline) == VK_SUCCESS);
   v->DestroyShaderModule(dev, module, NULL);
   const VkDescriptorPoolSize pool_size = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
   const VkDescriptorPoolCreateInfo dpi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &pool_size};
   VkDescriptorPool descriptor_pool;
   CHECK(v->CreateDescriptorPool(dev, &dpi, NULL, &descriptor_pool) == VK_SUCCESS);
   const VkDescriptorSetAllocateInfo dsai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = descriptor_pool, .descriptorSetCount = 1, .pSetLayouts = &set_layout};
   VkDescriptorSet set;
   CHECK(v->AllocateDescriptorSets(dev, &dsai, &set) == VK_SUCCESS);
   const VkDescriptorBufferInfo buffer_info = {local, 8192, 4096};
   const VkWriteDescriptorSet write = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = set, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
      .pBufferInfo = &buffer_info};
   v->UpdateDescriptorSets(dev, 1, &write, 0, NULL);

   VkCommandPool command_pool;
   const VkCommandPoolCreateInfo cpi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
   CHECK(v->CreateCommandPool(dev, &cpi, NULL, &command_pool) == VK_SUCCESS);
   const VkCommandBufferAllocateInfo cai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = command_pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
   VkCommandBuffer cmd;
   CHECK(v->AllocateCommandBuffers(dev, &cai, &cmd) == VK_SUCCESS);
   const VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
   CHECK(v->BeginCommandBuffer(cmd, &begin) == VK_SUCCESS);
   v->CmdResetQueryPool(cmd, pool, 0, 2);
   v->CmdFillBuffer(cmd, local, 0, 256, 0xabcd1234);
   const VkMemoryBarrier2 barrier = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_CLEAR_BIT,
      .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT, .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT};
   const VkDependencyInfo dependency = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .memoryBarrierCount = 1, .pMemoryBarriers = &barrier};
   v->CmdPipelineBarrier2(cmd, &dependency);
   const VkBufferCopy local_copy = {0, 1024 + 4, 256};
   v->CmdCopyBuffer(cmd, local, local, 1, &local_copy);
   const VkBufferCopy system_copy = {0, 2048, 128};
   v->CmdCopyBuffer(cmd, local, system, 1, &system_copy);
   const uint32_t update[4] = {0x11, 0x22, 0x33, 0x44};
   v->CmdUpdateBuffer(cmd, system, 512, sizeof(update), update);
   v->CmdSetEvent2(cmd, event, &dependency);
   v->CmdWaitEvents2(cmd, 1, &event, &dependency);
   v->CmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, pool, 1);
   v->CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
   v->CmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, 1, &set, 0, NULL);
   v->CmdDispatch(cmd, 3, 1, 1);
   CHECK(v->EndCommandBuffer(cmd) == VK_SUCCESS);

   const VkCommandBufferSubmitInfo cbi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
      .commandBuffer = cmd};
   const VkSubmitInfo2 submit = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
      .commandBufferInfoCount = 1, .pCommandBufferInfos = &cbi};
   CHECK(v->QueueSubmit2(queue, 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS);
   /* The first batch invalidates instruction caches after the program upload. */
   const uint32_t *ring = device.ring_bo.map;
   CHECK(device.ring.wptr == 12 && ring[0] == 0x240 && ring[1] == 0 &&
         ring[2] == (APEX_CP_CACHE_INSTRUCTION | APEX_CP_CACHE_L1 | APEX_CP_CACHE_TEXTURE));
   CHECK(ring[3] == 0x301 && ring[7] == 0x451 && ring[8] == (uint32_t)device.retire.va && ring[10] == 1);
   CHECK(!k.fence_count && !k.wait_count);
   mock_cp_run(&k);
   CHECK(k.rptr == 12 && !k.blocked);
   CHECK(k.packets[APEX_CP_INDIRECT] == 1 && k.packets[APEX_CP_COPY] == 2 &&
         k.packets[APEX_CP_FILL] == 2 && k.packets[APEX_CP_WRITE] == 3 &&
         k.packets[APEX_CP_WAIT] == 1 && k.packets[APEX_CP_TIMESTAMP] == 1 &&
         k.packets[APEX_CP_DISPATCH] == 1 && k.packets[APEX_CP_SIGNAL] == 1 &&
         !k.packets[APEX_CP_KFENCE]);
   /* Effects, read through the Vulkan mappings of the same bytes. */
   for (unsigned i = 0; i < 64; i++) {
      CHECK(((uint32_t *)map[0])[i] == 0xabcd1234);
      uint32_t copied;
      memcpy(&copied, map[0] + 1028 + i * 4, 4);
      CHECK(copied == 0xabcd1234);
   }
   for (unsigned i = 0; i < 32; i++)
      CHECK(((uint32_t *)(map[1] + 2048))[i] == 0xabcd1234);
   CHECK(!memcmp(map[1] + 512, update, sizeof(update)));
   CHECK(v->GetEventStatus(dev, event) == VK_EVENT_SET);
   uint64_t stamps[4] = {0};
   CHECK(v->GetQueryPoolResults(dev, pool, 0, 2, sizeof(stamps), stamps, 16,
                                VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT) ==
         VK_NOT_READY);
   CHECK(!stamps[1] && stamps[2] == 1000 && stamps[3] == 1);
   /* The dispatch: compute state, private arena, patched program and table GPUVAs. */
   struct apex_pipeline *p = apex_pipeline_from_handle(pipeline);
   CHECK(seen.count == 1 && seen.last.grid[0] == 3 && seen.last.grid[1] == 1 && seen.last.grid[2] == 1);
   CHECK(MOCK_U64(seen.last.state, APEX_STATE_COMPUTE_PROGRAM) == p->program.bo.va);
   CHECK(k.gems[p->program.bo.handle].vm_flags == (APEX_VM_READ | APEX_VM_EXEC));
   CHECK(MOCK_U64(seen.last.state, APEX_STATE_COMPUTE_PRIVATE) == k.gems[3].va);
   CHECK(seen.last.state[APEX_STATE_COMPUTE_LOCAL] == 16 &&
         seen.last.state[APEX_STATE_COMPUTE_LOCAL + 1] == 1);
   uint64_t table = MOCK_U64(seen.last.state, APEX_STATE_COMPUTE_USER);
   CHECK(table && !(table & 63));
   struct apex_buffer_descriptor row;
   mock_read(&k, table, &row, sizeof(row));
   CHECK(((uint64_t)row.high << 32 | row.low) == local_va + 8192 && row.bytes == 4096);
   /* User data s2:s3 address the trailer's grid. */
   uint64_t grid = MOCK_U64(seen.last.state, APEX_STATE_COMPUTE_USER + 2);
   CHECK(grid == table + apex_program_trailer(&p->program));
   CHECK(mock_read32(&k, grid) == 3 && mock_read32(&k, grid + 4) == 1 && mock_read32(&k, grid + 8) == 1);
   CHECK(v->QueueWaitIdle(queue) == VK_SUCCESS);

   /* Steady state: resubmission reuses the retired arena and publishes by
    * the doorbell alone, across many ring wraps. */
   unsigned ioctls = k.ioctls;
   for (unsigned i = 0; i < 3000; i++) {
      CHECK(v->QueueSubmit2(queue, 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS);
      mock_cp_run(&k);
      CHECK(*(uint64_t *)device.retire.map == i + 2);
   }
   CHECK(k.ioctls == ioctls && device.ring.wptr == 12 + 3000 * 9 && seen.count == 3001);
   CHECK(k.rptr == device.ring.wptr && mock_status32(&k, APEX_STATUS_RPTR) == k.rptr);

   /* A second command buffer in the same batch gets its own INDIRECT. */
   VkCommandBuffer second;
   CHECK(v->AllocateCommandBuffers(dev, &cai, &second) == VK_SUCCESS);
   CHECK(v->BeginCommandBuffer(second, &begin) == VK_SUCCESS);
   v->CmdFillBuffer(second, system, 4096, VK_WHOLE_SIZE, 0x5a5a5a5a);
   CHECK(v->EndCommandBuffer(second) == VK_SUCCESS);
   const VkCommandBufferSubmitInfo both[2] = {cbi, {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
      .commandBuffer = second}};
   const VkSubmitInfo2 submit2 = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
      .commandBufferInfoCount = 2, .pCommandBufferInfos = both};
   unsigned indirect = k.packets[APEX_CP_INDIRECT];
   CHECK(v->QueueSubmit2(queue, 1, &submit2, VK_NULL_HANDLE) == VK_SUCCESS);
   mock_cp_run(&k);
   CHECK(k.packets[APEX_CP_INDIRECT] == indirect + 2 && *(uint64_t *)device.retire.map == 3002);
   for (unsigned i = 4096; i < 16384; i += 4)
      CHECK(*(uint32_t *)(map[1] + i) == 0x5a5a5a5a);

   /* A queue fault reported on the status page loses the device. */
   ioctls = k.ioctls;
   mock_set_status32(&k, APEX_STATUS_STATE, APEX_STATUS_ENABLED | APEX_STATUS_FAULTED);
   CHECK(v->QueueSubmit2(queue, 1, &submit, VK_NULL_HANDLE) == VK_ERROR_DEVICE_LOST);
   CHECK(k.ioctls == ioctls + 1 && vk_device_is_lost(&device.vk));
   /* After loss the driver keeps VAs bound; the kernel reclaims at close. */
   k.bound_close = true;

   v->FreeCommandBuffers(dev, command_pool, 1, &second);
   v->DestroyCommandPool(dev, command_pool, NULL);
   v->DestroyDescriptorPool(dev, descriptor_pool, NULL);
   v->DestroyPipeline(dev, pipeline, NULL);
   v->DestroyPipelineLayout(dev, layout, NULL);
   v->DestroyDescriptorSetLayout(dev, set_layout, NULL);
   v->DestroyQueryPool(dev, pool, NULL);
   v->DestroyEvent(dev, event, NULL);
   v->DestroyBuffer(dev, local, NULL);
   v->DestroyBuffer(dev, system, NULL);
   for (unsigned t = 0; t < 2; t++)
      v->FreeMemory(dev, memory[t], NULL);
   apex_device_finish(&device);
   CHECK(!k.queue && !k.live);
   mock_kernel_finish(&k);
   vk_physical_device_finish(&physical);
   vk_instance_finish(&instance);
   puts("PASS Apex CP device: queue creation, packets from vkCmd*, patched dispatch state and table, "
        "doorbell-only steady-state submission over ring wraps, multi-IB batch, fault to device loss");
}

int
main(int argc, char **argv)
{
   CHECK(argc == 2);
   test_encoder();
   test_ring();
   test_device(argv[1]);
   return 0;
}
