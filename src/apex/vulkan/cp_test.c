/* SPDX-License-Identifier: MIT */
/* Offline command-processor gate: packet encodings for every opcode, IB
 * growth, ring wrap and space accounting, the vkQueueSubmit batch sequence,
 * a device recorded through Vulkan whose ring the command-processor model in
 * mock_kernel.h executes, and a render pass whose pass record, draw
 * registers and state blocks the model reads back, and transform feedback
 * with its dispatches run on the ISA model. Steady-state submission must
 * issue no ioctl. */
#include "apex_private.h"
#include "apex_job.h"
#include "apex_pipeline.h"
#include "mock_kernel.h"
#include "vk_alloc.h"
#include "vk_buffer.h"
#include "vk_instance.h"
#include "vk_physical_device.h"
#include <math.h>
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
   apex_cp_predicate(&ib, 0x3000000a04ull, APEX_CP_PREDICATE_ENABLE | APEX_CP_PREDICATE_INVERTED);
   apex_cp_predicate(&ib, 0, 0);
   EXPECT(&ib, 0x00000363, 0xa04, 0x30, 3, 0x00000363, 0, 0, 0);
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

static struct {
   unsigned count;
   struct mock_draw draws[8];
} drawn;

static void
on_draw(struct mock_kernel *kernel, const struct mock_draw *d)
{
   if (drawn.count < ARRAY_SIZE(drawn.draws))
      drawn.draws[drawn.count] = *d;
   drawn.count++;
}

static VkShaderModule
shader_module(const struct vk_device_dispatch_table *v, VkDevice dev, const char *path)
{
   size_t size;
   uint32_t *code = load_spirv(path, &size);
   VkShaderModule module;
   const VkShaderModuleCreateInfo mi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = size, .pCode = code};
   CHECK(v->CreateShaderModule(dev, &mi, NULL, &module) == VK_SUCCESS);
   free(code);
   return module;
}

/* Runs a dispatch on the ISA model over the mock's GEMs of at most 2 MiB:
 * submission arenas, programs, buffers and query pools. */
static void
simulate_dispatch(struct mock_kernel *kernel, const struct mock_dispatch *d)
{
   uint64_t program_va = MOCK_U64(d->state, APEX_STATE_COMPUTE_PROGRAM);
   uint32_t header[16];
   mock_read(kernel, program_va, header, sizeof(header));
   size_t size = 64 + header[2];
   uint8_t *program = malloc(size);
   CHECK(program);
   mock_read(kernel, program_va, program, size);
   struct apex_sim_region regions[64];
   unsigned count = 0;
   for (unsigned h = 1; h <= kernel->gem_count; h++) {
      const struct mock_gem *g = &kernel->gems[h];
      if (!g->live || !g->va || g->size > (2u << 20))
         continue;
      CHECK(count < ARRAY_SIZE(regions));
      regions[count] = (struct apex_sim_region){g->va, malloc(g->size), g->size};
      mock_read(kernel, g->va, regions[count].data, g->size);
      count++;
   }
   char diagnostic[256] = "";
   if (apex_simulate(program, size, &d->state[APEX_STATE_COMPUTE_USER], d->grid,
                     MOCK_U64(d->state, APEX_STATE_COMPUTE_PRIVATE), NULL, regions, count, diagnostic)) {
      fprintf(stderr, "simulation: %s\n", diagnostic);
      abort();
   }
   for (unsigned r = 0; r < count; r++) {
      mock_write(kernel, regions[r].gpuva, regions[r].data, regions[r].size);
      free(regions[r].data);
   }
   free(program);
   seen.count++;
}

/* Transform feedback through the CP model with dispatches on the ISA
 * model: a two-instance triangle strip captured into two buffers, the
 * first of which holds five of its six triangles; the stream query and
 * counter buffers receive the primitives written and needed and the byte
 * offsets. Resumed from the counters into rebound buffers, an indexed
 * strip with a restart index appends its three triangles. */
static void
test_transform_feedback(const struct vk_device_dispatch_table *v, VkDevice dev, VkQueue queue,
                        VkDeviceMemory memory, VkImageView view, const char *vertex_path,
                        const char *fragment_path)
{
   const uint64_t memory_va = ((struct apex_memory *)apex_memory_from_handle(memory))->storage->bo.va;
   /* Vertex buffer at 192 KiB, capture buffers at 256 and 264 KiB, counters at 272 KiB. */
   const VkDeviceSize places[4] = {196608, 262144, 270336, 278528};
   VkBuffer buffers[4];
   for (unsigned i = 0; i < 4; i++) {
      const VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 4096,
         .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                  VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_BUFFER_BIT_EXT |
                  VK_BUFFER_USAGE_TRANSFORM_FEEDBACK_COUNTER_BUFFER_BIT_EXT};
      CHECK(v->CreateBuffer(dev, &bi, NULL, &buffers[i]) == VK_SUCCESS);
      CHECK(v->BindBufferMemory(dev, buffers[i], memory, places[i]) == VK_SUCCESS);
   }
   const uint8_t zero[4096] = {0};
   for (unsigned i = 1; i < 4; i++)
      mock_write(&k, memory_va + places[i], zero, sizeof(zero));
   /* Vertex i: position (i, 2i, -i, 1) and BGRA8 color (10i, 20 + i, 30 + i, 255). */
   for (unsigned i = 0; i < 6; i++) {
      float position[4] = {i, 2.0f * i, -(float)i, 1.0f};
      uint8_t color[4] = {10 * i, 20 + i, 30 + i, 255};
      mock_write(&k, memory_va + places[0] + 32 * i, position, 16);
      mock_write(&k, memory_va + places[0] + 32 * i + 16, color, 4);
   }
   const uint16_t indices[8] = {0, 1, 2, 0xffff, 3, 4, 5, 2};
   mock_write(&k, memory_va + places[0] + 1024, indices, sizeof(indices));

   VkShaderModule modules[2] = {shader_module(v, dev, vertex_path), shader_module(v, dev, fragment_path)};
   const VkPipelineShaderStageCreateInfo stages[2] = {
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = modules[0], .pName = "main"},
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = modules[1], .pName = "main"},
   };
   const VkVertexInputBindingDescription binding = {0, 32, VK_VERTEX_INPUT_RATE_VERTEX};
   const VkVertexInputAttributeDescription attributes[2] = {
      {0, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 0}, {1, 0, VK_FORMAT_B8G8R8A8_UNORM, 16},
   };
   const VkPipelineVertexInputStateCreateInfo vis = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      .vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = &binding,
      .vertexAttributeDescriptionCount = 2, .pVertexAttributeDescriptions = attributes,
   };
   const VkPipelineInputAssemblyStateCreateInfo ia = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP, .primitiveRestartEnable = VK_TRUE,
   };
   const VkViewport viewport = {0, 0, 100, 70, 0, 1};
   const VkRect2D scissor = {{0, 0}, {100, 70}};
   const VkPipelineViewportStateCreateInfo vp = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1, .pViewports = &viewport, .scissorCount = 1, .pScissors = &scissor,
   };
   const VkPipelineRasterizationStateCreateInfo rs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .rasterizerDiscardEnable = VK_TRUE, .lineWidth = 1.0f,
   };
   const VkPipelineMultisampleStateCreateInfo ms = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
   };
   const VkPipelineColorBlendAttachmentState blend = {.colorWriteMask = 0xf};
   const VkPipelineColorBlendStateCreateInfo cb = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1, .pAttachments = &blend,
   };
   const VkFormat color = VK_FORMAT_R8G8B8A8_UNORM;
   const VkPipelineRenderingCreateInfo rendering = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .colorAttachmentCount = 1, .pColorAttachmentFormats = &color,
   };
   const VkPipelineLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
   VkPipelineLayout layout;
   CHECK(v->CreatePipelineLayout(dev, &li, NULL, &layout) == VK_SUCCESS);
   const VkGraphicsPipelineCreateInfo pi = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &rendering,
      .stageCount = 2, .pStages = stages, .pVertexInputState = &vis, .pInputAssemblyState = &ia,
      .pViewportState = &vp, .pRasterizationState = &rs, .pMultisampleState = &ms,
      .pColorBlendState = &cb, .layout = layout,
   };
   VkPipeline pipeline;
   CHECK(v->CreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &pi, NULL, &pipeline) == VK_SUCCESS);
   const VkQueryPoolCreateInfo qpi = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
      .queryType = VK_QUERY_TYPE_TRANSFORM_FEEDBACK_STREAM_EXT, .queryCount = 1};
   VkQueryPool pool;
   CHECK(v->CreateQueryPool(dev, &qpi, NULL, &pool) == VK_SUCCESS);

   VkCommandPool command_pool;
   const VkCommandPoolCreateInfo cpi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
   CHECK(v->CreateCommandPool(dev, &cpi, NULL, &command_pool) == VK_SUCCESS);
   const VkCommandBufferAllocateInfo cai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
      .commandPool = command_pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
   VkCommandBuffer cmd;
   CHECK(v->AllocateCommandBuffers(dev, &cai, &cmd) == VK_SUCCESS);
   const VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
   CHECK(v->BeginCommandBuffer(cmd, &begin) == VK_SUCCESS);
   v->CmdResetQueryPool(cmd, pool, 0, 1);
   const VkRenderingAttachmentInfo color_attachment = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = view,
      .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
   const VkRenderingInfo render = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = {{0, 0}, {100, 70}}, .layerCount = 1, .colorAttachmentCount = 1,
      .pColorAttachments = &color_attachment};
   v->CmdBeginRendering(cmd, &render);
   v->CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
   const VkDeviceSize vertex_offset = 0;
   v->CmdBindVertexBuffers(cmd, 0, 1, &buffers[0], &vertex_offset);
   const VkDeviceSize offsets[2] = {0, 0}, sizes[2] = {500, VK_WHOLE_SIZE}, counter_offsets[2] = {0, 4};
   v->CmdBindTransformFeedbackBuffersEXT(cmd, 0, 2, &buffers[1], offsets, sizes);
   v->CmdBeginQueryIndexedEXT(cmd, pool, 0, 0, 0);
   v->CmdBeginTransformFeedbackEXT(cmd, 0, 0, NULL, NULL);
   v->CmdDraw(cmd, 5, 2, 1, 0);
   const VkBuffer counters[2] = {buffers[3], buffers[3]};
   v->CmdEndTransformFeedbackEXT(cmd, 0, 2, counters, counter_offsets);
   v->CmdEndQueryIndexedEXT(cmd, pool, 0, 0);
   const VkDeviceSize whole[2] = {VK_WHOLE_SIZE, VK_WHOLE_SIZE};
   v->CmdBindTransformFeedbackBuffersEXT(cmd, 0, 2, &buffers[1], offsets, whole);
   v->CmdBindIndexBuffer(cmd, buffers[0], 1024, VK_INDEX_TYPE_UINT16);
   v->CmdBeginTransformFeedbackEXT(cmd, 0, 2, counters, counter_offsets);
   v->CmdDrawIndexed(cmd, 8, 1, 0, 0, 0);
   v->CmdEndTransformFeedbackEXT(cmd, 0, 2, counters, counter_offsets);
   v->CmdEndRendering(cmd);
   CHECK(v->EndCommandBuffer(cmd) == VK_SUCCESS);
   const VkCommandBufferSubmitInfo cbi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
      .commandBuffer = cmd};
   const VkSubmitInfo2 submit = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
      .commandBufferInfoCount = 1, .pCommandBufferInfos = &cbi};
   seen.count = 0;
   k.on_dispatch = simulate_dispatch;
   CHECK(v->QueueSubmit2(queue, 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS);
   mock_cp_run(&k);
   k.on_dispatch = NULL;
   /* Capture, bookkeeping and the counter store; then the counter load too. */
   CHECK(!k.blocked && seen.count == 7);

   /* Output vertex o is vertex o mod 3 of strip triangle t = o / 3 mod 3,
    * in Vulkan's order {t, t + 1 + t % 2, t + 2 - t % 2} from firstVertex 1.
   * The sixth triangle does not fit the first buffer, so neither buffer
    * stores it. The indexed strip's runs are {0, 1, 2} and {3, 4, 5, 2}. */
   const unsigned restarted[9] = {0, 1, 2, 3, 4, 5, 4, 2, 5};
   float captured[8], pair[2];
   for (unsigned o = 0; o < 25; o++) {
      unsigned t = o / 3 % 3, j = o % 3, instance = o < 15 ? o / 9 : 0;
      unsigned vertex = o >= 15 && o < 24 ? restarted[o - 15] :
                        1 + (j == 0 ? t : j == 1 ? t + 1 + t % 2 : t + 2 - t % 2);
      mock_read(&k, memory_va + places[1] + 32 * o, captured, sizeof(captured));
      mock_read(&k, memory_va + places[2] + 16 * o + 4, pair, sizeof(pair));
      if (o == 24) {
         for (unsigned c = 0; c < 8; c++)
            CHECK(captured[c] == 0.0f);
         CHECK(pair[0] == 0.0f && pair[1] == 0.0f);
         continue;
      }
      const float shade[4] = {(30 + vertex) / 255.0f, (20 + vertex) / 255.0f, 10 * vertex / 255.0f, 1.0f};
      const float position[4] = {vertex, 2.0f * vertex, -(float)vertex, 1.0f};
      for (unsigned c = 0; c < 4; c++)
         CHECK(fabsf(captured[c] - shade[c]) < 1e-6f && captured[4 + c] == position[c]);
      CHECK(pair[0] == position[0] + vertex && pair[1] == position[1] + 100.0f * instance);
   }
   uint32_t counted[2];
   mock_read(&k, memory_va + places[3], counted, sizeof(counted));
   CHECK(counted[0] == 24 * 32 && counted[1] == 24 * 16);
   uint64_t results[3];
   CHECK(v->GetQueryPoolResults(dev, pool, 0, 1, sizeof(results), results, sizeof(results),
                                VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT) == VK_SUCCESS);
   CHECK(results[0] == 5 && results[1] == 6 && results[2] == 1);
   CHECK(v->QueueWaitIdle(queue) == VK_SUCCESS);

   v->FreeCommandBuffers(dev, command_pool, 1, &cmd);
   v->DestroyCommandPool(dev, command_pool, NULL);
   v->DestroyQueryPool(dev, pool, NULL);
   v->DestroyPipeline(dev, pipeline, NULL);
   v->DestroyPipelineLayout(dev, layout, NULL);
   for (unsigned i = 0; i < 2; i++)
      v->DestroyShaderModule(dev, modules[i], NULL);
   for (unsigned i = 0; i < 4; i++)
      v->DestroyBuffer(dev, buffers[i], NULL);
}

/* A render pass and its draws through Vulkan, executed by the CP model:
 * the pass record in the bin pool, the draw registers, the state blocks
 * they name and the draw packets (Docs/architecture.md, Command processor
 * and user-mode rings; Fixed-function graphics). */
static void
test_graphics(const char *vertex_path, const char *fragment_path, const char *xfb_path)
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
   const struct vk_properties properties = {.subgroupSize = 16, .minSubgroupSize = 16, .maxSubgroupSize = 16};
   const struct vk_features features = {.dynamicRendering = true, .transformFeedback = true};
   CHECK(vk_physical_device_init(&physical, &instance, NULL, &features, &properties,
                                 &physical_dispatch) == VK_SUCCESS);
   (void)vk_physical_device_to_handle(&physical);
   mock_kernel_init(&k);
   k.on_draw = on_draw;
   struct apex_device device;
   const float priority = 1;
   const VkDeviceQueueCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueCount = 1, .pQueuePriorities = &priority};
   const VkDeviceCreateInfo di = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qi};
   CHECK(apex_device_init(&device, &physical, &di, NULL, k.fd) == VK_SUCCESS);
   VkDevice dev = apex_device_to_handle(&device);
   const struct vk_device_dispatch_table *v = &device.vk.dispatch_table;
   VkQueue queue;
   v->GetDeviceQueue(dev, 0, 0, &queue);

   /* A 100 x 70 RGBA8 target and a D24S8 depth-stencil image, tiled. */
   const VkImageCreateInfo color_info = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
      .imageType = VK_IMAGE_TYPE_2D, .format = VK_FORMAT_R8G8B8A8_UNORM, .extent = {100, 70, 1},
      .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT};
   VkImageCreateInfo depth_info = color_info;
   depth_info.format = VK_FORMAT_D24_UNORM_S8_UINT;
   depth_info.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
   VkImage images[2];
   CHECK(v->CreateImage(dev, &color_info, NULL, &images[0]) == VK_SUCCESS);
   CHECK(v->CreateImage(dev, &depth_info, NULL, &images[1]) == VK_SUCCESS);
   VkMemoryRequirements req;
   v->GetImageMemoryRequirements(dev, images[0], &req);
   /* 100 x 70 texels of 4 bytes: 4 x 3 tiles of 32 x 32. */
   CHECK(req.size == 12 * 4096 && req.alignment == 4096);
   const VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = 1 << 20, .memoryTypeIndex = 0};
   VkDeviceMemory memory;
   CHECK(v->AllocateMemory(dev, &ai, NULL, &memory) == VK_SUCCESS);
   CHECK(v->BindImageMemory(dev, images[0], memory, 0) == VK_SUCCESS);
   CHECK(v->BindImageMemory(dev, images[1], memory, 65536) == VK_SUCCESS);
   const uint64_t memory_va = ((struct apex_memory *)apex_memory_from_handle(memory))->storage->bo.va;
   VkImageView views[2];
   for (unsigned i = 0; i < 2; i++) {
      const VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
         .image = images[i], .viewType = VK_IMAGE_VIEW_TYPE_2D,
         .format = i ? VK_FORMAT_D24_UNORM_S8_UINT : VK_FORMAT_R8G8B8A8_UNORM,
         .subresourceRange = {i ? VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT :
                              VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
      CHECK(v->CreateImageView(dev, &vi, NULL, &views[i]) == VK_SUCCESS);
   }
   VkBuffer vertices;
   const VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 4096,
      .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
               VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT};
   CHECK(v->CreateBuffer(dev, &bi, NULL, &vertices) == VK_SUCCESS);
   CHECK(v->BindBufferMemory(dev, vertices, memory, 131072) == VK_SUCCESS);
   const uint64_t buffer_va = memory_va + 131072;

   VkShaderModule modules[2] = {shader_module(v, dev, vertex_path), shader_module(v, dev, fragment_path)};
   const VkPipelineShaderStageCreateInfo stages[2] = {
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = modules[0], .pName = "main"},
      {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
       .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = modules[1], .pName = "main"},
   };
   const VkVertexInputBindingDescription binding = {0, 32, VK_VERTEX_INPUT_RATE_VERTEX};
   const VkVertexInputAttributeDescription attributes[2] = {
      {0, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 0}, {1, 0, VK_FORMAT_R8G8B8A8_UNORM, 16},
   };
   const VkPipelineVertexInputStateCreateInfo vis = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
      .vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = &binding,
      .vertexAttributeDescriptionCount = 2, .pVertexAttributeDescriptions = attributes,
   };
   const VkPipelineInputAssemblyStateCreateInfo ia = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
      .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP, .primitiveRestartEnable = VK_TRUE,
   };
   const VkViewport viewport = {4, 6, 80, 60, 0.25f, 0.75f};
   const VkRect2D scissor = {{2, 3}, {90, 50}};
   const VkPipelineViewportStateCreateInfo vp = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
      .viewportCount = 1, .pViewports = &viewport, .scissorCount = 1, .pScissors = &scissor,
   };
   const VkPipelineRasterizationStateCreateInfo rs = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
      .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_BACK_BIT,
      .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1.0f,
   };
   const VkPipelineMultisampleStateCreateInfo ms = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
      .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
   };
   const VkPipelineDepthStencilStateCreateInfo dss = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
      .depthTestEnable = VK_TRUE, .depthWriteEnable = VK_TRUE, .depthCompareOp = VK_COMPARE_OP_LESS,
   };
   const VkPipelineColorBlendAttachmentState blend = {
      .blendEnable = VK_TRUE, .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
      .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, .colorBlendOp = VK_BLEND_OP_ADD,
      .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE, .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
      .alphaBlendOp = VK_BLEND_OP_ADD, .colorWriteMask = 0xf,
   };
   const VkPipelineColorBlendStateCreateInfo cb = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
      .attachmentCount = 1, .pAttachments = &blend,
   };
   const VkFormat color = VK_FORMAT_R8G8B8A8_UNORM;
   const VkPipelineRenderingCreateInfo rendering = {
      .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
      .colorAttachmentCount = 1, .pColorAttachmentFormats = &color,
      .depthAttachmentFormat = VK_FORMAT_D24_UNORM_S8_UINT, .stencilAttachmentFormat = VK_FORMAT_D24_UNORM_S8_UINT,
   };
   const VkPipelineLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
   VkPipelineLayout layout;
   CHECK(v->CreatePipelineLayout(dev, &li, NULL, &layout) == VK_SUCCESS);
   const VkGraphicsPipelineCreateInfo pi = {
      .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &rendering,
      .stageCount = 2, .pStages = stages, .pVertexInputState = &vis, .pInputAssemblyState = &ia,
      .pViewportState = &vp, .pRasterizationState = &rs, .pMultisampleState = &ms,
      .pDepthStencilState = &dss, .pColorBlendState = &cb, .layout = layout,
   };
   VkPipeline pipeline;
   CHECK(v->CreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &pi, NULL, &pipeline) == VK_SUCCESS);
   const VkQueryPoolCreateInfo qpi = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
      .queryType = VK_QUERY_TYPE_OCCLUSION, .queryCount = 2};
   VkQueryPool pool;
   CHECK(v->CreateQueryPool(dev, &qpi, NULL, &pool) == VK_SUCCESS);

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
   const VkRenderingAttachmentInfo color_attachment = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = views[0],
      .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_STORE, .clearValue.color.float32 = {1.0f, 0.0f, 0.5f, 1.0f}};
   const VkRenderingAttachmentInfo depth_attachment = {
      .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = views[1],
      .imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL, .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
      .storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE, .clearValue.depthStencil = {1.0f, 0x5a}};
   const VkRenderingInfo render = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
      .renderArea = {{0, 0}, {100, 70}}, .layerCount = 1, .colorAttachmentCount = 1,
      .pColorAttachments = &color_attachment, .pDepthAttachment = &depth_attachment,
      .pStencilAttachment = &depth_attachment};
   v->CmdBeginRendering(cmd, &render);
   v->CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
   const VkDeviceSize offset = 64;
   v->CmdBindVertexBuffers(cmd, 0, 1, &vertices, &offset);
   v->CmdBeginQuery(cmd, pool, 1, 0);
   v->CmdDraw(cmd, 4, 2, 1, 3);
   v->CmdEndQuery(cmd, pool, 1);
   v->CmdBindIndexBuffer(cmd, vertices, 1024, VK_INDEX_TYPE_UINT16);
   v->CmdDrawIndexed(cmd, 6, 1, 2, -1, 0);
   v->CmdDrawIndirect(cmd, vertices, 2048, 3, 16);
   v->CmdEndRendering(cmd);
   CHECK(v->EndCommandBuffer(cmd) == VK_SUCCESS);
   const VkCommandBufferSubmitInfo cbi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
      .commandBuffer = cmd};
   const VkSubmitInfo2 submit = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2,
      .commandBufferInfoCount = 1, .pCommandBufferInfos = &cbi};
   CHECK(v->QueueSubmit2(queue, 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS);
   mock_cp_run(&k);
   CHECK(!k.blocked && !k.in_pass && k.packets[APEX_CP_BEGIN_PASS] == 1 && k.packets[APEX_CP_END_PASS] == 1);
   CHECK(k.packets[APEX_CP_QUERY_BEGIN] == 1 && k.packets[APEX_CP_QUERY_END] == 1 && drawn.count == 3);

   /* The pass record in the bin pool: pool, 32 MiB, area, one layer, 64 x
    * 64 tiles of one sample, the regions; color attachment 0 cleared and
    * stored, tiled at 4 tiles per row; the depth-stencil attachment
    * cleared, not stored. */
   const uint64_t pool_va = device.bin_pool.va;
   uint32_t record[128];
   mock_read(&k, pool_va, record, sizeof(record));
   CHECK(MOCK_U64(record, 0) == pool_va && record[2] == APEX_BIN_POOL_BYTES && record[3] == 0 &&
         record[4] == (100 | 70u << 16) && record[5] == 1 && record[7] == 0);
   CHECK(record[8] == APEX_POOL_DRAW_BYTES && record[9] == APEX_POOL_VERTEX_BYTES &&
         record[10] == APEX_POOL_PRIMITIVE_BYTES);
   const uint32_t *a = &record[16];
   CHECK(a[0] == (1 | 1u << 1 | 1u << 5 | 3u << 16) && a[1] == (uint32_t)memory_va &&
         a[2] == ((uint32_t)(memory_va >> 32) | 4u << 16) && a[3] == 12 * 4096 / 64 && a[8] == 12 * 4096 / 64);
   CHECK(a[4] == (0xff | 0x80u << 16 | 0xffu << 24));
   a = &record[16 + 12 * 8];
   CHECK(a[0] == (1 | 1u << 1 | 1u << 3 | 1u << 5 | 1u << 8 | 3u << 16) && a[1] == (uint32_t)(memory_va + 65536));
   CHECK(a[4] == 0x3f800000 && a[6] == (0xffffff | 0x5au << 24));

   /* Draw 0: the vertex registers, the vertex-input block and the program. */
   const struct mock_draw *d = &drawn.draws[0];
   CHECK(d->op == APEX_CP_DRAW && d->payload[0] == 4 && d->payload[1] == 2 && d->payload[2] == 1 &&
         d->payload[3] == 3);
   uint64_t vs = MOCK_U64(d->state, APEX_STATE_VERTEX), fs = MOCK_U64(d->state, APEX_STATE_FRAGMENT);
   CHECK(!(vs & 63) && !(fs & 63) && mock_read32(&k, vs) == 0x50585041 && mock_read32(&k, fs) == 0x50585041);
   CHECK((mock_read32(&k, vs + 4) & 0xff) == 1 && (mock_read32(&k, fs + 4) & 0xff) == 2);
   uint32_t vi[128];
   mock_read(&k, MOCK_U64(d->state, APEX_STATE_VERTEX + 2), vi, sizeof(vi));
   CHECK(vi[0] == (uint32_t)(buffer_va + 64) && vi[1] == (((buffer_va + 64) >> 32) | 32u << 16) &&
         vi[2] == 4096 - 64);
   CHECK(vi[64] == ((1u << 2 | 3) << 8) && vi[65] == 2 && vi[66] == ((4u << 2 | 3) << 8 | 16u << 16) &&
         vi[67] == 6 && vi[96] == 0);
   /* Topology TRIANGLE_STRIP; restart applies to indexed draws. */
   CHECK(d->state[APEX_STATE_VERTEX + 8] == 4 && !d->state[APEX_STATE_VERTEX + 9]);
   /* Fragment registers: raster, depth-stencil, blend and viewport blocks. */
   uint32_t raster[16], ds[16], blend_block[64], viewport_block[256];
   mock_read(&k, MOCK_U64(d->state, APEX_STATE_FRAGMENT + 2), raster, sizeof(raster));
   mock_read(&k, MOCK_U64(d->state, APEX_STATE_FRAGMENT + 4), ds, sizeof(ds));
   mock_read(&k, MOCK_U64(d->state, APEX_STATE_FRAGMENT + 6), blend_block, sizeof(blend_block));
   mock_read(&k, MOCK_U64(d->state, APEX_STATE_FRAGMENT + 8), viewport_block, sizeof(viewport_block));
   CHECK(raster[0] == (2 | 1u << 2 | 1u << 9) && raster[1] == 1);
   CHECK(ds[0] == (1 | 2 | 1u << 2 | 1u << 8) && ds[2] == 0x33800000);
   CHECK(blend_block[0] == 0 && blend_block[1] == (0xf | 0u << 4 | 1u << 9 | 7u << 17 | 6u << 22 | 1u << 30));
   CHECK(viewport_block[0] == 0x40800000 && viewport_block[3] == 0x42700000 && viewport_block[4] == 0x3e800000 &&
         viewport_block[128] == (2 | 3u << 16) && viewport_block[129] == (92 | 53u << 16));
   CHECK(d->state[APEX_STATE_DYNAMIC + 12] == 1 && d->state[APEX_STATE_DYNAMIC + 11] == 0x3f800000);
   /* Both stages read one table; the vertex stage's view index is 0. */
   uint64_t table = MOCK_U64(d->state, APEX_STATE_VERTEX_USER);
   CHECK(table && table == MOCK_U64(d->state, APEX_STATE_FRAGMENT_USER) && !d->state[APEX_STATE_VERTEX_USER + 2]);

   /* Draw 1: indexed, 16-bit indices from byte 1024 with restart. */
   d = &drawn.draws[1];
   CHECK(d->op == APEX_CP_DRAW_INDEXED && d->payload[0] == 6 && d->payload[2] == 2 &&
         d->payload[3] == 0xffffffff && d->payload[4] == 0);
   CHECK(MOCK_U64(d->state, APEX_STATE_VERTEX + 4) == buffer_va + 1024 &&
         d->state[APEX_STATE_VERTEX + 6] == 4096 - 1024 && d->state[APEX_STATE_VERTEX + 7] == 1 &&
         d->state[APEX_STATE_VERTEX + 9] == 1 && d->state[APEX_STATE_VERTEX + 10] == 0xffffffff);
   /* Draw 2: three commands from the buffer at its 16-byte stride. */
   d = &drawn.draws[2];
   CHECK(d->op == APEX_CP_DRAW_INDIRECT && MOCK_U64(d->payload, 0) == buffer_va + 2048 &&
         d->payload[2] == 3 && d->payload[3] == 16 && !MOCK_U64(d->payload, 4));
   /* The query's availability follows END_PASS. */
   CHECK(mock_read32(&k, ((struct apex_query_pool *)apex_query_pool_from_handle(pool))->bo.va + APEX_QUERY_STRIDE + APEX_QUERY_AVAILABLE) == 1);
   CHECK(v->QueueWaitIdle(queue) == VK_SUCCESS);

   /* Conditional rendering: under a zero predicate the draw and the
    * application's attachment clear skip while the query copy, driver work,
    * runs with PREDICATE off; an inverted nonzero predicate skips, a plain
    * one draws. Each PREDICATE that turns skipping on reads the value. */
   const uint32_t predicates[2] = {0, 1};
   mock_write(&k, buffer_va + 3072, predicates, sizeof(predicates));
   CHECK(v->ResetCommandBuffer(cmd, 0) == VK_SUCCESS);
   CHECK(v->BeginCommandBuffer(cmd, &begin) == VK_SUCCESS);
   const VkConditionalRenderingBeginInfoEXT conditions[3] = {
      {.sType = VK_STRUCTURE_TYPE_CONDITIONAL_RENDERING_BEGIN_INFO_EXT, .buffer = vertices, .offset = 3072},
      {.sType = VK_STRUCTURE_TYPE_CONDITIONAL_RENDERING_BEGIN_INFO_EXT, .buffer = vertices, .offset = 3076,
       .flags = VK_CONDITIONAL_RENDERING_INVERTED_BIT_EXT},
      {.sType = VK_STRUCTURE_TYPE_CONDITIONAL_RENDERING_BEGIN_INFO_EXT, .buffer = vertices, .offset = 3076},
   };
   for (unsigned c = 0; c < 3; c++) {
      v->CmdBeginConditionalRenderingEXT(cmd, &conditions[c]);
      v->CmdBeginRendering(cmd, &render);
      v->CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
      v->CmdBindVertexBuffers(cmd, 0, 1, &vertices, &offset);
      v->CmdDraw(cmd, 4, 1, 0, 0);
      if (c == 0) {
         const VkClearAttachment clear = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .clearValue.color.float32 = {0, 1, 0, 1}};
         const VkClearRect rect = {.rect = {{0, 0}, {16, 16}}, .layerCount = 1};
         v->CmdClearAttachments(cmd, 1, &clear, 1, &rect);
      }
      v->CmdEndRendering(cmd);
      if (c == 0)
         v->CmdCopyQueryPoolResults(cmd, pool, 0, 2, vertices, 3584, 8, 0);
      v->CmdEndConditionalRenderingEXT(cmd);
   }
   CHECK(v->EndCommandBuffer(cmd) == VK_SUCCESS);
   unsigned draws_before = 0, dispatches = k.packets[APEX_CP_DISPATCH], drawn_before = drawn.count;
   unsigned predicates_before = k.packets[APEX_CP_PREDICATE], skipped = k.skipped;
   for (unsigned op = APEX_CP_DRAW; op <= APEX_CP_DRAW_INDEXED_INDIRECT; op++)
      draws_before += k.packets[op];
   CHECK(v->QueueSubmit2(queue, 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS);
   mock_cp_run(&k);
   unsigned draws = 0;
   for (unsigned op = APEX_CP_DRAW; op <= APEX_CP_DRAW_INDEXED_INDIRECT; op++)
      draws += k.packets[op];
   draws -= draws_before;
   /* The skipped draws: the first, the clear's (at least one) and the
    * second; the third draws. On, off for the query copy, on and off,
    * on and off. */
   CHECK(!k.blocked && !k.predicate && draws >= 4 && k.skipped - skipped == draws - 1 &&
         drawn.count == drawn_before + 1 && k.packets[APEX_CP_DISPATCH] > dispatches &&
         k.packets[APEX_CP_PREDICATE] - predicates_before == 6);
   CHECK(v->QueueWaitIdle(queue) == VK_SUCCESS);
   test_transform_feedback(v, dev, queue, memory, views[0], xfb_path, fragment_path);

   v->FreeCommandBuffers(dev, command_pool, 1, &cmd);
   v->DestroyCommandPool(dev, command_pool, NULL);
   v->DestroyQueryPool(dev, pool, NULL);
   v->DestroyPipeline(dev, pipeline, NULL);
   v->DestroyPipelineLayout(dev, layout, NULL);
   for (unsigned i = 0; i < 2; i++) {
      v->DestroyShaderModule(dev, modules[i], NULL);
      v->DestroyImageView(dev, views[i], NULL);
      v->DestroyImage(dev, images[i], NULL);
   }
   v->DestroyBuffer(dev, vertices, NULL);
   v->FreeMemory(dev, memory, NULL);
   apex_device_finish(&device);
   CHECK(!k.queue && !k.live);
   mock_kernel_finish(&k);
   vk_physical_device_finish(&physical);
   vk_instance_finish(&instance);
   puts("PASS Apex CP graphics: pass record and attachments in the bin pool, draw registers, vertex-input, "
        "raster, depth-stencil, blend and viewport blocks, DRAW, DRAW_INDEXED, DRAW_INDIRECT, queries, "
        "conditional rendering, transform feedback capture on the ISA model");
}

int
main(int argc, char **argv)
{
   CHECK(argc == 5);
   test_encoder();
   test_ring();
   test_device(argv[1]);
   test_graphics(argv[2], argv[3], argv[4]);
   return 0;
}
