/* SPDX-License-Identifier: MIT */
#include "apex_device.h"
#include "apex_pipeline.h"
#include "drm-uapi/apex_drm.h"
#include "vk_alloc.h"
#include "vk_buffer.h"
#include "vk_command_buffer.h"
#include "vk_drm_syncobj.h"
#include "vk_instance.h"
#include "vk_physical_device.h"
#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <sys/mman.h>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { \
   fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); abort(); \
} } while (0)

static unsigned calls, input_index, output_index, objects, live, fail, fail_at;
static uint64_t published, completed;
static bool terminal, timed_out, failed, grid_test, update_test;
static struct { uint64_t size, va; bool live; unsigned uploads; } gems[32];
static unsigned queries, waits, dispatch_index;
static unsigned grid_limit;

int __wrap_ioctl(int fd, unsigned long request, ...);
int __wrap_ioctl(int fd, unsigned long request, ...)
{
   va_list ap;
   va_start(ap, request);
   void *arg = va_arg(ap, void *);
   va_end(ap);
   if (request == DRM_IOCTL_APEX_INFO) {
      *(struct drm_apex_info *)arg = (struct drm_apex_info) {
         .version = 2, .capabilities = APEX_DRM_CAP_SHMEM | APEX_DRM_CAP_GPUVM | APEX_DRM_CAP_ASYNC,
      };
   } else if (request == DRM_IOCTL_SYNCOBJ_CREATE) {
      ((struct drm_syncobj_create *)arg)->handle = 99;
   } else if (request == DRM_IOCTL_SYNCOBJ_DESTROY) {
      CHECK(((struct drm_syncobj_destroy *)arg)->handle == 99);
   } else if (request == DRM_IOCTL_SYNCOBJ_QUERY) {
      struct drm_syncobj_timeline_array *r = arg;
      CHECK(r->count_handles == 1 && *(uint32_t *)(uintptr_t)r->handles == 99 && !r->flags);
      *(uint64_t *)(uintptr_t)r->points = completed;
   } else if (request == DRM_IOCTL_APEX_VM_STATUS) {
      struct drm_apex_vm_status *r = arg;
      CHECK(!r->error && !r->reserved);
      queries++;
      r->error = terminal ? -EIO : 0;
      errno = EIO; /* A successful status query need not preserve wait errno. */
   } else if (request == DRM_IOCTL_APEX_GEM_CREATE) {
      struct drm_apex_gem_create *r = arg;
      CHECK(completed == published && objects < 31); /* All preparation precedes waits. */
      r->handle = ++objects;
      gems[objects].size = r->size;
      gems[objects].live = true;
      live++;
      CHECK(!ftruncate(fd, (objects + 1) * 4096));
   } else if (request == DRM_IOCTL_APEX_GEM_MMAP) {
      struct drm_apex_gem_mmap *r = arg;
      r->offset = r->handle * 4096;
   } else if (request == DRM_IOCTL_APEX_VM_BIND) {
      struct drm_apex_vm_bind *r = arg;
      if (r->operation == APEX_DRM_VM_BIND_MAP) {
         CHECK(completed == published && gems[r->handle].live);
         gems[r->handle].va = r->va;
      } else {
         /* UNMAP waits the file's tail. After a partial enqueue failure it
          * would block cleanup behind an unsignaled dependency. */
         CHECK(!failed && r->operation == APEX_DRM_VM_BIND_UNMAP && completed == published);
         unsigned h;
         for (h = 1; h <= objects && gems[h].va != r->va; h++);
         CHECK(h <= objects && r->bytes == gems[h].size);
         gems[h].va = 0;
      }
   } else if (request == DRM_IOCTL_APEX_GEM_TRANSFER) {
      struct drm_apex_gem_transfer *r = arg;
      CHECK(completed == published && r->direction == APEX_DRM_TRANSFER_TO_LOCAL && gems[r->handle].live);
      gems[r->handle].uploads++;
      if (update_test && r->handle <= 2) {
         CHECK(r->bytes == (r->handle == 1 ? 52 : 60));
         uint32_t words[15];
         CHECK(pread(fd, words, r->bytes, r->handle * 4096) == r->bytes);
         for (unsigned i = 0; i < r->bytes / 4; i++)
            CHECK(words[i] == 0xa7010000 + r->handle * 123 + i * 37);
      }
   } else if (request == DRM_IOCTL_GEM_CLOSE) {
      struct drm_gem_close *r = arg;
      CHECK(gems[r->handle].live && (!gems[r->handle].va || failed));
      gems[r->handle].live = false;
      live--;
   } else {
      CHECK(request == DRM_IOCTL_APEX_VM_SUBMIT);
      struct drm_apex_vm_submit *r = arg;
      calls++;
      if (fail && (!fail_at || calls == fail_at)) { failed = true; errno = fail; return -1; }
      CHECK(!r->reserved && r->input_count <= 16 && r->output_count >= 1 && r->output_count <= 16);
      const struct drm_apex_sync *in = (void *)(uintptr_t)r->inputs;
      const struct drm_apex_sync *out = (void *)(uintptr_t)r->outputs;
      for (unsigned i = 0; i < r->input_count; i++, input_index++) {
         CHECK(in[i].handle == 200 + input_index && !in[i].flags);
         CHECK(in[i].point == (input_index % 2 ? 0 : 37 + input_index));
      }
      CHECK(out[0].handle == 99 && !out[0].flags && out[0].point == ++published);
      for (unsigned i = 1; i < r->output_count; i++, output_index++) {
         CHECK(out[i].handle == 300 + output_index && !out[i].flags);
         CHECK(out[i].point == (output_index % 2 ? 0 : 101 + output_index));
      }
      if (r->flags == APEX_DRM_SUBMIT_SYNC_ONLY) {
         CHECK(!r->program_va && !r->program_bytes && !r->data_va && !r->workgroups);
      } else if (update_test) {
         unsigned region = dispatch_index % 2;
         unsigned table = 4 + dispatch_index;
         CHECK(!r->flags && !r->input_count && r->output_count == 1);
         CHECK(r->program_va == gems[3].va && r->data_va == gems[table].va && r->workgroups == 1);
         CHECK(gems[1].uploads == 1 && gems[2].uploads == 1);
         uint32_t words[14];
         CHECK(pread(fd, words, sizeof(words), table * 4096) == sizeof(words));
         for (unsigned i = 0; i < 14; i++) words[i] = util_le32_to_cpu(words[i]);
         for (unsigned i = 0; i < 8; i++) CHECK(!words[i]);
         CHECK(((uint64_t)words[9] << 32 | words[8]) == gems[region + 1].va);
         CHECK(((uint64_t)words[11] << 32 | words[10]) == 0x30000004 + region * 64);
         CHECK(words[12] == (region ? 60 : 52));
         dispatch_index++;
      } else {
         CHECK(!r->flags && !r->input_count && r->output_count == 1);
         CHECK(objects == (grid_test ? 14 : 3) && live == objects && r->program_va == gems[1].va);
         CHECK(r->data_va == gems[dispatch_index + 2].va);
         uint32_t parameters[6];
         CHECK(pread(fd, parameters, sizeof(parameters), (dispatch_index + 2) * 4096 + 32) == sizeof(parameters));
         for (unsigned i = 0; i < 6; i++) parameters[i] = util_le32_to_cpu(parameters[i]);
         if (grid_test && dispatch_index < 12) {
            CHECK(r->workgroups == (dispatch_index % 2 ? 1 : grid_limit));
            CHECK(parameters[0] == (dispatch_index % 2 ? grid_limit : 0));
            CHECK(parameters[1] == dispatch_index / 2 % 2 && parameters[2] == dispatch_index / 4);
            CHECK(parameters[3] == grid_limit + 1 && parameters[4] == 2 && parameters[5] == 3);
         } else {
            CHECK(r->workgroups == (grid_test ? 2 : 1));
            CHECK(!parameters[0] && !parameters[1] && !parameters[2]);
            CHECK(parameters[3] == (grid_test ? 2 : 1) && parameters[4] == 1 && parameters[5] == 1);
         }
         dispatch_index++;
         char magic[4];
         CHECK(pread(fd, magic, 4, 4096) == 4 && !memcmp(magic, "APX2", 4));
      }
   }
   return 0;
}

static int create(struct util_sync_provider *p, uint32_t flags, uint32_t *h) { *h = 19; return 0; }
static int destroy(struct util_sync_provider *p, uint32_t h) { CHECK(h == 19); return 0; }
static void finalize(struct util_sync_provider *p) { }
static int wait_sync(struct util_sync_provider *p, uint32_t *h, unsigned n,
                     int64_t timeout, unsigned flags, uint32_t *first)
{
   waits++;
   if (!timed_out) return 0;
   errno = ETIME;
   return -1;
}
static int wait_timeline(struct util_sync_provider *p, uint32_t *h, uint64_t *points,
                         unsigned n, int64_t timeout, unsigned flags, uint32_t *first)
{
   return wait_sync(p, h, n, timeout, flags, first);
}
static int query(struct util_sync_provider *p, uint32_t *h, uint64_t *points, uint32_t n, uint32_t flags)
{
   CHECK(n == 1); *points = 113; return 0;
}

int main(int argc, char **argv)
{
   CHECK(argc == 2);
   FILE *f = fopen(argv[1], "rb");
   CHECK(f && !fseek(f, 0, SEEK_END));
   long bytes = ftell(f);
   CHECK(bytes > 0);
   rewind(f);
   uint32_t *spirv = malloc(bytes);
   CHECK(spirv && fread(spirv, 1, bytes, f) == bytes && !fclose(f));
   struct vk_instance instance;
   const struct vk_instance_extension_table extensions = {0};
   const struct vk_instance_dispatch_table instance_dispatch = {0};
   const VkInstanceCreateInfo instance_info = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
   CHECK(vk_instance_init(&instance, &extensions, &instance_dispatch, &instance_info,
                         vk_default_allocator()) == VK_SUCCESS);
   (void)vk_instance_to_handle(&instance);
   struct vk_physical_device physical;
   const struct vk_physical_device_dispatch_table physical_dispatch = {0};
   const struct vk_properties properties = {.subgroupSize = 16, .minSubgroupSize = 16, .maxSubgroupSize = 16,
      .maxComputeWorkGroupCount = {1024, 1, 1}, .maxComputeWorkGroupSize = {16, 1, 1}};
   CHECK(vk_physical_device_init(&physical, &instance, NULL, NULL, &properties,
                                &physical_dispatch) == VK_SUCCESS);
   (void)vk_physical_device_to_handle(&physical);
   struct util_sync_provider provider = {.create = create, .destroy = destroy, .finalize = finalize,
      .wait = wait_sync, .timeline_wait = wait_timeline, .query = query};
   struct vk_sync_type type = vk_drm_syncobj_get_type_from_provider(&provider);
   const struct vk_sync_type *types[] = {&type, NULL};
   physical.supported_sync_types = types;
   for (unsigned test = 0; test < 16; test++) {
      calls = input_index = output_index = objects = live = fail = fail_at = queries = waits = dispatch_index = 0;
      published = completed = 0;
      terminal = timed_out = failed = false;
      grid_test = test == 10 || test == 11;
      update_test = test >= 12;
      grid_limit = test == 11 ? 3 : 1024;
      memset(gems, 0, sizeof(gems));
      int fd = memfd_create("apex-async", MFD_CLOEXEC);
      CHECK(fd >= 0);
      struct apex_device device;
      const float priority = 1;
      VkDeviceQueueCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
         .queueCount = 1, .pQueuePriorities = &priority};
      VkDeviceCreateInfo di = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
         .queueCreateInfoCount = 1, .pQueueCreateInfos = &qi};
      CHECK(apex_device_init(&device, &physical, &di, NULL, fd, APEX_TRANSPORT_DRM) == VK_SUCCESS);
      device.vk.sync->finalize(device.vk.sync);
      device.vk.sync = &provider;
      VkDevice dev = apex_device_to_handle(&device);
      const struct vk_device_dispatch_table *v = &device.vk.dispatch_table;
      if (update_test) {
         VkBuffer buffer;
         VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = 256, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
         CHECK(v->CreateBuffer(dev, &bi, NULL, &buffer) == VK_SUCCESS);
         vk_buffer_from_handle(buffer)->device_address = 0x30000000;
         VkCommandPool pool;
         VkCommandPoolCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT};
         CHECK(v->CreateCommandPool(dev, &ci, NULL, &pool) == VK_SUCCESS);
         VkCommandBuffer cb;
         VkCommandBufferAllocateInfo ca = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = pool, .commandBufferCount = 1};
         CHECK(v->AllocateCommandBuffers(dev, &ca, &cb) == VK_SUCCESS);
         VkCommandBuffer primary = cb, secondary = VK_NULL_HANDLE;
         if (test >= 14) {
            ca.level = VK_COMMAND_BUFFER_LEVEL_SECONDARY;
            CHECK(v->AllocateCommandBuffers(dev, &ca, &secondary) == VK_SUCCESS);
            cb = secondary;
         }
         /* First discard an unsubmitted upload, then record immutable sources. */
         for (unsigned record = 0; record < 2; record++) {
            VkCommandBufferInheritanceInfo inheritance = {
               .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO,
            };
            VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
               .pInheritanceInfo = secondary ? &inheritance : NULL};
            CHECK(v->BeginCommandBuffer(cb, &begin) == VK_SUCCESS);
            uint32_t data[15];
            for (unsigned region = 0; region < 2; region++) {
               for (unsigned i = 0; i < 15; i++) data[i] = 0xa7010000 + (region + 1) * 123 + i * 37;
               v->CmdUpdateBuffer(cb, buffer, 4 + region * 64, region ? 60 : 52, data);
               memset(data, 0xcc, sizeof(data));
            }
            CHECK(v->EndCommandBuffer(cb) == VK_SUCCESS && !objects);
            if (!record) CHECK(v->ResetCommandBuffer(cb, 0) == VK_SUCCESS);
         }
         if (secondary) {
            VkCommandBufferBeginInfo begin = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            CHECK(v->BeginCommandBuffer(primary, &begin) == VK_SUCCESS);
            v->CmdExecuteCommands(primary, 1, &secondary);
            CHECK(v->EndCommandBuffer(primary) == VK_SUCCESS && !objects);
            cb = primary;
         }
         struct vk_command_buffer *command = vk_command_buffer_from_handle(cb);
         struct vk_queue_submit submit = {.command_buffer_count = 1, .command_buffers = &command};
         if (test == 13 || test == 15) {
            fail = ENOMEM;
            fail_at = 2;
            CHECK(device.queue.driver_submit(&device.queue, &submit) == VK_ERROR_DEVICE_LOST);
            CHECK(calls == 2 && published == 1 && live == 4);
         } else {
            for (unsigned repeat = 0; repeat < 2; repeat++) {
               CHECK(device.queue.driver_submit(&device.queue, &submit) == VK_SUCCESS);
               CHECK(dispatch_index == (repeat + 1) * 2 && calls == (repeat + 1) * 3);
               completed = published;
            }
            CHECK(objects == 7 && gems[1].uploads == 1 && gems[2].uploads == 1);
         }
         v->DestroyCommandPool(dev, pool, NULL);
         CHECK(!gems[1].live && !gems[2].live);
         v->DestroyBuffer(dev, buffer, NULL);
      } else if (test && test < 9) {
         if (test == 1 || test == 8) {
            struct vk_queue_submit empty = {0};
            fail = test == 1 ? EINTR : EBUSY;
            CHECK(device.queue.driver_submit(&device.queue, &empty) == VK_ERROR_DEVICE_LOST);
            CHECK(calls == 1 && !published); /* Never duplicate an interrupted enqueue. */
         } else if (test == 2 || test == 3 || test == 5) {
            VkFence fence;
            VkFenceCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            CHECK(v->CreateFence(dev, &fi, NULL, &fence) == VK_SUCCESS);
            terminal = true;
            timed_out = test == 3;
            CHECK((test == 2 ? v->GetFenceStatus(dev, fence) :
                   v->WaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX)) == VK_ERROR_DEVICE_LOST);
            CHECK(queries && waits == 1);
            v->DestroyFence(dev, fence, NULL);
         } else if (test == 4) {
            VkSemaphore semaphore;
            VkSemaphoreTypeCreateInfo ti = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
               .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE};
            VkSemaphoreCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &ti};
            CHECK(v->CreateSemaphore(dev, &si, NULL, &semaphore) == VK_SUCCESS);
            uint64_t value;
            terminal = true;
            CHECK(v->GetSemaphoreCounterValue(dev, semaphore, &value) == VK_ERROR_DEVICE_LOST && queries);
            v->DestroySemaphore(dev, semaphore, NULL);
         } else {
            struct vk_sync *sync;
            CHECK(vk_sync_create(&device.vk, &type, VK_SYNC_IS_TIMELINE, 0, &sync) == VK_SUCCESS);
            terminal = test == 6;
            timed_out = true;
            VkResult result = vk_sync_wait(&device.vk, sync, 79, VK_SYNC_WAIT_PENDING,
                                          terminal ? UINT64_MAX : 0);
            CHECK(result == (terminal ? VK_ERROR_DEVICE_LOST : VK_TIMEOUT));
            CHECK(queries && waits == 1);
            vk_sync_destroy(&device.vk, sync);
         }
      } else {
         VkPipelineLayout layout;
         VkPipelineLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
         CHECK(v->CreatePipelineLayout(dev, &li, NULL, &layout) == VK_SUCCESS);
         VkShaderModule shader;
         VkShaderModuleCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = bytes, .pCode = spirv};
         CHECK(v->CreateShaderModule(dev, &si, NULL, &shader) == VK_SUCCESS);
         VkPipeline pipeline;
         VkComputePipelineCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .layout = layout, .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
               .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = shader, .pName = "wide"}};
         CHECK(v->CreateComputePipelines(dev, VK_NULL_HANDLE, 1, &pi, NULL, &pipeline) == VK_SUCCESS);
         /* Exercise a smaller per-pipeline private-storage capacity. The
          * compiler fixture separately checks deriving it from padded lanes. */
         CHECK(apex_pipeline_from_handle(pipeline)->program.max_workgroups == 1024);
         apex_pipeline_from_handle(pipeline)->program.max_workgroups = grid_limit;
         VkCommandPool pool;
         VkCommandPoolCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT};
         CHECK(v->CreateCommandPool(dev, &ci, NULL, &pool) == VK_SUCCESS);
         VkCommandBuffer cb;
         VkCommandBufferAllocateInfo ca = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = pool, .commandBufferCount = 1};
         CHECK(v->AllocateCommandBuffers(dev, &ca, &cb) == VK_SUCCESS);
         VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
         if (grid_test) {
            for (unsigned axis = 0; axis < 3; axis++) {
               CHECK(v->BeginCommandBuffer(cb, &bi) == VK_SUCCESS);
               v->CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
               v->CmdDispatch(cb, axis == 0 ? 65536 : 1, axis == 1 ? 65536 : 1, axis == 2 ? 65536 : 1);
               CHECK(v->EndCommandBuffer(cb) == VK_ERROR_FEATURE_NOT_PRESENT);
               CHECK(v->ResetCommandBuffer(cb, 0) == VK_SUCCESS);
            }
         }
         CHECK(v->BeginCommandBuffer(cb, &bi) == VK_SUCCESS);
         v->CmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
         if (grid_test) {
            v->CmdDispatch(cb, 0, 2, 3);
            v->CmdDispatch(cb, 5, 0, 3);
            v->CmdDispatch(cb, 5, 2, 0);
         }
         v->CmdDispatch(cb, grid_test ? grid_limit + 1 : 1, grid_test ? 2 : 1, grid_test ? 3 : 1);
         v->CmdDispatch(cb, grid_test ? 2 : 1, 1, 1);
         CHECK(v->EndCommandBuffer(cb) == VK_SUCCESS);
         struct vk_drm_syncobj syncs[40];
         struct vk_sync_wait in[19];
         struct vk_sync_signal out[17];
         for (unsigned i = 0; i < 40; i++)
            syncs[i] = (struct vk_drm_syncobj) {.base = {.type = &type,
               .flags = i % 2 ? 0 : VK_SYNC_IS_TIMELINE}, .syncobj = i < 20 ? 200 + i : 280 + i};
         for (unsigned i = 0; i < 18; i++) in[i] = (struct vk_sync_wait) {
            .sync = &syncs[i].base, .wait_value = i % 2 ? 0 : 37 + i};
         in[18] = (struct vk_sync_wait) {.sync = &syncs[18].base}; /* Timeline zero is skipped. */
         for (unsigned i = 0; i < 17; i++) out[i] = (struct vk_sync_signal) {
            .sync = &syncs[20 + i].base, .signal_value = i % 2 ? 0 : 101 + i};
         struct vk_command_buffer *command = vk_command_buffer_from_handle(cb);
         struct vk_queue_submit submit = {.wait_count = 19, .waits = in,
            .signal_count = 17, .signals = out, .command_buffer_count = 1, .command_buffers = &command};
         if (test == 9) {
            fail = ENOMEM;
            fail_at = 4; /* Two waits and one shader accepted; next shader fails. */
            CHECK(device.queue.driver_submit(&device.queue, &submit) == VK_ERROR_DEVICE_LOST);
            CHECK(calls == 4 && published == 3 && !completed && live == 2);
         } else {
            CHECK(device.queue.driver_submit(&device.queue, &submit) == VK_SUCCESS);
            unsigned total = grid_test ? 17 : 6;
            CHECK(calls == total && published == total && input_index == 18 && output_index == 17 &&
                  live == (grid_test ? 14 : 3) && !waits);
            /* Reset/reuse cannot change a pending submission's tables. */
            CHECK(v->ResetCommandBuffer(cb, 0) == VK_SUCCESS);
            CHECK(v->BeginCommandBuffer(cb, &bi) == VK_SUCCESS);
            v->CmdDispatch(cb, 0, 1, 1);
            CHECK(v->EndCommandBuffer(cb) == VK_SUCCESS);
            struct vk_queue_submit empty = {0};
            CHECK(device.queue.driver_submit(&device.queue, &empty) == VK_SUCCESS);
            CHECK(live == (grid_test ? 14 : 3) && published == total + 1);
            completed = published;
            CHECK(device.queue.driver_submit(&device.queue, &empty) == VK_SUCCESS);
            CHECK(live == 1 && published == total + 2); /* Tables retired, program retained. */
            completed = published;
         }
         v->DestroyCommandPool(dev, pool, NULL);
         v->DestroyPipeline(dev, pipeline, NULL);
         v->DestroyShaderModule(dev, shader, NULL);
         v->DestroyPipelineLayout(dev, layout, NULL);
         CHECK(live == (test == 9 ? 1 : 0));
      }
      apex_device_finish(&device);
      CHECK(!live);
      CHECK(!close(fd));
   }
   free(spirv);
   vk_physical_device_finish(&physical);
   vk_instance_finish(&instance);
   puts("PASS Apex secondary updates: copied sources, reset/reuse, async retention and partial-enqueue cleanup (mock only)");
   puts("PASS Apex async transport: batched sync arrays, zero point, pending retention, retirement, backpressure, EINTR, fence/counter/kernel loss (mock only)");
   return 0;
}
