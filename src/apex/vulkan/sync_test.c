/* SPDX-License-Identifier: MIT */
#include "vk_alloc.h"
#include "vk_common_entrypoints.h"
#include "vk_device.h"
#include "vk_drm_syncobj.h"
#include "vk_instance.h"
#include "vk_physical_device.h"
#include "vk_queue.h"
#include "util/os_time.h"
#include "drm-uapi/drm.h"
#include <errno.h>
#include <semaphore.h>
#include <stdio.h>

#define CHECK(x) do { if (!(x)) { \
   fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); abort(); \
} } while (0)

/* Exercise the shared runtime without a GPU. The provider verifies wait
 * deadlines/flags and injects loss; it does not model GPU completion. */
static struct vk_device *waiting_device;
static bool wait_success, inject_loss;
static unsigned wait_flags, wait_calls;
static int64_t last_timeout;

static int
create(struct util_sync_provider *p, uint32_t flags, uint32_t *handle)
{
   *handle = 19;
   return 0;
}

static int
destroy(struct util_sync_provider *p, uint32_t handle)
{
   CHECK(handle == 19);
   return 0;
}

static void finalize(struct util_sync_provider *p) { }

static int
wait_sync(struct util_sync_provider *p, uint32_t *handles, unsigned count,
          int64_t timeout, unsigned flags, uint32_t *first)
{
   CHECK(count >= 1 && count <= 2 && handles[0] == 19);
   if (count == 2) CHECK(handles[1] == 19);
   CHECK(timeout >= 0 && timeout <= os_time_get_nano() + 100000000ull);
   wait_calls++;
   wait_flags = flags;
   last_timeout = timeout;
   if (wait_success)
      return 0;
   if (inject_loss)
      CHECK(vk_device_set_lost(waiting_device, "injected wait loss") == VK_ERROR_DEVICE_LOST);
   else {
      int64_t now = os_time_get_nano();
      if (timeout > now)
         os_time_sleep((timeout - now) / 1000 + 1);
   }
   errno = ETIME;
   return -1;
}

static int
wait_timeline(struct util_sync_provider *p, uint32_t *handles, uint64_t *points,
              unsigned count, int64_t timeout, unsigned flags, uint32_t *first)
{
   CHECK(points[0] == 73);
   if (count == 2) CHECK(points[1] == 91);
   return wait_sync(p, handles, count, timeout, flags, first);
}

static struct vk_queue *draining_queue;
static sem_t entered, release_submit, drain_sleeping;

int __real_cnd_wait(cnd_t *cond, mtx_t *mutex);
int __wrap_cnd_wait(cnd_t *cond, mtx_t *mutex);
int
__wrap_cnd_wait(cnd_t *cond, mtx_t *mutex)
{
   if (draining_queue && cond == &draining_queue->submit.pop)
      CHECK(!sem_post(&drain_sleeping));
   return __real_cnd_wait(cond, mutex);
}

static VkResult
fail_submit(struct vk_queue *queue, struct vk_queue_submit *submit)
{
   CHECK(!sem_post(&entered));
   CHECK(!sem_wait(&release_submit));
   return vk_queue_set_lost(queue, "injected submit failure");
}

static int
finish_queue(void *queue)
{
   vk_queue_finish(queue);
   return 0;
}

int main(void)
{
   struct vk_instance instance;
   const struct vk_instance_extension_table extensions = {0};
   const struct vk_instance_dispatch_table instance_dispatch = {0};
   const VkInstanceCreateInfo instance_info = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
   CHECK(vk_instance_init(&instance, &extensions, &instance_dispatch,
                         &instance_info, vk_default_allocator()) == VK_SUCCESS);
   (void)vk_instance_to_handle(&instance);
   struct vk_physical_device physical;
   const struct vk_physical_device_dispatch_table physical_dispatch = {0};
   const struct vk_properties properties = {0};
   CHECK(vk_physical_device_init(&physical, &instance, NULL, NULL,
                                &properties, &physical_dispatch) == VK_SUCCESS);
   (void)vk_physical_device_to_handle(&physical);
   struct util_sync_provider provider = {
      .create = create, .destroy = destroy, .wait = wait_sync,
      .timeline_wait = wait_timeline, .finalize = finalize,
   };
   wait_success = true;
   struct vk_sync_type type = vk_drm_syncobj_get_type_from_provider(&provider);
   const struct vk_sync_type *types[] = {&type, NULL};
   physical.supported_sync_types = types;
   struct vk_device device;
   const struct vk_device_dispatch_table dispatch = {0};
   const VkDeviceCreateInfo info = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
   CHECK(vk_device_init(&device, &physical, &dispatch, &info, NULL) == VK_SUCCESS);
   (void)vk_device_to_handle(&device);
   device.sync = &provider;
   waiting_device = &device;
   struct vk_sync *sync;
   CHECK(vk_sync_create(&device, &type, VK_SYNC_IS_TIMELINE, 0, &sync) == VK_SUCCESS);
   struct vk_sync_wait wait = {.sync = sync, .wait_value = 73};
   wait_success = false;
   CHECK(vk_sync_wait_many(&device, 1, &wait, VK_SYNC_WAIT_COMPLETE, 0) == VK_TIMEOUT);
   CHECK(last_timeout == 0 && wait_flags ==
         (DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT | DRM_SYNCOBJ_WAIT_FLAGS_WAIT_ALL));
   unsigned before = wait_calls;
   uint64_t deadline = os_time_get_nano() + 150000000ull;
   struct vk_sync_wait any[] = {wait, {.sync = sync, .wait_value = 91}};
   CHECK(vk_sync_wait_many(&device, 2, any, VK_SYNC_WAIT_ANY, deadline) == VK_TIMEOUT);
   CHECK(wait_calls >= before + 2 && last_timeout == deadline);
   CHECK(wait_flags == DRM_SYNCOBJ_WAIT_FLAGS_WAIT_FOR_SUBMIT);
   wait_success = true;
   CHECK(vk_sync_wait_many(&device, 1, &wait, VK_SYNC_WAIT_PENDING, 0) == VK_SUCCESS);
   CHECK(wait_flags & DRM_SYNCOBJ_WAIT_FLAGS_WAIT_AVAILABLE);
   wait_success = false;
   inject_loss = true;
   CHECK(vk_sync_wait_many(&device, 1, &wait, VK_SYNC_WAIT_COMPLETE, UINT64_MAX) == VK_ERROR_DEVICE_LOST);
   wait_success = true;
   CHECK(vk_sync_wait_many(&device, 1, &wait, VK_SYNC_WAIT_COMPLETE, 0) == VK_SUCCESS);
   vk_sync_destroy(&device, sync);
   vk_device_finish(&device);

   CHECK(vk_device_init(&device, &physical, &dispatch, &info, NULL) == VK_SUCCESS);
   (void)vk_device_to_handle(&device);
   vk_device_enable_threaded_submit(&device);
   struct vk_queue queue;
   const float priority = 1.0f;
   const VkDeviceQueueCreateInfo queue_info = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueCount = 1, .pQueuePriorities = &priority,
   };
   CHECK(vk_queue_init(&queue, &device, &queue_info, 0) == VK_SUCCESS);
   queue.driver_submit = fail_submit;
   CHECK(!sem_init(&entered, 0, 0) && !sem_init(&release_submit, 0, 0) &&
         !sem_init(&drain_sleeping, 0, 0));
   draining_queue = &queue;
   CHECK(vk_queue_enable_submit_thread(&queue) == VK_SUCCESS);
   const VkSubmitInfo2 submit = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
   CHECK(vk_common_QueueSubmit2(vk_queue_to_handle(&queue), 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS);
   CHECK(!sem_wait(&entered));
   /* Keep a trailing submit behind the failure and a drain already asleep.
    * The failure must wake the drain, join, then free both retained submits. */
   CHECK(vk_common_QueueSubmit2(vk_queue_to_handle(&queue), 1, &submit, VK_NULL_HANDLE) == VK_SUCCESS);
   thrd_t finish;
   CHECK(thrd_create(&finish, finish_queue, &queue) == thrd_success);
   CHECK(!sem_wait(&drain_sleeping));
   CHECK(!sem_post(&release_submit));
   int result;
   CHECK(thrd_join(finish, &result) == thrd_success && !result);
   draining_queue = NULL;
   CHECK(vk_device_is_lost(&device));
   CHECK(!sem_destroy(&entered) && !sem_destroy(&release_submit) && !sem_destroy(&drain_sleeping));
   vk_device_finish(&device);
   vk_physical_device_finish(&physical);
   vk_instance_finish(&instance);
   puts("PASS Mesa sync recovery: deadlines, ANY/ALL, pending, loss, zero probe, sleeping drain, retained submits");
   return 0;
}
