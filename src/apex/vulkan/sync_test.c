/* SPDX-License-Identifier: MIT */
#include "vk_alloc.h"
#include "vk_common_entrypoints.h"
#include "vk_device.h"
#include "vk_drm_syncobj.h"
#include "vk_instance.h"
#include "vk_physical_device.h"
#include "vk_queue.h"
#include "vk_fence.h"
#include "vk_semaphore.h"
#include "util/os_time.h"
#include "drm-uapi/drm.h"
#include <errno.h>
#include <fcntl.h>
#include <semaphore.h>
#include <stdio.h>
#include <sys/mman.h>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { \
   fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); abort(); \
} } while (0)

/* Exercise the shared runtime without a GPU. The provider verifies wait
 * deadlines/flags and injects loss; it does not model GPU completion. */
static struct vk_device *waiting_device;
static bool wait_success, inject_loss;
static unsigned wait_flags, wait_calls;
static int64_t last_timeout;
static unsigned reset_calls;
static bool reject_fd, checking_external;

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
export_fd(struct util_sync_provider *p, uint32_t handle, int *fd)
{
   CHECK(handle == 19);
   *fd = memfd_create("mock-sync-payload", MFD_CLOEXEC);
   CHECK(*fd >= 0);
   return 0;
}

static int
import_fd(struct util_sync_provider *p, int fd, uint32_t *handle)
{
   CHECK(fcntl(fd, F_GETFD) >= 0);
   if (reject_fd) { errno = EINVAL; return -1; }
   *handle = 19;
   return 0;
}

static int
import_file(struct util_sync_provider *p, uint32_t handle, int fd)
{
   CHECK(handle == 19);
   return import_fd(p, fd, &handle);
}

static int
reset(struct util_sync_provider *p, const uint32_t *handles, uint32_t count)
{
   CHECK(count == 1 && handles[0] == 19);
   reset_calls++;
   return 0;
}

/* FD ownership and copy/reference transference use the real runtime. The
 * provider only records API operations; it supplies no completion evidence. */
static void
external_sync(VkDevice device)
{
   VkExportFenceCreateInfo export_fence = {.sType = VK_STRUCTURE_TYPE_EXPORT_FENCE_CREATE_INFO,
      .handleTypes = VK_EXTERNAL_FENCE_HANDLE_TYPE_OPAQUE_FD_BIT | VK_EXTERNAL_FENCE_HANDLE_TYPE_SYNC_FD_BIT};
   VkFenceCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, .pNext = &export_fence,
      .flags = VK_FENCE_CREATE_SIGNALED_BIT};
   VkFence fence;
   CHECK(vk_common_CreateFence(device, &fi, NULL, &fence) == VK_SUCCESS);
   VkFenceGetFdInfoKHR get_fence = {.sType = VK_STRUCTURE_TYPE_FENCE_GET_FD_INFO_KHR,
      .fence = fence, .handleType = VK_EXTERNAL_FENCE_HANDLE_TYPE_OPAQUE_FD_BIT};
   int fd;
   CHECK(vk_common_GetFenceFdKHR(device, &get_fence, &fd) == VK_SUCCESS);
   CHECK(!reset_calls && (fcntl(fd, F_GETFD) & FD_CLOEXEC));
   VkImportFenceFdInfoKHR import_fence = {.sType = VK_STRUCTURE_TYPE_IMPORT_FENCE_FD_INFO_KHR,
      .fence = fence, .handleType = get_fence.handleType, .fd = fd};
   reject_fd = true;
   CHECK(vk_common_ImportFenceFdKHR(device, &import_fence) != VK_SUCCESS && fcntl(fd, F_GETFD) >= 0);
   reject_fd = false;
   CHECK(vk_common_ImportFenceFdKHR(device, &import_fence) == VK_SUCCESS);
   CHECK(fcntl(fd, F_GETFD) == -1 && errno == EBADF);
   get_fence.handleType = VK_EXTERNAL_FENCE_HANDLE_TYPE_SYNC_FD_BIT;
   CHECK(vk_common_GetFenceFdKHR(device, &get_fence, &fd) == VK_SUCCESS && reset_calls == 1);
   import_fence.handleType = get_fence.handleType;
   import_fence.flags = VK_FENCE_IMPORT_TEMPORARY_BIT;
   import_fence.fd = fd;
   CHECK(vk_common_ImportFenceFdKHR(device, &import_fence) == VK_SUCCESS);
   CHECK(fcntl(fd, F_GETFD) == -1 && errno == EBADF);
   CHECK(vk_fence_from_handle(fence)->temporary);
   CHECK(vk_common_ResetFences(device, 1, &fence) == VK_SUCCESS && reset_calls == 2);
   CHECK(!vk_fence_from_handle(fence)->temporary);
   vk_common_DestroyFence(device, fence, NULL);

   VkExportSemaphoreCreateInfo export_sem = {.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO,
      .handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT | VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT};
   VkSemaphoreCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &export_sem};
   VkSemaphore semaphore;
   CHECK(vk_common_CreateSemaphore(device, &si, NULL, &semaphore) == VK_SUCCESS);
   VkSemaphoreGetFdInfoKHR get_sem = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR,
      .semaphore = semaphore, .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT};
   CHECK(vk_common_GetSemaphoreFdKHR(device, &get_sem, &fd) == VK_SUCCESS && reset_calls == 2);
   VkImportSemaphoreFdInfoKHR import_sem = {.sType = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR,
      .semaphore = semaphore, .handleType = get_sem.handleType, .fd = fd};
   reject_fd = true;
   CHECK(vk_common_ImportSemaphoreFdKHR(device, &import_sem) != VK_SUCCESS && fcntl(fd, F_GETFD) >= 0);
   reject_fd = false;
   CHECK(vk_common_ImportSemaphoreFdKHR(device, &import_sem) == VK_SUCCESS);
   CHECK(fcntl(fd, F_GETFD) == -1 && errno == EBADF);
   get_sem.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
   CHECK(vk_common_GetSemaphoreFdKHR(device, &get_sem, &fd) == VK_SUCCESS && reset_calls == 3);
   import_sem.handleType = get_sem.handleType;
   import_sem.flags = VK_SEMAPHORE_IMPORT_TEMPORARY_BIT;
   import_sem.fd = fd;
   CHECK(vk_common_ImportSemaphoreFdKHR(device, &import_sem) == VK_SUCCESS);
   CHECK(fcntl(fd, F_GETFD) == -1 && errno == EBADF);
   CHECK(vk_semaphore_from_handle(semaphore)->temporary);
   CHECK(vk_common_GetSemaphoreFdKHR(device, &get_sem, &fd) == VK_SUCCESS && reset_calls == 3);
   CHECK(!vk_semaphore_from_handle(semaphore)->temporary && !close(fd));
   vk_common_DestroySemaphore(device, semaphore, NULL);
}

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
   if (checking_external) {
      CHECK(count == 1 && points[0] == 0 && (flags & DRM_SYNCOBJ_WAIT_FLAGS_WAIT_AVAILABLE));
   } else {
      CHECK(points[0] == 73);
      if (count == 2) CHECK(points[1] == 91);
   }
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
      .handle_to_fd = export_fd, .fd_to_handle = import_fd,
      .export_sync_file = export_fd, .import_sync_file = import_file, .reset = reset,
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
   checking_external = true;
   external_sync(vk_device_to_handle(&device));
   checking_external = false;
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
   puts("PASS Mesa sync: FD ownership/transference, temporary restoration, deadlines, ANY/ALL, pending, loss, sleeping drain (mock provider)");
   return 0;
}
