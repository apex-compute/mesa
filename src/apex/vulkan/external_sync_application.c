/* SPDX-License-Identifier: MIT */
/* Explicit public-loader gate with real cross-device DRM payloads. */
#include <vulkan/vulkan.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { \
   fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); abort(); \
} } while (0)
#define LOAD(name) PFN_vk##name name = (PFN_vk##name)gipa(instance, "vk" #name); CHECK(name)

static void consumed(int fd)
{
   if (fd >= 0) CHECK(fcntl(fd, F_GETFD) == -1 && errno == EBADF);
}

int main(int argc, char **argv)
{
   CHECK(argc == 2 && geteuid() != 0);
   CHECK(!setenv("VK_DRIVER_FILES", argv[1], 1) && !setenv("APEX_DEVELOPMENT", "1", 1));
   void *loader = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
   CHECK(loader);
   PFN_vkGetInstanceProcAddr gipa = (PFN_vkGetInstanceProcAddr)dlsym(loader, "vkGetInstanceProcAddr");
   CHECK(gipa);
   VkInstance instance = VK_NULL_HANDLE;
   LOAD(CreateInstance);
   const char *instance_extensions[] = {VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
      VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME, VK_KHR_EXTERNAL_FENCE_CAPABILITIES_EXTENSION_NAME};
   VkInstanceCreateInfo ii = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
      .enabledExtensionCount = 3, .ppEnabledExtensionNames = instance_extensions};
   CHECK(CreateInstance(&ii, NULL, &instance) == VK_SUCCESS);
   LOAD(DestroyInstance);
   LOAD(EnumeratePhysicalDevices);
   uint32_t count = 0;
   CHECK(EnumeratePhysicalDevices(instance, &count, NULL) == VK_SUCCESS);
   if (!count) {
      DestroyInstance(instance, NULL);
      dlclose(loader);
      puts("SKIP Apex external sync: no compatible render device");
      return 77;
   }
   VkPhysicalDevice physical;
   CHECK(count == 1 && EnumeratePhysicalDevices(instance, &count, &physical) == VK_SUCCESS);
   LOAD(CreateDevice);
   LOAD(DestroyDevice);
   LOAD(GetDeviceQueue);
   LOAD(CreateSemaphore);
   LOAD(DestroySemaphore);
   LOAD(SignalSemaphoreKHR);
   LOAD(GetSemaphoreCounterValueKHR);
   LOAD(GetSemaphoreFdKHR);
   LOAD(ImportSemaphoreFdKHR);
   LOAD(CreateFence);
   LOAD(DestroyFence);
   LOAD(GetFenceStatus);
   LOAD(WaitForFences);
   LOAD(ResetFences);
   LOAD(GetFenceFdKHR);
   LOAD(ImportFenceFdKHR);
   LOAD(QueueSubmit);
   LOAD(QueueWaitIdle);
   const char *device_extensions[] = {VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,
      VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME, VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
      VK_KHR_EXTERNAL_FENCE_EXTENSION_NAME, VK_KHR_EXTERNAL_FENCE_FD_EXTENSION_NAME};
   VkPhysicalDeviceTimelineSemaphoreFeatures feature = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES, .timelineSemaphore = VK_TRUE};
   float priority = 1;
   VkDeviceQueueCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueCount = 1, .pQueuePriorities = &priority};
   VkDeviceCreateInfo di = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &feature,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qi,
      .enabledExtensionCount = 5, .ppEnabledExtensionNames = device_extensions};
   VkDevice device[2];
   VkQueue queue[2];
   for (unsigned i = 0; i < 2; i++) {
      CHECK(CreateDevice(physical, &di, NULL, &device[i]) == VK_SUCCESS);
      GetDeviceQueue(device[i], 0, 0, &queue[i]);
   }
   VkExportSemaphoreCreateInfo export_sem = {.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO,
      .handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT};
   VkSemaphoreTypeCreateInfo timeline_type = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
      .pNext = &export_sem, .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE};
   VkSemaphoreCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &timeline_type};
   VkSemaphore timeline[2], binary[2];
   for (unsigned i = 0; i < 2; i++) CHECK(CreateSemaphore(device[i], &si, NULL, &timeline[i]) == VK_SUCCESS);
   VkSemaphoreGetFdInfoKHR get_sem = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR,
      .semaphore = timeline[0], .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT};
   int fd;
   CHECK(GetSemaphoreFdKHR(device[0], &get_sem, &fd) == VK_SUCCESS && fd >= 0);
   CHECK(fcntl(fd, F_GETFD) & FD_CLOEXEC);
   VkImportSemaphoreFdInfoKHR import_sem = {.sType = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR,
      .semaphore = timeline[1], .handleType = get_sem.handleType, .fd = fd};
   CHECK(ImportSemaphoreFdKHR(device[1], &import_sem) == VK_SUCCESS);
   consumed(fd);
   export_sem.handleTypes |= VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
   si.pNext = &export_sem;
   for (unsigned i = 0; i < 2; i++) CHECK(CreateSemaphore(device[i], &si, NULL, &binary[i]) == VK_SUCCESS);
   get_sem.semaphore = binary[1];
   CHECK(GetSemaphoreFdKHR(device[1], &get_sem, &fd) == VK_SUCCESS && fd >= 0);
   import_sem.semaphore = binary[0];
   import_sem.fd = fd;
   CHECK(ImportSemaphoreFdKHR(device[0], &import_sem) == VK_SUCCESS);
   consumed(fd);
   VkExportFenceCreateInfo export_fence = {.sType = VK_STRUCTURE_TYPE_EXPORT_FENCE_CREATE_INFO,
      .handleTypes = VK_EXTERNAL_FENCE_HANDLE_TYPE_OPAQUE_FD_BIT | VK_EXTERNAL_FENCE_HANDLE_TYPE_SYNC_FD_BIT};
   VkFenceCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, .pNext = &export_fence};
   VkFence fence[2];
   for (unsigned i = 0; i < 2; i++) CHECK(CreateFence(device[i], &fi, NULL, &fence[i]) == VK_SUCCESS);
   const VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
   const uint64_t values[] = {7, 19, 37};
   for (unsigned pass = 0; pass < 3; pass++) {
      uint64_t zero = 0;
      VkTimelineSemaphoreSubmitInfo ti = {.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
         .waitSemaphoreValueCount = 1, .pWaitSemaphoreValues = &values[pass],
         .signalSemaphoreValueCount = 1, .pSignalSemaphoreValues = &zero};
      VkSubmitInfo producer = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .pNext = &ti,
         .waitSemaphoreCount = 1, .pWaitSemaphores = &timeline[1], .pWaitDstStageMask = &stage,
         .signalSemaphoreCount = 1, .pSignalSemaphores = &binary[1]};
      VkSubmitInfo consumer = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
         .waitSemaphoreCount = 1, .pWaitSemaphores = &binary[0], .pWaitDstStageMask = &stage};
      CHECK(QueueSubmit(queue[1], 1, &producer, fence[1]) == VK_SUCCESS);
      if (pass != 1) CHECK(QueueSubmit(queue[0], 1, &consumer, fence[0]) == VK_SUCCESS);
      CHECK(WaitForFences(device[1], 1, &fence[1], VK_TRUE, 1000000) == VK_TIMEOUT);
      CHECK(GetFenceStatus(device[0], fence[0]) == VK_NOT_READY);
      VkSemaphoreSignalInfo signal = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO,
         .semaphore = timeline[0], .value = values[pass]};
      CHECK(SignalSemaphoreKHR(device[0], &signal) == VK_SUCCESS);
      CHECK(WaitForFences(device[1], 1, &fence[1], VK_TRUE, 10000000000ull) == VK_SUCCESS);
      if (pass == 1) {
         get_sem.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
         CHECK(GetSemaphoreFdKHR(device[1], &get_sem, &fd) == VK_SUCCESS);
         import_sem.handleType = get_sem.handleType;
         import_sem.flags = VK_SEMAPHORE_IMPORT_TEMPORARY_BIT;
         import_sem.fd = fd;
         CHECK(ImportSemaphoreFdKHR(device[0], &import_sem) == VK_SUCCESS);
         consumed(fd);
         CHECK(QueueSubmit(queue[0], 1, &consumer, fence[0]) == VK_SUCCESS);
      }
      CHECK(WaitForFences(device[0], 1, &fence[0], VK_TRUE, 10000000000ull) == VK_SUCCESS);
      uint64_t actual;
      CHECK(GetSemaphoreCounterValueKHR(device[1], timeline[1], &actual) == VK_SUCCESS && actual == values[pass]);
      if (pass != 2)
         for (unsigned i = 0; i < 2; i++) CHECK(ResetFences(device[i], 1, &fence[i]) == VK_SUCCESS);
   }
   /* Opaque fence resets affect both aliases; sync-file export copies the
    * completed payload and resets the source. Temporary import restores it. */
   VkFence alias;
   CHECK(CreateFence(device[0], &fi, NULL, &alias) == VK_SUCCESS);
   VkFenceGetFdInfoKHR get_fence = {.sType = VK_STRUCTURE_TYPE_FENCE_GET_FD_INFO_KHR,
      .fence = fence[1], .handleType = VK_EXTERNAL_FENCE_HANDLE_TYPE_OPAQUE_FD_BIT};
   CHECK(GetFenceFdKHR(device[1], &get_fence, &fd) == VK_SUCCESS && fd >= 0);
   VkImportFenceFdInfoKHR import_fence = {.sType = VK_STRUCTURE_TYPE_IMPORT_FENCE_FD_INFO_KHR,
      .fence = alias, .handleType = get_fence.handleType, .fd = fd};
   CHECK(ImportFenceFdKHR(device[0], &import_fence) == VK_SUCCESS);
   consumed(fd);
   CHECK(GetFenceStatus(device[0], alias) == VK_SUCCESS);
   CHECK(ResetFences(device[0], 1, &alias) == VK_SUCCESS);
   CHECK(GetFenceStatus(device[1], fence[1]) == VK_NOT_READY);
   VkSubmitInfo empty = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO};
   CHECK(QueueSubmit(queue[1], 1, &empty, fence[1]) == VK_SUCCESS);
   CHECK(WaitForFences(device[1], 1, &fence[1], VK_TRUE, 10000000000ull) == VK_SUCCESS);
   get_fence.handleType = VK_EXTERNAL_FENCE_HANDLE_TYPE_SYNC_FD_BIT;
   CHECK(GetFenceFdKHR(device[1], &get_fence, &fd) == VK_SUCCESS);
   CHECK(GetFenceStatus(device[1], fence[1]) == VK_NOT_READY && GetFenceStatus(device[0], alias) == VK_NOT_READY);
   import_fence.handleType = get_fence.handleType;
   import_fence.flags = VK_FENCE_IMPORT_TEMPORARY_BIT;
   import_fence.fd = fd;
   CHECK(ImportFenceFdKHR(device[0], &import_fence) == VK_SUCCESS);
   consumed(fd);
   CHECK(WaitForFences(device[0], 1, &alias, VK_TRUE, 10000000000ull) == VK_SUCCESS);
   CHECK(ResetFences(device[0], 1, &alias) == VK_SUCCESS && GetFenceStatus(device[0], alias) == VK_NOT_READY);
   DestroyFence(device[0], alias, NULL);
   for (unsigned i = 0; i < 2; i++) {
      CHECK(QueueWaitIdle(queue[i]) == VK_SUCCESS);
      DestroyFence(device[i], fence[i], NULL);
      DestroySemaphore(device[i], binary[i], NULL);
      DestroySemaphore(device[i], timeline[i], NULL);
      DestroyDevice(device[i], NULL);
   }
   DestroyInstance(instance, NULL);
   CHECK(!dlclose(loader));
   puts("PASS Apex external sync: two devices, timeline gate 7/19/37, opaque binary reuse, sync-file temporary restore, fence copy/reference reset, consumed FDs (sync-only, no shader)");
   return 0;
}
