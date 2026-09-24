/* SPDX-License-Identifier: MIT */
#include "apex_entrypoints.h"
#include "drm-uapi/apex_drm.h"
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <xf86drm.h>

#define CHECK(x) do { if (!(x)) { \
   fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); abort(); \
} } while (0)
#define PROC(type, name) PFN_vk##type name = (PFN_vk##type)gipa(instance, "vk" #type); CHECK(name)

static int fault, open_count, last_fd;
static bool mock;
static char *nodes[DRM_NODE_MAX] = {[DRM_NODE_RENDER] = "/apex-test/render"};
static drmPciDeviceInfo pci = {.vendor_id = 0x10ee, .device_id = 0xa15e};
static drmDevice drm = {.nodes = nodes, .available_nodes = 1 << DRM_NODE_RENDER,
                        .bustype = DRM_BUS_PCI, .deviceinfo.pci = &pci};
static drmVersion version = {.name = "apex-display"};

int __wrap_drmGetDevices2(uint32_t flags, drmDevicePtr devices[], int count);
void __wrap_drmFreeDevices(drmDevicePtr devices[], int count);
drmVersionPtr __wrap_drmGetVersion(int fd);
void __wrap_drmFreeVersion(drmVersionPtr ptr);
int __wrap_drmGetCap(int fd, uint64_t cap, uint64_t *value);
int __wrap_drmSyncobjCreate(int fd, uint32_t flags, uint32_t *handle);
int __wrap_drmSyncobjDestroy(int fd, uint32_t handle);
int __wrap_drmSyncobjWait(int fd, uint32_t *handles, unsigned count,
                        int64_t timeout, unsigned flags, uint32_t *first);
int __wrap_open64(const char *path, int flags, ...);
int __real_open64(const char *path, int flags, ...);
int __wrap_ioctl(int fd, unsigned long request, ...);

int __wrap_drmGetDevices2(uint32_t flags, drmDevicePtr devices[], int count)
{
   CHECK(mock && count > 0);
   devices[0] = &drm;
   return 1;
}
void __wrap_drmFreeDevices(drmDevicePtr devices[], int count)
{
   CHECK(count == 1 && devices[0] == &drm);
}
drmVersionPtr __wrap_drmGetVersion(int fd) { return &version; }
void __wrap_drmFreeVersion(drmVersionPtr ptr) { CHECK(ptr == &version); }
int __wrap_drmGetCap(int fd, uint64_t cap, uint64_t *value)
{
   CHECK(cap == DRM_CAP_SYNCOBJ_TIMELINE);
   *value = fault != 4;
   return 0;
}
int __wrap_drmSyncobjCreate(int fd, uint32_t flags, uint32_t *handle)
{
   CHECK(flags == DRM_SYNCOBJ_CREATE_SIGNALED);
   *handle = 7;
   return 0;
}
int __wrap_drmSyncobjDestroy(int fd, uint32_t handle) { CHECK(handle == 7); return 0; }
int __wrap_drmSyncobjWait(int fd, uint32_t *handles, unsigned count,
                        int64_t timeout, unsigned flags, uint32_t *first)
{
   CHECK(count == 1 && handles[0] == 7 && !timeout);
   return fault == 5 ? -1 : 0;
}
int __wrap_open64(const char *path, int flags, ...)
{
   if (!strcmp(path, nodes[DRM_NODE_RENDER])) {
      CHECK(mock && flags == (O_RDWR | O_CLOEXEC));
      open_count++;
      return last_fd = __real_open64("/dev/null", flags);
   }
   CHECK(!(flags & O_CREAT));
   return __real_open64(path, flags);
}
int __wrap_ioctl(int fd, unsigned long request, ...)
{
   CHECK(mock && request == DRM_IOCTL_APEX_INFO);
   va_list ap;
   va_start(ap, request);
   struct drm_apex_info *info = va_arg(ap, void *);
   va_end(ap);
   CHECK(!info->version && !info->capabilities && !info->max_buffer_bytes);
   *info = (struct drm_apex_info) {
      .version = fault == 2 ? 1 : 2,
      .capabilities = APEX_DRM_CAP_SHMEM | (fault == 3 ? 0 : APEX_DRM_CAP_GPUVM),
      .max_buffer_bytes = 64 * 1024 * 1024,
   };
   return 0;
}

static void
exercise(PFN_vkGetInstanceProcAddr gipa)
{
   VkInstance instance = VK_NULL_HANDLE;
   PROC(CreateInstance, create);
   CHECK(!gipa(VK_NULL_HANDLE, "vkDestroyInstance"));
   CHECK(!gipa(VK_NULL_HANDLE, "vkNotAFunction"));
   VkInstanceCreateInfo info = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
   CHECK(!unsetenv("APEX_DEVELOPMENT"));
   CHECK(create(&info, NULL, &instance) == VK_ERROR_INCOMPATIBLE_DRIVER);
   CHECK(instance == VK_NULL_HANDLE);
   CHECK(!setenv("APEX_DEVELOPMENT", "1", 1));
   const char *unsupported = "VK_KHR_surface";
   info.enabledExtensionCount = 1;
   info.ppEnabledExtensionNames = &unsupported;
   CHECK(create(&info, NULL, &instance) == VK_ERROR_EXTENSION_NOT_PRESENT);
   const char *properties_extension = VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME;
   info.ppEnabledExtensionNames = &properties_extension;
   CHECK(create(&info, NULL, &instance) == VK_SUCCESS);
   PROC(DestroyInstance, destroy);
   PROC(EnumeratePhysicalDevices, enumerate);
   uint32_t count = 0;
   VkResult enumerated = enumerate(instance, &count, NULL);
   if (!mock) {
      /* The system loader reports INITIALIZATION_FAILED when all ICDs return
       * zero devices. The orb has no Apex GPU; this is loader, not GPU evidence. */
      CHECK(enumerated == VK_SUCCESS ||
            (enumerated == VK_ERROR_INITIALIZATION_FAILED && count == 0));
      printf("PASS Apex loader: opt-in, extension rejection, instance lifecycle, %u compatible devices\n", count);
      destroy(instance, NULL);
      return;
   }
   CHECK(enumerated == VK_SUCCESS);
   CHECK(count == (fault ? 0 : 1));
   if (fault) {
      destroy(instance, NULL);
      return;
   }
   VkPhysicalDevice physical;
   count = 0;
   CHECK(enumerate(instance, &count, &physical) == VK_INCOMPLETE && !count);
   count = 1;
   CHECK(enumerate(instance, &count, &physical) == VK_SUCCESS && count == 1);
   PROC(GetPhysicalDeviceProperties, get_properties);
   VkPhysicalDeviceProperties props;
   get_properties(physical, &props);
   CHECK(props.limits.maxComputeWorkGroupCount[0] == 1024 &&
         props.limits.maxComputeWorkGroupCount[1] == 1 &&
         props.limits.maxComputeWorkGroupInvocations == 16);
   CHECK(strstr(props.deviceName, "non-conformant"));
   PROC(GetPhysicalDeviceFeatures2KHR, get_features2);
   VkPhysicalDeviceRobustness2FeaturesEXT robustness = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT,
      .robustImageAccess2 = VK_TRUE, .nullDescriptor = VK_TRUE,
   };
   VkPhysicalDeviceScalarBlockLayoutFeatures scalar = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES,
      .pNext = &robustness,
   };
   VkPhysicalDeviceFeatures2 features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &scalar,
   };
   get_features2(physical, &features);
   CHECK(scalar.scalarBlockLayout);
   CHECK(features.features.robustBufferAccess && robustness.robustBufferAccess2);
   CHECK(!robustness.robustImageAccess2 && !robustness.nullDescriptor);
   PROC(GetPhysicalDeviceProperties2KHR, get_properties2);
   VkPhysicalDeviceRobustness2PropertiesEXT robust_props = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_PROPERTIES_EXT,
   };
   VkPhysicalDeviceProperties2 properties2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &robust_props,
   };
   get_properties2(physical, &properties2);
   CHECK(robust_props.robustStorageBufferAccessSizeAlignment == 1 &&
         robust_props.robustUniformBufferAccessSizeAlignment == 1);
   PROC(GetPhysicalDeviceMemoryProperties, get_memory);
   VkPhysicalDeviceMemoryProperties mem;
   get_memory(physical, &mem);
   CHECK(mem.memoryTypeCount == 1 && mem.memoryHeapCount == 1);
   CHECK(mem.memoryTypes[0].propertyFlags ==
         (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT));
   PROC(GetPhysicalDeviceFormatProperties, get_format);
   VkFormatProperties format;
   memset(&format, 0xff, sizeof(format));
   get_format(physical, VK_FORMAT_R8G8B8A8_UNORM, &format);
   CHECK(!format.linearTilingFeatures && !format.optimalTilingFeatures && !format.bufferFeatures);
   PROC(GetPhysicalDeviceQueueFamilyProperties, get_queues);
   VkQueueFamilyProperties queue_props;
   count = 1;
   get_queues(physical, &count, &queue_props);
   CHECK(count == 1 && queue_props.queueCount == 1 && queue_props.queueFlags == VK_QUEUE_COMPUTE_BIT);
   PROC(CreateDevice, create_device);
   PROC(GetDeviceProcAddr, gdpa);
   float priority = 0;
   VkDeviceQueueCreateInfo queue = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                   .queueCount = 1, .pQueuePriorities = &priority};
   VkDeviceCreateInfo device_info = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                                     .queueCreateInfoCount = 1, .pQueueCreateInfos = &queue};
   VkPhysicalDeviceFeatures unsupported_features = {.geometryShader = VK_TRUE};
   device_info.pEnabledFeatures = &unsupported_features;
   VkDevice device;
   CHECK(create_device(physical, &device_info, NULL, &device) == VK_ERROR_FEATURE_NOT_PRESENT);
   CHECK(device == VK_NULL_HANDLE && fcntl(last_fd, F_GETFD) == -1 && errno == EBADF);
   device_info.pEnabledFeatures = NULL;
   const char *memory_extensions[] = {
      VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME, VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME,
      VK_EXT_ROBUSTNESS_2_EXTENSION_NAME, VK_EXT_SCALAR_BLOCK_LAYOUT_EXTENSION_NAME,
   };
   device_info.enabledExtensionCount = 4;
   device_info.ppEnabledExtensionNames = memory_extensions;
   device_info.pNext = &features;
   robustness.nullDescriptor = VK_TRUE;
   CHECK(create_device(physical, &device_info, NULL, &device) == VK_ERROR_FEATURE_NOT_PRESENT);
   CHECK(device == VK_NULL_HANDLE && fcntl(last_fd, F_GETFD) == -1 && errno == EBADF);
   robustness.nullDescriptor = VK_FALSE;
   /* CTS's compute-only robustness helper can duplicate the sole family.
    * Reject that queue list independently of the supported feature chain. */
   const VkDeviceQueueCreateInfo duplicate_queues[] = {queue, queue};
   device_info.queueCreateInfoCount = 2;
   device_info.pQueueCreateInfos = duplicate_queues;
   CHECK(create_device(physical, &device_info, NULL, &device) == VK_ERROR_FEATURE_NOT_PRESENT);
   CHECK(device == VK_NULL_HANDLE && fcntl(last_fd, F_GETFD) == -1 && errno == EBADF);
   device_info.queueCreateInfoCount = 1;
   device_info.pQueueCreateInfos = &queue;
   int opens = open_count;
   CHECK(create_device(physical, &device_info, NULL, &device) == VK_SUCCESS);
   PFN_vkGetBufferMemoryRequirements2KHR get_requirements =
      (PFN_vkGetBufferMemoryRequirements2KHR)gdpa(device, "vkGetBufferMemoryRequirements2KHR");
   PFN_vkCreateBuffer create_buffer = (PFN_vkCreateBuffer)gdpa(device, "vkCreateBuffer");
   PFN_vkDestroyBuffer destroy_buffer = (PFN_vkDestroyBuffer)gdpa(device, "vkDestroyBuffer");
   CHECK(get_requirements && create_buffer && destroy_buffer);
   const VkBufferCreateInfo buffer_info = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 65,
      .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT,
   };
   VkBuffer buffer;
   CHECK(create_buffer(device, &buffer_info, NULL, &buffer) == VK_SUCCESS);
   const VkBufferMemoryRequirementsInfo2 buffer_req = {
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2, .buffer = buffer,
   };
   VkMemoryDedicatedRequirements dedicated = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS,
      .prefersDedicatedAllocation = VK_TRUE, .requiresDedicatedAllocation = VK_TRUE,
   };
   VkMemoryRequirements2 requirements = {
      .sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2, .pNext = &dedicated,
   };
   get_requirements(device, &buffer_req, &requirements);
   CHECK(requirements.memoryRequirements.size == 128 &&
         requirements.memoryRequirements.alignment == 64 &&
         requirements.memoryRequirements.memoryTypeBits == 1);
   CHECK(!dedicated.prefersDedicatedAllocation && !dedicated.requiresDedicatedAllocation);
   destroy_buffer(device, buffer, NULL);
   int fd1 = last_fd;
   VkDevice second;
   CHECK(create_device(physical, &device_info, NULL, &second) == VK_SUCCESS);
   int fd2 = last_fd;
   CHECK(open_count == opens + 2 && fd1 != fd2);
   PFN_vkDestroyDevice destroy_device = (PFN_vkDestroyDevice)gdpa(device, "vkDestroyDevice");
   CHECK(destroy_device);
   destroy_device(device, NULL);
   destroy_device(second, NULL);
   CHECK(fcntl(fd1, F_GETFD) == -1 && errno == EBADF);
   CHECK(fcntl(fd2, F_GETFD) == -1 && errno == EBADF);
   destroy(instance, NULL);
}

int main(int argc, char **argv)
{
   CHECK(argc == 2);
   if (!strcmp(argv[1], "--mock")) {
      mock = true;
      for (fault = 0; fault <= 5; fault++) {
         version.name = fault == 1 ? "foreign-driver" : "apex-display";
         exercise(apex_GetInstanceProcAddr);
      }
      puts("PASS Apex instance: device/ABI/sync filtering, compute-only queries, fresh VM opens, cleanup (mock DRM)");
   } else {
      CHECK(!setenv("VK_DRIVER_FILES", argv[1], 1));
      /* The real system loader consumes the generated manifest and shared ICD. */
      void *loader = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
      CHECK(loader);
      PFN_vkGetInstanceProcAddr gipa = (PFN_vkGetInstanceProcAddr)dlsym(loader, "vkGetInstanceProcAddr");
      CHECK(gipa);
      exercise(gipa);
      CHECK(!dlclose(loader));
   }
   return 0;
}
