/* SPDX-License-Identifier: MIT */
#include "apex_entrypoints.h"
#include "mock_kernel.h"
#include "util/macros.h"
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
/* DRM_APEX_INFO timestamp_hz the mock kernel reports, when nonzero. */
static uint64_t timestamp_hz;
static bool mock;
/* Each render-node open is a fresh DRM file: a mock kernel on its memfd. */
static struct mock_kernel kernels[32];
static unsigned kernel_count;
static char *nodes[DRM_NODE_MAX] = {[DRM_NODE_RENDER] = "/apex-test/render"};
static drmPciDeviceInfo pci = {.vendor_id = 0x10ee, .device_id = 0xa15e};
static drmPciBusInfo bus = {.domain = 0x1234, .bus = 7, .dev = 3, .func = 1};
static drmDevice drm = {.nodes = nodes, .available_nodes = 1 << DRM_NODE_RENDER | 1 << DRM_NODE_PRIMARY,
                        .bustype = DRM_BUS_PCI, .deviceinfo.pci = &pci, .businfo.pci = &bus};
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
      CHECK(kernel_count < ARRAY_SIZE(kernels));
      mock_kernel_init(&kernels[kernel_count]);
      return last_fd = kernels[kernel_count++].fd;
   }
   CHECK(!(flags & O_CREAT));
   return __real_open64(path, flags);
}
int __wrap_ioctl(int fd, unsigned long request, ...)
{
   CHECK(mock);
   va_list ap;
   va_start(ap, request);
   void *arg = va_arg(ap, void *);
   va_end(ap);
   struct mock_kernel *kernel = NULL;
   for (unsigned i = kernel_count; i-- && !kernel;)
      if (kernels[i].fd == fd)
         kernel = &kernels[i];
   CHECK(kernel);
   if (request == DRM_IOCTL_APEX_INFO) {
      const struct drm_apex_info zero = {0};
      CHECK(!memcmp(arg, &zero, sizeof(zero)));
      if (fault == 2) {
         errno = ENOTTY;
         return -1;
      }
      mock_kernel_ioctl(kernel, request, arg);
      if (timestamp_hz)
         ((struct drm_apex_info *)arg)->timestamp_hz = timestamp_hz;
      if (fault == 3)
         ((struct drm_apex_info *)arg)->timestamp_hz = 0;
      return 0;
   }
   return mock_kernel_ioctl(kernel, request, arg);
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
   const char *unsupported = "VK_KHR_xcb_surface";
   info.enabledExtensionCount = 1;
   info.ppEnabledExtensionNames = &unsupported;
   CHECK(create(&info, NULL, &instance) == VK_ERROR_EXTENSION_NOT_PRESENT);
   const char *instance_extensions[] = {VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
      VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME,
      VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME,
      VK_KHR_EXTERNAL_FENCE_CAPABILITIES_EXTENSION_NAME};
   info.enabledExtensionCount = ARRAY_SIZE(instance_extensions);
   info.ppEnabledExtensionNames = instance_extensions;
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
   /* The timestamp period follows the clock the kernel reports. */
   CHECK(props.limits.timestampPeriod == 1e9f / (timestamp_hz ? timestamp_hz : 250000000));
   CHECK(props.limits.maxComputeWorkGroupCount[0] == 65535 &&
         props.limits.maxComputeWorkGroupCount[1] == 65535 &&
         props.limits.maxComputeWorkGroupCount[2] == 65535 &&
         props.limits.maxComputeWorkGroupInvocations == 256 &&
         props.limits.maxComputeWorkGroupSize[0] == 256 &&
         props.limits.maxComputeWorkGroupSize[1] == 256 &&
         props.limits.maxComputeWorkGroupSize[2] == 64);
   CHECK(strstr(props.deviceName, "non-conformant"));
   PROC(GetPhysicalDeviceFeatures2KHR, get_features2);
   VkPhysicalDeviceRobustness2FeaturesEXT robustness = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_FEATURES_EXT,
      .robustImageAccess2 = VK_TRUE, .nullDescriptor = VK_TRUE,
   };
   VkPhysicalDeviceMaintenance5FeaturesKHR maintenance5 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES_KHR, .pNext = &robustness,
   };
   VkPhysicalDeviceTransformFeedbackFeaturesEXT xfb = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_FEATURES_EXT, .pNext = &maintenance5,
   };
   VkPhysicalDeviceProvokingVertexFeaturesEXT provoking = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROVOKING_VERTEX_FEATURES_EXT, .pNext = &xfb,
      .transformFeedbackPreservesProvokingVertex = VK_TRUE,
   };
   VkPhysicalDeviceCustomBorderColorFeaturesEXT border = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_CUSTOM_BORDER_COLOR_FEATURES_EXT, .pNext = &provoking,
   };
   VkPhysicalDeviceBorderColorSwizzleFeaturesEXT swizzle = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BORDER_COLOR_SWIZZLE_FEATURES_EXT, .pNext = &border,
   };
   VkPhysicalDeviceDepthClipEnableFeaturesEXT depth_clip = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEPTH_CLIP_ENABLE_FEATURES_EXT, .pNext = &swizzle,
   };
   VkPhysicalDeviceScalarBlockLayoutFeatures scalar = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES,
      .pNext = &depth_clip,
   };
   VkPhysicalDeviceFeatures2 features = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &scalar,
   };
   get_features2(physical, &features);
   CHECK(scalar.scalarBlockLayout);
   CHECK(features.features.robustBufferAccess && robustness.robustBufferAccess2);
   CHECK(!robustness.robustImageAccess2 && robustness.nullDescriptor);
   CHECK(maintenance5.maintenance5 && provoking.provokingVertexLast &&
         provoking.transformFeedbackPreservesProvokingVertex && xfb.transformFeedback && !xfb.geometryStreams);
   CHECK(border.customBorderColors && border.customBorderColorWithoutFormat);
   CHECK(swizzle.borderColorSwizzle && swizzle.borderColorSwizzleFromImage);
   CHECK(depth_clip.depthClipEnable);
   PROC(GetPhysicalDeviceProperties2KHR, get_properties2);
   VkPhysicalDeviceDrmPropertiesEXT drm_props = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRM_PROPERTIES_EXT,
   };
   VkPhysicalDeviceMaintenance5PropertiesKHR maintenance5_props = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_PROPERTIES_KHR, .pNext = &drm_props,
   };
   VkPhysicalDeviceProvokingVertexPropertiesEXT provoking_props = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROVOKING_VERTEX_PROPERTIES_EXT,
      .pNext = &maintenance5_props,
   };
   VkPhysicalDeviceTransformFeedbackPropertiesEXT xfb_props = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TRANSFORM_FEEDBACK_PROPERTIES_EXT,
      .pNext = &provoking_props,
   };
   VkPhysicalDeviceRobustness2PropertiesEXT robust_props = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ROBUSTNESS_2_PROPERTIES_EXT,
      .pNext = &xfb_props,
   };
   VkPhysicalDeviceProperties2 properties2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &robust_props,
   };
   get_properties2(physical, &properties2);
   CHECK(robust_props.robustStorageBufferAccessSizeAlignment == 1 &&
         robust_props.robustUniformBufferAccessSizeAlignment == 1);
   /* Zink matches its DRM fd's render node against these numbers. */
   CHECK(drm_props.hasRender && drm_props.renderMajor == 1 && drm_props.renderMinor == 3);
   CHECK(drm_props.hasPrimary && drm_props.primaryMajor == 1 && drm_props.primaryMinor == 5);
   CHECK(provoking_props.provokingVertexModePerPipeline &&
         provoking_props.transformFeedbackPreservesTriangleFanProvokingVertex);
   CHECK(xfb_props.maxTransformFeedbackStreams == 1 && xfb_props.maxTransformFeedbackBuffers == 4 &&
         xfb_props.transformFeedbackQueries && !xfb_props.transformFeedbackDraw);
   CHECK(maintenance5_props.earlyFragmentSampleMaskTestBeforeSampleCounting &&
         !maintenance5_props.earlyFragmentMultisampleCoverageAfterSampleCounting &&
         maintenance5_props.depthStencilSwizzleOneSupport && !maintenance5_props.polygonModePointSize &&
         !maintenance5_props.nonStrictSinglePixelWideLinesUseParallelogram &&
         !maintenance5_props.nonStrictWideLinesUseParallelogram);
   PROC(EnumerateDeviceExtensionProperties, enumerate_extensions);
   VkExtensionProperties extensions[128];
   uint32_t extension_count = ARRAY_SIZE(extensions);
   CHECK(enumerate_extensions(physical, NULL, &extension_count, extensions) == VK_SUCCESS);
   const char *zink_extensions[] = {VK_EXT_PHYSICAL_DEVICE_DRM_EXTENSION_NAME,
      VK_KHR_MAINTENANCE_5_EXTENSION_NAME, VK_EXT_PROVOKING_VERTEX_EXTENSION_NAME,
      VK_EXT_CUSTOM_BORDER_COLOR_EXTENSION_NAME, VK_EXT_BORDER_COLOR_SWIZZLE_EXTENSION_NAME,
      VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME, VK_EXT_IMAGE_DRM_FORMAT_MODIFIER_EXTENSION_NAME,
      VK_EXT_TRANSFORM_FEEDBACK_EXTENSION_NAME,
      VK_KHR_SWAPCHAIN_MUTABLE_FORMAT_EXTENSION_NAME};
   for (unsigned i = 0; i < ARRAY_SIZE(zink_extensions); i++) {
      bool found = false;
      for (unsigned e = 0; e < extension_count; e++)
         found |= !strcmp(extensions[e].extensionName, zink_extensions[i]);
      /* Foreign queue ownership, modifiers and swapchains accompany dma-buf memory. */
      CHECK(found == (i < 5 || coherent));
   }
   PROC(GetPhysicalDeviceMemoryProperties, get_memory);
   VkPhysicalDeviceMemoryProperties mem;
   get_memory(physical, &mem);
   /* LOCAL through BAR2 and cached SYSTEM, both host coherent. */
   CHECK(mem.memoryTypeCount == 2 && mem.memoryHeapCount == 2);
   CHECK(mem.memoryTypes[0].propertyFlags ==
         (VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) && !mem.memoryTypes[0].heapIndex);
   CHECK(mem.memoryTypes[1].propertyFlags ==
         (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT |
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) && mem.memoryTypes[1].heapIndex == 1);
   CHECK(mem.memoryHeaps[0].flags == VK_MEMORY_HEAP_DEVICE_LOCAL_BIT &&
         mem.memoryHeaps[0].size == 8ull << 30 && !mem.memoryHeaps[1].flags &&
         mem.memoryHeaps[1].size == 1024ull * 1024 * 1024);
   PROC(GetPhysicalDeviceExternalBufferPropertiesKHR, get_external);
   VkPhysicalDeviceExternalBufferInfo external = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_BUFFER_INFO,
      .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
   };
   VkExternalBufferProperties external_props = {.sType = VK_STRUCTURE_TYPE_EXTERNAL_BUFFER_PROPERTIES};
   get_external(physical, &external, &external_props);
   CHECK(external_props.externalMemoryProperties.externalMemoryFeatures ==
      (VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT | VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT));
   CHECK(external_props.externalMemoryProperties.compatibleHandleTypes ==
      (VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT | VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT));
   CHECK(external_props.externalMemoryProperties.exportFromImportedHandleTypes ==
         external_props.externalMemoryProperties.compatibleHandleTypes);
   external.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
   get_external(physical, &external, &external_props);
   CHECK(!external_props.externalMemoryProperties.externalMemoryFeatures);
   VkPhysicalDeviceIDProperties id = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
   properties2.pNext = &id;
   get_properties2(physical, &properties2);
   const uint8_t expected_id[VK_UUID_SIZE] = {0xee, 0x10, 0x5e, 0xa1, 0x34, 0x12, 7, 3, 1};
   CHECK(!memcmp(id.deviceUUID, expected_id, sizeof(expected_id)));
   CHECK(!memcmp(id.driverUUID, "Apex GEM bytes 1", VK_UUID_SIZE));
   PROC(GetPhysicalDeviceExternalSemaphorePropertiesKHR, get_external_semaphore);
   PROC(GetPhysicalDeviceExternalFencePropertiesKHR, get_external_fence);
   const uint32_t fd_types[] = {VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT,
      VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT};
   for (unsigned i = 0; i < ARRAY_SIZE(fd_types); i++) {
      VkPhysicalDeviceExternalSemaphoreInfo sem_info = {
         .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_SEMAPHORE_INFO,
         .handleType = fd_types[i]};
      VkExternalSemaphoreProperties sem_props = {.sType = VK_STRUCTURE_TYPE_EXTERNAL_SEMAPHORE_PROPERTIES};
      get_external_semaphore(physical, &sem_info, &sem_props);
      CHECK(sem_props.externalSemaphoreFeatures ==
         (VK_EXTERNAL_SEMAPHORE_FEATURE_IMPORTABLE_BIT | VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT));
      CHECK(sem_props.compatibleHandleTypes == (fd_types[0] | fd_types[1]));
      VkSemaphoreTypeCreateInfo timeline = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
         .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE};
      sem_info.pNext = &timeline;
      get_external_semaphore(physical, &sem_info, &sem_props);
      CHECK(sem_props.externalSemaphoreFeatures == (i ? 0 :
         VK_EXTERNAL_SEMAPHORE_FEATURE_IMPORTABLE_BIT | VK_EXTERNAL_SEMAPHORE_FEATURE_EXPORTABLE_BIT));
      CHECK(sem_props.compatibleHandleTypes == (i ? 0 : fd_types[0]));
      VkPhysicalDeviceExternalFenceInfo fence_info = {
         .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_FENCE_INFO,
         .handleType = i ? VK_EXTERNAL_FENCE_HANDLE_TYPE_SYNC_FD_BIT : VK_EXTERNAL_FENCE_HANDLE_TYPE_OPAQUE_FD_BIT};
      VkExternalFenceProperties fence_props = {.sType = VK_STRUCTURE_TYPE_EXTERNAL_FENCE_PROPERTIES};
      get_external_fence(physical, &fence_info, &fence_props);
      CHECK(fence_props.externalFenceFeatures ==
         (VK_EXTERNAL_FENCE_FEATURE_IMPORTABLE_BIT | VK_EXTERNAL_FENCE_FEATURE_EXPORTABLE_BIT));
      CHECK(fence_props.compatibleHandleTypes ==
         (VK_EXTERNAL_FENCE_HANDLE_TYPE_OPAQUE_FD_BIT | VK_EXTERNAL_FENCE_HANDLE_TYPE_SYNC_FD_BIT));
   }
   PROC(GetPhysicalDeviceFormatProperties, get_format);
   VkFormatProperties format;
   memset(&format, 0xff, sizeof(format));
   get_format(physical, VK_FORMAT_R8G8B8A8_UNORM, &format);
   const VkFormatFeatureFlags transfer_features = VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
                                                  VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
   const VkFormatFeatureFlags sampled = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
   const VkFormatFeatureFlags texel = VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT |
                                      VK_FORMAT_FEATURE_UNIFORM_TEXEL_BUFFER_BIT |
                                      VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_BIT;
   CHECK(format.linearTilingFeatures == (transfer_features | VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT |
            VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT |
            sampled | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT |
            VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_MINMAX_BIT |
            VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT) &&
         format.optimalTilingFeatures == format.linearTilingFeatures && format.bufferFeatures == texel);
   /* Maintenance5: enumerants beyond the defined formats report nothing. */
   for (unsigned i = 0; i < 5; i++) {
      memset(&format, 0xff, sizeof(format));
      get_format(physical, VK_FORMAT_MAX_ENUM - i, &format);
      CHECK(!format.linearTilingFeatures && !format.optimalTilingFeatures && !format.bufferFeatures);
   }
   /* The packed floats sample and filter through class 11 and do not
    * render; 4444 samples, 1555 also renders and blends. */
   const VkFormatFeatureFlags filtered = sampled | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT |
      VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_MINMAX_BIT | VK_FORMAT_FEATURE_BLIT_SRC_BIT;
   const VkFormatFeatureFlags rendered = VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT |
      VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT;
   const struct { VkFormat format; bool render; } new_formats[] = {
      {VK_FORMAT_B10G11R11_UFLOAT_PACK32, false}, {VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, false},
      {VK_FORMAT_R4G4B4A4_UNORM_PACK16, false}, {VK_FORMAT_A4B4G4R4_UNORM_PACK16, false},
      {VK_FORMAT_A1R5G5B5_UNORM_PACK16, true}, {VK_FORMAT_B5G5R5A1_UNORM_PACK16, true},
   };
   for (unsigned i = 0; i < ARRAY_SIZE(new_formats); i++) {
      get_format(physical, new_formats[i].format, &format);
      CHECK((format.optimalTilingFeatures & (filtered | rendered)) ==
            (filtered | (new_formats[i].render ? rendered : 0)));
   }
   get_format(physical, VK_FORMAT_R32_UINT, &format);
   VkFormatFeatureFlags image_features = VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT |
      VK_FORMAT_FEATURE_STORAGE_IMAGE_ATOMIC_BIT | VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT |
      VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT | transfer_features | sampled;
   CHECK(format.linearTilingFeatures == image_features && format.optimalTilingFeatures == image_features &&
         format.bufferFeatures == (texel | VK_FORMAT_FEATURE_STORAGE_TEXEL_BUFFER_ATOMIC_BIT));
   CHECK(props.limits.maxImageDimension2D == 8192 && props.limits.maxImageArrayLayers == 2048);
   /* DRM_FORMAT_MOD_LINEAR only, for linear color formats. */
   PROC(GetPhysicalDeviceFormatProperties2KHR, get_format2);
   const VkFormat modifier_formats[] = {VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_D32_SFLOAT,
                                        VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK};
   for (unsigned i = 0; i < ARRAY_SIZE(modifier_formats); i++) {
      VkDrmFormatModifierProperties2EXT modifier2 = {0};
      VkDrmFormatModifierPropertiesList2EXT list2 = {
         .sType = VK_STRUCTURE_TYPE_DRM_FORMAT_MODIFIER_PROPERTIES_LIST_2_EXT,
         .drmFormatModifierCount = 1, .pDrmFormatModifierProperties = &modifier2};
      VkDrmFormatModifierPropertiesListEXT list = {
         .sType = VK_STRUCTURE_TYPE_DRM_FORMAT_MODIFIER_PROPERTIES_LIST_EXT, .pNext = &list2};
      VkFormatProperties2 format2 = {.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2, .pNext = &list};
      get_format2(physical, modifier_formats[i], &format2);
      CHECK(list.drmFormatModifierCount == !i && list2.drmFormatModifierCount == !i);
      if (!i)
         CHECK(!modifier2.drmFormatModifier && modifier2.drmFormatModifierPlaneCount == 1 &&
               (modifier2.drmFormatModifierTilingFeatures & 0x7fffffff) ==
               format2.formatProperties.linearTilingFeatures);
   }
   /* External images: importable and exportable dma-bufs and opaque fds. */
   PROC(GetPhysicalDeviceImageFormatProperties2KHR, get_image_format2);
   VkPhysicalDeviceImageDrmFormatModifierInfoEXT modifier_info = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_DRM_FORMAT_MODIFIER_INFO_EXT};
   VkPhysicalDeviceExternalImageFormatInfo external_image = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO, .pNext = &modifier_info,
      .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT};
   VkPhysicalDeviceImageFormatInfo2 image_info2 = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2, .pNext = &external_image,
      .format = VK_FORMAT_R8G8B8A8_UNORM, .type = VK_IMAGE_TYPE_2D,
      .tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT,
      .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT};
   VkExternalImageFormatProperties external_props2 = {
      .sType = VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES};
   VkImageFormatProperties2 image_props2 = {.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2,
                                            .pNext = &external_props2};
   CHECK(get_image_format2(physical, &image_info2, &image_props2) ==
         (coherent ? VK_SUCCESS : VK_ERROR_FORMAT_NOT_SUPPORTED));
   if (coherent) {
      CHECK(image_props2.imageFormatProperties.maxMipLevels == 1 &&
            image_props2.imageFormatProperties.maxArrayLayers == 1);
      CHECK(external_props2.externalMemoryProperties.externalMemoryFeatures ==
            (VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT | VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT) &&
            external_props2.externalMemoryProperties.compatibleHandleTypes ==
            (VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT | VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT));
   }
   modifier_info.drmFormatModifier = 0x0100000000000001ull;
   CHECK(get_image_format2(physical, &image_info2, &image_props2) == VK_ERROR_FORMAT_NOT_SUPPORTED);
   external_image.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
   image_info2.tiling = VK_IMAGE_TILING_OPTIMAL;
   image_info2.format = VK_FORMAT_ETC2_R8G8B8_UNORM_BLOCK;
   image_info2.usage = VK_IMAGE_USAGE_SAMPLED_BIT;
   CHECK(get_image_format2(physical, &image_info2, &image_props2) == VK_ERROR_FORMAT_NOT_SUPPORTED);
   /* Mutable swapchains: storage on an sRGB image through its UNORM view. */
   const VkFormat view_formats[] = {VK_FORMAT_B8G8R8A8_SRGB, VK_FORMAT_B8G8R8A8_UNORM};
   VkImageFormatListCreateInfo view_list = {.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO,
      .viewFormatCount = 2, .pViewFormats = view_formats};
   image_info2 = (VkPhysicalDeviceImageFormatInfo2) {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2, .pNext = &view_list,
      .format = VK_FORMAT_B8G8R8A8_SRGB, .type = VK_IMAGE_TYPE_2D, .tiling = VK_IMAGE_TILING_OPTIMAL,
      .usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT};
   image_props2.pNext = NULL;
   CHECK(get_image_format2(physical, &image_info2, &image_props2) == VK_ERROR_FORMAT_NOT_SUPPORTED);
   image_info2.flags = VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT | VK_IMAGE_CREATE_EXTENDED_USAGE_BIT;
   CHECK(get_image_format2(physical, &image_info2, &image_props2) == VK_SUCCESS);
   PROC(GetPhysicalDeviceImageFormatProperties, get_image_format);
   VkImageFormatProperties image_props;
   CHECK(get_image_format(physical, VK_FORMAT_R32_UINT, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL,
      VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT, 0, &image_props) == VK_SUCCESS);
   CHECK(image_props.maxExtent.width == 8192 && image_props.maxExtent.height == 8192 && image_props.maxExtent.depth == 1);
   CHECK(image_props.maxMipLevels == 14 && image_props.maxArrayLayers == 2048 &&
         image_props.sampleCounts == VK_SAMPLE_COUNT_1_BIT && image_props.maxResourceSize == (1ull << 31));
   CHECK(get_image_format(physical, VK_FORMAT_R32_UINT, VK_IMAGE_TYPE_3D, VK_IMAGE_TILING_OPTIMAL,
      VK_IMAGE_USAGE_STORAGE_BIT, 0, &image_props) == VK_SUCCESS);
   CHECK(image_props.maxExtent.depth == 512 && image_props.sampleCounts == VK_SAMPLE_COUNT_1_BIT);
   CHECK(get_image_format(physical, VK_FORMAT_R32_UINT, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_LINEAR,
      VK_IMAGE_USAGE_SAMPLED_BIT, 0, &image_props) == VK_SUCCESS);
   CHECK(get_image_format(physical, VK_FORMAT_BC1_RGB_UNORM_BLOCK, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL,
      VK_IMAGE_USAGE_SAMPLED_BIT, 0, &image_props) == VK_ERROR_FORMAT_NOT_SUPPORTED);
   CHECK(get_image_format(physical, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TYPE_2D,
      VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
      0, &image_props) == VK_SUCCESS);
   CHECK(get_image_format(physical, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TYPE_2D,
      VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
      0, &image_props) == VK_SUCCESS);
   CHECK(get_image_format(physical, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TYPE_2D,
      VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, 0,
      &image_props) == VK_ERROR_FORMAT_NOT_SUPPORTED);
   CHECK(get_image_format(physical, VK_FORMAT_R32_SINT, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_LINEAR,
      VK_IMAGE_USAGE_STORAGE_BIT, 0, &image_props) == VK_SUCCESS);
   PROC(GetPhysicalDeviceQueueFamilyProperties, get_queues);
   VkQueueFamilyProperties queue_props;
   count = 1;
   get_queues(physical, &count, &queue_props);
   CHECK(count == 1 && queue_props.queueCount == 1 &&
         queue_props.queueFlags == (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT));
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
      VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME, VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
      VK_KHR_EXTERNAL_FENCE_EXTENSION_NAME, VK_KHR_EXTERNAL_FENCE_FD_EXTENSION_NAME,
      VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME, VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
      VK_EXT_EXTERNAL_MEMORY_DMA_BUF_EXTENSION_NAME,
   };
   device_info.enabledExtensionCount = ARRAY_SIZE(memory_extensions);
   device_info.ppEnabledExtensionNames = memory_extensions;
   device_info.pNext = &features;
   robustness.robustImageAccess2 = VK_TRUE;
   CHECK(create_device(physical, &device_info, NULL, &device) == VK_ERROR_FEATURE_NOT_PRESENT);
   CHECK(device == VK_NULL_HANDLE && fcntl(last_fd, F_GETFD) == -1 && errno == EBADF);
   robustness.robustImageAccess2 = VK_FALSE;
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
   /* Compute-only clients may destroy an unused null buffer view. Exercise
    * both direct lookup and the instance trampoline used by Amber cleanup. */
   PFN_vkDestroyBufferView destroy_view =
      (PFN_vkDestroyBufferView)gdpa(device, "vkDestroyBufferView");
   CHECK(destroy_view);
   destroy_view(device, VK_NULL_HANDLE, NULL);
   PROC(DestroyBufferView, destroy_view_instance);
   destroy_view_instance(device, VK_NULL_HANDLE, NULL);
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
         requirements.memoryRequirements.memoryTypeBits == 3);
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
      /* The M4.1 fabric's 125 MHz timebase and the M4.2 target's 250 MHz. */
      fault = 0;
      for (unsigned i = 0; i < 2; i++) {
         timestamp_hz = i ? 250000000 : 125000000;
         exercise(apex_GetInstanceProcAddr);
      }
      puts("PASS Apex instance: device/ABI/sync filtering, compute-only queries, fresh VM opens, queue creation, cleanup, timestamp period at 125 and 250 MHz (mock DRM)");
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
