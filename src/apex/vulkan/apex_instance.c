/* SPDX-License-Identifier: MIT */
#include "apex_device.h"
#include "apex_entrypoints.h"
#include "drm-uapi/apex_drm.h"
#include "vk_alloc.h"
#include "vk_drm_syncobj.h"
#include "vk_instance.h"
#include "vk_limits.h"
#include "vk_log.h"
#include "vk_physical_device.h"
#include "util/log.h"
#include "util/os_misc.h"
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <xf86drm.h>

/* Development protocol only. This compute subset is not a conformant Vulkan
 * device. The library/manifest are uninstalled and instance creation is opt-in. */
#define APEX_DEVELOPMENT_API VK_API_VERSION_1_0

struct apex_physical_device {
   struct vk_physical_device vk;
   char *render_node;
   bool prime_coherent;
   bool host_coherent;
   struct vk_sync_type sync_type;
   const struct vk_sync_type *sync_types[2];
};
VK_DEFINE_HANDLE_CASTS(apex_physical_device, vk.base, VkPhysicalDevice,
                      VK_OBJECT_TYPE_PHYSICAL_DEVICE);

static const struct vk_instance_extension_table instance_extensions = {
   .KHR_get_physical_device_properties2 = true,
   .KHR_external_memory_capabilities = true,
   .KHR_external_semaphore_capabilities = true,
   .KHR_external_fence_capabilities = true,
   .EXT_debug_utils = true,
};

VKAPI_ATTR VkResult VKAPI_CALL
apex_EnumerateInstanceVersion(uint32_t *version)
{
   /* Mesa implements the instance protocol independently of device support. */
   *version = VK_API_VERSION_1_4;
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
apex_EnumerateInstanceLayerProperties(uint32_t *count, VkLayerProperties *properties)
{
   *count = 0;
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
apex_EnumerateInstanceExtensionProperties(const char *layer, uint32_t *count,
                                         VkExtensionProperties *properties)
{
   if (layer)
      return VK_ERROR_LAYER_NOT_PRESENT;
   return vk_enumerate_instance_extension_properties(&instance_extensions, count, properties);
}

VKAPI_ATTR void VKAPI_CALL
apex_GetPhysicalDeviceQueueFamilyProperties2(VkPhysicalDevice physical, uint32_t *count,
                                             VkQueueFamilyProperties2 *properties)
{
   VK_OUTARRAY_MAKE_TYPED(VkQueueFamilyProperties2, out, properties, count);
   vk_outarray_append_typed(VkQueueFamilyProperties2, &out, prop) {
      prop->queueFamilyProperties = (VkQueueFamilyProperties) {
         .queueFlags = VK_QUEUE_COMPUTE_BIT,
         .queueCount = 1,
      };
   }
}

VKAPI_ATTR void VKAPI_CALL
apex_GetPhysicalDeviceMemoryProperties2(VkPhysicalDevice physical,
                                       VkPhysicalDeviceMemoryProperties2 *properties)
{
   VK_FROM_HANDLE(apex_physical_device, device, physical);
   properties->memoryProperties = (VkPhysicalDeviceMemoryProperties) {
      .memoryTypeCount = device->prime_coherent ? 2 : 1,
      .memoryTypes[0] = {.propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                        VK_MEMORY_PROPERTY_HOST_CACHED_BIT},
      .memoryTypes[1] = {.propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT},
      .memoryHeapCount = 1,
      .memoryHeaps[0] = {.size = 64 * 1024 * 1024,
                        .flags = device->prime_coherent ? VK_MEMORY_HEAP_DEVICE_LOCAL_BIT : 0},
   };
   if (device->host_coherent) {
      VkPhysicalDeviceMemoryProperties *mem = &properties->memoryProperties;
      mem->memoryTypes[mem->memoryTypeCount++] = (VkMemoryType) {
         .propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                          VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
         .heapIndex = mem->memoryHeapCount++,
      };
      mem->memoryHeaps[1] = (VkMemoryHeap) {.size = 64 * 1024 * 1024};
   }
}

VKAPI_ATTR void VKAPI_CALL
apex_GetPhysicalDeviceExternalBufferProperties(VkPhysicalDevice physical,
   const VkPhysicalDeviceExternalBufferInfo *info, VkExternalBufferProperties *properties)
{
   VK_FROM_HANDLE(apex_physical_device, device, physical);
   properties->externalMemoryProperties = (VkExternalMemoryProperties){0};
   if (!device->prime_coherent || info->flags ||
       (info->usage & ~(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                       VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT)) ||
       (info->handleType != VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT &&
        info->handleType != VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT))
      return;
   properties->externalMemoryProperties = (VkExternalMemoryProperties) {
      .externalMemoryFeatures = VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT | VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT,
      .exportFromImportedHandleTypes = APEX_EXTERNAL_MEMORY_TYPES,
      .compatibleHandleTypes = APEX_EXTERNAL_MEMORY_TYPES,
   };
}

VKAPI_ATTR void VKAPI_CALL
apex_GetPhysicalDeviceFormatProperties2(VkPhysicalDevice physical, VkFormat format,
                                       VkFormatProperties2 *properties)
{
   VkFormatFeatureFlags features = format == VK_FORMAT_R32_UINT ?
      VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_STORAGE_IMAGE_ATOMIC_BIT |
      VK_FORMAT_FEATURE_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT : 0;
   properties->formatProperties = (VkFormatProperties) {
      .linearTilingFeatures = features, .optimalTilingFeatures = features,
   };
   VkFormatProperties3 *props3 = vk_find_struct(properties->pNext, FORMAT_PROPERTIES_3);
   if (props3) {
      props3->linearTilingFeatures = props3->optimalTilingFeatures = features;
      props3->bufferFeatures = 0;
   }
}

VKAPI_ATTR VkResult VKAPI_CALL
apex_GetPhysicalDeviceImageFormatProperties2(VkPhysicalDevice physical,
                                            const VkPhysicalDeviceImageFormatInfo2 *info,
                                            VkImageFormatProperties2 *properties)
{
   return apex_image_format_properties(info, properties);
}

VKAPI_ATTR void VKAPI_CALL
apex_GetPhysicalDeviceSparseImageFormatProperties2(VkPhysicalDevice physical,
   const VkPhysicalDeviceSparseImageFormatInfo2 *info, uint32_t *count,
   VkSparseImageFormatProperties2 *properties)
{
   *count = 0;
}

VKAPI_ATTR void VKAPI_CALL
apex_DestroyDevice(VkDevice handle, const VkAllocationCallbacks *alloc)
{
   VK_FROM_HANDLE(apex_device, device, handle);
   if (!device)
      return;
   int fd = device->fd;
   VkAllocationCallbacks allocator = device->vk.alloc;
   apex_device_finish(device);
   close(fd);
   vk_free2(&allocator, alloc, device);
}

VKAPI_ATTR VkResult VKAPI_CALL
apex_CreateDevice(VkPhysicalDevice handle, const VkDeviceCreateInfo *info,
                   const VkAllocationCallbacks *alloc, VkDevice *out)
{
   VK_FROM_HANDLE(apex_physical_device, physical, handle);
   *out = VK_NULL_HANDLE;
   /* Each logical device needs a fresh open, not dup(): the DRM file owns its VM. */
   int fd = open(physical->render_node, O_RDWR | O_CLOEXEC);
   if (fd < 0)
      return VK_ERROR_INITIALIZATION_FAILED;
   struct apex_device *device = vk_zalloc2(&physical->vk.instance->alloc, alloc,
      sizeof(*device), 8, VK_SYSTEM_ALLOCATION_SCOPE_DEVICE);
   if (!device) {
      close(fd);
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }
   VkResult result = apex_device_init(device, &physical->vk, info, alloc, fd, APEX_TRANSPORT_DRM);
   if (result != VK_SUCCESS) {
      vk_free2(&physical->vk.instance->alloc, alloc, device);
      close(fd);
      return result;
   }
   device->vk.dispatch_table.DestroyDevice = apex_DestroyDevice;
   *out = apex_device_to_handle(device);
   return VK_SUCCESS;
}

static void
destroy_physical(struct vk_physical_device *vk)
{
   struct apex_physical_device *physical = (void *)vk;
   const VkAllocationCallbacks *alloc = &vk->instance->alloc;
   vk_physical_device_finish(vk);
   vk_free(alloc, physical->render_node);
   vk_free(alloc, physical);
}

static VkResult
try_create_physical(struct vk_instance *instance, drmDevicePtr drm,
                    struct vk_physical_device **out)
{
   if (!(drm->available_nodes & (1 << DRM_NODE_RENDER)) || drm->bustype != DRM_BUS_PCI ||
       drm->deviceinfo.pci->vendor_id != 0x10ee || drm->deviceinfo.pci->device_id != 0xa15e)
      return VK_ERROR_INCOMPATIBLE_DRIVER;
   const char *node = drm->nodes[DRM_NODE_RENDER];
   int fd = open(node, O_RDWR | O_CLOEXEC);
   if (fd < 0)
      return VK_ERROR_INCOMPATIBLE_DRIVER;
   drmVersionPtr version = drmGetVersion(fd);
   bool matches = version && !strcmp(version->name, "apex-display");
   drmFreeVersion(version);
   struct drm_apex_info caps = {0};
   if (!matches || ioctl(fd, DRM_IOCTL_APEX_INFO, &caps) || caps.version != 2 ||
       (caps.capabilities & (APEX_DRM_CAP_SHMEM | APEX_DRM_CAP_GPUVM)) !=
       (APEX_DRM_CAP_SHMEM | APEX_DRM_CAP_GPUVM)) {
      close(fd);
      return VK_ERROR_INCOMPATIBLE_DRIVER;
   }
   struct vk_sync_type sync_type = vk_drm_syncobj_get_type(fd);
   close(fd);
   const uint32_t required_sync = VK_SYNC_FEATURE_BINARY | VK_SYNC_FEATURE_TIMELINE |
                                  VK_SYNC_FEATURE_CPU_WAIT | VK_SYNC_FEATURE_CPU_RESET |
                                  VK_SYNC_FEATURE_GPU_WAIT | VK_SYNC_FEATURE_WAIT_PENDING;
   if ((sync_type.features & required_sync) != required_sync)
      return VK_ERROR_INCOMPATIBLE_DRIVER;
   struct apex_physical_device *physical = vk_zalloc(&instance->alloc,
      sizeof(*physical), 8, VK_SYSTEM_ALLOCATION_SCOPE_INSTANCE);
   if (!physical)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   struct vk_physical_device_dispatch_table dispatch;
   vk_physical_device_dispatch_table_from_entrypoints(&dispatch, &apex_physical_device_entrypoints, true);
   const struct vk_device_extension_table extensions = {
      .KHR_get_memory_requirements2 = true,
      .KHR_dedicated_allocation = true,
      .KHR_external_memory = caps.capabilities & APEX_DRM_CAP_PRIME_COHERENT,
      .KHR_external_memory_fd = caps.capabilities & APEX_DRM_CAP_PRIME_COHERENT,
      .EXT_external_memory_dma_buf = caps.capabilities & APEX_DRM_CAP_PRIME_COHERENT,
      /* Mesa's DRM syncobj type supplies opaque-FD and sync-file payloads. */
      .KHR_external_semaphore = true,
      .KHR_external_semaphore_fd = true,
      .KHR_external_fence = true,
      .KHR_external_fence_fd = true,
      .KHR_storage_buffer_storage_class = true,
      .KHR_timeline_semaphore = true,
      .EXT_robustness2 = true,
      .EXT_scalar_block_layout = true,
   };
   const struct vk_features features = {
      .timelineSemaphore = true,
      .robustBufferAccess = true,
      .robustBufferAccess2 = true,
      .scalarBlockLayout = true,
   };
   const bool multiwave = caps.capabilities & APEX_DRM_CAP_MULTIWAVE;
   struct vk_properties properties = {
      .apiVersion = APEX_DEVELOPMENT_API,
      .vendorID = 0x10ee, .deviceID = 0xa15e,
      .deviceType = VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU,
      .deviceName = "Apex development compute (non-conformant)",
      .maxMemoryAllocationCount = 4096,
      .maxBoundDescriptorSets = MESA_VK_MAX_DESCRIPTOR_SETS,
      .maxPerStageDescriptorUniformBuffers = APEX_MAX_DESCRIPTORS,
      .maxPerStageDescriptorStorageBuffers = APEX_MAX_DESCRIPTORS,
      .maxPerStageDescriptorStorageImages = APEX_MAX_DESCRIPTORS,
      .maxPerStageResources = APEX_MAX_DESCRIPTORS,
      .maxDescriptorSetUniformBuffers = APEX_MAX_DESCRIPTORS,
      .maxDescriptorSetStorageBuffers = APEX_MAX_DESCRIPTORS,
      .maxDescriptorSetStorageImages = APEX_MAX_DESCRIPTORS,
      .maxDescriptorSetUniformBuffersDynamic = APEX_MAX_DESCRIPTORS,
      .maxDescriptorSetStorageBuffersDynamic = APEX_MAX_DESCRIPTORS,
      .maxUniformBufferRange = 64 * 1024 * 1024,
      .maxStorageBufferRange = 64 * 1024 * 1024,
      .maxPushConstantsSize = APEX_MAX_PUSH_CONSTANTS,
      .maxComputeSharedMemorySize = 32768,
      .maxComputeWorkGroupCount = {1024, 1, 1},
      .maxComputeWorkGroupInvocations = multiwave ? 256 : 16,
      .maxComputeWorkGroupSize = {multiwave ? 256 : 16,
                                  multiwave ? 256 : 16, multiwave ? 64 : 16},
      .maxImageDimension2D = 4096, .maxImageArrayLayers = 256,
      .storageImageSampleCounts = VK_SAMPLE_COUNT_1_BIT,
      .minMemoryMapAlignment = 4096,
      .minUniformBufferOffsetAlignment = 4,
      .minStorageBufferOffsetAlignment = 4,
      .nonCoherentAtomSize = 1,
      /* Descriptor bounds are checked without rounding, per 32-bit component. */
      .robustStorageBufferAccessSizeAlignment = 1,
      .robustUniformBufferAccessSizeAlignment = 1,
      .subgroupSize = 16, .minSubgroupSize = 16, .maxSubgroupSize = 16,
      .maxTimelineSemaphoreValueDifference = UINT64_MAX,
   };
   /* Opaque-fd compatibility is the flat GEM byte layout, revision 1. */
   memcpy(properties.driverUUID, "Apex GEM bytes 1", VK_UUID_SIZE);
   properties.deviceUUID[0] = 0xee; properties.deviceUUID[1] = 0x10;
   properties.deviceUUID[2] = 0x5e; properties.deviceUUID[3] = 0xa1;
   properties.deviceUUID[4] = drm->businfo.pci->domain;
   properties.deviceUUID[5] = drm->businfo.pci->domain >> 8;
   properties.deviceUUID[6] = drm->businfo.pci->bus;
   properties.deviceUUID[7] = drm->businfo.pci->dev;
   properties.deviceUUID[8] = drm->businfo.pci->func;
   VkResult result = vk_physical_device_init(&physical->vk, instance, &extensions,
                                            &features, &properties, &dispatch);
   if (result != VK_SUCCESS) {
      vk_free(&instance->alloc, physical);
      return result;
   }
   physical->render_node = vk_strdup(&instance->alloc, node, VK_SYSTEM_ALLOCATION_SCOPE_INSTANCE);
   if (!physical->render_node) {
      destroy_physical(&physical->vk);
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   }
   physical->sync_type = sync_type;
   physical->prime_coherent = caps.capabilities & APEX_DRM_CAP_PRIME_COHERENT;
   physical->host_coherent = caps.capabilities & APEX_DRM_CAP_HOST_COHERENT;
   physical->sync_types[0] = &physical->sync_type;
   physical->vk.supported_sync_types = physical->sync_types;
   *out = &physical->vk;
   return VK_SUCCESS;
}

VKAPI_ATTR VkResult VKAPI_CALL
apex_CreateInstance(const VkInstanceCreateInfo *info, const VkAllocationCallbacks *alloc,
                     VkInstance *out)
{
   *out = VK_NULL_HANDLE;
   const char *development = os_get_option("APEX_DEVELOPMENT");
   if (!development || strcmp(development, "1"))
      return VK_ERROR_INCOMPATIBLE_DRIVER;
   alloc = alloc ? alloc : vk_default_allocator();
   struct vk_instance *instance = vk_zalloc(alloc, sizeof(*instance), 8,
                                           VK_SYSTEM_ALLOCATION_SCOPE_INSTANCE);
   if (!instance)
      return VK_ERROR_OUT_OF_HOST_MEMORY;
   struct vk_instance_dispatch_table dispatch;
   vk_instance_dispatch_table_from_entrypoints(&dispatch, &apex_instance_entrypoints, true);
   VkResult result = vk_instance_init(instance, &instance_extensions, &dispatch, info, alloc);
   if (result != VK_SUCCESS) {
      vk_free(alloc, instance);
      return result;
   }
   instance->physical_devices.try_create_for_drm = try_create_physical;
   instance->physical_devices.destroy = destroy_physical;
   mesa_logw("Apex development compute driver: incomplete and non-conformant; no graphics or WSI");
   *out = vk_instance_to_handle(instance);
   return VK_SUCCESS;
}

VKAPI_ATTR void VKAPI_CALL
apex_DestroyInstance(VkInstance handle, const VkAllocationCallbacks *alloc)
{
   VK_FROM_HANDLE(vk_instance, instance, handle);
   if (!instance)
      return;
   VkAllocationCallbacks allocator = instance->alloc;
   vk_instance_finish(instance);
   vk_free2(&allocator, alloc, instance);
}

VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
apex_GetInstanceProcAddr(VkInstance handle, const char *name)
{
   VK_FROM_HANDLE(vk_instance, instance, handle);
   return vk_instance_get_proc_addr(instance, &apex_instance_entrypoints, name);
}

PUBLIC VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
vk_icdGetInstanceProcAddr(VkInstance instance, const char *name)
{
   return apex_GetInstanceProcAddr(instance, name);
}
