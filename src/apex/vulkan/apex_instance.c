/* SPDX-License-Identifier: MIT */
#include "apex_device.h"
#include "apex_draw.h"
#include "apex_entrypoints.h"
#include "drm-uapi/apex_drm.h"
#include "drm-uapi/drm_fourcc.h"
#include "vk_alloc.h"
#include "vk_drm_syncobj.h"
#include "vk_instance.h"
#include "vk_limits.h"
#include "vk_log.h"
#include "vk_physical_device.h"
#include "wsi_common.h"
#include "util/log.h"
#include "util/os_misc.h"
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>
#include <xf86drm.h>

/* Development protocol only. This compute subset is not a conformant Vulkan
 * device. The library/manifest are uninstalled and instance creation is opt-in. */
#define APEX_DEVELOPMENT_API VK_API_VERSION_1_3

struct apex_physical_device {
   struct vk_physical_device vk;
   char *render_node;
   bool prime_coherent;
   bool host_coherent;
   struct vk_sync_type sync_type;
   const struct vk_sync_type *sync_types[2];
   /* Presentation through Mesa's display WSI on the device's primary node. */
   struct wsi_device wsi_device;
   int display_fd;
};
VK_DEFINE_HANDLE_CASTS(apex_physical_device, vk.base, VkPhysicalDevice,
                      VK_OBJECT_TYPE_PHYSICAL_DEVICE);

static const struct vk_instance_extension_table instance_extensions = {
   .KHR_get_physical_device_properties2 = true,
   .KHR_external_memory_capabilities = true,
   .KHR_external_semaphore_capabilities = true,
   .KHR_external_fence_capabilities = true,
   .EXT_debug_utils = true,
   .KHR_surface = true,
   .KHR_display = true,
   .KHR_get_surface_capabilities2 = true,
   .KHR_get_display_properties2 = true,
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
         .queueFlags = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT,
         .timestampValidBits = 64,
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
      .memoryHeaps[0] = {.size = APEX_MAX_ALLOCATION,
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
      mem->memoryHeaps[1] = (VkMemoryHeap) {.size = APEX_MAX_ALLOCATION};
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
   VkFormatFeatureFlags2 features = apex_format_features(format, false);
   VkFormatFeatureFlags2 buffer = apex_format_features(format, true);
   /* Legacy flags hold bits 0..30; higher bits exist only in FormatFeatureFlags2. */
   const VkFormatFeatureFlags2 legacy = 0x7fffffffull;
   /* ETC2/EAC images are optimal-tiling only. */
   VkFormatFeatureFlags2 linear = apex_decoded_format(format, NULL) ? 0 : features;
   properties->formatProperties = (VkFormatProperties) {
      .linearTilingFeatures = (VkFormatFeatureFlags)(linear & legacy),
      .optimalTilingFeatures = (VkFormatFeatureFlags)(features & legacy),
      .bufferFeatures = (VkFormatFeatureFlags)(buffer & legacy),
   };
   VkFormatProperties3 *props3 = vk_find_struct(properties->pNext, FORMAT_PROPERTIES_3);
   if (props3) {
      props3->linearTilingFeatures = linear;
      props3->optimalTilingFeatures = features;
      props3->bufferFeatures = buffer;
   }
   /* DRM_FORMAT_MOD_LINEAR, one plane, with the linear tiling features. */
   bool modifier = apex_format_modifier_supported(format);
   VkDrmFormatModifierPropertiesListEXT *list =
      vk_find_struct(properties->pNext, DRM_FORMAT_MODIFIER_PROPERTIES_LIST_EXT);
   if (list) {
      VK_OUTARRAY_MAKE_TYPED(VkDrmFormatModifierPropertiesEXT, out, list->pDrmFormatModifierProperties,
                             &list->drmFormatModifierCount);
      if (modifier) {
         vk_outarray_append_typed(VkDrmFormatModifierPropertiesEXT, &out, p)
            *p = (VkDrmFormatModifierPropertiesEXT) {DRM_FORMAT_MOD_LINEAR, 1,
                                                     (VkFormatFeatureFlags)(linear & legacy)};
      }
   }
   VkDrmFormatModifierPropertiesList2EXT *list2 =
      vk_find_struct(properties->pNext, DRM_FORMAT_MODIFIER_PROPERTIES_LIST_2_EXT);
   if (list2) {
      VK_OUTARRAY_MAKE_TYPED(VkDrmFormatModifierProperties2EXT, out, list2->pDrmFormatModifierProperties,
                             &list2->drmFormatModifierCount);
      if (modifier) {
         vk_outarray_append_typed(VkDrmFormatModifierProperties2EXT, &out, p)
            *p = (VkDrmFormatModifierProperties2EXT) {DRM_FORMAT_MOD_LINEAR, 1, linear};
      }
   }
}

VKAPI_ATTR VkResult VKAPI_CALL
apex_GetPhysicalDeviceImageFormatProperties2(VkPhysicalDevice physical,
                                            const VkPhysicalDeviceImageFormatInfo2 *info,
                                            VkImageFormatProperties2 *properties)
{
   VK_FROM_HANDLE(apex_physical_device, device, physical);
   return apex_image_format_properties(info, device->prime_coherent, properties);
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

static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL
wsi_proc_addr(VkPhysicalDevice physical, const char *name)
{
   VK_FROM_HANDLE(apex_physical_device, device, physical);
   return vk_instance_get_proc_addr_unchecked(device->vk.instance, name);
}

static void
destroy_physical(struct vk_physical_device *vk)
{
   struct apex_physical_device *physical = (void *)vk;
   const VkAllocationCallbacks *alloc = &vk->instance->alloc;
   if (physical->vk.wsi_device) {
      physical->vk.wsi_device = NULL;
      wsi_device_finish(&physical->wsi_device, alloc);
   }
   if (physical->display_fd >= 0)
      close(physical->display_fd);
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
   /* VK_EXT_physical_device_drm: device numbers of the render and primary nodes. */
   struct stat render_stat, primary_stat;
   bool has_render = !fstat(fd, &render_stat);
   bool has_primary = (drm->available_nodes & (1 << DRM_NODE_PRIMARY)) &&
                      !stat(drm->nodes[DRM_NODE_PRIMARY], &primary_stat);
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
   vk_physical_device_dispatch_table_from_entrypoints(&dispatch, &wsi_physical_device_entrypoints, false);
   const struct vk_device_extension_table extensions = {
      .KHR_swapchain = caps.capabilities & APEX_DRM_CAP_PRIME_COHERENT,
      .KHR_get_memory_requirements2 = true,
      .KHR_dedicated_allocation = true,
      .KHR_external_memory = caps.capabilities & APEX_DRM_CAP_PRIME_COHERENT,
      .KHR_external_memory_fd = caps.capabilities & APEX_DRM_CAP_PRIME_COHERENT,
      .EXT_external_memory_dma_buf = caps.capabilities & APEX_DRM_CAP_PRIME_COHERENT,
      .EXT_image_drm_format_modifier = caps.capabilities & APEX_DRM_CAP_PRIME_COHERENT,
      /* Swapchain images take MUTABLE_FORMAT and a view format list. */
      .KHR_swapchain_mutable_format = caps.capabilities & APEX_DRM_CAP_PRIME_COHERENT,
      /* Mesa's DRM syncobj type supplies opaque-FD and sync-file payloads. */
      .KHR_external_semaphore = true,
      .KHR_external_semaphore_fd = true,
      .KHR_external_fence = true,
      .KHR_external_fence_fd = true,
      .KHR_storage_buffer_storage_class = true,
      .KHR_timeline_semaphore = true,
      .EXT_robustness2 = true,
      .EXT_physical_device_drm = true,
      /* Foreign ownership transfers, like external ones, need no operation:
       * jobs serialize and PRIME memory is coherent at submission. */
      .EXT_queue_family_foreign = caps.capabilities & APEX_DRM_CAP_PRIME_COHERENT,
      .KHR_maintenance5 = true,
      .EXT_provoking_vertex = true,
      .EXT_custom_border_color = true,
      .EXT_border_color_swizzle = true,
      .KHR_line_rasterization = true,
      .EXT_line_rasterization = true,
      .EXT_depth_clip_enable = true,
      .EXT_transform_feedback = true,
      .KHR_vertex_attribute_divisor = true,
      .EXT_vertex_attribute_divisor = true,
      .EXT_scalar_block_layout = true,
      .KHR_sampler_mirror_clamp_to_edge = true,
      .KHR_index_type_uint8 = true,
      .KHR_draw_indirect_count = true,
      .KHR_multiview = true,
      .KHR_push_descriptor = true,
      .KHR_descriptor_update_template = true,
      .EXT_shader_viewport_index_layer = true,
      /* Promoted to Vulkan 1.1-1.3. */
      .KHR_bind_memory2 = true,
      .KHR_maintenance1 = true,
      .KHR_maintenance2 = true,
      .KHR_maintenance3 = true,
      .KHR_maintenance4 = true,
      .KHR_relaxed_block_layout = true,
      .KHR_create_renderpass2 = true,
      .KHR_depth_stencil_resolve = true,
      .KHR_driver_properties = true,
      .KHR_image_format_list = true,
      .KHR_imageless_framebuffer = true,
      .KHR_separate_depth_stencil_layouts = true,
      .KHR_shader_float_controls = true,
      .KHR_shader_subgroup_extended_types = true,
      .KHR_spirv_1_4 = true,
      .KHR_uniform_buffer_standard_layout = true,
      .KHR_buffer_device_address = true,
      .KHR_vulkan_memory_model = true,
      .EXT_host_query_reset = true,
      .EXT_sampler_filter_minmax = true,
      .EXT_separate_stencil_usage = true,
      .EXT_private_data = true,
      .EXT_inline_uniform_block = true,
      .KHR_dynamic_rendering = true,
      .KHR_synchronization2 = true,
      .EXT_pipeline_creation_cache_control = true,
      .EXT_subgroup_size_control = true,
      .EXT_shader_demote_to_helper_invocation = true,
      .KHR_shader_terminate_invocation = true,
      .KHR_zero_initialize_workgroup_memory = true,
      .KHR_shader_integer_dot_product = true,
      .EXT_image_robustness = true,
      .KHR_copy_commands2 = true,
      .KHR_format_feature_flags2 = true,
      .EXT_texel_buffer_alignment = true,
      .KHR_shader_non_semantic_info = true,
      .EXT_extended_dynamic_state = true,
      .EXT_extended_dynamic_state2 = true,
   };
   const struct vk_features features = {
      .timelineSemaphore = true,
      .robustBufferAccess = true,
      .robustBufferAccess2 = true,
      /* Null descriptors are zero rows (see write_descriptor). */
      .nullDescriptor = true,
      .maintenance5 = true,
      .provokingVertexLast = true,
      /* Border colors need no format; they follow the view's component
       * mapping (apex_texture.c). */
      .customBorderColors = true,
      .customBorderColorWithoutFormat = true,
      .borderColorSwizzle = true,
      .borderColorSwizzleFromImage = true,
      /* GLES 2 correctness in the software raster (setup and fragment kernels). */
      .logicOp = true,
      .fillModeNonSolid = true,
      .alphaToOne = true,
      .shaderClipDistance = true,
      .shaderCullDistance = true,
      /* The setup kernel's z planes follow depth clipping, independent of clamp. */
      .depthClipEnable = true,
      /* One stream captured by the setup kernel in primitive order. */
      .transformFeedback = true,
      /* Vertex fetch divides the instance index by any divisor, zero repeating
       * the first instance. */
      .vertexAttributeInstanceRateDivisor = true,
      .vertexAttributeInstanceRateZeroDivisor = true,
      .wideLines = true,
      .rectangularLines = true,
      .bresenhamLines = true,
      .smoothLines = true,
      .stippledRectangularLines = true,
      .stippledBresenhamLines = true,
      .stippledSmoothLines = true,
      .scalarBlockLayout = true,
      .samplerMirrorClampToEdge = true,
      .largePoints = true,
      .indexTypeUint8 = true,
      .multiDrawIndirect = true,
      .drawIndirectFirstInstance = true,
      .drawIndirectCount = true,
      .depthClamp = true,
      .multiViewport = true,
      .shaderOutputViewportIndex = true,
      .shaderOutputLayer = true,
      .multiview = true,
      .sampleRateShading = true,
      .shaderStorageImageExtendedFormats = true,
      .shaderStorageImageReadWithoutFormat = true,
      .shaderStorageImageWriteWithoutFormat = true,
      .fragmentStoresAndAtomics = true,
      .vertexPipelineStoresAndAtomics = true,
      /* Implemented 1.0 features. */
      .fullDrawIndexUint32 = true,
      .imageCubeArray = true,
      .independentBlend = true,
      .depthBiasClamp = true,
      .occlusionQueryPrecise = true,
      .shaderImageGatherExtended = true,
      .shaderInt64 = true,
      .shaderUniformBufferArrayDynamicIndexing = true,
      .shaderSampledImageArrayDynamicIndexing = true,
      .shaderStorageBufferArrayDynamicIndexing = true,
      .shaderStorageImageArrayDynamicIndexing = true,
      /* Vulkan 1.2. */
      .uniformBufferStandardLayout = true,
      .shaderSubgroupExtendedTypes = true,
      .separateDepthStencilLayouts = true,
      .hostQueryReset = true,
      .imagelessFramebuffer = true,
      .subgroupBroadcastDynamicId = true,
      .samplerFilterMinmax = true,
      .bufferDeviceAddress = true,
      /* Vulkan 1.3. */
      .robustImageAccess = true,
      .inlineUniformBlock = true,
      .pipelineCreationCacheControl = true,
      .privateData = true,
      .shaderDemoteToHelperInvocation = true,
      .shaderTerminateInvocation = true,
      .subgroupSizeControl = true,
      .computeFullSubgroups = true,
      .synchronization2 = true,
      .shaderZeroInitializeWorkgroupMemory = true,
      .dynamicRendering = true,
      .shaderIntegerDotProduct = true,
      .maintenance4 = true,
      .vulkanMemoryModel = true,
      .vulkanMemoryModelDeviceScope = true,
      .textureCompressionETC2 = true,
      .texelBufferAlignment = true,
      .extendedDynamicState = true,
      .extendedDynamicState2 = true,
   };
   const bool multiwave = caps.capabilities & APEX_DRM_CAP_MULTIWAVE;
   struct vk_properties properties = {
      .apiVersion = APEX_DEVELOPMENT_API,
      .vendorID = 0x10ee, .deviceID = 0xa15e,
      .deviceType = VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU,
      .deviceName = "Apex development (non-conformant)",
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
      .maxStorageBufferRange = APEX_MAX_ALLOCATION,
      .maxPushConstantsSize = APEX_MAX_PUSH_CONSTANTS,
      .maxComputeSharedMemorySize = 32768,
      /* Recording splits dispatches into native chunks. */
      .maxComputeWorkGroupCount = {65535, 65535, 65535},
      .maxComputeWorkGroupInvocations = multiwave ? 256 : 16,
      .maxComputeWorkGroupSize = {multiwave ? 256 : 16,
                                  multiwave ? 256 : 16, multiwave ? 64 : 16},
      .maxImageDimension1D = 4096, .maxImageDimension2D = 4096, .maxImageDimension3D = 2048,
      .maxImageDimensionCube = 4096, .maxImageArrayLayers = 256,
      .maxTexelBufferElements = 1 << 27,
      .maxSamplerAllocationCount = 4000,
      .bufferImageGranularity = 64,
      .maxPerStageDescriptorSamplers = APEX_MAX_DESCRIPTORS,
      .maxPerStageDescriptorSampledImages = APEX_MAX_DESCRIPTORS,
      .maxPerStageDescriptorInputAttachments = 8,
      .maxDescriptorSetSamplers = APEX_MAX_DESCRIPTORS,
      .maxDescriptorSetSampledImages = APEX_MAX_DESCRIPTORS,
      .maxDescriptorSetInputAttachments = 8,
      .maxVertexInputAttributes = 32, .maxVertexInputBindings = APEX_DRAW_MAX_BINDINGS,
      .maxVertexInputAttributeOffset = 2047, .maxVertexInputBindingStride = 2048,
      .maxVertexOutputComponents = 128, .maxFragmentInputComponents = 128,
      .maxFragmentOutputAttachments = 8, .maxFragmentCombinedOutputResources = 16,
      .maxColorAttachments = 8,
      .maxDrawIndexedIndexValue = UINT32_MAX, .maxDrawIndirectCount = 65535,
      .maxSamplerLodBias = 16.0f, .maxSamplerAnisotropy = 1.0f,
      .maxPushDescriptors = 32, .maxMultiviewViewCount = 32, .maxMultiviewInstanceIndex = (1u << 27) - 1,
      .maxViewports = APEX_DRAW_MAX_VIEWPORTS, .maxViewportDimensions = {4096, 4096},
      .viewportBoundsRange = {-8192.0f, 8191.0f}, .viewportSubPixelBits = 8,
      .subPixelPrecisionBits = 8, .subTexelPrecisionBits = 8, .mipmapPrecisionBits = 8,
      .minTexelBufferOffsetAlignment = 4,
      .minTexelOffset = -8, .maxTexelOffset = 7, .minTexelGatherOffset = -8, .maxTexelGatherOffset = 7,
      .minInterpolationOffset = -0.5f, .maxInterpolationOffset = 0.4375f,
      .subPixelInterpolationOffsetBits = 4,
      .maxFramebufferWidth = 4096, .maxFramebufferHeight = 4096, .maxFramebufferLayers = 256,
      .framebufferColorSampleCounts = VK_SAMPLE_COUNT_1_BIT | VK_SAMPLE_COUNT_4_BIT,
      .framebufferDepthSampleCounts = VK_SAMPLE_COUNT_1_BIT | VK_SAMPLE_COUNT_4_BIT,
      .framebufferStencilSampleCounts = VK_SAMPLE_COUNT_1_BIT | VK_SAMPLE_COUNT_4_BIT,
      .framebufferNoAttachmentsSampleCounts = VK_SAMPLE_COUNT_1_BIT | VK_SAMPLE_COUNT_4_BIT,
      .sampledImageColorSampleCounts = VK_SAMPLE_COUNT_1_BIT | VK_SAMPLE_COUNT_4_BIT,
      .sampledImageIntegerSampleCounts = VK_SAMPLE_COUNT_1_BIT | VK_SAMPLE_COUNT_4_BIT,
      .sampledImageDepthSampleCounts = VK_SAMPLE_COUNT_1_BIT | VK_SAMPLE_COUNT_4_BIT,
      .sampledImageStencilSampleCounts = VK_SAMPLE_COUNT_1_BIT | VK_SAMPLE_COUNT_4_BIT,
      .maxSampleMaskWords = 1, .discreteQueuePriorities = 2,
      .pointSizeRange = {1.0f, 64.0f}, .pointSizeGranularity = 1.0f / 128.0f, .lineWidthRange = {1.0f, 16.0f},
      .lineWidthGranularity = 1.0f / 8.0f, .lineSubPixelPrecisionBits = 8,
      .maxClipDistances = 8, .maxCullDistances = 8, .maxCombinedClipAndCullDistances = 8,
      .standardSampleLocations = true,
      .storageImageSampleCounts = VK_SAMPLE_COUNT_1_BIT,
      .minMemoryMapAlignment = 4096,
      .minUniformBufferOffsetAlignment = 4,
      .minStorageBufferOffsetAlignment = 4,
      .nonCoherentAtomSize = 1,
      .optimalBufferCopyOffsetAlignment = 64,
      /* Device timebase: 250 MHz on both profiles. */
      .timestampPeriod = 4.0f, .timestampComputeAndGraphics = true,
      .optimalBufferCopyRowPitchAlignment = 64,
      /* Descriptor bounds are checked without rounding, per 32-bit component. */
      .robustStorageBufferAccessSizeAlignment = 1,
      .robustUniformBufferAccessSizeAlignment = 1,
      .subgroupSize = 16, .minSubgroupSize = 16, .maxSubgroupSize = 16,
      .subgroupSupportedStages = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
      .subgroupSupportedOperations = VK_SUBGROUP_FEATURE_BASIC_BIT | VK_SUBGROUP_FEATURE_VOTE_BIT |
         VK_SUBGROUP_FEATURE_BALLOT_BIT | VK_SUBGROUP_FEATURE_SHUFFLE_BIT |
         VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT |
         VK_SUBGROUP_FEATURE_QUAD_BIT,
      /* Points are rejected only by the z planes and the guard band. */
      .pointClippingBehavior = VK_POINT_CLIPPING_BEHAVIOR_USER_CLIP_PLANES_ONLY,
      .maxPerSetDescriptors = APEX_MAX_DESCRIPTORS,
      .maxMemoryAllocationSize = APEX_MAX_ALLOCATION,
      /* Vulkan 1.2: depth/stencil resolves take sample 0. */
      .driverID = VK_DRIVER_ID_MESA_LLVMPIPE,
      .driverName = "Apex", .driverInfo = "Mesa development driver (non-conformant)",
      .supportedDepthResolveModes = VK_RESOLVE_MODE_SAMPLE_ZERO_BIT,
      .supportedStencilResolveModes = VK_RESOLVE_MODE_SAMPLE_ZERO_BIT,
      .independentResolveNone = true, .independentResolve = true,
      .filterMinmaxSingleComponentFormats = true, .filterMinmaxImageComponentMapping = true,
      .framebufferIntegerColorSampleCounts = VK_SAMPLE_COUNT_1_BIT | VK_SAMPLE_COUNT_4_BIT,
      .denormBehaviorIndependence = VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_ALL,
      .roundingModeIndependence = VK_SHADER_FLOAT_CONTROLS_INDEPENDENCE_ALL,
      .maxTimelineSemaphoreValueDifference = UINT64_MAX,
      /* Vulkan 1.3. */
      .requiredSubgroupSizeStages = VK_SHADER_STAGE_COMPUTE_BIT,
      .maxComputeWorkgroupSubgroups = multiwave ? 16 : 1,
      .maxInlineUniformBlockSize = APEX_MAX_INLINE_BYTES,
      .maxInlineUniformTotalSize = APEX_MAX_INLINE_BYTES,
      .maxPerStageDescriptorInlineUniformBlocks = 4,
      .maxPerStageDescriptorUpdateAfterBindInlineUniformBlocks = 4,
      .maxDescriptorSetInlineUniformBlocks = 4,
      .maxDescriptorSetUpdateAfterBindInlineUniformBlocks = 4,
      .storageTexelBufferOffsetAlignmentBytes = 4,
      .uniformTexelBufferOffsetAlignmentBytes = 4,
      .maxBufferSize = APEX_MAX_ALLOCATION,
      /* VK_KHR_maintenance5: early-test shaders count samples after the
       * sample mask and multisample coverage; ONE swizzles of depth/stencil
       * views read one; non-strict lines of any width are minor-axis parallelograms. */
      .earlyFragmentMultisampleCoverageAfterSampleCounting = false,
      .earlyFragmentSampleMaskTestBeforeSampleCounting = true,
      .depthStencilSwizzleOneSupport = true,
      .polygonModePointSize = false,
      .nonStrictSinglePixelWideLinesUseParallelogram = true,
      .nonStrictWideLinesUseParallelogram = true,
      .provokingVertexModePerPipeline = true,
      .transformFeedbackPreservesTriangleFanProvokingVertex = false,
      .maxCustomBorderColorSamplers = 4000,
      .maxVertexAttribDivisor = UINT32_MAX, .supportsNonZeroFirstInstance = true,
      .maxTransformFeedbackStreams = 1, .maxTransformFeedbackBuffers = APEX_DRAW_MAX_XFB_BUFFERS,
      .maxTransformFeedbackBufferSize = APEX_MAX_ALLOCATION,
      .maxTransformFeedbackStreamDataSize = APEX_DRAW_MAX_XFB_OUTPUTS * 16,
      .maxTransformFeedbackBufferDataSize = APEX_DRAW_MAX_XFB_OUTPUTS * 16,
      .maxTransformFeedbackBufferDataStride = 2048, .transformFeedbackQueries = true,
      .drmHasRender = has_render,
      .drmRenderMajor = has_render ? major(render_stat.st_rdev) : 0,
      .drmRenderMinor = has_render ? minor(render_stat.st_rdev) : 0,
      .drmHasPrimary = has_primary,
      .drmPrimaryMajor = has_primary ? major(primary_stat.st_rdev) : 0,
      .drmPrimaryMinor = has_primary ? minor(primary_stat.st_rdev) : 0,
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
   /* Swapchain images blit into exported PRIME buffers that KMS scans out;
    * the primary node needs the video group or seat access. */
   physical->display_fd = -1;
   if (instance->enabled_extensions.KHR_display && (drm->available_nodes & (1 << DRM_NODE_PRIMARY)))
      physical->display_fd = open(drm->nodes[DRM_NODE_PRIMARY], O_RDWR | O_CLOEXEC);
   result = wsi_device_init(&physical->wsi_device, apex_physical_device_to_handle(physical),
                            wsi_proc_addr, &instance->alloc, physical->display_fd, NULL,
                            &(struct wsi_device_options){.sw_device = false});
   if (result != VK_SUCCESS) {
      destroy_physical(&physical->vk);
      return result;
   }
   physical->wsi_device.supports_scanout = false;
   physical->wsi_device.supports_modifiers = false;
   physical->vk.wsi_device = &physical->wsi_device;
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
   vk_instance_dispatch_table_from_entrypoints(&dispatch, &wsi_instance_entrypoints, false);
   VkResult result = vk_instance_init(instance, &instance_extensions, &dispatch, info, alloc);
   if (result != VK_SUCCESS) {
      vk_free(alloc, instance);
      return result;
   }
   instance->physical_devices.try_create_for_drm = try_create_physical;
   instance->physical_devices.destroy = destroy_physical;
   mesa_logw("Apex development driver: incomplete and non-conformant");
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
