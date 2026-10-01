/* SPDX-License-Identifier: MIT */
#include "apex_device.h"
#include "drm-uapi/apex_drm.h"
#include "vk_alloc.h"
#include "vk_buffer.h"
#include "vk_instance.h"
#include "vk_physical_device.h"
#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define CHECK(x) do { if (!(x)) { \
   fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); abort(); \
} } while (0)

/* Model PRIME's single handle per dma-buf, including repeated imports that
 * acquire no additional kernel handle reference. No shader execution is mocked. */
static int objects[16];
static unsigned object_count, handle_count, live, closes, maps, binds;
static bool coherent = true, host_coherent, fail_bind, fail_create;
static uint32_t expected_create_flags;
static unsigned creates;
static struct { unsigned object; bool live; uint64_t va, size; } handles[64];

static unsigned
new_handle(unsigned object)
{
   unsigned h = ++handle_count;
   CHECK(h < ARRAY_SIZE(handles));
   struct stat st;
   CHECK(!fstat(objects[object], &st));
   handles[h].object = object;
   handles[h].size = st.st_size;
   handles[h].live = true;
   live++;
   return h;
}

int __wrap_ioctl(int fd, unsigned long request, ...);
int
__wrap_ioctl(int fd, unsigned long request, ...)
{
   va_list ap;
   va_start(ap, request);
   void *arg = va_arg(ap, void *);
   va_end(ap);
   if (request == DRM_IOCTL_APEX_INFO) {
      *(struct drm_apex_info *)arg = (struct drm_apex_info) {
         .version = 2, .capabilities = APEX_DRM_CAP_SHMEM | APEX_DRM_CAP_GPUVM |
                                      (coherent ? APEX_DRM_CAP_PRIME_COHERENT : 0) |
                                      (host_coherent ? APEX_DRM_CAP_HOST_COHERENT : 0),
      };
   } else if (request == DRM_IOCTL_APEX_GEM_CREATE) {
      struct drm_apex_gem_create *r = arg;
      creates++;
      CHECK(r->flags == expected_create_flags && !r->handle);
      if (fail_create) { errno = EOPNOTSUPP; return -1; }
      CHECK(object_count < ARRAY_SIZE(objects));
      objects[object_count] = memfd_create("apex-owned", MFD_CLOEXEC);
      CHECK(objects[object_count] >= 0 && !ftruncate(objects[object_count], r->size));
      r->handle = new_handle(object_count++);
      CHECK(!ftruncate(fd, (r->handle + 1) * 65536));
   } else if (request == DRM_IOCTL_APEX_GEM_MMAP) {
      struct drm_apex_gem_mmap *r = arg;
      struct stat st;
      CHECK(handles[r->handle].live && !fstat(fd, &st));
      r->offset = r->handle * 65536;
      if (st.st_size < (off_t)(r->offset + 65536))
         CHECK(!ftruncate(fd, r->offset + 65536));
      maps++;
   } else if (request == DRM_IOCTL_PRIME_FD_TO_HANDLE) {
      struct drm_prime_handle *r = arg;
      struct stat st;
      CHECK(!r->flags);
      if (fstat(r->fd, &st)) return -1;
      unsigned object;
      for (object = 0; object < object_count; object++) {
         struct stat candidate;
         CHECK(!fstat(objects[object], &candidate));
         if (candidate.st_ino == st.st_ino && candidate.st_dev == st.st_dev) break;
      }
      if (object == object_count) { errno = EINVAL; return -1; }
      for (unsigned h = 1; h <= handle_count; h++) {
         if (handles[h].live && handles[h].object == object) {
            r->handle = h;
            return 0;
         }
      }
      r->handle = new_handle(object);
   } else if (request == DRM_IOCTL_PRIME_HANDLE_TO_FD) {
      struct drm_prime_handle *r = arg;
      CHECK(handles[r->handle].live && r->flags == (DRM_CLOEXEC | DRM_RDWR));
      r->fd = fcntl(objects[handles[r->handle].object], F_DUPFD_CLOEXEC, 0);
      CHECK(r->fd >= 0);
   } else if (request == DRM_IOCTL_APEX_VM_BIND) {
      struct drm_apex_vm_bind *r = arg;
      if (r->operation == APEX_DRM_VM_BIND_MAP) {
         CHECK(handles[r->handle].live && !handles[r->handle].va);
         CHECK(r->bytes == handles[r->handle].size && !r->offset);
         CHECK(r->flags == (APEX_DRM_VM_READ | APEX_DRM_VM_WRITE));
         if (fail_bind) { errno = ENOMEM; return -1; }
         handles[r->handle].va = r->va;
         binds++;
      } else {
         CHECK(r->operation == APEX_DRM_VM_BIND_UNMAP);
         unsigned h;
         for (h = 1; h <= handle_count && handles[h].va != r->va; h++);
         CHECK(h <= handle_count && handles[h].live && handles[h].size == r->bytes);
         handles[h].va = 0;
      }
   } else {
      CHECK(request == DRM_IOCTL_GEM_CLOSE); /* No GEM_TRANSFER for imports. */
      struct drm_gem_close *r = arg;
      CHECK(handles[r->handle].live && !handles[r->handle].va);
      handles[r->handle].live = false;
      live--;
      closes++;
   }
   return 0;
}

int main(void)
{
   objects[object_count++] = memfd_create("foreign", MFD_CLOEXEC);
   CHECK(objects[0] >= 0 && !ftruncate(objects[0], 8192));
   struct vk_instance instance;
   const struct vk_instance_extension_table extensions = {0};
   const struct vk_instance_dispatch_table instance_dispatch = {0};
   VkInstanceCreateInfo ii = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
   CHECK(vk_instance_init(&instance, &extensions, &instance_dispatch, &ii, vk_default_allocator()) == VK_SUCCESS);
   (void)vk_instance_to_handle(&instance);
   struct vk_physical_device physical;
   const struct vk_physical_device_dispatch_table physical_dispatch = {0};
   const struct vk_properties properties = {0};
   CHECK(vk_physical_device_init(&physical, &instance, NULL, NULL, &properties, &physical_dispatch) == VK_SUCCESS);
   (void)vk_physical_device_to_handle(&physical);
   int fd = memfd_create("apex-file", MFD_CLOEXEC);
   CHECK(fd >= 0);
   struct apex_device device;
   float priority = 1;
   VkDeviceQueueCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueCount = 1, .pQueuePriorities = &priority};
   VkDeviceCreateInfo di = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
      .queueCreateInfoCount = 1, .pQueueCreateInfos = &qi};
   CHECK(apex_device_init(&device, &physical, &di, NULL, fd, APEX_TRANSPORT_DRM) == VK_SUCCESS);
   VkDevice dev = apex_device_to_handle(&device);
   const struct vk_device_dispatch_table *v = &device.vk.dispatch_table;
   VkMemoryFdPropertiesKHR props = {.sType = VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR};
   CHECK(v->GetMemoryFdPropertiesKHR(dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, objects[0], &props) == VK_SUCCESS);
   /* Imports take either device-local type: device-only PRIME or LOCAL-resident. */
   CHECK(props.memoryTypeBits == 6 && !live && closes == 1 && fcntl(objects[0], F_GETFD) >= 0);
   CHECK(!ftruncate(fd, 4096)); /* Valid extent, but not a PRIME-exported object. */
   CHECK(v->GetMemoryFdPropertiesKHR(dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, fd, &props) == VK_ERROR_INVALID_EXTERNAL_HANDLE);
   VkExportMemoryAllocateInfo export = {.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
      .handleTypes = APEX_EXTERNAL_MEMORY_TYPES};
   VkImportMemoryFdInfoKHR import = {.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
      .pNext = &export, .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT};
   VkMemoryAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .pNext = &import, .allocationSize = 8193, .memoryTypeIndex = 1};
   VkDeviceMemory memory[3];
   import.fd = dup(objects[0]);
   CHECK(import.fd >= 0);
   CHECK(v->AllocateMemory(dev, &ai, NULL, &memory[0]) == VK_ERROR_INVALID_EXTERNAL_HANDLE);
   CHECK(memory[0] == VK_NULL_HANDLE && fcntl(import.fd, F_GETFD) >= 0 && !live);
   ai.allocationSize = 8192;
   fail_bind = true;
   CHECK(v->AllocateMemory(dev, &ai, NULL, &memory[0]) == VK_ERROR_OUT_OF_DEVICE_MEMORY);
   CHECK(!live && fcntl(import.fd, F_GETFD) >= 0);
   fail_bind = false;
   unsigned before = closes;
   CHECK(v->AllocateMemory(dev, &ai, NULL, &memory[0]) == VK_SUCCESS);
   CHECK(fcntl(import.fd, F_GETFD) == -1 && errno == EBADF);
   import.fd = dup(objects[0]);
   /* Zink allocates with device addresses; capture/replay is unsupported. */
   VkMemoryAllocateFlagsInfo flags = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, .pNext = &import,
      .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT | VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_CAPTURE_REPLAY_BIT};
   ai.pNext = &flags;
   CHECK(v->AllocateMemory(dev, &ai, NULL, &memory[1]) == VK_ERROR_FEATURE_NOT_PRESENT);
   CHECK(fcntl(import.fd, F_GETFD) >= 0);
   flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
   CHECK(v->AllocateMemory(dev, &ai, NULL, &memory[1]) == VK_SUCCESS);
   ai.pNext = &import;
   CHECK(fcntl(import.fd, F_GETFD) == -1 && errno == EBADF);
   CHECK(live == 1 && binds == 1 && !maps && closes == before);
   CHECK(v->GetMemoryFdPropertiesKHR(dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, objects[0], &props) == VK_SUCCESS);
   CHECK(live == 1 && closes == before);
   void *map = (void *)1;
   CHECK(v->MapMemory(dev, memory[0], 0, VK_WHOLE_SIZE, 0, &map) == VK_ERROR_MEMORY_MAP_FAILED && !map);
   VkExternalMemoryBufferCreateInfo external = {.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
      .handleTypes = APEX_EXTERNAL_MEMORY_TYPES};
   VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .pNext = &external,
      .size = 193, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT};
   VkBuffer buffers[2];
   for (unsigned i = 0; i < 2; i++) {
      CHECK(v->CreateBuffer(dev, &bi, NULL, &buffers[i]) == VK_SUCCESS);
      VkMemoryRequirements req;
      v->GetBufferMemoryRequirements(dev, buffers[i], &req);
      CHECK(req.size == 256 && req.alignment == 64 && req.memoryTypeBits == 2);
      CHECK(v->BindBufferMemory(dev, buffers[i], memory[i], 64 + i * 128) == VK_SUCCESS);
   }
   CHECK(vk_buffer_from_handle(buffers[1])->device_address - vk_buffer_from_handle(buffers[0])->device_address == 128);
   v->DestroyBuffer(dev, buffers[0], NULL);
   v->FreeMemory(dev, memory[0], NULL);
   CHECK(live == 1 && closes == before);
   VkMemoryGetFdInfoKHR get = {.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
      .memory = memory[1], .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT};
   int exported;
   CHECK(v->GetMemoryFdKHR(dev, &get, &exported) == VK_SUCCESS);
   CHECK(fcntl(exported, F_GETFD) & FD_CLOEXEC);
   v->DestroyBuffer(dev, buffers[1], NULL);
   v->FreeMemory(dev, memory[1], NULL);
   CHECK(!live && closes == before + 1 && fcntl(exported, F_GETFD) >= 0);
   CHECK(!close(exported));
   /* A zero-sized import retains its backing but exposes no bindable bytes. */
   import.pNext = NULL;
   import.fd = dup(objects[0]);
   ai.allocationSize = 0;
   CHECK(v->AllocateMemory(dev, &ai, NULL, &memory[0]) == VK_SUCCESS);
   CHECK(fcntl(import.fd, F_GETFD) == -1 && errno == EBADF);
   CHECK(live == 1 && !maps);
   CHECK(v->CreateBuffer(dev, &bi, NULL, &buffers[0]) == VK_SUCCESS);
   CHECK(v->BindBufferMemory(dev, buffers[0], memory[0], 0) == VK_ERROR_OUT_OF_DEVICE_MEMORY);
   v->DestroyBuffer(dev, buffers[0], NULL);
   v->FreeMemory(dev, memory[0], NULL);
   CHECK(!live);
   /* handleType=0 ignores the import structure, including its FD. */
   import.handleType = 0;
   import.fd = objects[0];
   ai.allocationSize = 8192;
   ai.memoryTypeIndex = 0;
   CHECK(v->AllocateMemory(dev, &ai, NULL, &memory[0]) == VK_SUCCESS);
   CHECK(fcntl(import.fd, F_GETFD) >= 0 && live == 1 && maps == 1);
   CHECK(v->MapMemory(dev, memory[0], 0, VK_WHOLE_SIZE, 0, &map) == VK_SUCCESS && map);
   v->UnmapMemory(dev, memory[0]);
   v->FreeMemory(dev, memory[0], NULL);
   CHECK(!live);
   ai.memoryTypeIndex = 1;
   import.pNext = &export;
   /* Owned export/self-import must share the original handle in either free order. */
   for (unsigned order = 0; order < 2; order++) {
      ai.pNext = &export;
      CHECK(v->AllocateMemory(dev, &ai, NULL, &memory[0]) == VK_SUCCESS);
      get.memory = memory[0];
      CHECK(v->GetMemoryFdKHR(dev, &get, &import.fd) == VK_SUCCESS);
      import.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
      ai.pNext = &import;
      CHECK(v->AllocateMemory(dev, &ai, NULL, &memory[1]) == VK_SUCCESS);
      CHECK(live == 1);
      before = closes;
      v->FreeMemory(dev, memory[order], NULL);
      CHECK(live == 1 && closes == before);
      get.memory = memory[1 - order];
      CHECK(v->GetMemoryFdKHR(dev, &get, &exported) == VK_SUCCESS && !close(exported));
      v->FreeMemory(dev, memory[1 - order], NULL);
      CHECK(!live && closes == before + 1);
   }
   /* External images: LOCAL-resident memory, dma-buf import, LINEAR modifier. */
   VkExternalMemoryImageCreateInfo external_image = {
      .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
      .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT};
   VkImageCreateInfo image_info = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .pNext = &external_image,
      .imageType = VK_IMAGE_TYPE_2D, .format = VK_FORMAT_R8G8B8A8_UNORM, .extent = {7, 3, 1},
      .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
      .tiling = VK_IMAGE_TILING_LINEAR,
      .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT};
   VkImage image;
   CHECK(v->CreateImage(dev, &image_info, NULL, &image) == VK_SUCCESS);
   VkMemoryRequirements image_req;
   v->GetImageMemoryRequirements(dev, image, &image_req);
   CHECK(image_req.size == 192 && image_req.memoryTypeBits == 4);
   VkMemoryDedicatedAllocateInfo dedicated = {.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
      .image = image};
   import = (VkImportMemoryFdInfoKHR) {.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
      .pNext = &dedicated, .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
      .fd = dup(objects[0])};
   ai = (VkMemoryAllocateInfo) {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &import,
      .allocationSize = 192, .memoryTypeIndex = 1};
   CHECK(v->AllocateMemory(dev, &ai, NULL, &memory[0]) == VK_SUCCESS);
   CHECK(v->BindImageMemory(dev, image, memory[0], 0) == VK_ERROR_INVALID_EXTERNAL_HANDLE);
   v->FreeMemory(dev, memory[0], NULL);
   /* The LOCAL-resident import maps on first use, once per GEM storage. */
   import.fd = dup(objects[0]);
   ai.memoryTypeIndex = 2;
   unsigned maps_before = maps;
   CHECK(v->AllocateMemory(dev, &ai, NULL, &memory[0]) == VK_SUCCESS && maps == maps_before);
   CHECK(v->BindImageMemory(dev, image, memory[0], 0) == VK_SUCCESS);
   CHECK(v->MapMemory(dev, memory[0], 64, VK_WHOLE_SIZE, 0, &map) == VK_SUCCESS && map);
   CHECK(maps == maps_before + 1);
   ((uint8_t *)map)[0] = 0x5a;
   v->UnmapMemory(dev, memory[0]);
   CHECK(v->MapMemory(dev, memory[0], 0, VK_WHOLE_SIZE, 0, &map) == VK_SUCCESS);
   CHECK(maps == maps_before + 1 && ((uint8_t *)map)[64] == 0x5a);
   v->UnmapMemory(dev, memory[0]);
   ai = (VkMemoryAllocateInfo) {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = 4096, .memoryTypeIndex = 0};
   CHECK(v->AllocateMemory(dev, &ai, NULL, &memory[1]) == VK_SUCCESS);
   VkImage second;
   CHECK(v->CreateImage(dev, &image_info, NULL, &second) == VK_SUCCESS);
   CHECK(v->BindImageMemory(dev, second, memory[1], 0) == VK_ERROR_INVALID_EXTERNAL_HANDLE);
   v->DestroyImage(dev, second, NULL);
   v->DestroyImage(dev, image, NULL);
   const uint64_t linear = 0, other = 0x0100000000000001ull;
   VkImageDrmFormatModifierListCreateInfoEXT list = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_LIST_CREATE_INFO_EXT,
      .drmFormatModifierCount = 1, .pDrmFormatModifiers = &other};
   external_image.pNext = &list;
   image_info.tiling = VK_IMAGE_TILING_DRM_FORMAT_MODIFIER_EXT;
   CHECK(v->CreateImage(dev, &image_info, NULL, &image) == VK_ERROR_FORMAT_NOT_SUPPORTED);
   list.pDrmFormatModifiers = &linear;
   image_info.mipLevels = 2;
   CHECK(v->CreateImage(dev, &image_info, NULL, &image) == VK_ERROR_FORMAT_NOT_SUPPORTED);
   image_info.mipLevels = 1;
   CHECK(v->CreateImage(dev, &image_info, NULL, &image) == VK_SUCCESS);
   VkImageDrmFormatModifierPropertiesEXT modifier = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_PROPERTIES_EXT, .drmFormatModifier = other};
   CHECK(v->GetImageDrmFormatModifierPropertiesEXT(dev, image, &modifier) == VK_SUCCESS &&
         modifier.drmFormatModifier == linear);
   VkImageSubresource plane = {.aspectMask = VK_IMAGE_ASPECT_MEMORY_PLANE_0_BIT_EXT};
   VkSubresourceLayout layout;
   v->GetImageSubresourceLayout(dev, image, &plane, &layout);
   CHECK(!layout.offset && layout.rowPitch == 64 && layout.size == 192);
   v->DestroyImage(dev, image, NULL);
   /* An explicit layout must use the linear pitch; its offset moves the image. */
   VkSubresourceLayout plane_layout = {.offset = 128, .rowPitch = 128};
   VkImageDrmFormatModifierExplicitCreateInfoEXT explicit_layout = {
      .sType = VK_STRUCTURE_TYPE_IMAGE_DRM_FORMAT_MODIFIER_EXPLICIT_CREATE_INFO_EXT,
      .drmFormatModifier = linear, .drmFormatModifierPlaneCount = 1, .pPlaneLayouts = &plane_layout};
   external_image.pNext = &explicit_layout;
   CHECK(v->CreateImage(dev, &image_info, NULL, &image) == VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT);
   plane_layout.rowPitch = 64;
   plane_layout.offset = 96;
   CHECK(v->CreateImage(dev, &image_info, NULL, &image) == VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT);
   plane_layout.offset = 128;
   CHECK(v->CreateImage(dev, &image_info, NULL, &image) == VK_SUCCESS);
   v->GetImageMemoryRequirements(dev, image, &image_req);
   CHECK(image_req.size == 320 && image_req.memoryTypeBits == 4);
   VkDeviceImageMemoryRequirements device_req = {
      .sType = VK_STRUCTURE_TYPE_DEVICE_IMAGE_MEMORY_REQUIREMENTS, .pCreateInfo = &image_info};
   VkMemoryRequirements2 req2 = {.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
   v->GetDeviceImageMemoryRequirements(dev, &device_req, &req2);
   CHECK(req2.memoryRequirements.size == 320 && req2.memoryRequirements.memoryTypeBits == 4);
   v->GetImageSubresourceLayout(dev, image, &plane, &layout);
   CHECK(layout.offset == 128 && layout.rowPitch == 64);
   CHECK(v->BindImageMemory(dev, image, memory[0], 0) == VK_ERROR_OUT_OF_DEVICE_MEMORY);
   v->DestroyImage(dev, image, NULL);
   v->FreeMemory(dev, memory[1], NULL);
   v->FreeMemory(dev, memory[0], NULL);
   CHECK(!live);
   /* Owned LOCAL-resident storage: GEM_LOCAL, a coherent mapping (the mock
    * rejects every GEM_TRANSFER), dma-buf export and external images. */
   expected_create_flags = APEX_DRM_GEM_LOCAL;
   ai = (VkMemoryAllocateInfo) {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .pNext = &export, .allocationSize = 8192, .memoryTypeIndex = 2};
   CHECK(v->AllocateMemory(dev, &ai, NULL, &memory[0]) == VK_SUCCESS);
   CHECK(v->MapMemory(dev, memory[0], 0, VK_WHOLE_SIZE, 0, &map) == VK_SUCCESS && map);
   memset(map, 0x3c, 8192);
   VkMappedMemoryRange range = {.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
      .memory = memory[0], .size = VK_WHOLE_SIZE};
   CHECK(v->FlushMappedMemoryRanges(dev, 1, &range) == VK_SUCCESS);
   CHECK(v->InvalidateMappedMemoryRanges(dev, 1, &range) == VK_SUCCESS);
   CHECK(((uint8_t *)map)[8191] == 0x3c);
   get = (VkMemoryGetFdInfoKHR) {.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR,
      .memory = memory[0], .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT};
   CHECK(v->GetMemoryFdKHR(dev, &get, &exported) == VK_SUCCESS && !close(exported));
   external_image.pNext = NULL;
   image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
   CHECK(v->CreateImage(dev, &image_info, NULL, &image) == VK_SUCCESS);
   CHECK(v->BindImageMemory(dev, image, memory[0], 4096) == VK_SUCCESS);
   v->DestroyImage(dev, image, NULL);
   v->UnmapMemory(dev, memory[0]);
   v->FreeMemory(dev, memory[0], NULL);
   CHECK(!live);
   expected_create_flags = 0;
   ai = (VkMemoryAllocateInfo) {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
      .allocationSize = 8192, .memoryTypeIndex = 1};
   import.pNext = &export;
   apex_device_finish(&device);
   coherent = false;
   CHECK(apex_device_init(&device, &physical, &di, NULL, fd, APEX_TRANSPORT_DRM) == VK_SUCCESS);
   image_info.pNext = &external_image;
   external_image.pNext = NULL;
   image_info.tiling = VK_IMAGE_TILING_LINEAR;
   CHECK(v->CreateImage(dev, &image_info, NULL, &image) == VK_ERROR_FORMAT_NOT_SUPPORTED);
   CHECK(v->GetMemoryFdPropertiesKHR(dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT, objects[0], &props) == VK_ERROR_INVALID_EXTERNAL_HANDLE);
   ai.pNext = &export;
   CHECK(v->AllocateMemory(dev, &ai, NULL, &memory[0]) == VK_ERROR_FEATURE_NOT_PRESENT);
   CHECK(v->CreateBuffer(dev, &bi, NULL, &buffers[0]) == VK_ERROR_INVALID_EXTERNAL_HANDLE);
   apex_device_finish(&device);
   /* Host coherence is independent of PRIME. Check both type orderings and
    * never turn a failed coherent GEM request into a shadow allocation. */
   for (unsigned caps = 0; caps < 4; caps++) {
      coherent = caps & 1;
      host_coherent = caps & 2;
      CHECK(apex_device_init(&device, &physical, &di, NULL, fd, APEX_TRANSPORT_DRM) == VK_SUCCESS);
      ai = (VkMemoryAllocateInfo) {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
         .allocationSize = 8192, .memoryTypeIndex = coherent ? 2 : 1};
      before = creates;
      if (!host_coherent) {
         ai.memoryTypeIndex = 1 + 2 * coherent; /* One past the last type. */
         CHECK(v->AllocateMemory(dev, &ai, NULL, &memory[0]) == VK_ERROR_FEATURE_NOT_PRESENT);
         CHECK(!memory[0] && creates == before);
         apex_device_finish(&device);
         continue;
      }
      expected_create_flags = APEX_DRM_GEM_HOST_COHERENT;
      fail_create = true;
      CHECK(v->AllocateMemory(dev, &ai, NULL, &memory[0]) == VK_ERROR_OUT_OF_DEVICE_MEMORY);
      CHECK(!memory[0] && !live && creates == before + 1);
      fail_create = false;
      fail_bind = true;
      CHECK(v->AllocateMemory(dev, &ai, NULL, &memory[0]) == VK_ERROR_OUT_OF_DEVICE_MEMORY);
      CHECK(!memory[0] && !live);
      fail_bind = false;
      CHECK(v->AllocateMemory(dev, &ai, NULL, &memory[0]) == VK_SUCCESS);
      CHECK(v->MapMemory(dev, memory[0], 4096, 4096, 0, &map) == VK_SUCCESS);
      uint32_t *words = map;
      for (unsigned i = 0; i < 1024; i++) words[i] = 0x59f10000 + i * 37;
      VkMappedMemoryRange range = {.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE,
         .memory = memory[0], .offset = 4096, .size = VK_WHOLE_SIZE};
      /* The mock rejects every GEM_TRANSFER. Coherent flush/invalidate are
       * valid no-ops, including subranges, without touching adjacent bytes. */
      CHECK(v->FlushMappedMemoryRanges(dev, 1, &range) == VK_SUCCESS);
      range.size = 64;
      CHECK(v->InvalidateMappedMemoryRanges(dev, 1, &range) == VK_SUCCESS);
      for (unsigned i = 0; i < 1024; i++) CHECK(words[i] == 0x59f10000 + i * 37);
      range.size = 4097;
      CHECK(v->FlushMappedMemoryRanges(dev, 1, &range) == VK_ERROR_MEMORY_MAP_FAILED);
      void *invalid = (void *)1;
      CHECK(v->MapMemory(dev, memory[0], 8192, 1, 0, &invalid) == VK_ERROR_MEMORY_MAP_FAILED && !invalid);
      bi.pNext = NULL;
      CHECK(v->CreateBuffer(dev, &bi, NULL, &buffers[0]) == VK_SUCCESS);
      VkMemoryRequirements req;
      v->GetBufferMemoryRequirements(dev, buffers[0], &req);
      CHECK(req.memoryTypeBits == (coherent ? 15 : 3));
      CHECK(v->BindBufferMemory(dev, buffers[0], memory[0], 4096) == VK_SUCCESS);
      image_info = (VkImageCreateInfo) {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
         .imageType = VK_IMAGE_TYPE_2D, .format = VK_FORMAT_R32_UINT,
         .extent = {7, 3, 1}, .mipLevels = 1, .arrayLayers = 1,
         .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_LINEAR,
         .usage = VK_IMAGE_USAGE_STORAGE_BIT};
      CHECK(v->CreateImage(dev, &image_info, NULL, &image) == VK_SUCCESS);
      v->GetImageMemoryRequirements(dev, image, &req);
      CHECK(req.memoryTypeBits == (coherent ? 15 : 3));
      CHECK(v->BindImageMemory(dev, image, memory[0], 0) == VK_SUCCESS);
      ai.pNext = &export;
      CHECK(v->AllocateMemory(dev, &ai, NULL, &memory[1]) == VK_ERROR_INVALID_EXTERNAL_HANDLE);
      import.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
      import.fd = objects[0];
      ai.pNext = &import;
      CHECK(v->AllocateMemory(dev, &ai, NULL, &memory[1]) == VK_ERROR_INVALID_EXTERNAL_HANDLE);
      CHECK(fcntl(import.fd, F_GETFD) >= 0);
      get.memory = memory[0];
      CHECK(v->GetMemoryFdKHR(dev, &get, &exported) == VK_ERROR_INVALID_EXTERNAL_HANDLE);
      v->DestroyImage(dev, image, NULL);
      v->DestroyBuffer(dev, buffers[0], NULL);
      v->UnmapMemory(dev, memory[0]);
      v->FreeMemory(dev, memory[0], NULL);
      CHECK(!live);
      /* LOCAL-resident storage follows the host-coherent type. */
      ai = (VkMemoryAllocateInfo) {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
         .allocationSize = 8192, .memoryTypeIndex = 3};
      expected_create_flags = APEX_DRM_GEM_LOCAL;
      CHECK(v->AllocateMemory(dev, &ai, NULL, &memory[0]) ==
            (coherent ? VK_SUCCESS : VK_ERROR_FEATURE_NOT_PRESENT));
      if (coherent)
         v->FreeMemory(dev, memory[0], NULL);
      expected_create_flags = 0;
      CHECK(!live);
      apex_device_finish(&device);
   }
   puts("PASS Apex host-coherent memory: independent capability, GEM flag/no fallback, range no-ops, buffer/image types, external rejection, cleanup (mock DRM)");
   CHECK(!close(fd));
   for (unsigned i = 0; i < object_count; i++) CHECK(!close(objects[i]));
   vk_physical_device_finish(&physical);
   vk_instance_finish(&instance);
   puts("PASS Apex external memory: device-local import types, LOCAL-resident external images, lazy import maps and LINEAR modifiers, fd ownership, same-file handle dedup, alias offsets, both free orders, failure cleanup (mock PRIME, no GPU execution)");
   return 0;
}
