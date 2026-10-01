/* SPDX-License-Identifier: MIT */
/* Hardware bring-up of GLES 2 through Zink on Apex. Explicit use only: it
 * opens the named DRM node.
 *   apex-gles-application egl NODE  GLES 2 context on the EGL device whose
 *                                    primary or render node is NODE; renders
 *                                    a clear and a textured triangle into an
 *                                    FBO and checks pixels.
 *   apex-gles-application gbm NODE   GBM allocation (implicit and explicit
 *                                    LINEAR modifier), CPU mapping, and EGL
 *                                    on the GBM platform rendering into a GBM
 *                                    buffer imported as an EGLImage.
 *   apex-gles-application caps NODE  Desktop GL and GLES 3 contexts on the
 *                                    device: versions and the extensions
 *                                    Mesa requires for GLES 3.0 and 3.1.
 *   apex-gles-application scanout CARD
 *                                    GBM scanout buffers rendered through
 *                                    Zink become KMS framebuffers; as DRM
 *                                    master with a connected sink, page-flip
 *                                    between two of them through each of
 *                                    three clear colors and check that the
 *                                    shown buffer holds the frame's color. */
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>
#include "drm-uapi/drm.h"
#include "drm-uapi/drm_mode.h"
#include "gbm.h"

#define SIZE 16

static int failures;

static void
check(bool ok, const char *what)
{
   printf("%s %s\n", ok ? "ok  " : "FAIL", what);
   failures += !ok;
}

static bool
near(unsigned value, unsigned expected)
{
   return value + 1 >= expected && value <= expected + 1;
}

static GLuint
program(void)
{
   static const char *vs =
      "attribute vec2 position;\n"
      "attribute float u;\n"
      "varying vec4 constant_color;\n"
      "varying float gradient;\n"
      "void main() {\n"
      "   constant_color = vec4(1.0, 0.0, 0.0, 1.0);\n"
      "   gradient = u;\n"
      "   gl_Position = vec4(position, 0.0, 1.0);\n"
      "}\n";
   static const char *fs =
      "precision mediump float;\n"
      "uniform sampler2D tex;\n"
      "varying vec4 constant_color;\n"
      "varying float gradient;\n"
      "void main() {\n"
      "   gl_FragColor = vec4(constant_color.r, gradient, texture2D(tex, vec2(0.75, 0.25)).g, 1.0);\n"
      "}\n";
   GLuint p = glCreateProgram();
   const char *sources[2] = {vs, fs};
   const GLenum types[2] = {GL_VERTEX_SHADER, GL_FRAGMENT_SHADER};
   for (unsigned i = 0; i < 2; i++) {
      GLuint s = glCreateShader(types[i]);
      glShaderSource(s, 1, &sources[i], NULL);
      glCompileShader(s);
      GLint ok;
      glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
      check(ok, i ? "fragment shader compiles" : "vertex shader compiles");
      glAttachShader(p, s);
   }
   glBindAttribLocation(p, 0, "position");
   glBindAttribLocation(p, 1, "u");
   glLinkProgram(p);
   GLint linked;
   glGetProgramiv(p, GL_LINK_STATUS, &linked);
   check(linked, "program links");
   return p;
}

/* Clear to (0, 0, 64, 255), then the lower-left half triangle writes
 * (255, round((x + 0.5) / 16 * 255), 255, 255): a constant varying, an
 * interpolated one and texel (1, 0) of a 2x2 nearest texture. */
static void
render_and_check(GLuint framebuffer)
{
   glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
   check(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE, "framebuffer complete");
   static const uint8_t texels[16] = {255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255};
   GLuint texture;
   glGenTextures(1, &texture);
   glBindTexture(GL_TEXTURE_2D, texture);
   glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 2, 2, 0, GL_RGBA, GL_UNSIGNED_BYTE, texels);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
   glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
   GLuint p = program();
   glUseProgram(p);
   glUniform1i(glGetUniformLocation(p, "tex"), 0);
   static const float position[] = {-1, -1, 1, -1, -1, 1};
   static const float u[] = {0, 1, 0};
   glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, position);
   glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, u);
   glEnableVertexAttribArray(0);
   glEnableVertexAttribArray(1);
   glViewport(0, 0, SIZE, SIZE);
   glClearColor(0.0f, 0.0f, 64.0f / 255.0f, 1.0f);
   glClear(GL_COLOR_BUFFER_BIT);
   glDrawArrays(GL_TRIANGLES, 0, 3);
   uint8_t pixels[SIZE * SIZE * 4];
   glReadPixels(0, 0, SIZE, SIZE, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
   check(glGetError() == GL_NO_ERROR, "no GL error");
   const uint8_t *clear = &pixels[(15 * SIZE + 15) * 4];
   check(clear[0] == 0 && clear[1] == 0 && clear[2] == 64 && clear[3] == 255, "clear color at (15, 15)");
   bool triangle = true;
   for (unsigned y = 0; y < SIZE; y++) {
      for (unsigned x = 0; x + y <= 12; x++) {
         const uint8_t *px = &pixels[(y * SIZE + x) * 4];
         unsigned green = (unsigned)((x + 0.5) / 16.0 * 255.0 + 0.5);
         if (px[0] != 255 || !near(px[1], green) || px[2] != 255 || px[3] != 255) {
            if (triangle)
               printf("     (%u, %u) = (%u, %u, %u, %u), expected (255, %u, 255, 255)\n",
                      x, y, px[0], px[1], px[2], px[3], green);
            triangle = false;
         }
      }
   }
   check(triangle, "triangle: constant and interpolated varyings, texture sample");
}

static bool
extension(const char *list, const char *name)
{
   size_t n = strlen(name);
   for (const char *p = list; p && (p = strstr(p, name)); p += n)
      if ((p == list || p[-1] == ' ') && (p[n] == ' ' || !p[n]))
         return true;
   return false;
}

static EGLContext
gles2_context(EGLDisplay display, EGLConfig *config_out)
{
   EGLint major, minor;
   check(eglInitialize(display, &major, &minor), "eglInitialize");
   printf("     EGL %d.%d %s\n", major, minor, eglQueryString(display, EGL_VENDOR));
   eglBindAPI(EGL_OPENGL_ES_API);
   const EGLint attributes[] = {EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT, EGL_SURFACE_TYPE, 0, EGL_NONE};
   EGLConfig config = NULL;
   EGLint count = 0;
   eglChooseConfig(display, attributes, &config, 1, &count);
   const EGLint context_attributes[] = {EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE};
   EGLContext context = eglCreateContext(display, count ? config : EGL_NO_CONFIG_KHR,
                                         EGL_NO_CONTEXT, context_attributes);
   check(context != EGL_NO_CONTEXT, "GLES 2 context");
   check(context && eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, context),
         "surfaceless make current");
   if (context) {
      printf("     GL_VENDOR   %s\n     GL_RENDERER %s\n     GL_VERSION  %s\n",
             glGetString(GL_VENDOR), glGetString(GL_RENDERER), glGetString(GL_VERSION));
   }
   if (config_out)
      *config_out = config;
   return context;
}

/* The EGL device with NODE as its primary or render node, so the host's
 * other GPUs are never chosen. */
static int
run_egl(const char *node)
{
   PFNEGLQUERYDEVICESEXTPROC query_devices =
      (PFNEGLQUERYDEVICESEXTPROC)eglGetProcAddress("eglQueryDevicesEXT");
   PFNEGLQUERYDEVICESTRINGEXTPROC query_string =
      (PFNEGLQUERYDEVICESTRINGEXTPROC)eglGetProcAddress("eglQueryDeviceStringEXT");
   if (!query_devices || !query_string) {
      check(false, "EGL_EXT_device_enumeration and EGL_EXT_device_query");
      return 1;
   }
   EGLDeviceEXT devices[16], device = EGL_NO_DEVICE_EXT;
   EGLint count = 0;
   query_devices(16, devices, &count);
   for (EGLint i = 0; i < count && device == EGL_NO_DEVICE_EXT; i++) {
      const char *extensions = query_string(devices[i], EGL_EXTENSIONS);
      const char *primary = extension(extensions, "EGL_EXT_device_drm") ?
         query_string(devices[i], EGL_DRM_DEVICE_FILE_EXT) : NULL;
      const char *render = extension(extensions, "EGL_EXT_device_drm_render_node") ?
         query_string(devices[i], EGL_DRM_RENDER_NODE_FILE_EXT) : NULL;
      if ((primary && !strcmp(primary, node)) || (render && !strcmp(render, node)))
         device = devices[i];
   }
   check(device != EGL_NO_DEVICE_EXT, "EGL device for the node");
   if (device == EGL_NO_DEVICE_EXT)
      return 1;
   EGLDisplay display = eglGetPlatformDisplay(EGL_PLATFORM_DEVICE_EXT, device, NULL);
   check(display != EGL_NO_DISPLAY, "device platform display");
   if (display == EGL_NO_DISPLAY || !gles2_context(display, NULL))
      return 1;
   GLuint framebuffer, renderbuffer;
   glGenFramebuffers(1, &framebuffer);
   glGenRenderbuffers(1, &renderbuffer);
   glBindRenderbuffer(GL_RENDERBUFFER, renderbuffer);
   /* GLES 2 guarantees RGBA4; RGBA8 needs OES_rgb8_rgba8. */
   check(extension((const char *)glGetString(GL_EXTENSIONS), "GL_OES_rgb8_rgba8"), "GL_OES_rgb8_rgba8");
   glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8_OES, SIZE, SIZE);
   glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
   glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, renderbuffer);
   render_and_check(framebuffer);
   eglTerminate(display);
   return failures != 0;
}

#define GBM_FUNCTIONS(X) X(gbm_create_device) X(gbm_device_destroy) X(gbm_device_get_backend_name) \
   X(gbm_bo_create) X(gbm_bo_destroy) X(gbm_bo_get_fd) X(gbm_bo_get_stride) X(gbm_bo_get_modifier) \
   X(gbm_bo_map) X(gbm_bo_unmap) X(gbm_bo_create_with_modifiers2) X(gbm_bo_get_handle)
#define DECLARE(f) static __typeof__(f) *p_##f;

GBM_FUNCTIONS(DECLARE)

/* Allocation, mapping and, for rendering buffers, an EGLImage render target
 * whose clear must reach the buffer's memory. */
static void
gbm_case(struct gbm_device *device, EGLDisplay display, uint32_t usage, bool explicit_linear,
         const char *name)
{
   /* An explicit modifier list selects Zink's VK_EXT_image_drm_format_modifier path. */
   static const uint64_t linear = 0; /* DRM_FORMAT_MOD_LINEAR */
   struct gbm_bo *bo = explicit_linear ?
      p_gbm_bo_create_with_modifiers2(device, SIZE, SIZE, GBM_FORMAT_ABGR8888, &linear, 1, usage) :
      p_gbm_bo_create(device, SIZE, SIZE, GBM_FORMAT_ABGR8888, usage);
   char what[128];
   snprintf(what, sizeof(what), "%s: gbm_bo_create", name);
   check(bo, what);
   if (!bo)
      return;
   if (explicit_linear) {
      snprintf(what, sizeof(what), "%s: modifier is DRM_FORMAT_MOD_LINEAR", name);
      check(p_gbm_bo_get_modifier(bo) == linear, what);
   }
   int fd = p_gbm_bo_get_fd(bo);
   uint32_t stride = p_gbm_bo_get_stride(bo);
   printf("     fd %d stride %u modifier 0x%llx\n", fd, stride,
          (unsigned long long)p_gbm_bo_get_modifier(bo));
   snprintf(what, sizeof(what), "%s: dma-buf export", name);
   check(fd >= 0 && stride >= SIZE * 4, what);
   uint32_t map_stride = 0;
   void *data = NULL;
   uint32_t *words = p_gbm_bo_map(bo, 0, 0, SIZE, SIZE, GBM_BO_TRANSFER_READ_WRITE, &map_stride, &data);
   snprintf(what, sizeof(what), "%s: CPU map", name);
   check(words, what);
   if (words) {
      for (unsigned y = 0; y < SIZE; y++)
         for (unsigned x = 0; x < SIZE; x++)
            words[y * map_stride / 4 + x] = 0x01000000u * y + x;
      p_gbm_bo_unmap(bo, data);
      data = NULL;
      words = p_gbm_bo_map(bo, 0, 0, SIZE, SIZE, GBM_BO_TRANSFER_READ, &map_stride, &data);
      bool same = words;
      for (unsigned y = 0; words && y < SIZE; y++)
         for (unsigned x = 0; x < SIZE; x++)
            same &= words[y * map_stride / 4 + x] == 0x01000000u * y + x;
      if (words)
         p_gbm_bo_unmap(bo, data);
      /* gbm_bo_map requires *map_data to be NULL. */
      data = NULL;
      snprintf(what, sizeof(what), "%s: CPU writes read back", name);
      check(same, what);
   }
   if (display != EGL_NO_DISPLAY && (usage & GBM_BO_USE_RENDERING) && fd >= 0) {
      PFNEGLCREATEIMAGEKHRPROC create_image = (PFNEGLCREATEIMAGEKHRPROC)eglGetProcAddress("eglCreateImageKHR");
      PFNGLEGLIMAGETARGETRENDERBUFFERSTORAGEOESPROC image_storage =
         (PFNGLEGLIMAGETARGETRENDERBUFFERSTORAGEOESPROC)eglGetProcAddress("glEGLImageTargetRenderbufferStorageOES");
      uint64_t modifier = p_gbm_bo_get_modifier(bo);
      const EGLint attributes[] = {
         EGL_WIDTH, SIZE, EGL_HEIGHT, SIZE, EGL_LINUX_DRM_FOURCC_EXT, GBM_FORMAT_ABGR8888,
         EGL_DMA_BUF_PLANE0_FD_EXT, fd, EGL_DMA_BUF_PLANE0_OFFSET_EXT, 0,
         EGL_DMA_BUF_PLANE0_PITCH_EXT, (EGLint)stride,
         EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT, (EGLint)(uint32_t)modifier,
         EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT, (EGLint)(uint32_t)(modifier >> 32), EGL_NONE,
      };
      EGLImageKHR image = create_image ?
         create_image(display, EGL_NO_CONTEXT, EGL_LINUX_DMA_BUF_EXT, NULL, attributes) : EGL_NO_IMAGE_KHR;
      snprintf(what, sizeof(what), "%s: EGLImage from the dma-buf", name);
      check(image != EGL_NO_IMAGE_KHR, what);
      if (image != EGL_NO_IMAGE_KHR && image_storage) {
         GLuint framebuffer, renderbuffer;
         glGenFramebuffers(1, &framebuffer);
         glGenRenderbuffers(1, &renderbuffer);
         glBindRenderbuffer(GL_RENDERBUFFER, renderbuffer);
         image_storage(GL_RENDERBUFFER, image);
         glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
         glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, renderbuffer);
         render_and_check(framebuffer);
         glFinish();
         /* ABGR8888 is R, G, B, A in byte order: the clear at (15, 15) is
          * 0xff400000 and the triangle at (0, 0) is 0xffff08ff. */
         words = p_gbm_bo_map(bo, 0, 0, SIZE, SIZE, GBM_BO_TRANSFER_READ, &map_stride, &data);
         bool seen = words && words[15 * map_stride / 4 + 15] == 0xff400000u &&
                     (words[0] & 0xffff00ffu) == 0xffff00ffu;
         if (words && !seen)
            printf("     read (15, 15) 0x%08x, (0, 0) 0x%08x\n", words[15 * map_stride / 4 + 15], words[0]);
         if (words)
            p_gbm_bo_unmap(bo, data);
         snprintf(what, sizeof(what), "%s: rendering reaches the GBM buffer", name);
         check(seen, what);
      }
   }
   if (fd >= 0)
      close(fd);
   p_gbm_bo_destroy(bo);
}

static int
run_gbm(const char *node)
{
   void *library = dlopen("libgbm.so.1", RTLD_NOW | RTLD_GLOBAL);
   check(library, "libgbm.so.1");
   if (!library)
      return 1;
#define RESOLVE(f) p_##f = (__typeof__(f) *)dlsym(library, #f); if (!p_##f) { check(false, #f); return 1; }
   GBM_FUNCTIONS(RESOLVE)
   int fd = open(node, O_RDWR | O_CLOEXEC);
   check(fd >= 0, "open node");
   if (fd < 0)
      return 1;
   struct gbm_device *device = p_gbm_create_device(fd);
   check(device, "gbm_create_device");
   if (!device)
      return 1;
   printf("     backend %s\n", p_gbm_device_get_backend_name(device));
   EGLDisplay display = eglGetPlatformDisplay(EGL_PLATFORM_GBM_KHR, device, NULL);
   check(display != EGL_NO_DISPLAY, "GBM platform display");
   if (display != EGL_NO_DISPLAY && !gles2_context(display, NULL))
      display = EGL_NO_DISPLAY;
   if (display != EGL_NO_DISPLAY) {
      PFNEGLQUERYDMABUFMODIFIERSEXTPROC query_modifiers =
         (PFNEGLQUERYDMABUFMODIFIERSEXTPROC)eglGetProcAddress("eglQueryDmaBufModifiersEXT");
      EGLuint64KHR modifiers[16];
      EGLint count = 0;
      bool linear = false;
      if (query_modifiers && query_modifiers(display, GBM_FORMAT_ABGR8888, 16, modifiers, NULL, &count))
         for (EGLint i = 0; i < count; i++)
            linear |= modifiers[i] == 0;
      printf("     %d dma-buf modifiers for ABGR8888\n", count);
      check(linear, "eglQueryDmaBufModifiersEXT lists DRM_FORMAT_MOD_LINEAR");
   }
   gbm_case(device, display, GBM_BO_USE_LINEAR | GBM_BO_USE_RENDERING, false, "linear rendering");
   gbm_case(device, display, GBM_BO_USE_LINEAR | GBM_BO_USE_SCANOUT | GBM_BO_USE_RENDERING, false,
            "linear scanout");
   gbm_case(device, display, GBM_BO_USE_SCANOUT | GBM_BO_USE_RENDERING, true, "LINEAR modifier");
   if (display != EGL_NO_DISPLAY)
      eglTerminate(display);
   p_gbm_device_destroy(device);
   close(fd);
   return failures != 0;
}

struct scanout_buffer {
   struct gbm_bo *bo;
   GLuint framebuffer;
   uint32_t fb;
};

/* A GBM scanout buffer as a GL render target and a KMS framebuffer. The
 * kernel admits only LOCAL-resident objects, so ADDFB2 succeeding shows the
 * Zink allocation scans out in place. */
static bool
scanout_buffer(int fd, struct gbm_device *device, EGLDisplay display, uint32_t width,
               uint32_t height, struct scanout_buffer *buffer)
{
   buffer->bo = p_gbm_bo_create(device, width, height, GBM_FORMAT_XRGB8888,
                                GBM_BO_USE_SCANOUT | GBM_BO_USE_RENDERING);
   check(buffer->bo, "scanout gbm_bo_create");
   if (!buffer->bo)
      return false;
   int dmabuf = p_gbm_bo_get_fd(buffer->bo);
   uint32_t stride = p_gbm_bo_get_stride(buffer->bo);
   PFNEGLCREATEIMAGEKHRPROC create_image = (PFNEGLCREATEIMAGEKHRPROC)eglGetProcAddress("eglCreateImageKHR");
   PFNGLEGLIMAGETARGETRENDERBUFFERSTORAGEOESPROC image_storage =
      (PFNGLEGLIMAGETARGETRENDERBUFFERSTORAGEOESPROC)eglGetProcAddress("glEGLImageTargetRenderbufferStorageOES");
   const EGLint attributes[] = {
      EGL_WIDTH, (EGLint)width, EGL_HEIGHT, (EGLint)height,
      EGL_LINUX_DRM_FOURCC_EXT, GBM_FORMAT_XRGB8888, EGL_DMA_BUF_PLANE0_FD_EXT, dmabuf,
      EGL_DMA_BUF_PLANE0_OFFSET_EXT, 0, EGL_DMA_BUF_PLANE0_PITCH_EXT, (EGLint)stride, EGL_NONE,
   };
   EGLImageKHR image = dmabuf >= 0 && create_image ?
      create_image(display, EGL_NO_CONTEXT, EGL_LINUX_DMA_BUF_EXT, NULL, attributes) : EGL_NO_IMAGE_KHR;
   if (dmabuf >= 0)
      close(dmabuf);
   check(image != EGL_NO_IMAGE_KHR && image_storage, "scanout EGLImage");
   if (image == EGL_NO_IMAGE_KHR || !image_storage)
      return false;
   GLuint renderbuffer;
   glGenFramebuffers(1, &buffer->framebuffer);
   glGenRenderbuffers(1, &renderbuffer);
   glBindRenderbuffer(GL_RENDERBUFFER, renderbuffer);
   image_storage(GL_RENDERBUFFER, image);
   glBindFramebuffer(GL_FRAMEBUFFER, buffer->framebuffer);
   glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, renderbuffer);
   check(glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE, "scanout framebuffer complete");
   struct drm_mode_fb_cmd2 command = {
      .width = width, .height = height, .pixel_format = GBM_FORMAT_XRGB8888,
      .handles[0] = p_gbm_bo_get_handle(buffer->bo).u32, .pitches[0] = stride,
   };
   bool added = !ioctl(fd, DRM_IOCTL_MODE_ADDFB2, &command);
   printf("     %ux%u stride %u: ADDFB2 %s\n", width, height, stride, added ? "accepted" : strerror(errno));
   check(added, "KMS accepts the Zink scanout buffer");
   buffer->fb = command.fb_id;
   return added;
}

static uint32_t
scanout_color(unsigned frame)
{
   return 0xbfu << (16 - 8 * (frame % 3)); /* XRGB8888 0.75 red, green, blue */
}

/* Clears flush without waiting: each flip relies on the kernel waiting for
 * the buffer's implicit fences before scanning it out. */
static void
scanout_frame(const struct scanout_buffer *buffer, unsigned frame, uint32_t width, uint32_t height)
{
   glBindFramebuffer(GL_FRAMEBUFFER, buffer->framebuffer);
   glViewport(0, 0, width, height);
   glClearColor((frame % 3 == 0) * 0.75f, (frame % 3 == 1) * 0.75f, (frame % 3 == 2) * 0.75f, 1.0f);
   glClear(GL_COLOR_BUFFER_BIT);
   glFlush();
}

/* The display reads the buffer in place, so once its flip completes the
 * buffer's memory is what reaches the sink: sample corners and center. */
static bool
scanout_shows(const struct scanout_buffer *buffer, unsigned frame, uint32_t width, uint32_t height)
{
   const uint32_t points[5][2] = {
      {0, 0}, {width - 1, 0}, {width / 2, height / 2}, {0, height - 1}, {width - 1, height - 1},
   };
   bool ok = true;
   for (unsigned i = 0; i < 5; i++) {
      uint32_t stride;
      void *data = NULL;
      uint32_t *texel = p_gbm_bo_map(buffer->bo, points[i][0], points[i][1], 1, 1,
                                     GBM_BO_TRANSFER_READ, &stride, &data);
      uint32_t value = texel ? *texel & 0xffffff : ~0u;
      if (texel)
         p_gbm_bo_unmap(buffer->bo, data);
      if (value != scanout_color(frame)) {
         printf("     frame %u (%u,%u): %#08x, expected %#08x\n", frame, points[i][0], points[i][1],
                value, scanout_color(frame));
         ok = false;
      }
   }
   return ok;
}

static bool
flip(int fd, uint32_t crtc, uint32_t fb, int timeout_ms)
{
   struct drm_mode_crtc_page_flip request = {
      .crtc_id = crtc, .fb_id = fb, .flags = DRM_MODE_PAGE_FLIP_EVENT,
   };
   if (ioctl(fd, DRM_IOCTL_MODE_PAGE_FLIP, &request))
      return false;
   struct pollfd poll_fd = {.fd = fd, .events = POLLIN};
   struct drm_event_vblank event;
   return poll(&poll_fd, 1, timeout_ms) == 1 && read(fd, &event, sizeof(event)) >= (ssize_t)sizeof(event.base) &&
          event.base.type == DRM_EVENT_FLIP_COMPLETE;
}

static double
seconds_since(const struct timespec *begin)
{
   struct timespec now;
   clock_gettime(CLOCK_MONOTONIC, &now);
   return (now.tv_sec - begin->tv_sec) + (now.tv_nsec - begin->tv_nsec) * 1e-9;
}

static int
run_scanout(const char *node)
{
   void *library = dlopen("libgbm.so.1", RTLD_NOW | RTLD_GLOBAL);
   check(library, "libgbm.so.1");
   if (!library)
      return 1;
   GBM_FUNCTIONS(RESOLVE)
   int fd = open(node, O_RDWR | O_CLOEXEC);
   check(fd >= 0, "open card node");
   if (fd < 0)
      return 1;
   struct gbm_device *device = p_gbm_create_device(fd);
   check(device, "gbm_create_device");
   if (!device)
      return 1;
   EGLDisplay display = eglGetPlatformDisplay(EGL_PLATFORM_GBM_KHR, device, NULL);
   check(display != EGL_NO_DISPLAY, "GBM platform display");
   if (display == EGL_NO_DISPLAY || !gles2_context(display, NULL))
      return 1;
   /* The first connected connector's first (preferred) mode on the CRTC. */
   uint32_t crtcs[4], connectors[4];
   struct drm_mode_card_res resources = {
      .crtc_id_ptr = (uintptr_t)crtcs, .connector_id_ptr = (uintptr_t)connectors,
      .count_crtcs = 4, .count_connectors = 4,
   };
   struct drm_mode_modeinfo modes[64], mode = {0};
   uint32_t connector = 0;
   if (!ioctl(fd, DRM_IOCTL_MODE_GETRESOURCES, &resources) && resources.count_crtcs) {
      for (unsigned i = 0; i < resources.count_connectors && i < 4 && !connector; i++) {
         struct drm_mode_get_connector query = {.connector_id = connectors[i]};
         if (ioctl(fd, DRM_IOCTL_MODE_GETCONNECTOR, &query) || query.connection != 1 || !query.count_modes)
            continue;
         query = (struct drm_mode_get_connector) {.connector_id = connectors[i],
            .modes_ptr = (uintptr_t)modes, .count_modes = 64};
         if (!ioctl(fd, DRM_IOCTL_MODE_GETCONNECTOR, &query) && query.count_modes) {
            connector = connectors[i];
            mode = modes[0];
         }
      }
   }
   uint32_t width = connector ? mode.hdisplay : 256, height = connector ? mode.vdisplay : 64;
   struct scanout_buffer buffers[2] = {0};
   bool ready = scanout_buffer(fd, device, display, width, height, &buffers[0]) &&
                scanout_buffer(fd, device, display, width, height, &buffers[1]);
   if (ready && !connector)
      puts("     no connected sink: flips skipped");
   else if (ready && ioctl(fd, DRM_IOCTL_SET_MASTER, NULL))
      puts("     not DRM master: flips skipped");
   else if (ready) {
      /* Software rasterization takes seconds per frame: the flip deadline
       * scales with the measured time of a finished first frame. */
      struct timespec begin;
      clock_gettime(CLOCK_MONOTONIC, &begin);
      scanout_frame(&buffers[0], 0, width, height);
      glFinish();
      double frame_seconds = seconds_since(&begin);
      int timeout_ms = 2000 + (int)(4000 * frame_seconds);
      struct drm_mode_crtc crtc = {
         .crtc_id = crtcs[0], .fb_id = buffers[0].fb, .set_connectors_ptr = (uintptr_t)&connector,
         .count_connectors = 1, .mode = mode, .mode_valid = 1,
      };
      bool set = !ioctl(fd, DRM_IOCTL_MODE_SETCRTC, &crtc);
      printf("     %s %ux%u@%u, first frame %.2f s\n", set ? "modeset" : "modeset failed", width, height,
             mode.vrefresh, frame_seconds);
      check(set, "modeset onto the Zink scanout buffer");
      check(!set || scanout_shows(&buffers[0], 0, width, height), "modeset buffer holds its color");
      /* Six flips show each color from each buffer. */
      const unsigned frames = 6;
      unsigned flipped = 0, shown = 0;
      clock_gettime(CLOCK_MONOTONIC, &begin);
      for (unsigned frame = 1; set && frame <= frames; frame++, flipped++) {
         scanout_frame(&buffers[frame & 1], frame, width, height);
         if (!flip(fd, crtcs[0], buffers[frame & 1].fb, timeout_ms))
            break;
         shown += scanout_shows(&buffers[frame & 1], frame, width, height);
      }
      double seconds = seconds_since(&begin);
      printf("     %u flips in %.2f s (%.2f s per flip), %u showing their color\n", flipped, seconds,
             flipped ? seconds / flipped : 0.0, shown);
      check(flipped == frames, "page flips between Zink scanout buffers");
      check(shown == frames, "flipped buffers hold their frame's color");
      crtc = (struct drm_mode_crtc) {.crtc_id = crtcs[0]};
      ioctl(fd, DRM_IOCTL_MODE_SETCRTC, &crtc);
      ioctl(fd, DRM_IOCTL_DROP_MASTER, NULL);
   }
   for (unsigned i = 0; i < 2; i++) {
      if (buffers[i].fb)
         ioctl(fd, DRM_IOCTL_MODE_RMFB, &buffers[i].fb);
      if (buffers[i].bo)
         p_gbm_bo_destroy(buffers[i].bo);
   }
   eglTerminate(display);
   p_gbm_device_destroy(device);
   close(fd);
   return failures != 0;
}

/* Extensions of Mesa's GLES 3.0 and 3.1 version checks (compute_version_es2)
 * that have extension strings; the rest are implied by these or limits. */
static const char *const es30_extensions[] = {
   "GL_ARB_half_float_vertex", "GL_ARB_internalformat_query", "GL_ARB_map_buffer_range",
   "GL_ARB_shader_texture_lod", "GL_ARB_texture_rg", "GL_ARB_depth_buffer_float",
   "GL_ARB_framebuffer_object", "GL_EXT_packed_float", "GL_EXT_texture_array",
   "GL_EXT_texture_shared_exponent", "GL_EXT_texture_sRGB", "GL_EXT_transform_feedback",
   "GL_ARB_draw_instanced", "GL_ARB_instanced_arrays", "GL_ARB_uniform_buffer_object",
   "GL_EXT_texture_snorm", "GL_ARB_ES3_compatibility",
   "GL_ARB_vertex_type_2_10_10_10_rev",
};
/* GLES-only names of the GLES 3.0 check, read from a GLES 2 context. */
static const char *const es30_es_extensions[] = {
   "GL_OES_texture_float", "GL_OES_texture_half_float", "GL_OES_texture_half_float_linear",
   "GL_OES_depth_texture_cube_map",
};
/* Additional GL 3.0 features (compute_version). */
static const char *const gl30_extensions[] = {
   "GL_NV_conditional_render", "GL_ARB_color_buffer_float", "GL_ARB_texture_float",
   "GL_EXT_texture_integer", "GL_ARB_framebuffer_sRGB", "GL_EXT_draw_buffers2",
   "GL_EXT_texture_compression_rgtc",
};
static const char *const es31_extensions[] = {
   "GL_ARB_arrays_of_arrays", "GL_ARB_compute_shader", "GL_ARB_shader_storage_buffer_object",
   "GL_ARB_shader_atomic_counters", "GL_ARB_shader_image_load_store", "GL_ARB_draw_indirect",
   "GL_ARB_explicit_uniform_location", "GL_ARB_framebuffer_no_attachments",
   "GL_ARB_shading_language_packing", "GL_ARB_stencil_texturing", "GL_ARB_texture_multisample",
   "GL_ARB_texture_gather", "GL_MESA_shader_integer_functions", "GL_EXT_shader_integer_mix",
};

static int
run_caps(const char *node)
{
   PFNEGLQUERYDEVICESEXTPROC query_devices =
      (PFNEGLQUERYDEVICESEXTPROC)eglGetProcAddress("eglQueryDevicesEXT");
   PFNEGLQUERYDEVICESTRINGEXTPROC query_string =
      (PFNEGLQUERYDEVICESTRINGEXTPROC)eglGetProcAddress("eglQueryDeviceStringEXT");
   EGLDeviceEXT devices[16], device = EGL_NO_DEVICE_EXT;
   EGLint count = 0;
   if (query_devices && query_string)
      query_devices(16, devices, &count);
   for (EGLint i = 0; i < count && device == EGL_NO_DEVICE_EXT; i++) {
      const char *extensions = query_string(devices[i], EGL_EXTENSIONS);
      const char *render = extension(extensions, "EGL_EXT_device_drm_render_node") ?
         query_string(devices[i], EGL_DRM_RENDER_NODE_FILE_EXT) : NULL;
      const char *primary = extension(extensions, "EGL_EXT_device_drm") ?
         query_string(devices[i], EGL_DRM_DEVICE_FILE_EXT) : NULL;
      if ((render && !strcmp(render, node)) || (primary && !strcmp(primary, node)))
         device = devices[i];
   }
   check(device != EGL_NO_DEVICE_EXT, "EGL device for the node");
   if (device == EGL_NO_DEVICE_EXT)
      return 1;
   EGLDisplay display = eglGetPlatformDisplay(EGL_PLATFORM_DEVICE_EXT, device, NULL);
   EGLint major, minor;
   check(display != EGL_NO_DISPLAY && eglInitialize(display, &major, &minor), "eglInitialize");
   /* GLES 3: the highest version the context reports. */
   eglBindAPI(EGL_OPENGL_ES_API);
   const EGLint es3[] = {EGL_CONTEXT_MAJOR_VERSION, 3, EGL_NONE};
   EGLContext context = eglCreateContext(display, EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, es3);
   printf("     GLES 3 context: %s\n", context != EGL_NO_CONTEXT ? "created" : "unavailable");
   if (context != EGL_NO_CONTEXT) {
      if (eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, context))
         printf("     GLES VERSION %s\n     GLSL ES %s\n", glGetString(GL_VERSION),
                glGetString(GL_SHADING_LANGUAGE_VERSION));
      eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
      eglDestroyContext(display, context);
   }
   const EGLint es2[] = {EGL_CONTEXT_MAJOR_VERSION, 2, EGL_NONE};
   context = eglCreateContext(display, EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, es2);
   if (context != EGL_NO_CONTEXT && eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, context)) {
      const char *es_extensions = (const char *)glGetString(GL_EXTENSIONS);
      for (unsigned i = 0; i < sizeof(es30_es_extensions) / sizeof(es30_es_extensions[0]); i++)
         if (!extension(es_extensions, es30_es_extensions[i]))
            printf("     missing for GLES 3.0: %s\n", es30_es_extensions[i]);
      eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
      eglDestroyContext(display, context);
   }
   /* Desktop GL compatibility context: its extension string names Mesa's
    * internal features. */
   check(eglBindAPI(EGL_OPENGL_API), "eglBindAPI(EGL_OPENGL_API)");
   context = eglCreateContext(display, EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, NULL);
   check(context != EGL_NO_CONTEXT && eglMakeCurrent(display, EGL_NO_SURFACE, EGL_NO_SURFACE, context),
         "desktop GL context");
   if (context == EGL_NO_CONTEXT)
      return 1;
   const GLubyte *(*get_string)(GLenum) = (const GLubyte *(*)(GLenum))eglGetProcAddress("glGetString");
   const char *extensions = (const char *)get_string(GL_EXTENSIONS);
   printf("     GL_RENDERER %s\n     GL_VERSION  %s\n     GLSL        %s\n", get_string(GL_RENDERER),
          get_string(GL_VERSION), get_string(GL_SHADING_LANGUAGE_VERSION));
   for (unsigned i = 0; i < sizeof(es30_extensions) / sizeof(es30_extensions[0]); i++)
      if (!extension(extensions, es30_extensions[i]))
         printf("     missing for GLES 3.0 / GL 3.x: %s\n", es30_extensions[i]);
   for (unsigned i = 0; i < sizeof(gl30_extensions) / sizeof(gl30_extensions[0]); i++)
      if (!extension(extensions, gl30_extensions[i]))
         printf("     missing for GL 3.0: %s\n", gl30_extensions[i]);
   for (unsigned i = 0; i < sizeof(es31_extensions) / sizeof(es31_extensions[0]); i++)
      if (!extension(extensions, es31_extensions[i]))
         printf("     missing for GLES 3.1: %s\n", es31_extensions[i]);
   eglTerminate(display);
   return failures != 0;
}

int
main(int argc, char **argv)
{
   if (argc != 3 || (strcmp(argv[1], "egl") && strcmp(argv[1], "gbm") && strcmp(argv[1], "caps") &&
                     strcmp(argv[1], "scanout"))) {
      fprintf(stderr, "usage: %s egl|gbm|caps|scanout DRM-NODE\n", argv[0]);
      return 2;
   }
   int result = !strcmp(argv[1], "egl") ? run_egl(argv[2]) :
                !strcmp(argv[1], "gbm") ? run_gbm(argv[2]) :
                !strcmp(argv[1], "scanout") ? run_scanout(argv[2]) : run_caps(argv[2]);
   puts(result ? "FAIL" : "PASS");
   return result;
}
