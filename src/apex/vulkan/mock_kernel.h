/* SPDX-License-Identifier: MIT */
/* Offline model of the Apex DRM file and command processor for host tests.
 * A memfd stands in for the DRM file: GEM objects, the status page and the
 * doorbell page are ranges of it, so the driver's mmap()s see the same bytes
 * the model reads and writes. mock_cp_run() executes the ring the way the
 * card does (Docs/architecture.md, Command processor and user-mode rings),
 * with dispatches reported to a callback instead of run. No device node is
 * opened and no FPGA is touched. */
#ifndef APEX_MOCK_KERNEL_H
#define APEX_MOCK_KERNEL_H
#include "apex_cp.h"
#include "drm-uapi/apex_drm.h"
#include <errno.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define MOCK_CHECK(x) do { if (!(x)) { \
   fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); abort(); \
} } while (0)

#define MOCK_STATUS_VA 0x1000ull
#define MOCK_QUEUE_ID 5u

struct mock_gem {
   uint64_t offset, size, va;
   uint32_t flags, vm_flags;
   bool live;
};

struct mock_dispatch {
   uint32_t state[0x100];
   uint32_t grid[3];
};

/* A draw packet with the registers it snapshots. */
struct mock_draw {
   uint32_t state[0x100];
   uint32_t op, payload[6];
};

struct mock_kernel {
   int fd;
   uint64_t next_offset;
   struct mock_gem gems[256];
   unsigned gem_count, live, creates, binds, ioctls;
   /* A lost device closes GEMs it keeps bound; the kernel unmaps at close. */
   bool bound_close;
   /* Queue: at most one per file in these tests. */
   bool queue;
   uint64_t ring_va, status_offset, doorbell_offset;
   uint32_t ring_bytes, rptr;
   struct drm_apex_queue_fence fences[64];
   struct drm_apex_queue_wait waits[64];
   unsigned fence_count, wait_count;
   /* Command-processor model. */
   uint32_t state[0x100];
   uint64_t time;
   unsigned packets[256];
   bool blocked, in_pass;
   /* PREDICATE: draws and dispatches skip while set; `skipped` counts them. */
   bool predicate;
   unsigned skipped;
   void (*on_dispatch)(struct mock_kernel *, const struct mock_dispatch *);
   void (*on_draw)(struct mock_kernel *, const struct mock_draw *);
   void *data;
};

static inline void
mock_kernel_init(struct mock_kernel *k)
{
   memset(k, 0, sizeof(*k));
   k->fd = memfd_create("apex-mock-drm", MFD_CLOEXEC);
   MOCK_CHECK(k->fd >= 0);
   k->next_offset = 1 << 20;
}

static inline void
mock_kernel_finish(struct mock_kernel *k)
{
   MOCK_CHECK(!close(k->fd));
}

static inline uint64_t
mock_alloc(struct mock_kernel *k, uint64_t size)
{
   uint64_t offset = k->next_offset;
   k->next_offset += (size + 4095) / 4096 * 4096 + 4096;
   MOCK_CHECK(!ftruncate(k->fd, k->next_offset));
   return offset;
}

/* File offset of `bytes` at GPUVA `va`: a bound GEM or the status page. */
static inline uint64_t
mock_offset(struct mock_kernel *k, uint64_t va, uint64_t bytes)
{
   if (k->queue && va >= MOCK_STATUS_VA && va + bytes <= MOCK_STATUS_VA + 4096)
      return k->status_offset + va - MOCK_STATUS_VA;
   for (unsigned h = 1; h <= k->gem_count; h++) {
      const struct mock_gem *g = &k->gems[h];
      if (g->live && g->va && va >= g->va && va + bytes <= g->va + g->size)
         return g->offset + va - g->va;
   }
   fprintf(stderr, "mock: unmapped GPUVA %#llx+%llu\n", (unsigned long long)va,
           (unsigned long long)bytes);
   abort();
}

static inline void
mock_read(struct mock_kernel *k, uint64_t va, void *data, uint64_t bytes)
{
   MOCK_CHECK(pread(k->fd, data, bytes, mock_offset(k, va, bytes)) == (ssize_t)bytes);
}

static inline void
mock_write(struct mock_kernel *k, uint64_t va, const void *data, uint64_t bytes)
{
   MOCK_CHECK(pwrite(k->fd, data, bytes, mock_offset(k, va, bytes)) == (ssize_t)bytes);
}

static inline uint32_t
mock_read32(struct mock_kernel *k, uint64_t va)
{
   uint32_t value;
   mock_read(k, va, &value, 4);
   return value;
}

static inline uint64_t
mock_read64(struct mock_kernel *k, uint64_t va)
{
   uint64_t value;
   mock_read(k, va, &value, 8);
   return value;
}

static inline void
mock_write64(struct mock_kernel *k, uint64_t va, uint64_t value)
{
   mock_write(k, va, &value, 8);
}

static inline uint32_t
mock_status32(struct mock_kernel *k, unsigned offset)
{
   uint32_t value;
   MOCK_CHECK(pread(k->fd, &value, 4, k->status_offset + offset) == 4);
   return value;
}

static inline void
mock_set_status32(struct mock_kernel *k, unsigned offset, uint32_t value)
{
   MOCK_CHECK(pwrite(k->fd, &value, 4, k->status_offset + offset) == 4);
}

/* The DRM ioctls of the uapi plus GEM_CLOSE. */
static inline int
mock_kernel_ioctl(struct mock_kernel *k, unsigned long request, void *arg)
{
   k->ioctls++;
   switch (request) {
   case DRM_IOCTL_APEX_INFO:
      *(struct drm_apex_info *)arg = (struct drm_apex_info) {
         .local_bytes = 8ull << 30, .visible_bytes = 8ull << 30, .timestamp_hz = 250000000,
         .queue_slots = 64, .vm_slots = 64, .queues_per_file = 8,
         .cores = 2, .texture_units = 1, .tile_planes = 16,
      };
      return 0;
   case DRM_IOCTL_APEX_GEM_CREATE: {
      struct drm_apex_gem_create *r = arg;
      MOCK_CHECK(!r->handle && r->size && !(r->size % 4096) && !(r->flags & ~APEX_GEM_SYSTEM));
      unsigned h = ++k->gem_count;
      MOCK_CHECK(h < 256);
      k->gems[h] = (struct mock_gem){.offset = mock_alloc(k, r->size), .size = r->size,
                                     .flags = r->flags, .live = true};
      r->handle = h;
      k->live++;
      k->creates++;
      return 0;
   }
   case DRM_IOCTL_APEX_GEM_MMAP: {
      struct drm_apex_gem_mmap *r = arg;
      MOCK_CHECK(!r->reserved && r->handle <= k->gem_count && k->gems[r->handle].live);
      r->offset = k->gems[r->handle].offset;
      return 0;
   }
   case DRM_IOCTL_APEX_VM_BIND: {
      struct drm_apex_vm_bind *r = arg;
      MOCK_CHECK(!r->reserved && r->op_count);
      const struct drm_apex_vm_bind_op *ops = (const void *)(uintptr_t)r->ops;
      for (unsigned i = 0; i < r->op_count; i++) {
         const struct drm_apex_vm_bind_op *op = &ops[i];
         MOCK_CHECK(!op->reserved && !op->offset && op->bytes && !(op->bytes % 4096) &&
                    !(op->va % 4096) && op->va >= 2u << 20 && op->va + op->bytes <= 1ull << 40);
         if (op->op == APEX_VM_MAP) {
            struct mock_gem *g = &k->gems[op->handle];
            MOCK_CHECK(op->handle && op->handle <= k->gem_count && g->live && !g->va &&
                       op->bytes == g->size && op->flags && !(op->flags & ~7u));
            g->va = op->va;
            g->vm_flags = op->flags;
            k->binds++;
         } else {
            MOCK_CHECK(op->op == APEX_VM_UNMAP && !op->flags && !op->handle);
            unsigned h;
            for (h = 1; h <= k->gem_count && !(k->gems[h].live && k->gems[h].va == op->va); h++);
            MOCK_CHECK(h <= k->gem_count && k->gems[h].size == op->bytes);
            k->gems[h].va = 0;
         }
      }
      return 0;
   }
   case DRM_IOCTL_GEM_CLOSE: {
      struct drm_gem_close *r = arg;
      MOCK_CHECK(r->handle <= k->gem_count && k->gems[r->handle].live &&
                 (k->bound_close || !k->gems[r->handle].va));
      k->gems[r->handle].live = false;
      k->gems[r->handle].va = 0;
      k->live--;
      return 0;
   }
   case DRM_IOCTL_APEX_QUEUE_CREATE: {
      struct drm_apex_queue_create *r = arg;
      MOCK_CHECK(!k->queue && !r->reserved && !r->queue_id && !r->status_va);
      MOCK_CHECK(r->ring_bytes >= 4096 && r->ring_bytes <= 1u << 20 &&
                 !(r->ring_bytes & (r->ring_bytes - 1)) && !(r->ring_va % 4096) && r->priority <= 1);
      mock_offset(k, r->ring_va, r->ring_bytes);
      k->queue = true;
      k->ring_va = r->ring_va;
      k->ring_bytes = r->ring_bytes;
      k->status_offset = mock_alloc(k, 4096);
      k->doorbell_offset = mock_alloc(k, 4096);
      k->rptr = 0;
      mock_set_status32(k, APEX_STATUS_STATE, APEX_STATUS_ENABLED);
      r->queue_id = MOCK_QUEUE_ID;
      r->status_va = MOCK_STATUS_VA;
      r->status_offset = k->status_offset;
      r->doorbell_offset = k->doorbell_offset;
      return 0;
   }
   case DRM_IOCTL_APEX_QUEUE_DESTROY: {
      struct drm_apex_queue_destroy *r = arg;
      MOCK_CHECK(k->queue && r->queue_id == MOCK_QUEUE_ID && !r->reserved);
      k->queue = false;
      return 0;
   }
   case DRM_IOCTL_APEX_QUEUE_FENCE: {
      struct drm_apex_queue_fence *r = arg;
      MOCK_CHECK(r->queue_id == MOCK_QUEUE_ID && r->syncobj && k->fence_count < 64);
      k->fences[k->fence_count++] = *r;
      return 0;
   }
   case DRM_IOCTL_APEX_QUEUE_WAIT: {
      struct drm_apex_queue_wait *r = arg;
      MOCK_CHECK(r->queue_id == MOCK_QUEUE_ID && r->syncobj && r->value && k->wait_count < 64);
      k->waits[k->wait_count++] = *r;
      return 0;
   }
   case DRM_IOCTL_APEX_QUEUE_STATUS: {
      struct drm_apex_queue_status *r = arg;
      MOCK_CHECK(r->queue_id == MOCK_QUEUE_ID && !r->reserved);
      r->state = mock_status32(k, APEX_STATUS_STATE);
      r->fault_status = mock_status32(k, APEX_STATUS_FAULT);
      r->fault_unit = mock_status32(k, APEX_STATUS_FAULT + 4);
      r->fault_address = 0;
      r->error = r->state & APEX_STATUS_FAULTED ? -EIO : 0;
      return 0;
   }
   default:
      errno = ENOTTY;
      return -1;
   }
}

/* Payload dword counts: exact for fixed opcodes, a minimum otherwise. */
static inline bool
mock_count_ok(uint32_t op, uint32_t count)
{
   switch (op) {
   case APEX_CP_NOP: return true;
   case APEX_CP_INDIRECT: return count == 3;
   case APEX_CP_SET_STATE: return count >= 2;
   case APEX_CP_DISPATCH: return count == 3;
   case APEX_CP_DISPATCH_INDIRECT: return count == 2;
   case APEX_CP_DRAW: return count == 4;
   case APEX_CP_DRAW_INDEXED: return count == 5;
   case APEX_CP_DRAW_INDIRECT: case APEX_CP_DRAW_INDEXED_INDIRECT: return count == 6;
   case APEX_CP_BEGIN_PASS: case APEX_CP_END_PASS: return count == 0;
   case APEX_CP_COPY: return count == 6;
   case APEX_CP_FILL: return count == 5;
   case APEX_CP_COPY_RECT: return count == 11;
   case APEX_CP_BARRIER: return count == 2;
   case APEX_CP_WAIT: case APEX_CP_SIGNAL: return count == 4;
   case APEX_CP_KFENCE: return count == 2;
   case APEX_CP_WRITE: return count >= 4;
   case APEX_CP_TIMESTAMP: return count == 3;
   case APEX_CP_QUERY_BEGIN: return count == 1;
   case APEX_CP_QUERY_END: return count == 3;
   case APEX_CP_PREDICATE: return count == 3;
   default: return false;
   }
}

#define MOCK_U64(p, i) ((uint64_t)(p)[i] | (uint64_t)(p)[(i) + 1] << 32)

/* Executes `dwords` packets of one stream. Returns false when a WAIT is
 * unsatisfied (the stream stops before the WAIT). */
static inline bool
mock_cp_stream(struct mock_kernel *k, uint64_t base, uint32_t mask, uint32_t *position,
               uint32_t end, unsigned depth)
{
   while (*position != end) {
      uint32_t start = *position;
      uint32_t header;
      mock_read(k, base + (uint64_t)(start & mask) * 4, &header, 4);
      uint32_t op = header & 0xff, count = header >> 8 & 0xffff;
      MOCK_CHECK(!(header >> 24) && mock_count_ok(op, count) && end - start >= count + 1);
      uint32_t *p = calloc(count + 1, 4);
      MOCK_CHECK(p);
      for (uint32_t i = 0; i < count; i++)
         mock_read(k, base + (uint64_t)((start + 1 + i) & mask) * 4, &p[i], 4);
      k->packets[op]++;
      bool skippable = op == APEX_CP_DISPATCH || op == APEX_CP_DISPATCH_INDIRECT ||
                       (op >= APEX_CP_DRAW && op <= APEX_CP_DRAW_INDEXED_INDIRECT);
      if (skippable && k->predicate) {
         k->skipped++;
         op = APEX_CP_NOP;
      }
      switch (op) {
      case APEX_CP_INDIRECT: {
         MOCK_CHECK(depth < 2 && p[2] && !(p[0] & 3));
         uint32_t ib = 0;
         MOCK_CHECK(mock_cp_stream(k, MOCK_U64(p, 0), UINT32_MAX, &ib, p[2], depth + 1));
         break;
      }
      case APEX_CP_SET_STATE:
         /* The register file's accepted indices. */
         for (uint32_t r = p[0]; r < p[0] + count - 1; r++)
            MOCK_CHECK(r <= 0x009 || (r >= 0x010 && r <= 0x02a) || (r >= 0x030 && r <= 0x04b) ||
                       (r >= 0x050 && r <= 0x06f) || (r >= 0x080 && r <= 0x0ff));
         memcpy(&k->state[p[0]], &p[1], (count - 1) * 4);
         break;
      case APEX_CP_BEGIN_PASS: {
         /* The pass registers become the pass record at the pool's start;
          * the draw region holds at least one 1 KiB slot. */
         uint64_t pool = MOCK_U64(k->state, APEX_STATE_PASS);
         MOCK_CHECK(!k->in_pass && !(pool & 63) && k->state[APEX_STATE_PASS + 2] >= 1u << 20 &&
                    k->state[APEX_STATE_PASS + 8] >= 1024);
         mock_write(k, pool, &k->state[APEX_STATE_PASS], 512);
         k->in_pass = true;
         break;
      }
      case APEX_CP_END_PASS:
         MOCK_CHECK(k->in_pass);
         k->in_pass = false;
         break;
      case APEX_CP_DRAW: case APEX_CP_DRAW_INDEXED:
      case APEX_CP_DRAW_INDIRECT: case APEX_CP_DRAW_INDEXED_INDIRECT: {
         MOCK_CHECK(k->in_pass && MOCK_U64(k->state, APEX_STATE_VERTEX));
         struct mock_draw d = {.op = op};
         memcpy(d.state, k->state, sizeof(d.state));
         memcpy(d.payload, p, count * 4);
         if (k->on_draw)
            k->on_draw(k, &d);
         break;
      }
      case APEX_CP_QUERY_BEGIN:
         MOCK_CHECK(p[0] < 8);
         break;
      case APEX_CP_QUERY_END:
         MOCK_CHECK(p[0] < 8 && !(p[1] & 7));
         break;
      case APEX_CP_PREDICATE:
         /* Bit 0 reads the u32 as the packet executes; bit 1 inverts. */
         MOCK_CHECK(!(p[2] & ~3u));
         if (p[2] & 1) {
            MOCK_CHECK(!(p[0] & 3));
            k->predicate = (mock_read32(k, MOCK_U64(p, 0)) == 0) != ((p[2] >> 1) & 1);
         } else {
            k->predicate = false;
         }
         break;
      case APEX_CP_DISPATCH: {
         struct mock_dispatch d;
         memcpy(d.state, k->state, sizeof(d.state));
         memcpy(d.grid, p, sizeof(d.grid));
         if (k->on_dispatch && p[0] && p[1] && p[2])
            k->on_dispatch(k, &d);
         break;
      }
      case APEX_CP_COPY: {
         uint64_t bytes = MOCK_U64(p, 4);
         uint8_t *data = malloc(bytes ? bytes : 1);
         MOCK_CHECK(data);
         mock_read(k, MOCK_U64(p, 0), data, bytes);
         mock_write(k, MOCK_U64(p, 2), data, bytes);
         free(data);
         break;
      }
      case APEX_CP_FILL: {
         uint64_t va = MOCK_U64(p, 0), bytes = MOCK_U64(p, 2);
         MOCK_CHECK(!(va & 3) && !(bytes & 3));
         for (uint64_t i = 0; i < bytes; i += 4)
            mock_write(k, va + i, &p[4], 4);
         break;
      }
      case APEX_CP_WRITE:
         MOCK_CHECK(!(p[0] & 3) && p[2] <= 1);
         mock_write(k, MOCK_U64(p, 0), &p[3], (count - 3) * 4);
         break;
      case APEX_CP_WAIT:
         if (mock_read64(k, MOCK_U64(p, 0)) < MOCK_U64(p, 2)) {
            free(p);
            k->blocked = true;
            return false;
         }
         break;
      case APEX_CP_SIGNAL:
         mock_write64(k, MOCK_U64(p, 0), MOCK_U64(p, 2));
         break;
      case APEX_CP_KFENCE:
         MOCK_CHECK(depth == 0);
         MOCK_CHECK(pwrite(k->fd, p, 8, k->status_offset + APEX_STATUS_KFENCE) == 8);
         break;
      case APEX_CP_TIMESTAMP:
         mock_write64(k, MOCK_U64(p, 0), k->time += 1000);
         break;
      case APEX_CP_BARRIER:
         MOCK_CHECK(!(p[0] & ~7u) && !(p[1] & ~31u));
         break;
      default:
         break;
      }
      free(p);
      *position = start + count + 1;
   }
   return true;
}

/* Runs the ring from rptr to the doorbell's wptr, then publishes rptr. */
static inline void
mock_cp_run(struct mock_kernel *k)
{
   uint32_t wptr;
   MOCK_CHECK(pread(k->fd, &wptr, 4, k->doorbell_offset) == 4);
   MOCK_CHECK(wptr - k->rptr <= k->ring_bytes / 4);
   k->blocked = false;
   mock_cp_stream(k, k->ring_va, k->ring_bytes / 4 - 1, &k->rptr, wptr, 0);
   mock_set_status32(k, APEX_STATUS_RPTR, k->rptr);
   /* The doorbell page reads back the live rptr. */
   MOCK_CHECK(pwrite(k->fd, &k->rptr, 4, k->doorbell_offset) == 4);
}
#endif
