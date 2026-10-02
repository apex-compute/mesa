/* SPDX-License-Identifier: MIT */
#include "apex_cp.h"
#include "util/macros.h"
#include "util/os_time.h"
#include <assert.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void
apex_ib_init(struct apex_ib *ib)
{
   *ib = (struct apex_ib){0};
}

void
apex_ib_finish(struct apex_ib *ib)
{
   free(ib->words);
   *ib = (struct apex_ib){0};
}

uint32_t *
apex_ib_reserve(struct apex_ib *ib, uint32_t count)
{
   if (ib->failed)
      return NULL;
   if (count > UINT32_MAX - ib->count) {
      ib->failed = true;
      return NULL;
   }
   if (ib->count + count > ib->capacity) {
      uint32_t capacity = MAX2(ib->capacity, 256);
      while (capacity < ib->count + count)
         capacity = capacity > UINT32_MAX / 2 ? UINT32_MAX : capacity * 2;
      uint32_t *words = realloc(ib->words, (size_t)capacity * sizeof(*words));
      if (!words) {
         ib->failed = true;
         return NULL;
      }
      ib->words = words;
      ib->capacity = capacity;
   }
   uint32_t *out = ib->words + ib->count;
   ib->count += count;
   return out;
}

void
apex_cp_packet(struct apex_ib *ib, enum apex_cp_op op, uint32_t count, const uint32_t *payload)
{
   assert(count <= APEX_CP_MAX_COUNT);
   uint32_t *out = apex_ib_reserve(ib, count + 1);
   if (!out)
      return;
   out[0] = APEX_CP_HEADER(op, count);
   if (count)
      memcpy(out + 1, payload, (size_t)count * sizeof(*payload));
}

#define LO(v) ((uint32_t)(v))
#define HI(v) ((uint32_t)((uint64_t)(v) >> 32))

void
apex_cp_indirect(struct apex_ib *ib, uint64_t va, uint32_t dwords)
{
   assert(dwords && dwords <= 1u << 24 && !(va & 3));
   apex_cp_packet(ib, APEX_CP_INDIRECT, 3, (uint32_t[]){LO(va), HI(va), dwords});
}

void
apex_cp_set_state(struct apex_ib *ib, uint32_t index, uint32_t count, const uint32_t *values)
{
   assert(count && count < APEX_CP_MAX_COUNT && index + count <= 0x100);
   uint32_t *out = apex_ib_reserve(ib, count + 2);
   if (!out)
      return;
   out[0] = APEX_CP_HEADER(APEX_CP_SET_STATE, count + 1);
   out[1] = index;
   memcpy(out + 2, values, (size_t)count * sizeof(*values));
}

void
apex_cp_dispatch(struct apex_ib *ib, uint32_t x, uint32_t y, uint32_t z)
{
   apex_cp_packet(ib, APEX_CP_DISPATCH, 3, (uint32_t[]){x, y, z});
}

void
apex_cp_dispatch_indirect(struct apex_ib *ib, uint64_t va)
{
   apex_cp_packet(ib, APEX_CP_DISPATCH_INDIRECT, 2, (uint32_t[]){LO(va), HI(va)});
}

void
apex_cp_draw(struct apex_ib *ib, uint32_t vertices, uint32_t instances, uint32_t first_vertex,
             uint32_t first_instance)
{
   apex_cp_packet(ib, APEX_CP_DRAW, 4, (uint32_t[]){vertices, instances, first_vertex, first_instance});
}

void
apex_cp_draw_indexed(struct apex_ib *ib, uint32_t indices, uint32_t instances, uint32_t first_index,
                     int32_t vertex_offset, uint32_t first_instance)
{
   apex_cp_packet(ib, APEX_CP_DRAW_INDEXED, 5,
                  (uint32_t[]){indices, instances, first_index, (uint32_t)vertex_offset, first_instance});
}

void
apex_cp_draw_indirect(struct apex_ib *ib, bool indexed, uint64_t va, uint32_t count, uint32_t stride,
                      uint64_t count_va)
{
   apex_cp_packet(ib, indexed ? APEX_CP_DRAW_INDEXED_INDIRECT : APEX_CP_DRAW_INDIRECT, 6,
                  (uint32_t[]){LO(va), HI(va), count, stride, LO(count_va), HI(count_va)});
}

void
apex_cp_begin_pass(struct apex_ib *ib)
{
   apex_cp_packet(ib, APEX_CP_BEGIN_PASS, 0, NULL);
}

void
apex_cp_end_pass(struct apex_ib *ib)
{
   apex_cp_packet(ib, APEX_CP_END_PASS, 0, NULL);
}

void
apex_cp_copy(struct apex_ib *ib, uint64_t src, uint64_t dst, uint64_t bytes)
{
   apex_cp_packet(ib, APEX_CP_COPY, 6,
                  (uint32_t[]){LO(src), HI(src), LO(dst), HI(dst), LO(bytes), HI(bytes)});
}

void
apex_cp_fill(struct apex_ib *ib, uint64_t dst, uint64_t bytes, uint32_t value)
{
   assert(!(dst & 3) && !(bytes & 3));
   apex_cp_packet(ib, APEX_CP_FILL, 5, (uint32_t[]){LO(dst), HI(dst), LO(bytes), HI(bytes), value});
}

void
apex_cp_copy_rect(struct apex_ib *ib, const struct apex_cp_rect *r)
{
   apex_cp_packet(ib, APEX_CP_COPY_RECT, 11, (uint32_t[]){
      LO(r->src), HI(r->src), r->src_pitch, LO(r->dst), HI(r->dst), r->dst_pitch,
      r->row_bytes, r->rows, r->slices, r->src_slice_pitch, r->dst_slice_pitch});
}

void
apex_cp_barrier(struct apex_ib *ib, uint32_t wait_mask, uint32_t cache_mask)
{
   apex_cp_packet(ib, APEX_CP_BARRIER, 2, (uint32_t[]){wait_mask, cache_mask});
}

void
apex_cp_wait(struct apex_ib *ib, uint64_t va, uint64_t value)
{
   assert(!(va & 7));
   apex_cp_packet(ib, APEX_CP_WAIT, 4, (uint32_t[]){LO(va), HI(va), LO(value), HI(value)});
}

void
apex_cp_signal(struct apex_ib *ib, uint64_t va, uint64_t value)
{
   assert(!(va & 7));
   apex_cp_packet(ib, APEX_CP_SIGNAL, 4, (uint32_t[]){LO(va), HI(va), LO(value), HI(value)});
}

void
apex_cp_kfence(struct apex_ib *ib, uint64_t value)
{
   apex_cp_packet(ib, APEX_CP_KFENCE, 2, (uint32_t[]){LO(value), HI(value)});
}

void
apex_cp_write(struct apex_ib *ib, uint64_t va, uint32_t flags, uint32_t dwords, const uint32_t *data)
{
   assert(dwords && dwords <= APEX_CP_MAX_COUNT - 3 && !(va & 3));
   uint32_t *out = apex_ib_reserve(ib, dwords + 4);
   if (!out)
      return;
   out[0] = APEX_CP_HEADER(APEX_CP_WRITE, dwords + 3);
   out[1] = LO(va);
   out[2] = HI(va);
   out[3] = flags;
   memcpy(out + 4, data, (size_t)dwords * sizeof(*data));
}

void
apex_cp_timestamp(struct apex_ib *ib, uint64_t va, uint32_t flags)
{
   assert(!(va & 7));
   apex_cp_packet(ib, APEX_CP_TIMESTAMP, 3, (uint32_t[]){LO(va), HI(va), flags});
}

void
apex_cp_query_begin(struct apex_ib *ib, uint32_t slot)
{
   assert(slot < 8);
   apex_cp_packet(ib, APEX_CP_QUERY_BEGIN, 1, &slot);
}

void
apex_cp_query_end(struct apex_ib *ib, uint32_t slot, uint64_t va)
{
   assert(slot < 8 && !(va & 7));
   apex_cp_packet(ib, APEX_CP_QUERY_END, 3, (uint32_t[]){slot, LO(va), HI(va)});
}

void
apex_cp_predicate(struct apex_ib *ib, uint64_t va, uint32_t flags)
{
   assert(!(va & 3) && !(flags & ~(APEX_CP_PREDICATE_ENABLE | APEX_CP_PREDICATE_INVERTED)));
   apex_cp_packet(ib, APEX_CP_PREDICATE, 3, (uint32_t[]){LO(va), HI(va), flags});
}

void
apex_cp_batch(struct apex_ib *ib, const struct apex_cp_batch *batch)
{
   if (batch->acquire_cache)
      apex_cp_barrier(ib, 0, batch->acquire_cache);
   if (batch->kwait_value)
      apex_cp_wait(ib, batch->kwait_va, batch->kwait_value);
   for (uint32_t i = 0; i < batch->ib_count; i++)
      apex_cp_indirect(ib, batch->ib_va[i], batch->ib_dwords[i]);
   apex_cp_signal(ib, batch->retire_va, batch->sequence);
   if (batch->kfence)
      apex_cp_kfence(ib, batch->sequence);
}

bool
apex_ring_reserve(struct apex_ring *ring, uint32_t count, uint64_t timeout_ns)
{
   assert(count <= ring->dwords);
   /* The status page's rptr moves at INDIRECT and KFENCE retirement; the
    * doorbell read returns the live rptr at the cost of an uncached read. */
   if (ring->wptr - ring->status[APEX_STATUS_RPTR / 4] + count <= ring->dwords)
      return true;
   uint64_t deadline = os_time_get_nano() + timeout_ns;
   for (;;) {
      if (ring->status[APEX_STATUS_STATE / 4] & APEX_STATUS_FAULTED)
         return false;
      if (ring->wptr - ring->doorbell[0] + count <= ring->dwords)
         return true;
      if (os_time_get_nano() >= deadline)
         return false;
      nanosleep(&(struct timespec){.tv_nsec = 20000}, NULL);
   }
}

void
apex_ring_write(struct apex_ring *ring, const uint32_t *words, uint32_t count)
{
   uint32_t mask = ring->dwords - 1;
   for (uint32_t i = 0; i < count; i++)
      ring->map[(ring->wptr + i) & mask] = words[i];
   ring->wptr += count;
}

void
apex_ring_publish(struct apex_ring *ring)
{
   /* Write-combined ring stores reach BAR2 before the doorbell (sfence on
    * x86); the completer holds the doorbell behind earlier BAR2 writes. */
   atomic_thread_fence(memory_order_seq_cst);
#if defined(__x86_64__) || defined(__i386__)
   __builtin_ia32_sfence();
#endif
   ring->doorbell[0] = ring->wptr;
}
