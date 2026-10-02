/* SPDX-License-Identifier: MIT */
/* Command-processor packets, indirect buffers and the user-mode ring
 * (Docs/architecture.md, Command processor and user-mode rings). */
#ifndef APEX_CP_H
#define APEX_CP_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

enum apex_cp_op {
   APEX_CP_NOP = 0x00,
   APEX_CP_INDIRECT = 0x01,
   APEX_CP_SET_STATE = 0x02,
   APEX_CP_DISPATCH = 0x10,
   APEX_CP_DISPATCH_INDIRECT = 0x11,
   APEX_CP_DRAW = 0x20,
   APEX_CP_DRAW_INDEXED = 0x21,
   APEX_CP_DRAW_INDIRECT = 0x22,
   APEX_CP_DRAW_INDEXED_INDIRECT = 0x23,
   APEX_CP_BEGIN_PASS = 0x28,
   APEX_CP_END_PASS = 0x29,
   APEX_CP_COPY = 0x30,
   APEX_CP_FILL = 0x31,
   APEX_CP_COPY_RECT = 0x32,
   APEX_CP_BARRIER = 0x40,
   APEX_CP_WAIT = 0x50,
   APEX_CP_SIGNAL = 0x51,
   APEX_CP_KFENCE = 0x52,
   APEX_CP_WRITE = 0x53,
   APEX_CP_TIMESTAMP = 0x60,
   APEX_CP_QUERY_BEGIN = 0x61,
   APEX_CP_QUERY_END = 0x62,
   APEX_CP_PREDICATE = 0x63,
   APEX_CP_TLB_INVALIDATE = 0x80,
   APEX_CP_VM_DRAIN = 0x81,
};

/* Header: opcode 7:0, payload dwords 23:8, bits 31:24 zero. */
#define APEX_CP_HEADER(op, count) ((uint32_t)(op) | (uint32_t)(count) << 8)
#define APEX_CP_MAX_COUNT 0xffffu

/* State register indices. */
#define APEX_STATE_COMPUTE_PROGRAM 0x000   /* low, high */
#define APEX_STATE_COMPUTE_LOCAL 0x002     /* x, y, z */
#define APEX_STATE_COMPUTE_BASE 0x005      /* x, y, z */
#define APEX_STATE_COMPUTE_PRIVATE 0x008   /* low, high */
#define APEX_STATE_COMPUTE_USER 0x010      /* 16 dwords */
#define APEX_STATE_VERTEX 0x020
#define APEX_STATE_VERTEX_USER 0x030
#define APEX_STATE_FRAGMENT 0x040
#define APEX_STATE_FRAGMENT_USER 0x050
#define APEX_STATE_DYNAMIC 0x060
#define APEX_STATE_PASS 0x080              /* 128-dword pass record */
#define APEX_STATE_USER_DWORDS 16

/* BARRIER wait-mask classes and cache actions. */
#define APEX_CP_CLASS_COMPUTE (1u << 0)
#define APEX_CP_CLASS_GRAPHICS (1u << 1)
#define APEX_CP_CLASS_COPY (1u << 2)
#define APEX_CP_CLASS_ALL 7u
#define APEX_CP_CACHE_L1 (1u << 0)
#define APEX_CP_CACHE_INSTRUCTION (1u << 1)
#define APEX_CP_CACHE_TEXTURE (1u << 2)
#define APEX_CP_CACHE_L2_WRITEBACK (1u << 3)
#define APEX_CP_CACHE_TLB (1u << 4)

/* WRITE and TIMESTAMP flag bit 0: after prior work of the queue. */
#define APEX_CP_AFTER_PRIOR_WORK 1u
/* PREDICATE flags: skip draws and dispatches while the u32 at the address
 * is zero, or nonzero when inverted; no flags stop skipping. */
#define APEX_CP_PREDICATE_ENABLE (1u << 0)
#define APEX_CP_PREDICATE_INVERTED (1u << 1)

/* Status page, written by the CP in host memory. */
#define APEX_STATUS_RPTR 0
#define APEX_STATUS_STATE 4
#define APEX_STATUS_KFENCE 8
#define APEX_STATUS_KWAIT 16
#define APEX_STATUS_FAULT 24
#define APEX_STATUS_FAULT_ADDRESS 32
#define APEX_STATUS_FAULT_PACKET 40
#define APEX_STATUS_ENABLED (1u << 0)
#define APEX_STATUS_WAITING (1u << 1)
#define APEX_STATUS_FAULTED (1u << 2)
#define APEX_STATUS_RESET (1u << 3)

/* Host-built dword stream; `failed` latches allocation failure. */
struct apex_ib {
   uint32_t *words;
   uint32_t count, capacity;
   bool failed;
};

void apex_ib_init(struct apex_ib *ib);
void apex_ib_finish(struct apex_ib *ib);
static inline void apex_ib_reset(struct apex_ib *ib) { ib->count = 0; ib->failed = false; }
/* Reserves `count` dwords; NULL (and `failed`) when growth fails. */
uint32_t *apex_ib_reserve(struct apex_ib *ib, uint32_t count);
/* Appends one packet: header plus `count` payload dwords. */
void apex_cp_packet(struct apex_ib *ib, enum apex_cp_op op, uint32_t count, const uint32_t *payload);

void apex_cp_indirect(struct apex_ib *ib, uint64_t va, uint32_t dwords);
void apex_cp_set_state(struct apex_ib *ib, uint32_t index, uint32_t count, const uint32_t *values);
void apex_cp_dispatch(struct apex_ib *ib, uint32_t x, uint32_t y, uint32_t z);
void apex_cp_dispatch_indirect(struct apex_ib *ib, uint64_t va);
void apex_cp_draw(struct apex_ib *ib, uint32_t vertices, uint32_t instances, uint32_t first_vertex,
                  uint32_t first_instance);
void apex_cp_draw_indexed(struct apex_ib *ib, uint32_t indices, uint32_t instances, uint32_t first_index,
                          int32_t vertex_offset, uint32_t first_instance);
void apex_cp_draw_indirect(struct apex_ib *ib, bool indexed, uint64_t va, uint32_t count, uint32_t stride,
                           uint64_t count_va);
void apex_cp_begin_pass(struct apex_ib *ib);
void apex_cp_end_pass(struct apex_ib *ib);
void apex_cp_copy(struct apex_ib *ib, uint64_t src, uint64_t dst, uint64_t bytes);
void apex_cp_fill(struct apex_ib *ib, uint64_t dst, uint64_t bytes, uint32_t value);
struct apex_cp_rect {
   uint64_t src, dst;
   uint32_t src_pitch, dst_pitch, row_bytes, rows, slices, src_slice_pitch, dst_slice_pitch;
};
void apex_cp_copy_rect(struct apex_ib *ib, const struct apex_cp_rect *rect);
void apex_cp_barrier(struct apex_ib *ib, uint32_t wait_mask, uint32_t cache_mask);
void apex_cp_wait(struct apex_ib *ib, uint64_t va, uint64_t value);
void apex_cp_signal(struct apex_ib *ib, uint64_t va, uint64_t value);
void apex_cp_kfence(struct apex_ib *ib, uint64_t value);
/* WRITE carries 1..APEX_CP_MAX_COUNT - 3 dwords. */
void apex_cp_write(struct apex_ib *ib, uint64_t va, uint32_t flags, uint32_t dwords, const uint32_t *data);
void apex_cp_timestamp(struct apex_ib *ib, uint64_t va, uint32_t flags);
void apex_cp_query_begin(struct apex_ib *ib, uint32_t slot);
void apex_cp_query_end(struct apex_ib *ib, uint32_t slot, uint64_t va);
/* `va` is 4-byte aligned; ignored without APEX_CP_PREDICATE_ENABLE. */
void apex_cp_predicate(struct apex_ib *ib, uint64_t va, uint32_t flags);

/* The per-queue user-mode ring. `status` is the read-only status page and
 * `doorbell` the queue's doorbell page (offset 0: wptr on write, live rptr
 * on read). wptr and rptr are free-running dword counts. */
struct apex_ring {
   uint32_t *map;
   uint32_t dwords;
   uint32_t wptr;
   const volatile uint32_t *status;
   volatile uint32_t *doorbell;
};

/* Waits for room for `count` dwords; false when the queue faulted or the
 * CP did not consume within `timeout_ns`. */
bool apex_ring_reserve(struct apex_ring *ring, uint32_t count, uint64_t timeout_ns);
/* Copies words at wptr, wrapping at the ring end. */
void apex_ring_write(struct apex_ring *ring, const uint32_t *words, uint32_t count);
/* Orders the ring stores before the uncached doorbell store of wptr. */
void apex_ring_publish(struct apex_ring *ring);

/* One vkQueueSubmit batch as ring packets: acquire barrier, the kernel-wait
 * WAIT, one INDIRECT per command buffer, the retirement SIGNAL, then KFENCE
 * when a kernel fence needs it. */
struct apex_cp_batch {
   uint32_t acquire_cache;         /* BARRIER cache mask before the batch; 0 omits it */
   uint64_t kwait_va, kwait_value; /* WAIT(status + 16, value) when value is nonzero */
   uint32_t ib_count;
   const uint64_t *ib_va;
   const uint32_t *ib_dwords;
   uint64_t retire_va, sequence;   /* SIGNAL(retire_va, sequence) */
   bool kfence;                    /* KFENCE(sequence) */
};
void apex_cp_batch(struct apex_ib *ib, const struct apex_cp_batch *batch);

static inline uint64_t
apex_status_u64(const volatile uint32_t *status, unsigned offset)
{
   return status[offset / 4] | (uint64_t)status[offset / 4 + 1] << 32;
}
#endif
