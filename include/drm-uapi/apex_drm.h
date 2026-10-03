/* SPDX-License-Identifier: GPL-2.0-only WITH Linux-syscall-note */
/* Apex render and queue interface (Docs/architecture.md, Linux uapi). All
 * structs are naturally aligned; padding and reserved fields are zero. */
#ifndef APEX_DRM_H
#define APEX_DRM_H
#ifdef __KERNEL__
#include <drm/drm.h>
#else
#include <libdrm/drm.h>
#endif

#define DRM_APEX_INFO 0x00
#define DRM_APEX_GEM_CREATE 0x01
#define DRM_APEX_GEM_MMAP 0x02
#define DRM_APEX_VM_BIND 0x03
#define DRM_APEX_QUEUE_CREATE 0x04
#define DRM_APEX_QUEUE_DESTROY 0x05
#define DRM_APEX_QUEUE_FENCE 0x06
#define DRM_APEX_QUEUE_WAIT 0x07
#define DRM_APEX_QUEUE_STATUS 0x08

struct drm_apex_info {
	__u64 local_bytes;   /* allocatable LOCAL */
	__u64 visible_bytes; /* BAR2 extent; equals local_bytes on U50 */
	__u64 timestamp_hz;  /* device timestamp clock, BAR0 TIMEBASE */
	__u32 queue_slots, vm_slots; /* per device; zero without a command processor */
	__u32 queues_per_file;       /* 8 */
	__u32 cores, texture_units, tile_planes;
	__u64 reserved[4];
};

#define APEX_GEM_SYSTEM (1u << 0) /* cached host shmem, GPU access over PCIe */
/* Without SYSTEM: contiguous zero-filled LOCAL, the class of dumb buffers and
 * the only one KMS scans out. */
struct drm_apex_gem_create {
	__u64 size; /* multiple of 4 KiB */
	__u32 flags;
	__u32 handle; /* out */
};

struct drm_apex_gem_mmap {
	__u32 handle, reserved;
	__u64 offset; /* out: LOCAL maps BAR2 write-combined, SYSTEM maps shmem cached */
};

#define APEX_VM_MAP 1
#define APEX_VM_UNMAP 2
#define APEX_VM_MAP_NULL 3 /* sparse NULL leaves: reads zero, writes dropped */
#define APEX_VM_READ (1u << 0)
#define APEX_VM_WRITE (1u << 1)
#define APEX_VM_EXEC (1u << 2)
struct drm_apex_vm_bind_op {
	__u32 op, flags;
	__u32 handle, reserved;
	__u64 va, offset, bytes; /* 4 KiB aligned; va in [2 MiB, 2^40) */
};
struct drm_apex_vm_bind { /* synchronous */
	__u64 ops; /* user pointer to drm_apex_vm_bind_op[] */
	__u32 op_count, reserved;
};

struct drm_apex_queue_create {
	__u64 ring_va;    /* 4 KiB aligned, mapped readable in the file's VM */
	__u32 ring_bytes; /* power of two, 4 KiB..1 MiB */
	__u32 priority;   /* 0 normal, 1 high (DRM master or CAP_SYS_NICE) */
	__u32 queue_id;   /* out */
	__u32 reserved;
	__u64 status_va;       /* out: read-only GPUVA of the status page */
	__u64 status_offset;   /* out: mmap offset of the status page */
	__u64 doorbell_offset; /* out: mmap offset of the doorbell page */
};

struct drm_apex_queue_destroy {
	__u32 queue_id, reserved;
};

/* Attach a fence that signals when the queue's kfence >= seqno. Mesa has
 * already appended the KFENCE packet. point 0 replaces a binary syncobj's
 * fence; a nonzero point adds a timeline point. */
struct drm_apex_queue_fence {
	__u32 queue_id, syncobj;
	__u64 seqno, point;
};

/* When the syncobj point signals, the kernel writes value to the queue's
 * kwait and rechecks its waits. Mesa appends WAIT(status_va + 16, value). */
struct drm_apex_queue_wait {
	__u32 queue_id, syncobj;
	__u64 point, value;
};

struct drm_apex_queue_status {
	__u32 queue_id;
	__u32 state; /* out: status-page state */
	__u32 fault_status, fault_unit;
	__u64 fault_address;
	__s32 error; /* out: first terminal errno of the file, sticky */
	__u32 reserved;
};

#define DRM_IOCTL_APEX_INFO \
	DRM_IOR(DRM_COMMAND_BASE + DRM_APEX_INFO, struct drm_apex_info)
#define DRM_IOCTL_APEX_GEM_CREATE \
	DRM_IOWR(DRM_COMMAND_BASE + DRM_APEX_GEM_CREATE, struct drm_apex_gem_create)
#define DRM_IOCTL_APEX_GEM_MMAP \
	DRM_IOWR(DRM_COMMAND_BASE + DRM_APEX_GEM_MMAP, struct drm_apex_gem_mmap)
#define DRM_IOCTL_APEX_VM_BIND \
	DRM_IOW(DRM_COMMAND_BASE + DRM_APEX_VM_BIND, struct drm_apex_vm_bind)
#define DRM_IOCTL_APEX_QUEUE_CREATE \
	DRM_IOWR(DRM_COMMAND_BASE + DRM_APEX_QUEUE_CREATE, struct drm_apex_queue_create)
#define DRM_IOCTL_APEX_QUEUE_DESTROY \
	DRM_IOW(DRM_COMMAND_BASE + DRM_APEX_QUEUE_DESTROY, struct drm_apex_queue_destroy)
#define DRM_IOCTL_APEX_QUEUE_FENCE \
	DRM_IOW(DRM_COMMAND_BASE + DRM_APEX_QUEUE_FENCE, struct drm_apex_queue_fence)
#define DRM_IOCTL_APEX_QUEUE_WAIT \
	DRM_IOW(DRM_COMMAND_BASE + DRM_APEX_QUEUE_WAIT, struct drm_apex_queue_wait)
#define DRM_IOCTL_APEX_QUEUE_STATUS \
	DRM_IOWR(DRM_COMMAND_BASE + DRM_APEX_QUEUE_STATUS, struct drm_apex_queue_status)

#endif
