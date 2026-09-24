/* SPDX-License-Identifier: GPL-2.0-only WITH Linux-syscall-note */
/* From apex-compute/gpu Driver/apex_drm.h; internal experimental ABI. */
#ifndef APEX_DRM_H
#define APEX_DRM_H
#include "drm.h"

#define APEX_DRM_CAP_SHMEM (1U << 0)
#define APEX_DRM_CAP_EXEC (1U << 1)
#define APEX_DRM_CAP_GPUVM (1U << 2)
#define APEX_DRM_CAP_ASYNC (1U << 3)

/* Output only. Capabilities describe this DRM interface, not the raw device. */
struct drm_apex_info {
	__u32 version;
	__u32 capabilities;
	__u64 max_buffer_bytes;
};

/* Host-coherent shmem. size returns the page-rounded extent. flags and handle
 * must be zero on input. Handles belong to the calling DRM file. */
struct drm_apex_gem_create {
	__u64 size;
	__u32 flags;
	__u32 handle;
};

/* Returns a GEM mmap offset, never a BAR or GPU address. flags must be zero. */
struct drm_apex_gem_mmap {
	__u32 handle;
	__u32 flags;
	__u64 offset;
};

/* Synchronous APX dispatch over file-local shmem ranges. The kernel snapshots
 * inputs, owns the GPUVA layout and waits for release before copying output.
 * No raw GPU addresses or imported dma-bufs are accepted. Workgroups: 1..1024.
 * APX is at most 1 MiB; all local allocations must fit the 64 MiB native arena.
 * Callers synchronize CPU/PRIME access to both ranges through return; this
 * snapshot interface neither imports external fences nor publishes dma_fences.
 * flags and output fields must be zero on entry. status: success=1, cancelled=2,
 * invalid=3, memory fault=4, timeout=5. Only success changes the data range.
 * Signals and the five-second host deadline cancel/drain and return an errno.
 * A failed release returns EIO, suppresses output and retains unsafe backing. */
struct drm_apex_exec {
	__u32 program_handle;
	__u32 data_handle;
	__u32 workgroups;
	__u32 flags;
	__u64 program_offset;
	__u64 program_bytes;
	__u64 data_offset;
	__u64 data_bytes;
	__u32 status;
	__u32 reason;
	__u64 timestamp;
};

#define APEX_DRM_VM_BIND_MAP 1U
#define APEX_DRM_VM_BIND_UNMAP 2U
#define APEX_DRM_VM_READ (1U << 0)
#define APEX_DRM_VM_WRITE (1U << 1)
#define APEX_DRM_VM_EXEC (1U << 2)

/* Update the calling DRM file's implicit VM. MAP installs one fixed,
 * page-aligned GEM range; overlapping mappings are rejected. EXEC and WRITE
 * permissions are mutually exclusive. UNMAP removes one exact mapping and
 * requires flags, handle, pad and offset to be zero. Bind changes are
 * synchronous: no queue can retain the old page tables on return. */
struct drm_apex_vm_bind {
	__u32 operation;
	__u32 flags;
	__u32 handle;
	__u32 pad;
	__u64 va;
	__u64 offset;
	__u64 bytes;
};

#define APEX_DRM_TRANSFER_TO_LOCAL 1U
#define APEX_DRM_TRANSFER_FROM_LOCAL 2U

/* Explicitly copy one byte range between a GEM object's shmem CPU view and
 * its GEM-owned LOCAL backing. TO_LOCAL is a CPU flush/upload; FROM_LOCAL is
 * a CPU invalidate/download. Bytes outside the range are unchanged. */
struct drm_apex_gem_transfer {
	__u32 handle;
	__u32 direction;
	__u32 flags;
	__u32 pad;
	__u64 offset;
	__u64 bytes;
};

/* Synchronous dispatch through the calling DRM file's persistent VM. The
 * exact program extent must be covered by an RX mapping and pass APX
 * admission; data_va must select an RW mapping. This operation performs no
 * implicit shmem/LOCAL transfer. flags and output fields must be zero on
 * entry. Status values match drm_apex_exec. */
struct drm_apex_vm_exec {
	__u64 program_va;
	__u64 program_bytes;
	__u64 data_va;
	__u32 workgroups;
	__u32 flags;
	__u32 status;
	__u32 reason;
	__u64 timestamp;
};

#define APEX_DRM_SUBMIT_SYNC_ONLY (1U << 0)
#define APEX_DRM_MAX_SYNCS 16U

/* A handle belongs to the submitting DRM file. point=0 selects a binary
 * syncobj; a nonzero point selects a timeline syncobj. flags must be zero.
 * Input fences must already be published (including timeline points). */
struct drm_apex_sync {
	__u32 handle;
	__u32 flags;
	__u64 point;
};

/* Enqueue on the calling file's ordered VM entity. No hardware execution or
 * dependency wait occurs in the ioctl. The output fences report terminal
 * errors and are published before return, including on program/data BO
 * reservations. count is 0..16 for inputs, 1..16 for outputs. A sync-only
 * submission has flags=SYNC_ONLY and zero program/data/workgroup fields;
 * otherwise flags=0 and the execution fields match VM_EXEC. Reserved fields
 * must be zero. An output handle may appear only once in the output array.
 * Published input fences can wait arbitrarily long while the file is open;
 * close/unplug cancels waiting jobs with an error fence. */
struct drm_apex_vm_submit {
	__u64 program_va;
	__u64 program_bytes;
	__u64 data_va;
	__u64 inputs;  /* userspace array of drm_apex_sync */
	__u64 outputs; /* userspace array of drm_apex_sync */
	__u32 workgroups;
	__u32 flags;
	__u32 input_count;
	__u32 output_count;
	__u64 reserved;
};

/* Sticky first terminal failure for the calling file's asynchronous queue.
 * A syncobj wait reports signal, not fence->error; inspect this after a wait
 * before reporting GPU success. Zero means no failed submission observed.
 * This query does not clear the error. All fields must be zero on input. */
struct drm_apex_vm_status {
	__s32 error;
	__u32 reserved;
};

#define DRM_APEX_INFO 0x00
#define DRM_APEX_GEM_CREATE 0x01
#define DRM_APEX_GEM_MMAP 0x02
#define DRM_APEX_EXEC 0x03
#define DRM_APEX_VM_BIND 0x04
#define DRM_APEX_GEM_TRANSFER 0x05
#define DRM_APEX_VM_EXEC 0x06
#define DRM_APEX_VM_SUBMIT 0x07
#define DRM_APEX_VM_STATUS 0x08
#define DRM_IOCTL_APEX_INFO DRM_IOR(DRM_COMMAND_BASE + DRM_APEX_INFO, struct drm_apex_info)
#define DRM_IOCTL_APEX_GEM_CREATE DRM_IOWR(DRM_COMMAND_BASE + DRM_APEX_GEM_CREATE, struct drm_apex_gem_create)
#define DRM_IOCTL_APEX_GEM_MMAP DRM_IOWR(DRM_COMMAND_BASE + DRM_APEX_GEM_MMAP, struct drm_apex_gem_mmap)
#define DRM_IOCTL_APEX_EXEC DRM_IOWR(DRM_COMMAND_BASE + DRM_APEX_EXEC, struct drm_apex_exec)
#define DRM_IOCTL_APEX_VM_BIND DRM_IOW(DRM_COMMAND_BASE + DRM_APEX_VM_BIND, struct drm_apex_vm_bind)
#define DRM_IOCTL_APEX_GEM_TRANSFER DRM_IOW(DRM_COMMAND_BASE + DRM_APEX_GEM_TRANSFER, struct drm_apex_gem_transfer)
#define DRM_IOCTL_APEX_VM_EXEC DRM_IOWR(DRM_COMMAND_BASE + DRM_APEX_VM_EXEC, struct drm_apex_vm_exec)
#define DRM_IOCTL_APEX_VM_SUBMIT DRM_IOW(DRM_COMMAND_BASE + DRM_APEX_VM_SUBMIT, struct drm_apex_vm_submit)
#define DRM_IOCTL_APEX_VM_STATUS DRM_IOWR(DRM_COMMAND_BASE + DRM_APEX_VM_STATUS, struct drm_apex_vm_status)
#endif
