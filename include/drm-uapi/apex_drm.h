/* SPDX-License-Identifier: GPL-2.0-only WITH Linux-syscall-note */
/* From apex-compute/gpu Driver/apex_drm.h; internal experimental ABI. */
#ifndef APEX_DRM_H
#define APEX_DRM_H
#include "drm.h"

#define APEX_DRM_CAP_SHMEM (1U << 0)
#define APEX_DRM_CAP_EXEC (1U << 1)
#define APEX_DRM_CAP_GPUVM (1U << 2)
#define APEX_DRM_CAP_ASYNC (1U << 3)
#define APEX_DRM_CAP_PRIME_COHERENT (1U << 4)
#define APEX_DRM_CAP_MULTIWAVE (1U << 5)
#define APEX_DRM_CAP_HOST_COHERENT (1U << 6)

/* Output only. PRIME_COHERENT distinguishes GPUVM implementations that
 * refresh shared dma-buf backing before execution and copy writable results
 * back before completion. MULTIWAVE admits APX2 groups of up to 256 invocations
 * on the bound image. HOST_COHERENT requires the image's SYSTEM atomic feature
 * and a coherent DMA mapping; every allocation is checked again. Capabilities
 * describe this interface, not raw PCI. */
struct drm_apex_info {
	__u32 version;
	__u32 capabilities;
	__u64 max_buffer_bytes;
};

#define APEX_DRM_GEM_HOST_COHERENT (1U << 0)

/* Opt-in HOST_COHERENT uses pinned, bidirectionally DMA-mapped shmem as the
 * GPU's SYSTEM backing; CPU and GPU share its bytes without GEM_TRANSFER.
 * size returns the page-rounded extent; handle must be zero on input.
 * Handles belong to the calling DRM file. Other flags are invalid. */
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

/* Private Apex GEM: explicitly copy one byte range between shmem and LOCAL;
 * TO_LOCAL uploads and FROM_LOCAL downloads without changing other bytes.
 * Once exported, shared shmem is canonical and either direction only waits
 * and rechecks reservation fences (no copy). HOST_COHERENT also only waits:
 * its shmem is the GPU's SYSTEM backing. Foreign imports return EOPNOTSUPP;
 * use their exporter for CPU access and synchronization. */
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
 * admission; data_va must select an RW mapping. Private BOs retain explicit
 * transfer semantics. Shared BOs refresh whole-object canonical dma-buf
 * backing into LOCAL before execution and copy writable results back after
 * drain, before returning. HOST_COHERENT uses pinned SYSTEM backing directly,
 * with no LOCAL shadow or transfer. Coherent PRIME export is unsupported;
 * foreign imported PRIME retains the noncoherent LOCAL shadow semantics.
 * flags and output fields must be zero on entry.
 * Status values match drm_apex_exec. */
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
 * reservations. Execution refreshes shared canonical backing and copies
 * writable results back before signaling completion; private BOs keep their
 * explicit transfer semantics. count is 0..16 for inputs, 1..16 for outputs.
 * A sync-only submission copies no BO storage; it has flags=SYNC_ONLY and
 * zero program/data/workgroup fields;
 * otherwise flags=0 and the execution fields match VM_EXEC. Reserved fields
 * must be zero. One underlying output syncobj may appear only once, even
 * through two distinct handles imported from the same opaque syncobj fd.
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
