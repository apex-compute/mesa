/* SPDX-License-Identifier: GPL-2.0-only WITH Linux-syscall-note */
/* From apex-compute/gpu Driver/apex_drm.h; internal experimental ABI. */
#ifndef APEX_DRM_H
#define APEX_DRM_H
#include "drm.h"

#define APEX_DRM_CAP_SHMEM (1U << 0)
#define APEX_DRM_CAP_EXEC (1U << 1)

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

#define DRM_APEX_INFO 0x00
#define DRM_APEX_GEM_CREATE 0x01
#define DRM_APEX_GEM_MMAP 0x02
#define DRM_APEX_EXEC 0x03
#define DRM_IOCTL_APEX_INFO DRM_IOR(DRM_COMMAND_BASE + DRM_APEX_INFO, struct drm_apex_info)
#define DRM_IOCTL_APEX_GEM_CREATE DRM_IOWR(DRM_COMMAND_BASE + DRM_APEX_GEM_CREATE, struct drm_apex_gem_create)
#define DRM_IOCTL_APEX_GEM_MMAP DRM_IOWR(DRM_COMMAND_BASE + DRM_APEX_GEM_MMAP, struct drm_apex_gem_mmap)
#define DRM_IOCTL_APEX_EXEC DRM_IOWR(DRM_COMMAND_BASE + DRM_APEX_EXEC, struct drm_apex_exec)
#endif
