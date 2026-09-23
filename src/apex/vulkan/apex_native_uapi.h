/* SPDX-License-Identifier: GPL-2.0-only WITH Linux-syscall-note */
/* Native qualification ABI v1 from apex-compute/gpu Driver/apex_ioctl.h.
 * This is the privileged character-device interface, not a DRM UAPI. */
#ifndef APEX_NATIVE_UAPI_H
#define APEX_NATIVE_UAPI_H
#include <linux/ioctl.h>
#include <linux/types.h>
#define APEX_NATIVE_CREATE 0
#define APEX_NATIVE_ALLOC 1
#define APEX_NATIVE_UPLOAD 2
#define APEX_NATIVE_DOWNLOAD 3
#define APEX_NATIVE_START 4
#define APEX_NATIVE_SUBMIT 5
#define APEX_NATIVE_POLL 6
#define APEX_NATIVE_STOP 8
#define APEX_NATIVE_CLOSE 10
#define APEX_NATIVE_DATA 0
#define APEX_NATIVE_PROGRAM 1
#define APEX_NATIVE_COMPUTE 1
struct apex_ioctl_native {
   __u32 operation, kind, queue, handle;
   __u64 user_ptr, bytes, offset;
   __u32 data_handle, dependency, signal, workgroups;
   __u64 dependency_value, signal_value, identity;
   __u32 status, reason;
   __u64 timestamp;
};
#define APEX_IOCTL_NATIVE _IOWR('A', 0x10, struct apex_ioctl_native)
#endif
