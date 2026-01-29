#pragma once

#include "virtgpu-utils.h"
#include "virtgpu-shm.h"
#include "virtgpu-apir.h"

#include "backend/shared/api_remoting.h"
#include "backend/shared/apir_cs.h"

#include <fcntl.h>
#include <stdbool.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <threads.h>
#include <xf86drm.h>

#include <cstring>

#define VIRGL_RENDERER_UNSTABLE_APIS 1
#include "apir_hw.h"
#include <drm/virtgpu_drm.h>
#include "venus_hw.h"

#ifndef VIRTGPU_DRM_CAPSET_APIR
// Will be defined include/drm/virtgpu_drm.h when
// https://gitlab.freedesktop.org/virgl/virglrenderer/-/merge_requests/1590/diffs
// is merged
#define VIRTGPU_DRM_CAPSET_APIR 10
#endif

// Mesa/Virlgrenderer Venus internal. Only necessary during the
// Venus->APIR transition in Virglrenderer
#define VENUS_COMMAND_TYPE_LENGTH 331

#ifndef VIRTGPU_DRM_CAPSET_VENUS // only available with Linux >= v6.16
#define VIRTGPU_DRM_CAPSET_VENUS 4
#endif

typedef uint32_t virgl_renderer_capset;

/* from src/virtio/vulkan/vn_renderer_virtgpu.c */
#define VIRTGPU_PCI_VENDOR_ID       0x1af4
#define VIRTGPU_PCI_DEVICE_ID       0x1050
#define VIRTGPU_BLOB_MEM_GUEST_VRAM 0x0004
#define VIRTGPU_PARAM_GUEST_VRAM    9

#define SHMEM_DATA_SIZE  0x1830000  // 24MiB
#define SHMEM_REPLY_SIZE 0x4000

#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))

enum virt_gpu_result_t {
    APIR_SUCCESS                     = 0,
    APIR_ERROR_INITIALIZATION_FAILED = -1,
};

#define PRINTFLIKE(f, a) __attribute__((format(__printf__, f, a)))

struct virtgpu {
    bool use_apir_capset;

    int fd;

    struct {
        virgl_renderer_capset      id;
        uint32_t                   version;
        virgl_renderer_capset_apir data;
    } capset;

    util_sparse_array shmem_array;

    /* APIR communication pages */
    virtgpu_shmem reply_shmem;
    virtgpu_shmem data_shmem;
};

static inline int virtgpu_ioctl(virtgpu * gpu, unsigned long request, void * args) {
    return drmIoctl(gpu->fd, request, args);
}

virtgpu * create_virtgpu();

apir_encoder * remote_call_prepare(virtgpu * gpu, ApirCommandType apir_cmd_type, int32_t cmd_flags);

uint32_t remote_call(virtgpu *       gpu,
                     apir_encoder *  enc,
                     apir_decoder ** dec,
                     float           max_wait_ms,
                     long long *     call_duration_ns);

void remote_call_finish(virtgpu * gpu, apir_encoder * enc, apir_decoder * dec);
