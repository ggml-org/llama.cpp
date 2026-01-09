#pragma once

#include <xf86drm.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdbool.h>
#include <threads.h>
#include <cstring>
#include <sys/stat.h>
#include <sys/sysmacros.h>

#include "virtgpu-apir.h"
#include "virtgpu-utils.h"
#include "../ggml-remotingbackend/shared/api_remoting.h"
#include "../ggml-remotingbackend/shared/apir_cs.h"

#include "virtgpu-shm.h"

#define VIRGL_RENDERER_UNSTABLE_APIS 1
#include "drm-uapi/virtgpu_drm.h"
#include "apir_hw.h"
#include "venus_hw.h"

// must match https://gitlab.freedesktop.org/kpouget/virglrenderer/-/blob/main/src/virglrenderer_hw.h?ref_type=heads
enum virgl_renderer_capset {
    VIRGL_RENDERER_CAPSET_VIRGL                   = 1,
    VIRGL_RENDERER_CAPSET_VIRGL2                  = 2,
    /* 3 is reserved for gfxstream */
    VIRGL_RENDERER_CAPSET_VENUS                   = 4,
    /* 5 is reserved for cross-domain */
    VIRGL_RENDERER_CAPSET_DRM                     = 6,

    VIRGL_RENDERER_CAPSET_APIR                    = 10,
};

#define VENUS_COMMAND_TYPE_LENGTH 331

/* from src/virtio/vulkan/vn_renderer_virtgpu.c */
#define VIRTGPU_PCI_VENDOR_ID 0x1af4
#define VIRTGPU_PCI_DEVICE_ID 0x1050
#define VIRTGPU_BLOB_MEM_GUEST_VRAM 0x0004
#define VIRTGPU_PARAM_GUEST_VRAM 9

#define SHMEM_DATA_SIZE 0x1830000 // 24MiB
#define SHMEM_REPLY_SIZE 0x4000

#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))

typedef enum virt_gpu_result_t {
    APIR_SUCCESS = 0,
    APIR_ERROR_INITIALIZATION_FAILED = -1,
} virt_gpu_result_t;

#define PRINTFLIKE(f, a) __attribute__ ((format(__printf__, f, a)))

struct virtgpu {
    struct remoting_dev_instance *instance;

    bool use_apir_capset;

    int fd;

    struct {
        enum virgl_renderer_capset id;
        uint32_t version;
	struct virgl_renderer_capset_apir data;
    } capset;

    struct util_sparse_array shmem_array;

    /* APIR communication pages */
    struct virtgpu_shmem reply_shmem;
    struct virtgpu_shmem data_shmem;
};


static inline int
virtgpu_ioctl(struct virtgpu *gpu, unsigned long request, void *args)
{
    return drmIoctl(gpu->fd, request, args);
}

struct virtgpu *create_virtgpu();

struct apir_encoder *remote_call_prepare(
    struct virtgpu *gpu,
    ApirCommandType apir_cmd_type,
    int32_t cmd_flags);

uint32_t remote_call(
    struct virtgpu *gpu,
    struct apir_encoder *enc,
    struct apir_decoder **dec,
    float max_wait_ms,
    long long *call_duration_ns
    );

void remote_call_finish(
    struct virtgpu *gpu,
    struct apir_encoder *enc,
    struct apir_decoder *dec);
