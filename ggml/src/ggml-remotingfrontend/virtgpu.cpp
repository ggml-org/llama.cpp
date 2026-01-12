#include "virtgpu.h"

#include <stdio.h>
#include <unistd.h>

#include <cassert>
#include <cerrno>
#include <cstdlib>

static virt_gpu_result_t virtgpu_open_device(struct virtgpu * gpu, const drmDevicePtr dev);
static virt_gpu_result_t virtgpu_open(struct virtgpu * gpu);

static virt_gpu_result_t virtgpu_init_capset(struct virtgpu * gpu);
static virt_gpu_result_t virtgpu_init_context(struct virtgpu * gpu);

static int      virtgpu_ioctl_context_init(struct virtgpu * gpu, enum virgl_renderer_capset capset_id);
static int      virtgpu_ioctl_get_caps(struct virtgpu *           gpu,
                                       enum virgl_renderer_capset id,
                                       uint32_t                   version,
                                       void *                     capset,
                                       size_t                     capset_size);
static uint64_t virtgpu_ioctl_getparam(struct virtgpu * gpu, uint64_t param);
static void     virtgpu_init_renderer_info(struct virtgpu * gpu);

struct timer_data wait_host_reply_timer = { 0, 0, 0, "wait_host_reply" };

static void log_call_duration(long long call_duration_ns, const char * name);

const uint64_t APIR_HANDSHAKE_MAX_WAIT_MS   = 2 * 1000;   // 2s
const uint64_t APIR_LOADLIBRARY_MAX_WAIT_MS = 60 * 1000;  // 60s

static int virtgpu_handshake(struct virtgpu * gpu) {
    struct apir_encoder * encoder;
    struct apir_decoder * decoder;

    encoder = remote_call_prepare(gpu, APIR_COMMAND_TYPE_HandShake, 0);
    if (!encoder) {
        FATAL("%s: failed to prepare the remote call encoder :/", __func__);
        return 1;
    }

    /* write handshake props */

    uint32_t guest_major = APIR_PROTOCOL_MAJOR;
    uint32_t guest_minor = APIR_PROTOCOL_MINOR;
    apir_encode_uint32_t(encoder, &guest_major);
    apir_encode_uint32_t(encoder, &guest_minor);

    /* *** */

    uint32_t  ret_magic;
    long long call_duration_ns;
    ret_magic = remote_call(gpu, encoder, &decoder, APIR_HANDSHAKE_MAX_WAIT_MS, &call_duration_ns);
    log_call_duration(call_duration_ns, "API Remoting handshake");

    if (!decoder) {
        FATAL(
            "%s: failed to initiate the communication with the virglrenderer library. "
            "Most likely, the wrong virglrenderer library was loaded in the hypervisor.",
            __func__);
        return 1;
    }

    /* read handshake return values */

    uint32_t host_major;
    uint32_t host_minor;

    if (ret_magic != APIR_HANDSHAKE_MAGIC) {
        FATAL("%s: handshake with the virglrenderer failed (code=%d | %s):/", __func__, ret_magic,
              apir_backend_initialize_error(ret_magic));
    } else {
        apir_decode_uint32_t(decoder, &host_major);
        apir_decode_uint32_t(decoder, &host_minor);
    }

    remote_call_finish(gpu, encoder, decoder);

    if (ret_magic != APIR_HANDSHAKE_MAGIC) {
        return 1;
    }

    INFO("%s: Guest is running with %u.%u", __func__, guest_major, guest_minor);
    INFO("%s: Host is running with %u.%u", __func__, host_major, host_minor);

    if (guest_major != host_major) {
        ERROR("Host major (%d) and guest major (%d) version differ", host_major, guest_major);
    } else if (guest_minor != host_minor) {
        WARNING("Host minor (%d) and guest minor (%d) version differ", host_minor, guest_minor);
    }

    return 0;
}

static ApirLoadLibraryReturnCode virtgpu_load_library(struct virtgpu * gpu) {
    struct apir_encoder *     encoder;
    struct apir_decoder *     decoder;
    ApirLoadLibraryReturnCode ret;

    encoder = remote_call_prepare(gpu, APIR_COMMAND_TYPE_LoadLibrary, 0);
    if (!encoder) {
        FATAL("%s: hypercall error: failed to prepare the remote call encoder :/", __func__);
        return APIR_LOAD_LIBRARY_HYPERCALL_INITIALIZATION_ERROR;
    }

    long long call_duration_ns;

    ret = (ApirLoadLibraryReturnCode) remote_call(gpu, encoder, &decoder, APIR_LOADLIBRARY_MAX_WAIT_MS,
                                                  &call_duration_ns);
    log_call_duration(call_duration_ns, "API Remoting LoadLibrary");

    if (!decoder) {
        FATAL("%s: hypercall error: failed to kick the API remoting hypercall. :/", __func__);
        return APIR_LOAD_LIBRARY_HYPERCALL_INITIALIZATION_ERROR;
    }

    remote_call_finish(gpu, encoder, decoder);

    if (ret == APIR_LOAD_LIBRARY_SUCCESS) {
        INFO("%s: The API Remoting backend was successfully loaded and initialized", __func__);

        return ret;
    }

    // something wrong happened, find out what.

    if (ret < APIR_LOAD_LIBRARY_INIT_BASE_INDEX) {
        FATAL("%s: virglrenderer could not load the API Remoting backend library: %s (code %d)", __func__,
              apir_load_library_error(ret), ret);
        return ret;
    }

    INFO("%s: virglrenderer successfully loaded the API Remoting backend library", __func__);

    ApirLoadLibraryReturnCode apir_ret = (ApirLoadLibraryReturnCode) (ret - APIR_LOAD_LIBRARY_INIT_BASE_INDEX);

    if (apir_ret < APIR_LOAD_LIBRARY_INIT_BASE_INDEX) {
        FATAL("%s: the API Remoting backend library couldn't load the backend library: apir code=%d | %s):/", __func__,
              apir_ret, apir_load_library_error(apir_ret));
    } else {
        uint32_t lib_ret = apir_ret - APIR_LOAD_LIBRARY_INIT_BASE_INDEX;
        FATAL("%s: the API Remoting backend library initialize its backend library: apir code=%d):/", __func__,
              lib_ret);
    }
    return ret;
}

struct virtgpu * create_virtgpu() {
    struct virtgpu * gpu = new struct virtgpu();

    gpu->use_apir_capset = getenv("GGML_REMOTING_USE_APIR_CAPSET") != nullptr;

    util_sparse_array_init(&gpu->shmem_array, sizeof(struct virtgpu_shmem), 1024);

    if (virtgpu_open(gpu) != APIR_SUCCESS) {
        FATAL("%s: failed to open the virtgpu device :/", __func__);
        return NULL;
    }

    if (virtgpu_init_capset(gpu) != APIR_SUCCESS) {
        FATAL("%s: failed to initialize the GPU capset :/", __func__);
        return NULL;
    }

    if (virtgpu_init_context(gpu) != APIR_SUCCESS) {
        FATAL("%s: failed to initialize the GPU context :/", __func__);
        return NULL;
    }

    if (virtgpu_shmem_create(gpu, SHMEM_REPLY_SIZE, &gpu->reply_shmem)) {
        FATAL("%s: failed to create the shared reply memory pages :/", __func__);
        return NULL;
    }

    if (virtgpu_shmem_create(gpu, SHMEM_DATA_SIZE, &gpu->data_shmem)) {
        FATAL("%s: failed to create the shared data memory pages :/", __func__);
        return NULL;
    }

    if (virtgpu_handshake(gpu)) {
        FATAL("%s: failed to handshake with the virglrenderer library :/", __func__);
        return NULL;
    }

    if (virtgpu_load_library(gpu) != APIR_LOAD_LIBRARY_SUCCESS) {
        FATAL("%s: failed to load the backend library :/", __func__);
        return NULL;
    }

    return gpu;
}

static virt_gpu_result_t virtgpu_open(struct virtgpu * gpu) {
    drmDevicePtr devs[8];
    int          count = drmGetDevices2(0, devs, ARRAY_SIZE(devs));
    if (count < 0) {
        ERROR("%s: failed to enumerate DRM devices", __func__);
        return APIR_ERROR_INITIALIZATION_FAILED;
    }

    virt_gpu_result_t result = APIR_ERROR_INITIALIZATION_FAILED;
    for (int i = 0; i < count; i++) {
        result = virtgpu_open_device(gpu, devs[i]);
        if (result == APIR_SUCCESS) {
            break;
        }
    }

    drmFreeDevices(devs, count);

    return result;
}

static virt_gpu_result_t virtgpu_open_device(struct virtgpu * gpu, const drmDevicePtr dev) {
    const char * node_path = dev->nodes[DRM_NODE_RENDER];

    int fd = open(node_path, O_RDWR | O_CLOEXEC);
    if (fd < 0) {
        MESSAGE("failed to open %s", node_path);
        return APIR_ERROR_INITIALIZATION_FAILED;
    }

    drmVersionPtr version = drmGetVersion(fd);
    if (!version || strcmp(version->name, "virtio_gpu") || version->version_major != 0) {
        if (version) {
            MESSAGE("unknown DRM driver %s version %d", version->name, version->version_major);
        } else {
            MESSAGE("failed to get DRM driver version");
        }

        if (version) {
            drmFreeVersion(version);
        }
        close(fd);
        return APIR_ERROR_INITIALIZATION_FAILED;
    }

    gpu->fd = fd;

    drmFreeVersion(version);

    MESSAGE("using DRM device %s", node_path);

    return APIR_SUCCESS;
}

static virt_gpu_result_t virtgpu_init_context(struct virtgpu * gpu) {
    assert(!gpu->capset.version);
    const int ret = virtgpu_ioctl_context_init(gpu, gpu->capset.id);
    if (ret) {
        MESSAGE("failed to initialize context: %s", strerror(errno));
        return APIR_ERROR_INITIALIZATION_FAILED;
    }

    return APIR_SUCCESS;
}

static virt_gpu_result_t virtgpu_init_capset(struct virtgpu * gpu) {
    if (gpu->use_apir_capset) {
        MESSAGE("Using the APIR capset");
        gpu->capset.id = VIRGL_RENDERER_CAPSET_APIR;
    } else {
        MESSAGE("Using the Venus capset");
        gpu->capset.id = VIRGL_RENDERER_CAPSET_VENUS;
    }
    gpu->capset.version = 0;

    int ret =
        virtgpu_ioctl_get_caps(gpu, gpu->capset.id, gpu->capset.version, &gpu->capset.data, sizeof(gpu->capset.data));

    if (ret) {
        MESSAGE("failed to get APIR v%d capset: %s", gpu->capset.version, strerror(errno));
        return APIR_ERROR_INITIALIZATION_FAILED;
    }

    assert(gpu->capset.data.supports_blob_resources);

    return APIR_SUCCESS;
}

static int virtgpu_ioctl_context_init(struct virtgpu * gpu, enum virgl_renderer_capset capset_id) {
    struct drm_virtgpu_context_set_param ctx_set_params[3] = {
        {
         .param = VIRTGPU_CONTEXT_PARAM_CAPSET_ID,
         .value = capset_id,
         },
        {
         .param = VIRTGPU_CONTEXT_PARAM_NUM_RINGS,
         .value = 1,
         },
        {
         .param = VIRTGPU_CONTEXT_PARAM_POLL_RINGS_MASK,
         .value = 0, /* don't generate drm_events on fence signaling */
        },
    };

    struct drm_virtgpu_context_init args = {
        .num_params     = ARRAY_SIZE(ctx_set_params),
        .pad            = 0,
        .ctx_set_params = (uintptr_t) &ctx_set_params,
    };

    return virtgpu_ioctl(gpu, DRM_IOCTL_VIRTGPU_CONTEXT_INIT, &args);
}

static int virtgpu_ioctl_get_caps(struct virtgpu *           gpu,
                                  enum virgl_renderer_capset id,
                                  uint32_t                   version,
                                  void *                     capset,
                                  size_t                     capset_size) {
    struct drm_virtgpu_get_caps args = {
        .cap_set_id  = id,
        .cap_set_ver = version,
        .addr        = (uintptr_t) capset,
        .size        = (__u32) capset_size,
        .pad         = 0,
    };

    return virtgpu_ioctl(gpu, DRM_IOCTL_VIRTGPU_GET_CAPS, &args);
}

static uint64_t virtgpu_ioctl_getparam(struct virtgpu * gpu, uint64_t param) {
    /* val must be zeroed because kernel only writes the lower 32 bits */
    uint64_t                    val  = 0;
    struct drm_virtgpu_getparam args = {
        .param = param,
        .value = (uintptr_t) &val,
    };

    const int ret = virtgpu_ioctl(gpu, DRM_IOCTL_VIRTGPU_GETPARAM, &args);
    return ret ? 0 : val;
}

struct apir_encoder * remote_call_prepare(struct virtgpu * gpu, ApirCommandType apir_cmd_type, int32_t cmd_flags) {
    /*
     * Prepare the command encoder and its buffer
     */

    static char encoder_buffer[4096];

    static struct apir_encoder enc;
    enc = {
        .cur   = encoder_buffer,
        .start = encoder_buffer,
        .end   = encoder_buffer + sizeof(encoder_buffer),
        .fatal = false,
    };

    /*
     * Fill the command encoder with the common args:
     * - cmd_type (int32_t)
     * - cmd_flags (int32_t)
     * - reply res id (uint32_t)
   */

    int32_t cmd_type = apir_cmd_type;

    // for testing during the hypervisor transition
    if (!gpu->use_apir_capset) {
        cmd_type += VENUS_COMMAND_TYPE_LENGTH;
    }
    apir_encode_int32_t(&enc, &cmd_type);
    apir_encode_int32_t(&enc, &cmd_flags);

    uint32_t reply_res_id = gpu->reply_shmem.res_id;
    apir_encode_uint32_t(&enc, &reply_res_id);

    return &enc;
}

void remote_call_finish(struct virtgpu * gpu, struct apir_encoder * enc, struct apir_decoder * dec) {
    UNUSED(gpu);

    if (!enc) {
        ERROR("Invalid (null) encoder :/");
    }

    if (!dec) {
        ERROR("Invalid (null) decoder :/");
    }

    if (apir_encoder_get_fatal(enc)) {
        ERROR("Failed to encode the output parameters.");
    }

    if (apir_decoder_get_fatal(dec)) {
        ERROR("Failed to decode the input parameters.");
    }
}

uint32_t remote_call(struct virtgpu *       gpu,
                     struct apir_encoder *  encoder,
                     struct apir_decoder ** decoder,
                     float                  max_wait_ms,
                     long long *            call_duration_ns) {
    /*
     * Prepare the reply notification pointer
     */

    volatile std::atomic_uint * atomic_reply_notif = (volatile std::atomic_uint *) gpu->reply_shmem.mmap_ptr;
    *atomic_reply_notif                            = 0;

    /*
     * Trigger the execbuf ioctl
     */

    struct drm_virtgpu_execbuffer args = {
        .flags   = VIRTGPU_EXECBUF_RING_IDX,
        .size    = (uint32_t) (encoder->cur - encoder->start),
        .command = (uintptr_t) encoder->start,

        .bo_handles     = 0,
        .num_bo_handles = 0,

        .fence_fd         = 0,
        .ring_idx         = 0,
        .syncobj_stride   = 0,
        .num_in_syncobjs  = 0,
        .num_out_syncobjs = 0,
        .in_syncobjs      = 0,
        .out_syncobjs     = 0,
    };

    *decoder = NULL;

    int ret = drmIoctl(gpu->fd, DRM_IOCTL_VIRTGPU_EXECBUFFER, &args);

    if (ret != 0) {
        FATAL("%s: the virtgpu EXECBUFFER ioctl failed (%d) :/ \n", ret);
    }

    /*
     * Wait for the response notification
     */

    start_timer(&wait_host_reply_timer);

    struct timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    long long start_time = (long long) ts_start.tv_sec * 1000000000LL + ts_start.tv_nsec;

    bool     timedout    = false;
    uint32_t notif_value = 0;
    while (true) {
        notif_value = std::atomic_load_explicit(atomic_reply_notif, std::memory_order_acquire);

        if (notif_value != 0) {
            break;
        }

        int64_t base_sleep_us = 15;

        os_time_sleep(base_sleep_us);

        if (max_wait_ms) {
            clock_gettime(CLOCK_MONOTONIC, &ts_end);
            long long end_time    = (long long) ts_end.tv_sec * 1000000000LL + ts_end.tv_nsec;
            float     duration_ms = (end_time - start_time) / 1000000;

            if (duration_ms > max_wait_ms) {
                timedout = true;
                break;
            }
        }
    }

    if (call_duration_ns) {
        *call_duration_ns = stop_timer(&wait_host_reply_timer);
    }

    if (max_wait_ms && timedout) {
        ERROR("timed out waiting for the host answer...");
        return APIR_FORWARD_TIMEOUT;
    }

    /*
     * Prepare the decoder
     */
    static struct apir_decoder response_dec;
    response_dec.cur = (char *) gpu->reply_shmem.mmap_ptr + sizeof(*atomic_reply_notif);
    response_dec.end = (char *) gpu->reply_shmem.mmap_ptr + gpu->reply_shmem.mmap_size;
    *decoder         = &response_dec;

    // extract the actual return value from the notif flag
    uint32_t returned_value = notif_value - 1;
    return returned_value;
}

static void log_call_duration(long long call_duration_ns, const char * name) {
    double call_duration_ms = (double) call_duration_ns / 1e6;  // 1 millisecond = 1e6 nanoseconds
    double call_duration_s  = (double) call_duration_ns / 1e9;  // 1 second = 1e9 nanoseconds

    if (call_duration_s > 1) {
        MESSAGE("%s: waited %.2fs for the %s host reply...", __func__, call_duration_s, name);
    } else if (call_duration_ms > 1) {
        MESSAGE("%s: waited %.2fms for the %s host reply...", __func__, call_duration_ms, name);
    } else {
        MESSAGE("%s: waited %lldns for the %s host reply...", __func__, call_duration_ns, name);
    }
}
