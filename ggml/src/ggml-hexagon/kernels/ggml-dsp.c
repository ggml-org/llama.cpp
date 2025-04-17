#include "ggml-dsp.h"

// =================================================================================================
// tiny ggml-dsp, ported from original ggml
// =================================================================================================
static int32 g_thread_counts = 1;

void ggmlhexagon_log_internal(int level, const char *file, const char *func, int line, const char *format, ...) {
#if !GGMLHEXAGON_DEBUG
    return;
#endif
    static char s_ggmlhexagon_log_internal_buf[GGMLHEXAGON_LOGBUF_LEN];
    va_list args;
    va_start(args, format);
    int len_prefix = snprintf(s_ggmlhexagon_log_internal_buf, GGMLHEXAGON_LOGBUF_LEN, "[%s, %d]: ",
                              func, line);
    int len = vsnprintf(s_ggmlhexagon_log_internal_buf + len_prefix,
                        GGMLHEXAGON_LOGBUF_LEN - len_prefix, format, args);
    if (len < (GGMLHEXAGON_LOGBUF_LEN - len_prefix)) {
        FARF(ALWAYS, "%s\n", s_ggmlhexagon_log_internal_buf);
    }
    va_end(args);
}

void ggmlhexagon_dump_tensor_elements(const ggml_tensor * tensor) {
#if !GGMLHEXAGON_DEBUG
    return;
#endif
    float value = 0;
    char tmpbuf[GGMLHEXAGON_LOGBUF_LEN];
    size_t buflen = 0;
    if (tensor->type == GGML_TYPE_F32) {
        memset(tmpbuf, 0, GGMLHEXAGON_LOG_LEVEL_DEBUG);
        for (int h = 0; h < tensor->ne[3]; h++) {
            for (int i = 0; i < tensor->ne[2]; i++) {
                for (int j = 0; j < tensor->ne[1]; j++) {
                    for (int k = 0; k < tensor->ne[0]; k++) {
                        value = ((float *) tensor->data)[h * tensor->ne[2] + i * tensor->ne[1] +
                                                         j * tensor->ne[0] + k];
                        buflen += snprintf(tmpbuf + buflen, GGMLHEXAGON_LOGBUF_LEN - buflen, "%-4.2f\t", value);
                    }
                    buflen += snprintf(tmpbuf + buflen, GGMLHEXAGON_LOGBUF_LEN - buflen, "\n");
                }
            }
        }
        GGMLHEXAGON_LOG_DEBUG("\n%s\n", tmpbuf);
    }

    GGMLHEXAGON_LOG_DEBUG("\n");
}

void ggmlhexagon_dump_tensor(const ggml_tensor * tensor, int dump_tensor_data) {
    GGMLHEXAGON_LOG_DEBUG("ne = %5d x %5d x %5d x %5d , nb = (%5zi, %5zi, %5zi, %5zi)\n",
         tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3],
         tensor->nb[0], tensor->nb[1], tensor->nb[2], tensor->nb[3]);

    if ((1 == dump_tensor_data) && (ggml_nbytes(tensor) < 320)) {
        ggmlhexagon_dump_tensor_elements(tensor);
    }
}

size_t ggml_row_size(enum ggml_type type, int64_t ne) {
    return 4*ne;
}

size_t ggml_nbytes(const struct ggml_tensor * tensor) {
    size_t nbytes;
    const size_t blck_size = 1;
    if (blck_size == 1) {
        nbytes = 4;
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            nbytes += (tensor->ne[i] - 1)*tensor->nb[i];
        }
    } else {
        nbytes = tensor->ne[0]*tensor->nb[0]/blck_size;
        for (int i = 1; i < GGML_MAX_DIMS; ++i) {
            nbytes += (tensor->ne[i] - 1)*tensor->nb[i];
        }
    }

    return nbytes;
}

bool ggml_is_empty(const struct ggml_tensor * tensor) {
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (tensor->ne[i] == 0) {
            return true;
        }
    }
    return false;
}

bool ggml_can_repeat(const struct ggml_tensor * t0, const struct ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    return ggml_is_empty(t0) ? ggml_is_empty(t1) :
           (t1->ne[0]%t0->ne[0] == 0) &&
           (t1->ne[1]%t0->ne[1] == 0) &&
           (t1->ne[2]%t0->ne[2] == 0) &&
           (t1->ne[3]%t0->ne[3] == 0);
}

bool ggml_are_same_shape(const struct ggml_tensor * t0, const struct ggml_tensor * t1) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");
    return
            (t0->ne[0] == t1->ne[0]) &&
            (t0->ne[1] == t1->ne[1]) &&
            (t0->ne[2] == t1->ne[2]) &&
            (t0->ne[3] == t1->ne[3]);
}

int64_t ggml_nrows(const struct ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    return tensor->ne[1]*tensor->ne[2]*tensor->ne[3];
}

bool ggml_is_transposed(const struct ggml_tensor * tensor) {
    return tensor->nb[0] > tensor->nb[1];
}

bool ggml_is_contiguous_n(const struct ggml_tensor * tensor, int n) {
    size_t next_nb = 4;
    if (tensor->ne[0] != 1 && tensor->nb[0] != next_nb) {
        return false;
    }
    next_nb *= tensor->ne[0];
    for (int i = 1; i < GGML_MAX_DIMS; i++) {
        if (tensor->ne[i] != 1) {
            if (i > n) {
                if (tensor->nb[i] != next_nb) {
                    return false;
                }
                next_nb *= tensor->ne[i];
            } else {
                // this dimension does not need to be contiguous
                next_nb = tensor->ne[i]*tensor->nb[i];
            }
        }
    }
    return true;
}

int64_t ggml_nelements(const struct ggml_tensor * tensor) {
    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

    return tensor->ne[0]*tensor->ne[1]*tensor->ne[2]*tensor->ne[3];
}

static bool ggml_is_contiguous_0(const struct ggml_tensor * tensor) {
    return ggml_is_contiguous_n(tensor, 0);
}

bool ggml_is_contiguous(const struct ggml_tensor * tensor) {
    return ggml_is_contiguous_0(tensor);
}

void ggml_abort(const char * file, int line, const char * fmt, ...) {
    GGMLHEXAGON_LOG_DEBUG("enter ggml_abort");
    abort();
}

static inline uint64 hexagon_perf_get_time_us(void) {
    unsigned long long count;
    asm volatile (" %0 = c31:30 " : "=r"(count));
    return (uint64)(count) * 10ull / 192ull;
}

int64_t ggml_time_ms(void) {
    return hexagon_perf_get_time_us() * 1000;
}

int64_t ggml_time_us(void) {
    return hexagon_perf_get_time_us();
}

int ggmlop_get_thread_counts(void) {
    return g_thread_counts;
}

// =================================================================================================
//  implementation of ggml-hexagon kernel skel function
// =================================================================================================
int ggmlop_dsp_open(const char*uri, remote_handle64* handle) {
    void *tptr = NULL;
    GGMLHEXAGON_LOG_DEBUG("uri %s", uri);
    tptr = (void *)malloc(1);
    *handle = (remote_handle64)tptr;
    assert(*handle);

    GGMLHEXAGON_LOG_DEBUG("api_version = 0x%x", qurt_api_version());
    GGMLHEXAGON_LOG_DEBUG("hvx units = 0x%d", qurt_hvx_get_units());
    qurt_arch_version_t  vers;
    qurt_sysenv_get_arch_version(&vers);
    GGMLHEXAGON_LOG_DEBUG("arch_version=0x%x", vers.arch_version);
    qurt_sysenv_app_heap_t aheap;
    qurt_sysenv_get_app_heap(&aheap);
    GGMLHEXAGON_LOG_DEBUG("aheap.heap_base=0x%x, aheap.heap_limit=0x%x", aheap.heap_base, aheap.heap_limit);
    qurt_sysenv_max_hthreads_t mhwt;
    qurt_sysenv_get_max_hw_threads(&mhwt);
    GGMLHEXAGON_LOG_DEBUG("max hardware threads counts=%d", mhwt.max_hthreads);
    g_thread_counts = mhwt.max_hthreads;

    return 0;
}

int ggmlop_dsp_close(remote_handle64 handle) {
    if (handle)
        free((void*)handle);

    return 0;
}

AEEResult ggmlop_dsp_setclocks(remote_handle64 handle, int32 power_level, int32 latency, int32 dcvs_enabled, int32 thread_counts) {
    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__);
    HAP_power_request_t request;
    memset(&request, 0, sizeof(HAP_power_request_t));
    request.type = HAP_power_set_apptype;
    request.apptype = HAP_POWER_COMPUTE_CLIENT_CLASS;

    GGMLHEXAGON_LOG_DEBUG("user specified thread_counts %d", thread_counts);
    if (thread_counts > 1)
        g_thread_counts = (thread_counts > g_thread_counts) ? g_thread_counts : thread_counts;
    else
        g_thread_counts = 1;
    GGMLHEXAGON_LOG_DEBUG("real thread_counts %d", g_thread_counts);

    void * ggmop_ctx = (void*)(handle);
    int retval = HAP_power_set(ggmop_ctx, &request);
    if (retval)  {
        GGMLHEXAGON_LOG_DEBUG("failed first power vote");
        return AEE_EFAILED;
    }

    //configure clocks & DCVS mode
    memset(&request, 0, sizeof(HAP_power_request_t));
    request.type = HAP_power_set_DCVS_v2;
    request.dcvs_v2.dcvs_enable = TRUE;
    request.dcvs_v2.dcvs_params.target_corner = (HAP_dcvs_voltage_corner_t)power_level;
    if (dcvs_enabled) {
        request.dcvs_v2.dcvs_params.min_corner = HAP_DCVS_VCORNER_DISABLE;
        request.dcvs_v2.dcvs_params.max_corner = HAP_DCVS_VCORNER_DISABLE;
    } else {
        request.dcvs_v2.dcvs_params.min_corner = request.dcvs_v2.dcvs_params.target_corner;
        request.dcvs_v2.dcvs_params.max_corner = request.dcvs_v2.dcvs_params.target_corner;
    }
    request.dcvs_v2.dcvs_option     = HAP_DCVS_V2_PERFORMANCE_MODE;
    request.dcvs_v2.set_dcvs_params = TRUE;
    request.dcvs_v2.set_latency     = TRUE;
    request.dcvs_v2.latency         = latency;
    retval = HAP_power_set(ggmop_ctx, &request);
    if (retval) {
        GGMLHEXAGON_LOG_DEBUG("failed to vote for performance mode");
        return AEE_EFAILED;
    }

    memset(&request, 0, sizeof(HAP_power_request_t));
    request.type = HAP_power_set_HVX;
    request.hvx.power_up = TRUE;
    retval = HAP_power_set(ggmop_ctx, &request);
    if (retval) {
        GGMLHEXAGON_LOG_DEBUG("failed to vote for HVX power");
        return AEE_EFAILED;
    }
    GGMLHEXAGON_LOG_DEBUG("leave %s", __func__ );
    return AEE_SUCCESS;
}

// =================================================================================================
//  implementation of ggml-hexagon kernel, it's better to put every hexagon-kernel to a single file
// =================================================================================================
int ggmlop_dsp_softmax(remote_handle64 h, const dsptensor * src0, const dsptensor * src1, dsptensor * dst) {
    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__ );
    GGMLHEXAGON_LOG_DEBUG("leave %s", __func__ );
    return 0;
}

int ggmlop_dsp_rmsnorm(remote_handle64 h, const dsptensor * src0, const dsptensor * src1, dsptensor * dst) {
    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__ );
    GGMLHEXAGON_LOG_DEBUG("leave %s", __func__ );
    return 0;
}

int ggmlop_dsp_pool2d(remote_handle64 h, const dsptensor * src0, const dsptensor * src1, dsptensor * dst) {
    GGMLHEXAGON_LOG_DEBUG("enter %s", __func__ );
    GGMLHEXAGON_LOG_DEBUG("leave %s", __func__ );
    return 0;
}
