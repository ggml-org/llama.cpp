#include "ggml-dsp.h"

static int32 g_thread_counts = 1;

int ggmlop_dsp_open(const char * uri, remote_handle64 * handle) {
    void * tptr = NULL;
    GGMLHEXAGON_LOG_DEBUG("uri %s", uri);
    tptr = (void *)malloc(1);
    GGML_ASSERT(NULL != tptr);
    *handle = (remote_handle64)tptr;

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

AEEResult ggmlop_dsp_setclocks(remote_handle64 handle, int32 power_level, int32 latency, int32 mulmat_algo, int32 thread_counts) {
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
    if (mulmat_algo) {
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
