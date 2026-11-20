// SPDX-License-Identifier: MIT
#include "mmvq.hpp"
#include "mmvq-arl.hpp"
#include <mutex>

#ifdef GGML_SYCL_MMVQ_HAS_XMX
template <template <bool> class Traits>
static void ggml_sycl_mul_mat_vec_q_xmx_arl_run(bool            short_rows,
                                            const void *    vx,
                                            const void *    vgate,
                                            const char *    vy,
                                            float *         dst,
                                            int             ne00,
                                            int64_t         nrows,
                                            int64_t         ncols_dst,
                                            int             stride_y_bytes,
                                            int             ldd,
                                            int             glu_op,
                                            dpct::queue_ptr stream) {
    namespace esimd = ggml_sycl_esimd_arl;
    if (vgate) {
        if (short_rows) {
            esimd::xmx_mul_mat_ncols<Traits<true>, true>(vx, vgate, vy, dst, ne00, nrows, ncols_dst, stride_y_bytes,
                                                         ldd, glu_op, stream);
        } else {
            esimd::xmx_mul_mat_ncols<Traits<false>, true>(vx, vgate, vy, dst, ne00, nrows, ncols_dst, stride_y_bytes,
                                                          ldd, glu_op, stream);
        }
    } else if (short_rows) {
        esimd::xmx_mul_mat_ncols<Traits<true>>(vx, nullptr, vy, dst, ne00, nrows, ncols_dst, stride_y_bytes, ldd, 0,
                                               stream);
    } else {
        esimd::xmx_mul_mat_ncols<Traits<false>>(vx, nullptr, vy, dst, ne00, nrows, ncols_dst, stride_y_bytes, ldd, 0,
                                                stream);
    }
}

bool ggml_sycl_mul_mat_vec_q_xmx_arl(ggml_type       type,
                                             bool            short_rows,
                                             const void *    vx,
                                             const void *    vgate,
                                             const char *    vy,
                                             float *         dst,
                                             int             ne00,
                                             int64_t         nrows,
                                             int64_t         ncols_dst,
                                             int             stride_y_bytes,
                                             int             ldd,
                                             int             glu_op,
                                             dpct::queue_ptr stream) {
    if (!ggml_sycl_xmx_supports_type(type)) {
        return false;
    }
    static std::once_flag log_once;
    std::call_once(log_once, [] {
        GGML_LOG_INFO("ggml_sycl: using Xe-LPG+ XMX ESIMD (SLM=%d, gather=%d, native 2D=%d)\n",
                      XMX_ENABLE_SLM, XMX_ENABLE_GATHER_TRANSPOSE, XMX_ENABLE_TRANSPOSED_2D_LOADS);
    });
#    define XMX_RUN(TRAITS)                                                                                       \
        ggml_sycl_mul_mat_vec_q_xmx_arl_run<ggml_sycl_esimd_arl::TRAITS>(short_rows, vx, vgate, vy, dst, ne00, nrows,     \
                                                                 ncols_dst, stride_y_bytes, ldd, glu_op, stream); \
        return true
    switch (type) {
        case GGML_TYPE_Q2_K:
            XMX_RUN(xmx_traits_q2_k);
        case GGML_TYPE_Q3_K:
            XMX_RUN(xmx_traits_q3_k);
        case GGML_TYPE_Q4_K:
            XMX_RUN(xmx_traits_q4_k);
        case GGML_TYPE_Q5_K:
            XMX_RUN(xmx_traits_q5_k);
        case GGML_TYPE_Q6_K:
            XMX_RUN(xmx_traits_q6_k);
        case GGML_TYPE_Q8_0:
            XMX_RUN(xmx_traits_q8_0);
        default:
            return false;
    }
#    undef XMX_RUN
}

#endif // GGML_SYCL_MMVQ_HAS_XMX
