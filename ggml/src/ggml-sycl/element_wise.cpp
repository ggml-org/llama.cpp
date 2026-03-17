#include "common.hpp"
#include "ggml-sycl/presets.hpp"
#include "ggml.h"
#include "element_wise.hpp"

#define SYCL_GLOBAL_ID_LOOP(K, ITEM) \
    for (auto i = ITEM.get_global_id(0); i < (size_t)K; i += ITEM.get_global_range(0))

#define SYCL_LOCAL_ID_CALC(ITEM, IDX) \
    (ITEM.get_local_range(IDX) * ITEM.get_group(IDX) + ITEM.get_local_id(IDX))

static void acc_f32(const float * x, const float * y, float * dst, const int64_t ne,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const int64_t s11, const int64_t s12, const int64_t s13, const int64_t offset) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t i = SYCL_LOCAL_ID_CALC(item_ct1, 2);

    if (i >= ne) {
        return;
    }

    int64_t src1_idx = i - offset;

    int64_t tmp = src1_idx;
    const int64_t i13 = tmp / s13;
    tmp -= i13 * s13;
    const int64_t i12 = tmp / s12;
    tmp -= i12 * s12;
    const int64_t i11 = tmp / s11;
    tmp -= i11 * s11;
    const int64_t i10 = tmp;

    float val = x[i];
    if (src1_idx >= 0 && i10 < ne10 && i11 < ne11 && i12 < ne12 && i13 < ne13) {
        val += y[((i13*ne12 + i12) * ne11 + i11) * ne10 + i10];
    }
    dst[i] = val;
}

/* Unary OP funcs */
template<typename T>
static __dpct_inline__ T op_sgn(T x) {
    return x > static_cast<T>(0.f) ? static_cast<T>(1.f) : ((x < static_cast<T>(0.f) ? static_cast<T>(-1.f) : static_cast<T>(0.f)));
}

template<typename T>
static __dpct_inline__ T op_abs(T x) {
    return sycl::fabs(x);
}

template<typename T>
static __dpct_inline__ T op_elu(T x) {
    return (x > static_cast<T>(0.f)) ? x : sycl::expm1(x);
}

template<typename T>
static __dpct_inline__ T op_gelu(T x) {
    const T GELU_COEF_A    = static_cast<T>(0.044715f);
    const T SQRT_2_OVER_PI = static_cast<T>(0.79788456080286535587989211986876f);
    return static_cast<T>(0.5f) * x *
           (static_cast<T>(1.0f) +
            sycl::tanh(SQRT_2_OVER_PI * x * (static_cast<T>(1.0f) + GELU_COEF_A * x * x)));
}

template<typename T>
static __dpct_inline__ T op_silu(T x) {
    return x / (static_cast<T>(1.0f) + sycl::native::exp(-x));
}

template<typename T>
static __dpct_inline__ T op_gelu_quick(T x) {
    const T GELU_QUICK_COEF_LOCAL = static_cast<T>(-1.702f);
    return x * (static_cast<T>(1.0f) / (static_cast<T>(1.0f) + sycl::native::exp(GELU_QUICK_COEF_LOCAL * x)));
}

template<typename T>
static __dpct_inline__ T op_gelu_erf(T x) {
    const T SQRT_2_INV = static_cast<T>(0.70710678118654752440084436210484f);
    return static_cast<T>(0.5f) * x * (static_cast<T>(1.0f) + sycl::erf(x * SQRT_2_INV));
}

template<typename T>
static __dpct_inline__ T op_tanh(T x) {
    return sycl::tanh(x);
}

template<typename T>
static __dpct_inline__ T op_relu(T x) {
    return sycl::fmax(x, static_cast<T>(0));
}

template<typename T>
static __dpct_inline__ T op_sigmoid(T x) {
    return static_cast<T>(1.0f) / (static_cast<T>(1.0f) + sycl::native::exp(-x));
}

template<typename T>
static __dpct_inline__ T op_sqrt(T x) {
    return sycl::sqrt(x);
}

template<typename T>
static __dpct_inline__ T op_sin(T x) {
    return sycl::sin(x);
}

template<typename T>
static __dpct_inline__ T op_cos(T x) {
    return sycl::cos(x);
}

template<typename T>
static __dpct_inline__ T op_hardsigmoid(T x) {
    return sycl::fmin(static_cast<T>(1.0f), sycl::fmax(static_cast<T>(0.0f), (x + static_cast<T>(3.0f)) / static_cast<T>(6.0f)));
}

template<typename T>
static __dpct_inline__ T op_hardswish(T x) {
    return x * sycl::fmin(static_cast<T>(1.0f), sycl::fmax(static_cast<T>(0.0f), (x + static_cast<T>(3.0f)) / static_cast<T>(6.0f)));
}

template<typename T>
static __dpct_inline__ T op_exp(T x) {
    return sycl::exp(x);
}

template<typename T>
static __dpct_inline__ T op_log(T x) {
    if (x <= static_cast<T>(0)) {
        return neg_infinity<T>();
    }
    return sycl::log(x);
}

template<typename T>
static __dpct_inline__ T op_softplus(T x) {
    const float xf = (float) x;
    const float ax = sycl::fabs(xf);
    const float m  = sycl::fmax(xf, 0.0f);
    const float y  = m + sycl::log1p(sycl::exp(-ax));
    return (T) y;
}

template<typename T>
static __dpct_inline__ T op_neg(T x) {
    return -x;
}

template<typename T>
static __dpct_inline__ T op_step(T x) {
    return (x > static_cast<T>(0.0f)) ? static_cast<T>(1.0f) : static_cast<T>(0.0f);
}

template<typename T>
static __dpct_inline__ T op_leaky_relu(T x, float negative_slope) {
    T neg_slope_T = static_cast<T>(negative_slope);
    return sycl::fmax(x, static_cast<T>(0)) +
           sycl::fmin(x, static_cast<T>(0.0f)) * neg_slope_T;
}

template<typename T>
static __dpct_inline__ T op_sqr(T x) {
    return x * x;
}

template<typename T>
static __dpct_inline__ T op_clamp(T x, float min_val, float max_val) {
    return x < static_cast<T>(min_val) ? static_cast<T>(min_val) : (x > static_cast<T>(max_val) ? static_cast<T>(max_val) : x);
}

template<typename T>
static __dpct_inline__ T op_floor(T x) {
    return sycl::floor(x);
}

template<typename T>
static __dpct_inline__ T op_ceil(T x) {
    return sycl::ceil(x);
}

template<typename T>
static __dpct_inline__ T op_round(T x) {
    return sycl::round(x);
}

template<typename T>
static __dpct_inline__ T op_trunc(T x) {
    return sycl::trunc(x);
}

template<typename T, typename F>
static void unary_op_generic_kernel(
        const T * x,
        T * dst,
        const int k,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3,
        const size_t nb0,  const size_t nb1,  const size_t nb2,  const size_t nb3,
        const size_t nbd0, const size_t nbd1, const size_t nbd2, const size_t nbd3,
        const sycl::nd_item<1> & item_ct1,
        F func) {

        (void) ne3;
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        const int64_t i0 =  i % ne0;
        const int64_t i1 = (i / ne0)        % ne1;
        const int64_t i2 = (i / (ne0*ne1))  % ne2;
        const int64_t i3 =  i / (ne0*ne1*ne2);

        const char * src_base = (const char *) x;
        char       * dst_base = (char *) dst;

        const T * srcp = (const T *)(src_base + i0*nb0  + i1*nb1  + i2*nb2  + i3*nb3 );
        T *       dstp = (T *)(dst_base + i0*nbd0 + i1*nbd1 + i2*nbd2 + i3*nbd3);

        *dstp = func(*srcp);
    }
}

template<typename T>
static void unary_op_sqrt_kernel(const T * x, T * dst, const int k, const sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_sqrt(x[i]);
    }
}

template<typename T>
static void unary_op_sin_kernel(const T * x, T * dst, const int k, const sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_sin(x[i]);
    }
}

template<typename T>
static void unary_op_cos_kernel(const T * x, T * dst, const int k, const sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_cos(x[i]);
    }
}

template<typename T>
static void unary_op_log_kernel(const T * x, T * dst, const int k, const sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_log(x[i]);
    }
}


template<typename T>
static void unary_op_leaky_relu_kernel(const T * x, T * dst, const int k, float negative_slope, const sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_leaky_relu(x[i], negative_slope);
    }
}

template<typename T>
static void unary_op_sqr_kernel(const T * x, T * dst, const int k, const sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_sqr(x[i]);
    }
}

template<typename T>
static void unary_op_clamp_kernel(const T * x, T * dst, const int k, const sycl::nd_item<1> &item_ct1, float min_val, float max_val) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_clamp(x[i], min_val, max_val);
    }
}

template<typename T>
static void unary_op_floor_kernel(const T * x, T * dst, const int k, const sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_floor(x[i]);
    }
}

template<typename T>
static void unary_op_ceil_kernel(const T * x, T * dst, const int k, const sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_ceil(x[i]);
    }
}

template<typename T>
static void unary_op_round_kernel(const T * x, T * dst, const int k, const sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_round(x[i]);
    }
}

template<typename T>
static void unary_op_trunc_kernel(const T * x, T * dst, const int k, const sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_trunc(x[i]);
    }
}

template<typename T>
static void clamp(const T * x, T * dst, const float min, const float max, const int k,
                      const sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = x[i] < static_cast<T>(min) ? static_cast<T>(min) : (x[i] > static_cast<T>(max) ? static_cast<T>(max) : x[i]);
    }
}

template<typename T>
static void gated_op_fused_geglu(const T * x, const T * g, T * dst, const uint64_t k, const uint64_t n, const uint64_t o0, const uint64_t o1, const sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        const int64_t j0 = (i / n) * o0 + (i % n);
        const int64_t j1 = o0 == o1 ? j0 : (i / n) * o1 + (i % n);
        dst[i] = op_gelu(x[j0]) * g[j1];
    }
}

template<typename T>
static void gated_op_fused_reglu(const T * x, const T * g, T * dst, const uint64_t k, const uint64_t n, const uint64_t o0, const uint64_t o1, const sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        const int64_t j0 = (i / n) * o0 + (i % n);
        const int64_t j1 = o0 == o1 ? j0 : (i / n) * o1 + (i % n);
        dst[i] = op_relu(x[j0]) * g[j1];
    }
}

template<typename T>
static void gated_op_fused_swiglu(const T * x, const T * g, T * dst, const uint64_t k, const uint64_t n, const uint64_t o0, const uint64_t o1, const sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1)  {
        const int64_t j0 = (i / n) * o0 + (i % n);
        const int64_t j1 = o0 == o1 ? j0 : (i / n) * o1 + (i % n);
        dst[i] = op_silu(x[j0]) * g[j1];
    }
}

template<typename T>
static void gated_op_fused_geglu_erf(const T * x, const T * g, T * dst, const uint64_t k, const uint64_t n, const uint64_t o0, const uint64_t o1, const sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        const int64_t j0 = (i / n) * o0 + (i % n);
        const int64_t j1 = o0 == o1 ? j0 : (i / n) * o1 + (i % n);
        dst[i] = op_gelu_erf(x[j0]) * g[j1];
    }
}

template<typename T>
static void gated_op_fused_geglu_quick(const T * x, const T * g, T * dst, const uint64_t k, const uint64_t n, const uint64_t o0, const uint64_t o1, const sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        const int64_t j0 = (i / n) * o0 + (i % n);
        const int64_t j1 = o0 == o1 ? j0 : (i / n) * o1 + (i % n);
        dst[i] = op_gelu_quick(x[j0]) * g[j1];
    }
}

namespace ggml_sycl_detail {
static void acc_f32_sycl(const float *x, const float *y, float *dst,
                         const int64_t n_elements, const int64_t ne10, const int64_t ne11,
                         const int64_t ne12, const int64_t ne13, const int64_t s1, const int64_t s2, const int64_t s3,
                         const int64_t offset, queue_ptr stream) {
    const int num_blocks = (n_elements + SYCL_ACC_BLOCK_SIZE - 1) / SYCL_ACC_BLOCK_SIZE;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_ACC_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_ACC_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) {
                             acc_f32(x, y, dst, n_elements, ne10, ne11, ne12, ne13, s1, s2, s3, offset);
                         });
}

template<typename T>
static void arange_kernel(T * dst, const int k, T start, T step,
                         const sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = start + static_cast<T>(i) * step;
    }
}

template<typename KernelInvoker, typename... Args>
static inline void dispatch_ggml_sycl_op_unary(ggml_backend_sycl_context & ctx, ggml_tensor * dst, KernelInvoker kernel_invoker, Args&&... args) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->src[0]->type == dst->type);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    switch (dst->type) {
        case GGML_TYPE_F16:
            {
                auto data_pts = cast_data<sycl::half>(dst);
                kernel_invoker(data_pts.src, data_pts.dst, (int)ggml_nelements(dst->src[0]), main_stream, std::forward<Args>(args)...);
                break;
            }
        case GGML_TYPE_F32:
            {
                auto data_pts = cast_data<float>(dst);
                kernel_invoker(data_pts.src, data_pts.dst, (int)ggml_nelements(dst->src[0]), main_stream, std::forward<Args>(args)...);
                break;
            }
        default:
            GGML_ABORT("GGML tensor type not supported!\n");
    }
}

template<typename KernelInvoker, typename... Args>
static inline void dispatch_ggml_sycl_op_fused_glu(ggml_backend_sycl_context & ctx, ggml_tensor * dst, KernelInvoker kernel_invoker, Args&&... args) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->src[0]->type == dst->type);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const int64_t nc = src1 ? src0->ne[0] : src0->ne[0] / 2;;
    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_is_contiguous_1(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst));
    const int32_t swapped = ((const int32_t *) dst->op_params)[1];
    void * src0_d = src0->data;
    void * src1_d = src1 ? src1->data : src0->data;
    const int64_t src0_o = src0->nb[1];
    const int64_t src1_o = src1 ? src1->nb[1] : src0->nb[1];
    void * dst_d = dst->data;
    if (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->nb[0] == ggml_element_size(src1));
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(src0->type == src1->type);
    }
    switch (dst->type) {
        case GGML_TYPE_F16:
            {
                sycl::half * src0_p = (sycl::half *) src0_d;
                sycl::half * src1_p = (sycl::half *) src1_d;

                    if (!src1) {
                        src0_p += swapped ? nc : 0;
                        src1_p += swapped ? 0 : nc;
                    }
                kernel_invoker(src0_p,
                               src1_p,
                               (sycl::half *) dst_d,
                               ggml_nelements(dst),
                               nc,
                               src0_o / sizeof(sycl::half),
                               src1_o / sizeof(sycl::half),
                               main_stream,
                               std::forward<Args>(args)...);
                break;
            }
        case GGML_TYPE_F32:
            {
                float * src0_p = (float *) src0_d;
                float * src1_p = (float *) src1_d;

                    if (!src1) {
                        src0_p += swapped ? nc : 0;
                        src1_p += swapped ? 0 : nc;
                    }

                kernel_invoker(src0_p,
                               src1_p,
                               (float *) dst_d,
                               ggml_nelements(dst),
                               nc,
                               src0_o / sizeof(float),
                               src1_o / sizeof(float),
                               main_stream,
                               std::forward<Args>(args)...);
                break;
            }
        default:
            GGML_ABORT("GGML tensor type not supported!\n");
    }
}

template<typename F>
static inline void ggml_sycl_op_unary(
        ggml_backend_sycl_context & ctx, ggml_tensor * dst, F func) {

    ggml_tensor * src0 = dst->src[0];

    const int64_t ne0  = dst->ne[0];
    const int64_t ne1  = dst->ne[1];
    const int64_t ne2  = dst->ne[2];
    const int64_t ne3  = dst->ne[3];

    const size_t  nb0  = src0->nb[0];
    const size_t  nb1  = src0->nb[1];
    const size_t  nb2  = src0->nb[2];
    const size_t  nb3  = src0->nb[3];

    const size_t  nbd0 = dst->nb[0];
    const size_t  nbd1 = dst->nb[1];
    const size_t  nbd2 = dst->nb[2];
    const size_t  nbd3 = dst->nb[3];

    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [=](const auto* src, auto* dst_ptr, int k_elements, queue_ptr stream) {

            const int num_blocks = ceil_div(k_elements, 256);

            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(256),
                                  sycl::range<1>(256)),
                [=](sycl::nd_item<1> item_ct1) {
                    unary_op_generic_kernel(
                        src, dst_ptr, k_elements,
                        ne0, ne1, ne2, ne3,
                        nb0, nb1, nb2, nb3,
                        nbd0, nbd1, nbd2, nbd3,
                        item_ct1,
                        func
                    );
                });
        });
}


static inline void ggml_sycl_op_arange(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    float start, stop, step;
    memcpy(&start, dst->op_params, sizeof(float));
    memcpy(&stop, (float *) dst->op_params + 1, sizeof(float));
    memcpy(&step, (float *) dst->op_params + 2, sizeof(float));
    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    float * dst_ptr = (float *)dst->data;
    const int k = (int)ggml_nelements(dst);
    const int num_blocks = ceil_div(k, SYCL_ARANGE_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(SYCL_ARANGE_BLOCK_SIZE),
                          sycl::range<1>(SYCL_ARANGE_BLOCK_SIZE)),
        [=](sycl::nd_item<1> item_ct1) {
            arange_kernel(dst_ptr, k, start, step, item_ct1);
        });
}

} // namespace ggml_sycl_detail



static inline void ggml_sycl_op_sgn(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        return op_sgn(x);
    });
}


static inline void ggml_sycl_op_abs(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        return op_abs(x);
    });
}

static inline void ggml_sycl_op_elu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        return op_elu(x);
    });
}
static inline void ggml_sycl_op_silu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        return op_silu(x);
    });
}

static inline void ggml_sycl_op_gelu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        return op_gelu(x);
    });
}

static inline void ggml_sycl_op_gelu_quick(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        return op_gelu_quick(x);
    });
}

static inline void ggml_sycl_op_gelu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        return op_gelu_erf(x);
    });
}

static inline void ggml_sycl_op_tanh(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        return op_tanh(x);
    });
}

static inline void ggml_sycl_op_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        return op_relu(x);
    });
}

static inline void ggml_sycl_op_hardsigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        return op_hardsigmoid(x);
    });
}

static inline void ggml_sycl_op_hardswish(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        return op_hardswish(x);
    });
}

static inline void ggml_sycl_op_exp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        return op_exp(x);
    });
}

static inline void ggml_sycl_op_log(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](const auto* src, auto* dst_ptr, int k_elements, queue_ptr stream) {
            const int num_blocks = ceil_div(k_elements, SYCL_EXP_BLOCK_SIZE); // Using EXP block size
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(SYCL_EXP_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_EXP_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) {
                    unary_op_log_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

static inline void ggml_sycl_op_softplus(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        return op_softplus(x);
    });
}

static inline void ggml_sycl_op_neg(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        return op_neg(x);
    });
}


static inline void ggml_sycl_op_step(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        return op_step(x);
    });
}

static inline void ggml_sycl_op_sigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        return op_sigmoid(x);
    });
}

static inline void ggml_sycl_op_sqrt(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](const auto* src, auto* dst_ptr, int k_elements, queue_ptr stream) {
            const int num_blocks = ceil_div(k_elements, SYCL_SQRT_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(SYCL_SQRT_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_SQRT_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) {
                    unary_op_sqrt_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

static inline void ggml_sycl_op_sin(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](const auto* src, auto* dst_ptr, int k_elements, queue_ptr stream) {
            const int num_blocks = ceil_div(k_elements, SYCL_SIN_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(SYCL_SIN_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_SIN_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) {
                    unary_op_sin_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

static inline void ggml_sycl_op_cos(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](const auto* src, auto* dst_ptr, int k_elements, queue_ptr stream) {
            const int num_blocks = ceil_div(k_elements, SYCL_SIN_BLOCK_SIZE); // Using SIN block size
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(SYCL_SIN_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_SIN_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) {
                    unary_op_cos_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

static inline void ggml_sycl_op_leaky_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    float negative_slope;
    memcpy(&negative_slope, dst->op_params, sizeof(float));
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](const auto* src, auto* dst_ptr, int k_elements, queue_ptr stream, float slope) {
            const int num_blocks = ceil_div(k_elements, SYCL_RELU_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(SYCL_RELU_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_RELU_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) {
                    unary_op_leaky_relu_kernel(src, dst_ptr, k_elements, slope, item_ct1);
                });
        }, negative_slope);
}

static inline void ggml_sycl_op_sqr(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](const auto* src, auto* dst_ptr, int k_elements, queue_ptr stream) {
            const int num_blocks = ceil_div(k_elements, SYCL_SQR_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(SYCL_SQR_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_SQR_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) {
                    unary_op_sqr_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

static inline void ggml_sycl_op_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    float min_val;
    float max_val;
    memcpy(&min_val, dst->op_params, sizeof(float));
    memcpy(&max_val, (float *) dst->op_params + 1, sizeof(float));
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](const auto* src, auto* dst_ptr, int k_elements, queue_ptr stream, float min_arg, float max_arg) {
            const int num_blocks = ceil_div(k_elements, SYCL_CLAMP_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(SYCL_CLAMP_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_CLAMP_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) {
                    clamp(src, dst_ptr, min_arg, max_arg, k_elements, item_ct1);
                });
        }, min_val, max_val);
}

static inline void ggml_sycl_op_floor(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](const auto* src, auto* dst_ptr, int k_elements, queue_ptr stream) {
            const int num_blocks = ceil_div(k_elements, 256);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(256),
                                  sycl::range<1>(256)),
                [=](sycl::nd_item<1> item_ct1) {
                    unary_op_floor_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

static inline void ggml_sycl_op_ceil(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        return op_ceil(x);
    });
}

static inline void ggml_sycl_op_round(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](const auto* src, auto* dst_ptr, int k_elements, queue_ptr stream) {
            const int num_blocks = ceil_div(k_elements, 256);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(256),
                                  sycl::range<1>(256)),
                [=](sycl::nd_item<1> item_ct1) {
                    unary_op_round_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

static inline void ggml_sycl_op_trunc(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](const auto* src, auto* dst_ptr, int k_elements, queue_ptr stream) {
            const int num_blocks = ceil_div(k_elements, 256);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(256),
                                  sycl::range<1>(256)),
                [=](sycl::nd_item<1> item_ct1) {
                    unary_op_trunc_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

static inline void ggml_sycl_op_acc(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float       * dst_d  = (float       *)  dst->data;

    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(dst->nb[0] == ggml_element_size(dst));
    GGML_ASSERT(ggml_is_contiguously_allocated(dst));

    const int64_t s1     = dst->op_params[0] / sizeof(float);
    const int64_t s2     = dst->op_params[1] / sizeof(float);
    const int64_t s3     = dst->op_params[2] / sizeof(float);
    const int64_t offset = dst->op_params[3] / sizeof(float);

    ggml_sycl_detail::acc_f32_sycl(src0_d, src1_d, dst_d, ggml_nelements(dst),
        src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
        s1, s2, s3, offset, stream);
}

static inline void ggml_sycl_op_geglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_fused_glu(ctx, dst,
        [](const auto* x_ptr, const auto* g_ptr, auto* dst_ptr, uint64_t k, uint64_t n, uint64_t o0, uint64_t o1, queue_ptr main_stream) {
            const uint32_t num_blocks = ceil_div(k, SYCL_GELU_BLOCK_SIZE);
            main_stream->parallel_for(
                    sycl::nd_range<1>((num_blocks * sycl::range<1>(SYCL_GELU_BLOCK_SIZE)), sycl::range<1>(SYCL_GELU_BLOCK_SIZE)), [=](sycl::nd_item<1> item_ct1) {
                gated_op_fused_geglu(x_ptr, g_ptr, dst_ptr, k, n, o0, o1, item_ct1);
            });
        });
}

static inline void ggml_sycl_op_reglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_fused_glu(ctx, dst,
        [](const auto* x_ptr, const auto* g_ptr, auto* dst_ptr, uint64_t k, uint64_t n, uint64_t o0, uint64_t o1, queue_ptr main_stream) {
            const uint32_t num_blocks = ceil_div((uint32_t)k, SYCL_RELU_BLOCK_SIZE); // Using RELU block size for reglu
            main_stream->parallel_for(
                    sycl::nd_range<1>((num_blocks * sycl::range<1>(SYCL_RELU_BLOCK_SIZE)), sycl::range<1>(SYCL_RELU_BLOCK_SIZE)), [=](sycl::nd_item<1> item_ct1) {
                gated_op_fused_reglu(x_ptr, g_ptr, dst_ptr, k, n, o0, o1, item_ct1);
            });
        });
}

static inline void ggml_sycl_op_swiglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_fused_glu(ctx, dst,
        [](const auto* x_ptr, const auto* g_ptr, auto* dst_ptr, uint64_t k, uint64_t n, uint64_t o0, uint64_t o1, queue_ptr main_stream) {
            const uint32_t num_blocks = ceil_div((uint32_t)k, SYCL_SILU_BLOCK_SIZE); // Using SILU block size for swiglu
            main_stream->parallel_for(
                    sycl::nd_range<1>((num_blocks * sycl::range<1>(SYCL_SILU_BLOCK_SIZE)), sycl::range<1>(SYCL_SILU_BLOCK_SIZE)), [=](sycl::nd_item<1> item_ct1) {
                gated_op_fused_swiglu(x_ptr, g_ptr, dst_ptr, k, n, o0, o1, item_ct1);
            });
        });
}

__dpct_inline__ float ggml_sycl_op_swiglu_oai_single(float x, float g, float alpha = 1.702f, float limit = 7.0f) {
    x = sycl::fmin(x, limit);
    g = sycl::fmax(sycl::fmin(g, limit), -limit);

    float out_glu = x / (1.0f + sycl::native::exp(-x * alpha));
    out_glu = out_glu * (1.0f + g);
    return out_glu;
}


template <typename T>
static void swiglu_oai_kernel(const T * x, const T * g, T * dst, const int64_t k,
                              const int64_t n, const int64_t o0, const int64_t o1,
                              float alpha, float limit, sycl::nd_item<3> item_ct1) {
    const int64_t i = int64_t(item_ct1.get_local_range(2)) * item_ct1.get_group(2) + item_ct1.get_local_id(2);

    if (i >= k) {
        return;
    }

    const int64_t j0 = (i / n) * o0 + (i % n);
    const int64_t j1 = o0 == o1 ? j0 : (i / n) * o1 + (i % n);

    float xi = x[j0];
    float gi = g[j1];

    dst[i] = ggml_sycl_op_swiglu_oai_single(xi, gi, alpha, limit);
}

template <typename T>
static void swiglu_oai_sycl(const T *       x,
                            const T *       g,
                            T *             dst,
                            const int64_t   k,
                            const int64_t   n,
                            const int64_t   o0,
                            const int64_t   o1,
                            const float     alpha,
                            const float     limit,
                            dpct::queue_ptr stream) {
    const int64_t num_blocks = (k + SYCL_GLU_BLOCK_SIZE - 1) / SYCL_GLU_BLOCK_SIZE;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_GLU_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_GLU_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) {
                             swiglu_oai_kernel(x, g, dst, k, n, o0, o1, alpha, limit, item_ct1);
                         });
}

void ggml_sycl_op_swiglu_oai(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    void * src0_d = src0->data;
    void * src1_d = src1 ? src1->data : src0->data;
    const int64_t src0_o = src0->nb[1];
    const int64_t src1_o = src1 ? src1->nb[1] : src0->nb[1];
    void * dst_d = dst->data;
    const int64_t nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    dpct::queue_ptr     stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(src0->nb[0] == ggml_element_size(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == ggml_nrows(src0));

    if (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->nb[0] == ggml_element_size(src1));
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(src0->type == src1->type);
    }

    //const int32_t swapped = ((const int32_t *) dst->op_params)[1];
    const int32_t swapped = ggml_get_op_params_i32(dst, 1);
    const float alpha = ggml_get_op_params_f32(dst, 2);
    const float limit = ggml_get_op_params_f32(dst, 3);

    float * src0_p = (float *) src0_d;
    float * src1_p = (float *) src1_d;

    if (!src1) {
        src0_p += swapped ? nc : 0;
        src1_p += swapped ? 0 : nc;
    }

    swiglu_oai_sycl(src0_p, src1_p, (float *)dst_d, ggml_nelements(dst), nc, src0_o / sizeof(float), src1_o / sizeof(float), alpha, limit, stream);
}

static inline void ggml_sycl_op_geglu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_fused_glu(ctx, dst,
        [](const auto* x_ptr, const auto* g_ptr, auto* dst_ptr, uint64_t k, uint64_t n, uint64_t o0, uint64_t o1, queue_ptr main_stream) {
            const uint32_t num_blocks = ceil_div(k, SYCL_GELU_BLOCK_SIZE);
            main_stream->parallel_for(
                    sycl::nd_range<1>((num_blocks * sycl::range<1>(SYCL_GELU_BLOCK_SIZE)), sycl::range<1>(SYCL_GELU_BLOCK_SIZE)), [=](sycl::nd_item<1> item_ct1) {
                gated_op_fused_geglu_erf(x_ptr, g_ptr, dst_ptr, k, n, o0, o1, item_ct1);
            });
        });
}

static inline void ggml_sycl_op_geglu_quick(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_fused_glu(ctx, dst,
        [](const auto* x_ptr, const auto* g_ptr, auto* dst_ptr, uint64_t k, uint64_t n, uint64_t o0, uint64_t o1, queue_ptr main_stream) {
            const uint32_t num_blocks = ceil_div(k, SYCL_GELU_BLOCK_SIZE);
            main_stream->parallel_for(
                    sycl::nd_range<1>((num_blocks * sycl::range<1>(SYCL_GELU_BLOCK_SIZE)), sycl::range<1>(SYCL_GELU_BLOCK_SIZE)), [=](sycl::nd_item<1> item_ct1) {
                gated_op_fused_geglu_quick(x_ptr, g_ptr, dst_ptr, k, n, o0, o1, item_ct1);
            });
        });
}


void ggml_sycl_sqrt(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_sqrt(ctx, dst);
}

void ggml_sycl_sin(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_sin(ctx, dst);
}

void ggml_sycl_cos(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_cos(ctx, dst);
}

void ggml_sycl_acc(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_acc(ctx, dst);
}

void ggml_sycl_gelu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_gelu(ctx, dst);
}

void ggml_sycl_silu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_silu(ctx, dst);
}

void ggml_sycl_gelu_quick(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_gelu_quick(ctx, dst);
}

void ggml_sycl_gelu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_gelu_erf(ctx, dst);
}

void ggml_sycl_tanh(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_tanh(ctx, dst);
}

void ggml_sycl_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_relu(ctx, dst);
}

void ggml_sycl_sigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_sigmoid(ctx, dst);
}

void ggml_sycl_hardsigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_hardsigmoid(ctx, dst);
}

void ggml_sycl_hardswish(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_hardswish(ctx, dst);
}

void ggml_sycl_exp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_exp(ctx, dst);
}

void ggml_sycl_log(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_log(ctx, dst);
}

void ggml_sycl_softplus(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_softplus(ctx, dst);
}

void ggml_sycl_neg(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_neg(ctx, dst);
}

void ggml_sycl_step(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_step(ctx, dst);
}

void ggml_sycl_leaky_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_leaky_relu(ctx, dst);
}

void ggml_sycl_sqr(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_sqr(ctx, dst);
}

void ggml_sycl_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_clamp(ctx, dst);
}

void ggml_sycl_sgn(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_sgn(ctx, dst);
}

void ggml_sycl_abs(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_abs(ctx, dst);
}

void ggml_sycl_elu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_elu(ctx, dst);
}

void ggml_sycl_geglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_geglu(ctx, dst);
}

void ggml_sycl_reglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_reglu(ctx, dst);
}

void ggml_sycl_swiglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_swiglu(ctx, dst);
}

void ggml_sycl_swiglu_oai(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_swiglu_oai(ctx, dst);
}

void ggml_sycl_geglu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_geglu_erf(ctx, dst);
}

void ggml_sycl_geglu_quick(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_geglu_quick(ctx, dst);
}

void ggml_sycl_arange(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/0);
    ggml_sycl_detail::ggml_sycl_op_arange(ctx, dst);
}

void ggml_sycl_floor(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_floor(ctx, dst);
}

void ggml_sycl_ceil(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_ceil(ctx, dst);
}

void ggml_sycl_round(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_round(ctx, dst);
}

void ggml_sycl_trunc(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_trunc(ctx, dst);
}
