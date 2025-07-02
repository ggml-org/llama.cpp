#include "common.hpp"
#include "ggml-sycl/presets.hpp"
#include "ggml.h"
#include "element_wise.hpp"

#define SYCL_GLOBAL_ID_LOOP(K, ITEM) \
    for (auto i = ITEM.get_global_id(0); i < (size_t)K; i += ITEM.get_global_range(0))

#define SYCL_LOCAL_ID_CALC(ITEM, IDX) \
    (ITEM.get_local_range(IDX) * ITEM.get_group(IDX) + ITEM.get_local_id(IDX))


static void acc_f32(const float * x, const float * y, float * dst, const int ne,
    const int ne10, const int ne11, const int ne12,
    const int nb1, const int nb2, int offset, const sycl::nd_item<1> &item_ct1) {
    const int i = SYCL_LOCAL_ID_CALC(item_ct1, 0);
    if (i >= ne) {
        return;
    }
    int src1_idx = i - offset;
    int oz = src1_idx / nb2;
    int oy = (src1_idx - (oz * nb2)) / nb1;
    int ox = src1_idx % nb1;
    if (src1_idx >= 0 && ox < ne10 && oy < ne11 && oz < ne12) {
        dst[i] = x[i] + y[ox + oy * ne10 + oz * ne10 * ne11];
    } else {
        dst[i] = x[i];
    }
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
static void unary_op_sgn_kernel(const T * x, T * dst, const int k, const sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_sgn(x[i]);
    }
}

template<typename T>
static void unary_op_abs_kernel(const T * x, T * dst, const int k, const sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_abs(x[i]);
    }
}

template<typename T>
static void unary_op_elu_kernel(const T * x, T * dst, const int k, const sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_elu(x[i]);
    }
}

template<typename T>
static void unary_op_gelu_kernel(const T * x, T * dst, const int k, const sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_gelu(x[i]);
    }
}

template<typename T>
static void unary_op_silu_kernel(const T * x, T * dst, const int k, const sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_silu(x[i]);
    }
}

template<typename T>
static void unary_op_gelu_quick_kernel(const T * x, T * dst, const int k, const sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_gelu_quick(x[i]);
    }
}

template<typename T>
static void unary_op_gelu_erf_kernel(const T * x, T * dst, const int k, const sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_gelu_erf(x[i]);
    }
}

template<typename T>
static void unary_op_tanh_kernel(const T * x, T * dst, const int k, const sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_tanh(x[i]);
    }
}

template<typename T>
static void unary_op_relu_kernel(const T * x, T * dst, const int k, const sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_relu(x[i]);
    }
}

template<typename T>
static void unary_op_sigmoid_kernel(const T * x, T * dst, const int k, const sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_sigmoid(x[i]);
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
static void unary_op_hardsigmoid_kernel(const T * x, T * dst, const int k, const sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_hardsigmoid(x[i]);
    }
}

template<typename T>
static void unary_op_hardswish_kernel(const T * x, T * dst, const int k, const sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_hardswish(x[i]);
    }
}

template<typename T>
static void unary_op_exp_kernel(const T * x, T * dst, const int k, const sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_exp(x[i]);
    }
}

template<typename T>
static void unary_op_log_kernel(const T * x, T * dst, const int k, const sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_log(x[i]);
    }
}

template<typename T>
static void unary_op_neg_kernel(const T * x, T * dst, const int k, const sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_neg(x[i]);
    }
}

template<typename T>
static void unary_op_step_kernel(const T * x, T * dst, const int k, const sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_step(x[i]);
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

template<typename  T>
static void upscale(const T  *x, T *dst, const int nb00, const int nb01,
                        const int nb02, const int nb03, const int ne10, const int ne11,
                        const int ne12, const int ne13, const float sf0, const float sf1,
                        const float sf2, const float sf3, const sycl::nd_item<1> &item_ct1) {
    int index = item_ct1.get_local_id(0) +
               item_ct1.get_group(0) * item_ct1.get_local_range(0);
    if (index >= ne10 * ne11 * ne12 * ne13) {
        return;
    }
    // operation
    int i10 = index % ne10;
    int i11 = (index / ne10) % ne11;
    int i12 = (index / (ne10 * ne11)) % ne12;
    int i13 = (index / (ne10 * ne11 * ne12)) % ne13;

    int i00 = static_cast<int>(i10 / sf0);
    int i01 = static_cast<int>(i11 / sf1);
    int i02 = static_cast<int>(i12 / sf2);
    int i03 = static_cast<int>(i13 / sf3);

    dst[index] = *(const T *)((const char *)x + i03 * nb03 + i02 * nb02 + i01 * nb01 + i00 * nb00);
}

template <typename T>
static void pad(const T  *x, T *dst, const int ne0, const int ne00, const int ne01, const int ne02,
                    const sycl::nd_item<3> &item_ct1) {
    int nidx = SYCL_LOCAL_ID_CALC(item_ct1, 2);
    if (nidx >= ne0) {
        return;
    }

    // operation
    int offset_dst = nidx + item_ct1.get_group(1) * ne0 +
                     item_ct1.get_group(0) * ne0 * item_ct1.get_group_range(1);
    if (nidx < ne00 && item_ct1.get_group(1) < (size_t) ne01 && item_ct1.get_group(0) < (size_t) ne02) {
        int offset_src = nidx + item_ct1.get_group(1) * ne00 +
                         item_ct1.get_group(0) * ne00 * ne01;
            dst[offset_dst] = x[offset_src];
    } else {
        dst[offset_dst] = static_cast<T>(0.0f);
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

namespace ggml_sycl_detail {
static void acc_f32_sycl(const float *x, const float *y, float *dst,
                         const int n_elements, const int ne10, const int ne11,
                         const int ne12, const int nb1, const int nb2,
                         const int offset, queue_ptr stream) {
    int num_blocks = ceil_div(n_elements, SYCL_ACC_BLOCK_SIZE);
    sycl_parallel_for(stream,
        sycl::nd_range<1>(sycl::range<1>(num_blocks) *
                              sycl::range<1>(SYCL_ACC_BLOCK_SIZE),
                          sycl::range<1>(SYCL_ACC_BLOCK_SIZE)),
        [=](sycl::nd_item<1> item_ct1) {
            acc_f32(x, y, dst, n_elements, ne10, ne11, ne12, nb1, nb2, offset,
                    item_ct1);
        });
}

template<typename T>
static void upscale_sycl(const T *x, T *dst, const int nb00, const int nb01,
                             const int nb02, const int nb03, const int ne10, const int ne11,
                             const int ne12, const int ne13, const float sf0, const float sf1,
                             const float sf2, const float sf3, queue_ptr stream) {
    int dst_size = ne10 * ne11 * ne12 * ne13;
    int num_blocks = ceil_div(dst_size, SYCL_UPSCALE_BLOCK_SIZE);
    sycl::range<1> gridDim(num_blocks * SYCL_UPSCALE_BLOCK_SIZE);
    sycl_parallel_for<1>(
        stream, sycl::nd_range<1>(gridDim, sycl::range<1>(SYCL_UPSCALE_BLOCK_SIZE)), [=](sycl::nd_item<1> item_ct1) {
            upscale(x, dst, nb00, nb01, nb02, nb03, ne10, ne11, ne12, ne13, sf0, sf1, sf2, sf3, item_ct1);
        });
}

template<typename T>
static void pad_sycl(const T *x, T *dst, const int ne00,
                         const int ne01, const int ne02, const int ne0,
                         const int ne1, const int ne2, queue_ptr stream) {
    int num_blocks = ceil_div(ne0, SYCL_PAD_BLOCK_SIZE);
    sycl::range<3> gridDim(ne2, ne1, num_blocks);
    sycl_parallel_for(stream,
                      sycl::nd_range<3>(gridDim * sycl::range<3>(1, 1, SYCL_PAD_BLOCK_SIZE),
                                        sycl::range<3>(1, 1, SYCL_PAD_BLOCK_SIZE)),
                      [=](sycl::nd_item<3> item_ct1) { pad(x, dst, ne0, ne00, ne01, ne02, item_ct1); });
}

template<typename KernelInvoker, typename... Args>
static inline void dispatch_ggml_sycl_op_unary(ggml_backend_sycl_context & ctx, ggml_tensor * dst, KernelInvoker kernel_invoker, Args&&... args) {
#if defined (GGML_SYCL_F16)
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
#else
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
#endif
    GGML_ASSERT(dst->src[0]->type == dst->type);
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    switch (dst->type) {
#if defined (GGML_SYCL_F16)
        case GGML_TYPE_F16:
            {
                auto data_pts = cast_data<sycl::half>(dst);
                kernel_invoker(data_pts.src, data_pts.dst, (int)ggml_nelements(dst->src[0]), main_stream, std::forward<Args>(args)...);
                break;
            }
#endif
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
#if defined (GGML_SYCL_F16)
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
#else
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
#endif
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
#if defined (GGML_SYCL_F16)
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
#endif
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

template<typename KernelInvoker, typename... Args>
static inline void dispatch_ggml_sycl_op_upscale(ggml_backend_sycl_context & ctx, ggml_tensor * dst, KernelInvoker kernel_invoker, Args&&... args) {
#if defined (GGML_SYCL_F16)
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
#else
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
#endif
    GGML_ASSERT(dst->src[0]->type == dst->type);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    const float sf0 = (float) dst->ne[0] / dst->src[0]->ne[0];
    const float sf1 = (float) dst->ne[1] / dst->src[0]->ne[1];
    const float sf2 = (float) dst->ne[2] / dst->src[0]->ne[2];
    const float sf3 = (float) dst->ne[3] / dst->src[0]->ne[3];
    switch (dst->type) {
#if defined (GGML_SYCL_F16)
        case GGML_TYPE_F16:
            {
                auto data_pts = cast_data<sycl::half>(dst);
                kernel_invoker(data_pts.src, data_pts.dst, (int)dst->src[0]->nb[0], (int)dst->src[0]->nb[1], (int)dst->src[0]->nb[2],
                               (int)dst->src[0]->nb[3], (int)dst->ne[0], (int)dst->ne[1], (int)dst->ne[2], (int)dst->ne[3], sf0, sf1, sf2, sf3,
                               main_stream, std::forward<Args>(args)...);
                break;
            }
#endif
        case GGML_TYPE_F32:
            {
                auto data_pts = cast_data<float>(dst);
                kernel_invoker(data_pts.src, data_pts.dst, (int)dst->src[0]->nb[0], (int)dst->src[0]->nb[1], (int)dst->src[0]->nb[2],
                               (int)dst->src[0]->nb[3], (int)dst->ne[0], (int)dst->ne[1], (int)dst->ne[2], (int)dst->ne[3], sf0, sf1, sf2, sf3,
                               main_stream, std::forward<Args>(args)...);
                break;
            }
        default:
            GGML_ABORT("GGML tensor type not supported!\n");
    }
}

template<typename KernelInvoker, typename... Args>
static inline void dispatch_ggml_sycl_op_pad(ggml_backend_sycl_context & ctx, ggml_tensor * dst, KernelInvoker kernel_invoker, Args&&... args) {
#if defined (GGML_SYCL_F16)
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
#else
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
#endif
    GGML_ASSERT(dst->src[0]->type == dst->type);
    GGML_ASSERT(dst->src[0]->ne[3] == 1 && dst->ne[3] == 1); // just 3D tensors
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    switch (dst->type) {
#if defined (GGML_SYCL_F16)
        case GGML_TYPE_F16:
            {
                auto data_pts = cast_data<sycl::half>(dst);
                kernel_invoker(data_pts.src, data_pts.dst, (int)dst->src[0]->ne[0], (int)dst->src[0]->ne[1], (int)dst->src[0]->ne[2], (int)dst->ne[0],
                               (int)dst->ne[1], (int)dst->ne[2], main_stream, std::forward<Args>(args)...);
                break;
            }
#endif
        case GGML_TYPE_F32:
            {
                auto data_pts = cast_data<float>(dst);
                kernel_invoker(data_pts.src, data_pts.dst, (int)dst->src[0]->ne[0], (int)dst->src[0]->ne[1], (int)dst->src[0]->ne[2], (int)dst->ne[0],
                               (int)dst->ne[1], (int)dst->ne[2], main_stream, std::forward<Args>(args)...);
                break;
            }
        default:
            GGML_ABORT("GGML tensor type not supported!\n");
    }
}

} // namespace ggml_sycl_detail



static inline void ggml_sycl_op_sgn(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](const auto* src, auto* dst_ptr, int k_elements, queue_ptr stream) {
            const int num_blocks = ceil_div(k_elements, 256);
            sycl_parallel_for(stream,
                sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(256),
                                  sycl::range<1>(256)),
                [=](sycl::nd_item<1> item_ct1) {
                    unary_op_sgn_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

static inline void ggml_sycl_op_abs(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](const auto* src, auto* dst_ptr, int k_elements, queue_ptr stream) {
            const int num_blocks = ceil_div(k_elements, 256);
            sycl_parallel_for(stream,
                sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(256),
                                  sycl::range<1>(256)),
                [=](sycl::nd_item<1> item_ct1) {
                    unary_op_abs_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

static inline void ggml_sycl_op_elu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](const auto* src, auto* dst_ptr, int k_elements, queue_ptr stream) {
            const int num_blocks = ceil_div(k_elements, 256);
            sycl_parallel_for(stream,
                sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(256),
                                  sycl::range<1>(256)),
                [=](sycl::nd_item<1> item_ct1) {
                    unary_op_elu_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

static inline void ggml_sycl_op_silu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](const auto* src, auto* dst_ptr, int k_elements, queue_ptr stream) {
            const int num_blocks = ceil_div(k_elements, SYCL_SILU_BLOCK_SIZE);
            sycl_parallel_for(stream,
                sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(SYCL_SILU_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_SILU_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) {
                    unary_op_silu_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

static inline void ggml_sycl_op_gelu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](const auto* src, auto* dst_ptr, int k_elements, queue_ptr stream) {
            const int num_blocks = ceil_div(k_elements, SYCL_GELU_BLOCK_SIZE);
            sycl_parallel_for(stream,
                sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(SYCL_GELU_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_GELU_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) {
                    unary_op_gelu_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

static inline void ggml_sycl_op_gelu_quick(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](const auto* src, auto* dst_ptr, int k_elements, queue_ptr stream) {
            const int num_blocks = ceil_div(k_elements, SYCL_GELU_BLOCK_SIZE);
            sycl_parallel_for(stream,
                sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(SYCL_GELU_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_GELU_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) {
                    unary_op_gelu_quick_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

static inline void ggml_sycl_op_gelu_erf(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](const auto* src, auto* dst_ptr, int k_elements, queue_ptr stream) {
            const int num_blocks = ceil_div(k_elements, SYCL_GELU_BLOCK_SIZE);
            sycl_parallel_for(stream,
                sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(SYCL_GELU_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_GELU_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) {
                    unary_op_gelu_erf_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

static inline void ggml_sycl_op_tanh(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](const auto* src, auto* dst_ptr, int k_elements, queue_ptr stream) {
            const int num_blocks = ceil_div(k_elements, SYCL_TANH_BLOCK_SIZE);
            sycl_parallel_for(stream,
                sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(SYCL_TANH_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_TANH_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) {
                    unary_op_tanh_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

static inline void ggml_sycl_op_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](const auto* src, auto* dst_ptr, int k_elements, queue_ptr stream) {
            const int num_blocks = ceil_div(k_elements, SYCL_RELU_BLOCK_SIZE);
            sycl_parallel_for(stream,
                sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(SYCL_RELU_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_RELU_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) {
                    unary_op_relu_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

static inline void ggml_sycl_op_hardsigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](const auto* src, auto* dst_ptr, int k_elements, queue_ptr stream) {
            const int num_blocks = ceil_div(k_elements, SYCL_HARDSIGMOID_BLOCK_SIZE);
            sycl_parallel_for(stream,
                sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(SYCL_HARDSIGMOID_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_HARDSIGMOID_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) {
                    unary_op_hardsigmoid_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

static inline void ggml_sycl_op_hardswish(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](const auto* src, auto* dst_ptr, int k_elements, queue_ptr stream) {
            const int num_blocks = ceil_div(k_elements, SYCL_HARDSWISH_BLOCK_SIZE);
            sycl_parallel_for(stream,
                sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(SYCL_HARDSWISH_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_HARDSWISH_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) {
                    unary_op_hardswish_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

static inline void ggml_sycl_op_exp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](const auto* src, auto* dst_ptr, int k_elements, queue_ptr stream) {
            const int num_blocks = ceil_div(k_elements, SYCL_EXP_BLOCK_SIZE);
            sycl_parallel_for(stream,
                sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(SYCL_EXP_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_EXP_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) {
                    unary_op_exp_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

static inline void ggml_sycl_op_log(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](const auto* src, auto* dst_ptr, int k_elements, queue_ptr stream) {
            const int num_blocks = ceil_div(k_elements, SYCL_EXP_BLOCK_SIZE); // Using EXP block size
            sycl_parallel_for(stream,
                sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(SYCL_EXP_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_EXP_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) {
                    unary_op_log_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

static inline void ggml_sycl_op_neg(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](const auto* src, auto* dst_ptr, int k_elements, queue_ptr stream) {
            const int num_blocks = ceil_div(k_elements, SYCL_NEG_BLOCK_SIZE);
            sycl_parallel_for(stream,
                sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(SYCL_NEG_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_NEG_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) {
                    unary_op_neg_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

static inline void ggml_sycl_op_step(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](const auto* src, auto* dst_ptr, int k_elements, queue_ptr stream) {
            const int num_blocks = ceil_div(k_elements, SYCL_NEG_BLOCK_SIZE); // Using NEG block size
            sycl_parallel_for(stream,
                sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(SYCL_NEG_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_NEG_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) {
                    unary_op_step_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

static inline void ggml_sycl_op_sigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](const auto* src, auto* dst_ptr, int k_elements, queue_ptr stream) {
            const int num_blocks = ceil_div(k_elements, SYCL_SIGMOID_BLOCK_SIZE);
            sycl_parallel_for(stream,
                sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(SYCL_SIGMOID_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_SIGMOID_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) {
                    unary_op_sigmoid_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

static inline void ggml_sycl_op_sqrt(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](const auto* src, auto* dst_ptr, int k_elements, queue_ptr stream) {
            const int num_blocks = ceil_div(k_elements, SYCL_SQRT_BLOCK_SIZE);
            sycl_parallel_for(stream,
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
            sycl_parallel_for(stream,
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
            sycl_parallel_for(stream,
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
            sycl_parallel_for(stream,
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
            sycl_parallel_for(stream,
                sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(SYCL_SQR_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_SQR_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) {
                    unary_op_sqr_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

static inline void ggml_sycl_op_upscale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_upscale(ctx, dst,
        [](const auto* src, auto* dst_ptr, int nb00, int nb01, int nb02, int nb03,
           int ne10, int ne11, int ne12, int ne13, float sf0, float sf1, float sf2, float sf3,
           queue_ptr stream) {
            ggml_sycl_detail::upscale_sycl(src, dst_ptr, nb00, nb01, nb02, nb03, ne10, ne11, ne12, ne13, sf0, sf1, sf2, sf3, stream);
        });
}

static inline void ggml_sycl_op_pad(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_pad(ctx, dst,
        [](const auto* src, auto* dst_ptr, int ne00, int ne01, int ne02, int ne0, int ne1, int ne2,
           queue_ptr stream) {
            ggml_sycl_detail::pad_sycl(src, dst_ptr, ne00, ne01, ne02, ne0, ne1, ne2, stream);
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
            sycl_parallel_for(stream,
                sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(SYCL_CLAMP_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_CLAMP_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) {
                    clamp(src, dst_ptr, min_arg, max_arg, k_elements, item_ct1);
                });
        }, min_val, max_val);
}

static inline void ggml_sycl_op_acc(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->ne[3] == 1); // just 3D tensors supported
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    const float * src1_dd = static_cast<const float*>(dst->src[1]->data);
    float *       dst_dd  = static_cast<float *>(dst->data);

    int nb1 = dst->op_params[0] / 4; // 4 bytes of float32
    int nb2 = dst->op_params[1] / 4; // 4 bytes of float32
    // int nb3 = dst->op_params[2] / 4; // 4 bytes of float32 - unused
    int offset = dst->op_params[3] / 4; // offset in bytes

    ggml_sycl_detail::acc_f32_sycl(src0_dd, src1_dd, dst_dd, (int)ggml_nelements(dst), (int)dst->src[1]->ne[0], (int)dst->src[1]->ne[1], (int)dst->src[1]->ne[2], nb1, nb2, offset, main_stream);
}

static inline void ggml_sycl_op_geglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_fused_glu(ctx, dst,
        [](const auto* x_ptr, const auto* g_ptr, auto* dst_ptr, uint64_t k, uint64_t n, uint64_t o0, uint64_t o1, queue_ptr main_stream) {
            const uint32_t num_blocks = ceil_div(k, SYCL_GELU_BLOCK_SIZE);
            sycl_parallel_for(main_stream,
                    sycl::nd_range<1>((num_blocks * sycl::range<1>(SYCL_GELU_BLOCK_SIZE)), sycl::range<1>(SYCL_GELU_BLOCK_SIZE)), [=](sycl::nd_item<1> item_ct1) {
                gated_op_fused_geglu(x_ptr, g_ptr, dst_ptr, k, n, o0, o1, item_ct1);
            });
        });
}

static inline void ggml_sycl_op_reglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_fused_glu(ctx, dst,
        [](const auto* x_ptr, const auto* g_ptr, auto* dst_ptr, uint64_t k, uint64_t n, uint64_t o0, uint64_t o1, queue_ptr main_stream) {
            const uint32_t num_blocks = ceil_div((uint32_t)k, SYCL_RELU_BLOCK_SIZE); // Using RELU block size for reglu
            sycl_parallel_for(main_stream,
                    sycl::nd_range<1>((num_blocks * sycl::range<1>(SYCL_RELU_BLOCK_SIZE)), sycl::range<1>(SYCL_RELU_BLOCK_SIZE)), [=](sycl::nd_item<1> item_ct1) {
                gated_op_fused_reglu(x_ptr, g_ptr, dst_ptr, k, n, o0, o1, item_ct1);
            });
        });
}

static inline void ggml_sycl_op_swiglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_fused_glu(ctx, dst,
        [](const auto* x_ptr, const auto* g_ptr, auto* dst_ptr, uint64_t k, uint64_t n, uint64_t o0, uint64_t o1, queue_ptr main_stream) {
            const uint32_t num_blocks = ceil_div((uint32_t)k, SYCL_SILU_BLOCK_SIZE); // Using SILU block size for swiglu
            sycl_parallel_for(main_stream,
                    sycl::nd_range<1>((num_blocks * sycl::range<1>(SYCL_SILU_BLOCK_SIZE)), sycl::range<1>(SYCL_SILU_BLOCK_SIZE)), [=](sycl::nd_item<1> item_ct1) {
                gated_op_fused_swiglu(x_ptr, g_ptr, dst_ptr, k, n, o0, o1, item_ct1);
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

void ggml_sycl_upscale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_upscale(ctx, dst);
}

void ggml_sycl_pad(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_pad(ctx, dst);
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
