#include "common.hpp"
#include "ggml-sycl/presets.hpp"
#include "ggml.h"
#include "element_wise.hpp"

// --- Helper Macros for Kernel Indexing ---
#define SYCL_GLOBAL_ID_LOOP(K, ITEM) \
    for (auto i = ITEM.get_global_id(0); i < (size_t)K; i += ITEM.get_global_range(0))

#define SYCL_LOCAL_ID_CALC(ITEM, IDX) \
    (ITEM.get_local_range(IDX) * ITEM.get_group(IDX) + ITEM.get_local_id(IDX))

// --- Original Kernels (non-_sycl) - Modified to use indexing macros and cast literals ---

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

template<typename T>
static void sgn(const T * x, T * dst, const int k, const sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = x[i] > static_cast<T>(0.f) ? static_cast<T>(1.f) : ((x[i] < static_cast<T>(0.f) ? static_cast<T>(-1.f) : static_cast<T>(0.f)));
    }
}

template<typename T>
static void abs_op(const T * x, T * dst, const int k, const sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = sycl::fabs(x[i]);
    }
}

template<typename T>
static void elu_op(const T * x, T * dst, const int k, const sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = (x[i] > static_cast<T>(0.f)) ? x[i] : sycl::expm1(x[i]);
    }
}

template<typename T>
static void gelu(const T * x, T * dst, const int k,
                     const sycl::nd_item<1> &item_ct1) {
    const T GELU_COEF_A    = static_cast<T>(0.044715f);
    const T SQRT_2_OVER_PI = static_cast<T>(0.79788456080286535587989211986876f);
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = static_cast<T>(0.5f) * x[i] *
                 (static_cast<T>(1.0f) +
                  sycl::tanh(SQRT_2_OVER_PI * x[i] * (static_cast<T>(1.0f) + GELU_COEF_A * x[i] * x[i])));
    }
}

template<typename T>
static void silu(const T * x, T * dst, const int k,
                     const sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = x[i] / (static_cast<T>(1.0f) + sycl::native::exp(-x[i]));
    }
}

template<typename T>
static void gelu_quick(const T *x, T *dst, int k,
                           const sycl::nd_item<1> &item_ct1) {
    const T GELU_QUICK_COEF_LOCAL = static_cast<T>(-1.702f);
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = x[i] * (static_cast<T>(1.0f) / (static_cast<T>(1.0f) + sycl::native::exp(GELU_QUICK_COEF_LOCAL * x[i])));
    }
}

template<typename T>
static void gelu_erf(const T * x, T * dst, const int k, const sycl::nd_item<1> &item_ct1) {
    const T SQRT_2_INV = static_cast<T>(0.70710678118654752440084436210484f);
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
       auto x_i = x[i];
        dst[i] = static_cast<T>(0.5f) * x_i * (static_cast<T>(1.0f) + sycl::erf(x_i * SQRT_2_INV));
    }
}

template<typename T>
static void tanh(const T *x, T *dst, int k,
                     const sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = sycl::tanh((x[i]));
    }
}

template<typename T>
static void relu(const T * x, T * dst, const int k,
                     const sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = sycl::fmax((x[i]), static_cast<T>(0));
    }
}

template<typename T>
static void sigmoid(const T * x, T * dst, const int k,
                            const sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = static_cast<T>(1.0f) / (static_cast<T>(1.0f) + sycl::native::exp(-x[i]));
    }
}

template<typename T>
static void sqrt(const T * x, T * dst, const int k,
                            const sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = sycl::sqrt(x[i]);
    }
}

template<typename T>
static void sin(const T * x, T * dst, const int k,
                            const sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = sycl::sin(x[i]);
    }
}

template<typename T>
static void cos(const T * x, T * dst, const int k,
                            const sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = sycl::cos(x[i]);
    }
}

template<typename T>
static void hardsigmoid(const T * x, T * dst, const int k,
                            const sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = sycl::fmin(static_cast<T>(1.0f), sycl::fmax(static_cast<T>(0.0f), (x[i] + static_cast<T>(3.0f)) / static_cast<T>(6.0f)));
    }
}

template<typename T>
static void hardswish(const T * x, T * dst, const int k,
                          const sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = x[i] * sycl::fmin(static_cast<T>(1.0f), sycl::fmax(static_cast<T>(0.0f), (x[i] + static_cast<T>(3.0f)) / static_cast<T>(6.0f)));
    }
}

template<typename T>
static void exp(const T * x, T * dst, const int k,
                          const sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = sycl::exp(x[i]);
    }
}

template<typename T>
static void log(const T * x, T * dst, const int k,
                          const sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        T xi = x[i];
        if (xi <= static_cast<T>(0)) {
            dst[i] = neg_infinity<T>();
        } else {
            dst[i] = sycl::log(xi);
        }
    }
}

template<typename T>
static void neg(const T * x, T * dst, const int k,
                          const sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = -x[i];
    }
}

template<typename T>
static void step(const T * x, T * dst, const int k,
                          const sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = (x[i] > static_cast<T>(0.0f)) ? static_cast<T>(1.0f) : static_cast<T>(0.0f);
    }
}

template<typename T>
static void leaky_relu(const T *x, T *dst, const int k, const float negative_slope,
                           const sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        T neg_slope_T = static_cast<T>(negative_slope);
        dst[i] = sycl::fmax((x[i]), static_cast<T>(0)) +
                 sycl::fmin((x[i]), static_cast<T>(0.0f)) * neg_slope_T;
    }
}

template<typename T>
static void sqr(const T * x, T * dst, const int k,
                    const sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = x[i] * x[i];
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

// Fused GLU kernels (unchanged logic)
template<typename T>
static void gated_op_fused_geglu(const T * x, const T * g, T * dst, const uint64_t k, const uint64_t n, const uint64_t o0, const uint64_t o1, const sycl::nd_item<1> &item_ct1) {
    const T GELU_QUICK_COEF_LOCAL = static_cast<T>(-1.702f);
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        const int64_t j0 = (i / n) * o0 + (i % n);
        const int64_t j1 = o0 == o1 ? j0 : (i / n) * o1 + (i % n);
        const T x_val = x[j0];
        const T gelu_val = x_val * (static_cast<T>(1.0f) / (static_cast<T>(1.0f) + sycl::native::exp(GELU_QUICK_COEF_LOCAL * x_val)));

        dst[i] = gelu_val * g[j1];
    }
}

template<typename T>
static void gated_op_fused_reglu(const T * x, const T * g, T * dst, const uint64_t k, const uint64_t n, const uint64_t o0, const uint64_t o1, const sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        const int64_t j0 = (i / n) * o0 + (i % n);
        const int64_t j1 = o0 == o1 ? j0 : (i / n) * o1 + (i % n);
        dst[i] = sycl::max((x[j0]), static_cast<T>(0)) * g[j1];
    }
}

template<typename T>
static void gated_op_fused_swiglu(const T * x, const T * g, T * dst, const uint64_t k, const uint64_t n, const uint64_t o0, const uint64_t o1, const sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1)  {
        const int64_t j0 = (i / n) * o0 + (i % n);
        const int64_t j1 = o0 == o1 ? j0 : (i / n) * o1 + (i % n);
        dst[i] = (x[j0] / (static_cast<T>(1) + sycl::native::exp(-x[j0]))) * g[j1];
    }
}

// --- Generic SYCL Kernel Launchers ---
namespace ggml_sycl_detail {
// acc_f32_sycl remains specific
static void acc_f32_sycl(const float *x, const float *y, float *dst,
                         const int n_elements, const int ne10, const int ne11,
                         const int ne12, const int nb1, const int nb2,
                         const int offset, queue_ptr stream) {
    int num_blocks = ceil_div(n_elements, SYCL_ACC_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(num_blocks) *
                              sycl::range<1>(SYCL_ACC_BLOCK_SIZE),
                          sycl::range<1>(SYCL_ACC_BLOCK_SIZE)),
        [=](sycl::nd_item<1> item_ct1) {
            acc_f32(x, y, dst, n_elements, ne10, ne11, ne12, nb1, nb2, offset,
                    item_ct1);
        });
}

// upscale_sycl remains specific
template<typename T>
static void upscale_sycl(const T *x, T *dst, const int nb00, const int nb01,
                             const int nb02, const int nb03, const int ne10, const int ne11,
                             const int ne12, const int ne13, const float sf0, const float sf1,
                             const float sf2, const float sf3, queue_ptr stream) {
    int dst_size = ne10 * ne11 * ne12 * ne13;
    int num_blocks = ceil_div(dst_size, SYCL_UPSCALE_BLOCK_SIZE);
    sycl::range<1> gridDim(num_blocks * SYCL_UPSCALE_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<1>(gridDim, sycl::range<1>(SYCL_UPSCALE_BLOCK_SIZE)),
        [=](sycl::nd_item<1> item_ct1) {
            upscale(x, dst, nb00, nb01, nb02, nb03, ne10, ne11, ne12, ne13, sf0, sf1, sf2, sf3, item_ct1);
        });
}

// pad_sycl remains specific
template<typename T>
static void pad_sycl(const T *x, T *dst, const int ne00,
                         const int ne01, const int ne02, const int ne0,
                         const int ne1, const int ne2, queue_ptr stream) {
    int num_blocks = ceil_div(ne0, SYCL_PAD_BLOCK_SIZE);
    sycl::range<3> gridDim(ne2, ne1, num_blocks);
    stream->parallel_for(
        sycl::nd_range<3>(gridDim * sycl::range<3>(1, 1, SYCL_PAD_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_PAD_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            pad(x, dst, ne0, ne00, ne01, ne02, item_ct1);
        });
}

// Common dispatcher for 1-input, 1-output element-wise ops, handling type switching.
// KernelInvoker is a lambda that takes (const T* src, T* dst, int k, queue_ptr stream, Args...)
template<typename KernelInvoker, typename... Args>
inline void dispatch_ggml_sycl_op_unary(ggml_backend_sycl_context & ctx, ggml_tensor * dst, KernelInvoker kernel_invoker, Args&&... args) {
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

// Dispatcher for fused GLU ops, handling specific input pointer setup and type switching.
template<typename KernelInvoker, typename... Args>
inline void dispatch_ggml_sycl_op_fused_glu(ggml_backend_sycl_context & ctx, ggml_tensor * dst, KernelInvoker kernel_invoker, Args&&... args) {
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

// Dispatcher for upscale
template<typename KernelInvoker, typename... Args>
inline void dispatch_ggml_sycl_op_upscale(ggml_backend_sycl_context & ctx, ggml_tensor * dst, KernelInvoker kernel_invoker, Args&&... args) {
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

// Dispatcher for pad
template<typename KernelInvoker, typename... Args>
inline void dispatch_ggml_sycl_op_pad(ggml_backend_sycl_context & ctx, ggml_tensor * dst, KernelInvoker kernel_invoker, Args&&... args) {
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


// --- Backend Operation Functions (ggml_sycl_op_...) ---

inline void ggml_sycl_op_sgn(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](const auto* src, auto* dst_ptr, int k_elements, queue_ptr stream) {
            const int num_blocks = ceil_div(k_elements, 256);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(256),
                                  sycl::range<1>(256)),
                [=](sycl::nd_item<1> item_ct1) {
                    sgn(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

inline void ggml_sycl_op_abs(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](const auto* src, auto* dst_ptr, int k_elements, queue_ptr stream) {
            const int num_blocks = ceil_div(k_elements, 256);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(256),
                                  sycl::range<1>(256)),
                [=](sycl::nd_item<1> item_ct1) {
                    abs_op(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

inline void ggml_sycl_op_elu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](const auto* src, auto* dst_ptr, int k_elements, queue_ptr stream) {
            const int num_blocks = ceil_div(k_elements, 256);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(256),
                                  sycl::range<1>(256)),
                [=](sycl::nd_item<1> item_ct1) {
                    elu_op(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

inline void ggml_sycl_op_silu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](const auto* src, auto* dst_ptr, int k_elements, queue_ptr stream) {
            const int num_blocks = ceil_div(k_elements, SYCL_SILU_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(SYCL_SILU_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_SILU_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) {
                    silu(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

inline void ggml_sycl_op_gelu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](const auto* src, auto* dst_ptr, int k_elements, queue_ptr stream) {
            const int num_blocks = ceil_div(k_elements, SYCL_GELU_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(SYCL_GELU_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_GELU_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) {
                    gelu(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

inline void ggml_sycl_op_gelu_quick(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](const auto* src, auto* dst_ptr, int k_elements, queue_ptr stream) {
            const int num_blocks = ceil_div(k_elements, SYCL_GELU_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(SYCL_GELU_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_GELU_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) {
                    gelu_quick(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

inline void ggml_sycl_op_gelu_erf(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](const auto* src, auto* dst_ptr, int k_elements, queue_ptr stream) {
            const int num_blocks = ceil_div(k_elements, SYCL_GELU_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(SYCL_GELU_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_GELU_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) {
                    gelu_erf(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

inline void ggml_sycl_op_tanh(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](const auto* src, auto* dst_ptr, int k_elements, queue_ptr stream) {
            const int num_blocks = ceil_div(k_elements, SYCL_TANH_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(SYCL_TANH_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_TANH_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) {
                    tanh(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

inline void ggml_sycl_op_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](const auto* src, auto* dst_ptr, int k_elements, queue_ptr stream) {
            const int num_blocks = ceil_div(k_elements, SYCL_RELU_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(SYCL_RELU_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_RELU_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) {
                    relu(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

inline void ggml_sycl_op_hardsigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](const auto* src, auto* dst_ptr, int k_elements, queue_ptr stream) {
            const int num_blocks = ceil_div(k_elements, SYCL_HARDSIGMOID_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(SYCL_HARDSIGMOID_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_HARDSIGMOID_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) {
                    hardsigmoid(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

inline void ggml_sycl_op_hardswish(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](const auto* src, auto* dst_ptr, int k_elements, queue_ptr stream) {
            const int num_blocks = ceil_div(k_elements, SYCL_HARDSWISH_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(SYCL_HARDSWISH_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_HARDSWISH_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) {
                    hardswish(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

inline void ggml_sycl_op_exp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](const auto* src, auto* dst_ptr, int k_elements, queue_ptr stream) {
            const int num_blocks = ceil_div(k_elements, SYCL_EXP_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(SYCL_EXP_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_EXP_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) {
                    exp(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

inline void ggml_sycl_op_log(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](const auto* src, auto* dst_ptr, int k_elements, queue_ptr stream) {
            const int num_blocks = ceil_div(k_elements, SYCL_EXP_BLOCK_SIZE); // Using EXP block size
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(SYCL_EXP_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_EXP_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) {
                    log(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

inline void ggml_sycl_op_neg(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](const auto* src, auto* dst_ptr, int k_elements, queue_ptr stream) {
            const int num_blocks = ceil_div(k_elements, SYCL_NEG_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(SYCL_NEG_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_NEG_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) {
                    neg(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

inline void ggml_sycl_op_step(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](const auto* src, auto* dst_ptr, int k_elements, queue_ptr stream) {
            const int num_blocks = ceil_div(k_elements, SYCL_NEG_BLOCK_SIZE); // Using NEG block size
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(SYCL_NEG_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_NEG_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) {
                    step(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

inline void ggml_sycl_op_sigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](const auto* src, auto* dst_ptr, int k_elements, queue_ptr stream) {
            const int num_blocks = ceil_div(k_elements, SYCL_SIGMOID_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(SYCL_SIGMOID_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_SIGMOID_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) {
                    sigmoid(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

inline void ggml_sycl_op_sqrt(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](const auto* src, auto* dst_ptr, int k_elements, queue_ptr stream) {
            const int num_blocks = ceil_div(k_elements, SYCL_SQRT_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(SYCL_SQRT_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_SQRT_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) {
                    sqrt(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

inline void ggml_sycl_op_sin(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](const auto* src, auto* dst_ptr, int k_elements, queue_ptr stream) {
            const int num_blocks = ceil_div(k_elements, SYCL_SIN_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(SYCL_SIN_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_SIN_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) {
                    sin(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

inline void ggml_sycl_op_cos(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](const auto* src, auto* dst_ptr, int k_elements, queue_ptr stream) {
            const int num_blocks = ceil_div(k_elements, SYCL_SIN_BLOCK_SIZE); // Using SIN block size
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(SYCL_SIN_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_SIN_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) {
                    cos(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

inline void ggml_sycl_op_leaky_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    float negative_slope;
    memcpy(&negative_slope, dst->op_params, sizeof(float));
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](const auto* src, auto* dst_ptr, int k_elements, queue_ptr stream, float slope) {
            const int num_blocks = ceil_div(k_elements, SYCL_RELU_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(SYCL_RELU_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_RELU_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) {
                    leaky_relu(src, dst_ptr, k_elements, slope, item_ct1);
                });
        }, negative_slope);
}

inline void ggml_sycl_op_sqr(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](const auto* src, auto* dst_ptr, int k_elements, queue_ptr stream) {
            const int num_blocks = ceil_div(k_elements, SYCL_SQR_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(SYCL_SQR_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_SQR_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) {
                    sqr(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

inline void ggml_sycl_op_upscale(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_upscale(ctx, dst,
        [](const auto* src, auto* dst_ptr, int nb00, int nb01, int nb02, int nb03,
           int ne10, int ne11, int ne12, int ne13, float sf0, float sf1, float sf2, float sf3,
           queue_ptr stream) {
            ggml_sycl_detail::upscale_sycl(src, dst_ptr, nb00, nb01, nb02, nb03, ne10, ne11, ne12, ne13, sf0, sf1, sf2, sf3, stream);
        });
}

inline void ggml_sycl_op_pad(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_pad(ctx, dst,
        [](const auto* src, auto* dst_ptr, int ne00, int ne01, int ne02, int ne0, int ne1, int ne2,
           queue_ptr stream) {
            ggml_sycl_detail::pad_sycl(src, dst_ptr, ne00, ne01, ne02, ne0, ne1, ne2, stream);
        });
}

inline void ggml_sycl_op_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
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

inline void ggml_sycl_op_acc(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
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

inline void ggml_sycl_op_geglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_fused_glu(ctx, dst,
        [](const auto* x_ptr, const auto* g_ptr, auto* dst_ptr, uint64_t k, uint64_t n, uint64_t o0, uint64_t o1, queue_ptr main_stream) {
            const uint32_t num_blocks = ceil_div(k, SYCL_GELU_BLOCK_SIZE);
            main_stream->parallel_for(
                    sycl::nd_range<1>((num_blocks * sycl::range<1>(SYCL_GELU_BLOCK_SIZE)), sycl::range<1>(SYCL_GELU_BLOCK_SIZE)), [=](sycl::nd_item<1> item_ct1) {
                gated_op_fused_geglu(x_ptr, g_ptr, dst_ptr, k, n, o0, o1, item_ct1);
            });
        });
}

inline void ggml_sycl_op_reglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_fused_glu(ctx, dst,
        [](const auto* x_ptr, const auto* g_ptr, auto* dst_ptr, uint64_t k, uint64_t n, uint64_t o0, uint64_t o1, queue_ptr main_stream) {
            const uint32_t num_blocks = ceil_div((uint32_t)k, SYCL_RELU_BLOCK_SIZE); // Using RELU block size for reglu
            main_stream->parallel_for(
                    sycl::nd_range<1>((num_blocks * sycl::range<1>(SYCL_RELU_BLOCK_SIZE)), sycl::range<1>(SYCL_RELU_BLOCK_SIZE)), [=](sycl::nd_item<1> item_ct1) {
                gated_op_fused_reglu(x_ptr, g_ptr, dst_ptr, k, n, o0, o1, item_ct1);
            });
        });
}

inline void ggml_sycl_op_swiglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_fused_glu(ctx, dst,
        [](const auto* x_ptr, const auto* g_ptr, auto* dst_ptr, uint64_t k, uint64_t n, uint64_t o0, uint64_t o1, queue_ptr main_stream) {
            const uint32_t num_blocks = ceil_div((uint32_t)k, SYCL_SILU_BLOCK_SIZE); // Using SILU block size for swiglu
            main_stream->parallel_for(
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
