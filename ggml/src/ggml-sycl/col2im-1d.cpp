#include "col2im-1d.hpp"

template <typename T>
static void col2im_1d_sycl(
        const T * col,
        T * dst,
        const int64_t T_in,
        const int64_t T_out,
        const int64_t OC,
        const int64_t K,
        const int64_t K_OC,
        const int32_t s0,
        const int32_t p0,
        dpct::queue_ptr stream) {

    const int64_t total = T_out * OC;
    const uint32_t block_size = 256;
    const uint32_t num_blocks = (uint32_t) ((total + block_size - 1) / block_size);

    stream->parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, num_blocks * block_size),
            sycl::range<3>(1, 1, block_size)),
        [=](sycl::nd_item<3> item_ct1) {
            const int64_t idx = item_ct1.get_global_id(2);
            if (idx >= total) {
                return;
            }

            const int64_t oc = idx / T_out;
            const int64_t t_out = idx - oc * T_out;
            const int64_t t_abs = t_out + p0;

            int64_t t_in_min = (t_abs - K + s0) / s0;
            if (t_in_min < 0) {
                t_in_min = 0;
            }
            int64_t t_in_max = t_abs / s0;
            if (t_in_max >= T_in) {
                t_in_max = T_in - 1;
            }

            float sum = 0.0f;
            for (int64_t t_in = t_in_min; t_in <= t_in_max; ++t_in) {
                const int64_t k = t_abs - t_in * s0;
                sum += (float) col[(oc * K + k) + t_in * K_OC];
            }

            dst[idx] = (T) sum;
        });
}

void ggml_sycl_op_col2im_1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0 != nullptr);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(src0->type == dst->type);

    const int32_t s0 = ((const int32_t *) dst->op_params)[0];
    const int32_t OC = ((const int32_t *) dst->op_params)[1];
    const int32_t p0 = ((const int32_t *) dst->op_params)[2];

    const int64_t K_OC = src0->ne[0];
    const int64_t T_in = src0->ne[1];
    const int64_t K = K_OC / OC;
    const int64_t T_out = dst->ne[0];

    GGML_ASSERT(OC > 0);
    GGML_ASSERT(K_OC % OC == 0);

    dpct::queue_ptr stream = ctx.stream();

    switch (src0->type) {
        case GGML_TYPE_F32:
            col2im_1d_sycl<float>(
                (const float *) src0->data,
                (float *) dst->data,
                T_in, T_out, OC, K, K_OC, s0, p0, stream);
            break;
        case GGML_TYPE_F16:
            col2im_1d_sycl<sycl::half>(
                (const sycl::half *) src0->data,
                (sycl::half *) dst->data,
                T_in, T_out, OC, K, K_OC, s0, p0, stream);
            break;
#ifdef GGML_SYCL_HAS_BF16
        case GGML_TYPE_BF16:
            col2im_1d_sycl<sycl::ext::oneapi::bfloat16>(
                (const sycl::ext::oneapi::bfloat16 *) src0->data,
                (sycl::ext::oneapi::bfloat16 *) dst->data,
                T_in, T_out, OC, K, K_OC, s0, p0, stream);
            break;
#endif
        default:
            GGML_ABORT("col2im_1d: unsupported type %d", src0->type);
    }
}
