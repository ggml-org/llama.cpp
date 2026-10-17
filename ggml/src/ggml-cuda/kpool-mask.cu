#include "kpool-mask.cuh"
#include "convert.cuh"

#define CUDA_KPOOL_MASK_BLOCK_SIZE 512

// one block per token: fill the row with -inf, then copy the mask cells of the live picked pools and the tail
template <typename T>
static __global__ void kpool_mask_kernel(
        const T * mask, const float * score, const int32_t * top_k, const int32_t * pools, const int32_t * tail, T * dst,
        const int64_t n_kv, const int64_t ne1, const int64_t ne2,
        const int64_t n_pool, const int64_t kpool, const int64_t n_top, const int64_t n_tail,
        const int64_t sm1, const int64_t sm2, const int64_t sm3,
        const int64_t sd1, const int64_t sd2, const int64_t sd3,
        const int64_t ss0, const int64_t ss1,
        const int64_t sk0, const int64_t sk1,
        const int64_t sp0, const int64_t sp1,
        const int64_t st0, const int64_t st1,
        const bool vec) {
    ggml_cuda_pdl_lc();

    const int64_t t  = blockIdx.x;
    const int64_t i1 = t % ne1;
    const int64_t i2 = (t / ne1) % ne2;
    const int64_t i3 = t / (ne1*ne2);

    const T * mask_row = mask + i1*sm1 + i2*sm2 + i3*sm3;
    T       * dst_row  = dst  + i1*sd1 + i2*sd2 + i3*sd3;

    ggml_cuda_pdl_sync();

    const T ninf = ggml_cuda_cast<T>(-INFINITY);
    if (vec) {
        constexpr int nv = sizeof(int4)/sizeof(T);
        T fill[nv];
#pragma unroll
        for (int j = 0; j < nv; ++j) {
            fill[j] = ninf;
        }
        const int4 fill4 = *(const int4 *) fill;
        for (int64_t i = threadIdx.x; i < n_kv/nv; i += blockDim.x) {
            ((int4 *) dst_row)[i] = fill4;
        }
    } else {
        for (int64_t i = threadIdx.x; i < n_kv; i += blockDim.x) {
            dst_row[i] = ninf;
        }
    }

    __syncthreads();

    // one thread per picked pool or tail cell; cells picked twice get the same value
    for (int64_t i = threadIdx.x; i < n_top + n_tail; i += blockDim.x) {
        if (i < n_top) {
            const int32_t p = top_k[i*sk0 + t*sk1];
            if (p < 0 || p >= n_pool || !(score[p*ss0 + t*ss1] > -INFINITY)) {
                continue;
            }
            for (int64_t k = 0; k < kpool; ++k) {
                const int32_t c = pools[k*sp0 + p*sp1];
                if (c >= 0 && c < n_kv) {
                    dst_row[c] = mask_row[c];
                }
            }
        } else {
            const int32_t c = tail[(i - n_top)*st0 + t*st1];
            if (c >= 0 && c < n_kv) {
                dst_row[c] = mask_row[c];
            }
        }
    }
}

template <typename T>
static void kpool_mask_cuda(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * mask  = dst->src[0];
    const ggml_tensor * score = dst->src[1];
    const ggml_tensor * top_k = dst->src[2];
    const ggml_tensor * pools = dst->src[3];
    const ggml_tensor * tail  = dst->src[4];

    const int64_t n_kv     = mask->ne[0];
    const int64_t n_tokens = mask->ne[1]*mask->ne[2]*mask->ne[3];

    GGML_ASSERT(dst->nb[0] == sizeof(T) && mask->nb[0] == sizeof(T));
    GGML_ASSERT(dst->nb[1] % sizeof(T) == 0 && mask->nb[1] % sizeof(T) == 0);

    // 16-byte stores need every dst row aligned
    const bool vec = (n_kv*sizeof(T)) % sizeof(int4) == 0 &&
        dst->nb[1] % sizeof(int4) == 0 && dst->nb[2] % sizeof(int4) == 0 && dst->nb[3] % sizeof(int4) == 0 &&
        (uintptr_t) dst->data % sizeof(int4) == 0;

    const int32_t * tail_d = tail ? (const int32_t *) tail->data : nullptr;

    const dim3 block_dims(CUDA_KPOOL_MASK_BLOCK_SIZE, 1, 1);
    const dim3 grid_dims(n_tokens, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(kpool_mask_kernel<T>, launch_params,
            (const T *) mask->data, (const float *) score->data, (const int32_t *) top_k->data, (const int32_t *) pools->data,
            tail_d, (T *) dst->data,
            n_kv, mask->ne[1], mask->ne[2],
            pools->ne[1], pools->ne[0], top_k->ne[0], tail ? tail->ne[0] : 0,
            mask->nb[1]/sizeof(T), mask->nb[2]/sizeof(T), mask->nb[3]/sizeof(T),
            dst->nb[1]/sizeof(T),  dst->nb[2]/sizeof(T),  dst->nb[3]/sizeof(T),
            score->nb[0]/sizeof(float), score->nb[1]/sizeof(float),
            top_k->nb[0]/sizeof(int32_t), top_k->nb[1]/sizeof(int32_t),
            pools->nb[0]/sizeof(int32_t), pools->nb[1]/sizeof(int32_t),
            tail ? tail->nb[0]/sizeof(int32_t) : 0, tail ? tail->nb[1]/sizeof(int32_t) : 0,
            vec);
}

void ggml_cuda_op_kpool_mask(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    switch (dst->type) {
        case GGML_TYPE_F16:
            kpool_mask_cuda<half>(ctx, dst);
            break;
        case GGML_TYPE_F32:
            kpool_mask_cuda<float>(ctx, dst);
            break;
        default:
            GGML_ABORT("unsupported type");
    }
}
