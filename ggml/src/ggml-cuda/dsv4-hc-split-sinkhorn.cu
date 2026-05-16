#include "dsv4-hc-split-sinkhorn.cuh"

// Maximum n_hc supported (matches CPU reference assert at ops.cpp:11014 and
// the dst comb matrix scratch buffer size below).
#define DSV4_HC_SINKHORN_MAX_N_HC 16

// One block per row. Inside the block:
//   - threads cooperate (parallel for) on the pre/post slices and the final
//     copy of the comb matrix back to dst.
//   - tid == 0 runs the n_hc x n_hc Sinkhorn iterations serially. n_hc <= 16
//     so this is at most a few thousand FLOPs per row.
//
// The comb matrix lives in shared memory (sized for the worst case 16x16
// = 256 floats = 1 KiB per block, well within any device's shared-mem
// budget).
static __global__ void dsv4_hc_split_sinkhorn_f32(
        const float * __restrict__ mixes,
        const float * __restrict__ scale,
        const float * __restrict__ base,
        float       * __restrict__ dst,
        const int   n_hc,
        const int   sinkhorn_iters,
        const int   n_rows,
        const int   mix_hc,
        const int   nb01,    // input  row stride in bytes
        const int   nb1,     // output row stride in bytes
        const float eps) {
    const int row = blockIdx.x;
    if (row >= n_rows) {
        return;
    }

    const int tid   = threadIdx.x;
    const int blksz = blockDim.x;

    const float pre_scale  = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];

    const float * row_in  = (const float *) ((const char *) mixes + row * nb01);
    float       * row_out = (float *)       ((char *)       dst   + row * nb1);

    // ---------------- Section 1: pre slice ----------------
    // out[i] = sigmoid(mix[i] * pre_scale + base[i]) + eps
    for (int i = tid; i < n_hc; i += blksz) {
        const float z = row_in[i] * pre_scale + base[i];
        row_out[i] = 1.0f / (1.0f + expf(-z)) + eps;
    }

    // ---------------- Section 2: post slice ----------------
    // out[n_hc + i] = 2 * sigmoid(mix[n_hc + i] * post_scale + base[n_hc + i])
    for (int i = tid; i < n_hc; i += blksz) {
        const int off = n_hc + i;
        const float z = row_in[off] * post_scale + base[off];
        row_out[off] = 2.0f / (1.0f + expf(-z));
    }

    // ---------------- Section 3: comb matrix Sinkhorn ----------------
    //
    // c[src_hc + dst_hc * n_hc] layout (matches CPU reference at
    // ggml-cpu/ops.cpp:11055).
    extern __shared__ float shmem[];
    float * c = shmem;            // n_hc * n_hc floats

    // Load the comb logits = mix * comb_scale + base (parallel over the block).
    for (int i = tid; i < n_hc * n_hc; i += blksz) {
        const int off = 2 * n_hc + i;
        c[i] = row_in[off] * comb_scale + base[off];
    }
    __syncthreads();

    // Sinkhorn iterations run on thread 0; n_hc <= 16 keeps the inner loops
    // trivially cheap (~ 1k FLOPs per row total).
    if (tid == 0) {
        // First pass: per-dst_hc softmax (max-subtract for numerical stability,
        // exp, normalize) + eps stabilizer.
        for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
            float row_max = -INFINITY;
            for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
                row_max = fmaxf(row_max, c[src_hc + dst_hc * n_hc]);
            }

            float row_sum = 0.0f;
            for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
                const int idx = src_hc + dst_hc * n_hc;
                const float v = expf(c[idx] - row_max);
                c[idx] = v;
                row_sum += v;
            }

            const float inv_sum = 1.0f / row_sum;
            for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
                const int idx = src_hc + dst_hc * n_hc;
                c[idx] = c[idx] * inv_sum + eps;
            }
        }

        // First column-normalize: per src_hc, divide by (column sum + eps).
        for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
            float sum = 0.0f;
            for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
                sum += c[src_hc + dst_hc * n_hc];
            }
            const float inv_denom = 1.0f / (sum + eps);
            for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
                c[src_hc + dst_hc * n_hc] *= inv_denom;
            }
        }

        // Remaining sinkhorn_iters - 1 alternations: row-normalize then column-normalize.
        for (int it = 1; it < sinkhorn_iters; ++it) {
            // Row-normalize: per dst_hc, divide by (row sum + eps).
            for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
                float sum = 0.0f;
                for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
                    sum += c[src_hc + dst_hc * n_hc];
                }
                const float inv_denom = 1.0f / (sum + eps);
                for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
                    c[src_hc + dst_hc * n_hc] *= inv_denom;
                }
            }
            // Column-normalize: per src_hc, divide by (column sum + eps).
            for (int src_hc = 0; src_hc < n_hc; ++src_hc) {
                float sum = 0.0f;
                for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
                    sum += c[src_hc + dst_hc * n_hc];
                }
                const float inv_denom = 1.0f / (sum + eps);
                for (int dst_hc = 0; dst_hc < n_hc; ++dst_hc) {
                    c[src_hc + dst_hc * n_hc] *= inv_denom;
                }
            }
        }
    }
    __syncthreads();

    // Copy the comb matrix back to dst (parallel over the block).
    for (int i = tid; i < n_hc * n_hc; i += blksz) {
        row_out[2 * n_hc + i] = c[i];
    }

    // Suppress unused-warning for mix_hc; it's covered by the host-side asserts.
    (void) mix_hc;
}

void ggml_cuda_op_dsv4_hc_split_sinkhorn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * mixes = dst->src[0];
    const ggml_tensor * scale = dst->src[1];
    const ggml_tensor * base  = dst->src[2];

    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type  == GGML_TYPE_F32);
    GGML_ASSERT(dst->type   == GGML_TYPE_F32);
    GGML_ASSERT(mixes->nb[0] == sizeof(float));
    GGML_ASSERT(scale->nb[0] == sizeof(float));
    GGML_ASSERT(base->nb[0]  == sizeof(float));
    GGML_ASSERT(dst->nb[0]   == sizeof(float));

    const int   n_hc           = ggml_get_op_params_i32(dst, 0);
    const int   sinkhorn_iters = ggml_get_op_params_i32(dst, 1);
    const float eps            = ggml_get_op_params_f32(dst, 2);

    GGML_ASSERT(n_hc > 0 && n_hc <= DSV4_HC_SINKHORN_MAX_N_HC);
    GGML_ASSERT(sinkhorn_iters > 0);

    const int n_rows = (int) ggml_nrows(mixes);
    const int mix_hc = (int) mixes->ne[0];
    const int nb01   = (int) mixes->nb[1];
    const int nb1    = (int) dst->nb[1];

    GGML_ASSERT(mix_hc == (2 + n_hc) * n_hc);
    GGML_ASSERT((int) ggml_nrows(dst) == n_rows);

    // Block size MUST be a warp multiple (>= 32) so that the in-block
    // __syncthreads() barriers are well-formed and any future warp-wide
    // shuffle has a complete mask. With mix_hc in {24, 80} the natural
    // size is rounded up to 32 or 96.
    constexpr int CUDA_WARP_SIZE = 32;
    constexpr int CUDA_MAX_BLOCK = 256;
    const int rounded = ((mix_hc + CUDA_WARP_SIZE - 1) / CUDA_WARP_SIZE) * CUDA_WARP_SIZE;
    const int threads_per_block = std::min(CUDA_MAX_BLOCK, std::max(CUDA_WARP_SIZE, rounded));

    const dim3 grid(n_rows);
    const dim3 block(threads_per_block);
    const size_t shared = (size_t) n_hc * (size_t) n_hc * sizeof(float);

    cudaStream_t stream = ctx.stream();
    dsv4_hc_split_sinkhorn_f32<<<grid, block, shared, stream>>>(
        (const float *) mixes->data,
        (const float *) scale->data,
        (const float *) base->data,
        (float *)       dst->data,
        n_hc, sinkhorn_iters, n_rows, mix_hc,
        nb01, nb1, eps);
    CUDA_CHECK(cudaGetLastError());
}
