#include "turbo-quant.cuh"
#include "turbo-wht.cuh"

// ─── CUDA kernel ──────────────────────────────────────────────────────────────
//
// One block per 128-element group, 128 threads per block.
// direction: 0 = forward (signs1 → WHT → signs2), 1 = inverse (signs2 → WHT → signs1)
//
// Algorithm mirrors the CPU implementation in ggml-cpu/ops.cpp:
//   1. Apply s_first elementwise
//   2. Radix-2 Hadamard butterfly (7 stages, in-place)
//   3. Normalize by 1/sqrt(128) and apply s_second elementwise

template <int direction>
static __global__ void k_turbo_wht_f32(const float * __restrict__ src,
                                        float * __restrict__ dst,
                                        int64_t n_groups) {
    const int64_t g = blockIdx.x;
    if (g >= n_groups) return;

    const int t = threadIdx.x;  // 0 .. 127

    __shared__ float x[128];

    // Load from global memory
    x[t] = src[g * 128 + t];
    __syncthreads();

    // Apply first sign array
    x[t] *= (direction == 0) ? TURBO_WHT_SIGNS1[t] : TURBO_WHT_SIGNS2[t];
    __syncthreads();

    // WHT butterfly — 7 stages for 128 elements.
    // In stage h, threads where (t % (2h)) < h read x[t] and x[t+h],
    // then write x[t] = a+b and x[t+h] = a-b.  Each active thread
    // owns a disjoint pair, so no intra-stage conflicts exist.
#define WHT_STAGE(h) \
    if (t % (2*(h)) < (h)) { float a = x[t], b = x[t+(h)]; x[t] = a+b; x[t+(h)] = a-b; } \
    __syncthreads();

    WHT_STAGE(1)
    WHT_STAGE(2)
    WHT_STAGE(4)
    WHT_STAGE(8)
    WHT_STAGE(16)
    WHT_STAGE(32)
    WHT_STAGE(64)
#undef WHT_STAGE

    // Normalize and apply second sign array, write to output
    constexpr float inv_sqrt_128 = 0.08838834764831845f;  // 1/sqrt(128)
    dst[g * 128 + t] = x[t] * inv_sqrt_128 *
        ((direction == 0) ? TURBO_WHT_SIGNS2[t] : TURBO_WHT_SIGNS1[t]);
}

// ─── Dispatch ─────────────────────────────────────────────────────────────────

void ggml_cuda_turbo_wht(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];

    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src));
    GGML_ASSERT(ggml_is_contiguous(dst));

    int direction;
    memcpy(&direction, dst->op_params, sizeof(int));

    const int64_t n_total  = ggml_nelements(src);
    GGML_ASSERT(n_total % 128 == 0);
    const int64_t n_groups = n_total / 128;

    const float * src_ptr = (const float *) src->data;
    float       * dst_ptr = (float       *) dst->data;

    dim3 blocks(n_groups);
    dim3 threads(128);

    cudaStream_t stream = ctx.stream();
    if (direction == 0) {
        k_turbo_wht_f32<0><<<blocks, threads, 0, stream>>>(src_ptr, dst_ptr, n_groups);
    } else {
        k_turbo_wht_f32<1><<<blocks, threads, 0, stream>>>(src_ptr, dst_ptr, n_groups);
    }
}
