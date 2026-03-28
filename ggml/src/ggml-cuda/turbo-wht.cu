#include "turbo-quant.cuh"
#include "turbo-wht.cuh"

// ─── CUDA kernel ──────────────────────────────────────────────────────────────
//
// One block per 128-element group, 128 threads per block.
// direction: 0 = forward (signs1 → WHT → signs2), 1 = inverse (signs2 → WHT → signs1)
//
// When head_dim is not a multiple of 128, only the full 128-element groups
// within each head are processed.  Tail elements are left unchanged (identity).
//
// Algorithm mirrors the CPU implementation in ggml-cpu/ops.cpp:
//   1. Apply s_first elementwise
//   2. Radix-2 Hadamard butterfly (7 stages, in-place)
//   3. Normalize by 1/sqrt(128) and apply s_second elementwise

template <int direction>
static __global__ void k_turbo_wht_f32(const float * __restrict__ src,
                                        float * __restrict__ dst,
                                        int64_t n_groups,
                                        int64_t head_dim,
                                        int64_t groups_per_head) {
    const int64_t g = blockIdx.x;
    if (g >= n_groups) return;

    const int t = threadIdx.x;  // 0 .. 127

    // Map group index to position in the (possibly non-128-aligned) tensor:
    // each head has groups_per_head full groups, then a gap of (head_dim - groups_per_head*128) tail elements.
    const int64_t head_idx     = g / groups_per_head;
    const int64_t grp_in_head  = g % groups_per_head;
    const int64_t base         = head_idx * head_dim + grp_in_head * 128;

    __shared__ float x[128];

    // Load from global memory
    x[t] = src[base + t];
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
    dst[base + t] = x[t] * inv_sqrt_128 *
        ((direction == 0) ? TURBO_WHT_SIGNS2[t] : TURBO_WHT_SIGNS1[t]);
}

// ─── Simple copy kernel for tail elements (identity pass-through) ────────────

static __global__ void k_turbo_wht_copy_tail(const float * __restrict__ src,
                                              float * __restrict__ dst,
                                              int64_t n_heads,
                                              int64_t head_dim,
                                              int64_t tail_offset,
                                              int tail_size) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_heads * tail_size) return;

    const int64_t head_idx  = i / tail_size;
    const int64_t tail_elem = i % tail_size;
    const int64_t offset    = head_idx * head_dim + tail_offset + tail_elem;
    dst[offset] = src[offset];
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

    const int64_t head_dim       = src->ne[0];
    const int64_t n_heads        = ggml_nelements(src) / head_dim;
    const int64_t groups_per_head = head_dim / 128;
    const int     tail_size      = (int)(head_dim % 128);
    const int64_t n_groups       = groups_per_head * n_heads;

    const float * src_ptr = (const float *) src->data;
    float       * dst_ptr = (float       *) dst->data;

    cudaStream_t stream = ctx.stream();

    // Process full 128-element groups
    if (n_groups > 0) {
        dim3 blocks(n_groups);
        dim3 threads(128);
        if (direction == 0) {
            k_turbo_wht_f32<0><<<blocks, threads, 0, stream>>>(src_ptr, dst_ptr, n_groups, head_dim, groups_per_head);
        } else {
            k_turbo_wht_f32<1><<<blocks, threads, 0, stream>>>(src_ptr, dst_ptr, n_groups, head_dim, groups_per_head);
        }
    }

    // Pass through tail elements unchanged (no rotation)
    if (tail_size > 0) {
        const int64_t total_tail = n_heads * tail_size;
        const int block_sz = 256;
        const int n_blocks = (int)((total_tail + block_sz - 1) / block_sz);
        k_turbo_wht_copy_tail<<<n_blocks, block_sz, 0, stream>>>(
            src_ptr, dst_ptr, n_heads, head_dim, groups_per_head * 128, tail_size);
    }
}
