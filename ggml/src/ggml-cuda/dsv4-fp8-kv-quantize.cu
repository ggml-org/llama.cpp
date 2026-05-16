#include "dsv4-fp8-kv-quantize.cuh"

#if __CUDA_ARCH__ >= 890
#include <cuda_fp8.h>
#endif

#include <cstdint>

// E4M3FN code value: 0..127.
// Format: 1 sign + 4 exponent + 3 mantissa, bias 7, no inf/nan reserved.
// (i >> 3) & 0xf = exponent, i & 7 = mantissa. Code 0 is +0.
// Mirrors the CPU helper at ggml-cpu/ops.cpp:11245-11247 and the Metal
// helper dsv4_e4m3fn_value at ggml-metal.metal:2302-2308.
static __device__ __forceinline__ float dsv4_e4m3fn_value(int i) {
    const int e = (i >> 3) & 0x0f;
    const int m = i & 0x07;
    return e == 0
        ? float(m) * 0.001953125f                              // 2^-9 * m  (subnormal)
        : (1.0f + float(m) * 0.125f) * exp2f(float(e - 7));    // normal
}

// Round |x| to the nearest E4M3FN positive code value, breaking ties
// toward the EVEN code (matches CPU reference ops.cpp:11242-11253 exactly).
// Returns the dequantized F32, sign-preserved.
static __device__ __forceinline__ float dsv4_e4m3fn_dequant_sw(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax   = fminf(fabsf(x), 448.0f);

    int   best      = 0;
    float best_diff = ax;
    #pragma unroll
    for (int i = 1; i < 127; ++i) {
        const float val  = dsv4_e4m3fn_value(i);
        const float diff = fabsf(ax - val);
        if (diff < best_diff || (diff == best_diff && (i & 1) == 0 && (best & 1) != 0)) {
            best      = i;
            best_diff = diff;
        }
    }
    return sign * dsv4_e4m3fn_value(best);
}

// Dual-path E4M3FN quantize+dequantize round-trip with saturation.
//
// Native path uses NVIDIA's documented FP8 class API. The constructor
// __nv_fp8_e4m3(float) applies round-to-nearest-even and saturates to
// the finite E4M3 range (+/-448). The explicit float() conversion expands
// the FP8 storage back to F32. This is the supported public API per
// NVIDIA's cuda_fp8.h headers (CUDA toolkit >= 11.8).
//
// (We intentionally avoid the lower-level __nv_cvt_fp8_to_halfraw +
// __half2float chain: the class wrapper is clearer and avoids a half
// hop on F32-only data. There is no __nv_cvt_fp8_to_float intrinsic.)
static __device__ __forceinline__ float dsv4_e4m3fn_roundtrip(float x) {
#if __CUDA_ARCH__ >= 890
    const __nv_fp8_e4m3 q(x);
    return float(q);
#else
    // Software emulation: matches CPU reference bit-for-bit.
    return dsv4_e4m3fn_dequant_sw(x);
#endif
}

// Warp-level (32 threads) max-reduction via __shfl_xor_sync.
static __device__ __forceinline__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, offset));
    }
    return v;
}

// One block per row. blockDim.x == 64 (two warps).
static __global__ void dsv4_fp8_kv_quantize_f32(
        const char * __restrict__ src,
        char       * __restrict__ dst,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t nb00, const int64_t nb01, const int64_t nb02, const int64_t nb03,
        const int64_t nb0,  const int64_t nb1,  const int64_t nb2,  const int64_t nb3,
        const int     n_rot) {

    const int64_t n_rows = ne01 * ne02 * ne03;
    const int64_t row    = blockIdx.x;
    if (row >= n_rows) return;

    const int     tid     = threadIdx.x;        // 0..63
    const int     warp_id = tid >> 5;           // 0 or 1
    const int     lane    = tid & 31;

    const int64_t i1 = row % ne01;
    const int64_t i2 = (row / ne01) % ne02;
    const int64_t i3 = row / (ne01 * ne02);

    const char * src_base = src + i1*nb01 + i2*nb02 + i3*nb03;
    char       * dst_base = dst + i1*nb1  + i2*nb2  + i3*nb3;

    const int64_t n_nope = ne00 - (int64_t) n_rot;

    // Shared-mem slot for the two warps' partial max.
    __shared__ float warp_max[2];

    // Prefix loop: 64-element blocks.
    for (int64_t off = 0; off < n_nope; off += 64) {
        const float v = *(const float *)(src_base + (off + tid) * nb00);

        // Two-stage block-max reduction across 64 threads.
        // Stage 1: each warp reduces its 32 lanes via shfl_xor; lane 0 stores
        //          the warp's max to shared memory.
        // Stage 2: a single thread (warp 0, lane 0) combines the two warp maxes
        //          and writes the final block max back to warp_max[0].
        float m = warp_reduce_max(fabsf(v));
        if (lane == 0) warp_max[warp_id] = m;
        __syncthreads();
        if (warp_id == 0 && lane == 0) {
            warp_max[0] = fmaxf(warp_max[0], warp_max[1]);
        }
        __syncthreads();

        const float amax  = fmaxf(warp_max[0], 1.0e-4f);
        const float scale = exp2f(ceilf(log2f(amax / 448.0f)));

        const float q = dsv4_e4m3fn_roundtrip(fminf(fmaxf(v / scale, -448.0f), 448.0f)) * scale;
        *(float *)(dst_base + (off + tid) * nb0) = q;

        __syncthreads();  // protect warp_max for the next block
    }

    // Tail loop: copy n_rot elements per row through unchanged.
    // 64 threads stride through the tail.
    for (int64_t i = n_nope + tid; i < ne00; i += 64) {
        *(float *)(dst_base + i * nb0) = *(const float *)(src_base + i * nb00);
    }
}

void ggml_cuda_op_dsv4_fp8_kv_quantize(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];

    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_are_same_shape(src, dst));

    const int n_rot = ggml_get_op_params_i32(dst, 0);
    const int64_t head_dim = src->ne[0];
    const int64_t n_nope = head_dim - (int64_t) n_rot;

    GGML_ASSERT(n_rot >= 0);
    GGML_ASSERT(n_nope > 0);
    GGML_ASSERT(n_nope % 64 == 0);

    const int64_t n_rows = src->ne[1] * src->ne[2] * src->ne[3];

    const dim3 grid((unsigned) n_rows, 1, 1);
    const dim3 block(64, 1, 1);

    cudaStream_t stream = ctx.stream();
    dsv4_fp8_kv_quantize_f32<<<grid, block, 0, stream>>>(
        (const char *) src->data,
        (      char *) dst->data,
        src->ne[0], src->ne[1], src->ne[2], src->ne[3],
        (int64_t) src->nb[0], (int64_t) src->nb[1], (int64_t) src->nb[2], (int64_t) src->nb[3],
        (int64_t) dst->nb[0], (int64_t) dst->nb[1], (int64_t) dst->nb[2], (int64_t) dst->nb[3],
        n_rot);
    CUDA_CHECK(cudaGetLastError());
}
