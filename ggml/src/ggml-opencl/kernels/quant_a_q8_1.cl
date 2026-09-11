#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Quantize a contiguous [N, K] f32 activation buffer (token-major, K contiguous
// per token) into q8_1 blocks of 32: int8 quants + per-block scale d + per-block
// sum s (= d * Sum(qs)). Consumed by kernel_gemm_noshuffle_q4_k_q8_1_dp4a for the
// dp4a (int8) dense q4_K prefill GEMM. One work-item per 32-element block.
__kernel void kernel_quant_a_q8_1(
        __global const float * src,   // [N * K]
        __global       char  * qa,    // [N * K]
        __global       half  * da,    // [N * (K/32)]
        __global       half  * sa,    // [N * (K/32)]
        int total_blocks              // N * (K/32)
) {
    const int blk = get_global_id(0);
    if (blk >= total_blocks) {
        return;
    }

    const int base = blk * 32;

    float v[32];
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        v[i] = src[base + i];
        amax = fmax(amax, fabs(v[i]));
    }

    const float d  = amax / 127.0f;
    const float id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;

    int sum = 0;
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        const int q = (int)rint(v[i] * id);
        qa[base + i] = (char)q;
        sum += q;
    }

    da[blk] = (half)d;
    sa[blk] = (half)(d * (float)sum);
}
