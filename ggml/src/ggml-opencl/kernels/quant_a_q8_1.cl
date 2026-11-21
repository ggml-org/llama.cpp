#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Quantize a contiguous K-major [N, K] f32 activation buffer (K contiguous) into 32-value Q8_1 SoA blocks.
// Store int8 quants, per-block scale d, and scaled sum s = d * sum(qs); one work-item handles each block.
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

inline int4 quant_b_a7_q8(float4 v, int lid, __local float * red, float * d) {
    const float4 a = fabs(v);
    red[lid] = fmax(fmax(a.x, a.y), fmax(a.z, a.w));
    barrier(CLK_LOCAL_MEM_FENCE);
    float amax = 0.0f;
    for (int i = 0; i < 8; ++i) {
        amax = fmax(amax, red[(lid & ~7) + i]);
    }
    *d = amax / 127.0f;
    const float id = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
    return convert_int4(rint(v * id));
}

inline int quant_b_a7_sum(int4 q, int lid, int n, __local int * rsum) {
    rsum[lid] = q.x + q.y + q.z + q.w;
    barrier(CLK_LOCAL_MEM_FENCE);
    int sum = 0;
    for (int i = 0; i < n; ++i) {
        sum += rsum[(lid & ~(n - 1)) + i];
    }
    return sum;
}

__kernel void kernel_quant_b_q8_1_dp4a_a7(
        __global const char * src,
        ulong                 offset,
        __global       uint * qb,
        int N,
        int nb,
        ulong nb11,
        int tile_words,
        int order,
        int bias
) {
    __local float red[256];
    __local int   rsum[256];
    const int lid = get_local_id(0);
    const int nn  = lid >> 3;
    const int w   = lid & 7;
    const int b   = get_group_id(1);
    const int t   = get_group_id(2);
    const int n   = t * 32 + nn;
    __global uint * tile = qb + ((size_t)t * nb + b) * tile_words;

    float4 v = (float4)(0.0f);
    if (n < N) {
        __global const float * p = (__global const float *)(src + offset + (ulong)n * nb11) + b * 32;
        if (order == 0) {
            const float8 v8 = vload8(w & 3, p);
            v = (w & 4) ? v8.s1357 : v8.s0246;
        } else if (order == 1) {
            v = vload4(w, p);
        } else {
            p += 16 * (w >> 2) + (w & 3);
            v = (float4)(p[0], p[8], p[4], p[12]);
        }
    }
    float d;
    const int4 q = quant_b_a7_q8(v, lid, red, &d);
    tile[w * 32 + nn] = as_uint(convert_char4(q));
    const int group = (order == 2) ? 4 : 8;
    const int sum   = quant_b_a7_sum(q, lid, group, rsum);
    if ((w & (group - 1)) == 0) {
        tile[288 + 8 * w + nn] = bias ? (uint)(0x4B400000 - bias * sum) : as_uint(d * (float)sum);
        if (w == 0) {
            tile[256 + nn] = as_uint(d);
        }
    }
}
