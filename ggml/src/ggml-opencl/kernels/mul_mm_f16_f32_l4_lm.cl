#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define LOAD_VEC_A 4
#define LOAD_VEC_B 4

#define BM 64
#ifndef BN
#define BN 64
#endif
#define BK 16
#define TM 4
#ifndef TN
#define TN 8
#endif

#ifndef KERNEL_NAME_LM
#define KERNEL_NAME_LM kernel_mul_mm_f16_f32_l4_lm
#endif

kernel void KERNEL_NAME_LM(
    global half4 * src0,
    ulong offset0,
    global float4 * src1,
    ulong offset1,
    global float * dst,
    ulong offsetd,

    int ne00,
    int ne01,
    int ne02,
    int ne11,
    int ne12,

    int stride_a,
    int stride_b,
    int stride_d,

    int batch_stride_a,
    int batch_stride_b,
    int batch_stride_d,

    int r2,
    int r3
) {
    src0 = (global half4*)((global char*)src0 + offset0);
    src1 = (global float4*)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);

    local half  buf_a[BM * BK];
    local float buf_b[BN * BK];

    const int batch_idx = get_global_id(2);

    const int i13 = batch_idx / ne12;
    const int i12 = batch_idx % ne12;

    const int i03 = i13 / r3;
    const int i02 = i12 / r2;

    const int batch_idx_a = i03 * ne02 + i02;

    const int ir = get_group_id(0);
    const int ic = get_group_id(1);

    const int tid = get_local_id(0);
    const int th_r  = tid % (BM / TM);
    const int th_c  = tid / (BM / TM);

    const int loadr_a = get_local_id(0) % (BK / LOAD_VEC_A);
    const int loadc_a = get_local_id(0) / (BK / LOAD_VEC_A);
    const int loadr_b = get_local_id(0) % (BK / LOAD_VEC_B);
    const int loadc_b = get_local_id(0) / (BK / LOAD_VEC_B);

    const int loadstride_a = get_local_size(0) * LOAD_VEC_A / BK;
    const int loadstride_b = get_local_size(0) * LOAD_VEC_B / BK;

    int pos_a = (batch_idx_a * batch_stride_a + ir * BM * stride_a) / LOAD_VEC_A;
    int pos_b = (batch_idx   * batch_stride_b + ic * BN * stride_b) / LOAD_VEC_B;

    float sums[TM * TN];
    half  cache_a[TM];
    float cache_b[TN];

    for (int i = 0; i < TM * TN; i++) {
        sums[i] = 0.0f;
    }

    for (int block = 0; block < ne00; block += BK) {
        for (int l = 0; l < BM; l += loadstride_a) {
            if (ir*BM + loadc_a + l < ne01) {
                const int idx = pos_a + (loadc_a + l) * stride_a / LOAD_VEC_A + loadr_a;
                buf_a[(loadr_a * LOAD_VEC_A + 0) * BM + loadc_a + l] = src0[idx].s0;
                buf_a[(loadr_a * LOAD_VEC_A + 1) * BM + loadc_a + l] = src0[idx].s1;
                buf_a[(loadr_a * LOAD_VEC_A + 2) * BM + loadc_a + l] = src0[idx].s2;
                buf_a[(loadr_a * LOAD_VEC_A + 3) * BM + loadc_a + l] = src0[idx].s3;
            } else {
                buf_a[(loadr_a * LOAD_VEC_A + 0) * BM + loadc_a + l] = 0.0h;
                buf_a[(loadr_a * LOAD_VEC_A + 1) * BM + loadc_a + l] = 0.0h;
                buf_a[(loadr_a * LOAD_VEC_A + 2) * BM + loadc_a + l] = 0.0h;
                buf_a[(loadr_a * LOAD_VEC_A + 3) * BM + loadc_a + l] = 0.0h;
            }
        }

        // BN=8 uses only the threads that load a column inside the tile.
        for (int l = 0; loadc_b + l < BN; l += loadstride_b) {
            if (ic*BN + loadc_b + l < ne11) {
                const int idx = pos_b + (loadc_b + l) * stride_b / LOAD_VEC_B + loadr_b;
                buf_b[(loadr_b * LOAD_VEC_B + 0) * BN + loadc_b + l] = src1[idx].s0;
                buf_b[(loadr_b * LOAD_VEC_B + 1) * BN + loadc_b + l] = src1[idx].s1;
                buf_b[(loadr_b * LOAD_VEC_B + 2) * BN + loadc_b + l] = src1[idx].s2;
                buf_b[(loadr_b * LOAD_VEC_B + 3) * BN + loadc_b + l] = src1[idx].s3;
            } else {
                buf_b[(loadr_b * LOAD_VEC_B + 0) * BN + loadc_b + l] = 0.0h;
                buf_b[(loadr_b * LOAD_VEC_B + 1) * BN + loadc_b + l] = 0.0h;
                buf_b[(loadr_b * LOAD_VEC_B + 2) * BN + loadc_b + l] = 0.0h;
                buf_b[(loadr_b * LOAD_VEC_B + 3) * BN + loadc_b + l] = 0.0h;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        pos_a += BK / LOAD_VEC_A;
        pos_b += BK / LOAD_VEC_B;

        for (int i = 0; i < BK; i++) {
            for (int j = 0; j < TM; j++) {
                cache_a[j] = buf_a[(i) * BM + th_r * TM + j];
            }
            for (int j = 0; j < TN; j++) {
                cache_b[j] = buf_b[(i) * BN + th_c * TN + j];
            }

            for (int cc = 0; cc < TN; cc++) {
                for (int cr = 0; cr < TM; cr++) {
                    const int sums_idx = cc*TM + cr;
                    sums[sums_idx] = mad(convert_float(cache_a[cr]), cache_b[cc], sums[sums_idx]);
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const int dr = ir * BM + th_r * TM;
    const int dc = ic * BN + th_c * TN;

    const int offsets = batch_idx * batch_stride_d;

    for (int cc = 0; cc < TN; cc++) {
        for (int cr = 0; cr < TM; cr++) {
            if (dr + cr < ne01 && dc + cc < ne11) {
                dst[offsets + (dc + cc) * stride_d + dr + cr] = sums[cc * TM + cr];
            }
        }
    }
}

#define SKINNY_DIRECT_REDUCE(N, ACC, ROW)                                                        \
    _Pragma("unroll")                                                                           \
    for (uint col = 0; col < N; ++col) {                                                        \
        reduce[col * 256 + lid] = ACC[col];                                                     \
    }                                                                                            \
    barrier(CLK_LOCAL_MEM_FENCE);                                                                \
                                                                                                 \
    if (lane < 16) {                                                                             \
        _Pragma("unroll")                                                                       \
        for (uint col = 0; col < N; ++col) {                                                    \
            reduce[col * 256 + lid] += reduce[col * 256 + lid + 16];                            \
        }                                                                                        \
    }                                                                                            \
    barrier(CLK_LOCAL_MEM_FENCE);                                                                \
                                                                                                 \
    if (lane < 8) {                                                                              \
        _Pragma("unroll")                                                                       \
        for (uint col = 0; col < N; ++col) {                                                    \
            reduce[col * 256 + lid] += reduce[col * 256 + lid + 8];                             \
        }                                                                                        \
    }                                                                                            \
    barrier(CLK_LOCAL_MEM_FENCE);                                                                \
                                                                                                 \
    if (lane < 4) {                                                                              \
        _Pragma("unroll")                                                                       \
        for (uint col = 0; col < N; ++col) {                                                    \
            reduce[col * 256 + lid] += reduce[col * 256 + lid + 4];                             \
        }                                                                                        \
    }                                                                                            \
    barrier(CLK_LOCAL_MEM_FENCE);                                                                \
                                                                                                 \
    if (lane < 2) {                                                                              \
        _Pragma("unroll")                                                                       \
        for (uint col = 0; col < N; ++col) {                                                    \
            reduce[col * 256 + lid] += reduce[col * 256 + lid + 2];                             \
        }                                                                                        \
    }                                                                                            \
    barrier(CLK_LOCAL_MEM_FENCE);                                                                \
                                                                                                 \
    if (lane == 0 && ROW < (uint) ne01) {                                                       \
        _Pragma("unroll")                                                                       \
        for (uint col = 0; col < N; ++col) {                                                    \
            dst[(ulong) col * (uint) stride_d + ROW] = reduce[col * 256 + lid] + reduce[col * 256 + lid + 1]; \
        }                                                                                        \
    }

#define DECLARE_SKINNY_DIRECT_KERNEL(NAME, N)                                                   \
kernel void NAME(                                                                                \
    global half4 * src0, ulong offset0,                                                         \
    read_only image1d_buffer_t src1, ulong offset1,                                             \
    global float * dst, ulong offsetd,                                                          \
    int ne00, int ne01, int ne11, int stride_a, int stride_b, int stride_d) {                   \
    src0 = (global half4 *) ((global char *) src0 + offset0);                                    \
    dst  = (global float  *) ((global char *) dst  + offsetd);                                   \
    const uint lid  = get_local_id(0);                                                           \
    const uint row0 = get_group_id(0) * 16 + (lid / 32) * 2;                                    \
    const uint row1 = row0 + 1;                                                                  \
    const uint lane = lid % 32;                                                                  \
    const uint k4   = (uint) ne00 / 4;                                                          \
    float acc0[N] = { 0.0f };                                                                    \
    float acc1[N] = { 0.0f };                                                                    \
    global half4 * wrow0 = row0 < (uint) ne01 ? src0 + ((ulong) row0 * (uint) stride_a) / 4 : src0; \
    global half4 * wrow1 = row1 < (uint) ne01 ? src0 + ((ulong) row1 * (uint) stride_a) / 4 : src0; \
    if (row0 < (uint) ne01) {                                                                    \
        for (uint k = lane; k < k4; k += 32) {                                                  \
            const float4 w0 = convert_float4(wrow0[k]);                                         \
            const float4 w1 = convert_float4(wrow1[k]);                                         \
            _Pragma("unroll")                                                                   \
            for (uint col = 0; col < N; ++col) {                                                \
                const float4 x = read_imagef(src1, col * ((uint) stride_b / 4) + k);             \
                acc0[col] += w0.s0*x.s0 + w0.s1*x.s1 + w0.s2*x.s2 + w0.s3*x.s3;                \
                acc1[col] += w1.s0*x.s0 + w1.s1*x.s1 + w1.s2*x.s2 + w1.s3*x.s3;                \
            }                                                                                    \
        }                                                                                        \
    }                                                                                            \
    local float reduce[N * 256];                                                                 \
    SKINNY_DIRECT_REDUCE(N, acc0, row0)                                                         \
    barrier(CLK_LOCAL_MEM_FENCE);                                                                \
    SKINNY_DIRECT_REDUCE(N, acc1, row1)                                                         \
}

DECLARE_SKINNY_DIRECT_KERNEL(kernel_mul_mm_f16_f32_skinny_n9_direct_img, 9)
DECLARE_SKINNY_DIRECT_KERNEL(kernel_mul_mm_f16_f32_skinny_n10_direct_img, 10)

#undef DECLARE_SKINNY_DIRECT_KERNEL
#undef SKINNY_DIRECT_REDUCE
