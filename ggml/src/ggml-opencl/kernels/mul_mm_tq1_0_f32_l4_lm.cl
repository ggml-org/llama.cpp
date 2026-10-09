#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_intel_required_subgroup_size
#define INTEL_GPU 1
#endif

#define QK_K 256

// base 3: 5 ternary values per byte in qs, 4 per byte in qh
typedef struct {
    uchar qs[(QK_K - 4*QK_K/64)/5];
    uchar qh[QK_K/64];
    half  d;
} block_tq1_0;

constant uchar pow3[6] = { 1, 3, 9, 27, 81, 243 };

#define LOAD_VEC_A 4
#define LOAD_VEC_B 4

#define BM 64
#define BN 64
// K tile. buf_a and buf_b are 2*BM*BK*4 bytes, so BK=16 halves local memory per workgroup and doubles
// resident workgroups; that helps only when occupancy is local-memory bound, so the host picks it.
#ifndef BK
#define BK 32
#endif
#ifdef INTEL_GPU
#define TM 8
#else
#define TM 4
#endif
#define TN 8

// LM_HALF=1 stages the two LDS tiles as half instead of float, halving local memory again without
// adding barriers (unlike a smaller BK). Accumulation stays in float. The host enables it per device.
#ifndef LM_HALF
#define LM_HALF 0
#endif
#if LM_HALF
typedef half lm_st;
#define LM_LD4(p, i) convert_float4(vload4((i), (p)))
#else
typedef float lm_st;
#define LM_LD4(p, i) vload4((i), (p))
#endif

kernel void kernel_mul_mm_tq1_0_f32_l4_lm(
    global char   * src0,
    ulong offset0,
    global float4 * src1,
    ulong offset1,
    global float  * dst,
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
    global block_tq1_0 * src0_b = (global block_tq1_0 *)(src0 + offset0);
    src1 = (global float4*)((global char*)src1 + offset1);
    dst  = (global float *)((global char*)dst  + offsetd);

    local lm_st buf_a[BM * BK];
    local lm_st buf_b[BN * BK];

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

    // pos_a counts elements, not blocks
    int pos_a = batch_idx_a * batch_stride_a + ir * BM * stride_a;
    int pos_b = (batch_idx   * batch_stride_b + ic * BN * stride_b) / LOAD_VEC_B;

    // Accumulate four rows at a time. buf_a is contiguous in the row index, so a
    // whole TM slice arrives as float4 loads instead of TM scalar ones, and each
    // vector mad replaces four scalar ones. Same operands in the same order, so
    // the result is unchanged.
    float4 sums4[(TM/4) * TN];
    float4 cache_a4[TM/4];

    for (int i = 0; i < (TM/4) * TN; i++) {
        sums4[i] = (float4)(0.0f);
    }

    for (int block = 0; block < ne00; block += BK) {
        for (int l = 0; l < BM; l += loadstride_a) {
            if (ir*BM + loadc_a + l < ne01) {
                int idx = pos_a + (loadc_a + l) * stride_a + loadr_a * LOAD_VEC_A;
                int ib  = idx / QK_K;
                int e   = idx % QK_K;

                global block_tq1_0 * xb = src0_b + ib;

                // 4 consecutive elements always share the section and the power
                int n;
                global uchar * bp;
                if (e < 160)      { n = e >> 5;                    bp = xb->qs + (e & 31); }
                else if (e < 240) { int e2 = e - 160; n = e2 >> 4; bp = xb->qs + 32 + (e2 & 15); }
                else              { int e3 = e - 240; n = e3 >> 2; bp = xb->qh + (e3 & 3); }

                uchar4 b = vload4(0, bp);
                uchar  p = pow3[n];
                float  d = (float)xb->d;

                float4 v1;
                v1.s0 = (float)((int)(((uint)(uchar)(b.s0 * p) * 3) >> 8) - 1) * d;
                v1.s1 = (float)((int)(((uint)(uchar)(b.s1 * p) * 3) >> 8) - 1) * d;
                v1.s2 = (float)((int)(((uint)(uchar)(b.s2 * p) * 3) >> 8) - 1) * d;
                v1.s3 = (float)((int)(((uint)(uchar)(b.s3 * p) * 3) >> 8) - 1) * d;

                buf_a[(loadr_a * LOAD_VEC_A + 0) * BM + loadc_a + l] = v1.s0;
                buf_a[(loadr_a * LOAD_VEC_A + 1) * BM + loadc_a + l] = v1.s1;
                buf_a[(loadr_a * LOAD_VEC_A + 2) * BM + loadc_a + l] = v1.s2;
                buf_a[(loadr_a * LOAD_VEC_A + 3) * BM + loadc_a + l] = v1.s3;
            } else {
                buf_a[(loadr_a * LOAD_VEC_A + 0) * BM + loadc_a + l] = 0.0f;
                buf_a[(loadr_a * LOAD_VEC_A + 1) * BM + loadc_a + l] = 0.0f;
                buf_a[(loadr_a * LOAD_VEC_A + 2) * BM + loadc_a + l] = 0.0f;
                buf_a[(loadr_a * LOAD_VEC_A + 3) * BM + loadc_a + l] = 0.0f;
            }
        }

        for (int l = 0; l < BN; l += loadstride_b) {
            if (ic*BN + loadc_b + l < ne11) {
                int idx = pos_b + (loadc_b + l) * stride_b / LOAD_VEC_B + loadr_b;
                buf_b[(loadr_b * LOAD_VEC_B + 0) * BN + loadc_b + l] = src1[idx].s0;
                buf_b[(loadr_b * LOAD_VEC_B + 1) * BN + loadc_b + l] = src1[idx].s1;
                buf_b[(loadr_b * LOAD_VEC_B + 2) * BN + loadc_b + l] = src1[idx].s2;
                buf_b[(loadr_b * LOAD_VEC_B + 3) * BN + loadc_b + l] = src1[idx].s3;
            } else {
                buf_b[(loadr_b * LOAD_VEC_B + 0) * BN + loadc_b + l] = 0.0f;
                buf_b[(loadr_b * LOAD_VEC_B + 1) * BN + loadc_b + l] = 0.0f;
                buf_b[(loadr_b * LOAD_VEC_B + 2) * BN + loadc_b + l] = 0.0f;
                buf_b[(loadr_b * LOAD_VEC_B + 3) * BN + loadc_b + l] = 0.0f;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        pos_a += BK;
        pos_b += BK / LOAD_VEC_B;

        for (int i = 0; i < BK; i++) {
            for (int a = 0; a < TM/4; a++) {
                cache_a4[a] = LM_LD4(buf_a + (i) * BM + th_r * TM, a);
            }

            for (int cc = 0; cc < TN; cc++) {
                const float cache_b = buf_b[(i) * BN + th_c * TN + cc];
                for (int a = 0; a < TM/4; a++) {
                    const int sums_idx = cc*(TM/4) + a;
                    sums4[sums_idx] = mad(cache_a4[a], (float4)cache_b, sums4[sums_idx]);
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const int dr = ir * BM + th_r * TM;
    const int dc = ic * BN + th_c * TN;

    const int offsets = batch_idx * batch_stride_d;

    for (int cc = 0; cc < TN; cc++) {
        for (int a = 0; a < TM/4; a++) {
            const float4 v = sums4[cc * (TM/4) + a];
            const float  vs[4] = { v.s0, v.s1, v.s2, v.s3 };
            for (int k = 0; k < 4; k++) {
                if (dr + 4*a + k < ne01 && dc + cc < ne11) {
                    dst[offsets + (dc + cc) * stride_d + dr + 4*a + k] = vs[k];
                }
            }
        }
    }
}
