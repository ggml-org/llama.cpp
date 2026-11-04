#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_intel_required_subgroup_size
#define INTEL_GPU 1
#endif

#define QK_K 256

struct block_q3_K {
    uchar hmask[QK_K/8];
    uchar qs[QK_K/4];
    uchar scales[12];
    half d;
};

#define LOAD_VEC_A 4
#define LOAD_VEC_B 4

#define BM 64
#define BN 64
#ifndef BK
#define BK 32
#endif
#ifdef INTEL_GPU
#define TM 8
#else
#define TM 4
#endif
#define TN 8

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

kernel void kernel_mul_mm_q3_k_f32_l4_lm(
    global uchar * src0_qs,
    global uchar * src0_hm,
    global uchar * src0_sc,
    global half * src0_d,
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

    // pos_a counts elements, not blocks - one q3_K super block holds QK_K of them
    int pos_a = batch_idx_a * batch_stride_a + ir * BM * stride_a;
    int pos_b = (batch_idx   * batch_stride_b + ic * BN * stride_b) / LOAD_VEC_B;

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
                ib += offset0/sizeof(struct block_q3_K);
                int e   = idx % QK_K;

                // Only qs advances in the second half of a super-block.
                int n     = e >> 7;
                int rem   = e & 127;
                int j     = rem >> 5;
                int s     = (rem >> 4) & 1;
                int l16   = rem & 15;
                int is    = 8*n + 2*j + s;
                int shift = 2*j;
                uchar m   = (uchar)(1u << (4*n + j));

                global ushort * a = (global ushort *)(src0_sc + 12*ib);
                uint a0  = (uint)a[0] | ((uint)a[1] << 16);
                uint a1  = (uint)a[2] | ((uint)a[3] << 16);
                uint tmp = (uint)a[4] | ((uint)a[5] << 16);

                uint g   = is >> 2;
                uint w   = (g & 1u) ? a1 : a0;
                w = ((w >> ((g >> 1)*4)) & 0x0f0f0f0fu) | (((tmp >> (2*g)) & 0x03030303u) << 4);

                uint  sc6 = (w >> (8*(is & 3))) & 0xFFu;
                float dl  = (float)src0_d[ib] * ((float)sc6 - 32.0f);

                uchar4 q  = vload4(0, src0_qs + 64*ib + 32*n + 16*s + l16);
                uchar4 hb = vload4(0, src0_hm + 32*ib + 16*s + l16);

                float4 qv = convert_float4((uchar4)((q.s0 >> shift) & 3,
                                                    (q.s1 >> shift) & 3,
                                                    (q.s2 >> shift) & 3,
                                                    (q.s3 >> shift) & 3));
                float4 hv = (float4)((hb.s0 & m) ? 0.f : 4.f,
                                     (hb.s1 & m) ? 0.f : 4.f,
                                     (hb.s2 & m) ? 0.f : 4.f,
                                     (hb.s3 & m) ? 0.f : 4.f);
                float4 v1 = (qv - hv) * dl;

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

    const ulong offsets = batch_idx * batch_stride_d;

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
