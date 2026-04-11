#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define LOAD_VEC_A 4
#define LOAD_VEC_B 4

#define BM 64
#define BN 64
#define BK 32
#define TM 4
#define TN 8

kernel void kernel_mul_mm_q5_k_f32_l4_lm(
    global uchar4 * src0_q,
    global uchar  * src0_qh,
    global uchar  * src0_s,
    global half   * src0_d,
    global half   * src0_dm,
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

    local float buf_a[BM * BK];
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
    float cache_a[TM];
    float cache_b[TN];

    for (int i = 0; i < TM * TN; i++) {
        sums[i] = 0.0f;
    }

    for (int block = 0; block < ne00; block += BK) {
        for (int l = 0; l < BM; l += loadstride_a) {
            if (ir*BM + loadc_a + l < ne01) {
                int idx = pos_a + (loadc_a + l) * stride_a / LOAD_VEC_A + loadr_a;
                int ib  = idx / 64;
                int iqs = (idx % 64) * 2;

                int n   = iqs / 32;
                int b   = (iqs % 32) / 16;
                int is  = 2 * n + b;
                int qsi = n * 32 + (iqs % 16) * 2;

                global uchar * scales = src0_s + ib * 12;

                int scidx0      = (is < 4) ? is     : (is + 4);
                int scidx1      = (is < 4) ? is     : (is - 4);
                int scidxmask1  = (is < 4) ? 0x30   : 0xC0;
                int scidxshift1 = (is < 4) ? 0      : 2;
                int mbidx0      = is + 4;
                int mbidx1      = (is < 4) ? is + 4 : is;
                int mbidxmask0  = (is < 4) ? 0xF    : 0xF0;
                int mbidxshift0 = (is < 4) ? 0      : 4;
                int mbidxmask1  = (is < 4) ? 0x30   : 0xC0;
                int mbidxshift1 = (is < 4) ? 0      : 2;

                uchar sc    = (scales[scidx0] & 0xF) | ((scales[scidx1] & scidxmask1) >> scidxshift1);
                uchar mbyte = ((scales[mbidx0] & mbidxmask0) >> mbidxshift0) | ((scales[mbidx1] & mbidxmask1) >> mbidxshift1);

                float d =  (float)src0_d[ib]  * (float)sc;
                float m = -(float)src0_dm[ib] * (float)mbyte;

                int qh_base = (iqs % 16) * 2;
                int bit_pos  = 2*n + b;
                uchar h0 = (src0_qh[ib*32 + qh_base + 0] >> bit_pos) & 1;
                uchar h1 = (src0_qh[ib*32 + qh_base + 1] >> bit_pos) & 1;
                uchar h2 = (src0_qh[ib*32 + qh_base + 2] >> bit_pos) & 1;
                uchar h3 = (src0_qh[ib*32 + qh_base + 3] >> bit_pos) & 1;

                global uchar4 * qs = src0_q + ib*32 + (qsi >> 2);
                uchar4 q = *qs;
                float4 v1 = (convert_float4((uchar4)(
                    ((q.s0 >> (b * 4))&0x0F) | (h0 << 4),
                    ((q.s1 >> (b * 4))&0x0F) | (h1 << 4),
                    ((q.s2 >> (b * 4))&0x0F) | (h2 << 4),
                    ((q.s3 >> (b * 4))&0x0F) | (h3 << 4)
                )))*d + m;

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
                    sums[sums_idx] = mad(cache_a[cr], cache_b[cc], sums[sums_idx]);
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
