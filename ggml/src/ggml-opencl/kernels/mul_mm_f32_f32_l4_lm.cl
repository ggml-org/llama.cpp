#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define LOAD_VEC_A 4
#define LOAD_VEC_B 4

#define BM 64
#define BN 64
#define BK 16
#define TM 4
#define TN 8

kernel void kernel_mul_mm_f32_f32_l4_lm(
    global float4 * src0,
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
    src0 = (global float4*)((global char*)src0 + offset0);
    src1 = (global float4*)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);

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
                const int idx = pos_a + (loadc_a + l) * stride_a / LOAD_VEC_A + loadr_a;
                buf_a[(loadr_a * LOAD_VEC_A + 0) * BM + loadc_a + l] = src0[idx].s0;
                buf_a[(loadr_a * LOAD_VEC_A + 1) * BM + loadc_a + l] = src0[idx].s1;
                buf_a[(loadr_a * LOAD_VEC_A + 2) * BM + loadc_a + l] = src0[idx].s2;
                buf_a[(loadr_a * LOAD_VEC_A + 3) * BM + loadc_a + l] = src0[idx].s3;
            } else {
                buf_a[(loadr_a * LOAD_VEC_A + 0) * BM + loadc_a + l] = 0.0f;
                buf_a[(loadr_a * LOAD_VEC_A + 1) * BM + loadc_a + l] = 0.0f;
                buf_a[(loadr_a * LOAD_VEC_A + 2) * BM + loadc_a + l] = 0.0f;
                buf_a[(loadr_a * LOAD_VEC_A + 3) * BM + loadc_a + l] = 0.0f;
            }
        }

        for (int l = 0; l < BN; l += loadstride_b) {
            if (ic*BN + loadc_b + l < ne11) {
                const int idx = pos_b + (loadc_b + l) * stride_b / LOAD_VEC_B + loadr_b;
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

// Multi-column f32 GEMV for the small-N (spec/MTP verify) batch. The tiled GEMM
// above always computes a full BM x BN = 64 x 64 output tile, so at ne11=3 with a
// skinny weight (e.g. GDN ssm_alpha/ssm_beta, M=32) it launches ONE under-occupied
// workgroup at ~2.3% tile utilization. This kernel assigns one 64-thread workgroup
// per output element (m,n): the 64 threads split the K reduction (float4) and
// tree-reduce in __local (no subgroup ops -> portable). ne01*ne11 workgroups.
// Weight row is re-read per column (N small -> negligible). Summation order differs
// from the tiled GEMM (lane-strided + tree) -> f32-exact-ish, not bit-identical.
kernel void kernel_gemv_f32_f32_mc(
    global float * src0, ulong offset0,   // weight: row m at m*stride_a (elements)
    global float * src1, ulong offset1,   // activations: col n at n*stride_b
    global float * dst,  ulong offsetd,   // dst [M x N] col-major: (m,n) at n*stride_d+m
    int ne00,        // K
    int ne01,        // M
    int ne11,        // N
    int stride_a,    // weight row stride (elements) = K
    int stride_b,    // activation col stride (elements) = K
    int stride_d)    // dst column stride (elements) = M
{
    src0 = (global float*)((global char*)src0 + offset0);
    src1 = (global float*)((global char*)src1 + offset1);
    dst  = (global float*)((global char*)dst  + offsetd);

    uint lane = get_local_id(0);            // 0..63
    uint out  = get_global_id(1);           // 0 .. ne01*ne11 - 1
    uint m = out % (uint)ne01;
    uint n = out / (uint)ne01;

    global float4 * wrow = (global float4*)(src0 + (ulong)m * (uint)stride_a);
    global float4 * xcol = (global float4*)(src1 + (ulong)n * (uint)stride_b);
    uint k4 = (uint)ne00 >> 2;

    float acc = 0.0f;
    for (uint k = lane; k < k4; k += 64) {
        float4 w = wrow[k];
        float4 x = xcol[k];
        acc += w.s0*x.s0 + w.s1*x.s1 + w.s2*x.s2 + w.s3*x.s3;
    }

    local float red[64];
    red[lane] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint s = 32; s > 0; s >>= 1) {
        if (lane < s) red[lane] += red[lane + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lane == 0) dst[(ulong)n * (uint)stride_d + m] = red[0];
}
