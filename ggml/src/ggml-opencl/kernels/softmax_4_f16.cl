#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_intel_subgroups
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#else
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif

#ifdef cl_intel_required_subgroup_size
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#define INTEL_GPU 1
#define REQD_SUBGROUP_SIZE_16 __attribute__((intel_reqd_sub_group_size(16)))
#define REQD_SUBGROUP_SIZE_32 __attribute__((intel_reqd_sub_group_size(32)))
#elif defined(cl_qcom_reqd_sub_group_size)
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64  __attribute__((qcom_reqd_sub_group_size("half")))
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#endif

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_soft_max_4_f16(
        global char * src0,
        ulong offset0,
        global char * src1,
        ulong offset1,
        global char * src2,
        ulong offset2,
        global char * dst,
        ulong offsetd,
        int ne00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne12,
        int ne13,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        ulong nb1,
        ulong nb2,
        ulong nb3,
        float scale,
        float max_bias,
        float m0,
        float m1,
        int n_head_log2
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    src2 = src2 + offset2;
    dst  = dst  + offsetd;

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0);

    int i13 = i03%ne13;
    int i12 = i02%ne12;
    int i11 = i01;

    global float4 * psrc4 = (global float4 *)(src0 + i01*nb01 + i02*nb02 + i03*nb03);
    global half4  * pmask = src1 != src0 ? (global half4 *)(src1 + i11*nb11 + i12*nb12 + i13*nb13) : 0;
    global float  * psrc2 = src2 != src0 ? (global float *)(src2) : 0;
    global float4 * pdst4 = (global float4 *)(dst  + i01*nb1 + i02*nb2 + i03*nb3);

    float slope = 1.0f;

    // ALiBi
    if (max_bias > 0.0f) {
        int h = i02;

        float base = h < n_head_log2 ? m0 : m1;
        int   exp  = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;

        slope = pow(base, exp);
    }

    // parallel max
    float4 lmax4 = psrc2 ? psrc2[i02] : -INFINITY;
    for (int i00 = get_local_id(0); i00 < ne00/4; i00 += get_local_size(0)) {
        lmax4 = fmax(lmax4, psrc4[i00]*scale + slope*(pmask ? convert_float4(pmask[i00]) : 0.0f));
    }
    float lmax = fmax(fmax(lmax4.s0, lmax4.s1), fmax(lmax4.s2, lmax4.s3));

    const float max = sub_group_reduce_max(lmax);

    // parallel sum
    float4 lsum4 = 0.0f;
    for (int i00 = get_local_id(0); i00 < ne00/4; i00 += get_local_size(0)) {
        const float4 exp_psrc4 = exp((psrc4[i00]*scale + slope*(pmask ? convert_float4(pmask[i00]) : 0.0f)) - max);
        lsum4 += exp_psrc4;
        pdst4[i00] = exp_psrc4;
    }
    float lsum = lsum4.s0 + lsum4.s1 + lsum4.s2 + lsum4.s3;

    float sum = sub_group_reduce_add(lsum);

    if (psrc2) {
        sum += exp(psrc2[i02] - max);
    }

    for (int i00 = get_local_id(0); i00 < ne00/4; i00 += get_local_size(0)) {
        pdst4[i00] /= sum;
    }
}

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_soft_max_4_f16_nonorm(
        global float * psums,
        ulong offset_sums,
        global char * src0,
        ulong offset0,
        global char * src1,
        ulong offset1,
        global char * src2,
        ulong offset2,
        global char * dst,
        ulong offsetd,
        int ne00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne12,
        int ne13,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        ulong nb1,
        ulong nb2,
        ulong nb3,
        float scale,
        float max_bias,
        float m0,
        float m1,
        int n_head_log2
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    src2 = src2 + offset2;
    dst  = dst  + offsetd;

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0);

    int i13 = i03%ne13;
    int i12 = i02%ne12;
    int i11 = i01;

    global float4 * psrc4 = (global float4 *)(src0 + i01*nb01 + i02*nb02 + i03*nb03);
    global half4  * pmask = src1 != src0 ? (global half4 *)(src1 + i11*nb11 + i12*nb12 + i13*nb13) : 0;
    global float  * psrc2 = src2 != src0 ? (global float *)(src2) : 0;
    global float4 * pdst4 = (global float4 *)(dst  + i01*nb1 + i02*nb2 + i03*nb3);

    float slope = 1.0f;

    // ALiBi
    if (max_bias > 0.0f) {
        int h = i02;

        float base = h < n_head_log2 ? m0 : m1;
        int   exp  = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;

        slope = pow(base, exp);
    }

    // parallel max
    float4 lmax4 = psrc2 ? psrc2[i02] : -INFINITY;
    for (int i00 = get_local_id(0); i00 < ne00/4; i00 += get_local_size(0)) {
        lmax4 = fmax(lmax4, psrc4[i00]*scale + slope*(pmask ? convert_float4(pmask[i00]) : 0.0f));
    }
    float lmax = fmax(fmax(lmax4.s0, lmax4.s1), fmax(lmax4.s2, lmax4.s3));

    const float max = sub_group_reduce_max(lmax);

    // parallel sum
    float4 lsum4 = 0.0f;
    for (int i00 = get_local_id(0); i00 < ne00/4; i00 += get_local_size(0)) {
        const float4 exp_psrc4 = exp((psrc4[i00]*scale + slope*(pmask ? convert_float4(pmask[i00]) : 0.0f)) - max);
        lsum4 += exp_psrc4;
        pdst4[i00] = exp_psrc4;
    }
    float lsum = lsum4.s0 + lsum4.s1 + lsum4.s2 + lsum4.s3;

    float sum = sub_group_reduce_add(lsum);

    if (psrc2) {
        sum += exp(psrc2[i02] - max);
    }

    // The normalize pass is deferred: dst holds exp(x - max) and the row sum is published; the
    // consumer folds 1/sum into its much smaller output, saving a read+write of the score matrix.
    if (get_local_id(0) == 0) {
        psums = (global float *)((global char *)psums + offset_sums);
        psums[i03*get_num_groups(1)*get_num_groups(0) + i02*get_num_groups(0) + i01] = sum;
    }
}


// Applies the deferred softmax normalisation to the KQV result: out[:, n, h] /= sums[h*ne1 + n].
// The KQV output is dv x n_q x n_head, which is orders of magnitude smaller than the score
// matrix, so doing the division here instead of over the scores is nearly free.
kernel void kernel_fa_scale_rows_f32(
        global float * dst,
        ulong offsetd,
        global float * sums,
        ulong offset_sums,
        int dv,
        int ne1
) {
    dst  = (global float *)((global char *)dst  + offsetd);
    sums = (global float *)((global char *)sums + offset_sums);

    const int n = get_group_id(0);
    const int h = get_group_id(1);

    const float s = sums[h*ne1 + n];
    const float r = s > 0.0f ? 1.0f/s : 0.0f;

    global float * row = dst + (size_t)h*ne1*dv + (size_t)n*dv;
    for (int i = get_local_id(0); i < dv; i += get_local_size(0)) {
        row[i] *= r;
    }
}

// Deferred-normalisation softmax that emits P as u8 + per-32-block scale instead of f32.
// A single row-wide scale loses too much precision on the KQV result; a per-32-block scale does
// not. The block layout matches kernel_moe_reorder_quant_a_q8_1, which the dp4a GEMM consumes.
// The score matrix shrinks from 4 bytes per element to 1 + a half per 32.
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
// The parameter list after the three output pairs matches kernel_soft_max_4_f16_nonorm exactly,
// so the host sets the arguments with the same loop and only the leading count changes. dst and
// nb1..nb3 are unused here - P goes to qp/dp instead of the f32 score matrix.
kernel void kernel_soft_max_4_f16_q8(
        global uchar * qp,
        ulong offset_qp,
        global float * dp,
        ulong offset_dp,
        int qp_pitch,               // bytes per P row, >= ne00, multiple of 32
        global float * psums,
        ulong offset_sums,
        global char * src0,
        ulong offset0,
        global char * src1,
        ulong offset1,
        global char * src2,
        ulong offset2,
        global char * dst,
        ulong offsetd,
        int ne00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne12,
        int ne13,
        ulong nb11,
        ulong nb12,
        ulong nb13,
        ulong nb1,
        ulong nb2,
        ulong nb3,
        float scale,
        float max_bias,
        float m0,
        float m1,
        int n_head_log2
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    src2 = src2 + offset2;

    (void) dst; (void) offsetd; (void) nb1; (void) nb2; (void) nb3;

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0);

    int i13 = i03%ne13;
    int i12 = i02%ne12;
    int i11 = i01;

    global float4 * psrc4 = (global float4 *)(src0 + i01*nb01 + i02*nb02 + i03*nb03);
    global half4  * pmask = src1 != src0 ? (global half4 *)(src1 + i11*nb11 + i12*nb12 + i13*nb13) : 0;
    global float  * psrc2 = src2 != src0 ? (global float *)(src2) : 0;

    float slope = 1.0f;

    if (max_bias > 0.0f) {
        int h = i02;

        float base = h < n_head_log2 ? m0 : m1;
        int   exp  = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;

        slope = pow(base, exp);
    }

    // pass 1: row max, strided so the reads coalesce across lanes
    float4 lmax4 = psrc2 ? psrc2[i02] : -INFINITY;
    for (int i00 = get_local_id(0); i00 < ne00/4; i00 += get_local_size(0)) {
        lmax4 = fmax(lmax4, psrc4[i00]*scale + slope*(pmask ? convert_float4(pmask[i00]) : 0.0f));
    }
    float lmax = fmax(fmax(lmax4.s0, lmax4.s1), fmax(lmax4.s2, lmax4.s3));

    const float max = sub_group_reduce_max(lmax);

    // pass 2: one 32-block per lane, so a lane owns the whole block it has to scale.
    // A lane reads 32 contiguous floats rather than a strided float4 - same bytes, and the
    // block reduction stays in registers.
    const int nblk = ne00 / 32;
    const uint row  = (uint)(i03*get_num_groups(1)*get_num_groups(0) + i02*get_num_groups(0) + i01);

    global uchar * qrow = (global uchar *)((global char *)qp + offset_qp) + (size_t)row*qp_pitch;
    global float * drow = (global float *)((global char *)dp + offset_dp) + (size_t)row*(qp_pitch/32);

    float lsum = 0.0f;
    for (int b = get_local_id(0); b < nblk; b += get_local_size(0)) {
        float v[32];
        float amax = 0.0f;

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float4 e = exp((psrc4[b*8 + i]*scale + slope*(pmask ? convert_float4(pmask[b*8 + i]) : 0.0f)) - max);
            v[i*4 + 0] = e.s0; v[i*4 + 1] = e.s1; v[i*4 + 2] = e.s2; v[i*4 + 3] = e.s3;
            amax = fmax(amax, fmax(fmax(e.s0, e.s1), fmax(e.s2, e.s3)));
        }

        // P >= 0, so quantise unsigned: 255 levels instead of 127. The scale stays f32: it is
        // exp(blockmax - rowmax)/255, and a row whose max sits far above its scores (a large
        // attention sink) puts it below the half normal range, where a device that flushes
        // subnormals drops the whole block.
        const float d  = amax / 255.0f;
        const float id = amax > 0.0f ? 255.0f / amax : 0.0f;

        drow[b] = d;

        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            const int q = (int)rint(v[i] * id);
            qrow[b*32 + i] = (uchar)clamp(q, 0, 255);
            lsum += v[i];
        }
    }

    float sum = sub_group_reduce_add(lsum);

    if (psrc2) {
        sum += exp(psrc2[i02] - max);
    }

    if (get_local_id(0) == 0) {
        psums = (global float *)((global char *)psums + offset_sums);
        psums[row] = sum;
    }
}

// Second half of the fused KQ+softmax path (mul_mm_f16_f32_kq_p8). One workgroup per
// (query, head) row: reduce the per-32-block maxima to the row max (sinks join it as an extra
// column), then emit the per-block f32 scale exp(bmax - rowmax)/255 that kernel_mul_mm_q8_kqv
// consumes and the deferred-norm row sum. Reads 8 bytes per 32 scores.
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_fa_p8_fixup(
        global float * bmax,
        global float * bsum,
        global float * dp,
        global float * psums,
        global char  * sinks,
        ulong offset_sinks,
        int has_sinks,
        int nblk,
        int row_blk,                // blocks per row of bmax/bsum/dp (the padded pitch), >= nblk
        int N
) {
    const int n    = get_group_id(0);
    const int head = get_group_id(1);
    const uint row = (uint)head*(uint)N + (uint)n;
    global float * bm = bmax + (size_t)row*row_blk;
    global float * bs = bsum + (size_t)row*row_blk;
    global float * dr = dp   + (size_t)row*row_blk;

    float lmax = -INFINITY;
    for (int b = get_local_id(0); b < nblk; b += get_local_size(0)) {
        lmax = fmax(lmax, bm[b]);
    }
    float rowmax = sub_group_reduce_max(lmax);
    float sink = 0.0f;
    if (has_sinks) {
        sink = ((global float *)(sinks + offset_sinks))[head];
        rowmax = fmax(rowmax, sink);
    }

    float lsum = 0.0f;
    for (int b = get_local_id(0); b < nblk; b += get_local_size(0)) {
        const float m = bm[b];
        // finite sentinels rather than -INFINITY compares: the program builds with
        // -cl-finite-math-only, which may fold a test against infinity itself.
        const float f = (m > -1.0e30f && rowmax > -1.0e30f) ? exp(m - rowmax) : 0.0f;
        dr[b] = f / 255.0f;   // f32: see kernel_soft_max_4_f16_q8
        lsum += bs[b]*f;
    }
    float sum = sub_group_reduce_add(lsum);
    if (has_sinks && rowmax > -1.0e30f) {
        sum += exp(sink - rowmax);
    }
    if (get_local_id(0) == 0) {
        psums[row] = sum;
    }
}

// KQV for a batch narrower than one tile (2-16 query verify), reading V in the cache's row layout
// instead of rebuilding V^T per call. Columns are the (query, head) pairs of one KV group (at most
// FA_KQVD_MAXC); each lane owns two d of every V row, and per 32-row block P and its scale are
// unpacked to f32 in local memory. KV is split across work-groups (get_group_id(2));
// kernel_fa_kqv_direct_reduce adds the partials.
//   P u8 [head][n_q][p_pitch], P scales f32 [head][n_q][p_pitch/32], part f32 [split][head][n_q][dv]
#ifndef FA_KQVD_MAXC
#define FA_KQVD_MAXC 32
#endif
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_fa_kqv_direct_f16(
        global const char  * v,
        ulong                off_v,
        ulong                v_nb1,         // bytes per kv row
        ulong                v_nb2,         // bytes per kv head
        global const uchar * pq,
        global const float * pd,
        global       float * part,
        int                  dv,
        int                  n_q,
        int                  gqa,
        int                  n_head,
        int                  nblk,          // 32-row blocks of kv
        int                  blk_per_split,
        int                  p_pitch
) {
    const int lid     = get_local_id(0);
    const int d       = get_group_id(0)*128 + lid*2;
    const int head_kv = get_group_id(1);
    const int split   = get_group_id(2);
    const int ncol    = n_q*gqa;            // column c = query*gqa + head of the group
    const int b0      = split*blk_per_split;
    const int b1      = min(b0 + blk_per_split, nblk);
    const int nbp     = p_pitch / 32;

    __local float sh_p[32][FA_KQVD_MAXC];

    float2 acc[FA_KQVD_MAXC];
    #pragma unroll
    for (int c = 0; c < FA_KQVD_MAXC; ++c) {
        acc[c] = (float2)(0.0f);
    }

    global const char * vbase = v + off_v + (size_t)head_kv*v_nb2 + (size_t)d*sizeof(half);

    for (int b = b0; b < b1; ++b) {
        barrier(CLK_LOCAL_MEM_FENCE);
        // 32 rows x ncol columns of P: lane i unpacks column i%32's 16 rows (i/32 picks the half)
        for (int i = lid; i < 2*FA_KQVD_MAXC; i += 64) {
            const int c  = i % FA_KQVD_MAXC;
            const int r0 = (i / FA_KQVD_MAXC)*16;
            if (c < ncol) {
                const int n = c / gqa;
                const int h = head_kv*gqa + c % gqa;
                const size_t row = (size_t)h*n_q + n;
                const float sc = pd[row*nbp + b];
                const uchar16 u = vload16(0, pq + row*p_pitch + (size_t)b*32 + r0);
                const float16 f = convert_float16(u)*sc;
                sh_p[r0+ 0][c] = f.s0; sh_p[r0+ 1][c] = f.s1; sh_p[r0+ 2][c] = f.s2; sh_p[r0+ 3][c] = f.s3;
                sh_p[r0+ 4][c] = f.s4; sh_p[r0+ 5][c] = f.s5; sh_p[r0+ 6][c] = f.s6; sh_p[r0+ 7][c] = f.s7;
                sh_p[r0+ 8][c] = f.s8; sh_p[r0+ 9][c] = f.s9; sh_p[r0+10][c] = f.sa; sh_p[r0+11][c] = f.sb;
                sh_p[r0+12][c] = f.sc; sh_p[r0+13][c] = f.sd; sh_p[r0+14][c] = f.se; sh_p[r0+15][c] = f.sf;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (d < dv) {
            global const char * vr = vbase + (size_t)b*32*v_nb1;
            for (int r = 0; r < 32; ++r) {
                const float2 x = vload_half2(0, (global const half *)(vr + (size_t)r*v_nb1));
                // four columns per local read; columns past ncol hold stale values and are
                // never stored
                #pragma unroll
                for (int c = 0; c < FA_KQVD_MAXC; c += 4) {
                    if (c < ncol) {
                        const float4 pv = vload4(0, &sh_p[r][c]);
                        acc[c+0] = mad((float2)(pv.x), x, acc[c+0]);
                        acc[c+1] = mad((float2)(pv.y), x, acc[c+1]);
                        acc[c+2] = mad((float2)(pv.z), x, acc[c+2]);
                        acc[c+3] = mad((float2)(pv.w), x, acc[c+3]);
                    }
                }
            }
        }
    }

    if (d < dv) {
        #pragma unroll
        for (int c = 0; c < FA_KQVD_MAXC; ++c) {
            if (c < ncol) {
                const int n = c / gqa;
                const int h = head_kv*gqa + c % gqa;
                vstore2(acc[c], 0, part + (((size_t)split*n_head + h)*n_q + n)*dv + d);
            }
        }
    }
}

// Sum the KV-split partials of kernel_fa_kqv_direct_f16 into [head][n_q][dv].
kernel void kernel_fa_kqv_direct_reduce(
        global const float * part,
        global       float * dst,
        ulong                off_dst,
        int                  n_elem,        // n_head*n_q*dv
        int                  n_split
) {
    const int i = get_global_id(0);
    if (i >= n_elem) {
        return;
    }
    float s = 0.0f;
    for (int k = 0; k < n_split; ++k) {
        s += part[(size_t)k*n_elem + i];
    }
    ((global float *)((global char *)dst + off_dst))[i] = s;
}
