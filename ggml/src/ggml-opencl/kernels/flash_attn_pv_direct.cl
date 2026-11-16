#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable

#define FA_KQVD_MAXC 32

// Read F16 V in its cache layout. Share each V row across the queries and heads of one KV group.
// P8 and its block scales are expanded in local memory; KV splits write separate partial sums.
__attribute__((qcom_reqd_sub_group_size("half")))
kernel void kernel_fa_kqv_direct_f16(
        global const char  * v,
        ulong                off_v,
        ulong                v_nb1,
        ulong                v_nb2,
        global const uchar * pq,
        global const float * pd,
        global       float * part,       // [split][head][n_q][dv]
        int                  dv,
        int                  n_q,
        int                  gqa,
        int                  n_head,
        int                  nblk,
        int                  blk_per_split,
        int                  p_pitch
) {
    const int lid     = get_local_id(0);
    const int d       = get_group_id(0)*128 + lid*2;
    const int head_kv = get_group_id(1);
    const int split   = get_group_id(2);
    const int ncol    = n_q*gqa;
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
        // Each lane expands 16 KV entries for one column.
        const int c  = lid % FA_KQVD_MAXC;
        const int r0 = (lid / FA_KQVD_MAXC)*16;
        float16 f = (float16)(0.0f);
        if (c < ncol) {
            const int n = c / gqa;
            const int h = head_kv*gqa + c % gqa;
            const size_t row = (size_t)h*n_q + n;
            const float sc = pd[row*nbp + b];
            const uchar16 u = vload16(0, pq + row*p_pitch + (size_t)b*32 + r0);
            f = convert_float16(u)*sc;
        }
        sh_p[r0+ 0][c] = f.s0; sh_p[r0+ 1][c] = f.s1; sh_p[r0+ 2][c] = f.s2; sh_p[r0+ 3][c] = f.s3;
        sh_p[r0+ 4][c] = f.s4; sh_p[r0+ 5][c] = f.s5; sh_p[r0+ 6][c] = f.s6; sh_p[r0+ 7][c] = f.s7;
        sh_p[r0+ 8][c] = f.s8; sh_p[r0+ 9][c] = f.s9; sh_p[r0+10][c] = f.sa; sh_p[r0+11][c] = f.sb;
        sh_p[r0+12][c] = f.sc; sh_p[r0+13][c] = f.sd; sh_p[r0+14][c] = f.se; sh_p[r0+15][c] = f.sf;
        barrier(CLK_LOCAL_MEM_FENCE);

        global const char * vr = vbase + (size_t)b*32*v_nb1;
        for (int r = 0; r < 32; ++r) {
            const float2 x = vload_half2(0, (global const half *)(vr + (size_t)r*v_nb1));
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

    #pragma unroll
    for (int c = 0; c < FA_KQVD_MAXC; ++c) {
        if (c < ncol) {
            const int n = c / gqa;
            const int h = head_kv*gqa + c % gqa;
            vstore2(acc[c], 0, part + (((size_t)split*n_head + h)*n_q + n)*dv + d);
        }
    }
}

kernel void kernel_fa_kqv_direct_reduce(
        global const float * part,
        global       float * dst,
        int                  n_elem,
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
    dst[i] = s;
}
