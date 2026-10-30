#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable

#define LM_FIRST_256B   0
#define LM_SECOND_256B  64
#define LM_THIRD_256B   128
#define LM_FOURTH_256B  192


inline float16 mm_load_a(
    image1d_buffer_t matrix_A,
    uint subMatrixAStartInElements,
    int nb01,
    int line_stride_matrix_A_in_bytes
) {
    __private float8 regA;
    size_t sub_block_id_m = get_local_id(0);

#ifdef KQV
    uint a_texCoord = subMatrixAStartInElements/2 + (sub_block_id_m * nb01/4);
#else // KQ
    uint a_texCoord = subMatrixAStartInElements/2 + (sub_block_id_m * line_stride_matrix_A_in_bytes/4);
#endif

    regA.s0123  = read_imagef(matrix_A, a_texCoord/4);
    regA.s4567  = read_imagef(matrix_A, (a_texCoord+4)/4);

    return convert_float16(as_half16(regA));
}

inline float4 alu_32(
    float16 regA,
    __local float4* matrix_B_vec
) {

    __private float4 rC = 0;
    int i = get_sub_group_id() * 64;

    rC += regA.s0  * matrix_B_vec[i];
    rC += regA.s1  * matrix_B_vec[i + 16];
    rC += regA.s4  * matrix_B_vec[i + 1];
    rC += regA.s5  * matrix_B_vec[i + 17];
    rC += regA.s8  * matrix_B_vec[i + 2];
    rC += regA.s9  * matrix_B_vec[i + 18];
    rC += regA.sc  * matrix_B_vec[i + 3];
    rC += regA.sd  * matrix_B_vec[i + 19];

    i += 32;

    rC += regA.s2  * matrix_B_vec[i];
     rC += regA.s3  * matrix_B_vec[i + 16];
    rC += regA.s6  * matrix_B_vec[i + 1];
    rC += regA.s7  * matrix_B_vec[i + 17];
    rC += regA.sa  * matrix_B_vec[i + 2];
    rC += regA.sb  * matrix_B_vec[i + 18];
    rC += regA.se  * matrix_B_vec[i + 3];
    rC += regA.sf  * matrix_B_vec[i + 19];

    return rC;
}

inline float16 alu_16(
    float16 regA,
    __local float* matrix_B_local
) {
    float16 out;
    __local float4* matrix_B_vec = (__local float4*)matrix_B_local;

    out.s0123 = alu_32(regA, matrix_B_vec);
    out.s4567 = alu_32(regA, matrix_B_vec + 4);
    out.s89ab = alu_32(regA, matrix_B_vec + 8);
    out.scdef = alu_32(regA, matrix_B_vec + 12);

    return out;
}

inline void mm_mad(
    __local float* matrix_B_local,
    float16 regA,
    float8 regB,
    uint b_localOffsetInWords,
    float16* regC0_ptr,
    float16* regC1_ptr
) {
    int offset = b_localOffsetInWords + get_sub_group_id() * 256;

    matrix_B_local[offset + LM_FIRST_256B] = regB.s0;
    matrix_B_local[offset + LM_SECOND_256B] = regB.s1;
    matrix_B_local[offset + LM_THIRD_256B] = regB.s2;
    matrix_B_local[offset + LM_FOURTH_256B] = regB.s3;

    float16 add0 = alu_16(regA, matrix_B_local);
    *regC0_ptr += add0;

    matrix_B_local[offset + LM_FIRST_256B] = regB.s4;
    matrix_B_local[offset + LM_SECOND_256B] = regB.s5;
    matrix_B_local[offset + LM_THIRD_256B] = regB.s6;
    matrix_B_local[offset + LM_FOURTH_256B] = regB.s7;

    float16 add1 = alu_16(regA, matrix_B_local);
    *regC1_ptr += add1;
}

inline void mm_store_c_N(
    __write_only image1d_buffer_t matrix_C,
    float16 regC0,
    float16 regC1,
    uint subMatrixCStartInElements,
    int line_stride_matrix_C_in_bytes,
    int mask
) {
    size_t sub_block_id_m = get_local_id(0);

    uint strideInWords     = line_stride_matrix_C_in_bytes/4;
    uint c_coordInWords_0  = (subMatrixCStartInElements + sub_block_id_m);

    uint c_coordInWords_1  = c_coordInWords_0 + 1  * strideInWords;
    uint c_coordInWords_2  = c_coordInWords_0 + 2  * strideInWords;
    uint c_coordInWords_3  = c_coordInWords_0 + 3  * strideInWords;
    uint c_coordInWords_4  = c_coordInWords_0 + 4  * strideInWords;
    uint c_coordInWords_5  = c_coordInWords_0 + 5  * strideInWords;
    uint c_coordInWords_6  = c_coordInWords_0 + 6  * strideInWords;
    uint c_coordInWords_7  = c_coordInWords_0 + 7  * strideInWords;
    uint c_coordInWords_8  = c_coordInWords_0 + 8  * strideInWords;
    uint c_coordInWords_9  = c_coordInWords_0 + 9  * strideInWords;
    uint c_coordInWords_10 = c_coordInWords_0 + 10 * strideInWords;
    uint c_coordInWords_11 = c_coordInWords_0 + 11 * strideInWords;
    uint c_coordInWords_12 = c_coordInWords_0 + 12 * strideInWords;
    uint c_coordInWords_13 = c_coordInWords_0 + 13 * strideInWords;
    uint c_coordInWords_14 = c_coordInWords_0 + 14 * strideInWords;
    uint c_coordInWords_15 = c_coordInWords_0 + 15 * strideInWords;
    uint c_coordInWords_16 = c_coordInWords_0 + 16 * strideInWords;
    uint c_coordInWords_17 = c_coordInWords_0 + 17 * strideInWords;
    uint c_coordInWords_18 = c_coordInWords_0 + 18 * strideInWords;
    uint c_coordInWords_19 = c_coordInWords_0 + 19 * strideInWords;
    uint c_coordInWords_20 = c_coordInWords_0 + 20 * strideInWords;
    uint c_coordInWords_21 = c_coordInWords_0 + 21 * strideInWords;
    uint c_coordInWords_22 = c_coordInWords_0 + 22 * strideInWords;
    uint c_coordInWords_23 = c_coordInWords_0 + 23 * strideInWords;
    uint c_coordInWords_24 = c_coordInWords_0 + 24 * strideInWords;
    uint c_coordInWords_25 = c_coordInWords_0 + 25 * strideInWords;
    uint c_coordInWords_26 = c_coordInWords_0 + 26 * strideInWords;
    uint c_coordInWords_27 = c_coordInWords_0 + 27 * strideInWords;
    uint c_coordInWords_28 = c_coordInWords_0 + 28 * strideInWords;
    uint c_coordInWords_29 = c_coordInWords_0 + 29 * strideInWords;
    uint c_coordInWords_30 = c_coordInWords_0 + 30 * strideInWords;
    uint c_coordInWords_31 = c_coordInWords_0 + 31 * strideInWords;

    if (mask > 0)  { write_imagef(matrix_C, c_coordInWords_0, regC0.s0);  }
    if (mask > 1)  { write_imagef(matrix_C, c_coordInWords_1, regC0.s1);  }
    if (mask > 2)  { write_imagef(matrix_C, c_coordInWords_2, regC0.s2);  }
    if (mask > 3)  { write_imagef(matrix_C, c_coordInWords_3, regC0.s3);  }
    if (mask > 4)  { write_imagef(matrix_C, c_coordInWords_4, regC0.s4);  }
    if (mask > 5)  { write_imagef(matrix_C, c_coordInWords_5, regC0.s5);  }
    if (mask > 6)  { write_imagef(matrix_C, c_coordInWords_6, regC0.s6);  }
    if (mask > 7)  { write_imagef(matrix_C, c_coordInWords_7, regC0.s7);  }
    if (mask > 8)  { write_imagef(matrix_C, c_coordInWords_8, regC0.s8);  }
    if (mask > 9)  { write_imagef(matrix_C, c_coordInWords_9, regC0.s9);  }
    if (mask > 10) { write_imagef(matrix_C, c_coordInWords_10, regC0.sa); }
    if (mask > 11) { write_imagef(matrix_C, c_coordInWords_11, regC0.sb); }
    if (mask > 12) { write_imagef(matrix_C, c_coordInWords_12, regC0.sc); }
    if (mask > 13) { write_imagef(matrix_C, c_coordInWords_13, regC0.sd); }
    if (mask > 14) { write_imagef(matrix_C, c_coordInWords_14, regC0.se); }
    if (mask > 15) { write_imagef(matrix_C, c_coordInWords_15, regC0.sf); }
    if (mask > 16) { write_imagef(matrix_C, c_coordInWords_16, regC1.s0); }
    if (mask > 17) { write_imagef(matrix_C, c_coordInWords_17, regC1.s1); }
    if (mask > 18) { write_imagef(matrix_C, c_coordInWords_18, regC1.s2); }
    if (mask > 19) { write_imagef(matrix_C, c_coordInWords_19, regC1.s3); }
    if (mask > 20) { write_imagef(matrix_C, c_coordInWords_20, regC1.s4); }
    if (mask > 21) { write_imagef(matrix_C, c_coordInWords_21, regC1.s5); }
    if (mask > 22) { write_imagef(matrix_C, c_coordInWords_22, regC1.s6); }
    if (mask > 23) { write_imagef(matrix_C, c_coordInWords_23, regC1.s7); }
    if (mask > 24) { write_imagef(matrix_C, c_coordInWords_24, regC1.s8); }
    if (mask > 25) { write_imagef(matrix_C, c_coordInWords_25, regC1.s9); }
    if (mask > 26) { write_imagef(matrix_C, c_coordInWords_26, regC1.sa); }
    if (mask > 27) { write_imagef(matrix_C, c_coordInWords_27, regC1.sb); }
    if (mask > 28) { write_imagef(matrix_C, c_coordInWords_28, regC1.sc); }
    if (mask > 29) { write_imagef(matrix_C, c_coordInWords_29, regC1.sd); }
    if (mask > 30) { write_imagef(matrix_C, c_coordInWords_30, regC1.se); }
    if (mask > 31) { write_imagef(matrix_C, c_coordInWords_31, regC1.sf); }
}

#define TILESIZE_K 16
#define TILESIZE_M 64
#define TILESIZE_N 32
#if defined(KQ_P8)
// KQ with the softmax folded into the epilogue (-DKQ_P8): applies scale + mask, quantises each
// 32-row block of a query column to u8 against the block max, and writes the block max and
// sum of exp(s - blockmax). kernel_fa_p8_fixup turns those into per-block scales and the
// deferred-norm row sum. No f32 score matrix and no CL_R dst image (no image cap on the chunk).
__kernel void mul_mm_f16_f32_kq_p8(
        __read_only  image1d_buffer_t matrix_A,
        int offset0,
        __global float* matrix_B,
        int offset1,
        __global uchar* qp,      // u8 P, [D_B][N][p_pitch]
        __global float* bmax,    // per 32-row block max,          [D_B][N][p_pitch/32]
        __global float* bsum,    // per block sum of exp(s - bmax), [D_B][N][p_pitch/32]
        __global char*  mask,    // f16 [>=M][N] view (chunk rows), row stride mask_nb1 bytes
        ulong offset_mask,
        ulong mask_nb1,
        float scale,
        float max_bias,
        float m0,
        float m1,
        int n_head_log2,
        int M, int K, int N,
        int D_A,
        int D_B,
        int nb01,
        int kqv_mblock_fast,
        int p_pitch,             // bytes per P row, >= M, multiple of 32
        int gqa_fold             // > 1: the query heads of one KV group share the n-tiles
) {
#elif defined(KQV)
__kernel void mul_mm_f16_f32_kqv(
        __read_only  image1d_buffer_t matrix_A,
        int offset0,
        __global float* matrix_B,
        int offset1,
        __write_only image1d_buffer_t matrix_C,
        int offsetd,
        int M, int K, int N,
        int D_A,
        int D_B,
        int nb01
) {
#else
__kernel void mul_mm_f16_f32_kq(
        __read_only  image1d_buffer_t matrix_A,
        int offset0,
        __global float* matrix_B,
        int offset1,
        __write_only image1d_buffer_t matrix_C,
        int offsetd,
        int M, int K, int N,
        int D_A,
        int D_B,
        int nb01
) {
#endif

#ifdef KQ_P8
    // The fused-softmax KQ is dispatched n-tile-fast (ggml_cl_fa_kq_p8_dispatch): two adjacent
    // n-tiles per work-group share the A panel of an m-block, and each subgroup keeps its own
    // local-memory partition in the epilogue. The grid and these derivations are a matched pair.
    const uint m_blocks = (M+TILESIZE_M-1)/TILESIZE_M;
    uint block_id_n = get_global_id(1);
    uint block_id_m = get_global_id(2) % m_blocks;
    uint block_id_d = get_global_id(2) / m_blocks;
#else
    uint block_id_m = get_global_id(1);
    uint block_id_n = get_global_id(2) % ((N+TILESIZE_N-1)/TILESIZE_N);
    uint block_id_d = get_global_id(2) / ((N+TILESIZE_N-1)/TILESIZE_N);
#endif

    __private float16  regA;
    __private float8   regB;
    __private float16 regC0;
    __private float16 regC1;

    const uint col   = block_id_m * TILESIZE_M;
    const uint row   = block_id_n * TILESIZE_N;
#ifdef KQ_P8
    // gqa_fold > 1: block_id_d is a KV head and the n-tile columns run over (query, head of the
    // group), column c = query*gqa_fold + head. A short batch (a speculative verify of 2-16
    // queries) then fills the 32-wide tile and streams each K row once per KV head instead of
    // once per query head.
    const uint fold    = gqa_fold > 1 ? (uint)gqa_fold : 1u;
    const uint depth_A = fold > 1 ? block_id_d : block_id_d / (D_B/D_A);
    const uint depth_B = block_id_d;
#else
    const uint depth_A = block_id_d / (D_B/D_A);
    const uint depth_B = block_id_d;
#endif

#ifdef KQV
    int line_stride_matrix_A_in_bytes = nb01 * M;
    int line_stride_matrix_B_in_bytes = K * N * 4;
#else
    int line_stride_matrix_A_in_bytes = K * D_A * 2;
    int line_stride_matrix_B_in_bytes = K * D_B * 4;
#endif

    int line_stride_matrix_C_in_bytes = M * 4;

    const uint strideAinElements = line_stride_matrix_A_in_bytes / 2;
    const uint strideBinElements = line_stride_matrix_B_in_bytes / 4;

    size_t sub_block_id_m = get_local_id(0);

    uint b_localOffsetInWords = (sub_block_id_m/16)*16
                           + ((((sub_block_id_m)>>0)&1)<<2)
                           + ((((sub_block_id_m)>>1)&1)<<3)
                           + ((((sub_block_id_m)>>2)&1)<<0)
                           + ((((sub_block_id_m)>>3)&1)<<1);

    uint2 b_globalOffsetInWords_xy = {((sub_block_id_m%4)*4), (sub_block_id_m>>2)};
    uint b_globalOffsetInWords00, b_globalOffsetInWords16;
#ifdef KQV
    b_globalOffsetInWords00 = b_globalOffsetInWords_xy.x + b_globalOffsetInWords_xy.y*K;
    b_globalOffsetInWords16 = b_globalOffsetInWords00 + (16 * K);
    uint subMatrixAStartInElements = depth_A * strideAinElements + col * nb01 / 2;
    uint subMatrixBStartInElements = depth_B * strideBinElements + row * K;
#else
    b_globalOffsetInWords00 = b_globalOffsetInWords_xy.x + b_globalOffsetInWords_xy.y*line_stride_matrix_B_in_bytes/4;
    b_globalOffsetInWords16 = b_globalOffsetInWords00 + (16 * line_stride_matrix_B_in_bytes/4);
    uint subMatrixAStartInElements = col * strideAinElements + depth_A * K;
    uint subMatrixBStartInElements = row * strideBinElements + depth_B * K;
#endif

    __local float matrix_B_local[1024];

#ifdef KQ_P8
    if (fold > 1) {
        // this lane's two B columns, clamped to the last real one past the batch (read, unused)
        const uint nv = (uint)N * fold;
        uint c0 = row + b_globalOffsetInWords_xy.y;
        uint c1 = c0 + 16;
        c0 = min(c0, nv - 1);
        c1 = min(c1, nv - 1);
        b_globalOffsetInWords00 = (c0/fold)*strideBinElements + (depth_B*fold + c0%fold)*K + b_globalOffsetInWords_xy.x;
        b_globalOffsetInWords16 = (c1/fold)*strideBinElements + (depth_B*fold + c1%fold)*K + b_globalOffsetInWords_xy.x;
        subMatrixBStartInElements = 0;
    }
#endif

    for (uint step=0; step < K; step+=TILESIZE_K) {
        size_t sub_block_id_m = get_local_id(0);
        regA = mm_load_a(matrix_A, subMatrixAStartInElements, nb01, line_stride_matrix_A_in_bytes);

        uint b_coordInWords00 = subMatrixBStartInElements + b_globalOffsetInWords00;
        uint b_coordInWords16 = subMatrixBStartInElements + b_globalOffsetInWords16;

        regB.s0123 = vload4(b_coordInWords00/4, matrix_B);
        regB.s4567 = vload4(b_coordInWords16/4, matrix_B);

        mm_mad(matrix_B_local, regA, regB, b_localOffsetInWords, &regC0, &regC1);

        subMatrixAStartInElements += TILESIZE_K;
        subMatrixBStartInElements += TILESIZE_K;
    }

#ifdef KQ_P8
    // ---- fused block softmax epilogue ----
    // Lane = kv row col+lane of this 64-row tile; regC0/regC1 = query columns row+0..15 / row+16..31.
    // Each 32-row block of a column is one u8 block of kernel_mul_mm_q8_kqv's P. The tile is
    // transposed through local memory in two 16-column halves; one lane then owns one (column,
    // block) and does scale + mask, block max, exp, sum and u8 conversion on 16-wide vectors.
    __local float p8_lds[2048];
    __local float * lds = p8_lds + get_sub_group_id() * 1024;
    const uint lane = get_local_id(0);
    __global const half * mrow = (__global const half *)(mask + offset_mask);
    const uint mstride = (uint)(mask_nb1 / 2);
    const int nblk = p_pitch / 32;

    #pragma unroll
    for (int half_ = 0; half_ < 2; ++half_) {
        barrier(CLK_LOCAL_MEM_FENCE);
        const float16 cc = half_ == 0 ? regC0 : regC1;
        lds[ 0*64 + lane] = cc.s0; lds[ 1*64 + lane] = cc.s1; lds[ 2*64 + lane] = cc.s2; lds[ 3*64 + lane] = cc.s3;
        lds[ 4*64 + lane] = cc.s4; lds[ 5*64 + lane] = cc.s5; lds[ 6*64 + lane] = cc.s6; lds[ 7*64 + lane] = cc.s7;
        lds[ 8*64 + lane] = cc.s8; lds[ 9*64 + lane] = cc.s9; lds[10*64 + lane] = cc.sa; lds[11*64 + lane] = cc.sb;
        lds[12*64 + lane] = cc.sc; lds[13*64 + lane] = cc.sd; lds[14*64 + lane] = cc.se; lds[15*64 + lane] = cc.sf;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lane < 32) {
            const int  j = lane >> 1;
            const int  b = lane & 1;
            const uint c = row + half_*16 + j;          // tile column
            const uint n = c / fold;                     // query
            const uint h = depth_B*fold + c % fold;      // query head
            if (c < (uint)N * fold) {
                float slope = 1.0f;
                if (max_bias > 0.0f) {
                    const float base = (int)h < n_head_log2 ? m0 : m1;
                    const int   ex   = (int)h < n_head_log2 ? (int)h + 1 : 2*((int)h - n_head_log2) + 1;
                    slope = pow(base, ex);
                }
                const uint kvb = col + b*32;               // first kv row of this block
                const __local float * src = lds + j*64 + b*32;
                float16 s0 = vload16(0, src);
                float16 s1 = vload16(1, src);
                const __global half * mp = mrow + (size_t)n*mstride + kvb;
                const float16 m0v = convert_float16(vload16(0, mp));
                const float16 m1v = convert_float16(vload16(1, mp));
                s0 = s0*scale + slope*m0v;
                s1 = s1*scale + slope*m1v;
                const float16 mx16 = fmax(s0, s1);
                const float8  mx8  = fmax(mx16.lo, mx16.hi);
                const float4  mx4  = fmax(mx8.lo, mx8.hi);
                const float2  mx2  = fmax(mx4.lo, mx4.hi);
                const float   amax = fmax(mx2.x, mx2.y);
                const uint prow = h*(uint)N + n;
                const uint blk  = kvb >> 5;
                float sum = 0.0f;
                uchar16 q0 = (uchar16)(0);
                uchar16 q1 = (uchar16)(0);
                // Masked scores are -inf; a block that is entirely masked (every kv row
                // past this query under a causal mask) has amax = -inf and must quantise
                // to zeros with a zero sum, never exp(-inf - -inf). The program builds
                // with -cl-finite-math-only, so compare against a finite sentinel: a test
                // against -INFINITY itself may be folded away.
                if (amax > -1.0e30f) {
                    const float16 e0 = exp(s0 - amax);
                    const float16 e1 = exp(s1 - amax);
                    const float16 a16 = e0 + e1;
                    const float8  a8  = a16.lo + a16.hi;
                    const float4  a4  = a8.lo + a8.hi;
                    const float2  a2  = a4.lo + a4.hi;
                    sum = a2.x + a2.y;
                    q0 = convert_uchar16_sat_rte(e0*255.0f);
                    q1 = convert_uchar16_sat_rte(e1*255.0f);
                }
                __global uchar * qdst = qp + (size_t)prow*(size_t)p_pitch + (size_t)kvb;
                vstore16(q0, 0, qdst);
                vstore16(q1, 1, qdst);
                bmax[(size_t)prow*nblk + blk] = amax;
                bsum[(size_t)prow*nblk + blk] = sum;
            }
        }
    }
#else
    uint subMatrixCStartInElements = depth_B * N * M + row * M + col;
    mm_store_c_N(matrix_C, regC0, regC1, subMatrixCStartInElements, line_stride_matrix_C_in_bytes, (N-block_id_n*32));
#endif
}

