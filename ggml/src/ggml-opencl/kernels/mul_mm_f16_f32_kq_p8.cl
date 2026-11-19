#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable

#define LM_FIRST_256B   0
#define LM_SECOND_256B  64
#define LM_THIRD_256B   128
#define LM_FOURTH_256B  192


inline float16 mm_load_a(
    image1d_buffer_t matrix_A,
    uint subMatrixAStartInElements,
    int line_stride_matrix_A_in_bytes
) {
    __private float8 regA;
    size_t sub_block_id_m = get_local_id(0);

    uint a_texCoord = subMatrixAStartInElements/2 + (sub_block_id_m * line_stride_matrix_A_in_bytes/4);

    regA.s0123  = read_imagef(matrix_A, a_texCoord/4);
    regA.s4567  = read_imagef(matrix_A, (a_texCoord+4)/4);

    return convert_float16(as_half16(regA));
}

inline float4 alu_32(
    float16 regA,
    __local float4 * matrix_B_vec
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
    __local float * matrix_B_local
) {
    float16 out;
    __local float4 * matrix_B_vec = (__local float4*)matrix_B_local;

    out.s0123 = alu_32(regA, matrix_B_vec);
    out.s4567 = alu_32(regA, matrix_B_vec + 4);
    out.s89ab = alu_32(regA, matrix_B_vec + 8);
    out.scdef = alu_32(regA, matrix_B_vec + 12);

    return out;
}

inline void mm_mad(
    __local float * matrix_B_local,
    float16 regA,
    float8 regB,
    uint b_localOffsetInWords,
    float16 * regC0_ptr,
    float16 * regC1_ptr
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

#define TILESIZE_K 16
#define TILESIZE_M 64
#define TILESIZE_N 32
// Fuse KQ and block softmax into U8 P, block maxima and sums. No F32 score matrix is stored.
__kernel void mul_mm_f16_f32_kq_p8(
        __read_only  image1d_buffer_t matrix_A,
        __global float * matrix_B,
        __global uchar * qp,      // u8 P, [D_B][N][p_pitch]
        __global float * bmax,    // per 32-row block max,          [D_B][N][p_pitch/32]
        __global float * bsum,    // per block sum of exp(s - bmax), [D_B][N][p_pitch/32]
        __global char * mask,    // f16 [>=M][N] view (chunk rows), row stride mask_nb1 bytes
        ulong offset_mask,
        ulong mask_nb1,
        float scale,
        float max_bias,
        float m0,
        float m1,
        int n_head_log2,
        int M,
        int K,
        int N,
        int D_A,
        int D_B,
        int p_pitch              // bytes per P row, >= M, multiple of 32
) {

    // Match the n-tile-first grid in ggml_cl_fa_kq_p8_dispatch.
    const uint m_blocks = (M+TILESIZE_M-1)/TILESIZE_M;
    uint block_id_n = get_global_id(1);
    uint block_id_m = get_global_id(2) % m_blocks;
    uint block_id_d = get_global_id(2) / m_blocks;

    __private float16  regA;
    __private float8   regB;
    __private float16 regC0 = (float16)(0.0f);
    __private float16 regC1 = (float16)(0.0f);

    const uint col   = block_id_m * TILESIZE_M;
    const uint row   = block_id_n * TILESIZE_N;
    const uint depth_A = block_id_d / (D_B/D_A);
    const uint depth_B = block_id_d;

    int line_stride_matrix_A_in_bytes = K * D_A * 2;
    int line_stride_matrix_B_in_bytes = K * D_B * 4;

    const uint strideAinElements = line_stride_matrix_A_in_bytes / 2;
    const uint strideBinElements = line_stride_matrix_B_in_bytes / 4;

    size_t sub_block_id_m = get_local_id(0);

    uint b_localOffsetInWords = (sub_block_id_m/16)*16
                           + ((((sub_block_id_m)>>0)&1)<<2)
                           + ((((sub_block_id_m)>>1)&1)<<3)
                           + ((((sub_block_id_m)>>2)&1)<<0)
                           + ((((sub_block_id_m)>>3)&1)<<1);

    uint2 b_globalOffsetInWords_xy = {((sub_block_id_m%4)*4), (sub_block_id_m>>2)};
    uint subMatrixAStartInElements = col * strideAinElements + depth_A * K;
    uint subMatrixBStartInElements = 0;

    __local float matrix_B_local[1024];

    // Clamp Q loads in the last tile; columns past N are not stored.
    const uint c0 = min(row + b_globalOffsetInWords_xy.y, (uint)N - 1);
    const uint c1 = min(row + b_globalOffsetInWords_xy.y + 16, (uint)N - 1);
    const uint b_globalOffsetInWords00 = c0*strideBinElements + depth_B*K + b_globalOffsetInWords_xy.x;
    const uint b_globalOffsetInWords16 = c1*strideBinElements + depth_B*K + b_globalOffsetInWords_xy.x;

    for (uint step=0; step < K; step+=TILESIZE_K) {
        regA = mm_load_a(matrix_A, subMatrixAStartInElements, line_stride_matrix_A_in_bytes);

        uint b_coordInWords00 = subMatrixBStartInElements + b_globalOffsetInWords00;
        uint b_coordInWords16 = subMatrixBStartInElements + b_globalOffsetInWords16;

        regB.s0123 = vload4(b_coordInWords00/4, matrix_B);
        regB.s4567 = vload4(b_coordInWords16/4, matrix_B);

        mm_mad(matrix_B_local, regA, regB, b_localOffsetInWords, &regC0, &regC1);

        subMatrixAStartInElements += TILESIZE_K;
        subMatrixBStartInElements += TILESIZE_K;
    }

    // Transpose each score tile in local memory so one lane owns a 32-row softmax block.
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
            const uint n = row + half_*16 + j;
            const uint h = depth_B;
            if (n < (uint)N) {
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
                // Use a finite sentinel: -cl-finite-math-only can remove infinity checks.
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
}
