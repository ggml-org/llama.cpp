#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#ifdef cl_khr_integer_dot_product
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#endif

// Generic int8 dp4a MoE GEMM, specialized versions also exist
// MOE_QT:
//   4 (q4_K)/41(q4_1)/40(q4_0)   NIBBLE   image low nibbles -> EXP4
//   5 (q5_K)/51(q5_1)/50(q5_0)   NIBBLE+HI image nibbles + qh high-bit plane
//   6 (q6_K)                     Q6       image nibbles + qh 2-bit -> SIGN6((nibble|hi2))
//   80(q8_0)/82(mxfp4)           INT8     global int8 codes (mxfp4: convert applies kvalues LUT)

#define TILESIZE_M 64
#define TILESIZE_N 32
#define QK_K 256

#ifndef MOE_QT
#define MOE_QT 4
#endif

// 4 nibbles in low 16 bits of u -> 4 bytes (value 0..15)
#define EXP4(u)  ( ((uint)((u) & 0x000Fu))        | \
                  (((uint)((u) & 0x00F0u)) << 4)  | \
                  (((uint)((u) & 0x0F00u)) << 8)  | \
                  (((uint)((u) & 0xF000u)) << 12) )
// 4 2-bit highs in byte b -> 4 bytes, bits 4-5 (q6_K)
#define EXP2(b)  ( (((uint)((b) & 0x03u)) << 4)   | \
                  (((uint)((b) & 0x0Cu)) << 10)  | \
                  (((uint)((b) & 0x30u)) << 16)  | \
                  (((uint)((b) & 0xC0u)) << 22) )

// q6 (0..63) -> (q6-32) signed int8/byte (no inter-byte carry)
inline uint SIGN6(uint q6p){ uint x=q6p^0x20202020u; uint s=x&0x20202020u; return x|(s<<1)|(s<<2); }

// 4 high bits (one per element, in bits 0..3 of h) -> bit4 of each of 4 bytes (5-bit hi)
#define EXP1(h)  ( (((uint)((h) & 0x1u)) << 4)   | \
                  (((uint)((h) & 0x2u)) << 11)  | \
                  (((uint)((h) & 0x4u)) << 18)  | \
                  (((uint)((h) & 0x8u)) << 25) )

// per-type weight params + per-32-step unpack into qw[8] (8 int8 uints)
#if MOE_QT == 4 || MOE_QT == 41 || MOE_QT == 40
  #define WEIGHT_PARAMS __read_only image1d_buffer_t src0_q,
  #define LOAD_QW(step, sub) \
      uint qw[8]; { \
        const uint qoff0 = row + ((ne01*(step))>>3)      + ((expert_id*ne00*ne01)>>3); \
        const uint qoff1 = row + ((ne01*((step)+16))>>3) + ((expert_id*ne00*ne01)>>3); \
        const uint r0=read_imageui(src0_q,qoff0+lid).x, r1=read_imageui(src0_q,qoff0+lid+ne01).x; \
        const uint r2=read_imageui(src0_q,qoff1+lid).x, r3=read_imageui(src0_q,qoff1+lid+ne01).x; \
        qw[0]=EXP4(r0); qw[1]=EXP4(r0>>16); qw[2]=EXP4(r1); qw[3]=EXP4(r1>>16); \
        qw[4]=EXP4(r2); qw[5]=EXP4(r2>>16); qw[6]=EXP4(r3); qw[7]=EXP4(r3>>16); }

#elif MOE_QT == 5 || MOE_QT == 51 || MOE_QT == 50
  // low nibbles via image (q4_K layout) + high-bit plane src0_qh: 1 uint per 32-block
  // (bit i = high bit of element i). qh laid out [expert][block][row] to match the
  // existing q5_0 trans4 convert
  #define WEIGHT_PARAMS __read_only image1d_buffer_t src0_q, __global uint * src0_qh,
  #define LOAD_QW(step, sub) \
      uint qw[8]; { \
        const uint qoff0 = row + ((ne01*(step))>>3)      + ((expert_id*ne00*ne01)>>3); \
        const uint qoff1 = row + ((ne01*((step)+16))>>3) + ((expert_id*ne00*ne01)>>3); \
        const uint r0=read_imageui(src0_q,qoff0+lid).x, r1=read_imageui(src0_q,qoff0+lid+ne01).x; \
        const uint r2=read_imageui(src0_q,qoff1+lid).x, r3=read_imageui(src0_q,qoff1+lid+ne01).x; \
        const uint h = src0_qh[row_idx + (sub)*ne01 + expert_id*(ne00>>5)*ne01]; \
        qw[0]=EXP4(r0)|EXP1(h);        qw[1]=EXP4(r0>>16)|EXP1(h>>4); \
        qw[2]=EXP4(r1)|EXP1(h>>8);     qw[3]=EXP4(r1>>16)|EXP1(h>>12); \
        qw[4]=EXP4(r2)|EXP1(h>>16);    qw[5]=EXP4(r2>>16)|EXP1(h>>20); \
        qw[6]=EXP4(r3)|EXP1(h>>24);    qw[7]=EXP4(r3>>16)|EXP1(h>>28); }

#elif MOE_QT == 6
  #define WEIGHT_PARAMS __read_only image1d_buffer_t src0_ql, __global uint * src0_qh,
  #define LOAD_QW(step, sub) \
      uint qw[8]; { \
        const uint qoff0 = row + ((ne01*(step))>>3)      + ((expert_id*ne00*ne01)>>3); \
        const uint qoff1 = row + ((ne01*((step)+16))>>3) + ((expert_id*ne00*ne01)>>3); \
        const uint r0=read_imageui(src0_ql,qoff0+lid).x, r1=read_imageui(src0_ql,qoff0+lid+ne01).x; \
        const uint r2=read_imageui(src0_ql,qoff1+lid).x, r3=read_imageui(src0_ql,qoff1+lid+ne01).x; \
        const uint qhb = row + ((sub)*2)*ne01 + expert_id*((ne00>>5)*2)*ne01 + lid; \
        const uint qh1=src0_qh[qhb], qh2=src0_qh[qhb+ne01]; \
        qw[0]=SIGN6(EXP4(r0)|EXP2(qh1&0xFFu));        qw[1]=SIGN6(EXP4(r0>>16)|EXP2((qh1>>8)&0xFFu)); \
        qw[2]=SIGN6(EXP4(r1)|EXP2((qh1>>16)&0xFFu));  qw[3]=SIGN6(EXP4(r1>>16)|EXP2((qh1>>24)&0xFFu)); \
        qw[4]=SIGN6(EXP4(r2)|EXP2(qh2&0xFFu));        qw[5]=SIGN6(EXP4(r2>>16)|EXP2((qh2>>8)&0xFFu)); \
        qw[6]=SIGN6(EXP4(r3)|EXP2((qh2>>16)&0xFFu));  qw[7]=SIGN6(EXP4(r3>>16)|EXP2((qh2>>24)&0xFFu)); }

#elif MOE_QT == 80 || MOE_QT == 82
  // 8-bit direct: int8 codes 8 uints / 32-block, [expert][row][8*sub]. mxfp4: the
  // convert resolves kvalues_mxfp4[nibble] -> int8 and stores the e8m0_half scale.
  #define WEIGHT_PARAMS __global uint * src0_q8,
  #define LOAD_QW(step, sub) \
      uint qw[8]; { \
        const uint qb = (expert_id*ne01 + row_idx)*(ne00>>2) + (sub)*8; \
        qw[0]=src0_q8[qb+0]; qw[1]=src0_q8[qb+1]; qw[2]=src0_q8[qb+2]; qw[3]=src0_q8[qb+3]; \
        qw[4]=src0_q8[qb+4]; qw[5]=src0_q8[qb+5]; qw[6]=src0_q8[qb+6]; qw[7]=src0_q8[qb+7]; }
#else
  #error "unknown MOE_QT"
#endif

inline int dp4a4(uint w0,uint w1,uint w2,uint w3,uint a0,uint a1,uint a2,uint a3){
    int r=0; r=dot_acc_sat_4x8packed_ss_int(w0,a0,r); r=dot_acc_sat_4x8packed_ss_int(w1,a1,r);
    r=dot_acc_sat_4x8packed_ss_int(w2,a2,r); r=dot_acc_sat_4x8packed_ss_int(w3,a3,r); return r; }

// One token's two-half dp4a + uniform scale/min epilogue into acc[t].
#define MOE_DP4A_T(t) do {                                                                  \
        uint4 a0 = vload4(0, &sh_qa[t][0]);                                                 \
        uint4 a1 = vload4(0, &sh_qa[t][4]);                                                 \
        const int raw1 = dp4a4(qw[0],qw[1],qw[2],qw[3], a0.s0,a0.s1,a0.s2,a0.s3);           \
        const int raw2 = dp4a4(qw[4],qw[5],qw[6],qw[7], a1.s0,a1.s1,a1.s2,a1.s3);           \
        const float a_d = (float)sh_d[t];                                                   \
        acc[t] += sc0*a_d*(float)raw1 + sc1*a_d*(float)raw2 - mn*(float)sh_s[t];             \
    } while (0)

__attribute__((qcom_wave_pair_mode(1)))
kernel void kernel_gemm_moe_q8_1_dp4a(
        WEIGHT_PARAMS                            // per-type native weight buffer(s)
        __global     half *           src0_scale,// uniform f16 16/superblock (per-16), [expert,row]
        __global     half *           src0_min,  // uniform f16  8/superblock (per-32), [expert,row]
        __global     uint *           src1_qa,   // q8_1 activations int8 (as uint, 4/elem)
        __global     half *           src1_da,   // q8_1 per-block scale [tok_slot * ne00/32]
        __global     half *           src1_sa,   // q8_1 per-block sum*d [tok_slot * ne00/32]
        __global     uint *           src2,      // post-router (orig out positions)
        __global     ushort *         src2_emap, // tile -> expert id
        __write_only image1d_buffer_t dst,
        __global     int *            total_tiles,
        uint ne00,
        uint ne01,
        int  is_ragged,
        int  has_min                             // 0 for symmetric types (q8_0/q6_K/q4_0/...): skip min read
) {
    const uint block_id_m = get_global_id(1);
    const uint block_id_n = get_global_id(2);
    if (block_id_n >= total_tiles[0]) return;

    const uint lid = get_local_id(0);            // 0..63 -> output row within M-tile
    const ushort expert_id = src2_emap[block_id_n];
    const uint   row = block_id_m * TILESIZE_M;
    const uint   col = block_id_n * TILESIZE_N;
    const uint   row_idx = row + lid;

    // Scale/min are laid out FLAT per-32-block (2 per-16-segment scales + 1 min per
    // 32-block), so K only needs to be a multiple of 32 — works for the 32-block
    // types (q8_0/q5_0/q4_0/...) as well as the K-quants (K%256==0, same bytes).
    const uint nblk32     = ne00 / 32;
    const uint sc_per_row = nblk32 * 2;
    const uint mn_per_row = nblk32;
    const uint ne00_u = ne00 >> 2;
    const uint ne00_b = ne00 >> 5;

    __local uint sh_qa[TILESIZE_N][8];
    __local half sh_d[TILESIZE_N];
    __local half sh_s[TILESIZE_N];

    __local uint sh_src2[TILESIZE_N];
    __local int  sh_nreal;
    if (lid < TILESIZE_N) sh_src2[lid] = src2[col + lid];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0) {
        int nr = TILESIZE_N;
        if (is_ragged) { nr = 0;
            #pragma unroll
            for (int t = 0; t < TILESIZE_N; ++t) if (sh_src2[t] != 0xFFFFFFFFu) ++nr; }
        sh_nreal = nr;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const int n_real = sh_nreal;

    float acc[TILESIZE_N];
    #pragma unroll
    for (int t = 0; t < TILESIZE_N; ++t) acc[t] = 0.0f;

    for (uint step = 0; step < ne00; step += 32) {
        const uint sub = step >> 5;        // 32-block index along K

        // uniform pre-decoded scale (2 per-16-seg) + min (1) for this row, this 32-block
        __global half * scl = src0_scale + (expert_id*ne01 + row_idx)*sc_per_row + sub*2;
        const float sc0 = (float)scl[0];
        const float sc1 = (float)scl[1];
        float mn = 0.0f;
        if (has_min) mn = (float)src0_min[(expert_id*ne01 + row_idx)*mn_per_row + sub];

        LOAD_QW(step, sub)

        // Stage each token's 8 activation uints as two 128-bit uint4 loads/stores.
        const uint vlim = (uint)n_real * 2;
        for (uint idx = lid; idx < vlim; idx += 64) {
            const uint t = idx >> 1;
            const uint h = (idx & 1) << 2;   // 0 or 4
            uint4 v = vload4(0, &src1_qa[(col + t) * ne00_u + (step >> 2) + h]);
            vstore4(v, 0, &sh_qa[t][h]);
        }
        if (lid < (uint)n_real) {
            sh_d[lid] = src1_da[(col + lid) * ne00_b + sub];
            sh_s[lid] = src1_sa[(col + lid) * ne00_b + sub];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (n_real == TILESIZE_N) {
            #pragma unroll
            for (int t = 0; t < TILESIZE_N; ++t) { MOE_DP4A_T(t); }
        } else {
            #pragma unroll 4
            for (int t = 0; t < n_real; ++t) { MOE_DP4A_T(t); }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row_idx >= ne01) return;

    __local uint out_idx[TILESIZE_N];
    if (lid < TILESIZE_N) {
        uint idx = sh_src2[lid];
        if (idx == 0xFFFFFFFF) idx = sh_src2[0];
        out_idx[lid] = idx * ne01;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const uint m_offset = row + lid;
    if (n_real == TILESIZE_N) {
        #pragma unroll
        for (int t = 1; t < TILESIZE_N; ++t) write_imagef(dst, out_idx[t] + m_offset, acc[t]);
        barrier(CLK_GLOBAL_MEM_FENCE);
        write_imagef(dst, out_idx[0] + m_offset, acc[0]);
    } else {
        for (int t = 0; t < n_real; ++t) write_imagef(dst, out_idx[t] + m_offset, acc[t]);
    }
}
