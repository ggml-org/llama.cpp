#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#ifdef cl_khr_integer_dot_product
#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable
#endif

#define TILESIZE_N 32
#define QK_K 256
#define K_SCALE_SIZE 12

inline void get_scale_min_k4(
    int j,
    global const uchar * q,
    uchar * d,
    uchar * m,
    uchar mask_d6,
    uchar mask_d4,
    uchar mask_hi2
) {
    if (j < 4) {
        *d = q[j]   & mask_d6;
        *m = q[j+4] & mask_d6;
    } else {
        *d = (q[j+4] & mask_d4) | ((q[j-4] & mask_hi2) >> 2);
        *m = ((q[j+4] >> 4) & mask_d4) | ((q[j]   & mask_hi2) >> 2);
    }
}

// 4 nibbles in the low 16 bits of `u` -> 4 bytes (value 0..15, bits 0-3).
#define EXP4(u)  ( ((uint)((u) & 0x000Fu))        | \
                  (((uint)((u) & 0x00F0u)) << 4)  | \
                  (((uint)((u) & 0x0F00u)) << 8)  | \
                  (((uint)((u) & 0xF000u)) << 12) )

// 4 high bits (one per element, in bits 0-3 of h) -> bit 4 of each of 4 bytes,
// so OR with EXP4 forms the 5-bit q5_K code 0..31.
#define EXP1(h)  ( (((uint)((h) & 0x1u)) << 4)   | \
                  (((uint)((h) & 0x2u)) << 11)  | \
                  (((uint)((h) & 0x4u)) << 18)  | \
                  (((uint)((h) & 0x8u)) << 25) )

inline int dot8_q8a(uint8 qw, __local const uint * a) {
    int r = 0;
    r = dot_acc_sat_4x8packed_ss_int(qw.s0, a[0], r);
    r = dot_acc_sat_4x8packed_ss_int(qw.s1, a[1], r);
    r = dot_acc_sat_4x8packed_ss_int(qw.s2, a[2], r);
    r = dot_acc_sat_4x8packed_ss_int(qw.s3, a[3], r);
    r = dot_acc_sat_4x8packed_ss_int(qw.s4, a[4], r);
    r = dot_acc_sat_4x8packed_ss_int(qw.s5, a[5], r);
    r = dot_acc_sat_4x8packed_ss_int(qw.s6, a[6], r);
    r = dot_acc_sat_4x8packed_ss_int(qw.s7, a[7], r);
    return r;
}

__attribute__((qcom_wave_pair_mode(1)))
kernel void kernel_gemm_noshuffle_q5_k_q8_1_dp4a(
        __global const ushort * src0_q,    // q5_K low nibbles (transposed, ushort = 4 nibbles)
        __global const uchar  * src0_qh,   // q5_K high bits (transposed, uchar = 8 elems/byte)
        __global const uchar  * src0_s,    // 6-bit scale/min codes [row][superblock][12]
        __global const half   * src0_d,    // per-superblock scale (transposed)
        __global const half   * src0_dm,   // per-superblock min (transposed)
        __global const uint   * src1_qa,   // q8_1 activations int8 (as uint, 4/elem) [N, K]
        __global const half   * src1_da,   // q8_1 per-block scale [N, K/32]
        __global const half   * src1_sa,   // q8_1 per-block sum*d [N, K/32]
        __global       float  * dst,
        ulong  offsetd,
        int    m,                          // output features (rows)
        int    n_no_padding,               // tokens (cols)
        int    k,                          // K (== ne00)
        uchar  mask_d6,
        uchar  mask_d4,
        uchar  mask_hi2
) {
    dst = (global float *)((global char *)dst + offsetd);

    const uint lid = get_local_id(0);          // 0..63 -> row within the M-tile
    const uint block_id_m = get_global_id(1);
    const uint block_id_n = get_global_id(2);

    const uint row      = block_id_m * 64 + lid;
    const uint col_base = block_id_n * TILESIZE_N;
    const bool row_valid = row < (uint)m;
    const uint rrow     = row_valid ? row : 0;

    const uint num_superblocks = (uint)k / QK_K;
    const uint k_u = (uint)k >> 2;
    const uint k_b = (uint)k >> 5;

    __local uint sh_qa[TILESIZE_N][8];
    __local half sh_d[TILESIZE_N];
    __local half sh_s[TILESIZE_N];

#define NGROUPS (TILESIZE_N / 4)
    float4 acc[NGROUPS];
    #pragma unroll
    for (int g = 0; g < NGROUPS; ++g) acc[g] = (float4)(0.0f);

    for (uint step = 0; step < (uint)k; step += 32) {
        const uint sub     = step >> 5;
        const uint sb_idx  = step / QK_K;
        const uint sub_idx = sub & 7;

        const float dd  = (float)src0_d [rrow + sb_idx * m];
        const float dmm = (float)src0_dm[rrow + sb_idx * m];
        global const uchar * sc = src0_s + rrow * num_superblocks * K_SCALE_SIZE + sb_idx * K_SCALE_SIZE;
        uchar sv, mn;
        get_scale_min_k4(sub_idx, sc, &sv, &mn, mask_d6, mask_d4, mask_hi2);
        const float scale = dd  * (float)sv;
        const float minv  = dmm * (float)mn;

        // repack this row's 32 weights (nibble | high-bit) into 8 dp4a uints.
        // ushort u -> 4 elements at K = step + u*4; its 4 high bits are nibble
        // (u&1) of qh byte (step/8 + u/2).
        const uint wbase  = rrow + (step >> 2) * (uint)m;
        const uint qhbase = rrow + (step >> 3) * (uint)m;
        uint8 qw;
#define QWU(u) ( EXP4((uint)src0_q[wbase + (uint)(u) * m]) \
               | EXP1( (uint)((src0_qh[qhbase + (uint)((u) >> 1) * m] >> (((u) & 1) * 4)) & 0x0Fu) ) )
        qw.s0 = QWU(0); qw.s1 = QWU(1); qw.s2 = QWU(2); qw.s3 = QWU(3);
        qw.s4 = QWU(4); qw.s5 = QWU(5); qw.s6 = QWU(6); qw.s7 = QWU(7);
#undef QWU

        for (uint idx = lid; idx < TILESIZE_N * 8; idx += 64) {
            const uint t = idx >> 3;
            const uint u = idx & 7;
            const uint c = col_base + t;
            sh_qa[t][u] = (c < (uint)n_no_padding) ? src1_qa[c * k_u + (step >> 2) + u] : 0u;
        }
        if (lid < TILESIZE_N) {
            const uint c = col_base + lid;
            sh_d[lid] = (c < (uint)n_no_padding) ? src1_da[c * k_b + sub] : (half)0;
            sh_s[lid] = (c < (uint)n_no_padding) ? src1_sa[c * k_b + sub] : (half)0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

#define LD4(arr, b) ((float4)((float)arr[(b)+0], (float)arr[(b)+1], (float)arr[(b)+2], (float)arr[(b)+3]))
        #pragma unroll
        for (int g = 0; g < NGROUPS; ++g) {
            const int b = g * 4;
            float4 rf;
            rf.s0 = (float)dot8_q8a(qw, sh_qa[b+0]);  rf.s1 = (float)dot8_q8a(qw, sh_qa[b+1]);
            rf.s2 = (float)dot8_q8a(qw, sh_qa[b+2]);  rf.s3 = (float)dot8_q8a(qw, sh_qa[b+3]);
            acc[g] += scale * LD4(sh_d, b) * rf - minv * LD4(sh_s, b);
        }
#undef LD4
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (!row_valid) {
        return;
    }

    #pragma unroll
    for (int g = 0; g < NGROUPS; ++g) {
        const uint b = (uint)(g * 4);
        const float4 a = acc[g];
        const uint c0 = col_base + b;
        if (c0 + 0 < (uint)n_no_padding) dst[(c0 + 0) * (uint)m + row] = a.s0;
        if (c0 + 1 < (uint)n_no_padding) dst[(c0 + 1) * (uint)m + row] = a.s1;
        if (c0 + 2 < (uint)n_no_padding) dst[(c0 + 2) * (uint)m + row] = a.s2;
        if (c0 + 3 < (uint)n_no_padding) dst[(c0 + 3) * (uint)m + row] = a.s3;
    }
#undef NGROUPS
}
