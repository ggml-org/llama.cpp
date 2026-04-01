#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Q4_K × F32 batch GEMM — reorganized qs buffer (coalesced for Apple M1)
//
// Two-buffer design:
//   src0_meta : original packed block_q4_K (144 bytes/block), used ONLY for
//               d, dmin, and scales metadata — these 16 bytes/block are read
//               once per super-block and cached in private registers for the
//               remaining 7 BK steps.
//   src0_qs_t : qs data in [batch, nb, ne01, 128] order — block-major,
//               row-minor — so threads working on the same tile (same ib_in_row
//               but consecutive rows) access stride-128-byte-apart addresses
//               instead of the packed format's stride-1728-byte gap.
//
// Memory coalescing improvement:
//   Packed : stride = nb * 144 = 12 * 144 = 1728 bytes between adjacent rows
//   qst    : stride = 128 bytes between adjacent rows (fully coalesced)
//
// See mul_mm_q4_k_f32_l4_lm_packed.cl for the packed fallback.
// Tile: BM=BN=64, BK=32, TM=4, TN=8, LOAD_VEC_A=8, 128 threads.

#define LOAD_VEC_A 8
#define LOAD_VEC_B 4

#define BM 64
#define BN 64
#define BK 32
#define TM 4
#define TN 8

#define QK_K          256
#define K_SCALE_SIZE  12
#define BLOCK_Q4_K_SZ 144  // sizeof(block_q4_K) = 2+2+12+128

kernel void kernel_mul_mm_q4_k_f32_l4_lm_qst(
    global char   * src0_meta,  // packed block_q4_K for metadata only (d, dmin, scales)
    ulong           offset0,
    global uchar  * src0_qs_t,  // qs in [batch, nb, ne01, 128] order (coalesced)
    global float4 * src1,
    ulong           offset1,
    global float  * dst,
    ulong           offsetd,
    int ne00,           // K  (must be divisible by QK_K=256)
    int ne01,           // M  (weight rows)
    int ne02,           // batch dim of weights
    int ne11,           // N  (activation tokens)
    int ne12,           // batch dim of activations
    int stride_b,       // = ne10 = ne00
    int stride_d,       // = ne01
    int batch_stride_b, // = ne10 * ne11
    int batch_stride_d, // = ne0  * ne1
    int r2,
    int r3
) {
    src0_meta = src0_meta + offset0;
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

    const int ir  = get_group_id(0);
    const int ic  = get_group_id(1);
    const int tid = get_local_id(0);

    const int th_r = tid % (BM / TM);  // 0..15
    const int th_c = tid / (BM / TM);  // 0..7

    const int loadr_a = tid % (BK / LOAD_VEC_A);  // 0..3
    const int loadc_a = tid / (BK / LOAD_VEC_A);  // 0..31
    const int loadr_b = tid % (BK / LOAD_VEC_B);  // 0..7
    const int loadc_b = tid / (BK / LOAD_VEC_B);  // 0..15

    const int loadstride_a = get_local_size(0) * LOAD_VEC_A / BK;  // 32
    const int loadstride_b = get_local_size(0) * LOAD_VEC_B / BK;  // 16

    int pos_b = (batch_idx * batch_stride_b + ic * BN * stride_b) / LOAD_VEC_B;

    const int nb = ne00 / QK_K;  // super-blocks per weight row

    float sums[TM * TN];
    float cache_a[TM];
    float cache_b[TN];

    #pragma unroll
    for (int i = 0; i < TM * TN; i++) sums[i] = 0.0f;

    // Private register cache for d, dmin, scales per row.
    // Each thread handles at most 2 rows (l=0 and l=loadstride_a).
    // Reloaded from src0_meta only when sg==0 (start of each new super-block).
    float priv_dall[2];
    float priv_dmin[2];
    uchar priv_sc[2][K_SCALE_SIZE];

    for (int block = 0; block < ne00; block += BK) {
        const int ib_in_row   = block / QK_K;
        const int sg          = (block / BK) % 8;   // scale-group 0..7
        const int qs_byte_off = (sg / 2) * 32 + loadr_a * 8;
        const int nibble_hi   = sg & 1;
        const int sg_m4 = sg < 4 ? sg : sg - 4;

        // ---- Load tile of A (Q4_K weights) ----
        for (int l = 0; l < BM; l += loadstride_a) {
            const int row = ir * BM + loadc_a + l;
            const int li  = l / loadstride_a;  // 0 or 1
            if (row < ne01) {
                // qs from coalesced [batch, nb, ne01, 128] buffer:
                // stride between adjacent rows = 128 bytes (vs 1728 for packed)
                ulong qs_t_idx = ((ulong)batch_idx_a * nb + ib_in_row) * ne01 + row;
                uchar8 q = vload8(0, src0_qs_t + qs_t_idx * 128 + qs_byte_off);
                uchar8 nibbles = nibble_hi ? (q >> (uchar8)4) : (q & (uchar8)0x0F);

                // Reload d/dmin/scales cache from src0_meta only at start of each
                // new super-block. All 8 BK steps share the same metadata.
                if (sg == 0) {
                    ulong bi = (ulong)batch_idx_a * ne01 * nb
                             + (ulong)row * nb
                             + ib_in_row;
                    global char * bp = src0_meta + bi * BLOCK_Q4_K_SZ;
                    priv_dall[li] = (float)vload_half(0, (global half*)bp);
                    priv_dmin[li] = (float)vload_half(1, (global half*)bp);
                    global uchar * sc_g = (global uchar*)(bp + 4);
                    for (int i = 0; i < K_SCALE_SIZE; i++) {
                        priv_sc[li][i] = sc_g[i];
                    }
                }

                // Scale/min decode (get_scale_min_k4), using cached private data
                float scale = sg < 4
                    ? (float)(priv_sc[li][sg]   & 63)
                    : (float)((priv_sc[li][sg + 4] & 0x0F) | ((priv_sc[li][sg_m4] >> 6) << 4));
                float smin  = sg < 4
                    ? (float)(priv_sc[li][sg + 4] & 63)
                    : (float)((priv_sc[li][sg + 4] >>   4) | ((priv_sc[li][sg]   >> 6) << 4));
                float ds = priv_dall[li] * scale;
                float dm = priv_dmin[li] * smin;

                buf_a[(loadr_a * 8 + 0) * BM + loadc_a + l] = ds * (float)nibbles.s0 - dm;
                buf_a[(loadr_a * 8 + 1) * BM + loadc_a + l] = ds * (float)nibbles.s1 - dm;
                buf_a[(loadr_a * 8 + 2) * BM + loadc_a + l] = ds * (float)nibbles.s2 - dm;
                buf_a[(loadr_a * 8 + 3) * BM + loadc_a + l] = ds * (float)nibbles.s3 - dm;
                buf_a[(loadr_a * 8 + 4) * BM + loadc_a + l] = ds * (float)nibbles.s4 - dm;
                buf_a[(loadr_a * 8 + 5) * BM + loadc_a + l] = ds * (float)nibbles.s5 - dm;
                buf_a[(loadr_a * 8 + 6) * BM + loadc_a + l] = ds * (float)nibbles.s6 - dm;
                buf_a[(loadr_a * 8 + 7) * BM + loadc_a + l] = ds * (float)nibbles.s7 - dm;
            } else {
                buf_a[(loadr_a * 8 + 0) * BM + loadc_a + l] = 0.0f;
                buf_a[(loadr_a * 8 + 1) * BM + loadc_a + l] = 0.0f;
                buf_a[(loadr_a * 8 + 2) * BM + loadc_a + l] = 0.0f;
                buf_a[(loadr_a * 8 + 3) * BM + loadc_a + l] = 0.0f;
                buf_a[(loadr_a * 8 + 4) * BM + loadc_a + l] = 0.0f;
                buf_a[(loadr_a * 8 + 5) * BM + loadc_a + l] = 0.0f;
                buf_a[(loadr_a * 8 + 6) * BM + loadc_a + l] = 0.0f;
                buf_a[(loadr_a * 8 + 7) * BM + loadc_a + l] = 0.0f;
            }
        }

        // ---- Load tile of B (activations, F32) ----
        for (int l = 0; l < BN; l += loadstride_b) {
            if (ic * BN + loadc_b + l < ne11) {
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

        pos_b += BK / LOAD_VEC_B;

        // ---- Compute tile product ----
        // Unroll only the small inner loops (TM=4, TN=8 are compile-time constants).
        // Do NOT unroll the outer BK=32 loop — it causes register spill on Apple M1.
        for (int i = 0; i < BK; i++) {
            #pragma unroll
            for (int j = 0; j < TM; j++) cache_a[j] = buf_a[i * BM + th_r * TM + j];
            #pragma unroll
            for (int j = 0; j < TN; j++) cache_b[j] = buf_b[i * BN + th_c * TN + j];
            #pragma unroll
            for (int cc = 0; cc < TN; cc++) {
                #pragma unroll
                for (int cr = 0; cr < TM; cr++) {
                    sums[cc * TM + cr] = mad(cache_a[cr], cache_b[cc], sums[cc * TM + cr]);
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // ---- Write output ----
    const int dr      = ir * BM + th_r * TM;
    const int dc      = ic * BN + th_c * TN;
    const int offsets = batch_idx * batch_stride_d;

    for (int cc = 0; cc < TN; cc++) {
        for (int cr = 0; cr < TM; cr++) {
            if (dr + cr < ne01 && dc + cc < ne11) {
                dst[offsets + (dc + cc) * stride_d + dr + cr] = sums[cc * TM + cr];
            }
        }
    }
}
