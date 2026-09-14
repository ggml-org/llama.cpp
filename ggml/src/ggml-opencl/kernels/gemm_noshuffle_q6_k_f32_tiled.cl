// Batched (N>1) q6_K GEMM over the 64-row-TILED canonical layout produced by
// kernel_convert_block_q6_k_tiled_ns (cvt.cl). Companion to the decode kernel
// kernel_gemv_noshuffle_q6_K_f32_tiled: SAME pack, SAME canonical e-order
// dequant (correct by construction vs reference ggml q6_K), extended to N output
// columns. Makes the batched lm_head/embed (perplexity, spec-decode verify,
// batched serving) correct on GPU while keeping the tiled convert the fast decode
// GEMV depends on.
//
// One work-item owns one output ROW for a block of BN columns. A work-group is
// {64 lanes, NTILES subgroups} = NTILES*64 rows; the global z dimension tiles the
// N columns by BN. Each work-item computes its row's FULL K (no K-split, so no
// cross-subgroup reduction), which lets the whole work-group share one staged
// activation block:
//
//   __local activation staging — the BN columns of the current superblock (BN*256
//   floats) are loaded into __local once per superblock, cooperatively by all
//   NTILES*64 work-items, then every row reads its activation from __local. This
//   removes the ~Nrows-fold redundant image reads of the first version (each lane
//   re-read the activation), which made the batched GEMM ~2x slower than the plain
//   noshuffle GEMM.
//
// Weights are read from __global (coalesced) — matching the decode kernel; the
// lm_head weight is streamed with little reuse where coalesced global beats the
// Adreno texture cache.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64  __attribute__((qcom_reqd_sub_group_size("half")))
#endif

#define NTILES     4        // 64-row tiles per work-group (NTILES*64 = 256 rows)
#define TILE_ROWS  64
#define BN         16       // output columns handled per work-group (global z step)
#define WG_THREADS (NTILES * TILE_ROWS)

#if defined(ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_gemm_noshuffle_q6_K_f32_tiled(
    __global uint4 * src0_ql,   // tiled: 8 uint4 granules / superblock
    __global uint4 * src0_qh,   // tiled: 4 uint4 granules / superblock
    __global char  * src0_s,    // tiled: 16 chars / superblock
    __global half  * src0_d,    // tiled: 1 half  / superblock
    read_only image1d_buffer_t src1,   // activation [ne00, ne11] f32 (RGBA), column-major
    global float * dst,
    ulong offsetd,
    int ne00,
    int ne01,
    int ne11
) {
    int rit = get_local_id(0);          // 0..63  (lane within a tile; coalesces weight loads)
    int sg  = get_local_id(1);          // 0..NTILES-1
    int lid = sg * TILE_ROWS + rit;     // 0..WG_THREADS-1 (flat local id)
    int row = get_group_id(0) * WG_THREADS + lid;
    int rt  = row / TILE_ROWS;          // global 64-row tile index
    int col0 = get_global_id(2) * BN;   // first output column of this block

    int nb = ne00 / 256;                // superblocks per row
    int act_col_stride = ne00 / 4;      // activation float4 pixels per column

    const bool row_ok = row < ne01;

    // staged activation: BN columns x 256 elements for the current superblock
    __local float lact[BN * 256];

    float acc[BN];
    #pragma unroll
    for (int j = 0; j < BN; ++j) acc[j] = 0.0f;

    for (int sb = 0; sb < nb; ++sb) {
        // cooperatively stage BN columns' 256 activation elements (= BN*64 float4)
        for (int p = lid; p < BN * 64; p += WG_THREADS) {
            int j  = p >> 6;            // column within the BN block (p / 64)
            int e4 = p & 63;            // element-quad within the column (p % 64)
            int c  = col0 + j;
            float4 v = (c < ne11)
                ? read_imagef(src1, c * act_col_stride + sb * 64 + e4)
                : (float4)(0.0f);
            lact[p * 4 + 0] = v.x;
            lact[p * 4 + 1] = v.y;
            lact[p * 4 + 2] = v.z;
            lact[p * 4 + 3] = v.w;      // lact[j*256 + e], e = e4*4 + t
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (row_ok) {
            int tile_blk = rt * nb + sb;    // ne02 == 1 for lm_head/embed

            float dval = (float)src0_d[tile_blk * TILE_ROWS + rit];
            __global char * sc = src0_s + (tile_blk * TILE_ROWS + rit) * 16;

            uint ql[32];
            uint qh[16];
            #pragma unroll
            for (int g = 0; g < 8; ++g) {
                uint4 v = src0_ql[(tile_blk * 8 + g) * TILE_ROWS + rit];
                ql[g*4+0] = v.x; ql[g*4+1] = v.y; ql[g*4+2] = v.z; ql[g*4+3] = v.w;
            }
            #pragma unroll
            for (int g = 0; g < 4; ++g) {
                uint4 v = src0_qh[(tile_blk * 4 + g) * TILE_ROWS + rit];
                qh[g*4+0] = v.x; qh[g*4+1] = v.y; qh[g*4+2] = v.z; qh[g*4+3] = v.w;
            }

            // NOTE: the e loop (256) is deliberately NOT unrolled. Fully unrolling
            // 256*BN MACs overflows the in-process Adreno compiler (host stack
            // overflow at clBuildProgram, same class as the FA DK=512 OOM).
            for (int e = 0; e < 256; ++e) {
                uint low4 = (ql[e >> 3] >> ((e & 7) * 4)) & 0xF;
                uint hi2  = (qh[e >> 4] >> ((e & 15) * 2)) & 0x3;
                int  code = (int)(low4 | (hi2 << 4)) - 32;
                int  sidx = ((e >> 7) << 3) + (((e >> 5) & 3) << 1) + ((e >> 4) & 1);
                float cs  = (float)code * (float)sc[sidx] * dval;
                #pragma unroll
                for (int j = 0; j < BN; ++j) {
                    acc[j] += cs * lact[j * 256 + e];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row_ok) {
        dst = (global float*)((global char*)dst + offsetd);
        #pragma unroll
        for (int j = 0; j < BN; ++j) {
            int c = col0 + j;
            if (c < ne11) {
                dst[(ulong)c * ne01 + row] = acc[j];
            }
        }
    }
}
