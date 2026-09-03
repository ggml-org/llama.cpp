// Tiled-wide q6_K GEMV for the long-vocab lm_head/embed (decode path).
//
// Pairs with kernel_convert_block_q6_k_tiled_ns (cvt.cl): the weights are laid
// out CANONICALLY (6-bit code in element order e in [0,256)) and TILED by 64
// output rows so the 64-thread lane group coalesces every weight load. Both the
// pack (convert) and the unpack (here) are owned by us — correct by construction
// against the reference ggml q6_K dequant, no bit-interleave reverse-engineering.
//
// One work-item produces one output row. A work-group is {64 lanes, 4 subgroups}:
// the 64 lanes cover the 64 rows of one tile (coalesced reads), the 4 subgroups
// split the K-blocks and reduce through __local at the end.
//
// Weights are read from __global (coalesced) rather than image1d_buffer: the
// lm_head is read once per token with no reuse, and the Adreno texture cache
// caps such a streaming read well below the coalesced-global rate
// (see opencl_q6k_gemv_o4_shipped / x2-90 roofline notes).

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64  __attribute__((qcom_reqd_sub_group_size("half")))
#endif

#define NSUBGROUPS 4
#define TILE_ROWS  64

#if defined(ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_gemv_noshuffle_q6_K_f32_tiled(
    __global uint4 * src0_ql,   // tiled: 8 uint4 granules / superblock
    __global uint4 * src0_qh,   // tiled: 4 uint4 granules / superblock
    __global char  * src0_s,    // tiled: 16 chars / superblock
    __global half  * src0_d,    // tiled: 1 half  / superblock
    read_only image1d_buffer_t src1,   // activation (RGBA f32)
    global float * dst,
    ulong offsetd,
    int ne00,
    int ne01
) {
    int grp = get_local_id(1);          // subgroup index 0..3 (splits K)
    int row = get_global_id(0);         // output row along ne01
    int rt  = row / TILE_ROWS;
    int rit = row % TILE_ROWS;

    int nb = ne00 / 256;                // superblocks per row

    float acc = 0.0f;

    for (int sb = grp; sb < nb; sb += NSUBGROUPS) {
        int tile_blk = rt * nb + sb;    // ne02 == 1 for lm_head/embed

        // d + 16 scales for this (row, superblock)
        float dval = (float)src0_d[tile_blk * TILE_ROWS + rit];
        __global char * sc = src0_s + (tile_blk * TILE_ROWS + rit) * 16;

        // 32 ql-uints (8 codes/uint) + 16 qh-uints (16 codes/uint)
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

        // dequant 256 codes in canonical e-order, MAC with activation.
        int act_base = sb * 64;         // activation float4 pixel base (256/4)
        #pragma unroll
        for (int e4 = 0; e4 < 64; ++e4) {
            float4 a = read_imagef(src1, act_base + e4);
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                int  e    = e4 * 4 + t;
                uint low4 = (ql[e >> 3] >> ((e & 7) * 4)) & 0xF;
                uint hi2  = (qh[e >> 4] >> ((e & 15) * 2)) & 0x3;
                int  code = (int)(low4 | (hi2 << 4)) - 32;
                int  sidx = ((e >> 7) << 3) + (((e >> 5) & 3) << 1) + ((e >> 4) & 1);
                float scale = (float)sc[sidx] * dval;
                float av = (t == 0) ? a.x : (t == 1) ? a.y : (t == 2) ? a.z : a.w;
                acc += (float)code * scale * av;
            }
        }
    }

    // reduce across the NSUBGROUPS subgroups (same rit, different K-subset)
    local float reduce_lm[NSUBGROUPS * TILE_ROWS];
    reduce_lm[grp * TILE_ROWS + rit] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (grp == 0) {
        float total = reduce_lm[0 * TILE_ROWS + rit]
                    + reduce_lm[1 * TILE_ROWS + rit]
                    + reduce_lm[2 * TILE_ROWS + rit]
                    + reduce_lm[3 * TILE_ROWS + rit];
        dst = (global float*)((global char*)dst + offsetd);
        dst[row] = total;
    }
}

// Multi-column (N=3) variant of the tiled q6_K decode GEMV, for the speculative/
// MTP VERIFY lm_head/embed (ne1=3 = 2 drafts + 1 bonus). Identical tiled weight
// layout + unpack as the ne1=1 kernel above; each WI computes 3 output columns,
// streaming the (large) lm_head weight ONCE per superblock and reusing it across
// the 3 verify activation columns (dequant once per code, MAC into 3 accs). This
// is the lm_head analogue of the per-layer mc3 GEMV; the multiply order matches
// the ne1=1 kernel, so each column is byte-identical to a standalone tiled GEMV.
#if defined(ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_gemv_noshuffle_q6_K_f32_tiled_mc3(
    __global uint4 * src0_ql,
    __global uint4 * src0_qh,
    __global char  * src0_s,
    __global half  * src0_d,
    read_only image1d_buffer_t src1,
    global float * dst,
    ulong offsetd,
    int ne00,
    int ne01
) {
    int grp = get_local_id(1);
    int row = get_global_id(0);
    int rt  = row / TILE_ROWS;
    int rit = row % TILE_ROWS;

    int nb = ne00 / 256;
    int col_stride = ne00 / 4;          // activation float4 pixels per column

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f;

    for (int sb = grp; sb < nb; sb += NSUBGROUPS) {
        int tile_blk = rt * nb + sb;

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

        int act_base = sb * 64;
        #pragma unroll
        for (int e4 = 0; e4 < 64; ++e4) {
            float4 a0 = read_imagef(src1, 0*col_stride + act_base + e4);
            float4 a1 = read_imagef(src1, 1*col_stride + act_base + e4);
            float4 a2 = read_imagef(src1, 2*col_stride + act_base + e4);
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                int  e    = e4 * 4 + t;
                uint low4 = (ql[e >> 3] >> ((e & 7) * 4)) & 0xF;
                uint hi2  = (qh[e >> 4] >> ((e & 15) * 2)) & 0x3;
                int  code = (int)(low4 | (hi2 << 4)) - 32;
                int  sidx = ((e >> 7) << 3) + (((e >> 5) & 3) << 1) + ((e >> 4) & 1);
                float w   = (float)code * ((float)sc[sidx] * dval);  // dequant+scale once
                float av0 = (t == 0) ? a0.x : (t == 1) ? a0.y : (t == 2) ? a0.z : a0.w;
                float av1 = (t == 0) ? a1.x : (t == 1) ? a1.y : (t == 2) ? a1.z : a1.w;
                float av2 = (t == 0) ? a2.x : (t == 1) ? a2.y : (t == 2) ? a2.z : a2.w;
                acc0 += w * av0;
                acc1 += w * av1;
                acc2 += w * av2;
            }
        }
    }

    local float4 reduce_lm[NSUBGROUPS * TILE_ROWS];
    reduce_lm[grp * TILE_ROWS + rit] = (float4)(acc0, acc1, acc2, 0.0f);
    barrier(CLK_LOCAL_MEM_FENCE);

    if (grp == 0) {
        float4 total = reduce_lm[0 * TILE_ROWS + rit]
                     + reduce_lm[1 * TILE_ROWS + rit]
                     + reduce_lm[2 * TILE_ROWS + rit]
                     + reduce_lm[3 * TILE_ROWS + rit];
        dst = (global float*)((global char*)dst + offsetd);
        // dst column-major [ne01 rows x 3 cols]: (row, col) at col*ne01 + row
        dst[0*ne01 + row] = total.x;
        dst[1*ne01 + row] = total.y;
        dst[2*ne01 + row] = total.z;
    }
}
