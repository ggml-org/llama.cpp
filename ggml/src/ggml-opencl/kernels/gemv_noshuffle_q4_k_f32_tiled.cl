// Tiled-wide q4_K GEMV for the long-vocab lm_head/embed (decode path).
//
// Pairs with kernel_convert_block_q4_k_tiled_ns (cvt.cl): the weights are laid
// out CANONICALLY (4-bit code in element order e in [0,256)) and TILED by 64
// output rows so the 64-thread lane group coalesces every weight load. Both the
// pack (convert) and the unpack (here) are owned by us -> correct by
// construction vs the reference ggml q4_K dequant. Same structure as the q6_K
// tiled GEMV; the only differences are the 4-bit dequant and the q4_K
// scale/min decode (get_scale_min_k4 from the packed 12-byte block).
//
// One work-item produces one output row. WG = {64 lanes, 4 subgroups}: the 64
// lanes cover the 64 rows of one tile (coalesced uint4 reads), the 4 subgroups
// split the K-blocks and reduce through __local at the end. Weights read from
// __global (lm_head is streamed once per token; texture cache caps it below the
// coalesced-global rate).

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64  __attribute__((qcom_reqd_sub_group_size("half")))
#endif

#define QK_K       256
#define NSUBGROUPS 4
#define TILE_ROWS  64

// Decode one q4_K sub-block scale + min from the packed 12-byte block.
// Identical to the o4 kernel's helper (masks hard-coded: d6=0x3F, d4=0x0F, hi2=0xC0).
inline void q4k_scale_min(int j, __global const uchar * q, uchar * d, uchar * m) {
    if (j < 4) {
        *d = q[j]   & 0x3F;
        *m = q[j+4] & 0x3F;
    } else {
        *d = (q[j+4] & 0x0F) | ((q[j-4] & 0xC0) >> 2);
        *m = ((q[j+4] >> 4) & 0x0F) | ((q[j] & 0xC0) >> 2);
    }
}

#if defined(ADRENO_GPU)
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_gemv_noshuffle_q4_k_f32_tiled(
    __global uint4 * src0_q,    // tiled: 8 uint4 granules / superblock (4-bit codes)
    __global half  * src0_d,    // tiled: 1 half / superblock
    __global half  * src0_dm,   // tiled: 1 half / superblock
    __global uchar * src0_s,    // tiled: 12 bytes / superblock (packed scales)
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

    int nb = ne00 / QK_K;               // superblocks per row

    float acc = 0.0f;

    for (int sb = grp; sb < nb; sb += NSUBGROUPS) {
        int tile_blk = rt * nb + sb;    // ne02 == 1 for lm_head/embed

        float dval  = (float)src0_d [tile_blk * TILE_ROWS + rit];
        float dmval = (float)src0_dm[tile_blk * TILE_ROWS + rit];

        // decode the 8 sub-block (scale, min) pairs
        __global uchar * sc = src0_s + (tile_blk * TILE_ROWS + rit) * 12;
        float scale[8], minv[8];
        #pragma unroll
        for (int is = 0; is < 8; ++is) {
            uchar sd, sm;
            q4k_scale_min(is, sc, &sd, &sm);
            scale[is] = dval  * (float)sd;
            minv[is]  = dmval * (float)sm;
        }

        // 32 uints of 4-bit codes (8 codes/uint), e-order
        uint q[32];
        #pragma unroll
        for (int g = 0; g < 8; ++g) {
            uint4 v = src0_q[(tile_blk * 8 + g) * TILE_ROWS + rit];
            q[g*4+0] = v.x; q[g*4+1] = v.y; q[g*4+2] = v.z; q[g*4+3] = v.w;
        }

        // dequant 256 codes in canonical e-order, MAC with activation.
        int act_base = sb * 64;         // activation float4 pixel base (256/4)
        #pragma unroll
        for (int e4 = 0; e4 < 64; ++e4) {
            float4 a = read_imagef(src1, act_base + e4);
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                int  e    = e4 * 4 + t;
                uint code = (q[e >> 3] >> ((e & 7) * 4)) & 0xF;
                int  is   = e >> 5;     // sub-block index = e/32
                float av  = (t == 0) ? a.x : (t == 1) ? a.y : (t == 2) ? a.z : a.w;
                acc += ((float)code * scale[is] - minv[is]) * av;
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
