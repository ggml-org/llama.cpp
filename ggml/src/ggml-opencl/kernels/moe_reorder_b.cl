#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define QK4_0 32

kernel void kernel_moe_reorder_b(
    global float4 * src,
    global uint * router,
    global float4 * dst,
    global int * total_tiles,
    uint K,
    ushort map_ratio,
    uint tile_size
) {
    uint k_4 = get_global_id(0);
    uint post_router_idx = get_global_id(1);

    if ((k_4 >= (K / 4)) || (post_router_idx >= total_tiles[0] * tile_size)) {
        return;
    }

    uint router_idx = router[post_router_idx];

    // Padded slots need not be written at all. The MoE GEMMs accumulate per output
    // column and scatter only the real columns, so whatever sits in a padded slot
    // never reaches dst
    if (router_idx == 0xFFFFFFFF) {
        return;
    }

    ushort activation_idx = router_idx / map_ratio;
    dst[post_router_idx * K / 4 + k_4] = src[activation_idx * K / 4 + k_4];
}

// f16 variant for kernel_gemm_moe_mxfp4_f32_ns_coopmat, same arguments, one work-item per 8 k.
// The k order in each group of 8 is 0 4 1 5 2 6 3 7, the order of the coopmat mxfp4 dequant.
kernel void kernel_moe_reorder_b_f16(
    global float4 * src,
    global uint * router,
    global half * dst,
    global int * total_tiles,
    uint K,
    ushort map_ratio,
    uint tile_size
) {
    uint k_8 = get_global_id(0);
    uint post_router_idx = get_global_id(1);

    if ((k_8 >= (K / 8)) || (post_router_idx >= total_tiles[0] * tile_size)) {
        return;
    }

    // padded slots only affect their own (not stored) columns of C
    uint router_idx = router[post_router_idx];
    if (router_idx == 0xFFFFFFFF) {
        return;
    }

    uint activation_idx = router_idx / map_ratio;
    float4 a = src[activation_idx * K / 4 + k_8 * 2];
    float4 b = src[activation_idx * K / 4 + k_8 * 2 + 1];
    vstore8(convert_half8((float8)(a.x, b.x, a.y, b.y, a.z, b.z, a.w, b.w)), 0, dst + (ulong)post_router_idx * K + k_8 * 8);
}
