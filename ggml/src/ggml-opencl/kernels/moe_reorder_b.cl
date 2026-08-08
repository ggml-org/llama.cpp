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
    // never reaches dst -- verified by filling them with 1e30 and re-running
    // test-backend-ops MUL_MAT_ID (383 OK / 0 FAIL, unchanged), against a positive
    // control that poisons the real gather too and fails 188 of 383.
    if (router_idx == 0xFFFFFFFF) {
        return;
    }

    ushort activation_idx = router_idx / map_ratio;
    dst[post_router_idx * K / 4 + k_4] = src[activation_idx * K / 4 + k_4];
}
