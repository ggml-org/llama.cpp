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

    float4 out = (float4)(0);
    if (router_idx != 0xFFFFFFFF) {
        ushort activation_idx = router_idx / map_ratio;
        out = src[activation_idx * K / 4 + k_4];
    }

    dst[post_router_idx * K / 4 + k_4] = out;
}
