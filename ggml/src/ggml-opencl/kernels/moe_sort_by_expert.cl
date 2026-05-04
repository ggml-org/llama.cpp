#pragma OPENCL EXTENSION cl_khr_fp16 : enable

__kernel void kernel_moe_histogram(
    __global const int * input,
    __global int * hist,
    uint N,
    uint topK,
    uint n_experts
) {
    uint n = get_global_id(0);
    uint k = get_global_id(1);

    if (n >= N || k >= topK) {
        return;
    }

    int expert_id = input[n * n_experts + k];
    atomic_inc(&hist[expert_id]);
}

__kernel void kernel_moe_scan(
    __global int * hist,
    __global int * tile_offset,
    __global int * total_tiles,
    __global int * slot_counter,
    int tile_size,
    uint n_experts
) {
    int offset = 0;
    for (int v = 0; v < n_experts; v++) {
        int count = hist[v];
        int tiles = (count + tile_size - 1) / tile_size;
        tile_offset[v] = offset;
        offset += tiles;
        hist[v] = 0;
        slot_counter[v] = 0;
    }

    *total_tiles = offset;
}

__kernel void kernel_moe_scatter(
    __global const int * input,
    __global int * post_router,
    __global ushort * emap,
    __global const int * tile_offset,
    __global int * slot_counter,
    int N,
    int topK,
    uint n_experts
) {
    uint n = get_global_id(0);
    uint k = get_global_id(1);

    if (n >= N || k >= topK) {
        return;
    }

    int val = input[n * n_experts + k];

    int local_slot = atomic_inc(&slot_counter[val]);

    int tile_idx  = tile_offset[val] + (local_slot / 32);
    int lane      = local_slot % 32;
    int out_pos   = tile_idx * 32 + lane;

    post_router[out_pos] = n * topK + k;
    emap[tile_idx] = val;
}

__kernel void kernel_moe_fill(
    __global int * post_router,
    __global int * total_tiles,
    int tile_size
) {
    int tile_id = get_global_id(0);
    int vec_id_in_tile = get_global_id(1);

    if (tile_id < total_tiles[0]) {
        post_router[tile_id * tile_size + vec_id_in_tile] = 0xFFFFFFFF;
    }
}
