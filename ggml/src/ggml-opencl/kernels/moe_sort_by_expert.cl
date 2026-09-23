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

// Deterministic replacement for kernel_moe_scatter.
//
// kernel_moe_scatter takes each token's slot from atomic_inc(slot_counter[expert]),
// so the token -> slot packing inside an expert depends on which work-item wins the
// atomic and changes from run to run. The ragged prefill GEMM path is sensitive to
// that packing (the non-ragged path is not, since its padded slots alias slot 0 and
// are overwritten last), which makes MoE prompt processing non-reproducible: the same
// binary on the same prompt returns one of several outputs.
//
// Here the slot is the token's rank in flat (n, k) order among the tokens routed to
// the same expert - a fixed function of the routing input. One workgroup per expert
// walks the flat routing list in blocks of 64 and ranks its own tokens with a
// workgroup scan, carrying a running count between blocks. Cost is one pass over the
// routing list per expert; the list is a few KiB and stays in cache.
__kernel void kernel_moe_scatter_stable(
    __global const int * input,
    __global int * post_router,
    __global ushort * emap,
    __global const int * tile_offset,
    int N,
    int topK,
    uint n_experts
) {
    const int e   = get_group_id(1);
    const int lid = get_local_id(0);
    const int M   = N * topK;

    __local int scan[64];
    __local int running;

    if (lid == 0) {
        running = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int base = 0; base < M; base += 64) {
        const int j = base + lid;

        int pred = 0;
        if (j < M) {
            const int n = j / topK;
            const int k = j - n * topK;
            pred = (input[n * (int)n_experts + k] == e) ? 1 : 0;
        }

        scan[lid] = pred;
        barrier(CLK_LOCAL_MEM_FENCE);

        // Hillis-Steele inclusive scan over the 64 lanes
        for (int off = 1; off < 64; off <<= 1) {
            int add = (lid >= off) ? scan[lid - off] : 0;
            barrier(CLK_LOCAL_MEM_FENCE);
            scan[lid] += add;
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (pred) {
            const int local_slot = running + (scan[lid] - 1);   // exclusive rank
            const int tile_idx   = tile_offset[e] + (local_slot >> 5);
            const int lane       = local_slot & 31;

            post_router[tile_idx * 32 + lane] = j;
            emap[tile_idx] = (ushort)e;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid == 63) {
            running += scan[63];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
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
