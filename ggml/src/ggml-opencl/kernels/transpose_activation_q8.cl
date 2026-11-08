#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable

#define tile_size 32

kernel void kernel_transpose_activation_q8(
        __global uint * src1_qa,
        __global uint * src1_qa_trans,
        __global half * src1_sa,
        __global float * src1_sa_trans,
        __global const int * total_tiles,
        uint  K
) {
    uint row_in_tile = get_global_id(0);
    uint blk_k = get_group_id(1);
    uint col_in_tile = get_local_id(1);
    uint tile_n = get_global_id(2);

    if (blk_k >= (K / tile_size) || tile_n >= (uint)total_tiles[0]) {
        return;
    }

    uint src_idx = (tile_n * tile_size + row_in_tile) * K / 4 + blk_k * 8 + col_in_tile;
    uint dst_idx = tile_n * tile_size * K / 4 + (blk_k * 8 + col_in_tile) * tile_size + row_in_tile;

    src1_qa_trans[dst_idx] = src1_qa[src_idx];

    if (col_in_tile == 0) {
        src_idx = (tile_n * tile_size + row_in_tile) * K / 32 + blk_k;
        dst_idx = tile_n * tile_size * K / 32 + blk_k * tile_size + row_in_tile;

        src1_sa_trans[dst_idx] = (float)(src1_sa[src_idx]);
    }

}
