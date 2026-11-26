#define ARGMAX_LANES_PER_ROW 32
#define ARGMAX_ROWS_PER_GROUP 2
#define ARGMAX_BLOCK_SIZE (ARGMAX_LANES_PER_ROW * ARGMAX_ROWS_PER_GROUP)

kernel void kernel_argmax_f32_i32(
        global const char * src0,
        ulong               offset0,
        global char *       dst,
        ulong               offsetd,
        int                 ne00,
        int                 nrows) {
    const int local_id = get_local_id(0);
    const int col = local_id % ARGMAX_LANES_PER_ROW;
    const int row = get_group_id(1) * ARGMAX_ROWS_PER_GROUP + local_id / ARGMAX_LANES_PER_ROW;

    float max_value = -INFINITY;
    int max_index = 0;

    if (row < nrows) {
        global const float * src_row = (global const float *) (src0 + offset0) + row * ne00;
        for (int i = col; i < ne00; i += ARGMAX_LANES_PER_ROW) {
            const float value = src_row[i];
            if (value > max_value || (value == max_value && i > max_index)) {
                max_value = value;
                max_index = i;
            }
        }
    }

    local float max_values[ARGMAX_BLOCK_SIZE];
    local int max_indices[ARGMAX_BLOCK_SIZE];

    max_values[local_id] = max_value;
    max_indices[local_id] = max_index;

    barrier(CLK_LOCAL_MEM_FENCE);

    const int row_offset = local_id / ARGMAX_LANES_PER_ROW * ARGMAX_LANES_PER_ROW;
    for (int stride = ARGMAX_LANES_PER_ROW / 2; stride > 0; stride /= 2) {
        if (col < stride) {
            const float value = max_values[row_offset + col + stride];
            const int index = max_indices[row_offset + col + stride];
            if (value > max_values[row_offset + col] || (value == max_values[row_offset + col] && index > max_indices[row_offset + col])) {
                max_values[row_offset + col] = value;
                max_indices[row_offset + col] = index;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (col == 0 && row < nrows) {
        global int * dst_row = (global int *) (dst + offsetd) + row;
        dst_row[0] = max_indices[row_offset];
    }
}
