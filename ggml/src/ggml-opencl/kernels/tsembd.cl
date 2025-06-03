kernel void kernel_timestep_embedding(
    global const void * p_timesteps,
    ulong off_timesteps,
    global void * p_dst,
    ulong off_dst,
    int dst_nb1_bytes,
    int logical_dim,
    int max_period
) {
    int local_i;
    int local_j;
    int local_half_dim;
    float local_timestep_val;
    float local_freq;
    float local_arg;
    global float * local_embed_data_ptr;
    global const float * local_timesteps_input_ptr;
    global float * local_dst_output_base_ptr;

    local_timesteps_input_ptr = (global const float *)((global char *)p_timesteps + off_timesteps);
    local_dst_output_base_ptr = (global float *)((global char *)p_dst + off_dst);

    local_i = get_global_id(1);
    local_j = get_global_id(0);

    local_half_dim = logical_dim / 2;
    local_embed_data_ptr = (global float *)((global char *)local_dst_output_base_ptr + local_i * dst_nb1_bytes);

    if (logical_dim % 2 != 0 && local_j == ((logical_dim + 1) / 2)) {
        local_embed_data_ptr[logical_dim] = 0.0f;
    }

    if (local_j >= local_half_dim) {
        return;
    }

    local_timestep_val = local_timesteps_input_ptr[local_i];

    if (local_half_dim == 0) {
        local_freq = 1.0f;
    } else {
        local_freq = exp(-log((float)max_period) * (float)local_j / (float)local_half_dim);
    }

    local_arg = local_timestep_val * local_freq;
    local_embed_data_ptr[local_j] = cos(local_arg);
    local_embed_data_ptr[local_j + local_half_dim] = sin(local_arg);
}
