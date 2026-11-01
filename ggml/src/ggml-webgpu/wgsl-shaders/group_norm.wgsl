
// override WG_SIZE: u32 = 4u; // num of threads in a workgroup

// Tenth milestone: Parameterize params.epsilon and group size
// Eleventh milestone: parameterize based on what ggml will actually give us.
struct Params {
    width: u32,
    height: u32,
    num_channels: u32,
    batch_size: u32,
    // number of groups to do group norm over, NOT workgroup size!
    num_groups: u32,
    epsilon: f32,
    // input strides
    row_stride: u32,
    col_stride: u32,
    channel_stride: u32,
    batch_stride: u32,
    // output strides
    output_row_stride: u32,
    output_col_stride: u32,
    output_channel_stride: u32,
    output_batch_stride: u32,
    // offsets
    input_offset: u32,
    output_offset: u32,
};

// Input buffer: TODO change dtype
@group(0) @binding(0) var<storage, read_write> input : array<f32>;
#ifdef INPLACE
@group(0) @binding(1) var<uniform> params: Params;
#else
@group(0) @binding(1) var<storage, read_write> output : array<f32>;
@group(0) @binding(2) var<uniform> params: Params;
#endif

var<workgroup> partials: array<f32, WG_SIZE>;
var<workgroup> mean: f32;
var<workgroup> variance: f32;

fn input_index(logical_position: u32, group_base: u32) -> u32 {
    let col: u32 = logical_position % params.width;
    let row: u32 = (logical_position / params.width) % params.height;
    let channel: u32 = (logical_position / params.width) / params.height;
    return group_base 
        + (row * params.row_stride) 
        + (col * params.col_stride)
        + (channel * params.channel_stride);
}

fn output_index(logical_position: u32, group_base: u32) -> u32 {
    let col: u32 = logical_position % params.width;
    let row: u32 = (logical_position / params.width) % params.height;
    let channel: u32 = (logical_position / params.width) / params.height;
    return group_base 
        + (row * params.output_row_stride) 
        + (col * params.output_col_stride)
        + (channel * params.output_channel_stride);
}

@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let batch_index = workgroup_id.x / params.num_groups;
    let group_index = workgroup_id.x % params.num_groups;
    
    let channels_per_group = u32(ceil(f32(params.num_channels) / f32(params.num_groups)));
    let first_channel = group_index * channels_per_group;
    let last_channel = min(first_channel + channels_per_group, params.num_channels);
    
    let num_elems_per_group = (last_channel - first_channel) * params.width * params.height;
    let batch_base = params.input_offset + batch_index * params.batch_stride;
    let group_base = batch_base + first_channel * params.channel_stride;

    if (first_channel >= params.num_channels) {
        return;
    }

    if (batch_index >= params.batch_size) {
        return;
    }

    var partial_sum : f32 = 0.0f;
    for (var i = lane; i < num_elems_per_group; i += WG_SIZE) {
        partial_sum += input[input_index(i, group_base)];
    }
    partials[lane] = partial_sum;

    workgroupBarrier();

    if (lane == 0) {
        var total_sum : f32 = 0.0f;
        for (var i: u32 = 0u; i < WG_SIZE; i++) {
            total_sum += partials[i];
        }
        mean = total_sum / f32(num_elems_per_group); 
    }

    workgroupBarrier();

    var partial_sq_dev : f32 = 0.0f;
    for (var i = lane; i < num_elems_per_group; i += WG_SIZE) {
        partial_sq_dev += (input[input_index(i, group_base)] - mean) 
            * (input[input_index(i, group_base)] - mean);
    }
    partials[lane] = partial_sq_dev;

    workgroupBarrier();

    if (lane == 0) {
        var total_sum : f32 = 0.0f;
        for (var i = 0u; i < WG_SIZE; i++) {
            total_sum += partials[i];
        }
        variance = total_sum / f32(num_elems_per_group);
    } 

    workgroupBarrier();

    var group_norm : f32 = 0.0f;
    for (var i = lane; i < num_elems_per_group; i += WG_SIZE) {
        let x = input[input_index(i, group_base)];
        let output_group_base = params.output_offset
            + batch_index * params.output_batch_stride
            + first_channel * params.output_channel_stride;
#ifdef INPLACE
        input[input_index(i, group_base)] = (x - mean) / (sqrt(variance + params.epsilon));
#else
        output[output_index(i, output_group_base)] = (x - mean) / (sqrt(variance + params.epsilon));
#endif
    }


}
