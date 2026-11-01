#if defined(SRC_F16) || defined(DST_F16)
enable f16;
#endif

#ifdef SRC_F16
#define SRC_TYPE f16
#else
#define SRC_TYPE f32
#endif

#ifdef DST_F16
#define DST_TYPE f16
#else
#define DST_TYPE f32
#endif

struct Params {
    width: u32,
    height: u32,
    num_channels: u32,
    batch_size: u32,
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

@group(0) @binding(0) var<storage, read_write> input: array<SRC_TYPE>;
#ifdef INPLACE
@group(0) @binding(1) var<uniform> params: Params;
#else
@group(0) @binding(1) var<storage, read_write> output: array<DST_TYPE>;
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
    @builtin(local_invocation_index) lane: u32
) {
    if (params.num_groups == 0u) {
        return;
    }
    let batch_index = workgroup_id.x / params.num_groups;
    if (batch_index >= params.batch_size) {
        return;
    }
    let group_index = workgroup_id.x % params.num_groups;
    let channels_per_group = params.num_channels / params.num_groups
        + select(0u, 1u, params.num_channels % params.num_groups != 0u);
    if (group_index > params.num_channels / max(channels_per_group, 1u)) {
        return;
    }
    let first_channel = group_index * channels_per_group;
    if (first_channel >= params.num_channels) {
        return;
    }
    let group_channels = min(channels_per_group, params.num_channels - first_channel);
    let num_elems_per_group = group_channels * params.width * params.height;
    let batch_base = params.input_offset + batch_index * params.batch_stride;
    let group_base = batch_base + first_channel * params.channel_stride;

    var partial_sum : f32 = 0.0f;
    for (var i = lane; i < num_elems_per_group; i += WG_SIZE) {
        partial_sum += f32(input[input_index(i, group_base)]);
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
        let delta = f32(input[input_index(i, group_base)]) - mean;
        partial_sq_dev += delta * delta;
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

    let scale = inverseSqrt(variance + params.epsilon);
#ifndef INPLACE
    let output_group_base = params.output_offset
        + batch_index * params.output_batch_stride
        + first_channel * params.output_channel_stride;
#endif
    for (var i = lane; i < num_elems_per_group; i += WG_SIZE) {
        let index = input_index(i, group_base);
        let normalized = (f32(input[index]) - mean) * scale;
#ifdef INPLACE
        input[index] = SRC_TYPE(normalized);
#else
        output[output_index(i, output_group_base)] = DST_TYPE(normalized);
#endif
    }


}
