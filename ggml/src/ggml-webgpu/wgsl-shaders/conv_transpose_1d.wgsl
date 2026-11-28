enable f16;

@group(0) @binding(0)
var<storage, read_write> kernel: array<KERNEL_TYPE>;
@group(0) @binding(1)
var<storage, read_write> input: array<f32>;
@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

struct Params {
    kernel_offset: u32,
    input_offset: u32,
    output_offset: u32,
    kernel_size: u32,
    output_channels: u32,
    input_channels: u32,
    input_length: u32,
    output_length: u32,
    stride: u32,
    phases: u32,
    time_tiles: u32,
    total_tiles: u32,
};

@group(0) @binding(3)
var<uniform> params: Params;

var<workgroup> input_tile: array<array<f32, TILE>, TILE>;
var<workgroup> kernel_tile: array<array<f32, TILE>, TILE>;

@compute @workgroup_size(TILE, TILE)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(num_workgroups) groups: vec3<u32>,
        @builtin(local_invocation_id) lane: vec3<u32>) {
    let tile = group.x + groups.x * group.y;
    if (tile >= params.total_tiles) {
        return;
    }

    // Each tile covers one residue modulo stride, so its columns share the same kernel taps.
    let time_base = tile % params.time_tiles * TILE;
    let phase = tile / params.time_tiles % params.phases;
    let channel_base = tile / params.time_tiles / params.phases * TILE;
    let row = channel_base + lane.y;
    let weight_row = channel_base + lane.x;
    let time = time_base + lane.x;

    var sum = 0.0;
    if (phase < params.kernel_size) {
        let first = time_base + 1u - min(time_base + 1u, params.input_length);
        let last = min(time_base + TILE - 1u, (params.kernel_size - 1u - phase) / params.stride);
        // Visit overlapping input positions in ascending order.
        for (var step = last + 1u; step > first; ) {
            step -= 1u;
            let tap = phase + step * params.stride;
            let valid_time = time >= step && time - step < params.input_length;
            var dot = 0.0;
            for (var base = 0u; base < params.input_channels; base += TILE) {
                let col = base + lane.y;
                var value = 0.0;
                var weight = 0.0;
                if (col < params.input_channels) {
                    if (valid_time) {
                        value = input[params.input_offset + col * params.input_length + time - step];
                    }
                    if (weight_row < params.output_channels) {
                        let index = (col * params.output_channels + weight_row) * params.kernel_size + tap;
                        weight = f32(kernel[params.kernel_offset + index]);
                    }
                }
                input_tile[lane.y][lane.x] = value;
                kernel_tile[lane.y][lane.x] = weight;
                workgroupBarrier();
                for (var k = 0u; k < TILE; k++) {
                    dot = fma(input_tile[k][lane.x], kernel_tile[k][lane.y], dot);
                }
                workgroupBarrier();
            }
            if (valid_time) {
                sum += dot;
            }
        }
    }

    if (row < params.output_channels && time <= (params.output_length - 1u - phase) / params.stride) {
        let out_time = time * params.stride + phase;
        output[params.output_offset + row * params.output_length + out_time] = sum;
    }
}
