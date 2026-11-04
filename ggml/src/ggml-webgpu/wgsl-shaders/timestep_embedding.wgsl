struct Params {
    dim: u32,
    max_period: f32,
    num_timesteps: u32,
    input_offset: u32,
    output_offset: u32,
};

@group(0) @binding(0)
var<storage, read_write> timesteps: array<f32>;

@group(0) @binding(1)
var<storage, read_write> output: array<f32>;

@group(0) @binding(2)
var<uniform> params: Params;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let global_thread_index = gid.x;

    let H: u32 = params.dim / 2;

    // If dim = 1, zero out every output since we expect dim >= 2 for well defined output
    if (params.dim == 1 && global_thread_index < params.num_timesteps) {
        output[params.output_offset + global_thread_index] = 0.0;
    }

    if (global_thread_index >= params.num_timesteps * H) { return; }

    let timestep_index: u32 = global_thread_index / H;
    let frequency_index: u32 = global_thread_index % H;
    
    let frequency: f32 = exp(-log(params.max_period) * f32(frequency_index) / f32(H));
    let angle: f32 = timesteps[params.input_offset + timestep_index] * frequency;

    output[params.output_offset + timestep_index * params.dim + frequency_index] = cos(angle);
    output[params.output_offset + timestep_index * params.dim + frequency_index + H] = sin(angle);

    // For odd dims, we zero out the last output via the first thread of each timestep block, 
    // since we expect cos(x) sin(x) pairs for each frequency.
    if (params.dim % 2 == 1 && frequency_index == 0) {
        output[params.output_offset + timestep_index * params.dim + (params.dim - 1)] = 0.0;
    }

}
