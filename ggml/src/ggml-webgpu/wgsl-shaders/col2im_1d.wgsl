@group(0) @binding(0)
var<storage, read_write> src: array<f32>;

@group(0) @binding(1)
var<storage, read_write> dst: array<f32>;

struct Params {
    ne: u32,
    offset_src: u32,
    offset_dst: u32,
    src_ne0: u32,
    t_in: u32,
    t_out: u32,
    kernel: u32,
    stride: u32,
    padding: u32,
};

@group(0) @binding(2)
var<uniform> params: Params;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(num_workgroups) num_wg: vec3<u32>,
        @builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + num_wg.x * WG_SIZE * gid.y;
    if (idx >= params.ne) {
        return;
    }

    let channel = idx / params.t_out;
    let t_abs = idx % params.t_out + params.padding;
    var first: u32 = 0;
    if (t_abs >= params.kernel) {
        first = (t_abs - params.kernel) / params.stride + 1;
    }
    let last = min(t_abs / params.stride, params.t_in - 1);

    // Gather overlapping columns in the same order as the CPU reference.
    var sum: f32 = 0.0;
    for (var t = first; t <= last; t++) {
        let k = t_abs - t * params.stride;
        sum += src[params.offset_src + t * params.src_ne0 + channel * params.kernel + k];
    }
    dst[params.offset_dst + idx] = sum;
}
