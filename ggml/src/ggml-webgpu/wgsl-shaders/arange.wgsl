@group(0) @binding(0)
var<storage, read_write> dst: array<f32>;

struct Params {
    ne: u32,
    offset_dst: u32,
    start: f32,
    step: f32,
};

@group(0) @binding(1)
var<uniform> params: Params;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(num_workgroups) num_wg: vec3<u32>, @builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x + (num_wg.x * u32(WG_SIZE)) * gid.y;
    if (i >= params.ne) {
        return;
    }

    dst[params.offset_dst + i] = params.start + params.step * f32(i);
}
