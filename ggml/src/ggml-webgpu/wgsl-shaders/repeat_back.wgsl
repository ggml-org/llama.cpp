// sum over the repeats of src0 that map to each dst element

struct Params {
    ne: u32,

    offset_src0: u32,
    offset_dst: u32,

    ne00: u32,
    ne01: u32,
    ne02: u32,
    ne03: u32,

    stride_src0_0: u32,
    stride_src0_1: u32,
    stride_src0_2: u32,
    stride_src0_3: u32,

    ne0: u32,
    ne1: u32,
    ne2: u32,
    ne3: u32,
};

@group(0) @binding(0)
var<storage, read_write> src0: array<f32>;

@group(0) @binding(1)
var<storage, read_write> dst: array<f32>;

@group(0) @binding(2)
var<uniform> params: Params;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups)       num_wg: vec3<u32>) {
    let idx = gid.x + (num_wg.x * u32(WG_SIZE)) * gid.y;
    if (idx >= params.ne) {
        return;
    }

    var i = idx;
    let i0 = i % params.ne0;
    i = i / params.ne0;
    let i1 = i % params.ne1;
    i = i / params.ne1;
    let i2 = i % params.ne2;
    let i3 = i / params.ne2;

    var acc = 0.0f;
    for (var s3 = i3; s3 < params.ne03; s3 += params.ne3) {
        for (var s2 = i2; s2 < params.ne02; s2 += params.ne2) {
            for (var s1 = i1; s1 < params.ne01; s1 += params.ne1) {
                for (var s0 = i0; s0 < params.ne00; s0 += params.ne0) {
                    acc += src0[params.offset_src0 + s3 * params.stride_src0_3 + s2 * params.stride_src0_2 +
                                s1 * params.stride_src0_1 + s0 * params.stride_src0_0];
                }
            }
        }
    }

    dst[params.offset_dst + idx] = acc;
}
