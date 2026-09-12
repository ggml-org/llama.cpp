struct Params {
    ne: u32,

    offset_src0: u32,
    offset_src1: u32,
    offset_dst: u32,

    ne00: u32,
    ne01: u32,  // reduction dim, equal to src1 ne1
    ne02: u32,
    ne03: u32,

    stride_src0_0: u32,
    stride_src0_1: u32,
    stride_src0_2: u32,
    stride_src0_3: u32,

    ne10: u32,

    stride_src1_0: u32,
    stride_src1_1: u32,
    stride_src1_2: u32,
    stride_src1_3: u32,

    ne0: u32,
    ne1: u32,
    ne2: u32,
    ne3: u32,

    stride_dst_0: u32,
    stride_dst_1: u32,
    stride_dst_2: u32,
    stride_dst_3: u32,
};

@group(0) @binding(0)
var<storage, read_write> src0: array<f32>;

@group(0) @binding(1)
var<storage, read_write> src1: array<f32>;

@group(0) @binding(2)
var<storage, read_write> dst: array<f32>;

@group(0) @binding(3)
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

    // src0 is broadcast over dims 2 and 3
    let a_i0 = i0 % params.ne00;
    let a_i2 = i2 / (params.ne2 / params.ne02);
    let a_i3 = i3 / (params.ne3 / params.ne03);

    let b_i0 = i1 % params.ne10;

    let a_base = params.offset_src0 + a_i3 * params.stride_src0_3 + a_i2 * params.stride_src0_2 + a_i0 * params.stride_src0_0;
    let b_base = params.offset_src1 + i3 * params.stride_src1_3 + i2 * params.stride_src1_2 + b_i0 * params.stride_src1_0;

    var sum = 0.0f;
    for (var k: u32 = 0; k < params.ne01; k++) {
        sum += src0[a_base + k * params.stride_src0_1] * src1[b_base + k * params.stride_src1_1];
    }

    dst[params.offset_dst + i3 * params.stride_dst_3 + i2 * params.stride_dst_2 + i1 * params.stride_dst_1 + i0 * params.stride_dst_0] = sum;
}
