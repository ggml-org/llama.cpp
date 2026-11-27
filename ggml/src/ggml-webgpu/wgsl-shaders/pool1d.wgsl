struct Params {
    ne: u32,
    offset_src: u32,
    offset_dst: u32,
    src_ne0: u32,
    dst_ne0: u32,
    k0: i32,
    s0: i32,
    p0: i32,
};

@group(0) @binding(0) var<storage, read_write> src: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(num_workgroups) num_wg: vec3<u32>, @builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x + num_wg.x * u32(WG_SIZE) * gid.y;
    if (idx >= params.ne) {
        return;
    }

    let row = params.offset_src + (idx / params.dst_ne0) * params.src_ne0;
    let x0 = i32(idx % params.dst_ne0) * params.s0 - params.p0;
    let x_begin = max(x0, 0);
    let x_end = min(x0 + params.k0, i32(params.src_ne0));
#ifdef POOL_MAX
    var value = bitcast<f32>(0xff7fffffu);
#else
    var value = 0.0f;
#endif
    for (var x = x_begin; x < x_end; x++) {
        let v = src[row + u32(x)];
#ifdef POOL_MAX
        value = max(value, v);
#else
        value += v;
#endif
    }
#ifndef POOL_MAX
    value /= f32(max(x_end - x_begin, 1));
#endif
    dst[params.offset_dst + idx] = value;
}
