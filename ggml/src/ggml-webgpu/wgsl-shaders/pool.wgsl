struct Params {
    ne: u32,
    offset_src: u32,
    offset_dst: u32,
    si0: u32, si1: u32, si2: u32, si3: u32,
    iw: u32, ih: u32,
    ne0: u32, ne1: u32, ne2: u32,
    k0: i32, k1: i32,
    s0: i32, s1: i32,
    p0: i32, p1: i32,
};

@group(0) @binding(0)
var<storage, read_write> src: array<f32>;
@group(0) @binding(1)
var<storage, read_write> dst: array<f32>;
@group(0) @binding(2)
var<uniform> params: Params;

@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(num_workgroups) num_wg: vec3<u32>,
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x + num_wg.x * u32(WG_SIZE) * gid.y;
    if (idx >= params.ne) {
        return;
    }

    let i0 = idx % params.ne0;
    let i1 = (idx / params.ne0) % params.ne1;
    let i2 = (idx / (params.ne0 * params.ne1)) % params.ne2;
    let i3 = idx / (params.ne0 * params.ne1 * params.ne2);
    let ix = i32(i0) * params.s0 - params.p0;
    let begin_x = max(ix, 0);
    let end_x = min(ix + params.k0, i32(params.iw));

#ifdef POOL_2D
    let iy = i32(i1) * params.s1 - params.p1;
    let begin_y = max(iy, 0);
    let end_y = min(iy + params.k1, i32(params.ih));
    let base = params.offset_src + i2 * params.si2 + i3 * params.si3;
#else
    let begin_y = 0;
    let end_y = 1;
    let base = params.offset_src + i1 * params.si1 + i2 * params.si2 + i3 * params.si3;
#endif

#ifdef POOL_MAX
    var value = bitcast<f32>(0xff7fffffu);
#else
    var value = 0.0f;
#endif

    for (var y = begin_y; y < end_y; y++) {
        let row = base + u32(y) * params.si1;
        for (var x = begin_x; x < end_x; x++) {
            let v = src[row + u32(x) * params.si0];
#ifdef POOL_MAX
            value = max(v, value);
#else
            value += v;
#endif
        }
    }

#ifndef POOL_MAX
#ifdef POOL_2D
    value /= f32(params.k0 * params.k1);
#else
    if (end_x > begin_x) {
        value /= f32(end_x - begin_x);
    }
#endif
#endif

    dst[params.offset_dst + idx] = value;
}
