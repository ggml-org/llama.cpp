@group(0) @binding(0)
var<storage, read_write> src: array<f32>;

@group(0) @binding(1)
var<storage, read_write> dst: array<f32>;

struct Params {
    ne: u32,            // total number of elements
    offset_src: u32,    // in elements
    offset_dst: u32,    // in elements

    // Logical shapes
    src_ne0: u32,
    src_ne1: u32,

    dst_ne0: u32,
    dst_ne1: u32,

    k0: u32,
    k1: u32,
    s0: u32,
    s1: u32,
    p0: u32,
    p1: u32
}


@group(0) @binding(2)
var<uniform> params: Params;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(num_workgroups) num_wg: vec3<u32>,
        @builtin(global_invocation_id) gid: vec3<u32>) {

    let gid_i = gid.x + (num_wg.x * u32(WG_SIZE)) * gid.y;
    if (gid_i >= params.ne) {
        return;
    }

    let out_x = gid_i % params.dst_ne0;
    let out_y = (gid_i / params.dst_ne0) % params.dst_ne1;
    let out_z = (gid_i / (params.dst_ne0 * params.dst_ne1));

    let x0 = i32(out_x * params.s0) - i32(params.p0);
    let y0 = i32(out_y * params.s1) - i32(params.p1);
    let in_x_start = max(0, x0);
    let in_x_end = min(i32(params.src_ne0), x0 + i32(params.k0));
    let in_y_start = max(0, y0);
    let in_y_end = min(i32(params.src_ne1), y0 + i32(params.k1));

    let base = params.offset_src + out_z * params.src_ne0 * params.src_ne1;

#ifdef POOL_MAX
    var res: f32 = bitcast<f32>(0xff7fffffu);
#else
    var res: f32 = 0.0;
#endif

    for(var y = in_y_start; y < in_y_end; y++){
        let row = base + u32(y) * params.src_ne0;
        for(var x = in_x_start; x < in_x_end; x++){
#ifdef POOL_MAX
                res = max(res, src[row + u32(x)]);
#else
                res += src[row + u32(x)];
#endif
        }
    }

#ifndef POOL_MAX
        res = res / f32(params.k0 * params.k1);
#endif
        dst[params.offset_dst + gid_i] = res;
}