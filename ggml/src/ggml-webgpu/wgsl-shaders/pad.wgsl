@group(0) @binding(0)
var<storage, read_write> src: array<f32>;

@group(0) @binding(1)
var<storage, read_write> dst: array<f32>;

struct Params {
    ne: u32,            // total number of elements
    offset_src: u32,    // in elements
    offset_dst: u32,    // in elements

    // Strides (in elements)
    stride_src0: u32,
    stride_src1: u32,
    stride_src2: u32,
    stride_src3: u32,

    // Logical shapes
    src_ne0: u32,
    src_ne1: u32,
    src_ne2: u32,
    src_ne3: u32,

    dst_ne0: u32,
    dst_ne1: u32,
    dst_ne2: u32,
    dst_ne3: u32,

    // Pad sizes (in elements)
    lp0: u32,
    rp0: u32,
    lp1: u32,
    rp1: u32,
    lp2: u32,
    rp2: u32,
    lp3: u32,
    rp3: u32,
};

@group(0) @binding(2)
var<uniform> params: Params;

fn wrap_around(idx: i32, n: u32) -> u32 {
    return u32(idx + i32(n)) % n;
}

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.ne) {
        return;
    }

    var i = gid.x;
    let dst_plane = params.dst_ne2 * params.dst_ne1 * params.dst_ne0;
    let i3 = i / dst_plane;
    i = i % dst_plane;
    let i2 = i / (params.dst_ne1 * params.dst_ne0);
    i = i % (params.dst_ne1 * params.dst_ne0);
    let i1 = i / params.dst_ne0;
    let i0 = i % params.dst_ne0;

    var value: f32 = 0.0;

#ifdef CIRCULAR
    let ci0 = wrap_around(i32(i0) - i32(params.lp0), params.src_ne0);
    let ci1 = wrap_around(i32(i1) - i32(params.lp1), params.src_ne1);
    let ci2 = wrap_around(i32(i2) - i32(params.lp2), params.src_ne2);
    let ci3 = wrap_around(i32(i3) - i32(params.lp3), params.src_ne3);
    let circular_src_idx = ci0 * params.stride_src0 + ci1 * params.stride_src1 +
                           ci2 * params.stride_src2 + ci3 * params.stride_src3;
    value = src[params.offset_src + circular_src_idx];
#else
    let is_src =
        (i0 >= params.lp0 && i0 < params.dst_ne0 - params.rp0) &&
        (i1 >= params.lp1 && i1 < params.dst_ne1 - params.rp1) &&
        (i2 >= params.lp2 && i2 < params.dst_ne2 - params.rp2) &&
        (i3 >= params.lp3 && i3 < params.dst_ne3 - params.rp3);
    if (is_src) {
        let src_idx = (i0 - params.lp0) * params.stride_src0 + (i1 - params.lp1) * params.stride_src1 +
                      (i2 - params.lp2) * params.stride_src2 + (i3 - params.lp3) * params.stride_src3;
        value = src[params.offset_src + src_idx];
    }
#endif

    dst[params.offset_dst + gid.x] = value;
}
