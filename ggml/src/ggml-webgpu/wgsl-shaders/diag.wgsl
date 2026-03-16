@group(0) @binding(0)
var<storage, read_write> src: array<f32>;

@group(0) @binding(1)
var<storage, read_write> dst: array<f32>;

struct Params {
    ne: u32,
    offset_src: u32,
    offset_dst: u32,

    stride_src0: u32,
    stride_src2: u32,
    stride_src3: u32,

    src_ne0: u32,
    src_ne2: u32,
    src_ne3: u32,
};

@group(0) @binding(2)
var<uniform> params: Params;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.ne) {
        return;
    }

    var i = gid.x;
    let i3 = i / (params.src_ne2 * params.src_ne0 * params.src_ne0);
    i = i % (params.src_ne2 * params.src_ne0 * params.src_ne0);
    let i2 = i / (params.src_ne0 * params.src_ne0);
    i = i % (params.src_ne0 * params.src_ne0);
    let i1 = i / params.src_ne0;
    let i0 = i % params.src_ne0;

    if (i0 == i1) {
        let src_idx = params.offset_src + i0 * params.stride_src0 + i2 * params.stride_src2 + i3 * params.stride_src3;
        dst[params.offset_dst + gid.x] = src[src_idx];
    } else {
        dst[params.offset_dst + gid.x] = 0.0;
    }
}
