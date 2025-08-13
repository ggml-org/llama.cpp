enable f16;

@group(0) @binding(0)
var<storage, read_write> src: array<f32>;

@group(0) @binding(1)
var<storage, read_write> dst: array<f16>;

struct Params {
    ne: u32,            // total number of elements
    offset_src: u32,    // in elements
    offset_dst: u32,    // in elements

    // Strides (in elements) — may be permuted
    stride_src0: u32,
    stride_src1: u32,
    stride_src2: u32,
    stride_src3: u32,

    stride_dst0: u32,
    stride_dst1: u32,
    stride_dst2: u32,
    stride_dst3: u32,

    // Logical shape (same for both tensors)
    ne0: u32,
    ne1: u32,
    ne2: u32,
    ne3: u32,
};

@group(0) @binding(2)
var<uniform> params: Params;

override wg_size: u32;
@compute @workgroup_size(wg_size)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.ne) {
        return;
    }

    var i = gid.x;

    let i3 = i / (params.ne2 * params.ne1 * params.ne0);
    i = i % (params.ne2 * params.ne1 * params.ne0);

    let i2 = i / (params.ne1 * params.ne0);
    i = i % (params.ne1 * params.ne0);

    let i1 = i / params.ne0;
    let i0 = i % params.ne0;

    let src_idx = i0 * params.stride_src0 + i1 * params.stride_src1 +
                  i2 * params.stride_src2 + i3 * params.stride_src3;

    let dst_idx = i0 * params.stride_dst0 + i1 * params.stride_dst1 +
                  i2 * params.stride_dst2 + i3 * params.stride_dst3;

    dst[params.offset_dst + dst_idx] = f16(src[params.offset_src + src_idx]);
}
