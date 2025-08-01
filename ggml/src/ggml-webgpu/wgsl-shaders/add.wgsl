enable f16;

@group(0) @binding(0)
var<storage, read_write> src0: array<f32>;

@group(0) @binding(1)
var<storage, read_write> src1: array<f32>;

@group(0) @binding(2)
var<storage, read_write> dst: array<f32>;

struct Params {
    ne: u32,             // total number of elements

    stride_src0_0: u32,
    stride_src0_1: u32,
    stride_src0_2: u32,
    stride_src0_3: u32,

    stride_src1_0: u32,
    stride_src1_1: u32,
    stride_src1_2: u32,
    stride_src1_3: u32,

    stride_dst_0: u32,
    stride_dst_1: u32,
    stride_dst_2: u32,
    stride_dst_3: u32,

    ne0: u32,
    ne1: u32,
    ne2: u32,
    ne3: u32,

    // offsets in elements
    offset_src0: u32,
    offset_src1: u32,
    offset_dst: u32,
};

@group(0) @binding(3)
var<uniform> params: Params;

override wg_size: u32;
@compute @workgroup_size(wg_size)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.ne) {
        return;
    }

    var i = gid.x; // i = thread id

    // compute indexes for each dimension of the tensor 
    let i3 = i / (params.ne2 * params.ne1 * params.ne0);
    i = i % (params.ne2 * params.ne1 * params.ne0);

    let i2 = i / (params.ne1 * params.ne0);
    i = i % (params.ne1 * params.ne0);

    let i1 = i / params.ne0;

    let i0 = i % params.ne0;

    // compute indexes for position in each flat array
    let src0_idx = i0 * params.stride_src0_0 + i1 * params.stride_src0_1 +
                   i2 * params.stride_src0_2 + i3 * params.stride_src0_3;

    let src1_idx = i0 * params.stride_src1_0 + i1 * params.stride_src1_1 +
                   i2 * params.stride_src1_2 + i3 * params.stride_src1_3;

    let dst_idx = i0 * params.stride_dst_0 + i1 * params.stride_dst_1 +
                  i2 * params.stride_dst_2 + i3 * params.stride_dst_3;


    // dst[dst_idx] = src0[src0_idx] + src1[src1_idx];

    dst[params.offset_dst + dst_idx] = src0[params.offset_src0 + src0_idx] + src1[params.offset_src1 + src1_idx];
}
