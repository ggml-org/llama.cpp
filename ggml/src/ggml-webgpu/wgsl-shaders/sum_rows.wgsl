@group(0) @binding(0)
var<storage, read_write> src: array<f32>;

@group(0) @binding(1)
var<storage, read_write> dst: array<f32>;

struct Params {
    offset_src: u32, // in elements
    offset_dst: u32, // in elements

    // Strides (in elements)
    stride_src1: u32,
    stride_src2: u32,
    stride_src3: u32,

    ne0: u32,
    ne1: u32,
    ne2: u32
};

@group(0) @binding(2)
var<uniform> params: Params;

var<workgroup> shared_sum: array<f32, WG_SIZE>;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {

    var i = wid.x;
    let i3 = i / (params.ne2 * params.ne1);
    i = i % (params.ne2 * params.ne1);
    let i2 = i / params.ne1;
    let i1 = i % params.ne1;
    let i_src_row = params.offset_src + i3 * params.stride_src3 + i2 * params.stride_src2 + i1 * params.stride_src1;
    var local_sum: f32 = 0.0;
    for (var col = lid.x; col < params.ne0; col += WG_SIZE) {
        local_sum += src[i_src_row + col];
    }
    shared_sum[lid.x] = local_sum;
    workgroupBarrier();
    // reduce within workgroup
    var offset: u32 = WG_SIZE >> 1;
    while (offset > 0) {
        if (lid.x < offset) {
            shared_sum[lid.x] = shared_sum[lid.x] + shared_sum[lid.x + offset];
        }
        workgroupBarrier();
        offset >>= 1;
    }

    if (lid.x == 0) {
        dst[params.offset_dst + wid.x] = shared_sum[0];
    }
}
