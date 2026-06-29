// log_soft_max over ne0 (per row), f32, contiguous rows.
// out = (x - row_max) - log(sum(exp(x - row_max))), matching the CPU/CUDA ops.

struct Params {
    offset_src0: u32,
    offset_dst: u32,

    // Strides (in elements)
    stride_src01: u32,
    stride_src02: u32,
    stride_src03: u32,

    stride_dst1: u32,
    stride_dst2: u32,
    stride_dst3: u32,

    // shape of src0/dst
    ne: u32,
    ne0: u32,
    ne1: u32,
    ne2: u32,
};

@group(0) @binding(0)
var<storage, read_write> src: array<f32>;

#ifdef INPLACE
@group(0) @binding(1)
var<uniform> params: Params;

fn update(i: u32, val: f32) {
    src[i] = val;
}
#else
@group(0) @binding(1)
var<storage, read_write> dst: array<f32>;
@group(0) @binding(2)
var<uniform> params: Params;

fn update(i: u32, val: f32) {
    dst[i] = val;
}
#endif

var<workgroup> scratch: array<f32, WG_SIZE>;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {

    var i = wid.x;
    let i3 = i / (params.ne2 * params.ne1);
    i = i % (params.ne2 * params.ne1);
    let i2 = i / params.ne1;
    let i1 = i % params.ne1;
    let i_src0_row = params.offset_src0 + i3 * params.stride_src03 + i2 * params.stride_src02 + i1 * params.stride_src01;
    let i_dst_row = params.offset_dst + i3 * params.stride_dst3 + i2 * params.stride_dst2 + i1 * params.stride_dst1;
    let elems = (params.ne0 + WG_SIZE - 1) / WG_SIZE;

    // parallel max over the row
    var max_val = -1e30;
    var col = lid.x;
    for (var j: u32 = 0; j < elems; j++) {
        if (col >= params.ne0) {
            break;
        }
        max_val = max(max_val, src[i_src0_row + col]);
        col += WG_SIZE;
    }

    scratch[lid.x] = max_val;
    workgroupBarrier();
    var offset: u32 = WG_SIZE / 2;
    while (offset > 0) {
        if (lid.x < offset) {
            scratch[lid.x] = max(scratch[lid.x], scratch[lid.x + offset]);
        }
        offset = offset / 2;
        workgroupBarrier();
    }
    let row_max = scratch[0];
    workgroupBarrier();

    // parallel sum of exp(x - row_max)
    var sum = 0.0f;
    col = lid.x;
    for (var j: u32 = 0; j < elems; j++) {
        if (col >= params.ne0) {
            break;
        }
        sum += exp(src[i_src0_row + col] - row_max);
        col += WG_SIZE;
    }

    scratch[lid.x] = sum;
    workgroupBarrier();
    offset = WG_SIZE / 2;
    while (offset > 0) {
        if (lid.x < offset) {
            scratch[lid.x] += scratch[lid.x + offset];
        }
        offset = offset / 2;
        workgroupBarrier();
    }
    let log_sum = log(scratch[0]);
    workgroupBarrier();

    // out = (x - row_max) - log_sum
    col = lid.x;
    for (var j: u32 = 0; j < elems; j++) {
        if (col >= params.ne0) {
            break;
        }
        update(i_dst_row + col, (src[i_src0_row + col] - row_max) - log_sum);
        col += WG_SIZE;
    }
}
