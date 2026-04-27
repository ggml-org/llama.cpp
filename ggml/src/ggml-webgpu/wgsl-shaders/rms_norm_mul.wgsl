#ifdef OVERLAP

@group(0) @binding(0)
var<storage, read_write> rn_src: array<f32>;

@group(0) @binding(1)
var<storage, read_write> mul_src: array<f32>;

@group(0) @binding(2)
var<uniform> params: Params;

fn update(rn_src_offset: u32, dst_offset: u32, scale: f32, mul_src_offset: u32) {
    mul_src[dst_offset] = scale * rn_src[rn_src_offset] * mul_src[mul_src_offset];
}

#elif INPLACE

@group(0) @binding(0)
var<storage, read_write> rn_src: array<f32>;

@group(0) @binding(1)
var<storage, read_write> mul_src: array<f32>;

@group(0) @binding(2)
var<uniform> params: Params;

fn update(rn_src_offset: u32, dst_offset: u32, scale: f32, mul_src_offset: u32) {
    rn_src[dst_offset] = scale * rn_src[rn_src_offset] * mul_src[mul_src_offset];
}

#elif SRC_OVERLAP

@group(0) @binding(0)
var<storage, read_write> merged_src: array<f32>;

@group(0) @binding(1)
var<storage, read_write> dst: array<f32>;

@group(0) @binding(2)
var<uniform> params: Params;

fn update(rn_src_offset: u32, dst_offset: u32, scale: f32, mul_src_offset: u32) {
    dst[dst_offset] = scale * merged_src[rn_src_offset] * merged_src[mul_src_offset];
}

#else

@group(0) @binding(0)
var<storage, read_write> rn_src: array<f32>;

@group(0) @binding(1)
var<storage, read_write> mul_src: array<f32>;

@group(0) @binding(2)
var<storage, read_write> dst: array<f32>;

@group(0) @binding(3)
var<uniform> params: Params;

fn update(rn_src_offset: u32, dst_offset: u32, scale: f32, mul_src_offset: u32) {
    dst[dst_offset] = scale * rn_src[rn_src_offset] * mul_src[mul_src_offset];
}

#endif

struct Params {
    offset_rn_src: u32,
    offset_mul_src: u32,
    offset_merged_rn_src: u32,
    offset_merged_mul_src: u32,
    offset_dst: u32,

    stride_rn_src1: u32,
    stride_rn_src2: u32,
    stride_rn_src3: u32,

    stride_mul_src1: u32,
    stride_mul_src2: u32,
    stride_mul_src3: u32,

    stride_dst1: u32,
    stride_dst2: u32,
    stride_dst3: u32,

    mul_src_ne0: u32,
    mul_src_ne1: u32,
    mul_src_ne2: u32,
    mul_src_ne3: u32,

    ne0: u32,
    ne1: u32,
    ne2: u32,
    ne3: u32,

    eps: f32
};

var<workgroup> scratch: array<f32, WG_SIZE>;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {

    // one thread per row
    var i = wid.x;
    let i3 = i / (params.ne2 * params.ne1);
    i = i % (params.ne2 * params.ne1);
    let i2 = i / params.ne1;
    let i1 = i % params.ne1;
    let i_rn_src_row = params.offset_rn_src + params.offset_merged_rn_src + i3 * params.stride_rn_src3 + i2 * params.stride_rn_src2 + i1 * params.stride_rn_src1;
    let i_mul_src_row = params.offset_mul_src + params.offset_merged_mul_src + (i3 % params.mul_src_ne3) * params.stride_mul_src3 + (i2 % params.mul_src_ne2) * params.stride_mul_src2 + (i1 % params.mul_src_ne1) * params.stride_mul_src1;
    let i_dst_row = params.offset_dst + i3 * params.stride_dst3 + i2 * params.stride_dst2 + i1 * params.stride_dst1;

    let elems = (params.ne0 + WG_SIZE - 1) / WG_SIZE;

    var sum = 0.0f;
    var col = lid.x;
    for (var j: u32 = 0; j < elems; j++) {
        if (col >= params.ne0) {
            break;
        }
#ifdef SRC_OVERLAP
        sum += pow(merged_src[i_rn_src_row + col], 2.0);
#else
        sum += pow(rn_src[i_rn_src_row + col], 2.0);
#endif
        col += WG_SIZE;
    }

    scratch[lid.x] = sum;

    workgroupBarrier();

    var offset: u32 = WG_SIZE / 2;
    while (offset > 0) {
        if (lid.x < offset) {
            scratch[lid.x] += scratch[lid.x + offset];
        }
        offset = offset / 2;
        workgroupBarrier();
    }
    sum = scratch[0];

    let scale = 1.0/sqrt(sum/f32(params.ne0) + params.eps);

    col = lid.x;
    for (var j: u32 = 0; j < elems; j++) {
        if (col >= params.ne0) {
            break;
        }
        update(i_rn_src_row + col, i_dst_row + col, scale, i_mul_src_row + col % params.mul_src_ne0);
        col += WG_SIZE;
    }
}
