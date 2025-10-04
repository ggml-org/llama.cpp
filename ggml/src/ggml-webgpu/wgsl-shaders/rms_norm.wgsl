#define(VARIANTS)

[
  {
    "DECLS": ["NOT_INPLACE"]
  },
  {
    "SHADER_SUFFIX": "inplace",
    "DECLS": ["INPLACE"]
  },
]

#end(VARIANTS)

#define(DECLS)

#decl(NOT_INPLACE)

fn update(src_offset: u32, dst_offset: u32, scale: f32) {
    dst[dst_offset] = scale * src[src_offset];
}

@group(0) @binding(1)
var<storage, read_write> dst: array<f32>;

@group(0) @binding(2)
var<uniform> params: Params;

#enddecl(NOT_INPLACE)

#decl(INPLACE)

fn update(src_offset: u32, dst_offset: u32, scale: f32) {
    src[dst_offset] = scale * src[src_offset];
}

@group(0) @binding(1)
var<uniform> params: Params;

#enddecl(INPLACE)

#end(DECLS)

#define(SHADER)

struct Params {
    offset_src: u32, // in elements
    offset_dst: u32, // in elements

    // Strides (in elements)
    stride_src1: u32,
    stride_src2: u32,
    stride_src3: u32,

    stride_dst1: u32,
    stride_dst2: u32,
    stride_dst3: u32,

    // Shape of src/dst
    ne0: u32,
    ne1: u32,
    ne2: u32,
    ne3: u32,

    eps: f32
};

@group(0) @binding(0)
var<storage, read_write> src: array<f32>;

DECLS

override wg_size: u32;
var<workgroup> scratch: array<f32, wg_size>;

@compute @workgroup_size(wg_size)
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {

    // one thread per row
    var i = wid.x;
    let i3 = i / (params.ne2 * params.ne1);
    i = i % (params.ne2 * params.ne1);
    let i2 = i / params.ne1;
    let i1 = i % params.ne1;
    let i_src_row = params.offset_src + i3 * params.stride_src3 + i2 * params.stride_src2 + i1 * params.stride_src1;
    let i_dst_row = params.offset_dst + i3 * params.stride_dst3 + i2 * params.stride_dst2 + i1 * params.stride_dst1;

    let elems = (params.ne0 + wg_size - 1) / wg_size;

    var sum = 0.0f;
    var col = lid.x;
    for (var j: u32 = 0; j < elems; j++) {
        if (col >= params.ne0) {
            break;
        }
        sum += pow(src[i_src_row + col], 2.0);
        col += wg_size;
    }

    scratch[lid.x] = sum;
    workgroupBarrier();
    var offset = wg_size / 2;
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
        update(i_src_row + col, i_dst_row + col, scale);
        col += wg_size;
    }
}
#end(SHADER)
