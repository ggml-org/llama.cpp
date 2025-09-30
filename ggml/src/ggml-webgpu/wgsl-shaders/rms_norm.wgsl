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
@compute @workgroup_size(wg_size)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.ne1 * params.ne2 * params.ne3) {
        return;
    }

    // one thread per row
    var i = gid.x;
    let i3 = i / (params.ne2 * params.ne1);
    i = i % (params.ne2 * params.ne1);
    let i2 = i / params.ne1;
    let i1 = i % params.ne1;
    let i_src_row = params.offset_src + i3 * params.stride_src3 + i2 * params.stride_src2 + i1 * params.stride_src1;
    let i_dst_row = params.offset_src + i3 * params.stride_dst3 + i2 * params.stride_dst2 + i1 * params.stride_dst1;

    var sum = 0.0f;
    for (var j: u32 = 0; j < params.ne0; j++) {
        sum += src[i_src_row + j] * src[i_src_row + j];
    }
    let scale = 1.0/sqrt(sum/f32(params.ne0) + params.eps);
    for (var j: u32 = 0; j < params.ne0; j++) {
        update(i_src_row + j, i_dst_row + j, scale);
    }
}
#end(SHADER)
