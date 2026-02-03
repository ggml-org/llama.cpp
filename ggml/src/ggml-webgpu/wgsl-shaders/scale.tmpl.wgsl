#define(VARIANTS)

[
  {
    "SHADER_NAME": "scale_f32",
    "DECLS": ["NOT_INPLACE"]
  },
  {
    "SHADER_NAME": "scale_f32_inplace",
    "DECLS": ["INPLACE"]
  }
]

#end(VARIANTS)

#define(DECLS)

#decl(NOT_INPLACE)
@group(0) @binding(1)
var<storage, read_write> dst: array<f32>;

@group(0) @binding(2)
var<uniform> params: Params;

fn store_scale(val: f32, offset: u32) {
    dst[offset] = val;
}
#enddecl(NOT_INPLACE)

#decl(INPLACE)
@group(0) @binding(1)
var<uniform> params: Params;

fn store_scale(val: f32, offset: u32) {
    src[offset] = val;
}
#enddecl(INPLACE)

#end(DECLS)

#define(SHADER)

struct Params {
    offset_src: u32,
    offset_dst: u32,

    // Strides (in elements)
    stride_src1: u32,
    stride_src2: u32,
    stride_src3: u32,

    stride_dst1: u32,
    stride_dst2: u32,
    stride_dst3: u32,

    ne: u32,
    ne0: u32,
    ne1: u32,
    ne2: u32,

    scale: f32,
    bias: f32
};

@group(0) @binding(0)
var<storage, read_write> src: array<f32>;

DECLS

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

    let i_src = params.offset_src + i3 * params.stride_src3 + i2 * params.stride_src2 + i1 * params.stride_src1 + i0;
    let i_dst = params.offset_dst + i3 * params.stride_dst3 + i2 * params.stride_dst2 + i1 * params.stride_dst1 + i0;

    store_scale(src[i_src] * params.scale + params.bias, i_dst);
}
#end(SHADER)
