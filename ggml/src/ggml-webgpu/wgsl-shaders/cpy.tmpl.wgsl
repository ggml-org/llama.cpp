#define(VARIANTS)

[
  {
    "REPLS": {
      "SRC_TYPE": "f32",
      "DST_TYPE": "f32"
    }
  },
  {
    "REPLS": {
      "SRC_TYPE": "f32",
      "DST_TYPE": "i32"
    }
  },
  {
    "REPLS": {
      "SRC_TYPE": "f32",
      "DST_TYPE": "f16"
    }
  },
  {
    "REPLS": {
      "SRC_TYPE": "f16",
      "DST_TYPE": "f16"
    }
  },
  {
    "REPLS": {
      "SRC_TYPE": "f16",
      "DST_TYPE": "f32"
    }
  }
]

#end(VARIANTS)

#define(SHADER)
enable f16;

@group(0) @binding(0)
var<storage, read_write> src: array<{{SRC_TYPE}}>;

@group(0) @binding(1)
var<storage, read_write> dst: array<{{DST_TYPE}}>;

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

    // Logical shapes
    src_ne0: u32,
    src_ne1: u32,
    src_ne2: u32,

    dst_ne0: u32,
    dst_ne1: u32,
    dst_ne2: u32
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
    let i3 = i / (params.src_ne2 * params.src_ne1 * params.src_ne0);
    i = i % (params.src_ne2 * params.src_ne1 * params.src_ne0);
    let i2 = i / (params.src_ne1 * params.src_ne0);
    i = i % (params.src_ne1 * params.src_ne0);
    let i1 = i / params.src_ne0;
    let i0 = i % params.src_ne0;

    var j = gid.x;
    let j3 = j / (params.dst_ne2 * params.dst_ne1 * params.dst_ne0);
    j = j % (params.dst_ne2 * params.dst_ne1 * params.dst_ne0);
    let j2 = j / (params.dst_ne1 * params.dst_ne0);
    j = j % (params.dst_ne1 * params.dst_ne0);
    let j1 = j / params.dst_ne0;
    let j0 = j % params.dst_ne0;

    let src_idx = i0 * params.stride_src0 + i1 * params.stride_src1 +
                  i2 * params.stride_src2 + i3 * params.stride_src3;

    let dst_idx = j0 * params.stride_dst0 + j1 * params.stride_dst1 +
                  j2 * params.stride_dst2 + j3 * params.stride_dst3;

    dst[params.offset_dst + dst_idx] = {{DST_TYPE}}((src[params.offset_src + src_idx]));
}
#end(SHADER)
