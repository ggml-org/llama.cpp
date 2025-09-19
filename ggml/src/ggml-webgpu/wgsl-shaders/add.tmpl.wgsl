#define(VARIANTS)

[
  {
    "REPLS": {
      "TYPE" : "f32",
    }
  },
  {
    "REPLS": {
      "TYPE" : "f16",
    }
  }
]

#end(VARIANTS)

#define(SHADER)

enable f16;

#include "binary_head.tmpl"

@group(0) @binding(0)
var<storage, read_write> src0: array<{{TYPE}}>;

@group(0) @binding(1)
var<storage, read_write> src1: array<{{TYPE}}>;

@group(0) @binding(2)
var<storage, read_write> dst: array<{{TYPE}}>;

@group(0) @binding(3)
var<uniform> params: Params;

override wg_size: u32;
@compute @workgroup_size(wg_size)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < params.ne) {
        dst[params.offset_dst + gid.x] = src0[params.offset_src0 + gid.x] + src1[params.offset_src1 + src1_index(gid.x)];
    }
}

#end(SHADER)
