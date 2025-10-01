#define(VARIANTS)

[
  {
    "SHADER_NAME": "add_f32",
    "REPLS": {
      "TYPE" : "f32",
      "OP": "+"
    },
    "DECLS": ["NOT_INPLACE"]
  },
  {
    "SHADER_NAME": "add_f16",
    "REPLS": {
      "TYPE" : "f16",
      "OP": "+"
    },
    "DECLS": ["NOT_INPLACE"]
  },
  {
    "SHADER_NAME": "add_f32_inplace",
    "REPLS": {
      "TYPE" : "f32",
      "OP": "+"
    },
    "DECLS": ["INPLACE"]
  },
  {
    "SHADER_NAME": "add_f16_inplace",
    "REPLS": {
      "TYPE" : "f16",
      "OP": "+"
    },
    "DECLS": ["INPLACE"]
  },
  {
    "SHADER_NAME": "mul_f32",
    "REPLS": {
      "TYPE" : "f32",
      "OP": "*"
    },
    "DECLS": ["NOT_INPLACE"]
  },
  {
    "SHADER_NAME": "mul_f16",
    "REPLS": {
      "TYPE" : "f16",
      "OP": "*"
    },
    "DECLS": ["NOT_INPLACE"]
  },
  {
    "SHADER_NAME": "mul_f32_inplace",
    "REPLS": {
      "TYPE" : "f32",
      "OP": "*"
    },
    "DECLS": ["INPLACE"]
  },
  {
    "SHADER_NAME": "mul_f16_inplace",
    "REPLS": {
      "TYPE" : "f16",
      "OP": "*"
    },
    "DECLS": ["INPLACE"]
  },
  {
    "SHADER_NAME": "sub_f32",
    "REPLS": {
      "TYPE" : "f32",
      "OP": "-"
    },
    "DECLS": ["NOT_INPLACE"]
  },
  {
    "SHADER_NAME": "sub_f16",
    "REPLS": {
      "TYPE" : "f16",
      "OP": "-"
    },
    "DECLS": ["NOT_INPLACE"]
  },
  {
    "SHADER_NAME": "sub_f32_inplace",
    "REPLS": {
      "TYPE" : "f32",
      "OP": "-"
    },
    "DECLS": ["INPLACE"]
  },
  {
    "SHADER_NAME": "sub_f16_inplace",
    "REPLS": {
      "TYPE" : "f16",
      "OP": "-"
    },
    "DECLS": ["INPLACE"]
  },
  {
    "SHADER_NAME": "div_f32",
    "REPLS": {
      "TYPE" : "f32",
      "OP": "/"
    },
    "DECLS": ["NOT_INPLACE"]
  },
  {
    "SHADER_NAME": "div_f16",
    "REPLS": {
      "TYPE" : "f16",
      "OP": "/"
    },
    "DECLS": ["NOT_INPLACE"]
  },
  {
    "SHADER_NAME": "div_f32_inplace",
    "REPLS": {
      "TYPE" : "f32",
      "OP": "/"
    },
    "DECLS": ["INPLACE"]
  },
  {
    "SHADER_NAME": "div_f16_inplace",
    "REPLS": {
      "TYPE" : "f16",
      "OP": "/"
    },
    "DECLS": ["INPLACE"]
  }
]

#end(VARIANTS)

#define(DECLS)

#decl(NOT_INPLACE)

fn update(dst_i: u32, src0_i: u32, src1_i: u32) {
    dst[dst_i] = src0[src0_i] {{OP}} src1[src1_i];
}

@group(0) @binding(2)
var<storage, read_write> dst: array<{{TYPE}}>;

@group(0) @binding(3)
var<uniform> params: Params;

#enddecl(NOT_INPLACE)

#decl(INPLACE)

fn update(dst_i: u32, src0_i: u32, src1_i: u32) {
    src0[dst_i] = src0[src0_i] {{OP}} src1[src1_i];
}

@group(0) @binding(2)
var<uniform> params: Params;

#enddecl(INPLACE)

#end(DECLS)


#define(SHADER)

enable f16;

#include "binary_head.tmpl"

@group(0) @binding(0)
var<storage, read_write> src0: array<{{TYPE}}>;

@group(0) @binding(1)
var<storage, read_write> src1: array<{{TYPE}}>;

DECLS

override wg_size: u32;
@compute @workgroup_size(wg_size)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < params.ne) {
        update(params.offset_dst + gid.x, params.offset_src0 + gid.x, params.offset_src1 + src1_index(gid.x));
    }
}

#end(SHADER)
