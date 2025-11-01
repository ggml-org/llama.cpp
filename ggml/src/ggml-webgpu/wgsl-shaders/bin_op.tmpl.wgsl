#define(VARIANTS)

[
  {
    "SHADER_NAME": "add_f32",
    "REPLS": {
      "TYPE" : "f32",
      "SRC1_BUF": "src1",
      "DST_BUF": "dst",
      "OP": "+",
      "PARAMS_BINDING": 3
    },
    "DECLS": ["NOT_INPLACE"]
  },
  {
    "SHADER_NAME": "add_f32_inplace",
    "REPLS": {
      "TYPE" : "f32",
      "SRC1_BUF": "src1",
      "DST_BUF": "src0",
      "OP": "+",
      "PARAMS_BINDING": 2
    },
    "DECLS": ["INPLACE"]
  },
  {
    "SHADER_NAME": "add_f32_overlap",
    "REPLS": {
      "TYPE" : "f32",
      "SRC1_BUF": "src0",
      "DST_BUF": "dst",
      "OP": "+",
      "PARAMS_BINDING": 2
    },
    "DECLS": ["OVERLAP"]
  },
  {
    "SHADER_NAME": "add_f32_inplace_overlap",
    "REPLS": {
      "TYPE" : "f32",
      "SRC1_BUF": "src0",
      "DST_BUF": "src0",
      "OP": "+",
      "PARAMS_BINDING": 1
    },
    "DECLS": ["INPLACE_OVERLAP"]
  },
  {
    "SHADER_NAME": "add_f16",
    "REPLS": {
      "TYPE" : "f16",
      "SRC1_BUF": "src1",
      "DST_BUF": "dst",
      "OP": "+",
      "PARAMS_BINDING": 3
    },
    "DECLS": ["NOT_INPLACE"]
  },
  {
    "SHADER_NAME": "add_f16_inplace",
    "REPLS": {
      "TYPE" : "f16",
      "SRC1_BUF": "src1",
      "DST_BUF": "src0",
      "OP": "+",
      "PARAMS_BINDING": 2
    },
    "DECLS": ["INPLACE"]
  },
  {
    "SHADER_NAME": "add_f16_overlap",
    "REPLS": {
      "TYPE" : "f16",
      "SRC1_BUF": "src0",
      "DST_BUF": "dst",
      "OP": "+",
      "PARAMS_BINDING": 2
    },
    "DECLS": ["OVERLAP"]
  },
  {
    "SHADER_NAME": "add_f16_inplace_overlap",
    "REPLS": {
      "TYPE" : "f16",
      "SRC1_BUF": "src0",
      "DST_BUF": "src0",
      "OP": "+",
      "PARAMS_BINDING": 1
    },
    "DECLS": ["INPLACE_OVERLAP"]
  },
  {
    "SHADER_NAME": "mul_f32",
    "REPLS": {
      "TYPE" : "f32",
      "SRC1_BUF": "src1",
      "DST_BUF": "dst",
      "OP": "*",
      "PARAMS_BINDING": 3
    },
    "DECLS": ["NOT_INPLACE"]
  },
  {
    "SHADER_NAME": "mul_f32_inplace",
    "REPLS": {
      "TYPE" : "f32",
      "SRC1_BUF": "src1",
      "DST_BUF": "src0",
      "OP": "*",
      "PARAMS_BINDING": 2
    },
    "DECLS": ["INPLACE"]
  },
  {
    "SHADER_NAME": "mul_f32_overlap",
    "REPLS": {
      "TYPE" : "f32",
      "SRC1_BUF": "src0",
      "DST_BUF": "dst",
      "OP": "*",
      "PARAMS_BINDING": 2
    },
    "DECLS": ["OVERLAP"]
  },
  {
    "SHADER_NAME": "mul_f32_inplace_overlap",
    "REPLS": {
      "TYPE" : "f32",
      "SRC1_BUF": "src0",
      "DST_BUF": "src0",
      "OP": "*",
      "PARAMS_BINDING": 1
    },
    "DECLS": ["INPLACE_OVERLAP"]
  },
  {
    "SHADER_NAME": "mul_f16",
    "REPLS": {
      "TYPE" : "f16",
      "SRC1_BUF": "src1",
      "DST_BUF": "dst",
      "OP": "*",
      "PARAMS_BINDING": 3
    },
    "DECLS": ["NOT_INPLACE"]
  },
  {
    "SHADER_NAME": "mul_f16_inplace",
    "REPLS": {
      "TYPE" : "f16",
      "SRC1_BUF": "src1",
      "DST_BUF": "src0",
      "OP": "*",
      "PARAMS_BINDING": 2
    },
    "DECLS": ["INPLACE"]
  },
  {
    "SHADER_NAME": "mul_f16_overlap",
    "REPLS": {
      "TYPE" : "f16",
      "SRC1_BUF": "src0",
      "DST_BUF": "dst",
      "OP": "*",
      "PARAMS_BINDING": 2
    },
    "DECLS": ["OVERLAP"]
  },
  {
    "SHADER_NAME": "mul_f16_inplace_overlap",
    "REPLS": {
      "TYPE" : "f16",
      "SRC1_BUF": "src0",
      "DST_BUF": "src0",
      "OP": "*",
      "PARAMS_BINDING": 1
    },
    "DECLS": ["INPLACE_OVERLAP"]
  },
  {
    "SHADER_NAME": "sub_f32",
    "REPLS": {
      "TYPE" : "f32",
      "SRC1_BUF": "src1",
      "DST_BUF": "dst",
      "OP": "-",
      "PARAMS_BINDING": 3
    },
    "DECLS": ["NOT_INPLACE"]
  },
  {
    "SHADER_NAME": "sub_f32_inplace",
    "REPLS": {
      "TYPE" : "f32",
      "SRC1_BUF": "src1",
      "DST_BUF": "src0",
      "OP": "-",
      "PARAMS_BINDING": 2
    },
    "DECLS": ["INPLACE"]
  },
  {
    "SHADER_NAME": "sub_f32_overlap",
    "REPLS": {
      "TYPE" : "f32",
      "SRC1_BUF": "src0",
      "DST_BUF": "dst",
      "OP": "-",
      "PARAMS_BINDING": 2
    },
    "DECLS": ["OVERLAP"]
  },
  {
    "SHADER_NAME": "sub_f32_inplace_overlap",
    "REPLS": {
      "TYPE" : "f32",
      "SRC1_BUF": "src0",
      "DST_BUF": "src0",
      "OP": "-",
      "PARAMS_BINDING": 1
    },
    "DECLS": ["INPLACE_OVERLAP"]
  },
  {
    "SHADER_NAME": "sub_f16",
    "REPLS": {
      "TYPE" : "f16",
      "SRC1_BUF": "src1",
      "DST_BUF": "dst",
      "OP": "-",
      "PARAMS_BINDING": 3
    },
    "DECLS": ["NOT_INPLACE"]
  },
  {
    "SHADER_NAME": "sub_f16_inplace",
    "REPLS": {
      "TYPE" : "f16",
      "SRC1_BUF": "src1",
      "DST_BUF": "src0",
      "OP": "-",
      "PARAMS_BINDING": 2
    },
    "DECLS": ["INPLACE"]
  },
  {
    "SHADER_NAME": "sub_f16_overlap",
    "REPLS": {
      "TYPE" : "f16",
      "SRC1_BUF": "src0",
      "DST_BUF": "dst",
      "OP": "-",
      "PARAMS_BINDING": 2
    },
    "DECLS": ["OVERLAP"]
  },
  {
    "SHADER_NAME": "sub_f16_inplace_overlap",
    "REPLS": {
      "TYPE" : "f16",
      "SRC1_BUF": "src0",
      "DST_BUF": "src0",
      "OP": "-",
      "PARAMS_BINDING": 1
    },
    "DECLS": ["INPLACE_OVERLAP"]
  },

  {
    "SHADER_NAME": "div_f32",
    "REPLS": {
      "TYPE" : "f32",
      "SRC1_BUF": "src1",
      "DST_BUF": "dst",
      "OP": "/",
      "PARAMS_BINDING": 3
    },
    "DECLS": ["NOT_INPLACE"]
  },
  {
    "SHADER_NAME": "div_f32_inplace",
    "REPLS": {
      "TYPE" : "f32",
      "SRC1_BUF": "src1",
      "DST_BUF": "src0",
      "OP": "/",
      "PARAMS_BINDING": 2
    },
    "DECLS": ["INPLACE"]
  },
  {
    "SHADER_NAME": "div_f32_overlap",
    "REPLS": {
      "TYPE" : "f32",
      "SRC1_BUF": "src0",
      "DST_BUF": "dst",
      "OP": "/",
      "PARAMS_BINDING": 2
    },
    "DECLS": ["OVERLAP"]
  },
  {
    "SHADER_NAME": "div_f32_inplace_overlap",
    "REPLS": {
      "TYPE" : "f32",
      "SRC1_BUF": "src0",
      "DST_BUF": "src0",
      "OP": "/",
      "PARAMS_BINDING": 1
    },
    "DECLS": ["INPLACE_OVERLAP"]
  },
  {
    "SHADER_NAME": "div_f16",
    "REPLS": {
      "TYPE" : "f16",
      "SRC1_BUF": "src1",
      "DST_BUF": "dst",
      "OP": "/",
      "PARAMS_BINDING": 3
    },
    "DECLS": ["NOT_INPLACE"]
  },
  {
    "SHADER_NAME": "div_f16_inplace",
    "REPLS": {
      "TYPE" : "f16",
      "SRC1_BUF": "src1",
      "DST_BUF": "src0",
      "OP": "/",
      "PARAMS_BINDING": 2
    },
    "DECLS": ["INPLACE"]
  },
  {
    "SHADER_NAME": "div_f16_overlap",
    "REPLS": {
      "TYPE" : "f16",
      "SRC1_BUF": "src0",
      "DST_BUF": "dst",
      "OP": "/",
      "PARAMS_BINDING": 2
    },
    "DECLS": ["OVERLAP"]
  },
  {
    "SHADER_NAME": "div_f16_inplace_overlap",
    "REPLS": {
      "TYPE" : "f16",
      "SRC1_BUF": "src0",
      "DST_BUF": "src0",
      "OP": "/",
      "PARAMS_BINDING": 1
    },
    "DECLS": ["INPLACE_OVERLAP"]
  }
]

#end(VARIANTS)

#define(DECLS)

#decl(NOT_INPLACE)

@group(0) @binding(1)
var<storage, read_write> src1: array<{{TYPE}}>;

@group(0) @binding(2)
var<storage, read_write> dst: array<{{TYPE}}>;

#enddecl(NOT_INPLACE)

#decl(INPLACE)

@group(0) @binding(1)
var<storage, read_write> src1: array<{{TYPE}}>;

#enddecl(INPLACE)

#decl(OVERLAP)

@group(0) @binding(1)
var<storage, read_write> dst: array<{{TYPE}}>;

#enddecl(OVERLAP)

#decl(INPLACE_OVERLAP)

#enddecl(INPLACE_OVERLAP)

#end(DECLS)

#define(SHADER)

enable f16;

struct Params {
    ne: u32,

    // offsets in elements
    offset_src0: u32,
    offset_src1: u32,
    offset_dst: u32,

    stride_src1_0: u32,
    stride_src1_1: u32,
    stride_src1_2: u32,
    stride_src1_3: u32,

    a_ne0: u32,
    a_ne1: u32,
    a_ne2: u32,

    b_ne0: u32,
    b_ne1: u32,
    b_ne2: u32,
    b_ne3: u32,
};

fn src1_index(_i: u32) -> u32 {
    var i = _i;
    let a_i3 = i / (params.a_ne2 * params.a_ne1 * params.a_ne0);
    i = i % (params.a_ne2 * params.a_ne1 * params.a_ne0);
    let a_i2 = i / (params.a_ne1 * params.a_ne0);
    i = i % (params.a_ne1 * params.a_ne0);
    let a_i1 = i / params.a_ne0;
    let a_i0 = i % params.a_ne0;

    // handle repetition of b
    // index loops back to the beginning and repeats after elements are exhausted = modulo
    let b_i0 = a_i0 % params.b_ne0;
    let b_i1 = a_i1 % params.b_ne1;
    let b_i2 = a_i2 % params.b_ne2;
    let b_i3 = a_i3 % params.b_ne3;

    // compute index for position in b's flat array
    return b_i0 * params.stride_src1_0 +
           b_i1 * params.stride_src1_1 +
           b_i2 * params.stride_src1_2 +
           b_i3 * params.stride_src1_3;
}

@group(0) @binding(0)
var<storage, read_write> src0: array<{{TYPE}}>;

@group(0) @binding({{PARAMS_BINDING}})
var<uniform> params: Params;

DECLS

override wg_size: u32;
@compute @workgroup_size(wg_size)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < params.ne) {
        {{DST_BUF}}[params.offset_dst + gid.x] = src0[params.offset_src0 + gid.x] {{OP}} {{SRC1_BUF}}[params.offset_src1 + src1_index(gid.x)];
    }
}

#end(SHADER)
