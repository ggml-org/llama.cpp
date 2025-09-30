#define(VARIANTS)

[
  {
    "SHADER_NAME": "reglu_f32",
    "REPLS": {
      "TYPE" : "f32",
    },
    "DECLS": ["NO_SPLIT", "REGLU"]
  },
  {
    "SHADER_NAME": "reglu_f32_split",
    "REPLS": {
      "TYPE" : "f32",
    },
    "DECLS": ["SPLIT", "REGLU"]
  },
  {
    "SHADER_NAME": "reglu_f16",
    "REPLS": {
      "TYPE" : "f16",
    },
    "DECLS": ["NO_SPLIT", "REGLU"]
  },
  {
    "SHADER_NAME": "reglu_f16_split",
    "REPLS": {
      "TYPE" : "f16",
    },
    "DECLS": ["SPLIT", "REGLU"]
  },
  {
    "SHADER_NAME": "geglu_f32",
    "REPLS": {
      "TYPE" : "f32",
    },
    "DECLS": ["NO_SPLIT", "GEGLU"]
  },
  {
    "SHADER_NAME": "geglu_f32_split",
    "REPLS": {
      "TYPE" : "f32",
    },
    "DECLS": ["SPLIT", "GEGLU"]
  },
  {
    "SHADER_NAME": "geglu_f16",
    "REPLS": {
      "TYPE" : "f16",
    },
    "DECLS": ["NO_SPLIT", "GEGLU"]
  },
  {
    "SHADER_NAME": "geglu_f16_split",
    "REPLS": {
      "TYPE" : "f16",
    },
    "DECLS": ["SPLIT", "GEGLU"]
  },
  {
    "SHADER_NAME": "swiglu_f32",
    "REPLS": {
      "TYPE" : "f32",
    },
    "DECLS": ["NO_SPLIT", "SWIGLU"]
  },
  {
    "SHADER_NAME": "swiglu_f32_split",
    "REPLS": {
      "TYPE" : "f32",
    },
    "DECLS": ["SPLIT", "SWIGLU"]
  },
  {
    "SHADER_NAME": "swiglu_f16",
    "REPLS": {
      "TYPE" : "f16",
    },
    "DECLS": ["NO_SPLIT", "SWIGLU"]
  },
  {
    "SHADER_NAME": "swiglu_f16_split",
    "REPLS": {
      "TYPE" : "f16",
    },
    "DECLS": ["SPLIT", "SWIGLU"]
  },
  {
    "SHADER_NAME": "swiglu_oai_f32",
    "REPLS": {
      "TYPE" : "f32",
    },
    "DECLS": ["NO_SPLIT", "SWIGLU_OAI"]
  },
  {
    "SHADER_NAME": "swiglu_oai_f32_split",
    "REPLS": {
      "TYPE" : "f32",
    },
    "DECLS": ["SPLIT", "SWIGLU_OAI"]
  },
  {
    "SHADER_NAME": "geglu_erf_f32",
    "REPLS": {
      "TYPE" : "f32",
    },
    "DECLS": ["NO_SPLIT", "GEGLU_ERF"]
  },
  {
    "SHADER_NAME": "geglu_erf_f32_split",
    "REPLS": {
      "TYPE" : "f32",
    },
    "DECLS": ["SPLIT", "GEGLU_ERF"]
  },
  {
    "SHADER_NAME": "geglu_erf_f16",
    "REPLS": {
      "TYPE" : "f16",
    },
    "DECLS": ["NO_SPLIT", "GEGLU_ERF"]
  },
  {
    "SHADER_NAME": "geglu_erf_f16_split",
    "REPLS": {
      "TYPE" : "f16",
    },
    "DECLS": ["SPLIT", "GEGLU_ERF"]
  },
  {
    "SHADER_NAME": "geglu_quick_f32",
    "REPLS": {
      "TYPE" : "f32",
    },
    "DECLS": ["NO_SPLIT", "GEGLU_QUICK"]
  },
  {
    "SHADER_NAME": "geglu_quick_f32_split",
    "REPLS": {
      "TYPE" : "f32",
    },
    "DECLS": ["SPLIT", "GEGLU_QUICK"]
  },
  {
    "SHADER_NAME": "geglu_quick_f16",
    "REPLS": {
      "TYPE" : "f16",
    },
    "DECLS": ["NO_SPLIT", "GEGLU_QUICK"]
  },
  {
    "SHADER_NAME": "geglu_quick_f16_split",
    "REPLS": {
      "TYPE" : "f16",
    },
    "DECLS": ["SPLIT", "GEGLU_QUICK"]
  },
]

#end(VARIANTS)

#define(DECLS)

#decl(REGLU)
fn op(a: {{TYPE}}, b: {{TYPE}}) -> {{TYPE}} {
    return max(a, 0) * b;
}
#enddecl(REGLU)

#decl(GEGLU)
const SQRT_2_OVER_PI: {{TYPE}} = 0.79788456080286535587989211986876;
const GELU_COEF_A: {{TYPE}} = 0.044715;

fn op(a: {{TYPE}}, b: {{TYPE}}) -> {{TYPE}} {
    let val = SQRT_2_OVER_PI * a * (1.0 + GELU_COEF_A * a * a);
    return 0.5 * a * (2.0 - 2.0 / (exp(2 * val) + 1)) * b;
}
#enddecl(GEGLU)

#decl(SWIGLU)
fn op(a: {{TYPE}}, b: {{TYPE}}) -> {{TYPE}} {
    return a / (1.0 + exp(-a)) * b;
}
#enddecl(SWIGLU)

#decl(SWIGLU_OAI)
fn op(a: f32, b: f32) -> f32 {
  let xi = min(a, params.limit);
  let gi = max(min(b, params.limit), -params.limit);
  var out_glu = xi / (1.0 + exp(-xi * params.alpha));
  out_glu = out_glu * (1.0 + gi);
  return out_glu;
}
#enddecl(SWIGLU_OAI)

#decl(GEGLU_ERF)
const p_erf: {{TYPE}} = 0.3275911;
const a1_erf: {{TYPE}} = 0.254829592;
const a2_erf: {{TYPE}} = -0.284496736;
const a3_erf: {{TYPE}} = 1.421413741;
const a4_erf: {{TYPE}} = -1.453152027;
const a5_erf: {{TYPE}} = 1.061405429;
const SQRT_2_INV: {{TYPE}} = 0.7071067811865476;

fn op(a: {{TYPE}}, b: {{TYPE}}) -> {{TYPE}} {
  let a_div_sqr2 = a * SQRT_2_INV;
  let sign_x = sign(a_div_sqr2);
  let x = abs(a_div_sqr2);
  let t = 1.0 / (1.0 + p_erf * x);
  let y = 1.0 - (((((a5_erf * t + a4_erf) * t + a3_erf) * t + a2_erf) * t + a1_erf) * t * exp(-x * x));
  let erf_approx = sign_x * y;
  return 0.5 * a * (1.0 + erf_approx) * b;
}
#enddecl(GEGLU_ERF)

#decl(GEGLU_QUICK)
const GELU_QUICK_COEF: {{TYPE}} = -1.702;

fn op(a: {{TYPE}}, b: {{TYPE}}) -> {{TYPE}} {
    return a * (1.0 / (1.0 + exp(GELU_QUICK_COEF * a))) * b;
}
#enddecl(GEGLU_QUICK)

#decl(NO_SPLIT)
@group(0) @binding(1)
var<storage, read_write> dst: array<{{TYPE}}>;

@group(0) @binding(2)
var<uniform> params: Params;

fn a_value(base: u32) -> {{TYPE}} {
    let offset: u32 = select(0, params.ne0, params.swapped != 0);
    return src0[base + offset];
}

fn b_value(base: u32) -> {{TYPE}} {
    let offset: u32 = select(params.ne0, 0, params.swapped != 0);
    return src0[base + offset];
}
#enddecl(NO_SPLIT)

#decl(SPLIT)
@group(0) @binding(1)
var<storage, read_write> src1: array<{{TYPE}}>;

@group(0) @binding(2)
var<storage, read_write> dst: array<{{TYPE}}>;

@group(0) @binding(3)
var<uniform> params: Params;

fn a_value(base: u32) -> {{TYPE}} {
    return src0[base];
}

fn b_value(base: u32) -> {{TYPE}} {
    return src1[base];
}
#enddecl(SPLIT)

#end(DECLS)

#define(SHADER)

enable f16;

struct Params {
    offset_src0: u32,
    offset_src1: u32,
    offset_dst: u32,

    // Strides (in elements)
    stride_src01: u32,
    stride_src02: u32,
    stride_src03: u32,

    stride_src11: u32,
    stride_src12: u32,
    stride_src13: u32,

    stride_dst1: u32,
    stride_dst2: u32,
    stride_dst3: u32,

    // shape of dst
    ne: u32,
    ne0: u32,
    ne1: u32,
    ne2: u32,

    swapped: u32,
    alpha: f32,
    limit: f32,
}

@group(0) @binding(0)
var<storage, read_write> src0: array<{{TYPE}}>;

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

    let i_a = params.offset_src0 + i3 * params.stride_src03 + i2 * params.stride_src02 + i1 * params.stride_src01 + i0;
    let i_b = params.offset_src1 + i3 * params.stride_src13 + i2 * params.stride_src12 + i1 * params.stride_src11 + i0;
    let i_dst = params.offset_dst + i3 * params.stride_dst3 + i2 * params.stride_dst2 + i1 * params.stride_dst1 + i0;

    dst[i_dst] = op(a_value(i_a), b_value(i_b));
}

#end(SHADER)
