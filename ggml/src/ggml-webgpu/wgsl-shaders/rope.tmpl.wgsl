#define(VARIANTS)

[
  {
    "SHADER_SUFFIX": "f32_norm",
    "REPLS": {
      "TYPE" : "f32",
    },
    "DECLS": ["NO_FF_BINDINGS", "NO_FF_FUNC", "NORM", "ROTATE"]
  },
  {
    "SHADER_SUFFIX": "f32_norm_inplace",
    "REPLS": {
      "TYPE" : "f32",
    },
    "DECLS": ["NO_FF_BINDINGS_INPLACE", "NO_FF_FUNC", "NORM", "ROTATE_INPLACE"]
  },
  {
    "SHADER_SUFFIX": "f16_norm",
    "REPLS": {
      "TYPE" : "f16",
    },
    "DECLS": ["NO_FF_BINDINGS", "NO_FF_FUNC", "NORM", "ROTATE"]
  },
  {
    "SHADER_SUFFIX": "f16_norm_inplace",
    "REPLS": {
      "TYPE" : "f16",
    },
    "DECLS": ["NO_FF_BINDINGS_INPLACE", "NO_FF_FUNC", "NORM", "ROTATE_INPLACE"]
  },
  {
   "SHADER_SUFFIX": "f32_norm_ff",
    "REPLS": {
      "TYPE" : "f32",
    },
    "DECLS": ["FF_BINDINGS", "FF_FUNC", "NORM", "ROTATE"]
  },
  {
   "SHADER_SUFFIX": "f32_norm_ff_inplace",
    "REPLS": {
      "TYPE" : "f32",
    },
    "DECLS": ["FF_BINDINGS_INPLACE", "FF_FUNC", "NORM", "ROTATE_INPLACE"]
  },
  {
    "SHADER_SUFFIX": "f16_norm_ff",
    "REPLS": {
      "TYPE" : "f16",
    },
    "DECLS": ["FF_BINDINGS", "FF_FUNC", "NORM", "ROTATE"]
  },
  {
    "SHADER_SUFFIX": "f16_norm_ff_inplace",
    "REPLS": {
      "TYPE" : "f16",
    },
    "DECLS": ["FF_BINDINGS_INPLACE", "FF_FUNC", "NORM", "ROTATE_INPLACE"]
  },

  {
    "SHADER_SUFFIX": "f32_neox",
    "REPLS": {
      "TYPE" : "f32",
    },
    "DECLS": ["NO_FF_BINDINGS", "NO_FF_FUNC", "NEOX", "ROTATE"]
  },
  {
    "SHADER_SUFFIX": "f16_neox",
    "REPLS": {
      "TYPE" : "f16",
    },
    "DECLS": ["NO_FF_BINDINGS", "NO_FF_FUNC", "NEOX", "ROTATE"]
  },
  {
   "SHADER_SUFFIX": "f32_neox_ff",
    "REPLS": {
      "TYPE" : "f32",
    },
    "DECLS": ["FF_BINDINGS", "FF_FUNC", "NEOX", "ROTATE"]
  },
  {
    "SHADER_SUFFIX": "f16_neox_ff",
    "REPLS": {
      "TYPE" : "f16",
    },
    "DECLS": ["FF_BINDINGS", "FF_FUNC", "NEOX", "ROTATE"]
  }
]

#end(VARIANTS)

#define(DECLS)

#decl(ROTATE)
fn rotate(i_dst: u32, out0: f32, out1: f32) {
    dst[i_dst] = {{TYPE}}(out0);
    dst[i_dst + pair_offset()] = {{TYPE}}(out1);
}
#enddecl(ROTATE)

#decl(ROTATE_INPLACE)
fn rotate(i_dst: u32, out0: f32, out1: f32) {
    src0[i_dst] = {{TYPE}}(out0);
    src0[i_dst + pair_offset()] = {{TYPE}}(out1);
}
#enddecl(ROTATE_INPLACE)

#decl(NO_FF_FUNC)
fn freq_factor(i: u32) -> f32 {
    return 1.0f;
}
#enddecl(NO_FF_FUNC)

#decl(FF_FUNC)
fn freq_factor(i: u32) -> f32 {
    return src2[i/2];
}
#enddecl(FF_FUNC)

#decl(NO_FF_BINDINGS)

@group(0) @binding(2)
var<storage, read_write> dst: array<{{TYPE}}>;

@group(0) @binding(3)
var<uniform> params: Params;

#enddecl(NO_FF_BINDINGS)

#decl(NO_FF_BINDINGS_INPLACE)

@group(0) @binding(2)
var<uniform> params: Params;

#enddecl(NO_FF_BINDINGS_INPLACE)

#decl(FF_BINDINGS)

@group(0) @binding(2)
var<storage, read_write> src2: array<f32>;

@group(0) @binding(3)
var<storage, read_write> dst: array<{{TYPE}}>;

@group(0) @binding(4)
var<uniform> params: Params;

#enddecl(FF_BINDINGS)

#decl(FF_BINDINGS_INPLACE)

@group(0) @binding(2)
var<storage, read_write> src2: array<f32>;

@group(0) @binding(3)
var<uniform> params: Params;

#enddecl(FF_BINDINGS_INPLACE)

#decl(NORM)
fn pair_offset() -> u32 {
    return 1;
}
#enddecl(NORM)

#decl(NEOX)
fn pair_offset() -> u32 {
    return params.n_dims / 2;
}
#enddecl(NEOX)

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

    stride_dst1: u32,
    stride_dst2: u32,
    stride_dst3: u32,

    n_threads: u32,
    ne0: u32,
    ne1: u32,
    ne2: u32,

    n_dims: u32,
    theta_scale: f32,
    attn_factor: f32,
    freq_scale: f32,
    ext_factor: f32,
    corr_dims: vec2f
};


@group(0) @binding(0)
var<storage, read_write> src0: array<{{TYPE}}>;

@group(0) @binding(1)
var<storage, read_write> src1: array<i32>;

DECLS

fn rope_yarn_ramp(low: f32, high: f32, i: u32) -> f32 {
    let y = (f32(i / 2) - low) / max(0.001f, high - low);
    return 1.0f - min(1.0f, max(0.0f, y));
}

// returns vector of (cos_theta, sin_theta)
// TODO: check performance of instantiating once on the CPU and passed as buffer, since it's repeated per-row
fn rope_yarn(theta_extrap: f32, i: u32) -> vec2<f32> {
    var mscale = params.attn_factor;
    var theta = params.freq_scale * theta_extrap;
    if (params.ext_factor != 0.0f) {
        let ramp_mix = rope_yarn_ramp(params.corr_dims.x, params.corr_dims.y, i) * params.ext_factor;
        theta = theta * (1 - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * log(1.0f / params.freq_scale);
    }
    return vec2<f32>(cos(theta) * mscale, sin(theta) * mscale);
}

override wg_size: u32;
@compute @workgroup_size(wg_size)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // two elements per thread
    if (gid.x >= params.n_threads) {
        return;
    }

    var i = gid.x * 2; // start index for this thread
    let i3 = i / (params.ne2 * params.ne1 * params.ne0);
    i = i % (params.ne2 * params.ne1 * params.ne0);
    let i2 = i / (params.ne1 * params.ne0);
    i = i % (params.ne1 * params.ne0);
    let i1 = i / params.ne0;
    let i0 = i % params.ne0;

    let i_src = params.offset_src0 + i3 * params.stride_src03 + i2 * params.stride_src02 + i1 * params.stride_src01 + i0;
    let i_dst = params.offset_dst + i3 * params.stride_dst3 + i2 * params.stride_dst2 + i1 * params.stride_dst1 + i0;

    if (i0 >= params.n_dims) {
        rotate(i_dst, f32(src0[i_src]), f32(src0[i_src + 1]));
        return;
    }

    let theta_base = f32(src1[params.offset_src1 + i2]) * pow(params.theta_scale, f32(i0)/2.0f);
    let thetas = rope_yarn(theta_base/freq_factor(i0), i0);

    let x0 = f32(src0[i_src]);
    let x1 = f32(src0[i_src + pair_offset()]);
    rotate(i_dst, x0 * thetas.x - x1 * thetas.y, x0 * thetas.y + x1 * thetas.x);
}

#end(SHADER)
