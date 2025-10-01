#define(VARIANTS)

[
  {
    "REPLS": {
      "TYPE" : "f32",
    },
    "DECLS": ["NO_FF_BINDINGS", "NO_FF_FUNC", "ROTATE"]
  },
  {
    "SHADER_SUFFIX": "f32_inplace",
    "REPLS": {
      "TYPE" : "f32",
    },
    "DECLS": ["NO_FF_BINDINGS_INPLACE", "NO_FF_FUNC", "ROTATE_INPLACE"]
  },
  {
    "REPLS": {
      "TYPE" : "f16",
    },
    "DECLS": ["NO_FF_BINDINGS", "NO_FF_FUNC", "ROTATE"]
  },
  {
    "SHADER_SUFFIX": "f16_inplace",
    "REPLS": {
      "TYPE" : "f16",
    },
    "DECLS": ["NO_FF_BINDINGS_INPLACE", "NO_FF_FUNC", "ROTATE_INPLACE"]
  },
  {
   "SHADER_SUFFIX": "f32_ff",
    "REPLS": {
      "TYPE" : "f32",
    },
    "DECLS": ["FF_BINDINGS", "FF_FUNC", "ROTATE"]
  },
  {
   "SHADER_SUFFIX": "f32_ff_inplace",
    "REPLS": {
      "TYPE" : "f32",
    },
    "DECLS": ["FF_BINDINGS_INPLACE", "FF_FUNC", "ROTATE_INPLACE"]
  },
  {
    "SHADER_SUFFIX": "f16_ff",
    "REPLS": {
      "TYPE" : "f16",
    },
    "DECLS": ["FF_BINDINGS", "FF_FUNC", "ROTATE"]
  },
  {
    "SHADER_SUFFIX": "f16_ff_inplace",
    "REPLS": {
      "TYPE" : "f16",
    },
    "DECLS": ["FF_BINDINGS_INPLACE", "FF_FUNC", "ROTATE_INPLACE"]
  }
]

#end(VARIANTS)

#define(DECLS)

#decl(ROTATE)
fn rotate(i_dst0: u32, i_dst1: u32, out0: f32, out1: f32) {
    dst[i_dst0] = {{TYPE}}(out0);
    dst[i_dst1] = {{TYPE}}(out1);
}
#enddecl(ROTATE)

#decl(ROTATE_INPLACE)
fn rotate(i_dst0: u32, i_dst1: u32, out0: f32, out1: f32) {
    src0[i_dst0] = {{TYPE}}(out0);
    src0[i_dst1] = {{TYPE}}(out1);
}
#enddecl(ROTATE_INPLACE)

#decl(NO_FF_FUNC)
fn freq_factor(i: u32) -> f32 {
    return 1.0f;
}
#enddecl(NO_FF_FUNC)

#decl(FF_FUNC)
fn freq_factor(i: u32) -> f32 {
    return src2[params.offset_src2 + i/2];
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

#end(DECLS)

#define(SHADER)

enable f16;

struct Params {
    offset_src0: u32,
    offset_src1: u32,
    offset_src2: u32,
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
    mode: u32,
    theta_scale: f32,
    attn_factor: f32,
    freq_scale: f32,
    ext_factor: f32,
    corr_dim0: f32,
    corr_dim1: f32,
    sections0: u32,
    sections1: u32,
    sections2: u32,
    sections3: u32
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
        let ramp_mix = rope_yarn_ramp(params.corr_dim0, params.corr_dim1, i) * params.ext_factor;
        theta = theta * (1 - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * log(1.0f / params.freq_scale);
    }
    return vec2<f32>(cos(theta) * mscale, sin(theta) * mscale);
}

fn pair_base(i0: u32, div_2: bool) -> u32 {
    if (div_2) {
        return i0 / 2;
    } else {
        return i0;
    }
}

fn pair_offset(is_neox: bool, is_mrope: bool, is_vision: bool) -> u32 {
    if (is_vision) {
        return params.n_dims;
    } else if (is_neox || is_mrope) {
        return params.n_dims / 2;
    } else {
        return 1;
    }
}

override wg_size: u32;
@compute @workgroup_size(wg_size)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // two elements per thread
    if (gid.x >= params.n_threads) {
        return;
    }

    let is_neox = bool(params.mode & 2);
    let is_mrope = bool(params.mode & 8);
    let is_vision = params.mode == 24;

    var i = gid.x * 2; // start index for this thread
    let i3 = i / (params.ne2 * params.ne1 * params.ne0);
    i = i % (params.ne2 * params.ne1 * params.ne0);
    let i2 = i / (params.ne1 * params.ne0);
    i = i % (params.ne1 * params.ne0);
    let i1 = i / params.ne0;
    let i0 = i % params.ne0;

    let i_src_row = params.offset_src0 + i3 * params.stride_src03 + i2 * params.stride_src02 + i1 * params.stride_src01;
    let i_dst_row = params.offset_dst + i3 * params.stride_dst3 + i2 * params.stride_dst2 + i1 * params.stride_dst1;

    if (i0 >= params.n_dims && !is_vision) {
        let i_src = i_src_row + i0;
        let i_dst = i_dst_row + i0;
        rotate(i_dst, i_dst + 1, f32(src0[i_src]), f32(src0[i_src + 1]));
        return;
    }

    var theta_base_mult: u32 = 0;
    var theta_scale_pwr: u32 = i0 / 2;
    if (is_mrope) {
        let sect_dims = params.sections0 + params.sections1 + params.sections2 + params.sections3;
        let sec_w = params.sections1 + params.sections0;
        let sec_e = params.sections2 + sec_w;
        let sector = (i0 / 2) % sect_dims;
        if (sector >= params.sections0 && sector < sec_w) {
            theta_base_mult = 1;
            if (is_vision) {
                theta_scale_pwr = sector - params.sections0;
            }
        } else if (sector >= sec_w && sector < sec_e) {
            theta_base_mult = 2;
            if (is_vision) {
                theta_scale_pwr = sector - sec_w;
            }
        } else if (sector >= sec_e) {
            if (is_vision) {
                theta_scale_pwr = sector - sec_e;
                theta_scale_pwr = (i0 / 2) % sec_e;
            }
            theta_base_mult = 3;
        } else if (is_vision) {
            theta_scale_pwr = sector;
        }
    }
    let theta_base = f32(src1[params.offset_src1 + i2 + params.ne2 * theta_base_mult]) * pow(params.theta_scale, f32(theta_scale_pwr));
    let thetas = rope_yarn(theta_base/freq_factor(i0), i0);

    let i_src = i_src_row + pair_base(i0, is_neox || is_mrope || is_vision);
    let i_dst = i_dst_row + pair_base(i0, is_neox || is_mrope || is_vision);

    let x0 = f32(src0[i_src]);
    let x1 = f32(src0[i_src + pair_offset(is_neox, is_mrope, is_vision)]);
    rotate(i_dst, i_dst + pair_offset(is_neox, is_mrope, is_vision), x0 * thetas.x - x1 * thetas.y, x0 * thetas.y + x1 * thetas.x);
}

#end(SHADER)
