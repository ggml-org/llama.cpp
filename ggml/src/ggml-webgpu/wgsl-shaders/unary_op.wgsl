#define(VARIANTS)

[
  {
    "SHADER_NAME": "abs_f32",
    "REPLS": { "TYPE": "f32", "FUNC": "dst[dst_i] = abs(src[src_i]);", "EXT_PARAMS": "" },
    "DECLS": ["NOT_INPLACE"]
  },
  {
    "SHADER_NAME": "abs_f16",
    "REPLS": { "TYPE": "f16", "FUNC": "dst[dst_i] = abs(src[src_i]);", "EXT_PARAMS": "" },
    "DECLS": ["NOT_INPLACE"]
  },
  {
    "SHADER_NAME": "abs_in_place_f32",
    "REPLS": { "TYPE": "f32", "FUNC": "src[dst_i] = abs(src[src_i]);", "EXT_PARAMS": "" },
    "DECLS": ["INPLACE"]
  },
  {
    "SHADER_NAME": "abs_in_place_f16",
    "REPLS": { "TYPE": "f16", "FUNC": "src[dst_i] = abs(src[src_i]);", "EXT_PARAMS": "" },
    "DECLS": ["INPLACE"]
  },
  {
    "SHADER_NAME": "sgn_f32",
    "REPLS": { "TYPE": "f32", "FUNC": "dst[dst_i] = select(select(0.0, -1.0, src[src_i] < 0.0), 1.0, src[src_i] > 0.0);", "EXT_PARAMS": "" },
    "DECLS": ["NOT_INPLACE"]
  },
  {
    "SHADER_NAME": "sgn_f16",
    "REPLS": { "TYPE": "f16", "FUNC": "dst[dst_i] = select(select(0.0h, -1.0h, src[src_i] < 0.0h), 1.0h, src[src_i] > 0.0h);", "EXT_PARAMS": "" },
    "DECLS": ["NOT_INPLACE"]
  },
  {
    "SHADER_NAME": "sgn_in_place_f32",
    "REPLS": { "TYPE": "f32", "FUNC": "src[dst_i] = select(select(0.0, -1.0, src[src_i] < 0.0), 1.0, src[src_i] > 0.0);", "EXT_PARAMS": "" },
    "DECLS": ["INPLACE"]
  },
  {
    "SHADER_NAME": "sgn_in_place_f16",
    "REPLS": { "TYPE": "f16", "FUNC": "src[dst_i] = select(select(0.0h, -1.0h, src[src_i] < 0.0h), 1.0h, src[src_i] > 0.0h);", "EXT_PARAMS": "" },
    "DECLS": ["INPLACE"]
  },
  {
    "SHADER_NAME": "neg_f32",
    "REPLS": { "TYPE": "f32", "FUNC": "dst[dst_i] = -src[src_i];", "EXT_PARAMS": "" },
    "DECLS": ["NOT_INPLACE"]
  },
  {
    "SHADER_NAME": "neg_f16",
    "REPLS": { "TYPE": "f16", "FUNC": "dst[dst_i] = -src[src_i];", "EXT_PARAMS": "" },
    "DECLS": ["NOT_INPLACE"]
  },
  {
    "SHADER_NAME": "neg_in_place_f32",
    "REPLS": { "TYPE": "f32", "FUNC": "src[dst_i] = -src[src_i];", "EXT_PARAMS": "" },
    "DECLS": ["INPLACE"]
  },
  {
    "SHADER_NAME": "neg_in_place_f16",
    "REPLS": { "TYPE": "f16", "FUNC": "src[dst_i] = -src[src_i];", "EXT_PARAMS": "" },
    "DECLS": ["INPLACE"]
  },
  {
    "SHADER_NAME": "step_f32",
    "REPLS": { "TYPE": "f32", "FUNC": "dst[dst_i] = select(0.0, 1.0, src[src_i] > 0.0);", "EXT_PARAMS": "" },
    "DECLS": ["NOT_INPLACE"]
  },
  {
    "SHADER_NAME": "step_f16",
    "REPLS": { "TYPE": "f16", "FUNC": "dst[dst_i] = select(0.0h, 1.0h, src[src_i] > 0.0h);", "EXT_PARAMS": "" },
    "DECLS": ["NOT_INPLACE"]
  },
  {
    "SHADER_NAME": "step_in_place_f32",
    "REPLS": { "TYPE": "f32", "FUNC": "src[dst_i] = select(0.0, 1.0, src[src_i] > 0.0);", "EXT_PARAMS": "" },
    "DECLS": ["INPLACE"]
  },
  {
    "SHADER_NAME": "step_in_place_f16",
    "REPLS": { "TYPE": "f16", "FUNC": "src[dst_i] = select(0.0h, 1.0h, src[src_i] > 0.0h);", "EXT_PARAMS": "" },
    "DECLS": ["INPLACE"]
  },
  {
    "SHADER_NAME": "tanh_f32",
    "REPLS": { "TYPE": "f32", "FUNC": "dst[dst_i] = tanh(clamp(src[src_i], -9.010913, 9.010913)); // Regarding tanh() domain restrictions in wgsl https://github.com/gpuweb/gpuweb/issues/4458", "EXT_PARAMS": "" },
    "DECLS": ["NOT_INPLACE"]
  },
  {
    "SHADER_NAME": "tanh_f16",
    "REPLS": { "TYPE": "f16", "FUNC": "dst[dst_i] = tanh(clamp(src[src_i], -9.010913, 9.010913)); // Regarding tanh() domain restrictions in wgsl https://github.com/gpuweb/gpuweb/issues/4458", "EXT_PARAMS": "" },
    "DECLS": ["NOT_INPLACE"]
  },
  {
    "SHADER_NAME": "tanh_in_place_f32",
    "REPLS": { "TYPE": "f32", "FUNC": "src[dst_i] = tanh(clamp(src[src_i], -9.010913, 9.010913)); // Regarding tanh() domain restrictions in wgsl https://github.com/gpuweb/gpuweb/issues/4458", "EXT_PARAMS": "" },
    "DECLS": ["INPLACE"]
  },
  {
    "SHADER_NAME": "tanh_in_place_f16",
    "REPLS": { "TYPE": "f16", "FUNC": "src[dst_i] = tanh(clamp(src[src_i], -9.010913, 9.010913)); // Regarding tanh() domain restrictions in wgsl https://github.com/gpuweb/gpuweb/issues/4458", "EXT_PARAMS": "" },
    "DECLS": ["INPLACE"]
  },
  {
    "SHADER_NAME": "elu_f32",
    "REPLS": { "TYPE": "f32", "FUNC": "dst[dst_i] = select(exp(src[src_i]) - 1.0, src[src_i], src[src_i] > 0.0);", "EXT_PARAMS": "" },
    "DECLS": ["NOT_INPLACE"]
  },
  {
    "SHADER_NAME": "elu_f16",
    "REPLS": { "TYPE": "f16", "FUNC": "dst[dst_i] = select(exp(src[src_i]) - 1.0h, src[src_i], src[src_i] > 0.0h);", "EXT_PARAMS": "" },
    "DECLS": ["NOT_INPLACE"]
  },
  {
    "SHADER_NAME": "elu_in_place_f32",
    "REPLS": { "TYPE": "f32", "FUNC": "src[dst_i] = select(exp(src[src_i]) - 1.0, src[src_i], src[src_i] > 0.0);", "EXT_PARAMS": "" },
    "DECLS": ["INPLACE"]
  },
  {
    "SHADER_NAME": "elu_in_place_f16",
    "REPLS": { "TYPE": "f16", "FUNC": "src[dst_i] = select(exp(src[src_i]) - 1.0h, src[src_i], src[src_i] > 0.0h);", "EXT_PARAMS": "" },
    "DECLS": ["INPLACE"]
  },
  {
    "SHADER_NAME": "relu_f32",
    "REPLS": { "TYPE": "f32", "FUNC": "dst[dst_i] = select(0.0, src[src_i], src[src_i] > 0.0);", "EXT_PARAMS": "" },
    "DECLS": ["NOT_INPLACE"]
  },
  {
    "SHADER_NAME": "relu_f16",
    "REPLS": { "TYPE": "f16", "FUNC": "dst[dst_i] = select(0.0h, src[src_i], src[src_i] > 0.0h);", "EXT_PARAMS": "" },
    "DECLS": ["NOT_INPLACE"]
  },
  {
    "SHADER_NAME": "relu_in_place_f32",
    "REPLS": { "TYPE": "f32", "FUNC": "src[dst_i] = select(0.0, src[src_i], src[src_i] > 0.0);", "EXT_PARAMS": "" },
    "DECLS": ["INPLACE"]
  },
  {
    "SHADER_NAME": "relu_in_place_f16",
    "REPLS": { "TYPE": "f16", "FUNC": "src[dst_i] = select(0.0h, src[src_i], src[src_i] > 0.0h);", "EXT_PARAMS": "" },
    "DECLS": ["INPLACE"]
  },
  {
    "SHADER_NAME": "sigmoid_f32",
    "REPLS": { "TYPE": "f32", "FUNC": "dst[dst_i] = 1.0 / (1.0 + exp(-src[src_i]));", "EXT_PARAMS": "" },
    "DECLS": ["NOT_INPLACE"]
  },
  {
    "SHADER_NAME": "sigmoid_f16",
    "REPLS": { "TYPE": "f16", "FUNC": "dst[dst_i] = 1.0h / (1.0h + exp(-src[src_i]));", "EXT_PARAMS": "" },
    "DECLS": ["NOT_INPLACE"]
  },
  {
    "SHADER_NAME": "sigmoid_in_place_f32",
    "REPLS": { "TYPE": "f32", "FUNC": "src[dst_i] = 1.0 / (1.0 + exp(-src[src_i]));", "EXT_PARAMS": "" },
    "DECLS": ["INPLACE"]
  },
  {
    "SHADER_NAME": "sigmoid_in_place_f16",
    "REPLS": { "TYPE": "f16", "FUNC": "src[dst_i] = 1.0h / (1.0h + exp(-src[src_i]));", "EXT_PARAMS": "" },
    "DECLS": ["INPLACE"]
  },
  {
    "SHADER_NAME": "gelu_f32",
    "REPLS": { "TYPE": "f32", "FUNC": "dst[dst_i] = 0.5 * src[src_i] * (1.0 + tanh(clamp(sqrt(2.0 / 3.14159265) * (src[src_i] + 0.044715 * pow(src[src_i], 3.0)), -9.010913, 9.010913))); // Regarding tanh() domain restrictions in wgsl https://github.com/gpuweb/gpuweb/issues/4458", "EXT_PARAMS": "" },
    "DECLS": ["NOT_INPLACE"]
  },
  {
    "SHADER_NAME": "gelu_f16",
    "REPLS": { "TYPE": "f16", "FUNC": "dst[dst_i] = 0.5h * src[src_i] * (1.0h + tanh(clamp(sqrt(2.0h / 3.14159265h) * (src[src_i] + 0.044715h * pow(src[src_i], 3.0h)), -9.010913, 9.010913))); // Regarding tanh() domain restrictions in wgsl https://github.com/gpuweb/gpuweb/issues/4458", "EXT_PARAMS": "" },
    "DECLS": ["NOT_INPLACE"]
  },
  {
    "SHADER_NAME": "gelu_in_place_f32",
    "REPLS": { "TYPE": "f32", "FUNC": "src[dst_i] = 0.5 * src[src_i] * (1.0 + tanh(clamp(sqrt(2.0 / 3.14159265) * (src[src_i] + 0.044715 * pow(src[src_i], 3.0)), -9.010913, 9.010913))); // Regarding tanh() domain restrictions in wgsl https://github.com/gpuweb/gpuweb/issues/4458", "EXT_PARAMS": "" },
    "DECLS": ["INPLACE"]
  },
  {
    "SHADER_NAME": "gelu_in_place_f16",
    "REPLS": { "TYPE": "f16", "FUNC": "src[dst_i] = 0.5h * src[src_i] * (1.0h + tanh(clamp(sqrt(2.0h / 3.14159265h) * (src[src_i] + 0.044715h * pow(src[src_i], 3.0h)), -9.010913, 9.010913))); // Regarding tanh() domain restrictions in wgsl https://github.com/gpuweb/gpuweb/issues/4458", "EXT_PARAMS": "" },
    "DECLS": ["INPLACE"]
  },
  {
    "SHADER_NAME": "gelu_quick_f32",
    "REPLS": { "TYPE": "f32", "FUNC": "dst[dst_i] = src[src_i] * 0.5 * (1.0 + tanh(clamp(0.79788456 * src[src_i] * (1.0 + 0.044715 * src[src_i] * src[src_i]), -9.010913, 9.010913))); // Regarding tanh() domain restrictions in wgsl https://github.com/gpuweb/gpuweb/issues/4458", "EXT_PARAMS": "" },
    "DECLS": ["NOT_INPLACE"]
  },
  {
    "SHADER_NAME": "gelu_quick_f16",
    "REPLS": { "TYPE": "f16", "FUNC": "dst[dst_i] = src[src_i] * 0.5h * (1.0h + tanh(clamp(0.79788456h * src[src_i] * (1.0h + 0.044715h * src[src_i] * src[src_i]), -9.010913, 9.010913))); // Regarding tanh() domain restrictions in wgsl https://github.com/gpuweb/gpuweb/issues/4458", "EXT_PARAMS": "" },
    "DECLS": ["NOT_INPLACE"]
  },
  {
    "SHADER_NAME": "gelu_quick_in_place_f32",
    "REPLS": { "TYPE": "f32", "FUNC": "src[dst_i] = src[src_i] * 0.5 * (1.0 + tanh(clamp(0.79788456 * src[src_i] * (1.0 + 0.044715 * src[src_i] * src[src_i]), -9.010913, 9.010913))); // Regarding tanh() domain restrictions in wgsl https://github.com/gpuweb/gpuweb/issues/4458", "EXT_PARAMS": "" },
    "DECLS": ["INPLACE"]
  },
  {
    "SHADER_NAME": "gelu_quick_in_place_f16",
    "REPLS": { "TYPE": "f16", "FUNC": "src[dst_i] = src[src_i] * 0.5h * (1.0h + tanh(0.79788456h * src[src_i] * (1.0h + 0.044715h * src[src_i] * src[src_i]))); // Regarding tanh() domain restrictions in wgsl https://github.com/gpuweb/gpuweb/issues/4458", "EXT_PARAMS": "" },
    "DECLS": ["INPLACE"]
  },
  {
    "SHADER_NAME": "silu_f32",
    "REPLS": { "TYPE": "f32", "FUNC": "dst[dst_i] = src[src_i] / (1.0 + exp(-src[src_i]));", "EXT_PARAMS": "" },
    "DECLS": ["NOT_INPLACE"]
  },
  {
    "SHADER_NAME": "silu_f16",
    "REPLS": { "TYPE": "f16", "FUNC": "dst[dst_i] = src[src_i] / (1.0h + exp(-src[src_i]));", "EXT_PARAMS": "" },
    "DECLS": ["NOT_INPLACE"]
  },
  {
    "SHADER_NAME": "silu_in_place_f32",
    "REPLS": { "TYPE": "f32", "FUNC": "src[dst_i] = src[src_i] / (1.0 + exp(-src[src_i]));", "EXT_PARAMS": "" },
    "DECLS": ["INPLACE"]
  },
  {
    "SHADER_NAME": "silu_in_place_f16",
    "REPLS": { "TYPE": "f16", "FUNC": "src[dst_i] = src[src_i] / (1.0h + exp(-src[src_i]));", "EXT_PARAMS": "" },
    "DECLS": ["INPLACE"]
  },
  {
    "SHADER_NAME": "hardswish_f32",
    "REPLS": { "TYPE": "f32", "FUNC": "dst[dst_i] = src[src_i] * min(1.0, max(0.0, (src[src_i] + 3.0) / 6.0));", "EXT_PARAMS": "" },
    "DECLS": ["NOT_INPLACE"]
  },
  {
    "SHADER_NAME": "hardswish_f16",
    "REPLS": { "TYPE": "f16", "FUNC": "dst[dst_i] = src[src_i] * min(1.0h, max(0.0h, (src[src_i] + 3.0h) / 6.0h));", "EXT_PARAMS": "" },
    "DECLS": ["NOT_INPLACE"]
  },
  {
    "SHADER_NAME": "hardswish_in_place_f32",
    "REPLS": { "TYPE": "f32", "FUNC": "src[dst_i] = src[src_i] * min(1.0, max(0.0, (src[src_i] + 3.0) / 6.0));", "EXT_PARAMS": "" },
    "DECLS": ["INPLACE"]
  },
  {
    "SHADER_NAME": "hardswish_in_place_f16",
    "REPLS": { "TYPE": "f16", "FUNC": "src[dst_i] = src[src_i] * min(1.0h, max(0.0h, (src[src_i] + 3.0h) / 6.0h));", "EXT_PARAMS": "" },
    "DECLS": ["INPLACE"]
  },
  {
    "SHADER_NAME": "hardsigmoid_f32",
    "REPLS": { "TYPE": "f32", "FUNC": "dst[dst_i] = min(1.0, max(0.0, (src[src_i] + 3.0) / 6.0));", "EXT_PARAMS": "" },
    "DECLS": ["NOT_INPLACE"]
  },
  {
    "SHADER_NAME": "hardsigmoid_f16",
    "REPLS": { "TYPE": "f16", "FUNC": "dst[dst_i] = min(1.0h, max(0.0h, (src[src_i] + 3.0h) / 6.0h));", "EXT_PARAMS": "" },
    "DECLS": ["NOT_INPLACE"]
  },
  {
    "SHADER_NAME": "hardsigmoid_in_place_f32",
    "REPLS": { "TYPE": "f32", "FUNC": "src[dst_i] = min(1.0, max(0.0, (src[src_i] + 3.0) / 6.0));", "EXT_PARAMS": "" },
    "DECLS": ["INPLACE"]
  },
  {
    "SHADER_NAME": "hardsigmoid_in_place_f16",
    "REPLS": { "TYPE": "f16", "FUNC": "src[dst_i] = min(1.0h, max(0.0h, (src[src_i] + 3.0h) / 6.0h));", "EXT_PARAMS": "" },
    "DECLS": ["INPLACE"]
  },
  {
    "SHADER_NAME": "exp_f32",
    "REPLS": { "TYPE": "f32", "FUNC": "dst[dst_i] = exp(src[src_i]);", "EXT_PARAMS": "" },
    "DECLS": ["NOT_INPLACE"]
  },
  {
    "SHADER_NAME": "exp_f16",
    "REPLS": { "TYPE": "f16", "FUNC": "dst[dst_i] = exp(src[src_i]);", "EXT_PARAMS": "" },
    "DECLS": ["NOT_INPLACE"]
  },
  {
    "SHADER_NAME": "exp_in_place_f32",
    "REPLS": { "TYPE": "f32", "FUNC": "src[dst_i] = exp(src[src_i]);", "EXT_PARAMS": "" },
    "DECLS": ["INPLACE"]
  },
  {
    "SHADER_NAME": "exp_in_place_f16",
    "REPLS": { "TYPE": "f16", "FUNC": "src[dst_i] = exp(src[src_i]);", "EXT_PARAMS": "" },
    "DECLS": ["INPLACE"]
  },
  {
    "SHADER_NAME": "gelu_erf_f32",
    "REPLS": { "TYPE": "f32", "FUNC": "dst[dst_i] = 0.5 * src[src_i] * (1.0 + tanh(clamp(0.79788456 * (src[src_i] + 0.044715 * src[src_i] * src[src_i] * src[src_i]), -9.010913, 9.010913))); // Regarding tanh() domain restrictions in wgsl https://github.com/gpuweb/gpuweb/issues/4458", "EXT_PARAMS": "" },
    "DECLS": ["NOT_INPLACE"]
  },
  {
    "SHADER_NAME": "gelu_erf_f16",
    "REPLS": { "TYPE": "f16", "FUNC": "dst[dst_i] = 0.5h * src[src_i] * (1.0h + tanh(clamp(0.79788456h * (src[src_i] + 0.044715h * src[src_i] * src[src_i] * src[src_i]), -9.010913, 9.010913))); // Regarding tanh() domain restrictions in wgsl https://github.com/gpuweb/gpuweb/issues/4458", "EXT_PARAMS": "" },
    "DECLS": ["NOT_INPLACE"]
  },
  {
    "SHADER_NAME": "gelu_erf_in_place_f32",
    "REPLS": { "TYPE": "f32", "FUNC": "src[dst_i] = 0.5 * src[src_i] * (1.0 + tanh(clamp(0.79788456 * (src[src_i] + 0.044715 * src[src_i] * src[src_i] * src[src_i]), -9.010913, 9.010913))); // Regarding tanh() domain restrictions in wgsl https://github.com/gpuweb/gpuweb/issues/4458", "EXT_PARAMS": "" },
    "DECLS": ["INPLACE"]
  },
  {
    "SHADER_NAME": "gelu_erf_in_place_f16",
    "REPLS": { "TYPE": "f16", "FUNC": "src[dst_i] = 0.5h * src[src_i] * (1.0h + tanh(clamp(0.79788456h * (src[src_i] + 0.044715h * src[src_i] * src[src_i] * src[src_i]), -9.010913, 9.010913))); // Regarding tanh() domain restrictions in wgsl https://github.com/gpuweb/gpuweb/issues/4458", "EXT_PARAMS": "" },
    "DECLS": ["INPLACE"]
  },
  {
    "SHADER_NAME": "xielu_f32",
    "REPLS": {
      "TYPE": "f32",
      "FUNC": "dst[dst_i] = select(((exp(min(src[src_i], f32(params.eps))) - 1.0) - src[src_i]) * f32(params.alpha_n) + f32(params.beta) * src[src_i], f32(params.alpha_p) * src[src_i] * src[src_i] + f32(params.beta) * src[src_i], src[src_i] > 0.0);",
      "EXT_PARAMS": "alpha_n: u32, alpha_p: u32, beta: u32, eps: u32"
    },
    "DECLS": ["NOT_INPLACE"]
  },
  {
    "SHADER_NAME": "xielu_f16",
    "REPLS": {
      "TYPE": "f16",
      "FUNC": "dst[dst_i] = select(((exp(min(src[src_i], f16(params.eps))) - 1.0h) - src[src_i]) * f16(params.alpha_n) + f16(params.beta) * src[src_i], f16(params.alpha_p) * src[src_i] * src[src_i] + f16(params.beta) * src[src_i], src[src_i] > 0.0h);",
      "EXT_PARAMS": "alpha_n: u32, alpha_p: u32, beta: u32, eps: u32"
    },
    "DECLS": ["NOT_INPLACE"]
  },
  {
    "SHADER_NAME": "xielu_in_place_f32",
    "REPLS": {
      "TYPE": "f32",
      "FUNC": "src[dst_i] = select(((exp(min(src[src_i], f32(params.eps))) - 1.0) - src[src_i]) * f32(params.alpha_n) + f32(params.beta) * src[src_i], f32(params.alpha_p) * src[src_i] * src[src_i] + f32(params.beta) * src[src_i], src[src_i] > 0.0);",
      "EXT_PARAMS": "alpha_n: u32, alpha_p: u32, beta: u32, eps: u32"
    },
    "DECLS": ["INPLACE"]
  },
  {
    "SHADER_NAME": "xielu_in_place_f16",
      "REPLS": {
      "TYPE": "f16",
      "FUNC": "src[dst_i] = select(((exp(min(src[src_i], f16(params.eps))) - 1.0h) - src[src_i]) * f16(params.alpha_n) + f16(params.beta) * src[src_i], f16(params.alpha_p) * src[src_i] * src[src_i] + f16(params.beta) * src[src_i], src[src_i] > 0.0h);",
      "EXT_PARAMS": "alpha_n: u32, alpha_p: u32, beta: u32, eps: u32"
    },
    "DECLS": ["INPLACE"]
  }
]

#end(VARIANTS)

#define(DECLS)

#decl(INPLACE)

@group(0) @binding(1)
var<storage, read_write> dst: array<{{TYPE}}>;

@group(0) @binding(2)
var<uniform> params: Params;

#enddecl(INPLACE)

#decl(NOT_INPLACE)

@group(0) @binding(1)
var<storage, read_write> dst: array<{{TYPE}}>;

@group(0) @binding(2)
var<uniform> params: Params;

#enddecl(NOT_INPLACE)

#end(DECLS)

#define(SHADER)

enable f16;

fn update(dst_i: u32, src_i: u32) {
    {{FUNC}}
}

@group(0) @binding(0)
var<storage, read_write> src: array<{{TYPE}}>;

DECLS

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
    dst_ne2: u32,

    {{EXT_PARAMS}}
};

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


    update(params.offset_dst + dst_idx, params.offset_src + src_idx);
}

#end(SHADER)

