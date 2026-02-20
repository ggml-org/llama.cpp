#ifdef TYPE_F16
enable f16;
#define TYPE f16
#else
#define TYPE f32
#endif


@group(0) @binding(0)
var<storage, read_write> src: array<TYPE>;

#ifndef INPLACE
@group(0) @binding(1)
var<storage, read_write> dst: array<TYPE>;
#define PARAMS_BINDING 2
#else
#define PARAMS_BINDING 1
#endif

struct Params {
    ne: u32,            // total number of elements
    offset_src: u32,    // in elements
    offset_dst: u32,    // in elements

    // Strides (in elements)
    stride_src0: u32,
    stride_src1: u32,
    stride_src2: u32,
    stride_src3: u32,

    // Logical shapes
    ne0: u32,
    ne1: u32,
    ne2: u32,
#ifdef CLAMP
    clamp_min: f32,
    clamp_max: f32,
#endif
#ifdef FILL
    fill_val: f32,
#endif
#ifdef XIELU
    alpha_n: f32,
    alpha_p: f32,
    beta: f32,
    eps: f32,
#endif

};

@group(0) @binding(PARAMS_BINDING)
var<uniform> params: Params;

@compute @workgroup_size(WG_SIZE)
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

    let src_idx = i0 * params.stride_src0 + i1 * params.stride_src1 +
                  i2 * params.stride_src2 + i3 * params.stride_src3;

#ifdef ABS
    let res = abs(src[params.offset_src + src_idx]);
#endif
#ifdef SGN
    let res = select(TYPE(select(0.0, -1.0, src[params.offset_src + src_idx] < 0.0)), TYPE(1.0),
                     src[params.offset_src + src_idx] > 0.0);
#endif
#ifdef NEG
    let res = -src[params.offset_src + src_idx];
#endif
#ifdef STEP
    let res = TYPE(select(0.0, 1.0, src[params.offset_src + src_idx] > 0.0));
#endif
#ifdef TANH
    let res = tanh(clamp(src[params.offset_src + src_idx], -9.010913, 9.010913));
#endif
#ifdef RELU
    let res = select(0.0, src[params.offset_src + src_idx], src[params.offset_src + src_idx] > 0.0);
#endif
#ifdef ELU
    let res = select(exp(src[params.offset_src + src_idx]) - 1.0, src[params.offset_src + src_idx],
                     src[params.offset_src + src_idx] > 0.0);
#endif
#ifdef HARDSIGMOID
    let res = min(1.0, max(0.0, (src[params.offset_src + src_idx] + 3.0) / 6.0));
#endif
#ifdef SIGMOID
    let res = 1.0 / (1.0 + exp(-src[params.offset_src + src_idx]));
#endif
#ifdef SILU
    let res = src[params.offset_src + src_idx] / (1.0 + exp(-src[params.offset_src + src_idx]));
#endif
#ifdef EXP
    let res = exp(src[params.offset_src + src_idx]);
#endif
#ifdef LOG
    let res = TYPE(log(f32(src[params.offset_src + src_idx])));
#endif
#ifdef CLAMP
    let res = clamp(src[params.offset_src + src_idx], TYPE(params.clamp_min), TYPE(params.clamp_max));
#endif
#ifdef FILL
    let res = TYPE(params.fill_val);
#endif
#ifdef HARDSWISH
    let res = src[params.offset_src + src_idx] *
              min(1.0, max(0.0, (src[params.offset_src + src_idx] + 3.0) / 6.0));
#endif
#ifdef GELU
    let res = 0.5 * src[params.offset_src + src_idx] *
              (1.0 + tanh(clamp(sqrt(2.0 / 3.14159265) *
                               (src[params.offset_src + src_idx] +
                                0.044715 * pow(src[params.offset_src + src_idx], 3.0)),
                               -9.010913, 9.010913)));
#endif
#ifdef GELU_QUICK
    let res = src[params.offset_src + src_idx] * 0.5 *
              (1.0 + tanh(clamp(0.79788456 *
                               (src[params.offset_src + src_idx] +
                                0.044715 * src[params.offset_src + src_idx] *
                                    src[params.offset_src + src_idx] * src[params.offset_src + src_idx]),
                               -9.010913, 9.010913)));
#endif
#ifdef GELU_ERF
    let res = 0.5 * src[params.offset_src + src_idx] *
              (1.0 + tanh(clamp(0.79788456 *
                               (src[params.offset_src + src_idx] +
                                0.044715 * src[params.offset_src + src_idx] *
                                    src[params.offset_src + src_idx] * src[params.offset_src + src_idx]),
                               -9.010913, 9.010913)));
#endif
#ifdef XIELU
    let res =
        select(((exp(min(src[params.offset_src + src_idx], TYPE(params.eps))) - 1.0) -
                src[params.offset_src + src_idx]) *
                   TYPE(params.alpha_n) +
               TYPE(params.beta) * src[params.offset_src + src_idx],
               TYPE(params.alpha_p) * src[params.offset_src + src_idx] *
                   src[params.offset_src + src_idx] +
                   TYPE(params.beta) * src[params.offset_src + src_idx],
               src[params.offset_src + src_idx] > 0.0);
#endif
#ifdef SOFTPLUS
    let src_f32 = f32(src[params.offset_src + src_idx]);
    let res = TYPE(select(log(1.0 + exp(src_f32)), src_f32, src_f32 > 20.0));
#endif
#ifdef EXPM1
    let res = exp(src[params.offset_src + src_idx]) - 1.0;
#endif
#ifdef FLOOR
    let res = floor(src[params.offset_src + src_idx]);
#endif
#ifdef CEIL
    let res = ceil(src[params.offset_src + src_idx]);
#endif
#ifdef ROUND
    let src_f32 = f32(src[params.offset_src + src_idx]);
    let result = select(ceil(src_f32 - 0.5), floor(src_f32 + 0.5), src_f32 >= 0.0);
    let res = TYPE(result);
#endif
#ifdef TRUNC
    let res = trunc(src[params.offset_src + src_idx]);
#endif
#ifdef SQR
    let res = src[params.offset_src + src_idx] * src[params.offset_src + src_idx];
#endif
#ifdef SQRT
    let res = sqrt(src[params.offset_src + src_idx]);
#endif
#ifdef SIN
    let res_f32 = sin(f32(src[params.offset_src + src_idx]));
    let res = TYPE(res_f32);
#endif
#ifdef COS
    let res_f32 = cos(f32(src[params.offset_src + src_idx]));
    let res = TYPE(res_f32);
#endif

#ifdef INPLACE
    src[params.offset_src + src_idx] = res;
#else
    dst[params.offset_dst + gid.x] = res;
#endif
}
