#include "common_decls.tmpl"
enable f16;

@group(0) @binding(0)
#if defined(INPUT_F32)
var<storage, read_write> input: array<f32>;
#elif defined(INPUT_F16)
var<storage, read_write> input: array<f16>;
#endif

@group(0) @binding(1)
#if defined(OUTPUT_F32)
var<storage, read_write> output: array<f32>;
#elif defined(OUTPUT_F16)
var<storage, read_write> output: array<f16>;
#endif

struct Params {
    offset_i: u32,
    offset_o: u32,

    // element strides
    si0: u32, si1: u32, si2: u32, si3: u32,
    so0: u32, so1: u32, so2: u32, so3: u32,

    KW: u32, KH: u32, IC: u32,
    IW: u32, IH: u32, N: u32,
    OW: u32, OH: u32,

    // stride
    s0: u32, s1: u32,
    // padding
    p0: u32, p1: u32,
    // dilation
    d0: u32, d1: u32,
}

@group(0) @binding(2)
var<uniform> params: Params;

fn load_input(idx: u32) -> f32 {
    #if defined(INPUT_F32)
        return input[idx];
    #elif defined(INPUT_F16)
        return f32(input[idx]);
    #endif
}

fn store_output(idx: u32, val: f32) {
    #if defined(OUTPUT_F32)
        output[idx] = val;
    #elif defined(OUTPUT_F16)
        output[idx] = f16(val);
    #endif
}

@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>
) {

    let threads_per_group = u32(WG_SIZE);
    let i_out = gid.x + (num_wg.x * threads_per_group) * gid.y;
    let K = params.KW * params.KH * params.IC;
    let M = params.OW * params.OH;
    let total = K * M * params.N;

    if (i_out >= total) {
        return;
    }

    // decode (k, m, n)
    var i = i_out;
    let n = i / (K * M);
    i = i % (K * M);
    let m = i / K;
    let k = i % K;

    // decode (oh, ow)
    let oh = m / params.OW;
    let ow = m % params.OW;

    // decode (kw, kh, ic)
    let kw = k % params.KW;
    let tmp = k / params.KW;
    let kh = tmp % params.KH;
    let ic = tmp / params.KH;

    let iw_i32 = i32(ow * params.s0 + kw * params.d0) - i32(params.p0);
    let ih_i32 = i32(oh * params.s1 + kh * params.d1) - i32(params.p1);

    if (iw_i32 >= 0 && iw_i32 < i32(params.IW) && ih_i32 >= 0 && ih_i32 < i32(params.IH)) {
        let iw = u32(iw_i32);
        let ih = u32(ih_i32);
        let in_idx = params.offset_i + iw * params.si0 + ih * params.si1 + ic * params.si2 + n * params.si3;
        store_output(params.offset_o + k * params.so0 + ow * params.so1 + oh * params.so2 + n * params.so3, load_input(in_idx));
    } else {
        store_output(params.offset_o + k * params.so0 + ow * params.so1 + oh * params.so2 + n * params.so3, 0.0);
    }
}
