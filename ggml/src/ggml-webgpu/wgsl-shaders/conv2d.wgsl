#include "common_decls.tmpl"
enable f16;

@group(0) @binding(0)
var<storage, read_write> weights: array<WEIGHT_TYPE>;
@group(0) @binding(1)
var<storage, read_write> input: array<INPUT_TYPE>;
@group(0) @binding(2)
var<storage, read_write> output: array<OUTPUT_TYPE>;

struct Params {
    offset_w: u32,
    offset_i: u32,
    offset_o: u32,

    // element strides
    sw0: u32, sw1: u32, sw2: u32, sw3: u32,
    si0: u32, si1: u32, si2: u32, si3: u32,
    so0: u32, so1: u32, so2: u32, so3: u32,

    // kernel dimensions
    KW: u32, KH: u32, IC: u32,
    // input dimensions
    IW: u32, IH: u32,
    // output dimensions
    OW: u32, OH: u32, OC_out: u32, N_out: u32,

    // stride
    s0: u32, s1: u32,
    // padding
    p0: u32, p1: u32,
    // dilation
    d0: u32, d1: u32,
};

@group(0) @binding(3)
var<uniform> params: Params;

fn ceil_div_u32(x: u32, y: u32) -> u32 {
    return (x + y - 1) / y;
}

// returns the first valid kernel index k such that base + k * step >= 0
fn first_valid_k(base: i32, step: u32) -> u32 {
    if (base >= 0) {
        return 0;
    }

    return ceil_div_u32(u32(-base), step);
}

// returns the first invalid kernel index k such that base + k * step >= limit so valid k are in [0, end_valid_k)
fn end_valid_k(base: i32, step: u32, limit: u32, k_max: u32) -> u32 {
    let remaining = i32(limit) - base;
    if (remaining <= 0) {
        return 0;
    }

    return min(k_max, ceil_div_u32(u32(remaining), step));
}

@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>
) {

    let threads_per_group = u32(WG_SIZE);
    let i_out = gid.x + (num_wg.x * threads_per_group) * gid.y;
    let n_out = params.OW * params.OH * params.OC_out * params.N_out;

    var sum: f32 = 0.0;
    if (i_out >= n_out) {
        return;
    }

    // Kernel layout: [KW, KH, IC, ..]
    // Input layout:  [IW, IH, .., ..]
    // Output layout: [OW, OH, OC, N]

    var i = i_out;
    let n = i / (params.OC_out * params.OH * params.OW);
    i = i % (params.OC_out * params.OH * params.OW);
    let oc = i / (params.OH * params.OW);
    i = i % (params.OH * params.OW);
    let oh = i / params.OW;
    let ow = i % params.OW;

    let ow_base = i32(ow * params.s0) - i32(params.p0);
    let oh_base = i32(oh * params.s1) - i32(params.p1);

    // clip the valid kernel window once
    let kw_begin = first_valid_k(ow_base, params.d0);
    let kw_end = end_valid_k(ow_base, params.d0, params.IW, params.KW);
    let kh_begin = first_valid_k(oh_base, params.d1);
    let kh_end = end_valid_k(oh_base, params.d1, params.IH, params.KH);

    // entire receptive field is out of bounds
    if (kw_begin >= kw_end || kh_begin >= kh_end) {
        let out_idx = params.offset_o + ow * params.so0 + oh * params.so1 + oc * params.so2 + n * params.so3;
        output[out_idx] = OUTPUT_TYPE(0.0);
        return;
    }

    let weight_oc_base = params.offset_w + oc * params.sw3;
    let input_n_base = params.offset_i + n * params.si3;

    for (var ic: u32 = 0; ic < params.IC; ic += 1) {
        let w_base_ic = ic * params.sw2 + weight_oc_base;
        let in_base = ic * params.si2 + input_n_base;

        for (var kh: u32 = kh_begin; kh < kh_end; kh += 1) {
            let ih = u32(oh_base + i32(kh * params.d1));
            let w_row_base = w_base_ic + kh * params.sw1;
            let in_row_base = in_base + ih * params.si1;
            for (var kw: u32 = kw_begin; kw < kw_end; kw += 1) {
                let iw = u32(ow_base + i32(kw * params.d0));
                let w_idx = w_row_base + kw * params.sw0;
                let in_idx = in_row_base + iw * params.si0;
                sum += f32(weights[w_idx]) * f32(input[in_idx]);
            }
        }
    }

    let out_idx = params.offset_o + ow * params.so0 + oh * params.so1 + oc * params.so2 + n * params.so3;
    output[out_idx] = OUTPUT_TYPE(sum);
}
