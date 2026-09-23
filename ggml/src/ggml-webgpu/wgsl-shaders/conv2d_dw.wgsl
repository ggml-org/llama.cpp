#include "common_decls.tmpl"
enable f16;

// Ported from the Vulkan backend's conv2d_dw.comp. Two variants (based on WHCN)
// selected by the input (src1) layout: contiguous -> WHCN, else CWHN.
// weight (src0) is [KW,KH,1,C]; output matches the input layout.

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

    ne: u32,
    channels: u32,
    dst_w: u32, dst_h: u32,
    src_w: u32, src_h: u32,
    knl_w: u32, knl_h: u32,

    stride_x: i32, stride_y: i32,
    pad_x: i32, pad_y: i32,
    dilation_x: i32, dilation_y: i32,
};

@group(0) @binding(3)
var<uniform> params: Params;

#if defined(WHCN)
// Input/output/kernel contiguous in [W, H, C, N] order (kernel [KW,KH,C]).
fn conv_2d_dw(idx: u32) -> f32 {
    let i0    = idx / params.dst_w;
    let dst_x = idx - i0 * params.dst_w;
    let i1    = i0 / params.dst_h;
    let dst_y = i0 - i1 * params.dst_h;
    let n     = i1 / params.channels;
    let c     = i1 - n * params.channels;

    let src_i = params.offset_i + n * params.channels * params.src_h * params.src_w
                                + c * params.src_h * params.src_w;
    let knl_i = params.offset_w + c * params.knl_h * params.knl_w;

    var sum: f32 = 0.0;
    for (var ky: u32 = 0u; ky < params.knl_h; ky += 1u) {
        let src_y = i32(dst_y) * params.stride_y + i32(ky) * params.dilation_y - params.pad_y;
        if (src_y < 0 || src_y >= i32(params.src_h)) { continue; }
        for (var kx: u32 = 0u; kx < params.knl_w; kx += 1u) {
            let src_x = i32(dst_x) * params.stride_x + i32(kx) * params.dilation_x - params.pad_x;
            if (src_x < 0 || src_x >= i32(params.src_w)) { continue; }
            let v = f32(input[src_i + u32(src_y) * params.src_w + u32(src_x)]);
            let k = f32(weights[knl_i + ky * params.knl_w + kx]);
            sum += v * k;
        }
    }
    return sum;
}
#else
// Channels contiguous (CWHN): channel is the innermost axis.
fn conv_2d_dw(idx: u32) -> f32 {
    let i0    = idx / params.channels;
    let c     = idx - i0 * params.channels;
    let i1    = i0 / params.dst_w;
    let dst_x = i0 - i1 * params.dst_w;
    let n     = i1 / params.dst_h;
    let dst_y = i1 - n * params.dst_h;

    let src_i   = params.offset_i + n * params.channels * params.src_h * params.src_w;
    let src_row = params.src_w * params.channels;
    let knl_row = params.knl_w * params.channels;

    var sum: f32 = 0.0;
    for (var ky: u32 = 0u; ky < params.knl_h; ky += 1u) {
        let src_y = i32(dst_y) * params.stride_y + i32(ky) * params.dilation_y - params.pad_y;
        if (src_y < 0 || src_y >= i32(params.src_h)) { continue; }
        for (var kx: u32 = 0u; kx < params.knl_w; kx += 1u) {
            let src_x = i32(dst_x) * params.stride_x + i32(kx) * params.dilation_x - params.pad_x;
            if (src_x < 0 || src_x >= i32(params.src_w)) { continue; }
            let v = f32(input[src_i + u32(src_y) * src_row + u32(src_x) * params.channels + c]);
            let k = f32(weights[params.offset_w + ky * knl_row + kx * params.channels + c]);
            sum += v * k;
        }
    }
    return sum;
}
#endif

@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>
) {
    let idx = gid.x + (num_wg.x * u32(WG_SIZE)) * gid.y;
    if (idx >= params.ne) { return; }
    output[params.offset_o + idx] = OUTPUT_TYPE(conv_2d_dw(idx));
}
