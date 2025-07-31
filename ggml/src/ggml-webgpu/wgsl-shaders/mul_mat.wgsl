struct MulMatParams {
    m: u32,
    n: u32,
    k: u32,
    // all strides are in elements
    stride_01: u32,
    stride_11: u32,
    stride_02: u32,
    stride_12: u32,
    stride_03: u32,
    stride_13: u32,

    bs02: u32,
    bs03: u32,
    broadcast2: u32,
    broadcast3: u32
};

@group(0) @binding(0) var<storage, read_write> src0: array<f32>; // N rows, K columns
@group(0) @binding(1) var<storage, read_write> src1: array<f32>; // M rows, K columns (transposed)
@group(0) @binding(2) var<storage, read_write> dst: array<f32>; // M rows, N columns

@group(0) @binding(3) var<uniform> params: MulMatParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let total = params.m * params.n * params.bs02 * params.broadcast2 * params.bs03 * params.broadcast3;
    if (global_id.x >= total) {
        return;
    }

    let dst2_stride = params.m * params.n;
    let dst3_stride = dst2_stride * params.bs02 * params.broadcast2;

    let dst3_idx = global_id.x / dst3_stride;
    let src03_idx = dst3_idx / params.broadcast3; // src0 may be broadcast along the third dimension
    let src13_idx = dst3_idx; // src1 is not broadcast
    let dst3_rem = global_id.x % dst3_stride;

    let dst2_idx = dst3_rem / dst2_stride;
    let src02_idx = dst2_idx / params.broadcast2; // src0 may also be broadcast along the second dimension
    let src12_idx = dst2_idx; // src1 is not broadcast

    let dst2_rem = dst3_rem % dst2_stride;

    let row = dst2_rem / params.n; // output row
    let col = dst2_rem % params.n; // output column

    var sum = 0.0;
    for (var i: u32 = 0u; i < params.k; i = i + 1u) {
        let src0_idx = src03_idx * params.stride_03 + src02_idx * params.stride_02 + col * params.stride_01 + i;
        let src1_idx = src13_idx * params.stride_13 + src12_idx * params.stride_12 + row * params.stride_11 + i;
        sum = sum + src0[src0_idx] * src1[src1_idx];
    }
    dst[dst3_idx * dst3_stride + dst2_idx * dst2_stride + row * params.n + col] = sum;
}
