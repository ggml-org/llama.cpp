struct MulMatParams {
    m: u32,
    n: u32,
    k: u32
};

@group(0) @binding(0) var<storage, read> src0: array<f32>;
@group(0) @binding(1) var<storage, read> src1: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

@group(0) @binding(3) var<uniform> params: MulMatParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= params.m * params.n) {
        return;
    }
    let row = global_id.x / params.n;
    let col = global_id.x % params.n;
    var sum = 0.0;
    for (var i: u32 = 0u; i < params.k; i = i + 1u) {
        sum = sum + src0[col * params.k + i] * src1[row * params.k + i];
    }
    dst[row * params.n + col] = sum;
}