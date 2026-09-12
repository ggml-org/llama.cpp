// gradient of get_rows: each dst row gathers the grad rows that point to it,
// this avoids the atomics that a scatter would need

struct Params {
    ne: u32,

    offset_grad: u32,
    offset_idx: u32,
    offset_dst: u32,

    ne00: u32,  // row size
    n_idx: u32, // number of indices, can be less than the rows of grad

    stride_grad_1: u32,
    stride_idx_0: u32,

    ne0: u32,
    ne1: u32,
};

@group(0) @binding(0)
var<storage, read_write> grad: array<f32>;

@group(0) @binding(1)
var<storage, read_write> row_idx: array<i32>;

@group(0) @binding(2)
var<storage, read_write> dst: array<f32>;

@group(0) @binding(3)
var<uniform> params: Params;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups)       num_wg: vec3<u32>) {
    let idx = gid.x + (num_wg.x * u32(WG_SIZE)) * gid.y;
    if (idx >= params.ne) {
        return;
    }

    let col = idx % params.ne0;
    let row = idx / params.ne0;

    var sum = 0.0f;
    for (var i: u32 = 0; i < params.n_idx; i++) {
        if (u32(row_idx[params.offset_idx + i * params.stride_idx_0]) == row) {
            sum += grad[params.offset_grad + i * params.stride_grad_1 + col];
        }
    }

    dst[params.offset_dst + row * params.ne0 + col] = sum;
}
