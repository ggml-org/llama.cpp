@group(0) @binding(0)
var<storage, read> src0: array<f32>;

@group(0) @binding(1)
var<storage, read> src1: array<f32>;

@group(0) @binding(2)
var<storage, read_write> dst: array<f32>;

struct Params {
    offset_src0: u32,
    offset_src1: u32,
    offset_dst: u32,

    stride_src00: u32,
    stride_src01: u32,
    stride_src02: u32,
    stride_src03: u32,

    stride_src10: u32,
    stride_src11: u32,
    stride_src12: u32,
    stride_src13: u32,

    stride_dst0: u32,
    stride_dst1: u32,
    stride_dst2: u32,
    stride_dst3: u32,

    k: u32,
    ne2: u32,
    ne3: u32,
};

@group(0) @binding(3)
var<uniform> params: Params;

var<workgroup> shA: array<f32, BATCH_N * N>;
var<workgroup> shB: array<f32, BATCH_N * K_TILE>;

fn src0_idx(row: u32, col: u32, i2: u32, i3: u32) -> u32 {
    return params.offset_src0 +
           col * params.stride_src00 +
           row * params.stride_src01 +
           i2 * params.stride_src02 +
           i3 * params.stride_src03;
}

fn src1_idx(row: u32, col: u32, i2: u32, i3: u32) -> u32 {
    return params.offset_src1 +
           col * params.stride_src10 +
           row * params.stride_src11 +
           i2 * params.stride_src12 +
           i3 * params.stride_src13;
}

fn dst_idx(row: u32, col: u32, i2: u32, i3: u32) -> u32 {
    return params.offset_dst +
           col * params.stride_dst0 +
           row * params.stride_dst1 +
           i2 * params.stride_dst2 +
           i3 * params.stride_dst3;
}

@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let batch = workgroup_id.y;
    let col = workgroup_id.x * WG_SIZE + local_id.x;
    let i3 = batch / params.ne2;
    let i2 = batch % params.ne2;
    let active_lane = local_id.x < K_TILE;
    let active_col = active_lane && col < params.k;

    var X: array<f32, N>;

    for (var row_base = 0u; row_base < N; row_base += BATCH_N) {
        let cur_n = min(BATCH_N, N - row_base);

        for (var i = local_id.x; i < cur_n * N; i += WG_SIZE) {
            let tile_row = i / N;
            let tile_col = i % N;
            shA[i] = src0[src0_idx(row_base + tile_row, tile_col, i2, i3)];
        }

        for (var i = local_id.x; i < cur_n * K_TILE; i += WG_SIZE) {
            let tile_row = i / K_TILE;
            let tile_col = i % K_TILE;
            let global_col = workgroup_id.x * WG_SIZE + tile_col;
            let sh_idx = tile_row * K_TILE + tile_col;

            if (global_col < params.k) {
                shB[sh_idx] = src1[src1_idx(row_base + tile_row, global_col, i2, i3)];
            } else {
                shB[sh_idx] = 0.0;
            }
        }

        workgroupBarrier();

        if (active_col) {
            for (var row_offset = 0u; row_offset < cur_n; row_offset++) {
                let r = row_base + row_offset;
                var b = shB[row_offset * K_TILE + local_id.x];
                let a_row = row_offset * N;

                for (var t = 0u; t < r; t++) {
                    b -= shA[a_row + t] * X[t];
                }

                let x = b / shA[a_row + r];
                X[r] = x;
                dst[dst_idx(r, col, i2, i3)] = x;
            }
        }

        workgroupBarrier();
    }
}
