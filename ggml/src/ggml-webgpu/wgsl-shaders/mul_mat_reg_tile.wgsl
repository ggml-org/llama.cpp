enable f16;

#include "common_decls.tmpl"
#include "mul_mat_decls.tmpl"

#ifdef VEC
fn store_val(acc: array<array<f16, TILE_N>, TILE_M>, tn: u32, tm: u32) -> vec4<f32> {
    return vec4<f32>(f32(acc[tm][tn]), f32(acc[tm + 1][tn]), f32(acc[tm + 2][tn]), f32(acc[tm + 3][tn]));
}
#endif

#ifdef SCALAR
fn store_val(acc: array<array<f16, TILE_N>, TILE_M>, tn: u32, tm: u32) -> f32 {
    return f32(acc[tm][tn]);
}
#endif

struct MulMatParams {
    offset_src0: u32,
    offset_src1: u32,
    offset_dst: u32,
    m: u32,
    n: u32,
    k: u32,
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

@group(0) @binding(0) var<storage, read_write> src0: array<SRC0_TYPE>; // M rows, K columns
@group(0) @binding(1) var<storage, read_write> src1: array<SRC1_TYPE>; // K rows, N columns (transposed)
@group(0) @binding(2) var<storage, read_write> dst: array<DST_TYPE>; // M rows, N columns (transposed)

@group(0) @binding(3) var<uniform> params: MulMatParams;

fn get_local_n(thread_id: u32) -> u32 {
    return thread_id / WORKGROUP_SIZE_M;
}
fn get_local_m(thread_id: u32) -> u32 {
    return thread_id % WORKGROUP_SIZE_M;
}

const TOTAL_WORKGROUP_SIZE = WORKGROUP_SIZE_M * WORKGROUP_SIZE_N;
const TILE_SRC0_SHMEM = TILE_K * WORKGROUP_SIZE_M * TILE_M;
const TILE_SRC1_SHMEM = TILE_K * WORKGROUP_SIZE_N * TILE_N;
var<workgroup> shmem: array<f16, TILE_SRC0_SHMEM + TILE_SRC1_SHMEM>;

@compute @workgroup_size(TOTAL_WORKGROUP_SIZE)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {

    let thread_id = local_id.x;
    let local_m = get_local_m(thread_id);
    let local_n = get_local_n(thread_id);

    let wg_n_count = (params.n + WORKGROUP_SIZE_N * TILE_N - 1u) / (WORKGROUP_SIZE_N * TILE_N);
    let wg_m_count = (params.m + WORKGROUP_SIZE_M * TILE_M - 1u) / (WORKGROUP_SIZE_M * TILE_M);
    let wg_per_matrix = wg_m_count * wg_n_count;

    let batch_idx = wg_id.x / wg_per_matrix;

    let wg_in_batch = wg_id.x % wg_per_matrix;
    let wg_m = wg_in_batch % wg_m_count;
    let wg_n = wg_in_batch / wg_m_count;

    let output_row_base = wg_m * WORKGROUP_SIZE_M * TILE_M + local_m * TILE_M;
    let output_col_base = wg_n * WORKGROUP_SIZE_N * TILE_N + local_n * TILE_N;

    let dst2_stride = params.m * params.n;
    let dst3_stride = dst2_stride * params.bs02 * params.broadcast2;

    let dst3_idx = batch_idx / (params.bs02 * params.broadcast2);
    let src03_idx = dst3_idx / params.broadcast3;
    let src13_idx = dst3_idx;
    let dst2_idx = batch_idx % (params.bs02 * params.broadcast2);
    let src02_idx = dst2_idx / params.broadcast2;
    let src12_idx = dst2_idx;

    let src0_batch_offset = params.offset_src0 + src03_idx * params.stride_03 + src02_idx * params.stride_02;
    let src1_batch_offset = params.offset_src1 + src13_idx * params.stride_13 + src12_idx * params.stride_12;

    let offset_m = wg_m * WORKGROUP_SIZE_M * TILE_M;
    let offset_n = wg_n * WORKGROUP_SIZE_N * TILE_N;

    var acc: array<array<f16, TILE_N>, TILE_M>;

    for (var k_outer = 0u; k_outer < params.k; k_outer += TILE_K) {

        // see mul_mat_decls.tmpl
        init_shmem_src0(thread_id, src0_batch_offset, offset_m, k_outer);
        init_shmem_src1(thread_id, src1_batch_offset, offset_n, k_outer);

        workgroupBarrier();

        let k_end = min(TILE_K, params.k - k_outer);

        for (var k_inner = 0u; k_inner < k_end; k_inner++) {
            var src0_tile: array<f16, TILE_M>;
            for (var tm = 0u; tm < TILE_M; tm++) {
                let src0_m = local_m * TILE_M + tm;
                let src0_idx = k_inner + src0_m * TILE_K;
                src0_tile[tm] = shmem[src0_idx];
            }
            for (var tn = 0u; tn < TILE_N; tn++) {
                let src1_n = local_n * TILE_N + tn;
                let src1_idx = src1_n * TILE_K + k_inner;
                let src1_val = shmem[TILE_SRC0_SHMEM + src1_idx];
                for (var tm = 0u; tm < TILE_M; tm++) {
                      acc[tm][tn] += src0_tile[tm] * src1_val;
                }
            }
        }

        workgroupBarrier();
    }

    let dst_batch_offset = params.offset_dst + dst3_idx * dst3_stride + dst2_idx * dst2_stride;

    for (var tn = 0u; tn < TILE_N; tn++) {
        let global_col = output_col_base + tn;
        if (global_col < params.n) {
            for (var tm = 0u; tm < TILE_M; tm += VEC_SIZE) {
                let global_row = output_row_base + tm;
                if (global_row < params.m) {
                    let dst_idx = dst_batch_offset + global_col * params.m + global_row;
                    dst[dst_idx/VEC_SIZE] = store_val(acc, tn, tm);
                }
            }
        }
    }
}
