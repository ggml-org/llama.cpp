enable f16;

#define DECLARE_BYTE_LOADERS_SRC0
#include "common_decls.tmpl"

#include "mul_mat_decls.tmpl"

#ifdef VEC
fn store_val(acc: array<array<f16, TILE_M>, TILE_N>, tn: u32, tm: u32) -> vec4<f32> {
    return vec4<f32>(f32(acc[tn][tm]), f32(acc[tn][tm + 1]), f32(acc[tn][tm + 2]), f32(acc[tn][tm + 3]));
}
#endif

#ifdef SCALAR
fn store_val(acc: array<array<f16, TILE_M>, TILE_N>, tn: u32, tm: u32) -> f32 {
    return f32(acc[tn][tm]);
}
#endif

struct MulMatIdParams {
    offset_src0: u32,
    offset_src1: u32,
    offset_dst: u32,

    k: u32,
    m: u32,
    n_expert: u32,
    n_expert_used: u32,
    n_tokens: u32,
    b_ne1: u32,

    stride_01: u32,
    stride_11: u32,
    stride_02: u32,
    stride_12: u32,
};

@group(0) @binding(0) var<storage, read_write> src0: array<SRC0_TYPE>; // [cols, rows, n_expert]
@group(0) @binding(1) var<storage, read_write> src1: array<SRC1_TYPE>; // [cols, b_ne1, n_tokens]
@group(0) @binding(2) var<storage, read_write> dst: array<DST_TYPE>;   // [rows, n_expert_used, n_tokens]
@group(0) @binding(3) var<storage, read_write> global_gathered_expert_used: array<u32>; // [n_expert][n_tokens]
@group(0) @binding(4) var<storage, read_write> global_gathered_tokens: array<u32>; // [n_expert][n_tokens]
@group(0) @binding(5) var<storage, read_write> gathered_count_ids: array<u32>; // [n_expert]

@group(0) @binding(6) var<uniform> params: MulMatIdParams;

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
var<workgroup> gathered_expert_used: array<u32, TILE_N * WORKGROUP_SIZE_N>;
var<workgroup> gathered_tokens: array<u32, TILE_N * WORKGROUP_SIZE_N>;

#ifdef INIT_SRC1_SHMEM_FLOAT
fn init_shmem_id_src1(thread_id: u32, offset_src1: u32, rest_token_n: u32, k_outer: u32) {
    for (var elem_idx = thread_id * VEC_SIZE; elem_idx < TILE_SRC1_SHMEM; elem_idx += TOTAL_WORKGROUP_SIZE * VEC_SIZE) {
        let tile_n = elem_idx / TILE_K;
        let tile_k = elem_idx % TILE_K;
        if (tile_n < rest_token_n) {
            let global_src10 = k_outer + tile_k;
            let expert_used_idx = gathered_expert_used[tile_n] % params.b_ne1;
            let token_idx = gathered_tokens[tile_n];
            let src1_idx = offset_src1 + token_idx * params.stride_12 + expert_used_idx * params.stride_11 + global_src10;
            let src1_val = select(
                SRC1_TYPE(0.0),
                src1[src1_idx/VEC_SIZE],
                global_src10 < params.k);
            store_shmem(SHMEM_TYPE(src1_val), TILE_SRC0_SHMEM + elem_idx);
        } else {
            store_shmem(SHMEM_TYPE(0.0), TILE_SRC0_SHMEM + elem_idx);
        }
    }
}
#endif // INIT_SRC1_SHMEM_FLOAT

@compute @workgroup_size(TOTAL_WORKGROUP_SIZE)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(num_workgroups) num_wg: vec3<u32>) {

    let thread_id = local_id.x;
    let local_m = get_local_m(thread_id);
    let local_n = get_local_n(thread_id);

    var expert_idx:u32 = 0xFFFFFFFFu;
    var wg_in_batch:u32 = 0;
    var wg_sum:u32 = 0;
    let wg_m_count = (params.m + WORKGROUP_SIZE_M * TILE_M - 1u) / (WORKGROUP_SIZE_M * TILE_M);
    let wg_linear = wg_id.y * num_wg.x + wg_id.x;

    for (var i = 0u;i < params.n_expert;i += 1) {
        let wg_n_count = (gathered_count_ids[i] + WORKGROUP_SIZE_N * TILE_N - 1u) / (WORKGROUP_SIZE_N * TILE_N);
        let wg_per_matrix = wg_m_count * wg_n_count;
        if (wg_sum <= wg_linear && wg_linear < wg_sum + wg_per_matrix) {
            expert_idx = i;
            wg_in_batch = wg_linear - wg_sum;
            break;
        }
        wg_sum += wg_per_matrix;
    }

    let is_valid = expert_idx != 0xFFFFFFFFu;

    var wg_m: u32 = 0;
    var wg_n: u32 = 0;
    var offset_wg_m: u32 = 0;
    var offset_wg_n: u32 = 0;
    var rest_token_n: u32 = 0;
    var src0_batch_offset: u32 = 0;

    wg_m = wg_in_batch % wg_m_count;
    wg_n = wg_in_batch / wg_m_count;

    offset_wg_m = wg_m * WORKGROUP_SIZE_M * TILE_M;
    offset_wg_n = wg_n * WORKGROUP_SIZE_N * TILE_N;

    if (is_valid) {
        rest_token_n = gathered_count_ids[expert_idx] - offset_wg_n;
        let global_gathered_base = expert_idx * params.n_tokens + offset_wg_n;
        for (var i = thread_id; i < TILE_N * WORKGROUP_SIZE_N && offset_wg_n + i < gathered_count_ids[expert_idx]; i += TOTAL_WORKGROUP_SIZE) {
            gathered_expert_used[i] = global_gathered_expert_used[global_gathered_base + i];
            gathered_tokens[i] = global_gathered_tokens[global_gathered_base + i];
        }
        src0_batch_offset = params.offset_src0 + expert_idx * params.stride_02;
    }

    workgroupBarrier();

    let output_row_base = offset_wg_m + local_m * TILE_M;
    let output_col_base = offset_wg_n + local_n * TILE_N;

    let dst2_stride = params.m * params.n_expert_used;
    let dst1_stride = params.m;

    var acc: array<array<f16, TILE_M>, TILE_N>;

    for (var k_outer = 0u; k_outer < params.k; k_outer += TILE_K) {

        if (is_valid) {
            init_shmem_src0(thread_id, src0_batch_offset, offset_wg_m, k_outer);
            init_shmem_id_src1(thread_id, params.offset_src1, rest_token_n, k_outer);
        }

        workgroupBarrier();

        if (is_valid) {
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
                        acc[tn][tm] += src0_tile[tm] * src1_val;
                    }
                }
            }
        }

        workgroupBarrier();
    }

    if (is_valid) {
        for (var tn = 0u; tn < TILE_N; tn++) {
            let n_idx = output_col_base + tn;
            if (n_idx < gathered_count_ids[expert_idx]) {
                let dst1_idx = gathered_expert_used[n_idx - offset_wg_n];
                let dst2_idx = gathered_tokens[n_idx - offset_wg_n];
                let dst12_offset = params.offset_dst + dst2_idx * dst2_stride + dst1_idx * dst1_stride;
                for (var tm = 0u; tm < TILE_M; tm += VEC_SIZE) {
                    let global_row = output_row_base + tm;
                    if (global_row < params.m) {
                        let dst_idx = dst12_offset + global_row;
                        dst[dst_idx/VEC_SIZE] = store_val(acc, tn, tm);
                    }
                }
            }
        }
    }
}
