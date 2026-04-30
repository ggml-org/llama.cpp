#ifdef USE_SUBGROUP_REDUCTION
enable subgroups;
#endif
enable f16;

#define DECLARE_BYTE_LOADERS_SRC0
#include "common_decls.tmpl"

#include "mul_mat_vec_acc.tmpl"

struct MulMatIdVecParams {
    offset_src0: u32,
    offset_src1: u32,
    offset_ids: u32,
    offset_dst: u32,

    k: u32,
    m: u32,
    n_expert: u32,
    n_expert_used: u32,
    b_ne1: u32,

    stride_01: u32,
    stride_11: u32,
    stride_02: u32,
    stride_12: u32,
};

@group(0) @binding(0) var<storage, read_write> src0: array<SRC0_TYPE>; // [cols, rows, n_expert]
@group(0) @binding(1) var<storage, read_write> src1: array<SRC1_TYPE>; // [cols, b_ne1, n_tokens(1)]
@group(0) @binding(2) var<storage, read_write> ids: array<u32>;        // [n_experd_used, n_tokens(1)]
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;   // [rows, n_expert_used, n_tokens(1)]

// "mul_mat_vec_acc.tmpl" requires params.k, params.m, params.stride_01
@group(0) @binding(4) var<uniform> params: MulMatIdVecParams;

// Flattened as [row][thread] to keep each row's reduction contiguous in memory.
var<workgroup> partial_sums: array<f32, OUTPUTS_PER_WG * WG_SIZE>;

fn partial_index(row: u32, thread: u32) -> u32 {
    return row * WG_SIZE + thread;
}

var<workgroup> gathered_count_ids: array<u32, N_EXPERTS>;
var<workgroup> gathered_expert_used: array<u32, N_EXPERTS>;

@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>
#ifdef USE_SUBGROUP_REDUCTION
  , @builtin(subgroup_id) subgroup_id: u32,
    @builtin(subgroup_invocation_id) subgroup_invocation_id: u32,
    @builtin(num_subgroups) num_subgroups: u32,
    @builtin(subgroup_size) subgroup_size: u32
#endif
) {

    let thread_id = local_id.x;

    for (var i = thread_id;i < params.n_expert;i += WG_SIZE) {
        gathered_count_ids[i] = 0;
    }

    workgroupBarrier();

    // gather the selected experts for the target token.
    for (var col = thread_id;col < params.n_expert_used;col += WG_SIZE) {
        let expert = ids[params.offset_ids + col];
        gathered_count_ids[expert] = 1;
        gathered_expert_used[expert] = col;
    }

    workgroupBarrier();

    let output_groups:u32 = (params.m + OUTPUTS_PER_WG - 1u) / OUTPUTS_PER_WG;
    let wg_linear = wg_id.y * num_wg.x + wg_id.x;

    var own_expert:u32 = 0;
    var wg_in_batch:u32 = 0;
    var wg_sum:u32 = 0;

    for (var i = 0u;i < params.n_expert;i += 1) {
        let wg_vec_count = gathered_count_ids[i]; // 1 or 0
        let wg_per_matrix = output_groups * wg_vec_count;
        if (wg_sum <= wg_linear && wg_linear < wg_sum + wg_per_matrix) {
            own_expert = i;
            wg_in_batch = wg_linear - wg_sum;
            break;
        }
        wg_sum += wg_per_matrix;
    }

    let row_base = (wg_linear % output_groups) * OUTPUTS_PER_WG;
    let dst1_stride = params.m;

    let src0_batch_offset = params.offset_src0 + own_expert * params.stride_02;
    let src1_idx_base = params.offset_src1 + (gathered_expert_used[own_expert] % params.b_ne1) * params.stride_11;
    let dst_idx_base = params.offset_dst + gathered_expert_used[own_expert] * dst1_stride + row_base;

    let acc = accumulate_vec_dot(thread_id, row_base, src0_batch_offset, src1_idx_base);

#ifdef USE_SUBGROUP_REDUCTION
    for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
        let subgroup_total = subgroupAdd(acc[row]);
        if (subgroup_invocation_id == 0u) {
            partial_sums[partial_index(row, subgroup_id)] = subgroup_total;
        }
    }

    workgroupBarrier();

    for (var row = subgroup_id; (row < OUTPUTS_PER_WG) && (row_base + row < params.m); row += num_subgroups) {
        let output_row = row_base + row;
        var row_acc = 0.0f;
        for (var k = subgroup_invocation_id; k < num_subgroups; k += subgroup_size) {
            row_acc += partial_sums[partial_index(row, k)];
        }
        let row_total = subgroupAdd(row_acc);
        if (subgroup_invocation_id == 0) {
            dst[dst_idx_base + row] = row_total;
        }
    }
#endif

#ifdef USE_WORKGROUP_REDUCTION
    for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
        partial_sums[partial_index(row, thread_id)] = acc[row];
    }

    workgroupBarrier();

    var stride:u32 = WG_SIZE / 2u;

    while (stride > 0) {
        if (thread_id < stride) {
            for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
                partial_sums[partial_index(row, thread_id)] += partial_sums[partial_index(row, thread_id + stride)];
            }
        }

        workgroupBarrier();
        stride = stride / 2;
    }

    if (thread_id < OUTPUTS_PER_WG) {
        let output_row = row_base + thread_id;
        if (output_row < params.m) {
            dst[dst_idx_base + thread_id] = partial_sums[partial_index(thread_id, 0)];
        }
    }
#endif
}
