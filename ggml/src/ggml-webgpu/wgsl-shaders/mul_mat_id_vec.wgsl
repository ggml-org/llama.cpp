#ifdef USE_SUBGROUP_REDUCTION
enable subgroups;
#endif
enable f16;

#define DECLARE_BYTE_LOADERS_SRC0
#include "common_decls.tmpl"

#define SRC0_TYPE u32

fn byte_of(v: u32, b: u32) -> u32 {
    return (v >> (b * 8u)) & 0xFFu;
}

fn sbyte_of(v: u32, b: u32) -> i32 {
    let raw = i32((v >> (b * 8u)) & 0xFFu);
    return select(raw, raw - 256, raw >= 128);
}

#define VEC_SIZE 1u
#define SRC0_TYPE SRC0_INNER_TYPE
#define SRC1_TYPE SRC1_INNER_TYPE

fn inner_dot(src0_val: SRC0_TYPE, src1_val: SRC1_TYPE) -> f32 {
    return f32(src0_val) * f32(src1_val);
}

#ifdef MUL_ACC_Q8_0
#define BLOCK_SIZE 32
#define BLOCK_SIZE_BYTES 34
#define THREADS_PER_BLOCK 4
#define ELEMS_PER_THREAD (BLOCK_SIZE/THREADS_PER_BLOCK)
fn accumulate_vec_dot(thread_id: u32, row_base: u32, src0_batch_offset: u32, src1_idx_base: u32) -> array<f32, OUTPUTS_PER_WG> {
    var acc: array<f32, OUTPUTS_PER_WG>;

    let num_blocks = params.k / BLOCK_SIZE;
    let thread_within_block = thread_id % THREADS_PER_BLOCK;
        for (var block = thread_id / THREADS_PER_BLOCK; block < num_blocks; block += WG_SIZE / THREADS_PER_BLOCK) {
            let x_base = src1_idx_base + block * BLOCK_SIZE + thread_within_block * ELEMS_PER_THREAD;
            var x_block: array<f32, ELEMS_PER_THREAD>;
            for (var i = 0u; i < ELEMS_PER_THREAD; i++) {
                x_block[i] = f32(src1[x_base + i]);
            }

            for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
                let output_row = row_base + row;
                if (output_row < params.m) {
                    let block_byte_base = (src0_batch_offset + output_row * params.stride_01 + block) * BLOCK_SIZE_BYTES;
                    let d = f32(load_f16_at_src0(block_byte_base));
                    var row_sum = 0.0;

                    for (var packed_idx = 0u; packed_idx < ELEMS_PER_THREAD / 4u; packed_idx++) {
                        let q_packed = load_u32_at_src0(block_byte_base + 2u + 4u * (thread_within_block * 2u + packed_idx));
                        for (var byte_idx = 0u; byte_idx < 4u; byte_idx++) {
                            let q_val = f32(get_byte_i32(q_packed, byte_idx)) * d;
                            row_sum += q_val * x_block[packed_idx * 4u + byte_idx];
                        }
                    }
                    acc[row] += row_sum;
                }
            }
        }

    return acc;
}
#endif

#ifdef MUL_ACC_Q8_1
#define BLOCK_SIZE 32
#define BLOCK_SIZE_BYTES 36
#define THREADS_PER_BLOCK 4
#define ELEMS_PER_THREAD (BLOCK_SIZE/THREADS_PER_BLOCK)
fn accumulate_vec_dot(thread_id: u32, row_base: u32, src0_batch_offset: u32, src1_idx_base: u32) -> array<f32, OUTPUTS_PER_WG> {
    var acc: array<f32, OUTPUTS_PER_WG>;

    let num_blocks = params.k / BLOCK_SIZE;
    let thread_within_block = thread_id % THREADS_PER_BLOCK;
        for (var block = thread_id / THREADS_PER_BLOCK; block < num_blocks; block += WG_SIZE / THREADS_PER_BLOCK) {
            let x_base = src1_idx_base + block * BLOCK_SIZE + thread_within_block * ELEMS_PER_THREAD;
            var x_block: array<f32, ELEMS_PER_THREAD>;
            for (var i = 0u; i < ELEMS_PER_THREAD; i++) {
                x_block[i] = f32(src1[x_base + i]);
            }

            for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
                let output_row = row_base + row;
                if (output_row < params.m) {
                    let block_byte_base = (src0_batch_offset + output_row * params.stride_01 + block) * BLOCK_SIZE_BYTES;
                    let d = f32(load_f16_at_src0(block_byte_base));
                    let m = f32(load_f16_at_src0(block_byte_base + 2u));
                    var row_sum = 0.0;

                    for (var packed_idx = 0u; packed_idx < ELEMS_PER_THREAD / 4u; packed_idx++) {
                        let q_packed = load_u32_at_src0(block_byte_base + 4u + 4u * (thread_within_block * 2u + packed_idx));
                        for (var byte_idx = 0u; byte_idx < 4u; byte_idx++) {
                            let q_val = f32(get_byte_i32(q_packed, byte_idx)) * d + m;
                            row_sum += q_val * x_block[packed_idx * 4u + byte_idx];
                        }
                    }
                    acc[row] += row_sum;
                }
            }
        }

    return acc;
}
#endif

#ifdef MUL_ACC_Q6_K
#define BLOCK_SIZE 256
#define BLOCK_SIZE_BYTES 210
#define THREADS_PER_BLOCK 16

fn accumulate_vec_dot(thread_id: u32, row_base: u32, src0_batch_offset: u32, src1_idx_base: u32) -> array<f32, OUTPUTS_PER_WG> {
    var acc: array<f32, OUTPUTS_PER_WG>;

    let tid = thread_id % THREADS_PER_BLOCK;
    let block_group = thread_id / THREADS_PER_BLOCK;
    let num_block_groups: u32 = WG_SIZE / THREADS_PER_BLOCK;

    let ip = tid / 8u;
    let il = tid % 8u;
    let l0 = 4u * il;
    let is = 8u * ip + l0 / 16u;

    let y_offset = 128u * ip + l0;
    let q_offset_l = 64u * ip + l0;
    let q_offset_h = 32u * ip + l0;

    let num_blocks = params.k / BLOCK_SIZE;
    let sc_base_byte = 192u + (is & ~3u);
    let sc_byte_pos = is & 3u;

        for (var block = block_group; block < num_blocks; block += num_block_groups) {
            let x_base = src1_idx_base + block * BLOCK_SIZE + y_offset;
            var x_block: array<f32, 16>;
            for (var l = 0u; l < 4u; l++) {
                x_block[l]       = f32(src1[x_base + l]);
                x_block[l + 4u]  = f32(src1[x_base + 32u + l]);
                x_block[l + 8u]  = f32(src1[x_base + 64u + l]);
                x_block[l + 12u] = f32(src1[x_base + 96u + l]);
            }

            for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
                let output_row = row_base + row;
                if (output_row < params.m) {
                    let block_byte_base = (src0_batch_offset + output_row * params.stride_01 + block) * BLOCK_SIZE_BYTES;

                    let d = f32(load_f16_at_src0(block_byte_base + 208u));
                    let ql1_u32 = load_u32_at_src0(block_byte_base + q_offset_l);
                    let ql2_u32 = load_u32_at_src0(block_byte_base + q_offset_l + 32u);
                    let qh_u32 = load_u32_at_src0(block_byte_base + 128u + q_offset_h);
                    let sc_u32_0 = load_u32_at_src0(block_byte_base + sc_base_byte);
                    let sc_u32_1 = load_u32_at_src0(block_byte_base + sc_base_byte + 4u);

                    let sc0 = sbyte_of(sc_u32_0, sc_byte_pos);
                    let sc2 = sbyte_of(sc_u32_0, sc_byte_pos + 2u);
                    let sc4 = sbyte_of(sc_u32_1, sc_byte_pos);
                    let sc6 = sbyte_of(sc_u32_1, sc_byte_pos + 2u);

                    var sums = vec4<f32>(0.0, 0.0, 0.0, 0.0);

                    for (var l = 0u; l < 4u; l++) {
                        let q1b = byte_of(ql1_u32, l);
                        let q2b = byte_of(ql2_u32, l);
                        let qhb = byte_of(qh_u32, l);

                        let dq0 = f32(i32((q1b & 0x0Fu) | ((qhb & 0x03u) << 4u)) - 32);
                        let dq1 = f32(i32((q2b & 0x0Fu) | ((qhb & 0x0Cu) << 2u)) - 32);
                        let dq2 = f32(i32((q1b >> 4u) | (qhb & 0x30u)) - 32);
                        let dq3 = f32(i32((q2b >> 4u) | ((qhb & 0xC0u) >> 2u)) - 32);

                        sums[0] += x_block[l] * dq0;
                        sums[1] += x_block[l + 4u] * dq1;
                        sums[2] += x_block[l + 8u] * dq2;
                        sums[3] += x_block[l + 12u] * dq3;
                    }

                    acc[row] += d * (sums[0] * f32(sc0) + sums[1] * f32(sc2) +
                                     sums[2] * f32(sc4) + sums[3] * f32(sc6));
                }
            }
        }

    return acc;
}
#endif

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

    var own_expert:u32 = 0xFFFFFFFFu;
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
