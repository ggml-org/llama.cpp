#ifdef USE_SUBGROUP_REDUCTION
enable subgroups;
#endif
enable f16;

#include "common_decls.tmpl"

#ifdef U32_DEQUANT_HELPERS
#define SRC0_TYPE u32
#endif

#ifdef VEC
#define VEC_SIZE 4u
#define SRC0_TYPE vec4<SRC0_INNER_TYPE>
#define SRC1_TYPE vec4<SRC1_INNER_TYPE>

fn inner_dot(src0_val: SRC0_TYPE, src1_val: SRC1_TYPE) -> f32 {
    return f32(dot(SRC1_TYPE(src0_val), src1_val));
}
#endif

#ifdef SCALAR
#define VEC_SIZE 1u
#define SRC0_TYPE SRC0_INNER_TYPE
#define SRC1_TYPE SRC1_INNER_TYPE

fn inner_dot(src0_val: SRC0_TYPE, src1_val: SRC1_TYPE) -> f32 {
    return f32(src0_val) * f32(src1_val);
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

@group(0) @binding(0) var<storage, read_write> src0: array<SRC0_TYPE>;
@group(0) @binding(1) var<storage, read_write> src1: array<SRC1_TYPE>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

@group(0) @binding(3) var<uniform> params: MulMatParams;

// Flattened as [row][thread] to keep each row's reduction contiguous in memory.
var<workgroup> partial_sums: array<f32, OUTPUTS_PER_WG * WG_SIZE>;

fn partial_index(row: u32, thread: u32) -> u32 {
    return row * WG_SIZE + thread;
}

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

    let total_batches = params.bs02 * params.broadcast2 * params.bs03 * params.broadcast3;
    let wg_linear = wg_id.y * num_wg.x + wg_id.x;
    let output_groups = (params.m + OUTPUTS_PER_WG - 1u) / OUTPUTS_PER_WG;
    let batch_idx = wg_linear / output_groups;
    if (batch_idx >= total_batches) {
        return;
    }

    let row_base = (wg_linear % output_groups) * OUTPUTS_PER_WG;

    let dst2_stride = params.m * params.n;
    let dst2_idx = batch_idx % (params.bs02 * params.broadcast2);
    let dst3_stride = dst2_stride * params.bs02 * params.broadcast2;
    let dst3_idx = batch_idx / (params.bs02 * params.broadcast2);
    let src03_idx = dst3_idx / params.broadcast3;
    let src13_idx = dst3_idx;
    let src02_idx = dst2_idx / params.broadcast2;
    let src12_idx = dst2_idx;

    let src0_batch_offset = params.offset_src0 + src03_idx * params.stride_03 + src02_idx * params.stride_02;
    let src1_idx_base = params.offset_src1 + src13_idx * params.stride_13 + src12_idx * params.stride_12;
    let dst_idx_base = params.offset_dst + dst3_idx * dst3_stride + dst2_idx * dst2_stride + row_base;

    var acc: array<f32, OUTPUTS_PER_WG>;

#ifdef MUL_ACC_FLOAT
    let k_vec = params.k / VEC_SIZE;
    let src1_idx_base_vec = src1_idx_base / VEC_SIZE;

    // Each thread walks K, loads from the vector, and updates
    // a small block of output rows held in registers.
    for (var k = thread_id; k < k_vec; k += WG_SIZE) {
        let x = src1[src1_idx_base_vec + k];
        for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
            let output_row = row_base + row;
            if (output_row < params.m) {
                let src0_idx = (src0_batch_offset + output_row * params.stride_01) / VEC_SIZE + k;
                acc[row] += inner_dot(src0[src0_idx], x);
            }
        }
    }
#endif

#ifdef MUL_ACC_Q4_0
#define BLOCK_SIZE 32
#define BLOCK_SIZE_BYTES 18
#define THREADS_PER_BLOCK 4
#define ELEMS_PER_THREAD (BLOCK_SIZE/THREADS_PER_BLOCK)

    let num_blocks = params.k / BLOCK_SIZE;
    let thread_within_block = thread_id % 4;
    for (var block = thread_id/THREADS_PER_BLOCK; block < num_blocks; block += WG_SIZE/THREADS_PER_BLOCK) {
        let x_base = src1_idx_base + block * BLOCK_SIZE + thread_within_block * 4;
        var x_block: array<f32, ELEMS_PER_THREAD>;
        for (var i = 0u; i < ELEMS_PER_THREAD / 2; i++) {
            x_block[i] = f32(src1[x_base + i]);
            x_block[i + 4] = f32(src1[x_base + i + 16]);
        }

        for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
            let output_row = row_base + row;
            if (output_row < params.m) {
                let block_byte_base = (src0_batch_offset + output_row * params.stride_01 + block) * BLOCK_SIZE_BYTES;
                let d = f32(load_src0_f16_at(block_byte_base));
                var row_sum = 0.0;

                let q_packed = load_src0_u32_at(block_byte_base + 2u + 4u * thread_within_block);
                for (var byte_idx = 0u; byte_idx < 4u; byte_idx++) {
                    let q_byte = get_byte(q_packed, byte_idx);
                    let q_lo = (f32(q_byte & 0xFu) - 8.0) * d;
                    let q_hi = (f32((q_byte >> 4u) & 0xFu) - 8.0) * d;
                    row_sum += q_lo * x_block[byte_idx];
                    row_sum += q_hi * x_block[byte_idx + 4u];
                }
                acc[row] += row_sum;
            }
        }
    }
#endif

#ifdef MUL_ACC_Q4_1
#define BLOCK_SIZE 32
#define BLOCK_SIZE_BYTES 20
#define THREADS_PER_BLOCK 4
#define ELEMS_PER_THREAD (BLOCK_SIZE/THREADS_PER_BLOCK)

    let num_blocks = params.k / BLOCK_SIZE;
    let thread_within_block = thread_id % THREADS_PER_BLOCK;
    for (var block = thread_id / THREADS_PER_BLOCK; block < num_blocks; block += WG_SIZE / THREADS_PER_BLOCK) {
        let x_base = src1_idx_base + block * BLOCK_SIZE + thread_within_block * 4;
        var x_block: array<f32, ELEMS_PER_THREAD>;
        for (var i = 0u; i < ELEMS_PER_THREAD / 2; i++) {
            x_block[i] = f32(src1[x_base + i]);
            x_block[i + 4] = f32(src1[x_base + i + 16]);
        }

        for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
            let output_row = row_base + row;
            if (output_row < params.m) {
                let block_byte_base = (src0_batch_offset + output_row * params.stride_01 + block) * BLOCK_SIZE_BYTES;
                let d = f32(load_src0_f16_at(block_byte_base));
                let m = f32(load_src0_f16_at(block_byte_base + 2u));
                var row_sum = 0.0;

                let q_packed = load_src0_u32_at(block_byte_base + 4u + 4u * thread_within_block);
                for (var byte_idx = 0u; byte_idx < 4u; byte_idx++) {
                    let q_byte = get_byte(q_packed, byte_idx);
                    let q_lo = f32(q_byte & 0xFu) * d + m;
                    let q_hi = f32((q_byte >> 4u) & 0xFu) * d + m;
                    row_sum += q_lo * x_block[byte_idx];
                    row_sum += q_hi * x_block[byte_idx + 4u];
                }
                acc[row] += row_sum;
            }
        }
    }
#endif

#ifdef MUL_ACC_Q5_0
#define BLOCK_SIZE 32
#define BLOCK_SIZE_BYTES 22
#define THREADS_PER_BLOCK 4
#define ELEMS_PER_THREAD (BLOCK_SIZE/THREADS_PER_BLOCK)

    let num_blocks = params.k / BLOCK_SIZE;
    let thread_within_block = thread_id % THREADS_PER_BLOCK;
    for (var block = thread_id / THREADS_PER_BLOCK; block < num_blocks; block += WG_SIZE / THREADS_PER_BLOCK) {
        let x_base = src1_idx_base + block * BLOCK_SIZE + thread_within_block * 4;
        var x_block: array<f32, ELEMS_PER_THREAD>;
        for (var i = 0u; i < ELEMS_PER_THREAD / 2; i++) {
            x_block[i] = f32(src1[x_base + i]);
            x_block[i + 4] = f32(src1[x_base + i + 16]);
        }

        for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
            let output_row = row_base + row;
            if (output_row < params.m) {
                let block_byte_base = (src0_batch_offset + output_row * params.stride_01 + block) * BLOCK_SIZE_BYTES;
                let d = f32(load_src0_f16_at(block_byte_base));
                let qh_packed = load_src0_u32_at(block_byte_base + 2u);
                let q_packed = load_src0_u32_at(block_byte_base + 6u + 4u * thread_within_block);
                let qh_shift = thread_within_block * 4u;
                var row_sum = 0.0;

                for (var byte_idx = 0u; byte_idx < 4u; byte_idx++) {
                    let q_byte = get_byte(q_packed, byte_idx);
                    let qh_lo = ((qh_packed >> (qh_shift + byte_idx)) << 4u) & 0x10u;
                    let qh_hi = (qh_packed >> (qh_shift + byte_idx + 12u)) & 0x10u;
                    let q_lo = (f32((q_byte & 0xFu) | qh_lo) - 16.0) * d;
                    let q_hi = (f32(((q_byte >> 4u) & 0xFu) | qh_hi) - 16.0) * d;
                    row_sum += q_lo * x_block[byte_idx];
                    row_sum += q_hi * x_block[byte_idx + 4u];
                }
                acc[row] += row_sum;
            }
        }
    }
#endif

#ifdef MUL_ACC_Q5_1
#define BLOCK_SIZE 32
#define BLOCK_SIZE_BYTES 24
#define THREADS_PER_BLOCK 4
#define ELEMS_PER_THREAD (BLOCK_SIZE/THREADS_PER_BLOCK)

    let num_blocks = params.k / BLOCK_SIZE;
    let thread_within_block = thread_id % THREADS_PER_BLOCK;
    for (var block = thread_id / THREADS_PER_BLOCK; block < num_blocks; block += WG_SIZE / THREADS_PER_BLOCK) {
        let x_base = src1_idx_base + block * BLOCK_SIZE + thread_within_block * 4;
        var x_block: array<f32, ELEMS_PER_THREAD>;
        for (var i = 0u; i < ELEMS_PER_THREAD / 2; i++) {
            x_block[i] = f32(src1[x_base + i]);
            x_block[i + 4] = f32(src1[x_base + i + 16]);
        }

        for (var row = 0u; row < OUTPUTS_PER_WG; row++) {
            let output_row = row_base + row;
            if (output_row < params.m) {
                let block_byte_base = (src0_batch_offset + output_row * params.stride_01 + block) * BLOCK_SIZE_BYTES;
                let d = f32(load_src0_f16_at(block_byte_base));
                let m = f32(load_src0_f16_at(block_byte_base + 2u));
                let qh_packed = load_src0_u32_at(block_byte_base + 4u);
                let q_packed = load_src0_u32_at(block_byte_base + 8u + 4u * thread_within_block);
                let qh_shift = thread_within_block * 4u;
                var row_sum = 0.0;

                for (var byte_idx = 0u; byte_idx < 4u; byte_idx++) {
                    let q_byte = get_byte(q_packed, byte_idx);
                    let qh_lo = ((qh_packed >> (qh_shift + byte_idx)) << 4u) & 0x10u;
                    let qh_hi = (qh_packed >> (qh_shift + byte_idx + 12u)) & 0x10u;
                    let q_lo = f32((q_byte & 0xFu) | qh_lo) * d + m;
                    let q_hi = f32(((q_byte >> 4u) & 0xFu) | qh_hi) * d + m;
                    row_sum += q_lo * x_block[byte_idx];
                    row_sum += q_hi * x_block[byte_idx + 4u];
                }
                acc[row] += row_sum;
            }
        }
    }
#endif

#ifdef MUL_ACC_Q8_0
#define BLOCK_SIZE 32
#define BLOCK_SIZE_BYTES 34
#define THREADS_PER_BLOCK 4
#define ELEMS_PER_THREAD (BLOCK_SIZE/THREADS_PER_BLOCK)

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
                let d = f32(load_src0_f16_at(block_byte_base));
                var row_sum = 0.0;

                for (var packed_idx = 0u; packed_idx < ELEMS_PER_THREAD / 4u; packed_idx++) {
                    let q_packed = load_src0_u32_at(block_byte_base + 2u + 4u * (thread_within_block * 2u + packed_idx));
                    for (var byte_idx = 0u; byte_idx < 4u; byte_idx++) {
                        let q_val = f32(get_byte_i32(q_packed, byte_idx)) * d;
                        row_sum += q_val * x_block[packed_idx * 4u + byte_idx];
                    }
                }
                acc[row] += row_sum;
            }
        }
    }
#endif

#ifdef MUL_ACC_Q8_1
#define BLOCK_SIZE 32
#define BLOCK_SIZE_BYTES 36
#define THREADS_PER_BLOCK 4
#define ELEMS_PER_THREAD (BLOCK_SIZE/THREADS_PER_BLOCK)

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
                let d = f32(load_src0_f16_at(block_byte_base));
                let m = f32(load_src0_f16_at(block_byte_base + 2u));
                var row_sum = 0.0;

                for (var packed_idx = 0u; packed_idx < ELEMS_PER_THREAD / 4u; packed_idx++) {
                    let q_packed = load_src0_u32_at(block_byte_base + 4u + 4u * (thread_within_block * 2u + packed_idx));
                    for (var byte_idx = 0u; byte_idx < 4u; byte_idx++) {
                        let q_val = f32(get_byte_i32(q_packed, byte_idx)) * d + m;
                        row_sum += q_val * x_block[packed_idx * 4u + byte_idx];
                    }
                }
                acc[row] += row_sum;
            }
        }
    }
#endif

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

    var stride = WG_SIZE / 2u;

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
