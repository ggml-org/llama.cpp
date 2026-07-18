#ifdef USE_SUBGROUP_REDUCTION
enable subgroups;
#endif
enable f16;

requires packed_4x8_integer_dot_product;

#include "common_decls.tmpl"

struct Params {
    offset_src1: u32,
    stride_11: u32,
    stride_12: u32,
    stride_13: u32,
    ne0: u32,
    ne1: u32,
    ne2: u32,
    ne3: u32,
};

#define SRC1_TYPE vec4<SRC1_INNER_TYPE>

@group(0) @binding(0) var<storage, read_write> src1: array<SRC1_TYPE>;
@group(0) @binding(1) var<storage, read_write> src1q: array<q8_1>;

@group(0) @binding(2) var<uniform> params: Params;

#ifdef USE_SUBGROUP_REDUCTION
fn cluster_max_8(v: f32) -> f32 {
    var r = v;
    r = max(r, subgroupShuffleXor(r, 1u));
    r = max(r, subgroupShuffleXor(r, 2u));
    r = max(r, subgroupShuffleXor(r, 4u));
    return r;
}

#if defined(MUL_ACC_Q4_0) || defined(MUL_ACC_Q4_1) || defined(MUL_ACC_Q4_K)
fn cluster_add_i4x8(v: i32) -> i32 {
    var r= v;
    r += subgroupShuffleXor(r, 1u);
    r += subgroupShuffleXor(r, 2u);
    r += subgroupShuffleXor(r, 4u);
    return r;
}
#endif
#endif

#ifdef USE_WORKGROUP_REDUCTION
#define CLUSTER_SIZE 8

var<workgroup> partial_amaxs: array<array<f32, CLUSTER_SIZE>, WG_SIZE / CLUSTER_SIZE>;
var<workgroup> partial_sums:  array<array<i32, CLUSTER_SIZE>, WG_SIZE / CLUSTER_SIZE>;
#endif

@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>
) {
    let thread_id = local_id.x;
    let ne0_vec4 = params.ne0 / 4u;

    let wg_per_vec = (ne0_vec4 + (WG_SIZE - 1u)) / WG_SIZE;
    let total_batches = wg_per_vec * params.ne1 * params.ne2 * params.ne3;

    let wg_linear = wg_id.y * num_wg.x + wg_id.x;
    if (wg_linear >= total_batches) {
        return;
    }

    let vec_idx = wg_linear / wg_per_vec;
    let src13_idx = vec_idx / (params.ne2 * params.ne1);
    let vec_ne12_num       = vec_idx % (params.ne2 * params.ne1);
    let src12_idx = vec_ne12_num / params.ne1;
    let src11_idx = vec_ne12_num % params.ne1;
    let src1_idx_base = params.offset_src1 + src13_idx * params.stride_13 + src12_idx * params.stride_12 + src11_idx * params.stride_11;
    let src1_idx_vec4_base = src1_idx_base / 4u;

    let blocks_per_row = params.ne0 / 32u;
    let blocks_per_wg = (WG_SIZE * 4u) / 32u;
    let src1q_idx_base = ((src13_idx * params.ne2 + src12_idx) * params.ne1 + src11_idx) * blocks_per_row;
    let src11_wg_idx = wg_linear % wg_per_vec;
    let src1q_idx = src1q_idx_base + src11_wg_idx * blocks_per_wg + thread_id / 8u;
    let qs_idx = thread_id % 8u;

    // reduction
    var q4 = vec4<f32>(0.0);
    var q4_quants = 0u;
    var thread_amax = 0.0;

    let src11_vec4_idx = src11_wg_idx * WG_SIZE + thread_id;
    let is_valid = src11_vec4_idx < ne0_vec4;

#ifdef USE_SUBGROUP_REDUCTION

    var d = 0.0;

    if (is_valid) {
        q4 = src1[src1_idx_vec4_base + src11_vec4_idx];
        let abs_q4 = abs(q4);
        thread_amax = max(max(abs_q4[0u], abs_q4[1u]), max(abs_q4[2], abs_q4[3]));
    }

    d = cluster_max_8(thread_amax) / 127.0;

    if (is_valid) {
        let id = select(0.0, 1.0 / d, d > 0.0);
        q4_quants = pack4xI8(vec4<i32>(round(q4 * id)));
        if (qs_idx == 0u) {
            src1q[src1q_idx].d = f16(d);
        }
        src1q[src1q_idx].qs[qs_idx] = q4_quants;
    }

#if defined(MUL_ACC_Q4_0) || defined(MUL_ACC_Q4_1) || defined(MUL_ACC_Q4_K)
    let q4_quants_sum = dot4I8Packed(q4_quants, 0x01010101u);
    let s = f16(d * f32(cluster_add_i4x8(q4_quants_sum)));

    if (is_valid) {
        if (qs_idx == 0u) {
            src1q[src1q_idx].s = s;
        }
    }
#endif
#endif

#ifdef USE_WORKGROUP_REDUCTION

    var d = 0.0;
    let cluster_id = thread_id / 8u;

    if (is_valid) {
        q4 = src1[src1_idx_vec4_base + src11_vec4_idx];
        let abs_q4 = abs(q4);
        thread_amax = max(max(abs_q4[0], abs_q4[1]), max(abs_q4[2], abs_q4[3]));
        partial_amaxs[cluster_id][qs_idx] = thread_amax;
    }

    workgroupBarrier();

    if (is_valid) {
        let amax = max(
                    max(
                        max(partial_amaxs[cluster_id][0], partial_amaxs[cluster_id][1]), max(partial_amaxs[cluster_id][2], partial_amaxs[cluster_id][3])),
                    max(
                        max(partial_amaxs[cluster_id][4], partial_amaxs[cluster_id][5]), max(partial_amaxs[cluster_id][6], partial_amaxs[cluster_id][7]))
                );

        d = amax / 127.0;
        let id = select(0.0f, 1.0f / d, d > 0.0f);

        q4_quants = pack4xI8(vec4<i32>(round(q4 * id)));
        src1q[src1q_idx].qs[qs_idx] = q4_quants;

        if (qs_idx == 0u) {
            src1q[src1q_idx].d = f16(d);
        }
    }

#if defined(MUL_ACC_Q4_0) || defined(MUL_ACC_Q4_1) || defined(MUL_ACC_Q4_K)

    partial_sums[cluster_id][qs_idx] = dot4I8Packed(q4_quants, 0x01010101u);

    workgroupBarrier();

    if (is_valid) {
        if (qs_idx == 0u) {
            let s = d * f32(partial_sums[cluster_id][0] + partial_sums[cluster_id][1] + partial_sums[cluster_id][2] + partial_sums[cluster_id][3]
                                    + partial_sums[cluster_id][4] + partial_sums[cluster_id][5] + partial_sums[cluster_id][6] + partial_sums[cluster_id][7]);
            src1q[src1q_idx].s = f16(s);
        }
    }

#endif
#endif

}
