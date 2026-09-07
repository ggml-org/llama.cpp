enable f16;
enable subgroups;

#define BYTE_HELPERS
#include "flash_attn_decls.tmpl"
#include "common_decls.tmpl"

// Default values
// The actual values are defined in shader-lib.
#define HEAD_DIM_QK 64
#define HEAD_DIM_V 64
#define Q_TILE 4
#define KV_TILE 64
#define WG_SIZE 128

const Q_CHUNKS: u32 = HEAD_DIM_QK / 4u;
const V_CHUNKS: u32 = HEAD_DIM_V / 4u;
const SCORE_REGS_PER_LANE: u32 = (KV_TILE + MIN_SUBGROUP_SIZE - 1u) / MIN_SUBGROUP_SIZE;
const OUT_REGS_PER_LANE: u32 = (V_CHUNKS + MIN_SUBGROUP_SIZE - 1u) / MIN_SUBGROUP_SIZE;

#if !defined(K_DIRECT) || !defined(V_DIRECT)
#define STAGING_SHMEM kv_shmem
#define STAGING_OUT_TYPE f16
#include "flash_attn_staging.tmpl"
const kv_shmem_size = KV_TILE * max(HEAD_DIM_QK, HEAD_DIM_V);
var<workgroup> kv_shmem: array<f16, kv_shmem_size>;
#endif

var<workgroup> q_shmem: array<Q_TYPE, Q_TILE * HEAD_DIM_QK>;
var<workgroup> p_shmem: array<f16, Q_TILE * KV_TILE>;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(subgroup_id) subgroup_id: u32,
        @builtin(subgroup_size) subgroup_size: u32,
        @builtin(num_subgroups) num_subgroups: u32,
        @builtin(subgroup_invocation_id) sg_inv_id: u32) {
    if (subgroup_size == 0u || num_subgroups < Q_TILE) {
        return;
    }

    let wg_per_head = (params.seq_len_q + Q_TILE - 1u) / Q_TILE;
    let wg_per_batch = wg_per_head * params.n_heads;

    let dst2_stride = HEAD_DIM_V * params.n_heads;
    let dst3_stride = dst2_stride * params.seq_len_q;

    let batch_idx = wg_id.x / wg_per_batch;
    let q_batch_offset = params.offset_q + batch_idx * params.stride_q3;
    let k_batch_offset = params.offset_k + batch_idx * params.stride_k3;
    let v_batch_offset = params.offset_v + batch_idx * params.stride_v3;
    let dst_batch_offset = params.offset_dst + batch_idx * dst3_stride;
    let wg_in_batch = wg_id.x % wg_per_batch;

    let head_idx = wg_in_batch / wg_per_head;
    let q_head_offset = q_batch_offset + head_idx * params.stride_q2;
    let k_head_idx = head_idx / params.q_per_kv;
    let v_head_offset = v_batch_offset + k_head_idx * params.stride_v2;
    let k_head_offset = k_batch_offset + k_head_idx * params.stride_k2;

    let wg_in_head = wg_in_batch % wg_per_head;
    let q_row_start = wg_in_head * Q_TILE;
    let global_q_row = q_row_start + subgroup_id;
    let row_active = subgroup_id < Q_TILE && global_q_row < params.seq_len_q;

#ifdef MASK
    let mask_global_offset = params.offset_mask + batch_idx * params.stride_mask3 + q_row_start * params.seq_len_kv;
#endif

    let dst_global_offset = dst_batch_offset + q_row_start * dst2_stride + head_idx * HEAD_DIM_V;

    let head = f32(head_idx);
    let slope = select(1.0,
        select(pow(params.m1, 2.0 * (head - params.n_head_log2) + 1.0),
                pow(params.m0, head + 1.0),
                head < params.n_head_log2),
        params.max_bias > 0.0);

    for (var elem_idx = local_id.x; elem_idx < Q_TILE * HEAD_DIM_QK; elem_idx += WG_SIZE) {
        let q_tile_row = elem_idx / HEAD_DIM_QK;
        let q_col = elem_idx % HEAD_DIM_QK;
        let head_q_row = q_row_start + q_tile_row;
        let global_q_row_offset = q_head_offset + head_q_row * params.stride_q1;
        q_shmem[elem_idx] = select(
            0.0,
            Q_TYPE(Q[global_q_row_offset + q_col]) * params.scale,
            head_q_row < params.seq_len_q);
    }

    workgroupBarrier();

    var row_max = FLOAT_MIN;
    var exp_sum = 0.0;
    var out_regs: array<vec4<f32>, OUT_REGS_PER_LANE>;
    for (var reg_idx = 0u; reg_idx < OUT_REGS_PER_LANE; reg_idx += 1u) {
        out_regs[reg_idx] = vec4<f32>(0.0);
    }

    let q_base = subgroup_id * HEAD_DIM_QK;
    let subgroup_p_offset = subgroup_id * KV_TILE;

    for (var kv_tile = 0u; kv_tile < params.seq_len_kv; kv_tile += KV_TILE) {
        let kv_count = min(KV_TILE, params.seq_len_kv - kv_tile);
        let score_slots = min(SCORE_REGS_PER_LANE, (kv_count + subgroup_size - 1u) / subgroup_size);
        let out_slots = min(OUT_REGS_PER_LANE, (V_CHUNKS + subgroup_size - 1u) / subgroup_size);
        var local_scores: array<f32, SCORE_REGS_PER_LANE>;
        for (var slot = 0u; slot < SCORE_REGS_PER_LANE; slot += 1u) {
            local_scores[slot] = FLOAT_MIN;
        }

        // The tile path stages K/V in shared memory so each tile can be reused across
        // Q_TILE query rows. It therefore does not use the direct path.
#ifndef K_DIRECT
        load_k_tile_block(local_id.x, kv_count, kv_tile, k_head_offset);
#endif

        workgroupBarrier();

        var local_max = FLOAT_MIN;
        if (row_active) {
            for (var slot = 0u; slot < score_slots; slot += 1u) {
                let kv_local = sg_inv_id + slot * subgroup_size;
                if (kv_local >= kv_count) {
                    continue;
                }

                let global_k_row = kv_tile + kv_local;
                var dot_val = 0.0;
                for (var chunk = 0u; chunk < Q_CHUNKS; chunk += 1u) {
                    let q_off = q_base + chunk * 4u;
                    let qv = vec4<Q_TYPE>(
                        q_shmem[q_off + 0u],
                        q_shmem[q_off + 1u],
                        q_shmem[q_off + 2u],
                        q_shmem[q_off + 3u]);
                    let kv_off = kv_local * HEAD_DIM_QK + chunk * 4u;
                    let kv = vec4<f16>(
                        kv_shmem[kv_off + 0u],
                        kv_shmem[kv_off + 1u],
                        kv_shmem[kv_off + 2u],
                        kv_shmem[kv_off + 3u]);
                    dot_val += dot(vec4<f32>(qv), vec4<f32>(kv));
                }
#ifdef LOGIT_SOFTCAP
                dot_val = params.logit_softcap * tanh(dot_val);
#endif
#ifdef MASK
                let mask_idx = mask_global_offset + subgroup_id * params.seq_len_kv + global_k_row;
                dot_val += slope * f32(mask[mask_idx]);
#endif
                local_scores[slot] = dot_val;
                local_max = max(local_max, dot_val);
            }
        }

        let tile_max = subgroupMax(local_max);
        let new_max = max(row_max, tile_max);
        let cur_exp = exp(row_max - new_max);
        exp_sum *= cur_exp;
        for (var reg_idx = 0u; reg_idx < OUT_REGS_PER_LANE; reg_idx += 1u) {
            out_regs[reg_idx] *= cur_exp;
        }

        var local_sum = 0.0;
        for (var slot = 0u; slot < score_slots; slot += 1u) {
            let kv_local = sg_inv_id + slot * subgroup_size;
            if (row_active && kv_local < kv_count) {
                let p = exp(local_scores[slot] - new_max);
                p_shmem[subgroup_p_offset + kv_local] = f16(p);
                local_sum += p;
            }
        }

        workgroupBarrier();

        // The tile path stages K/V in shared memory so each tile can be reused across
        // Q_TILE query rows. It therefore does not use the direct path.
#ifndef V_DIRECT
        load_v_tile_block(local_id.x, kv_count, kv_tile, v_head_offset);
#endif

        workgroupBarrier();

        let tile_sum = subgroupAdd(local_sum);
        exp_sum += tile_sum;
        row_max = new_max;

        if (row_active) {
            for (var reg_idx = 0u; reg_idx < out_slots; reg_idx += 1u) {
                let chunk = sg_inv_id + reg_idx * subgroup_size;
                if (chunk >= V_CHUNKS) {
                    continue;
                }

                var acc = out_regs[reg_idx];
                for (var kv_local = 0u; kv_local < kv_count; kv_local += 1u) {
                    let p = f32(p_shmem[subgroup_p_offset + kv_local]);
                    let kv_off = kv_local * HEAD_DIM_V + chunk * 4u;
                    let v4 = vec4<f16>(
                        kv_shmem[kv_off + 0u],
                        kv_shmem[kv_off + 1u],
                        kv_shmem[kv_off + 2u],
                        kv_shmem[kv_off + 3u]);
                    acc += p * vec4<f32>(v4);
                }
                out_regs[reg_idx] = acc;
            }
        }

        workgroupBarrier();
    }

#ifdef SINKS
    if (row_active) {
        let sink_score = sinks[params.offset_sinks + head_idx];
        let sink_max = max(row_max, sink_score);
        let sink_scale = exp(row_max - sink_max);
        for (var reg_idx = 0u; reg_idx < OUT_REGS_PER_LANE; reg_idx += 1u) {
            out_regs[reg_idx] *= sink_scale;
        }
        exp_sum = exp_sum * sink_scale + exp(sink_score - sink_max);
        row_max = sink_max;
    }
#endif

    if (row_active) {
        let inv_exp_sum = select(0.0, 1.0 / exp_sum, exp_sum != 0.0);
        let row_base = dst_global_offset + subgroup_id * dst2_stride;
        let out_slots = min(OUT_REGS_PER_LANE, (V_CHUNKS + subgroup_size - 1u) / subgroup_size);
        for (var reg_idx = 0u; reg_idx < out_slots; reg_idx += 1u) {
            let chunk = sg_inv_id + reg_idx * subgroup_size;
            if (chunk >= V_CHUNKS) {
                continue;
            }
            let dst_vec_index = (row_base + chunk * 4u) >> 2u;
            dst[dst_vec_index] = vec4<DST_TYPE>(out_regs[reg_idx] * inv_exp_sum);
        }
    }
}
