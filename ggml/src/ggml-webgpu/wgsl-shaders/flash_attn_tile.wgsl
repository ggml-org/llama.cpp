enable f16;
enable subgroups;

#ifdef Q_F16
#define Q_TYPE f16
#else
#define Q_TYPE f32
#endif

#ifdef KV_F32
#define KV_TYPE f32
#else
#define KV_TYPE f16
#endif

#ifdef DST_F16
#define DST_TYPE f16
#else
#define DST_TYPE f32
#endif

#define HEAD_DIM_QK 64
#define HEAD_DIM_V 64
#define KV_STAGE_STRIDE 64
#define Q_TILE 4
#define KV_TILE 64
#define WG_SIZE 128
#ifndef MIN_SUBGROUP_SIZE
#define MIN_SUBGROUP_SIZE MAX_SUBGROUP_SIZE
#endif

struct Params {
    offset_q: u32,
    offset_k: u32,
    offset_v: u32,
    offset_mask: u32,
    offset_sinks: u32,
    offset_dst: u32,

    n_heads: u32,
    seq_len_q: u32,
    seq_len_kv: u32,

    stride_q1: u32,
    stride_q2: u32,
    stride_q3: u32,
    stride_k1: u32,
    stride_k2: u32,
    stride_k3: u32,
    stride_v1: u32,
    stride_v2: u32,
    stride_v3: u32,
    stride_mask3: u32,

    q_per_kv: u32,

    scale: f32,
    max_bias: f32,
    logit_softcap: f32,
    n_head_log2: f32,
    m0: f32,
    m1: f32,
};

@group(0) @binding(0) var<storage, read_write> Q: array<Q_TYPE>;
#ifdef KV_OVERLAP
@group(0) @binding(1) var<storage, read_write> K: array<vec4<KV_TYPE>>;
#define V K
#else
@group(0) @binding(1) var<storage, read_write> K: array<vec4<KV_TYPE>>;
@group(0) @binding(2) var<storage, read_write> V: array<vec4<KV_TYPE>>;
#endif

#if defined(MASK) && defined(SINKS)
#ifdef KV_OVERLAP
@group(0) @binding(2) var<storage, read_write> mask: array<f16>;
@group(0) @binding(3) var<storage, read_write> sinks: array<f32>;
#define DST_BINDING 4
#define PARAMS_BINDING 5
#else
@group(0) @binding(3) var<storage, read_write> mask: array<f16>;
@group(0) @binding(4) var<storage, read_write> sinks: array<f32>;
#define DST_BINDING 5
#define PARAMS_BINDING 6
#endif
#elif defined(MASK)
#ifdef KV_OVERLAP
@group(0) @binding(2) var<storage, read_write> mask: array<f16>;
#define DST_BINDING 3
#define PARAMS_BINDING 4
#else
@group(0) @binding(3) var<storage, read_write> mask: array<f16>;
#define DST_BINDING 4
#define PARAMS_BINDING 5
#endif
#elif defined(SINKS)
#ifdef KV_OVERLAP
@group(0) @binding(2) var<storage, read_write> sinks: array<f32>;
#define DST_BINDING 3
#define PARAMS_BINDING 4
#else
@group(0) @binding(3) var<storage, read_write> sinks: array<f32>;
#define DST_BINDING 4
#define PARAMS_BINDING 5
#endif
#else
#ifdef KV_OVERLAP
#define DST_BINDING 2
#define PARAMS_BINDING 3
#else
#define DST_BINDING 3
#define PARAMS_BINDING 4
#endif
#endif

@group(0) @binding(DST_BINDING) var<storage, read_write> dst: array<vec4<DST_TYPE>>;
@group(0) @binding(PARAMS_BINDING) var<uniform> params: Params;

const FLOAT_MIN: f32 = -1.0e9;
const Q_CHUNKS: u32 = HEAD_DIM_QK / 4u;
const V_CHUNKS: u32 = HEAD_DIM_V / 4u;
const SCORE_REGS_PER_LANE: u32 = (KV_TILE + MIN_SUBGROUP_SIZE - 1u) / MIN_SUBGROUP_SIZE;
const OUT_REGS_PER_LANE: u32 = (V_CHUNKS + MIN_SUBGROUP_SIZE - 1u) / MIN_SUBGROUP_SIZE;

var<workgroup> q_shmem: array<f32, Q_TILE * HEAD_DIM_QK>;
var<workgroup> kv_shmem: array<f32, KV_TILE * KV_STAGE_STRIDE>;
var<workgroup> p_shmem: array<f32, Q_TILE * KV_TILE>;

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
            f32(Q[global_q_row_offset + q_col]) * params.scale,
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

        for (var vec_idx_local = local_id.x; vec_idx_local < kv_count * Q_CHUNKS; vec_idx_local += WG_SIZE) {
            let kv_local = vec_idx_local / Q_CHUNKS;
            let chunk = vec_idx_local % Q_CHUNKS;
            let global_k_row = kv_tile + kv_local;
            let k_vec_index = (k_head_offset + global_k_row * params.stride_k1 + chunk * 4u) >> 2u;
            let k4 = K[k_vec_index];
            let kv_off = kv_local * KV_STAGE_STRIDE + chunk * 4u;
            kv_shmem[kv_off + 0u] = f32(k4.x);
            kv_shmem[kv_off + 1u] = f32(k4.y);
            kv_shmem[kv_off + 2u] = f32(k4.z);
            kv_shmem[kv_off + 3u] = f32(k4.w);
        }

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
                    let qv = vec4<f32>(
                        q_shmem[q_off + 0u],
                        q_shmem[q_off + 1u],
                        q_shmem[q_off + 2u],
                        q_shmem[q_off + 3u]);
                    let kv_off = kv_local * KV_STAGE_STRIDE + chunk * 4u;
                    let kv = vec4<f32>(
                        kv_shmem[kv_off + 0u],
                        kv_shmem[kv_off + 1u],
                        kv_shmem[kv_off + 2u],
                        kv_shmem[kv_off + 3u]);
                    dot_val += dot(qv, kv);
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
                p_shmem[subgroup_p_offset + kv_local] = p;
                local_sum += p;
            }
        }

        workgroupBarrier();

        for (var vec_idx_local = local_id.x; vec_idx_local < kv_count * V_CHUNKS; vec_idx_local += WG_SIZE) {
            let kv_local = vec_idx_local / V_CHUNKS;
            let chunk = vec_idx_local % V_CHUNKS;
            let global_v_row = kv_tile + kv_local;
            let v_vec_index = (v_head_offset + global_v_row * params.stride_v1 + chunk * 4u) >> 2u;
            let v4 = V[v_vec_index];
            let kv_off = kv_local * KV_STAGE_STRIDE + chunk * 4u;
            kv_shmem[kv_off + 0u] = f32(v4.x);
            kv_shmem[kv_off + 1u] = f32(v4.y);
            kv_shmem[kv_off + 2u] = f32(v4.z);
            kv_shmem[kv_off + 3u] = f32(v4.w);
        }

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
                    let p = p_shmem[subgroup_p_offset + kv_local];
                    let kv_off = kv_local * KV_STAGE_STRIDE + chunk * 4u;
                    let v4 = vec4<f32>(
                        kv_shmem[kv_off + 0u],
                        kv_shmem[kv_off + 1u],
                        kv_shmem[kv_off + 2u],
                        kv_shmem[kv_off + 3u]);
                    acc += p * v4;
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
