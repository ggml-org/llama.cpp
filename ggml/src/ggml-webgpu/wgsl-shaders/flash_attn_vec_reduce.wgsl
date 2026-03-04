enable f16;
enable subgroups;

// Default values
#define HEAD_DIM_V 64
#define WG_SIZE 128

struct Params {
    // Total rows to reduce: nrows = batch * n_heads * seq_len_q.
    nrows: u32,
    seq_len_q: u32,
    n_heads: u32,
    // Number of split workgroups used in the vec-split pass.
    // Each split contributes one partial (o, l, m) per row.
    nwg: u32,
    // Bases into tmp for partial output vectors and partial stats.
    tmp_data_base: u32,
    tmp_stats_base: u32,
};

@group(0) @binding(0) var<storage, read_write> tmp: array<f32>;
@group(0) @binding(1) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Params;

const FLOAT_MIN: f32 = -1.0e9;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>,
        @builtin(subgroup_id) subgroup_id: u32,
        @builtin(num_subgroups) num_subgroups: u32,
        @builtin(subgroup_size) subgroup_size: u32,
        @builtin(subgroup_invocation_id) sg_inv_id: u32) {
    // One workgroup reduces one logical output row rid.
    let rid = wg_id.x;
    if (rid >= params.nrows) {
        return;
    }

    // Decode flattened row id back to (batch, head, q_row).
    let rows_per_batch = params.n_heads * params.seq_len_q;
    let batch_idx = rid / rows_per_batch;
    let rem = rid % rows_per_batch;
    let head_idx = rem / params.seq_len_q;
    let q_row = rem % params.seq_len_q;

    let dst2_stride = HEAD_DIM_V * params.n_heads;
    let dst3_stride = dst2_stride * params.seq_len_q;
    let row_base = batch_idx * dst3_stride + q_row * dst2_stride + head_idx * HEAD_DIM_V;

    // Each subgroup lane corresponds to one split index g in [0, nwg).
    // This kernel requires params.nwg <= subgroup_size.
    let lane = sg_inv_id;
    if (params.nwg > subgroup_size) {
        return;
    }

    // Load split stats for this row:
    //   si = l_g (exp sum), mi = m_g (row max) from split g.
    let stats_base = params.tmp_stats_base + rid * (2u * params.nwg);
    let active_lane = lane < params.nwg;
    let si = select(0.0,      tmp[stats_base + 2u * lane + 0u], active_lane);
    let mi = select(FLOAT_MIN, tmp[stats_base + 2u * lane + 1u], active_lane);

    // Merge split softmax normalizers:
    //   m = max_g m_g
    //   l = sum_g l_g * exp(m_g - m)
    let m = subgroupMax(mi);
    let ms = select(0.0, exp(mi - m), active_lane);
    let s = subgroupAdd(si * ms);
    let inv_s = select(0.0, 1.0 / s, s != 0.0);

    // Merge partial output vectors:
    //   O = (sum_g O_g * exp(m_g - m)) / l
    let row_tmp_base = params.tmp_data_base + rid * (HEAD_DIM_V * params.nwg);
    for (var elem_base = subgroup_id * 4u; elem_base < HEAD_DIM_V; elem_base += num_subgroups * 4u) {
        var weighted = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        if (active_lane) {
            let src = row_tmp_base + lane * HEAD_DIM_V + elem_base;
            weighted = vec4<f32>(tmp[src + 0u], tmp[src + 1u], tmp[src + 2u], tmp[src + 3u]) * ms;
        }

        let sum_x = subgroupAdd(weighted.x);
        let sum_y = subgroupAdd(weighted.y);
        let sum_z = subgroupAdd(weighted.z);
        let sum_w = subgroupAdd(weighted.w);

        // Lane 0 writes the final normalized vec4 chunk.
        if (lane == 0u) {
            let dst_vec_index = (row_base + elem_base) >> 2u;
            dst[dst_vec_index] = vec4<f32>(sum_x, sum_y, sum_z, sum_w) * inv_s;
        }
    }
}
