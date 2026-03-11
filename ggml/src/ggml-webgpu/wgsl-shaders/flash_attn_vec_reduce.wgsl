enable f16;
enable subgroups;

// Default values
#define HEAD_DIM_V 64
#define WG_SIZE 128

struct Params {
    nrows: u32,
    seq_len_q: u32,
    n_heads: u32,
    offset_dst: u32,
    nwg: u32,
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
    let rid = wg_id.x;
    if (rid >= params.nrows) {
        return;
    }

    let rows_per_batch = params.n_heads * params.seq_len_q;
    let batch_idx = rid / rows_per_batch;
    let rem = rid % rows_per_batch;
    let head_idx = rem / params.seq_len_q;
    let q_row = rem % params.seq_len_q;

    let dst2_stride = HEAD_DIM_V * params.n_heads;
    let dst3_stride = dst2_stride * params.seq_len_q;
    let row_base = params.offset_dst + batch_idx * dst3_stride + q_row * dst2_stride + head_idx * HEAD_DIM_V;

    let thread = sg_inv_id;
    if (params.nwg > subgroup_size) {
        return;
    }

    let stats_base = params.tmp_stats_base + rid * (2u * params.nwg);
    let active_thread = thread < params.nwg;
    let si = select(0.0, tmp[stats_base + 2u * thread + 0u], active_thread);
    let mi = select(FLOAT_MIN, tmp[stats_base + 2u * thread + 1u], active_thread);
    let m = subgroupMax(mi);
    let ms = select(0.0, exp(mi - m), active_thread);
    let s = subgroupAdd(si * ms);
    let inv_s = select(0.0, 1.0 / s, s != 0.0);

    let row_tmp_base = params.tmp_data_base + rid * (HEAD_DIM_V * params.nwg);
    for (var elem_base = subgroup_id * 4u; elem_base < HEAD_DIM_V; elem_base += num_subgroups * 4u) {
        var weighted = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        if (active_thread) {
            let src = row_tmp_base + thread * HEAD_DIM_V + elem_base;
            weighted = vec4<f32>(tmp[src + 0u], tmp[src + 1u], tmp[src + 2u], tmp[src + 3u]) * ms;
        }

        let sum_x = subgroupAdd(weighted.x);
        let sum_y = subgroupAdd(weighted.y);
        let sum_z = subgroupAdd(weighted.z);
        let sum_w = subgroupAdd(weighted.w);

        if (thread == 0u) {
            let dst_vec_index = (row_base + elem_base) >> 2u;
            dst[dst_vec_index] = vec4<f32>(sum_x, sum_y, sum_z, sum_w) * inv_s;
        }
    }
}
