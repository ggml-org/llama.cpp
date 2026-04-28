#ifdef USE_SUBGROUP_REDUCTION
enable subgroups;
#endif

struct Params {
    offset_s: u32,
    offset_x: u32,
    offset_dt: u32,
    offset_A: u32,
    offset_B: u32,
    offset_C: u32,
    offset_ids: u32,
    offset_dst: u32,

    stride_s1: u32,
    stride_s2: u32,
    stride_s3: u32,

    stride_x1: u32,
    stride_x2: u32,
    stride_x3: u32,

    stride_dt1: u32,
    stride_dt2: u32,

    a_ne0: u32,
    stride_A1: u32,

    stride_B1: u32,
    stride_B2: u32,
    stride_B3: u32,

    stride_C1: u32,
    stride_C2: u32,
    stride_C3: u32,

    d_state: u32,
    d_inner: u32,
    n_head: u32,
    n_group: u32,
    n_seq_tokens: u32,
    n_seqs: u32,

    y_elems: u32,
};

@group(0) @binding(0) var<storage, read_write> s_in: array<f32>;
#ifdef XBC_OVERLAP
@group(0) @binding(1) var<storage, read_write> x_B_C_merged: array<f32>;
@group(0) @binding(2) var<storage, read_write> dt: array<f32>;
@group(0) @binding(3) var<storage, read_write> A: array<f32>;
@group(0) @binding(4) var<storage, read_write> ids: array<i32>;
@group(0) @binding(5) var<storage, read_write> dst: array<f32>;
@group(0) @binding(6) var<uniform> params: Params;
#else
@group(0) @binding(1) var<storage, read_write> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> dt: array<f32>;
@group(0) @binding(3) var<storage, read_write> A: array<f32>;
@group(0) @binding(4) var<storage, read_write> B: array<f32>;
@group(0) @binding(5) var<storage, read_write> C: array<f32>;
@group(0) @binding(6) var<storage, read_write> ids: array<i32>;
@group(0) @binding(7) var<storage, read_write> dst: array<f32>;
@group(0) @binding(8) var<uniform> params: Params;
#endif

var<workgroup> shared_x_dt: array<f32, TOKENS_PER_TILE>;
var<workgroup> shared_dtsp: array<f32, TOKENS_PER_TILE>;
var<workgroup> shared_reduce: array<f32, TOKENS_PER_TILE * WG_SIZE>;

fn reduce_base(token_in_tile: u32) -> u32 {
    return token_in_tile * WG_SIZE;
}

@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>
#ifdef USE_SUBGROUP_REDUCTION
  , @builtin(subgroup_id) subgroup_id: u32,
    @builtin(subgroup_invocation_id) subgroup_invocation_id: u32,
    @builtin(num_subgroups) num_subgroups: u32
#endif
) {
    let tid = local_id.x;
    let wg_linear = wg_id.y * num_wg.x + wg_id.x;

    let i1 = wg_linear % params.d_inner;
    let head_seq = wg_linear / params.d_inner;
    let ir = head_seq % params.n_head;
    let i3 = head_seq / params.n_head;

    let state_slot = u32(ids[params.offset_ids + i3]);
    let g = ir / (params.n_head / params.n_group);

    let s_idx = params.offset_s + tid + i1 * params.stride_s1 + ir * params.stride_s2 + state_slot * params.stride_s3;
    var s_prev = s_in[s_idx];

    let A0 = A[params.offset_A + (tid % params.a_ne0) + ir * params.stride_A1];

    for (var token_base = 0u; token_base < params.n_seq_tokens; token_base += TOKENS_PER_TILE) {
        if (tid < TOKENS_PER_TILE) {
            let token = token_base + tid;
            if (token < params.n_seq_tokens) {
                let x_idx = params.offset_x + i1 + ir * params.stride_x1 + token * params.stride_x2 + i3 * params.stride_x3;
                let dt_idx = params.offset_dt + ir + token * params.stride_dt1 + i3 * params.stride_dt2;
                let dt0 = dt[dt_idx];
                let dtsp = select(log(1.0 + exp(dt0)), dt0, dt0 > 20.0);
                shared_dtsp[tid] = dtsp;
#ifdef XBC_OVERLAP
                shared_x_dt[tid] = x_B_C_merged[x_idx] * dtsp;
#else
                shared_x_dt[tid] = x[x_idx] * dtsp;
#endif
            }
        }

        workgroupBarrier();

        for (var token_in_tile = 0u; token_in_tile < TOKENS_PER_TILE; token_in_tile++) {
            let token = token_base + token_in_tile;
            if (token >= params.n_seq_tokens) {
                break;
            }

            let x_dt = shared_x_dt[token_in_tile];
            let dA = exp(shared_dtsp[token_in_tile] * A0);
            let reduce_idx = reduce_base(token_in_tile) + tid;

            let b_idx = params.offset_B + tid + g * params.stride_B1 + token * params.stride_B2 + i3 * params.stride_B3;
            let c_idx = params.offset_C + tid + g * params.stride_C1 + token * params.stride_C2 + i3 * params.stride_C3;
#ifdef XBC_OVERLAP
            let s = s_prev * dA + x_B_C_merged[b_idx] * x_dt;
#else
            let s = s_prev * dA + B[b_idx] * x_dt;
#endif
            s_prev = s;

#ifdef USE_SUBGROUP_REDUCTION
#ifdef XBC_OVERLAP
            let subgroup_partial = subgroupAdd(s * x_B_C_merged[c_idx]);
#else
            let subgroup_partial = subgroupAdd(s * C[c_idx]);
#endif
            if (subgroup_invocation_id == 0u) {
                shared_reduce[reduce_idx - tid + subgroup_id] = subgroup_partial;
            }
#else
#ifdef XBC_OVERLAP
            shared_reduce[reduce_idx] = s * x_B_C_merged[c_idx];
#else
            shared_reduce[reduce_idx] = s * C[c_idx];
#endif
#endif

            workgroupBarrier();

#ifdef USE_SUBGROUP_REDUCTION
            if (tid == 0u) {
                var sum = 0.0;
                for (var sg = 0u; sg < num_subgroups; sg++) {
                    sum += shared_reduce[reduce_base(token_in_tile) + sg];
                }
                let y_idx =
                    params.offset_dst + i1 + ir * params.d_inner + token * (params.n_head * params.d_inner) +
                    i3 * (params.n_seq_tokens * params.n_head * params.d_inner);
                dst[y_idx] = sum;
            }
#else
            for (var stride = WG_SIZE / 2u; stride > 0u; stride >>= 1u) {
                if (tid < stride) {
                    shared_reduce[reduce_idx] += shared_reduce[reduce_idx + stride];
                }
                workgroupBarrier();
            }

            if (tid == 0u) {
                let y_idx =
                    params.offset_dst + i1 + ir * params.d_inner + token * (params.n_head * params.d_inner) +
                    i3 * (params.n_seq_tokens * params.n_head * params.d_inner);
                dst[y_idx] = shared_reduce[reduce_base(token_in_tile)];
            }
#endif

            workgroupBarrier();
        }
    }

    let state_idx =
        params.offset_dst + params.y_elems + tid + i1 * params.d_state + ir * (params.d_state * params.d_inner) +
        i3 * (params.d_state * params.d_inner * params.n_head);
    dst[state_idx] = s_prev;
}
