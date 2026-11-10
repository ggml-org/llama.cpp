@group(0) @binding(0)
var<storage, read_write> src_q: array<f32>;

@group(0) @binding(1)
var<storage, read_write> src_k: array<f32>;

@group(0) @binding(2)
var<storage, read_write> src_v: array<f32>;

@group(0) @binding(3)
var<storage, read_write> src_g: array<f32>;

@group(0) @binding(4)
var<storage, read_write> src_beta: array<f32>;

@group(0) @binding(5)
var<storage, read_write> src_state: array<f32>;

@group(0) @binding(6)
var<storage, read_write> dst: array<f32>;

#ifdef FUSED_CACHE
@group(0) @binding(7)
var<storage, read_write> dst_fuse: array<f32>;
#define DST_SNAP dst_fuse
#define NEXT_BINDING 8
#else
#define DST_SNAP dst
#define NEXT_BINDING 7
#endif

#ifdef REPLAY
// pending tokens to run before the new ones (see ggml_gated_delta_net)
@group(0) @binding(NEXT_BINDING)
var<storage, read_write> pend_k: array<f32>;

@group(0) @binding(NEXT_BINDING + 1)
var<storage, read_write> pend_v: array<f32>;

@group(0) @binding(NEXT_BINDING + 2)
var<storage, read_write> pend_gb: array<f32>;

@group(0) @binding(NEXT_BINDING + 3)
var<storage, read_write> n_rep: array<i32>;

var<workgroup> wg_n_rep: u32;

#define PARAMS_BINDING (NEXT_BINDING + 4)
#else
#define PARAMS_BINDING NEXT_BINDING
#endif

struct Params {
    h: u32,
    n_tokens: u32,
    n_seqs: u32,
    s_off: u32,

    sq1: u32,
    sq2: u32,
    sq3: u32,

    sv1: u32,
    sv2: u32,
    sv3: u32,

    sb1: u32,
    sb2: u32,
    sb3: u32,

    neq1: u32,
    rq3: u32,
    K: u32,
    scale: f32,
    dst_fuse_nb2: u32,
    dst_fuse_off: u32,

    n_pend: u32,
    keep: u32,

    spk1: u32,
    spk2: u32,
    spk3: u32,

    spv1: u32,
    spv2: u32,
    spv3: u32,

    spg1: u32,
    spg2: u32,
    spg3: u32,
};

@group(0) @binding(PARAMS_BINDING)
var<uniform> params: Params;

var<workgroup> sh_k: array<f32, S_V>;
var<workgroup> sh_q: array<f32, S_V>;
var<workgroup> sh_g: array<f32, S_V>;

@compute @workgroup_size(WG_SIZE)
fn main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let head_id = workgroup_id.x;
    let seq_id = workgroup_id.y;
    let col = local_id.x;

    let iq1 = head_id % params.neq1;
    let iq3 = seq_id / params.rq3;

    let state_size = S_V * S_V;
    // input state holds s0 only [S_v, S_v, H, n_seqs]: per-seq stride is H*D.
    let state_in_base = (seq_id * params.h + head_id) * state_size;
    let state_out_base = (seq_id * params.h + head_id) * state_size;

#ifdef FUSED_CACHE
    let state_size_per_snap = params.dst_fuse_nb2;
    let snap_off = params.dst_fuse_off;
#else
    let state_size_per_snap = state_size * params.h * params.n_seqs;
    let snap_off = params.s_off;
#endif

    var state: array<f32, S_V>;
    for (var i = 0u; i < S_V; i++) {
        state[i] = src_state[state_in_base + col * S_V + i];
    }

#ifdef REPLAY
    if (col == 0u) {
        wg_n_rep = u32(n_rep[seq_id]);
    }
    let n_r = workgroupUniformLoad(&wg_n_rep);

    for (var t = 0u; t < n_r; t++) {
        let k_off = seq_id * params.spk3 + t * params.spk2 + iq1 * params.spk1;
        let v_off = seq_id * params.spv3 + t * params.spv2 + head_id * params.spv1;
        let g_off = seq_id * params.spg3 + t * params.spg2 + head_id * params.spg1;

        sh_k[col] = pend_k[k_off + col];

#ifdef KDA
        sh_g[col] = exp(pend_gb[g_off + col]);
#endif

        workgroupBarrier();

        let v_val = pend_v[v_off + col];
        let beta_val = pend_gb[g_off + params.n_pend * params.spg2];

        // same update as below, without the attention output
        var kv_col = 0.0;
        var delta_col = 0.0;

#ifdef KDA
        for (var i = 0u; i < S_V; i++) {
            kv_col += (sh_g[i] * state[i]) * sh_k[i];
        }

        delta_col = (v_val - kv_col) * beta_val;

        for (var i = 0u; i < S_V; i++) {
            state[i] = sh_g[i] * state[i] + sh_k[i] * delta_col;
        }
#else
        let g_val = exp(pend_gb[g_off]);

        for (var i = 0u; i < S_V; i++) {
            kv_col += state[i] * sh_k[i];
        }

        delta_col = (v_val - g_val * kv_col) * beta_val;

        for (var i = 0u; i < S_V; i++) {
            state[i] = g_val * state[i] + sh_k[i] * delta_col;
        }
#endif

        workgroupBarrier();
    }

    if (params.keep != 0u) {
        for (var i = 0u; i < S_V; i++) {
            DST_SNAP[snap_off + state_out_base + col * S_V + i] = state[i];
        }
    }
#endif

    var attn_off = (seq_id * params.n_tokens * params.h + head_id) * S_V;

    for (var t = 0u; t < params.n_tokens; t++) {
        let q_off = iq3 * params.sq3 + t * params.sq2 + iq1 * params.sq1;
        let k_off = q_off;
        let v_off = seq_id * params.sv3 + t * params.sv2 + head_id * params.sv1;
        let gb_off = seq_id * params.sb3 + t * params.sb2 + head_id * params.sb1;

        sh_q[col] = src_q[q_off + col];
        sh_k[col] = src_k[k_off + col];

#ifdef KDA
        let g_base = gb_off * S_V;
        sh_g[col] = exp(src_g[g_base + col]);
#endif

        workgroupBarrier();

        let v_val = src_v[v_off + col];
        let beta_val = src_beta[gb_off];

        var kv_col = 0.0;
        var delta_col = 0.0;
        var attn_col = 0.0;

#ifdef KDA
        for (var i = 0u; i < S_V; i++) {
            kv_col += (sh_g[i] * state[i]) * sh_k[i];
        }

        delta_col = (v_val - kv_col) * beta_val;

        for (var i = 0u; i < S_V; i++) {
            state[i] = sh_g[i] * state[i] + sh_k[i] * delta_col;
            attn_col += state[i] * sh_q[i];
        }
#else
        let g_val = exp(src_g[gb_off]);

        for (var i = 0u; i < S_V; i++) {
            kv_col += state[i] * sh_k[i];
        }

        delta_col = (v_val - g_val * kv_col) * beta_val;

        for (var i = 0u; i < S_V; i++) {
            state[i] = g_val * state[i] + sh_k[i] * delta_col;
            attn_col += state[i] * sh_q[i];
        }
#endif

        dst[attn_off + col] = attn_col * params.scale;
        attn_off += S_V * params.h;

        if (params.K > 1u) {
            // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
            let target_slot = i32(params.n_tokens) - 1 - i32(t);
            if (target_slot >= 0 && target_slot < i32(params.K)) {
                let slot_base = snap_off + u32(target_slot) * state_size_per_snap + state_out_base;
                for (var i = 0u; i < S_V; i++) {
                    DST_SNAP[slot_base + col * S_V + i] = state[i];
                }
            }
        }

        workgroupBarrier();
    }

#ifdef REPLAY
    // replay with keep: the state output is the state after the replay
    let write_state = params.K == 1u && params.keep == 0u;
#else
    let write_state = params.K == 1u;
#endif

    if (write_state) {
        for (var i = 0u; i < S_V; i++) {
            DST_SNAP[snap_off + state_out_base + col * S_V + i] = state[i];
        }
    }
}
