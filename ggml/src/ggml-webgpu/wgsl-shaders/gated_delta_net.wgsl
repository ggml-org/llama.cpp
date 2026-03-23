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
    scale: f32,
};

@group(0) @binding(7)
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
    let state_base = (seq_id * params.h + head_id) * state_size;

    var state: array<f32, S_V>;
    for (var i = 0u; i < S_V; i++) {
        state[i] = src_state[state_base + col * S_V + i];
    }

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

        workgroupBarrier();
    }

    for (var i = 0u; i < S_V; i++) {
        dst[params.s_off + state_base + col * S_V + i] = state[i];
    }
}
