diagnostic(off, chromium.subgroup_matrix_uniformity);
enable f16;
enable subgroups;
enable chromium_experimental_subgroup_matrix;

struct Params {
    offset_q: u32,
    offset_k: u32,
    offset_v: u32,
    offset_mask: u32,
    offset_sinks: u32,
    offset_dst: u32,

    // shapes of Q/K/V
    head_dim_qk: u32,
    head_dim_v: u32,
    n_heads: u32,
    seq_len_q: u32,
    seq_len_kv: u32,

    // strides (in elements)
    stride_q1: u32,
    stride_q2: u32,
    stride_q3: u32,
    stride_k1: u32,
    stride_k2: u32,
    stride_k3: u32,
    stride_v1: u32,
    stride_v2: u32,
    stride_v3: u32,

    // TODO: still need to consider broadcast

    // softmax params
    scale: f32,
    max_bias: f32,
    n_head_log2: f32,
    m0: f32,
    m1: f32,
};

@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> K: array<f32>;
@group(0) @binding(2) var<storage, read> V: array<f32>;
@group(0) @binding(3) var<storage, read> mask: array<f16>;
@group(0) @binding(4) var<storage, read> sinks: array<f32>;
@group(0) @binding(5) var<storage, read_write> dst: array<f32>;

// The number of Q rows processed per workgroup
const Q_TILE = 8u;
var<workgroup> q_shmem: array<f16, Q_TILE * 64>; // assuming max head_dim_qk of 64

const KV_TILE = 16u;
// we can reuse the same shmem for K and V since we only need one at a time right?
var<workgroup> kv_shmem: array<f16, KV_TILE * 64>; // assuming max head_dim_qkv of 64

const WG_SIZE = 32;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(subgroup_id) subgroup_id: u32) {

          dst[0] = Q[0] + K[0] + V[0] + f32(mask[0]) + sinks[0]; // dummy line to avoid compile error

        // workgroups per head
        // batch index

        // load q into shared memory

        // for each kv tile
            // load k into shared memory

            // compute qk scores
            // apply mask
            // softmax

            // load v into shared memory

            // compute output

        // write output to dst
}