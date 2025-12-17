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
@group(0) @binding(6) var<storage, read_write> debug: array<f32>;
@group(0) @binding(7) var<uniform> params: Params;

// The number of Q rows processed per workgroup
const Q_TILE = 8u;
var<workgroup> q_shmem: array<f32, Q_TILE * 64>; // assumes max head_dim_qk of 64

const KV_TILE = 8u;
// we can reuse the same shmem for K and V since we only need one at a time right?
var<workgroup> k_shmem: array<f32, KV_TILE * 64>; // assuming max head_dim_qkv of 64
var<workgroup> v_shmem: array<f32, KV_TILE * 64>; // assuming max head_dim_qkv of 64

var<workgroup> o_shmem: array<f32, Q_TILE * 64>; // output shmem

// storage for output of Q*K^T scores for online softmax (S matrix from paper)
// also storage for diagonal matrix during online softmax (P matrix from paper)
// note that we reuse the same storage for both since we only need one at a time
var<workgroup> inter_shmem: array<f32, Q_TILE * KV_TILE>;

const WG_SIZE = 32u;
const SUBGROUP_SIZE = 32u;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(subgroup_id) subgroup_id: u32,
        @builtin(subgroup_invocation_id) sg_inv_id: u32) {

    debug[0] = 42;

    // each thread maintains its own cache for softmax intermediates
    var row_max = array<f32, Q_TILE>(-3.4e38, -3.4e38, -3.4e38, -3.4e38, -3.4e38, -3.4e38, -3.4e38, -3.4e38);
    var exp_sum = array<f32, Q_TILE>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

    // workgroups per head/batch
    let wg_per_head = (params.seq_len_q + Q_TILE - 1u) / Q_TILE;
    let wg_per_batch = wg_per_head * params.n_heads;

    let dst2_stride = params.head_dim_v * params.n_heads;
    let dst3_stride = dst2_stride * params.seq_len_q;

    // batch index
    let batch_idx = wg_id.x / wg_per_batch;
    let q_batch_offset = params.offset_q + batch_idx * params.stride_q3;
    let k_batch_offset = params.offset_k + batch_idx * params.stride_k3;
    let v_batch_offset = params.offset_v + batch_idx * params.stride_v3;
    let dst_batch_offset = params.offset_dst + batch_idx * dst3_stride;
    let wg_in_batch = wg_id.x % wg_per_batch;

    // head index
    let head_idx = wg_in_batch / wg_per_head;
    let q_head_offset = q_batch_offset + head_idx * params.stride_q2;
    let k_head_offset = k_batch_offset + head_idx * params.stride_k2;
    let v_head_offset = v_batch_offset + head_idx * params.stride_v2;
    let wg_in_head = wg_in_batch % wg_per_head;

    // starting Q row for this workgroup
    let q_row_start = wg_in_head * Q_TILE;

    // note that the output is permuted, the layout is [head_dim_v, n_heads, seq_len_q, batch_size]
    let dst_global_offset = dst_batch_offset + q_row_start * dst2_stride + head_idx * params.head_dim_v;

    // Which mask row to use. TODO: support broadcasting
    let mask_seq_offset = params.offset_mask + q_row_start * params.seq_len_kv;

    let head = f32(head_idx);
    let slope = select(1.0, select(pow(params.m1, 2.0 * (head - params.n_head_log2) + 1.0), pow(params.m0, head + 1.0), head < params.n_head_log2), params.max_bias > 0);

    // load q tile into shared memory
    for (var elem_idx = local_id.x; elem_idx < Q_TILE * params.head_dim_qk; elem_idx += WG_SIZE) {
        let q_row = elem_idx / params.head_dim_qk;
        let q_col = elem_idx % params.head_dim_qk;
        let head_q_row = q_row_start + q_row;
        let global_q_row_offset = q_head_offset + head_q_row * params.stride_q1;
        let q_val: f32 = select(
            0.0,
            Q[global_q_row_offset + q_col],
            head_q_row < params.seq_len_q && q_col < params.head_dim_qk);
        q_shmem[elem_idx] = q_val;
    }

    workgroupBarrier();

    for (var kv_tile = 0u; kv_tile < params.seq_len_kv; kv_tile += KV_TILE) {
      // load k tile into shared memory
      for (var elem_idx = local_id.x; elem_idx < KV_TILE * params.head_dim_qk; elem_idx += WG_SIZE) {
          let k_row = elem_idx / params.head_dim_qk;
          let k_col = elem_idx % params.head_dim_qk;
          let global_k_row = kv_tile + k_row;
          let global_k_row_offset = k_head_offset + global_k_row * params.stride_k1;
          let k_val: f32 = select(
              0.0,
              K[global_k_row_offset + k_col],
              global_k_row < params.seq_len_kv && k_col < params.head_dim_qk);
          k_shmem[elem_idx] = k_val;
      }

      workgroupBarrier();

      // accumulate q block * k block into registers
      var acc: subgroup_matrix_result<f32, 8, 8>;
      for (var head_dim_block = 0u; head_dim_block < params.head_dim_qk; head_dim_block += 8u) {
          // load q submatrix from shared memory
          var q_sg_mat: subgroup_matrix_left<f32, 8, 8> = subgroupMatrixLoad<subgroup_matrix_left<f32, 8, 8>>(
              &q_shmem,
              head_dim_block,
              false,
              params.head_dim_qk
          );

          // load k submatrix from shared memory
          var k_sg_mat: subgroup_matrix_right<f32, 8, 8> = subgroupMatrixLoad<subgroup_matrix_right<f32, 8, 8>>(
              &k_shmem,
              head_dim_block,
              true,
              params.head_dim_qk
          );

          acc = subgroupMatrixMultiplyAccumulate(q_sg_mat, k_sg_mat, acc);
      }

      // store acc to shared memory for softmax
      subgroupMatrixStore(&inter_shmem, 0, acc, false, KV_TILE);

      workgroupBarrier();

      // online softmax
      for (var q_tile_row = 0u; q_tile_row < Q_TILE; q_tile_row++) {
          // no need to process rows beyond seq_len_q
          let global_q_row = q_row_start + q_tile_row;
          if (global_q_row >= params.seq_len_q) {
              break;
          }

          // calculate running max
          let prev_max = row_max[q_tile_row];
          // The mask value for this Q row and K col
          let mask_val = select(0.0, f32(mask[mask_seq_offset + q_tile_row * params.seq_len_kv + kv_tile + sg_inv_id]), kv_tile + sg_inv_id < params.seq_len_kv && sg_inv_id < KV_TILE);
          let mask_term = slope * mask_val;
          let thread_tile_row_max = select(-3.4e38, inter_shmem[sg_inv_id + q_tile_row * KV_TILE] * params.scale + mask_term, sg_inv_id < KV_TILE);
          row_max[q_tile_row] = subgroupMax(max(prev_max, thread_tile_row_max));

          // calculate running exp sum
          let cur_exp = exp(prev_max - row_max[q_tile_row]);
          let cur_p = select(0.0, exp(thread_tile_row_max - row_max[q_tile_row]), kv_tile + sg_inv_id < params.seq_len_kv && sg_inv_id < KV_TILE);
          exp_sum[q_tile_row] = exp_sum[q_tile_row] * cur_exp + subgroupAdd(cur_p);

          // store back to shared memory (P matrix)
          if (sg_inv_id < KV_TILE) {
              inter_shmem[sg_inv_id + q_tile_row * KV_TILE] = cur_p;
          }

          for (var elem_idx = sg_inv_id; elem_idx < params.head_dim_v; elem_idx += SUBGROUP_SIZE) {
              let idx = q_tile_row * params.head_dim_v + elem_idx;
              let val = o_shmem[idx] * cur_exp;
              o_shmem[idx] = val;
          }
      }

      // load v tile into shared memory
      for (var elem_idx = local_id.x; elem_idx < KV_TILE * params.head_dim_v; elem_idx += WG_SIZE) {
          let v_row = elem_idx / params.head_dim_v;
          let v_col = elem_idx % params.head_dim_v;
          let global_v_row = kv_tile + v_row;
          let global_v_row_offset = v_head_offset + global_v_row * params.stride_v1;
          let v_val: f32 = select(
              0.0,
              f32(V[global_v_row_offset + v_col]),
              global_v_row < params.seq_len_kv && v_col < params.head_dim_v);
          v_shmem[elem_idx] = v_val;
      }

      workgroupBarrier();

      // we have P (8x8 tile, or Q_TILE x KV_TILE) in inter_shmem and V (8 x head_dim_v, or KV_TILE x head_dim_v) in v_shmem
      // we want to compute O += P * V
      // load P submatrix from shared memory
      var p_sg_mat: subgroup_matrix_left<f32, 8, 8> = subgroupMatrixLoad<subgroup_matrix_left<f32, 8, 8>>(
          &inter_shmem,
          0,
          false,
          KV_TILE
      );

      for (var head_dim_block = 0u; head_dim_block < params.head_dim_v; head_dim_block += 8u) {
          // load V submatrix from shared memory
          var v_sg_mat: subgroup_matrix_right<f32, 8, 8> = subgroupMatrixLoad<subgroup_matrix_right<f32, 8, 8>>(
              &v_shmem,
              head_dim_block,
              false, // or false? is this transposed?
              params.head_dim_v
          );

          // load O submatrix from shared memory
          var o_sg_mat: subgroup_matrix_result<f32, 8, 8> = subgroupMatrixLoad<subgroup_matrix_result<f32, 8, 8>>(
              &o_shmem,
              head_dim_block,
              false,
              params.head_dim_v
          );

          // O += P * V
          o_sg_mat = subgroupMatrixMultiplyAccumulate(p_sg_mat, v_sg_mat, o_sg_mat);

          // store O back to shared memory
          subgroupMatrixStore(&o_shmem, head_dim_block, o_sg_mat, false, params.head_dim_v);
      }

      workgroupBarrier();
    }

    // add sinks (applied once after processing all KV tiles)
    for (var q_tile_row = 0u; q_tile_row < Q_TILE; q_tile_row++) {
        // no need to process rows beyond seq_len_q
        let global_q_row = q_row_start + q_tile_row;
        if (global_q_row >= params.seq_len_q) {
            break;
        }

        let max_val = row_max[q_tile_row];
        // for non-sink threads, exp(-65504) effectively zeroes them out
        let sink_val = select(-3.4e38, sinks[params.offset_sinks + head_idx], sg_inv_id == 0);
        row_max[q_tile_row] = subgroupMax(max(max_val, sink_val));
        let max_exp = exp(max_val - row_max[q_tile_row]);
        let sink_exp = exp(sink_val - row_max[q_tile_row]);

        exp_sum[q_tile_row] = exp_sum[q_tile_row] * max_exp + subgroupAdd(sink_exp);
        for (var elem_idx = sg_inv_id; elem_idx < params.head_dim_v; elem_idx += SUBGROUP_SIZE) {
            let idx = q_tile_row * params.head_dim_v + elem_idx;
            let val = o_shmem[idx] * max_exp;
            o_shmem[idx] = val;
        }
    }

    // write output back to global memory
    for (var q_tile_row = 0u; q_tile_row < Q_TILE; q_tile_row++) {
        let global_q_row = q_row_start + q_tile_row;
        if (global_q_row >= params.seq_len_q) {
            break;
        }

        let scale = select(0.0, 1.0 / exp_sum[q_tile_row], exp_sum[q_tile_row] != 0);

        for (var elem_idx = sg_inv_id; elem_idx < params.head_dim_v; elem_idx += SUBGROUP_SIZE) {
            let o_val = o_shmem[q_tile_row * params.head_dim_v + elem_idx];
            let scaled = o_val * scale;
            dst[dst_global_offset + q_tile_row * dst2_stride + elem_idx] = scaled;
        }
    }
}
