diagnostic(off, chromium.subgroup_matrix_uniformity);
diagnostic(off, subgroup_uniformity);
enable f16;
enable subgroups;
enable chromium_experimental_subgroup_matrix;

#ifdef KV_F32
#define KV_TYPE f32
#else
#define KV_TYPE f16
#endif

// Default values
#define HEAD_DIM_QK 64
#define HEAD_DIM_V 64

// The number of rows/columns/k in a subgroup matrix. MxK * KxN = MxN
// Note that the "K" here does not correspond to the K in attention's Q/K/V, it's just the common dimension.
#define SG_MAT_M 8
#define SG_MAT_N 8
#define SG_MAT_K 8

// Each workgroup processes one subgroup matrix of Q rows
#define Q_TILE SG_MAT_M
#define KV_TILE 16
#define WG_SIZE 64

// Number of subgroup-matrix-width blocks that span the KV tile. SG_MAT_N must divide KV_TILE.
#define KV_BLOCKS (KV_TILE / SG_MAT_N)

// Quantization constants/helpers
#define BLOCK_SIZE 32
#define BLOCKS_K ((HEAD_DIM_QK + BLOCK_SIZE - 1) / BLOCK_SIZE)
#define BLOCKS_V ((HEAD_DIM_V + BLOCK_SIZE - 1) / BLOCK_SIZE)
// number of quantized elements processed per thread
#if defined(KV_Q4_0)
#define NQ 16
// Q4_0 has 32 elements, 1 f16 for scale, 8 f16 for 4-bit weights
#define F16_PER_BLOCK 9
#define WEIGHTS_PER_F16 4
#elif defined(KV_Q8_0)
#define NQ 8
// Q8_0 has 32 elements, 1 f16 for scale, 16 f16 for 8-bit weights
#define F16_PER_BLOCK 17
#define WEIGHTS_PER_F16 2
#endif
#define F16_PER_THREAD (NQ / WEIGHTS_PER_F16)

// Ok not to put these in a define block, compiler will remove if unused
fn get_byte(value: u32, index: u32) -> u32 {
    return (value >> (index * 8)) & 0xFF;
}

fn get_byte_i32(value: u32, index: u32) -> i32 {
    return bitcast<i32>(((value >> (index * 8)) & 0xFF) << 24) >> 24;
}

struct Params {
    offset_q: u32,
    offset_k: u32,
    offset_v: u32,
    offset_mask: u32,
    offset_sinks: u32,
    offset_dst: u32,

    // shapes of Q/K/V
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
    stride_mask3: u32,

    // repeat factors for K/V, e.g., MHA vs. MQA vs. GQA
    q_per_kv: u32,

    // softmax params
    scale: f32,
    max_bias: f32,
    logit_softcap: f32,
    n_head_log2: f32,
    m0: f32,
    m1: f32,
};

@group(0) @binding(0) var<storage, read_write> Q: array<f32>;
@group(0) @binding(1) var<storage, read_write> K: array<KV_TYPE>;
@group(0) @binding(2) var<storage, read_write> V: array<KV_TYPE>;

#if defined(MASK) && defined(SINKS)
@group(0) @binding(3) var<storage, read_write> mask: array<f16>;
@group(0) @binding(4) var<storage, read_write> sinks: array<f32>;
#define DST_BINDING 5
#define PARAMS_BINDING 6
#elif defined(MASK)
@group(0) @binding(3) var<storage, read_write> mask: array<f16>;
#define DST_BINDING 4
#define PARAMS_BINDING 5
#elif defined(SINKS)
@group(0) @binding(3) var<storage, read_write> sinks: array<f32>;
#define DST_BINDING 4
#define PARAMS_BINDING 5
#else
#define DST_BINDING 3
#define PARAMS_BINDING 4
#endif

@group(0) @binding(DST_BINDING) var<storage, read_write> dst: array<f32>;
@group(0) @binding(PARAMS_BINDING) var<uniform> params: Params;

// Just a very small float value.
const FLOAT_MIN: f32 = -1.0e9;

// The number of Q rows processed per workgroup
var<workgroup> q_shmem: array<f16, Q_TILE * HEAD_DIM_QK>;

#ifndef KV_DIRECT
const kv_shmem_size = KV_TILE * max(HEAD_DIM_QK, HEAD_DIM_V);
// we can reuse the same shmem for K and V since we only need one at a time
var<workgroup> kv_shmem: array<f16, kv_shmem_size>;
#endif

var<workgroup> o_shmem: array<f16, Q_TILE * HEAD_DIM_V>; // output shmem

#ifdef MASK
// storage for mask values
var<workgroup> mask_shmem: array<f16, Q_TILE * KV_TILE>;
#endif

// storage for output of Q*K^T scores for online softmax (S matrix from paper)
// also storage for diagonal matrix during online softmax (P matrix from paper)
// note that we reuse the same storage for both since we only need one at a time
var<workgroup> inter_shmem: array<f16, Q_TILE * KV_TILE>;

// Storage for row max and exp sum during online softmax
var<workgroup> row_max_shmem: array<f32, Q_TILE>;
var<workgroup> exp_sum_shmem: array<f32, Q_TILE>;

fn calc_softmax_term(kv_idx: u32, q_tile_row: u32, slope: f32) -> f32 {
    var v = select(FLOAT_MIN,
                   f32(inter_shmem[kv_idx + q_tile_row * KV_TILE]) * params.scale,
                   kv_idx < KV_TILE);
#ifdef LOGIT_SOFTCAP
    v = params.logit_softcap * tanh(v);
#endif
#ifdef MASK
    let mask_val = select(0.0, f32(mask_shmem[q_tile_row * KV_TILE + kv_idx]), kv_idx < KV_TILE);
    let mask_term = slope * mask_val;
    v += mask_term;
#endif
    return v;
}


@compute @workgroup_size(WG_SIZE)
fn main(@builtin(workgroup_id) wg_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(subgroup_id) subgroup_id: u32,
        @builtin(subgroup_size) subgroup_size: u32,
        @builtin(num_subgroups) num_subgroups: u32,
        @builtin(subgroup_invocation_id) sg_inv_id: u32) {

    // initialize row max for online softmax
    for (var i = local_id.x; i < Q_TILE; i += WG_SIZE) {
        row_max_shmem[i] = FLOAT_MIN;
        exp_sum_shmem[i] = 0.0;
    }

    for (var i = local_id.x; i < Q_TILE * HEAD_DIM_V; i += WG_SIZE) {
        o_shmem[i] = 0.0;
    }

    // workgroups per head/batch
    let wg_per_head = (params.seq_len_q + Q_TILE - 1u) / Q_TILE;
    let wg_per_batch = wg_per_head * params.n_heads;

    let dst2_stride = HEAD_DIM_V * params.n_heads;
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
    let k_head_idx = head_idx / params.q_per_kv;
    let v_head_idx = k_head_idx;
    let k_head_offset = k_batch_offset + k_head_idx * params.stride_k2;
    let v_head_offset = v_batch_offset + v_head_idx * params.stride_v2;

    // starting Q row for this workgroup
    let wg_in_head = wg_in_batch % wg_per_head;
    let q_row_start = wg_in_head * Q_TILE;

#ifdef MASK
    // mask offset
    let mask_global_offset = params.offset_mask + batch_idx * params.stride_mask3 + q_row_start * params.seq_len_kv;
#endif

    // note that the output is permuted, the layout is [head_dim_v, n_heads, seq_len_q, batch_size]
    let dst_global_offset = dst_batch_offset + q_row_start * dst2_stride + head_idx * HEAD_DIM_V;

    let head = f32(head_idx);
    let slope = select(1.0, select(pow(params.m1, 2.0 * (head - params.n_head_log2) + 1.0), pow(params.m0, head + 1.0), head < params.n_head_log2), params.max_bias > 0);

    // load q tile into shared memory
    for (var elem_idx = local_id.x; elem_idx < Q_TILE * HEAD_DIM_QK; elem_idx += WG_SIZE) {
        let q_row = elem_idx / HEAD_DIM_QK;
        let q_col = elem_idx % HEAD_DIM_QK;
        let head_q_row = q_row_start + q_row;
        let global_q_row_offset = q_head_offset + head_q_row * params.stride_q1;
        q_shmem[elem_idx] = f16(select(
            0.0,
            Q[global_q_row_offset + q_col],
            head_q_row < params.seq_len_q && q_col < HEAD_DIM_QK));
    }

    for (var kv_tile = 0u; kv_tile < params.seq_len_kv; kv_tile += KV_TILE) {
      // clear inter_shmem to ensure zero-initialized accumulators
      for (var elem_idx = local_id.x; elem_idx < Q_TILE * KV_TILE; elem_idx += WG_SIZE) {
          inter_shmem[elem_idx] = 0.0;
      }

      // load k tile into shared memory
#if defined(KV_Q4_0)
      for (var elem_idx = local_id.x * NQ; elem_idx < KV_TILE * HEAD_DIM_QK; elem_idx += WG_SIZE * NQ) {
          let blck_idx = elem_idx / BLOCK_SIZE;
          let block_offset = (elem_idx % BLOCK_SIZE) / WEIGHTS_PER_F16;
          let k_row = blck_idx / BLOCKS_K;
          let global_k_row = kv_tile + k_row;
          let block_k = blck_idx % BLOCKS_K;
          let row_offset = k_row * HEAD_DIM_QK;

          if (global_k_row < params.seq_len_kv) {
              let global_block_idx = k_head_offset + global_k_row * params.stride_k1 + block_k;
              let base_idx = global_block_idx * F16_PER_BLOCK;
              let d = K[base_idx]; // scale
              for (var j = 0u; j < F16_PER_THREAD; j += 2) {
                  let q_0 = K[base_idx + 1u + block_offset + j];
                  let q_1 = K[base_idx + 1u + block_offset + j + 1];
                  let q_packed = bitcast<u32>(vec2(q_0, q_1));
                  for (var k = 0u; k < 4u; k++) {
                      let q_byte = get_byte(q_packed, k);
                      let q_hi = (f16((q_byte >> 4) & 0xF) - 8.0) * d;
                      let q_lo = (f16(q_byte & 0xF) - 8.0) * d;
                      let idx = block_k * BLOCK_SIZE + block_offset * 2u + j * 2u + k;
                      kv_shmem[row_offset + idx] = q_lo;
                      kv_shmem[row_offset + idx + 16u] = q_hi;
                  }
              }
          }
      }
#elif defined(KV_Q8_0)
      for (var elem_idx = local_id.x * NQ; elem_idx < KV_TILE * HEAD_DIM_QK; elem_idx += WG_SIZE * NQ) {
          let blck_idx = elem_idx / BLOCK_SIZE;
          let block_offset = (elem_idx % BLOCK_SIZE) / WEIGHTS_PER_F16;
          let k_row = blck_idx / BLOCKS_K;
          let global_k_row = kv_tile + k_row;
          let block_k = blck_idx % BLOCKS_K;
          let row_offset = k_row * HEAD_DIM_QK;

          if (global_k_row < params.seq_len_kv) {
              let global_block_idx = k_head_offset + global_k_row * params.stride_k1 + block_k;
              let base_idx = global_block_idx * F16_PER_BLOCK;
              let d = K[base_idx]; // scale
              for (var j = 0u; j < F16_PER_THREAD; j += 2) {
                  let q_0 = K[base_idx + 1u + block_offset + j];
                  let q_1 = K[base_idx + 1u + block_offset + j + 1];
                  let q_packed = bitcast<u32>(vec2(q_0, q_1));
                  for (var k = 0u; k < 4u; k++) {
                      let q_byte = get_byte_i32(q_packed, k);
                      let q_val = f16(q_byte) * d;
                      let idx = block_k * BLOCK_SIZE + block_offset * 2u + j * 2u + k;
                      kv_shmem[row_offset + idx] = q_val;
                  }
              }
          }
      }
#elif defined(KV_DIRECT)
      // Direct global loads for KV
#else
      for (var elem_idx = local_id.x; elem_idx < KV_TILE * HEAD_DIM_QK; elem_idx += WG_SIZE) {
          let k_row = elem_idx / HEAD_DIM_QK;
          let k_col = elem_idx % HEAD_DIM_QK;
          let global_k_row = kv_tile + k_row;
          let global_k_row_offset = k_head_offset + global_k_row * params.stride_k1;
          kv_shmem[elem_idx] = f16(select(
              0.0,
              K[global_k_row_offset + k_col],
              global_k_row < params.seq_len_kv && k_col < HEAD_DIM_QK));
      }
#endif

      workgroupBarrier();

      // accumulate q block * k block into registers across the entire KV tile
      // TODO: this loop seems to be the current largest bottleneck
      for (var kv_block = subgroup_id; kv_block < KV_BLOCKS; kv_block += num_subgroups) {
          let inter_offset = kv_block * SG_MAT_N;
          var acc: subgroup_matrix_result<f16, SG_MAT_M, SG_MAT_N> = subgroupMatrixLoad<
              subgroup_matrix_result<f16, SG_MAT_M, SG_MAT_N>>(&inter_shmem, inter_offset, false, KV_TILE);
#ifdef KV_DIRECT
          let k_block_row = kv_tile + kv_block * SG_MAT_N;
          let k_global_offset = k_head_offset + k_block_row * params.stride_k1;
#else
          let k_block_offset = kv_block * SG_MAT_N * HEAD_DIM_QK;
#endif
          for (var head_dim_block = 0u; head_dim_block < HEAD_DIM_QK; head_dim_block += SG_MAT_K) {
              // load q submatrix from shared memory
              var q_sg_mat: subgroup_matrix_left<f16, SG_MAT_M, SG_MAT_K> = subgroupMatrixLoad<subgroup_matrix_left<f16, SG_MAT_M, SG_MAT_K>>(
                  &q_shmem,
                  head_dim_block,
                  false,
                  HEAD_DIM_QK
              );

              // load k submatrix from device or shared memory
#ifdef KV_DIRECT
              var k_sg_mat: subgroup_matrix_right<f16, SG_MAT_K, SG_MAT_N> = subgroupMatrixLoad<subgroup_matrix_right<f16, SG_MAT_K, SG_MAT_N>>(
                  &K,
                  k_global_offset + head_dim_block,
                  true,
                  params.stride_k1
              );
#else
              var k_sg_mat: subgroup_matrix_right<f16, SG_MAT_K, SG_MAT_N> = subgroupMatrixLoad<subgroup_matrix_right<f16, SG_MAT_K, SG_MAT_N>>(
                  &kv_shmem,
                  k_block_offset + head_dim_block,
                  true,
                  HEAD_DIM_QK
              );
#endif
              acc = subgroupMatrixMultiplyAccumulate(q_sg_mat, k_sg_mat, acc);
          }

          // store acc to shared memory for softmax (S matrix from paper)
          subgroupMatrixStore(&inter_shmem, inter_offset, acc, false, KV_TILE);
      }

#ifdef MASK
      // load mask tile into shared memory for this KV block
      // TODO: optimize and skip if mask is -INF for the entire tile
      for (var elem_idx = local_id.x; elem_idx < Q_TILE * KV_TILE; elem_idx += WG_SIZE) {
          let mask_row = elem_idx / KV_TILE;
          let mask_col = elem_idx % KV_TILE;
          let global_q_row = q_row_start + mask_row;
          let global_k_col = kv_tile + mask_col;
          let mask_in_bounds = global_q_row < params.seq_len_q && global_k_col < params.seq_len_kv;
          let mask_idx = mask_global_offset + mask_row * params.seq_len_kv + global_k_col;
          mask_shmem[elem_idx] = select(0.0, mask[mask_idx], mask_in_bounds);
      }
#endif

      workgroupBarrier();

      // online softmax
      for (var q_tile_row = subgroup_id; q_tile_row < Q_TILE; q_tile_row += num_subgroups) {
          let global_q_row = q_row_start + q_tile_row;
          if (global_q_row >= params.seq_len_q) {
              break;
          }

          // initialize running max for this row
          var prev_max = row_max_shmem[q_tile_row];
          var final_max = prev_max;
          // pass 1: compute final max across the full KV tile in chunks
          for (var kv_offset = 0u; kv_offset < KV_TILE; kv_offset += subgroup_size) {
              let kv_idx = kv_offset + sg_inv_id;
              let softmax_term = calc_softmax_term(kv_idx, q_tile_row, slope);
              final_max = subgroupMax(max(final_max, softmax_term));
          }

          var total_exp_term: f32 = 0.0;
          // pass 2: compute exp sum and write P using final_max
          for (var kv_offset = 0u; kv_offset < KV_TILE; kv_offset += subgroup_size) {
              let kv_idx = kv_offset + sg_inv_id;
              let softmax_term = calc_softmax_term(kv_idx, q_tile_row, slope);
              let cur_p = select(0.0,
                                 exp(softmax_term - final_max),
                                 kv_tile + kv_idx < params.seq_len_kv && kv_idx < KV_TILE);
              total_exp_term += subgroupAdd(cur_p);
              if (kv_idx < KV_TILE) {
                  inter_shmem[kv_idx + q_tile_row * KV_TILE] = f16(cur_p);
              }
          }

          let cur_exp = exp(prev_max - final_max);

          if (sg_inv_id == 0) {
              row_max_shmem[q_tile_row] = final_max;
              exp_sum_shmem[q_tile_row] = exp_sum_shmem[q_tile_row] * cur_exp + total_exp_term;
          }

          for (var elem_idx = sg_inv_id; elem_idx < HEAD_DIM_V; elem_idx += subgroup_size) {
              let idx = q_tile_row * HEAD_DIM_V + elem_idx;
              o_shmem[idx] = f16(f32(o_shmem[idx]) * cur_exp);
          }
      }

      // load v tile into shared memory
#if defined(KV_Q4_0)
      for (var elem_idx = local_id.x * NQ; elem_idx < KV_TILE * HEAD_DIM_V; elem_idx += WG_SIZE * NQ) {
          let blck_idx = elem_idx / BLOCK_SIZE;
          let block_offset = (elem_idx % BLOCK_SIZE) / WEIGHTS_PER_F16;
          let v_row = blck_idx / BLOCKS_V;
          let global_v_row = kv_tile + v_row;
          let block_k = blck_idx % BLOCKS_V;
          let row_offset = v_row * HEAD_DIM_V;

          if (global_v_row < params.seq_len_kv) {
              let global_block_idx = v_head_offset + global_v_row * params.stride_v1 + block_k;
              let base_idx = global_block_idx * F16_PER_BLOCK;
              let d = V[base_idx]; // scale
              for (var j = 0u; j < F16_PER_THREAD; j += 2) {
                  let q_0 = V[base_idx + 1u + block_offset + j];
                  let q_1 = V[base_idx + 1u + block_offset + j + 1];
                  let q_packed = bitcast<u32>(vec2(q_0, q_1));
                  for (var k = 0u; k < 4u; k++) {
                      let q_byte = get_byte(q_packed, k);
                      let q_hi = (f16((q_byte >> 4) & 0xF) - 8.0) * d;
                      let q_lo = (f16(q_byte & 0xF) - 8.0) * d;
                      let idx = block_k * BLOCK_SIZE + block_offset * 2u + j * 2u + k;
                      kv_shmem[row_offset + idx] = q_lo;
                      kv_shmem[row_offset + idx + 16u] = q_hi;
                  }
              }
          }
      }
#elif defined(KV_Q8_0)
      for (var elem_idx = local_id.x * NQ; elem_idx < KV_TILE * HEAD_DIM_V; elem_idx += WG_SIZE * NQ) {
          let blck_idx = elem_idx / BLOCK_SIZE;
          let block_offset = (elem_idx % BLOCK_SIZE) / WEIGHTS_PER_F16;
          let v_row = blck_idx / BLOCKS_V;
          let global_v_row = kv_tile + v_row;
          let block_k = blck_idx % BLOCKS_V;
          let row_offset = v_row * HEAD_DIM_V;

          if (global_v_row < params.seq_len_kv) {
              let global_block_idx = v_head_offset + global_v_row * params.stride_v1 + block_k;
              let base_idx = global_block_idx * F16_PER_BLOCK;
              let d = V[base_idx]; // scale
              for (var j = 0u; j < F16_PER_THREAD; j += 2) {
                  let q_0 = V[base_idx + 1u + block_offset + j];
                  let q_1 = V[base_idx + 1u + block_offset + j + 1];
                  let q_packed = bitcast<u32>(vec2(q_0, q_1));
                  for (var k = 0u; k < 4u; k++) {
                      let q_byte = get_byte_i32(q_packed, k);
                      let q_val = f16(q_byte) * d;
                      let idx = block_k * BLOCK_SIZE + block_offset * 2u + j * 2u + k;
                      kv_shmem[row_offset + idx] = q_val;
                  }
              }
          }
      }
#elif defined(KV_DIRECT)
      // Direct global loads for KV
#else
      for (var elem_idx = local_id.x; elem_idx < KV_TILE * HEAD_DIM_V; elem_idx += WG_SIZE) {
          let v_row = elem_idx / HEAD_DIM_V;
          let v_col = elem_idx % HEAD_DIM_V;
          let global_v_row = kv_tile + v_row;
          let global_v_row_offset = v_head_offset + global_v_row * params.stride_v1;
          kv_shmem[elem_idx] = f16(select(
              0.0,
              V[global_v_row_offset + v_col],
              global_v_row < params.seq_len_kv && v_col < HEAD_DIM_V));
      }
#endif

      workgroupBarrier();

      // we have P (Q_TILE x KV_TILE) in inter_shmem and V (KV_TILE x head_dim_v) in kv_shmem
      // we want to compute O += P * V across the full KV tile
      for (var head_dim_block = subgroup_id * SG_MAT_N;
           head_dim_block < HEAD_DIM_V;
           head_dim_block += num_subgroups * SG_MAT_N) {
              // load O submatrix from shared memory
              var o_sg_mat: subgroup_matrix_result<f16, SG_MAT_M, SG_MAT_N> = subgroupMatrixLoad<subgroup_matrix_result<f16, SG_MAT_M, SG_MAT_N>>(
                  &o_shmem,
                  head_dim_block,
                  false,
                  HEAD_DIM_V
              );

              for (var kv_block = 0u; kv_block < KV_BLOCKS; kv_block++) {
                  let p_offset = kv_block * SG_MAT_N;
                  var p_sg_mat: subgroup_matrix_left<f16, SG_MAT_M, SG_MAT_K> = subgroupMatrixLoad<subgroup_matrix_left<f16, SG_MAT_M, SG_MAT_K>>(
                      &inter_shmem,
                      p_offset,
                      false,
                      KV_TILE
                  );

                  // load V submatrix from global or shared memory
#ifdef KV_DIRECT
                  let v_block_row = kv_tile + kv_block * SG_MAT_N;
                  let v_global_offset = v_head_offset + v_block_row * params.stride_v1 + head_dim_block;
                  var v_sg_mat: subgroup_matrix_right<f16, SG_MAT_K, SG_MAT_N> = subgroupMatrixLoad<subgroup_matrix_right<f16, SG_MAT_K, SG_MAT_N>>(
                      &V,
                      v_global_offset,
                      false,
                      params.stride_v1
                  );
#else
                  let v_block_offset = kv_block * SG_MAT_N * HEAD_DIM_V;
                  var v_sg_mat: subgroup_matrix_right<f16, SG_MAT_K, SG_MAT_N> = subgroupMatrixLoad<subgroup_matrix_right<f16, SG_MAT_K, SG_MAT_N>>(
                      &kv_shmem,
                      v_block_offset + head_dim_block,
                      false,
                      HEAD_DIM_V
                  );
#endif
                  // O += P * V
                  o_sg_mat = subgroupMatrixMultiplyAccumulate(p_sg_mat, v_sg_mat, o_sg_mat);
              }

              // store O back to shared memory
              subgroupMatrixStore(&o_shmem, head_dim_block, o_sg_mat, false, HEAD_DIM_V);
      }

      workgroupBarrier();
    }

#ifdef SINKS
    // add sinks (applied once after processing all KV tiles)
    for (var q_tile_row = subgroup_id;
         q_tile_row < Q_TILE;
         q_tile_row += num_subgroups) {
            // no need to process rows beyond seq_len_q
            let global_q_row = q_row_start + q_tile_row;
            if (global_q_row >= params.seq_len_q) {
                break;
            }

            var prev_max = row_max_shmem[q_tile_row];

            // for non-sink threads, exp(FLOAT_MIN) effectively zeroes out their contribution to the sum
            let sink_val = select(FLOAT_MIN, sinks[params.offset_sinks + head_idx], sg_inv_id == 0);
            let new_max = subgroupMax(max(prev_max, sink_val));
            let max_exp = exp(prev_max - new_max);
            let sink_exp = exp(sink_val - new_max);

            let sink_exp_sum = subgroupAdd(sink_exp);

            if (sg_inv_id == 0) {
                exp_sum_shmem[q_tile_row] = exp_sum_shmem[q_tile_row] * max_exp + sink_exp_sum;
            }

            for (var elem_idx = sg_inv_id; elem_idx < HEAD_DIM_V; elem_idx += subgroup_size) {
                let idx = q_tile_row * HEAD_DIM_V + elem_idx;
                let val = f32(o_shmem[idx]) * max_exp;
                o_shmem[idx] = f16(val);
            }
    }

    workgroupBarrier();
#endif

    // write output back to global memory
    for (var q_tile_row = subgroup_id;
         q_tile_row < Q_TILE;
         q_tile_row += num_subgroups) {
            let global_q_row = q_row_start + q_tile_row;
            if (global_q_row >= params.seq_len_q) {
                break;
            }

            let exp_sum = exp_sum_shmem[q_tile_row];
            let scale = select(0.0, 1.0 / exp_sum, exp_sum != 0);

            for (var elem_idx = sg_inv_id; elem_idx < HEAD_DIM_V; elem_idx += subgroup_size) {
                let o_val = o_shmem[q_tile_row * HEAD_DIM_V + elem_idx];
                let scaled = f32(o_val) * scale;
                dst[dst_global_offset + q_tile_row * dst2_stride + elem_idx] = scaled;
            }
    }
}
