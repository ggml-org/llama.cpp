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

#define HEAD_DIM_QK 64
#define HEAD_DIM_V 64


#define SG_MAT_M 8
#define SG_MAT_N 8
#define SG_MAT_K 8

#define Q_TILE SG_MAT_M
#define KV_TILE 16
#define WG_SIZE 64
#ifndef VEC_NE
#define VEC_NE 4u
#endif

#define KV_BLOCKS (KV_TILE / SG_MAT_N)

#define BLOCK_SIZE 32
#define BLOCKS_K ((HEAD_DIM_QK + BLOCK_SIZE - 1) / BLOCK_SIZE)
#define BLOCKS_V ((HEAD_DIM_V + BLOCK_SIZE - 1) / BLOCK_SIZE)
#if defined(KV_Q4_0)
#define NQ 16
#define F16_PER_BLOCK 9
#define WEIGHTS_PER_F16 4
#elif defined(KV_Q8_0)
#define NQ 8
#define F16_PER_BLOCK 17
#define WEIGHTS_PER_F16 2
#endif
#define F16_PER_THREAD (NQ / WEIGHTS_PER_F16)

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

#ifdef BLK
    blk_base: u32,
    blk_nblk0: u32,
    blk_nblk1: u32,
#endif

    tmp_data_base: u32,
    tmp_stats_base: u32,
    nwg: u32,
};

@group(0) @binding(0) var<storage, read_write> Q: array<f32>;
#if defined(KV_Q4_0) || defined(KV_Q8_0)
@group(0) @binding(1) var<storage, read_write> K: array<KV_TYPE>;
#else
@group(0) @binding(1) var<storage, read_write> K: array<vec4<KV_TYPE>>;
#endif
#if defined(KV_Q4_0) || defined(KV_Q8_0)
@group(0) @binding(2) var<storage, read_write> V: array<KV_TYPE>;
#else
@group(0) @binding(2) var<storage, read_write> V: array<vec4<KV_TYPE>>;
#endif
#if defined(MASK) && defined(SINKS)
@group(0) @binding(3) var<storage, read_write> mask: array<f16>;
@group(0) @binding(4) var<storage, read_write> sinks: array<f32>;
#ifdef BLK
#define BLK_BINDING 5
#define TMP_BINDING 6
#define DST_BINDING 7
#define PARAMS_BINDING 8
#else
#define TMP_BINDING 5
#define DST_BINDING 6
#define PARAMS_BINDING 7
#endif
#elif defined(MASK)
@group(0) @binding(3) var<storage, read_write> mask: array<f16>;
#ifdef BLK
#define BLK_BINDING 4
#define TMP_BINDING 5
#define DST_BINDING 6
#define PARAMS_BINDING 7
#else
#define TMP_BINDING 4
#define DST_BINDING 5
#define PARAMS_BINDING 6
#endif
#elif defined(SINKS)
@group(0) @binding(3) var<storage, read_write> sinks: array<f32>;
#define TMP_BINDING 4
#define DST_BINDING 5
#define PARAMS_BINDING 6
#else
#define TMP_BINDING 3
#define DST_BINDING 4
#define PARAMS_BINDING 5
#endif

#ifdef BLK
@group(0) @binding(BLK_BINDING) var<storage, read_write> blk: array<u32>;
#endif
@group(0) @binding(TMP_BINDING) var<storage, read_write> tmp: array<f32>;
@group(0) @binding(DST_BINDING) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(PARAMS_BINDING) var<uniform> params: Params;

// Just a very small float value.
const FLOAT_MIN: f32 = -1.0e9;

var<workgroup> q_shmem: array<f16, Q_TILE * HEAD_DIM_QK>;

#ifndef KV_DIRECT
const kv_shmem_size = KV_TILE * max(HEAD_DIM_QK, HEAD_DIM_V);
// we can reuse the same shmem for K and V since we only need one at a time
var<workgroup> kv_shmem: array<f16, kv_shmem_size>;
#endif

var<workgroup> o_shmem: array<f16, Q_TILE * HEAD_DIM_V>;

#ifdef MASK
// storage for mask values
var<workgroup> mask_shmem: array<f16, Q_TILE * KV_TILE>;
#endif

// note that we reuse the same storage for both since we only need one at a time
var<workgroup> inter_shmem: array<f16, Q_TILE * KV_TILE>;

// Storage for row max and exp sum during online softmax
var<workgroup> row_max_shmem: array<f32, Q_TILE>;
var<workgroup> exp_sum_shmem: array<f32, Q_TILE>;
var<workgroup> blk_state_wg: u32;

fn calc_softmax_term(kv_idx: u32, q_tile_row: u32, slope: f32, has_bias: bool, apply_mask: bool) -> f32 {
    var v = select(FLOAT_MIN,
                   f32(inter_shmem[kv_idx + q_tile_row * KV_TILE]) * params.scale,
                   kv_idx < KV_TILE);
#ifdef LOGIT_SOFTCAP
    v = params.logit_softcap * tanh(v);
#endif
#ifdef MASK
    if (apply_mask) {
        var mask_val = select(0.0,f32(mask_shmem[q_tile_row * KV_TILE + kv_idx]), kv_idx < KV_TILE);
        v += select(mask_val, slope * mask_val, has_bias);
    }
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

    let iwg = wg_id.x % params.nwg;
    let base_wg_id = wg_id.x / params.nwg;

    // batch index
    let batch_idx = base_wg_id / wg_per_batch;
    let q_batch_offset = params.offset_q + batch_idx * params.stride_q3;
    let k_batch_offset = params.offset_k + batch_idx * params.stride_k3;
    let v_batch_offset = params.offset_v + batch_idx * params.stride_v3;
    let wg_in_batch = base_wg_id % wg_per_batch;

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

    let head = f32(head_idx);
    let has_bias = params.max_bias > 0.0;
    let slope = select(1.0, select(pow(params.m1, 2.0 * (head - params.n_head_log2) + 1.0), pow(params.m0, head + 1.0), head < params.n_head_log2), has_bias);

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

    for (var kv_tile = iwg * KV_TILE; kv_tile < params.seq_len_kv; kv_tile += KV_TILE * params.nwg) {
#ifdef BLK
        let q_blk = q_row_start / Q_TILE;
        let kv_blk = kv_tile / KV_TILE;
        let blk_batch = select(0u, batch_idx, params.stride_mask3 > 0u);
        let blk_idx = params.blk_base + (blk_batch * params.blk_nblk1 + q_blk) * params.blk_nblk0 + kv_blk;
        let blk_state_local = blk[blk_idx];
#else
        let blk_state_local = 1u;
#endif
        if (local_id.x == 0u) {
            blk_state_wg = blk_state_local;
        }
        workgroupBarrier();
        let blk_state = blk_state_wg;
        let skip_tile = blk_state == 0u;
        for (var elem_idx = local_id.x; elem_idx < Q_TILE * KV_TILE; elem_idx += WG_SIZE) {
            inter_shmem[elem_idx] = f16(0.0);
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
              let d = K[base_idx];
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
              let d = K[base_idx];
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
      for (var elem_idx = local_id.x * 4u; elem_idx < KV_TILE * HEAD_DIM_QK; elem_idx += WG_SIZE * 4u) {
          let k_row = elem_idx / HEAD_DIM_QK;
          let k_col = elem_idx % HEAD_DIM_QK;
          let global_k_row = kv_tile + k_row;
          let global_k_row_offset = k_head_offset + global_k_row * params.stride_k1;
          let in_bounds = global_k_row < params.seq_len_kv && (k_col + 3u) < HEAD_DIM_QK;
          let vec_idx = (global_k_row_offset + k_col) >> 2u;
          let k4 = select(vec4<KV_TYPE>(0.0), K[vec_idx], in_bounds);
          kv_shmem[elem_idx + 0u] = f16(k4.x);
          kv_shmem[elem_idx + 1u] = f16(k4.y);
          kv_shmem[elem_idx + 2u] = f16(k4.z);
          kv_shmem[elem_idx + 3u] = f16(k4.w);
      }
#endif

      workgroupBarrier();

      // accumulate q block * k block into registers across the entire KV tile
      // TODO: this loop seems to be the current largest bottleneck
      // this bracket exists to scope the lifetime of variables, reducing register pressure
      if (!skip_tile) {
        let num_of_threads = subgroup_size / VEC_NE;
        let tx = sg_inv_id % num_of_threads;
        let ty = sg_inv_id / num_of_threads;
          for (var q_tile_row = subgroup_id; q_tile_row < Q_TILE; q_tile_row += num_subgroups) {
              let global_q_row = q_row_start + q_tile_row;
              if (global_q_row >= params.seq_len_q) {
                  continue;
              }
              let local_q_row_offset = q_tile_row * HEAD_DIM_QK;

              for (var kv_base : u32 = 0u; kv_base < KV_TILE; kv_base += VEC_NE) {
                  let kv_idx = kv_base + ty;
                  var partial_sum: f32 = 0.0;
                  let kv_valid = kv_idx < KV_TILE && (kv_tile + kv_idx) < params.seq_len_kv;
                  
                  if (kv_valid) {
                    for (var i = tx; i < (HEAD_DIM_QK / 4u); i += num_of_threads) {
                        let q_off = local_q_row_offset + i * 4u;

                        let qv = vec4<f32>(
                            f32(q_shmem[q_off + 0u]),
                            f32(q_shmem[q_off + 1u]),
                            f32(q_shmem[q_off + 2u]),
                            f32(q_shmem[q_off + 3u]));
#ifdef KV_DIRECT
                        let idx = k_head_offset + (kv_tile + kv_idx) * params.stride_k1 + (i * 4u);
                        let kv = vec4<f32>(K[idx >> 2u]);
#else
                        let idx = kv_idx * HEAD_DIM_QK + (i * 4u);
                        let kv = vec4<f32>(
                            f32(kv_shmem[idx + 0u]),
                            f32(kv_shmem[idx + 1u]),
                            f32(kv_shmem[idx + 2u]),
                            f32(kv_shmem[idx + 3u]));
#endif
                        partial_sum += dot(qv, kv);
                    }
                  }
                  var sum = partial_sum;
                  // Reduce over tx threads (NL) for this ty stripe.
                  var tx_delta = num_of_threads >> 1u;
                  loop {
                      if (tx_delta == 0u) {
                          break;
                      }
                      let sh = subgroupShuffleDown(sum, tx_delta);
                      if (tx < tx_delta) {
                          sum += sh;
                      }
                      tx_delta >>= 1u;
                  }

                  let sum_bcast = subgroupShuffle(sum, num_of_threads * ty);
                  if (tx == 0u && kv_valid) {
                      let dst_idx = q_tile_row * KV_TILE + kv_idx;
                      inter_shmem[dst_idx] = f16(sum_bcast);
                  }
              }
          }
      }


#ifdef MASK
      let apply_mask = !skip_tile && (blk_state != 2u);
      if (apply_mask) {
          // load mask tile into shared memory for this KV block
          for (var elem_idx = local_id.x; elem_idx < Q_TILE * KV_TILE; elem_idx += WG_SIZE) {
              let mask_row = elem_idx / KV_TILE;
              let mask_col = elem_idx % KV_TILE;
              let global_q_row = q_row_start + mask_row;
              let global_k_col = kv_tile + mask_col;
              let mask_in_bounds = global_q_row < params.seq_len_q && global_k_col < params.seq_len_kv;
              let mask_idx = mask_global_offset + mask_row * params.seq_len_kv + global_k_col;
              mask_shmem[elem_idx] = select(0.0, mask[mask_idx], mask_in_bounds);
          }
      }
#else
      let apply_mask = false;
#endif

      workgroupBarrier();

      // online softmax
      if (!skip_tile) {
          for (var q_tile_row = subgroup_id; q_tile_row < Q_TILE; q_tile_row += num_subgroups) {
              let global_q_row = q_row_start + q_tile_row;
              if (global_q_row >= params.seq_len_q) {
                  break;
              }

              var prev_max = row_max_shmem[q_tile_row];
              var final_max = prev_max;
              // pass 1: compute final max across the full KV tile in chunks
              for (var kv_offset = 0u; kv_offset < KV_TILE; kv_offset += subgroup_size) {
                  let kv_idx = kv_offset + sg_inv_id;
                  let kv_valid = kv_tile + kv_idx < params.seq_len_kv && kv_idx < KV_TILE;
                  let softmax_term = select(FLOAT_MIN,
                                            calc_softmax_term(kv_idx, q_tile_row, slope, has_bias, apply_mask),
                                            kv_valid);
                  final_max = subgroupMax(max(final_max, softmax_term));
              }

              var total_exp_term: f32 = 0.0;
              // pass 2: compute exp sum and write P using final_max
              for (var kv_offset = 0u; kv_offset < KV_TILE; kv_offset += subgroup_size) {
                  let kv_idx = kv_offset + sg_inv_id;
                  let softmax_term = calc_softmax_term(kv_idx, q_tile_row, slope, has_bias, apply_mask);
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
              let d = V[base_idx];
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
              let d = V[base_idx];
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
      for (var elem_idx = local_id.x * 4u; elem_idx < KV_TILE * HEAD_DIM_V; elem_idx += WG_SIZE * 4u) {
          let v_row = elem_idx / HEAD_DIM_V;
          let v_col = elem_idx % HEAD_DIM_V;
          let global_v_row = kv_tile + v_row;
          let global_v_row_offset = v_head_offset + global_v_row * params.stride_v1;
          let in_bounds = global_v_row < params.seq_len_kv && (v_col + 3u) < HEAD_DIM_V;
          let vec_idx = (global_v_row_offset + v_col) >> 2u;
          let v4 = select(vec4<KV_TYPE>(0.0), V[vec_idx], in_bounds);
          kv_shmem[elem_idx + 0u] = f16(v4.x);
          kv_shmem[elem_idx + 1u] = f16(v4.y);
          kv_shmem[elem_idx + 2u] = f16(v4.z);
          kv_shmem[elem_idx + 3u] = f16(v4.w);
      }
#endif

      workgroupBarrier();

      if (!skip_tile) {
          // we have P (Q_TILE x KV_TILE) in inter_shmem and V (KV_TILE x head_dim_v) in kv_shmem
          // we want to compute O += P * V across the full KV tile
          let ne_threads : u32 = VEC_NE;
          let nl_threads = max(1u, subgroup_size / ne_threads);
          let tx_pv = sg_inv_id % nl_threads;
          let ty_pv = sg_inv_id / nl_threads;
          for (var q_tile_row = subgroup_id;
               q_tile_row < Q_TILE;
               q_tile_row += num_subgroups) {
              for (var vec_col = tx_pv; vec_col < (HEAD_DIM_V / 4u); vec_col += nl_threads) {
                  var lo = vec4<f32>(0.0, 0.0, 0.0, 0.0);
                  for (var cc = 0u; cc < KV_TILE / ne_threads; cc += 1u) {
                      let kv_idx = cc * ne_threads + ty_pv;
                      let v_row = kv_tile + kv_idx;
                      if (v_row >= params.seq_len_kv) {
                          continue;
                      }

                      let p = f32(inter_shmem[kv_idx + q_tile_row * KV_TILE]);
#ifdef KV_DIRECT
                      let v_idx = v_head_offset + v_row * params.stride_v1 + vec_col * 4u;
                      let v4 = vec4<f32>(V[v_idx >> 2u]);
#else
                      let v_idx = kv_idx * HEAD_DIM_V + vec_col * 4u;
                      let v4 = vec4<f32>(
                          f32(kv_shmem[v_idx + 0u]),
                          f32(kv_shmem[v_idx + 1u]),
                          f32(kv_shmem[v_idx + 2u]),
                          f32(kv_shmem[v_idx + 3u]));
#endif
                      lo += p * v4;
                  }

                  var lo_x = lo.x;
                  var lo_y = lo.y;
                  var lo_z = lo.z;
                  var lo_w = lo.w;
                  // Reduce over ty threads (NE) for this tx thread.
                  var ty_delta = ne_threads >> 1u;
                  loop {
                      if (ty_delta == 0u) {
                          break;
                      }
                      let thread_delta = ty_delta * nl_threads;
                      let shx = subgroupShuffleDown(lo_x, thread_delta);
                      let shy = subgroupShuffleDown(lo_y, thread_delta);
                      let shz = subgroupShuffleDown(lo_z, thread_delta);
                      let shw = subgroupShuffleDown(lo_w, thread_delta);
                      if (ty_pv < ty_delta) {
                          lo_x += shx;
                          lo_y += shy;
                          lo_z += shz;
                          lo_w += shw;
                      }
                      ty_delta >>= 1u;
                  }

                  if (ty_pv == 0u) {
                      let elem_base = vec_col * 4u;
                      let o_base_idx = q_tile_row * HEAD_DIM_V + elem_base;
                      o_shmem[o_base_idx + 0u] = f16(f32(o_shmem[o_base_idx + 0u]) + lo_x);
                      o_shmem[o_base_idx + 1u] = f16(f32(o_shmem[o_base_idx + 1u]) + lo_y);
                      o_shmem[o_base_idx + 2u] = f16(f32(o_shmem[o_base_idx + 2u]) + lo_z);
                      o_shmem[o_base_idx + 3u] = f16(f32(o_shmem[o_base_idx + 3u]) + lo_w);
                  }
              }
          }
      }

        workgroupBarrier();
    }


#ifdef SINKS
    // Sinks are global terms and must be applied exactly once across split workgroups.
    if (iwg == 0u) {
        for (var q_tile_row = subgroup_id;
             q_tile_row < Q_TILE;
             q_tile_row += num_subgroups) {
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
                    row_max_shmem[q_tile_row] = new_max;
                    exp_sum_shmem[q_tile_row] = exp_sum_shmem[q_tile_row] * max_exp + sink_exp_sum;
                }

            for (var elem_idx = sg_inv_id; elem_idx < HEAD_DIM_V; elem_idx += subgroup_size) {
                let idx = q_tile_row * HEAD_DIM_V + elem_idx;
                o_shmem[idx] = f16(f32(o_shmem[idx]) * max_exp);
            }
        }
        workgroupBarrier();
    }
#endif
    let rows_per_batch = params.n_heads * params.seq_len_q;
    for (var q_tile_row = subgroup_id;
         q_tile_row < Q_TILE;
         q_tile_row += num_subgroups) {

        let global_q_row = q_row_start + q_tile_row;
        if (global_q_row >= params.seq_len_q) { break; }

        if (params.nwg == 1u) {
            let exp_sum = exp_sum_shmem[q_tile_row];
            let scale = select(0.0, 1.0 / exp_sum, exp_sum != 0.0);
            let row_base: u32 =
                params.offset_dst + batch_idx * dst3_stride + global_q_row * dst2_stride + head_idx * HEAD_DIM_V;

            for (var elem_base = sg_inv_id * 4u; elem_base < HEAD_DIM_V; elem_base += subgroup_size * 4u) {
                let i0 = q_tile_row * HEAD_DIM_V + (elem_base + 0u);
                let i1 = q_tile_row * HEAD_DIM_V + (elem_base + 1u);
                let i2 = q_tile_row * HEAD_DIM_V + (elem_base + 2u);
                let i3 = q_tile_row * HEAD_DIM_V + (elem_base + 3u);

                let v = vec4<f32>(
                    f32(o_shmem[i0]) * scale,
                    f32(o_shmem[i1]) * scale,
                    f32(o_shmem[i2]) * scale,
                    f32(o_shmem[i3]) * scale
                );

                let dst_vec_index: u32 = (row_base + elem_base) >> 2u;
                dst[dst_vec_index] = v;
            }
        } else {
            let rid = batch_idx * rows_per_batch + head_idx * params.seq_len_q + global_q_row;
            let tmp_row_data_base = params.tmp_data_base + rid * (HEAD_DIM_V * params.nwg) + iwg * HEAD_DIM_V;
            let tmp_row_stats_base = params.tmp_stats_base + rid * (2u * params.nwg) + 2u * iwg;

            for (var elem_base = sg_inv_id * 4u;
                elem_base < HEAD_DIM_V;
                elem_base += subgroup_size * 4u) {

                let i0 = q_tile_row * HEAD_DIM_V + (elem_base + 0u);
                let i1 = q_tile_row * HEAD_DIM_V + (elem_base + 1u);
                let i2 = q_tile_row * HEAD_DIM_V + (elem_base + 2u);
                let i3 = q_tile_row * HEAD_DIM_V + (elem_base + 3u);

                let tbase = tmp_row_data_base + elem_base;
                tmp[tbase + 0u] = f32(o_shmem[i0]);
                tmp[tbase + 1u] = f32(o_shmem[i1]);
                tmp[tbase + 2u] = f32(o_shmem[i2]);
                tmp[tbase + 3u] = f32(o_shmem[i3]);
            }

            if (sg_inv_id == 0u) {
                tmp[tmp_row_stats_base + 0u] = exp_sum_shmem[q_tile_row];
                tmp[tmp_row_stats_base + 1u] = row_max_shmem[q_tile_row];
            }
        }
    }
}
