#define(VARIANTS)
[
  {
    "SHADER_SUFFIX": "f32_f32_vec",
    "REPLS": {
      "SRC0_TYPE" : "vec4<f32>",
      "SRC1_TYPE" : "vec4<f32>",
      "DST_TYPE": "vec4<f32>",
      "VEC_SIZE" : 4,
    },
    "DECLS": ["VEC", "MUL_ACC_FLOAT"]
  },
  {
    "SHADER_SUFFIX": "f32_f32",
    "REPLS": {
      "SRC0_TYPE" : "f32",
      "SRC1_TYPE" : "f32",
      "DST_TYPE": "f32",
      "VEC_SIZE" : 1,
    },
    "DECLS": ["SCALAR", "MUL_ACC_FLOAT"]
  },
  {
    "SHADER_SUFFIX": "f16_f32_vec",
    "REPLS": {
      "SRC0_TYPE" : "vec4<f16>",
      "SRC1_TYPE" : "vec4<f32>",
      "DST_TYPE": "vec4<f32>",
      "VEC_SIZE" : 4,
    },
    "DECLS": ["VEC", "MUL_ACC_FLOAT"]
  },
  {
    "SHADER_SUFFIX": "f16_f32",
    "REPLS": {
      "SRC0_TYPE" : "f16",
      "SRC1_TYPE" : "f32",
      "DST_TYPE": "f32",
      "VEC_SIZE" : 1,
    },
    "DECLS": ["SCALAR", "MUL_ACC_FLOAT"]
  },
  {
    "SHADER_SUFFIX": "f16_f16_vec",
    "REPLS": {
      "SRC0_TYPE" : "vec4<f16>",
      "SRC1_TYPE" : "vec4<f16>",
      "DST_TYPE": "vec4<f32>",
      "VEC_SIZE" : 4,
    },
    "DECLS": ["VEC", "MUL_ACC_FLOAT"]
  },
  {
    "SHADER_SUFFIX": "f16_f16",
    "REPLS": {
      "SRC0_TYPE" : "f16",
      "SRC1_TYPE" : "f16",
      "DST_TYPE": "f32",
      "VEC_SIZE" : 1,
    },
    "DECLS": ["SCALAR", "MUL_ACC_FLOAT"]
  },
  {
    "SHADER_SUFFIX": "q4_0_f32",
    "REPLS": {
      "SRC0_TYPE" : "f16",
      "SRC1_TYPE" : "f32",
      "DST_TYPE": "f32",
      "VEC_SIZE" : 1,
    },
    "DECLS": ["BYTE_HELPERS", "SCALAR", "MUL_ACC_Q4_0"]
  }
]

#end(VARIANTS)

#define(DECLS)

#decl(VEC)
fn inner_dot(src0_val: {{SRC0_TYPE}}, src1_val: {{SRC1_TYPE}}) -> f32 {
    return f32(dot({{SRC1_TYPE}}(src0_val), src1_val));
}

fn store_val(group_base: u32) -> vec4<f32> {
    return vec4<f32>(partial_sums[group_base],
                     partial_sums[group_base + THREADS_PER_OUTPUT],
                     partial_sums[group_base + THREADS_PER_OUTPUT * 2],
                     partial_sums[group_base + THREADS_PER_OUTPUT * 3]);
}
#enddecl(VEC)

#decl(SCALAR)
fn inner_dot(src0_val: {{SRC0_TYPE}}, src1_val: {{SRC1_TYPE}}) -> f32 {
    return f32(src0_val) * f32(src1_val);
}

fn store_val(group_base: u32) -> f32 {
    return partial_sums[group_base];
}
#enddecl(SCALAR)

#decl(MUL_ACC_FLOAT)

fn mul_acc(tig:u32, tile_size: u32, idx_base: u32, k_outer: u32) -> f32 {
    var local_sum = 0.0;
    for (var i = tig * {{VEC_SIZE}}; i < tile_size; i += THREADS_PER_OUTPUT * {{VEC_SIZE}}) {
        let a = src0[(idx_base + k_outer + i) / {{VEC_SIZE}}];
        let b = shared_vector[i / {{VEC_SIZE}}];
        local_sum += inner_dot(a, b);
    }
    return local_sum;
}

#enddecl(MUL_ACC_FLOAT)

#decl(MUL_ACC_Q4_0)

const BLOCK_SIZE = 32;
const NQ = 16u; // number of weights per thread
const F16_PER_BLOCK = 9u; // 1 scale + 8x4 packed weights
const WEIGHTS_PER_F16 = 4u; // 4 weights per f16
const F16_PER_THREAD = NQ / WEIGHTS_PER_F16;

fn mul_acc(tig:u32, tile_size: u32, idx_base: u32, k_outer: u32) -> f32 {
    var local_sum = 0.0;
    for (var i = tig * NQ; i < tile_size; i += THREADS_PER_OUTPUT * NQ) {
        let blck_idx = i / BLOCK_SIZE;
        let block_offset = (i % BLOCK_SIZE) / WEIGHTS_PER_F16;
        let scale_idx = (idx_base + k_outer / BLOCK_SIZE + blck_idx) * F16_PER_BLOCK;
        // each f16 contains offsets [block_offset, block_offset + 1] and [block_offset + 16, block_offset + 17]
        let shmem_idx = blck_idx * BLOCK_SIZE + block_offset * 2u;
        let d = f32(src0[scale_idx]);
        for (var j = 0u; j < F16_PER_THREAD; j += 2) {
            let q_0 = src0[scale_idx + 1 + block_offset + j];
            let q_1 = src0[scale_idx + 1 + block_offset + j + 1];
            let q_packed = bitcast<u32>(vec2(q_0, q_1));
            for (var k: u32 = 0; k < 4; k++) {
                let q_byte = get_byte(q_packed, k);
                let q_hi = (f32((q_byte >> 4) & 0xF) - 8.0) * d;
                let q_lo = (f32(q_byte & 0xF) - 8.0) * d;
                local_sum += q_lo * shared_vector[shmem_idx + j * 2 + k];
                local_sum += q_hi * shared_vector[shmem_idx + j * 2 + k + 16];
            }
        }
    }
    return local_sum;
}

#enddecl(MUL_ACC_Q4_0)

#end(DECLS)

#define(SHADER)
enable f16;

DECLS

struct MulMatParams {
    offset_src0: u32,
    offset_src1: u32,
    offset_dst: u32,
    m: u32,
    n: u32,
    k: u32,
    stride_01: u32,
    stride_11: u32,
    stride_02: u32,
    stride_12: u32,
    stride_03: u32,
    stride_13: u32,
    bs02: u32,
    bs03: u32,
    broadcast2: u32,
    broadcast3: u32
};

@group(0) @binding(0) var<storage, read_write> src0: array<{{SRC0_TYPE}}>; // Matrix (M x K)
@group(0) @binding(1) var<storage, read_write> src1: array<{{SRC1_TYPE}}>; // Vector (K x 1, transposed)
@group(0) @binding(2) var<storage, read_write> dst: array<{{DST_TYPE}}>;  // Result vector (transposed)

@group(0) @binding(3) var<uniform> params: MulMatParams;

override WORKGROUP_SIZE: u32;
override TILE_K: u32;
override OUTPUTS_PER_WG: u32;
override THREADS_PER_OUTPUT = WORKGROUP_SIZE / OUTPUTS_PER_WG;

// Shared memory for collaborative loading and reduction
var<workgroup> shared_vector: array<{{SRC1_TYPE}}, TILE_K/{{VEC_SIZE}}>;  // Cache vector tile
var<workgroup> partial_sums: array<f32, WORKGROUP_SIZE>;   // For reduction

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>) {
    let thread_id = local_id.x;

    // Handle batch dimensions
    let total_batches = params.bs02 * params.broadcast2 * params.bs03 * params.broadcast3;
    let wg_linear = wg_id.y * num_wg.x + wg_id.x;
    let output_groups = (params.m + OUTPUTS_PER_WG - 1u) / OUTPUTS_PER_WG;
    let batch_idx = wg_linear / output_groups;
    if (batch_idx >= total_batches) {
        return;
    }

    // Which of the outputs does this thread belong to?
    let thread_group = thread_id / THREADS_PER_OUTPUT;
    let thread_in_group = thread_id % THREADS_PER_OUTPUT;

    // Each workgroup computes OUTPUTS_PER_WG consecutive outputs
    let output_row = (wg_linear % output_groups) * OUTPUTS_PER_WG + thread_group;

    let dst2_stride = params.m * params.n;
    let dst2_idx = batch_idx % (params.bs02 * params.broadcast2);
    let dst3_stride = dst2_stride * params.bs02 * params.broadcast2;
    let dst3_idx = batch_idx / (params.bs02 * params.broadcast2);
    let src03_idx = dst3_idx / params.broadcast3;
    let src13_idx = dst3_idx;
    let src02_idx = dst2_idx / params.broadcast2;
    let src12_idx = dst2_idx;

    let src0_idx_base = params.offset_src0 + src03_idx * params.stride_03 + src02_idx * params.stride_02 + output_row * params.stride_01;
    let src1_idx_base = params.offset_src1 + src13_idx * params.stride_13 + src12_idx * params.stride_12;
    let dst_idx = params.offset_dst + dst3_idx * dst3_stride + dst2_idx * dst2_stride + output_row;

    var local_sum = 0.0;

    // Each thread processes multiple K elements and accumulates
    for (var k_tile = 0u; k_tile < params.k; k_tile += TILE_K) {
        let tile_size = min(TILE_K, params.k - k_tile);

        // Cooperatively load vector tile into shared memory (all threads)
        for (var i = thread_id * {{VEC_SIZE}}; i < tile_size; i += WORKGROUP_SIZE * {{VEC_SIZE}}) {
            shared_vector[i / {{VEC_SIZE}}] = src1[(src1_idx_base + k_tile + i) / {{VEC_SIZE}}];
        }

        workgroupBarrier();

        if (output_row < params.m) {
            local_sum += mul_acc(thread_in_group, tile_size, src0_idx_base, k_tile);
        }

        workgroupBarrier();
    }

    // Store partial sums and reduce within each partition
    partial_sums[thread_id] = local_sum;
    workgroupBarrier();
    let group_base = thread_group * THREADS_PER_OUTPUT;
    let thread_base = group_base + thread_in_group;
    var offset = THREADS_PER_OUTPUT / 2;
    while (offset > 0) {
        if (thread_in_group < offset) {
            partial_sums[thread_base] += partial_sums[thread_base + offset];
        }
        offset = offset / 2;
        workgroupBarrier();
    }

    // Store back to global memory
    if (output_row < params.m && thread_group % {{VEC_SIZE}} == 0 && thread_in_group == 0) {
        dst[dst_idx / {{VEC_SIZE}}] = store_val(group_base);
    }
}
#end(SHADER)
