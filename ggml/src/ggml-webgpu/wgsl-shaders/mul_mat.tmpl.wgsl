#define(VARIANTS)

[
  {
    "REPLS": {
      "SRC0_TYPE" : "f32",
      "SRC1_TYPE" : "f32",
      "BLOCK_SIZE" : 1
    },
    "DECLS" : "FLOAT"
  },
  {
    "REPLS": {
      "SRC0_TYPE" : "f16",
      "SRC1_TYPE" : "f16",
      "BLOCK_SIZE" : 1
    },
    "DECLS" : "FLOAT"
  },
  {
    "REPLS": {
      "SRC0_TYPE" : "f16",
      "SRC1_TYPE" : "f32",
      "BLOCK_SIZE" : 1
    },
    "DECLS" : "FLOAT"
  },
  {
    "REPLS": {
      "SRC0_TYPE" : "f32",
      "SRC1_TYPE" : "f16",
      "BLOCK_SIZE" : 1
    },
    "DECLS" : "FLOAT"
  },
  {
    "REPLS": {
      "SRC0_TYPE": "q4_0",
      "SRC1_TYPE": "f32",
      "BLOCK_SIZE": 32
    },
    "DECLS": "Q4_0"
  },
  {
    "REPLS": {
      "SRC0_TYPE": "q4_0",
      "SRC1_TYPE": "f16",
      "BLOCK_SIZE": 32
    },
    "DECLS": "Q4_0"
  }

]

#end(VARIANTS)

#define(DECLS)

#decl(FLOAT)
fn multiply_add(src0_idx_base: u32, src1_idx_base: u32, offset: u32) -> f32 {
    return f32(src0[src0_idx_base + offset]) * f32(src1[src1_idx_base + offset]);
}
#enddecl(FLOAT)

#decl(Q4_0)
struct q4_0 {
    d: f16,
    qs: array<f16, 8>
};

fn multiply_add(src0_idx_base: u32, src1_idx_base: u32, offset: u32) -> f32 {
    let block_q4_0: q4_0 = src0[src0_idx_base + offset];
    let d = f32(block_q4_0.d);
    var sum: f32 = 0.0;
    for (var j: u32 = 0; j < 4; j++) {
        let q_packed = bitcast<u32>(vec2(block_q4_0.qs[2 * j], block_q4_0.qs[2 * j + 1]));
        for (var k: u32 = 0; k < 4; k++) {
            let q_byte = (q_packed >> (k * 8)) & 0xFF;
            let q_hi = (f32((q_byte >> 4) & 0xF) - 8.0f) * d;
            let q_lo = (f32(q_byte & 0xF) - 8.0f) * d;
            let src1_offset = src1_idx_base + offset * 32 + j * 4 + k;
            sum += q_lo * f32(src1[src1_offset]);
            sum += q_hi * f32(src1[src1_offset + 16]);
        }
    }
    return sum;
}
#enddecl(Q4_0)
#end(DECLS)

#define(SHADER)

enable f16;

DECLS

struct MulMatParams {
    offset_src0: u32, // in elements
    offset_src1: u32, // in elements
    offset_dst: u32, // in elements
    m: u32,
    n: u32,
    k: u32,
    // all strides are in elements
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

@group(0) @binding(0) var<storage, read_write> src0: array<SRC0_TYPE>; // N rows, K columns
@group(0) @binding(1) var<storage, read_write> src1: array<SRC1_TYPE>; // M rows, K columns (transposed)
@group(0) @binding(2) var<storage, read_write> dst: array<f32>; // M rows, N columns

@group(0) @binding(3) var<uniform> params: MulMatParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let total = params.m * params.n * params.bs02 * params.broadcast2 * params.bs03 * params.broadcast3;
    if (global_id.x >= total) {
        return;
    }

    let dst2_stride = params.m * params.n;
    let dst3_stride = dst2_stride * params.bs02 * params.broadcast2;

    let dst3_idx = global_id.x / dst3_stride;
    let src03_idx = dst3_idx / params.broadcast3; // src0 may be broadcast along the third dimension
    let src13_idx = dst3_idx; // src1 is not broadcast
    let dst3_rem = global_id.x % dst3_stride;

    let dst2_idx = dst3_rem / dst2_stride;
    let src02_idx = dst2_idx / params.broadcast2; // src0 may also be broadcast along the second dimension
    let src12_idx = dst2_idx; // src1 is not broadcast

    let dst2_rem = dst3_rem % dst2_stride;

    let row = dst2_rem / params.n; // output row
    let col = dst2_rem % params.n; // output column

    let src0_idx_base = params.offset_src0 + src03_idx * params.stride_03 + src02_idx * params.stride_02 + col * params.stride_01;
    let src1_idx_base = params.offset_src1 + src13_idx * params.stride_13 + src12_idx * params.stride_12 + row * params.stride_11;

    var sum = 0.0;
    for (var i: u32 = 0u; i < params.k/BLOCK_SIZE; i = i + 1u) {
        sum += multiply_add(src0_idx_base, src1_idx_base, i);
    }
    dst[params.offset_dst + dst3_idx * dst3_stride + dst2_idx * dst2_stride + row * params.n + col] = sum;
}

#end(SHADER)
