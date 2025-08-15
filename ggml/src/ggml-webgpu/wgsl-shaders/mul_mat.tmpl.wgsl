#define(VARIANTS)

[
  {
    "REPLS": {
      "SRC0_TYPE" : "f32",
      "SRC1_TYPE" : "f32",
      "BLOCK_SIZE" : 1
    },
    "DECLS" : ["FLOAT"]
  },
  {
    "REPLS": {
      "SRC0_TYPE" : "f16",
      "SRC1_TYPE" : "f16",
      "BLOCK_SIZE" : 1
    },
    "DECLS" : ["FLOAT"]
  },
  {
    "REPLS": {
      "SRC0_TYPE" : "f16",
      "SRC1_TYPE" : "f32",
      "BLOCK_SIZE" : 1
    },
    "DECLS" : ["FLOAT"]
  },
  {
    "REPLS": {
      "SRC0_TYPE": "q4_0",
      "SRC1_TYPE": "f32",
      "BLOCK_SIZE": 32
    },
    "DECLS": ["BYTE_HELPERS", "Q4_0"]
  },
  {
    "REPLS": {
      "SRC0_TYPE": "q4_1",
      "SRC1_TYPE": "f32",
      "BLOCK_SIZE": 32
    },
    "DECLS": ["BYTE_HELPERS", "Q4_1"]
  },
  {
    "REPLS": {
      "SRC0_TYPE": "q5_0",
      "SRC1_TYPE": "f32",
      "BLOCK_SIZE": 32
    },
    "DECLS": ["BYTE_HELPERS", "Q5_0"]
  },
  {
    "REPLS": {
      "SRC0_TYPE": "q5_1",
      "SRC1_TYPE": "f32",
      "BLOCK_SIZE": 32
    },
    "DECLS": ["BYTE_HELPERS", "Q5_1"]
  },
  {
    "REPLS": {
      "SRC0_TYPE": "q8_0",
      "SRC1_TYPE": "f32",
      "BLOCK_SIZE": 32
    },
    "DECLS": ["BYTE_HELPERS", "Q8_0"]
  },
  {
    "REPLS": {
      "SRC0_TYPE": "q2_k",
      "SRC1_TYPE": "f32",
      "BLOCK_SIZE": 256
    },
    "DECLS": ["BYTE_HELPERS", "Q2_K"]
  },
  {
    "REPLS": {
      "SRC0_TYPE": "q3_k",
      "SRC1_TYPE": "f32",
      "BLOCK_SIZE": 256
    },
    "DECLS": ["BYTE_HELPERS", "Q3_K"]
  },
  {
    "REPLS": {
      "SRC0_TYPE": "q4_k",
      "SRC1_TYPE": "f32",
      "BLOCK_SIZE": 256
    },
    "DECLS": ["Q45_K_SCALE_MIN", "BYTE_HELPERS", "Q4_K"]
  },
  {
    "REPLS": {
      "SRC0_TYPE": "q5_k",
      "SRC1_TYPE": "f32",
      "BLOCK_SIZE": 256
    },
    "DECLS": ["Q45_K_SCALE_MIN", "BYTE_HELPERS", "Q5_K"]
  },
  {
    "REPLS": {
      "SRC0_TYPE": "q6_k",
      "SRC1_TYPE": "f32",
      "BLOCK_SIZE": 256
    },
    "DECLS": ["BYTE_HELPERS", "Q6_K"]
  },
  {
    "REPLS": {
      "SRC0_TYPE": "iq2_xxs",
      "SRC1_TYPE": "f32",
      "BLOCK_SIZE": 256
    },
    "DECLS": ["BYTE_HELPERS", "IQ2_TABLES", "IQ2_XXS"]
  }
]

#end(VARIANTS)

#define(DECLS)

#decl(BYTE_HELPERS)

fn get_byte(value: u32, index: u32) -> u32 {
    return (value >> (index * 8)) & 0xFF;
}

fn get_byte_i32(value: u32, index: u32) -> i32 {
    return bitcast<i32>(((value >> (index * 8)) & 0xFF) << 24) >> 24;
}

#enddecl(BYTE_HELPERS)

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
    let block_q4_0 = src0[src0_idx_base + offset];
    let d = f32(block_q4_0.d);
    var sum: f32 = 0.0;
    for (var j: u32 = 0; j < 4; j++) {
        let q_packed = bitcast<u32>(vec2(block_q4_0.qs[2 * j], block_q4_0.qs[2 * j + 1]));
        for (var k: u32 = 0; k < 4; k++) {
            let q_byte = get_byte(q_packed, k);
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

#decl(Q4_1)
struct q4_1 {
    d: f16,
    m: f16,
    qs: array<u32, 4>
};

fn multiply_add(src0_idx_base: u32, src1_idx_base: u32, offset: u32) -> f32 {
    let block_q4_1 = src0[src0_idx_base + offset];
    let d = f32(block_q4_1.d);
    let m = f32(block_q4_1.m);
    var sum: f32 = 0.0;
    for (var j: u32 = 0; j < 4; j++) {
        let q_packed = block_q4_1.qs[j];
        for (var k: u32 = 0; k < 4; k++) {
            let q_byte = get_byte(q_packed, k);
            let q_hi = f32((q_byte >> 4) & 0xF) * d + m;
            let q_lo = f32(q_byte & 0xF) * d + m;
            let src1_offset = src1_idx_base + offset * 32 + j * 4 + k;
            sum += q_lo * f32(src1[src1_offset]);
            sum += q_hi * f32(src1[src1_offset + 16]);
        }
    }
    return sum;
}
#enddecl(Q4_1)

#decl(Q5_0)
struct q5_0 {
    d: f16,
    qh: array<f16, 2>,
    qs: array<f16, 8>
};

fn multiply_add(src0_idx_base: u32, src1_idx_base: u32, offset: u32) -> f32 {
    let block_q5_0 = src0[src0_idx_base + offset];
    let d = f32(block_q5_0.d);
    var sum: f32 = 0.0;
    let qh_packed = bitcast<u32>(vec2(block_q5_0.qh[0], block_q5_0.qh[1]));
    for (var j: u32 = 0; j < 4; j++) {
        let q_packed = bitcast<u32>(vec2(block_q5_0.qs[2 * j], block_q5_0.qs[2 * j + 1]));
        for (var k: u32 = 0; k < 4; k++) {
            let q_byte = get_byte(q_packed, k);
            let qh_hi = (qh_packed >> (j * 4 + k + 12)) & 0x10;
            let q_hi = (f32(((q_byte >> 4) & 0xF) | qh_hi) - 16.0) * d;
            let qh_lo = ((qh_packed >> (j * 4 + k)) << 4) & 0x10;
            let q_lo = (f32((q_byte & 0xF) | qh_lo) - 16.0) * d;
            let src1_offset = src1_idx_base + offset * 32 + j * 4 + k;
            sum += q_lo * f32(src1[src1_offset]);
            sum += q_hi * f32(src1[src1_offset + 16]);
        }
    }
    return sum;
}
#enddecl(Q5_0)

#decl(Q5_1)
struct q5_1 {
    d: f16,
    m: f16,
    qh: u32,
    qs: array<u32, 4>
};

fn multiply_add(src0_idx_base: u32, src1_idx_base: u32, offset: u32) -> f32 {
    let block_q5_1 = src0[src0_idx_base + offset];
    let d = f32(block_q5_1.d);
    let m = f32(block_q5_1.m);
    var sum: f32 = 0.0;
    for (var j: u32 = 0; j < 4; j++) {
        let q_packed = block_q5_1.qs[j];
        for (var k: u32 = 0; k < 4; k++) {
            let q_byte = get_byte(q_packed, k);
            let qh_hi = (block_q5_1.qh >> (j * 4 + k + 12)) & 0x10;
            let q_hi = f32(((q_byte >> 4) & 0xF) | qh_hi) * d + m;
            let qh_lo = ((block_q5_1.qh >> (j * 4 + k)) << 4) & 0x10;
            let q_lo = f32((q_byte & 0xF) | qh_lo) * d + m;
            let src1_offset = src1_idx_base + offset * 32 + j * 4 + k;
            sum += q_lo * f32(src1[src1_offset]);
            sum += q_hi * f32(src1[src1_offset + 16]);
        }
    }
    return sum;
}
#enddecl(Q5_1)

#decl(Q8_0)
struct q8_0 {
    d: f16,
    qs: array<f16, 16>
};

fn multiply_add(src0_idx_base: u32, src1_idx_base: u32, offset: u32) -> f32 {
    let block_q8_0 = src0[src0_idx_base + offset];
    let d = f32(block_q8_0.d);
    var sum: f32 = 0.0;
    for (var j: u32 = 0; j < 8; j++) {
        let q_packed = bitcast<u32>(vec2(block_q8_0.qs[2 * j], block_q8_0.qs[2 * j + 1]));
        for (var k: u32 = 0; k < 4; k++) {
            let q_byte = get_byte_i32(q_packed, k);
            let q_val = f32(q_byte) * d;
            let src1_offset = src1_idx_base + offset * 32 + j * 4 + k;
            sum += q_val * f32(src1[src1_offset]);
        }
    }
    return sum;
}
#enddecl(Q8_0)

#decl(Q8_1)
struct q8_1 {
    d: f16,
    m: f16,
    qs: array<u32, 8>
};

fn multiply_add(src0_idx_base: u32, src1_idx_base: u32, offset: u32) -> f32 {
    let block_q8_1 = src0[src0_idx_base + offset];
    let d = f32(block_q8_1.d);
    let m = f32(block_q8_1.m);
    var sum: f32 = 0.0;
    for (var j: u32 = 0; j < 8; j++) {
        let q_packed = block_q8_1.qs[j];
        for (var k: u32 = 0; k < 4; k++) {
            let q_byte = get_byte_i32(q_packed, k);
            let q_val = f32(q_byte) * d + m;
            let src1_offset = src1_idx_base + offset * 32 + j * 4 + k;
            sum += q_val * f32(src1[src1_offset]);
        }
    }
    return sum;
}
#enddecl(Q8_1)

#decl(Q2_K)
// 16 blocks of 16 elements each
struct q2_k {
    scales: array<u32, 4>,
    qs: array<u32, 16>,
    d: f16,
    dmin: f16
};

fn multiply_add(src0_idx_base: u32, src1_idx_base: u32, offset: u32) -> f32 {
    let block = src0[src0_idx_base + offset];
    let d = f32(block.d);
    let m = f32(block.dmin);
    var sum = 0.0;
    var src1_i = src1_idx_base + offset * 256;
    var is: u32 = 0;
    // 2 halves of the block (128 elements each)
    for (var q_b_idx: u32 = 0; q_b_idx < 64; q_b_idx += 32) {
        // 4 groups (each group has 2 blocks of 16 elements)
        for (var shift: u32 = 0; shift < 8; shift += 2) {
            // 2 blocks
            for (var k: u32 = 0; k < 32; k += 16) {
                let sc = get_byte(block.scales[is / 4], is % 4);
                is++;
                let dl = d * f32(sc & 0xF);
                let ml = m * f32(sc >> 4);
                for (var l: u32 = 0u; l < 16; l++) {
                    let q_idx = q_b_idx + k + l;
                    let q_byte = get_byte(block.qs[q_idx / 4], q_idx % 4);
                    let qs_val = (q_byte >> shift) & 3;
                    sum += (f32(qs_val) * dl - ml) * src1[src1_i];
                    src1_i++;
                }
            }
        }
    }
    return sum;
}

#enddecl(Q2_K)

#decl(Q3_K)
// 16 blocks of 16 elements each
struct q3_k {
    hmask: array<f16, 16>,
    qs: array<f16, 32>,
    scales: array<f16, 6>, // 6-bit quantized values
    d: f16
};

fn multiply_add(src0_idx_base: u32, src1_idx_base: u32, offset: u32) -> f32 {
    let block = src0[src0_idx_base + offset];
    let d = f32(block.d);

    // extract 6-bit scales, which consist of 4-bits from first 8 bytes of scale,
    // and 2-bits from the last 4 bytes
    let kmask1: u32 = 0x03030303;
    let kmask2: u32 = 0x0f0f0f0f;
    var scale_vals: array<u32, 4>;
    for (var i: u32 = 0; i < 4; i++) {
        scale_vals[i] = bitcast<u32>(vec2(block.scales[2 * i], block.scales[2 * i + 1]));
    }
    var tmp: u32 = scale_vals[2];
    scale_vals[2] = ((scale_vals[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    scale_vals[3] = ((scale_vals[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    scale_vals[0] = (scale_vals[0] & kmask2) | ((tmp & kmask1) << 4);
    scale_vals[1] = (scale_vals[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

    // convert arrays of f16 -> u32
    var hmask_vals: array<u32, 8>;
    for (var i: u32 = 0; i < 8; i++) {
        hmask_vals[i] = bitcast<u32>(vec2(block.hmask[2 * i], block.hmask[2 * i + 1]));
    }
    var qs_vals: array<u32, 16>;
    for (var i: u32 = 0; i < 16; i++) {
        qs_vals[i] = bitcast<u32>(vec2(block.qs[2 * i], block.qs[2 * i + 1]));
    }

    var sum = 0.0;
    var src1_i = src1_idx_base + offset * 256;
    var is: u32 = 0;
    var m: u32 = 1;
    // 2 halves of the block (128 elements each)
    for (var q_b_idx: u32 = 0; q_b_idx < 64; q_b_idx += 32) {
        // 4 groups (each group has 2 blocks of 16 elements)
        for (var shift: u32 = 0; shift < 8; shift += 2) {
            // 2 blocks
            for (var k: u32 = 0; k < 32; k += 16) {
                let sc = get_byte(scale_vals[is / 4], is % 4);
                is++;
                let dl = d * (f32(sc) - 32.0);
                for (var l: u32 = 0u; l < 16u; l++) {
                    let q_idx = q_b_idx + k + l;
                    let hm_idx = k + l;
                    let q_byte = get_byte(qs_vals[q_idx / 4], q_idx % 4);
                    let hmask_byte = get_byte(hmask_vals[hm_idx / 4], hm_idx % 4);
                    let hm = select(4.0, 0.0, (hmask_byte & m) != 0);
                    let qs_val = (q_byte >> shift) & 3;
                    sum += ((f32(qs_val) - hm) * dl) * src1[src1_i];
                    src1_i++;
                }
            }
            m <<= 1;
        }
    }
    return sum;
}

#enddecl(Q3_K)

#decl(Q45_K_SCALE_MIN)

fn get_scale_min(is: u32, scales: array<u32, 3>) -> vec2<f32> {
    if (is < 4) {
        let sc_byte = get_byte(scales[is / 4], is % 4);
        let min_byte = get_byte(scales[(is + 4) / 4], is % 4);
        return vec2(f32(sc_byte & 63), f32(min_byte & 63));
    } else {
        let sc_min_lo = get_byte(scales[(is + 4) / 4], (is + 4) % 4);
        let sc_hi = get_byte(scales[(is - 4) / 4], (is - 4) % 4);
        let min_hi = get_byte(scales[is / 4], is % 4);
        let sc = (sc_min_lo & 0xF) | ((sc_hi >> 6) << 4);
        let m = (sc_min_lo >> 4) | ((min_hi >> 6) << 4);
        return vec2(f32(sc), f32(m));
    }
}

#enddecl(Q45_K_SCALE_MIN)

#decl(Q4_K)
// 8 blocks of 32 elements each
struct q4_k {
    d: f16,
    dmin: f16,
    scales: array<u32, 3>,
    qs: array<u32, 32>
};

fn multiply_add(src0_idx_base: u32, src1_idx_base: u32, offset: u32) -> f32 {
    let block = src0[src0_idx_base + offset];
    let d = f32(block.d);
    let m = f32(block.dmin);
    var sum = 0.0;
    var src1_i = src1_idx_base + offset * 256;
    var is: u32 = 0;
    // 2 blocks each iteration
    for (var q_b_idx: u32 = 0; q_b_idx < 128; q_b_idx += 32) {
        for (var shift: u32 = 0; shift < 8; shift += 4) {
            let scale_min = get_scale_min(is, block.scales);
            is++;
            let dl = d * scale_min.x;
            let ml = m * scale_min.y;
            for (var l: u32 = 0; l < 32; l++) {
                let q_idx = q_b_idx + l;
                let q_byte = get_byte(block.qs[q_idx / 4], q_idx % 4);
                let qs_val = (q_byte >> shift) & 0xF;
                sum += (f32(qs_val) * dl - ml) * src1[src1_i];
                src1_i++;
            }
        }
    }
    return sum;
}

#enddecl(Q4_K)

#decl(Q5_K)
// 8 blocks of 32 elements each
struct q5_k {
    d: f16,
    dmin: f16,
    scales: array<u32, 3>,
    qh: array<u32, 8>,
    qs: array<u32, 32>
};

fn multiply_add(src0_idx_base: u32, src1_idx_base: u32, offset: u32) -> f32 {
    let block = src0[src0_idx_base + offset];
    let d = f32(block.d);
    let m = f32(block.dmin);
    var sum = 0.0;
    var src1_i = src1_idx_base + offset * 256;
    var is: u32 = 0;
    var u: u32 = 1;
    // 2 blocks each iteration
    for (var q_b_idx: u32 = 0; q_b_idx < 128; q_b_idx += 32) {
        for (var shift: u32 = 0; shift < 8; shift += 4) {
            let scale_min = get_scale_min(is, block.scales);
            is++;
            let dl = d * scale_min.x;
            let ml = m * scale_min.y;
            for (var l: u32 = 0; l < 32; l++) {
                let q_idx = q_b_idx + l;
                let q_byte = get_byte(block.qs[q_idx / 4], q_idx % 4);
                let qh_byte = get_byte(block.qh[l / 4], l % 4);
                let qs_val = (q_byte >> shift) & 0xF;
                let qh_val = select(0.0, 16.0, (qh_byte & u) != 0);
                sum += ((f32(qs_val) + qh_val) * dl - ml) * src1[src1_i];
               src1_i++;
            }
            u <<= 1;
        }
    }
    return sum;
}

#enddecl(Q5_K)

#decl(Q6_K)
// 16 blocks of 16 elements each
struct q6_k {
    ql: array<f16, 64>,
    qh: array<f16, 32>,
    scales: array<f16, 8>,
    d: f16
};

fn multiply_add(src0_idx_base: u32, src1_idx_base: u32, offset: u32) -> f32 {
    let block = src0[src0_idx_base + offset];
    let d = f32(block.d);

    // convert arrays of f16 -> u32
    var ql_vals: array<u32, 32>;
    for (var i: u32 = 0; i < 32; i++) {
        ql_vals[i] = bitcast<u32>(vec2(block.ql[2 * i], block.ql[2 * i + 1]));
    }
    var qh_vals: array<u32, 16>;
    for (var i: u32 = 0; i < 16; i++) {
        qh_vals[i] = bitcast<u32>(vec2(block.qh[2 * i], block.qh[2 * i + 1]));
    }
    var scale_vals: array<u32, 4>;
    for (var i: u32 = 0; i < 4; i++) {
        scale_vals[i] = bitcast<u32>(vec2(block.scales[2 * i], block.scales[2 * i + 1]));
    }

    var sum = 0.0;
    var src1_i = src1_idx_base + offset * 256;
    var qh_b_idx: u32 = 0;
    var sc_b_idx: u32 = 0;
    for (var ql_b_idx: u32 = 0; ql_b_idx < 128; ql_b_idx += 64) {
        for (var l: u32 = 0; l < 32; l++) {
            let ql13_b = get_byte(ql_vals[(ql_b_idx + l) / 4], (ql_b_idx + l) % 4);
            let ql24_b = get_byte(ql_vals[(ql_b_idx + l + 32) / 4], (ql_b_idx + l + 32) % 4);
            let qh_b = get_byte(qh_vals[(qh_b_idx + l) / 4], (qh_b_idx + l) % 4);

            let q1 = f32((ql13_b & 0xF) | ((qh_b & 3) << 4)) - 32.0;
            let q2 = f32((ql24_b & 0xF) | (((qh_b >> 2) & 3) << 4)) - 32.0;
            let q3 = f32((ql13_b >> 4) | (((qh_b >> 4) & 3) << 4)) - 32.0;
            let q4 = f32((ql24_b >> 4) | (((qh_b >> 6) & 3) << 4)) - 32.0;

            let is = l/16;
            let is1 = sc_b_idx + is;
            let sc1 = get_byte_i32(scale_vals[is1 / 4], is1 % 4);
            let is2 = sc_b_idx + is + 2;
            let sc2 = get_byte_i32(scale_vals[is2 / 4], is2 % 4);
            let is3 = sc_b_idx + is + 4;
            let sc3 = get_byte_i32(scale_vals[is3 / 4], is3 % 4);
            let is4 = sc_b_idx + is + 6;
            let sc4 = get_byte_i32(scale_vals[is4 / 4], is4 % 4);

            sum += d * f32(sc1) * q1 * src1[src1_i + l];
            sum += d * f32(sc2) * q2 * src1[src1_i + l + 32];
            sum += d * f32(sc3) * q3 * src1[src1_i + l + 64];
            sum += d * f32(sc4) * q4 * src1[src1_i + l + 96];
        }
        src1_i += 128;
        qh_b_idx += 32;
        sc_b_idx += 8;
    }
    return sum;
}

#enddecl(Q6_K)

#decl(IQ2_TABLES)
const kmask_iq2xs : array<u32, 2> = array<u32, 2>(
    0x08040201u, // 1, 2, 4, 8
    0x80402010u  // 16, 32, 64, 128
);

const kmask_iq2xs_u32 : array<u32, 8> = array<u32, 8>(
    1, 2, 4, 8, 16, 32, 64, 128
);

const ksigns_iq2xs: array<u32, 32> = array<u32, 32>(
    0x03828100,0x87060584,0x8b0a0988,0x0f8e8d0c,
    0x93121190,0x17969514,0x1b9a9918,0x9f1e1d9c,
    0xa32221a0,0x27a6a524,0x2baaa928,0xaf2e2dac,
    0x33b2b130,0xb73635b4,0xbb3a39b8,0x3fbebd3c,
    0xc34241c0,0x47c6c544,0x4bcac948,0xcf4e4dcc,
    0x53d2d150,0xd75655d4,0xdb5a59d8,0x5fdedd5c,
    0x63e2e160,0xe76665e4,0xeb6a69e8,0x6feeed6c,
    0xf37271f0,0x77f6f574,0x7bfaf978,0xff7e7dfc
);

const ksigns_iq2xs_u32 : array<u32, 128> = array<u32, 128>(
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255
);

const iq2xxs_grid = array<u32, 512>(
    0x08080808, 0x08080808, 0x0808082b, 0x08080808, 0x08081919, 0x08080808, 0x08082b08, 0x08080808,
    0x08082b2b, 0x08080808, 0x08190819, 0x08080808, 0x08191908, 0x08080808, 0x082b0808, 0x08080808,
    0x082b082b, 0x08080808, 0x082b2b08, 0x08080808, 0x082b2b2b, 0x08080808, 0x19080819, 0x08080808,
    0x19081908, 0x08080808, 0x19190808, 0x08080808, 0x19192b08, 0x08080808, 0x192b0819, 0x08080808,
    0x192b1908, 0x08080808, 0x2b080808, 0x08080808, 0x2b08082b, 0x08080808, 0x2b082b2b, 0x08080808,
    0x2b2b082b, 0x08080808, 0x08080819, 0x08080819, 0x08081908, 0x08080819, 0x08190808, 0x08080819,
    0x08191919, 0x08080819, 0x19080808, 0x08080819, 0x2b081908, 0x08080819, 0x2b192b08, 0x08080819,
    0x08080808, 0x0808082b, 0x0808082b, 0x0808082b, 0x082b082b, 0x0808082b, 0x2b08082b, 0x0808082b,
    0x08080819, 0x08081908, 0x08081908, 0x08081908, 0x08190808, 0x08081908, 0x082b0819, 0x08081908,
    0x082b1908, 0x08081908, 0x19080808, 0x08081908, 0x1908082b, 0x08081908, 0x19082b08, 0x08081908,
    0x192b0808, 0x08081908, 0x2b080819, 0x08081908, 0x2b081908, 0x08081908, 0x2b190808, 0x08081908,
    0x2b2b1908, 0x08081908, 0x08080808, 0x08081919, 0x0808082b, 0x08081919, 0x08082b08, 0x08081919,
    0x082b0808, 0x08081919, 0x1908192b, 0x08081919, 0x192b2b19, 0x08081919, 0x2b080808, 0x08081919,
    0x2b190819, 0x08081919, 0x08082b19, 0x0808192b, 0x08190808, 0x0808192b, 0x19080808, 0x0808192b,
    0x2b081908, 0x0808192b, 0x2b2b1908, 0x0808192b, 0x08080808, 0x08082b08, 0x08081919, 0x08082b08,
    0x08082b08, 0x08082b08, 0x08191908, 0x08082b08, 0x082b2b08, 0x08082b08, 0x19080819, 0x08082b08,
    0x19081908, 0x08082b08, 0x19190808, 0x08082b08, 0x1919082b, 0x08082b08, 0x2b082b08, 0x08082b08,
    0x08081908, 0x08082b19, 0x19080808, 0x08082b19, 0x0808082b, 0x08082b2b, 0x08191908, 0x08082b2b,
    0x08080819, 0x08190808, 0x08081908, 0x08190808, 0x08190808, 0x08190808, 0x082b0819, 0x08190808,
    0x19080808, 0x08190808, 0x192b0808, 0x08190808, 0x2b081908, 0x08190808, 0x2b190808, 0x08190808,
    0x2b191919, 0x08190808, 0x08080808, 0x08190819, 0x08082b08, 0x08190819, 0x082b0808, 0x08190819,
    0x19190808, 0x08190819, 0x19192b2b, 0x08190819, 0x2b080808, 0x08190819, 0x082b1908, 0x0819082b,
    0x19081919, 0x0819082b, 0x08080808, 0x08191908, 0x08082b08, 0x08191908, 0x082b0808, 0x08191908,
    0x082b1919, 0x08191908, 0x19082b19, 0x08191908, 0x2b080808, 0x08191908, 0x08192b08, 0x08191919,
    0x192b082b, 0x08191919, 0x08080808, 0x0819192b, 0x0819192b, 0x0819192b, 0x08080819, 0x08192b08,
    0x08081908, 0x08192b08, 0x08190808, 0x08192b08, 0x19080808, 0x08192b08, 0x2b080819, 0x08192b08,
    0x08080808, 0x08192b19, 0x08081919, 0x08192b19, 0x2b2b0808, 0x08192b19, 0x19190819, 0x08192b2b,
    0x08080808, 0x082b0808, 0x0808082b, 0x082b0808, 0x08082b2b, 0x082b0808, 0x19081908, 0x082b0808,
    0x192b0819, 0x082b0808, 0x2b080808, 0x082b0808, 0x2b08082b, 0x082b0808, 0x082b2b19, 0x082b0819,
    0x19082b08, 0x082b0819, 0x08080808, 0x082b082b, 0x0808082b, 0x082b082b, 0x08080819, 0x082b1908,
    0x08081908, 0x082b1908, 0x08190808, 0x082b1908, 0x19080808, 0x082b1908, 0x1919192b, 0x082b1908,
    0x08080808, 0x082b1919, 0x19080819, 0x082b1919, 0x192b1908, 0x082b1919, 0x2b190808, 0x082b192b,
    0x08082b08, 0x082b2b08, 0x082b0808, 0x082b2b08, 0x2b191908, 0x082b2b08, 0x19081908, 0x082b2b2b,
    0x08080819, 0x19080808, 0x08081908, 0x19080808, 0x08190808, 0x19080808, 0x08192b08, 0x19080808,
    0x082b0819, 0x19080808, 0x082b1908, 0x19080808, 0x19080808, 0x19080808, 0x19082b08, 0x19080808,
    0x1919192b, 0x19080808, 0x192b0808, 0x19080808, 0x2b080819, 0x19080808, 0x2b081908, 0x19080808,
    0x2b190808, 0x19080808, 0x08080808, 0x19080819, 0x082b0808, 0x19080819, 0x192b0819, 0x19080819,
    0x2b080808, 0x19080819, 0x2b081919, 0x19080819, 0x08080819, 0x1908082b, 0x08190808, 0x1908082b,
    0x19082b08, 0x1908082b, 0x1919192b, 0x1908082b, 0x192b2b08, 0x1908082b, 0x08080808, 0x19081908,
    0x08082b08, 0x19081908, 0x082b0808, 0x19081908, 0x2b080808, 0x19081908, 0x2b192b19, 0x19081908,
    0x0819082b, 0x19081919, 0x082b1908, 0x19081919, 0x08080808, 0x1908192b, 0x08080819, 0x19082b08,
    0x08081908, 0x19082b08, 0x08190808, 0x19082b08, 0x19080808, 0x19082b08, 0x19081919, 0x19082b08,
    0x08080808, 0x19082b19, 0x19192b08, 0x19082b19, 0x192b0819, 0x19082b19, 0x2b08082b, 0x19082b19,
    0x19081919, 0x19082b2b, 0x2b190808, 0x19082b2b, 0x08080808, 0x19190808, 0x08082b08, 0x19190808,
    0x08190819, 0x19190808, 0x08192b19, 0x19190808, 0x082b0808, 0x19190808, 0x2b080808, 0x19190808,
    0x2b082b08, 0x19190808, 0x08081908, 0x19190819, 0x1908082b, 0x19190819, 0x2b2b1908, 0x19190819,
    0x2b190819, 0x1919082b, 0x2b190808, 0x19191908, 0x2b19082b, 0x19191908, 0x08082b2b, 0x19191919,
    0x08080819, 0x1919192b, 0x19191908, 0x1919192b, 0x08080808, 0x19192b08, 0x08190819, 0x19192b08,
    0x08192b19, 0x19192b08, 0x192b1908, 0x19192b08, 0x19080808, 0x19192b19, 0x08082b08, 0x19192b2b,
    0x08081908, 0x192b0808, 0x08190808, 0x192b0808, 0x19080808, 0x192b0808, 0x192b2b08, 0x192b0808,
    0x08080808, 0x192b0819, 0x19191919, 0x192b0819, 0x08192b08, 0x192b082b, 0x192b0808, 0x192b082b,
    0x08080808, 0x192b1908, 0x08081919, 0x192b1908, 0x08190808, 0x192b1919, 0x0819082b, 0x192b1919,
    0x2b081908, 0x192b1919, 0x1908082b, 0x192b2b08, 0x08080808, 0x2b080808, 0x0808082b, 0x2b080808,
    0x08082b2b, 0x2b080808, 0x19080819, 0x2b080808, 0x2b08082b, 0x2b080808, 0x08081908, 0x2b080819,
    0x08192b08, 0x2b080819, 0x19080808, 0x2b080819, 0x08190819, 0x2b08082b, 0x08080819, 0x2b081908,
    0x08081908, 0x2b081908, 0x08190808, 0x2b081908, 0x08191919, 0x2b081908, 0x19080808, 0x2b081908,
    0x192b0808, 0x2b081908, 0x08080808, 0x2b081919, 0x1908192b, 0x2b081919, 0x2b191908, 0x2b081919,
    0x08082b19, 0x2b08192b, 0x19080808, 0x2b08192b, 0x192b0808, 0x2b08192b, 0x0808082b, 0x2b082b08,
    0x08081908, 0x2b082b19, 0x08190819, 0x2b082b2b, 0x08081908, 0x2b190808, 0x08190808, 0x2b190808,
    0x082b1908, 0x2b190808, 0x19080808, 0x2b190808, 0x2b2b0819, 0x2b190808, 0x0819192b, 0x2b190819,
    0x2b080808, 0x2b190819, 0x19081919, 0x2b19082b, 0x08080808, 0x2b191908, 0x082b082b, 0x2b191908,
    0x19081908, 0x2b191908, 0x19190819, 0x2b191919, 0x2b080819, 0x2b192b08, 0x082b0808, 0x2b192b19,
    0x0808082b, 0x2b2b0808, 0x19190808, 0x2b2b0808, 0x2b081919, 0x2b2b0808, 0x08082b19, 0x2b2b0819,
    0x08080808, 0x2b2b082b, 0x08192b08, 0x2b2b1908, 0x19190808, 0x2b2b2b08, 0x08081908, 0x2b2b2b19
);

#enddecl(IQ2_TABLES)

#decl(IQ2_XXS)
struct iq2_xxs {
    d: f16,
    qs: array<f16, 32>
};

fn multiply_add(src0_idx_base: u32, src1_idx_base: u32, offset: u32) -> f32 {
    let block = src0[src0_idx_base + offset];
    let d = f32(block.d);
    var src1_i = src1_idx_base + offset * 256;
    var sum = 0.0;
    for (var ib: u32 = 0; ib < 32; ib += 4) {
        var aux: array<u32, 2>;
        aux[0] = bitcast<u32>(vec2(block.qs[ib], block.qs[ib + 1]));
        aux[1] = bitcast<u32>(vec2(block.qs[ib + 2], block.qs[ib + 3]));
        let db = d * (0.5 + f32(aux[1] >> 28)) * 0.25;
        for (var l: u32 = 0; l < 4; l++) {
            let ig = get_byte(aux[0], l) * 8;
            let is = (aux[1] >> (7 * l)) & 127;
            let signs = get_byte(ksigns_iq2xs[is / 4], is % 4);
            //let signs = ksigns_iq2xs_u32[is];
            for (var j: u32 = 0; j < 8; j++) {
                let g = get_byte(iq2xxs_grid[(ig + j) / 4], (ig + j) % 4);
                let m = select(1.0, -1.0, (get_byte(kmask_iq2xs[j / 4], j % 4) & signs) != 0);
                //let m = select(1.0, -1.0, (kmask_iq2xs_u32[j] & signs) != 0);
                sum += db * f32(g) * m * src1[src1_i];
                src1_i++;
            }
        }
    }
    return sum;
}

#enddecl(IQ2_XXS)

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
//@group(0) @binding(3) var<storage, read_write> debug: array<f32>;

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
