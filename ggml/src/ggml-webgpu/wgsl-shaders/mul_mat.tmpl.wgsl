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
    "DECLS": ["BYTE_HELPERS", "Q4_0_T", "Q4_0"]
  },
  {
    "REPLS": {
      "SRC0_TYPE": "q4_1",
      "SRC1_TYPE": "f32",
      "BLOCK_SIZE": 32
    },
    "DECLS": ["BYTE_HELPERS", "Q4_1_T", "Q4_1"]
  },
  {
    "REPLS": {
      "SRC0_TYPE": "q5_0",
      "SRC1_TYPE": "f32",
      "BLOCK_SIZE": 32
    },
    "DECLS": ["BYTE_HELPERS", "Q5_0_T", "Q5_0"]
  },
  {
    "REPLS": {
      "SRC0_TYPE": "q5_1",
      "SRC1_TYPE": "f32",
      "BLOCK_SIZE": 32
    },
    "DECLS": ["BYTE_HELPERS", "Q5_1_T", "Q5_1"]
  },
  {
    "REPLS": {
      "SRC0_TYPE": "q8_0",
      "SRC1_TYPE": "f32",
      "BLOCK_SIZE": 32
    },
    "DECLS": ["BYTE_HELPERS", "Q8_0_T", "Q8_0"]
  },
  {
    "REPLS": {
      "SRC0_TYPE": "q2_k",
      "SRC1_TYPE": "f32",
      "BLOCK_SIZE": 256
    },
    "DECLS": ["BYTE_HELPERS", "Q2_K_T", "Q2_K"]
  },
  {
    "REPLS": {
      "SRC0_TYPE": "q3_k",
      "SRC1_TYPE": "f32",
      "BLOCK_SIZE": 256
    },
    "DECLS": ["BYTE_HELPERS", "Q3_K_T", "Q3_K"]
  },
  {
    "REPLS": {
      "SRC0_TYPE": "q4_k",
      "SRC1_TYPE": "f32",
      "BLOCK_SIZE": 256
    },
    "DECLS": ["Q45_K_SCALE_MIN", "BYTE_HELPERS", "Q4_K_T", "Q4_K"]
  },
  {
    "REPLS": {
      "SRC0_TYPE": "q5_k",
      "SRC1_TYPE": "f32",
      "BLOCK_SIZE": 256
    },
    "DECLS": ["Q45_K_SCALE_MIN", "BYTE_HELPERS", "Q5_K_T", "Q5_K"]
  },
  {
    "REPLS": {
      "SRC0_TYPE": "q6_k",
      "SRC1_TYPE": "f32",
      "BLOCK_SIZE": 256
    },
    "DECLS": ["BYTE_HELPERS", "Q6_K_T", "Q6_K"]
  },
  {
    "REPLS": {
      "SRC0_TYPE": "iq2_xxs",
      "SRC1_TYPE": "f32",
      "BLOCK_SIZE": 256
    },
    "DECLS": ["BYTE_HELPERS", "IQ23_TABLES", "IQ2_XXS_GRID", "IQ2_XXS_T", "IQ2_XXS"]
  },
  {
    "REPLS": {
      "SRC0_TYPE": "iq2_xs",
      "SRC1_TYPE": "f32",
      "BLOCK_SIZE": 256
    },
    "DECLS": ["BYTE_HELPERS", "IQ23_TABLES", "IQ2_XS_GRID", "IQ2_XS_T", "IQ2_XS"]
  },
  {
    "REPLS": {
      "SRC0_TYPE": "iq2_s",
      "SRC1_TYPE": "f32",
      "BLOCK_SIZE": 256
    },
    "DECLS": ["BYTE_HELPERS", "IQ23_TABLES", "IQ2_S_GRID", "IQ2_S_T", "IQ2_S"]
  },
  {
    "REPLS": {
      "SRC0_TYPE": "iq3_xxs",
      "SRC1_TYPE": "f32",
      "BLOCK_SIZE": 256
    },
    "DECLS": ["BYTE_HELPERS", "IQ23_TABLES", "IQ3_XSS_GRID", "IQ3_XSS_T", "IQ3_XSS"]
  },
  {
    "REPLS": {
      "SRC0_TYPE": "iq3_s",
      "SRC1_TYPE": "f32",
      "BLOCK_SIZE": 256
    },
    "DECLS": ["BYTE_HELPERS", "IQ23_TABLES", "IQ3_S_GRID", "IQ3_S_T", "IQ3_S"]
  },
  {
    "REPLS": {
      "SRC0_TYPE": "iq1_s",
      "SRC1_TYPE": "f32",
      "BLOCK_SIZE": 256
    },
    "DECLS": ["BYTE_HELPERS", "IQ1_GRID", "IQ1_S_T", "IQ1_S"]
  },
  {
    "REPLS": {
      "SRC0_TYPE": "iq1_m",
      "SRC1_TYPE": "f32",
      "BLOCK_SIZE": 256
    },
    "DECLS": ["BYTE_HELPERS", "IQ1_GRID", "IQ1_M_T", "IQ1_M"]
  },
  {
    "REPLS": {
      "SRC0_TYPE": "iq4_nl",
      "SRC1_TYPE": "f32",
      "BLOCK_SIZE": 32,
    },
    "DECLS": ["BYTE_HELPERS", "IQ4_GRID", "IQ4_NL_T", "IQ4_NL"]
  },
  {
    "REPLS": {
      "SRC0_TYPE": "iq4_xs",
      "SRC1_TYPE": "f32",
      "BLOCK_SIZE": 256,
    },
    "DECLS": ["BYTE_HELPERS", "IQ4_GRID", "IQ4_XS_T", "IQ4_XS"]
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

#decl(Q4_K)
// 8 blocks of 32 elements each
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

#decl(IQ2_XXS)
fn multiply_add(src0_idx_base: u32, src1_idx_base: u32, offset: u32) -> f32 {
    let block = src0[src0_idx_base + offset];
    let d = f32(block.d);
    var src1_i = src1_idx_base + offset * 256;
    var sum = 0.0;
    for (var ib: u32 = 0; ib < 32; ib += 4) {
        let aux0 = bitcast<u32>(vec2(block.qs[ib], block.qs[ib + 1]));
        let aux1 = bitcast<u32>(vec2(block.qs[ib + 2], block.qs[ib + 3]));
        let db = d * (0.5 + f32(aux1 >> 28)) * 0.25;
        for (var l: u32 = 0; l < 4; l++) {
            let ig = get_byte(aux0, l) * 8;
            let is = (aux1 >> (7 * l)) & 127;
            let signs = get_byte(ksigns_iq2xs[is / 4], is % 4);
            for (var j: u32 = 0; j < 8; j++) {
                let g = get_byte(iq2xxs_grid[(ig + j) / 4], (ig + j) % 4);
                let m = select(1.0, -1.0, (get_byte(kmask_iq2xs[j / 4], j % 4) & signs) != 0);
                sum += db * f32(g) * m * src1[src1_i];
                src1_i++;
            }
        }
    }
    return sum;
}

#enddecl(IQ2_XXS)

#decl(IQ2_XS)
fn multiply_add(src0_idx_base: u32, src1_idx_base: u32, offset: u32) -> f32 {
    let block = src0[src0_idx_base + offset];
    let d = f32(block.d);
    var src1_i = src1_idx_base + offset * 256;
    var scale_vals = array<u32, 2>(
        bitcast<u32>(vec2(block.scales[0], block.scales[1])),
        bitcast<u32>(vec2(block.scales[2], block.scales[3]))
    );
    var sum = 0.0;
    for (var ib: u32 = 0; ib < 32; ib += 4) {
        let s = get_byte(scale_vals[ib / 16], (ib % 16) / 4);
        let db = array<f32, 2>(
            d * (0.5 + f32(s & 0xF)) * 0.25,
            d * (0.5 + f32(s >> 4)) * 0.25
        );
        for (var l: u32 = 0; l < 4; l++) {
            let qs_val = bitcast<u32>(vec2(block.qs[ib + l], 0.0));
            let ig = (qs_val & 511) * 8;
            let is = qs_val >> 9;
            let signs = get_byte(ksigns_iq2xs[is / 4], is % 4);
            let dl = db[l/2];
            for (var j: u32 = 0; j < 8; j++) {
                let g = get_byte(iq2xs_grid[(ig + j) / 4], (ig + j) % 4);
                let m = select(1.0, -1.0, (get_byte(kmask_iq2xs[j / 4], j % 4) & signs) != 0);
                sum += dl * f32(g) * m * src1[src1_i];
                src1_i++;
            }
        }
    }
    return sum;
}

#enddecl(IQ2_XS)

#decl(IQ2_S)
fn multiply_add(src0_idx_base: u32, src1_idx_base: u32, offset: u32) -> f32 {
    let block = src0[src0_idx_base + offset];
    let d = f32(block.d);
    var src1_i = src1_idx_base + offset * 256;
    var qs_vals : array<u32, 16>;
    for (var i: u32 = 0; i < 16; i++) {
        qs_vals[i] = bitcast<u32>(vec2(block.qs[i * 2], block.qs[i * 2 + 1]));
    }
    var qh_vals = array<u32, 2>(
        bitcast<u32>(vec2(block.qh[0], block.qh[1])),
        bitcast<u32>(vec2(block.qh[2], block.qh[3]))
    );
    var scale_vals = array<u32, 2>(
        bitcast<u32>(vec2(block.scales[0], block.scales[1])),
        bitcast<u32>(vec2(block.scales[2], block.scales[3]))
    );
    var sum = 0.0;
    for (var ib: u32 = 0; ib < 8; ib ++) {
        let s = get_byte(scale_vals[ib / 4], ib % 4);
        let db = array<f32, 2>(
            d * (0.5 + f32(s & 0xF)) * 0.25,
            d * (0.5 + f32(s >> 4)) * 0.25
        );
        let qs_w = qs_vals[ib];
        for (var l: u32 = 0; l < 4; l++) {
            let qh_b = (get_byte(qh_vals[ib / 4], ib % 4) << (8 - 2 * l)) & 0x300;
            let ig = (get_byte(qs_w, l) | qh_b) * 8;
            let signs = get_byte(qs_vals[ib + 8], l);
            let dl = db[l/2];
            for (var j: u32 = 0; j < 8; j++) {
                let g = get_byte(iq2s_grid[(ig + j) / 4], (ig + j) % 4);
                let m = select(1.0, -1.0, (get_byte(kmask_iq2xs[j / 4], j % 4) & signs) != 0);
                sum += dl * f32(g) * m * src1[src1_i];
                src1_i++;
            }
        }
    }
    return sum;
}


#enddecl(IQ2_S)

#decl(IQ3_XSS)
fn multiply_add(src0_idx_base: u32, src1_idx_base: u32, offset: u32) -> f32 {
    let block = src0[src0_idx_base + offset];
    let d = f32(block.d);
    var src1_i = src1_idx_base + offset * 256;
    var sum = 0.0;
    for (var ib: u32 = 0; ib < 16; ib += 2) {
        let sc_sign = bitcast<u32>(vec2(block.qs[ib + 32], block.qs[ib + 33]));
        let db = d * (0.5 + f32(sc_sign >> 28)) * 0.5;
        for (var l: u32 = 0; l < 4; l++) {
            let is = (sc_sign >> (7 * l)) & 127;
            let signs = get_byte(ksigns_iq2xs[is / 4], is % 4);
            let ig_val = bitcast<u32>(vec2(block.qs[ib * 2 + l], 0.0));
            let ig1 = get_byte(ig_val, 0);
            let ig2 = get_byte(ig_val, 1);
            for (var j: u32 = 0; j < 4; j++) {
                let g1 = get_byte(iq3xxs_grid[ig1], j);
                let g2 = get_byte(iq3xxs_grid[ig2], j);
                let m1 = select(1.0, -1.0, (get_byte(kmask_iq2xs[0], j) & signs) != 0);
                let m2 = select(1.0, -1.0, (get_byte(kmask_iq2xs[1], j) & signs) != 0);
                sum += db * f32(g1) * m1 * src1[src1_i];
                sum += db * f32(g2) * m2 * src1[src1_i + 4];
                src1_i++;
            }
            src1_i += 4;
        }
    }
    return sum;
}

#enddecl(IQ3_XSS)

#decl(IQ3_S)
fn multiply_add(src0_idx_base: u32, src1_idx_base: u32, offset: u32) -> f32 {
    let block = src0[src0_idx_base + offset];
    let d = f32(block.d);
    var src1_i = src1_idx_base + offset * 256;
    var qh_vals = array<u32, 2>(
        bitcast<u32>(vec2(block.qh[0], block.qh[1])),
        bitcast<u32>(vec2(block.qh[2], block.qh[3]))
    );
    var sign_vals: array<u32, 8>;
    for (var i: u32 = 0; i < 8; i++) {
        sign_vals[i] = bitcast<u32>(vec2(block.signs[i * 2], block.signs[i * 2 + 1]));
    }
    var scale_vals = bitcast<u32>(vec2(block.scales[0], block.scales[1]));
    var sum = 0.0;
    for (var ib: u32 = 0; ib < 4; ib++) {
        let s = get_byte(scale_vals, ib);
        let db = array<f32, 2>(
            d * (1.0 + 2.0 * f32(s & 0xF)),
            d * (1.0 + 2.0 * f32(s >> 4))
        );
        for (var k: u32 = 0; k < 2; k++) {
            let dl = db[k];
            let qh_byte = get_byte(qh_vals[ib / 2], (ib % 2) * 2 + k);
            let sign_w = sign_vals[ib * 2 + k];
            for (var l: u32 = 0; l < 4; l++) {
                let signs = get_byte(sign_w, l);
                let ig_val = bitcast<u32>(vec2(block.qs[ib * 8 + k * 4 + l], 0.0));
                let ig1 = get_byte(ig_val, 0) | ((qh_byte << ((8 - (2 * l)))) & 256);
                let ig2 = get_byte(ig_val, 1) | ((qh_byte << ((7 - (2 * l)))) & 256);
                for (var j: u32 = 0; j < 4; j++) {
                    let g1 = get_byte(iq3s_grid[ig1], j);
                    let g2 = get_byte(iq3s_grid[ig2], j);
                    let m1 = select(1.0, -1.0, (get_byte(kmask_iq2xs[0], j) & signs) != 0);
                    let m2 = select(1.0, -1.0, (get_byte(kmask_iq2xs[1], j) & signs) != 0);
                    sum += dl * f32(g1) * m1 * src1[src1_i];
                    sum += dl * f32(g2) * m2 * src1[src1_i + 4];
                    src1_i++;
                }
                src1_i += 4;
            }
        }
    }
    return sum;
}
#enddecl(IQ3_S)

#decl(IQ1_S)
fn multiply_add(src0_idx_base: u32, src1_idx_base: u32, offset: u32) -> f32 {
    let block = src0[src0_idx_base + offset];
    let d = f32(block.d);
    var src1_i = src1_idx_base + offset * 256;
    var sum = 0.0;
    for (var ib: u32 = 0; ib < 8; ib++) {
        let qh = bitcast<u32>(vec2(block.qh[ib], 0.0));
        let dl = d * (2 * f32((qh >> 12) & 7) + 1);
        let delta = select(IQ1_DELTA, -IQ1_DELTA, (qh & 0x8000) != 0);
        let qs_w = bitcast<u32>(vec2(block.qs[ib * 2], block.qs[ib * 2 + 1]));
        for (var l: u32 = 0; l < 4; l++) {
            let ig = (get_byte(qs_w, l) | (((qh >> (3 * l)) & 7) << 8)) * 8;
            for (var j: u32 = 0; j < 8; j++) {
                let gw = iq1_grid[(ig + j) / 16];
                let g = (gw >> (((ig + j) % 16) * 2)) & 3;
                let gs = bitcast<i32>(g << 30) >> 30;
                sum += dl * (f32(gs) + delta) * src1[src1_i];
                src1_i++;
            }
        }
    }
    return sum;
}

#enddecl(IQ1_S)

#decl(IQ1_M)
fn multiply_add(src0_idx_base: u32, src1_idx_base: u32, offset: u32) -> f32 {
    let block = src0[src0_idx_base + offset];

    let scale = ((block.scales[0] >> 12) & 0xF) | ((block.scales[0] >> 24) & 0x00F0) | ((block.scales[1] >> 4) & 0x0F00) | ((block.scales[1] >> 16) & 0xF000);
    let d = f32(bitcast<vec2<f16>>(scale).x);
    var src1_i = src1_idx_base + offset * 256;
    var sum = 0.0;
    for (var ib: u32 = 0; ib < 8; ib++) {
        let sw = (block.scales[ib / 4] >> (16 * ((ib / 2) % 2))) & 0xFFFF;
        let s1 : u32 = (sw >> (6 * (ib % 2))) & 0x7;
        let s2 : u32 = (sw >> (6 * (ib % 2) + 3)) & 0x7;
        var dl = array<f32, 2>(
            d * f32(2 * s1 + 1),
            d * f32(2 * s2 + 1)
        );

        let qh = block.qh[ib / 2] >> (16 * (ib % 2));
        var idx = array<u32, 4>(
            get_byte(block.qs[ib], 0) | ((qh << 8) & 0x700),
            get_byte(block.qs[ib], 1) | ((qh << 4) & 0x700),
            get_byte(block.qs[ib], 2) | ((qh) & 0x700),
            get_byte(block.qs[ib], 3) | ((qh >> 4) & 0x700)
        );
        var delta = array<f32, 4>(
            select(IQ1_DELTA, -IQ1_DELTA, (qh & 0x08) != 0),
            select(IQ1_DELTA, -IQ1_DELTA, (qh & 0x80) != 0),
            select(IQ1_DELTA, -IQ1_DELTA, ((qh >> 8) & 0x08) != 0),
            select(IQ1_DELTA, -IQ1_DELTA, ((qh >> 8) & 0x80) != 0)
        );
        for (var l: u32 = 0; l < 4; l++) {
            let ig = idx[l] * 8;
            for (var j: u32 = 0; j < 8; j++) {
                let gw = iq1_grid[(ig + j) / 16];
                let g = (gw >> (((ig + j) % 16) * 2)) & 3;
                let gs = bitcast<i32>(g << 30) >> 30;
                sum += dl[l/2] * (f32(gs) + delta[l]) * src1[src1_i];
                src1_i++;
            }
        }
    }
    return sum;
}

#enddecl(IQ1_M)

#decl(IQ4_NL)
fn multiply_add(src0_idx_base: u32, src1_idx_base: u32, offset: u32) -> f32 {
    let block = src0[src0_idx_base + offset];
    let d = f32(block.d);
    var src1_i = src1_idx_base + offset * 32;
    var sum = 0.0;
    var qs: array<u32, 4>;
    for (var i: u32 = 0; i < 4; i++) {
        qs[i] = bitcast<u32>(vec2(block.qs[i * 2], block.qs[i * 2 + 1]));
    }
    for (var j: u32 = 0; j < 16; j++) {
        let qsb = get_byte(qs[j / 4], j % 4);
        sum += d * f32(kvalues_iq4nl[qsb & 0xF]) * src1[src1_i];
        sum += d * f32(kvalues_iq4nl[qsb >> 4]) * src1[src1_i + 16];
        src1_i++;
    }
    return sum;
}

#enddecl(IQ4_NL)

#decl(IQ4_XS)
fn multiply_add(src0_idx_base: u32, src1_idx_base: u32, offset: u32) -> f32 {
    let block = src0[src0_idx_base + offset];
    let d = f32(block.d);
    let scales_h = bitcast<u32>(vec2(block.scales_h, 0.0));
    var src1_i = src1_idx_base + offset * 256;
    var sum = 0.0;
    for (var ib: u32 = 0; ib < 8; ib++) {
        let ls = ((get_byte(block.scales_l, ib / 2) >> (4 * (ib % 2))) & 0xF) | (((scales_h >> (2 * ib)) & 3) << 4);
        let dl = d * (f32(ls) - 32.0);
        for (var j: u32 = 0; j < 16; j++) {
            let iqs = ib * 16 + j;
            let qsb = get_byte(block.qs[iqs / 4], iqs % 4);
            sum += dl * f32(kvalues_iq4nl[qsb & 0xF]) * src1[src1_i];
            sum += dl * f32(kvalues_iq4nl[qsb >> 4]) * src1[src1_i + 16];
            src1_i++;
        }
        src1_i += 16;
    }
    return sum;
}

#enddecl(IQ4_XS)

#end(DECLS)

#define(SHADER)

enable f16;

DECLS

struct MulMatParams {
    offset_src0: u32, // in elements/blocks
    offset_src1: u32, // in elements/blocks
    offset_dst: u32, // in elements/blocks
    m: u32,
    n: u32,
    k: u32,
    // all strides are in elements/blocks
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

@group(0) @binding(0) var<storage, read_write> src0: array<{{SRC0_TYPE}}>; // M rows, K columns
@group(0) @binding(1) var<storage, read_write> src1: array<{{SRC1_TYPE}}>; // K rows, N columns (transposed)
@group(0) @binding(2) var<storage, read_write> dst: array<f32>; // M rows, N columns

@group(0) @binding(3) var<uniform> params: MulMatParams;

@compute @workgroup_size(256)
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

    let row = dst2_rem / params.m; // output row
    let col = dst2_rem % params.m; // output column

    let src0_idx_base = params.offset_src0 + src03_idx * params.stride_03 + src02_idx * params.stride_02 + col * params.stride_01;
    let src1_idx_base = params.offset_src1 + src13_idx * params.stride_13 + src12_idx * params.stride_12 + row * params.stride_11;

    var sum = 0.0;
    for (var i: u32 = 0u; i < params.k/{{BLOCK_SIZE}}; i = i + 1u) {
        sum += multiply_add(src0_idx_base, src1_idx_base, i);
    }
    dst[params.offset_dst + dst3_idx * dst3_stride + dst2_idx * dst2_stride + row * params.m + col] = sum;
}

#end(SHADER)
