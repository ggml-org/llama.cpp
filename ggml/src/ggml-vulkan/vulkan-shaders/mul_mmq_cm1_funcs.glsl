// Per-quant-type data structures and functions for the cm1 int8 coopmat path.
// Each quant type defines:
//   struct block_a_prefetch  — register data for one A-block per thread
//   block_a_load()           — load from global memory into a block_a_prefetch
//   block_a_to_shmem()       — unpack and write to shared memory
//   struct block_a_scale     - raw scale data of one A-block, loaded by a single thread
//   block_a_scale_load()     - load from global memory into a block_a_scale
//   block_a_scale_to_shmem() - decode and write to shared memory with a_scale_store()
//   A_IB_LO_MASK             - the low A-block index bits the decode depends on, passed as ib_lo
//                              (sub-block of a superblock, byte alignment of an unaligned block)
// Types with a common scale layout define A_SCALE_F16_D, A_SCALE_F16_DM or A_SCALE_K4 instead of
// the block_a_scale functions; the shared implementations follow the type list.
// block_a_load() should only issue loads: ALU work on loaded data belongs to the *_to_shmem functions,
// otherwise the prefetch has to wait for its own loads before the WMMA work it is meant to overlap.

// The two dwords covering the 4 bytes at byte_off, for block layouts that are not dword aligned.
uvec2 a_load_window(uint byte_off) {
    const uint dw = byte_off >> 2;
    return uvec2(data_a_u32[dw], data_a_u32[dw + 1u]);
}

// Blocks of 2 mod 4 bytes, field at 2 mod 4 bytes: the field is dword aligned in odd blocks only.
uint32_t a_window_dword(uvec2 w, uint ib_lo) {
    return (ib_lo & 1u) == 0u ? (w.x >> 16u) | (w.y << 16u) : w.x;
}

#ifdef A_KVALUES
// Four 4-bit indices, one per byte, to their int8 values.
uint32_t a_kvalues4(uint32_t idx) {
    const u8vec4 i = unpack8(idx);
    return pack32(i8vec4(cm1_kvalues[i.x], cm1_kvalues[i.y], cm1_kvalues[i.z], cm1_kvalues[i.w]));
}
#endif

// a_scale_fetch() returns (scale, -ACC_BIAS_F * scale, min) of scale group g for the epilogue.
#ifdef A_HAS_MIN
void a_scale_store(uint ks, uint row, float d, float m) {
    buf_a_dm[ks * BM + row] = vec2(d, m);
}
void a_scale_zero(uint ks, uint row) {
    a_scale_store(ks, row, 0.0, 0.0);
}
vec3 a_scale_fetch(uint ks, uint g, uint row) {
    const vec2 dm = buf_a_dm[ks * BM + row];
    return vec3(dm.x, -ACC_BIAS_F * dm.x, dm.y);
}
#elif KSCALES == 2
void a_scale_store(uint ks, uint row, float d0, float d1) {
    buf_a_d[(ks * KSCALES    ) * BM + row] = d0;
    buf_a_d[(ks * KSCALES + 1) * BM + row] = d1;
}
void a_scale_zero(uint ks, uint row) {
    a_scale_store(ks, row, 0.0, 0.0);
}
vec3 a_scale_fetch(uint ks, uint g, uint row) {
    const float d = buf_a_d[(ks * KSCALES + g) * BM + row];
    return vec3(d, -ACC_BIAS_F * d, 0.0);
}
#else
void a_scale_store(uint ks, uint row, float d) {
    buf_a_dn[ks * BM + row] = vec2(d, -ACC_BIAS_F * d);
}
void a_scale_zero(uint ks, uint row) {
    a_scale_store(ks, row, 0.0);
}
vec3 a_scale_fetch(uint ks, uint g, uint row) {
    return vec3(buf_a_dn[ks * BM + row], 0.0);
}
#endif

#if defined(DATA_A_Q4_0)

#define A_IB_LO_MASK 1u
#define A_SCALE_F16_D
struct block_a_prefetch {
    uvec2 qs;
};

block_a_prefetch block_a_load(uint ib, uint loadr) {
    block_a_prefetch blk;
    blk.qs = a_load_window(ib * 18u + 2u + loadr * 4u);
    return blk;
}

void block_a_to_shmem(block_a_prefetch blk, uint ib_lo, uint buf_ib, uint ks, uint loadr) {
    const uint32_t qs = a_window_dword(blk.qs, ib_lo);
    uint32_t lo4 = qs & 0x0F0F0F0F;
    uint32_t hi4 = (qs >> 4) & 0x0F0F0F0F;
    lo4 = ((lo4 | 0x80808080) - 0x08080808) ^ 0x80808080;
    hi4 = ((hi4 | 0x80808080) - 0x08080808) ^ 0x80808080;
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr    ] = lo4;
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr + 4] = hi4;
}

#elif defined(DATA_A_Q4_1)

#define A_IB_LO_MASK 0u
#define A_SCALE_F16_DM
struct block_a_prefetch {
    uint32_t qs;
};

block_a_prefetch block_a_load(uint ib, uint loadr) {
    block_a_prefetch blk;
    blk.qs = data_a_packed32[ib].qs[loadr];
    return blk;
}

void block_a_to_shmem(block_a_prefetch blk, uint ib_lo, uint buf_ib, uint ks, uint loadr) {
    // Store raw unsigned nibbles; the -8 offset is absorbed by the min term.
    uint32_t lo4 = blk.qs & 0x0F0F0F0F;
    uint32_t hi4 = (blk.qs >> 4) & 0x0F0F0F0F;
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr    ] = lo4;
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr + 4] = hi4;
}

#elif defined(DATA_A_Q5_0)

#define A_IB_LO_MASK 1u
#define A_SCALE_F16_D
struct block_a_prefetch {
    uvec2 qs;
    uvec2 qh;
};

block_a_prefetch block_a_load(uint ib, uint loadr) {
    block_a_prefetch blk;
    blk.qs = a_load_window(ib * 22u + 6u + loadr * 4u);
    blk.qh = a_load_window(ib * 22u + 2u);
    return blk;
}

void block_a_to_shmem(block_a_prefetch blk, uint ib_lo, uint buf_ib, uint ks, uint loadr) {
    const uint32_t qs = a_window_dword(blk.qs, ib_lo);
    const uint32_t qh = a_window_dword(blk.qh, ib_lo);
    uint32_t lo4 = qs & 0x0F0F0F0F;
    uint32_t hi4 = (qs >> 4) & 0x0F0F0F0F;
    lo4 |= ((qh >> (4u * loadr       )) & 0xFu) * 0x02040810u & 0x10101010u;
    hi4 |= ((qh >> (4u * loadr + 16u )) & 0xFu) * 0x02040810u & 0x10101010u;
    lo4 = ((lo4 | 0x80808080) - 0x10101010) ^ 0x80808080;
    hi4 = ((hi4 | 0x80808080) - 0x10101010) ^ 0x80808080;
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr    ] = lo4;
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr + 4] = hi4;
}

#elif defined(DATA_A_Q5_1)

#define A_IB_LO_MASK 0u
#define A_SCALE_F16_DM
struct block_a_prefetch {
    uint32_t qs;
    uint32_t qh;
};

block_a_prefetch block_a_load(uint ib, uint loadr) {
    block_a_prefetch blk;
    blk.qs = data_a_packed32[ib].qs[loadr];
    blk.qh = data_a_packed32[ib].qh;
    return blk;
}

void block_a_to_shmem(block_a_prefetch blk, uint ib_lo, uint buf_ib, uint ks, uint loadr) {
    // Store raw unsigned 5-bit values; the -16 offset is absorbed by the min term.
    uint32_t lo4 = blk.qs & 0x0F0F0F0F;
    uint32_t hi4 = (blk.qs >> 4) & 0x0F0F0F0F;
    lo4 |= ((blk.qh >> (4u * loadr       )) & 0xFu) * 0x02040810u & 0x10101010u;
    hi4 |= ((blk.qh >> (4u * loadr + 16u )) & 0xFu) * 0x02040810u & 0x10101010u;
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr    ] = lo4;
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr + 4] = hi4;
}

#elif defined(DATA_A_Q8_0)

#define A_IB_LO_MASK 1u
#define A_SCALE_F16_D
struct block_a_prefetch {
    uvec2 qs;
};

block_a_prefetch block_a_load(uint ib, uint loadr) {
    block_a_prefetch blk;
    blk.qs = a_load_window(ib * 34u + 2u + loadr * 4u);
    return blk;
}

void block_a_to_shmem(block_a_prefetch blk, uint ib_lo, uint buf_ib, uint ks, uint loadr) {
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr] = a_window_dword(blk.qs, ib_lo);
}

#elif defined(DATA_A_IQ4_NL)

#define A_IB_LO_MASK 1u
#define A_SCALE_F16_D
struct block_a_prefetch {
    uvec2 qs;
};

block_a_prefetch block_a_load(uint ib, uint loadr) {
    block_a_prefetch blk;
    blk.qs = a_load_window(ib * 18u + 2u + loadr * 4u);
    return blk;
}

void block_a_to_shmem(block_a_prefetch blk, uint ib_lo, uint buf_ib, uint ks, uint loadr) {
    const uint32_t qs = a_window_dword(blk.qs, ib_lo);
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr    ] = a_kvalues4(qs & 0x0F0F0F0F);
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr + 4] = a_kvalues4((qs >> 4) & 0x0F0F0F0F);
}

#elif defined(DATA_A_IQ4_XS)

#define A_IB_LO_MASK 7u
struct block_a_prefetch {
    uint32_t qs;
};

block_a_prefetch block_a_load(uint ib, uint loadr) {
    block_a_prefetch blk;
    const uint ib_k = ib / 8;
    const uint ib32 = ib % 8;
    blk.qs = data_a_packed32[ib_k].qs[4 * ib32 + loadr];
    return blk;
}

void block_a_to_shmem(block_a_prefetch blk, uint ib_lo, uint buf_ib, uint ks, uint loadr) {
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr    ] = a_kvalues4(blk.qs & 0x0F0F0F0F);
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr + 4] = a_kvalues4((blk.qs >> 4) & 0x0F0F0F0F);
}

struct block_a_scale {
    float16_t d;
    uint16_t scales_h;
    uint32_t scales_l;
};

block_a_scale block_a_scale_load(uint ib) {
    block_a_scale sc;
    const uint ib_k = ib / 8;
    sc.d = data_a_packed32[ib_k].d;
    sc.scales_h = data_a_packed32[ib_k].scales_h;
    sc.scales_l = data_a_packed32[ib_k].scales_l;
    return sc;
}

void block_a_scale_to_shmem(block_a_scale sc, uint ib_lo, uint ks, uint row) {
    const uint sl = (sc.scales_l >> (4 * ib_lo)) & 0xF;
    const uint sh = (uint(sc.scales_h) >> (2 * ib_lo)) & 3;
    a_scale_store(ks, row, float(sc.d) * float(int(sl | (sh << 4)) - 32));
}

#elif defined(DATA_A_MXFP4)

// 17-byte blocks: block_a_load packs u8 loads (a dword window was not a consistent win).
#define A_IB_LO_MASK 0u
struct block_a_prefetch {
    uint32_t qs;
};

block_a_prefetch block_a_load(uint ib, uint loadr) {
    block_a_prefetch blk;
    blk.qs = pack32(u8vec4(data_a[ib].qs[loadr * 4],
                            data_a[ib].qs[loadr * 4 + 1],
                            data_a[ib].qs[loadr * 4 + 2],
                            data_a[ib].qs[loadr * 4 + 3]));
    return blk;
}

void block_a_to_shmem(block_a_prefetch blk, uint ib_lo, uint buf_ib, uint ks, uint loadr) {
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr    ] = a_kvalues4(blk.qs & 0x0F0F0F0F);
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr + 4] = a_kvalues4((blk.qs >> 4) & 0x0F0F0F0F);
}

struct block_a_scale {
    uint8_t e;
};

block_a_scale block_a_scale_load(uint ib) {
    block_a_scale sc;
    sc.e = data_a[ib].e;
    return sc;
}

void block_a_scale_to_shmem(block_a_scale sc, uint ib_lo, uint ks, uint row) {
    a_scale_store(ks, row, e8m0_to_fp32(sc.e) * 0.5);
}

// LOAD_VEC_A=8 for k-quants and NVFP4: loadr has 4 positions, each writes 2 uint32

#elif defined(DATA_A_Q4_K)

#define A_IB_LO_MASK 7u
#define A_SCALE_K4
struct block_a_prefetch {
    uvec2 qs;
};

block_a_prefetch block_a_load(uint ib, uint loadr) {
    block_a_prefetch blk;
    const uint ib_k = ib / 8;
    const uint qs_base = ((ib % 8) >> 1) * 8;
    blk.qs = uvec2(data_a_packed32[ib_k].qs[qs_base + loadr * 2],
                   data_a_packed32[ib_k].qs[qs_base + loadr * 2 + 1]);
    return blk;
}

void block_a_to_shmem(block_a_prefetch blk, uint ib_lo, uint buf_ib, uint ks, uint loadr) {
    // Store raw unsigned nibbles; no -8 recentering needed.
    const uvec2 v = (blk.qs >> ((ib_lo & 1u) * 4u)) & 0x0F0F0F0Fu;
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr * 2    ] = v.x;
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr * 2 + 1] = v.y;
}

#elif defined(DATA_A_Q5_K)

#define A_IB_LO_MASK 7u
#define A_SCALE_K4
struct block_a_prefetch {
    uvec2 qs;
    uvec2 qh;
};

block_a_prefetch block_a_load(uint ib, uint loadr) {
    block_a_prefetch blk;
    const uint ib_k = ib / 8;
    const uint qs_base = ((ib % 8) >> 1) * 8;
    blk.qs = uvec2(data_a_packed32[ib_k].qs[qs_base + loadr * 2],
                   data_a_packed32[ib_k].qs[qs_base + loadr * 2 + 1]);
    blk.qh = uvec2(data_a_packed32[ib_k].qh[loadr * 2], data_a_packed32[ib_k].qh[loadr * 2 + 1]);
    return blk;
}

void block_a_to_shmem(block_a_prefetch blk, uint ib_lo, uint buf_ib, uint ks, uint loadr) {
    // Store raw unsigned 5-bit values (qs nibble | qh bit); no -16 recentering needed.
    const uvec2 v = ((blk.qs >> ((ib_lo & 1u) * 4u)) & 0x0F0F0F0Fu) | (((blk.qh >> ib_lo) & 0x01010101u) << 4);
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr * 2    ] = v.x;
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr * 2 + 1] = v.y;
}

#elif defined(DATA_A_Q6_K)

// 210-byte blocks: block_a_load packs u16 loads and decodes; a dword window would cost ~32 VGPRs.
#define A_IB_LO_MASK 7u
struct block_a_prefetch {
    uint32_t qs0;
    uint32_t qs1;
};

block_a_prefetch block_a_load(uint ib, uint loadr) {
    block_a_prefetch blk;
    const uint ib_k = ib / 8;
    const uint sub = ib % 8;
    const uint g = sub / 4;
    const uint j = sub % 4;

    const uint ql_u16 = g * 32 + (j & 1) * 16 + loadr * 4;
    const uint qh_u16 = g * 16 + loadr * 4;
    const uint qh_shift = j * 2;

    uint32_t ql0 = pack32(u16vec2(data_a_packed16[ib_k].ql[ql_u16    ],
                                   data_a_packed16[ib_k].ql[ql_u16 + 1]));
    uint32_t ql1 = pack32(u16vec2(data_a_packed16[ib_k].ql[ql_u16 + 2],
                                   data_a_packed16[ib_k].ql[ql_u16 + 3]));
    if (j >= 2) {
        ql0 = (ql0 >> 4) & 0x0F0F0F0F;
        ql1 = (ql1 >> 4) & 0x0F0F0F0F;
    } else {
        ql0 = ql0 & 0x0F0F0F0F;
        ql1 = ql1 & 0x0F0F0F0F;
    }

    uint32_t qh0 = pack32(u16vec2(data_a_packed16[ib_k].qh[qh_u16    ],
                                   data_a_packed16[ib_k].qh[qh_u16 + 1]));
    uint32_t qh1 = pack32(u16vec2(data_a_packed16[ib_k].qh[qh_u16 + 2],
                                   data_a_packed16[ib_k].qh[qh_u16 + 3]));

    blk.qs0 = ql0 | (((qh0 >> qh_shift) & 0x03030303) << 4);
    blk.qs1 = ql1 | (((qh1 >> qh_shift) & 0x03030303) << 4);

    return blk;
}

void block_a_to_shmem(block_a_prefetch blk, uint ib_lo, uint buf_ib, uint ks, uint loadr) {
    uint32_t v0 = ((blk.qs0 | 0x80808080) - 0x20202020) ^ 0x80808080;
    uint32_t v1 = ((blk.qs1 | 0x80808080) - 0x20202020) ^ 0x80808080;
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr * 2    ] = v0;
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr * 2 + 1] = v1;
}

struct block_a_scale {
    int16_t sc;
    float16_t d;
};

block_a_scale block_a_scale_load(uint ib) {
    block_a_scale sc;
    sc.sc = data_a_packed16[ib / 8].scales[ib % 8];
    sc.d = data_a_packed16[ib / 8].d;
    return sc;
}

void block_a_scale_to_shmem(block_a_scale sc, uint ib_lo, uint ks, uint row) {
    const i8vec2 s = unpack8(int32_t(sc.sc)).xy;
    const float d = float(sc.d);
    a_scale_store(ks, row, d * float(s.x), d * float(s.y));
}

#elif defined(DATA_A_Q3_K)

// 110-byte blocks: block_a_load packs u16 loads and decodes; a dword window would cost ~32 VGPRs.
#define A_IB_LO_MASK 7u
struct block_a_prefetch {
    uint32_t qs0;
    uint32_t qs1;
};

block_a_prefetch block_a_load(uint ib, uint loadr) {
    block_a_prefetch blk;
    const uint ib_k = ib / 8;
    const uint sub = ib % 8;
    const uint g = sub / 4;
    const uint j = sub % 4;
    const uint qs_shift = j * 2;
    const uint hm_bit = j + g * 4;

    const uint qs_u16 = g * 16 + loadr * 4;
    uint32_t qs0 = pack32(u16vec2(data_a_packed16[ib_k].qs[qs_u16    ],
                                   data_a_packed16[ib_k].qs[qs_u16 + 1]));
    uint32_t qs1 = pack32(u16vec2(data_a_packed16[ib_k].qs[qs_u16 + 2],
                                   data_a_packed16[ib_k].qs[qs_u16 + 3]));

    const uint hm_u16 = loadr * 4;
    uint32_t hm0 = pack32(u16vec2(data_a_packed16[ib_k].hmask[hm_u16    ],
                                   data_a_packed16[ib_k].hmask[hm_u16 + 1]));
    uint32_t hm1 = pack32(u16vec2(data_a_packed16[ib_k].hmask[hm_u16 + 2],
                                   data_a_packed16[ib_k].hmask[hm_u16 + 3]));

    blk.qs0 = ((qs0 >> qs_shift) & 0x03030303) | (((hm0 >> hm_bit) & 0x01010101) << 2);
    blk.qs1 = ((qs1 >> qs_shift) & 0x03030303) | (((hm1 >> hm_bit) & 0x01010101) << 2);

    return blk;
}

void block_a_to_shmem(block_a_prefetch blk, uint ib_lo, uint buf_ib, uint ks, uint loadr) {
    uint32_t v0 = ((blk.qs0 | 0x80808080) - 0x04040404) ^ 0x80808080;
    uint32_t v1 = ((blk.qs1 | 0x80808080) - 0x04040404) ^ 0x80808080;
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr * 2    ] = v0;
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr * 2 + 1] = v1;
}

struct block_a_scale {
    uint16_t lo;
    uint16_t hi;
    float16_t d;
};

block_a_scale block_a_scale_load(uint ib) {
    block_a_scale sc;
    const uint ib_k = ib / 8;
    const uint is = (ib % 8) * 2;
    sc.lo = data_a_packed16[ib_k].scales[(is % 8) / 2];
    sc.hi = data_a_packed16[ib_k].scales[(8 + (is % 4)) / 2];
    sc.d = data_a_packed16[ib_k].d;
    return sc;
}

void block_a_scale_to_shmem(block_a_scale sc, uint ib_lo, uint ks, uint row) {
    const uint is = ib_lo * 2;
    const uint lo = (uint(sc.lo) >> (4 * (is / 8))) & 0x0F0Fu;
    const uint hi = (uint(sc.hi) >> (2 * (is / 4))) & 0x0303u;
    const i8vec2 s = unpack8(int32_t(lo | (hi << 4))).xy;
    const float d = float(sc.d);
    a_scale_store(ks, row, d * float(int(s.x) - 32), d * float(int(s.y) - 32));
}

#elif defined(DATA_A_NVFP4)

#define A_IB_LO_MASK 0u
struct block_a_prefetch {
    uint32_t qs;
};

block_a_prefetch block_a_load(uint ib, uint loadr) {
    block_a_prefetch blk;
    const uint ib_k = ib / 2;
    const uint ihalf = ib % 2;
    const uint sub = ihalf * 2 + (loadr >> 1);
    const uint byte_group = loadr & 1u;

    // 36-byte blocks, qs at byte 4: always dword aligned.
    blk.qs = data_a_u32[ib_k * 9u + 1u + sub * 2u + byte_group];

    return blk;
}

void block_a_to_shmem(block_a_prefetch blk, uint ib_lo, uint buf_ib, uint ks, uint loadr) {
    const uint sub_base = (loadr >> 1) * 4;
    const uint byte_group = loadr & 1u;
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + sub_base + byte_group    ] = a_kvalues4(blk.qs & 0x0F0F0F0F);
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + sub_base + byte_group + 2] = a_kvalues4((blk.qs >> 4) & 0x0F0F0F0F);
}

struct block_a_scale {
    uint8_t d0;
    uint8_t d1;
};

block_a_scale block_a_scale_load(uint ib) {
    block_a_scale sc;
    const uint ib_k = ib / 2;
    const uint ihalf = ib % 2;
    sc.d0 = data_a[ib_k].d[ihalf * 2];
    sc.d1 = data_a[ib_k].d[ihalf * 2 + 1];
    return sc;
}

void block_a_scale_to_shmem(block_a_scale sc, uint ib_lo, uint ks, uint row) {
    a_scale_store(ks, row, ue4m3_to_fp32(sc.d0) * 0.5, ue4m3_to_fp32(sc.d1) * 0.5);
}

#endif

// ===== Shared A scale layouts =====

#if defined(A_SCALE_F16_D)

struct block_a_scale {
    float16_t d;
};

block_a_scale block_a_scale_load(uint ib) {
    block_a_scale sc;
    sc.d = data_a_packed16[ib].d;
    return sc;
}

void block_a_scale_to_shmem(block_a_scale sc, uint ib_lo, uint ks, uint row) {
    a_scale_store(ks, row, float(sc.d));
}

#elif defined(A_SCALE_F16_DM)

struct block_a_scale {
    f16vec2 dm;
};

block_a_scale block_a_scale_load(uint ib) {
    block_a_scale sc;
    sc.dm = data_a_packed32[ib].dm;
    return sc;
}

void block_a_scale_to_shmem(block_a_scale sc, uint ib_lo, uint ks, uint row) {
    a_scale_store(ks, row, float(sc.dm.x), float(sc.dm.y));
}

#elif defined(A_SCALE_K4)

struct block_a_scale {
    f16vec2 dm;
    uvec3 scales;
};

block_a_scale block_a_scale_load(uint ib) {
    block_a_scale sc;
    const uint ib_k = ib / 8;
    sc.dm = data_a_packed32[ib_k].dm;
    sc.scales = uvec3(data_a_packed32[ib_k].scales[0],
                      data_a_packed32[ib_k].scales[1],
                      data_a_packed32[ib_k].scales[2]);
    return sc;
}

void block_a_scale_to_shmem(block_a_scale sc, uint ib_lo, uint ks, uint row) {
    const uint sub = ib_lo;
    const uint j = sub & 3u;
    const uint s_j  = (sc.scales.x >> (8 * j)) & 0xFFu;
    const uint s_j4 = (sc.scales.y >> (8 * j)) & 0xFFu;
    const uint s_j8 = (sc.scales.z >> (8 * j)) & 0xFFu;
    const uint sc_val = (sub < 4) ? (s_j  & 0x3Fu) : ((s_j8 & 0x0Fu) | ((s_j  >> 6) << 4));
    const uint mn_val = (sub < 4) ? (s_j4 & 0x3Fu) : ((s_j8 >> 4)    | ((s_j4 >> 6) << 4));
    const vec2 dm = vec2(sc.dm);
    a_scale_store(ks, row, dm.x * float(sc_val), -(dm.y * float(mn_val)));
}

#endif

// ===== B-side: load and store =====

struct block_b_prefetch {
    ivec4 qs;
    float16_t d;
#ifdef A_HAS_MIN
    float16_t s;
#endif
};

block_b_prefetch block_b_load(uint ib_outer, uint ib_inner, uint loadr) {
    block_b_prefetch blk;
    blk.qs = data_b[ib_outer].qs[ib_inner * 2 + loadr];
    blk.d = data_b[ib_outer].ds[ib_inner].x;
#ifdef A_HAS_MIN
    blk.s = data_b[ib_outer].ds[ib_inner].y;
#endif
    return blk;
}

void block_b_to_shmem(block_b_prefetch blk, uint buf_ib, uint ks, uint loadr, bool in_bounds) {
    const ivec4 v = in_bounds ? blk.qs : ivec4(0);
    buf_b_qs[(buf_ib * QPITCH + ks * (BK / 4)) / 4 + loadr] = uvec4(v);
    if (loadr == 0) {
        buf_b_d[ks * BN + buf_ib] = in_bounds ? float(blk.d) : 0.0f;
#ifdef A_HAS_MIN
        buf_b_s[ks * BN + buf_ib] = in_bounds ? float(blk.s) : 0.0f;
#endif
    }
}

// ===== Framework macros =====

#ifdef MUL_MAT_ID
#define B_IB_CALC                                                                               \
            const u16vec2 row_idx = row_ids[buf_ib];                                            \
            const uint ib = pos_b_ib + row_idx.y * p.batch_stride_b / BK                        \
                          + (row_idx.x % p.ne11) * p.stride_b / BK;
#else
#define B_IB_CALC                                                                               \
            const uint ib = pos_b_ib + buf_ib * p.stride_b / BK;
#endif

#define A_STORE_LOOP(IB)                                                                        \
    [[unroll]] for (uint li = 0; li < A_LOADS; li++) {                                          \
        const uint buf_ib = loadc_a + li * loadstride_a;                                        \
        if (buf_ib < BM) {                                                                      \
            [[unroll]] for (uint ks = 0; ks < BK_STEP; ks++) {                                  \
                block_a_to_shmem(pre_a[li * BK_STEP + ks], (IB) & A_IB_LO_MASK, buf_ib, ks, loadr_a); \
            }                                                                                   \
        }                                                                                       \
    }

// When all rows share ib_lo and pos_a_ib is aligned, the ib_lo bits below BK_STEP are compile-time
// constants (the & ~ only tells the compiler what the condition guarantees).
#define A_IB_LO_FIXED (A_IB_LO_MASK & (BK_STEP - 1u))

// A scales are loaded with one (row, ks) block per thread instead of one per loadr group.
// A k step past end_k re-reads an in-bounds block; its scale is zeroed in STORE_BLOCK_TO_LDS.
#define PREFETCH_BLOCK(blk)                                                                     \
    [[unroll]] for (uint li = 0; li < A_SCALE_LOADS; li++) {                                    \
        const uint s_idx = gl_LocalInvocationID.x + li * BLOCK_SIZE;                            \
        if (s_idx < BM * BK_STEP) {                                                             \
            const uint ks = s_idx % BK_STEP;                                                    \
            const uint ks_ld = ((blk) + ks * BK < end_k) ? ks : 0u;                             \
            pre_sc[li] = block_a_scale_load(pos_a_ib + (s_idx / BK_STEP) * p.stride_a / BK      \
                                            + ks_ld);                                           \
        }                                                                                       \
    }                                                                                           \
    [[unroll]] for (uint li = 0; li < A_LOADS; li++) {                                          \
        const uint buf_ib = loadc_a + li * loadstride_a;                                        \
        if (buf_ib < BM) {                                                                      \
            const uint ib = pos_a_ib + buf_ib * p.stride_a / BK;                                \
            [[unroll]] for (uint ks = 0; ks < BK_STEP; ks++) {                                  \
                const uint ib_k = ((blk) + ks * BK < end_k) ? (ib + ks) : ib;                   \
                pre_a[li * BK_STEP + ks] = block_a_load(ib_k, loadr_a);                         \
            }                                                                                   \
        }                                                                                       \
    }                                                                                           \
    [[unroll]] for (uint li = 0; li < B_LOADS; li++) {                                          \
        const uint buf_ib = loadc_b + li * loadstride_b;                                        \
        if (buf_ib < BN) {                                                                      \
            B_IB_CALC                                                                           \
            [[unroll]] for (uint ks = 0; ks < BK_STEP; ks++) {                                  \
                const uint ib_k = ((blk) + ks * BK < end_k) ? (ib + ks) : ib;                   \
                pre_b[li * BK_STEP + ks] = block_b_load(ib_k / 4, ib_k % 4, loadr_b);          \
            }                                                                                   \
        }                                                                                       \
    }

#define STORE_BLOCK_TO_LDS(blk)                                                                 \
    [[unroll]] for (uint li = 0; li < A_SCALE_LOADS; li++) {                                    \
        const uint s_idx = gl_LocalInvocationID.x + li * BLOCK_SIZE;                            \
        if (s_idx < BM * BK_STEP) {                                                             \
            const uint ks = s_idx % BK_STEP;                                                    \
            if ((blk) + ks * BK < end_k) {                                                      \
                const uint ib = pos_a_ib + (s_idx / BK_STEP) * p.stride_a / BK + ks;            \
                block_a_scale_to_shmem(pre_sc[li], ib & A_IB_LO_MASK, ks, s_idx / BK_STEP);     \
            } else {                                                                            \
                a_scale_zero(ks, s_idx / BK_STEP);                                              \
            }                                                                                   \
        }                                                                                       \
    }                                                                                           \
    if ((p.stride_a / BK) % (A_IB_LO_MASK + 1u) == 0u && (pos_a_ib & A_IB_LO_FIXED) == 0u) {    \
        A_STORE_LOOP((pos_a_ib & ~A_IB_LO_FIXED) + ks)                                          \
    } else {                                                                                    \
        A_STORE_LOOP(pos_a_ib + buf_ib * p.stride_a / BK + ks)                                  \
    }                                                                                           \
    [[unroll]] for (uint li = 0; li < B_LOADS; li++) {                                          \
        const uint buf_ib = loadc_b + li * loadstride_b;                                        \
        if (buf_ib < BN) {                                                                      \
            [[unroll]] for (uint ks = 0; ks < BK_STEP; ks++) {                                  \
                const bool in_bounds = (blk) + ks * BK < end_k;                                 \
                block_b_to_shmem(pre_b[li * BK_STEP + ks], buf_ib, ks, loadr_b, in_bounds);     \
            }                                                                                   \
        }                                                                                       \
    }
