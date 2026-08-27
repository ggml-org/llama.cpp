// Per-quant-type data structures and functions for the cm1 int8 coopmat path.
// Each quant type defines:
//   struct block_a_prefetch  — register data for one A-block per thread
//   block_a_load()           — load from global memory into a block_a_prefetch
//   block_a_to_shmem()       — unpack and write to shared memory

#if defined(DATA_A_Q4_0)

struct block_a_prefetch {
    uint32_t qs;
    float16_t d;
};

block_a_prefetch block_a_load(uint ib, uint loadr) {
    block_a_prefetch blk;
    blk.qs = pack32(u16vec2(data_a_packed16[ib].qs[loadr * 2],
                             data_a_packed16[ib].qs[loadr * 2 + 1]));
    blk.d = data_a_packed16[ib].d;
    return blk;
}

void block_a_to_shmem(block_a_prefetch blk, uint buf_ib, uint ks, uint loadr) {
    uint32_t lo4 = blk.qs & 0x0F0F0F0F;
    uint32_t hi4 = (blk.qs >> 4) & 0x0F0F0F0F;
    lo4 = ((lo4 | 0x80808080) - 0x08080808) ^ 0x80808080;
    hi4 = ((hi4 | 0x80808080) - 0x08080808) ^ 0x80808080;
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr    ] = lo4;
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr + 4] = hi4;

    if (loadr == 0) {
        buf_a_d[ks * BM + buf_ib] = float(blk.d);
    }
}

#elif defined(DATA_A_Q4_1)

struct block_a_prefetch {
    uint32_t qs;
    f16vec2 dm;
};

block_a_prefetch block_a_load(uint ib, uint loadr) {
    block_a_prefetch blk;
    blk.qs = data_a_packed32[ib].qs[loadr];
    blk.dm = data_a_packed32[ib].dm;
    return blk;
}

void block_a_to_shmem(block_a_prefetch blk, uint buf_ib, uint ks, uint loadr) {
    uint32_t lo4 = blk.qs & 0x0F0F0F0F;
    uint32_t hi4 = (blk.qs >> 4) & 0x0F0F0F0F;
    lo4 = ((lo4 | 0x80808080) - 0x08080808) ^ 0x80808080;
    hi4 = ((hi4 | 0x80808080) - 0x08080808) ^ 0x80808080;
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr    ] = lo4;
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr + 4] = hi4;

    if (loadr == 0) {
        const float d = float(blk.dm.x);
        buf_a_dm[ks * BM + buf_ib] = vec2(d, 8.0 * d + float(blk.dm.y));
    }
}

#elif defined(DATA_A_Q5_0)

struct block_a_prefetch {
    uint32_t qs;
    float16_t d;
    uint32_t qh;
};

block_a_prefetch block_a_load(uint ib, uint loadr) {
    block_a_prefetch blk;
    blk.qs = pack32(u16vec2(data_a_packed16[ib].qs[loadr * 2],
                             data_a_packed16[ib].qs[loadr * 2 + 1]));
    blk.d = data_a_packed16[ib].d;
    blk.qh = pack32(u16vec2(data_a_packed16[ib].qh[0], data_a_packed16[ib].qh[1]));
    return blk;
}

void block_a_to_shmem(block_a_prefetch blk, uint buf_ib, uint ks, uint loadr) {
    uint32_t lo4 = blk.qs & 0x0F0F0F0F;
    uint32_t hi4 = (blk.qs >> 4) & 0x0F0F0F0F;
    lo4 |= ((blk.qh >> (4u * loadr       )) & 0xFu) * 0x02040810u & 0x10101010u;
    hi4 |= ((blk.qh >> (4u * loadr + 16u )) & 0xFu) * 0x02040810u & 0x10101010u;
    lo4 = ((lo4 | 0x80808080) - 0x10101010) ^ 0x80808080;
    hi4 = ((hi4 | 0x80808080) - 0x10101010) ^ 0x80808080;
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr    ] = lo4;
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr + 4] = hi4;

    if (loadr == 0) {
        buf_a_d[ks * BM + buf_ib] = float(blk.d);
    }
}

#elif defined(DATA_A_Q5_1)

struct block_a_prefetch {
    uint32_t qs;
    f16vec2 dm;
    uint32_t qh;
};

block_a_prefetch block_a_load(uint ib, uint loadr) {
    block_a_prefetch blk;
    blk.qs = data_a_packed32[ib].qs[loadr];
    blk.dm = data_a_packed32[ib].dm;
    blk.qh = data_a_packed32[ib].qh;
    return blk;
}

void block_a_to_shmem(block_a_prefetch blk, uint buf_ib, uint ks, uint loadr) {
    uint32_t lo4 = blk.qs & 0x0F0F0F0F;
    uint32_t hi4 = (blk.qs >> 4) & 0x0F0F0F0F;
    lo4 |= ((blk.qh >> (4u * loadr       )) & 0xFu) * 0x02040810u & 0x10101010u;
    hi4 |= ((blk.qh >> (4u * loadr + 16u )) & 0xFu) * 0x02040810u & 0x10101010u;
    lo4 = ((lo4 | 0x80808080) - 0x10101010) ^ 0x80808080;
    hi4 = ((hi4 | 0x80808080) - 0x10101010) ^ 0x80808080;
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr    ] = lo4;
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr + 4] = hi4;

    if (loadr == 0) {
        const float d = float(blk.dm.x);
        buf_a_dm[ks * BM + buf_ib] = vec2(d, 16.0 * d + float(blk.dm.y));
    }
}

#elif defined(DATA_A_Q8_0)

struct block_a_prefetch {
    uint32_t qs;
    float16_t d;
};

block_a_prefetch block_a_load(uint ib, uint loadr) {
    block_a_prefetch blk;
    blk.qs = pack32(u16vec2(data_a_packed16[ib].qs[loadr * 2],
                             data_a_packed16[ib].qs[loadr * 2 + 1]));
    blk.d = data_a_packed16[ib].d;
    return blk;
}

void block_a_to_shmem(block_a_prefetch blk, uint buf_ib, uint ks, uint loadr) {
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr] = blk.qs;

    if (loadr == 0) {
        buf_a_d[ks * BM + buf_ib] = float(blk.d);
    }
}

#elif defined(DATA_A_IQ4_NL)

struct block_a_prefetch {
    uint32_t qs;
    float16_t d;
};

block_a_prefetch block_a_load(uint ib, uint loadr) {
    block_a_prefetch blk;
    blk.qs = pack32(u16vec2(data_a_packed16[ib].qs[loadr * 2],
                             data_a_packed16[ib].qs[loadr * 2 + 1]));
    blk.d = data_a_packed16[ib].d;
    return blk;
}

void block_a_to_shmem(block_a_prefetch blk, uint buf_ib, uint ks, uint loadr) {
    const u8vec4 lo_idx = unpack8(blk.qs & 0x0F0F0F0F);
    const u8vec4 hi_idx = unpack8((blk.qs >> 4) & 0x0F0F0F0F);
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr    ] =
        pack32(i8vec4(cm1_kvalues[lo_idx.x], cm1_kvalues[lo_idx.y],
                      cm1_kvalues[lo_idx.z], cm1_kvalues[lo_idx.w]));
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr + 4] =
        pack32(i8vec4(cm1_kvalues[hi_idx.x], cm1_kvalues[hi_idx.y],
                      cm1_kvalues[hi_idx.z], cm1_kvalues[hi_idx.w]));

    if (loadr == 0) {
        buf_a_d[ks * BM + buf_ib] = float(blk.d);
    }
}

#elif defined(DATA_A_MXFP4)

struct block_a_prefetch {
    uint32_t qs;
    uint8_t e;
};

block_a_prefetch block_a_load(uint ib, uint loadr) {
    block_a_prefetch blk;
    blk.qs = pack32(u8vec4(data_a[ib].qs[loadr * 4],
                            data_a[ib].qs[loadr * 4 + 1],
                            data_a[ib].qs[loadr * 4 + 2],
                            data_a[ib].qs[loadr * 4 + 3]));
    blk.e = data_a[ib].e;
    return blk;
}

void block_a_to_shmem(block_a_prefetch blk, uint buf_ib, uint ks, uint loadr) {
    const u8vec4 lo_idx = unpack8(blk.qs & 0x0F0F0F0F);
    const u8vec4 hi_idx = unpack8((blk.qs >> 4) & 0x0F0F0F0F);
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr    ] =
        pack32(i8vec4(cm1_kvalues[lo_idx.x], cm1_kvalues[lo_idx.y],
                      cm1_kvalues[lo_idx.z], cm1_kvalues[lo_idx.w]));
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr + 4] =
        pack32(i8vec4(cm1_kvalues[hi_idx.x], cm1_kvalues[hi_idx.y],
                      cm1_kvalues[hi_idx.z], cm1_kvalues[hi_idx.w]));

    if (loadr == 0) {
        buf_a_d[ks * BM + buf_ib] = e8m0_to_fp32(blk.e) * 0.5;
    }
}

// LOAD_VEC_A=8 for k-quants and NVFP4: loadr has 4 positions, each writes 2 uint32

#elif defined(DATA_A_Q4_K)

struct block_a_prefetch {
    uint32_t qs0;
    uint32_t qs1;
    uint ib;
};

block_a_prefetch block_a_load(uint ib, uint loadr) {
    block_a_prefetch blk;
    const uint ib_k = ib / 8;
    const uint sub = ib % 8;
    const uint qs_base = (sub >> 1) * 8;

    uint32_t raw0 = data_a_packed32[ib_k].qs[qs_base + loadr * 2];
    uint32_t raw1 = data_a_packed32[ib_k].qs[qs_base + loadr * 2 + 1];
    if ((sub & 1u) != 0u) {
        blk.qs0 = (raw0 >> 4) & 0x0F0F0F0F;
        blk.qs1 = (raw1 >> 4) & 0x0F0F0F0F;
    } else {
        blk.qs0 = raw0 & 0x0F0F0F0F;
        blk.qs1 = raw1 & 0x0F0F0F0F;
    }
    blk.ib = ib;

    return blk;
}

void block_a_to_shmem(block_a_prefetch blk, uint buf_ib, uint ks, uint loadr) {
    uint32_t v0 = ((blk.qs0 | 0x80808080) - 0x08080808) ^ 0x80808080;
    uint32_t v1 = ((blk.qs1 | 0x80808080) - 0x08080808) ^ 0x80808080;
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr * 2    ] = v0;
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr * 2 + 1] = v1;

    if (loadr == 0) {
        const uint ib_k = blk.ib / 8;
        const uint sub = blk.ib % 8;
        uint sc_val, mn_val;
        if (sub < 4) {
            sc_val = uint(data_a[ib_k].scales[sub]) & 0x3Fu;
            mn_val = uint(data_a[ib_k].scales[sub + 4]) & 0x3Fu;
        } else {
            sc_val = (uint(data_a[ib_k].scales[sub + 4]) & 0xFu) | ((uint(data_a[ib_k].scales[sub - 4]) & 0xC0u) >> 2);
            mn_val = (uint(data_a[ib_k].scales[sub + 4]) >> 4) | ((uint(data_a[ib_k].scales[sub]) & 0xC0u) >> 2);
        }
        vec2 dm = vec2(data_a_packed32[ib_k].dm);
        float d_scaled = dm.x * float(sc_val);
        buf_a_dm[ks * BM + buf_ib] = vec2(d_scaled, 8.0 * d_scaled - (dm.y * float(mn_val)));
    }
}

#elif defined(DATA_A_Q5_K)

struct block_a_prefetch {
    uint32_t qs0;
    uint32_t qs1;
    uint32_t qh0;
    uint32_t qh1;
    uint ib;
};

block_a_prefetch block_a_load(uint ib, uint loadr) {
    block_a_prefetch blk;
    const uint ib_k = ib / 8;
    const uint sub = ib % 8;
    const uint qs_base = (sub >> 1) * 8;

    uint32_t raw0 = data_a_packed32[ib_k].qs[qs_base + loadr * 2];
    uint32_t raw1 = data_a_packed32[ib_k].qs[qs_base + loadr * 2 + 1];
    if ((sub & 1u) != 0u) {
        blk.qs0 = (raw0 >> 4) & 0x0F0F0F0F;
        blk.qs1 = (raw1 >> 4) & 0x0F0F0F0F;
    } else {
        blk.qs0 = raw0 & 0x0F0F0F0F;
        blk.qs1 = raw1 & 0x0F0F0F0F;
    }
    blk.qh0 = ((data_a_packed32[ib_k].qh[loadr * 2    ] >> sub) & 0x01010101) << 4;
    blk.qh1 = ((data_a_packed32[ib_k].qh[loadr * 2 + 1] >> sub) & 0x01010101) << 4;
    blk.ib = ib;

    return blk;
}

void block_a_to_shmem(block_a_prefetch blk, uint buf_ib, uint ks, uint loadr) {
    uint32_t v0 = blk.qs0 | blk.qh0;
    uint32_t v1 = blk.qs1 | blk.qh1;
    v0 = ((v0 | 0x80808080) - 0x10101010) ^ 0x80808080;
    v1 = ((v1 | 0x80808080) - 0x10101010) ^ 0x80808080;
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr * 2    ] = v0;
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr * 2 + 1] = v1;

    if (loadr == 0) {
        const uint ib_k = blk.ib / 8;
        const uint sub = blk.ib % 8;
        uint sc_val, mn_val;
        if (sub < 4) {
            sc_val = uint(data_a[ib_k].scales[sub]) & 0x3Fu;
            mn_val = uint(data_a[ib_k].scales[sub + 4]) & 0x3Fu;
        } else {
            sc_val = (uint(data_a[ib_k].scales[sub + 4]) & 0xFu) | ((uint(data_a[ib_k].scales[sub - 4]) & 0xC0u) >> 2);
            mn_val = (uint(data_a[ib_k].scales[sub + 4]) >> 4) | ((uint(data_a[ib_k].scales[sub]) & 0xC0u) >> 2);
        }
        vec2 dm = vec2(data_a_packed32[ib_k].dm);
        float d_scaled = dm.x * float(sc_val);
        buf_a_dm[ks * BM + buf_ib] = vec2(d_scaled, 16.0 * d_scaled - (dm.y * float(mn_val)));
    }
}

#elif defined(DATA_A_Q6_K)

struct block_a_prefetch {
    uint32_t qs0;
    uint32_t qs1;
    uint ib;
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
    blk.ib = ib;

    return blk;
}

void block_a_to_shmem(block_a_prefetch blk, uint buf_ib, uint ks, uint loadr) {
    uint32_t v0 = ((blk.qs0 | 0x80808080) - 0x20202020) ^ 0x80808080;
    uint32_t v1 = ((blk.qs1 | 0x80808080) - 0x20202020) ^ 0x80808080;
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr * 2    ] = v0;
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr * 2 + 1] = v1;

    if (loadr == 0) {
        const uint ib_k = blk.ib / 8;
        const uint sub = blk.ib % 8;
        i8vec2 sc = unpack8(int32_t(int16_t(data_a_packed16[ib_k].scales[sub]))).xy;
        buf_a_d[(ks * KSCALES    ) * BM + buf_ib] = float(data_a_packed16[ib_k].d) * float(sc.x);
        buf_a_d[(ks * KSCALES + 1) * BM + buf_ib] = float(data_a_packed16[ib_k].d) * float(sc.y);
    }
}

#elif defined(DATA_A_Q3_K)

struct block_a_prefetch {
    uint32_t qs0;
    uint32_t qs1;
    uint ib;
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
    blk.ib = ib;

    return blk;
}

void block_a_to_shmem(block_a_prefetch blk, uint buf_ib, uint ks, uint loadr) {
    uint32_t v0 = ((blk.qs0 | 0x80808080) - 0x04040404) ^ 0x80808080;
    uint32_t v1 = ((blk.qs1 | 0x80808080) - 0x04040404) ^ 0x80808080;
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr * 2    ] = v0;
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr * 2 + 1] = v1;

    if (loadr == 0) {
        const uint ib_k = blk.ib / 8;
        const uint sub = blk.ib % 8;
        const uint is = sub * 2;
        uint lo = uint(data_a_packed16[ib_k].scales[(is % 8) / 2]);
        lo = (lo >> (4 * (is / 8))) & 0x0F0Fu;
        uint hi = uint(data_a_packed16[ib_k].scales[(8 + (is % 4)) / 2]);
        hi = (hi >> (2 * (is / 4))) & 0x0303u;
        uint combined = lo | (hi << 4);
        i8vec2 sc = unpack8(int32_t(combined)).xy;
        float d = float(data_a_packed16[ib_k].d);
        buf_a_d[(ks * KSCALES    ) * BM + buf_ib] = d * float(int(sc.x) - 32);
        buf_a_d[(ks * KSCALES + 1) * BM + buf_ib] = d * float(int(sc.y) - 32);
    }
}

#elif defined(DATA_A_NVFP4)

struct block_a_prefetch {
    uint32_t qs;
    uint8_t d0;
    uint8_t d1;
};

block_a_prefetch block_a_load(uint ib, uint loadr) {
    block_a_prefetch blk;
    const uint ib_k = ib / 2;
    const uint ihalf = ib % 2;
    const uint sub = ihalf * 2 + (loadr >> 1);
    const uint byte_group = loadr & 1u;

    blk.qs = pack32(u8vec4(data_a[ib_k].qs[sub * 8 + byte_group * 4],
                            data_a[ib_k].qs[sub * 8 + byte_group * 4 + 1],
                            data_a[ib_k].qs[sub * 8 + byte_group * 4 + 2],
                            data_a[ib_k].qs[sub * 8 + byte_group * 4 + 3]));
    blk.d0 = data_a[ib_k].d[ihalf * 2];
    blk.d1 = data_a[ib_k].d[ihalf * 2 + 1];

    return blk;
}

void block_a_to_shmem(block_a_prefetch blk, uint buf_ib, uint ks, uint loadr) {
    const u8vec4 lo_idx = unpack8(blk.qs & 0x0F0F0F0F);
    const u8vec4 hi_idx = unpack8((blk.qs >> 4) & 0x0F0F0F0F);
    const uint sub_base = (loadr >> 1) * 4;
    const uint byte_group = loadr & 1u;
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + sub_base + byte_group] =
        pack32(i8vec4(cm1_kvalues[lo_idx.x], cm1_kvalues[lo_idx.y],
                      cm1_kvalues[lo_idx.z], cm1_kvalues[lo_idx.w]));
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + sub_base + 2 + byte_group] =
        pack32(i8vec4(cm1_kvalues[hi_idx.x], cm1_kvalues[hi_idx.y],
                      cm1_kvalues[hi_idx.z], cm1_kvalues[hi_idx.w]));

    if (loadr == 0) {
        buf_a_d[(ks * KSCALES    ) * BM + buf_ib] = ue4m3_to_fp32(blk.d0) * 0.5;
        buf_a_d[(ks * KSCALES + 1) * BM + buf_ib] = ue4m3_to_fp32(blk.d1) * 0.5;
    }
}

#endif

// ===== B-side: load and store =====

struct block_b_prefetch {
    ivec4 qs;
    float16_t d;
#if defined(DATA_A_Q4_1) || defined(DATA_A_Q5_1) || defined(DATA_A_Q4_K) || defined(DATA_A_Q5_K)
    float16_t s;
#endif
};

block_b_prefetch block_b_load(uint ib_outer, uint ib_inner, uint loadr) {
    block_b_prefetch blk;
    blk.qs = data_b[ib_outer].qs[ib_inner * 2 + loadr];
    blk.d = data_b[ib_outer].ds[ib_inner].x;
#if defined(DATA_A_Q4_1) || defined(DATA_A_Q5_1) || defined(DATA_A_Q4_K) || defined(DATA_A_Q5_K)
    blk.s = data_b[ib_outer].ds[ib_inner].y;
#endif
    return blk;
}

void block_b_to_shmem(block_b_prefetch blk, uint buf_ib, uint ks, uint loadr, bool in_bounds) {
    const ivec4 v = in_bounds ? blk.qs : ivec4(0);
    const uint base = buf_ib * QPITCH + ks * (BK / 4) + loadr * 4;
    buf_b_qs[base    ] = v.x;
    buf_b_qs[base + 1] = v.y;
    buf_b_qs[base + 2] = v.z;
    buf_b_qs[base + 3] = v.w;
    if (loadr == 0) {
        buf_b_d[ks * BN + buf_ib] = in_bounds ? float(blk.d) : 0.0f;
#if defined(DATA_A_Q4_1) || defined(DATA_A_Q5_1) || defined(DATA_A_Q4_K) || defined(DATA_A_Q5_K)
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

#define PREFETCH_BLOCK(blk)                                                                     \
    [[unroll]] for (uint li = 0; li < A_LOADS; li++) {                                          \
        const uint buf_ib = loadc_a + li * loadstride_a;                                        \
        if (buf_ib < BM) {                                                                      \
            const uint ib = pos_a_ib + buf_ib * p.stride_a / BK;                                \
            [[unroll]] for (uint ks = 0; ks < BK_STEP; ks++) {                                  \
                pre_a[li * BK_STEP + ks] = block_a_load(ib + ks, loadr_a);                      \
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
    [[unroll]] for (uint li = 0; li < A_LOADS; li++) {                                          \
        const uint buf_ib = loadc_a + li * loadstride_a;                                        \
        if (buf_ib < BM) {                                                                      \
            [[unroll]] for (uint ks = 0; ks < BK_STEP; ks++) {                                  \
                block_a_to_shmem(pre_a[li * BK_STEP + ks], buf_ib, ks, loadr_a);                \
            }                                                                                   \
        }                                                                                       \
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
