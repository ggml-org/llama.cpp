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
    float16_t d;
    float16_t m;
};

block_a_prefetch block_a_load(uint ib, uint loadr) {
    block_a_prefetch blk;
    blk.qs = pack32(u16vec2(data_a_packed16[ib].qs[loadr * 2],
                             data_a_packed16[ib].qs[loadr * 2 + 1]));
    blk.d = data_a_packed16[ib].d;
    blk.m = data_a_packed16[ib].m;
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
        buf_a_m[ks * BM + buf_ib] = float(blk.m);
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
    float16_t d;
    float16_t m;
    uint32_t qh;
};

block_a_prefetch block_a_load(uint ib, uint loadr) {
    block_a_prefetch blk;
    blk.qs = pack32(u16vec2(data_a_packed16[ib].qs[loadr * 2],
                             data_a_packed16[ib].qs[loadr * 2 + 1]));
    blk.d = data_a_packed16[ib].d;
    blk.m = data_a_packed16[ib].m;
    blk.qh = data_a_packed16[ib].qh;
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
        buf_a_m[ks * BM + buf_ib] = float(blk.m);
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
        pack32(i8vec4(kvalues_iq4nl_const[lo_idx.x], kvalues_iq4nl_const[lo_idx.y],
                      kvalues_iq4nl_const[lo_idx.z], kvalues_iq4nl_const[lo_idx.w]));
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr + 4] =
        pack32(i8vec4(kvalues_iq4nl_const[hi_idx.x], kvalues_iq4nl_const[hi_idx.y],
                      kvalues_iq4nl_const[hi_idx.z], kvalues_iq4nl_const[hi_idx.w]));

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
        pack32(i8vec4(kvalues_mxfp4_const[lo_idx.x], kvalues_mxfp4_const[lo_idx.y],
                      kvalues_mxfp4_const[lo_idx.z], kvalues_mxfp4_const[lo_idx.w]));
    buf_a_qs[buf_ib * QPITCH + ks * (BK / 4) + loadr + 4] =
        pack32(i8vec4(kvalues_mxfp4_const[hi_idx.x], kvalues_mxfp4_const[hi_idx.y],
                      kvalues_mxfp4_const[hi_idx.z], kvalues_mxfp4_const[hi_idx.w]));

    if (loadr == 0) {
        buf_a_d[ks * BM + buf_ib] = e8m0_to_fp32(blk.e) * 0.5;
    }
}

#endif

// ===== B-side: load and store =====

struct block_b_prefetch {
    ivec4 qs;
    float16_t d;
#if defined(DATA_A_Q4_1) || defined(DATA_A_Q5_1)
    float16_t s;
#endif
};

block_b_prefetch block_b_load(uint ib_outer, uint ib_inner, uint loadr) {
    block_b_prefetch blk;
    blk.qs = data_b[ib_outer].qs[ib_inner * 2 + loadr];
    blk.d = data_b[ib_outer].ds[ib_inner].x;
#if defined(DATA_A_Q4_1) || defined(DATA_A_Q5_1)
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
#if defined(DATA_A_Q4_1) || defined(DATA_A_Q5_1)
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
