// Quant-specific A-side unpacking: registers → shared memory
#if defined(DATA_A_Q4_0) || defined(DATA_A_Q4_1)
#define STORE_A_QS(buf_ib, ks, idx)                                                             \
                uint32_t lo4 = pre_a_qs[idx] & 0x0F0F0F0F;                                      \
                uint32_t hi4 = (pre_a_qs[idx] >> 4) & 0x0F0F0F0F;                               \
                lo4 = ((lo4 | 0x80808080) - 0x08080808) ^ 0x80808080;                           \
                hi4 = ((hi4 | 0x80808080) - 0x08080808) ^ 0x80808080;                           \
                buf_a_qs[(buf_ib) * QPITCH + (ks) * (BK / 4) + loadr_a    ] = lo4;              \
                buf_a_qs[(buf_ib) * QPITCH + (ks) * (BK / 4) + loadr_a + 4] = hi4;
#elif defined(DATA_A_Q5_0) || defined(DATA_A_Q5_1)
#define STORE_A_QS(buf_ib, ks, idx)                                                             \
                uint32_t lo4 = pre_a_qs[idx] & 0x0F0F0F0F;                                      \
                uint32_t hi4 = (pre_a_qs[idx] >> 4) & 0x0F0F0F0F;                               \
                const uint32_t qh = pre_a_qh[idx];                                              \
                lo4 |= ((qh >> (4u * loadr_a       )) & 0xFu) * 0x02040810u & 0x10101010u;      \
                hi4 |= ((qh >> (4u * loadr_a + 16u )) & 0xFu) * 0x02040810u & 0x10101010u;     \
                lo4 = ((lo4 | 0x80808080) - 0x10101010) ^ 0x80808080;                           \
                hi4 = ((hi4 | 0x80808080) - 0x10101010) ^ 0x80808080;                           \
                buf_a_qs[(buf_ib) * QPITCH + (ks) * (BK / 4) + loadr_a    ] = lo4;              \
                buf_a_qs[(buf_ib) * QPITCH + (ks) * (BK / 4) + loadr_a + 4] = hi4;
#elif defined(DATA_A_Q8_0)
#define STORE_A_QS(buf_ib, ks, idx)                                                             \
                buf_a_qs[(buf_ib) * QPITCH + (ks) * (BK / 4) + loadr_a] = pre_a_qs[idx];
#endif

// Quant-specific extra prefetch helpers (no-ops for types that don't need them)
#if defined(DATA_A_Q5_0)
#define PREFETCH_A_QH(li, ks, ib)                                                               \
                pre_a_qh[(li) * BK_STEP + (ks)] =                                              \
                    pack32(u16vec2(data_a_packed16[(ib) + (ks)].qh[0],                          \
                                   data_a_packed16[(ib) + (ks)].qh[1]));
#elif defined(DATA_A_Q5_1)
#define PREFETCH_A_QH(li, ks, ib)                                                               \
                pre_a_qh[(li) * BK_STEP + (ks)] = data_a_packed16[(ib) + (ks)].qh;
#else
#define PREFETCH_A_QH(li, ks, ib)
#endif

#if defined(DATA_A_Q4_1) || defined(DATA_A_Q5_1)
#define PREFETCH_A_M(li, ks, ib)                                                                \
                pre_a_m[(li) * BK_STEP + (ks)] = data_a_packed16[(ib) + (ks)].m;
#define PREFETCH_B_S(li, ks, ib_outer, ib_inner)                                                \
                pre_b_s[(li) * BK_STEP + (ks)] = data_b[(ib_outer)].ds[(ib_inner)].y;
#define STORE_A_M(buf_ib, ks, idx)                                                              \
                    buf_a_m[(ks) * BM + (buf_ib)] = float(pre_a_m[idx]);
#define STORE_B_S(in_bounds, buf_ib, ks, idx)                                                   \
                    buf_b_s[(ks) * BN + (buf_ib)] = (in_bounds) ? float(pre_b_s[idx]) : 0.0f;
#else
#define PREFETCH_A_M(li, ks, ib)
#define PREFETCH_B_S(li, ks, ib_outer, ib_inner)
#define STORE_A_M(buf_ib, ks, idx)
#define STORE_B_S(in_bounds, buf_ib, ks, idx)
#endif

#ifdef MUL_MAT_ID
#define B_IB_CALC                                                                               \
            const u16vec2 row_idx = row_ids[buf_ib];                                            \
            const uint ib = pos_b_ib + row_idx.y * p.batch_stride_b / BK                        \
                          + (row_idx.x % p.ne11) * p.stride_b / BK;
#else
#define B_IB_CALC                                                                               \
            const uint ib = pos_b_ib + buf_ib * p.stride_b / BK;
#endif

// Prefetch: global memory → registers
#define PREFETCH_BLOCK(blk)                                                                     \
    [[unroll]] for (uint li = 0; li < A_LOADS; li++) {                                          \
        const uint buf_ib = loadc_a + li * loadstride_a;                                        \
        if (buf_ib < BM) {                                                                      \
            const uint ib = pos_a_ib + buf_ib * p.stride_a / BK;                                \
            [[unroll]] for (uint ks = 0; ks < BK_STEP; ks++) {                                  \
                pre_a_qs[li * BK_STEP + ks] =                                                   \
                    pack32(u16vec2(data_a_packed16[ib + ks].qs[loadr_a * 2],                    \
                                   data_a_packed16[ib + ks].qs[loadr_a * 2 + 1]));              \
                pre_a_d[li * BK_STEP + ks] = data_a_packed16[ib + ks].d;                        \
                PREFETCH_A_QH(li, ks, ib)                                                       \
                PREFETCH_A_M(li, ks, ib)                                                        \
            }                                                                                   \
        }                                                                                       \
    }                                                                                           \
    [[unroll]] for (uint li = 0; li < B_LOADS; li++) {                                          \
        const uint buf_ib = loadc_b + li * loadstride_b;                                        \
        if (buf_ib < BN) {                                                                      \
            B_IB_CALC                                                                           \
            [[unroll]] for (uint ks = 0; ks < BK_STEP; ks++) {                                  \
                const uint ib_k = ((blk) + ks * BK < end_k) ? (ib + ks) : ib;                   \
                const uint ib_outer = ib_k / 4;                                                 \
                const uint ib_inner = ib_k % 4;                                                 \
                pre_b_qs[li * BK_STEP + ks] = data_b[ib_outer].qs[ib_inner * 2 + loadr_b];      \
                pre_b_d[li * BK_STEP + ks] = data_b[ib_outer].ds[ib_inner].x;                   \
                PREFETCH_B_S(li, ks, ib_outer, ib_inner)                                        \
            }                                                                                   \
        }                                                                                       \
    }

// Store: registers → shared memory (with quant-specific unpacking)
#define STORE_BLOCK_TO_LDS(blk)                                                                 \
    [[unroll]] for (uint li = 0; li < A_LOADS; li++) {                                          \
        const uint buf_ib = loadc_a + li * loadstride_a;                                        \
        if (buf_ib < BM) {                                                                      \
            [[unroll]] for (uint ks = 0; ks < BK_STEP; ks++) {                                  \
                const uint idx = li * BK_STEP + ks;                                             \
                STORE_A_QS(buf_ib, ks, idx)                                                     \
                if (loadr_a == 0) {                                                             \
                    buf_a_d[ks * BM + buf_ib] = float(pre_a_d[idx]);                            \
                    STORE_A_M(buf_ib, ks, idx)                                                  \
                }                                                                               \
            }                                                                                   \
        }                                                                                       \
    }                                                                                           \
    [[unroll]] for (uint li = 0; li < B_LOADS; li++) {                                          \
        const uint buf_ib = loadc_b + li * loadstride_b;                                        \
        if (buf_ib < BN) {                                                                      \
            [[unroll]] for (uint ks = 0; ks < BK_STEP; ks++) {                                  \
                const bool in_bounds = (blk) + ks * BK < end_k;                                 \
                const uint idx = li * BK_STEP + ks;                                             \
                const ivec4 v = in_bounds ? pre_b_qs[idx] : ivec4(0);                           \
                const uint base = buf_ib * QPITCH + ks * (BK / 4) + loadr_b * 4;                \
                buf_b_qs[base    ] = v.x;                                                       \
                buf_b_qs[base + 1] = v.y;                                                       \
                buf_b_qs[base + 2] = v.z;                                                       \
                buf_b_qs[base + 3] = v.w;                                                       \
                if (loadr_b == 0) {                                                             \
                    buf_b_d[ks * BN + buf_ib] = in_bounds ? float(pre_b_d[idx]) : 0.0f;         \
                    STORE_B_S(in_bounds, buf_ib, ks, idx)                                       \
                }                                                                               \
            }                                                                                   \
        }                                                                                       \
    }
