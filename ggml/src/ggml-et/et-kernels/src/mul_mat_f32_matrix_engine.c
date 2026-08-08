#include <etsoc/common/utils.h>
#include <stdint.h>
#include "ggml_tensor.h"
#include "platform.h"
#include "tensor.h"

/*
 * High-performance F32 Matrix Multiply for ET-SoC-1 — TensorFMA32.
 * Double-buffered producer/consumer implementation.
 *
 * Hart 1 (producer): Transposes weights into double-buffered L2 SCP.
 * Hart 0 (consumer): Tensor engine compute (FMA, reduce, store).
 *
 * Two execution paths:
 *   * REUSE path (n_tiles >= 2): prepared weight tiles are reused across
 *     ru_n N-tiles to minimize DRAM reads.
 *   * ORIGINAL path (n_tiles == 1, GEMV): one output tile at a time, no reuse.
 */

#define NUM_COMPUTE_SHIRES 32
#define MINIONS_PER_SHIRE  32

#define TILE_M  16
#define TILE_N  16
#define TILE_K  16

#ifndef REUSE_MAX
#define REUSE_MAX 32
#endif
#ifndef KWIN
#define KWIN    16      // K-blocks per window
#endif

#define MACHINE_SLOTS (NUM_COMPUTE_SHIRES * MINIONS_PER_SHIRE) // 1024

#define CACHEOP_MAX 0
#define REP_RATE    0

#define A_L1_START 0    // L1 SCP lines  0..15 for A (buffer 0)
#define A_L1_ALT   16   // L1 SCP lines 16..31 for A (buffer 1)

#define SCP_PANEL_SIZE   (TILE_K * TILE_M * (uint64_t)sizeof(float))  // 1024 bytes

#define RU_BUF_BYTES     (KWIN * SCP_PANEL_SIZE)
#define RU_CACHE_BYTES   (2 * RU_BUF_BYTES)
#define RU_CSCRATCH_BYTES (REUSE_MAX * 16 * 64ULL)
#define SCP_READY_OFF    (RU_CACHE_BYTES + RU_CSCRATCH_BYTES)
#define SCP_CONSUMED_OFF (SCP_READY_OFF + 64)
#define SCP_PER_MINION   (SCP_CONSUMED_OFF + 64)

static inline void __attribute__((always_inline))
pack_b_f32_transpose(float *out, const char *src0_batch,
                     int64_t mb, int64_t kb, int64_t nb1_0) {
    static const int32_t __attribute__((aligned(32))) scatter_idx[8] = {
        0, 64, 128, 192, 256, 320, 384, 448
    };

    unsigned long old_mask;
    __asm__ volatile(
        "mova.x.m  %[ms]            \n\t"
        "mov.m.x   m0, x0, 0xFF     \n\t"   // all 8 lanes active
        "flw.ps    f1, (%[idx])     \n\t"   // f1 = scatter offsets
        : [ms] "=&r"(old_mask)
        : [idx] "r"(scatter_idx)
        : "f1"
    );

    char *pbase = (char *) out;
    for (int j = 0; j < TILE_M; ++j) {
        const float *row = (const float *)(src0_batch + (mb + j) * nb1_0) + kb;
        char *col = pbase + j * 4;           // column m=j of the panel

        __asm__ volatile(
            "flw.ps      f2, 0(%[src])  \n\t"   // load row[0..7] (32 bytes)
            "flw.ps      f3, 32(%[src]) \n\t"   // load row[8..15] (32 bytes)
            "fscw.ps     f2, f1(%[c0])  \n\t"   // store row[0..7] to lines 0..7, col j
            "fscw.ps     f3, f1(%[c8])  \n\t"   // store row[8..15] to lines 8..15, col j
            :
            : [src] "r"(row), [c0] "r"(col), [c8] "r"(col + 8 * 64)
            : "f2", "f3", "memory"
        );
    }

    __asm__ volatile("mova.m.x %0" :: "r"(old_mask));
}

#define C_ROW_PAIR_ST(n0, n1, base)                                              \
    __asm__ volatile("fsw.ps f" #n0 ", (%0)\n\t fsw.ps f" #n1 ", (%1)\n\t"       \
                     :: "r"((base)), "r"((base) + 32) : "memory")
#define C_ROW_PAIR_LD(n0, n1, base)                                              \
    __asm__ volatile("flw.ps f" #n0 ", (%0)\n\t flw.ps f" #n1 ", (%1)\n\t"       \
                     :: "r"((base)), "r"((base) + 32) : "f" #n0, "f" #n1)

static inline void __attribute__((always_inline))
c_spill(char *s) {
    C_ROW_PAIR_ST(0,  1,  s + 0  * 64); C_ROW_PAIR_ST(2,  3,  s + 1  * 64);
    C_ROW_PAIR_ST(4,  5,  s + 2  * 64); C_ROW_PAIR_ST(6,  7,  s + 3  * 64);
    C_ROW_PAIR_ST(8,  9,  s + 4  * 64); C_ROW_PAIR_ST(10, 11, s + 5  * 64);
    C_ROW_PAIR_ST(12, 13, s + 6  * 64); C_ROW_PAIR_ST(14, 15, s + 7  * 64);
    C_ROW_PAIR_ST(16, 17, s + 8  * 64); C_ROW_PAIR_ST(18, 19, s + 9  * 64);
    C_ROW_PAIR_ST(20, 21, s + 10 * 64); C_ROW_PAIR_ST(22, 23, s + 11 * 64);
    C_ROW_PAIR_ST(24, 25, s + 12 * 64); C_ROW_PAIR_ST(26, 27, s + 13 * 64);
    C_ROW_PAIR_ST(28, 29, s + 14 * 64); C_ROW_PAIR_ST(30, 31, s + 15 * 64);
}

static inline void __attribute__((always_inline))
c_seed(char *s) {
    C_ROW_PAIR_LD(0,  1,  s + 0  * 64); C_ROW_PAIR_LD(2,  3,  s + 1  * 64);
    C_ROW_PAIR_LD(4,  5,  s + 2  * 64); C_ROW_PAIR_LD(6,  7,  s + 3  * 64);
    C_ROW_PAIR_LD(8,  9,  s + 4  * 64); C_ROW_PAIR_LD(10, 11, s + 5  * 64);
    C_ROW_PAIR_LD(12, 13, s + 6  * 64); C_ROW_PAIR_LD(14, 15, s + 7  * 64);
    C_ROW_PAIR_LD(16, 17, s + 8  * 64); C_ROW_PAIR_LD(18, 19, s + 9  * 64);
    C_ROW_PAIR_LD(20, 21, s + 10 * 64); C_ROW_PAIR_LD(22, 23, s + 11 * 64);
    C_ROW_PAIR_LD(24, 25, s + 12 * 64); C_ROW_PAIR_LD(26, 27, s + 13 * 64);
    C_ROW_PAIR_LD(28, 29, s + 14 * 64); C_ROW_PAIR_LD(30, 31, s + 15 * 64);
}

int entry_point(struct ggml_et_binary_params* params, void* env) {
    (void) env;

    uint64_t hart_id = get_hart_id();
    uint64_t shire_id = get_shire_id();

    if (shire_id >= NUM_COMPUTE_SHIRES) return 0;

    const int is_hart1 = hart_id & 1;
    uint64_t local_minion = (hart_id >> 1) & 0x1F;

    const int64_t K = params->src0.ne[0];
    const int64_t M = params->src0.ne[1];
    const int64_t N = params->src1.ne[1];

    if ((M % TILE_M) != 0) return 0;
    if ((K % TILE_K) != 0) return 0;

    const int64_t ne2_0 = params->src0.ne[2], ne3_0 = params->src0.ne[3];
    const int64_t ne2_1 = params->src1.ne[2], ne3_1 = params->src1.ne[3];

    const int64_t nb1_0 = params->src0.nb[1];
    const int64_t nb2_0 = params->src0.nb[2], nb3_0 = params->src0.nb[3];
    const int64_t nb1_1 = params->src1.nb[1];
    const int64_t nb2_1 = params->src1.nb[2], nb3_1 = params->src1.nb[3];
    const int64_t nb1_d = params->dst.nb[1];
    const int64_t nb2_d = params->dst.nb[2],  nb3_d = params->dst.nb[3];

    const char* src0_base = (const char*)params->src0.data;
    const char* src1_base = (const char*)params->src1.data;
    char*       dst_base  = (char*)params->dst.data;

    const int64_t m_tiles = M / TILE_M;
    const int64_t n_tiles = (N + TILE_N - 1) / TILE_N;
    const int64_t batch_count = ne2_1 * ne3_1;

    const int64_t r2 = ne2_1 / ne2_0;
    const int64_t r3 = ne3_1 / ne3_0;

    const int64_t k_steps = K / TILE_K;

    // L2 SCP pointers for this minion
    const uint64_t scp_base = local_minion * SCP_PER_MINION;
    volatile uint32_t *ready_ctr =
        (volatile uint32_t *) et_shire_l2scp_local(scp_base + SCP_READY_OFF);
    volatile uint32_t *consumed_ctr =
        (volatile uint32_t *) et_shire_l2scp_local(scp_base + SCP_CONSUMED_OFF);

    int64_t n_groups = (n_tiles + REUSE_MAX - 1) / REUSE_MAX;
    if (n_groups < 1) n_groups = 1;
    int64_t ru_n = (n_tiles + n_groups - 1) / n_groups;
    if (ru_n < 1) ru_n = 1;

    const int64_t units_pb   = m_tiles * n_groups;
    const int64_t base_units = units_pb * batch_count;

    int64_t k_splits = 1;
    {
        int64_t ks = 1;
        while (ks * 2 <= MINIONS_PER_SHIRE &&
               base_units * ks * 2 <= MACHINE_SLOTS &&
               (k_steps % (ks * 2)) == 0) {
            ks *= 2;
        }
        k_splits = ks;
    }

    const int64_t tiles_per_shire = MINIONS_PER_SHIRE / k_splits;
    const int64_t k_split         = local_minion % k_splits;
    const int64_t local_tile_idx  = local_minion / k_splits;
    const int64_t tiles_stride    = (int64_t) NUM_COMPUTE_SHIRES * tiles_per_shire;
    const int64_t my_start        = (int64_t) shire_id + local_tile_idx * NUM_COMPUTE_SHIRES;

    const int64_t k_steps_per_split = k_steps / k_splits;
    const int64_t k_start_block     = k_split * k_steps_per_split;
    const int64_t k_start           = k_start_block * TILE_K;
    const int64_t k_end             = k_start + k_steps_per_split * TILE_K;

    const int reuse_ok = (ru_n >= 2);

    // =====================================================================
    // REUSE path: transpose each K-window once, reuse across ru_n N-tiles.
    // =====================================================================
    if (reuse_ok) {
        char *cache_buf[2] = {
            (char *) et_shire_l2scp_local(scp_base),
            (char *) et_shire_l2scp_local(scp_base + RU_BUF_BYTES),
        };
        char *cscratch = (char *) et_shire_l2scp_local(scp_base + RU_CACHE_BYTES);

        const int64_t k_end_block = k_start_block + k_steps_per_split;
        const int64_t n_windows   = (k_steps_per_split + KWIN - 1) / KWIN;

        // ----- Hart 1: producer -----
        if (is_hart1) {
            scp_signal(ready_ctr, 0);
            scp_signal(consumed_ctr, 0);

            et_barrier(ET_BARRIER_MINION);
            uint32_t wid = 0;

            for (int64_t unit = my_start; unit < base_units; unit += tiles_stride) {
                const int64_t batch_idx    = unit / units_pb;
                const int64_t unit_in_b    = unit % units_pb;
                const int64_t mb_idx       = unit_in_b % m_tiles;

                const int64_t i3   = batch_idx / ne2_1;
                const int64_t i2   = batch_idx % ne2_1;
                const int64_t i2_0 = i2 / r2;
                const int64_t i3_0 = i3 / r3;

                const char *src0_batch = src0_base + i3_0 * nb3_0 + i2_0 * nb2_0;
                const int64_t mb = mb_idx * TILE_M;

                // Prefetch first window's weights
                {
                    const int64_t kbn_first = (k_start_block + KWIN <= k_end_block) ? KWIN : (k_end_block - k_start_block);
                    for (int64_t i = 0; i < kbn_first; ++i) {
                        l2_prefetch(src0_batch + mb * nb1_0 + (k_start_block + i) * TILE_K * sizeof(float), 16, nb1_0);
                    }
                }

                for (int64_t kw = 0; kw < n_windows; ++kw) {
                    const int buf = wid & 1;
                    if (wid >= 2) scp_wait(consumed_ctr, wid - 1);

                    const int64_t kb0 = k_start_block + kw * KWIN;
                    const int64_t kbn = (kb0 + KWIN <= k_end_block) ? KWIN : (k_end_block - kb0);

                    // Prefetch next window's weights
                    if (kw + 1 < n_windows) {
                        const int64_t kb_next = kb0 + KWIN;
                        const int64_t kbn_next = (kb_next + KWIN <= k_end_block) ? KWIN : (k_end_block - kb_next);
                        for (int64_t i = 0; i < kbn_next; ++i) {
                            l2_prefetch(src0_batch + mb * nb1_0 + (kb_next + i) * TILE_K * sizeof(float), 16, nb1_0);
                        }
                    }

                    float *cf = (float *) cache_buf[buf];
                    for (int64_t i = 0; i < kbn; ++i) {
                        pack_b_f32_transpose(cf + i * (SCP_PANEL_SIZE / sizeof(float)),
                                             src0_batch, mb, (kb0 + i) * TILE_K, nb1_0);
                    }
                    FENCE;
                    flush_to_l2_multi(cache_buf[buf], kbn * TILE_K, 64);
                    WAIT_CACHEOPS;

                    wid++;
                    scp_signal(ready_ctr, wid);
                }
            }
            FENCE;
            return 0;
        }

        // ----- Hart 0: consumer -----
        setup_cache_scp();
#if CACHEOP_MAX > 0 || REP_RATE > 0
        ucache_control(1, REP_RATE, CACHEOP_MAX);
#endif
        CLEAR_TENSOR_ERROR;

        et_barrier(ET_BARRIER_MINION);
        evict_to_l2((const void *)(uintptr_t) ready_ctr, 1, 64);    WAIT_CACHEOPS;
        evict_to_l2((const void *)(uintptr_t) consumed_ctr, 1, 64); WAIT_CACHEOPS;

        const uint64_t group_base_global = get_minion_id() - (uint64_t) k_split;

        uint32_t wid = 0;
        for (int64_t unit = my_start; unit < base_units; unit += tiles_stride) {
            const int64_t batch_idx = unit / units_pb;
            const int64_t unit_in_b = unit % units_pb;
            const int64_t g_idx     = unit_in_b / m_tiles;
            const int64_t mb_idx    = unit_in_b % m_tiles;

            const int64_t i3 = batch_idx / ne2_1;
            const int64_t i2 = batch_idx % ne2_1;

            const char *src1_batch = src1_base + i3 * nb3_1 + i2 * nb2_1;
            char       *dst_batch  = dst_base  + i3 * nb3_d + i2 * nb2_d;

            const int64_t mb        = mb_idx * TILE_M;
            const int64_t nb_base_t = g_idx * ru_n;
            int64_t r_count = n_tiles - nb_base_t;
            if (r_count > ru_n) r_count = ru_n;

            for (int64_t kw = 0; kw < n_windows; ++kw) {
                const int buf = wid & 1;
                wid++;
                scp_wait(ready_ctr, wid);

                const int64_t kb0 = k_start_block + kw * KWIN;
                const int64_t kbn = (kb0 + KWIN <= k_end_block) ? KWIN : (k_end_block - kb0);
                const int is_last = (kw == n_windows - 1);
                float *cf = (float *) cache_buf[buf];

                for (int64_t r = 0; r < r_count; ++r) {
                    const int64_t nb = (nb_base_t + r) * TILE_N;
                    const int64_t n_cur = (nb + TILE_N <= N) ? TILE_N : (N - nb);
                    const int64_t arows_fma = (n_cur == 4) ? 4 : (n_cur - 1);

                    char *cs = cscratch + r * (16 * 64);

                    int first;
                    if (kw == 0) {
                        first = 1;
                    } else {
                        c_seed(cs);
                        first = 0;
                    }

                    const int64_t nsteps = kbn;
#define A_TILE_ADDR(st)                                                                  \
    ((uint64_t)(src1_batch + nb * nb1_1 +                                                \
        (kb0 + (st)) * TILE_K * (int64_t) sizeof(float)))

                    // prologue: prefetch step 0 and 1, and load step 0
                    l2_prefetch((const void *) A_TILE_ADDR(0), n_cur, nb1_1);
                    if (1 < nsteps) {
                        l2_prefetch((const void *) A_TILE_ADDR(1), n_cur, nb1_1);
                    }
                    tensor_load(false, false, A_L1_START, TENSOR_LOAD_PLAIN, 0,
                                A_TILE_ADDR(0), 0, n_cur - 1, (uint64_t) nb1_1, 0);

                    for (int64_t s = 0; s < nsteps; ++s) {
                        const uint64_t a_cur = (s & 1) ? A_L1_ALT : A_L1_START;

                        tensor_wait(TENSOR_LOAD_WAIT_0);

                        if (s + 1 < nsteps) {
                            const uint64_t a_nxt = ((s + 1) & 1) ? A_L1_ALT : A_L1_START;
                            tensor_load(false, false, a_nxt, TENSOR_LOAD_PLAIN, 0,
                                        A_TILE_ADDR(s + 1), 0, n_cur - 1, (uint64_t) nb1_1, 0);
                        }

                        if (s + 2 < nsteps) {
                            l2_prefetch((const void *) A_TILE_ADDR(s + 2), n_cur, nb1_1);
                        }

                        tensor_load_setup_b(
                            false,
                            (uint64_t)(cf + s * (SCP_PANEL_SIZE / sizeof(float))),
                            TILE_K - 1, 64, 1);

                        tensor_wait(TENSOR_LOAD_WAIT_1);

                        tensor_fma(
                            false, 3, arows_fma, TILE_K - 1, 0,
                            false, false, false, true,
                            16, a_cur, TENSOR_FMA_OP_FP32, first);
                        tensor_wait(TENSOR_FMA_WAIT);
                        first = 0;
                    }
#undef A_TILE_ADDR

                    if (is_last) {
                        if (k_splits > 1) {
                            const uint64_t num_regs = (uint64_t) n_cur * 2;
                            if (k_split > 0) {
                                tensor_reduce_recv(0, TENSOR_REDUCE_OP_FADD, num_regs,
                                                   group_base_global + (uint64_t)(k_split - 1));
                                tensor_wait(TENSOR_REDUCE_WAIT);
                            }
                            if (k_split < k_splits - 1) {
                                tensor_reduce_send(0, num_regs,
                                                   group_base_global + (uint64_t)(k_split + 1));
                                tensor_wait(TENSOR_REDUCE_WAIT);
                            }
                        }
                        if (k_split == k_splits - 1) {
                            tensor_store(
                                0, 0, 3, n_cur - 1,
                                (uint64_t)(dst_batch + nb * nb1_d + mb * (int64_t) sizeof(float)),
                                0, (uint64_t) nb1_d);
                            tensor_wait(TENSOR_STORE_WAIT);
                        }
                    } else {
                        c_spill(cs);
                    }
                }
                scp_signal(consumed_ctr, wid);
            }
        }
        FENCE;
        return 0;
    }

    // =====================================================================
    // ORIGINAL path: one output tile at a time (no reuse).
    // =====================================================================
    const int64_t base_tiles = m_tiles * n_tiles * batch_count;
    float *scp_panel[2] = {
        (float *) et_shire_l2scp_local(scp_base),
        (float *) et_shire_l2scp_local(scp_base + SCP_PANEL_SIZE),
    };

    if (is_hart1) {
        scp_signal(ready_ctr, 0);
        scp_signal(consumed_ctr, 0);

        et_barrier(ET_BARRIER_MINION);
        uint32_t chunk_id = 0;

        for (int64_t tile = my_start; tile < base_tiles; tile += tiles_stride) {
            const int64_t tiles_per_batch = m_tiles * n_tiles;
            const int64_t batch_idx       = tile / tiles_per_batch;
            const int64_t tile_in_batch   = tile % tiles_per_batch;
            const int64_t mb_idx          = tile_in_batch % m_tiles;

            const int64_t i3   = batch_idx / ne2_1;
            const int64_t i2   = batch_idx % ne2_1;
            const int64_t i2_0 = i2 / r2;
            const int64_t i3_0 = i3 / r3;

            const char *src0_batch = src0_base + i3_0 * nb3_0 + i2_0 * nb2_0;
            const int64_t mb = mb_idx * TILE_M;

            // Prefetch first block
            l2_prefetch(src0_batch + mb * nb1_0 + k_start * sizeof(float), 16, nb1_0);

            for (int64_t kb = k_start; kb < k_end; kb += TILE_K) {
                int buf = chunk_id & 1;
                if (chunk_id >= 2) scp_wait(consumed_ctr, chunk_id - 1);

                // Prefetch next block
                if (kb + TILE_K < k_end) {
                    l2_prefetch(src0_batch + mb * nb1_0 + (kb + TILE_K) * sizeof(float), 16, nb1_0);
                }

                pack_b_f32_transpose(scp_panel[buf], src0_batch, mb, kb, nb1_0);

                FENCE;
                flush_to_l2(scp_panel[buf], 16, 64);
                WAIT_CACHEOPS;

                chunk_id++;
                scp_signal(ready_ctr, chunk_id);
            }
        }
        FENCE;
        return 0;
    }

    setup_cache_scp();
#if CACHEOP_MAX > 0 || REP_RATE > 0
    ucache_control(1, REP_RATE, CACHEOP_MAX);
#endif
    CLEAR_TENSOR_ERROR;

    et_barrier(ET_BARRIER_MINION);
    evict_to_l2((const void *)(uintptr_t) ready_ctr, 1, 64);    WAIT_CACHEOPS;
    evict_to_l2((const void *)(uintptr_t) consumed_ctr, 1, 64); WAIT_CACHEOPS;

    const uint64_t group_base_global = get_minion_id() - (uint64_t) k_split;
    uint32_t chunk_id = 0;

    for (int64_t tile = my_start; tile < base_tiles; tile += tiles_stride) {
        const int64_t tiles_per_batch = m_tiles * n_tiles;
        const int64_t batch_idx       = tile / tiles_per_batch;
        const int64_t tile_in_batch   = tile % tiles_per_batch;
        const int64_t nb_idx          = tile_in_batch / m_tiles;
        const int64_t mb_idx          = tile_in_batch % m_tiles;

        const int64_t i3 = batch_idx / ne2_1;
        const int64_t i2 = batch_idx % ne2_1;

        const char *src1_batch = src1_base + i3 * nb3_1 + i2 * nb2_1;
        char       *dst_batch  = dst_base  + i3 * nb3_d + i2 * nb2_d;

        const int64_t mb = mb_idx * TILE_M;
        const int64_t nb = nb_idx * TILE_N;
        const int64_t n_cur = (nb + TILE_N <= N) ? TILE_N : (N - nb);

        int first = 1;
        // Prologue: prefetch first two blocks, load first block
        l2_prefetch((const void *)(src1_batch + nb * nb1_1 + k_start * sizeof(float)), n_cur, nb1_1);
        if (k_start + TILE_K < k_end) {
            l2_prefetch((const void *)(src1_batch + nb * nb1_1 + (k_start + TILE_K) * sizeof(float)), n_cur, nb1_1);
        }
        tensor_load(
            false, false, A_L1_START, TENSOR_LOAD_PLAIN, 0,
            (uint64_t)(src1_batch + nb * nb1_1 + k_start * (int64_t) sizeof(float)),
            0, n_cur - 1, (uint64_t) nb1_1, 0);

        for (int64_t kb = k_start; kb < k_end; kb += TILE_K) {
            int buf = chunk_id & 1;
            const uint64_t a_cur = ((kb - k_start) / TILE_K & 1) ? A_L1_ALT : A_L1_START;

            tensor_wait(TENSOR_LOAD_WAIT_0);

            if (kb + TILE_K < k_end) {
                const uint64_t a_nxt = (((kb + TILE_K) - k_start) / TILE_K & 1) ? A_L1_ALT : A_L1_START;
                tensor_load(
                    false, false, a_nxt, TENSOR_LOAD_PLAIN, 0,
                    (uint64_t)(src1_batch + nb * nb1_1 + (kb + TILE_K) * (int64_t) sizeof(float)),
                    0, n_cur - 1, (uint64_t) nb1_1, 0);
            }

            if (kb + 2 * TILE_K < k_end) {
                l2_prefetch((const void *)(src1_batch + nb * nb1_1 + (kb + 2 * TILE_K) * sizeof(float)), n_cur, nb1_1);
            }

            chunk_id++;
            scp_wait(ready_ctr, chunk_id);

            tensor_load_setup_b(
                false,
                (uint64_t) scp_panel[buf],
                15, 64, 1);
            tensor_wait(TENSOR_LOAD_WAIT_1);

            tensor_fma(
                false, 3, n_cur - 1, TILE_K - 1, 0,
                false, false, false, true,
                16, a_cur, TENSOR_FMA_OP_FP32, first);
            tensor_wait(TENSOR_FMA_WAIT);
            first = 0;

            scp_signal(consumed_ctr, chunk_id);
        }

        // K-split ring reduce
        if (k_splits > 1) {
            const uint64_t num_regs = (uint64_t) n_cur * 2;

            if (k_split > 0) {
                tensor_reduce_recv(0, TENSOR_REDUCE_OP_FADD, num_regs, group_base_global + k_split - 1);
                tensor_wait(TENSOR_REDUCE_WAIT);
            }

            if (k_split < k_splits - 1) {
                tensor_reduce_send(0, num_regs, group_base_global + k_split + 1);
                tensor_wait(TENSOR_REDUCE_WAIT);
            }
        }

        if (k_split == k_splits - 1) {
            tensor_store(
                0, 0, 3, n_cur - 1,
                (uint64_t)(dst_batch + nb * nb1_d + mb * (int64_t) sizeof(float)),
                0, (uint64_t) nb1_d);
            tensor_wait(TENSOR_STORE_WAIT);
        }
    }

    FENCE;
    return 0;
}
