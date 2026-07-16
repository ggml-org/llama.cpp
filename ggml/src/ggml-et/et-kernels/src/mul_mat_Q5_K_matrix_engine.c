//******************************************************************************
// MUL_MAT Kernel
// Matrix multiplication: C[M,N] = A[M,K] * B[K,N]
//******************************************************************************

#include <etsoc/common/utils.h>
#include <stdint.h>
#include "ggml_tensor.h"
#include "platform.h"
#include "tensor.h"
#include "quants.h"
#include "math_fp.h"

// Q5_K x F32 -> F32 MUL_MAT on the tensor (matrix) engine, TensorFMA32.
// Identical producer/consumer, tiling and tensor-engine loop to
// mul_mat_Q4_K_matrix_engine.c; only the weight dequant differs (Q5_K uses a
// 6-bit quant with a per-16-element int8 scale and no min term).
// Hart 1: dequantize Q5_K weights to FP32 into double-buffered L2 SCP.
// Hart 0: tensor engine compute (FMA, reduce, store).
//
// Two execution paths (selected at runtime by N % TILE_N):
//   * REUSE path  (N % TILE_N == 0): dequantize each weight K-window ONCE and
//     reuse it across ru_n consecutive N-tiles, so the (producer-bound)
//     dequant work is cut by ~ru_n. Partial C is round-tripped through an
//     L2-SCP scratch between K-windows (the FMA C accumulator is a single fixed
//     register-file tile, so multiple output tiles cannot be resident at once).
//   * ORIGINAL path (N % TILE_N != 0): one output tile at a time, no reuse.

#define NUM_COMPUTE_SHIRES 32
#define MINIONS_PER_SHIRE  32

#define TILE_M  16
#define TILE_N  16
#define BLOCK_K 32      // one Q5_K group (32 elements) per panel
#define FMA_K   16      // tensor FMA k-width for FP32 (a_num_cols = FMA_K-1)

// --- Reuse knobs ----------------------------------------------------------
// REUSE_MAX caps the L2-SCP C-scratch footprint; the actual reuse factor is
// chosen at runtime as the largest value that still keeps the whole
// machine busy. KWIN is the dequant-cache depth (K-blocks per window).
#ifndef REUSE_MAX
#define REUSE_MAX 15
#endif
#ifndef KWIN
#define KWIN    16      // K-blocks per dequant window (cache depth)
#endif

#define MACHINE_SLOTS (NUM_COMPUTE_SHIRES * MINIONS_PER_SHIRE)  // 1024

#define CACHEOP_MAX 0
#define REP_RATE    0

#define A_L1_START 0    // L1 SCP lines  0..15 for A (activations)
#define B_L1_START 16   // L1 SCP lines 16..31 for B (dequantized weights)

// Single dequant panel: BLOCK_K k-lines x TILE_M m (FP32) = 32*64 = 2048 bytes,
// [k][m] order: panel[k*TILE_M + m].
#define SCP_PANEL_SIZE   (BLOCK_K * TILE_M * (uint64_t)sizeof(float))  // 2048

// L2 SCP layout per minion. The REUSE path needs the larger footprint, so the
// per-minion stride uses it for both paths (mutually exclusive at runtime).
//   [0 .. RU_BUF_BYTES)            cache buffer 0 (KWIN panels)
//   [RU_BUF_BYTES .. 2*..)         cache buffer 1 (KWIN panels)
//   [RU_CACHE_BYTES .. +R*1024)    REUSE_MAX C-scratch tiles (16 rows*64B each)
//   ready_ctr, consumed_ctr        sync counters
// The ORIGINAL path reuses [0,2048) and [2048,4096) as its two panels and the
// same ready/consumed counters (which sit above the cache region).
#define RU_BUF_BYTES     (KWIN * SCP_PANEL_SIZE)
#define RU_CACHE_BYTES   (2 * RU_BUF_BYTES)
#define RU_CSCRATCH_BYTES (REUSE_MAX * 16 * 64ULL)
#define SCP_READY_OFF    (RU_CACHE_BYTES + RU_CSCRATCH_BYTES)
#define SCP_CONSUMED_OFF (SCP_READY_OFF + 64)
#define SCP_PER_MINION   (SCP_CONSUMED_OFF + 64)

// Software fp16->fp32 (pure integer). The hardware fcvt.ps.f16 returns wrong
// values after the attention block (shared conversion-unit state), which would
// corrupt the weight scales here; software conversion avoids that instruction
// entirely. Only the super-block d is fp16, so cost is negligible.
static inline float __attribute__((always_inline)) me_sw_fp16(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) { f = sign; }
        else {
            exp = 127 - 15 + 1;
            while ((mant & 0x400) == 0) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            f = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float out; __builtin_memcpy(&out, &f, 4); return out;
}

// EXPERIMENT: use the hardware fcvt.ps.f16 (fp16_to_fp32, math_fp.h) for the
// super-block scale instead of the software me_sw_fp16 above. Hardware fcvt was
// observed to corrupt Q4_K scales after the attention block; trying it here for
// the other K-quants. If it produces garbage, flip ME_FP16 back to me_sw_fp16.
// #define ME_FP16(h) fp16_to_fp32(h)  // hardware fcvt: garbage for Q3/Q5/Q6 (fcvt-after-attention bug)
#define ME_FP16(h) me_sw_fp16(h)

// Dequantize one 32-element Q5_K GROUP of TILE_M weight rows into the FP32 panel,
// written in TenB [k][m] order: panel[k*TILE_M + m].
//
// A Q5_K super-block has the same affine layout as Q4_K (8 groups of 32) with an
// extra high bit per weight drawn from qh:
//     w = d*sc*((nibble) + (qh_bit ? 16 : 0)) - dmin*m
// kb_group is the global 32-element group index: super-block = kb_group/8,
// group g = kb_group%8 -> qs pair p = g/2, low/high nibble ab = g%1, qh bit
// 1<<(2p+ab) (see dequantize_q5_K_block).
static inline void __attribute__((always_inline))
dequant_q5_K_panel(float *panel, const char *src0_batch,
                   int64_t mb, int64_t kb_group, int64_t nb1_0) {
    const int64_t sb    = kb_group >> 3;
    const int     g     = (int) (kb_group & 7);
    const int     p     = g >> 1;             // qs pair (0..3), 32-byte block
    const int     ab    = g & 1;              // 0 low nibble, 1 high nibble
    const int     qloff = p * 32;
    const uint8_t ubit  = (uint8_t) (1u << (2 * p + ab));

    for (int j = 0; j < TILE_M; ++j) {
        const block_q5_K *blk =
            (const block_q5_K *)(src0_batch + (mb + j) * nb1_0) + sb;
        const float          d  = ME_FP16(blk->d);
        const float          dm = ME_FP16(blk->dmin);
        const uint8_t      * ql = blk->qs + qloff;
        const uint8_t      * qh = blk->qh;
        uint8_t sc, mm;
        get_scale_min_k4(g, blk->scales, &sc, &mm);
        const float dl = d  * (float) sc;
        const float ml = dm * (float) mm;

        for (int l = 0; l < 32; ++l) {
            const uint8_t nib = ab ? (ql[l] >> 4) : (ql[l] & 0xF);
            const int     hi5 = (qh[l] & ubit) ? 16 : 0;
            panel[l * TILE_M + j] = dl * (float) (nib + hi5) - ml;
        }
    }
}

// Spill / seed the FP32 C accumulator (16x16 tile in the vector register file,
// row n -> f2n[cols 0..7], f2n+1[cols 8..15]) to/from a 1 KB L2-SCP scratch.
// scratch layout: row n at byte offset n*64. Always moves all 16 rows; rows
// beyond a partial n_cur carry harmless garbage (never stored / recomputed).
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

int entry_point(struct ggml_et_binary_params *params, void *env) {
    (void) env;

    uint64_t hart_id  = get_hart_id();
    uint64_t shire_id = get_shire_id();

    if (shire_id >= NUM_COMPUTE_SHIRES) return 0;

    const int is_hart1 = hart_id & 1;
    uint64_t local_minion = (hart_id >> 1) & 0x1F;

    // Dimensions (both harts need these for tile assignment)
    const int64_t K = params->src0.ne[0];
    const int64_t M = params->src0.ne[1];
    const int64_t N = params->src1.ne[1];

    if ((M % TILE_M) != 0)  return 0;
    if ((K % QK_K) != 0)    return 0;

    const int64_t ne2_0 = params->src0.ne[2], ne3_0 = params->src0.ne[3];
    const int64_t ne2_1 = params->src1.ne[2], ne3_1 = params->src1.ne[3];

    const int64_t nb1_0 = params->src0.nb[1];
    const int64_t nb2_0 = params->src0.nb[2], nb3_0 = params->src0.nb[3];

    const int64_t nb1_1 = params->src1.nb[1];
    const int64_t nb2_1 = params->src1.nb[2], nb3_1 = params->src1.nb[3];

    const int64_t nb1_d = params->dst.nb[1];
    const int64_t nb2_d = params->dst.nb[2], nb3_d = params->dst.nb[3];

    const char *src0_base = (const char *) params->src0.data;
    const char *src1_base = (const char *) params->src1.data;
    char       *dst_base  = (char *) params->dst.data;

    const int64_t m_tiles = M / TILE_M;
    const int64_t n_tiles = (N + TILE_N - 1) / TILE_N;
    const int64_t batch_count = ne2_1 * ne3_1;

    const int64_t r2 = ne2_1 / ne2_0;
    const int64_t r3 = ne3_1 / ne3_0;

    const int64_t k_steps = K / BLOCK_K;        // number of 32-element groups

    const int64_t tiles_per_shire = MINIONS_PER_SHIRE;
    const int64_t local_tile_idx  = local_minion;
    const int64_t tiles_stride    = (int64_t) NUM_COMPUTE_SHIRES * tiles_per_shire;
    const int64_t my_start        = (int64_t) shire_id + local_tile_idx * NUM_COMPUTE_SHIRES;

    // L2 SCP pointers for this minion.
    const uint64_t scp_base = local_minion * SCP_PER_MINION;
    volatile uint32_t *ready_ctr =
        (volatile uint32_t *) et_shire_l2scp_local(scp_base + SCP_READY_OFF);
    volatile uint32_t *consumed_ctr =
        (volatile uint32_t *) et_shire_l2scp_local(scp_base + SCP_CONSUMED_OFF);

    // Calculate ru_n to perfectly minimize hardware waves while avoiding Consumer bottleneck.
    // The pipeline is perfectly balanced at r=8. Score = waves * max(8, r).
    // We find the r that minimizes Score.
    int64_t best_r = 1;
    int64_t min_score = INT64_MAX;
    int64_t max_search_r = REUSE_MAX;
    if (max_search_r > n_tiles) max_search_r = n_tiles;

    for (int64_t r = 1; r <= max_search_r; r++) {
        int64_t n_groups = (n_tiles + r - 1) / r;
        int64_t base_units = m_tiles * n_groups * batch_count;
        int64_t waves = (base_units + MACHINE_SLOTS - 1) / MACHINE_SLOTS;

        int64_t penalty = (r > 8) ? r : 8;
        int64_t score = waves * penalty;

        if (score < min_score) {
            min_score = score;
            best_r = r;
        }
    }
    int64_t ru_n = best_r;

    // Reuse pays only when it groups >=2 N-tiles; otherwise the windowing /
    // C round-trip is pure overhead, so use the one-tile-at-a-time path.
    const int reuse_ok = (ru_n >= 2);

    // =====================================================================
    // REUSE path: dequant each K-window once, reuse across ru_n N-tiles.
    // =====================================================================
    if (reuse_ok) {
        char *cache_buf[2] = {
            (char *) et_shire_l2scp_local(scp_base),
            (char *) et_shire_l2scp_local(scp_base + RU_BUF_BYTES),
        };
        char *cscratch = (char *) et_shire_l2scp_local(scp_base + RU_CACHE_BYTES);

        const int64_t n_groups   = (n_tiles + ru_n - 1) / ru_n;
        const int64_t units_pb   = m_tiles * n_groups;
        const int64_t base_units = units_pb * batch_count;
        const int64_t n_windows  = (k_steps + KWIN - 1) / KWIN;

        // ----- Hart 1: producer -----
        if (is_hart1) {
            scp_signal(ready_ctr, 0);
            scp_signal(consumed_ctr, 0);
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

                for (int64_t kw = 0; kw < n_windows; ++kw) {
                    const int buf = wid & 1;
                    if (wid >= 2) scp_wait(consumed_ctr, wid - 1);

                    const int64_t kb0 = kw * KWIN;
                    const int64_t kbn = (kb0 + KWIN <= k_steps) ? KWIN : (k_steps - kb0);

                    float *cf = (float *) cache_buf[buf];
                    for (int64_t i = 0; i < kbn; ++i) {
                        dequant_q5_K_panel(cf + i * (SCP_PANEL_SIZE / 4),
                                           src0_batch, mb, kb0 + i, nb1_0);
                    }
                    FENCE;
                    flush_to_l2(cache_buf[buf], kbn * BLOCK_K, 64);
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
        evict_to_l2((const void *) ready_ctr, 1, 64);    WAIT_CACHEOPS;
        evict_to_l2((const void *) consumed_ctr, 1, 64); WAIT_CACHEOPS;

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
            const int64_t nb_base_t = g_idx * ru_n;                    // first N-tile
            int64_t r_count = n_tiles - nb_base_t;
            if (r_count > ru_n) r_count = ru_n;

            for (int64_t kw = 0; kw < n_windows; ++kw) {
                const int buf = wid & 1;
                wid++;
                scp_wait(ready_ctr, wid);

                const int64_t kb0 = kw * KWIN;
                const int64_t kbn = (kb0 + KWIN <= k_steps) ? KWIN : (k_steps - kb0);
                const int is_last = (kw == n_windows - 1);
                float *cf = (float *) cache_buf[buf];

                for (int64_t r = 0; r < r_count; ++r) {
                    const int64_t nb = (nb_base_t + r) * TILE_N;
                    const int64_t n_cur = (nb + TILE_N <= N) ? TILE_N : (N - nb);
                    const int64_t arows_fma = (n_cur == 4) ? 4 : (n_cur - 1);

                    char *cs = cscratch + r * (16 * 64);

                    if (kw > 0) c_seed(cs);
                    int first = (kw == 0) ? 1 : 0;

                    if (n_cur == 4) {
                        static const float __attribute__((aligned(64))) zero_line[16] = {0};
                        tensor_load(false, false, A_L1_START + 4, TENSOR_LOAD_PLAIN, 0,
                                    (uint64_t) zero_line, 0, 0, 64, 0);
                        tensor_wait(TENSOR_LOAD_WAIT_0);
                    }

                    for (int64_t i = 0; i < kbn; ++i) {
                        for (int half = 0; half < 2; ++half) {
                            const int64_t k_elem = (kb0 + i) * BLOCK_K + half * FMA_K;
                            tensor_load(
                                false, false, A_L1_START, TENSOR_LOAD_PLAIN, 0,
                                (uint64_t)(src1_batch + nb * nb1_1 + k_elem * (int64_t) sizeof(float)),
                                0, n_cur - 1, (uint64_t) nb1_1, 0);
                            tensor_wait(TENSOR_LOAD_WAIT_0);

                            tensor_load_setup_b(
                                false,
                                (uint64_t)(cf + i * (SCP_PANEL_SIZE / 4) + half * FMA_K * TILE_M),
                                FMA_K - 1, 64, 1);

                            tensor_fma(
                                false, 3, arows_fma, FMA_K - 1, 0,
                                false, false, false, true,
                                B_L1_START, A_L1_START, TENSOR_FMA_OP_FP32, first);
                            tensor_wait(TENSOR_FMA_WAIT);
                            first = 0;
                        }
                    }

                    if (is_last) {
                        tensor_store(
                            0, 0, 3, n_cur - 1,
                            (uint64_t)(dst_batch + nb * nb1_d + mb * (int64_t) sizeof(float)),
                            0, (uint64_t) nb1_d);
                        tensor_wait(TENSOR_STORE_WAIT);
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
    // ORIGINAL path: one output tile at a time (N % TILE_N != 0). No reuse.
    // =====================================================================
    const int64_t base_tiles = m_tiles * n_tiles * batch_count;
    float *scp_panel[2] = {
        (float *) et_shire_l2scp_local(scp_base),
        (float *) et_shire_l2scp_local(scp_base + SCP_PANEL_SIZE),
    };

    if (is_hart1) {
        scp_signal(ready_ctr, 0);
        scp_signal(consumed_ctr, 0);
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

            for (int64_t kb = 0; kb < k_steps; ++kb) {
                int buf = chunk_id & 1;
                if (chunk_id >= 2) scp_wait(consumed_ctr, chunk_id - 1);

                dequant_q5_K_panel(scp_panel[buf], src0_batch, mb, kb, nb1_0);

                FENCE;
                flush_to_l2(scp_panel[buf], BLOCK_K, 64);
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
    evict_to_l2((const void *) ready_ctr, 1, 64);    WAIT_CACHEOPS;
    evict_to_l2((const void *) consumed_ctr, 1, 64); WAIT_CACHEOPS;

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
        const int64_t arows_fma = (n_cur == 4) ? 4 : (n_cur - 1);

        if (n_cur == 4) {
            static const float __attribute__((aligned(64))) zero_line[16] = {0};
            tensor_load(false, false, A_L1_START + 4, TENSOR_LOAD_PLAIN, 0,
                        (uint64_t) zero_line, 0, 0, 64, 0);
            tensor_wait(TENSOR_LOAD_WAIT_0);
        }

        int first = 1;
        for (int64_t kb = 0; kb < k_steps; ++kb) {
            int buf = chunk_id & 1;
            chunk_id++;
            scp_wait(ready_ctr, chunk_id);

            for (int half = 0; half < 2; ++half) {
                const int64_t k_elem = kb * BLOCK_K + half * FMA_K;
                tensor_load(
                    false, false, A_L1_START, TENSOR_LOAD_PLAIN, 0,
                    (uint64_t)(src1_batch + nb * nb1_1 + k_elem * (int64_t) sizeof(float)),
                    0, n_cur - 1, (uint64_t) nb1_1, 0);
                tensor_wait(TENSOR_LOAD_WAIT_0);

                tensor_load_setup_b(
                    false,
                    (uint64_t)(scp_panel[buf] + (int64_t) half * FMA_K * TILE_M),
                    FMA_K - 1, 64, 1);

                tensor_fma(
                    false, 3, arows_fma, FMA_K - 1, 0,
                    false, false, false, true,
                    B_L1_START, A_L1_START, TENSOR_FMA_OP_FP32, first);
                tensor_wait(TENSOR_FMA_WAIT);
                first = 0;
            }

            scp_signal(consumed_ctr, chunk_id);
        }

        tensor_store(
            0, 0, 3, n_cur - 1,
            (uint64_t)(dst_batch + nb * nb1_d + mb * (int64_t) sizeof(float)),
            0, (uint64_t) nb1_d);
        tensor_wait(TENSOR_STORE_WAIT);
    }

    FENCE;
    return 0;
}
