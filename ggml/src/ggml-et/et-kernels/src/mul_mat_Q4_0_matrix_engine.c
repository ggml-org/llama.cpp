#include <etsoc/common/utils.h>
#include <stdint.h>
#include "ggml_tensor.h"
#include "platform.h"
#include "tensor.h"
#include "quants.h"
#include "math_fp.h"

// Q4_0 x F32 -> F32 MUL_MAT on the tensor (matrix) engine, TensorFMA32.
//
// Weights (src0, Q4_0) are dequantized to FP32 *directly into TenB layout* and
// fed to the FP32 tensor FMA against FP32 activations (src1). No activation
// quantization. v1: correctness-first, scalar dequant.
//
// Why two harts: the TensorFMA32 accumulator lives in hart 0's FP/vector
// register file across the whole K reduction (first_pass=1 then accumulate).
// Dequant's scale multiply is a scalar/vector FP op that uses those same
// registers, so hart 0 cannot dequant between its accumulating FMAs without
// corrupting the accumulator. The harts have separate register files, so:
//   Hart 1: dequantize the next Q4_0 weight panel into double-buffered L2 SCP.
//   Hart 0: tensor engine only (load A, load B from SCP, FMA, reduce, store).
// Sync: monotonic counters in L2 SCP with evict-based coherency (mirrors
// mul_mat_f16_matrix_engine.c).
//
// Fused dequant+transpose: while decoding each Q4_0 block, each FP32 value is
// scattered straight into TenB's [k][m] order (line = k, within-line = m), so a
// plain tensor_load (no TRANSPOSE32) brings it into L1 already as B.

#define NUM_COMPUTE_SHIRES 32
#define MINIONS_PER_SHIRE  32

#define TILE_M  16
#define TILE_N  16
#define BLOCK_K QK4_0   // 32 elements per Q4_0 block
#define FMA_K   16      // tensor FMA k-width for FP32 (a_num_cols = FMA_K-1)

#define CACHEOP_MAX 0
#define REP_RATE    0

#define A_L1_START 0    // L1 SCP lines  0..15 for A (activations)
#define B_L1_START 16   // L1 SCP lines 16..31 for B (dequantized weights)

// L2 SCP layout per minion (double-buffered dequant panel + sync counters).
// panel = BLOCK_K k-lines x TILE_M m (FP32) = 32 * 64 = 2048 bytes, in TenB
// [k][m] order: panel[k*TILE_M + m].
#define SCP_PANEL_SIZE   (BLOCK_K * TILE_M * (uint64_t)sizeof(float))  // 2048
#define SCP_READY_OFF    (2 * SCP_PANEL_SIZE)                          // 4096
#define SCP_CONSUMED_OFF (SCP_READY_OFF + 64)                          // 4160
#define SCP_PER_MINION   (SCP_CONSUMED_OFF + 64)                       // 4224

// Signal a counter value to the other hart via L2 SCP.
static inline void __attribute__((always_inline))
scp_signal(volatile uint32_t *flag, uint32_t value) {
    *flag = value;
    FENCE;
    evict_to_l2((const void *)flag, 1, 64);
    WAIT_CACHEOPS;
}

// Wait for a counter in L2 SCP to reach the expected value.
static inline void __attribute__((always_inline))
scp_wait(volatile uint32_t *flag, uint32_t expected) {
    while (1) {
        evict_to_l2((const void *)flag, 1, 64);
        WAIT_CACHEOPS;
        if (*flag >= expected) return;
    }
}

// Dequantize one 32-element Q4_0 block of TILE_M weight rows into the FP32
// panel, written directly in TenB [k][m] order: panel[k*TILE_M + m].
//   Low  nibble of byte i -> k = i
//   High nibble of byte i -> k = i + 16
//   value = d * (nibble - 8)
static inline void __attribute__((always_inline))
dequant_q4_0_panel(float *panel, const char *src0_batch,
                   int64_t mb, int64_t kb_block, int64_t nb1_0) {
    for (int j = 0; j < TILE_M; ++j) {
        const block_q4_0 *blk =
            (const block_q4_0 *)(src0_batch + (mb + j) * nb1_0) + kb_block;
        const float d = fp16_to_fp32(blk->d);
        for (int i = 0; i < QK4_0 / 2; ++i) {       // 16 packed bytes
            const uint8_t b = blk->qs[i];
            panel[(i)      * TILE_M + j] = d * (float)((int)(b & 0xF) - 8);
            panel[(i + 16) * TILE_M + j] = d * (float)((int)(b >> 4)  - 8);
        }
    }
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
    if ((K % BLOCK_K) != 0) return 0;

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
    const int64_t base_tiles = m_tiles * n_tiles * batch_count;

    const int64_t r2 = ne2_1 / ne2_0;
    const int64_t r3 = ne3_1 / ne3_0;

    const int64_t k_steps = K / BLOCK_K;        // number of Q4_0 blocks

    // v1: force a single K-split. The multi-minion K-split + ring-reduce path
    // has a known correctness bug on partial-N tiles; it is disabled until that
    // is fixed separately. Each minion computes the full K reduction for its
    // own output tiles.
    const int64_t k_splits = 1;

    const int64_t tiles_per_shire = MINIONS_PER_SHIRE / k_splits;
    const int64_t k_split = local_minion % k_splits;
    const int64_t local_tile_idx = local_minion / k_splits;
    const int64_t tiles_stride = (int64_t) NUM_COMPUTE_SHIRES * tiles_per_shire;

    const int64_t k_steps_per_split = k_steps / k_splits;
    const int64_t kb_start = k_split * k_steps_per_split;       // first block
    const int64_t kb_end   = kb_start + k_steps_per_split;      // one past last

    // L2 SCP pointers for this minion's double-buffered panels + sync.
    uint64_t scp_base = local_minion * SCP_PER_MINION;
    float *scp_panel[2] = {
        (float *) et_shire_l2scp_local(scp_base),
        (float *) et_shire_l2scp_local(scp_base + SCP_PANEL_SIZE),
    };
    volatile uint32_t *ready_ctr =
        (volatile uint32_t *) et_shire_l2scp_local(scp_base + SCP_READY_OFF);
    volatile uint32_t *consumed_ctr =
        (volatile uint32_t *) et_shire_l2scp_local(scp_base + SCP_CONSUMED_OFF);

    // ================================================================
    // Hart 1: Q4_0 weight dequant producer
    // ================================================================
    if (is_hart1) {
        scp_signal(ready_ctr, 0);
        scp_signal(consumed_ctr, 0);

        uint32_t chunk_id = 0;

        for (int64_t tile = (int64_t) shire_id + local_tile_idx * NUM_COMPUTE_SHIRES;
             tile < base_tiles;
             tile += tiles_stride) {

            const int64_t tiles_per_batch = m_tiles * n_tiles;
            const int64_t batch_idx       = tile / tiles_per_batch;
            const int64_t tile_in_batch   = tile % tiles_per_batch;

            const int64_t mb_idx = tile_in_batch % m_tiles;

            const int64_t i3   = batch_idx / ne2_1;
            const int64_t i2   = batch_idx % ne2_1;
            const int64_t i2_0 = i2 / r2;
            const int64_t i3_0 = i3 / r3;

            const char *src0_batch = src0_base + i3_0 * nb3_0 + i2_0 * nb2_0;
            const int64_t mb = mb_idx * TILE_M;

            for (int64_t kb = kb_start; kb < kb_end; ++kb) {
                int buf = chunk_id & 1;

                // Back-pressure: wait for hart 0 to finish with this buffer.
                if (chunk_id >= 2) {
                    scp_wait(consumed_ctr, chunk_id - 1);
                }

                dequant_q4_0_panel(scp_panel[buf], src0_batch, mb, kb, nb1_0);

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

    // ================================================================
    // Hart 0: tensor engine compute
    // ================================================================
    uint64_t my_minion_id = get_minion_id();
    const uint64_t group_base_global = my_minion_id - k_split;

    setup_cache_scp();
#if CACHEOP_MAX > 0 || REP_RATE > 0
    ucache_control(1, REP_RATE, CACHEOP_MAX);
#endif
    CLEAR_TENSOR_ERROR;

    evict_to_l2((const void *) ready_ctr, 1, 64);
    WAIT_CACHEOPS;
    evict_to_l2((const void *) consumed_ctr, 1, 64);
    WAIT_CACHEOPS;

    uint32_t chunk_id = 0;

    for (int64_t tile = (int64_t) shire_id + local_tile_idx * NUM_COMPUTE_SHIRES;
         tile < base_tiles;
         tile += tiles_stride) {

        const int64_t tiles_per_batch = m_tiles * n_tiles;
        const int64_t batch_idx       = tile / tiles_per_batch;
        const int64_t tile_in_batch   = tile % tiles_per_batch;

        const int64_t nb_idx = tile_in_batch / m_tiles;
        const int64_t mb_idx = tile_in_batch % m_tiles;

        const int64_t i3 = batch_idx / ne2_1;
        const int64_t i2 = batch_idx % ne2_1;

        const char *src1_batch = src1_base + i3 * nb3_1 + i2 * nb2_1;
        char       *dst_batch  = dst_base  + i3 * nb3_d + i2 * nb2_d;

        const int64_t mb = mb_idx * TILE_M;
        const int64_t nb = nb_idx * TILE_N;
        const int64_t n_cur = (nb + TILE_N <= N) ? TILE_N : (N - nb);

        // Partial-N is handled by a_num_rows = n_cur-1 alone (as in the F32
        // matrix engine); no tensor_mask needed for the plain FP32 layout.
        int first = 1;  // first_pass=1 only for the very first FMA of the tile

        for (int64_t kb = kb_start; kb < kb_end; ++kb) {
            int buf = chunk_id & 1;

            // Wait for hart 1 to finish dequantizing this block.
            chunk_id++;
            scp_wait(ready_ctr, chunk_id);

            // Two FMA passes over the 32-wide block (16 K-cols each).
            for (int half = 0; half < 2; ++half) {
                const int64_t k_elem = kb * BLOCK_K + half * FMA_K;

                // Load A (activations) for this 16-K sub-tile, PLAIN.
                tensor_load(
                    false, false,
                    A_L1_START,
                    TENSOR_LOAD_PLAIN,
                    0,
                    (uint64_t)(src1_batch + nb * nb1_1 + k_elem * (int64_t) sizeof(float)),
                    0,
                    n_cur - 1,
                    (uint64_t) nb1_1,
                    0
                );

                // Load B (dequantized weights) half from L2 SCP panel, PLAIN.
                tensor_load(
                    false, false,
                    B_L1_START,
                    TENSOR_LOAD_PLAIN,
                    0,
                    (uint64_t)(scp_panel[buf] + (int64_t) half * FMA_K * TILE_M),
                    0,
                    FMA_K - 1,
                    64,
                    1
                );

                tensor_wait(TENSOR_LOAD_WAIT_0);
                tensor_wait(TENSOR_LOAD_WAIT_1);

                tensor_fma(
                    false,
                    3,              // b_num_col: (16/4)-1
                    n_cur - 1,      // a_num_rows
                    FMA_K - 1,      // a_num_cols
                    0,
                    false,
                    false,
                    false,
                    false,
                    B_L1_START,
                    A_L1_START,
                    TENSOR_FMA_OP_FP32,
                    first
                );

                tensor_wait(TENSOR_FMA_WAIT);
                first = 0;
            }

            // Signal that this buffer is free for hart 1 to reuse.
            scp_signal(consumed_ctr, chunk_id);
        }

        // K-split ring reduce.
        if (k_splits > 1) {
            const uint64_t num_regs = (uint64_t) n_cur * 2;

            if (k_split > 0) {
                tensor_reduce_recv(
                    0, TENSOR_REDUCE_OP_FADD,
                    num_regs,
                    group_base_global + k_split - 1
                );
                tensor_wait(TENSOR_REDUCE_WAIT);
            }

            if (k_split < k_splits - 1) {
                tensor_reduce_send(
                    0, num_regs,
                    group_base_global + k_split + 1
                );
                tensor_wait(TENSOR_REDUCE_WAIT);
            }
        }

        // Store FP32 result tile (only the last k-split owns the final sum).
        if (k_split == k_splits - 1) {
            tensor_store(
                0, 0, 3, n_cur - 1,
                (uint64_t)(dst_batch + nb * nb1_d + mb * (int64_t) sizeof(float)),
                0, (uint64_t) nb1_d
            );
            tensor_wait(TENSOR_STORE_WAIT);
        }
    }

    FENCE;
    return 0;
}
