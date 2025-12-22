#define GGML_COMMON_DECL_CPP
#include "ggml-ifairy-lut-impl.h"
#include "ggml-common.h"
#include "ggml-quants.h"
#include "ggml-impl.h"
#include "ggml-backend.h"

#ifndef GGML_FP16_TO_FP32
#define GGML_FP16_TO_FP32 ggml_fp16_to_fp32
#endif

#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <atomic>

static std::atomic<bool> g_ifairy_lut_warned_bad_layout(false);
static std::atomic<bool> g_ifairy_lut_warned_bad_kernel(false);

static inline size_t ggml_ifairy_checked_mul_size(size_t a, size_t b) {
    GGML_ASSERT(a == 0 || b <= SIZE_MAX / a);
    return a * b;
}

static inline size_t ggml_ifairy_checked_add_size(size_t a, size_t b) {
    GGML_ASSERT(a <= SIZE_MAX - b);
    return a + b;
}

ggml_ifairy_lut_layout ggml_ifairy_lut_layout_from_env(int n) {
    const char * env = getenv("GGML_IFAIRY_LUT_LAYOUT");
    if (env) {
        if (strcmp(env, "legacy") == 0) {
            return GGML_IFAIRY_LUT_LAYOUT_LEGACY;
        }
        if (strcmp(env, "compact") == 0) {
            return GGML_IFAIRY_LUT_LAYOUT_COMPACT;
        }
        if (strcmp(env, "auto") != 0) {
            if (ggml_ifairy_env_enabled("GGML_IFAIRY_LUT_DEBUG") && !g_ifairy_lut_warned_bad_layout.exchange(true)) {
                GGML_LOG_WARN("ifairy_lut: unknown GGML_IFAIRY_LUT_LAYOUT='%s' (expected: legacy|compact|auto), using default\n", env);
            }
        }
        // "auto" or unknown -> default policy below
    }

    // Default policy: prefer the legacy layout to avoid performance regressions.
    // The compact layout can be enabled explicitly via GGML_IFAIRY_LUT_LAYOUT=compact.
    (void) n;
    return GGML_IFAIRY_LUT_LAYOUT_LEGACY;
}

ggml_ifairy_lut_kernel ggml_ifairy_lut_kernel_from_env(void) {
    static std::atomic<int> cached(-1); // -1=unset, else ggml_ifairy_lut_kernel
    int v = cached.load(std::memory_order_relaxed);
    if (v >= 0) {
        return (ggml_ifairy_lut_kernel) v;
    }

    const char * env = getenv("GGML_IFAIRY_LUT_KERNEL");
    if (env) {
        if (strcmp(env, "auto") == 0) {
            v = GGML_IFAIRY_LUT_KERNEL_AUTO;
        } else if (strcmp(env, "sdot") == 0) {
            v = GGML_IFAIRY_LUT_KERNEL_SDOT;
        } else if (strcmp(env, "tbl") == 0) {
            v = GGML_IFAIRY_LUT_KERNEL_TBL;
        } else if (strcmp(env, "merged64") == 0) {
            v = GGML_IFAIRY_LUT_KERNEL_MERGED64;
        } else {
            v = GGML_IFAIRY_LUT_KERNEL_AUTO;
            if (ggml_ifairy_env_enabled("GGML_IFAIRY_LUT_DEBUG") && !g_ifairy_lut_warned_bad_kernel.exchange(true)) {
                GGML_LOG_WARN("ifairy_lut: unknown GGML_IFAIRY_LUT_KERNEL='%s' (expected: auto|sdot|tbl|merged64), using default\n", env);
            }
        }
    } else {
        v = GGML_IFAIRY_LUT_KERNEL_AUTO;
    }

    cached.store(v, std::memory_order_relaxed);
    return (ggml_ifairy_lut_kernel) v;
}

static inline uint64_t ggml_ifairy_pack_u8_8(
        uint8_t b0, uint8_t b1, uint8_t b2, uint8_t b3,
        uint8_t b4, uint8_t b5, uint8_t b6, uint8_t b7) {
    return ((uint64_t) b0) |
           ((uint64_t) b1 << 8) |
           ((uint64_t) b2 << 16) |
           ((uint64_t) b3 << 24) |
           ((uint64_t) b4 << 32) |
           ((uint64_t) b5 << 40) |
           ((uint64_t) b6 << 48) |
           ((uint64_t) b7 << 56);
}

bool ggml_ifairy_lut_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst) {
    const bool dbg = ggml_ifairy_env_enabled("GGML_IFAIRY_LUT_DEBUG");
    const char * enabled_env = getenv("GGML_IFAIRY_LUT");
    if (enabled_env && strcmp(enabled_env, "0") == 0) {
        if (dbg) { GGML_LOG_WARN("ifairy_lut: disabled by env GGML_IFAIRY_LUT=0\n"); }
        return false;
    }

#if !defined(__ARM_NEON) || !defined(__aarch64__)
    if (dbg) { GGML_LOG_WARN("ifairy_lut: disabled (requires __aarch64__ + __ARM_NEON)\n"); }
    return false;
#endif

    if (src0->type != GGML_TYPE_IFAIRY || (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_IFAIRY_Q16)) {
        if (dbg) {
            GGML_LOG_WARN("ifairy_lut: type mismatch src0=%s src1=%s dst=%s\n",
                          ggml_type_name(src0->type), ggml_type_name(src1->type), ggml_type_name(dst->type));
        }
        return false;
    }
    if (dst->type != GGML_TYPE_F32) {
        if (dbg) {
            GGML_LOG_WARN("ifairy_lut: dst type not F32 (%s)\n", ggml_type_name(dst->type));
        }
        return false;
    }
    // require logical K aligned to block
    if (src0->ne[0] % QK_K != 0 || src1->ne[0] != src0->ne[0]) {
        if (dbg) {
            GGML_LOG_WARN("ifairy_lut: K misaligned K0=%lld K1=%lld QK_K=%d\n",
                          (long long) src0->ne[0], (long long) src1->ne[0], QK_K);
        }
        return false;
    }
    if (dbg) { GGML_LOG_INFO("ifairy_lut: can_mul_mat=true\n"); }
    return true;
}

size_t ggml_ifairy_lut_get_wsize(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst, int n_threads) {
    if (!ggml_ifairy_lut_can_mul_mat(src0, src1, dst)) {
        return 0;
    }
    GGML_ASSERT(n_threads > 0);
    const bool strict = ggml_ifairy_env_enabled("GGML_IFAIRY_LUT_VALIDATE_STRICT");

    const int64_t M = src0->ne[1];
    const int64_t K = src0->ne[0];
    const int64_t N = src1->ne[1];
    const int64_t blocks_per_col = K / QK_K;
    const int64_t groups_per_block = (QK_K + 2) / 3;

    GGML_ASSERT(M >= 0);
    GGML_ASSERT(K >= 0);
    GGML_ASSERT(N >= 0);
    GGML_ASSERT(blocks_per_col >= 0);
    GGML_ASSERT(groups_per_block > 0);

    size_t quant_bytes = 0;
    if (src1->type == GGML_TYPE_F32) {
        const size_t q_elems = ggml_ifairy_checked_mul_size((size_t) N, (size_t) blocks_per_col);
        quant_bytes = GGML_PAD(ggml_ifairy_checked_mul_size(q_elems, sizeof(block_ifairy_q16)), 64);
    }

    // Optional BK tiling (by whole 256-element blocks) to reduce LUT working set.
    // Disabled under strict validation (strict currently assumes full-K single-pass).
    int tile_blocks = 0;
    if (!strict) {
        tile_blocks = ggml_ifairy_env_get_int_nonzero("GGML_IFAIRY_LUT_BK_BLOCKS", 0);
    }
    if (tile_blocks < 0) {
        if (ggml_ifairy_env_enabled("GGML_IFAIRY_LUT_DEBUG")) {
            GGML_LOG_WARN("ifairy_lut: GGML_IFAIRY_LUT_BK_BLOCKS < 0, clamping to 0\n");
        }
        tile_blocks = 0;
    }

    int bm = 64;
    if (tile_blocks > 0) {
        bm = ggml_ifairy_env_get_int_nonzero("GGML_IFAIRY_LUT_BM", 64);
        if (bm < 1) {
            if (ggml_ifairy_env_enabled("GGML_IFAIRY_LUT_DEBUG")) {
                GGML_LOG_WARN("ifairy_lut: GGML_IFAIRY_LUT_BM < 1, clamping to 1\n");
            }
            bm = 1;
        }
    }

    const int64_t blocks_tile = tile_blocks > 0 ? MIN((int64_t) tile_blocks, blocks_per_col) : blocks_per_col;
    const int64_t groups_tile = blocks_tile * groups_per_block;

    GGML_ASSERT(blocks_tile >= 0);
    GGML_ASSERT(groups_tile >= 0);

    const ggml_ifairy_lut_layout layout = ggml_ifairy_lut_layout_from_env((int) N);
    const size_t lut_groups = ggml_ifairy_checked_mul_size((size_t) N, (size_t) groups_tile);
    const size_t lut_bytes = layout == GGML_IFAIRY_LUT_LAYOUT_LEGACY
            ? ggml_ifairy_checked_mul_size(
                    ggml_ifairy_checked_mul_size(lut_groups, (size_t) (k_ifairy_lut_channels * k_ifairy_lut_patterns)),
                    sizeof(int16_t))
            : ggml_ifairy_checked_mul_size(lut_groups, (size_t) k_ifairy_lut_group_bytes);
    // activation scales are per-block (shared by all groups in the block)
    const size_t scale_bytes = ggml_ifairy_checked_mul_size(
            ggml_ifairy_checked_mul_size(ggml_ifairy_checked_mul_size((size_t) N, (size_t) blocks_tile), 2u),
            sizeof(float));

    size_t shared_bytes = GGML_PAD(ggml_ifairy_checked_add_size(lut_bytes, scale_bytes), 64);

    // Optional "full accumulator" mode (tiled only):
    // for small N, keep a single shared accumulator of size M*(4*N) floats so we can
    // preprocess each K-tile only once (instead of once per BM row-block), reducing barriers.
    bool fullacc = false;
    if (tile_blocks > 0 && M > 0) {
        const char * env = getenv("GGML_IFAIRY_LUT_FULLACC");
        const size_t acc_elems = ggml_ifairy_checked_mul_size((size_t) M, ggml_ifairy_checked_mul_size(4u, (size_t) N));
        const size_t acc_bytes = ggml_ifairy_checked_mul_size(acc_elems, sizeof(float));
        const bool auto_ok = (N <= 2) && (acc_bytes <= (size_t) (8 * 1024 * 1024));
        if (env && strcmp(env, "0") == 0) {
            fullacc = false;
        } else if (env && strcmp(env, "0") != 0) {
            fullacc = true;
        } else {
            fullacc = auto_ok;
        }
        if (fullacc) {
            shared_bytes = ggml_ifairy_checked_add_size(shared_bytes, GGML_PAD(acc_bytes, 64));
        }
    }

    // tmp buffer:
    // - non-tiled: N floats (bf16-pair packed into F32)
    // - tiled+BM:  BM*(4*N) floats accumulator (ac/ad/bc/bd), then combined and packed
    // - tiled+fullacc: minimal per-thread scratch
    const size_t tmp_elems = tile_blocks == 0
            ? (size_t) N
            : (fullacc ? 16u : ggml_ifairy_checked_mul_size((size_t) bm, ggml_ifairy_checked_mul_size(4u, (size_t) N)));
    const size_t tmp_bytes = GGML_PAD(ggml_ifairy_checked_mul_size(tmp_elems, sizeof(float)), 64);

    return ggml_ifairy_checked_add_size(
            ggml_ifairy_checked_add_size(quant_bytes, shared_bytes),
            ggml_ifairy_checked_mul_size(tmp_bytes, (size_t) n_threads));
}
