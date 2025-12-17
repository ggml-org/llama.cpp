#define GGML_COMMON_DECL_CPP
#include "ggml-ifairy-lut.h"
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
#include <vector>
#include <unordered_map>
#include <mutex>

#if defined(__ARM_NEON) && defined(__aarch64__)
#include <arm_neon.h>
#endif

static std::vector<ifairy_lut_extra *> g_ifairy_lut_extras;
static std::mutex g_ifairy_lut_mutex;

struct ifairy_lut_index_cache_key {
    const void * data;
    size_t       nbytes;
    int64_t      k;
    int64_t      rows;

    bool operator==(const ifairy_lut_index_cache_key & other) const noexcept {
        return data == other.data && nbytes == other.nbytes && k == other.k && rows == other.rows;
    }
};

struct ifairy_lut_index_cache_key_hash {
    size_t operator()(const ifairy_lut_index_cache_key & key) const noexcept {
        size_t h = std::hash<uintptr_t>{}((uintptr_t) key.data);
        h ^= std::hash<size_t>{}(key.nbytes) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= std::hash<int64_t>{}(key.k)      + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= std::hash<int64_t>{}(key.rows)   + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return h;
    }
};

struct ifairy_lut_index_cache_entry {
    ggml_backend_buffer_t buffer = nullptr;
    uint8_t *             base   = nullptr;
    size_t                size   = 0;
};

static std::unordered_map<ifairy_lut_index_cache_key, ifairy_lut_index_cache_entry, ifairy_lut_index_cache_key_hash> g_ifairy_lut_index_cache;

static inline bool ggml_ifairy_env_enabled(const char * name) {
    const char * env = getenv(name);
    return env && strcmp(env, "0") != 0;
}

// Prefetch is enabled by default; set GGML_IFAIRY_LUT_PREFETCH=0 to disable for tuning.
static inline bool ggml_ifairy_lut_prefetch_enabled(void) {
    const char * env = getenv("GGML_IFAIRY_LUT_PREFETCH");
    return !(env && strcmp(env, "0") == 0);
}

static inline size_t ggml_ifairy_checked_mul_size(size_t a, size_t b) {
    GGML_ASSERT(a == 0 || b <= SIZE_MAX / a);
    return a * b;
}

static inline size_t ggml_ifairy_checked_add_size(size_t a, size_t b) {
    GGML_ASSERT(a <= SIZE_MAX - b);
    return a + b;
}

enum ggml_ifairy_lut_layout {
    GGML_IFAIRY_LUT_LAYOUT_LEGACY  = 0, // 4x64 int16 per-group tables
    GGML_IFAIRY_LUT_LAYOUT_COMPACT = 1, // int8 per-position tables (3 positions × 4 codes × 4 channels)
};

static ggml_ifairy_lut_layout ggml_ifairy_lut_layout_from_env(int n) {
    const char * env = getenv("GGML_IFAIRY_LUT_LAYOUT");
    if (env) {
        if (strcmp(env, "legacy") == 0) {
            return GGML_IFAIRY_LUT_LAYOUT_LEGACY;
        }
        if (strcmp(env, "compact") == 0) {
            return GGML_IFAIRY_LUT_LAYOUT_COMPACT;
        }
        // "auto" or unknown -> default policy below
    }

    // Default policy: prefer the legacy layout to avoid performance regressions.
    // The compact layout can be enabled explicitly via GGML_IFAIRY_LUT_LAYOUT=compact.
    (void) n;
    return GGML_IFAIRY_LUT_LAYOUT_LEGACY;
}

static const int k_ifairy_lut_patterns = 64; // legacy table size
static const int k_ifairy_lut_codes     = 4;
static const int k_ifairy_lut_channels  = 4;

static const size_t k_ifairy_lut_pos_bytes   = (size_t) k_ifairy_lut_codes    * (size_t) k_ifairy_lut_channels;  // 16
// Compact layout per-group payload is 3 * 16B = 48B.
static const size_t k_ifairy_lut_group_bytes = GGML_IFAIRY_LUT_COMPACT_GROUP_BYTES;
static_assert(k_ifairy_lut_group_bytes >= 3 * k_ifairy_lut_pos_bytes, "compact LUT group size must fit 3 position tables");

#if defined(__ARM_NEON) && defined(__aarch64__)
// wr(code) / wi(code) coefficients for all 64 3-weight patterns (direct 6-bit encoding).
// code -> (wr, wi): 0 -> (-1,0), 1 -> (1,0), 2 -> (0,-1), 3 -> (0,1)
static const int8_t k_ifairy_wr0[64] = { -1, 1, 0, 0, -1, 1, 0, 0, -1, 1, 0, 0, -1, 1, 0, 0, -1, 1, 0, 0, -1, 1, 0, 0, -1, 1, 0, 0, -1, 1, 0, 0, -1, 1, 0, 0, -1, 1, 0, 0, -1, 1, 0, 0, -1, 1, 0, 0, -1, 1, 0, 0, -1, 1, 0, 0, -1, 1, 0, 0, -1, 1, 0, 0 };
static const int8_t k_ifairy_wr1[64] = { -1, -1, -1, -1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0 };
static const int8_t k_ifairy_wr2[64] = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
static const int8_t k_ifairy_wi0[64] = { 0, 0, -1, 1, 0, 0, -1, 1, 0, 0, -1, 1, 0, 0, -1, 1, 0, 0, -1, 1, 0, 0, -1, 1, 0, 0, -1, 1, 0, 0, -1, 1, 0, 0, -1, 1, 0, 0, -1, 1, 0, 0, -1, 1, 0, 0, -1, 1, 0, 0, -1, 1, 0, 0, -1, 1, 0, 0, -1, 1, 0, 0, -1, 1 };
static const int8_t k_ifairy_wi1[64] = { 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 1, 1, 1, 1 };
static const int8_t k_ifairy_wi2[64] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
#endif

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

// iFairy 3-weight LUT implementation (CPU backend).
// Integrated into ggml mul_mat routing under GGML_IFAIRY_ARM_LUT.

void ggml_ifairy_lut_init(void) {
    // No global initialization needed yet.
}

void ggml_ifairy_lut_free(void) {
    // free any extras allocated by transform_tensor
    std::lock_guard<std::mutex> lock(g_ifairy_lut_mutex);
    for (auto & it : g_ifairy_lut_index_cache) {
        if (it.second.buffer) {
            ggml_backend_buffer_free(it.second.buffer);
        }
    }
    g_ifairy_lut_index_cache.clear();
    for (auto * e : g_ifairy_lut_extras) {
        if (e) {
            if (e->indexes && e->index_tensor == NULL && e->index_buffer == NULL) {
                ggml_aligned_free(e->indexes, e->size);
            }
            delete e;
        }
    }
    g_ifairy_lut_extras.clear();
}

bool ggml_ifairy_lut_can_mul_mat(const struct ggml_tensor * src0, const struct ggml_tensor * src1, const struct ggml_tensor * dst) {
    const bool dbg = ggml_ifairy_env_enabled("GGML_IFAIRY_LUT_DEBUG");
    const char * enabled_env = getenv("GGML_IFAIRY_LUT");
    if (enabled_env && strcmp(enabled_env, "0") == 0) {
        if (dbg) { GGML_LOG_WARN("ifairy_lut: disabled by env GGML_IFAIRY_LUT=0\n"); }
        return false;
    }

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
        const char * env = getenv("GGML_IFAIRY_LUT_BK_BLOCKS");
        if (env && strcmp(env, "0") != 0) {
            tile_blocks = (int) strtol(env, NULL, 10);
        }
    }

    int bm = 64;
    if (tile_blocks > 0) {
        const char * env = getenv("GGML_IFAIRY_LUT_BM");
        if (env && strcmp(env, "0") != 0) {
            bm = (int) strtol(env, NULL, 10);
        }
        if (bm < 1) {
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

bool ggml_ifairy_lut_transform_tensor(struct ggml_tensor * tensor, struct ggml_tensor ** index_tensor_out) {
    if (!tensor || tensor->type != GGML_TYPE_IFAIRY) {
        if (index_tensor_out) {
            *index_tensor_out = NULL;
        }
        return false;
    }

    ifairy_lut_extra * extra = (ifairy_lut_extra *) tensor->extra;
    if (extra && extra->indexes) {
        if (index_tensor_out) {
            *index_tensor_out = NULL;
        }
        return true;
    }

    const int64_t k = tensor->ne[0];
    const int64_t rows = tensor->ne[1];
    if (k % QK_K != 0 || rows <= 0) {
        return false;
    }

    const struct ggml_ifairy_3w_index_info info = ggml_ifairy_3w_get_index_info(k);
    const size_t index_bytes = ggml_ifairy_3w_index_buffer_size(&info, rows);

    const ifairy_lut_index_cache_key key = {
        /* .data   = */ tensor->data,
        /* .nbytes = */ ggml_nbytes(tensor),
        /* .k      = */ k,
        /* .rows   = */ rows,
    };

    {
        std::lock_guard<std::mutex> lock(g_ifairy_lut_mutex);
        const auto it = g_ifairy_lut_index_cache.find(key);
        if (it != g_ifairy_lut_index_cache.end() && it->second.base && it->second.size == index_bytes) {
            const bool need_push = (extra == nullptr);
            if (!extra) {
                extra = new ifairy_lut_extra;
                tensor->extra = extra;
            }

            extra->indexes       = it->second.base;
            extra->size          = it->second.size;
            extra->index_tensor  = NULL;
            extra->index_buffer  = it->second.buffer;

            if (need_push) {
                g_ifairy_lut_extras.push_back(extra);
            }

            if (index_tensor_out) {
                *index_tensor_out = NULL;
            }
            return true;
        }
    }

    ggml_backend_buffer_t index_buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), index_bytes);
    uint8_t * buf = index_buffer ? (uint8_t *) ggml_backend_buffer_get_base(index_buffer) : nullptr;
    if (index_buffer) {
        ggml_backend_buffer_set_usage(index_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }

    // fallback (shouldn't normally happen, but keeps LUT usable under allocation failures)
    if (!buf) {
        if (index_buffer) {
            ggml_backend_buffer_free(index_buffer);
            index_buffer = nullptr;
        }
        buf = (uint8_t *) ggml_aligned_malloc(index_bytes);
        if (!buf) {
            return false;
        }
    }

    const bool ok = ggml_ifairy_3w_encode((const block_ifairy *) tensor->data, k, rows, buf, index_bytes);
    if (!ok) {
        if (index_buffer) {
            ggml_backend_buffer_free(index_buffer);
        } else {
            ggml_aligned_free(buf, index_bytes);
        }
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(g_ifairy_lut_mutex);
        if (index_buffer) {
            const auto it = g_ifairy_lut_index_cache.find(key);
            if (it == g_ifairy_lut_index_cache.end()) {
                g_ifairy_lut_index_cache.emplace(key, ifairy_lut_index_cache_entry{
                    /* .buffer = */ index_buffer,
                    /* .base   = */ buf,
                    /* .size   = */ index_bytes,
                });
            } else {
                // Another thread may have populated the cache meanwhile; reuse it and free ours.
                ggml_backend_buffer_free(index_buffer);
                index_buffer = it->second.buffer;
                buf          = it->second.base;
            }
        }

        const bool need_push = (extra == nullptr);
        if (!extra) {
            extra = new ifairy_lut_extra;
            tensor->extra = extra;
        }

        extra->indexes       = buf;
        extra->size          = index_bytes;
        extra->index_tensor  = NULL;
        extra->index_buffer  = index_buffer;

        if (need_push) {
            g_ifairy_lut_extras.push_back(extra);
        }
    }

    if (index_tensor_out) {
        *index_tensor_out = NULL;
    }
    return true;
}

static void ggml_ifairy_lut_preprocess_legacy(int m, int k, int n, const void * act, size_t act_stride, void * lut_scales, void * lut_buf) {
    (void) m; // rows unused in preprocess (per-column)
    if (!act || !lut_scales || !lut_buf) {
        return;
    }

    const int64_t K  = k;
    const int64_t blocks = K / QK_K;
    const int64_t groups_per_block = (QK_K + 2) / 3;
    const int64_t groups = blocks * groups_per_block;

    for (int col = 0; col < n; ++col) {
        const uint8_t * act_col_bytes = (const uint8_t *) act + (size_t) col * act_stride;
        const block_ifairy_q16 * act_blocks = (const block_ifairy_q16 *) act_col_bytes;
        float * scales_out = (float *) lut_scales + (size_t) col * (size_t) blocks * 2;

        // Layout: per-group, 64 patterns, interleaved 4 channels:
        //   tbl[(pat*4) + 0..3] = { sum_ac, sum_ad, sum_bc, sum_bd } (int16)
        int16_t * lut_out = (int16_t *) ((uint8_t *) lut_buf + (size_t) col * (size_t) groups * (size_t) (k_ifairy_lut_channels * k_ifairy_lut_patterns) * sizeof(int16_t));

        // per-block activation scales (shared by all groups in the block)
        for (int64_t blk = 0; blk < blocks; ++blk) {
            scales_out[blk * 2 + 0] = GGML_FP16_TO_FP32(act_blocks[blk].d_real);
            scales_out[blk * 2 + 1] = GGML_FP16_TO_FP32(act_blocks[blk].d_imag);
        }

        for (int64_t g = 0; g < groups; ++g) {
            const int64_t blk   = g / groups_per_block;
            const int64_t intra = g - blk * groups_per_block;

            const bool tail = intra == groups_per_block - 1;
            const int64_t base_off = tail ? (QK_K - 1) : intra * 3;
            const int64_t idx0 = blk * QK_K + base_off + 0;

            const int blk0 = (int) blk;
            const int off0 = (int) base_off;
            const int blk1 = (int) blk;
            const int blk2 = (int) blk;
            const int off1 = (int) (base_off + 1);
            const int off2 = (int) (base_off + 2);

            int xr0 = 0, xi0 = 0;
            int xr1 = 0, xi1 = 0;
            int xr2 = 0, xi2 = 0;

            if (idx0 < K) {
                xr0 = (int8_t) act_blocks[blk0].x_real[off0];
                xi0 = (int8_t) act_blocks[blk0].x_imag[off0];
            }
            if (!tail) {
                xr1 = (int8_t) act_blocks[blk1].x_real[off1];
                xi1 = (int8_t) act_blocks[blk1].x_imag[off1];
                xr2 = (int8_t) act_blocks[blk2].x_real[off2];
                xi2 = (int8_t) act_blocks[blk2].x_imag[off2];
            }

            int16_t * tbl = lut_out + (size_t) g * (size_t) (k_ifairy_lut_patterns * k_ifairy_lut_channels);

#if defined(__ARM_NEON) && defined(__aarch64__)
            const int8_t xr0_s8 = (int8_t) xr0;
            const int8_t xr1_s8 = (int8_t) xr1;
            const int8_t xr2_s8 = (int8_t) xr2;
            const int8_t xi0_s8 = (int8_t) xi0;
            const int8_t xi1_s8 = (int8_t) xi1;
            const int8_t xi2_s8 = (int8_t) xi2;

            for (int pat = 0; pat < 64; pat += 16) {
                const int8x16_t wr0 = vld1q_s8(k_ifairy_wr0 + pat);
                const int8x16_t wr1 = vld1q_s8(k_ifairy_wr1 + pat);
                const int8x16_t wr2 = vld1q_s8(k_ifairy_wr2 + pat);
                const int8x16_t wi0 = vld1q_s8(k_ifairy_wi0 + pat);
                const int8x16_t wi1 = vld1q_s8(k_ifairy_wi1 + pat);
                const int8x16_t wi2 = vld1q_s8(k_ifairy_wi2 + pat);

                int16x8_t ac0 = vmull_s8(vget_low_s8(wr0), vdup_n_s8(xr0_s8));
                ac0 = vmlal_s8(ac0, vget_low_s8(wr1), vdup_n_s8(xr1_s8));
                ac0 = vmlal_s8(ac0, vget_low_s8(wr2), vdup_n_s8(xr2_s8));

                int16x8_t ad0 = vmull_s8(vget_low_s8(wr0), vdup_n_s8(xi0_s8));
                ad0 = vmlal_s8(ad0, vget_low_s8(wr1), vdup_n_s8(xi1_s8));
                ad0 = vmlal_s8(ad0, vget_low_s8(wr2), vdup_n_s8(xi2_s8));

                int16x8_t bc0 = vmull_s8(vget_low_s8(wi0), vdup_n_s8(xr0_s8));
                bc0 = vmlal_s8(bc0, vget_low_s8(wi1), vdup_n_s8(xr1_s8));
                bc0 = vmlal_s8(bc0, vget_low_s8(wi2), vdup_n_s8(xr2_s8));

                int16x8_t bd0 = vmull_s8(vget_low_s8(wi0), vdup_n_s8(xi0_s8));
                bd0 = vmlal_s8(bd0, vget_low_s8(wi1), vdup_n_s8(xi1_s8));
                bd0 = vmlal_s8(bd0, vget_low_s8(wi2), vdup_n_s8(xi2_s8));

                int16x8_t ac1 = vmull_s8(vget_high_s8(wr0), vdup_n_s8(xr0_s8));
                ac1 = vmlal_s8(ac1, vget_high_s8(wr1), vdup_n_s8(xr1_s8));
                ac1 = vmlal_s8(ac1, vget_high_s8(wr2), vdup_n_s8(xr2_s8));

                int16x8_t ad1 = vmull_s8(vget_high_s8(wr0), vdup_n_s8(xi0_s8));
                ad1 = vmlal_s8(ad1, vget_high_s8(wr1), vdup_n_s8(xi1_s8));
                ad1 = vmlal_s8(ad1, vget_high_s8(wr2), vdup_n_s8(xi2_s8));

                int16x8_t bc1 = vmull_s8(vget_high_s8(wi0), vdup_n_s8(xr0_s8));
                bc1 = vmlal_s8(bc1, vget_high_s8(wi1), vdup_n_s8(xr1_s8));
                bc1 = vmlal_s8(bc1, vget_high_s8(wi2), vdup_n_s8(xr2_s8));

                int16x8_t bd1 = vmull_s8(vget_high_s8(wi0), vdup_n_s8(xi0_s8));
                bd1 = vmlal_s8(bd1, vget_high_s8(wi1), vdup_n_s8(xi1_s8));
                bd1 = vmlal_s8(bd1, vget_high_s8(wi2), vdup_n_s8(xi2_s8));

                int16x8x4_t out0;
                out0.val[0] = ac0;
                out0.val[1] = ad0;
                out0.val[2] = bc0;
                out0.val[3] = bd0;
                vst4q_s16(tbl + (size_t) pat * 4, out0);

                int16x8x4_t out1;
                out1.val[0] = ac1;
                out1.val[1] = ad1;
                out1.val[2] = bc1;
                out1.val[3] = bd1;
                vst4q_s16(tbl + (size_t) (pat + 8) * 4, out1);
            }
#else
            for (int pat = 0; pat < 64; ++pat) {
                const uint8_t c0 = (uint8_t) (pat & 3);
                const uint8_t c1 = (uint8_t) ((pat >> 2) & 3);
                const uint8_t c2 = (uint8_t) ((pat >> 4) & 3);

                int wr0 = 0, wi0 = 0;
                int wr1 = 0, wi1 = 0;
                int wr2 = 0, wi2 = 0;

                switch (c0) { case 0: wr0 = -1; break; case 1: wr0 =  1; break; case 2: wi0 = -1; break; case 3: wi0 =  1; break; }
                switch (c1) { case 0: wr1 = -1; break; case 1: wr1 =  1; break; case 2: wi1 = -1; break; case 3: wi1 =  1; break; }
                switch (c2) { case 0: wr2 = -1; break; case 1: wr2 =  1; break; case 2: wi2 = -1; break; case 3: wi2 =  1; break; }

                const int sum_ac = xr0 * wr0 + xr1 * wr1 + xr2 * wr2;
                const int sum_ad = xi0 * wr0 + xi1 * wr1 + xi2 * wr2;
                const int sum_bc = xr0 * wi0 + xr1 * wi1 + xr2 * wi2;
                const int sum_bd = xi0 * wi0 + xi1 * wi1 + xi2 * wi2;

                tbl[pat * 4 + 0] = (int16_t) sum_ac;
                tbl[pat * 4 + 1] = (int16_t) sum_ad;
                tbl[pat * 4 + 2] = (int16_t) sum_bc;
                tbl[pat * 4 + 3] = (int16_t) sum_bd;
            }
#endif
        }
    }
}

void ggml_ifairy_lut_preprocess(int m, int k, int n, const void * act, size_t act_stride, void * lut_scales, void * lut_buf) {
    ggml_ifairy_lut_preprocess_ex(m, k, n, act, act_stride, lut_scales, lut_buf, 0, 1);
}

void ggml_ifairy_lut_preprocess_ex(int m, int k, int n, const void * act, size_t act_stride, void * lut_scales, void * lut_buf, int ith, int nth) {
    const ggml_ifairy_lut_layout layout = ggml_ifairy_lut_layout_from_env(n);
    if (layout == GGML_IFAIRY_LUT_LAYOUT_LEGACY) {
        if (ith == 0) {
            ggml_ifairy_lut_preprocess_legacy(m, k, n, act, act_stride, lut_scales, lut_buf);
        }
        return;
    }

    (void) m; // rows unused in preprocess (per-column)
    if (!act || !lut_scales || !lut_buf) {
        return;
    }

    if (nth < 1) {
        nth = 1;
    }
    if (ith < 0 || ith >= nth) {
        return;
    }

    const int64_t K  = k;
    const int64_t blocks = K / QK_K;
    const int64_t groups_per_block = (QK_K + 2) / 3;
    const int64_t groups = blocks * groups_per_block;

    const bool shard_by_col = n >= nth;

    const int col_start = shard_by_col ? ith : 0;
    const int col_step  = shard_by_col ? nth : 1;
    const int col_end   = n;

    for (int col = col_start; col < col_end; col += col_step) {
        const uint8_t * act_col_bytes = (const uint8_t *) act + (size_t) col * act_stride;
        const block_ifairy_q16 * act_blocks = (const block_ifairy_q16 *) act_col_bytes;
        float * scales_out = (float *) lut_scales + (size_t) col * (size_t) blocks * 2;

        // per-block activation scales (shared by all groups in the block)
        // If n < nth we use group-sharding below and only thread 0 fills scales (small, avoids false sharing).
        if (shard_by_col || ith == 0) {
            for (int64_t blk = 0; blk < blocks; ++blk) {
                scales_out[blk * 2 + 0] = GGML_FP16_TO_FP32(act_blocks[blk].d_real);
                scales_out[blk * 2 + 1] = GGML_FP16_TO_FP32(act_blocks[blk].d_imag);
            }
        }

        // Layout: per-group, 3 positions (c0/c1/c2), each position is a 16B table:
        //   tbl_pos[code*4 + 0..3] = { ac, ad, bc, bd } (int8)
        // where code -> (wr,wi):
        //   0 -> (-1,0), 1 -> (1,0), 2 -> (0,-1), 3 -> (0,1)
        int8_t * lut_out = (int8_t *) ((uint8_t *) lut_buf + (size_t) col * (size_t) groups * k_ifairy_lut_group_bytes);

        // For decode-like cases (n < nth), shard by groups in a strided fashion to avoid false sharing:
        // group size is 48B, so consecutive groups share cache lines; striding keeps threads off the same lines.
        const int64_t g0 = shard_by_col ? 0 : ith;
        const int64_t gstep = shard_by_col ? 1 : (int64_t) nth;

        for (int64_t g = g0; g < groups; g += gstep) {
            const int64_t blk   = g / groups_per_block;
            const int64_t intra = g - blk * groups_per_block;

            const bool tail = intra == groups_per_block - 1;
            const int64_t base_off = tail ? (QK_K - 1) : intra * 3;
            const int64_t idx0 = blk * QK_K + base_off + 0;

            const int blk0 = (int) blk;
            const int off0 = (int) base_off;
            const int blk1 = (int) blk;
            const int blk2 = (int) blk;
            const int off1 = (int) (base_off + 1);
            const int off2 = (int) (base_off + 2);

            int8_t xr0 = 0, xi0 = 0;
            int8_t xr1 = 0, xi1 = 0;
            int8_t xr2 = 0, xi2 = 0;

            if (idx0 < K) {
                xr0 = (int8_t) act_blocks[blk0].x_real[off0];
                xi0 = (int8_t) act_blocks[blk0].x_imag[off0];
            }
            if (!tail) {
                xr1 = (int8_t) act_blocks[blk1].x_real[off1];
                xi1 = (int8_t) act_blocks[blk1].x_imag[off1];
                xr2 = (int8_t) act_blocks[blk2].x_real[off2];
                xi2 = (int8_t) act_blocks[blk2].x_imag[off2];
            }

            int8_t * grp = lut_out + (size_t) g * k_ifairy_lut_group_bytes;
            int8_t * tbl0 = grp + 0 * k_ifairy_lut_pos_bytes;
            int8_t * tbl1 = grp + 1 * k_ifairy_lut_pos_bytes;
            int8_t * tbl2 = grp + 2 * k_ifairy_lut_pos_bytes;

            // code 0: (-1,0) -> { -xr, -xi, 0, 0 }
            // code 1: ( 1,0) -> {  xr,  xi, 0, 0 }
            // code 2: (0,-1) -> { 0, 0, -xr, -xi }
            // code 3: (0, 1) -> { 0, 0,  xr,  xi }
            const uint8_t xr0_p = (uint8_t) xr0;
            const uint8_t xi0_p = (uint8_t) xi0;
            const uint8_t xr1_p = (uint8_t) xr1;
            const uint8_t xi1_p = (uint8_t) xi1;
            const uint8_t xr2_p = (uint8_t) xr2;
            const uint8_t xi2_p = (uint8_t) xi2;

            const uint8_t xr0_n = (uint8_t) (int8_t) -xr0;
            const uint8_t xi0_n = (uint8_t) (int8_t) -xi0;
            const uint8_t xr1_n = (uint8_t) (int8_t) -xr1;
            const uint8_t xi1_n = (uint8_t) (int8_t) -xi1;
            const uint8_t xr2_n = (uint8_t) (int8_t) -xr2;
            const uint8_t xi2_n = (uint8_t) (int8_t) -xi2;

            // Position table layout (16B):
            //   [-xr,-xi,0,0,  xr,xi,0,0,  0,0,-xr,-xi,  0,0,xr,xi]
            const uint64_t tbl0_lo = ggml_ifairy_pack_u8_8(xr0_n, xi0_n, 0, 0, xr0_p, xi0_p, 0, 0);
            const uint64_t tbl0_hi = ggml_ifairy_pack_u8_8(0, 0, xr0_n, xi0_n, 0, 0, xr0_p, xi0_p);

            const uint64_t tbl1_lo = ggml_ifairy_pack_u8_8(xr1_n, xi1_n, 0, 0, xr1_p, xi1_p, 0, 0);
            const uint64_t tbl1_hi = ggml_ifairy_pack_u8_8(0, 0, xr1_n, xi1_n, 0, 0, xr1_p, xi1_p);

            const uint64_t tbl2_lo = ggml_ifairy_pack_u8_8(xr2_n, xi2_n, 0, 0, xr2_p, xi2_p, 0, 0);
            const uint64_t tbl2_hi = ggml_ifairy_pack_u8_8(0, 0, xr2_n, xi2_n, 0, 0, xr2_p, xi2_p);

#if defined(__ARM_NEON) && defined(__aarch64__)
            const uint8x16_t v0 = vcombine_u8(vcreate_u8(tbl0_lo), vcreate_u8(tbl0_hi));
            const uint8x16_t v1 = vcombine_u8(vcreate_u8(tbl1_lo), vcreate_u8(tbl1_hi));
            const uint8x16_t v2 = vcombine_u8(vcreate_u8(tbl2_lo), vcreate_u8(tbl2_hi));
            vst1q_u8((uint8_t *) tbl0, v0);
            vst1q_u8((uint8_t *) tbl1, v1);
            vst1q_u8((uint8_t *) tbl2, v2);
#else
            memcpy(tbl0 + 0, &tbl0_lo, sizeof(tbl0_lo));
            memcpy(tbl0 + 8, &tbl0_hi, sizeof(tbl0_hi));
            memcpy(tbl1 + 0, &tbl1_lo, sizeof(tbl1_lo));
            memcpy(tbl1 + 8, &tbl1_hi, sizeof(tbl1_hi));
            memcpy(tbl2 + 0, &tbl2_lo, sizeof(tbl2_lo));
            memcpy(tbl2 + 8, &tbl2_hi, sizeof(tbl2_hi));
#endif
        }
    }
}

static void ggml_ifairy_lut_qgemm_ex_legacy(int m, int k, int n, const void * qweights, const uint8_t * indexes, const void * lut, const void * lut_scales, const void * act, size_t act_stride, float * dst, size_t dst_col_stride, size_t dst_row_stride, bool pack_bf16, bool strict, bool add) {
    if (!indexes || !dst || !qweights || !lut || !lut_scales) {
        return;
    }
    if (strict) {
        GGML_ASSERT(add == false);
    }

    const bool prefetch = ggml_ifairy_lut_prefetch_enabled();
    (void) prefetch;

    const int64_t K = k;
    const int64_t blocks = K / QK_K;
    const int64_t groups_per_block = (QK_K + 2) / 3;
    const int64_t groups = blocks * groups_per_block;

    const block_ifairy * w_blocks = (const block_ifairy *) qweights;

#if 0
    // Fast-path for decode: N == 1 avoids the col loop and some pointer arithmetic.
    if (n == 1) {
        const int8_t * lut_base = (const int8_t *) lut;
        const float * scales = (const float *) lut_scales;
        const block_ifairy_q16 * act_blocks = act ? (const block_ifairy_q16 *) act : NULL;

        for (int row = 0; row < m; ++row) {
            const block_ifairy * w_row = w_blocks + (size_t) row * (size_t) blocks;
            const uint8_t * idx_row = indexes + (size_t) row * (size_t) groups;

            const float coeff_w_real = GGML_FP16_TO_FP32(w_row[0].d_real);
            const float coeff_w_imag = GGML_FP16_TO_FP32(w_row[0].d_imag);

            float acc_ac_xr = 0.0f;
            float acc_ad_xi = 0.0f;
            float acc_bc_xr = 0.0f;
            float acc_bd_xi = 0.0f;

#if defined(__ARM_NEON) && defined(__aarch64__)
            float32x4_t accv = vdupq_n_f32(0.0f); // {ac, ad, bc, bd}
            for (int64_t blk = 0; blk < blocks; ++blk) {
                int32x4_t isum0 = vdupq_n_s32(0);
                int32x4_t isum1 = vdupq_n_s32(0);

                const uint8_t * idx_g = idx_row + (size_t) blk * (size_t) groups_per_block;
                const int8_t * grp   = lut_base + (size_t) blk * (size_t) groups_per_block * k_ifairy_lut_group_bytes;

                int64_t gi = 0;
                for (; gi + 3 < groups_per_block; gi += 4) {
                    const uint8_t pat0 = (uint8_t) (idx_g[0] & 0x3f);
                    const uint8_t pat1 = (uint8_t) (idx_g[1] & 0x3f);
                    const uint8_t pat2 = (uint8_t) (idx_g[2] & 0x3f);
                    const uint8_t pat3 = (uint8_t) (idx_g[3] & 0x3f);

                    const uint8_t c00 = (uint8_t) (pat0 & 3);
                    const uint8_t c01 = (uint8_t) ((pat0 >> 2) & 3);
                    const uint8_t c02 = (uint8_t) ((pat0 >> 4) & 3);

                    const uint8_t c10 = (uint8_t) (pat1 & 3);
                    const uint8_t c11 = (uint8_t) ((pat1 >> 2) & 3);
                    const uint8_t c12 = (uint8_t) ((pat1 >> 4) & 3);

                    const uint8_t c20 = (uint8_t) (pat2 & 3);
                    const uint8_t c21 = (uint8_t) ((pat2 >> 2) & 3);
                    const uint8_t c22 = (uint8_t) ((pat2 >> 4) & 3);

                    const uint8_t c30 = (uint8_t) (pat3 & 3);
                    const uint8_t c31 = (uint8_t) ((pat3 >> 2) & 3);
                    const uint8_t c32 = (uint8_t) ((pat3 >> 4) & 3);

                    const int8_t * grp0 = grp + 0 * k_ifairy_lut_group_bytes;
                    const int8_t * grp1 = grp + 1 * k_ifairy_lut_group_bytes;
                    const int8_t * grp2 = grp + 2 * k_ifairy_lut_group_bytes;
                    const int8_t * grp3 = grp + 3 * k_ifairy_lut_group_bytes;

                    if (prefetch) {
                        __builtin_prefetch(grp0 + 4 * k_ifairy_lut_group_bytes, 0, 1);
                    }

                    const int32_t * t00 = (const int32_t *) (grp0 + 0 * k_ifairy_lut_pos_bytes);
                    const int32_t * t01 = (const int32_t *) (grp0 + 1 * k_ifairy_lut_pos_bytes);
                    const int32_t * t02 = (const int32_t *) (grp0 + 2 * k_ifairy_lut_pos_bytes);

                    const int32x2_t p00 = vld1_dup_s32(t00 + c00);
                    const int32x2_t p01 = vld1_dup_s32(t01 + c01);
                    const int32x2_t p02 = vld1_dup_s32(t02 + c02);

                    int16x8_t s160 = vmovl_s8(vreinterpret_s8_s32(p00));
                    s160 = vaddq_s16(s160, vmovl_s8(vreinterpret_s8_s32(p01)));
                    s160 = vaddq_s16(s160, vmovl_s8(vreinterpret_s8_s32(p02)));
                    isum0 = vaddw_s16(isum0, vget_low_s16(s160));

                    const int32_t * t10 = (const int32_t *) (grp1 + 0 * k_ifairy_lut_pos_bytes);
                    const int32_t * t11 = (const int32_t *) (grp1 + 1 * k_ifairy_lut_pos_bytes);
                    const int32_t * t12 = (const int32_t *) (grp1 + 2 * k_ifairy_lut_pos_bytes);

                    const int32x2_t p10 = vld1_dup_s32(t10 + c10);
                    const int32x2_t p11 = vld1_dup_s32(t11 + c11);
                    const int32x2_t p12 = vld1_dup_s32(t12 + c12);

                    int16x8_t s161 = vmovl_s8(vreinterpret_s8_s32(p10));
                    s161 = vaddq_s16(s161, vmovl_s8(vreinterpret_s8_s32(p11)));
                    s161 = vaddq_s16(s161, vmovl_s8(vreinterpret_s8_s32(p12)));
                    isum1 = vaddw_s16(isum1, vget_low_s16(s161));

                    const int32_t * t20 = (const int32_t *) (grp2 + 0 * k_ifairy_lut_pos_bytes);
                    const int32_t * t21 = (const int32_t *) (grp2 + 1 * k_ifairy_lut_pos_bytes);
                    const int32_t * t22 = (const int32_t *) (grp2 + 2 * k_ifairy_lut_pos_bytes);

                    const int32x2_t p20 = vld1_dup_s32(t20 + c20);
                    const int32x2_t p21 = vld1_dup_s32(t21 + c21);
                    const int32x2_t p22 = vld1_dup_s32(t22 + c22);

                    int16x8_t s162 = vmovl_s8(vreinterpret_s8_s32(p20));
                    s162 = vaddq_s16(s162, vmovl_s8(vreinterpret_s8_s32(p21)));
                    s162 = vaddq_s16(s162, vmovl_s8(vreinterpret_s8_s32(p22)));
                    isum0 = vaddw_s16(isum0, vget_low_s16(s162));

                    const int32_t * t30 = (const int32_t *) (grp3 + 0 * k_ifairy_lut_pos_bytes);
                    const int32_t * t31 = (const int32_t *) (grp3 + 1 * k_ifairy_lut_pos_bytes);
                    const int32_t * t32 = (const int32_t *) (grp3 + 2 * k_ifairy_lut_pos_bytes);

                    const int32x2_t p30 = vld1_dup_s32(t30 + c30);
                    const int32x2_t p31 = vld1_dup_s32(t31 + c31);
                    const int32x2_t p32 = vld1_dup_s32(t32 + c32);

                    int16x8_t s163 = vmovl_s8(vreinterpret_s8_s32(p30));
                    s163 = vaddq_s16(s163, vmovl_s8(vreinterpret_s8_s32(p31)));
                    s163 = vaddq_s16(s163, vmovl_s8(vreinterpret_s8_s32(p32)));
                    isum1 = vaddw_s16(isum1, vget_low_s16(s163));

                    idx_g += 4;
                    grp   += 4 * k_ifairy_lut_group_bytes;
                }
                for (; gi < groups_per_block; ++gi, ++idx_g, grp += k_ifairy_lut_group_bytes) {
                    const uint8_t pat = (uint8_t) (*idx_g & 0x3f);
                    const uint8_t c0 = (uint8_t) (pat & 3);
                    const uint8_t c1 = (uint8_t) ((pat >> 2) & 3);
                    const uint8_t c2 = (uint8_t) ((pat >> 4) & 3);

                    if (prefetch) {
                        __builtin_prefetch(grp + k_ifairy_lut_group_bytes, 0, 1);
                    }

                    const int32_t * t0 = (const int32_t *) (grp + 0 * k_ifairy_lut_pos_bytes);
                    const int32_t * t1 = (const int32_t *) (grp + 1 * k_ifairy_lut_pos_bytes);
                    const int32_t * t2 = (const int32_t *) (grp + 2 * k_ifairy_lut_pos_bytes);

                    const int32x2_t p0 = vld1_dup_s32(t0 + c0);
                    const int32x2_t p1 = vld1_dup_s32(t1 + c1);
                    const int32x2_t p2 = vld1_dup_s32(t2 + c2);

                    int16x8_t s16 = vmovl_s8(vreinterpret_s8_s32(p0));
                    s16 = vaddq_s16(s16, vmovl_s8(vreinterpret_s8_s32(p1)));
                    s16 = vaddq_s16(s16, vmovl_s8(vreinterpret_s8_s32(p2)));
                    isum0 = vaddw_s16(isum0, vget_low_s16(s16));
                }

                const float32x2_t srsi = vld1_f32(scales + (size_t) blk * 2);
                const float32x4_t scv = vcombine_f32(srsi, srsi); // {sr, si, sr, si}
                const float32x4_t sumsf = vcvtq_f32_s32(vaddq_s32(isum0, isum1));
                accv = vmlaq_f32(accv, sumsf, scv);
            }

            acc_ac_xr = vgetq_lane_f32(accv, 0);
            acc_ad_xi = vgetq_lane_f32(accv, 1);
            acc_bc_xr = vgetq_lane_f32(accv, 2);
            acc_bd_xi = vgetq_lane_f32(accv, 3);
#else
            for (int64_t blk = 0; blk < blocks; ++blk) {
                int32_t sum_ac = 0;
                int32_t sum_ad = 0;
                int32_t sum_bc = 0;
                int32_t sum_bd = 0;

                const uint8_t * idx_g = idx_row + (size_t) blk * (size_t) groups_per_block;
                const int8_t * grp   = lut_base + (size_t) blk * (size_t) groups_per_block * k_ifairy_lut_group_bytes;

                for (int64_t gi = 0; gi < groups_per_block; ++gi, ++idx_g, grp += k_ifairy_lut_group_bytes) {
                    const uint8_t pat = (uint8_t) (*idx_g & 0x3f);
                    const uint8_t c0 = (uint8_t) (pat & 3);
                    const uint8_t c1 = (uint8_t) ((pat >> 2) & 3);
                    const uint8_t c2 = (uint8_t) ((pat >> 4) & 3);

                    const int32_t * t0 = (const int32_t *) (grp + 0 * k_ifairy_lut_pos_bytes);
                    const int32_t * t1 = (const int32_t *) (grp + 1 * k_ifairy_lut_pos_bytes);
                    const int32_t * t2 = (const int32_t *) (grp + 2 * k_ifairy_lut_pos_bytes);

                    const int8_t * e0 = (const int8_t *) &t0[c0];
                    const int8_t * e1 = (const int8_t *) &t1[c1];
                    const int8_t * e2 = (const int8_t *) &t2[c2];

                    sum_ac += (int32_t) e0[0] + (int32_t) e1[0] + (int32_t) e2[0];
                    sum_ad += (int32_t) e0[1] + (int32_t) e1[1] + (int32_t) e2[1];
                    sum_bc += (int32_t) e0[2] + (int32_t) e1[2] + (int32_t) e2[2];
                    sum_bd += (int32_t) e0[3] + (int32_t) e1[3] + (int32_t) e2[3];
                }

                const float act_scale_r = scales[blk * 2 + 0];
                const float act_scale_i = scales[blk * 2 + 1];
                acc_ac_xr += act_scale_r * (float) sum_ac;
                acc_ad_xi += act_scale_i * (float) sum_ad;
                acc_bc_xr += act_scale_r * (float) sum_bc;
                acc_bd_xi += act_scale_i * (float) sum_bd;
            }
#endif

            const float out_r = coeff_w_real * acc_ac_xr + coeff_w_imag * acc_bd_xi;
            const float out_i = coeff_w_imag * acc_bc_xr - coeff_w_real * acc_ad_xi;

            if (!isfinite(out_r) || !isfinite(out_i)) {
                ggml_abort(__FILE__, __LINE__, "ifairy_lut_qgemm: non-finite output (row=%d col=%d acc_r=%f acc_i=%f)",
                           row, 0, out_r, out_i);
            }

            if (strict) {
                GGML_ASSERT(act_blocks != NULL);
                double ref_ac_xr = 0.0;
                double ref_ad_xi = 0.0;
                double ref_bc_xr = 0.0;
                double ref_bd_xi = 0.0;

                for (int blk = 0; blk < (int) blocks; ++blk) {
                    const uint8_t * GGML_RESTRICT w_ptr   = w_row[blk].qs;
                    const int8_t  * GGML_RESTRICT x_r_ptr = (const int8_t *) act_blocks[blk].x_real;
                    const int8_t  * GGML_RESTRICT x_i_ptr = (const int8_t *) act_blocks[blk].x_imag;

                    int32_t sum_ac = 0;
                    int32_t sum_ad = 0;
                    int32_t sum_bc = 0;
                    int32_t sum_bd = 0;

                    for (int j = 0; j < QK_K; ++j) {
                        const int chunk    = j >> 6;
                        const int lane     = j & 0xF;
                        const int part     = (j >> 4) & 0x3;
                        const int byte_idx = (chunk << 4) + lane;
                        const int bit_off  = part * 2;

                        const uint8_t packed = w_ptr[byte_idx];
                        const uint8_t code   = (packed >> bit_off) & 0x3;

                        int wr = 0;
                        int wi = 0;
                        switch (code) {
                            case 0: wr = -1; wi =  0; break;
                            case 1: wr =  1; wi =  0; break;
                            case 2: wr =  0; wi = -1; break;
                            case 3: wr =  0; wi =  1; break;
                        }

                        const int xr = (int) x_r_ptr[j];
                        const int xi = (int) x_i_ptr[j];

                        sum_ac += xr * wr;
                        sum_ad += xi * wr;
                        sum_bc += xr * wi;
                        sum_bd += xi * wi;
                    }

                    const double x_real = (double) GGML_FP16_TO_FP32(act_blocks[blk].d_real);
                    const double x_imag = (double) GGML_FP16_TO_FP32(act_blocks[blk].d_imag);

                    ref_ac_xr += x_real * (double) sum_ac;
                    ref_ad_xi += x_imag * (double) sum_ad;
                    ref_bc_xr += x_real * (double) sum_bc;
                    ref_bd_xi += x_imag * (double) sum_bd;
                }

                const double ref_r = (double) coeff_w_real * ref_ac_xr + (double) coeff_w_imag * ref_bd_xi;
                const double ref_i = (double) coeff_w_imag * ref_bc_xr - (double) coeff_w_real * ref_ad_xi;

                const float dr = out_r - (float) ref_r;
                const float di = out_i - (float) ref_i;
                GGML_ASSERT(fabsf(dr) <= 1e-3f && fabsf(di) <= 1e-3f);
            }

            uint8_t * out_base = (uint8_t *) dst + (size_t) row * dst_row_stride;
            if (pack_bf16) {
                ggml_bf16_t br = GGML_FP32_TO_BF16(out_r);
                ggml_bf16_t bi = GGML_FP32_TO_BF16(out_i);
                ((ggml_bf16_t *) out_base)[0] = br;
                ((ggml_bf16_t *) out_base)[1] = bi;
            } else {
                float * out_ptr = (float *) out_base;
                if (add) {
                    out_ptr[0] += out_r;
                    out_ptr[1] += out_i;
                } else {
                    out_ptr[0] = out_r;
                    out_ptr[1] = out_i;
                }
            }
        }
        return;
    }
#endif

    // Fast-path for decode: N == 1 avoids the col loop and some pointer arithmetic.
    // Keep strict mode on the generic path (strict validation assumes the generic structure).
    if (n == 1 && !strict) {
        const int16_t * lut_base = (const int16_t *) lut;
        const float * scales = (const float *) lut_scales;
        const block_ifairy_q16 * act_blocks = act ? (const block_ifairy_q16 *) act : NULL;

        const size_t group_stride = (size_t) (k_ifairy_lut_patterns * k_ifairy_lut_channels);

        for (int row = 0; row < m; ++row) {
            const block_ifairy * w_row = w_blocks + (size_t) row * (size_t) blocks;
            const uint8_t * idx_row = indexes + (size_t) row * (size_t) groups;

            const float coeff_w_real = GGML_FP16_TO_FP32(w_row[0].d_real);
            const float coeff_w_imag = GGML_FP16_TO_FP32(w_row[0].d_imag);

            float acc_ac_xr = 0.0f;
            float acc_ad_xi = 0.0f;
            float acc_bc_xr = 0.0f;
            float acc_bd_xi = 0.0f;

#if defined(__ARM_NEON) && defined(__aarch64__)
            float32x4_t accv = vdupq_n_f32(0.0f); // {ac, ad, bc, bd}
            for (int64_t blk = 0; blk < blocks; ++blk) {
                int32x4_t isum0 = vdupq_n_s32(0);
                int32x4_t isum1 = vdupq_n_s32(0);

                const uint8_t * idx_blk = idx_row + (size_t) blk * (size_t) groups_per_block;
                const int16_t * lut_blk = lut_base + (size_t) blk * (size_t) groups_per_block * group_stride;

                int64_t gi = 0;
                for (; gi + 3 < groups_per_block; gi += 4) {
                    const uint8_t pat0 = (uint8_t) (idx_blk[gi + 0] & 0x3f);
                    const uint8_t pat1 = (uint8_t) (idx_blk[gi + 1] & 0x3f);
                    const uint8_t pat2 = (uint8_t) (idx_blk[gi + 2] & 0x3f);
                    const uint8_t pat3 = (uint8_t) (idx_blk[gi + 3] & 0x3f);

                    const int16_t * grp0 = lut_blk + (size_t) (gi + 0) * group_stride;
                    const int16_t * grp1 = lut_blk + (size_t) (gi + 1) * group_stride;
                    const int16_t * grp2 = lut_blk + (size_t) (gi + 2) * group_stride;
                    const int16_t * grp3 = lut_blk + (size_t) (gi + 3) * group_stride;

                    if (prefetch) {
                        __builtin_prefetch(grp0 + group_stride, 0, 1);
                        __builtin_prefetch(grp1 + group_stride, 0, 1);
                    }

                    const int16_t * tbl0 = grp0 + (size_t) pat0 * k_ifairy_lut_channels;
                    const int16_t * tbl1 = grp1 + (size_t) pat1 * k_ifairy_lut_channels;
                    const int16_t * tbl2 = grp2 + (size_t) pat2 * k_ifairy_lut_channels;
                    const int16_t * tbl3 = grp3 + (size_t) pat3 * k_ifairy_lut_channels;

                    const int16x4_t s0 = vld1_s16(tbl0);
                    const int16x4_t s1 = vld1_s16(tbl1);
                    const int16x4_t s2 = vld1_s16(tbl2);
                    const int16x4_t s3 = vld1_s16(tbl3);

                    isum0 = vaddw_s16(isum0, s0);
                    isum1 = vaddw_s16(isum1, s1);
                    isum0 = vaddw_s16(isum0, s2);
                    isum1 = vaddw_s16(isum1, s3);
                }
                for (; gi < groups_per_block; ++gi) {
                    const uint8_t pat = (uint8_t) (idx_blk[gi] & 0x3f);
                    const int16_t * tbl = lut_blk + (size_t) gi * group_stride + (size_t) pat * k_ifairy_lut_channels;
                    const int16x4_t sums16 = vld1_s16(tbl);
                    isum0 = vaddw_s16(isum0, sums16);
                }

                const float32x2_t srsi = vld1_f32(scales + (size_t) blk * 2);
                const float32x4_t scv = vcombine_f32(srsi, srsi); // {sr, si, sr, si}
                const float32x4_t sumsf = vcvtq_f32_s32(vaddq_s32(isum0, isum1));
                accv = vmlaq_f32(accv, sumsf, scv);
            }

            acc_ac_xr = vgetq_lane_f32(accv, 0);
            acc_ad_xi = vgetq_lane_f32(accv, 1);
            acc_bc_xr = vgetq_lane_f32(accv, 2);
            acc_bd_xi = vgetq_lane_f32(accv, 3);
#else
            for (int64_t blk = 0; blk < blocks; ++blk) {
                int32_t sum_ac = 0;
                int32_t sum_ad = 0;
                int32_t sum_bc = 0;
                int32_t sum_bd = 0;

                const uint8_t * idx_blk = idx_row + (size_t) blk * (size_t) groups_per_block;
                const int16_t * lut_blk = lut_base + (size_t) blk * (size_t) groups_per_block * group_stride;
                for (int64_t gi = 0; gi < groups_per_block; ++gi) {
                    const uint8_t pat = (uint8_t) (idx_blk[gi] & 0x3f);
                    const int16_t * tbl = lut_blk + (size_t) gi * group_stride + (size_t) pat * k_ifairy_lut_channels;
                    sum_ac += (int32_t) tbl[0];
                    sum_ad += (int32_t) tbl[1];
                    sum_bc += (int32_t) tbl[2];
                    sum_bd += (int32_t) tbl[3];
                }

                const float act_scale_r = scales[blk * 2 + 0];
                const float act_scale_i = scales[blk * 2 + 1];
                acc_ac_xr += act_scale_r * (float) sum_ac;
                acc_ad_xi += act_scale_i * (float) sum_ad;
                acc_bc_xr += act_scale_r * (float) sum_bc;
                acc_bd_xi += act_scale_i * (float) sum_bd;
            }
#endif

            const float out_r = coeff_w_real * acc_ac_xr + coeff_w_imag * acc_bd_xi;
            const float out_i = coeff_w_imag * acc_bc_xr - coeff_w_real * acc_ad_xi;

            if (!isfinite(out_r) || !isfinite(out_i)) {
                ggml_abort(__FILE__, __LINE__, "ifairy_lut_qgemm: non-finite output (row=%d col=%d acc_r=%f acc_i=%f)",
                           row, 0, out_r, out_i);
            }

            (void) act_blocks;

            uint8_t * out_base = (uint8_t *) dst + (size_t) row * dst_row_stride;
            if (pack_bf16) {
                ggml_bf16_t br = GGML_FP32_TO_BF16(out_r);
                ggml_bf16_t bi = GGML_FP32_TO_BF16(out_i);
                ((ggml_bf16_t *) out_base)[0] = br;
                ((ggml_bf16_t *) out_base)[1] = bi;
            } else {
                float * out_ptr = (float *) out_base;
                if (add) {
                    out_ptr[0] += out_r;
                    out_ptr[1] += out_i;
                } else {
                    out_ptr[0] = out_r;
                    out_ptr[1] = out_i;
                }
            }
        }
        return;
    }

    for (int row = 0; row < m; ++row) {
        const block_ifairy * w_row = w_blocks + (size_t) row * (size_t) blocks;
        const uint8_t * idx_row = indexes + (size_t) row * (size_t) groups;

        const float coeff_w_real = GGML_FP16_TO_FP32(w_row[0].d_real);
        const float coeff_w_imag = GGML_FP16_TO_FP32(w_row[0].d_imag);

        for (int col = 0; col < n; ++col) {
            const int16_t * lut_base = (const int16_t *) ((const uint8_t *) lut + (size_t) col * (size_t) groups * (size_t) (k_ifairy_lut_channels * k_ifairy_lut_patterns) * sizeof(int16_t));
            const float * scales = (const float *) lut_scales + (size_t) col * (size_t) blocks * 2;
            const block_ifairy_q16 * act_blocks = act ? (const block_ifairy_q16 *) ((const uint8_t *) act + (size_t) col * act_stride) : NULL;

            float acc_ac_xr = 0.0f;
            float acc_ad_xi = 0.0f;
            float acc_bc_xr = 0.0f;
            float acc_bd_xi = 0.0f;

#if defined(__ARM_NEON) && defined(__aarch64__)
            float32x4_t accv = vdupq_n_f32(0.0f); // {ac, ad, bc, bd}
            for (int64_t blk = 0; blk < blocks; ++blk) {
                int32x4_t isum0 = vdupq_n_s32(0);
                int32x4_t isum1 = vdupq_n_s32(0);

                const uint8_t * idx_blk = idx_row + (size_t) blk * (size_t) groups_per_block;
                const int16_t * lut_blk = lut_base + (size_t) blk * (size_t) groups_per_block * (size_t) (k_ifairy_lut_patterns * k_ifairy_lut_channels);

                const size_t group_stride = (size_t) (k_ifairy_lut_patterns * k_ifairy_lut_channels);
                int64_t gi = 0;
                for (; gi + 3 < groups_per_block; gi += 4) {
                    const uint8_t pat0 = (uint8_t) (idx_blk[gi + 0] & 0x3f);
                    const uint8_t pat1 = (uint8_t) (idx_blk[gi + 1] & 0x3f);
                    const uint8_t pat2 = (uint8_t) (idx_blk[gi + 2] & 0x3f);
                    const uint8_t pat3 = (uint8_t) (idx_blk[gi + 3] & 0x3f);

                    const int16_t * grp0 = lut_blk + (size_t) (gi + 0) * group_stride;
                    const int16_t * grp1 = lut_blk + (size_t) (gi + 1) * group_stride;
                    const int16_t * grp2 = lut_blk + (size_t) (gi + 2) * group_stride;
                    const int16_t * grp3 = lut_blk + (size_t) (gi + 3) * group_stride;

                    if (prefetch) {
                        __builtin_prefetch(grp0 + group_stride, 0, 1);
                        __builtin_prefetch(grp1 + group_stride, 0, 1);
                    }

                    const int16_t * tbl0 = grp0 + (size_t) pat0 * k_ifairy_lut_channels;
                    const int16_t * tbl1 = grp1 + (size_t) pat1 * k_ifairy_lut_channels;
                    const int16_t * tbl2 = grp2 + (size_t) pat2 * k_ifairy_lut_channels;
                    const int16_t * tbl3 = grp3 + (size_t) pat3 * k_ifairy_lut_channels;

                    const int16x4_t s0 = vld1_s16(tbl0);
                    const int16x4_t s1 = vld1_s16(tbl1);
                    const int16x4_t s2 = vld1_s16(tbl2);
                    const int16x4_t s3 = vld1_s16(tbl3);

                    isum0 = vaddw_s16(isum0, s0);
                    isum1 = vaddw_s16(isum1, s1);
                    isum0 = vaddw_s16(isum0, s2);
                    isum1 = vaddw_s16(isum1, s3);
                }
                for (; gi < groups_per_block; ++gi) {
                    const uint8_t pat = (uint8_t) (idx_blk[gi] & 0x3f);
                    const int16_t * tbl = lut_blk + (size_t) gi * group_stride + (size_t) pat * k_ifairy_lut_channels;
                    const int16x4_t sums16 = vld1_s16(tbl);
                    isum0 = vaddw_s16(isum0, sums16);
                }

                const float32x2_t srsi = vld1_f32(scales + (size_t) blk * 2);
                const float32x4_t scv = vcombine_f32(srsi, srsi); // {sr, si, sr, si}
                const float32x4_t sumsf = vcvtq_f32_s32(vaddq_s32(isum0, isum1));
                accv = vmlaq_f32(accv, sumsf, scv);
            }

            acc_ac_xr = vgetq_lane_f32(accv, 0);
            acc_ad_xi = vgetq_lane_f32(accv, 1);
            acc_bc_xr = vgetq_lane_f32(accv, 2);
            acc_bd_xi = vgetq_lane_f32(accv, 3);
#else
            for (int64_t blk = 0; blk < blocks; ++blk) {
                int32_t sum_ac = 0;
                int32_t sum_ad = 0;
                int32_t sum_bc = 0;
                int32_t sum_bd = 0;

                const uint8_t * idx_blk = idx_row + (size_t) blk * (size_t) groups_per_block;
                const int16_t * lut_blk = lut_base + (size_t) blk * (size_t) groups_per_block * (size_t) (k_ifairy_lut_patterns * k_ifairy_lut_channels);

                for (int64_t gi = 0; gi < groups_per_block; ++gi) {
                    const uint8_t pat = (uint8_t) (idx_blk[gi] & 0x3f);
                    const int16_t * tbl = lut_blk + (size_t) gi * (size_t) (k_ifairy_lut_patterns * k_ifairy_lut_channels) + (size_t) pat * k_ifairy_lut_channels;
                    sum_ac += (int32_t) tbl[0];
                    sum_ad += (int32_t) tbl[1];
                    sum_bc += (int32_t) tbl[2];
                    sum_bd += (int32_t) tbl[3];
                }

                const float act_scale_r = scales[blk * 2 + 0];
                const float act_scale_i = scales[blk * 2 + 1];
                acc_ac_xr += act_scale_r * (float) sum_ac;
                acc_ad_xi += act_scale_i * (float) sum_ad;
                acc_bc_xr += act_scale_r * (float) sum_bc;
                acc_bd_xi += act_scale_i * (float) sum_bd;
            }
#endif

            const float out_r = coeff_w_real * acc_ac_xr + coeff_w_imag * acc_bd_xi;
            const float out_i = coeff_w_imag * acc_bc_xr - coeff_w_real * acc_ad_xi;

            if (!isfinite(out_r) || !isfinite(out_i)) {
                ggml_abort(__FILE__, __LINE__, "ifairy_lut_qgemm: non-finite output (row=%d col=%d acc_r=%f acc_i=%f)",
                           row, col, out_r, out_i);
            }

            if (strict) {
                GGML_ASSERT(act_blocks != NULL);
                double ref_ac_xr = 0.0;
                double ref_ad_xi = 0.0;
                double ref_bc_xr = 0.0;
                double ref_bd_xi = 0.0;

                for (int64_t blk = 0; blk < blocks; ++blk) {
                    int32_t sum_ac = 0;
                    int32_t sum_ad = 0;
                    int32_t sum_bc = 0;
                    int32_t sum_bd = 0;

                    const uint8_t * w_ptr   = w_row[blk].qs;
                    const int8_t  * x_r_ptr = (const int8_t *) act_blocks[blk].x_real;
                    const int8_t  * x_i_ptr = (const int8_t *) act_blocks[blk].x_imag;

                    for (int j = 0; j < QK_K; ++j) {
                        const int chunk    = j >> 6;
                        const int lane     = j & 0xF;
                        const int part     = (j >> 4) & 0x3;
                        const int byte_idx = (chunk << 4) + lane;
                        const int bit_off  = part * 2;

                        const uint8_t packed = w_ptr[byte_idx];
                        const uint8_t code   = (packed >> bit_off) & 0x3;

                        int wr = 0, wi = 0;
                        switch (code) {
                            case 0: wr = -1; wi =  0; break;
                            case 1: wr =  1; wi =  0; break;
                            case 2: wr =  0; wi = -1; break;
                            case 3: wr =  0; wi =  1; break;
                        }

                        const int xr = (int) x_r_ptr[j];
                        const int xi = (int) x_i_ptr[j];

                        sum_ac += xr * wr;
                        sum_ad += xi * wr;
                        sum_bc += xr * wi;
                        sum_bd += xi * wi;
                    }

                    const double x_real = (double) GGML_FP16_TO_FP32(act_blocks[blk].d_real);
                    const double x_imag = (double) GGML_FP16_TO_FP32(act_blocks[blk].d_imag);

                    ref_ac_xr += x_real * (double) sum_ac;
                    ref_ad_xi += x_imag * (double) sum_ad;
                    ref_bc_xr += x_real * (double) sum_bc;
                    ref_bd_xi += x_imag * (double) sum_bd;
                }

                const double ref_r = (double) coeff_w_real * ref_ac_xr + (double) coeff_w_imag * ref_bd_xi;
                const double ref_i = (double) coeff_w_imag * ref_bc_xr - (double) coeff_w_real * ref_ad_xi;

                const float dr = out_r - (float) ref_r;
                const float di = out_i - (float) ref_i;
                GGML_ASSERT(fabsf(dr) <= 1e-3f && fabsf(di) <= 1e-3f);
            }

            uint8_t * out_base = (uint8_t *) dst + (size_t) col * dst_col_stride + (size_t) row * dst_row_stride;
            if (pack_bf16) {
                ggml_bf16_t br = GGML_FP32_TO_BF16(out_r);
                ggml_bf16_t bi = GGML_FP32_TO_BF16(out_i);
                ((ggml_bf16_t *) out_base)[0] = br;
                ((ggml_bf16_t *) out_base)[1] = bi;
            } else {
                float * out_ptr = (float *) out_base;
                if (add) {
                    out_ptr[0] += out_r;
                    out_ptr[1] += out_i;
                } else {
                    out_ptr[0] = out_r;
                    out_ptr[1] = out_i;
                }
            }
        }
    }
}

void ggml_ifairy_lut_qgemm_ex(int m, int k, int n, const void * qweights, const uint8_t * indexes, const void * lut, const void * lut_scales, const void * act, size_t act_stride, float * dst, size_t dst_col_stride, size_t dst_row_stride, bool pack_bf16, bool strict, bool add) {
    const ggml_ifairy_lut_layout layout = ggml_ifairy_lut_layout_from_env(n);
    if (layout == GGML_IFAIRY_LUT_LAYOUT_LEGACY) {
        ggml_ifairy_lut_qgemm_ex_legacy(m, k, n, qweights, indexes, lut, lut_scales, act, act_stride, dst, dst_col_stride, dst_row_stride, pack_bf16, strict, add);
        return;
    }

    if (!indexes || !dst || !qweights || !lut || !lut_scales) {
        return;
    }
    if (strict) {
        GGML_ASSERT(add == false);
    }

    const int64_t K = k;
    const int64_t blocks = K / QK_K;
    const int64_t groups_per_block = (QK_K + 2) / 3;
    const int64_t groups = blocks * groups_per_block;

    const block_ifairy * w_blocks = (const block_ifairy *) qweights;
    const bool prefetch = ggml_ifairy_lut_prefetch_enabled();
    (void) prefetch;

    // Fast-path for decode: N == 1 avoids the col loop and some pointer arithmetic.
    // Keep strict mode on the generic path (strict validation assumes the generic structure).
    if (n == 1 && !strict) {
        const int8_t * lut_base = (const int8_t *) lut;
        const float * scales = (const float *) lut_scales;
        (void) act;

        for (int row = 0; row < m; ++row) {
            const block_ifairy * w_row = w_blocks + (size_t) row * (size_t) blocks;
            const uint8_t * idx_row = indexes + (size_t) row * (size_t) groups;

            const float coeff_w_real = GGML_FP16_TO_FP32(w_row[0].d_real);
            const float coeff_w_imag = GGML_FP16_TO_FP32(w_row[0].d_imag);

            float acc_ac_xr = 0.0f;
            float acc_ad_xi = 0.0f;
            float acc_bc_xr = 0.0f;
            float acc_bd_xi = 0.0f;

#if defined(__ARM_NEON) && defined(__aarch64__)
            float32x4_t accv = vdupq_n_f32(0.0f); // {ac, ad, bc, bd}
            for (int64_t blk = 0; blk < blocks; ++blk) {
                int32x4_t isum0 = vdupq_n_s32(0);
                int32x4_t isum1 = vdupq_n_s32(0);

                const uint8_t * idx_g = idx_row + (size_t) blk * (size_t) groups_per_block;
                const int8_t * grp   = lut_base + (size_t) blk * (size_t) groups_per_block * k_ifairy_lut_group_bytes;

                int64_t gi = 0;
                for (; gi + 3 < groups_per_block; gi += 4) {
                    const uint8_t pat0 = (uint8_t) (idx_g[0] & 0x3f);
                    const uint8_t pat1 = (uint8_t) (idx_g[1] & 0x3f);
                    const uint8_t pat2 = (uint8_t) (idx_g[2] & 0x3f);
                    const uint8_t pat3 = (uint8_t) (idx_g[3] & 0x3f);

                    const uint8_t c00 = (uint8_t) (pat0 & 3);
                    const uint8_t c01 = (uint8_t) ((pat0 >> 2) & 3);
                    const uint8_t c02 = (uint8_t) ((pat0 >> 4) & 3);

                    const uint8_t c10 = (uint8_t) (pat1 & 3);
                    const uint8_t c11 = (uint8_t) ((pat1 >> 2) & 3);
                    const uint8_t c12 = (uint8_t) ((pat1 >> 4) & 3);

                    const uint8_t c20 = (uint8_t) (pat2 & 3);
                    const uint8_t c21 = (uint8_t) ((pat2 >> 2) & 3);
                    const uint8_t c22 = (uint8_t) ((pat2 >> 4) & 3);

                    const uint8_t c30 = (uint8_t) (pat3 & 3);
                    const uint8_t c31 = (uint8_t) ((pat3 >> 2) & 3);
                    const uint8_t c32 = (uint8_t) ((pat3 >> 4) & 3);

                    const int8_t * grp0 = grp + 0 * k_ifairy_lut_group_bytes;
                    const int8_t * grp1 = grp + 1 * k_ifairy_lut_group_bytes;
                    const int8_t * grp2 = grp + 2 * k_ifairy_lut_group_bytes;
                    const int8_t * grp3 = grp + 3 * k_ifairy_lut_group_bytes;

                    if (prefetch) {
                        __builtin_prefetch(grp0 + 4 * k_ifairy_lut_group_bytes, 0, 1);
                    }

                    const int32_t * t00 = (const int32_t *) (grp0 + 0 * k_ifairy_lut_pos_bytes);
                    const int32_t * t01 = (const int32_t *) (grp0 + 1 * k_ifairy_lut_pos_bytes);
                    const int32_t * t02 = (const int32_t *) (grp0 + 2 * k_ifairy_lut_pos_bytes);

                    const int32x2_t p00 = vld1_dup_s32(t00 + c00);
                    const int32x2_t p01 = vld1_dup_s32(t01 + c01);
                    const int32x2_t p02 = vld1_dup_s32(t02 + c02);

                    int16x8_t s160 = vmovl_s8(vreinterpret_s8_s32(p00));
                    s160 = vaddq_s16(s160, vmovl_s8(vreinterpret_s8_s32(p01)));
                    s160 = vaddq_s16(s160, vmovl_s8(vreinterpret_s8_s32(p02)));
                    isum0 = vaddw_s16(isum0, vget_low_s16(s160));

                    const int32_t * t10 = (const int32_t *) (grp1 + 0 * k_ifairy_lut_pos_bytes);
                    const int32_t * t11 = (const int32_t *) (grp1 + 1 * k_ifairy_lut_pos_bytes);
                    const int32_t * t12 = (const int32_t *) (grp1 + 2 * k_ifairy_lut_pos_bytes);

                    const int32x2_t p10 = vld1_dup_s32(t10 + c10);
                    const int32x2_t p11 = vld1_dup_s32(t11 + c11);
                    const int32x2_t p12 = vld1_dup_s32(t12 + c12);

                    int16x8_t s161 = vmovl_s8(vreinterpret_s8_s32(p10));
                    s161 = vaddq_s16(s161, vmovl_s8(vreinterpret_s8_s32(p11)));
                    s161 = vaddq_s16(s161, vmovl_s8(vreinterpret_s8_s32(p12)));
                    isum1 = vaddw_s16(isum1, vget_low_s16(s161));

                    const int32_t * t20 = (const int32_t *) (grp2 + 0 * k_ifairy_lut_pos_bytes);
                    const int32_t * t21 = (const int32_t *) (grp2 + 1 * k_ifairy_lut_pos_bytes);
                    const int32_t * t22 = (const int32_t *) (grp2 + 2 * k_ifairy_lut_pos_bytes);

                    const int32x2_t p20 = vld1_dup_s32(t20 + c20);
                    const int32x2_t p21 = vld1_dup_s32(t21 + c21);
                    const int32x2_t p22 = vld1_dup_s32(t22 + c22);

                    int16x8_t s162 = vmovl_s8(vreinterpret_s8_s32(p20));
                    s162 = vaddq_s16(s162, vmovl_s8(vreinterpret_s8_s32(p21)));
                    s162 = vaddq_s16(s162, vmovl_s8(vreinterpret_s8_s32(p22)));
                    isum0 = vaddw_s16(isum0, vget_low_s16(s162));

                    const int32_t * t30 = (const int32_t *) (grp3 + 0 * k_ifairy_lut_pos_bytes);
                    const int32_t * t31 = (const int32_t *) (grp3 + 1 * k_ifairy_lut_pos_bytes);
                    const int32_t * t32 = (const int32_t *) (grp3 + 2 * k_ifairy_lut_pos_bytes);

                    const int32x2_t p30 = vld1_dup_s32(t30 + c30);
                    const int32x2_t p31 = vld1_dup_s32(t31 + c31);
                    const int32x2_t p32 = vld1_dup_s32(t32 + c32);

                    int16x8_t s163 = vmovl_s8(vreinterpret_s8_s32(p30));
                    s163 = vaddq_s16(s163, vmovl_s8(vreinterpret_s8_s32(p31)));
                    s163 = vaddq_s16(s163, vmovl_s8(vreinterpret_s8_s32(p32)));
                    isum1 = vaddw_s16(isum1, vget_low_s16(s163));

                    idx_g += 4;
                    grp   += 4 * k_ifairy_lut_group_bytes;
                }
                for (; gi + 1 < groups_per_block; gi += 2) {
                    const uint8_t pat0 = (uint8_t) (idx_g[0] & 0x3f);
                    const uint8_t pat1 = (uint8_t) (idx_g[1] & 0x3f);

                    const uint8_t c00 = (uint8_t) (pat0 & 3);
                    const uint8_t c01 = (uint8_t) ((pat0 >> 2) & 3);
                    const uint8_t c02 = (uint8_t) ((pat0 >> 4) & 3);

                    const uint8_t c10 = (uint8_t) (pat1 & 3);
                    const uint8_t c11 = (uint8_t) ((pat1 >> 2) & 3);
                    const uint8_t c12 = (uint8_t) ((pat1 >> 4) & 3);

                    const int8_t * grp0 = grp;
                    const int8_t * grp1 = grp + k_ifairy_lut_group_bytes;

                    if (prefetch) {
                        __builtin_prefetch(grp0 + 2 * k_ifairy_lut_group_bytes, 0, 1);
                    }

                    const int32_t * t00 = (const int32_t *) (grp0 + 0 * k_ifairy_lut_pos_bytes);
                    const int32_t * t01 = (const int32_t *) (grp0 + 1 * k_ifairy_lut_pos_bytes);
                    const int32_t * t02 = (const int32_t *) (grp0 + 2 * k_ifairy_lut_pos_bytes);

                    const int32x2_t p00 = vld1_dup_s32(t00 + c00);
                    const int32x2_t p01 = vld1_dup_s32(t01 + c01);
                    const int32x2_t p02 = vld1_dup_s32(t02 + c02);

                    int16x8_t s160 = vmovl_s8(vreinterpret_s8_s32(p00));
                    s160 = vaddq_s16(s160, vmovl_s8(vreinterpret_s8_s32(p01)));
                    s160 = vaddq_s16(s160, vmovl_s8(vreinterpret_s8_s32(p02)));
                    isum0 = vaddw_s16(isum0, vget_low_s16(s160));

                    const int32_t * t10 = (const int32_t *) (grp1 + 0 * k_ifairy_lut_pos_bytes);
                    const int32_t * t11 = (const int32_t *) (grp1 + 1 * k_ifairy_lut_pos_bytes);
                    const int32_t * t12 = (const int32_t *) (grp1 + 2 * k_ifairy_lut_pos_bytes);

                    const int32x2_t p10 = vld1_dup_s32(t10 + c10);
                    const int32x2_t p11 = vld1_dup_s32(t11 + c11);
                    const int32x2_t p12 = vld1_dup_s32(t12 + c12);

                    int16x8_t s161 = vmovl_s8(vreinterpret_s8_s32(p10));
                    s161 = vaddq_s16(s161, vmovl_s8(vreinterpret_s8_s32(p11)));
                    s161 = vaddq_s16(s161, vmovl_s8(vreinterpret_s8_s32(p12)));
                    isum1 = vaddw_s16(isum1, vget_low_s16(s161));

                    idx_g += 2;
                    grp   += 2 * k_ifairy_lut_group_bytes;
                }
                for (; gi < groups_per_block; ++gi, ++idx_g, grp += k_ifairy_lut_group_bytes) {
                    const uint8_t pat = (uint8_t) (*idx_g & 0x3f);
                    const uint8_t c0 = (uint8_t) (pat & 3);
                    const uint8_t c1 = (uint8_t) ((pat >> 2) & 3);
                    const uint8_t c2 = (uint8_t) ((pat >> 4) & 3);

                    if (prefetch) {
                        __builtin_prefetch(grp + k_ifairy_lut_group_bytes, 0, 1);
                    }

                    const int32_t * t0 = (const int32_t *) (grp + 0 * k_ifairy_lut_pos_bytes);
                    const int32_t * t1 = (const int32_t *) (grp + 1 * k_ifairy_lut_pos_bytes);
                    const int32_t * t2 = (const int32_t *) (grp + 2 * k_ifairy_lut_pos_bytes);

                    const int32x2_t p0 = vld1_dup_s32(t0 + c0);
                    const int32x2_t p1 = vld1_dup_s32(t1 + c1);
                    const int32x2_t p2 = vld1_dup_s32(t2 + c2);

                    int16x8_t s16 = vmovl_s8(vreinterpret_s8_s32(p0));
                    s16 = vaddq_s16(s16, vmovl_s8(vreinterpret_s8_s32(p1)));
                    s16 = vaddq_s16(s16, vmovl_s8(vreinterpret_s8_s32(p2)));
                    isum0 = vaddw_s16(isum0, vget_low_s16(s16));
                }

                const float32x2_t srsi = vld1_f32(scales + (size_t) blk * 2);
                const float32x4_t scv = vcombine_f32(srsi, srsi); // {sr, si, sr, si}
                const float32x4_t sumsf = vcvtq_f32_s32(vaddq_s32(isum0, isum1));
                accv = vmlaq_f32(accv, sumsf, scv);
            }

            acc_ac_xr = vgetq_lane_f32(accv, 0);
            acc_ad_xi = vgetq_lane_f32(accv, 1);
            acc_bc_xr = vgetq_lane_f32(accv, 2);
            acc_bd_xi = vgetq_lane_f32(accv, 3);
#else
            for (int64_t blk = 0; blk < blocks; ++blk) {
                int32_t sum_ac = 0;
                int32_t sum_ad = 0;
                int32_t sum_bc = 0;
                int32_t sum_bd = 0;

                const uint8_t * idx_g = idx_row + (size_t) blk * (size_t) groups_per_block;
                const int8_t * grp   = lut_base + (size_t) blk * (size_t) groups_per_block * k_ifairy_lut_group_bytes;

                for (int64_t gi = 0; gi < groups_per_block; ++gi, ++idx_g, grp += k_ifairy_lut_group_bytes) {
                    const uint8_t pat = (uint8_t) (*idx_g & 0x3f);
                    const uint8_t c0 = (uint8_t) (pat & 3);
                    const uint8_t c1 = (uint8_t) ((pat >> 2) & 3);
                    const uint8_t c2 = (uint8_t) ((pat >> 4) & 3);

                    const int32_t * t0 = (const int32_t *) (grp + 0 * k_ifairy_lut_pos_bytes);
                    const int32_t * t1 = (const int32_t *) (grp + 1 * k_ifairy_lut_pos_bytes);
                    const int32_t * t2 = (const int32_t *) (grp + 2 * k_ifairy_lut_pos_bytes);

                    const int8_t * e0 = (const int8_t *) &t0[c0];
                    const int8_t * e1 = (const int8_t *) &t1[c1];
                    const int8_t * e2 = (const int8_t *) &t2[c2];

                    sum_ac += (int32_t) e0[0] + (int32_t) e1[0] + (int32_t) e2[0];
                    sum_ad += (int32_t) e0[1] + (int32_t) e1[1] + (int32_t) e2[1];
                    sum_bc += (int32_t) e0[2] + (int32_t) e1[2] + (int32_t) e2[2];
                    sum_bd += (int32_t) e0[3] + (int32_t) e1[3] + (int32_t) e2[3];
                }

                const float act_scale_r = scales[blk * 2 + 0];
                const float act_scale_i = scales[blk * 2 + 1];
                acc_ac_xr += act_scale_r * (float) sum_ac;
                acc_ad_xi += act_scale_i * (float) sum_ad;
                acc_bc_xr += act_scale_r * (float) sum_bc;
                acc_bd_xi += act_scale_i * (float) sum_bd;
            }
#endif

            const float out_r = coeff_w_real * acc_ac_xr + coeff_w_imag * acc_bd_xi;
            const float out_i = coeff_w_imag * acc_bc_xr - coeff_w_real * acc_ad_xi;

            if (!isfinite(out_r) || !isfinite(out_i)) {
                ggml_abort(__FILE__, __LINE__, "ifairy_lut_qgemm: non-finite output (row=%d col=%d acc_r=%f acc_i=%f)",
                           row, 0, out_r, out_i);
            }

            uint8_t * out_base = (uint8_t *) dst + (size_t) row * dst_row_stride;
            if (pack_bf16) {
                ggml_bf16_t br = GGML_FP32_TO_BF16(out_r);
                ggml_bf16_t bi = GGML_FP32_TO_BF16(out_i);
                ((ggml_bf16_t *) out_base)[0] = br;
                ((ggml_bf16_t *) out_base)[1] = bi;
            } else {
                float * out_ptr = (float *) out_base;
                if (add) {
                    out_ptr[0] += out_r;
                    out_ptr[1] += out_i;
                } else {
                    out_ptr[0] = out_r;
                    out_ptr[1] = out_i;
                }
            }
        }

        return;
    }

    for (int row = 0; row < m; ++row) {
        const block_ifairy * w_row = w_blocks + (size_t) row * (size_t) blocks;
        const uint8_t * idx_row = indexes + (size_t) row * (size_t) groups;

        const float coeff_w_real = GGML_FP16_TO_FP32(w_row[0].d_real);
        const float coeff_w_imag = GGML_FP16_TO_FP32(w_row[0].d_imag);

        for (int col = 0; col < n; ++col) {
            const int8_t * lut_base = (const int8_t *) ((const uint8_t *) lut + (size_t) col * (size_t) groups * k_ifairy_lut_group_bytes);
            const float * scales = (const float *) lut_scales + (size_t) col * (size_t) blocks * 2;

            float acc_ac_xr = 0.0f;
            float acc_ad_xi = 0.0f;
            float acc_bc_xr = 0.0f;
            float acc_bd_xi = 0.0f;

#if defined(__ARM_NEON) && defined(__aarch64__)
            float32x4_t accv = vdupq_n_f32(0.0f); // {ac, ad, bc, bd}
            for (int64_t blk = 0; blk < blocks; ++blk) {
                int32x4_t isum0 = vdupq_n_s32(0);
                int32x4_t isum1 = vdupq_n_s32(0);

                const uint8_t * idx_g = idx_row + (size_t) blk * (size_t) groups_per_block;
                const int8_t * grp   = lut_base + (size_t) blk * (size_t) groups_per_block * k_ifairy_lut_group_bytes;

                int64_t gi = 0;
                for (; gi + 3 < groups_per_block; gi += 4) {
                    const uint8_t pat0 = (uint8_t) (idx_g[0] & 0x3f);
                    const uint8_t pat1 = (uint8_t) (idx_g[1] & 0x3f);
                    const uint8_t pat2 = (uint8_t) (idx_g[2] & 0x3f);
                    const uint8_t pat3 = (uint8_t) (idx_g[3] & 0x3f);

                    const uint8_t c00 = (uint8_t) (pat0 & 3);
                    const uint8_t c01 = (uint8_t) ((pat0 >> 2) & 3);
                    const uint8_t c02 = (uint8_t) ((pat0 >> 4) & 3);

                    const uint8_t c10 = (uint8_t) (pat1 & 3);
                    const uint8_t c11 = (uint8_t) ((pat1 >> 2) & 3);
                    const uint8_t c12 = (uint8_t) ((pat1 >> 4) & 3);

                    const uint8_t c20 = (uint8_t) (pat2 & 3);
                    const uint8_t c21 = (uint8_t) ((pat2 >> 2) & 3);
                    const uint8_t c22 = (uint8_t) ((pat2 >> 4) & 3);

                    const uint8_t c30 = (uint8_t) (pat3 & 3);
                    const uint8_t c31 = (uint8_t) ((pat3 >> 2) & 3);
                    const uint8_t c32 = (uint8_t) ((pat3 >> 4) & 3);

                    const int8_t * grp0 = grp + 0 * k_ifairy_lut_group_bytes;
                    const int8_t * grp1 = grp + 1 * k_ifairy_lut_group_bytes;
                    const int8_t * grp2 = grp + 2 * k_ifairy_lut_group_bytes;
                    const int8_t * grp3 = grp + 3 * k_ifairy_lut_group_bytes;

                    if (prefetch) {
                        __builtin_prefetch(grp0 + 4 * k_ifairy_lut_group_bytes, 0, 1);
                    }

                    const int32_t * t00 = (const int32_t *) (grp0 + 0 * k_ifairy_lut_pos_bytes);
                    const int32_t * t01 = (const int32_t *) (grp0 + 1 * k_ifairy_lut_pos_bytes);
                    const int32_t * t02 = (const int32_t *) (grp0 + 2 * k_ifairy_lut_pos_bytes);

                    const int32x2_t p00 = vld1_dup_s32(t00 + c00);
                    const int32x2_t p01 = vld1_dup_s32(t01 + c01);
                    const int32x2_t p02 = vld1_dup_s32(t02 + c02);

                    int16x8_t s160 = vmovl_s8(vreinterpret_s8_s32(p00));
                    s160 = vaddq_s16(s160, vmovl_s8(vreinterpret_s8_s32(p01)));
                    s160 = vaddq_s16(s160, vmovl_s8(vreinterpret_s8_s32(p02)));
                    isum0 = vaddw_s16(isum0, vget_low_s16(s160));

                    const int32_t * t10 = (const int32_t *) (grp1 + 0 * k_ifairy_lut_pos_bytes);
                    const int32_t * t11 = (const int32_t *) (grp1 + 1 * k_ifairy_lut_pos_bytes);
                    const int32_t * t12 = (const int32_t *) (grp1 + 2 * k_ifairy_lut_pos_bytes);

                    const int32x2_t p10 = vld1_dup_s32(t10 + c10);
                    const int32x2_t p11 = vld1_dup_s32(t11 + c11);
                    const int32x2_t p12 = vld1_dup_s32(t12 + c12);

                    int16x8_t s161 = vmovl_s8(vreinterpret_s8_s32(p10));
                    s161 = vaddq_s16(s161, vmovl_s8(vreinterpret_s8_s32(p11)));
                    s161 = vaddq_s16(s161, vmovl_s8(vreinterpret_s8_s32(p12)));
                    isum1 = vaddw_s16(isum1, vget_low_s16(s161));

                    const int32_t * t20 = (const int32_t *) (grp2 + 0 * k_ifairy_lut_pos_bytes);
                    const int32_t * t21 = (const int32_t *) (grp2 + 1 * k_ifairy_lut_pos_bytes);
                    const int32_t * t22 = (const int32_t *) (grp2 + 2 * k_ifairy_lut_pos_bytes);

                    const int32x2_t p20 = vld1_dup_s32(t20 + c20);
                    const int32x2_t p21 = vld1_dup_s32(t21 + c21);
                    const int32x2_t p22 = vld1_dup_s32(t22 + c22);

                    int16x8_t s162 = vmovl_s8(vreinterpret_s8_s32(p20));
                    s162 = vaddq_s16(s162, vmovl_s8(vreinterpret_s8_s32(p21)));
                    s162 = vaddq_s16(s162, vmovl_s8(vreinterpret_s8_s32(p22)));
                    isum0 = vaddw_s16(isum0, vget_low_s16(s162));

                    const int32_t * t30 = (const int32_t *) (grp3 + 0 * k_ifairy_lut_pos_bytes);
                    const int32_t * t31 = (const int32_t *) (grp3 + 1 * k_ifairy_lut_pos_bytes);
                    const int32_t * t32 = (const int32_t *) (grp3 + 2 * k_ifairy_lut_pos_bytes);

                    const int32x2_t p30 = vld1_dup_s32(t30 + c30);
                    const int32x2_t p31 = vld1_dup_s32(t31 + c31);
                    const int32x2_t p32 = vld1_dup_s32(t32 + c32);

                    int16x8_t s163 = vmovl_s8(vreinterpret_s8_s32(p30));
                    s163 = vaddq_s16(s163, vmovl_s8(vreinterpret_s8_s32(p31)));
                    s163 = vaddq_s16(s163, vmovl_s8(vreinterpret_s8_s32(p32)));
                    isum1 = vaddw_s16(isum1, vget_low_s16(s163));

                    idx_g += 4;
                    grp   += 4 * k_ifairy_lut_group_bytes;
                }
                for (; gi + 1 < groups_per_block; gi += 2) {
                    const uint8_t pat0 = (uint8_t) (idx_g[0] & 0x3f);
                    const uint8_t pat1 = (uint8_t) (idx_g[1] & 0x3f);

                    const uint8_t c00 = (uint8_t) (pat0 & 3);
                    const uint8_t c01 = (uint8_t) ((pat0 >> 2) & 3);
                    const uint8_t c02 = (uint8_t) ((pat0 >> 4) & 3);

                    const uint8_t c10 = (uint8_t) (pat1 & 3);
                    const uint8_t c11 = (uint8_t) ((pat1 >> 2) & 3);
                    const uint8_t c12 = (uint8_t) ((pat1 >> 4) & 3);

                    const int8_t * grp0 = grp;
                    const int8_t * grp1 = grp + k_ifairy_lut_group_bytes;

                    if (prefetch) {
                        __builtin_prefetch(grp0 + 2 * k_ifairy_lut_group_bytes, 0, 1);
                    }

                    const int32_t * t00 = (const int32_t *) (grp0 + 0 * k_ifairy_lut_pos_bytes);
                    const int32_t * t01 = (const int32_t *) (grp0 + 1 * k_ifairy_lut_pos_bytes);
                    const int32_t * t02 = (const int32_t *) (grp0 + 2 * k_ifairy_lut_pos_bytes);

                    const int32x2_t p00 = vld1_dup_s32(t00 + c00);
                    const int32x2_t p01 = vld1_dup_s32(t01 + c01);
                    const int32x2_t p02 = vld1_dup_s32(t02 + c02);

                    int16x8_t s160 = vmovl_s8(vreinterpret_s8_s32(p00));
                    s160 = vaddq_s16(s160, vmovl_s8(vreinterpret_s8_s32(p01)));
                    s160 = vaddq_s16(s160, vmovl_s8(vreinterpret_s8_s32(p02)));
                    isum0 = vaddw_s16(isum0, vget_low_s16(s160));

                    const int32_t * t10 = (const int32_t *) (grp1 + 0 * k_ifairy_lut_pos_bytes);
                    const int32_t * t11 = (const int32_t *) (grp1 + 1 * k_ifairy_lut_pos_bytes);
                    const int32_t * t12 = (const int32_t *) (grp1 + 2 * k_ifairy_lut_pos_bytes);

                    const int32x2_t p10 = vld1_dup_s32(t10 + c10);
                    const int32x2_t p11 = vld1_dup_s32(t11 + c11);
                    const int32x2_t p12 = vld1_dup_s32(t12 + c12);

                    int16x8_t s161 = vmovl_s8(vreinterpret_s8_s32(p10));
                    s161 = vaddq_s16(s161, vmovl_s8(vreinterpret_s8_s32(p11)));
                    s161 = vaddq_s16(s161, vmovl_s8(vreinterpret_s8_s32(p12)));
                    isum1 = vaddw_s16(isum1, vget_low_s16(s161));

                    idx_g += 2;
                    grp   += 2 * k_ifairy_lut_group_bytes;
                }
                for (; gi < groups_per_block; ++gi, ++idx_g, grp += k_ifairy_lut_group_bytes) {
                    const uint8_t pat = (uint8_t) (*idx_g & 0x3f);
                    const uint8_t c0 = (uint8_t) (pat & 3);
                    const uint8_t c1 = (uint8_t) ((pat >> 2) & 3);
                    const uint8_t c2 = (uint8_t) ((pat >> 4) & 3);

                    if (prefetch) {
                        __builtin_prefetch(grp + k_ifairy_lut_group_bytes, 0, 1);
                    }

                    const int32_t * t0 = (const int32_t *) (grp + 0 * k_ifairy_lut_pos_bytes);
                    const int32_t * t1 = (const int32_t *) (grp + 1 * k_ifairy_lut_pos_bytes);
                    const int32_t * t2 = (const int32_t *) (grp + 2 * k_ifairy_lut_pos_bytes);

                    const int32x2_t p0 = vld1_dup_s32(t0 + c0);
                    const int32x2_t p1 = vld1_dup_s32(t1 + c1);
                    const int32x2_t p2 = vld1_dup_s32(t2 + c2);

                    int16x8_t s16 = vmovl_s8(vreinterpret_s8_s32(p0));
                    s16 = vaddq_s16(s16, vmovl_s8(vreinterpret_s8_s32(p1)));
                    s16 = vaddq_s16(s16, vmovl_s8(vreinterpret_s8_s32(p2)));
                    isum0 = vaddw_s16(isum0, vget_low_s16(s16));
                }

                const float32x2_t srsi = vld1_f32(scales + (size_t) blk * 2);
                const float32x4_t scv = vcombine_f32(srsi, srsi); // {sr, si, sr, si}
                const float32x4_t sumsf = vcvtq_f32_s32(vaddq_s32(isum0, isum1));
                accv = vmlaq_f32(accv, sumsf, scv);
            }

            acc_ac_xr = vgetq_lane_f32(accv, 0);
            acc_ad_xi = vgetq_lane_f32(accv, 1);
            acc_bc_xr = vgetq_lane_f32(accv, 2);
            acc_bd_xi = vgetq_lane_f32(accv, 3);
#else
            for (int64_t blk = 0; blk < blocks; ++blk) {
                int32_t sum_ac = 0;
                int32_t sum_ad = 0;
                int32_t sum_bc = 0;
                int32_t sum_bd = 0;

                const uint8_t * idx_g = idx_row + (size_t) blk * (size_t) groups_per_block;
                const int8_t * grp   = lut_base + (size_t) blk * (size_t) groups_per_block * k_ifairy_lut_group_bytes;

                for (int64_t gi = 0; gi < groups_per_block; ++gi, ++idx_g, grp += k_ifairy_lut_group_bytes) {
                    const uint8_t pat = (uint8_t) (*idx_g & 0x3f);
                    const uint8_t c0 = (uint8_t) (pat & 3);
                    const uint8_t c1 = (uint8_t) ((pat >> 2) & 3);
                    const uint8_t c2 = (uint8_t) ((pat >> 4) & 3);

                    const int32_t * t0 = (const int32_t *) (grp + 0 * k_ifairy_lut_pos_bytes);
                    const int32_t * t1 = (const int32_t *) (grp + 1 * k_ifairy_lut_pos_bytes);
                    const int32_t * t2 = (const int32_t *) (grp + 2 * k_ifairy_lut_pos_bytes);

                    const int8_t * e0 = (const int8_t *) &t0[c0];
                    const int8_t * e1 = (const int8_t *) &t1[c1];
                    const int8_t * e2 = (const int8_t *) &t2[c2];

                    sum_ac += (int32_t) e0[0] + (int32_t) e1[0] + (int32_t) e2[0];
                    sum_ad += (int32_t) e0[1] + (int32_t) e1[1] + (int32_t) e2[1];
                    sum_bc += (int32_t) e0[2] + (int32_t) e1[2] + (int32_t) e2[2];
                    sum_bd += (int32_t) e0[3] + (int32_t) e1[3] + (int32_t) e2[3];
                }

                const float act_scale_r = scales[blk * 2 + 0];
                const float act_scale_i = scales[blk * 2 + 1];
                acc_ac_xr += act_scale_r * (float) sum_ac;
                acc_ad_xi += act_scale_i * (float) sum_ad;
                acc_bc_xr += act_scale_r * (float) sum_bc;
                acc_bd_xi += act_scale_i * (float) sum_bd;
            }
#endif

            const float out_r = coeff_w_real * acc_ac_xr + coeff_w_imag * acc_bd_xi;
            const float out_i = coeff_w_imag * acc_bc_xr - coeff_w_real * acc_ad_xi;

            if (!isfinite(out_r) || !isfinite(out_i)) {
                ggml_abort(__FILE__, __LINE__, "ifairy_lut_qgemm: non-finite output (row=%d col=%d acc_r=%f acc_i=%f)",
                           row, col, out_r, out_i);
            }

            if (strict) {
                GGML_ASSERT(act != NULL);
                const block_ifairy_q16 * act_blocks = (const block_ifairy_q16 *) ((const uint8_t *) act + (size_t) col * act_stride);
                double ref_ac_xr = 0.0;
                double ref_ad_xi = 0.0;
                double ref_bc_xr = 0.0;
                double ref_bd_xi = 0.0;

                for (int blk = 0; blk < (int) blocks; ++blk) {
                    const uint8_t * GGML_RESTRICT w_ptr   = w_row[blk].qs;
                    const int8_t  * GGML_RESTRICT x_r_ptr = (const int8_t *) act_blocks[blk].x_real;
                    const int8_t  * GGML_RESTRICT x_i_ptr = (const int8_t *) act_blocks[blk].x_imag;

                    int32_t sum_ac = 0;
                    int32_t sum_ad = 0;
                    int32_t sum_bc = 0;
                    int32_t sum_bd = 0;

                    for (int j = 0; j < QK_K; ++j) {
                        const int chunk    = j >> 6;
                        const int lane     = j & 0xF;
                        const int part     = (j >> 4) & 0x3;
                        const int byte_idx = (chunk << 4) + lane;
                        const int bit_off  = part * 2;

                        const uint8_t packed = w_ptr[byte_idx];
                        const uint8_t code   = (packed >> bit_off) & 0x3;

                        int wr = 0;
                        int wi = 0;
                        switch (code) {
                            case 0: wr = -1; wi =  0; break;
                            case 1: wr =  1; wi =  0; break;
                            case 2: wr =  0; wi = -1; break;
                            case 3: wr =  0; wi =  1; break;
                        }

                        const int xr = (int) x_r_ptr[j];
                        const int xi = (int) x_i_ptr[j];

                        sum_ac += xr * wr;
                        sum_ad += xi * wr;
                        sum_bc += xr * wi;
                        sum_bd += xi * wi;
                    }

                    const double x_real = (double) GGML_FP16_TO_FP32(act_blocks[blk].d_real);
                    const double x_imag = (double) GGML_FP16_TO_FP32(act_blocks[blk].d_imag);

                    ref_ac_xr += x_real * (double) sum_ac;
                    ref_ad_xi += x_imag * (double) sum_ad;
                    ref_bc_xr += x_real * (double) sum_bc;
                    ref_bd_xi += x_imag * (double) sum_bd;
                }

                const double ref_r = (double) coeff_w_real * ref_ac_xr + (double) coeff_w_imag * ref_bd_xi;
                const double ref_i = (double) coeff_w_imag * ref_bc_xr - (double) coeff_w_real * ref_ad_xi;

                const float dr = out_r - (float) ref_r;
                const float di = out_i - (float) ref_i;
                GGML_ASSERT(fabsf(dr) <= 1e-3f && fabsf(di) <= 1e-3f);
            }

            uint8_t * out_base = (uint8_t *) dst + (size_t) col * dst_col_stride + (size_t) row * dst_row_stride;
            if (pack_bf16) {
                ggml_bf16_t br = GGML_FP32_TO_BF16(out_r);
                ggml_bf16_t bi = GGML_FP32_TO_BF16(out_i);
                ((ggml_bf16_t *) out_base)[0] = br;
                ((ggml_bf16_t *) out_base)[1] = bi;
            } else {
                float * out_ptr = (float *) out_base;
                if (add) {
                    out_ptr[0] += out_r;
                    out_ptr[1] += out_i;
                } else {
                    out_ptr[0] = out_r;
                    out_ptr[1] = out_i;
                }
            }
        }
    }
}

void ggml_ifairy_lut_qgemm(int m, int k, int n, const void * qweights, const uint8_t * indexes, const void * lut, const void * lut_scales, const void * act, size_t act_stride, float * dst, size_t dst_col_stride, size_t dst_row_stride, bool pack_bf16, bool strict) {
    ggml_ifairy_lut_qgemm_ex(m, k, n, qweights, indexes, lut, lut_scales, act, act_stride, dst, dst_col_stride, dst_row_stride, pack_bf16, strict, false);
}

// Accumulate the 4 basis sums for the iFairy complex dot product:
//   dst[c*dst_col_stride + 0..3] += { acc_ac_xr, acc_ad_xi, acc_bc_xr, acc_bd_xi }
// where each term already includes the per-block activation scales (real/imag).
static void ggml_ifairy_lut_accum4_ex_legacy(int k, int n, const uint8_t * indexes, const void * lut, const void * lut_scales, float * dst, size_t dst_col_stride, bool add) {
    if (!indexes || !dst || !lut || !lut_scales) {
        return;
    }

    const bool prefetch = ggml_ifairy_lut_prefetch_enabled();
    (void) prefetch;

    const int64_t K = k;
    const int64_t blocks = K / QK_K;
    const int64_t groups_per_block = (QK_K + 2) / 3;
    const int64_t groups = blocks * groups_per_block;

    for (int col = 0; col < n; ++col) {
        const int16_t * lut_base = (const int16_t *) ((const uint8_t *) lut + (size_t) col * (size_t) groups * (size_t) (k_ifairy_lut_channels * k_ifairy_lut_patterns) * sizeof(int16_t));
        const float * scales = (const float *) lut_scales + (size_t) col * (size_t) blocks * 2;
        float * out = (float *) ((uint8_t *) dst + (size_t) col * dst_col_stride);

#if defined(__ARM_NEON) && defined(__aarch64__)
        float32x4_t accv = add ? vld1q_f32(out) : vdupq_n_f32(0.0f);
        for (int64_t blk = 0; blk < blocks; ++blk) {
            int32x4_t isum0 = vdupq_n_s32(0);
            int32x4_t isum1 = vdupq_n_s32(0);
            const uint8_t * idx_blk = indexes + (size_t) blk * (size_t) groups_per_block;
            const int16_t * lut_blk = lut_base + (size_t) blk * (size_t) groups_per_block * (size_t) (k_ifairy_lut_patterns * k_ifairy_lut_channels);

            const size_t group_stride = (size_t) (k_ifairy_lut_patterns * k_ifairy_lut_channels);
            int64_t gi = 0;
            for (; gi + 3 < groups_per_block; gi += 4) {
                const uint8_t pat0 = (uint8_t) (idx_blk[gi + 0] & 0x3f);
                const uint8_t pat1 = (uint8_t) (idx_blk[gi + 1] & 0x3f);
                const uint8_t pat2 = (uint8_t) (idx_blk[gi + 2] & 0x3f);
                const uint8_t pat3 = (uint8_t) (idx_blk[gi + 3] & 0x3f);

                const int16_t * grp0 = lut_blk + (size_t) (gi + 0) * group_stride;
                const int16_t * grp1 = lut_blk + (size_t) (gi + 1) * group_stride;
                const int16_t * grp2 = lut_blk + (size_t) (gi + 2) * group_stride;
                const int16_t * grp3 = lut_blk + (size_t) (gi + 3) * group_stride;

                if (prefetch) {
                    __builtin_prefetch(grp0 + group_stride, 0, 1);
                    __builtin_prefetch(grp1 + group_stride, 0, 1);
                }

                const int16_t * tbl0 = grp0 + (size_t) pat0 * k_ifairy_lut_channels;
                const int16_t * tbl1 = grp1 + (size_t) pat1 * k_ifairy_lut_channels;
                const int16_t * tbl2 = grp2 + (size_t) pat2 * k_ifairy_lut_channels;
                const int16_t * tbl3 = grp3 + (size_t) pat3 * k_ifairy_lut_channels;

                const int16x4_t s0 = vld1_s16(tbl0);
                const int16x4_t s1 = vld1_s16(tbl1);
                const int16x4_t s2 = vld1_s16(tbl2);
                const int16x4_t s3 = vld1_s16(tbl3);

                isum0 = vaddw_s16(isum0, s0);
                isum1 = vaddw_s16(isum1, s1);
                isum0 = vaddw_s16(isum0, s2);
                isum1 = vaddw_s16(isum1, s3);
            }
            for (; gi < groups_per_block; ++gi) {
                const uint8_t pat = (uint8_t) (idx_blk[gi] & 0x3f);
                const int16_t * tbl = lut_blk + (size_t) gi * group_stride + (size_t) pat * k_ifairy_lut_channels;
                const int16x4_t sums16 = vld1_s16(tbl); // {ac, ad, bc, bd}
                isum0 = vaddw_s16(isum0, sums16);
            }

            const float32x2_t srsi = vld1_f32(scales + (size_t) blk * 2);
            const float32x4_t scv = vcombine_f32(srsi, srsi); // {sr, si, sr, si}
            const float32x4_t sumsf = vcvtq_f32_s32(vaddq_s32(isum0, isum1));
            accv = vmlaq_f32(accv, sumsf, scv);
        }
        vst1q_f32(out, accv);
#else
        float acc_ac_xr = add ? out[0] : 0.0f;
        float acc_ad_xi = add ? out[1] : 0.0f;
        float acc_bc_xr = add ? out[2] : 0.0f;
        float acc_bd_xi = add ? out[3] : 0.0f;

        for (int64_t blk = 0; blk < blocks; ++blk) {
            int32_t sum_ac = 0;
            int32_t sum_ad = 0;
            int32_t sum_bc = 0;
            int32_t sum_bd = 0;

            const uint8_t * idx_blk = indexes + (size_t) blk * (size_t) groups_per_block;
            const int16_t * lut_blk = lut_base + (size_t) blk * (size_t) groups_per_block * (size_t) (k_ifairy_lut_patterns * k_ifairy_lut_channels);

            for (int64_t gi = 0; gi < groups_per_block; ++gi) {
                const uint8_t pat = (uint8_t) (idx_blk[gi] & 0x3f);
                const int16_t * tbl = lut_blk + (size_t) gi * (size_t) (k_ifairy_lut_patterns * k_ifairy_lut_channels) + (size_t) pat * k_ifairy_lut_channels;
                sum_ac += (int32_t) tbl[0];
                sum_ad += (int32_t) tbl[1];
                sum_bc += (int32_t) tbl[2];
                sum_bd += (int32_t) tbl[3];
            }

            const float act_scale_r = scales[blk * 2 + 0];
            const float act_scale_i = scales[blk * 2 + 1];
            acc_ac_xr += act_scale_r * (float) sum_ac;
            acc_ad_xi += act_scale_i * (float) sum_ad;
            acc_bc_xr += act_scale_r * (float) sum_bc;
            acc_bd_xi += act_scale_i * (float) sum_bd;
        }

        out[0] = acc_ac_xr;
        out[1] = acc_ad_xi;
        out[2] = acc_bc_xr;
        out[3] = acc_bd_xi;
#endif
    }
}

void ggml_ifairy_lut_accum4_ex(int k, int n, const uint8_t * indexes, const void * lut, const void * lut_scales, float * dst, size_t dst_col_stride, bool add) {
    const ggml_ifairy_lut_layout layout = ggml_ifairy_lut_layout_from_env(n);
    if (layout == GGML_IFAIRY_LUT_LAYOUT_LEGACY) {
        ggml_ifairy_lut_accum4_ex_legacy(k, n, indexes, lut, lut_scales, dst, dst_col_stride, add);
        return;
    }

    if (!indexes || !dst || !lut || !lut_scales) {
        return;
    }

    const bool prefetch = ggml_ifairy_lut_prefetch_enabled();
    (void) prefetch;

    const int64_t K = k;
    const int64_t blocks = K / QK_K;
    const int64_t groups_per_block = (QK_K + 2) / 3;
    const int64_t groups = blocks * groups_per_block;

    for (int col = 0; col < n; ++col) {
        const int8_t * lut_base = (const int8_t *) ((const uint8_t *) lut + (size_t) col * (size_t) groups * k_ifairy_lut_group_bytes);
        const float * scales = (const float *) lut_scales + (size_t) col * (size_t) blocks * 2;
        float * out = (float *) ((uint8_t *) dst + (size_t) col * dst_col_stride);

#if defined(__ARM_NEON) && defined(__aarch64__)
        float32x4_t accv = add ? vld1q_f32(out) : vdupq_n_f32(0.0f);
        for (int64_t blk = 0; blk < blocks; ++blk) {
            int32x4_t isum0 = vdupq_n_s32(0);
            int32x4_t isum1 = vdupq_n_s32(0);
            const uint8_t * idx_g = indexes + (size_t) blk * (size_t) groups_per_block;
            const int8_t * grp   = lut_base + (size_t) blk * (size_t) groups_per_block * k_ifairy_lut_group_bytes;

            int64_t gi = 0;
            for (; gi + 1 < groups_per_block; gi += 2) {
                const uint8_t pat0 = (uint8_t) (idx_g[0] & 0x3f);
                const uint8_t pat1 = (uint8_t) (idx_g[1] & 0x3f);

                const uint8_t c00 = (uint8_t) (pat0 & 3);
                const uint8_t c01 = (uint8_t) ((pat0 >> 2) & 3);
                const uint8_t c02 = (uint8_t) ((pat0 >> 4) & 3);

                const uint8_t c10 = (uint8_t) (pat1 & 3);
                const uint8_t c11 = (uint8_t) ((pat1 >> 2) & 3);
                const uint8_t c12 = (uint8_t) ((pat1 >> 4) & 3);

                const int8_t * grp0 = grp;
                const int8_t * grp1 = grp + k_ifairy_lut_group_bytes;

                if (prefetch) {
                    __builtin_prefetch(grp0 + 2 * k_ifairy_lut_group_bytes, 0, 1);
                }

                const int32_t * t00 = (const int32_t *) (grp0 + 0 * k_ifairy_lut_pos_bytes);
                const int32_t * t01 = (const int32_t *) (grp0 + 1 * k_ifairy_lut_pos_bytes);
                const int32_t * t02 = (const int32_t *) (grp0 + 2 * k_ifairy_lut_pos_bytes);

                const int32x2_t p00 = vld1_dup_s32(t00 + c00);
                const int32x2_t p01 = vld1_dup_s32(t01 + c01);
                const int32x2_t p02 = vld1_dup_s32(t02 + c02);

                int16x8_t s160 = vmovl_s8(vreinterpret_s8_s32(p00));
                s160 = vaddq_s16(s160, vmovl_s8(vreinterpret_s8_s32(p01)));
                s160 = vaddq_s16(s160, vmovl_s8(vreinterpret_s8_s32(p02)));
                isum0 = vaddw_s16(isum0, vget_low_s16(s160));

                const int32_t * t10 = (const int32_t *) (grp1 + 0 * k_ifairy_lut_pos_bytes);
                const int32_t * t11 = (const int32_t *) (grp1 + 1 * k_ifairy_lut_pos_bytes);
                const int32_t * t12 = (const int32_t *) (grp1 + 2 * k_ifairy_lut_pos_bytes);

                const int32x2_t p10 = vld1_dup_s32(t10 + c10);
                const int32x2_t p11 = vld1_dup_s32(t11 + c11);
                const int32x2_t p12 = vld1_dup_s32(t12 + c12);

                int16x8_t s161 = vmovl_s8(vreinterpret_s8_s32(p10));
                s161 = vaddq_s16(s161, vmovl_s8(vreinterpret_s8_s32(p11)));
                s161 = vaddq_s16(s161, vmovl_s8(vreinterpret_s8_s32(p12)));
                isum1 = vaddw_s16(isum1, vget_low_s16(s161));

                idx_g += 2;
                grp   += 2 * k_ifairy_lut_group_bytes;
            }
            for (; gi < groups_per_block; ++gi, ++idx_g, grp += k_ifairy_lut_group_bytes) {
                const uint8_t pat = (uint8_t) (*idx_g & 0x3f);
                const uint8_t c0 = (uint8_t) (pat & 3);
                const uint8_t c1 = (uint8_t) ((pat >> 2) & 3);
                const uint8_t c2 = (uint8_t) ((pat >> 4) & 3);

                if (prefetch) {
                    __builtin_prefetch(grp + k_ifairy_lut_group_bytes, 0, 1);
                }

                const int32_t * t0 = (const int32_t *) (grp + 0 * k_ifairy_lut_pos_bytes);
                const int32_t * t1 = (const int32_t *) (grp + 1 * k_ifairy_lut_pos_bytes);
                const int32_t * t2 = (const int32_t *) (grp + 2 * k_ifairy_lut_pos_bytes);

                const int32x2_t p0 = vld1_dup_s32(t0 + c0);
                const int32x2_t p1 = vld1_dup_s32(t1 + c1);
                const int32x2_t p2 = vld1_dup_s32(t2 + c2);

                int16x8_t s16 = vmovl_s8(vreinterpret_s8_s32(p0));
                s16 = vaddq_s16(s16, vmovl_s8(vreinterpret_s8_s32(p1)));
                s16 = vaddq_s16(s16, vmovl_s8(vreinterpret_s8_s32(p2)));
                isum0 = vaddw_s16(isum0, vget_low_s16(s16));
            }

            const float32x2_t srsi = vld1_f32(scales + (size_t) blk * 2);
            const float32x4_t scv = vcombine_f32(srsi, srsi); // {sr, si, sr, si}
            const float32x4_t sumsf = vcvtq_f32_s32(vaddq_s32(isum0, isum1));
            accv = vmlaq_f32(accv, sumsf, scv);
        }
        vst1q_f32(out, accv);
#else
        float acc_ac_xr = add ? out[0] : 0.0f;
        float acc_ad_xi = add ? out[1] : 0.0f;
        float acc_bc_xr = add ? out[2] : 0.0f;
        float acc_bd_xi = add ? out[3] : 0.0f;

        for (int64_t blk = 0; blk < blocks; ++blk) {
            int32_t sum_ac = 0;
            int32_t sum_ad = 0;
            int32_t sum_bc = 0;
            int32_t sum_bd = 0;

            const uint8_t * idx_g = indexes + (size_t) blk * (size_t) groups_per_block;
            const int8_t * grp   = lut_base + (size_t) blk * (size_t) groups_per_block * k_ifairy_lut_group_bytes;

            for (int64_t gi = 0; gi < groups_per_block; ++gi, ++idx_g, grp += k_ifairy_lut_group_bytes) {
                const uint8_t pat = (uint8_t) (*idx_g & 0x3f);
                const uint8_t c0 = (uint8_t) (pat & 3);
                const uint8_t c1 = (uint8_t) ((pat >> 2) & 3);
                const uint8_t c2 = (uint8_t) ((pat >> 4) & 3);

                const int32_t * t0 = (const int32_t *) (grp + 0 * k_ifairy_lut_pos_bytes);
                const int32_t * t1 = (const int32_t *) (grp + 1 * k_ifairy_lut_pos_bytes);
                const int32_t * t2 = (const int32_t *) (grp + 2 * k_ifairy_lut_pos_bytes);

                const int8_t * e0 = (const int8_t *) &t0[c0];
                const int8_t * e1 = (const int8_t *) &t1[c1];
                const int8_t * e2 = (const int8_t *) &t2[c2];

                sum_ac += (int32_t) e0[0] + (int32_t) e1[0] + (int32_t) e2[0];
                sum_ad += (int32_t) e0[1] + (int32_t) e1[1] + (int32_t) e2[1];
                sum_bc += (int32_t) e0[2] + (int32_t) e1[2] + (int32_t) e2[2];
                sum_bd += (int32_t) e0[3] + (int32_t) e1[3] + (int32_t) e2[3];
            }

            const float act_scale_r = scales[blk * 2 + 0];
            const float act_scale_i = scales[blk * 2 + 1];
            acc_ac_xr += act_scale_r * (float) sum_ac;
            acc_ad_xi += act_scale_i * (float) sum_ad;
            acc_bc_xr += act_scale_r * (float) sum_bc;
            acc_bd_xi += act_scale_i * (float) sum_bd;
        }

        out[0] = acc_ac_xr;
        out[1] = acc_ad_xi;
        out[2] = acc_bc_xr;
        out[3] = acc_bd_xi;
#endif
    }
}

static void ggml_ifairy_lut_mul_mat_scalar_internal(int m, int k, int n, const void * qweights, const void * act, size_t act_stride,
                                                    const uint8_t * indexes, int8_t * lut, float * scales, float * dst, size_t dst_col_stride) {
    if (!qweights || !act || !dst || !indexes || !lut || !scales) {
        return;
    }

    const bool strict = getenv("GGML_IFAIRY_LUT_VALIDATE_STRICT") && strcmp(getenv("GGML_IFAIRY_LUT_VALIDATE_STRICT"), "0") != 0;

    // preprocess activations -> LUT per column
    ggml_ifairy_lut_preprocess(m, k, n, act, act_stride, scales, lut);
    const size_t dst_row_stride = 2 * sizeof(float);
    ggml_ifairy_lut_qgemm(m, k, n, qweights, indexes, lut, scales, act, act_stride, dst, dst_col_stride, dst_row_stride, false, strict);
}

void ggml_ifairy_lut_mul_mat_scalar(int m, int k, int n, const void * qweights, const void * act, size_t act_stride, float * dst) {
    if (!qweights || !act || !dst) {
        return;
    }

    const int64_t K = k;
    const int64_t blocks = K / QK_K;
    const int64_t groups_per_block = (QK_K + 2) / 3;
    const int64_t groups = blocks * groups_per_block;

    // workspace: indexes + LUT + scales
    const size_t index_bytes_raw = (size_t) m * (size_t) groups;
    const size_t index_bytes = GGML_PAD(index_bytes_raw, 64);
    const ggml_ifairy_lut_layout layout = ggml_ifairy_lut_layout_from_env(n);
    const size_t lut_bytes = layout == GGML_IFAIRY_LUT_LAYOUT_LEGACY
            ? (size_t) n * (size_t) groups * (size_t) (k_ifairy_lut_channels * k_ifairy_lut_patterns) * sizeof(int16_t)
            : (size_t) n * (size_t) groups * (size_t) k_ifairy_lut_group_bytes;
    const size_t scale_bytes = (size_t) n * (size_t) blocks * 2 * sizeof(float);
    const size_t total_bytes = index_bytes + lut_bytes + scale_bytes;

    void * ptr = NULL;
    if (posix_memalign(&ptr, 64, total_bytes) != 0) {
        return;
    }
    uint8_t * buf = (uint8_t *) ptr;
    memset(buf, 0, total_bytes);
    uint8_t * indexes = buf;
    int8_t * lut = (int8_t *) (buf + index_bytes);
    float * scales = (float *) (buf + index_bytes + lut_bytes);

    // build indexes per row
    ggml_ifairy_3w_encode((const block_ifairy *) qweights, K, m, indexes, index_bytes_raw);
    ggml_ifairy_lut_mul_mat_scalar_internal(m, k, n, qweights, act, act_stride, indexes, lut, scales, dst, (size_t) m * 2 * sizeof(float));

    free(buf);
}
