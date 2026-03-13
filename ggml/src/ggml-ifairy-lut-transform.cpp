#define GGML_COMMON_DECL_CPP
#include "ggml-backend.h"
#include "ggml-common.h"
#include "ggml-ifairy-lut-impl.h"
#include "ggml-impl.h"
#include "ggml-quants.h"

#ifndef GGML_FP16_TO_FP32
#    define GGML_FP16_TO_FP32 ggml_fp16_to_fp32
#endif

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <mutex>
#include <unordered_map>
#include <vector>

static_assert(QK_IFAIRY == 256, "lut packing assumes QK_IFAIRY=256");

static std::vector<ifairy_lut_extra *> g_ifairy_lut_extras;
static std::mutex                      g_ifairy_lut_mutex;

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
        h ^= std::hash<int64_t>{}(key.k) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= std::hash<int64_t>{}(key.rows) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return h;
    }
};

struct ifairy_lut_index_cache_entry {
    ggml_backend_buffer_t buffer = nullptr;
    uint8_t *             base   = nullptr;
    size_t                size   = 0;  // total buffer bytes
};

static std::unordered_map<ifairy_lut_index_cache_key, ifairy_lut_index_cache_entry, ifairy_lut_index_cache_key_hash>
    g_ifairy_lut_index_cache;

// iFairy 2-weight LUT implementation (CPU backend).
// Integrated into ggml mul_mat routing under GGML_IFAIRY_LUT_CPU.

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
                const size_t index_bytes_aligned = GGML_PAD(e->size, GGML_IFAIRY_LUT_WTILE_ALIGNMENT);
                const size_t total_bytes         = index_bytes_aligned + e->packed_w_size;
                ggml_aligned_free(e->indexes, total_bytes);
            }
            e->indexes       = NULL;
            e->size          = 0;
            e->packed_w      = NULL;
            e->packed_w_size = 0;
            e->index_tensor  = NULL;
            e->index_buffer  = NULL;
        }
    }
}

bool ggml_ifairy_lut_transform_tensor(struct ggml_tensor * tensor, struct ggml_tensor ** index_tensor_out) {
    if (!tensor || tensor->type != GGML_TYPE_IFAIRY) {
        if (index_tensor_out) {
            *index_tensor_out = NULL;
        }
        return false;
    }

    const bool dbg = ggml_ifairy_env_enabled("GGML_IFAIRY_LUT_DEBUG");

    ifairy_lut_extra * extra = (ifairy_lut_extra *) tensor->extra;
    if (extra && extra->indexes) {
        if (index_tensor_out) {
            *index_tensor_out = NULL;
        }
        return true;
    }

    const int64_t k    = tensor->ne[0];
    const int64_t rows = tensor->ne[1];
    if (k % QK_IFAIRY != 0 || rows <= 0) {
        if (dbg) {
            GGML_LOG_WARN("ifairy_lut: transform_tensor: invalid shape k=%lld rows=%lld QK_IFAIRY=%d\n", (long long) k,
                          (long long) rows, QK_IFAIRY);
        }
        return false;
    }

    const struct ggml_ifairy_2w_index_info info        = ggml_ifairy_2w_get_index_info(k);
    const size_t                           index_bytes = ggml_ifairy_2w_index_buffer_size(&info, rows);
    if (index_bytes == 0) {
        if (dbg) {
            GGML_LOG_WARN("ifairy_lut: transform_tensor: index_bytes==0 (k=%lld rows=%lld)\n", (long long) k,
                          (long long) rows);
        }
        return false;
    }

    const ifairy_lut_index_cache_key key = {
        /* .data   = */ tensor->data,
        /* .nbytes = */ ggml_nbytes(tensor),
        /* .k      = */ k,
        /* .rows   = */ rows,
    };

    const int64_t blocks_per_row = k / QK_IFAIRY;
    const int64_t tiles          = (rows + 15) / 16;
    const size_t  packed_bytes   = (size_t) tiles * (size_t) blocks_per_row * sizeof(struct ifairy_lut_wtile_16);

    // Layout within a single cached buffer:
    //   [indexes (padded to tile alignment)] [packed_wtiles]
    const size_t index_bytes_aligned = GGML_PAD(index_bytes, GGML_IFAIRY_LUT_WTILE_ALIGNMENT);
    const size_t total_bytes         = index_bytes_aligned + packed_bytes;

    {
        std::lock_guard<std::mutex> lock(g_ifairy_lut_mutex);
        extra = (ifairy_lut_extra *) tensor->extra;
        if (extra && extra->indexes && extra->packed_w) {
            if (index_tensor_out) {
                *index_tensor_out = NULL;
            }
            return true;
        }

        const auto it = g_ifairy_lut_index_cache.find(key);
        if (it != g_ifairy_lut_index_cache.end() && it->second.base && it->second.size == total_bytes) {
            if (!extra) {
                extra         = new ifairy_lut_extra;
                tensor->extra = extra;
                g_ifairy_lut_extras.push_back(extra);
            }

            extra->indexes       = it->second.base;
            extra->size          = index_bytes;
            extra->packed_w      = it->second.base + index_bytes_aligned;
            extra->packed_w_size = packed_bytes;
            extra->index_tensor  = NULL;
            extra->index_buffer  = it->second.buffer;

            if (index_tensor_out) {
                *index_tensor_out = NULL;
            }
            return true;
        }
    }

    ggml_backend_buffer_t index_buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), total_bytes);
    uint8_t *             buf = index_buffer ? (uint8_t *) ggml_backend_buffer_get_base(index_buffer) : nullptr;
    if (index_buffer) {
        ggml_backend_buffer_set_usage(index_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }

    // fallback (shouldn't normally happen, but keeps LUT usable under allocation failures)
    if (!buf) {
        if (index_buffer) {
            ggml_backend_buffer_free(index_buffer);
            index_buffer = nullptr;
        }
        buf = (uint8_t *) ggml_aligned_malloc(total_bytes);
        if (!buf) {
            if (dbg) {
                GGML_LOG_WARN("ifairy_lut: transform_tensor: allocation failed (bytes=%zu)\n", total_bytes);
            }
            return false;
        }
    }

    memset(buf, 0, total_bytes);

    uint8_t *                    indexes  = buf;
    struct ifairy_lut_wtile_16 * packed_w = (struct ifairy_lut_wtile_16 *) (buf + index_bytes_aligned);

    const bool ok = ggml_ifairy_2w_encode((const block_ifairy *) tensor->data, k, rows, indexes, index_bytes);
    if (!ok) {
        if (dbg) {
            GGML_LOG_WARN("ifairy_lut: transform_tensor: ggml_ifairy_2w_encode failed (bytes=%zu)\n", index_bytes);
        }
        if (index_buffer) {
            ggml_backend_buffer_free(index_buffer);
        } else {
            ggml_aligned_free(buf, total_bytes);
        }
        return false;
    }

    // Build packed 16-lane weights from the per-row indexes.
    // 2-weight encoding: 4-bit pattern is the direct LUT index.
    const block_ifairy * w_blocks = (const block_ifairy *) tensor->data;
    for (int64_t row = 0; row < rows; ++row) {
        const int64_t tile = row >> 4;
        const int64_t lane = row & 15;

        const uint8_t * row_indexes =
            indexes + (size_t) row * (size_t) blocks_per_row * (size_t) QK_IFAIRY_GROUPS_PER_BLOCK;

        for (int64_t blk = 0; blk < blocks_per_row; ++blk) {
            struct ifairy_lut_wtile_16 * t = packed_w + (size_t) tile * (size_t) blocks_per_row + (size_t) blk;

            const block_ifairy * wb = w_blocks + (size_t) row * (size_t) blocks_per_row + (size_t) blk;
            t->d_real[lane]         = GGML_FP16_TO_FP32(wb->d_real);
            t->d_imag[lane]         = GGML_FP16_TO_FP32(wb->d_imag);

            const uint8_t * blk_idx = row_indexes + (size_t) blk * (size_t) QK_IFAIRY_GROUPS_PER_BLOCK;
            for (int gi = 0; gi < QK_IFAIRY_GROUPS_PER_BLOCK; gi += 2) {
                const uint8_t lo    = blk_idx[gi + 0] & 0x0fu;
                const uint8_t hi    = blk_idx[gi + 1] & 0x0fu;
                t->qs[gi / 2][lane] = lo | (uint8_t) (hi << 4);
            }
        }
    }

    {
        std::lock_guard<std::mutex> lock(g_ifairy_lut_mutex);
        extra = (ifairy_lut_extra *) tensor->extra;
        if (extra && extra->indexes && extra->packed_w) {
            // Another thread finished while we were encoding; discard our buffer.
            if (index_buffer) {
                ggml_backend_buffer_free(index_buffer);
            } else {
                ggml_aligned_free(buf, total_bytes);
            }
            if (index_tensor_out) {
                *index_tensor_out = NULL;
            }
            return true;
        }

        const auto it = g_ifairy_lut_index_cache.find(key);
        if (it != g_ifairy_lut_index_cache.end() && it->second.base && it->second.size == total_bytes) {
            // Another thread populated the cache meanwhile; reuse it and free ours.
            if (index_buffer) {
                ggml_backend_buffer_free(index_buffer);
            } else {
                ggml_aligned_free(buf, total_bytes);
            }
            index_buffer = it->second.buffer;
            buf          = it->second.base;
        } else if (index_buffer) {
            g_ifairy_lut_index_cache.emplace(key, ifairy_lut_index_cache_entry{
                                                      /* .buffer = */ index_buffer,
                                                      /* .base   = */ buf,
                                                      /* .size   = */ total_bytes,
                                                  });
        }

        if (!extra) {
            extra         = new ifairy_lut_extra;
            tensor->extra = extra;
            g_ifairy_lut_extras.push_back(extra);
        }

        extra->indexes       = buf;
        extra->size          = index_bytes;
        extra->packed_w      = buf + index_bytes_aligned;
        extra->packed_w_size = packed_bytes;
        extra->index_tensor  = NULL;
        extra->index_buffer  = index_buffer;
    }

    if (index_tensor_out) {
        *index_tensor_out = NULL;
    }
    return true;
}
