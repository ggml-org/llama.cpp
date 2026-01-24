#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <sycl/sycl.hpp>

#include "ggml.h"
#include "ggml-quants.h"
#include "ggml-sycl/common.hpp"

#define GGML_COMMON_DECL_CPP
#include "ggml-common.h"

namespace sycl_bench {

static inline int tile_count(int blocks) {
    int count = 0;
    int remaining = blocks;
    while (remaining > 0) {
        int ts = 1;
        while (ts * 2 <= remaining && ts < 32) {
            ts *= 2;
        }
        remaining -= ts;
        count++;
    }
    return count;
}

static inline int tile_size_at(int blocks, int tile_idx) {
    int remaining = blocks;
    for (int i = 0; i <= tile_idx && remaining > 0; i++) {
        int ts = 1;
        while (ts * 2 <= remaining && ts < 32) {
            ts *= 2;
        }
        if (i == tile_idx) {
            return ts;
        }
        remaining -= ts;
    }
    return 0;
}

struct GeneratedWeights {
    std::vector<uint8_t> aos;
    std::vector<uint8_t> layout;
    std::vector<float>   fp32;
    ggml_layout_mode     layout_mode = GGML_LAYOUT_AOS;
    size_t               bytes_aos = 0;
    size_t               bytes_layout = 0;
};

struct GeneratedActivations {
    std::vector<uint8_t>    q8_1;
    std::vector<float>      fp32;
    std::vector<ggml_half>  fp16;
    int64_t                 k_padded = 0;
};

static inline int64_t padded_k(int64_t k, int64_t pad) {
    return GGML_PAD(k, pad);
}

static inline void fill_random(float * data, size_t count, std::mt19937 & rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < count; ++i) {
        data[i] = dist(rng);
    }
}

static inline void pack_half2(uint8_t * dst, float x, float y) {
    const sycl::half2 h2{ sycl::half(x), sycl::half(y) };
    std::memcpy(dst, &h2, sizeof(h2));
}

static inline void quantize_q8_1_block_ref(const float * src, int8_t * qs, float & d, float & s) {
    float amax = 0.0f;
    for (int i = 0; i < QK8_1; ++i) {
        amax = std::max(amax, std::fabs(src[i]));
    }
    d = amax / 127.0f;
    const float id = (d != 0.0f) ? 1.0f / d : 0.0f;
    int sum = 0;
    for (int i = 0; i < QK8_1; ++i) {
        const float v = src[i] * id;
        const int8_t q = static_cast<int8_t>(std::round(v));
        qs[i] = q;
        sum += q;
    }
    s = static_cast<float>(sum) * d;
}

static inline void quantize_q8_1_row_aos(const float * src, uint8_t * dst, int64_t k, int64_t k_padded) {
    const int64_t blocks = k / QK8_1;
    const int64_t pitch = k_padded / QK8_1;
    std::memset(dst, 0, (size_t) pitch * sizeof(block_q8_1));

    for (int64_t b = 0; b < blocks; ++b) {
        const float * block_src = src + b * QK8_1;
        block_q8_1 * block_dst = reinterpret_cast<block_q8_1 *>(dst) + b;
        float d = 0.0f;
        float s = 0.0f;
        quantize_q8_1_block_ref(block_src, block_dst->qs, d, s);
        pack_half2(reinterpret_cast<uint8_t *>(&block_dst->ds), d, s);
    }
}

static inline void quantize_q8_1_row_soa(const float * src, uint8_t * dst, int64_t k, int64_t k_padded) {
    const int64_t blocks = k / QK8_1;
    const size_t row_bytes = (size_t) (k_padded / QK8_1) * sizeof(block_q8_1);
    std::memset(dst, 0, row_bytes);

    int8_t * qs_dst = reinterpret_cast<int8_t *>(dst);
    uint8_t * ds_dst = dst + k;

    for (int64_t b = 0; b < blocks; ++b) {
        const float * block_src = src + b * QK8_1;
        float d = 0.0f;
        float s = 0.0f;
        quantize_q8_1_block_ref(block_src, qs_dst + b * QK8_1, d, s);
        pack_half2(ds_dst + b * sizeof(sycl::half2), d, s);
    }
}

static inline bool requires_block_alignment(ggml_type type) {
    return (ggml_blck_size(type) > 1);
}

// === CPU-side layout helpers (copied from ggml-sycl.cpp for staging) ===
static void reorder_q4_0_cpu(void * dst_soa, const void * src_aos, int ncols, int nrows) {
    const size_t blocks_per_row = ncols / QK4_0;
    const size_t nblocks        = blocks_per_row * nrows;
    const uint8_t * aos    = (const uint8_t *) src_aos;
    uint8_t *       soa_qs = (uint8_t *) dst_soa;
    uint8_t *       soa_d  = soa_qs + nblocks * (QK4_0 / 2);
    for (size_t ib = 0; ib < nblocks; ib++) {
        const uint8_t * block_aos = aos + ib * sizeof(block_q4_0);
        memcpy(soa_qs + ib * (QK4_0 / 2), block_aos + sizeof(ggml_half), QK4_0 / 2);
        memcpy(soa_d + ib * sizeof(ggml_half), block_aos, sizeof(ggml_half));
    }
}

static bool reorder_q4_0_coalesced_cpu(void * dst_coalesced, const void * src_aos, int ncols, int nrows) {
    constexpr int TILE_BLOCKS       = MMVQ_COALESCED_TILE_BLOCKS;
    constexpr int BYTES_PER_BLOCK   = QK4_0 / 2;
    constexpr int WORDS_PER_BLOCK   = 4;
    constexpr int WORD_PLANE_STRIDE = TILE_BLOCKS * 4;

    const size_t blocks_per_row = ncols / QK4_0;
    const size_t nblocks        = blocks_per_row * nrows;
    const size_t row_quants_bytes   = ncols / 2;
    const size_t total_quants_bytes = nrows * row_quants_bytes;

    const uint8_t * aos = (const uint8_t *) src_aos;
    uint8_t * coal_qs = (uint8_t *) dst_coalesced;
    uint8_t * coal_d  = coal_qs + total_quants_bytes;

    if (blocks_per_row % TILE_BLOCKS != 0) {
        std::fprintf(stderr, "[sycl-kernel-bench] Q4_0 coalesced: blocks_per_row=%zu not divisible by %d\n",
                     blocks_per_row, TILE_BLOCKS);
        return false;
    }

    for (size_t ib = 0; ib < nblocks; ib++) {
        const uint8_t * block_aos = aos + ib * sizeof(block_q4_0);
        const uint8_t * src_qs    = block_aos + sizeof(ggml_half);
        const size_t row           = ib / blocks_per_row;
        const size_t col_block     = ib % blocks_per_row;
        const size_t tile          = col_block / TILE_BLOCKS;
        const size_t block_in_tile = col_block % TILE_BLOCKS;
        const size_t tile_base = row * row_quants_bytes + tile * (TILE_BLOCKS * BYTES_PER_BLOCK);

        for (int word = 0; word < WORDS_PER_BLOCK; word++) {
            const size_t word_offset = tile_base + word * WORD_PLANE_STRIDE + block_in_tile * 4;
            memcpy(coal_qs + word_offset, src_qs + word * 4, 4);
        }
        memcpy(coal_d + ib * sizeof(ggml_half), block_aos, sizeof(ggml_half));
    }
    return true;
}

static void reorder_q8_0_cpu(void * dst_soa, const void * src_aos, int ncols, int nrows) {
    const size_t blocks_per_row = ncols / QK8_0;
    const size_t nblocks        = blocks_per_row * nrows;
    const uint8_t * aos    = (const uint8_t *) src_aos;
    uint8_t *       soa_qs = (uint8_t *) dst_soa;
    uint8_t *       soa_d  = soa_qs + nblocks * QK8_0;
    for (size_t ib = 0; ib < nblocks; ib++) {
        const uint8_t * block_aos = aos + ib * sizeof(block_q8_0);
        memcpy(soa_qs + ib * QK8_0, block_aos + sizeof(ggml_half), QK8_0);
        memcpy(soa_d + ib * sizeof(ggml_half), block_aos, sizeof(ggml_half));
    }
}

static bool reorder_q8_0_coalesced_cpu(void * dst_coalesced, const void * src_aos, int ncols, int nrows) {
    constexpr int TILE_BLOCKS       = MMVQ_COALESCED_TILE_BLOCKS;
    constexpr int BYTES_PER_BLOCK   = QK8_0;
    constexpr int WORDS_PER_BLOCK   = 8;
    constexpr int WORD_PLANE_STRIDE = TILE_BLOCKS * 4;

    const size_t blocks_per_row = ncols / QK8_0;
    const size_t nblocks        = blocks_per_row * nrows;
    const size_t row_quants_bytes   = ncols;
    const size_t total_quants_bytes = nrows * row_quants_bytes;

    const uint8_t * aos = (const uint8_t *) src_aos;
    uint8_t * coal_qs = (uint8_t *) dst_coalesced;
    uint8_t * coal_d  = coal_qs + total_quants_bytes;

    if (blocks_per_row % TILE_BLOCKS != 0) {
        std::fprintf(stderr, "[sycl-kernel-bench] Q8_0 coalesced: blocks_per_row=%zu not divisible by %d\n",
                     blocks_per_row, TILE_BLOCKS);
        return false;
    }

    for (size_t ib = 0; ib < nblocks; ib++) {
        const uint8_t * block_aos = aos + ib * sizeof(block_q8_0);
        const uint8_t * src_qs    = block_aos + sizeof(ggml_half);
        const size_t row           = ib / blocks_per_row;
        const size_t col_block     = ib % blocks_per_row;
        const size_t tile          = col_block / TILE_BLOCKS;
        const size_t block_in_tile = col_block % TILE_BLOCKS;
        const size_t tile_base = row * row_quants_bytes + tile * (TILE_BLOCKS * BYTES_PER_BLOCK);

        for (int word = 0; word < WORDS_PER_BLOCK; word++) {
            const size_t word_offset = tile_base + word * WORD_PLANE_STRIDE + block_in_tile * 4;
            memcpy(coal_qs + word_offset, src_qs + word * 4, 4);
        }
        memcpy(coal_d + ib * sizeof(ggml_half), block_aos, sizeof(ggml_half));
    }
    return true;
}

static void reorder_q6_k_cpu(void * dst_soa, const void * src_aos, size_t nblocks) {
    const uint8_t * aos        = (const uint8_t *) src_aos;
    uint8_t *       soa_ql     = (uint8_t *) dst_soa;
    uint8_t *       soa_qh     = soa_ql + nblocks * (QK_K / 2);
    uint8_t *       soa_scales = soa_qh + nblocks * (QK_K / 4);
    uint8_t *       soa_d      = soa_scales + nblocks * (QK_K / 16);

    for (size_t ib = 0; ib < nblocks; ib++) {
        const uint8_t * block_aos = aos + ib * sizeof(block_q6_K);
        memcpy(soa_ql + ib * (QK_K / 2), block_aos, QK_K / 2);
        memcpy(soa_qh + ib * (QK_K / 4), block_aos + (QK_K / 2), QK_K / 4);
        memcpy(soa_scales + ib * (QK_K / 16), block_aos + (QK_K / 2) + (QK_K / 4), QK_K / 16);
        memcpy(soa_d + ib * sizeof(ggml_half), block_aos + (QK_K / 2) + (QK_K / 4) + (QK_K / 16), sizeof(ggml_half));
    }
}

static void reorder_q2_k_cpu(void * dst_soa, const void * src_aos, size_t nblocks) {
    const uint8_t * aos           = (const uint8_t *) src_aos;
    uint8_t *       soa_qs        = (uint8_t *) dst_soa;
    uint8_t *       soa_scales    = soa_qs + nblocks * (QK_K / 4);
    uint8_t *       soa_dm        = soa_scales + nblocks * (QK_K / 16);

    for (size_t ib = 0; ib < nblocks; ib++) {
        const uint8_t * block_aos = aos + ib * sizeof(block_q2_K);
        memcpy(soa_scales + ib * (QK_K / 16), block_aos, QK_K / 16);
        memcpy(soa_qs + ib * (QK_K / 4), block_aos + (QK_K / 16), QK_K / 4);
        memcpy(soa_dm + ib * sizeof(ggml_half2), block_aos + (QK_K / 16) + (QK_K / 4), sizeof(ggml_half2));
    }
}

static void reorder_q3_k_cpu(void * dst_soa, const void * src_aos, size_t nblocks) {
    const uint8_t * aos        = (const uint8_t *) src_aos;
    uint8_t *       soa_qs     = (uint8_t *) dst_soa;
    uint8_t *       soa_hmask  = soa_qs + nblocks * (QK_K / 4);
    uint8_t *       soa_scales = soa_hmask + nblocks * (QK_K / 8);
    uint8_t *       soa_d      = soa_scales + nblocks * 12;

    for (size_t ib = 0; ib < nblocks; ib++) {
        const uint8_t * block_aos = aos + ib * sizeof(block_q3_K);
        memcpy(soa_hmask + ib * (QK_K / 8), block_aos, QK_K / 8);
        memcpy(soa_qs + ib * (QK_K / 4), block_aos + (QK_K / 8), QK_K / 4);
        memcpy(soa_scales + ib * 12, block_aos + (QK_K / 8) + (QK_K / 4), 12);
        memcpy(soa_d + ib * sizeof(ggml_half), block_aos + (QK_K / 8) + (QK_K / 4) + 12, sizeof(ggml_half));
    }
}

static void reorder_q5_k_cpu(void * dst_soa, const void * src_aos, size_t nblocks) {
    const uint8_t * aos        = (const uint8_t *) src_aos;
    uint8_t *       soa_qs     = (uint8_t *) dst_soa;
    uint8_t *       soa_qh     = soa_qs + nblocks * (QK_K / 2);
    uint8_t *       soa_scales = soa_qh + nblocks * (QK_K / 8);
    uint8_t *       soa_dm     = soa_scales + nblocks * K_SCALE_SIZE;

    for (size_t ib = 0; ib < nblocks; ib++) {
        const uint8_t * block_aos = aos + ib * sizeof(block_q5_K);
        memcpy(soa_scales + ib * K_SCALE_SIZE, block_aos + sizeof(ggml_half2), K_SCALE_SIZE);
        memcpy(soa_qh + ib * (QK_K / 8), block_aos + sizeof(ggml_half2) + K_SCALE_SIZE, QK_K / 8);
        memcpy(soa_qs + ib * (QK_K / 2), block_aos + sizeof(ggml_half2) + K_SCALE_SIZE + (QK_K / 8), QK_K / 2);
        memcpy(soa_dm + ib * sizeof(ggml_half2), block_aos, sizeof(ggml_half2));
    }
}

static bool reorder_q6_k_coalesced_cpu(void * dst_coalesced, const void * src_aos, int ncols, int nrows) {
    const int blocks_per_row = ncols / QK_K;
    const size_t nblocks     = (size_t) blocks_per_row * nrows;
    (void) nblocks;
    const int num_tiles      = tile_count(blocks_per_row);

    size_t row_quants_bytes = 0;
    for (int t = 0; t < num_tiles; t++) {
        const int ts = tile_size_at(blocks_per_row, t);
        row_quants_bytes += (size_t) ts * (128 + 64 + 16);
    }

    const uint8_t * aos = (const uint8_t *) src_aos;
    uint8_t * coal_d = (uint8_t *) dst_coalesced + (size_t) nrows * row_quants_bytes;

    for (int row = 0; row < nrows; row++) {
        uint8_t * row_dst = (uint8_t *) dst_coalesced + row * row_quants_bytes;
        int block_idx = 0;
        for (int tile = 0; tile < num_tiles; tile++) {
            const int tile_size = tile_size_at(blocks_per_row, tile);
            const int word_plane_stride = tile_size * 4;
            uint8_t * tile_ql = row_dst;
            uint8_t * tile_qh = tile_ql + tile_size * 128;
            uint8_t * tile_sc = tile_qh + tile_size * 64;

            for (int b = 0; b < tile_size; b++) {
                const size_t global_block = (size_t) row * blocks_per_row + block_idx + b;
                const uint8_t * block_aos = aos + global_block * sizeof(block_q6_K);

                const uint8_t * src_ql = block_aos;
                for (int word = 0; word < 32; word++) {
                    memcpy(tile_ql + word * word_plane_stride + b * 4, src_ql + word * 4, 4);
                }

                const uint8_t * src_qh = block_aos + 128;
                for (int word = 0; word < 16; word++) {
                    memcpy(tile_qh + word * word_plane_stride + b * 4, src_qh + word * 4, 4);
                }

                const uint8_t * src_sc = block_aos + 128 + 64;
                for (int word = 0; word < 4; word++) {
                    memcpy(tile_sc + word * word_plane_stride + b * 4, src_sc + word * 4, 4);
                }

                const uint8_t * src_d = block_aos + 128 + 64 + 16;
                memcpy(coal_d + global_block * sizeof(ggml_half), src_d, sizeof(ggml_half));
            }

            row_dst += tile_size * (128 + 64 + 16);
            block_idx += tile_size;
        }
    }
    return true;
}

static void reorder_mxfp4_cpu(void * dst_soa, const void * src_aos, int ncols, int nrows) {
    const size_t blocks_per_row = ncols / QK_MXFP4;
    const size_t nblocks        = blocks_per_row * nrows;
    const uint8_t * aos    = (const uint8_t *) src_aos;
    uint8_t *       soa_qs = (uint8_t *) dst_soa;
    uint8_t *       soa_e  = soa_qs + nblocks * (QK_MXFP4 / 2);

    for (size_t ib = 0; ib < nblocks; ib++) {
        const block_mxfp4 * block = (const block_mxfp4 *) (aos + ib * sizeof(block_mxfp4));
        memcpy(soa_qs + ib * (QK_MXFP4 / 2), block->qs, QK_MXFP4 / 2);
        soa_e[ib] = block->e;
    }
}

static bool reorder_mxfp4_coalesced_cpu(void * dst_coalesced, const void * src_aos, int ncols, int nrows) {
    constexpr int TILE_BLOCKS       = MMVQ_COALESCED_TILE_BLOCKS;
    constexpr int BYTES_PER_BLOCK   = QK_MXFP4 / 2;
    constexpr int WORDS_PER_BLOCK   = 4;
    constexpr int WORD_PLANE_STRIDE = TILE_BLOCKS * 4;

    const size_t blocks_per_row = ncols / QK_MXFP4;
    const size_t nblocks        = blocks_per_row * nrows;
    const size_t row_quants_bytes   = ncols / 2;
    const size_t total_quants_bytes = nrows * row_quants_bytes;

    const uint8_t * aos = (const uint8_t *) src_aos;
    uint8_t * coal_qs = (uint8_t *) dst_coalesced;
    uint8_t * coal_e  = coal_qs + total_quants_bytes;

    if (blocks_per_row % TILE_BLOCKS != 0) {
        std::fprintf(stderr, "[sycl-kernel-bench] MXFP4 coalesced: blocks_per_row=%zu not divisible by %d\n",
                     blocks_per_row, TILE_BLOCKS);
        return false;
    }

    for (size_t ib = 0; ib < nblocks; ib++) {
        const uint8_t * block_aos = aos + ib * sizeof(block_mxfp4);
        const uint8_t * src_qs    = block_aos + 1;
        const size_t row           = ib / blocks_per_row;
        const size_t col_block     = ib % blocks_per_row;
        const size_t tile          = col_block / TILE_BLOCKS;
        const size_t block_in_tile = col_block % TILE_BLOCKS;
        const size_t tile_base = row * row_quants_bytes + tile * (TILE_BLOCKS * BYTES_PER_BLOCK);

        for (int word = 0; word < WORDS_PER_BLOCK; word++) {
            const size_t word_offset = tile_base + word * WORD_PLANE_STRIDE + block_in_tile * 4;
            memcpy(coal_qs + word_offset, src_qs + word * 4, 4);
        }
        coal_e[ib] = block_aos[0];
    }
    return true;
}

// Generate quantized weights (AoS), optionally plus layout-specific buffers.
static inline bool generate_quantized_weights(const ggml_type type,
                                              const ggml_layout_mode layout_mode,
                                              int64_t m,
                                              int64_t k,
                                              bool keep_fp32,
                                              GeneratedWeights & out) {
    if (requires_block_alignment(type) && (k % ggml_blck_size(type) != 0)) {
        std::fprintf(stderr, "[sycl-kernel-bench] Unsupported K=%lld for type %d (block size=%lld)\n",
                     (long long) k, type, (long long) ggml_blck_size(type));
        return false;
    }

    const size_t row_size = ggml_row_size(type, k);
    const size_t total_bytes = row_size * m;

    out.aos.assign(total_bytes, 0);
    out.bytes_aos = total_bytes;
    out.layout_mode = layout_mode;

    std::mt19937 rng(42);
    std::vector<float> row_fp32((size_t) k);

    if (keep_fp32) {
        out.fp32.resize((size_t) m * k);
    }

    for (int64_t row = 0; row < m; ++row) {
        fill_random(row_fp32.data(), (size_t) k, rng);
        if (keep_fp32) {
            memcpy(out.fp32.data() + row * k, row_fp32.data(), (size_t) k * sizeof(float));
        }
        uint8_t * row_dst = out.aos.data() + row * row_size;
        ggml_quantize_chunk(type, row_fp32.data(), row_dst, 0, 1, k, nullptr);
    }

    if (layout_mode == GGML_LAYOUT_AOS) {
        out.layout.clear();
        out.bytes_layout = 0;
        return true;
    }

    // Prepare layout buffer
    size_t layout_bytes = 0;
    if (layout_mode == GGML_LAYOUT_SOA) {
        layout_bytes = ggml_row_size(type, k) * m;
    } else if (layout_mode == GGML_LAYOUT_COALESCED) {
        layout_bytes = ggml_row_size(type, k) * m; // coarse allocation; may differ for Q6_K
        if (type == GGML_TYPE_Q6_K) {
            const int blocks_per_row = (int) (k / QK_K);
            const int num_tiles = tile_count(blocks_per_row);
            size_t row_quants_bytes = 0;
            for (int t = 0; t < num_tiles; t++) {
                const int ts = tile_size_at(blocks_per_row, t);
                row_quants_bytes += (size_t) ts * (128 + 64 + 16);
            }
            const size_t nblocks = (size_t) blocks_per_row * m;
            layout_bytes = (size_t) m * row_quants_bytes + nblocks * sizeof(ggml_half);
        }
    }

    out.layout.assign(layout_bytes, 0);
    out.bytes_layout = layout_bytes;

    switch (layout_mode) {
        case GGML_LAYOUT_SOA:
            switch (type) {
                case GGML_TYPE_Q4_0: reorder_q4_0_cpu(out.layout.data(), out.aos.data(), (int) k, (int) m); break;
                case GGML_TYPE_Q8_0: reorder_q8_0_cpu(out.layout.data(), out.aos.data(), (int) k, (int) m); break;
                case GGML_TYPE_Q6_K: reorder_q6_k_cpu(out.layout.data(), out.aos.data(), (size_t) (m * (k / QK_K))); break;
                case GGML_TYPE_MXFP4: reorder_mxfp4_cpu(out.layout.data(), out.aos.data(), (int) k, (int) m); break;
                case GGML_TYPE_Q4_K: {
                    const size_t nblocks = (size_t) (m * (k / QK_K));
                    const uint8_t * aos = out.aos.data();
                    uint8_t * soa_qs = out.layout.data();
                    uint8_t * soa_scales = soa_qs + nblocks * (QK_K / 2);
                    uint8_t * soa_dm = soa_scales + nblocks * K_SCALE_SIZE;
                    for (size_t ib = 0; ib < nblocks; ib++) {
                        const uint8_t * block_aos = aos + ib * sizeof(block_q4_K);
                        memcpy(soa_qs + ib * (QK_K / 2), block_aos + 4 + K_SCALE_SIZE, QK_K / 2);
                        memcpy(soa_scales + ib * K_SCALE_SIZE, block_aos + 4, K_SCALE_SIZE);
                        memcpy(soa_dm + ib * 4, block_aos, 4);
                    }
                    break;
                }
                case GGML_TYPE_Q2_K: reorder_q2_k_cpu(out.layout.data(), out.aos.data(), (size_t) (m * (k / QK_K))); break;
                case GGML_TYPE_Q3_K: reorder_q3_k_cpu(out.layout.data(), out.aos.data(), (size_t) (m * (k / QK_K))); break;
                case GGML_TYPE_Q5_K: reorder_q5_k_cpu(out.layout.data(), out.aos.data(), (size_t) (m * (k / QK_K))); break;
                default:
                    std::fprintf(stderr, "[sycl-kernel-bench] SoA layout unsupported for type %d\n", type);
                    return false;
            }
            break;

        case GGML_LAYOUT_COALESCED:
            switch (type) {
                case GGML_TYPE_Q4_0: return reorder_q4_0_coalesced_cpu(out.layout.data(), out.aos.data(), (int) k, (int) m);
                case GGML_TYPE_Q8_0: return reorder_q8_0_coalesced_cpu(out.layout.data(), out.aos.data(), (int) k, (int) m);
                case GGML_TYPE_Q6_K: return reorder_q6_k_coalesced_cpu(out.layout.data(), out.aos.data(), (int) k, (int) m);
                case GGML_TYPE_MXFP4: return reorder_mxfp4_coalesced_cpu(out.layout.data(), out.aos.data(), (int) k, (int) m);
                default:
                    std::fprintf(stderr, "[sycl-kernel-bench] Coalesced layout unsupported for type %d\n", type);
                    return false;
            }
        default:
            break;
    }

    return true;
}

static inline GeneratedActivations generate_activations(int64_t batch,
                                                        int64_t k,
                                                        int64_t k_padded,
                                                        bool keep_fp32,
                                                        bool keep_fp16,
                                                        bool use_soa_layout) {
    GeneratedActivations out;
    out.k_padded = k_padded;
    const size_t row_bytes = (size_t) (k_padded / QK8_1) * sizeof(block_q8_1);
    out.q8_1.resize((size_t) batch * row_bytes);
    if (keep_fp32) {
        out.fp32.resize((size_t) (batch * k_padded));
    }
    if (keep_fp16) {
        out.fp16.resize((size_t) (batch * k_padded));
    }

    std::mt19937 rng(42);
    std::vector<float> row_fp32((size_t) k_padded);

    for (int64_t row = 0; row < batch; ++row) {
        fill_random(row_fp32.data(), (size_t) k, rng);
        if (k_padded > k) {
            std::fill(row_fp32.begin() + k, row_fp32.end(), 0.0f);
        }
        if (keep_fp32) {
            memcpy(out.fp32.data() + row * k_padded, row_fp32.data(), (size_t) k_padded * sizeof(float));
        }
        if (keep_fp16) {
            for (int64_t i = 0; i < k_padded; ++i) {
                out.fp16[(size_t) row * k_padded + (size_t) i] = ggml_fp32_to_fp16(row_fp32[(size_t) i]);
            }
        }
        uint8_t * row_dst = out.q8_1.data() + (size_t) row * row_bytes;
        if (use_soa_layout) {
            quantize_q8_1_row_soa(row_fp32.data(), row_dst, k, k_padded);
        } else {
            quantize_q8_1_row_aos(row_fp32.data(), row_dst, k, k_padded);
        }
    }

    return out;
}

}  // namespace sycl_bench
