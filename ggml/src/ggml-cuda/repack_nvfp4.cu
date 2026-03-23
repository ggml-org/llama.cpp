#include <cstring>

#include "ggml.h"
#include "repack_nvfp4.cuh"

static void ggml_cuda_repack_row_nvfp4(const block_nvfp4 * src, uint8_t * dst, int64_t ne0) {
    GGML_ASSERT(ne0 % QK_NVFP4 == 0);

    const int lanes_per_cuda_block = QK_K / QK_NVFP4;
    const int64_t n_upstream_blocks = ne0 / QK_NVFP4;
    const int64_t n_cuda_blocks = n_upstream_blocks / lanes_per_cuda_block;
    const int64_t tail_lanes = n_upstream_blocks % lanes_per_cuda_block;

    for (int64_t ib = 0; ib < n_cuda_blocks; ++ib) {
        const block_nvfp4 * in = src + ib * lanes_per_cuda_block;
        uint8_t * out = dst + ib * sizeof(block_nvfp4_cuda);

        for (int lane = 0; lane < lanes_per_cuda_block; ++lane) {
            uint8_t * out_qs = out + lane * sizeof(in[lane].qs);
            uint8_t * out_scales = out + lanes_per_cuda_block * sizeof(in[lane].qs) + lane * sizeof(in[lane].d);

            for (int pack = 0; pack < 8; ++pack) {
                const uint32_t packed = ggml_cuda_nvfp4_pack(in[lane].qs, pack);
                memcpy(out_qs + pack * sizeof(packed), &packed, sizeof(packed));
            }
            memcpy(out_scales, in[lane].d, sizeof(in[lane].d));
        }
    }

    if (tail_lanes > 0) {
        const block_nvfp4 * in_tail = src + n_cuda_blocks * lanes_per_cuda_block;
        uint8_t * tail = dst + n_cuda_blocks * sizeof(block_nvfp4_cuda); // Last short block stays compact too

        for (int64_t lane = 0; lane < tail_lanes; ++lane) {
            uint8_t * tail_qs = tail + lane * sizeof(in_tail[lane].qs);
            uint8_t * tail_scales = tail + tail_lanes * sizeof(in_tail[lane].qs) + lane * sizeof(in_tail[lane].d);

            for (int pack = 0; pack < 8; ++pack) {
                const uint32_t packed = ggml_cuda_nvfp4_pack(in_tail[lane].qs, pack);
                memcpy(tail_qs + pack * sizeof(packed), &packed, sizeof(packed));
            }
            memcpy(tail_scales, in_tail[lane].d, sizeof(in_tail[lane].d));
        }
    }
}

static void ggml_cuda_unpack_weights_nvfp4(const uint8_t * src, uint8_t * dst) {
    for (int scale = 0; scale < 4; ++scale) {
        uint32_t packed_lo;
        uint32_t packed_hi;
        memcpy(&packed_lo, src + (scale * 2 + 0) * sizeof(packed_lo), sizeof(packed_lo));
        memcpy(&packed_hi, src + (scale * 2 + 1) * sizeof(packed_hi), sizeof(packed_hi));

        for (int value = 0; value < 8; ++value) {
            dst[scale * 8 + value] =
                    ggml_cuda_nvfp4_unpack(packed_lo, value) |
                    (ggml_cuda_nvfp4_unpack(packed_hi, value) << 4);
        }
    }
}

static void ggml_cuda_unpack_row_nvfp4(const uint8_t * src, block_nvfp4 * dst, int64_t ne0) {
    GGML_ASSERT(ne0 % QK_NVFP4 == 0);

    const int lanes_per_cuda_block = QK_K / QK_NVFP4;
    const int64_t n_upstream_blocks = ne0 / QK_NVFP4;
    const int64_t n_cuda_blocks = n_upstream_blocks / lanes_per_cuda_block;
    const int64_t tail_lanes = n_upstream_blocks % lanes_per_cuda_block;

    for (int64_t ib = 0; ib < n_cuda_blocks; ++ib) {
        const uint8_t * in = src + ib * sizeof(block_nvfp4_cuda);
        block_nvfp4 * out = dst + ib * lanes_per_cuda_block;

        for (int lane = 0; lane < lanes_per_cuda_block; ++lane) {
            const uint8_t * in_qs = in + lane * sizeof(out[lane].qs);
            const uint8_t * in_scales = in + lanes_per_cuda_block * sizeof(out[lane].qs) + lane * sizeof(out[lane].d);

            ggml_cuda_unpack_weights_nvfp4(in_qs, out[lane].qs);
            memcpy(out[lane].d, in_scales, sizeof(out[lane].d));
        }
    }

    if (tail_lanes > 0) {
        const uint8_t * tail = src + n_cuda_blocks * sizeof(block_nvfp4_cuda);
        block_nvfp4 * out_tail = dst + n_cuda_blocks * lanes_per_cuda_block; // Same compact tail on unpack

        for (int64_t lane = 0; lane < tail_lanes; ++lane) {
            const uint8_t * tail_qs = tail + lane * sizeof(out_tail[lane].qs);
            const uint8_t * tail_scales = tail + tail_lanes * sizeof(out_tail[lane].qs) + lane * sizeof(out_tail[lane].d);

            ggml_cuda_unpack_weights_nvfp4(tail_qs, out_tail[lane].qs);
            memcpy(out_tail[lane].d, tail_scales, sizeof(out_tail[lane].d));
        }
    }
}

void ggml_cuda_repack_rows_nvfp4(int64_t ne0, int64_t nrows, const void * src, void * dst) {
    const size_t row_size = ggml_row_size(GGML_TYPE_NVFP4, ne0);

    for (int64_t row = 0; row < nrows; ++row) {
        ggml_cuda_repack_row_nvfp4((const block_nvfp4 *) ((const uint8_t *) src + row * row_size), (uint8_t *) dst + row * row_size, ne0);
    }
}

void ggml_cuda_unpack_rows_nvfp4(int64_t ne0, int64_t nrows, const void * src, void * dst) {
    const size_t row_size = ggml_row_size(GGML_TYPE_NVFP4, ne0);

    for (int64_t row = 0; row < nrows; ++row) {
        ggml_cuda_unpack_row_nvfp4((const uint8_t *) src + row * row_size, (block_nvfp4 *) ((uint8_t *) dst + row * row_size), ne0);
    }
}
