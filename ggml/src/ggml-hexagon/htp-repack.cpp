#include "htp-repack.h"

#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"
#include "ggml-impl.h"
#include "ggml-quants.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <initializer_list>

// --- 4-bit block-32 nibble helpers (shared by Q4_0, IQ4_NL, Q4_1, MXFP4) ---

static inline void unpack_nibbles_32(uint8_t * dst, const uint8_t * qs) {
    for (unsigned int i = 0; i < 16; ++i) {
        dst[i + 0]  = qs[i] & 0x0F;
        dst[i + 16] = qs[i] >> 4;
    }
}

static inline void pack_nibbles_32(uint8_t * qs, const uint8_t * src) {
    for (unsigned int i = 0; i < 16; ++i) {
        qs[i] = src[i + 0] | (src[i + 16] << 4);
    }
}

static inline void transpose_nibbles_to_tile(uint8_t * tile_dst, const uint8_t tile_quants[32][32]) {
    for (int cp = 0; cp < 16; cp++) {
        for (int row = 0; row < 32; row++) {
            tile_dst[cp * 32 + row] = (tile_quants[row][2 * cp + 1] << 4) | tile_quants[row][2 * cp];
        }
    }
}

static inline void transpose_tile_to_nibbles(uint8_t tile_quants[32][32], const uint8_t * tile_src) {
    for (int cp = 0; cp < 16; cp++) {
        for (int row = 0; row < 32; row++) {
            const uint8_t val = tile_src[cp * 32 + row];
            tile_quants[row][2 * cp + 0] = val & 0x0F;
            tile_quants[row][2 * cp + 1] = val >> 4;
        }
    }
}

// --- Generic 32x32 tile grid walkers for block-32 and float types ---

template <typename SrcElemT, int ELEMS_PER_KTILE, typename PackTileFn>
static void repack_to_tiled_grid_32(ggml_tensor * t, const void * data, size_t offset, size_t size,
                                    size_t tile_size, PackTileFn && pack_tile) {
    const SrcElemT * src_matrix = (const SrcElemT *) data;
    const int64_t ne0 = t->ne[0];
    const int64_t ne1 = t->ne[1];
    const int64_t ne2 = t->ne[2];
    const int64_t ne3 = t->ne[3];
    const int n_col_tiles = hex_round_up(ne1, 32) / 32;
    const int n_k_tiles   = hex_round_up(ne0, 32) / 32;
    const size_t matrix_size = (size_t) n_col_tiles * n_k_tiles * tile_size;

    const size_t slice_size = (size_t) ne1 * ggml_row_size(t->type, ne0);
    const int64_t start_slice = offset / slice_size;
    int64_t end_slice = (offset + size + slice_size - 1) / slice_size;
    if (end_slice > ne2 * ne3) {
        end_slice = ne2 * ne3;
    }

    const int64_t row_stride = (ne0 / (32 / ELEMS_PER_KTILE));

    for (int64_t slice_idx = start_slice; slice_idx < end_slice; slice_idx++) {
        const SrcElemT * src_slice = src_matrix + (slice_idx - start_slice) * (ne1 * row_stride);
        uint8_t * matrix_dst = (uint8_t *) t->data + slice_idx * matrix_size;

        for (int ct = 0; ct < n_col_tiles; ct++) {
            for (int kt = 0; kt < n_k_tiles; kt++) {
                uint8_t * tile_dst = matrix_dst + ((size_t) ct * n_k_tiles + kt) * tile_size;
                pack_tile(tile_dst, src_slice, ne0, ne1, row_stride, ct, kt);
            }
        }
    }
}

template <typename DstElemT, int ELEMS_PER_KTILE, typename UnpackTileFn>
static void unpack_from_tiled_grid_32(void * data, const ggml_tensor * t, size_t offset, size_t size,
                                      size_t tile_size, UnpackTileFn && unpack_tile) {
    uint8_t * dst_bytes = (uint8_t *) data;
    const int64_t ne0 = t->ne[0];
    const int64_t ne1 = t->ne[1];
    const int64_t ne2 = t->ne[2];
    const int64_t ne3 = t->ne[3];
    const int n_col_tiles = hex_round_up(ne1, 32) / 32;
    const int n_k_tiles   = hex_round_up(ne0, 32) / 32;
    const size_t matrix_size = (size_t) n_col_tiles * n_k_tiles * tile_size;

    const size_t row_size_bytes = ggml_row_size(t->type, ne0);
    const size_t slice_size     = (size_t) ne1 * row_size_bytes;
    const int64_t start_slice   = offset / slice_size;
    int64_t end_slice           = (offset + size + slice_size - 1) / slice_size;
    if (end_slice > ne2 * ne3) {
        end_slice = ne2 * ne3;
    }

    const int64_t row_stride = (ne0 / (32 / ELEMS_PER_KTILE));

    for (int64_t slice_idx = start_slice; slice_idx < end_slice; slice_idx++) {
        const size_t cur_start_byte     = (std::max)(offset, (size_t) slice_idx * slice_size);
        const size_t cur_end_byte       = (std::min)(offset + size, (size_t) (slice_idx + 1) * slice_size);
        const size_t slice_offset_start = cur_start_byte - (size_t) slice_idx * slice_size;
        const size_t slice_offset_end   = cur_end_byte - (size_t) slice_idx * slice_size;

        const int64_t start_row = slice_offset_start / row_size_bytes;
        const int64_t end_row   = (std::min)((int64_t) ((slice_offset_end + row_size_bytes - 1) / row_size_bytes), ne1);

        const int start_ct = (int) (start_row / 32);
        const int end_ct   = (std::min)((int) ((end_row + 31) / 32), n_col_tiles);

        DstElemT * dst_slice = (DstElemT *) (dst_bytes + (cur_start_byte - offset));
        const uint8_t * matrix_src = (const uint8_t *) t->data + slice_idx * matrix_size;

        for (int ct = start_ct; ct < end_ct; ct++) {
            for (int kt = 0; kt < n_k_tiles; kt++) {
                const uint8_t * tile_src = matrix_src + ((size_t) ct * n_k_tiles + kt) * tile_size;
                unpack_tile(dst_slice, tile_src, ne0, start_row, end_row, row_stride, ct, kt);
            }
        }
    }
}

// --- Block-32 Quantized Types (Q4_0, IQ4_NL, Q4_1, Q8_0, MXFP4) ---

static void repack_q4_0_tiled(ggml_tensor * t, const void * data, size_t offset, size_t size) {
    repack_to_tiled_grid_32<block_q4_0, 1>(t, data, offset, size, HTP_TENSOR_TILE_SIZE_Q4_0,
        [](uint8_t * tile_dst, const block_q4_0 * src_slice, int64_t ne0, int64_t ne1, int64_t row_stride, int ct, int kt) {
            uint8_t tile_quants[32][32];
            ggml_half * scale_dst = (ggml_half *) (tile_dst + 512);
            for (int row = 0; row < 32; row++) {
                const int64_t r = ct * 32 + row;
                if (r < ne1 && kt < ne0 / 32) {
                    const block_q4_0 & b = src_slice[r * row_stride + kt];
                    unpack_nibbles_32(tile_quants[row], b.qs);
                    scale_dst[row] = b.d;
                } else {
                    memset(tile_quants[row], 8, 32);
                    scale_dst[row] = 0;
                }
            }
            transpose_nibbles_to_tile(tile_dst, tile_quants);
        });
}

static void repack_tiled_q4_0(void * data, const ggml_tensor * t, size_t offset, size_t size) {
    unpack_from_tiled_grid_32<block_q4_0, 1>(data, t, offset, size, HTP_TENSOR_TILE_SIZE_Q4_0,
        [](block_q4_0 * dst_slice, const uint8_t * tile_src, int64_t ne0, int64_t start_row, int64_t end_row, int64_t row_stride, int ct, int kt) {
            uint8_t tile_quants[32][32];
            transpose_tile_to_nibbles(tile_quants, tile_src);
            const ggml_half * scale_src = (const ggml_half *) (tile_src + 512);
            for (int row = 0; row < 32; row++) {
                const int64_t r = ct * 32 + row;
                if (r >= start_row && r < end_row && kt < ne0 / 32) {
                    block_q4_0 & b = dst_slice[(r - start_row) * row_stride + kt];
                    pack_nibbles_32(b.qs, tile_quants[row]);
                    b.d = scale_src[row];
                }
            }
        });
}

static void repack_q4_1_tiled(ggml_tensor * t, const void * data, size_t offset, size_t size) {
    repack_to_tiled_grid_32<block_q4_1, 1>(t, data, offset, size, HTP_TENSOR_TILE_SIZE_Q4_1,
        [](uint8_t * tile_dst, const block_q4_1 * src_slice, int64_t ne0, int64_t ne1, int64_t row_stride, int ct, int kt) {
            uint8_t tile_quants[32][32];
            ggml_half * scale_dst = (ggml_half *) (tile_dst + 512);
            for (int row = 0; row < 32; row++) {
                const int64_t r = ct * 32 + row;
                if (r < ne1 && kt < ne0 / 32) {
                    const block_q4_1 & b = src_slice[r * row_stride + kt];
                    unpack_nibbles_32(tile_quants[row], b.qs);
                    scale_dst[2 * row + 0] = b.d;
                    scale_dst[2 * row + 1] = b.m;
                } else {
                    memset(tile_quants[row], 0, 32);
                    scale_dst[2 * row + 0] = 0;
                    scale_dst[2 * row + 1] = 0;
                }
            }
            transpose_nibbles_to_tile(tile_dst, tile_quants);
        });
}

static void repack_tiled_q4_1(void * data, const ggml_tensor * t, size_t offset, size_t size) {
    unpack_from_tiled_grid_32<block_q4_1, 1>(data, t, offset, size, HTP_TENSOR_TILE_SIZE_Q4_1,
        [](block_q4_1 * dst_slice, const uint8_t * tile_src, int64_t ne0, int64_t start_row, int64_t end_row, int64_t row_stride, int ct, int kt) {
            uint8_t tile_quants[32][32];
            transpose_tile_to_nibbles(tile_quants, tile_src);
            const ggml_half * scale_src = (const ggml_half *) (tile_src + 512);
            for (int row = 0; row < 32; row++) {
                const int64_t r = ct * 32 + row;
                if (r >= start_row && r < end_row && kt < ne0 / 32) {
                    block_q4_1 & b = dst_slice[(r - start_row) * row_stride + kt];
                    pack_nibbles_32(b.qs, tile_quants[row]);
                    b.d = scale_src[2 * row + 0];
                    b.m = scale_src[2 * row + 1];
                }
            }
        });
}

static void repack_mxfp4_tiled(ggml_tensor * t, const void * data, size_t offset, size_t size) {
    repack_to_tiled_grid_32<block_mxfp4, 1>(t, data, offset, size, HTP_TENSOR_TILE_SIZE_MXFP4,
        [](uint8_t * tile_dst, const block_mxfp4 * src_slice, int64_t ne0, int64_t ne1, int64_t row_stride, int ct, int kt) {
            uint8_t tile_quants[32][32];
            uint8_t * scale_dst = tile_dst + 512;
            for (int row = 0; row < 32; row++) {
                const int64_t r = ct * 32 + row;
                if (r < ne1 && kt < ne0 / 32) {
                    const block_mxfp4 & b = src_slice[r * row_stride + kt];
                    unpack_nibbles_32(tile_quants[row], b.qs);
                    scale_dst[row] = b.e;
                } else {
                    memset(tile_quants[row], 0, 32);
                    scale_dst[row] = 0;
                }
            }
            transpose_nibbles_to_tile(tile_dst, tile_quants);
        });
}

static void repack_tiled_mxfp4(void * data, const ggml_tensor * t, size_t offset, size_t size) {
    unpack_from_tiled_grid_32<block_mxfp4, 1>(data, t, offset, size, HTP_TENSOR_TILE_SIZE_MXFP4,
        [](block_mxfp4 * dst_slice, const uint8_t * tile_src, int64_t ne0, int64_t start_row, int64_t end_row, int64_t row_stride, int ct, int kt) {
            uint8_t tile_quants[32][32];
            transpose_tile_to_nibbles(tile_quants, tile_src);
            const uint8_t * scale_src = tile_src + 512;
            for (int row = 0; row < 32; row++) {
                const int64_t r = ct * 32 + row;
                if (r >= start_row && r < end_row && kt < ne0 / 32) {
                    block_mxfp4 & b = dst_slice[(r - start_row) * row_stride + kt];
                    pack_nibbles_32(b.qs, tile_quants[row]);
                    b.e = scale_src[row];
                }
            }
        });
}

static void repack_q8_0_tiled(ggml_tensor * t, const void * data, size_t offset, size_t size) {
    repack_to_tiled_grid_32<block_q8_0, 1>(t, data, offset, size, HTP_TENSOR_TILE_SIZE_Q8_0,
        [](uint8_t * tile_dst, const block_q8_0 * src_slice, int64_t ne0, int64_t ne1, int64_t row_stride, int ct, int kt) {
            for (int cp = 0; cp < 16; cp++) {
                const int col0 = cp * 2;
                const int col1 = col0 + 1;
                for (int row = 0; row < 32; row++) {
                    const int64_t r = ct * 32 + row;
                    const block_q8_0 * b = (r < ne1 && kt < ne0 / 32) ? &src_slice[r * row_stride + kt] : nullptr;
                    tile_dst[cp * 64 + 2 * row + 0] = b ? b->qs[col0] : 0;
                    tile_dst[cp * 64 + 2 * row + 1] = b ? b->qs[col1] : 0;
                }
            }
            ggml_half * scale_dst = (ggml_half *) (tile_dst + 1024);
            for (int row = 0; row < 32; row++) {
                const int64_t r = ct * 32 + row;
                scale_dst[row] = (r < ne1 && kt < ne0 / 32) ? src_slice[r * row_stride + kt].d : 0;
            }
        });
}

static void repack_tiled_q8_0(void * data, const ggml_tensor * t, size_t offset, size_t size) {
    unpack_from_tiled_grid_32<block_q8_0, 1>(data, t, offset, size, HTP_TENSOR_TILE_SIZE_Q8_0,
        [](block_q8_0 * dst_slice, const uint8_t * tile_src, int64_t ne0, int64_t start_row, int64_t end_row, int64_t row_stride, int ct, int kt) {
            for (int cp = 0; cp < 16; cp++) {
                const int col0 = cp * 2;
                const int col1 = col0 + 1;
                for (int row = 0; row < 32; row++) {
                    const int64_t r = ct * 32 + row;
                    if (r >= start_row && r < end_row && kt < ne0 / 32) {
                        block_q8_0 & b = dst_slice[(r - start_row) * row_stride + kt];
                        b.qs[col0] = tile_src[cp * 64 + 2 * row + 0];
                        b.qs[col1] = tile_src[cp * 64 + 2 * row + 1];
                    }
                }
            }
            const ggml_half * scale_src = (const ggml_half *) (tile_src + 1024);
            for (int row = 0; row < 32; row++) {
                const int64_t r = ct * 32 + row;
                if (r >= start_row && r < end_row && kt < ne0 / 32) {
                    dst_slice[(r - start_row) * row_stride + kt].d = scale_src[row];
                }
            }
        });
}

// --- Float Types (F16, F32, BF16 -> f16_tiled HMX layout) ---

static void repack_q2_0_tiled(ggml_tensor * t, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(t->ne[0] % 64 == 0);
    GGML_ASSERT(size == ggml_nbytes(t));

    const block_q2_0 * src = (const block_q2_0 *) data;
    const int64_t blocks_per_row = t->ne[0] / 64;
    const int64_t n_col_tiles = hex_round_up(t->ne[1], 32) / 32;
    const int64_t n_weight_tiles = blocks_per_row;
    const size_t matrix_size = (size_t) n_col_tiles * n_weight_tiles * HTP_TENSOR_TILE_SIZE_Q2_0;

    for (int64_t slice = 0; slice < t->ne[2] * t->ne[3]; ++slice) {
        const block_q2_0 * src_slice = src + slice * t->ne[1] * blocks_per_row;
        uint8_t * dst_slice = (uint8_t *) t->data + slice * matrix_size;
        memset(dst_slice, 0, matrix_size);

        for (int64_t row_index = 0; row_index < t->ne[1]; ++row_index) {
            const int64_t ct = row_index / 32;
            const int row = (int) (row_index % 32);
            for (int64_t kt = 0; kt < n_weight_tiles; ++kt) {
                const block_q2_0 * block = src_slice + row_index * blocks_per_row + kt;
                uint8_t * tile = dst_slice + (ct * n_weight_tiles + kt) * HTP_TENSOR_TILE_SIZE_Q2_0;
                for (int group = 0; group < 16; ++group) {
                    for (int byte = 0; byte < 4; ++byte) {
                        const int k = 4 * group + byte;
                        const uint8_t code = (block->qs[k / 4] >> (2 * (k % 4))) & 3;
                        tile[(group >> 2) * 128 + 4 * row + byte] |= code << (2 * (group & 3));
                    }
                }
                memcpy(tile + 512 + 2 * row, &block->d, sizeof(block->d));
            }
        }
    }
}

static void repack_tiled_q2_0(void * data, const ggml_tensor * t, size_t offset, size_t size) {
    GGML_ASSERT(t->ne[0] % 64 == 0);
    GGML_ASSERT(offset + size <= ggml_nbytes(t));

    const size_t first_block = offset / sizeof(block_q2_0);
    const size_t end_block = (offset + size + sizeof(block_q2_0) - 1) / sizeof(block_q2_0);
    const size_t blocks_per_row = t->ne[0] / 64;
    const size_t blocks_per_slice = t->ne[1] * blocks_per_row;
    const size_t n_weight_tiles = blocks_per_row;
    const size_t n_col_tiles = hex_round_up(t->ne[1], 32) / 32;
    const size_t matrix_size = n_col_tiles * n_weight_tiles * HTP_TENSOR_TILE_SIZE_Q2_0;

    for (size_t block_index = first_block; block_index < end_block; ++block_index) {
        const size_t slice = block_index / blocks_per_slice;
        const size_t in_slice = block_index % blocks_per_slice;
        const size_t row_index = in_slice / blocks_per_row;
        const size_t kt = in_slice % blocks_per_row;
        const size_t ct = row_index / 32;
        const size_t row = row_index % 32;
        const uint8_t * tile = (const uint8_t *) t->data + slice * matrix_size +
            (ct * n_weight_tiles + kt) * HTP_TENSOR_TILE_SIZE_Q2_0;

        block_q2_0 block = {};
        memcpy(&block.d, tile + 512 + 2 * row, sizeof(block.d));
        for (int group = 0; group < 16; ++group) {
            for (int byte = 0; byte < 4; ++byte) {
                const uint8_t packed = tile[(group >> 2) * 128 + 4 * row + byte];
                const uint8_t code = (packed >> (2 * (group & 3))) & 3;
                block.qs[group] |= code << (2 * byte);
            }
        }

        const size_t block_begin = block_index * sizeof(block_q2_0);
        const size_t copy_begin = (std::max)(block_begin, offset);
        const size_t copy_end = (std::min)(block_begin + sizeof(block_q2_0), offset + size);
        memcpy((uint8_t *) data + copy_begin - offset, (const uint8_t *) &block + copy_begin - block_begin, copy_end - copy_begin);
    }
}

template <typename SrcT, typename ToFp16Fn>
static void repack_float_to_f16_tiled(ggml_tensor * t, const void * data, size_t offset, size_t size, ToFp16Fn && to_fp16) {
    repack_to_tiled_grid_32<SrcT, 32>(t, data, offset, size, HTP_TENSOR_TILE_SIZE_F16,
        [&](uint8_t * tile_bytes, const SrcT * src_slice, int64_t ne0, int64_t ne1, int64_t row_stride, int ct, int kt) {
            uint16_t * tile_dst = (uint16_t *) tile_bytes;
            for (int cp = 0; cp < 16; cp++) {
                const int64_t c0 = kt * 32 + cp * 2;
                const int64_t c1 = c0 + 1;
                for (int row = 0; row < 32; row++) {
                    const int64_t r = ct * 32 + row;
                    const SrcT * r_ptr = (r < ne1) ? &src_slice[r * row_stride] : nullptr;
                    tile_dst[cp * 64 + row * 2 + 0] = (r_ptr && c0 < ne0) ? to_fp16(r_ptr[c0]) : 0;
                    tile_dst[cp * 64 + row * 2 + 1] = (r_ptr && c1 < ne0) ? to_fp16(r_ptr[c1]) : 0;
                }
            }
        });
}

template <typename DstT, typename FromFp16Fn>
static void unpack_f16_tiled_to_float(void * data, const ggml_tensor * t, size_t offset, size_t size, FromFp16Fn && from_fp16) {
    unpack_from_tiled_grid_32<DstT, 32>(data, t, offset, size, HTP_TENSOR_TILE_SIZE_F16,
        [&](DstT * dst_slice, const uint8_t * tile_bytes, int64_t ne0, int64_t start_row, int64_t end_row, int64_t row_stride, int ct, int kt) {
            const uint16_t * tile_src = (const uint16_t *) tile_bytes;
            for (int cp = 0; cp < 16; cp++) {
                const int64_t c0 = kt * 32 + cp * 2;
                const int64_t c1 = c0 + 1;
                for (int row = 0; row < 32; row++) {
                    const int64_t r = ct * 32 + row;
                    if (r >= start_row && r < end_row) {
                        DstT * r_ptr = dst_slice + (r - start_row) * row_stride;
                        if (c0 < ne0) r_ptr[c0] = from_fp16(tile_src[cp * 64 + row * 2 + 0]);
                        if (c1 < ne0) r_ptr[c1] = from_fp16(tile_src[cp * 64 + row * 2 + 1]);
                    }
                }
            }
        });
}

// --- Generic K-Quant (super-block 256) grid walkers ---

template <typename BlockT, typename PackKtFn>
static void repack_kquant_to_tiled_grid(ggml_tensor * t, const void * data, size_t offset, size_t size,
                                        size_t tile_size, PackKtFn && pack_kt) {
    GGML_ASSERT(offset == 0);
    GGML_UNUSED(size);

    const BlockT * src_matrix = (const BlockT *) data;
    const int64_t ne0 = t->ne[0];
    const int64_t ne1 = t->ne[1];
    const int64_t ne2 = t->ne[2];
    const int64_t ne3 = t->ne[3];
    GGML_ASSERT(ne0 % QK_K == 0);

    const int n_col_tiles    = hex_round_up(ne1, 32) / 32;
    const int n_k_tiles      = hex_round_up(ne0, 32) / 32;
    const size_t matrix_size = (size_t) n_col_tiles * n_k_tiles * tile_size;
    const int64_t sb_per_row = ne0 / QK_K;

    for (int64_t slice_idx = 0; slice_idx < ne2 * ne3; slice_idx++) {
        const BlockT * src_slice = src_matrix + slice_idx * (ne1 * sb_per_row);
        uint8_t * matrix_dst     = (uint8_t *) t->data + slice_idx * matrix_size;

        // Padding rows and OR-ed bit planes require zeroed tiles
        memset(matrix_dst, 0, matrix_size);

        for (int64_t r = 0; r < ne1; r++) {
            const int ct  = (int) (r / 32);
            const int row = (int) (r % 32);
            const BlockT * src_row = src_slice + r * sb_per_row;

            for (int kt = 0; kt < n_k_tiles; kt++) {
                uint8_t * tile = matrix_dst + ((size_t) ct * n_k_tiles + kt) * tile_size;
                pack_kt(tile, &src_row[kt / 8], row, kt % 8);
            }
        }
    }
}

template <typename BlockT, typename UnpackSbFn>
static void unpack_kquant_from_tiled_grid(void * data, const ggml_tensor * t, size_t offset, size_t size,
                                          size_t tile_size, UnpackSbFn && unpack_sb) {
    uint8_t * dst_bytes = (uint8_t *) data;
    const int64_t ne0 = t->ne[0];
    const int64_t ne1 = t->ne[1];
    const int64_t ne2 = t->ne[2];
    const int64_t ne3 = t->ne[3];
    GGML_ASSERT(ne0 % QK_K == 0);

    const int n_col_tiles    = hex_round_up(ne1, 32) / 32;
    const int n_k_tiles      = hex_round_up(ne0, 32) / 32;
    const size_t matrix_size = (size_t) n_col_tiles * n_k_tiles * tile_size;
    const int64_t sb_per_row = ne0 / QK_K;

    const size_t row_size_bytes = ggml_row_size(t->type, ne0);
    const size_t slice_size     = (size_t) ne1 * row_size_bytes;
    const int64_t start_slice   = offset / slice_size;
    int64_t end_slice           = (offset + size + slice_size - 1) / slice_size;
    if (end_slice > ne2 * ne3) {
        end_slice = ne2 * ne3;
    }

    for (int64_t slice_idx = start_slice; slice_idx < end_slice; slice_idx++) {
        const size_t cur_start_byte     = (std::max)(offset, (size_t) slice_idx * slice_size);
        const size_t cur_end_byte       = (std::min)(offset + size, (size_t) (slice_idx + 1) * slice_size);
        const size_t slice_offset_start = cur_start_byte - (size_t) slice_idx * slice_size;
        const size_t slice_offset_end   = cur_end_byte - (size_t) slice_idx * slice_size;

        const int64_t start_row = slice_offset_start / row_size_bytes;
        const int64_t end_row   = (std::min)((int64_t) ((slice_offset_end + row_size_bytes - 1) / row_size_bytes), ne1);

        BlockT * dst_slice         = (BlockT *) (dst_bytes + (cur_start_byte - offset));
        const uint8_t * matrix_src = (const uint8_t *) t->data + slice_idx * matrix_size;

        for (int64_t r = start_row; r < end_row; r++) {
            const int ct  = (int) (r / 32);
            const int row = (int) (r % 32);
            BlockT * dst_row = dst_slice + (r - start_row) * sb_per_row;

            for (int64_t sb = 0; sb < sb_per_row; sb++) {
                BlockT * b = &dst_row[sb];
                memset(b, 0, sizeof(BlockT));
                const uint8_t * sb_tiles = matrix_src + ((size_t) ct * n_k_tiles + sb * 8) * tile_size;
                unpack_sb(b, sb_tiles, tile_size, row);
            }
        }
    }
}

// --- K-Quant Helpers & Per-Type Implementations (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K) ---

// unsigned 6-bit value (0..63) of element e of a Q6_K block, same bit layout as dequantize_row_q6_K
static inline uint8_t q6_K_get_quant(const block_q6_K * b, int e) {
    const int c = e / 128;
    const int w = e % 128;
    const int g = w / 32;
    const int l = w % 32;
    const uint8_t * ql = b->ql + c * 64;
    const uint8_t * qh = b->qh + c * 32;
    uint8_t lo, hi;
    switch (g) {
        case 0:  lo = ql[l]      & 0xF; hi = (qh[l] >> 0) & 3; break;
        case 1:  lo = ql[l + 32] & 0xF; hi = (qh[l] >> 2) & 3; break;
        case 2:  lo = ql[l]      >> 4;  hi = (qh[l] >> 4) & 3; break;
        default: lo = ql[l + 32] >> 4;  hi = (qh[l] >> 6) & 3; break;
    }
    return (uint8_t) (lo | (hi << 4));
}

static void repack_q6_K_tiled(ggml_tensor * t, const void * data, size_t offset, size_t size) {
    repack_kquant_to_tiled_grid<block_q6_K>(t, data, offset, size, HTP_TENSOR_TILE_SIZE_Q6_K,
        [](uint8_t * tile, const block_q6_K * b, int row, int kt_local) {
            const float d     = GGML_FP16_TO_FP32(b->d);
            uint8_t * lo_pl   = tile;
            uint8_t * hi_pl   = tile + 512;
            ggml_half * sc_pl = (ggml_half *) (tile + 768);

            for (int lk = 0; lk < 32; lk++) {
                const uint8_t q6 = q6_K_get_quant(b, kt_local * 32 + lk);
                const int g   = lk >> 2;
                const int pos = row * 4 + (lk & 3);
                lo_pl[(g >> 1) * 128 + pos] |= (uint8_t) ((q6 & 0xF) << ((g & 1) * 4));
                hi_pl[(g >> 2) * 128 + pos] |= (uint8_t) ((q6 >> 4) << ((g & 3) * 2));
            }
            for (int sub = 0; sub < 2; sub++) {
                sc_pl[sub * 32 + row] = GGML_FP32_TO_FP16(d * (float) b->scales[kt_local * 2 + sub]);
            }
        });
}

static void repack_tiled_q6_K(void * data, const ggml_tensor * t, size_t offset, size_t size) {
    unpack_kquant_from_tiled_grid<block_q6_K>(data, t, offset, size, HTP_TENSOR_TILE_SIZE_Q6_K,
        [](block_q6_K * b, const uint8_t * sb_tiles, size_t tile_size, int row) {
            float sub_scales[16];
            for (int kt_local = 0; kt_local < 8; kt_local++) {
                const uint8_t *   tile  = sb_tiles + kt_local * tile_size;
                const uint8_t *   lo_pl = tile;
                const uint8_t *   hi_pl = tile + 512;
                const ggml_half * sc_pl = (const ggml_half *) (tile + 768);

                const int c  = kt_local / 4;
                const int g  = kt_local % 4;
                uint8_t * ql = b->ql + c * 64;
                uint8_t * qh = b->qh + c * 32;

                for (int lk = 0; lk < 32; lk++) {
                    const int g_tile = lk >> 2;
                    const int pos    = row * 4 + (lk & 3);
                    const uint8_t lo = (lo_pl[(g_tile >> 1) * 128 + pos] >> ((g_tile & 1) * 4)) & 0xF;
                    const uint8_t hi = (hi_pl[(g_tile >> 2) * 128 + pos] >> ((g_tile & 3) * 2)) & 3;

                    switch (g) {
                        case 0:  ql[lk]      |= lo;        qh[lk] |= (hi << 0); break;
                        case 1:  ql[lk + 32] |= lo;        qh[lk] |= (hi << 2); break;
                        case 2:  ql[lk]      |= (lo << 4); qh[lk] |= (hi << 4); break;
                        default: ql[lk + 32] |= (lo << 4); qh[lk] |= (hi << 6); break;
                    }
                }

                for (int sub = 0; sub < 2; sub++) {
                    sub_scales[kt_local * 2 + sub] = GGML_FP16_TO_FP32(sc_pl[sub * 32 + row]);
                }
            }

            float max_abs_scale = 0.0f;
            for (int s = 0; s < 16; s++) {
                max_abs_scale = (std::max)(max_abs_scale, fabsf(sub_scales[s]));
            }

            if (max_abs_scale == 0.0f) {
                b->d = GGML_FP32_TO_FP16(0.0f);
            } else {
                b->d = GGML_FP32_TO_FP16(max_abs_scale / 127.0f);
                const float d_actual = GGML_FP16_TO_FP32(b->d);
                const float inv_d    = (d_actual != 0.0f) ? (1.0f / d_actual) : 0.0f;
                for (int s = 0; s < 16; s++) {
                    const int sc = (int) roundf(sub_scales[s] * inv_d);
                    b->scales[s] = (int8_t) (std::max)(-128, (std::min)(127, sc));
                }
            }
        });
}

// low 2 bits (0..3) of element e of a Q2_K or Q3_K block, same bit layout as dequantize_row_q2_K / q3_K
static inline uint8_t q2_3_K_get_low2(const uint8_t * qs, int e) {
    const int c = e / 128;
    const int j = (e % 128) / 32;
    const int l = e % 32;
    return (qs[c * 32 + l] >> (2 * j)) & 3;
}

// hmask bit of element e of a Q3_K block, same bit layout as dequantize_row_q3_K
static inline bool q3_K_get_hbit(const block_q3_K * b, int e) {
    const int c = e / 128;
    const int j = (e % 128) / 32;
    const int l = e % 32;
    return (b->hmask[l] >> (c * 4 + j)) & 1;
}

// signed 6-bit scale j (-32..31) of a Q3_K block, same packing as quantize_row_q3_K_ref
static inline int q3_K_get_scale(const uint8_t * scales, int j) {
    const int lo = (j < 8) ? (scales[j] & 0xF) : (scales[j - 8] >> 4);
    const int hi = (scales[8 + j % 4] >> (2 * (j / 4))) & 3;
    return (lo | (hi << 4)) - 32;
}

// read-back: find fp16 d and l[j] in [lmin, lmax] with fp16(d * l[j]) == prod[j] for all j, false if none
static bool hexagon_recover_k_scales(const ggml_half * prod, int n, int lmin, int lmax, ggml_half * d_out, int * l_out) {
    int jmax = 0;
    for (int j = 1; j < n; j++) {
        if (fabsf(GGML_FP16_TO_FP32(prod[j])) > fabsf(GGML_FP16_TO_FP32(prod[jmax]))) {
            jmax = j;
        }
    }
    const float pmax = GGML_FP16_TO_FP32(prod[jmax]);
    if (pmax == 0.0f) {
        *d_out = GGML_FP32_TO_FP16(0.0f);
        for (int j = 0; j < n; j++) {
            l_out[j] = 0;
        }
        return true;
    }

    // the quantizers put the largest scale at or near the range end, so try large |l| first
    const int lext = (std::max)(-lmin, lmax);
    for (int a = lext; a >= 1; a--) {
        for (int sign : { 1, -1 }) {
            const int lj = sign * a;
            if (lj < lmin || lj > lmax) {
                continue;
            }
            const ggml_half d0 = GGML_FP32_TO_FP16(pmax / (float) lj);
            for (int ulp : { 0, -1, 1 }) {
                ggml_half d = d0;
                uint16_t bits;
                memcpy(&bits, &d, sizeof(bits));
                bits = (uint16_t) (bits + ulp);
                memcpy(&d, &bits, sizeof(bits));

                const float df = GGML_FP16_TO_FP32(d);
                if (!std::isfinite(df) || df == 0.0f) {
                    continue;
                }
                bool ok = true;
                for (int j = 0; j < n && ok; j++) {
                    const int l = (int) roundf(GGML_FP16_TO_FP32(prod[j]) / df);
                    const ggml_half p = GGML_FP32_TO_FP16(df * (float) l);
                    ok = l >= lmin && l <= lmax && memcmp(&p, &prod[j], sizeof(p)) == 0;
                    l_out[j] = l;
                }
                if (ok) {
                    *d_out = d;
                    return true;
                }
            }
        }
    }
    return false;
}

static void repack_q3_K_tiled(ggml_tensor * t, const void * data, size_t offset, size_t size) {
    repack_kquant_to_tiled_grid<block_q3_K>(t, data, offset, size, HTP_TENSOR_TILE_SIZE_Q3_K,
        [](uint8_t * tile, const block_q3_K * b, int row, int kt_local) {
            const float d     = GGML_FP16_TO_FP32(b->d);
            uint8_t * lo_pl   = tile;
            uint8_t * neg_pl  = tile + 256;
            ggml_half * sc_pl = (ggml_half *) (tile + 384);

            for (int lk = 0; lk < 32; lk++) {
                const int e   = kt_local * 32 + lk;
                const int g   = lk >> 2;
                const int pos = row * 4 + (lk & 3);
                lo_pl[(g >> 2) * 128 + pos] |= (uint8_t) (q2_3_K_get_low2(b->qs, e) << ((g & 3) * 2));
                if (!q3_K_get_hbit(b, e)) {
                    neg_pl[pos] |= (uint8_t) (1 << g);
                }
            }
            for (int sub = 0; sub < 2; sub++) {
                sc_pl[sub * 32 + row] = GGML_FP32_TO_FP16(d * (float) q3_K_get_scale(b->scales, kt_local * 2 + sub));
            }
        });
}

static void repack_tiled_q3_K(void * data, const ggml_tensor * t, size_t offset, size_t size) {
    unpack_kquant_from_tiled_grid<block_q3_K>(data, t, offset, size, HTP_TENSOR_TILE_SIZE_Q3_K,
        [](block_q3_K * b, const uint8_t * sb_tiles, size_t tile_size, int row) {
            ggml_half sub_scales[16];
            for (int kt_local = 0; kt_local < 8; kt_local++) {
                const uint8_t *   tile   = sb_tiles + kt_local * tile_size;
                const uint8_t *   lo_pl  = tile;
                const uint8_t *   neg_pl = tile + 256;
                const ggml_half * sc_pl  = (const ggml_half *) (tile + 384);

                for (int lk = 0; lk < 32; lk++) {
                    const int e   = kt_local * 32 + lk;
                    const int g   = lk >> 2;
                    const int pos = row * 4 + (lk & 3);
                    const uint8_t lo = (lo_pl[(g >> 2) * 128 + pos] >> ((g & 3) * 2)) & 3;

                    const int c = e / 128;
                    const int j = (e % 128) / 32;
                    const int l = e % 32;
                    b->qs[c * 32 + l] |= (uint8_t) (lo << (2 * j));
                    if (!((neg_pl[pos] >> g) & 1)) {
                        b->hmask[l] |= (uint8_t) (1 << (c * 4 + j));
                    }
                }

                for (int sub = 0; sub < 2; sub++) {
                    sub_scales[kt_local * 2 + sub] = sc_pl[sub * 32 + row];
                }
            }

            int ls[16];
            if (!hexagon_recover_k_scales(sub_scales, 16, -32, 31, &b->d, ls)) {
                float max_scale = 0.0f;
                for (int s = 0; s < 16; s++) {
                    if (fabsf(GGML_FP16_TO_FP32(sub_scales[s])) > fabsf(max_scale)) {
                        max_scale = GGML_FP16_TO_FP32(sub_scales[s]);
                    }
                }
                b->d = GGML_FP32_TO_FP16(-max_scale / 32.0f);
                const float d_actual = GGML_FP16_TO_FP32(b->d);
                const float inv_d    = (d_actual != 0.0f) ? (1.0f / d_actual) : 0.0f;
                for (int s = 0; s < 16; s++) {
                    ls[s] = (std::max)(-32, (std::min)(31, (int) roundf(GGML_FP16_TO_FP32(sub_scales[s]) * inv_d)));
                }
            }

            for (int s = 0; s < 16; s++) {
                const int l = ls[s] + 32;
                if (s < 8) {
                    b->scales[s] = l & 0xF;
                } else {
                    b->scales[s - 8] |= (uint8_t) ((l & 0xF) << 4);
                }
                b->scales[s % 4 + 8] |= (uint8_t) ((l >> 4) << (2 * (s / 4)));
            }
        });
}

static void repack_q2_K_tiled(ggml_tensor * t, const void * data, size_t offset, size_t size) {
    repack_kquant_to_tiled_grid<block_q2_K>(t, data, offset, size, HTP_TENSOR_TILE_SIZE_Q2_K,
        [](uint8_t * tile, const block_q2_K * b, int row, int kt_local) {
            uint8_t * lo_pl   = tile;
            uint8_t * sc_pl   = tile + 256;
            ggml_half * dd_pl = (ggml_half *) (tile + 320);

            for (int lk = 0; lk < 32; lk++) {
                const int g   = lk >> 2;
                const int pos = row * 4 + (lk & 3);
                lo_pl[(g >> 2) * 128 + pos] |= (uint8_t) (q2_3_K_get_low2(b->qs, kt_local * 32 + lk) << ((g & 3) * 2));
            }
            for (int sub = 0; sub < 2; sub++) {
                sc_pl[sub * 32 + row] = b->scales[kt_local * 2 + sub];
            }
            if (row / 8 == kt_local % 4) {
                dd_pl[row % 8] = kt_local < 4 ? b->d : (ggml_half) (b->dmin ^ 0x8000);
            }
        });
}

static void repack_tiled_q2_K(void * data, const ggml_tensor * t, size_t offset, size_t size) {
    unpack_kquant_from_tiled_grid<block_q2_K>(data, t, offset, size, HTP_TENSOR_TILE_SIZE_Q2_K,
        [](block_q2_K * b, const uint8_t * sb_tiles, size_t tile_size, int row) {
            for (int kt_local = 0; kt_local < 8; kt_local++) {
                const uint8_t *   tile  = sb_tiles + kt_local * tile_size;
                const uint8_t *   lo_pl = tile;
                const uint8_t *   sc_pl = tile + 256;
                const ggml_half * dd_pl = (const ggml_half *) (tile + 320);

                for (int lk = 0; lk < 32; lk++) {
                    const int e   = kt_local * 32 + lk;
                    const int g   = lk >> 2;
                    const int pos = row * 4 + (lk & 3);
                    const uint8_t lo = (lo_pl[(g >> 2) * 128 + pos] >> ((g & 3) * 2)) & 3;
                    b->qs[(e / 128) * 32 + e % 32] |= (uint8_t) (lo << (2 * ((e % 128) / 32)));
                }

                for (int sub = 0; sub < 2; sub++) {
                    b->scales[kt_local * 2 + sub] = sc_pl[sub * 32 + row];
                }
                if (row / 8 == kt_local % 4) {
                    if (kt_local < 4) {
                        b->d = dd_pl[row % 8];
                    } else {
                        b->dmin = (ggml_half) (dd_pl[row % 8] ^ 0x8000);
                    }
                }
            }
        });
}

static inline void get_scale_min_k4(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >>  4) | ((q[j - 0] >> 6) << 4);
    }
}

static inline void pack_k4_scales_mins(uint8_t * scales, ggml_half * d, ggml_half * dmin,
                                       const float * sub_scales, const float * sub_mins) {
    float max_scale = 0.0f;
    float max_min   = 0.0f;
    for (int j = 0; j < 8; j++) {
        if (sub_scales[j] > max_scale) max_scale = sub_scales[j];
        if (sub_mins[j]   > max_min)   max_min   = sub_mins[j];
    }

    float inv_scale = 0.0f;
    if (max_scale > 0.0f) {
        *d = GGML_FP32_TO_FP16(max_scale / 63.0f);
        const float d_actual = GGML_FP16_TO_FP32(*d);
        inv_scale = (d_actual > 0.0f) ? (1.0f / d_actual) : 0.0f;
    } else {
        *d = GGML_FP32_TO_FP16(0.0f);
    }

    float inv_min = 0.0f;
    if (max_min > 0.0f) {
        *dmin = GGML_FP32_TO_FP16(max_min / 63.0f);
        const float dmin_actual = GGML_FP16_TO_FP32(*dmin);
        inv_min = (dmin_actual > 0.0f) ? (1.0f / dmin_actual) : 0.0f;
    } else {
        *dmin = GGML_FP32_TO_FP16(0.0f);
    }

    for (int j = 0; j < 8; j++) {
        uint8_t ls = (std::min)((uint8_t) 63, (uint8_t) roundf(inv_scale * sub_scales[j]));
        uint8_t lm = (std::min)((uint8_t) 63, (uint8_t) roundf(inv_min * sub_mins[j]));
        if (j < 4) {
            scales[j]     = ls;
            scales[j + 4] = lm;
        } else {
            scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
            scales[j - 4] |= ((ls >> 4) << 6);
            scales[j - 0] |= ((lm >> 4) << 6);
        }
    }
}

static void repack_q4_K_tiled(ggml_tensor * t, const void * data, size_t offset, size_t size) {
    repack_kquant_to_tiled_grid<block_q4_K>(t, data, offset, size, HTP_TENSOR_TILE_SIZE_Q4_K,
        [](uint8_t * tile_dst, const block_q4_K * b, int row, int kt_local) {
            const float d    = GGML_FP16_TO_FP32(b->d);
            const float dmin = GGML_FP16_TO_FP32(b->dmin);

            uint8_t sc, m;
            get_scale_min_k4(kt_local, b->scales, &sc, &m);

            const uint8_t * qs_sub = b->qs + (kt_local / 2) * 32;
            const int shift = (kt_local & 1) ? 4 : 0;

            for (int cp = 0; cp < 16; cp++) {
                const uint8_t q0 = (qs_sub[2 * cp + 0] >> shift) & 0x0F;
                const uint8_t q1 = (qs_sub[2 * cp + 1] >> shift) & 0x0F;
                tile_dst[cp * 32 + row] = (uint8_t) ((q1 << 4) | q0);
            }

            ggml_half * scale_dst = (ggml_half *) (tile_dst + 512);
            scale_dst[2 * row + 0] = GGML_FP32_TO_FP16(d * (float) sc);
            scale_dst[2 * row + 1] = GGML_FP32_TO_FP16(-dmin * (float) m);
        });
}

static void repack_tiled_q4_K(void * data, const ggml_tensor * t, size_t offset, size_t size) {
    unpack_kquant_from_tiled_grid<block_q4_K>(data, t, offset, size, HTP_TENSOR_TILE_SIZE_Q4_K,
        [](block_q4_K * b, const uint8_t * sb_tiles, size_t tile_size, int row) {
            float sub_scales[8];
            float sub_mins[8];

            for (int kt_local = 0; kt_local < 8; kt_local++) {
                const uint8_t *   tile_src  = sb_tiles + kt_local * tile_size;
                const ggml_half * scale_src = (const ggml_half *) (tile_src + 512);

                uint8_t * qs_sub = b->qs + (kt_local / 2) * 32;
                const int shift  = (kt_local & 1) ? 4 : 0;

                for (int cp = 0; cp < 16; cp++) {
                    const uint8_t val = tile_src[cp * 32 + row];
                    qs_sub[2 * cp + 0] |= (uint8_t) ((val & 0x0F) << shift);
                    qs_sub[2 * cp + 1] |= (uint8_t) ((val >> 4)   << shift);
                }

                const float D = GGML_FP16_TO_FP32(scale_src[2 * row + 0]);
                const float M = GGML_FP16_TO_FP32(scale_src[2 * row + 1]);
                sub_scales[kt_local] = (D > 0.0f) ? D : 0.0f;
                sub_mins[kt_local]   = (-M > 0.0f) ? -M : 0.0f;
            }

            pack_k4_scales_mins(b->scales, &b->d, &b->dmin, sub_scales, sub_mins);
        });
}

static void repack_q5_K_tiled(ggml_tensor * t, const void * data, size_t offset, size_t size) {
    repack_kquant_to_tiled_grid<block_q5_K>(t, data, offset, size, HTP_TENSOR_TILE_SIZE_Q5_K,
        [](uint8_t * tile_dst, const block_q5_K * b, int row, int kt_local) {
            const float d     = GGML_FP16_TO_FP32(b->d);
            const float dmin  = GGML_FP16_TO_FP32(b->dmin);
            uint8_t * plane   = tile_dst + 640;

            uint8_t sc, m;
            get_scale_min_k4(kt_local, b->scales, &sc, &m);

            const uint8_t * qs_sub = b->qs + (kt_local / 2) * 32;
            const int shift    = (kt_local & 1) ? 4 : 0;
            const uint8_t hbit = (uint8_t) (1 << kt_local);

            for (int cp = 0; cp < 16; cp++) {
                const uint8_t q0 = (qs_sub[2 * cp + 0] >> shift) & 0x0F;
                const uint8_t q1 = (qs_sub[2 * cp + 1] >> shift) & 0x0F;
                tile_dst[cp * 32 + row] = (uint8_t) ((q1 << 4) | q0);

                const int i    = cp / 4;
                const int lane = (cp % 4) * 32 + row;
                if (b->qh[2 * cp + 0] & hbit) {
                    plane[lane] |= (uint8_t) (1 << (2 * i));
                }
                if (b->qh[2 * cp + 1] & hbit) {
                    plane[lane] |= (uint8_t) (1 << (2 * i + 1));
                }
            }

            ggml_half * scale_dst = (ggml_half *) (tile_dst + 512);
            scale_dst[2 * row + 0] = GGML_FP32_TO_FP16(d * (float) sc);
            scale_dst[2 * row + 1] = GGML_FP32_TO_FP16(-dmin * (float) m);
        });
}

static void repack_tiled_q5_K(void * data, const ggml_tensor * t, size_t offset, size_t size) {
    unpack_kquant_from_tiled_grid<block_q5_K>(data, t, offset, size, HTP_TENSOR_TILE_SIZE_Q5_K,
        [](block_q5_K * b, const uint8_t * sb_tiles, size_t tile_size, int row) {
            float sub_scales[8];
            float sub_mins[8];

            for (int kt_local = 0; kt_local < 8; kt_local++) {
                const uint8_t *   tile_src  = sb_tiles + kt_local * tile_size;
                const uint8_t *   plane     = tile_src + 640;
                const ggml_half * scale_src = (const ggml_half *) (tile_src + 512);

                uint8_t * qs_sub   = b->qs + (kt_local / 2) * 32;
                const int shift    = (kt_local & 1) ? 4 : 0;
                const uint8_t hbit = (uint8_t) (1 << kt_local);

                for (int cp = 0; cp < 16; cp++) {
                    const uint8_t val = tile_src[cp * 32 + row];
                    qs_sub[2 * cp + 0] |= (uint8_t) ((val & 0x0F) << shift);
                    qs_sub[2 * cp + 1] |= (uint8_t) ((val >> 4)   << shift);

                    const int i    = cp / 4;
                    const int lane = (cp % 4) * 32 + row;
                    if (plane[lane] & (1 << (2 * i))) {
                        b->qh[2 * cp + 0] |= hbit;
                    }
                    if (plane[lane] & (1 << (2 * i + 1))) {
                        b->qh[2 * cp + 1] |= hbit;
                    }
                }

                const float D = GGML_FP16_TO_FP32(scale_src[2 * row + 0]);
                const float M = GGML_FP16_TO_FP32(scale_src[2 * row + 1]);
                sub_scales[kt_local] = (D > 0.0f) ? D : 0.0f;
                sub_mins[kt_local]   = (-M > 0.0f) ? -M : 0.0f;
            }

            pack_k4_scales_mins(b->scales, &b->d, &b->dmin, sub_scales, sub_mins);
        });
}

static inline void q1_0_tiled_bit_location(int k, int row, int & byte_offset, int & bit) {
    const int quarter = k / 32;
    const int kq      = k % 32;

    byte_offset = quarter * 128 + 4 * row + (kq % 4);
    bit         = kq / 4;
}

static void repack_q1_0_tiled(ggml_tensor * t, const void * data, size_t offset, size_t size) {
    const block_q1_0 * src_matrix = (const block_q1_0 *) data;
    const int64_t ne0 = t->ne[0];
    const int64_t ne1 = t->ne[1];
    const int64_t ne2 = t->ne[2];
    const int64_t ne3 = t->ne[3];
    const int n_col_tiles = hex_round_up(ne1, 32) / 32;
    const int n_k_tiles = ne0 / QK1_0;
    const size_t tile_size = HTP_TENSOR_TILE_SIZE_Q1_0;
    const size_t matrix_size = (size_t) n_col_tiles * n_k_tiles * tile_size;
    const size_t slice_size = ne1 * ggml_row_size(t->type, ne0);
    const int64_t start_slice = offset / slice_size;
    const int64_t end_slice = (std::min)((int64_t) ((offset + size + slice_size - 1) / slice_size), ne2 * ne3);

    GGML_ASSERT(ne0 % QK1_0 == 0);

    for (int64_t slice_idx = start_slice; slice_idx < end_slice; slice_idx++) {
        const block_q1_0 * src_slice = src_matrix + (slice_idx - start_slice) * ne1 * n_k_tiles;
        uint8_t * matrix_dst = (uint8_t *) t->data + slice_idx * matrix_size;

        for (int ct = 0; ct < n_col_tiles; ct++) {
            for (int kt = 0; kt < n_k_tiles; kt++) {
                uint8_t * tile_dst = matrix_dst + ((size_t) ct * n_k_tiles + kt) * tile_size;
                ggml_half * scale_dst = (ggml_half *) (tile_dst + 512);
                memset(tile_dst, 0, tile_size);

                for (int row = 0; row < 32; row++) {
                    const int64_t r = ct * 32 + row;
                    if (r >= ne1) {
                        continue;
                    }

                    const block_q1_0 & block = src_slice[r * n_k_tiles + kt];
                    scale_dst[row] = block.d;
                    for (int k = 0; k < QK1_0; k++) {
                        int byte_offset;
                        int bit;
                        q1_0_tiled_bit_location(k, row, byte_offset, bit);
                        if (block.qs[k / 8] & (1u << (k % 8))) {
                            tile_dst[byte_offset] |= (uint8_t) (1u << bit);
                        }
                    }
                }
            }
        }
    }
}

static void repack_tiled_q1_0(void * data, const ggml_tensor * t, size_t offset, size_t size) {
    block_q1_0 * dst_matrix = (block_q1_0 *) data;
    const int64_t ne0 = t->ne[0];
    const int64_t ne1 = t->ne[1];
    const int64_t ne2 = t->ne[2];
    const int64_t ne3 = t->ne[3];
    const int n_col_tiles = hex_round_up(ne1, 32) / 32;
    const int n_k_tiles = ne0 / QK1_0;
    const size_t tile_size = HTP_TENSOR_TILE_SIZE_Q1_0;
    const size_t matrix_size = (size_t) n_col_tiles * n_k_tiles * tile_size;
    const size_t row_size_bytes = ggml_row_size(t->type, ne0);
    const size_t slice_size = ne1 * row_size_bytes;
    const int64_t start_slice = offset / slice_size;
    const int64_t end_slice = (std::min)((int64_t) ((offset + size + slice_size - 1) / slice_size), ne2 * ne3);

    GGML_ASSERT(ne0 % QK1_0 == 0);

    for (int64_t slice_idx = start_slice; slice_idx < end_slice; slice_idx++) {
        const size_t cur_start_byte = (std::max)(offset, (size_t) slice_idx * slice_size);
        const size_t cur_end_byte = (std::min)(offset + size, (size_t) (slice_idx + 1) * slice_size);
        const int64_t start_row = (cur_start_byte - (size_t) slice_idx * slice_size) / row_size_bytes;
        const int64_t end_row = (std::min)((int64_t) ((cur_end_byte - (size_t) slice_idx * slice_size + row_size_bytes - 1) / row_size_bytes), ne1);
        block_q1_0 * dst_slice = dst_matrix + (cur_start_byte - offset) / sizeof(block_q1_0);
        const uint8_t * matrix_src = (const uint8_t *) t->data + slice_idx * matrix_size;

        for (int ct = start_row / 32; ct < (end_row + 31) / 32; ct++) {
            for (int kt = 0; kt < n_k_tiles; kt++) {
                const uint8_t * tile_src = matrix_src + ((size_t) ct * n_k_tiles + kt) * tile_size;
                const ggml_half * scale_src = (const ggml_half *) (tile_src + 512);

                for (int row = 0; row < 32; row++) {
                    const int64_t r = ct * 32 + row;
                    if (r < start_row || r >= end_row) {
                        continue;
                    }

                    block_q1_0 & block = dst_slice[(r - start_row) * n_k_tiles + kt];
                    memset(block.qs, 0, sizeof(block.qs));
                    block.d = scale_src[row];
                    for (int k = 0; k < QK1_0; k++) {
                        int byte_offset;
                        int bit;
                        q1_0_tiled_bit_location(k, row, byte_offset, bit);
                        if (tile_src[byte_offset] & (1u << bit)) {
                            block.qs[k / 8] |= (uint8_t) (1u << (k % 8));
                        }
                    }
                }
            }
        }
    }
}

// --- Top-level Repack / Unpack Dispatchers ---

void htp_repack_tensor_tiled(ggml_tensor * tensor, const void * data, size_t size) {
    switch (tensor->type) {
        case GGML_TYPE_Q1_0:
            repack_q1_0_tiled(tensor, data, 0, size);
            break;
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_IQ4_NL:
            repack_q4_0_tiled(tensor, data, 0, size);
            break;
        case GGML_TYPE_Q4_1:
            repack_q4_1_tiled(tensor, data, 0, size);
            break;
        case GGML_TYPE_Q8_0:
            repack_q8_0_tiled(tensor, data, 0, size);
            break;
        case GGML_TYPE_MXFP4:
            repack_mxfp4_tiled(tensor, data, 0, size);
            break;
        case GGML_TYPE_Q2_0:
            repack_q2_0_tiled(tensor, data, 0, size);
            break;
        case GGML_TYPE_Q2_K:
            repack_q2_K_tiled(tensor, data, 0, size);
            break;
        case GGML_TYPE_Q3_K:
            repack_q3_K_tiled(tensor, data, 0, size);
            break;
        case GGML_TYPE_Q4_K:
            repack_q4_K_tiled(tensor, data, 0, size);
            break;
        case GGML_TYPE_Q5_K:
            repack_q5_K_tiled(tensor, data, 0, size);
            break;
        case GGML_TYPE_Q6_K:
            repack_q6_K_tiled(tensor, data, 0, size);
            break;
        case GGML_TYPE_F16:
            repack_float_to_f16_tiled<uint16_t>(tensor, data, 0, size,
                [](uint16_t v) { return v; });
            break;
        case GGML_TYPE_F32:
            repack_float_to_f16_tiled<float>(tensor, data, 0, size,
                [](float v) { return (uint16_t) ggml_fp32_to_fp16(v); });
            break;
        case GGML_TYPE_BF16:
            repack_float_to_f16_tiled<ggml_bf16_t>(tensor, data, 0, size,
                [](ggml_bf16_t v) { return (uint16_t) ggml_fp32_to_fp16(ggml_bf16_to_fp32(v)); });
            break;
        default:
            break;
    }
}

void htp_unpack_tensor_tiled(void * data, const ggml_tensor * tensor, size_t offset, size_t size) {
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor));

    switch (tensor->type) {
        case GGML_TYPE_Q1_0:
            repack_tiled_q1_0(data, tensor, offset, size);
            break;
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_IQ4_NL:
            repack_tiled_q4_0(data, tensor, offset, size);
            break;
        case GGML_TYPE_Q4_1:
            repack_tiled_q4_1(data, tensor, offset, size);
            break;
        case GGML_TYPE_Q8_0:
            repack_tiled_q8_0(data, tensor, offset, size);
            break;
        case GGML_TYPE_MXFP4:
            repack_tiled_mxfp4(data, tensor, offset, size);
            break;
        case GGML_TYPE_Q2_0:
            repack_tiled_q2_0(data, tensor, offset, size);
            break;
        case GGML_TYPE_Q2_K:
            repack_tiled_q2_K(data, tensor, offset, size);
            break;
        case GGML_TYPE_Q3_K:
            repack_tiled_q3_K(data, tensor, offset, size);
            break;
        case GGML_TYPE_Q4_K:
            repack_tiled_q4_K(data, tensor, offset, size);
            break;
        case GGML_TYPE_Q5_K:
            repack_tiled_q5_K(data, tensor, offset, size);
            break;
        case GGML_TYPE_Q6_K:
            repack_tiled_q6_K(data, tensor, offset, size);
            break;
        case GGML_TYPE_F16:
            unpack_f16_tiled_to_float<uint16_t>(data, tensor, offset, size,
                [](uint16_t v) { return v; });
            break;
        case GGML_TYPE_F32:
            unpack_f16_tiled_to_float<float>(data, tensor, offset, size,
                [](uint16_t v) { return ggml_fp16_to_fp32((ggml_fp16_t) v); });
            break;
        case GGML_TYPE_BF16:
            unpack_f16_tiled_to_float<ggml_bf16_t>(data, tensor, offset, size,
                [](uint16_t v) { return ggml_fp32_to_bf16(ggml_fp16_to_fp32((ggml_fp16_t) v)); });
            break;
        default:
            memcpy(data, (const char *) tensor->data + offset, size);
            break;
    }
}
