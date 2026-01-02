#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

// Simplified Q6_K block for testing (matches ggml-common.h)
struct block_q6_K {
    uint8_t  ql[128];     // quants, lower 4 bits
    uint8_t  qh[64];      // quants, upper 2 bits
    int8_t   scales[16];  // scales
    uint16_t d;           // super-block scale (ggml_half)
};

// Helper from common.hpp (inline for test)
inline int tile_count(int blocks) {
    int count = 0;
    while (blocks > 0) {
        int tile_size = 1;
        while (tile_size * 2 <= blocks && tile_size < 32) {
            tile_size *= 2;
        }
        count++;
        blocks -= tile_size;
    }
    return count;
}

inline int tile_size_at(int blocks, int tile_idx) {
    int idx = 0;
    while (blocks > 0) {
        int tile_size = 1;
        while (tile_size * 2 <= blocks && tile_size < 32) {
            tile_size *= 2;
        }
        if (idx == tile_idx) {
            return tile_size;
        }
        blocks -= tile_size;
        idx++;
    }
    return 0;
}

// Q6_K variable tile reorder: AoS -> word-major coalesced layout
// Each tile has its ql, qh, scales in word-major order
void reorder_q6_K_variable_tile(const block_q6_K * src,
                                uint8_t *          dst,
                                int                nrows,
                                int                blocks_per_row,
                                int64_t            row_stride) {
    const int num_tiles = tile_count(blocks_per_row);

    for (int row = 0; row < nrows; row++) {
        uint8_t * row_dst   = dst + row * row_stride;
        int       block_idx = 0;

        for (int tile = 0; tile < num_tiles; tile++) {
            int tile_size = tile_size_at(blocks_per_row, tile);

            // Reorder ql: word-major (32 words of 4 bytes each per block)
            for (int word = 0; word < 32; word++) {
                for (int b = 0; b < tile_size; b++) {
                    const block_q6_K & blk = src[row * blocks_per_row + block_idx + b];
                    memcpy(row_dst, &blk.ql[word * 4], 4);
                    row_dst += 4;
                }
            }

            // Reorder qh: word-major (16 words of 4 bytes each per block)
            for (int word = 0; word < 16; word++) {
                for (int b = 0; b < tile_size; b++) {
                    const block_q6_K & blk = src[row * blocks_per_row + block_idx + b];
                    memcpy(row_dst, &blk.qh[word * 4], 4);
                    row_dst += 4;
                }
            }

            // Reorder scales: word-major (4 words of 4 bytes each per block)
            for (int word = 0; word < 4; word++) {
                for (int b = 0; b < tile_size; b++) {
                    const block_q6_K & blk = src[row * blocks_per_row + block_idx + b];
                    memcpy(row_dst, &blk.scales[word * 4], 4);
                    row_dst += 4;
                }
            }

            block_idx += tile_size;
        }
    }
}

// Compute row stride for variable tile layout
int64_t compute_row_stride(int blocks_per_row) {
    const int num_tiles = tile_count(blocks_per_row);
    int64_t   stride    = 0;
    for (int t = 0; t < num_tiles; t++) {
        int ts = tile_size_at(blocks_per_row, t);
        stride += ts * (128 + 64 + 16);  // ql + qh + scales bytes per block
    }
    return stride;
}

int main() {
    // Test with 56 blocks (Mistral FFN dimension / QK_K)
    const int blocks_per_row = 56;
    const int nrows          = 2;

    std::vector<block_q6_K> src(nrows * blocks_per_row);

    // Fill with identifiable pattern
    for (int r = 0; r < nrows; r++) {
        for (int b = 0; b < blocks_per_row; b++) {
            block_q6_K & blk = src[r * blocks_per_row + b];
            // ql: fill with block index pattern
            for (int i = 0; i < 128; i++) {
                blk.ql[i] = ((r * 100 + b) + i) & 0xFF;
            }
            // qh: fill with different pattern
            for (int i = 0; i < 64; i++) {
                blk.qh[i] = ((r * 100 + b) * 2 + i) & 0xFF;
            }
            // scales: fill with signed values
            for (int i = 0; i < 16; i++) {
                blk.scales[i] = ((r * 100 + b) + i - 50) & 0x7F;
            }
            blk.d = r * 100 + b;
        }
    }

    // Compute expected sizes
    int     num_tiles  = tile_count(blocks_per_row);
    int64_t row_stride = compute_row_stride(blocks_per_row);

    std::vector<uint8_t> dst(nrows * row_stride);
    reorder_q6_K_variable_tile(src.data(), dst.data(), nrows, blocks_per_row, row_stride);

    // Verify tile structure: 56 = 32 + 16 + 8
    assert(num_tiles == 3);
    printf("PASS: 56 blocks -> %d tiles (32+16+8)\n", num_tiles);

    // Verify row stride calculation
    // 32*(128+64+16) + 16*(128+64+16) + 8*(128+64+16) = 56*208 = 11648
    assert(row_stride == 56 * 208);
    printf("PASS: Row stride = %ld bytes\n", (long) row_stride);

    // Verify first tile (32 blocks) - first word of ql should have block 0,1,2,...,31 data
    // Word 0 of ql: bytes 0-3 of block 0, then bytes 0-3 of block 1, etc.
    // Block 0 ql[0..3] = 0, 1, 2, 3 (since pattern is (0+i)&0xFF)
    assert(dst[0] == 0);  // Block 0, ql[0]
    assert(dst[1] == 1);  // Block 0, ql[1]
    assert(dst[2] == 2);  // Block 0, ql[2]
    assert(dst[3] == 3);  // Block 0, ql[3]
    // Next 4 bytes are block 1 ql[0..3] = 1, 2, 3, 4
    assert(dst[4] == 1);  // Block 1, ql[0]
    assert(dst[5] == 2);  // Block 1, ql[1]
    printf("PASS: First tile ql word-major layout correct\n");

    // Verify second tile starts after first tile data
    // First tile: 32 blocks * (128+64+16) = 6656 bytes
    int64_t tile2_offset = 32 * (128 + 64 + 16);
    // Second tile starts with block 32's data
    // Block 32 ql[0..3] = 32, 33, 34, 35
    assert(dst[tile2_offset + 0] == 32);
    assert(dst[tile2_offset + 1] == 33);
    printf("PASS: Second tile starts at correct offset (%ld)\n", (long) tile2_offset);

    // Verify row 1 starts at row_stride offset
    int64_t row1_offset = row_stride;
    // Row 1, Block 0 ql[0..3] = 100, 101, 102, 103
    assert(dst[row1_offset + 0] == 100);
    assert(dst[row1_offset + 1] == 101);
    printf("PASS: Row 1 data at correct offset\n");

    printf("\nAll variable tile reorder tests passed!\n");
    return 0;
}
