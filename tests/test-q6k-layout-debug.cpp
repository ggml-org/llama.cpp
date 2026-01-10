// Test Q6_K variable tile layout matches between CPU reorder and GPU kernel access
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cstdint>

// Constants from ggml (QK_K = 256)
constexpr int QK_K = 256;
constexpr int QI6_K = 32;
constexpr int QR6_K = 2;
constexpr int QI8_1 = 8;
constexpr int WARP_SIZE = 32;

// Q6_K block structure (AoS format)
struct block_q6_K {
    uint8_t ql[128];   // 0-127
    uint8_t qh[64];    // 128-191
    int8_t scales[16]; // 192-207
    uint16_t d;        // 208-209 (ggml_half)
};
static_assert(sizeof(block_q6_K) == 210, "block_q6_K size mismatch");

// Tile helpers (same as in common.hpp)
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
        if (idx == tile_idx) return tile_size;
        blocks -= tile_size;
        idx++;
    }
    return 0;
}

// CPU reorder (matching ggml-sycl.cpp)
void reorder_q6_k_coalesced_cpu(uint8_t* dst, const block_q6_K* src, int blocks_per_row, int nrows) {
    const int num_tiles = tile_count(blocks_per_row);

    // Compute row stride
    size_t row_quants_bytes = 0;
    for (int t = 0; t < num_tiles; t++) {
        int ts = tile_size_at(blocks_per_row, t);
        row_quants_bytes += (size_t)ts * (128 + 64 + 16);
    }

    // D values at end of all quant data
    uint8_t* coal_d = dst + (size_t)nrows * row_quants_bytes;

    for (int row = 0; row < nrows; row++) {
        uint8_t* row_dst = dst + row * row_quants_bytes;
        int block_idx = 0;

        for (int tile = 0; tile < num_tiles; tile++) {
            const int tile_size = tile_size_at(blocks_per_row, tile);
            const int word_plane_stride = tile_size * 4;

            uint8_t* tile_ql = row_dst;
            uint8_t* tile_qh = tile_ql + tile_size * 128;
            int8_t* tile_sc = (int8_t*)(tile_qh + tile_size * 64);

            for (int b = 0; b < tile_size; b++) {
                const size_t global_block = row * blocks_per_row + block_idx + b;
                const block_q6_K* block = &src[global_block];

                // ql: 128 bytes = 32 words
                for (int word = 0; word < 32; word++) {
                    memcpy(tile_ql + word * word_plane_stride + b * 4, block->ql + word * 4, 4);
                }

                // qh: 64 bytes = 16 words
                for (int word = 0; word < 16; word++) {
                    memcpy(tile_qh + word * word_plane_stride + b * 4, block->qh + word * 4, 4);
                }

                // scales: 16 bytes = 4 words
                for (int word = 0; word < 4; word++) {
                    memcpy((uint8_t*)tile_sc + word * word_plane_stride + b * 4, block->scales + word * 4, 4);
                }

                // d at end
                memcpy(coal_d + global_block * sizeof(uint16_t), &block->d, sizeof(uint16_t));
            }

            row_dst = (uint8_t*)tile_sc + tile_size * 16;
            block_idx += tile_size;
        }
    }
}

// Simulate GPU kernel access pattern
// Returns the values accessed by lane_id in warp_id for block block_in_tile
void simulate_gpu_access(
    const uint8_t* x_base,
    int global_row, int row_quants_bytes,
    int warp_id, int blocks_per_row,
    // Output: accessed values
    int* out_vl, int* out_vh, int8_t* out_sc0, int8_t* out_sc1,
    int block_in_tile, int iqs)
{
    const int num_tiles = tile_count(blocks_per_row);

    // Compute tile_size and tile_offset (matching kernel logic)
    int tile_size = 0;
    int tile_offset = 0;
    {
        int remaining = blocks_per_row;
        int current_offset = 0;
        for (int t = 0; t <= warp_id && remaining > 0; t++) {
            int ts = 1;
            while (ts * 2 <= remaining && ts < 32) {
                ts *= 2;
            }
            if (t == warp_id) {
                tile_size = ts;
                tile_offset = current_offset;
            }
            current_offset += ts;
            remaining -= ts;
        }
    }

    // Compute tile_byte_offset (matching kernel logic)
    int tile_byte_offset = 0;
    {
        int remaining = blocks_per_row;
        int current_offset = 0;
        for (int t = 0; t < warp_id && remaining > 0; t++) {
            int ts = 1;
            while (ts * 2 <= remaining && ts < 32) {
                ts *= 2;
            }
            tile_byte_offset += ts * (128 + 64 + 16);
            current_offset += ts;
            remaining -= ts;
        }
    }

    const int word_plane_stride = tile_size * 4;

    const uint8_t* tile_base = x_base + global_row * row_quants_bytes + tile_byte_offset;
    const uint8_t* tile_ql = tile_base;
    const uint8_t* tile_qh = tile_ql + tile_size * 128;
    const int8_t* tile_sc = (const int8_t*)(tile_qh + tile_size * 64);

    // Access pattern for iqs (matching kernel)
    const int ql_offset = iqs * word_plane_stride + block_in_tile * 4;
    const int qh_word_idx = (QI6_K / 4) * (iqs / (QI6_K / 2)) + iqs % (QI6_K / 4);
    const int qh_offset = qh_word_idx * word_plane_stride + block_in_tile * 4;

    *out_vl = *((const int*)(tile_ql + ql_offset));
    *out_vh = *((const int*)(tile_qh + qh_offset));

    const int scale_offset = (QI6_K / 4) * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 8);
    const int sc_idx0 = scale_offset;
    const int sc_idx1 = scale_offset + 4;
    const int sc_word0 = sc_idx0 / 4;
    const int sc_byte0 = sc_idx0 % 4;
    const int sc_word1 = sc_idx1 / 4;
    const int sc_byte1 = sc_idx1 % 4;
    const int sc_offset0 = sc_word0 * word_plane_stride + block_in_tile * 4 + sc_byte0;
    const int sc_offset1 = sc_word1 * word_plane_stride + block_in_tile * 4 + sc_byte1;
    *out_sc0 = tile_sc[sc_offset0];
    *out_sc1 = tile_sc[sc_offset1];
}

int main() {
    // Test with Mistral FFN dimensions: 14336 elements = 56 blocks
    // This creates variable tiles: 32 + 16 + 8
    const int ncols = 14336;
    const int blocks_per_row = ncols / QK_K;  // 56
    const int nrows = 2;  // Just test 2 rows

    printf("Testing Q6_K variable tile layout:\n");
    printf("  ncols=%d, blocks_per_row=%d, nrows=%d\n", ncols, blocks_per_row, nrows);
    printf("  Tile decomposition: ");
    int num_tiles = tile_count(blocks_per_row);
    for (int t = 0; t < num_tiles; t++) {
        if (t > 0) printf("+");
        printf("%d", tile_size_at(blocks_per_row, t));
    }
    printf(" = %d tiles\n", num_tiles);

    // Create AoS source with deterministic values
    std::vector<block_q6_K> aos(blocks_per_row * nrows);
    for (int row = 0; row < nrows; row++) {
        for (int b = 0; b < blocks_per_row; b++) {
            size_t idx = row * blocks_per_row + b;
            block_q6_K& block = aos[idx];

            // Fill ql with pattern: row*1000 + block*10 + byte
            for (int i = 0; i < 128; i++) {
                block.ql[i] = (uint8_t)((idx * 1000 + i) & 0xFF);
            }
            // Fill qh
            for (int i = 0; i < 64; i++) {
                block.qh[i] = (uint8_t)((idx * 2000 + i) & 0xFF);
            }
            // Fill scales with signed values
            for (int i = 0; i < 16; i++) {
                block.scales[i] = (int8_t)((idx * 3 + i) & 0x7F);
            }
            // D value
            block.d = (uint16_t)(idx * 7 + 1);
        }
    }

    // Compute row_quants_bytes
    size_t row_quants_bytes = 0;
    for (int t = 0; t < num_tiles; t++) {
        int ts = tile_size_at(blocks_per_row, t);
        row_quants_bytes += (size_t)ts * (128 + 64 + 16);
    }
    printf("  row_quants_bytes=%zu\n", row_quants_bytes);

    // Create coalesced destination
    size_t total_size = nrows * row_quants_bytes + (size_t)blocks_per_row * nrows * sizeof(uint16_t);
    std::vector<uint8_t> coalesced(total_size, 0);

    // Reorder
    reorder_q6_k_coalesced_cpu(coalesced.data(), aos.data(), blocks_per_row, nrows);

    // Verify GPU access pattern matches expected values
    int errors = 0;
    for (int row = 0; row < nrows; row++) {
        for (int warp_id = 0; warp_id < num_tiles; warp_id++) {
            int tile_size = tile_size_at(blocks_per_row, warp_id);
            int tile_offset = 0;
            for (int t = 0; t < warp_id; t++) {
                tile_offset += tile_size_at(blocks_per_row, t);
            }

            for (int block_in_tile = 0; block_in_tile < tile_size && block_in_tile < 4; block_in_tile++) {
                int global_block = row * blocks_per_row + tile_offset + block_in_tile;
                const block_q6_K& expected_block = aos[global_block];

                // Test a few iqs values
                for (int iqs : {0, 8, 16, 24}) {
                    int gpu_vl, gpu_vh;
                    int8_t gpu_sc0, gpu_sc1;

                    simulate_gpu_access(
                        coalesced.data(), row, row_quants_bytes,
                        warp_id, blocks_per_row,
                        &gpu_vl, &gpu_vh, &gpu_sc0, &gpu_sc1,
                        block_in_tile, iqs);

                    // Expected ql value: 4 bytes at offset iqs*4
                    int expected_vl;
                    memcpy(&expected_vl, expected_block.ql + iqs * 4, 4);

                    // Expected qh value
                    int qh_word_idx = (QI6_K / 4) * (iqs / (QI6_K / 2)) + iqs % (QI6_K / 4);
                    int expected_vh;
                    memcpy(&expected_vh, expected_block.qh + qh_word_idx * 4, 4);

                    // Expected scales
                    int scale_offset = (QI6_K / 4) * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 8);
                    int8_t expected_sc0 = expected_block.scales[scale_offset];
                    int8_t expected_sc1 = expected_block.scales[scale_offset + 4];

                    if (gpu_vl != expected_vl) {
                        printf("MISMATCH row=%d warp=%d block=%d iqs=%d: vl GPU=0x%08X expected=0x%08X\n",
                               row, warp_id, block_in_tile, iqs, gpu_vl, expected_vl);
                        errors++;
                    }
                    if (gpu_vh != expected_vh) {
                        printf("MISMATCH row=%d warp=%d block=%d iqs=%d: vh GPU=0x%08X expected=0x%08X\n",
                               row, warp_id, block_in_tile, iqs, gpu_vh, expected_vh);
                        errors++;
                    }
                    if (gpu_sc0 != expected_sc0 || gpu_sc1 != expected_sc1) {
                        printf("MISMATCH row=%d warp=%d block=%d iqs=%d: scales GPU=%d,%d expected=%d,%d\n",
                               row, warp_id, block_in_tile, iqs, gpu_sc0, gpu_sc1, expected_sc0, expected_sc1);
                        errors++;
                    }
                }
            }
        }
    }

    // Verify D values
    const uint16_t* d_vals = (const uint16_t*)(coalesced.data() + nrows * row_quants_bytes);
    for (int row = 0; row < nrows; row++) {
        for (int b = 0; b < blocks_per_row; b++) {
            int global_block = row * blocks_per_row + b;
            uint16_t expected_d = aos[global_block].d;
            uint16_t actual_d = d_vals[global_block];
            if (expected_d != actual_d) {
                printf("MISMATCH D value row=%d block=%d: GPU=%d expected=%d\n",
                       row, b, actual_d, expected_d);
                errors++;
            }
        }
    }

    if (errors == 0) {
        printf("\nAll layout verification tests PASSED!\n");
    } else {
        printf("\nFAILED: %d mismatches\n", errors);
        return 1;
    }

    return 0;
}
