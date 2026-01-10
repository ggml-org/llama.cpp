// tests/test-q6k-56block-debug.cpp
// Debug test for Q6_K 56-block variable tile coalesced layout
// This tests the exact case that Mistral's FFN layers use (14336 / 256 = 56 blocks per row)

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <cassert>

// Q6_K block structure (210 bytes)
#define QK_K 256
struct block_q6_K {
    uint8_t ql[QK_K/2];     // 128 bytes: lower 4 bits of quants
    uint8_t qh[QK_K/4];     // 64 bytes: upper 2 bits of quants
    int8_t  scales[QK_K/16]; // 16 bytes: scales
    uint16_t d;              // 2 bytes: super-block scale (really ggml_half)
};

static_assert(sizeof(block_q6_K) == 210, "block_q6_K size mismatch");

// Tile decomposition helpers (must match common.hpp)
inline int tile_count(int blocks) {
    int count = 0;
    while (blocks > 0) {
        int ts = 1;
        while (ts * 2 <= blocks && ts < 32) ts *= 2;
        count++;
        blocks -= ts;
    }
    return count;
}

inline int tile_size_at(int blocks, int tile_idx) {
    for (int t = 0; t < tile_idx; t++) {
        int ts = 1;
        while (ts * 2 <= blocks && ts < 32) ts *= 2;
        blocks -= ts;
    }
    int ts = 1;
    while (ts * 2 <= blocks && ts < 32) ts *= 2;
    return ts;
}

inline int tile_offset_at(int blocks, int tile_idx) {
    int offset = 0;
    for (int t = 0; t < tile_idx; t++) {
        int ts = 1;
        while (ts * 2 <= blocks && ts < 32) ts *= 2;
        offset += ts;
        blocks -= ts;
    }
    return offset;
}

// CPU reorder: AoS -> Variable Tile Coalesced (copy from ggml-sycl.cpp)
void reorder_q6_k_coalesced_cpu(void* dst_coalesced, const void* src_aos, int ncols, int nrows) {
    const int blocks_per_row = ncols / QK_K;
    const int num_tiles = tile_count(blocks_per_row);

    // Compute row stride
    size_t row_quants_bytes = 0;
    for (int t = 0; t < num_tiles; t++) {
        int ts = tile_size_at(blocks_per_row, t);
        row_quants_bytes += (size_t)ts * (128 + 64 + 16);
    }

    // AoS input
    const uint8_t* aos = (const uint8_t*)src_aos;

    // Coalesced layout: [row0_tiles][row1_tiles]...[all d values]
    uint8_t* coal_d = (uint8_t*)dst_coalesced + (size_t)nrows * row_quants_bytes;

    printf("[CPU REORDER] blocks_per_row=%d, num_tiles=%d, row_stride=%zu, d_offset=%zu\n",
           blocks_per_row, num_tiles, row_quants_bytes, (size_t)nrows * row_quants_bytes);

    for (int row = 0; row < nrows; row++) {
        uint8_t* row_dst = (uint8_t*)dst_coalesced + row * row_quants_bytes;

        int block_idx = 0;
        for (int tile = 0; tile < num_tiles; tile++) {
            int tile_size = tile_size_at(blocks_per_row, tile);
            int word_plane_stride = tile_size * 4;

            // Tile layout: [ql][qh][scales]
            uint8_t* tile_ql = row_dst;
            uint8_t* tile_qh = tile_ql + tile_size * 128;
            int8_t*  tile_sc = (int8_t*)(tile_qh + tile_size * 64);

            if (row == 0 && tile == 0) {
                printf("[CPU REORDER] row=0 tile=0: tile_size=%d, tile_ql=0, tile_qh=%d, tile_sc=%d\n",
                       tile_size, (int)(tile_qh - (uint8_t*)dst_coalesced), (int)((uint8_t*)tile_sc - (uint8_t*)dst_coalesced));
            }

            for (int b = 0; b < tile_size; b++) {
                int global_block = row * blocks_per_row + block_idx + b;
                const block_q6_K* src_block = (const block_q6_K*)(aos + global_block * sizeof(block_q6_K));

                const uint8_t* src_ql = src_block->ql;
                const uint8_t* src_qh = src_block->qh;
                const int8_t*  src_sc = src_block->scales;
                const uint8_t* src_d  = (const uint8_t*)&src_block->d;

                // Word-major layout: word 0 of all blocks, then word 1 of all blocks, etc.
                // ql: 32 words of 4 bytes each
                for (int word = 0; word < 32; word++) {
                    memcpy(tile_ql + word * word_plane_stride + b * 4, src_ql + word * 4, 4);
                }
                // qh: 16 words of 4 bytes each
                for (int word = 0; word < 16; word++) {
                    memcpy(tile_qh + word * word_plane_stride + b * 4, src_qh + word * 4, 4);
                }
                // scales: 4 words of 4 bytes each
                for (int word = 0; word < 4; word++) {
                    memcpy(tile_sc + word * word_plane_stride + b * 4, src_sc + word * 4, 4);
                }
                // D value stored contiguously
                memcpy(coal_d + global_block * sizeof(uint16_t), src_d, sizeof(uint16_t));
            }

            // Advance to next tile
            row_dst = (uint8_t*)tile_sc + tile_size * 16;
            block_idx += tile_size;
        }
    }
}

// Simulate GPU kernel access for one block
struct BlockData {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    uint16_t d;
};

void simulate_gpu_read_block(const uint8_t* x_base, int total_nrows, int row_quants_bytes,
                              int blocks_per_row, int num_tiles,
                              int global_row, int warp_id, int lane_id,
                              BlockData& result) {
    // Compute tile info for this warp
    int tile_size = 0;
    int tile_offset = 0;
    {
        int remaining = blocks_per_row;
        int current_offset = 0;
        for (int t = 0; t <= warp_id && remaining > 0; t++) {
            int ts = 1;
            while (ts * 2 <= remaining && ts < 32) ts *= 2;
            if (t == warp_id) {
                tile_size = ts;
                tile_offset = current_offset;
            }
            current_offset += ts;
            remaining -= ts;
        }
    }

    // Compute byte offset to this tile
    int tile_byte_offset = 0;
    {
        int remaining = blocks_per_row;
        for (int t = 0; t < warp_id && remaining > 0; t++) {
            int ts = 1;
            while (ts * 2 <= remaining && ts < 32) ts *= 2;
            tile_byte_offset += ts * (128 + 64 + 16);
            remaining -= ts;
        }
    }

    int word_plane_stride = tile_size * 4;

    // Tile base pointer
    const uint8_t* tile_base = x_base + global_row * row_quants_bytes + tile_byte_offset;
    const uint8_t* tile_ql = tile_base;
    const uint8_t* tile_qh = tile_ql + tile_size * 128;
    const int8_t* tile_sc = (const int8_t*)(tile_qh + tile_size * 64);

    // D values at end of tensor
    const uint16_t* x_d = (const uint16_t*)(x_base + (size_t)total_nrows * row_quants_bytes);

    int block_in_tile = lane_id;
    if (lane_id >= tile_size) {
        return;  // This lane doesn't process a block
    }

    // Read ql (32 words, word-major)
    for (int word = 0; word < 32; word++) {
        int offset = word * word_plane_stride + block_in_tile * 4;
        memcpy(result.ql + word * 4, tile_ql + offset, 4);
    }

    // Read qh (16 words, word-major)
    for (int word = 0; word < 16; word++) {
        int offset = word * word_plane_stride + block_in_tile * 4;
        memcpy(result.qh + word * 4, tile_qh + offset, 4);
    }

    // Read scales (4 words, word-major)
    for (int word = 0; word < 4; word++) {
        int offset = word * word_plane_stride + block_in_tile * 4;
        memcpy(result.scales + word * 4, tile_sc + offset, 4);
    }

    // Read D value
    int global_block_idx = global_row * blocks_per_row + tile_offset + block_in_tile;
    result.d = x_d[global_block_idx];
}

int main() {
    printf("=== Q6_K 56-block Variable Tile Debug Test ===\n\n");

    const int ncols = 14336;  // Mistral FFN dimension
    const int nrows = 4;      // Just a few rows for testing
    const int blocks_per_row = ncols / QK_K;  // 56 blocks

    printf("Configuration:\n");
    printf("  ncols = %d\n", ncols);
    printf("  nrows = %d\n", nrows);
    printf("  blocks_per_row = %d\n", blocks_per_row);

    int num_tiles = tile_count(blocks_per_row);
    printf("  num_tiles = %d\n", num_tiles);

    printf("\nTile decomposition:\n");
    int check_blocks = 0;
    for (int t = 0; t < num_tiles; t++) {
        int ts = tile_size_at(blocks_per_row, t);
        int to = tile_offset_at(blocks_per_row, t);
        printf("  Tile %d: size=%d, offset=%d\n", t, ts, to);
        check_blocks += ts;
    }
    printf("  Total blocks: %d (expected %d)\n", check_blocks, blocks_per_row);
    assert(check_blocks == blocks_per_row);

    // Compute row stride
    size_t row_quants_bytes = 0;
    for (int t = 0; t < num_tiles; t++) {
        int ts = tile_size_at(blocks_per_row, t);
        row_quants_bytes += (size_t)ts * (128 + 64 + 16);
    }
    printf("\nMemory layout:\n");
    printf("  row_quants_bytes = %zu\n", row_quants_bytes);
    printf("  total quants size = %zu bytes\n", (size_t)nrows * row_quants_bytes);
    printf("  d values at offset = %zu\n", (size_t)nrows * row_quants_bytes);
    printf("  d values size = %zu bytes\n", (size_t)nrows * blocks_per_row * sizeof(uint16_t));

    // Create test data in AoS format
    size_t aos_size = (size_t)nrows * blocks_per_row * sizeof(block_q6_K);
    std::vector<uint8_t> aos_data(aos_size);

    // Fill with recognizable patterns
    for (int row = 0; row < nrows; row++) {
        for (int block = 0; block < blocks_per_row; block++) {
            block_q6_K* b = (block_q6_K*)(aos_data.data() + (row * blocks_per_row + block) * sizeof(block_q6_K));

            // Fill ql with pattern: row in high nibble, block in low nibble (mod 16)
            for (int i = 0; i < 128; i++) {
                b->ql[i] = ((row & 0xF) << 4) | ((block + i) & 0xF);
            }
            // Fill qh similarly
            for (int i = 0; i < 64; i++) {
                b->qh[i] = ((row & 0xF) << 4) | ((block + i + 1) & 0xF);
            }
            // Fill scales
            for (int i = 0; i < 16; i++) {
                b->scales[i] = (int8_t)((row * 16 + block + i) & 0x7F);
            }
            // D value: row * 1000 + block
            b->d = (uint16_t)(row * 1000 + block);
        }
    }

    printf("\nCreated AoS test data: %zu bytes\n", aos_size);

    // Allocate coalesced buffer
    size_t coal_size = (size_t)nrows * row_quants_bytes + (size_t)nrows * blocks_per_row * sizeof(uint16_t);
    std::vector<uint8_t> coal_data(coal_size, 0xCC);  // Fill with 0xCC to detect unwritten areas

    printf("Allocated coalesced buffer: %zu bytes\n", coal_size);

    // Perform CPU reorder
    printf("\n--- Performing CPU reorder ---\n");
    reorder_q6_k_coalesced_cpu(coal_data.data(), aos_data.data(), ncols, nrows);

    // Now verify by simulating GPU reads
    printf("\n--- Simulating GPU reads ---\n");

    int errors = 0;
    for (int row = 0; row < nrows; row++) {
        for (int tile = 0; tile < num_tiles; tile++) {
            int tile_size = tile_size_at(blocks_per_row, tile);
            int tile_offset = tile_offset_at(blocks_per_row, tile);

            for (int lane = 0; lane < tile_size; lane++) {
                int global_block = row * blocks_per_row + tile_offset + lane;

                // Get original AoS data
                const block_q6_K* orig = (const block_q6_K*)(aos_data.data() + global_block * sizeof(block_q6_K));

                // Simulate GPU read
                BlockData gpu_read = {};
                simulate_gpu_read_block(coal_data.data(), nrows, row_quants_bytes,
                                        blocks_per_row, num_tiles,
                                        row, tile, lane, gpu_read);

                // Compare
                bool match = true;
                if (memcmp(gpu_read.ql, orig->ql, 128) != 0) {
                    match = false;
                    if (errors < 5) {
                        printf("MISMATCH: row=%d tile=%d lane=%d block=%d: ql differs\n",
                               row, tile, lane, global_block);
                        printf("  Expected ql[0-7]: %02x %02x %02x %02x %02x %02x %02x %02x\n",
                               orig->ql[0], orig->ql[1], orig->ql[2], orig->ql[3],
                               orig->ql[4], orig->ql[5], orig->ql[6], orig->ql[7]);
                        printf("  Got ql[0-7]:      %02x %02x %02x %02x %02x %02x %02x %02x\n",
                               gpu_read.ql[0], gpu_read.ql[1], gpu_read.ql[2], gpu_read.ql[3],
                               gpu_read.ql[4], gpu_read.ql[5], gpu_read.ql[6], gpu_read.ql[7]);
                    }
                }
                if (memcmp(gpu_read.qh, orig->qh, 64) != 0) {
                    match = false;
                    if (errors < 5) {
                        printf("MISMATCH: row=%d tile=%d lane=%d block=%d: qh differs\n",
                               row, tile, lane, global_block);
                    }
                }
                if (memcmp(gpu_read.scales, orig->scales, 16) != 0) {
                    match = false;
                    if (errors < 5) {
                        printf("MISMATCH: row=%d tile=%d lane=%d block=%d: scales differs\n",
                               row, tile, lane, global_block);
                    }
                }
                if (gpu_read.d != orig->d) {
                    match = false;
                    if (errors < 5) {
                        printf("MISMATCH: row=%d tile=%d lane=%d block=%d: d differs (got %u, expected %u)\n",
                               row, tile, lane, global_block, gpu_read.d, orig->d);
                    }
                }

                if (!match) {
                    errors++;
                }
            }
        }
    }

    printf("\n=== Summary ===\n");
    printf("Total blocks tested: %d\n", nrows * blocks_per_row);
    printf("Errors: %d\n", errors);

    if (errors == 0) {
        printf("PASS: All GPU reads match original AoS data\n");
        return 0;
    } else {
        printf("FAIL: %d mismatches detected\n", errors);
        return 1;
    }
}
