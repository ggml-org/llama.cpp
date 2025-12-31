// Test 4: Tile index verification test for Q6_K SoA kernel
// Dumps all tile indices each thread computes, compares SoA vs AoS
// This tests whether index calculations match between the two kernels

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>

// Constants
#define QK_K 256
#define QI6_K 32
#define QR6_K 2
#define WARP_SIZE 32
#define MMQ_TILE_NE_K 32

// Tile dimensions
constexpr int mmq_y = 128;
constexpr int nwarps = 4;

// Structure to capture indices computed by each thread
struct ThreadIndices {
    // For x_ql loading (per iteration of i0 loop)
    int ql_global_row[mmq_y / nwarps];
    int ql_global_block[mmq_y / nwarps];
    int ql_kq0[mmq_y / nwarps];
    int ql_kq1[mmq_y / nwarps];
    int ql_tile_idx0[mmq_y / nwarps];
    int ql_tile_idx1[mmq_y / nwarps];

    // For x_dm loading
    int dm_i;
    int dm_global_row;
    int dm_global_block;
    int dm_tile_idx;

    // For x_sc loading
    int sc_i;
    int sc_global_row;
    int sc_global_block;
    int sc_tile_idx;
};

// Compute SoA indices (matches load_tiles_q6_K_soa)
void compute_soa_indices(
    int i_offset, int k,
    int blocks_per_row, int row_offset, int block_offset, int row_low,
    ThreadIndices *indices)
{
    const int kbx = k / QI6_K;  // 0 for QK_K=256
    const int kqsx = k % QI6_K; // = k

    // x_ql indices
    int iter = 0;
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        const int global_row = row_low + row_offset + i;
        const int global_block = global_row * blocks_per_row + block_offset + kbx;

        const int ky = QR6_K * kqsx;
        const int kq0 = ky - ky % QI6_K + k % (QI6_K / 2) + 0;
        const int kq1 = ky - ky % QI6_K + k % (QI6_K / 2) + (QI6_K / 2);

        indices->ql_global_row[iter] = global_row;
        indices->ql_global_block[iter] = global_block;
        indices->ql_kq0[iter] = kq0;
        indices->ql_kq1[iter] = kq1;
        indices->ql_tile_idx0[iter] = i * (2 * MMQ_TILE_NE_K + 1) + kq0;
        indices->ql_tile_idx1[iter] = i * (2 * MMQ_TILE_NE_K + 1) + kq1;
        iter++;
    }

    // x_dm indices
    constexpr int blocks_per_tile_x_row = QI6_K > MMQ_TILE_NE_K ? 1 : MMQ_TILE_NE_K / QI6_K;
    const int kbxd = k % blocks_per_tile_x_row;

    int dm_i = (0 + i_offset * QI6_K + k / blocks_per_tile_x_row) % mmq_y;
    int dm_global_row = row_low + row_offset + dm_i;
    int dm_global_block = dm_global_row * blocks_per_row + block_offset + kbxd;

    indices->dm_i = dm_i;
    indices->dm_global_row = dm_global_row;
    indices->dm_global_block = dm_global_block;
    indices->dm_tile_idx = dm_i * (MMQ_TILE_NE_K / QI6_K) + dm_i / QI6_K + kbxd;

    // x_sc indices
    int sc_i = (0 + i_offset * 8 + k / (MMQ_TILE_NE_K / 8)) % mmq_y;
    int sc_global_row = row_low + row_offset + sc_i;
    int sc_global_block = sc_global_row * blocks_per_row + block_offset + (k % (MMQ_TILE_NE_K / 8)) / 4;

    indices->sc_i = sc_i;
    indices->sc_global_row = sc_global_row;
    indices->sc_global_block = sc_global_block;
    indices->sc_tile_idx = sc_i * (MMQ_TILE_NE_K / 8) + sc_i / 8 + k % (MMQ_TILE_NE_K / 8);
}

// Compute AoS indices (matches load_tiles_q6_K with pre-offset pointer)
// In AoS, the pointer is already offset to row_x_0 * blocks_per_row + ib0
// So global_block = i * blocks_per_row + kbx (relative to offset pointer)
void compute_aos_indices(
    int i_offset, int k,
    int blocks_per_row,
    ThreadIndices *indices)
{
    const int kbx = k / QI6_K;
    const int kqsx = k % QI6_K;

    int iter = 0;
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;

        // AoS uses relative indexing (pointer already offset)
        const int relative_block = i * blocks_per_row + kbx;

        const int ky = QR6_K * kqsx;
        const int kq0 = ky - ky % QI6_K + k % (QI6_K / 2) + 0;
        const int kq1 = ky - ky % QI6_K + k % (QI6_K / 2) + (QI6_K / 2);

        indices->ql_global_row[iter] = i;  // Local row
        indices->ql_global_block[iter] = relative_block;
        indices->ql_kq0[iter] = kq0;
        indices->ql_kq1[iter] = kq1;
        indices->ql_tile_idx0[iter] = i * (2 * MMQ_TILE_NE_K + 1) + kq0;
        indices->ql_tile_idx1[iter] = i * (2 * MMQ_TILE_NE_K + 1) + kq1;
        iter++;
    }

    // x_dm indices
    constexpr int blocks_per_tile_x_row = QI6_K > MMQ_TILE_NE_K ? 1 : MMQ_TILE_NE_K / QI6_K;
    const int kbxd = k % blocks_per_tile_x_row;

    int dm_i = (0 + i_offset * QI6_K + k / blocks_per_tile_x_row) % mmq_y;

    indices->dm_i = dm_i;
    indices->dm_global_row = dm_i;
    indices->dm_global_block = dm_i * blocks_per_row + kbxd;
    indices->dm_tile_idx = dm_i * (MMQ_TILE_NE_K / QI6_K) + dm_i / QI6_K + kbxd;

    // x_sc indices
    int sc_i = (0 + i_offset * 8 + k / (MMQ_TILE_NE_K / 8)) % mmq_y;

    indices->sc_i = sc_i;
    indices->sc_global_row = sc_i;
    indices->sc_global_block = sc_i * blocks_per_row + (k % (MMQ_TILE_NE_K / 8)) / 4;
    indices->sc_tile_idx = sc_i * (MMQ_TILE_NE_K / 8) + sc_i / 8 + k % (MMQ_TILE_NE_K / 8);
}

int main() {
    printf("=== Test 4: Tile index verification ===\n");
    printf("Compares index calculations between SoA and AoS kernels\n\n");

    const int blocks_per_row = 1;  // Single block per row
    const int row_offset = 0;      // Start at row 0
    const int block_offset = 0;    // Start at block 0
    const int row_low = 0;         // No row offset in split

    printf("Parameters: blocks_per_row=%d, row_offset=%d, block_offset=%d, row_low=%d\n\n",
           blocks_per_row, row_offset, block_offset, row_low);

    // Compare indices for all thread combinations
    int total_threads = nwarps * WARP_SIZE;
    ThreadIndices *soa_indices = new ThreadIndices[total_threads];
    ThreadIndices *aos_indices = new ThreadIndices[total_threads];

    for (int i_offset = 0; i_offset < nwarps; i_offset++) {
        for (int k = 0; k < WARP_SIZE; k++) {
            int tid = i_offset * WARP_SIZE + k;
            compute_soa_indices(i_offset, k, blocks_per_row, row_offset, block_offset, row_low, &soa_indices[tid]);
            compute_aos_indices(i_offset, k, blocks_per_row, &aos_indices[tid]);
        }
    }

    // Compare tile indices (these should match between SoA and AoS)
    printf("Checking tile indices match (SoA vs AoS)...\n");

    int ql_tile_mismatches = 0;
    int dm_tile_mismatches = 0;
    int sc_tile_mismatches = 0;

    for (int tid = 0; tid < total_threads; tid++) {
        for (int iter = 0; iter < mmq_y / nwarps; iter++) {
            if (soa_indices[tid].ql_tile_idx0[iter] != aos_indices[tid].ql_tile_idx0[iter]) {
                if (ql_tile_mismatches == 0) {
                    printf("  MISMATCH x_ql tile_idx0: tid=%d, iter=%d, SoA=%d, AoS=%d\n",
                           tid, iter, soa_indices[tid].ql_tile_idx0[iter], aos_indices[tid].ql_tile_idx0[iter]);
                }
                ql_tile_mismatches++;
            }
            if (soa_indices[tid].ql_tile_idx1[iter] != aos_indices[tid].ql_tile_idx1[iter]) {
                if (ql_tile_mismatches == 0) {
                    printf("  MISMATCH x_ql tile_idx1: tid=%d, iter=%d, SoA=%d, AoS=%d\n",
                           tid, iter, soa_indices[tid].ql_tile_idx1[iter], aos_indices[tid].ql_tile_idx1[iter]);
                }
                ql_tile_mismatches++;
            }
        }

        if (soa_indices[tid].dm_tile_idx != aos_indices[tid].dm_tile_idx) {
            if (dm_tile_mismatches == 0) {
                printf("  MISMATCH x_dm tile_idx: tid=%d, SoA=%d, AoS=%d\n",
                       tid, soa_indices[tid].dm_tile_idx, aos_indices[tid].dm_tile_idx);
            }
            dm_tile_mismatches++;
        }

        if (soa_indices[tid].sc_tile_idx != aos_indices[tid].sc_tile_idx) {
            if (sc_tile_mismatches == 0) {
                printf("  MISMATCH x_sc tile_idx: tid=%d, SoA=%d, AoS=%d\n",
                       tid, soa_indices[tid].sc_tile_idx, aos_indices[tid].sc_tile_idx);
            }
            sc_tile_mismatches++;
        }
    }

    printf("x_ql tile index mismatches: %d\n", ql_tile_mismatches);
    printf("x_dm tile index mismatches: %d\n", dm_tile_mismatches);
    printf("x_sc tile index mismatches: %d\n", sc_tile_mismatches);

    // Show sample indices for verification
    printf("\n--- Sample indices for thread (i_offset=0, k=0) ---\n");
    int tid0 = 0;
    printf("SoA: dm_i=%d, dm_global_row=%d, dm_global_block=%d, dm_tile_idx=%d\n",
           soa_indices[tid0].dm_i, soa_indices[tid0].dm_global_row,
           soa_indices[tid0].dm_global_block, soa_indices[tid0].dm_tile_idx);
    printf("AoS: dm_i=%d, dm_global_row=%d, dm_global_block=%d, dm_tile_idx=%d\n",
           aos_indices[tid0].dm_i, aos_indices[tid0].dm_global_row,
           aos_indices[tid0].dm_global_block, aos_indices[tid0].dm_tile_idx);

    printf("\n--- Sample indices for thread (i_offset=1, k=16) ---\n");
    int tid1 = 1 * WARP_SIZE + 16;
    printf("SoA: ql_global_row[0]=%d, ql_global_block[0]=%d, ql_kq0[0]=%d\n",
           soa_indices[tid1].ql_global_row[0], soa_indices[tid1].ql_global_block[0],
           soa_indices[tid1].ql_kq0[0]);
    printf("AoS: ql_global_row[0]=%d, ql_global_block[0]=%d, ql_kq0[0]=%d\n",
           aos_indices[tid1].ql_global_row[0], aos_indices[tid1].ql_global_block[0],
           aos_indices[tid1].ql_kq0[0]);

    // Verify global block indices make sense for SoA
    printf("\n--- Verifying SoA global block indices ---\n");
    bool soa_blocks_valid = true;
    int max_block = 0;
    for (int tid = 0; tid < total_threads; tid++) {
        for (int iter = 0; iter < mmq_y / nwarps; iter++) {
            int block = soa_indices[tid].ql_global_block[iter];
            if (block > max_block) max_block = block;
            if (block < 0) {
                printf("ERROR: Negative block index at tid=%d, iter=%d: %d\n", tid, iter, block);
                soa_blocks_valid = false;
            }
        }
    }
    printf("Max global block index: %d (expected max: %d)\n", max_block, mmq_y * blocks_per_row - 1);

    // Summary
    bool pass = (ql_tile_mismatches == 0 && dm_tile_mismatches == 0 &&
                 sc_tile_mismatches == 0 && soa_blocks_valid);
    printf("\n=== %s ===\n", pass ? "PASS" : "FAIL");

    if (!pass) {
        printf("\nNote: If tile indices match but global block indices differ,\n");
        printf("that's expected because SoA uses absolute indices while AoS uses relative.\n");
        printf("The key test is that TILE indices match (what gets written to shared memory).\n");
    }

    delete[] soa_indices;
    delete[] aos_indices;

    return pass ? 0 : 1;
}
