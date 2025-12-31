// GPU Pipeline Test for Q4_0 MMQ SoA vs AoS - Tests actual tile loading and vec_dot functions
// This test calls the same functions used by the real MMQ kernel to find where SoA diverges from AoS
//
// Build: cmake --build build --target test-mmq-soa-q4-0-pipeline
// Run: ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-mmq-soa-q4-0-pipeline

#include <sycl/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>

// ============================================================================
// Constants from mmq.cpp (must match exactly)
// ============================================================================
#define WARP_SIZE 32
#define MMQ_TILE_NE_K 32
#define QK4_0 32
#define QI4_0 4
#define QR4_0 2
#define QK8_1 32
#define QI8_1 4
#define VDR_Q4_0_Q8_1_MMQ 4

// ============================================================================
// Block structures (from ggml-common.h)
// ============================================================================
typedef struct {
    sycl::half d;
    uint8_t qs[QK4_0/2];
} block_q4_0;

typedef struct {
    sycl::half2 ds;  // d and sum
    int8_t qs[QK8_1];
} block_q8_1;

static_assert(sizeof(block_q4_0) == 18, "block_q4_0 size mismatch");
static_assert(sizeof(block_q8_1) == 36, "block_q8_1 size mismatch");

// ============================================================================
// Helper functions (from vecdotq.hpp)
// ============================================================================
static inline int get_int_from_uint8(const uint8_t* x8, const int& i32) {
    const uint16_t* x16 = (const uint16_t*)(x8 + sizeof(int) * i32);
    int x32 = 0;
    x32 |= x16[0] << 0;
    x32 |= x16[1] << 16;
    return x32;
}

// dpct::dp4a equivalent - dot product of 4 int8 values
static inline int dp4a(int a, int b, int c) {
    int8_t* a8 = (int8_t*)&a;
    int8_t* b8 = (int8_t*)&b;
    return c + a8[0]*b8[0] + a8[1]*b8[1] + a8[2]*b8[2] + a8[3]*b8[3];
}

// ============================================================================
// vec_dot_q4_0_q8_1_impl (from vecdotq.hpp line 564)
// ============================================================================
template <int vdr>
static inline float vec_dot_q4_0_q8_1_impl(const int* v, const int* u, const float& d4,
                                           const sycl::half2& ds8) {
    int sumi = 0;
    for (int i = 0; i < vdr; ++i) {
        const int vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        const int vi1 = (v[i] >> 4) & 0x0F0F0F0F;
        sumi = dp4a(vi0, u[2 * i + 0], sumi);
        sumi = dp4a(vi1, u[2 * i + 1], sumi);
    }
    float ds8_x = (float)ds8.x();
    float ds8_y = (float)ds8.y();
    return d4 * (sumi * ds8_x - (8 * vdr / QI4_0) * ds8_y);
}

// ============================================================================
// load_tiles_q4_0 - AoS version (from mmq.cpp lines 82-128)
// ============================================================================
template <int mmq_y, int nwarps, bool need_check>
static void load_tiles_q4_0_aos(
    const void* __restrict__ vx,
    int* __restrict__ x_ql,
    float* __restrict__ x_dmf,
    const int& i_offset, const int& i_max,
    const int& k, const int& blocks_per_row)
{
    const int kbx  = k / QI4_0;
    const int kqsx = k % QI4_0;
    const block_q4_0* bx0 = (const block_q4_0*)vx;

    // Load quantized values
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;
        if (need_check) {
            i = std::min(i, i_max);
        }
        const block_q4_0* bxi = bx0 + i * blocks_per_row + kbx;
        x_ql[i * (WARP_SIZE + 1) + k] = get_int_from_uint8(bxi->qs, kqsx);
    }

    // Load scales
    const int blocks_per_tile_x_row = WARP_SIZE / QI4_0;
    const int kbxd = k % blocks_per_tile_x_row;

    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_0) {
        int i = i0 + i_offset * QI4_0 + k / blocks_per_tile_x_row;
        if (need_check) {
            i = std::min(i, i_max);
        }
        const block_q4_0* bxi = bx0 + i * blocks_per_row + kbxd;
        x_dmf[i * (WARP_SIZE/QI4_0) + i / QI4_0 + kbxd] = (float)bxi->d;
    }
}

// ============================================================================
// load_tiles_q4_0_soa - SoA version (from mmq.cpp lines 137-198)
// ============================================================================
template <int mmq_y, int nwarps, bool need_check>
static void load_tiles_q4_0_soa(
    const uint8_t* __restrict__ qs_base,
    const size_t d_offset,
    int* __restrict__ x_ql,
    float* __restrict__ x_dmf,
    const int& i_offset, const int& i_max,
    const int& k, const int& blocks_per_row,
    const int& row_offset, const int& block_offset)
{
    const int kbx  = k / QI4_0;
    const int kqsx = k % QI4_0;

    const sycl::half* d_base = (const sycl::half*)(qs_base + d_offset);

    // Load quantized values from SoA layout
    for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
        int i = i0 + i_offset;
        if (need_check) {
            i = std::min(i, i_max);
        }
        const int global_row = row_offset + i;
        const int global_block = global_row * blocks_per_row + block_offset + kbx;
        const uint8_t* qs_ptr = qs_base + global_block * (QK4_0/2);
        x_ql[i * (WARP_SIZE + 1) + k] = get_int_from_uint8(qs_ptr, kqsx);
    }

    // Load scales from SoA layout
    const int ints_per_block = QK4_0 / 8;  // 4
    const int blocks_per_tile_x_row = WARP_SIZE / ints_per_block;  // 8
    const int kbxd = k % blocks_per_tile_x_row;

    for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_0) {
        int i = i0 + i_offset * QI4_0 + k / blocks_per_tile_x_row;
        if (need_check) {
            i = std::min(i, i_max);
        }
        const int global_row = row_offset + i;
        const int global_block = global_row * blocks_per_row + block_offset + kbxd;
        x_dmf[i * (WARP_SIZE/QI4_0) + i / QI4_0 + kbxd] = (float)d_base[global_block];
    }
}

// ============================================================================
// vec_dot_q4_0_q8_1_mul_mat - AoS/SoA vec_dot (same for both, from mmq.cpp lines 200-225)
// ============================================================================
static float vec_dot_q4_0_q8_1_mul_mat(
    const int* __restrict__ x_ql, const float* __restrict__ x_dmf,
    const int* __restrict__ y_qs, const sycl::half2* __restrict__ y_ds,
    const int& i, const int& j, const int& k)
{
    const int kyqs = k % (QI8_1/2) + QI8_1 * (k / (QI8_1/2));

    int u[2*VDR_Q4_0_Q8_1_MMQ];
    for (int l = 0; l < VDR_Q4_0_Q8_1_MMQ; ++l) {
        u[2*l+0] = y_qs[j * MMQ_TILE_NE_K + (kyqs + l)         % MMQ_TILE_NE_K];
        u[2*l+1] = y_qs[j * MMQ_TILE_NE_K + (kyqs + l + QI4_0) % MMQ_TILE_NE_K];
    }

    return vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMQ>(
        &x_ql[i * (WARP_SIZE + 1) + k], u, x_dmf[i * (WARP_SIZE/QI4_0) + i/QI4_0 + k/QI4_0],
        y_ds[j * (MMQ_TILE_NE_K/QI8_1) + (2*k/QI8_1) % (MMQ_TILE_NE_K/QI8_1)]);
}

// ============================================================================
// Reorder function (from ggml-sycl.cpp)
// ============================================================================
void reorder_aos_to_soa(const block_q4_0* aos, uint8_t* soa, int nrows, int ncols) {
    const int nblocks = nrows * (ncols / QK4_0);
    const size_t d_offset = (size_t)nrows * ncols / 2;
    uint8_t* qs_ptr = soa;
    sycl::half* d_ptr = (sycl::half*)(soa + d_offset);

    for (int ib = 0; ib < nblocks; ib++) {
        for (int j = 0; j < QK4_0/2; j++) {
            qs_ptr[ib * (QK4_0/2) + j] = aos[ib].qs[j];
        }
        d_ptr[ib] = aos[ib].d;
    }
}

// ============================================================================
// Create test data
// ============================================================================
void create_q4_0_data(block_q4_0* data, int nblocks, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.1f, 2.0f);

    for (int i = 0; i < nblocks; i++) {
        data[i].d = sycl::half(dist(rng));
        for (int j = 0; j < QK4_0/2; j++) {
            data[i].qs[j] = (uint8_t)(rng() & 0xFF);
        }
    }
}

void create_q8_1_data(block_q8_1* data, int nblocks, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.1f, 2.0f);

    for (int i = 0; i < nblocks; i++) {
        float sum = 0.0f;
        for (int j = 0; j < QK8_1; j++) {
            data[i].qs[j] = (int8_t)((rng() % 256) - 128);
            sum += data[i].qs[j];
        }
        data[i].ds = sycl::half2(sycl::half(dist(rng)), sycl::half(sum));
    }
}

// ============================================================================
// Test 1: Compare tile loading between AoS and SoA
// ============================================================================
bool test_tile_loading() {
    printf("Test 1: Tile loading comparison (AoS vs SoA)\n");

    constexpr int mmq_y = 64;
    constexpr int nwarps = 4;
    constexpr int nrows = 128;
    constexpr int ncols = 4096;
    constexpr int blocks_per_row = ncols / QK4_0;
    constexpr int nblocks = nrows * blocks_per_row;

    // Create Q4_0 test data
    std::vector<block_q4_0> aos_data(nblocks);
    create_q4_0_data(aos_data.data(), nblocks, 42);

    // Create SoA version
    size_t soa_size = nblocks * sizeof(block_q4_0);
    std::vector<uint8_t> soa_data(soa_size);
    reorder_aos_to_soa(aos_data.data(), soa_data.data(), nrows, ncols);

    const size_t d_offset = (size_t)nrows * ncols / 2;

    // Allocate tiles
    constexpr int tile_ql_size = mmq_y * (WARP_SIZE + 1);
    constexpr int tile_dm_size = mmq_y * (WARP_SIZE/QI4_0) + mmq_y / QI4_0 + WARP_SIZE/QI4_0;

    std::vector<int> aos_x_ql(tile_ql_size, 0);
    std::vector<float> aos_x_dm(tile_dm_size, 0.0f);
    std::vector<int> soa_x_ql(tile_ql_size, 0);
    std::vector<float> soa_x_dm(tile_dm_size, 0.0f);

    int errors = 0;

    // Test multiple tiles
    for (int tile_row = 0; tile_row < 2 && errors < 10; tile_row++) {
        for (int tile_col = 0; tile_col < 2 && errors < 10; tile_col++) {
            // Reset tiles
            std::fill(aos_x_ql.begin(), aos_x_ql.end(), 0);
            std::fill(aos_x_dm.begin(), aos_x_dm.end(), 0.0f);
            std::fill(soa_x_ql.begin(), soa_x_ql.end(), 0);
            std::fill(soa_x_dm.begin(), soa_x_dm.end(), 0.0f);

            int row_offset = tile_row * mmq_y;
            int block_offset = tile_col * (WARP_SIZE / QI4_0);

            // Simulate all threads loading tiles
            for (int warp = 0; warp < nwarps; warp++) {
                for (int lane = 0; lane < WARP_SIZE; lane++) {
                    int i_offset = warp;
                    int k = lane;
                    int i_max = std::min(mmq_y - 1, nrows - row_offset - 1);

                    // AoS load - pointer adjusted by BOTH row_offset AND block_offset
                    // This matches how mul_mat_q calls: x + row_x_0 * blocks_per_row_x + ib0
                    const block_q4_0* tile_data = aos_data.data() + row_offset * blocks_per_row + block_offset;
                    load_tiles_q4_0_aos<mmq_y, nwarps, true>(
                        tile_data, aos_x_ql.data(), aos_x_dm.data(),
                        i_offset, i_max, k, blocks_per_row);

                    // SoA load - receives row_offset and block_offset as parameters
                    load_tiles_q4_0_soa<mmq_y, nwarps, true>(
                        soa_data.data(), d_offset,
                        soa_x_ql.data(), soa_x_dm.data(),
                        i_offset, i_max, k, blocks_per_row,
                        row_offset, block_offset);
                }
            }

            // Compare x_ql tiles
            for (int i = 0; i < tile_ql_size && errors < 10; i++) {
                if (aos_x_ql[i] != soa_x_ql[i]) {
                    printf("  ERROR tile[%d,%d]: x_ql[%d] AoS=0x%08x SoA=0x%08x\n",
                           tile_row, tile_col, i, aos_x_ql[i], soa_x_ql[i]);
                    errors++;
                }
            }

            // Compare x_dm tiles
            for (int i = 0; i < tile_dm_size && errors < 10; i++) {
                if (fabsf(aos_x_dm[i] - soa_x_dm[i]) > 1e-4f) {
                    printf("  ERROR tile[%d,%d]: x_dm[%d] AoS=%.6f SoA=%.6f\n",
                           tile_row, tile_col, i, aos_x_dm[i], soa_x_dm[i]);
                    errors++;
                }
            }
        }
    }

    if (errors > 0) {
        printf("  FAIL: %d tile loading mismatches\n", errors);
        return false;
    }
    printf("  PASS: Tile loading matches between AoS and SoA\n");
    return true;
}

// ============================================================================
// Test 2: Compare vec_dot output between AoS and SoA loaded tiles
// ============================================================================
bool test_vec_dot_comparison() {
    printf("Test 2: vec_dot comparison with loaded tiles\n");

    constexpr int mmq_y = 64;
    constexpr int mmq_x = 64;
    constexpr int nwarps = 4;
    constexpr int nrows = 128;
    constexpr int ncols = 4096;
    constexpr int blocks_per_row = ncols / QK4_0;
    constexpr int nblocks = nrows * blocks_per_row;

    // Create Q4_0 test data (X matrix)
    std::vector<block_q4_0> aos_data(nblocks);
    create_q4_0_data(aos_data.data(), nblocks, 42);

    // Create SoA version
    size_t soa_size = nblocks * sizeof(block_q4_0);
    std::vector<uint8_t> soa_data(soa_size);
    reorder_aos_to_soa(aos_data.data(), soa_data.data(), nrows, ncols);

    const size_t d_offset = (size_t)nrows * ncols / 2;

    // Create Q8_1 test data (Y matrix - activations)
    constexpr int y_blocks = mmq_x * (ncols / QK8_1);
    std::vector<block_q8_1> y_data(y_blocks);
    create_q8_1_data(y_data.data(), y_blocks, 123);

    // Convert Y to tile format
    constexpr int tile_y_qs_size = mmq_x * MMQ_TILE_NE_K;
    constexpr int tile_y_ds_size = mmq_x * (MMQ_TILE_NE_K / QI8_1);
    std::vector<int> y_qs(tile_y_qs_size);
    std::vector<sycl::half2> y_ds(tile_y_ds_size);

    // Simple Y tile loading (just first tile for test)
    for (int j = 0; j < mmq_x; j++) {
        for (int k = 0; k < MMQ_TILE_NE_K; k++) {
            int block_idx = j * (ncols / QK8_1) + k / QI8_1;
            int in_block = k % QI8_1;
            // Pack 4 int8 values into one int
            int val = 0;
            for (int b = 0; b < 4; b++) {
                int8_t qval = y_data[block_idx].qs[in_block * 4 + b];
                val |= ((uint8_t)qval) << (8 * b);
            }
            y_qs[j * MMQ_TILE_NE_K + k] = val;
        }
        for (int k = 0; k < MMQ_TILE_NE_K / QI8_1; k++) {
            int block_idx = j * (ncols / QK8_1) + k;
            y_ds[j * (MMQ_TILE_NE_K / QI8_1) + k] = y_data[block_idx].ds;
        }
    }

    // Allocate X tiles
    constexpr int tile_ql_size = mmq_y * (WARP_SIZE + 1);
    constexpr int tile_dm_size = mmq_y * (WARP_SIZE/QI4_0) + mmq_y / QI4_0 + WARP_SIZE/QI4_0;

    std::vector<int> aos_x_ql(tile_ql_size, 0);
    std::vector<float> aos_x_dm(tile_dm_size, 0.0f);
    std::vector<int> soa_x_ql(tile_ql_size, 0);
    std::vector<float> soa_x_dm(tile_dm_size, 0.0f);

    int row_offset = 0;
    int block_offset = 0;

    // Load tiles (simulate all threads)
    for (int warp = 0; warp < nwarps; warp++) {
        for (int lane = 0; lane < WARP_SIZE; lane++) {
            int i_offset = warp;
            int k = lane;
            int i_max = mmq_y - 1;

            load_tiles_q4_0_aos<mmq_y, nwarps, false>(
                aos_data.data(), aos_x_ql.data(), aos_x_dm.data(),
                i_offset, i_max, k, blocks_per_row);

            load_tiles_q4_0_soa<mmq_y, nwarps, false>(
                soa_data.data(), d_offset,
                soa_x_ql.data(), soa_x_dm.data(),
                i_offset, i_max, k, blocks_per_row,
                row_offset, block_offset);
        }
    }

    // Compare vec_dot results
    int errors = 0;
    float max_diff = 0.0f;

    for (int i = 0; i < std::min(8, mmq_y) && errors < 10; i++) {
        for (int j = 0; j < std::min(8, mmq_x) && errors < 10; j++) {
            for (int k = 0; k < WARP_SIZE && errors < 10; k++) {
                float aos_result = vec_dot_q4_0_q8_1_mul_mat(
                    aos_x_ql.data(), aos_x_dm.data(),
                    y_qs.data(), y_ds.data(), i, j, k);

                float soa_result = vec_dot_q4_0_q8_1_mul_mat(
                    soa_x_ql.data(), soa_x_dm.data(),
                    y_qs.data(), y_ds.data(), i, j, k);

                float diff = fabsf(aos_result - soa_result);
                max_diff = std::max(max_diff, diff);

                if (diff > 1e-3f) {
                    printf("  ERROR vec_dot[i=%d,j=%d,k=%d]: AoS=%.6f SoA=%.6f diff=%.6f\n",
                           i, j, k, aos_result, soa_result, diff);
                    errors++;
                }
            }
        }
    }

    printf("  Max diff: %.6f\n", max_diff);

    if (errors > 0) {
        printf("  FAIL: %d vec_dot mismatches\n", errors);
        return false;
    }
    printf("  PASS: vec_dot outputs match between AoS and SoA tiles\n");
    return true;
}

// ============================================================================
// Test 3: Full MMQ-style matrix multiply comparison
// ============================================================================
bool test_full_mmq_comparison() {
    printf("Test 3: Full MMQ-style matrix multiply\n");

    constexpr int mmq_y = 32;  // Smaller for faster test
    constexpr int mmq_x = 32;
    constexpr int nwarps = 4;
    constexpr int nrows = 64;
    constexpr int ncols = 2048;
    constexpr int blocks_per_row = ncols / QK4_0;
    constexpr int nblocks = nrows * blocks_per_row;

    // Create Q4_0 test data
    std::vector<block_q4_0> aos_data(nblocks);
    create_q4_0_data(aos_data.data(), nblocks, 42);

    // Create SoA version
    size_t soa_size = nblocks * sizeof(block_q4_0);
    std::vector<uint8_t> soa_data(soa_size);
    reorder_aos_to_soa(aos_data.data(), soa_data.data(), nrows, ncols);

    const size_t d_offset = (size_t)nrows * ncols / 2;

    // Create Y data
    constexpr int y_blocks = mmq_x * (ncols / QK8_1);
    std::vector<block_q8_1> y_data(y_blocks);
    create_q8_1_data(y_data.data(), y_blocks, 123);

    // Output matrices
    std::vector<float> aos_output(mmq_y * mmq_x, 0.0f);
    std::vector<float> soa_output(mmq_y * mmq_x, 0.0f);

    // Tile buffers
    constexpr int tile_ql_size = mmq_y * (WARP_SIZE + 1);
    constexpr int tile_dm_size = mmq_y * (WARP_SIZE/QI4_0) + mmq_y / QI4_0 + WARP_SIZE/QI4_0;
    constexpr int tile_y_qs_size = mmq_x * MMQ_TILE_NE_K;
    constexpr int tile_y_ds_size = mmq_x * (MMQ_TILE_NE_K / QI8_1);

    std::vector<int> x_ql(tile_ql_size);
    std::vector<float> x_dm(tile_dm_size);
    std::vector<int> y_qs(tile_y_qs_size);
    std::vector<sycl::half2> y_ds(tile_y_ds_size);

    // Process tiles along K dimension
    for (int kblock = 0; kblock < blocks_per_row; kblock += WARP_SIZE / QI4_0) {
        // Load Y tile (same for both)
        for (int j = 0; j < mmq_x; j++) {
            for (int k = 0; k < MMQ_TILE_NE_K; k++) {
                int block_idx = j * (ncols / QK8_1) + kblock + k / QI8_1;
                if (block_idx < y_blocks) {
                    int in_block = k % QI8_1;
                    int val = 0;
                    for (int b = 0; b < 4; b++) {
                        int8_t qval = y_data[block_idx].qs[in_block * 4 + b];
                        val |= ((uint8_t)qval) << (8 * b);
                    }
                    y_qs[j * MMQ_TILE_NE_K + k] = val;
                }
            }
            for (int k = 0; k < MMQ_TILE_NE_K / QI8_1; k++) {
                int block_idx = j * (ncols / QK8_1) + kblock + k;
                if (block_idx < y_blocks) {
                    y_ds[j * (MMQ_TILE_NE_K / QI8_1) + k] = y_data[block_idx].ds;
                }
            }
        }

        // AoS path
        std::fill(x_ql.begin(), x_ql.end(), 0);
        std::fill(x_dm.begin(), x_dm.end(), 0.0f);

        for (int warp = 0; warp < nwarps; warp++) {
            for (int lane = 0; lane < WARP_SIZE; lane++) {
                load_tiles_q4_0_aos<mmq_y, nwarps, false>(
                    aos_data.data() + kblock, x_ql.data(), x_dm.data(),
                    warp, mmq_y - 1, lane, blocks_per_row);
            }
        }

        for (int i = 0; i < mmq_y; i++) {
            for (int j = 0; j < mmq_x; j++) {
                for (int k = 0; k < WARP_SIZE; k++) {
                    aos_output[i * mmq_x + j] += vec_dot_q4_0_q8_1_mul_mat(
                        x_ql.data(), x_dm.data(), y_qs.data(), y_ds.data(), i, j, k);
                }
            }
        }

        // SoA path
        std::fill(x_ql.begin(), x_ql.end(), 0);
        std::fill(x_dm.begin(), x_dm.end(), 0.0f);

        for (int warp = 0; warp < nwarps; warp++) {
            for (int lane = 0; lane < WARP_SIZE; lane++) {
                load_tiles_q4_0_soa<mmq_y, nwarps, false>(
                    soa_data.data(), d_offset,
                    x_ql.data(), x_dm.data(),
                    warp, mmq_y - 1, lane, blocks_per_row,
                    0, kblock);
            }
        }

        for (int i = 0; i < mmq_y; i++) {
            for (int j = 0; j < mmq_x; j++) {
                for (int k = 0; k < WARP_SIZE; k++) {
                    soa_output[i * mmq_x + j] += vec_dot_q4_0_q8_1_mul_mat(
                        x_ql.data(), x_dm.data(), y_qs.data(), y_ds.data(), i, j, k);
                }
            }
        }
    }

    // Compare outputs
    int errors = 0;
    float max_diff = 0.0f;
    float max_rel_diff = 0.0f;

    for (int i = 0; i < mmq_y * mmq_x; i++) {
        float diff = fabsf(aos_output[i] - soa_output[i]);
        float rel_diff = (fabsf(aos_output[i]) > 1e-6f) ?
                         diff / fabsf(aos_output[i]) : diff;
        max_diff = std::max(max_diff, diff);
        max_rel_diff = std::max(max_rel_diff, rel_diff);

        if (rel_diff > 1e-3f && diff > 1e-3f) {
            if (errors < 5) {
                printf("  ERROR output[%d]: AoS=%.6f SoA=%.6f diff=%.6f rel=%.4f\n",
                       i, aos_output[i], soa_output[i], diff, rel_diff);
            }
            errors++;
        }
    }

    printf("  Max absolute diff: %.6f\n", max_diff);
    printf("  Max relative diff: %.6f\n", max_rel_diff);

    if (errors > 0) {
        printf("  FAIL: %d output mismatches\n", errors);
        return false;
    }
    printf("  PASS: Full MMQ outputs match\n");
    return true;
}

// ============================================================================
// Main
// ============================================================================
int main() {
    printf("=== Q4_0 MMQ Pipeline Tests (AoS vs SoA) ===\n\n");

    int passed = 0;
    int failed = 0;

    if (test_tile_loading()) passed++; else failed++;
    printf("\n");

    if (test_vec_dot_comparison()) passed++; else failed++;
    printf("\n");

    if (test_full_mmq_comparison()) passed++; else failed++;
    printf("\n");

    printf("=================================\n");
    printf("Results: %d passed, %d failed\n", passed, failed);

    return failed > 0 ? 1 : 0;
}
