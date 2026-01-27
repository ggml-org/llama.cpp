//
// Unit tests for XMX kernel configuration types
// Tests the xmx-kernel-config.hpp template infrastructure
//
// TDD: This test is written FIRST before the implementation exists.
// Expected to fail initially with "file not found" or compilation errors.
//

#include <cassert>
#include <cstdio>
#include <cstring>

// This include should fail initially - file doesn't exist yet
#include "ggml-sycl/xmx-kernel-config.hpp"

//
// Test: XMXKernelConfig basic type traits
//
static bool test_basic_config_traits() {
    printf("  test_basic_config_traits...\n");

    // Test default BasicConfig (8x16x32 single-tile)
    using Basic = ggml_sycl_xmx::BasicConfig;

    // Verify compile-time constants
    static_assert(Basic::TILE_M == 8, "BasicConfig TILE_M should be 8");
    static_assert(Basic::TILE_N == 16, "BasicConfig TILE_N should be 16");
    static_assert(Basic::TILE_K == 32, "BasicConfig TILE_K should be 32");
    static_assert(Basic::TILES_M == 1, "BasicConfig TILES_M should be 1");
    static_assert(Basic::TILES_N == 1, "BasicConfig TILES_N should be 1");
    static_assert(Basic::OUTPUT_M == 8, "BasicConfig OUTPUT_M should be 8");
    static_assert(Basic::OUTPUT_N == 16, "BasicConfig OUTPUT_N should be 16");
    static_assert(Basic::BUFFER == ggml_sycl_xmx::BufferStrategy::SINGLE,
                  "BasicConfig should use SINGLE buffer");
    static_assert(Basic::REDUCE == ggml_sycl_xmx::ReductionStrategy::WARP_REDUCE,
                  "BasicConfig should use WARP_REDUCE");

    printf("    PASS\n");
    return true;
}

//
// Test: Double-buffer configuration
//
static bool test_doublebuf_config() {
    printf("  test_doublebuf_config...\n");

    using DoubleBuf = ggml_sycl_xmx::DoubleBufConfig;

    static_assert(DoubleBuf::TILE_M == 8, "DoubleBufConfig TILE_M should be 8");
    static_assert(DoubleBuf::TILE_N == 16, "DoubleBufConfig TILE_N should be 16");
    static_assert(DoubleBuf::TILE_K == 32, "DoubleBufConfig TILE_K should be 32");
    static_assert(DoubleBuf::BUFFER == ggml_sycl_xmx::BufferStrategy::DOUBLE_BUFFER,
                  "DoubleBufConfig should use DOUBLE_BUFFER");

    printf("    PASS\n");
    return true;
}

//
// Test: Multi-tile configuration
//
static bool test_multitile_config() {
    printf("  test_multitile_config...\n");

    using MultiTile = ggml_sycl_xmx::MultiTileConfig;

    static_assert(MultiTile::TILE_M == 8, "MultiTileConfig TILE_M should be 8");
    static_assert(MultiTile::TILE_N == 16, "MultiTileConfig TILE_N should be 16");
    static_assert(MultiTile::TILE_K == 32, "MultiTileConfig TILE_K should be 32");
    static_assert(MultiTile::TILES_M == 4, "MultiTileConfig TILES_M should be 4");
    static_assert(MultiTile::TILES_N == 1, "MultiTileConfig TILES_N should be 1");
    static_assert(MultiTile::OUTPUT_M == 32, "MultiTileConfig OUTPUT_M should be 32");
    static_assert(MultiTile::OUTPUT_N == 16, "MultiTileConfig OUTPUT_N should be 16");

    printf("    PASS\n");
    return true;
}

//
// Test: Large-tile configuration
//
static bool test_largetile_config() {
    printf("  test_largetile_config...\n");

    using LargeTile = ggml_sycl_xmx::LargeTileConfig;

    static_assert(LargeTile::TILE_M == 8, "LargeTileConfig TILE_M should be 8");
    static_assert(LargeTile::TILE_N == 16, "LargeTileConfig TILE_N should be 16");
    static_assert(LargeTile::TILES_M == 8, "LargeTileConfig TILES_M should be 8");
    static_assert(LargeTile::TILES_N == 4, "LargeTileConfig TILES_N should be 4");
    static_assert(LargeTile::OUTPUT_M == 64, "LargeTileConfig OUTPUT_M should be 64");
    static_assert(LargeTile::OUTPUT_N == 64, "LargeTileConfig OUTPUT_N should be 64");

    printf("    PASS\n");
    return true;
}

//
// Test: Column-fused configuration
//
static bool test_colfused_config() {
    printf("  test_colfused_config...\n");

    using ColFused = ggml_sycl_xmx::ColFusedConfig;

    static_assert(ColFused::TILE_M == 8, "ColFusedConfig TILE_M should be 8");
    static_assert(ColFused::TILE_N == 16, "ColFusedConfig TILE_N should be 16");
    static_assert(ColFused::REDUCE == ggml_sycl_xmx::ReductionStrategy::COLUMN_FUSED,
                  "ColFusedConfig should use COLUMN_FUSED");

    printf("    PASS\n");
    return true;
}

//
// Test: Custom configuration with explicit parameters
//
static bool test_custom_config() {
    printf("  test_custom_config...\n");

    using Custom = ggml_sycl_xmx::XMXKernelConfig<8, 16, 32, 2, 2,
                                                   ggml_sycl_xmx::BufferStrategy::DOUBLE_BUFFER,
                                                   ggml_sycl_xmx::ReductionStrategy::WARP_REDUCE>;

    static_assert(Custom::TILE_M == 8, "Custom TILE_M should be 8");
    static_assert(Custom::TILE_N == 16, "Custom TILE_N should be 16");
    static_assert(Custom::TILE_K == 32, "Custom TILE_K should be 32");
    static_assert(Custom::TILES_M == 2, "Custom TILES_M should be 2");
    static_assert(Custom::TILES_N == 2, "Custom TILES_N should be 2");
    static_assert(Custom::OUTPUT_M == 16, "Custom OUTPUT_M should be 16");
    static_assert(Custom::OUTPUT_N == 32, "Custom OUTPUT_N should be 32");

    printf("    PASS\n");
    return true;
}

//
// Test: SLM size calculations
//
static bool test_slm_calculations() {
    printf("  test_slm_calculations...\n");

    using Basic = ggml_sycl_xmx::BasicConfig;

    // Basic config SLM should fit in hardware limits
    // A tile: TILE_M * TILE_K = 8 * 32 = 256 bytes
    // B tile: TILE_K * TILE_N = 32 * 16 = 512 bytes
    // Scales/sums: small overhead

    constexpr int expected_a_size = Basic::TILE_M * Basic::TILE_K;  // 256
    constexpr int expected_b_size = Basic::TILE_K * Basic::TILE_N;  // 512

    // These would be provided by the config if we add SLM calculation helpers
    static_assert(expected_a_size == 256, "A tile size calculation");
    static_assert(expected_b_size == 512, "B tile size calculation");

    printf("    PASS\n");
    return true;
}

//
// Test: Workgroup size calculations
//
static bool test_workgroup_size() {
    printf("  test_workgroup_size...\n");

    // Multi-tile config with 4 sub-groups
    using MultiTile = ggml_sycl_xmx::MultiTileConfig;

    // Each sub-group is 16 threads (XMX_SG_SIZE)
    // 4 tiles vertically = 4 sub-groups = 64 threads
    constexpr int sg_size = 16;
    constexpr int expected_wg_threads = MultiTile::TILES_M * sg_size;

    static_assert(expected_wg_threads == 64, "MultiTile should use 64 threads");

    printf("    PASS\n");
    return true;
}

int main() {
    printf("Testing XMX kernel configuration types...\n\n");

    int num_failed = 0;

    if (!test_basic_config_traits()) num_failed++;
    if (!test_doublebuf_config()) num_failed++;
    if (!test_multitile_config()) num_failed++;
    if (!test_largetile_config()) num_failed++;
    if (!test_colfused_config()) num_failed++;
    if (!test_custom_config()) num_failed++;
    if (!test_slm_calculations()) num_failed++;
    if (!test_workgroup_size()) num_failed++;

    printf("\n");
    if (num_failed == 0) {
        printf("All tests PASSED!\n");
        return 0;
    } else {
        printf("%d test(s) FAILED!\n", num_failed);
        return 1;
    }
}
