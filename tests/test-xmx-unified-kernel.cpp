//
// Unit tests for XMX unified Q4_0 kernel template
// Tests that the unified kernel matches outputs of all 7 original kernel variants
//
// TDD: This test is written FIRST before the unified kernel implementation exists.
// Expected to fail initially with "file not found" or compilation errors.
//

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include <random>

// Include the configuration types (already implemented)
#include "ggml-sycl/xmx-kernel-config.hpp"

// Include the xmx-quant-loaders.hpp for QuantTraits (already exists)
#include "ggml-sycl/xmx-quant-loaders.hpp"

// This include should fail initially - file doesn't exist yet
#include "ggml-sycl/xmx-unified-q4-kernel.hpp"

// =============================================================================
// Test Data Generation
// =============================================================================

// Q4_0 block structure (from ggml-common.h)
// struct block_q4_0 {
//     ggml_half d;           // delta (scale)
//     uint8_t qs[QK4_0 / 2]; // nibbles / quants (16 bytes)
// };
constexpr int QK4_0_SIZE = 32;
constexpr int BLOCK_Q4_0_BYTES = 18;  // 2 (half) + 16 (nibbles)

// Q8_1 block structure (from ggml-common.h)
// struct block_q8_1 {
//     ggml_half d;           // delta (scale)
//     ggml_half s;           // d * sum(qs[i])
//     int8_t  qs[QK8_1];     // quants (32 bytes)
// };
constexpr int QK8_1_SIZE = 32;
constexpr int BLOCK_Q8_1_BYTES = 36;  // 2 (d) + 2 (s) + 32 (qs)

/**
 * @brief Generate random Q4_0 quantized weights
 */
static void generate_q4_0_weights(int nrows, int ncols_k, std::vector<uint8_t>& weights, std::mt19937& rng) {
    assert(ncols_k % QK4_0_SIZE == 0);
    int num_blocks_per_row = ncols_k / QK4_0_SIZE;
    int total_blocks = nrows * num_blocks_per_row;
    weights.resize(total_blocks * BLOCK_Q4_0_BYTES);

    std::uniform_real_distribution<float> scale_dist(0.001f, 0.1f);
    std::uniform_int_distribution<int> nibble_dist(0, 15);

    for (int block = 0; block < total_blocks; block++) {
        uint8_t* block_ptr = weights.data() + block * BLOCK_Q4_0_BYTES;

        // Generate scale (fp16)
        float d = scale_dist(rng);
        uint16_t d_bits;
        // Simple fp32 to fp16 conversion (rough approximation for test)
        uint32_t f32_bits;
        memcpy(&f32_bits, &d, 4);
        uint32_t sign = (f32_bits >> 31) & 1;
        int32_t exp = ((f32_bits >> 23) & 0xFF) - 127 + 15;
        uint32_t mant = (f32_bits >> 13) & 0x3FF;
        if (exp <= 0) { exp = 0; mant = 0; }
        if (exp >= 31) { exp = 31; mant = 0; }
        d_bits = (sign << 15) | (exp << 10) | mant;
        memcpy(block_ptr, &d_bits, 2);

        // Generate packed nibbles
        for (int i = 0; i < 16; i++) {
            int lo = nibble_dist(rng);
            int hi = nibble_dist(rng);
            block_ptr[2 + i] = (hi << 4) | lo;
        }
    }
}

/**
 * @brief Generate random Q8_1 quantized activations
 */
static void generate_q8_1_activations(int ncols_y, int ncols_k, std::vector<uint8_t>& acts, std::mt19937& rng) {
    assert(ncols_k % QK8_1_SIZE == 0);
    int num_blocks_per_col = ncols_k / QK8_1_SIZE;
    int total_blocks = ncols_y * num_blocks_per_col;
    acts.resize(total_blocks * BLOCK_Q8_1_BYTES);

    std::uniform_real_distribution<float> scale_dist(0.001f, 0.1f);
    std::uniform_int_distribution<int> qs_dist(-127, 127);

    for (int block = 0; block < total_blocks; block++) {
        uint8_t* block_ptr = acts.data() + block * BLOCK_Q8_1_BYTES;

        // Generate scale d (fp16)
        float d = scale_dist(rng);

        // Generate int8 quantized values and compute sum
        int32_t sum = 0;
        for (int i = 0; i < 32; i++) {
            int8_t qs = static_cast<int8_t>(qs_dist(rng));
            block_ptr[4 + i] = static_cast<uint8_t>(qs);
            sum += qs;
        }

        // s = d * sum(qs)
        float s = d * static_cast<float>(sum);

        // Store d and s as fp16
        auto to_fp16 = [](float f) -> uint16_t {
            uint32_t f32_bits;
            memcpy(&f32_bits, &f, 4);
            uint32_t sign = (f32_bits >> 31) & 1;
            int32_t exp = ((f32_bits >> 23) & 0xFF) - 127 + 15;
            uint32_t mant = (f32_bits >> 13) & 0x3FF;
            if (exp <= 0) { exp = 0; mant = 0; }
            if (exp >= 31) { exp = 31; mant = 0; }
            return (sign << 15) | (exp << 10) | mant;
        };

        uint16_t d_bits = to_fp16(d);
        uint16_t s_bits = to_fp16(s);
        memcpy(block_ptr, &d_bits, 2);
        memcpy(block_ptr + 2, &s_bits, 2);
    }
}

/**
 * @brief Reference CPU implementation of Q4_0 x Q8_1 matrix multiply
 *
 * Computes: C[M,N] = A[M,K] * B[K,N]
 * Where A is Q4_0 weights, B is Q8_1 activations
 */
static void reference_q4_0_q8_1_matmul(
    const uint8_t* weights,  // Q4_0 [nrows_x, ncols_k/32 blocks]
    const uint8_t* acts,     // Q8_1 [ncols_y, ncols_k/32 blocks]
    float* output,           // [nrows_x, ncols_y] column-major
    int nrows_x,
    int ncols_k,
    int ncols_y
) {
    const int num_k_blocks = ncols_k / 32;

    for (int row = 0; row < nrows_x; row++) {
        for (int col = 0; col < ncols_y; col++) {
            float acc = 0.0f;

            for (int k_block = 0; k_block < num_k_blocks; k_block++) {
                // Get weight block
                const uint8_t* w_block = weights + (row * num_k_blocks + k_block) * BLOCK_Q4_0_BYTES;
                uint16_t d_w_bits;
                memcpy(&d_w_bits, w_block, 2);
                float d_w = xmx::fp16_to_float(d_w_bits);

                // Get activation block
                const uint8_t* a_block = acts + (col * num_k_blocks + k_block) * BLOCK_Q8_1_BYTES;
                uint16_t d_a_bits, s_a_bits;
                memcpy(&d_a_bits, a_block, 2);
                memcpy(&s_a_bits, a_block + 2, 2);
                float d_a = xmx::fp16_to_float(d_a_bits);
                float s_a = xmx::fp16_to_float(s_a_bits);

                // Compute dot product
                int32_t dot = 0;
                for (int i = 0; i < 16; i++) {
                    uint8_t packed = w_block[2 + i];
                    int lo = (packed & 0x0F);      // Raw nibble 0-15
                    int hi = (packed >> 4);        // Raw nibble 0-15
                    int8_t qs_lo = static_cast<int8_t>(a_block[4 + i]);
                    int8_t qs_hi = static_cast<int8_t>(a_block[4 + i + 16]);
                    dot += lo * qs_lo + hi * qs_hi;
                }

                // Q4_0 x Q8_1 formula: d_w * (dot * d_a - 8 * s_a)
                // The -8 accounts for Q4_0 zero-point offset
                acc += d_w * (static_cast<float>(dot) * d_a - 8.0f * s_a);
            }

            // Column-major output
            output[col * nrows_x + row] = acc;
        }
    }
}

// =============================================================================
// Test Functions
// =============================================================================

static constexpr float TOLERANCE = 1e-3f;

/**
 * @brief Test that unified kernel exists and has correct template interface
 */
static bool test_unified_kernel_exists() {
    printf("  test_unified_kernel_exists...\n");

    // This will fail to compile if the unified kernel template doesn't exist
    // or doesn't have the expected interface
    using Config = ggml_sycl_xmx::BasicConfig;

    // Check that the unified kernel type traits exist
    static_assert(ggml_sycl_xmx::UnifiedXMXKernel<Config, GGML_TYPE_Q4_0>::TILE_M == 8,
                  "UnifiedXMXKernel should expose Config::TILE_M");
    static_assert(ggml_sycl_xmx::UnifiedXMXKernel<Config, GGML_TYPE_Q4_0>::TILE_N == 16,
                  "UnifiedXMXKernel should expose Config::TILE_N");

    printf("    PASS\n");
    return true;
}

/**
 * @brief Test that SLM requirements are computed correctly
 */
static bool test_slm_requirements() {
    printf("  test_slm_requirements...\n");

    using Basic = ggml_sycl_xmx::UnifiedXMXKernel<ggml_sycl_xmx::BasicConfig, GGML_TYPE_Q4_0>;
    using MultiTile = ggml_sycl_xmx::UnifiedXMXKernel<ggml_sycl_xmx::MultiTileConfig, GGML_TYPE_Q4_0>;

    // Basic: 8x32 A + 32x16 B + scales
    static_assert(Basic::SLM_A_SIZE == 8 * 32, "Basic SLM_A_SIZE");
    static_assert(Basic::SLM_B_SIZE == 32 * 16, "Basic SLM_B_SIZE");

    // MultiTile: 4 sub-groups, each with 8x32 A tile
    static_assert(MultiTile::SLM_A_SIZE == 4 * 8 * 32, "MultiTile SLM_A_SIZE");

    printf("    PASS\n");
    return true;
}

/**
 * @brief Test BasicConfig kernel against reference
 */
static bool test_basic_config_correctness() {
    printf("  test_basic_config_correctness...\n");

    const int M = 64;   // rows of weights
    const int K = 256;  // K dimension (must be multiple of 32)
    const int N = 32;   // batch size / columns

    std::mt19937 rng(42);
    std::vector<uint8_t> weights, acts;
    std::vector<float> output_ref(M * N, 0.0f);
    std::vector<float> output_test(M * N, 0.0f);

    generate_q4_0_weights(M, K, weights, rng);
    generate_q8_1_activations(N, K, acts, rng);

    // Compute reference
    reference_q4_0_q8_1_matmul(weights.data(), acts.data(), output_ref.data(), M, K, N);

    // Compute using unified kernel helper
    // (This is a CPU simulation - actual SYCL execution would be in integration tests)
    ggml_sycl_xmx::compute_reference_q4_0_q8_1<ggml_sycl_xmx::BasicConfig>(
        weights.data(), acts.data(), output_test.data(), M, K, N);

    // Compare
    float max_error = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float error = std::abs(output_ref[i] - output_test[i]);
        max_error = std::max(max_error, error);
        if (error > TOLERANCE) {
            printf("    FAIL: element %d differs: ref=%f, test=%f, error=%f\n",
                   i, output_ref[i], output_test[i], error);
            return false;
        }
    }

    printf("    PASS (max_error=%e)\n", max_error);
    return true;
}

/**
 * @brief Test MultiTileConfig kernel against reference
 */
static bool test_multitile_config_correctness() {
    printf("  test_multitile_config_correctness...\n");

    const int M = 128;  // rows of weights
    const int K = 256;  // K dimension
    const int N = 32;   // batch size

    std::mt19937 rng(123);
    std::vector<uint8_t> weights, acts;
    std::vector<float> output_ref(M * N, 0.0f);
    std::vector<float> output_test(M * N, 0.0f);

    generate_q4_0_weights(M, K, weights, rng);
    generate_q8_1_activations(N, K, acts, rng);

    reference_q4_0_q8_1_matmul(weights.data(), acts.data(), output_ref.data(), M, K, N);

    ggml_sycl_xmx::compute_reference_q4_0_q8_1<ggml_sycl_xmx::MultiTileConfig>(
        weights.data(), acts.data(), output_test.data(), M, K, N);

    float max_error = 0.0f;
    for (int i = 0; i < M * N; i++) {
        float error = std::abs(output_ref[i] - output_test[i]);
        max_error = std::max(max_error, error);
        if (error > TOLERANCE) {
            printf("    FAIL: element %d differs: ref=%f, test=%f\n",
                   i, output_ref[i], output_test[i]);
            return false;
        }
    }

    printf("    PASS (max_error=%e)\n", max_error);
    return true;
}

/**
 * @brief Test all 7 configurations produce consistent results
 */
static bool test_all_configs_equivalent() {
    printf("  test_all_configs_equivalent...\n");

    const int M = 64;
    const int K = 128;
    const int N = 64;

    std::mt19937 rng(456);
    std::vector<uint8_t> weights, acts;
    std::vector<float> output_ref(M * N, 0.0f);

    generate_q4_0_weights(M, K, weights, rng);
    generate_q8_1_activations(N, K, acts, rng);

    // Reference computation
    reference_q4_0_q8_1_matmul(weights.data(), acts.data(), output_ref.data(), M, K, N);

    // Test each configuration
    auto test_config = [&](const char* name, auto compute_fn) -> bool {
        std::vector<float> output_test(M * N, 0.0f);
        compute_fn(weights.data(), acts.data(), output_test.data(), M, K, N);

        float max_error = 0.0f;
        for (int i = 0; i < M * N; i++) {
            float error = std::abs(output_ref[i] - output_test[i]);
            max_error = std::max(max_error, error);
        }

        if (max_error > TOLERANCE) {
            printf("    FAIL: %s max_error=%e\n", name, max_error);
            return false;
        }
        printf("    %s: OK (max_error=%e)\n", name, max_error);
        return true;
    };

    bool all_pass = true;

    all_pass &= test_config("BasicConfig",
        ggml_sycl_xmx::compute_reference_q4_0_q8_1<ggml_sycl_xmx::BasicConfig>);
    all_pass &= test_config("DoubleBufConfig",
        ggml_sycl_xmx::compute_reference_q4_0_q8_1<ggml_sycl_xmx::DoubleBufConfig>);
    all_pass &= test_config("MultiTileConfig",
        ggml_sycl_xmx::compute_reference_q4_0_q8_1<ggml_sycl_xmx::MultiTileConfig>);
    all_pass &= test_config("LargeTileConfig",
        ggml_sycl_xmx::compute_reference_q4_0_q8_1<ggml_sycl_xmx::LargeTileConfig>);
    all_pass &= test_config("MultiTileDbConfig",
        ggml_sycl_xmx::compute_reference_q4_0_q8_1<ggml_sycl_xmx::MultiTileDbConfig>);
    all_pass &= test_config("ColFusedConfig",
        ggml_sycl_xmx::compute_reference_q4_0_q8_1<ggml_sycl_xmx::ColFusedConfig>);
    all_pass &= test_config("ColFused4TileConfig",
        ggml_sycl_xmx::compute_reference_q4_0_q8_1<ggml_sycl_xmx::ColFused4TileConfig>);

    if (all_pass) {
        printf("    PASS\n");
    }
    return all_pass;
}

// =============================================================================
// Main
// =============================================================================

int main() {
    printf("Testing XMX unified Q4_0 kernel template...\n\n");

    int num_failed = 0;

    if (!test_unified_kernel_exists()) num_failed++;
    if (!test_slm_requirements()) num_failed++;
    if (!test_basic_config_correctness()) num_failed++;
    if (!test_multitile_config_correctness()) num_failed++;
    if (!test_all_configs_equivalent()) num_failed++;

    printf("\n");
    if (num_failed == 0) {
        printf("All tests PASSED!\n");
        return 0;
    } else {
        printf("%d test(s) FAILED!\n", num_failed);
        return 1;
    }
}
