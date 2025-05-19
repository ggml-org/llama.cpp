#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <thread>
#include <future>

#include <math.h>
#include <float.h>
#include <stdio.h>

// Helper clamp function in case std::clamp is not available
template <typename T>
T clamp(T value, T min_val, T max_val) {
    return std::max(min_val, std::min(max_val, value));
}

static void print_vector(const std::string& name, const std::vector<float>& vec, int n) {
    printf("%s (first %d elements):\n", name.c_str(), n);
    for (int i = 0; i < std::min(n, (int)vec.size()); i++) {
        printf("[%d] = %f\n", i, vec[i]);
    }
    printf("\n");
}

static void random_fill(std::vector<float>& data, float min, float max) {
    size_t nels = data.size();
    static const size_t n_threads = std::thread::hardware_concurrency();
    
    static std::vector<std::default_random_engine> generators = []() {
        std::random_device rd;
        std::vector<std::default_random_engine> vec;
        vec.reserve(n_threads);
        for (size_t i = 0; i < n_threads; i++) { 
            vec.emplace_back(rd()); 
        }
        return vec;
    }();

    auto init_thread = [&](size_t ith, size_t start, size_t end) {
        std::uniform_real_distribution<float> distribution(min, max);
        auto & gen = generators[ith];
        for (size_t i = start; i < end; i++) {
            data[i] = distribution(gen);
        }
    };

    std::vector<std::future<void>> tasks;
    tasks.reserve(n_threads);
    for (size_t i = 0; i < n_threads; i++) {
        size_t start =     i*nels/n_threads;
        size_t end   = (i+1)*nels/n_threads;
        tasks.push_back(std::async(std::launch::async, init_thread, i, start, end));
    }
    
    for (auto & t : tasks) {
        t.get();
    }
}

static void compute_stats(const std::vector<float>& original, const std::vector<float>& reconstructed) {
    if (original.size() != reconstructed.size()) {
        printf("Error: vector sizes don't match for statistics computation\n");
        return;
    }

    float max_diff = 0.0f;
    float sum_squared_diff = 0.0f;
    int max_diff_idx = 0;
    
    for (size_t i = 0; i < original.size(); i++) {
        float diff = std::abs(original[i] - reconstructed[i]);
        if (diff > max_diff) {
            max_diff = diff;
            max_diff_idx = i;
        }
        sum_squared_diff += diff * diff;
    }
    
    float rmse = std::sqrt(sum_squared_diff / original.size());
    
    printf("Quantization Stats:\n");
    printf("  RMSE: %f\n", rmse);
    printf("  Max Diff: %f at index %d (original: %f, reconstructed: %f)\n", 
           max_diff, max_diff_idx, original[max_diff_idx], reconstructed[max_diff_idx]);
    
    // 显示前10个值的对比
    printf("Original vs Reconstructed (showing first 10 values):\n");
    int show_n = std::min(128, (int)original.size());
    for (int i = 0; i < show_n; i++) {
        printf("[%d] %.6f -> %.6f (diff: %.6f)\n", 
               i, original[i], reconstructed[i], std::abs(original[i] - reconstructed[i]));
    }
    printf("\n");
}

static float clampf(float v, float lo, float hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static void pseudo_quantize_qlutattn(
    const float* input,
    int8_t* quantized,
    float* scales,
    float* zeros,
    int n,
    int n_bit,
    int q_group_size
) {
    int num_groups;
    if (q_group_size > 0) {
        if (n % q_group_size != 0) {
            printf("Error: input size must be divisible by q_group_size\n");
            return;
        }
        num_groups = n / q_group_size;
    } else if (q_group_size == -1) {
        num_groups = 1;
        q_group_size = n;
    } else {
        num_groups = 1;
        q_group_size = n;
    }

    const int max_int = (1 << n_bit) - 1;
    const int min_int = 0;
    
    for (int g = 0; g < num_groups; ++g) {
        int start_idx = g * q_group_size;
        int end_idx = start_idx + q_group_size;
        
        float min_val = FLT_MAX;
        float max_val = -FLT_MAX;

        // Find min/max for group
        for (int i = start_idx; i < end_idx; ++i) {
            if (input[i] > max_val) max_val = input[i];
            if (input[i] < min_val) min_val = input[i];
        }

        // Calculate scales and zeros
        scales[g] = (max_val - min_val < 1e-5f ? 1e-5f : (max_val - min_val)) / max_int;
        float zeros_int = clampf(-roundf(min_val / scales[g]), 0.0f, (float)max_int);
        zeros[g] = (zeros_int - (1 << (n_bit - 1))) * scales[g];

        // Quantize values
        for (int i = start_idx; i < end_idx; ++i) {
            int quantized_val = (int)roundf(input[i] / scales[g]) + (int)zeros_int;
            quantized_val = quantized_val < min_int ? min_int : (quantized_val > max_int ? max_int : quantized_val);
            quantized[i] = static_cast<int8_t>(quantized_val);
        }
    }
}

static void pseudo_dequantize_qlutattn(
    const int8_t* quantized,
    float* dequantized,
    const float* scales,
    const float* zeros,
    int n,
    int n_bit,
    int q_group_size
) {
    int num_groups;
    if (q_group_size > 0) {
        if (n % q_group_size != 0) {
            printf("Error: input size must be divisible by q_group_size\n");
            return;
        }
        num_groups = n / q_group_size;
    } else if (q_group_size == -1) {
        num_groups = 1;
        q_group_size = n;
    } else {
        num_groups = 1;
        q_group_size = n;
    }

    const int K = 1 << (n_bit - 1);  // Zero point offset

    for (int g = 0; g < num_groups; ++g) {
        int start_idx = g * q_group_size;
        int end_idx = start_idx + q_group_size;
        
        // Calculate zero point in integer space
        float zero_point = zeros[g]; 

        for (int i = start_idx; i < end_idx; ++i) {
            // Convert quantized value back to float
            float val = quantized[i] * scales[g] - zero_point - (scales[g] * K);
            dequantized[i] = val;
        }
    }
}


static void test_qlutattn_quantization(int n_elements) {
    printf("\n=== Testing QLUTATTN Quantization (n_elements = %d) ===\n", n_elements);

    std::vector<float> original_data(n_elements);
    random_fill(original_data, -1.0f, 1.0f);
    
    const int n_bit = 4;
    const int q_group_size = 128;
    const int num_groups = n_elements / q_group_size;

    std::vector<int8_t> quantized_data(n_elements, 0);
    std::vector<float> scales(num_groups, 0.0f);
    std::vector<float> zeros(num_groups, 0.0f);

    pseudo_quantize_qlutattn(
        original_data.data(),
        quantized_data.data(),
        scales.data(),
        zeros.data(),
        n_elements,
        n_bit,
        q_group_size
    );

    std::vector<float> dequantized_data(n_elements, 0.0f);

    pseudo_dequantize_qlutattn(
        quantized_data.data(),
        dequantized_data.data(),
        scales.data(),
        zeros.data(),
        n_elements,
        n_bit,
        q_group_size
    );
    
    // Print quantized values for inspection
    printf("\nQuantized values:\n");
    for (int i = 0; i < n_elements; ++i) {
        printf("%d ", quantized_data[i]);
        if ((i+1) % 16 == 0) printf("\n");
    }
    printf("\n");

    // Print scale and zero point values
    printf("\nScale and zero point values per group:\n");
    for (int g = 0; g < num_groups; ++g) {
        printf("Group %d: scale = %f, zero = %f\n", g, scales[g], zeros[g]);
    }
    printf("\n");

    // 反量化（直接用量化输出即为反量化结果，因为pseudo_quantize_qlutattn输出的output就是反量化的浮点值）
    // 计算误差
    float max_abs_err = 0.0f;
    float mse = 0.0f;
    for (int i = 0; i < n_elements; ++i) {
        float err = dequantized_data[i] - original_data[i];
        float abs_err = fabsf(err);
        if (abs_err > max_abs_err) max_abs_err = abs_err;

        printf("dequantized_data[%d] = %f, original_data[%d] = %f, err = %f\n", i, dequantized_data[i], i, original_data[i], err);

        mse += err * err;
    }
    mse /= n_elements;

    printf("Max abs error: %f\n", max_abs_err);
    printf("MSE: %e\n", mse);

    // 简单断言
    if (max_abs_err > 0.15f) {
        printf("Test failed: max abs error too large!\n");
        exit(1);
    } else {
        printf("Test passed: quantization error within acceptable range.\n");
    }
}

int main() {
    printf("Running quantization tests...\n");
    
    // Test with different sizes
    test_qlutattn_quantization(256);  // One group
   
    printf("\nAll quantization tests completed successfully.\n");
    
    return 0;
}
