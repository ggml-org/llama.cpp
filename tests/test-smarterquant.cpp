// Unit tests for SmarterQuant functionality

#include "ggml.h"
#include "ggml-cpu.h" // For ggml_get_rows_smarterquant and potentially other CPU specific ops if needed for setup
#include "llama.h"      // For llama_ftype and other general llama types/macros
#include "llama-quant.h" // For SmarterQuantTensorInfo, load_smarter_quant_config, llama_tensor_quantize_smarter_blocks

#undef NDEBUG // Ensure asserts are enabled
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string>
#include <vector>
#include <numeric>   // For std::iota
#include <algorithm> // For std::random_shuffle (if needed, or use C++11 <random>)
#include <iostream>  // For printing detailed error messages
#include <iomanip>   // For std::fixed, std::setprecision
#include <stdexcept> // For std::runtime_error

// Forward declare ggml_get_rows_smarterquant as it's not in a public header
// and we removed static from its definition for testing.
extern "C" void ggml_get_rows_smarterquant(const struct ggml_tensor * tensor, const char * src_row_base, float * dst_row_final_target);


// Helper function to compare float arrays with tolerance
static bool compare_float_arrays(const float* arr1, const float* arr2, size_t size, float tolerance) {
    for (size_t i = 0; i < size; ++i) {
        if (fabs(arr1[i] - arr2[i]) > tolerance) {
            std::cerr << std::fixed << std::setprecision(8)
                      << "Mismatch at index " << i << ": arr1 = " << arr1[i]
                      << ", arr2 = " << arr2[i] << ", diff = " << fabs(arr1[i] - arr2[i])
                      << ", tolerance = " << tolerance << std::endl;
            return false;
        }
    }
    return true;
}

// Helper function to print a float array (for debugging)
static void print_float_array(const float* arr, size_t size, const std::string& name) {
    std::cout << name << " (" << size << " elements): [";
    for (size_t i = 0; i < size; ++i) {
        std::cout << arr[i] << (i == size - 1 ? "" : ", ");
        if (i > 0 && (i + 1) % 8 == 0 && i < size -1) std::cout << std::endl << "  ";
    }
    std::cout << "]" << std::endl;
}


int main(int argc, char **argv) {
    GGML_UNUSED(argc);
    GGML_UNUSED(argv);

    // Initialize ggml (needed for type traits, etc.)
    // ggml_cpu_init(); // May not be needed if llama_init does it, or if using specific functions
    // ggml_quantize_init(); // For quantization tables

    printf("Testing SmarterQuant functionality...\n");

    // Test cases will be added here

    // Test Case 1: Basic Quantization and Dequantization
    {
        printf("  Test Case 1: Basic Quantization and Dequantization with Q4_0, Q5_1, Q8_0, Q2_K and permutation...\n");
        bool test_case_1_success = true;

        const int64_t n_cols = 1024; // Must be multiple of 256 for easy block handling in this test
        const int64_t n_rows = 2;
        const int64_t nelements = n_rows * n_cols;

        // 1. Test Setup
        std::vector<float> original_f32_data(nelements);
        for (int64_t i = 0; i < nelements; ++i) {
            original_f32_data[i] = static_cast<float>(i % 256) - 128.0f; // Simple repeating pattern
            if (i % 3 == 0) original_f32_data[i] *= -0.5f;
        }

        SmarterQuantTensorInfo sq_info;
        sq_info.enabled = true;
        sq_info.compression_types[0] = GGML_TYPE_Q4_0;
        sq_info.compression_types[1] = GGML_TYPE_Q5_1;
        sq_info.compression_types[2] = GGML_TYPE_Q8_0;
        sq_info.compression_types[3] = GGML_TYPE_Q2_K; // For the 4th 256-block segment

        sq_info.n_cols_for_permutation = n_cols;
        sq_info.column_permutation = new int32_t[n_cols];
        for (int64_t i = 0; i < n_cols; ++i) {
            sq_info.column_permutation[i] = (n_cols - 1) - i; // Reverse permutation
        }

        int64_t tensor_ne[GGML_MAX_DIMS] = {n_cols, n_rows, 1, 1};

        // 2. Permutation
        std::vector<float> permuted_f32_data = original_f32_data; // Start with a copy
        if (sq_info.column_permutation && sq_info.n_cols_for_permutation == n_cols) {
            std::vector<float> temp_row(n_cols);
            for (int64_t r = 0; r < n_rows; ++r) {
                const float * original_row_start = original_f32_data.data() + r * n_cols;
                float * permuted_row_start = permuted_f32_data.data() + r * n_cols;
                for (int64_t c_new = 0; c_new < n_cols; ++c_new) {
                    temp_row[c_new] = original_row_start[sq_info.column_permutation[c_new]];
                }
                std::copy(temp_row.begin(), temp_row.end(), permuted_row_start);
            }
            printf("    Applied column permutation.\n");
        }


        // 3. Quantization
        std::vector<uint8_t> quantized_data(nelements * sizeof(float)); // Overestimate, will be resized

        // For K-quants, an imatrix might be expected by some ggml_quantize_chunk paths.
        // For this test, we'll pass nullptr, assuming the specific types chosen (like Q2_K)
        // can handle it for testing purposes or that the path taken doesn't strictly need it.
        // If a real imatrix is needed, this test setup would need to be more complex.
        // For Q2_K, imatrix is generally recommended for good quality.
        std::vector<float> imatrix_dummy(n_cols, 1.0f); // Dummy imatrix, all ones.

        size_t actual_quantized_size = llama_tensor_quantize_smarter_blocks(
            permuted_f32_data.data(),
            quantized_data.data(),
            tensor_ne,
            sq_info,
            imatrix_dummy.data(), // Pass dummy imatrix
            1 // nthread
        );
        quantized_data.resize(actual_quantized_size);
        printf("    Quantized data size: %zu bytes.\n", actual_quantized_size);


        // 4. Dequantization
        std::vector<float> dequantized_f32_data(nelements);

        // Simulate a ggml_tensor for dequantization
        struct ggml_tensor test_tensor;
        test_tensor.ne[0] = n_cols;
        test_tensor.ne[1] = n_rows;
        test_tensor.ne[2] = 1;
        test_tensor.ne[3] = 1;

        // nb calculation for mixed types is complex. ggml_get_rows_smarterquant primarily uses sq_info.
        // We set nb to what a contiguous F32 buffer would have for shape, as that's what dst_row_final_target expects for indexing.
        // The actual layout of quantized_data is handled internally by ggml_get_rows_smarterquant based on sq_info.
        test_tensor.nb[0] = sizeof(uint8_t); // This is not strictly correct for mixed types, but ggml_get_rows_smarterquant uses sq_info
        test_tensor.nb[1] = actual_quantized_size / n_rows; // Approximate stride for the quantized data per row
        test_tensor.nb[2] = actual_quantized_size;
        test_tensor.nb[3] = actual_quantized_size;

        test_tensor.type = static_cast<enum ggml_type>(sq_info.compression_types[3]); // Base type, cast to enum
        test_tensor.op = GGML_OP_NONE;
        test_tensor.sq_info = &sq_info; // CRITICAL: Link the SmarterQuant info
        test_tensor.data = quantized_data.data();
        test_tensor.buffer = NULL; // Not strictly needed for this CPU path test if data is set

        for (int64_t r = 0; r < n_rows; ++r) {
            const char * current_quantized_row_ptr = (const char*)test_tensor.data + r * (actual_quantized_size / n_rows);
            float * current_dequantized_row_ptr = dequantized_f32_data.data() + r * n_cols;
            ggml_get_rows_smarterquant(&test_tensor, current_quantized_row_ptr, current_dequantized_row_ptr);
        }
        printf("    Dequantization complete.\n");

        // 5. Verification
        // Q2_K is particularly lossy, Q4_0, Q5_1 also. Q8_0 is less so.
        // Need a higher tolerance.
        float tolerance = 0.15f;
        if (!compare_float_arrays(original_f32_data.data(), dequantized_f32_data.data(), nelements, tolerance)) {
            test_case_1_success = false;
            printf("    ERROR: Original and dequantized data differ more than tolerance.\n");
            // print_float_array(original_f32_data.data(), 256, "Original (first 256)"); // Print only a subset for brevity
            // print_float_array(dequantized_f32_data.data(), 256, "Dequantized (first 256)");
        }

        delete[] sq_info.column_permutation; // Clean up allocated permutation array

        if (test_case_1_success) {
            printf("    Test Case 1: PASSED\n");
        } else {
            printf("    Test Case 1: FAILED\n");
            return 1;
        }
    }

    printf("All SmarterQuant tests finished.\n");
    return 0; // Indicate success
}
