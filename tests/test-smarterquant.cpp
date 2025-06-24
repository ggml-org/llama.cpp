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
static bool compare_float_arrays(const float* arr1, const float* arr2, size_t size, float tolerance, const std::string& test_name) {
    for (size_t i = 0; i < size; ++i) {
        if (fabs(arr1[i] - arr2[i]) > tolerance) {
            std::cerr << std::fixed << std::setprecision(8)
                      << "Test: " << test_name << " - Mismatch at index " << i << ": arr1 = " << arr1[i]
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

// Test function
static bool run_smarterquant_test(
    const std::string& test_name,
    int64_t n_cols,
    int64_t n_rows,
    const std::vector<int32_t>& permutation_indices, // Empty for identity
    ggml_type type0, ggml_type type1, ggml_type type2, ggml_type type3,
    float tolerance
) {
    printf("  Test Case: %s (%" PRId64 " cols, %" PRId64 " rows)\n", test_name.c_str(), n_cols, n_rows);
    bool success = true;
    const int64_t nelements = n_rows * n_cols;

    // 1. Test Setup
    std::vector<float> original_f32_data(nelements);
    for (int64_t i = 0; i < nelements; ++i) {
        original_f32_data[i] = static_cast<float>((i % 253) * (i % 3 == 0 ? -1 : 1)) - 127.0f; // Varied pattern
        if (i % 5 == 0) original_f32_data[i] *= 0.5f;
        if (i % 7 == 0) original_f32_data[i] = fmodf(original_f32_data[i], 64.f) - 32.f; // Add some smaller values
    }

    SmarterQuantTensorInfo sq_info;
    sq_info.enabled = true;
    sq_info.compression_types[0] = type0;
    sq_info.compression_types[1] = type1;
    sq_info.compression_types[2] = type2;
    sq_info.compression_types[3] = type3;

    sq_info.n_cols_for_permutation = n_cols;
    if (!permutation_indices.empty()) {
        if (permutation_indices.size() != (size_t)n_cols) {
            fprintf(stderr, "    ERROR: Test %s - Permutation size %zu does not match n_cols %" PRId64 "\n", test_name.c_str(), permutation_indices.size(), n_cols);
            return false;
        }
        sq_info.column_permutation = new int32_t[n_cols];
        std::copy(permutation_indices.begin(), permutation_indices.end(), sq_info.column_permutation);
        printf("    Using custom column permutation (size %zu).\n", permutation_indices.size());
    } else {
        sq_info.column_permutation = new int32_t[n_cols];
        for (int64_t i = 0; i < n_cols; ++i) {
            sq_info.column_permutation[i] = i; // Identity permutation
        }
        printf("    Using identity column permutation.\n");
    }

    int64_t tensor_ne[GGML_MAX_DIMS] = {n_cols, n_rows, 1, 1};
    for(int i=2; i<GGML_MAX_DIMS; ++i) if(tensor_ne[i] == 0) tensor_ne[i] = 1;


    // 2. Permutation of input data (before quantization)
    std::vector<float> permuted_f32_data = original_f32_data;
    if (sq_info.column_permutation && sq_info.n_cols_for_permutation == n_cols) {
        std::vector<float> temp_row(n_cols);
        for (int64_t r = 0; r < n_rows; ++r) {
            const float * original_row_start = original_f32_data.data() + r * n_cols;
            float * permuted_row_start = permuted_f32_data.data() + r * n_cols; // Apply to a copy
            for (int64_t c_new = 0; c_new < n_cols; ++c_new) {
                temp_row[c_new] = original_row_start[sq_info.column_permutation[c_new]];
            }
            std::copy(temp_row.begin(), temp_row.end(), permuted_row_start);
        }
        printf("    Applied column permutation to input data for quantization.\n");
    }


    // 3. Quantization
    std::vector<uint8_t> quantized_data(nelements * sizeof(float) + 1024); // Overestimate, will be resized

    std::vector<float> imatrix_dummy(n_cols, 1.0f); // Dummy imatrix

    size_t actual_quantized_size = llama_tensor_quantize_smarter_blocks(
        permuted_f32_data.data(), // Use permuted data for quantization
        quantized_data.data(),
        tensor_ne,
        sq_info,
        imatrix_dummy.data(),
        1 // nthread
    );
    if (actual_quantized_size == 0 && nelements > 0) {
        fprintf(stderr, "    ERROR: Test %s - llama_tensor_quantize_smarter_blocks returned 0 size for non-empty tensor.\n", test_name.c_str());
        delete[] sq_info.column_permutation;
        return false;
    }
    quantized_data.resize(actual_quantized_size);
    printf("    Quantized data size: %zu bytes.\n", actual_quantized_size);

    // 4. Dequantization
    std::vector<float> dequantized_f32_data(nelements);

    struct ggml_tensor test_tensor;
    test_tensor.ne[0] = n_cols;
    test_tensor.ne[1] = n_rows;
    test_tensor.ne[2] = 1;
    test_tensor.ne[3] = 1;
    for(int i=2; i<GGML_MAX_DIMS; ++i) if(test_tensor.ne[i] == 0) test_tensor.ne[i] = 1;


    test_tensor.nb[0] = sizeof(uint8_t);
    test_tensor.nb[1] = (n_rows > 0 && actual_quantized_size > 0) ? (actual_quantized_size / n_rows) : actual_quantized_size;
    test_tensor.nb[2] = actual_quantized_size;
    test_tensor.nb[3] = actual_quantized_size;
    if (n_rows == 0) { // Handle case for 0-row tensor if it makes sense for the test
        test_tensor.nb[1] = 0;
    }


    test_tensor.type = static_cast<enum ggml_type>(sq_info.compression_types[0]); // Base type for GGUF, but sq_info drives dequant
    test_tensor.op = GGML_OP_NONE;
    test_tensor.sq_info = &sq_info;
    test_tensor.data = quantized_data.data();
    test_tensor.buffer = NULL;

    for (int64_t r = 0; r < n_rows; ++r) {
        const char * current_quantized_row_ptr = nullptr;
        if (actual_quantized_size > 0 && n_rows > 0) {
             current_quantized_row_ptr = (const char*)test_tensor.data + r * (actual_quantized_size / n_rows);
        } else if (actual_quantized_size == 0 && n_rows > 0) {
            // This case should ideally not happen if nelements > 0 leads to actual_quantized_size > 0
            // but as a safeguard if a row is empty or quantization results in zero size for a row.
            current_quantized_row_ptr = (const char*)test_tensor.data;
        } else {
             // No data to dequantize or no rows
        }

        float * current_dequantized_row_ptr = dequantized_f32_data.data() + r * n_cols;
        if (current_quantized_row_ptr || (n_cols > 0 && n_rows > 0 && actual_quantized_size == 0) ) { // Ensure there's something to dequantize or a row structure
             ggml_get_rows_smarterquant(&test_tensor, current_quantized_row_ptr, current_dequantized_row_ptr);
        }
    }
    if (n_rows > 0) printf("    Dequantization complete.\n");


    // 5. Verification
    if (!compare_float_arrays(original_f32_data.data(), dequantized_f32_data.data(), nelements, tolerance, test_name)) {
        success = false;
        printf("    ERROR: Test %s - Original and dequantized data differ more than tolerance.\n", test_name.c_str());
        // print_float_array(original_f32_data.data(), std::min((size_t)256, (size_t)nelements), "Original (first up to 256)");
        // print_float_array(dequantized_f32_data.data(), std::min((size_t)256, (size_t)nelements), "Dequantized (first up to 256)");
    }

    delete[] sq_info.column_permutation;

    if (success) {
        printf("    Test %s: PASSED\n", test_name.c_str());
    } else {
        printf("    Test %s: FAILED\n", test_name.c_str());
    }
    return success;
}


int main(int argc, char **argv) {
    GGML_UNUSED(argc);
    GGML_UNUSED(argv);
    printf("Testing SmarterQuant functionality...\n");
    int overall_status = 0;

    // Test Case 1: Original test (1024 cols, reverse permutation)
    std::vector<int32_t> reverse_perm_1024(1024);
    for (int64_t i = 0; i < 1024; ++i) reverse_perm_1024[i] = (1024 - 1) - i;
    if (!run_smarterquant_test("Basic Quant/Dequant (1024 cols, reverse perm)", 1024, 2, reverse_perm_1024, GGML_TYPE_Q4_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0, GGML_TYPE_Q2_K, 0.15f)) {
        overall_status = 1;
    }

    // Edge Case: 128 columns (less than one 256-block segment)
    std::vector<int32_t> identity_perm_128(128); std::iota(identity_perm_128.begin(), identity_perm_128.end(), 0);
    if (!run_smarterquant_test("Edge Case (128 cols, identity perm)", 128, 2, identity_perm_128, GGML_TYPE_Q4_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0, GGML_TYPE_Q2_K, 0.15f)) {
        overall_status = 1;
    }

    // Edge Case: 300 columns (spans two 256-block segments, second is partial)
    std::vector<int32_t> identity_perm_300(300); std::iota(identity_perm_300.begin(), identity_perm_300.end(), 0);
    if (!run_smarterquant_test("Edge Case (300 cols, identity perm)", 300, 2, identity_perm_300, GGML_TYPE_Q4_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0, GGML_TYPE_Q2_K, 0.15f)) {
        overall_status = 1;
    }

    // Edge Case: 512 columns (exactly two 256-block segments)
    std::vector<int32_t> identity_perm_512(512); std::iota(identity_perm_512.begin(), identity_perm_512.end(), 0);
    if (!run_smarterquant_test("Edge Case (512 cols, identity perm)", 512, 2, identity_perm_512, GGML_TYPE_Q4_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0, GGML_TYPE_Q2_K, 0.15f)) {
        overall_status = 1;
    }

    // Edge Case: 768 columns (exactly three 256-block segments)
    std::vector<int32_t> identity_perm_768(768); std::iota(identity_perm_768.begin(), identity_perm_768.end(), 0);
    if (!run_smarterquant_test("Edge Case (768 cols, identity perm)", 768, 2, identity_perm_768, GGML_TYPE_Q4_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0, GGML_TYPE_Q2_K, 0.15f)) {
        overall_status = 1;
    }

    // Test with a few swaps permutation (on 512 columns)
    std::vector<int32_t> swap_perm_512(512); std::iota(swap_perm_512.begin(), swap_perm_512.end(), 0);
    if (swap_perm_512.size() > 20) { // Ensure size is large enough for these swaps
        std::swap(swap_perm_512[0], swap_perm_512[1]);
        std::swap(swap_perm_512[10], swap_perm_512[20]);
    }
    if (swap_perm_512.size() > 256) { // Ensure size is large enough
         std::swap(swap_perm_512[255], swap_perm_512[256]); // Swap across a 256-block boundary
    }
    if (!run_smarterquant_test("Permutation (512 cols, few swaps)", 512, 2, swap_perm_512, GGML_TYPE_Q4_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0, GGML_TYPE_Q2_K, 0.15f)) {
        overall_status = 1;
    }

    // Test with no permutation (empty vector passed)
    if (!run_smarterquant_test("No Permutation (512 cols)", 512, 2, {}, GGML_TYPE_Q4_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0, GGML_TYPE_Q2_K, 0.15f)) {
        overall_status = 1;
    }


    printf("All SmarterQuant tests finished.\n");
    return overall_status;
}
