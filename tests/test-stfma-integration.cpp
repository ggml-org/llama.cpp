#include "ggml.h"
#include "ggml-quants.h"
#include "ggml-stfma-adapter.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define N_TEST_VALUES 256

// Helper to generate random float
float random_float(float min, float max) {
    return min + ((float)rand() / RAND_MAX) * (max - min);
}

// Helper to compare results
int compare_results(float res_orig, float res_stfma, const char* test_name) {
    float diff = fabsf(res_orig - res_stfma);
    float tolerance = 1e-4f;
    if (diff > tolerance) {
        printf("[%s] FAILED: Original=%.6f, STFMA=%.6f, Diff=%.6f\n", test_name, res_orig, res_stfma, diff);
        return 1;
    }
    printf("[%s] PASSED: Original=%.6f, STFMA=%.6f\n", test_name, res_orig, res_stfma);
    return 0;
}

int main(void) {
    srand(time(NULL));

    // 1. Allocate memory and generate random data
    float* src_x = (float*)malloc(sizeof(float) * N_TEST_VALUES);
    float* src_y = (float*)malloc(sizeof(float) * N_TEST_VALUES);

    for (int i = 0; i < N_TEST_VALUES; ++i) {
        src_x[i] = random_float(-1.0f, 1.0f);
        src_y[i] = random_float(-1.0f, 1.0f);
    }

    block_tq2_0* block_x = (block_tq2_0*)malloc(sizeof(block_tq2_0));
    block_q8_K* block_y = (block_q8_K*)malloc(sizeof(block_q8_K));

    // 2. Quantize the data
    quantize_row_tq2_0_ref(src_x, block_x, N_TEST_VALUES);
    quantize_row_q8_K_ref(src_y, block_y, N_TEST_VALUES);

    // 3. Run both original and STFMA implementations
    float result_original = 0.0f;
    float result_stfma = 0.0f;

    // Call original generic function
    ggml_vec_dot_tq2_0_q8_K_generic(N_TEST_VALUES, &result_original, 0, block_x, 0, block_y, 0, 1);

    // Call STFMA function
    ggml_vec_dot_tq2_0_q8_K_stfma(N_TEST_VALUES, &result_stfma, 0, block_x, 0, block_y, 0, 1);

    // 4. Compare results
    int failures = 0;
    failures += compare_results(result_original, result_stfma, "TQ2_0 <-> Q8_K Dot Product");

    // 5. Test encoding conversion
    uint8_t tq2_byte = 0b10010011; // Represents +1, 0, -1, invalid
    uint8_t stfma_byte = convert_tq2_to_stfma_byte(tq2_byte);
    uint8_t expected_stfma = 0b01001011;
    if (stfma_byte != expected_stfma) {
        printf("[Encoding] FAILED: Input=0x%02x, Got=0x%02x, Expected=0x%02x\n", tq2_byte, stfma_byte, expected_stfma);
        failures++;
    } else {
        printf("[Encoding] PASSED: Input=0x%02x -> Got=0x%02x\n", tq2_byte, stfma_byte);
    }

    free(src_x);
    free(src_y);
    free(block_x);
    free(block_y);

    if (failures > 0) {
        printf("\n%d tests failed.\n", failures);
        return 1;
    }

    printf("\nAll tests passed successfully!\n");
    return 0;
}
