#include "ggml.h"
#include "ggml-quants.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <cstdlib>

// Test that quantize_row_q4_0_blue_ref produces different output from quantize_row_q4_0_ref
// for the same input (verifying dithering is active)
int main() {
    static const int qk = 32;
    const int n = qk * 4; // 4 blocks
    float x[n];
    // Fill with deterministic data - not perfectly aligned with quantization grid
    for (int i = 0; i < n; i++) {
        x[i] = 0.5f + 0.1f * (float)(i % 7);
    }

    // Allocate output blocks
    block_q4_0 out_normal[n / qk];
    block_q4_0 out_blue[n / qk];

    // Quantize with normal method
    quantize_row_q4_0_ref(x, out_normal, n);
    // Quantize with blue-noise dithering
    quantize_row_q4_0_blue_ref(x, out_blue, n);

    // Compare results
    bool identical = true;
    for (int i = 0; i < n / qk; i++) {
        if (out_normal[i].d != out_blue[i].d) {
            identical = false;
        }
        for (int j = 0; j < qk/2; j++) {
            if (out_normal[i].qs[j] != out_blue[i].qs[j]) {
                identical = false;
            }
        }
    }

    if (identical) {
        printf("FAIL: q4_0_blue output is identical to q4_0 - dithering not active!\n");
        return 1;
    }

    printf("PASS: q4_0_blue differs from q4_0 (dithering is active)\n");

    // Test 2: Verify determinism (same input -> same output)
    block_q4_0 out_blue2[n / qk];
    quantize_row_q4_0_blue_ref(x, out_blue2, n);

    bool deterministic = true;
    for (int i = 0; i < n / qk; i++) {
        if (out_blue[i].d != out_blue2[i].d) {
            deterministic = false;
        }
        for (int j = 0; j < qk/2; j++) {
            if (out_blue[i].qs[j] != out_blue2[i].qs[j]) {
                deterministic = false;
            }
        }
    }

    if (!deterministic) {
        printf("FAIL: q4_0_blue is not deterministic!\n");
        return 1;
    }

    printf("PASS: q4_0_blue is deterministic\n");

    // Test 3: Verify dequantization produces valid output (block format is correct)
    float dequant_normal[n], dequant_blue[n];
    dequantize_row_q4_0(out_normal, dequant_normal, n);
    dequantize_row_q4_0(out_blue, dequant_blue, n);

    // Check that dequantized values are finite
    for (int i = 0; i < n; i++) {
        if (!std::isfinite(dequant_normal[i]) || !std::isfinite(dequant_blue[i])) {
            printf("FAIL: dequantized values are not finite!\n");
            return 1;
        }
    }
    printf("PASS: dequantization produces valid finite values\n");

    // Test 4: Verify type name
    const char * name = ggml_type_name(GGML_TYPE_Q4_0_BLUE);
    if (strcmp(name, "q4_0_blue") != 0) {
        printf("FAIL: type name is '%s', expected 'q4_0_blue'\n", name);
        return 1;
    }
    printf("PASS: type name is '%s'\n", name);

    // Test 5: Verify block sizes match q4_0
    if (ggml_type_size(GGML_TYPE_Q4_0_BLUE) != ggml_type_size(GGML_TYPE_Q4_0)) {
        printf("FAIL: block sizes don't match!\n");
        return 1;
    }
    printf("PASS: block sizes match Q4_0 (%zu bytes)\n", ggml_type_size(GGML_TYPE_Q4_0_BLUE));

    printf("\nAll tests PASSED!\n");
    return 0;
}
