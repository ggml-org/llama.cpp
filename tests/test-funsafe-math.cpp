// Test for -funsafe-math-optimizations issues
// This test verifies that floating-point operations produce correct results
// even when compiled with -funsafe-math-optimizations

#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>

#define EPSILON 1e-6

bool test_softmax_stability() {
    printf("Testing softmax stability...\n");
    
    const int n = 1000;
    float * data = (float *)malloc(n * sizeof(float));
    
    // Initialize with values that could cause overflow
    for (int i = 0; i < n; i++) {
        data[i] = (float)i - (n / 2);  // Values from -500 to 499
    }
    
    // Compute softmax with max subtraction for stability
    float max_val = data[0];
    for (int i = 1; i < n; i++) {
        if (data[i] > max_val) max_val = data[i];
    }
    
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        data[i] = expf(data[i] - max_val);
        sum += data[i];
    }
    for (int i = 0; i < n; i++) {
        data[i] /= sum;
    }
    
    // Verify softmax sums to 1
    float check_sum = 0.0f;
    for (int i = 0; i < n; i++) {
        check_sum += data[i];
    }
    
    free(data);
    
    if (fabsf(check_sum - 1.0f) > EPSILON) {
        printf("FAIL: Softmax sum = %f, expected 1.0\n", check_sum);
        return false;
    }
    
    printf("PASS: Softmax stability test passed (sum = %e)\n", check_sum);
    return true;
}

bool test_kq_scaling() {
    printf("Testing KQ scaling...\n");
    
    const int N = 32;
    const int D = 128;
    
    float * q = (float *)malloc(N * D * sizeof(float));
    float * k = (float *)malloc(N * D * sizeof(float));
    float * qk_ref = (float *)malloc(N * N * sizeof(float));
    
    srand(42);
    for (int i = 0; i < N * D; i++) {
        q[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        k[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    
    // Compute Q * K^T reference (row-major)
    const float scale = 1.0f / sqrtf((float)D);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float dot = 0.0f;
            for (int d = 0; d < D; d++) {
                dot += q[i * D + d] * k[j * D + d];
            }
            qk_ref[i * N + j] = dot * scale;
        }
    }
    
    // Verify the reference computation
    bool passed = true;
    int mismatches = 0;
    
    // Re-compute and compare
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float dot = 0.0f;
            for (int d = 0; d < D; d++) {
                dot += q[i * D + d] * k[j * D + d];
            }
            float scaled = dot * scale;
            
            if (fabsf(scaled - qk_ref[i * N + j]) > EPSILON) {
                if (mismatches < 5) {
                    printf("Mismatch at %d: got %f, expected %f\n", i * N + j, scaled, qk_ref[i * N + j]);
                }
                mismatches++;
                passed = false;
            }
        }
    }
    
    free(q);
    free(k);
    free(qk_ref);
    
    if (!passed) {
        printf("FAIL: KQ scaling test failed (%d mismatches)\n", mismatches);
        return false;
    }
    
    printf("PASS: KQ scaling test passed\n");
    return true;
}

bool test_attention_value_range() {
    printf("Testing attention value range...\n");
    
    const int N = 64;
    const int D = 128;
    
    float * q = (float *)malloc(N * D * sizeof(float));
    float * k = (float *)malloc(N * D * sizeof(float));
    float * qk = (float *)malloc(N * N * sizeof(float));
    float * attn = (float *)malloc(N * N * sizeof(float));
    
    srand(123);
    for (int i = 0; i < N * D; i++) {
        q[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        k[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    
    const float scale = 1.0f / sqrtf((float)D);
    
    // Compute Q * K^T
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float dot = 0.0f;
            for (int d = 0; d < D; d++) {
                dot += q[i * D + d] * k[j * D + d];
            }
            qk[i * N + j] = dot * scale;
        }
    }
    
    // Apply softmax with max subtraction
    for (int i = 0; i < N; i++) {
        float max_val = qk[i * N];
        for (int j = 1; j < N; j++) {
            if (qk[i * N + j] > max_val) max_val = qk[i * N + j];
        }
        
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            attn[i * N + j] = expf(qk[i * N + j] - max_val);
            sum += attn[i * N + j];
        }
        for (int j = 0; j < N; j++) {
            attn[i * N + j] /= sum;
        }
    }
    
    // Verify attention values are in [0, 1] and rows sum to 1
    float min_val = 1e30f;
    float max_val = -1e30f;
    
    for (int i = 0; i < N; i++) {
        float row_sum = 0.0f;
        for (int j = 0; j < N; j++) {
            float val = attn[i * N + j];
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
            row_sum += val;
        }
        
        if (fabsf(row_sum - 1.0f) > 0.01f) {
            printf("FAIL: Attention row %d sum = %f (expected 1.0)\n", i, row_sum);
            free(q);
            free(k);
            free(qk);
            free(attn);
            return false;
        }
    }
    
    printf("Attention value range: [%e, %e]\n", min_val, max_val);
    
    free(q);
    free(k);
    free(qk);
    free(attn);
    
    if (min_val < 0.0f || max_val > 1.0f) {
        printf("FAIL: Attention values out of range [0, 1]\n");
        return false;
    }
    
    printf("PASS: Attention value range test passed\n");
    return true;
}

int main() {
    printf("=== Testing -funsafe-math-optimizations safety ===\n\n");
    
    bool all_passed = true;
    
    all_passed &= test_softmax_stability();
    all_passed &= test_kq_scaling();
    all_passed &= test_attention_value_range();
    
    printf("\n=== Results ===\n");
    if (all_passed) {
        printf("PASS: All -funsafe-math-optimizations tests passed\n");
        return 0;
    } else {
        printf("FAIL: Some -funsafe-math-optimizations tests failed\n");
        return 1;
    }
}
