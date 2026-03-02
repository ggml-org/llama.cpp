/**
 * Simple Example: Using the Sparse Ternary FMA Kernel
 * 
 * This example demonstrates basic usage of the sparse-ternary-fma library.
 * It shows how to encode ternary values, pack them, and perform the FMA operation.
 * 
 * Copyright 2025 HyperFold Technologies UK Ltd
 * Author: Maurice Wilson
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "../include/sparse_ternary_fma.h"
#include <stdio.h>
#include <string.h>

int main(void) {
    printf("Sparse Ternary FMA - Simple Example\n");
    printf("====================================\n\n");
    
    /* Check CPU features */
    printf("CPU Features:\n");
    printf("  AVX-512 Support: %s\n", has_avx512_support() ? "Yes" : "No");
    printf("  Implementation: %s\n\n", get_fma_implementation());
    
    /* Example 1: Basic encoding/decoding */
    printf("Example 1: Encoding and Decoding\n");
    printf("---------------------------------\n");
    
    int8_t values[] = {-1, 0, 1};
    for (int i = 0; i < 3; i++) {
        uint8_t encoded = encode_trit(values[i]);
        int8_t decoded = decode_trit(encoded);
        printf("  Value %2d → Encoded 0x%02X → Decoded %2d\n",
               values[i], encoded, decoded);
    }
    printf("\n");
    
    /* Example 2: Packing and unpacking */
    printf("Example 2: Packing and Unpacking\n");
    printf("---------------------------------\n");
    
    const size_t N = 16;
    int8_t original[16] = {1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1};
    uint8_t packed[4];  /* 16 trits = 4 bytes */
    int8_t unpacked[16];
    
    pack_trit_array(original, packed, N);
    unpack_trit_array(packed, unpacked, N);
    
    printf("  Original:  ");
    for (size_t i = 0; i < N; i++) {
        printf("%2d ", original[i]);
    }
    printf("\n");
    
    printf("  Packed:    ");
    for (size_t i = 0; i < N / 4; i++) {
        printf("0x%02X ", packed[i]);
    }
    printf("\n");
    
    printf("  Unpacked:  ");
    for (size_t i = 0; i < N; i++) {
        printf("%2d ", unpacked[i]);
    }
    printf("\n\n");
    
    /* Example 3: Sparse Ternary FMA */
    printf("Example 3: Sparse Ternary FMA\n");
    printf("------------------------------\n");
    
    /* Dense coefficients A */
    int64_t A[8] = {100, 200, 300, 400, 500, 600, 700, 800};
    
    /* Ternary key B */
    int8_t B[8] = {1, -1, 0, 1, -1, 0, 1, -1};
    
    /* Pack B */
    uint8_t B_packed[2];  /* 8 trits = 2 bytes */
    pack_trit_array(B, B_packed, 8);
    
    /* Accumulator C (initialized to zero) */
    int64_t C[8];
    memset(C, 0, 8 * sizeof(int64_t));
    
    /* Perform FMA: C = A * B + C */
    sparse_ternary_fma(A, B_packed, C, 8);
    
    printf("  A:      ");
    for (int i = 0; i < 8; i++) {
        printf("%4lld ", (long long)A[i]);
    }
    printf("\n");
    
    printf("  B:      ");
    for (int i = 0; i < 8; i++) {
        printf("%4d ", B[i]);
    }
    printf("\n");
    
    printf("  C:      ");
    for (int i = 0; i < 8; i++) {
        printf("%4lld ", (long long)C[i]);
    }
    printf("\n");
    
    printf("\n  Expected: A[i] * B[i] for each i\n");
    printf("  Result:   ");
    for (int i = 0; i < 8; i++) {
        int64_t expected = A[i] * B[i];
        printf("%s ", (C[i] == expected) ? "✓" : "✗");
    }
    printf("\n\n");
    
    /* Example 4: Accumulation */
    printf("Example 4: Accumulation (Multiple FMAs)\n");
    printf("----------------------------------------\n");
    
    /* Reset C */
    memset(C, 0, 8 * sizeof(int64_t));
    
    /* Perform FMA multiple times */
    for (int iter = 0; iter < 3; iter++) {
        sparse_ternary_fma(A, B_packed, C, 8);
        printf("  After iteration %d: C[0] = %lld\n",
               iter + 1, (long long)C[0]);
    }
    
    printf("  Expected: A[0] * B[0] * 3 = %lld\n",
           (long long)(A[0] * B[0] * 3));
    printf("  Result: %s\n\n",
           (C[0] == A[0] * B[0] * 3) ? "✓ Correct" : "✗ Incorrect");
    
    printf("All examples completed successfully!\n");
    
    return 0;
}
