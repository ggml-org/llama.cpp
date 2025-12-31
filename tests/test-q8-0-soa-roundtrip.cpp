// Test Q8_0 SoA quantize/dequantize round-trip
// Verifies that values can be correctly read back after SoA reordering

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <cstdint>

// Constants from ggml
#define QK8_0 32
typedef uint16_t ggml_half;

// AoS block structure
struct block_q8_0 {
    ggml_half d;          // 2 bytes: delta (scale)
    int8_t    qs[QK8_0];  // 32 bytes: quants
};
static_assert(sizeof(block_q8_0) == 34, "block_q8_0 size mismatch");

// Convert half to float (simple implementation)
static float half_to_float(ggml_half h) {
    // Use memcpy to avoid aliasing issues
    uint16_t bits = h;
    uint32_t sign = (bits >> 15) & 0x1;
    uint32_t exp = (bits >> 10) & 0x1F;
    uint32_t mant = bits & 0x3FF;

    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        // Denormal
        float f = mant / 1024.0f;
        f *= powf(2.0f, -14.0f);
        return sign ? -f : f;
    } else if (exp == 31) {
        return sign ? -INFINITY : INFINITY;
    }

    float f = 1.0f + mant / 1024.0f;
    f *= powf(2.0f, (float)exp - 15.0f);
    return sign ? -f : f;
}

// Convert float to half (simple implementation)
static ggml_half float_to_half(float f) {
    if (f == 0.0f) return 0;
    if (std::isinf(f)) return (f > 0) ? 0x7C00 : 0xFC00;
    if (std::isnan(f)) return 0x7E00;

    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));

    uint32_t sign = (bits >> 31) & 0x1;
    int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = bits & 0x7FFFFF;

    if (exp <= 0) {
        return sign << 15;  // Underflow to zero
    } else if (exp >= 31) {
        return (sign << 15) | 0x7C00;  // Overflow to infinity
    }

    return (sign << 15) | (exp << 10) | (mant >> 13);
}

// Q8_0 quantization (from ggml)
static void quantize_row_q8_0(const float* x, block_q8_0* y, int k) {
    const int nb = k / QK8_0;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f;

        for (int j = 0; j < QK8_0; j++) {
            float v = fabsf(x[i*QK8_0 + j]);
            amax = fmaxf(amax, v);
        }

        const float d = amax / 127.0f;
        const float id = (d != 0.0f) ? 127.0f / amax : 0.0f;

        y[i].d = float_to_half(d);

        for (int j = 0; j < QK8_0; j++) {
            y[i].qs[j] = (int8_t)roundf(x[i*QK8_0 + j] * id);
        }
    }
}

// Reorder AoS to SoA (matches reorder_q8_0_cpu)
static void reorder_q8_0_to_soa(void* dst_soa, const void* src_aos, int ncols, int nrows) {
    const size_t blocks_per_row = ncols / QK8_0;
    const size_t nblocks = blocks_per_row * nrows;

    const uint8_t* aos = (const uint8_t*)src_aos;
    uint8_t* soa_qs = (uint8_t*)dst_soa;
    uint8_t* soa_d = soa_qs + nblocks * QK8_0;  // d values start after all qs

    for (size_t ib = 0; ib < nblocks; ib++) {
        const uint8_t* block_aos = aos + ib * sizeof(block_q8_0);

        // Copy qs (32 bytes at offset 2 in AoS block)
        memcpy(soa_qs + ib * QK8_0, block_aos + sizeof(ggml_half), QK8_0);

        // Copy d (2 bytes at offset 0 in AoS block)
        memcpy(soa_d + ib * sizeof(ggml_half), block_aos, sizeof(ggml_half));
    }
}

// Dequantize from SoA layout (matches dequantize_q8_0_reorder)
static void dequantize_q8_0_soa(const void* soa_data, int ncols, int nrows, float* output) {
    const size_t blocks_per_row = ncols / QK8_0;
    const size_t nblocks = blocks_per_row * nrows;

    const int8_t* soa_qs = (const int8_t*)soa_data;
    const ggml_half* soa_d = (const ggml_half*)((const uint8_t*)soa_data + nblocks * QK8_0);

    for (int row = 0; row < nrows; row++) {
        for (int col = 0; col < ncols; col++) {
            int ib = row * blocks_per_row + col / QK8_0;  // global block index
            int iqs = col % QK8_0;  // element within block

            float d = half_to_float(soa_d[ib]);
            int8_t q = soa_qs[ib * QK8_0 + iqs];

            output[row * ncols + col] = d * q;
        }
    }
}

// Dequantize from AoS layout (reference)
static void dequantize_q8_0_aos(const block_q8_0* blocks, int ncols, int nrows, float* output) {
    const size_t blocks_per_row = ncols / QK8_0;

    for (int row = 0; row < nrows; row++) {
        for (int col = 0; col < ncols; col++) {
            int ib = row * blocks_per_row + col / QK8_0;
            int iqs = col % QK8_0;

            float d = half_to_float(blocks[ib].d);
            int8_t q = blocks[ib].qs[iqs];

            output[row * ncols + col] = d * q;
        }
    }
}

int main() {
    printf("=== Q8_0 SoA Round-Trip Test ===\n\n");

    // Test dimensions (must be multiples of QK8_0=32)
    const int ncols = 128;  // 4 blocks per row
    const int nrows = 4;
    const size_t blocks_per_row = ncols / QK8_0;
    const size_t nblocks = blocks_per_row * nrows;

    printf("ncols=%d, nrows=%d, blocks_per_row=%zu, nblocks=%zu\n\n",
           ncols, nrows, blocks_per_row, nblocks);

    // 1. Create input float data
    std::vector<float> input(ncols * nrows);
    for (int i = 0; i < ncols * nrows; i++) {
        input[i] = (float)(i % 100) * 0.1f - 5.0f;  // Values from -5.0 to 4.9
    }

    // 2. Quantize to AoS
    std::vector<block_q8_0> aos_blocks(nblocks);
    for (int row = 0; row < nrows; row++) {
        quantize_row_q8_0(&input[row * ncols], &aos_blocks[row * blocks_per_row], ncols);
    }

    // 3. Dequantize from AoS (reference)
    std::vector<float> deq_aos(ncols * nrows);
    dequantize_q8_0_aos(aos_blocks.data(), ncols, nrows, deq_aos.data());

    // 4. Reorder to SoA
    size_t soa_size = nblocks * QK8_0 + nblocks * sizeof(ggml_half);
    std::vector<uint8_t> soa_data(soa_size);
    reorder_q8_0_to_soa(soa_data.data(), aos_blocks.data(), ncols, nrows);

    // 5. Dequantize from SoA
    std::vector<float> deq_soa(ncols * nrows);
    dequantize_q8_0_soa(soa_data.data(), ncols, nrows, deq_soa.data());

    // 6. Compare AoS vs SoA dequantization
    printf("Comparing AoS vs SoA dequantization:\n");
    int errors = 0;
    float max_diff = 0.0f;
    int first_error_idx = -1;

    for (int i = 0; i < ncols * nrows; i++) {
        float diff = fabsf(deq_aos[i] - deq_soa[i]);
        if (diff > max_diff) max_diff = diff;

        if (diff > 1e-6f) {
            if (first_error_idx < 0) first_error_idx = i;
            errors++;
            if (errors <= 10) {
                int row = i / ncols;
                int col = i % ncols;
                int ib = row * blocks_per_row + col / QK8_0;
                int iqs = col % QK8_0;
                printf("  [%d] row=%d col=%d ib=%d iqs=%d: AoS=%.6f SoA=%.6f diff=%.6f\n",
                       i, row, col, ib, iqs, deq_aos[i], deq_soa[i], diff);
            }
        }
    }

    if (errors > 0) {
        printf("FAIL: %d errors (max diff = %.6f)\n", errors, max_diff);

        // Debug: print first few SoA bytes
        printf("\nFirst SoA qs bytes:\n");
        for (int i = 0; i < 64 && i < (int)(nblocks * QK8_0); i++) {
            printf("%3d ", (int)(int8_t)soa_data[i]);
            if ((i+1) % 32 == 0) printf("\n");
        }

        printf("\nFirst SoA d values:\n");
        const ggml_half* soa_d = (const ggml_half*)(soa_data.data() + nblocks * QK8_0);
        for (int i = 0; i < 8 && i < (int)nblocks; i++) {
            printf("d[%d]=%.6f ", i, half_to_float(soa_d[i]));
        }
        printf("\n");

        printf("\nFirst AoS blocks:\n");
        for (int i = 0; i < 2 && i < (int)nblocks; i++) {
            printf("Block %d: d=%.6f, qs[0..3]=[%d,%d,%d,%d]\n",
                   i, half_to_float(aos_blocks[i].d),
                   aos_blocks[i].qs[0], aos_blocks[i].qs[1],
                   aos_blocks[i].qs[2], aos_blocks[i].qs[3]);
        }

        return 1;
    }

    printf("PASS: All %d values match (max diff = %.9f)\n\n", ncols * nrows, max_diff);

    // 7. Also verify original input vs dequantized (quantization error)
    printf("Checking quantization error (input vs dequantized):\n");
    float max_quant_error = 0.0f;
    for (int i = 0; i < ncols * nrows; i++) {
        float err = fabsf(input[i] - deq_aos[i]);
        if (err > max_quant_error) max_quant_error = err;
    }
    printf("Max quantization error: %.6f (expected due to int8 quantization)\n\n", max_quant_error);

    // 8. Test the indexing used by DMMV kernel
    printf("Testing DMMV-style indexing:\n");

    const int8_t* qs_base = (const int8_t*)soa_data.data();
    const int64_t d_offset = nrows * ncols;  // d values after all qs
    const ggml_half* d_base = (const ggml_half*)(soa_data.data() + d_offset);

    int dmmv_errors = 0;
    for (int abs_row = 0; abs_row < nrows; abs_row++) {
        for (int col = 0; col < ncols; col++) {
            // DMMV kernel indexing
            int ib = (abs_row * ncols + col) / QK8_0;
            int iqs = col % QK8_0;

            // Read using DMMV-style offsets
            float d = half_to_float(d_base[ib]);
            int8_t q = qs_base[ib * QK8_0 + iqs];
            float val = d * q;

            // Compare with reference
            float ref = deq_aos[abs_row * ncols + col];
            float diff = fabsf(val - ref);

            if (diff > 1e-6f) {
                dmmv_errors++;
                if (dmmv_errors <= 5) {
                    printf("  DMMV err: row=%d col=%d ib=%d iqs=%d: ref=%.6f got=%.6f\n",
                           abs_row, col, ib, iqs, ref, val);
                }
            }
        }
    }

    if (dmmv_errors > 0) {
        printf("FAIL: DMMV indexing has %d errors\n", dmmv_errors);
        return 1;
    }
    printf("PASS: DMMV indexing matches reference\n\n");

    printf("=== All tests passed! ===\n");
    return 0;
}
