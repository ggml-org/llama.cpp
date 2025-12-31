// GPU Step-by-step Q6_K SoA kernel debug test
// Runs ACTUAL GPU kernel logic with intermediate value capture

#include <CL/sycl.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>

// Q6_K constants
#define QK_K 256
#define QI6_K (QK_K / 8)   // 32
#define QR6_K 2

// MMQ tile constants
#define MMQ_TILE_NE_K 32
#define WARP_SIZE 32

// Debug buffer struct for capturing intermediate values
struct debug_values {
    float d_value;           // d after load
    int ql_int;              // ql before masking
    int ql0;                 // ql0 after masking
    int ql1;                 // ql1 after masking
    int qh_int;              // qh before masking
    int qh0;                 // qh0 after masking
    int qh1;                 // qh1 after masking
    int x_ql0;               // after sub_sat
    int x_ql1;               // after sub_sat
    int sc_int;              // scale value
    float partial_sum;       // partial dot product
    float final_result;      // final output
    int global_block;        // which block was accessed
    int ql_offset;           // ql offset used
    int qh_offset_used;      // qh offset used
    int d_offset_used;       // d offset used
};

// Helper: get int from uint8 array (GPU version)
static inline int get_int_from_uint8_gpu(const uint8_t* x, int i) {
    return x[4*i] | (x[4*i+1] << 8) | (x[4*i+2] << 16) | (x[4*i+3] << 24);
}

// Helper: get int from int8 array (GPU version)
static inline int get_int_from_int8_gpu(const int8_t* x, int i) {
    const uint8_t* xu = (const uint8_t*)x;
    return xu[4*i] | (xu[4*i+1] << 8) | (xu[4*i+2] << 16) | (xu[4*i+3] << 24);
}

// GPU kernel that captures all intermediate values
void debug_soa_kernel(
    const uint8_t* __restrict__ qs_base,
    size_t qh_offset,
    size_t scales_offset,
    size_t d_offset,
    int nblocks,
    int blocks_per_row,
    int row,
    int block_idx,
    debug_values* __restrict__ dbg,
    sycl::nd_item<1> item)
{
    int tid = item.get_local_id(0);
    if (tid != 0) return;  // Only first thread does the work

    // Compute base pointers
    const uint8_t* qh_base = qs_base + qh_offset;
    const int8_t* scales_base = (const int8_t*)(qs_base + scales_offset);
    const sycl::half* d_base = (const sycl::half*)(qs_base + d_offset);

    int global_block = row * blocks_per_row + block_idx;
    dbg->global_block = global_block;

    // Load d value - THIS IS WHERE NaN MIGHT COME FROM
    dbg->d_offset_used = d_offset + global_block * sizeof(sycl::half);
    float d_val = (float)d_base[global_block];
    dbg->d_value = d_val;

    // ql pointer for this block (128 bytes per block)
    const uint8_t* ql_ptr = qs_base + global_block * (QK_K/2);
    dbg->ql_offset = global_block * (QK_K/2);

    // qh pointer for this block (64 bytes per block)
    const uint8_t* qh_ptr = qh_base + global_block * (QK_K/4);
    dbg->qh_offset_used = qh_offset + global_block * (QK_K/4);

    // Compute for k=0
    const int k = 0;
    const int kqsx = k % QI6_K;

    // Load ql
    int ql_int = get_int_from_uint8_gpu(ql_ptr, kqsx);
    dbg->ql_int = ql_int;

    int ql0 = (ql_int >> 0) & 0x0F0F0F0F;
    int ql1 = (ql_int >> 4) & 0x0F0F0F0F;
    dbg->ql0 = ql0;
    dbg->ql1 = ql1;

    // Load qh
    int qh_idx = (QI6_K/4) * (kqsx / (QI6_K/2)) + kqsx % (QI6_K/4);
    int qh_int = get_int_from_uint8_gpu(qh_ptr, qh_idx);
    dbg->qh_int = qh_int;

    int shift = 2 * ((kqsx % (QI6_K/2)) / (QI6_K/4));
    int qh0 = ((qh_int >> shift) << 4) & 0x30303030;
    int qh1 = (qh_int >> shift) & 0x30303030;
    dbg->qh0 = qh0;
    dbg->qh1 = qh1;

    // Combine and subtract 32 (bias)
    // Use manual saturating subtract
    auto sat_sub_char4 = [](int a, int b) -> int {
        int8_t a0 = (a >> 0) & 0xFF;
        int8_t a1 = (a >> 8) & 0xFF;
        int8_t a2 = (a >> 16) & 0xFF;
        int8_t a3 = (a >> 24) & 0xFF;
        int8_t b0 = (b >> 0) & 0xFF;
        int8_t b1 = (b >> 8) & 0xFF;
        int8_t b2 = (b >> 16) & 0xFF;
        int8_t b3 = (b >> 24) & 0xFF;

        auto sat_sub = [](int8_t x, int8_t y) -> int8_t {
            int result = (int)x - (int)y;
            if (result < -128) return -128;
            if (result > 127) return 127;
            return (int8_t)result;
        };

        uint8_t r0 = (uint8_t)sat_sub(a0, b0);
        uint8_t r1 = (uint8_t)sat_sub(a1, b1);
        uint8_t r2 = (uint8_t)sat_sub(a2, b2);
        uint8_t r3 = (uint8_t)sat_sub(a3, b3);

        return r0 | (r1 << 8) | (r2 << 16) | (r3 << 24);
    };

    int x_ql0 = sat_sub_char4(ql0 | qh0, 0x20202020);
    int x_ql1 = sat_sub_char4(ql1 | qh1, 0x20202020);
    dbg->x_ql0 = x_ql0;
    dbg->x_ql1 = x_ql1;

    // Load scale
    const int8_t* sc_ptr = scales_base + global_block * (QK_K/16);
    int sc_int = get_int_from_int8_gpu(sc_ptr, 0);
    dbg->sc_int = sc_int;

    // Compute a simple dot product with y=1
    // Extract q values and compute sum
    int8_t q0 = (x_ql0 >> 0) & 0xFF;
    int8_t q1 = (x_ql0 >> 8) & 0xFF;
    int8_t q2 = (x_ql0 >> 16) & 0xFF;
    int8_t q3 = (x_ql0 >> 24) & 0xFF;

    int sum = q0 + q1 + q2 + q3;
    int8_t scale0 = (sc_int >> 0) & 0xFF;

    float partial = d_val * (float)scale0 * (float)sum;
    dbg->partial_sum = partial;
    dbg->final_result = partial;  // Simplified for debugging
}

// CPU reference implementation for comparison
void cpu_reference(
    const uint8_t* qs_base,
    size_t qh_offset,
    size_t scales_offset,
    size_t d_offset,
    int nblocks,
    int blocks_per_row,
    int row,
    int block_idx,
    debug_values* dbg)
{
    const uint8_t* qh_base = qs_base + qh_offset;
    const int8_t* scales_base = (const int8_t*)(qs_base + scales_offset);
    const uint16_t* d_base = (const uint16_t*)(qs_base + d_offset);

    int global_block = row * blocks_per_row + block_idx;
    dbg->global_block = global_block;

    // Load d value
    dbg->d_offset_used = d_offset + global_block * 2;
    uint16_t d_raw = d_base[global_block];
    float d_val = 0.0f;
    // Manual fp16 to fp32 conversion
    uint16_t h = d_raw;
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    if (exp == 0) {
        if (mant == 0) {
            uint32_t f = sign;
            memcpy(&d_val, &f, 4);
        } else {
            // Denormal
            while ((mant & 0x400) == 0) {
                mant <<= 1;
                exp--;
            }
            exp++;
            mant &= 0x3FF;
            uint32_t f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
            memcpy(&d_val, &f, 4);
        }
    } else if (exp == 31) {
        uint32_t f = sign | 0x7F800000 | (mant << 13);
        memcpy(&d_val, &f, 4);
    } else {
        uint32_t f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        memcpy(&d_val, &f, 4);
    }
    dbg->d_value = d_val;

    // ql pointer
    const uint8_t* ql_ptr = qs_base + global_block * (QK_K/2);
    dbg->ql_offset = global_block * (QK_K/2);

    // qh pointer
    const uint8_t* qh_ptr = qh_base + global_block * (QK_K/4);
    dbg->qh_offset_used = qh_offset + global_block * (QK_K/4);

    // k=0
    const int k = 0;
    const int kqsx = k % QI6_K;

    int ql_int = ql_ptr[0] | (ql_ptr[1] << 8) | (ql_ptr[2] << 16) | (ql_ptr[3] << 24);
    dbg->ql_int = ql_int;

    int ql0 = (ql_int >> 0) & 0x0F0F0F0F;
    int ql1 = (ql_int >> 4) & 0x0F0F0F0F;
    dbg->ql0 = ql0;
    dbg->ql1 = ql1;

    int qh_idx = (QI6_K/4) * (kqsx / (QI6_K/2)) + kqsx % (QI6_K/4);
    int qh_int = qh_ptr[qh_idx*4] | (qh_ptr[qh_idx*4+1] << 8) |
                 (qh_ptr[qh_idx*4+2] << 16) | (qh_ptr[qh_idx*4+3] << 24);
    dbg->qh_int = qh_int;

    int shift = 2 * ((kqsx % (QI6_K/2)) / (QI6_K/4));
    int qh0 = ((qh_int >> shift) << 4) & 0x30303030;
    int qh1 = (qh_int >> shift) & 0x30303030;
    dbg->qh0 = qh0;
    dbg->qh1 = qh1;

    // Saturating subtract
    auto sat_sub_char4 = [](int a, int b) -> int {
        auto sat_sub = [](int8_t x, int8_t y) -> int8_t {
            int result = (int)x - (int)y;
            if (result < -128) return -128;
            if (result > 127) return 127;
            return (int8_t)result;
        };
        int8_t a0 = (a >> 0) & 0xFF;
        int8_t a1 = (a >> 8) & 0xFF;
        int8_t a2 = (a >> 16) & 0xFF;
        int8_t a3 = (a >> 24) & 0xFF;
        int8_t b0 = (b >> 0) & 0xFF;
        int8_t b1 = (b >> 8) & 0xFF;
        int8_t b2 = (b >> 16) & 0xFF;
        int8_t b3 = (b >> 24) & 0xFF;
        uint8_t r0 = (uint8_t)sat_sub(a0, b0);
        uint8_t r1 = (uint8_t)sat_sub(a1, b1);
        uint8_t r2 = (uint8_t)sat_sub(a2, b2);
        uint8_t r3 = (uint8_t)sat_sub(a3, b3);
        return r0 | (r1 << 8) | (r2 << 16) | (r3 << 24);
    };

    int x_ql0 = sat_sub_char4(ql0 | qh0, 0x20202020);
    int x_ql1 = sat_sub_char4(ql1 | qh1, 0x20202020);
    dbg->x_ql0 = x_ql0;
    dbg->x_ql1 = x_ql1;

    const int8_t* sc_ptr = scales_base + global_block * (QK_K/16);
    int sc_int = sc_ptr[0] | ((uint8_t)sc_ptr[1] << 8) |
                 ((uint8_t)sc_ptr[2] << 16) | ((uint8_t)sc_ptr[3] << 24);
    dbg->sc_int = sc_int;

    int8_t q0 = (x_ql0 >> 0) & 0xFF;
    int8_t q1 = (x_ql0 >> 8) & 0xFF;
    int8_t q2 = (x_ql0 >> 16) & 0xFF;
    int8_t q3 = (x_ql0 >> 24) & 0xFF;
    int sum = q0 + q1 + q2 + q3;
    int8_t scale0 = (sc_int >> 0) & 0xFF;

    float partial = d_val * (float)scale0 * (float)sum;
    dbg->partial_sum = partial;
    dbg->final_result = partial;
}

void print_debug_values(const char* label, const debug_values& dbg) {
    printf("\n=== %s ===\n", label);
    printf("global_block: %d\n", dbg.global_block);
    printf("d_offset_used: %d, d_value: %.6f %s\n",
           dbg.d_offset_used, dbg.d_value,
           std::isnan(dbg.d_value) ? "*** NaN! ***" : (std::isinf(dbg.d_value) ? "*** Inf! ***" : ""));
    printf("ql_offset: %d\n", dbg.ql_offset);
    printf("qh_offset_used: %d\n", dbg.qh_offset_used);
    printf("ql_int: 0x%08x\n", dbg.ql_int);
    printf("ql0: 0x%08x, ql1: 0x%08x\n", dbg.ql0, dbg.ql1);
    printf("qh_int: 0x%08x\n", dbg.qh_int);
    printf("qh0: 0x%08x, qh1: 0x%08x\n", dbg.qh0, dbg.qh1);
    printf("x_ql0: 0x%08x, x_ql1: 0x%08x\n", dbg.x_ql0, dbg.x_ql1);
    printf("sc_int: 0x%08x\n", dbg.sc_int);
    printf("partial_sum: %.6f %s\n", dbg.partial_sum,
           std::isnan(dbg.partial_sum) ? "*** NaN! ***" : "");
    printf("final_result: %.6f %s\n", dbg.final_result,
           std::isnan(dbg.final_result) ? "*** NaN! ***" : "");
}

int main() {
    printf("=== GPU Step-by-Step Q6_K SoA Kernel Debug Test ===\n\n");

    // Create test data: 2 rows, 1 block per row = 2 blocks total
    const int nrows = 2;
    const int ncols = QK_K;  // 256 = 1 block worth
    const int nblocks = nrows;  // 1 block per row
    const int blocks_per_row = 1;

    // Q6_K block sizes (per block)
    const size_t ql_size_per_block = QK_K / 2;     // 128 bytes
    const size_t qh_size_per_block = QK_K / 4;     // 64 bytes
    const size_t scales_size_per_block = QK_K / 16; // 16 bytes
    const size_t d_size_per_block = 2;              // fp16

    // Total SoA layout sizes
    const size_t total_ql = nblocks * ql_size_per_block;      // 256
    const size_t total_qh = nblocks * qh_size_per_block;      // 128
    const size_t total_scales = nblocks * scales_size_per_block; // 32
    const size_t total_d = nblocks * d_size_per_block;        // 4

    // SoA offsets
    const size_t qh_offset = total_ql;
    const size_t scales_offset = qh_offset + total_qh;
    const size_t d_offset = scales_offset + total_scales;
    const size_t total_size = d_offset + total_d;

    printf("SoA layout:\n");
    printf("  ql: 0 - %zu (size %zu)\n", total_ql, total_ql);
    printf("  qh: %zu - %zu (size %zu)\n", qh_offset, qh_offset + total_qh, total_qh);
    printf("  scales: %zu - %zu (size %zu)\n", scales_offset, scales_offset + total_scales, total_scales);
    printf("  d: %zu - %zu (size %zu)\n", d_offset, d_offset + total_d, total_d);
    printf("  total: %zu bytes\n\n", total_size);

    // Allocate and fill test data
    std::vector<uint8_t> soa_data(total_size, 0);

    // Fill with known patterns
    std::mt19937 rng(42);

    // Fill ql (random 0-255)
    for (size_t i = 0; i < total_ql; i++) {
        soa_data[i] = rng() & 0xFF;
    }

    // Fill qh (random 0-255)
    for (size_t i = 0; i < total_qh; i++) {
        soa_data[qh_offset + i] = rng() & 0xFF;
    }

    // Fill scales (random -127 to 127)
    for (size_t i = 0; i < total_scales; i++) {
        soa_data[scales_offset + i] = (rng() % 255) - 127;
    }

    // Fill d values with known fp16 values
    // Use a simple value: 0.1 (fp16 = 0x2E66)
    uint16_t d_val_fp16 = 0x2E66;  // approximately 0.1
    for (int i = 0; i < nblocks; i++) {
        memcpy(&soa_data[d_offset + i * 2], &d_val_fp16, 2);
    }

    printf("Test data created with %d blocks\n", nblocks);
    printf("d values set to fp16 0x%04x\n\n", d_val_fp16);

    // Run CPU reference first
    debug_values cpu_dbg = {};
    cpu_reference(soa_data.data(), qh_offset, scales_offset, d_offset,
                  nblocks, blocks_per_row, 0, 0, &cpu_dbg);
    print_debug_values("CPU Reference (row 0, block 0)", cpu_dbg);

    // Now run on GPU
    try {
        // Get GPU device
        sycl::queue q{sycl::gpu_selector_v};
        printf("\nUsing GPU: %s\n",
               q.get_device().get_info<sycl::info::device::name>().c_str());

        // Allocate device memory
        uint8_t* d_soa_data = sycl::malloc_device<uint8_t>(total_size, q);
        debug_values* d_dbg = sycl::malloc_device<debug_values>(1, q);

        // Copy data to device
        q.memcpy(d_soa_data, soa_data.data(), total_size).wait();

        // Zero debug buffer
        debug_values zero_dbg = {};
        q.memcpy(d_dbg, &zero_dbg, sizeof(debug_values)).wait();

        // Run debug kernel
        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::nd_range<1>(32, 32), [=](sycl::nd_item<1> item) {
                debug_soa_kernel(d_soa_data, qh_offset, scales_offset, d_offset,
                                nblocks, blocks_per_row, 0, 0, d_dbg, item);
            });
        }).wait();

        // Copy results back
        debug_values gpu_dbg;
        q.memcpy(&gpu_dbg, d_dbg, sizeof(debug_values)).wait();

        print_debug_values("GPU Kernel (row 0, block 0)", gpu_dbg);

        // Compare CPU vs GPU
        printf("\n=== CPU vs GPU Comparison ===\n");
        bool match = true;

        #define CHECK_FIELD(name) \
            if (cpu_dbg.name != gpu_dbg.name) { \
                printf("MISMATCH: " #name " CPU=0x%x GPU=0x%x\n", \
                       (unsigned)cpu_dbg.name, (unsigned)gpu_dbg.name); \
                match = false; \
            }

        #define CHECK_FLOAT(name) \
            if (std::abs(cpu_dbg.name - gpu_dbg.name) > 1e-6 || \
                std::isnan(cpu_dbg.name) != std::isnan(gpu_dbg.name)) { \
                printf("MISMATCH: " #name " CPU=%.6f GPU=%.6f\n", \
                       cpu_dbg.name, gpu_dbg.name); \
                match = false; \
            }

        CHECK_FIELD(global_block);
        CHECK_FIELD(ql_offset);
        CHECK_FIELD(qh_offset_used);
        CHECK_FIELD(d_offset_used);
        CHECK_FLOAT(d_value);
        CHECK_FIELD(ql_int);
        CHECK_FIELD(ql0);
        CHECK_FIELD(ql1);
        CHECK_FIELD(qh_int);
        CHECK_FIELD(qh0);
        CHECK_FIELD(qh1);
        CHECK_FIELD(x_ql0);
        CHECK_FIELD(x_ql1);
        CHECK_FIELD(sc_int);
        CHECK_FLOAT(partial_sum);
        CHECK_FLOAT(final_result);

        if (match) {
            printf("ALL VALUES MATCH!\n");
        } else {
            printf("\n*** MISMATCHES FOUND - GPU kernel behaves differently! ***\n");
        }

        // Check for NaN in GPU results
        if (std::isnan(gpu_dbg.d_value)) {
            printf("\n*** BUG: GPU d_value is NaN! ***\n");
            printf("    Check if sycl::half conversion is broken\n");
        }
        if (std::isnan(gpu_dbg.partial_sum) && !std::isnan(cpu_dbg.partial_sum)) {
            printf("\n*** BUG: GPU partial_sum is NaN but CPU is not! ***\n");
            printf("    NaN introduced during GPU computation\n");
        }

        // Cleanup
        sycl::free(d_soa_data, q);
        sycl::free(d_dbg, q);

        return match ? 0 : 1;

    } catch (const sycl::exception& e) {
        printf("SYCL exception: %s\n", e.what());
        return 1;
    }
}
