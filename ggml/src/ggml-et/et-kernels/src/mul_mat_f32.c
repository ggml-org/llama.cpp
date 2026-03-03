#include <stdint.h>
#include "ggml_tensor.h"
#include "platform.h"
// #include "tensor.h"

#include "math_fp.h"
#include "quants.h"
#include "block_ops.h"

#include <stdio.h>


// //
// // MCACHE_CONTROL
// //
// inline void __attribute__((always_inline))
// mcache_control(uint64_t d1_split, uint64_t scp_en, uint64_t cacheop_rate, uint64_t cacheop_max)
// {
//     uint64_t csr_enc = ((cacheop_max & 0x1F) << 6) | ((cacheop_rate & 0x7) << 2) |
//                        ((scp_en & 0x1) << 1) | ((d1_split & 0x1) << 0);

//     __asm__ __volatile__("csrw 0x7e0, %[csr_enc]\n" : : [csr_enc] "r"(csr_enc) : "x31");
// }

/*
 * L1 SCP
 */
// Dcache configuration
#define L1D_NUM_SETS      16
#define L1D_NUM_WAYS      4
#define L1D_LINE_SIZE     64

#define NOP  __asm__ __volatile__ ("nop\n");
#define FENCE __asm__ __volatile__ ("fence\n");
#define WFI __asm__ __volatile__ ("wfi\n");
#define WAIT_TENSOR_LOAD_0     __asm__ __volatile__ ( "csrwi 0x830, 0\n" : : );
#define WAIT_TENSOR_LOAD_1     __asm__ __volatile__ ( "csrwi 0x830, 1\n" : : );
#define WAIT_TENSOR_LOAD_L2_0  __asm__ __volatile__ ( "csrwi 0x830, 2\n" : : );
#define WAIT_TENSOR_LOAD_L2_1  __asm__ __volatile__ ( "csrwi 0x830, 3\n" : : );
#define WAIT_PREFETCH_0        __asm__ __volatile__ ( "csrwi 0x830, 4\n" : : );
#define WAIT_PREFETCH_1        __asm__ __volatile__ ( "csrwi 0x830, 5\n" : : );
#define WAIT_CACHEOPS          __asm__ __volatile__ ( "csrwi 0x830, 6\n" : : );
#define WAIT_TENSOR_FMA        __asm__ __volatile__ ( "csrwi 0x830, 7\n" : : );
#define WAIT_TENSOR_STORE      __asm__ __volatile__ ( "csrwi 0x830, 8\n" : : );
#define WAIT_TENSOR_REDUCE     __asm__ __volatile__ ( "csrwi 0x830, 9\n" : : );
#define WAIT_TENSOR_QUANT      __asm__ __volatile__ ( "csrwi 0x830, 10\n" : : );
#define STALL                  __asm__ __volatile__ ( "csrw stall, x0\n" : : );
#define CLEAR_TENSOR_ERROR     __asm__ __volatile__ ( "csrwi 0x808, 0" : : );

#define EXCL_MODE(val) __asm__ __volatile__("csrw 0x7d3, %[csr_enc]\n" : : [csr_enc] "r"(val) : "x31");
#define MCACHE_CONTROL(x1, x2, x3, x4) __asm__ __volatile__("csrw 0x7e0, %0\n" : : "r"(((x1 & 0x1F) << 6) | ((x2 & 0x7) << 2) | ((x3 & 0x1) << 1) | ((x4 & 0x1) << 0)) : "x31");

static inline void evict_dcache(void)
{
    for (uint64_t set = 0; set < L1D_NUM_SETS; set++)
    {
        uint64_t evict_val =
            (1ull << 58) |
            ((set & 0xF) << 14) |
            15ull;

        __asm__ __volatile__(
            "fence\n"

            "csrw evict_sw, %[val]\n"
            "addi %[val], %[val], 64\n"
            "csrw evict_sw, %[val]\n"
            "addi %[val], %[val], 64\n"
            "csrw evict_sw, %[val]\n"
            "addi %[val], %[val], 64\n"
            "csrw evict_sw, %[val]\n"
            "addi %[val], %[val], 64\n"

            "csrwi tensor_wait, 6\n"
            :
            : [val] "r"(evict_val)
            : "memory"
        );
    }
}

void setup_cache_scp(){
    // PRM-8: Cache Control Extension
    EXCL_MODE(1);
    // Evict the whole L1$

    // evict_dcache();

    // Shared Mode
    MCACHE_CONTROL(0, 0, 0, 0);
    WAIT_CACHEOPS;
    // D1Split Mode
    MCACHE_CONTROL(0, 0, 0, 1);
    WAIT_CACHEOPS;
    // Scratchpad Mode
    MCACHE_CONTROL(0, 0, 1, 1);
    WAIT_CACHEOPS;
    EXCL_MODE(0);
}

#include "tensor.h"
int entry_point(struct ggml_et_binary_params* params, void* env) {
    uint64_t hart_id = get_hart_id();
    uint64_t global_id = ((hart_id >> 6) << 5) + ((hart_id >> 1) & 0x1F);
    if (hart_id & 1) {
        return 0;  // Odd thread - skip work
    }
    const int64_t K = params->src0.ne[0];
    const int64_t M = params->src0.ne[1];
    const int64_t N = params->src1.ne[1];

    float* src0_data = ( float*)params->src0.data;
    const float* src1_data = (const float*)params->src1.data;
    float* dst_data       = (float*)params->dst.data;
    // printf("ET-MatMul: M=%ld, N=%ld, K=%ld\n", M, N, K);


    // Initialize SCP
    setup_cache_scp();
// CLEAR_TENSOR_ERROR
    const int64_t block_size = 16;
    // The hardware shifts the stride LEFT by 6.
    // Should we must shift our byte-stride RIGHT by 6 to compensate???
    // const uint64_t stride_b = (K * sizeof(float)) >> 6;
    const uint64_t stride_b = (K * sizeof(float));
    // const uint64_t stride_dst = (M * sizeof(float)) >> 6;

    for (int64_t m = global_id; m < M; m += 1024) {
        for (int64_t n = 0; n < N; n += block_size) {
            for (int64_t kb = 0; kb < K; kb += block_size) {

                // 1. Load Matrix A (1 row x 16 cols) -> SCP Line 0
                tensor_load(false, false, 0, 0, 0,
                            (uint64_t)(&src0_data[m * K + kb]), 0,
                            0, // 0+1 = 1 line
                            0, // Stride ignored for 1 line
                            0);

                // 2. Load Matrix B (16 rows x 16 cols) -> SCP Lines 1-16
                // Note: We pass stride_b which is already >> 6
                tensor_load(false, false, 1, 0, 0,
                            (uint64_t)(&src1_data[n * K + kb]), 0,
                            15, // 15+1 = 16 lines
                            stride_b << 6, // tensor_load() masks low bits, so we provide the aligned byte address
                            1);

                tensor_wait(TENSOR_LOAD_WAIT_0);
                tensor_wait(TENSOR_LOAD_WAIT_1);

                // 3. FMA: C = A * B
                tensor_fma(
                    false,      // use_tmask
                    3,          // b_num_col: (4*3 + 4) = 16 columns
                    0,          // a_num_rows: (0 + 1) = 1 row
                    15,         // a_num_cols: (15 + 1) = 16 elements
                    0,          // offset: AOFFSET
                    0,          // tenc_loc: (Manual says C is always registers for this opcode)
                    false,      // tenb_unsigned
                    false,      // tena_unsigned
                    0,          // tenb_loc: 0 = L1 Scratchpad
                    1,          // scp_loc_b: Starting SCP line 1
                    0,          // scp_loc_a: Starting SCP line 0
                    0,          // opcode: Manual says TensorType = 000 for this FP32 FMA
                    (kb == 0)   // first_pass: If 1, MUL=1 (overwrite C). If 0, MUL=0 (add to C).
                );
                tensor_wait(TENSOR_FMA_WAIT);
            }

            // 4. Store from REGISTERS to Memory
            // According to manual: C row 0 is in f0 and f1 (because BCOLS=3 is >= 2)
            // tensor_store(reg_stride, start_reg, cols, Arows, addr, coop_store, stride)
            tensor_store(
                0,          // reg_stride
                0,          // start_reg: f0
                3,          // cols: BCOLS=3 (16 columns)
                0,          // Arows: 1 row
                (uint64_t)(&dst_data[n * M + m]),
                0,          // coop_store
                (M * sizeof(float)) // byte stride in memory
            );

            tensor_wait(TENSOR_STORE_WAIT);
        }
    }

    if(global_id == 0) {
        unsigned long error = get_tensor_error();
        ((uint64_t*)src0_data)[0] = error;
    }
    return 0;
}








// KERNEL_TRAMPOLINE();


// int entry_point(struct ggml_et_binary_params* params, void* env) {
//     uint64_t hart_id = get_hart_id();
//     uint64_t global_id = ((hart_id >> 6) << 5) + ((hart_id >> 1) & 0x1F);

//     // Matrix dimensions
//     const int64_t K = params->src0.ne[0];
//     const int64_t M = params->src0.ne[1];
//     const int64_t N = params->src1.ne[1];

//     // Block size is 8: K_blocks = K / 8
//     const int64_t K_blocks = K >> 4;

//     // Data pointers
//     const float* src0_data = (const float*)params->src0.data;
//     const float* src1_data = (const float*)params->src1.data;
//     float* dst_data       = (float*)params->dst.data;

//     // Parallelize over M (rows)
//     for (int64_t m = global_id; m < M; m += 1024) {
//         for (int64_t n = 0; n < N; n++) {
//             float sum = 0.0f;

//             // Direct row/column pointers
//             const float* q_row = src0_data + (m * K);
//             const float* b_col = src1_data + (n * K);

//             // Process blocks of 16
//             for (int64_t kb = 0; kb < K_blocks; kb++) {
//                 // (kb << 4) moves 16 float elements forward
//                 sum += compute_block_dot_product_f32(q_row + (kb << 4), b_col + (kb << 4));
//             }

//             // Atomic store to the destination
//             atomic_store_f32((volatile float*)(dst_data + (n * M) + m), sum);
//         }
//     }
//     return 0;
// }
