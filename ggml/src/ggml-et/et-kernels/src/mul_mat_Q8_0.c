//******************************************************************************
// MUL_MAT Kernel
// Matrix multiplication: C[M,N] = A[M,K] * B[K,N]
//******************************************************************************

#include <stdint.h>
#include "ggml_tensor.h"
#include "platform.h"
#include "math_fp.h"
#include "quants.h"
#include "block_ops.h"

// Using the block prefetch logic
static inline void prefetch_weight_row(const void* start_ptr, int64_t num_blocks, uint32_t worker_id) {
    const uint64_t cache_line_size = 64;
    uintptr_t self_ptr = (uintptr_t)start_ptr;
    uintptr_t self_ptr_end = self_ptr + (num_blocks * sizeof(block_q8_0));

    // 1. Align to cache lines and calculate range
    uint64_t startCL = (self_ptr + 63) >> 6;
    uint64_t endCL = (self_ptr_end + 63) >> 6;
    if (endCL >= startCL) {
        uint64_t total_lines = endCL - startCL + 1;
        // 2. Load balance across the minions in the Shire (assuming 8 per group for this logic)
        // Adjust worker_id if using global_id
        uint32_t local_worker_id = worker_id % 8;
        uint64_t lines_per_minion = total_lines >> 3;
        uint64_t extra = total_lines & 7;

        uint64_t offset = local_worker_id * lines_per_minion;
        if (local_worker_id < extra) {
            offset += local_worker_id;
            lines_per_minion++;
        } else {
            offset += extra;
        }
        self_ptr = (startCL + offset) << 6;
        int pending_lines = lines_per_minion;

        // 3. Hardware Prefetch Loop (16 lines at a time)
        for (; pending_lines > 0; pending_lines -= 16) {
            uint64_t current_batch = (pending_lines > 16 ? 16 : pending_lines) - 1;
            uint64_t self_size = current_batch; // bits 3:0

            __asm__ __volatile__ (
                "li    x1, 0x400000000000000 \n"  // Dest = L2 (bits 59:58 = 01)
                "addi  x31, zero, 64\n"  // Stride = 64 bytes
                "or    x3, x1, %[ptr]\n"  // Combine Dest + VA
                "or    x3, x3, %[sz]\n"  // Combine with NumLines
                "csrw  0x81f, x3\n"  // prefetch_va
                :
                : [ptr] "r" (self_ptr),
                  [sz] "r" (self_size)
                : "x1", "x3", "x31", "memory"
            );
            self_ptr += (16 * 64);
        }
    }
}

int entry_point(struct ggml_et_binary_params* params, void* env) {
    uint64_t hart_id = get_hart_id();
    uint64_t global_id = ((hart_id >> 6) << 5) + ((hart_id >> 1) & 0x1F);
    // if (hart_id & 1) {
    //     return 0;  // Odd thread - skip work
    // }
    const int64_t K = params->src0.ne[0];
    const int64_t M = params->src0.ne[1];
    const int64_t N = params->src1.ne[1];
    const int64_t K_blocks = K >> 5;

    const block_q8_0* src0_data = (const block_q8_0*)params->src0.data;
    const float* src1_data = (const float*)params->src1.data;
    float* dst_data = (float*)params->dst.data;

    for (int64_t m = hart_id; m < M; m += 2048) {
    // for (int64_t m = global_id; m < M; m += 1024) {
        for (int64_t n = 0; n < N; n++) {
            float sum = 0.0f;

            const block_q8_0* q_row = src0_data + (m * K_blocks);
            const float* b_col = src1_data + (n * K);

            for (int64_t kb = 0; kb < K_blocks; kb++) {
                sum += compute_block_dot_product_q8_0(q_row + kb, b_col + (kb << 5)); // b_col + (kb * 32) = b_col + (kb << 5)
            }
            atomic_store_f32((volatile float*)(dst_data + (n * M) + m), sum);
        }
    }
    return 0;
}
