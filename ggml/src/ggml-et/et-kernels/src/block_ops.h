//******************************************************************************
// ET Vectorized Block Operations Library
// Provides optimized block-level operations using ET hardware vector instructions
//******************************************************************************

#ifndef BLOCK_OPS_H
#define BLOCK_OPS_H

#include <stdint.h>
#include "math_fp.h"
#include "quants.h"

//******************************************************************************
// Block Dot Product Operations
//******************************************************************************
inline void __attribute__((always_inline)) excl_mode(uint64_t val)
{
    __asm__ __volatile__("csrw 0x7d3, %[csr_enc]\n" : : [csr_enc] "r"(val) : "x31");
}

// Compute dot product between dequantized q8_0 block and f32 column vector
// Vectorized: processes 8 elements at a time using ET vector instructions
// Block size: 32 int8 values (QK8_0)
static inline float compute_block_dot_product_q8_0(const block_q8_0* a_block, const float* b_col_start) {

    // Set mask register to enable all 8 vector elements
    unsigned long temp_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(temp_mask));  // Save current mask
    __asm__ volatile("mov.m.x m0, x0, 0xFF");           // Enable all 8 elements
    __asm__ volatile("fbci.pi f10, 0" ::: "f10");       // Use f10 as accumulator, init to 0

    static const int32_t gather_pattern[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    __asm__ volatile("flw.ps f31, %[gather]\n" : : [gather] "m"(*(const int32_t(*)[8])gather_pattern) : "f31");

    // Process 32 elements in 4 chunks of 8 elements each
    for (int chunk = 0; chunk < 4; chunk++) {
        int offset = chunk << 3; // chunk * 8

        __asm__ volatile(
            "flw.ps f12, %[b_vec]\n"                 // Load 8 B values (floats)
            "fgb.ps f11, f31(%[a_ptr])\n"            // Gather 8 int8 bytes from A using pattern
            "fcvt.ps.pw f11, f11\n"                  // Convert int8 vector to float vector
            "fmadd.ps f10, f11, f12, f10\n"          // acc += a_vec * b_vec (8-wide)
            :
            : [a_ptr] "r"(&a_block->qs[offset]),
              [b_vec] "m"(*(const float(*)[8])&b_col_start[offset]),
              [scale] "m"(a_block->d)
            : "f10", "f11", "f12"
        );
    }

    // Horizontal sum: reduce f10 into a single scalar
    float final_sum;
    __asm__ __volatile__ (
        // Pairwise sum within each 128-bit half
        "fswizz.ps f1, f10, 0xB1 \n\t"             // Swaps: e0<->e1 and e2<->e3
        "fadd.ps   f2, f10, f1, rne \n\t"
        // Complete the sum for each 128-bit half
        "fswizz.ps f3, f2, 0x4E \n\t"              // Swaps: e0,e1 <-> e2,e3
        "fadd.ps   f4, f2, f3, rne \n\t"
        // Sum across the two 128b halfs
        "fmvz.x.ps t0, f4, 4 \n\t"
        "fbcx.ps   f5, t0 \n\t"
        "fadd.ps   %[vout], f4, f5, rne \n\t"
        : [vout] "=f" (final_sum)
        :: "t0", "f10", "f2", "f3", "f4", "f5"
    );

    // Restore original mask
    __asm__ volatile("mova.m.x %0" :: "r"(temp_mask));

    const float scale = fp16_to_fp32(a_block->d);
    return final_sum * scale;
}

//******************************************************************************
// Split-phase Q8_0 dot product API
//
//   q8_dot_begin(st)      — save mask, set mask 0xFF
//   q8_dot_reset()        — zero vector accumulator f20
//   q8_dot_tile(q, b, n)  — accumulate n Q8_0 blocks into f20
//   q8_dot_reduce()       — horizontal sum of f20, return scalar float
//   q8_dot_teardown(st)   — restore original mask
//
// Register contract:
//   f20       — row accumulator (persistent across tiles, reset per row)
//   f31       — gather pattern (reloaded per q8_dot_tile call)
//   f10-f12   — scratch within tile
//   f15       — scale broadcast within tile
//   f1-f5, t0 — scratch within reduce
//******************************************************************************

static inline void __attribute__((always_inline))
q8_dot_reset(void) {
    __asm__ volatile("fbci.pi f20, 0" ::: "f20");
}

// Accumulate n_blocks Q8_0 blocks into f20.
// Uses fg32b.ps (fast gather with scalar pattern) for aligned chunks,
// falls back to fgb.ps for chunks crossing a 32-byte boundary.
static inline void __attribute__((always_inline))
q8_dot_tile(const block_q8_0* q_row, const float* b_col, int64_t n_blocks) {
    const int32_t gather_pattern[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    const uint64_t gather_0_to_7 = 0x398a418820ULL;

    __asm__ volatile("flw.ps f31, %[g]\n"
                     : : [g] "m"(*(const int32_t(*)[8])gather_pattern)
                     : "f31");

    for (int64_t kb = 0; kb < n_blocks; kb++) {
        const block_q8_0* blk = q_row + kb;
        const float* b_ptr = b_col + (kb << 5);
        const uintptr_t qs_addr = (uintptr_t)blk->qs;
        const uintptr_t qs_aligned = qs_addr & ~(uintptr_t)31;
        const uintptr_t qs_low = qs_addr & 31;
        const int fast_chunks = (int)((32 - qs_low) >> 3);

        if (fast_chunks >= 3) {
            __asm__ volatile(
                "fbci.pi     f10, 0\n"
                "flw.ps      f12, %[bv0]\n"
                "fg32b.ps    f11, %[gi](%[ap0])\n"
                "fcvt.ps.pw  f11, f11\n"
                "fmadd.ps    f10, f11, f12, f10\n"
                "flw.ps      f12, %[bv1]\n"
                "fg32b.ps    f11, %[gi](%[ap1])\n"
                "fcvt.ps.pw  f11, f11\n"
                "fmadd.ps    f10, f11, f12, f10\n"
                "flw.ps      f12, %[bv2]\n"
                "fg32b.ps    f11, %[gi](%[ap2])\n"
                "fcvt.ps.pw  f11, f11\n"
                "fmadd.ps    f10, f11, f12, f10\n"
                "flw.ps      f12, %[bv3]\n"
                "fgb.ps      f11, f31(%[ap3])\n"
                "fcvt.ps.pw  f11, f11\n"
                "fmadd.ps    f10, f11, f12, f10\n"
                :
                : [gi]  "r"(gather_0_to_7),
                  [ap0] "r"(qs_addr),
                  [ap1] "r"(qs_aligned | ((qs_addr + 8)  & 31)),
                  [ap2] "r"(qs_aligned | ((qs_addr + 16) & 31)),
                  [ap3] "r"(&blk->qs[24]),
                  [bv0] "m"(*(const float(*)[8])&b_ptr[0]),
                  [bv1] "m"(*(const float(*)[8])&b_ptr[8]),
                  [bv2] "m"(*(const float(*)[8])&b_ptr[16]),
                  [bv3] "m"(*(const float(*)[8])&b_ptr[24])
                : "f10", "f11", "f12"
            );
        } else if (fast_chunks == 2) {
            __asm__ volatile(
                "fbci.pi     f10, 0\n"
                "flw.ps      f12, %[bv0]\n"
                "fg32b.ps    f11, %[gi](%[ap0])\n"
                "fcvt.ps.pw  f11, f11\n"
                "fmadd.ps    f10, f11, f12, f10\n"
                "flw.ps      f12, %[bv1]\n"
                "fg32b.ps    f11, %[gi](%[ap1])\n"
                "fcvt.ps.pw  f11, f11\n"
                "fmadd.ps    f10, f11, f12, f10\n"
                "flw.ps      f12, %[bv2]\n"
                "fgb.ps      f11, f31(%[ap2])\n"
                "fcvt.ps.pw  f11, f11\n"
                "fmadd.ps    f10, f11, f12, f10\n"
                "flw.ps      f12, %[bv3]\n"
                "fgb.ps      f11, f31(%[ap3])\n"
                "fcvt.ps.pw  f11, f11\n"
                "fmadd.ps    f10, f11, f12, f10\n"
                :
                : [gi]  "r"(gather_0_to_7),
                  [ap0] "r"(qs_addr),
                  [ap1] "r"(qs_aligned | ((qs_addr + 8) & 31)),
                  [ap2] "r"(&blk->qs[16]),
                  [ap3] "r"(&blk->qs[24]),
                  [bv0] "m"(*(const float(*)[8])&b_ptr[0]),
                  [bv1] "m"(*(const float(*)[8])&b_ptr[8]),
                  [bv2] "m"(*(const float(*)[8])&b_ptr[16]),
                  [bv3] "m"(*(const float(*)[8])&b_ptr[24])
                : "f10", "f11", "f12"
            );
        } else if (fast_chunks == 1) {
            __asm__ volatile(
                "fbci.pi     f10, 0\n"
                "flw.ps      f12, %[bv0]\n"
                "fg32b.ps    f11, %[gi](%[ap0])\n"
                "fcvt.ps.pw  f11, f11\n"
                "fmadd.ps    f10, f11, f12, f10\n"
                "flw.ps      f12, %[bv1]\n"
                "fgb.ps      f11, f31(%[ap1])\n"
                "fcvt.ps.pw  f11, f11\n"
                "fmadd.ps    f10, f11, f12, f10\n"
                "flw.ps      f12, %[bv2]\n"
                "fgb.ps      f11, f31(%[ap2])\n"
                "fcvt.ps.pw  f11, f11\n"
                "fmadd.ps    f10, f11, f12, f10\n"
                "flw.ps      f12, %[bv3]\n"
                "fgb.ps      f11, f31(%[ap3])\n"
                "fcvt.ps.pw  f11, f11\n"
                "fmadd.ps    f10, f11, f12, f10\n"
                :
                : [gi]  "r"(gather_0_to_7),
                  [ap0] "r"(qs_addr),
                  [ap1] "r"(&blk->qs[8]),
                  [ap2] "r"(&blk->qs[16]),
                  [ap3] "r"(&blk->qs[24]),
                  [bv0] "m"(*(const float(*)[8])&b_ptr[0]),
                  [bv1] "m"(*(const float(*)[8])&b_ptr[8]),
                  [bv2] "m"(*(const float(*)[8])&b_ptr[16]),
                  [bv3] "m"(*(const float(*)[8])&b_ptr[24])
                : "f10", "f11", "f12"
            );
        } else {
            __asm__ volatile(
                "fbci.pi     f10, 0\n"
                "flw.ps      f12, %[bv0]\n"
                "fgb.ps      f11, f31(%[ap0])\n"
                "fcvt.ps.pw  f11, f11\n"
                "fmadd.ps    f10, f11, f12, f10\n"
                "flw.ps      f12, %[bv1]\n"
                "fgb.ps      f11, f31(%[ap1])\n"
                "fcvt.ps.pw  f11, f11\n"
                "fmadd.ps    f10, f11, f12, f10\n"
                "flw.ps      f12, %[bv2]\n"
                "fgb.ps      f11, f31(%[ap2])\n"
                "fcvt.ps.pw  f11, f11\n"
                "fmadd.ps    f10, f11, f12, f10\n"
                "flw.ps      f12, %[bv3]\n"
                "fgb.ps      f11, f31(%[ap3])\n"
                "fcvt.ps.pw  f11, f11\n"
                "fmadd.ps    f10, f11, f12, f10\n"
                :
                : [ap0] "r"(&blk->qs[0]),
                  [ap1] "r"(&blk->qs[8]),
                  [ap2] "r"(&blk->qs[16]),
                  [ap3] "r"(&blk->qs[24]),
                  [bv0] "m"(*(const float(*)[8])&b_ptr[0]),
                  [bv1] "m"(*(const float(*)[8])&b_ptr[8]),
                  [bv2] "m"(*(const float(*)[8])&b_ptr[16]),
                  [bv3] "m"(*(const float(*)[8])&b_ptr[24])
                : "f10", "f11", "f12"
            );
        }

        // f20 += f10 * broadcast(scale) — hardware fp16→fp32 via FCVT.PS.F16
        uint32_t scale_raw = (uint32_t)blk->d;
        __asm__ volatile(
            "fbcx.ps f15, %[sb]\n"
            "fcvt.ps.f16 f15, f15\n"
            "fmadd.ps f20, f10, f15, f20\n"
            :
            : [sb] "r"(scale_raw)
            : "f15", "f20"
        );
    }
}

// Horizontal sum of 8-element vector accumulator f20.
static inline float __attribute__((always_inline))
q8_dot_reduce(void) {
    float result;
    __asm__ __volatile__ (
        "fswizz.ps f1, f20, 0xB1 \n\t"
        "fadd.ps   f2, f20, f1, rne \n\t"
        "fswizz.ps f3, f2, 0x4E \n\t"
        "fadd.ps   f4, f2, f3, rne \n\t"
        "fmvz.x.ps t0, f4, 4 \n\t"
        "fbcx.ps   f5, t0 \n\t"
        "fadd.ps   %[vout], f4, f5, rne \n\t"
        : [vout] "=f" (result)
        :: "t0", "f1", "f2", "f3", "f4", "f5"
    );
    return result;
}

// Full-row dot product (convenience wrapper)
static inline float compute_row_dot_q8_0(const block_q8_0* q_row,
                                         const float* b_col,
                                         int64_t K_blocks) {
    unsigned long saved_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(saved_mask));
    __asm__ volatile("mov.m.x m0, x0, 0xFF");
    q8_dot_reset();
    q8_dot_tile(q_row, b_col, K_blocks);
    float result = q8_dot_reduce();
    __asm__ volatile("mova.m.x %0" :: "r"(saved_mask));
    return result;
}


// Compute dot product between f16 block and f32 column vector (NAIVE VERSION)
// Scalar implementation for debugging - no vectorization
// Block size: 32 f16 values (64 bytes = 1 cache line)
static inline float compute_block_dot_product_f16_naive(const uint16_t* a_block, const float* b_col_start) {
    float acc_vec[8] __attribute__ ((aligned (32))) = {0.0f};
    // Byte offsets for 16-bit (half-word) elements
    static const int32_t gather_pattern[8] = {0, 2, 4, 6, 8, 10, 12, 14};
    unsigned long temp_mask;

    __asm__ volatile("mova.x.m %0" : "=r"(temp_mask));
    __asm__ volatile("mov.m.x m0, x0, 0xFF");

    // Load the pattern once into f31 for the duration of all 4 chunks
    __asm__ volatile("flw.ps f31, %[gather]\n" : : [gather] "m"(*(const int32_t(*)[8])gather_pattern) : "f31");

    for (int chunk = 0; chunk < 4; chunk++) {
        // Correct pointers:
        // a_block elements are 2 bytes, b_col elements are 4 bytes
        const uint16_t* a_ptr = &a_block[chunk << 3]; // chunk * 8
        const float* b_ptr = &b_col_start[chunk << 3]; // chunk * 8

        __asm__ volatile(
            "flw.ps f10, %[acc]\n"
            "fgh.ps f11, f31(%[a_p])\n"      // Uses {0,2,4,6,8,10,12,14} byte offsets
            "fcvt.ps.f16 f11, f11\n"
            "flw.ps f12, (%[b_p])\n"         // Standard vector load (32-bit floats)
            "fmadd.ps f10, f11, f12, f10\n"
            "fsw.ps f10, %[result]\n"

            : [result] "=m"(*(float(*)[8])acc_vec)
            : [acc] "m"(*(const float(*)[8])acc_vec),
              [a_p] "r"(a_ptr),
              [b_p] "r"(b_ptr)
            : "f10", "f11", "f12"
        );
    }

    __asm__ volatile("mova.m.x %0" :: "r"(temp_mask));

    return acc_vec[0] + acc_vec[1] + acc_vec[2] + acc_vec[3] +
           acc_vec[4] + acc_vec[5] + acc_vec[6] + acc_vec[7];

}


// Compute dot product between f16 block and f32 column vector
// SCALAR implementation for partial blocks
// Block size: up to 32 f16 values (can handle partial blocks for misaligned K)
static inline float compute_block_dot_product_f16_partial(const uint16_t* a_block, const float* b_col_start, int elements) {
    // This matches compute_block_dot_product_f16_naive behavior
    float sum = 0.0f;

    for (int i = 0; i < elements; i++) {
        float a_val = fp16_to_fp32(a_block[i]);
        float b_val = b_col_start[i];
        sum += a_val * b_val;
    }

    return sum;
}

// Compute dot product between f16 block and f16 column vector
// Scalar implementation for generic non-matrix-engine fallback paths.
static inline float compute_block_dot_product_f16_f16_partial(const uint16_t* a_block, const uint16_t* b_col_start, int elements) {
    float sum = 0.0f;

    for (int i = 0; i < elements; i++) {
        sum += fp16_to_fp32(a_block[i]) * fp16_to_fp32(b_col_start[i]);
    }

    return sum;
}

// Compute dot product between f16 block and f32 column vector
// Vectorized: processes 8 elements at a time using ET vector instructions
// Block size: 32 f16 values (64 bytes = 1 cache line)
static inline float compute_block_dot_product_f16(const uint16_t* a_block, const float* b_col_start) {
    return compute_block_dot_product_f16_partial(a_block, b_col_start, QK_F16);
}

// Compute dot product between f32 block and f32 column vector
// Vectorized: processes 8 elements at a time using ET vector instructions
// Block size: up to 16 f32 values (can handle partial blocks for misaligned K)
static inline float compute_block_dot_product_f32_partial(const float* a_block, const float* b_col_start, int elements) {
    float acc_vec[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}; // Accumulator vector

    // Calculate how many full 8-element chunks we can process
    int vec_end = (elements / 8) * 8;

    if (vec_end > 0) {
        // Set mask register to enable all 8 vector elements
        unsigned long temp_mask;
        __asm__ volatile("mova.x.m %0" : "=r"(temp_mask));  // Save current mask
        __asm__ volatile("mov.m.x m0, x0, 0xFF");           // Enable all 8 elements

        // Process full 8-element chunks
        for (int i = 0; i < vec_end; i += 8) {
            // Vectorized f32 multiply-accumulate
            __asm__ volatile(
                "flw.ps f10, %[acc]\n"                   // Load current accumulator (8 floats)
                "flw.ps f11, %[a_vec]\n"                 // Load 8 A values (f32)
                "flw.ps f12, %[b_vec]\n"                 // Load 8 B values (f32)
                "fmadd.ps f10, f11, f12, f10\n"          // acc += a_vec * b_vec (8-wide)
                "fsw.ps f10, %[result]\n"                // Store back to accumulator

                : [result] "=m"(*(float(*)[8])acc_vec)
                : [acc] "m"(*(const float(*)[8])acc_vec),
                  [a_vec] "m"(*(const float(*)[8])(a_block + i)),
                  [b_vec] "m"(*(const float(*)[8])(b_col_start + i))
                : "f10", "f11", "f12"
            );
        }

        // Restore original mask
        __asm__ volatile("mova.m.x %0" :: "r"(temp_mask));
    }

    // Horizontal sum: reduce 8 accumulator elements to single scalar
    float final_sum = 0.0f;
    for (int i = 0; i < 8; i++) {
        final_sum += acc_vec[i];
    }

    // Handle remaining elements (< 8) with scalar operations
    for (int i = vec_end; i < elements; i++) {
        final_sum += a_block[i] * b_col_start[i];
    }

    return final_sum;
}

// Compute dot product between f32 block and f16 column vector
// Scalar implementation for generic non-matrix-engine fallback paths.
static inline float compute_block_dot_product_f32_f16_partial(const float* a_block, const uint16_t* b_col_start, int elements) {
    float sum = 0.0f;

    for (int i = 0; i < elements; i++) {
        sum += a_block[i] * fp16_to_fp32(b_col_start[i]);
    }

    return sum;
}







// Compute dot product between f32 block and f32 column vector
// Vectorized: processes 8 elements at a time using ET vector instructions
// Block size: 16 f32 values (64 bytes = 1 cache line)
static inline float compute_block_dot_product_f32(const float* a_block, const float* b_col_start) {
    return compute_block_dot_product_f32_partial(a_block, b_col_start, QK_F32);

    // float acc_vec[8];
    // unsigned long old_mask;
    // __asm__ volatile(
    //     // Save current mask
    //     "mova.x.m %[old_mask]\n"
    //     // Enable all 8 lanes
    //     "mov.m.x m0, x0, 0xFF\n"

    //     "flw.ps  f11, %[a]\n"
    //     "flw.ps  f12, %[b]\n"
    //     "fmadd.ps f10, f11, f12, f10\n"
    //     "fsw.ps  f10, %[out]\n"
    //     "mova.m.x %[old_mask]\n"

    //     : [out] "=m" (*(float(*)[8])acc_vec),
    //       [old_mask] "=r"(old_mask)
    //     : [a] "m" (*(const float(*)[8])a_block),
    //       [b] "m" (*(const float(*)[8])b_col_start)
    //     : "f10", "f11", "f12"
    // );

    // // Horizontal reduction
    // return acc_vec[0] + acc_vec[1] + acc_vec[2] + acc_vec[3] +
    //        acc_vec[4] + acc_vec[5] + acc_vec[6] + acc_vec[7];

}

#endif // BLOCK_OPS_H
