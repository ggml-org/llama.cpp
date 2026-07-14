//******************************************************************************
// ET Vectorized Block Operations Library
// Provides optimized block-level operations using ET hardware vector instructions
//******************************************************************************

#ifndef BLOCK_OPS_H
#    define BLOCK_OPS_H

#    include "math_fp.h"
#    include "platform.h"
#    include "quants.h"

#    include <stdint.h>

//******************************************************************************
// Block Dot Product Operations
//******************************************************************************
inline void __attribute__((always_inline)) excl_mode(uint64_t val) {
    __asm__ __volatile__("csrw 0x7d3, %[csr_enc]\n" : : [csr_enc] "r"(val) : "x31");
}

static inline float compute_block_dot_product_q4_0(const block_q4_0 * a_block, const float * b_col_start) {
    // Set mask register to enable all 8 vector elements
    unsigned long temp_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(temp_mask));  // Save current mask
    __asm__ volatile("mov.m.x m0, x0, 0xFF");           // Enable all 8 elements

    // Use f10 as accumulator, init to 0
    __asm__ volatile("fbci.ps f10, 0" ::: "f10");

    static const int32_t gather_pattern[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    __asm__ volatile("flw.ps f31, %[gather]\n" : : [gather] "m"(*(const int32_t (*)[8]) gather_pattern) : "f31");

    // Process 32 elements in 2 chunks of 16 elements (8 bytes) each
    for (int chunk = 0; chunk < 2; chunk++) {
        int offset_a      = chunk * 8;
        int offset_b_low  = chunk * 8;       // Activations for lower nibbles
        int offset_b_high = chunk * 8 + 16;  // Activations for upper nibbles (16 elements later)

        __asm__ volatile(
            "fgb.ps f11, f31(%[a_ptr])\n"  // Gather 8 bytes (16 packed q4_0 weights)

            // 1. Extract & Multiply Lower Nibbles
            "fandi.pi f12, f11, 15\n"             // Mask lower 4 bits (x & 0xF)
            "faddi.pi f12, f12, -8\n"             // GGML offset to signed: (x & 0xF) - 8
            "fcvt.ps.pw f12, f12, rne\n"          // Convert INT32 to FP32
            "flw.ps f13, 0(%[b_low])\n"           // Load 8 B values (floats)
            "fmadd.ps f10, f12, f13, f10, rne\n"  // acc += A_low * B_low

            // 2. Extract & Multiply Upper Nibbles
            "fsrli.pi f14, f11, 4\n"              // Shift upper 4 bits down
            "fandi.pi f14, f14, 15\n"             // Mask new lower 4 bits
            "faddi.pi f14, f14, -8\n"             // GGML offset to signed
            "fcvt.ps.pw f14, f14, rne\n"          // Convert INT32 to FP32
            "flw.ps f15, 0(%[b_high])\n"          // Load next 8 B values (floats)
            "fmadd.ps f10, f14, f15, f10, rne\n"  // acc += A_high * B_high
            :
            : [a_ptr] "r"(&a_block->qs[offset_a]), [b_low] "r"(&b_col_start[offset_b_low]),
              [b_high] "r"(&b_col_start[offset_b_high])
            // Note: f10 is explicitly NOT listed in the clobbers here to ensure the compiler
            // preserves the running sum across C loop iterations safely.
            : "f11", "f12", "f13", "f14", "f15");
    }

    // Horizontal sum: reduce f10 into a single scalar
    float final_sum;
    __asm__ __volatile__(
        // Pairwise sum within each 128-bit half
        "fswizz.ps f1, f10, 0xB1 \n\t"  // Swaps: e0<->e1 and e2<->e3
        "fadd.ps   f2, f10, f1, rne \n\t"
        // Complete the sum for each 128-bit half
        "fswizz.ps f3, f2, 0x4E \n\t"  // Swaps: e0,e1 <-> e2,e3
        "fadd.ps   f4, f2, f3, rne \n\t"
        // Sum across the two 128b halfs
        "fmvz.x.ps t0, f4, 4 \n\t"
        "fbcx.ps   f5, t0 \n\t"
        "fadd.ps   %[vout], f4, f5, rne \n\t"
        : [vout] "=f"(final_sum)::"t0", "f1", "f2", "f3", "f4", "f5", "f10");

    // Restore original mask
    __asm__ volatile("mova.m.x %0" ::"r"(temp_mask));

    const float scale = fp16_to_fp32(a_block->d);
    return final_sum * scale;
}

// Compute dot product between dequantized q8_0 block and f32 column vector
// Vectorized: processes 8 elements at a time using ET vector instructions
// Block size: 32 int8 values (QK8_0)
static inline float compute_block_dot_product_q8_0(const block_q8_0 * a_block, const float * b_col_start) {
    // Set mask register to enable all 8 vector elements
    unsigned long temp_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(temp_mask));  // Save current mask
    __asm__ volatile("mov.m.x m0, x0, 0xFF");           // Enable all 8 elements
    __asm__ volatile("fbci.pi f10, 0" ::: "f10");       // Use f10 as accumulator, init to 0

    static const int32_t gather_pattern[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };

    __asm__ volatile("flw.ps f31, %[gather]\n" : : [gather] "m"(*(const int32_t (*)[8]) gather_pattern) : "f31");

    // Process 32 elements in 4 chunks of 8 elements each
    for (int chunk = 0; chunk < 4; chunk++) {
        int offset = chunk << 3;  // chunk * 8

        __asm__ volatile(
            "flw.ps f12, %[b_vec]\n"         // Load 8 B values (floats)
            "fgb.ps f11, f31(%[a_ptr])\n"    // Gather 8 int8 bytes from A using pattern
            "fcvt.ps.pw f11, f11\n"          // Convert int8 vector to float vector
            "fmadd.ps f10, f11, f12, f10\n"  // acc += a_vec * b_vec (8-wide)
            :
            : [a_ptr] "r"(&a_block->qs[offset]), [b_vec] "m"(*(const float (*)[8]) & b_col_start[offset]),
              [scale] "m"(a_block->d)
            : "f10", "f11", "f12");
    }

    // Horizontal sum: reduce f10 into a single scalar
    float final_sum;
    __asm__ __volatile__(
        // Pairwise sum within each 128-bit half
        "fswizz.ps f1, f10, 0xB1 \n\t"  // Swaps: e0<->e1 and e2<->e3
        "fadd.ps   f2, f10, f1, rne \n\t"
        // Complete the sum for each 128-bit half
        "fswizz.ps f3, f2, 0x4E \n\t"  // Swaps: e0,e1 <-> e2,e3
        "fadd.ps   f4, f2, f3, rne \n\t"
        // Sum across the two 128b halfs
        "fmvz.x.ps t0, f4, 4 \n\t"
        "fbcx.ps   f5, t0 \n\t"
        "fadd.ps   %[vout], f4, f5, rne \n\t"
        : [vout] "=f"(final_sum)::"t0", "f10", "f2", "f3", "f4", "f5");

    // Restore original mask
    __asm__ volatile("mova.m.x %0" ::"r"(temp_mask));

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

static inline void __attribute__((always_inline)) q8_dot_reset(void) {
    __asm__ volatile("fbci.pi f20, 0" ::: "f20");
}

// Accumulate n_blocks Q8_0 blocks into f20.
// Uses fg32b.ps (fast gather with scalar pattern) for aligned chunks,
// falls back to fgb.ps for chunks crossing a 32-byte boundary.
static inline void __attribute__((always_inline)) q8_dot_tile(const block_q8_0 * q_row,
                                                              const float *      b_col,
                                                              int64_t            n_blocks) {
    const int32_t  gather_pattern[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    const uint64_t gather_0_to_7     = 0x398a418820ULL;

    __asm__ volatile("flw.ps f31, %[g]\n" : : [g] "m"(*(const int32_t (*)[8]) gather_pattern) : "f31");

    for (int64_t kb = 0; kb < n_blocks; kb++) {
        const block_q8_0 * blk         = q_row + kb;
        const float *      b_ptr       = b_col + (kb << 5);
        const uintptr_t    qs_addr     = (uintptr_t) blk->qs;
        const uintptr_t    qs_aligned  = qs_addr & ~(uintptr_t) 31;
        const uintptr_t    qs_low      = qs_addr & 31;
        const int          fast_chunks = (int) ((32 - qs_low) >> 3);

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
                : [gi] "r"(gather_0_to_7), [ap0] "r"(qs_addr), [ap1] "r"(qs_aligned | ((qs_addr + 8) & 31)),
                  [ap2] "r"(qs_aligned | ((qs_addr + 16) & 31)), [ap3] "r"(&blk->qs[24]),
                  [bv0] "m"(*(const float (*)[8]) & b_ptr[0]), [bv1] "m"(*(const float (*)[8]) & b_ptr[8]),
                  [bv2] "m"(*(const float (*)[8]) & b_ptr[16]), [bv3] "m"(*(const float (*)[8]) & b_ptr[24])
                : "f10", "f11", "f12");
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
                : [gi] "r"(gather_0_to_7), [ap0] "r"(qs_addr), [ap1] "r"(qs_aligned | ((qs_addr + 8) & 31)),
                  [ap2] "r"(&blk->qs[16]), [ap3] "r"(&blk->qs[24]), [bv0] "m"(*(const float (*)[8]) & b_ptr[0]),
                  [bv1] "m"(*(const float (*)[8]) & b_ptr[8]), [bv2] "m"(*(const float (*)[8]) & b_ptr[16]),
                  [bv3] "m"(*(const float (*)[8]) & b_ptr[24])
                : "f10", "f11", "f12");
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
                : [gi] "r"(gather_0_to_7), [ap0] "r"(qs_addr), [ap1] "r"(&blk->qs[8]), [ap2] "r"(&blk->qs[16]),
                  [ap3] "r"(&blk->qs[24]), [bv0] "m"(*(const float (*)[8]) & b_ptr[0]),
                  [bv1] "m"(*(const float (*)[8]) & b_ptr[8]), [bv2] "m"(*(const float (*)[8]) & b_ptr[16]),
                  [bv3] "m"(*(const float (*)[8]) & b_ptr[24])
                : "f10", "f11", "f12");
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
                : [ap0] "r"(&blk->qs[0]), [ap1] "r"(&blk->qs[8]), [ap2] "r"(&blk->qs[16]), [ap3] "r"(&blk->qs[24]),
                  [bv0] "m"(*(const float (*)[8]) & b_ptr[0]), [bv1] "m"(*(const float (*)[8]) & b_ptr[8]),
                  [bv2] "m"(*(const float (*)[8]) & b_ptr[16]), [bv3] "m"(*(const float (*)[8]) & b_ptr[24])
                : "f10", "f11", "f12");
        }

        // f20 += f10 * broadcast(scale) — hardware fp16→fp32 via FCVT.PS.F16
        uint32_t scale_raw = (uint32_t) blk->d;
        __asm__ volatile(
            "fbcx.ps f15, %[sb]\n"
            "fcvt.ps.f16 f15, f15\n"
            "fmadd.ps f20, f10, f15, f20\n"
            :
            : [sb] "r"(scale_raw)
            : "f15", "f20");
    }
}

// Horizontal sum of 8-element vector accumulator f20.
static inline float __attribute__((always_inline)) q8_dot_reduce(void) {
    float result;
    __asm__ __volatile__(
        "fswizz.ps f1, f20, 0xB1 \n\t"
        "fadd.ps   f2, f20, f1, rne \n\t"
        "fswizz.ps f3, f2, 0x4E \n\t"
        "fadd.ps   f4, f2, f3, rne \n\t"
        "fmvz.x.ps t0, f4, 4 \n\t"
        "fbcx.ps   f5, t0 \n\t"
        "fadd.ps   %[vout], f4, f5, rne \n\t"
        : [vout] "=f"(result)::"t0", "f1", "f2", "f3", "f4", "f5");
    return result;
}

// Full-row dot product (convenience wrapper)
static inline float compute_row_dot_q8_0(const block_q8_0 * q_row, const float * b_col, int64_t K_blocks) {
    unsigned long saved_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(saved_mask));
    __asm__ volatile("mov.m.x m0, x0, 0xFF");
    q8_dot_reset();
    q8_dot_tile(q_row, b_col, K_blocks);
    float result = q8_dot_reduce();
    __asm__ volatile("mova.m.x %0" ::"r"(saved_mask));
    return result;
}

//******************************************************************************
// Hoisted Q8_0 dot API
//
// q8_dot_begin/end save/restore the vector mask once around a long sequence of
// dot products, so the per-row mask shuffles are hoisted out of the inner
// loops. q8_dot_compute does a full-row dot (no mask handling). The _x2
// variant computes two rows together while reusing each loaded B chunk —
// only safe when both row pointers share the same 32-byte alignment phase
// (i.e. the Q8 row stride is a multiple of 32).
//******************************************************************************

typedef struct {
    unsigned long saved_mask;
} q8_dot_state;

static inline void q8_dot_begin(q8_dot_state * state) {
    __asm__ volatile("mova.x.m %0" : "=r"(state->saved_mask));
    __asm__ volatile("mov.m.x m0, x0, 0xFF");
}

static inline void q8_dot_end(const q8_dot_state * state) {
    __asm__ volatile("mova.m.x %0" ::"r"(state->saved_mask));
}

// Equivalent to q8_dot_reset+tile+reduce, without touching the mask register.
// Caller is responsible for q8_dot_begin/end around the surrounding loop.
static inline float q8_dot_compute(const block_q8_0 * q_row, const float * b_col, int64_t K_blocks) {
    q8_dot_reset();
    q8_dot_tile(q_row, b_col, K_blocks);
    return q8_dot_reduce();
}

// Compute two row dots together while reusing the same loaded B chunks.
//
// Safe when every row starts at the same 32-byte offset, i.e. the Q8 row stride
// is a multiple of 32. In that case the gather/alignment pattern is the same
// for both rows at a given `kb`, so one set of B vector loads feeds both row
// accumulators.
static inline void q8_dot_compute_x2_aligned(const block_q8_0 * q_row0,
                                             const block_q8_0 * q_row1,
                                             const float *      b_col,
                                             int64_t            K_blocks,
                                             float *            out0,
                                             float *            out1) {
    const int32_t  gather_pattern[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    const uint64_t gather_0_to_7     = 0x398a418820ULL;
    __asm__ volatile("flw.ps f31, %[g]\n" : : [g] "m"(*(const int32_t (*)[8]) gather_pattern) : "f31");
    __asm__ volatile(
        "fbci.pi f20, 0\n"
        "fbci.pi f21, 0\n" ::
            : "f20", "f21");

    for (int64_t kb = 0; kb < K_blocks; kb++) {
        const block_q8_0 * blk0  = q_row0 + kb;
        const block_q8_0 * blk1  = q_row1 + kb;
        const float *      b_ptr = b_col + (kb << 5);

        const uintptr_t qs_addr0    = (uintptr_t) blk0->qs;
        const uintptr_t qs_addr1    = (uintptr_t) blk1->qs;
        const uintptr_t qs_aligned0 = qs_addr0 & ~(uintptr_t) 31;
        const uintptr_t qs_aligned1 = qs_addr1 & ~(uintptr_t) 31;
        const int       fast_chunks = (int) ((32 - (qs_addr0 & 31)) >> 3);

        if (fast_chunks >= 3) {
            __asm__ volatile(
                "fbci.pi     f10, 0\n"
                "fbci.pi     f11, 0\n"

                "flw.ps      f12, %[bv0]\n"
                "fg32b.ps    f16, %[gi](%[r0ap0])\n"
                "fcvt.ps.pw  f16, f16\n"
                "fmadd.ps    f10, f16, f12, f10\n"
                "fg32b.ps    f17, %[gi](%[r1ap0])\n"
                "fcvt.ps.pw  f17, f17\n"
                "fmadd.ps    f11, f17, f12, f11\n"

                "flw.ps      f13, %[bv1]\n"
                "fg32b.ps    f16, %[gi](%[r0ap1])\n"
                "fcvt.ps.pw  f16, f16\n"
                "fmadd.ps    f10, f16, f13, f10\n"
                "fg32b.ps    f17, %[gi](%[r1ap1])\n"
                "fcvt.ps.pw  f17, f17\n"
                "fmadd.ps    f11, f17, f13, f11\n"

                "flw.ps      f14, %[bv2]\n"
                "fg32b.ps    f16, %[gi](%[r0ap2])\n"
                "fcvt.ps.pw  f16, f16\n"
                "fmadd.ps    f10, f16, f14, f10\n"
                "fg32b.ps    f17, %[gi](%[r1ap2])\n"
                "fcvt.ps.pw  f17, f17\n"
                "fmadd.ps    f11, f17, f14, f11\n"

                "flw.ps      f15, %[bv3]\n"
                "fgb.ps      f16, f31(%[r0ap3])\n"
                "fcvt.ps.pw  f16, f16\n"
                "fmadd.ps    f10, f16, f15, f10\n"
                "fgb.ps      f17, f31(%[r1ap3])\n"
                "fcvt.ps.pw  f17, f17\n"
                "fmadd.ps    f11, f17, f15, f11\n"
                :
                : [gi] "r"(gather_0_to_7), [r0ap0] "r"(qs_addr0), [r0ap1] "r"(qs_aligned0 | ((qs_addr0 + 8) & 31)),
                  [r0ap2] "r"(qs_aligned0 | ((qs_addr0 + 16) & 31)), [r0ap3] "r"(&blk0->qs[24]), [r1ap0] "r"(qs_addr1),
                  [r1ap1] "r"(qs_aligned1 | ((qs_addr1 + 8) & 31)), [r1ap2] "r"(qs_aligned1 | ((qs_addr1 + 16) & 31)),
                  [r1ap3] "r"(&blk1->qs[24]), [bv0] "m"(*(const float (*)[8]) & b_ptr[0]),
                  [bv1] "m"(*(const float (*)[8]) & b_ptr[8]), [bv2] "m"(*(const float (*)[8]) & b_ptr[16]),
                  [bv3] "m"(*(const float (*)[8]) & b_ptr[24])
                : "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17");
        } else if (fast_chunks == 2) {
            __asm__ volatile(
                "fbci.pi     f10, 0\n"
                "fbci.pi     f11, 0\n"

                "flw.ps      f12, %[bv0]\n"
                "fg32b.ps    f16, %[gi](%[r0ap0])\n"
                "fcvt.ps.pw  f16, f16\n"
                "fmadd.ps    f10, f16, f12, f10\n"
                "fg32b.ps    f17, %[gi](%[r1ap0])\n"
                "fcvt.ps.pw  f17, f17\n"
                "fmadd.ps    f11, f17, f12, f11\n"

                "flw.ps      f13, %[bv1]\n"
                "fg32b.ps    f16, %[gi](%[r0ap1])\n"
                "fcvt.ps.pw  f16, f16\n"
                "fmadd.ps    f10, f16, f13, f10\n"
                "fg32b.ps    f17, %[gi](%[r1ap1])\n"
                "fcvt.ps.pw  f17, f17\n"
                "fmadd.ps    f11, f17, f13, f11\n"

                "flw.ps      f14, %[bv2]\n"
                "fgb.ps      f16, f31(%[r0ap2])\n"
                "fcvt.ps.pw  f16, f16\n"
                "fmadd.ps    f10, f16, f14, f10\n"
                "fgb.ps      f17, f31(%[r1ap2])\n"
                "fcvt.ps.pw  f17, f17\n"
                "fmadd.ps    f11, f17, f14, f11\n"

                "flw.ps      f15, %[bv3]\n"
                "fgb.ps      f16, f31(%[r0ap3])\n"
                "fcvt.ps.pw  f16, f16\n"
                "fmadd.ps    f10, f16, f15, f10\n"
                "fgb.ps      f17, f31(%[r1ap3])\n"
                "fcvt.ps.pw  f17, f17\n"
                "fmadd.ps    f11, f17, f15, f11\n"
                :
                : [gi] "r"(gather_0_to_7), [r0ap0] "r"(qs_addr0), [r0ap1] "r"(qs_aligned0 | ((qs_addr0 + 8) & 31)),
                  [r0ap2] "r"(&blk0->qs[16]), [r0ap3] "r"(&blk0->qs[24]), [r1ap0] "r"(qs_addr1),
                  [r1ap1] "r"(qs_aligned1 | ((qs_addr1 + 8) & 31)), [r1ap2] "r"(&blk1->qs[16]),
                  [r1ap3] "r"(&blk1->qs[24]), [bv0] "m"(*(const float (*)[8]) & b_ptr[0]),
                  [bv1] "m"(*(const float (*)[8]) & b_ptr[8]), [bv2] "m"(*(const float (*)[8]) & b_ptr[16]),
                  [bv3] "m"(*(const float (*)[8]) & b_ptr[24])
                : "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17");
        } else if (fast_chunks == 1) {
            __asm__ volatile(
                "fbci.pi     f10, 0\n"
                "fbci.pi     f11, 0\n"

                "flw.ps      f12, %[bv0]\n"
                "fg32b.ps    f16, %[gi](%[r0ap0])\n"
                "fcvt.ps.pw  f16, f16\n"
                "fmadd.ps    f10, f16, f12, f10\n"
                "fg32b.ps    f17, %[gi](%[r1ap0])\n"
                "fcvt.ps.pw  f17, f17\n"
                "fmadd.ps    f11, f17, f12, f11\n"

                "flw.ps      f13, %[bv1]\n"
                "fgb.ps      f16, f31(%[r0ap1])\n"
                "fcvt.ps.pw  f16, f16\n"
                "fmadd.ps    f10, f16, f13, f10\n"
                "fgb.ps      f17, f31(%[r1ap1])\n"
                "fcvt.ps.pw  f17, f17\n"
                "fmadd.ps    f11, f17, f13, f11\n"

                "flw.ps      f14, %[bv2]\n"
                "fgb.ps      f16, f31(%[r0ap2])\n"
                "fcvt.ps.pw  f16, f16\n"
                "fmadd.ps    f10, f16, f14, f10\n"
                "fgb.ps      f17, f31(%[r1ap2])\n"
                "fcvt.ps.pw  f17, f17\n"
                "fmadd.ps    f11, f17, f14, f11\n"

                "flw.ps      f15, %[bv3]\n"
                "fgb.ps      f16, f31(%[r0ap3])\n"
                "fcvt.ps.pw  f16, f16\n"
                "fmadd.ps    f10, f16, f15, f10\n"
                "fgb.ps      f17, f31(%[r1ap3])\n"
                "fcvt.ps.pw  f17, f17\n"
                "fmadd.ps    f11, f17, f15, f11\n"
                :
                : [gi] "r"(gather_0_to_7), [r0ap0] "r"(qs_addr0), [r0ap1] "r"(&blk0->qs[8]), [r0ap2] "r"(&blk0->qs[16]),
                  [r0ap3] "r"(&blk0->qs[24]), [r1ap0] "r"(qs_addr1), [r1ap1] "r"(&blk1->qs[8]),
                  [r1ap2] "r"(&blk1->qs[16]), [r1ap3] "r"(&blk1->qs[24]), [bv0] "m"(*(const float (*)[8]) & b_ptr[0]),
                  [bv1] "m"(*(const float (*)[8]) & b_ptr[8]), [bv2] "m"(*(const float (*)[8]) & b_ptr[16]),
                  [bv3] "m"(*(const float (*)[8]) & b_ptr[24])
                : "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17");
        } else {
            __asm__ volatile(
                "fbci.pi     f10, 0\n"
                "fbci.pi     f11, 0\n"

                "flw.ps      f12, %[bv0]\n"
                "fgb.ps      f16, f31(%[r0ap0])\n"
                "fcvt.ps.pw  f16, f16\n"
                "fmadd.ps    f10, f16, f12, f10\n"
                "fgb.ps      f17, f31(%[r1ap0])\n"
                "fcvt.ps.pw  f17, f17\n"
                "fmadd.ps    f11, f17, f12, f11\n"

                "flw.ps      f13, %[bv1]\n"
                "fgb.ps      f16, f31(%[r0ap1])\n"
                "fcvt.ps.pw  f16, f16\n"
                "fmadd.ps    f10, f16, f13, f10\n"
                "fgb.ps      f17, f31(%[r1ap1])\n"
                "fcvt.ps.pw  f17, f17\n"
                "fmadd.ps    f11, f17, f13, f11\n"

                "flw.ps      f14, %[bv2]\n"
                "fgb.ps      f16, f31(%[r0ap2])\n"
                "fcvt.ps.pw  f16, f16\n"
                "fmadd.ps    f10, f16, f14, f10\n"
                "fgb.ps      f17, f31(%[r1ap2])\n"
                "fcvt.ps.pw  f17, f17\n"
                "fmadd.ps    f11, f17, f14, f11\n"

                "flw.ps      f15, %[bv3]\n"
                "fgb.ps      f16, f31(%[r0ap3])\n"
                "fcvt.ps.pw  f16, f16\n"
                "fmadd.ps    f10, f16, f15, f10\n"
                "fgb.ps      f17, f31(%[r1ap3])\n"
                "fcvt.ps.pw  f17, f17\n"
                "fmadd.ps    f11, f17, f15, f11\n"
                :
                : [r0ap0] "r"(&blk0->qs[0]), [r0ap1] "r"(&blk0->qs[8]), [r0ap2] "r"(&blk0->qs[16]),
                  [r0ap3] "r"(&blk0->qs[24]), [r1ap0] "r"(&blk1->qs[0]), [r1ap1] "r"(&blk1->qs[8]),
                  [r1ap2] "r"(&blk1->qs[16]), [r1ap3] "r"(&blk1->qs[24]), [bv0] "m"(*(const float (*)[8]) & b_ptr[0]),
                  [bv1] "m"(*(const float (*)[8]) & b_ptr[8]), [bv2] "m"(*(const float (*)[8]) & b_ptr[16]),
                  [bv3] "m"(*(const float (*)[8]) & b_ptr[24])
                : "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17");
        }

        const uint32_t scale_raw0 = (uint32_t) blk0->d;
        const uint32_t scale_raw1 = (uint32_t) blk1->d;
        __asm__ volatile(
            "fbcx.ps     f24, %[s0]\n"
            "fcvt.ps.f16 f24, f24\n"
            "fmadd.ps    f20, f10, f24, f20\n"
            "fbcx.ps     f25, %[s1]\n"
            "fcvt.ps.f16 f25, f25\n"
            "fmadd.ps    f21, f11, f25, f21\n"
            :
            : [s0] "r"(scale_raw0), [s1] "r"(scale_raw1)
            : "f20", "f21", "f24", "f25");
    }

    float result0;
    float result1;
    __asm__ __volatile__(
        "fswizz.ps f1, f20, 0xB1 \n\t"
        "fadd.ps   f2, f20, f1, rne \n\t"
        "fswizz.ps f3, f2, 0x4E \n\t"
        "fadd.ps   f4, f2, f3, rne \n\t"
        "fmvz.x.ps t0, f4, 4 \n\t"
        "fbcx.ps   f5, t0 \n\t"
        "fadd.ps   %[vout], f4, f5, rne \n\t"
        : [vout] "=f"(result0)::"t0", "f1", "f2", "f3", "f4", "f5");
    __asm__ __volatile__(
        "fswizz.ps f1, f21, 0xB1 \n\t"
        "fadd.ps   f2, f21, f1, rne \n\t"
        "fswizz.ps f3, f2, 0x4E \n\t"
        "fadd.ps   f4, f2, f3, rne \n\t"
        "fmvz.x.ps t0, f4, 4 \n\t"
        "fbcx.ps   f5, t0 \n\t"
        "fadd.ps   %[vout], f4, f5, rne \n\t"
        : [vout] "=f"(result1)::"t0", "f1", "f2", "f3", "f4", "f5");

    *out0 = result0;
    *out1 = result1;
}

// Compute dot product between f16 block and f32 column vector (NAIVE VERSION)
// Scalar implementation for debugging - no vectorization
// Block size: 32 f16 values (64 bytes = 1 cache line)
static inline float compute_block_dot_product_f16_naive(const uint16_t * a_block, const float * b_col_start) {
    float                acc_vec[8] __attribute__((aligned(32))) = { 0.0f };
    // Byte offsets for 16-bit (half-word) elements
    static const int32_t gather_pattern[8]                       = { 0, 2, 4, 6, 8, 10, 12, 14 };
    unsigned long        temp_mask;

    __asm__ volatile("mova.x.m %0" : "=r"(temp_mask));
    __asm__ volatile("mov.m.x m0, x0, 0xFF");

    // Load the pattern once into f31 for the duration of all 4 chunks
    __asm__ volatile("flw.ps f31, %[gather]\n" : : [gather] "m"(*(const int32_t (*)[8]) gather_pattern) : "f31");

    for (int chunk = 0; chunk < 4; chunk++) {
        // Correct pointers:
        // a_block elements are 2 bytes, b_col elements are 4 bytes
        const uint16_t * a_ptr = &a_block[chunk << 3];      // chunk * 8
        const float *    b_ptr = &b_col_start[chunk << 3];  // chunk * 8

        __asm__ volatile(
            "flw.ps f10, %[acc]\n"
            "fgh.ps f11, f31(%[a_p])\n"  // Uses {0,2,4,6,8,10,12,14} byte offsets
            "fcvt.ps.f16 f11, f11\n"
            "flw.ps f12, (%[b_p])\n"     // Standard vector load (32-bit floats)
            "fmadd.ps f10, f11, f12, f10\n"
            "fsw.ps f10, %[result]\n"

            : [result] "=m"(*(float (*)[8]) acc_vec)
            : [acc] "m"(*(const float (*)[8]) acc_vec), [a_p] "r"(a_ptr), [b_p] "r"(b_ptr)
            : "f10", "f11", "f12");
    }

    __asm__ volatile("mova.m.x %0" ::"r"(temp_mask));

    return acc_vec[0] + acc_vec[1] + acc_vec[2] + acc_vec[3] + acc_vec[4] + acc_vec[5] + acc_vec[6] + acc_vec[7];
}

// Compute dot product between f16 block and f32 column vector
// SCALAR implementation for partial blocks
// Block size: up to 32 f16 values (can handle partial blocks for misaligned K)
static inline float compute_block_dot_product_f16_partial(const uint16_t * a_block,
                                                          const float *    b_col_start,
                                                          int              elements) {
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
static inline float compute_block_dot_product_f16_f16_partial(const uint16_t * a_block,
                                                              const uint16_t * b_col_start,
                                                              int              elements) {
    float sum = 0.0f;

    for (int i = 0; i < elements; i++) {
        sum += fp16_to_fp32(a_block[i]) * fp16_to_fp32(b_col_start[i]);
    }

    return sum;
}

// Compute dot product between f16 block and f32 column vector
// Vectorized: processes 8 elements at a time using ET vector instructions
// Block size: 32 f16 values (64 bytes = 1 cache line)
static inline float compute_block_dot_product_f16(const uint16_t * a_block, const float * b_col_start) {
    return compute_block_dot_product_f16_partial(a_block, b_col_start, QK_F16);
}

// Compute dot product between f32 block and f32 column vector
// Vectorized: processes 8 elements at a time using ET vector instructions
// Block size: up to 16 f32 values (can handle partial blocks for misaligned K)
static inline float compute_block_dot_product_f32_partial(const float * a_block,
                                                          const float * b_col_start,
                                                          int           elements) {
    float acc_vec[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };  // Accumulator vector

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
                "flw.ps f10, %[acc]\n"           // Load current accumulator (8 floats)
                "flw.ps f11, %[a_vec]\n"         // Load 8 A values (f32)
                "flw.ps f12, %[b_vec]\n"         // Load 8 B values (f32)
                "fmadd.ps f10, f11, f12, f10\n"  // acc += a_vec * b_vec (8-wide)
                "fsw.ps f10, %[result]\n"        // Store back to accumulator

                : [result] "=m"(*(float (*)[8]) acc_vec)
                : [acc] "m"(*(const float (*)[8]) acc_vec), [a_vec] "m"(*(const float (*)[8])(a_block + i)),
                  [b_vec] "m"(*(const float (*)[8])(b_col_start + i))
                : "f10", "f11", "f12");
        }

        // Restore original mask
        __asm__ volatile("mova.m.x %0" ::"r"(temp_mask));
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
static inline float compute_block_dot_product_f32_f16_partial(const float *    a_block,
                                                              const uint16_t * b_col_start,
                                                              int              elements) {
    float sum = 0.0f;

    for (int i = 0; i < elements; i++) {
        sum += a_block[i] * fp16_to_fp32(b_col_start[i]);
    }

    return sum;
}

//******************************************************************************
// F32 x F32 register-resident row dot (decode GEMV). The running sum lives in
// vector registers f20..f23 for the whole row instead of being spilled to
// memory every few elements, so the inner loop is just paired flw.ps loads and
// fmadd, no gather or convert needed since both operands are already f32.
// Four independent lane accumulators break the fmadd dependency chain so
// several A/B loads stay in flight, hiding load latency.
//******************************************************************************
static inline void __attribute__((always_inline))
f32_dot_reset(void) {
    __asm__ volatile(
        "fbci.pi f20, 0\n"
        "fbci.pi f21, 0\n"
        "fbci.pi f22, 0\n"
        "fbci.pi f23, 0\n"
        ::: "f20", "f21", "f22", "f23");
}

// Accumulate n_blocks blocks of 32 f32 A values times 32 f32 B into f20..f23.
static inline void __attribute__((always_inline))
f32_dot_tile(const float * a_row, const float * b_col, int64_t n_blocks) {
    // Prefetch A ahead into L2, not L1. The weight row is streamed once and
    // never reused, but L1 is shared by both harts of a minion, so a simple
    // L1 prefetch gets evicted before the load consumes it and the line is
    // re-fetched from DRAM. l2_prefetch stages the line into L2 instead, which
    // is large enough to hold the in-flight window, so a re-touch hits L2
    // rather than causing a second DRAM read. B lives in on-chip L2 SCP
    // already, so it needs no prefetch.
    //
    // Each block is 128B = 2 lines. Prefetch 16 lines (8 blocks) at a time,
    // staying PF_AHEAD blocks in front of the consuming load.
    const int64_t PF_AHEAD = 16;
    for (int64_t kb = 0; kb < n_blocks; kb++) {
        const float * a_ptr = a_row + (kb << 5);   // 32 f32 per block = 128B
        const float * b_ptr = b_col + (kb << 5);
        if ((kb & 7) == 0) {
            const int64_t pf_block = kb + PF_AHEAD;
            if (pf_block + 8 <= n_blocks) {
                l2_prefetch(a_row + (pf_block << 5), 16, 64);   // 16 lines = 8 blocks
            }
        }
        __asm__ volatile(
            // Issue all 8 independent loads first (4 A + 4 B) so several misses
            // are outstanding before any consumer stalls in-order issue.
            "flw.ps   f11, %[a0]\n"
            "flw.ps   f13, %[a1]\n"
            "flw.ps   f15, %[a2]\n"
            "flw.ps   f17, %[a3]\n"
            "flw.ps   f12, %[b0]\n"
            "flw.ps   f14, %[b1]\n"
            "flw.ps   f16, %[b2]\n"
            "flw.ps   f18, %[b3]\n"
            "fmadd.ps f20, f11, f12, f20\n"
            "fmadd.ps f21, f13, f14, f21\n"
            "fmadd.ps f22, f15, f16, f22\n"
            "fmadd.ps f23, f17, f18, f23\n"
            :
            : [a0] "m"(*(const float(*)[8])&a_ptr[0]),
              [a1] "m"(*(const float(*)[8])&a_ptr[8]),
              [a2] "m"(*(const float(*)[8])&a_ptr[16]),
              [a3] "m"(*(const float(*)[8])&a_ptr[24]),
              [b0] "m"(*(const float(*)[8])&b_ptr[0]),
              [b1] "m"(*(const float(*)[8])&b_ptr[8]),
              [b2] "m"(*(const float(*)[8])&b_ptr[16]),
              [b3] "m"(*(const float(*)[8])&b_ptr[24])
            : "f11", "f12", "f13", "f14", "f15", "f16", "f17", "f18",
              "f20", "f21", "f22", "f23"
        );
    }
}

static inline float __attribute__((always_inline))
f32_dot_reduce(void) {
    float result;
    __asm__ __volatile__ (
        // Combine the 4 lane accumulators, then horizontal-sum the 8 lanes.
        "fadd.ps   f20, f20, f21, rne \n\t"
        "fadd.ps   f22, f22, f23, rne \n\t"
        "fadd.ps   f20, f20, f22, rne \n\t"
        "fswizz.ps f1, f20, 0xB1 \n\t"
        "fadd.ps   f2, f20, f1, rne \n\t"
        "fswizz.ps f3, f2, 0x4E \n\t"
        "fadd.ps   f4, f2, f3, rne \n\t"
        "fmvz.x.ps t0, f4, 4 \n\t"
        "fbcx.ps   f5, t0 \n\t"
        "fadd.ps   %[vout], f4, f5, rne \n\t"
        : [vout] "=f" (result)
        :: "t0", "f1", "f2", "f3", "f4", "f5", "f20", "f21", "f22", "f23"
    );
    return result;
}

// Compute dot product between f32 block and f32 column vector
// Vectorized: processes 8 elements at a time using ET vector instructions
// Block size: 16 f32 values (64 bytes = 1 cache line)
static inline float compute_block_dot_product_f32(const float * a_block, const float * b_col_start) {
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

#endif  // BLOCK_OPS_H

static inline void __attribute__((always_inline)) q4_dot_reset(void) {
    __asm__ volatile("fbci.pi f20, 0" ::: "f20");
}

static inline void __attribute__((always_inline)) q4_dot_tile(const block_q4_0 * q_row,
                                                              const float *      b_col,
                                                              int64_t            n_blocks) {
    const int32_t gather_pattern[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    __asm__ volatile("flw.ps f31, %[g]\n" : : [g] "m"(*(const int32_t (*)[8]) gather_pattern) : "f31");

    for (int64_t kb = 0; kb < n_blocks; kb++) {
        const block_q4_0 * blk   = q_row + kb;
        const float *      b_ptr = b_col + (kb << 5);

        __asm__ volatile(
            "fbci.pi     f10, 0\n"

            "fgb.ps      f11, f31(%[a_ptr0])\n"
            "fandi.pi    f12, f11, 15\n"
            "faddi.pi    f12, f12, -8\n"
            "fcvt.ps.pw  f12, f12, rne\n"
            "flw.ps      f13, %[b_low0]\n"
            "fmadd.ps    f10, f12, f13, f10, rne\n"

            "fsrli.pi    f14, f11, 4\n"
            "fandi.pi    f14, f14, 15\n"
            "faddi.pi    f14, f14, -8\n"
            "fcvt.ps.pw  f14, f14, rne\n"
            "flw.ps      f15, %[b_high0]\n"
            "fmadd.ps    f10, f14, f15, f10, rne\n"

            "fgb.ps      f11, f31(%[a_ptr1])\n"
            "fandi.pi    f12, f11, 15\n"
            "faddi.pi    f12, f12, -8\n"
            "fcvt.ps.pw  f12, f12, rne\n"
            "flw.ps      f13, %[b_low1]\n"
            "fmadd.ps    f10, f12, f13, f10, rne\n"

            "fsrli.pi    f14, f11, 4\n"
            "fandi.pi    f14, f14, 15\n"
            "faddi.pi    f14, f14, -8\n"
            "fcvt.ps.pw  f14, f14, rne\n"
            "flw.ps      f15, %[b_high1]\n"
            "fmadd.ps    f10, f14, f15, f10, rne\n"
            :
            : [a_ptr0] "r"(&blk->qs[0]), [b_low0] "m"(*(const float (*)[8]) & b_ptr[0]),
              [b_high0] "m"(*(const float (*)[8]) & b_ptr[16]), [a_ptr1] "r"(&blk->qs[8]),
              [b_low1] "m"(*(const float (*)[8]) & b_ptr[8]), [b_high1] "m"(*(const float (*)[8]) & b_ptr[24])
            : "f10", "f11", "f12", "f13", "f14", "f15");

        uint32_t scale_raw = (uint32_t) blk->d;
        __asm__ volatile(
            "fbcx.ps f15, %[sb]\n"
            "fcvt.ps.f16 f15, f15\n"
            "fmadd.ps f20, f10, f15, f20\n"
            :
            : [sb] "r"(scale_raw)
            : "f15", "f20");
    }
}

static inline float __attribute__((always_inline)) q4_dot_reduce(void) {
    float result;
    __asm__ __volatile__(
        "fswizz.ps f1, f20, 0xB1 \n\t"
        "fadd.ps   f2, f20, f1, rne \n\t"
        "fswizz.ps f3, f2, 0x4E \n\t"
        "fadd.ps   f4, f2, f3, rne \n\t"
        "fmvz.x.ps t0, f4, 4 \n\t"
        "fbcx.ps   f5, t0 \n\t"
        "fadd.ps   %[vout], f4, f5, rne \n\t"
        : [vout] "=f"(result)::"t0", "f1", "f2", "f3", "f4", "f5");
    return result;
}

static inline float compute_row_dot_q4_0(const block_q4_0 * q_row, const float * b_col, int64_t K_blocks) {
    unsigned long saved_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(saved_mask));
    __asm__ volatile("mov.m.x m0, x0, 0xFF");
    q4_dot_reset();
    q4_dot_tile(q_row, b_col, K_blocks);
    float result = q4_dot_reduce();
    __asm__ volatile("mova.m.x %0" ::"r"(saved_mask));
    return result;
}

typedef struct {
    unsigned long saved_mask;
} q4_dot_state;

static inline void q4_dot_begin(q4_dot_state * state) {
    __asm__ volatile("mova.x.m %0" : "=r"(state->saved_mask));
    __asm__ volatile("mov.m.x m0, x0, 0xFF");
}

static inline void q4_dot_end(const q4_dot_state * state) {
    __asm__ volatile("mova.m.x %0" ::"r"(state->saved_mask));
}

static inline float q4_dot_compute(const block_q4_0 * q_row, const float * b_col, int64_t K_blocks) {
    q4_dot_reset();
    q4_dot_tile(q_row, b_col, K_blocks);
    return q4_dot_reduce();
}

static inline void q4_dot_compute_x2_aligned(const block_q4_0 * q_row0,
                                             const block_q4_0 * q_row1,
                                             const float *      b_col,
                                             int64_t            K_blocks,
                                             float *            out0,
                                             float *            out1) {
    const int32_t gather_pattern[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    __asm__ volatile("flw.ps f31, %[g]\n" : : [g] "m"(*(const int32_t (*)[8]) gather_pattern) : "f31");

    __asm__ volatile(
        "fbci.pi f20, 0\n"
        "fbci.pi f21, 0\n" ::
            : "f20", "f21");

    for (int64_t kb = 0; kb < K_blocks; kb++) {
        const block_q4_0 * blk0  = q_row0 + kb;
        const block_q4_0 * blk1  = q_row1 + kb;
        const float *      b_ptr = b_col + (kb << 5);

        __asm__ volatile(
            "fbci.pi     f10, 0\n"
            "fbci.pi     f16, 0\n"

            "flw.ps      f13, %[b_low0]\n"
            "flw.ps      f15, %[b_high0]\n"

            "fgb.ps      f11, f31(%[a_ptr0_0])\n"
            "fgb.ps      f17, f31(%[a_ptr1_0])\n"

            "fandi.pi    f12, f11, 15\n"
            "faddi.pi    f12, f12, -8\n"
            "fcvt.ps.pw  f12, f12, rne\n"
            "fmadd.ps    f10, f12, f13, f10, rne\n"

            "fandi.pi    f18, f17, 15\n"
            "faddi.pi    f18, f18, -8\n"
            "fcvt.ps.pw  f18, f18, rne\n"
            "fmadd.ps    f16, f18, f13, f16, rne\n"

            "fsrli.pi    f14, f11, 4\n"
            "fandi.pi    f14, f14, 15\n"
            "faddi.pi    f14, f14, -8\n"
            "fcvt.ps.pw  f14, f14, rne\n"
            "fmadd.ps    f10, f14, f15, f10, rne\n"

            "fsrli.pi    f19, f17, 4\n"
            "fandi.pi    f19, f19, 15\n"
            "faddi.pi    f19, f19, -8\n"
            "fcvt.ps.pw  f19, f19, rne\n"
            "fmadd.ps    f16, f19, f15, f16, rne\n"

            "flw.ps      f13, %[b_low1]\n"
            "flw.ps      f15, %[b_high1]\n"

            "fgb.ps      f11, f31(%[a_ptr0_1])\n"
            "fgb.ps      f17, f31(%[a_ptr1_1])\n"

            "fandi.pi    f12, f11, 15\n"
            "faddi.pi    f12, f12, -8\n"
            "fcvt.ps.pw  f12, f12, rne\n"
            "fmadd.ps    f10, f12, f13, f10, rne\n"

            "fandi.pi    f18, f17, 15\n"
            "faddi.pi    f18, f18, -8\n"
            "fcvt.ps.pw  f18, f18, rne\n"
            "fmadd.ps    f16, f18, f13, f16, rne\n"

            "fsrli.pi    f14, f11, 4\n"
            "fandi.pi    f14, f14, 15\n"
            "faddi.pi    f14, f14, -8\n"
            "fcvt.ps.pw  f14, f14, rne\n"
            "fmadd.ps    f10, f14, f15, f10, rne\n"

            "fsrli.pi    f19, f17, 4\n"
            "fandi.pi    f19, f19, 15\n"
            "faddi.pi    f19, f19, -8\n"
            "fcvt.ps.pw  f19, f19, rne\n"
            "fmadd.ps    f16, f19, f15, f16, rne\n"
            :
            : [a_ptr0_0] "r"(&blk0->qs[0]), [a_ptr0_1] "r"(&blk0->qs[8]), [a_ptr1_0] "r"(&blk1->qs[0]),
              [a_ptr1_1] "r"(&blk1->qs[8]), [b_low0] "m"(*(const float (*)[8]) & b_ptr[0]),
              [b_high0] "m"(*(const float (*)[8]) & b_ptr[16]), [b_low1] "m"(*(const float (*)[8]) & b_ptr[8]),
              [b_high1] "m"(*(const float (*)[8]) & b_ptr[24])
            : "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17", "f18", "f19");

        const uint32_t scale_raw0 = (uint32_t) blk0->d;
        const uint32_t scale_raw1 = (uint32_t) blk1->d;
        __asm__ volatile(
            "fbcx.ps     f24, %[s0]\n"
            "fcvt.ps.f16 f24, f24\n"
            "fmadd.ps    f20, f10, f24, f20\n"
            "fbcx.ps     f25, %[s1]\n"
            "fcvt.ps.f16 f25, f25\n"
            "fmadd.ps    f21, f16, f25, f21\n"
            :
            : [s0] "r"(scale_raw0), [s1] "r"(scale_raw1)
            : "f20", "f21", "f24", "f25");
    }

    float result0, result1;
    __asm__ __volatile__(
        "fswizz.ps f1, f20, 0xB1 \n\t"
        "fadd.ps   f2, f20, f1, rne \n\t"
        "fswizz.ps f3, f2, 0x4E \n\t"
        "fadd.ps   f4, f2, f3, rne \n\t"
        "fmvz.x.ps t0, f4, 4 \n\t"
        "fbcx.ps   f5, t0 \n\t"
        "fadd.ps   %[vout], f4, f5, rne \n\t"
        : [vout] "=f"(result0)::"t0", "f1", "f2", "f3", "f4", "f5");
    __asm__ __volatile__(
        "fswizz.ps f1, f21, 0xB1 \n\t"
        "fadd.ps   f2, f21, f1, rne \n\t"
        "fswizz.ps f3, f2, 0x4E \n\t"
        "fadd.ps   f4, f2, f3, rne \n\t"
        "fmvz.x.ps t0, f4, 4 \n\t"
        "fbcx.ps   f5, t0 \n\t"
        "fadd.ps   %[vout], f4, f5, rne \n\t"
        : [vout] "=f"(result1)::"t0", "f1", "f2", "f3", "f4", "f5");

    *out0 = result0;
    *out1 = result1;
}


// Full-row dot product for Q4_K weights against an F32 activation column.
//
// Unlike Q4_0/Q8_0 (whose dequant is a pure per-block scale, so the scale can
// be factored out of the dot product), Q4_K reconstructs each weight via an
// affine transform `w = d*scale*q - dmin*min` with per-group scales/mins inside
// each 256-element super-block. That makes the cheap "scale the integer dot"
// trick inapplicable.
//
// The dequant math mirrors dequantize_q4_K_block exactly, but the per-element
// product is folded straight into a scalar accumulator instead of being staged
// through a temporary buffer. This deliberately avoids a large (1KB) on-stack
// dequant buffer and the vector-mask save/restore of the F32 dot helper, both
// of which are unsafe in the uberkernel context.
//
// K_sblocks is the number of QK_K (256) element super-blocks in the row
// (i.e. K / QK_K).
static inline float sw_fp16_to_fp32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) { f = sign; }
        else {
            exp = 127 - 15 + 1;
            while ((mant & 0x400) == 0) { mant <<= 1; exp--; }
            mant &= 0x3FF;
            f = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float out; __builtin_memcpy(&out, &f, 4); return out;
}

static inline float compute_row_dot_q4_K(const block_q4_K* q_row,
                                         const float* b_col,
                                         int64_t K_sblocks) {
    float acc = 0.0f;
    for (int64_t sb = 0; sb < K_sblocks; sb++) {
        const block_q4_K* block = q_row + sb;
        const float* b = b_col + sb * QK_K;
        const uint8_t* q = block->qs;
        const float d   = sw_fp16_to_fp32(block->d);
        const float min = sw_fp16_to_fp32(block->dmin);

        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, block->scales, &sc, &m);
            const float d1 = d * sc;
            const float m1 = min * m;
            get_scale_min_k4(is + 1, block->scales, &sc, &m);
            const float d2 = d * sc;
            const float m2 = min * m;
            for (int l = 0; l < 32; ++l) {
                acc += (d1 * (float)(q[l] & 0xF) - m1) * (*b++);
            }
            for (int l = 0; l < 32; ++l) {
                acc += (d2 * (float)(q[l] >> 4) - m2) * (*b++);
            }
            q += 32;
            is += 2;
        }
    }
    return acc;
}

// Vectorized (8-wide) full-row dot for Q4_K. Affine 8-groups-of-32 layout
// w = d*sc*nibble - dmin*m (no qh bit). Group g -> qs pair g/2, nibble
// low(g even)/high(g odd). f10 scale term, f9 min term; result = f10 - f9.
#define Q4V_LO "fandi.pi f12, f11, 15\n\t"
#define Q4V_HI "fsrli.pi f12, f11, 4\n\t fandi.pi f12, f12, 15\n\t"
#define Q4V_CHUNK(NIB, qlp, bp, dscb, dmb)                               \
    __asm__ volatile(                                                    \
        "fgb.ps     f11, f31(%[q])\n\t"                                  \
        NIB                                                              \
        "fcvt.ps.pw f12, f12, rne\n\t"                                   \
        "fbcx.ps    f16, %[dsc]\n\t"                                     \
        "flw.ps     f15, 0(%[b])\n\t"                                    \
        "fmul.ps    f12, f12, f16\n\t"                                   \
        "fmadd.ps   f10, f12, f15, f10\n\t"                              \
        "fbcx.ps    f17, %[dm]\n\t"                                      \
        "fmadd.ps   f9, f15, f17, f9\n\t"                                \
        :: [q] "r"(qlp), [b] "r"(bp), [dsc] "r"(dscb), [dm] "r"(dmb)     \
        : "f11", "f12", "f15", "f16", "f17")
#define Q4V_GROUP(NIB, pp)                                               \
    do {                                                                 \
        const uint8_t* qlb = block->qs + (pp) * 32;                      \
        Q4V_CHUNK(NIB, qlb + 0,  bg + 0,  dscb, dmb);                    \
        Q4V_CHUNK(NIB, qlb + 8,  bg + 8,  dscb, dmb);                    \
        Q4V_CHUNK(NIB, qlb + 16, bg + 16, dscb, dmb);                    \
        Q4V_CHUNK(NIB, qlb + 24, bg + 24, dscb, dmb);                    \
    } while (0)

static inline float compute_row_dot_q4_K_vec(const block_q4_K* q_row,
                                             const float* b_col,
                                             int64_t K_sblocks) {
    unsigned long saved_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(saved_mask));
    __asm__ volatile("mov.m.x m0, x0, 0xFF");

    static const int32_t __attribute__((aligned(64))) gp[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    __asm__ volatile("flw.ps f31, %[g]\n\t"
                     "fbci.ps f10, 0\n\t"
                     "fbci.ps f9, 0\n\t"
                     :: [g] "m"(*(const int32_t(*)[8]) gp)
                     : "f31", "f10", "f9");

    for (int64_t sb = 0; sb < K_sblocks; sb++) {
        const block_q4_K* block = q_row + sb;
        const float* b = b_col + sb * QK_K;
        const float d   = sw_fp16_to_fp32(block->d);
        const float min = sw_fp16_to_fp32(block->dmin);

        for (int g = 0; g < 8; ++g) {
            uint8_t sc, m;
            get_scale_min_k4(g, block->scales, &sc, &m);
            const float dscf = d * (float) sc;
            const float dmf  = min * (float) m;
            uint32_t dscb, dmb;
            __builtin_memcpy(&dscb, &dscf, 4);
            __builtin_memcpy(&dmb, &dmf, 4);
            const float* bg = b + g * 32;
            const int p = g >> 1;

            if (g & 1) {
                Q4V_GROUP(Q4V_HI, p);
            } else {
                Q4V_GROUP(Q4V_LO, p);
            }
        }
    }

    float final_sum;
    __asm__ volatile(
        "fswizz.ps f1, f10, 0xB1 \n\t fadd.ps f2, f10, f1, rne \n\t"
        "fswizz.ps f3, f2, 0x4E \n\t fadd.ps f4, f2, f3, rne \n\t"
        "fmvz.x.ps t0, f4, 4 \n\t fbcx.ps f5, t0 \n\t fadd.ps f6, f4, f5, rne \n\t"
        "fswizz.ps f1, f9, 0xB1 \n\t fadd.ps f2, f9, f1, rne \n\t"
        "fswizz.ps f3, f2, 0x4E \n\t fadd.ps f4, f2, f3, rne \n\t"
        "fmvz.x.ps t0, f4, 4 \n\t fbcx.ps f5, t0 \n\t fadd.ps f7, f4, f5, rne \n\t"
        "fsub.ps   %[out], f6, f7, rne \n\t"
        : [out] "=f"(final_sum)
        :: "t0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f9", "f10");

    __asm__ volatile("mova.m.x %0" ::"r"(saved_mask));
    return final_sum;
}

// Vectorized (8-wide) full-row dot for Q6_K, modeled on the q4_0/q8_0 vector
// dots. Each 32-element group of contiguous output elements maps to contiguous
// ql/qh bytes and a fixed nibble/shift, so a group is 4 chunks of 8 lanes.
// The per-group factor d*scale is folded into the weight vector (fmul), so a
// single fp32 accumulator (f10) covers the whole row and is reduced once at the
// end. f30 holds -32.0 (the Q6_K zero point), f31 the byte-gather pattern.
#define Q6V_NIB_LO "fandi.pi f12, f11, 15\n\t"
#define Q6V_NIB_HI "fsrli.pi f12, f11, 4\n\t fandi.pi f12, f12, 15\n\t"
#define Q6V_CHUNK(NIB, SH, qlp, qhp, bp, facbits)                        \
    __asm__ volatile(                                                    \
        "fgb.ps     f11, f31(%[q])\n\t"                                  \
        "fgb.ps     f13, f31(%[h])\n\t"                                  \
        NIB                                                              \
        "fsrli.pi   f13, f13, " #SH "\n\t"                               \
        "fandi.pi   f13, f13, 3\n\t"                                     \
        "fslli.pi   f13, f13, 4\n\t"                                     \
        "fcvt.ps.pw f12, f12, rne\n\t"                                   \
        "fcvt.ps.pw f13, f13, rne\n\t"                                   \
        "fadd.ps    f12, f12, f13, rne\n\t"                              \
        "fadd.ps    f12, f12, f30, rne\n\t"                              \
        "fbcx.ps    f16, %[f]\n\t"                                       \
        "flw.ps     f15, 0(%[b])\n\t"                                    \
        "fmul.ps    f12, f12, f16\n\t"                                   \
        "fmadd.ps   f10, f12, f15, f10\n\t"                              \
        :: [q] "r"(qlp), [h] "r"(qhp), [b] "r"(bp), [f] "r"(facbits)     \
        : "f11", "f12", "f13", "f15", "f16")

static inline float compute_row_dot_q6_K_vec(const block_q6_K* q_row,
                                             const float* b_col,
                                             int64_t K_sblocks) {
    unsigned long saved_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(saved_mask));
    __asm__ volatile("mov.m.x m0, x0, 0xFF");

    static const int32_t __attribute__((aligned(64))) gp[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    const uint32_t neg32 = 0xC2000000u;  // -32.0f
    __asm__ volatile("flw.ps f31, %[g]\n\t"
                     "fbcx.ps f30, %[n]\n\t"
                     "fbci.ps f10, 0\n\t"
                     :: [g] "m"(*(const int32_t(*)[8]) gp), [n] "r"(neg32)
                     : "f31", "f30", "f10");

    for (int64_t sb = 0; sb < K_sblocks; sb++) {
        const block_q6_K* block = q_row + sb;
        const float* b = b_col + sb * QK_K;
        const float d = sw_fp16_to_fp32(block->d);

        for (int g = 0; g < 8; ++g) {
            const int c   = g >> 2;
            const int sub = g & 3;
            const uint8_t* qlbase = block->ql + 64 * c + (sub & 1) * 32;
            const uint8_t* qhbase = block->qh + 32 * c;
            const float* bg = b + g * 32;
            const float f0f = d * (float) block->scales[8 * c + 2 * sub + 0];
            const float f1f = d * (float) block->scales[8 * c + 2 * sub + 1];
            uint32_t f0, f1;
            __builtin_memcpy(&f0, &f0f, 4);
            __builtin_memcpy(&f1, &f1f, 4);

            switch (sub) {
                case 0:
                    Q6V_CHUNK(Q6V_NIB_LO, 0, qlbase + 0,  qhbase + 0,  bg + 0,  f0);
                    Q6V_CHUNK(Q6V_NIB_LO, 0, qlbase + 8,  qhbase + 8,  bg + 8,  f0);
                    Q6V_CHUNK(Q6V_NIB_LO, 0, qlbase + 16, qhbase + 16, bg + 16, f1);
                    Q6V_CHUNK(Q6V_NIB_LO, 0, qlbase + 24, qhbase + 24, bg + 24, f1);
                    break;
                case 1:
                    Q6V_CHUNK(Q6V_NIB_LO, 2, qlbase + 0,  qhbase + 0,  bg + 0,  f0);
                    Q6V_CHUNK(Q6V_NIB_LO, 2, qlbase + 8,  qhbase + 8,  bg + 8,  f0);
                    Q6V_CHUNK(Q6V_NIB_LO, 2, qlbase + 16, qhbase + 16, bg + 16, f1);
                    Q6V_CHUNK(Q6V_NIB_LO, 2, qlbase + 24, qhbase + 24, bg + 24, f1);
                    break;
                case 2:
                    Q6V_CHUNK(Q6V_NIB_HI, 4, qlbase + 0,  qhbase + 0,  bg + 0,  f0);
                    Q6V_CHUNK(Q6V_NIB_HI, 4, qlbase + 8,  qhbase + 8,  bg + 8,  f0);
                    Q6V_CHUNK(Q6V_NIB_HI, 4, qlbase + 16, qhbase + 16, bg + 16, f1);
                    Q6V_CHUNK(Q6V_NIB_HI, 4, qlbase + 24, qhbase + 24, bg + 24, f1);
                    break;
                default:
                    Q6V_CHUNK(Q6V_NIB_HI, 6, qlbase + 0,  qhbase + 0,  bg + 0,  f0);
                    Q6V_CHUNK(Q6V_NIB_HI, 6, qlbase + 8,  qhbase + 8,  bg + 8,  f0);
                    Q6V_CHUNK(Q6V_NIB_HI, 6, qlbase + 16, qhbase + 16, bg + 16, f1);
                    Q6V_CHUNK(Q6V_NIB_HI, 6, qlbase + 24, qhbase + 24, bg + 24, f1);
                    break;
            }
        }
    }

    float final_sum;
    __asm__ volatile(
        "fswizz.ps f1, f10, 0xB1 \n\t"
        "fadd.ps   f2, f10, f1, rne \n\t"
        "fswizz.ps f3, f2, 0x4E \n\t"
        "fadd.ps   f4, f2, f3, rne \n\t"
        "fmvz.x.ps t0, f4, 4 \n\t"
        "fbcx.ps   f5, t0 \n\t"
        "fadd.ps   %[out], f4, f5, rne \n\t"
        : [out] "=f"(final_sum)
        :: "t0", "f1", "f2", "f3", "f4", "f5", "f10");

    __asm__ volatile("mova.m.x %0" ::"r"(saved_mask));
    return final_sum;
}

// NOTE: two further Q6_K dot experiments were tried and REMOVED as neither
// helped - the generation dot is memory/overhead bound, not compute bound:
//   * gather-reduced (ql/qh gathered once, reused across sub-groups, 64->24
//     fgb/super-block) - perf-neutral (14.95 vs 14.93 t/s).
//   * 4 rotating accumulators to pipeline the fmadd chain - also neutral (15.02).

// Vectorized (8-wide) full-row dot for Q5_K. Same 8-groups-of-32 affine layout
// as Q4_K plus a 5th bit from qh: w = d*sc*(nibble + qh_bit*16) - dmin*m.
// Group g -> qs pair g/2, nibble low(g even)/high(g odd), qh bit at position g.
// f10 accumulates the scale term, f9 the min term (sum of b*dmin*m per group);
// result = reduce(f10) - reduce(f9).
#define Q5V_LO "fandi.pi f12, f11, 15\n\t"
#define Q5V_HI "fsrli.pi f12, f11, 4\n\t fandi.pi f12, f12, 15\n\t"
#define Q5V_CHUNK(NIB, BP, qlp, qhp, bp, dscb, dmb)                      \
    __asm__ volatile(                                                    \
        "fgb.ps     f11, f31(%[q])\n\t"                                  \
        "fgb.ps     f13, f31(%[h])\n\t"                                  \
        NIB                                                              \
        "fsrli.pi   f13, f13, " #BP "\n\t"                               \
        "fandi.pi   f13, f13, 1\n\t"                                     \
        "fslli.pi   f13, f13, 4\n\t"                                     \
        "fcvt.ps.pw f12, f12, rne\n\t"                                   \
        "fcvt.ps.pw f13, f13, rne\n\t"                                   \
        "fadd.ps    f12, f12, f13, rne\n\t"                              \
        "fbcx.ps    f16, %[dsc]\n\t"                                     \
        "flw.ps     f15, 0(%[b])\n\t"                                    \
        "fmul.ps    f12, f12, f16\n\t"                                   \
        "fmadd.ps   f10, f12, f15, f10\n\t"                              \
        "fbcx.ps    f17, %[dm]\n\t"                                      \
        "fmadd.ps   f9, f15, f17, f9\n\t"                                \
        :: [q] "r"(qlp), [h] "r"(qhp), [b] "r"(bp), [dsc] "r"(dscb), [dm] "r"(dmb) \
        : "f11", "f12", "f13", "f15", "f16", "f17")
#define Q5V_GROUP(NIB, BP, pp)                                           \
    do {                                                                 \
        const uint8_t* qlb = block->qs + (pp) * 32;                      \
        const uint8_t* qhb = block->qh;                                  \
        Q5V_CHUNK(NIB, BP, qlb + 0,  qhb + 0,  bg + 0,  dscb, dmb);      \
        Q5V_CHUNK(NIB, BP, qlb + 8,  qhb + 8,  bg + 8,  dscb, dmb);      \
        Q5V_CHUNK(NIB, BP, qlb + 16, qhb + 16, bg + 16, dscb, dmb);      \
        Q5V_CHUNK(NIB, BP, qlb + 24, qhb + 24, bg + 24, dscb, dmb);      \
    } while (0)

static inline float compute_row_dot_q5_K_vec(const block_q5_K* q_row,
                                             const float* b_col,
                                             int64_t K_sblocks) {
    unsigned long saved_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(saved_mask));
    __asm__ volatile("mov.m.x m0, x0, 0xFF");

    static const int32_t __attribute__((aligned(64))) gp[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    __asm__ volatile("flw.ps f31, %[g]\n\t"
                     "fbci.ps f10, 0\n\t"
                     "fbci.ps f9, 0\n\t"
                     :: [g] "m"(*(const int32_t(*)[8]) gp)
                     : "f31", "f10", "f9");

    for (int64_t sb = 0; sb < K_sblocks; sb++) {
        const block_q5_K* block = q_row + sb;
        const float* b = b_col + sb * QK_K;
        const float d   = sw_fp16_to_fp32(block->d);
        const float min = sw_fp16_to_fp32(block->dmin);

        for (int g = 0; g < 8; ++g) {
            uint8_t sc, m;
            get_scale_min_k4(g, block->scales, &sc, &m);
            const float dscf = d * (float) sc;
            const float dmf  = min * (float) m;
            uint32_t dscb, dmb;
            __builtin_memcpy(&dscb, &dscf, 4);
            __builtin_memcpy(&dmb, &dmf, 4);
            const float* bg = b + g * 32;
            const int p = g >> 1;

            switch (g) {
                case 0: Q5V_GROUP(Q5V_LO, 0, p); break;
                case 1: Q5V_GROUP(Q5V_HI, 1, p); break;
                case 2: Q5V_GROUP(Q5V_LO, 2, p); break;
                case 3: Q5V_GROUP(Q5V_HI, 3, p); break;
                case 4: Q5V_GROUP(Q5V_LO, 4, p); break;
                case 5: Q5V_GROUP(Q5V_HI, 5, p); break;
                case 6: Q5V_GROUP(Q5V_LO, 6, p); break;
                default: Q5V_GROUP(Q5V_HI, 7, p); break;
            }
        }
    }

    float final_sum;
    __asm__ volatile(
        "fswizz.ps f1, f10, 0xB1 \n\t fadd.ps f2, f10, f1, rne \n\t"
        "fswizz.ps f3, f2, 0x4E \n\t fadd.ps f4, f2, f3, rne \n\t"
        "fmvz.x.ps t0, f4, 4 \n\t fbcx.ps f5, t0 \n\t fadd.ps f6, f4, f5, rne \n\t"
        "fswizz.ps f1, f9, 0xB1 \n\t fadd.ps f2, f9, f1, rne \n\t"
        "fswizz.ps f3, f2, 0x4E \n\t fadd.ps f4, f2, f3, rne \n\t"
        "fmvz.x.ps t0, f4, 4 \n\t fbcx.ps f5, t0 \n\t fadd.ps f7, f4, f5, rne \n\t"
        "fsub.ps   %[out], f6, f7, rne \n\t"
        : [out] "=f"(final_sum)
        :: "t0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f9", "f10");

    __asm__ volatile("mova.m.x %0" ::"r"(saved_mask));
    return final_sum;
}

// Vectorized (8-wide) full-row dot for Q2_K. Affine 2-bit
// w = d*(sc&0xF)*q - dmin*(sc>>4), 16 groups of 16. Sub-group sg -> chunk
// c=sg/8, shift 2*((sg&7)/2), A/B nibble-set ab=(sg&7)&1 selecting qs/scale.
// f10 scale term, f9 min term; result = reduce(f10) - reduce(f9).
#define Q2V_CHUNK(SH, qp, bp, dlb, mlb)                                  \
    __asm__ volatile(                                                    \
        "fgb.ps     f11, f31(%[q])\n\t"                                  \
        "fsrli.pi   f12, f11, " #SH "\n\t"                               \
        "fandi.pi   f12, f12, 3\n\t"                                     \
        "fcvt.ps.pw f12, f12, rne\n\t"                                   \
        "fbcx.ps    f16, %[dl]\n\t"                                      \
        "flw.ps     f15, 0(%[b])\n\t"                                    \
        "fmul.ps    f12, f12, f16\n\t"                                   \
        "fmadd.ps   f10, f12, f15, f10\n\t"                              \
        "fbcx.ps    f17, %[ml]\n\t"                                      \
        "fmadd.ps   f9, f15, f17, f9\n\t"                                \
        :: [q] "r"(qp), [b] "r"(bp), [dl] "r"(dlb), [ml] "r"(mlb)        \
        : "f11", "f12", "f15", "f16", "f17")
#define Q2V_GROUP(SH, qb, bg)                                            \
    do {                                                                 \
        Q2V_CHUNK(SH, (qb) + 0, (bg) + 0, dlb, mlb);                     \
        Q2V_CHUNK(SH, (qb) + 8, (bg) + 8, dlb, mlb);                     \
    } while (0)

static inline float compute_row_dot_q2_K_vec(const block_q2_K* q_row,
                                             const float* b_col,
                                             int64_t K_sblocks) {
    unsigned long saved_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(saved_mask));
    __asm__ volatile("mov.m.x m0, x0, 0xFF");

    static const int32_t __attribute__((aligned(64))) gp[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    __asm__ volatile("flw.ps f31, %[g]\n\t"
                     "fbci.ps f10, 0\n\t"
                     "fbci.ps f9, 0\n\t"
                     :: [g] "m"(*(const int32_t(*)[8]) gp)
                     : "f31", "f10", "f9");

    for (int64_t sb = 0; sb < K_sblocks; sb++) {
        const block_q2_K* block = q_row + sb;
        const float* b = b_col + sb * QK_K;
        const float d   = sw_fp16_to_fp32(block->d);
        const float min = sw_fp16_to_fp32(block->dmin);

        for (int sg = 0; sg < 16; ++sg) {
            const int c   = sg >> 3;
            const int sgi = sg & 7;
            const int j   = sgi >> 1;
            const int ab  = sgi & 1;
            const uint8_t sc = block->scales[c * 8 + 2 * j + ab];
            const float dlf = d   * (float) (sc & 0xF);
            const float mlf = min * (float) (sc >> 4);
            uint32_t dlb, mlb;
            __builtin_memcpy(&dlb, &dlf, 4);
            __builtin_memcpy(&mlb, &mlf, 4);
            const uint8_t* qb = block->qs + c * 32 + ab * 16;
            const float* bg = b + sg * 16;

            switch (j) {
                case 0: Q2V_GROUP(0, qb, bg); break;
                case 1: Q2V_GROUP(2, qb, bg); break;
                case 2: Q2V_GROUP(4, qb, bg); break;
                default: Q2V_GROUP(6, qb, bg); break;
            }
        }
    }

    float final_sum;
    __asm__ volatile(
        "fswizz.ps f1, f10, 0xB1 \n\t fadd.ps f2, f10, f1, rne \n\t"
        "fswizz.ps f3, f2, 0x4E \n\t fadd.ps f4, f2, f3, rne \n\t"
        "fmvz.x.ps t0, f4, 4 \n\t fbcx.ps f5, t0 \n\t fadd.ps f6, f4, f5, rne \n\t"
        "fswizz.ps f1, f9, 0xB1 \n\t fadd.ps f2, f9, f1, rne \n\t"
        "fswizz.ps f3, f2, 0x4E \n\t fadd.ps f4, f2, f3, rne \n\t"
        "fmvz.x.ps t0, f4, 4 \n\t fbcx.ps f5, t0 \n\t fadd.ps f7, f4, f5, rne \n\t"
        "fsub.ps   %[out], f6, f7, rne \n\t"
        : [out] "=f"(final_sum)
        :: "t0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f9", "f10");

    __asm__ volatile("mova.m.x %0" ::"r"(saved_mask));
    return final_sum;
}

// Vectorized (8-wide) full-row dot for Q3_K. 3-bit
// w = d*(scale-32)*(q2 + hbit*4 - 4), 16 groups of 16, no min term. Sub-group
// sg -> c=sg/8, j=(sg&7)/2 (qs shift 2*j), ab=(sg&7)&1 (qs/hmask offset),
// hmask bit at position c*4+j. f30 holds -4.0.
#define Q3V_CHUNK(SH, BP, qp, hp, bp, dlb)                               \
    __asm__ volatile(                                                    \
        "fgb.ps     f11, f31(%[q])\n\t"                                  \
        "fgb.ps     f13, f31(%[h])\n\t"                                  \
        "fsrli.pi   f12, f11, " #SH "\n\t fandi.pi f12, f12, 3\n\t"      \
        "fsrli.pi   f13, f13, " #BP "\n\t fandi.pi f13, f13, 1\n\t fslli.pi f13, f13, 2\n\t" \
        "fcvt.ps.pw f12, f12, rne\n\t"                                   \
        "fcvt.ps.pw f13, f13, rne\n\t"                                   \
        "fadd.ps    f12, f12, f13, rne\n\t"                              \
        "fadd.ps    f12, f12, f30, rne\n\t"                              \
        "fbcx.ps    f16, %[dl]\n\t"                                      \
        "flw.ps     f15, 0(%[b])\n\t"                                    \
        "fmul.ps    f12, f12, f16\n\t"                                   \
        "fmadd.ps   f10, f12, f15, f10\n\t"                              \
        :: [q] "r"(qp), [h] "r"(hp), [b] "r"(bp), [dl] "r"(dlb)          \
        : "f11", "f12", "f13", "f15", "f16")
#define Q3V_GROUP(SH, BP, qb, hb, bg)                                    \
    do {                                                                 \
        Q3V_CHUNK(SH, BP, (qb) + 0, (hb) + 0, (bg) + 0, dlb);            \
        Q3V_CHUNK(SH, BP, (qb) + 8, (hb) + 8, (bg) + 8, dlb);            \
    } while (0)

static inline float compute_row_dot_q3_K_vec(const block_q3_K* q_row,
                                             const float* b_col,
                                             int64_t K_sblocks) {
    unsigned long saved_mask;
    __asm__ volatile("mova.x.m %0" : "=r"(saved_mask));
    __asm__ volatile("mov.m.x m0, x0, 0xFF");

    static const int32_t __attribute__((aligned(64))) gp[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    const uint32_t neg4 = 0xC0800000u;  // -4.0f
    __asm__ volatile("flw.ps f31, %[g]\n\t"
                     "fbcx.ps f30, %[n]\n\t"
                     "fbci.ps f10, 0\n\t"
                     :: [g] "m"(*(const int32_t(*)[8]) gp), [n] "r"(neg4)
                     : "f31", "f30", "f10");

    for (int64_t sb = 0; sb < K_sblocks; sb++) {
        const block_q3_K* block = q_row + sb;
        const float* b = b_col + sb * QK_K;
        const float d = sw_fp16_to_fp32(block->d);
        int8_t scales[16];
        unpack_q3_K_scales(block->scales, scales);

        for (int sg = 0; sg < 16; ++sg) {
            const int c   = sg >> 3;
            const int sgi = sg & 7;
            const int j   = sgi >> 1;
            const int ab  = sgi & 1;
            const float dlf = d * (float) (scales[c * 8 + 2 * j + ab] - 32);
            uint32_t dlb;
            __builtin_memcpy(&dlb, &dlf, 4);
            const uint8_t* qb = block->qs + c * 32 + ab * 16;
            const uint8_t* hb = block->hmask + ab * 16;
            const float* bg = b + sg * 16;
            const int cj = c * 4 + j;  // hmask bit position

            switch (cj) {
                case 0: Q3V_GROUP(0, 0, qb, hb, bg); break;
                case 1: Q3V_GROUP(2, 1, qb, hb, bg); break;
                case 2: Q3V_GROUP(4, 2, qb, hb, bg); break;
                case 3: Q3V_GROUP(6, 3, qb, hb, bg); break;
                case 4: Q3V_GROUP(0, 4, qb, hb, bg); break;
                case 5: Q3V_GROUP(2, 5, qb, hb, bg); break;
                case 6: Q3V_GROUP(4, 6, qb, hb, bg); break;
                default: Q3V_GROUP(6, 7, qb, hb, bg); break;
            }
        }
    }

    float final_sum;
    __asm__ volatile(
        "fswizz.ps f1, f10, 0xB1 \n\t fadd.ps f2, f10, f1, rne \n\t"
        "fswizz.ps f3, f2, 0x4E \n\t fadd.ps f4, f2, f3, rne \n\t"
        "fmvz.x.ps t0, f4, 4 \n\t fbcx.ps f5, t0 \n\t fadd.ps %[out], f4, f5, rne \n\t"
        : [out] "=f"(final_sum)
        :: "t0", "f1", "f2", "f3", "f4", "f5", "f10");

    __asm__ volatile("mova.m.x %0" ::"r"(saved_mask));
    return final_sum;
}
