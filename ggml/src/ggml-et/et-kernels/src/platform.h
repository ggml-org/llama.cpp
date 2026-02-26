//******************************************************************************
// ET Platform Hardware Abstraction Layer
// Provides thread coordination, kernel infrastructure, and platform primitives
// for bare metal ET kernels
//******************************************************************************

#ifndef PLATFORM_H
#define PLATFORM_H

#include <stdint.h>

#define SOC_MINIONS_PER_SHIRE 32
#define NUM_HARTS_PER_MINION 2

// Environment structure definition
typedef struct {
    uint32_t version;           // Version of the ABI (offset 0)
    uint32_t padding1;          // Padding to align shire_mask to offset 8
    uint64_t shire_mask;        // Bitmask of active compute shires (offset 8)
    uint32_t frequency;         // Frequency of Minion cores in MHz (offset 16)
    uint32_t padding2;          // Padding to maintain alignment
} __attribute__((packed, aligned(64))) kernel_environment_t;

// Get absolute hart ID using inline assembly
static inline uint64_t get_hart_id(void) {
    uint64_t hart_id;
    __asm__ volatile("csrr %0, hartid" : "=r"(hart_id));
    return hart_id;
}

// Manual implementation of count trailing zeros for bare metal environment
// NOTE: This simple loop-based implementation is used for portability.
// Production implementations (like libgcc's __ctzdi2) use optimized bit manipulation
// algorithms with lookup tables and parallel bit operations for O(log n) performance.
static inline int manual_ctzll(uint64_t x) {
    if (x == 0) return 64;
    int count = 0;
    while ((x & 1) == 0) {
        x >>= 1;
        count++;
    }
    return count;
}

// Manual implementation of population count for bare metal environment
// NOTE: This simple loop-based implementation is used for portability.
// Production implementations (like libgcc's __popcountdi2) use optimized bit-parallel
// algorithms with magic constants and bit manipulation tricks for O(1) performance.
static inline int manual_popcountll(uint64_t x) {
    int count = 0;
    while (x) {
        count += x & 1;
        x >>= 1;
    }
    return count;
}

// Calculate relative thread ID from absolute hart ID using shire mask
// Returns -1 if this hart is not active (not in shire mask)
static inline int get_relative_thread_id(uint64_t shire_mask) {
    int hart_id = (int)get_hart_id();

    // Find starting hart offset from lowest active shire
    int starting_hart = manual_ctzll(shire_mask) * SOC_MINIONS_PER_SHIRE * NUM_HARTS_PER_MINION;

    // Return -1 if not an active thread
    if (hart_id < starting_hart) {
        return -1;
    }

    // Calculate relative thread ID
    int thread_id = hart_id - starting_hart;
    return thread_id;
}

// Calculate total number of threads from shire mask
static inline int get_num_threads(uint64_t shire_mask) {
    // Count active shires using popcount, multiply by minions per shire and harts per minion
    return manual_popcountll(shire_mask) * SOC_MINIONS_PER_SHIRE * NUM_HARTS_PER_MINION;
}

//******************************************************************************
// Atomic Operations
//******************************************************************************

// Atomic store for F32 values to global memory
// Uses ET hardware's custom amoswapg.w instruction for global atomic swap
// This ensures cache coherency when multiple threads write to nearby addresses
static inline void atomic_store_f32(volatile float* addr, float value) {
    uint32_t value_bits = *(uint32_t*)&value;
    __asm__ volatile(
        "amoswapg.w zero, %1, (%0)"
        :
        : "r"(addr), "r"(value_bits)
        : "memory"
    );
}

// Atomic store for F16 values to global memory
// Uses ET hardware's custom shg instruction (store halfword global)
// This ensures cache coherency when multiple threads write to nearby addresses
// Address must be 16-bit aligned
static inline void atomic_store_f16(volatile uint16_t* addr, uint16_t value) {
    __asm__ volatile(
        "shg %1, (%0)"
        :
        : "r"(addr), "r"(value)
        : "memory"
    );
}

#endif // PLATFORM_H
