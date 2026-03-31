//******************************************************************************
// ET Platform Hardware Abstraction Layer
// Provides thread coordination, kernel infrastructure, and platform primitives
// for bare metal ET kernels
//******************************************************************************

#ifndef PLATFORM_H
#define PLATFORM_H

#include <stdint.h>
#include "etsoc/isa/hart.h"
#include "etsoc/common/utils.h"
#include "etsoc/isa/barriers.h"

#define SOC_MINIONS_PER_SHIRE 32
#define NUM_HARTS_PER_MINION 2
#define ET_CACHE_LINE_SIZE_BYTES 64

// Environment structure definition
typedef struct {
    uint32_t version;           // Version of the ABI (offset 0)
    uint32_t padding1;          // Padding to align shire_mask to offset 8
    uint64_t shire_mask;        // Bitmask of active compute shires (offset 8)
    uint32_t frequency;         // Frequency of Minion cores in MHz (offset 16)
    uint32_t padding2;          // Padding to maintain alignment
} __attribute__((packed, aligned(64))) kernel_environment_t;

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

static inline int64_t et_gcd_i64(int64_t a, int64_t b) {
    while (b) {
        const int64_t t = b;
        b = a % b;
        a = t;
    }
    return a;
}

// Return the number of consecutive rows of width row_elems needed so the
// combined write footprint spans an integer number of cache lines.
static inline int64_t et_rows_per_cacheline_group(int64_t row_elems, int64_t elem_size_bytes) {
    if (row_elems <= 0 || elem_size_bytes <= 0) {
        return 1;
    }

    const int64_t row_bytes = row_elems * elem_size_bytes;
    const int64_t gcd = et_gcd_i64(ET_CACHE_LINE_SIZE_BYTES, row_bytes);
    return ET_CACHE_LINE_SIZE_BYTES / gcd;
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
// Synchronization Primitives
//******************************************************************************

#define NOP   __asm__ __volatile__ ("nop\n");
#define FENCE __asm__ __volatile__ ("fence\n" ::: "memory");
#define WFI   __asm__ __volatile__ ("wfi\n");

//******************************************************************************
// Atomic Operations
//******************************************************************************

// Global AMO primitives — ET custom 'g' suffix instructions that go through
// the NoC coherence fabric for chip-wide atomicity.

// Atomic swap (word), returns previous value.
static inline uint32_t __attribute__((always_inline))
et_global_swap_w(volatile void *addr, uint32_t val)
{
    uint32_t ret;
    __asm__ __volatile__(
        "amoswapg.w %0, %1, (%2)"
        : "=r"(ret) : "r"(val), "r"(addr) : "memory"
    );
    return ret;
}

// Atomic add (word), returns previous value.
static inline uint32_t __attribute__((always_inline))
et_global_add_w(volatile void *addr, uint32_t val)
{
    uint32_t ret;
    __asm__ __volatile__(
        "amoaddg.w %0, %1, (%2)"
        : "=r"(ret) : "r"(val), "r"(addr) : "memory"
    );
    return ret;
}

// Atomic store (halfword, global). Address must be 16-bit aligned.
static inline void __attribute__((always_inline))
et_global_store_hw(volatile void *addr, uint16_t val)
{
    __asm__ __volatile__(
        "shg %0, (%1)"
        : : "r"(val), "r"(addr) : "memory"
    );
}

// Convenience wrappers — float types, fire-and-forget (old value discarded).
static inline void atomic_store_f32(volatile float *addr, float value) {
    et_global_swap_w(addr, *(uint32_t *)&value);
}

static inline void atomic_add_f32(volatile float *addr, float value) {
    et_global_add_w(addr, *(uint32_t *)&value);
}

static inline void atomic_store_f16(volatile uint16_t *addr, uint16_t value) {
    et_global_store_hw(addr, value);
}

//******************************************************************************
// Barrier Primitives
//
// Hardware resources used (per shire):
//   - 32 FLBs: 8-bit atomic counters, non-blocking (CSR 0x820)
//   - 2 FCCs per hart: credit counters, hardware-stall on consume (CSR 0x821)
//
// Convention:
//   MINION barriers: FLB = local_minion_id (0-31), FCC 0
//   SHIRE  barriers: FLB 0,                        FCC 1
//
// MINION and SHIRE barriers MUST NOT be concurrent. All minion barriers
// must complete before a shire barrier, and vice versa. FLB 0 is shared
// between minion 0's barrier and the shire barrier — safe only because
// the FLB counter auto-resets on match.
//
// FCC 0 is safe for all 32 concurrent minion barriers because each
// barrier's fcc_send targets only its own minion (per-hart private
// counters, scoped by CREDINC mask). FCC 1 is reserved for shire-wide
// broadcast.
//******************************************************************************

#define ET_DEFAULT_SHIRE_MASK 0xFFFFFFFFULL

typedef enum {
    ET_BARRIER_MINION,  // sync both harts within each minion (FLB=minion_id, FCC 0)
    ET_BARRIER_SHIRE,   // sync all harts across the shire   (FLB=0, FCC 1)
    ET_BARRIER_GLOBAL,  // sync all harts across all active shires (FLB+global AMO+FCC)
} et_barrier_scope_t;

//******************************************************************************
// Global Barrier (cross-shire)
//
// Synchronizes all harts across multiple shires on the chip.
// Algorithm:
//   1. FLB within each shire to elect one representative hart
//   2. Elected hart does a global atomic increment on a shared counter
//   3. The last shire to arrive resets the counter and sends FCC credits
//      to all active shires to release them
//   4. All harts wait on FCC to complete the barrier
//
// Uses FLB 0, FCC 1 (same as ET_BARRIER_SHIRE, these must not overlap).
// The counter lives in a cache-line-aligned global to avoid coherency problems
//******************************************************************************

// Barrier counter cache-line aligned to avoid coherency problems
// Must be zero-initialized (BSS).
static uint32_t __attribute__((aligned(64)))
et_global_barrier_count[64 / sizeof(uint32_t)] = {0};

// Cross-shire barrier: all harts in num_active_shires shires synchronize.
// Returns 1 if this hart was the globally-last to arrive, 0 otherwise.
//
//   num_active_shires - number of shires participating
//                       (typically popcount(shire_mask) from kernel_environment_t)
static inline uint64_t __attribute__((always_inline))
et_barrier_global(uint64_t num_active_shires)
{
    uint64_t last_global = 0;

    // FLB within this shire. Elect one hart per shire.
    // Master shire has only 16 minions (32 harts), others have 32 (64 harts).
    uint64_t shire_id = get_shire_id();
    uint32_t harts_in_shire = (shire_id == SHIRE_MASTER)
        ? (SOC_MINIONS_PER_SHIRE / 2) * NUM_HARTS_PER_MINION
        : SOC_MINIONS_PER_SHIRE * NUM_HARTS_PER_MINION;
    uint64_t last_in_shire = flbarrier(0, harts_in_shire - 1);

    if (last_in_shire) {
        // Global atomic increment. Count arriving shires
        uint32_t prev = et_global_add_w(et_global_barrier_count, 1);

        if (prev == num_active_shires - 1) {
            // Last shire. reset counter and fan out FCC to all shires
            last_global = 1;
            et_global_swap_w(et_global_barrier_count, 0);

            for (uint64_t sid = 0; sid < 33; sid++) {
                // Send FCC 1 credit to all harts (both threads) in each shire
                fcc_send(sid, THREAD_0, FCC_1, 0xFFFFFFFF);
                fcc_send(sid, THREAD_1, FCC_1, 0xFFFFFFFF);
            }
        }
    }

    // All harts wait for the FCC credit from the last shire
    fcc_consume(FCC_1);
    return last_global;
}

// Barrier with scope-derived parameters.
// Returns 1 if this hart was the last to arrive, 0 otherwise.
//
// ET_BARRIER_GLOBAL uses ET_DEFAULT_SHIRE_MASK (32 shires). For a different
// shire count, use et_barrier_global(n) directly.
static inline uint64_t __attribute__((always_inline))
et_barrier(et_barrier_scope_t scope)
{
    if (scope == ET_BARRIER_MINION) {
        uint32_t local_minion = (get_hart_id() >> 1) & 0x1F;
        uint32_t mask = 1u << local_minion;
        return shire_barrier(local_minion, 0, 2, mask, mask);
    } else if (scope == ET_BARRIER_SHIRE) {
        uint64_t shire_id = get_shire_id();
        uint32_t thread_count = (shire_id == SHIRE_MASTER) ? 32 : 64;
        uint32_t mask = (shire_id == SHIRE_MASTER) ? 0xFFFF0000U : 0xFFFFFFFFU;
        return shire_barrier(0, 1, thread_count, mask, mask);
    } else { /* ET_BARRIER_GLOBAL */
        return et_barrier_global(manual_popcountll(ET_DEFAULT_SHIRE_MASK));
    }
}

// Raw barrier — caller manages FLB/FCC allocation.
// Use when et_barrier() doesn't fit (custom thread counts, subgroups,
// only even harts active, etc).
//
//   flb          - which FLB counter (0-31)
//   fcc          - which FCC counter (0 or 1)
//   thread_count - number of harts that will call this barrier
//   mask_t0      - CREDINC bitmask: which minions' hart 0 gets a credit
//   mask_t1      - CREDINC bitmask: which minions' hart 1 gets a credit
static inline uint64_t __attribute__((always_inline))
et_barrier_raw(uint32_t flb, uint32_t fcc, uint32_t thread_count,
               uint32_t mask_t0, uint32_t mask_t1)
{
    return shire_barrier(flb, fcc, thread_count, mask_t0, mask_t1);
}

// One-way semaphore between harts (non-blocking post, blocking wait).
//
// et_sem_post(): increment the partner hart's semaphore. Non-blocking.
//   the caller continues immediately. Multiple posts accumulate.
//
// et_sem_wait(): block until the semaphore is non-zero, then decrement it.
//
// Backed by hardware FCC (Flow Control Credit) counters. Uses FCC 0 for
// ET_BARRIER_MINION scope. Counters are per-hart private, so both harts
// can post/wait on the same scope independently.
//
// Must not be mixed with et_barrier() of the same scope in the
// same kernel (shared FCC channel).
static inline void __attribute__((always_inline))
et_sem_post(et_barrier_scope_t scope)
{
    if (scope == ET_BARRIER_MINION) {
        uint64_t hart_id = get_hart_id();
        uint32_t local_minion = (hart_id >> 1) & 0x1F;
        uint32_t mask = 1u << local_minion;
        uint64_t shire_id = get_shire_id();

        if (hart_id & 1) {
            // Hart 1 → hart 0
            fcc_send(shire_id, THREAD_0, FCC_0, mask);
        } else {
            // Hart 0 → hart 1
            fcc_send(shire_id, THREAD_1, FCC_0, mask);
        }
    }
}

// Block until a post from et_sem_post() is available, then consume it.
static inline void __attribute__((always_inline))
et_sem_wait(et_barrier_scope_t scope)
{
    if (scope == ET_BARRIER_MINION) {
        fcc_consume(FCC_0);
    }
}

//******************************************************************************
// Tensor Engine Wait & Error Macros
//
// These write to CSR 0x830 (tensor_wait) to stall the hart until the specified
// tensor unit completes its current operation.  The immediate encodes which
// unit to wait on.
//******************************************************************************

#define WAIT_TENSOR_LOAD_0     __asm__ __volatile__ ( "csrwi 0x830, 0\n"  : : );
#define WAIT_TENSOR_LOAD_1     __asm__ __volatile__ ( "csrwi 0x830, 1\n"  : : );
#define WAIT_TENSOR_LOAD_L2_0  __asm__ __volatile__ ( "csrwi 0x830, 2\n"  : : );
#define WAIT_TENSOR_LOAD_L2_1  __asm__ __volatile__ ( "csrwi 0x830, 3\n"  : : );
#define WAIT_PREFETCH_0        __asm__ __volatile__ ( "csrwi 0x830, 4\n"  : : );
#define WAIT_PREFETCH_1        __asm__ __volatile__ ( "csrwi 0x830, 5\n"  : : );
#define WAIT_CACHEOPS          __asm__ __volatile__ ( "csrwi 0x830, 6\n"  : : );
#define WAIT_TENSOR_FMA        __asm__ __volatile__ ( "csrwi 0x830, 7\n"  : : );
#define WAIT_TENSOR_STORE      __asm__ __volatile__ ( "csrwi 0x830, 8\n"  : : );
#define WAIT_TENSOR_REDUCE     __asm__ __volatile__ ( "csrwi 0x830, 9\n"  : : );
#define WAIT_TENSOR_QUANT      __asm__ __volatile__ ( "csrwi 0x830, 10\n" : : );
#define STALL                  __asm__ __volatile__ ( "csrw stall, x0\n"  : : );

// Write 0 to CSR 0x808 (tensor_error) to clear any latched tensor error bits.
// Must be issued before the first tensor operation in a kernel to avoid stale
// errors from a previous invocation causing spurious faults.
#define CLEAR_TENSOR_ERROR     __asm__ __volatile__ ( "csrwi 0x808, 0" : : );

//******************************************************************************
// L1 Data Cache / Scratchpad (SCP) Configuration
//
// The ET-SoC-1 L1 data cache can be split so that half its ways operate as a
// software-managed scratchpad (SCP).  Tensor load/store/FMA instructions
// require SCP mode to be active.
//
// CSR 0x810 — ucache_control:
//
//   Bit(s)  Field         Description
//   ──────  ────────────  ──────────────────────────────────────────────────
//   [0]     D1Split       1 = L1 is split (half cache, half SCP).
//                          Read-only from U-mode; set by M-mode firmware
//                          before kernel launch.  Writing ScpEnable while
//                          D1Split=0 is silently ignored.
//   [1]     ScpEnable     1 = scratchpad is active and zeroed.
//   [4:2]   RepRate       Cache-op replay rate (0 = no delay between ops).
//   [10:6]  CacheOpMax    Max outstanding cache ops (0 = unlimited).
//
// Typical kernel prologue for tensor operations:
//     setup_cache_scp();   // enables SCP, waits for zeroing
//     CLEAR_TENSOR_ERROR;  // clear stale error bits
//******************************************************************************

// Write the ucache_control CSR (0x810).
//
//   scp_en       — 1 to enable SCP mode (requires D1Split already set)
//   cacheop_rate — cache-op replay rate (0–7; 0 = no delay)
//   cacheop_max  — max outstanding cache ops (0–31; 0 = unlimited)
static inline void __attribute__((always_inline))
ucache_control(uint64_t scp_en, uint64_t cacheop_rate, uint64_t cacheop_max)
{
    uint64_t csr_enc = ((cacheop_max & 0x1F) << 6) |
                       ((cacheop_rate & 0x7)  << 2) |
                       ((scp_en & 0x1)        << 1);

    __asm__ __volatile__("csrw 0x810, %[csr_enc]\n" : : [csr_enc] "r"(csr_enc) : "x31");
}

// Enable L1 scratchpad mode and wait for the transition to complete.
// After this call the SCP lines are zeroed and ready for tensor operations.
//
// Prerequisites:
//   - D1Split must already be 1 (set by M-mode firmware at boot).
//   - Only even harts (hart 0 per minion) should call this, as only they
//     can issue tensor instructions.
static inline void setup_cache_scp(void)
{
    FENCE;                    // drain pending stores before reconfiguring cache
    ucache_control(1, 0, 0);  // ScpEnable=1
    WAIT_CACHEOPS;            // wait for SCP mode transition + zeroing
}

//******************************************************************************
// L2 Scratchpad (L2 SCP) Address Computation
//
// Each shire has 4 MB of SRAM that can be split across L2 cache, L3 cache,
// and scratchpad.  The scratchpad region occupies 0x00_8000_0000~0x00_FFFF_FFFF
// and is accessible via regular load/store from any minion core.
//
// Two addressing formats (differentiated by address bit 30):
//
// Format 0 (bit[30]=0): Direct shire addressing
//   [29:23] = shire ID (0–33, or 0x7F for local shire)
//   [22:0]  = byte offset within shire's scratchpad
//
// Format 1 (bit[30]=1): Striped (round-robin) addressing
//   [29:28] = shire ID[6:5]
//   [27:11] = offset[22:6]   (cache-line-aligned upper bits)
//   [10:6]  = shire ID[4:0]
//   [5:0]   = offset[5:0]    (byte within cache line)
//   Consecutive 64-byte cache lines cycle through different shires,
//   distributing bandwidth across the mesh.
//
// Shire ID 0x7F always targets the local shire (instead of figureing out which
// shire you are on).
//******************************************************************************

#define L2SCP_BASE         0x0080000000ULL
#define L2SCP_SHIRE_LOCAL  0x7FULL

// Format 0: direct address into a specific shire's L2 SCP.
//   shire: 0–33 for explicit shire, L2SCP_SHIRE_LOCAL (0x7F) for local
//   offset: byte offset within the shire's scratchpad
static inline void * __attribute__((always_inline))
et_shire_l2scp(uint64_t shire, uint64_t offset)
{
    return (void *)(L2SCP_BASE
                    | ((shire & 0x7F) << 23)
                    | (offset & 0x7FFFFF));
}

// Format 0: local shire shorthand — no cross-shire traffic.
static inline void * __attribute__((always_inline))
et_shire_l2scp_local(uint64_t offset)
{
    return (void *)(L2SCP_BASE
                    | (L2SCP_SHIRE_LOCAL << 23)
                    | (offset & 0x7FFFFF));
}

// Format 1: flat offset into a hardware-striped global address space.
// Consecutive 64-byte cache lines automatically land on different shires,
// distributing bandwidth across the mesh.  No shire parameter — the
// hardware derives the target shire from the address bits.
static inline void * __attribute__((always_inline))
et_global_l2scp(uint64_t offset)
{
    return (void *)(L2SCP_BASE
                    | (1ULL << 30)
                    | (offset & 0x3FFFFFFF));
}

//******************************************************************************
// Cache Operatons
//******************************************************************************

// Prefetch nlines cache lines into L2 starting at addr, with stride bytes
// between each line.  Uses PrefetchVA (CSR 0x81F) with dest=L2 (bits 59:58=01).
//
// The hardware fetches nlines consecutive cache-line-sized (64B) blocks from
// DRAM/L3 into L2, starting at addr and advancing by stride bytes per line.
// This is asynchronous — use WAIT_PREFETCH_0 or WAIT_PREFETCH_1 if the hart
// must stall until the prefetch completes.
//
// NOTE: nlines is encoded in a 4-bit field (max 16). Passing nlines > 16
// silently truncates. DO NOT pass nlines > 16.
static inline void __attribute__((always_inline))
l2_prefetch(const void *addr, uint64_t nlines, uint64_t stride)
{
    uint64_t csr_val = (0x1ULL << 58) |
                       ((uint64_t)addr & 0xFFFFFFFFFFC0ULL) |
                       ((nlines - 1) & 0xF);

    __asm__ __volatile__ (
        "mv    x31, %[stride]\n"
        "csrw  0x81f, %[val]\n"
        :
        : [stride] "r" (stride & 0xFFFFFFFFFFC0ULL),
          [val] "r" (csr_val)
        : "x31", "memory"
    );
}

// Flush nlines cache lines at stride apart starting at addr from L1 to L2.
// Uses FlushVA (CSR 0x8BF).  Caller must FENCE before (to drain stores to L1)
// and WAIT_CACHEOPS after (to ensure flush completes before tensor loads).
//
// NOTE: nlines is encoded in a 4-bit field (max 16). Passing nlines > 16
// silently truncates. DO NOT pass nlines > 16.
static inline void __attribute__((always_inline))
flush_to_l2(const void *addr, uint64_t nlines, uint64_t stride)
{
    // dest=01 (L2) in bits 59:58, VA in bits 47:6, numlines-1 in bits 3:0
    uint64_t csr_val = (0x1ULL << 58) |
                       ((uint64_t)addr & 0xFFFFFFFFFFC0ULL) |
                       ((nlines - 1) & 0xF);
    uint64_t x31_val = stride & 0xFFFFFFFFFFC0ULL;

    __asm__ __volatile__(
        "mv x31, %[x31]\n"
        "csrw 0x8BF, %[val]\n"
        :
        : [x31] "r"(x31_val), [val] "r"(csr_val)
        : "x31", "memory"
    );
}

// Evict nlines cache lines at stride apart starting at addr from L1 to L2.
// Uses EvictVA (CSR 0x89F).  Unlike flush_to_l2, this guarantees the line is
// NOT present in L1 after the operation - subsequent loads will miss and go
// to L2/SCP. Caller must FENCE before and WAIT_CACHEOPS after.
//
// NOTE: nlines is encoded in a 4-bit field (max 16). DO NOT pass nlines > 16.
static inline void __attribute__((always_inline))
evict_to_l2(const void *addr, uint64_t nlines, uint64_t stride)
{
    // dest=01 (L2) in bits 59:58, VA in bits 47:6, numlines-1 in bits 3:0
    uint64_t csr_val = (0x1ULL << 58) |
                       ((uint64_t)addr & 0xFFFFFFFFFFC0ULL) |
                       ((nlines - 1) & 0xF);
    uint64_t x31_val = stride & 0xFFFFFFFFFFC0ULL;

    __asm__ __volatile__(
        "mv x31, %[x31]\n"
        "csrw 0x89F, %[val]\n"
        :
        : [x31] "r"(x31_val), [val] "r"(csr_val)
        : "x31", "memory"
    );
}

#endif // PLATFORM_H
