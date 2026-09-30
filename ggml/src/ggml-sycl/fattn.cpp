//
// MIT license
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//


#include <sycl/sycl.hpp>
#include "dpct/helper.hpp"
#include "common.hpp"
#include "fattn-common.hpp"
#include "fattn-tile.hpp"
#include "fattn-vec.hpp"
#include "fattn-xmx.hpp"
#include "fattn.hpp"
#include "fattn-onednn.hpp"
#include "fattn-sparse.hpp"

bool ggml_sycl_fa_large_grf_variants() {
    return GGML_SYCL_FA_LARGE_GRF_VARIANTS != 0;
}

int ggml_sycl_fa_large_grf_mode() {
    static const int mode = [] {
        const int v = ggml_sycl_get_env("GGML_SYCL_FA_LARGE_GRF", 0);
        if (v < 0 || v > 2) {
            GGML_LOG_WARN("ggml_sycl_fa_large_grf_mode: ignoring invalid GGML_SYCL_FA_LARGE_GRF=%d (0, 1 or 2)\n", v);
            return 0;
        }
        if (v != 0 && !ggml_sycl_fa_large_grf_variants()) {
            GGML_LOG_WARN("ggml_sycl_fa_large_grf_mode: GGML_SYCL_FA_LARGE_GRF=%d ignored: built without the 256-GRF "
                          "FA variants (CMake GGML_SYCL_FA_LARGE_GRF=OFF)\n", v);
            return 0;
        }
        return v;
    }();
    return mode;
}
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <mutex>


#define FATTN_VEC_CASE(D, type_K, type_V)                                                                        \
    {                                                                                                            \
        const bool type_K_okay = K->type == (type_K) || (K->type == GGML_TYPE_F32 && (type_K) == GGML_TYPE_F16); \
        const bool type_V_okay = V->type == (type_V) || (V->type == GGML_TYPE_F32 && (type_V) == GGML_TYPE_F16); \
        if (Q->ne[0] == (D) && type_K_okay && type_V_okay) {                                                     \
            if constexpr ((D) == 128 && (type_K) == GGML_TYPE_Q8_0 && (type_V) == GGML_TYPE_Q8_0) {              \
                const bool K_quants_first = ggml_sycl_tensor_is_kv_q8_quants_first(K);                                     \
                const bool V_quants_first = ggml_sycl_tensor_is_kv_q8_quants_first(V);                                     \
                GGML_ASSERT(K_quants_first == V_quants_first);                                                     \
                if (K_quants_first) {                                                                             \
                    ggml_sycl_flash_attn_ext_vec_case_q8_quants_first<D>(ctx, dst);                               \
                    return;                                                                                      \
                }                                                                                               \
            }                                                                                                   \
            ggml_sycl_flash_attn_ext_vec_case<D, type_K, type_V>(ctx, dst);                                      \
            return;                                                                                              \
        }                                                                                                       \
    }                                                                    \

#define FATTN_VEC_CASES_ALL_D(type_K, type_V) \
    FATTN_VEC_CASE( 64, type_K, type_V)       \
    FATTN_VEC_CASE(128, type_K, type_V)       \
    FATTN_VEC_CASE(256, type_K, type_V)       \
    FATTN_VEC_CASE(512, type_K, type_V)       \

// Turbo blocks span 128 elements, so turbo K/V only supports head sizes that are
// multiples of 128 (see the D % 128 gate in ggml_sycl_get_best_fattn_kernel).
#define FATTN_VEC_CASES_TURBO_D(type_K, type_V) \
    FATTN_VEC_CASE(128, type_K, type_V)         \
    FATTN_VEC_CASE(256, type_K, type_V)         \
    FATTN_VEC_CASE(512, type_K, type_V)         \

static void ggml_sycl_flash_attn_ext_vec(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];

#ifdef GGML_SYCL_FA_ALL_QUANTS  // P3.2.2a2a first cut: leave the
       // mixed-K flash_attn_ext_vec<...,42/43/44,...> dispatch block
       // available, but with the macro UNDEFINED in
       // ggml/src/ggml-sycl/common.hpp:48 by default. The runtime
       // rejection path in ggml_sycl_get_best_fattn_kernel()
       // (fattn.cpp:224,237) is behind #ifndef, so the macro
       // state is the single source of truth - flipping it in
       // common.hpp is the only edit needed to switch dispatch
       // modes end-to-end. See ASSUMPTIONS.md:553-583 +
       // RALPH_TASKS.md:1237-1251.
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_TURBO2_0,GGML_TYPE_F16)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_TURBO3_0,GGML_TYPE_F16)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_TURBO4_0,GGML_TYPE_F16)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_TURBO2_0,GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_TURBO3_0,GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_TURBO4_0,GGML_TYPE_Q4_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_TURBO2_0,GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_TURBO3_0,GGML_TYPE_Q4_1)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_TURBO4_0,GGML_TYPE_Q4_1)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_TURBO2_0,GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_TURBO3_0,GGML_TYPE_Q5_0)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_TURBO4_0,GGML_TYPE_Q5_0)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_TURBO2_0,GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_TURBO3_0,GGML_TYPE_Q5_1)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_TURBO4_0,GGML_TYPE_Q5_1)

    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_1, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q5_1, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_TURBO2_0,GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_TURBO3_0,GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_TURBO4_0,GGML_TYPE_Q8_0)

    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_F16,  GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_Q4_0, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_Q4_1, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_Q5_0, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_Q5_1, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_Q8_0, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_TURBO2_0,GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_TURBO3_0,GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_TURBO4_0,GGML_TYPE_TURBO2_0)

    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_F16,  GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_Q4_0, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_Q4_1, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_Q5_0, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_Q5_1, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_Q8_0, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_TURBO2_0,GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_TURBO3_0,GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_TURBO4_0,GGML_TYPE_TURBO3_0)

    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_F16,  GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_Q4_0, GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_Q4_1, GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_Q5_0, GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_Q5_1, GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_Q8_0, GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_TURBO2_0,GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_TURBO3_0,GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_TURBO4_0,GGML_TYPE_TURBO4_0)
#else
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_F16,  GGML_TYPE_F16)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_CASES_ALL_D(GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_TURBO2_0,GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_TURBO3_0,GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_TURBO4_0,GGML_TYPE_TURBO4_0)
    // P3.2.2b0a1b4aab: post-auto-asymmetric K=q8_0 + V=turbo mix rows;
    // already in the GGML_SYCL_FA_ALL_QUANTS matrix at :126/:136/:146
    // for the macro-DEF branch, but the cheap-cut #else branch compiles
    // only same-type K=V pairs, so the post-auto-asymmetric path needs
    // its own row here. Dispatches require explicit -ctk q8_0 -ctv turbo
    // (or auto-asymmetric K downgrade produces this pair at runtime).
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_Q8_0,GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_Q8_0,GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_Q8_0,GGML_TYPE_TURBO4_0)
#endif // GGML_SYCL_FA_ALL_QUANTS

#ifdef GGML_SYCL_TURBO_QUANT
    // Turbo KV cache type combos. Turbo blocks span 128 elements (D % 128 == 0
    // required), so every row here uses FATTN_VEC_CASES_TURBO_D (D in
    // {128,256,512}), never FATTN_VEC_CASES_ALL_D (which includes D=64).
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_TURBO2_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_F16,       GGML_TYPE_TURBO2_0)

    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_TURBO3_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_F16,       GGML_TYPE_TURBO3_0)

    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_TURBO4_0, GGML_TYPE_F16)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_F16,       GGML_TYPE_TURBO4_0)

    // q8_0 + turbo combos
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_Q8_0,      GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_TURBO2_0,  GGML_TYPE_Q8_0)

    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_Q8_0,      GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_TURBO3_0,  GGML_TYPE_Q8_0)

    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_Q8_0,      GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_TURBO4_0,  GGML_TYPE_Q8_0)

    // Cross-turbo K/V combos
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO3_0)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO2_0)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_TURBO3_0, GGML_TYPE_TURBO4_0)
    FATTN_VEC_CASES_TURBO_D(GGML_TYPE_TURBO4_0, GGML_TYPE_TURBO3_0)
#endif // GGML_SYCL_TURBO_QUANT

    GGML_ABORT("Not match KV type in vec");
}

// Best FlashAttention kernel for a specific GPU:
enum best_fattn_kernel {
    BEST_FATTN_KERNEL_NONE     =   0,
    BEST_FATTN_KERNEL_VEC      = 100,
    BEST_FATTN_KERNEL_ONEDNN   = 150, // oneDNN SDPA: native F16 (PR #25222)
    BEST_FATTN_KERNEL_TILE     = 200,
    BEST_FATTN_KERNEL_XMX      = 300,
    BEST_FATTN_KERNEL_MKL      = 350,
};

struct ggml_sycl_fattn_profile_bucket {
    uint64_t launches = 0;
    uint64_t conversion_us = 0;
    uint64_t conversion_bytes = 0;
    uint64_t stage1_us = 0;
    uint64_t combine_us = 0;
    uint64_t gqa_ratio = 0;
    uint64_t repeated_packed_kv_bytes = 0;
    // Launch geometry of the most recent launch in this bucket. These are shapes,
    // not accumulators, so they record last-value-wins like gqa_ratio.
    uint64_t parallel_blocks = 0;
    uint64_t ntiles_total = 0;
    uint64_t blocks_total = 0;
    uint64_t work_items_total = 0;
    uint64_t max_wg_per_cu = 0;
    uint64_t nsm = 0;
};

class ggml_sycl_fattn_profile_collector {
public:
    ~ggml_sycl_fattn_profile_collector() {
        for (int route = 0; route < 2; ++route) {
            for (int layout = 0; layout < 2; ++layout) {
                const ggml_sycl_fattn_profile_bucket & bucket = buckets[route][layout];
                if (bucket.launches == 0) {
                    continue;
                }
                fprintf(
                    stderr,
                    "GGML_SYCL_FA_PROFILE: route=%s layout=%s launches=%llu "
                    "conversion_us=%llu conversion_bytes=%llu stage1_us=%llu "
                    "combine_us=%llu gqa=%llu repeated_packed_kv_bytes=%llu "
                    "parallel_blocks=%llu ntiles_total=%llu blocks_total=%llu "
                    "work_items_total=%llu max_wg_per_cu=%llu nsm=%llu\n",
                    route == 0 ? "VEC" : "TILE",
                    layout == 0 ? "canonical" : "quants-first",
                    (unsigned long long) bucket.launches,
                    (unsigned long long) bucket.conversion_us,
                    (unsigned long long) bucket.conversion_bytes,
                    (unsigned long long) bucket.stage1_us,
                    (unsigned long long) bucket.combine_us,
                    (unsigned long long) bucket.gqa_ratio,
                    (unsigned long long) bucket.repeated_packed_kv_bytes,
                    (unsigned long long) bucket.parallel_blocks,
                    (unsigned long long) bucket.ntiles_total,
                    (unsigned long long) bucket.blocks_total,
                    (unsigned long long) bucket.work_items_total,
                    (unsigned long long) bucket.max_wg_per_cu,
                    (unsigned long long) bucket.nsm);
            }
        }
        fflush(stderr);
    }

    void record(
        bool tile_route,
        bool quants_first,
        uint64_t conversion_us,
        uint64_t conversion_bytes,
        uint64_t stage1_us,
        uint64_t combine_us,
        uint64_t gqa_ratio,
        uint64_t repeated_packed_kv_bytes,
        uint64_t parallel_blocks,
        uint64_t ntiles_total,
        uint64_t blocks_total,
        uint64_t work_items_total,
        uint64_t max_wg_per_cu,
        uint64_t nsm) {
        std::lock_guard<std::mutex> lock(mutex);
        ggml_sycl_fattn_profile_bucket & bucket =
            buckets[tile_route ? 1 : 0][quants_first ? 1 : 0];
        bucket.launches++;
        bucket.conversion_us += conversion_us;
        bucket.conversion_bytes += conversion_bytes;
        bucket.stage1_us += stage1_us;
        bucket.combine_us += combine_us;
        bucket.gqa_ratio = gqa_ratio;
        bucket.repeated_packed_kv_bytes += repeated_packed_kv_bytes;
        bucket.parallel_blocks = parallel_blocks;
        bucket.ntiles_total = ntiles_total;
        bucket.blocks_total = blocks_total;
        bucket.work_items_total = work_items_total;
        bucket.max_wg_per_cu = max_wg_per_cu;
        bucket.nsm = nsm;
    }

private:
    std::mutex mutex;
    ggml_sycl_fattn_profile_bucket buckets[2][2] = {};
};

bool ggml_sycl_fattn_profile_enabled() {
    static const bool enabled = [] {
        const char * value = getenv("GGML_SYCL_FA_PROFILE");
        return value != nullptr && value[0] != '\0' && !(value[0] == '0' && value[1] == '\0');
    }();
    return enabled;
}

// Sync-free geometry table, keyed by route x phase. Values are shapes, so the last
// launch in each bucket wins; only the launch counter accumulates.
struct ggml_sycl_fattn_geometry_bucket {
    uint64_t launches = 0;
    char type_k[16] = {};
    uint64_t parallel_blocks = 0;
    uint64_t ntiles_total = 0;
    uint64_t blocks_total = 0;
    uint64_t work_items_total = 0;
    uint64_t max_wg_per_cu = 0;
    uint64_t nsm = 0;
    uint64_t stream_k = 0;
};

class ggml_sycl_fattn_geometry_collector {
public:
    ~ggml_sycl_fattn_geometry_collector() {
        for (int route = 0; route < 2; ++route) {
            for (int phase = 0; phase < 2; ++phase) {
                const ggml_sycl_fattn_geometry_bucket & bucket = buckets[route][phase];
                if (bucket.launches == 0) {
                    continue;
                }
                fprintf(
                    stderr,
                    "GGML_SYCL_FA_GEOMETRY: route=%s phase=%s type_k=%s launches=%llu "
                    "parallel_blocks=%llu ntiles_total=%llu blocks_total=%llu "
                    "work_items_total=%llu max_wg_per_cu=%llu nsm=%llu stream_k=%llu\n",
                    route == 0 ? "VEC" : "TILE",
                    phase == 0 ? "decode" : "prefill",
                    bucket.type_k,
                    (unsigned long long) bucket.launches,
                    (unsigned long long) bucket.parallel_blocks,
                    (unsigned long long) bucket.ntiles_total,
                    (unsigned long long) bucket.blocks_total,
                    (unsigned long long) bucket.work_items_total,
                    (unsigned long long) bucket.max_wg_per_cu,
                    (unsigned long long) bucket.nsm,
                    (unsigned long long) bucket.stream_k);
            }
        }
        fflush(stderr);
    }

    void record(
        bool tile_route,
        bool decode,
        const char * type_k,
        uint64_t parallel_blocks,
        uint64_t ntiles_total,
        uint64_t blocks_total,
        uint64_t work_items_total,
        uint64_t max_wg_per_cu,
        uint64_t nsm,
        uint64_t stream_k) {
        std::lock_guard<std::mutex> lock(mutex);
        ggml_sycl_fattn_geometry_bucket & bucket =
            buckets[tile_route ? 1 : 0][decode ? 0 : 1];
        bucket.launches++;
        if (type_k != nullptr) {
            std::snprintf(bucket.type_k, sizeof(bucket.type_k), "%s", type_k);
        }
        bucket.parallel_blocks  = parallel_blocks;
        bucket.ntiles_total     = ntiles_total;
        bucket.blocks_total     = blocks_total;
        bucket.work_items_total = work_items_total;
        bucket.max_wg_per_cu    = max_wg_per_cu;
        bucket.nsm              = nsm;
        bucket.stream_k         = stream_k;
    }

private:
    std::mutex mutex;
    ggml_sycl_fattn_geometry_bucket buckets[2][2] = {};
};

void ggml_sycl_fattn_profile_record_geometry(
    bool tile_route,
    bool decode,
    const char * type_k,
    uint64_t parallel_blocks,
    uint64_t ntiles_total,
    uint64_t blocks_total,
    uint64_t work_items_total,
    uint64_t max_wg_per_cu,
    uint64_t nsm,
    uint64_t stream_k) {
    static ggml_sycl_fattn_geometry_collector collector;
    collector.record(
        tile_route,
        decode,
        type_k,
        parallel_blocks,
        ntiles_total,
        blocks_total,
        work_items_total,
        max_wg_per_cu,
        nsm,
        stream_k);
}

void ggml_sycl_fattn_profile_record(
    bool tile_route,
    bool quants_first,
    uint64_t conversion_us,
    uint64_t conversion_bytes,
    uint64_t stage1_us,
    uint64_t combine_us,
    uint64_t gqa_ratio,
    uint64_t repeated_packed_kv_bytes,
    uint64_t parallel_blocks,
    uint64_t ntiles_total,
    uint64_t blocks_total,
    uint64_t work_items_total,
    uint64_t max_wg_per_cu,
    uint64_t nsm) {
    static ggml_sycl_fattn_profile_collector collector;
    collector.record(
        tile_route,
        quants_first,
        conversion_us,
        conversion_bytes,
        stage1_us,
        combine_us,
        gqa_ratio,
        repeated_packed_kv_bytes,
        parallel_blocks,
        ntiles_total,
        blocks_total,
        work_items_total,
        max_wg_per_cu,
        nsm);
}

static void ggml_sycl_log_fattn_route_once(best_fattn_kernel route, const ggml_tensor * dst) {
    if (!ggml_sycl_fattn_profile_enabled() || route == BEST_FATTN_KERNEL_NONE) {
        return;
    }

    int route_index = 0;
    const char * route_name = "VEC";
    switch (route) {
        case BEST_FATTN_KERNEL_VEC:
            break;
        case BEST_FATTN_KERNEL_TILE:
            route_index = 1;
            route_name = "TILE";
            break;
        case BEST_FATTN_KERNEL_XMX:
            route_index = 2;
            route_name = "XMX";
            break;
        case BEST_FATTN_KERNEL_ONEDNN:
            route_index = 3;
            route_name = "ONEDNN";
            break;
        case BEST_FATTN_KERNEL_MKL:
            route_index = 4;
            route_name = "MKL";
            break;
        case BEST_FATTN_KERNEL_NONE:
            return;
    }

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const bool is_decode = Q->ne[1] == 1;
    static std::atomic<bool> logged[5][2] = {};
    if (logged[route_index][is_decode ? 0 : 1].exchange(true, std::memory_order_relaxed)) {
        return;
    }

    const bool K_quants_first =
        K->type == GGML_TYPE_Q8_0 && ggml_sycl_tensor_is_kv_q8_quants_first(K);
    const bool V_quants_first =
        V->type == GGML_TYPE_Q8_0 && ggml_sycl_tensor_is_kv_q8_quants_first(V);
    // llama-bench suppresses GGML_LOG_INFO in JSON mode; trace output must bypass its callback.
    fprintf(
        stderr,
        "GGML_SYCL_FA_ROUTE: route=%s phase=%s q_tokens=%lld head_dim=%lld gqa=%lld "
        "type_k=%s type_v=%s k_quants_first=%d v_quants_first=%d\n",
        route_name,
        is_decode ? "decode" : "prefill",
        (long long) Q->ne[1],
        (long long) K->ne[0],
        (long long) (Q->ne[2] / K->ne[2]),
        ggml_type_name(K->type),
        ggml_type_name(V->type),
        K_quants_first ? 1 : 0,
        V_quants_first ? 1 : 0);
    fflush(stderr);
}

static best_fattn_kernel ggml_sycl_get_best_fattn_kernel(const int device, const ggml_tensor * dst) {
#ifndef SYCL_FLASH_ATTN
    GGML_UNUSED(dst);
    return BEST_FATTN_KERNEL_NONE;
#endif// SYCL_FLASH_ATTN

    if(!g_ggml_sycl_enable_flash_attention) return BEST_FATTN_KERNEL_NONE;

    const ggml_tensor * KQV   = dst;
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    const int gqa_ratio = Q->ne[2] / K->ne[2];
    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    float logit_softcap = 0.0f;
    memcpy(&logit_softcap, (const float *) KQV->op_params + 2, sizeof(float));
    // The XMX FA kernel ignores ALiBi (max_bias), logit soft-capping, attention
    // sinks (src[4]) and multi-sequence batches (ne[3] > 1). Route any of those to
    // the existing VEC/TILE kernels so GGML_SYCL_FA_XMX cannot silently change results.
    const bool xmx_features_ok =
        max_bias == 0.0f && logit_softcap == 0.0f && dst->src[4] == nullptr &&
        Q->ne[3] == 1 && K->ne[3] == 1 && V->ne[3] == 1 && (mask == nullptr || mask->ne[3] == 1);

    bool gqa_opt_applies = gqa_ratio >= 2 && mask && max_bias == 0.0f && K->ne[1] % FATTN_KQ_STRIDE == 0;

    for (const ggml_tensor * t : {Q, K, V, mask}) {
        if (t == nullptr || ggml_is_quantized(t->type)) {
            continue;
        }
        for (size_t i = 1; i < GGML_MAX_DIMS; ++i) {
            if (t->nb[i] % 16 != 0) {
                gqa_opt_applies = false;
                break;
            }
        }
    }

    switch (K->ne[0]) {
        case  40:
        case  64:
        case  72:
        case  80:
        case  96:
        case 128:
        case 112:
        case 256:
        case 512:
            if (V->ne[0] != K->ne[0]) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        case 576:
            if (V->ne[0] != 512) {
                return BEST_FATTN_KERNEL_NONE;
            }
            if (!gqa_opt_applies) {
                return BEST_FATTN_KERNEL_NONE;
            }
            break;
        default:
            return BEST_FATTN_KERNEL_NONE;
    }

    const bool K_is_turbo = (K->type == GGML_TYPE_TURBO2_0 || K->type == GGML_TYPE_TURBO3_0 || K->type == GGML_TYPE_TURBO4_0);
    const bool V_is_turbo = (V->type == GGML_TYPE_TURBO2_0 || V->type == GGML_TYPE_TURBO3_0 || V->type == GGML_TYPE_TURBO4_0);

#ifndef GGML_SYCL_TURBO_QUANT
    if (K_is_turbo || V_is_turbo) {
        return BEST_FATTN_KERNEL_NONE;
    }
#endif // GGML_SYCL_TURBO_QUANT

#ifndef GGML_SYCL_FA_ALL_QUANTS
    // Mixed K/V types are rejected by default, except turbo combos: same-type
    // turbo, cross-turbo (turbo2<->turbo3<->turbo4), and turbo+F16/turbo+Q8_0
    // (e.g. the post-auto-asymmetric K=q8_0 + V=turbo pair so Qwen3 GQA 8:1,
    // after the auto-asymmetric K downgrade at src/llama-kv-cache.cpp:152
    // fires, can reach the VEC kernel). The precise combinations are
    // re-validated below; this gate only needs to defer to it.
    if (K->type != V->type && !K_is_turbo && !V_is_turbo) {
        return BEST_FATTN_KERNEL_NONE;
    }
#endif // GGML_SYCL_FA_ALL_QUANTS

    switch (K->type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
            break;
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
#ifndef GGML_SYCL_FA_ALL_QUANTS
            return BEST_FATTN_KERNEL_NONE;
#endif // GGML_SYCL_FA_ALL_QUANTS
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_TURBO2_0:
        case GGML_TYPE_TURBO3_0:
        case GGML_TYPE_TURBO4_0:
            break;
        default:
            return BEST_FATTN_KERNEL_NONE;
    }

    if (V_is_turbo || K_is_turbo) {
        if (Q->ne[0] > 512) {
            return BEST_FATTN_KERNEL_NONE;
        }
        // Cross-turbo K/V combos supported (turbo2<->turbo3, turbo2<->turbo4, turbo3<->turbo4)
        if (K_is_turbo && !V_is_turbo && V->type != GGML_TYPE_F16 && V->type != GGML_TYPE_Q8_0) {
            return BEST_FATTN_KERNEL_NONE;
        }
        if (V_is_turbo && !K_is_turbo && K->type != GGML_TYPE_F16 && K->type != GGML_TYPE_Q8_0) {
            return BEST_FATTN_KERNEL_NONE;
        }
        // softcap+turbo is valid for D<=512: softcap is applied post-dot-product, independent of K/V quantization.
    }

    if (mask && mask->ne[2] != 1) {
        return BEST_FATTN_KERNEL_NONE;
    }

    // Turbo KV uses the VEC kernel exclusively: it is the only SYCL turbo FA path
    // with complete K and V dequant (need_f16 = false). VEC tiles over Q columns,
    // so it serves both decode and prefill. TILE turbo is unsupported. Turbo blocks
    // span 128 elements, so only head sizes that are multiples of 128 are usable.
    const bool K_turbo = ggml_type_is_turbo(K->type);
    const bool V_turbo = ggml_type_is_turbo(V->type);
    if (K_turbo || V_turbo) {
        if (K->ne[0] % 128 != 0) {
            return BEST_FATTN_KERNEL_NONE;
        }
        // XMX turbo (opt-in, off by default): same turbo type on K and V, D in
        // {128, 256}. The DPAS kernel dequants turbo blocks into its staging tiles
        // (rotated domain, same as VEC). D=512 exceeds the 64KB SLM budget, and
        // mixed turbo or turbo+f16 KV stay on VEC.
        if (getenv("GGML_SYCL_FA_XMX") && xmx_features_ok && K->type == V->type && (K->ne[0] == 128 || K->ne[0] == 256)) {
            return BEST_FATTN_KERNEL_XMX;
        }
        return BEST_FATTN_KERNEL_VEC;
    }

    // MKL path: XMX-accelerated GEMM for prompt processing (all KV cache types,
    // reached here only for non-turbo K/V since the turbo branch above always
    // returns). The MKL kernel converts non-F16 K/V to F16 via to_fp16_sycl
    // before GEMM, so quantized, F16, BF16, and F32 caches all qualify.
    // Validated envelope: grouped-query (gqa_ratio >= 2), head_dim a multiple
    // of 64 in [64,512] with matching K/V head size, mask present, no
    // sinks/ALiBi/softcap, single-sequence batch. Everything else falls
    // through to XMX/oneDNN/VEC/TILE below. Set GGML_SYCL_ENABLE_MKL_FA=0 to
    // force the fallback path for A/B testing.
    if (g_ggml_sycl_enable_mkl_fa == 1 && mask && !sinks && gqa_ratio >= 2 &&
        Q->ne[0] >= 64 && Q->ne[0] <= 512 && Q->ne[0] % 64 == 0 &&
        Q->ne[0] == V->ne[0] &&
        Q->ne[1] >= 32 && K->ne[1] >= 1024 &&
        max_bias == 0.0f && logit_softcap == 0.0f &&
        (Q->ne[3] == K->ne[3] || K->ne[3] == 1)) {
        // F16 K/V strides must be a multiple of ne[0]*2 (the natural row size
        // in bytes). This passes both dense (nb1 == ne0*2) and interleaved
        // (nb1 == H * ne0*2) layouts; only pathological test strides fall
        // through to TILE.
        bool kv_strides_ok = true;
        for (const ggml_tensor * t : {K, V}) {
            if (t->type == GGML_TYPE_F16 && t->nb[1] % (t->ne[0] * 2) != 0) {
                kv_strides_ok = false;
                break;
            }
        }
        if (kv_strides_ok) {
            return BEST_FATTN_KERNEL_MKL;
        }
    }

    // BF16 is excluded: the VEC kernel has no BF16 template (it needs GGML_SYCL_FA_ALL_QUANTS for non-F16/Q4_0/Q8_0).
    const bool has_bf16 = (K->type == GGML_TYPE_BF16 || V->type == GGML_TYPE_BF16);
    const bool can_use_vector_kernel = Q->ne[0] <= 512 && Q->ne[0] % 64 == 0 && K->ne[1] % FATTN_KQ_STRIDE == 0
        && !has_bf16;
    // TILE's quantized staging uses the source-aware non-contiguous converter,
    // so both canonical and quants-first q8_0 rows are supported here.
    const bool force_q8_gqa_tile =
        g_ggml_sycl_fa_q8_gqa_tile == 1 &&
        Q->ne[1] == 1 &&
        K->ne[0] == 128 &&
        K->type == GGML_TYPE_Q8_0 &&
        V->type == GGML_TYPE_Q8_0 &&
        gqa_opt_applies &&
        gqa_ratio >= 2;
    if (force_q8_gqa_tile) {
        return BEST_FATTN_KERNEL_TILE;
    }
    const bool force_vec_standard =
        g_ggml_sycl_fa_force_vec_standard == 1 &&
        Q->ne[1] == 1 &&
        K->ne[0] == 128 &&
        K->type == V->type &&
        (K->type == GGML_TYPE_F16 || K->type == GGML_TYPE_Q8_0) &&
        can_use_vector_kernel;
    if (force_vec_standard) {
        return BEST_FATTN_KERNEL_VEC;
    }


    // XMX (DPAS) path -- opt-in via GGML_SYCL_FA_XMX. Scope: f16 or q8_0 KV,
    // D in {128, 256}, additive mask (ne[2]==1). The math is validated
    // (docs/research/xmx_fa_*.cpp) and oracle-gated; off by default. D=512 is
    // excluded (staging tiles exceed the 64KB SLM budget).
    const bool xmx_kv_ok =
        (K->type == GGML_TYPE_F16 || K->type == GGML_TYPE_Q8_0) &&
        K->type == V->type &&
        !ggml_sycl_tensor_is_kv_q8_quants_first(K) &&
        !ggml_sycl_tensor_is_kv_q8_quants_first(V);
    if (getenv("GGML_SYCL_FA_XMX") && xmx_features_ok && xmx_kv_ok && (Q->ne[0] == 128 || Q->ne[0] == 256)) {
        return BEST_FATTN_KERNEL_XMX;
    }

    // Fused-XMX path: oneDNN Graph SDPA (flash attention). Strictly
    // additive -- taken only when statically supported, otherwise falls through to VEC/TILE below.
    if (ggml_sycl_flash_attn_ext_onednn_supported(dst)) {
        return BEST_FATTN_KERNEL_ONEDNN;
    }

    // If there are no tensor cores available, use the generic tile kernel:
    if (can_use_vector_kernel) {
        if (!ggml_is_quantized(K->type) && !ggml_is_quantized(V->type)) {
            if (Q->ne[1] == 1) {
                if (!gqa_opt_applies) {
                    return BEST_FATTN_KERNEL_VEC;
                }
            }
        } else {
            if (Q->ne[1] <= 2) {
                // TILE is faster for quantized KV decode on Xe2 (BMG); keep VEC on untested archs
                const gpu_arch arch = ggml_sycl_info().devices[device].hw_info.arch;
                if (arch == gpu_arch::intel_gpu_bmg_g21 || arch == gpu_arch::intel_gpu_bmg_g31) {
                    return BEST_FATTN_KERNEL_TILE;
                }
                return BEST_FATTN_KERNEL_VEC;
            }
        }
    }
    if (V_is_turbo || K_is_turbo) {
        return BEST_FATTN_KERNEL_NONE;
    }
    return BEST_FATTN_KERNEL_TILE;
}

bool ggml_backend_sycl_flash_attn_ext_uses_mkl(int device, const ggml_tensor * op) {
    return op && op->op == GGML_OP_FLASH_ATTN_EXT &&
        ggml_sycl_get_best_fattn_kernel(device, op) == BEST_FATTN_KERNEL_MKL;
}

void ggml_sycl_flash_attn_ext(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_set_device(ctx.device);

    // sparse nodes are gathered down to n_kv_max rows and re-dispatched here
    if (ggml_sycl_flash_attn_ext_sparse(ctx, dst)) {
        return;
    }

    const best_fattn_kernel route =
        ggml_sycl_get_best_fattn_kernel(ggml_sycl_get_device(), dst);
    ggml_sycl_log_fattn_route_once(route, dst);

    // n_kv watchdog: log when n_kv differs from the last FA call with
    // the same D — helps detect cache-truncation issues.
    static int nkv_debug = ggml_sycl_get_env("GGML_SYCL_MKL_FA_DEBUG", 0);
    if (nkv_debug == 1) {
        const ggml_tensor * K_dbg = dst->src[1];
        const ggml_tensor * V_dbg = dst->src[2];
        static int64_t last_nkv_d256 = 0, last_nkv_d512 = 0;
        static int fa_call_seq = 0;
        fa_call_seq++;
        int64_t cur_nkv = K_dbg->ne[1];
        int Dk = (int)K_dbg->ne[0];
        const char * kname = "TILE";
        if (route == BEST_FATTN_KERNEL_MKL)  kname = "MKL";
        if (route == BEST_FATTN_KERNEL_ONEDNN)  kname = "ONEDNN";
        if (route == BEST_FATTN_KERNEL_VEC)  kname = "VEC";
        int64_t delta = 0;
        if (Dk == 256) {
            delta = cur_nkv - last_nkv_d256;
            last_nkv_d256 = cur_nkv;
        } else if (Dk == 512) {
            delta = cur_nkv - last_nkv_d512;
            last_nkv_d512 = cur_nkv;
        }
        GGML_LOG_INFO("[FA-DISP] #%d %s D=%d n_kv=%lld delta=%lld "
                "V_ne1=%lld\n",
                fa_call_seq, kname, Dk,
                (long long)cur_nkv, (long long)delta,
                (long long)V_dbg->ne[1]);
    }

    switch (route) {
        case BEST_FATTN_KERNEL_NONE:
            GGML_ABORT("Not support Flash-Attention");
        case BEST_FATTN_KERNEL_ONEDNN:
            // guarded: ggml_sycl_flash_attn_ext_onednn() is only defined under GGML_SYCL_DNNL;
            // the reference must be compiled out here or the GGML_SYCL_DNNL=0 build fails to link.
#if GGML_SYCL_DNNL
            ggml_sycl_flash_attn_ext_onednn(ctx, dst);
#endif
            break;
        case BEST_FATTN_KERNEL_TILE:
            ggml_sycl_flash_attn_ext_tile(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_VEC:
            ggml_sycl_flash_attn_ext_vec(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_XMX:
            ggml_sycl_flash_attn_ext_xmx(ctx, dst);
            break;
        case BEST_FATTN_KERNEL_MKL:
            ggml_sycl_flash_attn_ext_mkl(ctx, dst);
            break;
    }

    // --- Output fingerprint (GGML_SYCL_MKL_FA_DIAG=1) ---
    // Copy first 64 float output values to host for fingerprinting.
    // Compare MKL vs TILE (GGML_SYCL_ENABLE_MKL_FA=0) to detect divergence.
    // Only fingerprints the first 6 FA calls with n_kv >= 1024.
    static int fa_diag = ggml_sycl_get_env("GGML_SYCL_MKL_FA_DIAG", 0);
    static int fa_diag_count = 0;
    if (fa_diag == 1 && fa_diag_count < 6) {
        const ggml_tensor * K_diag = dst->src[1];
        const ggml_tensor * V_diag = dst->src[2];
        const ggml_tensor * Q_diag = dst->src[0];
        if (K_diag->ne[1] >= 1024) {
            fa_diag_count++;
            float diag_buf[64];
            dpct::queue_ptr q = ctx.stream();
            q->memcpy(diag_buf, dst->data, 64 * sizeof(float));
            q->wait();
            const char * kname = "???";
            best_fattn_kernel kb = ggml_sycl_get_best_fattn_kernel(ctx.device, dst);
            if (kb == BEST_FATTN_KERNEL_ONEDNN) kname = "ONEDNN";
            if (kb == BEST_FATTN_KERNEL_MKL) kname = "MKL";
            if (kb == BEST_FATTN_KERNEL_TILE) kname = "TILE";
            if (kb == BEST_FATTN_KERNEL_VEC) kname = "VEC";
            GGML_LOG_INFO("[FA-DIAG] #%d %s D=%d n_kv=%lld n_q=%lld "
                    "n_qh=%lld n_kvh=%lld K=%s V=%s "
                    "nb1=%zu nb2=%zu first 64 floats:\n",
                    fa_diag_count, kname,
                    (int)K_diag->ne[0], (long long)K_diag->ne[1],
                    (long long)Q_diag->ne[1],
                    (long long)Q_diag->ne[2], (long long)K_diag->ne[2],
                    ggml_type_name(K_diag->type),
                    ggml_type_name(V_diag->type),
                    K_diag->nb[1], K_diag->nb[2]);
            for (int i = 0; i < 64; i += 8) {
                GGML_LOG_INFO("  [%2d] %08x %08x %08x %08x %08x %08x %08x %08x\n",
                        i,
                        *(unsigned *)&diag_buf[i+0], *(unsigned *)&diag_buf[i+1],
                        *(unsigned *)&diag_buf[i+2], *(unsigned *)&diag_buf[i+3],
                        *(unsigned *)&diag_buf[i+4], *(unsigned *)&diag_buf[i+5],
                        *(unsigned *)&diag_buf[i+6], *(unsigned *)&diag_buf[i+7]);
            }
        }
    }

}

bool ggml_sycl_flash_attn_ext_supported(int device, const ggml_tensor * dst) {
    // Turbo KV runs on the VEC kernel. The historical veto here (IGC hang + garbage
    // output) traced to vec_dot_fattn_vec_KQ_turbo_generic reading Q_v as a full
    // row instead of the caller's per-thread register slice; fixed in fattn-common.hpp.
    return ggml_sycl_get_best_fattn_kernel(device, dst) != BEST_FATTN_KERNEL_NONE;
}

bool ggml_sycl_flash_attn_ext_uses_library(int device, const ggml_tensor * dst) {
    const best_fattn_kernel kernel = ggml_sycl_get_best_fattn_kernel(device, dst);
    return kernel == BEST_FATTN_KERNEL_ONEDNN || kernel == BEST_FATTN_KERNEL_MKL;
}

static uintptr_t ggml_sycl_fattn_reserve_halves(ggml_sycl_fattn_extra & extra, size_t n_halves) {
    if (n_halves == 0) {
        return 0;
    }
    extra.end = GGML_PAD(extra.end, SYCL_BUFFER_ALIGNMENT);
    const uintptr_t block = extra.end;
    extra.end += n_halves * sizeof(sycl::half);
    return block;
}

ggml_sycl_fattn_extra ggml_sycl_fattn_get_extra(const ggml_tensor * dst) {
    ggml_sycl_fattn_extra extra;

    extra.end = (uintptr_t) dst->data + ggml_nbytes(dst);

    if (dst->op != GGML_OP_FLASH_ATTN_EXT) {
        return extra;
    }

    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    if (!Q || !K || !V) {
        return extra;
    }

    const int64_t d = K->ne[0];
    const int64_t H = Q->ne[2];
    const int64_t q = Q->ne[1];

    // calculate the worst-case memory consumption across all kernels
    const bool onednn_supported = ggml_sycl_flash_attn_ext_onednn_supported(dst, /* use_shape_limit */ false);

    const bool tile_needs_K = K->type != GGML_TYPE_F16;
    const bool tile_needs_V = V->type != GGML_TYPE_F16;

    const bool V_is_K_view = V->view_src &&
        (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));

    size_t need_K = 0, need_V = 0, need_Q = 0, need_out = 0, need_scale = 0;
    if (onednn_supported) {
        need_Q     = (size_t) H * q * d;
        need_out   = (size_t) H * q * d;
        need_scale = 1;
        // an f16 cache is bound in place, so it needs no staging copy
        if (!ggml_sycl_fattn_onednn_binds_kv(K, V)) {
            need_K = (size_t) ggml_nelements(K);
            need_V = (size_t) ggml_nelements(V);
        }
    }
    if (tile_needs_K) {
        need_K = std::max(need_K, (size_t) ggml_nelements(K));
    }
    if (tile_needs_V) {
        need_V = std::max(need_V, (size_t) ggml_nelements(V));
    }

    extra.Q_buffer_ptr = ggml_sycl_fattn_reserve_halves(extra, need_Q);
    extra.K_buffer_ptr = ggml_sycl_fattn_reserve_halves(extra, need_K);
    extra.V_buffer_ptr = (V_is_K_view && !onednn_supported && need_V)
                       ? extra.K_buffer_ptr
                       : ggml_sycl_fattn_reserve_halves(extra, need_V);
    extra.scale_buffer_ptr = ggml_sycl_fattn_reserve_halves(extra, need_scale);
    extra.out_buffer_ptr   = ggml_sycl_fattn_reserve_halves(extra, need_out);

    return extra;
}

size_t ggml_sycl_flash_attn_ext_get_alloc_size(const ggml_tensor * dst) {
    const ggml_sycl_fattn_extra extra = ggml_sycl_fattn_get_extra(dst);
    return (size_t) (extra.end - (uintptr_t) dst->data);
}
