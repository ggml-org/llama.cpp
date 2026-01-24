#pragma once

#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

#include <sycl/sycl.hpp>

#include "../../dpas_config.hpp"

#if __has_include(<sycl/ext/intel/esimd.hpp>) && __has_include(<sycl/ext/intel/esimd/xmx/dpas.hpp>)
#    include <sycl/ext/intel/esimd.hpp>
#    include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#    define DPAS_EXPLORATION_ESIMD_AVAILABLE 1
#else
#    define DPAS_EXPLORATION_ESIMD_AVAILABLE 0
#endif

#if defined(__has_cpp_attribute)
#    if __has_cpp_attribute(intel::num_registers)
#        define DPAS_NUM_REGS(N) [[intel::num_registers(N)]]
#    else
#        define DPAS_NUM_REGS(N)
#    endif
#elif defined(__INTEL_LLVM_COMPILER)
#    define DPAS_NUM_REGS(N) [[intel::num_registers(N)]]
#else
#    define DPAS_NUM_REGS(N)
#endif

namespace sycl_bench {

struct DpasBenchArgs {
    const void * a = nullptr;
    const void * b = nullptr;
    void *       c = nullptr;
    int64_t      m = 0;
    int64_t      n = 0;
    int64_t      k = 0;
    DpasType     type_a = DpasType::INT8;
    DpasType     type_b = DpasType::INT8;
    DpasAccType  type_acc = DpasAccType::INT32;
    DpasMemoryPattern memory_pattern = DpasMemoryPattern::DIRECT_GLOBAL;
    DpasGrfMode  grf_mode = DpasGrfMode::GRF_128;
    int          repeat = 8;
    int          n_tile_repeats = 1;
    bool         misaligned = false;
    sycl::queue * stream = nullptr;
};

template <typename TA, typename TB, typename TAcc, int Repeat, DpasMemoryPattern Pattern, DpasGrfMode GrfMode>
struct dpas_kernel_tag {};

inline int dpas_k_per_tile(DpasType type_a, DpasType type_b) {
    const int bits_a = (type_a == DpasType::INT8) ? 8 : 16;
    const int bits_b = (type_b == DpasType::INT8) ? 8 : 16;
    const int max_bits = (bits_a > bits_b) ? bits_a : bits_b;
    const int max_elems = 32 / max_bits;
    const int ops_per_channel = (max_elems > 8) ? 8 : (max_elems < 1 ? 1 : max_elems);
    return 8 * ops_per_channel;
}

template <typename T>
constexpr int dpas_elem_bits() {
    return static_cast<int>(sizeof(T) * 8);
}

template <typename TA, typename TB>
constexpr int dpas_k_per_tile_typed() {
    constexpr int bits_a = dpas_elem_bits<TA>();
    constexpr int bits_b = dpas_elem_bits<TB>();
    constexpr int max_bits = (bits_a > bits_b) ? bits_a : bits_b;
    constexpr int max_elems = 32 / max_bits;
    constexpr int ops_per_channel = (max_elems > 8) ? 8 : (max_elems < 1 ? 1 : max_elems);
    return 8 * ops_per_channel;
}

template <typename T, int N, DpasMemoryPattern Pattern>
SYCL_ESIMD_FUNCTION inline sycl::ext::intel::esimd::simd<T, N> dpas_block_load(const T * ptr, bool misaligned) {
    using namespace sycl::ext::intel::esimd;
    constexpr bool use_streaming =
        (Pattern == DpasMemoryPattern::LSC_STREAMING || Pattern == DpasMemoryPattern::LSC_PREFETCH ||
         Pattern == DpasMemoryPattern::LSC_PREFETCH_2 || Pattern == DpasMemoryPattern::LSC_PREFETCH_3 ||
         Pattern == DpasMemoryPattern::LSC_PREFETCH_4 || Pattern == DpasMemoryPattern::LSC_PREFETCH_5 ||
         Pattern == DpasMemoryPattern::LSC_PREFETCH_6 || Pattern == DpasMemoryPattern::LSC_PREFETCH_8 ||
         Pattern == DpasMemoryPattern::LSC_PREFETCH_10);
    const auto load_props_aligned_16 =
        properties{alignment<16>, cache_hint_L1<cache_hint::cached>, cache_hint_L2<cache_hint::cached>};
    const auto load_props_aligned_64 =
        properties{alignment<64>, cache_hint_L1<cache_hint::cached>, cache_hint_L2<cache_hint::cached>};
    const auto load_props_streaming_16 =
        properties{alignment<16>, cache_hint_L1<cache_hint::streaming>, cache_hint_L2<cache_hint::uncached>};
    const auto load_props_streaming_64 =
        properties{alignment<64>, cache_hint_L1<cache_hint::streaming>, cache_hint_L2<cache_hint::uncached>};

    if (misaligned) {
        simd<T, N> tmp;
#pragma unroll
        for (int i = 0; i < N; ++i) {
            tmp[i] = ptr[i];
        }
        return tmp;
    }

    const uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    const bool aligned64 = (addr & 63) == 0;

    if constexpr (use_streaming && N <= 64) {
        return aligned64 ? block_load<T, N>(ptr, load_props_streaming_64)
                         : block_load<T, N>(ptr, load_props_streaming_16);
    }

    return aligned64 ? block_load<T, N>(ptr, load_props_aligned_64)
                     : block_load<T, N>(ptr, load_props_aligned_16);
}

template <typename T>
SYCL_ESIMD_FUNCTION inline void dpas_prefetch_streaming(const T * ptr) {
    using namespace sycl::ext::intel::esimd;
    const uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    if ((addr & 0x3u) != 0u) {
        return;
    }
    auto ptr_u32 = reinterpret_cast<const uint32_t *>(ptr);
    if ((addr & 63u) == 0u) {
        prefetch(ptr_u32,
                 properties{cache_hint_L1<cache_hint::streaming>,
                            cache_hint_L2<cache_hint::uncached>,
                            alignment<64>});
        return;
    }
    if ((addr & 15u) == 0u) {
        prefetch(ptr_u32,
                 properties{cache_hint_L1<cache_hint::streaming>,
                            cache_hint_L2<cache_hint::uncached>,
                            alignment<16>});
        return;
    }
    prefetch(ptr_u32,
             properties{cache_hint_L1<cache_hint::streaming>,
                        cache_hint_L2<cache_hint::uncached>,
                        alignment<4>});
}

template <typename TA, typename TB, typename TAcc, int Repeat, DpasMemoryPattern Pattern, DpasGrfMode GrfMode>
inline bool launch_dpas_kernel(const DpasBenchArgs & args,
                               std::vector<sycl::event> * events,
                               std::string & error) {
#if !DPAS_EXPLORATION_ESIMD_AVAILABLE
    (void)args;
    (void)events;
    error = "SYCL ESIMD unavailable; dpas exploration kernels disabled.";
    return false;
#else
    if (!args.stream) {
        error = "SYCL stream is null.";
        return false;
    }
    constexpr int ExecN = 16;
    constexpr int KPer = dpas_k_per_tile_typed<TA, TB>();
    constexpr int AN = Repeat * KPer;
    constexpr int BN = KPer * ExecN;

    if (args.m % Repeat != 0 || args.n % ExecN != 0 || args.k % KPer != 0) {
        error = "DPAS dims must be multiples of repeat, 16, and K tile.";
        return false;
    }

    const int64_t m_tiles = args.m / Repeat;
    const int n_tile_repeats = args.n_tile_repeats;
    const int64_t n_tiles = args.n / ExecN;
    const int64_t n_tile_groups = n_tiles / n_tile_repeats;
    const int64_t k_tiles = args.k / KPer;

    const auto * a_base = static_cast<const TA *>(args.a);
    const auto * b_base = static_cast<const TB *>(args.b);
    auto * c_base = static_cast<TAcc *>(args.c);

    if (!a_base || !b_base || !c_base) {
        error = "DPAS buffers are null.";
        return false;
    }

    if (n_tile_repeats != 1 && n_tile_repeats != 2 && n_tile_repeats != 4 && n_tile_repeats != 8) {
        error = "DPAS n-tile repeats must be 1, 2, 4, or 8.";
        return false;
    }
    if constexpr (Pattern == DpasMemoryPattern::SLM_BUFFER || Pattern == DpasMemoryPattern::DOUBLE_BUFFER) {
        if (n_tile_repeats > 4) {
            error = "DPAS SLM patterns only support n-tile repeats = 1, 2, or 4.";
            return false;
        }
    }
    constexpr uint32_t slm_a_bytes = static_cast<uint32_t>(AN * sizeof(TA));
    constexpr uint32_t slm_b_bytes = static_cast<uint32_t>(BN * sizeof(TB));
    constexpr uint32_t slm_pad = 64;
    constexpr uint32_t slm_a_offset = 0;
    constexpr uint32_t slm_b_offset = ((slm_a_bytes + slm_pad - 1) / slm_pad) * slm_pad;
    constexpr uint32_t slm_b_stride = ((slm_b_bytes + slm_pad - 1) / slm_pad) * slm_pad;
    constexpr uint32_t slm_tile_bytes_4 = slm_b_offset + slm_b_stride * 4;
    constexpr uint32_t slm_total_bytes_max =
        (Pattern == DpasMemoryPattern::DOUBLE_BUFFER) ? slm_tile_bytes_4 * 2 : slm_tile_bytes_4;
    const uint32_t slm_tile_bytes =
        slm_b_offset + slm_b_stride * static_cast<uint32_t>(n_tile_repeats);

    const int64_t tile_count = m_tiles * n_tile_groups;
    sycl::range<1> grid(static_cast<size_t>(tile_count));
    sycl::range<1> block(1);

    using KernelTag = dpas_kernel_tag<TA, TB, TAcc, Repeat, Pattern, GrfMode>;
    [[maybe_unused]] constexpr int NumRegs = (GrfMode == DpasGrfMode::GRF_256) ? 256 : 128;

    sycl::event ev = args.stream->submit([&](sycl::handler & h) {
        h.parallel_for<KernelTag>(
            sycl::nd_range<1>(grid, block),
            [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL DPAS_NUM_REGS(NumRegs) {
                if constexpr (Pattern == DpasMemoryPattern::SLM_BUFFER ||
                              Pattern == DpasMemoryPattern::DOUBLE_BUFFER) {
                    sycl::ext::intel::esimd::slm_init<slm_total_bytes_max>();
                }
                const int64_t tile_idx = static_cast<int64_t>(item.get_global_id(0));
                (void)tile_count;
                const int64_t tile_m = tile_idx / n_tile_groups;
                const int64_t tile_group_n = tile_idx - tile_m * n_tile_groups;
                const int64_t tile_n_base = tile_group_n * n_tile_repeats;

                using namespace sycl::ext::intel::esimd;
                if (n_tile_repeats == 1) {
                    simd<TAcc, Repeat * ExecN> acc = TAcc(0);
                    simd<TA, AN> a_vec;
                    simd<TB, BN> b_vec;

                    if constexpr (Pattern == DpasMemoryPattern::REG_PREFETCH) {
                        const TA * a_ptr = a_base + (tile_m * k_tiles) * AN;
                        const TB * b_ptr = b_base + (tile_n_base * k_tiles) * BN;
                        a_vec = dpas_block_load<TA, AN, Pattern>(a_ptr, args.misaligned);
                        b_vec = dpas_block_load<TB, BN, Pattern>(b_ptr, args.misaligned);
                        simd<TA, AN> a_next;
                        simd<TB, BN> b_next;
                        const bool has_next = k_tiles > 1;
                        if (has_next) {
                            a_next = dpas_block_load<TA, AN, Pattern>(a_ptr + AN, args.misaligned);
                            b_next = dpas_block_load<TB, BN, Pattern>(b_ptr + BN, args.misaligned);
                        }
                        for (int64_t kt = 0; kt < k_tiles; ++kt) {
                            acc = xmx::dpas<8, Repeat, TAcc, TAcc, TB, TA>(acc, b_vec, a_vec);
                            const bool has_next2 = (kt + 2) < k_tiles;
                            if (has_next2) {
                                simd<TA, AN> a_next2 =
                                    dpas_block_load<TA, AN, Pattern>(a_ptr + 2 * AN, args.misaligned);
                                simd<TB, BN> b_next2 =
                                    dpas_block_load<TB, BN, Pattern>(b_ptr + 2 * BN, args.misaligned);
                                a_vec = a_next;
                                b_vec = b_next;
                                a_next = a_next2;
                                b_next = b_next2;
                                a_ptr += AN;
                                b_ptr += BN;
                            } else if ((kt + 1) < k_tiles) {
                                a_vec = a_next;
                                b_vec = b_next;
                                a_ptr += AN;
                                b_ptr += BN;
                            }
                        }
                    } else if constexpr (Pattern == DpasMemoryPattern::DOUBLE_BUFFER) {
                        int buf_idx = 0;
                        const TA * a_ptr = a_base + (tile_m * k_tiles) * AN;
                        const TB * b_ptr = b_base + (tile_n_base * k_tiles) * BN;
                        {
                            auto a_tmp = dpas_block_load<TA, AN, Pattern>(a_ptr, args.misaligned);
                            auto b_tmp = dpas_block_load<TB, BN, Pattern>(b_ptr, args.misaligned);
                            slm_block_store<TA, AN>(slm_a_offset, a_tmp);
                            slm_block_store<TB, BN>(slm_b_offset, b_tmp);
                        }
                        for (int64_t kt = 0; kt < k_tiles; ++kt) {
                            const uint32_t slm_offset = (buf_idx == 0) ? 0 : slm_tile_bytes;
                            a_vec = slm_block_load<TA, AN>(slm_offset + slm_a_offset);
                            b_vec = slm_block_load<TB, BN>(slm_offset + slm_b_offset);
                            acc = xmx::dpas<8, Repeat, TAcc, TAcc, TB, TA>(acc, b_vec, a_vec);
                            const bool has_next = (kt + 1) < k_tiles;
                            if (has_next) {
                                const uint32_t next_offset = (buf_idx == 0) ? slm_tile_bytes : 0;
                                auto a_next = dpas_block_load<TA, AN, Pattern>(a_ptr + AN, args.misaligned);
                                auto b_next = dpas_block_load<TB, BN, Pattern>(b_ptr + BN, args.misaligned);
                                slm_block_store<TA, AN>(next_offset + slm_a_offset, a_next);
                                slm_block_store<TB, BN>(next_offset + slm_b_offset, b_next);
                            }
                            buf_idx ^= 1;
                            a_ptr += AN;
                            b_ptr += BN;
                        }
                    } else {
                        const TA * a_ptr = a_base + (tile_m * k_tiles) * AN;
                        const TB * b_ptr = b_base + (tile_n_base * k_tiles) * BN;
                        for (int64_t kt = 0; kt < k_tiles; ++kt) {
                            if constexpr (Pattern == DpasMemoryPattern::LSC_PREFETCH ||
                                          Pattern == DpasMemoryPattern::LSC_PREFETCH_2 ||
                                          Pattern == DpasMemoryPattern::LSC_PREFETCH_3 ||
                                          Pattern == DpasMemoryPattern::LSC_PREFETCH_4 ||
                                          Pattern == DpasMemoryPattern::LSC_PREFETCH_5 ||
                                          Pattern == DpasMemoryPattern::LSC_PREFETCH_6 ||
                                          Pattern == DpasMemoryPattern::LSC_PREFETCH_8 ||
                                          Pattern == DpasMemoryPattern::LSC_PREFETCH_10) {
                                constexpr int PrefetchDist =
                                    (Pattern == DpasMemoryPattern::LSC_PREFETCH_10) ? 10
                                    : (Pattern == DpasMemoryPattern::LSC_PREFETCH_8) ? 8
                                    : (Pattern == DpasMemoryPattern::LSC_PREFETCH_6) ? 6
                                    : (Pattern == DpasMemoryPattern::LSC_PREFETCH_5) ? 5
                                    : (Pattern == DpasMemoryPattern::LSC_PREFETCH_4) ? 4
                                    : (Pattern == DpasMemoryPattern::LSC_PREFETCH_3) ? 3
                                    : (Pattern == DpasMemoryPattern::LSC_PREFETCH_2) ? 2
                                                                                      : 1;
                                const int64_t next = kt + PrefetchDist;
                                if (next < k_tiles) {
                                    const TA * pf_a = a_ptr + PrefetchDist * AN;
                                    const TB * pf_b = b_ptr + PrefetchDist * BN;
                                    dpas_prefetch_streaming(pf_a);
                                    dpas_prefetch_streaming(pf_b);
                                }
                            }

                            if constexpr (Pattern == DpasMemoryPattern::SLM_BUFFER) {
                                auto a_tmp = dpas_block_load<TA, AN, Pattern>(a_ptr, args.misaligned);
                                auto b_tmp = dpas_block_load<TB, BN, Pattern>(b_ptr, args.misaligned);
                                slm_block_store<TA, AN>(slm_a_offset, a_tmp);
                                slm_block_store<TB, BN>(slm_b_offset, b_tmp);
                                a_vec = slm_block_load<TA, AN>(slm_a_offset);
                                b_vec = slm_block_load<TB, BN>(slm_b_offset);
                            } else {
                                a_vec = dpas_block_load<TA, AN, Pattern>(a_ptr, args.misaligned);
                                b_vec = dpas_block_load<TB, BN, Pattern>(b_ptr, args.misaligned);
                            }

                            acc = xmx::dpas<8, Repeat, TAcc, TAcc, TB, TA>(acc, b_vec, a_vec);
                            a_ptr += AN;
                            b_ptr += BN;
                        }
                    }

                    for (int r = 0; r < Repeat; ++r) {
                        auto * out_ptr = c_base + (tile_m * Repeat + r) * args.n + tile_n_base * ExecN;
                        simd<TAcc, ExecN> row = acc.template select<ExecN, 1>(r * ExecN);
                        block_store(out_ptr, row);
                    }
                } else if (n_tile_repeats == 2) {
                    simd<TAcc, Repeat * ExecN> acc0 = TAcc(0);
                    simd<TAcc, Repeat * ExecN> acc1 = TAcc(0);
                    const TA * a_ptr = a_base + (tile_m * k_tiles) * AN;
                    const TB * b_ptr0 = b_base + (tile_n_base * k_tiles) * BN;
                    const TB * b_ptr1 = b_ptr0 + BN * k_tiles;

                    if constexpr (Pattern == DpasMemoryPattern::DOUBLE_BUFFER) {
                        int buf_idx = 0;
                        {
                            auto a_tmp = dpas_block_load<TA, AN, Pattern>(a_ptr, args.misaligned);
                            auto b_tmp0 = dpas_block_load<TB, BN, Pattern>(b_ptr0, args.misaligned);
                            auto b_tmp1 = dpas_block_load<TB, BN, Pattern>(b_ptr1, args.misaligned);
                            slm_block_store<TA, AN>(slm_a_offset, a_tmp);
                            slm_block_store<TB, BN>(slm_b_offset, b_tmp0);
                            slm_block_store<TB, BN>(slm_b_offset + slm_b_stride, b_tmp1);
                        }
                        for (int64_t kt = 0; kt < k_tiles; ++kt) {
                            const uint32_t slm_offset = (buf_idx == 0) ? 0 : slm_tile_bytes;
                            simd<TA, AN> a_vec = slm_block_load<TA, AN>(slm_offset + slm_a_offset);
                            simd<TB, BN> b_vec0 = slm_block_load<TB, BN>(slm_offset + slm_b_offset);
                            simd<TB, BN> b_vec1 = slm_block_load<TB, BN>(slm_offset + slm_b_offset + slm_b_stride);
                            acc0 = xmx::dpas<8, Repeat, TAcc, TAcc, TB, TA>(acc0, b_vec0, a_vec);
                            acc1 = xmx::dpas<8, Repeat, TAcc, TAcc, TB, TA>(acc1, b_vec1, a_vec);
                            const bool has_next = (kt + 1) < k_tiles;
                            if (has_next) {
                                const uint32_t next_offset = (buf_idx == 0) ? slm_tile_bytes : 0;
                                auto a_next = dpas_block_load<TA, AN, Pattern>(a_ptr + AN, args.misaligned);
                                auto b_next0 = dpas_block_load<TB, BN, Pattern>(b_ptr0 + BN, args.misaligned);
                                auto b_next1 = dpas_block_load<TB, BN, Pattern>(b_ptr1 + BN, args.misaligned);
                                slm_block_store<TA, AN>(next_offset + slm_a_offset, a_next);
                                slm_block_store<TB, BN>(next_offset + slm_b_offset, b_next0);
                                slm_block_store<TB, BN>(next_offset + slm_b_offset + slm_b_stride, b_next1);
                            }
                            buf_idx ^= 1;
                            a_ptr += AN;
                            b_ptr0 += BN;
                            b_ptr1 += BN;
                        }
                    } else if constexpr (Pattern == DpasMemoryPattern::REG_PREFETCH) {
                        simd<TA, AN> a_vec = dpas_block_load<TA, AN, Pattern>(a_ptr, args.misaligned);
                        simd<TB, BN> b_vec0 = dpas_block_load<TB, BN, Pattern>(b_ptr0, args.misaligned);
                        simd<TB, BN> b_vec1 = dpas_block_load<TB, BN, Pattern>(b_ptr1, args.misaligned);
                        simd<TA, AN> a_next;
                        simd<TB, BN> b_next0;
                        simd<TB, BN> b_next1;
                        if (k_tiles > 1) {
                            a_next = dpas_block_load<TA, AN, Pattern>(a_ptr + AN, args.misaligned);
                            b_next0 = dpas_block_load<TB, BN, Pattern>(b_ptr0 + BN, args.misaligned);
                            b_next1 = dpas_block_load<TB, BN, Pattern>(b_ptr1 + BN, args.misaligned);
                        }
                        for (int64_t kt = 0; kt < k_tiles; ++kt) {
                            acc0 = xmx::dpas<8, Repeat, TAcc, TAcc, TB, TA>(acc0, b_vec0, a_vec);
                            acc1 = xmx::dpas<8, Repeat, TAcc, TAcc, TB, TA>(acc1, b_vec1, a_vec);
                            if ((kt + 1) < k_tiles) {
                                const bool has_next2 = (kt + 2) < k_tiles;
                                if (has_next2) {
                                    simd<TA, AN> a_next2 =
                                        dpas_block_load<TA, AN, Pattern>(a_ptr + 2 * AN, args.misaligned);
                                    simd<TB, BN> b_next2_0 =
                                        dpas_block_load<TB, BN, Pattern>(b_ptr0 + 2 * BN, args.misaligned);
                                    simd<TB, BN> b_next2_1 =
                                        dpas_block_load<TB, BN, Pattern>(b_ptr1 + 2 * BN, args.misaligned);
                                    a_vec = a_next;
                                    b_vec0 = b_next0;
                                    b_vec1 = b_next1;
                                    a_next = a_next2;
                                    b_next0 = b_next2_0;
                                    b_next1 = b_next2_1;
                                } else {
                                    a_vec = a_next;
                                    b_vec0 = b_next0;
                                    b_vec1 = b_next1;
                                }
                                a_ptr += AN;
                                b_ptr0 += BN;
                                b_ptr1 += BN;
                            }
                        }
                    } else {
                        for (int64_t kt = 0; kt < k_tiles; ++kt) {
                            if constexpr (Pattern == DpasMemoryPattern::LSC_PREFETCH ||
                                          Pattern == DpasMemoryPattern::LSC_PREFETCH_2 ||
                                          Pattern == DpasMemoryPattern::LSC_PREFETCH_3 ||
                                          Pattern == DpasMemoryPattern::LSC_PREFETCH_4 ||
                                          Pattern == DpasMemoryPattern::LSC_PREFETCH_5 ||
                                          Pattern == DpasMemoryPattern::LSC_PREFETCH_6 ||
                                          Pattern == DpasMemoryPattern::LSC_PREFETCH_8 ||
                                          Pattern == DpasMemoryPattern::LSC_PREFETCH_10) {
                                constexpr int PrefetchDist =
                                    (Pattern == DpasMemoryPattern::LSC_PREFETCH_10) ? 10
                                    : (Pattern == DpasMemoryPattern::LSC_PREFETCH_8) ? 8
                                    : (Pattern == DpasMemoryPattern::LSC_PREFETCH_6) ? 6
                                    : (Pattern == DpasMemoryPattern::LSC_PREFETCH_5) ? 5
                                    : (Pattern == DpasMemoryPattern::LSC_PREFETCH_4) ? 4
                                    : (Pattern == DpasMemoryPattern::LSC_PREFETCH_3) ? 3
                                    : (Pattern == DpasMemoryPattern::LSC_PREFETCH_2) ? 2
                                                                                      : 1;
                                const int64_t next = kt + PrefetchDist;
                                if (next < k_tiles) {
                                    const TA * pf_a = a_ptr + PrefetchDist * AN;
                                    const TB * pf_b0 = b_ptr0 + PrefetchDist * BN;
                                    const TB * pf_b1 = b_ptr1 + PrefetchDist * BN;
                                    dpas_prefetch_streaming(pf_a);
                                    dpas_prefetch_streaming(pf_b0);
                                    dpas_prefetch_streaming(pf_b1);
                                }
                            }

                            simd<TA, AN> a_vec;
                            simd<TB, BN> b_vec0;
                            simd<TB, BN> b_vec1;
                            if constexpr (Pattern == DpasMemoryPattern::SLM_BUFFER) {
                                auto a_tmp = dpas_block_load<TA, AN, Pattern>(a_ptr, args.misaligned);
                                auto b_tmp0 = dpas_block_load<TB, BN, Pattern>(b_ptr0, args.misaligned);
                                auto b_tmp1 = dpas_block_load<TB, BN, Pattern>(b_ptr1, args.misaligned);
                                slm_block_store<TA, AN>(slm_a_offset, a_tmp);
                                slm_block_store<TB, BN>(slm_b_offset, b_tmp0);
                                slm_block_store<TB, BN>(slm_b_offset + slm_b_stride, b_tmp1);
                                a_vec = slm_block_load<TA, AN>(slm_a_offset);
                                b_vec0 = slm_block_load<TB, BN>(slm_b_offset);
                                b_vec1 = slm_block_load<TB, BN>(slm_b_offset + slm_b_stride);
                            } else {
                                a_vec = dpas_block_load<TA, AN, Pattern>(a_ptr, args.misaligned);
                                b_vec0 = dpas_block_load<TB, BN, Pattern>(b_ptr0, args.misaligned);
                                b_vec1 = dpas_block_load<TB, BN, Pattern>(b_ptr1, args.misaligned);
                            }
                            acc0 = xmx::dpas<8, Repeat, TAcc, TAcc, TB, TA>(acc0, b_vec0, a_vec);
                            acc1 = xmx::dpas<8, Repeat, TAcc, TAcc, TB, TA>(acc1, b_vec1, a_vec);
                            a_ptr += AN;
                            b_ptr0 += BN;
                            b_ptr1 += BN;
                        }
                    }

                    for (int r = 0; r < Repeat; ++r) {
                        auto * out_ptr0 = c_base + (tile_m * Repeat + r) * args.n + tile_n_base * ExecN;
                        auto * out_ptr1 = out_ptr0 + ExecN;
                        simd<TAcc, ExecN> row0 = acc0.template select<ExecN, 1>(r * ExecN);
                        simd<TAcc, ExecN> row1 = acc1.template select<ExecN, 1>(r * ExecN);
                        block_store(out_ptr0, row0);
                        block_store(out_ptr1, row1);
                    }
                } else if (n_tile_repeats == 4) {
                    simd<TAcc, Repeat * ExecN> acc[4] = { TAcc(0), TAcc(0), TAcc(0), TAcc(0) };
                    const TA * a_ptr = a_base + (tile_m * k_tiles) * AN;
                    const TB * b_ptr[4];
                    b_ptr[0] = b_base + (tile_n_base * k_tiles) * BN;
                    b_ptr[1] = b_ptr[0] + BN * k_tiles;
                    b_ptr[2] = b_ptr[1] + BN * k_tiles;
                    b_ptr[3] = b_ptr[2] + BN * k_tiles;
                    if constexpr (Pattern == DpasMemoryPattern::DOUBLE_BUFFER) {
                        int buf_idx = 0;
                        {
                            auto a_tmp = dpas_block_load<TA, AN, Pattern>(a_ptr, args.misaligned);
                            slm_block_store<TA, AN>(slm_a_offset, a_tmp);
#pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                auto b_tmp = dpas_block_load<TB, BN, Pattern>(b_ptr[i], args.misaligned);
                                slm_block_store<TB, BN>(slm_b_offset + i * slm_b_stride, b_tmp);
                            }
                        }
                        for (int64_t kt = 0; kt < k_tiles; ++kt) {
                            const uint32_t slm_offset = (buf_idx == 0) ? 0 : slm_tile_bytes;
                            simd<TA, AN> a_vec = slm_block_load<TA, AN>(slm_offset + slm_a_offset);
                            simd<TB, BN> b_vec[4];
#pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                b_vec[i] = slm_block_load<TB, BN>(slm_offset + slm_b_offset + i * slm_b_stride);
                            }
#pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                acc[i] = xmx::dpas<8, Repeat, TAcc, TAcc, TB, TA>(acc[i], b_vec[i], a_vec);
                            }
                            const bool has_next = (kt + 1) < k_tiles;
                            if (has_next) {
                                const uint32_t next_offset = (buf_idx == 0) ? slm_tile_bytes : 0;
                                auto a_next = dpas_block_load<TA, AN, Pattern>(a_ptr + AN, args.misaligned);
                                slm_block_store<TA, AN>(next_offset + slm_a_offset, a_next);
#pragma unroll
                                for (int i = 0; i < 4; ++i) {
                                    auto b_next = dpas_block_load<TB, BN, Pattern>(b_ptr[i] + BN, args.misaligned);
                                    slm_block_store<TB, BN>(next_offset + slm_b_offset + i * slm_b_stride, b_next);
                                }
                            }
                            buf_idx ^= 1;
                            a_ptr += AN;
#pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                b_ptr[i] += BN;
                            }
                        }
                    } else if constexpr (Pattern == DpasMemoryPattern::REG_PREFETCH) {
                        simd<TA, AN> a_vec = dpas_block_load<TA, AN, Pattern>(a_ptr, args.misaligned);
                        simd<TB, BN> b_vec[4];
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            b_vec[i] = dpas_block_load<TB, BN, Pattern>(b_ptr[i], args.misaligned);
                        }
                        simd<TA, AN> a_next;
                        simd<TB, BN> b_next[4];
                        if (k_tiles > 1) {
                            a_next = dpas_block_load<TA, AN, Pattern>(a_ptr + AN, args.misaligned);
#pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                b_next[i] = dpas_block_load<TB, BN, Pattern>(b_ptr[i] + BN, args.misaligned);
                            }
                        }
                        for (int64_t kt = 0; kt < k_tiles; ++kt) {
#pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                acc[i] = xmx::dpas<8, Repeat, TAcc, TAcc, TB, TA>(acc[i], b_vec[i], a_vec);
                            }
                            if ((kt + 1) < k_tiles) {
                                const bool has_next2 = (kt + 2) < k_tiles;
                                if (has_next2) {
                                    simd<TA, AN> a_next2 =
                                        dpas_block_load<TA, AN, Pattern>(a_ptr + 2 * AN, args.misaligned);
                                    simd<TB, BN> b_next2[4];
#pragma unroll
                                    for (int i = 0; i < 4; ++i) {
                                        b_next2[i] =
                                            dpas_block_load<TB, BN, Pattern>(b_ptr[i] + 2 * BN, args.misaligned);
                                    }
                                    a_vec = a_next;
#pragma unroll
                                    for (int i = 0; i < 4; ++i) {
                                        b_vec[i] = b_next[i];
                                    }
                                    a_next = a_next2;
#pragma unroll
                                    for (int i = 0; i < 4; ++i) {
                                        b_next[i] = b_next2[i];
                                    }
                                } else {
                                    a_vec = a_next;
#pragma unroll
                                    for (int i = 0; i < 4; ++i) {
                                        b_vec[i] = b_next[i];
                                    }
                                }
                                a_ptr += AN;
#pragma unroll
                                for (int i = 0; i < 4; ++i) {
                                    b_ptr[i] += BN;
                                }
                            }
                        }
                    } else {
                        for (int64_t kt = 0; kt < k_tiles; ++kt) {
                            if constexpr (Pattern == DpasMemoryPattern::LSC_PREFETCH ||
                                          Pattern == DpasMemoryPattern::LSC_PREFETCH_2 ||
                                          Pattern == DpasMemoryPattern::LSC_PREFETCH_3 ||
                                          Pattern == DpasMemoryPattern::LSC_PREFETCH_4 ||
                                          Pattern == DpasMemoryPattern::LSC_PREFETCH_5 ||
                                          Pattern == DpasMemoryPattern::LSC_PREFETCH_6 ||
                                          Pattern == DpasMemoryPattern::LSC_PREFETCH_8 ||
                                          Pattern == DpasMemoryPattern::LSC_PREFETCH_10) {
                                constexpr int PrefetchDist =
                                    (Pattern == DpasMemoryPattern::LSC_PREFETCH_10) ? 10
                                    : (Pattern == DpasMemoryPattern::LSC_PREFETCH_8) ? 8
                                    : (Pattern == DpasMemoryPattern::LSC_PREFETCH_6) ? 6
                                    : (Pattern == DpasMemoryPattern::LSC_PREFETCH_5) ? 5
                                    : (Pattern == DpasMemoryPattern::LSC_PREFETCH_4) ? 4
                                    : (Pattern == DpasMemoryPattern::LSC_PREFETCH_3) ? 3
                                    : (Pattern == DpasMemoryPattern::LSC_PREFETCH_2) ? 2
                                                                                      : 1;
                                const int64_t next = kt + PrefetchDist;
                                if (next < k_tiles) {
                                    const TA * pf_a = a_ptr + PrefetchDist * AN;
                                    dpas_prefetch_streaming(pf_a);
                                    for (int i = 0; i < 4; ++i) {
                                        const TB * pf_b = b_ptr[i] + PrefetchDist * BN;
                                        dpas_prefetch_streaming(pf_b);
                                    }
                                }
                            }

                            simd<TA, AN> a_vec;
                            simd<TB, BN> b_vec[4];
                            if constexpr (Pattern == DpasMemoryPattern::SLM_BUFFER) {
                                auto a_tmp = dpas_block_load<TA, AN, Pattern>(a_ptr, args.misaligned);
                                slm_block_store<TA, AN>(slm_a_offset, a_tmp);
#pragma unroll
                                for (int i = 0; i < 4; ++i) {
                                    auto b_tmp = dpas_block_load<TB, BN, Pattern>(b_ptr[i], args.misaligned);
                                    slm_block_store<TB, BN>(slm_b_offset + i * slm_b_stride, b_tmp);
                                }
                                a_vec = slm_block_load<TA, AN>(slm_a_offset);
#pragma unroll
                                for (int i = 0; i < 4; ++i) {
                                    b_vec[i] = slm_block_load<TB, BN>(slm_b_offset + i * slm_b_stride);
                                }
                            } else {
                                a_vec = dpas_block_load<TA, AN, Pattern>(a_ptr, args.misaligned);
#pragma unroll
                                for (int i = 0; i < 4; ++i) {
                                    b_vec[i] = dpas_block_load<TB, BN, Pattern>(b_ptr[i], args.misaligned);
                                }
                            }
#pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                acc[i] = xmx::dpas<8, Repeat, TAcc, TAcc, TB, TA>(acc[i], b_vec[i], a_vec);
                            }
                            a_ptr += AN;
#pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                b_ptr[i] += BN;
                            }
                        }
                    }

                    for (int r = 0; r < Repeat; ++r) {
                        auto * out_ptr0 = c_base + (tile_m * Repeat + r) * args.n + tile_n_base * ExecN;
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            simd<TAcc, ExecN> row = acc[i].template select<ExecN, 1>(r * ExecN);
                            block_store(out_ptr0 + i * ExecN, row);
                        }
                    }
                } else if (n_tile_repeats == 8) {
                    simd<TAcc, Repeat * ExecN> acc[8] = {
                        TAcc(0), TAcc(0), TAcc(0), TAcc(0),
                        TAcc(0), TAcc(0), TAcc(0), TAcc(0)
                    };
                    const TA * a_ptr = a_base + (tile_m * k_tiles) * AN;
                    const TB * b_ptr[8];
                    b_ptr[0] = b_base + (tile_n_base * k_tiles) * BN;
                    for (int i = 1; i < 8; ++i) {
                        b_ptr[i] = b_ptr[i - 1] + BN * k_tiles;
                    }
                    if constexpr (Pattern == DpasMemoryPattern::DOUBLE_BUFFER) {
                        int buf_idx = 0;
                        {
                            auto a_tmp = dpas_block_load<TA, AN, Pattern>(a_ptr, args.misaligned);
                            slm_block_store<TA, AN>(slm_a_offset, a_tmp);
#pragma unroll
                            for (int i = 0; i < 8; ++i) {
                                auto b_tmp = dpas_block_load<TB, BN, Pattern>(b_ptr[i], args.misaligned);
                                slm_block_store<TB, BN>(slm_b_offset + i * slm_b_stride, b_tmp);
                            }
                        }
                        for (int64_t kt = 0; kt < k_tiles; ++kt) {
                            const uint32_t slm_offset = (buf_idx == 0) ? 0 : slm_tile_bytes;
                            simd<TA, AN> a_vec = slm_block_load<TA, AN>(slm_offset + slm_a_offset);
                            simd<TB, BN> b_vec[8];
#pragma unroll
                            for (int i = 0; i < 8; ++i) {
                                b_vec[i] = slm_block_load<TB, BN>(slm_offset + slm_b_offset + i * slm_b_stride);
                            }
#pragma unroll
                            for (int i = 0; i < 8; ++i) {
                                acc[i] = xmx::dpas<8, Repeat, TAcc, TAcc, TB, TA>(acc[i], b_vec[i], a_vec);
                            }
                            const bool has_next = (kt + 1) < k_tiles;
                            if (has_next) {
                                const uint32_t next_offset = (buf_idx == 0) ? slm_tile_bytes : 0;
                                auto a_next = dpas_block_load<TA, AN, Pattern>(a_ptr + AN, args.misaligned);
                                slm_block_store<TA, AN>(next_offset + slm_a_offset, a_next);
#pragma unroll
                                for (int i = 0; i < 8; ++i) {
                                    auto b_next = dpas_block_load<TB, BN, Pattern>(b_ptr[i] + BN, args.misaligned);
                                    slm_block_store<TB, BN>(next_offset + slm_b_offset + i * slm_b_stride, b_next);
                                }
                            }
                            buf_idx ^= 1;
                            a_ptr += AN;
#pragma unroll
                            for (int i = 0; i < 8; ++i) {
                                b_ptr[i] += BN;
                            }
                        }
                    } else if constexpr (Pattern == DpasMemoryPattern::REG_PREFETCH) {
                        simd<TA, AN> a_vec = dpas_block_load<TA, AN, Pattern>(a_ptr, args.misaligned);
                        simd<TB, BN> b_vec[8];
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            b_vec[i] = dpas_block_load<TB, BN, Pattern>(b_ptr[i], args.misaligned);
                        }
                        simd<TA, AN> a_next;
                        simd<TB, BN> b_next[8];
                        if (k_tiles > 1) {
                            a_next = dpas_block_load<TA, AN, Pattern>(a_ptr + AN, args.misaligned);
#pragma unroll
                            for (int i = 0; i < 8; ++i) {
                                b_next[i] = dpas_block_load<TB, BN, Pattern>(b_ptr[i] + BN, args.misaligned);
                            }
                        }
                        for (int64_t kt = 0; kt < k_tiles; ++kt) {
#pragma unroll
                            for (int i = 0; i < 8; ++i) {
                                acc[i] = xmx::dpas<8, Repeat, TAcc, TAcc, TB, TA>(acc[i], b_vec[i], a_vec);
                            }
                            if ((kt + 1) < k_tiles) {
                                const bool has_next2 = (kt + 2) < k_tiles;
                                if (has_next2) {
                                    simd<TA, AN> a_next2 =
                                        dpas_block_load<TA, AN, Pattern>(a_ptr + 2 * AN, args.misaligned);
                                    simd<TB, BN> b_next2[8];
#pragma unroll
                                    for (int i = 0; i < 8; ++i) {
                                        b_next2[i] =
                                            dpas_block_load<TB, BN, Pattern>(b_ptr[i] + 2 * BN, args.misaligned);
                                    }
                                    a_vec = a_next;
#pragma unroll
                                    for (int i = 0; i < 8; ++i) {
                                        b_vec[i] = b_next[i];
                                    }
                                    a_next = a_next2;
#pragma unroll
                                    for (int i = 0; i < 8; ++i) {
                                        b_next[i] = b_next2[i];
                                    }
                                } else {
                                    a_vec = a_next;
#pragma unroll
                                    for (int i = 0; i < 8; ++i) {
                                        b_vec[i] = b_next[i];
                                    }
                                }
                                a_ptr += AN;
#pragma unroll
                                for (int i = 0; i < 8; ++i) {
                                    b_ptr[i] += BN;
                                }
                            }
                        }
                    } else {
                        for (int64_t kt = 0; kt < k_tiles; ++kt) {
                            if constexpr (Pattern == DpasMemoryPattern::LSC_PREFETCH ||
                                          Pattern == DpasMemoryPattern::LSC_PREFETCH_2 ||
                                          Pattern == DpasMemoryPattern::LSC_PREFETCH_3 ||
                                          Pattern == DpasMemoryPattern::LSC_PREFETCH_4 ||
                                          Pattern == DpasMemoryPattern::LSC_PREFETCH_5 ||
                                          Pattern == DpasMemoryPattern::LSC_PREFETCH_6 ||
                                          Pattern == DpasMemoryPattern::LSC_PREFETCH_8 ||
                                          Pattern == DpasMemoryPattern::LSC_PREFETCH_10) {
                                constexpr int PrefetchDist =
                                    (Pattern == DpasMemoryPattern::LSC_PREFETCH_10) ? 10
                                    : (Pattern == DpasMemoryPattern::LSC_PREFETCH_8) ? 8
                                    : (Pattern == DpasMemoryPattern::LSC_PREFETCH_6) ? 6
                                    : (Pattern == DpasMemoryPattern::LSC_PREFETCH_5) ? 5
                                    : (Pattern == DpasMemoryPattern::LSC_PREFETCH_4) ? 4
                                    : (Pattern == DpasMemoryPattern::LSC_PREFETCH_3) ? 3
                                    : (Pattern == DpasMemoryPattern::LSC_PREFETCH_2) ? 2
                                                                                      : 1;
                                const int64_t next = kt + PrefetchDist;
                                if (next < k_tiles) {
                                    const TA * pf_a = a_ptr + PrefetchDist * AN;
                                    dpas_prefetch_streaming(pf_a);
#pragma unroll
                                    for (int i = 0; i < 8; ++i) {
                                        const TB * pf_b = b_ptr[i] + PrefetchDist * BN;
                                        dpas_prefetch_streaming(pf_b);
                                    }
                                }
                            }

                            simd<TA, AN> a_vec;
                            simd<TB, BN> b_vec[8];
                            if constexpr (Pattern == DpasMemoryPattern::SLM_BUFFER) {
                                auto a_tmp = dpas_block_load<TA, AN, Pattern>(a_ptr, args.misaligned);
                                slm_block_store<TA, AN>(slm_a_offset, a_tmp);
#pragma unroll
                                for (int i = 0; i < 8; ++i) {
                                    auto b_tmp = dpas_block_load<TB, BN, Pattern>(b_ptr[i], args.misaligned);
                                    slm_block_store<TB, BN>(slm_b_offset + i * slm_b_stride, b_tmp);
                                }
                                a_vec = slm_block_load<TA, AN>(slm_a_offset);
#pragma unroll
                                for (int i = 0; i < 8; ++i) {
                                    b_vec[i] = slm_block_load<TB, BN>(slm_b_offset + i * slm_b_stride);
                                }
                            } else {
                                a_vec = dpas_block_load<TA, AN, Pattern>(a_ptr, args.misaligned);
#pragma unroll
                                for (int i = 0; i < 8; ++i) {
                                    b_vec[i] = dpas_block_load<TB, BN, Pattern>(b_ptr[i], args.misaligned);
                                }
                            }
#pragma unroll
                            for (int i = 0; i < 8; ++i) {
                                acc[i] = xmx::dpas<8, Repeat, TAcc, TAcc, TB, TA>(acc[i], b_vec[i], a_vec);
                            }
                            a_ptr += AN;
#pragma unroll
                            for (int i = 0; i < 8; ++i) {
                                b_ptr[i] += BN;
                            }
                        }
                    }

                    for (int r = 0; r < Repeat; ++r) {
                        auto * out_ptr0 = c_base + (tile_m * Repeat + r) * args.n + tile_n_base * ExecN;
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            simd<TAcc, ExecN> row = acc[i].template select<ExecN, 1>(r * ExecN);
                            block_store(out_ptr0 + i * ExecN, row);
                        }
                    }
                }
            });
    });

    if (events) {
        events->push_back(ev);
    }
    return true;
#endif
}

template <typename TA, typename TB, typename TAcc, int Repeat, DpasMemoryPattern Pattern>
inline bool dispatch_dpas_grf(const DpasBenchArgs & args,
                              std::vector<sycl::event> * events,
                              std::string & error) {
    if (args.grf_mode == DpasGrfMode::GRF_256) {
        return launch_dpas_kernel<TA, TB, TAcc, Repeat, Pattern, DpasGrfMode::GRF_256>(args, events, error);
    }
    return launch_dpas_kernel<TA, TB, TAcc, Repeat, Pattern, DpasGrfMode::GRF_128>(args, events, error);
}

template <typename TA, typename TB, typename TAcc, int Repeat>
inline bool dispatch_dpas_pattern(const DpasBenchArgs & args,
                                  std::vector<sycl::event> * events,
                                  std::string & error) {
    switch (args.memory_pattern) {
        case DpasMemoryPattern::DIRECT_GLOBAL:
            return dispatch_dpas_grf<TA, TB, TAcc, Repeat, DpasMemoryPattern::DIRECT_GLOBAL>(args, events, error);
        case DpasMemoryPattern::SLM_BUFFER:
            return dispatch_dpas_grf<TA, TB, TAcc, Repeat, DpasMemoryPattern::SLM_BUFFER>(args, events, error);
        case DpasMemoryPattern::REG_PREFETCH:
            return dispatch_dpas_grf<TA, TB, TAcc, Repeat, DpasMemoryPattern::REG_PREFETCH>(args, events, error);
        case DpasMemoryPattern::DOUBLE_BUFFER:
            return dispatch_dpas_grf<TA, TB, TAcc, Repeat, DpasMemoryPattern::DOUBLE_BUFFER>(args, events, error);
        case DpasMemoryPattern::LSC_STREAMING:
            return dispatch_dpas_grf<TA, TB, TAcc, Repeat, DpasMemoryPattern::LSC_STREAMING>(args, events, error);
        case DpasMemoryPattern::LSC_PREFETCH:
            return dispatch_dpas_grf<TA, TB, TAcc, Repeat, DpasMemoryPattern::LSC_PREFETCH>(args, events, error);
        case DpasMemoryPattern::LSC_PREFETCH_2:
            return dispatch_dpas_grf<TA, TB, TAcc, Repeat, DpasMemoryPattern::LSC_PREFETCH_2>(args, events, error);
        case DpasMemoryPattern::LSC_PREFETCH_3:
            return dispatch_dpas_grf<TA, TB, TAcc, Repeat, DpasMemoryPattern::LSC_PREFETCH_3>(args, events, error);
        case DpasMemoryPattern::LSC_PREFETCH_4:
            return dispatch_dpas_grf<TA, TB, TAcc, Repeat, DpasMemoryPattern::LSC_PREFETCH_4>(args, events, error);
        case DpasMemoryPattern::LSC_PREFETCH_5:
            return dispatch_dpas_grf<TA, TB, TAcc, Repeat, DpasMemoryPattern::LSC_PREFETCH_5>(args, events, error);
        case DpasMemoryPattern::LSC_PREFETCH_6:
            return dispatch_dpas_grf<TA, TB, TAcc, Repeat, DpasMemoryPattern::LSC_PREFETCH_6>(args, events, error);
        case DpasMemoryPattern::LSC_PREFETCH_8:
            return dispatch_dpas_grf<TA, TB, TAcc, Repeat, DpasMemoryPattern::LSC_PREFETCH_8>(args, events, error);
        case DpasMemoryPattern::LSC_PREFETCH_10:
            return dispatch_dpas_grf<TA, TB, TAcc, Repeat, DpasMemoryPattern::LSC_PREFETCH_10>(args, events, error);
        default:
            error = "Unknown DPAS memory pattern.";
            return false;
    }
}

inline bool dispatch_dpas(const DpasBenchArgs & args,
                          std::vector<sycl::event> * events,
                          std::string & error) {
    if (args.type_a != args.type_b) {
        error = "DPAS exploration only supports matching A/B types.";
        return false;
    }

    auto dispatch_for_types = [&](auto ta_tag, auto tb_tag, auto acc_tag) -> bool {
        using TA = decltype(ta_tag);
        using TB = decltype(tb_tag);
        using TAcc = decltype(acc_tag);

        switch (args.repeat) {
            case 1: return dispatch_dpas_pattern<TA, TB, TAcc, 1>(args, events, error);
            case 2: return dispatch_dpas_pattern<TA, TB, TAcc, 2>(args, events, error);
            case 4: return dispatch_dpas_pattern<TA, TB, TAcc, 4>(args, events, error);
            case 8: return dispatch_dpas_pattern<TA, TB, TAcc, 8>(args, events, error);
            default: error = "DPAS repeat must be 1,2,4,8."; return false;
        }
    };

    if (args.type_a == DpasType::INT8) {
        if (args.type_acc == DpasAccType::INT32) {
            return dispatch_for_types(int8_t{}, int8_t{}, int{});
        }
        return dispatch_for_types(int8_t{}, int8_t{}, float{});
    }

    if (args.type_a == DpasType::FP16) {
        return dispatch_for_types(sycl::half{}, sycl::half{}, float{});
    }

    if (args.type_a == DpasType::BF16) {
        return dispatch_for_types(sycl::ext::oneapi::bfloat16{}, sycl::ext::oneapi::bfloat16{}, float{});
    }

    error = "Unsupported DPAS type.";
    return false;
}

}  // namespace sycl_bench
