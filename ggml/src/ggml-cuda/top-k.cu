#include "argsort.cuh"
#include "top-k.cuh"

#include <limits>
#include <string>
#include <vector>
#include <cstdint>
#include <charconv>
#include <string_view>

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
#    if (CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2)
#        define CUB_TOP_K_AVAILABLE
#        include <cuda/iterator>
using namespace cub;
#    endif  // CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2
#endif      // GGML_CUDA_USE_CUB

#ifdef CUB_TOP_K_AVAILABLE

static void top_k_cub(ggml_cuda_pool & pool,
                      const float *    src,
                      int *            dst,
                      const int        ncols,
                      const int        k,
                      cudaStream_t     stream) {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto stream_env   = cuda::stream_ref{ stream };
    auto env          = cuda::std::execution::env{ stream_env, requirements };

    auto indexes_in = cuda::make_counting_iterator(0);

    size_t temp_storage_bytes = 0;
    CUDA_CHECK(DeviceTopK::MaxPairs(nullptr, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst, ncols, k,
                         env));

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    CUDA_CHECK(DeviceTopK::MaxPairs(d_temp_storage, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst,
                         ncols, k, env));
}

#endif // CUB_TOP_K_AVAILABLE

static __device__ __forceinline__ uint32_t top_k_float_to_ordered(float value) {
    const uint32_t bits = __float_as_uint(value);
    const uint32_t mask = (uint32_t) (-(int32_t) (bits >> 31)) | 0x80000000U;
    return bits ^ mask;
}

struct top_k_radix_state {
    uint32_t prefix;
    uint32_t prefix_mask;
    int rank;
    int greater_count;
    int equal_count;
};

static __global__ void top_k_radix_init(top_k_radix_state * states, int nrows, int k) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row] = {0, 0, k, 0, 0};
    }
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_histogram(
        const float * __restrict__ src,
        const top_k_radix_state * __restrict__ states,
        int * __restrict__ block_histograms,
        int ncols,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    __shared__ int histogram[NBINS];

    histogram[tid] = 0;
    __syncthreads();

    const top_k_radix_state state = states[row];
    for (int col = row_block * BLOCK_SIZE + tid;
         col < ncols;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if ((key & state.prefix_mask) == state.prefix) {
            atomicAdd(&histogram[(key >> shift) & (NBINS - 1)], 1);
        }
    }
    __syncthreads();

    const size_t histogram_offset =
        ((size_t) row * blocks_per_row + row_block) * NBINS;
    block_histograms[histogram_offset + tid] = histogram[tid];
}

template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_select(
        const int * __restrict__ block_histograms,
        top_k_radix_state * __restrict__ states,
        int blocks_per_row,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ int histogram[NBINS];

    int count = 0;
    for (int row_block = 0; row_block < blocks_per_row; ++row_block) {
        const size_t offset = ((size_t) row * blocks_per_row + row_block) * NBINS;
        count += block_histograms[offset + tid];
    }
    histogram[tid] = count;
    __syncthreads();

    if (tid == 0) {
        top_k_radix_state state = states[row];
        int bin = NBINS - 1;
        while (bin > 0 && histogram[bin] < state.rank) {
            state.rank -= histogram[bin--];
        }
        state.prefix |= (uint32_t) bin << shift;
        state.prefix_mask |= (uint32_t) (NBINS - 1) << shift;
        states[row] = state;
    }
}

static __global__ void top_k_radix_reset_counters(top_k_radix_state * states, int nrows) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        states[row].greater_count = 0;
        states[row].equal_count = 0;
    }
}

template<int BLOCK_SIZE>
static __global__ void top_k_radix_gather(
        const float * __restrict__ src,
        int * __restrict__ dst,
        top_k_radix_state * __restrict__ states,
        int ncols,
        int k,
        int blocks_per_row) {
    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    int * row_dst = dst + (size_t) row * k;
    top_k_radix_state * state = &states[row];

    for (int col = row_block * BLOCK_SIZE + tid;
         col < ncols;
         col += blocks_per_row * BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if (key > state->prefix) {
            const int pos = atomicAdd(&state->greater_count, 1);
            row_dst[pos] = col;
        } else if (key == state->prefix) {
            const int pos = atomicAdd(&state->equal_count, 1);
            if (pos < state->rank) {
                row_dst[k - state->rank + pos] = col;
            }
        }
    }
}

static void top_k_radix_cuda(
        ggml_cuda_pool & pool,
        const float * src, int * dst, int ncols, int nrows, int k, cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 256;
    constexpr int RADIX_BITS = 8;
    constexpr int NBINS = 1 << RADIX_BITS;
    const int blocks_per_row = std::min((ncols + 1023) / 1024, 64);

    ggml_cuda_pool_alloc<top_k_radix_state> states_alloc(pool, nrows);
    ggml_cuda_pool_alloc<int> histograms_alloc(pool, (size_t) nrows * blocks_per_row * NBINS);
    top_k_radix_state * states = states_alloc.get();
    int * histograms = histograms_alloc.get();

    top_k_radix_init<<<(nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, nrows, k);

    const dim3 row_grid(blocks_per_row * nrows);
    for (int shift = 32 - RADIX_BITS; shift >= 0; shift -= RADIX_BITS) {
        top_k_radix_histogram<BLOCK_SIZE, RADIX_BITS>
            <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                src, states, histograms, ncols, blocks_per_row, shift);
        top_k_radix_select<BLOCK_SIZE, RADIX_BITS>
            <<<nrows, BLOCK_SIZE, 0, stream>>>(histograms, states, blocks_per_row, shift);
    }

    top_k_radix_reset_counters
        <<<(nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, nrows);
    top_k_radix_gather<BLOCK_SIZE>
        <<<row_grid, BLOCK_SIZE, 0, stream>>>(
            src, dst, states, ncols, k, blocks_per_row);
}

enum topk_impl {
    TOPK_IMPL_BITONIC,
    TOPK_IMPL_ARGSORT,
    TOPK_IMPL_DEVICETOPK,
    TOPK_IMPL_RADIX,
};

enum topk_param {
    TOPK_PARAM_NCOLS,
    TOPK_PARAM_NROWS,
    TOPK_PARAM_K,
    TOPK_PARAM_DEV,
};

struct topk_param_rule {
    topk_param param;
    std::vector<std::pair<int64_t, int64_t>> values;
    std::string str;
};

struct topk_selection_rule {
    std::vector<topk_param_rule> param_rules;
    topk_impl impl;
    std::string str;
};

static std::vector<topk_selection_rule> parse_topk_selection_rules(std::string_view input) {
    auto parse_impl_name = [&](std::string_view s) -> topk_impl {
        if (s == "bitonic"   ) { return TOPK_IMPL_BITONIC;    }
        if (s == "argsort"   ) { return TOPK_IMPL_ARGSORT;    }
        if (s == "devicetopk") { return TOPK_IMPL_DEVICETOPK; }
        if (s == "radix"     ) { return TOPK_IMPL_RADIX;      }
        GGML_ABORT("Unknown TOP_K implementation name %.*s", static_cast<int>(s.length()), s.data());
        return TOPK_IMPL_BITONIC; // should be unreachable
    };

    auto parse_param_name = [&](std::string_view s) -> topk_param {
        if (s == "ncols") { return TOPK_PARAM_NCOLS; }
        if (s == "nrows") { return TOPK_PARAM_NROWS; }
        if (s == "k"    ) { return TOPK_PARAM_K;     }
        if (s == "dev"  ) { return TOPK_PARAM_DEV;   }
        GGML_ABORT("Unknown TOP_K param name %.*s", static_cast<int>(s.length()), s.data());
        return TOPK_PARAM_NCOLS; // should be unreachable
    };

    auto parse_param_value = [&](std::string_view s) -> int64_t {
        if (s.empty()) {
            GGML_ABORT("TOP_K param value can't be empty");
        }
        int64_t value;
        const auto result = std::from_chars(s.data(), s.data() + s.size(), value);
        if (result.ec != std::errc{} || result.ptr != s.data() + s.size()) {
            GGML_ABORT("Invalid TOP_K param rule value %.*s", static_cast<int>(s.length()), s.data());
        }
        return value;
    };

    std::vector<topk_selection_rule> rules;
    while(true) {
        const auto semicolon = input.find(';');
        const auto rule = input.substr(0, semicolon);
        if (rule.empty()) {
            GGML_ABORT("Empty TOP_K implementation selection rule in %.*s", static_cast<int>(input.length()), input.data());
        }

        const auto equals = rule.find('=');
        const bool has_param_rules = equals != std::string_view::npos;
        topk_selection_rule parsed{{}, parse_impl_name(has_param_rules ? rule.substr(equals + 1) : rule), std::string(rule)};
        if (has_param_rules) {
            auto params = rule.substr(0, equals);
            if (params.empty()) {
                GGML_ABORT("Empty TOP_K parameter value rule in %.*s", static_cast<int>(rule.length()), rule.data());
            }
            while(true) {
                const auto open = params.find('[');
                if (open == std::string_view::npos) {
                    GGML_ABORT("Couldn't find TOP_K parameter value rule opening square bracket in %.*s", static_cast<int>(params.length()), params.data());
                }
                const auto close = params.find(']', open + 1);
                if (close == std::string_view::npos) {
                    GGML_ABORT("Couldn't find TOP_K parameter value rule closing square bracket in %.*s", static_cast<int>(params.length()), params.data());
                }

                topk_param_rule param_rule{parse_param_name(params.substr(0, open)), {}, std::string(params.substr(0, close + 1))};
                auto values = params.substr(open + 1, close - open - 1);

                while(true) {
                    const auto comma = values.find(',');
                    const auto value = values.substr(0, comma);
                    if (value.empty()) {
                        GGML_ABORT("Empty TOP_K parameter value in %.*s", static_cast<int>(params.length()), params.data());
                    }

                    const auto colon = value.find(':');
                    if (colon == std::string_view::npos) {
                        const auto n = parse_param_value(value);
                        param_rule.values.emplace_back(n, n);
                    } else {
                        const auto from = value.substr(0, colon);
                        const auto to = value.substr(colon + 1);
                        const auto lo = from.empty() ? int64_t{0} : parse_param_value(from);
                        const auto hi = to.empty() ? std::numeric_limits<int64_t>::max() : parse_param_value(to);
                        param_rule.values.emplace_back(lo, hi);
                    }

                    if (comma == std::string_view::npos) break;
                    values.remove_prefix(comma + 1);
                }

                parsed.param_rules.push_back(std::move(param_rule));
                params.remove_prefix(close + 1);
                if (params.empty()) break;
                if (params.front() != ',') {
                    GGML_ABORT("Missing comma before TOP_K parameter value rules %.*s", static_cast<int>(params.length()), params.data());
                }
                params.remove_prefix(1);
                if (params.empty()) {
                    GGML_ABORT("Empty TOP_K parameter value rule in %.*s", static_cast<int>(rule.length()), rule.data());
                }
            }
        }

        rules.push_back(std::move(parsed));
        if (semicolon == std::string_view::npos) {
            break;
        }
        input.remove_prefix(semicolon + 1);
    }
    return rules;
}

static topk_impl select_topk_impl(const char * selection_rules, int64_t ncols, int64_t nrows, int64_t k) {
    if (selection_rules == NULL) {
#if defined(GGML_USE_HIP)
        selection_rules = "ncols[:1024]=bitonic;radix";
#elif defined(CUB_TOP_K_AVAILABLE)
        selection_rules = "ncols[:1024]=bitonic;nrows[:2]=devicetopk;argsort";
#elif defined(GGML_CUDA_USE_CUB)
        selection_rules = "ncols[:1024]=bitonic;argsort";
#else
        selection_rules = "ncols[:1024]=bitonic;radix";
#endif
    }
    if (strcmp(selection_rules, "deterministic") == 0) {
#if defined(GGML_CUDA_USE_CUB)
        selection_rules = "ncols[:1024]=bitonic;argsort";
#else // GGML_CUDA_USE_CUB
        GGML_ABORT("Deterministic TOP_K implementation requires CUDA CUB");
#endif // GGML_CUDA_USE_CUB
    }

    // parse selection rules just once
    static std::vector<topk_selection_rule> selection_rules_parsed = parse_topk_selection_rules(selection_rules);

    GGML_LOG_DEBUG("%s: Searching for TOP_K implementation for ncols=%ld, nrows=%ld, k=%ld, dev=%d\n", __func__, ncols, nrows, k, ggml_cuda_get_device());
    for (const auto & selection_rule : selection_rules_parsed) {
        GGML_LOG_DEBUG("%s: Trying TOP_K implementation selection rule %s\n", __func__, selection_rule.str.c_str());
        bool match = true;
        for (const auto & param_rule : selection_rule.param_rules) {
            GGML_LOG_DEBUG("%s: Trying TOP_K parameter value rule %s\n", __func__, param_rule.str.c_str());
            int64_t param_value = 0;
            switch (param_rule.param) {
                case TOPK_PARAM_NCOLS: {
                    param_value = ncols;
                } break;
                case TOPK_PARAM_NROWS: {
                    param_value = nrows;
                } break;
                case TOPK_PARAM_K: {
                    param_value = k;
                } break;
                case TOPK_PARAM_DEV: {
                    param_value = ggml_cuda_get_device();
                } break;
            }
            bool value_match = false;
            for (const auto & param_values : param_rule.values) {
                if (param_value >= param_values.first && param_value <= param_values.second) {
                    value_match = true;
                    break;
                }
            }
            if (!value_match) {
                match=false;
                break;
            }
        }
        if (match) {
            GGML_LOG_DEBUG("%s: Found matching TOP_K implementation selection rule %s\n", __func__, selection_rule.str.c_str());
            return selection_rule.impl;
        }
    }

    GGML_ABORT("No matching TOP_K implementation selection rules found.");
}

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    int *               dst_d  = (int *) dst->data;
    cudaStream_t        stream = ctx.stream();

    // are these asserts truly necessary?
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t    ncols = src0->ne[0];
    const int64_t    nrows = ggml_nrows(src0);
    const int64_t    k     = dst->ne[0];
    ggml_cuda_pool & pool  = ctx.pool();

    const char * topk_selection_rules = getenv("GGML_CUDA_TOP_K_IMPL");
    switch (select_topk_impl(topk_selection_rules, ncols, nrows, k)) {
        case TOPK_IMPL_BITONIC: {
            ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
            int *                     tmp_dst = temp_dst_alloc.get();
            argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
            CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                         cudaMemcpyDeviceToDevice, stream));
        } break;
        case TOPK_IMPL_ARGSORT: {
#ifdef GGML_CUDA_USE_CUB
            const int chunk_nrows = argsort_f32_i32_cuda_cub_chunk_nrows(src0->nb[1], nrows);
            ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * chunk_nrows);
            int * tmp_dst = temp_dst_alloc.get();

            for (int64_t i = 0; i < nrows; i += chunk_nrows) {
                int iter_nrows = std::min((int64_t) chunk_nrows, nrows - i);

                argsort_f32_i32_cuda_cub(pool, src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
                CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int),
                                             iter_nrows, cudaMemcpyDeviceToDevice, stream));

                src0_d += ncols * iter_nrows;
                dst_d  += k     * iter_nrows;
            }
#else  // GGML_CUDA_USE_CUB
            GGML_ABORT("argsort TOP_K implementation is not available.");
#endif // GGML_CUDA_USE_CUB
        } break;
        case TOPK_IMPL_DEVICETOPK: {
#ifdef CUB_TOP_K_AVAILABLE
            // TODO: Switch to `DeviceBatchedTopK` for multi-row TopK once implemented
            // https://github.com/NVIDIA/cccl/issues/8360
            for (int i = 0; i < nrows; i++) {
                top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
            }
#else  // CUB_TOP_K_AVAILABLE
            GGML_ABORT("devicetopk TOP_K implementation is not available.");
#endif // CUB_TOP_K_AVAILABLE
        } break;
        case TOPK_IMPL_RADIX: {
            top_k_radix_cuda(pool, src0_d, dst_d, ncols, nrows, k, stream);
        } break;
    }
}
