#pragma once

#include <string>
#include <vector>

#include <sycl/sycl.hpp>

#include "ggml.h"
#include "ggml-quants.h"
#include "ggml-sycl/common.hpp"
#include "ggml-sycl/ggml-sycl-bench.hpp"

#if __has_include(<sycl/ext/oneapi/matrix/matrix.hpp>)
#    include <sycl/ext/oneapi/matrix/matrix.hpp>
#    define MMVQ_TIER2_XMX_AVAILABLE 1
#else
#    define MMVQ_TIER2_XMX_AVAILABLE 0
#endif

#if __has_include(<sycl/ext/intel/esimd.hpp>) && __has_include(<sycl/ext/intel/esimd/xmx/dpas.hpp>)
#    define MMVQ_TIER2_ESIMD_AVAILABLE 1
#    include <sycl/ext/intel/esimd.hpp>
#    include <sycl/ext/intel/esimd/xmx/dpas.hpp>
#else
#    define MMVQ_TIER2_ESIMD_AVAILABLE 0
#endif

namespace sycl_bench {
namespace mmvq_tier2 {

#if MMVQ_TIER2_XMX_AVAILABLE

namespace sycl_xmx = sycl::ext::oneapi::experimental::matrix;
#if MMVQ_TIER2_ESIMD_AVAILABLE
namespace esimd = sycl::ext::intel::esimd;
namespace esimd_xmx = sycl::ext::intel::esimd::xmx;
#endif

constexpr int XMX_M       = 8;
constexpr int XMX_N       = 16;
constexpr int XMX_K       = 32;
constexpr int XMX_SG_SIZE = 16;

constexpr int SLM_A_SIZE        = XMX_M * XMX_K;
constexpr int SLM_B_SIZE        = XMX_K * XMX_N;
constexpr int SLM_C_SIZE        = XMX_M * XMX_N;
constexpr int SLM_SCALES_A_SIZE = XMX_M * 2;
constexpr int SLM_OFFSETS_A_SIZE = XMX_M * 2;
constexpr int SLM_SCALES_B_SIZE = XMX_N;
constexpr int SLM_SUMS_B_SIZE   = XMX_N * 2;

struct AParams {
    float scale0  = 0.0f;
    float offset0 = 0.0f;
    float scale1  = 0.0f;
    float offset1 = 0.0f;
};

template <ggml_type T> struct block_traits;

template <> struct block_traits<GGML_TYPE_Q4_0> {
    using block_t = block_q4_0;
    static constexpr int block_size = QK4_0;
    static constexpr bool per_16 = false;
};

template <> struct block_traits<GGML_TYPE_Q8_0> {
    using block_t = block_q8_0;
    static constexpr int block_size = QK8_0;
    static constexpr bool per_16 = false;
};

template <> struct block_traits<GGML_TYPE_MXFP4> {
    using block_t = block_mxfp4;
    static constexpr int block_size = QK_MXFP4;
    static constexpr bool per_16 = false;
};

template <> struct block_traits<GGML_TYPE_Q2_K> {
    using block_t = block_q2_K;
    static constexpr int block_size = QK_K;
    static constexpr bool per_16 = true;
};

template <> struct block_traits<GGML_TYPE_Q3_K> {
    using block_t = block_q3_K;
    static constexpr int block_size = QK_K;
    static constexpr bool per_16 = true;
};

template <> struct block_traits<GGML_TYPE_Q4_K> {
    using block_t = block_q4_K;
    static constexpr int block_size = QK_K;
    static constexpr bool per_16 = false;
};

template <> struct block_traits<GGML_TYPE_Q5_K> {
    using block_t = block_q5_K;
    static constexpr int block_size = QK_K;
    static constexpr bool per_16 = false;
};

template <> struct block_traits<GGML_TYPE_Q6_K> {
    using block_t = block_q6_K;
    static constexpr int block_size = QK_K;
    static constexpr bool per_16 = true;
};

static inline bool is_tier2_quant(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q5_K:
            return true;
        default:
            return false;
    }
}

static inline int block_size_for_type(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0: return QK4_0;
        case GGML_TYPE_Q8_0: return QK8_0;
        case GGML_TYPE_MXFP4: return QK_MXFP4;
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
            return QK_K;
        default:
            return 0;
    }
}

static inline bool validate_xmx_args_common(const ggml_sycl::mmvq_bench_args & args,
                                            ggml_layout_mode layout,
                                            std::string & error) {
    if (args.stream == nullptr) {
        error = "SYCL stream is null.";
        return false;
    }
    if (args.layout != layout) {
        error = "Tier2 kernel layout mismatch.";
        return false;
    }
    if (layout == GGML_LAYOUT_SOA && args.layout_base == nullptr) {
        error = "Tier2 SoA kernels require layout_base pointer.";
        return false;
    }
    if (!is_tier2_quant(args.weight_type)) {
        error = "Tier2 kernels: unsupported quant type.";
        return false;
    }
    if (args.ncols <= 0 || args.nrows <= 0 || args.batch <= 0) {
        error = "Invalid dimensions for tier2 kernel.";
        return false;
    }
    if ((args.ncols % XMX_K) != 0) {
        error = "K dimension must be multiple of 32 for tier2 kernel.";
        return false;
    }
    if ((args.src1_padded_col_size % XMX_K) != 0) {
        error = "Padded K dimension must be multiple of 32 for tier2 kernel.";
        return false;
    }
    if (args.src1_padded_col_size < args.ncols) {
        error = "Padded K dimension is smaller than K for tier2 kernel.";
        return false;
    }
    const int block_size = block_size_for_type(args.weight_type);
    if (block_size <= 0 || (args.ncols % block_size) != 0) {
        error = "K dimension must be multiple of quant block size for tier2 kernel.";
        return false;
    }
    return true;
}

static inline bool validate_xmx_args(const ggml_sycl::mmvq_bench_args & args,
                                     std::string & error) {
    return validate_xmx_args_common(args, GGML_LAYOUT_AOS, error);
}

static inline bool validate_xmx_soa_args(const ggml_sycl::mmvq_bench_args & args,
                                         std::string & error) {
    return validate_xmx_args_common(args, GGML_LAYOUT_SOA, error);
}

static inline void decode_nibbles(const uint8_t * nibbles, int8_t * q_out) {
#pragma unroll
    for (int j = 0; j < 16; j++) {
        const uint8_t packed = nibbles[j];
        q_out[j]      = static_cast<int8_t>(packed & 0x0F);
        q_out[j + 16] = static_cast<int8_t>(packed >> 4);
    }
}

static inline int8_t q3_k_scale(const uint8_t * scales, int is) {
    int8_t us;
    if (is < 4) {
        us = (scales[is] & 0xF) | (((scales[is + 8] >> 0) & 3) << 4);
    } else if (is < 8) {
        us = (scales[is] & 0xF) | (((scales[is + 4] >> 2) & 3) << 4);
    } else if (is < 12) {
        us = (scales[is - 8] >> 4) | (((scales[is + 0] >> 4) & 3) << 4);
    } else {
        us = (scales[is - 8] >> 4) | (((scales[is - 4] >> 6) & 3) << 4);
    }
    return us;
}

#if QK_K == 256
static inline void get_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}
#endif

template <ggml_type T>
static inline void decode_block_aos(const void * weights,
                                    int row,
                                    int blocks_per_row,
                                    int block_idx,
                                    int sub_idx,
                                    int8_t * q_out,
                                    AParams & params) {
    using traits = block_traits<T>;
    const int64_t global_block = static_cast<int64_t>(row) * blocks_per_row + block_idx;
    const uint8_t * block_ptr = static_cast<const uint8_t *>(weights) + global_block * sizeof(typename traits::block_t);

    if constexpr (T == GGML_TYPE_Q4_0) {
        const sycl::half d = *reinterpret_cast<const sycl::half *>(block_ptr);
        params.scale0  = float(d);
        params.offset0 = -8.0f * params.scale0;
        decode_nibbles(block_ptr + sizeof(ggml_half), q_out);
    } else if constexpr (T == GGML_TYPE_Q8_0) {
        const sycl::half d = *reinterpret_cast<const sycl::half *>(block_ptr);
        params.scale0 = float(d);
        params.offset0 = 0.0f;
        const int8_t * qs = reinterpret_cast<const int8_t *>(block_ptr + sizeof(ggml_half));
#pragma unroll
        for (int j = 0; j < XMX_K; j++) {
            q_out[j] = qs[j];
        }
    } else if constexpr (T == GGML_TYPE_MXFP4) {
        const uint8_t * qs = block_ptr + sizeof(uint8_t);
        for (int j = 0; j < 16; ++j) {
            const uint8_t packed = qs[j];
            q_out[j]      = kvalues_mxfp4[packed & 0x0F];
            q_out[j + 16] = kvalues_mxfp4[packed >> 4];
        }
        const uint8_t e8m0 = *reinterpret_cast<const uint8_t *>(block_ptr);
        params.scale0  = sycl_e8m0_to_fp32_half(e8m0);
        params.offset0 = 0.0f;
    } else if constexpr (T == GGML_TYPE_Q4_K) {
        const block_q4_K * block = reinterpret_cast<const block_q4_K *>(block_ptr);
        const int chunk = sub_idx / 2;
        const bool high = (sub_idx & 1) != 0;
        const uint8_t * qs = block->qs + chunk * 32;
        for (int i = 0; i < XMX_K; ++i) {
            const uint8_t packed = qs[i];
            q_out[i] = static_cast<int8_t>(high ? (packed >> 4) : (packed & 0x0F));
        }
        uint8_t sc, m;
        get_scale_min_k4(sub_idx, block->scales, sc, m);
        const sycl::half d = *reinterpret_cast<const sycl::half *>(block_ptr);
        const sycl::half dmin = *reinterpret_cast<const sycl::half *>(block_ptr + sizeof(ggml_half));
        params.scale0  = float(d) * sc;
        params.offset0 = -float(dmin) * m;
    } else if constexpr (T == GGML_TYPE_Q5_K) {
        const block_q5_K * block = reinterpret_cast<const block_q5_K *>(block_ptr);
        const int chunk = sub_idx / 2;
        const bool high = (sub_idx & 1) != 0;
        const uint8_t * qs = block->qs + chunk * 32;
        const uint8_t * qh = block->qh;
        const uint8_t mask = static_cast<uint8_t>(1u << sub_idx);
        for (int i = 0; i < XMX_K; i++) {
            const uint8_t packed = qs[i];
            const uint8_t low = high ? (packed >> 4) : (packed & 0x0F);
            const uint8_t hbit = (qh[i] & mask) ? 1 : 0;
            q_out[i] = static_cast<int8_t>(low + (hbit << 4));
        }
        uint8_t sc, m;
        get_scale_min_k4(sub_idx, block->scales, sc, m);
        const sycl::half d = *reinterpret_cast<const sycl::half *>(block_ptr);
        const sycl::half dmin = *reinterpret_cast<const sycl::half *>(block_ptr + sizeof(ggml_half));
        params.scale0  = float(d) * sc;
        params.offset0 = -float(dmin) * m;
    } else if constexpr (T == GGML_TYPE_Q2_K) {
        const block_q2_K * block = reinterpret_cast<const block_q2_K *>(block_ptr);
        const uint8_t * qs = block->qs;
        const int block128 = sub_idx / 4;
        const int sub = sub_idx % 4;
        const int base = block128 * 32;
        const int shift = sub * 2;
        for (int i = 0; i < 16; ++i) {
            const uint8_t qbyte = qs[base + i];
            q_out[i] = static_cast<int8_t>((qbyte >> shift) & 0x3);
        }
        for (int i = 0; i < 16; ++i) {
            const uint8_t qbyte = qs[base + 16 + i];
            q_out[i + 16] = static_cast<int8_t>((qbyte >> shift) & 0x3);
        }
        const uint8_t * scales = block->scales;
        const uint8_t * dm_ptr = block_ptr + (QK_K / 16) + (QK_K / 4);
        const sycl::half d = *reinterpret_cast<const sycl::half *>(dm_ptr);
        const sycl::half dmin = *reinterpret_cast<const sycl::half *>(dm_ptr + sizeof(ggml_half));
        const int sc0 = scales[sub_idx * 2 + 0];
        const int sc1 = scales[sub_idx * 2 + 1];
        params.scale0  = float(d) * (sc0 & 0xF);
        params.offset0 = -float(dmin) * (sc0 >> 4);
        params.scale1  = float(d) * (sc1 & 0xF);
        params.offset1 = -float(dmin) * (sc1 >> 4);
    } else if constexpr (T == GGML_TYPE_Q3_K) {
        const block_q3_K * block = reinterpret_cast<const block_q3_K *>(block_ptr);
        const uint8_t * qs = block->qs;
        const uint8_t * hmask = block->hmask;
        const int block128 = sub_idx / 4;
        const int sub = sub_idx % 4;
        const int base = block128 * 32;
        const int shift = sub * 2;
        const uint8_t mask = static_cast<uint8_t>(1u << sub_idx);
        for (int i = 0; i < 16; ++i) {
            const uint8_t qbyte = qs[base + i];
            const int low = (qbyte >> shift) & 0x3;
            const int hbit = (hmask[i] & mask) ? 1 : 0;
            q_out[i] = static_cast<int8_t>(low - (hbit ? 0 : 4));
        }
        for (int i = 0; i < 16; ++i) {
            const uint8_t qbyte = qs[base + 16 + i];
            const int low = (qbyte >> shift) & 0x3;
            const int hbit = (hmask[16 + i] & mask) ? 1 : 0;
            q_out[i + 16] = static_cast<int8_t>(low - (hbit ? 0 : 4));
        }
        const uint8_t * scales = block->scales;
        const float d = float(block->d);
        const int8_t sc0 = q3_k_scale(scales, sub_idx * 2 + 0);
        const int8_t sc1 = q3_k_scale(scales, sub_idx * 2 + 1);
        params.scale0  = d * (sc0 - 32);
        params.offset0 = 0.0f;
        params.scale1  = d * (sc1 - 32);
        params.offset1 = 0.0f;
    } else if constexpr (T == GGML_TYPE_Q6_K) {
        const block_q6_K * block = reinterpret_cast<const block_q6_K *>(block_ptr);
        const uint8_t * ql = block->ql;
        const uint8_t * qh = block->qh;
        const int block128 = sub_idx / 4;
        const int sub = sub_idx % 4;
        const int ql_base = block128 * 64;
        const int qh_base = block128 * 32;
        const bool high = sub >= 2;
        const int ql_offset = ql_base + ((sub & 1) ? 32 : 0);
        const int qh_shift = sub * 2;
        for (int i = 0; i < XMX_K; i++) {
            const uint8_t ql_byte = ql[ql_offset + i];
            const uint8_t low = high ? (ql_byte >> 4) : (ql_byte & 0x0F);
            const uint8_t qh_byte = qh[qh_base + i];
            const uint8_t highbits = (qh_byte >> qh_shift) & 0x3;
            q_out[i] = static_cast<int8_t>((low | (highbits << 4)) - 32);
        }
        const int8_t * scales = block->scales;
        const float d = float(block->d);
        params.scale0  = d * scales[sub_idx * 2 + 0];
        params.offset0 = 0.0f;
        params.scale1  = d * scales[sub_idx * 2 + 1];
        params.offset1 = 0.0f;
    }
}

template <ggml_type T>
static inline void decode_block_soa(const void * layout_base,
                                    int row,
                                    int blocks_per_row,
                                    int block_idx,
                                    int sub_idx,
                                    size_t nblocks_total,
                                    int8_t * q_out,
                                    AParams & params) {
    const size_t global_block = static_cast<size_t>(row) * blocks_per_row + static_cast<size_t>(block_idx);
    const uint8_t * base = static_cast<const uint8_t *>(layout_base);

    if constexpr (T == GGML_TYPE_Q4_0) {
        const uint8_t * qs_base = base;
        const sycl::half * d_base =
            reinterpret_cast<const sycl::half *>(qs_base + nblocks_total * (QK4_0 / 2));
        const uint8_t * qs = qs_base + global_block * (QK4_0 / 2);
        params.scale0  = float(d_base[global_block]);
        params.offset0 = -8.0f * params.scale0;
        decode_nibbles(qs, q_out);
    } else if constexpr (T == GGML_TYPE_Q8_0) {
        const uint8_t * qs_base = base;
        const sycl::half * d_base =
            reinterpret_cast<const sycl::half *>(qs_base + nblocks_total * QK8_0);
        const int8_t * qs = reinterpret_cast<const int8_t *>(qs_base + global_block * QK8_0);
        params.scale0 = float(d_base[global_block]);
        params.offset0 = 0.0f;
#pragma unroll
        for (int j = 0; j < XMX_K; j++) {
            q_out[j] = qs[j];
        }
    } else if constexpr (T == GGML_TYPE_MXFP4) {
        const uint8_t * qs_base = base;
        const uint8_t * e_base = qs_base + nblocks_total * (QK_MXFP4 / 2);
        const uint8_t * qs = qs_base + global_block * (QK_MXFP4 / 2);
        for (int j = 0; j < 16; ++j) {
            const uint8_t packed = qs[j];
            q_out[j]      = kvalues_mxfp4[packed & 0x0F];
            q_out[j + 16] = kvalues_mxfp4[packed >> 4];
        }
        params.scale0  = sycl_e8m0_to_fp32_half(e_base[global_block]);
        params.offset0 = 0.0f;
    } else if constexpr (T == GGML_TYPE_Q6_K) {
        const uint8_t * ql_base = base;
        const uint8_t * qh_base = ql_base + nblocks_total * (QK_K / 2);
        const uint8_t * scales_base = qh_base + nblocks_total * (QK_K / 4);
        const sycl::half * d_base =
            reinterpret_cast<const sycl::half *>(scales_base + nblocks_total * (QK_K / 16));
        const uint8_t * ql = ql_base + global_block * (QK_K / 2);
        const uint8_t * qh = qh_base + global_block * (QK_K / 4);
        const int8_t * scales = reinterpret_cast<const int8_t *>(scales_base + global_block * (QK_K / 16));
        const int block128 = sub_idx / 4;
        const int sub = sub_idx % 4;
        const int ql_base_offset = block128 * 64;
        const int qh_base_offset = block128 * 32;
        const bool high = sub >= 2;
        const int ql_offset = ql_base_offset + ((sub & 1) ? 32 : 0);
        const int qh_shift = sub * 2;
        for (int i = 0; i < XMX_K; i++) {
            const uint8_t ql_byte = ql[ql_offset + i];
            const uint8_t low = high ? (ql_byte >> 4) : (ql_byte & 0x0F);
            const uint8_t qh_byte = qh[qh_base_offset + i];
            const uint8_t highbits = (qh_byte >> qh_shift) & 0x3;
            q_out[i] = static_cast<int8_t>((low | (highbits << 4)) - 32);
        }
        const float d = float(d_base[global_block]);
        params.scale0  = d * scales[sub_idx * 2 + 0];
        params.offset0 = 0.0f;
        params.scale1  = d * scales[sub_idx * 2 + 1];
        params.offset1 = 0.0f;
    } else if constexpr (T == GGML_TYPE_Q4_K) {
        const uint8_t * qs_base = base;
        const uint8_t * scales_base = qs_base + nblocks_total * (QK_K / 2);
        const sycl::half * dm_base =
            reinterpret_cast<const sycl::half *>(scales_base + nblocks_total * K_SCALE_SIZE);
        const int chunk = sub_idx / 2;
        const bool high = (sub_idx & 1) != 0;
        const uint8_t * qs = qs_base + global_block * (QK_K / 2) + chunk * 32;
        for (int i = 0; i < XMX_K; ++i) {
            const uint8_t packed = qs[i];
            q_out[i] = static_cast<int8_t>(high ? (packed >> 4) : (packed & 0x0F));
        }
        uint8_t sc, m;
        get_scale_min_k4(sub_idx, scales_base + global_block * K_SCALE_SIZE, sc, m);
        const sycl::half d = dm_base[global_block * 2 + 0];
        const sycl::half dmin = dm_base[global_block * 2 + 1];
        params.scale0  = float(d) * sc;
        params.offset0 = -float(dmin) * m;
    } else if constexpr (T == GGML_TYPE_Q5_K) {
        const uint8_t * qs_base = base;
        const uint8_t * qh_base = qs_base + nblocks_total * (QK_K / 2);
        const uint8_t * scales_base = qh_base + nblocks_total * (QK_K / 8);
        const sycl::half * dm_base =
            reinterpret_cast<const sycl::half *>(scales_base + nblocks_total * K_SCALE_SIZE);
        const uint8_t * qs = qs_base + global_block * (QK_K / 2);
        const uint8_t * qh = qh_base + global_block * (QK_K / 8);
        const int chunk = sub_idx / 2;
        const bool high = (sub_idx & 1) != 0;
        const uint8_t mask = static_cast<uint8_t>(1u << sub_idx);
        const uint8_t * qs_chunk = qs + chunk * 32;
        for (int i = 0; i < XMX_K; i++) {
            const uint8_t packed = qs_chunk[i];
            const uint8_t low = high ? (packed >> 4) : (packed & 0x0F);
            const uint8_t hbit = (qh[i] & mask) ? 1 : 0;
            q_out[i] = static_cast<int8_t>(low + (hbit << 4));
        }
        uint8_t sc, m;
        get_scale_min_k4(sub_idx, scales_base + global_block * K_SCALE_SIZE, sc, m);
        const sycl::half d = dm_base[global_block * 2 + 0];
        const sycl::half dmin = dm_base[global_block * 2 + 1];
        params.scale0  = float(d) * sc;
        params.offset0 = -float(dmin) * m;
    } else if constexpr (T == GGML_TYPE_Q2_K) {
        const uint8_t * qs_base = base;
        const uint8_t * scales_base = qs_base + nblocks_total * (QK_K / 4);
        const sycl::half * dm_base =
            reinterpret_cast<const sycl::half *>(scales_base + nblocks_total * (QK_K / 16));
        const uint8_t * qs = qs_base + global_block * (QK_K / 4);
        const uint8_t * scales = scales_base + global_block * (QK_K / 16);
        const int block128 = sub_idx / 4;
        const int sub = sub_idx % 4;
        const int base_idx = block128 * 32;
        const int shift = sub * 2;
        for (int i = 0; i < 16; ++i) {
            const uint8_t qbyte = qs[base_idx + i];
            q_out[i] = static_cast<int8_t>((qbyte >> shift) & 0x3);
        }
        for (int i = 0; i < 16; ++i) {
            const uint8_t qbyte = qs[base_idx + 16 + i];
            q_out[i + 16] = static_cast<int8_t>((qbyte >> shift) & 0x3);
        }
        const sycl::half d = dm_base[global_block * 2 + 0];
        const sycl::half dmin = dm_base[global_block * 2 + 1];
        const int sc0 = scales[sub_idx * 2 + 0];
        const int sc1 = scales[sub_idx * 2 + 1];
        params.scale0  = float(d) * (sc0 & 0xF);
        params.offset0 = -float(dmin) * (sc0 >> 4);
        params.scale1  = float(d) * (sc1 & 0xF);
        params.offset1 = -float(dmin) * (sc1 >> 4);
    } else if constexpr (T == GGML_TYPE_Q3_K) {
        const uint8_t * qs_base = base;
        const uint8_t * hmask_base = qs_base + nblocks_total * (QK_K / 4);
        const uint8_t * scales_base = hmask_base + nblocks_total * (QK_K / 8);
        const sycl::half * d_base =
            reinterpret_cast<const sycl::half *>(scales_base + nblocks_total * 12);
        const uint8_t * qs = qs_base + global_block * (QK_K / 4);
        const uint8_t * hmask = hmask_base + global_block * (QK_K / 8);
        const int block128 = sub_idx / 4;
        const int sub = sub_idx % 4;
        const int base_idx = block128 * 32;
        const int shift = sub * 2;
        const uint8_t mask = static_cast<uint8_t>(1u << sub_idx);
        for (int i = 0; i < 16; ++i) {
            const uint8_t qbyte = qs[base_idx + i];
            const int low = (qbyte >> shift) & 0x3;
            const int hbit = (hmask[i] & mask) ? 1 : 0;
            q_out[i] = static_cast<int8_t>(low - (hbit ? 0 : 4));
        }
        for (int i = 0; i < 16; ++i) {
            const uint8_t qbyte = qs[base_idx + 16 + i];
            const int low = (qbyte >> shift) & 0x3;
            const int hbit = (hmask[16 + i] & mask) ? 1 : 0;
            q_out[i + 16] = static_cast<int8_t>(low - (hbit ? 0 : 4));
        }
        const float d = float(d_base[global_block]);
        const int8_t sc0 = q3_k_scale(scales_base + global_block * 12, sub_idx * 2 + 0);
        const int8_t sc1 = q3_k_scale(scales_base + global_block * 12, sub_idx * 2 + 1);
        params.scale0  = d * (sc0 - 32);
        params.offset0 = 0.0f;
        params.scale1  = d * (sc1 - 32);
        params.offset1 = 0.0f;
    }
}

template <ggml_type T, bool SoA, int RowSubgroups, int TileCols>
static inline void mmvq_tile_kernel(const void * __restrict__ weights,
                                    const void * __restrict__ layout_base,
                                    const void * __restrict__ activations,
                                    float * __restrict__ dst,
                                    const int ncols_x,
                                    const int ncols_y,
                                    const int nrows_dst,
                                    const int row_low,
                                    const int row_high,
                                    const int src1_stride_blocks,
                                    const int blocks_per_row,
                                    const size_t nblocks_total,
                                    int8_t * __restrict__ slm_A,
                                    int8_t * __restrict__ slm_B,
                                    int32_t * __restrict__ slm_C,
                                    float * __restrict__ slm_scales_A,
                                    float * __restrict__ slm_offsets_A,
                                    float * __restrict__ slm_scales_B,
                                    float * __restrict__ slm_sums_B,
                                    sycl::nd_item<2> item) {
    if constexpr (SoA) {
        (void) weights;
    } else {
        (void) nblocks_total;
    }
    if constexpr (!SoA) {
        (void) layout_base;
    }
    const auto sg      = item.get_sub_group();
    const int  lane_id = sg.get_local_id()[0];
    const int  row_sg  = item.get_local_id(0);

    const int tile_rows = RowSubgroups * XMX_M;
    const int row_base_group = item.get_group(0) * tile_rows + row_low;
    const int col_base = item.get_group(1) * TileCols;

    if (row_base_group >= row_high || col_base >= ncols_y) {
        return;
    }

    const int row_base = row_base_group + row_sg * XMX_M;
    const int num_k_blocks = ncols_x / XMX_K;

    const char * src1 = static_cast<const char *>(activations);
    constexpr int Q8_1_BLOCK_SIZE = sizeof(block_q8_1);
    constexpr int SUBBLOCKS = block_traits<T>::block_size / XMX_K;
    constexpr int PASSES = block_traits<T>::per_16 ? 2 : 1;

    float acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    for (int k_block = 0; k_block < num_k_blocks; k_block++) {
        int8_t q_local[XMX_K];
        AParams params{};

        if (lane_id < XMX_M) {
            const int row = row_base + lane_id;
            if (row < row_high) {
                const int block_idx = k_block / SUBBLOCKS;
                const int sub_idx = k_block - block_idx * SUBBLOCKS;
                if constexpr (SoA) {
                    decode_block_soa<T>(layout_base, row, blocks_per_row, block_idx, sub_idx, nblocks_total, q_local, params);
                } else {
                    decode_block_aos<T>(weights, row, blocks_per_row, block_idx, sub_idx, q_local, params);
                }
            } else {
#pragma unroll
                for (int j = 0; j < XMX_K; j++) {
                    q_local[j] = 0;
                }
                params.scale0  = 0.0f;
                params.offset0 = 0.0f;
                params.scale1  = 0.0f;
                params.offset1 = 0.0f;
            }
        }

        if (lane_id < XMX_N) {
            const int col = col_base + lane_id;
            const bool active_col = (lane_id < TileCols) && (col < ncols_y);
            if (active_col) {
                const char * block_ptr = src1 +
                    (static_cast<int64_t>(col) * src1_stride_blocks + k_block) * Q8_1_BLOCK_SIZE;
                const sycl::half d = *reinterpret_cast<const sycl::half *>(block_ptr);
                const sycl::half s = *reinterpret_cast<const sycl::half *>(block_ptr + 2);

                slm_scales_B[lane_id] = float(d);

                float sum0 = 0.0f;
                float sum1 = 0.0f;
                const int8_t * qs = reinterpret_cast<const int8_t *>(block_ptr + 4);
#pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    const int8_t v = qs[k];
                    slm_B[lane_id * XMX_K + k] = v;
                    if constexpr (T == GGML_TYPE_Q2_K) {
                        if (k < 16) {
                            sum0 += v;
                        } else {
                            sum1 += v;
                        }
                    }
                }
                if constexpr (T == GGML_TYPE_Q2_K) {
                    const float d_f = float(d);
                    slm_sums_B[0 * XMX_N + lane_id] = d_f * sum0;
                    slm_sums_B[1 * XMX_N + lane_id] = d_f * sum1;
                } else {
                    slm_sums_B[lane_id] = float(s);
                }
            } else {
                slm_scales_B[lane_id] = 0.0f;
                if constexpr (T == GGML_TYPE_Q2_K) {
                    slm_sums_B[0 * XMX_N + lane_id] = 0.0f;
                    slm_sums_B[1 * XMX_N + lane_id] = 0.0f;
                } else {
                    slm_sums_B[lane_id] = 0.0f;
                }
#pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    slm_B[lane_id * XMX_K + k] = 0;
                }
            }
        }

        item.barrier(sycl::access::fence_space::local_space);

        for (int pass = 0; pass < PASSES; pass++) {
            if (lane_id < XMX_M) {
                const float scale = (pass == 0) ? params.scale0 : params.scale1;
                const float offset = (pass == 0) ? params.offset0 : params.offset1;
                slm_scales_A[pass * XMX_M + lane_id] = scale;
                slm_offsets_A[pass * XMX_M + lane_id] = offset;
                const int base_idx = lane_id * XMX_K;
#pragma unroll
                for (int j = 0; j < XMX_K; j++) {
                    if constexpr (PASSES == 1) {
                        slm_A[base_idx + j] = q_local[j];
                    } else {
                        const bool keep = (pass == 0) ? (j < 16) : (j >= 16);
                        slm_A[base_idx + j] = keep ? q_local[j] : 0;
                    }
                }
            }

            item.barrier(sycl::access::fence_space::local_space);

            sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::a, XMX_M, XMX_K, sycl_xmx::layout::row_major>
                matA;
            sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::b, XMX_K, XMX_N, sycl_xmx::layout::col_major>
                matB;
            sycl_xmx::joint_matrix<sycl::sub_group, int32_t, sycl_xmx::use::accumulator, XMX_M, XMX_N> matC;

            auto A_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
                                                  sycl::access::decorated::no>(slm_A);
            auto B_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
                                                  sycl::access::decorated::no>(slm_B);
            auto C_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
                                                  sycl::access::decorated::no>(slm_C);

            sycl_xmx::joint_matrix_load(sg, matA, A_ptr, XMX_K);
            sycl_xmx::joint_matrix_load(sg, matB, B_ptr, XMX_K);
            sycl_xmx::joint_matrix_fill(sg, matC, 0);
            sycl_xmx::joint_matrix_mad(sg, matC, matA, matB, matC);
            sycl_xmx::joint_matrix_store(sg, matC, C_ptr, XMX_N, sycl_xmx::layout::row_major);

#pragma unroll
            for (int idx = lane_id, acc_idx = 0; idx < XMX_M * XMX_N; idx += XMX_SG_SIZE, acc_idx++) {
                const int   i    = idx / XMX_N;
                const int   j    = idx % XMX_N;
                const float d_A  = slm_scales_A[pass * XMX_M + i];
                const float o_A  = slm_offsets_A[pass * XMX_M + i];
                const float d_B  = slm_scales_B[j];
                float s_B = 0.0f;
                if constexpr (T == GGML_TYPE_Q2_K) {
                    s_B = slm_sums_B[pass * XMX_N + j];
                } else {
                    s_B = slm_sums_B[j];
                }
                const float C_ij = static_cast<float>(slm_C[idx]);
                acc[acc_idx] += d_A * (C_ij * d_B) + o_A * s_B;
            }

            item.barrier(sycl::access::fence_space::local_space);
        }

        item.barrier(sycl::access::fence_space::local_space);
    }

#pragma unroll
    for (int idx = lane_id, acc_idx = 0; idx < XMX_M * XMX_N; idx += XMX_SG_SIZE, acc_idx++) {
        const int i   = idx / XMX_N;
        const int j   = idx % XMX_N;
        const int row = row_base + i;
        const int col = col_base + j;
        if (row < row_high && col < ncols_y && j < TileCols) {
            dst[col * nrows_dst + row] = acc[acc_idx];
        }
    }
}

template <ggml_type T, int RowSubgroups, int TileCols>
static inline void mmvq_tile_kernel_db(const void * __restrict__ weights,
                                       const void * __restrict__ activations,
                                       float * __restrict__ dst,
                                       const int ncols_x,
                                       const int ncols_y,
                                       const int nrows_dst,
                                       const int row_low,
                                       const int row_high,
                                       const int src1_stride_blocks,
                                       const int blocks_per_row,
                                       int8_t * __restrict__ slm_A,
                                       int8_t * __restrict__ slm_B,
                                       int32_t * __restrict__ slm_C,
                                       float * __restrict__ slm_scales_A,
                                       float * __restrict__ slm_offsets_A,
                                       float * __restrict__ slm_scales_B,
                                       float * __restrict__ slm_sums_B,
                                       sycl::nd_item<2> item) {
    const auto sg      = item.get_sub_group();
    const int  lane_id = sg.get_local_id()[0];
    const int  row_sg  = item.get_local_id(0);

    const int tile_rows = RowSubgroups * XMX_M;
    const int row_base_group = item.get_group(0) * tile_rows + row_low;
    const int col_base = item.get_group(1) * TileCols;

    if (row_base_group >= row_high || col_base >= ncols_y) {
        return;
    }

    const int row_base = row_base_group + row_sg * XMX_M;
    const int num_k_blocks = ncols_x / XMX_K;

    const char * src1 = static_cast<const char *>(activations);
    constexpr int Q8_1_BLOCK_SIZE = sizeof(block_q8_1);
    constexpr int SUBBLOCKS = block_traits<T>::block_size / XMX_K;
    constexpr int PASSES = block_traits<T>::per_16 ? 2 : 1;

    float acc[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    auto load_k_block = [&](int k_block, int buf) {
        if (lane_id < XMX_M) {
            const int row = row_base + lane_id;
            if (row < row_high) {
                const int block_idx = k_block / SUBBLOCKS;
                const int sub_idx = k_block - block_idx * SUBBLOCKS;
                int8_t q_local[XMX_K];
                AParams params{};
                decode_block_aos<T>(weights, row, blocks_per_row, block_idx, sub_idx, q_local, params);
                const int base_idx = buf * SLM_A_SIZE + lane_id * XMX_K;
#pragma unroll
                for (int j = 0; j < XMX_K; j++) {
                    slm_A[base_idx + j] = q_local[j];
                }
                slm_scales_A[buf * SLM_SCALES_A_SIZE + lane_id] = params.scale0;
                slm_offsets_A[buf * SLM_OFFSETS_A_SIZE + lane_id] = params.offset0;
                slm_scales_A[buf * SLM_SCALES_A_SIZE + XMX_M + lane_id] = params.scale1;
                slm_offsets_A[buf * SLM_OFFSETS_A_SIZE + XMX_M + lane_id] = params.offset1;
            } else {
                const int base_idx = buf * SLM_A_SIZE + lane_id * XMX_K;
#pragma unroll
                for (int j = 0; j < XMX_K; j++) {
                    slm_A[base_idx + j] = 0;
                }
                slm_scales_A[buf * SLM_SCALES_A_SIZE + lane_id] = 0.0f;
                slm_offsets_A[buf * SLM_OFFSETS_A_SIZE + lane_id] = 0.0f;
                slm_scales_A[buf * SLM_SCALES_A_SIZE + XMX_M + lane_id] = 0.0f;
                slm_offsets_A[buf * SLM_OFFSETS_A_SIZE + XMX_M + lane_id] = 0.0f;
            }
        }

        if (lane_id < XMX_N) {
            const int col = col_base + lane_id;
            const bool active_col = (lane_id < TileCols) && (col < ncols_y);
            if (active_col) {
                const char * block_ptr = src1 +
                    (static_cast<int64_t>(col) * src1_stride_blocks + k_block) * Q8_1_BLOCK_SIZE;
                const sycl::half d = *reinterpret_cast<const sycl::half *>(block_ptr);
                const sycl::half s = *reinterpret_cast<const sycl::half *>(block_ptr + 2);

                slm_scales_B[buf * SLM_SCALES_B_SIZE + lane_id] = float(d);

                float sum0 = 0.0f;
                float sum1 = 0.0f;
                const int8_t * qs = reinterpret_cast<const int8_t *>(block_ptr + 4);
                const int base_idx = buf * SLM_B_SIZE + lane_id * XMX_K;
#pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    const int8_t v = qs[k];
                    slm_B[base_idx + k] = v;
                    if constexpr (T == GGML_TYPE_Q2_K) {
                        if (k < 16) {
                            sum0 += v;
                        } else {
                            sum1 += v;
                        }
                    }
                }
                if constexpr (T == GGML_TYPE_Q2_K) {
                    const float d_f = float(d);
                    const int sum_base = buf * SLM_SUMS_B_SIZE;
                    slm_sums_B[sum_base + 0 * XMX_N + lane_id] = d_f * sum0;
                    slm_sums_B[sum_base + 1 * XMX_N + lane_id] = d_f * sum1;
                } else {
                    slm_sums_B[buf * SLM_SUMS_B_SIZE + lane_id] = float(s);
                }
            } else {
                slm_scales_B[buf * SLM_SCALES_B_SIZE + lane_id] = 0.0f;
                if constexpr (T == GGML_TYPE_Q2_K) {
                    const int sum_base = buf * SLM_SUMS_B_SIZE;
                    slm_sums_B[sum_base + 0 * XMX_N + lane_id] = 0.0f;
                    slm_sums_B[sum_base + 1 * XMX_N + lane_id] = 0.0f;
                } else {
                    slm_sums_B[buf * SLM_SUMS_B_SIZE + lane_id] = 0.0f;
                }
                const int base_idx = buf * SLM_B_SIZE + lane_id * XMX_K;
#pragma unroll
                for (int k = 0; k < XMX_K; k++) {
                    slm_B[base_idx + k] = 0;
                }
            }
        }
    };

    if (num_k_blocks > 0) {
        load_k_block(0, 0);
        item.barrier(sycl::access::fence_space::local_space);
    }

    for (int k_block = 0; k_block < num_k_blocks; k_block++) {
        const int buf_comp = k_block & 1;
        const int buf_load = (k_block + 1) & 1;

        if (k_block + 1 < num_k_blocks) {
            load_k_block(k_block + 1, buf_load);
        }

        [[maybe_unused]] int8_t q_local[XMX_K];
        if constexpr (PASSES == 2) {
            if (lane_id < XMX_M) {
                const int base_idx = buf_comp * SLM_A_SIZE + lane_id * XMX_K;
#pragma unroll
                for (int j = 0; j < XMX_K; j++) {
                    q_local[j] = slm_A[base_idx + j];
                }
            }
        }

        for (int pass = 0; pass < PASSES; pass++) {
            if (lane_id < XMX_M) {
                const int base_idx = buf_comp * SLM_A_SIZE + lane_id * XMX_K;
#pragma unroll
                for (int j = 0; j < XMX_K; j++) {
                    if constexpr (PASSES == 2) {
                        const bool keep = (pass == 0) ? (j < 16) : (j >= 16);
                        slm_A[base_idx + j] = keep ? q_local[j] : 0;
                    }
                }
            }

            item.barrier(sycl::access::fence_space::local_space);

            sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::a, XMX_M, XMX_K, sycl_xmx::layout::row_major>
                matA;
            sycl_xmx::joint_matrix<sycl::sub_group, int8_t, sycl_xmx::use::b, XMX_K, XMX_N, sycl_xmx::layout::col_major>
                matB;
            sycl_xmx::joint_matrix<sycl::sub_group, int32_t, sycl_xmx::use::accumulator, XMX_M, XMX_N> matC;

            auto A_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
                                                  sycl::access::decorated::no>(slm_A + buf_comp * SLM_A_SIZE);
            auto B_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
                                                  sycl::access::decorated::no>(slm_B + buf_comp * SLM_B_SIZE);
            auto C_ptr = sycl::address_space_cast<sycl::access::address_space::local_space,
                                                  sycl::access::decorated::no>(slm_C);

            sycl_xmx::joint_matrix_load(sg, matA, A_ptr, XMX_K);
            sycl_xmx::joint_matrix_load(sg, matB, B_ptr, XMX_K);
            sycl_xmx::joint_matrix_fill(sg, matC, 0);
            sycl_xmx::joint_matrix_mad(sg, matC, matA, matB, matC);
            sycl_xmx::joint_matrix_store(sg, matC, C_ptr, XMX_N, sycl_xmx::layout::row_major);

#pragma unroll
            for (int idx = lane_id, acc_idx = 0; idx < XMX_M * XMX_N; idx += XMX_SG_SIZE, acc_idx++) {
                const int   i    = idx / XMX_N;
                const int   j    = idx % XMX_N;
                const float d_A  = slm_scales_A[buf_comp * SLM_SCALES_A_SIZE + pass * XMX_M + i];
                const float o_A  = slm_offsets_A[buf_comp * SLM_OFFSETS_A_SIZE + pass * XMX_M + i];
                const float d_B  = slm_scales_B[buf_comp * SLM_SCALES_B_SIZE + j];
                float s_B = 0.0f;
                if constexpr (T == GGML_TYPE_Q2_K) {
                    s_B = slm_sums_B[buf_comp * SLM_SUMS_B_SIZE + pass * XMX_N + j];
                } else {
                    s_B = slm_sums_B[buf_comp * SLM_SUMS_B_SIZE + j];
                }
                const float C_ij = static_cast<float>(slm_C[idx]);
                acc[acc_idx] += d_A * (C_ij * d_B) + o_A * s_B;
            }

            item.barrier(sycl::access::fence_space::local_space);
        }

        item.barrier(sycl::access::fence_space::local_space);
    }

#pragma unroll
    for (int idx = lane_id, acc_idx = 0; idx < XMX_M * XMX_N; idx += XMX_SG_SIZE, acc_idx++) {
        const int i   = idx / XMX_N;
        const int j   = idx % XMX_N;
        const int row = row_base + i;
        const int col = col_base + j;
        if (row < row_high && col < ncols_y && j < TileCols) {
            dst[col * nrows_dst + row] = acc[acc_idx];
        }
    }
}

template <ggml_type T, bool SoA, int RowSubgroups, int TileCols>
static inline bool launch_xmx_typed(const ggml_sycl::mmvq_bench_args & args,
                                    std::vector<sycl::event> * events,
                                    std::string & error) {
    (void) error;
    const int row_low  = static_cast<int>(args.row_low);
    const int row_high = static_cast<int>(args.row_high);
    const int rows_span = row_high - row_low;
    const int tile_rows = RowSubgroups * XMX_M;
    const int num_row_tiles = (rows_span + tile_rows - 1) / tile_rows;
    const int num_col_tiles = (static_cast<int>(args.batch) + TileCols - 1) / TileCols;

    const int ncols_x = static_cast<int>(args.ncols);
    const int ncols_y = static_cast<int>(args.batch);
    const int nrows_dst = static_cast<int>(args.dst_row_stride);
    const int src1_stride_blocks = static_cast<int>(args.src1_padded_col_size / XMX_K);

    const int block_size = block_traits<T>::block_size;
    const int blocks_per_row = ncols_x / block_size;
    const size_t nblocks_total = static_cast<size_t>(blocks_per_row) * static_cast<size_t>(args.nrows);

    const void * weights = args.weights;
    const void * layout_base = args.layout_base;
    const void * activations = args.activations;
    float * output = args.output;

    sycl::queue & queue = *args.stream;
    const sycl::range<2> grid(num_row_tiles, num_col_tiles);
    const sycl::range<2> block(RowSubgroups, XMX_SG_SIZE);

    constexpr int TOTAL_SLM_SIZE =
        SLM_A_SIZE + SLM_B_SIZE + SLM_C_SIZE * 4 +
        SLM_SCALES_A_SIZE * 4 + SLM_OFFSETS_A_SIZE * 4 +
        SLM_SCALES_B_SIZE * 4 + SLM_SUMS_B_SIZE * 4;

    sycl::event ev = queue.submit([&](sycl::handler & h) {
        sycl::local_accessor<char, 1> shared_acc(sycl::range<1>(TOTAL_SLM_SIZE * RowSubgroups), h);

        h.parallel_for(sycl::nd_range<2>(grid * block, block),
                       [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(XMX_SG_SIZE)]] {
                           char * shared = shared_acc.get_multi_ptr<sycl::access::decorated::no>().get();

                           const int row_sg = item.get_local_id(0);
                           int offset = row_sg * TOTAL_SLM_SIZE;
                           int8_t * slm_A = reinterpret_cast<int8_t *>(shared + offset);
                           offset += SLM_A_SIZE;
                           int8_t * slm_B = reinterpret_cast<int8_t *>(shared + offset);
                           offset += SLM_B_SIZE;
                           int32_t * slm_C = reinterpret_cast<int32_t *>(shared + offset);
                           offset += SLM_C_SIZE * 4;
                           float * slm_scales_A = reinterpret_cast<float *>(shared + offset);
                           offset += SLM_SCALES_A_SIZE * 4;
                           float * slm_offsets_A = reinterpret_cast<float *>(shared + offset);
                           offset += SLM_OFFSETS_A_SIZE * 4;
                           float * slm_scales_B = reinterpret_cast<float *>(shared + offset);
                           offset += SLM_SCALES_B_SIZE * 4;
                           float * slm_sums_B = reinterpret_cast<float *>(shared + offset);

                           mmvq_tile_kernel<T, SoA, RowSubgroups, TileCols>(
                               weights, layout_base, activations, output,
                               ncols_x, ncols_y, nrows_dst,
                               row_low, row_high, src1_stride_blocks,
                               blocks_per_row, nblocks_total,
                               slm_A, slm_B, slm_C, slm_scales_A, slm_offsets_A, slm_scales_B, slm_sums_B, item);
                       });
    });

    if (events != nullptr) {
        events->push_back(ev);
    }
    return true;
}

template <ggml_type T, int RowSubgroups, int TileCols>
static inline bool launch_xmx_typed_db(const ggml_sycl::mmvq_bench_args & args,
                                       std::vector<sycl::event> * events,
                                       std::string & error) {
    (void) error;
    const int row_low  = static_cast<int>(args.row_low);
    const int row_high = static_cast<int>(args.row_high);
    const int rows_span = row_high - row_low;
    const int tile_rows = RowSubgroups * XMX_M;
    const int num_row_tiles = (rows_span + tile_rows - 1) / tile_rows;
    const int num_col_tiles = (static_cast<int>(args.batch) + TileCols - 1) / TileCols;

    const int ncols_x = static_cast<int>(args.ncols);
    const int ncols_y = static_cast<int>(args.batch);
    const int nrows_dst = static_cast<int>(args.dst_row_stride);
    const int src1_stride_blocks = static_cast<int>(args.src1_padded_col_size / XMX_K);

    const int block_size = block_traits<T>::block_size;
    const int blocks_per_row = ncols_x / block_size;

    const void * weights = args.weights;
    const void * activations = args.activations;
    float * output = args.output;

    sycl::queue & queue = *args.stream;
    const sycl::range<2> grid(num_row_tiles, num_col_tiles);
    const sycl::range<2> block(RowSubgroups, XMX_SG_SIZE);

    constexpr int SLM_A_DB_SIZE = SLM_A_SIZE * 2;
    constexpr int SLM_B_DB_SIZE = SLM_B_SIZE * 2;
    constexpr int SLM_SCALES_A_DB_SIZE = SLM_SCALES_A_SIZE * 2;
    constexpr int SLM_OFFSETS_A_DB_SIZE = SLM_OFFSETS_A_SIZE * 2;
    constexpr int SLM_SCALES_B_DB_SIZE = SLM_SCALES_B_SIZE * 2;
    constexpr int SLM_SUMS_B_DB_SIZE = SLM_SUMS_B_SIZE * 2;

    constexpr int TOTAL_SLM_SIZE =
        SLM_A_DB_SIZE + SLM_B_DB_SIZE + SLM_C_SIZE * 4 +
        SLM_SCALES_A_DB_SIZE * 4 + SLM_OFFSETS_A_DB_SIZE * 4 +
        SLM_SCALES_B_DB_SIZE * 4 + SLM_SUMS_B_DB_SIZE * 4;

    sycl::event ev = queue.submit([&](sycl::handler & h) {
        sycl::local_accessor<char, 1> shared_acc(sycl::range<1>(TOTAL_SLM_SIZE * RowSubgroups), h);

        h.parallel_for(sycl::nd_range<2>(grid * block, block),
                       [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(XMX_SG_SIZE)]] {
                           char * shared = shared_acc.get_multi_ptr<sycl::access::decorated::no>().get();

                           const int row_sg = item.get_local_id(0);
                           int offset = row_sg * TOTAL_SLM_SIZE;
                           int8_t * slm_A = reinterpret_cast<int8_t *>(shared + offset);
                           offset += SLM_A_DB_SIZE;
                           int8_t * slm_B = reinterpret_cast<int8_t *>(shared + offset);
                           offset += SLM_B_DB_SIZE;
                           int32_t * slm_C = reinterpret_cast<int32_t *>(shared + offset);
                           offset += SLM_C_SIZE * 4;
                           float * slm_scales_A = reinterpret_cast<float *>(shared + offset);
                           offset += SLM_SCALES_A_DB_SIZE * 4;
                           float * slm_offsets_A = reinterpret_cast<float *>(shared + offset);
                           offset += SLM_OFFSETS_A_DB_SIZE * 4;
                           float * slm_scales_B = reinterpret_cast<float *>(shared + offset);
                           offset += SLM_SCALES_B_DB_SIZE * 4;
                           float * slm_sums_B = reinterpret_cast<float *>(shared + offset);

                           mmvq_tile_kernel_db<T, RowSubgroups, TileCols>(
                               weights, activations, output,
                               ncols_x, ncols_y, nrows_dst,
                               row_low, row_high, src1_stride_blocks,
                               blocks_per_row,
                               slm_A, slm_B, slm_C,
                               slm_scales_A, slm_offsets_A, slm_scales_B, slm_sums_B, item);
                       });
    });

    if (events != nullptr) {
        events->push_back(ev);
    }
    return true;
}

template <int RowSubgroups, int TileCols>
static inline bool launch_xmx_dispatch_db(const ggml_sycl::mmvq_bench_args & args,
                                          std::vector<sycl::event> * events,
                                          std::string & error) {
    switch (args.weight_type) {
        case GGML_TYPE_Q4_0:
            return launch_xmx_typed_db<GGML_TYPE_Q4_0, RowSubgroups, TileCols>(args, events, error);
        case GGML_TYPE_Q8_0:
            return launch_xmx_typed_db<GGML_TYPE_Q8_0, RowSubgroups, TileCols>(args, events, error);
        case GGML_TYPE_Q6_K:
            return launch_xmx_typed_db<GGML_TYPE_Q6_K, RowSubgroups, TileCols>(args, events, error);
        case GGML_TYPE_MXFP4:
            return launch_xmx_typed_db<GGML_TYPE_MXFP4, RowSubgroups, TileCols>(args, events, error);
        case GGML_TYPE_Q4_K:
            return launch_xmx_typed_db<GGML_TYPE_Q4_K, RowSubgroups, TileCols>(args, events, error);
        case GGML_TYPE_Q2_K:
            return launch_xmx_typed_db<GGML_TYPE_Q2_K, RowSubgroups, TileCols>(args, events, error);
        case GGML_TYPE_Q3_K:
            return launch_xmx_typed_db<GGML_TYPE_Q3_K, RowSubgroups, TileCols>(args, events, error);
        case GGML_TYPE_Q5_K:
            return launch_xmx_typed_db<GGML_TYPE_Q5_K, RowSubgroups, TileCols>(args, events, error);
        default:
            error = "Tier2 kernel: unsupported quant type.";
            return false;
    }
}

template <bool SoA, int RowSubgroups, int TileCols>
static inline bool launch_xmx_dispatch(const ggml_sycl::mmvq_bench_args & args,
                                       std::vector<sycl::event> * events,
                                       std::string & error) {
    switch (args.weight_type) {
        case GGML_TYPE_Q4_0:
            return launch_xmx_typed<GGML_TYPE_Q4_0, SoA, RowSubgroups, TileCols>(args, events, error);
        case GGML_TYPE_Q8_0:
            return launch_xmx_typed<GGML_TYPE_Q8_0, SoA, RowSubgroups, TileCols>(args, events, error);
        case GGML_TYPE_Q6_K:
            return launch_xmx_typed<GGML_TYPE_Q6_K, SoA, RowSubgroups, TileCols>(args, events, error);
        case GGML_TYPE_MXFP4:
            return launch_xmx_typed<GGML_TYPE_MXFP4, SoA, RowSubgroups, TileCols>(args, events, error);
        case GGML_TYPE_Q4_K:
            return launch_xmx_typed<GGML_TYPE_Q4_K, SoA, RowSubgroups, TileCols>(args, events, error);
        case GGML_TYPE_Q2_K:
            return launch_xmx_typed<GGML_TYPE_Q2_K, SoA, RowSubgroups, TileCols>(args, events, error);
        case GGML_TYPE_Q3_K:
            return launch_xmx_typed<GGML_TYPE_Q3_K, SoA, RowSubgroups, TileCols>(args, events, error);
        case GGML_TYPE_Q5_K:
            return launch_xmx_typed<GGML_TYPE_Q5_K, SoA, RowSubgroups, TileCols>(args, events, error);
        default:
            error = "Tier2 kernel: unsupported quant type.";
            return false;
    }
}

#if MMVQ_TIER2_ESIMD_AVAILABLE

template <int TypeId, int RowsPerTile, int TileCols, bool ChainK>
class mmvq_esimd_dpas_kernel;

template <ggml_type T, int RowsPerTile, int TileCols, bool ChainK>
static inline bool launch_esimd_dpas_typed(const ggml_sycl::mmvq_bench_args & args,
                                           std::vector<sycl::event> * events,
                                           std::string & error) {
    (void) error;
    static_assert(TileCols == 16, "ESIMD dpas kernels assume 16 output columns.");
    static_assert(RowsPerTile >= 1 && RowsPerTile <= 8, "ESIMD dpas RepeatCount must be in [1, 8].");

    const int row_low  = static_cast<int>(args.row_low);
    const int row_high = static_cast<int>(args.row_high);
    const int rows_span = row_high - row_low;
    const int num_row_tiles = (rows_span + RowsPerTile - 1) / RowsPerTile;
    const int num_col_tiles = (static_cast<int>(args.batch) + TileCols - 1) / TileCols;

    const int ncols_x = static_cast<int>(args.ncols);
    const int ncols_y = static_cast<int>(args.batch);
    const int nrows_dst = static_cast<int>(args.dst_row_stride);
    const int src1_stride_blocks = static_cast<int>(args.src1_padded_col_size / XMX_K);

    const int block_size = block_traits<T>::block_size;
    const int blocks_per_row = ncols_x / block_size;

    const void * weights = args.weights;
    const void * activations = args.activations;
    float * output = args.output;

    sycl::queue & queue = *args.stream;
    const sycl::range<2> grid(num_row_tiles, num_col_tiles);
    const sycl::range<2> block(1, 1);

    sycl::event ev = queue.submit([&](sycl::handler & h) {
        h.parallel_for<mmvq_esimd_dpas_kernel<static_cast<int>(T), RowsPerTile, TileCols, ChainK>>(
            sycl::nd_range<2>(grid * block, block),
            [=](sycl::nd_item<2> item) SYCL_ESIMD_KERNEL {
                constexpr int HALF_COLS = XMX_N;
                constexpr int NUM_HALVES = TileCols / HALF_COLS;
                constexpr int SUBBLOCKS = block_traits<T>::block_size / XMX_K;
                constexpr int PASSES = block_traits<T>::per_16 ? 2 : 1;
                constexpr int AN = RowsPerTile * XMX_K;
                constexpr int BN = XMX_K * HALF_COLS;

                const int tile_row = static_cast<int>(item.get_global_id(0));
                const int tile_col = static_cast<int>(item.get_global_id(1));
                const int row_base = row_low + tile_row * RowsPerTile;
                const int col_base = tile_col * TileCols;

                if (row_base >= row_high || col_base >= ncols_y) {
                    return;
                }

                float acc[RowsPerTile][TileCols];
                for (int r = 0; r < RowsPerTile; ++r) {
                    for (int c = 0; c < TileCols; ++c) {
                        acc[r][c] = 0.0f;
                    }
                }

                const int num_k_blocks = ncols_x / XMX_K;
                const char * src1 = static_cast<const char *>(activations);
                constexpr int Q8_1_BLOCK_SIZE = sizeof(block_q8_1);

                constexpr int chain_iters = ChainK ? 2 : 1;
                for (int k_block = 0; k_block < num_k_blocks; k_block += chain_iters) {
                    for (int chain = 0; chain < chain_iters; ++chain) {
                        const int kb = k_block + chain;
                        if (kb >= num_k_blocks) {
                            break;
                        }

                        int8_t a_vals[RowsPerTile][XMX_K];
                        AParams params[RowsPerTile];
                        for (int r = 0; r < RowsPerTile; ++r) {
                            const int row = row_base + r;
                            if (row < row_high) {
                                const int block_idx = kb / SUBBLOCKS;
                                const int sub_idx = kb - block_idx * SUBBLOCKS;
                                decode_block_aos<T>(weights, row, blocks_per_row, block_idx, sub_idx, a_vals[r], params[r]);
                            } else {
#pragma unroll
                                for (int k = 0; k < XMX_K; ++k) {
                                    a_vals[r][k] = 0;
                                }
                                params[r].scale0  = 0.0f;
                                params[r].offset0 = 0.0f;
                                params[r].scale1  = 0.0f;
                                params[r].offset1 = 0.0f;
                            }
                        }

                        int8_t b_vals[TileCols][XMX_K];
                        float d_B[TileCols];
                        float s_B[2][TileCols];
                        for (int c = 0; c < TileCols; ++c) {
                            const int col = col_base + c;
                            if (col < ncols_y) {
                                const char * block_ptr = src1 +
                                    (static_cast<int64_t>(col) * src1_stride_blocks + kb) * Q8_1_BLOCK_SIZE;
                                const sycl::half d = *reinterpret_cast<const sycl::half *>(block_ptr);
                                const sycl::half s = *reinterpret_cast<const sycl::half *>(block_ptr + 2);
                                d_B[c] = float(d);
                                const int8_t * qs = reinterpret_cast<const int8_t *>(block_ptr + 4);
                                if constexpr (T == GGML_TYPE_Q2_K) {
                                    float sum0 = 0.0f;
                                    float sum1 = 0.0f;
                                    for (int k = 0; k < XMX_K; ++k) {
                                        const int8_t v = qs[k];
                                        if (k < 16) {
                                            sum0 += v;
                                        } else {
                                            sum1 += v;
                                        }
                                    }
                                    s_B[0][c] = d_B[c] * sum0;
                                    s_B[1][c] = d_B[c] * sum1;
                                } else {
                                    s_B[0][c] = float(s);
                                    s_B[1][c] = 0.0f;
                                }
#pragma unroll
                                for (int k = 0; k < XMX_K; ++k) {
                                    b_vals[c][k] = qs[k];
                                }
                            } else {
                                d_B[c] = 0.0f;
                                s_B[0][c] = 0.0f;
                                s_B[1][c] = 0.0f;
#pragma unroll
                                for (int k = 0; k < XMX_K; ++k) {
                                    b_vals[c][k] = 0;
                                }
                            }
                        }

                            for (int pass = 0; pass < PASSES; ++pass) {
                                esimd::simd<int8_t, AN> a_vec;
                                int a_idx = 0;
                                for (int r = 0; r < RowsPerTile; ++r) {
                                    for (int k = 0; k < XMX_K; ++k) {
                                        int8_t v = a_vals[r][k];
                                        if constexpr (PASSES == 2) {
                                            const bool keep = (pass == 0) ? (k < 16) : (k >= 16);
                                            v = keep ? v : 0;
                                        }
                                        a_vec[a_idx++] = v;
                                    }
                                }

                            for (int half = 0; half < NUM_HALVES; ++half) {
                                esimd::simd<int8_t, BN> b_vec;
                                int b_idx = 0;
                                const int col_offset = half * HALF_COLS;
                                for (int k4 = 0; k4 < (XMX_K / 4); ++k4) {
                                    for (int j = 0; j < HALF_COLS; ++j) {
                                        const int col = col_offset + j;
                                        const int base = k4 * 4;
                                        b_vec[b_idx++] = b_vals[col][base + 0];
                                        b_vec[b_idx++] = b_vals[col][base + 1];
                                        b_vec[b_idx++] = b_vals[col][base + 2];
                                        b_vec[b_idx++] = b_vals[col][base + 3];
                                    }
                                }

                                auto c_vec = esimd_xmx::dpas<8, RowsPerTile, int, int8_t, int8_t>(b_vec, a_vec);
                                for (int r = 0; r < RowsPerTile; ++r) {
                                    const float d_A = (pass == 0) ? params[r].scale0 : params[r].scale1;
                                    const float o_A = (pass == 0) ? params[r].offset0 : params[r].offset1;
                                    for (int j = 0; j < HALF_COLS; ++j) {
                                        const int col = col_offset + j;
                                        const int idx = r * HALF_COLS + j;
                                        const float C_ij = static_cast<float>(c_vec[idx]);
                                        float s_val = 0.0f;
                                        if constexpr (T == GGML_TYPE_Q2_K) {
                                            s_val = s_B[pass][col];
                                        } else {
                                            s_val = s_B[0][col];
                                        }
                                        acc[r][col] += d_A * (C_ij * d_B[col]) + o_A * s_val;
                                    }
                                }
                            }
                        }
                    }
                }

                for (int r = 0; r < RowsPerTile; ++r) {
                    const int row = row_base + r;
                    if (row >= row_high) {
                        continue;
                    }
                    for (int c = 0; c < TileCols; ++c) {
                        const int col = col_base + c;
                        if (col < ncols_y) {
                            output[col * nrows_dst + row] = acc[r][c];
                        }
                    }
                }
            });
    });

    if (events != nullptr) {
        events->push_back(ev);
    }
    return true;
}

template <int RowsPerTile, int TileCols, bool ChainK>
static inline bool launch_esimd_dpas_dispatch(const ggml_sycl::mmvq_bench_args & args,
                                              std::vector<sycl::event> * events,
                                              std::string & error) {
    switch (args.weight_type) {
        case GGML_TYPE_Q4_0:
            return launch_esimd_dpas_typed<GGML_TYPE_Q4_0, RowsPerTile, TileCols, ChainK>(args, events, error);
        case GGML_TYPE_Q8_0:
            return launch_esimd_dpas_typed<GGML_TYPE_Q8_0, RowsPerTile, TileCols, ChainK>(args, events, error);
        case GGML_TYPE_Q6_K:
            return launch_esimd_dpas_typed<GGML_TYPE_Q6_K, RowsPerTile, TileCols, ChainK>(args, events, error);
        case GGML_TYPE_MXFP4:
            return launch_esimd_dpas_typed<GGML_TYPE_MXFP4, RowsPerTile, TileCols, ChainK>(args, events, error);
        case GGML_TYPE_Q4_K:
            return launch_esimd_dpas_typed<GGML_TYPE_Q4_K, RowsPerTile, TileCols, ChainK>(args, events, error);
        case GGML_TYPE_Q2_K:
            return launch_esimd_dpas_typed<GGML_TYPE_Q2_K, RowsPerTile, TileCols, ChainK>(args, events, error);
        case GGML_TYPE_Q3_K:
            return launch_esimd_dpas_typed<GGML_TYPE_Q3_K, RowsPerTile, TileCols, ChainK>(args, events, error);
        case GGML_TYPE_Q5_K:
            return launch_esimd_dpas_typed<GGML_TYPE_Q5_K, RowsPerTile, TileCols, ChainK>(args, events, error);
        default:
            error = "Tier2 kernel: unsupported quant type.";
            return false;
    }
}

#endif  // MMVQ_TIER2_ESIMD_AVAILABLE

inline bool launch_xmx_tile_8x8(const ggml_sycl::mmvq_bench_args & args,
                                std::vector<sycl::event> * events,
                                std::string & error) {
    if (!validate_xmx_args(args, error)) {
        return false;
    }
    // Uses 8x16 joint_matrix with column masking to emulate 8x8 output.
    return launch_xmx_dispatch<false, 1, 8>(args, events, error);
}

inline bool launch_xmx_tile_16x16(const ggml_sycl::mmvq_bench_args & args,
                                  std::vector<sycl::event> * events,
                                  std::string & error) {
    if (!validate_xmx_args(args, error)) {
        return false;
    }
    return launch_xmx_dispatch<false, 2, 16>(args, events, error);
}

inline bool launch_xmx_aos_direct(const ggml_sycl::mmvq_bench_args & args,
                                  std::vector<sycl::event> * events,
                                  std::string & error) {
    if (!validate_xmx_args(args, error)) {
        return false;
    }
    return launch_xmx_dispatch<false, 1, 16>(args, events, error);
}

inline bool launch_xmx_soa_direct(const ggml_sycl::mmvq_bench_args & args,
                                  std::vector<sycl::event> * events,
                                  std::string & error) {
    if (!validate_xmx_soa_args(args, error)) {
        return false;
    }
    return launch_xmx_dispatch<true, 1, 16>(args, events, error);
}

inline bool launch_xmx_double_buffer(const ggml_sycl::mmvq_bench_args & args,
                                     std::vector<sycl::event> * events,
                                     std::string & error) {
    if (!validate_xmx_args(args, error)) {
        return false;
    }
    return launch_xmx_dispatch_db<1, 16>(args, events, error);
}

inline bool launch_esimd_dpas_1x16x32(const ggml_sycl::mmvq_bench_args & args,
                                      std::vector<sycl::event> * events,
                                      std::string & error) {
    if (!validate_xmx_args(args, error)) {
        return false;
    }
#if !MMVQ_TIER2_ESIMD_AVAILABLE
    error = "SYCL ESIMD unavailable; ESIMD dpas kernels disabled.";
    return false;
#else
    return launch_esimd_dpas_dispatch<1, 16, false>(args, events, error);
#endif
}

inline bool launch_esimd_dpas_8x16x32(const ggml_sycl::mmvq_bench_args & args,
                                      std::vector<sycl::event> * events,
                                      std::string & error) {
    if (!validate_xmx_args(args, error)) {
        return false;
    }
#if !MMVQ_TIER2_ESIMD_AVAILABLE
    error = "SYCL ESIMD unavailable; ESIMD dpas kernels disabled.";
    return false;
#else
    return launch_esimd_dpas_dispatch<8, 16, false>(args, events, error);
#endif
}

inline bool launch_esimd_dpas_chained(const ggml_sycl::mmvq_bench_args & args,
                                      std::vector<sycl::event> * events,
                                      std::string & error) {
    if (!validate_xmx_args(args, error)) {
        return false;
    }
#if !MMVQ_TIER2_ESIMD_AVAILABLE
    error = "SYCL ESIMD unavailable; ESIMD dpas kernels disabled.";
    return false;
#else
    return launch_esimd_dpas_dispatch<8, 16, true>(args, events, error);
#endif
}

#else  // MMVQ_TIER2_XMX_AVAILABLE

inline bool launch_xmx_tile_8x8(const ggml_sycl::mmvq_bench_args &, std::vector<sycl::event> *,
                                std::string & error) {
    error = "SYCL joint_matrix unavailable; tier2 kernels disabled.";
    return false;
}

inline bool launch_xmx_tile_16x16(const ggml_sycl::mmvq_bench_args &, std::vector<sycl::event> *,
                                  std::string & error) {
    error = "SYCL joint_matrix unavailable; tier2 kernels disabled.";
    return false;
}

inline bool launch_xmx_aos_direct(const ggml_sycl::mmvq_bench_args &, std::vector<sycl::event> *,
                                  std::string & error) {
    error = "SYCL joint_matrix unavailable; tier2 kernels disabled.";
    return false;
}

inline bool launch_xmx_soa_direct(const ggml_sycl::mmvq_bench_args &, std::vector<sycl::event> *,
                                  std::string & error) {
    error = "SYCL joint_matrix unavailable; tier2 kernels disabled.";
    return false;
}

inline bool launch_xmx_double_buffer(const ggml_sycl::mmvq_bench_args &, std::vector<sycl::event> *,
                                     std::string & error) {
    error = "SYCL joint_matrix unavailable; tier2 kernels disabled.";
    return false;
}

inline bool launch_esimd_dpas_1x16x32(const ggml_sycl::mmvq_bench_args &, std::vector<sycl::event> *,
                                      std::string & error) {
    error = "SYCL joint_matrix unavailable; tier2 kernels disabled.";
    return false;
}

inline bool launch_esimd_dpas_8x16x32(const ggml_sycl::mmvq_bench_args &, std::vector<sycl::event> *,
                                      std::string & error) {
    error = "SYCL joint_matrix unavailable; tier2 kernels disabled.";
    return false;
}

inline bool launch_esimd_dpas_chained(const ggml_sycl::mmvq_bench_args &, std::vector<sycl::event> *,
                                      std::string & error) {
    error = "SYCL joint_matrix unavailable; tier2 kernels disabled.";
    return false;
}

#endif  // MMVQ_TIER2_XMX_AVAILABLE

}  // namespace mmvq_tier2
}  // namespace sycl_bench
