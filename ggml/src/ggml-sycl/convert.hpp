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

#ifndef GGML_SYCL_CONVERT_HPP
#define GGML_SYCL_CONVERT_HPP

#include "common.hpp"

template <typename T>
using to_t_sycl_t = void (*)(const void * __restrict__ x, T * __restrict__ y, int64_t k, dpct::queue_ptr stream);
typedef to_t_sycl_t<float>      to_fp32_sycl_t;
typedef to_t_sycl_t<sycl::half> to_fp16_sycl_t;

to_fp16_sycl_t ggml_get_to_fp16_sycl(ggml_type type, ggml_tensor * dst);
to_fp32_sycl_t ggml_get_to_fp32_sycl(ggml_type type, ggml_tensor * dst);

#ifdef GGML_SYCL_HAS_BF16
typedef to_t_sycl_t<sycl::ext::oneapi::bfloat16> to_bf16_sycl_t;
to_bf16_sycl_t ggml_get_to_bf16_sycl(ggml_type type, ggml_tensor * dst);
#endif

// Nc = Non-contiguous
template <typename T>
using to_t_nc_sycl_t = void (*)(const void * x, T * y, int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
                                   int64_t s01, int64_t s02, int64_t s03, dpct::queue_ptr queue);

typedef to_t_nc_sycl_t<sycl::half> to_fp16_nc_sycl_t;
to_fp16_nc_sycl_t ggml_get_to_fp16_nc_sycl(ggml_type type);

// ggml_fp32_to_f8_e4m3 for device code: the same steps, with the frexpf of a normal value read from its bits.
// AOT device images have no frexpf, and the kernels that call it share an image with every ADD and REPEAT kernel
// (binbcast.cpp), so none of those could load.
static inline uint8_t ggml_sycl_fp32_to_f8_e4m3(float x) {
    const uint8_t sign = sycl::signbit(x) ? 0x80 : 0;
    x                  = sycl::fabs(x);
    if (sycl::isnan(x)) {
        return 0x7F;
    }
    if (x == 0.0f) {
        return sign;
    }
    if (sycl::isinf(x) || x >= 448.0f) {
        return sign | 0x7E;
    }
    if (x < 0.015625f) {
        return sign | (uint8_t) ggml_round_to_nearest_even(x * 512.0f);
    }
    // x >= 2^-6 is normal: frexpf gives exponent (biased - 126) and twice its mantissa is x with exponent 0
    const uint32_t bits        = sycl::bit_cast<uint32_t>(x);
    const float    mantissa    = sycl::bit_cast<float>((bits & 0x7FFFFFu) | 0x3F800000u);
    int            encoded_exp = (int) ((bits >> 23) & 0xFF) - 126 + 6;
    int            encoded_man = ggml_round_to_nearest_even((mantissa - 1.0f) * 8.0f);
    if (encoded_man == 8) {
        encoded_man = 0;
        encoded_exp++;
    }
    if (encoded_exp > 15 || (encoded_exp == 15 && encoded_man > 6)) {
        return sign | 0x7E;
    }
    return sign | (uint8_t) (encoded_exp << 3) | (uint8_t) encoded_man;
}

template<typename dst_t, typename src_t>
 inline dst_t ggml_sycl_cast(src_t x) {
    if constexpr (std::is_same_v<dst_t, src_t>) {
        return x;
    } else if constexpr (std::is_same_v<dst_t, ggml_fp8_e4m3_t>) {
        // ggml_fp8_e4m3_t is a struct, so this branch comes before the float(x) paths
        return { ggml_sycl_fp32_to_f8_e4m3(float(x)) };
    } else if constexpr (std::is_same_v<src_t, ggml_fp8_e4m3_t>) {
        return dst_t(ggml_f8_e4m3_to_fp32(x.bits));
#ifdef GGML_SYCL_HAS_BF16
    } else if constexpr (std::is_same_v<dst_t, sycl::ext::oneapi::bfloat16>) {
        return sycl::ext::oneapi::bfloat16(float(x));
    } else if constexpr (std::is_same_v<src_t, sycl::ext::oneapi::bfloat16>) {
        return static_cast<float>(x);
#endif
    } else if constexpr (std::is_same_v<src_t, sycl::float2> && std::is_same_v<dst_t, sycl::half2>) {
        return x.template convert<sycl::half, sycl::rounding_mode::rte>();
#ifdef GGML_SYCL_HAS_BF16
    } else if constexpr (std::is_same_v<src_t, sycl::float2> &&
                         std::is_same_v<dst_t, sycl::vec<sycl::ext::oneapi::bfloat16, 2>>) {
        return {x.x, x.y};
#endif
    } else if constexpr(std::is_same_v<dst_t, int32_t>) {
        return int32_t(x);
    } else {
        return float(x);
    }
}


#endif  // GGML_SYCL_CONVERT_HPP
