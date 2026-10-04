//
// MIT license
// SPDX-License-Identifier: MIT
//

#ifndef GGML_SYCL_IQ4NL_HPP
#define GGML_SYCL_IQ4NL_HPP

#include <cstdint>

#include "dpct/helper.hpp"

// One quarter (4 entries, selected by index bits 0 and 1) of the kvalues_iq4nl lookup in
// iq4nl_lookup4, written as k0 ^ (d1 & m0) ^ (d2 & m1) ^ (d3 & m0 & m1) with byte-replicated
// constants, so each constant is the immediate of a single and/xor.
static __dpct_inline__ uint32_t iq4nl_quarter(const int q, const uint32_t m0, const uint32_t m1, const uint32_t m01) {
    constexpr uint8_t k[16] = { 0x81, 0x98, 0xAD, 0xBF, 0xCF, 0xDD, 0xEA, 0xF6, 0x01, 0x0D, 0x19, 0x26, 0x35, 0x45, 0x59, 0x71 };
    const uint32_t k0 = k[4 * q + 0] * 0x01010101u;
    const uint32_t d1 = (k[4 * q + 0] ^ k[4 * q + 1]) * 0x01010101u;
    const uint32_t d2 = (k[4 * q + 0] ^ k[4 * q + 2]) * 0x01010101u;
    const uint32_t d3 = (k[4 * q + 0] ^ k[4 * q + 1] ^ k[4 * q + 2] ^ k[4 * q + 3]) * 0x01010101u;
    return k0 ^ ((d1 & m0) ^ (d2 & m1) ^ (d3 & m01));
}

// Looks up kvalues_iq4nl for the four indices held in the low nibbles of the bytes of x (the high
// nibbles must be zero), entirely in registers: the byte gathers of get_int_from_table_16 saturate
// the send queue on Xe. Byte-wide masks of the index bits pick the entry from each quarter of the
// table and then the quarter.
static __dpct_inline__ int iq4nl_lookup4(const uint32_t x) {
    const uint32_t m0  = ((x >> 0) & 0x01010101u) * 0xffu;
    const uint32_t m1  = ((x >> 1) & 0x01010101u) * 0xffu;
    const uint32_t m2  = ((x >> 2) & 0x01010101u) * 0xffu;
    const uint32_t m3  = ((x >> 3) & 0x01010101u) * 0xffu;
    const uint32_t m01 = m0 & m1;

    const uint32_t g0 = iq4nl_quarter(0, m0, m1, m01);
    const uint32_t g1 = iq4nl_quarter(1, m0, m1, m01);
    const uint32_t g2 = iq4nl_quarter(2, m0, m1, m01);
    const uint32_t g3 = iq4nl_quarter(3, m0, m1, m01);

    auto sel = [](const uint32_t a, const uint32_t b, const uint32_t m) { return (a & ~m) | (b & m); };

    return (int) sel(sel(g0, g1, m2), sel(g2, g3, m2), m3);
}

#endif // GGML_SYCL_IQ4NL_HPP
