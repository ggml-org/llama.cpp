// Unit tests for ggml_fp32_to_ue4m3 / ggml_ue4m3_to_fp32 (Fixes #31010)

#include "ggml.h"
#include "../ggml/src/ggml-impl.h"

#undef NDEBUG
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

static float bits_to_f32(uint32_t bits) {
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static float ue4m3_raw(uint8_t code) {
    // Decode without the 0.5 kvalues convention factor
    if (code == 0 || code == 0x7F) {
        return 0.0f;
    }
    const int exp = (code >> 3) & 0xF;
    const int man = code & 0x7;
    if (exp == 0) {
        return ldexpf((float) man, -9);
    }
    return ldexpf(1.0f + (float) man / 8.0f, exp - 7);
}

int main(void) {
    // Cases from https://github.com/ggml-org/llama.cpp/issues/31010
    struct {
        uint32_t in_bits;
        uint8_t  expected;
    } cases[] = {
        { 0x43800000u, 0x78 }, // 256
        { 0x43960000u, 0x79 }, // 300 -> 288
        { 0x43C80000u, 0x7C }, // 400 -> 384
        { 0x43C00000u, 0x7C }, // 384
        { 0x43D7FFFFu, 0x7D }, // 431.99997 -> 416
        { 0x3C700000u, 0x08 }, // 0.0146484375 (tie) -> 0.015625
        { 0x3C757CA8u, 0x08 }, // 0.0149833 -> 0.015625
        { 0x3A800000u, 0x00 }, // 0.0009765625 (tie) -> 0
        { 0x3F880000u, 0x38 }, // 1.0625 (tie) -> 1
        { 0x43480000u, 0x74 }, // 200 (tie) -> 192
    };

    int n_fail = 0;
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); ++i) {
        const float   x   = bits_to_f32(cases[i].in_bits);
        const uint8_t got = ggml_fp32_to_ue4m3(x);
        if (got != cases[i].expected) {
            printf("FAIL case %zu: bits=0x%08X x=%g got=0x%02X expected=0x%02X (raw got=%g exp=%g)\n",
                   i, cases[i].in_bits, x, got, cases[i].expected,
                   ue4m3_raw(got), ue4m3_raw(cases[i].expected));
            n_fail++;
        }
    }

    // Edges around 256 / max finite / subnormals
    struct {
        float   x;
        uint8_t expected;
    } edges[] = {
        { 0.0f,        0x00 },
        { -1.0f,       0x00 },
        { 448.0f,      0x7E },
        { 449.0f,      0x7E },
        { 240.0f,      0x77 },
        { 248.0f,      0x78 }, // midpoint 240|256 -> even 256
        { 256.0f,      0x78 },
        { 416.0f,      0x7D },
        { 432.0f,      0x7E }, // midpoint 416|448 -> even 448
        { 0.015625f,   0x08 }, // smallest normal
        { 0.001953125f,0x01 }, // 2^-9
        { 1.0f,        0x38 },
        { 1.125f,      0x39 },
    };

    for (size_t i = 0; i < sizeof(edges) / sizeof(edges[0]); ++i) {
        const uint8_t got = ggml_fp32_to_ue4m3(edges[i].x);
        if (got != edges[i].expected) {
            printf("FAIL edge %zu: x=%g got=0x%02X expected=0x%02X\n",
                   i, edges[i].x, got, edges[i].expected);
            n_fail++;
        }
    }

    // Round-trip: every finite code's raw value encodes back to itself
    for (int code = 1; code <= 0x7E; ++code) {
        const float   raw = ue4m3_raw((uint8_t) code);
        const uint8_t got = ggml_fp32_to_ue4m3(raw);
        if (got != (uint8_t) code) {
            printf("FAIL roundtrip code=0x%02X raw=%g got=0x%02X\n", code, raw, got);
            n_fail++;
        }
    }

    // Decoder convention: ggml_ue4m3_to_fp32 returns raw * 0.5
    assert(ggml_ue4m3_to_fp32(0x78) == 128.0f); // 256 * 0.5
    assert(ggml_ue4m3_to_fp32(0x7E) == 224.0f); // 448 * 0.5
    assert(ggml_ue4m3_to_fp32(0x7F) == 0.0f);
    assert(ggml_ue4m3_to_fp32(0x00) == 0.0f);

    if (n_fail != 0) {
        printf("%d failure(s)\n", n_fail);
        return 1;
    }
    printf("ue4m3 conversion tests OK\n");
    return 0;
}
