// These tests may take a long time!
// They are to prove that conversion from double to float of various functions in ggml.c doesn't affect the result.
// This is done by checking all finite (non-NaN, non-infinite) floats.

#undef NDEBUG
#include <cassert>
#if !defined(__riscv) && !defined(__s390__) && !defined(__ARM_NEON)
#include <immintrin.h>
#endif
#include <cmath>
#include <cstdint>
#include <cstring>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdouble-promotion"

// ggml.c::quantize_row_q4_0_ref
inline static uint8_t round_orig(float v0) { return ((int8_t) (round(v0))) + 8; }

// ggml.c::ggml_silu_f32
inline static float silu_orig(float x) {
    return x/(1.0 + exp(-x));
}

#pragma GCC diagnostic pop

// ggml.c::quantize_row_q4_0_ref
inline static uint8_t round_float(float v0) { return (int8_t)roundf(v0) + 8; }

// ggml.c::ggml_silu_f32
inline static float silu_float(float x) {
    return x/(1.0f + expf(-x));
}

int main(void) {
    uint32_t x = UINT32_MAX;
    do {
        float f;
        memcpy(&f, &x, sizeof(x));
        assert(!std::isfinite(f) || (round_orig(f) == round_float(f)));
    } while (x--);

#ifdef __F16C__
    // GELU and SILU implementations are used with a FP16 lookup table.
    // The original and float-only results are not equal for all inputs after converting to FP16.
    // GELU is an approximation anyway (tanh), not tested here.
    // For SILU, verify that the results are at least the closest floating point numbers, if the FP16 values don't match.
    for (x = 0; x <= UINT16_MAX; x++) {
        float f = _cvtsh_ss(x);
        const float so = silu_orig(f);
        const float sf = silu_float(f);
        assert(   (_cvtss_sh(so, 0) == _cvtss_sh(sf, 0))
               || (nextafterf(so, sf) == sf)
               || (nextafterf(sf, so) == so));
    }
#endif
}
