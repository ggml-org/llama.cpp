#pragma once

// TurboQuant helpers used by the CPU quantizers.

#include "ggml.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void turboq_rotate_forward(float * y, const float * x, int64_t d, uint64_t seed);

void turboq_rotate_inverse(float * x, const float * y, int64_t d, uint64_t seed);

uint64_t turboq_seed_from_row(int64_t row_idx);

#ifdef __cplusplus
}
#endif
