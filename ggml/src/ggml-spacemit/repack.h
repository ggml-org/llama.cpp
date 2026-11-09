#pragma once

#include "ggml-common.h"
#include "ggml.h"

#include <cstddef>
#include <cstdint>

namespace ggml::cpu::riscv64_spacemit {

struct uint4x2_t {
    uint8_t val{ 0 };

    operator uint8_t() const noexcept { return (val & 0x0F); }

    explicit uint4x2_t(uint8_t v) noexcept { val = v & 0x0F; }
};

using _spe_fp16_ = _Float16;

// Each row/K32 block has its own scale and uint8 zero point; Q4_0 uses zp=8.
template <size_t, size_t, size_t> struct Q41Block;

template <> struct Q41Block<32, 32, 2> {
    // 32 rows x K32; each byte stores even K in the low nibble and odd K in the high nibble.
    uint4x2_t data[4 * 8 * 16];

    uint8_t    zp[32];
    _spe_fp16_ scale[32];
};

static_assert(sizeof(Q41Block<32, 32, 2>) == 608, "Q41Block layout changed");

template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS>
int repack(ggml_tensor * t, const void * data, size_t data_size);

// Q4_0 repack is lossless.
int repack_q4_0_to_q41(ggml_tensor * t, const void * data, size_t data_size);

// Q4_1/Q4_K retain nibbles and round offsets to uint8 zp; Q4_K also folds scales into FP16.
// Zero scale uses zp=0; fractional, clipped and nonzero constant offsets remain lossy.
int repack_q4_1_to_q41(ggml_tensor * t, const void * data, size_t data_size);
int repack_q4_k_to_q41(ggml_tensor * t, const void * data, size_t data_size);

// Readback restores source layout, not original Q4_1/Q4_K values; callers pass source byte counts.
int unpack_q41_to_q4_0_common(const ggml_tensor * t, void * data, size_t data_size);

// Q5_0 tensor converted exactly to Q8_0 and repacked into the q8_0 32x32 layout (allocation: Q8_0 size)
int repack_q5_0_to_q8_0_32(ggml_tensor * t, const void * data, size_t data_size);
int repack_f32_to_q8_0_32(ggml_tensor * t, const void * data, size_t data_size);
int unpack_q8_0_32_to_f32(const ggml_tensor * t, void * data, size_t data_size);

}  // namespace ggml::cpu::riscv64_spacemit
