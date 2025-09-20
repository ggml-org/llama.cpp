#pragma once

#include <cstddef>

namespace sqnbitgemm_spacemit_ime
{
size_t
SQ4BitGemmKernel_CompInt8(size_t BlkLen,
                          const std::byte* QuantA,
                          const std::byte* QuantBData,
                          const float* QuantBScale,
                          const std::byte* QuantBZeroPoint,
                          float* C,
                          size_t CountM,
                          size_t CountN,
                          size_t CountK,
                          size_t BlockCountK,
                          size_t ldc,
                          const float* Bias,
                          const size_t ScaleStride);
void
QuantizeARow_CompInt8(size_t BlkLen, const float* A, size_t CountK, std::byte* QuantA);

void
QuantizeAM4Row_CompInt8(size_t BlkLen, const float* A, size_t CountK, std::byte* QuantA);

}
