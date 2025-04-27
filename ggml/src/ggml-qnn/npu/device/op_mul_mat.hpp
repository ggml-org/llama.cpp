#pragma once

#include <hexagon_types.h>

#include <cstdint>

#include "tensor.hpp"

namespace hexagon {

constexpr const size_t kBytesPerVector  = sizeof(HVX_Vector);  // 128 for v73
constexpr const size_t kFloatsPerVector = kBytesPerVector / sizeof(float);
constexpr const size_t kAlignMask       = kBytesPerVector - 1;

inline size_t unaligned_bytes(const void * addr) {
    return ((size_t) addr) & kAlignMask;
}

inline bool is_addr_aligned(void * addr) {
    return unaligned_bytes(addr) == 0;
}

bool mul_mat_f32(tensor * out, size_t tidx, size_t tcnt);
bool is_mul_mat_supported(const npu_device_tensor_spec & src0, const npu_device_tensor_spec & src1,
                          const npu_device_tensor_spec & dst, npu_device_tensor_op op);

}  // namespace hexagon
