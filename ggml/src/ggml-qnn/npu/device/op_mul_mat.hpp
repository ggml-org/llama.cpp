#pragma once

#include <hexagon_types.h>

#include "op_types.hpp"
#include "tensor.hpp"

namespace hexagon {

constexpr const size_t kBytesPerVector      = sizeof(HVX_Vector);  // 128 for v73
constexpr const size_t kAlignMask           = kBytesPerVector - 1;
constexpr const size_t kL2CacheSize         = 8 * 1024;            // // 8KB L2 cache
constexpr const size_t kL2FetchAheadVectors = kL2CacheSize / kBytesPerVector;

inline size_t unaligned_bytes(const void * addr) {
    return ((size_t) addr) & kAlignMask;
}

inline bool is_addr_aligned(void * addr) {
    return unaligned_bytes(addr) == 0;
}

inline void l2fetch(const void * p, uint32_t stride, uint32_t width, uint32_t height, uint32_t dir) {
    uint64_t control = HEXAGON_V64_CREATE_H(dir, stride, width, height);
    __asm__ __volatile__(" l2fetch(%0,%1) " : : "r"(p), "r"(control));
}

inline void l2fetch_row(const uint8_t * curr_row, size_t bytes) {
    // TODO: should we use small kL2FetchAheadVectors?
    int32_t l2fetch_vectors = Q6_R_min_RR(bytes / kBytesPerVector, kL2FetchAheadVectors);
    hexagon::l2fetch(curr_row, kBytesPerVector, kBytesPerVector, l2fetch_vectors, 0);
}

inline float get_flt0_from_fltv(HVX_Vector vect) {
    // See also: tools\HEXAGON_Tools\8.6.07\Examples\StandAlone_Applications\QFloat\QFloat.c

    union {
        int32_t i;
        float   f;
    } cvt;

    cvt.i = vect[0];
    return cvt.f;
}

bool mul_mat_f32(tensor * out, compute_params * params);
bool is_mul_mat_supported(const npu_device_tensor_spec & src0, const npu_device_tensor_spec & src1,
                          const npu_device_tensor_spec & dst, npu_device_tensor_op op);

}  // namespace hexagon
