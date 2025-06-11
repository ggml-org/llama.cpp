/***************************************************************************
 *
 *  Copyright (C) 2025 Codeplay Software Ltd.
 *  Copyright (C) 2025 Intel Corporation
 *
 *  MIT License
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  builtins.hpp
 *
 *  Description:
 *     Intel builtin specifics
 **************************************************************************/

#pragma once

#include <sycl/marray.hpp>

#include "ggml-sycl/dpct/helper.hpp"

#define GGML_SYCL_UNREACHABLE(x) \
    assert(0 && x);              \
    printf(x);

#ifdef __SYCL_DEVICE_ONLY__
template <class T, int N> using vector_t = T __attribute__((ext_vector_type(N)));
#else
template <class T, int N> using vector_t = sycl::marray<T, N>;
#endif

#ifdef __SYCL_DEVICE_ONLY__
#    define SYCL_DEVICE_BUILTIN(x) SYCL_EXTERNAL extern "C" x
#else
#    define SYCL_DEVICE_BUILTIN(x)                                                      \
        inline x {                                                                      \
            GGML_SYCL_UNREACHABLE("Attempting to use a device built-in in host code."); \
        }
#endif

#ifdef __SYCL_DEVICE_ONLY__
#    define SYCL_DEVICE_OCL(x) SYCL_EXTERNAL extern "C" x
#else
#    define SYCL_DEVICE_OCL(x)
#endif

namespace sycl::vector_types {

using ushort2  = vector_t<uint16_t, 2>;
using ushort4  = vector_t<uint16_t, 4>;
using ushort8  = vector_t<uint16_t, 8>;
using ushort16 = vector_t<uint16_t, 16>;

using uint16_t2  = vector_t<uint16_t, 2>;
using uint16_t4  = vector_t<uint16_t, 4>;
using uint16_t8  = vector_t<uint16_t, 8>;
using uint16_t16 = vector_t<uint16_t, 16>;

using uint32_t2  = vector_t<uint32_t, 2>;
using uint32_t8  = vector_t<uint32_t, 8>;
using uint32_t16 = vector_t<uint32_t, 16>;
using uint32_t32 = vector_t<uint32_t, 32>;

using int32_t2 = vector_t<int32_t, 2>;
}  // namespace sycl::vector_types

using coord_t = sycl::vector_types::int32_t2;

namespace sycl::detail {

// Avoid compilation warnings in the host side
// These are expected to be unused
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wreturn-type"

SYCL_DEVICE_BUILTIN(uint16_t __builtin_IB_subgroup_block_read_flat_u16_m1k16v1(intptr_t baseoffset, int width_minus_one,
                                                                               int height_minus_one,
                                                                               int pitch_minus_one, coord_t coord));
SYCL_DEVICE_BUILTIN(sycl::vector_types::ushort2 __builtin_IB_subgroup_block_read_flat_u16_m2k16v1(
    intptr_t baseoffset, int width_minus_one, int height_minus_one, int pitch_minus_one, coord_t coord));
SYCL_DEVICE_BUILTIN(sycl::vector_types::ushort4 __builtin_IB_subgroup_block_read_flat_u16_m4k16v1(
    intptr_t baseoffset, int width_minus_one, int height_minus_one, int pitch_minus_one, coord_t coord));
SYCL_DEVICE_BUILTIN(sycl::vector_types::ushort8 __builtin_IB_subgroup_block_read_flat_u16_m8k16v1(
    intptr_t baseoffset, int width_minus_one, int height_minus_one, int pitch_minus_one, coord_t coord));
SYCL_DEVICE_BUILTIN(sycl::vector_types::ushort16 __builtin_IB_subgroup_block_read_flat_u16_m16k16v1(
    intptr_t baseoffset, int width_minus_one, int height_minus_one, int pitch_minus_one, coord_t coord));
SYCL_DEVICE_BUILTIN(uint32_t __builtin_IB_subgroup_block_read_flat_u32_m1k16v1(
    intptr_t baseoffset, int width_minus_one, int height_minus_one, int pitch_minus_one, coord_t coord));

SYCL_EXTERNAL extern "C" int __builtin_IB_dp4a_ss(int c, int a, int b) __attribute__((const));

#pragma clang diagnostic pop

template <int ElementSize, int BlockWidth, int BlockHeight, int BlockCount> struct XeSubgroup2DBlockLoad {
    static_assert(false, "Unsupported 2D Block Load Configuration.");
};

template <> struct XeSubgroup2DBlockLoad<2, 16, 1, 1> {
    template <typename T>
    __dpct_inline__ void operator()(const void * srcBasePointer, int memoryWidth, int memoryHeight, int memoryPitch,
                                    coord_t coordinate, T * dstPointer) {
        *reinterpret_cast<uint16_t *>(dstPointer) = __builtin_IB_subgroup_block_read_flat_u16_m1k16v1(
            (intptr_t) (srcBasePointer), memoryWidth - 1, memoryHeight - 1, memoryPitch - 1, coordinate);
    }
};

template <> struct XeSubgroup2DBlockLoad<2, 16, 2, 1> {
    template <typename T>
    __dpct_inline__ void operator()(const void * srcBasePointer, int memoryWidth, int memoryHeight, int memoryPitch,
                                    coord_t coordinate, T * dstPointer) {
        *reinterpret_cast<sycl::vector_types::ushort2 *>(dstPointer) =
            __builtin_IB_subgroup_block_read_flat_u16_m2k16v1((intptr_t) (srcBasePointer), memoryWidth - 1,
                                                              memoryHeight - 1, memoryPitch - 1, coordinate);
    }
};

template <> struct XeSubgroup2DBlockLoad<2, 16, 4, 1> {
    template <typename T>
    __dpct_inline__ void operator()(const void * srcBasePointer, int memoryWidth, int memoryHeight, int memoryPitch,
                                    coord_t coordinate, T * dstPointer) {
        *reinterpret_cast<sycl::vector_types::ushort4 *>(dstPointer) =
            __builtin_IB_subgroup_block_read_flat_u16_m4k16v1((intptr_t) (srcBasePointer), memoryWidth - 1,
                                                              memoryHeight - 1, memoryPitch - 1, coordinate);
    }
};

template <> struct XeSubgroup2DBlockLoad<2, 16, 8, 1> {
    template <typename T>
    __dpct_inline__ void operator()(const void * srcBasePointer, int memoryWidth, int memoryHeight, int memoryPitch,
                                    coord_t coordinate, T * dstPointer) {
        *reinterpret_cast<sycl::vector_types::ushort8 *>(dstPointer) =
            __builtin_IB_subgroup_block_read_flat_u16_m8k16v1((intptr_t) (srcBasePointer), memoryWidth - 1,
                                                              memoryHeight - 1, memoryPitch - 1, coordinate);
    }
};

template <> struct XeSubgroup2DBlockLoad<2, 16, 16, 1> {
    template <typename T>
    __dpct_inline__ void operator()(const void * srcBasePointer, int memoryWidth, int memoryHeight, int memoryPitch,
                                    coord_t coordinate, T * dstPointer) {
        *reinterpret_cast<sycl::vector_types::ushort16 *>(dstPointer) =
            __builtin_IB_subgroup_block_read_flat_u16_m16k16v1((intptr_t) (srcBasePointer), memoryWidth - 1,
                                                               memoryHeight - 1, memoryPitch - 1, coordinate);
    }
};

template <> struct XeSubgroup2DBlockLoad<4, 16, 1, 1> {
    template <typename T>
    __dpct_inline__ void operator()(const void * srcBasePointer, int memoryWidth, int memoryHeight, int memoryPitch,
                                    coord_t coordinate, T * dstPointer) {
        *reinterpret_cast<uint32_t *>(dstPointer) = __builtin_IB_subgroup_block_read_flat_u32_m1k16v1(
            reinterpret_cast<long>(srcBasePointer), memoryWidth - 1, memoryHeight - 1, memoryPitch - 1, coordinate);
    }
};

}  // namespace sycl::detail
