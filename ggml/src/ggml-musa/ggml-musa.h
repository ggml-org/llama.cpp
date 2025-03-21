#pragma once

#include <map>
#include <iostream>
#include <unordered_map>
#include <musa_runtime.h>
#include <mudnn.h>

// FULL_MASK
#define MASK_SHFL_128 ((~(128 - 1)) & 0x7f) << 7 | (128 - 1)
#define MASK_SHFL_64 ((~(64 - 1)) & 0x7f) << 7 | (64 - 1)
#define MASK_SHFL_32 ((~(32 - 1)) & 0x7f) << 7 | (32 - 1)
#define MASK_SHFL_16 ((~(16 - 1)) & 0x7f) << 7 | (16 - 1)
#define MASK_SHFL_8 ((~(8 - 1)) & 0x7f) << 7 | (8 - 1)
#define MASK_SHFL_4 ((~(4 - 1)) & 0x7f) << 7 | (4 - 1)
#define MASK_SHFL_2 ((~(2 - 1)) & 0x7f) << 7 | (2 - 1)

#define MASK_SHFL_UP_128 ((~(128 - 1)) & 0x7f) << 7
#define MASK_SHFL_UP_64 ((~(64 - 1)) & 0x7f) << 7
#define MASK_SHFL_UP_32 ((~(32 - 1)) & 0x7f) << 7
#define MASK_SHFL_UP_16 ((~(16 - 1)) & 0x7f) << 7
#define MASK_SHFL_UP_8 ((~(8 - 1)) & 0x7f) << 7
#define MASK_SHFL_UP_4 ((~(4 - 1)) & 0x7f) << 7
#define MASK_SHFL_UP_2 ((~(2 - 1)) & 0x7f) << 7

template <typename T, int width>
__device__ __forceinline__ T musa_shfl_xor_sync(T val, int lane_mask) {
#if (defined(__MUSA_ARCH__) && (__MUSA_ARCH__ > 220))  // MUSIFY_EXCL_LINE
  return __shfl_xor_sync(0xffffffff, val, lane_mask, width);
#elif (defined(__MUSA_ARCH__) && (__MUSA_ARCH__ > 210))  // MUSIFY_EXCL_LINE
  static_assert((width >= 2) && (width <= 128) && ((width & (width - 1)) == 0));

  auto shfl_func = [&](int& var) {
    if constexpr (width == 128) {
      return __musa_shfl_xor_sync_i32(var, lane_mask & 0x7f, MASK_SHFL_128);
    } else if constexpr (width == 64) {
      return __musa_shfl_xor_sync_i32(var, lane_mask & 0x3f, MASK_SHFL_64);
    } else if constexpr (width == 32) {
      return __musa_shfl_xor_sync_i32(var, lane_mask & 0x1f, MASK_SHFL_32);
    } else if constexpr (width == 16) {
      return __musa_shfl_xor_sync_i32(var, lane_mask & 0xf, MASK_SHFL_16);
    } else if constexpr (width == 8) {
      return __musa_shfl_xor_sync_i32(var, lane_mask & 0x7, MASK_SHFL_8);
    } else if constexpr (width == 4) {
      return __musa_shfl_xor_sync_i32(var, lane_mask & 0x3, MASK_SHFL_4);
    } else if constexpr (width == 2) {
      return __musa_shfl_xor_sync_i32(var, lane_mask & 0x1, MASK_SHFL_2);
    }
  };

  if constexpr (sizeof(T) == 4) {
    int var = *(reinterpret_cast<int32_t*>(&val));
    int ret = shfl_func(var);
    return *(reinterpret_cast<T*>(&ret));
  } else {
    struct __Bits {
      int __a, __b;
    };
    __Bits __tmp;
    memcpy(&__tmp, &val, sizeof(val));
    __tmp.__a = shfl_func(__tmp.__a);
    __tmp.__b = shfl_func(__tmp.__b);
    int64_t ret = *(reinterpret_cast<int64_t*>(&__tmp));
    return *(reinterpret_cast<T*>(&ret));
  }
#else
  return 0;
#endif
}
