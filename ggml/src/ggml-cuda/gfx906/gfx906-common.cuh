#pragma once

#include "gfx906-config.h"

#ifdef GGML_USE_HIP

static __device__ __forceinline__ float sgpr_broadcast_f32(float value) {
    int i = __float_as_int(value);
    i = __builtin_amdgcn_readfirstlane(i);
    return __int_as_float(i);
}

static __device__ __forceinline__ int sgpr_broadcast_i32(int value) {
    return __builtin_amdgcn_readfirstlane(value);
}

static __device__ __forceinline__ half sgpr_broadcast_f16(half value) {
    int i = *reinterpret_cast<const short*>(&value);
    i = __builtin_amdgcn_readfirstlane(i);
    short s = static_cast<short>(i);
    return *reinterpret_cast<half*>(&s);
}

static __device__ __forceinline__ float fast_exp_f32(float x) {
    constexpr float LOG2_E = 1.4426950408889634f;
    float result;
    asm volatile(
        "v_exp_f32 %0, %1"
        : "=v"(result)
        : "v"(x * LOG2_E)
    );
    return result;
}

static __device__ __forceinline__ float fast_exp2_f32(float x) {
    float result;
    asm volatile(
        "v_exp_f32 %0, %1"
        : "=v"(result)
        : "v"(x)
    );
    return result;
}

static __device__ __forceinline__ float fast_log2_f32(float x) {
    float result;
    asm volatile(
        "v_log_f32 %0, %1"
        : "=v"(result)
        : "v"(x)
    );
    return result;
}

static __device__ __forceinline__ float fast_tanh_f32(float x) {
    if (x > 10.0f) return 1.0f;
    if (x < -10.0f) return -1.0f;

    const float exp2x = fast_exp_f32(2.0f * x);
    return 1.0f - 2.0f / (exp2x + 1.0f);
}

static __device__ __forceinline__ float fast_rcp_f32(float x) {
    float result;
    asm volatile(
        "v_rcp_f32 %0, %1"
        : "=v"(result)
        : "v"(x)
    );
    return result;
}

#define DEFINE_FUSED_DPP_F32(name, barrier, dpp_ctrl, vop_instr)           \
    static __device__ __forceinline__ float name(float x) {                \
        float result;                                                       \
        asm volatile(                                                       \
            barrier                                                         \
            vop_instr " %0, %1, %1 " dpp_ctrl " row_mask:0xf bank_mask:0xf" \
            : "=v"(result) : "v"(x) : "memory"                             \
        );                                                                  \
        return result;                                                      \
    }

DEFINE_FUSED_DPP_F32(hip_add_xor1_f32, "s_nop 4\n", "quad_perm:[1,0,3,2]", "v_add_f32_dpp")
DEFINE_FUSED_DPP_F32(hip_max_xor1_f32, "s_nop 4\n", "quad_perm:[1,0,3,2]", "v_max_f32_dpp")

DEFINE_FUSED_DPP_F32(hip_add_xor2_f32, "s_nop 1\n", "quad_perm:[2,3,0,1]", "v_add_f32_dpp")
DEFINE_FUSED_DPP_F32(hip_max_xor2_f32, "s_nop 1\n", "quad_perm:[2,3,0,1]", "v_max_f32_dpp")

DEFINE_FUSED_DPP_F32(hip_add_xor8_f32, "s_nop 1\n", "row_ror:8", "v_add_f32_dpp")
DEFINE_FUSED_DPP_F32(hip_max_xor8_f32, "s_nop 1\n", "row_ror:8", "v_max_f32_dpp")

#undef DEFINE_FUSED_DPP_F32

static __device__ __forceinline__ float hip_shuffle_xor4_f32(float x) {
    int v_src = __float_as_int(x);
    int v_dst;
    asm volatile(
        "v_mov_b32 %0, %1\n"
        "s_nop 1\n"
        "v_mov_b32_dpp %0, %1 row_shl:4 row_mask:0xf bank_mask:0x5\n"
        "v_mov_b32_dpp %0, %1 row_shr:4 row_mask:0xf bank_mask:0xa\n"
        : "=v"(v_dst) : "v"(v_src) : "memory"
    );
    return __int_as_float(v_dst);
}

static __device__ __forceinline__ float hip_shuffle_xor16_f32(float x) {
    int int_val = __float_as_int(x);
    int result;
    asm volatile(
        "ds_swizzle_b32 %0, %1 offset:swizzle(SWAP,16)\n"
        "s_waitcnt lgkmcnt(0)\n"
        : "=v"(result) : "v"(int_val) : "memory"
    );
    return __int_as_float(result);
}

struct AddOp {
    static __device__ __forceinline__ float apply(float a, float b) { return a + b; }
    static __device__ __forceinline__ float xor1(float x) { return hip_add_xor1_f32(x); }
    static __device__ __forceinline__ float xor2(float x) { return hip_add_xor2_f32(x); }
    static __device__ __forceinline__ float xor8(float x) { return hip_add_xor8_f32(x); }
};

struct MaxOp {
    static __device__ __forceinline__ float apply(float a, float b) { return fmaxf(a, b); }
    static __device__ __forceinline__ float xor1(float x) { return hip_max_xor1_f32(x); }
    static __device__ __forceinline__ float xor2(float x) { return hip_max_xor2_f32(x); }
    static __device__ __forceinline__ float xor8(float x) { return hip_max_xor8_f32(x); }
};

template<int width = WARP_SIZE, typename Op>
static __device__ __forceinline__ float warp_reduce_amd_f32(float x) {
    if (width >= 2)  x = Op::xor1(x);
    if (width >= 4)  x = Op::xor2(x);
    if (width >= 8)  x = Op::apply(x, hip_shuffle_xor4_f32(x));
    if (width >= 16) x = Op::xor8(x);
    if (width >= 32) x = Op::apply(x, hip_shuffle_xor16_f32(x));
    if (width == 64) x = Op::apply(x, __shfl_xor(x, 32, 64));
    return x;
}

template<typename T>
static __device__ __forceinline__ T hip_dpp_xor1(T value) {
    static_assert(sizeof(T) == 4, "DPP operations require 32-bit types");
    int int_val = *reinterpret_cast<int*>(&value);
    int result;
    asm volatile(
        "s_nop 4\n"
        "v_mov_b32_dpp %0, %1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf"
        : "=v"(result) : "v"(int_val) : "memory"
    );
    return *reinterpret_cast<T*>(&result);
}

template<typename T>
static __device__ __forceinline__ T hip_dpp_xor2(T value) {
    static_assert(sizeof(T) == 4, "DPP operations require 32-bit types");
    int int_val = *reinterpret_cast<int*>(&value);
    int result;
    asm volatile(
        "s_nop 1\n"
        "v_mov_b32_dpp %0, %1 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf"
        : "=v"(result) : "v"(int_val) : "memory"
    );
    return *reinterpret_cast<T*>(&result);
}

template<typename T>
static __device__ __forceinline__ T hip_dpp_xor4(T value) {
    static_assert(sizeof(T) == 4, "DPP operations require 32-bit types");
    int v_src = *reinterpret_cast<int*>(&value);
    int v_dst;
    asm volatile(
        "v_mov_b32 %0, %1\n"
        "s_nop 1\n"
        "v_mov_b32_dpp %0, %1 row_shl:4 row_mask:0xf bank_mask:0x5\n"
        "v_mov_b32_dpp %0, %1 row_shr:4 row_mask:0xf bank_mask:0xa\n"
        : "=v"(v_dst) : "v"(v_src) : "memory"
    );
    return *reinterpret_cast<T*>(&v_dst);
}

template<typename T>
static __device__ __forceinline__ T hip_dpp_xor8(T value) {
    static_assert(sizeof(T) == 4, "DPP operations require 32-bit types");
    int int_val = *reinterpret_cast<int*>(&value);
    int result;
    asm volatile(
        "s_nop 1\n"
        "v_mov_b32_dpp %0, %1 row_ror:8 row_mask:0xf bank_mask:0xf"
        : "=v"(result) : "v"(int_val) : "memory"
    );
    return *reinterpret_cast<T*>(&result);
}

template<typename T>
static __device__ __forceinline__ T hip_dpp_xor16(T value) {
    static_assert(sizeof(T) == 4, "DPP operations require 32-bit types");
    int int_val = *reinterpret_cast<int*>(&value);
    int result;
    asm volatile(
        "ds_swizzle_b32 %0, %1 offset:swizzle(SWAP,16)\n"
        "s_waitcnt lgkmcnt(0)\n"
        : "=v"(result) : "v"(int_val) : "memory"
    );
    return *reinterpret_cast<T*>(&result);
}

template<int width = WARP_SIZE, typename T>
static __device__ __forceinline__ T gfx906_shfl_xor_sync(T x, int offset) {
    switch (~offset) {
        case ~1:  return hip_dpp_xor1(x);
        case ~2:  return hip_dpp_xor2(x);
        case ~4:  return hip_dpp_xor4(x);
        case ~8:  return hip_dpp_xor8(x);
        case ~16: return hip_dpp_xor16(x);
        default:  return __shfl_xor(x, offset, width);
    }
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ float gfx906_warp_reduce_sum_f32(float x) {
    return warp_reduce_amd_f32<width, AddOp>(x);
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ float gfx906_warp_reduce_max_f32(float x) {
    return warp_reduce_amd_f32<width, MaxOp>(x);
}

template<int width = WARP_SIZE, typename T>
static __device__ __forceinline__ T gfx906_warp_reduce_sum_generic(T x) {
    #pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        x += gfx906_shfl_xor_sync<width>(x, offset);
    }
    return x;
}

// ================================================================================================
// Specialized N-thread reductions for quantization and other group operations
// These provide optimized DPP-based reductions for common group sizes (4, 8, 16, 32)
// ================================================================================================

// Generic N-thread reduction using DPP operations (N must be power of 2)
template<int N, typename Op>
static __device__ __forceinline__ float warp_reduce_n_amd_f32(float x) {
    static_assert((N & (N-1)) == 0 && N <= 64, "N must be power of 2 <= 64");
    // Build reduction tree from largest to smallest offset
    if constexpr (N > 32) x = Op::apply(x, __shfl_xor(x, 32, 64));
    if constexpr (N > 16) x = Op::apply(x, hip_shuffle_xor16_f32(x));
    if constexpr (N > 8)  x = Op::apply(x, hip_shuffle_xor4_f32(x));
    if constexpr (N > 4)  x = Op::xor2(x);
    if constexpr (N > 2)  x = Op::xor1(x);
    if constexpr (N > 1)  {
        // For N=2, we still need xor1
        if constexpr (N == 2) x = Op::xor1(x);
    }
    return x;
}

// Specialized 8-thread reduction for Q8_1 quantization (most common case)
// This is used when 8 threads cooperate to quantize a 32-element block
template<typename Op>
static __device__ __forceinline__ float warp_reduce_8_amd_f32(float x) {
    // 8-thread reduction tree: xor4 -> xor2 -> xor1
    x = Op::apply(x, hip_shuffle_xor4_f32(x));  // Reduce 8->4
    x = Op::xor2(x);                             // Reduce 4->2  
    x = Op::xor1(x);                             // Reduce 2->1
    return x;
}

// Public interface: 8-thread sum reduction
static __device__ __forceinline__ float warp_reduce_sum_8_f32(float x) {
    return warp_reduce_8_amd_f32<AddOp>(x);
}

// Public interface: 8-thread max reduction
static __device__ __forceinline__ float warp_reduce_max_8_f32(float x) {
    return warp_reduce_8_amd_f32<MaxOp>(x);
}

// Generic N-thread sum reduction (compile-time N)
template<int N>
static __device__ __forceinline__ float warp_reduce_sum_n_f32(float x) {
    return warp_reduce_n_amd_f32<N, AddOp>(x);
}

// Generic N-thread max reduction (compile-time N)
template<int N>
static __device__ __forceinline__ float warp_reduce_max_n_f32(float x) {
    return warp_reduce_n_amd_f32<N, MaxOp>(x);
}

// Convenience aliases that match common.cuh naming
template<int width = WARP_SIZE>
static __device__ __forceinline__ float gfx906_warp_reduce_sum(float x) {
    return gfx906_warp_reduce_sum_f32<width>(x);
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ float gfx906_warp_reduce_max(float x) {
    return gfx906_warp_reduce_max_f32<width>(x);
}

#endif
