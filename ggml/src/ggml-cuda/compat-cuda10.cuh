#pragma once

// Compatibility polyfills for CUDA 10.2 (Jetson TX2)
// bf16 types are mapped to fp16 since compute 6.2 has no hardware bf16 support.

#include <cuda_fp16.h>

// bf16 type polyfills (mapped to fp16 for compute 6.2)
typedef __half nv_bfloat16;

struct nv_bfloat162 {
    nv_bfloat16 x;
    nv_bfloat16 y;
};

static __host__ __device__ __forceinline__ nv_bfloat16 __float2bfloat16(float f) {
    return __float2half(f);
}

static __host__ __device__ __forceinline__ float __bfloat162float(nv_bfloat16 h) {
    return __half2float(h);
}

static __host__ __device__ __forceinline__ nv_bfloat162 make_bfloat162(nv_bfloat16 a, nv_bfloat16 b) {
    nv_bfloat162 r;
    r.x = a;
    r.y = b;
    return r;
}

static __host__ __device__ __forceinline__ nv_bfloat162 __float22bfloat162_rn(float2 f) {
    return make_bfloat162(__float2bfloat16(f.x), __float2bfloat16(f.y));
}

static __host__ __device__ __forceinline__ float2 __bfloat1622float2(nv_bfloat162 h) {
    return make_float2(__bfloat162float(h.x), __bfloat162float(h.y));
}

static __host__ __device__ __forceinline__ nv_bfloat16 __low2bfloat16(nv_bfloat162 h) {
    return h.x;
}

static __host__ __device__ __forceinline__ nv_bfloat16 __high2bfloat16(nv_bfloat162 h) {
    return h.y;
}

static __host__ __device__ __forceinline__ nv_bfloat162 __halves2bfloat162(nv_bfloat16 a, nv_bfloat16 b) {
    nv_bfloat162 r;
    r.x = a;
    r.y = b;
    return r;
}

// CUDA_R_16BF cublas data type (not defined in CUDA 10.2)
#ifndef CUDA_R_16BF
#define CUDA_R_16BF CUDA_R_16F
#endif
