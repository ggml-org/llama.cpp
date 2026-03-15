// Copyright (C) 2025 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT

#include "hadamard.cuh"

template <int nh>
static __global__ void hadamard_f32(const char * src, char * dst, int ne0,
        size_t nb01, size_t nb02, size_t nb03, size_t nb1, size_t nb2, size_t nb3) {

    constexpr float ksqrt2 = 0.707106781f;

    int nc  = ne0/nh;
    int ii1 = blockIdx.x;
    int i1  = ii1 / nc;
    int ic  = ii1 % nc;
    int i2  = blockIdx.y;
    int i3  = blockIdx.z;

    int tid = threadIdx.x;

    const float * x = (const float *)((const char *)src + i1*nb01 + i2*nb02 + i3*nb03) + ic*nh;
          float * y = (      float *)((const char *)dst + i1*nb1  + i2*nb2  + i3*nb3)  + ic*nh;

    __shared__ float ys[nh];

    ys[2*tid+0] = x[2*tid+0] + x[2*tid+1];
    ys[2*tid+1] = x[2*tid+0] - x[2*tid+1];

    float scale = ksqrt2;

#pragma unroll
    for (int h = 2; h < nh; h <<= 2) {
        __syncthreads();
        int ii = tid/h, jj = tid%h;
        int j = 2*h*ii+jj;
        float u = ys[j], v = ys[j+h];
        ys[j+0] = u + v;
        ys[j+h] = u - v;
        scale *= ksqrt2;
    }

    __syncthreads();
    y[2*tid+0] = ys[2*tid+0] * scale;
    y[2*tid+1] = ys[2*tid+1] * scale;
}

static void hadamard_f32_cuda(int nh, const char * x, char * y, int ne0, int ne1, int ne2, int ne3,
        size_t nb01, size_t nb02, size_t nb03, size_t nb1, size_t nb2, size_t nb3, cudaStream_t stream) {
    int nc = ne0/nh;
    int nrows = nc*ne1;
    dim3 num_blocks = dim3(nrows, ne2, ne3);
    switch (nh) {
        case  64: hadamard_f32< 64><<<num_blocks,  32, 0, stream>>>(x, y, ne0, nb01, nb02, nb03, nb1, nb2, nb3); break;
        case 128: hadamard_f32<128><<<num_blocks,  64, 0, stream>>>(x, y, ne0, nb01, nb02, nb03, nb1, nb2, nb3); break;
        case 256: hadamard_f32<256><<<num_blocks, 128, 0, stream>>>(x, y, ne0, nb01, nb02, nb03, nb1, nb2, nb3); break;
        default: GGML_ABORT("Unsupported Hadamard block size");
    }
}

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#include <intrin.h>
#include <ammintrin.h>
#include <nmmintrin.h>
#include <immintrin.h>
#include <stdlib.h>
static inline int popcount(uint32_t x) { return __popcnt(x); }
#else
static inline int popcount(uint32_t x) { return __builtin_popcount(x); }
#endif


void ggml_cuda_op_hadamard(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    GGML_ASSERT(src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_are_same_shape(src, dst));

    int nh = dst->op_params[0];
    GGML_ASSERT(dst->ne[0]%nh == 0);
    GGML_ASSERT(nh > 1 && popcount(nh) == 1);

    hadamard_f32_cuda(nh, (const char *)src->data, (char *)dst->data, src->ne[0], src->ne[1], src->ne[2], src->ne[3],
            src->nb[1], src->nb[2], src->nb[3], dst->nb[1], dst->nb[2], dst->nb[3], ctx.stream());
}
