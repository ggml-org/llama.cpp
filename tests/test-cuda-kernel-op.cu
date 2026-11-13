// the kernels of test-cuda-kernel-op.cpp, nvcc compiles them to PTX (see CMakeLists.txt)

#include "test-cuda-kernel-op.h"

static __device__ uint64_t test_time() {
    uint64_t t = 0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// each block waits
static __device__ void test_delay(const uint64_t delay_ns) {
    if (threadIdx.x == 0 && delay_ns > 0) {
        const uint64_t t0 = test_time();
        while (test_time() - t0 < delay_ns) {
        }
    }
    __syncthreads();
}

// dst = x*scale + add
extern "C" __global__ void test_affine(float * dst, const float * x, const test_kernel_params p) {
    test_delay(p.delay_ns);
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < p.n) {
        dst[i] = x[i]*p.scale + p.add;
    }
}

// dst = x + host
extern "C" __global__ void test_add_host(float * dst, const float * x, const test_kernel_params p) {
    test_delay(p.delay_ns);
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < p.n) {
        dst[i] = x[i] + p.host[i];
    }
}

// dst = x with each block reversed, through dynamic shared memory
extern "C" __global__ void test_reverse(float * dst, const float * x, const test_kernel_params p) {
    extern __shared__ float s[];
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    s[threadIdx.x] = i < p.n ? x[i] : 0.0f;
    __syncthreads();
    if (i < p.n) {
        dst[i] = s[blockDim.x - 1 - threadIdx.x];
    }
}

// dst = a + b + c
extern "C" __global__ void test_sum3(float * dst, const float * a, const float * b, const float * c, const test_kernel_params p) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < p.n) {
        dst[i] = a[i] + b[i] + c[i];
    }
}
