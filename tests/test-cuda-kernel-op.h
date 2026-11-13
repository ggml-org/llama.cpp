#pragma once

#include <stdint.h>

// shared by the kernels (test-cuda-kernel-op.cu) and the test

#define TEST_BLOCK_SIZE 256

// the last argument of each kernel
struct test_kernel_params {
    const float * host;     // [n] in pinned host memory
    float         scale;
    float         add;
    int32_t       n;
    uint64_t      delay_ns; // the kernel waits before it writes: a consumer that does not wait for it reads old data
};
