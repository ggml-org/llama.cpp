#include "common.cuh"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>

// Pass operands as launch arguments, as in the backend kernels.
static __global__ void divide(uint32_t n, uint3 divisor, uint2 * result) {
    *result = fast_div_modulo(n, divisor);
}

static void check_runtime(cudaError_t status, const char * operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(status));
        std::exit(2);
    }
}

int main() {
    const uint2 cases[] = {
        {UINT32_MAX, 1},              // shift by zero
        {0xbfffffffu, 3},             // immediately before the carry
        {0xc0000000u, 3},             // hi + n == 2^32
        {UINT32_MAX - 1, UINT32_MAX}, // shift by 32, no carry
        {UINT32_MAX, UINT32_MAX},     // shift by 32 with carry
    };

    uint2 * device_result = nullptr;
    check_runtime(cudaMalloc((void **) &device_result, sizeof(uint2)), "allocate result");
    constexpr size_t case_count = sizeof(cases) / sizeof(cases[0]);
    size_t failures = 0;
    for (const auto & input : cases) {
        const uint32_t n = input.x;
        const uint32_t d = input.y;
        divide<<<1, 1>>>(n, init_fastdiv_values(d), device_result);
        check_runtime(cudaGetLastError(), "launch divide");
        uint2 result;
        check_runtime(cudaMemcpy(&result, device_result, sizeof(result), cudaMemcpyDeviceToHost), "read result");
        if (result.x != n / d || result.y != n % d) {
            std::printf("FAIL n=%" PRIu32 " d=%" PRIu32 " got=(%" PRIu32 ",%" PRIu32
                        ") expected=(%" PRIu32 ",%" PRIu32 ")\n", n, d, result.x, result.y, n / d, n % d);
            ++failures;
        }
    }
    check_runtime(cudaFree(device_result), "free result");
    std::printf("%zu/%zu cases passed\n", case_count - failures, case_count);
    return failures == 0 ? 0 : 1;
}
