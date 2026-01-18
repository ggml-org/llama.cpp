// Validate layout-specific byte calculations for SYCL layouts.
//
// Usage:
//   ONEAPI_DEVICE_SELECTOR=level_zero:1 ./build/bin/test-layout-bytes

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-sycl.h"
#include "ggml-sycl/ggml-sycl-test.hpp"
#include "ggml-sycl/common.hpp"
#include "ggml-quants.h"

#if !defined(GGML_USE_SYCL)
int main() {
    std::fprintf(stderr, "GGML_USE_SYCL not enabled; skipping test.\n");
    return 0;
}
#else

static bool expect_eq(const char * label, size_t got, size_t expected) {
    if (got != expected) {
        std::fprintf(stderr, "%s: expected %zu, got %zu\n", label, expected, got);
        return false;
    }
    return true;
}

int main() {
    if (!std::getenv("ONEAPI_DEVICE_SELECTOR")) {
        setenv("ONEAPI_DEVICE_SELECTOR", "level_zero:1", 1);
    }

    ggml_backend_t backend = ggml_backend_sycl_init(0);
    if (!backend) {
        std::fprintf(stderr, "SKIP: SYCL backend unavailable\n");
        return 0;
    }

    ggml_init_params params = {
        /*.mem_size   =*/ 8 * 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        ggml_backend_free(backend);
        std::fprintf(stderr, "FAIL: failed to init ggml context\n");
        return 1;
    }

    bool ok = true;
    const int device = 0;

    // Q4_0 coalesced bytes.
    {
        const int64_t ncols = QK4_0 * 8;
        const int64_t nrows = 3;
        ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, ncols, nrows);
        const size_t blocks_per_row = static_cast<size_t>(ncols / QK4_0);
        const size_t nblocks = blocks_per_row * static_cast<size_t>(nrows);
        const size_t expected = static_cast<size_t>(nrows) * static_cast<size_t>(ncols / 2) +
                                nblocks * sizeof(ggml_half);
        const size_t got = ggml_sycl::test_layout_bytes(t, GGML_LAYOUT_COALESCED, device);
        ok &= expect_eq("Q4_0 coalesced bytes", got, expected);
    }

    // Q8_0 coalesced bytes.
    {
        const int64_t ncols = QK8_0 * 5;
        const int64_t nrows = 2;
        ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, ncols, nrows);
        const size_t blocks_per_row = static_cast<size_t>(ncols / QK8_0);
        const size_t nblocks = blocks_per_row * static_cast<size_t>(nrows);
        const size_t expected = static_cast<size_t>(nrows) * static_cast<size_t>(ncols) +
                                nblocks * sizeof(ggml_half);
        const size_t got = ggml_sycl::test_layout_bytes(t, GGML_LAYOUT_COALESCED, device);
        ok &= expect_eq("Q8_0 coalesced bytes", got, expected);
    }

    // Q6_K coalesced bytes.
    {
        const int64_t ncols = QK_K * 7;
        const int64_t nrows = 4;
        ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, ncols, nrows);
        const size_t blocks_per_row = static_cast<size_t>(ncols / QK_K);
        const size_t nblocks = blocks_per_row * static_cast<size_t>(nrows);
        const size_t row_quants = blocks_per_row * (128 + 64 + 16);
        const size_t expected = row_quants * static_cast<size_t>(nrows) + nblocks * sizeof(ggml_half);
        const size_t got = ggml_sycl::test_layout_bytes(t, GGML_LAYOUT_COALESCED, device);
        ok &= expect_eq("Q6_K coalesced bytes", got, expected);
    }

    // MXFP4 coalesced bytes.
    {
        const int64_t ncols = QK_MXFP4 * 4;
        const int64_t nrows = 5;
        ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_MXFP4, ncols, nrows);
        const size_t blocks_per_row = static_cast<size_t>(ncols / QK_MXFP4);
        const size_t nblocks = blocks_per_row * static_cast<size_t>(nrows);
        const size_t expected = static_cast<size_t>(nrows) * static_cast<size_t>(ncols / 2) + nblocks;
        const size_t got = ggml_sycl::test_layout_bytes(t, GGML_LAYOUT_COALESCED, device);
        ok &= expect_eq("MXFP4 coalesced bytes", got, expected);
    }

    // XMX GEMM tiled bytes (if supported).
    {
        const auto & caps = ggml_sycl_info().devices[device].xmx_caps;
        if (caps.supported && caps.supports_int8 && caps.M > 0) {
            const int64_t ncols = QK8_0 * 4;
            const int64_t nrows = static_cast<int64_t>(caps.M) + 1;
            ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, ncols, nrows);
            const int64_t tile_m = static_cast<int64_t>(caps.M);
            const int64_t tiles_m = (nrows + tile_m - 1) / tile_m;
            const int64_t blocks_per_row = ncols / QK8_0;
            const size_t block_bytes = ggml_type_size(GGML_TYPE_Q8_0);
            const size_t expected = static_cast<size_t>(tiles_m) * static_cast<size_t>(blocks_per_row) *
                                    static_cast<size_t>(tile_m) * block_bytes;
            const size_t got = ggml_sycl::test_layout_bytes(t, GGML_LAYOUT_XMX_GEMM_TILED, device);
            ok &= expect_eq("XMX GEMM tiled bytes", got, expected);
        }
    }

    ggml_free(ctx);
    ggml_backend_free(backend);

    if (!ok) {
        std::fprintf(stderr, "Layout bytes test: FAIL\n");
        return 1;
    }
    std::printf("Layout bytes test: PASS\n");
    return 0;
}

#endif
