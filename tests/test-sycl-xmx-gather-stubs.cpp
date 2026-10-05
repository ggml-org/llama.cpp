#include "../ggml/src/ggml-sycl/fused-gemm.hpp"

#include <cstdio>
#include <cstdlib>

struct unused_pool final : ggml_sycl_pool {
    void * alloc(size_t, size_t *) override {
        std::fputs("XMX gather stub attempted an allocation\n", stderr);
        std::abort();
    }

    void free(void *, size_t) override {
        std::fputs("XMX gather stub attempted a free\n", stderr);
        std::abort();
    }
};

int main() {
    unused_pool pool;
    std::vector<ggml_sycl_gg_tile> tiles;
    const int64_t offsets[] = { 0, 32 };

    // Valid shapes still return false without reading buffers or requiring a device queue.
    const bool results[] = {
        ggml_sycl_fused_dequant_gemm_f16_device_ok(nullptr),
        ggml_sycl_fused_dequant_gemm_f16(
            GGML_TYPE_IQ4_NL, nullptr, nullptr, nullptr, 16, 32, 256, 16, pool, nullptr),
        ggml_sycl_grouped_dequant_gemm_f16(
            GGML_TYPE_IQ4_NL, nullptr, 0, nullptr, nullptr, offsets,
            1, 16, 256, 32, tiles, pool, nullptr),
    };
    for (size_t i = 0; i < sizeof(results) / sizeof(results[0]); ++i) {
        if (results[i]) {
            std::fprintf(stderr, "XMX gather stub %zu unexpectedly accepted work\n", i);
            return 1;
        }
    }
    if (!tiles.empty()) {
        std::fputs("XMX gather stub modified tile scratch\n", stderr);
        return 1;
    }
    return 0;
}
