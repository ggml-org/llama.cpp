#include <sycl/sycl.hpp>

#include <cstdio>
#include <cstdlib>
#include <string>

// Standalone link harness: satisfy the gather object's ggml imports without a backend or queue.
extern int g_ggml_sycl_xmx_gather_types;
int g_ggml_sycl_xmx_gather_types = 0;

extern "C" [[noreturn]] void ggml_abort(const char *, int, const char *, ...);
extern "C" [[noreturn]] void ggml_abort(const char *, int, const char *, ...) {
    std::abort();
}

bool ggml_sycl_fused_dequant_gemm_f16_device_ok(sycl::queue *);

int main() {
    // A static consumer pulls the archive member through an ordinary public entry point.
    auto * volatile device_ok = &ggml_sycl_fused_dequant_gemm_f16_device_ok;
    if (!device_ok) { return 1; }
    int named = 0;
    for (const auto & id : sycl::get_kernel_ids()) {
        if (std::string(id.get_name()).find("fused_dequant_gemm_kernel") != std::string::npos) {
            ++named;
        }
    }
    // One named kernel per supported IQ format must survive the separate device link.
    std::printf("registered gather kernels: %d (expected 9)\n", named);
    return named == 9 ? 0 : 1;
}
