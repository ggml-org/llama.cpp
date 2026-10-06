#include "ggml-cuda/common.cuh"
#include "ggml-cuda/fattn-mma-f16.cuh"

// Unused launch helpers can survive an unoptimized compile. Metadata tests must never enter them.
int ggml_cuda_get_device() { std::abort(); }
[[noreturn]] void ggml_cuda_error(const char *, const char *, const char *, int, const char *) { std::abort(); }

// Compile these checks for every device target, without launching a kernel.
#if defined(__CUDA_ARCH__)
#if __CUDA_ARCH__ == 610
#if !defined(FP16_AVAILABLE) || defined(FAST_FP16_AVAILABLE) || defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMPERE_MMA_AVAILABLE) || defined(BLACKWELL_MMA_AVAILABLE) || defined(CP_ASYNC_AVAILABLE)
#error "SM61 must use the stock slow-FP16, non-MMA, non-cp.async feature set"
#endif
#endif
#if __CUDA_ARCH__ == 750 || __CUDA_ARCH__ == 860
static_assert(ggml_cuda_fattn_mma_get_nstages(256,256,4,2) == (__CUDA_ARCH__ == 750 ? 0 : 2),"keep stock generation-specific staging");
static_assert(ggml_cuda_fattn_mma_get_nstages(256,256,2,8) == (__CUDA_ARCH__ == 750 ? 0 : 2),"keep TG2 stock staging");
static_assert(ggml_cuda_fattn_mma_get_nstages(256,256,4,8) == (__CUDA_ARCH__ == 750 ? 0 : 2),"keep TG3/TG4 stock staging");
static_assert(ggml_cuda_fattn_mma_get_nthreads(256,256,8) == (__CUDA_ARCH__ == 750 ? 128 : 64),"keep stock thread geometry");
#endif
#if __CUDA_ARCH__ < 750 && defined(TURING_MMA_AVAILABLE)
#error "Turing MMA is not available on this target"
#endif
#if __CUDA_ARCH__ < 800 && (defined(AMPERE_MMA_AVAILABLE) || defined(CP_ASYNC_AVAILABLE))
#error "Ampere MMA and cp.async are not available on this target"
#endif
#if __CUDA_ARCH__ < 1200 && defined(BLACKWELL_MMA_AVAILABLE)
#error "Blackwell MMA is not available on this target"
#endif
#if __CUDA_ARCH__ >= 750 && !defined(TURING_MMA_AVAILABLE)
#error "Turing MMA is missing from this target"
#endif
#if __CUDA_ARCH__ >= 800 && (!defined(AMPERE_MMA_AVAILABLE) || !defined(CP_ASYNC_AVAILABLE))
#error "Ampere MMA or cp.async is missing from this target"
#endif
#if __CUDA_ARCH__ >= 1200 && __CUDA_ARCH__ < 1300 && !defined(BLACKWELL_MMA_AVAILABLE)
#error "Blackwell MMA is missing from this target"
#endif
#endif

// Exercise stock compiled-feature queries without initializing a CUDA device.
int main() {
#ifndef __CUDA_ARCH_LIST__
    fprintf(stderr, "SKIP: this compiler does not expose NVCC's compiled architecture list\n");
    return 77;
#else
    const int architectures[] = {__CUDA_ARCH_LIST__};
    printf("Compiled CUDA architectures:");
    for (const int arch : architectures) {
        printf(" %d", arch);
    }
    printf("\n");
#ifdef GGML_USE_VMM
    printf("VMM pool: compiled (driver-gated)\n");
#else
    printf("VMM pool: disabled\n");
#endif
#ifdef GGML_CUDA_USE_GRAPHS
    printf("CUDA graph cache: compiled (device-gated)\n");
#else
    printf("CUDA graph cache: disabled\n");
#endif
#ifdef GGML_CUDA_USE_PDL
    printf("PDL launch wrapper: compiled (kernel/environment-gated)\n");
#else
    printf("PDL launch wrapper: unavailable\n");
#endif
#ifdef GGML_CUDA_NO_FA
    printf("Flash attention: disabled\n");
#elif defined(GGML_CUDA_FA_ALL_QUANTS)
    printf("Flash attention: all supported KV quant pairs compiled\n");
#else
    printf("Flash attention: default KV quant pairs compiled\n");
#endif

    const int devices[] = {500, 600, 610, 620, 700, 750, 800, 860, 890, 900, 1000, 1200, 1210, 1300};
    for (const int device : devices) {
        int expected = -1;
        for (const int arch : architectures) {
            if (arch <= device && arch > expected) {
                expected = arch;
            }
        }
        const bool fp16 = expected >= 600;
        if (ggml_cuda_highest_compiled_arch(device) != expected ||
            fp16_available(device) != fp16 ||
            fast_fp16_available(device) != (fp16 && expected != 610) ||
            volta_mma_available(device) != (expected == 700) ||
            turing_mma_available(device) != (expected >= 750) ||
            ampere_mma_available(device) != (expected >= 800) ||
            cp_async_available(device) != (expected >= 800) ||
            blackwell_mma_available(device) != (expected >= 1200 && expected < 1300)) {
            fprintf(stderr, "Incorrect compiled-feature report for device %d (compiled %d)\n", device, expected);
            return 1;
        }
        printf("Query CC %d: compiled=%d fp16=%d fast_fp16=%d turing_mma=%d ampere_mma=%d cp_async=%d blackwell_mma=%d\n",
                device, expected, fp16_available(device), fast_fp16_available(device),
                turing_mma_available(device), ampere_mma_available(device),
                cp_async_available(device), blackwell_mma_available(device));
        if (turing_mma_available(device)) {
            for (auto columns : {8,16,32}) {
                const auto config=ggml_cuda_fattn_mma_get_config(256,256,columns,device);
                const int c2=columns == 8 ? 2 : 8, c1=columns/c2;
                const int stages=ggml_cuda_fattn_mma_get_nstages(256,256,c1,c2,device);
                if (stages != (expected >= 800 ? 2 : 0) ||
                        config.nthreads != (expected >= 800 && columns <= 16 ? 64 : 128)) return 1;
                printf("  head-256 columns=%d threads=%d stages=%d\n",columns,config.nthreads,stages);
            }
        }
    }
    printf("Compiled-feature checks passed without a GPU\n");
    return 0;
#endif
}
