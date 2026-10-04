#include "ggml-cuda/common.cuh"

// Compile these checks for every device target, without launching a kernel.
#if defined(__CUDA_ARCH__)
#if __CUDA_ARCH__ == 610
#if !defined(FP16_AVAILABLE) || defined(FAST_FP16_AVAILABLE) || defined(VOLTA_MMA_AVAILABLE) || defined(TURING_MMA_AVAILABLE) || defined(AMPERE_MMA_AVAILABLE) || defined(BLACKWELL_MMA_AVAILABLE) || defined(CP_ASYNC_AVAILABLE)
#error "SM61 must use the stock slow-FP16, non-MMA, non-cp.async feature set"
#endif
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
    }
    printf("Compiled-feature checks passed without a GPU\n");
    return 0;
#endif
}
