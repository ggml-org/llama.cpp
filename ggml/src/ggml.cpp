#include "ggml-impl.h"
#include "ggml-cuda.h"

#if defined(GGML_USE_CUDA) || defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)
#include <atomic>
#endif
#include <cstdlib>
#include <exception>

static std::terminate_handler previous_terminate_handler;

GGML_NORETURN static void ggml_uncaught_exception() {
    ggml_print_backtrace();
    if (previous_terminate_handler) {
        previous_terminate_handler();
    }
    abort(); // unreachable unless previous_terminate_handler was nullptr
}

static bool ggml_uncaught_exception_init = []{
    const char * GGML_NO_BACKTRACE = getenv("GGML_NO_BACKTRACE");
    if (GGML_NO_BACKTRACE) {
        return false;
    }
    const auto prev{std::get_terminate()};
    GGML_ASSERT(prev != ggml_uncaught_exception);
    previous_terminate_handler = prev;
    std::set_terminate(ggml_uncaught_exception);
    return true;
}();

#if defined(GGML_USE_CUDA) || defined(GGML_USE_HIP) || defined(GGML_USE_MUSA)

#ifndef __cuda_cuda_h__
typedef void * cudaStream_t;
#endif

extern "C" int ggml_backend_cuda_get_device_count(void) __attribute__((weak));
extern "C" void ggml_cuda_nvfp4_register_autotune() __attribute__((weak));
extern "C" void ggml_cuda_nvfp4_autotune(const float * x, const float * qw, int64_t n, float * best_a, float * best_b, cudaStream_t stream) __attribute__((weak));
extern "C" bool ggml_cuda_nvfp4_quantize(
    const void * x, bool x_bf16, float x_scale,
    void * vy, int64_t nrow, int64_t n_per_row, const float * qw,
    float a, float b, cudaStream_t stream) __attribute__((weak));
extern "C" bool ggml_cuda_nvfp4_quantize_cfg(
    const void * x, bool x_bf16, float x_scale,
    void * vy, int64_t nrow, int64_t n_per_row, const float * qw,
    float a, float b, const nvfp4_cuda_runtime_cfg * cfg, cudaStream_t stream) __attribute__((weak));

static std::atomic<int> g_nvfp4_ab_valid{0};
static float g_nvfp4_a = NVFP4_A0;
static float g_nvfp4_b = NVFP4_B0;

static inline bool nvfp4_cuda_available() {
    return ggml_backend_cuda_get_device_count && ggml_backend_cuda_get_device_count() > 0;
}

static inline void nvfp4_register_autotune() {
    if (ggml_cuda_nvfp4_register_autotune) {
        ggml_cuda_nvfp4_register_autotune();
    }
}

static inline void nvfp4_get_ab(float * a_out, float * b_out) {
    float a = NVFP4_A0;
    float b = NVFP4_B0;
    if (g_nvfp4_ab_valid.load(std::memory_order_acquire) != 0) {
        a = g_nvfp4_a;
        b = g_nvfp4_b;
    }
    *a_out = a;
    *b_out = b;
}

extern "C" bool nvfp4_autotune(const float * x, const float * qw, int64_t n, float * best_a, float * best_b) {
    if (!nvfp4_cuda_available()) {
        return false;
    }
    nvfp4_register_autotune();
    if (!ggml_cuda_nvfp4_autotune) {
        return false;
    }
    ggml_cuda_nvfp4_autotune(x, qw, n, best_a, best_b, nullptr);
    return true;
}

extern "C" bool nvfp4_autotune_cuda(const float * x, const float * qw, int64_t n, float * best_a, float * best_b, void * stream) {
    if (!nvfp4_cuda_available()) {
        return false;
    }
    nvfp4_register_autotune();
    if (!ggml_cuda_nvfp4_autotune) {
        return false;
    }
    ggml_cuda_nvfp4_autotune(x, qw, n, best_a, best_b, (cudaStream_t) stream);
    return true;
}

extern "C" void nvfp4_set_ab(float a, float b) {
    g_nvfp4_a = a;
    g_nvfp4_b = b;
    g_nvfp4_ab_valid.store(1, std::memory_order_release);
}

extern "C" void nvfp4_clear_ab(void) {
    g_nvfp4_ab_valid.store(0, std::memory_order_release);
}

extern "C" bool nvfp4_quantize_cuda_cfg(
    const void * x, bool x_bf16, void * vy,
    int64_t nrow, int64_t n_per_row,
    const float * qw, float x_scale,
    const nvfp4_cuda_runtime_cfg * cfg, void * stream) {
    if (!nvfp4_cuda_available()) {
        return false;
    }

    float a = NVFP4_A0;
    float b = NVFP4_B0;
    nvfp4_get_ab(&a, &b);

    if (ggml_cuda_nvfp4_quantize_cfg) {
        return ggml_cuda_nvfp4_quantize_cfg(x, x_bf16, x_scale, vy, nrow, n_per_row, qw, a, b, cfg, (cudaStream_t) stream);
    }
    if (!ggml_cuda_nvfp4_quantize) {
        return false;
    }
    return ggml_cuda_nvfp4_quantize(x, x_bf16, x_scale, vy, nrow, n_per_row, qw, a, b, (cudaStream_t) stream);
}

extern "C" bool nvfp4_quantize_cuda(const void * x, bool x_bf16, void * vy, int64_t nrow, int64_t n_per_row, const float * qw, float x_scale, void * stream) {
    return nvfp4_quantize_cuda_cfg(x, x_bf16, vy, nrow, n_per_row, qw, x_scale, nullptr, stream);
}

extern "C" bool nvfp4_quantize_cuda_ab(const void * x, bool x_bf16, void * vy, int64_t nrow, int64_t n_per_row, const float * qw, float x_scale, float a, float b, void * stream) {
    if (!nvfp4_cuda_available()) {
        return false;
    }
    if (ggml_cuda_nvfp4_quantize_cfg) {
        return ggml_cuda_nvfp4_quantize_cfg(x, x_bf16, x_scale, vy, nrow, n_per_row, qw, a, b, nullptr, (cudaStream_t) stream);
    }
    if (!ggml_cuda_nvfp4_quantize) {
        return false;
    }
    return ggml_cuda_nvfp4_quantize(x, x_bf16, x_scale, vy, nrow, n_per_row, qw, a, b, (cudaStream_t) stream);
}

#else

extern "C" bool nvfp4_autotune(const float * x, const float * qw, int64_t n, float * best_a, float * best_b) {
    (void) x; (void) qw; (void) n; (void) best_a; (void) best_b;
    return false;
}

extern "C" bool nvfp4_autotune_cuda(const float * x, const float * qw, int64_t n, float * best_a, float * best_b, void * stream) {
    (void) x; (void) qw; (void) n; (void) best_a; (void) best_b; (void) stream;
    return false;
}

extern "C" void nvfp4_set_ab(float a, float b) {
    (void) a; (void) b;
}

extern "C" void nvfp4_clear_ab(void) {
}

extern "C" bool nvfp4_quantize_cuda_cfg(
    const void * x, bool x_bf16, void * vy,
    int64_t nrow, int64_t n_per_row,
    const float * qw, float x_scale,
    const nvfp4_cuda_runtime_cfg * cfg, void * stream) {
    (void) x; (void) x_bf16; (void) vy; (void) nrow; (void) n_per_row;
    (void) qw; (void) x_scale; (void) cfg; (void) stream;
    return false;
}

extern "C" bool nvfp4_quantize_cuda(const void * x, bool x_bf16, void * vy, int64_t nrow, int64_t n_per_row, const float * qw, float x_scale, void * stream) {
    return nvfp4_quantize_cuda_cfg(x, x_bf16, vy, nrow, n_per_row, qw, x_scale, nullptr, stream);
}

extern "C" bool nvfp4_quantize_cuda_ab(const void * x, bool x_bf16, void * vy, int64_t nrow, int64_t n_per_row, const float * qw, float x_scale, float a, float b, void * stream) {
    (void) a; (void) b;
    return nvfp4_quantize_cuda_cfg(x, x_bf16, vy, nrow, n_per_row, qw, x_scale, nullptr, stream);
}

#endif
