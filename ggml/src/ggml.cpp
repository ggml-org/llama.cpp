#include "ggml-impl.h"

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

extern "C" bool nvfp4_autotune(const float * x, const float * qw, int64_t n, float * best_a, float * best_b) {
    // This is the public CUDA autotune hook for NVFP4 export.
    // The nvfp4-cpu branch keeps CUDA disabled and leaves CPU tuning in place.
    GGML_UNUSED(x);
    GGML_UNUSED(qw);
    GGML_UNUSED(n);
    GGML_UNUSED(best_a);
    GGML_UNUSED(best_b);
    return false;
}

extern "C" bool nvfp4_autotune_cuda(const float * x, const float * qw, int64_t n, float * best_a, float * best_b, void * stream) {
    // This is the stream-aware CUDA autotune hook for NVFP4 export.
    // The nvfp4-cpu branch leaves the symbol in place but does not use CUDA.
    GGML_UNUSED(x);
    GGML_UNUSED(qw);
    GGML_UNUSED(n);
    GGML_UNUSED(best_a);
    GGML_UNUSED(best_b);
    GGML_UNUSED(stream);
    return false;
}

extern "C" void nvfp4_set_ab(float a, float b) {
    // This stores CUDA-side tuning parameters for NVFP4 quantization.
    // The nvfp4-cpu branch keeps the API but does not retain CUDA state.
    GGML_UNUSED(a);
    GGML_UNUSED(b);
}

extern "C" void nvfp4_clear_ab(void) {
    // This clears CUDA-side tuning parameters for NVFP4 quantization.
    // The nvfp4-cpu branch keeps the API but has no CUDA state to clear.
}

extern "C" bool nvfp4_quantize_cuda_cfg(
    const void * x, bool x_bf16, void * vy,
    int64_t nrow, int64_t n_per_row,
    const float * qw, float x_scale,
    const void * cfg, void * stream) {
    // This is the configurable CUDA quantization hook for NVFP4 export.
    // The nvfp4-cpu branch keeps the call shape stable and always falls back to CPU.
    GGML_UNUSED(x);
    GGML_UNUSED(x_bf16);
    GGML_UNUSED(vy);
    GGML_UNUSED(nrow);
    GGML_UNUSED(n_per_row);
    GGML_UNUSED(qw);
    GGML_UNUSED(x_scale);
    GGML_UNUSED(cfg);
    GGML_UNUSED(stream);
    return false;
}

extern "C" bool nvfp4_quantize_cuda(const void * x, bool x_bf16, void * vy, int64_t nrow, int64_t n_per_row, const float * qw, float x_scale, void * stream) {
    return nvfp4_quantize_cuda_cfg(x, x_bf16, vy, nrow, n_per_row, qw, x_scale, nullptr, stream);
}

extern "C" bool nvfp4_quantize_cuda_ab(const void * x, bool x_bf16, void * vy, int64_t nrow, int64_t n_per_row, const float * qw, float x_scale, float a, float b, void * stream) {
    // This is the explicit-(a,b) CUDA quantization hook for NVFP4 export.
    // The nvfp4-cpu branch keeps the API but always declines CUDA quantization.
    GGML_UNUSED(x);
    GGML_UNUSED(x_bf16);
    GGML_UNUSED(vy);
    GGML_UNUSED(nrow);
    GGML_UNUSED(n_per_row);
    GGML_UNUSED(qw);
    GGML_UNUSED(x_scale);
    GGML_UNUSED(a);
    GGML_UNUSED(b);
    GGML_UNUSED(stream);
    return false;
}
