#include <asm/hwprobe.h>
#include <asm/unistd.h>
#include <unistd.h>

#include "ggml-cpu.h"
#include "quants.h"

extern "C" {
#include "kernels.inc"
}

#if defined(__riscv_v) && __riscv_v >= 1000000

// helper macros for runtime kernel dispatch

#define RVV_VEC_DOT_DISPATCH_PAIR(func_name, MINVLEN, SUFFIX)  \
    if (vlenb >= MINVLEN) {                                    \
        return func_name##SUFFIX;                              \
    }

#define RVV_VEC_DOT_DISPATCH_2(func_name, c1, s1)      \
    RVV_VEC_DOT_DISPATCH_PAIR(func_name, c1, s1)

#define RVV_VEC_DOT_DISPATCH_4(func_name, c1, s1, ...) \
    RVV_VEC_DOT_DISPATCH_PAIR(func_name, c1, s1)       \
    RVV_VEC_DOT_DISPATCH_2(func_name, __VA_ARGS__)

#define RVV_VEC_DOT_DISPATCH_6(func_name, c1, s1, ...) \
    RVV_VEC_DOT_DISPATCH_PAIR(func_name, c1, s1)       \
    RVV_VEC_DOT_DISPATCH_4(func_name, __VA_ARGS__)
// add more if needed

#define GET_RVV_VEC_DOT_DISPATCH_MACRO(_1, _2, _3, _4, _5, _6, NAME, ...) NAME

#define RVV_VEC_DOT_DISPATCH_CHECKS(func_name, ...)                                           \
    GET_RVV_VEC_DOT_DISPATCH_MACRO(__VA_ARGS__, RVV_VEC_DOT_DISPATCH_6,                       \
                           SKIP, RVV_VEC_DOT_DISPATCH_4,                                      \
                           SKIP, RVV_VEC_DOT_DISPATCH_2)(func_name, __VA_ARGS__)

#define RVV_VEC_DOT_DISPATCH(func_name, ...)                                          \
    static ggml_vec_dot_t func_name##_kernel_sel() {                                  \
        int vlenb = dispatch_vlenb;                                                   \
        RVV_VEC_DOT_DISPATCH_CHECKS(func_name, __VA_ARGS__)                           \
        return func_name##_generic;                                                   \
    }                                                                                 \
    static ggml_vec_dot_t func_name##_kernel = func_name##_kernel_sel();              \
    void func_name(int n, float * GGML_RESTRICT s, size_t bs,                         \
                   const void * GGML_RESTRICT vx, size_t bx,                          \
                   const void * GGML_RESTRICT vy, size_t by, int nrc) {               \
        (func_name##_kernel)(n, s, bs, vx, bx, vy, by, nrc);                          \
    }

#include <riscv_vector.h>

static bool probe_rvv() {
    bool has_rvv = false;

    struct riscv_hwprobe probe;
    probe.key = RISCV_HWPROBE_KEY_IMA_EXT_0;
    probe.value = 0;

    int ret = syscall(__NR_riscv_hwprobe, &probe, 1, 0, NULL, 0);

    if (0 == ret) {
        has_rvv = !!(probe.value & RISCV_HWPROBE_IMA_V);
    }

    return has_rvv;
}

static int probe_vlenb() {
    if (probe_rvv()) {
        return __riscv_vlenb();
    }
    return 0;
}
static int dispatch_vlenb = probe_vlenb();

#elif defined(__riscv_xtheadvector)

void ggml_vec_dot_q5_K_q8_K_071(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy,  size_t by, int nrc) {
    ggml_vec_dot_q5_K_q8_K_generic(n, s, bs, vx, bx, vy, by, nrc);
}

#define RVV_VEC_DOT_DISPATCH(func_name, ...)                                          \
    void func_name(int n, float * GGML_RESTRICT s, size_t bs,                         \
                   const void * GGML_RESTRICT vx, size_t bx,                          \
                   const void * GGML_RESTRICT vy, size_t by, int nrc) {               \
        (func_name##_071)(n, s, bs, vx, bx, vy, by, nrc);                             \
    }

#else

#define RVV_VEC_DOT_DISPATCH(func_name, ...)                                          \
    void func_name(int n, float * GGML_RESTRICT s, size_t bs,                         \
                   const void * GGML_RESTRICT vx, size_t bx,                          \
                   const void * GGML_RESTRICT vy, size_t by, int nrc) {               \
        (func_name##_generic)(n, s, bs, vx, bx, vy, by, nrc);                         \
    }

#endif

extern "C" {

RVV_VEC_DOT_DISPATCH(ggml_vec_dot_q2_K_q8_K, 32, _256, 16, _128)
RVV_VEC_DOT_DISPATCH(ggml_vec_dot_q3_K_q8_K, 32, _256, 16, _128)
RVV_VEC_DOT_DISPATCH(ggml_vec_dot_q4_K_q8_K, 32, _256, 16, _128)
RVV_VEC_DOT_DISPATCH(ggml_vec_dot_q5_K_q8_K, 16, _128)
RVV_VEC_DOT_DISPATCH(ggml_vec_dot_q6_K_q8_K, 32, _256, 16, _128)

}

