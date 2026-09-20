#ifndef GGML_SYCL_BASE_HPP
#define GGML_SYCL_BASE_HPP

/**
 * Module: base
 *
 * Description:
 * Provides zero-dependency, foundational primitives, core abstractions,
 * and low-level system interfaces. This module acts as the lowest layer
 * of the architecture and is consumed globally across all subsystems.
 *
 * Constraints:
 * - STRICTLY zero upstream dependencies (leaf module).
 * - High stability and backward compatibility required.
 */

#include <cstdio>

extern int g_ggml_sycl_debug;
extern int g_ggml_sycl_dev_debug;

#if defined(__clang__) && __has_builtin(__builtin_expect)
// Hint the optimizer to pipeline the more likely following instruction in branches
#    define LIKELY(expr)   __builtin_expect(expr, true)
#    define UNLIKELY(expr) __builtin_expect(expr, false)
#else
#    define LIKELY(expr)   (expr)
#    define UNLIKELY(expr) (expr)
#endif

// Always-inline marker for device-code helpers - replaces the SYCLomatic
// __dpct_inline__.
#if defined(_MSC_VER) && !defined(__clang__)
#    define GGML_SYCL_INLINE __forceinline
#else
#    define GGML_SYCL_INLINE inline __attribute__((always_inline))
#endif

// Variable alignment attribute - replaces the SYCLomatic __dpct_align__.
#define GGML_SYCL_ALIGN(n) __attribute__((aligned(n)))

#define GGML_SYCL_DEBUG(...)              \
    do {                                  \
        if (UNLIKELY(g_ggml_sycl_debug))  \
            fprintf(stderr, __VA_ARGS__); \
    } while (0)

#define GGML_SYCL_DEV_DEBUG(...)          \
    do {                                  \
        if (UNLIKELY(g_ggml_sycl_dev_debug))  \
            fprintf(stderr, __VA_ARGS__); \
    } while (0)

#endif  // GGML_SYCL_BASE_HPP
