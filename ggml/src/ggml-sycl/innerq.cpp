// SPDX-License-Identifier: MIT
//
// SYCL-side InnerQ device kernels. The host-side reference implementation
// of ggml_innerq_state_decide, ggml_innerq_state_k_squared_scale,
// ggml_innerq_compute_k_squared_profile, and ggml_innerq_state_recover
// lives in ggml/src/ggml-innerq.c (always compiled into ggml-base so
// src/llama-context.cpp can link without SYCL). This file only provides
// the SYCL device kernel for ggml_innerq_compute_k_squared_profile_sycl.

#include "ggml-innerq.h"

#include <sycl/sycl.hpp>

extern "C" void ggml_innerq_compute_k_squared_profile(
    const float * probe, int n_probe, int head_dim, float * out_scales);

// P3.2.3.2: SYCL device kernel for K^2 profile computation.
//
// Real SYCL parallel_for reduction (replaces the P3.2.3.2a "delegate to
// C reference" stub). The kernel computes the per-position sum-of-
// squares across the probe tokens on the SYCL device, then derives
// the per-position K^2 scale (1 / sqrt(1 + mean-square)). When no
// SYCL device is available (e.g. host CPU emulator without a GPU),
// the wrapper falls back to the C reference.
//
// API: ggml_innerq_compute_k_squared_profile_sycl(probe, n_probe,
// head_dim, out_scales) -- mirrors the C reference signature.

// P3.2.3.2: SYCL device kernel for K^2 profile computation.
//
// This is a parallel_for reduction that computes the per-position sum
// of squares across the probe tokens, then derives the per-position
// K^2 scale (1 / sqrt(1 + mean-square)). On any SYCL device
// (GPU or host CPU emulator), this produces the same result as the
// C reference within float tolerance.
//
// API mirrors the C reference:
//   ggml_innerq_compute_k_squared_profile_sycl
// takes (probe, n_probe, head_dim, out_scales) and fills out_scales.
// The host-callable wrapper below tries to acquire a SYCL device
// and dispatch the parallel_for; if no SYCL device is available
// (which is the case on the A770 when the AOT device kernel
// doesn't include the new kernel, or when the harness is run
// without a GPU), the function falls back to the C reference.
// The C reference is the binding correctness oracle; the SYCL
// kernel's only job is to match it within float tolerance.

extern "C" void ggml_innerq_compute_k_squared_profile_sycl(
    const float * probe, int n_probe, int head_dim, float * out_scales) {
    // Defensive: same as the C reference.
    for (int d = 0; d < head_dim; ++d) {
        out_scales[d] = 1.0f;
    }
    if (probe == nullptr || out_scales == nullptr || n_probe < 1) {
        return;
    }
    if (head_dim != 16 && head_dim != 32 && head_dim != 64 && head_dim != 128) {
        return;
    }

    // Try to acquire a SYCL device. If none is available (no GPU,
    // host emulator not enabled, etc.), fall back to the C reference
    // and return. The C reference is the binding correctness oracle.
    sycl::queue q;
    try {
        q = sycl::queue{sycl::default_selector{}};
    } catch (...) {
        // No SYCL device available; fall back to the C reference.
        ggml_innerq_compute_k_squared_profile(probe, n_probe, head_dim, out_scales);
        return;
    }

    // The parallel_for reduction computes per-position sum-of-squares
    // across the probe tokens. We use a separate accumulator per
    // position (one accumulator per head_dim slot). For head_dim=128
    // and n_probe up to ~256, the reduction is small enough to fit
    // in a single work-group's local memory.
    const int D = head_dim;
    const int N = n_probe;
    const size_t buf_probe_n = (size_t) N * (size_t) D;
    const size_t buf_out_n   = (size_t) D;

    sycl::buffer<float, 1> buf_probe{const_cast<float *>(probe), sycl::range<1>(buf_probe_n)};
    sycl::buffer<float, 1> buf_out{out_scales, sycl::range<1>(buf_out_n)};

    // Pass 1: per-position sum of squares.
    q.submit([&](sycl::handler & h) {
        sycl::accessor acc_probe{buf_probe, h, sycl::read_only};
        sycl::accessor acc_out{buf_out, h, sycl::write_only, sycl::no_init};
        h.parallel_for(sycl::range<1>(D), [=](sycl::id<1> d) {
            double sumsq = 0.0;
            const int d_idx = (int) d[0];
            for (int i = 0; i < N; ++i) {
                const double v = (double) acc_probe[(size_t) i * (size_t) D + (size_t) d_idx];
                sumsq += v * v;
            }
            acc_out[d_idx] = (float) (sumsq / (double) N);  // mean square
        });
    });
    // Pass 2: convert mean-square to 1 / sqrt(1 + mean-square).
    q.submit([&](sycl::handler & h) {
        sycl::accessor acc_out{buf_out, h, sycl::read_write};
        h.parallel_for(sycl::range<1>(D), [=](sycl::id<1> d) {
            const int d_idx = (int) d[0];
            const double ms = (double) acc_out[d_idx];
            acc_out[d_idx] = (float) (1.0 / sycl::sqrt(1.0 + ms));
        });
    });
    q.wait_and_throw();
    // The buffer's writeback is implicit (sycl::buffer created with
    // out_scales as the host pointer); the kernel's writes to acc_out
    // are propagated back to out_scales at q.wait_and_throw().
}

// P3.2.3.3: Static-turbo fallback + init-only retry policy placeholder.
//
// The P3.2 policy contract specifies:
//   - On InnerQ failure, fall back to STATIC turbo4 (NEVER f16).
//   - Recalibration: 1 retry on init-only anomalies, no retry on
//     mid-stream NaN.
// The full implementation lives in the SYCL backend's FA dispatch
// path; it is gated on the policy's `decide()` return value and
// inspects the per-request abort signal (P3.2.3.3 will wire it
// into ggml-sycl.cpp:4828). For now, P3.2.3.1 (C reference) and
// P3.2.3.2 (SYCL kernel) are the artifacts shipped to date.
