// SPDX-License-Identifier: MIT
#pragma once
#include "tq2_reference.hpp"
#include <sycl/sycl.hpp>

namespace maple_w2 {
enum class Layout { native_tq2, tile8, s2tile8 };
enum class ActivationType { f32, f16 };
// All pointers must be device/shared USM or compatible external-memory imports
// in q.get_context(). DO NOT pass raw Vulkan addresses or CPU allocations.
// Weights must first pass validate_tq2(). Views are contiguous in this pilot.
struct DeviceProblem {
    Shape shape;
    uint32_t tokens=1, topk=8;
    bool per_selection=false; // gate/up=false, down=true
    Layout layout=Layout::native_tq2;
    const uint8_t *w0=nullptr;
    const uint8_t *w1=nullptr; // optional fused second projection, same shape/layout
    const void *x=nullptr;    // F32 -> F16 in registers, or existing F16 input
    const int32_t *ids=nullptr; // [tokens, topk]
    float *y0=nullptr, *y1=nullptr; // [tokens, topk, M], NOT router-reduced
    int32_t *status=nullptr; // at least tokens*topk entries, initialized to zero
    ActivationType activation_type=ActivationType::f32;
};
struct Options { uint32_t split_k=1, local_size=4; };
// scratch size in floats. Reuse across sequential calls, not in-flight calls.
size_t scratch_floats(const DeviceProblem&, Options);
struct Run {
    sycl::event main, done;
    bool has_reduce=false;
    double main_us() const;
    double reduce_us() const;
};
Run enqueue(sycl::queue&, const DeviceProblem&, Options, float *scratch,
            const std::vector<sycl::event>& dependencies={});
// Experimental F32 non-XMX comparator, not an implementation of the exact
// Vulkan shader. Native TQ2 input only. Useful for within-queue diagnostics.
Run enqueue_fma_reference(sycl::queue&, const DeviceProblem&,
                          const std::vector<sycl::event>& dependencies={});
} // namespace maple_w2
