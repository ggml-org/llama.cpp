// SPDX-License-Identifier: MIT
#pragma once
#include "maple_w2a16.hpp"
#include "w2a8_reference.hpp"
#include "expert_token_tiles.hpp"
namespace maple_w2 {
// prequantized: a producer has already written q/scales/invalid for this call.
// The producer completion event MUST be in dependencies on an out-of-order queue.
// This is NOT stale-activation reuse across tokens.
enum class A8Mode { staged, fused, prequantized };
struct A8Options {
    uint32_t group=32; A8Mode mode=A8Mode::staged; Options kernel{};
    // Optional GPU-generated permutation. Only execution order changes: all
    // activation/output/status addressing still uses ORIGINAL token/selection.
    const int32_t* job_order=nullptr;
    // Optional grouped multi-assignment register reuse. G32/s2tile8/prequantized.
    // Each token still uses the original RepeatCount=1 DPAS arithmetic.
    ExpertTokenTileView token_tiles{};
};
// Persistent scratch, context-compatible USM. No allocations in enqueue_a8.
// q/scales are required for staged/prequantized; fused writes them only with capture=true.
// invalid has rows*(K/group) elements. Staged overwrites it; prequantized requires
// that the dependency producer has written it for this invocation.
struct A8Workspace {
    int8_t* q=nullptr;
    float* scales=nullptr;
    int32_t* invalid=nullptr;
    bool capture=false;
};
struct A8Run {
    sycl::event quant, main, done;
    bool has_quant=false, has_reduce=false;
    double quant_us() const;
    double main_us() const;
    double reduce_us() const;
    sycl::event first_event() const { return has_quant?quant:main; }
};
A8Run enqueue_a8(sycl::queue&,const DeviceProblem&,A8Options,A8Workspace,float* split_scratch,
                const std::vector<sycl::event>& dependencies={});
// Expose the UNCHANGED v0.4 staged quantizer as a producer. Useful for fork/join
// and stage isolation. Does not inspect IDs or weights; dependencies own x lifetime.
sycl::event enqueue_a8_quant(sycl::queue&,const void* x,ActivationType,uint32_t rows,
    uint32_t k,uint32_t group,A8Workspace,const std::vector<sycl::event>& dependencies={});
// Internal opt-in grouped reuse dispatch. Public enqueue_a8 validates its views.
A8Run enqueue_a8_token_tiles(sycl::queue&,const DeviceProblem&,A8Options,A8Workspace,
    float* split_scratch,const std::vector<sycl::event>& dependencies);
// Exact signed-INT2/INT8 ISA probe. Throws on wrong results; no emulated fallback.
void run_s2s8_probe(sycl::queue&);
} // namespace maple_w2
