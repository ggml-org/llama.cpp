// SPDX-License-Identifier: MIT
#pragma once
#include "maple_w2a8.hpp"
#include <array>
#include "expert_grouping.hpp"
#include "moe_dispatch_policy.hpp"
namespace maple_w2 {
enum class MoePath { a16, a8_separate, a8_gluquant };
struct MoeOptions {
    MoePath path=MoePath::a8_gluquant;
    uint32_t input_group=32,hidden_group=32;
    Options gate_kernel{},down_kernel{};
    float clamp=7.f;
    // Diagnostic only: intentionally recreate three host boundaries.
    // Default false: enqueue_moe returns before GPU completion.
    bool diagnostic_host_waits=false;
    // v0.4 sole optimization: stable expert-major job order, built every call.
    // No multi-token DPAS tile or GEMM; original dot arithmetic is retained.
    bool expert_grouping=false;
    // Appended fields preserve source-level aggregate initialization of v0.4.
    MoeSchedule schedule=MoeSchedule::inherit_v04;
    MoeAutoPolicy auto_policy{};
    bool overlap_grouping_quant=false; // fork/join; no overlap guarantee
    uint32_t gate_tokens_per_tile=1,down_tokens_per_tile=1;
    // Caller supplies fresh input_a8 AND completion dependencies. Not a cache.
    bool input_prequantized=false;
};
struct MoeProblem {
    Shape gate_shape; // {input width, expert hidden width, experts}; down={hidden,input,experts}
    uint32_t tokens=1,topk=8;
    Layout layout=Layout::s2tile8;
    const uint8_t *gate=nullptr,*up=nullptr,*down=nullptr;
    const float *x=nullptr,*routes=nullptr; // x [Q,K], normalized routes [Q,topk]
    const int32_t *ids=nullptr;
    float *y=nullptr; // [Q,K], before residual add
};
struct MoeWorkspace {
    float *gate=nullptr,*up=nullptr,*hidden=nullptr,*down=nullptr;
    float *gate_split=nullptr,*down_split=nullptr;
    A8Workspace input_a8{},hidden_a8{};
    int32_t *gate_status=nullptr,*down_status=nullptr; // each Q*topk; zeroed by caller before launch
    int32_t *hidden_invalid=nullptr; // Q*topk*H, overwritten by epilogue
    int32_t *output_invalid=nullptr; // Q*K, overwritten by weighted sum
    ExpertGroupingWorkspace grouping{};
    ExpertTokenTileView gate_tiles{},down_tiles{};
};
struct MoeStage {const char* label="";sycl::event event;uint32_t parents=0;};
struct MoeRun {
    std::array<MoeStage,20> stages{};size_t count=0;sycl::event done;
    MoeScheduleDecision dispatch{};
    bool fork_join_requested=false,queue_in_order=false;
    uint32_t gate_tile=1,down_tile=1;
    void add(const char* label,const sycl::event& e,uint32_t parents=0){if(count==stages.size())throw std::logic_error("event list full");stages[count++]={label,e,parents};done=e;}
};
// USM allocations must share this queue's context. No explicit device allocations or readback inside.
// Same workspace cannot be shared by concurrent in-flight runs without dependencies.
// Input/route/ID values are caller validated; device status must also be checked.
// On a submission exception, drain the queue before reusing/freeing this workspace.
// Never schedule a fallback against buffers that may still have in-flight writes.
MoeRun enqueue_moe(sycl::queue&,const MoeProblem&,const MoeOptions&,const MoeWorkspace&,
                   const std::vector<sycl::event>& dependencies={});
} // namespace maple_w2
