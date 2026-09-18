// SPDX-License-Identifier: MIT
#pragma once
#include "expert_grouping_reference.hpp"
#include "expert_token_tiles.hpp"
#include <sycl/sycl.hpp>
namespace maple_w2 {
struct ExpertGroupingWorkspace {
    int32_t* counts=nullptr;        // experts+1; includes invalid-ID bucket
    int32_t* offsets=nullptr;       // experts+2; last entry equals jobs
    int32_t* sorted_to_job=nullptr; // jobs; permutation of original job indices
    size_t job_capacity=0;
    uint32_t expert_capacity=0;
};
struct ExpertGroupingRun {sycl::event count,prefix,scatter;};
// Three dependent GPU kernels, rebuilt from this call's IDs on EVERY enqueue.
// No allocation, ID readback, cached old routing, or host wait in this function.
// Scratch must be in the queue's context and exclusively owned until completion.
ExpertGroupingRun enqueue_expert_grouping(sycl::queue&,const int32_t* ids,size_t jobs,uint32_t experts,
    ExpertGroupingWorkspace,const std::vector<sycl::event>& dependencies={});
// Build bounded descriptors from fresh count/offset/order data. No host readback.
sycl::event enqueue_expert_token_tiles(sycl::queue&,size_t jobs,uint32_t experts,
    ExpertGroupingWorkspace,ExpertTokenTileView,const std::vector<sycl::event>& dependencies={});
} // namespace maple_w2
