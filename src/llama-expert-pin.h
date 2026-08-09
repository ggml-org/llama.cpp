#pragma once

#include <cstdint>

struct llama_expert_heatmap;
struct llama_model;

// mmap page hints for the expert tier. replaces the unconditional MADV_DONTNEED
// on GPU-bound experts: a periodic pass drops the hottest GPU experts' pages
// and warms the ones most likely to be needed next.
namespace llama_expert_pin {

    // dials, all env-overridable with these defaults
    struct config {
        int   period        = 32;    // tokens between passes
        int   start_tokens  = 128;   // first pass at this many tokens
        float dontneed_gpu  = 0.20f; // top fraction of GPU experts to drop
        float willneed_gpu  = 0.20f; // bottom fraction of GPU experts to warm
        float willneed_cold = 0.40f; // top fraction of cold experts to warm
    };

    const config & get_config();

    // true when pinning is enabled (LLAMA_EXPERT_PIN set)
    bool active();

    // periodic madvise pass. heatmap feeds the rankings; is_gpu_resident
    // (ud, il, e) marks store residents, nullptr = standalone (all cold).
    void maybe_run(const llama_model * model,
                   const llama_expert_heatmap * heatmap,
                   bool (*is_gpu_resident)(void * ud, int il, int e),
                   void * ud);

}
