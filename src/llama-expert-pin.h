#pragma once

#include <cstdint>

struct llama_expert_heatmap;
struct llama_model;

// mmap page hints for the expert tier. periodic madvise pass: drop the hottest
// GPU experts' pages, warm the ones most likely needed next.
namespace llama_expert_pin {

    // dials, all env-overridable with these defaults
    struct config {
        int   period        = 32;    // tokens between passes
        int   start_tokens  = 128;   // first pass at this many tokens
        float dontneed_gpu  = 0.20f; // top fraction of GPU experts to drop
        float willneed_gpu  = 0.20f; // bottom fraction of GPU experts to warm
        float willneed_cold = 0.70f; // top fraction of cold experts to warm
    };

    const config & get_config();

    // resolved pin fraction from --expert-pin (percent, -1 = unset)
    void set_pct(int pct);
    int  get_pct();

    // true when pinning is enabled (LLAMA_EXPERT_PIN set or pct > 0)
    bool active();

    // periodic madvise pass. is_gpu_resident marks store residents,
    // nullptr = standalone (all cold)
    void maybe_run(const llama_model * model,
                   const llama_expert_heatmap * heatmap,
                   bool (*is_gpu_resident)(void * ud, int il, int e),
                   void * ud);

}
