#include "mmproj-evict-draft.h"

#include "log.h"

#include "mtmd.h"

#include "llama.h"
#include "../src/llama-ext.h"

#include <chrono>

namespace server {

bool mmproj_evict_draft_swap::init(mtmd_context * mctx_, llama_model * model_dft_, llama_context * ctx_dft_) {
    // the caller has already verified the draft has swappable GPU weights (llama_model_weights_swap_prepare)
    if (mctx_ == nullptr || model_dft_ == nullptr) {
        return false;
    }
    mctx      = mctx_;
    model_dft = model_dft_;
    ctx_dft   = ctx_dft_;
    ready     = true;
    LOG_INF("mmproj_evict_draft : swap armed (draft weight buffers recorded for evict/restore)\n");
    return true;
}

bool mmproj_evict_draft_swap::enter_mtmd(enum mtmd_mmproj_modality mod) {
    std::lock_guard<std::mutex> lock(mu);
    if (!ready) {
        return true; // swap disabled; nothing to do
    }
    const bool first_overall = (vision_depth + audio_depth) == 0;
    int & mod_depth = (mod == MTMD_MMPROJ_MOD_AUDIO) ? audio_depth : vision_depth;
    const bool first_mod   = (mod_depth == 0);
    ++mod_depth;

    auto t0 = std::chrono::steady_clock::now();
    // Evict the draft's unique weights (free its VRAM) before streaming the modality's weights,
    // so the new GPU buffer can reuse that space with no transient VRAM growth. Done once per
    // window, when the first modality enters.
    if (first_overall) {
        // wait for the last draft decode to finish before freeing its device buffer; the D2H copies
        // in weight_swap_evict are host-blocking but do not order against the compute stream
        if (ctx_dft) {
            llama_synchronize(ctx_dft);
        }
        if (!llama_model_weights_swap_evict(model_dft)) {
            llama_model_weights_swap_restore(model_dft); // roll back the partially evicted buffers
            --mod_depth;
            LOG_ERR("mmproj_evict_draft : failed to evict draft before entering %s mode\n", mod_name(mod));
            return false;
        }
    }
    // Stream the requested modality's weights to GPU (once per window for that modality; a no-op
    // if the encoder is absent from the mmproj).
    if (first_mod && !mtmd_set_mmproj_modality_weights_gpu(mctx, mod, true)) {
        if (first_overall) {
            llama_model_weights_swap_restore(model_dft);
        }
        --mod_depth;
        LOG_ERR("mmproj_evict_draft : failed to stream %s mmproj weights to GPU; rolling back\n", mod_name(mod));
        return false;
    }
    if (first_mod) {
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        LOG_INF("mmproj_evict_draft : enter %s mode: %s in %.1f ms\n", mod_name(mod),
                first_overall ? "draft evict + weights -> GPU" : "weights -> GPU", ms);
    }
    return true;
}

bool mmproj_evict_draft_swap::exit_mtmd(enum mtmd_mmproj_modality mod) {
    std::lock_guard<std::mutex> lock(mu);
    if (!ready) {
        return true;
    }
    int & mod_depth = (mod == MTMD_MMPROJ_MOD_AUDIO) ? audio_depth : vision_depth;
    if (mod_depth <= 0) {
        LOG_WRN("mmproj_evict_draft : unbalanced exit_mtmd(%s), nothing to do\n", mod_name(mod));
        return true;
    }
    --mod_depth;
    const bool last_mod     = (mod_depth == 0);
    const bool last_overall = (vision_depth + audio_depth) == 0;

    auto t0 = std::chrono::steady_clock::now();
    // Move the modality's weights back to host (free its VRAM) once the last encode of that
    // modality exits, then restore the draft when the last encode of any modality exits.
    bool ok = true;
    if (last_mod) {
        ok = mtmd_set_mmproj_modality_weights_gpu(mctx, mod, false) && ok;
    }
    if (last_overall) {
        ok = llama_model_weights_swap_restore(model_dft) && ok;
    }
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (!ok) {
        LOG_ERR("mmproj_evict_draft : failed to exit %s mode\n", mod_name(mod));
    } else if (last_mod) {
        LOG_INF("mmproj_evict_draft : exit %s mode: weights -> host%s in %.1f ms\n", mod_name(mod),
                last_overall ? " + draft restore" : "", ms);
    }
    return ok;
}

} // namespace server
