#pragma once

// --mmproj-evict-draft (EXPERIMENTAL)
//
// Keeps each mmproj modality encoder (vision, audio) weights resident in host RAM and streams
// them to the compute device on demand, only while encoding that modality, evicting the
// speculative-decoding draft model's unique weights to host RAM for that window, then restoring
// them.
//
// The mmproj side and the draft side are each swappable weight units:
//   - mmproj: one ggml_backend_buffer per modality encoder (clip_ctx::buf), backed by a persistent
//             host copy when weights_evict is set (mtmd_set_mmproj_modality_weights_gpu).
//   - draft : the tensors the draft gguf actually contains (tok_embd / output are shared with the
//             target via ctx_other and are NOT in the draft's buffers, so the draft buffers are
//             exactly the unique weights that are safe to evict; llama_model_weights_swap_*).

#include <mutex>

#include "mtmd.h"

struct llama_model;
struct llama_context;

namespace server {

// Depth-tracked, mutex-guarded controller for the mmproj/draft weight swap; the media-decode path
// uses it to enter/exit "media mode" per modality (mmproj on GPU, draft evicted). All depths 0 =
// "decode mode" (draft weights on GPU, every mmproj modality on host); any depth > 0 = "media
// mode". A single mmproj may hold several modality encoders (vision, audio), tracked
// independently so only the modalities in use are streamed. The draft is evicted once when the
// first modality enters and restored once when the last modality exits; nested encodes of the
// same modality are depth-tracked, so overlapping encodes from multiple slots serialize safely.
class mmproj_evict_draft_swap {
public:
    static const char * mod_name(enum mtmd_mmproj_modality mod) {
        return mod == MTMD_MMPROJ_MOD_AUDIO ? "audio" : "vision";
    }

    // Arm the controller. The caller must have already run llama_model_weights_swap_prepare
    // successfully (the swap is only armed in that case) and loaded the mmproj with weights_evict
    // (host-resident weights). ctx_dft is the draft context; it is drained (llama_synchronize)
    // before the first evict so an in-flight draft decode cannot read the freed device buffer.
    // Returns false only on null inputs.
    bool init(mtmd_context * mctx, llama_model * model_dft, llama_context * ctx_dft);

    bool active() const { return ready; }

    // Stream the given mmproj modality's weights to GPU and evict the draft's unique weights
    // (the draft is evicted once, when the first modality enters). Idempotent w.r.t. nested
    // encodes of the same modality.
    bool enter_mtmd(enum mtmd_mmproj_modality mod);
    // Move the given mmproj modality's weights back to host and restore the draft (the draft is
    // restored once, when the last modality exits).
    bool exit_mtmd(enum mtmd_mmproj_modality mod);

private:
    mtmd_context * mctx = nullptr;
    llama_model * model_dft = nullptr;
    llama_context * ctx_dft = nullptr;
    bool ready = false;
    int vision_depth = 0;
    int audio_depth  = 0;
    std::mutex mu;
};

}
