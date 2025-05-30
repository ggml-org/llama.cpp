#pragma once

#include "llama.h" // This now provides the primary definitions

// #include <array> // llama.h includes this

// Internal constant if not defined in the public API
#define LLAMA_MAX_EXPERTS 256  // DeepSeekV3

// Internal helper structs if they are not part of the public API
// and are used by files including src/llama-hparams.h
// If these are actually part of the public llama_hparams, they should be in include/llama.h
// For now, assuming they might be used by other src files that include this.
struct llama_hparams_posnet {
    uint32_t n_embd;
    uint32_t n_layer;
};

struct llama_hparams_convnext {
    uint32_t n_embd;
    uint32_t n_layer;
};

// All other definitions previously in this file (LLAMA_MAX_LAYERS,
// enum llama_expert_gating_func_type, enum llama_swa_type,
// struct llama_hparams, and the static_assert) are removed
// to defer to the definitions in "llama.h".
