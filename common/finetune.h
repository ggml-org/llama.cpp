#pragma once

#include "ggml-opt.h"
#include "llama.h"

#include <string>

// Build an SFT dataset (tokens, next-token labels, assistant-only loss masks) from JSONL chat
// conversations rendered through the chat template (the model's own one when chat_template is empty).
// Returns nullptr when the template cannot be used or the data is malformed.
ggml_opt_dataset_t common_opt_sft_dataset_init(
        struct llama_context * ctx,
        const std::string    & json_content,
        const std::string    & chat_template = "");
