#pragma once

#include "chat.h"
#include "llama.h"

#include <cstdint>
#include <string>
#include <vector>

enum class llama_prune_ppl_mask {
    ALL,
    ASSISTANT,
    REASONING,
    CONTENT,
};

struct llama_prune_dataset_record {
    std::vector<llama_token> tokens;
    std::vector<uint8_t> token_fields;
    int64_t line = 0;
};

struct llama_prune_dataset {
    std::vector<llama_prune_dataset_record> records;
    int64_t total_tokens = 0;
};

llama_prune_ppl_mask llama_prune_ppl_mask_parse(const std::string & value);
const char * llama_prune_ppl_mask_name(llama_prune_ppl_mask value);
bool llama_prune_token_is_evaluated(const llama_prune_dataset_record & record, size_t token_index, llama_prune_ppl_mask mask);

llama_prune_dataset llama_prune_dataset_load(
        const std::string & path,
        const llama_model * model,
        const common_chat_templates * templates,
        int32_t n_threads = 0);
