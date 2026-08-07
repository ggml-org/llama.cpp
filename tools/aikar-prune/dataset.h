#pragma once

#include "chat.h"
#include "llama.h"

#include <cstdint>
#include <string>
#include <vector>

enum class aikar_ppl_mask {
    ALL,
    ASSISTANT,
    REASONING,
    CONTENT,
};

struct aikar_dataset_record {
    std::vector<llama_token> tokens;
    std::vector<uint8_t> token_fields;
    int64_t line = 0;
};

struct aikar_dataset {
    std::vector<aikar_dataset_record> records;
    int64_t total_tokens = 0;
};

aikar_ppl_mask aikar_ppl_mask_parse(const std::string & value);
const char * aikar_ppl_mask_name(aikar_ppl_mask value);
bool aikar_token_is_evaluated(const aikar_dataset_record & record, size_t token_index, aikar_ppl_mask mask);

aikar_dataset aikar_dataset_load(
        const std::string & path,
        const llama_model * model,
        const common_chat_templates * templates);
