#pragma once

#include <cstdint>

namespace ggml::hrx {

struct LlmMoeDispatchProfile {
    const char * name               = "";
    int64_t      hidden_size        = 0;
    int64_t      expert_hidden_size = 0;
    int64_t      expert_count       = 0;
    int64_t      route_count        = 0;
    int64_t      max_token_count    = 0;
    float        rms_norm_epsilon   = 0.0f;
};

constexpr LlmMoeDispatchProfile kLlmMoeQwen30BDispatchProfile = {
    "qwen30b", 2048, 768, 128, 8, 2048, 0.000001f,
};

static constexpr const LlmMoeDispatchProfile & kActiveLlmMoeDispatchProfile = kLlmMoeQwen30BDispatchProfile;
static constexpr const LlmMoeDispatchProfile & kQwen30BMoeDispatchProfile   = kLlmMoeQwen30BDispatchProfile;

constexpr bool is_llm_supported_query_length(const LlmMoeDispatchProfile & profile, int64_t query_length) {
    return query_length >= 1 && query_length <= profile.max_token_count;
}

constexpr bool is_llm_decode_query_length(int64_t query_length) {
    return query_length == 1;
}

constexpr bool is_llm_prefill_query_length(const LlmMoeDispatchProfile & profile, int64_t query_length) {
    return query_length > 1 && query_length <= profile.max_token_count;
}

}  // namespace ggml::hrx
