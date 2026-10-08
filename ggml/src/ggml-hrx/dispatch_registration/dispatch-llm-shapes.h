#pragma once

#include "dispatch-llm-profiles.h"

#include <cstdint>

namespace ggml::hrx {

constexpr bool is_llm_prefill_512_query_length(const LlmMoeDispatchProfile & profile, int64_t query_length) {
    return is_llm_prefill_query_length(profile, query_length) && query_length == 512;
}

constexpr bool is_qwen_supported_query_length(int64_t query_length) {
    return is_llm_supported_query_length(kQwen30BMoeDispatchProfile, query_length);
}

constexpr bool is_qwen_decode_query_length(int64_t query_length) {
    return is_llm_decode_query_length(query_length);
}

constexpr bool is_qwen_prefill_query_length(int64_t query_length) {
    return is_llm_prefill_query_length(kQwen30BMoeDispatchProfile, query_length);
}

constexpr bool is_qwen_prefill_512_query_length(int64_t query_length) {
    return is_llm_prefill_512_query_length(kQwen30BMoeDispatchProfile, query_length);
}

}  // namespace ggml::hrx
