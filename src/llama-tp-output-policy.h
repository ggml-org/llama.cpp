#pragma once

enum class llama_tp_output_split_mode {
    mirrored,
    hidden,
    vocabulary,
};

struct llama_tp_output_policy_input {
    bool sharding_enabled;
    bool sharding_blocked;
    bool tensor_parallel;
    bool supported_arch;
    bool vocabulary_requested;
    bool primary_head;
};

constexpr llama_tp_output_split_mode llama_tp_output_policy_select(
        const llama_tp_output_policy_input & input) {
    if (!input.sharding_enabled || input.sharding_blocked ||
            !input.tensor_parallel || !input.supported_arch) {
        return llama_tp_output_split_mode::mirrored;
    }
    return input.vocabulary_requested && input.primary_head
        ? llama_tp_output_split_mode::vocabulary
        : llama_tp_output_split_mode::hidden;
}
