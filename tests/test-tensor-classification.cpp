//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

#include "../ggml/src/ggml-sycl/tensor-types.hpp"

#include <cassert>
#include <iostream>
#include <string>

using namespace ggml_sycl;

void test_tensor_classification() {
    // Embeddings (priority 0)
    assert(classify_tensor("token_embd.weight") == tensor_class::EMBEDDING);

    // Output head (priority 0)
    assert(classify_tensor("lm_head.weight") == tensor_class::OUTPUT);
    assert(classify_tensor("output.weight") == tensor_class::OUTPUT);

    // Attention (priority 1)
    assert(classify_tensor("blk.0.attn_q.weight") == tensor_class::ATTENTION);
    assert(classify_tensor("blk.5.attn_k.weight") == tensor_class::ATTENTION);
    assert(classify_tensor("blk.10.attn_v.weight") == tensor_class::ATTENTION);
    assert(classify_tensor("blk.15.attn_output.weight") == tensor_class::ATTENTION);

    // Router (priority 2)
    assert(classify_tensor("blk.0.ffn_gate_inp.weight") == tensor_class::ROUTER);

    // Dense FFN (priority 2)
    assert(classify_tensor("blk.0.ffn_up.weight") == tensor_class::FFN);
    assert(classify_tensor("blk.5.ffn_down.weight") == tensor_class::FFN);
    assert(classify_tensor("blk.10.ffn_gate.weight") == tensor_class::FFN);

    // Experts (priority 3)
    assert(classify_tensor("blk.0.ffn_down_exps.weight") == tensor_class::EXPERT);
    assert(classify_tensor("blk.5.ffn_gate_exps.weight") == tensor_class::EXPERT);
    assert(classify_tensor("blk.10.ffn_up_exps.weight") == tensor_class::EXPERT);

    // Norms (priority 4)
    assert(classify_tensor("blk.0.attn_norm.weight") == tensor_class::NORM);
    assert(classify_tensor("blk.0.ffn_norm.weight") == tensor_class::NORM);
    assert(classify_tensor("output_norm.weight") == tensor_class::NORM);

    // Other
    assert(classify_tensor("rope_freqs.weight") == tensor_class::OTHER);

    std::cout << "test_tensor_classification: PASSED\n";
}

void test_priority_from_type() {
    assert(priority_from_type(tensor_class::EMBEDDING) == 0);
    assert(priority_from_type(tensor_class::OUTPUT) == 0);
    assert(priority_from_type(tensor_class::ATTENTION) == 1);
    assert(priority_from_type(tensor_class::FFN) == 2);
    assert(priority_from_type(tensor_class::ROUTER) == 2);
    assert(priority_from_type(tensor_class::EXPERT) == 3);
    assert(priority_from_type(tensor_class::NORM) == 4);
    assert(priority_from_type(tensor_class::OTHER) == 5);

    std::cout << "test_priority_from_type: PASSED\n";
}

void test_extract_layer_id() {
    assert(extract_layer_id("blk.0.attn_q.weight") == 0);
    assert(extract_layer_id("blk.15.ffn_up.weight") == 15);
    assert(extract_layer_id("blk.127.attn_k.weight") == 127);
    assert(extract_layer_id("token_embd.weight") == -1);
    assert(extract_layer_id("output_norm.weight") == -1);

    std::cout << "test_extract_layer_id: PASSED\n";
}

void test_extract_expert_id() {
    // Non-expert tensors return -1
    assert(extract_expert_id("blk.0.attn_q.weight") == -1);
    assert(extract_expert_id("blk.0.ffn_up.weight") == -1);

    // Expert tensors - expert ID is inferred from context, not tensor name
    // The tensor name just indicates it's an expert tensor
    // Actual expert ID comes from the tensor's position in the expert array
    assert(extract_expert_id("blk.0.ffn_down_exps.weight") == -1);  // Array of all experts

    std::cout << "test_extract_expert_id: PASSED\n";
}

void test_make_tensor_info() {
    auto info = make_tensor_info("blk.5.attn_q.weight", 1024 * 1024);
    assert(info.name == "blk.5.attn_q.weight");
    assert(info.size == 1024 * 1024);
    assert(info.layer_id == 5);
    assert(info.type == tensor_class::ATTENTION);
    assert(info.static_priority == 1);

    std::cout << "test_make_tensor_info: PASSED\n";
}

int main() {
    try {
        test_tensor_classification();
        test_priority_from_type();
        test_extract_layer_id();
        test_extract_expert_id();
        test_make_tensor_info();
        std::cout << "\nAll tensor classification tests PASSED!\n";
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "Test FAILED: " << e.what() << "\n";
        return 1;
    }
}
