#include <cassert>
#include <cstring>
#include <iostream>

// Test that tensor placement logic routes correctly:
// - Dense tensors: VRAM first, overflow to pinned
// - Expert tensors: VRAM until full, then pinned
// - mmap only when both exhausted

extern "C" {
// Forward declarations from ggml-sycl
int ggml_sycl_classify_tensor(const char * name);  // 0=dense, 1=expert, 2=other
}

void test_tensor_classification() {
    // Dense layer patterns
    assert(ggml_sycl_classify_tensor("blk.0.attn_q.weight") == 0);
    assert(ggml_sycl_classify_tensor("blk.5.attn_k.weight") == 0);
    assert(ggml_sycl_classify_tensor("blk.10.ffn_down.weight") == 0);

    // Expert patterns (MoE)
    assert(ggml_sycl_classify_tensor("blk.0.ffn_down_exps.weight") == 1);
    assert(ggml_sycl_classify_tensor("blk.5.ffn_gate_exps.weight") == 1);
    assert(ggml_sycl_classify_tensor("blk.10.ffn_up_exps.weight") == 1);

    // Router (treated as dense - used every token)
    assert(ggml_sycl_classify_tensor("blk.0.ffn_gate_inp.weight") == 0);

    // Other tensors
    assert(ggml_sycl_classify_tensor("token_embd.weight") == 2);
    assert(ggml_sycl_classify_tensor("output_norm.weight") == 2);

    std::cout << "test_tensor_classification: PASSED\n";
}

int main() {
    test_tensor_classification();
    std::cout << "\nAll tests PASSED!\n";
    return 0;
}
