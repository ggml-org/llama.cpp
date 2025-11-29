// Unit tests for new GGML Flash Attention extension functions
//
// Tests ggml_flash_attn_ext_set_seq_ids() and ggml_flash_attn_ext_set_paged()
// which enable multi-sequence batching and paged attention for flash attention.
//
// These functions traverse through view/reshape operations to find the actual
// flash attention tensor and set additional source tensors for:
// - Sequence IDs: src[5] = q_seq_ids, src[6] = kv_seq_ids
// - Paged attention: src[7] = block_table, src[8] = seq_lens

#include <cstdio>
#include <cassert>
#include <cstring>
#include "ggml.h"

// =============================================================================
// Test Framework
// =============================================================================

#define TEST(name) static void test_##name()
#define RUN_TEST(name) do { \
    printf("Running test: %s... ", #name); \
    test_##name(); \
    printf("PASSED\n"); \
} while(0)

// Helper to pad up to GGML_KQ_MASK_PAD (64)
static inline int pad_to_mask(int n) {
    return ((n + GGML_KQ_MASK_PAD - 1) / GGML_KQ_MASK_PAD) * GGML_KQ_MASK_PAD;
}

// Helper to create a test context
static struct ggml_context * create_test_ctx(size_t mem_size = 256 * 1024) {
    struct ggml_init_params params = {
        .mem_size   = mem_size,
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };
    return ggml_init(params);
}

// =============================================================================
// Tests for ggml_flash_attn_ext_set_seq_ids
// =============================================================================

TEST(set_seq_ids_null_inputs) {
    // When q_seq_ids or kv_seq_ids is NULL, function should return early
    struct ggml_context * ctx = create_test_ctx();

    struct ggml_tensor * dummy = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 128);

    // Should not crash with NULL inputs
    ggml_flash_attn_ext_set_seq_ids(dummy, nullptr, nullptr);
    ggml_flash_attn_ext_set_seq_ids(nullptr, dummy, dummy);

    ggml_free(ctx);
}

TEST(set_seq_ids_direct_fattn) {
    // Test setting seq_ids directly on a flash attention tensor
    struct ggml_context * ctx = create_test_ctx();

    const int n_tokens = 16;
    const int n_kv = 64;
    const int n_head = 8;
    const int d_head = 64;

    // Create Q, K, V tensors for flash attention
    const int n_tokens_pad = pad_to_mask(n_tokens);

    struct ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d_head, n_tokens, n_head, 1);
    struct ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d_head, n_kv, n_head, 1);
    struct ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d_head, n_kv, n_head, 1);
    struct ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv, n_tokens_pad);

    // Create flash attention tensor
    struct ggml_tensor * fattn = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f, 0.0f, 0.0f);

    // Create sequence ID tensors
    struct ggml_tensor * q_seq_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    struct ggml_tensor * kv_seq_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_kv);

    // Set seq_ids
    ggml_flash_attn_ext_set_seq_ids(fattn, q_seq_ids, kv_seq_ids);

    // Verify they were set
    assert(fattn->src[5] == q_seq_ids);
    assert(fattn->src[6] == kv_seq_ids);

    ggml_free(ctx);
}

TEST(set_seq_ids_through_view) {
    // Test setting seq_ids on a flash attention tensor wrapped in a VIEW
    struct ggml_context * ctx = create_test_ctx();

    const int n_tokens = 16;
    const int n_kv = 64;
    const int n_head = 8;
    const int d_head = 64;

    const int n_tokens_pad = pad_to_mask(n_tokens);

    struct ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d_head, n_tokens, n_head, 1);
    struct ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d_head, n_kv, n_head, 1);
    struct ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d_head, n_kv, n_head, 1);
    struct ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv, n_tokens_pad);

    struct ggml_tensor * fattn = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f, 0.0f, 0.0f);

    // Wrap in a view
    struct ggml_tensor * view = ggml_view_1d(ctx, fattn, ggml_nelements(fattn), 0);

    struct ggml_tensor * q_seq_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    struct ggml_tensor * kv_seq_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_kv);

    // Set seq_ids through the view
    ggml_flash_attn_ext_set_seq_ids(view, q_seq_ids, kv_seq_ids);

    // Verify they were set on the underlying flash attention tensor
    assert(fattn->src[5] == q_seq_ids);
    assert(fattn->src[6] == kv_seq_ids);

    ggml_free(ctx);
}

TEST(set_seq_ids_through_reshape) {
    // Test setting seq_ids on a flash attention tensor wrapped in RESHAPE
    struct ggml_context * ctx = create_test_ctx();

    const int n_tokens = 16;
    const int n_kv = 64;
    const int n_head = 8;
    const int d_head = 64;

    struct ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d_head, n_tokens, n_head, 1);
    struct ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d_head, n_kv, n_head, 1);
    struct ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d_head, n_kv, n_head, 1);
    const int n_tokens_pad = pad_to_mask(n_tokens);
    struct ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv, n_tokens_pad);

    struct ggml_tensor * fattn = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f, 0.0f, 0.0f);

    // Wrap in reshape (2D)
    struct ggml_tensor * reshaped = ggml_reshape_2d(ctx, fattn, d_head, n_tokens * n_head);

    struct ggml_tensor * q_seq_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    struct ggml_tensor * kv_seq_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_kv);

    // Set seq_ids through the reshape
    ggml_flash_attn_ext_set_seq_ids(reshaped, q_seq_ids, kv_seq_ids);

    // Verify they were set on the underlying flash attention tensor
    assert(fattn->src[5] == q_seq_ids);
    assert(fattn->src[6] == kv_seq_ids);

    ggml_free(ctx);
}

TEST(set_seq_ids_through_cont) {
    // Test setting seq_ids on a flash attention tensor wrapped in CONT
    struct ggml_context * ctx = create_test_ctx();

    const int n_tokens = 16;
    const int n_kv = 64;
    const int n_head = 8;
    const int d_head = 64;

    struct ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d_head, n_tokens, n_head, 1);
    struct ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d_head, n_kv, n_head, 1);
    struct ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d_head, n_kv, n_head, 1);
    const int n_tokens_pad = pad_to_mask(n_tokens);
    struct ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv, n_tokens_pad);

    struct ggml_tensor * fattn = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f, 0.0f, 0.0f);

    // Wrap in cont (makes tensor contiguous)
    struct ggml_tensor * cont = ggml_cont(ctx, fattn);

    struct ggml_tensor * q_seq_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    struct ggml_tensor * kv_seq_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_kv);

    // Set seq_ids through cont
    ggml_flash_attn_ext_set_seq_ids(cont, q_seq_ids, kv_seq_ids);

    // Verify they were set on the underlying flash attention tensor
    assert(fattn->src[5] == q_seq_ids);
    assert(fattn->src[6] == kv_seq_ids);

    ggml_free(ctx);
}

TEST(set_seq_ids_through_permute) {
    // Test setting seq_ids on a flash attention tensor wrapped in PERMUTE
    struct ggml_context * ctx = create_test_ctx();

    const int n_tokens = 16;
    const int n_kv = 64;
    const int n_head = 8;
    const int d_head = 64;

    struct ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d_head, n_tokens, n_head, 1);
    struct ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d_head, n_kv, n_head, 1);
    struct ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d_head, n_kv, n_head, 1);
    const int n_tokens_pad = pad_to_mask(n_tokens);
    struct ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv, n_tokens_pad);

    struct ggml_tensor * fattn = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f, 0.0f, 0.0f);

    // Wrap in permute
    struct ggml_tensor * permuted = ggml_permute(ctx, fattn, 0, 2, 1, 3);

    struct ggml_tensor * q_seq_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    struct ggml_tensor * kv_seq_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_kv);

    // Set seq_ids through permute
    ggml_flash_attn_ext_set_seq_ids(permuted, q_seq_ids, kv_seq_ids);

    // Verify they were set on the underlying flash attention tensor
    assert(fattn->src[5] == q_seq_ids);
    assert(fattn->src[6] == kv_seq_ids);

    ggml_free(ctx);
}

TEST(set_seq_ids_through_transpose) {
    // Test setting seq_ids on a flash attention tensor wrapped in TRANSPOSE
    struct ggml_context * ctx = create_test_ctx();

    const int n_tokens = 16;
    const int n_kv = 64;
    const int n_head = 8;
    const int d_head = 64;

    struct ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d_head, n_tokens, n_head, 1);
    struct ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d_head, n_kv, n_head, 1);
    struct ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d_head, n_kv, n_head, 1);
    const int n_tokens_pad = pad_to_mask(n_tokens);
    struct ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv, n_tokens_pad);

    struct ggml_tensor * fattn = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f, 0.0f, 0.0f);

    // Wrap in transpose
    struct ggml_tensor * transposed = ggml_transpose(ctx, fattn);

    struct ggml_tensor * q_seq_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    struct ggml_tensor * kv_seq_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_kv);

    // Set seq_ids through transpose
    ggml_flash_attn_ext_set_seq_ids(transposed, q_seq_ids, kv_seq_ids);

    // Verify they were set on the underlying flash attention tensor
    assert(fattn->src[5] == q_seq_ids);
    assert(fattn->src[6] == kv_seq_ids);

    ggml_free(ctx);
}

TEST(set_seq_ids_chain) {
    // Test setting seq_ids through a chain of view/reshape operations
    struct ggml_context * ctx = create_test_ctx();

    const int n_tokens = 16;
    const int n_kv = 64;
    const int n_head = 8;
    const int d_head = 64;

    struct ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d_head, n_tokens, n_head, 1);
    struct ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d_head, n_kv, n_head, 1);
    struct ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d_head, n_kv, n_head, 1);
    const int n_tokens_pad = pad_to_mask(n_tokens);
    struct ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv, n_tokens_pad);

    struct ggml_tensor * fattn = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f, 0.0f, 0.0f);

    // Chain of operations: fattn -> cont -> reshape -> view
    struct ggml_tensor * cont = ggml_cont(ctx, fattn);
    struct ggml_tensor * reshaped = ggml_reshape_2d(ctx, cont, d_head, n_tokens * n_head);
    struct ggml_tensor * view = ggml_view_1d(ctx, reshaped, d_head * n_tokens * n_head, 0);

    struct ggml_tensor * q_seq_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    struct ggml_tensor * kv_seq_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_kv);

    // Set seq_ids through the chain
    ggml_flash_attn_ext_set_seq_ids(view, q_seq_ids, kv_seq_ids);

    // Verify they were set on the underlying flash attention tensor
    assert(fattn->src[5] == q_seq_ids);
    assert(fattn->src[6] == kv_seq_ids);

    ggml_free(ctx);
}

TEST(set_seq_ids_no_fattn_tensor) {
    // Test that function doesn't crash when no flash attention tensor is found
    struct ggml_context * ctx = create_test_ctx();

    // Create a non-flash-attention tensor
    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
    struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
    struct ggml_tensor * matmul = ggml_mul_mat(ctx, a, b);

    struct ggml_tensor * q_seq_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 16);
    struct ggml_tensor * kv_seq_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 64);

    // Should silently return without setting anything (no crash)
    ggml_flash_attn_ext_set_seq_ids(matmul, q_seq_ids, kv_seq_ids);

    ggml_free(ctx);
}

// =============================================================================
// Tests for ggml_flash_attn_ext_set_paged
// =============================================================================

TEST(set_paged_null_inputs) {
    // When block_table or seq_lens is NULL, function should return early
    struct ggml_context * ctx = create_test_ctx();

    struct ggml_tensor * dummy = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 128);

    // Should not crash with NULL inputs
    ggml_flash_attn_ext_set_paged(dummy, nullptr, nullptr);
    ggml_flash_attn_ext_set_paged(nullptr, dummy, dummy);

    ggml_free(ctx);
}

TEST(set_paged_direct_fattn) {
    // Test setting paged attention tensors directly on a flash attention tensor
    struct ggml_context * ctx = create_test_ctx();

    const int n_tokens = 16;
    const int n_kv = 64;
    const int n_head = 8;
    const int d_head = 64;
    const int batch_size = 4;
    const int max_blocks_per_seq = 8;

    struct ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d_head, n_tokens, n_head, 1);
    struct ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d_head, n_kv, n_head, 1);
    struct ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d_head, n_kv, n_head, 1);
    const int n_tokens_pad = pad_to_mask(n_tokens);
    struct ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv, n_tokens_pad);

    struct ggml_tensor * fattn = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f, 0.0f, 0.0f);

    // Create paged attention tensors
    struct ggml_tensor * block_table = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, max_blocks_per_seq, batch_size);
    struct ggml_tensor * seq_lens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch_size);

    // Set paged attention
    ggml_flash_attn_ext_set_paged(fattn, block_table, seq_lens);

    // Verify they were set
    assert(fattn->src[7] == block_table);
    assert(fattn->src[8] == seq_lens);

    ggml_free(ctx);
}

TEST(set_paged_through_view) {
    // Test setting paged attention through a VIEW operation
    struct ggml_context * ctx = create_test_ctx();

    const int n_tokens = 16;
    const int n_kv = 64;
    const int n_head = 8;
    const int d_head = 64;
    const int batch_size = 4;
    const int max_blocks_per_seq = 8;

    struct ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d_head, n_tokens, n_head, 1);
    struct ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d_head, n_kv, n_head, 1);
    struct ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d_head, n_kv, n_head, 1);
    const int n_tokens_pad = pad_to_mask(n_tokens);
    struct ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv, n_tokens_pad);

    struct ggml_tensor * fattn = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f, 0.0f, 0.0f);

    // Wrap in view
    struct ggml_tensor * view = ggml_view_1d(ctx, fattn, ggml_nelements(fattn), 0);

    struct ggml_tensor * block_table = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, max_blocks_per_seq, batch_size);
    struct ggml_tensor * seq_lens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch_size);

    // Set paged attention through view
    ggml_flash_attn_ext_set_paged(view, block_table, seq_lens);

    // Verify they were set on the underlying flash attention tensor
    assert(fattn->src[7] == block_table);
    assert(fattn->src[8] == seq_lens);

    ggml_free(ctx);
}

TEST(set_paged_through_reshape) {
    // Test setting paged attention through a RESHAPE operation
    struct ggml_context * ctx = create_test_ctx();

    const int n_tokens = 16;
    const int n_kv = 64;
    const int n_head = 8;
    const int d_head = 64;
    const int batch_size = 4;
    const int max_blocks_per_seq = 8;

    struct ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d_head, n_tokens, n_head, 1);
    struct ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d_head, n_kv, n_head, 1);
    struct ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d_head, n_kv, n_head, 1);
    const int n_tokens_pad = pad_to_mask(n_tokens);
    struct ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv, n_tokens_pad);

    struct ggml_tensor * fattn = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f, 0.0f, 0.0f);

    // Wrap in reshape
    struct ggml_tensor * reshaped = ggml_reshape_2d(ctx, fattn, d_head, n_tokens * n_head);

    struct ggml_tensor * block_table = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, max_blocks_per_seq, batch_size);
    struct ggml_tensor * seq_lens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch_size);

    // Set paged attention through reshape
    ggml_flash_attn_ext_set_paged(reshaped, block_table, seq_lens);

    // Verify they were set on the underlying flash attention tensor
    assert(fattn->src[7] == block_table);
    assert(fattn->src[8] == seq_lens);

    ggml_free(ctx);
}

TEST(set_paged_chain) {
    // Test setting paged attention through a chain of operations
    struct ggml_context * ctx = create_test_ctx();

    const int n_tokens = 16;
    const int n_kv = 64;
    const int n_head = 8;
    const int d_head = 64;
    const int batch_size = 4;
    const int max_blocks_per_seq = 8;

    struct ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d_head, n_tokens, n_head, 1);
    struct ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d_head, n_kv, n_head, 1);
    struct ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d_head, n_kv, n_head, 1);
    const int n_tokens_pad = pad_to_mask(n_tokens);
    struct ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv, n_tokens_pad);

    struct ggml_tensor * fattn = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f, 0.0f, 0.0f);

    // Chain: fattn -> cont -> reshape -> view
    struct ggml_tensor * cont = ggml_cont(ctx, fattn);
    struct ggml_tensor * reshaped = ggml_reshape_2d(ctx, cont, d_head, n_tokens * n_head);
    struct ggml_tensor * view = ggml_view_1d(ctx, reshaped, d_head * n_tokens * n_head, 0);

    struct ggml_tensor * block_table = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, max_blocks_per_seq, batch_size);
    struct ggml_tensor * seq_lens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch_size);

    // Set paged attention through chain
    ggml_flash_attn_ext_set_paged(view, block_table, seq_lens);

    // Verify they were set on the underlying flash attention tensor
    assert(fattn->src[7] == block_table);
    assert(fattn->src[8] == seq_lens);

    ggml_free(ctx);
}

TEST(set_paged_no_fattn_tensor) {
    // Test that function doesn't crash when no flash attention tensor is found
    struct ggml_context * ctx = create_test_ctx();

    struct ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
    struct ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
    struct ggml_tensor * matmul = ggml_mul_mat(ctx, a, b);

    struct ggml_tensor * block_table = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 8, 4);
    struct ggml_tensor * seq_lens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 4);

    // Should silently return without setting anything (no crash)
    ggml_flash_attn_ext_set_paged(matmul, block_table, seq_lens);

    ggml_free(ctx);
}

// =============================================================================
// Combined Tests
// =============================================================================

TEST(set_both_seq_ids_and_paged) {
    // Test setting both seq_ids and paged attention on the same flash attention tensor
    struct ggml_context * ctx = create_test_ctx();

    const int n_tokens = 16;
    const int n_kv = 64;
    const int n_head = 8;
    const int d_head = 64;
    const int batch_size = 4;
    const int max_blocks_per_seq = 8;

    struct ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d_head, n_tokens, n_head, 1);
    struct ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d_head, n_kv, n_head, 1);
    struct ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d_head, n_kv, n_head, 1);
    const int n_tokens_pad = pad_to_mask(n_tokens);
    struct ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv, n_tokens_pad);

    struct ggml_tensor * fattn = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f, 0.0f, 0.0f);

    // Create all extension tensors
    struct ggml_tensor * q_seq_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    struct ggml_tensor * kv_seq_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_kv);
    struct ggml_tensor * block_table = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, max_blocks_per_seq, batch_size);
    struct ggml_tensor * seq_lens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch_size);

    // Set both seq_ids and paged attention
    ggml_flash_attn_ext_set_seq_ids(fattn, q_seq_ids, kv_seq_ids);
    ggml_flash_attn_ext_set_paged(fattn, block_table, seq_lens);

    // Verify all were set correctly
    assert(fattn->src[5] == q_seq_ids);
    assert(fattn->src[6] == kv_seq_ids);
    assert(fattn->src[7] == block_table);
    assert(fattn->src[8] == seq_lens);

    ggml_free(ctx);
}

TEST(set_through_mixed_chain) {
    // Test setting both through a mixed chain of view/reshape operations
    struct ggml_context * ctx = create_test_ctx();

    const int n_tokens = 16;
    const int n_kv = 64;
    const int n_head = 8;
    const int d_head = 64;
    const int batch_size = 4;
    const int max_blocks_per_seq = 8;

    struct ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d_head, n_tokens, n_head, 1);
    struct ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d_head, n_kv, n_head, 1);
    struct ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d_head, n_kv, n_head, 1);
    const int n_tokens_pad = pad_to_mask(n_tokens);
    struct ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv, n_tokens_pad);

    struct ggml_tensor * fattn = ggml_flash_attn_ext(ctx, q, k, v, mask, 1.0f, 0.0f, 0.0f);

    // Create a deep chain: fattn -> permute -> cont -> reshape -> view
    struct ggml_tensor * permuted = ggml_permute(ctx, fattn, 0, 2, 1, 3);
    struct ggml_tensor * cont = ggml_cont(ctx, permuted);
    struct ggml_tensor * reshaped = ggml_reshape_2d(ctx, cont, d_head * n_head, n_tokens);
    struct ggml_tensor * view = ggml_view_1d(ctx, reshaped, d_head * n_tokens * n_head, 0);

    struct ggml_tensor * q_seq_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    struct ggml_tensor * kv_seq_ids = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_kv);
    struct ggml_tensor * block_table = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, max_blocks_per_seq, batch_size);
    struct ggml_tensor * seq_lens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, batch_size);

    // Set both through the deep chain
    ggml_flash_attn_ext_set_seq_ids(view, q_seq_ids, kv_seq_ids);
    ggml_flash_attn_ext_set_paged(view, block_table, seq_lens);

    // Verify all were set on the underlying flash attention tensor
    assert(fattn->src[5] == q_seq_ids);
    assert(fattn->src[6] == kv_seq_ids);
    assert(fattn->src[7] == block_table);
    assert(fattn->src[8] == seq_lens);

    ggml_free(ctx);
}

// =============================================================================
// Main
// =============================================================================

int main() {
    printf("=== GGML Flash Attention Extension Tests ===\n\n");

    printf("ggml_flash_attn_ext_set_seq_ids tests:\n");
    RUN_TEST(set_seq_ids_null_inputs);
    RUN_TEST(set_seq_ids_direct_fattn);
    RUN_TEST(set_seq_ids_through_view);
    RUN_TEST(set_seq_ids_through_reshape);
    RUN_TEST(set_seq_ids_through_cont);
    RUN_TEST(set_seq_ids_through_permute);
    RUN_TEST(set_seq_ids_through_transpose);
    RUN_TEST(set_seq_ids_chain);
    RUN_TEST(set_seq_ids_no_fattn_tensor);

    printf("\n");

    printf("ggml_flash_attn_ext_set_paged tests:\n");
    RUN_TEST(set_paged_null_inputs);
    RUN_TEST(set_paged_direct_fattn);
    RUN_TEST(set_paged_through_view);
    RUN_TEST(set_paged_through_reshape);
    RUN_TEST(set_paged_chain);
    RUN_TEST(set_paged_no_fattn_tensor);

    printf("\n");

    printf("Combined tests:\n");
    RUN_TEST(set_both_seq_ids_and_paged);
    RUN_TEST(set_through_mixed_chain);

    printf("\n=== All tests passed! ===\n");

    return 0;
}
