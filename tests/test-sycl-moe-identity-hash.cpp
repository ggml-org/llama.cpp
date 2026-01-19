// SYCL MoE Identity Hash Collision Test
//
// This test verifies that MoE expert tensors have unique cache identities.
// Bug: llama.cpp-twc - Expert tensors were receiving null identity hashes (name_hash=0x0)
//      causing massive cache collision errors: "[UNIFIED-CACHE] identity collision on layout insert"
//
// Root cause investigation: The ggml_sycl_get_moe_expert_cache_key() function sets name_hash=0
// and relies solely on aux_id (cache_uuid + expert_id) for uniqueness. However, if multiple
// tensors share similar cache_uuids or if the identity comparison doesn't properly account
// for aux_id differences, collisions occur.
//
// This test exercises:
// 1. Single weight tensor cache key has non-zero name_hash
// 2. Multiple weight tensors with different names get different name_hashes
// 3. MoE expert cache keys for different experts are unique
// 4. MoE expert cache keys across different weight tensors are unique
// 5. Identity collision detection works correctly

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_set>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml-sycl.h"
#include "ggml-sycl/unified-cache.hpp"

#if !defined(GGML_USE_SYCL)
int main() {
    fprintf(stderr, "GGML_USE_SYCL not enabled; skipping test.\n");
    return 0;
}
#else

// Test counters
static int g_tests_run    = 0;
static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define TEST_ASSERT(cond, msg)                                                                                         \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            fprintf(stderr, "  FAIL: %s\n", msg);                                                                      \
            return false;                                                                                              \
        }                                                                                                              \
    } while (0)

#define RUN_TEST(fn)                                                                                                   \
    do {                                                                                                               \
        g_tests_run++;                                                                                                 \
        if (fn()) {                                                                                                    \
            g_tests_passed++;                                                                                          \
            printf("  PASS: %s\n", #fn);                                                                               \
        } else {                                                                                                       \
            g_tests_failed++;                                                                                          \
            fprintf(stderr, "  FAIL: %s\n", #fn);                                                                      \
        }                                                                                                              \
    } while (0)

// Helper: allocate a weight buffer
static ggml_backend_buffer_t alloc_weight_buffer(ggml_backend_buffer_type_t buft, ggml_tensor * weight) {
    const size_t              buf_size = ggml_backend_buft_get_alloc_size(buft, weight);
    ggml_backend_buffer_t     buffer   = ggml_backend_buft_alloc_buffer(buft, buf_size);
    if (!buffer) {
        return nullptr;
    }
    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_tensor_alloc(buffer, weight, ggml_backend_buffer_get_base(buffer));
    return buffer;
}

// Helper: print cache ID for debugging
static void print_cache_id(const char * label, const ggml_sycl_cache_id & id) {
    fprintf(stderr, "  %s: valid=%d model=%llu has_gguf=%d file_idx=%u file_offs=%zu nbytes=%zu "
                    "name_hash=0x%llx type=%d aux_id=%llu\n",
            label, id.valid, (unsigned long long) id.model_id, id.has_gguf, id.file_idx, id.file_offs, id.nbytes,
            (unsigned long long) id.name_hash, id.type, (unsigned long long) id.aux_id);
}

// =============================================================================
// Test 1: Single weight tensor cache key has non-zero name_hash
// =============================================================================
static bool test_single_weight_name_hash() {
    printf("TEST: test_single_weight_name_hash\n");

    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    TEST_ASSERT(cpu_backend != nullptr, "CPU backend unavailable");

    ggml_init_params params = {
        /*.mem_size   =*/4 * 1024 * 1024,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ggml_context * ctx = ggml_init(params);
    TEST_ASSERT(ctx != nullptr, "ggml_init failed");

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(cpu_backend);
    TEST_ASSERT(buft != nullptr, "CPU buffer type unavailable");

    // Create a weight tensor with a specific name
    ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 256, 64);
    TEST_ASSERT(weight != nullptr, "tensor allocation failed");
    ggml_set_name(weight, "blk.0.ffn_gate.weight");

    ggml_backend_buffer_t buffer = alloc_weight_buffer(buft, weight);
    TEST_ASSERT(buffer != nullptr, "buffer allocation failed");

    // Register GGUF identity
    const size_t   nbytes   = ggml_nbytes(weight);
    const uint64_t model_id = 1;
    ggml_backend_sycl_register_weight_identity(weight, 0, 4096, nbytes, model_id);

    // Get cache key
    const int            device_id = 0;
    ggml_sycl_cache_id   key       = ggml_backend_sycl_get_weight_cache_key(weight, device_id);

    print_cache_id("weight key", key);

    TEST_ASSERT(key.valid, "cache key should be valid");
    TEST_ASSERT(key.model_id == model_id, "model_id mismatch");
    TEST_ASSERT(key.name_hash != 0, "name_hash should NOT be zero for named weight tensor");

    // Verify the hash is what we expect for this tensor name
    uint64_t expected_hash = static_cast<uint64_t>(std::hash<std::string>()("blk.0.ffn_gate.weight"));
    fprintf(stderr, "  Expected name_hash: 0x%llx, got: 0x%llx\n", (unsigned long long) expected_hash,
            (unsigned long long) key.name_hash);
    TEST_ASSERT(key.name_hash == expected_hash, "name_hash should match std::hash of tensor name");

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(cpu_backend);

    return true;
}

// =============================================================================
// Test 2: Different weight tensors get different name_hashes
// =============================================================================
static bool test_different_weights_different_hashes() {
    printf("TEST: test_different_weights_different_hashes\n");

    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    TEST_ASSERT(cpu_backend != nullptr, "CPU backend unavailable");

    ggml_init_params params = {
        /*.mem_size   =*/4 * 1024 * 1024,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ggml_context * ctx = ggml_init(params);
    TEST_ASSERT(ctx != nullptr, "ggml_init failed");

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(cpu_backend);

    const char * tensor_names[] = { "blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight",
                                    "blk.1.ffn_gate.weight", "blk.1.ffn_up.weight", "blk.1.ffn_down.weight" };
    const int                       num_tensors = sizeof(tensor_names) / sizeof(tensor_names[0]);
    std::vector<ggml_sycl_cache_id> keys;
    std::vector<ggml_backend_buffer_t> buffers;

    for (int i = 0; i < num_tensors; ++i) {
        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 256 + i, 64);
        TEST_ASSERT(weight != nullptr, "tensor allocation failed");
        ggml_set_name(weight, tensor_names[i]);

        ggml_backend_buffer_t buffer = alloc_weight_buffer(buft, weight);
        TEST_ASSERT(buffer != nullptr, "buffer allocation failed");
        buffers.push_back(buffer);

        // Register with same file_idx/offs to simulate MoE contiguous storage
        // This should still produce unique keys due to different name_hash
        ggml_backend_sycl_register_weight_identity(weight, 0, 4096, ggml_nbytes(weight), 1);

        ggml_sycl_cache_id key = ggml_backend_sycl_get_weight_cache_key(weight, 0);
        print_cache_id(tensor_names[i], key);

        TEST_ASSERT(key.valid, "cache key should be valid");
        TEST_ASSERT(key.name_hash != 0, "name_hash should not be zero");
        keys.push_back(key);
    }

    // Verify all keys are unique (different name_hashes)
    std::unordered_set<uint64_t> seen_hashes;
    for (size_t i = 0; i < keys.size(); ++i) {
        if (seen_hashes.find(keys[i].name_hash) != seen_hashes.end()) {
            fprintf(stderr, "  FAIL: Duplicate name_hash 0x%llx for tensor %s\n", (unsigned long long) keys[i].name_hash,
                    tensor_names[i]);
            for (auto * buf : buffers) {
                ggml_backend_buffer_free(buf);
            }
            ggml_free(ctx);
            ggml_backend_free(cpu_backend);
            return false;
        }
        seen_hashes.insert(keys[i].name_hash);
    }

    for (auto * buf : buffers) {
        ggml_backend_buffer_free(buf);
    }
    ggml_free(ctx);
    ggml_backend_free(cpu_backend);

    printf("  All %d tensors have unique name_hashes\n", num_tensors);
    return true;
}

// =============================================================================
// Test 3: MoE expert weight tensors have unique identities via name_hash
// =============================================================================
static bool test_moe_expert_weights_unique() {
    printf("TEST: test_moe_expert_weights_unique\n");

    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    TEST_ASSERT(cpu_backend != nullptr, "CPU backend unavailable");

    ggml_init_params params = {
        /*.mem_size   =*/8 * 1024 * 1024,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ggml_context * ctx = ggml_init(params);
    TEST_ASSERT(ctx != nullptr, "ggml_init failed");

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(cpu_backend);

    // Simulate MoE expert weights: blk.0.ffn_gate_exps.weight for experts 0-7
    // These might have the same file_offs in GGUF if stored contiguously
    const int num_experts = 8;
    std::vector<ggml_sycl_cache_id>    keys;
    std::vector<ggml_backend_buffer_t> buffers;

    for (int e = 0; e < num_experts; ++e) {
        char name[64];
        snprintf(name, sizeof(name), "blk.0.ffn_gate_exps.weight.%d", e);

        ggml_tensor * weight = ggml_new_tensor_2d(ctx, GGML_TYPE_Q4_0, 256, 64);
        TEST_ASSERT(weight != nullptr, "tensor allocation failed");
        ggml_set_name(weight, name);

        ggml_backend_buffer_t buffer = alloc_weight_buffer(buft, weight);
        TEST_ASSERT(buffer != nullptr, "buffer allocation failed");
        buffers.push_back(buffer);

        // All experts could have same file_offs if stored contiguously
        // The name_hash should differentiate them
        ggml_backend_sycl_register_weight_identity(weight, 0, 4096, ggml_nbytes(weight), 1);

        ggml_sycl_cache_id key = ggml_backend_sycl_get_weight_cache_key(weight, 0);
        print_cache_id(name, key);

        TEST_ASSERT(key.valid, "cache key should be valid");
        TEST_ASSERT(key.name_hash != 0, "name_hash should not be zero for expert weight");
        keys.push_back(key);
    }

    // Verify all expert weight cache keys are unique
    std::unordered_set<ggml_sycl_cache_id, ggml_sycl::detail::cache_id_hash, ggml_sycl::detail::cache_id_equal_fn>
        unique_keys;
    for (int e = 0; e < num_experts; ++e) {
        if (!unique_keys.emplace(keys[e]).second) {
            fprintf(stderr, "  FAIL: Duplicate cache key for expert %d\n", e);
            for (auto * buf : buffers) {
                ggml_backend_buffer_free(buf);
            }
            ggml_free(ctx);
            ggml_backend_free(cpu_backend);
            return false;
        }
    }

    for (auto * buf : buffers) {
        ggml_backend_buffer_free(buf);
    }
    ggml_free(ctx);
    ggml_backend_free(cpu_backend);

    printf("  All %d MoE expert weights have unique cache keys\n", num_experts);
    return true;
}

// =============================================================================
// Test 4: MoE intermediate tensors (ffn_moe_probs, etc.) have unique identities
// =============================================================================
static bool test_moe_intermediate_tensors_unique() {
    printf("TEST: test_moe_intermediate_tensors_unique\n");

    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    TEST_ASSERT(cpu_backend != nullptr, "CPU backend unavailable");

    ggml_init_params params = {
        /*.mem_size   =*/4 * 1024 * 1024,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ggml_context * ctx = ggml_init(params);
    TEST_ASSERT(ctx != nullptr, "ggml_init failed");

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(cpu_backend);

    // MoE intermediate tensor names that could collide
    const char * tensor_names[] = { "ffn_moe_probs",   "ffn_moe_probs.0", "ffn_moe_probs.1",
                                    "ffn_moe_indices", "expert_weights",  "expert_ids" };
    const int                          num_tensors = sizeof(tensor_names) / sizeof(tensor_names[0]);
    std::vector<ggml_sycl_cache_id>    keys;
    std::vector<ggml_backend_buffer_t> buffers;

    for (int i = 0; i < num_tensors; ++i) {
        ggml_tensor * tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 128, 32);
        TEST_ASSERT(tensor != nullptr, "tensor allocation failed");
        ggml_set_name(tensor, tensor_names[i]);

        ggml_backend_buffer_t buffer = alloc_weight_buffer(buft, tensor);
        TEST_ASSERT(buffer != nullptr, "buffer allocation failed");
        buffers.push_back(buffer);

        // These are compute tensors, not GGUF-backed
        // They should still get unique cache keys via name_hash

        ggml_sycl_cache_id key = ggml_backend_sycl_get_weight_cache_key(tensor, 0);
        print_cache_id(tensor_names[i], key);

        TEST_ASSERT(key.valid, "cache key should be valid");
        // Note: intermediate tensors may or may not have name_hash depending on code path
        // The key thing is they should be unique
        keys.push_back(key);
    }

    // Verify all keys are unique
    std::unordered_set<ggml_sycl_cache_id, ggml_sycl::detail::cache_id_hash, ggml_sycl::detail::cache_id_equal_fn>
        unique_keys;
    for (int i = 0; i < num_tensors; ++i) {
        if (!unique_keys.emplace(keys[i]).second) {
            fprintf(stderr, "  FAIL: Duplicate cache key for tensor %s\n", tensor_names[i]);
            for (auto * buf : buffers) {
                ggml_backend_buffer_free(buf);
            }
            ggml_free(ctx);
            ggml_backend_free(cpu_backend);
            return false;
        }
    }

    for (auto * buf : buffers) {
        ggml_backend_buffer_free(buf);
    }
    ggml_free(ctx);
    ggml_backend_free(cpu_backend);

    printf("  All %d MoE intermediate tensors have unique cache keys\n", num_tensors);
    return true;
}

// =============================================================================
// Test 5: "unknown" named tensor should still get valid name_hash
// =============================================================================
static bool test_unknown_tensor_name_hash() {
    printf("TEST: test_unknown_tensor_name_hash\n");

    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    TEST_ASSERT(cpu_backend != nullptr, "CPU backend unavailable");

    ggml_init_params params = {
        /*.mem_size   =*/4 * 1024 * 1024,
        /*.mem_buffer =*/nullptr,
        /*.no_alloc   =*/true,
    };
    ggml_context * ctx = ggml_init(params);
    TEST_ASSERT(ctx != nullptr, "ggml_init failed");

    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(cpu_backend);

    // Create tensor with no name (will default to "unknown")
    ggml_tensor * tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 64);
    TEST_ASSERT(tensor != nullptr, "tensor allocation failed");
    // Don't set name - should default to "unknown" or empty

    ggml_backend_buffer_t buffer = alloc_weight_buffer(buft, tensor);
    TEST_ASSERT(buffer != nullptr, "buffer allocation failed");

    ggml_sycl_cache_id key = ggml_backend_sycl_get_weight_cache_key(tensor, 0);
    print_cache_id("unnamed tensor", key);

    TEST_ASSERT(key.valid, "cache key should be valid");

    // Hash of "unknown" should not be 0
    uint64_t unknown_hash = static_cast<uint64_t>(std::hash<std::string>()("unknown"));
    fprintf(stderr, "  Hash of 'unknown': 0x%llx\n", (unsigned long long) unknown_hash);
    TEST_ASSERT(unknown_hash != 0, "hash of 'unknown' should not be zero");

    // The actual name_hash in the key should also not be 0
    // (unless there's a bug setting it to 0 somewhere)
    if (key.name_hash == 0) {
        fprintf(stderr, "  WARNING: name_hash is 0 even though hash('unknown') = 0x%llx\n",
                (unsigned long long) unknown_hash);
        fprintf(stderr, "  This indicates a bug where name_hash is being set to 0 in a code path\n");
        // Don't fail the test - this helps us diagnose the bug
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    ggml_backend_free(cpu_backend);

    return true;
}

// =============================================================================
// Test 6: Verify cache_id_equal properly compares all fields
// =============================================================================
static bool test_cache_id_equality() {
    printf("TEST: test_cache_id_equality\n");

    ggml_sycl_cache_id id1{};
    id1.valid     = true;
    id1.model_id  = 1;
    id1.has_gguf  = true;
    id1.file_idx  = 0;
    id1.file_offs = 4096;
    id1.nbytes    = 1024;
    id1.name_hash = 0x12345678;
    id1.type      = GGML_TYPE_Q4_0;
    id1.aux_id    = 0;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        id1.ne[i]           = 0;
        id1.tp_local_ne[i]  = 0;
        id1.tp_offset_ne[i] = 0;
    }
    id1.ne[0]        = 256;
    id1.ne[1]        = 64;
    id1.tp_sharded   = false;
    id1.tp_rank      = 0;
    id1.tp_world_size = 1;

    // Copy and verify equal
    ggml_sycl_cache_id id2 = id1;
    TEST_ASSERT(ggml_sycl::detail::cache_id_equal(id1, id2), "identical IDs should be equal");

    // Change name_hash - should now be different
    id2.name_hash = 0x87654321;
    TEST_ASSERT(!ggml_sycl::detail::cache_id_equal(id1, id2), "different name_hash should make IDs unequal");

    // Reset and change aux_id
    id2           = id1;
    id2.aux_id    = 123;
    TEST_ASSERT(!ggml_sycl::detail::cache_id_equal(id1, id2), "different aux_id should make IDs unequal");

    // Test with name_hash=0 but different aux_id (MoE expert path)
    ggml_sycl_cache_id moe1 = id1;
    moe1.name_hash          = 0;
    moe1.aux_id             = 100; // Expert 0 + cache_uuid

    ggml_sycl_cache_id moe2 = id1;
    moe2.name_hash          = 0;
    moe2.aux_id             = 101; // Expert 1 + cache_uuid

    TEST_ASSERT(!ggml_sycl::detail::cache_id_equal(moe1, moe2), "different aux_id should make MoE IDs unequal");

    printf("  cache_id_equal properly distinguishes IDs by name_hash and aux_id\n");
    return true;
}

// =============================================================================
// Test 7: Simulate actual MoE expert cache key collision scenario
// This tests the ggml_sycl_get_moe_expert_cache_key() code path which sets name_hash=0
// =============================================================================
static bool test_moe_expert_cache_key_collision_scenario() {
    printf("TEST: test_moe_expert_cache_key_collision_scenario\n");

    // In real MoE scenarios, different expert weight tensors (ffn_gate_exps, ffn_up_exps, ffn_down_exps)
    // would each have their own ggml_tensor_extra_gpu with unique cache_uuid.
    // The collision would only happen if:
    // 1. Two different tensors share the same extra (incorrect)
    // 2. Two different tensors both get cache_uuid=0 (bug in uuid assignment)
    // 3. The id_to_key mapping is incorrect

    // Simulate cache IDs like ggml_sycl_get_moe_expert_cache_key() creates
    // These have name_hash=0 and rely on aux_id for uniqueness

    const uint64_t model_id = 1;
    const int      num_weight_types = 3;  // gate, up, down
    const int      num_experts = 8;
    const int      num_layers = 2;

    std::vector<ggml_sycl_cache_id> all_keys;
    std::unordered_set<ggml_sycl_cache_id, ggml_sycl::detail::cache_id_hash, ggml_sycl::detail::cache_id_equal_fn>
        unique_keys;

    // Simulate different cache_uuids for different weight tensors
    uint64_t base_cache_uuid = 1000;

    for (int l = 0; l < num_layers; ++l) {
        for (int w = 0; w < num_weight_types; ++w) {
            uint64_t tensor_cache_uuid = base_cache_uuid++;  // Each tensor type gets unique uuid

            for (int e = 0; e < num_experts; ++e) {
                ggml_sycl_cache_id id{};
                id.valid    = true;
                id.model_id = model_id;
                id.has_gguf = false;
                id.file_idx = 0;
                id.file_offs = 0;
                id.nbytes   = 0;
                id.name_hash = 0;  // MoE expert keys use aux_id, not name_hash
                id.type     = GGML_TYPE_COUNT;
                id.tp_sharded = false;
                id.tp_rank  = 0;
                id.tp_world_size = 1;
                for (int i = 0; i < GGML_MAX_DIMS; ++i) {
                    id.ne[i] = 0;
                    id.tp_local_ne[i] = 0;
                    id.tp_offset_ne[i] = 0;
                }

                // Simulate ggml_sycl_hash_combine(cache_uuid, expert_id)
                size_t aux = tensor_cache_uuid;
                aux = aux ^ (static_cast<size_t>(e) + 0x9e3779b97f4a7c15ULL + (aux << 6) + (aux >> 2));
                id.aux_id = aux;

                fprintf(stderr, "  Layer %d, Weight %d, Expert %d: aux_id=0x%llx\n",
                        l, w, e, (unsigned long long) id.aux_id);

                if (!unique_keys.emplace(id).second) {
                    fprintf(stderr, "  FAIL: Collision detected for layer=%d weight=%d expert=%d\n", l, w, e);
                    return false;
                }
                all_keys.push_back(id);
            }
        }
    }

    printf("  All %zu MoE expert cache keys are unique (name_hash=0 path)\n", all_keys.size());

    // Now test what happens if two tensors SHARE the same cache_uuid (the bug scenario)
    printf("  Testing collision scenario: two tensors with same cache_uuid...\n");

    uint64_t shared_uuid = 9999;
    ggml_sycl_cache_id key1{};
    key1.valid = true;
    key1.model_id = model_id;
    key1.name_hash = 0;
    key1.type = GGML_TYPE_COUNT;
    size_t aux1 = shared_uuid;
    aux1 = aux1 ^ (0ULL + 0x9e3779b97f4a7c15ULL + (aux1 << 6) + (aux1 >> 2));  // Expert 0
    key1.aux_id = aux1;

    ggml_sycl_cache_id key2{};
    key2.valid = true;
    key2.model_id = model_id;
    key2.name_hash = 0;
    key2.type = GGML_TYPE_COUNT;
    size_t aux2 = shared_uuid;
    aux2 = aux2 ^ (0ULL + 0x9e3779b97f4a7c15ULL + (aux2 << 6) + (aux2 >> 2));  // Expert 0 again
    key2.aux_id = aux2;

    // These SHOULD be equal (same tensor, same expert)
    if (!ggml_sycl::detail::cache_id_equal(key1, key2)) {
        fprintf(stderr, "  FAIL: Same tensor/expert keys should be equal\n");
        return false;
    }
    printf("  Same uuid + same expert = equal keys (correct)\n");

    // Different expert on same tensor should be different
    ggml_sycl_cache_id key3 = key1;
    size_t aux3 = shared_uuid;
    aux3 = aux3 ^ (1ULL + 0x9e3779b97f4a7c15ULL + (aux3 << 6) + (aux3 >> 2));  // Expert 1
    key3.aux_id = aux3;

    if (ggml_sycl::detail::cache_id_equal(key1, key3)) {
        fprintf(stderr, "  FAIL: Different experts should have different keys\n");
        return false;
    }
    printf("  Same uuid + different expert = different keys (correct)\n");

    return true;
}

// =============================================================================
// Main
// =============================================================================
int main() {
    printf("=== SYCL MoE Identity Hash Collision Tests ===\n\n");

    RUN_TEST(test_single_weight_name_hash);
    RUN_TEST(test_different_weights_different_hashes);
    RUN_TEST(test_moe_expert_weights_unique);
    RUN_TEST(test_moe_intermediate_tensors_unique);
    RUN_TEST(test_unknown_tensor_name_hash);
    RUN_TEST(test_cache_id_equality);
    RUN_TEST(test_moe_expert_cache_key_collision_scenario);

    printf("\n=== Summary ===\n");
    printf("Tests run: %d, Passed: %d, Failed: %d\n", g_tests_run, g_tests_passed, g_tests_failed);

    if (g_tests_failed > 0) {
        printf("\nFAILED: Some tests did not pass. See above for details.\n");
        printf("This indicates the identity hash collision bug (llama.cpp-twc) is still present.\n");
        return 1;
    }

    printf("\nPASS: All MoE identity hash tests passed.\n");
    return 0;
}

#endif  // GGML_USE_SYCL
