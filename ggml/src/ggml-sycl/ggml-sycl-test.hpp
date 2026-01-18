#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-sycl.h"

#include <cstdint>

namespace ggml_sycl {

// Test-only layout override hooks (no env handling inside the library).
// Use the guard in tests to temporarily force a layout during a scoped operation.
void test_set_layout_override(ggml_layout_mode layout);
void test_clear_layout_override();
bool test_get_layout_override(ggml_layout_mode * out);
void test_clear_host_weight_registry();
bool test_backend_supports_graphs(ggml_backend_t backend);
bool test_backend_graphs_disabled(ggml_backend_t backend);
size_t test_graph_pinned_entry_count(ggml_backend_t backend);
size_t test_layout_bytes(const ggml_tensor * tensor, ggml_layout_mode layout, int device);

inline ggml_sycl_cache_id test_make_cache_id(const void * tag, uint64_t model_id = 1) {
    ggml_sycl_cache_id id{};
    id.valid    = true;
    id.model_id = model_id;
    id.aux_id   = reinterpret_cast<uintptr_t>(tag);
    return id;
}

struct test_layout_override_guard {
    explicit test_layout_override_guard(ggml_layout_mode layout) {
        test_set_layout_override(layout);
    }
    ~test_layout_override_guard() {
        test_clear_layout_override();
    }
};

}  // namespace ggml_sycl
