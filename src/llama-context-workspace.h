#pragma once

#include "../ggml/src/ggml-backend-memory.h"

#include <memory>

struct llama_compute_arena_deleter {
    // Release the arena reference owned by one context binding.
    void operator()(ggml_backend_memory_arena_t arena) const noexcept {
        ggml_backend_memory_arena_free(arena);
    }
};

using llama_compute_arena_ptr = std::unique_ptr<ggml_backend_memory_arena, llama_compute_arena_deleter>;

struct llama_compute_arena_binding {
    ggml_backend_buffer_type_t buft = nullptr;
    size_t capacity = 0;
    size_t first_slot = 0;
    llama_compute_arena_ptr arena;
};
