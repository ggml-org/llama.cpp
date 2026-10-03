#pragma once

#include "mtmd.h"
#include "ggml-cpp.h"

#include <array>
#include <cstdio>
#include <memory>
#include <string>

struct mtmd_projector_tensor_source {
    std::string name;
    ggml_type type;
    std::array<int64_t, GGML_MAX_DIMS> shape;
    size_t offset = 0, bytes = 0;
};

// Shared file handle and copied tensor manifest; no second copy of weight bytes is kept in RAM.
class MTMD_API mtmd_projector_source {
public:
    // Open one reload source; the supplied GGUF metadata need not outlive this call.
    static std::shared_ptr<const mtmd_projector_source> open(const char * path, const gguf_context * metadata);
    // Take ownership of file, including on failure. Reads share a checked, serialized file cursor.
    static std::shared_ptr<const mtmd_projector_source> from_file(FILE * file, const gguf_context * metadata);
    ~mtmd_projector_source();
    const mtmd_projector_tensor_source * find(const char * name) const noexcept;
    // Read exactly one tensor payload; reject unknown names and wrong sizes before writing output.
    bool read(const char * name, void * output, size_t bytes) const;

private:
    struct implementation;
    explicit mtmd_projector_source(std::unique_ptr<implementation> impl);
    std::unique_ptr<implementation> impl;
};

// Stable tensor descriptors retain their reload source independently of device storage.
// Binding changes are owner-thread-only; shared references protect lifetime, not parallel mutation.
class MTMD_API mtmd_projector_metadata {
public:
    // Adopt an unbound tensor context whose names, shapes and types match the source manifest.
    static std::shared_ptr<mtmd_projector_metadata> create(ggml_context_ptr context,
        std::shared_ptr<const mtmd_projector_source> source);
    ggml_context * context() const noexcept;
    const std::shared_ptr<const mtmd_projector_source> & source() const noexcept;

private:
    friend class mtmd_projector_weights;
    mtmd_projector_metadata() = default;
    ggml_context_ptr tensors;
    std::shared_ptr<const mtmd_projector_source> reload;
    bool loading = false;
};

// Eager resident binding. Shared owners keep both descriptors and backend storage alive.
// Callers drain execution and retire captures before returning their last owner; no runtime unload API yet.
class MTMD_API mtmd_projector_weights {
public:
    // Allocate and load once; reject reentry or an already-bound metadata context.
    static std::shared_ptr<mtmd_projector_weights> allocate(std::shared_ptr<mtmd_projector_metadata> metadata,
        ggml_backend_buffer_type_t type, bool skip_upload = false,
        mtmd_progress_callback progress = nullptr, void * user_data = nullptr);
    ~mtmd_projector_weights();
    ggml_backend_buffer_t buffer() const noexcept;
    const std::shared_ptr<mtmd_projector_metadata> & metadata() const noexcept;

private:
    mtmd_projector_weights() = default;
    std::shared_ptr<mtmd_projector_metadata> descriptors;
    ggml_backend_buffer_ptr storage;
};

// Internal ownership seam for lifecycle adapters and tests, not a new public mtmd C API.
MTMD_API std::shared_ptr<const mtmd_projector_weights> mtmd_acquire_projector_weights(const mtmd_context * ctx) noexcept;
struct clip_ctx;
std::shared_ptr<const mtmd_projector_weights> clip_acquire_projector_weights(const clip_ctx * ctx) noexcept;
