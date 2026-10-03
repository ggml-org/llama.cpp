#pragma once

#include "mtmd.h"
#include "ggml-cpp.h"
#include "../../ggml/src/ggml-backend-memory.h"

#include <array>
#include <cstdio>
#include <memory>
#include <string>
#include <functional>

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
// Use the residency lifecycle to drain execution and retire captures before returning its last owner.
class MTMD_API mtmd_projector_weights {
public:
    // Allocate and load once; reject reentry or an already-bound metadata context.
    static std::shared_ptr<mtmd_projector_weights> allocate(std::shared_ptr<mtmd_projector_metadata> metadata,
        ggml_backend_buffer_type_t type, bool skip_upload = false,
        mtmd_progress_callback progress = nullptr, void * user_data = nullptr);
    ~mtmd_projector_weights();
    ggml_backend_buffer_t buffer() const noexcept;
    const std::shared_ptr<mtmd_projector_metadata> & metadata() const noexcept;
    // Measurement-only buffers have storage but cannot be admitted for execution.
    bool uploaded() const noexcept;
    // Bind and upload within one retained lease; never allocate a replacement backend buffer.
    static std::shared_ptr<mtmd_projector_weights> allocate_in(std::shared_ptr<mtmd_projector_metadata> metadata,
        ggml_backend_memory_lease_t lease,mtmd_progress_callback progress = nullptr,void * user_data = nullptr);
    ggml_backend_memory_lease_t lease() const noexcept;

private:
    mtmd_projector_weights() = default;
    std::shared_ptr<mtmd_projector_metadata> descriptors;
    ggml_backend_buffer_ptr storage;
    bool payload_uploaded = false;
    std::shared_ptr<ggml_backend_memory_lease> grant;
    static std::shared_ptr<mtmd_projector_weights> allocate_impl(std::shared_ptr<mtmd_projector_metadata> metadata,
        ggml_backend_buffer_type_t type,bool skip_upload,mtmd_progress_callback progress,void * user_data,
        ggml_backend_memory_lease_t lease);
};

struct mtmd_projector_residency_hooks {
    std::function<bool()> drain;
    std::function<bool()> invalidate;
};

// Owner-thread lifecycle; execution pins retain the current binding until drained retirement.
class MTMD_API mtmd_projector_residency {
public:
    static std::unique_ptr<mtmd_projector_residency> create(std::shared_ptr<mtmd_projector_weights> weights,
        ggml_backend_buffer_type_t type, mtmd_projector_residency_hooks hooks);
    // Start with metadata/source only, without eager device storage.
    static std::unique_ptr<mtmd_projector_residency> create_unloaded(std::shared_ptr<mtmd_projector_metadata> metadata,
        ggml_backend_buffer_type_t type,mtmd_projector_residency_hooks hooks);
    ~mtmd_projector_residency();
    bool ready() const noexcept;
    uint64_t generation() const noexcept;
    std::shared_ptr<const mtmd_projector_weights> acquire() const noexcept;
    // Reject stale generations and reentry; end closes host submission, not device completion.
    bool begin(uint64_t generation) noexcept;
    void end() noexcept;
    // Refuse retained readers; drain and invalidate before returning the last internal weight owner.
    bool unload() noexcept;
    // Failed loading remains unbound and retryable; no implicit graph execution is admitted.
    bool reload(mtmd_progress_callback progress = nullptr, void * user_data = nullptr) noexcept;
    // Publish a loaded lease binding with its execution dependency and new generation.
    bool reload_in(ggml_backend_memory_lease_t lease,mtmd_progress_callback progress = nullptr,void * user_data = nullptr) noexcept;

private:
    struct implementation;
    explicit mtmd_projector_residency(std::unique_ptr<implementation> impl);
    std::unique_ptr<implementation> impl;
    bool reload_impl(ggml_backend_memory_lease_t lease,mtmd_progress_callback progress,void * user_data) noexcept;
};

// Internal ownership seam for lifecycle adapters and tests, not a new public mtmd C API.
MTMD_API std::shared_ptr<const mtmd_projector_weights> mtmd_acquire_projector_weights(const mtmd_context * ctx) noexcept;
MTMD_API bool mtmd_unload_projector_weights(mtmd_context * ctx) noexcept;
MTMD_API bool mtmd_reload_projector_weights(mtmd_context * ctx,
    mtmd_progress_callback progress = nullptr, void * user_data = nullptr) noexcept;
MTMD_API mtmd_context * mtmd_init_from_file_deferred(const char * path,const llama_model * model,mtmd_context_params params);
MTMD_API bool mtmd_reload_projector_weights_in(mtmd_context * ctx,ggml_backend_memory_lease_t lease,
    mtmd_progress_callback progress = nullptr,void * user_data = nullptr) noexcept;
struct clip_ctx;
std::shared_ptr<const mtmd_projector_weights> clip_acquire_projector_weights(const clip_ctx * ctx) noexcept;
bool clip_unload_projector_weights(clip_ctx * ctx) noexcept;
bool clip_reload_projector_weights(clip_ctx * ctx,mtmd_progress_callback progress,void * user_data) noexcept;
bool clip_reload_projector_weights_in(clip_ctx * ctx,ggml_backend_memory_lease_t lease,mtmd_progress_callback progress,void * user_data) noexcept;
