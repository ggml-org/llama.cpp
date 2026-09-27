#pragma once
#include "llama-kv-stream-session.h"
#include "llama-kv-stream-layer-lease.h"
class llama_kv_stream_logical_cache;


// Exact device-local grants contributed to the common text compute/KV parent.
struct llama_kv_stream_memory_requirements {
    ggml_backend_buffer_type_t buffer_type = nullptr;
    size_t pool_bytes = 0, writer_bytes = 0;
    size_t attention_prefill_bytes = 0, attention_decode_bytes = 0;
    size_t alignment = 1;
    size_t shared_device_memory_bytes = 0;
};

// Borrowed candidate leases; the model retains them only after complete session reconstruction.
struct llama_kv_stream_memory_binding {
    ggml_backend_buffer_t parent = nullptr;
    ggml_backend_memory_lease_t pool = nullptr, writer = nullptr, attention = nullptr;
    llama_memory_resource_id pool_resource = 0, writer_resource = 0, attention_resource = 0;
    llama_memory_stage_id prefill_stage = 0, decode_stage = 0;
};

struct llama_kv_stream_runtime_diagnostics {
    uint64_t layout_revision = 0;
    size_t pool_bytes = 0, writer_bytes = 0, attention_bytes = 0;
    uint32_t resident_pages_per_layer = 0, ring_slots = 0, active_pages = 0;
    size_t last_copy_bytes = 0, last_copy_calls = 0;
    double last_copy_ms = 0, last_elapsed_ms = 0;
    bool streaming_active = false;
};

struct llama_kv_stream_model_config {
    ggml_backend_t backend = nullptr;
    llama_kv_stream_host_config host;
    size_t pool_bytes = 0;
    uint32_t max_batch_rows = 0, query_heads = 0;
    bool measure = false;
    bool resume_decode = true;
    bool cross_token_prefetch = true;
    size_t shared_device_memory_bytes = 0;
    // Opt-in one-layer MTP host cache in the same physical KV policy.
    uint32_t auxiliary_cache_layers = 0;
};

// Own the host-KV execution buffer and a serial session; proxy-buffer references retain the runtime state.
class llama_kv_stream_model {
public:
    static std::unique_ptr<llama_kv_stream_model> create(const llama_kv_stream_model_config & config);
    ~llama_kv_stream_model();
    ggml_backend_buffer_t buffer() const noexcept;
    std::shared_ptr<llama_kv_stream_host> host() const noexcept;
    std::shared_ptr<llama_kv_stream_logical_cache> auxiliary_cache() const noexcept;
    // Metadata snapshot only; it does not retain the binding lease.
    llama_kv_stream_binding_view binding_view() const noexcept;
    // Borrow the retained MTP owner's occupied-ring snapshot for target appends.
    // Clear it while idle before releasing/replanning the physical MTP lease.
    bool set_ring_guard(std::shared_ptr<const llama_kv_stream_ring_guard> guard);
    // Populate the MTP layer once and retain one physical reservation for TG1-TG4.
    // Borrowed plans remain valid only until release_mtp_layer().
    bool acquire_mtp_layer(size_t future_tokens = 0);
    size_t mtp_reserved_tokens() const noexcept;
    // Copy only newly published MTP host rows into the retained physical lease.
    bool advance_mtp_layer_tail();
    // Queue encoded tail rows directly into protected spans while host publication
    // is pending; the caller keeps the returned provisional lease through use.
    bool stage_mtp_tail_async(ggml_backend_t backend, size_t first, bool value,
        const ggml_tensor * encoded, size_t row, size_t count,
        llama_kv_stream_population_stats & staged);
    llama_kv_stream_complete_layer_lease_t provisional_mtp_layer(
        uint32_t query_tokens, size_t active_tokens);
    // After writer completion, adopt staged bytes and refresh committed plans.
    bool advance_mtp_layer_tail_staged(const llama_kv_stream_population_stats & staged);
    // Retire a rejected MTP suffix while retaining its unchanged physical prefix.
    bool truncate_mtp_layer(size_t tokens);
    bool release_mtp_layer();
    ggml_kv_stream_span_plan_t mtp_layer_plan(uint32_t query_tokens) const noexcept;
    bool has_mtp_layer() const noexcept;
    llama_kv_stream_population_stats mtp_layer_population() const noexcept;
    bool begin(size_t active_tokens, uint32_t query_tokens, bool decode);
    bool complete() const noexcept;
    void abort();
    bool reset(bool clear_bytes);
    bool restore(size_t tokens);
    bool truncate(size_t tokens);
    size_t tokens() const noexcept;
    size_t granted_bytes() const noexcept;
    bool set_workspaces(const std::vector<ggml_backend_memory_lease_t> & leases);
    void release_graphs();
    size_t captured_layers() const;
    bool memory_requirements(llama_kv_stream_memory_requirements & output) const noexcept;
    // Suspend private grants before parent allocation; attach validates all regions before publication.
    bool prepare_shared_memory();
    bool resume_private_memory();
    bool attach_shared_memory(const llama_kv_stream_memory_binding & binding);
    // The shared owner calls detach only after destroying its coordinator and workspace consumer.
    bool detach_shared_memory() noexcept;
    // Borrow only during the serial MTP phase.
    ggml_backend_buffer_t mtp_attention_workspace() const noexcept;
    ggml_backend_buffer_t mtp_writer_workspace() const noexcept;
    llama_memory_consumer * memory_consumer() noexcept;
    bool uses_shared_memory() const noexcept;
    ggml_backend_buffer_t shared_parent() const noexcept;
    size_t device_grant_bytes() const noexcept;
    size_t pool_grant_bytes() const noexcept;
    size_t writer_grant_bytes() const noexcept;
    size_t attention_grant_bytes() const noexcept;
    bool runtime_diagnostics(llama_kv_stream_runtime_diagnostics & output) const noexcept;
private:
    llama_kv_stream_model() = default;
    struct implementation;
    std::shared_ptr<implementation> impl;
    ggml_backend_buffer_t proxy = nullptr;
};
