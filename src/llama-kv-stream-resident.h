#pragma once

#include "llama-kv-stream-binding.h"
#include "llama-kv-stream-content.h"

struct llama_kv_stream_write_stats {
    size_t graph_submissions = 0, d2h_bytes = 0, d2h_calls = 0, d2d_bytes = 0, d2d_calls = 0;
    size_t device_scratch_bytes = 0, host_payload_bytes = 0, tile_rows = 0;
};

// Idle native resources owned by the coarse binding; backend and external graph contexts must outlive their users.
// Hold a binding execution pin for the entire lifetime of any graph referencing these planes, including captures.
class llama_kv_stream_resident : public llama_memory_executable {
public:
    ~llama_kv_stream_resident() override;
    static std::unique_ptr<llama_kv_stream_resident> create(const llama_kv_stream_binding_view & binding,
            std::shared_ptr<llama_kv_stream_content> content, ggml_backend_t backend);

    // Drain prior backend work, then upload dirty rows into the fixed policy-derived resident planes.
    // Reject contexts that need streaming; no layout adaptation or conversion fallback is enabled here.
    bool synchronize(size_t active_tokens);
    bool ready(size_t active_tokens) const noexcept;
    size_t last_upload_bytes() const noexcept;
    size_t last_upload_calls() const noexcept;

    // Configure a physical-batch ceiling. Quantization scratch and indices borrow the unused ring region.
    bool configure_writes(size_t max_batch_rows);
    // Source is a completed, dense F32 [head_dim * heads, rows] device tensor. Rows are consecutive.
    // Hold a binding pin and the source owner until return; call synchronize() before attention.
    bool write_rows(uint32_t layer, ggml_kv_stream_operand operand, size_t first_row, const ggml_tensor * source);
    llama_kv_stream_write_stats last_write_stats() const noexcept;
    // Retire the one cached writer graph and release its retained source buffer before reclaiming workspace.
    void release_write_workspace();

    // Build an ordinary FLASH_ATTN_EXT node over leased views, without allocating K/V device buffers.
    // A supplied mask must hide padded/future keys. Without a mask, active_tokens must need no padding.
    // The caller allocates Q/mask/output and gates execution on ready(); invalid/unsupported metadata returns null.
    ggml_tensor * attention(ggml_context * context, uint32_t layer, ggml_tensor * q, ggml_tensor * mask,
            size_t active_tokens, float scale);

private:
    llama_kv_stream_resident() = default;
    struct implementation;
    std::unique_ptr<implementation> impl;
};
