#pragma once

#include "llama-kv-stream-binding.h"
#include "llama-kv-stream-content.h"
#include "llama-kv-stream-publication.h"

struct llama_kv_stream_write_stats {
    size_t graph_submissions = 0, d2h_bytes = 0, d2h_calls = 0, d2d_bytes = 0, d2d_calls = 0;
    size_t device_scratch_bytes = 0, host_payload_bytes = 0, tile_rows = 0;
};

struct llama_kv_stream_prefetch_stats {
    size_t pending_pages = 0, peak_pages = 0, max_layer_distance = 0;
    size_t copy_bytes = 0, copy_calls = 0;
    size_t feedback_stride = 1, feedback_uploads = 0;
    size_t ready_pages = 0;
    bool demand_ready = false, primed = false, adopted = false;
};
struct ggml_kv_stream_copy_feedback;
struct llama_kv_stream_feedback_context {
    uint32_t query_tokens = 0;
    bool decode = false;
};
struct llama_kv_stream_capture_stamp {
    ggml_backend_buffer_t buffer = nullptr;
    uint64_t binding_revision = 0, residency_revision = 0, mirror_epoch = 0;
    size_t padded_tokens = 0;
};

// Idle native resources owned by the coarse binding; backend and external graph contexts must outlive their users.
// Hold a binding execution pin for the entire lifetime of any graph referencing these planes, including captures.
class llama_kv_stream_resident : public llama_memory_executable {
public:
    ~llama_kv_stream_resident() override;
    // Optional validated placement selects an idle layout; no live repartition or policy publication occurs here.
    static std::unique_ptr<llama_kv_stream_resident> create(const llama_kv_stream_binding_view & binding,
            std::shared_ptr<llama_kv_stream_content> content, ggml_backend_t backend,
            const llama_kv_stream_policy_state * placement = nullptr);

    // Drain prior backend work, then upload dirty rows into the fixed policy-derived resident planes.
    // Reject contexts that exceed resident capacity; this refresh copies encoded bytes without conversion or repartition.
    bool synchronize(size_t active_tokens);
    bool ready(size_t active_tokens) const noexcept;
    // Capture identity excludes ordinary data writes and live-length changes within one padded page extent.
    bool capture_state(ggml_backend_t backend, size_t active_tokens, llama_kv_stream_capture_stamp & output) const;
    bool capture_attention(const ggml_tensor * attention, size_t active_tokens) const;
    // Ordered one-block path. Hold a binding pin; Q/mask/output and the disjoint workspace lease remain live until return.
    // Caller supplies padded causal mask values. Query backend capabilities before planning native or converted K/V.
    bool compute_one_block(uint32_t layer, ggml_tensor * q, ggml_tensor * mask, ggml_tensor * output,
            size_t active_tokens, float scale, ggml_backend_memory_lease_t workspace);
    // Traversal over any number of tail blocks; optional overlap retains the ordered correctness control.
    // Ordered sequences may return after queueing an intermediate layer; the final layer drains all retained work.
    // Non-sequence calls complete before return. Host backing must remain pinned through the sequence.
    // A positive span ceiling groups physical neighbors without increasing ring or conversion storage.
    bool compute_streamed(uint32_t layer, ggml_tensor * q, ggml_tensor * mask, ggml_tensor * output,
            size_t active_tokens, float scale, ggml_backend_memory_lease_t workspace, bool overlap = false,
            size_t span_pages = 1, bool decode_feedback = false);
    size_t last_upload_bytes() const noexcept;
    size_t last_upload_calls() const noexcept;
    // Successful partial-attention submissions in the last started streamed execution.
    size_t last_attention_calls() const noexcept;
    // Hold the binding pin until completion/cancel. Only sequence-tail publication APIs may mutate host content meanwhile.
    // Default stable_tokens declares a fully ready snapshot; online producers must pass the immutable prefix explicitly.
    // Declare both decode intent and query count before prefetch starts. Unknown phase and single-token prompts stay unprofiled.
    bool begin_sequence(const std::vector<uint32_t> & layers, size_t active_tokens, size_t span_pages = 1,
            size_t stable_tokens = SIZE_MAX, llama_kv_stream_feedback_context feedback = {});
    // Speculatively start the next append while idle; adoption validates the complete logical identity.
    // Both operations are backend-neutral and retain the existing opaque copy queue until adoption or cancellation.
    bool prime_sequence(const std::vector<uint32_t> & layers, size_t active_tokens, size_t span_pages,
            size_t stable_tokens, llama_kv_stream_feedback_context feedback);
    bool adopt_sequence(const std::vector<uint32_t> & layers, size_t active_tokens, size_t span_pages,
            size_t stable_tokens, llama_kv_stream_feedback_context feedback);
    bool publish_sequence_tail(const std::vector<llama_kv_stream_write_span> & spans);
    // An invalid or failed layer call cancels outstanding prefetch; the final valid layer ends the sequence.
    void cancel_sequence();
    bool sequence_active() const noexcept;
    // Readiness is an observation, not a storage-lifetime fence; distance is measured in the supplied layer order.
    llama_kv_stream_prefetch_stats sequence_stats() const noexcept;

    // Opt-in diagnostics and span trials. Change only while idle; no KV storage is resized here.
    bool configure_feedback(bool enable, size_t bounded_span_pages = 32);
    // Graph callers reserve native FA extras for their output; low-level minimal-workspace callers leave this off.
    bool configure_native_graph_attention(bool enable);
    // Only explicitly marked decode sequences may use the optional native resumable contract.
    bool configure_resumed_decode(bool enable);
    size_t suggested_span_pages() const noexcept;
    // Poll completed snapshots without waiting; the latest successful run may still be pending.
    llama_kv_stream_feedback feedback() const noexcept;
    ggml_kv_stream_copy_feedback copy_feedback() const noexcept;
    // A successful owner-session commit may advance authoritative content identity without discarding timing history.
    bool advance_feedback_identity();
    // Read-only proposal. Decode feedback requires explicit intent; accept device layout before publishing decision.next.
    bool recommend_policy(const llama_kv_stream_policy_state & previous, size_t active_tokens,
            uint32_t query_tokens, llama_kv_stream_policy_decision & decision, bool decode_feedback = false, bool uniform_prefill = false) const;

    // Configure a physical-batch ceiling. Default scratch borrows the idle ring; external scratch must be a disjoint lease.
    bool configure_writes(size_t max_batch_rows, ggml_backend_memory_lease_t workspace = nullptr);
    // Publish both completed GPU producer planes atomically while historical prefetch remains live.
    // Requires disjoint leased writer workspace configured before begin_sequence().
    // Rows must cover [stable_tokens, active_tokens). Submission failure cancels prefetch but preserves host bytes.
    bool write_sequence_rows(uint32_t layer, size_t first_row, const ggml_tensor * k, const ggml_tensor * v);
    bool prepare_write_pair(uint32_t layer, size_t first_row, const ggml_tensor * k, const ggml_tensor * v,
            llama_kv_stream_publication_ticket & ticket, uint32_t pair,
            std::unique_ptr<llama_kv_stream_publication_pair> & output);
    // After all host completions are ready, atomically publish every layer and close the sequence.
    bool commit_sequence_writes();


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
