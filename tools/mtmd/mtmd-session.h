#pragma once

#include "mtmd-embeddings.h"
#include "mtmd-helper.h"

enum class mtmd_session_phase { text_prefill, vision_encode, embedding_prefill };
enum class mtmd_session_status { idle, ready, complete, failed, cancelled };

struct mtmd_session_step {
    mtmd_session_phase phase;
    size_t chunk;
    size_t batch;
};

// Owner-thread-only adapter. Operations finish using their host views before returning.
struct mtmd_session_backend {
    virtual ~mtmd_session_backend() = default;
    // No model work. Return 0 for accepted, 2/3 for a batch boundary, or an error.
    virtual int32_t validate_batch(const std::vector<const mtmd_input_chunk *> & entries,
        const mtmd_input_chunk * chunk) = 0;
    // Return one retained view per input chunk, in the same order.
    virtual int32_t encode(const std::vector<const mtmd_input_chunk *> & chunks,
        std::vector<mtmd_embedding_view> & outputs) = 0;
    // A null view denotes text. Keep target positions and persistent state outside the plan.
    virtual int32_t prefill(const mtmd_input_chunk * chunk, const mtmd_embedding_view * embedding) = 0;
};

// Input metadata and the prepared backend outlive the plan. Device allocations and target KV stay external.
class MTMD_API mtmd_session_plan {
public:
    mtmd_session_plan() = default;
    mtmd_session_plan(const mtmd_session_plan &) = delete;
    mtmd_session_plan & operator=(const mtmd_session_plan &) = delete;
    // Plan compatible media batches without executing either model. Speculation is not admitted yet.
    bool prepare(const std::vector<const mtmd_input_chunk *> & chunks, size_t embedding_width,
        mtmd_session_backend & backend, bool speculative_execution = false);
    // A failure closes this request; no later text/image step can run.
    int32_t advance(mtmd_session_backend & backend);
    // During a callback, cancellation takes effect after that operation completes.
    void cancel() noexcept;
    mtmd_session_status status() const noexcept;
    const std::vector<mtmd_session_step> & steps() const noexcept;
    size_t next_step() const noexcept;

private:
    std::vector<const mtmd_input_chunk *> chunks;
    std::vector<mtmd_session_step> execution;
    std::vector<std::vector<size_t>> batches;
    std::vector<mtmd_embedding_view> embeddings;
    size_t width = 0;
    size_t cursor = 0;
    mtmd_session_status state = mtmd_session_status::idle;
    bool busy = false;
    bool cancelled = false;
    mtmd_session_backend * owner = nullptr;
};

// Baseline adapter: existing target/projector allocations, ordinary position helpers, and no MTP callback.
MTMD_API int32_t mtmd_session_eval_chunks(mtmd_context * ctx, llama_context * lctx,
    const mtmd_input_chunks * chunks, llama_pos n_past, llama_seq_id seq_id, int32_t n_batch,
    bool logits_last, llama_pos * new_n_past);

// Opt-in internal adapter: vision borrows bounded target scratch and returns it before embedding prefill.
MTMD_API int32_t mtmd_session_eval_chunks_shared(mtmd_context * ctx, llama_context * lctx,
    const mtmd_input_chunks * chunks, llama_pos n_past, llama_seq_id seq_id, int32_t n_batch,
    bool logits_last, llama_pos * new_n_past);
