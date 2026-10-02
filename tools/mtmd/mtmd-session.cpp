#include "mtmd-session.h"

#include <limits>
#include <utility>

namespace {
struct session_gate {
    bool & busy;
    explicit session_gate(bool & busy) : busy(busy) { busy = true; }
    ~session_gate() { busy = false; }
};
}

// Build all compatibility decisions before submitting any target or projector work.
bool mtmd_session_plan::prepare(const std::vector<const mtmd_input_chunk *> & input, size_t embedding_width,
        mtmd_session_backend & backend, bool speculative_execution) {
    if (busy || state == mtmd_session_status::ready || state == mtmd_session_status::failed || speculative_execution) return false;
    session_gate gate(busy);
    cancelled = false;
    try {
        auto next_chunks = input;
        std::vector<size_t> media;
        for (size_t i = 0; i < input.size(); ++i) {
            if (!input[i]) return false;
            const auto type = mtmd_input_chunk_get_type(input[i]);
            if (type == MTMD_INPUT_CHUNK_TYPE_TEXT) continue;
            const size_t rows = mtmd_input_chunk_get_n_tokens(input[i]);
            if (type != MTMD_INPUT_CHUNK_TYPE_IMAGE || !embedding_width || !rows ||
                    rows > SIZE_MAX / sizeof(float) / embedding_width) return false;
            media.push_back(i);
        }
        std::vector<std::vector<size_t>> next_batches;
        std::vector<size_t> batch_for_chunk(input.size(), SIZE_MAX);
        for (size_t first = 0; first < media.size();) {
            std::vector<const mtmd_input_chunk *> members;
            std::vector<size_t> indices;
            size_t end = first;
            size_t rows = 0;
            for (; end < media.size(); ++end) {
                const size_t index = media[end];
                const int32_t result = backend.validate_batch(members, input[index]);
                if (cancelled) { state = mtmd_session_status::cancelled; embeddings.clear(); return false; }
                if (result != 0) {
                    if (members.empty() || (result != 2 && result != 3)) return false;
                    break;
                }
                const size_t added = mtmd_input_chunk_get_n_tokens(input[index]);
                if (added > SIZE_MAX / sizeof(float) / embedding_width - rows) return false;
                rows += added;
                members.push_back(input[index]);
                indices.push_back(index);
                batch_for_chunk[index] = next_batches.size();
            }
            next_batches.push_back(std::move(indices));
            first = end;
        }
        std::vector<mtmd_session_step> next_steps;
        for (size_t i = 0; i < input.size(); ++i) {
            const size_t group = batch_for_chunk[i];
            if (group == SIZE_MAX) next_steps.push_back({mtmd_session_phase::text_prefill, i, SIZE_MAX});
            else {
                if (next_batches[group].front() == i) next_steps.push_back({mtmd_session_phase::vision_encode, i, group});
                next_steps.push_back({mtmd_session_phase::embedding_prefill, i, group});
            }
        }
        std::vector<mtmd_embedding_view> next_embeddings(input.size());
        chunks = std::move(next_chunks);
        batches = std::move(next_batches);
        execution = std::move(next_steps);
        embeddings = std::move(next_embeddings);
        width = embedding_width;
        cursor = 0;
        owner = &backend;
        state = execution.empty() ? mtmd_session_status::complete : mtmd_session_status::ready;
        return true;
    } catch (...) { return false; }
}

// A successful encode can run ahead, but target prefill never skips an earlier prompt chunk.
int32_t mtmd_session_plan::advance(mtmd_session_backend & backend) {
    if (busy) return -1;
    if (state == mtmd_session_status::complete) return 0;
    if (state != mtmd_session_status::ready) return -1;
    if (&backend != owner) return -1;
    session_gate gate(busy);
    auto fail = [&](int32_t result) {
        embeddings.clear();
        state = cancelled ? mtmd_session_status::cancelled : mtmd_session_status::failed;
        return result ? result : -1;
    };
    try {
        const auto & step = execution[cursor];
        int32_t result = 0;
        if (step.phase == mtmd_session_phase::vision_encode) {
            std::vector<const mtmd_input_chunk *> members;
            for (size_t index : batches[step.batch]) members.push_back(chunks[index]);
            std::vector<mtmd_embedding_view> outputs;
            result = backend.encode(members, outputs);
            if (result || cancelled) return fail(result);
            if (outputs.size() != members.size()) return fail(-1);
            for (size_t i = 0; i < outputs.size(); ++i) {
                if (!outputs[i].data() || outputs[i].n_embd() != width ||
                        outputs[i].n_tokens() != mtmd_input_chunk_get_n_tokens(members[i])) return fail(-1);
            }
            for (size_t i = 0; i < outputs.size(); ++i) embeddings[batches[step.batch][i]] = std::move(outputs[i]);
        } else {
            const auto * view = step.phase == mtmd_session_phase::embedding_prefill ? &embeddings[step.chunk] : nullptr;
            if (view && !view->data()) return fail(-1);
            result = backend.prefill(chunks[step.chunk], view);
            if (result || cancelled) return fail(result);
            embeddings[step.chunk] = {};
        }
        ++cursor;
        if (cursor == execution.size()) { state = mtmd_session_status::complete; embeddings.clear(); }
        return 0;
    } catch (...) { return fail(-1); }
}

// Defer view destruction until the active callback has finished using its inputs.
void mtmd_session_plan::cancel() noexcept {
    if (!busy && state == mtmd_session_status::complete) return;
    cancelled = true;
    if (busy) return;
    embeddings.clear();
    if (state != mtmd_session_status::failed) state = mtmd_session_status::cancelled;
}
mtmd_session_status mtmd_session_plan::status() const noexcept { return state; }
const std::vector<mtmd_session_step> & mtmd_session_plan::steps() const noexcept { return execution; }
size_t mtmd_session_plan::next_step() const noexcept { return cursor; }

namespace {
struct session_adapter : mtmd_session_backend {
    mtmd_context * ctx;
    llama_context * lctx;
    llama_pos position;
    llama_seq_id sequence;
    int32_t batch_size;
    bool logits_last;
    size_t n_chunks;
    size_t prefills = 0;

    session_adapter(mtmd_context * ctx, llama_context * lctx, llama_pos position, llama_seq_id sequence,
            int32_t batch_size, bool logits_last, size_t n_chunks) : ctx(ctx), lctx(lctx), position(position),
        sequence(sequence), batch_size(batch_size), logits_last(logits_last), n_chunks(n_chunks) {}

    // Reuse the real context's capability and batch-size checks without allocating graph storage.
    int32_t validate_batch(const std::vector<const mtmd_input_chunk *> & entries,
            const mtmd_input_chunk * chunk) override {
        mtmd::batch_ptr batch(mtmd_batch_init(ctx));
        if (!batch) return -1;
        for (const auto * entry : entries) {
            const auto result = mtmd_batch_add_chunk(batch.get(), entry);
            if (result) return result;
        }
        return mtmd_batch_add_chunk(batch.get(), chunk);
    }

    // The transient batch can disappear immediately after its host outputs are retained.
    int32_t encode(const std::vector<const mtmd_input_chunk *> & chunks,
            std::vector<mtmd_embedding_view> & outputs) override {
        mtmd::batch_ptr batch(mtmd_batch_init(ctx));
        if (!batch) return -1;
        for (const auto * chunk : chunks) {
            const auto result = mtmd_batch_add_chunk(batch.get(), chunk);
            if (result) return result;
        }
        const auto result = mtmd_batch_encode(batch.get());
        if (result) return result;
        for (const auto * chunk : chunks) {
            mtmd_embedding_view view;
            if (!mtmd_batch_acquire_output_embd(batch.get(), chunk, view)) return -1;
            outputs.push_back(std::move(view));
        }
        return 0;
    }

    // Keep ordinary text batching, M-RoPE and non-causal setup; no draft-model callback runs here.
    int32_t prefill(const mtmd_input_chunk * chunk, const mtmd_embedding_view * view) override {
        const auto positions = mtmd_input_chunk_get_n_pos(chunk);
        if (positions < 0 || positions > std::numeric_limits<llama_pos>::max() - position) return -1;
        llama_pos next = position;
        const auto result = view ? mtmd_helper_decode_image_chunk(ctx, lctx, chunk,
            const_cast<float *>(view->data()), position, sequence, batch_size, &next, nullptr, nullptr) :
            mtmd_helper_eval_chunk_single(ctx, lctx, chunk, position, sequence, batch_size,
                logits_last && prefills + 1 == n_chunks, &next);
        position = next;
        if (!result) ++prefills;
        return result;
    }
};
}

// No workspace sharing or target-state reset: the supplied prefix continues through this ordered plan.
int32_t mtmd_session_eval_chunks(mtmd_context * ctx, llama_context * lctx, const mtmd_input_chunks * input,
        llama_pos n_past, llama_seq_id seq_id, int32_t n_batch, bool logits_last, llama_pos * new_n_past) {
    if (!ctx || !lctx || !input || !new_n_past || n_past < 0 || seq_id < 0 || n_batch <= 0 ||
            size_t(n_batch) > llama_n_batch(lctx)) return -1;
    try {
        const auto width = llama_model_n_embd_inp(llama_get_model(lctx));
        if (width <= 0) return -1;
        std::vector<const mtmd_input_chunk *> chunks;
        for (size_t i = 0; i < mtmd_input_chunks_size(input); ++i) chunks.push_back(mtmd_input_chunks_get(input, i));
        session_adapter backend(ctx, lctx, n_past, seq_id, n_batch, logits_last, chunks.size());
        mtmd_session_plan plan;
        if (!plan.prepare(chunks, size_t(width), backend)) return -1;
        *new_n_past = n_past;
        while (plan.status() == mtmd_session_status::ready) {
            const auto result = plan.advance(backend);
            *new_n_past = backend.position;
            if (result) return result;
        }
        return 0;
    } catch (...) { return -1; }
}
