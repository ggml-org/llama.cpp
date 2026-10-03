#include "mtmd-session.h"
#include "mtmd-workspace.h"
#include "mtmd-projector-storage.h"
#include "../../src/llama-context-memory.h"

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
    bool shared = false;
    bool arena = false;

    session_adapter(mtmd_context * ctx, llama_context * lctx, llama_pos position, llama_seq_id sequence,
            int32_t batch_size, bool logits_last, size_t n_chunks, bool shared, bool arena) : ctx(ctx), lctx(lctx), position(position),
        sequence(sequence), batch_size(batch_size), logits_last(logits_last), n_chunks(n_chunks), shared(shared),arena(arena) {}

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
        if (arena) {
            const auto result = mtmd_batch_encode_arena(ctx,batch.get(),lctx);
            if (result) return result;
            for (const auto * chunk : chunks) {
                mtmd_embedding_view view;
                if (!mtmd_batch_acquire_output_embd(batch.get(),chunk,view)) return -1;
                outputs.push_back(std::move(view));
            }
            return 0;
        }
        struct workspace_return {
            mtmd_context * ctx = nullptr;
            llama_context * target = nullptr;
            // Release vision addresses before rebuilding target graphs, including failure exits.
            bool finish() noexcept {
                auto * vision = ctx; ctx = nullptr;
                auto * text = target; target = nullptr;
                try {
                    if (vision && !mtmd_release_compute_workspace(vision)) return false;
                    return !text || llama_context_resume_kv_device(text,llama_memory_text_phase::prefill);
                } catch (...) { return false; }
            }
            ~workspace_return() { finish(); }
        } returned;
        if (shared) {
            auto * parent = llama_context_compute_memory(lctx);
            auto * buffer = parent ? parent->shared_parent() : nullptr;
            if (!buffer || !parent->valid() || parent->kv_device_suspended()) return -1;
            std::vector<ggml_backend_memory_workspace_group> groups;
            if (!mtmd_batch_measure_compute_workspace(batch.get(), groups, ggml_backend_buffer_get_type(buffer))) return -1;
            for (const auto & group : groups)
                if (group.buft == ggml_backend_buffer_get_type(buffer) && group.size > ggml_backend_buffer_get_size(buffer)) return -1;
            if (parent->shares_kv_memory()) {
                if (!llama_context_suspend_kv_device(lctx)) return -1;
                returned.target = lctx;
            }
            returned.ctx = ctx;
            if (!mtmd_borrow_compute_workspace(ctx, *parent)) return -1;
        }
        const auto result = mtmd_batch_encode(batch.get());
        if (result) return result;
        for (const auto * chunk : chunks) {
            mtmd_embedding_view view;
            if (!mtmd_batch_acquire_output_embd(batch.get(), chunk, view)) return -1;
            outputs.push_back(std::move(view));
        }
        if (!returned.finish()) return -1;
        return 0;
    }

    // Keep ordinary text batching, M-RoPE and non-causal setup; no draft-model callback runs here.
    int32_t prefill(const mtmd_input_chunk * chunk, const mtmd_embedding_view * view) override {
        const auto positions = mtmd_input_chunk_get_n_pos(chunk);
        if (positions < 0 || positions > std::numeric_limits<llama_pos>::max() - position) return -1;
        llama_pos next = position;
        llama_set_kv_stream_decode(lctx, false);
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

// Continue the supplied prefix; shared encoding returns all target KV grants until host outputs are retained.
static int32_t session_eval_chunks(mtmd_context * ctx, llama_context * lctx, const mtmd_input_chunks * input,
        llama_pos n_past, llama_seq_id seq_id, int32_t n_batch, bool logits_last, llama_pos * new_n_past, bool shared,bool arena = false) {
    if (!ctx || !lctx || !input || !new_n_past || n_past < 0 || seq_id < 0 || n_batch <= 0 ||
            size_t(n_batch) > llama_n_batch(lctx)) return -1;
    try {
        const auto width = llama_model_n_embd_inp(llama_get_model(lctx));
        if (width <= 0) return -1;
        std::vector<const mtmd_input_chunk *> chunks;
        for (size_t i = 0; i < mtmd_input_chunks_size(input); ++i) chunks.push_back(mtmd_input_chunks_get(input, i));
        session_adapter backend(ctx, lctx, n_past, seq_id, n_batch, logits_last, chunks.size(), shared,arena);
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

int32_t mtmd_session_eval_chunks(mtmd_context * ctx, llama_context * lctx, const mtmd_input_chunks * input,
        llama_pos n_past, llama_seq_id seq_id, int32_t n_batch, bool logits_last, llama_pos * new_n_past) {
    return session_eval_chunks(ctx, lctx, input, n_past, seq_id, n_batch, logits_last, new_n_past, false);
}

int32_t mtmd_session_eval_chunks_shared(mtmd_context * ctx, llama_context * lctx, const mtmd_input_chunks * input,
        llama_pos n_past, llama_seq_id seq_id, int32_t n_batch, bool logits_last, llama_pos * new_n_past) {
    return session_eval_chunks(ctx, lctx, input, n_past, seq_id, n_batch, logits_last, new_n_past, true);
}

int32_t mtmd_session_eval_chunks_arena(mtmd_context * ctx,llama_context * lctx,const mtmd_input_chunks * chunks,
        llama_pos n_past,llama_seq_id seq_id,int32_t n_batch,bool logits_last,llama_pos * new_n_past) {
    return session_eval_chunks(ctx,lctx,chunks,n_past,seq_id,n_batch,logits_last,new_n_past,false,true);
}

// The parent lease counts remain authoritative even if a reader outlives the projector or this request.
int32_t mtmd_batch_encode_arena(mtmd_context * ctx,mtmd_batch * batch,llama_context * target,mtmd_progress_callback progress,void * user_data) {
    if (!ctx || !batch || !target || mtmd_batch_context(batch) != ctx) return -1;
    auto * owner = llama_context_compute_memory(target);
    auto * parent = owner ? owner->shared_parent() : nullptr;
    if (!owner || !owner->valid() || owner->kv_device_suspended() || !owner->shares_kv_memory() ||
            owner->has_speculative_consumer() || !parent) return -1;
    struct phase_return {
        mtmd_context * ctx;
        llama_context * target;
        std::vector<ggml_backend_memory_lease_t> grants;
        std::vector<std::unique_ptr<ggml_backend_memory_arena,decltype(&ggml_backend_memory_arena_free)>> host;
        bool suspended = false, finished = false, armed = false;
        phase_return(mtmd_context * ctx,llama_context * target) : ctx(ctx),target(target) {}
        bool finish() noexcept {
            if (finished) return true;
            finished = true;
            if (!armed) return true;
            bool released = false;
            try { released = mtmd_release_compute_workspace(ctx) && mtmd_unload_projector_weights(ctx); } catch (...) {}
            for (auto * lease : grants) ggml_backend_memory_lease_free(lease);
            grants.clear(); host.clear();
            return released && (!suspended || llama_context_resume_kv_device(target,llama_memory_text_phase::prefill));
        }
        ~phase_return() { finish(); }
    } returned{ctx,target};
    try {
        if (!mtmd_unload_projector_weights(ctx)) return -1;
        returned.armed = true;
        mtmd_vision_phase_requirements plan;
        if (!mtmd_batch_measure_vision_phase(batch,parent,plan)) return -1;
        if (!llama_context_suspend_kv_device(target)) return -1;
        returned.suspended = true;
        if (!owner->lend_suspended({plan.weight_bytes,plan.device_compute_bytes},returned.grants)) return -1;
        if (!mtmd_reload_projector_weights_in(ctx,returned.grants[0],progress,user_data)) return -1;
        std::vector<ggml_backend_memory_lease_t> compute;
        for (size_t i = 0; i < plan.groups.size(); ++i) {
            const auto & group = plan.groups[i];
            if (group.buft == ggml_backend_buffer_get_type(parent)) compute.push_back(returned.grants[1]);
            else {
                returned.host.emplace_back(ggml_backend_memory_arena_new(group.buft,group.size),ggml_backend_memory_arena_free);
                auto * arena = returned.host.back().get();
                if (!arena || !ggml_backend_memory_arena_begin(arena,0) ||
                        !ggml_backend_memory_arena_reserve(arena,i+1,group.size,group.alignment,0,nullptr) ||
                        !ggml_backend_memory_arena_commit(arena)) return -1;
                auto * lease = ggml_backend_memory_arena_acquire(arena,i+1);
                if (!lease) return -1;
                returned.grants.push_back(lease); compute.push_back(lease);
            }
        }
        if (!mtmd_attach_compute_workspace(ctx,compute)) return -1;
        const auto result = mtmd_batch_encode(batch);
        if (result) return result;
        return returned.finish() ? 0 : -1;
    } catch (...) { return -1; }
}
