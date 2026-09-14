#include "llama-kv-stream-resident.h"
#include "ggml-cpp.h"
#include "llama-kv-stream-writer.h"
#include "../ggml/src/ggml-kv-stream-device.h"
#include "../ggml/src/ggml-kv-stream-copy.h"
#include "llama-kv-stream-prefetch.h"
#include "llama-kv-stream-feedback.h"
#include "../ggml/src/ggml-cuda-graph.h"

#include <cmath>
#include <chrono>
#include <atomic>
#include <cstring>
#include <limits>
#include <utility>

// Binding recreation must not make a new counter history look like an earlier runtime's history.
static std::atomic<uint64_t> next_feedback_epoch{1};

struct llama_kv_stream_resident::implementation {
    llama_kv_stream_binding_view binding;
    std::shared_ptr<llama_kv_stream_content> content;
    ggml_backend_t backend = nullptr;
    ggml_context_ptr context;
    llama_kv_stream_policy_layout layout;
    std::vector<std::pair<ggml_tensor *, ggml_tensor *>> roots;
    ggml_tensor * stage[2] = {};
    ggml_kv_stream_layout page, conversion;
    ggml_tensor * converted[2] = {};
    bool fallback = false;
    bool native_graph_attention = false;
    bool resumed_decode = false;
    std::vector<llama_kv_stream_rows> ranges;
    std::vector<llama_kv_stream_rows> layer_ranges{{0,ggml_kv_stream_operand::k,0,0},{0,ggml_kv_stream_operand::v,0,0}};
    size_t active = 0, bytes = 0, calls = 0;
    size_t attention_calls = 0;
    uint64_t generation = 0, epoch = 0;
    bool initialized = false, valid = false, busy = false, poisoned = false;
    uint64_t residency_revision = 1;
    ggml_backend_cuda_graph_is_capturing_t query_capture = nullptr;
    // Cached writer graphs retire before their leased workspace is released.
    std::unique_ptr<ggml_backend_memory_lease,decltype(&ggml_backend_memory_lease_free)>
        writer_lease{nullptr,ggml_backend_memory_lease_free};
    uintptr_t writer_base = 0;
    size_t writer_bytes = 0;
    std::unique_ptr<llama_kv_stream_writer> writer;
    llama_kv_stream_write_stats write_stats;
    const ggml_kv_stream_copy_ops * copy_ops = nullptr;
    ggml_backend_buffer_t copy_host = nullptr;
    struct prefetch_sequence {
        llama_kv_stream_prefetch_plan plan;
        std::vector<uint32_t> layers;
        std::vector<uint8_t> resident_dirty;
        size_t active = 0, padded = 0, span = 0, stable = 0, next = 0;
        uint64_t generation = 0, epoch = 0;
        bool started = false, feedback_valid = true;
        uint32_t queries = 0;
        bool profile = false;
        bool decode = false;
        llama_kv_stream_prefetch_stats stats;
    };
    std::unique_ptr<prefetch_sequence> sequence;
    llama_kv_stream_prefetch_stats finished_sequence;
    bool measuring = false;
    size_t bounded_span = 32, feedback_pages = 0;
    uint32_t feedback_queries = 0;
    uint64_t feedback_generation = 0, feedback_epoch = 0;
    std::vector<uint32_t> feedback_layers;
    llama_kv_stream_feedback_window window;
    llama_kv_stream_span_tuner tuner;
    ggml_kv_stream_copy_feedback last_feedback;
    struct pending_feedback {
        uint64_t id = 0, epoch = 0;
        size_t span = 0;
        uint32_t queries = 0;
        double elapsed_ms = 0;
    };
    std::array<pending_feedback,ggml_kv_stream_feedback_slots::capacity> feedback_pending{};
    std::chrono::steady_clock::time_point run_started;
    // Destroy the copy queue before tensor metadata and authoritative host storage.
    std::unique_ptr<void,void(*)(void*)> copies{nullptr,nullptr};

    bool capturing() const { return query_capture && query_capture(backend); }
    // A return to the same extent after streaming must not resurrect a previous resident capture.
    bool enter_streamed() {
        if (residency_revision == UINT64_MAX) return false;
        ++residency_revision; return true;
    }

    // Only future layers' mutable suffixes may change while historical DMA remains in flight.
    bool tail_valid(const std::vector<llama_kv_stream_write_span> & spans, bool generated) const {
        const auto & seq = *sequence;
        for (const auto & span : spans) {
            const auto found = std::find(seq.layers.begin()+seq.next,seq.layers.end(),span.layer);
            if (found == seq.layers.end() || (generated ? span.data != nullptr : span.data == nullptr) || !span.bytes ||
                    (span.operand != ggml_kv_stream_operand::k && span.operand != ggml_kv_stream_operand::v)) return false;
            const size_t stride = span.operand == ggml_kv_stream_operand::k ? page.k_token_bytes : page.v_token_bytes;
            if (span.offset%stride || span.bytes%stride || span.offset/stride < seq.stable ||
                    span.offset/stride > seq.active || span.bytes/stride > seq.active-span.offset/stride) return false;
        }
        return true;
    }

    // Publish the new content identity only after the complete K/V transaction succeeds.
    void tail_committed(const std::vector<llama_kv_stream_write_span> & spans) {
        auto & seq = *sequence;
        const bool tracked = feedback_generation == seq.generation && feedback_epoch == seq.epoch;
        for (const auto & span : spans) {
            const size_t index = size_t(std::find(seq.layers.begin()+seq.next,seq.layers.end(),span.layer)-seq.layers.begin());
            const size_t stride = span.operand == ggml_kv_stream_operand::k ? page.k_token_bytes : page.v_token_bytes;
            if (span.offset/stride < layout.layers[span.layer].planes.tokens) seq.resident_dirty[index] = true;
        }
        seq.generation = content->generation(); seq.epoch = content->mirror_epoch(); valid = false;
        if (tracked) { feedback_generation = seq.generation; feedback_epoch = seq.epoch; }
    }

    // Discard both counter continuity and timing trials after unknown content or execution changes.
    void reset_feedback() {
        window.reset(next_feedback_epoch.fetch_add(1,std::memory_order_relaxed)); tuner.reset(); last_feedback = {};
        feedback_pending = {};
        feedback_generation = feedback_epoch = 0;
    }
    // Poll without waiting; an old or cancelled snapshot can free storage but cannot train a new epoch.
    void collect_feedback(bool accept = true) {
        if (!measuring || !copies || copy_ops->version < 5) return;
        ggml_kv_stream_copy_snapshot snapshot;
        while (copy_ops->poll_feedback(copies.get(),&snapshot)) {
            auto found = std::find_if(feedback_pending.begin(),feedback_pending.end(),
                [&](const auto & entry) { return entry.id == snapshot.id; });
            if (found == feedback_pending.end()) continue;
            const auto entry = *found; *found = {};
            if (!accept || entry.epoch != window.snapshot().epoch) continue;
            if (!window.add(snapshot.value)) { reset_feedback(); continue; }
            last_feedback = snapshot.value;
            const size_t slots = binding.initial_policy.ring_slots, bounded = std::min(slots,bounded_span);
            if (entry.queries == 1 && bounded < slots && (entry.span == bounded || entry.span >= slots))
                tuner.observe(entry.elapsed_ms,entry.span == bounded);
        }
    }
    void prepare_feedback(size_t tokens, uint32_t queries, const std::vector<uint32_t> & layers) {
        if (!measuring) return;
        const size_t pages = (tokens-1)/page.tokens+1;
        if (feedback_generation != content->generation() || feedback_epoch != content->mirror_epoch() ||
                pages != feedback_pages || queries != feedback_queries || layers != feedback_layers) reset_feedback();
        feedback_pages = pages; feedback_queries = queries; feedback_layers = layers;
        feedback_generation = content->generation(); feedback_epoch = content->mirror_epoch();
        collect_feedback();
    }
    // Only successful complete executions train the policy or end-to-end span trials.
    void record_feedback(size_t span) {
        if (!measuring) return;
        const uint64_t id = copy_ops->feedback_id(copies.get());
        if (id) {
            const auto found = std::find_if(feedback_pending.begin(),feedback_pending.end(),[](const auto & entry) { return !entry.id; });
            // The backend also has two slots; never replace metadata for an uncollected result.
            GGML_ASSERT(found != feedback_pending.end());
            *found = {id,window.snapshot().epoch,span,feedback_queries,
                std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-run_started).count()};
        }
        feedback_generation = content->generation(); feedback_epoch = content->mirror_epoch();
        collect_feedback();
    }

    // Reuse event resources only while their retained host allocation still matches the cache.
    bool ensure_copies(size_t slots) {
        if (copies && copy_host != content->host()->buffer()) {
            if (measuring) reset_feedback();
            copies.reset(); copy_host = nullptr;
        }
        if (copies) return true;
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
        auto get = reinterpret_cast<ggml_kv_stream_copy_ops_get>(ggml_backend_reg_get_proc_address(reg,"ggml_backend_kv_stream_copy_ops"));
        copy_ops = get ? get() : nullptr;
        if (!copy_ops || copy_ops->version < 2 || !copy_ops->enqueue_span || !copy_ops->stats) return false;
        copies = decltype(copies)(copy_ops->create(backend,binding.buffer,content->host()->buffer(),binding.config.shape,slots),copy_ops->free);
        if (!copies) return false;
        if (measuring && (copy_ops->version < 7 || !copy_ops->measure || !copy_ops->acquire_span ||
                !copy_ops->feedback_id || !copy_ops->poll_feedback ||
                !copy_ops->begin_with_feedback || !copy_ops->enqueue_span_with_feedback ||
                !copy_ops->measure(copies.get(),true))) { copies.reset(); return false; }
        copy_host = content->host()->buffer(); return true;
    }

    // Unknown cache mutations invalidate queued snapshots; authorized tail writes update these tokens.
    bool sequence_valid() const {
        return sequence && sequence->generation == content->generation() && sequence->epoch == content->mirror_epoch();
    }

    // Fill free physical slots in demand order, but do not copy an unready mutable tail.
    bool fill_sequence(bool newly_ready = false) {
        auto & seq = *sequence;
        const auto submit = [&](size_t index) {
            const auto & r = *seq.plan.request(index);
            if (r.submitted || !seq.plan.can_submit(r)) return true;
            if (!seq.started) {
                if (!ensure_copies(binding.initial_policy.ring_slots)) return false;
                if (!(measuring ? copy_ops->begin_with_feedback(copies.get(),seq.profile) : copy_ops->begin(copies.get()))) return false;
                seq.started = true;
            }
            llama_kv_stream_host_layer host;
            if (!content->host()->layer(seq.layers[r.layer],host)) return false;
            const size_t live = std::min(r.tokens,seq.active-r.first);
            const void * k = static_cast<const char *>(host.k)+r.first*page.k_token_bytes;
            const void * v = static_cast<const char *>(host.v)+r.first*page.v_token_bytes;
            if (!(measuring ? copy_ops->enqueue_span_with_feedback(copies.get(),r.slot,k,v,live,r.tokens,r.stable) :
                    copy_ops->enqueue_span(copies.get(),r.slot,k,v,live,r.tokens))) return false;
            if (!seq.plan.mark_submitted(index)) return false;
            seq.stats.max_layer_distance = std::max(seq.stats.max_layer_distance,r.layer-seq.next);
            return true;
        };
        // Only layer entry changes readiness of existing reservations; refill checks only newly added requests.
        if (newly_ready) for (size_t i = 0; i < seq.plan.pending(); ++i) if (!submit(i)) return false;
        llama_kv_stream_prefetch_request r;
        while (seq.plan.reserve(r)) {
            seq.stats.peak_pages = std::max(seq.stats.peak_pages,seq.plan.used());
            if (!submit(seq.plan.pending()-1)) return false;
        }
        return true;
    }

    // Retire pending DMA before releasing the logical session, including failure/cancellation.
    void stop_sequence(bool success = false) {
        if (!sequence) return;
        finished_sequence = sequence->stats;
        if (sequence->started) {
            copy_ops->drain(copies.get());
            const auto copied = copy_ops->stats(copies.get());
            finished_sequence.copy_bytes = copied.bytes; finished_sequence.copy_calls = copied.calls;
        }
        if (measuring) {
            if (success && sequence->started && sequence->feedback_valid && sequence->profile) record_feedback(sequence->span);
            else reset_feedback();
        }
        finished_sequence.pending_pages = 0;
        if (copies && copy_host != content->host()->buffer()) { copies.reset(); copy_host = nullptr; }
        sequence.reset();
    }

    // Storage encoding must match the fixed device layout; replacement alone cannot change it.
    bool compatible() const {
        const auto host = content->host();
        const auto & a = binding.config.shape;
        const auto & b = host->config().shape;
        return binding.cache_id == host->cache_id() && binding.config.layers == host->config().layers &&
            a.type_k == b.type_k && a.type_v == b.type_v && a.head_dim_k == b.head_dim_k &&
            a.head_dim_v == b.head_dim_v && a.heads == b.heads && a.page_tokens == b.page_tokens && a.alignment == b.alignment;
    }

    // Refresh only the capacity assigned to each layer; concentrated layouts may assign zero rows.
    bool refresh(size_t padded, uint32_t only_layer = UINT32_MAX) {
        bytes = calls = 0;
        ggml_backend_synchronize(backend);
        if (!initialized) {
            if (!content->reset_mirror()) return false;
            initialized = true;
        }
        auto & selected = only_layer == UINT32_MAX ? ranges : layer_ranges;
        for (auto & range : selected) {
            if (only_layer != UINT32_MAX) range.layer = only_layer;
            range.count = std::min(padded, layout.layers[range.layer].planes.tokens);
        }
        if (!content->flush(selected, [&](const llama_kv_stream_copy_span & span) {
            const bool value = span.rows.operand == ggml_kv_stream_operand::v;
            const auto & planes = layout.layers[span.rows.layer].planes;
            const size_t stride = value ? planes.v_token_bytes : planes.k_token_bytes;
            auto * root = value ? roots[span.rows.layer].second : roots[span.rows.layer].first;
            ggml_backend_tensor_set(root, span.data, span.rows.first*stride, span.bytes);
            bytes += span.bytes; ++calls;
            return true;
        })) return false;
        return true;
    }

    // Ordinary CUDA attention needs padded keys, but every padded row must remain resident.
    bool extent(size_t tokens, size_t & padded) const {
        if (poisoned || !compatible() || !tokens || tokens > content->host()->config().context_tokens || tokens > size_t(INT32_MAX) - 255) return false;
        padded = (tokens + 255)/256*256;
        for (const auto & entry : layout.layers) if (padded > entry.planes.tokens) return false;
        return padded <= content->host()->layout().tokens;
    }

    // Describe token-major heads using the same strides as the original resident fast path.
    ggml_tensor descriptor(uint32_t layer, bool value, size_t padded) const {
        ggml_tensor tensor = *(value ? roots[layer].second : roots[layer].first);
        const auto & shape = binding.config.shape;
        const auto & planes = layout.layers[layer].planes;
        tensor.ne[0] = value ? shape.head_dim_v : shape.head_dim_k;
        tensor.ne[1] = int64_t(padded);
        tensor.ne[2] = shape.heads;
        tensor.ne[3] = 1;
        tensor.nb[1] = value ? planes.v_token_bytes : planes.k_token_bytes;
        tensor.nb[2] = value ? planes.v_row_bytes : planes.k_row_bytes;
        tensor.nb[3] = value ? planes.v_bytes : planes.k_bytes;
        return tensor;
    }
};

// Reuse the existing optional copy adapter; enabling diagnostics does not alter default spans.
bool llama_kv_stream_resident::configure_feedback(bool enable, size_t bounded) {
    auto & s = *impl;
    if (s.busy || s.sequence || s.capturing() || !bounded) return false;
    if (enable && (!s.ensure_copies(s.binding.initial_policy.ring_slots) || s.copy_ops->version < 7 ||
            !s.copy_ops->measure || !s.copy_ops->poll_feedback || !s.copy_ops->feedback_id || !s.copy_ops->acquire_span ||
            !s.copy_ops->begin_with_feedback || !s.copy_ops->enqueue_span_with_feedback)) return false;
    if (s.copies && s.copy_ops->version >= 4 && !s.copy_ops->measure(s.copies.get(),enable)) return false;
    s.measuring = enable; s.bounded_span = bounded; s.reset_feedback();
    return true;
}

bool llama_kv_stream_resident::configure_native_graph_attention(bool enable) {
    auto & s = *impl;
    if (s.busy || s.sequence || s.capturing()) return false;
    auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(s.backend));
    auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(reg,"ggml_backend_kv_stream_partial_ops"));
    if (enable && (!get || !get() || get()->version < 4 || !get()->direct)) return false;
    s.native_graph_attention = enable; return true;
}
bool llama_kv_stream_resident::configure_resumed_decode(bool enable) {
    auto & s = *impl;
    if (s.busy || s.sequence || s.capturing()) return false;
    s.resumed_decode = enable; return true;
}
size_t llama_kv_stream_resident::suggested_span_pages() const noexcept {
    (void) feedback();
    const auto & s = *impl;
    return s.tuner.bounded() ? std::min(size_t(s.binding.initial_policy.ring_slots),s.bounded_span) : s.binding.initial_policy.ring_slots;
}
llama_kv_stream_feedback llama_kv_stream_resident::feedback() const noexcept {
    auto & s = *impl;
    if (!s.measuring || s.busy || s.sequence) return {};
    const bool current = s.feedback_generation == s.content->generation() && s.feedback_epoch == s.content->mirror_epoch();
    s.collect_feedback(current);
    if (!current) return {};
    return s.window.snapshot();
}
ggml_kv_stream_copy_feedback llama_kv_stream_resident::copy_feedback() const noexcept {
    return feedback().available ? impl->last_feedback : ggml_kv_stream_copy_feedback{};
}
// Reading feedback never advances the accepted policy cursor or changes device addresses.
bool llama_kv_stream_resident::recommend_policy(const llama_kv_stream_policy_state & previous, size_t tokens,
        uint32_t queries, llama_kv_stream_policy_decision & decision, bool decode_feedback, bool uniform_prefill) const {
    if (impl->busy || impl->sequence) return false;
    const bool same = decode_feedback && queries == 1 && tokens &&
        (tokens-1)/impl->page.tokens+1 == impl->feedback_pages && queries == impl->feedback_queries;
    return llama_kv_stream_policy_step(impl->binding.config,previous,
            {tokens,queries,same ? feedback() : llama_kv_stream_feedback{},uniform_prefill},decision).status == llama_kv_stream_policy_status::success;
}

// External captures must be retired before their binding execution pins are returned.
llama_kv_stream_resident::~llama_kv_stream_resident() = default;

// Establish a serial execution order and the host prefix that is already immutable for every layer.
bool llama_kv_stream_resident::begin_sequence(const std::vector<uint32_t> & layers, size_t active, size_t span, size_t stable, llama_kv_stream_feedback_context feedback_context) {
    auto & s = *impl;
    if (s.busy || s.sequence || s.capturing() || s.poisoned || !s.compatible() || layers.empty() || layers.size() > s.layout.layers.size() || !active || !span ||
            active > size_t(INT32_MAX)-255 || active > s.content->host()->config().context_tokens) return false;
    if (s.measuring) s.run_started = std::chrono::steady_clock::now();
    (void) feedback();
    auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(s.backend));
    auto partial_ops = reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(reg,"ggml_backend_kv_stream_partial_ops"));
    auto copy_ops = reinterpret_cast<ggml_kv_stream_copy_ops_get>(ggml_backend_reg_get_proc_address(reg,"ggml_backend_kv_stream_copy_ops"));
    if (!partial_ops || !partial_ops() || !copy_ops || !copy_ops() || copy_ops()->version < 2) return false;
    if (stable == SIZE_MAX) stable = active;
    if (stable > active) return false;
    try {
        auto seq = std::make_unique<implementation::prefetch_sequence>();
        seq->layers = layers; seq->resident_dirty.resize(layers.size(),false);
        seq->active = active; seq->padded = (active+255)/256*256; seq->span = span; seq->stable = stable;
        seq->queries = feedback_context.query_tokens;
        seq->decode = feedback_context.decode;
        std::vector<size_t> prefixes;
        for (size_t i = 0; i < layers.size(); ++i) {
            if (layers[i] >= s.layout.layers.size() || std::find(layers.begin(),layers.begin()+i,layers[i]) != layers.begin()+i) return false;
            prefixes.push_back(std::min(seq->padded,s.layout.layers[layers[i]].planes.tokens));
            const size_t immutable = stable == active ? seq->padded : stable/s.page.tokens*s.page.tokens;
            seq->profile |= feedback_context.decode && seq->queries == 1 && prefixes.back() < immutable;
        }
        if (!seq->plan.start(prefixes,active,seq->padded,s.page.tokens,s.binding.initial_policy.ring_slots,span,stable)) return false;
        if (std::any_of(prefixes.begin(),prefixes.end(),[&](size_t prefix) { return prefix < seq->padded; }) && !s.enter_streamed()) return false;
        if (!s.writer_lease) release_write_workspace();
        s.busy = true; s.valid = false;
        struct guard { bool & busy; ~guard() { busy = false; } } operation{s.busy};
        if (!s.refresh(seq->padded)) return false;
        seq->generation = s.content->generation(); seq->epoch = s.content->mirror_epoch();
        s.sequence = std::move(seq);
        if (!s.fill_sequence()) { s.stop_sequence(); return false; }
        return true;
    } catch (const std::bad_alloc &) { s.stop_sequence(); return false; }
}

// Only unconsumed tail rows may change while stable historical spans are in flight.
bool llama_kv_stream_resident::publish_sequence_tail(const std::vector<llama_kv_stream_write_span> & spans) {
    auto & s = *impl;
    if (s.busy || s.capturing() || !s.sequence) return false;
    if (!s.sequence_valid()) { s.stop_sequence(); return false; }
    if (!s.tail_valid(spans,false)) return false;
    llama_kv_stream_write write;
    if (!s.content->prepare(spans,write) || !s.content->commit(write)) return false;
    s.tail_committed(spans);
    return true;
}

// Caller may cancel only between owner-thread layer calls.
void llama_kv_stream_resident::cancel_sequence() { if (!impl->busy) impl->stop_sequence(); }
bool llama_kv_stream_resident::sequence_active() const noexcept { return bool(impl->sequence); }
llama_kv_stream_prefetch_stats llama_kv_stream_resident::sequence_stats() const noexcept {
    const auto & s = *impl;
    if (!s.sequence) return s.finished_sequence;
    auto result = s.sequence->stats; result.pending_pages = s.sequence->plan.used();
    if (s.sequence->started) {
        auto copy = s.copy_ops->stats(s.copies.get()); result.copy_bytes = copy.bytes; result.copy_calls = copy.calls;
        for (size_t i = 0; i < s.sequence->plan.pending(); ++i) {
            const auto * r = s.sequence->plan.request(i);
            if (!r->submitted) continue;
            for (size_t page = 0; page < r->pages; ++page) result.ready_pages += s.copy_ops->ready(s.copies.get(),r->slot+page);
        }
        const auto * front = s.sequence->plan.front();
        result.demand_ready = front && front->submitted && s.copy_ops->ready(s.copies.get(),front->slot);
    }
    return result;
}

// Construct only metadata and borrowed tensor bindings; do not copy or clear device memory in the factory.
std::unique_ptr<llama_kv_stream_resident> llama_kv_stream_resident::create(
        const llama_kv_stream_binding_view & binding, std::shared_ptr<llama_kv_stream_content> content, ggml_backend_t backend,
        const llama_kv_stream_policy_state * placement) {
    if (!content || !backend || !binding.buffer || !binding.base || binding.capacity != binding.config.pool_bytes ||
            binding.capacity > ggml_backend_buffer_get_size(binding.buffer) ||
            binding.base != ggml_backend_buffer_get_base(binding.buffer) ||
            !ggml_backend_supports_buft(backend, ggml_backend_buffer_get_type(binding.buffer))) return {};
    const auto & shape = binding.config.shape;
    if (shape.page_tokens <= 0 || shape.page_tokens % 256 || shape.head_dim_k > INT32_MAX ||
            shape.head_dim_v > INT32_MAX || shape.heads > INT32_MAX) return {};
    try {
        std::unique_ptr<llama_kv_stream_resident> result(new llama_kv_stream_resident);
        result->impl = std::make_unique<implementation>();
        auto & s = *result->impl;
        s.binding = binding; s.content = std::move(content); s.backend = backend;
        if (placement) s.binding.initial_policy = *placement;
        const auto & state = s.binding.initial_policy;
        if (size_t(state.decode_active_pages) > SIZE_MAX/size_t(shape.page_tokens)) return {};
        if (!s.compatible() || llama_kv_stream_policy_layout_make(binding.config, state,
                size_t(state.decode_active_pages)*size_t(shape.page_tokens), s.layout).status != llama_kv_stream_policy_status::success) return {};
        s.fallback = state.budget.page.attention == ggml_kv_stream_attention::f16;
        s.conversion = state.budget.page.conversion;
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
        s.query_capture = reinterpret_cast<ggml_backend_cuda_graph_is_capturing_t>(
            ggml_backend_reg_get_proc_address(reg,"ggml_backend_cuda_graph_is_capturing"));
        auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(reg,"ggml_backend_kv_stream_partial_ops"));
        const auto * ops = get ? get() : nullptr;
        if (s.fallback && (!ops || ops->version < 3 || !ops->capabilities || !ops->supports_conversion || !ops->convert)) return {};
        if (ops && ops->version >= 3 && ops->capabilities) {
            auto actual = ops->capabilities(backend,shape.type_k,shape.type_v);
            if (s.fallback) actual.direct_pair = false;
            ggml_kv_stream_execution execution;
            if (ggml_kv_stream_resolve(shape,actual,size_t(shape.page_tokens),execution).status != ggml_kv_stream_status::success ||
                    execution.attention != state.budget.page.attention) return {};
        }
        const size_t layers = s.layout.layers.size();
        if (layers > (SIZE_MAX/ggml_tensor_overhead() - 5)/2) return {};
        s.context.reset(ggml_init({(2*layers + 5)*ggml_tensor_overhead(), nullptr, true}));
        if (!s.context) return {};
        s.roots.reserve(layers);
        s.ranges.reserve(2*layers);
        for (uint32_t layer = 0; layer < layers; ++layer) {
            const auto & entry = s.layout.layers[layer];
            ggml_tensor * roots[2];
            for (int value = 0; value < 2; ++value) {
                const auto type = ggml_type(value ? shape.type_v : shape.type_k);
                const size_t dim = size_t(value ? shape.head_dim_v : shape.head_dim_k);
                const size_t bytes = value ? entry.planes.v_bytes : entry.planes.k_bytes;
                if (entry.planes.tokens > size_t(INT64_MAX)/size_t(shape.heads)/dim) return {};
                const size_t elements = entry.planes.tokens*size_t(shape.heads)*dim;
                // Flat roots naturally satisfy quantized matrix padding without adding bytes between the K/V planes.
                roots[value] = ggml_new_tensor_1d(s.context.get(), type, int64_t(elements));
                if (!bytes) continue;
                if (ggml_nbytes(roots[value]) != bytes ||
                        ggml_backend_buffer_get_alloc_size(binding.buffer, roots[value]) != bytes) return {};
                const size_t offset = entry.offset + (value ? entry.planes.v_offset : 0);
                if (offset > binding.capacity || bytes > binding.capacity - offset) return {};
                if (ggml_backend_tensor_alloc(binding.buffer, roots[value], static_cast<char *>(binding.base) + offset) != GGML_STATUS_SUCCESS) return {};
            }
            s.roots.emplace_back(roots[0], roots[1]);
            s.ranges.push_back({layer, ggml_kv_stream_operand::k, 0, 0});
            s.ranges.push_back({layer, ggml_kv_stream_operand::v, 0, 0});
        }
        if (ggml_kv_stream_layout_make(shape, size_t(shape.page_tokens), s.page).status != ggml_kv_stream_status::success ||
                s.page.bytes > s.layout.ring.bytes) return {};
        for (int value = 0; value < 2; ++value) {
            const auto type = ggml_type(value ? shape.type_v : shape.type_k);
            const size_t dim = size_t(value ? shape.head_dim_v : shape.head_dim_k);
            const size_t bytes = value ? s.page.v_bytes : s.page.k_bytes;
            if (s.page.tokens > size_t(INT64_MAX)/size_t(shape.heads)/dim) return {};
            s.stage[value] = ggml_new_tensor_1d(s.context.get(), type, int64_t(s.page.tokens*size_t(shape.heads)*dim));
            if (ggml_backend_buffer_get_alloc_size(binding.buffer,s.stage[value]) != bytes ||
                    ggml_backend_tensor_alloc(binding.buffer,s.stage[value],static_cast<char *>(binding.base)+(value ? s.layout.ring.v_offset : 0)) != GGML_STATUS_SUCCESS) return {};
        }
        if (s.fallback) for (int value = 0; value < 2; ++value) {
            const size_t bytes = value ? s.conversion.v_bytes : s.conversion.k_bytes;
            const size_t offset = s.layout.conversion_offset+(value ? s.conversion.v_offset : 0);
            if (s.conversion.bytes != s.layout.conversion_bytes || offset > binding.capacity ||
                    bytes > binding.capacity-offset || bytes/sizeof(ggml_fp16_t) > size_t(INT64_MAX)) return {};
            s.converted[value] = ggml_new_tensor_1d(s.context.get(),GGML_TYPE_F16,int64_t(bytes/sizeof(ggml_fp16_t)));
            if (ggml_backend_buffer_get_alloc_size(binding.buffer,s.converted[value]) != bytes ||
                    ggml_backend_tensor_alloc(binding.buffer,s.converted[value],static_cast<char *>(binding.base)+offset) != GGML_STATUS_SUCCESS) return {};
        }
        return result;
    } catch (const std::bad_alloc &) {
        return {};
    }
}

// Complete older attention before overwriting its inputs; acknowledgement follows completed synchronous copies.
bool llama_kv_stream_resident::synchronize(size_t active_tokens) {
    auto & s = *impl;
    if (s.busy || s.sequence || s.capturing()) return false;
    s.valid = false; s.bytes = s.calls = 0;
    size_t padded;
    if (!s.extent(active_tokens, padded)) return false;
    s.busy = true;
    struct guard { bool & busy; ~guard() { busy = false; } } operation{s.busy};
    if (!s.refresh(padded)) return false;
    s.active = active_tokens;
    s.generation = s.content->generation();
    s.epoch = s.content->mirror_epoch();
    s.valid = true;
    return true;
}

// Physical address stability alone does not prove that the resident contents are current.
bool llama_kv_stream_resident::ready(size_t active_tokens) const noexcept {
    const auto & s = *impl;
    return !s.busy && s.valid && s.active == active_tokens && s.generation == s.content->generation() && s.epoch == s.content->mirror_epoch();
}

// A data refresh at the same padded extent can reuse a capture; a replaced mirror or streamed epoch cannot.
bool llama_kv_stream_resident::capture_state(ggml_backend_t backend, size_t tokens, llama_kv_stream_capture_stamp & output) const {
    const auto & s = *impl;
    size_t padded;
    if (backend != s.backend || !s.query_capture || s.busy || s.sequence || s.fallback || !s.extent(tokens,padded)) return false;
    output = {s.binding.buffer,s.binding.revision,s.residency_revision,s.content->mirror_epoch(),padded};
    return true;
}

// Only ordinary native TG1 attention over this consumer's exact resident planes is capturable here.
bool llama_kv_stream_resident::capture_attention(const ggml_tensor * op, size_t tokens) const {
    const auto & s = *impl;
    llama_kv_stream_capture_stamp state;
    if (!op || op->op != GGML_OP_FLASH_ATTN_EXT || !capture_state(s.backend,tokens,state) ||
            !op->src[0] || op->src[0]->ne[1] != 1 || (!op->src[3] && tokens != state.padded_tokens)) return false;
    const auto equal = [](const ggml_tensor * actual, const ggml_tensor & expected) {
        return actual && actual->buffer == expected.buffer && actual->data == expected.data && actual->type == expected.type &&
            !std::memcmp(actual->ne,expected.ne,sizeof(expected.ne)) && !std::memcmp(actual->nb,expected.nb,sizeof(expected.nb));
    };
    bool planes = false;
    for (uint32_t layer = 0; layer < s.roots.size(); ++layer)
        if (equal(op->src[1],s.descriptor(layer,false,state.padded_tokens)) &&
                equal(op->src[2],s.descriptor(layer,true,state.padded_tokens)) &&
                op->src[1]->view_src == s.roots[layer].first && op->src[1]->src[0] == s.roots[layer].first &&
                op->src[2]->view_src == s.roots[layer].second && op->src[2]->src[0] == s.roots[layer].second &&
                !op->src[1]->view_offs && !op->src[2]->view_offs) { planes = true; break; }
    if (!planes) return false;
    ggml_kv_stream_execution execution;
    const auto & shape = s.binding.config.shape;
    return ggml_kv_stream_attention_validate(op,{shape.head_dim_k,shape.head_dim_v,256,shape.page_tokens,shape.alignment},
        s.binding.config.capabilities,state.padded_tokens,execution).status == ggml_kv_stream_status::success &&
        execution.attention == ggml_kv_stream_attention::direct && ggml_backend_supports_op(s.backend,op);
}

// Report completed uploads in the last synchronization attempt, not PCIe utilization.
size_t llama_kv_stream_resident::last_upload_bytes() const noexcept { return impl->bytes; }
size_t llama_kv_stream_resident::last_upload_calls() const noexcept { return impl->calls; }
size_t llama_kv_stream_resident::last_attention_calls() const noexcept { return impl->attention_calls; }

// Validate stack descriptors before GGML constructors can assert on malformed attention metadata.
ggml_tensor * llama_kv_stream_resident::attention(ggml_context * context, uint32_t layer, ggml_tensor * q,
        ggml_tensor * mask, size_t active_tokens, float scale) {
    auto & s = *impl;
    size_t padded;
    if (s.busy || s.sequence || !context || !q || layer >= s.roots.size() || !std::isfinite(scale) || scale <= 0 ||
            !s.extent(active_tokens, padded) || (!mask && active_tokens != padded) ||
            (mask && !ggml_is_contiguous(mask))) return nullptr;
    auto k = s.descriptor(layer, false, padded), v = s.descriptor(layer, true, padded);
    ggml_tensor prototype = {};
    prototype.type = GGML_TYPE_F32; prototype.op = GGML_OP_FLASH_ATTN_EXT;
    prototype.ne[0] = v.ne[0]; prototype.ne[1] = q->ne[2]; prototype.ne[2] = q->ne[1]; prototype.ne[3] = 1;
    prototype.nb[0] = sizeof(float);
    for (int i = 0; i < 3; ++i) {
        if (prototype.ne[i] <= 0 || uint64_t(prototype.ne[i]) > SIZE_MAX/prototype.nb[i]) return nullptr;
        prototype.nb[i+1] = prototype.nb[i]*size_t(prototype.ne[i]);
    }
    prototype.src[0] = q; prototype.src[1] = &k; prototype.src[2] = &v; prototype.src[3] = mask;
    std::memcpy(prototype.op_params, &scale, sizeof(scale));
    ggml_flash_attn_ext_set_prec(&prototype, GGML_PREC_F32);
    const auto & shape = s.binding.config.shape;
    const ggml_kv_stream_attention_limits limits{shape.head_dim_k, shape.head_dim_v, 256, shape.page_tokens, shape.alignment};
    ggml_kv_stream_execution execution;
    if (ggml_kv_stream_attention_validate(&prototype, limits, s.binding.config.capabilities, padded, execution).status !=
            ggml_kv_stream_status::success || execution.attention != ggml_kv_stream_attention::direct ||
            !ggml_backend_supports_op(s.backend, &prototype)) return nullptr;
    auto * kt = ggml_view_4d(context, s.roots[layer].first, k.ne[0], k.ne[1], k.ne[2], 1, k.nb[1], k.nb[2], k.nb[3], 0);
    auto * vt = ggml_view_4d(context, s.roots[layer].second, v.ne[0], v.ne[1], v.ne[2], 1, v.nb[1], v.nb[2], v.nb[3], 0);
    if (ggml_backend_view_init(kt) != GGML_STATUS_SUCCESS || ggml_backend_view_init(vt) != GGML_STATUS_SUCCESS) return nullptr;
    auto * out = ggml_flash_attn_ext(context, q, kt, vt, mask, scale, 0, 0);
    ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
    return out;
}

// Default scratch borrows the idle ring; a separate leased region also permits online sequence producers.
bool llama_kv_stream_resident::configure_writes(size_t maximum, ggml_backend_memory_lease_t workspace) {
    auto & s = *impl;
    if (s.busy || s.sequence || s.capturing() || s.poisoned || !s.compatible() || !maximum) return false;
    auto * buffer = s.binding.buffer;
    size_t bytes = s.layout.ring.bytes;
    if (workspace) {
        ggml_backend_memory_region region;
        buffer = ggml_backend_memory_lease_buffer(workspace);
        if (!buffer || !ggml_backend_memory_lease_get_region(workspace,&region) || !region.size ||
                region.size != ggml_backend_buffer_get_size(buffer) ||
                ggml_backend_buffer_get_type(buffer) != ggml_backend_buffer_get_type(s.binding.buffer)) return false;
        const auto base = uintptr_t(ggml_backend_buffer_get_base(buffer)), pool = uintptr_t(s.binding.base);
        bytes = region.size;
        if (!base || base%128 || base > UINTPTR_MAX-bytes || pool > UINTPTR_MAX-s.binding.capacity ||
                (base < pool+s.binding.capacity && pool < base+bytes)) return false;
    }
    decltype(s.writer_lease) lease(ggml_backend_memory_lease_retain(workspace),ggml_backend_memory_lease_free);
    s.busy = true;
    struct guard { bool & busy; ~guard() { busy = false; } } operation{s.busy};
    ggml_backend_synchronize(s.backend);
    s.writer.reset();
    s.writer_lease.reset();
    s.writer_base = s.writer_bytes = 0;
    s.write_stats = {};
    s.writer = llama_kv_stream_writer::create(s.backend,buffer,bytes,s.binding.config.shape,maximum);
    if (s.writer && lease) {
        s.writer_base = uintptr_t(ggml_backend_buffer_get_base(buffer)); s.writer_bytes = bytes;
        s.writer_lease = std::move(lease);
    }
    return bool(s.writer);
}

// Complete quantization/download into a private ticket before making either producer plane visible.
bool llama_kv_stream_resident::write_sequence_rows(uint32_t layer, size_t first, const ggml_tensor * k, const ggml_tensor * v) {
    auto & s = *impl;
    if (s.busy || s.capturing() || !s.sequence || !s.writer || !s.writer_lease) return false;
    s.write_stats = {};
    if (!s.sequence_valid()) { s.stop_sequence(); return false; }
    if (!s.writer->accepts(k,false) || !s.writer->accepts(v,true) || k->ne[1] != v->ne[1] ||
            first != s.sequence->stable || size_t(k->ne[1]) != s.sequence->active-first) return false;
    const size_t rows = size_t(k->ne[1]);
    const std::vector<llama_kv_stream_write_span> spans{
        {layer,ggml_kv_stream_operand::k,first*s.page.k_token_bytes,nullptr,rows*s.page.k_token_bytes},
        {layer,ggml_kv_stream_operand::v,first*s.page.v_token_bytes,nullptr,rows*s.page.v_token_bytes}};
    if (!s.tail_valid(spans,true)) return false;
    s.busy = true;
    struct guard { bool & busy; ~guard() { busy = false; } } operation{s.busy};
    const auto accumulate = [&] {
        const auto part = s.writer->stats();
        s.write_stats.graph_submissions += part.graph_submissions;
        s.write_stats.d2h_bytes += part.d2h_bytes; s.write_stats.d2h_calls += part.d2h_calls;
        s.write_stats.device_scratch_bytes = std::max(s.write_stats.device_scratch_bytes,part.device_scratch_bytes);
        s.write_stats.host_payload_bytes += part.host_payload_bytes; s.write_stats.tile_rows = part.tile_rows;
    };
    try {
        llama_kv_stream_write write;
        const bool generated = s.content->prepare_generated(spans,[&](const auto & span, void * payload) {
            const bool value = span.operand == ggml_kv_stream_operand::v;
            try {
                const bool result = s.writer->generate(value ? v : k,value,payload,{});
                accumulate(); return result;
            } catch (...) { accumulate(); throw; }
        },write);
        if (!generated || !s.content->commit(write)) { s.stop_sequence(); return false; }
        s.tail_committed(spans);
        return true;
    } catch (...) { s.stop_sequence(); throw; }
}

// Publish host bytes atomically after all tiled generation completes; completed D2D tiles need no H2D refresh.
bool llama_kv_stream_resident::write_rows(uint32_t layer, ggml_kv_stream_operand operand, size_t first, const ggml_tensor * source) {
    auto & s = *impl;
    s.write_stats = {};
    const bool value = operand == ggml_kv_stream_operand::v;
    if (s.busy || s.sequence || s.capturing() || s.poisoned || !s.writer || !s.compatible() || layer >= s.roots.size() ||
            (operand != ggml_kv_stream_operand::k && !value) || !s.writer->accepts(source, value)) return false;
    const size_t rows = size_t(source->ne[1]);
    const auto host = s.content->host();
    if (first > host->config().context_tokens || rows > host->config().context_tokens - first ||
            first > s.layout.layers[layer].planes.tokens || rows > s.layout.layers[layer].planes.tokens - first) return false;
    s.busy = true; s.valid = false;
    struct guard { bool & busy; ~guard() { busy = false; } } operation{s.busy};
    // SET_ROWS queues behind earlier backend work; each tile drains before any resident overwrite.
    if (!s.initialized) {
        if (!s.content->reset_mirror()) return false;
        s.initialized = true;
    }
    const size_t stride = value ? host->layout().v_token_bytes : host->layout().k_token_bytes;
    const size_t width = size_t(value ? s.binding.config.shape.head_dim_v : s.binding.config.shape.head_dim_k)*size_t(s.binding.config.shape.heads);
    bool touched = false, generation_started = false;
    // If speculative mirror publication fails, the unchanged authoritative bytes must overwrite it on retry.
    auto rollback = [&] {
        if (touched && !s.content->reset_mirror()) s.poisoned = true;
    };
    try {
        const std::vector<llama_kv_stream_rows> acknowledge{{layer, operand, first, rows}};
        const std::function<bool(const llama_kv_stream_copy_span &)> already_copied = [&](const auto & span) {
            return span.rows.layer == layer && span.rows.operand == operand && span.rows.first >= first &&
                span.rows.first - first <= rows && span.rows.count <= rows - (span.rows.first - first);
        };
        llama_kv_stream_write write;
        const bool generated = s.content->prepare_generated({{layer, operand, first*stride, nullptr, rows*stride}},
            [&](const auto &, void * payload) {
                generation_started = true;
                return s.writer->generate(source, value, payload, [&](const ggml_tensor * stage, size_t offset, size_t count) {
                    ggml_tensor src = *stage, dst = *stage;
                    src.ne[0] = dst.ne[0] = int64_t(width*count);
                    src.ne[1] = src.ne[2] = src.ne[3] = dst.ne[1] = dst.ne[2] = dst.ne[3] = 1;
                    src.nb[1] = src.nb[2] = src.nb[3] = dst.nb[1] = dst.nb[2] = dst.nb[3] = count*stride;
                    auto * root = value ? s.roots[layer].second : s.roots[layer].first;
                    dst.buffer = root->buffer;
                    dst.data = static_cast<char *>(root->data) + (first + offset)*stride;
                    touched = true;
                    ggml_backend_tensor_copy_async(s.backend, s.backend, &src, &dst);
                    return true;
                });
            }, write);
        s.write_stats = generation_started ? s.writer->stats() : llama_kv_stream_write_stats{};
        if (!generated || !s.content->commit(write)) { rollback(); return false; }
        if (!s.content->flush(acknowledge, already_copied)) { rollback(); return false; }
        return true;
    } catch (...) {
        s.write_stats = generation_started ? s.writer->stats() : llama_kv_stream_write_stats{};
        rollback();
        throw;
    }
}

// Keep the previous one-block admission boundary while sharing the traversal implementation.
bool llama_kv_stream_resident::compute_one_block(uint32_t layer, ggml_tensor * q, ggml_tensor * mask,
        ggml_tensor * output, size_t active_tokens, float scale, ggml_backend_memory_lease_t workspace) {
    if (layer >= impl->layout.layers.size()) return false;
    const size_t prefix = impl->layout.layers[layer].planes.tokens;
    if (active_tokens <= prefix || active_tokens-prefix > impl->page.tokens) return false;
    return compute_streamed(layer,q,mask,output,active_tokens,scale,workspace);
}

// Traverse resident and streamed ranges with one reusable export and an unnormalized accumulator.
bool llama_kv_stream_resident::compute_streamed(uint32_t layer, ggml_tensor * q, ggml_tensor * mask,
        ggml_tensor * output, size_t active_tokens, float scale, ggml_backend_memory_lease_t workspace, bool overlap, size_t span_pages, bool decode_feedback) {
    auto & s = *impl;
    if (s.busy || s.capturing()) return false; // Rejection must not drain/cancel an outer call or active capture.
    const bool cross = bool(s.sequence);
    if (!cross && s.measuring) s.run_started = std::chrono::steady_clock::now();
    struct session_guard {
        implementation & s; bool enabled, keep = false;
        ~session_guard() { if (enabled && !keep) s.stop_sequence(); }
    } session_call{s,cross};
    if (cross && (!s.sequence_valid() || !overlap || active_tokens != s.sequence->active || span_pages != s.sequence->span ||
            s.sequence->next >= s.sequence->layers.size() || layer != s.sequence->layers[s.sequence->next])) return false;
    if (s.busy || s.poisoned || !s.compatible() || layer >= s.roots.size() || !q || !mask || !output ||
            !workspace || !std::isfinite(scale) || scale <= 0 || active_tokens > size_t(INT32_MAX)-255 ||
            active_tokens > s.content->host()->config().context_tokens || q->ne[1] <= 0 || q->ne[2] <= 0 ||
            q->ne[1] > INT32_MAX/512/q->ne[2]) return false;
    if (!active_tokens) return false;
    if (cross && s.sequence->queries && q->ne[1] != s.sequence->queries) return false;
    const size_t padded = (active_tokens+255)/256*256;
    const size_t prefix = std::min(padded,s.layout.layers[layer].planes.tokens);
    const size_t slots = s.binding.initial_policy.ring_slots;
    if (!slots || !span_pages || slots > s.layout.ring.bytes/s.page.bytes) return false;
    const size_t blocks = (padded-prefix)/s.page.tokens + ((padded-prefix)%s.page.tokens != 0);
    const auto width = [&](size_t block) { return ggml_kv_stream_contiguous_pages(block,blocks,slots,span_pages); };
    if (padded > s.content->host()->layout().tokens || mask->ne[0] < int64_t(padded) ||
            mask->type != GGML_TYPE_F16 || !mask->data || mask->nb[0] != sizeof(ggml_fp16_t) ||
            mask->ne[0] > INT32_MAX || mask->nb[1] < size_t(mask->ne[0])*sizeof(ggml_fp16_t)) return false;
    auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(s.backend));
    auto get = reinterpret_cast<ggml_kv_stream_partial_ops_get>(ggml_backend_reg_get_proc_address(reg,"ggml_backend_kv_stream_partial_ops"));
    const auto * ops = get ? get() : nullptr;
    if (!ops || ops->version < 2 || !ops->supports || !ops->partial || !ops->merge || !ops->fold || !ops->clear) return false;
    ggml_kv_stream_block_layout work;
    ggml_kv_stream_layout gathered;
    ggml_kv_stream_resume_plan resume_plan;
    const bool resumed = s.resumed_decode && cross && s.sequence->decode && q->ne[1] == 1 && !s.fallback &&
        ops->version >= 5 && ops->resume_plan && ops->resume &&
        ops->resume_plan(s.backend,s.binding.config.shape.type_k,s.binding.config.shape.type_v,
            uint32_t(q->ne[2]),uint32_t(s.binding.config.shape.heads),padded,resume_plan);
    const bool native = s.native_graph_attention && !resumed;
    if (ggml_kv_stream_block_layout_make(size_t(q->ne[1])*size_t(q->ne[2]),size_t(output->ne[0]),work).status !=
            ggml_kv_stream_partial_status::success) return false;
    if (native && ggml_kv_stream_layout_make(s.binding.config.shape,padded,gathered).status != ggml_kv_stream_status::success) return false;
    std::unique_ptr<ggml_backend_memory_lease, decltype(&ggml_backend_memory_lease_free)>
        lease(ggml_backend_memory_lease_retain(workspace),ggml_backend_memory_lease_free);
    auto * wb = ggml_backend_memory_lease_buffer(lease.get());
    ggml_backend_memory_region region;
    if (!wb || !ggml_backend_memory_lease_get_region(lease.get(),&region) ||
            region.size != ggml_backend_buffer_get_size(wb) || region.size < (resumed ? resume_plan.bytes : native ? gathered.bytes : work.bytes) ||
            ggml_backend_buffer_get_type(wb) != ggml_backend_buffer_get_type(s.binding.buffer)) return false;
    const auto scratch = uintptr_t(ggml_backend_buffer_get_base(wb));
    const auto pool = uintptr_t(s.binding.base);
    if (!scratch || scratch%128 || scratch > UINTPTR_MAX-region.size || pool > UINTPTR_MAX-s.binding.capacity) return false;
    const auto overlaps = [](uintptr_t a, size_t na, uintptr_t b, size_t nb) {
        return a < b+nb && b < a+na;
    };
    if (overlaps(scratch,region.size,pool,s.binding.capacity)) return false;
    if (s.writer_bytes && overlaps(scratch,region.size,s.writer_base,s.writer_bytes)) return false;
    ggml_tensor k, v, ck, cv, slice, op;
    // Each slot selects one page from the separate contiguous ring K and V planes.
    const auto describe = [&](size_t first, size_t count, size_t slot, bool resident) {
        k = s.descriptor(layer,false,count); v = s.descriptor(layer,true,count);
        if (!resident) {
            k.buffer = v.buffer = s.binding.buffer;
            k.data = static_cast<char *>(s.stage[0]->data)+slot*s.page.k_bytes;
            v.data = static_cast<char *>(s.stage[1]->data)+slot*s.page.v_bytes;
            k.nb[3] = s.page.k_bytes; v.nb[3] = s.page.v_bytes;
        } else {
            k.data = static_cast<char *>(k.data)+first*s.page.k_token_bytes;
            v.data = static_cast<char *>(v.data)+first*s.page.v_token_bytes;
        }
        if (s.fallback) {
            if (count > s.conversion.tokens) return false;
            ck = *s.converted[0]; cv = *s.converted[1];
            for (auto item : {std::pair{&ck,false},std::pair{&cv,true}}) {
                auto * tensor = item.first;
                tensor->ne[0] = item.second ? s.binding.config.shape.head_dim_v : s.binding.config.shape.head_dim_k;
                tensor->ne[1] = int64_t(count); tensor->ne[2] = s.binding.config.shape.heads; tensor->ne[3] = 1;
                tensor->nb[1] = item.second ? s.conversion.v_token_bytes : s.conversion.k_token_bytes;
                tensor->nb[2] = item.second ? s.conversion.v_row_bytes : s.conversion.k_row_bytes;
                tensor->nb[3] = item.second ? s.conversion.v_bytes : s.conversion.k_bytes;
            }
            if (!ops->supports_conversion(s.backend,&k,&ck) || !ops->supports_conversion(s.backend,&v,&cv)) return false;
        }
        slice = *mask; slice.ne[0] = int64_t(count);
        if (uintptr_t(mask->data) > UINTPTR_MAX-first*sizeof(ggml_fp16_t)) return false;
        slice.data = reinterpret_cast<void *>(uintptr_t(mask->data)+first*sizeof(ggml_fp16_t));
        op = *output; op.op = GGML_OP_FLASH_ATTN_EXT;
        std::memset(op.src,0,sizeof(op.src)); std::memset(op.op_params,0,sizeof(op.op_params));
        std::memcpy(op.op_params,&scale,sizeof(scale)); ggml_flash_attn_ext_set_prec(&op,GGML_PREC_F32);
        op.src[0] = q; op.src[1] = s.fallback ? &ck : &k; op.src[2] = s.fallback ? &cv : &v; op.src[3] = &slice;
        return ops->supports(s.backend,&op);
    };
    const size_t resident_chunk = s.fallback ? s.page.tokens : prefix;
    for (size_t first = 0; first < prefix; first += resident_chunk)
        if (!describe(first,std::min(resident_chunk,prefix-first),0,true)) return false;
    // Validate bounded covering spans here; each actual queue span is checked again before execution.
    for (size_t block = 0; block < blocks; block += width(block)) {
        const size_t pages = width(block);
        for (size_t offset = 0; offset < pages; offset += s.fallback ? 1 : pages) {
            const size_t first = prefix+(block+offset)*s.page.tokens;
            const size_t count = std::min((s.fallback ? 1 : pages)*s.page.tokens,padded-first);
            if (!describe(first,count,block%slots+offset,false)) return false;
        }
    }
    // Backend validation checks shape arithmetic; also exclude aliases with unused pool/lease bytes.
    for (auto * tensor : {q,mask,output}) {
        const auto data = uintptr_t(tensor->data);
        const size_t bytes = ggml_nbytes(tensor);
        if (data > UINTPTR_MAX-bytes || overlaps(data,bytes,scratch,region.size) ||
                overlaps(data,bytes,pool,s.binding.capacity) ||
                (s.writer_bytes && overlaps(data,bytes,s.writer_base,s.writer_bytes))) return false;
    }
    if (!cross && !s.writer_lease) release_write_workspace();
    if (!cross && prefix < padded && !s.enter_streamed()) return false;
    s.busy = true; s.valid = false;
    bool succeeded = false;
    struct guard {
        implementation & s; bool & succeeded; bool cross;
        ~guard() { s.busy = false; if (!cross && !succeeded && s.measuring) s.reset_feedback(); }
    } operation{s,succeeded,cross};
    if (!cross) {
        if (!s.refresh(padded)) return false;
    } else {
        s.bytes = s.calls = 0;
        // begin_sequence refreshed the mirror; only authorized resident-tail publications can dirty it.
        if (s.sequence->resident_dirty[s.sequence->next]) {
            if (!s.refresh(padded,layer)) return false;
            s.sequence->resident_dirty[s.sequence->next] = false;
        }
    }
    llama_kv_stream_host_layer host;
    if (!s.content->host()->layer(layer,host)) return false;
    if (s.measuring) {
        if (!cross) s.prepare_feedback(active_tokens,uint32_t(q->ne[1]),{layer});
        else if (s.sequence->next == 0) s.prepare_feedback(active_tokens,s.sequence->queries,s.sequence->layers);
        else if (s.feedback_queries != uint32_t(q->ne[1])) s.sequence->feedback_valid = false;
    }
    s.attention_calls = 0;
    const auto partial = [&](bool second) {
        if (!ops->partial(s.backend,&op,wb,second)) return false;
        ++s.attention_calls;
        return true;
    };
    struct copy_guard {
        const ggml_kv_stream_copy_ops * ops = nullptr;
        void * queue = nullptr;
        ~copy_guard() { if (queue) ops->drain(queue); }
    } copying;
    bool direct_result = false;
    const auto finish = [&] {
        if (cross && s.sequence->plan.front() && s.sequence->plan.front()->layer == s.sequence->next) return false;
        if (!direct_result && !ops->merge(s.backend,output,wb)) return false;
        if (cross) {
            if (++s.sequence->next == s.sequence->layers.size()) s.stop_sequence(true);
            session_call.keep = true;
        } else if (s.measuring) {
            if (overlap && prefix < padded) {
                s.copy_ops->drain(s.copies.get()); copying.queue = nullptr;
                if (decode_feedback && q->ne[1] == 1) s.record_feedback(span_pages); else s.reset_feedback();
            }
            else s.reset_feedback();
        }
        succeeded = true;
        return true;
    };
    const bool prefetch = overlap && prefix < padded;
    // Ordered calls after a replacement must not keep the old pinned cache alive either.
    if (s.copies && s.copy_host != s.content->host()->buffer()) { s.copies.reset(); s.copy_host = nullptr; }
    if (cross) {
        if (!s.sequence->plan.make_ready(s.sequence->next) || !s.fill_sequence(true)) return false;
    } else if (prefetch) {
        if (!s.ensure_copies(slots)) return false;
        if (!(s.measuring ? s.copy_ops->begin_with_feedback(s.copies.get(),decode_feedback && q->ne[1] == 1) : s.copy_ops->begin(s.copies.get()))) return false;
        copying.ops = s.copy_ops; copying.queue = s.copies.get();
    }
    const auto enqueue = [&](size_t block) {
        const size_t first = prefix+block*s.page.tokens;
        const size_t count = std::min(width(block)*s.page.tokens,padded-first), live = std::min(count,active_tokens-first);
        if (!s.copy_ops->enqueue_span(s.copies.get(),block%slots,
                static_cast<const char *>(host.k)+first*s.page.k_token_bytes,
                static_cast<const char *>(host.v)+first*s.page.v_token_bytes,live,count)) return false;
        s.bytes += live*(s.page.k_token_bytes+s.page.v_token_bytes); s.calls += 2;
        return true;
    };
    if (prefetch && !cross) for (size_t block = 0; block < std::min(slots,blocks); block += width(block)) if (!enqueue(block)) return false;
    if ((native || resumed) && !s.fallback && prefix == padded) {
        if (!describe(0,prefix,0,true) || !ops->direct(s.backend,&op)) return false;
        ++s.attention_calls; direct_result = true; return finish();
    }
    // Assemble one full logical layer, without changing its encoded bytes or native reduction order.
    const auto gather = [&](const void * key,const void * value,size_t first,size_t count) {
        for (bool is_value : {false,true}) {
            const size_t stride = is_value ? gathered.v_token_bytes : gathered.k_token_bytes;
            ggml_tensor source{},destination{};
            source.type = destination.type = GGML_TYPE_I8;
            source.buffer = s.binding.buffer; destination.buffer = wb;
            source.data = const_cast<void *>(is_value ? value : key);
            destination.data = reinterpret_cast<char *>(scratch)+(is_value ? gathered.v_offset : 0)+first*stride;
            source.ne[0] = destination.ne[0] = int64_t(count*stride);
            source.nb[0] = destination.nb[0] = 1;
            for (int i = 1; i < 4; ++i) { source.ne[i] = destination.ne[i] = 1; source.nb[i] = destination.nb[i] = count*stride; }
            ggml_backend_tensor_copy_async(s.backend,s.backend,&source,&destination);
        }
        // The next ring reuse must not race this encoded-device copy.
        ggml_backend_synchronize(s.backend); return true;
    };
    const auto finish_gather = [&] {
        k = s.descriptor(layer,false,padded); v = s.descriptor(layer,true,padded);
        k.buffer = v.buffer = wb;
        k.data = reinterpret_cast<void *>(scratch); v.data = reinterpret_cast<void *>(scratch+gathered.v_offset);
        k.nb[1] = gathered.k_token_bytes; k.nb[2] = gathered.k_row_bytes; k.nb[3] = gathered.k_bytes;
        v.nb[1] = gathered.v_token_bytes; v.nb[2] = gathered.v_row_bytes; v.nb[3] = gathered.v_bytes;
        slice = *mask; slice.ne[0] = int64_t(padded);
        op = *output; op.op = GGML_OP_FLASH_ATTN_EXT;
        std::memset(op.src,0,sizeof(op.src)); std::memset(op.op_params,0,sizeof(op.op_params));
        std::memcpy(op.op_params,&scale,sizeof(scale)); ggml_flash_attn_ext_set_prec(&op,GGML_PREC_F32);
        op.src[0] = q; op.src[1] = &k; op.src[2] = &v; op.src[3] = &slice;
        if (!ops->direct(s.backend,&op)) return false;
        ++s.attention_calls; direct_result = true; return finish();
    };
    const auto convert_inputs = [&] {
        return !s.fallback || (ops->convert(s.backend,&k,&ck) && ops->convert(s.backend,&v,&cv));
    };
    if (resumed) {
        if (prefix && (!describe(0,prefix,0,true) || !ops->resume(s.backend,&op,wb,resume_plan,padded,0,false))) return false;
        s.attention_calls += prefix != 0;
    } else if (native) {
        if (prefix && !gather(s.roots[layer].first->data,s.roots[layer].second->data,0,prefix)) return false;
        if (prefix == padded) return finish_gather();
    } else if (prefix && !s.fallback) {
        if (!describe(0,prefix,0,true) || !partial(false)) return false;
    } else {
        if (!ops->clear(s.backend,output,wb,false)) return false;
        for (size_t first = 0; first < prefix; first += resident_chunk) {
            if (!describe(first,std::min(resident_chunk,prefix-first),0,true) || !convert_inputs() ||
                    !partial(true)) return false;
            if (first+std::min(resident_chunk,prefix-first) == padded) return finish();
            if (!ops->fold(s.backend,output,wb)) return false;
        }
    }
    for (size_t block = 0; block < blocks;) {
        const size_t first = prefix+block*s.page.tokens;
        size_t pages = width(block), slot = block%slots;
        if (cross) {
            const auto * request = s.sequence->plan.front();
            if (!request || request->layer != s.sequence->next || request->first != first || !request->submitted) return false;
            pages = request->pages; slot = request->slot;
        }
        const size_t count = std::min(pages*s.page.tokens,padded-first);
        const size_t live = std::min(count,active_tokens-first);
        if (!prefetch) for (int value = 0; value < 2; ++value) {
            auto staged = *s.stage[value];
            staged.data = static_cast<char *>(staged.data)+slot*(value ? s.page.v_bytes : s.page.k_bytes);
            staged.ne[0] = int64_t(count)*s.binding.config.shape.heads*(value ? s.binding.config.shape.head_dim_v : s.binding.config.shape.head_dim_k);
            staged.nb[1] = staged.nb[2] = staged.nb[3] = ggml_row_size(staged.type,staged.ne[0]);
            const size_t stride = value ? s.page.v_token_bytes : s.page.k_token_bytes;
            const auto * source = static_cast<const char *>(value ? host.v : host.k);
            if (count > live) ggml_backend_tensor_memset(&staged,0,live*stride,(count-live)*stride);
            ggml_backend_tensor_set(&staged,source+first*stride,0,live*stride);
            s.bytes += live*stride; ++s.calls;
        }
        // Fallback shares the batched transfer, but never grows its one-page conversion workspace.
        for (size_t offset = 0; offset < pages; offset += s.fallback ? 1 : pages) {
            const size_t consumed = s.fallback ? 1 : pages;
            const size_t token = first+offset*s.page.tokens;
            const size_t tokens = std::min(consumed*s.page.tokens,padded-token);
            if (!describe(token,tokens,slot+offset,false)) return false;
            if (prefetch) {
                if (s.measuring) {
                    if (!s.copy_ops->acquire_span(s.copies.get(),slot+offset,consumed)) return false;
                } else for (size_t i = 0; i < consumed; ++i)
                    if (!s.copy_ops->acquire(s.copies.get(),slot+offset+i)) return false;
            }
            if (!native && !resumed && !convert_inputs()) return false;
            const auto recycle = [&] {
                for (size_t i = 0; i < consumed; ++i) {
                    // The current partial/convert callbacks synchronize before returning.
                    const bool released = cross && s.copy_ops->version >= 3 && s.copy_ops->release_completed ?
                        s.copy_ops->release_completed(s.copies.get(),slot+offset+i) : s.copy_ops->release(s.copies.get(),slot+offset+i);
                    if (!released) return false;
                }
                if (cross) return s.sequence->plan.consume(consumed) && s.fill_sequence();
                // A contiguous DMA must wait for the complete destination run to become reusable.
                return offset+consumed != pages || block+slots >= blocks || enqueue(block+slots);
            };
            if (resumed) {
                const bool last = token+tokens == padded;
                if (!ops->resume(s.backend,&op,wb,resume_plan,padded,token,last)) return false;
                ++s.attention_calls;
                if (prefetch && !recycle()) return false;
                if (last) { direct_result = true; return finish(); }
                continue;
            }
            if (native) {
                if (!gather(k.data,v.data,token,tokens) || (prefetch && !recycle())) return false;
                if (token+tokens == padded) return finish_gather();
                continue;
            }
            if (prefetch && s.fallback && !recycle()) return false;
            if (!partial(true)) return false;
            if (prefetch && !s.fallback && !recycle()) return false;
            if (token+tokens == padded) return finish();
            if (!ops->fold(s.backend,output,wb)) return false;
        }
        block += pages;
    }
    return ops->clear(s.backend,output,wb,true) && finish();
}

// Expose exact submitted tile/copy counts and configured scratch bounds for tests and diagnostics.
llama_kv_stream_write_stats llama_kv_stream_resident::last_write_stats() const noexcept { return impl->write_stats; }

// Drop the single cached graph/source reference before a caller reclaims prefill input workspace.
void llama_kv_stream_resident::release_write_workspace() {
    if (impl->busy || impl->sequence || impl->capturing()) return;
    ggml_backend_synchronize(impl->backend);
    impl->writer.reset();
    impl->writer_lease.reset();
    impl->writer_base = impl->writer_bytes = 0;
}
