#include "llama-kv-stream-layer-lease.h"
#include "llama-kv-stream-logical-cache.h"
#include "ggml-cpp.h"

#include "llama-impl.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <limits>
#include <mutex>
#include <new>
#include <utility>
#include <vector>

using pool_ptr = std::unique_ptr<ggml_backend_memory_lease, decltype(&ggml_backend_memory_lease_free)>;

static bool multiply(size_t a, size_t b, size_t & output) {
    if (b && a > SIZE_MAX/b) return false;
    output = a*b;
    return true;
}

static size_t ceil_div(size_t value, size_t divisor) {
    return value/divisor + (value%divisor != 0);
}

struct layer_lease_state;

struct layer_reservation {
    std::shared_ptr<layer_lease_state> owner;
    uint32_t layer = 0;
    size_t active_tokens = 0;
    size_t resident_tokens = 0;
    size_t ring_first = SIZE_MAX;
    size_t ring_slots = 0;
    uint64_t layout_revision = 0;
    uint64_t content_generation = 0;
    bool published = false;
    bool populating = false, populated = false;
    bool resident_mirror = false;
    size_t populated_tokens = 0;
    llama_kv_stream_population_stats population;

    ~layer_reservation();
};

struct layer_lease_state {
    mutable std::mutex mutex;
    pool_ptr pool{nullptr, ggml_backend_memory_lease_free};
    llama_kv_stream_layer_lease_layout identity;
    llama_kv_stream_policy_layout layout;
    std::vector<uint8_t> occupied;
    std::vector<std::weak_ptr<layer_reservation>> layers;
    size_t reservations = 0;
    size_t leases = 0;
    size_t guards = 0;
    bool closed = false;
};

llama_kv_stream_ring_guard::llama_kv_stream_ring_guard(
        std::shared_ptr<layer_lease_state> state, std::vector<uint8_t> blocked) :
    state(std::move(state)), blocked(std::move(blocked)) {}

llama_kv_stream_ring_guard::~llama_kv_stream_ring_guard() {
    std::lock_guard<std::mutex> lock(state->mutex);
    GGML_ASSERT(state->guards > 0);
    --state->guards;
}

const std::vector<uint8_t> & llama_kv_stream_ring_guard::blocked_slots() const noexcept { return blocked; }
const ggml_kv_stream_layout & llama_kv_stream_ring_guard::ring_layout() const noexcept {
    return state->layout.ring;
}

ggml_backend_buffer_t llama_kv_stream_ring_guard::pool_buffer() const noexcept {
    return ggml_backend_memory_lease_buffer(state->pool.get());
}


layer_reservation::~layer_reservation() {
    if (!published || !owner) return;
    std::lock_guard<std::mutex> lock(owner->mutex);
    for (size_t i = 0; i < ring_slots; ++i) {
        GGML_ASSERT(ring_first + i < owner->occupied.size() && owner->occupied[ring_first + i]);
        owner->occupied[ring_first + i] = 0;
    }
    GGML_ASSERT(owner->reservations > 0);
    --owner->reservations;
}

struct llama_kv_stream_complete_layer_lease {
    std::atomic<uint32_t> references{1};
    std::shared_ptr<layer_reservation> reservation;
    ggml_kv_stream_span_plan_t plan = nullptr;
    size_t active_tokens = 0;
    uint64_t content_generation = 0;

    ~llama_kv_stream_complete_layer_lease() {
        ggml_kv_stream_span_plan_free(plan);
        auto state = reservation ? reservation->owner : nullptr;
        if (state) {
            std::lock_guard<std::mutex> lock(state->mutex);
            GGML_ASSERT(state->leases > 0);
            --state->leases;
        }
    }
};

struct llama_kv_stream_layer_lease_owner::implementation {
    std::shared_ptr<layer_lease_state> state;
};

// Validate and materialize a complete immutable layout before publishing any owner state.
static bool materialize(
        ggml_backend_memory_lease_t pool,
        const llama_kv_stream_layer_lease_layout & identity,
        llama_kv_stream_policy_layout & output) {
    if (!pool || !identity.active_tokens || !identity.layout_revision || !identity.content_generation)
        return false;
    const size_t placement = identity.placement_tokens ? identity.placement_tokens : identity.active_tokens;
    if (placement > identity.active_tokens) return false;
    auto * buffer = ggml_backend_memory_lease_buffer(pool);
    ggml_backend_memory_region region;
    if (!buffer || !ggml_backend_memory_lease_get_region(pool, &region) ||
            ggml_backend_buffer_get_size(buffer) != region.size ||
            identity.policy.pool_bytes > region.size) return false;
    llama_kv_stream_policy_layout next;
    if (llama_kv_stream_policy_layout_make(
            identity.policy, identity.state, placement, next).status !=
            llama_kv_stream_policy_status::success) return false;
    output = std::move(next);
    return true;
}

llama_kv_stream_layer_lease_owner::llama_kv_stream_layer_lease_owner() = default;

llama_kv_stream_layer_lease_owner::~llama_kv_stream_layer_lease_owner() {
    close();
}

std::unique_ptr<llama_kv_stream_layer_lease_owner> llama_kv_stream_layer_lease_owner::create(
        ggml_backend_memory_lease_t pool,
        const llama_kv_stream_layer_lease_layout & identity) {
    llama_kv_stream_policy_layout layout;
    if (!materialize(pool, identity, layout)) return {};
    try {
        auto result = std::unique_ptr<llama_kv_stream_layer_lease_owner>(
            new llama_kv_stream_layer_lease_owner);
        result->impl = std::make_unique<implementation>();
        auto state = std::make_shared<layer_lease_state>();
        state->pool.reset(ggml_backend_memory_lease_retain(pool));
        if (!state->pool) return {};
        state->identity = identity;
        state->layout = std::move(layout);
        state->occupied.assign(identity.state.ring_slots, 0);
        state->layers.resize(identity.policy.layers);
        result->impl->state = std::move(state);
        return result;
    } catch (const std::bad_alloc &) {
        return {};
    }
}

// Find one nonwrapping physical run so the suffix is one contiguous K span and one contiguous V span.
static bool find_run(const std::vector<uint8_t> & occupied, size_t count, size_t & first) {
    if (count == 0) {
        first = SIZE_MAX;
        return true;
    }
    if (count > occupied.size()) return false;
    for (size_t candidate = 0; candidate <= occupied.size() - count; ++candidate) {
        bool free = true;
        for (size_t i = 0; i < count; ++i) free &= !occupied[candidate + i];
        if (free) {
            first = candidate;
            return true;
        }
    }
    return false;
}

// Build a query-specific plan over one shared physical reservation.
static ggml_kv_stream_span_plan_t build_plan(
        const layer_lease_state & state,
        const layer_reservation & reservation,
        uint32_t query_tokens, size_t active_tokens) {
    const auto & layer = state.layout.layers[reservation.layer];
    const size_t page_tokens = size_t(state.identity.policy.shape.page_tokens);
    std::vector<ggml_kv_stream_span_source> sources;
    try {
        sources.reserve(2);
        const size_t resident_tokens = std::min(active_tokens, reservation.resident_tokens);
        if (resident_tokens) {
            sources.push_back({
                state.pool.get(), state.pool.get(), 0, resident_tokens,
                layer.offset, layer.offset + layer.planes.v_offset});
        }
        const size_t streamed = active_tokens - resident_tokens;
        if (streamed) {
            size_t ring_token, k_offset, v_offset;
            if (!multiply(reservation.ring_first, page_tokens, ring_token) ||
                    !multiply(ring_token, state.layout.ring.k_token_bytes, k_offset) ||
                    !multiply(ring_token, state.layout.ring.v_token_bytes, v_offset) ||
                    v_offset > SIZE_MAX - state.layout.ring.v_offset) return nullptr;
            v_offset += state.layout.ring.v_offset;
            sources.push_back({
                state.pool.get(), state.pool.get(), resident_tokens, streamed,
                k_offset, v_offset});
        }
        ggml_kv_stream_span_plan_t plan = nullptr;
        return ggml_kv_stream_span_plan_make(
            state.identity.policy.shape, sources.data(), sources.size(),
            active_tokens, query_tokens, plan).status ==
            ggml_kv_stream_status::success ? plan : nullptr;
    } catch (const std::bad_alloc &) {
        return nullptr;
    }
}

std::shared_ptr<llama_kv_stream_ring_guard> llama_kv_stream_layer_lease_owner::hold_ring() {
    if (!impl || !impl->state) return {};
    auto state = impl->state;
    std::lock_guard<std::mutex> lock(state->mutex);
    if (state->closed || state->guards == SIZE_MAX) return {};
    try {
        auto result = std::shared_ptr<llama_kv_stream_ring_guard>(
            new llama_kv_stream_ring_guard(state, state->occupied));
        ++state->guards;
        return result;
    } catch (const std::bad_alloc &) {
        return {};
    }
}


llama_kv_stream_complete_layer_lease_t llama_kv_stream_layer_lease_owner::acquire(
        const llama_kv_stream_complete_layer_request & request) {
    if (!impl || !impl->state) return nullptr;
    auto state = impl->state;
    std::lock_guard<std::mutex> lock(state->mutex);
    const size_t active = request.active_tokens ? request.active_tokens : state->identity.active_tokens;
    if (state->closed || request.layer >= state->layout.layers.size() ||
            !request.query_tokens || !active || active > state->identity.active_tokens ||
            request.query_tokens > active ||
            request.layout_revision != state->identity.layout_revision ||
            request.content_generation != state->identity.content_generation ||
            request.cache_id != state->layout.layers[request.layer].cache_id) return nullptr;

    std::shared_ptr<layer_reservation> reservation = state->layers[request.layer].lock();
    bool created = false;
    if (!reservation) {
        if (state->guards) return nullptr;
        const auto & layer = state->layout.layers[request.layer];
        const size_t page_tokens = size_t(state->identity.policy.shape.page_tokens);
        size_t capacity;
        if (!multiply(size_t(layer.capacity_pages), page_tokens, capacity)) return nullptr;
        const size_t resident_tokens = std::min(state->identity.active_tokens, capacity);
        const size_t ring_slots = ceil_div(state->identity.active_tokens - resident_tokens, page_tokens);
        size_t ring_first;
        if (!find_run(state->occupied, ring_slots, ring_first)) return nullptr;
        try {
            reservation = std::make_shared<layer_reservation>();
        } catch (const std::bad_alloc &) {
            return nullptr;
        }
        reservation->owner = state;
        reservation->layer = request.layer;
        reservation->active_tokens = state->identity.active_tokens;
        reservation->resident_tokens = resident_tokens;
        reservation->ring_first = ring_first;
        reservation->ring_slots = ring_slots;
        reservation->layout_revision = state->identity.layout_revision;
        reservation->content_generation = state->identity.content_generation;
        created = true;
    }

    ggml_kv_stream_span_plan_t plan = build_plan(*state, *reservation, request.query_tokens, active);
    if (!plan) return nullptr;
    auto * lease = new (std::nothrow) llama_kv_stream_complete_layer_lease;
    if (!lease) {
        ggml_kv_stream_span_plan_free(plan);
        return nullptr;
    }
    lease->content_generation = request.content_generation;
    lease->reservation = reservation;
    lease->plan = plan;
    lease->active_tokens = active;
    if (created) {
        for (size_t i = 0; i < reservation->ring_slots; ++i) {
            GGML_ASSERT(reservation->ring_first + i < state->occupied.size() &&
                !state->occupied[reservation->ring_first + i]);
            state->occupied[reservation->ring_first + i] = 1;
        }
        reservation->published = true;
        state->layers[request.layer] = reservation;
        ++state->reservations;
    }
    ++state->leases;
    return lease;
}

static bool same_population_shape(const ggml_kv_stream_shape & a, const ggml_kv_stream_shape & b) {
    return a.type_k == b.type_k && a.type_v == b.type_v &&
        a.head_dim_k == b.head_dim_k && a.head_dim_v == b.head_dim_v &&
        a.heads == b.heads && a.page_tokens == b.page_tokens && a.alignment == b.alignment;
}

static bool populate_layer(ggml_backend_t backend, llama_kv_stream_complete_layer_lease_t lease,
        const llama_kv_stream_logical_cache & cache, llama_kv_stream_population_stats & stats, bool reuse_resident) {
    if (!backend || !lease || !lease->reservation) return false;
    const auto & reservation = *lease->reservation;
    const auto & state = *reservation.owner;
    const auto & physical = state.layout.layers[reservation.layer];
    const auto host = cache.host();
    const auto frontiers = cache.frontiers();
    if (!host || !physical.cache_id || physical.cache_id != cache.identity().id ||
            physical.cache_layer >= host->config().layers ||
            !same_population_shape(host->config().shape, state.identity.policy.shape) ||
            cache.identity().generation != reservation.content_generation ||
            cache.content()->generation() != reservation.content_generation ||
            cache.tokens() != lease->active_tokens ||
            frontiers.reserved != cache.tokens() || frontiers.host != cache.tokens() ||
            frontiers.committed != cache.tokens()) return false;
    llama_kv_stream_host_layer source;
    if (!host->layer(physical.cache_layer, source)) return false;
    ggml_kv_stream_span_plan_view view;
    if (!ggml_kv_stream_span_plan_get_view(lease->plan, view) || !view.count || view.count > 2 ||
            view.active_tokens != lease->active_tokens) return false;

    ggml_context_ptr context(ggml_init({16384, nullptr, true}));
    if (!context) return false;
    struct upload { ggml_tensor * tensor = nullptr; const void * source = nullptr; size_t bytes = 0; };
    std::array<upload, 4> uploads{};
    size_t count = 0;
    llama_kv_stream_population_stats next;
    const auto add = [&](ggml_backend_buffer_t buffer, size_t offset, const void * data, size_t bytes) {
        if (!buffer || !data || !bytes || bytes > INT64_MAX || bytes > SIZE_MAX - next.bytes) return false;
        auto * type = ggml_backend_buffer_get_type(buffer);
        auto * device = ggml_backend_buft_get_device(type);
        if (!ggml_backend_supports_buft(backend, type) || (device && device != ggml_backend_get_device(backend)))
            return false;
        const size_t capacity = ggml_backend_buffer_get_size(buffer);
        if (offset > capacity || bytes > capacity - offset) return false;
        auto * tensor = ggml_new_tensor_1d(context.get(), GGML_TYPE_I8, int64_t(bytes));
        if (!tensor || ggml_backend_buffer_get_alloc_size(buffer, tensor) > capacity - offset) return false;
        const auto base = reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(buffer));
        if (!base || offset > UINTPTR_MAX - base || bytes > UINTPTR_MAX - base - offset ||
                ggml_backend_tensor_alloc(buffer, tensor, reinterpret_cast<void *>(base + offset)) != GGML_STATUS_SUCCESS)
            return false;
        uploads[count++] = {tensor, data, bytes};
        next.bytes += bytes;
        ++next.calls;
        return true;
    };
    const auto & host_layout = host->layout();
    for (size_t i = 0; i < view.count; ++i) {
        const auto & span = view.spans[i];
        if (reuse_resident && span.token_begin < reservation.resident_tokens) {
            if (span.token_begin != 0 || span.tokens > reservation.resident_tokens) return false;
            const std::vector<llama_kv_stream_rows> ranges{
                {physical.cache_layer,ggml_kv_stream_operand::k,0,span.tokens},
                {physical.cache_layer,ggml_kv_stream_operand::v,0,span.tokens}};
            const bool copied = cache.content()->flush(ranges,[&](const llama_kv_stream_copy_span & dirty) {
                const bool value=dirty.rows.operand == ggml_kv_stream_operand::v;
                const size_t stride=value ? host_layout.v_token_bytes : host_layout.k_token_bytes;
                auto * buffer=value ? span.v_buffer : span.k_buffer;
                const size_t base=value ? span.v_offset : span.k_offset;
                size_t delta;
                if (!multiply(dirty.rows.first,stride,delta) || delta > SIZE_MAX-base ||
                        dirty.bytes > INT64_MAX || dirty.bytes > SIZE_MAX-next.bytes || next.calls == SIZE_MAX) return false;
                const size_t offset=base+delta, capacity=ggml_backend_buffer_get_size(buffer);
                if (offset > capacity || dirty.bytes > capacity-offset) return false;
                ggml_tensor tensor{};
                tensor.type=GGML_TYPE_I8;
                tensor.ne[0]=int64_t(dirty.bytes); tensor.nb[0]=1;
                for (int axis=1; axis<4; ++axis) { tensor.ne[axis]=1; tensor.nb[axis]=dirty.bytes; }
                auto * address=static_cast<char *>(ggml_backend_buffer_get_base(buffer))+offset;
                if (ggml_backend_tensor_alloc(buffer,&tensor,address) != GGML_STATUS_SUCCESS) return false;
                ggml_backend_tensor_set_async(backend,&tensor,dirty.data,0,dirty.bytes);
                ggml_backend_synchronize(backend);
                next.bytes+=dirty.bytes; ++next.calls;
                return true;
            });
            if (!copied) return false;
            continue;
        }
        size_t k_first, v_first, k_bytes, v_bytes;
        if (!multiply(span.token_begin, host_layout.k_token_bytes, k_first) ||
                !multiply(span.token_begin, host_layout.v_token_bytes, v_first) ||
                !multiply(span.tokens, host_layout.k_token_bytes, k_bytes) ||
                !multiply(span.tokens, host_layout.v_token_bytes, v_bytes) ||
                k_first > host_layout.k_bytes || k_bytes > host_layout.k_bytes - k_first ||
                v_first > host_layout.v_bytes || v_bytes > host_layout.v_bytes - v_first ||
                !add(span.k_buffer, span.k_offset, static_cast<const char *>(source.k) + k_first, k_bytes) ||
                !add(span.v_buffer, span.v_offset, static_cast<const char *>(source.v) + v_first, v_bytes))
            return false;
    }
    try {
        for (size_t i = 0; i < count; ++i) {
            ggml_backend_tensor_set_async(backend, uploads[i].tensor, uploads[i].source, 0, uploads[i].bytes);
        }
        if (count) ggml_backend_synchronize(backend);
    } catch (...) {
        try { ggml_backend_synchronize(backend); } catch (...) {}
        return false;
    }
    stats = next;
    return true;
}

llama_kv_stream_complete_layer_lease_t llama_kv_stream_layer_lease_owner::acquire_populated(
        ggml_backend_t backend, const llama_kv_stream_complete_layer_request & request,
        const llama_kv_stream_logical_cache & cache, bool reuse_resident) {
    if (!impl || !impl->state || !backend) return nullptr;
    auto * buffer = ggml_backend_memory_lease_buffer(impl->state->pool.get());
    if (!buffer) return nullptr;
    auto * type = ggml_backend_buffer_get_type(buffer);
    auto * device = ggml_backend_buft_get_device(type);
    if (!ggml_backend_supports_buft(backend, type) ||
            (device && device != ggml_backend_get_device(backend))) return nullptr;
    auto * lease = acquire(request);
    if (!lease) return nullptr;
    auto reservation = lease->reservation;
    auto state = reservation->owner;
    bool populate = false;
    bool ready = false;
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        ready = reservation->populated;
        if (!ready && !reservation->populating && !state->closed) {
            reservation->populating = true;
            populate = true;
        }
    }
    if (ready) {
        const auto frontiers = cache.frontiers();
        if (cache.identity().id == state->layout.layers[request.layer].cache_id &&
                cache.identity().generation == reservation->content_generation &&
                cache.content()->generation() == reservation->content_generation &&
                cache.tokens() == lease->active_tokens &&
                reservation->populated_tokens == lease->active_tokens &&
                frontiers.reserved == cache.tokens() && frontiers.host == cache.tokens() &&
                frontiers.committed == cache.tokens()) return lease;
        llama_kv_stream_complete_layer_lease_free(lease);
        return nullptr;
    }
    if (!populate) {
        llama_kv_stream_complete_layer_lease_free(lease);
        return nullptr;
    }
    llama_kv_stream_population_stats stats;
    bool copied=false;
    try { copied=populate_layer(backend,lease,cache,stats,reuse_resident); }
    catch (...) { try { ggml_backend_synchronize(backend); } catch (...) {} }
    const auto frontiers = cache.frontiers();
    const bool current = copied && cache.identity().generation == reservation->content_generation &&
        cache.content()->generation() == reservation->content_generation &&
        cache.tokens() == lease->active_tokens &&
        frontiers.reserved == cache.tokens() && frontiers.host == cache.tokens() &&
        frontiers.committed == cache.tokens();
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        reservation->populating = false;
        if (current && !state->closed) {
            reservation->population = stats;
            reservation->populated = true;
            reservation->populated_tokens = lease->active_tokens;
            reservation->resident_mirror = reuse_resident;
            ready = true;
        }
    }
    if (!ready) {
        llama_kv_stream_complete_layer_lease_free(lease);
        return nullptr;
    }
    return lease;
}

// A completed tail copy acknowledges only its resident rows; ring contents remain lease-local.
static bool acknowledge_resident_tail(const layer_reservation & reservation,
        const llama_kv_stream_logical_cache & cache,size_t first,size_t next) {
    if (!reservation.resident_mirror || first >= reservation.resident_tokens) return true;
    const size_t end=std::min(next,reservation.resident_tokens);
    const auto layer=reservation.owner->layout.layers[reservation.layer].cache_layer;
    return cache.content()->flush({{layer,ggml_kv_stream_operand::k,first,end-first},
        {layer,ggml_kv_stream_operand::v,first,end-first}},[](const auto &) { return true; });
}

// Publish only a newly committed suffix into the already protected resident/ring
// placement. The old plan remains valid until the generation is advanced atomically.
bool llama_kv_stream_layer_lease_owner::publish_tail(
        ggml_backend_t backend, llama_kv_stream_complete_layer_lease_t previous,
        const llama_kv_stream_logical_cache & cache, llama_kv_stream_population_stats & delta) {
    if (!impl || !impl->state || !backend || !previous || !previous->reservation) return false;
    auto state = impl->state;
    auto reservation = previous->reservation;
    if (reservation->owner != state) return false;
    const auto host = cache.host();
    if (!host || !same_population_shape(host->config().shape, state->identity.policy.shape) ||
            reservation->layer >= state->layout.layers.size() ||
            state->layout.layers[reservation->layer].cache_id != cache.identity().id ||
            state->layout.layers[reservation->layer].cache_layer >= host->config().layers)
        return false;
    const auto frontiers = cache.frontiers();
    const size_t next_tokens = cache.tokens();
    const uint64_t next_generation = cache.identity().generation;
    if (next_tokens <= previous->active_tokens ||
            next_tokens > reservation->active_tokens ||
            next_generation <= previous->content_generation ||
            cache.content()->generation() != next_generation ||
            frontiers.reserved != next_tokens || frontiers.host != next_tokens ||
            frontiers.committed != next_tokens || frontiers.device != 0) return false;
    auto * pool = ggml_backend_memory_lease_buffer(state->pool.get());
    auto * type = pool ? ggml_backend_buffer_get_type(pool) : nullptr;
    auto * device = type ? ggml_backend_buft_get_device(type) : nullptr;
    if (!pool || !ggml_backend_supports_buft(backend, type) ||
            (device && device != ggml_backend_get_device(backend))) return false;
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        if (state->closed || !reservation->populated || reservation->populating ||
                reservation->populated_tokens != previous->active_tokens ||
                reservation->content_generation != previous->content_generation ||
                state->identity.content_generation != previous->content_generation)
            return false;
        reservation->populating = true;
    }
    struct population_gate {
        std::shared_ptr<layer_reservation> reservation;
        bool active = true;
        ~population_gate() {
            if (!active) return;
            std::lock_guard<std::mutex> lock(reservation->owner->mutex);
            reservation->populating = false;
        }
    } gate{reservation};
    using plan_ptr = std::unique_ptr<ggml_kv_stream_span_plan, decltype(&ggml_kv_stream_span_plan_free)>;
    plan_ptr plan(build_plan(*state, *reservation, 1, next_tokens), ggml_kv_stream_span_plan_free);
    ggml_kv_stream_span_plan_view view;
    if (!plan || !ggml_kv_stream_span_plan_get_view(plan.get(), view) ||
            !view.count || view.count > 2 || view.active_tokens != next_tokens) return false;
    llama_kv_stream_host_layer source;
    if (!host->layer(state->layout.layers[reservation->layer].cache_layer, source)) return false;
    ggml_context_ptr context(ggml_init({16384, nullptr, true}));
    if (!context) return false;
    struct upload { ggml_tensor * tensor = nullptr; const void * source = nullptr; size_t bytes = 0; };
    std::array<upload, 4> uploads{};
    size_t count = 0;
    llama_kv_stream_population_stats next;
    const auto add = [&](ggml_backend_buffer_t buffer, size_t offset, const void * data, size_t bytes) {
        if (!buffer || !data || !bytes || count >= uploads.size() || bytes > INT64_MAX ||
                bytes > SIZE_MAX - next.bytes) return false;
        const size_t capacity = ggml_backend_buffer_get_size(buffer);
        if (offset > capacity || bytes > capacity - offset) return false;
        auto * tensor = ggml_new_tensor_1d(context.get(), GGML_TYPE_I8, int64_t(bytes));
        if (!tensor || ggml_backend_buffer_get_alloc_size(buffer, tensor) > capacity - offset) return false;
        auto * base = static_cast<char *>(ggml_backend_buffer_get_base(buffer));
        if (!base || ggml_backend_tensor_alloc(buffer, tensor, base + offset) != GGML_STATUS_SUCCESS)
            return false;
        uploads[count++] = {tensor, data, bytes};
        next.bytes += bytes;
        ++next.calls;
        return true;
    };
    const auto & host_layout = host->layout();
    for (size_t i = 0; i < view.count; ++i) {
        const auto & span = view.spans[i];
        const size_t first = std::max(previous->active_tokens, span.token_begin);
        const size_t end = std::min(next_tokens, span.token_begin + span.tokens);
        if (first >= end) continue;
        const size_t offset_tokens = first - span.token_begin;
        size_t k_delta, v_delta, k_first, v_first, k_bytes, v_bytes;
        if (!multiply(offset_tokens, host_layout.k_token_bytes, k_delta) ||
                !multiply(offset_tokens, host_layout.v_token_bytes, v_delta) ||
                !multiply(first, host_layout.k_token_bytes, k_first) ||
                !multiply(first, host_layout.v_token_bytes, v_first) ||
                !multiply(end - first, host_layout.k_token_bytes, k_bytes) ||
                !multiply(end - first, host_layout.v_token_bytes, v_bytes) ||
                k_delta > SIZE_MAX - span.k_offset ||
                v_delta > SIZE_MAX - span.v_offset ||
                k_first > host_layout.k_bytes || k_bytes > host_layout.k_bytes - k_first ||
                v_first > host_layout.v_bytes || v_bytes > host_layout.v_bytes - v_first ||
                !add(span.k_buffer, span.k_offset + k_delta,
                    static_cast<const char *>(source.k) + k_first, k_bytes) ||
                !add(span.v_buffer, span.v_offset + v_delta,
                    static_cast<const char *>(source.v) + v_first, v_bytes))
            return false;
    }
    if (!count) return false;
    try {
        ggml_backend_synchronize(backend);
        for (size_t i = 0; i < count; ++i)
            ggml_backend_tensor_set_async(backend, uploads[i].tensor, uploads[i].source, 0, uploads[i].bytes);
        ggml_backend_synchronize(backend);
    } catch (...) {
        try { ggml_backend_synchronize(backend); } catch (...) {}
        return false;
    }
    const auto current = cache.frontiers();
    if (cache.tokens() != next_tokens || cache.identity().generation != next_generation ||
            cache.content()->generation() != next_generation ||
            current.reserved != next_tokens || current.host != next_tokens ||
            current.committed != next_tokens || current.device != 0) return false;
    if (!acknowledge_resident_tail(*reservation,cache,previous->active_tokens,next_tokens)) return false;
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        if (state->closed || reservation->content_generation != previous->content_generation ||
                state->identity.content_generation != previous->content_generation ||
                reservation->population.bytes > SIZE_MAX - next.bytes ||
                reservation->population.calls > SIZE_MAX - next.calls) return false;
        reservation->content_generation = next_generation;
        reservation->populated_tokens = next_tokens;
        reservation->population.bytes += next.bytes;
        reservation->population.calls += next.calls;
        state->identity.content_generation = next_generation;
        reservation->populating = false;
    }
    gate.active = false;
    delta = next;
    return true;
}

// Queue only newly encoded rows into the protected resident/ring placement.
// The writer's completion event fences these D2D copies as well as its host copy.
bool llama_kv_stream_layer_lease_owner::stage_tail_async(
        ggml_backend_t backend, llama_kv_stream_complete_layer_lease_t previous,
        size_t first, bool value, const ggml_tensor * encoded, size_t row, size_t count,
        llama_kv_stream_population_stats & staged) {
    if (!impl || !impl->state || !backend || !previous || !previous->reservation ||
            !encoded || !encoded->buffer || !encoded->data || !count ||
            previous->reservation->owner != impl->state) return false;
    auto state = impl->state;
    auto reservation = previous->reservation;
    const auto & shape = state->identity.policy.shape;
    const size_t stride = ggml_row_size(ggml_type(value ? shape.type_v : shape.type_k),
        int64_t(size_t(value ? shape.head_dim_v : shape.head_dim_k)*shape.heads));
    if (!stride || encoded->type != (value ? shape.type_v : shape.type_k) ||
            first != previous->active_tokens || row > SIZE_MAX - first ||
            count > SIZE_MAX - first - row || count > SIZE_MAX/stride ||
            count*stride > ggml_nbytes(encoded) ||
            staged.bytes > SIZE_MAX - count*stride || staged.calls == SIZE_MAX) return false;
    const size_t begin = first + row, end = begin + count;
    {
        std::lock_guard<std::mutex> lock(state->mutex);
        if (state->closed || !reservation->populated || reservation->populating ||
                reservation->populated_tokens != first || end > reservation->active_tokens ||
                reservation->content_generation != previous->content_generation ||
                state->identity.content_generation != previous->content_generation)
            return false;
    }
    using plan_ptr = std::unique_ptr<ggml_kv_stream_span_plan, decltype(&ggml_kv_stream_span_plan_free)>;
    plan_ptr plan(build_plan(*state, *reservation, 1, end), ggml_kv_stream_span_plan_free);
    ggml_kv_stream_span_plan_view view;
    if (!plan || !ggml_kv_stream_span_plan_get_view(plan.get(), view) ||
            !view.count || view.count > 2 || view.active_tokens != end) return false;
    size_t copied = 0, calls = 0;
    try {
        for (size_t i = 0; i < view.count; ++i) {
            const auto & span = view.spans[i];
            const size_t part_first = std::max(begin, span.token_begin);
            const size_t part_end = std::min(end, span.token_begin + span.tokens);
            if (part_first >= part_end) continue;
            const size_t bytes = (part_end - part_first)*stride;
            auto * destination = value ? span.v_buffer : span.k_buffer;
            const size_t offset = (value ? span.v_offset : span.k_offset) +
                (part_first - span.token_begin)*stride;
            if (!destination || offset > ggml_backend_buffer_get_size(destination) ||
                    bytes > ggml_backend_buffer_get_size(destination) - offset)
                return false;
            ggml_tensor src{}, dst{};
            src.type = dst.type = GGML_TYPE_I8;
            src.buffer = encoded->buffer;
            dst.buffer = destination;
            src.data = static_cast<char *>(encoded->data) + (part_first - begin)*stride;
            dst.data = static_cast<char *>(ggml_backend_buffer_get_base(destination)) + offset;
            src.ne[0] = dst.ne[0] = int64_t(bytes);
            src.nb[0] = dst.nb[0] = 1;
            for (int axis = 1; axis < 4; ++axis) {
                src.ne[axis] = dst.ne[axis] = 1;
                src.nb[axis] = dst.nb[axis] = bytes;
            }
            ggml_backend_tensor_copy_async(backend, backend, &src, &dst);
            copied += bytes;
            ++calls;
        }
    } catch (...) {
        try { ggml_backend_synchronize(backend); } catch (...) {}
        return false;
    }
    if (copied != count*stride || !calls || staged.calls > SIZE_MAX - calls) return false;
    staged.bytes += copied;
    staged.calls += calls;
    return true;
}

// A completed writer made host and directly staged device bytes identical.
// Update only metadata here; no host-to-device copy is necessary.
bool llama_kv_stream_layer_lease_owner::adopt_staged_tail(
        llama_kv_stream_complete_layer_lease_t previous,
        const llama_kv_stream_logical_cache & cache,
        const llama_kv_stream_population_stats & staged) {
    if (!impl || !impl->state || !previous || !previous->reservation ||
            previous->reservation->owner != impl->state) return false;
    auto state = impl->state;
    auto reservation = previous->reservation;
    const auto host = cache.host();
    const auto frontier = cache.frontiers();
    const size_t next = cache.tokens();
    const uint64_t generation = cache.identity().generation;
    const size_t appended = next >= previous->active_tokens ? next - previous->active_tokens : 0;
    const size_t k_stride = host ? host->layout().k_token_bytes : 0;
    const size_t v_stride = host ? host->layout().v_token_bytes : 0;
    if (k_stride > SIZE_MAX - v_stride ||
            (k_stride + v_stride && appended > SIZE_MAX/(k_stride + v_stride))) return false;
    if (!host || !same_population_shape(host->config().shape, state->identity.policy.shape) ||
            reservation->layer >= state->layout.layers.size() ||
            state->layout.layers[reservation->layer].cache_id != cache.identity().id ||
            next <= previous->active_tokens || next > reservation->active_tokens ||
            generation <= previous->content_generation ||
            cache.content()->generation() != generation ||
            frontier.reserved != next || frontier.host != next ||
            frontier.committed != next || frontier.device != 0 ||
            staged.bytes != appended*(k_stride + v_stride) ||
            staged.calls < 2) return false;
    std::lock_guard<std::mutex> lock(state->mutex);
    if (state->closed || !reservation->populated || reservation->populating ||
            reservation->populated_tokens != previous->active_tokens ||
            reservation->content_generation != previous->content_generation ||
            state->identity.content_generation != previous->content_generation ||
            reservation->population.bytes > SIZE_MAX - staged.bytes ||
            reservation->population.calls > SIZE_MAX - staged.calls) return false;
    if (!acknowledge_resident_tail(*reservation,cache,previous->active_tokens,next)) return false;
    reservation->content_generation = generation;
    reservation->populated_tokens = next;
    reservation->population.bytes += staged.bytes;
    reservation->population.calls += staged.calls;
    state->identity.content_generation = generation;
    return true;
}


bool llama_kv_stream_layer_lease_owner::adopt_truncated_prefix(
        llama_kv_stream_complete_layer_lease_t previous,
        const llama_kv_stream_logical_cache & cache) {
    if (!impl || !impl->state || !previous || !previous->reservation) return false;
    const auto state = impl->state;
    const auto reservation = previous->reservation;
    const auto host = cache.host();
    const auto frontier = cache.frontiers();
    const size_t tokens = cache.tokens();
    const uint64_t generation = cache.identity().generation;
    if (reservation->owner != state || !host ||
            !same_population_shape(host->config().shape, state->identity.policy.shape) ||
            reservation->layer >= state->layout.layers.size() ||
            state->layout.layers[reservation->layer].cache_id != cache.identity().id ||
            state->layout.layers[reservation->layer].cache_layer >= host->config().layers ||
            tokens < 4 || tokens >= previous->active_tokens ||
            generation <= previous->content_generation ||
            cache.content()->generation() != generation ||
            frontier.reserved != tokens || frontier.host != tokens ||
            frontier.committed != tokens || frontier.device != 0) return false;
    std::lock_guard<std::mutex> lock(state->mutex);
    if (state->closed || !reservation->populated || reservation->populating ||
            reservation->populated_tokens != previous->active_tokens ||
            reservation->content_generation != previous->content_generation ||
            state->identity.content_generation != previous->content_generation) return false;
    // No device copy occurs: the immutable accepted prefix retains its exact
    // resident/ring placement. The generation change invalidates every old plan.
    reservation->populated_tokens = tokens;
    reservation->content_generation = generation;
    state->identity.content_generation = generation;
    return true;
}
bool llama_kv_stream_layer_lease_owner::rebind(
        const llama_kv_stream_layer_lease_layout & identity) {
    if (!impl || !impl->state) return false;
    auto state = impl->state;
    llama_kv_stream_policy_layout layout;
    if (!materialize(state->pool.get(), identity, layout)) return false;
    try {
        std::vector<uint8_t> occupied(identity.state.ring_slots, 0);
        std::vector<std::weak_ptr<layer_reservation>> layers(identity.policy.layers);
        std::lock_guard<std::mutex> lock(state->mutex);
        if (state->closed || state->reservations || state->leases || state->guards ||
                identity.layout_revision <= state->identity.layout_revision) return false;
        state->identity = identity;
        state->layout = std::move(layout);
        state->occupied = std::move(occupied);
        state->layers = std::move(layers);
        return true;
    } catch (const std::bad_alloc &) {
        return false;
    }
}

bool llama_kv_stream_layer_lease_owner::can_repartition() const {
    if (!impl || !impl->state) return false;
    std::lock_guard<std::mutex> lock(impl->state->mutex);
    return !impl->state->closed && !impl->state->reservations && !impl->state->leases && !impl->state->guards;
}

void llama_kv_stream_layer_lease_owner::close() {
    if (!impl || !impl->state) return;
    std::lock_guard<std::mutex> lock(impl->state->mutex);
    impl->state->closed = true;
}

bool llama_kv_stream_layer_lease_owner::closed() const {
    if (!impl || !impl->state) return true;
    std::lock_guard<std::mutex> lock(impl->state->mutex);
    return impl->state->closed;
}

size_t llama_kv_stream_layer_lease_owner::active_reservations() const {
    if (!impl || !impl->state) return 0;
    std::lock_guard<std::mutex> lock(impl->state->mutex);
    return impl->state->reservations;
}

size_t llama_kv_stream_layer_lease_owner::active_leases() const {
    if (!impl || !impl->state) return 0;
    std::lock_guard<std::mutex> lock(impl->state->mutex);
    return impl->state->leases;
}

size_t llama_kv_stream_layer_lease_owner::ring_slots_used() const {
    if (!impl || !impl->state) return 0;
    std::lock_guard<std::mutex> lock(impl->state->mutex);
    return std::count(impl->state->occupied.begin(), impl->state->occupied.end(), uint8_t(1));
}

uint64_t llama_kv_stream_layer_lease_owner::layout_revision() const {
    if (!impl || !impl->state) return 0;
    std::lock_guard<std::mutex> lock(impl->state->mutex);
    return impl->state->identity.layout_revision;
}

uint64_t llama_kv_stream_layer_lease_owner::content_generation() const {
    if (!impl || !impl->state) return 0;
    std::lock_guard<std::mutex> lock(impl->state->mutex);
    return impl->state->identity.content_generation;
}

llama_kv_stream_complete_layer_lease_t llama_kv_stream_complete_layer_lease_retain(
        llama_kv_stream_complete_layer_lease_t lease) {
    if (!lease) return nullptr;
    uint32_t count = lease->references.load(std::memory_order_relaxed);
    do {
        GGML_ASSERT(count > 0 && count < std::numeric_limits<uint32_t>::max());
    } while (!lease->references.compare_exchange_weak(
        count, count + 1, std::memory_order_relaxed, std::memory_order_relaxed));
    return lease;
}

void llama_kv_stream_complete_layer_lease_free(
        llama_kv_stream_complete_layer_lease_t lease) {
    if (!lease) return;
    const uint32_t count = lease->references.fetch_sub(1, std::memory_order_acq_rel);
    GGML_ASSERT(count > 0);
    if (count == 1) delete lease;
}

ggml_kv_stream_span_plan_t llama_kv_stream_complete_layer_lease_plan(
        llama_kv_stream_complete_layer_lease_t lease) {
    return lease && lease->reservation &&
        lease->content_generation == lease->reservation->content_generation ? lease->plan : nullptr;
}

uint32_t llama_kv_stream_complete_layer_lease_layer(
        llama_kv_stream_complete_layer_lease_t lease) {
    return lease && lease->reservation ? lease->reservation->layer : UINT32_MAX;
}

size_t llama_kv_stream_complete_layer_lease_resident_tokens(
        llama_kv_stream_complete_layer_lease_t lease) {
    return lease && lease->reservation ? std::min(lease->active_tokens, lease->reservation->resident_tokens) : 0;
}

size_t llama_kv_stream_complete_layer_lease_ring_first(
        llama_kv_stream_complete_layer_lease_t lease) {
    return lease && lease->reservation ? lease->reservation->ring_first : SIZE_MAX;
}

size_t llama_kv_stream_complete_layer_lease_ring_slots(
        llama_kv_stream_complete_layer_lease_t lease) {
    return lease && lease->reservation ? lease->reservation->ring_slots : 0;
}

uint64_t llama_kv_stream_complete_layer_lease_layout_revision(
        llama_kv_stream_complete_layer_lease_t lease) {
    return lease && lease->reservation ? lease->reservation->layout_revision : 0;
}

uint64_t llama_kv_stream_complete_layer_lease_content_generation(
        llama_kv_stream_complete_layer_lease_t lease) {
    return lease ? lease->content_generation : 0;
}

llama_kv_stream_population_stats llama_kv_stream_complete_layer_lease_population(
        llama_kv_stream_complete_layer_lease_t lease) {
    if (!lease || !lease->reservation) return {};
    auto reservation = lease->reservation;
    std::lock_guard<std::mutex> lock(reservation->owner->mutex);
    return reservation->populated ? reservation->population : llama_kv_stream_population_stats{};
}
