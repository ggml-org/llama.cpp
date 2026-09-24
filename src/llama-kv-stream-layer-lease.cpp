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
    auto * buffer = ggml_backend_memory_lease_buffer(pool);
    ggml_backend_memory_region region;
    if (!buffer || !ggml_backend_memory_lease_get_region(pool, &region) ||
            ggml_backend_buffer_get_size(buffer) != region.size ||
            identity.policy.pool_bytes > region.size) return false;
    llama_kv_stream_policy_layout next;
    if (llama_kv_stream_policy_layout_make(
            identity.policy, identity.state, identity.active_tokens, next).status !=
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
        if (!multiply(size_t(layer.resident_live_pages), page_tokens, capacity)) return nullptr;
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
        const llama_kv_stream_logical_cache & cache, llama_kv_stream_population_stats & stats) {
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
        ggml_backend_synchronize(backend);
    } catch (...) {
        try { ggml_backend_synchronize(backend); } catch (...) {}
        return false;
    }
    stats = next;
    return true;
}

llama_kv_stream_complete_layer_lease_t llama_kv_stream_layer_lease_owner::acquire_populated(
        ggml_backend_t backend, const llama_kv_stream_complete_layer_request & request,
        const llama_kv_stream_logical_cache & cache) {
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
    const bool copied = populate_layer(backend, lease, cache, stats);
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
            ready = true;
        }
    }
    if (!ready) {
        llama_kv_stream_complete_layer_lease_free(lease);
        return nullptr;
    }
    return lease;
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
    return lease ? lease->plan : nullptr;
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
    return lease && lease->reservation ? lease->reservation->content_generation : 0;
}

llama_kv_stream_population_stats llama_kv_stream_complete_layer_lease_population(
        llama_kv_stream_complete_layer_lease_t lease) {
    if (!lease || !lease->reservation) return {};
    auto reservation = lease->reservation;
    std::lock_guard<std::mutex> lock(reservation->owner->mutex);
    return reservation->populated ? reservation->population : llama_kv_stream_population_stats{};
}
