#include "llama-kv-stream-policy.h"

#include <algorithm>
#include <cmath>
#include <new>
#include <utility>

using status = llama_kv_stream_policy_status;

constexpr uint32_t DECODE_QUERY_LIMIT = 32;
constexpr double MISS_THRESHOLD = .01;
constexpr double COPY_SATURATED = .90;
constexpr double COPY_LIGHT = .50;
constexpr double RING_LIGHT = .50;

// Page counts never need an overflowing numerator + denominator - 1.
static uint64_t ceil_div(uint64_t n, uint64_t d) { return n/d + (n % d != 0); }
// Saturate long-running hysteresis counters instead of wrapping to a fresh window.
static uint32_t increment(uint32_t n) { return n == UINT32_MAX ? n : n + 1; }

// Derive the fixed page budget from the shared quant geometry, reserving conversion first.
static llama_kv_stream_policy_result budget_make(
        const llama_kv_stream_policy_config & c, llama_kv_stream_policy_budget & b) {
    if (!c.layers || !std::isfinite(c.overlap_ratio) || c.overlap_ratio <= 0 ||
            !c.grow_evaluations || !c.shrink_evaluations || !c.cooldown_evaluations) {
        return {status::invalid_config, {}};
    }
    if (!c.caches.empty()) {
        uint64_t mapped = 0;
        for (size_t i = 0; i < c.caches.size(); ++i) {
            const auto & cache = c.caches[i];
            if (!cache.id || !cache.layers || cache.layers > c.layers - mapped) return {status::invalid_config, {}};
            for (size_t j = 0; j < i; ++j) {
                if (c.caches[j].id == cache.id) return {status::invalid_config, {}};
            }
            mapped += cache.layers;
        }
        if (mapped != c.layers) return {status::invalid_config, {}};
    }
    if (c.shape.page_tokens <= 0 || uint64_t(c.shape.page_tokens) > SIZE_MAX) return {status::geometry_error, {ggml_kv_stream_status::invalid_shape}};
    const auto result = ggml_kv_stream_resolve(c.shape, c.capabilities, size_t(c.shape.page_tokens), b.page);
    if (result.status != ggml_kv_stream_status::success) return {status::geometry_error, result};
    const auto & page = b.page.storage;
    // Fixed page accounting is valid only when concatenated planes need no per-page padding.
    if (!page.bytes || page.k_bytes % c.shape.alignment || page.v_bytes % c.shape.alignment ||
            page.v_offset != page.k_bytes) return {status::geometry_error, {ggml_kv_stream_status::invalid_alignment}};
    if (c.pool_bytes <= b.page.conversion.bytes) return {status::invalid_budget, {}};
    const size_t pages = (c.pool_bytes - b.page.conversion.bytes)/page.bytes;
    if (pages > UINT32_MAX) return {status::overflow, {}};
    if (pages <= c.layers) return {status::invalid_budget, {}};
    const uint32_t hint = c.initial_ring_slots ? c.initial_ring_slots :
        std::min(uint32_t(8), uint32_t(pages) - c.layers);
    if (hint > pages - c.layers) return {status::invalid_budget, {}};
    b.shape = c.shape;
    b.pool_bytes = c.pool_bytes;
    b.pages = uint32_t(pages);
    try {
        b.caches = c.caches;
    } catch (const std::bad_alloc &) {
        return {status::allocation_failed, {}};
    }
    b.layers = c.layers;
    b.minimum_ring_slots = b.pages - ((b.pages - hint)/b.layers)*b.layers;
    b.conversion_offset = pages*page.bytes;
    b.unused_bytes = c.pool_bytes - b.conversion_offset - b.page.conversion.bytes;
    return {};
}

llama_kv_stream_policy_result llama_kv_stream_policy_minimum_pool_bytes(
        const llama_kv_stream_policy_config & config, size_t & output) {
    ggml_kv_stream_execution execution;
    const auto geometry = ggml_kv_stream_resolve(
        config.shape,config.capabilities,size_t(config.shape.page_tokens),execution);
    if (geometry.status != ggml_kv_stream_status::success) {
        return {status::geometry_error,geometry};
    }
    if (!config.layers) return {status::invalid_config,{}};
    if (config.layers == UINT32_MAX || !execution.storage.bytes ||
            size_t(config.layers)+1 >
                (SIZE_MAX-execution.conversion.bytes)/execution.storage.bytes) {
        return {status::overflow,{}};
    }
    llama_kv_stream_policy_config minimum;
    try {
        minimum = config;
    } catch (const std::bad_alloc &) {
        return {status::allocation_failed, {}};
    }
    minimum.pool_bytes = (size_t(config.layers)+1)*execution.storage.bytes +
        execution.conversion.bytes;
    llama_kv_stream_policy_budget budget;
    const auto result = budget_make(minimum,budget);
    if (result.status != status::success) return result;
    output = minimum.pool_bytes;
    return {};
}

// Preserve the fixed pool's encoding and page geometry across proposals.
static bool same_shape(const ggml_kv_stream_shape & a, const ggml_kv_stream_shape & b) {
    return a.type_k == b.type_k && a.type_v == b.type_v && a.head_dim_k == b.head_dim_k &&
        a.head_dim_v == b.head_dim_v && a.heads == b.heads && a.page_tokens == b.page_tokens && a.alignment == b.alignment;
}
// Detect altered derived metadata before trusting a saved budget.
static bool same_planes(const ggml_kv_stream_layout & a, const ggml_kv_stream_layout & b) {
    return a.k_row_bytes == b.k_row_bytes && a.v_row_bytes == b.v_row_bytes &&
        a.k_token_bytes == b.k_token_bytes && a.v_token_bytes == b.v_token_bytes &&
        a.k_bytes == b.k_bytes && a.v_offset == b.v_offset && a.v_bytes == b.v_bytes &&
        a.bytes == b.bytes && a.tokens == b.tokens && a.pages == b.pages && a.tail_tokens == b.tail_tokens;
}

// Reject stale budget identity and malformed state before using counters or calculating offsets.
static bool state_valid(const llama_kv_stream_policy_budget & b, const llama_kv_stream_policy_state & s) {
    const auto & old = s.budget;
    if (b.caches.size() != old.caches.size()) return false;
    for (size_t i = 0; i < b.caches.size(); ++i) {
        if (b.caches[i].id != old.caches[i].id || b.caches[i].layers != old.caches[i].layers) return false;
    }
    return same_shape(b.shape, old.shape) && same_planes(b.page.storage, old.page.storage) &&
        same_planes(b.page.conversion, old.page.conversion) && b.page.attention == old.page.attention &&
        b.pool_bytes == old.pool_bytes && b.pages == old.pages && b.layers == old.layers &&
        b.minimum_ring_slots == old.minimum_ring_slots && b.conversion_offset == old.conversion_offset &&
        b.unused_bytes == old.unused_bytes && s.ring_slots >= b.minimum_ring_slots &&
        uint64_t(s.resident_pages_per_layer)*b.layers + s.ring_slots == b.pages &&
        (!s.decode_active_pages || s.decode_active_pages > s.resident_pages_per_layer) &&
        (!s.feedback_initialized || (s.misses <= s.samples &&
            s.layer_misses <= s.layer_samples));
}

struct capacity_profile {
    uint32_t mean = 0, active = 0, layers = 0, splits = 0;
    uint32_t streamed_base = 0, remainder = 0;
};

// This is the production concentrated layout, bounded by both ring size and per-layer active capacity.
static capacity_profile profile(const llama_kv_stream_policy_state & s) {
    capacity_profile p;
    p.mean = s.resident_pages_per_layer;
    p.active = s.decode_active_pages;
    p.layers = s.budget.layers;
    if (p.active <= p.mean) return p;
    const uint64_t deficit = uint64_t(p.active - p.mean)*p.layers;
    const uint64_t minimum = ceil_div(deficit, p.active);
    p.splits = s.spread_streaming ? p.layers : uint32_t(std::max(
        std::min(uint64_t(p.layers), ceil_div(deficit, s.ring_slots)),
        minimum));
    p.streamed_base = uint32_t(deficit/p.splits);
    p.remainder = uint32_t(deficit % p.splits);
    return p;
}

// Invert floor(split * layers / splits) without scanning a layer list.
static uint32_t capacity(const capacity_profile & p, uint32_t layer) {
    if (!p.splits) return p.mean;
    const uint64_t split = ((uint64_t(layer) + 1)*p.splits - 1)/p.layers;
    if (split*p.layers/p.splits != layer) return p.active;
    return p.active - p.streamed_base - (split < p.remainder ? 1u : 0u);
}

// All-layer splits are uniform too; extent changes alone need not move any addresses.
static bool same_capacity(const capacity_profile & a, const capacity_profile & b) {
    const bool uniform_a = !a.splits || a.splits == a.layers;
    const bool uniform_b = !b.splits || b.splits == b.layers;
    if (uniform_a || uniform_b) return uniform_a && uniform_b && a.mean == b.mean;
    return a.active == b.active && a.splits == b.splits &&
        a.streamed_base == b.streamed_base && a.remainder == b.remainder;
}

// Preserve the descending reference predicate with a bounded search instead of a page-by-page scan.
static uint32_t overlap_target(const llama_kv_stream_policy_budget & b, uint32_t active, double ratio) {
    const uint32_t maximum = std::min(active, (b.pages - b.minimum_ring_slots)/b.layers);
    auto fits = [&](uint32_t resident) {
        return resident == active || double(b.pages - resident*b.layers) >= ratio*double(active - resident);
    };
    if (fits(maximum)) return maximum;
    // If ratio >= layers, demotion cannot improve the inequality. Zero is the reference's fallback.
    if (ratio >= double(b.layers) || !fits(0)) return 0;
    uint32_t low = 0, high = maximum;
    while (low < high) {
        const uint32_t distance = high - low;
        const uint32_t middle = low + distance/2 + distance%2;
        if (fits(middle)) low = middle;
        else high = middle - 1;
    }
    return low;
}

// A new/reset/invalid sample establishes a baseline; it never invents a light-load evaluation.
static void ingest_feedback(const llama_kv_stream_feedback & f, llama_kv_stream_policy_decision & d,
        double & miss_ratio) {
    if (!f.available) return;
    auto & s = d.next;
    const bool totals_valid = f.misses <= f.samples && f.layer_misses <= f.layer_samples;
    const bool metrics_valid = std::isfinite(f.copy_busy_ratio) && f.copy_busy_ratio >= 0 && f.copy_busy_ratio <= 1;
    const bool layer_available = f.layer_samples || s.layer_samples;
    bool reset = !s.feedback_initialized || f.epoch != s.feedback_epoch || !totals_valid || !metrics_valid ||
        f.samples < s.samples || f.misses < s.misses ||
        f.layer_samples < s.layer_samples || f.layer_misses < s.layer_misses;
    if (!reset) {
        const uint64_t samples = layer_available ? f.layer_samples-s.layer_samples : f.samples-s.samples;
        const uint64_t misses = layer_available ? f.layer_misses-s.layer_misses : f.misses-s.misses;
        reset = misses > samples;
        if (!reset && samples) {
            d.feedback_used = true;
            miss_ratio = double(misses)/double(samples);
            d.layer_feedback_used = layer_available;
        }
    }
    s.feedback_initialized = totals_valid;
    d.layer_feedback_available = layer_available;
    s.feedback_epoch = f.epoch;
    s.samples = totals_valid ? f.samples : 0;
    s.misses = totals_valid ? f.misses : 0;
    s.layer_samples = totals_valid ? f.layer_samples : 0;
    s.layer_misses = totals_valid ? f.layer_misses : 0;
    if (reset) {
        s.starved = s.overprovisioned = 0;
        d.feedback_reset = true;
    }
}

// Allocate one initial resident page per layer at minimum, and put the page remainder into the ring.
llama_kv_stream_policy_result llama_kv_stream_policy_initialize(
        const llama_kv_stream_policy_config & c, llama_kv_stream_policy_state & output) {
    llama_kv_stream_policy_state next;
    const auto result = budget_make(c, next.budget);
    if (result.status != status::success) return result;
    next.ring_slots = next.budget.minimum_ring_slots;
    next.resident_pages_per_layer = (next.budget.pages - next.ring_slots)/next.budget.layers;
    output = std::move(next);
    return {};
}

// Propose quotas and feedback cursors; the runtime adapter owns transactional publication and GPU ordering.
llama_kv_stream_policy_result llama_kv_stream_policy_step(
        const llama_kv_stream_policy_config & c, const llama_kv_stream_policy_state & previous,
        const llama_kv_stream_policy_observation & observation, llama_kv_stream_policy_decision & output) {
    llama_kv_stream_policy_budget b;
    auto result = budget_make(c, b);
    if (result.status != status::success) return result;
    if (!state_valid(b, previous)) return {status::invalid_state, {}};
    if (!observation.query_tokens) return {status::invalid_observation, {}};
    const uint64_t active_wide = ceil_div(observation.active_tokens, uint64_t(c.shape.page_tokens));
    if (active_wide > UINT32_MAX) return {status::overflow, {}};
    const uint32_t active = uint32_t(active_wide);
    llama_kv_stream_policy_decision d;
    try {
        d.next = previous;
    } catch (const std::bad_alloc &) {
        return {status::allocation_failed, {}};
    }
    auto & s = d.next;
    double miss_ratio = 0;
    ingest_feedback(observation.feedback,d,miss_ratio);
    const bool pressure = active > s.resident_pages_per_layer;
    const uint32_t requested_decode = !observation.uniform_prefill && observation.query_tokens <= DECODE_QUERY_LIMIT && pressure ? active : 0;
    const bool entering = requested_decode && requested_decode != previous.decode_active_pages;
    d.target_resident_pages = overlap_target(b, active, c.overlap_ratio);
    if (!pressure) {
        s.starved = s.overprovisioned = 0; s.spread_streaming = false;
    } else if (!c.fixed_ring && (entering || d.feedback_used)) {
        if (d.feedback_used) s.evaluations_since_repartition = increment(s.evaluations_since_repartition);
        const uint32_t target = std::max(1u, d.target_resident_pages);
        const bool below = s.resident_pages_per_layer > target;
        const bool feedback_starved = d.feedback_used && miss_ratio > MISS_THRESHOLD &&
            observation.feedback.copy_busy_ratio < COPY_SATURATED;
        const bool light = d.feedback_used && miss_ratio <= MISS_THRESHOLD && observation.feedback.copy_busy_ratio < COPY_LIGHT &&
            double(observation.feedback.peak_slots)/s.ring_slots < RING_LIGHT;
        if (d.feedback_used) {
            s.starved = below || feedback_starved ? increment(s.starved) : 0;
            s.overprovisioned = light ? increment(s.overprovisioned) : 0;
        }
        const bool cooldown = s.evaluations_since_repartition >= c.cooldown_evaluations;
        const bool hard_deadline = d.layer_feedback_used && feedback_starved;
        if (entering && below) {
            s.resident_pages_per_layer = target;
        } else if ((cooldown || hard_deadline) && d.feedback_used && (below || feedback_starved) &&
                s.starved >= c.grow_evaluations && s.resident_pages_per_layer > 1) {
            s.resident_pages_per_layer = below ? target : s.resident_pages_per_layer - 1;
            // Extra slots are useful only if placement does not enlarge the streamed spans to consume
            // them. Keep pressure placement distributed until the next uniform/no-pressure phase.
            if (feedback_starved) s.spread_streaming = true;
        } else if (cooldown && light && s.overprovisioned >= c.shrink_evaluations &&
                s.resident_pages_per_layer < target &&
                uint64_t(s.ring_slots) >= uint64_t(b.minimum_ring_slots) + b.layers) {
            ++s.resident_pages_per_layer;
        }
        s.ring_slots = b.pages - s.resident_pages_per_layer*b.layers;
    }
    d.partition_changed = s.ring_slots != previous.ring_slots;
    if (d.partition_changed) {
        s.starved = s.overprovisioned = s.evaluations_since_repartition = 0;
    }
    s.decode_active_pages = !observation.uniform_prefill && observation.query_tokens <= DECODE_QUERY_LIMIT && active > s.resident_pages_per_layer ? active : 0;
    if (!s.decode_active_pages) s.spread_streaming = false;
    d.layout_changed = d.partition_changed || !same_capacity(profile(previous), profile(s));
    output = std::move(d);
    return {};
}

llama_kv_stream_policy_result llama_kv_stream_policy_reserve_layer(
        const llama_kv_stream_policy_config & c, const llama_kv_stream_policy_state & previous,
        size_t placement_tokens, size_t reserved_tokens, uint32_t layer,
        llama_kv_stream_policy_decision & output) {
    if (!placement_tokens || reserved_tokens < placement_tokens || layer >= c.layers)
        return {status::invalid_observation, {}};
    llama_kv_stream_policy_layout layout;
    auto result = llama_kv_stream_policy_layout_make(c, previous, placement_tokens, layout);
    if (result.status != status::success) return result;
    const auto & b = previous.budget;
    const uint64_t active = ceil_div(reserved_tokens, uint64_t(c.shape.page_tokens));
    const uint64_t capacity = layout.layers[layer].capacity_pages;
    const uint64_t streamed = active > capacity ? active - capacity : 0;
    if (active > b.pages) return {status::invalid_budget, {}};
    llama_kv_stream_policy_decision next;
    try {
        next.next = previous;
    } catch (const std::bad_alloc &) {
        return {status::allocation_failed, {}};
    }
    if (double(previous.ring_slots) >= c.overlap_ratio * double(streamed) ||
            (c.fixed_ring && previous.ring_slots >= streamed)) {
        output = std::move(next);
        return {};
    }
    if (c.fixed_ring) return {status::invalid_budget, {}};
    auto & state = next.next;
    next.target_resident_pages = overlap_target(b, uint32_t(active), c.overlap_ratio);
    state.resident_pages_per_layer = std::min(previous.resident_pages_per_layer, next.target_resident_pages);
    state.ring_slots = b.pages - state.resident_pages_per_layer * b.layers;
    // Concentrated placement can leave this layer with no resident capacity.
    state.decode_active_pages = 0;
    state.spread_streaming = false;
    const uint64_t suffix = active > state.resident_pages_per_layer ?
        active - state.resident_pages_per_layer : 0;
    if (state.ring_slots < suffix) return {status::invalid_budget, {}};
    next.partition_changed = state.ring_slots != previous.ring_slots;
    next.layout_changed = next.partition_changed || !same_capacity(profile(previous), profile(state));
    if (next.layout_changed) state.starved = state.overprovisioned = state.evaluations_since_repartition = 0;
    output = std::move(next);
    return {};
}

static llama_kv_stream_policy_result resize_plan(
        const llama_kv_stream_policy_config & current, size_t active_tokens,
        bool decode, size_t pool_bytes, bool growing, llama_kv_stream_policy_rebind & output) {
    if ((growing && pool_bytes <= current.pool_bytes) ||
            (!growing && pool_bytes >= current.pool_bytes)) return {status::invalid_budget, {}};
    llama_kv_stream_policy_rebind next;
    try {
        next.config = current;
    } catch (const std::bad_alloc &) {
        return {status::allocation_failed, {}};
    }
    next.config.pool_bytes = pool_bytes;
    auto result = llama_kv_stream_policy_initialize(next.config,next.state);
    if (result.status != status::success) return result;
    llama_kv_stream_policy_decision decision;
    result = llama_kv_stream_policy_step(next.config,next.state,
        {active_tokens,1,{},!decode},decision);
    if (result.status != status::success) return result;
    next.state = decision.next;
    result = llama_kv_stream_policy_layout_make(
        next.config,next.state,active_tokens,next.layout);
    if (result.status != status::success) return result;
    output = std::move(next);
    return {};
}

llama_kv_stream_policy_result llama_kv_stream_policy_grow(
        const llama_kv_stream_policy_config & current, size_t active_tokens,
        bool decode, size_t pool_bytes, llama_kv_stream_policy_rebind & output) {
    return resize_plan(current,active_tokens,decode,pool_bytes,true,output);
}

llama_kv_stream_policy_result llama_kv_stream_policy_shrink(
        const llama_kv_stream_policy_config & current, size_t active_tokens,
        bool decode, size_t pool_bytes, llama_kv_stream_policy_rebind & output) {
    return resize_plan(current,active_tokens,decode,pool_bytes,false,output);
}

// Ring first, then compact per-layer planes, then aligned conversion scratch; only the final tail is unused.
llama_kv_stream_policy_result llama_kv_stream_policy_layout_make(
        const llama_kv_stream_policy_config & c, const llama_kv_stream_policy_state & s,
        size_t active_tokens, llama_kv_stream_policy_layout & output) {
    llama_kv_stream_policy_budget b;
    const auto result = budget_make(c, b);
    if (result.status != status::success) return result;
    if (!state_valid(b, s)) return {status::invalid_state, {}};
    const uint64_t active_wide = ceil_div(active_tokens, uint64_t(c.shape.page_tokens));
    if (active_wide > UINT32_MAX) return {status::overflow, {}};
    const uint32_t active = uint32_t(active_wide);
    if (s.decode_active_pages && s.decode_active_pages != active) return {status::invalid_observation, {}};
    try {
        llama_kv_stream_policy_layout next;
        const size_t page_tokens = size_t(c.shape.page_tokens);
        if (s.ring_slots > SIZE_MAX/page_tokens) return {status::overflow, {}};
        auto geometry = ggml_kv_stream_layout_make(c.shape, size_t(s.ring_slots)*page_tokens, next.ring);
        if (geometry.status != ggml_kv_stream_status::success) return {status::geometry_error, geometry};
        if (b.layers > next.layers.max_size()) return {status::overflow, {}};
        next.layers.resize(b.layers);
        const auto p = profile(s);
        size_t offset = next.ring.bytes;
        size_t cache_index = 0;
        uint32_t cache_layer = 0;
        for (uint32_t layer = 0; layer < b.layers; ++layer) {
            auto & entry = next.layers[layer];
            if (b.caches.empty()) {
                entry.cache_layer = layer;
            } else {
                entry.cache_id = b.caches[cache_index].id;
                entry.cache_layer = cache_layer++;
                if (cache_layer == b.caches[cache_index].layers) {
                    ++cache_index;
                    cache_layer = 0;
                }
            }
            entry.capacity_pages = capacity(p, layer);
            entry.resident_live_pages = std::min(active, entry.capacity_pages);
            entry.streamed_pages = active - entry.resident_live_pages;
            entry.waves = uint32_t(ceil_div(entry.streamed_pages, s.ring_slots));
            entry.offset = offset;
            if (entry.capacity_pages > SIZE_MAX/page_tokens) return {status::overflow, {}};
            geometry = ggml_kv_stream_layout_make(c.shape, size_t(entry.capacity_pages)*page_tokens, entry.planes);
            if (geometry.status != ggml_kv_stream_status::success) return {status::geometry_error, geometry};
            if (offset > b.conversion_offset || entry.planes.bytes > b.conversion_offset - offset) return {status::invalid_state, {}};
            offset += entry.planes.bytes;
        }
        if (offset != b.conversion_offset) return {status::invalid_state, {}};
        next.conversion_offset = b.conversion_offset;
        next.conversion_bytes = b.page.conversion.bytes;
        next.unused_bytes = b.unused_bytes;
        output = std::move(next);
        return {};
    } catch (const std::bad_alloc &) {
        return {status::allocation_failed, {}};
    }
}
