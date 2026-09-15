#include "llama-kv-stream-publication.h"

#include <algorithm>
#include <array>
#include <deque>
#include <limits>
#include <new>
#include <utility>

using lease_ptr = std::unique_ptr<ggml_backend_memory_lease, decltype(&ggml_backend_memory_lease_free)>;

enum class publication_operation : uint8_t {
    idle,
    pending,
    complete,
    failed,
};

struct publication_pair {
    std::array<std::array<publication_operation, 2>, 2> operations{};
};

struct llama_kv_stream_publication_entry {
    uint64_t sequence = 0;
    uint64_t generation = 0;
    size_t first = 0;
    size_t count = 0;
    std::vector<publication_pair> pairs;
    std::vector<std::shared_ptr<void>> owners;
    std::vector<lease_ptr> leases;
    size_t outstanding = 0;
    bool committed = false;
    bool failed = false;
    bool cancelled = false;
    bool abandoned = false;
};

static bool valid(llama_kv_stream_publication_plane value) {
    return value == llama_kv_stream_publication_plane::k || value == llama_kv_stream_publication_plane::v;
}

static bool valid(llama_kv_stream_publication_domain value) {
    return value == llama_kv_stream_publication_domain::device || value == llama_kv_stream_publication_domain::host;
}

static size_t index(llama_kv_stream_publication_plane value) {
    return value == llama_kv_stream_publication_plane::v;
}

static size_t index(llama_kv_stream_publication_domain value) {
    return value == llama_kv_stream_publication_domain::host;
}

static bool pair_ready(const publication_pair & pair, llama_kv_stream_publication_domain domain) {
    const auto & operations = pair.operations[index(domain)];
    return operations[index(llama_kv_stream_publication_plane::k)] == publication_operation::complete &&
        operations[index(llama_kv_stream_publication_plane::v)] == publication_operation::complete;
}

static bool entry_ready(const llama_kv_stream_publication_entry & entry, llama_kv_stream_publication_domain domain) {
    return !entry.failed && !entry.cancelled && std::all_of(entry.pairs.begin(), entry.pairs.end(),
        [domain](const publication_pair & pair) { return pair_ready(pair, domain); });
}

static bool terminal(const llama_kv_stream_publication_entry & entry) {
    return entry.committed || entry.failed || entry.cancelled;
}

struct llama_kv_stream_publication_state {
    llama_kv_stream_publication_config config;
    llama_kv_stream_publication_frontiers frontiers;
    std::deque<std::shared_ptr<llama_kv_stream_publication_entry>> entries;
    uint64_t next_sequence = 0;
    bool closed = false;
    bool sequence_exhausted = false;

    size_t advance(size_t frontier, llama_kv_stream_publication_domain domain) const {
        for (const auto & entry : entries) {
            if (entry->first + entry->count <= frontier) continue;
            if (entry->first != frontier || !entry_ready(*entry, domain)) break;
            frontier = entry->first + entry->count;
        }
        return frontier;
    }

    void advance() {
        if (closed) return;
        frontiers.device = advance(frontiers.device, llama_kv_stream_publication_domain::device);
        frontiers.host = advance(frontiers.host, llama_kv_stream_publication_domain::host);
        size_t committed = frontiers.committed;
        for (const auto & entry : entries) {
            if (entry->first + entry->count <= committed) continue;
            if (entry->first != committed || !entry_ready(*entry, llama_kv_stream_publication_domain::device) ||
                    !entry_ready(*entry, llama_kv_stream_publication_domain::host)) break;
            committed = entry->first + entry->count;
        }
        frontiers.committed = committed;
        for (const auto & entry : entries) if (entry->first + entry->count <= committed) entry->committed = true;
    }

    void collect() {
        entries.erase(std::remove_if(entries.begin(), entries.end(), [](const auto & entry) {
            return entry->abandoned && entry->outstanding == 0 && terminal(*entry);
        }), entries.end());
    }

    bool submit(const std::shared_ptr<llama_kv_stream_publication_entry> & entry, uint32_t pair,
            llama_kv_stream_publication_plane plane, llama_kv_stream_publication_domain domain) {
        if (closed || !entry || entry->generation != config.generation || terminal(*entry) ||
                pair >= entry->pairs.size() || !valid(plane) || !valid(domain)) return false;
        auto & operation = entry->pairs[pair].operations[index(domain)][index(plane)];
        if (operation != publication_operation::idle || entry->outstanding == std::numeric_limits<size_t>::max()) return false;
        operation = publication_operation::pending;
        ++entry->outstanding;
        return true;
    }

    bool finish(const std::shared_ptr<llama_kv_stream_publication_entry> & entry, uint32_t pair,
            llama_kv_stream_publication_plane plane, llama_kv_stream_publication_domain domain, bool success) {
        if (!entry || entry->generation != config.generation || pair >= entry->pairs.size() || !valid(plane) || !valid(domain)) return false;
        auto & operation = entry->pairs[pair].operations[index(domain)][index(plane)];
        if (operation != publication_operation::pending || entry->outstanding == 0) return false;
        operation = success ? publication_operation::complete : publication_operation::failed;
        --entry->outstanding;
        if (!success) {
            entry->failed = true;
            closed = true;
        } else {
            advance();
        }
        collect();
        return true;
    }

    bool cancel(const std::shared_ptr<llama_kv_stream_publication_entry> & entry) {
        if (!entry || entry->generation != config.generation || entry->committed || entry->failed) return false;
        if (!entry->cancelled) {
            entry->cancelled = true;
            closed = true;
        }
        collect();
        return true;
    }

    void shutdown() {
        closed = true;
        for (const auto & entry : entries) if (!terminal(*entry)) entry->cancelled = true;
        collect();
    }
};

llama_kv_stream_publication_completion::llama_kv_stream_publication_completion(
        std::shared_ptr<llama_kv_stream_publication_state> state,
        std::shared_ptr<llama_kv_stream_publication_entry> entry, uint32_t pair,
        llama_kv_stream_publication_plane plane, llama_kv_stream_publication_domain domain) :
    state(std::move(state)), entry(std::move(entry)), pair(pair), plane(plane), domain(domain) {}

llama_kv_stream_publication_completion::~llama_kv_stream_publication_completion() {
    finish(false);
}

llama_kv_stream_publication_completion::llama_kv_stream_publication_completion(
        llama_kv_stream_publication_completion && other) noexcept :
    state(std::move(other.state)), entry(std::move(other.entry)), pair(other.pair), plane(other.plane), domain(other.domain) {}

llama_kv_stream_publication_completion & llama_kv_stream_publication_completion::operator=(
        llama_kv_stream_publication_completion && other) noexcept {
    if (this != &other) {
        finish(false);
        state = std::move(other.state);
        entry = std::move(other.entry);
        pair = other.pair;
        plane = other.plane;
        domain = other.domain;
    }
    return *this;
}

bool llama_kv_stream_publication_completion::pending() const noexcept {
    return state && entry;
}

bool llama_kv_stream_publication_completion::finish(bool success) noexcept {
    if (!pending()) return false;
    auto retained_state = std::move(state);
    auto retained_entry = std::move(entry);
    return retained_state->finish(retained_entry, pair, plane, domain, success);
}

llama_kv_stream_publication_ticket::llama_kv_stream_publication_ticket(
        std::shared_ptr<llama_kv_stream_publication_state> state,
        std::shared_ptr<llama_kv_stream_publication_entry> entry) :
    state(std::move(state)), entry(std::move(entry)) {}

llama_kv_stream_publication_ticket::~llama_kv_stream_publication_ticket() {
    abandon();
}

llama_kv_stream_publication_ticket::llama_kv_stream_publication_ticket(
        llama_kv_stream_publication_ticket && other) noexcept :
    state(std::move(other.state)), entry(std::move(other.entry)) {}

llama_kv_stream_publication_ticket & llama_kv_stream_publication_ticket::operator=(
        llama_kv_stream_publication_ticket && other) noexcept {
    if (this != &other) {
        abandon();
        state = std::move(other.state);
        entry = std::move(other.entry);
    }
    return *this;
}

bool llama_kv_stream_publication_ticket::pending() const noexcept {
    return state && entry && !entry->abandoned;
}

uint64_t llama_kv_stream_publication_ticket::sequence() const noexcept {
    return pending() ? entry->sequence : 0;
}

uint64_t llama_kv_stream_publication_ticket::generation() const noexcept {
    return pending() ? entry->generation : 0;
}

size_t llama_kv_stream_publication_ticket::first() const noexcept {
    return pending() ? entry->first : 0;
}

size_t llama_kv_stream_publication_ticket::count() const noexcept {
    return pending() ? entry->count : 0;
}

uint32_t llama_kv_stream_publication_ticket::pairs() const noexcept {
    return pending() ? uint32_t(entry->pairs.size()) : 0;
}

bool llama_kv_stream_publication_ticket::ready(uint32_t pair, llama_kv_stream_publication_domain domain) const noexcept {
    return pending() && valid(domain) && pair < entry->pairs.size() && pair_ready(entry->pairs[pair], domain);
}

bool llama_kv_stream_publication_ticket::committed() const noexcept {
    return pending() && entry->committed;
}

bool llama_kv_stream_publication_ticket::failed() const noexcept {
    return pending() && entry->failed;
}

bool llama_kv_stream_publication_ticket::submit(uint32_t pair, llama_kv_stream_publication_plane plane,
        llama_kv_stream_publication_domain domain, llama_kv_stream_publication_completion & output) noexcept {
    if (!pending() || output.pending() || !state->submit(entry, pair, plane, domain)) return false;
    output = llama_kv_stream_publication_completion(state, entry, pair, plane, domain);
    return true;
}

bool llama_kv_stream_publication_ticket::cancel() noexcept {
    return pending() && state->cancel(entry);
}

bool llama_kv_stream_publication_ticket::retire() noexcept {
    if (!pending() || !terminal(*entry) || entry->outstanding != 0) return false;
    auto retained_state = std::move(state);
    auto retained_entry = std::move(entry);
    retained_entry->abandoned = true;
    retained_state->collect();
    return true;
}
bool llama_kv_stream_publication_complete_sync(llama_kv_stream_publication_ticket & ticket, uint32_t pair) noexcept {
    std::array<llama_kv_stream_publication_completion, 4> completions;
    size_t next = 0;
    for (auto domain : {llama_kv_stream_publication_domain::device, llama_kv_stream_publication_domain::host}) {
        for (auto plane : {llama_kv_stream_publication_plane::k, llama_kv_stream_publication_plane::v}) {
            if (!ticket.submit(pair, plane, domain, completions[next])) return false;
            ++next;
        }
    }
    for (auto & completion : completions) if (!completion.finish()) return false;
    return ticket.ready(pair, llama_kv_stream_publication_domain::device) &&
        ticket.ready(pair, llama_kv_stream_publication_domain::host);
}


void llama_kv_stream_publication_ticket::abandon() noexcept {
    if (!pending()) return;
    auto retained_state = std::move(state);
    auto retained_entry = std::move(entry);
    if (!terminal(*retained_entry)) retained_state->cancel(retained_entry);
    retained_entry->abandoned = true;
    retained_state->collect();
}

std::unique_ptr<llama_kv_stream_publications> llama_kv_stream_publications::create(
        const llama_kv_stream_publication_config & config) {
    if (!config.generation || !config.next_sequence || config.tokens > config.capacity) return {};
    try {
        auto result = std::unique_ptr<llama_kv_stream_publications>(new llama_kv_stream_publications);
        result->state = std::make_shared<llama_kv_stream_publication_state>();
        result->state->config = config;
        result->state->frontiers = {config.tokens, config.tokens, config.tokens, config.tokens};
        result->state->next_sequence = config.next_sequence;
        return result;
    } catch (const std::bad_alloc &) {
        return {};
    }
}

llama_kv_stream_publications::~llama_kv_stream_publications() {
    if (state) state->shutdown();
}

bool llama_kv_stream_publications::reserve(size_t first, size_t count, uint32_t pairs,
        const std::vector<std::shared_ptr<void>> & owners,
        const std::vector<ggml_backend_memory_lease_t> & leases,
        llama_kv_stream_publication_ticket & output) {
    if (!state || state->closed || state->sequence_exhausted || output.pending() || !count || !pairs ||
            first != state->frontiers.reserved || count > state->config.capacity - first) return false;
    for (const auto & owner : owners) if (!owner) return false;
    for (auto * lease : leases) {
        ggml_backend_memory_region region;
        if (!lease || !ggml_backend_memory_lease_get_region(lease, &region) || !ggml_backend_memory_lease_buffer(lease)) return false;
    }
    try {
        auto entry = std::make_shared<llama_kv_stream_publication_entry>();
        if (pairs > entry->pairs.max_size() || owners.size() > entry->owners.max_size() || leases.size() > entry->leases.max_size()) return false;
        entry->sequence = state->next_sequence;
        entry->generation = state->config.generation;
        entry->first = first;
        entry->count = count;
        entry->pairs.resize(pairs);
        entry->owners = owners;
        entry->leases.reserve(leases.size());
        for (auto * lease : leases) {
            const bool duplicate = std::any_of(entry->leases.begin(), entry->leases.end(),
                [lease](const lease_ptr & existing) { return existing.get() == lease; });
            if (!duplicate) entry->leases.emplace_back(ggml_backend_memory_lease_retain(lease), ggml_backend_memory_lease_free);
        }
        state->entries.push_back(entry);
        state->frontiers.reserved = first + count;
        if (state->next_sequence == UINT64_MAX) state->sequence_exhausted = true;
        else ++state->next_sequence;
        output = llama_kv_stream_publication_ticket(state, std::move(entry));
        return true;
    } catch (const std::bad_alloc &) {
        return false;
    }
}

llama_kv_stream_publication_frontiers llama_kv_stream_publications::frontiers() const noexcept {
    return state ? state->frontiers : llama_kv_stream_publication_frontiers{};
}

uint64_t llama_kv_stream_publications::generation() const noexcept {
    return state ? state->config.generation : 0;
}

size_t llama_kv_stream_publications::pending() const noexcept {
    return state ? state->entries.size() : 0;
}

bool llama_kv_stream_publications::failed() const noexcept {
    return !state || state->closed;
}

bool llama_kv_stream_publications::exhausted() const noexcept {
    return !state || state->sequence_exhausted;
}
