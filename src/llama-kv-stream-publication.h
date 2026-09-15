#pragma once

#include "../ggml/src/ggml-backend-memory.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

enum class llama_kv_stream_publication_domain : uint8_t {
    device,
    host,
};

enum class llama_kv_stream_publication_plane : uint8_t {
    k,
    v,
};

struct llama_kv_stream_publication_config {
    size_t tokens = 0;
    size_t capacity = 0;
    uint64_t generation = 0;
    uint64_t next_sequence = 0;
};

struct llama_kv_stream_publication_frontiers {
    size_t reserved = 0;
    size_t device = 0;
    size_t host = 0;
    size_t committed = 0;
};

struct llama_kv_stream_publication_state;
struct llama_kv_stream_publication_entry;

// One submitted plane/domain operation. Dropping it unfinished fails the publication.
class llama_kv_stream_publication_completion {
public:
    llama_kv_stream_publication_completion() = default;
    ~llama_kv_stream_publication_completion();
    llama_kv_stream_publication_completion(llama_kv_stream_publication_completion && other) noexcept;
    llama_kv_stream_publication_completion & operator=(llama_kv_stream_publication_completion && other) noexcept;
    llama_kv_stream_publication_completion(const llama_kv_stream_publication_completion &) = delete;
    llama_kv_stream_publication_completion & operator=(const llama_kv_stream_publication_completion &) = delete;

    bool pending() const noexcept;
    bool finish(bool success = true) noexcept;

private:
    friend class llama_kv_stream_publication_ticket;
    llama_kv_stream_publication_completion(std::shared_ptr<llama_kv_stream_publication_state> state,
            std::shared_ptr<llama_kv_stream_publication_entry> entry, uint32_t pair,
            llama_kv_stream_publication_plane plane, llama_kv_stream_publication_domain domain);

    std::shared_ptr<llama_kv_stream_publication_state> state;
    std::shared_ptr<llama_kv_stream_publication_entry> entry;
    uint32_t pair = 0;
    llama_kv_stream_publication_plane plane = llama_kv_stream_publication_plane::k;
    llama_kv_stream_publication_domain domain = llama_kv_stream_publication_domain::device;
};

// Move-only handle for one token range containing one K/V pair per participating layer.
class llama_kv_stream_publication_ticket {
public:
    llama_kv_stream_publication_ticket() = default;
    ~llama_kv_stream_publication_ticket();
    llama_kv_stream_publication_ticket(llama_kv_stream_publication_ticket && other) noexcept;
    llama_kv_stream_publication_ticket & operator=(llama_kv_stream_publication_ticket && other) noexcept;
    llama_kv_stream_publication_ticket(const llama_kv_stream_publication_ticket &) = delete;
    llama_kv_stream_publication_ticket & operator=(const llama_kv_stream_publication_ticket &) = delete;

    bool pending() const noexcept;
    uint64_t sequence() const noexcept;
    uint64_t generation() const noexcept;
    size_t first() const noexcept;
    size_t count() const noexcept;
    uint32_t pairs() const noexcept;
    bool ready(uint32_t pair, llama_kv_stream_publication_domain domain) const noexcept;
    bool committed() const noexcept;
    bool failed() const noexcept;

    // Submission reserves one completion handle before backend work is queued.
    bool submit(uint32_t pair, llama_kv_stream_publication_plane plane,
            llama_kv_stream_publication_domain domain, llama_kv_stream_publication_completion & output) noexcept;
    bool cancel() noexcept;
    bool retire() noexcept;

private:
    friend class llama_kv_stream_publications;
    llama_kv_stream_publication_ticket(std::shared_ptr<llama_kv_stream_publication_state> state,
            std::shared_ptr<llama_kv_stream_publication_entry> entry);
    void abandon() noexcept;

    std::shared_ptr<llama_kv_stream_publication_state> state;
    std::shared_ptr<llama_kv_stream_publication_entry> entry;
};
// Mark one pair device-ready and host-ready through the common completion protocol.
bool llama_kv_stream_publication_complete_sync(llama_kv_stream_publication_ticket & ticket, uint32_t pair) noexcept;


// Owner-thread-only ordered publication state. Backend callbacks must return completion to its owner thread.
class llama_kv_stream_publications {
public:
    static std::unique_ptr<llama_kv_stream_publications> create(const llama_kv_stream_publication_config & config);
    ~llama_kv_stream_publications();
    llama_kv_stream_publications(const llama_kv_stream_publications &) = delete;
    llama_kv_stream_publications & operator=(const llama_kv_stream_publications &) = delete;

    bool reserve(size_t first, size_t count, uint32_t pairs,
            const std::vector<std::shared_ptr<void>> & owners,
            const std::vector<ggml_backend_memory_lease_t> & leases,
            llama_kv_stream_publication_ticket & output);
    llama_kv_stream_publication_frontiers frontiers() const noexcept;
    uint64_t generation() const noexcept;
    size_t pending() const noexcept;
    bool failed() const noexcept;
    bool exhausted() const noexcept;

private:
    llama_kv_stream_publications() = default;
    std::shared_ptr<llama_kv_stream_publication_state> state;
};
