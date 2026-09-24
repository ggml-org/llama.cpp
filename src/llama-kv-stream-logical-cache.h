#pragma once

#include "llama-kv-stream-content.h"
#include "llama-kv-stream-writer.h"
#include "llama-kv-stream-publication.h"

struct llama_kv_stream_logical_identity {
    uint64_t id = 0;
    uint64_t generation = 0;
};

struct llama_kv_stream_logical_checkpoint {
    uint64_t id = 0;
    uint64_t generation = 0;
    size_t tokens = 0;
};

// Owns one authoritative cache and its ordered K/V publication frontier.
class llama_kv_stream_logical_cache {
public:
    static std::unique_ptr<llama_kv_stream_logical_cache> create(std::shared_ptr<llama_kv_stream_host> host);
    ~llama_kv_stream_logical_cache();

    std::shared_ptr<llama_kv_stream_host> host() const noexcept;
    std::shared_ptr<llama_kv_stream_content> content() const noexcept;
    llama_kv_stream_logical_identity identity() const noexcept;
    llama_kv_stream_logical_checkpoint checkpoint() const noexcept;
    llama_kv_stream_publication_frontiers frontiers() const noexcept;
    size_t tokens() const noexcept;

    // One append is active at a time. Host publication does not claim a device mirror.
    bool begin(size_t count);
    bool publish_host(llama_kv_stream_write & write);
    bool begin_generated(std::shared_ptr<llama_kv_stream_writer> writer, const ggml_tensor * k, const ggml_tensor * v);
    bool complete_generated();
    bool finish();
    // Cancel only before host publication and after queued backend work is drained.
    bool cancel();
    bool truncate(size_t tokens);
    // Caller restores saved host bytes before invoking this method.
    bool restore(const llama_kv_stream_logical_checkpoint & checkpoint);

private:
    explicit llama_kv_stream_logical_cache(std::shared_ptr<llama_kv_stream_host> host);
    bool complete(llama_kv_stream_publication_domain domain);
    bool covers(const llama_kv_stream_write & write) const noexcept;
    bool drain_generated();
    bool reset_frontier(size_t tokens);

    std::shared_ptr<llama_kv_stream_host> backing;
    std::shared_ptr<llama_kv_stream_content> authoritative;
    std::unique_ptr<llama_kv_stream_publications> publications;
    llama_kv_stream_publication_ticket pending;
    size_t committed = 0;
    uint64_t expected_generation = 0;
    bool host_ready = false;
    bool failed = false;
    std::shared_ptr<llama_kv_stream_writer> active_writer;
    llama_kv_stream_writer_completion generated[2];
    llama_kv_stream_write generated_write;
};
