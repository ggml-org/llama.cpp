#pragma once

#include "server-http.h"

#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>
#include <string>

// streaming buffer for one generation, survives HTTP disconnect. the producer appends SSE bytes,
// readers drain from any offset via read_from. keyed by conversation_id, one conv = one live session

struct stream_session;

using stream_session_ptr = std::shared_ptr<stream_session>;

// base of the producer/consumer pipe ends. virtual dtor so each runs its own teardown:
// the producer finalizes the session, the consumer leaves it untouched
struct stream_pipe {
    virtual ~stream_pipe() = default;

    bool is_cancelled() const;

protected:
    explicit stream_pipe(stream_session_ptr session);

    stream_session_ptr session_;
};

// producer end: writes chunks into the ring buffer and owns the session lifetime, finalizing it
// on destruction.
struct stream_pipe_producer : stream_pipe {
    ~stream_pipe_producer() override;

    bool write(const char * data, size_t len);

    static stream_pipe_producer * create(stream_session_ptr session);

private:
    explicit stream_pipe_producer(stream_session_ptr session);
};

void server_stream_session_manager_start();
void server_stream_session_manager_stop();

// route handler factories wired under /v1/stream/* by server.cpp
server_http_context::handler_t server_stream_make_get_handler();
server_http_context::handler_t server_stream_make_lookup_handler();
server_http_context::handler_t server_stream_make_delete_handler();

// extract the X-Conversation-Id header value (case-insensitive), empty when absent
std::string server_stream_conv_id_from_headers(const std::map<std::string, std::string> & headers);

// implement tee-style pipe (spipe) for "stream replay" functionality
struct server_res_spipe : server_http_res {
private:
    // if set, the stream survives a client disconnect:
    // connection kept alive, output is forwarded to spipe and reuse later
    std::unique_ptr<stream_pipe_producer> spipe;
    // if spipe is set, use this next_orig to implement tee-style pipe
    std::function<bool(std::string &)> next_orig;
    const server_http_req * req = nullptr;
    // set once next_orig reports no more data, so on_complete() doesn't re-drain a finished stream
    bool next_finished = false;

public:
    void set_req(const server_http_req * req);
    bool conn_alive();
    bool should_stop();
    void on_complete() override;
    void set_next(std::function<bool(std::string &)> next_fn);
};
