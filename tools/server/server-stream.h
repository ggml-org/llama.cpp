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
//
// lifetime safety: holds a shared_ptr<atomic<bool>> alive also captured by the session's
// stop_producer hook. cleanup() sets alive=false and clears the hook; it must run while the
// response the hook calls stop() on is still alive. ~server_res_generator() does this explicitly.
struct stream_pipe_producer : stream_pipe {
    ~stream_pipe_producer() override;

    bool write(const char * data, size_t len);

    // mark the natural end on the wire so a later close() is a no-op
    void done();

    // on a peer drop, pump the response next() into the ring buffer until done. runs on the http
    // worker from on_complete, no-op after done() or cancel
    void close();

    // disarm the stop hook and drop the alive guard, must run while the response the hook
    // references is still alive. idempotent, the destructor calls it too
    void cleanup();

    // res.stop() is invoked when the session is cancelled, the alive guard ensures stop() is not
    // called after cleanup() has run
    static std::shared_ptr<stream_pipe_producer> create(stream_session_ptr session, server_http_res & res);

private:
    explicit stream_pipe_producer(stream_session_ptr session);

    bool                                done_ = false;
    std::shared_ptr<std::atomic<bool>>  alive_;
    server_http_res *                   res_ = nullptr;
};

void server_stream_session_manager_start();
void server_stream_session_manager_stop();

// route handler factories wired under /v1/stream/* by server.cpp
server_http_context::handler_t server_stream_make_get_handler();
server_http_context::handler_t server_stream_make_lookup_handler();
server_http_context::handler_t server_stream_make_delete_handler();

// extract the X-Conversation-Id header value (case-insensitive), empty when absent
std::string server_stream_conv_id_from_headers(const std::map<std::string, std::string> & headers);

// on an X-Conversation-Id header, create or replace the session and attach a producer pipe to res
void server_stream_session_attach_pipe(server_http_res & res, const std::map<std::string, std::string> & headers);

// should_stop closure that ignores peer disconnect when a pipe is attached, so only an explicit
// DELETE stops the producer and generation keeps flowing into the ring buffer. without a pipe it
// delegates to fallback, the legacy non-resumable flow
std::function<bool()> server_stream_aware_should_stop(server_http_res * res, std::function<bool()> fallback);
