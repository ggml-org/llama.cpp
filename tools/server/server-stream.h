#pragma once

#include "server-http.h"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

enum class stream_read_status {
    OK,
    OFFSET_LOST,
};

// streaming buffer for one generation, survives HTTP disconnect.
// the producer side pushes raw SSE bytes via append. HTTP readers drain from
// any offset via read_from. read_from blocks until new bytes arrive or the
// session is finalized. identity of the session is the conversation_id, no
// extra opaque token: one conv = at most one live session at a time
struct stream_session {
    std::string conversation_id;
    int64_t     started_ts; // unix seconds at construction, used by /v1/streams listing

    stream_session(std::string conversation_id_, size_t max_bytes_);
    stream_session(const stream_session &)             = delete;
    stream_session & operator=(const stream_session &) = delete;

    // append raw bytes, drops from the front if the cap is reached.
    // returns false if the session is already finalized
    bool append(const char * data, size_t len);

    // mark the session as complete, wakes all pending readers
    void finalize();

    // drain bytes from offset, calling sink for each chunk. blocks until more
    // bytes arrive or finalize is called. returns OK on clean exit, OFFSET_LOST
    // if offset falls below the dropped prefix
    stream_read_status read_from(size_t offset,
        const std::function<bool(const char *, size_t)> & sink,
        const std::function<bool()> & should_stop);

    bool    is_done() const;
    bool    is_cancelled() const;   // true when cancel() has been invoked
    size_t  total_size() const;     // bytes that ever entered the session
    size_t  dropped_prefix() const; // bytes evicted from the front due to cap
    int64_t completed_at() const;   // 0 while alive, unix seconds after finalize

    // attach a producer side stop hook, the drain sets this on startup so we can cancel its
    // underlying reader. pass an empty function to detach (drain must clear before destroying
    // its reader)
    void set_stop_producer(std::function<void()> fn);

    // invoke the stop hook if attached, signals the producer to abort its inference asap,
    // idempotent
    void cancel();

private:
    mutable std::mutex      mu;
    std::condition_variable cv;
    std::vector<char>       buffer;
    size_t                  prefix_dropped;
    size_t                  cap_bytes;
    std::atomic<bool>       done;
    std::atomic<bool>       cancelled;
    std::atomic<int64_t>    completed_ts;
    std::function<void()>   stop_producer; // protected by mu
};

using stream_session_ptr = std::shared_ptr<stream_session>;

// owns all live sessions, runs a periodic GC to evict expired ones.
// the map is keyed by conversation_id, so the invariant "one conv = at most one
// live session" is enforced at the type level
class stream_session_manager {
public:
    stream_session_manager();
    ~stream_session_manager();

    stream_session_manager(const stream_session_manager &)             = delete;
    stream_session_manager & operator=(const stream_session_manager &) = delete;

    // install a new session for this conversation, evicting and cancelling any previous one.
    // the conversation_id must be non empty, the caller is responsible for that check.
    // returns the new session
    stream_session_ptr create_or_replace(const std::string & conversation_id);

    // lookup, returns null if unknown or already evicted
    stream_session_ptr get(const std::string & conversation_id);

    // list every live or recently completed session, used by GET /v1/streams without filter
    std::vector<stream_session_ptr> list_all() const;

    // remove from the map and finalize, wakes any pending readers
    void evict(const std::string & conversation_id);

    // signal the producer to cancel asap then evict, used by the explicit user Stop path
    void evict_and_cancel(const std::string & conversation_id);

    void start_gc();
    void stop_gc();

    // shared atomic flipped to true on stop_gc, drain threads poll it to exit cleanly
    std::shared_ptr<std::atomic<bool>> shutdown_flag() const { return drain_shutdown; }

private:
    void gc_loop();

    mutable std::shared_mutex                           map_mu;
    std::unordered_map<std::string, stream_session_ptr> sessions; // key: conversation_id
    std::thread                                         gc_thread;
    std::atomic<bool>                                   running;
    std::mutex                                          gc_wake_mu;
    std::condition_variable                             gc_wake_cv;
    std::shared_ptr<std::atomic<bool>>                  drain_shutdown;
};

// the process wide stream session manager. defined in server-stream.cpp so the symbol
// resolves through the server-context static lib, both llama-server and llama-cli link it.
// start_gc() and stop_gc() are called explicitly from llama-server main(), llama-cli never
// touches it and leaves it idle. the destructor calls stop_gc() unconditionally so the
// process exit path is safe whether or not the GC thread was started
extern stream_session_manager g_stream_sessions;

// route handler factories. each builds a server_http_context::handler_t that operates
// directly on g_stream_sessions, server.cpp wires them under /v1/stream/* without going
// through server-context's server_routes. keeps the resumable stream surface confined to
// server-stream and server-http
server_http_context::handler_t make_stream_get_handler();
server_http_context::handler_t make_streams_lookup_handler();
server_http_context::handler_t make_stream_delete_handler();

// attach a resumable session to an outgoing streaming response. inspects the request headers
// for X-Conversation-Id, and when present creates or replaces a session on the global manager,
// then wires three closures on the response: tee mirrors each SSE chunk into the ring buffer,
// on_stream_end finalizes the session, and the session's stop_producer hook calls rd.stop()
// so the explicit DELETE /v1/stream/<conv_id> route can abort the underlying reader. no op
// when the header is absent. server-context just calls this and never touches the manager
struct server_response_reader; // forward declare to avoid pulling server-queue.h into the header
void stream_session_attach_hooks(server_http_res & res, server_response_reader & rd, const std::map<std::string, std::string> & headers);

// build a should_stop closure that suppresses the peer disconnect signal when a tee is
// attached to the response. used by handler lambdas that pump a server_response_reader so
// the producer keeps running past F5, only an explicit DELETE through the stop_producer
// hook is allowed to abort it. without a tee the returned closure is bit identical to the
// fallback, the legacy non resumable flow is unchanged
std::function<bool()> stream_aware_should_stop(server_http_res * res, std::function<bool()> fallback);
