#include "server-stream.h"
#include "server-common.h"
#include "server-http.h"

#include <chrono>
#include <memory>
#include <utility>

namespace {
constexpr int64_t STREAM_SESSION_TTL_SECONDS         = 300;
constexpr size_t  STREAM_SESSION_MAX_BYTES           = 4 * 1024 * 1024;
constexpr int64_t STREAM_SESSION_GC_INTERVAL_SECONDS = 60;
constexpr int64_t STREAM_READ_WAKE_INTERVAL_MS       = 200;

// returns unix time in seconds
int64_t now_seconds() {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();
}
}

stream_session::stream_session(std::string conversation_id_, size_t max_bytes_)
    : conversation_id(std::move(conversation_id_))
    , started_ts(now_seconds())
    , prefix_dropped(0)
    , cap_bytes(max_bytes_)
    , done(false)
    , completed_ts(0) {
    buffer.reserve(64 * 1024);
}

bool stream_session::append(const char * data, size_t len) {
    if (len == 0) {
        return true;
    }
    {
        std::lock_guard<std::mutex> lock(mu);
        if (done.load(std::memory_order_relaxed)) {
            return false;
        }
        if (len >= cap_bytes) {
            // single chunk bigger than the cap, keep only the tail that fits
            size_t skip = len - cap_bytes;
            prefix_dropped += buffer.size() + skip;
            buffer.clear();
            buffer.insert(buffer.end(), data + skip, data + len);
        } else {
            size_t needed = buffer.size() + len;
            if (needed > cap_bytes) {
                size_t to_drop = needed - cap_bytes;
                buffer.erase(buffer.begin(), buffer.begin() + to_drop);
                prefix_dropped += to_drop;
            }
            buffer.insert(buffer.end(), data, data + len);
        }
    }
    cv.notify_all();
    return true;
}

void stream_session::finalize() {
    bool was_done = done.exchange(true, std::memory_order_acq_rel);
    if (was_done) {
        return;
    }
    completed_ts.store(now_seconds(), std::memory_order_release);
    cv.notify_all();
}

stream_read_status stream_session::read_from(size_t offset,
        const std::function<bool(const char *, size_t)> & sink,
        const std::function<bool()> & should_stop) {
    std::unique_lock<std::mutex> lock(mu);
    while (true) {
        if (should_stop && should_stop()) {
            return stream_read_status::OK;
        }
        if (offset < prefix_dropped) {
            return stream_read_status::OFFSET_LOST;
        }
        size_t logical_end = prefix_dropped + buffer.size();
        if (offset < logical_end) {
            size_t local_off = offset - prefix_dropped;
            size_t n         = buffer.size() - local_off;
            // copy the available chunk under the lock, release before calling the sink
            std::vector<char> chunk(buffer.begin() + local_off, buffer.begin() + local_off + n);
            offset += n;
            lock.unlock();
            bool keep_going = sink(chunk.data(), chunk.size());
            if (!keep_going) {
                return stream_read_status::OK;
            }
            lock.lock();
            continue;
        }
        if (done.load(std::memory_order_acquire)) {
            return stream_read_status::OK;
        }
        // wait for new bytes, finalize, or a periodic wake to re check should_stop
        cv.wait_for(lock, std::chrono::milliseconds(STREAM_READ_WAKE_INTERVAL_MS));
    }
}

bool stream_session::is_done() const {
    return done.load(std::memory_order_acquire);
}

size_t stream_session::total_size() const {
    std::lock_guard<std::mutex> lock(mu);
    return prefix_dropped + buffer.size();
}

size_t stream_session::dropped_prefix() const {
    std::lock_guard<std::mutex> lock(mu);
    return prefix_dropped;
}

int64_t stream_session::completed_at() const {
    return completed_ts.load(std::memory_order_acquire);
}

void stream_session::set_stop_producer(std::function<void()> fn) {
    std::lock_guard<std::mutex> lock(mu);
    stop_producer = std::move(fn);
}

void stream_session::cancel() {
    // copy the hook under the lock then invoke outside, the producer side may grab queue locks
    // and we do not want to hold our mu across that path
    std::function<void()> fn;
    {
        std::lock_guard<std::mutex> lock(mu);
        fn = stop_producer;
    }
    if (fn) {
        fn();
    }
}

stream_session_manager::stream_session_manager()
    : running(false)
    , drain_shutdown(std::make_shared<std::atomic<bool>>(false)) {
}

stream_session_manager::~stream_session_manager() {
    stop_gc();
}

stream_session_ptr stream_session_manager::create_or_replace(const std::string & conversation_id) {
    // evict any previous session on the same conv, this guarantees the invariant
    // "one conv = at most one live session" and propagates cancel to its producer
    stream_session_ptr previous;
    auto fresh = std::make_shared<stream_session>(conversation_id, STREAM_SESSION_MAX_BYTES);
    {
        std::unique_lock<std::shared_mutex> lock(map_mu);
        auto it = sessions.find(conversation_id);
        if (it != sessions.end()) {
            previous = it->second;
            it->second = fresh;
        } else {
            sessions.emplace(conversation_id, fresh);
        }
    }
    if (previous) {
        previous->cancel();
        previous->finalize();
    }
    return fresh;
}

stream_session_ptr stream_session_manager::get(const std::string & conversation_id) {
    std::shared_lock<std::shared_mutex> lock(map_mu);
    auto it = sessions.find(conversation_id);
    if (it == sessions.end()) {
        return nullptr;
    }
    return it->second;
}

std::vector<stream_session_ptr> stream_session_manager::list_all() const {
    std::vector<stream_session_ptr> out;
    std::shared_lock<std::shared_mutex> lock(map_mu);
    out.reserve(sessions.size());
    for (auto & kv : sessions) {
        out.push_back(kv.second);
    }
    return out;
}

void stream_session_manager::evict(const std::string & conversation_id) {
    stream_session_ptr s;
    {
        std::unique_lock<std::shared_mutex> lock(map_mu);
        auto it = sessions.find(conversation_id);
        if (it == sessions.end()) {
            return;
        }
        s = it->second;
        sessions.erase(it);
    }
    // finalize outside the map lock so any pending readers wake up and exit
    s->finalize();
}

void stream_session_manager::evict_and_cancel(const std::string & conversation_id) {
    stream_session_ptr s;
    {
        std::unique_lock<std::shared_mutex> lock(map_mu);
        auto it = sessions.find(conversation_id);
        if (it == sessions.end()) {
            return;
        }
        s = it->second;
        sessions.erase(it);
    }
    // signal the producer side first so the inference is cancelled at the queue level,
    // then finalize, which wakes any pending HTTP reader and lets the drain exit naturally
    s->cancel();
    s->finalize();
}

void stream_session_manager::start_gc() {
    if (running.exchange(true)) {
        return;
    }
    gc_thread = std::thread([this] { gc_loop(); });
}

void stream_session_manager::stop_gc() {
    drain_shutdown->store(true, std::memory_order_release);
    bool was_running = running.exchange(false);
    if (was_running) {
        {
            std::lock_guard<std::mutex> lock(gc_wake_mu);
        }
        gc_wake_cv.notify_all();
        if (gc_thread.joinable()) {
            gc_thread.join();
        }
    }
    // finalize all live sessions so no reader ever hangs
    std::vector<stream_session_ptr> snapshot;
    {
        std::unique_lock<std::shared_mutex> lock(map_mu);
        snapshot.reserve(sessions.size());
        for (auto & kv : sessions) {
            snapshot.push_back(kv.second);
        }
        sessions.clear();
    }
    for (auto & s : snapshot) {
        s->finalize();
    }
}

void stream_session_manager::gc_loop() {
    while (running.load(std::memory_order_acquire)) {
        {
            std::unique_lock<std::mutex> lock(gc_wake_mu);
            gc_wake_cv.wait_for(lock,
                std::chrono::seconds(STREAM_SESSION_GC_INTERVAL_SECONDS),
                [this] { return !running.load(std::memory_order_acquire); });
        }
        if (!running.load(std::memory_order_acquire)) {
            return;
        }
        int64_t cutoff = now_seconds() - STREAM_SESSION_TTL_SECONDS;
        std::vector<stream_session_ptr> to_drop;
        {
            std::unique_lock<std::shared_mutex> lock(map_mu);
            for (auto it = sessions.begin(); it != sessions.end(); ) {
                int64_t completed = it->second->completed_at();
                if (completed != 0 && completed <= cutoff) {
                    to_drop.push_back(it->second);
                    it = sessions.erase(it);
                } else {
                    ++it;
                }
            }
        }
        // finalize outside the map lock, idempotent if the session was already done
        for (auto & s : to_drop) {
            s->finalize();
        }
    }
}

// process wide manager, lifecycle controlled by llama-server main() via start_gc/stop_gc
stream_session_manager g_stream_sessions;

// helper, builds the standard error response and assigns it to a brand new http_res
static server_http_res_ptr make_error_response(int status, const std::string & message, error_type type) {
    auto res = std::make_unique<server_http_res>();
    json err = format_error_response(message, type);
    res->status = json_value(err, "code", status);
    res->content_type = "application/json; charset=utf-8";
    res->data = safe_json_to_str({{"error", err}});
    return res;
}

server_http_context::handler_t make_stream_get_handler() {
    return [](const server_http_req & req) -> server_http_res_ptr {
        // GET /v1/stream/<conv_id>?from=N replays the SSE bytes already buffered for the
        // session, blocks for more bytes when the session is still running, returns when
        // the session is finalized. the body is streamed back as text/event-stream so the
        // browser EventSource can attach to it like a fresh request
        std::string conv_id = req.get_param("conv_id");
        if (conv_id.empty()) {
            return make_error_response(400, "Missing conversation id in path", ERROR_TYPE_INVALID_REQUEST);
        }
        auto session = g_stream_sessions.get(conv_id);
        if (!session) {
            return make_error_response(404, "Stream not found or expired", ERROR_TYPE_NOT_FOUND);
        }
        size_t from = 0;
        std::string from_str = req.get_param("from");
        if (!from_str.empty()) {
            try {
                from = static_cast<size_t>(std::stoull(from_str));
            } catch (const std::exception &) {
                return make_error_response(400, "Invalid 'from' offset", ERROR_TYPE_INVALID_REQUEST);
            }
        }
        if (from < session->dropped_prefix()) {
            return make_error_response(400, "Stream offset lost, please restart", ERROR_TYPE_INVALID_REQUEST);
        }
        auto res = std::make_unique<server_http_res>();
        res->status = 200;
        res->content_type = "text/event-stream";
        // the next closure reads from the ring buffer at the requested offset, blocks until
        // bytes arrive or the session finalizes. exit each call after draining the available
        // chunk so set_chunked_content_provider gets a chance to flush to the socket
        auto offset_ptr      = std::make_shared<size_t>(from);
        auto session_capture = session;
        res->next = [session_capture, offset_ptr, &req](std::string & output) -> bool {
            bool got_any = false;
            session_capture->read_from(*offset_ptr,
                [&](const char * d, size_t n) {
                    output.append(d, n);
                    *offset_ptr += n;
                    got_any = true;
                    return false;
                },
                req.should_stop);
            return got_any;
        };
        return res;
    };
}

server_http_context::handler_t make_streams_list_handler() {
    return [](const server_http_req & req) -> server_http_res_ptr {
        // GET /v1/streams returns sessions as a JSON array. with conversation_id set, every
        // session whose key is exactly that id or starts with "<id>::" matches, so a single
        // call returns every per model variant for a given conv. without conversation_id,
        // every live or recently completed session known to this server, used by the WebUI
        // at mount and on visibilitychange to populate the sidebar spinners
        std::string conversation_id = req.get_param("conversation_id");
        std::vector<stream_session_ptr> sessions;
        if (conversation_id.empty()) {
            sessions = g_stream_sessions.list_all();
        } else {
            const std::string with_sep = conversation_id + "::";
            auto all = g_stream_sessions.list_all();
            for (auto & s : all) {
                if (s->conversation_id == conversation_id) {
                    sessions.push_back(s);
                } else if (s->conversation_id.compare(0, with_sep.size(), with_sep) == 0) {
                    sessions.push_back(s);
                }
            }
        }
        json arr = json::array();
        for (auto & s : sessions) {
            arr.push_back({
                {"conversation_id", s->conversation_id},
                {"is_done",         s->is_done()},
                {"total_bytes",     s->total_size()},
                {"started_at",      s->started_ts},
                {"completed_at",    s->completed_at()},
            });
        }
        auto res = std::make_unique<server_http_res>();
        res->status = 200;
        res->content_type = "application/json; charset=utf-8";
        res->data = safe_json_to_str(arr);
        return res;
    };
}

server_http_context::handler_t make_stream_delete_handler() {
    return [](const server_http_req & req) -> server_http_res_ptr {
        // DELETE /v1/stream/<conv_id> is the explicit user Stop, cancels the producer hook
        // wired by handle_completions_impl and evicts the buffer. idempotent, a session that
        // already finalized or was never created returns 204 either way
        std::string conv_id = req.get_param("conv_id");
        if (conv_id.empty()) {
            return make_error_response(400, "Missing conversation id in path", ERROR_TYPE_INVALID_REQUEST);
        }
        SRV_INF("DELETE /v1/stream/%s -> evict_and_cancel\n", conv_id.c_str());
        g_stream_sessions.evict_and_cancel(conv_id);
        auto res = std::make_unique<server_http_res>();
        res->status = 204;
        res->content_type = "application/json";
        return res;
    };
}
