#include "server-stream.h"
#include "server-common.h"
#include "server-http.h"
#include "server-queue.h"

#include <chrono>
#include <memory>
#include <utility>
#include <shared_mutex>

enum class stream_read_status {
    OK,
    OFFSET_LOST,
};

namespace {
constexpr int64_t STREAM_SESSION_TTL_SECONDS         = 300;
constexpr size_t  STREAM_SESSION_MAX_BYTES           = 4 * 1024 * 1024;
constexpr int64_t STREAM_SESSION_GC_INTERVAL_SECONDS = 60;
constexpr int64_t STREAM_READ_WAKE_INTERVAL_MS       = 200;

int64_t now_seconds() {
    return std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();
}
}

// owns all live sessions keyed by conversation_id, one conv = at most one live session.
// a periodic GC evicts expired ones
class stream_session_manager {
public:
    stream_session_manager();
    ~stream_session_manager();

    stream_session_manager(const stream_session_manager &)             = delete;
    stream_session_manager & operator=(const stream_session_manager &) = delete;

    // install a new session, evicting and cancelling any previous one. conversation_id must be non empty
    stream_session_ptr create_or_replace(const std::string & conversation_id);

    stream_session_ptr get(const std::string & conversation_id);

    std::vector<stream_session_ptr> list_all() const;

    void evict(const std::string & conversation_id);

    void evict_and_cancel(const std::string & conversation_id);

    void start_gc();
    void stop_gc();

private:
    void gc_loop();

    mutable std::shared_mutex                           map_mu;
    std::unordered_map<std::string, stream_session_ptr> sessions; // key: conversation_id
    std::thread                                         gc_thread;
    bool                                                running;
    std::mutex                                          gc_wake_mu;
    std::condition_variable                             gc_wake_cv;
};

// process wide manager, lifecycle controlled by llama-server main() via start_gc/stop_gc
static stream_session_manager g_stream_sessions;

void server_stream_session_manager_start() {
    g_stream_sessions.start_gc();
}

void server_stream_session_manager_stop() {
    g_stream_sessions.stop_gc();
}

struct stream_session {
    std::string conversation_id;
    int64_t     started_ts; // unix seconds at construction

    stream_session(std::string conversation_id_, size_t max_bytes_);
    stream_session(const stream_session &)             = delete;
    stream_session & operator=(const stream_session &) = delete;

    bool append(const char * data, size_t len);

    void finalize();

    // drain from offset into sink, blocking for more bytes or finalize. OFFSET_LOST if offset
    // fell below the dropped prefix
    stream_read_status read_from(size_t offset,
        const std::function<bool(const char *, size_t)> & sink,
        const std::function<bool()> & should_stop);

    bool    is_done() const;
    bool    is_cancelled() const;
    size_t  total_size() const;     // bytes that ever entered the session
    size_t  dropped_prefix() const; // bytes evicted from the front due to cap
    int64_t completed_at() const;   // 0 while alive, unix seconds after finalize

    void cancel();

private:
    mutable std::mutex      mu;
    std::condition_variable cv;
    std::vector<char>       buffer;
    size_t                  prefix_dropped;
    size_t                  cap_bytes;
    bool                    done;
    std::atomic<bool>       cancelled; // polled lock-free by the should_stop closure, no mu
    int64_t                 completed_ts;
};
stream_session::stream_session(std::string conversation_id_, size_t max_bytes_)
    : conversation_id(std::move(conversation_id_))
    , started_ts(now_seconds())
    , prefix_dropped(0)
    , cap_bytes(max_bytes_)
    , done(false)
    , cancelled(false)
    , completed_ts(0) {
    buffer.reserve(64 * 1024);
}

bool stream_session::append(const char * data, size_t len) {
    if (len == 0) {
        return true;
    }
    {
        std::lock_guard<std::mutex> lock(mu);
        if (done) {
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
    {
        std::lock_guard<std::mutex> lock(mu);
        if (done) {
            return;
        }
        done         = true;
        completed_ts = now_seconds();
    }
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
        if (done) {
            return stream_read_status::OK;
        }
        // wait for new bytes, finalize, or a periodic wake to re check should_stop
        cv.wait_for(lock, std::chrono::milliseconds(STREAM_READ_WAKE_INTERVAL_MS));
    }
}

bool stream_session::is_done() const {
    std::lock_guard<std::mutex> lock(mu);
    return done;
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
    std::lock_guard<std::mutex> lock(mu);
    return completed_ts;
}

void stream_session::cancel() {
    // the should_stop closure on both the producer and any HTTP reader polls is_cancelled()
    // so flipping this is the only signal needed to unwind both sides
    cancelled.store(true, std::memory_order_release);
}

bool stream_session::is_cancelled() const {
    return cancelled.load(std::memory_order_acquire);
}

stream_session_manager::stream_session_manager()
    : running(false) {
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
            std::string live;
            for (const auto & kv : sessions) {
                if (!live.empty()) live += ", ";
                live += kv.first;
            }
            SRV_WRN("stop on unknown stream session, conv_id=%s matched nothing, %zu live: [%s]\n",
                    conversation_id.c_str(), sessions.size(), live.c_str());
            return;
        }
        s = it->second;
        sessions.erase(it);
    }
    // cancel first so the producer's on_complete() drain loop and any pending HTTP reader
    // observe is_cancelled() and stop pulling further output, then finalize to wake readers
    // blocked in read_from(). note: this does not interrupt the underlying generation itself,
    // which keeps running to its own natural stop condition (EOS/max_tokens)
    s->cancel();
    s->finalize();
}

void stream_session_manager::start_gc() {
    {
        std::lock_guard<std::mutex> lock(gc_wake_mu);
        if (running) {
            return;
        }
        running = true;
    }
    gc_thread = std::thread([this] { gc_loop(); });
}

void stream_session_manager::stop_gc() {
    bool was_running;
    {
        std::lock_guard<std::mutex> lock(gc_wake_mu);
        was_running = running;
        running = false;
    }
    if (was_running) {
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
    while (true) {
        {
            std::unique_lock<std::mutex> lock(gc_wake_mu);
            gc_wake_cv.wait_for(lock,
                std::chrono::seconds(STREAM_SESSION_GC_INTERVAL_SECONDS),
                [this] { return !running; });
            if (!running) {
                return;
            }
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

// stream_pipe

// consumer end: read-only replay of the ring buffer, the destructor does not finalize the session
struct stream_pipe_consumer : stream_pipe {
    stream_read_status read(size_t & offset,
        const std::function<bool(const char *, size_t)> & sink,
        const std::function<bool()> & should_stop);

    static std::shared_ptr<stream_pipe_consumer> create(stream_session_ptr session);

private:
    explicit stream_pipe_consumer(stream_session_ptr session);
};

stream_pipe::stream_pipe(stream_session_ptr session)
    : session_(std::move(session)) {
}

bool stream_pipe::is_cancelled() const {
    return session_->is_cancelled();
}

// stream_pipe_producer

stream_pipe_producer::stream_pipe_producer(stream_session_ptr session)
    : stream_pipe(std::move(session)) {
}

stream_pipe_producer::~stream_pipe_producer() {
    session_->finalize();
}

bool stream_pipe_producer::write(const char * data, size_t len) {
    return session_->append(data, len);
}

stream_pipe_producer * stream_pipe_producer::create(stream_session_ptr session) {
    return new stream_pipe_producer(std::move(session));
}

// stream_pipe_consumer

stream_pipe_consumer::stream_pipe_consumer(stream_session_ptr session)
    : stream_pipe(std::move(session)) {
}

stream_read_status stream_pipe_consumer::read(size_t & offset,
        const std::function<bool(const char *, size_t)> & sink,
        const std::function<bool()> & should_stop) {
    return session_->read_from(offset, sink, should_stop);
}

std::shared_ptr<stream_pipe_consumer> stream_pipe_consumer::create(stream_session_ptr session) {
    return std::shared_ptr<stream_pipe_consumer>(new stream_pipe_consumer(std::move(session)));
}

// helper, builds the standard error response and assigns it to a brand new http_res
static server_http_res_ptr make_error_response(int status, const std::string & message, error_type type) {
    auto res = std::make_unique<server_http_res>();
    json err = format_error_response(message, type);
    res->status = json_value(err, "code", status);
    res->content_type = "application/json; charset=utf-8";
    res->data = safe_json_to_str({{"error", err}});
    return res;
}

server_http_context::handler_t server_stream_make_get_handler() {
    return [](const server_http_req & req) -> server_http_res_ptr {
        // GET /v1/stream/<conv_id>?from=N replays buffered SSE bytes then blocks for live
        // bytes until the session finalizes, streamed as text/event-stream for EventSource
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
        auto offset_ptr = std::make_shared<size_t>(from);
        // consumer pipe: read-only, does not finalize the session on destruction
        auto pipe = stream_pipe_consumer::create(session);
        res->next = [pipe, offset_ptr, &req](std::string & output) -> bool {
            bool got_any = false;
            pipe->read(*offset_ptr,
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

server_http_context::handler_t server_stream_make_lookup_handler() {
    return [](const server_http_req & req) -> server_http_res_ptr {
        // POST /v1/streams/lookup returns the matching sessions, only for ids the caller already
        // knows. each id matches the exact key and any "<id>::<model>" per model variant
        std::vector<std::string> requested;
        try {
            json body = json::parse(req.body);
            if (body.contains("conversation_ids") && body["conversation_ids"].is_array()) {
                for (const auto & v : body["conversation_ids"]) {
                    if (v.is_string()) {
                        std::string id = v.get<std::string>();
                        if (!id.empty()) {
                            requested.push_back(std::move(id));
                        }
                    }
                }
            }
        } catch (const std::exception & e) {
            auto res = std::make_unique<server_http_res>();
            res->status = 400;
            res->content_type = "application/json; charset=utf-8";
            res->data = safe_json_to_str({{"error", {{"message", std::string("invalid body: ") + e.what()},
                                                     {"type", "invalid_request_error"}}}});
            return res;
        }

        std::vector<stream_session_ptr> sessions;
        if (!requested.empty()) {
            auto all = g_stream_sessions.list_all();
            for (const auto & rid : requested) {
                const std::string with_sep = rid + "::";
                for (auto & s : all) {
                    if (s->conversation_id == rid ||
                        s->conversation_id.compare(0, with_sep.size(), with_sep) == 0) {
                        sessions.push_back(s);
                    }
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

server_http_context::handler_t server_stream_make_delete_handler() {
    return [](const server_http_req & req) -> server_http_res_ptr {
        // DELETE /v1/stream/<conv_id> is the explicit user Stop, cancels the producer and evicts
        // the buffer. idempotent, returns 204 even if the session was already gone
        std::string conv_id = req.get_param("conv_id");
        if (conv_id.empty()) {
            return make_error_response(400, "Missing conversation id in path", ERROR_TYPE_INVALID_REQUEST);
        }
        SRV_TRC("DELETE /v1/stream/%s -> evict_and_cancel\n", conv_id.c_str());
        g_stream_sessions.evict_and_cancel(conv_id);
        auto res = std::make_unique<server_http_res>();
        res->status = 204;
        res->content_type = "application/json";
        return res;
    };
}

std::string server_stream_conv_id_from_headers(const std::map<std::string, std::string> & headers) {
    // case-insensitive scan for x-conversation-id
    static constexpr char   target[]   = "x-conversation-id";
    static constexpr size_t target_len = sizeof(target) - 1;
    for (const auto & [hk, hv] : headers) {
        if (hk.size() != target_len) continue;
        bool match = true;
        for (size_t i = 0; i < target_len; ++i) {
            char c = hk[i];
            if (c >= 'A' && c <= 'Z') c = char(c + 32);
            if (c != target[i]) { match = false; break; }
        }
        if (match) {
            return hv;
        }
    }
    return std::string();
}

static stream_pipe_producer * server_stream_create_spipe(const std::map<std::string, std::string> & headers) {
    std::string conversation_id = server_stream_conv_id_from_headers(headers);
    SRV_TRC("conv_id=%s (empty=%d)\n", conversation_id.c_str(), conversation_id.empty() ? 1 : 0);
    if (conversation_id.empty()) {
        return nullptr;
    }
    auto session = g_stream_sessions.create_or_replace(conversation_id);
    return stream_pipe_producer::create(session);
}

//
// server_res_spipe
//

void server_res_spipe::set_req(const server_http_req * req) {
    this->req = req;
    // optionally attach spipe to the response when X-Conversation-Id is present
    spipe.reset(server_stream_create_spipe(req->headers));
}

bool server_res_spipe::conn_alive() {
    GGML_ASSERT(req != nullptr);
    return !req->should_stop();
}

bool server_res_spipe::should_stop() {
    if (spipe) {
        // note: if DELETE /v1/stream/<conv_id> is called, is_cancelled() will be true
        return spipe->is_cancelled();
    } else {
        return !conn_alive();
    }
}

void server_res_spipe::on_complete() {
    if (!spipe || next_finished) {
        return;
    }
    // an empty next_orig means set_next() never ran: the request failed before streaming
    // started, typically a params validation throw. evict the session installed by set_req()
    // so the failed request leaves nothing behind for discovery or replay
    if (!next_orig) {
        g_stream_sessions.evict(server_stream_conv_id_from_headers(req->headers));
        return;
    }
    std::string chunk;
    while (!spipe->is_cancelled()) {
        chunk.clear();
        bool has_next = next_orig(chunk);
        if (!chunk.empty()) {
            spipe->write(chunk.data(), chunk.size());
        }
        if (!has_next) {
            break;
        }
    }
}

void server_res_spipe::set_next(std::function<bool(std::string &)> next_fn) {
    next_orig = std::move(next_fn);
    next = [this](std::string & out) {
        bool has_next = next_orig(out);
        if (spipe) {
            // if spipe is set, tee-style pipe input to both HTTP and spipe
            spipe->write(out.data(), out.size());
        }
        if (!has_next) {
            next_finished = true;
        }
        return has_next;
    };
}
