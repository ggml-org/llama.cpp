#include "router-proxy.h"

#include "log.h"
#include "router-config.h"

#include <cpp-httplib/httplib.h>

#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <nlohmann/json.hpp>

namespace {
void copy_response_headers(const httplib::Headers & from, httplib::Response & to) {
    for (const auto & h : from) {
        if (h.first == "Transfer-Encoding" || h.first == "Content-Length") {
            continue;
        }
        to.set_header(h.first, h.second);
    }
}

bool matches_any_endpoint(const std::string & path, const std::vector<std::string> & patterns) {
    if (patterns.empty()) {
        return true;
    }

    for (const auto & pattern : patterns) {
        if (path.compare(0, pattern.size(), pattern) == 0) {
            return true;
        }
    }

    return false;
}
} // namespace

bool proxy_request(const httplib::Request & req,
                   httplib::Response &       res,
                   const std::string &       upstream_base,
                   const RouterOptions &     opts,
                   const std::vector<std::string> & proxy_endpoints,
                   const std::string &              override_path) {
    if (upstream_base.empty()) {
        res.status = 502;
        res.set_content("{\"error\":\"missing upstream\"}", "application/json");
        return false;
    }

    const std::string forwarded_path = !override_path.empty() ? override_path : (!req.target.empty() ? req.target : req.path);

    LOG_INF("Proxying %s %s to upstream %s\n", req.method.c_str(), forwarded_path.c_str(), upstream_base.c_str());
    auto client = std::make_shared<httplib::Client>(upstream_base.c_str());
    client->set_connection_timeout(opts.connection_timeout_s, 0);
    client->set_read_timeout(opts.read_timeout_s, 0);

    httplib::Headers headers = req.headers;
    headers.erase("Host");

    const std::string path         = forwarded_path;
    const std::string method       = req.method;
    const std::string request_body = req.body;

    if (!matches_any_endpoint(path, proxy_endpoints)) {
        LOG_WRN("Request %s not proxied because it does not match configured endpoints\n", path.c_str());
        res.status = 404;
        res.set_content("{\"error\":\"endpoint not proxied\"}", "application/json");
        return false;
    }

    const std::string content_type = req.get_header_value("Content-Type", "application/json");

    const auto accept_header = req.get_header_value("Accept");
    bool       wants_stream  = accept_header.find("text/event-stream") != std::string::npos;

    try {
        auto body_json = nlohmann::json::parse(req.body);
        if (body_json.value("stream", false)) {
            wants_stream = true;
        }
    } catch (const std::exception &) {
        if (req.body.find("\"stream\":true") != std::string::npos) {
            wants_stream = true;
        }
    }

    httplib::Result result;
    if (wants_stream) {
        struct StreamState {
            std::mutex              mutex;
            std::condition_variable cv;
            std::queue<std::string> chunks;
            bool                    done   = false;
            std::string             content_type = "text/event-stream";
            int                     status       = 200;
            std::string             reason       = "OK";
            httplib::Headers        upstream_headers;
            std::string             upstream_base;
            std::string             path;
            std::string             method;
            std::string             body;
            httplib::Headers        headers;
            std::string             request_content_type;
            int                     connection_timeout;
            int                     read_timeout;
        };

        auto state_ptr                = std::make_shared<StreamState>();
        state_ptr->upstream_base      = upstream_base;
        state_ptr->path               = path;
        state_ptr->method             = method;
        state_ptr->body               = request_body;
        state_ptr->headers            = headers;
        state_ptr->request_content_type = content_type;
        state_ptr->connection_timeout = opts.connection_timeout_s;
        state_ptr->read_timeout       = opts.read_timeout_s;

        auto content_receiver = [state_ptr](const char * data, size_t len) {
            {
                std::lock_guard<std::mutex> lock(state_ptr->mutex);
                state_ptr->chunks.emplace(data, len);
            }
            state_ptr->cv.notify_one();
            return true;
        };

        auto upstream_thread = std::make_shared<std::thread>([state_ptr, content_receiver]() {
            auto local_client = std::make_shared<httplib::Client>(state_ptr->upstream_base.c_str());
            local_client->set_connection_timeout(state_ptr->connection_timeout, 0);
            local_client->set_read_timeout(state_ptr->read_timeout, 0);

            const auto & headers     = state_ptr->headers;
            const auto & method      = state_ptr->method;
            const auto & request_body = state_ptr->body;
            const auto & path        = state_ptr->path;
            const auto & content_type = state_ptr->request_content_type;

            httplib::Result result;
            if (method == "POST") {
                result = local_client->Post(path.c_str(), headers, request_body, content_type.c_str(), content_receiver);
                if (result) {
                    std::lock_guard<std::mutex> lock(state_ptr->mutex);
                    state_ptr->status           = result->status;
                    state_ptr->reason           = result->reason;
                    state_ptr->upstream_headers = result->headers;
                    state_ptr->content_type     = result->get_header_value("Content-Type", "text/event-stream");
                }
            } else {
                auto response_handler = [state_ptr](const httplib::Response & upstream) {
                    std::lock_guard<std::mutex> lock(state_ptr->mutex);
                    state_ptr->status           = upstream.status;
                    state_ptr->reason           = upstream.reason;
                    state_ptr->upstream_headers = upstream.headers;
                    state_ptr->content_type     = upstream.get_header_value("Content-Type", "text/event-stream");
                    return true;
                };
                result = local_client->Get(path.c_str(), headers, response_handler, content_receiver);
            }

            std::lock_guard<std::mutex> lock(state_ptr->mutex);
            if (!result) {
                state_ptr->status       = 502;
                state_ptr->reason       = "Bad Gateway";
                state_ptr->content_type = "application/json";
                state_ptr->chunks.emplace("{\"error\":\"upstream unavailable\"}");
            }
            state_ptr->done = true;
            state_ptr->cv.notify_all();
        });

        res.status = 200;
        res.set_chunked_content_provider(
            "text/event-stream",
            [state_ptr, upstream_thread, &res](size_t, httplib::DataSink & sink) {
                std::unique_lock<std::mutex> lock(state_ptr->mutex);
                state_ptr->cv.wait(lock, [&] { return !state_ptr->chunks.empty() || state_ptr->done; });

                if (!state_ptr->chunks.empty()) {

                    // Chunks available: send next chunk to client
                    auto chunk = std::move(state_ptr->chunks.front());
                    state_ptr->chunks.pop();
                    if (!state_ptr->upstream_headers.empty()) {

                        // Apply response headers on first chunk
                        res.status = state_ptr->status;
                        res.reason = state_ptr->reason;
                        copy_response_headers(state_ptr->upstream_headers, res);
                        state_ptr->upstream_headers.clear();
                        res.set_header("Content-Type", state_ptr->content_type);
                    }
                    lock.unlock();

                    // sink.write() returns true -> provider continues immediately
                    return sink.write(chunk.data(), chunk.size());
                }

                // No chunks available: determine if stream should continue or terminate
                if (state_ptr->done) {
                    // Upstream finished and all chunks have been sent
                    lock.unlock();
                    sink.done();  // Explicitly signal stream completion to httplib
                    return false; // Stop provider -> httplib closes connection gracefully
                }

                // Spurious wakeup or transient empty queue: upstream still processing
                lock.unlock();
                return false;     // Pause provider -> httplib retries after timeout/new data
            },
            [state_ptr, upstream_thread](bool) {
                (void) state_ptr;
                if (upstream_thread && upstream_thread->joinable()) {
                    upstream_thread->join();
                }
            });

        return true;
    }

    if (method == "POST") {
        result = client->Post(path.c_str(), headers, request_body, content_type.c_str());
    } else {
        result = client->Get(path.c_str(), headers);
    }

    if (!result) {
        LOG_ERR("Upstream %s unavailable for %s %s\n", upstream_base.c_str(), method.c_str(), path.c_str());
        res.status = 502;
        res.set_content("{\"error\":\"upstream unavailable\"}", "application/json");
        return false;
    }

    res.status = result->status;
    res.reason = result->reason;
    for (const auto & h : result->headers) {
        res.set_header(h.first, h.second);
    }

    const auto ct = result->get_header_value("Content-Type", "application/octet-stream");
    res.set_content(result->body, ct.c_str());
    LOG_INF("Upstream response %d (%s) relayed for %s\n", res.status, res.reason.c_str(), path.c_str());
    return true;
}
