#pragma once

#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include <cstdint>
#include <unordered_map>

struct common_params;
struct stream_pipe_producer; // defined in server-stream.h

// generator-like API for HTTP response generation
// this object response with one of the 2 modes:
// 1) normal response: `data` contains the full response body
// 2) streaming response: each call to next(output) generates the next chunk
//    when next(output) returns false, no more data after the current chunk
//    note: some chunks can be empty, in which case no data is sent for that chunk
struct server_http_res {
    std::string content_type = "application/json; charset=utf-8";
    int status = 200;
    std::string data;
    std::map<std::string, std::string> headers;

    // if set, the stream survives a client disconnect: the producer pipe keeps draining into the
    // ring buffer and finalizes the session on destruction, so no explicit on_stream_end is needed.
    // shared_ptr (not unique_ptr) so the forward-declared type is safe to delete here.
    std::shared_ptr<stream_pipe_producer> spipe;

    std::function<bool(std::string &)> next = nullptr;
    bool is_stream() const {
        return next != nullptr;
    }

    // called when the session is cancelled (e.g. DELETE /v1/stream/<conv_id>).
    // server_res_generator overrides this to stop its reader; the default is a no-op.
    virtual void stop() {}

    virtual ~server_http_res() = default;
};

// unique pointer, used by set_chunked_content_provider
// httplib requires the stream provider to be stored in heap
using server_http_res_ptr = std::unique_ptr<server_http_res>;
using raw_buffer = std::vector<uint8_t>;

struct uploaded_file {
    raw_buffer data;
    std::string filename;
    std::string content_type;
};

struct server_http_req {
    std::map<std::string, std::string> params; // path_params + query_params
    std::map<std::string, std::string> headers; // used by MCP proxy
    std::string path;
    std::string query_string; // query parameters string (e.g. "action=save")
    std::string body;
    std::map<std::string, uploaded_file> files; // used for file uploads (form data)
    const std::function<bool()> & should_stop;

    std::string get_param(const std::string & key, const std::string & def = "") const {
        auto it = params.find(key);
        if (it != params.end()) {
            return it->second;
        }
        return def;
    }
};

struct server_http_context {
    class Impl;
    std::unique_ptr<Impl> pimpl;

    std::thread thread; // server thread
    std::atomic<bool> is_ready = false;

    // note: the handler should never throw exceptions
    using handler_t = std::function<server_http_res_ptr(const server_http_req & req)>;
    mutable std::unordered_map<std::string, handler_t> handlers;

    std::string path_prefix;
    std::string hostname;
    int port    = 8080;
    bool is_ssl = false;

    server_http_context();
    ~server_http_context();

    bool init(const common_params & params);
    bool start();
    void stop() const;

    void get(const std::string & path, const handler_t & handler) const;
    void post(const std::string & path, const handler_t & handler) const;
    void del(const std::string & path, const handler_t & handler) const;

    // Register the Google Cloud Platform (Vertex AI) compat (AIP_PREDICT_ROUTE env var, or /predict)
    // Must be called AFTER all other API routes are registered
    void register_gcp_compat() const;

    // for debugging
    std::string listening_address;
};
