#pragma once

#include <functional>
#include <string>

// openai-like client for CLI
struct cli_client {
    std::string server_base; // base url, for example "http://127.0.0.1:8080"
    std::string last_error;  // set when wait_health() fails

    std::string model; // optional, set when the server has multiple models (router mode)

    // simple GET request, returns the raw response body
    // throws std::runtime_error on transport error or non-2xx status
    std::string get(const std::string & path);

    // simple POST request, returns the raw response body
    // throws std::runtime_error on transport error or non-2xx status
    std::string post(const std::string & path, const std::string & body);

    // POST request with an SSE streaming response
    // on_data is invoked per "data:" event with the raw event payload
    // returns after the stream is finished (empty string on graceful exit)
    // otherwise, the raw error response body
    std::string post_sse(const std::string & path,
                          const std::string & body,
                          const std::function<bool()> & should_stop,
                          const std::function<void(const std::string &)> & on_data);

    // poll /health until the server is ready to accept requests
    // returns false if is_aborted returned true or the server is unreachable
    bool wait_health(const std::function<bool()> & is_aborted);
};
