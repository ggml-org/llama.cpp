#pragma once

#include "ggml.h"

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

#include <functional>
#include <string>

using json = nlohmann::ordered_json;

// openai-like client for CLI
struct cli_client {
    std::string server_base; // base url, for example "http://127.0.0.1:8080"
    std::string last_error;  // set when wait_health() fails

    // simple GET request, returns the response json
    // throws std::runtime_error on transport error or non-2xx status
    json get(const std::string & path);

    // simple POST request, returns the response json
    // throws std::runtime_error on transport error or non-2xx status
    json post(const std::string & path, const json & body);

    // POST request with an SSE streaming response; on_data is invoked once
    // per "data:" event; the function returns after the stream is finished:
    // a null json on graceful exit (incl. cancellation via should_stop),
    // the error response json otherwise
    json post_sse(const std::string & path,
                  const json & body,
                  const std::function<bool()> & should_stop,
                  const std::function<void(const json &)> & on_data);

    // poll /health until the server is ready to accept requests
    // returns false if is_aborted returned true or the server is unreachable
    bool wait_health(const std::function<bool()> & is_aborted);

    //
    // higher-level wrappers
    //

    json create_chat_completion(const json & request,
                                const std::function<bool()> & should_stop,
                                const std::function<void(const json &)> & on_data) {
        return post_sse("/v1/chat/completions", request, should_stop, on_data);
    }

    json get_props() {
        return get("/props");
    }
};
