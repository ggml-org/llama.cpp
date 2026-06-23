#include "cli-client.h"

#include "http.h"

#include <algorithm>
#include <chrono>
#include <thread>

// generation can stall for a long time during prompt processing, so the
// read timeout must be generous
static constexpr time_t CLI_HTTP_READ_TIMEOUT_SEC = 3600;

// upper bound for the accumulated response body kept for error reporting
static constexpr size_t CLI_HTTP_MAX_ERROR_BODY = 1024 * 1024;

// returns the path with the base url's path prefix prepended (if any)
static std::string join_path(const common_http_url & parts, const std::string & path) {
    if (parts.path.empty() || parts.path == "/") {
        return path;
    }
    std::string prefix = parts.path;
    if (prefix.back() == '/') {
        prefix.pop_back();
    }
    return prefix + path;
}

json cli_client::get(const std::string & path) {
    auto [cli, parts] = common_http_client(server_base);
    cli.set_read_timeout(CLI_HTTP_READ_TIMEOUT_SEC, 0);
    auto res = cli.Get(join_path(parts, path));
    if (!res) {
        throw std::runtime_error("failed to connect to " + server_base + ": " + httplib::to_string(res.error()));
    }
    if (res->status < 200 || res->status >= 300) {
        throw std::runtime_error("GET " + path + " failed with status " + std::to_string(res->status) + ": " + res->body);
    }
    json result = json::parse(res->body, nullptr, false);
    if (result.is_discarded()) {
        throw std::runtime_error("GET " + path + " returned invalid JSON");
    }
    return result;
}

json cli_client::post(const std::string & path, const json & body) {
    auto [cli, parts] = common_http_client(server_base);
    cli.set_read_timeout(CLI_HTTP_READ_TIMEOUT_SEC, 0);
    auto res = cli.Post(join_path(parts, path), body.dump(), "application/json");
    if (!res) {
        throw std::runtime_error("failed to connect to " + server_base + ": " + httplib::to_string(res.error()));
    }
    if (res->status < 200 || res->status >= 300) {
        throw std::runtime_error("POST " + path + " failed with status " + std::to_string(res->status) + ": " + res->body);
    }
    json result = json::parse(res->body, nullptr, false);
    if (result.is_discarded()) {
        throw std::runtime_error("POST " + path + " returned invalid JSON");
    }
    return result;
}

json cli_client::post_sse(const std::string & path,
                          const json & body,
                          const std::function<bool()> & should_stop,
                          const std::function<void(const json &)> & on_data) {
    auto [cli, parts] = common_http_client(server_base);
    cli.set_read_timeout(CLI_HTTP_READ_TIMEOUT_SEC, 0);

    std::string pending;  // buffer for incomplete SSE lines
    std::string raw_body; // accumulated body, used only for error reporting

    auto receiver = [&](const char * data, size_t len) -> bool {
        if (should_stop()) {
            return false; // aborts the request
        }
        if (raw_body.size() < CLI_HTTP_MAX_ERROR_BODY) {
            raw_body.append(data, std::min(len, CLI_HTTP_MAX_ERROR_BODY - raw_body.size()));
        }
        pending.append(data, len);
        size_t pos;
        while ((pos = pending.find('\n')) != std::string::npos) {
            std::string line = pending.substr(0, pos);
            pending.erase(0, pos + 1);
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            if (line.rfind("data: ", 0) != 0) {
                continue;
            }
            std::string payload = line.substr(6);
            if (payload == "[DONE]") {
                continue;
            }
            json event = json::parse(payload, nullptr, false);
            if (!event.is_discarded()) {
                on_data(event);
            }
        }
        return true;
    };

    httplib::Headers headers = {{"Accept", "text/event-stream"}};
    auto res = cli.Post(join_path(parts, path), headers, body.dump(), "application/json", receiver);

    if (!res) {
        if (res.error() == httplib::Error::Canceled && should_stop()) {
            return json(); // cancelled by the user
        }
        return json {{"error", {{"message", "failed to connect to " + server_base + ": " + httplib::to_string(res.error())}}}};
    }
    if (res->status < 200 || res->status >= 300) {
        json error_body = json::parse(raw_body, nullptr, false);
        if (!error_body.is_discarded() && error_body.contains("error")) {
            return error_body;
        }
        return json {{"error", {{"message", "request failed with status " + std::to_string(res->status)}}}};
    }
    return json();
}

bool cli_client::wait_health(const std::function<bool()> & is_aborted) {
    int connect_attempts = 0;
    while (!is_aborted()) {
        auto [cli, parts] = common_http_client(server_base);
        cli.set_connection_timeout(1, 0);
        auto res = cli.Get(join_path(parts, "/health"));
        if (res) {
            if (res->status == 200) {
                return true;
            }
            // any other status means the server is up but not ready yet
            // (e.g. 503 while the model is still loading)
        } else if (++connect_attempts >= 10) {
            last_error = "failed to connect to " + server_base + ": " + httplib::to_string(res.error());
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
    }
    last_error = "aborted while waiting for the server to become ready";
    return false;
}
