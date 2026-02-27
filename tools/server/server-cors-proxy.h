#pragma once

#include "common.h"
#include "http.h"

#include <string>
#include <unordered_set>
#include <list>
#include <map>

#include "server-http.h"

static server_http_res_ptr proxy_request(const server_http_req & req, std::string method) {
    std::string target_url = req.get_param("url");
    common_http_url parsed_url = common_http_parse_url(target_url);

    if (parsed_url.host.empty()) {
        throw std::runtime_error("invalid target URL: missing host");
    }

    if (parsed_url.path.empty()) {
        parsed_url.path = "/";
    }

    if (!parsed_url.password.empty()) {
        throw std::runtime_error("authentication in target URL is not supported");
    }

    if (parsed_url.scheme != "http" && parsed_url.scheme != "https") {
        throw std::runtime_error("unsupported URL scheme in target URL: " + parsed_url.scheme);
    }

    SRV_INF("proxying %s request to %s://%s%s\n", method.c_str(), parsed_url.scheme.c_str(), parsed_url.host.c_str(), parsed_url.path.c_str());

    auto proxy = std::make_unique<server_http_proxy>(
            method,
            parsed_url.host,
            parsed_url.scheme == "http" ? 80 : 443,
            parsed_url.path,
            req.headers,
            req.body,
            req.should_stop,
            600, // timeout_read (default to 10 minutes)
            600  // timeout_write (default to 10 minutes)
            );

    return proxy;
}

static server_http_context::handler_t proxy_handler_post = [](const server_http_req & req) -> server_http_res_ptr {
    return proxy_request(req, "POST");
};

static server_http_context::handler_t proxy_handler_get = [](const server_http_req & req) -> server_http_res_ptr {
    return proxy_request(req, "GET");
};
