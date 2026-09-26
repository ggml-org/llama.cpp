#pragma once

#include "common.h"
#include "http.h"
#include "server-common.h"
#include "server-http.h"
#include "server-models.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <ws2tcpip.h>
#else
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#endif

struct proxy_target_policy_result {
    bool allowed;
    std::string message;
};

static std::string proxy_target_url_syntax_error(const std::string & target_url) {
    const size_t scheme_end = target_url.find("://");
    if (scheme_end == std::string::npos) {
        return "";
    }

    const size_t authority_begin = scheme_end + 3;
    const size_t authority_end = target_url.find_first_of("/?#", authority_begin);
    if (authority_end != std::string::npos && target_url[authority_end] != '/') {
        return "target URL query or fragment must follow a path";
    }

    const std::string authority = target_url.substr(
            authority_begin,
            authority_end == std::string::npos ? std::string::npos : authority_end - authority_begin);
    if (authority.find('@') != std::string::npos) {
        return "userinfo in target URLs is not allowed";
    }
    if (std::any_of(authority.begin(), authority.end(), [](unsigned char c) {
            return std::isspace(c) != 0 || std::iscntrl(c) != 0;
        })) {
        return "whitespace in target URL authority is not allowed";
    }

    bool has_port = false;
    std::string port;
    if (!authority.empty() && authority.front() == '[') {
        const size_t close = authority.find(']');
        if (close != std::string::npos && close + 1 < authority.size()) {
            if (authority[close + 1] != ':') {
                return "invalid bracketed target URL authority";
            }
            has_port = true;
            port = authority.substr(close + 2);
        }
    } else {
        const size_t colon = authority.find(':');
        if (colon != std::string::npos) {
            if (authority.find(':', colon + 1) != std::string::npos) {
                return "IPv6 target URL literals must be bracketed";
            }
            has_port = true;
            port = authority.substr(colon + 1);
        }
    }

    if (has_port) {
        if (port.empty() || !std::all_of(port.begin(), port.end(), [](unsigned char c) {
                return std::isdigit(c) != 0;
            })) {
            return "invalid target URL port";
        }
        uint32_t port_number = 0;
        for (const unsigned char c : port) {
            port_number = port_number * 10 + static_cast<uint32_t>(c - '0');
            if (port_number > 65535) {
                return "target URL port is out of range";
            }
        }
        if (port_number == 0) {
            return "target URL port is out of range";
        }
    }

    return "";
}

static std::string proxy_normalize_host(std::string host) {
    std::transform(host.begin(), host.end(), host.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (!host.empty() && host.back() == '.') {
        host.pop_back();
    }
    return host;
}

static bool proxy_host_is_or_has_suffix(const std::string & host, const std::string & suffix) {
    return host == suffix ||
           (host.size() > suffix.size() &&
            host.compare(host.size() - suffix.size(), suffix.size(), suffix) == 0 &&
            host[host.size() - suffix.size() - 1] == '.');
}

static bool proxy_ipv4_has_prefix(uint32_t address, uint32_t network, uint32_t mask) {
    return (address & mask) == network;
}

static bool proxy_ipv4_is_global(uint32_t address) {
    // IANA protocol-assignment anycasts are globally reachable exceptions
    // inside the otherwise non-global 192.0.0.0/24 range.
    if (address == 0xc0000009U || address == 0xc000000aU) {
        return true;
    }

    return !proxy_ipv4_has_prefix(address, 0x00000000U, 0xff000000U) &&
           !proxy_ipv4_has_prefix(address, 0x0a000000U, 0xff000000U) &&
           !proxy_ipv4_has_prefix(address, 0x64400000U, 0xffc00000U) &&
           !proxy_ipv4_has_prefix(address, 0x7f000000U, 0xff000000U) &&
           !proxy_ipv4_has_prefix(address, 0xa9fe0000U, 0xffff0000U) &&
           !proxy_ipv4_has_prefix(address, 0xac100000U, 0xfff00000U) &&
           !proxy_ipv4_has_prefix(address, 0xc0000000U, 0xffffff00U) &&
           !proxy_ipv4_has_prefix(address, 0xc0000200U, 0xffffff00U) &&
           !proxy_ipv4_has_prefix(address, 0xc0586300U, 0xffffff00U) &&
           !proxy_ipv4_has_prefix(address, 0xc0a80000U, 0xffff0000U) &&
           !proxy_ipv4_has_prefix(address, 0xc6120000U, 0xfffe0000U) &&
           !proxy_ipv4_has_prefix(address, 0xc6336400U, 0xffffff00U) &&
           !proxy_ipv4_has_prefix(address, 0xcb007100U, 0xffffff00U) &&
           !proxy_ipv4_has_prefix(address, 0xe0000000U, 0xf0000000U) &&
           !proxy_ipv4_has_prefix(address, 0xf0000000U, 0xf0000000U);
}

static bool proxy_ipv6_has_prefix(
        const uint8_t * address,
        const std::array<uint8_t, 16> & prefix,
        size_t prefix_bits) {
    const size_t whole_bytes = prefix_bits / 8;
    const size_t remaining_bits = prefix_bits % 8;
    for (size_t i = 0; i < whole_bytes; ++i) {
        if (address[i] != prefix[i]) {
            return false;
        }
    }
    if (remaining_bits != 0) {
        const uint8_t mask = static_cast<uint8_t>(0xffU << (8 - remaining_bits));
        return (address[whole_bytes] & mask) == (prefix[whole_bytes] & mask);
    }
    return true;
}

static uint32_t proxy_ipv4_from_ipv6_tail(const uint8_t * bytes) {
    return (static_cast<uint32_t>(bytes[12]) << 24) |
           (static_cast<uint32_t>(bytes[13]) << 16) |
           (static_cast<uint32_t>(bytes[14]) << 8) |
           static_cast<uint32_t>(bytes[15]);
}

static bool proxy_ipv6_is_global(const in6_addr & address) {
    static const std::array<uint8_t, 16> IPV4_MAPPED       = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff};
    static const std::array<uint8_t, 16> NAT64_GLOBAL      = {0x00, 0x64, 0xff, 0x9b};
    static const std::array<uint8_t, 16> IANA_SPECIAL      = {0x20, 0x01};
    static const std::array<uint8_t, 16> PCP_ANYCAST       = {0x20, 0x01, 0x00, 0x01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
    static const std::array<uint8_t, 16> TURN_ANYCAST      = {0x20, 0x01, 0x00, 0x01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2};
    static const std::array<uint8_t, 16> DNS_SD_ANYCAST    = {0x20, 0x01, 0x00, 0x01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3};
    static const std::array<uint8_t, 16> AMT               = {0x20, 0x01, 0x00, 0x03};
    static const std::array<uint8_t, 16> AS112_V6          = {0x20, 0x01, 0x00, 0x04, 0x01, 0x12};
    static const std::array<uint8_t, 16> ORCHID_V2         = {0x20, 0x01, 0x00, 0x20};
    static const std::array<uint8_t, 16> DRONE_REMOTE_ID   = {0x20, 0x01, 0x00, 0x30};
    static const std::array<uint8_t, 16> DOCUMENTATION_V6  = {0x20, 0x01, 0x0d, 0xb8};
    static const std::array<uint8_t, 16> SIX_TO_FOUR       = {0x20, 0x02};
    static const std::array<uint8_t, 16> DOCUMENTATION_V2  = {0x3f, 0xff, 0x00};
    static const std::array<uint8_t, 16> SEGMENT_ROUTING   = {0x5f, 0x00};

    const uint8_t * bytes = reinterpret_cast<const uint8_t *>(&address);

    if (proxy_ipv6_has_prefix(bytes, IPV4_MAPPED, 96)) {
        return proxy_ipv4_is_global(proxy_ipv4_from_ipv6_tail(bytes));
    }
    if (proxy_ipv6_has_prefix(bytes, NAT64_GLOBAL, 96)) {
        return proxy_ipv4_is_global(proxy_ipv4_from_ipv6_tail(bytes));
    }

    // Globally routable unicast space is 2000::/3. Other address families
    // include unspecified, loopback, ULA, link-local, site-local, and multicast.
    if ((bytes[0] & 0xe0U) != 0x20U) {
        return false;
    }

    // The IANA 2001::/23 registry is non-global by default. Only allocations
    // explicitly marked globally reachable are allowed.
    if (proxy_ipv6_has_prefix(bytes, IANA_SPECIAL, 23)) {
        return proxy_ipv6_has_prefix(bytes, PCP_ANYCAST, 128) ||
               proxy_ipv6_has_prefix(bytes, TURN_ANYCAST, 128) ||
               proxy_ipv6_has_prefix(bytes, DNS_SD_ANYCAST, 128) ||
               proxy_ipv6_has_prefix(bytes, AMT, 32) ||
               proxy_ipv6_has_prefix(bytes, AS112_V6, 48) ||
               proxy_ipv6_has_prefix(bytes, ORCHID_V2, 28) ||
               proxy_ipv6_has_prefix(bytes, DRONE_REMOTE_ID, 28);
    }

    return !proxy_ipv6_has_prefix(bytes, DOCUMENTATION_V6, 32) &&
           !proxy_ipv6_has_prefix(bytes, SIX_TO_FOUR, 16) &&
           !proxy_ipv6_has_prefix(bytes, DOCUMENTATION_V2, 20) &&
           !proxy_ipv6_has_prefix(bytes, SEGMENT_ROUTING, 16);
}

static bool proxy_sockaddr_is_global(const sockaddr * address) {
    if (address->sa_family == AF_INET) {
        const auto * ipv4 = reinterpret_cast<const sockaddr_in *>(address);
        return proxy_ipv4_is_global(ntohl(ipv4->sin_addr.s_addr));
    }
    if (address->sa_family == AF_INET6) {
        const auto * ipv6 = reinterpret_cast<const sockaddr_in6 *>(address);
        return proxy_ipv6_is_global(ipv6->sin6_addr);
    }
    return false;
}

// Residual risk, documented and not fixed here: this checks the target host string once, at
// request time. A hostname target (not a numeric IP) is not resolved here (see the AI_NUMERICHOST
// probe and its "Tier 4a" comment below), and the outbound request is made later by
// server_http_proxy (server-models.cpp), which does its own DNS lookup and follows HTTP redirects
// (cpp-httplib set_follow_location(true)) without calling back into this policy. So a hostname
// that resolves to a non-global address at connection time (DNS rebinding) or a redirect to a
// non-global target is not caught. Closing this needs the HTTP client to connect to the specific
// IP validated here (and to re-validate each redirect hop), which is a client-level change outside
// this policy function's scope.
static proxy_target_policy_result proxy_target_policy(
        const common_http_url & parsed_url,
        const std::vector<std::string> & allowlist) {
    if (parsed_url.scheme != "http" && parsed_url.scheme != "https") {
        return {false, "unsupported target URL scheme"};
    }
    if (parsed_url.host.empty()) {
        return {false, "target URL is missing a host"};
    }
    if (!parsed_url.user.empty() || !parsed_url.password.empty()) {
        return {false, "userinfo in target URLs is not allowed"};
    }

    const std::string host = proxy_normalize_host(parsed_url.host);
    if (host.empty()) {
        return {false, "target URL is missing a host"};
    }

    for (const std::string & allowed_host : allowlist) {
        if (host == proxy_normalize_host(allowed_host)) {
            return {true, ""};
        }
    }

    if (proxy_host_is_or_has_suffix(host, "localhost") ||
        proxy_host_is_or_has_suffix(host, "local") ||
        proxy_host_is_or_has_suffix(host, "localdomain") ||
        proxy_host_is_or_has_suffix(host, "internal") ||
        proxy_host_is_or_has_suffix(host, "home.arpa")) {
        return {false, "local target host is not allowed"};
    }
    if (host == "metadata.google.internal" || host == "metadata.goog") {
        return {false, "metadata target host is not allowed"};
    }

    addrinfo hints = {};
    hints.ai_family = AF_UNSPEC;
    hints.ai_flags = AI_NUMERICHOST;
    addrinfo * addresses = nullptr;
    const int lookup_result = getaddrinfo(host.c_str(), nullptr, &hints, &addresses);
    if (lookup_result != 0) {
        return {true, ""}; // Hostname: Tier 4a intentionally does not resolve DNS.
    }

    bool global = addresses != nullptr;
    for (const addrinfo * current = addresses; current != nullptr; current = current->ai_next) {
        if (current->ai_addr == nullptr || !proxy_sockaddr_is_global(current->ai_addr)) {
            global = false;
            break;
        }
    }
    freeaddrinfo(addresses);

    return global
            ? proxy_target_policy_result{true, ""}
            : proxy_target_policy_result{false, "non-global numeric target host is not allowed"};
}

static std::string proxy_header_to_lower(std::string header) {
    std::transform(header.begin(), header.end(), header.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return header;
}

static bool proxy_header_is_forbidden(const std::string & header) {
    static constexpr std::array<const char *, 11> forbidden_headers = {
        "connection",
        "content-length",
        "host",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    };
    const std::string lowered = proxy_header_to_lower(header);
    return std::find(forbidden_headers.begin(), forbidden_headers.end(), lowered) != forbidden_headers.end();
}

static std::map<std::string, std::string> proxy_extract_forward_headers(
        const std::map<std::string, std::string> & request_headers) {
    const std::string proxy_header_prefix = "x-llama-server-proxy-header-";
    std::set<std::string> connection_headers;
    for (const auto & entry : request_headers) {
        const std::string lowered_key = proxy_header_to_lower(entry.first);
        if (!string_starts_with(lowered_key, proxy_header_prefix)) {
            continue;
        }

        const std::string forwarded_key = lowered_key.substr(proxy_header_prefix.size());
        if (forwarded_key == "connection") {
            for (const auto & token : string_split<std::string>(entry.second, ',')) {
                const std::string lowered_token = proxy_header_to_lower(string_strip(token));
                if (!lowered_token.empty()) {
                    connection_headers.insert(lowered_token);
                }
            }
        }
    }

    std::map<std::string, std::string> headers;
    for (const auto & entry : request_headers) {
        const std::string lowered_key = proxy_header_to_lower(entry.first);
        if (!string_starts_with(lowered_key, proxy_header_prefix)) {
            continue;
        }

        const std::string new_key = entry.first.substr(proxy_header_prefix.size());
        const std::string lowered_new_key = lowered_key.substr(proxy_header_prefix.size());
        if (new_key.empty() ||
            proxy_header_is_forbidden(lowered_new_key) ||
            connection_headers.count(lowered_new_key) != 0) {
            continue;
        }

        headers[new_key] = entry.second;
    }
    return headers;
}

static server_http_res_ptr proxy_policy_error(const std::string & message) {
    auto res = std::make_unique<server_http_res>();
    res->status = 400;
    res->data = safe_json_to_str({
        {"error", format_error_response(message, ERROR_TYPE_INVALID_REQUEST)},
    });
    return res;
}

static server_http_res_ptr proxy_request(
        const server_http_req & req,
        const std::string & method,
        const std::vector<std::string> & allowlist) {
    const std::string target_url = req.get_param("url");
    const std::string syntax_error = proxy_target_url_syntax_error(target_url);
    if (!syntax_error.empty()) {
        return proxy_policy_error(syntax_error);
    }

    common_http_url parsed_url;
    try {
        parsed_url = common_http_parse_url(target_url);
    } catch (const std::exception & e) {
        return proxy_policy_error(std::string("invalid target URL: ") + e.what());
    }

    const proxy_target_policy_result policy = proxy_target_policy(parsed_url, allowlist);
    if (!policy.allowed) {
        return proxy_policy_error(policy.message);
    }

    if (parsed_url.path.empty()) {
        parsed_url.path = "/";
    }

    SRV_INF("proxying %s request to %s://%s:%i%s\n", method.c_str(), parsed_url.scheme.c_str(), common_http_format_host(parsed_url.host).c_str(), parsed_url.port, parsed_url.path.c_str());

    const std::map<std::string, std::string> headers = proxy_extract_forward_headers(req.headers);

    return std::make_unique<server_http_proxy>(
            method,
            parsed_url.scheme,
            parsed_url.host,
            parsed_url.port,
            parsed_url.path,
            headers,
            req.body,
            req.files,
            req.should_stop,
            600,
            600);
}

static server_http_context::handler_t proxy_handler_post(std::vector<std::string> allowlist) {
    return [allowlist = std::move(allowlist)](const server_http_req & req) -> server_http_res_ptr {
        return proxy_request(req, "POST", allowlist);
    };
}

static server_http_context::handler_t proxy_handler_get(std::vector<std::string> allowlist) {
    return [allowlist = std::move(allowlist)](const server_http_req & req) -> server_http_res_ptr {
        return proxy_request(req, "GET", allowlist);
    };
}
