#include "server-cors-proxy.h"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <winsock2.h>
#endif

static void expect_policy(
        const std::string & url,
        const std::vector<std::string> & allowlist,
        bool expected_allowed) {
    const common_http_url parsed = common_http_parse_url(url);
    const proxy_target_policy_result result = proxy_target_policy(parsed, allowlist);
    if (result.allowed != expected_allowed) {
        std::cerr << "unexpected policy result for " << url << ": " << result.message << std::endl;
        std::exit(1);
    }
}

int main() {
#ifdef _WIN32
    WSADATA wsa_data;
    if (WSAStartup(MAKEWORD(2, 2), &wsa_data) != 0) {
        std::cerr << "WSAStartup failed" << std::endl;
        return 1;
    }
#endif

    for (const char * url : {
            "http://@example.com/",
            "http://example.com:/",
            "http://example.com:0/",
            "http://example.com:65536/",
            "http://example.com:not-a-port/",
            "http://[::1]suffix/",
            "http://example.com?query",
        }) {
        if (proxy_target_url_syntax_error(url).empty()) {
            std::cerr << "expected malformed target URL to be rejected: " << url << std::endl;
            return 1;
        }
    }

    for (const char * url : {
            "http://8.8.8.8/",
            "https://[2606:4700:4700::1111]/",
            "http://192.0.0.9/",
            "http://192.0.0.10/",
            "https://[2001:1::1]/",
            "https://[2001:1::2]/",
            "https://[2001:1::3]/",
            "https://[2001:3::1]/",
            "https://[2001:4:112::1]/",
            "https://[2001:20::1]/",
            "https://[2001:30::1]/",
            "https://[64:ff9b::808:808]/",
            "https://mcp.example.com/",
        }) {
        expect_policy(url, {}, true);
    }

    for (const char * url : {
            "http://0.0.0.0/",
            "http://127.0.0.1/",
            "http://10.0.0.1/",
            "http://100.64.0.1/",
            "http://169.254.169.254/",
            "http://172.16.0.1/",
            "http://192.168.0.1/",
            "http://192.0.2.1/",
            "http://198.18.0.1/",
            "http://198.51.100.1/",
            "http://203.0.113.1/",
            "http://224.0.0.1/",
            "http://240.0.0.1/",
            "http://255.255.255.255/",
            "http://127.1/",
            "http://2130706433/",
            "http://0177.0.0.1/",
            "http://0x7f000001/",
            "http://[::]/",
            "http://[::1]/",
            "http://[fc00::1]/",
            "http://[fd00::1]/",
            "http://[fe80::1]/",
            "http://[ff02::1]/",
            "http://[100::1]/",
            "http://[2001:db8::1]/",
            "http://[2001:1::4]/",
            "http://[2001:5::1]/",
            "http://[2002::1]/",
            "http://[3fff::1]/",
            "http://[5f00::1]/",
            "http://[64:ff9b:1::1]/",
            "http://[64:ff9b::a00:1]/",
            "http://[::ffff:127.0.0.1]/",
            "http://[::ffff:10.0.0.1]/",
            "http://localhost/",
            "http://api.localhost/",
            "http://localhost./",
            "http://service.local/",
            "http://localdomain/",
            "http://service.localdomain/",
            "http://internal/",
            "http://service.internal/",
            "http://metadata.google.internal/",
            "http://metadata.google.internal./",
            "http://metadata.goog/",
            "http://metadata.goog./",
        }) {
        expect_policy(url, {}, false);
    }

    expect_policy("http://user@example.com/", {"example.com"}, false);
    expect_policy("http://user:password@example.com/", {"example.com"}, false);

    expect_policy("http://127.0.0.1/", {"127.0.0.1"}, true);
    expect_policy("http://LOCALHOST./", {"localhost."}, true);
    expect_policy("http://127.0.0.1/", {"127.0.0.10"}, false);
    expect_policy("http://localhost/", {"localhost.example"}, false);

    for (const char * header : {
            "Host",
            "Content-Length",
            "Transfer-Encoding",
            "Connection",
            "Proxy-Connection",
            "Keep-Alive",
            "Proxy-Authenticate",
            "Proxy-Authorization",
            "TE",
            "Trailer",
            "Upgrade",
        }) {
        if (!proxy_header_is_forbidden(header)) {
            std::cerr << "expected proxy header to be blocked: " << header << std::endl;
            return 1;
        }
    }
    for (const char * header : {"Authorization", "Content-Type", "X-MCP-Token"}) {
        if (proxy_header_is_forbidden(header)) {
            std::cerr << "expected proxy header to be forwarded: " << header << std::endl;
            return 1;
        }
    }

    const std::map<std::string, std::string> request_headers = {
        {"X-Llama-Server-Proxy-Header-Authorization", "Bearer allowed"},
        {"X-Llama-Server-Proxy-Header-Connection", "X-Drop-Me, X-Also-Drop"},
        {"X-Llama-Server-Proxy-Header-Content-Type", "application/json"},
        {"X-Llama-Server-Proxy-Header-Host", "internal.example"},
        {"X-Llama-Server-Proxy-Header-X-Also-Drop", "blocked"},
        {"X-Llama-Server-Proxy-Header-X-Drop-Me", "blocked"},
    };
    const auto forwarded_headers = proxy_extract_forward_headers(request_headers);
    if (forwarded_headers.size() != 2 ||
        forwarded_headers.at("Authorization") != "Bearer allowed" ||
        forwarded_headers.at("Content-Type") != "application/json") {
        std::cerr << "proxy header filtering did not preserve only end-to-end headers" << std::endl;
        return 1;
    }

#ifdef _WIN32
    WSACleanup();
#endif
    return 0;
}
