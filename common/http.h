#pragma once

#include <cpp-httplib/httplib.h>

#include <cstdint>
#include <cstring>
#ifdef _WIN32
#  include <ws2tcpip.h>
#else
#  include <arpa/inet.h>
#  include <netdb.h>
#  include <netinet/in.h>
#  include <sys/socket.h>
#endif

struct common_http_url {
    std::string scheme;
    std::string user;
    std::string password;
    std::string host;
    int port;
    std::string path;
};

static common_http_url common_http_parse_url(const std::string & url) {
    common_http_url parts;
    auto scheme_end = url.find("://");

    if (scheme_end == std::string::npos) {
        throw std::runtime_error("invalid URL: no scheme");
    }
    parts.scheme = url.substr(0, scheme_end);

    if (parts.scheme != "http" && parts.scheme != "https") {
        throw std::runtime_error("unsupported URL scheme: " + parts.scheme);
    }

    auto rest = url.substr(scheme_end + 3);
    auto at_pos = rest.find('@');

    if (at_pos != std::string::npos) {
        auto auth = rest.substr(0, at_pos);
        auto colon_pos = auth.find(':');
        if (colon_pos != std::string::npos) {
            parts.user = auth.substr(0, colon_pos);
            parts.password = auth.substr(colon_pos + 1);
        } else {
            parts.user = auth;
        }
        rest = rest.substr(at_pos + 1);
    }

    auto slash_pos = rest.find('/');

    if (slash_pos != std::string::npos) {
        parts.host = rest.substr(0, slash_pos);
        parts.path = rest.substr(slash_pos);
    } else {
        parts.host = rest;
        parts.path = "/";
    }

    auto colon_pos = parts.host.find(':');

    if (colon_pos != std::string::npos) {
        parts.port = std::stoi(parts.host.substr(colon_pos + 1));
        parts.host = parts.host.substr(0, colon_pos);
    } else if (parts.scheme == "http") {
        parts.port = 80;
    } else if (parts.scheme == "https") {
        parts.port = 443;
    } else {
        throw std::runtime_error("unsupported URL scheme: " + parts.scheme);
    }

    return parts;
}

static std::pair<httplib::Client, common_http_url> common_http_client(const std::string & url) {
    common_http_url parts = common_http_parse_url(url);

    if (parts.host.empty()) {
        throw std::runtime_error("error: invalid URL format");
    }

#ifndef CPPHTTPLIB_OPENSSL_SUPPORT
    if (parts.scheme == "https") {
        throw std::runtime_error(
            "HTTPS is not supported. Please rebuild with one of:\n"
            "  -DLLAMA_BUILD_BORINGSSL=ON\n"
            "  -DLLAMA_BUILD_LIBRESSL=ON\n"
            "  -DLLAMA_OPENSSL=ON (default, requires OpenSSL dev files installed)"
        );
    }
#endif

    httplib::Client cli(parts.scheme + "://" + parts.host + ":" + std::to_string(parts.port));

    if (!parts.user.empty()) {
        cli.set_basic_auth(parts.user, parts.password);
    }

    cli.set_follow_location(true);

    return { std::move(cli), std::move(parts) };
}

// Returns true only if every resolved address for `host` is a public,
// routable address. Resolving first normalizes decimal/octal/hex IPv4
// encodings; IPv4-mapped/6to4 IPv6 wrappers are unwrapped and re-checked.
// Fails closed (returns false) on resolution failure.
static bool common_host_is_safe(const std::string & host) {
    auto v4_blocked = [](uint32_t a /*host order*/) {
        auto in = [&](uint32_t net, int bits){ return (a >> (32-bits)) == (net >> (32-bits)); };
        return in(0x00000000u,8) || in(0x0A000000u,8)  || in(0x64400000u,10) ||
               in(0x7F000000u,8) || in(0xA9FE0000u,16) || in(0xAC100000u,12) ||
               in(0xC0A80000u,16)|| in(0xC6120000u,15) || in(0xE0000000u,4)  ||
               in(0xF0000000u,4) || (a == 0xFFFFFFFFu);
    };
    auto v6_blocked = [&](const uint8_t b[16]) {
        static const uint8_t loop[16]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1};
        static const uint8_t any[16]={0};
        static const uint8_t mapped[12]={0,0,0,0,0,0,0,0,0,0,0xFF,0xFF};
        if (!memcmp(b,loop,16) || !memcmp(b,any,16)) return true;
        if ((b[0]&0xFE)==0xFC) return true;                 // fc00::/7
        if (b[0]==0xFE && (b[1]&0xC0)==0x80) return true;   // fe80::/10
        if (b[0]==0xFF) return true;                        // ff00::/8
        if (!memcmp(b,mapped,12))                           // ::ffff:v4
            return v4_blocked((b[12]<<24)|(b[13]<<16)|(b[14]<<8)|b[15]);
        if (b[0]==0x20 && b[1]==0x02)                       // 2002:v4::/16
            return v4_blocked((b[2]<<24)|(b[3]<<16)|(b[4]<<8)|b[5]);
        return false;
    };
    addrinfo hints{}; hints.ai_family = AF_UNSPEC; hints.ai_socktype = SOCK_STREAM;
    addrinfo * res = nullptr;
    if (getaddrinfo(host.c_str(), nullptr, &hints, &res) != 0) return false;
    bool safe = true;
    for (addrinfo * p = res; p && safe; p = p->ai_next) {
        if (p->ai_family == AF_INET)
            safe = !v4_blocked(ntohl(((sockaddr_in*)p->ai_addr)->sin_addr.s_addr));
        else if (p->ai_family == AF_INET6)
            safe = !v6_blocked(((sockaddr_in6*)p->ai_addr)->sin6_addr.s6_addr);
    }
    freeaddrinfo(res);
    return safe;
}

static std::string common_http_show_masked_url(const common_http_url & parts) {
    return parts.scheme + "://" + (parts.user.empty() ? "" : "****:****@") + parts.host + parts.path;
}
