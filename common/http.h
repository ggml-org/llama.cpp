#pragma once

#include <cpp-httplib/httplib.h>

#include <memory>

#ifdef _WIN32
#include <winsock2.h>
#include <windows.h>
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#endif

struct common_http_url {
    std::string scheme;
    std::string user;
    std::string password;
    std::string host;
    int port;
    std::string path;
};

// bracket an IPv6 literal host for a URL authority (RFC 3986)
static std::string common_http_format_host(const std::string & host) {
    return host.find(':') != std::string::npos ? "[" + host + "]" : host;
}

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

    // split the authority into host and optional port, a bracketed IPv6 literal keeps its inner colons (RFC 3986)
    std::string port_str;
    if (!parts.host.empty() && parts.host.front() == '[') {
        auto close = parts.host.find(']');
        if (close == std::string::npos) {
            throw std::runtime_error("invalid IPv6 URL authority: " + parts.host);
        }
        auto after = parts.host.substr(close + 1);
        if (!after.empty() && after.front() == ':') {
            port_str = after.substr(1);
        }
        parts.host = parts.host.substr(1, close - 1);
    } else {
        auto colon_pos = parts.host.find(':');
        if (colon_pos != std::string::npos) {
            port_str = parts.host.substr(colon_pos + 1);
            parts.host = parts.host.substr(0, colon_pos);
        }
    }

    if (!port_str.empty()) {
        parts.port = std::stoi(port_str);
    } else if (parts.scheme == "http") {
        parts.port = 80;
    } else if (parts.scheme == "https") {
        parts.port = 443;
    } else {
        throw std::runtime_error("unsupported URL scheme: " + parts.scheme);
    }

    return parts;
}

class common_http_client {
    httplib::Client cli;
public:
    common_http_client(const std::string & url) : cli(url) {
        cli.set_follow_location(true);
    }
    virtual ~common_http_client() = default;

    virtual httplib::Result Head(const std::string & path) { return cli.Head(path); }
    virtual httplib::Result Get (const std::string & path) { return cli.Get(path); }
    virtual httplib::Result Get (const std::string & path, const httplib::Headers & headers) { return cli.Get(path, headers); }
    virtual httplib::Result Get (const std::string & path, const httplib::Headers & headers, httplib::ContentReceiver receiver, httplib::DownloadProgress progress = nullptr) { return cli.Get(path, headers, std::move(receiver), std::move(progress)); }
    virtual httplib::Result Post(const std::string & path, const std::string & body, const std::string & content_type) { return cli.Post(path, body, content_type); }
    virtual httplib::Result Post(const std::string & path, const httplib::Headers & headers, const std::string & body, const std::string & content_type, httplib::ContentReceiver receiver) { return cli.Post(path, headers, body, content_type, std::move(receiver)); }

    void set_default_headers   (httplib::Headers headers) { cli.set_default_headers(std::move(headers)); }
    void set_basic_auth        (const std::string & username, const std::string & password) { cli.set_basic_auth(username, password); }
    void set_read_timeout      (time_t sec, time_t usec) { cli.set_read_timeout(sec, usec); }
    void set_write_timeout     (time_t sec, time_t usec) { cli.set_write_timeout(sec, usec); }
    void set_connection_timeout(time_t sec, time_t usec) { cli.set_connection_timeout(sec, usec); }

    // the ranged pull path uses the underlying client directly
    httplib::Client & raw() { return cli; }
};

using common_http_client_ptr = std::unique_ptr<common_http_client>;

// create an HTTP client through the substitutable factory below
common_http_client_ptr common_http_client_create(const std::string & url);

// substitute the client creation, e.g. with a stub (ONLY for testing)
void common_http_client_set_factory(common_http_client_ptr (*factory)(const std::string & url));

static std::pair<common_http_client_ptr, common_http_url> common_http_client_init(const std::string & url) {
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

    auto cli = common_http_client_create(parts.scheme + "://" + common_http_format_host(parts.host) + ":" + std::to_string(parts.port));

    if (!parts.user.empty()) {
        cli->set_basic_auth(parts.user, parts.password);
    }

    return { std::move(cli), std::move(parts) };
}

static std::string common_http_show_masked_url(const common_http_url & parts) {
    return parts.scheme + "://" + (parts.user.empty() ? "" : "****:****@") + common_http_format_host(parts.host) + parts.path;
}

static int common_http_get_free_port() {
#ifdef _WIN32
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        return -1;
    }
    typedef SOCKET native_socket_t;
#define INVALID_SOCKET_VAL INVALID_SOCKET
#define CLOSE_SOCKET(s) closesocket(s)
#else
    typedef int native_socket_t;
#define INVALID_SOCKET_VAL -1
#define CLOSE_SOCKET(s) close(s)
#endif

    native_socket_t sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock == INVALID_SOCKET_VAL) {
#ifdef _WIN32
        WSACleanup();
#endif
        return -1;
    }

    struct sockaddr_in serv_addr;
    std::memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    serv_addr.sin_port = htons(0);

    if (bind(sock, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) != 0) {
        CLOSE_SOCKET(sock);
#ifdef _WIN32
        WSACleanup();
#endif
        return -1;
    }

#ifdef _WIN32
    int namelen = sizeof(serv_addr);
#else
    socklen_t namelen = sizeof(serv_addr);
#endif
    if (getsockname(sock, (struct sockaddr*)&serv_addr, &namelen) != 0) {
        CLOSE_SOCKET(sock);
#ifdef _WIN32
        WSACleanup();
#endif
        return -1;
    }

    int port = ntohs(serv_addr.sin_port);

    CLOSE_SOCKET(sock);
#ifdef _WIN32
    WSACleanup();
#endif

    return port;
}
