#pragma once

#include "log.h"

#include <cpp-httplib/httplib.h>

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

// match a host against a NO_PROXY list: comma separated entries, "*" bypasses the proxy for every host,
// an entry matches the host itself or any of its subdomains, a leading dot is optional, ports are ignored
static bool common_http_proxy_bypass(const std::string & no_proxy, const std::string & host) {
    auto lower = [](std::string s) {
        for (auto & c : s) {
            if (c >= 'A' && c <= 'Z') {
                c += 'a' - 'A';
            }
        }
        return s;
    };

    const std::string h = lower(host);

    for (size_t pos = 0; pos < no_proxy.size(); ) {
        size_t end = no_proxy.find(',', pos);
        if (end == std::string::npos) {
            end = no_proxy.size();
        }

        std::string entry = lower(no_proxy.substr(pos, end - pos));
        pos = end + 1;

        entry.erase(0, entry.find_first_not_of(" \t"));
        auto last = entry.find_last_not_of(" \t");
        if (last == std::string::npos) {
            continue;
        }
        entry.erase(last + 1);

        if (entry == "*") {
            return true;
        }

        if (entry.front() == '.') {
            entry.erase(0, 1);
        }

        if (entry.empty()) {
            continue;
        }

        if (h == entry) {
            return true;
        }

        if (h.size() > entry.size() && h[h.size() - entry.size() - 1] == '.' &&
            h.compare(h.size() - entry.size(), entry.size(), entry) == 0) {
            return true;
        }
    }

    return false;
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

    // Honor HTTP_PROXY / HTTPS_PROXY environment variables (both lower- and upper-case).
    auto getenv_s = [](const char * upper, const char * lower) -> std::string {
        const char * val = std::getenv(upper);
        if (!val) {
            val = std::getenv(lower);
        }
        return val ? val : "";
    };

    const std::string proxy_url = (parts.scheme == "https")
        ? getenv_s("HTTPS_PROXY", "https_proxy")
        : getenv_s("HTTP_PROXY",  "http_proxy");

    if (!proxy_url.empty() && !common_http_proxy_bypass(getenv_s("NO_PROXY", "no_proxy"), parts.host)) {
        try {
            common_http_url proxy = common_http_parse_url(proxy_url);
            cli.set_proxy(proxy.host, proxy.port);
            if (!proxy.user.empty()) {
                cli.set_proxy_basic_auth(proxy.user, proxy.password);
            }
        } catch (const std::exception & e) {
            // fall back to a direct connection, the URL itself is not logged as it can carry credentials
            LOG_WRN("%s: ignoring malformed proxy URL from the environment: %s\n", __func__, e.what());
        }
    }

    return { std::move(cli), std::move(parts) };
}

static std::string common_http_show_masked_url(const common_http_url & parts) {
    return parts.scheme + "://" + (parts.user.empty() ? "" : "****:****@") + parts.host + parts.path;
}
