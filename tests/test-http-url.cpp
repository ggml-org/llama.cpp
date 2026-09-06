// tests common_http_parse_url, which splits a URL into the parts the http client is
// built from. The authority boundary is the interesting part: getting it wrong folds a
// query string into the hostname, which then fails to resolve.

#include "http.h"

#include <cstdio>
#include <string>
#include <vector>

static int g_failures = 0;

static void check(const std::string & url,
                  const std::string & user,
                  const std::string & password,
                  const std::string & host,
                  int                 port,
                  const std::string & path) {
    common_http_url parts;

    try {
        parts = common_http_parse_url(url);
    } catch (const std::exception & e) {
        printf("FAIL %-46s threw: %s\n", url.c_str(), e.what());
        g_failures++;
        return;
    }

    if (parts.user == user && parts.password == password && parts.host == host &&
        parts.port == port && parts.path == path) {
        return;
    }

    printf("FAIL %s\n", url.c_str());
    printf("     expected user=[%s] password=[%s] host=[%s] port=%d path=[%s]\n",
           user.c_str(), password.c_str(), host.c_str(), port, path.c_str());
    printf("     got      user=[%s] password=[%s] host=[%s] port=%d path=[%s]\n",
           parts.user.c_str(), parts.password.c_str(), parts.host.c_str(), parts.port,
           parts.path.c_str());
    g_failures++;
}

static void check_throws(const std::string & url) {
    try {
        common_http_parse_url(url);
    } catch (const std::exception &) {
        return;
    }
    printf("FAIL %s: expected a throw\n", url.c_str());
    g_failures++;
}

int main() {
    // scheme, host and path
    check("http://example.com",                 "", "", "example.com",  80, "/");
    check("https://example.com",                "", "", "example.com", 443, "/");
    check("https://example.com/",               "", "", "example.com", 443, "/");
    check("https://example.com/model.gguf",     "", "", "example.com", 443, "/model.gguf");
    check("https://example.com:8443/a/b",       "", "", "example.com", 8443, "/a/b");

    // a query or fragment ends the authority just as a '/' does, and the path is still
    // rooted, so neither is absorbed into the host
    check("https://example.com?download=1",     "", "", "example.com", 443, "/?download=1");
    check("https://example.com#frag",           "", "", "example.com", 443, "/#frag");
    check("http://example.com:8080?a=1",        "", "", "example.com", 8080, "/?a=1");
    check("https://example.com/m.gguf?a=1",     "", "", "example.com", 443, "/m.gguf?a=1");

    // userinfo
    check("https://user:pass@example.com/x",    "user", "pass", "example.com", 443, "/x");
    check("https://user@example.com/x",         "user", "",     "example.com", 443, "/x");
    check("https://user@example.com?a=1",       "user", "",     "example.com", 443, "/?a=1");

    // '@' past the end of the authority belongs to the path or query, not to userinfo
    check("https://example.com/p?to=a@b.com",   "", "", "example.com", 443, "/p?to=a@b.com");
    check("https://example.com/a@b",            "", "", "example.com", 443, "/a@b");
    check("https://example.com?to=a@b.com",     "", "", "example.com", 443, "/?to=a@b.com");

    // IPv6 literals keep their inner colons and lose their brackets
    check("http://[::1]",                       "", "", "::1", 80, "/");
    check("http://[::1]:8080/x",                "", "", "::1", 8080, "/x");
    check("http://[::1]?a=1",                   "", "", "::1", 80, "/?a=1");
    check("http://u:p@[::1]:8080/x",            "u", "p", "::1", 8080, "/x");

    check_throws("example.com/model.gguf");
    check_throws("ftp://example.com/model.gguf");

    if (g_failures > 0) {
        printf("%d failure(s)\n", g_failures);
        return 1;
    }

    printf("OK\n");
    return 0;
}
