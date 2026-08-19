#include "server-task.h"

#include <cstdio>
#include <initializer_list>

static server_tokens tokens(std::initializer_list<llama_token> values, bool has_mtmd = false) {
    return server_tokens(llama_tokens(values), has_mtmd);
}

static bool check(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        return false;
    }
    return true;
}

int main() {
    bool ok = true;

    {
        auto resident = tokens({ 1, 2, 3 });
        auto incoming = tokens({ 1, 2, 3, 4, 5 });
        ok &= check(server_prompt_cache_can_reuse_in_place(resident, incoming, true),
                    "an exact text prefix should reuse the resident state");
    }
    {
        auto resident = tokens({ 1, 2, 3 });
        auto incoming = tokens({ 1, 2, 3 });
        ok &= check(server_prompt_cache_can_reuse_in_place(resident, incoming, true),
                    "an identical text prompt should reuse the resident state");
    }
    {
        auto resident = tokens({});
        auto incoming = tokens({ 1 });
        ok &= check(!server_prompt_cache_can_reuse_in_place(resident, incoming, true),
                    "an empty resident prompt must use the established path");
    }
    {
        auto resident = tokens({ 1, 2, 3 });
        auto incoming = tokens({ 1, 2 });
        ok &= check(!server_prompt_cache_can_reuse_in_place(resident, incoming, true),
                    "a rewind must use the established path");
    }
    {
        auto resident = tokens({ 1, 2, 3 });
        auto incoming = tokens({ 1, 9, 3, 4 });
        ok &= check(!server_prompt_cache_can_reuse_in_place(resident, incoming, true),
                    "a divergent prompt must use the established path");
    }
    {
        auto resident = tokens({ 1, 2, 3 });
        auto incoming = tokens({ 1, 2, 3, 4 });
        ok &= check(!server_prompt_cache_can_reuse_in_place(resident, incoming, false),
                    "cache_prompt=false must use the established path");
    }
    {
        auto resident = tokens({ 1, 2, 3 }, true);
        auto incoming = tokens({ 1, 2, 3, 4 });
        ok &= check(!server_prompt_cache_can_reuse_in_place(resident, incoming, true),
                    "a multimedia resident prompt must use the established path");
    }
    {
        auto resident = tokens({ 1, 2, 3 });
        auto incoming = tokens({ 1, 2, 3, 4 }, true);
        ok &= check(!server_prompt_cache_can_reuse_in_place(resident, incoming, true),
                    "a multimedia incoming prompt must use the established path");
    }

    if (!ok) {
        return 1;
    }

    std::puts("prompt cache in-place policy: PASS");
    return 0;
}
