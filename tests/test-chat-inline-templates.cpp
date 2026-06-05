// Tests for the built-in ("inline") chat template fallback.
//
// Some models ship GGUF files without an embedded chat_template. For those,
// common/chat.cpp falls back to a built-in Jinja template selected by the model
// architecture name. The templates are embedded at build time by
// scripts/gen-chat-inline-templates.py from the manifest
// common/chat-inline-templates.h.in.
//
// This test runs with the repository root as the working directory (see
// tests/CMakeLists.txt) so it can read the original .jinja files from disk and
// confirm they were embedded verbatim.

#include "chat.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

static std::string read_file(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        fprintf(stderr, "failed to open '%s' (is the working directory the repo root?)\n", path.c_str());
        assert(false && "could not open template file");
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

int main() {
    // A registered architecture returns its embedded template, byte-for-byte
    // identical to the source .jinja file.
    {
        const std::string expected = read_file("models/templates/deepseek-ai-DeepSeek-V4.jinja");
        const std::string actual   = common_chat_template_inline("deepseek-v4-flash");

        assert(!actual.empty() && "deepseek-v4-flash should have a built-in inline template");
        assert(actual == expected && "inline template must match the source .jinja byte-for-byte");
        printf("ok: deepseek-v4-flash inline template matches source (%zu bytes)\n", actual.size());
    }

    // An unregistered architecture returns an empty string (no fallback).
    {
        const std::string actual = common_chat_template_inline("this-arch-does-not-exist");
        assert(actual.empty() && "unregistered arch must not return a template");
        printf("ok: unregistered arch returns empty\n");
    }

    // An empty architecture name must not match anything either.
    {
        const std::string actual = common_chat_template_inline("");
        assert(actual.empty() && "empty arch must not return a template");
        printf("ok: empty arch returns empty\n");
    }

    printf("All inline chat template tests passed.\n");
    return 0;
}
