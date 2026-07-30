// test-server-mtp-trigger.cpp
//
// Verifies the gating of the auto-MTP path in tools/server/server-context.cpp
// (around line 1067). The server auto-enables `COMMON_SPECULATIVE_TYPE_DRAFT_MTP`
// when `common_model_has_embedded_mtp(model_path)` returns true and the user has
// not passed `--no-embedded-mtp`.
//
// The audit (docs/audit-2026-07-29.md, section 2 + 6) flags this auto-trigger as
// a workaround: combined with the stubbed `mtp_context()`/`ane_mtp_program()`
// accessors, it made every gemma4 12B F16 with MTP fail to load until the
// `--no-embedded-mtp` flag was added as a mitigation.
//
// What this test verifies
// -----------------------
// 1. The probe is cheap and read-only: a non-existent path, an empty path, and
//    an unparseable file all return false instead of throwing or crashing.
// 2. A GGUF that explicitly carries `mtp.component.present = true` is
//    detected as an embedded-MTP model.
// 3. The probe is type-strict: a GGUF that has the same key with the wrong
//    type, or with the value false, returns false.
// 4. A GGUF that lacks the key entirely returns false. This is the critical
//    regression case: a non-DFLASH / non-MTP model must not be auto-triggered.
//
// We construct minimal GGUFv3 files in memory using gguf.h, write them to a
// temp file, and hand the path to `common_model_has_embedded_mtp`. The probe
// itself only reads the GGUF header (no tensor allocation), so each subtest
// is sub-millisecond and well within the <5s budget.

#include "gguf.h"

#include "speculative.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>

// assert() is a no-op in Release builds (-DNDEBUG), so the test would
// silently become a no-op. Use TEST_ASSERT for production-grade coverage.
#define TEST_ASSERT(cond)                                                        \
    do {                                                                          \
        if (!(cond)) {                                                            \
            std::fprintf(stderr, "test-server-mtp-trigger: assertion failed: "   \
                                 "%s (at %s:%d)\n",                               \
                         #cond, __FILE__, __LINE__);                              \
            std::abort();                                                         \
        }                                                                         \
    } while (0)

#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

// Write `bytes` to a freshly-created temp file and return its absolute path.
// The caller is expected to unlink() it when done.
std::string write_temp_file(const std::string & suffix, const std::vector<unsigned char> & bytes) {
    const std::string tmpl_str = "/tmp/test-mtp-trigger-XXXXXX" + suffix;
    std::vector<char> path_buf(tmpl_str.size() + 1);
    std::snprintf(path_buf.data(), path_buf.size(), "%s", tmpl_str.c_str());
    int fd = mkstemps(path_buf.data(), (int) suffix.size());
    if (fd < 0) {
        std::perror("mkstemps");
        std::abort();
    }
    const ssize_t written = ::write(fd, bytes.data(), bytes.size());
    if (written < 0 || (size_t) written != bytes.size()) {
        std::perror("write");
        std::abort();
    }
    ::close(fd);
    return std::string(path_buf.data());
}

void unlink_file(const std::string & path) {
    ::unlink(path.c_str());
}

// Build a minimal GGUFv3 file that carries a single `mtp.component.present`
// key with the given boolean value. The file is otherwise empty (no tensors,
// no other KVs) because the probe only inspects the header.
std::vector<unsigned char> build_gguf_with_mtp_flag(bool value) {
    gguf_context * ctx = gguf_init_empty();
    TEST_ASSERT(ctx != nullptr);

    gguf_set_val_bool(ctx, "mtp.component.present", value);

    const size_t meta_size = gguf_get_meta_size(ctx);
    std::vector<unsigned char> out(meta_size);
    gguf_get_meta_data(ctx, out.data());
    gguf_free(ctx);
    return out;
}

// Build a GGUF where the same key is encoded with the wrong type. The probe
// must reject non-BOOL entries (the source explicitly checks
// `gguf_get_kv_type(ctx, key) == GGUF_TYPE_BOOL`). We use GGUF_TYPE_STRING
// for the negative test.
std::vector<unsigned char> build_gguf_with_wrong_type_mtp_flag() {
    gguf_context * ctx = gguf_init_empty();
    TEST_ASSERT(ctx != nullptr);

    gguf_set_val_str(ctx, "mtp.component.present", "true");

    const size_t meta_size = gguf_get_meta_size(ctx);
    std::vector<unsigned char> out(meta_size);
    gguf_get_meta_data(ctx, out.data());
    gguf_free(ctx);
    return out;
}

// Build a GGUF with the unrelated KV `general.architecture = "llama"`. The
// probe must return false because the model has no MTP marker at all.
std::vector<unsigned char> build_gguf_without_mtp_flag() {
    gguf_context * ctx = gguf_init_empty();
    TEST_ASSERT(ctx != nullptr);

    gguf_set_val_str(ctx, "general.architecture", "llama");

    const size_t meta_size = gguf_get_meta_size(ctx);
    std::vector<unsigned char> out(meta_size);
    gguf_get_meta_data(ctx, out.data());
    gguf_free(ctx);
    return out;
}

}  // namespace

int main() {
    // 1. Defensive: empty / non-existent / unparseable paths must return
    //    false. The server-context.cpp gate relies on this never throwing,
    //    since the call site has no try/catch.
    TEST_ASSERT(!common_model_has_embedded_mtp(""));
    TEST_ASSERT(!common_model_has_embedded_mtp("/tmp/this-path-should-not-exist-1234567890.gguf"));

    {
        // A file that exists but is not a GGUF (4-byte header "NOPE").
        std::vector<unsigned char> nope = { 'N', 'O', 'P', 'E' };
        const std::string path = write_temp_file(".gguf", nope);
        TEST_ASSERT(!common_model_has_embedded_mtp(path));
        unlink_file(path);
    }

    // 2. Positive: a GGUF with `mtp.component.present = true` must be
    //    recognized. This is the trigger condition for the auto-MTP path.
    {
        const auto bytes = build_gguf_with_mtp_flag(true);
        const std::string path = write_temp_file(".gguf", bytes);
        TEST_ASSERT(common_model_has_embedded_mtp(path));
        unlink_file(path);
    }

    // 3. Type strictness: a GGUF where the same key is a STRING must NOT
    //    trigger the path. This guards against a future change that
    //    accidentally coerces the type.
    {
        const auto bytes = build_gguf_with_wrong_type_mtp_flag();
        const std::string path = write_temp_file(".gguf", bytes);
        TEST_ASSERT(!common_model_has_embedded_mtp(path));
        unlink_file(path);
    }

    // 4. Negative: a GGUF that explicitly sets the flag to false.
    {
        const auto bytes = build_gguf_with_mtp_flag(false);
        const std::string path = write_temp_file(".gguf", bytes);
        TEST_ASSERT(!common_model_has_embedded_mtp(path));
        unlink_file(path);
    }

    // 5. Critical regression case: a non-MTP model that simply lacks the
    //    key. This is the configuration for every llama.cpp model that
    //    does not use the dflash/draft-mtp drafter; the probe must not
    //    false-positive.
    {
        const auto bytes = build_gguf_without_mtp_flag();
        const std::string path = write_temp_file(".gguf", bytes);
        TEST_ASSERT(!common_model_has_embedded_mtp(path));
        unlink_file(path);
    }

    return 0;
}
