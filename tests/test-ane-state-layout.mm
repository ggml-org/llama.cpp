// Unit tests for the shared ane_state_layout.v1 reader
// (common/ane-state-layout.h). The reader is shared between
// ggml/src/ggml-ane/ggml-ane.mm (the W0/W1 ggml backend) and
// common/ane-mtp.mm (the multifunction prefill/MTP/DFlash backend).
// These tests are the C++ side of the contract; the Python side
// lives in tools/ane-mtp/test_state_layout.py and has its own
// 24 tests covering the same fields from the other direction.
//
// What we test:
//   - Reading a valid W0 manifest (real fixture, no mocking).
//   - Rejecting a missing file with a clear error string.
//   - Rejecting an empty file.
//   - Rejecting a non-JSON file.
//   - Rejecting a version mismatch (we write a JSON with
//     version=2 and confirm the reader refuses).
//   - Field-level spot-checks: bundle_name, state_size_bytes,
//     slot count, slot names, function name, role, model_type.

#include "ane-state-layout.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

namespace {

int g_failures = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL [%s:%d] %s\n", __FILE__, __LINE__, msg); \
        ++g_failures; \
    } else { \
        std::fprintf(stdout, "ok   %s\n", msg); \
    } \
} while (0)

std::string resolve_w0_manifest_path() {
    if (const char * env = std::getenv("TESSERA_ANE_STATE_LAYOUT_MANIFEST");
            env != nullptr && env[0] != '\0') {
        return env;
    }
    std::string candidate = "./tools/ane-mtp/fixtures/w0-matmul/w0-256x256.ane_state.v1.json";
    for (int i = 0; i < 8; ++i) {
        struct stat st;
        if (::stat(candidate.c_str(), &st) == 0) {
            return candidate;
        }
        auto slash = candidate.find("/tools/");
        if (slash == std::string::npos) break;
        candidate = ".." + candidate.substr(slash);
    }
    return {};
}

bool write_temp(const std::string & contents, std::string * path_out) {
    char tmpl[] = "/tmp/ane_state_test.XXXXXX";
    int fd = ::mkstemp(tmpl);
    if (fd < 0) return false;
    ::write(fd, contents.data(), contents.size());
    ::close(fd);
    *path_out = tmpl;
    return true;
}

}  // namespace

int main() {
    std::fprintf(stdout, "ane state-layout reader test\n");

    // --- Test 1: read the manifest at the path resolved by the
    //     env var or the W0 fallback. We branch on which manifest
    //     is being tested so the W0-specific assertions don't
    //     fire when an external manifest (e.g. a real multifunction
    //     bundle like gemma4-ane-prefill-bundle) is provided.
    const std::string manifest_path = resolve_w0_manifest_path();
    CHECK(!manifest_path.empty(), "manifest path resolves");
    if (manifest_path.empty()) return 1;
    const char * env = std::getenv("TESSERA_ANE_STATE_LAYOUT_MANIFEST");
    const bool is_w0 = (env == nullptr) ||
        std::strstr(manifest_path.c_str(), "w0-256x256") != nullptr;

    {
        ane_state_layout_v1_t layout;
        std::string error;
        const bool ok = ane_layout::read_state_layout(
            manifest_path.c_str(), &layout, &error);
        CHECK(ok, "read_state_layout accepts the manifest");
        if (ok) {
            CHECK(layout.version == 1, "version == 1");
            CHECK(layout.state_size_bytes >= 65536,
                  "state_size_bytes >= 64KB (ANE minimum)");
            CHECK(layout.model_type == ANE_MODEL_TYPE_NEURAL_NETWORK ||
                  layout.model_type == ANE_MODEL_TYPE_ML_PROGRAM,
                  "model_type is NeuralNetwork or ML Program");
            CHECK(layout.n_slots > 0, "at least one slot declared");
            CHECK(layout.n_functions > 0,
                  "at least one function declared");
            // All slot offsets are 16-byte aligned (SIMD safety).
            for (uint32_t i = 0; i < layout.n_slots; ++i) {
                CHECK(layout.slots[i].offset % 16 == 0,
                      "slot offset is 16-byte aligned");
                CHECK(layout.slots[i].size_bytes % 16 == 0,
                      "slot size is 16-byte aligned");
            }
            if (is_w0) {
                CHECK(std::strcmp(layout.bundle_name, "w0-256x256") == 0,
                      "W0 bundle_name == w0-256x256");
                CHECK(layout.state_size_bytes == 65536,
                      "W0 state_size_bytes == 65536 (ANE minimum alloc)");
                CHECK(layout.model_type == ANE_MODEL_TYPE_NEURAL_NETWORK,
                      "W0 model_type == NEURAL_NETWORK");
                CHECK(layout.n_slots == 2, "W0 has 2 slots");
                if (layout.n_slots == 2) {
                    CHECK(std::strcmp(layout.slots[0].name, "x") == 0,
                          "W0 slots[0].name == x");
                    CHECK(layout.slots[0].kind == ANE_SLOT_KIND_INPUT,
                          "W0 slots[0].kind == INPUT");
                    CHECK(layout.slots[0].dtype == ANE_DTYPE_F32,
                          "W0 slots[0].dtype == F32");
                    CHECK(layout.slots[0].shape[0] == 256,
                          "W0 slots[0].shape[0] == 256");
                    CHECK(std::strcmp(layout.slots[1].name, "y") == 0,
                          "W0 slots[1].name == y");
                    CHECK(layout.slots[1].kind == ANE_SLOT_KIND_OUTPUT,
                          "W0 slots[1].kind == OUTPUT");
                }
                CHECK(layout.n_functions == 1, "W0 has 1 function");
                if (layout.n_functions == 1) {
                    CHECK(std::strcmp(layout.functions[0].name, "main") == 0,
                          "W0 functions[0].name == main");
                    CHECK(layout.functions[0].role == ANE_ROLE_MATMUL,
                          "W0 functions[0].role == MATMUL");
                    CHECK(layout.functions[0].stateful == false,
                          "W0 functions[0].stateful == false");
                    CHECK(layout.functions[0].n_inputs == 1,
                          "W0 functions[0].n_inputs == 1");
                    CHECK(layout.functions[0].n_outputs == 1,
                          "W0 functions[0].n_outputs == 1");
                    CHECK(std::strcmp(layout.functions[0].core_ml_function_name,
                                      "main") == 0,
                          "W0 functions[0].core_ml_function_name == main");
                }
                CHECK(layout.n_deps == 0, "W0 has 0 dependencies");
            } else {
                // External manifest (e.g. multifunction prefill).
                // We only check manifest-shape invariants that
                // hold for any valid manifest; bundle-specific
                // assertions live in the manifest-emitter test
                // (tools/ane-mtp/test_emit_manifest.py).
                std::fprintf(stdout,
                    "  external manifest: %s (%u slots, %u functions)\n",
                    layout.bundle_name, layout.n_slots, layout.n_functions);
            }
        } else {
            std::fprintf(stderr, "  error: %s\n", error.c_str());
        }
    }

    // --- Test 2: missing file is rejected with a clear error ---
    {
        ane_state_layout_v1_t layout;
        std::string error;
        const bool ok = ane_layout::read_state_layout(
            "/tmp/this-path-does-not-exist-ane_state.ane_state.v1.json",
            &layout, &error);
        CHECK(!ok, "missing file is rejected");
        CHECK(!error.empty(), "missing file error string is non-empty");
        CHECK(error.find("manifest not found") != std::string::npos,
              "missing file error mentions 'manifest not found'");
    }

    // --- Test 3: empty file is rejected ---
    {
        std::string path;
        CHECK(write_temp("", &path), "write empty temp file");
        ane_state_layout_v1_t layout;
        std::string error;
        const bool ok = ane_layout::read_state_layout(
            path.c_str(), &layout, &error);
        CHECK(!ok, "empty file is rejected");
        ::unlink(path.c_str());
    }

    // --- Test 4: non-JSON file is rejected ---
    {
        std::string path;
        CHECK(write_temp("this is not json", &path),
              "write non-JSON temp file");
        ane_state_layout_v1_t layout;
        std::string error;
        const bool ok = ane_layout::read_state_layout(
            path.c_str(), &layout, &error);
        CHECK(!ok, "non-JSON file is rejected");
        ::unlink(path.c_str());
    }

    // --- Test 5: version mismatch (write version=2) is rejected ---
    {
        std::string path;
        const std::string bad_version =
            "{\"version\": 2, \"bundle_name\": \"future\", "
            "\"state_size_bytes\": 65536, \"slots\": [], \"functions\": []}";
        CHECK(write_temp(bad_version, &path), "write version=2 temp file");
        ane_state_layout_v1_t layout;
        std::string error;
        const bool ok = ane_layout::read_state_layout(
            path.c_str(), &layout, &error);
        CHECK(!ok, "version=2 is rejected");
        CHECK(error.find("version") != std::string::npos,
              "version-mismatch error mentions 'version'");
        ::unlink(path.c_str());
    }

    // --- Test 6: missing required field (bundle_name) is rejected ---
    {
        std::string path;
        const std::string no_bundle_name =
            "{\"version\": 1, \"state_size_bytes\": 65536, "
            "\"slots\": [], \"functions\": []}";
        CHECK(write_temp(no_bundle_name, &path),
              "write missing-bundle_name temp file");
        ane_state_layout_v1_t layout;
        std::string error;
        const bool ok = ane_layout::read_state_layout(
            path.c_str(), &layout, &error);
        CHECK(!ok, "missing bundle_name is rejected");
        CHECK(error.find("bundle_name") != std::string::npos,
              "missing-bundle_name error mentions 'bundle_name'");
        ::unlink(path.c_str());
    }

    // --- Test 7: manifest path resolver ---
    {
        const std::string resolved = ane_layout::manifest_path_for_mlmodelc_dir(
            "/tmp/foo/w0-256x256.mlmodelc");
        CHECK(resolved ==
              "/tmp/foo/w0-256x256.ane_state.v1.json",
              "manifest_path_for_mlmodelc_dir matches the convention");
    }
    {
        const std::string resolved = ane_layout::manifest_path_for_mlmodelc_dir(
            nullptr);
        CHECK(resolved.empty(),
              "manifest_path_for_mlmodelc_dir(nullptr) returns empty");
    }

    // --- Test 8: multifunction-shaped manifest (synthetic) round-trips ---
    {
        const std::string multifunction =
            "{\n"
            "  \"version\": 1,\n"
            "  \"bundle_name\": \"gemma4-prefill-mtp\",\n"
            "  \"state_size_bytes\": 8388608,\n"
            "  \"model_type\": \"ml_program\",\n"
            "  \"slots\": [\n"
            "    {\"name\": \"token_ids\", \"kind\": \"input\", "
            "     \"dtype\": \"i32\", \"shape\": [1, 32], "
            "     \"offset\": 0, \"size_bytes\": 2048},\n"
            "    {\"name\": \"positions\", \"kind\": \"input\", "
            "     \"dtype\": \"i32\", \"shape\": [1, 32], "
            "     \"offset\": 2048, \"size_bytes\": 2048},\n"
            "    {\"name\": \"hidden_states\", \"kind\": \"state\", "
            "     \"dtype\": \"f16\", \"shape\": [1, 32, 3072], "
            "     \"offset\": 16384, \"size_bytes\": 196608},\n"
            "    {\"name\": \"top_token\", \"kind\": \"output\", "
            "     \"dtype\": \"i32\", \"shape\": [1], "
            "     \"offset\": 262144, \"size_bytes\": 16}\n"
            "  ],\n"
            "  \"functions\": [\n"
            "    {\"name\": \"prefill_s32\", \"role\": \"prefill\", "
            "     \"bucket\": 32, \"stateful\": true, "
            "     \"input_slots\": [\"token_ids\", \"positions\"], "
            "     \"output_slots\": [\"hidden_states\"], "
            "     \"core_ml_function_name\": \"prefill_s32\", "
            "     \"use_ane\": true},\n"
            "    {\"name\": \"mtp_predict\", \"role\": \"mtp\", "
            "     \"bucket\": 1, \"stateful\": true, "
            "     \"input_slots\": [\"token_ids\", \"positions\", "
            "                        \"hidden_states\"], "
            "     \"output_slots\": [\"top_token\"], "
            "     \"core_ml_function_name\": \"mtp_predict\", "
            "     \"use_ane\": true}\n"
            "  ],\n"
            "  \"dependencies\": [\n"
            "    {\"producer\": \"prefill_s32\", "
            "     \"slot\": \"hidden_states\", "
            "     \"consumers\": [\"mtp_predict\"]}\n"
            "  ]\n"
            "}\n";
        std::string path;
        CHECK(write_temp(multifunction, &path), "write multifunction temp");
        ane_state_layout_v1_t layout;
        std::string error;
        const bool ok = ane_layout::read_state_layout(
            path.c_str(), &layout, &error);
        CHECK(ok, "multifunction manifest is read successfully");
        if (ok) {
            CHECK(layout.model_type == ANE_MODEL_TYPE_ML_PROGRAM,
                  "multifunction model_type == ML_PROGRAM");
            CHECK(layout.n_functions == 2,
                  "multifunction has 2 functions");
            CHECK(layout.n_deps == 1,
                  "multifunction has 1 dependency");
            if (layout.n_deps == 1) {
                CHECK(layout.deps[0].producer_function_id == 0,
                      "deps[0].producer_function_id == 0 (prefill_s32)");
                CHECK(layout.deps[0].slot_id == 2,
                      "deps[0].slot_id == 2 (hidden_states)");
                CHECK(layout.deps[0].n_consumers == 1,
                      "deps[0] has 1 consumer");
                CHECK(layout.deps[0].consumer_function_ids[0] == 1,
                      "deps[0] consumer is mtp_predict (id 1)");
            }
        } else {
            std::fprintf(stderr, "  error: %s\n", error.c_str());
        }
        ::unlink(path.c_str());
    }

    if (g_failures == 0) {
        std::fprintf(stdout, "\nALL PASSED\n");
        return 0;
    }
    std::fprintf(stderr, "\n%d FAILURE(S)\n", g_failures);
    return 1;
}
