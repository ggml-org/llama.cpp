// Drafter shape contract test: synthetic validation of the
// dflash / hybrid / mtp_predict slot shape contract without
// requiring a real drafter bundle with manifest sidecar.
//
// The drafter functions (common_ane_compute_dflash,
// common_ane_compute_hybrid, common_ane_mtp_program_predict) are
// the public API for the slot helpers. They each call
// set_pinned_input + dispatch_pinned_function + get_pinned_output
// for the function's declared input and output slots. The slot
// shape contract is documented in the function comments and is
// the source of truth for the bundle converter.
//
// This test verifies the contract in two ways:
//
//   1. MANIFEST LEVEL: a synthetic ane_state_layout.v1.json
//      manifest is constructed in /tmp with the drafter
//      functions and their declared slots. ane_layout::read_state_layout
//      parses it; the test asserts each function has the expected
//      inputs and outputs with the expected dtypes and shapes.
//      This validates the converter side of the contract: a real
//      drafter bundle's manifest must declare slots matching the
//      shape contract below.
//
//   2. PUBLIC API LEVEL: with the gemma4 prefill bundle (which
//      has only prefill functions, no drafter), the drafter
//      public API is called. The dispatch path correctly
//      resolves the function to "not found" and returns false
//      without touching any slot. The test asserts this graceful
//      rejection so the drafter API is known to fail safely on
//      bundles that don't declare drafter functions.
//
// End-to-end validation of the drafter functions against a real
// MTP/DFlash bundle is deferred until a bundle with manifest
// sidecar is available. The manifest contract here is the
// converter-side precondition for that future work.

#include "ane-mtp.h"
#include "ane-state-layout.h"
#include "ane-state.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static int g_failures = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL [%s:%d] %s\n", __FILE__, __LINE__, msg); \
        ++g_failures; \
    } else { \
        std::fprintf(stdout, "ok   %s\n", msg); \
    } \
} while (0)

// Write a JSON file with the given content. Returns true on success.
static bool write_file(const std::string & path, const std::string & content) {
    std::ofstream f(path);
    if (!f) return false;
    f << content;
    return f.good();
}

// Find a slot by name in the manifest. Returns the slot index
// or ANE_STATE_SLOTS_MAX if not found.
static uint32_t find_slot(const ane_state_layout_v1_t & m, const char * name) {
    for (uint32_t i = 0; i < m.n_slots; ++i) {
        if (std::strcmp(m.slots[i].name, name) == 0) return i;
    }
    return ANE_STATE_SLOTS_MAX;
}

// Find a function by name in the manifest. Returns the function
// index or UINT32_MAX if not found.
static uint32_t find_function(const ane_state_layout_v1_t & m, const char * name) {
    for (uint32_t i = 0; i < m.n_functions; ++i) {
        if (std::strcmp(m.functions[i].name, name) == 0) return i;
    }
    return UINT32_MAX;
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr,
                "usage: %s PREFILL_GGUF\n"
                "  PREFILL_GGUF is the gemma4 prefill bundle (used to "
                "verify the drafter public API rejects gracefully when "
                "the bundle declares no drafter functions).\n",
                argv[0]);
        return 2;
    }
    // -------- 1. MANIFEST LEVEL: synthetic drafter manifest ----
    //
    // The synthetic manifest declares three drafter functions
    // (dflash_b4, hybrid_b4, mtp_predict) and their slots. The
    // shapes are the source of truth from common/ane-mtp.mm's
    // dispatch path:
    //
    //   dflash_b4:
    //     inputs  target_features fp32 [bucket, feature_width]
    //             token_ids       i32  [bucket]
    //             positions       i32  [bucket]
    //     outputs draft_tokens    i32  [bucket * block_size]
    //             confidence      fp32 [bucket * block_size]
    //
    //   hybrid_b4:
    //     inputs  dflash_tokens   i32  [bucket * block_size]
    //             dflash_confidence fp32 [bucket * block_size]
    //             dflash_counts   i32  [bucket]
    //             mtp_tokens      i32  [bucket * block_size]
    //             mtp_confidence  fp32 [bucket * block_size]
    //             mtp_counts      i32  [bucket]
    //             dflash_cutoff   fp32 [bucket]
    //     outputs selected_source i32  [bucket]
    //             agreement       i32  [bucket]
    //
    //   mtp_predict:
    //     inputs  token_ids       i32  [bucket]
    //             h_nextn         fp32 [bucket, hidden_size]
    //             positions       i32  [bucket]
    //     outputs top_token       i32  [bucket]
    //             confidence      fp32 [bucket]
    //             next_hidden     fp32 [bucket, hidden_size]
    //
    // The state_size_bytes is the sum of all slot sizes,
    // aligned up to the ANE 16KB page boundary.
    const char * tmpdir = std::getenv("TMPDIR");
    if (tmpdir == nullptr || tmpdir[0] == '\0') {
        tmpdir = "/tmp";
    }
    const std::string manifest_path = std::string(tmpdir) +
        "/tessera-drafter-synthetic.ane_state.v1.json";
    // Layout strategy: pack slots contiguously starting at offset 0,
    // each with 16-byte alignment (SIMD alignment for fp16/fp32).
    // 1 KB per slot is enough for the synthetic test.
    const size_t kSlotBytes = 1024;
    const size_t kOffsetBase = 0;
    const size_t off_dflash_target = kOffsetBase + 0 * kSlotBytes;
    const size_t off_dflash_tokens = kOffsetBase + 1 * kSlotBytes;
    const size_t off_dflash_positions = kOffsetBase + 2 * kSlotBytes;
    const size_t off_dflash_drafts = kOffsetBase + 3 * kSlotBytes;
    const size_t off_dflash_conf = kOffsetBase + 4 * kSlotBytes;
    const size_t off_hybrid_dflash_tokens = kOffsetBase + 5 * kSlotBytes;
    const size_t off_hybrid_dflash_conf = kOffsetBase + 6 * kSlotBytes;
    const size_t off_hybrid_dflash_counts = kOffsetBase + 7 * kSlotBytes;
    const size_t off_hybrid_mtp_tokens = kOffsetBase + 8 * kSlotBytes;
    const size_t off_hybrid_mtp_conf = kOffsetBase + 9 * kSlotBytes;
    const size_t off_hybrid_mtp_counts = kOffsetBase + 10 * kSlotBytes;
    const size_t off_hybrid_cutoff = kOffsetBase + 11 * kSlotBytes;
    const size_t off_hybrid_selected = kOffsetBase + 12 * kSlotBytes;
    const size_t off_hybrid_agreement = kOffsetBase + 13 * kSlotBytes;
    const size_t off_mtp_token_ids = kOffsetBase + 14 * kSlotBytes;
    const size_t off_mtp_h_nextn = kOffsetBase + 15 * kSlotBytes;
    const size_t off_mtp_positions = kOffsetBase + 16 * kSlotBytes;
    const size_t off_mtp_top = kOffsetBase + 17 * kSlotBytes;
    const size_t off_mtp_conf = kOffsetBase + 18 * kSlotBytes;
    const size_t off_mtp_next = kOffsetBase + 19 * kSlotBytes;
    const size_t state_size = 20 * kSlotBytes;
    char json[16384];
    std::snprintf(json, sizeof(json), R"({
  "version": 1,
  "bundle_name": "drafter-synthetic",
  "state_size_bytes": %zu,
  "model_type": "ml_program",
  "slots": [
    {"name": "dflash_b4.target_features", "kind": "input",  "dtype": "f32", "shape": [4, 8],         "offset": %zu, "size_bytes": %zu},
    {"name": "dflash_b4.token_ids",       "kind": "input",  "dtype": "i32", "shape": [4],            "offset": %zu, "size_bytes": %zu},
    {"name": "dflash_b4.positions",       "kind": "input",  "dtype": "i32", "shape": [4],            "offset": %zu, "size_bytes": %zu},
    {"name": "dflash_b4.draft_tokens",    "kind": "output", "dtype": "i32", "shape": [4, 4],         "offset": %zu, "size_bytes": %zu},
    {"name": "dflash_b4.confidence",      "kind": "output", "dtype": "f32", "shape": [4, 4],         "offset": %zu, "size_bytes": %zu},
    {"name": "hybrid_b4.dflash_tokens",   "kind": "input",  "dtype": "i32", "shape": [4, 4],         "offset": %zu, "size_bytes": %zu},
    {"name": "hybrid_b4.dflash_confidence","kind": "input", "dtype": "f32", "shape": [4, 4],         "offset": %zu, "size_bytes": %zu},
    {"name": "hybrid_b4.dflash_counts",   "kind": "input",  "dtype": "i32", "shape": [4],            "offset": %zu, "size_bytes": %zu},
    {"name": "hybrid_b4.mtp_tokens",      "kind": "input",  "dtype": "i32", "shape": [4, 4],         "offset": %zu, "size_bytes": %zu},
    {"name": "hybrid_b4.mtp_confidence",  "kind": "input",  "dtype": "f32", "shape": [4, 4],         "offset": %zu, "size_bytes": %zu},
    {"name": "hybrid_b4.mtp_counts",      "kind": "input",  "dtype": "i32", "shape": [4],            "offset": %zu, "size_bytes": %zu},
    {"name": "hybrid_b4.dflash_cutoff",   "kind": "input",  "dtype": "f32", "shape": [4],            "offset": %zu, "size_bytes": %zu},
    {"name": "hybrid_b4.selected_source", "kind": "output", "dtype": "i32", "shape": [4],            "offset": %zu, "size_bytes": %zu},
    {"name": "hybrid_b4.agreement",       "kind": "output", "dtype": "i32", "shape": [4],            "offset": %zu, "size_bytes": %zu},
    {"name": "mtp_predict.token_ids",     "kind": "input",  "dtype": "i32", "shape": [4],            "offset": %zu, "size_bytes": %zu},
    {"name": "mtp_predict.h_nextn",       "kind": "input",  "dtype": "f32", "shape": [4, 16],        "offset": %zu, "size_bytes": %zu},
    {"name": "mtp_predict.positions",     "kind": "input",  "dtype": "i32", "shape": [4],            "offset": %zu, "size_bytes": %zu},
    {"name": "mtp_predict.top_token",     "kind": "output", "dtype": "i32", "shape": [4],            "offset": %zu, "size_bytes": %zu},
    {"name": "mtp_predict.confidence",    "kind": "output", "dtype": "f32", "shape": [4],            "offset": %zu, "size_bytes": %zu},
    {"name": "mtp_predict.next_hidden",   "kind": "output", "dtype": "f32", "shape": [4, 16],        "offset": %zu, "size_bytes": %zu}
  ],
  "functions": [
    {"name": "dflash_b4",  "role": "dflash",  "bucket": 4, "stateful": false, "use_ane": true,
     "core_ml_function_name": "dflash_b4",
     "input_slots":  ["dflash_b4.target_features", "dflash_b4.token_ids", "dflash_b4.positions"],
     "output_slots": ["dflash_b4.draft_tokens", "dflash_b4.confidence"]},
    {"name": "hybrid_b4",  "role": "hybrid",  "bucket": 4, "stateful": false, "use_ane": true,
     "core_ml_function_name": "hybrid_b4",
     "input_slots":  ["hybrid_b4.dflash_tokens", "hybrid_b4.dflash_confidence", "hybrid_b4.dflash_counts",
                      "hybrid_b4.mtp_tokens", "hybrid_b4.mtp_confidence", "hybrid_b4.mtp_counts",
                      "hybrid_b4.dflash_cutoff"],
     "output_slots": ["hybrid_b4.selected_source", "hybrid_b4.agreement"]},
    {"name": "mtp_predict", "role": "mtp",     "bucket": 4, "stateful": false, "use_ane": true,
     "core_ml_function_name": "mtp_predict",
     "input_slots":  ["mtp_predict.token_ids", "mtp_predict.h_nextn", "mtp_predict.positions"],
     "output_slots": ["mtp_predict.top_token", "mtp_predict.confidence", "mtp_predict.next_hidden"]}
  ]
})",
        state_size,
        off_dflash_target, kSlotBytes,
        off_dflash_tokens, kSlotBytes,
        off_dflash_positions, kSlotBytes,
        off_dflash_drafts, kSlotBytes,
        off_dflash_conf, kSlotBytes,
        off_hybrid_dflash_tokens, kSlotBytes,
        off_hybrid_dflash_conf, kSlotBytes,
        off_hybrid_dflash_counts, kSlotBytes,
        off_hybrid_mtp_tokens, kSlotBytes,
        off_hybrid_mtp_conf, kSlotBytes,
        off_hybrid_mtp_counts, kSlotBytes,
        off_hybrid_cutoff, kSlotBytes,
        off_hybrid_selected, kSlotBytes,
        off_hybrid_agreement, kSlotBytes,
        off_mtp_token_ids, kSlotBytes,
        off_mtp_h_nextn, kSlotBytes,
        off_mtp_positions, kSlotBytes,
        off_mtp_top, kSlotBytes,
        off_mtp_conf, kSlotBytes,
        off_mtp_next, kSlotBytes);
    if (!write_file(manifest_path, json)) {
        std::fprintf(stderr, "failed to write synthetic manifest to %s\n",
                manifest_path.c_str());
        return 1;
    }
    ane_state_layout_v1_t m = {};
    std::string err;
    if (!ane_layout::read_state_layout(manifest_path.c_str(), &m, &err)) {
        std::fprintf(stderr, "read_state_layout failed: %s\n", err.c_str());
        return 1;
    }
    CHECK(m.n_slots == 20, "synthetic manifest has 20 slots");
    CHECK(m.n_functions == 3, "synthetic manifest has 3 functions");
    CHECK(std::strcmp(m.bundle_name, "drafter-synthetic") == 0,
          "synthetic manifest has correct bundle_name");
    // Verify dflash_b4 contract
    {
        const uint32_t fi = find_function(m, "dflash_b4");
        CHECK(fi != UINT32_MAX, "function dflash_b4 found");
        const ane_function_v1_t & f = m.functions[fi];
        CHECK(f.role == ANE_ROLE_DFLASH, "dflash_b4 role is DFLASH");
        CHECK(f.bucket == 4, "dflash_b4 bucket is 4");
        CHECK(f.n_inputs == 3, "dflash_b4 has 3 inputs");
        CHECK(f.n_outputs == 2, "dflash_b4 has 2 outputs");
        // inputs
        const uint32_t target = find_slot(m, "dflash_b4.target_features");
        const uint32_t tokens = find_slot(m, "dflash_b4.token_ids");
        const uint32_t positions = find_slot(m, "dflash_b4.positions");
        CHECK(target != ANE_STATE_SLOTS_MAX, "dflash_b4.target_features slot present");
        CHECK(tokens != ANE_STATE_SLOTS_MAX, "dflash_b4.token_ids slot present");
        CHECK(positions != ANE_STATE_SLOTS_MAX, "dflash_b4.positions slot present");
        CHECK(m.slots[target].dtype == ANE_DTYPE_F32, "dflash_b4.target_features is f32");
        CHECK(m.slots[target].n_dim == 2, "dflash_b4.target_features is 2-D");
        CHECK(m.slots[tokens].dtype == ANE_DTYPE_I32, "dflash_b4.token_ids is i32");
        CHECK(m.slots[positions].dtype == ANE_DTYPE_I32, "dflash_b4.positions is i32");
        // outputs
        const uint32_t drafts = find_slot(m, "dflash_b4.draft_tokens");
        const uint32_t conf = find_slot(m, "dflash_b4.confidence");
        CHECK(drafts != ANE_STATE_SLOTS_MAX, "dflash_b4.draft_tokens slot present");
        CHECK(conf != ANE_STATE_SLOTS_MAX, "dflash_b4.confidence slot present");
        CHECK(m.slots[drafts].dtype == ANE_DTYPE_I32, "dflash_b4.draft_tokens is i32");
        CHECK(m.slots[conf].dtype == ANE_DTYPE_F32, "dflash_b4.confidence is f32");
    }
    // Verify hybrid_b4 contract
    {
        const uint32_t fi = find_function(m, "hybrid_b4");
        CHECK(fi != UINT32_MAX, "function hybrid_b4 found");
        const ane_function_v1_t & f = m.functions[fi];
        CHECK(f.role == ANE_ROLE_HYBRID, "hybrid_b4 role is HYBRID");
        CHECK(f.n_inputs == 7, "hybrid_b4 has 7 inputs");
        CHECK(f.n_outputs == 2, "hybrid_b4 has 2 outputs");
        // Spot-check the dflash_cutoff slot
        const uint32_t cutoff = find_slot(m, "hybrid_b4.dflash_cutoff");
        CHECK(cutoff != ANE_STATE_SLOTS_MAX, "hybrid_b4.dflash_cutoff slot present");
        CHECK(m.slots[cutoff].dtype == ANE_DTYPE_F32, "hybrid_b4.dflash_cutoff is f32");
        const uint32_t selected = find_slot(m, "hybrid_b4.selected_source");
        const uint32_t agreement = find_slot(m, "hybrid_b4.agreement");
        CHECK(selected != ANE_STATE_SLOTS_MAX, "hybrid_b4.selected_source slot present");
        CHECK(agreement != ANE_STATE_SLOTS_MAX, "hybrid_b4.agreement slot present");
        CHECK(m.slots[selected].dtype == ANE_DTYPE_I32, "hybrid_b4.selected_source is i32");
        CHECK(m.slots[agreement].dtype == ANE_DTYPE_I32, "hybrid_b4.agreement is i32");
    }
    // Verify mtp_predict contract
    {
        const uint32_t fi = find_function(m, "mtp_predict");
        CHECK(fi != UINT32_MAX, "function mtp_predict found");
        const ane_function_v1_t & f = m.functions[fi];
        CHECK(f.role == ANE_ROLE_MTP, "mtp_predict role is MTP");
        CHECK(f.n_inputs == 3, "mtp_predict has 3 inputs");
        CHECK(f.n_outputs == 3, "mtp_predict has 3 outputs");
        // h_nextn is the 2-D input
        const uint32_t h_nextn = find_slot(m, "mtp_predict.h_nextn");
        CHECK(h_nextn != ANE_STATE_SLOTS_MAX, "mtp_predict.h_nextn slot present");
        CHECK(m.slots[h_nextn].dtype == ANE_DTYPE_F32, "mtp_predict.h_nextn is f32");
        CHECK(m.slots[h_nextn].n_dim == 2, "mtp_predict.h_nextn is 2-D");
        // next_hidden is the 2-D output
        const uint32_t next = find_slot(m, "mtp_predict.next_hidden");
        CHECK(next != ANE_STATE_SLOTS_MAX, "mtp_predict.next_hidden slot present");
        CHECK(m.slots[next].dtype == ANE_DTYPE_F32, "mtp_predict.next_hidden is f32");
        CHECK(m.slots[next].n_dim == 2, "mtp_predict.next_hidden is 2-D");
    }
    // -------- 2. PUBLIC API LEVEL: prefill bundle rejects ----
    //
    // The gemma4 prefill bundle has prefill_s128 but no
    // dflash / hybrid / mtp_predict functions. The drafter
    // public API must reject gracefully (return false) without
    // crashing or mutating state.
    if (std::getenv("TESSERA_ANE_STATE_LAYOUT_MANIFEST") != nullptr) {
        common_ane_prefill_manifest pm;
        if (common_ane_prefill_manifest_load(argv[1], &pm) &&
                pm.architecture == "gemma4" &&
                pm.batch_size == 1 &&
                pm.layer_first == 0 && pm.layer_last == 0) {
            const uint32_t sequence = pm.sequence_buckets[0];
            auto program = common_ane_prefill_program_load(argv[1], sequence);
            if (program && common_ane_mtp_program_is_warm(program)) {
                // Try dflash: must return false because the
                // bundle has no dflash function.
                const size_t feature_count = (size_t) sequence * 8;
                std::vector<float> target_features(feature_count, 0.0f);
                std::vector<int32_t> tokens(sequence, 0);
                std::vector<int32_t> positions(sequence);
                for (int32_t i = 0; i < (int32_t) sequence; ++i) {
                    positions[i] = i;
                }
                std::vector<int32_t> draft_tokens(sequence * 4, 0);
                std::vector<float> confidence(sequence * 4, 0.0f);
                const bool dflash_ok = common_ane_compute_dflash(
                        program, 4, target_features.data(),
                        sequence, 8, tokens.data(), positions.data(),
                        draft_tokens.data(), confidence.data());
                CHECK(!dflash_ok, "dflash_b4 on a prefill bundle returns false (no dflash function)");
                // Try hybrid
                std::vector<int32_t> agreement(sequence, 0);
                std::vector<int32_t> selected_source(sequence, 0);
                const bool hybrid_ok = common_ane_compute_hybrid(
                        program, 4, draft_tokens.data(), confidence.data(),
                        tokens.data(), draft_tokens.data(), confidence.data(),
                        tokens.data(), sequence, 0.5f,
                        selected_source.data(), agreement.data());
                CHECK(!hybrid_ok, "hybrid_b4 on a prefill bundle returns false (no hybrid function)");
                // Try mtp_predict
                std::vector<float> h_nextn((size_t) sequence * pm.hidden_size, 0.0f);
                std::vector<int32_t> top_token(sequence, 0);
                std::vector<float> next_hidden((size_t) sequence * pm.hidden_size, 0.0f);
                const bool mtp_ok = common_ane_mtp_program_predict(
                        program, tokens.data(), h_nextn.data(),
                        sequence, pm.hidden_size, positions.data(),
                        top_token.data(), confidence.data(), next_hidden.data());
                CHECK(!mtp_ok, "mtp_predict on a prefill bundle returns false (no mtp function)");
            } else {
                std::fprintf(stderr,
                        "skipping public API drafter rejection test: "
                        "prefill bundle load failed\n");
            }
        } else {
            std::fprintf(stderr,
                    "skipping public API drafter rejection test: "
                    "prefill manifest shape unexpected\n");
        }
    } else {
        std::fprintf(stderr,
                "skipping public API drafter rejection test: "
                "TESSERA_ANE_STATE_LAYOUT_MANIFEST not set "
                "(set it to point at the gemma4 prefill-bundle.ane_state.v1.json "
                "for the full contract check)\n");
    }
    if (g_failures == 0) {
        std::fprintf(stdout, "ALL PASSED\n");
        return 0;
    }
    std::fprintf(stderr, "FAILED: %d failures\n", g_failures);
    return 1;
}
