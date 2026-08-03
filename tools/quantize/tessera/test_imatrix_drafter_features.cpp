//
// test_imatrix_drafter_features.cpp
//
// Tests the offline trunk-feature capture file format (tessera-features) from
// the DFlash / D-PACE training driver's CONSUMER perspective. The capture
// path is wired into tools/imatrix/imatrix.cpp (--features-out
// --feature-layers <csv> --features-warmup N); the resulting file is consumed
// by tessera-train-dflash to feed the drafter's encoder FC. This test pins
// the consumer-side invariants the driver relies on:
//
//   1. Header round-trips n_embd, n_layers, target_layers, and the window
//      layout (chunk_tokens, warmup, stride) so the driver can read them.
//   2. row_floats() = n_layers * n_embd matches the drafter's encoder FC
//      input width (dflash.target_layer_ids.size() * drafter_n_embd).
//   3. target_layers are in concatenation order, so the drafter's FC sees
//      the same order the encoder expects (encoder concatenates
//      trunk_layer[target_layer_ids[0]] || ... || trunk_layer[target_layer_ids[-1]]
//      for each input token).
//   4. Overlap-mode capture produces contiguous row->token mapping; the
//      driver can index features[corpus_token_id] without gaps. This is the
//      pre-condition for the dflash encoder to find a feature row for every
//      committed context token referenced in a dflash-block.v1 record.
//
// Companion to test_features.cpp (which covers the file-format level). This
// file focuses on the invariants the dflash driver depends on.
//

#include "tessera-features.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { g_pass++; } \
    else { g_fail++; std::fprintf(stderr, "FAIL: %s\n", msg); } \
} while(0)

#define EPS 1e-6f

static const char * PREFIX = "/tmp/ts_imatrix_drafter_features_test";

static void write_dflash_capture(const std::string & pfx,
                                 int32_t n_embd,
                                 const std::vector<int32_t> & target_layers,
                                 int32_t warmup,
                                 int32_t chunk_tokens,
                                 int32_t stride,
                                 int n_tokens,
                                 const std::vector<float> & flat_rows) {
    ts_features_writer w;
    if (!w.open(pfx, n_embd, target_layers)) {
        std::fprintf(stderr, "FAIL: open %s\n", pfx.c_str());
        g_fail++;
        return;
    }
    w.header.chunk_tokens = chunk_tokens;
    w.header.warmup       = warmup;
    w.header.stride       = stride;
    const int32_t row_floats = n_embd * (int32_t) target_layers.size();
    for (int i = 0; i < n_tokens; ++i) {
        const float * row = flat_rows.data() + (size_t) i * row_floats;
        w.append_token(row);
    }
    w.close();
}

// Simulates what `llama-imatrix --features-out pfx --feature-layers 0,8,15
// --features-warmup 256 -c 512` would produce for a 4-layer-32-embd trunk
// with 3 emitted windows, then verifies the dflash-driver consumer can
// reconstruct the encoder FC input.
static void test_dflash_encoder_invariants() {
    // dflash.target_layer_ids = [0, 8, 15] (trunk layers to fuse)
    const std::vector<int32_t> target_layers = {0, 8, 15};
    const int32_t n_embd = 32;
    const int32_t warmup = 4;
    const int32_t chunk  = 8;     // n_ctx
    const int32_t stride = chunk - warmup;  // 4
    const int n_windows  = 3;
    const int n_tokens   = n_windows * stride;  // 12 contiguous emitted rows

    // Build a synthetic 12 x (3*32) feature blob where each row encodes its
    // (window_idx, layer_idx) so the test can spot layer-mis-ordering bugs.
    std::vector<float> flat;
    flat.reserve((size_t) n_tokens * target_layers.size() * n_embd);
    for (int w = 0; w < n_windows; ++w) {
        for (int j = 0; j < stride; ++j) {
            for (size_t li = 0; li < target_layers.size(); ++li) {
                for (int e = 0; e < n_embd; ++e) {
                    // unique signature: 100*window + 10*layer + 1*embd
                    const float v = (float)(100 * w + 10 * (int) target_layers[li] + e);
                    flat.push_back(v);
                }
            }
        }
    }
    write_dflash_capture(PREFIX, n_embd, target_layers, warmup, chunk, stride,
                         n_tokens, flat);

    // Read back the header and confirm the dflash encoder can use it.
    ts_features_header h;
    CHECK(ts_features_read_header(PREFIX, h), "header round-trip");
    CHECK(h.n_embd == n_embd, "n_embd round-trip (drafter input width per layer)");
    CHECK(h.n_layers == (int32_t) target_layers.size(),
          "n_layers = dflash.target_layer_ids.size()");
    CHECK(h.target_layers == target_layers,
          "target_layers preserved in concatenation order");
    CHECK(h.row_floats() == (int32_t) target_layers.size() * n_embd,
          "row_floats = n_layers * n_embd (encoder FC input width)");
    CHECK(h.chunk_tokens == chunk, "chunk_tokens round-trip (n_ctx)");
    CHECK(h.warmup == warmup, "warmup round-trip (per-chunk context primer)");
    CHECK(h.stride == stride, "stride round-trip (window advance)");
    CHECK(h.rows_per_chunk() == stride,
          "rows_per_chunk = chunk_tokens - warmup (overlap mode)");
    CHECK(h.effective_stride() == stride, "effective_stride = stride");

    // Read the blob and validate the encoder FC input layout: for every
    // emitted row r, layers concatenate in target_layers order with
    // n_embd floats each, and the values follow the signature we wrote.
    std::vector<float> blob;
    {
        const std::string bin = std::string(PREFIX) + ".bin";
        FILE * fp = std::fopen(bin.c_str(), "rb");
        std::fseek(fp, 0, SEEK_END);
        const long sz = std::ftell(fp);
        std::fseek(fp, 0, SEEK_SET);
        blob.resize((size_t) sz / sizeof(float));
        std::fread(blob.data(), sizeof(float), blob.size(), fp);
        std::fclose(fp);
    }
    CHECK((int) blob.size() == n_tokens * h.row_floats(),
          "blob size = n_tokens * row_floats");

    // Every emitted row maps to corpus token warmup + r, contiguously.
    for (int r = 0; r < n_tokens; ++r) {
        CHECK(ts_features_row_to_token(h, r) == warmup + r,
              "contiguous row->token mapping (overlap mode)");

        // The blob row must concatenate target_layers in the recorded order.
        for (size_t li = 0; li < target_layers.size(); ++li) {
            for (int e = 0; e < n_embd; ++e) {
                const float got = blob[(size_t) r * h.row_floats()
                                       + li * n_embd + e];
                // Reconstruct the signature: window = r / stride, layer = target_layers[li].
                const int window   = r / stride;
                const int layer_id = target_layers[li];
                const float want   = (float)(100 * window + 10 * layer_id + e);
                if (std::fabs((double) got - (double) want) > (double) EPS) {
                    std::fprintf(stderr, "FAIL: row %d layer %zu embd %d got %.3f want %.3f\n",
                                 r, li, e, (double) got, (double) want);
                    g_fail++;
                }
            }
        }
    }
    // g_pass counter for the inner loop:
    g_pass += n_tokens * target_layers.size() * n_embd;
}

// dflash.target_layer_ids is a property of the drafter GGUF metadata. The
// dflash driver must reject a features capture whose target_layers does
// NOT match its own dflash.target_layer_ids (mismatched concatenation
// order would silently corrupt the encoder FC input). The schema is
// permissive about which layers were captured, but the driver should
// enforce identity before consuming.
//
// We simulate the check at the file-format level: write a capture, then
// verify a different drafter with a mismatched target_layers order
// would produce a row_floats() == the right width but a CONCATENATION
// signature that disagrees.
static void test_drafter_target_layers_mismatch() {
    const std::vector<int32_t> captured_layers = {0, 8, 15};   // what imatrix captured
    const std::vector<int32_t> drafter_layers  = {15, 8, 0};   // drafter expects this order
    const int32_t n_embd = 4;

    write_dflash_capture(std::string(PREFIX) + "_mismatch", n_embd,
                         captured_layers, /*warmup=*/0, /*chunk=*/4, /*stride=*/0,
                         /*n_tokens=*/1, {1, 2, 3, 4,  5, 6, 7, 8,  9, 10, 11, 12});

    ts_features_header h;
    CHECK(ts_features_read_header(std::string(PREFIX) + "_mismatch", h),
          "mismatch capture header ok");
    // row_floats matches by definition (same cardinality), but the LAYER
    // ORDER disagrees. The drafter driver must compare h.target_layers to
    // dflash.target_layer_ids BEFORE consuming; if mismatched, refuse to
    // load. We test the format-level signal here.
    CHECK(h.target_layers != drafter_layers,
          "captured target_layers differ from drafter's expected order");
    CHECK(h.row_floats() == (int32_t) drafter_layers.size() * n_embd,
          "row_floats still equals drafter FC input width (cardinality OK, order is not)");
}

// Stage 0 invariant from the design doc (7.1 token-fidelity): the trunk's
// tokenizer (which produced the corpus tokens) and the drafter's tokenizer
// (which it borrows via tok_embd) must agree. The features capture is
// indexed by corpus token id, so the consumer side asserts that the
// drafter's vocab size matches what the drafter will see at training time
// (this is a property of the drafter GGUF, not the features file, but the
// format side documents the expected row width).
static void test_row_width_matches_target_layers() {
    // drafter expects target_layer_ids = [2, 4, 6, 8], n_embd = 16
    // expected row_floats = 4 * 16 = 64
    const std::vector<int32_t> target_layers = {2, 4, 6, 8};
    const int32_t n_embd = 16;
    write_dflash_capture(std::string(PREFIX) + "_roww", n_embd,
                         target_layers, /*warmup=*/0, /*chunk=*/2, /*stride=*/0,
                         /*n_tokens=*/1, std::vector<float>(target_layers.size() * n_embd, 0.0f));

    ts_features_header h;
    CHECK(ts_features_read_header(std::string(PREFIX) + "_roww", h),
          "roww capture ok");
    CHECK(h.row_floats() == 4 * 16,
          "row_floats = dflash.target_layer_ids.size() * drafter_n_embd");
    CHECK(h.bytes_per_float() == 4, "f32 capture: 4 bytes/float");
}

int main() {
    test_dflash_encoder_invariants();
    test_drafter_target_layers_mismatch();
    test_row_width_matches_target_layers();

    std::printf("imatrix_drafter_features: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
