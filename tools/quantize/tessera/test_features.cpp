// test_features.cpp - offline tests for tessera-features (file format, no model)
#include "tessera-features.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { g_pass++; } \
    else { g_fail++; fprintf(stderr, "FAIL: %s\n", msg); } \
} while(0)

#define EPS 1e-6f

static const char * PREFIX = "/tmp/ts_features_test";

// read the whole blob back as f32 rows.
static std::vector<float> read_blob(const std::string & path) {
    std::vector<float> out;
    FILE * fp = std::fopen(path.c_str(), "rb");
    if (!fp) { return out; }
    std::fseek(fp, 0, SEEK_END);
    const long sz = std::ftell(fp);
    std::fseek(fp, 0, SEEK_SET);
    out.resize((size_t) sz / sizeof(float));
    if (!out.empty()) {
        std::fread(out.data(), sizeof(float), out.size(), fp);
    }
    std::fclose(fp);
    return out;
}

static void test_roundtrip_layers() {
    // 3 target layers, n_embd = 4, 2 tokens. Layer order {10, 2, 7} must be
    // preserved in the on-disk concatenation order.
    const int32_t n_embd = 4;
    const std::vector<int32_t> layers = {10, 2, 7};

    ts_features_writer w;
    CHECK(w.open(PREFIX, n_embd, layers), "open ok");

    // token 0: layer rows are distinct constants per layer.
    float t0_l0[4] = { 1,  2,  3,  4};   // target layer 10
    float t0_l1[4] = { 5,  6,  7,  8};   // target layer 2
    float t0_l2[4] = { 9, 10, 11, 12};   // target layer 7
    const float * t0[3] = {t0_l0, t0_l1, t0_l2};
    CHECK(w.append_token_layers(t0), "append token 0");

    // token 1
    float t1_l0[4] = {0.1f, 0.2f, 0.3f, 0.4f};
    float t1_l1[4] = {0.5f, 0.6f, 0.7f, 0.8f};
    float t1_l2[4] = {0.9f, 1.0f, 1.1f, 1.2f};
    const float * t1[3] = {t1_l0, t1_l1, t1_l2};
    CHECK(w.append_token_layers(t1), "append token 1");

    CHECK(w.n_tokens_written() == 2, "n_tokens_written == 2");
    CHECK(w.close(), "close ok");

    // header round-trip
    ts_features_header h;
    CHECK(ts_features_read_header(PREFIX, h), "read header ok");
    CHECK(h.n_tokens == 2, "header n_tokens");
    CHECK(h.n_embd == 4, "header n_embd");
    CHECK(h.n_layers == 3, "header n_layers");
    CHECK(h.target_layers == layers, "header target_layers order preserved");
    CHECK(h.dtype == TS_FEATURES_F32, "header dtype f32");
    CHECK(h.row_floats() == 12, "header row_floats = 3*4");
    CHECK(h.bytes_per_float() == 4, "header bytes_per_float = 4");

    // blob layout: row-major [n_tokens, n_layers*n_embd], layers concatenated
    // in target_layers order.
    std::vector<float> blob = read_blob(std::string(PREFIX) + ".bin");
    CHECK(blob.size() == 2u * 12u, "blob size = 2 rows * 12 floats");

    // token 0 row = [l0 | l1 | l2]
    const float * r0 = blob.data();
    CHECK(std::memcmp(r0 + 0, t0_l0, 4*sizeof(float)) == 0, "t0 layer0 block");
    CHECK(std::memcmp(r0 + 4, t0_l1, 4*sizeof(float)) == 0, "t0 layer1 block");
    CHECK(std::memcmp(r0 + 8, t0_l2, 4*sizeof(float)) == 0, "t0 layer2 block");

    // token 1 row
    const float * r1 = blob.data() + 12;
    bool t1_ok = true;
    for (int i = 0; i < 4; ++i) {
        t1_ok &= std::fabs(r1[0+i] - t1_l0[i]) < EPS;
        t1_ok &= std::fabs(r1[4+i] - t1_l1[i]) < EPS;
        t1_ok &= std::fabs(r1[8+i] - t1_l2[i]) < EPS;
    }
    CHECK(t1_ok, "t1 fused row values");
}

static void test_fused_matches_layered() {
    // append_token (pre-fused) must produce byte-identical output to
    // append_token_layers for the same data.
    const int32_t n_embd = 3;
    const std::vector<int32_t> layers = {0, 5};

    float l0[3] = {1, 2, 3};
    float l1[3] = {4, 5, 6};
    float fused[6] = {1, 2, 3, 4, 5, 6};

    ts_features_writer wa;
    CHECK(wa.open(std::string(PREFIX) + "_a", n_embd, layers), "open a");
    const float * lp[2] = {l0, l1};
    CHECK(wa.append_token_layers(lp), "append layered");
    CHECK(wa.close(), "close a");

    ts_features_writer wb;
    CHECK(wb.open(std::string(PREFIX) + "_b", n_embd, layers), "open b");
    CHECK(wb.append_token(fused), "append fused");
    CHECK(wb.close(), "close b");

    std::vector<float> ba = read_blob(std::string(PREFIX) + "_a.bin");
    std::vector<float> bb = read_blob(std::string(PREFIX) + "_b.bin");
    CHECK(ba.size() == 6 && bb.size() == 6, "both blobs 6 floats");
    CHECK(ba == bb, "fused == layered byte-for-byte");
}

static void test_error_paths() {
    ts_features_writer w;
    // empty layer order rejected
    CHECK(!w.open(PREFIX, 4, {}), "empty layers rejected");
    // bad n_embd rejected
    CHECK(!w.open(PREFIX, 0, {1, 2}), "zero n_embd rejected");
    // f16 not implemented yet -> rejected
    CHECK(!w.open(PREFIX, 4, {1, 2}, TS_FEATURES_F16), "f16 rejected (not impl)");

    // read_header on a missing prefix fails cleanly
    ts_features_header h;
    CHECK(!ts_features_read_header("/tmp/ts_features_does_not_exist", h), "missing header fails");
}

static void test_corrupt_header() {
    // write a valid file, then corrupt the schema and confirm the reader
    // rejects it.
    const std::string pfx = std::string(PREFIX) + "_corrupt";
    ts_features_writer w;
    CHECK(w.open(pfx, 2, {3}), "open corrupt-base");
    float row[2] = {1, 2};
    CHECK(w.append_token(row), "append corrupt-base");
    CHECK(w.close(), "close corrupt-base");

    // overwrite the json with a wrong schema version.
    FILE * fp = std::fopen((pfx + ".json").c_str(), "w");
    const char * bad = "{\"schema_version\":\"llama.tessera.features.v9\",\"n_tokens\":1,\"n_embd\":2,\"n_layers\":1,\"target_layers\":[3],\"dtype\":\"f32\"}";
    std::fwrite(bad, 1, std::strlen(bad), fp);
    std::fclose(fp);

    ts_features_header h;
    CHECK(!ts_features_read_header(pfx, h), "wrong schema rejected");

    // mismatched target_layers length vs n_layers is rejected.
    fp = std::fopen((pfx + ".json").c_str(), "w");
    const char * bad2 = "{\"schema_version\":\"llama.tessera.features.v1\",\"n_tokens\":1,\"n_embd\":2,\"n_layers\":2,\"target_layers\":[3],\"dtype\":\"f32\"}";
    std::fwrite(bad2, 1, std::strlen(bad2), fp);
    std::fclose(fp);
    CHECK(!ts_features_read_header(pfx, h), "layer-count mismatch rejected");
}

static void test_chunk_layout_roundtrip() {
    // warmup + chunk_tokens + stride survive the header round-trip so the
    // training driver can reconstruct row -> corpus-token alignment.
    const std::string pfx = std::string(PREFIX) + "_chunky";
    ts_features_writer w;
    CHECK(w.open(pfx, 2, {4, 9}), "open chunky");
    w.header.chunk_tokens = 128;
    w.header.warmup       = 8;
    w.header.stride       = 120;   // overlap mode: stride == chunk_tokens - warmup
    float row[2] = {1, 2};
    CHECK(w.append_token(row), "append chunky");
    CHECK(w.close(), "close chunky");

    ts_features_header h;
    CHECK(ts_features_read_header(pfx, h), "read chunky header");
    CHECK(h.chunk_tokens == 128, "chunk_tokens round-trip");
    CHECK(h.warmup == 8, "warmup round-trip");
    CHECK(h.stride == 120, "stride round-trip");
    CHECK(h.rows_per_chunk() == 120, "rows_per_chunk = 128 - 8");
    CHECK(h.effective_stride() == 120, "effective_stride = stride when set");

    // a header with no stride falls back to chunk_tokens (legacy files).
    ts_features_header legacy;
    legacy.chunk_tokens = 128;
    legacy.warmup       = 8;
    legacy.stride       = 0;
    CHECK(legacy.effective_stride() == 128, "effective_stride legacy fallback");
}

static void write_with_stride(const std::string & pfx, int32_t stride) {
    ts_features_writer w;
    if (!w.open(pfx, 2, {4})) { return; }
    w.header.chunk_tokens = 128;
    w.header.warmup       = 8;
    w.header.stride       = stride;
    float row[2] = {1, 2};
    w.append_token(row);
    w.close();
}

static void test_stride_validation() {
    // a stride that would double-emit (stride < rows_per_window) or skip
    // (stride > chunk_tokens) is rejected by the reader. rows_per_window here
    // is chunk_tokens - warmup = 128 - 8 = 120.
    ts_features_header h;

    write_with_stride(std::string(PREFIX) + "_stride_lo", 64);   // < 120
    CHECK(!ts_features_read_header(std::string(PREFIX) + "_stride_lo", h),
          "stride < rows_per_window rejected");

    write_with_stride(std::string(PREFIX) + "_stride_hi", 200);  // > 128
    CHECK(!ts_features_read_header(std::string(PREFIX) + "_stride_hi", h),
          "stride > chunk_tokens rejected");

    // stride == rows_per_window (overlap) and stride == chunk_tokens (legacy)
    // are both valid.
    write_with_stride(std::string(PREFIX) + "_stride_ok", 120);
    CHECK(ts_features_read_header(std::string(PREFIX) + "_stride_ok", h),
          "stride == rows_per_window accepted");
}

static void test_row_to_token() {
    // LEGACY layout (stride == 0): windows advanced by a full chunk_tokens and
    // discarded a warmup prefix per window, so the mapping has a per-window gap.
    // chunk_tokens=128, warmup=8 -> 120 emitted rows/window.
    ts_features_header h;
    h.n_tokens     = 240;   // 2 windows
    h.n_embd       = 2;
    h.n_layers     = 1;
    h.target_layers = {0};
    h.chunk_tokens = 128;
    h.warmup       = 8;
    h.stride       = 0;     // legacy

    CHECK(ts_features_row_to_token(h, 0)   == 8,   "legacy row0 -> token 8 (first after warmup)");
    CHECK(ts_features_row_to_token(h, 119) == 127, "legacy row119 -> token 127 (window0 end)");
    CHECK(ts_features_row_to_token(h, 120) == 136, "legacy row120 -> token 136 (window1 start: 128+8, gap)");
    CHECK(ts_features_row_to_token(h, 239) == 255, "legacy row239 -> token 255 (window1 end)");
    CHECK(ts_features_row_to_token(h, 240) == -1,  "row240 out of range");
    CHECK(ts_features_row_to_token(h, -1)  == -1,  "negative row rejected");

    // OVERLAP layout (stride == rows_per_window): windows overlap by `warmup`,
    // so the emitted rows are contiguous and row r -> token warmup + r.
    ts_features_header ov;
    ov.n_tokens     = 240;
    ov.n_embd       = 2;
    ov.n_layers     = 1;
    ov.target_layers = {0};
    ov.chunk_tokens = 128;
    ov.warmup       = 8;
    ov.stride       = 120;  // == chunk_tokens - warmup

    CHECK(ts_features_row_to_token(ov, 0)   == 8,   "overlap row0 -> token 8");
    CHECK(ts_features_row_to_token(ov, 119) == 127, "overlap row119 -> token 127");
    CHECK(ts_features_row_to_token(ov, 120) == 128, "overlap row120 -> token 128 (contiguous, no gap)");
    CHECK(ts_features_row_to_token(ov, 239) == 247, "overlap row239 -> token 247");
    // contiguous invariant: every row maps to warmup + row.
    bool contiguous = true;
    for (int64_t r = 0; r < ov.n_tokens; ++r) {
        contiguous &= (ts_features_row_to_token(ov, r) == 8 + r);
    }
    CHECK(contiguous, "overlap mapping is contiguous (row -> warmup + row)");

    // no chunk layout + no warmup -> identity.
    ts_features_header flat;
    flat.n_tokens = 10;
    flat.n_embd   = 2;
    flat.n_layers = 1;
    flat.target_layers = {0};
    CHECK(ts_features_row_to_token(flat, 5) == 5, "flat identity mapping");

    // warmup without chunk layout is inconsistent.
    ts_features_header bad;
    bad.n_tokens = 10;
    bad.n_embd   = 2;
    bad.n_layers = 1;
    bad.target_layers = {0};
    bad.warmup   = 8;   // chunk_tokens stays 0
    CHECK(ts_features_row_to_token(bad, 0) == -1, "warmup-without-chunk rejected");

    // zero emitted rows per chunk is rejected.
    ts_features_header zero;
    zero.n_tokens = 5;
    zero.n_embd   = 2;
    zero.n_layers = 1;
    zero.target_layers = {0};
    zero.chunk_tokens = 8;
    zero.warmup       = 8;
    CHECK(ts_features_row_to_token(zero, 0) == -1, "zero rows-per-chunk rejected");
}

int main() {
    test_roundtrip_layers();
    test_fused_matches_layered();
    test_error_paths();
    test_corrupt_header();
    test_chunk_layout_roundtrip();
    test_stride_validation();
    test_row_to_token();

    printf("features: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
