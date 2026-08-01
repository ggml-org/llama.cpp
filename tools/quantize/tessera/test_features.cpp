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

int main() {
    test_roundtrip_layers();
    test_fused_matches_layered();
    test_error_paths();
    test_corrupt_header();

    printf("features: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
