//
// test_gguf_write_lifetime.cpp
//
// Regression test for the ts_gguf_write_tensor_cluster data-lifetime bug.
// The GGUF tensor descriptors reference the quant-result buffers by data
// pointer, and gguf_write_to_file reads through those pointers after the
// dispatch walk completes. This mirrors the fixed dispatch pattern: the
// result lives in a function-scope deque (stable address) and the descriptors
// are allocated from a caller-owned ggml_context. We write a real GGUF, tear
// down the write side, read it back, and byte-compare the payload. A dangling
// data pointer (the old bug) would mismatch or crash.
//

#include "tessera-quant.h"
#include "tessera-gguf-writer.h"

#include "ggml.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <vector>

static int g_fail = 0;
static void check(const char * name, bool ok) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", name);
    if (!ok) g_fail++;
}

static uint32_t rng = 12345;
static float randn_f() {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    float u1 = (float)(rng & 0xFFFFFF) / (float)0x1000000 + 1e-7f;
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    float u2 = (float)(rng & 0xFFFFFF) / (float)0x1000000;
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

int main() {
    const int64_t out_dim = 16;
    const int64_t in_dim  = 1280; // two 640-pages -> pages_per_row == 2

    // synthetic weights + per-channel act scales
    std::vector<float> w((size_t)(out_dim * in_dim));
    std::vector<float> act((size_t)in_dim);
    for (auto & x : w)   x = randn_f();
    for (auto & a : act) a = 0.5f + 0.5f * (float)(rng++ & 0xFF) / 255.0f;

    // result lives in a function-scope deque, exactly like the fixed dispatch
    std::deque<ts_quant_result_2d> cluster_results;
    ts_quant_result_2d & qr = cluster_results.emplace_back();

    ts_quant_params_2d qp = {};
    qp.alpha          = 0.0f;
    qp.clip           = 1.0f;
    qp.max_outliers   = 4;
    qp.outlier_thresh = 2.0f;
    qp.awq_grid       = 5;

    int rc = ts_quantize_2d(w.data(), act.data(), nullptr, nullptr, act.data(),
                            out_dim, in_dim, 0, &qp, &qr);
    check("quantize rc == 0", rc == 0);
    check("packed non-empty", !qr.packed.empty());

    // pristine copy of the packed payload to compare after the round-trip
    const std::vector<uint32_t> packed_copy = qr.packed;

    // build the output GGUF with a caller-owned tensor context
    struct gguf_context * out_ctx = gguf_init_empty();
    check("gguf_init_empty", out_ctx != nullptr);

    struct ggml_init_params ip = { /*mem_size=*/ 64 * 1024, /*mem_buffer=*/ nullptr, /*no_alloc=*/ true };
    struct ggml_context * out_ggml_ctx = ggml_init(ip);
    check("ggml_init", out_ggml_ctx != nullptr);

    ts_gguf_writer_params mp = {};
    mp.seed = 1; mp.alpha = 0.0f; mp.clip = 1.0f; mp.outlier_frac = 2.0f;
    ts_gguf_write_metadata(out_ctx, &mp);

    ts_gguf_write_tensor_cluster(out_ctx, out_ggml_ctx, "blk.0.attn_q", &qr, out_dim, in_dim);

    check("cluster added >= 6 tensors", gguf_get_n_tensors(out_ctx) >= 6);

    const char * path = "tessera_gguf_write_test.gguf";
    check("gguf_write_to_file", gguf_write_to_file(out_ctx, path, false));

    // tear down the write side BEFORE readback, as dispatch does at cleanup
    ggml_free(out_ggml_ctx);
    gguf_free(out_ctx);

    // read it back and verify the payload survived (would fail if the
    // descriptors had pointed at a freed per-iteration local)
    struct ggml_context * rin_ctx = nullptr;
    struct gguf_init_params rp = { /*no_alloc=*/ false, /*ctx=*/ &rin_ctx };
    struct gguf_context * rin = gguf_init_from_file(path, rp);
    check("reopen gguf", rin != nullptr);

    if (rin) {
        struct ggml_tensor * pt = ggml_get_tensor(rin_ctx, "blk.0.attn_q.weight_packed");
        check("weight_packed present", pt != nullptr);
        if (pt) {
            const size_t cmp_bytes = packed_copy.size() * sizeof(uint32_t);
            const bool same = (ggml_nbytes(pt) >= cmp_bytes) &&
                              (memcmp(pt->data, packed_copy.data(), cmp_bytes) == 0);
            check("weight_packed round-trips byte-identical", same);
        }
        gguf_free(rin);
        ggml_free(rin_ctx);
    }

    std::remove(path);

    std::printf("\n%s (%d failures)\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
