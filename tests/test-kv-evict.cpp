// Tests for StreamingLLM-style KV cache eviction (--kv-evict-sink / --kv-evict-window)
//
// The feature keeps an attention-sink prefix plus a recent window, masks the
// middle, and recycles the oldest eligible masked-middle cell when the cache
// is full. Positions stay monotonic; the physical cache is bounded.
//
// Models are generated in-memory (no model files required), following the
// same approach as tests/test-llama-archs.cpp. The tests inspect the internal
// llama_kv_cache to verify the recycling invariants.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "ggml-cpp.h"
#include "llama.h"
#include "llama-cpp.h"

#include <unordered_map>
#include <unordered_set>

#include "../src/llama-arch.h"
#include "../src/llama-context.h"
#include "../src/llama-kv-cache.h"
#include "../src/llama-kv-cells.h"
#include "../src/llama-model-saver.h"

#include <bitset>
#include <cinttypes>
#include <cstdarg>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <vector>

static int g_failures = 0;

static llama_model * g_model_llama = nullptr;
static llama_model * g_model_qwen  = nullptr;
static llama_model * g_model_gemma = nullptr;

static void kv_check(bool cond, const char * fmt, ...) {
    if (cond) {
        return;
    }
    ++g_failures;
    fprintf(stderr, "    FAIL: ");
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
}

static void set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    size_t seed = *(const size_t *) userdata;
    std::hash<std::string> hasher;
    seed ^= hasher(tensor->name);
    std::mt19937 gen(seed);
    std::normal_distribution<float> dis(0.0f, 1.0e-2f);

    const int64_t ne = ggml_nelements(tensor);
    if (tensor->type == GGML_TYPE_F32) {
        std::vector<float> tmp(ne);
        for (int64_t i = 0; i < ne; i++) {
            tmp[i] = dis(gen);
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else {
        std::vector<ggml_fp16_t> tmp(ne);
        for (int64_t i = 0; i < ne; i++) {
            tmp[i] = ggml_fp32_to_fp16(dis(gen));
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    }
}

// Minimal in-memory GGUF fixture for the architectures the tests need.
// The loader auto-creates tensors missing from the metadata, so only the
// hyper-parameters matter here.
static gguf_context_ptr make_gguf(const llm_arch arch, const uint32_t nextn) {
    gguf_context_ptr ret(gguf_init_empty());
    llama_model_saver ms(arch, ret.get());

    const uint32_t n_vocab     = 128;
    const uint32_t n_embd      = 256;
    const uint32_t n_head      = 2;
    const uint32_t n_ff        = 384;
    const uint32_t n_layer     = 2;
    const uint32_t n_embd_head = n_embd / n_head;

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,      llm_arch_name(arch));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                n_vocab);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,            uint32_t(128));
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,          n_embd);
    ms.add_kv(LLM_KV_FEATURES_LENGTH,           n_embd);
    ms.add_kv(LLM_KV_BLOCK_COUNT,               n_layer);
    ms.add_kv(LLM_KV_LEADING_DENSE_BLOCK_COUNT, uint32_t(1));
    ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH,       n_ff);
    ms.add_kv(LLM_KV_USE_PARALLEL_RESIDUAL,     false);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT,      n_head);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV,   n_head);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, 1e-5f);
    ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW,  uint32_t(16));
    ms.add_kv(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, uint32_t(2));
    ms.add_kv(LLM_KV_ROPE_DIMENSION_COUNT,      n_embd_head);

    if (arch == LLM_ARCH_QWEN35) {
        // Qwen3.5 is a hybrid arch; used to exercise the MTP context path.
        ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS, std::vector<uint32_t>({n_embd_head/4, n_embd_head/4, n_embd_head/4, n_embd_head/4}));
        ms.add_kv(LLM_KV_SSM_INNER_SIZE,         uint32_t(256));
        ms.add_kv(LLM_KV_SSM_CONV_KERNEL,        uint32_t(4));
        ms.add_kv(LLM_KV_SSM_STATE_SIZE,         uint32_t(16));
        ms.add_kv(LLM_KV_SSM_TIME_STEP_RANK,     n_head);
        ms.add_kv(LLM_KV_SSM_GROUP_COUNT,        uint32_t(1));
        ms.add_kv(LLM_KV_FULL_ATTENTION_INTERVAL, uint32_t(4));
        ms.add_kv(LLM_KV_NEXTN_PREDICT_LAYERS,   nextn);
    }

    ms.add_kv(LLM_KV_TOKENIZER_MODEL, "no_vocab");

    return ret;
}

static llama_model_ptr make_model(const llm_arch arch, const uint32_t nextn) {
    gguf_context_ptr gguf_ctx = make_gguf(arch, nextn);

    llama_model_params params = llama_model_default_params();
    params.load_mtp = true; // needed so the nextn layer tensors are created for MTP contexts

    size_t seed = 42;
    return llama_model_ptr(llama_model_init_from_user(gguf_ctx.get(), set_tensor_data, &seed, params));
}

struct evict_ctx {
    llama_context * ctx = nullptr;
    llama_kv_cache * kv  = nullptr;

    ~evict_ctx() {
        llama_free(ctx);
    }

    evict_ctx() = default;
    evict_ctx(const evict_ctx &) = delete;
    evict_ctx & operator=(const evict_ctx &) = delete;
    evict_ctx(evict_ctx && other) noexcept : ctx(other.ctx), kv(other.kv) {
        other.ctx = nullptr;
        other.kv  = nullptr;
    }
    evict_ctx & operator=(evict_ctx && other) noexcept {
        if (this != &other) {
            llama_free(ctx);
            ctx = other.ctx;
            kv  = other.kv;
            other.ctx = nullptr;
            other.kv  = nullptr;
        }
        return *this;
    }
};

static evict_ctx make_evict_ctx(
        llama_model * model,
        const uint32_t sink,
        const uint32_t recent,
        const uint32_t n_ubatch,
        const uint32_t n_seq_max,
        const bool     unified,
        const llama_context_type ctype = LLAMA_CONTEXT_TYPE_DEFAULT) {
    llama_context_params params = llama_context_default_params();
    params.n_ctx            = 256;
    params.n_batch          = 256;
    params.n_ubatch         = n_ubatch;
    params.n_kv_sink        = sink;
    params.n_kv_recent      = recent;
    params.kv_unified       = unified;
    params.n_seq_max        = n_seq_max;
    params.ctx_type         = ctype;
    params.n_threads        = 4;
    params.n_threads_batch  = 4;

    evict_ctx res;
    res.ctx = llama_init_from_model(model, params);
    if (res.ctx == nullptr) {
        return res;
    }
    res.kv = dynamic_cast<llama_kv_cache *>(llama_get_memory(res.ctx));
    return res;
}

// decode n tokens at consecutive positions [pos0, pos0 + n) for the given
// sequence. with n_seq_id == 2 the tokens are shared by sequences 0 and 1.
static bool decode_range(
        llama_context * ctx,
        const int       n_vocab,
        const int       pos0,
        const int       n,
        const llama_seq_id seq_id,
        const int       n_seq_id = 1) {
    llama_batch batch = llama_batch_init(n, 0, n_seq_id > 1 ? 2 : 1);
    for (int i = 0; i < n; i++) {
        const llama_token tok = (pos0 + i) % n_vocab;
        if (n_seq_id == 1) {
            common_batch_add(batch, tok, pos0 + i, { seq_id }, i == n - 1);
        } else {
            common_batch_add(batch, tok, pos0 + i, { 0, 1 }, i == n - 1);
        }
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

static std::set<llama_pos> present_positions(const llama_kv_cache * kv, const llama_seq_id seq_id) {
    std::set<llama_pos> res;
    const auto & cells = kv->get_cells(seq_id);
    for (uint32_t i = 0; i < cells.size(); i++) {
        if (cells.is_empty(i)) {
            continue;
        }
        if (!cells.seq_has(i, seq_id)) {
            continue;
        }
        res.insert(cells.pos_get(i));
    }
    return res;
}

// check that every sink position and the full recent window [p1 - recent, p1]
// are still present in the cache. p1 is the maximum decoded position.
static bool check_sink_recent(
        const llama_kv_cache * kv,
        const llama_seq_id     seq_id,
        const uint32_t         sink,
        const uint32_t         recent,
        const llama_pos        p1) {
    const auto present = present_positions(kv, seq_id);

    // sink cells fill up as the first sink positions are decoded; only require
    // the ones that have been written so far
    for (llama_pos p = 0; p < (llama_pos) sink && p <= p1; p++) {
        if (!present.count(p)) {
            return false;
        }
    }

    const llama_pos lo = std::max<llama_pos>(0, p1 - (llama_pos) recent);
    for (llama_pos p = lo; p <= p1; p++) {
        if (!present.count(p)) {
            return false;
        }
    }
    return true;
}

// create an eviction context and check its memory is a llama_kv_cache
static bool make_evict_ctx_ready(
        evict_ctx &    ec,
        llama_model *  model,
        const uint32_t sink,
        const uint32_t recent,
        const uint32_t n_ubatch,
        const uint32_t n_seq_max,
        const bool     unified,
        const char *   what,
        const llama_context_type ctype = LLAMA_CONTEXT_TYPE_DEFAULT,
        const bool     need_kv = true) {
    ec = make_evict_ctx(model, sink, recent, n_ubatch, n_seq_max, unified, ctype);
    kv_check(ec.ctx != nullptr, "%s: failed to create context", what);
    if (need_kv) {
        kv_check(ec.kv  != nullptr, "%s: memory is not a llama_kv_cache", what);
    }
    return ec.ctx != nullptr && (!need_kv || ec.kv != nullptr);
}

// decode n single tokens at positions [pos0, pos0 + n), checking the
// sink/recent invariant after every step
static bool decode_invariant_run(
        llama_context * ctx,
        llama_kv_cache * kv,
        const int       n_vocab,
        const int       pos0,
        const int       n,
        const uint32_t  sink,
        const uint32_t  recent) {
    const int failures0 = g_failures;
    for (int pos = pos0; pos < pos0 + n; pos++) {
        kv_check(decode_range(ctx, n_vocab, pos, 1, 0), "decode failed at pos %d", pos);
        kv_check(check_sink_recent(kv, 0, sink, recent, pos),
                "sink/recent invariant broken at position %d", pos);
        if (g_failures != failures0) {
            return false;
        }
    }
    return true;
}

// case 1: sink cells are never recycled; content stays visible
static void test_sink_preservation() {
    const uint32_t sink = 4, recent = 16;
    const int n_vocab = 128;

    evict_ctx ec;
    if (!make_evict_ctx_ready(ec, g_model_llama, sink, recent, 4, 1, true, "sink preservation")) {
        return;
    }

    const int failures0 = g_failures;
    for (int pos = 0; pos < 300; pos += 4) {
        const int n = std::min(4, 300 - pos);
        kv_check(decode_range(ec.ctx, n_vocab, pos, n, 0), "decode failed at pos %d", pos);
        if (g_failures != failures0) {
            break;
        }
    }

    const auto present = present_positions(ec.kv, 0);
    for (llama_pos p = 0; p < (llama_pos) sink; p++) {
        kv_check(present.count(p) != 0, "sink cell at position %" PRId32 " was recycled", p);
    }
    kv_check(ec.kv->get_evict_stats().cells_recycled() > 0, "expected cells to be recycled");
    kv_check(check_sink_recent(ec.kv, 0, sink, recent, 299), "recent window not fully present at p1 = 299");
}

// case 2: cells inside the recent window of the current position are never
// recycled; a recycled victim is always outside [p1 - recent, p1)
static void test_recent_window_retention() {
    const uint32_t sink = 4, recent = 16;
    const int n_vocab = 128;

    evict_ctx ec;
    if (!make_evict_ctx_ready(ec, g_model_llama, sink, recent, 4, 1, true, "recent-window retention")) {
        return;
    }

    // single-token decode; sink and recent window must stay intact after each step
    if (!decode_invariant_run(ec.ctx, ec.kv, n_vocab, 0, 250, sink, recent)) {
        return;
    }

    kv_check(ec.kv->get_evict_stats().cells_recycled() > 0, "expected cells to be recycled");
}

// case 3: long synthetic run - bounded physical cache, monotonic positions,
// recycling and masking both active, consistent generation (no crash)
static void test_multiple_eviction_cycles() {
    const uint32_t sink = 4, recent = 16, n_ubatch = 4;
    const int n_vocab = 128;

    evict_ctx ec;
    if (!make_evict_ctx_ready(ec, g_model_llama, sink, recent, n_ubatch, 1, true, "multiple eviction cycles")) {
        return;
    }

    kv_check(ec.kv->get_size() == sink + recent + n_ubatch,
            "physical cache size %u, expected %u", ec.kv->get_size(), sink + recent + n_ubatch);

    int n_past = 0;
    const int n_total = 600;
    const int failures0 = g_failures;
    for (int pos = 0; pos < n_total; pos += (int) n_ubatch) {
        const int n = std::min((int) n_ubatch, n_total - pos);
        kv_check(decode_range(ec.ctx, n_vocab, pos, n, 0), "decode failed at pos %d", pos);
        if (g_failures != failures0) {
            break;
        }
        n_past += n;
    }

    kv_check(n_past == n_total, "n_past = %d, expected %d", n_past, n_total);

    // physical cache size stays bounded
    const auto & cells = ec.kv->get_cells(0);
    uint32_t n_used = 0;
    for (uint32_t i = 0; i < cells.size(); i++) {
        if (!cells.is_empty(i)) {
            n_used++;
        }
    }
    kv_check(n_used <= ec.kv->get_size(), "used cells %u exceed cache size %u", n_used, ec.kv->get_size());

    const auto stats = ec.kv->get_evict_stats();
    kv_check(stats.cells_recycled() > 0, "expected cells to be recycled");
    kv_check(stats.masked_cells()     > 0, "expected cells to be masked");
    kv_check(stats.victim_failures()  == 0, "expected no victim selection failures, got %llu",
            (unsigned long long) stats.victim_failures());

    // positions stay monotonic: the max cached position equals the last decoded one
    kv_check(cells.seq_pos_max(0) == n_total - 1, "seq_pos_max = %" PRId32 ", expected %d",
            cells.seq_pos_max(0), n_total - 1);
    kv_check(check_sink_recent(ec.kv, 0, sink, recent, n_total - 1),
            "sink/recent invariant broken after %d tokens", n_total);
}

// case 4: MTP disables eviction when n_kv_recent < 64, keeps it otherwise
static void test_mtp_disable() {
    // plain (non-MTP) context is the control case: eviction must stay on
    {
        evict_ctx ec;
        if (!make_evict_ctx_ready(ec, g_model_llama, 4, 32, 4, 1, true, "control")) {
            return;
        }
        kv_check(ec.ctx->get_cparams().n_kv_sink   == 4, "control ctx: sink disabled");
        kv_check(ec.ctx->get_cparams().n_kv_recent == 32, "control ctx: recent disabled");
    }

    // MTP with recent < 64 -> disabled
    {
        evict_ctx ec;
        if (!make_evict_ctx_ready(ec, g_model_qwen, 4, 32, 4, 1, true, "MTP (recent 32)", LLAMA_CONTEXT_TYPE_MTP)) {
            return;
        }
        kv_check(ec.ctx->get_cparams().n_kv_sink   == 0, "MTP (recent 32): sink not zeroed");
        kv_check(ec.ctx->get_cparams().n_kv_recent == 0, "MTP (recent 32): recent not zeroed");
    }

    // MTP with recent >= 64 -> eviction stays on and actually runs
    {
        evict_ctx ec;
        if (!make_evict_ctx_ready(ec, g_model_qwen, 4, 64, 4, 1, true, "MTP (recent 64)", LLAMA_CONTEXT_TYPE_MTP)) {
            return;
        }
        kv_check(ec.ctx->get_cparams().n_kv_sink   == 4, "MTP (recent 64): sink disabled");
        kv_check(ec.ctx->get_cparams().n_kv_recent == 64, "MTP (recent 64): recent disabled");

        bool ok = true;
        for (int pos = 0; pos < 200; pos += 4) {
            ok = ok && decode_range(ec.ctx, 128, pos, std::min(4, 200 - pos), 0);
        }
        kv_check(ok, "MTP (recent 64): decode failed");
        kv_check(ec.kv->get_evict_stats().cells_recycled() > 0, "MTP (recent 64): no recycling");
    }
}

// case 5: SWA (swa_type != NONE) always disables eviction
static void test_swa_disable() {
    // gemma2 is a sliding-window model (swa_type == STANDARD)
    for (const uint32_t recent : { uint32_t(16), uint32_t(64) }) {
        evict_ctx ec;
        if (!make_evict_ctx_ready(ec, g_model_gemma, 4, recent, 4, 1, true, "SWA", LLAMA_CONTEXT_TYPE_DEFAULT, false)) {
            return;
        }
        kv_check(ec.ctx->get_cparams().n_kv_sink   == 0, "SWA (recent %u): sink not zeroed", recent);
        kv_check(ec.ctx->get_cparams().n_kv_recent == 0, "SWA (recent %u): recent not zeroed", recent);
    }
}

// case 6: state write/read round-trip after eviction has fragmented the cache
static void test_state_roundtrip() {
    const uint32_t sink = 4, recent = 16;
    const int n_vocab = 128;

    evict_ctx src;
    if (!make_evict_ctx_ready(src, g_model_llama, sink, recent, 4, 1, true, "state save")) {
        return;
    }

    const int failures0 = g_failures;

    for (int pos = 0; pos < 300; pos += 4) {
        kv_check(decode_range(src.ctx, n_vocab, pos, std::min(4, 300 - pos), 0),
                "source decode failed at pos %d", pos);
        if (g_failures != failures0) {
            return;
        }
    }

    const std::set<llama_pos> positions_before = present_positions(src.kv, 0);
    kv_check(src.kv->get_evict_stats().cells_recycled() > 0, "expected recycling before state save");
    kv_check(check_sink_recent(src.kv, 0, sink, recent, 299), "source cache invariant broken before save");

    // serialize the whole state
    std::vector<uint8_t> state(llama_state_get_size(src.ctx));
    const size_t n_save = llama_state_get_data(src.ctx, state.data(), state.size());
    kv_check(n_save == state.size(), "state save size %zu, expected %zu", n_save, state.size());
    if (n_save != state.size()) {
        return;
    }

    // restore into a fresh context
    evict_ctx dst;
    if (!make_evict_ctx_ready(dst, g_model_llama, sink, recent, 4, 1, true, "state load")) {
        return;
    }

    const size_t n_load = llama_state_set_data(dst.ctx, state.data(), state.size());
    kv_check(n_load == state.size(), "state load size %zu, expected %zu", n_load, state.size());

    // positions and visibility preserved across the round-trip
    const std::set<llama_pos> positions_after = present_positions(dst.kv, 0);
    kv_check(positions_after == positions_before,
            "state round-trip changed cached positions (%zu before, %zu after)",
            positions_before.size(), positions_after.size());
    kv_check(check_sink_recent(dst.kv, 0, sink, recent, 299),
            "reloaded cache is missing sink/recent positions");

    // the reloaded cache keeps working: decode continues consistently
    bool ok = true;
    for (int pos = 300; pos < 400; pos += 4) {
        ok = ok && decode_range(dst.ctx, n_vocab, pos, std::min(4, 400 - pos), 0);
    }
    kv_check(ok, "decode failed after state restore");
    kv_check(check_sink_recent(dst.kv, 0, sink, recent, 399), "invariant broken after state restore");
}

// case 7: multi-sequence - recycling only touches cells owned by the current
// sequence; shared cells are never recycled; unified sizing accounts for n_seq_max
static void test_multi_sequence() {
    const uint32_t sink = 4, recent = 16, n_ubatch = 4, n_seq_max = 2;
    const int n_vocab = 128;

    evict_ctx ec;
    if (!make_evict_ctx_ready(ec, g_model_llama, sink, recent, n_ubatch, n_seq_max, true, "multi-sequence")) {
        return;
    }

    // physical cache sized per-sequence: n_seq_max * (sink + recent) + ubatch
    kv_check(ec.kv->get_size() == n_seq_max*(sink + recent) + n_ubatch,
            "physical cache size %u, expected %u", ec.kv->get_size(), n_seq_max*(sink + recent) + n_ubatch);

    // shared prefix: tokens 0..20 decoded for both sequences at once
    kv_check(decode_range(ec.ctx, n_vocab, 0, 21, 0, 2), "shared prefix decode failed");

    // sequence 0 alone outruns the cache, forcing recycling of its own cells
    bool ok = true;
    for (int pos = 21; pos < 300; pos++) {
        ok = ok && decode_range(ec.ctx, n_vocab, pos, 1, 0);
    }
    kv_check(ok, "sequence 0 decode failed");
    if (!ok) {
        return;
    }

    const auto present_0 = present_positions(ec.kv, 0);
    const auto present_1 = present_positions(ec.kv, 1);

    // shared prefix never recycled: every position 0..20 still visible to both sequences
    for (llama_pos p = 0; p <= 20; p++) {
        kv_check(present_0.count(p) != 0, "shared cell at position %" PRId32 " lost for sequence 0", p);
        kv_check(present_1.count(p) != 0, "shared cell at position %" PRId32 " lost for sequence 1", p);
    }

    // sequence 0 recent window intact
    kv_check(check_sink_recent(ec.kv, 0, sink, recent, 299), "sequence 0 recent window broken");

    // recycling only touched sequence-0-only cells: no cell is owned solely by
    // sequence 1, and every single-owner cell belongs to sequence 0
    const auto & cells = ec.kv->get_cells(0);
    for (uint32_t i = 0; i < cells.size(); i++) {
        if (cells.is_empty(i)) {
            continue;
        }
        kv_check(cells.seq_has(i, 0) || cells.seq_has(i, 1), "cell %u belongs to no active sequence", i);
        if (cells.seq_count(i) == 1) {
            kv_check(cells.seq_has(i, 0), "single-owner cell %u does not belong to sequence 0", i);
        }
    }

    kv_check(ec.kv->get_evict_stats().cells_recycled() > 0, "expected sequence 0 cells to be recycled");
}

// case 8: a ubatch spanning the recent-window boundary must not recycle cells
// that an earlier token in the same ubatch still needs
static void test_ubatch_boundary() {
    const uint32_t sink = 4, recent = 16, n_ubatch = 8;
    const int n_vocab = 128;

    evict_ctx ec;
    if (!make_evict_ctx_ready(ec, g_model_llama, sink, recent, n_ubatch, 1, true, "ubatch boundary")) {
        return;
    }

    kv_check(ec.kv->get_size() == sink + recent + n_ubatch,
            "physical cache size %u, expected %u", ec.kv->get_size(), sink + recent + n_ubatch);

    // fill the cache with single-token decodes so a later ubatch must recycle
    const int failures0 = g_failures;
    for (int pos = 0; pos < 200; pos++) {
        kv_check(decode_range(ec.ctx, n_vocab, pos, 1, 0), "single decode failed at pos %d", pos);
        if (g_failures != failures0) {
            return;
        }
    }

    // several full ubatches whose positions cross the recent-window boundary:
    // the first token of each batch sits at p0 and needs [p0 - recent, p0];
    // the recycling threshold must be derived from that first token
    for (int p0 = 200; p0 < 320; p0 += (int) n_ubatch) {
        kv_check(decode_range(ec.ctx, n_vocab, p0, (int) n_ubatch, 0),
                "ubatch decode failed starting at pos %d", p0);
        if (g_failures != failures0) {
            return;
        }

        // the whole span [p0 - recent, p0 + n_ubatch - 1] must be present:
        // cells needed by earlier tokens of the batch were never recycled
        const auto present = present_positions(ec.kv, 0);
        for (llama_pos p = p0 - (llama_pos) recent; p <= p0 + (llama_pos) n_ubatch - 1; p++) {
            kv_check(present.count(p) != 0,
                    "position %" PRId32 " (needed by the ubatch at %d) was recycled", p, p0);
            if (g_failures != failures0) {
                return;
            }
        }
    }

    kv_check(ec.kv->get_evict_stats().cells_recycled() > 0, "expected cells to be recycled");
    kv_check(ec.kv->get_evict_stats().victim_failures() == 0, "expected no victim selection failures");
}

// case 9: quantized K/V cache runs the full eviction cycle without misparsing.
// the cache-type selection is exposed on llama_context_params (type_k/type_v),
// the -ctk/-ctv CLI equivalents
static void test_quantized_kv() {
    const uint32_t sink = 4, recent = 16;
    const int n_vocab = 128;

    evict_ctx ec;
    {
        llama_context_params params = llama_context_default_params();
        params.n_ctx            = 256;
        params.n_batch          = 256;
        params.n_ubatch         = 4;
        params.n_kv_sink        = sink;
        params.n_kv_recent      = recent;
        params.kv_unified       = true;
        params.n_seq_max        = 1;
        params.type_k           = GGML_TYPE_Q4_0;
        params.type_v           = GGML_TYPE_Q4_0;
        params.n_threads        = 4;
        params.n_threads_batch  = 4;
        ec.ctx = llama_init_from_model(g_model_llama, params);
    }
    kv_check(ec.ctx != nullptr, "failed to create quantized KV context");
    if (ec.ctx == nullptr) {
        return;
    }
    ec.kv = dynamic_cast<llama_kv_cache *>(llama_get_memory(ec.ctx));
    kv_check(ec.kv != nullptr, "quantized context memory is not a llama_kv_cache");
    if (ec.kv == nullptr) {
        return;
    }

    kv_check(ec.kv->type_k() == GGML_TYPE_Q4_0, "K cache type %d, expected q4_0", (int) ec.kv->type_k());
    kv_check(ec.kv->type_v() == GGML_TYPE_Q4_0, "V cache type %d, expected q4_0", (int) ec.kv->type_v());

    if (!decode_invariant_run(ec.ctx, ec.kv, n_vocab, 0, 300, sink, recent)) {
        return;
    }

    kv_check(ec.kv->get_evict_stats().cells_recycled() > 0, "expected cells to be recycled");
    kv_check(ec.kv->get_evict_stats().victim_failures() == 0, "unexpected victim selection failure");
}

// snapshot of (position, sequence-membership) per physical cell of a stream
struct kv_cell_state {
    std::vector<llama_pos> pos;
    std::vector<std::bitset<LLAMA_MAX_SEQ>> seq;
};

static kv_cell_state kv_snapshot(const llama_kv_cache * kv, const llama_seq_id seq_id) {
    kv_cell_state s;
    const auto & cells = kv->get_cells(seq_id);
    s.pos.resize(cells.size(), -1);
    s.seq.resize(cells.size());
    for (uint32_t i = 0; i < cells.size(); i++) {
        if (cells.is_empty(i)) {
            continue;
        }
        s.pos[i] = cells.pos_get(i);
        for (llama_seq_id q = 0; q < LLAMA_MAX_SEQ; q++) {
            if (cells.seq_has(i, q)) {
                s.seq[i].set(q);
            }
        }
    }
    return s;
}

// physical cells whose content changed since the snapshot (i.e. written by the
// decode in between). one ubatch token must write exactly one distinct cell
static std::vector<uint32_t> kv_written_cells(
        const llama_kv_cache * kv,
        const llama_seq_id     seq_id,
        const kv_cell_state &  before) {
    std::vector<uint32_t> res;
    const auto & cells = kv->get_cells(seq_id);
    for (uint32_t i = 0; i < cells.size(); i++) {
        llama_pos cur_pos = -1;
        std::bitset<LLAMA_MAX_SEQ> cur_seq;
        if (!cells.is_empty(i)) {
            cur_pos = cells.pos_get(i);
            for (llama_seq_id q = 0; q < LLAMA_MAX_SEQ; q++) {
                if (cells.seq_has(i, q)) {
                    cur_seq.set(q);
                }
            }
        }
        if (cur_pos != before.pos[i] || cur_seq != before.seq[i]) {
            res.push_back(i);
        }
    }
    return res;
}

// prove that each expected position for a sequence maps to a DISTINCT physical
// cell. a duplicate assignment (two tokens of one ubatch sharing a cell) would
// clobber one write: either the position goes missing entirely, or two positions
// share one cell. both are caught here.
static bool kv_assert_distinct_cells(
        const llama_kv_cache *            kv,
        const llama_seq_id                seq_id,
        const std::vector<llama_pos> &    positions) {
    const auto & cells = kv->get_cells(seq_id);
    std::unordered_map<llama_pos, uint32_t> pos_to_cell;
    for (uint32_t i = 0; i < cells.size(); i++) {
        if (cells.is_empty(i) || !cells.seq_has(i, seq_id)) {
            continue;
        }
        pos_to_cell[cells.pos_get(i)] = i;
    }
    std::unordered_set<uint32_t> used_cells;
    for (const llama_pos p : positions) {
        const auto it = pos_to_cell.find(p);
        if (it == pos_to_cell.end()) {
            return false; // a token's write was clobbered (position missing)
        }
        if (!used_cells.insert(it->second).second) {
            return false; // two positions share one physical cell
        }
    }
    return true;
}

// decode a single ubatch whose tokens carry explicit (position, seq_id) pairs
static bool decode_ubatch(
        llama_context * ctx,
        const int       n_vocab,
        const std::vector<std::pair<llama_pos, llama_seq_id>> & tokens) {
    llama_batch batch = llama_batch_init(tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); i++) {
        common_batch_add(batch, tokens[i].first % n_vocab, tokens[i].first, { tokens[i].second }, false);
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

// case 10: unified vs non-unified multi-sequence sizing. the unified cache must
// size its single physical pool to n_seq_max*(sink + recent) + ubatch and never
// hand the same physical cell to two sequences of the same ubatch; the
// non-unified cache keeps one per-sequence pool of sink + recent + ubatch
static void test_unified_nonunified_multi_seq() {
    const uint32_t sink = 4, recent = 16, n_seq_max = 2;
    const int n_vocab = 128;

    // unified: one shared pool, simultaneous eviction in one ubatch
    {
        const uint32_t n_ubatch = 8;
        evict_ctx ec;
        if (!make_evict_ctx_ready(ec, g_model_llama, sink, recent, n_ubatch, n_seq_max, true, "unified multi-seq")) {
            return;
        }

        kv_check(ec.kv->get_n_stream() == 1, "unified cache has %u streams, expected 1", ec.kv->get_n_stream());
        kv_check(ec.kv->get_size() == n_seq_max*(sink + recent) + n_ubatch,
                "unified physical size %u, expected %u", ec.kv->get_size(), n_seq_max*(sink + recent) + n_ubatch);

        const int failures0 = g_failures;
        bool ok = true;
        for (int pos = 0; pos < 200; pos++) {
            ok = ok && decode_range(ec.ctx, n_vocab, pos, 1, 0);
            ok = ok && decode_range(ec.ctx, n_vocab, pos, 1, 1);
        }
        kv_check(ok, "unified interleaved decode failed");
        if (g_failures != failures0) {
            return;
        }

        // sequence 0 alone pushes the pool to capacity so the mix below recycles
        for (int pos = 200; pos < 204; pos++) {
            kv_check(decode_range(ec.ctx, n_vocab, pos, 1, 0), "seq 0 fill failed at pos %d", pos);
        }
        if (g_failures != failures0) {
            return;
        }

        kv_cell_state before = kv_snapshot(ec.kv, 0);
        std::vector<std::pair<llama_pos, llama_seq_id>> toks;
        for (int i = 0; i < 2; i++) { toks.push_back({204 + i, 0}); }
        for (int i = 0; i < 2; i++) { toks.push_back({200 + i, 1}); }
        kv_check(decode_ubatch(ec.ctx, n_vocab, toks), "unified simultaneous ubatch decode failed");
        const auto written = kv_written_cells(ec.kv, 0, before);
        kv_check(written.size() == toks.size(),
                "unified ubatch assigned %zu distinct cells to %zu tokens (duplicate cell)", written.size(), toks.size());
    }

    // non-unified: per-sequence pools
    {
        const uint32_t n_ubatch = 4;
        evict_ctx ec;
        if (!make_evict_ctx_ready(ec, g_model_llama, sink, recent, n_ubatch, n_seq_max, false, "non-unified multi-seq")) {
            return;
        }

        kv_check(ec.kv->get_n_stream() == n_seq_max, "non-unified cache has %u streams, expected %u",
                ec.kv->get_n_stream(), n_seq_max);
        kv_check(ec.kv->get_size() == sink + recent + n_ubatch,
                "non-unified physical size %u, expected %u", ec.kv->get_size(), sink + recent + n_ubatch);

        const int failures0 = g_failures;
        bool ok = true;
        for (int pos = 0; pos < 200; pos++) {
            ok = ok && decode_range(ec.ctx, n_vocab, pos, 1, 0);
            ok = ok && decode_range(ec.ctx, n_vocab, pos, 1, 1);
        }
        kv_check(ok, "non-unified interleaved decode failed");
        if (g_failures != failures0) {
            return;
        }

        kv_cell_state before0 = kv_snapshot(ec.kv, 0);
        kv_cell_state before1 = kv_snapshot(ec.kv, 1);
        std::vector<std::pair<llama_pos, llama_seq_id>> toks;
        for (int i = 0; i < 2; i++) { toks.push_back({200 + i, 0}); }
        for (int i = 0; i < 2; i++) { toks.push_back({200 + i, 1}); }
        kv_check(decode_ubatch(ec.ctx, n_vocab, toks), "non-unified simultaneous ubatch decode failed");
        const auto w0 = kv_written_cells(ec.kv, 0, before0);
        const auto w1 = kv_written_cells(ec.kv, 1, before1);
        kv_check(w0.size() == 2, "non-unified seq 0 wrote %zu cells for 2 tokens", w0.size());
        kv_check(w1.size() == 2, "non-unified seq 1 wrote %zu cells for 2 tokens", w1.size());
        kv_check(check_sink_recent(ec.kv, 0, sink, recent, 201), "non-unified seq 0 window broken");
        kv_check(check_sink_recent(ec.kv, 1, sink, recent, 201), "non-unified seq 1 window broken");
    }
}

// case 11: core regression - two sequences whose positions both trigger eviction
// in the same unified ubatch must receive distinct physical cells and keep their
// sink + recent windows intact
static void test_simultaneous_multi_seq_eviction() {
    const uint32_t sink = 4, recent = 16, n_ubatch = 8, n_seq_max = 2;
    const int n_vocab = 128;

    evict_ctx ec;
    if (!make_evict_ctx_ready(ec, g_model_llama, sink, recent, n_ubatch, n_seq_max, true, "simultaneous multi-seq eviction")) {
        return;
    }

    kv_check(ec.kv->get_size() == n_seq_max*(sink + recent) + n_ubatch,
            "physical size %u, expected %u", ec.kv->get_size(), n_seq_max*(sink + recent) + n_ubatch);

    const int failures0 = g_failures;

    // interleaved runs give both sequences their own cells in the shared pool
    bool ok = true;
    for (int pos = 0; pos < 200; pos++) {
        ok = ok && decode_range(ec.ctx, n_vocab, pos, 1, 0);
        ok = ok && decode_range(ec.ctx, n_vocab, pos, 1, 1);
    }
    kv_check(ok, "interleaved decode failed");
    if (g_failures != failures0) {
        return;
    }

    // sequence 0 alone pushes the pool to capacity, leaving its masked-middle
    // cells (positions just below its recent window) as the only recyclable ones
    for (int pos = 200; pos < 204; pos++) {
        kv_check(decode_range(ec.ctx, n_vocab, pos, 1, 0), "seq 0 fill failed at pos %d", pos);
    }
    if (g_failures != failures0) {
        return;
    }

    // one ubatch where both sequences trigger eviction at the same time
    const uint64_t recycled_before = ec.kv->get_evict_stats().cells_recycled();
    kv_cell_state before = kv_snapshot(ec.kv, 0);
    std::vector<std::pair<llama_pos, llama_seq_id>> toks;
    for (int i = 0; i < 2; i++) { toks.push_back({204 + i, 0}); }
    for (int i = 0; i < 2; i++) { toks.push_back({200 + i, 1}); }
    kv_check(decode_ubatch(ec.ctx, n_vocab, toks), "simultaneous eviction ubatch failed");

    // regression: the recycle scan must not hand the same cell to two tokens
    const auto written = kv_written_cells(ec.kv, 0, before);
    kv_check(written.size() == toks.size(),
            "simultaneous ubatch assigned %zu distinct cells to %zu tokens", written.size(), toks.size());
    // direct proof: every written position maps to a distinct physical cell
    kv_check(kv_assert_distinct_cells(ec.kv, 0, { 204, 205 }),
            "seq 0: two tokens in one ubatch share a physical cell (duplicate assignment)");
    kv_check(kv_assert_distinct_cells(ec.kv, 1, { 200, 201 }),
            "seq 1: two tokens in one ubatch share a physical cell (duplicate assignment)");

    // eviction actually ran inside the batch
    kv_check(ec.kv->get_evict_stats().cells_recycled() > recycled_before,
            "expected the ubatch to recycle cells");

    // both sequences keep their full sink + recent windows
    kv_check(check_sink_recent(ec.kv, 0, sink, recent, 205), "seq 0 window broken after simultaneous eviction");
    kv_check(check_sink_recent(ec.kv, 1, sink, recent, 201), "seq 1 window broken after simultaneous eviction");
}

// case 12: a long shared prefix can exhaust the pool so that no single-owner
// cell and no empty cell remains for a diverging sequence. the decode must fail
// explicitly (empty slot, WARN logged) instead of corrupting the existing state
static void test_shared_cell_exhaustion() {
    const uint32_t sink = 4, recent = 16, n_ubatch = 4, n_seq_max = 2;
    const int n_vocab = 128;

    evict_ctx ec;
    if (!make_evict_ctx_ready(ec, g_model_llama, sink, recent, n_ubatch, n_seq_max, true, "shared-cell exhaustion")) {
        return;
    }

    // shared prefix: every prefix cell is owned by both sequences and so can
    // never be recycled (recycling only touches single-owner cells)
    kv_check(decode_range(ec.ctx, n_vocab, 0, 21, 0, 2), "shared prefix decode failed");
    if (g_failures) {
        return;
    }

    // sequence 0 outruns the pool and starts recycling its own cells
    const int failures0 = g_failures;
    bool ok = true;
    for (int pos = 21; pos < 300; pos++) {
        ok = ok && decode_range(ec.ctx, n_vocab, pos, 1, 0);
    }
    kv_check(ok, "seq 0 run failed");
    if (g_failures != failures0) {
        return;
    }
    kv_check(ec.kv->get_evict_stats().cells_recycled() > 0, "expected seq 0 cells to be recycled");

    // the diverging sequence has no private cells and the pool is full: the
    // eviction must surface the explicit empty-slot failure (WARN) rather than
    // corrupting the windows
    const uint64_t victim_fail_before = ec.kv->get_evict_stats().victim_failures();
    const bool diverged = decode_range(ec.ctx, n_vocab, 21, 1, 1);
    kv_check(!diverged, "diverging sequence unexpectedly decoded");
    kv_check(ec.kv->get_evict_stats().victim_failures() > victim_fail_before,
            "expected a victim-selection failure (WARN) for the diverging sequence");

    // state is intact: seq 0 keeps its window and no cell lost its owner
    kv_check(check_sink_recent(ec.kv, 0, sink, recent, 299), "seq 0 window corrupted by failed divergence");
    const auto & cells = ec.kv->get_cells(0);
    bool consistent = true;
    for (uint32_t i = 0; i < cells.size(); i++) {
        if (!cells.is_empty(i) && cells.seq_count(i) == 0) {
            consistent = false;
        }
    }
    kv_check(consistent, "a non-empty cell lost its sequence membership");
}

// case 13: MTP context under eviction survives a forced rollback of the tail.
// a true speculative accept/reject cannot be forced through the public context
// API, so this tests the closest reachable path: remove the rejected tail with
// llama_memory_seq_rm and re-decode the same positions (as a verified token)
static void test_mtp_rollback_eviction() {
    const uint32_t sink = 4, recent = 64, n_ubatch = 8;
    const int n_vocab = 128;

    evict_ctx ec;
    if (!make_evict_ctx_ready(ec, g_model_qwen, sink, recent, n_ubatch, 1, true, "MTP rollback", LLAMA_CONTEXT_TYPE_MTP)) {
        return;
    }
    kv_check(ec.ctx->get_cparams().n_kv_recent == recent, "MTP eviction not enabled (recent %u)",
            ec.ctx->get_cparams().n_kv_recent);

    const int failures0 = g_failures;

    // long run: recycling is active inside the MTP context
    bool ok = true;
    for (int pos = 0; pos < 400; pos += (int) n_ubatch) {
        ok = ok && decode_range(ec.ctx, n_vocab, pos, std::min((int) n_ubatch, 400 - pos), 0);
    }
    kv_check(ok, "MTP decode failed");
    kv_check(ec.kv->get_evict_stats().cells_recycled() > 0, "MTP context did not recycle");
    if (g_failures != failures0) {
        return;
    }

    // force a rejection: roll the sequence back a few positions, then re-decode
    const llama_pos rb = 380;
    llama_memory_seq_rm(llama_get_memory(ec.ctx), 0, rb, -1);

    ok = true;
    for (int pos = rb; pos < 400; pos += 4) {
        ok = ok && decode_range(ec.ctx, n_vocab, pos, std::min(4, 400 - pos), 0);
    }
    kv_check(ok, "MTP re-decode after rollback failed");
    kv_check(ec.kv->get_cells(0).seq_pos_max(0) == 399, "seq_pos_max after rollback = %" PRId32 ", expected 399",
            ec.kv->get_cells(0).seq_pos_max(0));
    kv_check(check_sink_recent(ec.kv, 0, sink, recent, 399), "MTP window broken after rollback");
}

// case 14: boundary values for the sink/recent sizing
static void test_boundary_values() {
    const int n_vocab = 128;

    // sink == 1, recent == 1
    {
        evict_ctx ec;
        if (!make_evict_ctx_ready(ec, g_model_llama, 1, 1, 4, 1, true, "sink=1/recent=1")) {
            return;
        }
        if (!decode_invariant_run(ec.ctx, ec.kv, n_vocab, 0, 100, 1, 1)) {
            return;
        }
        kv_check(ec.kv->get_evict_stats().cells_recycled() > 0, "no recycling with sink=1/recent=1");
    }

    // recent < n_ubatch: a full ubatch spans several recent windows
    {
        evict_ctx ec;
        if (!make_evict_ctx_ready(ec, g_model_llama, 4, 2, 8, 1, true, "recent < n_ubatch")) {
            return;
        }
        if (!decode_invariant_run(ec.ctx, ec.kv, n_vocab, 0, 100, 4, 2)) {
            return;
        }
        kv_check(ec.kv->get_evict_stats().cells_recycled() > 0, "no recycling with recent < n_ubatch");
    }

    // sink + recent == n_ctx_seq: the full logical window fits exactly
    {
        evict_ctx ec;
        if (!make_evict_ctx_ready(ec, g_model_llama, 8, 248, 4, 1, true, "sink+recent == n_ctx_seq")) {
            return;
        }
        bool ok = true;
        for (int pos = 0; pos < 400; pos += 4) {
            ok = ok && decode_range(ec.ctx, n_vocab, pos, std::min(4, 400 - pos), 0);
        }
        kv_check(ok, "decode failed with sink+recent == n_ctx_seq");
        kv_check(check_sink_recent(ec.kv, 0, 8, 248, 399), "window broken with sink+recent == n_ctx_seq");
        kv_check(ec.kv->get_evict_stats().cells_recycled() > 0, "expected recycling with sink+recent == n_ctx_seq");
    }

    // window_start clamps to 0 while p1 < recent: the whole cache is in the recent window
    {
        evict_ctx ec;
        if (!make_evict_ctx_ready(ec, g_model_llama, 4, 16, 4, 1, true, "window clamp")) {
            return;
        }
        if (!decode_invariant_run(ec.ctx, ec.kv, n_vocab, 0, 40, 4, 16)) {
            return;
        }
        kv_check(ec.kv->get_evict_stats().cells_recycled() > 0, "no recycling in the window-clamp case");
    }
}

// case 15: a sparse ubatch (positions with holes) crossing the recent boundary.
// only M-RoPE models accept non-contiguous positions, so this runs on the
// Qwen3.5 MTP context (the smallest M-RoPE fixture available)
static void test_sparse_ubatch() {
    const uint32_t sink = 4, recent = 64, n_ubatch = 8;
    const int n_vocab = 128;

    evict_ctx ec;
    if (!make_evict_ctx_ready(ec, g_model_qwen, sink, recent, n_ubatch, 1, true, "sparse ubatch", LLAMA_CONTEXT_TYPE_MTP)) {
        return;
    }

    const int failures0 = g_failures;

    // fill the cache so the sparse ubatch must recycle
    bool ok = true;
    for (int pos = 0; pos < 130; pos++) {
        ok = ok && decode_range(ec.ctx, n_vocab, pos, 1, 0);
    }
    kv_check(ok, "pre-fill decode failed");
    kv_check(ec.kv->get_evict_stats().cells_recycled() > 0, "expected recycling before the sparse ubatch");
    if (g_failures != failures0) {
        return;
    }

    // one ubatch with holes; the min-position token starts just below the max
    // token's recent-window edge (199 - 64 = 135), crossing the boundary
    const std::vector<llama_pos> pos_list = { 130, 133, 196, 199 };
    const auto present_before = present_positions(ec.kv, 0);
    const uint64_t recycled_before = ec.kv->get_evict_stats().cells_recycled();

    llama_batch batch = llama_batch_init(pos_list.size(), 0, 1);
    for (size_t i = 0; i < pos_list.size(); i++) {
        common_batch_add(batch, pos_list[i] % n_vocab, pos_list[i], { 0 }, false);
    }
    const bool decoded = llama_decode(ec.ctx, batch) == 0;
    llama_batch_free(batch);
    kv_check(decoded, "sparse ubatch decode failed");
    if (!decoded) {
        return;
    }
    kv_check(ec.kv->get_evict_stats().cells_recycled() > recycled_before,
            "expected the sparse ubatch to recycle cells");

    // every batch position must be present...
    const auto present = present_positions(ec.kv, 0);
    for (const llama_pos p : pos_list) {
        kv_check(present.count(p) != 0, "sparse ubatch position %" PRId32 " missing after decode", p);
    }

    // ...and no pre-existing cell that any batch token still needs may have
    // been recycled (the victim threshold is derived from the min position)
    const llama_pos lo = pos_list.front() - (llama_pos) recent;
    const llama_pos hi = pos_list.back();
    for (llama_pos p = lo; p <= hi; p++) {
        if (present_before.count(p)) {
            kv_check(present.count(p) != 0,
                    "position %" PRId32 " needed by the sparse ubatch was recycled", p);
        }
    }
}

int main(int argc, char ** argv) {
    common_init();

    // self-contained test: no model file is required, so no common args parsing
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            fprintf(stderr, "Usage: %s\n", argv[0]);
            return 0;
        }
    }

    ggml_backend_load_all();

    llama_model_ptr model_llama = make_model(LLM_ARCH_LLAMA, 0);
    llama_model_ptr model_qwen  = make_model(LLM_ARCH_QWEN35, 1);
    llama_model_ptr model_gemma = make_model(LLM_ARCH_GEMMA2, 0);

    if (!model_llama || !model_qwen || !model_gemma) {
        fprintf(stderr, "failed to create test models\n");
        return 1;
    }

    g_model_llama = model_llama.get();
    g_model_qwen  = model_qwen.get();
    g_model_gemma = model_gemma.get();

    const char * names[] = {
        "sink preservation",
        "recent-window retention",
        "multiple eviction cycles",
        "MTP disable (recent < 64)",
        "SWA disable (swa_type != NONE)",
        "state save/load after fragmentation",
        "multi-sequence recycling",
        "ubatch crossing the eviction boundary",
        "quantized KV (q4_0)",
        "unified vs non-unified multi-sequence",
        "simultaneous multi-seq eviction",
        "shared-cell exhaustion",
        "MTP rollback under eviction",
        "boundary values",
        "sparse ubatch with holes",
    };

    void (*const fns[])() = {
        test_sink_preservation,
        test_recent_window_retention,
        test_multiple_eviction_cycles,
        test_mtp_disable,
        test_swa_disable,
        test_state_roundtrip,
        test_multi_sequence,
        test_ubatch_boundary,
        test_quantized_kv,
        test_unified_nonunified_multi_seq,
        test_simultaneous_multi_seq_eviction,
        test_shared_cell_exhaustion,
        test_mtp_rollback_eviction,
        test_boundary_values,
        test_sparse_ubatch,
    };

    const int n_total = (int) (sizeof(names)/sizeof(names[0]));
    int n_pass = 0;
    for (int i = 0; i < n_total; i++) {
        const int failures_before = g_failures;
        fprintf(stderr, "== %s ==\n", names[i]);
        fns[i]();
        if (g_failures == failures_before) {
            n_pass++;
        }
    }

    fprintf(stderr, "%d/%d test cases passed, %d checks failed\n", n_pass, n_total, g_failures);
    return g_failures == 0 ? 0 : 1;
}