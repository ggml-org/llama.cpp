// llama-trace-moe — GLM-5.2 / DeepSeek-MoE expert-routing tracer.
//
// Planned artifact referenced by GLM52_TRACE_PLAN.md (Phase 1, Story 1):
//   /Users/spotted/projects/llama.cpp/examples/trace-moe/
//
// Captures real MoE routing tensors (ffn_moe_topk, ffn_moe_weights) per
// (token, layer) via the ggml backend eval callback, pairs them, and writes a
// compact, asynchronous JSONL trace that gguf2mlx.tracing.analyze consumes.
//
// Heavy fields never run on the eval hot path: the callback only builds compact
// records and pushes them to a bounded queue; a dedicated writer thread drains
// the queue and writes JSONL + a <trace>.meta.json sidecar at the end.
//
// The --trace-* flags are pre-scanned out of argv so common_params_parse only
// sees standard llama-cli flags (-m, -p, -f, -ngl, -c, -n, --temp, --jinja,
// -cnv, -st, --chat-template-kwargs, ...).
//
// Build:
//   cmake --build build-metal --target llama-trace-moe
//
// Run:
//   build-metal/bin/llama-trace-moe \
//       -m /path/to/GLM-5.2-mixed-00001-of-00009.gguf -ngl 999 -c 32768 \
//       -f prompt.txt -n 256 --temp 0 --jinja -cnv -st \
//       --trace-out traces/run.jsonl --trace-task-label coding \
//       --trace-language en --trace-script Latin --trace-prompt-family coding \
//       --trace-test-id coding_01 --trace-phase both --trace-backpressure sample

#include "arg.h"
#include "common.h"
#include "log.h"
#include <filesystem>
#include <fstream>

#include <nlohmann/json.hpp>

#include "llama.h"
#include "gguf.h"  // Story 8 AC: read *.expert_count KV for n_expert_total

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <ctime>
#include <deque>
#include <mutex>
#include <regex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>

// Self-contained SHA-256 (FIPS-180-4) — no external crypto dependency.
// Used only for prompt_sha256 + model_sha256_prefix in the .meta.json sidecar;
// never participates in inference. ~60 LoC, public-domain reference impl.
namespace sha256_detail {
    struct SHA256 {
        uint32_t state[8];
        uint64_t bitlen;
        uint8_t  data[64];
        size_t   datalen;
    };
    static const uint32_t K[64] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
    };
    static inline uint32_t rotr(uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); }
    static inline void sha256_init(SHA256 & c) {
        c.bitlen = 0; c.datalen = 0;
        static const uint32_t H0[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
        for (int i = 0; i < 8; i++) c.state[i] = H0[i];
    }
    static inline void sha256_transform(SHA256 & c, const uint8_t * d) {
        uint32_t m[64], a,b,cc,dd,e,f,g,h, W1, W2;
        for (int i = 0; i < 16; i++)
            m[i] = (uint32_t(d[i*4]) << 24) | (uint32_t(d[i*4+1]) << 16) | (uint32_t(d[i*4+2]) << 8) | uint32_t(d[i*4+3]);
        for (int i = 16; i < 64; i++) {
            W1 = rotr(m[i-15],7) ^ rotr(m[i-15],18) ^ (m[i-15] >> 3);
            W2 = rotr(m[i-2],17) ^ rotr(m[i-2],19) ^ (m[i-2] >> 10);
            m[i] = m[i-16] + W1 + m[i-7] + W2;
        }
        a=c.state[0]; b=c.state[1]; cc=c.state[2]; dd=c.state[3];
        e=c.state[4]; f=c.state[5]; g=c.state[6]; h=c.state[7];
        for (int i = 0; i < 64; i++) {
            uint32_t S1 = rotr(e,6) ^ rotr(e,11) ^ rotr(e,25);
            uint32_t ch = (e & f) ^ (~e & g);
            uint32_t t1 = h + S1 + ch + K[i] + m[i];
            uint32_t S0 = rotr(a,2) ^ rotr(a,13) ^ rotr(a,22);
            uint32_t mj = (a & b) ^ (a & cc) ^ (b & cc);
            uint32_t t2 = S0 + mj;
            h=g; g=f; f=e; e=dd+t1; dd=cc; cc=b; b=a; a=t1+t2;
        }
        c.state[0]+=a; c.state[1]+=b; c.state[2]+=cc; c.state[3]+=dd;
        c.state[4]+=e; c.state[5]+=f; c.state[6]+=g; c.state[7]+=h;
    }
    static inline void sha256_update(SHA256 & c, const uint8_t * data, size_t len) {
        for (size_t i = 0; i < len; i++) {
            c.data[c.datalen++] = data[i];
            if (c.datalen == 64) { sha256_transform(c, c.data); c.bitlen += 512; c.datalen = 0; }
        }
    }
    static inline void sha256_final(SHA256 & c, uint8_t out[32]) {
        uint64_t bits = c.bitlen + uint64_t(c.datalen) * 8;
        c.data[c.datalen++] = 0x80;
        if (c.datalen > 56) { while (c.datalen < 64) c.data[c.datalen++] = 0; sha256_transform(c, c.data); c.datalen = 0; }
        while (c.datalen < 56) c.data[c.datalen++] = 0;
        for (int i = 7; i >= 0; i--) c.data[c.datalen++] = uint8_t((bits >> (i*8)) & 0xFF);
        sha256_transform(c, c.data);
        for (int i = 0; i < 8; i++) {
            out[i*4]   = uint8_t((c.state[i] >> 24) & 0xFF);
            out[i*4+1] = uint8_t((c.state[i] >> 16) & 0xFF);
            out[i*4+2] = uint8_t((c.state[i] >> 8) & 0xFF);
            out[i*4+3] = uint8_t(c.state[i] & 0xFF);
        }
    }
}
// Compute hex SHA-256 over a byte buffer. Returns lowercase 64-char hex.
static std::string sha256_hex(const uint8_t * data, size_t len) {
    sha256_detail::SHA256 ctx;
    sha256_detail::sha256_init(ctx);
    sha256_detail::sha256_update(ctx, data, len);
    uint8_t digest[32];
    sha256_detail::sha256_final(ctx, digest);
    static const char * hx = "0123456789abcdef";
    std::string out(64, '0');
    for (int i = 0; i < 32; i++) {
        out[i*2]   = hx[(digest[i] >> 4) & 0xF];
        out[i*2+1] = hx[digest[i] & 0xF];
    }
    return out;
}
static std::string sha256_hex(const std::string & s) {
    return sha256_hex(reinterpret_cast<const uint8_t *>(s.data()), s.size());
}
// ISO 8601 UTC timestamp, e.g. "2026-06-20T18:42:01Z".
static std::string iso_utc_now() {
    using std::chrono::system_clock;
    auto now = system_clock::now();
    std::time_t t = system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    char buf[24];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return std::string(buf);
}
#include <vector>

namespace {

constexpr int TRACE_SCHEMA_VERSION = 1;

enum class Backpressure { Block, Drop, Sample };

struct TraceConfig {
    std::string trace_out;
    std::string trace_prompts;              // path to JSONL prompts file (batched mode)
    std::string task_label     = "adhoc";
    std::string language       = "en";
    std::string script         = "Latin";
    std::string prompt_family  = "misc";
    std::string test_id        = "adhoc";
    std::string phase          = "both";  // prefill|generation|both
    std::string backpressure   = "sample";
    std::string trace_layers;             // "0,1,2,6" or "0..78" or "" = all
    int         trace_max_tokens = 0;     // 0 = unlimited
    // Story 6 AC: bounded activation summarization. Default off — full
    // activation dumps would dominate the JSONL for prefill on 20k-token
    // prompts. 'trace_activations' is a comma-separated list of tensor
    // stems to capture ('l_out', 'kqv_out', 'ffn_out', 'ffn_moe_out',
    // 'ffn_swiglu', 'attn_norm', 'ffn_norm', ...). Empty string = off.
    // 'trace_activation_topk' = N: top-N channels per (tensor, token) by
    // |magnitude|. 0 = use DEFAULT (10).
    std::string trace_activations;        // "" = off; "l_out" / "l_out,kqv_out" = on
    int         trace_activation_topk = 0; // 0 = default (10)
    int         trace_activation_stride = 2; // emit for every Nth layer to bound prefill JSONL
    // Reproducibility provenance (Story 9 AC): full argv joined as a string,
    // and process start time in UTC ISO 8601. Set once in main() before the
    // batch loop so every per-prompt sidecar carries the same provenance.
    std::string full_command_line;
    std::string started_at_iso;
    // chat-template flags that this example (LLAMA_EXAMPLE_COMMON) cannot parse and
    // that the tracer would ignore anyway (it tokenizes params.prompt verbatim).
    // Collected here so main() can log one honest notice.
    std::vector<std::string> stripped_chat_flags;
};

// ---------------------------------------------------------------------------
// JSON helpers (no external json dependency; records are small and flat).
// ---------------------------------------------------------------------------
std::string join(const std::vector<std::string> & v, const std::string & sep) {
    std::string out;
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) out += sep;
        out += v[i];
    }
    return out;
}

void json_escape_append(const std::string & s, std::string & out) {
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
}

std::string render_record(
        const std::string & run_id, const std::string & model, const char * phase,
        int token_index, int layer,
        const std::vector<int32_t> & experts, const std::vector<float> & weights,
        float entropy, int n_expert_total,
        const TraceConfig & cfg) {
    std::string s;
    s.reserve(256 + experts.size() * 24);
    s += "{\"schema_version\":";
    s += std::to_string(TRACE_SCHEMA_VERSION);
    s += ",\"event\":\"moe_topk\"";
    s += ",\"run_id\":\"";     json_escape_append(run_id, s); s += "\"";
    s += ",\"model\":\"";      json_escape_append(model, s);  s += "\"";
    s += ",\"phase\":\"";      s += phase;                     s += "\"";
    s += ",\"token_index\":";  s += std::to_string(token_index);
    s += ",\"layer\":";        s += std::to_string(layer);
    s += ",\"task_label\":\""; json_escape_append(cfg.task_label, s);  s += "\"";
    s += ",\"language\":\"";   json_escape_append(cfg.language, s);    s += "\"";
    s += ",\"script\":\"";     json_escape_append(cfg.script, s);      s += "\"";
    s += ",\"prompt_family\":\""; json_escape_append(cfg.prompt_family, s); s += "\"";
    s += ",\"test_id\":\"";    json_escape_append(cfg.test_id, s);     s += "\"";
    s += ",\"n_expert_used\":"; s += std::to_string((int) experts.size());
    if (n_expert_total > 0) { s += ",\"n_expert\":"; s += std::to_string(n_expert_total); }
    // experts
    s += ",\"experts\":[";
    for (size_t i = 0; i < experts.size(); ++i) {
        if (i) s += ",";
        s += std::to_string(experts[i]);
    }
    s += "],\"weights\":[";
    for (size_t i = 0; i < weights.size(); ++i) {
        if (i) s += ",";
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.6g", (double) weights[i]);
        s += buf;
    }
    s += "]";
    s += ",\"router_entropy\":"; { char buf[32]; std::snprintf(buf, sizeof(buf), "%.6g", (double) entropy); s += buf; }
    s += "}\n";
    return s;
}

// Story 6 AC: render a bounded activation-summary JSONL record. Mirrors
// render_record's provenance block (run_id / model / phase / token_index /
// layer / task_label / language / script / prompt_family / test_id) but
// adds the activation-specific fields: tensor_stem, n_channels, topk,
// top_k_channels [[idx, magnitude], ...] (sorted by |magnitude| desc),
// and per-token stat summaries (l2_norm / mean / std / max_abs).
//
// top_k_channels magnitude is the raw signed value so the analyzer can
// distinguish excitatory vs inhibitory channels; analyzer bucketing keys on
// channel_idx (frequency) not magnitude.
std::string render_activation_record(
        const std::string & run_id, const std::string & model, const char * phase,
        int token_index, int layer,
        const std::string & tensor_stem, int n_channels, int topk,
        const std::vector<std::pair<int, float>> & top_k_channels,
        double l2_norm, double mean, double std_dev, double max_abs,
        const TraceConfig & cfg) {
    std::string s;
    s.reserve(256 + top_k_channels.size() * 24);
    s += "{\"schema_version\":";
    s += std::to_string(TRACE_SCHEMA_VERSION);
    s += ",\"event\":\"activation_summary\"";
    s += ",\"run_id\":\"";     json_escape_append(run_id, s); s += "\"";
    s += ",\"model\":\"";      json_escape_append(model, s);  s += "\"";
    s += ",\"phase\":\"";      s += phase;                     s += "\"";
    s += ",\"token_index\":";  s += std::to_string(token_index);
    s += ",\"layer\":";        s += std::to_string(layer);
    s += ",\"task_label\":\""; json_escape_append(cfg.task_label, s);  s += "\"";
    s += ",\"language\":\"";   json_escape_append(cfg.language, s);    s += "\"";
    s += ",\"script\":\"";     json_escape_append(cfg.script, s);      s += "\"";
    s += ",\"prompt_family\":\""; json_escape_append(cfg.prompt_family, s); s += "\"";
    s += ",\"test_id\":\"";    json_escape_append(cfg.test_id, s);     s += "\"";
    s += ",\"tensor_stem\":\""; json_escape_append(tensor_stem, s);    s += "\"";
    s += ",\"n_channels\":"; s += std::to_string(n_channels);
    s += ",\"topk\":";        s += std::to_string(topk);
    // top_k_channels (already sorted desc by |magnitude|)
    s += ",\"top_k_channels\":[";
    for (size_t i = 0; i < top_k_channels.size(); ++i) {
        if (i) s += ",";
        char buf[64];
        std::snprintf(buf, sizeof(buf), "[%d,%.6g]",
            top_k_channels[i].first, (double) top_k_channels[i].second);
        s += buf;
    }
    s += "]";
    s += ",\"l2_norm\":";  { char buf[32]; std::snprintf(buf, sizeof(buf), "%.6g", l2_norm);  s += buf; }
    s += ",\"mean\":";      { char buf[32]; std::snprintf(buf, sizeof(buf), "%.6g", mean);     s += buf; }
    s += ",\"std\":";       { char buf[32]; std::snprintf(buf, sizeof(buf), "%.6g", std_dev);  s += buf; }
    s += ",\"max_abs\":";   { char buf[32]; std::snprintf(buf, sizeof(buf), "%.6g", max_abs);  s += buf; }
    s += "}\n";
    return s;
}

// ShortGPT-style Block Influence: emitted as a separate event alongside
// activation_summary, but ALWAYS (independent of activation_stride) when the
// user opted into l_out via --trace-activations. BI = 1 - cos(h_in, h_out)
// where h_in = previous layer's l_out residual and h_out = this layer's l_out
// residual, both for the same token. The cache is cleared between prompts in
// run_one_prompt. The 'layer:N' on the record is the layer whose transform
// produced h_out (i.e. layer N's BI, computed from l_out-{N-1} and l_out-N).
std::string render_bi_record(
        const std::string & run_id, const std::string & model, const char * phase,
        int token_index, int layer, int n_channels,
        double cos_sim, double bi_score, const TraceConfig & cfg) {
    // Build with std::string::push_back + arithmetic conversions to avoid the
    // multiple-escape fragility of chained s += "..." clauses for JSON.
    std::string s;
    s.reserve(320);
    s.push_back('{');
    s += "\"schema_version\":"; s += std::to_string(TRACE_SCHEMA_VERSION);
    s += ",\"event\":\"block_influence\"";
    s += ",\"run_id\":\"";  json_escape_append(run_id, s); s.push_back('"');
    s += ",\"model\":\"";   json_escape_append(model, s); s.push_back('"');
    s += ",\"phase\":\"";   s += phase; s.push_back('"');
    s += ",\"token_index\":"; s += std::to_string(token_index);
    s += ",\"layer\":";       s += std::to_string(layer);
    s += ",\"task_label\":\""; json_escape_append(cfg.task_label, s); s.push_back('"');
    s += ",\"language\":\"";   json_escape_append(cfg.language, s); s.push_back('"');
    s += ",\"script\":\"";     json_escape_append(cfg.script, s); s.push_back('"');
    s += ",\"prompt_family\":\""; json_escape_append(cfg.prompt_family, s); s.push_back('"');
    s += ",\"test_id\":\"";    json_escape_append(cfg.test_id, s); s.push_back('"');
    s += ",\"n_channels\":"; s += std::to_string(n_channels);
    { char buf[32]; std::snprintf(buf, sizeof(buf), "%.6g", cos_sim);  s += ",\"cos_sim\":";  s += buf; }
    { char buf[32]; std::snprintf(buf, sizeof(buf), "%.6g", bi_score); s += ",\"bi_score\":"; s += buf; }
    s += "}\n";
    return s;
}

// ---------------------------------------------------------------------------
// Async, bounded writer.
// ---------------------------------------------------------------------------
struct TraceWriter {
    std::FILE * fh = nullptr;
    std::string out_path;
    std::mutex  mu;
    std::condition_variable cv;
    std::deque<std::string> q;
    size_t      max_queue = 8192;
    Backpressure bp = Backpressure::Sample;
    std::atomic<bool> stop{false};
    std::thread writer;
    std::atomic<uint64_t> written{0};
    std::atomic<uint64_t> dropped{0};
    std::atomic<uint64_t> sampled{0};

    bool open(const std::string & path, Backpressure b, size_t qsize) {
        out_path = path;
        bp = b;
        max_queue = qsize > 0 ? qsize : 8192;
        // reset state for reuse across prompts in batched mode: a previous close()
        // set stop=true and joined the writer thread, so we must clear stop before
        // spawning a fresh thread (otherwise run() exits immediately on empty queue).
        stop.store(false);
        fh = std::fopen(path.c_str(), "w");
        if (!fh) return false;
        writer = std::thread([this]{ run(); });
        return true;
    }

    void run() {
        while (true) {
            std::string line;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [this]{ return stop.load() || !q.empty(); });
                if (q.empty() && stop.load()) break;
                line = std::move(q.front());
                q.pop_front();
            }
            std::fputs(line.c_str(), fh);
            ++written;
            if (written.load() % 64 == 0) std::fflush(fh);
        }
        if (std::fflush(fh) != 0) {
            LOG_ERR("%s: flush failed\n", __func__);
        }
    }

    void push(std::string && line) {
        bool accepted = false;
        if (bp == Backpressure::Sample) {
            // Adaptive sampling: when near-full, probabilistically shed load.
            double fill = 0.0;
            {
                std::lock_guard<std::mutex> lk(mu);
                fill = (double) q.size() / (double) max_queue;
            }
            if (fill > 0.9) {
                // keep roughly 1 of every 2 under pressure
                if ((written.load() ^ dropped.load()) & 1u) {
                    ++sampled;
                    return;
                }
            }
            std::lock_guard<std::mutex> lk(mu);
            if (q.size() < max_queue) { q.emplace_back(std::move(line)); accepted = true; }
        } else if (bp == Backpressure::Drop) {
            std::lock_guard<std::mutex> lk(mu);
            if (q.size() < max_queue) { q.emplace_back(std::move(line)); accepted = true; }
        } else { // Block
            std::unique_lock<std::mutex> lk(mu);
            cv.wait(lk, [this]{ return q.size() < max_queue || stop.load(); });
            if (!stop.load()) { q.emplace_back(std::move(line)); accepted = true; }
        }
        if (accepted) cv.notify_one();
        else ++dropped;
    }

    void close() {
        stop.store(true);
        cv.notify_all();
        if (writer.joinable()) writer.join();
        if (fh) { std::fclose(fh); fh = nullptr; }
    }
};

// ---------------------------------------------------------------------------
// Trace state shared with the eval callback.
// ---------------------------------------------------------------------------
struct TraceState {
    std::string run_id;
    std::string model_name;
    TraceConfig cfg;
    TraceWriter writer;

    // decode loop signals
    const char * current_phase = "prefill";  // "prefill" | "generation"
    int token_base = 0;        // token index of the first token of the current batch
    int batch_n_tokens = 0;    // n_tokens of the batch currently being decoded
    int phase_tokens_emitted[2] = {0, 0}; // [prefill, generation]
    int max_new_tokens = 0;    // mirror of params.n_predict
    int n_expert_total = 0;

    std::vector<int> selected_layers; // empty = all
    // Story 6 AC: activation tensor stems the user opted into (parsed from
    // --trace-activations 'a,b,c' cfg field). Empty = no activation records.
    std::vector<std::string> activation_stems;
    int activation_topk = 10;          // DEFAULT_ACTIVATION_TOPK mirrored from Python schema
    int activation_stride = 2;          // emit every Nth layer to bound prefill JSONL
    bool trace_prefill = true;
    bool trace_generation = true;

    // per-batch pending topk buffer keyed by layer
    std::unordered_map<int, std::vector<int32_t>> pending_topk;
    std::unordered_map<int, int> pending_topk_n_used;   // n_used of the pending topk
    std::unordered_map<int, int> pending_topk_n_tokens; // n_tokens of the pending topk

    // ShortGPT Block Influence cache: per-token, the previous layer's l_out
    // residual stream (full 6144-float F32 vector). When layer N's l_out fires
    // for token t, we look up this cache; if a vector is present (from layer
    // N-1), compute cos_sim(h_in, h_out) and emit a 'block_influence'
    // record. Memory cost: ~n_channels float per token currently in flight
    // (~24 KB/token at n_channels=6144). Resets every prompt in run_one_prompt.
    //
    // Note: the trace_cb_eval callback fires for EVERY ggml_tensor node in
    // graph-eval order. For a standard pre-norm residual stream, l_out-N's
    // output is layer N+1's input, so cos_sim(prev_l_out_per_token[t],
    // current_l_out[t]) gives BI for layer N (the transform that produced
    // current_l_out).
    std::unordered_map<int, std::vector<float>> prev_l_out_per_token;

    // (readback now uses a per-call fresh buffer; no shared scratch field)
};

int phase_index(const char * phase) {
    return std::strcmp(phase, "generation") == 0 ? 1 : 0;
}

int extract_layer(const char * name) {
    // tensors in the eval callback are named like "ffn_moe_topk-42" or "ffn_moe_weights-7"
    std::string s(name);
    auto pos = s.find_last_of('-');
    if (pos == std::string::npos) return -1;
    try { return std::stoi(s.substr(pos + 1)); }
    catch (...) { return -1; }
}

bool is_topk_tensor(const char * name) {
    std::string s(name);
    return s.rfind("ffn_moe_topk", 0) == 0;
}
bool is_weights_tensor(const char * name) {
    // avoid the _norm/_softmax/_sum variants: exact base name + layer suffix only
    std::string s(name);
    if (s.rfind("ffn_moe_weights", 0) != 0) return false;
    // accept "ffn_moe_weights" or "ffn_moe_weights-N"
    if (s.size() == std::string("ffn_moe_weights").size()) return true;
    return s[std::string("ffn_moe_weights").size()] == '-';
}

// Story 6 AC: detect named activation tensors the user opted into via
// --trace-activations. The callback is fired for EVERY intermediate tensor,
// so predicate must be tight: match '<stem>-N' exactly where stem is in the
// configured stems vector. Returns the stem via out_stem on match.
bool is_activation_tensor(const char * name, const std::vector<std::string> & stems,
                          std::string & out_stem) {
    if (stems.empty()) return false;
    std::string s(name);
    auto pos = s.find_last_of('-');
    if (pos == std::string::npos) return false;
    std::string stem = s.substr(0, pos);
    // layer suffix must be a non-negative integer (not another word)
    try {
        int layer = std::stoi(s.substr(pos + 1));
        (void) layer;
    } catch (...) { return false; }
    for (const auto & want : stems) {
        if (stem == want) {
            out_stem = stem;
            return true;
        }
    }
    return false;
}

bool layer_selected(const TraceState & st, int layer) {
    if (st.selected_layers.empty()) return true;
    for (int l : st.selected_layers) if (l == layer) return true;
    return false;
}

float compute_entropy(const std::vector<float> & w) {
    double total = 0.0;
    for (float x : w) total += x;
    if (total <= 0) return 0.0f;
    double h = 0.0;
    for (float x : w) {
        if (x <= 0) continue;
        double p = x / total;
        h -= p * std::log2(p);
    }
    return (float) h;
}

// Emit ONE record for a single (token-in-batch, layer) routing event.
// topk_ptr/weights_ptr each have n_used contiguous elements for this token.
void emit_one(TraceState & st, int layer, int n_used, int tok_in_batch,
             const int32_t * topk_ptr, const float * weights_ptr) {
    int pi = phase_index(st.current_phase);
    bool phase_on = (pi == 0) ? st.trace_prefill : st.trace_generation;
    if (!phase_on) return;
    if (st.cfg.trace_max_tokens > 0 && st.phase_tokens_emitted[pi] >= st.cfg.trace_max_tokens) {
        return;
    }
    std::vector<int32_t> experts(n_used);
    std::vector<float>    weights(n_used);
    for (int k = 0; k < n_used; ++k) {
        experts[k] = topk_ptr[k];
        weights[k] = weights_ptr[k];
    }
    float ent = compute_entropy(weights);
    std::string rec = render_record(
        st.run_id, st.model_name, st.current_phase,
        st.token_base + tok_in_batch, layer, experts, weights, ent, st.n_expert_total, st.cfg);
    st.writer.push(std::move(rec));
    st.phase_tokens_emitted[pi]++;
}

bool trace_cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    if (ask) return true;  // we are interested when data is available
    auto * st = static_cast<TraceState *>(user_data);
    const char * name = t->name;
    if (!name || !name[0]) return true;

    // Story 6 AC: activation summary path. If the user opted in via
    // --trace-activations and the current tensor is one of the requested
    // stems, compute top-K channels by |magnitude| + per-token stat
    // summaries and push an activation_summary record. This is separate from
    // the routing-event path below so the JSONL holds both event types.
    if (!st->activation_stems.empty()) {
        std::string matched_stem;
        if (is_activation_tensor(name, st->activation_stems, matched_stem)) {
            int layer = extract_layer(name);
            if (layer >= 0 && layer_selected(*st, layer)) {
                // --------------------------------------------------------------------
                // ShortGPT BI: ALWAYS (independent of activation_stride) when the
                // matched stem is 'l_out'. We do a separate host copy here so the
                // existing top-K activation_summary emission below is untouched,
                // at the cost of one redundant copy per layer only when l_out
                // is the matched stem. For other stems (kqv_out, ffn_out, ...)
                // this whole block is skipped.
                // --------------------------------------------------------------------
                if (matched_stem == "l_out" && t->type == GGML_TYPE_F32) {
                    size_t bi_nbytes = ggml_nbytes(t);
                    std::vector<uint8_t> bi_raw(bi_nbytes);
                    ggml_backend_tensor_get(t, bi_raw.data(), 0, bi_nbytes);
                    int n_channels = (int) t->ne[0];
                    int n_tokens   = (int) t->ne[1];
                    const float * bi_src =
                        reinterpret_cast<const float *>(bi_raw.data());
                    int pi = (std::string(st->current_phase) == "generation") ? 1 : 0;
                    bool phase_on = (pi == 0) ? st->trace_prefill : st->trace_generation;
                    for (int tok = 0; tok < n_tokens; ++tok) {
                        int tok_idx = st->token_base + tok;
                        const float * v = bi_src + (size_t) tok * n_channels;
                        auto it = st->prev_l_out_per_token.find(tok_idx);
                        if (it != st->prev_l_out_per_token.end() && phase_on) {
                            const float * prev = it->second.data();
                            double dot = 0.0, na = 0.0, nb = 0.0;
                            for (int c = 0; c < n_channels; ++c) {
                                double a = (double) v[c];
                                double b = (double) prev[c];
                                dot += a * b;
                                na  += a * a;
                                nb  += b * b;
                            }
                            double denom = std::sqrt(na) * std::sqrt(nb);
                            double cos_sim = denom > 0.0 ? dot / denom : 0.0;
                            double bi_score = 1.0 - cos_sim;
                            std::string rec = render_bi_record(
                                st->run_id, st->model_name, st->current_phase,
                                tok_idx, layer, n_channels,
                                cos_sim, bi_score, st->cfg);
                            st->writer.push(std::move(rec));
                        }
                        // Cache current as 'previous' for the next layer's
                        // cos_sim lookup, even when this phase's records are
                        // disabled — keeps continuity so that re-enabling the
                        // phase mid-prompt doesn't produce a bogus 'no prev'
                        // miss on the layer where tracing was off.
                        st->prev_l_out_per_token[tok_idx] =
                            std::vector<float>(v, v + n_channels);
                    }
                }

                if ((layer % st->activation_stride) == 0) {
                // ALWAYS copy via ggml_backend_tensor_get for the same Metal
                // sync reason as the routing-event path.
                size_t nbytes = ggml_nbytes(t);
                std::vector<uint8_t> raw(nbytes);
                ggml_backend_tensor_get(t, raw.data(), 0, nbytes);
                // Activation tensors are F32 [n_embd, n_tokens] contiguous.
                // (The probe on GLM-5.2 confirmed l_out-N is ne=[6144, T].)
                if (t->type == GGML_TYPE_F32) {
                    int n_channels = (int) t->ne[0];
                    int n_tokens  = (int) t->ne[1];
                    const float * src = reinterpret_cast<const float *>(raw.data());
                    int topk = st->activation_topk > 0 ? st->activation_topk : 10;
                    // Per-token: compute top-K channels + stats. Push one
                    // activation_summary record per token in the batch.
                    for (int tok = 0; tok < n_tokens; ++tok) {
                        // zero-copy view into this token's channel vector
                        const float * v = src + (size_t) tok * n_channels;
                        // Compute mean / max_abs in single pass; L2 needs sumsq.
                        double sum = 0.0, sumsq = 0.0, max_abs_val = 0.0;
                        for (int c = 0; c < n_channels; ++c) {
                            double x = (double) v[c];
                            sum += x;
                            sumsq += x * x;
                            double ax = std::fabs(x);
                            if (ax > max_abs_val) max_abs_val = ax;
                        }
                        double mean_v = sum / n_channels;
                        double var = (sumsq / n_channels) - mean_v * mean_v;
                        if (var < 0.0) var = 0.0;  // numerical robustness
                        double std_v = std::sqrt(var);
                        double l2 = std::sqrt(sumsq);
                        // Build top-K channels by |magnitude|. Use a partial
                        // sort to bound cost: copy (idx, val) pairs, partial_sort
                        // on |val| desc, take first topk.
                        // For 6144 channels × every-token, n_channels² scans must
                        // NOT happen per token. Use a min-heap of size topk for O(N log K).
                        std::vector<std::pair<float, int>> heap;  // (|val|, idx)
                        heap.reserve(topk + 1);
                        for (int c = 0; c < n_channels; ++c) {
                            float av = std::fabs(v[c]);
                            if ((int) heap.size() < topk) {
                                heap.push_back({av, c});
                                if ((int) heap.size() == topk) {
                                    // heapify to min-heap on |val|
                                    std::make_heap(heap.begin(), heap.end(),
                                        [](const auto & a, const auto & b){ return a.first > b.first; });
                                }
                            } else if (av > heap.front().first) {
                                std::pop_heap(heap.begin(), heap.end(),
                                    [](const auto & a, const auto & b){ return a.first > b.first; });
                                heap.back() = {av, c};
                                std::push_heap(heap.begin(), heap.end(),
                                    [](const auto & a, const auto & b){ return a.first > b.first; });
                            }
                        }
                        // Extract + sort by |val| descending for the JSONL row
                        std::sort(heap.begin(), heap.end(),
                            [](const auto & a, const auto & b){ return a.first > b.first; });
                        std::vector<std::pair<int, float>> top_channels;
                        top_channels.reserve(heap.size());
                        for (const auto & p : heap) {
                            top_channels.push_back({p.second, v[p.second]});
                        }
                        int pi = (std::string(st->current_phase) == "generation") ? 1 : 0;
                        int tok_idx = st->token_base + tok;
                        // Respect trace_max_tokens per phase
                        if (st->cfg.trace_max_tokens > 0 &&
                            st->phase_tokens_emitted[pi] >= st->cfg.trace_max_tokens) {
                            // skip this token's activation record too
                            continue;
                        }
                        std::string rec = render_activation_record(
                            st->run_id, st->model_name, st->current_phase,
                            tok_idx, layer,
                            matched_stem, n_channels, topk,
                            top_channels, l2, mean_v, std_v, max_abs_val,
                            st->cfg);
                        st->writer.push(std::move(rec));
                        st->phase_tokens_emitted[pi]++;
                    }
                }
                }
            }
            // either way (matched stem or not), don't fall through to MoE path
            return true;
        }
    }

    if (!is_topk_tensor(name) && !is_weights_tensor(name)) return true;

    int layer = extract_layer(name);
    if (layer < 0) return true;

    if (!layer_selected(*st, layer)) return true;

    // ALWAYS copy to a fresh host buffer via ggml_backend_tensor_get. We never
    // read t->data directly even when ggml_backend_buffer_is_host(t->buffer) is
    // true, because on the Metal backend intermediate MoE tensors can live in
    // shared host memory whose GPU write has not landed by the time the eval
    // callback fires (this produced garbage expert IDs during fast single-token
    // generation decodes). ggml_backend_tensor_get performs the backend's sync.
    // The MoE topk/weights tensors are byte-contiguous in the [n_used, n_tokens]
    // frame (verified for GLM-5.2: topk ne=[N,T] nb=[4,N*4]; weights ne=[1,N,T]
    // nb=[4,4,N*4] — the degenerate dim0 makes dim0+dim1 a flat N-element run),
    // so a flat element read by `idx = k + tok*n_used` is correct after the copy.
    size_t nbytes = ggml_nbytes(t);
    std::vector<uint8_t> raw(nbytes);
    ggml_backend_tensor_get(t, raw.data(), 0, nbytes);
    const uint8_t * data = raw.data();

    if (is_topk_tensor(name)) {
        // I32 [n_used, n_tokens] (contiguous)
        if (t->type != GGML_TYPE_I32) return true;
        int n_used   = (int) t->ne[0];
        int n_tokens = (int) t->ne[1];
        size_t n = (size_t) n_used * (size_t) n_tokens;
        std::vector<int32_t> topk(n);
        const int32_t * src = reinterpret_cast<const int32_t *>(data);
        for (size_t i = 0; i < n; ++i) topk[i] = src[i];
        st->pending_topk[layer] = std::move(topk);
        st->pending_topk_n_used[layer] = n_used;
        st->pending_topk_n_tokens[layer] = n_tokens;
        return true;
    }

    // weights tensor (F32). Its ne may be [1, n_used, n_tokens] (degenerate
    // dim0) but the copied bytes are contiguous in [n_used, n_tokens], so we
    // read flat using the pending TOPK's n_used/n_tokens (the source of truth
    // for which experts were selected and how many tokens are in this batch).
    if (t->type != GGML_TYPE_F32) return true;
    auto it = st->pending_topk.find(layer);
    if (it == st->pending_topk.end()) {
        return true;  // weights without pending topk — drop
    }
    std::vector<int32_t> topk = std::move(it->second);
    st->pending_topk.erase(it);
    int n_used   = st->pending_topk_n_used[layer];
    int n_tokens = st->pending_topk_n_tokens[layer];
    const float * wsrc = reinterpret_cast<const float *>(data);
    size_t total_elems = nbytes / sizeof(float);
    for (int tok = 0; tok < n_tokens; ++tok) {
        std::vector<float> w(n_used, 0.0f);
        for (int k = 0; k < n_used; ++k) {
            size_t idx = (size_t) k + (size_t) tok * (size_t) n_used;
            if (idx < total_elems) w[k] = wsrc[idx];
        }
        const int32_t * tk = topk.data() + (size_t) tok * (size_t) n_used;
        emit_one(*st, layer, n_used, tok, tk, w.data());
    }
    return true;
}

// ---------------------------------------------------------------------------
// argv pre-scan for --trace-* flags.
// ---------------------------------------------------------------------------
TraceConfig config_from_trace_flags(
        std::vector<std::string> & trace_args,
        int & argc, char ** & argv_out) {
    TraceConfig cfg;
    // rebuild argv without --trace-* args
    static thread_local std::vector<char*> kept;
    kept.clear();
    kept.push_back(argv_out[0]);
    for (size_t i = 1; i < trace_args.size(); ++i) {
        const std::string & a = trace_args[i];
        auto take_next = [&](std::string & dst) {
            if (i + 1 < trace_args.size()) { dst = trace_args[++i]; }
        };
        if (a == "--trace-out")              take_next(cfg.trace_out);
        else if (a == "--trace-prompts")         take_next(cfg.trace_prompts);
        else if (a == "--trace-task-label")  take_next(cfg.task_label);
        else if (a == "--trace-language")    take_next(cfg.language);
        else if (a == "--trace-script")      take_next(cfg.script);
        else if (a == "--trace-prompt-family") take_next(cfg.prompt_family);
        else if (a == "--trace-test-id")     take_next(cfg.test_id);
        else if (a == "--trace-phase")       take_next(cfg.phase);
        else if (a == "--trace-backpressure") take_next(cfg.backpressure);
        else if (a == "--trace-layers")      take_next(cfg.trace_layers);
        else if (a == "--trace-max-tokens")  {
            std::string tmp; take_next(tmp); cfg.trace_max_tokens = std::atoi(tmp.c_str());
        }
        // Story 6 AC: bounded activation summaries. Default off. Comma-
        // separated list of tensor stems to intercept (e.g. "l_out" or
        // "l_out,kqv_out,ffn_out"). Validation of which stems are real
        // happens at runtime — the eval callback simply won't fire for
        // stems that never appear in the graph.
        else if (a == "--trace-activations") {
            take_next(cfg.trace_activations);
        }
        else if (a == "--trace-activation-topk") {
            std::string tmp; take_next(tmp); cfg.trace_activation_topk = std::atoi(tmp.c_str());
        }
        else if (a == "--trace-activation-stride") {
            std::string tmp; take_next(tmp); cfg.trace_activation_stride = std::atoi(tmp.c_str());
        }
        // These flags are set_examples()-restricted to CLI/SERVER/MTMD and the
        // tracer would ignore them regardless (it never applies a chat template).
        // Strip them so llama-cli-style invocations don't fail with
        // "invalid argument"; main() warns the user they had no effect.
        else if (a == "--jinja" || a == "--no-jinja" ||
                 a == "-cnv" || a == "--conversation" ||
                 a == "-no-cnv" || a == "--no-conversation" ||
                 a == "-st" || a == "--single-turn") {
            cfg.stripped_chat_flags.push_back(a);
        }
        else if (a == "--chat-template-kwargs") {
            cfg.stripped_chat_flags.push_back(a);
            std::string tmp; take_next(tmp);  // swallow the JSON value
        }
        else {
            // keep this argument for common_params_parse
            kept.push_back(const_cast<char*>(a.c_str()));
        }
    }
    argv_out = kept.data();
    argc = (int) kept.size();
    return cfg;
}

std::vector<int> parse_layers(const std::string & spec) {
    std::vector<int> out;
    if (spec.empty()) return out;
    std::string s = spec;
    // handle "0..78"
    auto dd = s.find("..");
    if (dd != std::string::npos) {
        int lo = std::atoi(s.c_str());
        int hi = std::atoi(s.c_str() + dd + 2);
        for (int i = lo; i <= hi; ++i) out.push_back(i);
        return out;
    }
    std::string cur;
    for (char c : s + ",") {
        if (c == ',') { if (!cur.empty()) { out.push_back(std::atoi(cur.c_str())); cur.clear(); } }
        else cur += c;
    }
    return out;
}

// ---------------------------------------------------------------------------
// batched multi-prompt mode
// ---------------------------------------------------------------------------
// A single prompt spec parsed from the --trace-prompts JSONL file.
struct PromptSpec {
    std::string prompt;
    std::string task_label;
    std::string language;
    std::string script;
    std::string prompt_family;
    std::string test_id;
};

// Parse a JSONL file of prompts. Each line: {"prompt": "...", "task_label": "...",
// "language": "...", "test_id": "...", "script": "...", "prompt_family": "..."}.
// Fields other than "prompt" are optional and override the cfg defaults.
std::vector<PromptSpec> load_prompt_specs(const std::string & path) {
    std::vector<PromptSpec> out;
    std::FILE * f = std::fopen(path.c_str(), "r");
    if (!f) {
        LOG_ERR("%s: could not open --trace-prompts file: %s\n", __func__, path.c_str());
        return out;
    }
    char * line = nullptr;
    size_t cap = 0;
    ssize_t len;
    int lineno = 0;
    while ((len = getline(&line, &cap, f)) != -1) {
        ++lineno;
        std::string s(line, (size_t) len);
        // strip trailing newline
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
        if (s.empty()) continue;
        try {
            auto j = nlohmann::json::parse(s);
            PromptSpec ps;
            ps.prompt = j.value("prompt", "");
            if (ps.prompt.empty()) {
                LOG_WRN("%s: line %d has empty prompt, skipping\n", __func__, lineno);
                continue;
            }
            ps.task_label    = j.value("task_label",    "");
            ps.language      = j.value("language",      "");
            ps.script        = j.value("script",        "");
            ps.prompt_family = j.value("prompt_family", "");
            ps.test_id       = j.value("test_id",       "");
            out.push_back(std::move(ps));
        } catch (const std::exception & e) {
            LOG_WRN("%s: line %d JSON parse error: %s, skipping\n", __func__, lineno, e.what());
        }
    }
    free(line);
    std::fclose(f);
    return out;
}

// Result of tracing one prompt.
struct PromptResult {
    uint64_t prompt_n_tokens = 0;
    uint64_t gen_n_tokens    = 0;
    uint64_t records_written = 0;
    uint64_t records_dropped = 0;
    uint64_t records_sampled = 0;
    double    wall            = 0.0;
};

// Trace a single prompt end-to-end (tokenize -> prefill -> generate), writing
// one JSONL trace file (+ .meta.json) to out_path. The caller owns the model/
// context/sampler and is responsible for clearing the KV cache between prompts.
PromptResult run_one_prompt(
        TraceState & st,
        llama_context * ctx,
        const llama_vocab * vocab,
        llama_sampler * smpl,
        const common_params & params,
        const std::string & model_name,
        Backpressure bp,
        const std::string & run_id,
        const std::string & out_path,
        const PromptSpec & ps) {
    PromptResult res;
    // reset per-prompt counters AND per-prompt metadata in cfg (render_record
    // reads cfg.task_label/language/script/prompt_family/test_id into each record).
    st.cfg.task_label    = ps.task_label;
    st.cfg.language      = ps.language;
    st.cfg.script        = ps.script;
    st.cfg.prompt_family = ps.prompt_family;
    st.cfg.test_id       = ps.test_id;
    st.run_id = run_id;
    st.token_base = 0;
    st.batch_n_tokens = 0;
    st.current_phase = "prefill";
    st.phase_tokens_emitted[0] = 0;
    st.phase_tokens_emitted[1] = 0;
    st.pending_topk.clear();
    st.pending_topk_n_used.clear();
    st.pending_topk_n_tokens.clear();
    // ShortGPT BI: clear the per-token previous-layer residual cache so the
    // next prompt's token 0 doesn't cos-compare against this prompt's final
    // residual (which would be a bogus metric and rot memory across runs).
    st.prev_l_out_per_token.clear();
    st.writer.written.store(0);
    st.writer.dropped.store(0);
    st.writer.sampled.store(0);

    if (!st.writer.open(out_path, bp, /*qsize*/ 8192)) {
        LOG_ERR("ERROR: could not open trace output: %s\n", out_path.c_str());
        return res;
    }

    // Story 8 AC: reset perf counters so this prompt's prefill/gen timings are
    // isolated from prior prompts in batched mode. perf_reset zeroes
    // t_p_eval_us/n_p_eval/t_eval_us/n_eval but preserves t_load_us (model
    // load time, not needed here). After the decode loop we read perf_context
    // to compute per-prompt prompt-eval and gen tok/s.
    llama_perf_context_reset(ctx);

    auto t_start = std::chrono::steady_clock::now();

    const bool add_bos = llama_vocab_get_add_bos(vocab);
    std::vector<llama_token> tokens = common_tokenize(ctx, ps.prompt, add_bos, true);
    if (tokens.empty()) {
        LOG_ERR("%s: no input tokens for test_id=%s\n", __func__, ps.test_id.c_str());
        st.writer.close();
        return res;
    }
    LOG_INF("%s: [%s] prompt tokens = %zu\n", __func__, ps.test_id.c_str(), tokens.size());

    // ---------------- prefill ----------------
    if (st.trace_prefill) {
        st.current_phase = "prefill";
        st.token_base = 0;
        st.batch_n_tokens = (int) tokens.size();
        llama_batch batch = llama_batch_get_one(tokens.data(), (int) tokens.size());
        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("%s: prefill decode failed for test_id=%s\n", __func__, ps.test_id.c_str());
        }
    } else {
        // still seed KV cache for generation
        llama_batch batch = llama_batch_get_one(tokens.data(), (int) tokens.size());
        llama_decode(ctx, batch);
    }

    // ---------------- generation ----------------
    if (st.trace_generation && params.n_predict > 0) {
        st.current_phase = "generation";
        llama_token new_token_id = llama_sampler_sample(smpl, ctx, -1);
        for (int step = 0; step < params.n_predict; ++step) {
            if (llama_vocab_is_eog(vocab, new_token_id)) break;
            if (st.cfg.trace_max_tokens > 0 && st.phase_tokens_emitted[1] >= st.cfg.trace_max_tokens) break;
            st.token_base = (int) tokens.size() + step;
            st.batch_n_tokens = 1;
            llama_batch batch = llama_batch_get_one(&new_token_id, 1);
            int ret = llama_decode(ctx, batch);
            if (ret != 0) {
                LOG_WRN("%s: generation decode failed at step %d for test_id=%s\n",
                        __func__, step, ps.test_id.c_str());
                break;
            }
            ++res.gen_n_tokens;
            new_token_id = llama_sampler_sample(smpl, ctx, -1);
        }
    }
    res.prompt_n_tokens = tokens.size();

    auto t_end = std::chrono::steady_clock::now();
    res.wall = std::chrono::duration<double>(t_end - t_start).count();

    st.writer.close();
    res.records_written  = st.writer.written.load();
    res.records_dropped  = st.writer.dropped.load();
    res.records_sampled = st.writer.sampled.load();

    // ---------------- metadata sidecar ----------------
    std::string meta = out_path + ".meta.json";
    std::FILE * mf = std::fopen(meta.c_str(), "w");
    if (mf) {
        std::fprintf(mf, "{\n");
        std::fprintf(mf, "  \"schema_version\": %d,\n", TRACE_SCHEMA_VERSION);
        std::fprintf(mf, "  \"run_id\": \"%s\",\n", run_id.c_str());
        std::fprintf(mf, "  \"model\": \"%s\",\n", model_name.c_str());
        std::fprintf(mf, "  \"model_path\": \"%s\",\n", params.model.path.c_str());
        // Story 9 AC: real command_line (was "llama-trace-moe ..." placeholder).
        // JSON-escape because argv can contain quotes/backslashes.
        {
            std::string cl_escaped;
            json_escape_append(st.cfg.full_command_line, cl_escaped);
            std::fprintf(mf, "  \"command_line\": \"%s\",\n", cl_escaped.c_str());
        }
        // Story 9 AC: real prompt_sha256 (was "(see run log)" placeholder).
        // Hash the UTF-8 bytes of params.prompt — whether the prompt came from
        // -p or was loaded into params.prompt from -f, this is the actual text
        // the tokenizer saw (verbatim; the tracer does not apply a chat template).
        {
            std::string ph = sha256_hex(params.prompt);
            std::fprintf(mf, "  \"prompt_sha256\": \"%s\",\n", ph.c_str());
        }
        // Story 9 AC: prompt_path when -f <file> was used (single-prompt mode).
        // Batched mode (--trace-prompts) has no single prompt file; field stays null.
        if (!params.prompt_file.empty()) {
            std::string pf_escaped;
            json_escape_append(params.prompt_file, pf_escaped);
            std::fprintf(mf, "  \"prompt_path\": \"%s\",\n", pf_escaped.c_str());
        }
        // Story 9 AC: model_size_bytes + model_sha256_prefix (over first 1 MiB to
        // stay cheap; hashing the full 26 GB shard would dominate a trace run).
        // For multi-shard models (path matches *-of-N.gguf), also glob sibling
        // shards and write model_total_size_bytes so the sidecar reports the
        // full model size (per-shard size like 9.4 MiB looks misleadingly tiny
        // for shard 1, which only carries the GGUF header).
        {
            std::error_code ec;
            auto sz = std::filesystem::file_size(params.model.path, ec);
            if (!ec) {
                std::fprintf(mf, "  \"model_size_bytes\": %llu,\n", (unsigned long long) sz);
            }
            // Glob all sibling shards: same directory, same stem prefix before
            // the final -NNNNN-of-NNNNN suffix. Robust to single-shard models
            // (no glob match → skip field).
            {
                std::filesystem::path p(params.model.path);
                std::string stem = p.stem().string();  // e.g. "GLM-5.2-mixed-00001-of-00009"
                static const std::regex re_of("^(.*?)?-?\\d+-of-\\d+$");
                std::smatch ms;
                if (std::regex_match(stem, ms, re_of) && ms.size() == 2) {
                    std::string base = ms[1].str();
                    // strip trailing dash from base
                    if (!base.empty() && base.back() == '-') base.pop_back();
                    uint64_t total = 0; bool found_any = false;
                    std::error_code ec2;
                    for (auto & entry : std::filesystem::directory_iterator(p.parent_path(), ec2)) {
                        if (ec2) break;
                        if (entry.path().extension() != ".gguf") continue;
                        std::string s = entry.path().stem().string();
                        if (s.rfind(base, 0) != 0) continue;
                        std::error_code ec3;
                        auto esz = std::filesystem::file_size(entry, ec3);
                        if (!ec3) { total += esz; found_any = true; }
                    }
                    if (found_any) {
                        std::fprintf(mf, "  \"model_total_size_bytes\": %llu,\n", (unsigned long long) total);
                    }
                }
            }
            std::ifstream fh(params.model.path, std::ios::binary);
            if (fh) {
                std::vector<uint8_t> head(1u << 20);  // 1 MiB
                fh.read(reinterpret_cast<char *>(head.data()), head.size());
                auto got = fh.gcount();
                if (got > 0) {
                    std::string h = sha256_hex(head.data(), size_t(got));
                    // 16 hex chars = 64-bit prefix; enough for provenance diffing.
                    std::fprintf(mf, "  \"model_sha256_prefix\": \"%s\",\n", h.substr(0, 16).c_str());
                }
            }
        }
        // Story 9 AC: ISO timestamps (UTC). started_at captured once in main();
        // ended_at computed here at meta-write time.
        std::fprintf(mf, "  \"started_at\": \"%s\",\n", st.cfg.started_at_iso.c_str());
        std::fprintf(mf, "  \"ended_at\": \"%s\",\n", iso_utc_now().c_str());
        std::fprintf(mf, "  \"task_label\": \"%s\",\n", ps.task_label.c_str());
        std::fprintf(mf, "  \"language\": \"%s\",\n", ps.language.c_str());
        std::fprintf(mf, "  \"script\": \"%s\",\n", ps.script.c_str());
        std::fprintf(mf, "  \"prompt_family\": \"%s\",\n", ps.prompt_family.c_str());
        std::fprintf(mf, "  \"test_id\": \"%s\",\n", ps.test_id.c_str());
        std::string thinking_mode = "unknown";
        std::string reasoning_effort = "unknown";
        {
            auto it = params.default_template_kwargs.find("enable_thinking");
            if (it != params.default_template_kwargs.end()) {
                thinking_mode = (it->second == "false" || it->second == "0") ? "disabled" : "enabled";
            }
            auto ir = params.default_template_kwargs.find("reasoning_effort");
            if (ir != params.default_template_kwargs.end()) {
                reasoning_effort = ir->second.empty() ? "none" : ir->second;
            }
        }
        std::fprintf(mf, "  \"thinking_mode\": \"%s\",\n", thinking_mode.c_str());
        std::fprintf(mf, "  \"reasoning_effort\": \"%s\",\n", reasoning_effort.c_str());
        std::fprintf(mf, "  \"max_new_tokens\": %d,\n", params.n_predict);
        std::fprintf(mf, "  \"phase_modes\": {\"prefill\": %s, \"generation\": %s},\n",
            st.trace_prefill ? "true" : "false",
            st.trace_generation ? "true" : "false");
        std::fprintf(mf, "  \"trace_layers\": \"%s\",\n",
            st.cfg.trace_layers.empty() ? "all" : st.cfg.trace_layers.c_str());
        std::fprintf(mf, "  \"trace_max_tokens\": %d,\n", st.cfg.trace_max_tokens);
        std::fprintf(mf, "  \"backpressure\": \"%s\",\n", st.cfg.backpressure.c_str());
        // Story 6 AC: activation-summary sidecar fields. Absent (omitted) when
        // --trace-activations was not passed, so analyzer/report can detect.
        if (!st.activation_stems.empty()) {
            std::string joined_stems;
            for (size_t i = 0; i < st.activation_stems.size(); ++i) {
                if (i) joined_stems += ",";
                joined_stems += st.activation_stems[i];
            }
            std::fprintf(mf, "  \"activation_stems\": \"%s\",\n", joined_stems.c_str());
            std::fprintf(mf, "  \"activation_topk\": %d,\n", st.activation_topk);
            std::fprintf(mf, "  \"activation_stride\": %d,\n", st.activation_stride);
        }
        std::fprintf(mf, "  \"queue_size\": %zu,\n", st.writer.max_queue);
        std::fprintf(mf, "  \"records_written\": %llu,\n", (unsigned long long) res.records_written);
        std::fprintf(mf, "  \"records_dropped\": %llu,\n", (unsigned long long) res.records_dropped);
        std::fprintf(mf, "  \"records_sampled\": %llu,\n", (unsigned long long) res.records_sampled);
        std::fprintf(mf, "  \"prompt_token_count\": %llu,\n", (unsigned long long) res.prompt_n_tokens);
        std::fprintf(mf, "  \"gen_token_count\": %llu,\n", (unsigned long long) res.gen_n_tokens);
        // Story 8 AC: per-prompt speed metrics from llama_perf_context.
        // t_p_eval_ms is total pref-time for this prompt; t_eval_ms is total
        // gen-time. n_p_eval is prompt tokens processed; n_eval is tokens
        // generated. tok/s = tokens / (ms/1000) = tokens*1000/ms.
        {
            llama_perf_context_data perf = llama_perf_context(ctx);
            double p_per_sec = (perf.t_p_eval_ms > 0.0)
                ? (double) perf.n_p_eval * 1000.0 / perf.t_p_eval_ms
                : 0.0;
            double g_per_sec = (perf.t_eval_ms   > 0.0)
                ? (double) perf.n_eval   * 1000.0 / perf.t_eval_ms
                : 0.0;
            std::fprintf(mf, "  \"perf_prompt_eval_per_sec\": %.4f,\n", p_per_sec);
            std::fprintf(mf, "  \"perf_gen_per_sec\": %.4f,\n", g_per_sec);
            std::fprintf(mf, "  \"perf_prompt_eval_ms\": %.3f,\n", perf.t_p_eval_ms);
            std::fprintf(mf, "  \"perf_eval_ms\": %.3f,\n", perf.t_eval_ms);
            std::fprintf(mf, "  \"perf_n_prompt_eval\": %d,\n", (int) perf.n_p_eval);
            std::fprintf(mf, "  \"perf_n_eval\": %d,\n", (int) perf.n_eval);
        }
        // Story 8 AC: write n_expert_total to the sidecar too so the analyzer
        // can label experts as #X of N total even if records themselves don't
        // carry it (e.g. dense models, or older trace files).
        if (st.n_expert_total > 0) {
            std::fprintf(mf, "  \"n_expert_total\": %d,\n", st.n_expert_total);
        }
        std::fprintf(mf, "  \"wall_seconds\": %.3f\n", res.wall);
        std::fprintf(mf, "}\n");
        std::fclose(mf);
    }

    LOG("%s: [%s] trace written to %s (%llu records, %llu dropped, %llu sampled, %.2fs)\n",
        __func__, ps.test_id.c_str(), out_path.c_str(),
        (unsigned long long) res.records_written,
        (unsigned long long) res.records_dropped,
        (unsigned long long) res.records_sampled,
        res.wall);
    return res;
}

}  // namespace

// ---------------------------------------------------------------------------
// Story 8 AC: read n_expert_total from the GGUF file's metadata.
//
// There is no public llama API for "how many experts per layer". The
// hparams field (hparams.n_expert) is private/experimental. But the GGUF KV
// "<arch>.expert_count" is always written by llama_model_saver and is unique
// per model (one arch per file). We scan all KV keys for the suffix
// ".expert_count" — there is exactly one such key per multi-arch GGUF — and
// read its u32 value. Returns 0 if the file can't be opened or the key is
// absent (older sidecars / dense models). For GLM-5.2 this is 256, matching
// the verified expert ID range 0..255 observed in real traces.
// ---------------------------------------------------------------------------
static int read_n_expert_total_from_gguf(const std::string & model_path) {
    gguf_init_params params = {};
    params.no_alloc = true;  // metadata only; don't load tensor data
    params.ctx      = nullptr;
    gguf_context * gctx = gguf_init_from_file(model_path.c_str(), params);
    if (!gctx) return 0;
    int found = 0;
    const int64_t n_kv = gguf_get_n_kv(gctx);
    for (int64_t i = 0; i < n_kv; ++i) {
        const char * key = gguf_get_key(gctx, i);
        if (!key) continue;
        std::string ks(key);
        // Suffix match — every arch writes "<arch>.expert_count".
        static const std::string suffix = ".expert_count";
        if (ks.size() > suffix.size() &&
            ks.compare(ks.size() - suffix.size(), suffix.size(), suffix) == 0) {
            found = (int) gguf_get_val_u32(gctx, i);
            break;
        }
    }
    gguf_free(gctx);
    return found;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    // Pre-scan to peel --trace-* off argv before common_params_parse sees them.
    std::vector<std::string> raw_args(argv, argv + argc);
    char ** argv_kept = argv;
    int argc_kept = argc;
    TraceConfig cfg = config_from_trace_flags(raw_args, argc_kept, argv_kept);

    if (cfg.trace_out.empty()) {
        LOG_ERR("ERROR: --trace-out <path.jsonl> is required\n");
        return 2;
    }

    common_params params;
    common_init();

    // Reproducibility provenance (Story 9 AC): capture the full command line
    // (rejoined from argv) and the process start time. These are written into
    // every per-prompt .meta.json sidecar so a trace can be reproduced.
    {
        std::string cl;
        for (int i = 0; i < argc_kept; i++) {
            if (i > 0) cl += " ";
            cl += argv_kept[i];
        }
        cfg.full_command_line = cl;
        cfg.started_at_iso    = iso_utc_now();
    }

    if (!common_params_parse(argc_kept, argv_kept, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    if (!cfg.stripped_chat_flags.empty()) {
        LOG_WRN("%s: ignoring chat-template flags %s (the tracer tokenizes "
                "params.prompt verbatim; it does not apply a chat template). "
                "For templated prompts, pre-template and pass via -p. "
                "See traces/README.md.",
                __func__, join(cfg.stripped_chat_flags, ", ").c_str());
    }
    if (cfg.trace_prompts.empty()) {
        if (params.prompt.empty() && params.in_files.empty()) {
            LOG_ERR("ERROR: provide -p <prompt> or -f <file>, or --trace-prompts <file.jsonl>\n");
            return 2;
        }
    }

    // backpressure
    Backpressure bp = Backpressure::Sample;
    if (cfg.backpressure == "block") bp = Backpressure::Block;
    else if (cfg.backpressure == "drop") bp = Backpressure::Drop;
    else bp = Backpressure::Sample;

    TraceState st;
    st.cfg = cfg;
    const std::string model_name = std::filesystem::path(params.model.path).parent_path().filename().string();
    st.model_name = model_name.empty() ? params.model.path : model_name;
    st.max_new_tokens = params.n_predict;
    st.selected_layers = parse_layers(cfg.trace_layers);
    st.trace_prefill = (cfg.phase == "prefill" || cfg.phase == "both");
    st.trace_generation = (cfg.phase == "generation" || cfg.phase == "both");
    // Story 6 AC: parse activation-stem list. Comma-separated, e.g.
    // "l_out,kqv_out,ffn_out". Empty string → no activation tracers.
    if (!cfg.trace_activations.empty()) {
        std::string s = cfg.trace_activations;
        size_t pos;
        while ((pos = s.find(',')) != std::string::npos) {
            std::string tok = s.substr(0, pos);
            // trim whitespace
            while (!tok.empty() && std::isspace((unsigned char) tok.front())) tok.erase(0, 1);
            while (!tok.empty() && std::isspace((unsigned char) tok.back())) tok.pop_back();
            if (!tok.empty()) st.activation_stems.push_back(tok);
            s.erase(0, pos + 1);
        }
        while (!s.empty() && std::isspace((unsigned char) s.front())) s.erase(0, 1);
        while (!s.empty() && std::isspace((unsigned char) s.back())) s.pop_back();
        if (!s.empty()) st.activation_stems.push_back(s);
        st.activation_topk = cfg.trace_activation_topk > 0 ? cfg.trace_activation_topk : 10;
        st.activation_stride = cfg.trace_activation_stride > 0 ? cfg.trace_activation_stride : 2;
        LOG_INF("%s: activation tracing on — stems=[", __func__);
        for (size_t i = 0; i < st.activation_stems.size(); ++i) {
            if (i) LOG_INF(",");
            LOG_INF("%s", st.activation_stems[i].c_str());
        }
        LOG_INF("] topk=%d stride=%d\n", st.activation_topk, st.activation_stride);
    }

    params.cb_eval = trace_cb_eval;
    params.cb_eval_user_data = &st;
    params.warmup = false;

    // ---- load model ONCE (batched mode traces N prompts against this one model) ----
    auto t_total_start = std::chrono::steady_clock::now();
    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();
    if (!model || !ctx) {
        LOG_ERR("ERROR: failed to init llama\n");
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    (void) model;

    // Story 8 AC: populate n_expert_total once per model (it's a global KV —
    // all MoE layers in one GGUF share the same expert_count). render_record
    // emits "n_expert":N per routing event when this is >0, so the analyzer
    // can label experts as #X of N total. For GLM-5.2 this is 256.
    st.n_expert_total = read_n_expert_total_from_gguf(params.model.path);
    if (st.n_expert_total > 0) {
        LOG_INF("%s: n_expert_total = %d (from GGUF KV)", __func__, st.n_expert_total);
    }

    // ---- sampler for generation ----
    struct llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(params.sampling.temp));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    // ---- build prompt spec list ----
    // Batched mode (--trace-prompts): each JSONL line is one prompt with metadata.
    // Single mode: synthesize one spec from params.prompt + cfg defaults.
    const bool batched = !cfg.trace_prompts.empty();
    std::vector<PromptSpec> specs;
    if (batched) {
        specs = load_prompt_specs(cfg.trace_prompts);
        if (specs.empty()) {
            LOG_ERR("ERROR: no prompts parsed from %s\n", cfg.trace_prompts.c_str());
            llama_sampler_free(smpl);
            llama_backend_free();
            return 1;
        }
        // fill spec defaults from cfg where the JSONL entry omitted them
        for (auto & ps : specs) {
            if (ps.task_label.empty())    ps.task_label    = cfg.task_label;
            if (ps.language.empty())      ps.language      = cfg.language;
            if (ps.script.empty())        ps.script        = cfg.script;
            if (ps.prompt_family.empty()) ps.prompt_family = cfg.prompt_family;
            if (ps.test_id.empty())       ps.test_id       = cfg.test_id;
        }
    } else {
        PromptSpec ps;
        ps.prompt        = params.prompt;
        ps.task_label    = cfg.task_label;
        ps.language      = cfg.language;
        ps.script        = cfg.script;
        ps.prompt_family = cfg.prompt_family;
        ps.test_id       = cfg.test_id;
        specs.push_back(std::move(ps));
    }

    // ---- output path(s) ----
    // Batched: --trace-out is an output directory; each prompt writes
    // <dir>/<test_id>.jsonl (+ .meta.json). Single: --trace-out is the file.
    std::string out_dir;
    if (batched) {
        out_dir = cfg.trace_out;
        std::error_code ec;
        std::filesystem::create_directories(out_dir, ec);
        if (ec) {
            LOG_ERR("ERROR: could not create output dir %s: %s\n", out_dir.c_str(), ec.message().c_str());
            llama_sampler_free(smpl);
            llama_backend_free();
            return 2;
        }
    }

    // ---- run ----
    const std::string ts = std::to_string((long long) std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()));
    uint64_t total_records = 0, total_dropped = 0, total_sampled = 0;
    for (size_t i = 0; i < specs.size(); ++i) {
        PromptSpec & ps = specs[i];
        std::string out_path;
        if (batched) {
            // include language in the filename: the suite reuses test_id across
            // languages (same prompt translated), so test_id alone would collide.
            std::string fname = ps.test_id.empty() ? ("prompt_" + std::to_string(i)) : ps.test_id;
            if (!ps.language.empty()) fname += "-" + ps.language;
            out_path = out_dir + "/" + fname + ".jsonl";
        } else {
            out_path = cfg.trace_out;
        }
        std::string run_id = ps.test_id + "-" + ps.language + "-" + ts;

        PromptResult r = run_one_prompt(st, ctx, vocab, smpl, params, model_name, bp, run_id, out_path, ps);
        total_records  += r.records_written;
        total_dropped  += r.records_dropped;
        total_sampled += r.records_sampled;

        // clear KV cache between prompts so each starts with a clean slate
        llama_memory_clear(llama_get_memory(ctx), true);
    }

    llama_sampler_free(smpl);

    auto t_total_end = std::chrono::steady_clock::now();
    double wall_total = std::chrono::duration<double>(t_total_end - t_total_start).count();

    if (batched) {
        LOG("%s: batched %zu prompts -> %s (%llu total records, %llu dropped, %llu sampled, %.2fs total)\n",
            __func__, specs.size(), cfg.trace_out.c_str(),
            (unsigned long long) total_records,
            (unsigned long long) total_dropped,
            (unsigned long long) total_sampled,
            wall_total);
    }

    llama_backend_free();
    return 0;
}
