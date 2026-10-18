// next-layer expert prediction (src/llama-moe-predict.*): selection rule, hit accounting, adaptation of the number
// of guesses, popularity, work queue priority and stale guesses, persistence of the learned state, the byte
// accounting of the loads (hit / pending / credited / wasted / late), and llama_decode_progress
//
// test-moe-predict --progress <kimi-k3 test model>: only the decode progress test (needs a generated model)

#include "../src/llama-graph.h"
#include "../src/llama-moe-predict.h"

#include "ggml.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <numeric>
#include <set>
#include <string>
#include <utility>
#include <vector>

static int n_fail = 0;

#define CHECK(cond, ...) do {                                   \
    if (!(cond)) {                                              \
        fprintf(stderr, "FAIL %s:%d: %s: ", __FILE__, __LINE__, #cond); \
        fprintf(stderr, __VA_ARGS__);                           \
        fprintf(stderr, "\n");                                  \
        n_fail++;                                               \
    }                                                           \
} while (0)

static constexpr int N_LAYER  = 4;  // layer 0 dense, 1..3 MoE
static constexpr int N_EXPERT = 64;
static constexpr int N_USED   = 4;

struct fixture {
    ggml_context * ctx = nullptr;
    std::vector<float> mem;
    std::vector<llama_moe_predictor::layer_tensors> lt;

    fixture() {
        ggml_init_params ip = { 16*ggml_tensor_overhead(), nullptr, true };
        ctx = ggml_init(ip);
        lt.resize(N_LAYER);
        // small expert tensors backed by heap memory: the hints are harmless there
        mem.assign((size_t) N_LAYER*3*N_EXPERT*64, 0.0f);
        size_t off = 0;
        for (int il = 1; il < N_LAYER; ++il) {
            for (int k = 0; k < 3; ++k) {
                ggml_tensor * t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 8, 8, N_EXPERT);
                t->data = mem.data() + off;
                off += 64*N_EXPERT;
                lt[il].exps[k] = t;
            }
        }
    }
    ~fixture() { ggml_free(ctx); }
};

static llama_moe_predictor::params base_params(int mode) {
    llama_moe_predictor::params p;
    p.mode     = mode;
    p.n_layer  = N_LAYER;
    p.n_expert = N_EXPERT;
    p.n_used   = N_USED;
    p.model_id = 1234;
    return p;
}

// logits that rank the experts in `order` first (descending), then the rest
static std::vector<float> logits_for(const std::vector<int> & order) {
    std::vector<float> l(N_EXPERT, -10.0f);
    for (size_t i = 0; i < order.size(); ++i) {
        l[order[i]] = 5.0f - 0.1f*(float) i;
    }
    return l;
}

// expert at router rank r for the logits from logits_for(identity order)
static std::vector<int> identity_order() {
    std::vector<int> o(N_EXPERT);
    std::iota(o.begin(), o.end(), 0);
    return o;
}

// one decode step through MoE layers 1..3: layer il selects sel[il], and predicts il+1 from logits pred[il]
static void step(llama_moe_predictor & mp, const std::vector<std::vector<int>> & sel, const std::vector<float> & pred) {
    for (int il = 1; il < N_LAYER; ++il) {
        std::vector<int32_t> s(sel[il].begin(), sel[il].end());
        const int il_next = il + 1 < N_LAYER ? il + 1 : -1;
        mp.on_layer(il, s.data(), sizeof(int32_t), N_USED*sizeof(int32_t), 1,
                il_next, il_next >= 0 ? pred.data() : nullptr, N_EXPERT*sizeof(float), nullptr);
    }
}

static void test_rank_router() {
    std::vector<float> logits(N_EXPERT), bias(N_EXPERT);
    for (int e = 0; e < N_EXPERT; ++e) {
        logits[e] = std::sin(0.7f*e)*3.0f;
        bias[e]   = 0.01f*(e % 7);
    }
    std::vector<int> top;
    llama_moe_predictor::rank_router(logits.data(), bias.data(), N_EXPERT, 10, top);

    std::vector<int> ref(N_EXPERT);
    std::iota(ref.begin(), ref.end(), 0);
    auto score = [&](int e) { return 1.0f/(1.0f + std::exp(-logits[e])) + bias[e]; };
    std::stable_sort(ref.begin(), ref.end(), [&](int a, int b) { return score(a) > score(b); });
    ref.resize(10);
    CHECK(top == ref, "router ranking differs from sigmoid + bias order");
}

static void test_hits_and_growth() {
    fixture fx;
    llama_moe_predictor mp(base_params(llama_moe_predictor::MODE_OBSERVE), fx.lt);
    const auto pred = logits_for(identity_order());

    // each step: 2 experts at router ranks 0,1 and 2 at ranks 16,17 -> the 4 slots after n_r = 16 are 50% useful
    // (layer 2 is scored from the first step on: layer 1 predicts it earlier in the same step)
    std::vector<std::vector<int>> sel(N_LAYER, { 0, 1, 16, 17 });
    for (int i = 0; i < 16; ++i) {
        step(mp, sel, pred);
    }
    auto v = mp.view(2);
    CHECK(v.n_r == 20, "n_r should grow to 20 when ranks 16-19 are 50%% useful, got %d", v.n_r);
    CHECK(v.n_p == 0,  "n_p should stay 0, got %d", v.n_p);
    CHECK(std::fabs(v.recall - 0.5) < 1e-9, "recall with n_r 16 should be 0.5, got %f", v.recall);

    // with n_r 20 the same selection is fully predicted
    for (int i = 0; i < 16; ++i) {
        step(mp, sel, pred);
    }
    v = mp.view(2);
    CHECK(v.n_r == 20, "n_r should stay 20 (outer slots 50%% useful, next ones unused), got %d", v.n_r);
    CHECK(v.recall > 0.74 && v.recall < 0.76, "lifetime recall should be (16*0.5 + 16*1)/32 = 0.75, got %f", v.recall);

    // only ranks 0..3 used: the outermost slots are useless -> shrink back to the minimum
    std::vector<std::vector<int>> sel2(N_LAYER, { 0, 1, 2, 3 });
    for (int i = 0; i < 16; ++i) {
        step(mp, sel2, pred);
    }
    CHECK(mp.view(2).n_r == 16, "n_r should shrink to 16, got %d", mp.view(2).n_r);
}

static void test_popularity() {
    fixture fx;
    llama_moe_predictor mp(base_params(llama_moe_predictor::MODE_OBSERVE), fx.lt);
    const auto pred = logits_for(identity_order());

    // the router never ranks experts 60..63 in its top 16+4, but they are always selected
    std::vector<std::vector<int>> sel(N_LAYER, { 60, 61, 62, 63 });
    for (int i = 0; i < 1 + 16; ++i) {
        step(mp, sel, pred);
    }
    auto v = mp.view(2);
    CHECK(v.n_p == 4, "popularity slots should grow to 4, got %d", v.n_p);
    for (int i = 0; i < 16; ++i) {
        step(mp, sel, pred);
    }
    v = mp.view(2);
    CHECK(v.recall > 0.45, "recall should rise once popularity slots exist, got %f", v.recall);
    CHECK(v.n_r + v.n_p <= 32, "cap exceeded: %d + %d", v.n_r, v.n_p);

    // once only router guesses are selected, the popularity slots are useless and go away
    std::vector<std::vector<int>> top(N_LAYER, { 0, 1, 2, 3 });
    for (int i = 0; i < 32; ++i) {
        step(mp, top, pred);
    }
    CHECK(mp.view(2).n_p == 0, "useless popularity slots should shrink to 0, got %d", mp.view(2).n_p);
}

// hits of layer 2 so far (recall * actual, with N_USED actual experts per single-token step)
static double hits_l2(const llama_moe_predictor & mp) {
    const auto v = mp.view(2);
    return v.recall*N_USED*(double) v.tokens;
}

static void test_cap() {
    fixture fx;
    auto p = base_params(llama_moe_predictor::MODE_OBSERVE);
    p.cap = 20;
    llama_moe_predictor mp(p, fx.lt);
    const auto pred = logits_for(identity_order());

    // grow to the cap
    std::vector<std::vector<int>> sel(N_LAYER, { 0, 1, 16, 17 });
    for (int i = 0; i < 16; ++i) {
        step(mp, sel, pred);
    }
    CHECK(mp.view(2).n_r == 20, "n_r should grow to the cap 20, got %d", mp.view(2).n_r);

    // outer router slots 50% useful (keep), next router and popularity slots 50% useful (would grow): the cap holds
    std::vector<std::vector<int>> sel2(N_LAYER, { 16, 17, 20, 21 });
    for (int i = 0; i < 16; ++i) {
        step(mp, sel2, pred);
    }
    CHECK(mp.view(2).n_r == 20 && mp.view(2).n_p == 0, "cap 20: expected n_r 20, n_p 0, got %d, %d",
            mp.view(2).n_r, mp.view(2).n_p);
}

static void test_shrink() {
    fixture fx;
    llama_moe_predictor mp(base_params(llama_moe_predictor::MODE_OBSERVE), fx.lt);
    const auto pred = logits_for(identity_order());

    std::vector<std::vector<int>> grow(N_LAYER, { 0, 1, 16, 17 });
    for (int i = 0; i < 16; ++i) {
        step(mp, grow, pred);
    }
    CHECK(mp.view(2).n_r == 20, "setup: n_r should be 20, got %d", mp.view(2).n_r);

    // outermost router slots (ranks 16-19) 25% useful: below the 0.3 shrink threshold
    std::vector<std::vector<int>> quarter(N_LAYER, { 0, 1, 2, 16 });
    for (int i = 0; i < 16; ++i) {
        step(mp, quarter, pred);
    }
    CHECK(mp.view(2).n_r == 16, "25%% useful outer slots should shrink n_r to 16, got %d", mp.view(2).n_r);

    // never below the minimum, however useless the outer slots are
    std::vector<std::vector<int>> top(N_LAYER, { 0, 1, 2, 3 });
    for (int i = 0; i < 32; ++i) {
        step(mp, top, pred);
    }
    CHECK(mp.view(2).n_r == 16, "n_r must not drop below 16, got %d", mp.view(2).n_r);
}

static void test_multi_token() {
    fixture fx;
    llama_moe_predictor mp(base_params(llama_moe_predictor::MODE_OBSERVE), fx.lt);
    const auto ident = logits_for(identity_order());

    // layer 1 predicts layer 2: router top 16 = experts 0..15
    std::vector<int32_t> s1 = { 0, 1, 2, 3 };
    mp.on_layer(1, s1.data(), sizeof(int32_t), N_USED*sizeof(int32_t), 1, 2, ident.data(), N_EXPERT*sizeof(float), nullptr);

    // layer 2 runs 2 tokens: {0..3} and {16..19} -> 8 actual experts, 4 predicted
    // its prediction for layer 3 uses each token's own logits: token 1 ranks 60..63 first
    std::vector<int32_t> s2 = { 0, 1, 2, 3, 16, 17, 18, 19 };
    std::vector<int> o1 = { 60, 61, 62, 63 };
    for (int e = 0; e < N_EXPERT; ++e) {
        if (e < 60) { o1.push_back(e); }
    }
    std::vector<float> l2 = ident;
    const auto l2b = logits_for(o1);
    l2.insert(l2.end(), l2b.begin(), l2b.end());
    mp.on_layer(2, s2.data(), sizeof(int32_t), N_USED*sizeof(int32_t), 2, 3, l2.data(), N_EXPERT*sizeof(float), nullptr);
    CHECK(mp.view(2).tokens == 2, "2 tokens expected, got %llu", (unsigned long long) mp.view(2).tokens);
    CHECK(std::fabs(mp.view(2).recall - 0.5) < 1e-9, "union of both tokens: recall 4/8 expected, got %f", mp.view(2).recall);

    // layer 3 selects 60..63, which only token 1's logits put in front
    std::vector<int32_t> s3 = { 60, 61, 62, 63 };
    mp.on_layer(3, s3.data(), sizeof(int32_t), N_USED*sizeof(int32_t), 1, -1, nullptr, 0, nullptr);
    CHECK(std::fabs(mp.view(3).recall - 1.0) < 1e-9, "token 1's logits should predict layer 3, recall %f", mp.view(3).recall);
}

static void test_fresh_prediction_each_step() {
    fixture fx;
    llama_moe_predictor mp(base_params(llama_moe_predictor::MODE_OBSERVE), fx.lt);
    const auto ident = logits_for(identity_order());
    std::vector<int> rev(N_EXPERT);
    for (int e = 0; e < N_EXPERT; ++e) {
        rev[e] = N_EXPERT - 1 - e;
    }
    const auto revl = logits_for(rev);

    std::vector<int32_t> s1 = { 0, 1, 2, 3 };
    std::vector<int32_t> s2 = { 0, 1, 2, 3 };
    // step A: predicted 0..15, selected 0..3 -> 4 hits
    mp.on_layer(1, s1.data(), sizeof(int32_t), N_USED*sizeof(int32_t), 1, 2, ident.data(), N_EXPERT*sizeof(float), nullptr);
    mp.on_layer(2, s2.data(), sizeof(int32_t), N_USED*sizeof(int32_t), 1, -1, nullptr, 0, nullptr);
    // step B: predicted 63..48, selected 0..3 -> 0 hits (ranks of step A must not linger)
    mp.on_layer(1, s1.data(), sizeof(int32_t), N_USED*sizeof(int32_t), 1, 2, revl.data(), N_EXPERT*sizeof(float), nullptr);
    mp.on_layer(2, s2.data(), sizeof(int32_t), N_USED*sizeof(int32_t), 1, -1, nullptr, 0, nullptr);
    CHECK(std::fabs(mp.view(2).recall - 0.5) < 1e-9, "recall 4/8 expected, got %f", mp.view(2).recall);

    // layer 2 again without a new prediction: not scored a second time
    std::vector<int32_t> s3 = { 60, 61, 62, 63 };
    mp.on_layer(2, s3.data(), sizeof(int32_t), N_USED*sizeof(int32_t), 1, -1, nullptr, 0, nullptr);
    CHECK(mp.view(2).tokens == 2, "a prediction must be scored once, tokens %llu", (unsigned long long) mp.view(2).tokens);
}

static void test_popularity_decay_and_exclusion() {
    fixture fx;
    const auto pred = logits_for(identity_order());

    // decay: a long-gone favourite must give way to the current one
    // (cap 20 = 16 router + 4 popularity slots, so growing cannot cover both sets)
    {
        auto p = base_params(llama_moe_predictor::MODE_OBSERVE);
        p.cap = 20;
        llama_moe_predictor mp(p, fx.lt);
        std::vector<std::vector<int>> old_sel(N_LAYER, { 60, 61, 62, 63 });
        std::vector<std::vector<int>> new_sel(N_LAYER, { 50, 51, 52, 53 });
        for (int i = 0; i < 200; ++i) { step(mp, old_sel, pred); }
        for (int i = 0; i < 100; ++i) { step(mp, new_sel, pred); }
        const double h0 = hits_l2(mp);
        for (int i = 0; i < 16; ++i) { step(mp, new_sel, pred); }
        const double h = hits_l2(mp) - h0;
        CHECK(h > 0.9*16*4, "with decay the recent experts should be the popular guesses: %.1f of 64 hit", h);
    }

    // exclusion: popular experts the router already guesses must not take popularity slots
    {
        llama_moe_predictor mp(base_params(llama_moe_predictor::MODE_OBSERVE), fx.lt);
        std::vector<std::vector<int>> a(N_LAYER, { 0, 1, 2, 3 });
        std::vector<std::vector<int>> b(N_LAYER, { 0, 1, 60, 61 });
        for (int i = 0; i < 100; ++i) { step(mp, a, pred); }
        for (int i = 0; i < 32; ++i) { step(mp, b, pred); }
        CHECK(mp.view(2).n_p == 4, "60, 61 should earn popularity slots, n_p %d", mp.view(2).n_p);
        const double h0 = hits_l2(mp);
        for (int i = 0; i < 16; ++i) { step(mp, b, pred); }
        const double h = hits_l2(mp) - h0;
        CHECK(h > 0.99*16*4, "all of 0, 1, 60, 61 should be guessed: %.1f of 64 hit", h);
    }
}

static void test_queue_order_and_stale() {
    fixture fx;
    auto p = base_params(llama_moe_predictor::MODE_PREFETCH);
    p.paused = true;
    p.trace  = true;
    llama_moe_predictor mp(p, fx.lt);
    const auto pred = logits_for(identity_order());

    // layer 1 runs: its 4 actual experts go HIGH, 16 guesses for layer 2 go LOW
    std::vector<int32_t> s1 = { 0, 1, 2, 3 };
    mp.on_layer(1, s1.data(), sizeof(int32_t), N_USED*sizeof(int32_t), 1, 2, pred.data(), N_EXPERT*sizeof(float), nullptr);
    mp.resume();
    mp.wait_idle();
    auto tr = mp.get_trace();
    size_t first_low = tr.size();
    for (size_t i = 0; i < tr.size(); ++i) {
        if (tr[i] < 0) { first_low = std::min(first_low, i); }
    }
    size_t n_high_after_low = 0;
    for (size_t i = first_low; i < tr.size(); ++i) {
        n_high_after_low += tr[i] > 0;
    }
    CHECK(n_high_after_low == 0, "a HIGH expert was processed after a LOW guess");
    auto c = mp.get_counters();
    CHECK(c.high_done == 4, "expected 4 HIGH experts, got %llu", (unsigned long long) c.high_done);
    CHECK(c.low_done == 16, "expected 16 LOW guesses processed, got %llu", (unsigned long long) c.low_done);

    // stale guesses: layer 2 starts before the worker gets to its guesses -> all dropped
    llama_moe_predictor::params p2 = p;
    llama_moe_predictor mp2(p2, fx.lt);
    std::vector<int32_t> s2 = { 4, 5, 6, 7 };
    mp2.on_layer(1, s1.data(), sizeof(int32_t), N_USED*sizeof(int32_t), 1, 2, pred.data(), N_EXPERT*sizeof(float), nullptr);
    mp2.on_layer(2, s2.data(), sizeof(int32_t), N_USED*sizeof(int32_t), 1, 3, pred.data(), N_EXPERT*sizeof(float), nullptr);
    mp2.resume();
    mp2.wait_idle();
    auto c2 = mp2.get_counters();
    // the guesses for layer 2 are stale (layer 2 has started), those for layer 3 are not
    CHECK(c2.low_drops == 16, "expected the 16 guesses for layer 2 to be dropped, got %llu", (unsigned long long) c2.low_drops);
    CHECK(c2.low_done == 16, "expected the 16 guesses for layer 3 to be processed, got %llu", (unsigned long long) c2.low_done);
    // layer 1 has run by then too: its actual experts are dropped, layer 2's are issued
    CHECK(c2.high_done == 4, "expected the 4 HIGH experts of layer 2, got %llu", (unsigned long long) c2.high_done);
    CHECK(c2.high_drops == 4, "expected the 4 HIGH experts of layer 1 to be dropped, got %llu", (unsigned long long) c2.high_drops);
}

static void test_startup_probe_ignored() {
    fixture fx;
    auto p = base_params(llama_moe_predictor::MODE_PREFETCH);
    p.paused = true;
    llama_moe_predictor mp(p, fx.lt);
    const auto pred = logits_for(identity_order());
    const uint32_t cold = mp.checksum();

    // a 2-token step before any single-token step: the server's init probe, ignored entirely
    std::vector<int32_t> s2 = { 0, 0, 0, 0, 0, 0, 0, 0 };
    for (int il = 1; il < N_LAYER; ++il) {
        mp.on_layer(il, s2.data(), sizeof(int32_t), N_USED*sizeof(int32_t), 2, il + 1 < N_LAYER ? il + 1 : -1,
                pred.data(), N_EXPERT*sizeof(float), nullptr);
    }
    CHECK(mp.view(2).tokens == 0, "probe tokens must not be counted, got %llu", (unsigned long long) mp.view(2).tokens);
    CHECK(mp.checksum() == cold, "probe must not change the state");
    mp.resume();
    mp.wait_idle();
    const auto c = mp.get_counters();
    CHECK(c.high_done + c.low_done + c.high_drops + c.low_drops == 0, "probe must not enqueue prefetch work");

    // a prompt of more than 4 tokens always counts, but does not unlock the small batches
    std::vector<int32_t> s5(5*N_USED, 1);
    for (int il = 1; il < N_LAYER; ++il) {
        mp.on_layer(il, s5.data(), sizeof(int32_t), N_USED*sizeof(int32_t), 5, il + 1 < N_LAYER ? il + 1 : -1,
                pred.data(), N_EXPERT*sizeof(float), nullptr);
    }
    CHECK(mp.view(2).tokens == 5, "a 5-token prompt must count, got %llu", (unsigned long long) mp.view(2).tokens);
    for (int il = 1; il < N_LAYER; ++il) {
        mp.on_layer(il, s2.data(), sizeof(int32_t), N_USED*sizeof(int32_t), 2, il + 1 < N_LAYER ? il + 1 : -1,
                pred.data(), N_EXPERT*sizeof(float), nullptr);
    }
    CHECK(mp.view(2).tokens == 5, "a 2-token step is still ignored before any single-token step, got %llu",
            (unsigned long long) mp.view(2).tokens);

    // a single-token step unlocks; a later 2-token step (two sequences) counts
    std::vector<std::vector<int>> sel(N_LAYER, { 0, 1, 2, 3 });
    step(mp, sel, pred);
    for (int il = 1; il < N_LAYER; ++il) {
        mp.on_layer(il, s2.data(), sizeof(int32_t), N_USED*sizeof(int32_t), 2, il + 1 < N_LAYER ? il + 1 : -1,
                pred.data(), N_EXPERT*sizeof(float), nullptr);
    }
    CHECK(mp.view(2).tokens == 8, "5 + 1 + 2 tokens expected after unlocking, got %llu", (unsigned long long) mp.view(2).tokens);
    mp.wait_idle();
}

static void test_queue_limits() {
    fixture fx;
    const auto pred = logits_for(identity_order());
    std::vector<int32_t> s = { 0, 1, 2, 3 };

    // bounded LOW queue: 300 x 16 guesses for layer 3 while paused, at most 4096 kept
    {
        auto p = base_params(llama_moe_predictor::MODE_PREFETCH);
        p.paused = true;
        llama_moe_predictor mp(p, fx.lt);
        for (int i = 0; i < 300; ++i) {
            mp.on_layer(2, s.data(), sizeof(int32_t), N_USED*sizeof(int32_t), 1, 3, pred.data(), N_EXPERT*sizeof(float), nullptr);
        }
        mp.resume();
        mp.wait_idle();
        const auto c = mp.get_counters();
        CHECK(c.low_done == 4096, "4096 guesses kept, got %llu", (unsigned long long) c.low_done);
        CHECK(c.low_drops == 300*16 - 4096, "the rest dropped, got %llu", (unsigned long long) c.low_drops);
    }

    // guesses from an older step are dropped even for a layer the new step has not reached
    {
        auto p = base_params(llama_moe_predictor::MODE_PREFETCH);
        p.paused = true;
        llama_moe_predictor mp(p, fx.lt);
        mp.on_layer(2, s.data(), sizeof(int32_t), N_USED*sizeof(int32_t), 1, 3, pred.data(), N_EXPERT*sizeof(float), nullptr);
        mp.on_layer(1, s.data(), sizeof(int32_t), N_USED*sizeof(int32_t), 1, 2, pred.data(), N_EXPERT*sizeof(float), nullptr);
        mp.resume();
        mp.wait_idle();
        const auto c = mp.get_counters();
        CHECK(c.low_drops == 16, "the 16 guesses of the older step dropped, got %llu", (unsigned long long) c.low_drops);
        CHECK(c.low_done  == 16, "the 16 guesses of the new step processed, got %llu", (unsigned long long) c.low_done);
    }
}

static uint32_t fnv1a(const std::vector<char> & b, size_t n) {
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < n; ++i) {
        h ^= (uint8_t) b[i];
        h *= 16777619u;
    }
    return h;
}

static std::string tmp_path(const char * name) {
    const char * d = getenv("TMPDIR");
    return std::string(d && *d ? d : ".") + "/" + name;
}

static void test_persistence() {
    fixture fx;
    const std::string path = tmp_path("test-moe-predict.bin");
    std::remove(path.c_str());

    auto p = base_params(llama_moe_predictor::MODE_OBSERVE);
    p.stats_path = path;

    uint32_t sum = 0;
    int n_r = 0, n_p = 0;
    {
        llama_moe_predictor mp(p, fx.lt);
        CHECK(!mp.loaded(), "nothing to load yet");
        const auto pred = logits_for(identity_order());
        std::vector<std::vector<int>> sel(N_LAYER, { 0, 1, 16, 17 });
        for (int i = 0; i < 17; ++i) {
            step(mp, sel, pred);
        }
        CHECK(mp.save(), "save failed");
        sum = mp.checksum();
        n_r = mp.view(2).n_r;
        n_p = mp.view(2).n_p;
    } // destructor saves again: same state

    {
        llama_moe_predictor mp(p, fx.lt);
        CHECK(mp.loaded(), "saved state not loaded");
        CHECK(mp.checksum() == sum, "loaded checksum %08x != saved %08x", mp.checksum(), sum);
        CHECK(mp.view(2).n_r == n_r && mp.view(2).n_p == n_p, "slot counts not restored");
    }

    // a different model: ignored
    {
        auto p2 = p;
        p2.model_id = 999;
        llama_moe_predictor mp(p2, fx.lt);
        CHECK(!mp.loaded(), "state of another model was loaded");
        CHECK(mp.view(2).n_r == 16, "cold start expected");
    }
    // restore the file (the previous predictor saved a cold state of model 999 over it)
    {
        llama_moe_predictor mp(p, fx.lt);
        CHECK(!mp.loaded(), "file now belongs to model 999");
    }

    // corrupted and truncated files: ignored
    {
        llama_moe_predictor mp(p, fx.lt);
        CHECK(mp.save(), "save failed");
    }
    {
        std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
        f.seekp(40);
        const char x = 0x5a;
        f.write(&x, 1);
    }
    {
        llama_moe_predictor mp(p, fx.lt);
        CHECK(!mp.loaded(), "corrupted file was loaded");
    }
    {
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        f.write("KRP1", 4);
    }
    {
        llama_moe_predictor mp(p, fx.lt);
        CHECK(!mp.loaded(), "truncated file was loaded");
    }

    // a valid checksum over a truncated file: still rejected
    {
        llama_moe_predictor mp(p, fx.lt);
        CHECK(mp.save(), "save failed");
    }
    {
        std::ifstream in(path, std::ios::binary);
        std::vector<char> b((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        b.resize(b.size() - 4 - N_EXPERT*sizeof(float)); // drop the checksum and the last layer's counts
        const uint32_t sum = fnv1a(b, b.size());
        b.insert(b.end(), (const char *) &sum, (const char *) &sum + 4);
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out.write(b.data(), b.size());
    }
    {
        llama_moe_predictor mp(p, fx.lt);
        CHECK(!mp.loaded(), "truncated file with a valid checksum was loaded");
    }

    // same model id, different layer count: rejected
    {
        llama_moe_predictor mp(p, fx.lt);
        CHECK(mp.save(), "save failed");
    }
    {
        auto p2 = p;
        p2.n_layer = N_LAYER + 1;
        auto lt2 = fx.lt;
        lt2.push_back(fx.lt[N_LAYER - 1]);
        llama_moe_predictor mp(p2, lt2);
        CHECK(!mp.loaded(), "state with a different layer count was loaded");
    }

    // same model id, different expert count: rejected
    {
        llama_moe_predictor mp(p, fx.lt);
        CHECK(mp.save(), "save failed");
    }
    {
        auto p2 = p;
        p2.n_expert = 32;
        llama_moe_predictor mp(p2, fx.lt);
        CHECK(!mp.loaded(), "state with a different expert count was loaded");
    }

    // loading clamps the slot counts to the current cap
    {
        llama_moe_predictor mp(p, fx.lt);
        const auto pred = logits_for(identity_order());
        std::vector<std::vector<int>> sel(N_LAYER, { 0, 1, 16, 17 });
        for (int i = 0; i < 16; ++i) {
            step(mp, sel, pred);
        }
        CHECK(mp.view(2).n_r == 20, "setup: n_r 20 expected, got %d", mp.view(2).n_r);
    } // saved with n_r 20
    {
        auto p2 = p;
        p2.cap = 16;
        llama_moe_predictor mp(p2, fx.lt);
        CHECK(mp.loaded(), "state should load with a smaller cap");
        CHECK(mp.view(2).n_r == 16, "n_r should be clamped to the cap 16, got %d", mp.view(2).n_r);
    }
    std::remove(path.c_str());

    // a failed save leaves no temporary file behind: the target is a non-empty directory, so the rename fails
    {
        const std::string dir = tmp_path("test-moe-predict.dir");
        std::filesystem::create_directories(dir);
        std::ofstream(dir + "/keep") << "x";
        auto p2 = p;
        p2.stats_path = dir;
        {
            llama_moe_predictor mp(p2, fx.lt);
            CHECK(!mp.save(), "saving over a directory should fail");
        }
        std::ifstream t(dir + ".tmp");
        CHECK(!t.good(), "temporary file left behind after a failed save");
        std::filesystem::remove_all(dir);
    }
}


// one MoE layer with a single-token selection; predicts the next layer when pred is given
static void run_layer(llama_moe_predictor & mp, int il, const std::vector<int32_t> & sel, const std::vector<float> * pred) {
    const int il_next = pred != nullptr && il + 1 < N_LAYER ? il + 1 : -1;
    mp.on_layer(il, sel.data(), sizeof(int32_t), N_USED*sizeof(int32_t), 1,
            il_next, il_next >= 0 ? pred->data() : nullptr, N_EXPERT*sizeof(float), nullptr);
}

static void test_precision_recall() {
    fixture fx;
    llama_moe_predictor mp(base_params(llama_moe_predictor::MODE_OBSERVE), fx.lt);
    const auto ident = logits_for(identity_order());

    // 16 router guesses 0..15 for layer 2, which then uses 0, 1, 40, 41: 2 hits of 16 guesses, 2 of 4 actual
    run_layer(mp, 1, { 0, 1, 2, 3 }, &ident);
    run_layer(mp, 2, { 0, 1, 40, 41 }, nullptr);
    auto st = mp.get_stats();
    CHECK(st.hits == 2 && st.actual == 4 && st.guesses == 16, "hits %llu actual %llu guesses %llu, expected 2 4 16",
            (unsigned long long) st.hits, (unsigned long long) st.actual, (unsigned long long) st.guesses);
    CHECK((double) st.hits/st.guesses == 0.125 && (double) st.hits/st.actual == 0.5, "precision 0.125 and recall 0.5 expected");
    CHECK(st.bytes_read_predictor == 0 && st.bytes_pending == 0, "observe mode reads nothing");

    // router and popularity guesses together: an expert counts once, so recall and precision stay <= 1
    llama_moe_predictor mp2(base_params(llama_moe_predictor::MODE_OBSERVE), fx.lt);
    std::vector<std::vector<int>> sel(N_LAYER, { 0, 1, 60, 61 });
    for (int i = 0; i < 64; ++i) {
        step(mp2, sel, ident);
    }
    st = mp2.get_stats();
    CHECK(mp2.view(2).n_p == 4, "setup: popularity slots expected, n_p %d", mp2.view(2).n_p);
    CHECK(st.hits <= st.actual && st.hits <= st.guesses, "hits %llu > actual %llu or guesses %llu",
            (unsigned long long) st.hits, (unsigned long long) st.actual, (unsigned long long) st.guesses);
    double h = 0.0;
    for (int il = 2; il < N_LAYER; ++il) {
        h += mp2.view(il).recall*N_USED*(double) mp2.view(il).tokens;
    }
    CHECK(std::fabs(h - (double) st.hits) < 1e-6, "lifetime hits %llu differ from the per-layer hits %.1f", (unsigned long long) st.hits, h);
}

// simulated loads: LOW guesses read `low` bytes, HIGH loads `high`, except the (layer, expert) pairs in `zero`
struct io_sim {
    uint64_t low  = 1000;
    uint64_t high = 7;
    bool     resident = true;
    std::set<std::pair<int, int>> zero;
    std::function<void(int il, int e, bool high)> on_load; // on the worker thread, at the start of a load
};

static llama_moe_predictor::params sim_params(io_sim & io, bool with_resident = true) {
    auto p = base_params(llama_moe_predictor::MODE_PREFETCH);
    p.paused = true;
    p.test_load_bytes = [&io](int il, int e, bool high) -> uint64_t {
        if (io.on_load) {
            io.on_load(il, e, high);
        }
        return io.zero.count({ il, e }) ? 0 : high ? io.high : io.low;
    };
    if (with_resident) {
        p.test_resident = [&io](int, int) { return io.resident; };
    }
    return p;
}

static void run_loads(llama_moe_predictor & mp) {
    mp.resume();
    mp.wait_idle();
    mp.pause();
}

static std::vector<float> logits_first(std::vector<int> first) {
    for (int e = 0; e < N_EXPERT; ++e) {
        if (std::find(first.begin(), first.end(), e) == first.end()) {
            first.push_back(e);
        }
    }
    return logits_for(first);
}

#define CHECK_BYTES(st, hit_, pend_, wasted_, late_, cred_, readp_, act_) do { \
    CHECK((st).bytes_hit == (hit_) && (st).bytes_pending == (pend_) && (st).bytes_wasted == (wasted_) &&      \
          (st).bytes_late == (late_) && (st).bytes_credited == (cred_) && (st).bytes_read_predictor == (readp_) && \
          (st).bytes_actual == (act_),                                                                         \
          "hit %llu pending %llu wasted %llu late %llu credited %llu read %llu actual %llu, expected %llu %llu %llu %llu %llu %llu %llu", \
          (unsigned long long) (st).bytes_hit, (unsigned long long) (st).bytes_pending, (unsigned long long) (st).bytes_wasted, \
          (unsigned long long) (st).bytes_late, (unsigned long long) (st).bytes_credited,                     \
          (unsigned long long) (st).bytes_read_predictor, (unsigned long long) (st).bytes_actual,              \
          (unsigned long long) (hit_), (unsigned long long) (pend_), (unsigned long long) (wasted_),            \
          (unsigned long long) (late_), (unsigned long long) (cred_), (unsigned long long) (readp_),            \
          (unsigned long long) (act_));                                                                        \
} while (0)

static void test_bytes() {
    fixture fx;
    io_sim io;
    llama_moe_predictor mp(sim_params(io), fx.lt);
    const uint64_t B = io.low;
    const uint64_t H = io.high;
    const auto ident = logits_for(identity_order());
    const auto l20   = logits_first({ 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35 });

    // step 1: the 16 guesses for layer 2 are read before it runs, the 4 experts of layer 1 too
    run_layer(mp, 1, { 0, 1, 2, 3 }, &ident);
    run_loads(mp);
    auto st = mp.get_stats();
    CHECK_BYTES(st, 0, 0, 0, 0, 0, 16*B, 4*H);

    // layer 2 uses 0 and 1: 2 x B hit, the other 14 guesses pending
    run_layer(mp, 2, { 0, 1, 40, 41 }, nullptr);
    st = mp.get_stats();
    CHECK_BYTES(st, 2*B, 14*B, 0, 0, 0, 16*B, 4*H);

    // HIGH loads (layer 2's own experts) count as needed bytes only
    run_loads(mp);
    st = mp.get_stats();
    CHECK_BYTES(st, 2*B, 14*B, 0, 0, 0, 16*B, 8*H);

    // step 2: the guesses are not read before layer 2 runs (stale later: 0 bytes); layer 2 uses the pending expert 2
    // while it is still resident: credited
    run_layer(mp, 1, { 0, 1, 2, 3 }, &l20);
    io.resident = true;
    run_layer(mp, 2, { 2, 50, 51, 52 }, nullptr);
    st = mp.get_stats();
    CHECK_BYTES(st, 3*B, 13*B, 0, 0, B, 16*B, 8*H);
    const auto c0 = mp.get_counters();
    run_loads(mp);
    st = mp.get_stats();
    CHECK(mp.get_counters().low_drops - c0.low_drops == 16, "the 16 stale guesses must be dropped");
    CHECK_BYTES(st, 3*B, 13*B, 0, 0, B, 16*B, 12*H);

    // step 3: the pending expert 3 is used after its pages were evicted: wasted
    run_layer(mp, 1, { 0, 1, 2, 3 }, &l20);
    io.resident = false;
    run_layer(mp, 2, { 3, 50, 51, 52 }, nullptr);
    run_loads(mp);
    st = mp.get_stats();
    CHECK_BYTES(st, 3*B, 12*B, B, 0, B, 16*B, 16*H);

    // step 4: 4 and 5 (pending) are guessed again: reading 4 again means its old pages were gone (wasted),
    // 5 reads nothing (still resident, stays pending)
    io.zero = { { 2, 5 } };
    const auto l4 = logits_first({ 4, 5, 60, 61, 62, 63, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29 });
    run_layer(mp, 1, { 0, 1, 2, 3 }, &l4);
    run_loads(mp);
    st = mp.get_stats();
    CHECK_BYTES(st, 3*B, 11*B, 2*B, 0, B, 31*B, 20*H);

    // layer 2 uses 4 (read for this guess: hit) and 5 (pending, resident: credited); the other 14 guesses are pending
    io.resident = true;
    run_layer(mp, 2, { 4, 5, 40, 41 }, nullptr);
    st = mp.get_stats();
    CHECK_BYTES(st, 5*B, 24*B, 2*B, 0, 2*B, 31*B, 20*H);
    run_loads(mp);
    st = mp.get_stats();
    CHECK_BYTES(st, 5*B, 24*B, 2*B, 0, 2*B, 31*B, 23*H); // the HIGH load of 5 reads nothing too
    CHECK(st.bytes_hit + st.bytes_wasted + st.bytes_late + st.bytes_pending == st.bytes_read_predictor, "guess bytes not conserved");

    // step 5: layer 2 starts while the first guess is read: that load is late, the rest are dropped; the HIGH loads
    // of layer 2 run nested inside that load and are not counted in it
    io.zero.clear();
    bool fired = false;
    io.on_load = [&](int il, int e, bool high) {
        GGML_UNUSED(e);
        if (!high && il == 2 && !fired) {
            fired = true;
            run_layer(mp, 2, { 0, 1, 2, 3 }, nullptr);
        }
    };
    const auto c1 = mp.get_counters();
    run_layer(mp, 1, { 0, 1, 2, 3 }, &ident);
    run_loads(mp);
    io.on_load = nullptr;
    st = mp.get_stats();
    CHECK(fired, "setup: the late load did not run");
    CHECK_BYTES(st, 5*B, 24*B, 2*B, B, 2*B, 32*B, 31*H);
    CHECK(mp.get_counters().low_drops - c1.low_drops == 15, "the other 15 guesses must be dropped");
    CHECK(st.bytes_hit + st.bytes_wasted + st.bytes_late + st.bytes_pending == st.bytes_read_predictor, "guess bytes not conserved");
}

static void test_nested_high() {
    fixture fx;
    io_sim io;
    llama_moe_predictor mp(sim_params(io), fx.lt);
    const auto ident = logits_for(identity_order());

    // layer 2 runs before the worker starts: its 16 guesses for layer 3 wait; the first of them triggers a second
    // pass of layer 2 (as if a new selection arrived), whose 4 HIGH loads are drained inside that LOW load
    run_layer(mp, 1, { 0, 1, 2, 3 }, &ident);
    run_layer(mp, 2, { 0, 1, 2, 3 }, &ident);
    bool fired = false;
    io.on_load = [&](int il, int e, bool high) {
        GGML_UNUSED(e);
        if (!high && il == 3 && !fired) {
            fired = true;
            run_layer(mp, 2, { 0, 1, 2, 3 }, &ident);
        }
    };
    run_loads(mp);
    io.on_load = nullptr;
    const auto st = mp.get_stats();
    CHECK(fired, "setup: the nested load did not run");
    // HIGH: layer 1 dropped (layer 2 has started), layer 2 twice; LOW: layer 2 dropped, layer 3 16 + 16
    CHECK(st.bytes_actual == 8*io.high, "bytes_actual %llu, expected %llu", (unsigned long long) st.bytes_actual, (unsigned long long) (8*io.high));
    CHECK(st.bytes_read_predictor == 32*io.low, "the nested HIGH loads leaked into the LOW load: read %llu, expected %llu",
            (unsigned long long) st.bytes_read_predictor, (unsigned long long) (32*io.low));
    CHECK(st.bytes_late == 0, "no late loads expected, got %llu", (unsigned long long) st.bytes_late);
}

static void test_mincore_fallback() {
    fixture fx;
    io_sim io;
    llama_moe_predictor::params p = sim_params(io, false);
    p.mincore_ok = false;
    llama_moe_predictor mp(p, fx.lt);
    const uint64_t B = io.low;
    const auto ident = logits_for(identity_order());
    const auto l20   = logits_first({ 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35 });

    run_layer(mp, 1, { 0, 1, 2, 3 }, &ident);
    run_loads(mp);
    run_layer(mp, 2, { 0, 1, 40, 41 }, nullptr);
    run_loads(mp);

    // without mincore the HIGH load decides: 2 reads nothing (resident: credited), 3 reads its pages again (wasted)
    io.zero = { { 2, 2 } };
    run_layer(mp, 1, { 0, 1, 2, 3 }, &l20);
    run_layer(mp, 2, { 2, 3, 50, 51 }, nullptr);
    auto st = mp.get_stats();
    CHECK(st.bytes_credited == 0 && st.bytes_pending == 14*B, "no residency probe without mincore");
    run_loads(mp);
    st = mp.get_stats();
    CHECK(st.bytes_credited == B && st.bytes_wasted == B && st.bytes_pending == 12*B && st.bytes_hit == 3*B,
            "credited %llu wasted %llu pending %llu hit %llu", (unsigned long long) st.bytes_credited,
            (unsigned long long) st.bytes_wasted, (unsigned long long) st.bytes_pending, (unsigned long long) st.bytes_hit);

    // the real mincore on the fixture's heap memory (resident): credited at scoring
    io_sim io2;
    llama_moe_predictor::params p2 = sim_params(io2, false);
    llama_moe_predictor mp2(p2, fx.lt);
    run_layer(mp2, 1, { 0, 1, 2, 3 }, &ident);
    run_loads(mp2);
    run_layer(mp2, 2, { 0, 1, 40, 41 }, nullptr);
    run_layer(mp2, 1, { 0, 1, 2, 3 }, &l20);
    run_layer(mp2, 2, { 2, 3, 50, 51 }, nullptr);
    st = mp2.get_stats();
    CHECK(st.bytes_credited == 2*B && st.bytes_pending == 12*B, "mincore on resident memory: credited %llu pending %llu",
            (unsigned long long) st.bytes_credited, (unsigned long long) st.bytes_pending);
    CHECK(st.probes == 2, "2 residency probes expected, got %llu", (unsigned long long) st.probes);
    CHECK(mp.get_stats().probes == 0, "no probes without mincore, got %llu", (unsigned long long) mp.get_stats().probes);
}

// at most 8 experts are probed per layer: the rest stay pending
static void test_probe_cap() {
    fixture fx;
    io_sim io;
    llama_moe_predictor mp(sim_params(io), fx.lt);
    const uint64_t B = io.low;
    const auto ident = logits_for(identity_order());
    const auto l20   = logits_first({ 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35 });

    run_layer(mp, 1, { 0, 1, 2, 3 }, &ident);
    run_loads(mp);
    run_layer(mp, 2, { 0, 1, 40, 41 }, nullptr); // 2..15 pending
    run_layer(mp, 1, { 0, 1, 2, 3 }, &l20);

    // 4 tokens use 2..15 and 40, 41: 14 pending experts used, 8 probed
    std::vector<int32_t> sel = { 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 40, 41 };
    mp.on_layer(2, sel.data(), sizeof(int32_t), N_USED*sizeof(int32_t), 4, -1, nullptr, 0, nullptr);
    const auto st = mp.get_stats();
    CHECK(st.probes == 8, "probes %llu, expected the cap 8", (unsigned long long) st.probes);
    CHECK(st.bytes_credited == 8*B && st.bytes_pending == 6*B, "credited %llu pending %llu",
            (unsigned long long) st.bytes_credited, (unsigned long long) st.bytes_pending);
}

static void test_expiry() {
    fixture fx;
    const auto ident = logits_for(identity_order());

    // a pending read expires once the process has read more than the page cache holds since then
    {
        io_sim io;
        llama_moe_predictor mp(sim_params(io), fx.lt);
        mp.set_io_clock(1000, 5000);
        run_layer(mp, 1, { 0, 1, 2, 3 }, &ident);
        run_loads(mp);
        run_layer(mp, 2, { 0, 1, 40, 41 }, nullptr);
        CHECK(mp.get_stats().bytes_pending == 14*io.low, "setup: 14 pending guesses expected");
        mp.set_io_clock(1000 + 5000, 5000);
        CHECK(mp.get_stats().bytes_pending == 14*io.low && mp.get_stats().bytes_wasted == 0, "expired too early");
        mp.set_io_clock(1000 + 5001, 6000);
        const auto st = mp.get_stats();
        CHECK(st.bytes_pending == 0 && st.bytes_wasted == 14*io.low, "pending %llu wasted %llu after the page cache turned over",
                (unsigned long long) st.bytes_pending, (unsigned long long) st.bytes_wasted);
    }

    // reads that became pending while no clock was published count from the first published value
    {
        io_sim io;
        llama_moe_predictor mp(sim_params(io), fx.lt);
        run_layer(mp, 1, { 0, 1, 2, 3 }, &ident);
        run_loads(mp);
        run_layer(mp, 2, { 0, 1, 40, 41 }, nullptr);
        mp.set_io_clock(1000000000, 5000);
        CHECK(mp.get_stats().bytes_wasted == 0, "a stale clock must not expire everything at once");
        mp.set_io_clock(1000000000 + 5001, 5000);
        CHECK(mp.get_stats().bytes_wasted == 14*io.low, "expected expiry after the page cache turned over");
    }
}

// the progress op raises the reported layer from thread 0 only, and never lowers it (two writers: ops and uploads)
static void test_progress_op() {
    std::atomic<int32_t> il_out{-1};
    llama_decode_progress_ud ud[4];
    for (int i = 0; i < 4; ++i) {
        ud[i] = { &il_out, i };
    }
    llama_decode_progress_op(nullptr, 1, 4, &ud[2]);
    CHECK(il_out.load() == -1, "only thread 0 reports");
    for (int i = 0; i < 4; ++i) {
        llama_decode_progress_op(nullptr, 0, 4, &ud[i]);
        CHECK(il_out.load() == i, "layer %d expected, got %d", i, il_out.load());
    }
    llama_decode_progress_set(il_out, 1);
    CHECK(il_out.load() == 3, "a lower layer must not move the progress back, got %d", il_out.load());
}

// decode progress of a prompt batch with the predictor off, on a generated Kimi-K3 test model
struct progress_rec {
    llama_context * ctx = nullptr;
    std::vector<llama_decode_progress_data> v;
};

static bool progress_cb(ggml_tensor * t, bool ask, void * ud) {
    GGML_UNUSED(t);
    auto * r = (progress_rec *) ud;
    if (!ask && r->ctx != nullptr) {
        llama_decode_progress_data d;
        llama_decode_progress(r->ctx, &d);
        r->v.push_back(d);
    }
    return true;
}

static void test_decode_progress(const char * path) {
#if defined(_WIN32)
    _putenv_s("LLAMA_MOE_PREDICT", "");
#else
    unsetenv("LLAMA_MOE_PREDICT");
#endif
    llama_backend_init();

    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(path, mparams);
    CHECK(model != nullptr, "cannot load %s", path);
    if (model == nullptr) {
        return;
    }

    llama_moe_predict_data pd;
    llama_moe_predict_stats(model, &pd);
    CHECK(!pd.enabled, "no predictor expected");

    progress_rec rec;
    auto cparams = llama_context_default_params();
    cparams.n_ctx             = 64;
    cparams.n_batch           = 16;
    cparams.n_ubatch          = 16;
    cparams.cb_eval           = progress_cb;
    cparams.cb_eval_user_data = &rec;
    llama_context * ctx = llama_init_from_model(model, cparams);
    CHECK(ctx != nullptr, "cannot create a context");
    if (ctx == nullptr) {
        llama_model_free(model);
        return;
    }
    rec.ctx = ctx;

    const int n_layer = llama_model_n_layer(model);
    char buf[64] = {};
    int n_lead = 0;
    if (llama_model_meta_val_str(model, "kimi-k3.leading_dense_block_count", buf, sizeof(buf)) > 0) {
        n_lead = atoi(buf);
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    auto run = [&](int n_tok) {
        rec.v.clear();
        llama_batch batch = llama_batch_init(n_tok, 0, 1);
        for (int i = 0; i < n_tok; ++i) {
            batch.token[i]     = (1 + i) % n_vocab;
            batch.pos[i]       = i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = i == n_tok - 1;
        }
        batch.n_tokens = n_tok;
        const int ret = llama_decode(ctx, batch);
        llama_batch_free(batch);
        llama_memory_clear(llama_get_memory(ctx), true);
        return ret;
    };

    // an 8-token prompt batch: every MoE layer is reported once, in order
    CHECK(run(8) == 0, "decode failed");
    std::vector<int> seen;
    bool ok_tokens = !rec.v.empty();
    for (const auto & d : rec.v) {
        ok_tokens = ok_tokens && d.n_tokens == 8 && d.n_tokens_done == 0 && d.n_layer == n_layer;
        if (seen.empty() ? d.il >= 0 : d.il != seen.back()) {
            seen.push_back(d.il);
        }
    }
    CHECK(ok_tokens, "during the batch: n_tokens 8, n_tokens_done 0, n_layer %d expected", n_layer);
    std::vector<int> expect;
    for (int il = n_lead; il < n_layer; ++il) {
        expect.push_back(il);
    }
    std::string got;
    for (int il : seen) {
        got += std::to_string(il) + " ";
    }
    CHECK(seen == expect, "reported layers %s, expected %d..%d once each in order", got.c_str(), n_lead, n_layer - 1);
    CHECK(!rec.v.empty() && rec.v.front().il == -1, "the first node must see no layer yet");

    llama_decode_progress_data d;
    llama_decode_progress(ctx, &d);
    CHECK(d.n_tokens == 0, "no batch runs after llama_decode returned, n_tokens %d", d.n_tokens);

    // a 2-token batch is a decode step: no layer reports
    CHECK(run(2) == 0, "decode failed");
    bool all_none = !rec.v.empty();
    for (const auto & r : rec.v) {
        all_none = all_none && r.il == -1 && r.n_tokens == 2;
    }
    CHECK(all_none, "small batches report no layers");

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
}

int main(int argc, char ** argv) {
    if (argc == 3 && std::string(argv[1]) == "--progress") {
        test_decode_progress(argv[2]);
        if (n_fail) {
            printf("FAILED (%d checks)\n", n_fail);
            return 1;
        }
        printf("OK\n");
        return 0;
    }

    test_rank_router();
    test_hits_and_growth();
    test_popularity();
    test_cap();
    test_shrink();
    test_multi_token();
    test_fresh_prediction_each_step();
    test_popularity_decay_and_exclusion();
    test_queue_order_and_stale();
    test_startup_probe_ignored();
    test_queue_limits();
    test_persistence();
    test_precision_recall();
    test_bytes();
    test_nested_high();
    test_mincore_fallback();
    test_probe_cap();
    test_expiry();
    test_progress_op();

    if (n_fail) {
        printf("FAILED (%d checks)\n", n_fail);
        return 1;
    }
    printf("OK\n");
    return 0;
}
