// next-layer expert prediction (src/llama-moe-predict.*): selection rule, hit accounting, adaptation of the number
// of guesses, popularity, work queue priority and stale guesses, and persistence of the learned state

#include "../src/llama-moe-predict.h"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <numeric>
#include <string>
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

int main() {
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

    if (n_fail) {
        printf("FAILED (%d checks)\n", n_fail);
        return 1;
    }
    printf("OK\n");
    return 0;
}
