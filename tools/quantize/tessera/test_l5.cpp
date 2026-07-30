#include "tessera-l5.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>

static void test_percentile_rank() {
    ts_score_map scores;
    for (int i = 0; i < 10; i++) {
        char name[32];
        snprintf(name, sizeof(name), "tensor_%d", i);
        scores[name] = (float)(i * 10);
    }

    ts_score_map ranks = ts_l5_percentile_rank(&scores);
    assert(ranks.size() == 10);

    // all in [0, 1]
    for (const auto & kv : ranks) {
        assert(kv.second >= 0.0f && kv.second <= 1.0f);
    }

    // monotonic: higher score -> higher rank
    float prev_rank = -1.0f;
    for (int i = 0; i < 10; i++) {
        char name[32];
        snprintf(name, sizeof(name), "tensor_%d", i);
        float r = ranks[name];
        assert(r >= prev_rank);
        prev_rank = r;
    }

    printf("  percentile_rank: OK\n");
}

static void test_pick_top() {
    ts_score_map scores;
    for (int i = 0; i < 10; i++) {
        char name[32];
        snprintf(name, sizeof(name), "tensor_%d", i);
        scores[name] = (float)(i * 10);
    }

    std::vector<std::string> top = ts_l5_pick_top(&scores, 0.3f);
    assert(top.size() == 3);

    // top 3 should be tensor_9, tensor_8, tensor_7
    assert(top[0] == "tensor_9");
    assert(top[1] == "tensor_8");
    assert(top[2] == "tensor_7");

    printf("  pick_top: OK\n");
}

static void test_ladder() {
    const char * up = ts_l5_step_up("q4_k");
    assert(up != nullptr);
    assert(strcmp(up, "q5_k") == 0);

    const char * down = ts_l5_step_down("q8_0");
    assert(down != nullptr);
    assert(strcmp(down, "tessera_t640") == 0);

    // boundary: step_up from top returns nullptr
    assert(ts_l5_step_up("q8_0") == nullptr);

    // boundary: step_down from bottom returns nullptr
    assert(ts_l5_step_down("q2_k") == nullptr);

    // ladder_index
    assert(ts_l5_ladder_index("q4_k") >= 0);
    assert(ts_l5_ladder_index("nonexistent") == -1);

    printf("  ladder: OK\n");
}

static void test_orchestrator_step() {
    // 10 tensors with high-variance scores
    ts_score_map sensitivity;
    const char * names[10];
    const char * qtypes[10];
    char name_bufs[10][32];

    for (int i = 0; i < 10; i++) {
        snprintf(name_bufs[i], sizeof(name_bufs[i]), "blk.%d.attn_q", i);
        names[i] = name_bufs[i];
        qtypes[i] = "q4_k";
        // high variance: scores range from 0.05 to 0.95
        sensitivity[name_bufs[i]] = 0.05f + 0.9f * ((float)i / 9.0f);
    }

    ts_orchestrator_params params;
    params.max_generations = 10;
    params.top_fraction = 0.3f;
    params.delta_threshold = 0.01f;
    params.ema_beta = 0.9f;
    params.verbose = false;

    ts_requant_plan plan;
    int n_actions = ts_l5_orchestrate_step(&sensitivity, qtypes, 10, 1, &params, &plan);

    // should have actions for top tensors
    assert(n_actions > 0);
    assert(plan.actions.size() == (size_t)n_actions);
    assert(plan.generation == 1);

    // all actions should be step_up from q4_k to q5_k
    for (const auto & action : plan.actions) {
        assert(action.type == TS_REQUANT_STEP_UP);
        assert(action.from_qtype == "q4_k");
        assert(action.to_qtype == "q5_k");
        assert(action.expected_delta > params.delta_threshold);
    }

    assert(plan.total_expected_delta > 0.0f);

    printf("  orchestrator_step: OK (%d actions)\n", n_actions);
}

static void test_ema() {
    ts_l5_ema ema;
    ts_l5_ema_init(&ema, 0.9f);

    ts_score_map s1;
    s1["a"] = 1.0f;
    s1["b"] = 0.5f;
    ts_l5_ema_update(&ema, &s1);

    // first update seeds directly
    assert(fabsf(ema.state["a"] - 1.0f) < 1e-6f);
    assert(fabsf(ema.state["b"] - 0.5f) < 1e-6f);

    // second update applies decay
    ts_score_map s2;
    s2["a"] = 0.0f;
    ts_l5_ema_update(&ema, &s2);
    float expected = 0.9f * 1.0f + 0.1f * 0.0f;
    assert(fabsf(ema.state["a"] - expected) < 1e-6f);

    printf("  ema: OK\n");
}

int main() {
    printf("test_l5:\n");
    test_percentile_rank();
    test_pick_top();
    test_ladder();
    test_orchestrator_step();
    test_ema();
    printf("all tests passed\n");
    return 0;
}
