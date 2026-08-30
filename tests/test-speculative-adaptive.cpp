#include "speculative-adaptive.h"
#include "ngram-map.h"

#include <cstdio>
#include <cstdlib>

static void require(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "adaptive test failure: %s\n", message);
        std::abort();
    }
}

static void test_opt_out_parser() {
    require(common_speculative_adaptive_env_enabled(nullptr), "unset enables adaptive mode");
    require(common_speculative_adaptive_env_enabled("1"), "one enables adaptive mode");
    require(!common_speculative_adaptive_env_enabled("0"), "zero disables adaptive mode");
    require(!common_speculative_adaptive_env_enabled("off"), "off disables adaptive mode");
    require(!common_speculative_adaptive_env_enabled("false"), "false disables adaptive mode");
    require(!common_speculative_adaptive_env_enabled("no"), "no disables adaptive mode");
}

static void test_bounds_and_staircase() {
    common_speculative_adaptive ctrl;
    ctrl.reset(3, 48);
    require(ctrl.n_floor == 3 && ctrl.n_ceiling == 48 && ctrl.n_cur == 3,
            "floor/start/ceiling");

    // The first wider rung needs the controller's hysteresis barrier.
    for (int i = 0; i < 9; ++i) {
        ctrl.update(3, 3);
        require(ctrl.n_cur == 3, "3 remains stable before promotion threshold");
    }
    ctrl.update(3, 3);
    require(ctrl.n_cur == 4, "3 promotes to 4");

    // Wider rungs grow gradually; there is no 3 -> 48 jump.
    for (int i = 0; i < 5; ++i) ctrl.update(4, 4);
    require(ctrl.n_cur == 4, "4 waits for its threshold");
    ctrl.update(4, 4);
    require(ctrl.n_cur == 6, "4 promotes to 6");
    for (int i = 0; i < 2; ++i) {
        ctrl.update(6, 6);
        require(ctrl.n_cur == 6, "6 waits for its threshold");
    }
    ctrl.update(6, 6);
    require(ctrl.n_cur == 8, "6 promotes to 8");

    ctrl.n_cur = 48;
    ctrl.update(48, 47);
    require(ctrl.n_cur == 32, "partial 48-token round retreats one staircase rung");
    ctrl.update(32, 0);
    require(ctrl.n_cur == 24, "partial 32-token round retreats one staircase rung");
    ctrl.update(24, 0);
    require(ctrl.n_cur == 16, "partial 24-token round retreats one staircase rung");
}

static void test_ngram_map_width_override() {
    const llama_tokens prompt = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 1,
    };

    common_ngram_map normal(2, 6, true, 1);
    common_ngram_map_begin(normal, prompt);
    llama_tokens normal_draft;
    common_ngram_map_draft(normal, prompt, 2, normal_draft);
    require(normal_draft.size() == 6, "normal ngram map uses configured width");

    common_ngram_map adaptive(2, 6, true, 1);
    common_ngram_map_begin(adaptive, prompt);
    adaptive.adaptive_draft_limit = 3;
    llama_tokens adaptive_draft;
    common_ngram_map_draft(adaptive, prompt, 2, adaptive_draft);
    require(adaptive_draft.size() == 3, "adaptive ngram map honors current width");

    common_ngram_map complex_normal(2, 6, false, 1);
    common_ngram_map_begin(complex_normal, prompt);
    llama_tokens complex_draft;
    common_ngram_map_draft(complex_normal, prompt, 2, complex_draft);
    require(complex_draft.size() == 6, "complex ngram map uses configured width");

    common_ngram_map complex_adaptive(2, 6, false, 1);
    common_ngram_map_begin(complex_adaptive, prompt);
    complex_adaptive.adaptive_draft_limit = 3;
    llama_tokens complex_adaptive_draft;
    common_ngram_map_draft(complex_adaptive, prompt, 2, complex_adaptive_draft);
    require(complex_adaptive_draft.size() == 3, "complex adaptive ngram map honors current width");

    // The controller must not exceed the map's actual available value span.
    common_ngram_map short_value(2, 1, true, 1);
    common_ngram_map_begin(short_value, prompt);
    short_value.adaptive_draft_limit = 6;
    llama_tokens short_draft;
    common_ngram_map_draft(short_value, prompt, 2, short_draft);
    require(short_draft.size() == 1, "adaptive width preserves available value bound");
}

static void test_short_matches_and_floor() {
    common_speculative_adaptive ctrl;
    ctrl.reset(3, 48);
    ctrl.n_cur = 12;

    // A short, fully accepted match did not exercise width 12 and cannot
    // promote the controller.
    ctrl.update(3, 3);
    require(ctrl.n_cur == 12, "short full match does not promote");
    require(ctrl.n_climb == 0, "short full match resets climb evidence");

    // Poor rounds cannot retreat below the configured MTP floor.
    ctrl.n_cur = 4;
    ctrl.update(4, 0);
    require(ctrl.n_cur == 3, "partial round retreats to floor");
    ctrl.update(3, 0);
    require(ctrl.n_cur == 3, "floor is sticky");

    // A configured floor higher than one is respected.
    ctrl.reset(5, 48);
    ctrl.n_cur = 7;
    ctrl.update(7, 0);
    require(ctrl.n_cur == 5, "higher floor is respected");
}

int main() {
    test_opt_out_parser();
    test_bounds_and_staircase();
    test_ngram_map_width_override();
    test_short_matches_and_floor();
    std::puts("test-speculative-adaptive: PASS");
    return 0;
}
