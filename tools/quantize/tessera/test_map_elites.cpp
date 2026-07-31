#include "tessera-search.h"
#include "tessera-regime.h"

#include <cmath>
#include <cstdio>
#include <cstring>

static bool feq(float a, float b) {
    return fabsf(a - b) < 1e-5f;
}

static ts_regime_descriptor make_desc(const char * name, const char * family,
                                      float kurtosis, float eff_rank, int modality) {
    ts_regime_descriptor d = {};
    d.tensor_name = name;
    d.family      = family;
    d.kurtosis    = kurtosis;
    d.eff_rank    = eff_rank;
    d.modality    = modality;
    return d;
}

static int test_init_and_empty_summary() {
    ts_map_elites_archive a;
    ts_archive_init(&a, 3, 3, 4, 3);

    if ((int)a.cells.size() != 3 * 3 * 4 * 3) {
        printf("FAIL init: cell count %d, expected %d\n", (int)a.cells.size(), 3 * 3 * 4 * 3);
        return 1;
    }

    ts_archive_summary s = ts_archive_summarize(&a);
    if (s.total_cells != 108 || s.occupied_cells != 0 ||
        !feq(s.mean_fitness, 0.0f) || !feq(s.best_fitness, 0.0f) || !feq(s.worst_fitness, 0.0f)) {
        printf("FAIL empty summary: total=%d occupied=%d mean=%.4f\n",
               s.total_cells, s.occupied_cells, s.mean_fitness);
        return 1;
    }
    printf("PASS init + empty summary: %d cells\n", s.total_cells);
    return 0;
}

static int test_insert_and_displace() {
    ts_map_elites_archive a;
    ts_archive_init(&a, 3, 3, 4, 3);

    ts_regime_descriptor d = make_desc("blk.0.attn_q.weight", "attn_q", 10.0f, 0.5f, 0);

    // first insert occupies the cell
    if (!ts_archive_insert(&a, &d, 0.5f, 0.3f, 0.7f, "blk.0.attn_q.weight")) {
        printf("FAIL insert: first insert did not improve empty cell\n");
        return 1;
    }
    const ts_archive_cell * c = ts_archive_query(&a, &d);
    if (!c || !feq(c->best_fitness, 0.5f) || !feq(c->best_alpha, 0.3f) || c->eval_count != 1) {
        printf("FAIL insert: cell state after first insert incorrect\n");
        return 1;
    }

    // better candidate displaces it
    if (!ts_archive_insert(&a, &d, 0.2f, 0.4f, 0.6f, "blk.1.attn_q.weight")) {
        printf("FAIL insert: better candidate did not improve cell\n");
        return 1;
    }
    c = ts_archive_query(&a, &d);
    if (!c || !feq(c->best_fitness, 0.2f) || !feq(c->best_alpha, 0.4f) ||
        !feq(c->best_clip, 0.6f) || c->eval_count != 2 ||
        strcmp(c->tensor_name, "blk.1.attn_q.weight") != 0) {
        printf("FAIL insert: cell state after better insert incorrect (fit=%.4f eval=%lld)\n",
               c ? c->best_fitness : -1.0f, c ? (long long)c->eval_count : -1);
        return 1;
    }

    // worse candidate does NOT displace, but still bumps eval_count
    if (ts_archive_insert(&a, &d, 0.9f, 0.1f, 0.1f, "blk.2.attn_q.weight")) {
        printf("FAIL insert: worse candidate incorrectly improved cell\n");
        return 1;
    }
    c = ts_archive_query(&a, &d);
    if (!c || !feq(c->best_fitness, 0.2f) || !feq(c->best_alpha, 0.4f) || c->eval_count != 3) {
        printf("FAIL insert: worse candidate displaced the elite\n");
        return 1;
    }
    printf("PASS insert + displacement: elite=0.2 eval_count=3\n");
    return 0;
}

static int test_modality_separates_cells() {
    ts_map_elites_archive a;
    ts_archive_init(&a, 3, 3, 4, 3);

    ts_regime_descriptor text  = make_desc("t.attn_q.weight", "attn_q", 10.0f, 0.5f, 0);
    ts_regime_descriptor image = make_desc("i.attn_q.weight", "attn_q", 10.0f, 0.5f, 1);

    ts_archive_insert(&a, &text,  0.2f, 0.4f, 0.6f, "t.attn_q.weight");
    ts_archive_insert(&a, &image, 0.8f, 0.9f, 0.1f, "i.attn_q.weight");

    const ts_archive_cell * ct = ts_archive_query(&a, &text);
    const ts_archive_cell * ci = ts_archive_query(&a, &image);
    if (!ct || !ci) {
        printf("FAIL modality: expected both cells occupied\n");
        return 1;
    }
    if (!feq(ct->best_fitness, 0.2f) || !feq(ci->best_fitness, 0.8f)) {
        printf("FAIL modality: cells not independent (text=%.4f image=%.4f)\n",
               ct->best_fitness, ci->best_fitness);
        return 1;
    }
    if (ct->modality_bucket == ci->modality_bucket) {
        printf("FAIL modality: both descriptors mapped to the same modality bucket\n");
        return 1;
    }

    // an unoccupied cell (modality 2) queries to nullptr
    ts_regime_descriptor audio = make_desc("a.attn_q.weight", "attn_q", 10.0f, 0.5f, 2);
    if (ts_archive_query(&a, &audio) != nullptr) {
        printf("FAIL modality: unoccupied cell returned non-null\n");
        return 1;
    }
    printf("PASS modality axis separates cells\n");
    return 0;
}

static int test_summary_stats() {
    ts_map_elites_archive a;
    ts_archive_init(&a, 3, 3, 4, 3);

    ts_regime_descriptor text  = make_desc("t.weight", "attn_q", 10.0f, 0.5f, 0);
    ts_regime_descriptor image = make_desc("i.weight", "attn_q", 10.0f, 0.5f, 1);
    ts_archive_insert(&a, &text,  0.2f, 0.4f, 0.6f, "t.weight");
    ts_archive_insert(&a, &image, 0.8f, 0.9f, 0.1f, "i.weight");

    ts_archive_summary s = ts_archive_summarize(&a);
    if (s.occupied_cells != 2 || !feq(s.best_fitness, 0.2f) ||
        !feq(s.worst_fitness, 0.8f) || !feq(s.mean_fitness, 0.5f)) {
        printf("FAIL summary: occupied=%d best=%.4f worst=%.4f mean=%.4f\n",
               s.occupied_cells, s.best_fitness, s.worst_fitness, s.mean_fitness);
        return 1;
    }
    printf("PASS summary: occupied=2 best=0.2 worst=0.8 mean=0.5\n");
    return 0;
}

static int test_json_round_trip() {
    ts_map_elites_archive a;
    ts_archive_init(&a, 3, 3, 4, 3);

    ts_regime_descriptor text  = make_desc("t.weight", "attn_q", 10.0f, 0.5f, 0);
    ts_regime_descriptor image = make_desc("i.weight", "ffn_down", 30.0f, 0.9f, 1);
    ts_archive_insert(&a, &text,  0.2f, 0.4f, 0.6f, "t.weight");
    ts_archive_insert(&a, &text,  0.9f, 0.1f, 0.1f, "t2.weight");   // worse, no displace
    ts_archive_insert(&a, &image, 0.35f, 0.55f, 0.45f, "i.weight");

    std::string js = ts_archive_to_json(&a);

    ts_map_elites_archive b;
    if (!ts_archive_from_json(js.c_str(), &b)) {
        printf("FAIL json: from_json returned false\n");
        return 1;
    }
    if (b.n_kurtosis_bins != 3 || b.n_rank_bins != 3 ||
        b.n_family_bins != 4 || b.n_modality_bins != 3) {
        printf("FAIL json: bin dims not preserved (%d %d %d %d)\n",
               b.n_kurtosis_bins, b.n_rank_bins, b.n_family_bins, b.n_modality_bins);
        return 1;
    }

    ts_archive_summary sa = ts_archive_summarize(&a);
    ts_archive_summary sb = ts_archive_summarize(&b);
    if (sa.occupied_cells != sb.occupied_cells || !feq(sa.mean_fitness, sb.mean_fitness) ||
        !feq(sa.best_fitness, sb.best_fitness) || !feq(sa.worst_fitness, sb.worst_fitness)) {
        printf("FAIL json: summary mismatch after round-trip\n");
        return 1;
    }

    const ts_archive_cell * ct = ts_archive_query(&b, &text);
    const ts_archive_cell * ci = ts_archive_query(&b, &image);
    if (!ct || !feq(ct->best_fitness, 0.2f) || !feq(ct->best_alpha, 0.4f) ||
        ct->eval_count != 2 || strcmp(ct->tensor_name, "t.weight") != 0) {
        printf("FAIL json: text cell not restored correctly\n");
        return 1;
    }
    if (!ci || !feq(ci->best_fitness, 0.35f) || !feq(ci->best_alpha, 0.55f)) {
        printf("FAIL json: image cell not restored correctly\n");
        return 1;
    }
    printf("PASS json round-trip: %d occupied cells restored\n", sb.occupied_cells);
    return 0;
}

static int test_from_json_invalid() {
    ts_map_elites_archive a;
    if (ts_archive_from_json("not json at all", &a)) {
        printf("FAIL invalid json: parse should have failed\n");
        return 1;
    }
    if (ts_archive_from_json("{\"no_cells\": true}", &a)) {
        printf("FAIL invalid json: missing cells should have failed\n");
        return 1;
    }
    printf("PASS invalid json rejected\n");
    return 0;
}

int main() {
    int failures = 0;
    failures += test_init_and_empty_summary();
    failures += test_insert_and_displace();
    failures += test_modality_separates_cells();
    failures += test_summary_stats();
    failures += test_json_round_trip();
    failures += test_from_json_invalid();

    if (failures == 0) {
        printf("\nAll tests passed.\n");
    } else {
        printf("\n%d test(s) FAILED.\n", failures);
    }
    return failures;
}
