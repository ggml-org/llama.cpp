#include "tessera-awq.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <vector>

//
// PRNG - xorshift32
//

struct ts_awq_rng {
    uint32_t s;

    void init(uint32_t seed) {
        s = seed ? seed : 1;
    }

    uint32_t next() {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        return s;
    }

    float uniform() {
        return (float)(next() >> 8) * (1.0f / 16777216.0f);
    }

    float gauss(float sigma) {
        float u1 = std::max(uniform(), 1e-10f);
        float u2 = uniform();
        return sigma * sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265358979323846f * u2);
    }
};

//
// Gene access (alpha, clip, lrq_rank_frac, rotation_lr)
//

static const int TS_AWQ_N_GENES = 4;

static float ts_awq_get_gene(const ts_awq_candidate * c, int i) {
    switch (i) {
        case 0: return c->alpha;
        case 1: return c->clip;
        case 2: return c->lrq_rank_frac;
        case 3: return c->rotation_lr;
        default: return 0.0f;
    }
}

static void ts_awq_set_gene(ts_awq_candidate * c, int i, float v) {
    v = std::max(0.0f, std::min(1.0f, v));
    switch (i) {
        case 0: c->alpha = v; break;
        case 1: c->clip = v; break;
        case 2: c->lrq_rank_frac = v; break;
        case 3: c->rotation_lr = v; break;
        default: break;
    }
}

//
// GA operators
//

static ts_awq_candidate ts_awq_random_candidate(ts_awq_rng * rng) {
    ts_awq_candidate c;
    for (int i = 0; i < TS_AWQ_N_GENES; i++) {
        ts_awq_set_gene(&c, i, rng->uniform());
    }
    c.expert_hint = -1;
    return c;
}

static ts_awq_candidate ts_awq_mutate(const ts_awq_candidate * c, ts_awq_rng * rng, float sigma) {
    ts_awq_candidate out = *c;
    for (int i = 0; i < TS_AWQ_N_GENES; i++) {
        ts_awq_set_gene(&out, i, ts_awq_get_gene(c, i) + rng->gauss(sigma));
    }
    return out;
}

static ts_awq_candidate ts_awq_crossover(const ts_awq_candidate * a, const ts_awq_candidate * b,
                                          ts_awq_rng * rng, float rate) {
    ts_awq_candidate out;
    for (int i = 0; i < TS_AWQ_N_GENES; i++) {
        ts_awq_set_gene(&out, i, rng->uniform() < rate ? ts_awq_get_gene(a, i)
                                                        : ts_awq_get_gene(b, i));
    }
    out.expert_hint = -1;
    return out;
}

//
// Family index
//

static int32_t ts_awq_family_idx(const std::string & family) {
    if (family == "attention")        return 0;
    if (family == "ffn")              return 1;
    if (family == "router")           return 2;
    if (family == "routed_expert")    return 3;
    if (family == "shared_expert")    return 4;
    if (family == "fusion")           return 5;
    if (family == "output_embedding") return 6;
    return 1;
}

//
// Archive cell comparator
//

struct ts_awq_cell_cmp {
    bool operator()(const ts_awq_archive_cell & a, const ts_awq_archive_cell & b) const {
        if (a.kurtosis_bucket != b.kurtosis_bucket) return a.kurtosis_bucket < b.kurtosis_bucket;
        if (a.eff_rank_bucket != b.eff_rank_bucket) return a.eff_rank_bucket < b.eff_rank_bucket;
        return a.family_bucket < b.family_bucket;
    }
};

struct ts_awq_archive_entry {
    ts_awq_candidate cand;
    ts_awq_score     score;
};

//
// Public API
//

ts_awq_archive_cell ts_awq_make_cell(float kurtosis, float eff_rank, int32_t family_idx) {
    ts_awq_archive_cell cell;
    if      (kurtosis <  2.0f) cell.kurtosis_bucket = 0;
    else if (kurtosis <  5.0f) cell.kurtosis_bucket = 1;
    else if (kurtosis < 10.0f) cell.kurtosis_bucket = 2;
    else if (kurtosis < 50.0f) cell.kurtosis_bucket = 3;
    else                       cell.kurtosis_bucket = 4;
    cell.eff_rank_bucket = std::min(4, std::max(0, (int)(eff_rank * 5.0f)));
    cell.family_bucket = family_idx;
    return cell;
}

std::string ts_awq_candidate_json(const ts_awq_candidate * cand) {
    char buf[512];
    snprintf(buf, sizeof(buf),
             "{\"alpha\":%.6f,\"clip\":%.6f,\"lrq_rank_frac\":%.6f,"
             "\"rotation_lr\":%.6f,\"expert_hint\":%lld}",
             cand->alpha, cand->clip, cand->lrq_rank_frac, cand->rotation_lr,
             (long long)cand->expert_hint);
    return std::string(buf);
}

int ts_awq_evolve(const ts_awq_layer * layer,
                  ts_awq_eval_fn eval, void * eval_ctx,
                  const ts_awq_evolve_params * params,
                  ts_awq_evolve_result * result) {
    if (!layer || !eval || !params || !result) {
        return -1;
    }

    const int64_t pop_size   = std::max(params->population, (int64_t)2);
    const int64_t n_gen      = std::max(params->generations, (int64_t)1);
    const int64_t n_islands  = std::max(params->islands, (int64_t)1);
    const int64_t mig_interval = params->migration_interval > 0 ? params->migration_interval : 10;

    ts_awq_rng rng;
    rng.init(params->seed);

    // Initialize island populations
    std::vector<std::vector<ts_awq_candidate>> pops(n_islands);
    std::vector<std::vector<ts_awq_score>>     scores(n_islands);
    for (int64_t isl = 0; isl < n_islands; isl++) {
        pops[isl].resize(pop_size);
        scores[isl].resize(pop_size);
        for (int64_t i = 0; i < pop_size; i++) {
            pops[isl][i] = ts_awq_random_candidate(&rng);
        }
    }

    int64_t evaluations = 0;

    // Evaluate initial populations
    for (int64_t isl = 0; isl < n_islands; isl++) {
        for (int64_t i = 0; i < pop_size; i++) {
            scores[isl][i] = eval(&pops[isl][i], layer, eval_ctx);
            evaluations++;
        }
    }

    // Archive
    std::map<ts_awq_archive_cell, ts_awq_archive_entry, ts_awq_cell_cmp> archive;
    ts_awq_archive_cell cell = ts_awq_make_cell(layer->kurtosis, layer->eff_rank,
                                                 ts_awq_family_idx(layer->family));

    // Track global best
    ts_awq_candidate best = pops[0][0];
    ts_awq_score     best_score = scores[0][0];

    auto update_best = [&](const ts_awq_candidate & c, const ts_awq_score & s) {
        if (s.composite > best_score.composite) {
            best = c;
            best_score = s;
        }
    };

    for (int64_t isl = 0; isl < n_islands; isl++) {
        for (int64_t i = 0; i < pop_size; i++) {
            update_best(pops[isl][i], scores[isl][i]);
        }
    }

    // Main GA loop
    for (int64_t gen = 0; gen < n_gen; gen++) {
        for (int64_t isl = 0; isl < n_islands; isl++) {
            // Rank by composite descending (higher = better)
            std::vector<int64_t> idx(pop_size);
            for (int64_t i = 0; i < pop_size; i++) {
                idx[i] = i;
            }
            std::stable_sort(idx.begin(), idx.end(), [&](int64_t a, int64_t b) {
                return scores[isl][a].composite > scores[isl][b].composite;
            });

            // Update archive with island best
            {
                auto it = archive.find(cell);
                if (it == archive.end() || scores[isl][idx[0]].composite > it->second.score.composite) {
                    archive[cell] = {pops[isl][idx[0]], scores[isl][idx[0]]};
                }
            }

            // Elitism: keep top 25% (min 2)
            int64_t elite_count = std::max((int64_t)2, pop_size / 4);
            std::vector<const ts_awq_candidate *> elites;
            for (int64_t i = 0; i < elite_count; i++) {
                elites.push_back(&pops[isl][idx[i]]);
            }

            // Breed next generation
            std::vector<ts_awq_candidate> next_pop;
            next_pop.reserve(pop_size);
            for (auto * e : elites) {
                next_pop.push_back(*e);
            }
            while ((int64_t)next_pop.size() < pop_size) {
                int64_t li = (int64_t)(rng.next() % (uint32_t)elite_count);
                int64_t ri = (int64_t)(rng.next() % (uint32_t)elite_count);
                ts_awq_candidate child = ts_awq_crossover(elites[li], elites[ri], &rng,
                                                           params->crossover_rate);
                child = ts_awq_mutate(&child, &rng, params->mutation_sigma);
                next_pop.push_back(child);
            }
            pops[isl] = std::move(next_pop);

            // Evaluate new population
            for (int64_t i = 0; i < pop_size; i++) {
                scores[isl][i] = eval(&pops[isl][i], layer, eval_ctx);
                evaluations++;
                update_best(pops[isl][i], scores[isl][i]);
            }
        }

        // Migration: best of island i -> worst of island (i+1) % n
        if (n_islands > 1 && (gen + 1) % mig_interval == 0) {
            std::vector<ts_awq_candidate> migrants(n_islands);
            std::vector<ts_awq_score>     migrant_scores(n_islands);
            for (int64_t isl = 0; isl < n_islands; isl++) {
                int64_t bi = 0;
                for (int64_t i = 1; i < pop_size; i++) {
                    if (scores[isl][i].composite > scores[isl][bi].composite) {
                        bi = i;
                    }
                }
                migrants[isl] = pops[isl][bi];
                migrant_scores[isl] = scores[isl][bi];
            }
            for (int64_t isl = 0; isl < n_islands; isl++) {
                int64_t dst = (isl + 1) % n_islands;
                int64_t wi = 0;
                for (int64_t i = 1; i < pop_size; i++) {
                    if (scores[dst][i].composite < scores[dst][wi].composite) {
                        wi = i;
                    }
                }
                pops[dst][wi] = migrants[isl];
                scores[dst][wi] = migrant_scores[isl];
            }
        }
    }

    // Fill result
    result->best = best;
    result->best_score = best_score;
    result->generations_run = n_gen;
    result->evaluations = evaluations;
    result->archive.clear();
    for (auto & kv : archive) {
        result->archive.push_back({kv.first, kv.second.cand});
    }

    return 0;
}

int ts_awq_evolve_all(const ts_awq_layer * layers, int64_t n_layers,
                      ts_awq_eval_fn eval, void * eval_ctx,
                      const ts_awq_evolve_params * params,
                      std::vector<ts_awq_evolve_result> * results) {
    if (!layers || n_layers < 1 || !eval || !params || !results) {
        return -1;
    }

    results->resize(n_layers);

    // Progressive eval: screen on stratified 25% of layers first, then
    // promote survivors to full eval. For the per-layer GA, this reduces
    // to running a short screen on a subset of layers to warm-start seeds.
    // With <= 4 layers, skip screening.
    if (n_layers > 4) {
        int64_t screen_count = std::max((int64_t)1, n_layers / 4);
        ts_awq_evolve_params screen_params = *params;
        screen_params.generations = std::max((int64_t)1, params->generations / 4);

        // Stratified screen: evenly spaced layers
        std::vector<ts_awq_evolve_result> screen_results(screen_count);
        for (int64_t si = 0; si < screen_count; si++) {
            int64_t li = si * (n_layers - 1) / std::max((int64_t)1, screen_count - 1);
            screen_params.seed = params->seed + (uint32_t)li;
            int rc = ts_awq_evolve(&layers[li], eval, eval_ctx, &screen_params, &screen_results[si]);
            if (rc != 0) {
                return rc;
            }
        }
    }

    // Full per-layer evolution
    for (int64_t i = 0; i < n_layers; i++) {
        ts_awq_evolve_params layer_params = *params;
        layer_params.seed = params->seed + (uint32_t)i;
        int rc = ts_awq_evolve(&layers[i], eval, eval_ctx, &layer_params, &(*results)[i]);
        if (rc != 0) {
            return rc;
        }
    }

    return 0;
}
