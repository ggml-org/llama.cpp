#include "tessera-awq.h"
#include "tessera-sharded-map.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <thread>
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
// Gene access. The 6 genes are the fields of ts_policy_genes, in declaration
// order: alpha, clip, outlier_fraction, moment_mix, tail_guard,
// ternary_threshold. This is the same space Python's Candidate searches
// (awq-evolve.py), so set/get apply Python's per-gene clip ranges rather than
// a blanket [0,1].
//

static const int TS_AWQ_N_GENES = 6;

// Per-gene metadata: (min, max, sigma, is_multiplicative). Mirrors Python's
// Candidate.clipped() ranges and mutate() sigma scales. ternary_threshold is
// the one multiplicative gene in Python (its sigma scales the noise, not the
// gene). is_multiplicative is recorded for fidelity but mutate() applies the
// same additive `gene + gauss(sigma * scale)` form Python uses.
struct ts_awq_gene_spec {
    float lo;
    float hi;
    float sigma;
    bool  is_multiplicative;
};

// alpha, clip, outlier_fraction, moment_mix, tail_guard, ternary_threshold
static const ts_awq_gene_spec TS_AWQ_GENE_SPECS[TS_AWQ_N_GENES] = {
    {  0.0f,    1.0f, 0.20f, false },  // alpha
    {  0.70f,   1.0f, 0.08f, false },  // clip
    {  0.0001f, 0.05f, 0.0f, false },  // outlier_fraction (sigma is value-dependent)
    {  0.0f,    1.0f, 0.20f, false },  // moment_mix
    {  0.0f,    2.0f, 0.25f, false },  // tail_guard
    {  0.30f,   3.0f, 0.18f, true  },  // ternary_threshold (multiplicative)
};

static float ts_awq_get_gene(const ts_awq_candidate * c, int i) {
    switch (i) {
        case 0: return c->genes.alpha;
        case 1: return c->genes.clip;
        case 2: return c->genes.outlier_fraction;
        case 3: return c->genes.moment_mix;
        case 4: return c->genes.tail_guard;
        case 5: return c->genes.ternary_threshold;
        default: return 0.0f;
    }
}

static void ts_awq_set_gene(ts_awq_candidate * c, int i, float v) {
    if (i < 0 || i >= TS_AWQ_N_GENES) {
        return;
    }
    const ts_awq_gene_spec & s = TS_AWQ_GENE_SPECS[i];
    v = std::max(s.lo, std::min(s.hi, v));
    switch (i) {
        case 0: c->genes.alpha             = v; break;
        case 1: c->genes.clip              = v; break;
        case 2: c->genes.outlier_fraction  = v; break;
        case 3: c->genes.moment_mix        = v; break;
        case 4: c->genes.tail_guard        = v; break;
        case 5: c->genes.ternary_threshold = v; break;
        default: break;
    }
}

// Sigma for outlier_fraction is max(value, 0.001) in Python.
static float ts_awq_gene_sigma(int i, float value) {
    if (i == 2) {
        return std::max(value, 0.001f);
    }
    return TS_AWQ_GENE_SPECS[i].sigma;
}

//
// GA operators. random_candidate ports Python's biased init (clip in
// [0.78,1.0], ternary_threshold around 1.0, others uniform in range);
// mutate ports Python's per-gene sigma scales.
//

static ts_awq_candidate ts_awq_random_candidate(ts_awq_rng * rng) {
    ts_awq_candidate c;
    c.expert_hint = -1;
    // alpha, moment_mix: uniform in [lo, hi].
    ts_awq_set_gene(&c, 0, TS_AWQ_GENE_SPECS[0].lo +
                         rng->uniform() * (TS_AWQ_GENE_SPECS[0].hi - TS_AWQ_GENE_SPECS[0].lo));
    // clip: biased to [0.78, 1.0] (Python's init range).
    ts_awq_set_gene(&c, 1, 0.78f + rng->uniform() * (1.0f - 0.78f));
    // outlier_fraction: log-uniform across the full range (Python samples a
    // span around base_fraction/4 .. base_fraction*4; we cover the full
    // [0.0001, 0.05] log range).
    {
        float lo = std::max(TS_AWQ_GENE_SPECS[2].lo, 1e-7f);
        float hi = TS_AWQ_GENE_SPECS[2].hi;
        float v = expf(logf(lo) + rng->uniform() * (logf(hi) - logf(lo)));
        ts_awq_set_gene(&c, 2, v);
    }
    // moment_mix: uniform.
    ts_awq_set_gene(&c, 3, rng->uniform());
    // tail_guard: uniform in [0, 1.0] (Python's init upper bound).
    ts_awq_set_gene(&c, 4, rng->uniform());
    // ternary_threshold: bias around 1.0 - 75% log-uniform in [0.5, 2.0],
    // 25% uniform exploration in [0.4, 2.5] (Python's init).
    {
        float v;
        if (rng->uniform() < 0.75f) {
            v = expf(logf(0.5f) + rng->uniform() * (logf(2.0f) - logf(0.5f)));
        } else {
            v = 0.4f + rng->uniform() * (2.5f - 0.4f);
        }
        ts_awq_set_gene(&c, 5, v);
    }
    return c;
}

static ts_awq_candidate ts_awq_mutate(const ts_awq_candidate * c, ts_awq_rng * rng, float sigma) {
    ts_awq_candidate out = *c;
    for (int i = 0; i < TS_AWQ_N_GENES; i++) {
        float scale = ts_awq_gene_sigma(i, ts_awq_get_gene(c, i));
        ts_awq_set_gene(&out, i, ts_awq_get_gene(c, i) + rng->gauss(sigma) * scale);
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
    // snake_case keys match Python's policy_entry / ts_policy_read, so this
    // round-trips with the policy reader (B4).
    snprintf(buf, sizeof(buf),
             "{\"awq_alpha\":%.6f,\"awq_clip\":%.6f,\"outlier_fraction\":%.6f,"
             "\"moment_mix\":%.6f,\"tail_guard\":%.6f,\"ternary_threshold\":%.6f,"
             "\"expert_hint\":%lld}",
             cand->genes.alpha, cand->genes.clip, cand->genes.outlier_fraction,
             cand->genes.moment_mix, cand->genes.tail_guard,
             cand->genes.ternary_threshold, (long long)cand->expert_hint);
    return std::string(buf);
}

ts_awq_candidate ts_awq_candidate_from_genes(ts_policy_genes genes, int64_t expert_hint) {
    ts_awq_candidate c;
    c.genes       = genes;
    c.expert_hint = expert_hint;
    // Clamp each gene into range so a hand-edited or legacy policy cannot
    // seed the GA with out-of-band values.
    for (int i = 0; i < TS_AWQ_N_GENES; i++) {
        ts_awq_set_gene(&c, i, ts_awq_get_gene(&c, i));
    }
    return c;
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

    // One-shot hypothesis test: when a family seed candidate exists, evaluate
    // it ONCE against this layer's weights. If the composite score is within
    // seed_accept_ratio of the seed's recorded score (i.e. the family optimum
    // transfers to this tensor), accept it directly and skip the GA entirely.
    // This costs 1 evaluation instead of 640+ for tensors that share the
    // family optimum - which is most of them in a transformer (sibling layers
    // at similar depths have nearly identical weight distributions).
    if (params->seed_candidate != nullptr && params->seed_composite > 0.0f &&
        params->seed_accept_ratio > 0.0f) {
        ts_awq_score seed_score = eval(params->seed_candidate, layer, eval_ctx);
        float threshold = params->seed_composite * params->seed_accept_ratio;
        if (seed_score.composite >= threshold) {
            // Family optimum transfers: accept and skip the GA.
            result->best           = *params->seed_candidate;
            result->best_score     = seed_score;
            result->generations_run = 0;
            result->evaluations     = 1;
            result->converged       = true;
            result->warm_started    = true;
            result->family_matched  = true;
            result->archive.clear();
            return 0;
        }
        // Family optimum does NOT transfer (this tensor is an outlier).
        // Fall through to the full GA, warm-started from the seed.
    }

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
        // Family warm-start: inject the seed candidate (from a previously-
        // converged tensor in the same family) into the population, replacing
        // the second member. The first member stays random so the GA still
        // explores. The seed gives the GA a near-optimal starting point so
        // convergence happens in a few generations instead of dozens.
        if (params->seed_candidate != nullptr && pop_size > 1) {
            pops[isl][1] = *params->seed_candidate;
        }
    }

    std::atomic<int64_t> evaluations{0};

    // Parallel candidate evaluation helper: evaluates a batch of (island,
    // index) pairs across n_eval_threads threads. Each candidate is
    // independent; the weight buffer (layer->weights) is shared read-only.
    // gen is the generation index (-1 = initial population) so the optional
    // eval_recorder callback can stream rows tagged with the right generation.
    const int32_t n_eval = std::max((int32_t)1, params->n_eval_threads);
    auto eval_population = [&](std::vector<std::pair<int64_t,int64_t>> & tasks,
                               int64_t gen) {
        const bool has_recorder = (params->eval_recorder != nullptr);
        if (n_eval <= 1 || (int64_t)tasks.size() <= 1) {
            for (auto & t : tasks) {
                scores[t.first][t.second] = eval(&pops[t.first][t.second], layer, eval_ctx);
                evaluations++;
                if (has_recorder) {
                    params->eval_recorder(layer, (int32_t)gen, (int32_t)t.first,
                                          (int32_t)t.second,
                                          &pops[t.first][t.second],
                                          &scores[t.first][t.second],
                                          params->eval_recorder_user);
                }
            }
            return;
        }
        std::atomic<size_t> next(0);
        auto worker = [&]() {
            for (;;) {
                size_t k = next.fetch_add(1, std::memory_order_relaxed);
                if (k >= tasks.size()) return;
                auto & t = tasks[k];
                scores[t.first][t.second] = eval(&pops[t.first][t.second], layer, eval_ctx);
                evaluations.fetch_add(1, std::memory_order_relaxed);
                if (has_recorder) {
                    params->eval_recorder(layer, (int32_t)gen, (int32_t)t.first,
                                          (int32_t)t.second,
                                          &pops[t.first][t.second],
                                          &scores[t.first][t.second],
                                          params->eval_recorder_user);
                }
            }
        };
        std::vector<std::thread> pool;
        pool.reserve((size_t)std::min(n_eval, (int32_t)tasks.size()));
        for (int32_t t = 0; t < std::min(n_eval, (int32_t)tasks.size()); t++) {
            pool.emplace_back(worker);
        }
        for (auto & th : pool) th.join();
    };

    // Evaluate initial populations
    {
        std::vector<std::pair<int64_t,int64_t>> tasks;
        tasks.reserve((size_t)n_islands * (size_t)pop_size);
        for (int64_t isl = 0; isl < n_islands; isl++) {
            for (int64_t i = 0; i < pop_size; i++) {
                tasks.push_back({isl, i});
            }
        }
        eval_population(tasks, -1);
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

    // Early termination: track consecutive generations without improvement.
    // If stagnation_limit > 0 and the best score hasn't improved by more than
    // stagnation_epsilon for that many generations, the GA has converged and
    // we stop early. This cuts typical evolution from 100 gens to ~20-30.
    const bool   use_stagnation = (params->stagnation_limit > 0);
    const float  stag_eps       = params->stagnation_epsilon;
    int64_t      gens_since_improve = 0;
    float        prev_best_composite = best_score.composite;

    // Main GA loop
    int64_t gen = 0;
    for (; gen < n_gen; gen++) {
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

            // Evaluate new population (parallel across the full generation's
            // candidates: all islands x pop_size evaluated at once for better
            // load balancing when some candidates converge faster than others).
            {
                std::vector<std::pair<int64_t,int64_t>> gen_tasks;
                gen_tasks.reserve((size_t)n_islands * (size_t)pop_size);
                for (int64_t j = 0; j < n_islands; j++) {
                    for (int64_t i = 0; i < pop_size; i++) {
                        gen_tasks.push_back({j, i});
                    }
                }
                eval_population(gen_tasks, gen);
                // update_best is serial (mutates best/best_score)
                for (auto & t : gen_tasks) {
                    update_best(pops[t.first][t.second], scores[t.first][t.second]);
                }
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

        // Stagnation check: has the best score improved this generation?
        if (use_stagnation) {
            if (best_score.composite > prev_best_composite + stag_eps) {
                gens_since_improve = 0;
                prev_best_composite = best_score.composite;
            } else {
                gens_since_improve++;
                if (gens_since_improve >= params->stagnation_limit) {
                    // Converged: best has been stable for stagnation_limit
                    // generations, so further iterations are unlikely to
                    // improve. Surfaced via result->converged so callers
                    // (the persistent store, resume set) can treat this
                    // tensor as a stable warm-start seed.
                    result->converged = true;
                    break;
                }
            }
        }
    }

    // Fill result
    result->best = best;
    result->best_score = best_score;
    result->generations_run = gen + 1;  // actual generations completed (gen is 0-indexed)
    result->evaluations = evaluations.load();
    result->warm_started = (params->seed_candidate != nullptr);
    // A tensor is "converged" when the GA either:
    //   (a) hit the stagnation break (result->converged already true), or
    //   (b) ran all generations but the score didn't improve in the last
    //       generation (gens_since_improve > 0 at loop exit).
    // Case (b) covers the common small-budget scenario (evolve_iters=8,
    // stagnation_limit=10) where the loop ends naturally before the
    // stagnation break can trigger. The score IS stable, the GA just
    // didn't have enough generations to prove it via the break.
    if (!result->converged && gen >= n_gen - 1 && gens_since_improve > 0) {
        result->converged = true;
    }
    result->archive.clear();
    for (auto & kv : archive) {
        result->archive.push_back({kv.first, kv.second.cand});
    }

    return 0;
}

int ts_awq_evolve_all(ts_awq_layer * layers, int64_t n_layers,
                      ts_awq_eval_fn eval, void * eval_ctx,
                      const ts_awq_evolve_params * params,
                      std::vector<ts_awq_evolve_result> * results) {
    if (!layers || n_layers < 1 || !eval || !params || !results) {
        return -1;
    }

    results->resize(n_layers);

    // Family warm-start cache: maps family name -> best candidate from a
    // previously-converged tensor in that family. Lives across BOTH the
    // screening and evolution phases so screen results seed the evolution.
    // Layers in the same family (e.g. all blk.*.ffn_up) tend to have similar
    // optimal alpha/clip, so seeding the GA from a sibling's result converges
    // in a few generations instead of dozens. Thread-safe via ts_sharded_map.
    // Family warm-start cache: stores the best candidate AND its composite
    // score per family, so the one-shot hypothesis test can compare against
    // the seed's original fitness.
    struct family_seed_entry {
        ts_awq_candidate candidate;
        float            composite;
    };
    ts_sharded_map<std::string, family_seed_entry> family_seeds;

    // Progressive eval: screen on stratified 25% of layers first, then
    // promote survivors to full eval. For the per-layer GA, this reduces
    // to running a short screen on a subset of layers to warm-start seeds.
    // With <= 4 layers, skip screening. The screen fans out across the same
    // thread pool as the main evolution (layers are independent; the only
    // shared state is eval_ctx, which the caller guards).
    // Screen results are fed into family_seeds so the main evolution starts
    // warm for every family the screen touched (typically all of them, since
    // the stratified sample covers every block).
    if (n_layers > 4) {
        int64_t screen_count = std::max((int64_t)1, n_layers / 4);
        ts_awq_evolve_params screen_params = *params;
        screen_params.generations = std::max((int64_t)1, params->generations / 4);
        screen_params.n_threads    = 1;  // no nested fan-out per layer

        if (params->on_phase_change) {
            params->on_phase_change("screen", screen_count,
                                    params->on_phase_change_user);
        }

        // Stratified screen: evenly spaced layers
        std::vector<int64_t> screen_li(screen_count);
        for (int64_t si = 0; si < screen_count; si++) {
            screen_li[si] = si * (n_layers - 1) / std::max((int64_t)1, screen_count - 1);
        }

        std::vector<ts_awq_evolve_result> screen_results(screen_count);

        const int32_t screen_threads = std::max(1, std::min(
            (int32_t) std::thread::hardware_concurrency(),
            std::min(params->n_threads > 0 ? params->n_threads : 1,
                     (int32_t) screen_count)));

        if (screen_threads <= 1) {
            for (int64_t si = 0; si < screen_count; si++) {
                screen_params.seed = params->seed + (uint32_t)screen_li[si];
                // Reuse family seed from earlier screen layers if available.
                family_seed_entry fam_seed;
                if (family_seeds.find_copy(layers[screen_li[si]].family, fam_seed)) {
                    screen_params.seed_candidate = &fam_seed.candidate;
                    screen_params.seed_composite = fam_seed.composite;
                } else {
                    screen_params.seed_candidate = nullptr;
                }
                int rc = ts_awq_evolve(&layers[screen_li[si]], eval, eval_ctx,
                                       &screen_params, &screen_results[si]);
                if (rc != 0) {
                    return rc;
                }
                // Feed the result back into the family cache for later screen
                // layers AND the main evolution phase.
                family_seeds.assign(layers[screen_li[si]].family,
                                    {screen_results[si].best, screen_results[si].best_score.composite});
                if (params->on_layer_done) {
                    params->on_layer_done(si, screen_count,
                                          layers[screen_li[si]].name.c_str(),
                                          params->on_layer_done_user);
                }
            }
        } else {
            // Atomic work queue over screen indices. Same pattern as the main
            // evolution loop below. screen_results slots do not alias.
            std::atomic<int64_t> next_si(0);
            std::atomic<int>     first_rc(0);
            auto screen_worker = [&]() {
                ts_awq_evolve_params sp = screen_params;
                for (;;) {
                    int64_t si = next_si.fetch_add(1, std::memory_order_relaxed);
                    if (si >= screen_count) return;
                    sp.seed = params->seed + (uint32_t)screen_li[si];
                    family_seed_entry fam_seed;
                    if (family_seeds.find_copy(layers[screen_li[si]].family, fam_seed)) {
                        sp.seed_candidate = &fam_seed.candidate;
                    sp.seed_composite = fam_seed.composite;
                    } else {
                        sp.seed_candidate = nullptr;
                    }
                    int rc = ts_awq_evolve(&layers[screen_li[si]], eval, eval_ctx,
                                           &sp, &screen_results[si]);
                    if (rc != 0) {
                        int expected = 0;
                        first_rc.compare_exchange_strong(expected, rc,
                                                         std::memory_order_relaxed);
                    }
                    family_seeds.assign(layers[screen_li[si]].family,
                                        {screen_results[si].best, screen_results[si].best_score.composite});
                    if (params->on_layer_done) {
                        params->on_layer_done(si, screen_count,
                                              layers[screen_li[si]].name.c_str(),
                                              params->on_layer_done_user);
                    }
                }
            };
            std::vector<std::thread> pool;
            pool.reserve((size_t)screen_threads);
            for (int32_t t = 0; t < screen_threads; t++) {
                pool.emplace_back(screen_worker);
            }
            for (auto & th : pool) th.join();
            int rc = first_rc.load(std::memory_order_relaxed);
            if (rc != 0) {
                return rc;
            }
        }
    }

    // Full per-layer evolution. Each layer's GA (population, archive, rng) is
    // local to one ts_awq_evolve call, so layers are independent and can be
    // fanned out across threads. results slots do not alias. The only shared
    // mutable state is eval_ctx, which the caller must guard (the dispatch
    // eval_ctx carries its own mutex for the sidecar/best-t2 caches). n_threads
    // <= 1 keeps the historical serial behaviour (and the per-layer return-code
    // short-circuit, which the parallel path replaces with a join-then-check).
    const int32_t req_threads = params->n_threads > 1 ? params->n_threads : 1;
    const int32_t hw_threads  = req_threads > 0
        ? std::min(req_threads, (int32_t)std::thread::hardware_concurrency())
        : 1;
    const int32_t n_threads   = std::max(1, std::min(hw_threads, (int32_t)n_layers));

    if (params->on_phase_change) {
        params->on_phase_change("evolve", n_layers,
                                params->on_phase_change_user);
    }

    // family_seeds was declared before the screening phase and is already
    // populated with the screen's best candidate per family. The evolve
    // phase reads and updates it as each layer converges.

    if (n_threads <= 1) {
        for (int64_t i = 0; i < n_layers; i++) {
            // Resume hook: if the caller has a prior converged result for
            // this layer (e.g. from a DuckDB-backed store), skip the GA
            // entirely and use the stored result. The layer's weights are
            // not touched in this path, so no load/release is needed.
            if (params->layer_skip_lookup && (*results)[i].best.genes.alpha == 0.0f) {
                ts_awq_evolve_result skipped = {};
                if (params->layer_skip_lookup(&layers[i], &skipped,
                                              params->layer_skip_lookup_user)) {
                    (*results)[i] = std::move(skipped);
                    family_seeds.assign(layers[i].family, {(*results)[i].best, (*results)[i].best_score.composite});
                    if (params->on_layer_done) {
                        params->on_layer_done(i, n_layers, layers[i].name.c_str(),
                                              params->on_layer_done_user);
                    }
                    continue;
                }
            }
            ts_awq_evolve_params layer_params = *params;
            layer_params.seed = params->seed + (uint32_t)i;
            // Inject the family's best-known candidate if available. Try the
            // in-memory cache first (cheap), then the persistent-store lookup
            // (one SQL query) so the first layer of a family can warm-start
            // from prior runs on disk.
            family_seed_entry fam_seed;
            if (family_seeds.find_copy(layers[i].family, fam_seed)) {
                layer_params.seed_candidate = &fam_seed.candidate;
                    layer_params.seed_composite = fam_seed.composite;
            } else if (params->family_seed_lookup &&
                       params->family_seed_lookup(layers[i].family.c_str(),
                                                  &fam_seed.candidate,
                                                  &fam_seed.composite,
                                                  params->family_seed_lookup_user)) {
                layer_params.seed_candidate = &fam_seed.candidate;
                layer_params.seed_composite = fam_seed.composite;
            }
            // Streaming weight load: if the layer has a load callback, fetch
            // weights on demand instead of holding all layers in memory.
            const float * loaded_weights = nullptr;
            if (layers[i].weights_load_fn) {
                loaded_weights = layers[i].weights_load_fn(layers[i].weights_user_data);
                layers[i].weights = loaded_weights;
            }
            int rc = ts_awq_evolve(&layers[i], eval, eval_ctx, &layer_params, &(*results)[i]);
            if (layers[i].weights_release_fn && loaded_weights) {
                layers[i].weights_release_fn(layers[i].weights_user_data, loaded_weights);
                layers[i].weights = nullptr;
            }
            if (rc != 0) {
                return rc;
            }
            // Store this layer's best as the family seed for subsequent layers.
            family_seeds.assign(layers[i].family, {(*results)[i].best, (*results)[i].best_score.composite});
            if (params->on_layer_done) {
                params->on_layer_done(i, n_layers, layers[i].name.c_str(),
                                      params->on_layer_done_user);
            }
        }
        return 0;
    }

    // Parallel: atomic work queue over layer indices. A shared atomic rc
    // captures the first failure; workers keep draining the queue so a slow
    // layer does not strand the pool. The join below waits for all workers
    // before returning, so results is fully populated on return.
    std::atomic<int64_t>  next_idx(0);
    std::atomic<int>      first_rc(0);
    auto worker = [&]() {
        for (;;) {
            int64_t i = next_idx.fetch_add(1, std::memory_order_relaxed);
            if (i >= n_layers) {
                return;
            }
            // Resume hook (see serial path). (*results)[i] does not alias
            // across workers, so the move is race-free.
            if (params->layer_skip_lookup && (*results)[i].best.genes.alpha == 0.0f) {
                ts_awq_evolve_result skipped = {};
                if (params->layer_skip_lookup(&layers[i], &skipped,
                                              params->layer_skip_lookup_user)) {
                    (*results)[i] = std::move(skipped);
                    family_seeds.assign(layers[i].family, {(*results)[i].best, (*results)[i].best_score.composite});
                    if (params->on_layer_done) {
                        params->on_layer_done(i, n_layers, layers[i].name.c_str(),
                                              params->on_layer_done_user);
                    }
                    continue;
                }
            }
            ts_awq_evolve_params layer_params = *params;
            layer_params.n_threads = 1;  // no nested fan-out
            layer_params.seed = params->seed + (uint32_t)i;
            // Inject the family's best-known candidate if available. The
            // sharded map is thread-safe; if two layers in the same family
            // run concurrently, the second one to finish overwrites the seed,
            // which is fine (both are near-optimal). The persistent-store
            // fallback only fires when the in-memory cache has no entry for
            // this family yet (first layer of the family in this run).
            family_seed_entry fam_seed;
            if (family_seeds.find_copy(layers[i].family, fam_seed)) {
                layer_params.seed_candidate = &fam_seed.candidate;
                    layer_params.seed_composite = fam_seed.composite;
            } else if (params->family_seed_lookup &&
                       params->family_seed_lookup(layers[i].family.c_str(),
                                                  &fam_seed.candidate,
                                                  &fam_seed.composite,
                                                  params->family_seed_lookup_user)) {
                layer_params.seed_candidate = &fam_seed.candidate;
                layer_params.seed_composite = fam_seed.composite;
            }
            // Streaming weight load (see serial path above for details).
            const float * loaded_weights = nullptr;
            if (layers[i].weights_load_fn) {
                loaded_weights = layers[i].weights_load_fn(layers[i].weights_user_data);
                layers[i].weights = loaded_weights;
            }
            int rc = ts_awq_evolve(&layers[i], eval, eval_ctx, &layer_params, &(*results)[i]);
            if (layers[i].weights_release_fn && loaded_weights) {
                layers[i].weights_release_fn(layers[i].weights_user_data, loaded_weights);
                layers[i].weights = nullptr;
            }
            if (rc != 0) {
                int expected = 0;
                first_rc.compare_exchange_strong(expected, rc,
                                                 std::memory_order_relaxed);
            }
            // Store this layer's best as the family seed for subsequent layers.
            family_seeds.assign(layers[i].family, {(*results)[i].best, (*results)[i].best_score.composite});
            if (params->on_layer_done) {
                params->on_layer_done(i, n_layers, layers[i].name.c_str(),
                                      params->on_layer_done_user);
            }
        }
    };

    std::vector<std::thread> pool;
    pool.reserve((size_t)n_threads);
    for (int32_t t = 0; t < n_threads; t++) {
        pool.emplace_back(worker);
    }
    for (auto & th : pool) {
        th.join();
    }

    return first_rc.load(std::memory_order_relaxed);
}
