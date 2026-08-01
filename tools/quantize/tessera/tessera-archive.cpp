#include "tessera-archive.h"
#include "tessera-regime.h"

#include <nlohmann/json.hpp>

#include <cfloat>
#include <cstring>
#include <algorithm>

// --- GA search fitness ---

float ts_search_fitness(const float * t2, const ts_search_config * cfg) {
    if (!t2 || !cfg || cfg->n_layers <= 0) {
        return 0.0f;
    }
    float sum = 0.0f;
    for (int64_t l = 0; l < cfg->n_layers; l++) {
        float alpha = cfg->layer_alpha ? cfg->layer_alpha[l] : 1.0f;
        sum += alpha * t2[l];
    }
    return sum;
}

// --- Regime router ---

ts_expert_id ts_route_expert(float kurtosis, float eff_rank,
                             const char * family) {
    (void)family;
    if (kurtosis > 10.0f) {
        return TS_EXPERT_DARTQUANT;
    }
    if (eff_rank < 0.3f) {
        return TS_EXPERT_FLRQ;
    }
    return TS_EXPERT_AWQ;
}

// --- MAP-Elites archive ---

using json = nlohmann::json;

static const char * TS_ARCHIVE_SCHEMA = "tessera.map-elites-archive.v1";

// FNV-1a 32-bit string hash for the tensor-family axis.
static uint32_t ts_archive_family_hash(const char * s) {
    uint32_t h = 2166136261u;
    for (const unsigned char * p = (const unsigned char *)s; *p; p++) {
        h ^= *p;
        h *= 16777619u;
    }
    return h;
}

// Discretize a descriptor into the four archive axes.
// kurtosis: log10 scale over [1, 100] (reference edges 1, 3, 10, 30, 100).
// eff_rank: linear over [0, 1] (reference edges 0, 0.2, 0.4, 0.6, 0.8, 1.0).
// family:   FNV-1a hash mod n_family_bins.
// modality: direct clamp to [0, n_modality_bins).
static void ts_archive_bins(const ts_map_elites_archive * a,
                            const ts_regime_descriptor * desc,
                            int * k, int * r, int * f, int * m) {
    float kurt = desc->kurtosis;
    if (kurt < 1.0f)   kurt = 1.0f;
    if (kurt > 100.0f) kurt = 100.0f;
    float tk = log10f(kurt) / 2.0f;   // log10(1)=0, log10(100)=2 -> [0, 1]
    *k = (int)floorf(tk * a->n_kurtosis_bins);
    *k = std::max(0, std::min(a->n_kurtosis_bins - 1, *k));

    float er = desc->eff_rank;
    if (er < 0.0f) er = 0.0f;
    if (er > 1.0f) er = 1.0f;
    *r = (int)floorf(er * a->n_rank_bins);
    *r = std::max(0, std::min(a->n_rank_bins - 1, *r));

    *f = (int)(ts_archive_family_hash(desc->family.c_str()) % (uint32_t)a->n_family_bins);

    *m = (int)desc->modality;
    *m = std::max(0, std::min(a->n_modality_bins - 1, *m));
}

static int64_t ts_archive_index(const ts_map_elites_archive * a,
                                int k, int r, int f, int m) {
    return (((int64_t)k * a->n_rank_bins + r) * a->n_family_bins + f) * a->n_modality_bins + m;
}

void ts_archive_init(ts_map_elites_archive * archive,
                     int n_kurtosis_bins, int n_rank_bins,
                     int n_family_bins, int n_modality_bins) {
    if (!archive) {
        return;
    }
    archive->n_kurtosis_bins = std::max(1, n_kurtosis_bins);
    archive->n_rank_bins     = std::max(1, n_rank_bins);
    archive->n_family_bins   = std::max(1, n_family_bins);
    archive->n_modality_bins = std::max(1, n_modality_bins);

    int64_t total = (int64_t)archive->n_kurtosis_bins * archive->n_rank_bins
                  * archive->n_family_bins * archive->n_modality_bins;
    archive->cells.resize(total);

    for (int64_t idx = 0; idx < total; idx++) {
        int m = (int)(idx % archive->n_modality_bins);
        int f = (int)((idx / archive->n_modality_bins) % archive->n_family_bins);
        int r = (int)((idx / ((int64_t)archive->n_modality_bins * archive->n_family_bins))
                      % archive->n_rank_bins);
        int k = (int)(idx / ((int64_t)archive->n_modality_bins * archive->n_family_bins
                      * archive->n_rank_bins));

        ts_archive_cell & c = archive->cells[idx];
        c.kurtosis_bucket = (float)k;
        c.eff_rank_bucket = (float)r;
        c.family_bucket   = f;
        c.modality_bucket = m;
        c.best_fitness    = FLT_MAX;
        c.best_alpha      = 0.0f;
        c.best_clip       = 0.0f;
        c.eval_count      = 0;
        c.tensor_name[0]  = '\0';
    }
}

bool ts_archive_insert(ts_map_elites_archive * archive,
                       const ts_regime_descriptor * desc,
                       float fitness, float alpha, float clip,
                       const char * tensor_name) {
    if (!archive || !desc || archive->cells.empty()) {
        return false;
    }
    int k, r, f, m;
    ts_archive_bins(archive, desc, &k, &r, &f, &m);
    ts_archive_cell & c = archive->cells[ts_archive_index(archive, k, r, f, m)];

    c.eval_count++;

    if (fitness < c.best_fitness) {
        c.best_fitness = fitness;
        c.best_alpha   = alpha;
        c.best_clip    = clip;
        if (tensor_name) {
            strncpy(c.tensor_name, tensor_name, sizeof(c.tensor_name) - 1);
            c.tensor_name[sizeof(c.tensor_name) - 1] = '\0';
        } else {
            c.tensor_name[0] = '\0';
        }
        return true;
    }
    return false;
}

const ts_archive_cell * ts_archive_query(const ts_map_elites_archive * archive,
                                         const ts_regime_descriptor * desc) {
    if (!archive || !desc || archive->cells.empty()) {
        return nullptr;
    }
    int k, r, f, m;
    ts_archive_bins(archive, desc, &k, &r, &f, &m);
    const ts_archive_cell & c = archive->cells[ts_archive_index(archive, k, r, f, m)];
    return c.eval_count > 0 ? &c : nullptr;
}

std::string ts_archive_to_json(const ts_map_elites_archive * archive) {
    if (!archive) {
        return "{}";
    }
    json j;
    j["schema"]           = TS_ARCHIVE_SCHEMA;
    j["n_kurtosis_bins"]  = archive->n_kurtosis_bins;
    j["n_rank_bins"]      = archive->n_rank_bins;
    j["n_family_bins"]    = archive->n_family_bins;
    j["n_modality_bins"]  = archive->n_modality_bins;

    json cells = json::array();
    for (const ts_archive_cell & c : archive->cells) {
        if (c.eval_count <= 0) {
            continue;   // only persist occupied cells
        }
        json jc;
        jc["kurtosis_bucket"] = c.kurtosis_bucket;
        jc["eff_rank_bucket"] = c.eff_rank_bucket;
        jc["family_bucket"]   = c.family_bucket;
        jc["modality_bucket"] = c.modality_bucket;
        jc["best_fitness"]    = c.best_fitness;
        jc["best_alpha"]      = c.best_alpha;
        jc["best_clip"]       = c.best_clip;
        jc["eval_count"]      = (int64_t)c.eval_count;
        jc["tensor_name"]     = std::string(c.tensor_name);
        cells.push_back(jc);
    }
    j["cells"] = cells;
    return j.dump(2);
}

bool ts_archive_from_json(const char * json_str, ts_map_elites_archive * archive) {
    if (!json_str || !archive) {
        return false;
    }
    json j;
    try {
        j = json::parse(json_str);
    } catch (const json::exception &) {
        return false;
    }

    if (!j.is_object() || !j.contains("cells")) {
        return false;
    }

    ts_archive_init(archive,
                    j.value("n_kurtosis_bins", 5),
                    j.value("n_rank_bins", 5),
                    j.value("n_family_bins", 8),
                    j.value("n_modality_bins", 3));

    for (const json & jc : j["cells"]) {
        int k = jc.value("kurtosis_bucket", 0);
        int r = jc.value("eff_rank_bucket", 0);
        int f = jc.value("family_bucket", 0);
        int m = jc.value("modality_bucket", 0);
        k = std::max(0, std::min(archive->n_kurtosis_bins - 1, k));
        r = std::max(0, std::min(archive->n_rank_bins - 1, r));
        f = std::max(0, std::min(archive->n_family_bins - 1, f));
        m = std::max(0, std::min(archive->n_modality_bins - 1, m));

        ts_archive_cell & c = archive->cells[ts_archive_index(archive, k, r, f, m)];
        c.kurtosis_bucket = (float)k;
        c.eff_rank_bucket = (float)r;
        c.family_bucket   = f;
        c.modality_bucket = m;
        c.best_fitness    = jc.value("best_fitness", FLT_MAX);
        c.best_alpha      = jc.value("best_alpha", 0.0f);
        c.best_clip       = jc.value("best_clip", 0.0f);
        c.eval_count      = jc.value("eval_count", (int64_t)0);
        std::string name  = jc.value("tensor_name", std::string());
        strncpy(c.tensor_name, name.c_str(), sizeof(c.tensor_name) - 1);
        c.tensor_name[sizeof(c.tensor_name) - 1] = '\0';
    }
    return true;
}

ts_archive_summary ts_archive_summarize(const ts_map_elites_archive * archive) {
    ts_archive_summary s;
    s.total_cells    = archive ? (int)archive->cells.size() : 0;
    s.occupied_cells = 0;
    s.mean_fitness   = 0.0f;
    s.best_fitness   = 0.0f;
    s.worst_fitness  = 0.0f;

    if (!archive) {
        return s;
    }

    float sum = 0.0f;
    float best = FLT_MAX;
    float worst = -FLT_MAX;
    for (const ts_archive_cell & c : archive->cells) {
        if (c.eval_count <= 0) {
            continue;
        }
        s.occupied_cells++;
        sum += c.best_fitness;
        best  = std::min(best, c.best_fitness);
        worst = std::max(worst, c.best_fitness);
    }

    if (s.occupied_cells > 0) {
        s.mean_fitness  = sum / (float)s.occupied_cells;
        s.best_fitness  = best;
        s.worst_fitness = worst;
    }
    return s;
}
