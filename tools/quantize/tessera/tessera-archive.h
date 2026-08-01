#pragma once

//
// tessera-archive.h
//
// MAP-Elites quality-diversity archive, GA fitness, regime router,
// and expert enum. A grid of cells indexed by regime descriptors
// (kurtosis, effective rank, tensor family, modality). Each cell keeps
// the best policy found for that regime, turning the single-best GA
// into a regime-indexed archive that preserves one elite per cell.
//

#include <cstdint>
#include <string>
#include <vector>

// full definition lives in tessera-regime.h; only pointers are used here
struct ts_regime_descriptor;

// --- Expert ids + router ---

enum ts_expert_id {
    TS_EXPERT_AWQ       = 0,
    TS_EXPERT_LRQ       = 1,
    TS_EXPERT_DARTQUANT = 2,
    TS_EXPERT_FLRQ      = 3,
    TS_EXPERT_CHAMPQ    = 4,
    TS_EXPERT_SEPTQ     = 5,
    TS_EXPERT_COUNT     = 6,
};

// Route a tensor to its best expert based on regime descriptors.
// kurtosis: activation kurtosis from imatrix.
// eff_rank: effective rank (spectral entropy).
// family: tensor family string ("attn_q", "attn_k", "ffn_gate", etc).
ts_expert_id ts_route_expert(float kurtosis, float eff_rank,
                             const char * family);

// --- GA search fitness ---

struct ts_search_config {
    const float * layer_alpha;  // per-layer HIGGS weights (n_layers,), nullptr = uniform
    int64_t       n_layers;
};

// Composite fitness: Sum_l alpha_l * t_l^2.
// t2: per-layer relative Frobenius errors (n_layers,).
// cfg: search config with optional layer_alpha (nullptr = all alpha_l = 1).
float ts_search_fitness(const float * t2, const ts_search_config * cfg);

// MAP-Elites archive cell: the best policy found for a regime cell.
struct ts_archive_cell {
    float kurtosis_bucket;    // discretized kurtosis bin
    float eff_rank_bucket;    // discretized effective-rank bin
    int   family_bucket;      // hashed tensor family
    int   modality_bucket;    // modality ID (0=text, 1=image, 2=audio)

    float best_fitness;       // Sum alpha_l * t_l^2 (lower is better)
    float best_alpha;         // the AWQ alpha that achieved it
    float best_clip;          // the AWQ clip that achieved it
    int64_t eval_count;       // how many candidates evaluated in this cell
    char    tensor_name[256]; // representative tensor
};

// The archive: a grid of cells indexed by regime descriptors.
// Total cells = product of bins (default 5*5*8*3 = 600).
struct ts_map_elites_archive {
    std::vector<ts_archive_cell> cells;
    int n_kurtosis_bins;   // default 5
    int n_rank_bins;       // default 5
    int n_family_bins;     // default 8
    int n_modality_bins;   // default 3
};

void ts_archive_init(ts_map_elites_archive * archive,
                     int n_kurtosis_bins, int n_rank_bins,
                     int n_family_bins, int n_modality_bins);

// Insert a candidate into the archive. Returns true if it improved the cell.
bool ts_archive_insert(ts_map_elites_archive * archive,
                       const ts_regime_descriptor * desc,
                       float fitness, float alpha, float clip,
                       const char * tensor_name);

// Query the best policy for a regime cell. Returns nullptr if the cell
// is unoccupied.
const ts_archive_cell * ts_archive_query(const ts_map_elites_archive * archive,
                                         const ts_regime_descriptor * desc);

// Serialize to / from JSON (for sidecar persistence).
std::string ts_archive_to_json(const ts_map_elites_archive * archive);
bool ts_archive_from_json(const char * json, ts_map_elites_archive * archive);

// Summary stats.
struct ts_archive_summary {
    int total_cells;
    int occupied_cells;
    float mean_fitness;
    float best_fitness;
    float worst_fitness;
};
ts_archive_summary ts_archive_summarize(const ts_map_elites_archive * archive);
