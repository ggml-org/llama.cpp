#pragma once

#include "ggml-metal-device.h"  // enum ggml_metal_device_id
#include "ggml.h"

#include <cstdint>
#include <vector>

namespace ggml_metal_tuning {

// FA vec selection buckets. ne01 (query rows) splits decode (==1) from batch (>=2), the
// batch side refined into {2,3,4,5}: Q>1 reuses one K/V load across rows, so it only pays
// off once ne01 aligns with Q. ne11 (KV length) is bucketed too, as the Q>1 crossover is
// head-size dependent (small dk crosses late, large dk wins even at short KV).
constexpr int FA_VEC_NE11_BUCKETS[] = { 1024, 4096, 16384 };
constexpr int FA_VEC_NE01_BUCKETS[] = { 2, 3, 4, 5 };

int fa_vec_ne11_bucket(int64_t ne11);
int fa_vec_ne01_bucket(int64_t ne01);

// NE baked into each (dk,dv) baseline instantiation in kernels/fa_vec_*.metal.
// Hand-maintained mirror; keep in sync with those instantiations.
// The Metal test slice covers every legal config for dk=128 and dk=576.
int fa_vec_baseline_ne(int dk, int dv);

// Tuned table has two row kinds. Exact rows key a (ne11_b, ne01_b) bucket. Default rows
// collapse ne11 over one ne01 domain: ne11_b == FA_VEC_NE11_DEFAULT and ne01_b holds the
// domain. fa_vec_pick tries exact bucket -> domain default -> baseline; short KV
// (ne11 < FA_VEC_NE11_BUCKETS[0]) always uses baseline.
constexpr int8_t FA_VEC_NE11_DEFAULT  = -1;
constexpr int8_t FA_VEC_DOMAIN_DECODE = 0;  // ne01 == 1
constexpr int8_t FA_VEC_DOMAIN_BATCH  = 1;  // ne01 >= 2

struct fa_vec_key_t {
    int8_t  family;
    int8_t  dtype;
    int16_t dk;
    int16_t dv;
    int8_t  ne11_b;
    int8_t  ne01_b;
};

static_assert(sizeof(fa_vec_key_t) == 8, "fa_vec_key_t must be tightly packed for memcmp");

struct fa_vec_cfg_t {
    int8_t Q;
    int8_t NE;
};

struct fa_vec_entry_t {
    fa_vec_key_t key;
    fa_vec_cfg_t cfg;
};

// legal NE values for a (dk,dv): NL = 32/NE, require (dk/4)%NL==0 && (dv/4)%NL==0.
// single source shared by the offline tuner and test-backend-ops.
inline std::vector<int> fa_vec_legal_ne(int dk, int dv) {
    std::vector<int> r;
    for (int ne : { 1, 2, 4 }) {
        const int nl = 32 / ne;
        if ((dk / 4) % nl == 0 && (dv / 4) % nl == 0) {
            r.push_back(ne);
        }
    }
    return r;
}

// test/tune-only override; when set, fa_vec_pick returns it directly.
void         fa_vec_set_override(fa_vec_cfg_t cfg);
void         fa_vec_clear_override();
fa_vec_cfg_t fa_vec_baseline_cfg(int dk, int dv);

// Keyed by Apple GPU family; an untuned family matches no row and gets the baseline.
fa_vec_cfg_t fa_vec_pick(int gpu_family, int dtype, int dk, int dv, int64_t ne11, int64_t ne01);

// ---------------------------------------------------------------------------
// mul_mm tile selection
// ---------------------------------------------------------------------------

// Bucket edges partition the (N0, tokens) shape space into cells; one tuned
// row per cell holds the winning tile. Edges are DEVICE-INDEPENDENT: they track
// tile geometry (NR0/NR1 granularity) and real model-dimension clusters, not any
// single GPU's crossover point. Which tile wins inside a cell is per-device data
// held in mm_tile_tuned_table, so a device's flip point is never baked in here.
//
// K (in-feat, ne00) is NOT a key dimension: it scales every candidate's cost
// equally, so it never changes the tile ranking, and keying on it lets exact
// rows memorize deep-K quirks that then misroute unsampled neighbor shapes.
// The sweep still samples deep-K shapes; group_split aggregates across K.
//
// out-feat (ne01): 0=<2048, 1=2048-4607, 2=4608-8191, 3=8192-29999, 4=>=30000 (vocab isolated).
// Edge at 4608 splits the 3584/4096 cluster from 4608/5120: the crossover drifts with N0
// (baseline threadgroup count -> occupancy), so 5120 must not inherit the 3584/4096 winner.
constexpr int MM_TILE_N0_BUCKETS[]     = { 2048, 4608, 8192, 30000 };
// token (ne11): 0=<9, 1=9-31, 2=32-63, 3=64-127, 4=>=128. Split finely across the
// low-token range: the small-tile vs baseline crossover sits here and drifts
// per (dtype,N0), so each side of it gets its own bucket to key a tuned row.
// The edge at 9 separates the small-batch range that mm_tile_ne11_mm_min can route
// into mm (ne11 <= 8) from the range that always used mm: nr1=8 pads half as much
// as nr1=16 at ne11 <= 8, and the two swap ranking between 8 and 16, so one bucket
// spanning both would serve the wrong tile to one of them.
// There is no bucket above 128: mm_tile_pick short-circuits tokens >=
// MM_TILE_TOKENS_MAX_TUNED to baseline (every tuned device converged to baseline
// there), so bucket-4 rows effectively serve [128, MM_TILE_TOKENS_MAX_TUNED).
constexpr int MM_TILE_TOKEN_BUCKETS[]  = { 9, 32, 64, 128 };

int mm_tile_N0_bucket(int64_t N_out);
int mm_tile_token_bucket(int64_t tokens);

struct mm_tile_cfg_t {
    int16_t nr0;  // 32, 64, or 128 (int16: 128 exceeds int8_t range)
    int16_t nr1;  //  8, 16, or  32
};

// Geometries the sweep times, against the baseline anchor below. This is a sweep-budget
// list, NOT a legality constraint: mm_tile_cfg_is_legal defines what the kernel can serve,
// and the two are deliberately separate. nr0 is a function constant and nr1 is instantiated
// for 8/16/32, so a retune can add any legal geometry here - 128x16, 128x32, 96x8 - without
// touching mul_mm.metal. What bounds this list is sweep time: every entry is timed at every
// (dtype, shape, token) cell.
constexpr mm_tile_cfg_t MM_TILE_SWEEP_CANDIDATES[] = {
    { 32, 8 }, { 32, 16 }, { 64, 8 }, { 64, 16 },
};
constexpr mm_tile_cfg_t MM_TILE_BASELINE_CFG = { 64, 32 };

// Legality of a tile geometry, and the single source of truth for it. nr1 selects the
// instantiation (INST_MM_NR1 in mul_mm.metal, plus the bare names for nr1 = 32); nr0 is a
// function constant, so every nr0 that passes here is served by that same instantiation.
// SG_N/TM/TN derive from nr1 alone; nr0 only sets the simdgroup count and the
// shared-memory strides. sg_m >= tn is the "32*(nr0/16) threads must cover every B-tile
// block" condition, i.e. nr0 >= 2*nr1. The host asserts this before specializing a
// pipeline, so the kernel no longer has to static_assert it.
constexpr bool mm_tile_cfg_is_legal(int nr0, int nr1) {
    if (nr1 != 8 && nr1 != 16 && nr1 != 32) { return false; }
    if (nr0 % 16 != 0)                      { return false; }

    const int sg_n = (nr1 == 32) ? 2 : 1;
    if ((nr0 / 16) % sg_n != 0)             { return false; }

    const int sg_m = (nr0 / 16) / sg_n;
    const int tn   = nr1 / (sg_n * 8);
    if (sg_m < tn)                          { return false; }

    return 32 * (nr0 / 16) <= 1024;         // threads per threadgroup
}

static_assert(mm_tile_cfg_is_legal(MM_TILE_BASELINE_CFG.nr0, MM_TILE_BASELINE_CFG.nr1),
              "the baseline tile must be servable");

// Occupancy-saturation threshold for the pick-time veto, measured in baseline
// (64x32) threadgroup count: n_tg = ceil(N_out/64) * ceil(tokens/32). Once the
// baseline dispatch alone saturates the GPU (n_tg >= C_sat), a smaller tile has
// no occupancy headroom left to win, so any non-baseline table row is overridden
// back to baseline. This bounds table extrapolation error on shapes the sweep
// never sampled. Below MM_TILE_OCCUPANCY_MIN_TOKENS the veto is exempt: those
// wins come from baseline's token padding + bc_out slow path, which persist at
// any occupancy.
// Calibrated on M4 Max (40 cores x ~3.6 concurrent tg/core; accuracy is flat
// for C_sat in [128,160]). Per-device data like the tuned table: recalibrate
// when adding rows for a new device. Overestimating C_sat degrades to pure
// table behavior; underestimating only forfeits small-tile wins - neither
// direction can pick something slower than baseline.
// A single M4 Max slot suffices structurally: the veto only fires when a tuned
// row was found, and rows match on exact device_id, so no other device can ever
// consult this constant.
constexpr int MM_TILE_C_SAT_M4_MAX = 144;

// Veto exemption edge, tied to the token bucket that separates the padding-driven
// wins from the occupancy-driven ones.
constexpr int MM_TILE_OCCUPANCY_MIN_TOKENS = MM_TILE_TOKEN_BUCKETS[1];

// Above this the pick short-circuits to baseline. A measured result (every tuned
// device converged to baseline there), not an edge derived from the bucket list.
constexpr int MM_TILE_TOKENS_MAX_TUNED = 256;

// Tuned table has two row kinds.
// Exact rows key a (N0_b, tokens_b) bucket.
// Default rows collapse N0_b but ALWAYS keep tokens_b, per the pick lattice
// (folding order = weakest signal first, tokens never folds):
//   L1 exact -> L2 (N0_b=ANY, keep tokens_b) -> L3 baseline (64x32).
// tokens is the strongest tile signal (small tile at low tokens, large tile at
// prefill), so it is never collapsed. MM_TILE_BUCKET_ANY is the collapse
// sentinel; it applies to N0_b and is compared per-field.
constexpr int8_t MM_TILE_BUCKET_ANY    = -1;

struct mm_tile_key_t {
    int8_t  device_id;
    int8_t  dtype;      // ggml_type of src0
    int8_t  N0_b;       // out-feat (ne01) bucket, or MM_TILE_BUCKET_ANY when collapsed
    int8_t  tokens_b;   // token (ne11) bucket (never collapsed)
};
static_assert(sizeof(mm_tile_key_t) == 4, "mm_tile_key_t must have no padding for memcmp");

struct mm_tile_entry_t {
    mm_tile_key_t key;
    mm_tile_cfg_t cfg;
};

// test/tune-only override; when set, mm_tile_pick returns it directly.
void           mm_tile_set_override(mm_tile_cfg_t cfg);
void           mm_tile_clear_override();
mm_tile_cfg_t  mm_tile_baseline_cfg();

// Returns (64,32) unless a tuned row matches.
mm_tile_cfg_t  mm_tile_pick(enum ggml_metal_device_id device_id,
                             int dtype,
                             int64_t N_out,
                             int64_t tokens);

// Upstream mv_ext -> mm break-even, and the clamp for the tuned values below.
constexpr int MM_TILE_NE11_MM_MIN_DEFAULT = 8;

// Per-device/(dtype, N0-bucket) break-even: the small-batch mm tile can beat mv_ext
// below the default, but only at large N0 (small N0 lacks the occupancy for mm), so
// it is keyed on N0 bucket. Returned V is the upper bound of the mv_ext window:
// ne11 in [2,V] uses mv_ext, ne11 > V uses mm. Clamped to (and defaulting to)
// MM_TILE_NE11_MM_MIN_DEFAULT because mv_ext aborts above it, so an untuned
// (device, dtype, N0) keeps the upstream dispatch. Exact device only, like the
// tile table.
int            mm_tile_ne11_mm_min(enum ggml_metal_device_id device_id, int dtype, int64_t N_out);

// tune-only override for mm_tile_ne11_mm_min; -1 clears it (forces the mm path so
// the sweep can measure mm-tile vs mv_ext in the [2,8] range).
void           mm_tile_set_ne11_mm_min_override(int ne11_mm_min);

// Self-test for the pick lattice (L1 exact -> L2 N0-collapse -> L3 baseline) and
// for mm_tile_pick's short-circuit and occupancy veto. Runs the real lookup
// against a synthetic table, then mm_tile_pick against the tuned one; returns the
// number of failed assertions (0 = pass).
int            mm_tile_lattice_selftest();

}  // namespace ggml_metal_tuning
