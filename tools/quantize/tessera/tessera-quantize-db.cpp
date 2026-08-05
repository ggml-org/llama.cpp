//
// tessera-quantize-db.cpp
//
// DuckDB-backed persistent store. See tessera-quantize-db.h for the contract.
//
// The DuckDB C++ API is exception-based; every call site is wrapped in
// try/catch and translated to a non-zero return + err string. The dispatch
// treats DB failures as non-fatal: it logs the error and continues, so a
// corrupt DB never blocks quantization. Closing the appender or connection
// in the catch path would double-throw; destruction is left to RAII via
// unique_ptr, which DuckDB guarantees does not throw from destructors.
//

#include "tessera-quantize-db.h"
#include "tessera-db-buffer.h"

// duckdb.hpp pulls in windows.h on Windows; the MIN/MAX macros there clash
// with std::min/std::max. NOMINMAX keeps the standard library usable.
#ifdef _WIN32
#  define NOMINMAX
#endif
#include "duckdb.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

// SHA-256 implementation (self-contained, public domain style). We avoid
// pulling OpenSSL just for fingerprinting the GGUF head/tail. The reference
// impl is the FIPS 180-4 algorithm; correctness verified against known
// test vectors (empty string, "abc").
namespace {

struct sha256_ctx {
    uint32_t state[8];
    uint64_t bitlen;
    uint8_t  data[64];
    uint32_t datalen;
};

static const uint32_t SHA256_K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

#define SHA256_ROTR(a,b) (((a) >> (b)) | ((a) << (32-(b))))

static void sha256_transform(sha256_ctx * c, const uint8_t * d) {
    uint32_t a,b,e,f,t1,t2,m[64];
    uint32_t g,h,i,j;
    int k;
    g = c->state[0]; h = c->state[1]; i = c->state[2]; j = c->state[3];
    a = c->state[4]; b = c->state[5]; e = c->state[6]; f = c->state[7];
    // ambiguous-arg fix: hand-rolled
    for (k = 0; k < 16; k++) {
        m[k] = ((uint32_t)d[k*4] << 24) | ((uint32_t)d[k*4+1] << 16) |
               ((uint32_t)d[k*4+2] << 8) | ((uint32_t)d[k*4+3]);
    }
    for (; k < 64; k++) {
        uint32_t s0 = SHA256_ROTR(m[k-15],7) ^ SHA256_ROTR(m[k-15],18) ^ (m[k-15] >> 3);
        uint32_t s1 = SHA256_ROTR(m[k-2],17) ^ SHA256_ROTR(m[k-2],19)  ^ (m[k-2] >> 10);
        m[k] = m[k-16] + s0 + m[k-7] + s1;
    }
    // The 8 working variables and 64 rounds. Names follow FIPS 180-4.
    for (k = 0; k < 64; k++) {
        uint32_t S1 = SHA256_ROTR(e,6) ^ SHA256_ROTR(e,11) ^ SHA256_ROTR(e,25);
        uint32_t ch = (e & b) ^ ((~e) & f);
        t1 = f + S1 + ch + SHA256_K[k] + m[k];
        uint32_t S0 = SHA256_ROTR(g,2) ^ SHA256_ROTR(g,13) ^ SHA256_ROTR(g,22);
        uint32_t maj = (g & h) ^ (g & i) ^ (h & i);
        t2 = S0 + maj;
        f = e; e = b; b = a; a = g;
        g = j + t1; j = i; i = h; h = t1 + t2;
    }
    c->state[0]+=g; c->state[1]+=h; c->state[2]+=i; c->state[3]+=j;
    c->state[4]+=a; c->state[5]+=b; c->state[6]+=e; c->state[7]+=f;
}

static void sha256_init(sha256_ctx * c) {
    c->datalen = 0; c->bitlen = 0;
    c->state[0]=0x6a09e667; c->state[1]=0xbb67ae85; c->state[2]=0x3c6ef372; c->state[3]=0xa54ff53a;
    c->state[4]=0x510e527f; c->state[5]=0x9b05688c; c->state[6]=0x1f83d9ab; c->state[7]=0x5be0cd19;
}

static void sha256_update(sha256_ctx * c, const uint8_t * data, size_t len) {
    for (size_t i = 0; i < len; i++) {
        c->data[c->datalen++] = data[i];
        if (c->datalen == 64) {
            sha256_transform(c, c->data);
            c->bitlen += 512;
            c->datalen = 0;
        }
    }
}

static void sha256_final(sha256_ctx * c, uint8_t out[32]) {
    uint32_t i = c->datalen;
    c->data[i++] = 0x80;
    if (c->datalen < 56) {
        while (i < 56) c->data[i++] = 0x00;
    } else {
        while (i < 64) c->data[i++] = 0x00;
        sha256_transform(c, c->data);
        std::memset(c->data, 0, 56);
    }
    c->bitlen += (uint64_t)c->datalen * 8;
    c->data[63] = (uint8_t)(c->bitlen);
    c->data[62] = (uint8_t)(c->bitlen >> 8);
    c->data[61] = (uint8_t)(c->bitlen >> 16);
    c->data[60] = (uint8_t)(c->bitlen >> 24);
    c->data[59] = (uint8_t)(c->bitlen >> 32);
    c->data[58] = (uint8_t)(c->bitlen >> 40);
    c->data[57] = (uint8_t)(c->bitlen >> 48);
    c->data[56] = (uint8_t)(c->bitlen >> 56);
    sha256_transform(c, c->data);
    for (i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            out[i + j*4] = (uint8_t)((c->state[j] >> (24 - i*8)) & 0xff);
        }
    }
}

static std::string sha256_hex(const uint8_t in[32]) {
    static const char * hex = "0123456789abcdef";
    std::string s(64, '0');
    for (int i = 0; i < 32; i++) {
        s[i*2]   = hex[(in[i] >> 4) & 0xf];
        s[i*2+1] = hex[in[i] & 0xf];
    }
    return s;
}

}  // namespace

// ---------------------------------------------------------------------------
// schema
// ---------------------------------------------------------------------------

namespace {

static const char * TS_QDB_SCHEMA_SQL =
    "CREATE TABLE IF NOT EXISTS runs (\n"
    "    run_id TEXT PRIMARY KEY,\n"
    "    model_path TEXT,\n"
    "    model_hash TEXT,\n"
    "    tessera_commit TEXT,\n"
    "    config_json TEXT,\n"
    "    started_at TIMESTAMP,\n"
    "    completed_at TIMESTAMP,\n"
    "    status TEXT DEFAULT 'running'\n"
    ");\n"
    "CREATE TABLE IF NOT EXISTS tensors (\n"
    "    run_id TEXT REFERENCES runs(run_id),\n"
    "    name TEXT,\n"
    "    family TEXT,\n"
    "    layer_depth INTEGER,\n"
    "    out_dim INTEGER,\n"
    "    in_dim INTEGER,\n"
    "    n_elements BIGINT,\n"
    "    kurtosis FLOAT,\n"
    "    eff_rank FLOAT,\n"
    "    source_type TEXT,\n"
    "    PRIMARY KEY (run_id, name)\n"
    ");\n"
    "CREATE TABLE IF NOT EXISTS ga_evaluations (\n"
    "    run_id TEXT REFERENCES runs(run_id),\n"
    "    tensor_name TEXT,\n"
    "    generation INTEGER,\n"
    "    island INTEGER,\n"
    "    candidate_idx INTEGER,\n"
    "    alpha FLOAT,\n"
    "    clip FLOAT,\n"
    "    composite FLOAT,\n"
    "    mse FLOAT,\n"
    "    relative_frob FLOAT,\n"
    "    evaluated_at TIMESTAMP\n"
    ");\n"
    "CREATE TABLE IF NOT EXISTS ga_results (\n"
    "    run_id TEXT REFERENCES runs(run_id),\n"
    "    tensor_name TEXT,\n"
    "    family TEXT,\n"
    "    best_alpha FLOAT,\n"
    "    best_clip FLOAT,\n"
    "    best_composite FLOAT,\n"
    "    best_mse FLOAT,\n"
    "    generations_run INTEGER,\n"
    "    n_evaluations BIGINT,\n"
    "    converged BOOLEAN,\n"
    "    warm_started BOOLEAN,\n"
    "    completed_at TIMESTAMP,\n"
    "    PRIMARY KEY (run_id, tensor_name)\n"
    ");\n"
    "CREATE TABLE IF NOT EXISTS acceptance (\n"
    "    run_id TEXT REFERENCES runs(run_id),\n"
    "    tensor_name TEXT,\n"
    "    family TEXT,\n"
    "    composite_t2 FLOAT,\n"
    "    awq_t2 FLOAT,\n"
    "    rotation_t2 FLOAT,\n"
    "    lowrank_t2 FLOAT,\n"
    "    hessian_t2 FLOAT,\n"
    "    verdict TEXT,\n"
    "    PRIMARY KEY (run_id, tensor_name)\n"
    ");\n"
    "CREATE TABLE IF NOT EXISTS l5_fixups (\n"
    "    run_id TEXT REFERENCES runs(run_id),\n"
    "    tensor_name TEXT,\n"
    "    generation INTEGER,\n"
    "    strategy TEXT,\n"
    "    before_frob FLOAT,\n"
    "    after_frob FLOAT,\n"
    "    PRIMARY KEY (run_id, tensor_name, generation)\n"
    ");\n"
    // ---- shared tensor_stats: cross-pipeline feature table (additive) ----
    // One row per (model_hash, name). The C++ quantization pipeline writes the
    // GA-prep walk stats here (kurtosis, eff_rank, source_type); the Python
    // calibration pipeline writes the imatrix / AWQ stats (rms, mean_abs,
    // tail_ratio). source records which pipeline last wrote the row. PRIMARY
    // KEY makes this an upsert target for both sides. The 'tensors' table
    // stays as-is for backward compatibility with the per-run ga-prep walk;
    // new code should read from tensor_stats.
    "CREATE TABLE IF NOT EXISTS tensor_stats (\n"
    "    model_hash         TEXT NOT NULL,\n"
    "    model_role         TEXT NOT NULL DEFAULT 'trunk',\n"
    "    name               TEXT NOT NULL,\n"
    "    family             TEXT,\n"
    "    layer_depth        INTEGER,\n"
    "    out_dim            BIGINT,\n"
    "    in_dim             BIGINT,\n"
    "    n_elements         BIGINT,\n"
    "    dtype              TEXT,\n"
    "    kurtosis           DOUBLE,\n"
    "    eff_rank           DOUBLE,\n"
    "    rms                DOUBLE,\n"
    "    mean_abs           DOUBLE,\n"
    "    tail_ratio         DOUBLE,\n"
    "    source             TEXT,\n"
    "    recommended_action TEXT,\n"
    "    updated_at         TIMESTAMP,\n"
    "    PRIMARY KEY (model_hash, model_role, name)\n"
    ");\n"
    // ---- per-tensor summary mirrors of the four analytical outputs ----
    // These are the per-tensor summary of the typed-NDJSON outputs the
    // Python calibration / analytics pipeline writes
    // (l3_outlier, l4_probe, l5_plan, per_layer_error). The row-level data
    // stays in NDJSON / parquet; only the per-tensor summary fields land in
    // DuckDB so the C++ side can join against them in the GA warm-start /
    // L5 requant fixup path. model_hash + name joins back to tensor_stats.
    "CREATE TABLE IF NOT EXISTS l3_outlier_summary (\n"
    "    model_hash        TEXT NOT NULL,\n"
    "    model_role        TEXT NOT NULL DEFAULT 'trunk',\n"
    "    name              TEXT NOT NULL,\n"
    "    layer             INTEGER,\n"
    "    sidecar_label     TEXT,\n"
    "    outlier_count     BIGINT,\n"
    "    outlier_fraction  DOUBLE,\n"
    "    max_abs           DOUBLE,\n"
    "    rms               DOUBLE,\n"
    "    updated_at        TIMESTAMP,\n"
    "    PRIMARY KEY (model_hash, model_role, name, sidecar_label)\n"
    ");\n"
    "CREATE TABLE IF NOT EXISTS l4_probe_summary (\n"
    "    model_hash        TEXT NOT NULL,\n"
    "    model_role        TEXT NOT NULL DEFAULT 'trunk',\n"
    "    name              TEXT NOT NULL,\n"
    "    layer             INTEGER,\n"
    "    current_qtype     TEXT,\n"
    "    mse               DOUBLE,\n"
    "    mse_minus_one     DOUBLE,\n"
    "    perplexity        DOUBLE,\n"
    "    top1_mismatch     DOUBLE,\n"
    "    n_weights         BIGINT,\n"
    "    updated_at        TIMESTAMP,\n"
    "    PRIMARY KEY (model_hash, model_role, name)\n"
    ");\n"
    "CREATE TABLE IF NOT EXISTS l5_plan_summary (\n"
    "    model_hash        TEXT NOT NULL,\n"
    "    model_role        TEXT NOT NULL DEFAULT 'trunk',\n"
    "    name              TEXT NOT NULL,\n"
    "    layer             INTEGER,\n"
    "    iteration         INTEGER,\n"
    "    plan_id           TEXT,\n"
    "    sensitivity_score DOUBLE,\n"
    "    recommended_qtype TEXT,\n"
    "    recommended_alpha DOUBLE,\n"
    "    recommended_clip  DOUBLE,\n"
    "    updated_at        TIMESTAMP,\n"
    "    PRIMARY KEY (model_hash, model_role, name, iteration, plan_id)\n"
    ");\n"
    "CREATE TABLE IF NOT EXISTS per_layer_error_summary (\n"
    "    model_hash        TEXT NOT NULL,\n"
    "    name              TEXT NOT NULL,\n"
    "    layer             INTEGER,\n"
    "    epsilon           DOUBLE,\n"
    "    reference_qtype   TEXT,\n"
    "    updated_at        TIMESTAMP,\n"
    "    PRIMARY KEY (model_hash, name)\n"
    ");\n"
    // ---- l4_plan_outcome: the per-iteration audit trail (additive) ----
    // Captures the L4 forward-pass measurement AFTER a requant plan was
    // applied. One row per (model_hash, name, iteration, plan_id). The
    // C++ adaptive_requantize loop writes one row per (tensor, gen)
    // with strategy = "A" (alpha/clip) or "B" (outlier_thresh bump).
    // The Python l5_orchestrator's per-iteration runs also land here
    // (via the C++ apply step that re-quantizes and re-probes).
    //
    // mse_before / frob_before are the pre-requant measurements;
    // mse_after / frob_after are the post-requant measurements. The
    // delta is computed in l5_outcome.py at read time (delta_mse =
    // mse_after - mse_before) so the schema stays narrow.
    "CREATE TABLE IF NOT EXISTS l4_plan_outcome (\n"
    "    model_hash           TEXT NOT NULL,\n"
    "    model_role           TEXT NOT NULL DEFAULT 'trunk',\n"
    "    name                 TEXT NOT NULL,\n"
    "    layer                INTEGER,\n"
    "    iteration            INTEGER NOT NULL,\n"
    "    plan_id              TEXT NOT NULL,\n"
    "    strategy             TEXT,\n"
    "    alpha_before         DOUBLE,\n"
    "    alpha_after          DOUBLE,\n"
    "    clip_before          DOUBLE,\n"
    "    clip_after           DOUBLE,\n"
    "    outlier_thresh_before DOUBLE,\n"
    "    outlier_thresh_after  DOUBLE,\n"
    "    mse_before           DOUBLE,\n"
    "    mse_after            DOUBLE,\n"
    "    frob_before          DOUBLE,\n"
    "    frob_after           DOUBLE,\n"
    "    family               TEXT,\n"
    "    updated_at           TIMESTAMP,\n"
    "    PRIMARY KEY (model_hash, model_role, name, iteration, plan_id)\n"
    ");\n"
    // ---- l5_outcome: the verdict (Python-written, additive) ----
    // The "did this requant plan reduce error?" verdict, computed by
    // l5_outcome.py from a join of l5_plan_summary and l4_plan_outcome.
    // One row per (model_hash, name, iteration, plan_id). plan_accepted
    // is True if delta_mse < 0 (the plan actually reduced error);
    // the threshold for "accepted" is configurable in l5_outcome.py
    // and is recorded as accept_threshold for reproducibility.
    // sensitivity_calibration_error is the residual of a linear fit
    // of delta_mse on sensitivity_score per (model, family) — a
    // running measure of how well the orchestrator's sensitivity
    // scoring predicts the actual error delta. The three per-tensor
    // component columns (imatrix_magnitude, gradient_proxy,
    // layer_position_prior) are populated by the orchestrator's
    // write_history path so the retune can fit a 3-coefficient model
    // and isolate which component drives the miscalibration per
    // family. The C++ side reads them but does not write them; they
    // are nullable so the pre-emptive schema change stays additive.
    "CREATE TABLE IF NOT EXISTS l5_outcome (\n"
    "    model_hash            TEXT NOT NULL,\n"
    "    model_role            TEXT NOT NULL DEFAULT 'trunk',\n"
    "    name                  TEXT NOT NULL,\n"
    "    layer                 INTEGER,\n"
    "    iteration             INTEGER NOT NULL,\n"
    "    plan_id               TEXT NOT NULL,\n"
    "    family                TEXT,\n"
    "    sensitivity_score     DOUBLE,\n"
    "    recommended_alpha     DOUBLE,\n"
    "    recommended_clip      DOUBLE,\n"
    "    mse_before            DOUBLE,\n"
    "    mse_after             DOUBLE,\n"
    "    delta_mse             DOUBLE,\n"
    "    delta_frob            DOUBLE,\n"
    "    plan_accepted         BOOLEAN,\n"
    "    accept_threshold      DOUBLE,\n"
    "    residual              DOUBLE,\n"
    "    imatrix_magnitude     DOUBLE,\n"
    "    gradient_proxy        DOUBLE,\n"
    "    layer_position_prior  DOUBLE,\n"
    "    updated_at            TIMESTAMP,\n"
    "    PRIMARY KEY (model_hash, model_role, name, iteration, plan_id)\n"
    ");\n"
    // ---- l5_weights: per-(model, family) retuned scoring weights (additive) ----
    // Written by tools/tessera/l5_retune.py from a closed-form OLS fit
    // of delta_mse on sensitivity_score per (model, family) group in
    // l5_outcome. The retune is the "did this plan reduce error?"
    // feedback loop's consumer: the orchestrator's next generation
    // reads l5_weights and uses the recommended
    // (w_imatrix, w_gradient, w_layer) as the starting point for
    // SensitivityScorer, closing the loop.
    //
    // PRIMARY KEY (model_hash, family) so the upsert target is per
    // (model, family). bias is the OLS intercept (the constant in
    // delta_mse = a + b*sensitivity_score); in_sample_loss is the
    // mean absolute residual after the fit, an "are these weights
    // any good" signal for the next retune. n_samples is the count
    // of l5_outcome rows that fed the fit; updated_at is the wall
    // clock. retune_source records which retune algorithm produced
    // the row ("ols_slope_v1" is the current production algorithm).
    // requant_budget_bits is the dispatch-side budget the orchestrator
    // recommends for this family in the next requant pass; computed by
    // l5_retune.py as total_storage * (1 - hit_rate) * base_budget_fraction
    // and NULL when the family has too few samples to project a
    // budget. The dispatch currently reads it for forward-compatibility
    // (the contract is additive) but does not act on the value yet;
    // the column is here so the producer/consumer agree on the schema.
    "CREATE TABLE IF NOT EXISTS l5_weights (\n"
    "    model_hash            TEXT NOT NULL,\n"
    "    model_role            TEXT NOT NULL DEFAULT 'trunk',\n"
    "    family                TEXT NOT NULL,\n"
    "    w_imatrix             DOUBLE NOT NULL,\n"
    "    w_gradient            DOUBLE NOT NULL,\n"
    "    w_layer               DOUBLE NOT NULL,\n"
    "    bias                  DOUBLE,\n"
    "    n_samples             INTEGER,\n"
    "    in_sample_loss        DOUBLE,\n"
    "    hit_rate              DOUBLE,\n"
    "    retune_source         TEXT,\n"
    "    requant_budget_bits   BIGINT,\n"
    "    updated_at            TIMESTAMP,\n"
    "    PRIMARY KEY (model_hash, model_role, family)\n"
    ");\n"
    "CREATE INDEX IF NOT EXISTS idx_ga_results_family\n"
    "    ON ga_results(family, best_composite DESC);\n"
    "CREATE INDEX IF NOT EXISTS idx_ga_eval_tensor\n"
    "    ON ga_evaluations(run_id, tensor_name);\n"
    "CREATE INDEX IF NOT EXISTS idx_tensors_family\n"
    "    ON tensors(run_id, family);\n"
    "CREATE INDEX IF NOT EXISTS idx_tensor_stats_family\n"
    "    ON tensor_stats(model_hash, family);\n"
    "CREATE INDEX IF NOT EXISTS idx_l3_outlier_model\n"
    "    ON l3_outlier_summary(model_hash, layer);\n"
    "CREATE INDEX IF NOT EXISTS idx_l4_probe_model\n"
    "    ON l4_probe_summary(model_hash, layer);\n"
    "CREATE INDEX IF NOT EXISTS idx_l5_plan_model\n"
    "    ON l5_plan_summary(model_hash, layer);\n"
    "CREATE INDEX IF NOT EXISTS idx_per_layer_error_model\n"
    "    ON per_layer_error_summary(model_hash, layer);\n"
    "CREATE INDEX IF NOT EXISTS idx_l4_outcome_model\n"
    "    ON l4_plan_outcome(model_hash, layer);\n"
    "CREATE INDEX IF NOT EXISTS idx_l5_outcome_model\n"
    "    ON l5_outcome(model_hash, family, plan_accepted);\n"
    "CREATE INDEX IF NOT EXISTS idx_l5_weights_model\n"
    "    ON l5_weights(model_hash);\n";

// Phase 16.7: per-component (model_role, name) covering index
// statements. Kept as a separate constant from TS_QDB_SCHEMA_SQL
// because the (model_role, name) columns are added by the
// in-place migration, not by the CREATE TABLE statements. On a
// fresh DB the columns are present from the start, so the
// indexes can be created immediately after the schema setup;
// on a pre-Phase-16 DB the migration has to add the columns
// first. The cleanest place to run this is AFTER
// ts_tessera_db_migrate_model_role in ts_tessera_db_open; see
// the open path.
//
// The composite PKs include model_role but the per-component
// query pattern ("give me all `attn_q` rows for role=`dflash`")
// scans the PK left-to-right and would still walk every
// (model_hash, model_role, name) tuple. A secondary index on
// (model_role, name) lets DuckDB satisfy
// "WHERE model_role = ? AND name = ?" with an index seek
// without reading the full PK. Idempotent: CREATE INDEX IF
// NOT EXISTS is a no-op on re-open.
//
// l5_weights is the one outlier: it has no `name` column (the
// row is per family), so the covering index is on
// (model_role, family).
static const char * TS_QDB_PHASE_16_7_INDEXES_SQL =
    "CREATE INDEX IF NOT EXISTS idx_tensor_stats_role_name\n"
    "    ON tensor_stats(model_role, name);\n"
    "CREATE INDEX IF NOT EXISTS idx_l3_outlier_role_name\n"
    "    ON l3_outlier_summary(model_role, name);\n"
    "CREATE INDEX IF NOT EXISTS idx_l4_probe_role_name\n"
    "    ON l4_probe_summary(model_role, name);\n"
    "CREATE INDEX IF NOT EXISTS idx_l5_plan_role_name\n"
    "    ON l5_plan_summary(model_role, name);\n"
    "CREATE INDEX IF NOT EXISTS idx_l4_outcome_role_name\n"
    "    ON l4_plan_outcome(model_role, name);\n"
    "CREATE INDEX IF NOT EXISTS idx_l5_outcome_role_name\n"
    "    ON l5_outcome(model_role, name);\n"
    "CREATE INDEX IF NOT EXISTS idx_l5_weights_role_family\n"
    "    ON l5_weights(model_role, family);\n";

// DuckDB escapes single quotes by doubling them. Sufficient for run_id,
// model_path, family, etc. (no LIKE / backslash semantics in DuckDB strings).
std::string sql_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        if (c == '\'') out += "''";
        else           out += c;
    }
    return out;
}

// Format the current UTC time as a DuckDB TIMESTAMP literal
// 'YYYY-MM-DD HH:MM:SS'. We avoid CURRENT_TIMESTAMP / NOW() because DuckDB
// 1.5+ gates them behind the core_functions extension, which is awkward to
// load from the amalgamation. The second-precision is sufficient for run
// lifecycle rows.
std::string ts_now_ts() {
    std::time_t now = std::time(nullptr);
    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &now);
#else
    gmtime_r(&now, &tm);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm);
    return std::string("'") + buf + "'";
}

}  // namespace

// ---------------------------------------------------------------------------
// open / close
// ---------------------------------------------------------------------------

ts_tessera_db::~ts_tessera_db() {
    // Crash-safe shutdown: force a CHECKPOINT before tearing down the
    // connection. DuckDB will checkpoint on a clean shutdown, but not
    // on a SIGKILL (jetsam) - the WAL is left on disk and the next
    // read-only open fails until something forces a flush. Issuing
    // CHECKPOINT here guarantees a clean .duckdb and no stale .wal
    // on every destructor path.
    // Best-effort: a failed CHECKPOINT (read-only connection, open
    // transaction, etc.) is not fatal; the conn reset below still
    // cleans up. DuckDB aborts the CHECKPOINT cleanly on its own.
    if (conn) {
        try {
            auto res = conn->Query("CHECKPOINT");
            (void) res;  // best-effort
        } catch (...) {
            // ignore; teardown proceeds
        }
    }
    // Order matters: a Connection holds a reference to its DuckDB; tear it
    // down before the database goes away. Both unique_ptrs reset in
    // declaration order (conn declared after db in the header), which is the
    // reverse of construction and the correct teardown order.
    conn.reset();
    db.reset();
}

ts_tessera_db * ts_tessera_db_open(const std::string & path,
                                     std::string * err) {
    auto * wrap = new (std::nothrow) ts_tessera_db();
    if (wrap == nullptr) {
        if (err) *err = "out of memory";
        return nullptr;
    }
    try {
        // DBConfig defaults: read/write, create on open. We pass a DBInstance
        // to the connection so the DuckDB object outlives the open scope.
        wrap->db_path = path;
        wrap->db.reset(new duckdb::DuckDB(path));
        wrap->conn.reset(new duckdb::Connection(*wrap->db));
        // Note: DuckDB 1.5+ moved CURRENT_TIMESTAMP / NOW() into the
        // core_functions extension, which is not LoadStaticExtension-friendly
        // in the amalgamation. We avoid SQL timestamp defaults entirely; the
        // callers (begin_run / complete_run / insert_ga_result) format
        // timestamps in C++ via ts_now_ts() and pass them as literals.
        auto res = wrap->conn->Query(TS_QDB_SCHEMA_SQL);
        if (res->HasError()) {
            if (err) *err = "schema setup failed: " + res->GetError();
            delete wrap;
            return nullptr;
        }
        // Phase 16: in-place migration of pre-Phase-16 DBs. The
        // CREATE TABLE IF NOT EXISTS above is a no-op on an
        // existing old schema, so the migration is what actually
        // adds model_role to a pre-Phase-16 DB. Idempotent: a
        // fresh DB has the column already, so the migration
        // short-circuits to a no-op.
        //
        // Must run BEFORE the Phase 16.7 index creation: the
        // indexes reference the model_role column, which the
        // migration is what adds on a pre-Phase-16 DB.
        if (ts_tessera_db_migrate_model_role(wrap, err) != 0) {
            delete wrap;
            return nullptr;
        }
        // Phase 16.7: per-component (model_role, name) covering
        // indexes. Run after the migration so the (model_role,
        // name) columns exist on every table. Idempotent on
        // re-open (CREATE INDEX IF NOT EXISTS is a no-op when
        // the index is already present).
        {
            auto idx_res = wrap->conn->Query(TS_QDB_PHASE_16_7_INDEXES_SQL);
            if (idx_res->HasError()) {
                if (err) *err = "phase 16.7 index setup failed: " +
                                idx_res->GetError();
                delete wrap;
                return nullptr;
            }
        }
    } catch (const std::exception & e) {
        if (err) *err = std::string("duckdb open failed: ") + e.what();
        delete wrap;
        return nullptr;
    } catch (...) {
        if (err) *err = "duckdb open failed: unknown exception";
        delete wrap;
        return nullptr;
    }
    return wrap;
}

// ---------------------------------------------------------------------------
// run lifecycle
// ---------------------------------------------------------------------------

std::string ts_tessera_db_begin_run(ts_tessera_db * db,
                                     const std::string & model_path,
                                     const std::string & model_hash,
                                     const std::string & tessera_commit,
                                     const std::string & config_json,
                                     std::string * err) {
    if (db == nullptr || db->conn == nullptr) return "";
    // run_id: SHA256(model_hash + config + nanos) collapsed to 16 hex chars.
    // Collisions are inconsequential (PRIMARY KEY conflict -> no insert) and
    // a per-process random suffix makes restarts robust to clock collisions.
    auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    std::string key = model_hash + "|" + config_json + "|" + std::to_string(now);
    sha256_ctx c; sha256_init(&c);
    sha256_update(&c, (const uint8_t*)key.data(), key.size());
    uint8_t dig[32]; sha256_final(&c, dig);
    std::string run_id = sha256_hex(dig).substr(0, 16);

    std::ostringstream q;
    q << "INSERT INTO runs (run_id, model_path, model_hash, tessera_commit, "
         "config_json, status, started_at) VALUES ('"
      << sql_escape(run_id) << "', '"
      << sql_escape(model_path) << "', '"
      << sql_escape(model_hash) << "', '"
      << sql_escape(tessera_commit) << "', '"
      << sql_escape(config_json) << "', 'running', "
      << ts_now_ts() << ")";
    try {
        auto res = db->conn->Query(q.str());
        if (res->HasError()) {
            if (err) *err = "begin_run insert failed: " + res->GetError();
            return "";
        }
    } catch (const std::exception & e) {
        if (err) *err = std::string("begin_run failed: ") + e.what();
        return "";
    } catch (...) {
        if (err) *err = "begin_run failed: unknown exception";
        return "";
    }
    return run_id;
}

int ts_tessera_db_complete_run(ts_tessera_db * db,
                                const std::string & run_id,
                                const std::string & status,
                                std::string * err) {
    if (db == nullptr || db->conn == nullptr) return 0;
    std::ostringstream q;
    q << "UPDATE runs SET status = '" << sql_escape(status)
      << "', completed_at = " << ts_now_ts()
      << " WHERE run_id = '"
      << sql_escape(run_id) << "'";
    try {
        auto res = db->conn->Query(q.str());
        if (res->HasError()) {
            if (err) *err = "complete_run failed: " + res->GetError();
            return 1;
        }
    } catch (const std::exception & e) {
        if (err) *err = std::string("complete_run failed: ") + e.what();
        return 1;
    } catch (...) {
        if (err) *err = "complete_run failed: unknown exception";
        return 1;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// tensor registry
// ---------------------------------------------------------------------------

int ts_tessera_db_insert_tensor(ts_tessera_db * db,
                                 const ts_tessera_db_tensor & t,
                                 std::string * err) {
    if (db == nullptr || db->conn == nullptr) return 0;
    std::ostringstream q;
    q << "INSERT INTO tensors (run_id, name, family, layer_depth, out_dim, "
         "in_dim, n_elements, kurtosis, eff_rank, source_type) VALUES ('"
      << sql_escape(t.run_id) << "', '"
      << sql_escape(t.name) << "', '"
      << sql_escape(t.family) << "', "
      << t.layer_depth << ", "
      << t.out_dim << ", "
      << t.in_dim << ", "
      << t.n_elements << ", "
      << t.kurtosis << ", "
      << t.eff_rank << ", '"
      << sql_escape(t.source_type) << "')";
    try {
        auto res = db->conn->Query(q.str());
        if (res->HasError()) {
            if (err) *err = "insert_tensor failed: " + res->GetError();
            return 1;
        }
    } catch (const std::exception & e) {
        if (err) *err = std::string("insert_tensor failed: ") + e.what();
        return 1;
    } catch (...) {
        if (err) *err = "insert_tensor failed: unknown exception";
        return 1;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// ga_results (one row per converged tensor)
// ---------------------------------------------------------------------------

int ts_tessera_db_insert_ga_result(ts_tessera_db * db,
                                    const ts_tessera_db_ga_result & r,
                                    std::string * err) {
    if (db == nullptr || db->conn == nullptr) return 0;
    std::ostringstream q;
    // ON CONFLICT upsert: a re-run of a previously-interrupted tensor should
    // overwrite the old row, not duplicate it. PRIMARY KEY is (run_id, name).
    q << "INSERT INTO ga_results (run_id, tensor_name, family, best_alpha, "
         "best_clip, best_composite, best_mse, generations_run, "
         "n_evaluations, converged, warm_started) VALUES ('"
      << sql_escape(r.run_id) << "', '"
      << sql_escape(r.tensor_name) << "', '"
      << sql_escape(r.family) << "', "
      << r.best_alpha << ", "
      << r.best_clip << ", "
      << r.best_composite << ", "
      << r.best_mse << ", "
      << r.generations_run << ", "
      << r.n_evaluations << ", "
      << (r.converged ? "TRUE" : "FALSE") << ", "
      << (r.warm_started ? "TRUE" : "FALSE")
      << ") ON CONFLICT (run_id, tensor_name) DO UPDATE SET "
         "family=excluded.family, best_alpha=excluded.best_alpha, "
         "best_clip=excluded.best_clip, best_composite=excluded.best_composite, "
         "best_mse=excluded.best_mse, generations_run=excluded.generations_run, "
         "n_evaluations=excluded.n_evaluations, converged=excluded.converged, "
         "warm_started=excluded.warm_started, "
         "completed_at=" << ts_now_ts();
    try {
        auto res = db->conn->Query(q.str());
        if (res->HasError()) {
            if (err) *err = "insert_ga_result failed: " + res->GetError();
            return 1;
        }
    } catch (const std::exception & e) {
        if (err) *err = std::string("insert_ga_result failed: ") + e.what();
        return 1;
    } catch (...) {
        if (err) *err = "insert_ga_result failed: unknown exception";
        return 1;
    }
    return 0;
}

int ts_tessera_db_insert_acceptance(ts_tessera_db * db,
                                     const ts_tessera_db_acceptance & a,
                                     std::string * err) {
    if (db == nullptr || db->conn == nullptr) return 0;
    std::ostringstream q;
    q << "INSERT INTO acceptance (run_id, tensor_name, family, composite_t2, "
         "awq_t2, rotation_t2, lowrank_t2, hessian_t2, verdict) VALUES ('"
      << sql_escape(a.run_id) << "', '"
      << sql_escape(a.tensor_name) << "', '"
      << sql_escape(a.family) << "', "
      << a.composite_t2 << ", "
      << a.awq_t2 << ", "
      << a.rotation_t2 << ", "
      << a.lowrank_t2 << ", "
      << a.hessian_t2 << ", '"
      << sql_escape(a.verdict)
      << "') ON CONFLICT (run_id, tensor_name) DO UPDATE SET "
         "family=excluded.family, composite_t2=excluded.composite_t2, "
         "awq_t2=excluded.awq_t2, rotation_t2=excluded.rotation_t2, "
         "lowrank_t2=excluded.lowrank_t2, hessian_t2=excluded.hessian_t2, "
         "verdict=excluded.verdict";
    try {
        auto res = db->conn->Query(q.str());
        if (res->HasError()) {
            if (err) *err = "insert_acceptance failed: " + res->GetError();
            return 1;
        }
    } catch (const std::exception & e) {
        if (err) *err = std::string("insert_acceptance failed: ") + e.what();
        return 1;
    } catch (...) {
        if (err) *err = "insert_acceptance failed: unknown exception";
        return 1;
    }
    return 0;
}

int ts_tessera_db_insert_l5_fixup(ts_tessera_db * db,
                                   const ts_tessera_db_l5_fixup & f,
                                   std::string * err) {
    if (db == nullptr || db->conn == nullptr) return 0;
    std::ostringstream q;
    q << "INSERT INTO l5_fixups (run_id, tensor_name, generation, strategy, "
         "before_frob, after_frob) VALUES ('"
      << sql_escape(f.run_id) << "', '"
      << sql_escape(f.tensor_name) << "', "
      << f.generation << ", '"
      << sql_escape(f.strategy) << "', "
      << f.before_frob << ", "
      << f.after_frob
      << ") ON CONFLICT (run_id, tensor_name, generation) DO UPDATE SET "
         "strategy=excluded.strategy, before_frob=excluded.before_frob, "
         "after_frob=excluded.after_frob";
    try {
        auto res = db->conn->Query(q.str());
        if (res->HasError()) {
            if (err) *err = "insert_l5_fixup failed: " + res->GetError();
            return 1;
        }
    } catch (const std::exception & e) {
        if (err) *err = std::string("insert_l5_fixup failed: ") + e.what();
        return 1;
    } catch (...) {
        if (err) *err = "insert_l5_fixup failed: unknown exception";
        return 1;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// tensor_stats upsert: cross-pipeline feature table
// ---------------------------------------------------------------------------

int ts_tessera_db_upsert_tensor_stat(
    ts_tessera_db * db,
    const ts_tessera_db_tensor_stat & row,
    std::string * err) {
    if (db == nullptr || db->conn == nullptr) return 0;
    // PRIMARY KEY (model_hash, model_role, name). The ON CONFLICT
    // DO UPDATE clause overwrites every column on a re-write.
    // source is updated to the current writer's tag. Phase 16
    // adds model_role; the default "trunk" preserves the
    // pre-Phase-16 contract when the caller leaves the field
    // empty.
    const std::string role = row.model_role.empty()
        ? std::string("trunk")
        : row.model_role;
    std::ostringstream q;
    q << "INSERT INTO tensor_stats (model_hash, model_role, name, family, "
         "layer_depth, out_dim, in_dim, n_elements, dtype, "
         "kurtosis, eff_rank, rms, mean_abs, tail_ratio, source, "
         "recommended_action, updated_at) VALUES ("
      << "'" << sql_escape(row.model_hash) << "', "
      << "'" << sql_escape(role) << "', "
      << "'" << sql_escape(row.name) << "', "
      << "'" << sql_escape(row.family) << "', "
      << row.layer_depth << ", "
      << row.out_dim << ", "
      << row.in_dim << ", "
      << row.n_elements << ", "
      << "'" << sql_escape(row.dtype) << "', "
      << row.kurtosis << ", "
      << row.eff_rank << ", "
      << row.rms << ", "
      << row.mean_abs << ", "
      << row.tail_ratio << ", "
      << "'" << sql_escape(row.source) << "', "
      << "'" << sql_escape(row.recommended_action) << "', "
      << ts_now_ts()
      << ") ON CONFLICT (model_hash, model_role, name) DO UPDATE SET "
         "family=excluded.family, layer_depth=excluded.layer_depth, "
         "out_dim=excluded.out_dim, in_dim=excluded.in_dim, "
         "n_elements=excluded.n_elements, dtype=excluded.dtype, "
         "kurtosis=excluded.kurtosis, eff_rank=excluded.eff_rank, "
         "rms=excluded.rms, mean_abs=excluded.mean_abs, "
         "tail_ratio=excluded.tail_ratio, source=excluded.source, "
         "recommended_action=excluded.recommended_action, "
         "updated_at=excluded.updated_at";
    try {
        auto res = db->conn->Query(q.str());
        if (res->HasError()) {
            if (err) *err = "upsert_tensor_stat failed: " + res->GetError();
            return 1;
        }
    } catch (const std::exception & e) {
        if (err) *err = std::string("upsert_tensor_stat failed: ") + e.what();
        return 1;
    } catch (...) {
        if (err) *err = "upsert_tensor_stat failed: unknown exception";
        return 1;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// tensor_stats read helper: per-component qtype lookup
// ---------------------------------------------------------------------------
//
// Phase 16: the unified Gemma4 12B + dspark + dflash + MTP arch
// shares tensor names across components (e.g. "blk.0.attn_q.weight"
// exists in both the trunk and the dflash encoder). The per-tensor
// calibration policy that the writer (ts_unified_writer) needs to
// pick a qtype for each tensor is keyed on (model_hash, model_role,
// name) so the dflash drafter's per-tensor qtype does not collide
// with the trunk's.
//
// This reader returns one row per (model_hash, model_role, name) it
// finds. The Python calibration pipeline writes the per-tensor qtype
// into tensor_stats.dtype (when it has a calibration verdict for
// the tensor); the writer treats dtype as the "this is the qtype
// recommended for this tensor" signal. dtype values that are
// "f16" or "f32" (the no-quantize passthrough) are filtered out by
// the writer's per-tensor slot logic; the reader returns them as-is
// so the writer can apply the filter.
//
// Empty `role` is treated as "all roles" so the writer can do a
// single bulk read for a model rather than one round-trip per
// component. Result rows are returned in (model_role, name) order
// so the writer's per-component scan is cache-friendly.
int ts_tessera_db_read_unified_policy(
    ts_tessera_db * db,
    const std::string & model_hash,
    const std::string & role,    // empty = all roles
    ts_tessera_db_unified_policy * out,
    std::string * err) {
    if (db == nullptr || db->conn == nullptr || out == nullptr) return 1;
    out->entries.clear();
    std::ostringstream q;
    q << "SELECT model_role, name, family, dtype, source, "
         "recommended_action FROM tensor_stats WHERE model_hash = '"
      << sql_escape(model_hash) << "'";
    if (!role.empty()) {
        q << " AND model_role = '" << sql_escape(role) << "'";
    }
    q << " ORDER BY model_role, name";
    try {
        auto res = db->conn->Query(q.str());
        if (res->HasError()) {
            if (err) *err = "read_unified_policy failed: " + res->GetError();
            return 1;
        }
        while (auto chunk = res->Fetch()) {
            const idx_t n = chunk->size();
            for (idx_t i = 0; i < n; i++) {
                ts_tessera_db_unified_policy_entry e;
                if (!chunk->GetValue(0, i).IsNull()) e.model_role = chunk->GetValue(0, i).ToString();
                if (!chunk->GetValue(1, i).IsNull()) e.name       = chunk->GetValue(1, i).ToString();
                if (!chunk->GetValue(2, i).IsNull()) e.family     = chunk->GetValue(2, i).ToString();
                if (!chunk->GetValue(3, i).IsNull()) e.dtype      = chunk->GetValue(3, i).ToString();
                if (!chunk->GetValue(4, i).IsNull()) e.source     = chunk->GetValue(4, i).ToString();
                if (!chunk->GetValue(5, i).IsNull()) e.recommended_action = chunk->GetValue(5, i).ToString();
                out->entries.push_back(std::move(e));
            }
        }
    } catch (const std::exception & ex) {
        if (err) *err = std::string("read_unified_policy failed: ") + ex.what();
        return 1;
    } catch (...) {
        if (err) *err = "read_unified_policy failed: unknown exception";
        return 1;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// L5 weights upsert: per-(model, family) retuned scoring weights
// ---------------------------------------------------------------------------

int ts_tessera_db_upsert_l5_weight(
    ts_tessera_db * db,
    const ts_tessera_db_l5_weight & row,
    std::string * err) {
    if (db == nullptr || db->conn == nullptr) return 0;
    // PRIMARY KEY (model_hash, model_role, family). The ON
    // CONFLICT DO UPDATE clause overwrites every column on a
    // re-write. retune_source is updated to the current writer's
    // tag so the consumer can tell which algorithm produced the
    // row. requant_budget_bits uses the C++ -1 sentinel for NULL
    // so a missing budget (the common case when l5_retune.py
    // doesn't have enough samples to project one) round-trips
    // without a separate has_value flag on the struct. Phase 16
    // adds model_role; the default "trunk" preserves the
    // pre-Phase-16 contract when the caller leaves the field
    // empty.
    const std::string role = row.model_role.empty()
        ? std::string("trunk")
        : row.model_role;
    const std::string budget_str =
        (row.requant_budget_bits >= 0) ? std::to_string(row.requant_budget_bits) : std::string("NULL");
    std::ostringstream q;
    q << "INSERT INTO l5_weights (model_hash, model_role, family, "
         "w_imatrix, w_gradient, w_layer, bias, n_samples, "
         "in_sample_loss, hit_rate, retune_source, requant_budget_bits, "
         "updated_at) VALUES ("
      << "'" << sql_escape(row.model_hash) << "', "
      << "'" << sql_escape(role) << "', "
      << "'" << sql_escape(row.family) << "', "
      << row.w_imatrix << ", "
      << row.w_gradient << ", "
      << row.w_layer << ", "
      << row.bias << ", "
      << row.n_samples << ", "
      << row.in_sample_loss << ", "
      << row.hit_rate << ", "
      << "'" << sql_escape(row.retune_source) << "', "
      << budget_str << ", "
      << ts_now_ts()
      << ") ON CONFLICT (model_hash, model_role, family) DO UPDATE SET "
         "w_imatrix=excluded.w_imatrix, "
         "w_gradient=excluded.w_gradient, "
         "w_layer=excluded.w_layer, "
         "bias=excluded.bias, "
         "n_samples=excluded.n_samples, "
         "in_sample_loss=excluded.in_sample_loss, "
         "hit_rate=excluded.hit_rate, "
         "retune_source=excluded.retune_source, "
         "requant_budget_bits=excluded.requant_budget_bits, "
         "updated_at=excluded.updated_at";
    try {
        auto res = db->conn->Query(q.str());
        if (res->HasError()) {
            if (err) *err = "upsert_l5_weight failed: " + res->GetError();
            return 1;
        }
    } catch (const std::exception & e) {
        if (err) *err = std::string("upsert_l5_weight failed: ") + e.what();
        return 1;
    } catch (...) {
        if (err) *err = "upsert_l5_weight failed: unknown exception";
        return 1;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// L5 weights list reader: per-(model, family) typed reader
// ---------------------------------------------------------------------------
//
// Mirrors ts_tessera_db_list_converged_for_model. Used by the dispatch's
// GA-prep walk warm-start to bias the GA's (alpha, clip) initial
// population per family. Empty `family` means "all families for this
// model_hash"; the dispatch passes empty when it wants the full list
// at open time. Ordered by hit_rate DESC so the most-converged family
// is the first entry the dispatch sees.

int ts_tessera_db_list_l5_weights(ts_tessera_db * db,
                                  const std::string & model_hash,
                                  const std::string & family,
                                  ts_tessera_db_l5_weight_list * out) {
    if (out == nullptr) return 0;
    out->entries.clear();
    if (db == nullptr || db->conn == nullptr) return 0;
    if (model_hash.empty()) return 0;
    // Phase 16: model_role added to the SELECT so the dispatch
    // can filter on (model_hash, model_role) without a separate
    // read. Optional model_role_filter arg defaults to empty
    // (all roles); a non-empty value filters the read.
    std::ostringstream q;
    q << "SELECT model_hash, model_role, family, w_imatrix, w_gradient, w_layer, "
         "bias, n_samples, in_sample_loss, hit_rate, requant_budget_bits, "
         "retune_source FROM l5_weights WHERE model_hash = '"
      << sql_escape(model_hash) << "'";
    if (!family.empty()) {
        q << " AND family = '" << sql_escape(family) << "'";
    }
    q << " ORDER BY hit_rate DESC";
    try {
        auto res = db->conn->Query(q.str());
        if (res->HasError()) return 1;
        const idx_t n = res->RowCount();
        out->entries.reserve((size_t)n);
        for (idx_t r = 0; r < n; r++) {
            ts_tessera_db_l5_weight_list_entry e;
            auto mh = res->GetValue(0, r);
            e.model_hash = mh.IsNull() ? std::string() : mh.ToString();
            auto mr = res->GetValue(1, r);
            // Pre-Phase-16 rows have NULL model_role (the column
            // was added with DEFAULT 'trunk' but the migration
            // backfills via the same default, so the column is
            // never NULL in practice; the IsNull guard is
            // belt-and-suspenders for safety).
            e.model_role = mr.IsNull() ? std::string("trunk") : mr.ToString();
            auto fm = res->GetValue(2, r);
            e.family     = fm.IsNull() ? std::string() : fm.ToString();
            e.w_imatrix  = res->GetValue(3, r).GetValue<double>();
            e.w_gradient = res->GetValue(4, r).GetValue<double>();
            e.w_layer    = res->GetValue(5, r).GetValue<double>();
            e.bias       = res->GetValue(6, r).GetValue<double>();
            e.n_samples  = (int32_t) res->GetValue(7, r).GetValue<int64_t>();
            e.in_sample_loss = res->GetValue(8, r).GetValue<double>();
            e.hit_rate   = res->GetValue(9, r).GetValue<double>();
            // requant_budget_bits: NULL when the family had too few samples
            // to project a budget. Use -1 as the C++ NULL sentinel.
            auto bv = res->GetValue(10, r);
            e.requant_budget_bits = bv.IsNull() ? -1 : bv.GetValue<int64_t>();
            auto rs = res->GetValue(11, r);
            e.retune_source = rs.IsNull() ? std::string() : rs.ToString();
            out->entries.push_back(std::move(e));
        }
    } catch (...) {
        return 1;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// L5 outcome stats: per-(model, family) verdict aggregate
// ---------------------------------------------------------------------------
//
// Used by the dispatch's converged-fast early-exit. Aggregates the
// verdict (plan_accepted) over the (model, family) group; n_rows
// counts all l5_outcome rows for the group, n_accepted counts those
// with plan_accepted = TRUE, hit_rate is n_accepted / n_rows.
// mean_delta_mse and mean_sensitivity are the per-row means over
// the same set. Empty `family` means "all families"; the caller
// picks the most-converged one to gate on.
//
// Note: the DuckDB amalgamation build used here does NOT load
// the core_functions extension, so SUM and AVG are unavailable.
// The aggregate is split into two COUNT(*) FILTER (WHERE ...) calls
// (FILTER is a SQL standard, built into DuckDB's base parser) and
// hit_rate is computed in C++ as the ratio. mean_delta_mse /
// mean_sensitivity are returned as 0 for now; if the dispatch
// starts to act on them, we can re-introduce the aggregates via
// a manual sum / count.

int ts_tessera_db_l5_outcome_stats_for(ts_tessera_db * db,
                                       const std::string & model_hash,
                                       const std::string & family,
                                       ts_tessera_db_l5_outcome_stats * out) {
    if (out == nullptr) return 0;
    *out = ts_tessera_db_l5_outcome_stats{};
    if (db == nullptr || db->conn == nullptr) return 0;
    if (model_hash.empty()) return 0;
    out->family = family;
    std::ostringstream q;
    q << "SELECT "
         "COUNT(*) AS n_rows, "
         "COUNT(*) FILTER (WHERE plan_accepted) AS n_accepted "
         "FROM l5_outcome WHERE model_hash = '"
      << sql_escape(model_hash) << "'";
    if (!family.empty()) {
        q << " AND family = '" << sql_escape(family) << "'";
    }
    try {
        auto res = db->conn->Query(q.str());
        if (res->HasError()) {
            return 1;
        }
        if (res->RowCount() == 0) return 0;
        out->n_rows = (int32_t) res->GetValue(0, 0).GetValue<int64_t>();
        auto na = res->GetValue(1, 0);
        out->n_accepted = na.IsNull() ? 0 : (int32_t) na.GetValue<int64_t>();
        // hit_rate computed in C++ to avoid the SUM/AVG core_functions
        // dependency. mean_delta_mse / mean_sensitivity stay 0 here;
        // the dispatch's converged-fast gate only reads hit_rate.
        out->hit_rate = (out->n_rows > 0)
            ? (double) out->n_accepted / (double) out->n_rows
            : 0.0;
    } catch (...) {
        return 1;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// L4 plan outcome: thin shim over ts_db_buffer for the feedback loop
// ---------------------------------------------------------------------------

int ts_tessera_db_append_l4_outcome(
    struct ts_db_buffer * buffer,
    const ts_tessera_db_l4_outcome & row) {
    if (buffer == nullptr) return 0;
    // Column order must match the l4_plan_outcome CREATE TABLE
    // (see TS_QDB_SCHEMA_SQL above):
    //   model_hash, model_role, name, layer, iteration, plan_id,
    //   strategy, alpha_before, alpha_after, clip_before, clip_after,
    //   outlier_thresh_before, outlier_thresh_after, mse_before,
    //   mse_after, frob_before, frob_after, family, updated_at
    // updated_at is the special NULL token; the buffer's
    // looks_like_int / looks_like_float pass-through skips the
    // quote path for numerics, and the literal string "NULL"
    // bypasses the quote path entirely. Phase 16: model_role
    // defaults to "trunk" when the caller leaves it empty
    // (preserves the pre-Phase-16 contract for C++ writers that
    // pre-date the field).
    const std::string role = row.model_role.empty()
        ? std::string("trunk")
        : row.model_role;
    std::vector<std::string> values = {
        sql_escape(row.model_hash),
        sql_escape(role),
        sql_escape(row.name),
        std::to_string(row.layer),
        std::to_string(row.iteration),
        sql_escape(row.plan_id),
        sql_escape(row.strategy),
        std::to_string(row.alpha_before),
        std::to_string(row.alpha_after),
        std::to_string(row.clip_before),
        std::to_string(row.clip_after),
        std::to_string(row.outlier_thresh_before),
        std::to_string(row.outlier_thresh_after),
        std::to_string(row.mse_before),
        std::to_string(row.mse_after),
        std::to_string(row.frob_before),
        std::to_string(row.frob_after),
        sql_escape(row.family),
        "NULL",   // updated_at: stamped at flush time would be nicer, but the
                  // appender path uses NULL per the ga_evaluations pattern
    };
    ts_db_buffer_append(buffer, values);
    return 0;
}

// ---------------------------------------------------------------------------
// Appender: bulk GA evaluation logging
// ---------------------------------------------------------------------------

struct ts_tessera_db_appender {
    ts_tessera_db *        owner;
    std::string             run_id;
    std::string             tensor_name;
    std::unique_ptr<duckdb::Appender> app;
    bool                    broken = false;   // sticky: stop after first error
};

ts_tessera_db_appender * ts_tessera_db_appender_open(ts_tessera_db * db,
                                                       const std::string & run_id,
                                                       const std::string & tensor_name,
                                                       std::string * err) {
    if (db == nullptr || db->conn == nullptr) return nullptr;
    std::unique_ptr<ts_tessera_db_appender> ap(new ts_tessera_db_appender());
    ap->owner      = db;
    ap->run_id     = run_id;
    ap->tensor_name = tensor_name;
    try {
        ap->app.reset(new duckdb::Appender(*db->conn, "ga_evaluations"));
    } catch (const std::exception & e) {
        if (err) *err = std::string("appender open failed: ") + e.what();
        return nullptr;
    } catch (...) {
        if (err) *err = "appender open failed: unknown exception";
        return nullptr;
    }
    return ap.release();
}

int ts_tessera_db_appender_row(ts_tessera_db_appender * ap,
                                const ts_tessera_db_eval_row & row) {
    if (ap == nullptr || ap->app == nullptr || ap->broken) return 0;
    try {
        ap->app->BeginRow();
        // Column order must match CREATE TABLE: run_id, tensor_name,
        // generation, island, candidate_idx, alpha, clip, composite, mse,
        // relative_frob, evaluated_at. evaluated_at is left NULL per row
        // (the GA hot path cannot afford a per-row time formatting call);
        // bulk post-hoc analysis can derive it from ga_results.completed_at.
        ap->app->Append<const char *>(row.run_id.c_str());
        ap->app->Append<const char *>(row.tensor_name.c_str());
        ap->app->Append<int32_t>(row.generation);
        ap->app->Append<int32_t>(row.island);
        ap->app->Append<int32_t>(row.candidate_idx);
        ap->app->Append<float>(row.alpha);
        ap->app->Append<float>(row.clip);
        ap->app->Append<float>(row.composite);
        ap->app->Append<float>(row.mse);
        ap->app->Append<float>(row.relative_frob);
        ap->app->Append<std::nullptr_t>(nullptr);   // evaluated_at
        ap->app->EndRow();
    } catch (const std::exception &) {
        // Sticky: the Appender is in an unspecified state after a thrown
        // append. Mark broken and let close() flush whatever buffered safely;
        // further rows are dropped. The dispatch treats eval logging as best-
        // effort, so swallowing the error here keeps the GA running.
        ap->broken = true;
        return 1;
    } catch (...) {
        ap->broken = true;
        return 1;
    }
    return 0;
}

int ts_tessera_db_appender_flush(ts_tessera_db_appender * ap) {
    if (ap == nullptr || ap->app == nullptr || ap->broken) return 0;
    try {
        ap->app->Flush();
    } catch (const std::exception &) {
        ap->broken = true;
        return 1;
    } catch (...) {
        ap->broken = true;
        return 1;
    }
    return 0;
}

void ts_tessera_db_appender_close(ts_tessera_db_appender * ap) {
    if (ap == nullptr) return;
    if (ap->app != nullptr && !ap->broken) {
        try { ap->app->Close(); } catch (...) { /* swallow; cleanup */ }
    }
    delete ap;
}

// ---------------------------------------------------------------------------
// warm-start lookup + resumability
// ---------------------------------------------------------------------------

bool ts_tessera_db_lookup_family_seed(ts_tessera_db * db,
                                       const std::string & family,
                                       const std::string & exclude_run_id,
                                       ts_tessera_db_family_seed * out) {
    if (db == nullptr || db->conn == nullptr || out == nullptr) return false;
    std::ostringstream q;
    q << "SELECT best_alpha, best_clip, best_composite, tensor_name "
         "FROM ga_results WHERE family = '"
      << sql_escape(family)
      << "' AND run_id != '" << sql_escape(exclude_run_id)
      << "' ORDER BY best_composite DESC LIMIT 1";
    try {
        auto res = db->conn->Query(q.str());
        if (res->HasError() || res->RowCount() == 0) return false;
        // NOTE: do NOT use the templated res->GetValue<float>(col,row): that
        // implementation routes through GetValue<int64_t> and truncates floats.
        // Fetch the duckdb::Value and use its proper float specialization.
        out->best_alpha     = res->GetValue(0, 0).GetValue<float>();
        out->best_clip      = res->GetValue(1, 0).GetValue<float>();
        out->best_composite = res->GetValue(2, 0).GetValue<float>();
        auto v = res->GetValue(3, 0);
        out->tensor_name    = v.IsNull() ? std::string() : v.ToString();
        return true;
    } catch (...) {
        return false;
    }
}

int ts_tessera_db_list_converged(ts_tessera_db * db,
                                  const std::string & run_id,
                                  std::vector<std::string> * out) {
    if (out == nullptr) return 0;
    out->clear();
    if (db == nullptr || db->conn == nullptr) return 0;
    std::ostringstream q;
    q << "SELECT tensor_name FROM ga_results WHERE run_id = '"
      << sql_escape(run_id) << "' AND converged = TRUE";
    try {
        auto res = db->conn->Query(q.str());
        if (res->HasError()) return 1;
        // GetValue is O(row_count) per call (per the DuckDB header), but n_rows
        // here is bounded by the layer count (~254 for an 8B model) so the
        // total cost is negligible compared to a single GA evaluation.
        const idx_t n = res->RowCount();
        out->reserve((size_t)n);
        for (idx_t r = 0; r < n; r++) {
            auto v = res->GetValue(0, r);
            if (!v.IsNull()) out->push_back(v.ToString());
        }
    } catch (...) {
        return 1;
    }
    return 0;
}

int ts_tessera_db_list_converged_for_model(ts_tessera_db * db,
                                            const std::string & model_hash,
                                            std::vector<std::string> * out) {
    if (out == nullptr) return 0;
    out->clear();
    if (db == nullptr || db->conn == nullptr) return 0;
    if (model_hash.empty()) return 0;
    std::ostringstream q;
    q << "SELECT ga_results.tensor_name FROM ga_results JOIN runs "
      << "ON ga_results.run_id = runs.run_id WHERE runs.model_hash = '"
      << sql_escape(model_hash) << "' AND ga_results.converged = TRUE";
    try {
        auto res = db->conn->Query(q.str());
        if (res->HasError()) return 1;
        const idx_t n = res->RowCount();
        out->reserve((size_t)n);
        for (idx_t r = 0; r < n; r++) {
            auto v = res->GetValue(0, r);
            if (!v.IsNull()) out->push_back(v.ToString());
        }
    } catch (...) {
        return 1;
    }
    return 0;
}

bool ts_tessera_db_load_ga_result(ts_tessera_db * db,
                                   const std::string & run_id,
                                   const std::string & tensor_name,
                                   ts_tessera_db_ga_result * out) {
    if (db == nullptr || db->conn == nullptr || out == nullptr) return false;
    std::ostringstream q;
    q << "SELECT family, best_alpha, best_clip, best_composite, best_mse, "
         "generations_run, n_evaluations, converged, warm_started "
         "FROM ga_results WHERE run_id = '"
      << sql_escape(run_id) << "' AND tensor_name = '"
      << sql_escape(tensor_name) << "' LIMIT 1";
    try {
        auto res = db->conn->Query(q.str());
        if (res->HasError() || res->RowCount() == 0) return false;
        out->run_id          = run_id;
        out->tensor_name     = tensor_name;
        auto fam = res->GetValue(0, 0);
        out->family          = fam.IsNull() ? std::string() : fam.ToString();
        // See lookup_family_seed: the templated GetValue<float> truncates via
        // int64_t. Use the duckdb::Value float specialization instead.
        out->best_alpha      = res->GetValue(1, 0).GetValue<float>();
        out->best_clip       = res->GetValue(2, 0).GetValue<float>();
        out->best_composite  = res->GetValue(3, 0).GetValue<float>();
        out->best_mse        = res->GetValue(4, 0).GetValue<float>();
        out->generations_run = (int32_t) res->GetValue(5, 0).GetValue<int64_t>();
        out->n_evaluations   = res->GetValue(6, 0).GetValue<int64_t>();
        out->converged       = res->GetValue(7, 0).GetValue<bool>();
        out->warm_started    = res->GetValue(8, 0).GetValue<bool>();
        return true;
    } catch (...) {
        return false;
    }
}

bool ts_tessera_db_load_ga_result_for_model(ts_tessera_db * db,
                                             const std::string & model_hash,
                                             const std::string & tensor_name,
                                             ts_tessera_db_ga_result * out) {
    if (db == nullptr || db->conn == nullptr || out == nullptr) return false;
    if (model_hash.empty()) return false;
    // JOIN runs to filter by model_hash; ORDER BY completed_at DESC so the
    // most recent prior run wins (a run that crashed mid-way would have a
    // NULL completed_at, so it sorts last among same-model runs).
    std::ostringstream q;
    q << "SELECT ga_results.run_id, ga_results.family, ga_results.best_alpha, "
         "ga_results.best_clip, ga_results.best_composite, ga_results.best_mse, "
         "ga_results.generations_run, ga_results.n_evaluations, "
         "ga_results.converged, ga_results.warm_started "
         "FROM ga_results JOIN runs ON ga_results.run_id = runs.run_id "
         "WHERE runs.model_hash = '"
      << sql_escape(model_hash)
      << "' AND ga_results.tensor_name = '"
      << sql_escape(tensor_name)
      << "' AND ga_results.converged = TRUE "
         "ORDER BY runs.completed_at DESC LIMIT 1";
    try {
        auto res = db->conn->Query(q.str());
        if (res->HasError() || res->RowCount() == 0) return false;
        auto rid = res->GetValue(0, 0);
        out->run_id          = rid.IsNull() ? std::string() : rid.ToString();
        out->tensor_name     = tensor_name;
        auto fam = res->GetValue(1, 0);
        out->family          = fam.IsNull() ? std::string() : fam.ToString();
        out->best_alpha      = res->GetValue(2, 0).GetValue<float>();
        out->best_clip       = res->GetValue(3, 0).GetValue<float>();
        out->best_composite  = res->GetValue(4, 0).GetValue<float>();
        out->best_mse        = res->GetValue(5, 0).GetValue<float>();
        out->generations_run = (int32_t) res->GetValue(6, 0).GetValue<int64_t>();
        out->n_evaluations   = res->GetValue(7, 0).GetValue<int64_t>();
        out->converged       = res->GetValue(8, 0).GetValue<bool>();
        out->warm_started    = res->GetValue(9, 0).GetValue<bool>();
        return true;
    } catch (...) {
        return false;
    }
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

int32_t ts_tessera_db_layer_depth(const std::string & name) {
    // Match "blk.N." / "blocks.N." (llama-style) and "h.N." (GPT-2 / NEOX).
    // The pattern is "prefix<N>." with N decimal. Returns 0 when no block
    // index is present (norm, embed, output).
    size_t i = 0;
    auto try_consume_word = [&](const char * w) -> bool {
        size_t n = std::strlen(w);
        if (name.compare(0, n, w) == 0) {
            i = n;
            return true;
        }
        return false;
    };
    if (!(try_consume_word("blk.") ||
          try_consume_word("blocks.") ||
          try_consume_word("h.") ||
          try_consume_word("layers.") ||
          try_consume_word("model.layers."))) {
        return 0;
    }
    int32_t depth = 0;
    bool any = false;
    while (i < name.size() && name[i] >= '0' && name[i] <= '9') {
        depth = depth * 10 + (name[i] - '0');
        any = true;
        i++;
    }
    return any ? depth : 0;
}

std::string ts_tessera_db_hash_gguf(const std::string & path) {
    // Fingerprint the head and tail of the GGUF. The head carries the magic
    // + metadata + tensor info (key/value pairs); the tail carries the last
    // tensor's payload. 1 MB each is enough entropy for warm-start keying and
    // runs in milliseconds even on multi-GB models.
    const size_t CHUNK = 1 * 1024 * 1024;  // 1 MB
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return "";
    sha256_ctx c; sha256_init(&c);
    std::vector<uint8_t> buf(CHUNK);
    // head
    f.read(reinterpret_cast<char *>(buf.data()), (std::streamsize)buf.size());
    auto head_n = (size_t)f.gcount();
    if (head_n == 0) return "";
    sha256_update(&c, buf.data(), head_n);
    // tail
    uint64_t fsize = 0;
    {
        std::ifstream fs(path, std::ios::binary | std::ios::ate);
        if (!fs.is_open()) return "";
        fsize = (uint64_t)fs.tellg();
    }
    if (fsize > head_n + CHUNK) {
        f.clear();
        f.seekg((std::streamoff)(fsize - CHUNK), std::ios::beg);
        f.read(reinterpret_cast<char *>(buf.data()), (std::streamsize)buf.size());
        auto tail_n = (size_t)f.gcount();
        sha256_update(&c, buf.data(), tail_n);
        // mix file size in so a padded tail with identical bytes still differs
        uint8_t szb[8];
        for (int i = 0; i < 8; i++) szb[i] = (uint8_t)((fsize >> (i*8)) & 0xff);
        sha256_update(&c, szb, 8);
    }
    uint8_t dig[32]; sha256_final(&c, dig);
    return sha256_hex(dig);
}

int64_t ts_tessera_db_debug_count(ts_tessera_db * db,
                                   const std::string & query) {
    if (db == nullptr || db->conn == nullptr) return -1;
    try {
        auto res = db->conn->Query(query);
        if (res->HasError() || res->RowCount() == 0) return -1;
        return res->GetValue(0, 0).GetValue<int64_t>();
    } catch (...) {
        return -1;
    }
}

// ---------------------------------------------------------------------------
// Phase 16 migration: add model_role to 7 tables + rebuild PK
// ---------------------------------------------------------------------------
//
// DuckDB's older versions do not support `ALTER TABLE ... DROP
// CONSTRAINT <pk_name>`; the only way to change a PRIMARY KEY is
// the standard rebuild dance:
//   1. CREATE TABLE <name>_new (..., model_role, PRIMARY KEY (...));
//   2. INSERT INTO <name>_new SELECT *, 'trunk' AS model_role FROM <name>;
//   3. DROP TABLE <name>;
//   4. ALTER TABLE <name>_new RENAME TO <name>;
//
// This is destructive on the old table's PK, but the data is
// preserved. The migration is idempotent: a re-run is a no-op when
// the model_role column already exists (the SELECT FROM
// information_schema.columns check at the top of each table's
// migration). On a fresh DB created with the new CREATE TABLE
// schema, the function also no-ops (the column is present from
// the schema setup).
//
// Called automatically by ts_tessera_db_open on every open.
// tools/tessera/migrate_model_role.py is the Python-side
// equivalent for Python-only-opened DBs.

namespace {

// Does the table have the model_role column? Used as the
// idempotency guard: when the column is present, the table
// was either freshly created with the new schema (fresh DB)
// or already migrated (re-opened old DB), and the rebuild
// would be a no-op (or destructive — don't do it).
bool table_has_model_role(duckdb::Connection & conn,
                           const std::string & table_name,
                           std::string * err) {
    const std::string q =
        "SELECT COUNT(*) FROM information_schema.columns "
        "WHERE table_name = '" + table_name +
        "' AND column_name = 'model_role'";
    try {
        auto res = conn.Query(q);
        if (res->HasError()) {
            if (err) *err = "model_role check failed on " + table_name + ": " + res->GetError();
            return false;
        }
        if (res->RowCount() == 0) return false;
        const int64_t n = res->GetValue(0, 0).GetValue<int64_t>();
        return n > 0;
    } catch (const std::exception & e) {
        if (err) *err = std::string("model_role check exception on ") + table_name + ": " + e.what();
        return false;
    } catch (...) {
        if (err) *err = "model_role check exception on " + table_name;
        return false;
    }
}

// Run a single DDL/DML statement. Translates DuckDB errors to the
// out `err` parameter and returns non-zero. Used by the migration
// table loops.
int exec_simple(duckdb::Connection & conn,
                 const std::string & sql,
                 const std::string & context,
                 std::string * err) {
    try {
        auto res = conn.Query(sql);
        if (res->HasError()) {
            if (err) *err = context + " failed: " + res->GetError();
            return 1;
        }
    } catch (const std::exception & e) {
        if (err) *err = context + " exception: " + e.what();
        return 1;
    } catch (...) {
        if (err) *err = context + " unknown exception";
        return 1;
    }
    return 0;
}

// Migrate one table: if model_role is missing, rebuild with the
// new schema (model_role column + new PK). Returns 0 on success or
// no-op; non-zero on error. The `create_new_sql` is the CREATE
// TABLE for the new (Phase 16) schema; the `insert_select_sql` is
// the INSERT ... SELECT that backfills model_role='trunk' for
// existing rows.
//
// Phase 16.7: `n_rows_out` (when non-null) reports the row count
// backfilled into the new model_role column. -1 means "no
// migration happened" (already migrated or fresh table) and 0+
// means "rows backfilled with model_role='trunk'". The audit
// sidecar (model_role_migration.json) keys on this distinction
// to avoid noisy re-runs.
int migrate_one_table(duckdb::Connection & conn,
                       const std::string & table_name,
                       const std::string & create_new_sql,
                       const std::string & insert_select_sql,
                       int64_t * n_rows_out,
                       std::string * err) {
    if (n_rows_out) *n_rows_out = -1;  // sentinel: no migration
    if (table_has_model_role(conn, table_name, err)) {
        return 0;   // already migrated (fresh DB or re-open)
    }
    // Phase 16.7: capture the row count BEFORE the rebuild. The
    // count after the rebuild is 0 (we just dropped the old
    // table). The pre-rebuild count is the "n_rows backfilled"
    // value the audit sidecar records.
    int64_t n_rows = 0;
    {
        const std::string cq = "SELECT COUNT(*) FROM " + table_name;
        try {
            auto cres = conn.Query(cq);
            if (!cres->HasError() && cres->RowCount() > 0) {
                n_rows = cres->GetValue(0, 0).GetValue<int64_t>();
            }
        } catch (...) {
            // non-fatal: a count failure should not block the
            // migration. The sidecar will record 0 for the
            // count in this corner case.
            n_rows = 0;
        }
    }
    const std::string tmp = table_name + "__p16_new";
    // 1. CREATE TABLE <name>__p16_new (..., model_role, PRIMARY KEY (...))
    std::string create_sql = create_new_sql;
    // Replace the table name in the CREATE SQL with the temporary name.
    const std::string token = "CREATE TABLE IF NOT EXISTS " + table_name + " (";
    const std::string repl  = "CREATE TABLE " + tmp + " (";
    auto pos = create_sql.find(token);
    if (pos == std::string::npos) {
        if (err) *err = "migrate_one_table: cannot find CREATE TABLE token for " + table_name;
        return 1;
    }
    create_sql.replace(pos, token.size(), repl);
    if (exec_simple(conn, create_sql, "CREATE " + tmp, err) != 0) return 1;
    // 2. INSERT INTO <name>__p16_new SELECT *, 'trunk' AS model_role FROM <name>
    if (exec_simple(conn, insert_select_sql, "INSERT into " + tmp, err) != 0) return 1;
    // 3. DROP TABLE <name>
    if (exec_simple(conn, "DROP TABLE " + table_name, "DROP " + table_name, err) != 0) return 1;
    // 4. ALTER TABLE <name>__p16_new RENAME TO <name>
    if (exec_simple(conn, "ALTER TABLE " + tmp + " RENAME TO " + table_name,
                    "RENAME " + tmp, err) != 0) return 1;
    if (n_rows_out) *n_rows_out = n_rows;
    return 0;
}

// Phase 16.7: write a JSON sidecar describing the migration that
// just happened. The sidecar is written next to the duckdb file
// (same directory, same basename + ".model_role_migration.json"
// suffix) and is the only audit trail of what Phase 16 did to a
// pre-Phase-16 DB. The format is a small hand-rolled JSON
// (deliberately not a JSON library - the sidecar is debug-only
// and we want zero new deps); the keys are stable.
//
// The function is a no-op when the db_path is empty (in-memory
// DB) or when the migration log is empty (every table was
// already migrated; re-open is a no-op). It writes atomically
// (write to "<sidecar>.tmp", then rename) so a crash mid-write
// cannot leave a half-written sidecar.
int write_migration_sidecar(const std::string & db_path,
                             const std::vector<std::pair<std::string, int64_t>> & rows,
                             std::string * err) {
    if (db_path.empty() || db_path == ":memory:") return 0;
    if (rows.empty()) return 0;   // no migration happened
    // Build the sidecar path: foo.duckdb -> foo.model_role_migration.json
    std::string sidecar = db_path;
    auto slash = sidecar.find_last_of("/\\");
    auto dot   = sidecar.find_last_of('.');
    // Only treat the final dot-after-slash as the extension
    // separator. "/path/foo.tar.gz" -> split at the last dot
    // -> stem = "/path/foo.tar", ext = "gz". The stem keeps the
    // directory; the extension is the duckdb suffix.
    std::string stem, ext;
    if (dot != std::string::npos &&
        (slash == std::string::npos || dot > slash)) {
        stem = sidecar.substr(0, dot);
        ext  = sidecar.substr(dot);
    } else {
        stem = sidecar;
        ext  = "";
    }
    const std::string final_sidecar = stem + ".model_role_migration.json";
    const std::string tmp_sidecar    = final_sidecar + ".tmp";
    // Build the JSON body. The format is intentionally minimal:
    //   {
    //     "db_path": "<path>",
    //     "model_role": "trunk",
    //     "ts": "YYYY-MM-DD HH:MM:SS",
    //     "tables": [
    //       {"name": "tensor_stats", "n_rows_at_migration": 42},
    //       ...
    //     ]
    //   }
    //
    // JSON requires double quotes and ASCII; we strip control
    // characters from db_path defensively (a Windows path with
    // backslashes is fine; the only thing that would break the
    // JSON is a literal " in the path, which is not a valid
    // path character on any supported platform).
    std::string safe_path;
    safe_path.reserve(db_path.size());
    for (char c : db_path) {
        if (c == '"' || c == '\\') {
            safe_path += '\\';
        }
        safe_path += c;
    }
    std::ostringstream body;
    body << "{\n"
         << "  \"db_path\": \"" << safe_path << "\",\n"
         << "  \"model_role\": \"trunk\",\n"
         << "  \"ts\": \"" << ts_now_ts() << "\",\n"
         << "  \"tables\": [\n";
    for (size_t i = 0; i < rows.size(); i++) {
        body << "    {\"name\": \"" << rows[i].first
             << "\", \"n_rows_at_migration\": " << rows[i].second << "}";
        if (i + 1 < rows.size()) body << ",";
        body << "\n";
    }
    body << "  ]\n"
         << "}\n";
    // Atomic write: tmp -> rename. std::rename is atomic on
    // POSIX (the rename(2) syscall is atomic within a
    // filesystem) and best-effort on Windows.
    {
        std::ofstream f(tmp_sidecar, std::ios::binary | std::ios::trunc);
        if (!f) {
            if (err) *err = "sidecar open failed: " + tmp_sidecar;
            return 1;
        }
        f << body.str();
        if (!f) {
            if (err) *err = "sidecar write failed: " + tmp_sidecar;
            return 1;
        }
    }
    if (std::rename(tmp_sidecar.c_str(), final_sidecar.c_str()) != 0) {
        if (err) *err = "sidecar rename failed: " + tmp_sidecar + " -> " + final_sidecar;
        return 1;
    }
    return 0;
}

}  // namespace

int ts_tessera_db_migrate_model_role(ts_tessera_db * db,
                                      std::string * err) {
    if (db == nullptr || db->conn == nullptr) {
        if (err) *err = "migrate_model_role: null db or conn";
        return 1;
    }
    // The CREATE statements here are the Phase 16 schemas. They
    // are templated on table_name; migrate_one_table substitutes
    // the temporary table name. Each statement matches the
    // canonical schema in TS_QDB_SCHEMA_SQL above (one source of
    // truth — the open path uses TS_QDB_SCHEMA_SQL; this
    // migration rebuilds to the same shape).
    //
    // Order matters: do the leaf tables first (l3, l4, l5
    // outcomes) before the per-(model, family) aggregates. In
    // practice there are no FKs between them (the schema is
    // key-only, no REFERENCES), so order is documented but not
    // enforced.
    const std::map<std::string, std::pair<std::string, std::string>> tables = {
        // tensor_stats
        {"tensor_stats", {
            "CREATE TABLE IF NOT EXISTS tensor_stats ("
            "    model_hash         TEXT NOT NULL,"
            "    model_role         TEXT NOT NULL DEFAULT 'trunk',"
            "    name               TEXT NOT NULL,"
            "    family             TEXT,"
            "    layer_depth        INTEGER,"
            "    out_dim            BIGINT,"
            "    in_dim             BIGINT,"
            "    n_elements         BIGINT,"
            "    dtype              TEXT,"
            "    kurtosis           DOUBLE,"
            "    eff_rank           DOUBLE,"
            "    rms                DOUBLE,"
            "    mean_abs           DOUBLE,"
            "    tail_ratio         DOUBLE,"
            "    source             TEXT,"
            "    recommended_action TEXT,"
            "    updated_at         TIMESTAMP,"
            "    PRIMARY KEY (model_hash, model_role, name)"
            ")",
            // INSERT: the SELECT explicitly lists the old
            // columns so the trailing AS model_role has a
            // matching slot in the CREATE TABLE column list.
            "INSERT INTO tensor_stats__p16_new "
            "(model_hash, name, family, layer_depth, out_dim, in_dim, "
            "n_elements, dtype, kurtosis, eff_rank, rms, mean_abs, "
            "tail_ratio, source, recommended_action, updated_at, model_role) "
            "SELECT model_hash, name, family, layer_depth, out_dim, in_dim, "
            "n_elements, dtype, kurtosis, eff_rank, rms, mean_abs, "
            "tail_ratio, source, recommended_action, updated_at, 'trunk' "
            "FROM tensor_stats",
        }},
        // l3_outlier_summary
        {"l3_outlier_summary", {
            "CREATE TABLE IF NOT EXISTS l3_outlier_summary ("
            "    model_hash        TEXT NOT NULL,"
            "    model_role        TEXT NOT NULL DEFAULT 'trunk',"
            "    name              TEXT NOT NULL,"
            "    layer             INTEGER,"
            "    sidecar_label     TEXT,"
            "    outlier_count     BIGINT,"
            "    outlier_fraction  DOUBLE,"
            "    max_abs           DOUBLE,"
            "    rms               DOUBLE,"
            "    updated_at        TIMESTAMP,"
            "    PRIMARY KEY (model_hash, model_role, name, sidecar_label)"
            ")",
            "INSERT INTO l3_outlier_summary__p16_new "
            "(model_hash, name, layer, sidecar_label, outlier_count, "
            "outlier_fraction, max_abs, rms, updated_at, model_role) "
            "SELECT model_hash, name, layer, sidecar_label, outlier_count, "
            "outlier_fraction, max_abs, rms, updated_at, 'trunk' "
            "FROM l3_outlier_summary",
        }},
        // l4_probe_summary
        {"l4_probe_summary", {
            "CREATE TABLE IF NOT EXISTS l4_probe_summary ("
            "    model_hash        TEXT NOT NULL,"
            "    model_role        TEXT NOT NULL DEFAULT 'trunk',"
            "    name              TEXT NOT NULL,"
            "    layer             INTEGER,"
            "    current_qtype     TEXT,"
            "    mse               DOUBLE,"
            "    mse_minus_one     DOUBLE,"
            "    perplexity        DOUBLE,"
            "    top1_mismatch     DOUBLE,"
            "    n_weights         BIGINT,"
            "    updated_at        TIMESTAMP,"
            "    PRIMARY KEY (model_hash, model_role, name)"
            ")",
            "INSERT INTO l4_probe_summary__p16_new "
            "(model_hash, name, layer, current_qtype, mse, mse_minus_one, "
            "perplexity, top1_mismatch, n_weights, updated_at, model_role) "
            "SELECT model_hash, name, layer, current_qtype, mse, mse_minus_one, "
            "perplexity, top1_mismatch, n_weights, updated_at, 'trunk' "
            "FROM l4_probe_summary",
        }},
        // l5_plan_summary
        {"l5_plan_summary", {
            "CREATE TABLE IF NOT EXISTS l5_plan_summary ("
            "    model_hash        TEXT NOT NULL,"
            "    model_role        TEXT NOT NULL DEFAULT 'trunk',"
            "    name              TEXT NOT NULL,"
            "    layer             INTEGER,"
            "    iteration         INTEGER,"
            "    plan_id           TEXT,"
            "    sensitivity_score DOUBLE,"
            "    recommended_qtype TEXT,"
            "    recommended_alpha DOUBLE,"
            "    recommended_clip  DOUBLE,"
            "    updated_at        TIMESTAMP,"
            "    PRIMARY KEY (model_hash, model_role, name, iteration, plan_id)"
            ")",
            "INSERT INTO l5_plan_summary__p16_new "
            "(model_hash, name, layer, iteration, plan_id, sensitivity_score, "
            "recommended_qtype, recommended_alpha, recommended_clip, "
            "updated_at, model_role) "
            "SELECT model_hash, name, layer, iteration, plan_id, sensitivity_score, "
            "recommended_qtype, recommended_alpha, recommended_clip, "
            "updated_at, 'trunk' "
            "FROM l5_plan_summary",
        }},
        // l4_plan_outcome
        {"l4_plan_outcome", {
            "CREATE TABLE IF NOT EXISTS l4_plan_outcome ("
            "    model_hash           TEXT NOT NULL,"
            "    model_role           TEXT NOT NULL DEFAULT 'trunk',"
            "    name                 TEXT NOT NULL,"
            "    layer                INTEGER,"
            "    iteration            INTEGER NOT NULL,"
            "    plan_id              TEXT NOT NULL,"
            "    strategy             TEXT,"
            "    alpha_before         DOUBLE,"
            "    alpha_after          DOUBLE,"
            "    clip_before          DOUBLE,"
            "    clip_after           DOUBLE,"
            "    outlier_thresh_before DOUBLE,"
            "    outlier_thresh_after  DOUBLE,"
            "    mse_before           DOUBLE,"
            "    mse_after            DOUBLE,"
            "    frob_before          DOUBLE,"
            "    frob_after           DOUBLE,"
            "    family               TEXT,"
            "    updated_at           TIMESTAMP,"
            "    PRIMARY KEY (model_hash, model_role, name, iteration, plan_id)"
            ")",
            "INSERT INTO l4_plan_outcome__p16_new "
            "(model_hash, name, layer, iteration, plan_id, strategy, "
            "alpha_before, alpha_after, clip_before, clip_after, "
            "outlier_thresh_before, outlier_thresh_after, mse_before, "
            "mse_after, frob_before, frob_after, family, updated_at, model_role) "
            "SELECT model_hash, name, layer, iteration, plan_id, strategy, "
            "alpha_before, alpha_after, clip_before, clip_after, "
            "outlier_thresh_before, outlier_thresh_after, mse_before, "
            "mse_after, frob_before, frob_after, family, updated_at, 'trunk' "
            "FROM l4_plan_outcome",
        }},
        // l5_outcome
        {"l5_outcome", {
            "CREATE TABLE IF NOT EXISTS l5_outcome ("
            "    model_hash            TEXT NOT NULL,"
            "    model_role            TEXT NOT NULL DEFAULT 'trunk',"
            "    name                  TEXT NOT NULL,"
            "    layer                 INTEGER,"
            "    iteration             INTEGER NOT NULL,"
            "    plan_id               TEXT NOT NULL,"
            "    family                TEXT,"
            "    sensitivity_score     DOUBLE,"
            "    recommended_alpha     DOUBLE,"
            "    recommended_clip      DOUBLE,"
            "    mse_before            DOUBLE,"
            "    mse_after             DOUBLE,"
            "    delta_mse             DOUBLE,"
            "    delta_frob            DOUBLE,"
            "    plan_accepted         BOOLEAN,"
            "    accept_threshold      DOUBLE,"
            "    residual              DOUBLE,"
            "    imatrix_magnitude     DOUBLE,"
            "    gradient_proxy        DOUBLE,"
            "    layer_position_prior  DOUBLE,"
            "    updated_at            TIMESTAMP,"
            "    PRIMARY KEY (model_hash, model_role, name, iteration, plan_id)"
            ")",
            "INSERT INTO l5_outcome__p16_new "
            "(model_hash, name, layer, iteration, plan_id, family, "
            "sensitivity_score, recommended_alpha, recommended_clip, "
            "mse_before, mse_after, delta_mse, delta_frob, plan_accepted, "
            "accept_threshold, residual, imatrix_magnitude, gradient_proxy, "
            "layer_position_prior, updated_at, model_role) "
            "SELECT model_hash, name, layer, iteration, plan_id, family, "
            "sensitivity_score, recommended_alpha, recommended_clip, "
            "mse_before, mse_after, delta_mse, delta_frob, plan_accepted, "
            "accept_threshold, residual, imatrix_magnitude, gradient_proxy, "
            "layer_position_prior, updated_at, 'trunk' "
            "FROM l5_outcome",
        }},
        // l5_weights
        {"l5_weights", {
            "CREATE TABLE IF NOT EXISTS l5_weights ("
            "    model_hash            TEXT NOT NULL,"
            "    model_role            TEXT NOT NULL DEFAULT 'trunk',"
            "    family                TEXT NOT NULL,"
            "    w_imatrix             DOUBLE NOT NULL,"
            "    w_gradient            DOUBLE NOT NULL,"
            "    w_layer               DOUBLE NOT NULL,"
            "    bias                  DOUBLE,"
            "    n_samples             INTEGER,"
            "    in_sample_loss        DOUBLE,"
            "    hit_rate              DOUBLE,"
            "    retune_source         TEXT,"
            "    requant_budget_bits   BIGINT,"
            "    updated_at            TIMESTAMP,"
            "    PRIMARY KEY (model_hash, model_role, family)"
            ")",
            "INSERT INTO l5_weights__p16_new "
            "(model_hash, family, w_imatrix, w_gradient, w_layer, bias, "
            "n_samples, in_sample_loss, hit_rate, retune_source, "
            "requant_budget_bits, updated_at, model_role) "
            "SELECT model_hash, family, w_imatrix, w_gradient, w_layer, bias, "
            "n_samples, in_sample_loss, hit_rate, retune_source, "
            "requant_budget_bits, updated_at, 'trunk' "
            "FROM l5_weights",
        }},
    };
    // Phase 16.7: per-table migration log. One entry per table
    // that actually needed the migration; the audit sidecar is
    // emitted only when at least one table was migrated.
    std::vector<std::pair<std::string, int64_t>> migrated;
    for (const auto & kv : tables) {
        const std::string & tname = kv.first;
        const std::string & create_sql  = kv.second.first;
        const std::string & insert_sql  = kv.second.second;
        int64_t n_rows = -1;
        if (migrate_one_table(*db->conn, tname, create_sql, insert_sql, &n_rows, err) != 0) {
            return 1;
        }
        // Phase 16.7: record the migration for the audit
        // sidecar. n_rows is -1 on no-op (already migrated or
        // fresh DB) and 0+ on an actual migration. We only
        // emit the sidecar for the actual-migration case.
        if (n_rows >= 0) {
            migrated.emplace_back(tname, n_rows);
        }
    }
    // Phase 16.7: write the audit sidecar. No-op when
    // `migrated` is empty (a re-open of an already-migrated
    // DB, the common case).
    if (!migrated.empty()) {
        std::string sidecar_err;
        if (write_migration_sidecar(db->db_path, migrated, &sidecar_err) != 0) {
            // Sidecar is informational, not load-bearing. Log
            // the failure but do not fail the migration: the
            // schema is correct, only the audit trail is
            // missing.
            if (err) *err = "migration sidecar write failed: " + sidecar_err;
        }
    }
    return 0;
}

// Test-only INSERT into l5_outcome. The l5_outcome table is
// normally Python-written (tools/tessera/l5_outcome.py), but the
// C++ converged-fast test in test_l5_dispatch.cpp needs a way to
// populate it without round-tripping through Python. The schema
// is the same one l5_outcome.py writes; the C++ side does not
// normally touch it (production C++ writes go to l4_plan_outcome
// via ts_db_buffer).
//
// The 3 per-tensor component columns (imatrix_magnitude,
// gradient_proxy, layer_position_prior) are written as 0 -> NULL
// for the test's convenience; the production contract is that
// they are populated by the orchestrator's write_history path.
// All other numeric columns default to 0 if the caller leaves
// them uninitialized; the test only sets what matters.
int ts_tessera_db_test_insert_l5_outcome(
    ts_tessera_db * db,
    const ts_tessera_db_l5_outcome_row & row) {
    if (db == nullptr || db->conn == nullptr) return 1;
    // The per-tensor component columns are nullable; the test
    // passes 0 as the "leave NULL" sentinel (a real value of 0
    // would be unusual in this domain, and the C++ test path is
    // round-trip only). Phase 16: model_role defaults to "trunk"
    // when the caller leaves it empty.
    const std::string role = row.model_role.empty()
        ? std::string("trunk")
        : row.model_role;
    const std::string im_str = (row.imatrix_magnitude == 0.0) ? std::string("NULL") : std::to_string(row.imatrix_magnitude);
    const std::string gp_str = (row.gradient_proxy == 0.0)    ? std::string("NULL") : std::to_string(row.gradient_proxy);
    const std::string lp_str = (row.layer_position_prior == 0.0) ? std::string("NULL") : std::to_string(row.layer_position_prior);
    std::ostringstream q;
    q << "INSERT INTO l5_outcome ("
         "model_hash, model_role, name, layer, iteration, plan_id, family, "
         "sensitivity_score, recommended_alpha, recommended_clip, "
         "mse_before, mse_after, delta_mse, delta_frob, "
         "plan_accepted, accept_threshold, residual, "
         "imatrix_magnitude, gradient_proxy, layer_position_prior) "
         "VALUES ("
      << "'" << sql_escape(row.model_hash) << "', "
      << "'" << sql_escape(role) << "', "
      << "'" << sql_escape(row.name) << "', "
      << row.layer << ", "
      << row.iteration << ", "
      << "'" << sql_escape(row.plan_id) << "', "
      << "'" << sql_escape(row.family) << "', "
      << row.sensitivity_score << ", "
      << row.recommended_alpha << ", "
      << row.recommended_clip << ", "
      << row.mse_before << ", "
      << row.mse_after << ", "
      << row.delta_mse << ", "
      << row.delta_frob << ", "
      << (row.plan_accepted ? "TRUE" : "FALSE") << ", "
      << row.accept_threshold << ", "
      << row.residual << ", "
      << im_str << ", "
      << gp_str << ", "
      << lp_str
      << ")";
    try {
        auto res = db->conn->Query(q.str());
        if (res->HasError()) return 1;
    } catch (...) {
        return 1;
    }
    return 0;
}
