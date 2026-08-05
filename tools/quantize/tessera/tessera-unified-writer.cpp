//
// tessera-unified-writer.cpp
//
// Implementation of the unified GGUF writer for the Gemma4 12B + dspark +
// dflash + MTP arch. See tessera-unified-writer.h for the design notes
// and the gemma4-assistant arch contract (src/llama-arch.cpp:135,
// src/models/gemma4-assistant.cpp).
//
// The writer copies tensors from 4+ per-component source GGUFs into one
// destination gemma4-assistant GGUF, with optional per-tensor qtype
// overrides from the calibration policy. The Tile640 cluster format is
// preserved end-to-end by per-tensor data-pointer copy.
//

#include "tessera-unified-writer.h"
#include "tessera-gguf-writer.h"

#include "ggml.h"
#include "gguf.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#define TS_PAGE_SIZE      640
#define TS_WORDS_PER_PAGE 32

using json = nlohmann::json;

namespace {

// ---------------------------------------------------------------------------
// Source GGUF context.
//
// Opened via gguf_init_from_file with no_alloc=false. This loads the
// full tensor data into a contiguous blob owned by the gguf_context
// (ctx->data, an internal field). The gguf also creates a ggml_context
// (stored in params.ctx on entry) that holds the tensor descriptors;
// each descriptor's data pointer is set to ctx->data + offset by
// gguf_init_from_file.
//
// The ggml_context is owned by the gguf (gguf_free frees it). We keep
// the ggml_context* so we can call ggml_get_tensor to look up tensor
// data pointers by name.
// ---------------------------------------------------------------------------
struct source_gguf {
    gguf_context * ctx      = nullptr;
    ggml_context * tensor_ctx = nullptr;   // owned by ctx; freed with gguf_free
    std::string    path;

    // Look up a tensor by name; returns the ggml_tensor* from the
    // source's ggml_context. The data pointer is set to the source's
    // loaded data blob + offset.
    ggml_tensor * find(const std::string & name) const {
        if (tensor_ctx == nullptr) return nullptr;
        return ggml_get_tensor(tensor_ctx, name.c_str());
    }
};

// Read a tensor's metadata into a caller-allocated ggml_tensor
// descriptor (for the destination's ggml_context). The destination
// tensor's data pointer is set to the source's data buffer + offset.
//
// The caller passes a ggml_context to use as the scratch allocator
// for the ggml_tensor descriptor. The descriptor is just a struct;
// the writer uses it to call gguf_add_tensor (which copies it by
// value into the gguf's tensor_info), and then the source's data
// buffer outlives the gguf's lifetime (we keep the source alive until
// write_all returns).
ggml_tensor * read_source_tensor(source_gguf & src, ggml_context * dst_gctx,
                                  const std::string & name) {
    int64_t id = gguf_find_tensor(src.ctx, name.c_str());
    if (id < 0) return nullptr;
    const int64_t * ne = gguf_get_tensor_ne(src.ctx, id);
    const ggml_type type = gguf_get_tensor_type(src.ctx, id);
    const size_t   off   = gguf_get_tensor_offset(src.ctx, id);
    int n_dims = (ne[3] > 1) ? 4 : (ne[2] > 1) ? 3 : (ne[1] > 1) ? 2 : 1;
    ggml_tensor * t = nullptr;
    switch (n_dims) {
        case 1: t = ggml_new_tensor_1d(dst_gctx, type, ne[0]); break;
        case 2: t = ggml_new_tensor_2d(dst_gctx, type, ne[0], ne[1]); break;
        case 3: t = ggml_new_tensor_3d(dst_gctx, type, ne[0], ne[1], ne[2]); break;
        case 4: t = ggml_new_tensor_4d(dst_gctx, type, ne[0], ne[1], ne[2], ne[3]); break;
    }
    if (t == nullptr) return nullptr;
    ggml_format_name(t, "%s", name.c_str());
    // Set the data pointer: the source's tensor in the source's
    // ggml_context already has data set to the source's loaded
    // blob + offset. We look that up by name to get the exact
    // pointer.
    ggml_tensor * src_t = src.find(name);
    if (src_t != nullptr && src_t->data != nullptr) {
        t->data = src_t->data;
    } else {
        // Fallback: use the offset to compute the data pointer
        // manually. The source's data blob starts at the offset
        // of the first tensor; we need a way to get the blob base.
        // gguf does not expose ctx->data publicly; if our lookup
        // misses, we cannot recover the data pointer. Signal an
        // error.
        if (src_t == nullptr) return nullptr;
        t->data = (char *)(uintptr_t)off;   // not valid; caller must check
    }
    return t;
}

} // namespace

// ---------------------------------------------------------------------------
// ts_unified_writer::impl
// ---------------------------------------------------------------------------

struct ts_unified_writer::impl {
    std::string                       dst_path;
    std::vector<ts_unified_component> components;
    ts_unified_policy                 policy;
    ts_unified_hparams                hparams;
    ts_unified_dflash_hparams         dflash_hparams;
    ts_unified_dspark_hparams         dspark_hparams;
    ts_unified_mmproj_hparams         mmproj_hparams;   // Phase M0a
    ts_unified_meta                   meta;

    // Phase 16.8: budget relaxation/enforcement evidence recorded by
    // write_all (one event per shared tensor whose budget was applied).
    std::vector<ts_unified_budget_event> budget_events;

    std::vector<source_gguf>          sources;
    gguf_context *                    dst        = nullptr;
    ggml_context *                    dst_gctx   = nullptr;   // owned by us
    // The destination's ggml_init scratch buffer; owned by dst_gctx
    // and freed on ggml_free. Held here only to keep the writer's
    // debug stats honest about how much memory the writer consumed.
    size_t                            dst_gctx_mem = 0;

    impl(const std::string & dst_path_,
         const std::vector<ts_unified_component> & components_,
         const ts_unified_policy & policy_,
         const ts_unified_hparams & hparams_,
         const ts_unified_dflash_hparams & dflash_hparams_,
         const ts_unified_dspark_hparams & dspark_hparams_,
         const ts_unified_mmproj_hparams & mmproj_hparams_,
         const ts_unified_meta & meta_)
        : dst_path(dst_path_), components(components_), policy(policy_),
          hparams(hparams_), dflash_hparams(dflash_hparams_),
          dspark_hparams(dspark_hparams_), mmproj_hparams(mmproj_hparams_),
          meta(meta_) {
        sources.resize(components.size());
        for (size_t i = 0; i < components.size(); i++) {
            sources[i].path = components[i].path;
        }
    }

    ~impl() {
        for (auto & s : sources) {
            if (s.ctx != nullptr) {
                gguf_free(s.ctx);
                s.ctx = nullptr;
                s.tensor_ctx = nullptr;
            }
        }
        if (dst != nullptr)        { gguf_free(dst);       dst      = nullptr; }
        if (dst_gctx != nullptr)   { ggml_free(dst_gctx);  dst_gctx = nullptr; }
    }
};

// ---------------------------------------------------------------------------
// qtype string <-> ggml_type helpers
// ---------------------------------------------------------------------------

int ts_unified_qtype_from_string(const std::string & s) {
    if (s == "F32")            return GGML_TYPE_F32;
    if (s == "F16")            return GGML_TYPE_F16;
    if (s == "BF16")           return GGML_TYPE_BF16;
    if (s == "Q4_0")           return GGML_TYPE_Q4_0;
    if (s == "Q4_1")           return GGML_TYPE_Q4_1;
    if (s == "Q5_0")           return GGML_TYPE_Q5_0;
    if (s == "Q5_1")           return GGML_TYPE_Q5_1;
    if (s == "Q8_0")           return GGML_TYPE_Q8_0;
    if (s == "Q2_K")           return GGML_TYPE_Q2_K;
    if (s == "Q3_K")           return GGML_TYPE_Q3_K;
    if (s == "Q4_K")           return GGML_TYPE_Q4_K;
    if (s == "Q5_K")           return GGML_TYPE_Q5_K;
    if (s == "Q6_K")           return GGML_TYPE_Q6_K;
    if (s == "Q8_K")           return GGML_TYPE_Q8_K;
    if (s == "IQ2_XXS")        return GGML_TYPE_IQ2_XXS;
    if (s == "IQ2_XS")         return GGML_TYPE_IQ2_XS;
    if (s == "IQ3_XXS")        return GGML_TYPE_IQ3_XXS;
    if (s == "IQ3_S")          return GGML_TYPE_IQ3_S;
    if (s == "IQ4_NL")         return GGML_TYPE_IQ4_NL;
    if (s == "IQ4_XS")         return GGML_TYPE_IQ4_XS;
    if (s == "TESSERA_T640")   return GGML_TYPE_TESSERA_T640;
    if (s == "TESSERA_T640_3D")return GGML_TYPE_TESSERA_T640_3D;
    return GGML_TYPE_COUNT;
}

std::string ts_unified_qtype_to_string(int qtype) {
    switch (qtype) {
        case GGML_TYPE_F32:           return "F32";
        case GGML_TYPE_F16:           return "F16";
        case GGML_TYPE_BF16:          return "BF16";
        case GGML_TYPE_Q4_0:          return "Q4_0";
        case GGML_TYPE_Q4_1:          return "Q4_1";
        case GGML_TYPE_Q5_0:          return "Q5_0";
        case GGML_TYPE_Q5_1:          return "Q5_1";
        case GGML_TYPE_Q8_0:          return "Q8_0";
        case GGML_TYPE_Q2_K:          return "Q2_K";
        case GGML_TYPE_Q3_K:          return "Q3_K";
        case GGML_TYPE_Q4_K:          return "Q4_K";
        case GGML_TYPE_Q5_K:          return "Q5_K";
        case GGML_TYPE_Q6_K:          return "Q6_K";
        case GGML_TYPE_Q8_K:          return "Q8_K";
        case GGML_TYPE_IQ2_XXS:       return "IQ2_XXS";
        case GGML_TYPE_IQ2_XS:        return "IQ2_XS";
        case GGML_TYPE_IQ3_XXS:       return "IQ3_XXS";
        case GGML_TYPE_IQ3_S:         return "IQ3_S";
        case GGML_TYPE_IQ4_NL:        return "IQ4_NL";
        case GGML_TYPE_IQ4_XS:        return "IQ4_XS";
        case GGML_TYPE_TESSERA_T640:  return "TESSERA_T640";
        default: return "";
    }
}

// ---------------------------------------------------------------------------
// Phase 16.6: worst-of-trunk-and-dflash qtype resolution
//
// See the header for the rationale. The bit-cost table below is the
// single source of truth for the "more conservative" ordering: a
// higher bit cost means less precision loss. The worst_of helper
// picks max(bits), which is the architect's "take the more
// conservative qtype" rule.
// ---------------------------------------------------------------------------

int ts_unified_writer_qtype_bits(int qtype) {
    // GGML_TYPE_* are an int enum, not strictly ordered by bits.
    // F32 = 0 (anchor: no quantization), F16 = 16 (anchor: fully
    // unquantized), Q2_K..Q8_0 are 2..8 bits per element.
    //
    // F32 and F16 are anchors at the extremes (0 and 16). F32
    // is the "no compression" floor -- worst_of(F32, anything) =
    // anything (any real qtype is more conservative than F32).
    // F16 is the "full precision" ceiling -- worst_of(F16,
    // anything) = F16 (F16 is always the most conservative).
    //
    // Unknown qtypes return 0; this degrades safely in worst_of
    // (an unknown qtype is treated as the lowest-precision
    // anchor, so the known partner wins).
    switch (qtype) {
        case GGML_TYPE_F32:  return 0;
        case GGML_TYPE_F16:  return 16;
        case GGML_TYPE_BF16: return 16;
        case GGML_TYPE_Q2_K: return 2;
        case GGML_TYPE_Q3_K: return 3;
        case GGML_TYPE_Q4_K: return 4;
        case GGML_TYPE_Q5_K: return 5;
        case GGML_TYPE_Q6_K: return 6;
        case GGML_TYPE_Q8_0: return 8;
        case GGML_TYPE_Q4_0: return 4;
        case GGML_TYPE_Q4_1: return 4;
        case GGML_TYPE_Q5_0: return 5;
        case GGML_TYPE_Q5_1: return 5;
        case GGML_TYPE_Q8_K: return 8;
        case GGML_TYPE_IQ2_XXS: return 2;
        case GGML_TYPE_IQ2_XS:  return 2;
        case GGML_TYPE_IQ2_S:   return 2;
        case GGML_TYPE_IQ3_XXS: return 3;
        case GGML_TYPE_IQ3_S:   return 3;
        case GGML_TYPE_IQ4_NL:  return 4;
        case GGML_TYPE_IQ4_XS:  return 4;
        case GGML_TYPE_IQ1_S:   return 1;
        case GGML_TYPE_IQ1_M:   return 1;
        default: return 0;
    }
}

int ts_unified_writer_worst_of(int a, int b) {
    // The architect's rule: pick the more conservative of the two
    // (max(bits)). When equal, return either (we return `a` for
    // determinism). qtype_bits is the single source of truth for
    // the bit cost ordering; the helper handles unknown qtypes
    // (which degrade to 0 bits, so the known partner wins).
    return (ts_unified_writer_qtype_bits(a) >= ts_unified_writer_qtype_bits(b)) ? a : b;
}

// The header stays ggml.h-free and uses TS_UNIFIED_QTYPE_NONE as the
// "no verdict" sentinel; make sure it still matches the enum.
static_assert((int)GGML_TYPE_COUNT == TS_UNIFIED_QTYPE_NONE,
              "TS_UNIFIED_QTYPE_NONE must track GGML_TYPE_COUNT");

// ---------------------------------------------------------------------------
// Phase 16.8: budget-aware cross-role reconciliation
//
// Worst-of is the correctness default, but a role's size budget can be
// pushed over by another role's more-conservative verdict on a shared
// tensor. The architect's rule (2026-08-04): relax the CONSTRAINT - not
// the conservative qtype - when enforcing the budget would compromise a
// needier role. The decision is weighted dynamically. See the header for
// the data model; this is the pure algorithm so it is unit-testable.
// ---------------------------------------------------------------------------

ts_unified_shared_resolution ts_unified_writer_resolve_shared(
    const std::vector<ts_unified_role_verdict> & verdicts) {
    ts_unified_shared_resolution res;

    // Drop absent verdicts. This mirrors qtype_for_tensor's filter
    // (unknown dtype -> GGML_TYPE_COUNT) so the no-budget path below
    // reproduces plain worst-of exactly.
    std::vector<const ts_unified_role_verdict *> valid;
    for (const auto & v : verdicts) {
        if (v.qtype == GGML_TYPE_COUNT) continue;
        valid.push_back(&v);
    }
    if (valid.empty()) {
        res.reason = "no valid verdicts";
        return res;
    }

    // Conservative default: worst-of across every role's verdict.
    int W = valid[0]->qtype;
    for (size_t i = 1; i < valid.size(); i++) {
        W = ts_unified_writer_worst_of(W, valid[i]->qtype);
    }
    res.qtype = W;
    const int64_t W_bits = ts_unified_writer_qtype_bits(W);

    // No budget configured on any role -> plain worst-of (pre-16.8).
    bool any_budget = false;
    for (const auto * v : valid) {
        if (v->budget_bits >= 0) { any_budget = true; break; }
    }
    if (!any_budget) {
        res.reason = "worst-of (no role budget configured)";
        return res;
    }
    res.budget_applied = true;

    // The protector is the role whose verdict demands the most bits
    // (== the worst-of winner). Enforcing another role's budget below
    // protector_bits compromises it.
    const ts_unified_role_verdict * protector = valid[0];
    for (const auto * v : valid) {
        if (ts_unified_writer_qtype_bits(v->qtype) >
            ts_unified_writer_qtype_bits(protector->qtype)) {
            protector = v;
        }
    }
    const int64_t protector_bits = ts_unified_writer_qtype_bits(protector->qtype);

    int n_relaxed = 0;
    int64_t enforced_cap = W_bits;              // tightest enforced cap
    const ts_unified_role_verdict * enforcer = nullptr;
    for (const auto * r : valid) {
        if (r->budget_bits < 0) continue;               // unconstrained role
        if (W_bits <= r->budget_bits) continue;         // already within budget
        // Role r is over budget under the worst-of qtype. Enforcing its
        // budget would cap below protector_bits.
        if (protector->model_role != r->model_role &&
            protector_bits > r->budget_bits) {
            // Enforcement compromises the protector. Weight the decision.
            if (protector->weight >= r->weight) {
                // Relax r's constraint; keep the conservative qtype.
                res.relaxed = true;
                res.relaxed_role = r->model_role;
                res.protected_role = protector->model_role;
                n_relaxed++;
                continue;
            }
        }
        // Enforce r's budget (it outweighs the protector, or it IS the
        // protector capping itself). Keep the tightest cap seen.
        if (r->budget_bits < enforced_cap) {
            enforced_cap = r->budget_bits;
            enforcer = r;
        }
    }

    if (enforcer != nullptr && enforced_cap < W_bits) {
        // Cap the qtype at the enforced budget. Pick the largest-bits
        // verdict that still fits (as conservative as the budget allows).
        int best = GGML_TYPE_COUNT;
        int best_bits = -1;
        for (const auto * v : valid) {
            const int b = ts_unified_writer_qtype_bits(v->qtype);
            if ((int64_t)b <= enforced_cap && b > best_bits) {
                best = v->qtype;
                best_bits = b;
            }
        }
        if (best == GGML_TYPE_COUNT) {
            // No verdict fits the cap; fall back to the smallest verdict
            // (best-effort honor of the budget).
            best = valid[0]->qtype;
            for (const auto * v : valid) {
                if (ts_unified_writer_qtype_bits(v->qtype) <
                    ts_unified_writer_qtype_bits(best)) {
                    best = v->qtype;
                }
            }
        }
        res.qtype = best;
        res.enforced = true;
        res.enforced_role = enforcer->model_role;
    }

    // Evidence line summarizing the effective action.
    std::string s = "worst_of=" + std::to_string(W_bits) + "bit";
    if (res.relaxed) {
        s += "; relaxed " + res.relaxed_role + " budget to protect " +
             res.protected_role + "(" + std::to_string(protector_bits) + "bit)";
    }
    if (res.enforced) {
        s += "; enforced " + res.enforced_role + " budget cap=" +
             std::to_string(enforced_cap) + "bit -> " +
             std::to_string(ts_unified_writer_qtype_bits(res.qtype)) + "bit";
    }
    if (!res.relaxed && !res.enforced) {
        s += "; all roles within budget";
    }
    res.reason = s;
    (void)n_relaxed;
    return res;
}

int ts_unified_policy_load_json(const std::string & path,
                                 ts_unified_policy * out,
                                 std::string * err) {
    if (out == nullptr) return 1;
    out->entries.clear();
    std::ifstream f(path);
    if (!f) {
        if (err) *err = "policy_load_json: cannot open " + path;
        return 1;
    }
    json j;
    try {
        f >> j;
    } catch (const std::exception & e) {
        if (err) *err = std::string("policy_load_json: parse error: ") + e.what();
        return 1;
    }
    // The unified policy from unified_calibrate.py is a
    // llama.speculative.calibration-policy.v1 document with:
    //   "tensor_families": [{ "model_role": "...", "name": "...",
    //                         "dtype": "...", "family": "...", ... }, ...]
    // Older test fixtures use a bare list. Accept both.
    json families = j.contains("tensor_families") ? j["tensor_families"] : j;
    if (!families.is_array()) {
        if (err) *err = "policy_load_json: expected array, got " + std::string(families.type_name());
        return 1;
    }
    for (const auto & e : families) {
        ts_unified_policy_entry p;
        if (e.contains("model_role")) p.model_role = e["model_role"].get<std::string>();
        if (e.contains("name"))       p.name       = e["name"].get<std::string>();
        if (e.contains("dtype"))      p.dtype      = e["dtype"].get<std::string>();
        if (p.model_role.empty() || p.name.empty()) continue;  // skip malformed
        out->entries.push_back(std::move(p));
    }
    // Phase 16.8: optional per-role budgets (additive; absent = the
    // writer applies plain worst-of everywhere).
    //   "role_budgets": [{ "model_role": "...", "budget_bits": N,
    //                      "weight": W }, ...]
    if (j.contains("role_budgets") && j["role_budgets"].is_array()) {
        for (const auto & e : j["role_budgets"]) {
            ts_unified_role_budget rb;
            if (e.contains("model_role")) rb.model_role = e["model_role"].get<std::string>();
            if (e.contains("budget_bits")) rb.budget_bits = e["budget_bits"].get<int64_t>();
            if (e.contains("weight"))      rb.weight      = e["weight"].get<double>();
            if (rb.model_role.empty()) continue;  // skip malformed
            out->role_budgets.push_back(std::move(rb));
        }
    }
    return 0;
}

int ts_unified_policy_save_json(const std::string & path,
                                 const ts_unified_policy & policy,
                                 std::string * err) {
    json j;
    j["schema"] = "llama.speculative.calibration-policy.v1";
    j["model_role"] = nullptr;   // multi-component
    j["tensor_families"] = json::array();
    for (const auto & e : policy.entries) {
        j["tensor_families"].push_back({
            {"model_role", e.model_role},
            {"name",       e.name},
            {"dtype",      e.dtype},
        });
    }
    // Phase 16.8: emit role_budgets when present (additive; readers
    // without 16.8 support ignore the unknown key).
    if (!policy.role_budgets.empty()) {
        j["role_budgets"] = json::array();
        for (const auto & rb : policy.role_budgets) {
            j["role_budgets"].push_back({
                {"model_role",  rb.model_role},
                {"budget_bits", rb.budget_bits},
                {"weight",      rb.weight},
            });
        }
    }
    std::ofstream f(path);
    if (!f) {
        if (err) *err = "policy_save_json: cannot open " + path;
        return 1;
    }
    f << j.dump(2) << "\n";
    if (!f.good()) {
        if (err) *err = "policy_save_json: write failed";
        return 1;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// tile640 cluster copy
//
// A source tensor of type GGML_TYPE_TESSERA_T640 is one of a 6- (or
// 7- with act_scale) sub-tensor cluster. The writer treats the
// <base>.weight tensor as the cluster entry point and copies the
// 6 sub-tensors by data pointer. The destination's <base>.weight
// tensor is a TESSERA_T640 placeholder (its data is unused by the
// loader; the cluster sub-tensors carry the actual data).
// ---------------------------------------------------------------------------

namespace {

struct cluster_member_def {
    const char * suffix;
    ggml_type    type;
    int          n_dims;
    int64_t      ne[GGML_MAX_DIMS];
};

bool copy_tile640_cluster(
    source_gguf & src,
    ggml_context * src_meta_ctx,   // ggml context for src tensor metadata
    gguf_context * dst,
    ggml_context * dst_gctx,
    const std::string & src_base,
    const std::string & dst_base,
    int64_t * out_bytes,
    int * out_copied,
    std::string * err) {
    // Probe the base tensor for shape.
    int64_t base_id = gguf_find_tensor(src.ctx, src_base.c_str());
    if (base_id < 0) {
        if (err) *err = "tile640 base not found: " + src_base;
        return false;
    }
    ggml_tensor * base_meta = read_source_tensor(src, src_meta_ctx, src_base);
    if (base_meta == nullptr) {
        if (err) *err = "tile640 base metadata read failed: " + src_base;
        return false;
    }
    const int64_t in_dim  = base_meta->ne[0];
    const int64_t out_dim = base_meta->ne[1];
    const int64_t pages   = (in_dim + TS_PAGE_SIZE - 1) / TS_PAGE_SIZE;

    // Detect optional act_scale.
    const std::string act_name = src_base + ".weight_act_scale";
    const bool has_act_scale = (gguf_find_tensor(src.ctx, act_name.c_str()) >= 0);

    // Cluster sub-tensor defs (n_dims / ne for the destination; the
    // outlier_cols / outlier_vals read the source's ne[0] for the
    // real size).
    cluster_member_def members[] = {
        { "weight_packed",              GGML_TYPE_I32, 2, { pages * TS_WORDS_PER_PAGE, out_dim, 1, 1 } },
        { "weight_page_scales",         GGML_TYPE_F16, 2, { pages,                    out_dim, 1, 1 } },
        { "weight_lane_scales",         GGML_TYPE_I8,  2, { pages * TS_WORDS_PER_PAGE, out_dim, 1, 1 } },
        { "weight_outlier_row_offsets", GGML_TYPE_I32, 1, { out_dim + 1,                1, 1, 1 } },
        { "weight_outlier_cols",        GGML_TYPE_I32, 1, { 0,                          0, 0, 0 } }, // size from source
        { "weight_outlier_vals",        GGML_TYPE_F16, 1, { 0,                          0, 0, 0 } }, // size from source
    };
    if (has_act_scale) {
        cluster_member_def act = {
            "weight_act_scale", GGML_TYPE_F16, 1, { in_dim, 1, 1, 1 }
        };
        members[6] = act;
    }
    const size_t n_members = has_act_scale ? 7 : 6;

    for (size_t i = 0; i < n_members; i++) {
        const std::string src_name = src_base + "." + members[i].suffix;
        const std::string dst_name = dst_base + "." + members[i].suffix;
        ggml_tensor * s = read_source_tensor(src, src_meta_ctx, src_name);
        if (s == nullptr) {
            if (err) *err = "tile640 member missing in source: " + src_name;
            return false;
        }
        // Variable-size members use the source's ne[0].
        ggml_tensor * d = nullptr;
        if (std::strcmp(members[i].suffix, "weight_outlier_cols") == 0 ||
            std::strcmp(members[i].suffix, "weight_outlier_vals") == 0) {
            int64_t n = s->ne[0] > 0 ? s->ne[0] : 1;
            d = ggml_new_tensor_1d(dst_gctx, members[i].type, n);
        } else {
            // Use the source's ne[] verbatim (the writer's ne[] in
            // members[] is an upper bound for the static-sized
            // members; the source's ne[] is authoritative).
            int n_dims = members[i].n_dims;
            int64_t ne0 = s->ne[0];
            int64_t ne1 = s->ne[1];
            int64_t ne2 = s->ne[2];
            int64_t ne3 = s->ne[3];
            if (n_dims == 1)      d = ggml_new_tensor_1d(dst_gctx, members[i].type, ne0);
            else if (n_dims == 2) d = ggml_new_tensor_2d(dst_gctx, members[i].type, ne0, ne1);
            else if (n_dims == 3) d = ggml_new_tensor_3d(dst_gctx, members[i].type, ne0, ne1, ne2);
            else                  d = ggml_new_tensor_4d(dst_gctx, members[i].type, ne0, ne1, ne2, ne3);
        }
        if (d == nullptr) {
            if (err) *err = "ggml_new_tensor failed for " + dst_name;
            return false;
        }
        ggml_format_name(d, "%s", dst_name.c_str());
        // Repoint the destination tensor's data at the source's
        // data buffer. The source's data stays valid until close_source.
        d->data = s->data;
        gguf_add_tensor(dst, d);
        *out_bytes += (int64_t)ggml_nbytes(d);
    }
    // The base tensor: TESSERA_T640 placeholder, data points to a
    // static zero. The loader ignores the base tensor's data; the
    // cluster sub-tensors carry the actual data.
    ggml_tensor * base_dst = ggml_new_tensor_2d(dst_gctx, GGML_TYPE_TESSERA_T640, in_dim, out_dim);
    if (base_dst == nullptr) {
        if (err) *err = "ggml_new_tensor failed for base " + dst_base;
        return false;
    }
    ggml_format_name(base_dst, "%s", dst_base.c_str());
    static const int32_t t640_base_pad = 0;
    base_dst->data = (void *)&t640_base_pad;
    gguf_add_tensor(dst, base_dst);
    (*out_copied) += (int)n_members + 1;
    return true;
}

} // namespace

// ---------------------------------------------------------------------------
// ts_unified_writer constructor
// ---------------------------------------------------------------------------

ts_unified_writer::ts_unified_writer(const std::string & dst_path,
                                     const std::vector<ts_unified_component> & components,
                                     const ts_unified_policy & policy,
                                     const ts_unified_hparams & hparams,
                                     const ts_unified_dflash_hparams & dflash_hparams,
                                     const ts_unified_dspark_hparams & dspark_hparams,
                                     const ts_unified_mmproj_hparams & mmproj_hparams,
                                     const ts_unified_meta & meta,
                                     std::string * err)
    : p_(new impl(dst_path, components, policy, hparams, dflash_hparams, dspark_hparams, mmproj_hparams, meta)) {
    if (hparams.n_layer == 0) {
        if (err) *err = "ts_unified_writer: hparams.n_layer must be > 0";
        return;
    }
    if (components.empty()) {
        if (err) *err = "ts_unified_writer: no components supplied";
        return;
    }
    // The MTP constraint: the gemma4-assistant arch requires
    // n_layer_nextn == n_layer (gemma4-assistant.cpp:53). The
    // writer emits the same n_layer as both the trunk's block
    // count and the MTP's nextn_predict_layers. The
    // ts_unified_hparams struct deliberately does not carry
    // n_layer_nextn as a separate field; the equality is the
    // single source of truth.

    // Open each source GGUF.
    for (size_t i = 0; i < components.size(); i++) {
        ggml_context * gctx = nullptr;
        gguf_init_params ip = { /*no_alloc=*/ false, /*ctx=*/ &gctx };
        p_->sources[i].ctx = gguf_init_from_file(components[i].path.c_str(), ip);
        if (p_->sources[i].ctx == nullptr) {
            if (err) *err = "component " + std::to_string(i) + " (" + components[i].path + "): gguf_init_from_file failed";
            return;
        }
        if (gctx == nullptr) {
            if (err) *err = "component " + std::to_string(i) + " (" + components[i].path + "): gguf_init_from_file returned null ggml ctx";
            return;
        }
        // The ggml_context is owned by the gguf (gguf_free frees
        // it). We hold a pointer so we can call ggml_get_tensor
        // for data-pointer lookups.
        p_->sources[i].tensor_ctx = gctx;
    }

    // Open the destination gguf.
    p_->dst = gguf_init_empty();
    if (p_->dst == nullptr) {
        if (err) *err = "gguf_init_empty failed";
        return;
    }
    // ggml_init gives us a scratch buffer for the destination
    // ggml_tensors. 64 MB is generous for the small synthetic test
    // inputs and a real 12B model's destination; the writer uses
    // ~280 tensor descriptors (1 per cluster member + 1 per
    // non-cluster tensor) which is well under 1 MB.
    const size_t gctx_size = 64 * 1024 * 1024;
    p_->dst_gctx_mem = gctx_size;
    ggml_init_params ip = { /*mem_size=*/ gctx_size, /*mem_buffer=*/ nullptr, /*no_alloc=*/ true };
    p_->dst_gctx = ggml_init(ip);
    if (p_->dst_gctx == nullptr) {
        if (err) *err = "ggml_init failed for destination ggml ctx";
        return;
    }

    // Set arch + hparams.
    gguf_set_val_str(p_->dst, "general.architecture", "gemma4-assistant");
    gguf_set_val_u32(p_->dst, "gemma4-assistant.block_count",          p_->hparams.n_layer);
    gguf_set_val_u32(p_->dst, "gemma4-assistant.embedding_length",      p_->hparams.n_embd);
    gguf_set_val_u32(p_->dst, "gemma4-assistant.feed_forward_length",  p_->hparams.n_ff);
    gguf_set_val_u32(p_->dst, "gemma4-assistant.attention.head_count",          p_->hparams.n_head);
    gguf_set_val_u32(p_->dst, "gemma4-assistant.attention.head_count_kv",      p_->hparams.n_head_kv);
    gguf_set_val_u32(p_->dst, "gemma4-assistant.attention.key_length",          p_->hparams.n_embd_head_k);
    gguf_set_val_u32(p_->dst, "gemma4-assistant.attention.value_length",        p_->hparams.n_embd_head_v);
    gguf_set_val_u32(p_->dst, "gemma4-assistant.attention.key_length_swa",      p_->hparams.n_embd_head_k_swa);
    gguf_set_val_u32(p_->dst, "gemma4-assistant.attention.value_length_swa",    p_->hparams.n_embd_head_v_swa);
    gguf_set_val_u32(p_->dst, "gemma4-assistant.attention.sliding_window",      p_->hparams.n_swa);
    if (!p_->hparams.is_swa_impl.empty() && p_->hparams.is_swa_impl.size() == p_->hparams.n_layer) {
        gguf_set_arr_data(p_->dst, "gemma4-assistant.attention.sliding_window_pattern",
                          GGUF_TYPE_UINT8, p_->hparams.is_swa_impl.data(), p_->hparams.is_swa_impl.size());
    }
    if (p_->hparams.n_kv_shared_layers > 0) {
        gguf_set_val_u32(p_->dst, "gemma4-assistant.attention.shared_kv_layers",
                         p_->hparams.n_kv_shared_layers);
    }
    if (p_->hparams.rope_freq_base_train_swa > 0.0f) {
        gguf_set_val_f32(p_->dst, "gemma4-assistant.rope.freq_base_swa",
                         p_->hparams.rope_freq_base_train_swa);
    }
    gguf_set_val_f32(p_->dst, "gemma4-assistant.attention.layer_norm_rms_epsilon",
                     p_->hparams.f_norm_rms_eps);
    // n_layer_nextn: the gemma4-assistant arch's current constraint
    // is n_layer_nextn == n_layer (gemma4-assistant.cpp:53). The
    // writer emits the value from the supplied hparams (which the
    // caller may override); the loader asserts equality.
    gguf_set_val_u32(p_->dst, "gemma4-assistant.nextn_predict_layers", p_->hparams.n_layer);
    if (p_->hparams.n_vocab > 0) {
        gguf_set_val_u32(p_->dst, "gemma4-assistant.vocab_size", p_->hparams.n_vocab);
    }
    if (p_->hparams.n_embd_out > 0) {
        gguf_set_val_u32(p_->dst, "gemma4-assistant.embedding_length_out", p_->hparams.n_embd_out);
    }
    // Phase M0a: multimodal-projector hparams. All values are
    // additive -- the destination's loader treats absence of these
    // keys as "no mmproj in this GGUF" and falls back to the
    // non-multimodal gemma4-assistant build. The keys follow the
    // gemma4 mtmd companion's naming convention; the values come
    // from the CLI's --mmproj-hparams JSON (or a default-constructed
    // ts_unified_mmproj_hparams when no mmproj is in this run).
    if (p_->mmproj_hparams.vision_n_embd > 0) {
        gguf_set_val_u32(p_->dst, "gemma4-assistant.vision.embedding_length",
                         (uint32_t)p_->mmproj_hparams.vision_n_embd);
    }
    if (!p_->mmproj_hparams.vision_arch.empty()) {
        gguf_set_val_str(p_->dst, "gemma4-assistant.vision.architecture",
                         p_->mmproj_hparams.vision_arch.c_str());
    }
    if (p_->mmproj_hparams.audio_n_embd > 0) {
        gguf_set_val_u32(p_->dst, "gemma4-assistant.audio.embedding_length",
                         (uint32_t)p_->mmproj_hparams.audio_n_embd);
    }
    if (!p_->mmproj_hparams.audio_arch.empty()) {
        gguf_set_val_str(p_->dst, "gemma4-assistant.audio.architecture",
                         p_->mmproj_hparams.audio_arch.c_str());
    }
    if (p_->mmproj_hparams.projector_dim > 0) {
        gguf_set_val_u32(p_->dst, "gemma4-assistant.mm.projector_dim",
                         (uint32_t)p_->mmproj_hparams.projector_dim);
    }
    // Tessera provenance
    if (!p_->meta.build_info.empty()) {
        gguf_set_val_str(p_->dst, "tessera.provenance.build_info", p_->meta.build_info.c_str());
    }
    if (!p_->meta.main_tip.empty()) {
        gguf_set_val_str(p_->dst, "tessera.provenance.main_tip", p_->meta.main_tip.c_str());
    }
    gguf_set_val_u32(p_->dst, "tessera.unified.writer", 1);
}

ts_unified_writer::~ts_unified_writer() {
    delete p_;
}

// ---------------------------------------------------------------------------
// write_all: copy every tensor from every source into the destination
// ---------------------------------------------------------------------------

// --- per-component routing ---------------------------------------------------

// Returns the gemma4-assistant destination tensor name for a given
// (component, source_name) pair. The brief's spec routes the
// per-component tensors into specific gemma4-assistant slots:
//
//   * trunk:        copy as-is. The trunk's blk.{i}.attn_q.weight
//                   IS the gemma4-assistant MTP-side per-block
//                   attn_q (the MTP graph uses the same names).
//   * dflash:       copy with a "dflash." prefix to disambiguate
//                   from the trunk's identically-named tensors.
//                   The dflash drafter (loaded as ctx_other) is a
//                   separate file at runtime; the prefix is a
//                   future-proofing affordance for the case where
//                   the orchestrator wants the drafter to read
//                   from the same unified GGUF as the MTP.
//   * dspark:       copy as-is. The dspark heads (markov_w1,
//                   markov_w2, conf_proj) are unique to the dflash
//                   arch and don't conflict with anything.
//   * mtp_nextn:    copy as-is. The mtp_nextn tensors are
//                   blk.{i}.nextn.* with unique suffixes.
//   * shared_embd:  copy as-is. The shared embeddings
//                   (token_embd, output) are unique.
//   * vision_tower: copy as-is. Source tensors already carry the
//                   "v." prefix per tools/mtmd/clip.cpp:1831
//                   ("a" / "v" prefix is set on the source side
//                   based on modality). The writer does NOT add
//                   a second prefix; a "v.patch_embd.weight" in
//                   the source becomes "v.patch_embd.weight" in
//                   the destination.
//   * audio_tower:  copy as-is. Same convention as vision_tower;
//                   source tensors are already "a.*".
//   * mm_projector: copy as-is. Source tensors are already "mm.*"
//                   (multi_modal_projector weights per
//                   tools/mtmd/clip.cpp:2594+). If a "mm.*" tensor
//                   appears in BOTH vision_tower and mm_projector
//                   components (rare but possible for shared
//                   projector weights), the writer's name-based
//                   worst-of qtype lookup (qtype_for_tensor) treats
//                   it the same way token_embd.weight is treated
//                   across trunk / dflash / shared_embd -- the
//                   per-role verdicts reconcile, last-write-wins
//                   for the data pointer (the mm_projector
//                   component wins by component order; the writer
//                   does not error on the conflict, matching the
//                   pre-M0a shared_embd contract).
//
// Returns "" for tensors the writer does not route. The writer
// skips "" silently (the tensor is in the source but the unified
// arch does not need it).
static std::string route_destination_name(const std::string & component_role,
                                          const std::string & src_name) {
    if (component_role == "dflash") {
        return "dflash." + src_name;
    }
    // trunk / dspark / mtp_nextn / shared_embd / vision_tower /
    // audio_tower / mm_projector: copy as-is. The mmproj source
    // GGUFs already namespace their tensors with "v.", "a.", and
    // "mm." prefixes (see the per-component comment above), so the
    // destination names are byte-identical to the source names.
    return src_name;
}

// --- Phase 16.6 worst-of qtype resolution ------------------------------------
//
// Per-tensor qtype resolution (Phase 16.6 worst-of). The policy's
// tensor_families can have multiple entries for the same tensor
// name (e.g. one from the trunk calibration, one from the dflash
// drafter calibration, one from the shared_embd calibration --
// all targeting `token_embd.weight`).
//
// The worst-of rule: pick the more conservative qtype
// (max(bits)) across all matching entries. The drafter is more
// sensitive to the embedding than the trunk, so when in doubt,
// use more bits.
//
// For the trunk / dflash / dspark / mtp_nextn components, this is
// a no-op in practice (each per-block tensor's name is owned by
// exactly one component, so the policy has at most one matching
// entry per name). The rule only matters for shared_embd
// (token_embd / output), where the trunk's and dflash's
// verdicts can disagree.
//
// Lookup is by tensor NAME (not by (role, name)) because the
// shared tensor's qtype must reconcile across all components'
// calibration verdicts. The dflash prefix on the destination
// side keeps the destination tensors distinct even when the
// source names collide.
//
// Returns GGML_TYPE_COUNT (== 45) when no entry matches; the
// caller treats that as "no override" and copies the source's
// qtype as-is.
static int qtype_for_tensor(
    const std::string & name,
    const std::vector<ts_unified_policy_entry> & entries) {
    int result = GGML_TYPE_COUNT;
    for (const auto & e : entries) {
        if (e.name != name) continue;
        if (e.dtype.empty()) continue;
        int q = ts_unified_qtype_from_string(e.dtype);
        if (q == GGML_TYPE_COUNT) continue;   // unknown dtype, skip
        if (result == GGML_TYPE_COUNT) {
            result = q;
        } else {
            result = ts_unified_writer_worst_of(result, q);
        }
    }
    return result;
}

// --- Phase 16.8 budget-aware per-tensor resolution ---------------------------
//
// Resolves the qtype for one destination tensor. When the policy carries
// no role budgets, or the tensor has a single owning role, this is plain
// worst-of (byte-identical to the pre-16.8 path). Only when >= 2 roles
// have verdicts on the same name AND role budgets are configured does the
// budget-aware reconciliation run, appending relaxation/enforcement
// evidence to *events.
static int resolve_tensor_qtype(
    const std::string & name,
    const ts_unified_policy & policy,
    std::vector<ts_unified_budget_event> * events) {
    // Fast path: no budgets -> plain worst-of (pre-16.8 contract).
    if (policy.role_budgets.empty()) {
        return qtype_for_tensor(name, policy.entries);
    }
    // Collect per-role verdicts, folding duplicates within a role via
    // worst-of (a role may emit more than one entry for the same name).
    std::map<std::string, int> role_qtype;
    for (const auto & e : policy.entries) {
        if (e.name != name) continue;
        if (e.dtype.empty()) continue;
        int q = ts_unified_qtype_from_string(e.dtype);
        if (q == GGML_TYPE_COUNT) continue;   // unknown dtype, skip
        auto it = role_qtype.find(e.model_role);
        if (it == role_qtype.end()) {
            role_qtype[e.model_role] = q;
        } else {
            it->second = ts_unified_writer_worst_of(it->second, q);
        }
    }
    if (role_qtype.empty()) return GGML_TYPE_COUNT;
    if (role_qtype.size() < 2) {
        // Single owning role: cross-role budget propagation cannot
        // happen, so keep the plain worst-of result.
        return qtype_for_tensor(name, policy.entries);
    }
    // Build the verdict list with each role's budget + weight attached.
    std::vector<ts_unified_role_verdict> verdicts;
    for (const auto & kv : role_qtype) {
        ts_unified_role_verdict v;
        v.model_role = kv.first;
        v.qtype      = kv.second;
        for (const auto & rb : policy.role_budgets) {
            if (rb.model_role == kv.first) {
                v.budget_bits = rb.budget_bits;
                v.weight      = rb.weight;
                break;
            }
        }
        verdicts.push_back(std::move(v));
    }
    ts_unified_shared_resolution res = ts_unified_writer_resolve_shared(verdicts);
    if (events != nullptr) {
        if (res.relaxed) {
            ts_unified_budget_event ev;
            ev.tensor = name; ev.action = "relaxed";
            ev.role = res.relaxed_role; ev.other_role = res.protected_role;
            ev.qtype = res.qtype; ev.reason = res.reason;
            events->push_back(std::move(ev));
        }
        if (res.enforced) {
            ts_unified_budget_event ev;
            ev.tensor = name; ev.action = "enforced";
            ev.role = res.enforced_role;
            ev.qtype = res.qtype; ev.reason = res.reason;
            events->push_back(std::move(ev));
        }
    }
    return res.qtype;
}

int ts_unified_writer::write_all(std::string * err) {
    if (p_->dst == nullptr) {
        if (err) *err = "ts_unified_writer: not initialized";
        return 1;
    }
    // Phase 16.6: per-tensor qtype is resolved by name (not
    // (role, name)) with worst-of across all matching entries.
    // The old (role, name) map is gone; the qtype_for_tensor
    // helper does the lookup on demand for each source tensor.
    int64_t total_bytes = 0;
    int n_copied = 0;
    int n_skipped = 0;
    int n_overrides = 0;

    // Track which destination names we've already used so a name
    // collision in the source GGUFs (e.g. trunk and dflash both
    // exporting blk.0.attn_q.weight) is caught explicitly. The
    // writer's prefix logic (route_destination_name) prevents most
    // collisions, but the per-component source might still collide
    // with the destination's own meta-tensors.
    std::map<std::string, std::string> dst_seen;   // dst_name -> src path

    for (size_t ci = 0; ci < p_->components.size(); ci++) {
        const auto & comp = p_->components[ci];
        source_gguf & src = p_->sources[ci];
        const int64_t n_tensors = gguf_get_n_tensors(src.ctx);

        for (int64_t ti = 0; ti < n_tensors; ti++) {
            const char * name = gguf_get_tensor_name(src.ctx, ti);
            if (name == nullptr) continue;
            std::string src_name = name;

            // Skip cluster sub-tensors; we handle the base (which
            // has type TESSERA_T640) and copy the whole cluster as
            // a unit. The cluster sub-tensors' names end in
            // .weight_packed / .weight_page_scales / ...
            if (src_name.find(".weight_packed") != std::string::npos ||
                src_name.find(".weight_page_scales") != std::string::npos ||
                src_name.find(".weight_lane_scales") != std::string::npos ||
                src_name.find(".weight_outlier_row_offsets") != std::string::npos ||
                src_name.find(".weight_outlier_cols") != std::string::npos ||
                src_name.find(".weight_outlier_vals") != std::string::npos ||
                src_name.find(".weight_act_scale") != std::string::npos) {
                continue;
            }

            // Route the tensor name to the destination slot.
            const std::string dst_name = route_destination_name(comp.model_role, src_name);
            if (dst_name.empty()) {
                n_skipped++;
                continue;
            }
            // Duplicate detection: if a previous component already
            // claimed this destination name AND the source data
            // matches, this is a "shared" tensor (e.g. token_embd
            // provided by both trunk and shared_embd); the policy's
            // shared_embd component wins (the writer treats the
            // last write as authoritative). If the data is
            // different, this is a real conflict and we skip with
            // a clear error.
            auto prev = dst_seen.find(dst_name);
            if (prev != dst_seen.end()) {
                // Accept if the source path is the same (idempotent
                // re-run with the same component). The
                // gguf_add_tensor ABORT would catch this otherwise.
                n_skipped++;
                continue;
            }
            dst_seen[dst_name] = comp.path;

            // Get the source's type (used by both the override
            // classifier and the copy path).
            ggml_type src_type = gguf_get_tensor_type(src.ctx, ti);

            int override_qtype = resolve_tensor_qtype(src_name, p_->policy,
                                                       &p_->budget_events);
            if (override_qtype != GGML_TYPE_COUNT) {
                // Only count as an override when the policy's
                // resolved qtype actually differs from the
                // source's type. A policy entry that matches
                // the source's qtype is a no-op and should not
                // bump the override counter (the brief's spec is
                // "per-tensor qtype override", which implies a
                // change). With the worst-of rule, the resolved
                // qtype is the more conservative partner; the
                // override applies when the conservative
                // partner differs from the source's qtype.
                if (override_qtype != (int)src_type) {
                    n_overrides++;
                }
            }
            if (src_type == GGML_TYPE_TESSERA_T640) {
                // tile640 cluster copy. We need a ggml context to
                // allocate transient metadata tensors for the
                // sub-tensors; we use a static thread-local
                // scratch context (4 MB) so each call reuses the
                // same buffer.
                static thread_local ggml_context * cluster_meta_ctx = nullptr;
                static thread_local size_t         cluster_meta_size = 0;
                if (cluster_meta_ctx == nullptr) {
                    cluster_meta_size = 4 * 1024 * 1024;   // 4 MB scratch
                    ggml_init_params ip = { cluster_meta_size, nullptr, true };
                    cluster_meta_ctx = ggml_init(ip);
                }
                if (cluster_meta_ctx == nullptr) {
                    if (err) *err = "ggml_init failed for cluster metadata ctx";
                    return 1;
                }
                int copied_cluster = 0;
                if (!copy_tile640_cluster(src, cluster_meta_ctx,
                                          p_->dst, p_->dst_gctx,
                                          src_name, dst_name,
                                          &total_bytes, &copied_cluster, err)) {
                    if (err) *err = "cluster " + src_name + " (" + comp.model_role + "): " + *err;
                    return 1;
                }
                n_copied += copied_cluster;
            } else {
                // Standard qtype copy.
                ggml_tensor * s = src.find(src_name);
                if (s == nullptr || s->data == nullptr) {
                    if (err) *err = "source tensor not loaded: " + src_name;
                    return 1;
                }
                int64_t sid = gguf_find_tensor(src.ctx, src_name.c_str());
                const int64_t * sne = gguf_get_tensor_ne(src.ctx, sid);
                int n_dims = (sne[3] > 1) ? 4 : (sne[2] > 1) ? 3 : (sne[1] > 1) ? 2 : 1;
                ggml_tensor * d = nullptr;
                if (n_dims == 1)      d = ggml_new_tensor_1d(p_->dst_gctx, src_type, sne[0]);
                else if (n_dims == 2) d = ggml_new_tensor_2d(p_->dst_gctx, src_type, sne[0], sne[1]);
                else if (n_dims == 3) d = ggml_new_tensor_3d(p_->dst_gctx, src_type, sne[0], sne[1], sne[2]);
                else                  d = ggml_new_tensor_4d(p_->dst_gctx, src_type, sne[0], sne[1], sne[2], sne[3]);
                if (d == nullptr) {
                    if (err) *err = "ggml_new_tensor failed for " + dst_name;
                    return 1;
                }
                ggml_format_name(d, "%s", dst_name.c_str());
                d->data = s->data;
                if (override_qtype != GGML_TYPE_COUNT &&
                    override_qtype != (int)src_type) {
                    // Phase 16.6: the resolved qtype is the
                    // worst-of across all matching entries. The
                    // new tensor's ndims must match the source
                    // (qtype_blocks and qtype_size can vary
                    // between qtypes, so we allocate with the
                    // same dimensionality as the source).
                    ggml_tensor * od = nullptr;
                    if (n_dims == 1)      od = ggml_new_tensor_1d(p_->dst_gctx, (ggml_type)override_qtype, sne[0]);
                    else if (n_dims == 2) od = ggml_new_tensor_2d(p_->dst_gctx, (ggml_type)override_qtype, sne[0], sne[1]);
                    else if (n_dims == 3) od = ggml_new_tensor_3d(p_->dst_gctx, (ggml_type)override_qtype, sne[0], sne[1], sne[2]);
                    else                  od = ggml_new_tensor_4d(p_->dst_gctx, (ggml_type)override_qtype, sne[0], sne[1], sne[2], sne[3]);
                    if (od == nullptr) {
                        if (err) *err = "ggml_new_tensor (override) failed for " + dst_name;
                        return 1;
                    }
                    ggml_format_name(od, "%s", dst_name.c_str());
                    od->data = s->data;
                    d = od;
                }
                gguf_add_tensor(p_->dst, d);
                total_bytes += (int64_t)ggml_nbytes(d);
                n_copied++;
            }

            if      (comp.model_role == "trunk")        stats_.n_tensors_trunk++;
            else if (comp.model_role == "dflash")       stats_.n_tensors_dflash++;
            else if (comp.model_role == "dspark")       stats_.n_tensors_dspark++;
            else if (comp.model_role == "mtp_nextn")    stats_.n_tensors_mtp_nextn++;
            else if (comp.model_role == "shared_embd")  stats_.n_tensors_shared_embd++;
            else if (comp.model_role == "vision_tower") stats_.n_tensors_vision_tower++;
            else if (comp.model_role == "audio_tower")  stats_.n_tensors_audio_tower++;
            else if (comp.model_role == "mm_projector") stats_.n_tensors_mm_projector++;
        }
    }
    stats_.n_tensors_skipped = n_skipped;
    stats_.n_qtype_overrides = n_overrides;
    stats_.total_bytes       = total_bytes;
    // Phase 16.8: budget relaxation stats + audit trail. The events are
    // also serialized into the destination GGUF metadata so the
    // relaxation evidence travels with the artifact.
    for (const auto & ev : p_->budget_events) {
        if      (ev.action == "relaxed")  stats_.n_budget_relaxed++;
        else if (ev.action == "enforced") stats_.n_budget_enforced++;
    }
    if (!p_->budget_events.empty()) {
        std::string ev_json = "[";
        for (size_t i = 0; i < p_->budget_events.size(); i++) {
            const auto & ev = p_->budget_events[i];
            if (i != 0) ev_json += ",";
            ev_json += "{\"tensor\":\"" + ev.tensor +
                       "\",\"action\":\"" + ev.action +
                       "\",\"role\":\"" + ev.role + "\"";
            if (!ev.other_role.empty()) {
                ev_json += ",\"other_role\":\"" + ev.other_role + "\"";
            }
            ev_json += ",\"qtype\":\"" + ts_unified_qtype_to_string(ev.qtype) +
                       "\",\"reason\":\"" + ev.reason + "\"}";
        }
        ev_json += "]";
        gguf_set_val_str(p_->dst, "tessera.unified.budget_events", ev_json.c_str());
    }
    // Crash-safe publish: write to .tmp, then atomic rename. The unified
    // GGUF can be ~3.7 GB and take minutes; a SIGKILL or jetsam mid-write
    // without this guard would leave a truncated file at dst_path, and
    // the next quantize pass would either load garbage or fail with a
    // confusing "invalid GGUF" instead of a clean "previous run
    // interrupted" recovery. POSIX rename is atomic on the same
    // filesystem; mirrors the imatrix save pattern in
    // tools/imatrix/imatrix.cpp:save_imatrix.
    const std::string dst_tmp = p_->dst_path + ".tmp";
    if (!gguf_write_to_file(p_->dst, dst_tmp.c_str(), /*only_meta=*/false)) {
        std::remove(dst_tmp.c_str());
        if (err) *err = "gguf_write_to_file failed: " + dst_tmp;
        return 1;
    }
    if (std::rename(dst_tmp.c_str(), p_->dst_path.c_str()) != 0) {
        std::remove(dst_tmp.c_str());
        if (err) *err = "unified GGUF atomic rename failed: " + p_->dst_path;
        return 1;
    }
    return 0;
}

const std::vector<ts_unified_budget_event> & ts_unified_writer::get_budget_events() const {
    return p_->budget_events;
}
