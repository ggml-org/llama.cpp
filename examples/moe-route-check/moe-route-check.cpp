// Compare the MoE routing (sigmoid, bias, top-k, norm, scale) computed by the model's backend with the CPU backend
// on the same router logits, layer by layer, for one prompt.
//
//   llama-moe-route-check -m model.gguf -f prompt.txt -ngl 99 [-ub 1]
//
// Full MoE layer check: MOE_LAYER=<il,...> MOE_LAYER_DUMP=<file> saves the input (attn_norm) and output (l_out) of
// those layers; a second run on CPU (-dev none) with MOE_LAYER_REF=<file> feeds the saved input to the same layers
// and compares their output with the saved one. MOE_LAYER_PROBE=1 also saves and compares the layer intermediates.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstring>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

// same clamp as build_moe_ffn
static const float WEIGHTS_SUM_MIN = 6.103515625e-5f;
// same as the MUL_MAT tolerance of test-backend-ops
static const double LAYER_NMSE_MAX = 5e-4;
// intermediates of the LatentMoE layer; the latent up output is l_out - ffn_shexp
static const char * const LAYER_PROBES[] = {
    "ffn_moe_up", "ffn_moe_relu_sqr", "ffn_moe_down", "ffn_moe_weighted", "ffn_moe_out", "ffn_up", "ffn_sqr(relu)", "ffn_shexp",
};
static const char * const LATENT_UP_PROBE = "latent_up";

struct layer_stats {
    std::string buft;
    int64_t n_exp       = 0;
    int64_t n_used      = 0;
    int64_t n_tokens    = 0;
    int64_t set_diff    = 0;  // tokens whose selected expert set differs
    int64_t order_diff  = 0;  // tokens with the same set in a different order
    int64_t ties        = 0;  // tokens whose set differs only by experts tied at the k-th score
    float   min_margin  = INFINITY;  // smallest gap between the k-th and (k+1)-th selection score
    float   diff_margin = INFINITY;  // smallest gap among tokens with a different set
    float   max_w_diff  = 0.0f;
};

struct probe_stats {
    double  err = 0.0, nrm_ref = 0.0, nrm_x = 0.0;
    int64_t n = 0, zero_ref = 0, zero_x = 0, n_ub = 0, collapsed = 0;
    float   amax_ref = 0.0f;
};

struct route_check {
    ggml_backend_t cpu = nullptr;
    std::vector<uint8_t> buf;
    // per layer, captured from the backend before the weights arrive
    std::map<int, std::vector<float>>   logits;
    std::map<int, std::vector<float>>   bias;
    std::map<int, std::vector<int32_t>> argsort;
    std::map<int, layer_stats>          stats;
    // full layer check, per layer and ubatch
    std::set<int> layers;
    bool          ref = false;
    std::map<int, std::vector<std::vector<float>>> inp, out;
    std::map<int, size_t> n_inp, n_out;
    double nmse_max = 0.0;
    // intermediates by full tensor name, per ubatch
    bool probe = false;
    std::map<std::string, std::vector<std::vector<float>>> probes;
    std::map<std::string, size_t> n_probe;
    std::map<std::string, std::vector<float>> last_cpu;
    std::map<std::string, std::vector<std::vector<float>>> cpu_probes;
    std::map<std::string, probe_stats> pstats;
};


static void probe_add(probe_stats & s, const std::vector<float> & r, const std::vector<float> & x) {
    double nx = 0.0;
    for (size_t i = 0; i < r.size(); i++) {
        const double d = (double) x[i] - r[i];
        s.err      += d * d;
        s.nrm_ref  += (double) r[i] * r[i];
        nx         += (double) x[i] * x[i];
        s.zero_ref += r[i] == 0.0f;
        s.zero_x   += x[i] == 0.0f;
        s.amax_ref  = std::max(s.amax_ref, std::fabs(r[i]));
    }
    s.nrm_x += nx;
    s.n     += (int64_t) r.size();
    s.n_ub++;
    s.collapsed += nx == 0.0;
}

struct layer_out_stats {
    double nmse = 0.0, max_abs = 0.0, worst_tok_nmse = 0.0, worst_rms_ref = 0.0, worst_rms_x = 0.0;
    int64_t worst_tok = -1;
};

// reference r, test x, ne0 values per token
static layer_out_stats compare_out(const std::vector<float> & r, const std::vector<float> & x, int64_t ne0) {
    layer_out_stats s;
    double err = 0.0, nrm = 0.0;
    for (size_t t = 0; t < r.size() / ne0; t++) {
        double te = 0.0, tn = 0.0, tx = 0.0;
        for (int64_t i = 0; i < ne0; i++) {
            const double d = (double) x[t * ne0 + i] - r[t * ne0 + i];
            te += d * d;
            tn += (double) r[t * ne0 + i] * r[t * ne0 + i];
            tx += (double) x[t * ne0 + i] * x[t * ne0 + i];
            s.max_abs = std::max(s.max_abs, std::fabs(d));
        }
        err += te;
        nrm += tn;
        const double tok_nmse = te / tn;
        if (!(tok_nmse <= s.worst_tok_nmse)) {
            s.worst_tok_nmse = tok_nmse;
            s.worst_tok      = (int64_t) t;
            s.worst_rms_ref  = std::sqrt(tn / ne0);
            s.worst_rms_x    = std::sqrt(tx / ne0);
        }
    }
    s.nmse = err / nrm;
    return s;
}

static void write_vec(FILE * f, const std::vector<float> & v) {
    const uint64_t n = v.size();
    fwrite(&n, sizeof(n), 1, f);
    fwrite(v.data(), sizeof(float), n, f);
}

static bool read_vec(FILE * f, std::vector<float> & v) {
    uint64_t n;
    if (fread(&n, sizeof(n), 1, f) != 1) {
        return false;
    }
    v.resize(n);
    return fread(v.data(), sizeof(float), n, f) == n;
}

static bool layers_save(const route_check & rc, const char * path) {
    FILE * f = fopen(path, "wb");
    if (!f) {
        return false;
    }
    for (const auto & [il, outs] : rc.out) {
        for (size_t k = 0; k < outs.size(); k++) {
            const int32_t hdr[2] = { il, (int32_t) k };
            fwrite(hdr, sizeof(hdr), 1, f);
            write_vec(f, rc.inp.at(il)[k]);
            write_vec(f, outs[k]);
        }
    }
    fclose(f);
    return true;
}

static bool layers_load(route_check & rc, const char * path) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        return false;
    }
    int32_t hdr[2];
    while (fread(hdr, sizeof(hdr), 1, f) == 1) {
        std::vector<float> in, out;
        if (!read_vec(f, in) || !read_vec(f, out)) {
            fclose(f);
            return false;
        }
        rc.inp[hdr[0]].push_back(std::move(in));
        rc.out[hdr[0]].push_back(std::move(out));
    }
    fclose(f);
    return true;
}

static std::string probe_path(const char * path) {
    return std::string(path) + ".probe";
}

static bool probes_save(const std::map<std::string, std::vector<std::vector<float>>> & probes, const char * path) {
    FILE * f = fopen(probe_path(path).c_str(), "wb");
    if (!f) {
        return false;
    }
    for (const auto & [name, ubs] : probes) {
        for (const auto & v : ubs) {
            const uint32_t len = (uint32_t) name.size();
            fwrite(&len, sizeof(len), 1, f);
            fwrite(name.data(), 1, len, f);
            write_vec(f, v);
        }
    }
    fclose(f);
    return true;
}

static bool probes_load(route_check & rc, const char * path) {
    FILE * f = fopen(probe_path(path).c_str(), "rb");
    if (!f) {
        return false;
    }
    uint32_t len;
    while (fread(&len, sizeof(len), 1, f) == 1) {
        std::string name(len, '\0');
        std::vector<float> v;
        if (fread(name.data(), 1, len, f) != len || !read_vec(f, v)) {
            fclose(f);
            return false;
        }
        rc.probes[name].push_back(std::move(v));
    }
    fclose(f);
    return true;
}

static bool split_name(const char * name, std::string & base, int & il) {
    const char * dash = strrchr(name, '-');
    if (!dash) {
        return false;
    }
    base = std::string(name, dash - name);
    il   = atoi(dash + 1);
    return true;
}

template <typename T>
static std::vector<T> get_data(const ggml_tensor * t) {
    GGML_ASSERT(ggml_is_contiguous(t));
    std::vector<T> v(ggml_nelements(t));
    ggml_backend_tensor_get(t, v.data(), 0, ggml_nbytes(t));
    return v;
}

// the routing chain of build_moe_ffn for sigmoid gating with bias, norm and scale, on the CPU backend
static void cpu_route(route_check & rc, const std::vector<float> & logits, const std::vector<float> & bias,
                      int64_t n_exp, int64_t n_tok, int64_t k, float scale,
                      std::vector<int32_t> & ids, std::vector<float> & w, std::vector<float> & sel) {
    ggml_init_params ip = { ggml_tensor_overhead() * 32 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * lg    = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_exp, n_tok);
    ggml_tensor * b     = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_exp);
    ggml_tensor * probs = ggml_sigmoid(ctx, lg);
    ggml_tensor * selp  = ggml_add(ctx, probs, b);
    ggml_tensor * top   = ggml_argsort_top_k(ctx, selp, k);
    ggml_tensor * wt    = ggml_get_rows(ctx, ggml_reshape_3d(ctx, probs, 1, n_exp, n_tok), top);
    wt                  = ggml_reshape_2d(ctx, wt, k, n_tok);
    ggml_tensor * sum   = ggml_clamp(ctx, ggml_sum_rows(ctx, wt), WEIGHTS_SUM_MIN, INFINITY);
    wt                  = ggml_scale(ctx, ggml_div(ctx, wt, sum), scale);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, selp);
    ggml_build_forward_expand(gf, wt);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, rc.cpu);
    ggml_backend_tensor_set(lg, logits.data(), 0, ggml_nbytes(lg));
    ggml_backend_tensor_set(b, bias.data(), 0, ggml_nbytes(b));
    GGML_ASSERT(ggml_backend_graph_compute(rc.cpu, gf) == GGML_STATUS_SUCCESS);

    ids = get_data<int32_t>(top->src[0]);  // full argsort, the top-k view is its first k columns
    w   = get_data<float>(wt);
    sel = get_data<float>(selp);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

static void compare(route_check & rc, int il, const ggml_tensor * t_w) {
    const std::vector<float> & lg = rc.logits.at(il);
    const std::vector<int32_t> & as = rc.argsort.at(il);
    const int64_t n_used = t_w->ne[1];
    const int64_t n_tok  = t_w->ne[2];
    const int64_t n_exp  = (int64_t) lg.size() / n_tok;
    float scale;
    memcpy(&scale, t_w->op_params, sizeof(scale));  // GGML_OP_SCALE: s, b

    std::vector<float> w_dev = get_data<float>(t_w);
    std::vector<int32_t> ids;
    std::vector<float> w_cpu, sel;
    cpu_route(rc, lg, rc.bias.at(il), n_exp, n_tok, n_used, scale, ids, w_cpu, sel);

    layer_stats & s = rc.stats[il];
    s.buft     = t_w->buffer ? ggml_backend_buft_name(ggml_backend_buffer_get_type(t_w->buffer)) : "?";
    s.n_exp    = n_exp;
    s.n_used   = n_used;
    s.n_tokens += n_tok;

    for (int64_t t = 0; t < n_tok; t++) {
        const int32_t * d = &as[t * n_exp];
        const int32_t * c = &ids[t * n_exp];
        const float   * p = &sel[t * n_exp];
        const float margin = n_used < n_exp ? p[c[n_used - 1]] - p[c[n_used]] : INFINITY;
        s.min_margin = std::min(s.min_margin, margin);

        std::vector<int32_t> ds(d, d + n_used), cs(c, c + n_used);
        if (!std::equal(ds.begin(), ds.end(), cs.begin())) {
            std::sort(ds.begin(), ds.end());
            std::sort(cs.begin(), cs.end());
            // the device set is also a valid top-k when it differs only by experts tied with the k-th score
            const bool tie = std::all_of(ds.begin(), ds.end(), [&](int32_t e) { return p[e] >= p[c[n_used - 1]]; });
            if (ds != cs && tie) {
                s.ties++;
                continue;
            }
            if (ds != cs) {
                if (s.set_diff == 0) {
                    const auto [lmin, lmax] = std::minmax_element(lg.begin() + t * n_exp, lg.begin() + (t + 1) * n_exp);
                    const auto [bmin, bmax] = std::minmax_element(rc.bias.at(il).begin(), rc.bias.at(il).end());
                    LOG("layer %d token %lld: logits [%g, %g] bias [%g, %g]\n", il, (long long) t, *lmin, *lmax, *bmin, *bmax);
                    std::string nans;
                    int64_t n_nan = 0;
                    for (int64_t e = 0; e < n_exp; e++) {
                        if (std::isnan(lg[t * n_exp + e]) && n_nan++ < 16) {
                            nans += " " + std::to_string(e);
                        }
                    }
                    int64_t n_nan_all = 0;
                    for (float v : lg) {
                        n_nan_all += std::isnan(v);
                    }
                    LOG("  nan logits: token %lld/%lld, all %lld/%zu, at%s; logits[0..3] %g %g %g %g\n", (long long) n_nan,
                        (long long) n_exp, (long long) n_nan_all, lg.size(), nans.c_str(), lg[t * n_exp], lg[t * n_exp + 1],
                        lg[t * n_exp + 2], lg[t * n_exp + 3]);
                    for (int64_t i = 0; i < n_used + 1; i++) {
                        LOG("  #%lld dev %4d (%.6f)  cpu %4d (%.6f)\n", (long long) i, d[i], p[d[i]], c[i], p[c[i]]);
                    }
                }
                s.set_diff++;
                s.diff_margin = std::min(s.diff_margin, margin);
                continue;
            }
            s.order_diff++;
        }
        // same set: compare weights by expert id
        for (int64_t i = 0; i < n_used; i++) {
            const int64_t j = std::find(c, c + n_used, d[i]) - c;
            s.max_w_diff = std::max(s.max_w_diff, std::fabs(w_dev[t * n_used + i] - w_cpu[t * n_used + j]));
        }
    }
}

// MOE_ROUTE_DEBUG=2: ask for every node, log the first ones with NaN output
static bool nan_cb(ggml_tensor * t, bool ask, void * user_data) {
    auto * n_logged = (int *) user_data;
    if (ask) {
        return true;
    }
    if (*n_logged >= 8 || t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) {
        return true;
    }
    const auto x = get_data<float>(t);
    const long long n_nan = std::count_if(x.begin(), x.end(), [](float v) { return std::isnan(v); });
    if (n_nan) {
        std::string srcs;
        for (int i = 0; i < GGML_MAX_SRC && t->src[i]; i++) {
            srcs += std::string(" ") + t->src[i]->name + "(" + ggml_op_desc(t->src[i]) + ")";
        }
        LOG("nan: %s op %s nan %lld/%zu ne %lld %lld %lld srcs%s\n", t->name, ggml_op_desc(t), n_nan, x.size(),
            (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], srcs.c_str());
        (*n_logged)++;
    }
    return true;
}

static bool route_cb(ggml_tensor * t, bool ask, void * user_data) {
    auto * rc = (route_check *) user_data;
    std::string base;
    int il;
    if (!split_name(t->name, base, il)) {
        return ask ? false : true;
    }
    // the logits must be read at their own node: the sigmoid runs in place over them
    const bool debug = getenv("MOE_ROUTE_DEBUG") != nullptr;
    const bool want = base == "ffn_moe_logits" || base == "ffn_moe_probs_biased" || base == "ffn_moe_argsort" ||
                      base == "ffn_moe_weights_scaled" || (debug && base == "attn_norm");
    const bool layer_io = rc->layers.count(il) && (base == "attn_norm" || base == "l_out");
    const bool probe    = rc->probe && rc->layers.count(il) && t->view_src == nullptr && ggml_is_contiguous(t) &&
                       std::any_of(std::begin(LAYER_PROBES), std::end(LAYER_PROBES), [&](const char * p) { return base == p; });
    if (ask) {
        return want || layer_io || probe;
    }
    if (probe) {
        GGML_ASSERT(t->type == GGML_TYPE_F32);
        if (rc->ref) {
            const auto & dev = rc->probes.at(t->name).at(rc->n_probe[t->name]++);
            rc->last_cpu[t->name] = get_data<float>(t);
            probe_add(rc->pstats[t->name], rc->last_cpu[t->name], dev);
            rc->cpu_probes[t->name].push_back(rc->last_cpu[t->name]);
        } else {
            rc->probes[t->name].push_back(get_data<float>(t));
        }
        return true;
    }
    if (layer_io && base == "attn_norm") {
        if (rc->ref) {
            const auto & v = rc->inp.at(il).at(rc->n_inp[il]++);
            GGML_ASSERT(v.size() == (size_t) ggml_nelements(t) && ggml_is_contiguous(t));
            ggml_backend_tensor_set(t, v.data(), 0, ggml_nbytes(t));
        } else {
            rc->inp[il].push_back(get_data<float>(t));
        }
    }
    if (layer_io && base == "l_out") {
        if (rc->ref) {
            const auto & dev = rc->out.at(il).at(rc->n_out[il]++);
            const auto   cpu = get_data<float>(t);
            GGML_ASSERT(dev.size() == cpu.size());
            const layer_out_stats ls = compare_out(cpu, dev, t->ne[0]);
            if (rc->probe) {
                rc->cpu_probes[t->name].push_back(cpu);
            }
            const std::string shexp = "ffn_shexp-" + std::to_string(il);
            if (rc->probe && rc->last_cpu.count(shexp)) {
                const auto & dev_sh = rc->probes.at(shexp).at(rc->n_probe[shexp] - 1);
                const auto & cpu_sh = rc->last_cpu.at(shexp);
                std::vector<float> dev_up(dev.size()), cpu_up(cpu.size());
                for (size_t i = 0; i < cpu.size(); i++) {
                    dev_up[i] = dev[i] - dev_sh[i];
                    cpu_up[i] = cpu[i] - cpu_sh[i];
                }
                probe_add(rc->pstats[std::string(LATENT_UP_PROBE) + "-" + std::to_string(il)], cpu_up, dev_up);
            }
            LOG("layer-check %3d ubatch %zu tokens %5lld: nmse %.3e max_abs %.3e worst token %lld nmse %.3e rms cpu %.3e dev %.3e -> %s\n",
                il, rc->n_out[il] - 1, (long long) t->ne[1], ls.nmse, ls.max_abs, (long long) ls.worst_tok,
                ls.worst_tok_nmse, ls.worst_rms_ref, ls.worst_rms_x, ls.nmse <= LAYER_NMSE_MAX ? "OK" : "FAIL");
            rc->nmse_max = std::max(rc->nmse_max, ls.nmse);
        } else {
            rc->out[il].push_back(get_data<float>(t));
            if (rc->probe) {
                rc->probes[t->name].push_back(rc->out[il].back());
            }
        }
        return true;
    }
    if (base == "attn_norm") {
        if (!debug) {
            return true;
        }
        const auto x = get_data<float>(t);
        LOG("attn_norm-%d: op %s nan %lld/%zu\n", il, ggml_op_desc(t),
            (long long) std::count_if(x.begin(), x.end(), [](float v) { return std::isnan(v); }), x.size());
        return true;
    }
    if (base == "ffn_moe_logits") {
        rc->logits[il] = get_data<float>(t);
        if (debug) {
            // router input: an fp16 matmul path overflows on |x| > 65504
            const auto x = get_data<float>(t->src[1]);
            float amax = 0.0f;
            long long n_big = 0, n_bad_tok = 0;
            for (int64_t r = 0; r < t->src[1]->ne[1]; r++) {
                bool big = false;
                for (int64_t c = 0; c < t->src[1]->ne[0]; c++) {
                    const float a = std::fabs(x[r * t->src[1]->ne[0] + c]);
                    amax = std::max(amax, a);
                    big |= !(a <= 65504.0f);
                    n_big += !(a <= 65504.0f);
                }
                n_bad_tok += big;
            }
            const auto & lg = rc->logits[il];
            LOG("logits-%d: nan %lld/%zu; input %s max|x| %g, |x|>fp16 max %lld in %lld/%lld tokens\n", il,
                (long long) std::count_if(lg.begin(), lg.end(), [](float v) { return std::isnan(v); }), lg.size(),
                t->src[1]->name, amax, n_big, n_bad_tok, (long long) t->src[1]->ne[1]);
        }
    } else if (base == "ffn_moe_probs_biased") {
        rc->bias[il] = get_data<float>(t->src[1]);
    } else if (base == "ffn_moe_argsort") {
        rc->argsort[il] = get_data<int32_t>(t);
    } else if (base == "ffn_moe_weights_scaled") {
        compare(*rc, il, t);
    }
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    route_check rc;
    rc.cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    GGML_ASSERT(rc.cpu);

    const char * env_layers = getenv("MOE_LAYER");
    const char * env_dump   = getenv("MOE_LAYER_DUMP");
    const char * env_ref    = getenv("MOE_LAYER_REF");
    if (env_layers) {
        std::stringstream ss(env_layers);
        for (std::string tok; std::getline(ss, tok, ',');) {
            rc.layers.insert(std::stoi(tok));
        }
        rc.ref = env_ref != nullptr;
        rc.probe = getenv("MOE_LAYER_PROBE") != nullptr;
        if (rc.ref && (!layers_load(rc, env_ref) || (rc.probe && !probes_load(rc, env_ref)))) {
            LOG_ERR("%s: cannot read %s\n", __func__, env_ref);
            return 1;
        }
    }

    int n_nan_logged = 0;
    const char * dbg = getenv("MOE_ROUTE_DEBUG");
    params.cb_eval           = dbg && atoi(dbg) == 2 ? nan_cb : route_cb;
    params.cb_eval_user_data = dbg && atoi(dbg) == 2 ? (void *) &n_nan_logged : (void *) &rc;
    params.warmup            = false;

    auto llama_init = common_init_from_params(params);
    llama_context * ctx = llama_init->context();
    if (ctx == nullptr) {
        LOG_ERR("%s: failed to init\n", __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(llama_init->model());
    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, llama_vocab_get_add_bos(vocab), true);
    if (tokens.empty()) {
        LOG_ERR("%s: empty prompt, use -p or -f\n", __func__);
        return 1;
    }
    LOG_INF("prompt tokens = %zu\n", tokens.size());

    for (size_t i = 0; i < tokens.size(); i += params.n_batch) {
        const int n = (int) std::min<size_t>(params.n_batch, tokens.size() - i);
        if (llama_decode(ctx, llama_batch_get_one(tokens.data() + i, n))) {
            LOG_ERR("%s: decode failed\n", __func__);
            return 1;
        }
    }

    int64_t n_bad = 0, n_ties = 0;
    LOG("\n%5s %6s %4s %7s %8s %10s %5s %12s %12s %12s  %s\n", "layer", "n_exp", "k", "tokens", "set_diff",
        "order_diff", "ties", "min_margin", "diff_margin", "max_w_diff", "buffer");
    for (const auto & [il, s] : rc.stats) {
        LOG("%5d %6lld %4lld %7lld %8lld %10lld %5lld %12.3e %12.3e %12.3e  %s\n", il, (long long) s.n_exp,
            (long long) s.n_used, (long long) s.n_tokens, (long long) s.set_diff, (long long) s.order_diff,
            (long long) s.ties, s.min_margin, s.diff_margin, s.max_w_diff, s.buft.c_str());
        n_bad += s.set_diff;
        n_ties += s.ties;
    }
    LOG("layers = %zu, tokens with a different expert set = %lld (ties %lld) -> %s\n", rc.stats.size(),
        (long long) n_bad, (long long) n_ties, rc.stats.empty() ? "NO MOE LAYERS SEEN" : n_bad ? "MISMATCH" : "MATCH");

    bool layers_ok = true;
    if (!rc.layers.empty()) {
        if (rc.ref) {
            size_t n_checked = 0;
            for (const auto & [il, n] : rc.n_out) {
                n_checked += n;
            }
            layers_ok = n_checked > 0 && rc.nmse_max <= LAYER_NMSE_MAX;
            if (rc.probe) {
                LOG("\n%-24s %8s %10s %10s %10s %10s %10s %10s\n", "probe", "ubatches", "nmse", "rms_cpu", "rms_dev",
                    "amax_cpu", "zero_dev", "collapsed");
                for (const auto & [name, ps] : rc.pstats) {
                    LOG("%-24s %8lld %10.3e %10.3e %10.3e %10.3e %9.2f%% %10lld  (zero_cpu %.2f%%)\n", name.c_str(),
                        (long long) ps.n_ub, ps.err / ps.nrm_ref, std::sqrt(ps.nrm_ref / ps.n), std::sqrt(ps.nrm_x / ps.n),
                        ps.amax_ref, 100.0 * ps.zero_x / ps.n, (long long) ps.collapsed, 100.0 * ps.zero_ref / ps.n);
                }
            }
            if (rc.probe && env_dump) {
                LOG("layer-dump: cpu probes -> %s %s\n", probe_path(env_dump).c_str(),
                    probes_save(rc.cpu_probes, env_dump) ? "saved" : "FAILED");
            }
            LOG("layer-check: %zu layer outputs, max nmse %.3e (limit %.1e) -> %s\n", n_checked, rc.nmse_max,
                LAYER_NMSE_MAX, layers_ok ? "MATCH" : "MISMATCH");
        } else if (env_dump) {
            layers_ok = layers_save(rc, env_dump) && (!rc.probe || probes_save(rc.probes, env_dump));
            LOG("layer-dump: %zu layers -> %s %s\n", rc.out.size(), env_dump, layers_ok ? "saved" : "FAILED");
        }
    }

    ggml_backend_free(rc.cpu);
    llama_backend_free();
    return rc.stats.empty() || n_bad || !layers_ok ? 1 : 0;
}
