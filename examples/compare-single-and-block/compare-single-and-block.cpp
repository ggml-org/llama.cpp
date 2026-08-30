// Single-token vs multi-token (block) decode numerical comparison.
//
// What it measures: given identical token conditioning (both paths prefill the
// same prefix with the same batch shape, so the prefix KV is bit-identical),
// does a W-token batch decode produce the same logits as W single-token decode
// steps? This isolates floating-point batch-shape effects — NOT trajectory
// drift: all comparisons are teacher-forced along the original AR trajectory.
//
// Method:
//   1) AR greedy decode, 1 token at a time — the ground-truth trajectory
//   2) for each position t: rebuild prefix prompt + AR[0..t-1] on two contexts
//      with fully wiped KV, decode the same W inputs
//      [last, AR[t], ..., AR[t+W-2]] as one batch (all logits=true) on one
//      context, step those tokens one-by-one on the other, and compare logits
//      at every block offset j
//
// Output (default): run config (including prompt_tokens and n_gen) immediately
// followed by pairwise logit-diff statistics and flip details. A one-line
// status on stderr is erased before the report. Pass --verbose for the AR
// trajectory, per-window max|dlogit|, the banded dlogit table, and the
// per-cell absolute-logit table. Argmax disagreements are recorded and the
// run continues to the length cap (-n); every comparison stays teacher-forced
// on the original AR trajectory.
//
// Exit codes: 0 = no argmax mismatch, 1 = argmax mismatch found, 2 = error
//
// Usage:
// ./llama-compare-single-and-block \   # 
//     -m model.gguf \                          # model path
//     -p "Explain the Pythagorean theorem" \   # prompt (default if omitted)
//     -n 100 \                                  # max generate length (default 100)
//     -c 4096 \                                # max context (need >= prompt + max(n_gen, block_width))
//     -b 512 \                                 # logical max batch (must be >= block_width)
//     -ub 512 \                                # physical max batch (N < block_width tests ubatch split)
//     -np 1 \                                  # parallel sequences (this tool uses 1)
//     -ngl 0 \                                 # GPU layers (0 = CPU)
//     -t 4 \                                   # CPU threads for generation
//     -tb 4 \                                  # CPU threads for batch / prompt
//     -fa off \                                # flash attention: on|off|auto
//     --block-width 5 \                        # tokens per block decode (default 5)
//     --json /tmp/compare_report.json          # write JSON report (optional)
//     --verbose \                              # AR trajectory + dlogit/per-cell tables

#include "arg.h"
#include "chat.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

#ifdef _WIN32
#    include <io.h>
#    define isatty _isatty
#    define fileno _fileno
#else
#    include <unistd.h>
#endif

struct diag_config {
    int         block_width = 5;
    std::string json_path;
    bool        verbose = false;
};

static bool stderr_tty() {
    return isatty(fileno(stderr)) != 0;
}

// Single in-place status line on stderr. Kept short so it cannot wrap
// (a wrapped \r only rewinds the last visual row and concatenates junk).
// Erased before the report so run config and stats stay adjacent.
static void progress_bar(const char * label, int done, int total, const char * extra) {
    if (!stderr_tty() || total <= 0) {
        return;
    }
    fprintf(stderr, "\r\033[2K%s %d/%d", label, done, total);
    if (extra && extra[0]) {
        fprintf(stderr, " %s", extra);
    }
    fflush(stderr);
}

static void progress_clear() {
    if (!stderr_tty()) {
        return;
    }
    fprintf(stderr, "\r\033[2K");
    fflush(stderr);
}

static void print_run_config(
        const llama_model * model,
        const common_params & params,
        const diag_config & cfg,
        int prompt_tokens,
        int n_gen);

static std::string json_escape(const std::string & s) {
    std::string o;
    o.reserve(s.size() + 8);
    for (unsigned char c : s) {
        if (c == '\\' || c == '"') {
            o.push_back('\\');
            o.push_back((char) c);
        } else if (c == '\n') {
            o += "\\n";
        } else if (c == '\t') {
            o += "\\t";
        } else if (c < 32) {
            char buf[8];
            snprintf(buf, sizeof(buf), "\\u%04x", c);
            o += buf;
        } else {
            o.push_back((char) c);
        }
    }
    return o;
}

static std::string tok_str(const llama_context * ctx, llama_token id) {
    std::string s = common_token_to_piece(ctx, id);
    std::string out;
    out.reserve(s.size() * 2);
    for (unsigned char c : s) {
        if (c == '\n') {
            out += "\\n";
        } else if (c == '\t') {
            out += "\\t";
        } else if (c < 32 || c == 127) {
            char buf[8];
            snprintf(buf, sizeof(buf), "\\x%02x", c);
            out += buf;
        } else {
            out.push_back((char) c);
        }
    }
    return out;
}

// ASCII-safe, width-clamped token piece for table cells
static std::string piece_cell(const llama_context * ctx, llama_token id, size_t maxw) {
    const std::string s = common_token_to_piece(ctx, id);
    std::string out;
    for (unsigned char c : s) {
        if (c < 32 || c >= 127) {
            char buf[8];
            snprintf(buf, sizeof(buf), "\\x%02x", c);
            out += buf;
        } else {
            out.push_back((char) c);
        }
        if (out.size() > maxw + 4) {
            break; // one escape can overshoot maxw; final clamp below
        }
    }
    if (out.size() > maxw) {
        out = out.substr(0, maxw - 1) + "~";
    }
    return out;
}

static llama_token argmax_logit(const float * logits, int n_vocab, float * out_logit) {
    llama_token best = 0;
    float       best_l = logits[0];
    for (int i = 1; i < n_vocab; ++i) {
        if (logits[i] > best_l) {
            best_l = logits[i];
            best   = i;
        }
    }
    if (out_logit) {
        *out_logit = best_l;
    }
    return best;
}

// ------------------------------------------------------------------
// RAII batch wrapper to avoid leaks on early return
// ------------------------------------------------------------------
struct batch_raii {
    llama_batch b;
    batch_raii(int n_tokens, int n_embd, int n_seq_max) {
        b = llama_batch_init(n_tokens, n_embd, n_seq_max);
    }
    ~batch_raii() { llama_batch_free(b); }
    operator llama_batch&() { return b; }
    llama_batch* operator&() { return &b; }
};

// Prefill n tokens, chunked to respect the context's n_batch limit.
// llama_decode asserts n_tokens <= n_batch, so a long prompt submitted in one
// shot aborts the process instead of returning an error.
static bool decode_prefill(llama_context * ctx, const llama_token * toks, int n) {
    const int n_batch = (int) llama_n_batch(ctx);
    for (int i = 0; i < n; i += n_batch) {
        const int chunk = std::min(n_batch, n - i);
        if (llama_decode(ctx, llama_batch_get_one(const_cast<llama_token *>(toks + i), chunk)) != 0) {
            return false;
        }
    }
    return true;
}

// ------------------------------------------------------------------
// Chat prompt builder
// ------------------------------------------------------------------
static std::vector<llama_token> build_chat_prompt(const llama_model * model, const std::string & user_msg) {
    auto tmpls = common_chat_templates_init(model, "");
    common_chat_templates_inputs inputs;
    inputs.use_jinja             = true;
    inputs.enable_thinking       = false;
    inputs.add_generation_prompt = true;
    common_chat_msg user;
    user.role    = "user";
    user.content = user_msg;
    inputs.messages.push_back(user);
    inputs.chat_template_kwargs["enable_thinking"] = "false";

    auto params = common_chat_templates_apply(tmpls.get(), inputs);
    return common_tokenize(llama_model_get_vocab(model), params.prompt, true, true);
}

struct tok_rec {
    llama_token id = 0;
    float       logit = 0.0f;
};

// one table cell: trajectory token p = t + j predicted at block offset j
struct cmp_cell {
    bool        valid = false;
    float       l_s   = 0.0f;   // AR-token logit, single-token decode
    float       l_b   = 0.0f;   // AR-token logit, block decode
    llama_token amax_s  = 0;    // argmax id, single-token decode
    llama_token amax_b  = 0;    // argmax id, block decode
    float       lmax_s  = 0.0f; // argmax logit, single-token decode
    float       lmax_b  = 0.0f; // argmax logit, block decode
};

struct flip_rec {
    int         t = 0;
    int         j = 0;
    llama_token single_id = 0;    // argmax of the single-token path
    llama_token block_id  = 0;    // argmax of the block path
    float       l_single_max = 0.0f;
    float       l_block_max  = 0.0f;
    float       l_s = 0.0f;       // AR-token logit, single path
    float       l_b = 0.0f;       // AR-token logit, block path
};

// moment accumulator for a diff stream. |d|^2 == d^2, so three sums cover both
// the plain and the abs statistics; variances are population moments (the
// cells are the full population of this run, not a sample)
struct stat_acc {
    double n    = 0.0;
    double sum  = 0.0; // Σ d
    double sum2 = 0.0; // Σ d²  (== Σ |d|²)
    double suma = 0.0; // Σ |d|

    void add(double d) {
        n    += 1.0;
        sum  += d;
        sum2 += d * d;
        suma += std::abs(d);
    }

    double mean()     const { return n > 0.0 ? sum  / n : 0.0; }
    double abs_mean() const { return n > 0.0 ? suma / n : 0.0; }
    double var()      const { return n > 0.0 ? std::max(0.0, sum2 / n - mean()     * mean())     : 0.0; }
    double abs_var()  const { return n > 0.0 ? std::max(0.0, sum2 / n - abs_mean() * abs_mean()) : 0.0; }
};

// pairwise diffs over all valid cells: l_ar - l_single measures
// prefill-vs-incremental drift, l_single - l_block the block-shape effect
struct logit_stats {
    stat_acc ar_single;
    stat_acc ar_block;
    stat_acc single_block;
};

// ------------------------------------------------------------------
// AR greedy baseline
// ------------------------------------------------------------------
static std::string path_stem(const std::string & path) {
    size_t slash = path.find_last_of("/\\");
    std::string b = (slash == std::string::npos) ? path : path.substr(slash + 1);
    size_t dot = b.rfind('.');
    if (dot != std::string::npos) {
        b = b.substr(0, dot);
    }
    return b;
}

static std::string model_label(const llama_model * model, const std::string & path) {
    char name[256];
    if (llama_model_meta_val_str(model, "general.name", name, sizeof(name)) > 0 && name[0]) {
        return name;
    }
    return path_stem(path);
}

static std::string ftype_short(const llama_model * model) {
    const char * n = llama_ftype_name(llama_model_ftype(model));
    if (!n || !n[0]) {
        return "unknown";
    }
    std::string s(n);
    const size_t sp = s.find(' ');
    if (sp != std::string::npos) {
        s = s.substr(0, sp);
    }
    for (char & c : s) {
        if (c >= 'A' && c <= 'Z') {
            c = (char) (c - 'A' + 'a');
        }
    }
    return s;
}

static const char * fa_flag(llama_flash_attn_type t) {
    switch (t) {
        case LLAMA_FLASH_ATTN_TYPE_ENABLED:  return "on";
        case LLAMA_FLASH_ATTN_TYPE_DISABLED: return "off";
        default:                             return "auto";
    }
}

static std::vector<tok_rec> run_ar_greedy(
        llama_model * model,
        const common_params & params_base,
        const std::vector<llama_token> & prompt,
        int n_gen,
        bool verbose) {
    auto cparams = common_context_params_to_llama(params_base);
    llama_context_ptr ctx(llama_init_from_model(model, cparams));
    if (!ctx) {
        LOG_ERR("failed to create AR context\n");
        return {};
    }

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

    if (!decode_prefill(ctx.get(), prompt.data(), (int) prompt.size() - 1)) {
        LOG_ERR("AR prefill failed\n");
        return {};
    }

    llama_token id_last = prompt.back();
    std::vector<tok_rec> out;
    out.reserve(n_gen);

    if (verbose) {
        printf("\n=== Ground-truth: AR greedy (n_gen=%d) ===\n", n_gen);
    }
    for (int t = 0; t < n_gen; ++t) {
        if (llama_decode(ctx.get(), llama_batch_get_one(&id_last, 1)) != 0) {
            progress_clear();
            LOG_ERR("AR decode failed at t=%d\n", t);
            return {};
        }
        const float * logits = llama_get_logits_ith(ctx.get(), -1);
        float         best_l = 0.0f;
        llama_token   id     = argmax_logit(logits, n_vocab, &best_l);

        out.push_back({id, best_l});
        if (verbose) {
            printf("AR[%3d] id=%6d logit=%9.4f piece='%s'\n", t, id, best_l, tok_str(ctx.get(), id).c_str());
        } else {
            progress_bar("AR", t + 1, n_gen, nullptr);
        }
        if (llama_vocab_is_eog(vocab, id)) {
            break;
        }
        id_last = id;
    }
    return out;
}

// ------------------------------------------------------------------
// Table rendering
// ------------------------------------------------------------------
static logit_stats compute_logit_stats(
        const std::vector<tok_rec> & ar,
        const std::vector<std::vector<cmp_cell>> & cells) {
    logit_stats st;
    for (int p = 0; p < (int) cells.size(); ++p) {
        for (const auto & c : cells[p]) {
            if (!c.valid) {
                continue;
            }
            const double l_ar = ar[p].logit;
            st.ar_single.add(l_ar - c.l_s);
            st.ar_block.add(l_ar - c.l_b);
            st.single_block.add(c.l_s - c.l_b);
        }
    }
    return st;
}

static void print_stat_row(const char * name, const stat_acc & s) {
    printf("%-28s %+12.6f %12.6f %12.6f %12.6f\n",
           name, s.mean(), s.var(), s.abs_mean(), s.abs_var());
}

static void print_logit_stats(const logit_stats & st) {
    printf("\n=== pairwise logit-diff statistics (n=%d cells) ===\n",
           (int) st.single_block.n);
    printf("%-28s %12s %12s %12s %12s\n", "pair", "mean", "var", "mean|d|", "var|d|");
    print_stat_row("l_ar - l_single", st.ar_single);
    print_stat_row("l_ar - l_block",  st.ar_block);
    print_stat_row("l_single - l_block (dlogit)", st.single_block);
}

static void print_flip(
        const llama_context * ctx,
        const std::vector<tok_rec> & ar,
        const flip_rec & flip) {
    const int p = flip.t + flip.j;
    printf("FLIP at pos=%d (window t=%d, block offset j=%d):\n", p, flip.t, flip.j);
    printf("  single argmax=%6d('%s') l=%.4f | block argmax=%6d('%s') l=%.4f\n",
           flip.single_id, tok_str(ctx, flip.single_id).c_str(), flip.l_single_max,
           flip.block_id,  tok_str(ctx, flip.block_id).c_str(),  flip.l_block_max);
    printf("  AR token=%6d('%s') l_s=%.4f l_b=%.4f dlogit=%+.4f\n",
           ar[p].id, tok_str(ctx, ar[p].id).c_str(), flip.l_s, flip.l_b, flip.l_s - flip.l_b);
    if (flip.single_id != ar[p].id) {
        printf("  note: single-path argmax also deviates from the AR trajectory — "
               "prefill-vs-incremental drift, not a block effect\n");
    }
}

static int count_flip_positions(const std::vector<flip_rec> & flips) {
    std::vector<int> pos;
    pos.reserve(flips.size());
    for (const auto & f : flips) {
        pos.push_back(f.t + f.j);
    }
    std::sort(pos.begin(), pos.end());
    return (int) (std::unique(pos.begin(), pos.end()) - pos.begin());
}

static int count_checked(const std::vector<std::vector<cmp_cell>> & cells) {
    int n = 0;
    for (const auto & row : cells) {
        for (const auto & c : row) {
            if (c.valid) {
                ++n;
            }
        }
    }
    return n;
}

static void print_table(
        const llama_context * ctx,
        const std::vector<tok_rec> & ar,
        const std::vector<std::vector<cmp_cell>> & cells,
        const std::vector<flip_rec> & flips,
        int block_width,
        bool verbose) {
    if (verbose) {
        printf("\n=== dlogit = logit(single) - logit(block), per trajectory token ===\n");
        printf("%4s %7s %-14s", "pos", "id", "piece");
        for (int j = 0; j < block_width; ++j) {
            printf(" %7s%d", "j=", j);
        }
        printf("\n");

        for (int p = 0; p < (int) ar.size(); ++p) {
            bool any = false;
            bool row_flip = false;
            for (int j = 0; j < block_width; ++j) {
                if (cells[p][j].valid) {
                    any = true;
                    row_flip = row_flip || (cells[p][j].amax_s != cells[p][j].amax_b);
                }
            }
            if (!any) {
                continue; // never covered by a window
            }

            printf("%4d %7d %-14s", p, ar[p].id, piece_cell(ctx, ar[p].id, 14).c_str());
            for (int j = 0; j < block_width; ++j) {
                if (cells[p][j].valid) {
                    printf(" %+8.4f", cells[p][j].l_s - cells[p][j].l_b);
                } else {
                    printf(" %8s", ".");
                }
            }
            if (row_flip) {
                printf("  <-- FLIP");
            }
            printf("\n");
        }

        // per-cell detail: absolute logits of both paths + argmax of each.
        // l_ar is the AR-baseline logit for the same token; l_ar vs l_single
        // quantifies prefill-vs-incremental drift, l_single vs l_block the
        // block-shape effect.
        printf("\n=== per-cell logits: single vs block (teacher-forced on AR token) ===\n");
        printf("%4s %4s %2s %7s %-14s %9s %9s %9s %8s  %-14s %-14s\n",
               "pos", "t", "j", "id", "AR ground-truth",
               "l_ar", "l_single", "l_block", "dlogit", "argmax(single)", "argmax(block)");
        for (int p = 0; p < (int) ar.size(); ++p) {
            for (int j = 0; j < block_width; ++j) {
                const auto & c = cells[p][j];
                if (!c.valid) {
                    continue;
                }
                printf("%4d %4d %2d %7d %-14s %9.4f %9.4f %9.4f %+8.4f  %-14s %-14s%s\n",
                       p, p - j, j, ar[p].id, piece_cell(ctx, ar[p].id, 14).c_str(),
                       ar[p].logit, c.l_s, c.l_b, c.l_s - c.l_b,
                       piece_cell(ctx, c.amax_s, 14).c_str(),
                       piece_cell(ctx, c.amax_b, 14).c_str(),
                       c.amax_s != c.amax_b ? "  <-- FLIP" : "");
            }
        }
    }

    print_logit_stats(compute_logit_stats(ar, cells));

    printf("\nsummary: tokens=%zu checked=%d flips=%zu cells across %d positions\n",
           ar.size(), count_checked(cells), flips.size(), count_flip_positions(flips));
    if (flips.empty()) {
        printf("no argmax mismatch\n");
        return;
    }

    for (const auto & flip : flips) {
        print_flip(ctx, ar, flip);
    }
}

// ------------------------------------------------------------------
// Core comparison: single-step decode vs one W-token batch decode.
// Both contexts are created once and reused; the KV is fully wiped before each
// iteration so both sides re-prefill the same prefix with the same shape,
// keeping the prefix KV bit-identical between the two paths. 
// note: partial KV rollback would contaminate the prefix with prior-iteration shape effects
// ------------------------------------------------------------------
static bool compare_single_vs_block(
        llama_model * model,
        const common_params & params_base,
        const std::vector<llama_token> & prompt,
        const std::vector<tok_rec> & ar,
        const diag_config & cfg,
        std::vector<std::vector<cmp_cell>> & cells, // [pos][j], pos = t + j
        std::vector<flip_rec> & flips) {
    if (cfg.verbose) {
        printf("\n=== single vs block (width=%d) ===\n", cfg.block_width);
    }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    auto cparams = common_context_params_to_llama(params_base);

    llama_context_ptr ctx_block(llama_init_from_model(model, cparams));
    llama_context_ptr ctx_single(llama_init_from_model(model, cparams));
    if (!ctx_block || !ctx_single) {
        LOG_ERR("failed to create comparison contexts\n");
        return false;
    }

    const int n_pos = (int) ar.size();
    cells.assign(n_pos, std::vector<cmp_cell>(cfg.block_width));

    for (int t = 0; t < n_pos; ++t) {
        std::vector<llama_token> seq = prompt;
        for (int i = 0; i < t; ++i) {
            seq.push_back(ar[i].id);  // seq: prompt + AR[0..t-1]
        }

        // wipe all KV, then prefill both contexts with the identical prefix
        llama_memory_seq_rm(llama_get_memory(ctx_block.get()), 0, -1, -1);
        llama_memory_seq_rm(llama_get_memory(ctx_single.get()), 0, -1, -1);

        const int n_past = (int) seq.size() - 1;
        if (!decode_prefill(ctx_block.get(), seq.data(), n_past)) {
            progress_clear();
            LOG_ERR("block prefill failed at t=%d\n", t);
            return false;
        }
        if (!decode_prefill(ctx_single.get(), seq.data(), n_past)) {
            progress_clear();
            LOG_ERR("single prefill failed at t=%d\n", t);
            return false;
        }

        const int n_extra = std::min(cfg.block_width, n_pos - t);

        // --- block decode: the same W inputs as the single path, one llama_decode ---
        // [seq.back(), AR[t], ..., AR[t+n_extra-2]] — n_extra tokens, all logits=true
        batch_raii batch(n_extra, 0, 1);
        common_batch_clear(batch);
        for (int j = 0; j < n_extra; ++j) {
            const llama_token tok = (j == 0) ? seq.back() : ar[t + j - 1].id;
            common_batch_add(batch, tok, n_past + j, { 0 }, true);
        }
        // shape of this call:
        //   input:  [n_extra] tokens in ONE llama_decode, single seq {0},
        //           explicit positions n_past .. n_past+n_extra-1
        //           (token[0] = last prefix token, token[j>0] = AR[t+j-1])
        //   output: [n_extra] logits rows — every batch token has output=true.
        //           row j predicts trajectory position p = t + j, read via
        //           llama_get_logits_ith(ctx_block, j)  (j = batch token index)
        if (llama_decode(ctx_block.get(), batch) != 0) {
            progress_clear();
            LOG_ERR("block decode failed at t=%d\n", t);
            return false;
        }

        // --- single decode: step through the same tokens one-by-one ---
        llama_token single_id  = seq.back();
        float       win_max = 0.0f;
        int         win_flips = 0;

        for (int j = 0; j < n_extra; ++j) {
            // shape of this call:
            //   input:  [1] token per llama_decode. llama_batch_get_one leaves
            //           pos/seq_id/logits as nullptr, so llama auto-assigns:
            //           seq 0, pos = current KV end (= n_past + j at this
            //           step, matching the block path's explicit positions),
            //           and marks only the last token as output
            //   output: [1] logits row — the prediction for trajectory
            //           position p = t + j, read via
            //           llama_get_logits_ith(ctx_single, -1) (last output row)
            if (llama_decode(ctx_single.get(), llama_batch_get_one(&single_id, 1)) != 0) {
                progress_clear();
                LOG_ERR("single_id decode failed at t=%d j=%d\n", t, j);
                return false;
            }
            const float * logits_single = llama_get_logits_ith(ctx_single.get(), -1);
            const float * logits_block = llama_get_logits_ith(ctx_block.get(), j);

            const int p = t + j;

            float lr = 0.0f, lb = 0.0f;
            const llama_token arg_single = argmax_logit(logits_single, n_vocab, &lr);
            const llama_token arg_block  = argmax_logit(logits_block,  n_vocab, &lb);

            cmp_cell cell;
            cell.valid  = true;
            cell.l_s    = logits_single[ar[p].id];
            cell.l_b    = logits_block[ar[p].id];
            cell.amax_s = arg_single;
            cell.amax_b = arg_block;
            cell.lmax_s = lr;
            cell.lmax_b = lb;
            cells[p][j] = cell;
            win_max = std::max(win_max, std::abs(cell.l_s - cell.l_b));

            if (arg_single != arg_block) {
                flips.push_back({ t, j, arg_single, arg_block, lr, lb, cell.l_s, cell.l_b });
                ++win_flips;
            }
            single_id = ar[p].id; // teacher-force along the AR trajectory, including after a flip
        }
        if (cfg.verbose) {
            printf("t=%3d window=[%3d..%3d] max|dlogit|=%.4f%s\n",
                   t, t, t + n_extra - 1, win_max, win_flips ? "  <-- FLIP" : "");
        } else {
            char extra[32];
            snprintf(extra, sizeof(extra), "flips=%zu", flips.size());
            progress_bar("compare", t + 1, n_pos, extra);
        }
    }

    progress_clear();

    // determinism self-check: the t=0 single path replays the AR baseline's
    // exact compute graph (same prefill chunks, same 1-token steps), so on a
    // deterministic backend l_s must reproduce l_ar bit-for-bit. Any drift
    // means run-to-run non-determinism (e.g. GPU atomics), which invalidates
    // the bit-identical-prefix premise of every cell in this run.
    //
    // cells[p][j] is indexed by trajectory pos p = t+j, so the t=0 window is
    // the diagonal cells[j][j] — NOT cells[p][0]. cells[p][0] is window t=p,
    // whose prefix was re-prefilled as a batch and therefore measures
    // prefill-vs-incremental drift, not determinism.
    for (int j = 0; j < std::min(cfg.block_width, n_pos); ++j) {
        const cmp_cell & c = cells[j][j];
        if (c.valid && c.l_s != ar[j].logit) { // bitwise equality intended
            LOG_WRN("determinism check failed: t=0 single path deviates from AR baseline "
                    "at p=%d (l_s=%.6f l_ar=%.6f |d|=%g); absolute diffs in this run may be noise\n",
                    j, c.l_s, ar[j].logit, std::abs(c.l_s - ar[j].logit));
            break;
        }
    }

    print_run_config(model, params_base, cfg, (int) prompt.size(), (int) ar.size());
    print_table(ctx_single.get(), ar, cells, flips, cfg.block_width, cfg.verbose);
    return true;
}

// ------------------------------------------------------------------
// JSON export
// ------------------------------------------------------------------
static void write_json(
        const std::string & path,
        const common_params & params,
        const std::string & prompt,
        const std::vector<llama_token> & prompt_tokens,
        const std::vector<tok_rec> & ar,
        const std::vector<std::vector<cmp_cell>> & cells,
        const std::vector<flip_rec> & flips,
        const diag_config & cfg) {
    std::ofstream f(path);
    if (!f) {
        LOG_ERR("failed to write %s\n", path.c_str());
        return;
    }
    f << std::setprecision(9); // round-trip precision for float
    f << "{\n  \"prompt\": \"" << json_escape(prompt) << "\",\n";
    f << "  \"prompt_tokens\": [";
    for (size_t i = 0; i < prompt_tokens.size(); ++i) {
        if (i) f << ",";
        f << prompt_tokens[i];
    }
    f << "],\n";
    // absolute diffs are only comparable under identical backend settings, so
    // the report carries the environment it was produced in
    f << "  \"config\": {"
      << "\"model\": \"" << json_escape(params.model.path) << "\""
      << ", \"flash_attn\": " << (int) params.flash_attn_type
      << ", \"n_gpu_layers\": " << params.n_gpu_layers
      << ", \"n_threads\": " << params.cpuparams.n_threads
      << ", \"n_threads_batch\": " << params.cpuparams_batch.n_threads
      << ", \"n_batch\": " << params.n_batch
      << ", \"n_ubatch\": " << params.n_ubatch
      << ", \"n_ctx\": " << params.n_ctx
      << "},\n";
    f << "  \"block_width\": " << cfg.block_width << ",\n";
    f << "  \"ar\": [";
    for (size_t i = 0; i < ar.size(); ++i) {
        if (i) f << ",";
        f << "{\"id\":" << ar[i].id << ",\"logit\":" << ar[i].logit << "}";
    }
    f << "],\n  \"n_flips\": " << flips.size()
      << ", \"n_flip_positions\": " << count_flip_positions(flips)
      << ",\n  \"flips\": [";
    for (size_t i = 0; i < flips.size(); ++i) {
        const auto & flip = flips[i];
        if (i) f << ",";
        f << "{\"t\":" << flip.t
          << ",\"j\":" << flip.j
          << ",\"pos\":" << flip.t + flip.j
          << ",\"single_id\":" << flip.single_id
          << ",\"block_id\":" << flip.block_id
          << ",\"l_single_max\":" << flip.l_single_max
          << ",\"l_block_max\":" << flip.l_block_max
          << ",\"l_s\":" << flip.l_s
          << ",\"l_b\":" << flip.l_b
          << "}";
    }
    f << "],\n  \"stats\": {";
    {
        const logit_stats st = compute_logit_stats(ar, cells);
        const char      * names[3] = { "l_ar-l_single", "l_ar-l_block", "l_single-l_block" };
        const stat_acc  * accs[3]  = { &st.ar_single, &st.ar_block, &st.single_block };
        for (int i = 0; i < 3; ++i) {
            if (i) f << ",";
            f << "\"" << names[i] << "\": {\"n\":" << (long long) accs[i]->n
              << ",\"mean\":" << accs[i]->mean()
              << ",\"var\":" << accs[i]->var()
              << ",\"abs_mean\":" << accs[i]->abs_mean()
              << ",\"abs_var\":" << accs[i]->abs_var() << "}";
        }
    }
    f << "},\n  \"cells\": [";
    bool first = true;
    for (int p = 0; p < (int) cells.size(); ++p) {
        for (int j = 0; j < (int) cells[p].size(); ++j) {
            const auto & c = cells[p][j];
            if (!c.valid) {
                continue;
            }
            if (!first) f << ",";
            first = false;
            f << "{\"pos\":" << p
              << ",\"t\":" << p - j
              << ",\"j\":" << j
              << ",\"l_s\":" << c.l_s
              << ",\"l_b\":" << c.l_b
              << ",\"dlogit\":" << c.l_s - c.l_b
              << ",\"amax_s\":" << c.amax_s
              << ",\"amax_b\":" << c.amax_b
              << ",\"lmax_s\":" << c.lmax_s
              << ",\"lmax_b\":" << c.lmax_b
              << "}";
        }
    }
    f << "]\n}\n";
    // keep the report on stdout so it stays in order with the table when
    // stdout/stderr are merged into the same log file
    printf("wrote %s\n", path.c_str());
}

// ------------------------------------------------------------------
// CLI
// ------------------------------------------------------------------
static void parse_extra_flags(int & argc, char ** argv, diag_config & cfg) {
    std::vector<char *> kept;
    kept.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        const char * a = argv[i];
        if (std::strcmp(a, "--block-width") == 0 && i + 1 < argc) {
            cfg.block_width = std::atoi(argv[++i]);
        } else if (std::strcmp(a, "--json") == 0 && i + 1 < argc) {
            cfg.json_path = argv[++i];
        } else if (std::strcmp(a, "--verbose") == 0 || std::strcmp(a, "--full") == 0) {
            cfg.verbose = true;
        } else if (std::strcmp(a, "--help-extra") == 0) {
            printf(
                "extra flags:\n"
                "  --block-width N      tokens per block decode (default 5)\n"
                "  --json PATH          write report\n"
                "  --verbose, --full    AR trajectory, dlogit/per-cell tables\n"
                "  --help-extra\n"
                "\nnotes:\n"
                "  default stdout is run config + pairwise stats + flip details.\n"
                "  argmax disagreements are recorded; the run always continues\n"
                "  to the length cap (-n, default 100), teacher-forced on the AR\n"
                "  trajectory.\n"
                "  use -ub N with N < --block-width to test ubatch-split effects.\n"
                "  keep -t/-b/-ub/-fa fixed across runs: absolute diffs are only\n"
                "  comparable under identical backend settings.\n");
            std::exit(0);
        } else {
            kept.push_back(argv[i]);
        }
    }
    argc = (int) kept.size();
    for (int i = 0; i < argc; ++i) {
        argv[i] = kept[i];
    }
}

static void print_run_config(
        const llama_model * model,
        const common_params & params,
        const diag_config & cfg,
        int prompt_tokens,
        int n_gen) {
    printf("\n--- run config ---\n");
    printf("flash_attn: %d\n", (int) params.flash_attn_type);
    printf("n_gpu_layers: %d\n", params.n_gpu_layers);
    printf("n_threads: %d / batch: %d\n", params.cpuparams.n_threads, params.cpuparams_batch.n_threads);
    printf("n_batch: %d / n_ubatch: %d\n", params.n_batch, params.n_ubatch);
    printf("block_width: %d\n", cfg.block_width);
    printf("prompt_tokens: %d\n", prompt_tokens);
    printf("n_gen: %d\n", n_gen);
    printf("model: %s %s    backend: %s  fa=%s  t=%d  W=%d\n",
           model_label(model, params.model.path).c_str(),
           ftype_short(model).c_str(),
           params.n_gpu_layers > 0 ? "GPU" : "CPU",
           fa_flag(params.flash_attn_type),
           params.cpuparams.n_threads,
           cfg.block_width);
    printf("------------------\n");
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    common_init();

    diag_config cfg;
    parse_extra_flags(argc, argv, cfg);

    common_params params;
    params.n_predict   = 100;
    params.n_ctx       = 4096;
    params.n_batch     = 512;
    params.n_ubatch    = 512;
    params.n_parallel  = 1;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 2;
    }

    const int n_gen = params.n_predict > 0 ? params.n_predict : 100;

    if (cfg.block_width < 1) {
        LOG_ERR("--block-width must be >= 1\n");
        return 2;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    // model only: this tool creates its own contexts, so the default context
    // that common_init_from_params would build (with its KV buffer) sits unused
    auto init = common_init_from_params(params, /*model_only=*/true);
    llama_model * model = init->model();
    if (!model) {
        LOG_ERR("failed to load model\n");
        return 2;
    }

    std::string user = params.prompt.empty() ? "Explain the Pythagorean theorem" : params.prompt;
    std::vector<llama_token> prompt_tokens;
    if (user.find("<|im_start|>") != std::string::npos) {
        prompt_tokens = common_tokenize(llama_model_get_vocab(model), user, true, true);
    } else {
        prompt_tokens = build_chat_prompt(model, user);
    }

    if (prompt_tokens.empty()) {
        LOG_ERR("empty prompt (chat template apply failed?)\n");
        return 2;
    }

    // deepest pos: n_past + n_extra - 1 <= P + max(n_gen, W) - 2
    const int need = (int) prompt_tokens.size() + std::max(n_gen, cfg.block_width);
    if (need > params.n_ctx) {
        LOG_ERR("prompt(%zu) + max(n_gen=%d, block_width=%d) = %d exceeds n_ctx(%d); increase -c\n",
                prompt_tokens.size(), n_gen, cfg.block_width, need, params.n_ctx);
        return 2;
    }

    // llama_decode asserts n_tokens <= n_batch (abort, not error return).
    // keep the block as one logical decode; use -ub to test physical splits
    // (they kick in when n_ubatch < block_width)
    if (cfg.block_width > params.n_batch) {
        LOG_ERR("block batch (block_width = %d) exceeds n_batch (%d); increase -b\n",
                cfg.block_width, params.n_batch);
        return 2;
    }

    LOG_INF("prompt_tokens=%zu flash_attn=%d block_width=%d\n",
            prompt_tokens.size(), (int) params.flash_attn_type, cfg.block_width);

    auto ar = run_ar_greedy(model, params, prompt_tokens, n_gen, cfg.verbose);
    if (ar.empty()) {
        return 2;
    }

    std::vector<std::vector<cmp_cell>> cells;
    std::vector<flip_rec> flips;
    const bool ok = compare_single_vs_block(model, params, prompt_tokens, ar, cfg, cells, flips);

    if (!cfg.json_path.empty()) {
        write_json(cfg.json_path, params, user, prompt_tokens, ar, cells, flips, cfg);
    }

    llama_backend_free();

    if (!ok) {
        return 2;
    }
    return flips.empty() ? 0 : 1;
}
