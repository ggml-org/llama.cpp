#include "llama-moe-predict.h"

#include "llama-impl.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>

#if defined(__linux__) || defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#define LLAMA_MOE_PREDICT_MADVISE
#endif

#if defined(__linux__)
#include <fcntl.h>
#include <sys/stat.h>
#endif

static constexpr uint32_t MOE_PRED_MAGIC   = 0x3150524b; // "KRP1"
static constexpr uint32_t MOE_PRED_VERSION = 1;
static constexpr size_t   MOE_PRED_PIECE   = 2u*1024*1024;
static constexpr size_t   MOE_PRED_QMAX    = 4096;       // LOW entries kept at most
static constexpr int      MOE_PRED_NR_MIN  = 16;
static constexpr int      MOE_PRED_EVAL    = 16;         // tokens between adaptations
static constexpr int      MOE_PRED_SAVE    = 32;         // steps between saves
static constexpr int      MOE_PRED_LOG     = 64;         // steps between log lines
static constexpr size_t   MOE_PRED_PROBE   = 1024u*1024; // residency probe: bytes per expert tensor (one mincore each)
static constexpr int      MOE_PRED_NPROBE  = 8;          // experts probed per layer at most
static constexpr int64_t  MOE_PRED_IO_GAP  = 2500000;    // us: a longer gap between io clock updates restarts the clock

// bytes the calling thread caused to be read from storage so far (0 if unknown)
static uint64_t thread_read_bytes() {
#if defined(__linux__)
    struct io_file {
        int fd = -1;
        ~io_file() {
            if (fd >= 0) {
                close(fd);
            }
        }
    };
    thread_local io_file f;
    if (f.fd < 0) {
        f.fd = open("/proc/thread-self/io", O_RDONLY | O_CLOEXEC);
        if (f.fd < 0) {
            return 0;
        }
    }
    char buf[512];
    const ssize_t n = pread(f.fd, buf, sizeof(buf) - 1, 0);
    if (n <= 0) {
        return 0;
    }
    buf[n] = 0;
    const char * p = strstr(buf, "\nread_bytes: ");
    return p ? strtoull(p + 13, nullptr, 10) : 0;
#else
    return 0;
#endif
}

// HIGH loads of this thread, so that a LOW load that drains them in between can exclude their bytes
static thread_local uint64_t t_nested_bytes = 0;
// simulated read_bytes of this thread (params::test_load_bytes)
static thread_local uint64_t t_sim_bytes = 0;

static uint32_t fnv1a(const uint8_t * p, size_t n) {
    uint32_t h = 2166136261u;
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 16777619u;
    }
    return h;
}

template <typename T>
static void put(std::vector<uint8_t> & out, const T & v) {
    const uint8_t * p = (const uint8_t *) &v;
    out.insert(out.end(), p, p + sizeof(T));
}

template <typename T>
static bool get(const std::vector<uint8_t> & in, size_t & off, T & v) {
    if (off + sizeof(T) > in.size()) {
        return false;
    }
    memcpy(&v, in.data() + off, sizeof(T));
    off += sizeof(T);
    return true;
}

static int find_rank(const std::vector<int> & v, int e) {
    for (size_t i = 0; i < v.size(); ++i) {
        if (v[i] == e) {
            return (int) i;
        }
    }
    return -1;
}

llama_moe_predictor::llama_moe_predictor(const params & p, std::vector<layer_tensors> layers) :
        par(p), lt(std::move(layers)) {
    par.n_workers = std::max(1, std::min(par.n_workers, 2));
    par.cap       = std::max(MOE_PRED_NR_MIN, par.cap);

    ls.resize(par.n_layer);
    uds.resize(par.n_layer);
    for (int il = 0; il < par.n_layer; ++il) {
        if (lt[il].exps[0] != nullptr) {
            ls[il].pop.assign(par.n_expert, 0.0f);
            if (first_moe < 0) {
                first_moe = il;
            }
        }
    }
    decay = std::pow(0.5f, 1.0f/64.0f); // popularity half-life: 64 tokens
    scratch_mark.assign(par.n_expert, 0);
    scratch_score.assign(par.n_expert, 0.0f);
    scratch_rank_r.assign(par.n_expert, -1);
    scratch_rank_p.assign(par.n_expert, -1);

    if (par.mode == MODE_PREFETCH) {
        const size_t n = (size_t) par.n_layer*par.n_expert;
        pred_tag   .reset(new std::atomic<uint64_t>[n]());
        pred_bytes .reset(new std::atomic<uint64_t>[n]());
        pend_bytes .reset(new std::atomic<uint64_t>[n]());
        pend_clock .reset(new std::atomic<uint64_t>[n]());
        pend_cached.reset(new std::atomic<uint64_t>[n]());
    }

#if defined(LLAMA_MOE_PREDICT_MADVISE)
    const long pg = sysconf(_SC_PAGESIZE);
    page_size = pg > 0 ? (size_t) pg : 4096;
#endif
    t_start_us = ggml_time_us();
    paused     = par.paused;

    if (!par.stats_path.empty()) {
        load();
    }

    // the worker hints in prefetch mode, and saves in both modes
    const int n_workers = par.mode == MODE_PREFETCH ? par.n_workers : 1;
    for (int i = 0; i < n_workers; ++i) {
        workers.emplace_back([this] { worker_loop(); });
    }

    LLAMA_LOG_INFO("%s: MoE expert prediction %s, %d workers, slots cap %d, grow %.2f, shrink %.2f, stats %s, residency by %s\n",
            __func__, par.mode == MODE_PREFETCH ? "prefetch" : "observe", n_workers, par.cap, par.grow, par.shrink,
            par.stats_path.empty() ? "(none)" : par.stats_path.c_str(), par.mincore_ok ? "mincore" : "load deltas");
}

llama_moe_predictor::~llama_moe_predictor() {
    {
        std::lock_guard<std::mutex> lock(qmtx);
        stop = true;
    }
    qcv.notify_all();
    for (auto & t : workers) {
        t.join();
    }
    if (!par.stats_path.empty()) {
        save();
    }
}

llama_moe_predictor::hint_ud * llama_moe_predictor::get_hint_ud(int il, int il_next) {
    std::lock_guard<std::mutex> lock(mtx);
    if (!uds[il]) {
        uds[il] = std::make_unique<hint_ud>(hint_ud { this, il, il_next });
    }
    return uds[il].get();
}

void llama_moe_predictor::rank_router(const float * logits, const float * bias, int n_expert, int n, std::vector<int> & out) {
    std::vector<std::pair<float, int>> s(n_expert);
    for (int e = 0; e < n_expert; ++e) {
        s[e] = { 1.0f/(1.0f + std::exp(-logits[e])) + (bias ? bias[e] : 0.0f), e };
    }
    n = std::min(n, n_expert);
    std::partial_sort(s.begin(), s.begin() + n, s.end(), [](const auto & a, const auto & b) {
        return a.first > b.first || (a.first == b.first && a.second < b.second);
    });
    out.resize(n);
    for (int i = 0; i < n; ++i) {
        out[i] = s[i].second;
    }
}

void llama_moe_predictor::on_layer(int il, const int32_t * sel, size_t sel_nb0, size_t sel_nb1, int n_tok,
        int il_next, const float * pred, size_t pred_nb1, const float * bias) {
    if (il < 0 || il >= par.n_layer || ls[il].pop.empty() || n_tok <= 0) {
        return;
    }

    std::vector<int> act;
    std::vector<int> guess;
    bool do_save = false;

    {
        std::lock_guard<std::mutex> lock(mtx);

        // servers run a 2-token probe decode at context creation (common_context_can_seq_rm) before any request:
        // small multi-token steps are ignored until a single-token step shows real decoding has started
        if (n_tok == 1) {
            unlocked = true;
        } else if (n_tok <= 4 && !unlocked) {
            return;
        }

        if (il == first_moe) {
            step++;
        }
        cur_layer = il;

        // actual selection of this layer, union over the tokens
        for (int t = 0; t < n_tok; ++t) {
            for (int k = 0; k < par.n_used; ++k) {
                const int e = *(const int32_t *) ((const char *) sel + t*sel_nb1 + k*sel_nb0);
                if (e >= 0 && e < par.n_expert && !scratch_mark[e]) {
                    scratch_mark[e] = 1;
                    act.push_back(e);
                }
            }
        }
        for (int e : act) {
            scratch_mark[e] = 0;
        }

        // score the prediction that was made for this layer
        layer_state & s = ls[il];
        if (s.has_pred) {
            // ranks of the guessed experts (pred_r is a prefix of pred_r48)
            std::vector<int> & rank_r = scratch_rank_r;
            std::vector<int> & rank_p = scratch_rank_p;
            for (size_t i = 0; i < s.pred_r48.size(); ++i) {
                rank_r[s.pred_r48[i]] = (int) i;
            }
            for (size_t i = 0; i < s.pred_p.size(); ++i) {
                rank_p[s.pred_p[i]] = (int) i;
            }
            const int n_pred_r = (int) s.pred_r.size();

            // counterfactual: router top T-4 plus the 4 most popular experts that are not among them
            static const int cf_n[4] = { 16, 24, 32, 48 };
            int mix[3][4];
            for (int i = 0; i < 3; ++i) {
                const int T = cf_n[i];
                int taken = 0;
                for (int c : s.pred_p_all) {
                    if (taken == 4) {
                        break;
                    }
                    if (rank_r[c] >= 0 && rank_r[c] < T - 4) {
                        continue;
                    }
                    mix[i][taken++] = c;
                }
                for (; taken < 4; ++taken) {
                    mix[i][taken] = -1;
                }
            }

            uint64_t hits = 0;
            for (int e : act) {
                const int rr = rank_r[e] < n_pred_r ? rank_r[e] : -1;
                const int rp = rr >= 0 && rr < s.n_r ? -1 : rank_p[e];
                if (rr >= 0 && rr < s.n_r) {
                    hits++;
                    s.hit_r_outer += rr >= s.n_r - 4;
                } else if (rr >= s.n_r) {
                    s.hit_r_next++;
                }
                if (rp >= 0 && rp < s.n_p) {
                    hits++;
                    s.hit_p_outer += rp >= s.n_p - 4;
                } else if (rp >= s.n_p) {
                    s.hit_p_next++;
                }

                const int r48 = rank_r[e];
                for (int i = 0; i < 4; ++i) {
                    s.cf_hits[i] += r48 >= 0 && r48 < cf_n[i];
                }
                for (int i = 0; i < 3; ++i) {
                    bool hit = r48 >= 0 && r48 < cf_n[i] - 4;
                    for (int k = 0; k < 4 && !hit; ++k) {
                        hit = mix[i][k] == e;
                    }
                    s.cf_mix_hits[i] += hit;
                }
            }

            for (int e : s.pred_r48) {
                rank_r[e] = -1;
            }
            for (int e : s.pred_p) {
                rank_p[e] = -1;
            }

            s.hits   += hits;
            s.actual += act.size();
            n_hits    += hits;
            n_actual  += act.size();
            n_guesses += s.guess.size();

            // the bytes read for the guesses: used now, or pending until the expert is used or evicted
            if (pred_tag) {
                for (int e : act) {
                    scratch_mark[e] = 1;
                }
                const uint64_t clk = io_clock.load(std::memory_order_relaxed);
                const uint64_t cch = io_cached.load(std::memory_order_relaxed);
                for (int e : s.guess) {
                    const size_t i = (size_t) il*par.n_expert + e;
                    pred_tag[i].store(0, std::memory_order_relaxed); // loads that finish later are late
                    const uint64_t b = pred_bytes[i].exchange(0, std::memory_order_relaxed);
                    if (b == 0) {
                        continue;
                    }
                    if (scratch_mark[e]) {
                        bytes_hit += b;
                    } else {
                        bytes_pending += b;
                        pend_clock [i].store(clk, std::memory_order_relaxed);
                        pend_cached[i].store(cch, std::memory_order_relaxed);
                        pend_bytes [i] += b;
                    }
                }
                for (int e : act) {
                    scratch_mark[e] = 0;
                }
            }
            s.wasted += (uint64_t) std::max<int64_t>(0, (int64_t) (s.n_r + s.n_p) - (int64_t) hits);
            s.tokens += n_tok;
            s.n_eval += n_tok;
            // each prediction is scored once (a prompt batch in between leaves no stale guess behind)
            s.has_pred = false;
            if (s.n_eval >= MOE_PRED_EVAL) {
                adapt(s);
            }
        }

        // an earlier guess that was not used is used now: credited if its pages are still resident
        // (before the expert matmuls of this layer fault them in); without a trusted mincore the HIGH load decides
        if (pred_tag && (par.mincore_ok || par.test_resident)) {
            int n_probe = 0;
            for (int e : act) {
                const size_t i = (size_t) il*par.n_expert + e;
                if (n_probe == MOE_PRED_NPROBE) {
                    break;
                }
                if (pend_bytes[i].load(std::memory_order_relaxed) == 0) {
                    continue;
                }
                n_probe++;
                n_probes++;
                resolve_pending(i, resident(il, e));
            }
        }

        // popularity
        const float d = std::pow(decay, (float) n_tok);
        for (float & v : s.pop) {
            v *= d;
        }
        for (int e : act) {
            s.pop[e] += 1.0f;
        }

        // predict the next layer
        if (il_next >= 0 && il_next < par.n_layer && pred != nullptr && !ls[il_next].pop.empty()) {
            layer_state & nx = ls[il_next];

            // a guess for several tokens: the best score of any of them
            std::vector<float> & best = scratch_score;
            std::fill(best.begin(), best.end(), -INFINITY);
            for (int t = 0; t < n_tok; ++t) {
                const float * l = (const float *) ((const char *) pred + t*pred_nb1);
                for (int e = 0; e < par.n_expert; ++e) {
                    best[e] = std::max(best[e], l[e]);
                }
            }
            rank_router(best.data(), bias, par.n_expert, std::max(48, nx.n_r + 4), nx.pred_r48);
            nx.pred_r.assign(nx.pred_r48.begin(), nx.pred_r48.begin() + std::min<size_t>(nx.pred_r48.size(), nx.n_r + 4));

            // popularity ranking: overall, and excluding the router's chosen slots
            std::vector<std::pair<float, int>> pr(par.n_expert);
            for (int e = 0; e < par.n_expert; ++e) {
                pr[e] = { nx.pop[e], e };
            }
            const int n_pr = std::min<int>(par.n_expert, par.cap + 52);
            std::partial_sort(pr.begin(), pr.begin() + n_pr, pr.end(), [](const auto & a, const auto & b) {
                return a.first > b.first || (a.first == b.first && a.second < b.second);
            });
            nx.pred_p_all.clear();
            nx.pred_p.clear();
            for (int i = 0; i < n_pr; ++i) {
                const int e = pr[i].second;
                if ((int) nx.pred_p_all.size() < 52) {
                    nx.pred_p_all.push_back(e);
                }
                const int rr = find_rank(nx.pred_r, e);
                if ((rr < 0 || rr >= nx.n_r) && (int) nx.pred_p.size() < nx.n_p + 4) {
                    nx.pred_p.push_back(e);
                }
            }
            nx.has_pred = true;

            guess.assign(nx.pred_r.begin(), nx.pred_r.begin() + std::min<size_t>(nx.pred_r.size(), nx.n_r));
            guess.insert(guess.end(), nx.pred_p.begin(), nx.pred_p.begin() + std::min<size_t>(nx.pred_p.size(), nx.n_p));
            nx.guess = guess;

            if (pred_tag) {
                for (int e : guess) {
                    const size_t i = (size_t) il_next*par.n_expert + e;
                    // bytes of an older guess that landed after its scoring
                    bytes_late += pred_bytes[i].exchange(0, std::memory_order_relaxed);
                    pred_tag[i].store(step + 1, std::memory_order_relaxed);
                }
            }
        }

        if (il == first_moe) {
            if (step - last_save >= MOE_PRED_SAVE && !par.stats_path.empty()) {
                last_save = step;
                do_save   = true;
            }
            if (step - last_log >= MOE_PRED_LOG) {
                last_log = step;
                log_stats_locked();
            }
        }
    }

    step_seen.store(step, std::memory_order_relaxed);
    layer_seen.store(il,  std::memory_order_relaxed);

    if (par.mode == MODE_PREFETCH) {
        enqueue(true, il, act);
        if (!guess.empty()) {
            enqueue(false, il_next, guess);
        }
    }
    if (do_save) {
        {
            std::lock_guard<std::mutex> lock(qmtx);
            want_save = true;
        }
        qcv.notify_one();
    }
}

void llama_moe_predictor::adapt(layer_state & s) {
    const double slots = 4.0*s.n_eval;
    const double u_r_out  = s.hit_r_outer/slots;
    const double u_r_next = s.hit_r_next /slots;
    const double u_p_out  = s.n_p >= 4 ? s.hit_p_outer/slots : 1.0;
    const double u_p_next = s.hit_p_next /slots;

    const bool room = s.n_r + s.n_p + 4 <= par.cap;
    if (room && u_r_next >= par.grow && u_r_next >= u_p_next) {
        s.n_r += 4;
    } else if (room && u_p_next >= par.grow) {
        s.n_p += 4;
    } else {
        if (s.n_r - 4 >= MOE_PRED_NR_MIN && u_r_out < par.shrink) {
            s.n_r -= 4;
        }
        if (s.n_p >= 4 && u_p_out < par.shrink) {
            s.n_p -= 4;
        }
    }

    s.hit_r_outer = s.hit_r_next = s.hit_p_outer = s.hit_p_next = 0;
    s.n_eval = 0;
}

void llama_moe_predictor::log_stats_locked() {
    uint64_t hits = 0, actual = 0, wasted = 0, tokens = 0;
    uint64_t cf[4] = {}, cfm[3] = {};
    int nr_min = 1 << 30, nr_max = 0, np_min = 1 << 30, np_max = 0;
    for (const auto & s : ls) {
        if (s.pop.empty()) {
            continue;
        }
        hits += s.hits; actual += s.actual; wasted += s.wasted; tokens = std::max(tokens, s.tokens);
        for (int i = 0; i < 4; ++i) { cf[i]  += s.cf_hits[i]; }
        for (int i = 0; i < 3; ++i) { cfm[i] += s.cf_mix_hits[i]; }
        nr_min = std::min(nr_min, s.n_r); nr_max = std::max(nr_max, s.n_r);
        np_min = std::min(np_min, s.n_p); np_max = std::max(np_max, s.n_p);
    }
    if (actual == 0) {
        return;
    }
    const double a = (double) actual;
    const double elapsed = (double) std::max<int64_t>(1, ggml_time_us() - t_start_us);
    const uint64_t hn = high_n.load();
    LLAMA_LOG_INFO("%s: step %llu: recall %.3f (router top16/24/32/48 %.3f/%.3f/%.3f/%.3f, +pop4 at 16/24/32 %.3f/%.3f/%.3f), "
            "n_r %d..%d, n_p %d..%d, unused guesses %.1f/token, worker busy %.0f%%, high wait %.1f ms, low drops %llu\n",
            __func__, (unsigned long long) step, hits/a, cf[0]/a, cf[1]/a, cf[2]/a, cf[3]/a, cfm[0]/a, cfm[1]/a, cfm[2]/a,
            nr_min, nr_max, np_min, np_max, tokens ? (double) wasted/tokens : 0.0,
            100.0*busy_us.load()/elapsed, hn ? high_wait_us.load()/1000.0/hn : 0.0, (unsigned long long) low_drops.load());
}

llama_moe_predictor::layer_view llama_moe_predictor::view(int il) const {
    std::lock_guard<std::mutex> lock(mtx);
    const auto & s = ls[il];
    return { s.n_r, s.n_p, s.actual ? (double) s.hits/s.actual : 0.0, s.tokens };
}

//
// prefetch worker
//

void llama_moe_predictor::enqueue(bool high, int il, const std::vector<int> & experts) {
    const int64_t t = ggml_time_us();
    {
        std::lock_guard<std::mutex> lock(qmtx);
        for (int e : experts) {
            if (high) {
                q_high.push_back({ il, e, step_seen.load(std::memory_order_relaxed), t });
                n_high_pending++;
            } else if (q_low.size() < MOE_PRED_QMAX) {
                q_low.push_back({ il, e, step_seen.load(std::memory_order_relaxed), t });
            } else {
                low_drops++;
            }
        }
    }
    qcv.notify_one();
}

bool llama_moe_predictor::pop_high(work & w) {
    std::lock_guard<std::mutex> lock(qmtx);
    if (q_high.empty()) {
        return false;
    }
    w = q_high.front();
    q_high.pop_front();
    n_high_pending--;
    return true;
}

void llama_moe_predictor::drain_high() {
    work w;
    while (pop_high(w)) {
        // the layer has already run (its matmuls requested the pages themselves): nothing left to gain
        const uint64_t s = step_seen.load(std::memory_order_relaxed);
        const int      l = layer_seen.load(std::memory_order_relaxed);
        if (w.step < s || (w.step == s && w.il < l)) {
            high_drops++;
            continue;
        }
        high_wait_us += ggml_time_us() - w.t_us;
        high_n++;
        if (par.trace) {
            std::lock_guard<std::mutex> lock(qmtx);
            trace.push_back(w.il + 1);
        }
        prefetch_expert(w.il, w.e, false, w.step);
    }
}

llama_moe_predictor::counters llama_moe_predictor::get_counters() const {
    return { high_n.load(), low_done.load(), low_drops.load(), high_drops.load() };
}

void llama_moe_predictor::pause() {
    std::lock_guard<std::mutex> lock(qmtx);
    paused = true;
}

void llama_moe_predictor::resume() {
    {
        std::lock_guard<std::mutex> lock(qmtx);
        paused = false;
    }
    qcv.notify_all();
}

void llama_moe_predictor::wait_idle() {
    std::unique_lock<std::mutex> lock(qmtx);
    idle_cv.wait(lock, [this] { return q_high.empty() && q_low.empty() && in_progress == 0; });
}

std::vector<int> llama_moe_predictor::get_trace() {
    std::lock_guard<std::mutex> lock(qmtx);
    return trace;
}

void llama_moe_predictor::prefetch_expert(int il, int e, bool low, uint64_t step) {
    // the bytes this load reads from disk: the read_bytes delta of this thread, without the HIGH loads drained in between
    const bool     account = pred_tag != nullptr;
    const uint64_t io0     = account ? load_bytes(il, e, !low) : 0;
    const uint64_t nested0 = t_nested_bytes;
#if defined(LLAMA_MOE_PREDICT_MADVISE)
    for (const ggml_tensor * t : lt[il].exps) {
        if (t == nullptr || e >= t->ne[2]) {
            continue;
        }
        const size_t nb02 = t->nb[2];
        const uintptr_t slice = (uintptr_t) t->data + (uintptr_t) e*nb02;
        for (size_t off = 0; off < nb02; off += MOE_PRED_PIECE) {
            const uintptr_t p   = slice + off;
            const uintptr_t n   = std::min(MOE_PRED_PIECE, nb02 - off);
            const uintptr_t beg = p & ~(uintptr_t) (page_size - 1);
            const uintptr_t end = (p + n + page_size - 1) & ~(uintptr_t) (page_size - 1);
            (void) posix_madvise((void *) beg, end - beg, POSIX_MADV_WILLNEED);
            // the experts the current layer needs come before the guesses for the next one
            if (low && n_high_pending.load(std::memory_order_relaxed) > 0) {
                drain_high();
            }
        }
    }
#endif
    if (!account) {
        return;
    }
    const uint64_t io1    = par.test_load_bytes ? t_sim_bytes : thread_read_bytes();
    const uint64_t nested = t_nested_bytes - nested0;
    const uint64_t d      = io1 > io0 ? io1 - io0 : 0;
    const uint64_t b      = d > nested ? d - nested : 0;
    if (!low) {
        t_nested_bytes += b;
    }
    account_load(il, e, low, step, b);
}

// the read_bytes of this thread before a load (tests: the simulated load is done here)
uint64_t llama_moe_predictor::load_bytes(int il, int e, bool high) {
    if (par.test_load_bytes) {
        const uint64_t before = t_sim_bytes;
        t_sim_bytes += par.test_load_bytes(il, e, high);
        return before;
    }
    return thread_read_bytes();
}

void llama_moe_predictor::account_load(int il, int e, bool low, uint64_t step, uint64_t b) {
    const size_t i = (size_t) il*par.n_expert + e;
    if (!low) {
        // the current layer needs it: never late nor wasted
        bytes_actual += b;
        if (!par.mincore_ok && !par.test_resident && pend_bytes[i].load(std::memory_order_relaxed) > 0) {
            resolve_pending(i, b == 0);
        }
        return;
    }
    if (pred_tag[i].load(std::memory_order_relaxed) == step + 1) {
        pred_bytes[i] += b;
        // scored in between: the bytes are not in its count, so they are late
        if (pred_tag[i].load(std::memory_order_relaxed) != step + 1) {
            bytes_late += pred_bytes[i].exchange(0, std::memory_order_relaxed);
        }
    } else {
        bytes_late += b;
    }
    bytes_read_predictor += b;
    // guessed again while an earlier unused read is pending: nothing read means its pages are still there
    if (pend_bytes[i].load(std::memory_order_relaxed) > 0) {
        if (b == 0) {
            pend_clock [i].store(io_clock .load(std::memory_order_relaxed), std::memory_order_relaxed);
            pend_cached[i].store(io_cached.load(std::memory_order_relaxed), std::memory_order_relaxed);
        } else {
            resolve_pending(i, false);
        }
    }
}

void llama_moe_predictor::resolve_pending(size_t i, bool used) {
    const uint64_t p = pend_bytes[i].exchange(0, std::memory_order_relaxed);
    if (p == 0) {
        return;
    }
    if (used) {
        bytes_hit      += p;
        bytes_credited += p;
    } else {
        bytes_wasted += p;
    }
    bytes_pending -= p;
}

// sampled residency of an expert: the middle MOE_PRED_PROBE bytes of each of its tensors, resident if half the pages are
bool llama_moe_predictor::resident(int il, int e) {
    if (par.test_resident) {
        return par.test_resident(il, e);
    }
#if defined(__linux__)
    thread_local std::vector<unsigned char> vec;
    size_t n_res = 0;
    size_t n_all = 0;
    for (const ggml_tensor * t : lt[il].exps) {
        if (t == nullptr || e >= t->ne[2]) {
            continue;
        }
        const size_t nb02 = t->nb[2];
        const size_t len  = std::min(MOE_PRED_PROBE, nb02);
        const size_t pos  = (nb02 - len)/2;
        const uintptr_t p   = (uintptr_t) t->data + (uintptr_t) e*nb02 + pos;
        const uintptr_t beg = p & ~(uintptr_t) (page_size - 1);
        const uintptr_t end = (p + len + page_size - 1) & ~(uintptr_t) (page_size - 1);
        vec.resize((end - beg)/page_size);
        if (mincore((void *) beg, end - beg, vec.data()) != 0) {
            continue;
        }
        for (unsigned char v : vec) {
            n_res += v & 1;
        }
        n_all += vec.size();
    }
    return n_all > 0 && 2*n_res >= n_all;
#else
    GGML_UNUSED(il);
    GGML_UNUSED(e);
    return false;
#endif
}

llama_moe_predictor::stats llama_moe_predictor::get_stats() const {
    const auto ld = [](const std::atomic<uint64_t> & a) { return a.load(std::memory_order_relaxed); };
    return {
        par.mode, layer_seen.load(std::memory_order_relaxed), ld(step_seen),
        ld(n_hits), ld(n_actual), ld(n_guesses),
        ld(bytes_hit), ld(bytes_wasted), ld(bytes_late), ld(bytes_actual), ld(bytes_read_predictor),
        ld(bytes_pending), ld(bytes_credited), ld(n_probes),
    };
}

void llama_moe_predictor::set_io_clock(uint64_t read_bytes, uint64_t cached_bytes) {
    // pending reads stamped while nobody published the clock may be older than they look: count from now
    const int64_t t = ggml_time_us();
    if (io_last_us == 0 || t - io_last_us > MOE_PRED_IO_GAP) {
        io_floor = read_bytes;
    }
    io_last_us = t;
    io_clock .store(read_bytes,   std::memory_order_relaxed);
    io_cached.store(cached_bytes, std::memory_order_relaxed);

    if (!pred_tag || cached_bytes == 0) {
        return;
    }
    // the whole page cache was read over since the pending read: its pages are gone
    const size_t n = (size_t) par.n_layer*par.n_expert;
    for (size_t i = 0; i < n; ++i) {
        if (pend_bytes[i].load(std::memory_order_relaxed) == 0) {
            continue;
        }
        const uint64_t clk = std::max(pend_clock[i].load(std::memory_order_relaxed), io_floor);
        const uint64_t pc  = pend_cached[i].load(std::memory_order_relaxed);
        const uint64_t lim = pc > 0 ? std::min(pc, cached_bytes) : cached_bytes;
        if (read_bytes > clk && read_bytes - clk > lim) {
            resolve_pending(i, false);
        }
    }
}

bool llama_moe_predictor::mincore_trusted(const std::vector<std::string> & paths) {
#if defined(__linux__)
    // mincore reports every page as resident for a file the caller neither owns nor may write
    if (paths.empty()) {
        return false;
    }
    for (const auto & path : paths) {
        struct stat st{};
        if (stat(path.c_str(), &st) != 0 || (st.st_uid != geteuid() && access(path.c_str(), W_OK) != 0)) {
            return false;
        }
    }
    return true;
#else
    GGML_UNUSED(paths);
    return false;
#endif
}

void llama_moe_predictor::worker_loop() {
    for (;;) {
        work w;
        bool have_low = false;
        bool do_save  = false;
        {
            std::unique_lock<std::mutex> lock(qmtx);
            qcv.wait(lock, [this] { return stop || (!paused && (want_save || !q_high.empty() || !q_low.empty())); });
            if (stop) {
                return;
            }
            // save only when no prefetch work waits: the fsync must not hold it up
            if (want_save && q_high.empty() && q_low.empty()) {
                want_save = false;
                do_save   = true;
            } else if (q_high.empty() && !q_low.empty()) {
                w = q_low.front();
                q_low.pop_front();
                have_low = true;
            }
            in_progress++;
        }

        const int64_t t0 = ggml_time_us();
        if (do_save) {
            save();
        } else if (have_low) {
            // a guess for a layer that has already started (or an older step) is useless now
            const uint64_t s = step_seen.load(std::memory_order_relaxed);
            const int      l = layer_seen.load(std::memory_order_relaxed);
            if (w.step < s || (w.step == s && w.il <= l)) {
                low_drops++;
            } else {
                drain_high();
                if (par.trace) {
                    std::lock_guard<std::mutex> lock(qmtx);
                    trace.push_back(-(w.il + 1));
                }
                prefetch_expert(w.il, w.e, true, w.step);
                low_done++;
            }
        } else {
            drain_high();
        }
        busy_us += ggml_time_us() - t0;

        {
            std::lock_guard<std::mutex> lock(qmtx);
            in_progress--;
            if (q_high.empty() && q_low.empty() && in_progress == 0) {
                idle_cv.notify_all();
            }
        }
    }
}

//
// persistence
//

void llama_moe_predictor::serialize_locked(std::vector<uint8_t> & out) const {
    out.clear();
    put(out, MOE_PRED_MAGIC);
    put(out, MOE_PRED_VERSION);
    put(out, (uint32_t) par.n_layer);
    put(out, (uint32_t) par.n_expert);
    put(out, (uint64_t) par.model_id);
    for (const auto & s : ls) {
        const uint8_t moe = s.pop.empty() ? 0 : 1;
        put(out, moe);
        if (moe) {
            put(out, (int32_t) s.n_r);
            put(out, (int32_t) s.n_p);
            const uint8_t * p = (const uint8_t *) s.pop.data();
            out.insert(out.end(), p, p + s.pop.size()*sizeof(float));
        }
    }
}

uint32_t llama_moe_predictor::checksum() const {
    std::vector<uint8_t> buf;
    {
        std::lock_guard<std::mutex> lock(mtx);
        serialize_locked(buf);
    }
    return fnv1a(buf.data(), buf.size());
}

bool llama_moe_predictor::save() {
    std::vector<uint8_t> buf;
    {
        std::lock_guard<std::mutex> lock(mtx);
        serialize_locked(buf);
    }
    const uint32_t sum = fnv1a(buf.data(), buf.size());
    put(buf, sum);

    std::error_code ec;
    const std::filesystem::path path(par.stats_path);
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path(), ec);
    }
    const std::string tmp = par.stats_path + ".tmp";
    FILE * f = fopen(tmp.c_str(), "wb");
    if (!f) {
        LLAMA_LOG_WARN("%s: cannot write %s: %s\n", __func__, tmp.c_str(), strerror(errno));
        return false;
    }
    bool ok = fwrite(buf.data(), 1, buf.size(), f) == buf.size() && fflush(f) == 0;
#if defined(LLAMA_MOE_PREDICT_MADVISE)
    ok = ok && fsync(fileno(f)) == 0;
#endif
    ok = fclose(f) == 0 && ok;
    if (!ok || std::rename(tmp.c_str(), par.stats_path.c_str()) != 0) {
        LLAMA_LOG_WARN("%s: failed to save %s\n", __func__, par.stats_path.c_str());
        std::remove(tmp.c_str());
        return false;
    }
    LLAMA_LOG_DEBUG("%s: saved %s, checksum %08x\n", __func__, par.stats_path.c_str(), sum);
    return true;
}

bool llama_moe_predictor::load() {
    FILE * f = fopen(par.stats_path.c_str(), "rb");
    if (!f) {
        LLAMA_LOG_INFO("%s: no saved state at %s, starting cold\n", __func__, par.stats_path.c_str());
        return false;
    }
    std::vector<uint8_t> buf;
    uint8_t chunk[65536];
    size_t n;
    while ((n = fread(chunk, 1, sizeof(chunk), f)) > 0) {
        buf.insert(buf.end(), chunk, chunk + n);
    }
    fclose(f);

    auto reject = [&](const char * why) {
        LLAMA_LOG_WARN("llama_moe_predictor::load: ignoring %s (%s), starting cold\n", par.stats_path.c_str(), why);
        return false;
    };

    if (buf.size() < sizeof(uint32_t)) {
        return reject("too short");
    }
    uint32_t sum;
    memcpy(&sum, buf.data() + buf.size() - sizeof(uint32_t), sizeof(uint32_t));
    if (fnv1a(buf.data(), buf.size() - sizeof(uint32_t)) != sum) {
        return reject("checksum mismatch");
    }

    size_t off = 0;
    uint32_t magic, version, n_layer, n_expert;
    uint64_t model_id;
    if (!get(buf, off, magic) || !get(buf, off, version) || !get(buf, off, n_layer) || !get(buf, off, n_expert) ||
        !get(buf, off, model_id)) {
        return reject("truncated header");
    }
    if (magic != MOE_PRED_MAGIC || version != MOE_PRED_VERSION) {
        return reject("unknown format");
    }
    if ((int) n_layer != par.n_layer || (int) n_expert != par.n_expert || model_id != par.model_id) {
        return reject("different model");
    }

    std::vector<layer_state> tmp(ls.size());
    for (size_t il = 0; il < ls.size(); ++il) {
        uint8_t moe;
        if (!get(buf, off, moe) || moe != (ls[il].pop.empty() ? 0 : 1)) {
            return reject("layer layout mismatch");
        }
        if (!moe) {
            continue;
        }
        int32_t n_r, n_p;
        if (!get(buf, off, n_r) || !get(buf, off, n_p) || off + par.n_expert*sizeof(float) > buf.size() - sizeof(uint32_t)) {
            return reject("truncated layer");
        }
        tmp[il].n_r = std::max(MOE_PRED_NR_MIN, std::min<int>(n_r, par.cap));
        tmp[il].n_p = std::max(0, std::min<int>(n_p, par.cap - tmp[il].n_r));
        tmp[il].pop.resize(par.n_expert);
        memcpy(tmp[il].pop.data(), buf.data() + off, par.n_expert*sizeof(float));
        off += par.n_expert*sizeof(float);
    }
    if (off != buf.size() - sizeof(uint32_t)) {
        return reject("trailing data");
    }

    {
        std::lock_guard<std::mutex> lock(mtx);
        for (size_t il = 0; il < ls.size(); ++il) {
            if (!tmp[il].pop.empty()) {
                ls[il].n_r = tmp[il].n_r;
                ls[il].n_p = tmp[il].n_p;
                ls[il].pop = std::move(tmp[il].pop);
            }
        }
        loaded_ok = true;
    }
    LLAMA_LOG_INFO("%s: loaded %s, checksum %08x\n", __func__, par.stats_path.c_str(), checksum());
    return true;
}

//
// graph op
//

void llama_moe_predictor_hint_op(struct ggml_tensor * dst, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }
    const auto * ud = (const llama_moe_predictor::hint_ud *) userdata;

    const ggml_tensor * sel  = dst->src[1];
    const ggml_tensor * pl   = dst->src[2];
    const ggml_tensor * bias = dst->src[3];

    GGML_ASSERT(sel->type == GGML_TYPE_I32 && sel->ne[0] == ud->pred->n_used());

    ud->pred->on_layer(ud->il, (const int32_t *) sel->data, sel->nb[0], sel->nb[1], (int) sel->ne[1],
            pl ? ud->il_next : -1, pl ? (const float *) pl->data : nullptr, pl ? pl->nb[1] : 0,
            bias ? (const float *) bias->data : nullptr);
}
