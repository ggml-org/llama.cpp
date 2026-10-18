#include "server-perf.h"

#include <algorithm>
#include <cstdint>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(__linux__)
#include <dlfcn.h>
#include <limits.h>
#include <pthread.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>
#endif

static constexpr int64_t PERF_TICK_US     = 1000000; // sample period while polled
static constexpr int64_t PERF_FIRST_US    =  500000; // second sample after a wake
static constexpr int64_t PERF_ACTIVE_US   = 10000000; // keep sampling this long after the last poll
static constexpr int64_t PERF_STALE_US    = 2500000; // older samples belong to an earlier active period
static constexpr int64_t PERF_FRESH_US    = 1500000; // a poll is answered at once with samples this recent
static constexpr int64_t PERF_WAIT_US     = 1400000; // longest wait of a poll for fresh samples
static constexpr int64_t PERF_WINDOW_US   = 60000000;
static constexpr int64_t PERF_SMOOTH_US   = 10000000;
static constexpr size_t  PERF_RING        = 61;      // 60 s of 1 s intervals

//
// server_perf_counters
//

static constexpr size_t  PERF_HIST_MAX    = 64;      // batches kept for the cost model

std::vector<int32_t> server_prompt_plan(const server_prompt_plan_params & p) {
    std::vector<int32_t> out;
    if (p.n_batch <= 0) {
        return out;
    }
    const auto is_user = [&](int32_t pos) {
        return std::binary_search(p.user_starts.begin(), p.user_starts.end(), pos);
    };
    int32_t pos   = p.pos;
    int32_t last  = p.last_ckpt;
    bool    empty = p.mem_empty;
    while (pos < p.n_total) {
        const int32_t start = pos;
        int32_t n = 0;
        while (pos < p.n_total && n < p.n_batch) {
            pos++;
            n++;
            // the batch ends before a user message that gets a checkpoint
            if (p.checkpoints && is_user(pos) && (pos == p.last_user_pos || last < 0 || pos > last + p.min_step)) {
                break;
            }
            // and that many tokens before the end of the prompt
            if (p.checkpoints && (p.n_total == pos + std::min(p.n_batch, 4 + p.n_ubatch) || p.n_total == pos + std::min(p.n_batch, 4))) {
                break;
            }
        }
        out.push_back(n);

        // the checkpoint taken before this batch is decoded
        const bool near_end = p.n_total < pos + p.n_ubatch;
        bool ckpt = p.checkpoints && !empty;
        if (pos != p.n_total && !is_user(start) && !near_end) {
            ckpt = false;
        }
        ckpt = ckpt && (last < 0 || start == p.last_user_pos || near_end || start > last + p.min_step);
        if (ckpt) {
            last = start;
        }
        empty = false;
    }
    return out;
}

static void perf_plan_archive(server_perf_plan & plan) {
    plan.hist.insert(plan.hist.end(), plan.done.begin(), plan.done.end());
    if (plan.hist.size() > PERF_HIST_MAX) {
        plan.hist.erase(plan.hist.begin(), plan.hist.end() - PERF_HIST_MAX);
    }
    plan.done.clear();
    plan.remaining.clear();
    plan.cur_n = 0;
    plan.valid = true;
}

void server_perf_counters::on_prompt_plan(int32_t n_cur, std::vector<int32_t> remaining, bool valid) {
    std::lock_guard<std::mutex> lock(plan_mtx);
    plan.cur_n     = n_cur;
    plan.remaining = std::move(remaining);
    plan.valid     = plan.valid && valid; // once unknown, unknown for the rest of the request
}

void server_perf_counters::on_prompt_start(int64_t t_start_us) {
    {
        std::lock_guard<std::mutex> lock(plan_mtx);
        perf_plan_archive(plan);
    }
    cur_prompt_n.store(0, std::memory_order_relaxed);
    cur_prompt_t_start_us.store(t_start_us, std::memory_order_relaxed);
    cur_gen_n.store(0, std::memory_order_relaxed);
    generating.store(false, std::memory_order_relaxed);
    processing.store(true, std::memory_order_relaxed);
}

void server_perf_counters::on_prompt_batch(const server_slot_stats & st, uint64_t n_tokens, int64_t t_batch_us) {
    n_prompt_total.fetch_add(n_tokens, std::memory_order_relaxed);
    prompt_busy_us.fetch_add((uint64_t) std::max<int64_t>(0, t_batch_us), std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(plan_mtx);
        plan.done.push_back({ (int32_t) n_tokens, std::max<int64_t>(0, t_batch_us) });
        // a batch the server had to decode in parts (no room in the KV cache): the rest of it is still to come
        plan.cur_n = std::max<int32_t>(0, plan.cur_n - (int32_t) n_tokens);
    }
    if (st.is_set()) {
        cur_prompt_n.store(st.n_prompt_processed, std::memory_order_relaxed);
        cur_prompt_t_start_us.store(st.t_start, std::memory_order_relaxed);
    }
}

void server_perf_counters::on_gen(const server_slot_stats & st) {
    n_gen_total.fetch_add(1, std::memory_order_relaxed);
    cur_gen_n.store(st.n_gen, std::memory_order_relaxed);
    cur_gen_t_first_us.store(st.t_prompt_last, std::memory_order_relaxed);
    cur_gen_t_last_us.store(st.t_gen_last, std::memory_order_relaxed);
    // the prompt is done with the first token: its rate is final now (the totals follow at the request's end)
    if (st.n_gen == 1 && st.n_prompt_processed > 0 && st.t_prompt_last > st.t_start) {
        last_prompt_n .store(st.n_prompt_processed,         std::memory_order_relaxed);
        last_prompt_us.store(st.t_prompt_last - st.t_start, std::memory_order_relaxed);
    }
    processing.store(false, std::memory_order_relaxed);
    generating.store(true, std::memory_order_relaxed);
}

void server_perf_counters::on_request_end(const server_slot_stats & st) {
    if (st.n_gen > 0) {
        last_gen_n    .store(st.n_gen,         std::memory_order_relaxed);
        last_gen_steps.store(st.n_gen_steps(), std::memory_order_relaxed);
        last_gen_us   .store(st.t_gen_us(),    std::memory_order_relaxed);
        tot_gen_steps.fetch_add(st.n_gen_steps(), std::memory_order_relaxed);
        tot_gen_us   .fetch_add(st.t_gen_us(),    std::memory_order_relaxed);
    }
    if (st.n_prompt_processed > 0 && st.t_prompt_last > st.t_start) {
        const uint64_t t_us = st.t_prompt_last - st.t_start;
        last_prompt_n .store(st.n_prompt_processed, std::memory_order_relaxed);
        last_prompt_us.store(t_us,                  std::memory_order_relaxed);
        tot_prompt_n .fetch_add(st.n_prompt_processed, std::memory_order_relaxed);
        tot_prompt_us.fetch_add(t_us,                  std::memory_order_relaxed);
    }
    generating.store(false, std::memory_order_relaxed);
    processing.store(false, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(plan_mtx);
    perf_plan_archive(plan);
}

//
// server_perf_compute
//

static uint64_t perf_delta(uint64_t a, uint64_t b) {
    return b >= a ? b - a : 0; // a counter that went back (reload) counts as no change
}

static json perf_rate(double num, double den_us) {
    return den_us > 0.0 ? json(num/den_us*1e6) : json();
}

static json perf_ratio(uint64_t num, uint64_t den) {
    return den > 0 ? json((double) num/den) : json();
}

// index of the oldest sample not older than `span` before the newest one
static size_t perf_since(const std::deque<server_perf_tick> & ring, int64_t span) {
    const int64_t t0 = ring.back().t_us - span;
    size_t i = ring.size() - 1;
    while (i > 0 && ring[i - 1].t_us >= t0) {
        i--;
    }
    return i;
}

// projected duration (us) of the running request's prompt, from its batch plan:
// - with batch costs a + b x tokens fitted on the batches seen so far (two sizes at least), scaled by how this
//   request's finished batches compare to the fit: the running batch has (1 - fraction of its layers done) of its
//   cost left, the batches after it cost the fit
// - without a fit (a cold server): the running batch takes its elapsed time over its fraction of layers done, and
//   every full batch after it as long (one pass over the experts each)
// - batches of 4 tokens or less are decode-sized steps and cost their own average
static bool perf_prompt_projection(const server_perf_tick & t, int64_t now_us, double & total_us) {
    const auto & pl = t.plan;
    const auto & pg = t.progress;
    if (!pl.valid || (pl.done.empty() && pl.cur_n == 0 && pl.remaining.empty())) {
        return false;
    }

    // least squares over the full batches, average of the small ones
    double sn = 0, st = 0, snn = 0, snt = 0, n_big = 0, tiny = 0, n_tiny = 0;
    int32_t n_min = INT32_MAX, n_max = 0;
    double last_big = 0.0;
    const auto add = [&](const server_perf_batch & b) {
        if (b.n <= 4) {
            tiny += (double) b.us;
            n_tiny++;
            return;
        }
        sn += b.n; st += (double) b.us; snn += (double) b.n*b.n; snt += (double) b.n*b.us; n_big++;
        n_min = std::min(n_min, b.n);
        n_max = std::max(n_max, b.n);
        last_big = (double) b.us;
    };
    for (const auto & b : pl.hist) {
        add(b);
    }
    for (const auto & b : pl.done) {
        add(b);
    }
    const bool   fit = n_big >= 2 && n_max > n_min;
    double fb = fit ? (n_big*snt - sn*st)/(n_big*snn - sn*sn) : 0.0;
    fb = std::max(0.0, fb);
    const double fa = fit ? (st - fb*sn)/n_big : 0.0;
    const auto   model = [&](int32_t n) { return std::max(0.0, fa + fb*n); };
    const double t_tiny = n_tiny > 0 ? tiny/n_tiny : 0.0;

    // a running full batch without layer reports and nothing to compare it with: no basis for a projection
    const bool cur_layers = pg.n_tokens > 4 && pg.n_layer > 0 && pg.il >= 0;
    if (pl.cur_n > 4 && !fit && !cur_layers) {
        return false;
    }

    const double elapsed = (double) (now_us - t.cur_prompt_t_start_us);
    // how fast this request goes compared to the fit
    double scale = 1.0;
    if (fit) {
        double t_act = 0.0, t_fit = 0.0;
        for (const auto & b : pl.done) {
            if (b.n > 4) {
                t_act += (double) b.us;
                t_fit += model(b.n);
            }
        }
        if (t_act > 0.0 && t_fit > 0.0) {
            scale = std::clamp(t_act/t_fit, 0.5, 2.0);
        }
    }

    double cur_left = 0.0;
    double cur_tot  = 0.0; // projected duration of the running full batch, 0: unknown
    if (pl.cur_n > 0 && pg.n_tokens > 0) {
        const double cur_el = (double) std::max<int64_t>(0, now_us - pg.t_start_us);
        if (cur_layers) {
            const double f = std::clamp((pg.n_tokens_done + (double) pg.n_tokens*(pg.il + 1)/pg.n_layer)/pl.cur_n, 1e-3, 1.0);
            // the layers do not all cost the same (GPU layers, cached experts): trust the fit while few are done
            cur_tot = fit ? cur_el + (1.0 - f)*scale*model(pl.cur_n) : cur_el/f;
        } else if (pl.cur_n <= 4) {
            cur_tot = std::max(cur_el, t_tiny);
        } else if (fit) {
            cur_tot = std::max(cur_el, model(pl.cur_n));
        } else {
            cur_tot = cur_el;
        }
        cur_left = std::max(0.0, cur_tot - cur_el);
    }

    // a decode-sized batch not seen yet costs about a full pass over the used experts (logged: 4.6-14 s for 4 tokens on
    // Kimi-K3, ~0.06-0.18 of a full batch): the larger of the measured decode step and a tenth of the running batch
    const double t_step  = t.tot_gen_steps > 0 ? (double) t.tot_gen_us/t.tot_gen_steps : 0.0;
    const double t_small = n_tiny > 0 ? t_tiny : std::max(t_step, pl.cur_n > 4 && cur_tot > 0.0 ? 0.1*cur_tot : 0.0);
    double rest = 0.0;
    for (const int32_t n : pl.remaining) {
        if (n <= 4) {
            rest += t_small;
        } else if (fit) {
            rest += scale*model(n);
        } else {
            rest += pl.cur_n > 4 && cur_tot > 0.0 ? cur_tot : last_big;
        }
    }
    total_us = elapsed + cur_left + rest;
    return total_us > 0.0;
}

json server_perf_compute(const std::deque<server_perf_tick> & ring, const server_perf_baseline & base,
        const server_perf_info & info, int64_t now_us) {
    if (ring.empty()) {
        return json { { "ts_ms", nullptr }, { "window_s", PERF_WINDOW_US/1000000 } };
    }

    const server_perf_tick & last = ring.back();
    const server_perf_tick * prev = ring.size() >= 2 ? &ring[ring.size() - 2] : nullptr;
    const double dt_us = prev ? (double) (last.t_us - prev->t_us) : 0.0;
    const size_t iw    = perf_since(ring, PERF_WINDOW_US);
    const server_perf_tick & win = ring[iw];

    json res = {
        { "ts_ms",    last.ts_ms },
        { "window_s", PERF_WINDOW_US/1000000 },
    };

    // 1: CPU
    if (last.has_cpu) {
        json busy = nullptr;
        json proc = nullptr;
        if (prev && prev->has_cpu && dt_us > 0.0) {
            const uint64_t d_all  = perf_delta(prev->cpu_total, last.cpu_total);
            const uint64_t d_idle = perf_delta(prev->cpu_idle,  last.cpu_idle);
            if (d_all > 0) {
                busy = std::clamp(100.0*(double) (d_all - std::min(d_idle, d_all))/d_all, 0.0, 100.0);
            }
            if (info.n_cpu > 0 && info.clk_tck > 0) {
                const double ticks = (double) perf_delta(prev->proc_ticks, last.proc_ticks);
                proc = std::clamp(100.0*ticks/info.clk_tck/(dt_us/1e6)/info.n_cpu, 0.0, 100.0);
            }
        }
        res["cpu"] = { { "busy_pct", busy }, { "proc_pct", proc }, { "n_cpu", info.n_cpu } };
    } else {
        res["cpu"] = nullptr;
    }

    // 2, 4: GPU
    if (info.gpu_ok && last.has_gpu) {
        res["gpu"] = {
            { "busy_pct",        last.gpu_busy_pct },
            { "mem_used_bytes",  last.gpu_mem_used },
            { "mem_total_bytes", last.gpu_mem_total },
            { "name",            info.gpu_name },
        };
    } else {
        res["gpu"] = nullptr;
    }

    // 3: system memory, the page cache counted as occupied
    if (last.has_mem) {
        res["mem"] = {
            { "total_bytes",    last.mem_total },
            { "occupied_bytes", last.mem_total - std::min(last.mem_free, last.mem_total) },
            { "used_bytes",     last.mem_total - std::min(last.mem_available, last.mem_total) },
            { "cached_bytes",   last.mem_cached },
            { "free_bytes",     last.mem_free },
        };
    } else {
        res["mem"] = nullptr;
    }

    // 5: generation speed: the running request, else the last one
    {
        json live = nullptr;
        if (last.generating && last.cur_gen_n >= 2 && last.cur_gen_t_last_us > last.cur_gen_t_first_us) {
            live = perf_rate((double) (last.cur_gen_n - 1), (double) (last.cur_gen_t_last_us - last.cur_gen_t_first_us));
        }
        const json tps_last = perf_rate((double) last.last_gen_steps, (double) last.last_gen_us);

        // window: tokens per second over the seconds that produced tokens
        double n_win = 0.0;
        double t_win = 0.0;
        for (size_t i = iw + 1; i < ring.size(); ++i) {
            const uint64_t dn = perf_delta(ring[i - 1].n_gen_total, ring[i].n_gen_total);
            if (dn > 0) {
                n_win += (double) dn;
                t_win += (double) (ring[i].t_us - ring[i - 1].t_us);
            }
        }

        double n_tot = (double) last.tot_gen_steps;
        double t_tot = (double) last.tot_gen_us;
        if (last.generating && last.cur_gen_n >= 2 && last.cur_gen_t_last_us > last.cur_gen_t_first_us) {
            n_tot += (double) (last.cur_gen_n - 1);
            t_tot += (double) (last.cur_gen_t_last_us - last.cur_gen_t_first_us);
        }

        res["gen"] = {
            // generating: the running request's rate, null (the panel prints ...) until its second token
            { "tps_headline", last.generating ? live : tps_last },
            { "tps_window",   perf_rate(n_win, t_win) },
            { "tps_total",    perf_rate(n_tot, t_tot) },
            { "tps_last",     tps_last },
            { "generating",   last.generating },
            { "n_tokens",     last.generating ? last.cur_gen_n : last.last_gen_n },
        };
    }

    // 6: prompt speed: the running request including the layers done of the running batch, else the last one
    {
        const auto & pg = last.progress;
        const bool in_batch = pg.n_tokens > 4 && pg.n_layer > 0;
        json live  = nullptr;
        json layer = nullptr;
        uint64_t n_cur = last.cur_prompt_n;
        const auto & pl = last.plan;
        uint64_t n_req = 0; // the prompt tokens the request processes
        for (const auto & b : pl.done) {
            n_req += b.n;
        }
        n_req += pl.cur_n;
        for (const int32_t n : pl.remaining) {
            n_req += n;
        }
        bool estimate = false;
        if (last.processing && last.cur_prompt_t_start_us > 0) {
            double total_us = 0.0;
            if (n_req > 0 && perf_prompt_projection(last, now_us, total_us)) {
                // an estimate of the request's final rate
                live     = perf_rate((double) n_req, total_us);
                estimate = true;
            } else {
                double done = (double) last.cur_prompt_n;
                if (in_batch) {
                    done += pg.n_tokens_done + (double) pg.n_tokens*(pg.il + 1)/pg.n_layer;
                }
                live = perf_rate(done, (double) (now_us - last.cur_prompt_t_start_us));
            }
            if (in_batch) {
                n_cur += pg.n_tokens_done + pg.n_tokens;
            }
        }
        const size_t n_batches = pl.done.size() + (pl.cur_n > 0 ? 1 : 0) + pl.remaining.size();
        if (in_batch && pg.il >= 0) {
            layer = pg.il + 1;
        }
        const json tps_last = perf_rate((double) last.last_prompt_n, (double) last.last_prompt_us);

        const double n_win = (double) perf_delta(win.n_prompt_total, last.n_prompt_total);
        const double t_win = (double) perf_delta(win.prompt_busy_us, last.prompt_busy_us);

        res["prompt"] = {
            { "tps_headline", last.processing ? live : tps_last },
            { "tps_window",   n_win > 0.0 ? perf_rate(n_win, t_win) : json() },
            { "tps_total",    perf_rate((double) last.tot_prompt_n, (double) last.tot_prompt_us) },
            { "tps_last",     tps_last },
            { "processing",   last.processing },
            { "layer",        layer },
            { "n_layer",      pg.n_layer > 0 ? json(pg.n_layer) : json() },
            { "n_tokens",     last.processing ? n_cur : last.last_prompt_n },
            { "estimate",     last.processing && estimate },
            { "batch",        last.processing && pl.valid && n_batches > 0 ? json(pl.done.size() + (pl.cur_n > 0 ? 1 : 0)) : json() },
            { "n_batches",    last.processing && pl.valid && n_batches > 0 ? json(n_batches) : json() },
        };
    }

    // 7: NVMe read bandwidth of the device that holds the model file
    if (info.disk_ok && last.has_disk) {
        json bps    = nullptr;
        json bps_10 = nullptr;
        if (prev && prev->has_disk && dt_us > 0.0) {
            bps = perf_rate(512.0*perf_delta(prev->disk_sectors, last.disk_sectors), dt_us);
        }
        const server_perf_tick & s10 = ring[perf_since(ring, PERF_SMOOTH_US)];
        if (&s10 != &last && s10.has_disk) {
            bps_10 = perf_rate(512.0*perf_delta(s10.disk_sectors, last.disk_sectors), (double) (last.t_us - s10.t_us));
        }
        res["nvme"] = { { "device", info.disk_name }, { "read_bps", bps }, { "read_bps_10s", bps_10 } };
    } else {
        res["nvme"] = nullptr;
    }

    // 8: prediction: precision (guesses that were used) and recall (used experts that were guessed)
    const auto & pd = last.predict;
    if (pd.enabled) {
        const uint64_t hits    = perf_delta(base.predict.hits,    pd.hits);
        const uint64_t actual  = perf_delta(base.predict.actual,  pd.actual);
        const uint64_t guesses = perf_delta(base.predict.guesses, pd.guesses);
        const auto & w = win.predict;
        const bool   w_ok = w.enabled && &win != &last;
        res["predict"] = {
            { "mode",             pd.mode == 2 ? "prefetch" : "observe" },
            { "precision_window", w_ok ? perf_ratio(perf_delta(w.hits, pd.hits), perf_delta(w.guesses, pd.guesses)) : json() },
            { "precision_total",  perf_ratio(hits, guesses) },
            { "recall_window",    w_ok ? perf_ratio(perf_delta(w.hits, pd.hits), perf_delta(w.actual, pd.actual)) : json() },
            { "recall_total",     perf_ratio(hits, actual) },
            { "hits",             hits },
            { "actual",           actual },
            { "guesses",          guesses },
            // graph-thread cost of the residency probes, for the overhead checks
            { "probes",           perf_delta(base.predict.probes, pd.probes) },
            { "steps",            perf_delta(base.predict.step_seen, pd.step_seen) },
        };
    } else {
        res["predict"] = nullptr;
    }

    // 9: preload wasted: wasted and late bytes over all bytes the process read from disk, since the baseline
    if (pd.enabled && pd.mode == 2) {
        const uint64_t wasted = perf_delta(base.predict.bytes_wasted, pd.bytes_wasted);
        const uint64_t late   = perf_delta(base.predict.bytes_late,   pd.bytes_late);
        const uint64_t total  = last.has_io && base.has_io ? perf_delta(base.io_read_bytes, last.io_read_bytes) : 0;
        res["preload"] = {
            { "wasted_pct",           total > 0 ? json(std::clamp(100.0*(double) (wasted + late)/total, 0.0, 100.0)) : json() },
            { "bytes_wasted",         wasted },
            { "bytes_late",           late },
            { "bytes_hit",            perf_delta(base.predict.bytes_hit,            pd.bytes_hit) },
            { "bytes_credited",       perf_delta(base.predict.bytes_credited,       pd.bytes_credited) },
            { "bytes_actual",         perf_delta(base.predict.bytes_actual,         pd.bytes_actual) },
            { "bytes_pending",        pd.bytes_pending },
            { "bytes_read_predictor", perf_delta(base.predict.bytes_read_predictor, pd.bytes_read_predictor) },
            { "bytes_read_total",     total },
        };
    } else {
        res["preload"] = nullptr;
    }

    return res;
}

//
// sources (Linux)
//

#if defined(__linux__)
static bool perf_read_file(const char * path, std::string & out) {
    FILE * f = fopen(path, "re");
    if (!f) {
        return false;
    }
    out.clear();
    char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) {
        out.append(buf, n);
    }
    fclose(f);
    return !out.empty();
}

// value after "key" in a "key value" text (meminfo, io), 0 if missing
static uint64_t perf_field(const std::string & s, const char * key) {
    size_t p = 0;
    const size_t n = strlen(key);
    while ((p = s.find(key, p)) != std::string::npos) {
        if (p == 0 || s[p - 1] == '\n') {
            return strtoull(s.c_str() + p + n, nullptr, 10);
        }
        p += n;
    }
    return 0;
}

// the subset of NVML used here, loaded at run time
struct perf_nvml {
    struct util_t { unsigned int gpu; unsigned int memory; };
    struct mem_t  { unsigned long long total; unsigned long long free; unsigned long long used; };

    using init_fn   = int (*)(void);
    using handle_fn = int (*)(unsigned int, void **);
    using util_fn   = int (*)(void *, util_t *);
    using mem_fn    = int (*)(void *, mem_t *);
    using name_fn   = int (*)(void *, char *, unsigned int);

    bool        ok  = false;
    void *      dev = nullptr;
    util_fn     get_util = nullptr;
    mem_fn      get_mem  = nullptr;
    std::string name;

    perf_nvml() {
        void * lib = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
        if (lib == nullptr) {
            SRV_INF("%s", "perf: NVML not available, no GPU figures\n");
            return;
        }
        auto init   = reinterpret_cast<init_fn>  (dlsym(lib, "nvmlInit_v2"));
        auto handle = reinterpret_cast<handle_fn>(dlsym(lib, "nvmlDeviceGetHandleByIndex_v2"));
        auto gname  = reinterpret_cast<name_fn>  (dlsym(lib, "nvmlDeviceGetName"));
        get_util    = reinterpret_cast<util_fn>  (dlsym(lib, "nvmlDeviceGetUtilizationRates"));
        get_mem     = reinterpret_cast<mem_fn>   (dlsym(lib, "nvmlDeviceGetMemoryInfo"));
        if (!init || !handle || !get_util || !get_mem || init() != 0 || handle(0, &dev) != 0) {
            SRV_WRN("%s", "perf: NVML failed to initialize, no GPU figures\n");
            dlclose(lib);
            return;
        }
        char buf[96] = {};
        if (gname && gname(dev, buf, sizeof(buf)) == 0) {
            name = buf;
        }
        ok = true;
        SRV_INF("perf: GPU 0 via NVML: %s\n", name.c_str());
    }
};

static perf_nvml & perf_get_nvml() {
    static perf_nvml nvml;
    return nvml;
}

// the block device that holds the file, as named in /proc/diskstats (md0, nvme0n1p2, ...)
static std::string perf_disk_of(const std::string & path) {
    struct stat st{};
    if (path.empty() || stat(path.c_str(), &st) != 0) {
        return "";
    }
    char link[64];
    snprintf(link, sizeof(link), "/sys/dev/block/%u:%u", major(st.st_dev), minor(st.st_dev));
    char target[PATH_MAX];
    const ssize_t n = readlink(link, target, sizeof(target) - 1);
    if (n <= 0) {
        return "";
    }
    target[n] = 0;
    const char * slash = strrchr(target, '/');
    return slash ? slash + 1 : target;
}

static bool perf_disk_sectors(const std::string & name, uint64_t & sectors) {
    std::string s;
    if (name.empty() || !perf_read_file("/proc/diskstats", s)) {
        return false;
    }
    size_t p = 0;
    while (p < s.size()) {
        size_t e = s.find('\n', p);
        if (e == std::string::npos) {
            e = s.size();
        }
        unsigned int maj, min;
        char dev[64];
        unsigned long long reads, merged, sec;
        if (sscanf(s.c_str() + p, "%u %u %63s %llu %llu %llu", &maj, &min, dev, &reads, &merged, &sec) == 6 && name == dev) {
            sectors = sec;
            return true;
        }
        p = e + 1;
    }
    return false;
}
#endif

server_perf_tick server_perf::sample() const {
    server_perf_tick t;
    t.t_us  = ggml_time_us();
    t.ts_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

#if defined(__linux__)
    std::string s;
    if (perf_read_file("/proc/stat", s)) {
        unsigned long long v[8] = {};
        if (sscanf(s.c_str(), "cpu %llu %llu %llu %llu %llu %llu %llu %llu", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6], &v[7]) == 8) {
            t.cpu_total = v[0] + v[1] + v[2] + v[3] + v[4] + v[5] + v[6] + v[7];
            t.cpu_idle  = v[3] + v[4];
            t.has_cpu   = true;
        }
    }
    if (t.has_cpu && perf_read_file("/proc/self/stat", s)) {
        // the fields after the command name, which may contain spaces and parentheses
        const size_t p = s.rfind(')');
        unsigned long long ut = 0, stm = 0;
        if (p != std::string::npos &&
            sscanf(s.c_str() + p + 1, " %*c %*d %*d %*d %*d %*d %*u %*u %*u %*u %*u %llu %llu", &ut, &stm) == 2) {
            t.proc_ticks = ut + stm;
        }
    }
    if (perf_read_file("/proc/meminfo", s)) {
        t.mem_total     = perf_field(s, "MemTotal:")     * 1024;
        t.mem_free      = perf_field(s, "MemFree:")      * 1024;
        t.mem_available = perf_field(s, "MemAvailable:") * 1024;
        t.mem_cached    = perf_field(s, "Cached:")       * 1024;
        t.has_mem       = t.mem_total > 0;
    }
    if (perf_read_file("/proc/self/io", s)) {
        t.io_read_bytes = perf_field(s, "read_bytes:");
        t.has_io        = true;
    }
    if (info.disk_ok) {
        t.has_disk = perf_disk_sectors(info.disk_name, t.disk_sectors);
    }
    if (info.gpu_ok) {
        auto & nv = perf_get_nvml();
        perf_nvml::util_t u = {};
        perf_nvml::mem_t  m = {};
        if (nv.get_util(nv.dev, &u) == 0 && nv.get_mem(nv.dev, &m) == 0) {
            t.has_gpu       = true;
            t.gpu_busy_pct  = u.gpu;
            t.gpu_mem_used  = m.used;
            t.gpu_mem_total = m.total;
        }
    }
#endif

    const auto & c = counters;
    const auto   r = std::memory_order_relaxed;
    t.n_gen_total    = c.n_gen_total.load(r);
    t.n_prompt_total = c.n_prompt_total.load(r);
    t.prompt_busy_us = c.prompt_busy_us.load(r);
    t.generating     = c.generating.load(r);
    t.cur_gen_n      = c.cur_gen_n.load(r);
    t.cur_gen_t_first_us = c.cur_gen_t_first_us.load(r);
    t.cur_gen_t_last_us  = c.cur_gen_t_last_us.load(r);
    t.processing     = c.processing.load(r);
    t.cur_prompt_n   = c.cur_prompt_n.load(r);
    t.cur_prompt_t_start_us = c.cur_prompt_t_start_us.load(r);
    t.last_gen_n     = c.last_gen_n.load(r);
    t.last_gen_steps = c.last_gen_steps.load(r);
    t.last_gen_us    = c.last_gen_us.load(r);
    t.last_prompt_n  = c.last_prompt_n.load(r);
    t.last_prompt_us = c.last_prompt_us.load(r);
    t.tot_gen_steps  = c.tot_gen_steps.load(r);
    t.tot_gen_us     = c.tot_gen_us.load(r);
    t.tot_prompt_n   = c.tot_prompt_n.load(r);
    t.tot_prompt_us  = c.tot_prompt_us.load(r);
    {
        std::lock_guard<std::mutex> lock(counters.plan_mtx);
        t.plan = counters.plan;
    }

    if (ctx != nullptr) {
        llama_decode_progress(ctx, &t.progress);
    }
    llama_moe_predict_stats(model, &t.predict);
    return t;
}

//
// server_perf
//

server_perf::~server_perf() {
    stop();
}

void server_perf::start(const llama_model * model, const llama_context * ctx, const std::string & model_path) {
    stop();

    server_perf_info inf;
#if defined(__linux__)
    inf.n_cpu   = (int) sysconf(_SC_NPROCESSORS_ONLN);
    inf.clk_tck = sysconf(_SC_CLK_TCK);
    inf.disk_name = perf_disk_of(model_path);
    uint64_t sec = 0;
    inf.disk_ok = perf_disk_sectors(inf.disk_name, sec);
    SRV_INF("perf: model file on block device %s\n", inf.disk_ok ? inf.disk_name.c_str() : "(unknown, no NVMe figures)");
#else
    GGML_UNUSED(model_path);
#endif

    std::lock_guard<std::mutex> lock(mtx);
    this->model = model;
    this->ctx   = ctx;
    // NVML stays loaded across model reloads (sleep)
    inf.gpu_ok   = info.gpu_ok;
    inf.gpu_name = info.gpu_name;
    info = inf;
    ring.clear();
    last_poll_us = 0;

    // the totals count from here, whether or not anyone polls
    const server_perf_tick t0 = sample();
    base.has_io        = t0.has_io;
    base.io_read_bytes = t0.io_read_bytes;
    base.predict       = t0.predict;

    running  = true;
    stopping = false;
    worker = std::thread([this] { loop(); });
#if defined(__linux__)
    pthread_setname_np(worker.native_handle(), "srv-perf");
#endif
}

void server_perf::stop() {
    {
        std::lock_guard<std::mutex> lock(mtx);
        if (!worker.joinable()) {
            return;
        }
        stopping = true;
        running  = false;
    }
    cv.notify_all();
    cv_tick.notify_all();
    worker.join();

    std::lock_guard<std::mutex> lock(mtx);
    model = nullptr;
    ctx   = nullptr;
    ring.clear();
}

void server_perf::loop() {
    std::unique_lock<std::mutex> lock(mtx);
    while (!stopping) {
        const int64_t now = ggml_time_us();
        if (last_poll_us == 0 || now - last_poll_us > PERF_ACTIVE_US) {
            cv.wait(lock, [this] { return stopping || (last_poll_us > 0 && ggml_time_us() - last_poll_us <= PERF_ACTIVE_US); });
            continue;
        }
        // rates over a gap since an earlier active period would mean nothing
        if (!ring.empty() && now - ring.back().t_us > PERF_STALE_US) {
            ring.clear();
        }

        lock.unlock();
#if defined(__linux__)
        // NVML is loaded on the first poll only: a panel nobody opens costs nothing
        if (!nvml_checked) {
            auto & nv = perf_get_nvml();
            std::lock_guard<std::mutex> lk(mtx);
            info.gpu_ok   = nv.ok;
            info.gpu_name = nv.name;
            nvml_checked  = true;
        }
#endif
        const server_perf_tick t = sample();
        if (t.has_io && t.has_mem) {
            llama_moe_predict_set_io_clock(model, t.io_read_bytes, t.mem_cached);
        }
        lock.lock();

        ring.push_back(t);
        while (ring.size() > PERF_RING) {
            ring.pop_front();
        }
        cv_tick.notify_all();

        const int64_t wait = ring.size() < 2 ? PERF_FIRST_US : PERF_TICK_US;
        cv.wait_for(lock, std::chrono::microseconds(wait), [this] { return stopping; });
    }
}

bool server_perf::get(json & out) {
    std::unique_lock<std::mutex> lock(mtx);
    if (!running) {
        return false;
    }
    const auto fresh = [this] {
        return ring.size() >= 2 && ggml_time_us() - ring.back().t_us < PERF_FRESH_US;
    };
    last_poll_us = ggml_time_us();
    if (!fresh()) {
        cv.notify_all();
        cv_tick.wait_for(lock, std::chrono::microseconds(PERF_WAIT_US), [&] { return !running || fresh(); });
    }
    if (!running) {
        return false;
    }
    out = server_perf_compute(ring, base, info, ggml_time_us());
    return true;
}
