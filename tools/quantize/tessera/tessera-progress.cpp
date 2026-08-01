#include "tessera-progress.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

// ISO C does not define isatty; POSIX does. The header is environment-portable
// enough on the targets we build for (macOS/iOS/Linux). Windows builds would
// need _isatty from <io.h>, but tessera does not currently build there.
#ifdef _WIN32
#  include <io.h>
#  define TS_ISATTY(fd) _isatty(fd)
#else
#  define TS_ISATTY(fd) isatty(fd)
#endif

namespace {

// One ring of recent labels so the live line can show "what just finished"
// without worker threads needing to coordinate. Lock-free enough for our
// purposes: the slot is written under a relaxed atomic index, and the reader
// tolerates torn reads (it just shows a possibly-stale label). 8 slots is
// generous for the 5x/s ticker and 8-worker GA.
constexpr size_t TS_PLABEL_RING = 8;

struct ts_progress_impl {
    std::atomic<bool>        stop_{false};
    std::atomic<bool>        finished_{false};
    std::atomic<int64_t>     current_{0};
    std::atomic<int64_t>     total_{0};
    std::chrono::steady_clock::time_point start_;

    std::string              phase_;
    std::string              progress_file_;  // empty = no NDJSON
    FILE *                   ndjson_fp_ = nullptr;

    std::thread              ticker_;
    bool                     terminal_ = false;

    // Recent tensor/layer names; relaxed atomic rotation. Each entry is owned
    // by one slot at a time under the ring index; the write replaces an entry
    // the reader may be observing, but label strings are small and the worst
    // case is a slightly stale or truncated label on the live line.
    std::atomic<unsigned>    label_idx_{0};
    std::vector<std::string> labels_{TS_PLABEL_RING};

    explicit ts_progress_impl(std::chrono::steady_clock::time_point s)
        : start_(s), labels_(TS_PLABEL_RING) {}
};

// Format seconds as H:MM:SS for the live line. Buffer must hold >= 10 chars.
static void ts_format_hms(double seconds, char * out, size_t out_sz) {
    if (seconds < 0 || !std::isfinite(seconds)) seconds = 0;
    int64_t s = (int64_t)seconds;
    int h = (int)(s / 3600);
    int m = (int)((s % 3600) / 60);
    int sec = (int)(s % 60);
    if (h > 0) {
        snprintf(out, out_sz, "%dh%02dm%02ds", h, m, sec);
    } else {
        snprintf(out, out_sz, "%dm%02ds", m, sec);
    }
}

// Rate and ETA from progress + elapsed. Returns false when the rate is not
// meaningful yet (no items, or too little time to estimate).
static bool ts_compute_rate_eta(const ts_progress_impl * p,
                                double elapsed_s,
                                double * rate,
                                double * eta_s) {
    int64_t cur = p->current_.load(std::memory_order_relaxed);
    int64_t tot = p->total_.load(std::memory_order_relaxed);
    if (cur <= 0 || elapsed_s < 0.5) {
        if (rate) *rate = 0.0;
        if (eta_s) *eta_s = -1.0;
        return false;
    }
    double r = (double)cur / elapsed_s;
    if (rate) *rate = r;
    if (eta_s) {
        if (r <= 0.0 || tot <= cur) {
            *eta_s = -1.0;
        } else {
            *eta_s = (double)(tot - cur) / r;
        }
    }
    return r > 0.0;
}

// Sample one recent label. May return an empty string if the workers have not
// recorded one yet for this phase.
static std::string ts_sample_label(const ts_progress_impl * p) {
    if (p->labels_.empty()) return std::string();
    unsigned idx = p->label_idx_.load(std::memory_order_relaxed);
    // Index most-recent first; tolerate the ring being empty/uninitialized.
    for (size_t k = 0; k < TS_PLABEL_RING; k++) {
        size_t slot = (idx + TS_PLABEL_RING - k) % TS_PLABEL_RING;
        if (!p->labels_[slot].empty()) return p->labels_[slot];
    }
    return std::string();
}

// One render of the terminal live line. Writes to stderr.
static void ts_render_terminal(const ts_progress_impl * p, double elapsed_s) {
    int64_t cur = p->current_.load(std::memory_order_relaxed);
    int64_t tot = p->total_.load(std::memory_order_relaxed);
    double pct = (tot > 0) ? (100.0 * (double)cur / (double)tot) : 0.0;
    if (pct > 100.0) pct = 100.0;

    double rate = 0.0, eta_s = -1.0;
    ts_compute_rate_eta(p, elapsed_s, &rate, &eta_s);

    char elapsed_hms[16];
    ts_format_hms(elapsed_s, elapsed_hms, sizeof(elapsed_hms));

    char eta_hms[16] = "-";
    if (eta_s >= 0.0) {
        ts_format_hms(eta_s, eta_hms, sizeof(eta_hms));
    }

    std::string label = ts_sample_label(p);
    if (label.size() > 28) {
        label = "..." + label.substr(label.size() - 25);
    }

    // [phase]  pct%  cur/tot  elapsed  rate it/s  eta  label
    // Use carriage return + clear-to-EOL so the line redraws in place.
    if (rate > 0.0) {
        std::fprintf(stderr,
                     "\r\033[K[%s] %5.1f%%  %lld/%lld  %s  %.2f it/s  eta %s  %s",
                     p->phase_.c_str(), pct,
                     (long long)cur, (long long)tot,
                     elapsed_hms, rate, eta_hms, label.c_str());
    } else {
        std::fprintf(stderr,
                     "\r\033[K[%s] %5.1f%%  %lld/%lld  %s  %s",
                     p->phase_.c_str(), pct,
                     (long long)cur, (long long)tot,
                     elapsed_hms, label.c_str());
    }
    std::fflush(stderr);
}

// One NDJSON line. Appended to the progress file if one was configured.
static void ts_render_ndjson(ts_progress_impl * p, double elapsed_s) {
    if (!p->ndjson_fp_) return;
    int64_t cur = p->current_.load(std::memory_order_relaxed);
    int64_t tot = p->total_.load(std::memory_order_relaxed);
    double rate = 0.0, eta_s = -1.0;
    ts_compute_rate_eta(p, elapsed_s, &rate, &eta_s);

    auto now = std::chrono::system_clock::now();
    int64_t ts = (int64_t)std::chrono::duration_cast<std::chrono::seconds>(
        now.time_since_epoch()).count();

    std::string label = ts_sample_label(p);
    // Escape backslash and double-quote in the label so the JSON is valid even
    // if a tensor name contained them (unlikely, but cheap to handle).
    std::string esc;
    esc.reserve(label.size() + 4);
    for (char c : label) {
        if (c == '\\' || c == '"') esc.push_back('\\');
        esc.push_back(c);
    }

    std::fprintf(p->ndjson_fp_,
        "{\"ts\":%lld,\"phase\":\"%s\",\"current\":%lld,\"total\":%lld,"
        "\"elapsed_s\":%.3f,\"rate\":%.4f,\"eta_s\":%.3f,\"label\":\"%s\"}\n",
        (long long)ts, p->phase_.c_str(),
        (long long)cur, (long long)tot,
        elapsed_s, rate, eta_s, esc.c_str());
    std::fflush(p->ndjson_fp_);
}

// Ticker loop: roughly 5 renders/s. Both sinks share one loop so NDJSON and
// terminal stay in sync without per-render duplication of bookkeeping.
static void ts_progress_tick_loop(ts_progress_impl * p) {
    using clock = std::chrono::steady_clock;
    const auto interval = std::chrono::milliseconds(200);
    while (!p->stop_.load(std::memory_order_relaxed)) {
        double elapsed_s = std::chrono::duration<double>(
            clock::now() - p->start_).count();
        if (p->terminal_) {
            ts_render_terminal(p, elapsed_s);
        }
        if (p->ndjson_fp_) {
            ts_render_ndjson(p, elapsed_s);
        }
        std::this_thread::sleep_for(interval);
    }
}

}  // namespace

struct ts_progress * ts_progress_create(const char * initial_phase,
                                        int64_t       initial_total,
                                        const char * progress_file,
                                        bool          force_terminal) {
    auto * p = new ts_progress_impl(std::chrono::steady_clock::now());
    p->phase_ = initial_phase ? initial_phase : ts_progress_phase::SETUP;
    p->total_.store(initial_total, std::memory_order_relaxed);

    // Terminal: on when stderr is a TTY or when explicitly forced (e.g. the
    // user passed --verbose even with redirected stderr).
    p->terminal_ = force_terminal || (TS_ISATTY(fileno(stderr)) != 0);

    if (progress_file && progress_file[0] != '\0') {
        p->progress_file_ = progress_file;
        p->ndjson_fp_ = std::fopen(progress_file, "wb");
        if (!p->ndjson_fp_) {
            std::fprintf(stderr,
                "tessera-progress: warning: could not open '%s' for NDJSON output (%s)\n",
                progress_file, std::strerror(errno));
        }
    }

    if (p->terminal_ || p->ndjson_fp_) {
        p->ticker_ = std::thread(ts_progress_tick_loop, p);
    }
    return reinterpret_cast<struct ts_progress *>(p);
}

void ts_progress_set_phase(struct ts_progress * p_o,
                           const char * phase,
                           int64_t total,
                           const char * note) {
    auto * p = reinterpret_cast<ts_progress_impl *>(p_o);
    if (!p) return;
    if (phase) p->phase_ = phase;
    p->current_.store(0, std::memory_order_relaxed);
    p->total_.store(total, std::memory_order_relaxed);
    // Clear the label ring so a stale label from the previous phase does not
    // appear on the first render of the new one.
    for (auto & s : p->labels_) s.clear();
    p->label_idx_.store(0, std::memory_order_relaxed);

    // Emit a phase-boundary marker line to NDJSON so the UI can segment phases.
    if (p->ndjson_fp_ && note) {
        auto now = std::chrono::system_clock::now();
        int64_t ts = (int64_t)std::chrono::duration_cast<std::chrono::seconds>(
            now.time_since_epoch()).count();
        std::fprintf(p->ndjson_fp_,
            "{\"ts\":%lld,\"event\":\"phase\",\"phase\":\"%s\",\"total\":%lld,\"note\":\"%s\"}\n",
            (long long)ts, p->phase_.c_str(), (long long)total, note);
        std::fflush(p->ndjson_fp_);
    }
}

void ts_progress_inc(struct ts_progress * p_o,
                     int64_t delta,
                     const char * label) {
    auto * p = reinterpret_cast<ts_progress_impl *>(p_o);
    if (!p) return;
    p->current_.fetch_add(delta, std::memory_order_relaxed);
    if (label && label[0] != '\0') {
        unsigned idx = p->label_idx_.fetch_add(1, std::memory_order_relaxed);
        size_t slot = idx % TS_PLABEL_RING;
        p->labels_[slot] = label;
    }
}

void ts_progress_finish(struct ts_progress * p_o) {
    auto * p = reinterpret_cast<ts_progress_impl *>(p_o);
    if (!p || p->finished_.exchange(true)) return;
    p->stop_.store(true, std::memory_order_relaxed);
    if (p->ticker_.joinable()) {
        p->ticker_.join();
    }

    // Final summary line on the terminal: one complete line (newline, not CR).
    if (p->terminal_) {
        double elapsed_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - p->start_).count();
        int64_t cur = p->current_.load(std::memory_order_relaxed);
        int64_t tot = p->total_.load(std::memory_order_relaxed);
        char elapsed_hms[16];
        ts_format_hms(elapsed_s, elapsed_hms, sizeof(elapsed_hms));
        std::fprintf(stderr,
            "\r\033[K[%s] done: %lld/%lld in %s\n",
            p->phase_.c_str(),
            (long long)cur, (long long)tot, elapsed_hms);
        std::fflush(stderr);
    }

    // Final NDJSON marker.
    if (p->ndjson_fp_) {
        double elapsed_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - p->start_).count();
        int64_t cur = p->current_.load(std::memory_order_relaxed);
        int64_t tot = p->total_.load(std::memory_order_relaxed);
        auto now = std::chrono::system_clock::now();
        int64_t ts = (int64_t)std::chrono::duration_cast<std::chrono::seconds>(
            now.time_since_epoch()).count();
        std::fprintf(p->ndjson_fp_,
            "{\"ts\":%lld,\"event\":\"done\",\"phase\":\"%s\","
            "\"current\":%lld,\"total\":%lld,\"elapsed_s\":%.3f}\n",
            (long long)ts, p->phase_.c_str(),
            (long long)cur, (long long)tot, elapsed_s);
        std::fflush(p->ndjson_fp_);
    }
}

void ts_progress_destroy(struct ts_progress * p_o) {
    auto * p = reinterpret_cast<ts_progress_impl *>(p_o);
    if (!p) return;
    ts_progress_finish(p_o);
    if (p->ndjson_fp_) {
        std::fclose(p->ndjson_fp_);
        p->ndjson_fp_ = nullptr;
    }
    delete p;
}
