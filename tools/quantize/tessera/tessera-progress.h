#pragma once

// Structured progress reporting for the tessera quantize pipeline.
//
// Two sinks, driven from one set of atomic counters so worker threads can
// update progress without contending on output:
//
//   1. Terminal live-update. A background ticker thread redraws a single
//      status line (carriage-return + clear-to-EOL) roughly 5x/s when stderr
//      is a TTY. Suppressed when stderr is redirected so log files stay clean.
//
//   2. NDJSON event stream. When a progress file path is set (via the
//      --progress-file flag or TESSERA_PROGRESS_FILE), one JSON object per
//      ticker tick is appended. The Studio UI tails this file to render
//      progress bars. Format per line:
//
//        {"ts":1699999999,"phase":"ga-evolve","current":42,"total":290,
//         "elapsed_s":134.0,"rate":0.71,"eta_s":349.0,"label":"blk.12.ffn"}
//
// The reporter is a single handle (ts_progress) created at the start of the
// pipeline and destroyed at the end. Phase transitions reset the counters;
// the elapsed clock is monotonic across phases so the UI can show total
// pipeline time.

#include <cstdint>
#include <stdio.h>

struct ts_progress;

// Phase labels used across the pipeline. Kept short for the terminal line.
// Anything else passed to ts_progress_set_phase is printed verbatim.
struct ts_progress_phase {
    static constexpr const char * SETUP     = "setup";
    static constexpr const char * CALIB     = "calib";
    static constexpr const char * HIGGS     = "higgs";
    static constexpr const char * GA_PREP   = "ga-prep";
    static constexpr const char * GA_SCREEN = "ga-screen";
    static constexpr const char * GA_EVOLVE = "ga-evolve";
    static constexpr const char * QUANTIZE  = "quantize";
    static constexpr const char * FINALIZE  = "finalize";
};

// Create and start the reporter. `initial_phase`/`initial_total` seed the
// first phase. `progress_file` may be NULL/empty to disable NDJSON output.
// Terminal output is auto-enabled when stderr is a TTY; pass force_terminal
// = true to enable it unconditionally (used by --verbose).
struct ts_progress * ts_progress_create(const char * initial_phase,
                                        int64_t       initial_total,
                                        const char * progress_file,
                                        bool          force_terminal);

// Reset current to 0, total to `total`, and relabel the phase. The overall
// elapsed clock keeps running. Optional `note` is emitted once in the NDJSON
// stream as a phase-boundary marker.
void ts_progress_set_phase(struct ts_progress * p,
                           const char * phase,
                           int64_t total,
                           const char * note);

// Bump current by `delta` (default 1). Thread-safe. `label` is an optional
// per-item tag (e.g. the tensor name); may be NULL. Copied internally so the
// caller does not need to keep it alive.
void ts_progress_inc(struct ts_progress * p,
                     int64_t delta,
                     const char * label);

// Print a final summary line (phase, totals, total elapsed) and stop the
// ticker. Idempotent.
void ts_progress_finish(struct ts_progress * p);

// Stop the ticker and free resources. Implies finish.
void ts_progress_destroy(struct ts_progress * p);
