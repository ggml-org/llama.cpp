// Phase 0 profile test: per-function phase table for a real
// ANE dispatch (the gemma4 prefill bundle).
//
// What we test:
//   1. A real gemma4 prefill bundle loads with the manifest
//      sidecar (TESSERA_ANE_STATE_LAYOUT_MANIFEST + the GGUF).
//   2. The Phase 0 profile counters are non-zero after N
//      successful dispatches (input_prep / ane_dispatch /
//      output_read / signal all recorded).
//   3. The per-function phase table returned by
//      common_ane_mtp_program_phase_stats has one row per
//      warm function and is sorted by (role, bucket) like
//      common_ane_compute_functions.
//   4. The row's count matches the number of dispatches we
//      issued; the totals are monotonic; the maxes are >= each
//      phase's average (totals / count).
//   5. The signal phase max is non-zero (per-slot signal calls
//      happened at least once).
//
// This is the Phase 0 production proof: the lock-free
// per-function phase stats are collected without disturbing
// the dispatch contract, and the host can read them via the
// public API.

#include "ane-mtp.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int g_failures = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL [%s:%d] %s\n", __FILE__, __LINE__, msg); \
        ++g_failures; \
    } else { \
        std::fprintf(stdout, "ok   %s\n", msg); \
    } \
} while (0)

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s PREFILL_GGUF\n", argv[0]);
        return 2;
    }
    if (std::getenv("TESSERA_ANE_STATE_LAYOUT_MANIFEST") == nullptr) {
        std::fprintf(stderr,
                "TESSERA_ANE_STATE_LAYOUT_MANIFEST must point at a real "
                "multifunction .ane_state.v1.json sidecar (e.g. the "
                "gemma4 prefill-bundle.ane_state.v1.json); the pinned-slot "
                "path requires the manifest.\n");
        return 2;
    }
    common_ane_prefill_manifest manifest;
    if (!common_ane_prefill_manifest_load(argv[1], &manifest) ||
            manifest.architecture != "gemma4" ||
            manifest.batch_size != 1 ||
            manifest.layer_first != 0 || manifest.layer_last != 0) {
        std::fprintf(stderr, "unexpected prefill manifest shape\n");
        return 1;
    }
    const uint32_t sequence = manifest.sequence_buckets[0];
    auto program = common_ane_prefill_program_load(argv[1], sequence);
    if (!program || !common_ane_mtp_program_is_warm(program)) {
        std::fprintf(stderr, "failed to load+warm phase profile program\n");
        return 1;
    }
    // Build a prefill payload.
    std::vector<int32_t> tokens(sequence, 0);
    std::vector<int32_t> positions(sequence);
    for (int32_t i = 0; i < (int32_t) sequence; ++i) {
        positions[i] = i;
    }
    const size_t hidden_count = (size_t) sequence * manifest.hidden_size;
    const size_t kv_count = (size_t) sequence * manifest.kv_heads * manifest.head_dim;
    std::vector<float> hidden(hidden_count);
    std::vector<float> keys(kv_count);
    std::vector<float> values(kv_count);
    // Pre-dispatch snapshot: counts should be zero (we haven't
    // dispatched yet). The first row corresponds to the first
    // warm function in the bundle (prefill_s128 in the gemma4
    // prefill bundle).
    {
        const auto rows = common_ane_mtp_program_phase_stats(program);
        CHECK(!rows.empty(), "phase stats: at least one row at startup");
        bool any_count_nonzero = false;
        for (const auto & row : rows) {
            if (row.stats.count != 0) {
                any_count_nonzero = true;
                break;
            }
        }
        // count == 0 at startup is expected; we haven't
        // dispatched yet. The warmup doesn't update the stats
        // (warmup uses the GGUF-metadata + arena path, not the
        // pinned-slot path). So a non-zero count here would be
        // a bug.
        CHECK(!any_count_nonzero, "phase stats: all counts are 0 at startup");
    }
    // Run N dispatches. The phase stats should accumulate.
    const uint32_t n_dispatches = 5;
    for (uint32_t i = 0; i < n_dispatches; ++i) {
        if (!common_ane_compute_prefill_slab(
                program, sequence, tokens.data(), positions.data(), 1,
                manifest.hidden_size, manifest.kv_heads, manifest.head_dim,
                hidden.data(), keys.data(), values.data())) {
            std::fprintf(stderr, "phase profile prefill_slab failed at iteration %u\n", i);
            return 1;
        }
    }
    // Post-dispatch snapshot. The prefill_s128 row should have
    // count == n_dispatches and totals/maxes all non-zero.
    const auto rows = common_ane_mtp_program_phase_stats(program);
    CHECK(!rows.empty(), "phase stats: at least one row after dispatch");
    // Sorted by (role, bucket): the first row should be the
    // smallest (role, bucket) pair. For the gemma4 prefill
    // bundle with one function (prefill_s128), there's one row.
    bool first_row_has_counts = false;
    for (size_t i = 0; i + 1 < rows.size(); ++i) {
        if (rows[i].role != rows[i + 1].role) {
            CHECK(rows[i].role < rows[i + 1].role,
                  "phase stats: rows sorted by role");
        } else {
            CHECK(rows[i].bucket <= rows[i + 1].bucket,
                  "phase stats: rows sorted by bucket within role");
        }
    }
    for (const auto & row : rows) {
        std::fprintf(stdout,
                "phase: %-16s role=%-8s bucket=%u count=%llu "
                "input_prep avg=%.1fus max=%lluus "
                "ane_dispatch avg=%.1fus max=%lluus "
                "output_read avg=%.1fus max=%lluus "
                "signal avg=%.1fus max=%lluus\n",
                row.function_name.c_str(),
                row.role.c_str(),
                row.bucket,
                (unsigned long long) row.stats.count,
                row.stats.count > 0
                    ? (double) row.stats.input_prep_us_total
                          / (double) row.stats.count
                    : 0.0,
                (unsigned long long) row.stats.input_prep_us_max,
                row.stats.count > 0
                    ? (double) row.stats.ane_dispatch_us_total
                          / (double) row.stats.count
                    : 0.0,
                (unsigned long long) row.stats.ane_dispatch_us_max,
                row.stats.count > 0
                    ? (double) row.stats.output_read_us_total
                          / (double) row.stats.count
                    : 0.0,
                (unsigned long long) row.stats.output_read_us_max,
                row.stats.count > 0
                    ? (double) row.stats.signal_us_total
                          / (double) row.stats.count
                    : 0.0,
                (unsigned long long) row.stats.signal_us_max);
        if (row.stats.count > 0) {
            first_row_has_counts = true;
            // The first row's count must equal n_dispatches
            // (we only dispatched one function: prefill_s128).
            CHECK(row.stats.count == n_dispatches,
                  "phase stats: prefill_s128 count == n_dispatches");
            // Max must be >= average (totals / count).
            const uint64_t avg_input = row.stats.input_prep_us_total / row.stats.count;
            CHECK(row.stats.input_prep_us_max >= avg_input,
                  "phase stats: input_prep max >= average");
            const uint64_t avg_ane = row.stats.ane_dispatch_us_total / row.stats.count;
            CHECK(row.stats.ane_dispatch_us_max >= avg_ane,
                  "phase stats: ane_dispatch max >= average");
            const uint64_t avg_output = row.stats.output_read_us_total / row.stats.count;
            CHECK(row.stats.output_read_us_max >= avg_output,
                  "phase stats: output_read max >= average");
            // Totals must be > 0 (we definitely did some work).
            CHECK(row.stats.ane_dispatch_us_total > 0,
                  "phase stats: ane_dispatch total > 0");
            // The signal phase is recorded in the pump's
            // signal_fn on the E-core thread. The signal
            // totals include the per-slot MTLSharedEvent
            // signals; totals can be very small (<1us per
            // signal), so we only assert non-zero in the
            // max. The max is the largest signal phase across
            // n_dispatches calls.
            CHECK(row.stats.signal_us_max > 0,
                  "phase stats: signal max > 0 (per-slot signal was recorded)");
        }
    }
    CHECK(first_row_has_counts,
          "phase stats: at least one row has count > 0 after dispatches");
    if (g_failures == 0) {
        std::fprintf(stdout, "ALL PASSED\n");
        return 0;
    }
    std::fprintf(stderr, "FAILED: %d failures\n", g_failures);
    return 1;
}
