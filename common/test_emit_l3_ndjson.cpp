//
// test_emit_l3_ndjson.cpp
//
// Companion to test_tessera_analytical_io.cpp. Writes a small
// sample L3-outlier NDJSON file using tessera::analytical::write_record,
// so the Python end-to-end test (tools/tessera/test_analytical_io.py)
// can verify the cross-language round-trip:
//
//     C++ writer  ->  NDJSON on disk  ->  polars reader
//
// Run as:
//
//     test-emit-l3-ndjson <output.ndjson>
//
// Writes 3 records, one per (tensor, sidecar_label) combination.
// Mirrors the same shape as a real L3 outlier report from
// tools/tessera/l3_outlier_report.py.
//

#include "tessera-analytical-io.h"

#include <cstdio>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <output.ndjson>\n", argv[0]);
        return 1;
    }
    auto path = fs::path(argv[1]);
    try {
        // Three records across two sidecar labels. The first two share
        // a sidecar_label; the third is a different sidecar so the
        // polars read can verify that the per-row records are not
        // accidentally collapsed by the writer.
        tessera::analytical::write_record(path, tessera::analytical::SCHEMA_L3_OUTLIER, {
            {"sidecar_label", "ckpt-v3-q4_k"},
            {"tensor",        "blk.12.attn_q.weight"},
            {"layer",         "blk.12"},
            {"rows",          4096},
            {"cols",          4096},
            {"total_elements", 16777216},
            {"outlier_count",  17000},
            {"outlier_fraction", 0.001013},
            {"threshold",     6.0f},
            {"skipped",       false},
            {"kernel_version", "integrate/polars-scout-phase-a abc1234"},
            {"created_at",    "2026-08-03T20:30:00Z"},
            {"tessera_main_tip", "df3e9c6cd"},
        });
        tessera::analytical::write_record(path, tessera::analytical::SCHEMA_L3_OUTLIER, {
            {"sidecar_label", "ckpt-v3-q4_k"},
            {"tensor",        "blk.13.attn_q.weight"},
            {"layer",         "blk.13"},
            {"rows",          4096},
            {"cols",          4096},
            {"total_elements", 16777216},
            {"outlier_count",  18200},
            {"outlier_fraction", 0.001084},
            {"threshold",     6.0f},
            {"skipped",       false},
            {"kernel_version", "integrate/polars-scout-phase-a abc1234"},
            {"created_at",    "2026-08-03T20:30:00Z"},
            {"tessera_main_tip", "df3e9c6cd"},
        });
        tessera::analytical::write_record(path, tessera::analytical::SCHEMA_L3_OUTLIER, {
            {"sidecar_label", "ckpt-v3-q5_k"},
            {"tensor",        "blk.12.attn_q.weight"},
            {"layer",         "blk.12"},
            {"rows",          4096},
            {"cols",          4096},
            {"total_elements", 16777216},
            {"outlier_count",  9000},
            {"outlier_fraction", 0.000536},
            {"threshold",     6.0f},
            {"skipped",       false},
            {"kernel_version", "integrate/polars-scout-phase-a abc1234"},
            {"created_at",    "2026-08-03T20:30:00Z"},
            {"tessera_main_tip", "df3e9c6cd"},
        });
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
    std::fprintf(stdout, "wrote %s\n", path.string().c_str());
    return 0;
}
