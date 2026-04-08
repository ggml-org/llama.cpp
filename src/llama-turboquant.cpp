/**
 * llama-turboquant.cpp — quant.cpp (turboquant) integration for llama.cpp
 *
 * Thin shim between llama.cpp's CLI/init and quant.cpp's public C API.
 *
 * The heavy lifting (quantize, dequantize, vec_dot wrappers) lives in
 * ggml.c and ggml-cpu.c where it calls TQ_TRAITS[] from libturboquant.
 * This file only handles:
 *   - Startup validation (llama_turboquant_init)
 *   - CLI name resolution (llama_turboquant_type_from_str)
 *
 * Canonical type names (e.g. "polar_3b") are resolved via
 * tq_type_from_name() in libturboquant. Short aliases (e.g. "turbo3",
 * "tq-uniform-4b") are mapped here to keep the alias table close to
 * the CLI that uses it.
 */

#ifdef LLAMA_TURBOQUANT

#include "llama-turboquant.h"
#include "ggml.h"

extern "C" {
#include "turboquant/turboquant.h"
}

#include <cstdio>
#include <cstring>

// ---------------------------------------------------------------------------
// tq_type <-> ggml_type mapping
//
// The ggml_type enum values (GGML_TYPE_TQ_POLAR_3B = 42, ...) are in the
// same order as tq_type (TQ_TYPE_POLAR_3B = 0, ...), so the mapping is
// just an offset.
// ---------------------------------------------------------------------------

static enum ggml_type tq_to_ggml(tq_type t) {
    if (t < 0 || t >= TQ_TYPE_COUNT) { return GGML_TYPE_COUNT; }
    return (enum ggml_type)((int)GGML_TYPE_TQ_POLAR_3B + (int)t);
}

// ---------------------------------------------------------------------------
// CLI alias table
//
// Short-form names accepted by --cache-type-k / --cache-type-v.
// Canonical names ("polar_3b", "turbo_kv_4b", ...) are handled by
// tq_type_from_name() in libturboquant — no duplication needed.
// ---------------------------------------------------------------------------

struct tq_alias { const char * name; tq_type type; };

static const tq_alias aliases[] = {
    { "turbo3",          TQ_TYPE_TURBO_3B    },
    { "tq-turbo-3b",     TQ_TYPE_TURBO_3B    },
    { "turbo4",          TQ_TYPE_TURBO_4B    },
    { "tq-turbo-4b",     TQ_TYPE_TURBO_4B    },
    { "polar3",          TQ_TYPE_POLAR_3B    },
    { "tq-polar-3b",     TQ_TYPE_POLAR_3B    },
    { "polar4",          TQ_TYPE_POLAR_4B    },
    { "tq-polar-4b",     TQ_TYPE_POLAR_4B    },
    { "qjl1",            TQ_TYPE_QJL_1B      },
    { "tq-qjl-1b",       TQ_TYPE_QJL_1B      },
    { "uniform4",        TQ_TYPE_UNIFORM_4B  },
    { "tq-uniform-4b",   TQ_TYPE_UNIFORM_4B  },
    { "uniform2",        TQ_TYPE_UNIFORM_2B  },
    { "tq-uniform-2b",   TQ_TYPE_UNIFORM_2B  },
    { "uniform3",        TQ_TYPE_UNIFORM_3B  },
    { "tq-uniform-3b",   TQ_TYPE_UNIFORM_3B  },
    { "turbokv3",        TQ_TYPE_TURBO_KV_3B },
    { "tq-turbo-kv-3b",  TQ_TYPE_TURBO_KV_3B },
    { "turbokv4",        TQ_TYPE_TURBO_KV_4B },
    { "tq-turbo-kv-4b",  TQ_TYPE_TURBO_KV_4B },
    { "turbokv1",        TQ_TYPE_TURBO_KV_1B },
    { "tq-turbo-kv-1b",  TQ_TYPE_TURBO_KV_1B },
    { "turbokv2",        TQ_TYPE_TURBO_KV_2B },
    { "tq-turbo-kv-2b",  TQ_TYPE_TURBO_KV_2B },
    { "turbokv5",        TQ_TYPE_TURBO_KV_5B },
    { "tq-turbo-kv-5b",  TQ_TYPE_TURBO_KV_5B },
    { "turbokv4o",       TQ_TYPE_TURBO_KV_4BO },
    { "tq-turbo-kv-4bo", TQ_TYPE_TURBO_KV_4BO },
    { "turbokv3o",       TQ_TYPE_TURBO_KV_3BO },
    { "tq-turbo-kv-3bo", TQ_TYPE_TURBO_KV_3BO },
};

static const int n_aliases = sizeof(aliases) / sizeof(aliases[0]);

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

void llama_turboquant_init(void) {
    static bool done = false;
    if (done) { return; }
    done = true;

    // Validate that TQ_TRAITS entries are populated
    for (int i = 0; i < TQ_TYPE_COUNT; i++) {
        if (!TQ_TRAITS[i].quantize || !TQ_TRAITS[i].dequantize) {
            fprintf(stderr, "[turboquant] WARNING: type %s missing quantize/dequantize\n",
                    tq_type_name((tq_type)i));
        }
    }

    fprintf(stderr, "[turboquant] Registered %d KV cache types (ggml IDs %d..%d)\n",
            (int)TQ_TYPE_COUNT,
            (int)GGML_TYPE_TQ_POLAR_3B,
            (int)GGML_TYPE_TQ_TURBO_KV_3BO);
}

enum ggml_type llama_turboquant_type_from_str(const char * name) {
    if (!name) { return GGML_TYPE_COUNT; }

    // 1) Try canonical name via libturboquant (e.g. "polar_3b", "turbo_kv_4b")
    tq_type t = tq_type_from_name(name);
    if (t != TQ_TYPE_COUNT) { return tq_to_ggml(t); }

    // 2) Try short aliases (e.g. "turbo3", "tq-uniform-4b")
    for (int i = 0; i < n_aliases; i++) {
        if (strcmp(name, aliases[i].name) == 0) {
            return tq_to_ggml(aliases[i].type);
        }
    }

    // 3) Try matching the ggml type_name (e.g. "tq_polar_3b" with underscore prefix)
    for (int i = 0; i < TQ_TYPE_COUNT; i++) {
        // ggml type names are "tq_polar_3b", quant.cpp names are "polar_3b"
        // Check if name matches "tq_" + canonical name
        const char * canonical = tq_type_name((tq_type)i);
        if (!canonical) continue;
        char buf[64];
        snprintf(buf, sizeof(buf), "tq_%s", canonical);
        if (strcmp(name, buf) == 0) {
            return tq_to_ggml((tq_type)i);
        }
    }

    return GGML_TYPE_COUNT;
}

void llama_turboquant_print_types(void) {
    fprintf(stderr, "Available quant.cpp (turboquant) KV cache types:\n");
    for (int i = 0; i < TQ_TYPE_COUNT; i++) {
        float bpe = tq_type_bpe((tq_type)i);
        fprintf(stderr, "  tq_%-16s  %.1f bpe  %.1fx compression vs FP16\n",
                tq_type_name((tq_type)i), bpe,
                (bpe > 0.0f) ? 16.0f / bpe : 0.0f);
    }
}

#endif // LLAMA_TURBOQUANT
