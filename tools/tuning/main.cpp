#include "fa-vec.h"
#include "ggml-backend.h"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

struct tuner_def {
    const char * name;
    bool (*run)(ggml_backend_t, ggml_backend_dev_t, const tuner_opts &);
};

static const tuner_def k_tuners[] = {
    { "fa-vec", tuner_fa_vec_run },
};

static void usage(const char * argv0) {
    printf("usage: %s <tuner> [options]\n", argv0);
    printf("\n");
    printf("  offline kernel tuner for the Metal backend: sweeps a kernel's config grid and\n");
    printf("  prints pasteable table rows for the machine it runs on. never a pass/fail test.\n");
    printf("\n");
    printf("  tuners:\n");
    printf("    fa-vec            flash-attn vec (Q,NE) for ggml-metal-tuning.cpp\n");
    printf("\n");
    printf("  options:\n");
    printf("    -b <name>         backend device (default: first Metal device)\n");
    printf("    --dtype <list>    restrict KV dtypes, e.g. f16,q4_0 (default: all)\n");
    printf("    --dk <list>       restrict head sizes, e.g. 128,192 (default: all)\n");
    printf("    --reps <n>        timed reps per candidate, odd for an exact median (default: 7)\n");
    printf("    --seed <n>        RNG seed; per-cell seeds mix it with the shape (default: 1234)\n");
    printf("    --no-cooldown     do not pause/re-measure on thermal drift, only warn\n");
    printf("    --cool-drift <f>  anchor drift that triggers a cooldown (default: 0.10)\n");
    printf("    --cool-eps <f>    anchor tolerance to consider the GPU cool again (default: 0.03)\n");
    printf("    --cool-max-wait <s> give up cooling a cell after this many seconds (default: 120)\n");
    printf("    --cool-max-retry <n> re-measure rounds per cell before giving up (default: 2)\n");
    printf("\n");
    printf("  the table goes to stdout, all diagnostics to stderr:\n");
    printf("    %s fa-vec > rows.txt 2> sweep.log\n", argv0);
}

int main(int argc, char ** argv) {
    const char * tuner = nullptr;
    const char * bname = nullptr;
    tuner_opts   opts;

    for (int i = 1; i < argc; i++) {
        const char * a = argv[i];
        if (strcmp(a, "-h") == 0 || strcmp(a, "--help") == 0) {
            usage(argv[0]);
            return 0;
        } else if (strcmp(a, "-b") == 0 && i + 1 < argc) {
            bname = argv[++i];
        } else if (strcmp(a, "--dtype") == 0 && i + 1 < argc) {
            opts.dtype_filter = argv[++i];
        } else if (strcmp(a, "--dk") == 0 && i + 1 < argc) {
            opts.dk_filter = argv[++i];
        } else if (strcmp(a, "--reps") == 0 && i + 1 < argc) {
            opts.reps = atoi(argv[++i]);
        } else if (strcmp(a, "--seed") == 0 && i + 1 < argc) {
            opts.seed = (unsigned) strtoul(argv[++i], nullptr, 10);
        } else if (strcmp(a, "--no-cooldown") == 0) {
            opts.cooldown = false;
        } else if (strcmp(a, "--cool-drift") == 0 && i + 1 < argc) {
            opts.cool_drift = atof(argv[++i]);
        } else if (strcmp(a, "--cool-eps") == 0 && i + 1 < argc) {
            opts.cool_eps = atof(argv[++i]);
        } else if (strcmp(a, "--cool-max-wait") == 0 && i + 1 < argc) {
            opts.cool_max_wait = atoi(argv[++i]);
        } else if (strcmp(a, "--cool-max-retry") == 0 && i + 1 < argc) {
            opts.cool_max_retry = atoi(argv[++i]);
        } else if (a[0] != '-' && tuner == nullptr) {
            tuner = a;
        } else {
            fprintf(stderr, "error: unrecognized or incomplete argument: %s\n\n", a);
            usage(argv[0]);
            return 1;
        }
    }

    if (tuner == nullptr) {
        usage(argv[0]);
        return 1;
    }
    if (opts.reps < 1) {
        fprintf(stderr, "error: --reps must be >= 1\n");
        return 1;
    }

    const tuner_def * t = nullptr;
    for (const auto & cand : k_tuners) {
        if (strcmp(tuner, cand.name) == 0) {
            t = &cand;
            break;
        }
    }
    if (t == nullptr) {
        fprintf(stderr, "error: unknown tuner: %s\n\n", tuner);
        usage(argv[0]);
        return 1;
    }

    ggml_backend_load_all();

    ggml_backend_dev_t dev = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (bname) {
            if (strcmp(ggml_backend_dev_name(d), bname) == 0) {
                dev = d;
                break;
            }
        } else if (strncmp(ggml_backend_dev_name(d), "MTL", 3) == 0) {
            dev = d;
            break;
        }
    }

    if (dev == nullptr) {
        fprintf(stderr, "error: no %s device found\n", bname ? bname : "Metal");
        return 1;
    }

    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (backend == nullptr) {
        fprintf(stderr, "error: failed to init backend %s\n", ggml_backend_dev_name(dev));
        return 1;
    }

    fprintf(stderr, "device: %s (%s)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev));

    const bool ok = t->run(backend, dev, opts);

    ggml_backend_free(backend);
    ggml_quantize_free();

    return ok ? 0 : 1;
}
