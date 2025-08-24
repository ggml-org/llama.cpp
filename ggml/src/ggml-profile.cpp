#include "ggml-profile.h"

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <string>
#include <chrono>

#ifdef GGML_GRAPH_PROFILER

struct ggml_profile_output {
    const char * prefix;
    FILE *       stream;
};

extern "C" void ggml_graph_profile_init(struct ggml_cgraph *cg, int n_threads)
{
    // TODO: make this a param
    const char *env = getenv("GGML_GRAPH_PROFILE");
    if (!env) { return; }

    // The number of threads may change between passes (pp vs tg).
    // Allocate for max_n_threads for simplicity for now.
    // TODO: use aligned allocator

    size_t node_size = sizeof(struct ggml_profile_timing) * GGML_MAX_N_THREADS;
    size_t pvec_size = sizeof(std::intptr_t) * cg->n_nodes;
    size_t time_size = node_size * cg->n_nodes;
    size_t t_size    = pvec_size + time_size + sizeof(ggml_profile_output) + sizeof(ggml_profile_data);

    uint8_t * ptr = (uint8_t *) malloc(t_size);
    if (!ptr) {
        fprintf(stderr, "ggml-profile: failed to allocate profiling data : n_threads %d n_nodes %d\n", n_threads, cg->n_nodes);
        return;
    }
    memset(ptr, 0, t_size);

    // init all pointers
    cg->prof         = (ggml_profile_data *)    ptr; ptr += sizeof(ggml_profile_data);
    cg->prof->output = (ggml_profile_output *)  ptr; ptr += sizeof(ggml_profile_output);
    cg->prof->timing = (ggml_profile_timing **) ptr; ptr += pvec_size;
    for (int i=0; i < cg->n_nodes; i++) {
        cg->prof->timing[i] = (struct ggml_profile_timing *) ptr; ptr += node_size;
    }

    // init the output
    ggml_profile_output *out = cg->prof->output;
    if (!strcmp("stderr", env) || !strcmp("1", env)) {
        out->prefix = "ggml-profile:";
        out->stream = stderr;
    } else {
        out->prefix = "";
        out->stream = fopen(env, "w");
    }

}

extern "C" void ggml_graph_profile_start(struct ggml_cgraph *cg, int n_threads)
{
    if (!cg->prof) { ggml_graph_profile_init(cg, n_threads); }
    if (!cg->prof) { return; }
}

static inline int ggml_profile_format_tensor_dims(char *str, struct ggml_tensor *t)
{
    return sprintf(str, "%d:%d:%d:%d",
        (int) t->ne[0], (int) t->ne[1], (int) t->ne[3], (int) t->ne[3]);
}

static inline void ggml_profile_format_op_dims(char *str, struct ggml_tensor *t)
{
    char *p = str;

    // append src0 and src1 (if any)
    if (t->src[0]) {
       p += ggml_profile_format_tensor_dims(p, t->src[0]);

       for (int i = 1; i < GGML_MAX_SRC && t->src[i]; i++) {
           p += sprintf(p, " x ");
           p += ggml_profile_format_tensor_dims(p, t->src[i]);
       }

       p += sprintf(p, " -> ");
    }

    // format self dims separately for better visual alignment
    char self[64];
    ggml_profile_format_tensor_dims(self, t);

    p += sprintf(p, "%12s", self);
}

static inline void ggml_profile_format_op_types(char *str, struct ggml_tensor *t)
{
    char *p = str;

    // append src0 and src1 (if any)
    if (t->src[0]) {
       p += sprintf(p, "%s", ggml_type_name(t->src[0]->type));

       for (int i = 1; i < GGML_MAX_SRC && t->src[i]; i++) {
           p += sprintf(p, " x ");
           p += sprintf(p, "%s", ggml_type_name(t->src[i]->type));
       }

       p += sprintf(p, " -> ");
    }

    p += sprintf(p, "%3s", ggml_type_name(t->type));
}

static inline void ggml_profile_format_op_names(char *str, const struct ggml_tensor *t)
{
    char *p = str;

    // append src0 and src1 (if any)
    if (t->src[0]) {
       p += sprintf(p, "%s", t->src[0]->name);

       for (int i = 1; i < GGML_MAX_SRC && t->src[i]; i++) {
           p += sprintf(p, " x ");
           p += sprintf(p, "%s", t->src[i]->name);
       }

       p += sprintf(p, " -> ");
    }

    p += sprintf(p, "%s", t->name);
}


extern "C" void ggml_graph_profile_finish(struct ggml_cgraph *cg, int n_threads)
{
    if (!cg->prof) { return; }

    ggml_profile_output *out = cg->prof->output;

    fprintf(out->stream, "%s| node idx | op name | proc (nsec) | sync (nsec) | total (nsec) | op dims | op types | tensor names |\n", out->prefix);
    fprintf(out->stream, "%s| -------: | :------ | ----------: | ----------: | -----------: | ------: | -------: | -----------: |\n", out->prefix);

    char dims[64 * GGML_MAX_SRC];
    char types[16 * GGML_MAX_SRC];
    char names[128 * GGML_MAX_SRC];

    for (int i = 0; i < cg->n_nodes; i++) {
        uint64_t p_nsec = 0;
        uint64_t s_nsec = 0;
        uint64_t t_nsec = 0;

        // add up per thread counters and reset them
        for (int t=0; t < n_threads; t++) {
            ggml_profile_timing &timing = cg->prof->timing[i][t];

            p_nsec += timing.nsec[GGML_PROF_OP_SYNC] - timing.nsec[GGML_PROF_OP_START];
            s_nsec += timing.nsec[GGML_PROF_OP_END]  - timing.nsec[GGML_PROF_OP_SYNC];
            t_nsec += timing.nsec[GGML_PROF_OP_END]  - timing.nsec[GGML_PROF_OP_START];

            timing.nsec[GGML_PROF_OP_START] = 0;
            timing.nsec[GGML_PROF_OP_SYNC]  = 0;
            timing.nsec[GGML_PROF_OP_END]   = 0;
        }

        ggml_profile_format_op_dims(dims, cg->nodes[i]);
        ggml_profile_format_op_types(types, cg->nodes[i]);
        ggml_profile_format_op_names(names, cg->nodes[i]);

        fprintf(out->stream, "%s| %04d | %10s | %10lu | %10lu | %10lu | %46s | %22s | %20s |\n", out->prefix,
            i, ggml_op_name(cg->nodes[i]->op),
            (unsigned long) p_nsec, (unsigned long) s_nsec, (unsigned long) t_nsec,
            dims, types, names);
    }
    fprintf(out->stream, "%s   \n", out->prefix); // empty line to split tables
}

extern "C" void ggml_graph_profile_free(struct ggml_cgraph *cg)
{
    if (!cg->prof) { return; }

    ggml_profile_output *out = cg->prof->output;
    if (out->stream != stderr) {
        fclose(out->stream);
    }

    free(cg->prof); cg->prof = nullptr;
}

extern "C" void ggml_graph_profile_event(const struct ggml_cgraph *cg, enum ggml_profile_event e, int node_n, int ith)
{
    if (!cg->prof) { return; }

    using clock = std::chrono::high_resolution_clock;

    ggml_profile_timing &timing = cg->prof->timing[node_n][ith];
    timing.nsec[e] = std::chrono::nanoseconds(clock::now().time_since_epoch()).count();
}

#endif // GGML_GRAPH_PROFILER
