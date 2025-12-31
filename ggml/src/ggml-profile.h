#pragma once

#include "ggml-impl.h"

// GGML internal header

#ifdef __cplusplus
extern "C" {
#endif

// op profile events & timing (per op / per thread)
enum ggml_profile_event {
    GGML_PROF_OP_START,
    GGML_PROF_OP_SYNC,
    GGML_PROF_OP_END
};

struct ggml_profile_timing {
    uint64_t nsec[GGML_PROF_OP_END + 1]; // event times in nsec
};

struct ggml_profile_output;

struct ggml_profile_data {
    struct ggml_profile_output *output;
    struct ggml_profile_timing ** timing; // per op / per thread timing
};

// check if profiling is enabled for this graph
static inline bool ggml_graph_profile_enabled(const struct ggml_cgraph *cg)
{
    return cg->prof != NULL;
}

// get pointer to the timing data for specific node / thread
// can be used by the backends to populate data collected internally
static inline struct ggml_profile_timing * ggml_graph_profile_timing(const struct ggml_cgraph *cg, int node_n, int ith)
{
    if (!cg->prof) { return NULL; }
    return &cg->prof->timing[node_n][ith];
}

#ifndef GGML_GRAPH_PROFILER

// Stub out all profiler functions

static inline void ggml_graph_profile_init(struct ggml_cgraph *cg, int n_threads)
{
    GGML_UNUSED(cg);
    GGML_UNUSED(n_threads);
}

static inline void ggml_graph_profile_start(struct ggml_cgraph *cg, int n_threads)
{
    GGML_UNUSED(cg);
    GGML_UNUSED(n_threads);
}

static inline void ggml_graph_profile_finish(struct ggml_cgraph *cg, int n_threads)
{
    GGML_UNUSED(cg);
    GGML_UNUSED(n_threads);
}

static inline void ggml_graph_profile_free(struct ggml_cgraph *cg)
{
    GGML_UNUSED(cg);
}

static inline void ggml_graph_profile_event(const struct ggml_cgraph *cg, enum ggml_profile_event e, int node_n, int ith)
{
    GGML_UNUSED(cg);
    GGML_UNUSED(e);
    GGML_UNUSED(node_n);
    GGML_UNUSED(ith);
}

#else

void ggml_graph_profile_init(struct ggml_cgraph *cg, int n_threads);
void ggml_graph_profile_start(struct ggml_cgraph *cg, int n_threads);
void ggml_graph_profile_finish(struct ggml_cgraph *cg, int n_threads);
void ggml_graph_profile_free(struct ggml_cgraph *cg);
void ggml_graph_profile_event(const struct ggml_cgraph *cg, enum ggml_profile_event e, int node_n, int ith);

#endif // GGML_GRAPH_PROFILER

#ifdef __cplusplus
}
#endif
