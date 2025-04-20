// tensor-footprint-estimation.h
#pragma once

#include <stdio.h>
#include <stdint.h>

// Forward declaration for ggml_cgraph
struct ggml_cgraph;

#ifdef __cplusplus
extern "C" {
#endif

// Log the entire computation graph to CSV
void ggml_log_graph(struct ggml_cgraph* cgraph);

#ifdef __cplusplus
}
#endif