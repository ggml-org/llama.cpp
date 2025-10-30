/*
 * Axelera Compiler Integration
 *
 * This header defines the graph compilation and planning interface for the Axelera backend.
 */

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// Forward declarations
struct ggml_backend_axelera_graph_plan;
struct ggml_axelera_device_context;

// Graph plan handle (opaque)
typedef struct ggml_backend_axelera_graph_plan* ggml_axelera_graph_plan_t;

// Create a graph plan (compiles the graph)
ggml_backend_graph_plan_t ggml_axelera_graph_plan_create(
    ggml_backend_t backend,
    const ggml_cgraph* cgraph);

// Execute a compiled graph plan
ggml_status ggml_axelera_graph_plan_compute(
    ggml_backend_t backend,
    ggml_backend_graph_plan_t plan);

// Free a graph plan
void ggml_axelera_graph_plan_free(
    ggml_backend_t backend,
    ggml_backend_graph_plan_t plan);

#ifdef __cplusplus
}
#endif
