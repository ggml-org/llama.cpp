// ggml-graph-logging.c
#include "ggml-graph-logging.h"
#include <stdlib.h> // for getenv
#include <stdio.h>  // for fprintf, stderr
#include <stdbool.h>
#include <string.h>
#include <inttypes.h>

// Include the full definition of ggml structs
#include "ggml.h"
#include "ggml-impl.h" // This includes the full definition of ggml_cgraph


//
// Graph logging
//
// This is a simple logging system for the graph of a GGML model.
//
// The graph is logged to a CSV file.
//
// The CSV file contains the following columns:
//
// - node_id: The unique identifier for the node.
// - name: The name of the node.
// - op: The operation performed by the node.
// - dim0, dim1, dim2, dim3: The dimensions of the node.
// - bytes: The number of bytes in the node.
// - flags: The flags of the node.
// - src0..srcN: The source nodes of the node.
//
// The CSV file is written to the current working directory.
// The CSV file is overwritten if it already exists.
// The program will terminate after the graph is logged.
//
// The graph is logged when the environment variable GGML_LOG_GRAPH is set to 1.
// The filename for the log file can be set using the environment variable GGML_LOG_GRAPH_FILENAME.
//
// The graph is logged using the ggml_log_graph function.
//


static FILE* ggml_graph_log_init(const char* filename) {    
    FILE* file = fopen(filename, "w");
    if (file) {
        fprintf(stderr, "%s: Graph logging enabled, will write to '%s'\n", __func__, filename);
        
        // Write CSV header - now with dynamic source columns
        fprintf(file, "node_id,name,op,dim0,dim1,dim2,dim3,bytes,flags");
        
        // Add source columns based on GGML_MAX_SRC
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            fprintf(file, ",src%d", i);
        }
        fprintf(file, "\n");
    } else {
        fprintf(stderr, "%s: Error: Failed to open graph file '%s' for writing.\n", __func__, filename);
    }
    return file;
}

static void ggml_graph_log_free(FILE* file) {
    if (file) {
        fclose(file);
    }
}

static void write_tensor_to_csv(struct ggml_tensor* tensor, const char* custom_flags, FILE* file) {
    if (!tensor || !file) return;
    
    // Get flags
    const char* flags = custom_flags ? custom_flags : "-";
    if (!custom_flags) {
        if (tensor->flags & GGML_TENSOR_FLAG_PARAM) {
            flags = "PARAM";
        } else if (tensor->flags & GGML_TENSOR_FLAG_INPUT) {
            flags = "INPUT";
        } else if (tensor->flags & GGML_TENSOR_FLAG_OUTPUT) {
            flags = "OUTPUT";
        }
    }
    
    // Calculate size in bytes
    size_t total_size = ggml_nbytes(tensor);
    
    // Write base tensor info
    fprintf(file, 
            "%p,%s,%s,%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 ",%.2f,%s",
            (void*)tensor,                               // node_id (pointer for uniqueness)
            tensor->name[0] ? tensor->name : "unnamed",  // name
            ggml_op_name(tensor->op),                    // op
            tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3], // dimensions
            (double)total_size,                          // bytes
            flags);                                      // flags
    
    // Write all source tensors dynamically
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        fprintf(file, ",%p", (void*)tensor->src[i]);
    }
    
    fprintf(file, "\n");
}

void ggml_log_graph(struct ggml_cgraph* cgraph) {
    const char* log_graph_env = getenv("GGML_LOG_GRAPH");
    if (!log_graph_env || (strcmp(log_graph_env, "1") != 0 && strcmp(log_graph_env, "true") != 0)) {
        return;
    }    

    // Get the filename from the environment variable, or use the default
    const char* filename_env = getenv("GGML_LOG_GRAPH_FILENAME");
    const char* filename = filename_env ? filename_env : "ggml_graph.csv";

    FILE* file = ggml_graph_log_init(filename);
    if (!file || !cgraph) {
        return;
    }

    // Process all nodes in the graph
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor* node = cgraph->nodes[i];
        write_tensor_to_csv(node, NULL, file);
    }
    
    // Process all leaf nodes as well
    for (int i = 0; i < cgraph->n_leafs; i++) {
        struct ggml_tensor* leaf = cgraph->leafs[i];
        if (!leaf) continue;
        
        // Skip if already included in nodes
        bool already_processed = false;
        for (int j = 0; j < cgraph->n_nodes; j++) {
            if (cgraph->nodes[j] == leaf) {
                already_processed = true;
                break;
            }
        }
        if (already_processed) continue;
        
        write_tensor_to_csv(leaf, "LEAF", file);
    }
    
    // Flush the file to ensure all data is written
    fflush(file);
    ggml_graph_log_free(file);
    
    fprintf(stderr, "Graph logging complete: %d nodes and %d leafs written to CSV file. Terminating.\n", 
            cgraph->n_nodes, cgraph->n_leafs);    
    exit(0);
}

