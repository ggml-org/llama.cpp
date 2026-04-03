#pragma once

#include "ggml-backend.h"
#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

//
// Profiler
//

// Profile event types
enum ggml_profile_event_type {
    GGML_PROFILE_EVENT_OP,    // single operation execution (computation kernel)
    GGML_PROFILE_EVENT_COPY,  // data transfer between devices
};

// A single profiling record representing a timed interval
typedef struct ggml_profile_record {
    enum ggml_profile_event_type type;
    const char *                 name;        // operation name (e.g., "mul_mat", "copy_H2D")
    int                          backend_id;  // scheduler's backend index (0 = highest priority)
    int                          split_id;    // which graph split (0..n_splits-1)
    uint64_t                     start_ns;    // start timestamp in nanoseconds
    uint64_t                     end_ns;      // end timestamp in nanoseconds
    uint64_t                     bytes;       // bytes transferred (for copy) or tensor size (for ops)
    const char *                 extra;       // fusion name for fused ops, or NULL
    int64_t                      ne_src0[4];  // src[0] tensor dimensions (e.g. weight matrix for MUL_MAT)
    int64_t                      ne_src1[4];  // src[1] tensor dimensions (e.g. input matrix for MUL_MAT)
    int64_t                      ne_src2[4];  // src[2] tensor dimensions (e.g. ids for MUL_MAT_ID)
    int                          type_src0;   // src[0] tensor type (ggml_type), -1 if N/A
    int                          type_src1;   // src[1] tensor type (ggml_type), -1 if N/A
    int                          type_src2;   // src[2] tensor type (ggml_type), -1 if N/A
    int                          sub_op;      // sub-operation (ggml_unary_op or ggml_glu_op), -1 if N/A
} ggml_profile_record;

// Backend profiler interface - each backend optionally implements this
// to provide fine-grained operation timing
struct ggml_backend_profiler {
    void * context;  // backend-specific profiler context

    // Enable or disable profiling on this backend
    void (*enable)(void * context, bool enable);

    // Clear all recorded data
    void (*reset)(void * context);

    // Set the current split ID (called by scheduler before graph_compute)
    void (*set_split_id)(void * context, int split_id);

    // Get recorded profiling data
    // Returns the number of records; sets *out to point to internal storage
    // The returned pointer remains valid until the next reset or disable call
    int (*get_records)(void * context, const ggml_profile_record ** out);

    // Free the profiler context
    void (*free_context)(void * context);
};

typedef struct ggml_backend_profiler * ggml_backend_profiler_t;

// Register a profiler on a backend (called by backend during init)
// The profiler is owned by the backend and will be freed when the backend is freed
GGML_API void ggml_backend_set_profiler(ggml_backend_t backend, ggml_backend_profiler_t profiler);

// Get the profiler associated with a backend (returns NULL if none)
GGML_API ggml_backend_profiler_t ggml_backend_get_profiler(ggml_backend_t backend);

//
// Scheduler profiling API
//

// Enable or disable profiling on a scheduler
// When enabled, the scheduler will:
//   - Time data copy operations between backends
//   - Enable profiling on all backends that support it
//   - Collect profiling records from all backends after each graph compute
GGML_API void ggml_backend_sched_set_profiling(ggml_backend_sched_t sched, bool enable);

// Check if profiling is enabled on a scheduler
GGML_API bool ggml_backend_sched_get_profiling(ggml_backend_sched_t sched);

// Get profiling data from the last graph compute
// Records are owned by the scheduler; valid until the next compute or reset
// Returns the number of records
GGML_API int ggml_backend_sched_get_profiling_records(ggml_backend_sched_t sched, const ggml_profile_record ** records);

// Print a human-readable summary of the last profiling run to stdout
// Groups records by operation name and shows total/count/min/max/avg time
GGML_API void ggml_backend_sched_print_profiling(ggml_backend_sched_t sched);

// Reset profiling data (clear all recorded data)
GGML_API void ggml_backend_sched_reset_profiling(ggml_backend_sched_t sched);

// Get current time in nanoseconds (for manual profiling if needed)
GGML_API uint64_t ggml_profiler_time_ns(void);

// Export profiling data as JSON to a file
// Returns 0 on success, -1 on error
GGML_API int ggml_backend_sched_export_profiling_json(ggml_backend_sched_t sched, const char * filepath);

// Export profiling data as JSON to a FILE pointer
GGML_API int ggml_backend_sched_write_profiling_json(ggml_backend_sched_t sched, FILE * fp);

// Export profiling data as plain text statistics to a file
// Returns 0 on success, -1 on error
GGML_API int ggml_backend_sched_export_profiling_text(ggml_backend_sched_t sched, const char * filepath);

// Export profiling data as plain text statistics to a FILE pointer
GGML_API int ggml_backend_sched_write_profiling_text(ggml_backend_sched_t sched, FILE * fp);

#ifdef __cplusplus
}
#endif
