#pragma once

// thin C wrapper around the private AppleNeuralEngine framework (_ANEInMemoryModel)

#include <IOSurface/IOSurfaceRef.h>

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_ANE_MAX_NAME 64

// layout of one program input/output, as reported by the compiled model
// the logical shape is [n, c, h, w]; rows of w elements are row_stride bytes apart
typedef struct {
    char    name[GGML_ANE_MAX_NAME]; // MIL symbol name, "@output" suffix removed
    int64_t n, c, h, w;
    size_t  batch_stride;
    size_t  plane_stride;
    size_t  row_stride;
    size_t  esize;                   // 2 = fp16, 4 = fp32
} ggml_ane_io_desc;

typedef struct ggml_ane_program ggml_ane_program;

bool         ggml_ane_runtime_available(void);
const char * ggml_ane_runtime_arch(void);

IOSurfaceRef ggml_ane_surface_create(size_t size);
// surface over existing page-aligned memory (no copy), NULL if not possible
IOSurfaceRef ggml_ane_surface_wrap(void * ptr, size_t size);

// compile (or reuse the OS compile cache) and load a MIL program with a single Core ML weight blob
// on failure returns NULL and writes a message to err
ggml_ane_program * ggml_ane_program_create(const char * mil, size_t mil_len, const void * weights, size_t weights_len, char * err, size_t err_len);
void               ggml_ane_program_free(ggml_ane_program * prog);

// drop the OS's compiled copy of a loaded program (e.g. when it is replaced by a larger version)
void               ggml_ane_program_purge_compiled(ggml_ane_program * prog);

// a program has one procedure per MIL function (procedure index = position of the function name in sorted order)
int ggml_ane_program_n_procedures(const ggml_ane_program * prog);
int ggml_ane_program_procedure   (const ggml_ane_program * prog, const char * name); // -1 if not found

// inputs and outputs of a procedure, in symbol order
int                      ggml_ane_program_n_inputs (const ggml_ane_program * prog, int proc);
int                      ggml_ane_program_n_outputs(const ggml_ane_program * prog, int proc);
const ggml_ane_io_desc * ggml_ane_program_input (const ggml_ane_program * prog, int proc, int i);
const ggml_ane_io_desc * ggml_ane_program_output(const ggml_ane_program * prog, int proc, int i);

// each surface must be at least batch_stride bytes; data starts at offset 0 (the ANE ignores I/O offsets)
bool ggml_ane_program_eval(ggml_ane_program * prog, int proc, IOSurfaceRef * inputs, IOSurfaceRef * outputs, char * err, size_t err_len);

#ifdef __cplusplus
}
#endif
