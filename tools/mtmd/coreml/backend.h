#pragma once

// !!! Internal header for the mtmd CoreML backend !!!

#define MTMD_INTERNAL_HEADER

#include <cstdint>
#include <string>
#include <vector>

namespace mtmd_coreml::backend {

// dtype tag for a single input tensor in the generic predict call.
enum dtype {
    DTYPE_F32,
    DTYPE_I32,
};

// Description of one input fed to MLModel.predict. The buffer pointed to by
// `data` must remain valid for the duration of the predict() call.
struct input_tensor {
    const char *         name;
    const void *         data;
    std::vector<int64_t> shape;
    dtype                kind;
};

// Load a .mlmodelc bundle. Returns an opaque handle (bridged-retained MLModel)
// or nullptr on failure.
void * load   (const char * mlmodelc_path);
void   unload (void * handle);

// Generic prediction: assemble inputs by name, fetch one named output,
// and memcpy it into `out_buf` (caller-owned). Returns false on shape
// mismatch, missing output, or any MLModel error.
bool predict_single_output(void *                            handle,
                           const std::vector<input_tensor> & inputs,
                           const char *                      out_name,
                           float *                           out_buf);

} // namespace mtmd_coreml::backend
