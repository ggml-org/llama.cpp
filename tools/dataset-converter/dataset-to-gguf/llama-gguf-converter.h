#pragma once

#include "llama.h"  // For struct llama_model

// Class encapsulating the high-level logic for converting
// input data to the GGUF format.
struct llama_gguf_converter {
    // Default constructor.
    llama_gguf_converter() = default;

    // Method to execute the conversion process.
    // params: A structure containing all necessary parameters for conversion.
    // Returns true on successful conversion, false on error.
    bool llama_gguf_converter_convert(const struct common_params & params, const struct llama_model * model);
};
