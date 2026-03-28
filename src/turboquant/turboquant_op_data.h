#pragma once

#include "turboquant_impl.h"
#include <cstdint>
#include <cstddef>

namespace turboquant {

// Data structure for passing to custom op
struct TurboQuantOpData {
    TurboQuantKVCache * tq;           // TurboQuant instance
    uint8_t * quantized_buffer;       // Buffer to write packed indices to
    size_t bytes_per_vector;          // Bytes per quantized vector
    size_t buffer_offset;             // Current offset in buffer (for multiple calls)
    
    // Constructor
    TurboQuantOpData(TurboQuantKVCache * tq_, uint8_t * buffer_, size_t bytes_per_vector_)
        : tq(tq_), quantized_buffer(buffer_), bytes_per_vector(bytes_per_vector_), buffer_offset(0) {}
};

} // namespace turboquant