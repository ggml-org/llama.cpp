#pragma once

#include <cstddef>

// Optional probe failures use the ordinary pool; publish granularity only after both queries succeed.
template<typename Attribute, typename Granularity>
static bool ggml_cuda_probe_vmm(Attribute attribute, Granularity query_granularity, std::size_t & granularity) {
    int supported = 0;
    std::size_t bytes = 0;
    if (!attribute(supported) || !supported || !query_granularity(bytes) || !bytes) return false;
    granularity = bytes;
    return true;
}
