#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ggml::hrx {

enum class IQCodebookResource {
    IQ1,
    IQ2XXS,
    IQ2XS,
    IQ3XXS,
    IQ2S,
    IQ3S,
};

struct IQCodebookResourceSpec {
    const char *                 key;
    const std::vector<uint8_t> & data;
    size_t                       alignment;
};

const IQCodebookResourceSpec & iq_codebook_resource_spec(IQCodebookResource resource);

}  // namespace ggml::hrx
