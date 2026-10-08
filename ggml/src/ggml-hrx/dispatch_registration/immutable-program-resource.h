#pragma once

#include "dispatch-registry.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ggml::hrx {

struct ImmutableProgramResourceSpec {
    std::string                  key;
    const std::vector<uint8_t> & data;
    size_t                       alignment = 256;
};

ValueId require_immutable_program_resource(const DispatchMatchContext &         context,
                                           DispatchMatch &                      match,
                                           const ImmutableProgramResourceSpec & spec);

}  // namespace ggml::hrx
