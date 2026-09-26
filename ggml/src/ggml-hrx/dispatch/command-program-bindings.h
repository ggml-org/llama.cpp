#pragma once

#include "graph/value-map.h"
#include "status.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ggml::hrx {

struct CommandProgramBinding {
    // A buffer is directly bindable by an HRX command program. Host data requires residency or staging before
    // execution. These are alternate storage forms and should not both be populated.
    ValueId      value;
    hrx_buffer_t buffer     = nullptr;
    size_t       offset     = 0;
    size_t       length     = 0;
    uint64_t     identity   = 0;
    uint64_t     generation = 0;
    size_t       capacity   = 0;
    void *       host_data  = nullptr;
    bool         weight     = false;

    bool requires_materialization() const { return host_data != nullptr; }
};

struct CommandProgramBindingsFingerprint {
    std::string value;
};

struct CommandProgramBindingsHash {
    uint64_t value = 0;
};

class CommandProgramBindings {
  public:
    static CommandProgramBindings from_value_map(const ValueMap & values);
    static CommandProgramBindings from_bindings(std::vector<CommandProgramBinding> bindings,
                                                const Status &                     errors = {});

    const CommandProgramBinding * find(ValueId value) const;

    const std::vector<CommandProgramBinding> & bindings() const { return bindings_; }

    bool valid() const { return status.success(); }

    Status status;

  private:
    std::vector<CommandProgramBinding> bindings_;
};

CommandProgramBindingsHash        command_program_bindings_hash(const CommandProgramBindings & bindings);
CommandProgramBindingsFingerprint command_program_bindings_fingerprint(const CommandProgramBindings & bindings);

}  // namespace ggml::hrx
