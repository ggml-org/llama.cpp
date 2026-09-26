#pragma once

#include "command-program.h"

#include <vector>

namespace ggml::hrx {

class TransientAllocator {
  public:
    static TransientPlan allocate(const Graph &                graph,
                                  const CommandPlan &          command_plan,
                                  const std::vector<Command> & initialization_commands,
                                  const std::vector<Command> & commands,
                                  CompletionCounterPlan &      completion_counters,
                                  Status &                     errors);

    static bool allocations_can_overlap(const CommandProgram &      program,
                                        const TransientAllocation & lhs,
                                        const TransientAllocation & rhs);
};

}  // namespace ggml::hrx
