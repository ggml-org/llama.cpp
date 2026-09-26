#pragma once

#include "command-program-executor.h"
#include "dispatch/command-program-bindings.h"
#include "dispatch/command-program.h"
#include "runtime/graph-replay.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace ggml::hrx {

struct PreparedCommandProgramCacheStats {
    uint64_t builds = 0;
    uint64_t hits   = 0;
};

struct PreparedCommandProgramCacheExecutionResult {
    bool                success = false;
    Status              status;
    HrxGraphReplayEvent graph_replay_event = HrxGraphReplayEvent::Disabled;
    std::string         graph_replay_ineligible_reason;
    uint64_t            graph_replay_build_ns                     = 0;
    uint64_t            graph_replay_launch_ns                    = 0;
    uint64_t            graph_replay_total_ns                     = 0;
    size_t              graph_replay_dispatches                   = 0;
    bool                graph_replay_transient_allocation_changed = false;
};

class PreparedCommandProgramCache {
  public:
    bool execute(const CommandProgramExecutionContext & context,
                 uint64_t                               graph_uid,
                 const std::string &                    command_shape,
                 const CommandProgram &                 commands,
                 const CommandProgramBindings &         bindings);

    PreparedCommandProgramCacheExecutionResult execute_with_result(const CommandProgramExecutionContext & context,
                                                                   uint64_t                               graph_uid,
                                                                   const std::string &                    command_shape,
                                                                   const CommandProgram &                 commands,
                                                                   const CommandProgramBindings &         bindings);

    PreparedCommandProgramCacheStats stats() const;

    void clear();

  private:
    struct Key {
        uint64_t graph_uid          = 0;
        uint64_t target_hash        = 0;
        uint64_t command_shape_hash = 0;
        uint64_t bindings_hash      = 0;

        bool operator==(const Key & other) const {
            return graph_uid == other.graph_uid && target_hash == other.target_hash &&
                   command_shape_hash == other.command_shape_hash && bindings_hash == other.bindings_hash;
        }
    };

    struct KeyHash {
        size_t operator()(const Key & key) const;
    };

    Key cache_key(uint64_t                               graph_uid,
                  const CommandProgramExecutionContext & context,
                  const std::string &                    command_shape,
                  const CommandProgramBindings &         bindings) const;

    struct Entry {
        std::mutex             mutex;
        PreparedCommandProgram program;
        RecordedCommandGraph   recorded;
        bool                   has_program = false;
    };

    void record_build();
    void record_hit();

    mutable std::mutex                                       mutex_;
    std::unordered_map<Key, std::shared_ptr<Entry>, KeyHash> programs_;
    PreparedCommandProgramCacheStats                         stats_;
};

}  // namespace ggml::hrx
