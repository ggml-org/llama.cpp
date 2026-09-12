#include "prepared-command-program-cache.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

namespace ggml::hrx {
namespace {

static void mix_hash(uint64_t & hash, uint64_t value) {
    hash ^= value;
    hash *= UINT64_C(1099511628211);
}

static uint64_t hash_text(const char * text) {
    uint64_t hash = UINT64_C(1469598103934665603);
    if (text == nullptr) {
        return hash;
    }
    while (*text != 0) {
        mix_hash(hash, static_cast<unsigned char>(*text));
        ++text;
    }
    return hash;
}

static uint64_t hash_string(const std::string & text) {
    uint64_t hash = UINT64_C(1469598103934665603);
    for (const char c : text) {
        mix_hash(hash, static_cast<unsigned char>(c));
    }
    return hash;
}

static void apply_graph_replay_result(PreparedCommandProgramCacheExecutionResult & result,
                                      const RecordedCommandGraphExecutionResult &  replay) {
    result.graph_replay_event                        = replay.event;
    result.graph_replay_ineligible_reason            = replay.ineligible_reason;
    result.graph_replay_build_ns                     = replay.build_ns;
    result.graph_replay_launch_ns                    = replay.launch_ns;
    result.graph_replay_total_ns                     = replay.total_ns();
    result.graph_replay_dispatches                   = replay.dispatch_count;
    result.graph_replay_transient_allocation_changed = replay.transient_allocation_changed;
}

static bool graph_replay_should_fallback(HrxGraphReplayEvent event) {
    return event == HrxGraphReplayEvent::Ineligible || event == HrxGraphReplayEvent::BuildFailed;
}

}  // namespace

size_t PreparedCommandProgramCache::KeyHash::operator()(const Key & key) const {
    uint64_t hash = UINT64_C(1469598103934665603);
    mix_hash(hash, key.graph_uid);
    mix_hash(hash, key.target_hash);
    mix_hash(hash, key.command_shape_hash);
    mix_hash(hash, key.bindings_hash);
    return static_cast<size_t>(hash);
}

PreparedCommandProgramCache::Key PreparedCommandProgramCache::cache_key(uint64_t graph_uid,
                                                                        const CommandProgramExecutionContext & context,
                                                                        const std::string &            command_shape,
                                                                        const CommandProgramBindings & bindings) const {
    return {
        graph_uid,
        hash_text(context.target),
        hash_string(command_shape),
        command_program_bindings_hash(bindings).value,
    };
}

bool PreparedCommandProgramCache::execute(const CommandProgramExecutionContext & context,
                                          uint64_t                               graph_uid,
                                          const std::string &                    command_shape,
                                          const CommandProgram &                 commands,
                                          const CommandProgramBindings &         bindings) {
    return execute_with_result(context, graph_uid, command_shape, commands, bindings).success;
}

PreparedCommandProgramCacheExecutionResult PreparedCommandProgramCache::execute_with_result(
    const CommandProgramExecutionContext & context,
    uint64_t                               graph_uid,
    const std::string &                    command_shape,
    const CommandProgram &                 commands,
    const CommandProgramBindings &         bindings) {
    PreparedCommandProgramCacheExecutionResult result;
    if (graph_uid == 0 || !commands.valid() || !bindings.valid()) {
        result.graph_replay_event             = HrxGraphReplayEvent::Ineligible;
        result.graph_replay_ineligible_reason = "uncached_graph";
        PreparedCommandProgram prepared       = prepare_command_program(context, commands, bindings);
        if (!prepared.valid()) {
            result.status.append(prepared.status);
            return result;
        }
        result.success = bind_and_execute_prepared_command_program(context, commands, bindings, prepared);
        if (!result.success) {
            result.status.log("execute uncached HRX command program failed");
        }
        return result;
    }

    const Key              key = cache_key(graph_uid, context, command_shape, bindings);
    std::shared_ptr<Entry> entry;
    bool                   created_entry = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto                        found = programs_.find(key);
        if (found == programs_.end()) {
            entry = std::make_shared<Entry>();
            programs_.emplace(key, entry);
            created_entry = true;
        } else {
            entry = found->second;
        }
    }

    std::lock_guard<std::mutex> entry_lock(entry->mutex);
    if (entry->has_program && entry->program.valid()) {
        record_hit();
        const RecordedCommandGraphExecutionResult replay =
            bind_and_launch_recorded_command_graph(context, commands, bindings, entry->program, entry->recorded);
        apply_graph_replay_result(result, replay);
        if (replay.success) {
            result.success = true;
            return result;
        }
        if (!graph_replay_should_fallback(replay.event)) {
            result.status.append(replay.status);
            if (result.status.success()) {
                result.status.log("execute cached HRX graph replay failed");
            }
            return result;
        }
        result.success = bind_and_execute_prepared_command_program(context, commands, bindings, entry->program);
        if (!result.success) {
            result.status.log("execute cached HRX command program failed");
        }
        return result;
    }

    PreparedCommandProgram prepared = prepare_command_program(context, commands, bindings);
    if (!prepared.valid()) {
        if (created_entry) {
            std::lock_guard<std::mutex> lock(mutex_);
            const auto                  found = programs_.find(key);
            if (found != programs_.end() && found->second == entry && !entry->has_program) {
                programs_.erase(found);
            }
        }
        result.status.append(prepared.status);
        return result;
    }
    entry->program     = std::move(prepared);
    entry->has_program = true;
    record_build();

    const RecordedCommandGraphExecutionResult replay =
        bind_and_launch_recorded_command_graph(context, commands, bindings, entry->program, entry->recorded);
    apply_graph_replay_result(result, replay);
    if (replay.success) {
        result.success = true;
        return result;
    }
    if (!graph_replay_should_fallback(replay.event)) {
        result.status.append(replay.status);
        if (result.status.success()) {
            result.status.log("execute prepared HRX graph replay failed");
        }
        return result;
    }
    result.success = bind_and_execute_prepared_command_program(context, commands, bindings, entry->program);
    if (!result.success) {
        result.status.log("execute prepared HRX command program failed");
    }
    return result;
}

PreparedCommandProgramCacheStats PreparedCommandProgramCache::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

void PreparedCommandProgramCache::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    programs_.clear();
}

void PreparedCommandProgramCache::record_build() {
    std::lock_guard<std::mutex> lock(mutex_);
    ++stats_.builds;
}

void PreparedCommandProgramCache::record_hit() {
    std::lock_guard<std::mutex> lock(mutex_);
    ++stats_.hits;
}

}  // namespace ggml::hrx
