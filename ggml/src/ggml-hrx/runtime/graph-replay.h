#pragma once

#include <chrono>
#include <cstdint>

namespace ggml::hrx {

enum class HrxGraphReplayEvent {
    Disabled,
    Ineligible,
    MissBuild,
    Hit,
    RebuildTransient,
    BuildFailed,
    LaunchFailed,
};

inline const char * hrx_graph_replay_event_name(HrxGraphReplayEvent event) {
    switch (event) {
        case HrxGraphReplayEvent::Disabled:
            return "disabled";
        case HrxGraphReplayEvent::Ineligible:
            return "ineligible";
        case HrxGraphReplayEvent::MissBuild:
            return "miss_build";
        case HrxGraphReplayEvent::Hit:
            return "hit";
        case HrxGraphReplayEvent::RebuildTransient:
            return "rebuild_transient";
        case HrxGraphReplayEvent::BuildFailed:
            return "build_failed";
        case HrxGraphReplayEvent::LaunchFailed:
            return "launch_failed";
    }
    return "unknown";
}

inline uint64_t hrx_graph_replay_now_ns() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

}  // namespace ggml::hrx
