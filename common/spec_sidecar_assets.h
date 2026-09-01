// SPDX-License-Identifier: MIT
#pragma once

#include "spec_sidecar.h"

#include <cstdint>
#include <string>
#include <vector>

// Generates model-derived sidecar artifacts without Python or external tools.
// The caller must still provide a provider library built with llama-server.
// Artifacts are committed atomically and reused when their source/cache key
// matches. cache_root may be empty to use the normal llama.cpp cache.
bool common_spec_sidecar_prepare_artifacts(
        const common_spec_sidecar_profile & profile,
        const std::string & target_path,
        const std::string & draft_path,
        const std::string & cache_root,
        common_spec_sidecar_paths & paths,
        bool & cache_hit,
        std::string & error);

// Exposed for a small hermetic integrity test. The built-in map is the pinned,
// Apache-2.0 Qwen3.8 draft vocabulary used by the qualified sidecars.
bool common_spec_sidecar_builtin_draft_vocab_ids(
        std::vector<int32_t> & ids,
        std::string & error);
