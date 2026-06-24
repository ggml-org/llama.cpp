#pragma once

#include <string>
#include <vector>

// Ref: https://www.modelscope.cn/docs/models/download

namespace ms_cache {

struct ms_file {
    std::string path;        // repo-relative file path (e.g. "Q4_K_M/model.gguf")
    std::string url;         // full download URL
    std::string local_path;  // where the blob lives on disk (blobs/<sha256>)
    std::string final_path;  // where the symlink lives (snapshots/master/<path>)
    std::string oid;         // Sha256 hash from ModelScope API (64 hex chars)
    std::string repo_id;     // e.g. "Qwen/Qwen3-0.6B-GGUF"
};

using ms_files = std::vector<ms_file>;

// Get files from ModelScope API
ms_files get_repo_files(
    const std::string & repo_id,
    const std::string & token
);

// Scan local ModelScope cache
ms_files get_cached_files(const std::string & repo_id = {});

// Create snapshot path (link or move/copy) and return it
std::string finalize_file(const ms_file & file);

} // namespace ms_cache
