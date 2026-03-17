#pragma once

#include <filesystem>
#include <string>
#include <utility>
#include <vector>

// Ref: https://huggingface.co/docs/hub/local-cache.md

namespace hf_cache {

struct hf_file {
    std::string path;
    std::string url;
    std::string local_path;
    std::string link_path;
    std::string oid;
    std::string repo_id;
};

using hf_files = std::vector<hf_file>;

// Get files from HF API
hf_files get_repo_files(
    const std::string & repo_id,
    const std::string & bearer_token
);

hf_files get_cached_files(const std::string & repo_id);

// Create symlink if link_path is set and returns the snapshot path
std::string finalize_file(const hf_file & file);

// TODO: Remove later
void migrate_old_cache_to_hf_cache(const std::string & bearer_token, bool offline = false);

} // namespace hf_cache
