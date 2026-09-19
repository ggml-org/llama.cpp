#pragma once

#include <string>
#include <vector>

// Ref: https://www.modelscope.cn/docs/models/download

namespace ms_cache {

struct ms_file {
    std::string path;
    std::string url;
    std::string local_path;
    std::string final_path;
    std::string oid;
    std::string repo_id;
};

using ms_files = std::vector<ms_file>;

// Get files from ModelScope API
ms_files get_repo_files(
    const std::string & repo_id,
    const std::string & token
);

ms_files get_cached_files(const std::string & repo_id = {});

// Create snapshot path (link or move/copy) and return it
std::string finalize_file(const ms_file & file);

// Remove the entire cached directory for a repo, returns true if removed
bool remove_cached_repo(const std::string & repo_id);

} // namespace ms_cache
