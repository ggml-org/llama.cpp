#pragma once

#include <functional>
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
std::string finalize_file(const ms_file & file, const std::function<bool()> & keep_waiting = {});

// Lock file path for a blob: <cache>/.locks/<repo folder>/<oid>.lock
// Pure path computation: creates nothing. Returns an empty string on invalid repo or oid.
std::string get_lock_path(const std::string & repo_id, const std::string & oid);

// Remove the entire cached directory for a repo, returns true if removed
bool remove_cached_repo(const std::string & repo_id);

// Get the cache directory path as a string
std::string get_cache_path();

} // namespace ms_cache
