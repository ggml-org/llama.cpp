#pragma once

#include <functional>
#include <string>
#include <vector>

// Ref: https://huggingface.co/docs/hub/local-cache.md

namespace hf_cache {

struct hf_file {
    std::string path;
    std::string url;
    std::string local_path;
    std::string final_path;
    std::string oid;
    std::string repo_id;
};

using hf_files = std::vector<hf_file>;

// Get files from HF API
hf_files get_repo_files(
    const std::string & repo_id,
    const std::string & token
);

hf_files get_cached_files(const std::string & repo_id = {});

// Lock file path for a blob, shared with the hf CLI: <cache>/.locks/<repo folder>/<oid>.lock
// Pure path computation: creates nothing. Returns an empty string on invalid repo or oid.
std::string get_lock_path(const std::string & repo_id, const std::string & oid);

// Create snapshot path (link or move/copy) and return it.
// Throws when the lock cannot be taken or keep_waiting returns false.
// A completed snapshot (final_path present) is returned without the lock, so a read-only cache works.
std::string finalize_file(const hf_file & file, const std::function<bool()> & keep_waiting = {});

// Remove the entire cached directory for a repo, returns true if removed
bool remove_cached_repo(const std::string & repo_id);

// Returns the HuggingFace hub cache path
std::string get_cache_path();

} // namespace hf_cache
