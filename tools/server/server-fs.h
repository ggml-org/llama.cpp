#pragma once

#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

// Server-side filesystem search backing the /filesystem/* endpoints.

namespace server_fs {

enum class match_mode {
    substring,
    prefix,
};

enum class entry_type_filter {
    any,
    file,
    directory,
};

struct search_options {
    std::string query;
    // when set, search only inside this directory (must resolve within an allowed root)
    std::string context_path;
    match_mode match = match_mode::substring;
    entry_type_filter type = entry_type_filter::any;
    int limit = 50;
    int max_depth = 8;
    bool show_hidden = false; // include entries under "."-prefixed directories
};

struct search_entry {
    std::string name;            // basename
    std::string path;            // absolute path
    std::string parent;          // absolute path of the parent dir
    std::string type;            // "file" or "directory"
    int64_t size = 0;            // bytes, files only
    int64_t modified = 0;        // unix seconds
};

struct git_info {
    bool is_repo = false;
    std::string root;            // directory holding `.git`
    std::string branch;          // branch name; "detached"/"submodule" when no branch can be determined
};

// user's home directory from $HOME or %USERPROFILE%; empty if neither is set
std::string home_dir();

// expand a leading "~" or "~/" to the user's home directory; leaves "~user" forms unchanged
std::string expand_home(const std::string & path);

// directory names skipped during recursive walks
const std::unordered_set<std::string> & junk_dir_names();

// Compute the effective browse roots: `configured` canonicalized, or $HOME when empty.
std::vector<std::string> effective_roots(
        const std::vector<std::string> & configured,
        std::string & err);

// Resolve `path` to a canonical absolute path inside one of `allowed_roots`.
// Empty path resolves to the first root, relative paths resolve against it.
// Returns empty string on failure (path escapes all roots, or does not exist).
std::string resolve_path(
        const std::string & path,
        const std::vector<std::string> & allowed_roots,
        std::string & err);

// Walk `root` (canonical path of an existing directory) and populate `results`
// with entries matching `opts`, ranked by match quality.
// `root` must be inside `allowed_roots` (see resolve_path). A leading "~" in
// the query expands to the user's home directory; an absolute query that
// escapes `root` but lives under another allowed root re-roots the search.
bool search(
        const std::string & root,
        const std::vector<std::string> & allowed_roots,
        const search_options & opts,
        std::vector<search_entry> & results,
        std::string & err);

// Walk up from `path` looking for `.git` (directory or gitfile form), staying
// inside `allowed_roots`. Returns false when no repo is found within the depth cap.
bool git_status(
        const std::string & path,
        const std::vector<std::string> & allowed_roots,
        git_info & info,
        std::string & err);

} // namespace server_fs
