#pragma once

#include "hf-cache.h"

#include <string>
#include <vector>

struct common_params_model;

using common_header      = std::pair<std::string, std::string>;
using common_header_list = std::vector<common_header>;

struct common_remote_params {
    common_header_list headers;
    long timeout  = 0;           // in seconds, 0 means no timeout
    long max_size = 0;           // unlimited if 0
};

// get remote file content, returns <http_code, raw_response_body>
std::pair<long, std::vector<char>> common_remote_get_content(const std::string & url, const common_remote_params & params);

// split HF repo with tag into <repo, tag>
// for example: "user/model:tag" -> <"user/model", "tag">
// if tag is not present, default to "latest"
// example: "user/model" -> <"user/model", "latest">
std::pair<std::string, std::string> common_download_split_repo_tag(const std::string & hf_repo_with_tag);

// Options for common_download_model
struct common_download_model_opts {
    bool download_mmproj = false;
    bool offline         = false;
};

// Result of common_download_model
struct common_download_model_result {
    std::string model_path;  // path to downloaded model (empty on failure)
    std::string mmproj_path; // path to downloaded mmproj (empty if not downloaded)
};

/**
 * Allow getting the HF file from the HF repo with tag (like ollama), for example:
 * - bartowski/Llama-3.2-3B-Instruct-GGUF:q4
 * - bartowski/Llama-3.2-3B-Instruct-GGUF:Q4_K_M
 * - bartowski/Llama-3.2-3B-Instruct-GGUF:q5_k_s
 * Tag is optional, it checks for Q4_K_M first, then Q4_0, then if not found, return the first GGUF file in repo
 */
common_download_model_result common_download_model(
    const common_params_model & model,
    const std::string & bearer_token,
    const common_download_model_opts & opts = {},
    const common_header_list & headers = {}
);

// returns list of cached models
std::vector<std::string> common_list_cached_models();

// download single file from url to local path
// returns status code or -1 on error
// skip_etag: if true, don't read/write .etag files (for HF cache where filename is the hash)
int common_download_file_single(const std::string & url,
                                const std::string & path,
                                const std::string & bearer_token,
                                bool offline,
                                const common_header_list & headers = {},
                                bool skip_etag = false);

// resolve and download model from Docker registry
// return local path to downloaded model file
std::string common_docker_resolve_model(const std::string & docker);
