#pragma once

#include <string>
#include <vector>

struct common_params_model;

using common_header      = std::pair<std::string, std::string>;
using common_header_list = std::vector<common_header>;

struct common_download_progress {
    std::string url;
    size_t downloaded = 0;
    size_t total      = 0;
    bool cached       = false;
};

class common_download_callback {
public:
    virtual ~common_download_callback() = default;
    virtual void on_start(const common_download_progress & p) = 0;
    virtual void on_update(const common_download_progress & p) = 0;
    virtual void on_done(const common_download_progress & p, bool ok) = 0;
    virtual bool is_cancelled() const { return false; }
};

struct common_remote_params {
    common_header_list headers;
    long timeout  = 0;           // in seconds, 0 means no timeout
    long max_size = 0;           // unlimited if 0
};

// get remote file content, returns <http_code, raw_response_body>
std::pair<long, std::vector<char>> common_remote_get_content(const std::string & url, const common_remote_params & params);

// split HF repo with tag into <repo, tag>, for example:
// - "ggml-org/models:F16" -> <"ggml-org/models", "F16">
// tag is optional and can be empty
std::pair<std::string, std::string> common_download_split_repo_tag(const std::string & hf_repo_with_tag);

// Result of common_list_cached_models
struct common_cached_model_info {
    std::string repo;
    std::string tag;
    std::string to_string() const {
        return repo + ":" + tag;
    }
};

// Options for common_download_model and common_download_file_single
struct common_download_opts {
    std::string bearer_token;
    common_header_list headers;
    bool offline = false;
    common_download_callback * callback = nullptr;
};

// Result of common_download_model
struct common_download_model_result {
    std::string model_path;
    std::string mmproj_path;
};

// Download model from HuggingFace repo or URL
//
// input (via model struct):
// - model.hf_repo: HF repo with optional tag, see common_download_split_repo_tag
// - model.hf_file: specific file in the repo (requires hf_repo)
// - model.url: simple download (used if hf_repo is empty)
// - model.path: local file path
//
// tag matching (for HF repos without model.hf_file):
// - if tag is specified, searches for GGUF matching that quantization
// - if no tag, searches for Q4_K_M, then Q4_0, then first available GGUF
//
// split GGUF: multi-part files like "model-00001-of-00003.gguf" are automatically
// detected and all parts are downloaded
//
// caching:
// - HF repos: uses HuggingFace cache
// - URLs: uses ETag-based caching
//
// when opts.offline=true, no network requests are made
// when download_mmproj=true, searches for mmproj in same directory as model or any parent directory
// then with the closest quantization bits
//
// returns result with model_path and mmproj_path (empty on failure)
common_download_model_result common_download_model(
    const common_params_model & model,
    const common_download_opts & opts = {},
    bool download_mmproj = false
);

// returns list of cached models
std::vector<common_cached_model_info> common_list_cached_models();

// download single file from url to local path
// returns status code or -1 on error
// skip_etag: if true, don't read/write .etag files (for HF cache where filename is the hash)
int common_download_file_single(const std::string & url,
                                const std::string & path,
                                const common_download_opts & opts = {},
                                bool skip_etag = false);

// resolve and download model from Docker registry
// return local path to downloaded model file
std::string common_docker_resolve_model(const std::string & docker);
