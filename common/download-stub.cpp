// Stub implementation of download.h / hf-cache.h for LLAMA_DOWNLOAD=OFF builds.
// Keeps arg.cpp/preset.cpp source-compatible without cpp-httplib or OpenSSL.
// Pure string helpers behave like the real implementation; anything that would
// touch the network or the HF cache throws or reports empty state.

#include "download.h"

#include "common.h"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

static const char DOWNLOAD_DISABLED_MSG[] =
    "error: model download support is disabled in this build (LLAMA_DOWNLOAD=OFF); "
    "provide a local model path instead";

std::pair<long, std::vector<char>> common_remote_get_content(const std::string & /*url*/, const common_remote_params & /*params*/) {
    throw std::runtime_error(DOWNLOAD_DISABLED_MSG);
}

std::pair<std::string, std::string> common_download_split_repo_tag(const std::string & hf_repo_with_tag) {
    auto parts = string_split<std::string>(hf_repo_with_tag, ':');
    std::string tag = parts.size() > 1 ? parts.back() : "";
    std::string hf_repo = parts[0];
    if (string_split<std::string>(hf_repo, '/').size() != 2) {
        throw std::invalid_argument("error: invalid HF repo format, expected <user>/<model>[:quant]\n");
    }
    return {hf_repo, tag};
}

void common_download_run_tasks(const std::vector<common_download_task> & tasks) {
    // called unconditionally with an empty task list on plain local-model runs
    if (tasks.empty()) {
        return;
    }
    throw std::runtime_error(DOWNLOAD_DISABLED_MSG);
}

std::vector<std::string> common_download_get_all_parts(const std::string & url) {
    // no remote probing; treat every url as a single file
    return { url };
}

std::vector<common_cached_model_info> common_list_cached_models() {
    return {};
}

int common_download_file_single(const std::string & /*url*/,
                                const std::string & /*path*/,
                                const common_download_opts & /*opts*/,
                                bool /*skip_etag*/) {
    throw std::runtime_error(DOWNLOAD_DISABLED_MSG);
}

std::string common_docker_resolve_model(const std::string & /*docker*/) {
    throw std::runtime_error(DOWNLOAD_DISABLED_MSG);
}

bool common_download_remove(const std::string & /*hf_repo_with_tag*/) {
    return false;
}

common_download_hf_plan common_download_get_hf_plan(const common_params_model & /*model*/, const common_download_opts & /*opts*/) {
    throw std::runtime_error(DOWNLOAD_DISABLED_MSG);
}

namespace hf_cache {

hf_files get_repo_files(const std::string & /*repo_id*/, const std::string & /*token*/) {
    throw std::runtime_error(DOWNLOAD_DISABLED_MSG);
}

hf_files get_cached_files(const std::string & /*repo_id*/) {
    return {};
}

std::string get_cache_path() {
    return {};
}

std::string finalize_file(const hf_file & /*file*/) {
    throw std::runtime_error(DOWNLOAD_DISABLED_MSG);
}

bool remove_cached_repo(const std::string & /*repo_id*/) {
    return false;
}

} // namespace hf_cache
