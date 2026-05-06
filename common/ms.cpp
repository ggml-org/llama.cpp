/**
 * ModelScope Integration Module
 *
 * Handles model downloading and caching from ModelScope for llama.cpp.
 * Key features:
 * - Repository file listing & automatic model file selection based on tags
 * - Download progress tracking & automatic local caching
 *
 * Configuration:
 * - Endpoint: `MODEL_ENDPOINT` env var (default: https://modelscope.cn/)
 * - Authentication: Provide token via `-hft` CLI flag or `MS_TOKEN` env var.
 *
 * Usage: llama-cli -ms <repo_id> -hff <model_file> -hft <ms_token>  (e.g., "Qwen/Qwen3-0.6B-GGUF")
 */

#include "ms.h"

#include "common.h"
#include "log.h"
#include "download.h"

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

#include <filesystem>
#include <string>
#include <stdexcept>
#include <cctype>
#include <algorithm>
#include <map>
#include <regex>

#ifndef _WIN32
#include <unistd.h>
#include <pwd.h>
#endif

namespace nl = nlohmann;

namespace ms {

namespace fs = std::filesystem;

struct ms_file {
    std::string path;
    std::string url;
    std::string local_path;
    std::string repo_id;
    uint64_t size = 0;
};

static fs::path get_cache_directory() {
    static const fs::path cache = []() {
        if (auto * p = std::getenv("LLAMA_CACHE"); p && *p) return fs::path(p);
        if (auto * p = std::getenv("MODELSCOPE_CACHE"); p && *p) return fs::path(p);

        if (auto * p = std::getenv("XDG_CACHE_HOME"); p && *p) {
            return fs::path(p) / "modelscope";
        }

#ifndef _WIN32
        const struct passwd * pw = getpwuid(getuid());
        if (pw && pw->pw_dir && *pw->pw_dir) {
            return fs::path(pw->pw_dir) / ".cache" / "modelscope";
        }
#endif

#if defined(_WIN32)
        if (auto * p = std::getenv("USERPROFILE"); p && *p) {
            return fs::path(p) / ".cache" / "modelscope";
        }
#endif

        return fs::current_path() / ".cache" / "modelscope";
    }();
    return cache;
}

// Duplicated from hf-cache.cpp. Cannot be reused because
// those functions are static in hf-cache.cpp
static bool is_alphanum(const char c) {
    return (c >= 'A' && c <= 'Z') ||
           (c >= 'a' && c <= 'z') ||
           (c >= '0' && c <= '9');
}

static bool is_special_char(char c) {
    return c == '/' || c == '.' || c == '-';
}

static bool is_valid_repo_id(const std::string & repo_id) {
    if (repo_id.empty() || repo_id.length() > 256) {
        return false;
    }
    int slash = 0;
    bool special = true;

    for (const char c : repo_id) {
        if (is_alphanum(c) || c == '_') {
            special = false;
        } else if (is_special_char(c)) {
            if (special) {
                return false;
            }
            slash += (c == '/');
            special = true;
        } else {
            return false;
        }
    }
    return !special && slash == 1;
}

static bool is_valid_subpath(const fs::path & base, const fs::path & subpath) {
    if (subpath.is_absolute()) {
        return false;
    }
    std::error_code ec;
    auto abs_base = fs::absolute(base, ec);
    if (ec) return false;

    auto b = abs_base.lexically_normal();
    auto t = (b / subpath).lexically_normal();

    auto [b_end, _] = std::mismatch(b.begin(), b.end(), t.begin(), t.end());
    return b_end == b.end();
}

static nl::json api_get(const std::string & url, const std::string & token = "") {
    common_remote_params params;
    if (!token.empty()) {
        params.headers.emplace_back("Cookie", "m_session_id=" + token);
    }
    params.timeout = 30;

    auto [status, body] = common_remote_get_content(url, params);

    std::string body_str(body.begin(), body.end());

    if (status != 200) {
        if (!body_str.empty()) {
            try {
                auto json_error = nl::json::parse(body_str);
                if (json_error.contains("Message")) {
                    body_str = json_error["Message"].get<std::string>();
                } else if (json_error.contains("msg")) {
                    body_str = json_error["msg"].get<std::string>();
                }
            } catch (...) {
            }
        }
        throw std::runtime_error("HTTP " + std::to_string(status) + ": " + body_str);
    }

    return nl::json::parse(body_str);
}

static const std::string & get_modelscope_endpoint() {
    static const std::string endpoint = []() {
        const char * env = std::getenv("MODEL_ENDPOINT");
        std::string ep = env ? env : "https://modelscope.cn/";
        if (ep.back() != '/') {
            ep += '/';
        }
        return ep;
    }();
    return endpoint;
}

// Similar to hf_cache::get_repo_files() but cannot be shared:
// - Different API endpoint (ModelScope vs HuggingFace)
// - Different auth (Cookie vs Bearer)
static std::vector<ms_file> list_files(const std::string & repo_id, const std::string & token = "") {
    std::vector<ms_file> files;

    const std::string & endpoint = get_modelscope_endpoint();
    std::string api_url = endpoint + "api/v1/models/" + repo_id + "/repo/files?Revision=master&Recursive=true";

    try {
        auto response = api_get(api_url, token);

        if (response.contains("Data") && response["Data"].contains("Files")) {
            for (const auto & file_json : response["Data"]["Files"]) {
                ms_file file;
                file.repo_id = repo_id;
                file.path = file_json.value("Path", "");
                file.size = file_json.value("Size", 0ULL);

                if (!file.path.empty()) {
                    file.url = endpoint + "models/" + repo_id + "/resolve/master/" + file.path;
                    files.push_back(std::move(file));
                }
            }
        }
    } catch (const std::exception & e) {
        std::string err_msg = e.what();
        if (err_msg.find("404") != std::string::npos) {
            LOG_ERR("%s: repository not found: %s\n", __func__, repo_id.c_str());
        } else if (err_msg.find("401") != std::string::npos ||
                   err_msg.find("403") != std::string::npos) {
            if (token.empty()) {
                LOG_DBG("%s: remote list failed (no token), relying on cache.\n", __func__);
            } else {
                LOG_ERR("%s: auth failed: %s\n", __func__, err_msg.c_str());
            }
        } else {
            LOG_ERR("%s: failed to list files for %s: %s\n", __func__, repo_id.c_str(), err_msg.c_str());
        }
    }

    return files;
}

static bool matches_quant_tag(const std::string & filename, const std::string & quant_tag) {
    if (quant_tag.empty()) {
        return true;
    }
    std::regex pattern(quant_tag + "[._-]", std::regex::icase);
    return std::regex_search(filename, pattern);
}

// Similar to download.cpp find_best_model() but cannot be shared:
// - Different data types (ms_file vs hf_file)
static std::string find_best_model(const std::vector<ms_file> & files, const std::string & quant_tag) {
    if (files.empty()) {
        return "";
    }

    std::vector<std::string> tags;
    if (!quant_tag.empty()) {
        tags.push_back(quant_tag);
    } else {
        tags = {"Q4_K_M", "Q8_0"};
    }

    for (const auto & tag : tags) {
        for (const auto & file : files) {
            if (gguf_filename_is_model(file.path) && matches_quant_tag(file.path, tag)) {
                auto split = get_gguf_split_info(file.path);
                if (split.count > 1 && split.index != 1) {
                    continue;
                }
                return file.path;
            }
        }
    }

    if (quant_tag.empty()) {
        for (const auto & file : files) {
            if (gguf_filename_is_model(file.path)) {
                auto split = get_gguf_split_info(file.path);
                if (split.count > 1 && split.index != 1) {
                    continue;
                }
                return file.path;
            }
        }
    }

    return "";
}

static std::string get_local_path(const ms_file & file) {
    fs::path cache_dir = get_cache_directory();
    fs::path base_path = cache_dir / "hub" / "models" / file.repo_id;

    if (!is_valid_subpath(base_path, file.path)) {
        LOG_ERR("%s: security check failed for path: %s\n", __func__, file.path.c_str());
        return "";
    }

    fs::path local_path = base_path / file.path;
    return local_path.string();
}

static void collect_files_from_dir(const fs::path & base, const fs::path & dir, const std::string & clean_repo_id, std::vector<ms_file> & out) {
    std::error_code ec;
    for (const auto & entry : fs::directory_iterator(dir, ec)) {
        if (ec) continue;

        ec.clear();
        if (entry.is_directory(ec) && !ec) {
            collect_files_from_dir(base, entry.path(), clean_repo_id, out);
            continue;
        }

        ec.clear();
        if (!entry.is_regular_file(ec) || ec) continue;

        const std::string fname = entry.path().filename().string();
        if (fname.size() >= 19 && fname.compare(fname.size() - 19, 19, ".downloadInProgress") == 0) {
            continue;
        }

        ms_file file;
        file.repo_id = clean_repo_id;
        file.path = entry.path().lexically_relative(base).generic_string();
        file.local_path = entry.path().string();
        ec.clear();
        file.size = fs::file_size(entry.path(), ec);
        out.push_back(std::move(file));
    }
}

// Similar to hf_cache::get_cached_files() but cannot be shared:
// - Different cache structure (flat dir vs blob/snapshot/symlink)
// - Different directory naming (owner/repo vs models--owner--repo)
static std::vector<ms_file> scan_local_cache(const std::string & clean_repo_id) {
    std::vector<ms_file> cached_files;
    fs::path base_cache_dir = get_cache_directory() / "hub" / "models" / clean_repo_id;

    std::error_code ec;
    if (!fs::exists(base_cache_dir, ec) || !fs::is_directory(base_cache_dir, ec)) {
        return cached_files;
    }

    collect_files_from_dir(base_cache_dir, base_cache_dir, clean_repo_id, cached_files);
    return cached_files;
}

static bool is_file_valid(const ms_file & file, const fs::path & local_path) {
    std::error_code ec;
    if (!fs::exists(local_path, ec)) {
        return false;
    }
    auto size = fs::file_size(local_path, ec);
    if (ec || size == 0) {
        return false;
    }
    if (file.size > 0 && size != file.size) {
        return false;
    }
    return true;
}

static std::string download_file_with_common(const ms_file & selected_file, const fs::path & local_path, bool offline, const std::string & token = "") {
    common_download_opts opts;
    opts.offline = offline;
    if (!token.empty()) {
        opts.headers.emplace_back("Cookie", "m_session_id=" + token);
    }

    int status = common_download_file_single(selected_file.url, local_path.string(), opts, false);

    if (status >= 200 && status < 400) {
        return local_path.string();
    }
    LOG_ERR("%s: download failed with status: %d\n", __func__, status);
    return "";
}

static std::string resolve_file(const ms_file & file, bool offline, const std::string & token) {
    std::string local_path = file.local_path;
    if (local_path.empty()) {
        local_path = get_local_path(file);
    }
    if (local_path.empty()) {
        LOG_ERR("%s: failed to determine local path for %s\n", __func__, file.path.c_str());
        return "";
    }

    if (is_file_valid(file, local_path)) {
        return local_path;
    }

    if (offline || file.url.empty()) {
        return "";
    }

    return download_file_with_common(file, local_path, offline, token);
}

// Similar to common_download_model() but cannot be shared:
// - common_download_model hardcodes HF API via get_hf_plan → hf_cache::get_repo_files
// - Different cache structure (flat vs blob/snapshot/symlink with finalize_file)
// - Different auth mechanism (Cookie vs Bearer)
// - Shared infrastructure: common_download_file_single, gguf_filename_is_model, get_gguf_split_info
download_result download_model(const std::string & clean_repo_id, const std::string & filename, bool offline, const std::string & quant_tag, const std::string & token) {
    download_result result;

    if (!is_valid_repo_id(clean_repo_id)) {
        LOG_ERR("%s: invalid repository: %s\n", __func__, clean_repo_id.c_str());
        return result;
    }

    std::vector<ms_file> all_files = scan_local_cache(clean_repo_id);

    if (!offline) {
        auto remote_files = list_files(clean_repo_id, token);
        std::map<std::string, ms_file> file_map;
        for (const auto & f : remote_files) {
            file_map[f.path] = f;
        }
        for (const auto & f : all_files) {
            auto it = file_map.find(f.path);
            if (it == file_map.end()) {
                file_map[f.path] = f;
            } else if (!f.local_path.empty()) {
                it->second.local_path = f.local_path;
            }
        }
        all_files.clear();
        for (auto & [_, f] : file_map) {
            all_files.push_back(std::move(f));
        }
    }

    if (all_files.empty()) {
        LOG_ERR("%s: no files found for repository %s\n", __func__, clean_repo_id.c_str());
        return result;
    }

    std::string model_path;

    if (!filename.empty()) {
        bool found = false;
        for (const auto & file : all_files) {
            if (file.path == filename) {
                found = true;
                model_path = resolve_file(file, offline, token);
                break;
            }
        }
        if (model_path.empty()) {
            if (found) {
                LOG_ERR("%s: failed to download '%s' from repository %s\n", __func__, filename.c_str(), clean_repo_id.c_str());
            } else {
                LOG_ERR("%s: file '%s' not found in repository %s\n", __func__, filename.c_str(), clean_repo_id.c_str());
                for (const auto & file : all_files) {
                    if (file.path.find(".gguf") != std::string::npos) {
                        LOG_ERR("  %s\n", file.path.c_str());
                    }
                }
            }
            return result;
        }
    } else {
        std::string selected = find_best_model(all_files, quant_tag);
        if (selected.empty()) {
            LOG_ERR("%s: no suitable GGUF file found in repository %s\n", __func__, clean_repo_id.c_str());
            return result;
        }
        for (const auto & file : all_files) {
            if (file.path == selected) {
                model_path = resolve_file(file, offline, token);
                break;
            }
        }
        if (model_path.empty()) {
            LOG_ERR("%s: failed to download model from repository %s\n", __func__, clean_repo_id.c_str());
            return result;
        }
    }

    result.model_path = model_path;

    auto split = get_gguf_split_info(fs::path(model_path).filename().string());
    if (split.count > 1) {
        for (const auto & file : all_files) {
            auto f_split = get_gguf_split_info(file.path);
            if (f_split.prefix == split.prefix && f_split.count == split.count && f_split.index != 1) {
                resolve_file(file, offline, token);
            }
        }
    }

    for (const auto & file : all_files) {
        std::string lower = file.path;
        std::transform(lower.begin(), lower.end(), lower.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

        if (lower.find(".gguf") != std::string::npos && lower.find("mmproj") != std::string::npos) {
            std::string mmproj_path = resolve_file(file, offline, token);
            if (!mmproj_path.empty()) {
                result.mmproj_path = mmproj_path;
                break;
            }
        }
    }

    return result;
}

} // namespace ms
