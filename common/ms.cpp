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
 * Usage: llama-cli -ms <repo_id> -hff <model_file> -hft <ms_token> (e.g., "Qwen/Qwen3-0.6B-GGUF")
 */

#include "ms.h"

#include "common.h"
#include "log.h"
#include "http.h"
#include "download.h"
#include "build-info.h"

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <stdexcept>
#include <cctype>
#include <algorithm>
#include <regex>
#include <iomanip>
#include <mutex>
#include <map>

// isatty and system includes
#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#include <pwd.h>
#endif

namespace nl = nlohmann;

#if defined(_WIN32)
#define HOME_DIR "USERPROFILE"
#else
#define HOME_DIR "HOME"
#endif

namespace ms {

namespace fs = std::filesystem;

static fs::path get_cache_directory() {
    static const fs::path cache = []() {
        struct {
            const char * var;
            fs::path path;
        } entries[] = {
            {"LLAMA_CACHE",           fs::path()},
            {"MODELSCOPE_CACHE",      fs::path()},
            {"XDG_CACHE_HOME",        fs::path("modelscope")},
            {HOME_DIR,                fs::path(".cache") / "modelscope"}
        };
        for (const auto & entry : entries) {
            if (auto * p = std::getenv(entry.var); p && *p) {
                fs::path base(p);
                return entry.path.empty() ? base : base / entry.path;
            }
        }
#ifndef _WIN32
        const struct passwd * pw = getpwuid(getuid());
        if (pw && pw->pw_dir && *pw->pw_dir) {
            return fs::path(pw->pw_dir) / ".cache" / "modelscope";
        }
#endif
        return fs::current_path() / ".cache" / "modelscope";
    }();
    return cache;
}

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
    auto b = fs::absolute(base, ec).lexically_normal();
    auto t = (b / subpath).lexically_normal();
    
    if (ec) return false;

    auto [b_end, _] = std::mismatch(b.begin(), b.end(), t.begin(), t.end());
    return b_end == b.end();
}

static nl::json api_get(const std::string & url, const std::string & token = "") {
    auto [cli, parts] = common_http_client(url);
    
    httplib::Headers headers;
    headers.emplace("User-Agent", "llama-cpp/" + std::string(llama_build_info()));
    if (!token.empty()) {
        headers.emplace("Cookie", "m_session_id=" + token);
    }
    
    cli.set_read_timeout(30, 0);
    cli.set_write_timeout(30, 0);
    
    auto res = cli.Get(parts.path, headers);
    
    if (!res || res->status != 200) {
        std::string body = res ? res->body : "";
        if (!body.empty()) {
            try {
                auto json_error = nl::json::parse(body);
                if (json_error.contains("Message")) {
                    body = json_error["Message"].get<std::string>();
                } else if (json_error.contains("msg")) {
                    body = json_error["msg"].get<std::string>();
                }
            } catch (...) {
                // Keep original body
            }
        }
        long http_code = res ? res->status : -1;
        throw std::runtime_error("HTTP " + std::to_string(http_code) + ": " + body);
    }
    
    return nl::json::parse(res->body);
}

static std::vector<ms_file> list_files(const std::string & repo_id, const std::string & token = "") {
    std::vector<ms_file> files;
    
    const char * model_endpoint_env = std::getenv("MODEL_ENDPOINT");
    std::string endpoint = "https://www.modelscope.cn";
    
    if (model_endpoint_env) {
        endpoint = model_endpoint_env;
        if (!endpoint.empty() && endpoint.back() == '/') {
            endpoint.pop_back();
        }
    }
    
    std::string api_url = endpoint + "/api/v1/models/" + repo_id + "/repo/files";
    
    try {
        auto response = api_get(api_url, token);
        
        if (response.contains("Data") && response["Data"].contains("Files")) {
            for (const auto & file_json : response["Data"]["Files"]) {
                ms_file file;
                file.repo_id = repo_id;
                file.path = file_json.value("Path", "");
                file.size = file_json.value("Size", 0ULL);
                
                if (!file.path.empty()) {
                    file.url = endpoint + "/models/" + repo_id + "/resolve/master/" + file.path;
                    files.push_back(std::move(file));
                }
            }
        }
    } catch (const std::exception & e) {
        std::string err_msg = e.what();
        // 静默处理认证失败，依靠缓存降级
        if (err_msg.find("401") != std::string::npos || 
            err_msg.find("403") != std::string::npos || 
            err_msg.find("404") != std::string::npos) {
            
            if (token.empty()) {
                LOG_DBG("%s: remote list failed (no token), relying on cache.\n", __func__);
            } else {
                LOG_ERR("%s: auth failed or repo not found: %s\n", __func__, err_msg.c_str());
            }
        } else {
            LOG_ERR("%s: failed to list files for %s: %s\n", __func__, repo_id.c_str(), err_msg.c_str());
        }
    }
    
    return files;
}

static bool string_contains(const std::string & str, const std::string & substr) {
    return str.find(substr) != std::string::npos;
}

static std::string normalize_quant_tag(const std::string & tag) {
    std::string result = tag;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return result;
}

static bool matches_quant_tag(const std::string & filename, const std::string & quant_tag) {
    if (quant_tag.empty()) {
        return true;
    }
    
    std::string normalized_tag = normalize_quant_tag(quant_tag);
    std::string normalized_filename = normalize_quant_tag(filename);
    
    std::vector<std::string> patterns = {
        "-" + normalized_tag + ".",
        "." + normalized_tag + ".",
        "_" + normalized_tag + "."
    };
    
    for (const auto & pattern : patterns) {
        if (string_contains(normalized_filename, pattern)) {
            return true;
        }
    }
    
    return false;
}

static std::string find_best_model(const std::vector<ms_file> & files, const std::string & quant_tag) {
    if (files.empty()) {
        return "";
    }

    std::vector<std::string> preferred_quants = {
        "q4_k_m", "q4_0", "q5_k_m", "q5_0", "q6_k", "q8_0"
    };
    
    std::vector<std::string> tags;
    if (!quant_tag.empty()) {
        tags.push_back(quant_tag);
    }
    tags.insert(tags.end(), preferred_quants.begin(), preferred_quants.end());
    
    for (const auto & tag : tags) {
        for (const auto & file : files) {
            if (string_contains(file.path, ".gguf") && matches_quant_tag(file.path, tag)) {
                // LOG_INF 改为 LOG_DBG，或者完全移除，保持静默
                LOG_DBG("%s: selected best model: %s\n", __func__, file.path.c_str());
                return file.path;
            }
        }
    }
    
    // Fallback
    for (const auto & file : files) {
        if (string_contains(file.path, ".gguf")) {
            LOG_DBG("%s: selected fallback model: %s\n", __func__, file.path.c_str());
            return file.path;
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

class ProgressBar {
    static inline std::mutex mutex;
    static inline std::map<const ProgressBar *, int> lines;
    static inline int max_line = 0;

    std::string filename;
    size_t len = 0;

    static void cleanup(const ProgressBar * line) {
        lines.erase(line);
        if (lines.empty()) {
            max_line = 0;
        }
    }

    static bool is_output_a_tty() {
#if defined(_WIN32)
        return _isatty(_fileno(stdout));
#else
        return isatty(1);
#endif
    }

public:
    ProgressBar(const std::string & url = "") : filename(url) {
        if (auto pos = filename.rfind('/'); pos != std::string::npos) {
            filename = filename.substr(pos + 1);
        }
        if (auto pos = filename.find('?'); pos != std::string::npos) {
            filename = filename.substr(0, pos);
        }
        for (size_t i = 0; i < filename.size(); ++i) {
            if ((filename[i] & 0xC0) != 0x80) {
                if (len++ == 39) {
                    filename.resize(i);
                    filename += "…";
                    break;
                }
            }
        }
    }

    ~ProgressBar() {
        std::lock_guard<std::mutex> lock(mutex);
        cleanup(this);
    }

    void update(size_t current, size_t total) {
        if (!total || !is_output_a_tty()) {
            return;
        }

        std::lock_guard<std::mutex> lock(mutex);

        if (lines.find(this) == lines.end()) {
            lines[this] = max_line++;
            std::cout << "\n";
        }
        int lines_up = max_line - lines[this];

        size_t bar = 55 - len;
        size_t pct = (100 * current) / total;
        size_t pos = (bar * current) / total;

        if (lines_up > 0) {
            std::cout << "\033[" << lines_up << "A";
        }
        std::cout << '\r' << "Downloading " << filename << " ";

        for (size_t i = 0; i < bar; ++i) {
            std::cout << (i < pos ? "—" : " ");
        }
        std::cout << std::setw(4) << pct << "%\033[K";

        if (lines_up > 0) {
            std::cout << "\033[" << lines_up << "B";
        }
        std::cout << '\r' << std::flush;

        if (current == total) {
            cleanup(this);
        }
    }

    ProgressBar(const ProgressBar &) = delete;
    ProgressBar & operator=(const ProgressBar &) = delete;
};

static std::string download_file_with_progress(const ms_file & selected_file, const fs::path & local_path, const std::string & token = "") {
    fs::create_directories(local_path.parent_path());
    
    std::ofstream ofs(local_path, std::ios::binary);
    if (!ofs.is_open()) {
        LOG_ERR("%s: error opening local file for writing: %s\n", __func__, local_path.string().c_str());
        return "";
    }

    auto [cli, parts] = common_http_client(selected_file.url);
    
    httplib::Headers headers;
    headers.emplace("User-Agent", "llama-cpp/" + std::string(llama_build_info()));
    if (!token.empty()) {
        headers.emplace("Cookie", "m_session_id=" + token);
    }
    
    cli.set_read_timeout(300, 0);
    cli.set_write_timeout(300, 0);

    size_t downloaded = 0;
    size_t total_size = selected_file.size;
    size_t progress_step = 0;
    ProgressBar bar(selected_file.url);

    auto res = cli.Get(parts.path, headers,
        [&](const httplib::Response &response) {
            if (response.status != 200) {
                LOG_WRN("%s: download received non-successful status code: %d\n", __func__, response.status);
                return false;
            }
            if (total_size == 0 && response.has_header("Content-Length")) {
                try {
                    size_t content_length = std::stoull(response.get_header_value("Content-Length"));
                    total_size = content_length;
                } catch (const std::exception &e) {
                    LOG_WRN("%s: invalid Content-Length header: %s\n", __func__, e.what());
                }
            }
            return true;
        },
        [&](const char *data, size_t len) {
            ofs.write(data, len);
            if (!ofs) {
                LOG_ERR("%s: error writing to file: %s\n", __func__, local_path.string().c_str());
                return false;
            }
            downloaded += len;
            progress_step += len;

            if (progress_step >= total_size / 1000 || downloaded == total_size) {
                bar.update(downloaded, total_size);
                progress_step = 0;
            }
            return true;
        },
        nullptr
    );

    if (!res) {
        LOG_ERR("%s: download failed: %s (status: %d)\n",
                __func__,
                httplib::to_string(res.error()).c_str(),
                res ? res->status : -1);
        return "";
    }

    ofs.close();
    if (ofs.fail()) {
        LOG_ERR("%s: failed to write file: %s\n", __func__, local_path.string().c_str());
        fs::remove(local_path);
        return "";
    }

    LOG_INF("%s: successfully downloaded %s\n", __func__, local_path.string().c_str());
    return local_path.string();
}

// Helper to scan local cache directory
static std::vector<ms_file> scan_local_cache(const std::string & clean_repo_id) {
    std::vector<ms_file> cached_files;
    fs::path cache_dir = get_cache_directory() / "hub" / "models" / clean_repo_id;
    
    std::error_code ec;
    if (fs::exists(cache_dir, ec) && fs::is_directory(cache_dir, ec)) {
        for (const auto & entry : fs::directory_iterator(cache_dir, ec)) {
            if (ec || !fs::is_regular_file(entry, ec)) continue;
            
            ms_file file;
            file.repo_id = clean_repo_id;
            file.path = entry.path().filename().string();
            file.local_path = entry.path().string();
            file.size = fs::file_size(entry.path(), ec);
            cached_files.push_back(std::move(file));
        }
    }
    return cached_files;
}

std::string download_model(const std::string & clean_repo_id, const std::string & filename, bool offline, const std::string & quant_tag, const std::string & token) {
    if (!is_valid_repo_id(clean_repo_id)) {
        LOG_ERR("%s: invalid repository: %s\n", __func__, clean_repo_id.c_str());
        return "";
    }

    // 1. Scan Local Cache First
    std::vector<ms_file> cached_files = scan_local_cache(clean_repo_id);

    // 2. If specific filename is requested
    if (!filename.empty()) {
        // Check cache
        for (const auto & file : cached_files) {
            if (file.path == filename) {
                LOG_DBG("%s: found specified file in cache: %s\n", __func__, file.local_path.c_str());
                return file.local_path;
            }
        }
        
        // Not in cache, must download
        if (offline) {
            LOG_ERR("%s: file '%s' not found in cache and offline mode is enabled\n", __func__, filename.c_str());
            return "";
        }

        auto remote_files = list_files(clean_repo_id, token);
        for (const auto & remote_file : remote_files) {
            if (remote_file.path == filename) {
                std::string local_path = get_local_path(remote_file);
                if (local_path.empty()) return "";

                // Check size match
                if (fs::exists(local_path)) {
                    try {
                        if (remote_file.size > 0 && fs::file_size(local_path) == remote_file.size) {
                            return local_path;
                        }
                    } catch (...) {}
                }
                return download_file_with_progress(remote_file, local_path, token);
            }
        }
        LOG_ERR("%s: file '%s' not found in remote repository\n", __func__, filename.c_str());
        return "";
    }

    // 3. Auto-select Mode (No specific filename)
    
    // A. Try to find best match in CACHE first (Offline-first strategy)
    if (!cached_files.empty()) {
        std::string best_cached = find_best_model(cached_files, quant_tag);
        if (!best_cached.empty()) {
            for (const auto & file : cached_files) {
                if (file.path == best_cached) {
                    LOG_DBG("%s: using cached model '%s' (skipping network request)\n", __func__, file.path.c_str());
                    return file.local_path;
                }
            }
        }
    }

    // B. If cache miss or no good match, try Network
    if (!offline) {
        auto remote_files = list_files(clean_repo_id, token);
        
        // If network fails (e.g. private repo no token), fallback to ANY cached gguf if available
        if (remote_files.empty()) {
            if (!cached_files.empty()) {
                for (const auto & file : cached_files) {
                    if (string_contains(file.path, ".gguf")) {
                        LOG_WRN("%s: remote list failed, using available cached file: %s\n", __func__, file.path.c_str());
                        return file.local_path;
                    }
                }
            }
            LOG_ERR("%s: failed to list files and cache is empty/invalid\n", __func__);
            return "";
        }

        std::string best_remote = find_best_model(remote_files, quant_tag);
        if (best_remote.empty()) {
            LOG_ERR("%s: no suitable GGUF file found in remote repository\n", __func__);
            return "";
        }

        // Find the file object and download/check
        for (const auto & file : remote_files) {
            if (file.path == best_remote) {
                std::string local_path = get_local_path(file);
                if (local_path.empty()) return "";

                if (fs::exists(local_path)) {
                    try {
                        if (file.size > 0 && fs::file_size(local_path) == file.size) {
                            return local_path;
                        }
                    } catch (...) {}
                }
                return download_file_with_progress(file, local_path, token);
            }
        }
    }

    LOG_ERR("%s: failed to find model in cache or remote\n", __func__);
    return "";
}

download_result download_model_with_mmproj(const std::string & clean_repo_id, const std::string & filename, bool offline, const std::string & quant_tag, const std::string & token) {
    download_result result;

    if (!is_valid_repo_id(clean_repo_id)) {
        LOG_ERR("%s: invalid repository: %s\n", __func__, clean_repo_id.c_str());
        return result;
    }

    // 1. Scan Local Cache
    std::vector<ms_file> cached_files = scan_local_cache(clean_repo_id);

    // 2. Determine Model Filename
    std::string model_filename = filename;
    
    // A. Try to find in Cache first
    if (model_filename.empty()) {
        model_filename = find_best_model(cached_files, quant_tag);
    }

    // B. If not in cache, try Remote (Network)
    if (model_filename.empty() && !offline) {
        auto remote_files = list_files(clean_repo_id, token);
        if (!remote_files.empty()) {
            model_filename = find_best_model(remote_files, quant_tag);
        }
    }

    // 3. Resolve Main Model Path (Download if necessary)
    if (!model_filename.empty()) {
        // Check if it's already in cache
        bool found_in_cache = false;
        for (const auto & f : cached_files) {
            if (f.path == model_filename) {
                found_in_cache = true;
                result.model_path = f.local_path;
                break;
            }
        }

        // If not in cache, download it
        if (!found_in_cache && !offline) {
            result.model_path = download_model(clean_repo_id, model_filename, offline, quant_tag, token);
        } else if (!found_in_cache && offline) {
             LOG_ERR("%s: model '%s' not in cache and offline mode enabled\n", __func__, model_filename.c_str());
        }
    }

    // 4. Find mmproj File
    std::string mmproj_filename = "";
    auto is_mmproj = [](const std::string & path) {
        std::string lower = path;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
        return (string_contains(lower, "mmproj") || string_contains(lower, "clip") || string_contains(lower, "vision")) &&
               (string_contains(lower, ".gguf") || string_contains(lower, ".bin"));
    };

    // Search in Cache first
    for (const auto & f : cached_files) {
        if (is_mmproj(f.path) && f.path != model_filename) {
            result.mmproj_path = f.local_path;
            mmproj_filename = f.path;
            break;
        }
    }

    // If not in cache, try Remote
    if (result.mmproj_path.empty() && !offline) {
        auto remote_files = list_files(clean_repo_id, token);
        for (const auto & f : remote_files) {
            if (is_mmproj(f.path) && f.path != model_filename) {
                mmproj_filename = f.path;
                std::string local_path = get_local_path(f);
                if (local_path.empty()) continue;

                if (fs::exists(local_path)) {
                    try {
                        if (f.size > 0 && fs::file_size(local_path) == f.size) {
                            result.mmproj_path = local_path;
                            break;
                        } else {
                             fs::remove(local_path);
                        }
                    } catch (...) {}
                }
                
                if (result.mmproj_path.empty()) {
                    result.mmproj_path = download_file_with_progress(f, local_path, token);
                }
                break;
            }
        }
    }

    if (result.model_path.empty()) {
        LOG_ERR("%s: no model file found\n", __func__);
    }
    
    return result;
}

} // namespace ms
