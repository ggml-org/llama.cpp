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
    // Resolve both paths to absolute and normalize
    std::error_code ec;
    auto b = fs::absolute(base, ec).lexically_normal();
    auto t = (b / subpath).lexically_normal();
    
    if (ec) return false;

    // Check if t starts with b
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
    
    cli.set_read_timeout(30, 0); // 30 seconds timeout
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
    
    // Get ModelScope endpoint
    const char * model_endpoint_env = std::getenv("MODEL_ENDPOINT");
    std::string modelscope_endpoint = "https://modelscope.cn/";
    if (model_endpoint_env) {
        modelscope_endpoint = model_endpoint_env;
        if (modelscope_endpoint.back() != '/') {
            modelscope_endpoint += '/';
        }
    }
    
    std::string api_url = modelscope_endpoint + "api/v1/models/" + repo_id + "/repo/files";
    
    try {
        auto response = api_get(api_url, token);
        
        // ModelScope API returns files in Data.Files array
        if (response.contains("Data") && response["Data"].contains("Files")) {
            for (const auto & file_json : response["Data"]["Files"]) {
                ms_file file;
                file.repo_id = repo_id;
                file.path = file_json.value("Path", "");
                file.size = file_json.value("Size", 0ULL);
                
                if (!file.path.empty()) {
                    // Construct download URL using resolve endpoint
                    file.url = modelscope_endpoint + "models/" + repo_id + "/resolve/master/" + file.path;
                    files.push_back(std::move(file));
                }
            }
        }
    } catch (const std::exception & e) {
        LOG_ERR("%s: failed to list files for %s: %s\n", __func__, repo_id.c_str(), e.what());
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
    
    // Try different patterns: model-Q8_0.gguf, model.Q8_0.gguf, etc.
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
                return file.path;
            }
        }
    }
    
    // Fallback: return first GGUF file
    for (const auto & file : files) {
        if (string_contains(file.path, ".gguf")) {
            return file.path;
        }
    }
    
    return "";
}

static std::string get_local_path(const ms_file & file) {
    fs::path cache_dir = get_cache_directory();
    fs::path base_path = cache_dir / "hub" / "models" / file.repo_id;
    
    // Security check against path traversal
    if (!is_valid_subpath(base_path, file.path)) {
        LOG_ERR("%s: security check failed for path: %s\n", __func__, file.path.c_str());
        return "";
    }

    fs::path local_path = base_path / file.path;
    return local_path.string();
}

// ProgressBar class copied from download.cpp
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

// Download function with progress bar
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
    
    cli.set_read_timeout(300, 0); // 5 minutes timeout
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

std::string download_model(const std::string & clean_repo_id, const std::string & filename, bool offline, const std::string & quant_tag, const std::string & token) {
    if (!is_valid_repo_id(clean_repo_id)) {
        LOG_ERR("%s: invalid repository: %s\n", __func__, clean_repo_id.c_str());
        return "";
    }

    // Check cache first
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

    // If we have a specific filename, look for it directly
    if (!filename.empty()) {
        // First check cache
        for (const auto & file : cached_files) {
            if (file.path == filename) {
                LOG_DBG("%s: found file in cache: %s\n", __func__, file.local_path.c_str());
                return file.local_path;
            }
        }
        
        // Not in cache, need to download
        if (!offline) {
            auto remote_files = list_files(clean_repo_id, token);
            for (const auto & remote_file : remote_files) {
                if (remote_file.path == filename) {
                    ms_file selected_file = remote_file;
                    std::string local_path = get_local_path(selected_file);
                    if (local_path.empty()) {
                        LOG_ERR("%s: failed to determine local path for %s\n", __func__, selected_file.path.c_str());
                        return "";
                    }
                    
                    // Check if file already exists with correct size
                    if (fs::exists(local_path)) {
                        try {
                            uintmax_t file_size = fs::file_size(local_path);
                            if (selected_file.size > 0 && file_size == selected_file.size) {
                                LOG_DBG("%s: file already exists with correct size: %s\n", __func__, local_path.c_str());
                                selected_file.local_path = local_path;
                                return selected_file.local_path;
                            }
                        } catch (...) {
                            // Ignore errors, will re-download
                        }
                    }

                    return download_file_with_progress(selected_file, local_path, token);
                }
            }
            
            // File not found on remote
            LOG_ERR("%s: file '%s' not found in ModelScope repository %s\n", __func__, filename.c_str(), clean_repo_id.c_str());
            if (!cached_files.empty()) {
                LOG_ERR("%s: available files in cache:\n", __func__);
                for (const auto & file : cached_files) {
                    LOG_ERR("  %s\n", file.path.c_str());
                }
            }
            auto remote_files_list = list_files(clean_repo_id, token);
            if (!remote_files_list.empty()) {
                LOG_ERR("%s: available files in repository:\n", __func__);
                for (const auto & file : remote_files_list) {
                    if (string_contains(file.path, ".gguf")) {
                        LOG_ERR("  %s\n", file.path.c_str());
                    }
                }
            }
            return "";
        }
        
        // Offline mode and not in cache
        LOG_ERR("%s: file '%s' not found in cache and offline mode is enabled\n", __func__, filename.c_str());
        if (!cached_files.empty()) {
            LOG_ERR("%s: available files in cache:\n", __func__);
            for (const auto & file : cached_files) {
                LOG_ERR("  %s\n", file.path.c_str());
            }
        }
        return "";
    }

    // No specific filename, find best match based on quantization preference
    if (!cached_files.empty()) {
        std::string selected_file_path = find_best_model(cached_files, quant_tag);
        if (!selected_file_path.empty()) {
            // Find the corresponding file object
            for (const auto & file : cached_files) {
                if (file.path == selected_file_path) {
                    LOG_DBG("%s: found best match in cache: %s\n", __func__, file.local_path.c_str());
                    return file.local_path;
                }
            }
        }
    }

    // Not in cache or no good match, check remote
    if (!offline) {
        auto remote_files = list_files(clean_repo_id, token);
        if (remote_files.empty()) {
            LOG_ERR("%s: no files found in ModelScope repository %s\n", __func__, clean_repo_id.c_str());
            return "";
        }

        std::string selected_file_path = find_best_model(remote_files, quant_tag);
        if (selected_file_path.empty()) {
            LOG_ERR("%s: no suitable GGUF file found in ModelScope repository %s\n", __func__, clean_repo_id.c_str());
            LOG_ERR("%s: available GGUF files:\n", __func__);
            for (const auto & file : remote_files) {
                if (string_contains(file.path, ".gguf")) {
                    LOG_ERR("  %s\n", file.path.c_str());
                }
            }
            return "";
        }

        // Find the selected file
        for (const auto & file : remote_files) {
            if (file.path == selected_file_path) {
                std::string local_path = get_local_path(file);
                if (local_path.empty()) {
                    LOG_ERR("%s: failed to determine local path for %s\n", __func__, file.path.c_str());
                    return "";
                }
                
                // Check if file already exists with correct size
                if (fs::exists(local_path)) {
                    try {
                        uintmax_t file_size = fs::file_size(local_path);
                        if (file.size > 0 && file_size == file.size) {
                            LOG_DBG("%s: file already exists with correct size: %s\n", __func__, local_path.c_str());
                            return local_path;
                        }
                    } catch (...) {
                        // Ignore errors, will re-download
                    }
                }
                
                return download_file_with_progress(file, local_path, token);
            }
        }
    }

    LOG_ERR("%s: failed to find or download model from ModelScope repository %s\n", __func__, clean_repo_id.c_str());
    return "";
}

} // namespace ms