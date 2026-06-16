#include "ms-cache.h"

#include "build-info.h"
#include "common.h"
#include "log.h"
#include "http.h"

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <string>
#include <stdexcept>

namespace nl = nlohmann;

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define HOME_DIR "USERPROFILE"
#include <windows.h>
#else
#define HOME_DIR "HOME"
#include <unistd.h>
#include <pwd.h>
#endif

namespace ms_cache {

namespace fs = std::filesystem;

static fs::path get_cache_directory() {
    static const fs::path cache = []() {
        struct {
            const char * var;
            fs::path path;
        } entries[] = {
            {"LLAMA_CACHE",       fs::path()},
            {"MODELSCOPE_CACHE",  fs::path()},
            {"XDG_CACHE_HOME",    fs::path("modelscope") / "hub"},
            {HOME_DIR,            fs::path(".cache") / "modelscope" / "hub"}
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
            return fs::path(pw->pw_dir) / ".cache" / "modelscope" / "hub";
        }
#endif
        throw std::runtime_error("Failed to determine ModelScope cache directory");
    }();

    return cache;
}

static std::string folder_name_to_repo(const std::string & folder) {
    constexpr std::string_view prefix = "models--";
    if (folder.rfind(prefix, 0)) {
        return {};
    }
    std::string result = folder.substr(prefix.length());
    string_replace_all(result, "--", "/");
    return result;
}

static std::string repo_to_folder_name(const std::string & repo_id) {
    constexpr std::string_view prefix = "models--";
    std::string result = std::string(prefix) + repo_id;
    string_replace_all(result, "/", "--");
    return result;
}

static fs::path get_repo_path(const std::string & repo_id) {
    return get_cache_directory() / repo_to_folder_name(repo_id);
}

static bool is_hex_char(const char c) {
    return (c >= 'A' && c <= 'F') ||
           (c >= 'a' && c <= 'f') ||
           (c >= '0' && c <= '9');
}

static bool is_hex_string(const std::string & s, size_t expected_len) {
    if (s.length() != expected_len) {
        return false;
    }
    for (const char c : s) {
        if (!is_hex_char(c)) {
            return false;
        }
    }
    return true;
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

static bool is_valid_oid(const std::string & oid) {
    return is_hex_string(oid, 64);
}

static bool is_valid_subpath(const fs::path & path, const fs::path & subpath) {
    if (subpath.is_absolute()) {
        return false;
    }
    auto b = fs::absolute(path).lexically_normal();
    auto t = (b / subpath).lexically_normal();
    auto [b_end, _] = std::mismatch(b.begin(), b.end(), t.begin(), t.end());

    return b_end == b.end();
}

static void safe_write_file(const fs::path & path, const std::string & data) {
    fs::path path_tmp = path.string() + ".tmp";

    if (path.has_parent_path()) {
        fs::create_directories(path.parent_path());
    }

    std::ofstream file(path_tmp);
    file << data;
    file.close();

    std::error_code ec;

    if (!file.fail()) {
        fs::rename(path_tmp, path, ec);
    }
    if (file.fail() || ec) {
        fs::remove(path_tmp, ec);
        throw std::runtime_error("failed to write file: " + path.string());
    }
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

static nl::json api_get(const std::string & url,
                        const std::string & token) {
    auto [cli, parts] = common_http_client(url);

    httplib::Headers headers = {
        {"User-Agent", "llama-cpp/" + std::string(llama_build_info())},
        {"Accept", "application/json"}
    };

    if (!token.empty()) {
        headers.emplace("Cookie", "m_session_id=" + token);
    }

    if (auto res = cli.Get(parts.path, headers)) {
        auto body = res->body;

        if (res->status == 200) {
            return nl::json::parse(res->body);
        }
        try {
            auto json_error = nl::json::parse(res->body);
            if (json_error.contains("Message")) {
                body = json_error["Message"].get<std::string>();
            } else if (json_error.contains("msg")) {
                body = json_error["msg"].get<std::string>();
            }
        } catch (...) { }

        throw std::runtime_error("GET failed (" + std::to_string(res->status) + "): " + body);
    } else {
        throw std::runtime_error("HTTPLIB failed: " + httplib::to_string(res.error()));
    }
}

// ModelScope uses "master" as the snapshot key (no single repo-wide commit like HF)
static const std::string & get_snapshot_ref() {
    static const std::string ref = "master";
    return ref;
}

ms_files get_repo_files(const std::string & repo_id,
                        const std::string & token) {
    if (!is_valid_repo_id(repo_id)) {
        LOG_WRN("%s: invalid repository: %s\n", __func__, repo_id.c_str());
        return {};
    }

    const std::string & endpoint = get_modelscope_endpoint();
    std::string api_url = endpoint + "api/v1/models/" + repo_id + "/repo/files?Revision=master&Recursive=true";

    fs::path blobs_path = get_repo_path(repo_id) / "blobs";
    fs::path ref = get_snapshot_ref();
    fs::path commit_path = get_repo_path(repo_id) / "snapshots" / ref;

    ms_files files;

    try {
        auto response = api_get(api_url, token);

        if (!response.contains("Data") || !response["Data"].contains("Files")) {
            LOG_WRN("%s: unexpected response format for '%s'\n", __func__, repo_id.c_str());
            return {};
        }

        // write ref so get_cached_files can find the snapshot
        fs::path refs_path = get_repo_path(repo_id) / "refs";
        safe_write_file(refs_path / "main", ref);

        for (const auto & item : response["Data"]["Files"]) {
            if (!item.contains("Path") || !item["Path"].is_string()) {
                continue;
            }

            ms_file file;
            file.repo_id = repo_id;
            file.path = item["Path"].get<std::string>();

            if (!is_valid_subpath(commit_path, file.path)) {
                LOG_WRN("%s: skip invalid path: %s\n", __func__, file.path.c_str());
                continue;
            }

            if (item.contains("Sha256") && item["Sha256"].is_string()) {
                file.oid = item["Sha256"].get<std::string>();
            }

            if (!file.oid.empty() && !is_valid_oid(file.oid)) {
                LOG_WRN("%s: skip invalid oid: %s\n", __func__, file.oid.c_str());
                continue;
            }

            file.url = endpoint + "models/" + repo_id + "/resolve/master/" + file.path;

            fs::path final_path = commit_path / file.path;
            file.final_path = final_path.string();

            if (!file.oid.empty() && !fs::exists(final_path)) {
                fs::path local_path = blobs_path / file.oid;
                file.local_path = local_path.string();
            } else {
                file.local_path = file.final_path;
            }

            files.push_back(file);
        }
    } catch (const nl::json::exception & e) {
        LOG_ERR("%s: JSON error: %s\n", __func__, e.what());
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

static std::string get_cached_ref(const fs::path & repo_path) {
    fs::path refs_path = repo_path / "refs";
    if (!fs::is_directory(refs_path)) {
        return {};
    }

    for (const auto & entry : fs::directory_iterator(refs_path)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        std::ifstream f(entry.path());
        std::string ref;
        if (!f || !std::getline(f, ref) || ref.empty()) {
            continue;
        }
        if (entry.path().filename() == "main") {
            return ref;
        }
    }
    return {};
}

ms_files get_cached_files(const std::string & repo_id) {
    fs::path cache_dir = get_cache_directory();
    if (!fs::exists(cache_dir)) {
        return {};
    }

    if (!repo_id.empty() && !is_valid_repo_id(repo_id)) {
        LOG_WRN("%s: invalid repository: %s\n", __func__, repo_id.c_str());
        return {};
    }

    ms_files files;

    for (const auto & repo : fs::directory_iterator(cache_dir)) {
        if (!repo.is_directory()) {
            continue;
        }
        fs::path snapshots_path = repo.path() / "snapshots";

        if (!fs::exists(snapshots_path)) {
            continue;
        }
        std::string _repo_id = folder_name_to_repo(repo.path().filename().string());

        if (!is_valid_repo_id(_repo_id)) {
            continue;
        }
        if (!repo_id.empty() && _repo_id != repo_id) {
            continue;
        }
        std::string ref = get_cached_ref(repo.path());
        fs::path ref_path = snapshots_path / ref;

        if (ref.empty() || !fs::is_directory(ref_path)) {
            continue;
        }
        for (const auto & entry : fs::recursive_directory_iterator(ref_path)) {
            if (!entry.is_regular_file() && !entry.is_symlink()) {
                continue;
            }
            fs::path path = entry.path().lexically_relative(ref_path);

            if (!path.empty()) {
                ms_file file;
                file.repo_id = _repo_id;
                file.path = path.generic_string();
                file.local_path = entry.path().string();
                file.final_path = file.local_path;
                files.push_back(std::move(file));
            }
        }
    }

    return files;
}

std::string finalize_file(const ms_file & file) {
    static std::atomic<bool> symlinks_disabled{false};

    std::error_code ec;
    fs::path local_path(file.local_path);
    fs::path final_path(file.final_path);

    if (local_path == final_path || fs::exists(final_path, ec)) {
        return file.final_path;
    }

    if (!fs::exists(local_path, ec)) {
        return file.final_path;
    }

    fs::create_directories(final_path.parent_path(), ec);

    if (!symlinks_disabled) {
        fs::path target = fs::relative(local_path, final_path.parent_path(), ec);
        if (!ec) {
            fs::create_symlink(target, final_path, ec);
        }
        if (!ec) {
            return file.final_path;
        }
    }

    if (!symlinks_disabled.exchange(true)) {
        LOG_WRN("%s: failed to create symlink: %s\n", __func__, ec.message().c_str());
        LOG_WRN("%s: switching to degraded mode\n", __func__);
    }

    fs::rename(local_path, final_path, ec);
    if (ec) {
        LOG_WRN("%s: failed to move file to snapshots: %s\n", __func__, ec.message().c_str());
        fs::copy(local_path, final_path, ec);
        if (ec) {
            LOG_ERR("%s: failed to copy file to snapshots: %s\n", __func__, ec.message().c_str());
        }
    }
    return file.final_path;
}

} // namespace ms_cache
