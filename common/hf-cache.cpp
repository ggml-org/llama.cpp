#include "hf-cache.h"

#include "common.h"
#include "log.h"
#include "http.h"

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <atomic>
#include <regex> // migration only
#include <string>
#include <string_view>
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
#endif

namespace hf_cache {

namespace fs = std::filesystem;

static fs::path get_cache_directory() {
    static const fs::path cache = []() {
        struct {
            const char * var;
            fs::path path;
        } entries[] = {
            {"LLAMA_CACHE",           fs::path()},
            {"HF_HUB_CACHE",          fs::path()},
            {"HUGGINGFACE_HUB_CACHE", fs::path()},
            {"HF_HOME",               fs::path("hub")},
            {"XDG_CACHE_HOME",        fs::path("huggingface") / "hub"},
            {HOME_DIR,                fs::path(".cache") / "huggingface" / "hub"}
        };
        for (const auto & entry : entries) {
            if (auto * p = std::getenv(entry.var); p && *p) {
                fs::path base(p);
                return entry.path.empty() ? base : base / entry.path;
            }
        }
        throw std::runtime_error("Failed to determine HF cache directory");
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

// base chars [A-Za-z0-9_] are always valid
// special chars [/.-] must be surrounded by base chars
// exactly one '/' required
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

static bool is_valid_hf_token(const std::string & token) {
    if (token.length() < 37 || token.length() > 256 ||
        !string_starts_with(token, "hf_")) {
        return false;
    }
    for (size_t i = 3; i < token.length(); ++i) {
        if (!is_alphanum(token[i])) {
            return false;
        }
    }
    return true;
}

static bool is_valid_commit(const std::string & hash) {
    return is_hex_string(hash, 40);
}

static bool is_valid_oid(const std::string & oid) {
    return is_hex_string(oid, 40) || is_hex_string(oid, 64);
}

static bool is_valid_subpath(const fs::path & path, const fs::path & subpath) {
    if (subpath.is_absolute()) {
        return false; // never do a / b with b absolute
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

static nl::json api_get(const std::string & url,
                        const std::string & token) {
    auto [cli, parts] = common_http_client(url);

    httplib::Headers headers = {
        {"User-Agent", "llama-cpp/" + build_info},
        {"Accept", "application/json"}
    };

    if (is_valid_hf_token(token)) {
        headers.emplace("Authorization", "Bearer " + token);
    } else if (!token.empty()) {
        LOG_WRN("%s: invalid token, authentication disabled\n", __func__);
    }

    if (auto res = cli.Get(parts.path, headers)) {
        auto body = res->body;

        if (res->status == 200) {
            return nl::json::parse(res->body);
        }
        try {
            body = nl::json::parse(res->body)["error"].get<std::string>();
        } catch (...) { }

        throw std::runtime_error("GET failed (" + std::to_string(res->status) + "): " + body);
    } else {
        throw std::runtime_error("HTTPLIB failed: " + httplib::to_string(res.error()));
    }
}

static std::string get_repo_commit(const std::string & repo_id,
                                   const std::string & token) {
    try {
        auto endpoint = get_model_endpoint();
        auto json = api_get(endpoint + "api/models/" + repo_id + "/refs", token);

        if (!json.is_object() ||
            !json.contains("branches") || !json["branches"].is_array()) {
            LOG_WRN("%s: missing 'branches' for '%s'\n", __func__, repo_id.c_str());
            return {};
        }

        fs::path refs_path = get_repo_path(repo_id) / "refs";
        std::string name;
        std::string commit;

        for (const auto & branch : json["branches"]) {
            if (!branch.is_object() ||
                !branch.contains("name") || !branch["name"].is_string() ||
                !branch.contains("targetCommit") || !branch["targetCommit"].is_string()) {
                continue;
            }
            std::string _name = branch["name"].get<std::string>();
            std::string _commit = branch["targetCommit"].get<std::string>();

            if (!is_valid_subpath(refs_path, _name)) {
                LOG_WRN("%s: skip invalid branch: %s\n", __func__, _name.c_str());
                continue;
            }
            if (!is_valid_commit(_commit)) {
                LOG_WRN("%s: skip invalid commit: %s\n", __func__, _commit.c_str());
                continue;
            }

            if (_name == "main") {
                name = _name;
                commit = _commit;
                break;
            }

            if (name.empty() || commit.empty()) {
                name = _name;
                commit = _commit;
            }
        }

        if (name.empty() || commit.empty()) {
            LOG_WRN("%s: no valid branch for '%s'\n", __func__, repo_id.c_str());
            return {};
        }

        safe_write_file(refs_path / name, commit);
        return commit;

    } catch (const nl::json::exception & e) {
        LOG_ERR("%s: JSON error: %s\n", __func__, e.what());
    } catch (const std::exception & e) {
        LOG_ERR("%s: error: %s\n", __func__, e.what());
    }
    return {};
}

hf_files get_repo_files(const std::string & repo_id,
                        const std::string & token) {
    if (!is_valid_repo_id(repo_id)) {
        LOG_WRN("%s: invalid repository: %s\n", __func__, repo_id.c_str());
        return {};
    }

    std::string commit = get_repo_commit(repo_id, token);
    if (commit.empty()) {
        LOG_WRN("%s: failed to resolve commit for %s\n", __func__, repo_id.c_str());
        return {};
    }

    fs::path blobs_path = get_repo_path(repo_id) / "blobs";
    fs::path commit_path = get_repo_path(repo_id) / "snapshots" / commit;

    hf_files files;

    try {
        auto endpoint = get_model_endpoint();
        auto json = api_get(endpoint + "api/models/" + repo_id + "/tree/" + commit + "?recursive=true", token);

        if (!json.is_array()) {
            LOG_WRN("%s: response is not an array for '%s'\n", __func__, repo_id.c_str());
            return {};
        }

        for (const auto & item : json) {
            if (!item.is_object() ||
                !item.contains("type") || !item["type"].is_string() || item["type"] != "file" ||
                !item.contains("path") || !item["path"].is_string()) {
                continue;
            }

            hf_file file;
            file.repo_id = repo_id;
            file.path = item["path"].get<std::string>();

            if (!is_valid_subpath(commit_path, file.path)) {
                LOG_WRN("%s: skip invalid path: %s\n", __func__, file.path.c_str());
                continue;
            }

            if (item.contains("lfs") && item["lfs"].is_object()) {
                if (item["lfs"].contains("oid") && item["lfs"]["oid"].is_string()) {
                    file.oid = item["lfs"]["oid"].get<std::string>();
                }
                if (item["lfs"].contains("size") && item["lfs"]["size"].is_number()) {
                    file.size = item["lfs"]["size"].get<size_t>();
                }
            } else if (item.contains("oid") && item["oid"].is_string()) {
                file.oid = item["oid"].get<std::string>();
            }
            if (file.size == 0 && item.contains("size") && item["size"].is_number()) {
                file.size = item["size"].get<size_t>();
            }

            if (!file.oid.empty() && !is_valid_oid(file.oid)) {
                LOG_WRN("%s: skip invalid oid: %s\n", __func__, file.oid.c_str());
                continue;
            }

            file.url = endpoint + repo_id + "/resolve/" + commit + "/" + file.path;

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
        LOG_ERR("%s: error: %s\n", __func__, e.what());
    }
    return files;
}

static std::string get_cached_ref(const fs::path & repo_path) {
    fs::path refs_path = repo_path / "refs";
    if (!fs::is_directory(refs_path)) {
        return {};
    }
    std::string fallback;

    for (const auto & entry : fs::directory_iterator(refs_path)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        std::ifstream f(entry.path());
        std::string commit;
        if (!f || !std::getline(f, commit) || commit.empty()) {
            continue;
        }
        if (!is_valid_commit(commit)) {
            LOG_WRN("%s: skip invalid commit: %s\n", __func__, commit.c_str());
            continue;
        }
        if (entry.path().filename() == "main") {
            return commit;
        }
        if (fallback.empty()) {
            fallback = commit;
        }
    }
    return fallback;
}

hf_files get_cached_files(const std::string & repo_id) {
    fs::path cache_dir = get_cache_directory();
    if (!fs::exists(cache_dir)) {
        return {};
    }

    if (!repo_id.empty() && !is_valid_repo_id(repo_id)) {
        LOG_WRN("%s: invalid repository: %s\n", __func__, repo_id.c_str());
        return {};
    }

    hf_files files;

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
        std::string commit = get_cached_ref(repo.path());
        fs::path commit_path = snapshots_path / commit;

        if (commit.empty() || !fs::is_directory(commit_path)) {
            continue;
        }
        for (const auto & entry : fs::recursive_directory_iterator(commit_path)) {
            if (!entry.is_regular_file() && !entry.is_symlink()) {
                continue;
            }
            fs::path path = entry.path().lexically_relative(commit_path);

            if (!path.empty()) {
                hf_file file;
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

std::string finalize_file(const hf_file & file) {
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

// delete everything after this line, one day

// copied from download.cpp without the tag part
struct gguf_split_info {
    std::string prefix; // tag included
    int index;
    int count;
};

static gguf_split_info get_gguf_split_info(const std::string & path) {
    static const std::regex re_split("^(.+)-([0-9]{5})-of-([0-9]{5})$", std::regex::icase);
    std::smatch m;

    std::string prefix = path;
    if (!string_remove_suffix(prefix, ".gguf")) {
        return {};
    }

    int index = 1;
    int count = 1;

    if (std::regex_match(prefix, m, re_split)) {
        index = std::stoi(m[2].str());
        count = std::stoi(m[3].str());
        prefix = m[1].str();
    }

    return {std::move(prefix), index, count};
}

static std::pair<std::string, std::string> parse_manifest_name(std::string & filename) {
    static const std::regex re(R"(^manifest=([^=]+)=([^=]+)=.*\.json$)");
    std::smatch match;
    if (std::regex_match(filename, match, re)) {
        return {match[1].str(), match[2].str()};
    }
    return {};
}

static std::string make_old_cache_filename(const std::string & owner,
                                           const std::string & repo,
                                           const std::string & filename) {
    auto result = owner + "_" + repo + "_" + filename;
    string_replace_all(result, "/", "_");
    return result;
}

struct migrate_file {
    std::string path;
    std::string sha256;
    size_t size;
    fs::path old_path;
    fs::path etag_path;
    const hf_file * file;
};

using migrate_files = std::vector<migrate_file>;

static bool collect_file(const fs::path    & old_cache,
                         const std::string & owner,
                         const std::string & repo,
                         const std::string & path,
                         const std::string & sha256,
                         const hf_files    & files,
                         migrate_files     & to_migrate) {

    const hf_file * file = nullptr;

    for (const auto & f : files) {
        if (f.path == path) {
            file = &f;
            break;
        }
    }

    std::string old_filename = make_old_cache_filename(owner, repo, path);
    fs::path old_path = old_cache / old_filename;
    fs::path etag_path = old_path.string() + ".etag";

    if (!fs::exists(old_path)) {
        if (file && fs::exists(file->final_path)) {
            return true;
        }
        LOG_WRN("%s: %s not found in old cache or HF cache\n", __func__, old_filename.c_str());
        return false;
    }

    if (!file) {
        LOG_WRN("%s: %s not found in current repo\n", __func__, old_filename.c_str());
        return false;
    }

    if (!sha256.empty() && !file->oid.empty() && sha256 != file->oid) {
        LOG_WRN("%s: %s is not up to date (sha256 mismatch)\n", __func__, old_filename.c_str());
        return false;
    }

    if (file->size > 0) {
        size_t size = fs::file_size(old_path);
        if (size != file->size) {
            LOG_WRN("%s: %s has wrong size %zu (expected %zu)\n", __func__, old_filename.c_str(), size, file->size);
            return false;
        }
    }

    to_migrate.push_back({path, sha256, file->size, old_path, etag_path, file});
    return true;
}

static bool collect_files(const fs::path    & old_cache,
                          const std::string & owner,
                          const std::string & repo,
                          const nl::json    & node,
                          const hf_files    & files,
                          migrate_files     & to_migrate) {

    if (!node.contains("rfilename") ||
        !node.contains("lfs")       ||
        !node["lfs"].contains("sha256")) {
        return true;
    }

    std::string path = node["rfilename"];
    std::string sha256 = node["lfs"]["sha256"];

    auto split = get_gguf_split_info(path);

    if (split.count <= 1) {
        return collect_file(old_cache, owner, repo, path, sha256, files, to_migrate);
    }

    std::vector<std::pair<std::string, std::string>> splits;

    for (const auto & f : files) {
        auto split_f = get_gguf_split_info(f.path);
        if (split_f.count == split.count && split_f.prefix == split.prefix) {
            // sadly the manifest only provides the sha256 of the first file (index == 1)
            // the rest will be verified using the size...
            std::string f_sha256 = (split_f.index == 1) ? sha256 : "";
            splits.emplace_back(f.path, f_sha256);
        }
    }

    if ((int)splits.size() != split.count) {
        LOG_WRN("%s: expected %d split files but found %d in repo\n", __func__, split.count, (int)splits.size());
        return false;
    }

    for (const auto & [f_path, f_sha256] : splits) {
        if (!collect_file(old_cache, owner, repo, f_path, f_sha256, files, to_migrate)) {
            return false;
        }
    }

    return true;
}

static bool migrate_file(const migrate_file & file) {
    std::error_code ec;

    fs::path new_path(file.file->local_path);
    fs::create_directories(new_path.parent_path(), ec);

    if (!fs::exists(new_path, ec)) {
        fs::rename(file.old_path, new_path, ec);
        if (ec) {
            fs::copy_file(file.old_path, new_path, ec);
            if (ec) {
                LOG_ERR("%s: failed to move/copy %s: %s\n", __func__, file.old_path.string().c_str(), ec.message().c_str());
                return false;
            }
        }
        fs::remove(file.old_path, ec);
    }
    fs::remove(file.etag_path, ec);

    std::string filename = finalize_file(*file.file);
    LOG_INF("%s: migrated %s -> %s\n", __func__, file.old_path.filename().string().c_str(), filename.c_str());
    return true;
}

void migrate_old_cache_to_hf_cache(const std::string & token, bool offline) {
    fs::path old_cache = fs_get_cache_directory();
    if (!fs::exists(old_cache)) {
        return;
    }

    if (offline) {
        LOG_WRN("%s: skipping migration in offline mode (will run when online)\n", __func__);
        return; // -hf is not going to work
    }

    bool warned = false;

    for (const auto & entry : fs::directory_iterator(old_cache)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        auto filename = entry.path().filename().string();
        auto [owner, repo] = parse_manifest_name(filename);

        if (owner.empty() || repo.empty()) {
            continue;
        }

        if (!warned) {
            warned = true;
            LOG_WRN("================================================================================\n"
                    "WARNING: Migrating cache to HuggingFace cache directory\n"
                    "  Old cache: %s\n"
                    "  New cache: %s\n"
                    "This one-time migration moves models previously downloaded with -hf\n"
                    "from the legacy llama.cpp cache to the standard HuggingFace cache.\n"
                    "Models downloaded with --model-url are not affected.\n"
                    "================================================================================\n",
                    old_cache.string().c_str(), get_cache_directory().string().c_str());
        }

        auto repo_id = owner + "/" + repo;
        auto files = get_repo_files(repo_id, token);

        if (files.empty()) {
            LOG_WRN("%s: could not get repo files for %s, skipping\n", __func__, repo_id.c_str());
            continue;
        }

        migrate_files to_migrate;
        bool ok = true;

        try {
            std::ifstream manifest(entry.path());
            auto json = nl::json::parse(manifest);
            for (const char * key : {"ggufFile", "mmprojFile"}) {
                if (json.contains(key)) {
                    if (!collect_files(old_cache, owner, repo, json[key], files, to_migrate)) {
                        ok = false;
                        break;
                    }
                }
            }
        } catch (const std::exception & e) {
            LOG_WRN("%s: failed to parse manifest %s: %s\n", __func__, filename.c_str(), e.what());
            continue;
        }

        if (!ok) {
            LOG_WRN("%s: migration skipped: one or more files failed validation\n", __func__);
            continue;
        }

        for (const auto & file : to_migrate) {
            if (!migrate_file(file)) {
                ok = false;
                break;
            }
        }

        if (!ok) {
            LOG_WRN("%s: migration failed: could not migrate all files\n", __func__);
            continue;
        }

        LOG_INF("%s: migration complete, deleting manifest: %s\n", __func__, entry.path().string().c_str());
        fs::remove(entry.path());
    }
}

} // namespace hf_cache
