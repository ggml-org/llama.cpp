#include "hf-cache.h"

#include "common.h"
#include "log.h"
#include "http.h"

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

#include <ctime>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <regex> // migration only
#include <string>

namespace nl = nlohmann;

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace hf_cache {

namespace fs = std::filesystem;

static fs::path get_cache_directory() {
    const char * hf_hub_cache = std::getenv("HF_HUB_CACHE");
    if (hf_hub_cache && *hf_hub_cache) {
        return fs::path(hf_hub_cache);  // assume shell-expanded; add expand logic if you want full parity
    }

    const char * huggingface_hub_cache = std::getenv("HUGGINGFACE_HUB_CACHE");
    if (huggingface_hub_cache && *huggingface_hub_cache) {
        return fs::path(huggingface_hub_cache);
    }

    const char * hf_home = std::getenv("HF_HOME");
    if (hf_home && *hf_home) {
        return fs::path(hf_home) / "hub";
    }

    const char * xdg_cache_home = std::getenv("XDG_CACHE_HOME");
    if (xdg_cache_home && *xdg_cache_home) {
        return fs::path(xdg_cache_home) / "huggingface" / "hub";
    }
#if defined(_WIN32)
    const char * userprofile = std::getenv("USERPROFILE");
    if (userprofile && *userprofile) {
        return fs::path(userprofile) / ".cache" / "huggingface" / "hub";
    }
#else
    const char * home = std::getenv("HOME");
    if (home && *home) {
        return fs::path(home) / ".cache" / "huggingface" / "hub";
    }
#endif
    throw std::runtime_error("Failed to determine HF cache directory");
}

static bool symlinks_supported() {
#ifdef _WIN32
    static bool supported = false;
    static std::once_flag once;
    std::call_once(once, []() {
        fs::path link = get_cache_directory() / ("link_" + std::to_string(GetCurrentProcessId()));

        std::error_code ec;
        fs::create_directory_symlink("..", link, ec);
        supported = !ec;

        if (!ec) {
            fs::remove(link, ec);
        } else if (GetLastError() == ERROR_PRIVILEGE_NOT_HELD) {
            LOG_WRN("symlink creation requires Developer Mode or admin privileges on Windows\n");
        }
    });
    return supported;
#else
    return true;
#endif
}

static std::string folder_name_to_repo(const std::string & folder) {
    if (folder.size() < 8 || folder.substr(0, 8) != "models--") {
        return {};
    }
    std::string repo_id;
    for (size_t i = 8; i < folder.size(); ++i) {
        if (i + 1 < folder.size() && folder[i] == '-' && folder[i+1] == '-') {
            repo_id += '/';
            i++;
        } else {
            repo_id += folder[i];
        }
    }
    return repo_id;
}

static std::string repo_to_folder_name(const std::string & repo_id) {
    std::string name = "models--";
    for (char c : repo_id) {
        if (c == '/') {
            name += "--";
        } else {
            name += c;
        }
    }
    return name;
}

static fs::path get_repo_path(const std::string & repo_id) {
    return get_cache_directory() / repo_to_folder_name(repo_id);
}

static void write_ref(const std::string & repo_id,
                      const std::string & ref,
                      const std::string & commit) {
    fs::path refs_path = get_repo_path(repo_id) / "refs";
    std::error_code ec;
    fs::create_directories(refs_path, ec);

    fs::path ref_path = refs_path / ref;
    fs::path ref_path_tmp = refs_path / (ref + ".tmp");
    {
        std::ofstream file(ref_path_tmp);
        if (!file) {
            throw std::runtime_error("Failed to write ref file: " + ref_path.string());
        }
        file << commit;
    }
    std::error_code rename_ec;
    fs::rename(ref_path_tmp, ref_path, rename_ec);
    if (rename_ec) {
        LOG_ERR("%s: unable to rename file: %s to %s\n", __func__, ref_path_tmp.c_str(), ref_path.c_str());
        fs::remove(ref_path_tmp, ec);
    }
}

static std::string get_repo_ref(const std::string & repo_id,
                                const std::string & bearer_token) {
    std::string url = get_model_endpoint() + "api/models/" + repo_id + "/refs";
    auto [cli, parts] = common_http_client(url);

    httplib::Headers headers;
    headers.emplace("User-Agent", "llama-cpp/" + build_info);
    headers.emplace("Accept", "application/json");
    if (!bearer_token.empty()) {
        headers.emplace("Authorization", "Bearer " + bearer_token);
    }
    cli.set_default_headers(headers);

    auto res = cli.Get(parts.path);
    if (!res || res->status != 200) {
        LOG_WRN("%s: API request failed for %s, status: %d\n", __func__, url.c_str(), res ? res->status : -1);
        return {};
    }

    try {
        auto j = nl::json::parse(res->body);

        if (!j.contains("branches") || !j["branches"].is_array()) {
            return {};
        }

        std::string name;
        std::string commit;

        for (const auto & branch : j["branches"]) {
            if (!branch.contains("name") || !branch.contains("targetCommit")) {
                continue;
            }
            std::string _name = branch["name"].get<std::string>();
            std::string _commit = branch["targetCommit"].get<std::string>();

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

        if (!name.empty() && !commit.empty()) {
            write_ref(repo_id, name, commit);
        }
        return commit;
    } catch (const std::exception & e) {
        LOG_WRN("%s: failed to parse API response: %s\n", __func__, e.what());
        return {};
    }
}

hf_files get_repo_files(const std::string & repo_id,
                        const std::string & bearer_token) {
    std::string rev = get_repo_ref(repo_id, bearer_token);
    if (rev.empty()) {
        LOG_WRN("%s: failed to resolve commit hash for %s\n", __func__, repo_id.c_str());
        return {};
    }

    std::string url = get_model_endpoint() + "api/models/" + repo_id + "/tree/" + rev + "?recursive=true";

    auto [cli, parts] = common_http_client(url);

    httplib::Headers headers;
    headers.emplace("User-Agent", "llama-cpp/" + build_info);
    headers.emplace("Accept", "application/json");
    if (!bearer_token.empty()) {
        headers.emplace("Authorization", "Bearer " + bearer_token);
    }
    cli.set_default_headers(headers);

    auto res = cli.Get(parts.path);
    if (!res || res->status != 200) {
        LOG_WRN("%s: API request failed for %s, status: %d\n", __func__, url.c_str(), res ? res->status : -1);
        return {};
    }

    std::string endpoint = get_model_endpoint(); // TODO
    bool use_symlinks = symlinks_supported();
    hf_files files;

    try {
        auto j = nl::json::parse(res->body);

        if (!j.is_array()) {
            LOG_DBG("%s: response is not an array\n", __func__);
            return files;
        }

        for (const auto & item : j) {
            if (!item.contains("type") || item["type"] != "file") {
                continue;
            }
            if (!item.contains("path")) {
                continue;
            }

            hf_file file;
            file.repo_id = repo_id;
            file.path = item["path"].get<std::string>();

            if (item.contains("lfs") && item["lfs"].is_object()) {
                if (item["lfs"].contains("oid") && item["lfs"]["oid"].is_string()) {
                    file.oid = item["lfs"]["oid"].get<std::string>();
                }
            } else if (item.contains("oid") && item["oid"].is_string()) {
                file.oid = item["oid"].get<std::string>();
            }

            file.url = endpoint + repo_id + "/resolve/" + rev + "/" + file.path;

            fs::path path = file.path;
            fs::path repo_path = get_repo_path(repo_id);
            fs::path snapshots_path = repo_path / "snapshots" / rev / path;
            fs::path blobs_path = repo_path / "blobs" / file.oid;

            if (use_symlinks) {
                file.local_path = blobs_path.string();
                file.link_path = snapshots_path.string();
            } else { // degraded mode
                file.local_path = snapshots_path.string();
            }

            files.push_back(file);
        }
    } catch (const std::exception & e) {
        LOG_WRN("%s: failed to parse API response: %s\n", __func__, e.what());
        return {};
    }

    return files;
}

static std::string get_cached_ref(const fs::path & repo_path) {
    fs::path refs_path = repo_path / "refs";
    if (!fs::is_directory(refs_path)) {
        return {};
    }
    for (const auto & entry : fs::directory_iterator(refs_path)) {
        if (entry.is_regular_file()) {
            std::ifstream f(entry.path());
            std::string commit;
            if (f && std::getline(f, commit) && !commit.empty()) {
                return commit;
            }
        }
    }
    return {};
}

hf_files get_cached_files(const std::string & repo_id) {
    fs::path cache_dir = get_cache_directory();
    if (!fs::exists(cache_dir)) {
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

        if (_repo_id.empty()) {
            continue;
        }
        if (!repo_id.empty() && _repo_id != repo_id) {
            continue;
        }
        std::string commit = get_cached_ref(repo.path());
        fs::path rev_path = snapshots_path / commit;

        if (commit.empty() || !fs::is_directory(rev_path)) {
            continue;
        }
        for (const auto & entry : fs::recursive_directory_iterator(rev_path)) {
            if (!entry.is_regular_file() && !entry.is_symlink()) {
                continue;
            }
            fs::path path = entry.path().lexically_relative(rev_path);

            if (!path.empty()) {
                hf_file file;
                file.repo_id = _repo_id;
                file.path = path.generic_string();
                file.local_path = entry.path().string();
                files.push_back(std::move(file));
            }
        }
    }

    return files;
}

std::string finalize_file(const hf_file & file) {
    if (file.link_path.empty()) {
        return file.local_path;
    }

    fs::path link_path(file.link_path);
    fs::path local_path(file.local_path);

    std::error_code ec;
    fs::create_directories(link_path.parent_path(), ec);
    fs::path target_path = fs::relative(local_path, link_path.parent_path(), ec);
    fs::create_symlink(target_path, link_path, ec);

    if (fs::exists(link_path)) {
        return file.link_path;
    }

    LOG_WRN("%s: failed to create symlink: %s\n", __func__, file.link_path.c_str());
    return file.local_path;
}

// delete everything after this line, one day

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
    std::string name = owner + "_" + repo + "_" + filename;
    for (char & c : name) {
        if (c == '/') {
            c = '_';
        }
    }
    return name;
}

static bool migrate_single_file(const fs::path    & old_cache,
                                const std::string & owner,
                                const std::string & repo,
                                const nl::json    & node,
                                const hf_files    & files) {

    if (!node.contains("rfilename") ||
        !node.contains("lfs")       ||
        !node["lfs"].contains("sha256")) {
        return false;
    }

    std::string path = node["rfilename"];
    std::string sha256 = node["lfs"]["sha256"];

    const hf_file * file_info = nullptr;
    for (const auto & f : files) {
        if (f.path == path) {
            file_info = &f;
            break;
        }
    }

    std::string old_filename = make_old_cache_filename(owner, repo, path);
    fs::path old_path = old_cache / old_filename;
    fs::path etag_path = old_path.string() + ".etag";

    if (!fs::exists(old_path)) {
        if (fs::exists(etag_path)) {
            LOG_WRN("%s: %s is orphan, deleting...\n", __func__, etag_path.string().c_str());
            fs::remove(etag_path);
        }
        return false;
    }

    bool delete_old_path = false;

    if (!file_info) {
        LOG_WRN("%s: %s not found in current repo, deleting...\n", __func__, old_filename.c_str());
        delete_old_path = true;
    } else if (!sha256.empty() && !file_info->oid.empty() && sha256 != file_info->oid) {
        LOG_WRN("%s: %s is not up to date (sha256 mismatch), deleting...\n", __func__, old_filename.c_str());
        delete_old_path = true;
    }

    std::error_code ec;

    if (delete_old_path) {
        fs::remove(old_path, ec);
        fs::remove(etag_path, ec);
        return true;
    }

    fs::path new_path(file_info->local_path);
    fs::create_directories(new_path.parent_path(), ec);

    if (!fs::exists(new_path, ec)) {
        fs::rename(old_path, new_path, ec);
        if (ec) {
            fs::copy_file(old_path, new_path, ec);
            if (ec) {
                LOG_WRN("%s: failed to move/copy %s: %s\n", __func__, old_path.string().c_str(), ec.message().c_str());
                return false;
            }
        }
        fs::remove(old_path, ec);
    }
    fs::remove(etag_path, ec);

    std::string snapshot_path = finalize_file(*file_info);
    LOG_INF("%s: migrated %s -> %s\n", __func__, old_filename.c_str(), snapshot_path.c_str());

    return true;
}

void migrate_old_cache_to_hf_cache(const std::string & bearer_token, bool offline) {
    fs::path old_cache = fs_get_cache_directory();
    if (!fs::exists(old_cache)) {
        return;
    }

    if (offline) {
        LOG_WRN("%s: skipping migration in offline mode (will run when online)\n", __func__);
        return; // -hf is not going to work
    }

    for (const auto & entry : fs::directory_iterator(old_cache)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        auto filename = entry.path().filename().string();
        auto [owner, repo] = parse_manifest_name(filename);

        if (owner.empty() || repo.empty()) {
            continue;
        }

        auto repo_id = owner + "/" + repo;
        auto files = get_repo_files(repo_id, bearer_token);

        if (files.empty()) {
            LOG_WRN("%s: could not get repo files for %s, skipping\n", __func__, repo_id.c_str());
            continue;
        }

        try {
            std::ifstream manifest_stream(entry.path());
            std::string content((std::istreambuf_iterator<char>(manifest_stream)), std::istreambuf_iterator<char>());
            auto j = nl::json::parse(content);
            for (const char* key : {"ggufFile", "mmprojFile"}) {
                if (j.contains(key)) {
                    migrate_single_file(old_cache, owner, repo, j[key], files);
                }
            }
        } catch (const std::exception & e) {
            LOG_WRN("%s: failed to parse manifest %s: %s\n", __func__, filename.c_str(), e.what());
            continue;
        }
        fs::remove(entry.path());
    }
}

} // namespace hf_cache
