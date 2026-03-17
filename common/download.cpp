#include "arg.h"

#include "common.h"
#include "log.h"
#include "download.h"
#include "hf-cache.h"

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <future>
#include <map>
#include <mutex>
#include <regex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "http.h"

#ifndef __EMSCRIPTEN__
#ifdef __linux__
#include <linux/limits.h>
#elif defined(_WIN32)
#   if !defined(PATH_MAX)
#   define PATH_MAX MAX_PATH
#   endif
#elif defined(_AIX)
#include <sys/limits.h>
#else
#include <sys/syslimits.h>
#endif
#endif

// isatty
#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

using json = nlohmann::ordered_json;

//
// downloader
//

// validate repo name format: owner/repo
static void write_file(const std::string & fname, const std::string & content) {
    const std::string fname_tmp = fname + ".tmp";
    std::ofstream     file(fname_tmp);
    if (!file) {
        throw std::runtime_error(string_format("error: failed to open file '%s'\n", fname.c_str()));
    }

    try {
        file << content;
        file.close();

        // Makes write atomic
        if (rename(fname_tmp.c_str(), fname.c_str()) != 0) {
            LOG_ERR("%s: unable to rename file: %s to %s\n", __func__, fname_tmp.c_str(), fname.c_str());
            // If rename fails, try to delete the temporary file
            if (remove(fname_tmp.c_str()) != 0) {
                LOG_ERR("%s: unable to delete temporary file: %s\n", __func__, fname_tmp.c_str());
            }
        }
    } catch (...) {
        // If anything fails, try to delete the temporary file
        if (remove(fname_tmp.c_str()) != 0) {
            LOG_ERR("%s: unable to delete temporary file: %s\n", __func__, fname_tmp.c_str());
        }

        throw std::runtime_error(string_format("error: failed to write file '%s'\n", fname.c_str()));
    }
}

static void write_etag(const std::string & path, const std::string & etag) {
    const std::string etag_path = path + ".etag";
    write_file(etag_path, etag);
    LOG_DBG("%s: file etag saved: %s\n", __func__, etag_path.c_str());
}

static std::string read_etag(const std::string & path) {
    const std::string etag_path = path + ".etag";
    if (!std::filesystem::exists(etag_path)) {
        return {};
    }
    std::ifstream etag_in(etag_path);
    if (!etag_in) {
        LOG_ERR("%s: could not open .etag file for reading: %s\n", __func__, etag_path.c_str());
        return {};
    }
    std::string etag;
    std::getline(etag_in, etag);
    return etag;
}

static bool is_http_status_ok(int status) {
    return status >= 200 && status < 400;
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

class ProgressBar {
    static inline std::mutex mutex;
    static inline std::map<const ProgressBar *, int> lines;
    static inline int max_line = 0;

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
    ProgressBar() = default;

    ~ProgressBar() {
        std::lock_guard<std::mutex> lock(mutex);
        cleanup(this);
    }

    void update(size_t current, size_t total) {
        if (!is_output_a_tty()) {
            return;
        }

        if (!total) {
            return;
        }

        std::lock_guard<std::mutex> lock(mutex);

        if (lines.find(this) == lines.end()) {
            lines[this] = max_line++;
            std::cout << "\n";
        }
        int lines_up = max_line - lines[this];

        size_t width = 50;
        size_t pct = (100 * current) / total;
        size_t pos = (width * current) / total;

        std::cout << "\033[s";

        if (lines_up > 0) {
            std::cout << "\033[" << lines_up << "A";
        }
        std::cout << "\033[2K\r["
            << std::string(pos, '=')
            << (pos < width ? ">" : "")
            << std::string(width - pos, ' ')
            << "] " << std::setw(3) << pct << "%  ("
            << current / (1024 * 1024) << " MB / "
            << total / (1024 * 1024) << " MB) "
            << "\033[u";

        std::cout.flush();

        if (current == total) {
             cleanup(this);
        }
    }

    ProgressBar(const ProgressBar &) = delete;
    ProgressBar & operator=(const ProgressBar &) = delete;
};

static bool common_pull_file(httplib::Client & cli,
                             const std::string & resolve_path,
                             const std::string & path_tmp,
                             bool supports_ranges,
                             size_t existing_size,
                             size_t & total_size) {
    std::ofstream ofs(path_tmp, std::ios::binary | std::ios::app);
    if (!ofs.is_open()) {
        LOG_ERR("%s: error opening local file for writing: %s\n", __func__, path_tmp.c_str());
        return false;
    }

    httplib::Headers headers;
    if (supports_ranges && existing_size > 0) {
        headers.emplace("Range", "bytes=" + std::to_string(existing_size) + "-");
    }

    const char * func = __func__; // avoid __func__ inside a lambda
    size_t downloaded = existing_size;
    size_t progress_step = 0;
    ProgressBar bar;

    auto res = cli.Get(resolve_path, headers,
        [&](const httplib::Response &response) {
            if (existing_size > 0 && response.status != 206) {
                LOG_WRN("%s: server did not respond with 206 Partial Content for a resume request. Status: %d\n", func, response.status);
                return false;
            }
            if (existing_size == 0 && response.status != 200) {
                LOG_WRN("%s: download received non-successful status code: %d\n", func, response.status);
                return false;
            }
            if (total_size == 0 && response.has_header("Content-Length")) {
                try {
                    size_t content_length = std::stoull(response.get_header_value("Content-Length"));
                    total_size = existing_size + content_length;
                } catch (const std::exception &e) {
                    LOG_WRN("%s: invalid Content-Length header: %s\n", func, e.what());
                }
            }
            return true;
        },
        [&](const char *data, size_t len) {
            ofs.write(data, len);
            if (!ofs) {
                LOG_ERR("%s: error writing to file: %s\n", func, path_tmp.c_str());
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
        return false;
    }

    return true;
}

// download one single file from remote URL to local path
// returns status code or -1 on error
static int common_download_file_single_online(const std::string        & url,
                                              const std::string        & path,
                                              const std::string        & bearer_token,
                                              const common_header_list & custom_headers,
                                              bool                       skip_etag = false) {
    static const int max_attempts        = 3;
    static const int retry_delay_seconds = 2;

    auto [cli, parts] = common_http_client(url);

    httplib::Headers headers;
    for (const auto & h : custom_headers) {
        headers.emplace(h.first, h.second);
    }
    if (headers.find("User-Agent") == headers.end()) {
        headers.emplace("User-Agent", "llama-cpp/" + build_info);
    }
    if (!bearer_token.empty()) {
        headers.emplace("Authorization", "Bearer " + bearer_token);
    }
    cli.set_default_headers(headers);

    const bool file_exists = std::filesystem::exists(path);

    if (file_exists && skip_etag) {
        LOG_INF("%s: using cached file: %s\n", __func__, path.c_str());
        return 304; // 304 Not Modified - fake cached response
    }

    std::string last_etag;
    if (file_exists) {
        last_etag = read_etag(path);
    } else {
        LOG_INF("%s: no previous model file found %s\n", __func__, path.c_str());
    }

    auto head = cli.Head(parts.path);
    if (!head || head->status < 200 || head->status >= 300) {
        LOG_WRN("%s: HEAD failed, status: %d\n", __func__, head ? head->status : -1);
        if (file_exists) {
            LOG_INF("%s: using cached file (HEAD failed): %s\n", __func__, path.c_str());
            return 304; // 304 Not Modified - fake cached response
        }
        return head ? head->status : -1;
    }

    std::string etag;
    if (head->has_header("ETag")) {
        etag = head->get_header_value("ETag");
    }

    size_t total_size = 0;
    if (head->has_header("Content-Length")) {
        try {
            total_size = std::stoull(head->get_header_value("Content-Length"));
        } catch (const std::exception& e) {
            LOG_WRN("%s: invalid Content-Length in HEAD response: %s\n", __func__, e.what());
        }
    }

    bool supports_ranges = false;
    if (head->has_header("Accept-Ranges")) {
        supports_ranges = head->get_header_value("Accept-Ranges") != "none";
    }

    if (file_exists) {
        if (etag.empty()) {
            LOG_INF("%s: using cached file (no server etag): %s\n", __func__, path.c_str());
            return 304; // 304 Not Modified - fake cached response
        }
        if (!last_etag.empty() && last_etag == etag) {
            LOG_INF("%s: using cached file (same etag): %s\n", __func__, path.c_str());
            return 304; // 304 Not Modified - fake cached response
        }
        if (remove(path.c_str()) != 0) {
            LOG_ERR("%s: unable to delete file: %s\n", __func__, path.c_str());
            return -1;
        }
    }

    { // silent
        std::error_code ec;
        std::filesystem::path p(path);
        std::filesystem::create_directories(p.parent_path(), ec);
    }

    const std::string path_temporary = path + ".downloadInProgress";
    int delay = retry_delay_seconds;

    for (int i = 0; i < max_attempts; ++i) {
        if (i) {
            LOG_WRN("%s: retrying after %d seconds...\n", __func__, delay);
            std::this_thread::sleep_for(std::chrono::seconds(delay));
            delay *= retry_delay_seconds;
        }

        size_t existing_size = 0;

        if (std::filesystem::exists(path_temporary)) {
            if (supports_ranges) {
                existing_size = std::filesystem::file_size(path_temporary);
            } else if (remove(path_temporary.c_str()) != 0) {
                LOG_ERR("%s: unable to delete file: %s\n", __func__, path_temporary.c_str());
                return -1;
            }
        }

        LOG_INF("%s: downloading from %s to %s (etag:%s)...\n",
                __func__, common_http_show_masked_url(parts).c_str(),
                path_temporary.c_str(), etag.c_str());

        if (common_pull_file(cli, parts.path, path_temporary, supports_ranges, existing_size, total_size)) {
            if (std::rename(path_temporary.c_str(), path.c_str()) != 0) {
                LOG_ERR("%s: unable to rename file: %s to %s\n", __func__, path_temporary.c_str(), path.c_str());
                return -1;
            }
            if (!etag.empty() && !skip_etag) {
                write_etag(path, etag);
            }
            return head->status;
        }
    }

    LOG_ERR("%s: download failed after %d attempts\n", __func__, max_attempts);
    return -1; // max attempts reached
}

std::pair<long, std::vector<char>> common_remote_get_content(const std::string          & url,
                                                             const common_remote_params & params) {
    auto [cli, parts] = common_http_client(url);

    httplib::Headers headers;
    for (const auto & h : params.headers) {
        headers.emplace(h.first, h.second);
    }
    if (headers.find("User-Agent") == headers.end()) {
        headers.emplace("User-Agent", "llama-cpp/" + build_info);
    }

    if (params.timeout > 0) {
        cli.set_read_timeout(params.timeout, 0);
        cli.set_write_timeout(params.timeout, 0);
    }

    std::vector<char> buf;
    auto res = cli.Get(parts.path, headers,
        [&](const char *data, size_t len) {
            buf.insert(buf.end(), data, data + len);
            return params.max_size == 0 ||
                   buf.size() <= static_cast<size_t>(params.max_size);
        },
        nullptr
    );

    if (!res) {
        throw std::runtime_error("error: cannot make GET request");
    }

    return { res->status, std::move(buf) };
}

int common_download_file_single(const std::string & url,
                                const std::string & path,
                                const std::string & bearer_token,
                                bool offline,
                                const common_header_list & headers,
                                bool skip_etag) {
    if (!offline) {
        return common_download_file_single_online(url, path, bearer_token, headers, skip_etag);
    }

    if (!std::filesystem::exists(path)) {
        LOG_ERR("%s: required file is not available in cache (offline mode): %s\n", __func__, path.c_str());
        return -1;
    }

    LOG_INF("%s: using cached file (offline mode): %s\n", __func__, path.c_str());
    return 304; // Not Modified - fake cached response
}

// "subdir/model-00001-of-00002.gguf" -> "subdir/model", 1, 2
static std::tuple<std::string, int, int> get_gguf_split_info(const std::string & path) {
    if (path.empty()) {
        return {};
    }

    static const std::regex re(R"(^(.+)-([0-9]+)-of-([0-9]+)\.gguf$)", std::regex::icase);

    std::smatch m;
    if (std::regex_match(path, m, re)) {
        return {m[1].str(), std::stoi(m[2].str()), std::stoi(m[3].str())};
    }
    return {};
}

static hf_cache::hf_files get_split_files(const hf_cache::hf_files & all_files,
                                          const hf_cache::hf_file & primary_file) {
    hf_cache::hf_files result;
    auto [prefix, idx, count] = get_gguf_split_info(primary_file.path);

    if (count > 1) {
        for (const auto & f : all_files) {
            auto [sprefix, sidx, scount] = get_gguf_split_info(f.path);
            if (scount == count && sprefix == prefix) {
                result.push_back(f);
            }
        }
    } else {
        result.push_back(primary_file);
    }
    return result;
}

static hf_cache::hf_files filter_gguf_by_quant(const hf_cache::hf_files & files,
                                               const std::string & quant_tag) {
    hf_cache::hf_files matches;
    std::regex pattern(quant_tag + "[.-]", std::regex::icase);

    for (const auto & f : files) {
        if (!string_ends_with(f.path, ".gguf")) {
            continue;
        }
        if (f.path.find("mmproj") != std::string::npos) {
            continue;
        }
        if (std::regex_search(f.path, pattern)) {
            matches.push_back(f);
        }
    }
    return matches;
}

static void list_available_gguf_files(const hf_cache::hf_files & files) {
    LOG_INF("Available GGUF files:\n");
    for (const auto & f : files) {
        if (string_ends_with(f.path, ".gguf")) {
            LOG_INF(" - %s\n", f.path.c_str());
        }
    }
}

struct hf_plan {
    hf_cache::hf_file primary;
    hf_cache::hf_file mmproj;
    bool has_primary = false;
    bool has_mmproj = false;
    hf_cache::hf_files files;
};

static hf_plan get_hf_plan(const common_params_model        & model,
                           const std::string                & token,
                           const common_download_model_opts & opts) {
    hf_plan plan;
    auto [repo, tag] = common_download_split_repo_tag(model.hf_repo);

    auto all = opts.offline ? hf_cache::get_cached_files(repo)
                            : hf_cache::get_repo_files(repo, token);
    if (all.empty()) {
        return plan;
    }

    hf_cache::hf_files candidates;

    if (!model.hf_file.empty()) {
        const hf_cache::hf_file * found_file = nullptr;
        for (const auto & f : all) {
            if (f.path == model.hf_file) {
                found_file = &f;
                break;
            }
        }

        if (!found_file) {
            LOG_ERR("%s: --hf-file '%s' not found in repository\n", __func__, model.hf_file.c_str());
            list_available_gguf_files(all);
            return plan;
        }

        plan.primary = *found_file;
        plan.has_primary = true;
        candidates = get_split_files(all, *found_file);
    } else {
        std::vector<std::string> search_priority = {!tag.empty() ? tag : "Q4_K_M", "Q4_0"};

        for (const auto & q : search_priority) {
            candidates = filter_gguf_by_quant(all, q);
            if (!candidates.empty()) {
                candidates = get_split_files(all, candidates[0]);
                break;
            }
        }

        if (candidates.empty()) {
            for (const auto & f : all) {
                if (string_ends_with(f.path, ".gguf") &&
                    f.path.find("mmproj") == std::string::npos) {
                    candidates = get_split_files(all, f);
                    break;
                }
            }
        }

        if (candidates.empty()) {
            LOG_ERR("%s: no GGUF files found in repository %s\n", __func__, repo.c_str());
            list_available_gguf_files(all);
            return plan;
        }

        plan.primary = candidates[0];
        plan.has_primary = true;
    }

    for (const auto & f : candidates) {
        plan.files.push_back(f);
    }

    if (opts.download_mmproj) {
        for (const auto & f : all) {
            if (string_ends_with(f.path, ".gguf") &&
                f.path.find("mmproj") != std::string::npos) {
                plan.mmproj = f;
                plan.has_mmproj = true;
                plan.files.push_back(f);
                break;
            }
        }
    }

    return plan;
}

static std::vector<std::pair<std::string, std::string>> get_url_tasks(const common_params_model & model) {
    auto [prefix_url, idx, count] = get_gguf_split_info(model.url);

    if (count <= 1) {
        return {{model.url, model.path}};
    }

    std::vector<std::pair<std::string, std::string>> files;

    size_t pos = prefix_url.rfind('/');
    std::string prefix_filename = (pos != std::string::npos) ? prefix_url.substr(pos + 1) : prefix_url;
    std::string prefix_path = (std::filesystem::path(model.path).parent_path() / prefix_filename).string();

    for (int i = 1; i <= count; i++) {
        std::string suffix = string_format("-%05d-of-%05d.gguf", i, count);
        files.emplace_back(prefix_url + suffix, prefix_path + suffix);
    }
    return files;
}

common_download_model_result common_download_model(const common_params_model        & model,
                                                   const std::string                & bearer_token,
                                                   const common_download_model_opts & opts,
                                                   const common_header_list         & headers) {
    common_download_model_result result;
    std::vector<std::pair<std::string, std::string>> to_download;
    hf_plan hf;

    bool is_hf = !model.hf_repo.empty();

    if (is_hf) {
        hf = get_hf_plan(model, bearer_token, opts);
        for (const auto & f : hf.files) {
            to_download.emplace_back(f.url, f.local_path);
        }
    } else if (!model.url.empty()) {
        to_download = get_url_tasks(model);
    } else {
        result.model_path = model.path;
        return result;
    }

    if (to_download.empty()) {
        return result;
    }

    std::vector<std::future<bool>> futures;
    for (const auto & item : to_download) {
        futures.push_back(std::async(std::launch::async,
            [u = item.first, p = item.second, &bearer_token, offline = opts.offline, &headers, is_hf]() {
                int status = common_download_file_single(u, p, bearer_token, offline, headers, is_hf);
                return is_http_status_ok(status);
            }
        ));
    }

    for (auto & f : futures) {
        if (!f.get()) {
            return {};
        }
    }

    if (is_hf) {
        for (const auto & f : hf.files) {
            hf_cache::finalize_file(f);
        }
        if (hf.has_primary) {
            result.model_path = hf_cache::finalize_file(hf.primary);
        }
        if (hf.has_mmproj) {
            result.mmproj_path = hf_cache::finalize_file(hf.mmproj);
        }
    } else {
        result.model_path = model.path;
    }

    return result;
}

//
// Docker registry functions
//

static std::string common_docker_get_token(const std::string & repo) {
    std::string url = "https://auth.docker.io/token?service=registry.docker.io&scope=repository:" + repo + ":pull";

    common_remote_params params;
    auto                 res = common_remote_get_content(url, params);

    if (res.first != 200) {
        throw std::runtime_error("Failed to get Docker registry token, HTTP code: " + std::to_string(res.first));
    }

    std::string            response_str(res.second.begin(), res.second.end());
    nlohmann::ordered_json response = nlohmann::ordered_json::parse(response_str);

    if (!response.contains("token")) {
        throw std::runtime_error("Docker registry token response missing 'token' field");
    }

    return response["token"].get<std::string>();
}

std::string common_docker_resolve_model(const std::string & docker) {
    // Parse ai/smollm2:135M-Q4_0
    size_t      colon_pos = docker.find(':');
    std::string repo, tag;
    if (colon_pos != std::string::npos) {
        repo = docker.substr(0, colon_pos);
        tag  = docker.substr(colon_pos + 1);
    } else {
        repo = docker;
        tag  = "latest";
    }

    // ai/ is the default
    size_t      slash_pos = docker.find('/');
    if (slash_pos == std::string::npos) {
        repo.insert(0, "ai/");
    }

    LOG_INF("%s: Downloading Docker Model: %s:%s\n", __func__, repo.c_str(), tag.c_str());
    try {
        // --- helper: digest validation ---
        auto validate_oci_digest = [](const std::string & digest) -> std::string {
            // Expected: algo:hex ; start with sha256 (64 hex chars)
            // You can extend this map if supporting other algorithms in future.
            static const std::regex re("^sha256:([a-fA-F0-9]{64})$");
            std::smatch m;
            if (!std::regex_match(digest, m, re)) {
                throw std::runtime_error("Invalid OCI digest format received in manifest: " + digest);
            }
            // normalize hex to lowercase
            std::string normalized = digest;
            std::transform(normalized.begin()+7, normalized.end(), normalized.begin()+7, [](unsigned char c){
                return std::tolower(c);
            });
            return normalized;
        };

        std::string token = common_docker_get_token(repo);  // Get authentication token

        // Get manifest
        // TODO: cache the manifest response so that it appears in the model list
        const std::string    url_prefix = "https://registry-1.docker.io/v2/" + repo;
        std::string          manifest_url = url_prefix + "/manifests/" + tag;
        common_remote_params manifest_params;
        manifest_params.headers.push_back({"Authorization", "Bearer " + token});
        manifest_params.headers.push_back({"Accept",
            "application/vnd.docker.distribution.manifest.v2+json,application/vnd.oci.image.manifest.v1+json"
        });
        auto manifest_res = common_remote_get_content(manifest_url, manifest_params);
        if (manifest_res.first != 200) {
            throw std::runtime_error("Failed to get Docker manifest, HTTP code: " + std::to_string(manifest_res.first));
        }

        std::string            manifest_str(manifest_res.second.begin(), manifest_res.second.end());
        nlohmann::ordered_json manifest = nlohmann::ordered_json::parse(manifest_str);
        std::string            gguf_digest;  // Find the GGUF layer
        if (manifest.contains("layers")) {
            for (const auto & layer : manifest["layers"]) {
                if (layer.contains("mediaType")) {
                    std::string media_type = layer["mediaType"].get<std::string>();
                    if (media_type == "application/vnd.docker.ai.gguf.v3" ||
                        media_type.find("gguf") != std::string::npos) {
                        gguf_digest = layer["digest"].get<std::string>();
                        break;
                    }
                }
            }
        }

        if (gguf_digest.empty()) {
            throw std::runtime_error("No GGUF layer found in Docker manifest");
        }

        // Validate & normalize digest
        gguf_digest = validate_oci_digest(gguf_digest);
        LOG_DBG("%s: Using validated digest: %s\n", __func__, gguf_digest.c_str());

        // Prepare local filename
        std::string model_filename = repo;
        std::replace(model_filename.begin(), model_filename.end(), '/', '_');
        model_filename += "_" + tag + ".gguf";
        std::string local_path = fs_get_cache_file(model_filename);

        const std::string blob_url = url_prefix + "/blobs/" + gguf_digest;
        const int http_status = common_download_file_single(blob_url, local_path, token, false, {});
        if (!is_http_status_ok(http_status)) {
            throw std::runtime_error("Failed to download Docker Model");
        }

        LOG_INF("%s: Downloaded Docker Model to: %s\n", __func__, local_path.c_str());
        return local_path;
    } catch (const std::exception & e) {
        LOG_ERR("%s: Docker Model download failed: %s\n", __func__, e.what());
        throw;
    }
}

std::vector<std::string> common_list_cached_models() {
    auto files = hf_cache::get_cached_files("");
    std::set<std::string> models;

    for (const auto & f : files) {
        std::string tmp = f.path;

        if (!string_remove_suffix(tmp, ".gguf")) {
            continue;
        }
        if (tmp.find("mmproj") != std::string::npos) {
            continue;
        }
        auto split_pos = tmp.find("-00001-of-");

        if (split_pos == std::string::npos &&
            tmp.find("-of-") != std::string::npos) {
            continue;
        }
        if (split_pos != std::string::npos) {
            tmp.erase(split_pos);
        }
        auto sep_pos = tmp.find_last_of("-.");

        if (sep_pos == std::string::npos || sep_pos == tmp.size() - 1) {
            continue;
        }
        tmp.erase(0, sep_pos + 1);

        bool is_valid = true;
        for (char & c : tmp) {
            unsigned char uc = c;
            if (!std::isalnum(uc) && uc != '_') {
                is_valid = false;
                break;
            }
            c = std::toupper(uc);
        }
        if (is_valid) {
            models.insert(f.repo_id + ":" + tmp);
        }
    }

    return {models.begin(), models.end()};
}
