#include "arg.h"

#include "common.h"
#include "gguf.h" // for reading GGUF splits
#include "log.h"
#include "download.h"

#define JSON_ASSERT GGML_ASSERT
#include <nlohmann/json.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <future>
#include <map>
#include <mutex>
#include <regex>
#include <string>
#include <thread>
#include <vector>

#if defined(LLAMA_USE_HTTPLIB)
#include "http.h"
#endif

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

#define LLAMA_MAX_URL_LENGTH 2084 // Maximum URL Length in Chrome: 2083

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
static bool validate_repo_name(const std::string & repo) {
    static const std::regex repo_regex(R"(^[A-Za-z0-9_.\-]+\/[A-Za-z0-9_.\-]+$)");
    return std::regex_match(repo, repo_regex);
}

static std::string get_manifest_path(const std::string & repo, const std::string & tag) {
    // we use "=" to avoid clashing with other component, while still being allowed on windows
    std::string fname = "manifest=" + repo + "=" + tag + ".json";
    if (!validate_repo_name(repo)) {
        throw std::runtime_error("error: repo name must be in the format 'owner/repo'");
    }
    string_replace_all(fname, "/", "=");
    return fs_get_cache_file(fname);
}

static std::string read_file(const std::string & fname) {
    std::ifstream file(fname);
    if (!file) {
        throw std::runtime_error(string_format("error: failed to open file '%s'\n", fname.c_str()));
    }
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    file.close();
    return content;
}

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
    std::string none;
    const std::string etag_path = path + ".etag";

    if (std::filesystem::exists(etag_path)) {
        std::ifstream etag_in(etag_path);
        if (!etag_in) {
            LOG_ERR("%s: could not open .etag file for reading: %s\n", __func__, etag_path.c_str());
            return none;
        }
        std::string etag;
        std::getline(etag_in, etag);
        return etag;
    }

    // no etag file, but maybe there is an old .json
    // remove this code later
    const std::string metadata_path = path + ".json";

    if (std::filesystem::exists(metadata_path)) {
        std::ifstream metadata_in(metadata_path);
        try {
            nlohmann::json metadata_json;
            metadata_in >> metadata_json;
            LOG_DBG("%s: previous metadata file found %s: %s\n", __func__, metadata_path.c_str(),
                    metadata_json.dump().c_str());
            if (metadata_json.contains("etag") && metadata_json.at("etag").is_string()) {
                std::string etag = metadata_json.at("etag");
                write_etag(path, etag);
                if (!std::filesystem::remove(metadata_path)) {
                    LOG_WRN("%s: failed to delete old .json metadata file: %s\n", __func__, metadata_path.c_str());
                }
                return etag;
            }
        } catch (const nlohmann::json::exception & e) {
            LOG_ERR("%s: error reading metadata file %s: %s\n", __func__, metadata_path.c_str(), e.what());
        }
    }
    return none;
}

static bool is_http_status_ok(int status) {
    return status >= 200 && status < 400;
}

std::pair<std::string, std::string> common_download_split_repo_tag(const std::string & hf_repo_with_tag) {
    auto parts = string_split<std::string>(hf_repo_with_tag, ':');
    std::string tag = parts.size() > 1 ? parts.back() : "latest";
    std::string hf_repo = parts[0];
    if (string_split<std::string>(hf_repo, '/').size() != 2) {
        throw std::invalid_argument("error: invalid HF repo format, expected <user>/<model>[:quant]\n");
    }
    return {hf_repo, tag};
}

#if defined(LLAMA_USE_HTTPLIB)

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
        LOG_ERR("%s: error during download. Status: %d\n", __func__, res ? res->status : -1);
        return false;
    }

    return true;
}

// download one single file from remote URL to local path
// returns status code or -1 on error
static int common_download_file_single_online(const std::string & url,
                                               const std::string & path,
                                               const std::string & bearer_token,
                                               const common_header_list & custom_headers) {
    static const int max_attempts        = 3;
    static const int retry_delay_seconds = 2;

    auto [cli, parts] = common_http_client(url);

    httplib::Headers default_headers = {{"User-Agent", "llama-cpp"}};
    if (!bearer_token.empty()) {
        default_headers.insert({"Authorization", "Bearer " + bearer_token});
    }
    for (const auto & h : custom_headers) {
        default_headers.emplace(h.first, h.second);
    }
    cli.set_default_headers(default_headers);

    const bool file_exists = std::filesystem::exists(path);

    std::string last_etag;
    if (file_exists) {
        last_etag = read_etag(path);
    } else {
        LOG_INF("%s: no previous model file found %s\n", __func__, path.c_str());
    }

    for (int i = 0; i < max_attempts; ++i) {
        auto head = cli.Head(parts.path);
        bool head_ok = head && head->status >= 200 && head->status < 300;
        if (!head_ok) {
            LOG_WRN("%s: HEAD invalid http status code received: %d\n", __func__, head ? head->status : -1);
            if (file_exists) {
                LOG_INF("%s: Using cached file (HEAD failed): %s\n", __func__, path.c_str());
                return 304; // 304 Not Modified - fake cached response
            }
            return head->status; // cannot use cached file, return raw status code
            // TODO: maybe retry only on certain codes
        }

        std::string etag;
        if (head_ok && head->has_header("ETag")) {
            etag = head->get_header_value("ETag");
        }

        size_t total_size = 0;
        if (head_ok && head->has_header("Content-Length")) {
            try {
                total_size = std::stoull(head->get_header_value("Content-Length"));
            } catch (const std::exception& e) {
                LOG_WRN("%s: Invalid Content-Length in HEAD response: %s\n", __func__, e.what());
            }
        }

        bool supports_ranges = false;
        if (head_ok && head->has_header("Accept-Ranges")) {
            supports_ranges = head->get_header_value("Accept-Ranges") != "none";
        }

        bool should_download_from_scratch = false;
        if (!last_etag.empty() && !etag.empty() && last_etag != etag) {
            LOG_WRN("%s: ETag header is different (%s != %s): triggering a new download\n", __func__,
                    last_etag.c_str(), etag.c_str());
            should_download_from_scratch = true;
        }

        if (file_exists) {
            if (!should_download_from_scratch) {
                LOG_INF("%s: using cached file: %s\n", __func__, path.c_str());
                return 304; // 304 Not Modified - fake cached response
            }
            LOG_WRN("%s: deleting previous downloaded file: %s\n", __func__, path.c_str());
            if (remove(path.c_str()) != 0) {
                LOG_ERR("%s: unable to delete file: %s\n", __func__, path.c_str());
                return -1;
            }
        }

        const std::string path_temporary = path + ".downloadInProgress";
        size_t existing_size = 0;

        if (std::filesystem::exists(path_temporary)) {
            if (supports_ranges && !should_download_from_scratch) {
                existing_size = std::filesystem::file_size(path_temporary);
            } else if (remove(path_temporary.c_str()) != 0) {
                LOG_ERR("%s: unable to delete file: %s\n", __func__, path_temporary.c_str());
                return -1;
            }
        }

        // start the download
        LOG_INF("%s: trying to download model from %s to %s (etag:%s)...\n",
                __func__, common_http_show_masked_url(parts).c_str(), path_temporary.c_str(), etag.c_str());
        const bool was_pull_successful = common_pull_file(cli, parts.path, path_temporary, supports_ranges, existing_size, total_size);
        if (!was_pull_successful) {
            if (i + 1 < max_attempts) {
                const int exponential_backoff_delay = std::pow(retry_delay_seconds, i) * 1000;
                LOG_WRN("%s: retrying after %d milliseconds...\n", __func__, exponential_backoff_delay);
                std::this_thread::sleep_for(std::chrono::milliseconds(exponential_backoff_delay));
            } else {
                LOG_ERR("%s: download failed after %d attempts\n", __func__, max_attempts);
            }
            continue;
        }

        if (std::rename(path_temporary.c_str(), path.c_str()) != 0) {
            LOG_ERR("%s: unable to rename file: %s to %s\n", __func__, path_temporary.c_str(), path.c_str());
            return -1;
        }
        if (!etag.empty()) {
            write_etag(path, etag);
        }

        return head->status; // TODO: use actual GET status?
    }

    return -1; // max attempts reached
}

std::pair<long, std::vector<char>> common_remote_get_content(const std::string          & url,
                                                             const common_remote_params & params) {
    auto [cli, parts] = common_http_client(url);

    httplib::Headers headers = {{"User-Agent", "llama-cpp"}};

    for (const auto & header : params.headers) {
        headers.emplace(header.first, header.second);
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
                                const common_header_list & headers) {
    if (!offline) {
        return common_download_file_single_online(url, path, bearer_token, headers);
    }

    if (!std::filesystem::exists(path)) {
        LOG_ERR("%s: required file is not available in cache (offline mode): %s\n", __func__, path.c_str());
        return -1;
    }

    LOG_INF("%s: using cached file (offline mode): %s\n", __func__, path.c_str());
    return 304; // Not Modified - fake cached response
}

// download multiple files from remote URLs to local paths
// the input is a vector of pairs <url, path>
static bool common_download_file_multiple(const std::vector<std::pair<std::string, std::string>> & urls,
                                          const std::string & bearer_token,
                                          bool offline,
                                          const common_header_list & headers) {
    // Prepare download in parallel
    std::vector<std::future<bool>> futures_download;
    futures_download.reserve(urls.size());

    for (auto const & item : urls) {
        futures_download.push_back(
            std::async(
                std::launch::async,
                [&bearer_token, offline, &headers](const std::pair<std::string, std::string> & it) -> bool {
                    const int http_status = common_download_file_single(it.first, it.second, bearer_token, offline, headers);
                    return is_http_status_ok(http_status);
                },
                item
            )
        );
    }

    // Wait for all downloads to complete
    for (auto & f : futures_download) {
        if (!f.get()) {
            return false;
        }
    }

    return true;
}

bool common_download_model(const common_params_model & model,
                           const std::string & bearer_token,
                           bool offline,
                           const common_header_list & headers) {
    // Basic validation of the model.url
    if (model.url.empty()) {
        LOG_ERR("%s: invalid model url\n", __func__);
        return false;
    }

    const int http_status = common_download_file_single(model.url, model.path, bearer_token, offline, headers);
    if (!is_http_status_ok(http_status)) {
        return false;
    }

    // check for additional GGUFs split to download
    int n_split = 0;
    {
        struct gguf_init_params gguf_params = {
            /*.no_alloc = */ true,
            /*.ctx      = */ NULL,
        };
        auto * ctx_gguf = gguf_init_from_file(model.path.c_str(), gguf_params);
        if (!ctx_gguf) {
            LOG_ERR("\n%s:  failed to load input GGUF from %s\n", __func__, model.path.c_str());
            return false;
        }

        auto key_n_split = gguf_find_key(ctx_gguf, LLM_KV_SPLIT_COUNT);
        if (key_n_split >= 0) {
            n_split = gguf_get_val_u16(ctx_gguf, key_n_split);
        }

        gguf_free(ctx_gguf);
    }

    if (n_split > 1) {
        char split_prefix[PATH_MAX] = {0};
        char split_url_prefix[LLAMA_MAX_URL_LENGTH] = {0};

        // Verify the first split file format
        // and extract split URL and PATH prefixes
        {
            if (!llama_split_prefix(split_prefix, sizeof(split_prefix), model.path.c_str(), 0, n_split)) {
                LOG_ERR("\n%s: unexpected model file name: %s n_split=%d\n", __func__, model.path.c_str(), n_split);
                return false;
            }

            if (!llama_split_prefix(split_url_prefix, sizeof(split_url_prefix), model.url.c_str(), 0, n_split)) {
                LOG_ERR("\n%s: unexpected model url: %s n_split=%d\n", __func__, model.url.c_str(), n_split);
                return false;
            }
        }

        std::vector<std::pair<std::string, std::string>> urls;
        for (int idx = 1; idx < n_split; idx++) {
            char split_path[PATH_MAX] = {0};
            llama_split_path(split_path, sizeof(split_path), split_prefix, idx, n_split);

            char split_url[LLAMA_MAX_URL_LENGTH] = {0};
            llama_split_path(split_url, sizeof(split_url), split_url_prefix, idx, n_split);

            if (std::string(split_path) == model.path) {
                continue; // skip the already downloaded file
            }

            urls.push_back({split_url, split_path});
        }

        // Download in parallel
        common_download_file_multiple(urls, bearer_token, offline, headers);
    }

    return true;
}

common_hf_file_res common_get_hf_file(const std::string & hf_repo_with_tag,
                                      const std::string & bearer_token,
                                      bool offline,
                                      const common_header_list & custom_headers) {
    // the returned hf_repo is without tag
    auto [hf_repo, tag] = common_download_split_repo_tag(hf_repo_with_tag);

    std::string url = get_model_endpoint() + "v2/" + hf_repo + "/manifests/" + tag;

    // headers
    common_header_list headers = custom_headers;
    headers.push_back({"Accept", "application/json"});
    if (!bearer_token.empty()) {
        headers.push_back({"Authorization", "Bearer " + bearer_token});
    }
    // Important: the User-Agent must be "llama-cpp" to get the "ggufFile" field in the response
    // User-Agent header is already set in common_remote_get_content, no need to set it here

    // make the request
    common_remote_params params;
    params.headers = headers;
    long res_code = 0;
    std::string res_str;
    bool use_cache = false;
    std::string cached_response_path = get_manifest_path(hf_repo, tag);
    if (!offline) {
        try {
            auto res = common_remote_get_content(url, params);
            res_code = res.first;
            res_str = std::string(res.second.data(), res.second.size());
        } catch (const std::exception & e) {
            LOG_WRN("error: failed to get manifest at %s: %s\n", url.c_str(), e.what());
        }
    }
    if (res_code == 0) {
        if (std::filesystem::exists(cached_response_path)) {
            LOG_WRN("trying to read manifest from cache: %s\n", cached_response_path.c_str());
            res_str = read_file(cached_response_path);
            res_code = 200;
            use_cache = true;
        } else {
            throw std::runtime_error(
                offline ? "error: failed to get manifest (offline mode)"
                : "error: failed to get manifest (check your internet connection)");
        }
    }
    std::string ggufFile;
    std::string mmprojFile;

    if (res_code == 200 || res_code == 304) {
        try {
            auto j = json::parse(res_str);

            if (j.contains("ggufFile") && j["ggufFile"].contains("rfilename")) {
                ggufFile = j["ggufFile"]["rfilename"].get<std::string>();
            }
            if (j.contains("mmprojFile") && j["mmprojFile"].contains("rfilename")) {
                mmprojFile = j["mmprojFile"]["rfilename"].get<std::string>();
            }
        } catch (const std::exception & e) {
            throw std::runtime_error(std::string("error parsing manifest JSON: ") + e.what());
        }
        if (!use_cache) {
            // if not using cached response, update the cache file
            write_file(cached_response_path, res_str);
        }
    } else if (res_code == 401) {
        throw std::runtime_error("error: model is private or does not exist; if you are accessing a gated model, please provide a valid HF token");
    } else {
        throw std::runtime_error(string_format("error from HF API (%s), response code: %ld, data: %s", url.c_str(), res_code, res_str.c_str()));
    }

    // check response
    if (ggufFile.empty()) {
        throw std::runtime_error("error: model does not have ggufFile");
    }

    return { hf_repo, ggufFile, mmprojFile };
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

#else

common_hf_file_res common_get_hf_file(const std::string &, const std::string &, bool, const common_header_list &) {
    throw std::runtime_error("download functionality is not enabled in this build");
}

bool common_download_model(const common_params_model &, const std::string &, bool, const common_header_list &) {
    throw std::runtime_error("download functionality is not enabled in this build");
}

std::string common_docker_resolve_model(const std::string &) {
    throw std::runtime_error("download functionality is not enabled in this build");
}

int common_download_file_single(const std::string &,
                                const std::string &,
                                const std::string &,
                                bool,
                                const common_header_list &) {
    throw std::runtime_error("download functionality is not enabled in this build");
}

#endif // defined(LLAMA_USE_HTTPLIB)

std::vector<common_cached_model_info> common_list_cached_models() {
    std::vector<common_cached_model_info> models;
    const std::string cache_dir = fs_get_cache_directory();
    const std::vector<common_file_info> files = fs_list(cache_dir, false);
    for (const auto & file : files) {
        if (string_starts_with(file.name, "manifest=") && string_ends_with(file.name, ".json")) {
            common_cached_model_info model_info;
            model_info.manifest_path = file.path;
            std::string fname = file.name;
            string_replace_all(fname, ".json", ""); // remove extension
            auto parts = string_split<std::string>(fname, '=');
            if (parts.size() == 4) {
                // expect format: manifest=<user>=<model>=<tag>=<other>
                model_info.user  = parts[1];
                model_info.model = parts[2];
                model_info.tag   = parts[3];
            } else {
                // invalid format
                continue;
            }
            model_info.size = 0; // TODO: get GGUF size, not manifest size
            models.push_back(model_info);
        }
    }
    return models;
}
