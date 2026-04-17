#include "arg.h"

#include "build-info.h"
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
#include <unordered_set>
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

class ProgressBar : public common_download_callback {
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
    ProgressBar() = default;

    void on_start(const common_download_progress & p) override {
        filename = p.url;

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

    void on_done(const common_download_progress &, bool) override {
        std::lock_guard<std::mutex> lock(mutex);
        cleanup(this);
    }

    void on_update(const common_download_progress & p) override {
        if (!p.total || !is_output_a_tty()) {
            return;
        }

        std::lock_guard<std::mutex> lock(mutex);

        if (lines.find(this) == lines.end()) {
            lines[this] = max_line++;
            std::cout << "\n";
        }
        int lines_up = max_line - lines[this];

        size_t bar = (55 - len) * 2;
        size_t pct = (100 * p.downloaded) / p.total;
        size_t pos = (bar * p.downloaded) / p.total;

        if (lines_up > 0) {
            std::cout << "\033[" << lines_up << "A";
        }
        std::cout << '\r' << "Downloading " << filename << " ";

        for (size_t i = 0; i < bar; i += 2) {
            std::cout << (i + 1 < pos ? "─" : (i < pos ? "╴" : " "));
        }
        std::cout << std::setw(4) << pct << "%\033[K";

        if (lines_up > 0) {
            std::cout << "\033[" << lines_up << "B";
        }
        std::cout << '\r' << std::flush;

        if (p.downloaded == p.total) {
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
                             common_download_progress & p,
                             common_download_callback * callback) {
    std::ofstream ofs(path_tmp, std::ios::binary | std::ios::app);
    if (!ofs.is_open()) {
        LOG_ERR("%s: error opening local file for writing: %s\n", __func__, path_tmp.c_str());
        return false;
    }

    httplib::Headers headers;
    if (supports_ranges && p.downloaded > 0) {
        headers.emplace("Range", "bytes=" + std::to_string(p.downloaded) + "-");
    }

    const char * func = __func__; // avoid __func__ inside a lambda
    size_t progress_step = 0;

    auto res = cli.Get(resolve_path, headers,
        [&](const httplib::Response &response) {
            if (p.downloaded > 0 && response.status != 206) {
                LOG_WRN("%s: server did not respond with 206 Partial Content for a resume request. Status: %d\n", func, response.status);
                return false;
            }
            if (p.downloaded == 0 && response.status != 200) {
                LOG_WRN("%s: download received non-successful status code: %d\n", func, response.status);
                return false;
            }
            if (p.total == 0 && response.has_header("Content-Length")) {
                try {
                    size_t content_length = std::stoull(response.get_header_value("Content-Length"));
                    p.total = p.downloaded + content_length;
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
            p.downloaded += len;
            progress_step += len;

            if (progress_step >= p.total / 1000 || p.downloaded == p.total) {
                if (callback) {
                    callback->on_update(p);
                    if (callback->is_cancelled()) {
                        return false;
                    }
                }
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
static int common_download_file_single_online(const std::string & url,
                                              const std::string & path,
                                              const common_download_opts & opts,
                                              bool skip_etag) {
    static const int max_attempts        = 3;
    static const int retry_delay_seconds = 2;

    const bool file_exists = std::filesystem::exists(path);

    if (file_exists && skip_etag) {
        LOG_DBG("%s: using cached file: %s\n", __func__, path.c_str());
        return 304; // 304 Not Modified - fake cached response
    }

    auto [cli, parts] = common_http_client(url);

    httplib::Headers headers;
    for (const auto & h : opts.headers) {
        headers.emplace(h.first, h.second);
    }
    if (headers.find("User-Agent") == headers.end()) {
        headers.emplace("User-Agent", "llama-cpp/" + std::string(llama_build_info()));
    }
    if (!opts.bearer_token.empty()) {
        headers.emplace("Authorization", "Bearer " + opts.bearer_token);
    }
    cli.set_default_headers(headers);

    std::string last_etag;
    if (file_exists) {
        last_etag = read_etag(path);
    } else {
        LOG_DBG("%s: no previous model file found %s\n", __func__, path.c_str());
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

    common_download_progress p;
    p.url = url;
    if (head->has_header("Content-Length")) {
        try {
            p.total = std::stoull(head->get_header_value("Content-Length"));
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
            LOG_DBG("%s: using cached file (no server etag): %s\n", __func__, path.c_str());
            return 304; // 304 Not Modified - fake cached response
        }
        if (!last_etag.empty() && last_etag == etag) {
            LOG_DBG("%s: using cached file (same etag): %s\n", __func__, path.c_str());
            return 304; // 304 Not Modified - fake cached response
        }
        if (remove(path.c_str()) != 0) {
            LOG_ERR("%s: unable to delete file: %s\n", __func__, path.c_str());
            return -1;
        }
    }

    { // silent
        std::error_code ec;
        std::filesystem::create_directories(std::filesystem::path(path).parent_path(), ec);
    }

    bool success = false;
    const std::string path_temporary = path + ".downloadInProgress";
    int delay = retry_delay_seconds;

    if (opts.callback) {
        opts.callback->on_start(p);
    }

    for (int i = 0; i < max_attempts; ++i) {
        if (opts.callback && opts.callback->is_cancelled()) {
            break;
        }
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
                break;
            }
        }

        p.downloaded = existing_size;

        LOG_DBG("%s: downloading from %s to %s (etag:%s)...\n",
                __func__, common_http_show_masked_url(parts).c_str(),
                path_temporary.c_str(), etag.c_str());

        if (common_pull_file(cli, parts.path, path_temporary, supports_ranges, p, opts.callback)) {
            if (std::rename(path_temporary.c_str(), path.c_str()) != 0) {
                LOG_ERR("%s: unable to rename file: %s to %s\n", __func__, path_temporary.c_str(), path.c_str());
                break;
            }
            if (!etag.empty() && !skip_etag) {
                write_etag(path, etag);
            }
            success = true;
            break;
        }
    }

    if (opts.callback) {
        opts.callback->on_done(p, success);
    }
    if (opts.callback && opts.callback->is_cancelled() &&
        std::filesystem::exists(path_temporary)) {
        if (remove(path_temporary.c_str()) != 0) {
            LOG_ERR("%s: unable to delete temporary file: %s\n", __func__, path_temporary.c_str());
        }
    }
    if (!success) {
        LOG_ERR("%s: download failed after %d attempts\n", __func__, max_attempts);
        return -1; // max attempts reached
    }

    return head->status;
}

std::pair<long, std::vector<char>> common_remote_get_content(const std::string          & url,
                                                             const common_remote_params & params) {
    auto [cli, parts] = common_http_client(url);

    httplib::Headers headers;
    for (const auto & h : params.headers) {
        headers.emplace(h.first, h.second);
    }
    if (headers.find("User-Agent") == headers.end()) {
        headers.emplace("User-Agent", "llama-cpp/" + std::string(llama_build_info()));
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
                                const common_download_opts & opts,
                                bool skip_etag) {
    if (!opts.offline) {
        ProgressBar tty_cb;
        common_download_opts online_opts = opts;
        if (!online_opts.callback) {
            online_opts.callback = &tty_cb;
        }
        return common_download_file_single_online(url, path, online_opts, skip_etag);
    }

    if (!std::filesystem::exists(path)) {
        LOG_ERR("%s: required file is not available in cache (offline mode): %s\n", __func__, path.c_str());
        return -1;
    }

    LOG_DBG("%s: using cached file (offline mode): %s\n", __func__, path.c_str());

    // notify the callback that the file was cached
    if (opts.callback) {
        common_download_progress p;
        p.url = url;
        p.cached = true;
        opts.callback->on_start(p);
        opts.callback->on_done(p, true);
    }

    return 304; // Not Modified - fake cached response
}

struct gguf_split_info {
    std::string prefix; // tag included
    std::string tag;
    int index;
    int count;
};

static gguf_split_info get_gguf_split_info(const std::string & path) {
    static const std::regex re_split("^(.+)-([0-9]{5})-of-([0-9]{5})$", std::regex::icase);
    static const std::regex re_tag("[-.]([A-Z0-9_]+)$", std::regex::icase);
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

    std::string tag;
    if (std::regex_search(prefix, m, re_tag)) {
        tag = m[1].str();
        for (char & c : tag) {
            c = std::toupper((unsigned char)c);
        }
    }

    return {std::move(prefix), std::move(tag), index, count};
}

// Q4_0 -> 4, F16 -> 16, NVFP4 -> 4, Q8_K_M -> 8, etc
static int extract_quant_bits(const std::string & filename) {
    auto split = get_gguf_split_info(filename);

    auto pos = split.tag.find_first_of("0123456789");
    if (pos == std::string::npos) {
        return 0;
    }

    return std::stoi(split.tag.substr(pos));
}

static hf_cache::hf_files get_split_files(const hf_cache::hf_files & files,
                                          const hf_cache::hf_file  & file) {
    auto split = get_gguf_split_info(file.path);

    if (split.count <= 1) {
        return {file};
    }
    hf_cache::hf_files result;

    for (const auto & f : files) {
        auto split_f = get_gguf_split_info(f.path);
        if (split_f.count == split.count && split_f.prefix == split.prefix) {
            result.push_back(f);
        }
    }
    return result;
}

static hf_cache::hf_file find_best_mmproj(const hf_cache::hf_files & files,
                                          const std::string        & model) {
    hf_cache::hf_file best;
    size_t best_depth = 0;
    int best_diff = 0;
    bool found = false;

    auto model_bits = extract_quant_bits(model);
    auto model_parts = string_split<std::string>(model, '/');
    auto model_dir = model_parts.end() - 1;

    for (const auto & f : files) {
        if (!string_ends_with(f.path, ".gguf") ||
            f.path.find("mmproj") == std::string::npos) {
            continue;
        }

        auto mmproj_parts = string_split<std::string>(f.path, '/');
        auto mmproj_dir = mmproj_parts.end() - 1;

        auto [_, dir] = std::mismatch(model_parts.begin(), model_dir,
                                      mmproj_parts.begin(), mmproj_dir);
        if (dir != mmproj_dir) {
            continue;
        }

        size_t depth = dir - mmproj_parts.begin();
        auto bits = extract_quant_bits(f.path);
        auto diff = std::abs(bits - model_bits);

        if (!found || depth > best_depth || (depth == best_depth && diff < best_diff)) {
            best = f;
            best_depth = depth;
            best_diff = diff;
            found = true;
        }
    }
    return best;
}

static bool gguf_filename_is_model(const std::string & filepath) {
    if (!string_ends_with(filepath, ".gguf")) {
        return false;
    }

    std::string filename = filepath;
    if (auto pos = filename.rfind('/'); pos != std::string::npos) {
        filename = filename.substr(pos + 1);
    }

    return filename.find("mmproj")  == std::string::npos &&
           filename.find("imatrix") == std::string::npos;
}

static hf_cache::hf_file find_best_model(const hf_cache::hf_files & files,
                                         const std::string        & tag) {
    std::vector<std::string> tags;

    if (!tag.empty()) {
        tags.push_back(tag);
    } else {
        tags = {"Q4_K_M", "Q4_0"};
    }

    for (const auto & t : tags) {
        std::regex pattern(t + "[.-]", std::regex::icase);
        for (const auto & f : files) {
            if (gguf_filename_is_model(f.path) &&
                std::regex_search(f.path, pattern)) {
                auto split = get_gguf_split_info(f.path);
                if (split.count > 1 && split.index != 1) {
                    continue;
                }
                return f;
            }
        }
    }

    // fallback to first available model only if tag is empty
    if (tag.empty()) {
        for (const auto & f : files) {
            if (gguf_filename_is_model(f.path)) {
                auto split = get_gguf_split_info(f.path);
                if (split.count > 1 && split.index != 1) {
                    continue;
                }
                return f;
            }
        }
    }

    return {};
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
    hf_cache::hf_files model_files;
    hf_cache::hf_file mmproj;
};

static hf_plan get_hf_plan(const common_params_model  & model,
                           const common_download_opts & opts,
                           bool download_mmproj) {
    hf_plan plan;
    hf_cache::hf_files all;

    auto [repo, tag] = common_download_split_repo_tag(model.hf_repo);

    if (!opts.offline) {
        all = hf_cache::get_repo_files(repo, opts.bearer_token);
    }
    if (all.empty()) {
        all = hf_cache::get_cached_files(repo);
    }
    if (all.empty()) {
        return plan;
    }

    hf_cache::hf_file primary;

    if (!model.hf_file.empty()) {
        for (const auto & f : all) {
            if (f.path == model.hf_file) {
                primary = f;
                break;
            }
        }
        if (primary.path.empty()) {
            LOG_ERR("%s: file '%s' not found in repository\n", __func__, model.hf_file.c_str());
            list_available_gguf_files(all);
            return plan;
        }
    } else {
        primary = find_best_model(all, tag);
        if (primary.path.empty()) {
            LOG_ERR("%s: no GGUF files found in repository %s\n", __func__, repo.c_str());
            list_available_gguf_files(all);
            return plan;
        }
    }

    plan.primary = primary;
    plan.model_files = get_split_files(all, primary);

    if (download_mmproj) {
        plan.mmproj = find_best_mmproj(all, primary.path);
    }

    return plan;
}

struct download_task {
    std::string url;
    std::string path;
};

static std::vector<download_task> get_url_tasks(const common_params_model & model) {
    auto split = get_gguf_split_info(model.url);

    if (split.count <= 1) {
        return {{model.url, model.path}};
    }

    auto filename = split.prefix;
    if (auto pos = split.prefix.rfind('/'); pos != std::string::npos) {
        filename = split.prefix.substr(pos + 1);
    }

    auto parent_path = std::filesystem::path(model.path).parent_path();
    auto prefix_path = (parent_path / filename).string();

    std::vector<download_task> tasks;
    for (int i = 1; i <= split.count; i++) {
        auto suffix = string_format("-%05d-of-%05d.gguf", i, split.count);
        tasks.push_back({split.prefix + suffix, prefix_path + suffix});
    }
    return tasks;
}

common_download_model_result common_download_model(const common_params_model  & model,
                                                   const common_download_opts & opts,
                                                   bool download_mmproj) {
    common_download_model_result result;
    std::vector<download_task> tasks;
    hf_plan hf;

    bool is_hf = !model.hf_repo.empty();

    if (is_hf) {
        hf = get_hf_plan(model, opts, download_mmproj);
        for (const auto & f : hf.model_files) {
            tasks.push_back({f.url, f.local_path});
        }
        if (!hf.mmproj.path.empty()) {
            tasks.push_back({hf.mmproj.url, hf.mmproj.local_path});
        }
    } else if (!model.url.empty()) {
        tasks = get_url_tasks(model);
    } else {
        result.model_path = model.path;
        return result;
    }

    if (tasks.empty()) {
        return result;
    }

    std::vector<std::future<bool>> futures;
    for (const auto & task : tasks) {
        futures.push_back(std::async(std::launch::async,
            [&task, &opts, is_hf]() {
                int status = common_download_file_single(task.url, task.path, opts, is_hf);
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
        for (const auto & f : hf.model_files) {
            hf_cache::finalize_file(f);
        }
        result.model_path = hf.primary.final_path;

        if (!hf.mmproj.path.empty()) {
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
        common_download_opts opts;
        opts.bearer_token = token;
        const int http_status = common_download_file_single(blob_url, local_path, opts);
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

std::vector<common_cached_model_info> common_list_cached_models() {
    std::unordered_set<std::string> seen;
    std::vector<common_cached_model_info> result;

    auto files = hf_cache::get_cached_files();

    for (const auto & f : files) {
        auto split = get_gguf_split_info(f.path);
        if (split.index != 1 || split.tag.empty() ||
            split.prefix.find("mmproj") != std::string::npos) {
            continue;
        }
        if (seen.insert(f.repo_id + ":" + split.tag).second) {
            result.push_back({f.repo_id, split.tag});
        }
    }

    return result;
}
