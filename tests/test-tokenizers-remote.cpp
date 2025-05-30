#include "arg.h"
#include "common.h"

#include <string>
#include <fstream>
#include <vector>
#include <json.hpp>

using json = nlohmann::json;

#undef NDEBUG
#include <cassert>

std::string endpoint = "https://huggingface.co/";
std::string repo = "ggml-org/vocabs";

static void write_file(const std::string & fname, const std::string & content) {
    std::ofstream file(fname);
    if (file) {
        file << content;
        file.close();
    }
}

static json get_hf_repo_dir(const std::string & hf_repo_with_branch, bool recursive, const std::string & repo_path, const std::string & bearer_token) {
    auto parts = string_split<std::string>(hf_repo_with_branch, ':');
    std::string branch = parts.size() > 1 ? parts.back() : "main";
    std::string hf_repo = parts[0];
    std::string url = endpoint + "api/models/" + hf_repo + "/tree/" + branch;
    std::string path = repo_path;

    if (!path.empty()) {
        // FIXME: path should be properly url-encoded!
        string_replace_all(path, "/", "%2F");
        url += "/" + path;
    }

    if (recursive) {
        url += "?recursive=true";
    }

    // headers
    std::vector<std::string> headers;
    headers.push_back("Accept: application/json");
    if (!bearer_token.empty()) {
        headers.push_back("Authorization: Bearer " + bearer_token);
    }

    // we use "=" to avoid clashing with other component, while still being allowed on windows
    std::string cached_response_fname = "test_vocab=" + hf_repo + "/" + repo_path + "=" + branch + ".json";
    string_replace_all(cached_response_fname, "/", "_");
    std::string cached_response_path = fs_get_cache_file(cached_response_fname);

    // make the request
    common_remote_params params;
    params.headers = headers;
    json res_data;
    try {
        // TODO: For pagination links we need response headers, which is not provided by common_remote_get_content()
        auto res = common_remote_get_content(url, params);
        long res_code = res.first;
        std::string res_str = std::string(res.second.data(), res.second.size());

        if (res_code == 200) {
            write_file(cached_response_path, res_str);
        } else if (res_code == 401) {
            throw std::runtime_error("error: model is private or does not exist; if you are accessing a gated model, please provide a valid HF token");
        } else {
            throw std::runtime_error(string_format("error from HF API, response code: %ld, data: %s", res_code, res_str.c_str()));
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "error: failed to get repo tree: %s\n", e.what());
        fprintf(stderr, "try reading from cache\n");
    }

    // try to read from cache
    try {
        std::ifstream f(cached_response_path);
        res_data = json::parse(f);
    } catch (const std::exception & e) {
        fprintf(stderr, "error: failed to get repo tree (check your internet connection)\n");
    }

    return res_data;
}

int main(void) {
    if (common_has_curl()) {
        json tree = get_hf_repo_dir(repo, true, {}, {});

        if (!tree.empty()) {
            std::vector<std::pair<std::string, std::string>> files;

            for (const auto & item : tree) {
                if (item.at("type") == "file") {
                    std::string path = item.at("path");

                    if (string_ends_with(path, ".gguf") || string_ends_with(path, ".gguf.inp") || string_ends_with(path, ".gguf.out")) {
                        // this is to avoid different repo having same file name, or same file name in different subdirs
                        std::string filepath = repo + "_" + path;
                        // to make sure we don't have any slashes in the filename
                        string_replace_all(filepath, "/", "_");
                        // to make sure we don't have any quotes in the filename
                        string_replace_all(filepath, "'", "_");
                        filepath = fs_get_cache_file(filepath);

                        files.push_back({endpoint + repo + "/resolve/main/" + path, filepath});
                    }
                }
            }

            if (common_download_file_multiple(files, {}, false)) {
                std::string dir_sep(1, DIRECTORY_SEPARATOR);

                for (auto const & item : files) {
                    std::string filepath = item.second;

                    if (string_ends_with(filepath, ".gguf")) {
                        std::string vocab_inp = filepath + ".inp";
                        std::string vocab_out = filepath + ".out";
                        auto matching_inp = std::find_if(files.begin(), files.end(), [&vocab_inp](const auto & p) {
                            return p.second == vocab_inp;
                        });
                        auto matching_out = std::find_if(files.begin(), files.end(), [&vocab_out](const auto & p) {
                            return p.second == vocab_out;
                        });

                        if (matching_inp != files.end() && matching_out != files.end()) {
                            std::string test_command = "." + dir_sep + "test-tokenizer-0 '" + filepath + "'";
                            assert(std::system(test_command.c_str()) == 0);
                        } else {
                            printf("test-tokenizers-remote: %s found without .inp/out vocab files, skipping...\n", filepath.c_str());
                        }
                    }
                }
            } else {
                printf("test-tokenizers-remote: failed to download files, unable to perform tests...\n");
            }
        } else {
            printf("test-tokenizers-remote: failed to retrieve repository info, unable to perform tests...\n");
        }
    } else {
        printf("test-tokenizers-remote: no curl, unable to perform tests...\n");
    }
}
