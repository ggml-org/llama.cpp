#if defined(_WIN32)
#    include <windows.h>
#    include <io.h>
#else
#    include <sys/file.h>
#    include <sys/ioctl.h>
#    include <unistd.h>
#endif

#if defined(LLAMA_USE_CURL)
#    include <curl/curl.h>
#endif

#include <signal.h>

#include <climits>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <list>
#include <sstream>
#include <string>
#include <vector>

#include "chat.h"
#include "common.h"
#include "json.hpp"
#include "linenoise.cpp/linenoise.h"
#include "llama-cpp.h"
#include "log.h"

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__)) || defined(_WIN32)
[[noreturn]] static void sigint_handler(int) {
    printf("\n" LOG_COL_DEFAULT);
    exit(0);  // not ideal, but it's the only way to guarantee exit in all cases
}
#endif

GGML_ATTRIBUTE_FORMAT(1, 2)
static std::string fmt(const char * fmt, ...) {
    va_list ap;
    va_list ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    const int size = vsnprintf(NULL, 0, fmt, ap);
    GGML_ASSERT(size >= 0 && size < INT_MAX); // NOLINT
    std::string buf;
    buf.resize(size);
    const int size2 = vsnprintf(const_cast<char *>(buf.data()), buf.size() + 1, fmt, ap2);
    GGML_ASSERT(size2 == size);
    va_end(ap2);
    va_end(ap);

    return buf;
}

GGML_ATTRIBUTE_FORMAT(1, 2)
static int printe(const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    const int ret = vfprintf(stderr, fmt, args);
    va_end(args);

    return ret;
}

static std::string strftime_fmt(const char * fmt, const std::tm & tm) {
    std::ostringstream oss;
    oss << std::put_time(&tm, fmt);

    return oss.str();
}

class Opt {
  public:
    int init(int argc, const char ** argv) {
        ctx_params           = llama_context_default_params();
        model_params         = llama_model_default_params();
        context_size_default = ctx_params.n_batch;
        ngl_default          = model_params.n_gpu_layers;
        common_params_sampling sampling;
        temperature_default = sampling.temp;

        if (argc < 2) {
            printe("Error: No arguments provided.\n");
            print_help();
            return 1;
        }

        // Parse arguments
        if (parse(argc, argv)) {
            printe("Error: Failed to parse arguments.\n");
            print_help();
            return 1;
        }

        // If help is requested, show help and exit
        if (help) {
            print_help();
            return 2;
        }

        ctx_params.n_batch        = context_size >= 0 ? context_size : context_size_default;
        ctx_params.n_ctx          = ctx_params.n_batch;
        model_params.n_gpu_layers = ngl >= 0 ? ngl : ngl_default;
        temperature               = temperature >= 0 ? temperature : temperature_default;

        return 0;  // Success
    }

    llama_context_params ctx_params;
    llama_model_params   model_params;
    std::string model_;
    std::string chat_template_file;
    std::string          user;
    bool                 use_jinja   = false;
    int                  context_size = -1, ngl = -1;
    float                temperature = -1;
    bool                 verbose     = false;

  private:
    int   context_size_default = -1, ngl_default = -1;
    float temperature_default = -1;
    bool  help                = false;

    bool parse_flag(const char ** argv, int i, const char * short_opt, const char * long_opt) {
        return strcmp(argv[i], short_opt) == 0 || strcmp(argv[i], long_opt) == 0;
    }

    int handle_option_with_value(int argc, const char ** argv, int & i, int & option_value) {
        if (i + 1 >= argc) {
            return 1;
        }

        option_value = std::atoi(argv[++i]);

        return 0;
    }

    int handle_option_with_value(int argc, const char ** argv, int & i, float & option_value) {
        if (i + 1 >= argc) {
            return 1;
        }

        option_value = std::atof(argv[++i]);

        return 0;
    }

    int handle_option_with_value(int argc, const char ** argv, int & i, std::string & option_value) {
        if (i + 1 >= argc) {
            return 1;
        }

        option_value = argv[++i];

        return 0;
    }

    int parse(int argc, const char ** argv) {
        bool options_parsing   = true;
        for (int i = 1, positional_args_i = 0; i < argc; ++i) {
            if (options_parsing && (strcmp(argv[i], "-c") == 0 || strcmp(argv[i], "--context-size") == 0)) {
                if (handle_option_with_value(argc, argv, i, context_size) == 1) {
                    return 1;
                }
            } else if (options_parsing &&
                       (strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "-ngl") == 0 || strcmp(argv[i], "--ngl") == 0)) {
                if (handle_option_with_value(argc, argv, i, ngl) == 1) {
                    return 1;
                }
            } else if (options_parsing && strcmp(argv[i], "--temp") == 0) {
                if (handle_option_with_value(argc, argv, i, temperature) == 1) {
                    return 1;
                }
            } else if (options_parsing &&
                       (parse_flag(argv, i, "-v", "--verbose") || parse_flag(argv, i, "-v", "--log-verbose"))) {
                verbose = true;
            } else if (options_parsing && strcmp(argv[i], "--jinja") == 0) {
                use_jinja = true;
            } else if (options_parsing && strcmp(argv[i], "--chat-template-file") == 0){
                if (handle_option_with_value(argc, argv, i, chat_template_file) == 1) {
                    return 1;
                }
                use_jinja = true;
            } else if (options_parsing && parse_flag(argv, i, "-h", "--help")) {
                help = true;
                return 0;
            } else if (options_parsing && strcmp(argv[i], "--") == 0) {
                options_parsing = false;
            } else if (positional_args_i == 0) {
                if (!argv[i][0] || argv[i][0] == '-') {
                    return 1;
                }

                ++positional_args_i;
                model_ = argv[i];
            } else if (positional_args_i == 1) {
                ++positional_args_i;
                user = argv[i];
            } else {
                user += " " + std::string(argv[i]);
            }
        }

        if (model_.empty()){
            return 1;
        }

        return 0;
    }

    void print_help() const {
        printf(
            "Description:\n"
            "  Runs a llm\n"
            "\n"
            "Usage:\n"
            "  llama-run [options] model [prompt]\n"
            "\n"
            "Options:\n"
            "  -c, --context-size <value>\n"
            "      Context size (default: %d)\n"
            "  --chat-template-file <path>\n"
            "      Path to the file containing the chat template to use with the model.\n"
            "      Only supports jinja templates and implicitly sets the --jinja flag.\n"
            "  --jinja\n"
            "      Use jinja templating for the chat template of the model\n"
            "  -n, -ngl, --ngl <value>\n"
            "      Number of GPU layers (default: %d)\n"
            "  --temp <value>\n"
            "      Temperature (default: %.1f)\n"
            "  -v, --verbose, --log-verbose\n"
            "      Set verbosity level to infinity (i.e. log all messages, useful for debugging)\n"
            "  -h, --help\n"
            "      Show help message\n"
            "\n"
            "Commands:\n"
            "  model\n"
            "      Model is a string with an optional prefix of \n"
            "      huggingface:// (hf://), ollama://, https:// or file://.\n"
            "      If no protocol is specified and a file exists in the specified\n"
            "      path, file:// is assumed, otherwise if a file does not exist in\n"
            "      the specified path, ollama:// is assumed. Models that are being\n"
            "      pulled are downloaded with .partial extension while being\n"
            "      downloaded and then renamed as the file without the .partial\n"
            "      extension when complete.\n"
            "\n"
            "Examples:\n"
            "  llama-run llama3\n"
            "  llama-run ollama://granite-code\n"
            "  llama-run ollama://smollm:135m\n"
            "  llama-run hf://QuantFactory/SmolLM-135M-GGUF/SmolLM-135M.Q2_K.gguf\n"
            "  llama-run "
            "huggingface://bartowski/SmolLM-1.7B-Instruct-v0.2-GGUF/SmolLM-1.7B-Instruct-v0.2-IQ3_M.gguf\n"
            "  llama-run https://example.com/some-file1.gguf\n"
            "  llama-run some-file2.gguf\n"
            "  llama-run file://some-file3.gguf\n"
            "  llama-run --ngl 999 some-file4.gguf\n"
            "  llama-run --ngl 999 some-file5.gguf Hello World\n",
            context_size_default, ngl_default, temperature_default);
    }
};

struct progress_data {
    size_t                                file_size  = 0;
    std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
    bool                                  printed    = false;
};

static int get_terminal_width() {
#if defined(_WIN32)
    CONSOLE_SCREEN_BUFFER_INFO csbi;
    GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi);
    return csbi.srWindow.Right - csbi.srWindow.Left + 1;
#else
    struct winsize w;
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
    return w.ws_col;
#endif
}

class File {
  public:
    FILE * file = nullptr;

    FILE * open(const std::string & filename, const char * mode) {
        file = ggml_fopen(filename.c_str(), mode);

        return file;
    }

    int lock() {
        if (file) {
#    ifdef _WIN32
            fd    = _fileno(file);
            hFile = (HANDLE) _get_osfhandle(fd);
            if (hFile == INVALID_HANDLE_VALUE) {
                fd = -1;

                return 1;
            }

            OVERLAPPED overlapped = {};
            if (!LockFileEx(hFile, LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY, 0, MAXDWORD, MAXDWORD,
                            &overlapped)) {
                fd = -1;

                return 1;
            }
#    else
            fd = fileno(file);
            if (flock(fd, LOCK_EX | LOCK_NB) != 0) {
                fd = -1;

                return 1;
            }
#    endif
        }

        return 0;
    }

    std::string to_string() {
        fseek(file, 0, SEEK_END);
        const size_t size = ftell(file);
        fseek(file, 0, SEEK_SET);
        std::string out;
        out.resize(size);
        const size_t read_size = fread(&out[0], 1, size, file);
        if (read_size != size) {
            printe("Error reading file: %s", strerror(errno));
        }

        return out;
    }

    ~File() {
        if (fd >= 0) {
#    ifdef _WIN32
            if (hFile != INVALID_HANDLE_VALUE) {
                OVERLAPPED overlapped = {};
                UnlockFileEx(hFile, 0, MAXDWORD, MAXDWORD, &overlapped);
            }
#    else
            flock(fd, LOCK_UN);
#    endif
        }

        if (file) {
            fclose(file);
        }
    }

  private:
    int fd = -1;
#    ifdef _WIN32
    HANDLE hFile = nullptr;
#    endif
};

#ifdef LLAMA_USE_CURL
class HttpClient {
  public:
    int init(const std::string & url, const std::vector<std::string> & headers, const std::string & output_file,
             const bool progress, std::string * response_str = nullptr) {
        if (std::filesystem::exists(output_file)) {
            return 0;
        }

        std::string output_file_partial;
        curl = curl_easy_init();
        if (!curl) {
            return 1;
        }

        progress_data data;
        File          out;
        if (!output_file.empty()) {
            output_file_partial = output_file + ".partial";
            if (!out.open(output_file_partial, "ab")) {
                printe("Failed to open file for writing\n");

                return 1;
            }

            if (out.lock()) {
                printe("Failed to exclusively lock file\n");

                return 1;
            }
        }

        set_write_options(response_str, out);
        data.file_size = set_resume_point(output_file_partial);
        set_progress_options(progress, data);
        set_headers(headers);
        CURLcode res = perform(url);
        if (res != CURLE_OK){
            printe("Fetching resource '%s' failed: %s\n", url.c_str(), curl_easy_strerror(res));
            return 1;
        }
        if (!output_file.empty()) {
            std::filesystem::rename(output_file_partial, output_file);
        }

        return 0;
    }

    ~HttpClient() {
        if (chunk) {
            curl_slist_free_all(chunk);
        }

        if (curl) {
            curl_easy_cleanup(curl);
        }
    }

  private:
    CURL *              curl  = nullptr;
    struct curl_slist * chunk = nullptr;

    void set_write_options(std::string * response_str, const File & out) {
        if (response_str) {
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, capture_data);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, response_str);
        } else {
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, out.file);
        }
    }

    size_t set_resume_point(const std::string & output_file) {
        size_t file_size = 0;
        if (std::filesystem::exists(output_file)) {
            file_size = std::filesystem::file_size(output_file);
            curl_easy_setopt(curl, CURLOPT_RESUME_FROM_LARGE, static_cast<curl_off_t>(file_size));
        }

        return file_size;
    }

    void set_progress_options(bool progress, progress_data & data) {
        if (progress) {
            curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
            curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &data);
            curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, update_progress);
        }
    }

    void set_headers(const std::vector<std::string> & headers) {
        if (!headers.empty()) {
            if (chunk) {
                curl_slist_free_all(chunk);
                chunk = 0;
            }

            for (const auto & header : headers) {
                chunk = curl_slist_append(chunk, header.c_str());
            }

            curl_easy_setopt(curl, CURLOPT_HTTPHEADER, chunk);
        }
    }

    CURLcode perform(const std::string & url) {
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(curl, CURLOPT_DEFAULT_PROTOCOL, "https");
        curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);
        return curl_easy_perform(curl);
    }

    static std::string human_readable_time(double seconds) {
        int hrs  = static_cast<int>(seconds) / 3600;
        int mins = (static_cast<int>(seconds) % 3600) / 60;
        int secs = static_cast<int>(seconds) % 60;

        if (hrs > 0) {
            return fmt("%dh %02dm %02ds", hrs, mins, secs);
        } else if (mins > 0) {
            return fmt("%dm %02ds", mins, secs);
        } else {
            return fmt("%ds", secs);
        }
    }

    static std::string human_readable_size(curl_off_t size) {
        static const char * suffix[] = { "B", "KB", "MB", "GB", "TB" };
        char                length   = sizeof(suffix) / sizeof(suffix[0]);
        int                 i        = 0;
        double              dbl_size = size;
        if (size > 1024) {
            for (i = 0; (size / 1024) > 0 && i < length - 1; i++, size /= 1024) {
                dbl_size = size / 1024.0;
            }
        }

        return fmt("%.2f %s", dbl_size, suffix[i]);
    }

    static int update_progress(void * ptr, curl_off_t total_to_download, curl_off_t now_downloaded, curl_off_t,
                               curl_off_t) {
        progress_data * data = static_cast<progress_data *>(ptr);
        if (total_to_download <= 0) {
            return 0;
        }

        total_to_download += data->file_size;
        const curl_off_t now_downloaded_plus_file_size = now_downloaded + data->file_size;
        const curl_off_t percentage      = calculate_percentage(now_downloaded_plus_file_size, total_to_download);
        std::string      progress_prefix = generate_progress_prefix(percentage);

        const double speed = calculate_speed(now_downloaded, data->start_time);
        const double tim   = (total_to_download - now_downloaded) / speed;
        std::string  progress_suffix =
            generate_progress_suffix(now_downloaded_plus_file_size, total_to_download, speed, tim);

        int         progress_bar_width = calculate_progress_bar_width(progress_prefix, progress_suffix);
        std::string progress_bar;
        generate_progress_bar(progress_bar_width, percentage, progress_bar);

        print_progress(progress_prefix, progress_bar, progress_suffix);
        data->printed = true;

        return 0;
    }

    static curl_off_t calculate_percentage(curl_off_t now_downloaded_plus_file_size, curl_off_t total_to_download) {
        return (now_downloaded_plus_file_size * 100) / total_to_download;
    }

    static std::string generate_progress_prefix(curl_off_t percentage) { return fmt("%3ld%% |", static_cast<long int>(percentage)); }

    static double calculate_speed(curl_off_t now_downloaded, const std::chrono::steady_clock::time_point & start_time) {
        const auto                          now             = std::chrono::steady_clock::now();
        const std::chrono::duration<double> elapsed_seconds = now - start_time;
        return now_downloaded / elapsed_seconds.count();
    }

    static std::string generate_progress_suffix(curl_off_t now_downloaded_plus_file_size, curl_off_t total_to_download,
                                                double speed, double estimated_time) {
        const int width = 10;
        return fmt("%*s/%*s%*s/s%*s", width, human_readable_size(now_downloaded_plus_file_size).c_str(), width,
                   human_readable_size(total_to_download).c_str(), width, human_readable_size(speed).c_str(), width,
                   human_readable_time(estimated_time).c_str());
    }

    static int calculate_progress_bar_width(const std::string & progress_prefix, const std::string & progress_suffix) {
        int progress_bar_width = get_terminal_width() - progress_prefix.size() - progress_suffix.size() - 3;
        if (progress_bar_width < 1) {
            progress_bar_width = 1;
        }

        return progress_bar_width;
    }

    static std::string generate_progress_bar(int progress_bar_width, curl_off_t percentage,
                                             std::string & progress_bar) {
        const curl_off_t pos = (percentage * progress_bar_width) / 100;
        for (int i = 0; i < progress_bar_width; ++i) {
            progress_bar.append((i < pos) ? "█" : " ");
        }

        return progress_bar;
    }

    static void print_progress(const std::string & progress_prefix, const std::string & progress_bar,
                               const std::string & progress_suffix) {
        printe("\r" LOG_CLR_TO_EOL "%s%s| %s", progress_prefix.c_str(), progress_bar.c_str(), progress_suffix.c_str());
    }
    // Function to write data to a file
    static size_t write_data(void * ptr, size_t size, size_t nmemb, void * stream) {
        FILE * out = static_cast<FILE *>(stream);
        return fwrite(ptr, size, nmemb, out);
    }

    // Function to capture data into a string
    static size_t capture_data(void * ptr, size_t size, size_t nmemb, void * stream) {
        std::string * str = static_cast<std::string *>(stream);
        str->append(static_cast<char *>(ptr), size * nmemb);
        return size * nmemb;
    }
};
#endif

class LlamaData {
  public:
    llama_model_ptr                 model;
    llama_sampler_ptr               sampler;
    llama_context_ptr               context;
    std::vector<llama_chat_message> messages; // TODO: switch to common_chat_msg
    std::list<std::string>          msg_strs;
    std::vector<char>               fmtted;

    int init(Opt & opt) {
        model = initialize_model(opt);
        if (!model) {
            return 1;
        }

        context = initialize_context(model, opt);
        if (!context) {
            return 1;
        }

        sampler = initialize_sampler(opt);

        return 0;
    }

  private:
#ifdef LLAMA_USE_CURL
    int download(const std::string & url, const std::string & output_file, const bool progress,
                 const std::vector<std::string> & headers = {}, std::string * response_str = nullptr) {
        HttpClient http;
        if (http.init(url, headers, output_file, progress, response_str)) {
            return 1;
        }

        return 0;
    }
#else
    int download(const std::string &, const std::string &, const bool, const std::vector<std::string> & = {},
                 std::string * = nullptr) {
        printe("%s: llama.cpp built without libcurl, downloading from an url not supported.\n", __func__);

        return 1;
    }
#endif

    // Helper function to handle model tag extraction and URL construction
    std::pair<std::string, std::string> extract_model_and_tag(std::string & model, const std::string & base_url) {
        std::string  model_tag = "latest";
        const size_t colon_pos = model.find(':');
        if (colon_pos != std::string::npos) {
            model_tag = model.substr(colon_pos + 1);
            model     = model.substr(0, colon_pos);
        }

        std::string url = base_url + model + "/manifests/" + model_tag;

        return { model, url };
    }

    // Helper function to download and parse the manifest
    int download_and_parse_manifest(const std::string & url, const std::vector<std::string> & headers,
                                    nlohmann::json & manifest) {
        std::string manifest_str;
        int         ret = download(url, "", false, headers, &manifest_str);
        if (ret) {
            return ret;
        }

        manifest = nlohmann::json::parse(manifest_str);

        return 0;
    }

    int huggingface_dl(std::string & model, const std::string & bn) {
        // Find the second occurrence of '/' after protocol string
        size_t pos = model.find('/');
        pos        = model.find('/', pos + 1);
        std::string              hfr, hff;
        std::vector<std::string> headers = { "User-Agent: llama-cpp", "Accept: application/json" };
        std::string              url;

        if (pos == std::string::npos) {
            auto [model_name, manifest_url] = extract_model_and_tag(model, "https://huggingface.co/v2/");
            hfr                             = model_name;

            nlohmann::json manifest;
            int            ret = download_and_parse_manifest(manifest_url, headers, manifest);
            if (ret) {
                return ret;
            }

            hff = manifest["ggufFile"]["rfilename"];
        } else {
            hfr = model.substr(0, pos);
            hff = model.substr(pos + 1);
        }

        url = "https://huggingface.co/" + hfr + "/resolve/main/" + hff;

        return download(url, bn, true, headers);
    }

    int ollama_dl(std::string & model, const std::string & bn) {
        const std::vector<std::string> headers = { "Accept: application/vnd.docker.distribution.manifest.v2+json" };
        if (model.find('/') == std::string::npos) {
            model = "library/" + model;
        }

        auto [model_name, manifest_url] = extract_model_and_tag(model, "https://registry.ollama.ai/v2/");
        nlohmann::json manifest;
        int            ret = download_and_parse_manifest(manifest_url, {}, manifest);
        if (ret) {
            return ret;
        }

        std::string layer;
        for (const auto & l : manifest["layers"]) {
            if (l["mediaType"] == "application/vnd.ollama.image.model") {
                layer = l["digest"];
                break;
            }
        }

        std::string blob_url = "https://registry.ollama.ai/v2/" + model_name + "/blobs/" + layer;

        return download(blob_url, bn, true, headers);
    }

    int github_dl(const std::string & model, const std::string & bn) {
        std::string  repository = model;
        std::string  branch     = "main";
        const size_t at_pos     = model.find('@');
        if (at_pos != std::string::npos) {
            repository = model.substr(0, at_pos);
            branch     = model.substr(at_pos + 1);
        }

        const std::vector<std::string> repo_parts = string_split(repository, "/");
        if (repo_parts.size() < 3) {
            printe("Invalid GitHub repository format\n");
            return 1;
        }

        const std::string & org          = repo_parts[0];
        const std::string & project      = repo_parts[1];
        std::string         url          = "https://raw.githubusercontent.com/" + org + "/" + project + "/" + branch;
        for (size_t i = 2; i < repo_parts.size(); ++i) {
            url += "/" + repo_parts[i];
        }

        return download(url, bn, true);
    }

    int s3_dl(const std::string & model, const std::string & bn) {
        const size_t slash_pos = model.find('/');
        if (slash_pos == std::string::npos) {
            return 1;
        }

        const std::string bucket     = model.substr(0, slash_pos);
        const std::string key        = model.substr(slash_pos + 1);
        const char * access_key = std::getenv("AWS_ACCESS_KEY_ID");
        const char * secret_key = std::getenv("AWS_SECRET_ACCESS_KEY");
        if (!access_key || !secret_key) {
            printe("AWS credentials not found in environment\n");
            return 1;
        }

        // Generate AWS Signature Version 4 headers
        // (Implementation requires HMAC-SHA256 and date handling)
        // Get current timestamp
        const time_t                   now     = time(nullptr);
        const tm                       tm      = *gmtime(&now);
        const std::string              date     = strftime_fmt("%Y%m%d", tm);
        const std::string              datetime = strftime_fmt("%Y%m%dT%H%M%SZ", tm);
        const std::vector<std::string> headers  = {
            "Authorization: AWS4-HMAC-SHA256 Credential=" + std::string(access_key) + "/" + date +
                "/us-east-1/s3/aws4_request",
            "x-amz-content-sha256: UNSIGNED-PAYLOAD", "x-amz-date: " + datetime
        };

        const std::string url = "https://" + bucket + ".s3.amazonaws.com/" + key;

        return download(url, bn, true, headers);
    }

    std::string basename(const std::string & path) {
        const size_t pos = path.find_last_of("/\\");
        if (pos == std::string::npos) {
            return path;
        }

        return path.substr(pos + 1);
    }

    int rm_until_substring(std::string & model_, const std::string & substring) {
        const std::string::size_type pos = model_.find(substring);
        if (pos == std::string::npos) {
            return 1;
        }

        model_ = model_.substr(pos + substring.size());  // Skip past the substring
        return 0;
    }

    int resolve_model(std::string & model_) {
        int                            ret     = 0;
        if (string_starts_with(model_, "file://") || std::filesystem::exists(model_)) {
            rm_until_substring(model_, "://");

            return ret;
        }

        const std::string bn = basename(model_);
        if (string_starts_with(model_, "hf://") || string_starts_with(model_, "huggingface://") ||
            string_starts_with(model_, "hf.co/")) {
            rm_until_substring(model_, "hf.co/");
            rm_until_substring(model_, "://");
            ret = huggingface_dl(model_, bn);
        } else if ((string_starts_with(model_, "https://") || string_starts_with(model_, "http://")) &&
                   !string_starts_with(model_, "https://ollama.com/library/")) {
            ret = download(model_, bn, true);
        } else if (string_starts_with(model_, "github:") || string_starts_with(model_, "github://")) {
            rm_until_substring(model_, "github:");
            rm_until_substring(model_, "://");
            ret = github_dl(model_, bn);
        } else if (string_starts_with(model_, "s3://")) {
            rm_until_substring(model_, "://");
            ret = s3_dl(model_, bn);
        } else {  // ollama:// or nothing
            rm_until_substring(model_, "ollama.com/library/");
            rm_until_substring(model_, "://");
            ret = ollama_dl(model_, bn);
        }

        model_ = bn;

        return ret;
    }

    // Initializes the model and returns a unique pointer to it
    llama_model_ptr initialize_model(Opt & opt) {
        ggml_backend_load_all();
        resolve_model(opt.model_);
        printe("\r" LOG_CLR_TO_EOL "Loading model");
        llama_model_ptr model(llama_model_load_from_file(opt.model_.c_str(), opt.model_params));
        if (!model) {
            printe("%s: error: unable to load model from file: %s\n", __func__, opt.model_.c_str());
        }

        printe("\r" LOG_CLR_TO_EOL);
        return model;
    }

    // Initializes the context with the specified parameters
    llama_context_ptr initialize_context(const llama_model_ptr & model, const Opt & opt) {
        llama_context_ptr context(llama_init_from_model(model.get(), opt.ctx_params));
        if (!context) {
            printe("%s: error: failed to create the llama_context\n", __func__);
        }

        return context;
    }

    // Initializes and configures the sampler
    llama_sampler_ptr initialize_sampler(const Opt & opt) {
        llama_sampler_ptr sampler(llama_sampler_chain_init(llama_sampler_chain_default_params()));
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_min_p(0.05f, 1));
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_temp(opt.temperature));
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

        return sampler;
    }
};

// Add a message to `messages` and store its content in `msg_strs`
static void add_message(const char * role, const std::string & text, LlamaData & llama_data) {
    llama_data.msg_strs.push_back(std::move(text));
    llama_data.messages.push_back({ role, llama_data.msg_strs.back().c_str() });
}

// Function to apply the chat template and resize `formatted` if needed
static int apply_chat_template(const struct common_chat_templates * tmpls, LlamaData & llama_data, const bool append, bool use_jinja) {
    common_chat_templates_inputs inputs;
    for (const auto & msg : llama_data.messages) {
        common_chat_msg cmsg;
        cmsg.role    = msg.role;
        cmsg.content = msg.content;
        inputs.messages.push_back(cmsg);
    }
    inputs.add_generation_prompt = append;
    inputs.use_jinja = use_jinja;

    auto chat_params = common_chat_templates_apply(tmpls, inputs);
    // TODO: use other params for tool calls.
    auto result = chat_params.prompt;
    llama_data.fmtted.resize(result.size() + 1);
    memcpy(llama_data.fmtted.data(), result.c_str(), result.size() + 1);
    return result.size();
}

// Function to tokenize the prompt
static int tokenize_prompt(const llama_vocab * vocab, const std::string & prompt,
                           std::vector<llama_token> & prompt_tokens, const LlamaData & llama_data) {
    const bool is_first = llama_kv_self_used_cells(llama_data.context.get()) == 0;

    const int n_prompt_tokens = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, is_first, true);
    prompt_tokens.resize(n_prompt_tokens);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), is_first,
                       true) < 0) {
        printe("failed to tokenize the prompt\n");
        return -1;
    }

    return n_prompt_tokens;
}

// Check if we have enough space in the context to evaluate this batch
static int check_context_size(const llama_context_ptr & ctx, const llama_batch & batch) {
    const int n_ctx      = llama_n_ctx(ctx.get());
    const int n_ctx_used = llama_kv_self_used_cells(ctx.get());
    if (n_ctx_used + batch.n_tokens > n_ctx) {
        printf(LOG_COL_DEFAULT "\n");
        printe("context size exceeded\n");
        return 1;
    }

    return 0;
}

// convert the token to a string
static int convert_token_to_string(const llama_vocab * vocab, const llama_token token_id, std::string & piece) {
    char buf[256];
    int  n = llama_token_to_piece(vocab, token_id, buf, sizeof(buf), 0, true);
    if (n < 0) {
        printe("failed to convert token to piece\n");
        return 1;
    }

    piece = std::string(buf, n);
    return 0;
}

static void print_word_and_concatenate_to_response(const std::string & piece, std::string & response) {
    printf("%s", piece.c_str());
    fflush(stdout);
    response += piece;
}

// helper function to evaluate a prompt and generate a response
static int generate(LlamaData & llama_data, const std::string & prompt, std::string & response) {
    const llama_vocab * vocab = llama_model_get_vocab(llama_data.model.get());

    std::vector<llama_token> tokens;
    if (tokenize_prompt(vocab, prompt, tokens, llama_data) < 0) {
        return 1;
    }

    // prepare a batch for the prompt
    llama_batch batch = llama_batch_get_one(tokens.data(), tokens.size());
    llama_token new_token_id;
    while (true) {
        check_context_size(llama_data.context, batch);
        if (llama_decode(llama_data.context.get(), batch)) {
            printe("failed to decode\n");
            return 1;
        }

        // sample the next token, check is it an end of generation?
        new_token_id = llama_sampler_sample(llama_data.sampler.get(), llama_data.context.get(), -1);
        if (llama_vocab_is_eog(vocab, new_token_id)) {
            break;
        }

        std::string piece;
        if (convert_token_to_string(vocab, new_token_id, piece)) {
            return 1;
        }

        print_word_and_concatenate_to_response(piece, response);

        // prepare the next batch with the sampled token
        batch = llama_batch_get_one(&new_token_id, 1);
    }

    printf(LOG_COL_DEFAULT);
    return 0;
}

static int read_user_input(std::string & user_input) {
    static const char * prompt_prefix_env = std::getenv("LLAMA_PROMPT_PREFIX");
    static const char * prompt_prefix     = prompt_prefix_env ? prompt_prefix_env : "> ";
#ifdef WIN32
    printf("\r" LOG_CLR_TO_EOL LOG_COL_DEFAULT "%s", prompt_prefix);

    std::getline(std::cin, user_input);
    if (std::cin.eof()) {
        printf("\n");
        return 1;
    }
#else
    std::unique_ptr<char, decltype(&std::free)> line(const_cast<char *>(linenoise(prompt_prefix)), free);
    if (!line) {
        return 1;
    }

    user_input = line.get();
#endif

    if (user_input == "/bye") {
        return 1;
    }

    if (user_input.empty()) {
        return 2;
    }

#ifndef WIN32
    linenoiseHistoryAdd(line.get());
#endif

    return 0;  // Should have data in happy path
}

// Function to generate a response based on the prompt
static int generate_response(LlamaData & llama_data, const std::string & prompt, std::string & response,
                             const bool stdout_a_terminal) {
    // Set response color
    if (stdout_a_terminal) {
        printf(LOG_COL_YELLOW);
    }

    if (generate(llama_data, prompt, response)) {
        printe("failed to generate response\n");
        return 1;
    }

    // End response with color reset and newline
    printf("\n%s", stdout_a_terminal ? LOG_COL_DEFAULT : "");
    return 0;
}

// Helper function to apply the chat template and handle errors
static int apply_chat_template_with_error_handling(const common_chat_templates * tmpls, LlamaData & llama_data, const bool append, int & output_length, bool use_jinja) {
    const int new_len = apply_chat_template(tmpls, llama_data, append, use_jinja);
    if (new_len < 0) {
        printe("failed to apply the chat template\n");
        return -1;
    }

    output_length = new_len;
    return 0;
}

// Helper function to handle user input
static int handle_user_input(std::string & user_input, const std::string & user) {
    if (!user.empty()) {
        user_input = user;
        return 0;  // No need for interactive input
    }

    return read_user_input(user_input);  // Returns true if input ends the loop
}

static bool is_stdin_a_terminal() {
#if defined(_WIN32)
    HANDLE hStdin = GetStdHandle(STD_INPUT_HANDLE);
    DWORD  mode;
    return GetConsoleMode(hStdin, &mode);
#else
    return isatty(STDIN_FILENO);
#endif
}

static bool is_stdout_a_terminal() {
#if defined(_WIN32)
    HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD  mode;
    return GetConsoleMode(hStdout, &mode);
#else
    return isatty(STDOUT_FILENO);
#endif
}

// Function to handle user input
static int get_user_input(std::string & user_input, const std::string & user) {
    while (true) {
        const int ret = handle_user_input(user_input, user);
        if (ret == 1) {
            return 1;
        }

        if (ret == 2) {
            continue;
        }

        break;
    }

    return 0;
}

// Reads a chat template file to be used
static std::string read_chat_template_file(const std::string & chat_template_file) {
    File file;
    if (!file.open(chat_template_file, "r")) {
        printe("Error opening chat template file '%s': %s", chat_template_file.c_str(), strerror(errno));
        return "";
    }

    return file.to_string();
}

static int process_user_message(const Opt & opt, const std::string & user_input, LlamaData & llama_data,
                                const common_chat_templates_ptr & chat_templates, int & prev_len,
                                const bool stdout_a_terminal) {
    add_message("user", opt.user.empty() ? user_input : opt.user, llama_data);
    int new_len;
    if (apply_chat_template_with_error_handling(chat_templates.get(), llama_data, true, new_len, opt.use_jinja) < 0) {
        return 1;
    }

    std::string prompt(llama_data.fmtted.begin() + prev_len, llama_data.fmtted.begin() + new_len);
    std::string response;
    if (generate_response(llama_data, prompt, response, stdout_a_terminal)) {
        return 1;
    }

    if (!opt.user.empty()) {
        return 2;
    }

    add_message("assistant", response, llama_data);
    if (apply_chat_template_with_error_handling(chat_templates.get(), llama_data, false, prev_len, opt.use_jinja) < 0) {
        return 1;
    }

    return 0;
}

// Main chat loop function
static int chat_loop(LlamaData & llama_data, const Opt & opt) {
    int prev_len = 0;
    llama_data.fmtted.resize(llama_n_ctx(llama_data.context.get()));
    std::string chat_template;
    if (!opt.chat_template_file.empty()) {
        chat_template = read_chat_template_file(opt.chat_template_file);
    }

    common_chat_templates_ptr chat_templates    = common_chat_templates_init(llama_data.model.get(), chat_template);
    static const bool stdout_a_terminal = is_stdout_a_terminal();
    while (true) {
        // Get user input
        std::string user_input;
        if (get_user_input(user_input, opt.user) == 1) {
            return 0;
        }

        const int ret = process_user_message(opt, user_input, llama_data, chat_templates, prev_len, stdout_a_terminal);
        if (ret == 1) {
            return 1;
        } else if (ret == 2) {
            break;
        }
    }

    return 0;
}

static void log_callback(const enum ggml_log_level level, const char * text, void * p) {
    const Opt * opt = static_cast<Opt *>(p);
    if (opt->verbose || level == GGML_LOG_LEVEL_ERROR) {
        printe("%s", text);
    }
}

static std::string read_pipe_data() {
    std::ostringstream result;
    result << std::cin.rdbuf();  // Read all data from std::cin
    return result.str();
}

static void ctrl_c_handling() {
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
    struct sigaction sigint_action;
    sigint_action.sa_handler = sigint_handler;
    sigemptyset(&sigint_action.sa_mask);
    sigint_action.sa_flags = 0;
    sigaction(SIGINT, &sigint_action, NULL);
#elif defined(_WIN32)
    auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
        return (ctrl_type == CTRL_C_EVENT) ? (sigint_handler(SIGINT), true) : false;
    };
    SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif
}

int main(int argc, const char ** argv) {
    ctrl_c_handling();
    Opt       opt;
    const int ret = opt.init(argc, argv);
    if (ret == 2) {
        return 0;
    } else if (ret) {
        return 1;
    }

    if (!is_stdin_a_terminal()) {
        if (!opt.user.empty()) {
            opt.user += "\n\n";
        }

        opt.user += read_pipe_data();
    }

    llama_log_set(log_callback, &opt);
    LlamaData llama_data;
    if (llama_data.init(opt)) {
        return 1;
    }

    if (chat_loop(llama_data, opt)) {
        return 1;
    }

    return 0;
}
