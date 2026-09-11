#include "ggml-rpc.h"
#ifdef _WIN32
#  define NOMINMAX
#  define DIRECTORY_SEPARATOR '\\'
#  include <windows.h>
#  include <fcntl.h>
#  include <io.h>
#  define isatty _isatty
#  define fileno _fileno
#else
#  define DIRECTORY_SEPARATOR '/'
#  include <unistd.h>
#  include <sys/stat.h>
#endif
#include <algorithm>
#include <chrono>
#include <clocale>
#include <codecvt>
#include <cstdarg>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <regex>
#include <stdio.h>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <sys/types.h>
#include <pwd.h>
#endif

// NOTE: this is copied from common.cpp to avoid linking with libcommon
#ifdef _WIN32
static std::wstring utf8_to_wstring(const std::string & str) {
    if (str.empty()) {
        return std::wstring();
    }

    int size = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), (int)str.size(), NULL, 0);

    if (size <= 0) {
        return std::wstring();
    }

    std::wstring wstr(size, 0);
    MultiByteToWideChar(CP_UTF8, 0, str.c_str(), (int)str.size(), &wstr[0], size);

    return wstr;
}
#endif

// NOTE: this is copied from common.cpp to avoid linking with libcommon
// returns true if successful, false otherwise
static bool fs_create_directory_with_parents(const std::string & path) {
#ifdef _WIN32
    std::wstring wpath = utf8_to_wstring(path);

    // if the path already exists, check whether it's a directory
    const DWORD attributes = GetFileAttributesW(wpath.c_str());
    if ((attributes != INVALID_FILE_ATTRIBUTES) && (attributes & FILE_ATTRIBUTE_DIRECTORY)) {
        return true;
    }

    size_t pos_slash = 0;

    // process path from front to back, procedurally creating directories
    while ((pos_slash = path.find('\\', pos_slash)) != std::string::npos) {
        const std::wstring subpath = wpath.substr(0, pos_slash);

        pos_slash += 1;

        // skip the drive letter, in some systems it can return an access denied error
        if (subpath.length() == 2 && subpath[1] == ':') {
            continue;
        }

        const bool success = CreateDirectoryW(subpath.c_str(), NULL);

        if (!success) {
            const DWORD error = GetLastError();

            // if the path already exists, ensure that it's a directory
            if (error == ERROR_ALREADY_EXISTS) {
                const DWORD attributes = GetFileAttributesW(subpath.c_str());
                if (attributes == INVALID_FILE_ATTRIBUTES || !(attributes & FILE_ATTRIBUTE_DIRECTORY)) {
                    return false;
                }
            } else {
                return false;
            }
        }
    }

    return true;
#else
    // if the path already exists, check whether it's a directory
    struct stat info;
    if (stat(path.c_str(), &info) == 0) {
        return S_ISDIR(info.st_mode);
    }

    size_t pos_slash = 1; // skip leading slashes for directory creation

    // process path from front to back, procedurally creating directories
    while ((pos_slash = path.find('/', pos_slash)) != std::string::npos) {
        const std::string subpath = path.substr(0, pos_slash);
        struct stat info;

        // if the path already exists, ensure that it's a directory
        if (stat(subpath.c_str(), &info) == 0) {
            if (!S_ISDIR(info.st_mode)) {
                return false;
            }
        } else {
            // create parent directories
            const int ret = mkdir(subpath.c_str(), 0755);
            if (ret != 0) {
                return false;
            }
        }

        pos_slash += 1;
    }

    return true;
#endif // _WIN32
}

// NOTE: this is copied from common.cpp to avoid linking with libcommon
static std::string fs_get_cache_directory() {
    std::string cache_directory = "";
    auto ensure_trailing_slash = [](std::string p) {
        // Make sure to add trailing slash
        if (p.back() != DIRECTORY_SEPARATOR) {
            p += DIRECTORY_SEPARATOR;
        }
        return p;
    };
    if (getenv("LLAMA_CACHE")) {
        cache_directory = std::getenv("LLAMA_CACHE");
    } else {
#if defined(__linux__) || defined(__FreeBSD__) || defined(_AIX) || \
    defined(__OpenBSD__) || defined(__NetBSD__)
        if (std::getenv("XDG_CACHE_HOME")) {
            cache_directory = std::getenv("XDG_CACHE_HOME");
        } else if (std::getenv("HOME")) {
            cache_directory = std::getenv("HOME") + std::string("/.cache/");
        } else {
#if defined(__linux__)
            /* no $HOME is defined, fallback to getpwuid */
            struct passwd *pw = getpwuid(getuid());
            if ((!pw) || (!pw->pw_dir)) {
                throw std::runtime_error("Failed to find $HOME directory");
            }

            cache_directory = std::string(pw->pw_dir) + std::string("/.cache/");
#else /* defined(__linux__) */
            throw std::runtime_error("Failed to find $HOME directory");
#endif /* defined(__linux__) */
        }
#elif defined(__APPLE__)
        cache_directory = std::getenv("HOME") + std::string("/Library/Caches/");
#elif defined(_WIN32)
        cache_directory = std::getenv("LOCALAPPDATA");
#elif defined(__EMSCRIPTEN__)
        GGML_ABORT("not implemented on this platform");
#else
#  error Unknown architecture
#endif
        cache_directory = ensure_trailing_slash(cache_directory);
        cache_directory += "llama.cpp";
    }
    return ensure_trailing_slash(cache_directory);
}

// NOTE: this is copied from common/log.h to avoid linking with libcommon
#define LOG_LEVEL_DEBUG  5
#define LOG_LEVEL_TRACE  4
#define LOG_LEVEL_INFO   3
#define LOG_LEVEL_WARN   2
#define LOG_LEVEL_ERROR  1
#define LOG_LEVEL_OUTPUT 0 // output data from tools

static int              g_log_verbosity = LOG_LEVEL_INFO;
static bool             g_log_colors    = false;
static int64_t          g_log_t_start   = 0;
static std::mutex       g_log_mtx;

// NOTE: this is copied from common/log.cpp to avoid linking with libcommon
static int64_t t_us() {
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

// NOTE: this is copied from common.cpp to avoid linking with libcommon
static bool tty_can_use_colors() {
    // Check NO_COLOR environment variable (https://no-color.org/)
    if (const char * no_color = std::getenv("NO_COLOR")) {
        if (no_color[0] != '\0') {
            return false;
        }
    }

    // Check TERM environment variable
    if (const char * term = std::getenv("TERM")) {
        if (std::strcmp(term, "dumb") == 0) {
            return false;
        }
    }

    // Check if stdout and stderr are connected to a terminal
    // We check both because log messages can go to either
    bool stdout_is_tty = isatty(fileno(stdout));
    bool stderr_is_tty = isatty(fileno(stderr));

    return stdout_is_tty || stderr_is_tty;
}

// NOTE: this is copied from common/log.cpp to avoid linking with libcommon
static int common_get_verbosity(enum ggml_log_level level) {
    switch (level) {
        case GGML_LOG_LEVEL_DEBUG: return LOG_LEVEL_DEBUG;
        case GGML_LOG_LEVEL_INFO:  return LOG_LEVEL_TRACE;
        case GGML_LOG_LEVEL_WARN:  return LOG_LEVEL_WARN;
        case GGML_LOG_LEVEL_ERROR: return LOG_LEVEL_ERROR;
        case GGML_LOG_LEVEL_CONT:  return LOG_LEVEL_TRACE;
        case GGML_LOG_LEVEL_NONE:
        default:
            return LOG_LEVEL_OUTPUT;
    }
}

// NOTE: this is copied (and simplified) from common/log.cpp to avoid linking with libcommon
static void log_add(enum ggml_log_level level, const char * fmt, ...) {
    if (common_get_verbosity(level) > g_log_verbosity) {
        return;
    }

    va_list args;
    va_start(args, fmt);
    std::vector<char> buf(256);
    int n = vsnprintf(buf.data(), buf.size(), fmt, args);
    if (n >= (int) buf.size()) {
        buf.resize(n + 1);
        va_list args_copy;
        va_copy(args_copy, args);
        vsnprintf(buf.data(), buf.size(), fmt, args_copy);
        va_end(args_copy);
    }
    va_end(args);

    std::lock_guard<std::mutex> lock(g_log_mtx);

    FILE * f = stderr;

    if (level != GGML_LOG_LEVEL_NONE) {
        const char * col_blue    = g_log_colors ? "\033[34m" : "";
        const char * col_default = g_log_colors ? "\033[0m"  : "";

        const int64_t ts = t_us() - g_log_t_start;
        fprintf(f, "%s%d.%02d.%03d.%03d%s ",
                col_blue,
                (int) (ts / 1000000 / 60),
                (int) (ts / 1000000 % 60),
                (int) (ts / 1000 % 1000),
                (int) (ts % 1000),
                col_default);

        switch (level) {
            case GGML_LOG_LEVEL_INFO:  fprintf(f, "%sI %s", g_log_colors ? "\033[32m" : "", col_default); break;
            case GGML_LOG_LEVEL_WARN:  fprintf(f, "%sW %s", g_log_colors ? "\033[35m" : "", "");          break;
            case GGML_LOG_LEVEL_ERROR: fprintf(f, "%sE %s", g_log_colors ? "\033[31m" : "", "");          break;
            case GGML_LOG_LEVEL_DEBUG: fprintf(f, "%sD %s", g_log_colors ? "\033[33m" : "", "");          break;
            default:
                break;
        }
    }

    fprintf(f, "%s", buf.data());

    if (level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_DEBUG) {
        fprintf(f, "%s", g_log_colors ? "\033[0m" : "");
    }

    fflush(f);
}

static void log_callback(enum ggml_log_level level, const char * text, void * /*user_data*/) {
    log_add(level, "%s", text);
}

#define RPC_INF(fmt, ...) log_add(GGML_LOG_LEVEL_INFO,  "%s: " fmt, __func__, __VA_ARGS__)
#define RPC_WRN(fmt, ...) log_add(GGML_LOG_LEVEL_WARN,  "%s: " fmt, __func__, __VA_ARGS__)
#define RPC_ERR(fmt, ...) log_add(GGML_LOG_LEVEL_ERROR, "%s: " fmt, __func__, __VA_ARGS__)

struct rpc_server_params {
    std::string              host        = "127.0.0.1";
    int                      port        = 50052;
    bool                     use_cache   = false;
    int                      n_threads   = std::max(1U, std::thread::hardware_concurrency()/2);
    std::vector<std::string> devices;
};

static void print_usage(int /*argc*/, char ** argv, rpc_server_params params) {
    fprintf(stderr, "Usage: %s [options]\n\n", argv[0]);
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help                       show this help message and exit\n");
    fprintf(stderr, "  -t, --threads N                  number of threads for the CPU device (default: %d)\n", params.n_threads);
    fprintf(stderr, "  -d, --device <dev1,dev2,...>     comma-separated list of devices\n");
    fprintf(stderr, "  -H, --host HOST                  host to bind to (default: %s)\n", params.host.c_str());
    fprintf(stderr, "  -p, --port PORT                  port to bind to (default: %d)\n", params.port);
    fprintf(stderr, "  -c, --cache                      enable local file cache\n");
    fprintf(stderr, "\n");
}

static bool rpc_server_params_parse(int argc, char ** argv, rpc_server_params & params) {
    std::string arg;
    for (int i = 1; i < argc; i++) {
        arg = argv[i];
        if (arg == "-H" || arg == "--host") {
            if (++i >= argc) {
                return false;
            }
            params.host = argv[i];
        } else if (arg == "-t" || arg == "--threads") {
            if (++i >= argc) {
                return false;
            }
            params.n_threads = std::stoi(argv[i]);
            if (params.n_threads <= 0) {
                RPC_ERR("error: invalid number of threads: %d\n", params.n_threads);
                return false;
            }
        } else if (arg == "-d" || arg == "--device") {
            if (++i >= argc) {
                return false;
            }
            const std::regex regex{ R"([,/]+)" };
            std::string dev_str = argv[i];
            std::sregex_token_iterator iter(dev_str.begin(), dev_str.end(), regex, -1);
            std::sregex_token_iterator end;
            for ( ; iter != end; ++iter) {
                try {
                    params.devices.push_back(*iter);
                } catch (const std::exception & ) {
                    RPC_ERR("error: invalid device: %s\n", iter->str().c_str());
                    return false;
                }
            }
        } else if (arg == "-p" || arg == "--port") {
            if (++i >= argc) {
                return false;
            }
            params.port = std::stoi(argv[i]);
            if (params.port <= 0 || params.port > 65535) {
                return false;
            }
        } else if (arg == "-c" || arg == "--cache") {
            params.use_cache = true;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argc, argv, params);
            exit(0);
        } else {
            RPC_ERR("error: unknown argument: %s\n", arg.c_str());
            print_usage(argc, argv, params);
            exit(0);
        }
    }
    return true;
}

static std::vector<ggml_backend_dev_t> get_devices(const rpc_server_params & params) {
    std::vector<ggml_backend_dev_t> devices;
    if (!params.devices.empty()) {
        for (auto device : params.devices) {
            ggml_backend_dev_t dev = ggml_backend_dev_by_name(device.c_str());
            if (dev) {
                devices.push_back(dev);
            } else {
                RPC_ERR("error: unknown device: %s\n", device.c_str());
                RPC_INF("%s", "available devices:\n");
                for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
                    auto * dev = ggml_backend_dev_get(i);
                    size_t free, total;
                    ggml_backend_dev_memory(dev, &free, &total);
                    RPC_INF("  %s: %s (%zu MiB, %zu MiB free)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev), total / 1024 / 1024, free / 1024 / 1024);
                }
                return {};
            }
        }
    }

    // Try non-CPU devices first
    if (devices.empty()) {
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
                devices.push_back(dev);
            }
        }
    }

    // If there are no accelerators, fallback to CPU device
    if (devices.empty()) {
        ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        if (dev) {
            devices.push_back(dev);
        }
    }

    return devices;
}

int main(int argc, char * argv[]) {
    std::setlocale(LC_NUMERIC, "C");

    g_log_t_start = t_us();
    g_log_colors  = tty_can_use_colors();
    g_log_verbosity = std::getenv("GGML_RPC_DEBUG") ? LOG_LEVEL_DEBUG : common_get_verbosity(GGML_LOG_LEVEL_INFO);
    ggml_log_set(log_callback, NULL);

    rpc_server_params params;
    if (!rpc_server_params_parse(argc, argv, params)) {
        RPC_ERR("%s", "Invalid parameters\n");
        return 1;
    }

    ggml_backend_load_all();

    if (params.host != "127.0.0.1") {
        RPC_WRN("%s", "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        RPC_WRN("WARNING: Host ('%s') is != '127.0.0.1'\n", params.host.c_str());
        RPC_WRN("%s", "         Never expose the RPC server to an open network!\n");
        RPC_WRN("%s", "         This is an experimental feature and is not secure!\n");
        RPC_WRN("%s", "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    }

    auto devices = get_devices(params);
    if (devices.empty()) {
        RPC_ERR("%s", "No devices found\n");
        return 1;
    }
    std::string endpoint = params.host + ":" + std::to_string(params.port);
    const char * cache_dir = nullptr;
    std::string cache_dir_str;
    if (params.use_cache) {
        cache_dir_str = fs_get_cache_directory() + "rpc" + DIRECTORY_SEPARATOR;
        if (!fs_create_directory_with_parents(cache_dir_str)) {
            RPC_ERR("Failed to create cache directory: %s\n", cache_dir_str.c_str());
            return 1;
        }
        cache_dir = cache_dir_str.c_str();
    }

    ggml_backend_reg_t reg = ggml_backend_reg_by_name("RPC");
    if (!reg) {
        RPC_ERR("%s", "Failed to find RPC backend\n");
        return 1;
    }

    auto start_server_fn = (decltype(ggml_backend_rpc_start_server)*) ggml_backend_reg_get_proc_address(reg, "ggml_backend_rpc_start_server");
    if (!start_server_fn) {
        RPC_ERR("%s", "Failed to obtain RPC backend start server function\n");
        return 1;
    }

    start_server_fn(endpoint.c_str(), cache_dir, params.n_threads, devices.size(), devices.data());
    return 0;
}
