#include "ggml-backend.h"
#include "ggml-rpc.h"
#ifdef _WIN32
#  define NOMINMAX
#  include <windows.h>
#  include <fcntl.h>
#  include <io.h>
#else
#  include <unistd.h>
#endif
#include <algorithm>
#include <clocale>
#include <codecvt>
#include <filesystem>
#include <regex>
#include <stdio.h>
#include <string>
#include <thread>
#include <vector>

#if !defined(_WIN32)
#include <sys/types.h>
#include <pwd.h>
#endif


// NOTE: this is copied from common.cpp to avoid linking with libcommon
static std::string fs_path_to_utf8(const std::filesystem::path & path) {
    const auto value = path.u8string();
    return std::string(value.begin(), value.end());
}

// common_get_path_from_env() is adapted to avoid utf8_to_wstring
static std::filesystem::path common_get_path_from_env(const std::string & name) {
#ifdef _WIN32
    std::wstring wname;
    for (const char * p = name.c_str(); *p; ++p) {
        wname.push_back((wchar_t)*p);
    }
    const wchar_t * wvalue = _wgetenv(wname.c_str());
    return wvalue ? std::filesystem::path(wvalue) : std::filesystem::path();
#else
    const char * value = std::getenv(name.c_str());
    return value ? std::filesystem::path(value) : std::filesystem::path();
#endif
}

// NOTE: this is copied from common.cpp to avoid linking with libcommon
#if !defined(_WIN32)
static std::filesystem::path get_home_directory() {
    std::filesystem::path home = common_get_path_from_env("HOME");
    if (!home.empty()) {
        return home;
    }
    const struct passwd * pw = getpwuid(getuid());
    if (!pw || !pw->pw_dir || !*pw->pw_dir) {
        throw std::runtime_error("Failed to find $HOME directory");
    }
    return pw->pw_dir;
}
#endif

// NOTE: this is copied from common.cpp to avoid linking with libcommon
static std::filesystem::path fs_get_cache_directory() {
    std::filesystem::path cache_directory = common_get_path_from_env("LLAMA_CACHE");
    if (!cache_directory.empty()) {
        return cache_directory;
    }
#if defined(_WIN32)
    cache_directory = common_get_path_from_env("LOCALAPPDATA");
    if (cache_directory.empty()) {
        throw std::runtime_error("Failed to find %LOCALAPPDATA% directory");
    }
#elif defined(__APPLE__)
    cache_directory = get_home_directory() / "Library/Caches";
#else
    cache_directory = common_get_path_from_env("XDG_CACHE_HOME");
    if (cache_directory.empty()) {
        cache_directory = get_home_directory() / ".cache";
    }
#endif
    return cache_directory / "llama.cpp";
}

// NOTE: this is copied from common.h to avoid linking with libcommon
static bool common_create_directories(const std::filesystem::path & path, std::error_code & ec) {
#if defined(__linux__)
    return std::filesystem::create_directories(path / "", ec);
#else
    return std::filesystem::create_directories(path, ec);
#endif
}

struct rpc_server_params {
    std::string              host        = "::";
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
            // Strip brackets from IPv6 addresses (e.g., [::] -> ::)
            if (params.host.size() >= 2 && params.host.front() == '[' && params.host.back() == ']') {
                params.host = params.host.substr(1, params.host.size() - 2);
            }
        } else if (arg == "-t" || arg == "--threads") {
            if (++i >= argc) {
                return false;
            }
            params.n_threads = std::stoi(argv[i]);
            if (params.n_threads <= 0) {
                fprintf(stderr, "error: invalid number of threads: %d\n", params.n_threads);
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
                    fprintf(stderr, "error: invalid device: %s\n", iter->str().c_str());
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
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
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
                fprintf(stderr, "error: unknown device: %s\n", device.c_str());
                fprintf(stderr, "available devices:\n");
                for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
                    auto * dev = ggml_backend_dev_get(i);
                    size_t free, total;
                    ggml_backend_dev_memory(dev, &free, &total);
                    printf("  %s: %s (%zu MiB, %zu MiB free)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev), total / 1024 / 1024, free / 1024 / 1024);
                }
                return {};
            }
        }
    }

    // Try non-CPU devices first
    if (devices.empty()) {
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            enum ggml_backend_dev_type dev_type = ggml_backend_dev_type(dev);
            if (dev_type != GGML_BACKEND_DEVICE_TYPE_CPU && dev_type != GGML_BACKEND_DEVICE_TYPE_ACCEL) {
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

    ggml_backend_load_all();

    rpc_server_params params;
    if (!rpc_server_params_parse(argc, argv, params)) {
        fprintf(stderr, "Invalid parameters\n");
        return 1;
    }

    if (params.host != "127.0.0.1" && params.host != "::1") {
        fprintf(stderr, "\n");
        fprintf(stderr, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        fprintf(stderr, "WARNING: Host ('%s') is not a loopback address\n", params.host.c_str());
        fprintf(stderr, "         Never expose the RPC server to an open network!\n");
        fprintf(stderr, "         This is an experimental feature and is not secure!\n");
        fprintf(stderr, "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        fprintf(stderr, "\n");
    }

    auto devices = get_devices(params);
    if (devices.empty()) {
        fprintf(stderr, "No devices found\n");
        return 1;
    }
    std::string endpoint;
    if (params.host.find(':') != std::string::npos) {
        // IPv6 address needs bracket notation in endpoint
        endpoint = "[" + params.host + "]:" + std::to_string(params.port);
    } else {
        endpoint = params.host + ":" + std::to_string(params.port);
    }
    const char * cache_dir = nullptr;
    std::string cache_dir_str;
    if (params.use_cache) {
        const std::filesystem::path cache_dir_path = fs_get_cache_directory() / "rpc";
        std::error_code ec;
        common_create_directories(cache_dir_path, ec);
        if (ec) {
            fprintf(stderr, "Failed to create cache directory: %s\n", fs_path_to_utf8(cache_dir_path).c_str());
            return 1;
        }
        cache_dir_str = fs_path_to_utf8(cache_dir_path);
        cache_dir = cache_dir_str.c_str();
    }

    ggml_backend_reg_t reg = ggml_backend_reg_by_name("RPC");
    if (!reg) {
        fprintf(stderr, "Failed to find RPC backend\n");
        return 1;
    }

    auto start_server_fn = (decltype(ggml_backend_rpc_start_server)*) ggml_backend_reg_get_proc_address(reg, "ggml_backend_rpc_start_server");
    if (!start_server_fn) {
        fprintf(stderr, "Failed to obtain RPC backend start server function\n");
        return 1;
    }

    start_server_fn(endpoint.c_str(), cache_dir, params.n_threads, devices.size(), devices.data());
    return 0;
}
