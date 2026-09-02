#include "server-connect.h"

#include "server-common.h"
#include "subproc.h"

#include <cctype>
#include <chrono>
#include <exception>
#include <filesystem>
#include <random>
#include <string_view>
#include <system_error>
#include <thread>
#include <vector>

// share code = room code + pass code, must match llama-connect and the Web UI
// ref: https://github.com/ggml-org/llama-connect/blob/master/src/protocol.rs
static constexpr size_t CONNECT_ROOM_CODE_LEN = 8;
static constexpr size_t CONNECT_PASS_CODE_LEN = 32;

static const std::string CONNECT_CODE_CHARS = "ABCDEFGHJKMNPQRSTUVWXYZabcdefghjkmnpqrstuvwxyz23456789";

#if defined(_WIN32)
static const std::string CONNECT_EXE_NAME = "llama-connect.exe";
static constexpr char    PATH_SEPARATOR   = ';';
#else
static const std::string CONNECT_EXE_NAME = "llama-connect";
static constexpr char    PATH_SEPARATOR   = ':';
#endif

// how long to wait for the child to notice the closed stdin before killing it
static constexpr int CONNECT_STOP_TIMEOUT_MS = 3000;

// the pass code guards the tunnel, so do not use random_string(): its mt19937 is predictable
static std::string gen_share_code() {
    std::random_device rd;
    std::uniform_int_distribution<size_t> dist(0, CONNECT_CODE_CHARS.size() - 1);

    std::string code(CONNECT_ROOM_CODE_LEN + CONNECT_PASS_CODE_LEN, ' ');
    for (char & c : code) {
        c = CONNECT_CODE_CHARS[dist(rd)];
    }

    return code;
}

static std::string format_share_code(const std::string & code) {
    std::string out;
    for (size_t i = 0; i < code.size(); i += 8) {
        if (i > 0) {
            out += ' ';
        }
        out += code.substr(i, 8);
    }
    return out;
}

// whitespace is tolerated, the Web UI shows the code in blocks and users paste it back
// returns an empty string if the key is not a valid share code
static std::string normalize_share_code(const std::string & key) {
    std::string code;
    for (char c : key) {
        if (!std::isspace((unsigned char) c)) {
            code += c;
        }
    }

    if (code.size() != CONNECT_ROOM_CODE_LEN + CONNECT_PASS_CODE_LEN) {
        return "";
    }

    if (code.find_first_not_of(CONNECT_CODE_CHARS) != std::string::npos) {
        return "";
    }

    return code;
}

// llama-connect logs as "[LEVEL] text", forward at the same level so our verbosity filter applies
static void forward_child_log(const std::string & line) {
    static const std::pair<std::string_view, ggml_log_level> tags[] = {
        { "[ERROR] ", GGML_LOG_LEVEL_ERROR },
        { "[WARN] ",  GGML_LOG_LEVEL_WARN  },
        { "[INFO] ",  GGML_LOG_LEVEL_INFO  },
        { "[DEBUG] ", GGML_LOG_LEVEL_DEBUG },
        { "[TRACE] ", GGML_LOG_LEVEL_DEBUG },
    };

    // untagged lines are the startup banner, show them as info
    ggml_log_level level = GGML_LOG_LEVEL_INFO;
    const char * text = line.c_str();

    for (const auto & [tag, tag_level] : tags) {
        if (string_starts_with(line, tag)) {
            level = tag_level;
            text += tag.size();
            break;
        }
    }

    switch (level) {
        case GGML_LOG_LEVEL_ERROR: LOG_ERR("connect | %s", text); break;
        case GGML_LOG_LEVEL_WARN:  LOG_WRN("connect | %s", text); break;
        case GGML_LOG_LEVEL_DEBUG: LOG_DBG("connect | %s", text); break;
        default:                   LOG_INF("connect | %s", text); break;
    }
}

static bool path_is_file(const std::filesystem::path & p) {
    std::error_code ec;
    return std::filesystem::is_regular_file(p, ec);
}

std::string server_connect::find_binary() {
    // prefer the copy shipped next to llama-server over an unrelated one in PATH
    try {
        auto sibling = get_server_exec_path().parent_path() / CONNECT_EXE_NAME;
        if (path_is_file(sibling)) {
            return sibling.string();
        }
    } catch (const std::exception & e) {
        SRV_WRN("could not resolve the llama-server path (%s), looking for llama-connect in PATH only\n", e.what());
    }

    const std::string path_env = common_get_env("PATH");
    size_t start = 0;
    while (start <= path_env.size()) {
        size_t end = path_env.find(PATH_SEPARATOR, start);
        if (end == std::string::npos) {
            end = path_env.size();
        }
        const std::string dir = path_env.substr(start, end - start);
        if (!dir.empty()) {
            auto candidate = std::filesystem::path(dir) / CONNECT_EXE_NAME;
            if (path_is_file(candidate)) {
                return candidate.string();
            }
        }
        start = end + 1;
    }

    return "";
}

std::string server_connect::unavailable_reason(const common_params & params) {
    if (!common_subproc::is_supported()) {
        return "this build has subprocess support disabled, rebuild with -DLLAMA_SUBPROCESS=ON";
    }

    if (!params.server_connect_code.empty() && normalize_share_code(params.server_connect_code).empty()) {
        return "--connect-code must be " + std::to_string(CONNECT_ROOM_CODE_LEN + CONNECT_PASS_CODE_LEN)
             + " characters from '" + CONNECT_CODE_CHARS + "'";
    }

    if (find_binary().empty()) {
        return "could not find '" + CONNECT_EXE_NAME + "' next to llama-server or in PATH.\n"
               "    it is a separate binary: download it from https://github.com/ggml-org/llama-connect/releases\n"
               "    or build llama.cpp with -DLLAMA_CONNECT=ON to have it fetched automatically";
    }

    return "";
}

bool server_connect::start(const common_params & params) {
    const std::string bin = find_binary();
    if (bin.empty()) {
        SRV_ERR("%s", "llama-connect binary not found\n");
        return false;
    }

    // already validated by unavailable_reason()
    const std::string code = params.server_connect_code.empty()
                           ? gen_share_code()
                           : normalize_share_code(params.server_connect_code);

    // always loopback, params.hostname may be 0.0.0.0 or a unix socket which the child cannot dial
    const std::vector<std::string> args = {
        bin,
        "--host", "127.0.0.1",
        "--port", std::to_string(params.port),
        "--code", code,
        // the kernel closes our end of this pipe even if we are killed without cleanup,
        // so the child cannot outlive us
        "--exit-on-stdin-eof",
    };

    proc = std::make_unique<common_subproc>();

    const int options = subprocess_option_no_window
                      | subprocess_option_combined_stdout_stderr
                      | subprocess_option_inherit_environment;

    if (!proc->create(args, options)) {
        SRV_ERR("failed to spawn '%s'\n", bin.c_str());
        proc.reset();
        return false;
    }

    log_thread = std::thread([this]() {
        FILE * out = proc->stdout_file();
        if (out == nullptr) {
            SRV_ERR("%s", "failed to get stdout of the llama-connect process\n");
            return;
        }
        std::vector<char> buf(4096);
        while (fgets(buf.data(), (int) buf.size(), out) != nullptr) {
            forward_child_log(buf.data());
        }
        // EOF means the child is gone
        if (!stopping.load(std::memory_order_acquire)) {
            SRV_ERR("%s", "llama-connect exited on its own, remote access is no longer available\n");
        }
    });

    SRV_INF("%s", "-----------------\n");
    SRV_INF("%s", "remote access is enabled via llama-connect\n");
    SRV_INF("share code (enter it in the Web UI under Settings -> Remote Access): %s\n",
            format_share_code(code).c_str());
    SRV_WRN("%s", "anyone with this code can use this server, do not share it publicly\n");
    SRV_INF("%s", "-----------------\n");

    return true;
}

void server_connect::stop() {
    if (!proc) {
        return;
    }

    SRV_INF("%s", "stopping llama-connect...\n");

    stopping.store(true, std::memory_order_release);

    proc->close_stdin();

    for (int elapsed = 0; elapsed < CONNECT_STOP_TIMEOUT_MS && proc->alive(); elapsed += 100) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // no-op if the child already exited; also unblocks the log thread by closing its stdout
    proc->terminate();

    if (log_thread.joinable()) {
        try {
            log_thread.join();
        } catch (const std::system_error & e) {
            // ~thread() on a still-joinable thread calls std::terminate, detach instead
            SRV_ERR("failed to join the llama-connect log thread: %s\n", e.what());
            log_thread.detach();
        }
    }

    proc->join(); // reap the zombie
    proc.reset();
}

server_connect::server_connect() = default;

server_connect::~server_connect() {
    try {
        stop();
    } catch (const std::exception & e) {
        SRV_ERR("failed to stop llama-connect: %s\n", e.what());
    } catch (...) {
        SRV_ERR("%s", "failed to stop llama-connect\n");
    }
}
