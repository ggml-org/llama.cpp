#include "session-file.h"

#if __has_include("log.h")
#include "log.h"
#else
#include <cstdio>
#define LOG_ERR(fmt, ...) fprintf(stderr, fmt, ##__VA_ARGS__)
#define LOG_WRN(fmt, ...) fprintf(stderr, fmt, ##__VA_ARGS__)
#define LOG_INF(fmt, ...) fprintf(stderr, fmt, ##__VA_ARGS__)
#endif

#include <chrono>
#include <ctime>
#include <filesystem>
#include <random>
#include <sstream>

namespace fs = std::filesystem;

static std::string iso_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count() % 1000;
    struct tm tm_buf;
#ifdef _WIN32
    gmtime_s(&tm_buf, &time);
#else
    gmtime_r(&time, &tm_buf);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", &tm_buf);
    char result[40];
    std::snprintf(result, sizeof(result), "%s.%03dZ", buf, (int) ms);
    return result;
}

static std::string generate_uuid() {
    static std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<uint32_t> dist;
    char buf[40];
    uint32_t a = dist(rng), b = dist(rng), c = dist(rng), d = dist(rng);
    std::snprintf(buf, sizeof(buf), "%08x-%04x-%04x-%04x-%04x%08x",
        a, (b >> 16) & 0xffff, ((b & 0x0fff) | 0x4000),
        ((c >> 16) & 0x3fff) | 0x8000, c & 0xffff, d);
    return buf;
}

bool session_file::open(const std::string & path) {
    path_ = path;
    file_.open(path, std::ios::app | std::ios::binary);
    if (!file_.is_open()) {
        LOG_ERR("Failed to open session file: %s\n", path.c_str());
        return false;
    }
    return true;
}

void session_file::reopen() {
    file_.close();
    file_.open(path_, std::ios::trunc | std::ios::binary);
    messages_written_ = 0;
}

void session_file::write_header(const std::string & working_dir) {
    json header;
    header["type"]      = "session";
    header["version"]   = 1;
    header["id"]        = generate_uuid();
    header["timestamp"] = iso_timestamp();
    header["cwd"]       = working_dir;

    write_line(header);
}

void session_file::append_message(const json & msg) {
    json entry;
    entry["type"]      = "message";
    entry["timestamp"] = iso_timestamp();
    entry["message"]       = msg;

    write_line(entry);
    messages_written_++;
}

void session_file::append_compaction(const std::string & summary, size_t kept_from) {
    json entry;
    entry["type"]      = "compaction";
    entry["summary"]   = summary;
    entry["kept_from"] = kept_from;

    write_line(entry);
}

void session_file::write_line(const json & entry) {
    if (!file_.is_open()) {
        return;
    }
    file_ << entry.dump(-1, ' ', false, json::error_handler_t::replace) << '\n';
    file_.flush();
}

std::optional<loaded_session> session_file::load(const std::string & path) {
    if (!fs::exists(path)) {
        return std::nullopt;
    }

    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        LOG_ERR("Failed to open session file for reading: %s\n", path.c_str());
        return std::nullopt;
    }

    std::string line;
    bool header_seen = false;
    std::vector<json> msg_log;
    std::string last_summary;
    size_t last_kept_from = 0;
    bool has_compaction = false;

    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }

        json entry;
        try {
            entry = json::parse(line);
        } catch (const json::parse_error & e) {
            LOG_WRN("Session file: skipping malformed line: %s\n", e.what());
            continue;
        }

        std::string type = entry.value("type", "");

        if (!header_seen) {
            if (type != "session" && type != "header") {
                LOG_ERR("Session file: first line is not a session header\n");
                return std::nullopt;
            }
            int version = entry.value("version", 0);
            if (version != 1) {
                LOG_ERR("Session file: unsupported version %d\n", version);
                return std::nullopt;
            }
            header_seen = true;
            continue;
        }

        if (type == "message") {
            if (entry.contains("message")) {
                msg_log.push_back(entry["message"]);
            }
        } else if (type == "compaction") {
            last_summary   = entry.value("summary", "");
            last_kept_from = entry.value("kept_from", (size_t) 0);
            has_compaction = true;
        }
    }

    if (!header_seen) {
        LOG_ERR("Session file: no header found\n");
        return std::nullopt;
    }

    loaded_session result;

    if (has_compaction && last_kept_from < msg_log.size()) {
        result.messages = json::array();
        for (size_t i = last_kept_from; i < msg_log.size(); i++) {
            result.messages.push_back(msg_log[i]);
        }
        result.previous_summary = last_summary;
    } else {
        result.messages = json::array();
        for (const auto & m : msg_log) {
            result.messages.push_back(m);
        }
    }

    result.total_messages_in_file = msg_log.size();

    LOG_INF("Session loaded: %zu messages%s\n",
            result.messages.size(),
            has_compaction ? " (with compaction)" : "");

    return result;
}

std::string session_file::get_session_dir(const std::string & config_dir, const std::string & working_dir) {
    // Encode cwd into a readable directory name (matching Pi's format)
    std::string encoded = working_dir;

    // Strip leading slash
    while (!encoded.empty() && (encoded[0] == '/' || encoded[0] == '\\')) {
        encoded.erase(0, 1);
    }

    // Replace path separators and colons with dashes
    for (char & c : encoded) {
        if (c == '/' || c == '\\' || c == ':') {
            c = '-';
        }
    }

    return config_dir + "/sessions/" + encoded;
}

std::string session_file::new_session_path(const std::string & session_dir) {
    fs::create_directories(session_dir);

    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    struct tm tm_buf;
#ifdef _WIN32
    localtime_s(&tm_buf, &time);
#else
    localtime_r(&time, &tm_buf);
#endif

    char buf[64];
    std::strftime(buf, sizeof(buf), "%Y%m%d-%H%M%S", &tm_buf);

    // Append milliseconds for uniqueness within the same second
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count() % 1000;
    char ms_buf[8];
    std::snprintf(ms_buf, sizeof(ms_buf), "-%03d", (int) ms);

    return session_dir + "/" + std::string(buf) + ms_buf + ".jsonl";
}

std::string session_file::find_latest_session(const std::string & session_dir) {
    if (!fs::exists(session_dir) || !fs::is_directory(session_dir)) {
        return "";
    }

    std::string latest_path;
    fs::file_time_type latest_time{};

    for (const auto & entry : fs::directory_iterator(session_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".jsonl") {
            auto mtime = entry.last_write_time();
            if (latest_path.empty() || mtime > latest_time) {
                latest_path = entry.path().string();
                latest_time = mtime;
            }
        }
    }

    return latest_path;
}
