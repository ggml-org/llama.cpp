#include "server-mcp.h"

#include "subproc.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <functional>
#include <sstream>
#include <thread>

#if defined(_WIN32)
#  include <io.h>
#  include <windows.h>
#else
#  include <errno.h>
#  include <fcntl.h>
#  include <poll.h>
#  include <unistd.h>
extern char ** environ;
#endif

// read NDJSON lines from a child pipe, calling on_line per line until `running` clears, EOF/error, or on_line returns false.
// polled, not blocking: a grandchild can inherit the pipe's write end and hold it open (terminate() kills only the direct child), so a blocking read would hang teardown on an EOF that never comes.
static void mcp_pump_ndjson(FILE * f, std::atomic<bool> & running,
                            const std::function<bool(std::string &&)> & on_line) {
    if (!f) {
        return;
    }
    const int    poll_ms  = 50;
    const size_t max_line = 8 * 1024 * 1024; // drop any single NDJSON line larger than this, so a child that never emits '\n' can't grow buf without bound
#if defined(_WIN32)
    HANDLE h = (HANDLE) _get_osfhandle(_fileno(f));
#else
    int fd = fileno(f);
    int fl = fcntl(fd, F_GETFL, 0);
    if (fl >= 0) {
        fcntl(fd, F_SETFL, fl | O_NONBLOCK);
    }
#endif
    std::string buf;
    bool        skipping = false; // discarding an over-long line until its terminating newline
    char        chunk[4096];
    while (running.load()) {
        size_t n = 0;
#if defined(_WIN32)
        DWORD avail = 0;
        if (!PeekNamedPipe(h, NULL, 0, NULL, &avail, NULL)) {
            break; // pipe broken / child gone
        }
        if (avail == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(poll_ms));
            continue;
        }
        DWORD to_read = avail < (DWORD) sizeof(chunk) ? avail : (DWORD) sizeof(chunk);
        DWORD got     = 0;
        if (!ReadFile(h, chunk, to_read, &got, NULL) || got == 0) {
            break;
        }
        n = (size_t) got;
#else
        struct pollfd pfd;
        pfd.fd      = fd;
        pfd.events  = POLLIN;
        pfd.revents = 0;
        int pr      = poll(&pfd, 1, poll_ms);
        if (pr < 0) {
            if (errno == EINTR) {
                continue;
            }
            break;
        }
        if (pr == 0) {
            continue; // timeout -> re-check running
        }
        if (pfd.revents & (POLLERR | POLLNVAL)) {
            break;
        }
        ssize_t r = read(fd, chunk, sizeof(chunk));
        if (r < 0) {
            if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) {
                continue;
            }
            break;
        }
        if (r == 0) {
            break; // EOF: child (and any pipe writers) closed the stream
        }
        n = (size_t) r;
#endif
        buf.append(chunk, n);

        // resync after an over-long, unterminated line: discard bytes until the next newline
        if (skipping) {
            size_t nl = buf.find('\n');
            if (nl == std::string::npos) {
                if (buf.size() > max_line) {
                    buf.clear(); // stay bounded while waiting for a terminator
                }
                continue;
            }
            buf.erase(0, nl + 1);
            skipping = false;
        }

        size_t pos;
        while ((pos = buf.find('\n')) != std::string::npos) {
            std::string line = buf.substr(0, pos);
            buf.erase(0, pos + 1);
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            if (line.empty()) {
                continue;
            }
            if (!on_line(std::move(line))) {
                return;
            }
        }

        // a partial line already larger than the cap and still no newline: drop it to avoid unbounded growth
        if (buf.size() > max_line) {
            SRV_WRN("MCP: dropping oversized line (> %zu bytes) from child pipe\n", max_line);
            buf.clear();
            skipping = true;
        }
    }
}

//
// server_mcp_server_config
//

std::vector<server_mcp_server_config> server_mcp_server_config::parse_from_json(const std::string & json_str) {
    return parse_cursor_format(json::parse(json_str));
}

std::vector<server_mcp_server_config> server_mcp_server_config::parse_cursor_format(const json & j) {
    std::vector<server_mcp_server_config> result;

    if (!j.contains("mcpServers") || !j.at("mcpServers").is_object()) {
        return result;
    }

    for (const auto & [name, cfg] : j.at("mcpServers").items()) {
        server_mcp_server_config sc;
        sc.name = name;
        sc.command = cfg.value("command", std::string());
        sc.cwd = cfg.value("cwd", std::string());
        sc.timeout_ms = cfg.value("timeout_ms", sc.timeout_ms);

        if (cfg.contains("args") && cfg.at("args").is_array()) {
            for (const auto & a : cfg.at("args")) {
                sc.args.push_back(a.get<std::string>());
            }
        }
        if (cfg.contains("env") && cfg.at("env").is_object()) {
            for (const auto & [k, v] : cfg.at("env").items()) {
                sc.env[k] = v.get<std::string>();
            }
        }

        if (sc.command.empty()) {
            SRV_WRN("MCP server '%s' has no command, skipping\n", name.c_str());
            continue;
        }
        result.push_back(std::move(sc));
    }

    return result;
}


//
// server_mcp_transport
//

static constexpr const char * MCP_PROTOCOL_VERSION = "2024-11-05";

static std::string rpc_error_message(const json & resp) {
    if (resp.contains("error")) {
        const json & e = resp.at("error");
        if (e.is_object()) {
            return e.value("message", "unknown error");
        }
        if (e.is_string()) {
            return e.get<std::string>();
        }
    }
    return "unknown error";
}

// normalize an MCP tools/call result to the /tools contract (see README-dev.md):
// concat text parts of result.content[], and surface an isError result
static json mcp_result_to_response(const json & result) {
    std::string text;
    if (result.contains("content") && result.at("content").is_array()) {
        for (const auto & part : result.at("content")) {
            if (part.is_object() && part.value("type", "") == "text") {
                if (!text.empty()) {
                    text += "\n";
                }
                text += part.value("text", "");
            }
        }
    }
    if (result.is_object() && result.value("isError", false)) {
        return {{"error", text.empty() ? "MCP tool returned an error" : text}};
    }
    return {{"plain_text_response", text}};
}

json server_mcp_transport::send_rpc(const json & request, const std::function<bool()> & should_stop) {
    if (!to_server.write(request.dump())) {
        return {{"error", {{"code", -32603}, {"message", "transport closed"}}}};
    }

    const bool has_id = request.contains("id");
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    auto stop = [&]() {
        return (should_stop && should_stop()) || std::chrono::steady_clock::now() >= deadline;
    };

    std::string frame;
    while (from_server.read(frame, stop, false)) {
        json reply;
        try {
            reply = json::parse(frame);
        } catch (...) {
            if (std::chrono::steady_clock::now() >= deadline) {
                break;
            }
            continue; // skip malformed frame
        }
        // no id: a notification. mismatched id: a stale reply from a timed-out request (ids are monotonic, never a future one)
        if (!has_id || (reply.contains("id") && reply.at("id") == request.at("id"))) {
            return reply;
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            break; // a flood of notifications must not outrun the deadline
        }
    }

    if (should_stop && should_stop()) {
        return {{"error", {{"code", -32603}, {"message", "cancelled"}}}};
    }
    if (std::chrono::steady_clock::now() >= deadline) {
        return {{"error", {{"code", -32603}, {"message", "request timed out"}}}};
    }
    return {{"error", {{"code", -32603}, {"message", "transport closed"}}}};
}

bool server_mcp_transport::ensure_init(const std::function<bool()> & should_stop) {
    if (initialized) {
        return true;
    }

    json init_req = {
        {"jsonrpc", "2.0"},
        {"id", next_id++},
        {"method", "initialize"},
        {"params", {
            {"protocolVersion", MCP_PROTOCOL_VERSION},
            {"capabilities", json::object()},
            {"clientInfo", {{"name", "llama.cpp"}, {"version", "1.0"}}},
        }},
    };
    json resp = send_rpc(init_req, should_stop);
    if (!resp.contains("result")) {
        last_error = "initialize failed: " + rpc_error_message(resp);
        return false;
    }

    // notifications/initialized: no id, no reply expected
    json notif = {{"jsonrpc", "2.0"}, {"method", "notifications/initialized"}};
    to_server.write(notif.dump());

    initialized = true;
    return true;
}

std::vector<server_mcp_tool_def> server_mcp_transport::list_tools(const std::function<bool()> & should_stop) {
    std::lock_guard<std::mutex> lock(rpc_mutex);
    if (!ensure_init(should_stop)) {
        return {};
    }
    if (!tools.empty()) {
        return tools;
    }

    json req = {{"jsonrpc", "2.0"}, {"id", next_id++}, {"method", "tools/list"}};
    json resp = send_rpc(req, should_stop);
    if (!resp.contains("result")) {
        last_error = "tools/list failed: " + rpc_error_message(resp);
        return {};
    }

    const json & result = resp.at("result");
    if (result.contains("tools") && result.at("tools").is_array()) {
        for (const auto & t : result.at("tools")) {
            server_mcp_tool_def def;
            def.server_name = name;
            def.name = t.value("name", "");
            def.description = t.value("description", "");
            if (t.contains("inputSchema")) {
                def.input_schema = t.at("inputSchema");
            }
            tools.push_back(std::move(def));
        }
    }
    return tools;
}

json server_mcp_transport::call_tool(const std::string & tool_name,
                                     const json & arguments,
                                     const std::function<bool()> & should_stop) {
    std::lock_guard<std::mutex> lock(rpc_mutex);
    if (!ensure_init(should_stop)) {
        return {{"error", last_error}};
    }

    json req = {
        {"jsonrpc", "2.0"},
        {"id", next_id++},
        {"method", "tools/call"},
        {"params", {{"name", tool_name}, {"arguments", arguments}}},
    };
    json resp = send_rpc(req, should_stop);
    if (resp.contains("error")) {
        return {{"error", rpc_error_message(resp)}};
    }
    if (resp.contains("result")) {
        return mcp_result_to_response(resp.at("result"));
    }
    return {{"error", "invalid response from MCP server"}};
}

//
// server_mcp_stdio
//

struct server_mcp_stdio::process_handle {
    common_subproc sp;
    FILE * in  = nullptr; // child stdin
    FILE * out = nullptr; // child stdout
    FILE * err = nullptr; // child stderr
};

#if defined(_WIN32)
// config strings are UTF-8 (from JSON) and subprocess.h converts them with CP_UTF8, so inputs must be UTF-8, not the active code page
static std::wstring windows_utf8_to_wide(const std::string & s) {
    if (s.empty()) {
        return std::wstring();
    }
    int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int) s.size(), NULL, 0);
    if (n <= 0) {
        return std::wstring();
    }
    std::wstring w((size_t) n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), (int) s.size(), &w[0], n);
    return w;
}

static std::string windows_wide_to_utf8(const wchar_t * s, int len /* -1 for NUL-terminated */) {
    int n = WideCharToMultiByte(CP_UTF8, 0, s, len, NULL, 0, NULL, NULL);
    if (n <= 0) {
        return std::string();
    }
    std::string out((size_t) n, '\0');
    WideCharToMultiByte(CP_UTF8, 0, s, len, &out[0], n, NULL, NULL);
    if (len == -1 && !out.empty() && out.back() == '\0') {
        out.pop_back(); // drop the terminator WideCharToMultiByte counts for -1
    }
    return out;
}
#endif

static std::string mcp_resolve_command(const std::string & command) {
#if defined(_WIN32)
    // For Windows: make sure we handle ".exe" correctly, as well as UTF-8
    std::wstring wcmd = windows_utf8_to_wide(command);
    wchar_t      buf[MAX_PATH * 4];
    const DWORD  cap = (DWORD) (sizeof(buf) / sizeof(buf[0]));

    auto search = [&](const wchar_t * ext) -> std::string {
        DWORD n = SearchPathW(NULL, wcmd.c_str(), ext, cap, buf, NULL);
        return (n > 0 && n < cap) ? windows_wide_to_utf8(buf, (int) n) : std::string();
    };

    std::string found = search(NULL); // exact path / already-extensioned / .exe on PATH
    if (!found.empty()) {
        return found;
    }

    std::wstring pathext;
    DWORD        need = GetEnvironmentVariableW(L"PATHEXT", NULL, 0);
    if (need > 0) {
        pathext.resize(need);
        DWORD got = GetEnvironmentVariableW(L"PATHEXT", &pathext[0], need);
        pathext.resize(got);
    }
    if (pathext.empty()) {
        pathext = L".COM;.EXE;.BAT;.CMD";
    }
    for (size_t start = 0; start <= pathext.size();) {
        size_t       sep = pathext.find(L';', start);
        std::wstring ext = pathext.substr(start, sep == std::wstring::npos ? std::wstring::npos : sep - start);
        if (!ext.empty()) {
            found = search(ext.c_str());
            if (!found.empty()) {
                return found;
            }
        }
        if (sep == std::wstring::npos) {
            break;
        }
        start = sep + 1;
    }
    return command; // give up and let subprocess.h report the spawn error
#else
    return command;
#endif // _WIN32
}

static std::vector<std::string> mcp_parent_env() {
    std::vector<std::string> env;
#if defined(_WIN32)
    LPWCH block = GetEnvironmentStringsW();
    if (block) {
        for (LPWCH e = block; *e; e += wcslen(e) + 1) {
            env.emplace_back(windows_wide_to_utf8(e, -1));
        }
        FreeEnvironmentStringsW(block);
    }
#else
    if (environ) {
        for (char ** e = environ; *e; ++e) {
            env.emplace_back(*e);
        }
    }
#endif
    return env;
}

// parent env with the config overrides applied, in "KEY=VALUE" form
static std::vector<std::string> mcp_build_env(const std::map<std::string, std::string> & overrides) {
    std::vector<std::string> env;
    for (auto & e : mcp_parent_env()) {
        size_t eq = e.find('=');
        std::string key = eq == std::string::npos ? e : e.substr(0, eq);
        if (overrides.find(key) == overrides.end()) {
            env.push_back(e);
        }
    }
    for (auto & [k, v] : overrides) {
        env.push_back(k + "=" + v);
    }
    return env;
}

server_mcp_stdio::server_mcp_stdio(const server_mcp_server_config & config) : config(config) {
    name = config.name;
    timeout_ms = config.timeout_ms;
    // bound the reply queue: send_rpc only drains during a call, so unsolicited notifications would otherwise grow it without limit
    from_server.max_size = 65536;
}

server_mcp_stdio::~server_mcp_stdio() {
    join_pumps();
}

bool server_mcp_stdio::start() {
    std::vector<std::string> argv_s;
    argv_s.push_back(mcp_resolve_command(config.command));
    argv_s.insert(argv_s.end(), config.args.begin(), config.args.end());

    int options = subprocess_option_no_window | subprocess_option_search_user_path;
    std::vector<std::string> envp_s;
    if (config.env.empty()) {
        options |= subprocess_option_inherit_environment;
    } else {
        envp_s = mcp_build_env(config.env);
    }

    auto handle = std::make_unique<process_handle>();
    bool ok = handle->sp.create(argv_s, options, envp_s, config.cwd.empty() ? nullptr : config.cwd.c_str());
    if (!ok) {
        SRV_WRN("MCP '%s': failed to spawn '%s'\n", config.name.c_str(), config.command.c_str());
        return false;
    }
    handle->in  = handle->sp.stdin_file();
    handle->out = handle->sp.stdout_file();
    handle->err = handle->sp.stderr_file();

    proc = std::move(handle);
    running.store(true);
    reader = std::thread([this] { reader_loop(); });
    writer = std::thread([this] { writer_loop(); });
    errlog = std::thread([this] { errlog_loop(); });
    return true;
}

void server_mcp_stdio::close() {
    join_pumps();
}

bool server_mcp_stdio::is_alive() const {
    return running.load();
}

std::string server_mcp_stdio::diagnostics() {
    std::string out;
    {
        std::lock_guard<std::mutex> lock(rpc_mutex); // last_error is written by send_rpc's callers
        out = last_error;
    }
    std::lock_guard<std::mutex> lk(err_mu);
    if (!err_tail.empty()) {
        if (!out.empty()) {
            out += "; ";
        }
        out += "last stderr: " + err_tail;
    }
    return out;
}

void server_mcp_stdio::reader_loop() {
    mcp_pump_ndjson(proc->out, running, [this](std::string && line) {
        return from_server.write(std::move(line)); // false => consumer gone, stop
    });
    running.store(false);
    to_server.close_write();   // stop the writer
    from_server.close_write(); // EOF to any waiting caller
}

// write all of `data` to child stdin, non-blocking and polled so teardown never hangs (a grandchild can hold the read end of a full pipe open). returns false on error/close/shutdown.
static bool mcp_write_all(FILE * f, const std::string & data, std::atomic<bool> & running) {
    if (!f) {
        return false;
    }
    size_t total = 0;
#if defined(_WIN32)
    HANDLE h      = (HANDLE) _get_osfhandle(_fileno(f));
    DWORD  nowait = PIPE_NOWAIT;
    SetNamedPipeHandleState(h, &nowait, NULL, NULL);
    while (total < data.size() && running.load()) {
        DWORD written = 0;
        BOOL  ok      = WriteFile(h, data.data() + total, (DWORD) (data.size() - total), &written, NULL);
        if (ok && written > 0) {
            total += written;
            continue;
        }
        if (!ok) {
            DWORD err = GetLastError();
            if (err != ERROR_NO_DATA && err != ERROR_PIPE_BUSY) {
                return false;
            }
        }
        // backpressure (pipe full) is rare for small JSON-RPC frames; sleep rather than spin.
        // no writable-wait exists for a PIPE_NOWAIT anonymous pipe, so this polls like the POSIX poll() path.
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
#else
    int fd = fileno(f);
    int fl = fcntl(fd, F_GETFL, 0);
    if (fl >= 0) {
        fcntl(fd, F_SETFL, fl | O_NONBLOCK);
    }
    while (total < data.size() && running.load()) {
        ssize_t n = write(fd, data.data() + total, data.size() - total);
        if (n > 0) {
            total += (size_t) n;
            continue;
        }
        if (n == 0) {
            return false;
        }
        if (errno == EINTR) {
            continue;
        }
        if (errno != EAGAIN && errno != EWOULDBLOCK) {
            return false;
        }
        struct pollfd pfd;
        pfd.fd      = fd;
        pfd.events  = POLLOUT;
        pfd.revents = 0;
        int pr      = poll(&pfd, 1, 50);
        if (pr < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        if (pfd.revents & (POLLERR | POLLNVAL | POLLHUP)) {
            return false;
        }
    }
#endif
    return total == data.size();
}

void server_mcp_stdio::writer_loop() {
    auto should_stop = [this] { return !running.load(); };
    std::string msg;
    while (to_server.read(msg, should_stop)) {
        msg.push_back('\n');
        if (!mcp_write_all(proc->in, msg, running)) {
            break; // child gone or shutting down
        }
    }
    running.store(false);
    to_server.close_read();    // fail fast on any further send_rpc write
    from_server.close_write(); // wake any caller waiting for a reply
}

void server_mcp_stdio::errlog_loop() {
    static constexpr size_t ERR_TAIL_MAX = 4096;
    // drain stderr (an undrained pipe blocks the child):
    // log it, and keep a bounded tail for reporting when the server dies
    mcp_pump_ndjson(proc->err, running, [this](std::string && line) {
        SRV_DBG("MCP '%s' stderr: %s\n", name.c_str(), line.c_str());
        std::lock_guard<std::mutex> lk(err_mu);
        err_tail += line;
        err_tail += '\n';
        if (err_tail.size() > ERR_TAIL_MAX) {
            err_tail.erase(0, err_tail.size() - ERR_TAIL_MAX);
        }
        return true;
    });
}

void server_mcp_stdio::join_pumps() {
    if (!proc) {
        return;
    }
    running.store(false);
    to_server.close_write();   // wake the writer if it waits for a message
    from_server.close_write(); // wake any caller waiting for a reply

    proc->sp.terminate(); // child death unblocks the blocked fread/fwrite

    if (writer.joinable()) writer.join();
    if (reader.joinable()) reader.join();
    if (errlog.joinable()) errlog.join();

    proc->sp.join(); // reap the child: never waiting would leave the pid a zombie for the process lifetime
    proc.reset();
}


//
// server_mcp
//

static constexpr int MCP_COOLDOWN_SECONDS = 5;
static constexpr int MCP_WARMUP_TIMEOUT_SECONDS = 10; // cap per-server tool discovery at startup

server_mcp::~server_mcp() {
    shutdown();

    std::vector<std::shared_ptr<server_mcp_transport>> to_close;
    {
        std::lock_guard<std::mutex> lock(mutex);
        for (auto & [name, t] : transports) {
            to_close.push_back(std::move(t));
        }
        transports.clear();
    }
    for (auto & t : to_close) {
        t->close();
    }
}

std::shared_ptr<server_mcp_transport> server_mcp::create_transport(const server_mcp_server_config & cfg) {
    return std::make_shared<server_mcp_stdio>(cfg);
}

void server_mcp::shutdown() {
    stopping.store(true);
}

const server_mcp_server_config * server_mcp::find_config(const std::string & name) const {
    for (const auto & c : configs) {
        if (c.name == name) {
            return &c;
        }
    }
    return nullptr;
}

void server_mcp::start(const common_params & params) {
    auto append = [this](const std::string & json_str) {
        try {
            auto parsed = server_mcp_server_config::parse_from_json(json_str);
            if (parsed.empty()) {
                SRV_WRN("%s", "MCP config: no servers found in JSON\n");
            }
            for (auto & p : parsed) {
                // names must be unique across both config sources: get_or_create / find_config key on the name
                if (find_config(p.name)) {
                    SRV_WRN("MCP config: duplicate server name '%s', skipping\n", p.name.c_str());
                    continue;
                }
                configs.push_back(std::move(p));
            }
        } catch (const std::exception & e) {
            throw std::runtime_error(std::string("failed to parse MCP config JSON: ") + e.what());
        }
    };
    if (!params.mcp_servers_config.empty()) {
        std::ifstream f = fs_open_ifstream(params.mcp_servers_config, std::ios::in);
        if (!f) {
            throw std::runtime_error("failed to open MCP config file: " + params.mcp_servers_config);
        }
        std::stringstream ss;
        ss << f.rdbuf();
        append(ss.str());
    }
    if (!params.mcp_servers_json.empty()) {
        append(params.mcp_servers_json);
    }

    if (configs.empty()) {
        return;
    }

    std::vector<server_mcp_tool_def> discovered;
    for (const auto & cfg : configs) {
        auto t = create_transport(cfg);
        if (!t->start()) {
            SRV_WRN("MCP warmup: failed to spawn '%s': %s\n", cfg.name.c_str(), t->diagnostics().c_str());
            continue;
        }
        // bound warmup per server so an unresponsive one can't stall startup for the full per-call timeout
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(MCP_WARMUP_TIMEOUT_SECONDS);
        auto should_stop = [this, deadline]() {
            return stopping.load() || std::chrono::steady_clock::now() >= deadline;
        };
        auto tools = t->list_tools(should_stop);
        SRV_INF("MCP warmup: '%s' discovered %zu tools\n", cfg.name.c_str(), tools.size());
        discovered.insert(discovered.end(), tools.begin(), tools.end());
        t->close();
    }

    std::lock_guard<std::mutex> lock(mutex);
    registry.swap(discovered);
}

std::vector<server_mcp_tool_def> server_mcp::list_tools() const {
    std::lock_guard<std::mutex> lock(mutex);
    return registry;
}

json server_mcp::call_tool(const std::string & server_name,
                           const std::string & tool_name,
                           const json & arguments,
                           const std::function<bool()> & should_stop) {
    auto transport = get_or_create(server_name);
    if (!transport) {
        return {{"error", "MCP server unavailable: " + server_name}};
    }

    auto stop = [this, &should_stop]() {
        return stopping.load() || (should_stop && should_stop());
    };
    return transport->call_tool(tool_name, arguments, stop);
}

std::shared_ptr<server_mcp_transport> server_mcp::get_or_create(const std::string & name) {
    std::vector<std::shared_ptr<server_mcp_transport>> to_close; // closed after unlock
    std::shared_ptr<server_mcp_transport> result;

    {
        std::lock_guard<std::mutex> lock(mutex);
        if (stopping.load()) {
            return nullptr;
        }

        auto now = std::chrono::steady_clock::now();
        auto dead_it = dead_servers.find(name);
        if (dead_it != dead_servers.end()) {
            if (now < dead_it->second) {
                return nullptr;
            }
            dead_servers.erase(dead_it);
        }

        auto it = transports.find(name);
        if (it != transports.end()) {
            if (it->second->is_alive()) {
                return it->second;
            }
            SRV_WRN("MCP '%s' is no longer alive: %s\n", name.c_str(), it->second->diagnostics().c_str());
            to_close.push_back(std::move(it->second));
            transports.erase(it);
        }

        const server_mcp_server_config * cfg = find_config(name);
        if (cfg) {
            auto fresh = create_transport(*cfg);
            if (fresh->start() && fresh->is_alive()) {
                transports[name] = fresh;
                result = fresh;
            } else {
                SRV_WRN("MCP '%s': failed to start: %s\n", name.c_str(), fresh->diagnostics().c_str());
                to_close.push_back(std::move(fresh));
                dead_servers[name] = now + std::chrono::seconds(MCP_COOLDOWN_SECONDS);
            }
        }
    }

    for (auto & t : to_close) {
        t->close(); // blocking call, no leaks
    }

    return result;
}

