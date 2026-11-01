#include "server-service.h"
#include "server-common.h"
#include "server-models.h"

#include "build-info.h"
#include "common.h"
#include "log.h"

#include <cpp-httplib/httplib.h>

#include <cerrno>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <thread>

#if defined(_WIN32)
#include <windows.h>
#else
#include <signal.h>
#include <unistd.h>
#endif

std::filesystem::path server_service_dir() {
    return fs_get_config_directory();
}

std::filesystem::path server_service_lock_path() {
    return server_service_dir() / "service.json";
}

std::filesystem::path server_service_default_sock() {
    return server_service_dir() / "service.sock";
}

std::filesystem::path server_service_log_path() {
    return server_service_dir() / "service.log";
}

int server_service_current_pid() {
#if defined(_WIN32)
    return (int) GetCurrentProcessId();
#else
    return (int) getpid();
#endif
}

bool server_service_pid_alive(int pid) {
    if (pid <= 0) {
        return false;
    }
#if defined(_WIN32)
    HANDLE h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, (DWORD) pid);
    if (h == nullptr) {
        return false;
    }
    DWORD code = 0;
    const bool alive = GetExitCodeProcess(h, &code) && code == STILL_ACTIVE;
    CloseHandle(h);
    return alive;
#else
    return kill((pid_t) pid, 0) == 0 || errno == EPERM;
#endif
}

void server_service_pid_kill(int pid) {
    if (pid <= 0) {
        return;
    }
#if defined(_WIN32)
    HANDLE h = OpenProcess(PROCESS_TERMINATE, FALSE, (DWORD) pid);
    if (h != nullptr) {
        TerminateProcess(h, 1);
        CloseHandle(h);
    }
#else
    kill((pid_t) pid, SIGKILL);
#endif
}

std::optional<server_service_lock> server_service_lock::read() {
    std::ifstream in(server_service_lock_path());
    if (!in) {
        return std::nullopt;
    }
    std::string raw((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    try {
        json data = json::parse(raw);
        server_service_lock lock;
        lock.pid   = json_value(data, "pid",   0);
        lock.host  = json_value(data, "host",  std::string());
        lock.port  = json_value(data, "port",  0);
        lock.sock  = json_value(data, "sock",  std::string());
        lock.build = json_value(data, "build", std::string());
        lock.args  = json_value(data, "args",  std::vector<std::string>());
        return lock;
    } catch (const std::exception & e) {
        LOG_WRN("failed to read the service lock file: %s\n", e.what());
        return std::nullopt;
    }
}

void server_service_lock::write() const {
    std::error_code ec;
    common_create_directories(server_service_dir(), ec);
    json data = {
        {"pid",   pid},
        {"host",  host},
        {"port",  port},
        {"sock",  sock},
        {"build", build},
        {"args",  args},
    };
    fs_write_atomic(server_service_lock_path(), data.dump(2));
}

void server_service_lock::remove(bool force) {
    if (!force) {
        auto lock = read();
        if (!lock.has_value() || lock->pid != server_service_current_pid()) {
            return;
        }
    }
    std::error_code ec;
    std::filesystem::remove(server_service_lock_path(), ec);
}

bool server_service_lock::alive() const {
    return server_service_pid_alive(pid);
}

server_service_response server_service_request(const std::string & sock, const std::string & method, const std::string & path, const std::string & body) {
    httplib::Client cli(sock);
    cli.set_address_family(AF_UNIX);
    cli.set_connection_timeout(2, 0);
    cli.set_read_timeout(30, 0);

    httplib::Result res;
    if (method == "GET") {
        res = cli.Get(path);
    } else if (method == "POST") {
        res = cli.Post(path, body, "application/json");
    } else if (method == "DELETE") {
        res = cli.Delete(path);
    } else {
        throw std::invalid_argument("unsupported method: " + method);
    }
    if (!res) {
        throw std::runtime_error("cannot reach the service at " + sock + ": " + httplib::to_string(res.error()));
    }
    return { res->status, res->body };
}

//
// routes
//

static void res_ok(std::unique_ptr<server_http_res> & res, const json & response_data) {
    res->status = 200;
    res->data = safe_json_to_str(response_data);
}

static void res_err(std::unique_ptr<server_http_res> & res, const json & error_data) {
    res->status = json_value(error_data, "code", 500);
    res->data = safe_json_to_str({{ "error", error_data }});
}

bool server_service::init(const common_params & params,
                          server_models_routes & models_routes,
                          const std::function<void(server_http_context &)> & register_routes,
                          const wrap_t & wrap,
                          std::function<void()> stop_server) {
    sock = params.service_sock;
    host = params.hostnames.empty() ? "" : params.hostnames[0];

    auto lock = server_service_lock::read();
    if (lock.has_value() && lock->alive() && lock->pid != server_service_current_pid()) {
        SRV_ERR("a service is already running with pid %d\n", lock->pid);
        return false;
    }
    std::error_code ec;
    std::filesystem::remove(std::filesystem::u8path(sock), ec); // left by a service that died

    params_service = params;
    params_service.hostnames = { sock };
    params_service.api_keys.clear();
    params_service.ssl_file_key.clear();
    params_service.ssl_file_cert.clear();
    params_service.n_threads_http = 4;
    ctx.emplace();
    if (!ctx->init(params_service)) {
        SRV_ERR("%s", "failed to initialize the service socket\n");
        return false;
    }

    server_models & models = models_routes.models;

    this->get_status = [&models, &models_routes](const server_http_req &) {
        auto res = std::make_unique<server_http_res>();
        json running = json::array();
        for (const auto & meta : models.get_all_meta()) {
            if (meta.is_running()) {
                running.push_back(meta.name);
            }
        }
        res_ok(res, {
            {"pid",     server_service_current_pid()},
            {"host",    models_routes.params.hostnames[0]},
            {"port",    models_routes.params.port},
            {"sock",    models_routes.params.service_sock},
            {"build",   std::string(llama_build_info())},
            {"running", running},
        });
        return res;
    };

    this->post_spawn = [&models](const server_http_req & req) {
        auto res = std::make_unique<server_http_res>();
        json body = json::parse(req.body);

        server_models::spawn_options opts;
        opts.args    = json_value(body, "args",    std::vector<std::string>());
        opts.cwd     = json_value(body, "cwd",     std::string());
        opts.name    = json_value(body, "name",    std::string());
        opts.replace = json_value(body, "replace", false);
        if (body.contains("env")) {
            for (const auto & [key, value] : body.at("env").items()) {
                if (!value.is_string()) {
                    throw std::invalid_argument("env values must be strings");
                }
                opts.env[key] = value.get<std::string>();
            }
        }

        std::string name = models.spawn(opts);
        res_ok(res, {{"model", name}});
        return res;
    };

    this->post_stop = [stop_server](const server_http_req &) {
        auto res = std::make_unique<server_http_res>();
        // the answer goes out before the listener stops
        std::thread([stop_server]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            stop_server();
        }).detach();
        res_ok(res, {{"success", true}});
        return res;
    };

    this->get_logs = [&models, &models_routes](const server_http_req & req) {
        auto res = std::make_unique<server_http_res>();
        std::string name = req.get_param("model");
        if (name.empty()) {
            name = models.get_default_model();
        }
        auto meta = name.empty() ? std::nullopt : models.get_meta(name);
        if (!meta.has_value()) {
            res_err(res, format_error_response("model is not found", ERROR_TYPE_NOT_FOUND));
            return res;
        }
        auto ring = models.get_log_ring(meta->name);
        if (ring == nullptr) {
            res_err(res, format_error_response("model has not been started", ERROR_TYPE_NOT_FOUND));
            return res;
        }
        const bool follow = req.get_param("follow") == "1" || req.get_param("follow") == "true";

        res->status       = 200;
        res->content_type = "text/plain; charset=utf-8";
        auto seq = std::make_shared<uint64_t>(0);
        res->next = [ring, seq, follow, &req, &models_routes](std::string & output) -> bool {
            output.clear();
            while (true) {
                std::vector<std::string> lines;
                const bool more = ring->read(*seq, lines, 200);
                for (const auto & line : lines) {
                    output += line;
                }
                if (!output.empty()) {
                    return true;
                }
                if (!more || !follow || req.should_stop() || models_routes.stopping.load(std::memory_order_relaxed)) {
                    return false;
                }
            }
        };
        return res;
    };

    register_routes(*ctx);
    ctx->get ("/service/status", wrap(get_status));
    ctx->post("/service/spawn",  wrap(post_spawn));
    ctx->post("/service/stop",   wrap(post_stop));
    ctx->get ("/service/logs",   wrap(get_logs));
    return true;
}

bool server_service::start() {
    if (!ctx->start()) {
        SRV_ERR("%s", "failed to start the service socket\n");
        return false;
    }
    ctx->is_ready.store(true);
    return true;
}

void server_service::stop() {
    ctx->stop();
}

void server_service::write_lock(int port, int argc, char ** argv) {
    server_service_lock lock;
    lock.pid   = server_service_current_pid();
    lock.host  = host;
    lock.port  = port;
    lock.sock  = sock;
    lock.build = llama_build_info();
    for (int i = 1; i < argc; i++) {
        lock.args.push_back(argv[i]);
    }
    lock.write();
    SRV_INF("service listening on unix://%s\n", sock.c_str());
}

void server_service::shutdown() {
    ctx->stop();
    ctx->join();
    std::error_code ec;
    std::filesystem::remove(std::filesystem::u8path(sock), ec);
    server_service_lock::remove();
}
