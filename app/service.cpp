#include "server-common.h"
#include "server-service.h"

#include "common.h"
#include "subproc.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif

int llama_server(int argc, char ** argv);
int llama_service(int argc, char ** argv);

static void print_usage() {
    printf(
        "Usage: llama service <command> [server options]\n"
        "\n"
        "Commands:\n"
        "  start    Start the service in the background\n"
        "  stop     Stop the service and every model it runs\n"
        "  restart  Stop the service, then start it again with the same options\n"
        "  status   Show whether the service is running\n"
        "  run      Run the service in the foreground\n"
        "\n"
        "Server options such as --port, --host or --models-max are passed to the server.\n");
}

static bool has_arg(const std::vector<std::string> & args, const std::string & name) {
    for (const auto & a : args) {
        if (a == name) {
            return true;
        }
    }
    return false;
}

static void set_env(const char * key, const char * value) {
#if defined(_WIN32)
    _putenv_s(key, value);
#else
    setenv(key, value, 1);
#endif
}

static int service_run(std::vector<std::string> args, bool detached) {
    if (detached) {
#if !defined(_WIN32)
        setsid(); // the terminal that started the service must not reach it
#endif
        std::error_code ec;
        common_create_directories(server_service_dir(), ec);
        const std::string log = server_service_log_path().string();
        // the pipes of the process that started the service go away with it
        if (freopen(log.c_str(), "a", stdout) == nullptr || freopen(log.c_str(), "a", stderr) == nullptr) {
            return 1;
        }
        setvbuf(stdout, nullptr, _IOLBF, 0);
        setvbuf(stderr, nullptr, _IONBF, 0);
    }

    if (!has_arg(args, "--service-sock")) {
        args.push_back("--service-sock");
        args.push_back(server_service_default_sock().string());
    }

    // the router re-invokes this binary as 'llama serve' for its children
    set_env("LLAMA_APP_CMD", "serve");

    std::vector<std::string> argv_s = { "llama-server" };
    argv_s.insert(argv_s.end(), args.begin(), args.end());
    std::vector<char *> argv;
    for (auto & a : argv_s) {
        argv.push_back(a.data());
    }
    return llama_server((int) argv.size(), argv.data());
}

static int service_start(const std::vector<std::string> & args) {
    auto lock = server_service_lock::read();
    if (lock.has_value() && lock->alive()) {
        printf("service is already running (pid %d) at http://%s:%d\n", lock->pid, lock->host.c_str(), lock->port);
        return 0;
    }

    std::string exe;
    try {
        exe = get_server_exec_path().string();
    } catch (const std::exception & e) {
        fprintf(stderr, "error: cannot find the llama binary: %s\n", e.what());
        return 1;
    }

    std::vector<std::string> cmd = { exe, "service", "run", "--detached" };
    cmd.insert(cmd.end(), args.begin(), args.end());

    common_subproc proc;
    if (!proc.create(cmd, subprocess_option_no_window | subprocess_option_inherit_environment)) {
        fprintf(stderr, "error: failed to start the service\n");
        return 1;
    }

    // wait until the service answers on its socket
    for (int i = 0; i < 300; i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        lock = server_service_lock::read();
        if (lock.has_value() && lock->alive()) {
            try {
                auto res = server_service_request(lock->sock, "GET", "/service/status");
                if (res.status == 200) {
                    printf("service started (pid %d), listening on http://%s:%d\n", lock->pid, lock->host.c_str(), lock->port);
                    return 0;
                }
            } catch (const std::exception &) {
                // not listening yet
            }
        }
        if (!proc.alive()) {
            break;
        }
    }
    fprintf(stderr, "error: the service did not start, see %s\n", server_service_log_path().string().c_str());
    return 1;
}

static int service_stop() {
    auto lock = server_service_lock::read();
    if (!lock.has_value() || !lock->alive()) {
        printf("service is not running\n");
        server_service_lock::remove(/* force */ true);
        return 0;
    }

    try {
        server_service_request(lock->sock, "POST", "/service/stop", "{}");
    } catch (const std::exception & e) {
        fprintf(stderr, "warning: %s\n", e.what());
    }

    for (int i = 0; i < 300 && lock->alive(); i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (lock->alive()) {
        fprintf(stderr, "service did not stop in time, killing pid %d\n", lock->pid);
        server_service_pid_kill(lock->pid);
    }
    server_service_lock::remove(/* force */ true);
    printf("service stopped\n");
    return 0;
}

static int service_status() {
    auto lock = server_service_lock::read();
    if (!lock.has_value() || !lock->alive()) {
        printf("service is not running\n");
        return 1;
    }
    try {
        auto res = server_service_request(lock->sock, "GET", "/service/status");
        if (res.status != 200) {
            throw std::runtime_error("status " + std::to_string(res.status));
        }
        json status = json::parse(res.body);
        printf("service is running (pid %d)\n", json_value(status, "pid", lock->pid));
        printf("  address: http://%s:%d\n", lock->host.c_str(), lock->port);
        printf("  socket:  %s\n", lock->sock.c_str());
        printf("  build:   %s\n", json_value(status, "build", std::string()).c_str());
        const auto running = json_value(status, "running", std::vector<std::string>());
        printf("  running: %s\n", running.empty() ? "none" : string_join(running, ", ").c_str());
    } catch (const std::exception & e) {
        printf("service (pid %d) is not responding: %s\n", lock->pid, e.what());
        return 1;
    }
    return 0;
}

int llama_service(int argc, char ** argv) {
    const std::string cmd = argc >= 2 ? argv[1] : "";
    std::vector<std::string> args;
    for (int i = 2; i < argc; i++) {
        args.push_back(argv[i]);
    }

    if (cmd == "start") {
        return service_start(args);
    }
    if (cmd == "stop") {
        return service_stop();
    }
    if (cmd == "restart") {
        auto lock = server_service_lock::read();
        if (args.empty() && lock.has_value()) {
            args = lock->args;
        }
        service_stop();
        return service_start(args);
    }
    if (cmd == "status") {
        return service_status();
    }
    if (cmd == "run") {
        bool detached = false;
        for (auto it = args.begin(); it != args.end();) {
            if (*it == "--detached") {
                detached = true;
                it = args.erase(it);
            } else {
                ++it;
            }
        }
        return service_run(args, detached);
    }

    print_usage();
    if (!cmd.empty() && cmd != "help" && cmd != "--help" && cmd != "-h") {
        fprintf(stderr, "error: unknown command '%s'\n", cmd.c_str());
        return 1;
    }
    return 0;
}
