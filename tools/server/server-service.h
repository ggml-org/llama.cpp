#pragma once

#include "common.h"
#include "server-http.h"

#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

// the service: a router that also listens on a UNIX socket for 'llama service' and 'llama serve'.
// the socket carries the whole router API without an API key, plus the routes below. the lock
// file next to it says which process runs the service and how to reach it

struct server_models_routes;

// under the user config directory
std::filesystem::path server_service_dir();
std::filesystem::path server_service_lock_path();
std::filesystem::path server_service_default_sock();
std::filesystem::path server_service_log_path();

struct server_service_lock {
    int         pid  = 0;
    std::string host;
    int         port = 0;
    std::string sock;
    std::string build;
    std::vector<std::string> args; // the service was started with these, for a restart

    // nullopt when there is no lock file or it is unreadable
    static std::optional<server_service_lock> read();
    void write() const;
    // only when the lock names this process, or force is set
    static void remove(bool force = false);

    bool alive() const;
};

int  server_service_current_pid();
bool server_service_pid_alive(int pid);
void server_service_pid_kill(int pid);

// a request over the service socket; throws when the socket is unreachable
struct server_service_response {
    int         status = 0;
    std::string body;
};
server_service_response server_service_request(const std::string & sock, const std::string & method, const std::string & path, const std::string & body = "");

// the socket listener of a router, see server_run_router()
struct server_service {
    using wrap_t = std::function<server_http_context::handler_t(server_http_context::handler_t)>;

    // register_routes puts the router API on the socket, wrap is the exception wrapper of the router;
    // false when another service runs or the socket cannot be set up
    bool init(const common_params & params,
              server_models_routes & models_routes,
              const std::function<void(server_http_context &)> & register_routes,
              const wrap_t & wrap,
              std::function<void()> stop_server);

    bool start();
    void stop(); // from the shutdown handler
    void write_lock(int port, int argc, char ** argv); // throws when the lock file cannot be written
    void shutdown(); // join, remove the socket file and the lock

private:
    common_params params_service; // the listener keeps a reference to it
    std::string host; // of the TCP listener, for the lock file
    std::string sock;
    std::optional<server_http_context> ctx;

    server_http_context::handler_t get_status;
    server_http_context::handler_t post_spawn;
    server_http_context::handler_t post_stop;
    server_http_context::handler_t get_logs;
};
