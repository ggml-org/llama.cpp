#include "utils.hpp"
#include "server-models.h"

#include "download.h"

#include <cpp-httplib/httplib.h>

#include <functional>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <spawn.h>
#include <signal.h> // for kill()

#if defined(__APPLE__) && defined(__MACH__)
// macOS: use _NSGetExecutablePath to get the executable path
#include <mach-o/dyld.h>
#include <limits.h>
#endif

static std::filesystem::path get_server_exec_path() {
#if defined(_MSC_VER)
    wchar_t path[FILENAME_MAX] = { 0 };
    GetModuleFileNameW(nullptr, path, FILENAME_MAX);
    return std::filesystem::path(path);
#elif defined(__APPLE__) && defined(__MACH__)
    char small_path[PATH_MAX];
    uint32_t size = sizeof(small_path);

    if (_NSGetExecutablePath(small_path, &size) == 0) {
        // resolve any symlinks to get absolute path
        try {
            return std::filesystem::canonical(std::filesystem::path(small_path));
        } catch (...) {
            return std::filesystem::path(small_path);
        }
    } else {
        // buffer was too small, allocate required size and call again
        std::vector<char> buf(size);
        if (_NSGetExecutablePath(buf.data(), &size) == 0) {
            try {
                return std::filesystem::canonical(std::filesystem::path(buf.data()));
            } catch (...) {
                return std::filesystem::path(buf.data());
            }
        }
        return std::filesystem::path(std::string(buf.data(), (size > 0) ? size : 0));
    }
#else
    char path[FILENAME_MAX];
    ssize_t count = readlink("/proc/self/exe", path, FILENAME_MAX);
    return std::filesystem::path(std::string(path, (count > 0) ? count: 0));
#endif
}

//
// server_models
//

server_models::server_models(
        const common_params & params,
        int argc,
        char ** argv,
        char ** envp) : base_params(params) {
    for (int i = 0; i < argc; i++) {
        base_args.push_back(std::string(argv[i]));
    }
    for (char ** env = envp; *env != nullptr; env++) {
        base_env.push_back(std::string(*env));
    }
    // TODO: allow refreshing cached model list
    auto cached_models = common_list_cached_models();
    for (const auto & model : cached_models) {
        server_model_meta meta{
            /* name        */ model.to_string(),
            /* path        */ model.manifest_path,
            /* path_mmproj */ "",
            /* in_cache    */ true,
            /* port        */ 0,
            /* status      */ SERVER_MODEL_STATUS_UNLOADED
        };
        mapping[meta.name] = instance_t{0, std::thread(), meta};
    }
}

void server_models::update_meta(const std::string & name, const server_model_meta & meta) {
    std::lock_guard<std::mutex> lk(mutex);
    auto it = mapping.find(name);
    if (it != mapping.end()) {
        it->second.meta = meta;
    }
    cv.notify_all(); // notify wait_until_loaded
}

bool server_models::has_model(const std::string & name) {
    std::lock_guard<std::mutex> lk(mutex);
    return mapping.find(name) != mapping.end();
}

std::optional<server_model_meta> server_models::get_meta(const std::string & name) {
    std::lock_guard<std::mutex> lk(mutex);
    auto it = mapping.find(name);
    if (it != mapping.end()) {
        return it->second.meta;
    }
    return std::nullopt;
}

static int get_free_port(std::string host) {
    httplib::Server s;
    int port = s.bind_to_any_port(host.c_str());
    s.stop();
    return port;
}

// helper to convert vector<string> to char **
// pointers are only valid as long as the original vector is valid
static std::vector<char *> to_char_ptr_array(const std::vector<std::string> & vec) {
    std::vector<char *> result;
    result.reserve(vec.size() + 1);
    for (const auto & s : vec) {
        result.push_back(const_cast<char*>(s.c_str()));
    }
    result.push_back(nullptr);
    return result;
}

std::vector<server_model_meta> server_models::get_all_meta() {
    std::lock_guard<std::mutex> lk(mutex);
    std::vector<server_model_meta> result;
    for (const auto & [name, inst] : mapping) {
        result.push_back(inst.meta);
    }
    return result;
}

void server_models::load(const std::string & name) {
    auto meta = get_meta(name);
    if (!meta.has_value()) {
        throw std::runtime_error("model name=" + name + " is not found");
    }

    std::lock_guard<std::mutex> lk(mutex);
    if (meta->status != SERVER_MODEL_STATUS_FAILED && meta->status != SERVER_MODEL_STATUS_UNLOADED) {
        SRV_INF("model %s is not ready\n", name.c_str());
        return;
    }

    instance_t inst;
    inst.meta        = meta.value();
    inst.meta.port   = get_free_port(base_params.hostname);
    inst.meta.status = SERVER_MODEL_STATUS_LOADING;

    pid_t pid = 0;
    {
        std::string exec_path = get_server_exec_path().string();
        SRV_INF("spawning server instance with name=%s on port %d\n", inst.meta.name.c_str(), inst.meta.port);

        std::vector<std::string> child_args = base_args; // copy
        if (inst.meta.in_cache) {
            child_args.push_back("-hf");
            child_args.push_back(inst.meta.name);
        } else {
            child_args.push_back("-m");
            child_args.push_back(inst.meta.path);
            if (!inst.meta.path_mmproj.empty()) {
                child_args.push_back("--mmproj");
                child_args.push_back(inst.meta.path_mmproj);
            }
        }
        child_args.push_back("--alias");
        child_args.push_back(inst.meta.name);
        child_args.push_back("--port");
        child_args.push_back(std::to_string(inst.meta.port));

        std::vector<std::string> child_env = base_env; // copy
        child_env.push_back("LLAMA_SERVER_ROUTER_PORT=" + std::to_string(base_params.port));

        // TODO: add logging
        SRV_INF("%s", "spawning server instance with args:\n");
        for (const auto & arg : child_args) {
            SRV_INF("  %s\n", arg.c_str());
        }

        std::vector<char *> argv = to_char_ptr_array(child_args);
        std::vector<char *> envp = to_char_ptr_array(child_env);

        if (posix_spawn(&pid, exec_path.c_str(), NULL, NULL, argv.data(), envp.data()) != 0) {
            perror("posix_spawn");
            throw std::runtime_error("failed to spawn server instance");
        } else {
            inst.pid = pid;
            SRV_INF("spawned instance with pid %d\n", pid);
        }
    }

    inst.th = std::thread([this, name, pid]() {
        int exit_code = 0;
        waitpid(pid, &exit_code, 0);
        SRV_INF("instance with pid %d exited with status %d\n", pid, exit_code);
        // note: if this is reached before std::move(inst) happens,
        //       this will be blocked until lock_guard is released (no race condition)
        this->update_status(name, exit_code == 0 ? SERVER_MODEL_STATUS_UNLOADED : SERVER_MODEL_STATUS_FAILED);
    });
    if (inst.th.joinable()) {
        inst.th.detach();
    }

    mapping[name] = std::move(inst);
    cv.notify_all();
}

void server_models::unload(const std::string & name) {
    std::lock_guard<std::mutex> lk(mutex);
    auto it = mapping.find(name);
    if (it != mapping.end()) {
        if (it->second.pid != 0) {
            SRV_INF("killing instance %s with pid %d\n", name.c_str(), it->second.pid);
            kill(it->second.pid, SIGTERM);
        }
        it->second.meta.status = SERVER_MODEL_STATUS_UNLOADED;
        cv.notify_all(); // notify status change
    }
}

void server_models::unload_all() {
    auto all_meta = get_all_meta();
    for (const auto & meta : all_meta) {
        unload(meta.name);
    }
}

void server_models::update_status(const std::string & name, server_model_status status) {
    auto meta = get_meta(name);
    if (meta.has_value()) {
        meta->status = status;
        update_meta(name, meta.value());
    }
}

void server_models::wait_until_loaded(const std::string & name) {
    std::unique_lock<std::mutex> lk(mutex);
    cv.wait(lk, [this, &name]() {
        auto it = mapping.find(name);
        if (it != mapping.end()) {
            return it->second.meta.status == SERVER_MODEL_STATUS_LOADED ||
                   it->second.meta.status == SERVER_MODEL_STATUS_FAILED;
        }
        return false;
    });
}

void server_models::ensure_model_loaded(const std::string & name) {
    auto meta = get_meta(name);
    if (!meta.has_value()) {
        throw std::runtime_error("model name=" + name + " is not found");
    }
    if (meta->status == SERVER_MODEL_STATUS_LOADED) {
        return; // already loaded
    }
    load(name);
    wait_until_loaded(name);
}

server_http_res_ptr server_models::proxy_request(const server_http_req & req, const std::string & method, const std::string & name) {
    auto meta = get_meta(name);
    if (!meta.has_value()) {
        throw std::runtime_error("model name=" + name + " is not found");
    }
    ensure_model_loaded(name); // TODO: handle failure case
    SRV_INF("proxying request to model %s at port %d\n", name.c_str(), meta->port);
    auto proxy = std::make_unique<server_http_proxy>(
            method,
            base_params.hostname,
            meta->port,
            req.path,
            req.headers,
            req.body,
            req.should_stop);
    return proxy;
}

void server_models::notify_router_server_ready(const std::string & name) {
    // send a notification to the router server that a model instance is ready
    const char * router_port = std::getenv("LLAMA_SERVER_ROUTER_PORT");
    if (router_port == nullptr) {
        // no router server to notify, this is a standalone server
        return;
    }

    httplib::Client cli("localhost", std::atoi(router_port));
    cli.set_connection_timeout(0, 200000); // 200 milliseconds

    httplib::Request req;
    req.method = "POST";
    req.path   = "/models/status";
    req.set_header("Content-Type", "application/json");

    json body;
    body["model"] = name;
    body["value"] = server_model_status_to_string(SERVER_MODEL_STATUS_LOADED);
    req.body = body.dump();

    SRV_INF("notifying router server that model %s is ready\n", name.c_str());
    cli.send(std::move(req));
    // discard response
}


//
// server_http_proxy
//

// simple implementation of a pipe
// used for streaming data between threads
template<typename T>
struct pipe_t {
    std::mutex mutex;
    std::condition_variable cv;
    std::queue<T> queue;
    std::atomic<bool> writer_closed{false};
    std::atomic<bool> reader_closed{false};
    void close_write() {
        writer_closed.store(true);
        cv.notify_all();
    }
    void close_read() {
        reader_closed.store(true);
        cv.notify_all();
    }
    bool read(T & output, const std::function<bool()> & should_stop) {
        std::unique_lock<std::mutex> lk(mutex);
        constexpr auto poll_interval = std::chrono::milliseconds(500);
        while (true) {
            if (!queue.empty()) {
                output = std::move(queue.front());
                queue.pop();
                return true;
            }
            if (writer_closed.load()) {
                return false; // clean EOF
            }
            if (should_stop()) {
                close_read(); // signal broken pipe to writer
                return false; // cancelled / reader no longer alive
            }
            cv.wait_for(lk, poll_interval);
        }
    }
    bool write(T && data) {
        std::lock_guard<std::mutex> lk(mutex);
        if (reader_closed.load()) {
            return false; // broken pipe
        }
        queue.push(std::move(data));
        cv.notify_one();
        return true;
    }
};

server_http_proxy::server_http_proxy(
        const std::string & method,
        const std::string & host,
        int port,
        const std::string & path,
        const std::map<std::string, std::string> & headers,
        const std::string & body,
        const std::function<bool()> should_stop) {
    // shared between reader and writer threads
    auto cli  = std::make_shared<httplib::Client>(host, port);
    auto pipe = std::make_shared<pipe_t<msg_t>>();

    // setup Client
    cli->set_connection_timeout(0, 200000); // 200 milliseconds
    this->status = 500; // to be overwritten upon response
    this->cleanup = [pipe]() {
        pipe->close_read();
        pipe->close_write();
    };

    // wire up the receive end of the pipe
    this->next = [pipe, should_stop](std::string & out) -> bool {
        msg_t msg;
        bool has_next = pipe->read(msg, should_stop);
        if (!msg.data.empty()) {
            out = std::move(msg.data);
        }
        return has_next; // false if EOF or pipe broken
    };

    // wire up the HTTP client
    // note: do NOT capture `this` pointer, as it may be destroyed before the thread ends
    httplib::ResponseHandler response_handler = [pipe, cli](const httplib::Response & response) {
        msg_t msg;
        msg.status = response.status;
        for (const auto & [key, value] : response.headers) {
            msg.headers[key] = value;
        }
        return pipe->write(std::move(msg)); // send headers first
    };
    httplib::ContentReceiverWithProgress content_receiver = [pipe](const char * data, size_t data_length, size_t, size_t) {
        // send data chunks
        // returns false if pipe is closed / broken (signal to stop receiving)
        return pipe->write({{}, 0, std::string(data, data_length)});
    };

    // prepare the request to destination server
    httplib::Request req;
    {
        req.method = method;
        req.path = path;
        for (const auto & [key, value] : headers) {
            req.set_header(key, value);
        }
        req.body = body;
        req.response_handler = response_handler;
        req.content_receiver = content_receiver;
    }

    // start the proxy thread
    SRV_DBG("start proxy thread %s %s\n", req.method.c_str(), req.path.c_str());
    this->thread = std::thread([cli, pipe, req]() {
        auto result = cli->send(std::move(req));
        if (result.error() != httplib::Error::Success) {
            auto err_str = httplib::to_string(result.error());
            SRV_ERR("http client error: %s\n", err_str.c_str());
            pipe->write({{}, 500, ""}); // header
            pipe->write({{}, 0, "proxy error: " + err_str}); // body
        }
        pipe->close_write(); // signal EOF to reader
        SRV_DBG("%s", "client request thread ended\n");
    });
    this->thread.detach();

    // wait for the first chunk (headers)
    msg_t header;
    pipe->read(header, should_stop);
    SRV_DBG("%s", "received response headers\n");
    this->status  = header.status;
    this->headers = header.headers;
}
