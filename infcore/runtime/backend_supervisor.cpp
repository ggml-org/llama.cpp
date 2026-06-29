// infcore runtime — корпоративная лицензия.
#include "backend_supervisor.h"

#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <vector>

#include "httplib.h"

namespace infcore {

BackendSupervisor::BackendSupervisor(Options opt)
    : opt_(std::move(opt)), next_port_(opt_.port_range_start) {
    reaper_ = std::thread([this] { reaper_loop(); });
}

BackendSupervisor::~BackendSupervisor() {
    {
        std::lock_guard<std::mutex> lock(mu_);
        stop_ = true;
    }
    reaper_cv_.notify_all();
    if (reaper_.joinable()) reaper_.join();

    std::lock_guard<std::mutex> lock(mu_);
    for (auto& kv : backends_)
        if (kv.second->state == State::Ready || kv.second->state == State::Starting)
            stop_backend(*kv.second);
}

long long BackendSupervisor::now_ms() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

BackendSupervisor::Backend& BackendSupervisor::get_or_create(const std::string& name) {
    auto it = backends_.find(name);
    if (it == backends_.end())
        it = backends_.emplace(name, std::make_unique<Backend>()).first;
    return *it->second;
}

bool BackendSupervisor::wait_health(int port) {
    httplib::Client cli(opt_.host, port);
    cli.set_connection_timeout(1, 0);
    cli.set_read_timeout(2, 0);
    const long long deadline = now_ms() + opt_.startup_timeout_ms;
    while (now_ms() < deadline) {
        auto r = cli.Get("/health");
        if (r && r->status == 200) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }
    return false;
}

bool BackendSupervisor::spawn(const ModelEntry& e, Backend& b, std::string& err) {
    if (opt_.llama_server_bin.empty()) {
        err = "не задан runtime.llama_server_bin";
        return false;
    }
    if (b.port == 0) b.port = next_port_++;

    std::vector<std::string> args = {
        opt_.llama_server_bin,
        "--host", opt_.host,
        "--port", std::to_string(b.port),
        "--model", e.gguf_path,
        "--ctx-size", std::to_string(e.n_ctx),
        "--n-gpu-layers", std::to_string(e.n_gpu_layers),
    };
    if (e.modality == Modality::Embedding) args.push_back("--embedding");

    std::vector<char*> argv;
    argv.reserve(args.size() + 1);
    for (auto& a : args) argv.push_back(a.data());
    argv.push_back(nullptr);

    pid_t pid = fork();
    if (pid < 0) { err = "fork() не удался"; return false; }
    if (pid == 0) {
        // дочерний: новая группа процессов, чтобы сигналы не задевали gateway
        setpgid(0, 0);
        execvp(argv[0], argv.data());
        std::perror("infcore: execvp llama-server");
        _exit(127);
    }

    b.pid = pid;
    b.url = "http://" + opt_.host + ":" + std::to_string(b.port);
    return true;
}

void BackendSupervisor::stop_backend(Backend& b) {
    if (b.pid > 0) {
        kill(b.pid, SIGTERM);
        const long long deadline = now_ms() + 5000;
        bool reaped = false;
        while (now_ms() < deadline) {
            int st = 0;
            pid_t r = waitpid(b.pid, &st, WNOHANG);
            if (r == b.pid || r < 0) { reaped = true; break; }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        if (!reaped) {
            kill(b.pid, SIGKILL);
            waitpid(b.pid, nullptr, 0);
        }
    }
    b.pid = -1;
    b.url.clear();
    b.state = State::Stopped;
}

std::string BackendSupervisor::ensure_ready(const ModelEntry& e, std::string& err) {
    std::unique_lock<std::mutex> lock(mu_);
    Backend& b = get_or_create(e.logical_name);

    for (;;) {
        switch (b.state) {
            case State::Ready:
                b.last_used_ms = now_ms();
                return b.url;
            case State::Failed:
                err = b.last_error;
                b.state = State::Stopped;  // позволяем повторную попытку при следующем вызове
                return std::string();
            case State::Starting:
                b.cv.wait(lock);
                continue;
            case State::Stopped: {
                b.state = State::Starting;
                std::string serr;
                lock.unlock();  // fork + поллинг /health без блокировки остальных моделей
                bool ok = spawn(e, b, serr);
                if (ok) ok = wait_health(b.port);
                lock.lock();
                if (ok) {
                    b.state = State::Ready;
                    b.last_used_ms = now_ms();
                } else {
                    if (serr.empty()) serr = "бэкенд не прошёл health-check за startup_timeout_ms";
                    b.last_error = serr;
                    stop_backend(b);
                    b.state = State::Failed;
                }
                b.cv.notify_all();
                continue;
            }
        }
    }
}

void BackendSupervisor::acquire(const std::string& name) {
    std::lock_guard<std::mutex> lock(mu_);
    Backend& b = get_or_create(name);
    b.active++;
    b.last_used_ms = now_ms();
}

void BackendSupervisor::release(const std::string& name) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = backends_.find(name);
    if (it == backends_.end()) return;
    Backend& b = *it->second;
    if (b.active > 0) b.active--;
    b.last_used_ms = now_ms();
}

void BackendSupervisor::reaper_loop() {
    std::unique_lock<std::mutex> lock(mu_);
    while (!stop_) {
        reaper_cv_.wait_for(lock, std::chrono::seconds(5));
        if (stop_) break;
        const long long t = now_ms();
        for (auto& kv : backends_) {
            Backend& b = *kv.second;
            if (b.state != State::Ready) continue;
            // упавший процесс: пометить остановленным, подобрать зомби
            if (b.pid > 0 && kill(b.pid, 0) != 0) {
                waitpid(b.pid, nullptr, WNOHANG);
                b.pid = -1;
                b.url.clear();
                b.state = State::Stopped;
                continue;
            }
            if (b.active == 0 && t - b.last_used_ms > opt_.idle_timeout_ms)
                stop_backend(b);
        }
    }
}

}  // namespace infcore
