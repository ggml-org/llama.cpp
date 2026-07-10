// infcore runtime — корпоративная лицензия.
#include "backend_supervisor.h"

#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <vector>

#include "httplib.h"

namespace infcore {

namespace {
// Per-boot секрет для --api-key дочерних llama-server (offline, из /dev/urandom).
std::string gen_token() {
    std::ifstream ur("/dev/urandom", std::ios::binary);
    unsigned char buf[24];
    std::string tok;
    static const char* hx = "0123456789abcdef";
    if (ur.read(reinterpret_cast<char*>(buf), sizeof(buf))) {
        for (unsigned char c : buf) { tok.push_back(hx[c >> 4]); tok.push_back(hx[c & 0xF]); }
    }
    return tok;
}
}  // namespace

BackendSupervisor::BackendSupervisor(Options opt)
    : opt_(std::move(opt)), api_key_(gen_token()), next_port_(opt_.port_range_start) {
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
    httplib::Client cli("127.0.0.1", port);
    cli.set_connection_timeout(1, 0);
    cli.set_read_timeout(2, 0);
    httplib::Headers h;
    if (!api_key_.empty()) h.emplace("Authorization", "Bearer " + api_key_);
    const long long deadline = now_ms() + opt_.startup_timeout_ms;
    while (now_ms() < deadline) {
        auto r = cli.Get("/health", h);
        if (r && r->status == 200) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }
    return false;
}

// Управляемые бэкенды всегда слушают loopback и требуют per-boot --api-key:
// напрямую (в обход gateway/RBAC/audit) до них не достучаться.
bool BackendSupervisor::spawn(const ModelEntry& e, int port, pid_t& out_pid, std::string& err) {
    if (opt_.llama_server_bin.empty()) {
        err = "не задан runtime.llama_server_bin";
        return false;
    }

    std::vector<std::string> args = {
        opt_.llama_server_bin,
        "--host", "127.0.0.1",
        "--port", std::to_string(port),
        "--model", e.gguf_path,
        "--ctx-size", std::to_string(e.n_ctx),
        "--n-gpu-layers", std::to_string(e.n_gpu_layers),
    };
    if (!api_key_.empty()) { args.push_back("--api-key"); args.push_back(api_key_); }
    if (e.modality == Modality::Embedding) args.push_back("--embedding");
    if (!e.mmproj_path.empty()) { args.push_back("--mmproj"); args.push_back(e.mmproj_path); }

    std::vector<char*> argv;
    argv.reserve(args.size() + 1);
    for (auto& a : args) argv.push_back(a.data());
    argv.push_back(nullptr);

    pid_t pid = fork();
    if (pid < 0) { err = "fork() не удался"; return false; }
    if (pid == 0) {
        // дочерний: новая группа процессов, чтобы сигналы не задевали gateway
        setpgid(0, 0);
        // не наследуем дескрипторы gateway (слушающий сокет, клиентские соединения,
        // fd audit-журнала) - иначе ребёнок может держать порт/подорвать неизменяемость лога
        long maxfd = sysconf(_SC_OPEN_MAX);
        if (maxfd < 3 || maxfd > 4096) maxfd = 4096;
        for (int fd = 3; fd < (int)maxfd; ++fd) ::close(fd);
        execvp(argv[0], argv.data());
        std::perror("infcore: execvp llama-server");
        _exit(127);
    }

    out_pid = pid;
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
                // держим Failed до истечения backoff, затем разрешаем повторную попытку
                if (now_ms() < b.retry_after_ms) { err = b.last_error; return std::string(); }
                b.state = State::Stopped;
                continue;
            case State::Starting:
                b.cv.wait(lock);
                continue;
            case State::Stopped: {
                if (now_ms() < b.retry_after_ms) { err = b.last_error; return std::string(); }
                if (b.port == 0) b.port = next_port_++;  // порт назначаем под mu_ (без гонки)
                b.state = State::Starting;
                const int port = b.port;
                std::string serr;
                lock.unlock();  // fork + поллинг /health без блокировки остальных моделей
                pid_t pid = -1;
                bool ok = spawn(e, port, pid, serr);
                if (ok) ok = wait_health(port);
                lock.lock();
                b.pid = pid;  // присваиваем под mu_ (reaper читает pid под mu_)
                if (ok) {
                    b.url = "http://127.0.0.1:" + std::to_string(port);
                    b.state = State::Ready;
                    b.last_used_ms = now_ms();
                    b.fail_count = 0;
                    b.retry_after_ms = 0;
                } else {
                    if (serr.empty()) serr = "бэкенд не прошёл health-check за startup_timeout_ms";
                    b.last_error = serr;
                    stop_backend(b);
                    b.port = 0;  // при сбое отдаём порт (мог быть занят чужим процессом)
                    b.fail_count++;
                    int backoff = 5000 * b.fail_count;
                    if (backoff > 60000) backoff = 60000;
                    b.retry_after_ms = now_ms() + backoff;
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
            // упавший процесс: waitpid подбирает зомби и достоверно отличает живого от мёртвого
            // (kill(pid,0) для зомби возвращает успех и не детектил бы падение)
            if (b.pid > 0) {
                int st = 0;
                pid_t r = waitpid(b.pid, &st, WNOHANG);
                if (r != 0) {  // r==pid: завершился и подобран; r<0: процесса уже нет
                    b.pid = -1;
                    b.url.clear();
                    b.state = State::Stopped;
                    continue;
                }
            }
            if (b.active == 0 && t - b.last_used_ms > opt_.idle_timeout_ms)
                stop_backend(b);
        }
    }
}

}  // namespace infcore
