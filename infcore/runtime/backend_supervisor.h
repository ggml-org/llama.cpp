// infcore runtime — корпоративная лицензия.
// Lazy-supervisor бэкендов: по требованию поднимает дочерние процессы llama-server
// для управляемых моделей (backend_url пуст), гасит их по простою. Модели с явным
// backend_url считаются внешними и здесь не управляются.
#pragma once

#include <sys/types.h>

#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "registry/model_registry.h"

namespace infcore {

class BackendSupervisor {
public:
    struct Options {
        std::string llama_server_bin;          // путь к нашему llama-server (из сборки)
        std::string host = "127.0.0.1";        // адрес привязки поднимаемых бэкендов
        int port_range_start  = 8100;          // откуда раздаются локальные порты
        int idle_timeout_ms   = 300000;        // простой до выгрузки
        int startup_timeout_ms = 120000;       // ожидание /health при старте
    };

    explicit BackendSupervisor(Options opt);
    ~BackendSupervisor();

    BackendSupervisor(const BackendSupervisor&) = delete;
    BackendSupervisor& operator=(const BackendSupervisor&) = delete;

    // Гарантирует, что управляемый бэкенд для модели поднят и здоров.
    // Возвращает базовый URL бэкенда либо пустую строку (err заполнен).
    // Блокирует на время старта; параллельные вызовы для одной модели ждут один старт.
    std::string ensure_ready(const ModelEntry& e, std::string& err);

    // Учёт активных запросов: reaper не гасит бэкенд с active > 0 (иначе оборвёт stream).
    void acquire(const std::string& logical_name);
    void release(const std::string& logical_name);

private:
    enum class State { Stopped, Starting, Ready, Failed };

    struct Backend {
        State        state = State::Stopped;
        pid_t        pid   = -1;
        int          port  = 0;          // sticky: назначается один раз
        std::string  url;
        std::string  last_error;
        long long    last_used_ms = 0;
        int          active = 0;
        std::condition_variable cv;
    };

    Backend& get_or_create(const std::string& logical_name);  // mu_ удерживается вызывающим
    bool spawn(const ModelEntry& e, Backend& b, std::string& err);  // fork+exec, без блокировки mu_
    bool wait_health(int port);          // поллит /health до startup_timeout_ms
    void stop_backend(Backend& b);       // SIGTERM -> SIGKILL, waitpid
    void reaper_loop();
    long long now_ms();

    Options    opt_;
    std::mutex mu_;
    std::map<std::string, std::unique_ptr<Backend>> backends_;
    int        next_port_;
    std::thread reaper_;
    bool       stop_ = false;
    std::condition_variable reaper_cv_;
};

}  // namespace infcore
