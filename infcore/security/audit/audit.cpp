// infcore — корп. лицензия.
#include "audit/audit.h"

#include <cerrno>
#include <fcntl.h>
#include <unistd.h>

#include <ctime>

#include "nlohmann/json.hpp"

using json = nlohmann::json;

namespace infcore {

namespace {
std::string utc_now() {
    std::time_t t = std::time(nullptr);
    std::tm tm{};
    gmtime_r(&t, &tm);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return buf;
}
}  // namespace

AuditLog::~AuditLog() {
    {
        std::lock_guard<std::mutex> lock(mu_);
        stop_ = true;
    }
    cv_work_.notify_all();
    cv_commit_.notify_all();
    if (writer_.joinable()) writer_.join();   // дописывает остаток очереди перед выходом
    if (fd_ >= 0) ::close(fd_);
}

bool AuditLog::open(const std::string& path) {
    // O_CLOEXEC: дочерние llama-server не должны наследовать fd журнала.
    fd_ = ::open(path.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0640);
    if (fd_ < 0) return false;
    writer_ = std::thread([this] { writer_loop(); });
    return true;
}

void AuditLog::log(const AuditEvent& e) {
    if (fd_ < 0) return;
    json j = {
        {"ts", utc_now()},
        {"subject", e.subject},
        {"role", e.role},
        {"endpoint", e.endpoint},
        {"model", e.model},
        {"client_ip", e.client_ip},
        {"decision", e.decision},
        {"reason", e.reason},
        {"status", e.status},
    };
    std::string line = j.dump();
    line.push_back('\n');

    std::unique_lock<std::mutex> lock(mu_);
    if (stop_ || writer_failed_) return;   // писатель мёртв -> не залипаем навсегда
    const unsigned long long seq = ++enqueued_seq_;
    queue_.push_back(std::move(line));
    cv_work_.notify_one();
    // Ждём, пока наша запись зафиксирована на диске (делим fsync с соседями по батчу).
    // Очередь при этом ограничена числом одновременных запросов (каждый продьюсер
    // блокируется до коммита), так что расти неограниченно не может.
    cv_commit_.wait(lock, [&] { return committed_seq_ >= seq || writer_failed_ || stop_; });
}

// Поток-писатель: спит до появления работы, затем ЗАБИРАЕТ ВСЮ очередь одним
// батчем, пишет её и делает один fsync -> group-commit. После fsync поднимает
// committed_seq_ и будит всех ждущих продьюсеров этого батча.
void AuditLog::writer_loop() {
    std::unique_lock<std::mutex> lock(mu_);
    for (;;) {
        cv_work_.wait(lock, [&] { return !queue_.empty() || stop_; });
        if (queue_.empty()) {
            if (stop_) return;
            continue;
        }
        std::deque<std::string> batch;
        batch.swap(queue_);                       // забрали всё -> один fsync на всех
        const unsigned long long upto = enqueued_seq_;
        lock.unlock();

        bool ok = true;
        for (const auto& line : batch) {
            ssize_t off = 0, n = (ssize_t)line.size();
            while (off < n) {
                ssize_t w = ::write(fd_, line.data() + off, n - off);
                if (w < 0) {
                    if (errno == EINTR) continue;  // повтор, а не потеря записи
                    ok = false; break;             // фатальная I/O-ошибка
                }
                off += w;
            }
            if (!ok) break;
        }
        if (ok) ::fsync(fd_);

        lock.lock();
        if (ok) {
            committed_seq_ = upto;
        } else {
            writer_failed_ = true;   // разблокируем всех ждущих и больше не блокируем log()
        }
        cv_commit_.notify_all();
        if (writer_failed_) return;  // журнал сломан; продьюсеры увидят writer_failed_
    }
}

}  // namespace infcore
