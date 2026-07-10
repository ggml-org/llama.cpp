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
    if (fd_ >= 0) ::close(fd_);
}

bool AuditLog::open(const std::string& path) {
    // O_CLOEXEC: дочерние llama-server не должны наследовать fd журнала.
    fd_ = ::open(path.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0640);
    return fd_ >= 0;
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

    std::lock_guard<std::mutex> lock(mu_);
    ssize_t off = 0, n = (ssize_t)line.size();
    while (off < n) {
        ssize_t w = ::write(fd_, line.data() + off, n - off);
        if (w < 0) {
            if (errno == EINTR || errno == EAGAIN) continue;  // повтор, а не потеря записи
            break;   // фатальная ошибка записи; журнал не должен ронять gateway
        }
        off += w;
    }
    ::fsync(fd_);
}

}  // namespace infcore
