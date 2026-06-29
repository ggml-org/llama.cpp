// infcore — корп. лицензия. Неизменяемый аудит-журнал (кто/когда/что/модель/итог).
// Только локальная запись (append-only JSONL, fsync). Без внешнего экспорта.
// Истинная immutability — средствами ОС (chattr +a / auditd); здесь O_APPEND+fsync.
#pragma once

#include <mutex>
#include <string>

namespace infcore {

struct AuditEvent {
    std::string subject;
    std::string role;
    std::string endpoint;
    std::string model;
    std::string client_ip;
    std::string decision;   // "allow" | "deny"
    std::string reason;
    int         status = 0;
};

class AuditLog {
public:
    ~AuditLog();
    bool open(const std::string& path);   // false при ошибке открытия
    void log(const AuditEvent& e);
    bool enabled() const { return fd_ >= 0; }

private:
    std::mutex mu_;
    int fd_ = -1;
};

}  // namespace infcore
