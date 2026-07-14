// infcore — корп. лицензия. Неизменяемый аудит-журнал (кто/когда/что/модель/итог).
// Только локальная запись (append-only JSONL, fsync). Без внешнего экспорта.
// Истинная immutability — средствами ОС (chattr +a / auditd); здесь O_APPEND+fsync.
//
// Durability + пропускная способность: запись событий вынесена в выделенный
// поток-писатель с group-commit. log() кладёт строку в очередь и БЛОКИРУЕТСЯ до
// момента, когда его запись физически на диске (fsync). Писатель забирает всю
// накопленную очередь одним батчем и делает ОДИН fsync на всех -> N параллельных
// запросов делят один fsync вместо N последовательных. Ни одно событие не теряется
// (fsync до возврата из log()), при этом флуд не сериализуется на диске.
#pragma once

#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>
#include <thread>

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
    void writer_loop();                   // group-commit: батч + один fsync

    std::mutex               mu_;
    std::condition_variable  cv_work_;     // будит писателя: появилась работа/стоп
    std::condition_variable  cv_commit_;   // будит ждущих: их запись на диске
    std::deque<std::string>  queue_;       // строки в ожидании записи
    unsigned long long       enqueued_seq_  = 0;   // последний назначенный номер
    unsigned long long       committed_seq_ = 0;   // последний зафиксированный (fsync) номер
    bool                     stop_          = false;
    bool                     writer_failed_ = false; // фатальная ошибка I/O -> не блокируем навсегда
    std::thread              writer_;
    int                      fd_ = -1;
};

}  // namespace infcore
