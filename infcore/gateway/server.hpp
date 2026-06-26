// infcore gateway — корпоративная лицензия.
// OpenAI-совместимый gateway: control-plane (auth/registry/routing/metrics) перед
// бэкендами llama-server. Прокси с passthrough SSE для stream-ответов.
#pragma once

#include <atomic>
#include <map>
#include <mutex>
#include <string>

#include "config.hpp"
#include "registry/model_registry.h"

namespace infcore {

class GatewayServer {
public:
    explicit GatewayServer(GatewayConfig cfg);
    int run();   // блокирующий; возвращает код выхода

private:
    GatewayConfig cfg_;
    ModelRegistry registry_;

    // примитивные метрики (pull, /metrics)
    std::mutex                                metrics_mu_;
    std::map<std::string, std::atomic<long>>  counters_;

    void   inc(const std::string& key);
    long   get_counter(const std::string& key);
    std::string render_metrics();
};

}  // namespace infcore
