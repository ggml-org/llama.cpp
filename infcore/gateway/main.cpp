// infcore gateway — корпоративная лицензия.
// Точка входа: загрузка конфига → запуск OpenAI-совместимого gateway (control-plane
// перед бэкендами llama-server). Никаких исходящих соединений за пределы контура.
#include <cstdio>
#include <exception>

#include "config.hpp"
#include "server.hpp"

namespace infcore { void metrics_init(); }

int main(int argc, char** argv) {
    const char* cfg_path = (argc > 1) ? argv[1] : "infcore/config/gateway.yaml";
    std::printf("infcore gateway 0.1.0\n");

    try {
        infcore::metrics_init();
        infcore::GatewayConfig cfg = infcore::load_config(cfg_path);
        infcore::GatewayServer server(std::move(cfg));
        return server.run();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "infcore: фатальная ошибка: %s\n", e.what());
        return 1;
    }
}
