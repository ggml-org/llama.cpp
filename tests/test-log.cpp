#include "arg.h"
#include "common.h"
#include "log.h"

#include <cstdlib>
#include <thread>

int main(int argc, char ** argv) {
    common_params params;
    params.model.path = "."; // this test takes no model
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    LOG("%s: running\n", "test-log");

    const int n_thread = 8;

    std::thread threads[n_thread];
    for (int i = 0; i < n_thread; i++) {
        threads[i] = std::thread([i]() {
            const int n_msg = 1000;

            for (int j = 0; j < n_msg; j++) {
                const int log_type = std::rand() % 3;

                switch (log_type) {
                    case 0: LOG_INF("Thread %d: %d\n", i, j); break;
                    case 1: LOG_TRC("Thread %d: %d\n", i, j); break;
                    case 2: LOG_DBG("Thread %d: %d\n", i, j); break;
                    default:
                        break;
                }

                if (rand () % 10 < 5) {
                    common_log_set_timestamps(common_log_main(), rand() % 2);
                    common_log_set_prefix    (common_log_main(), rand() % 2);
                }
            }
        });
    }

    for (int i = 0; i < n_thread; i++) {
        threads[i].join();
    }

    LOG("%s: %s\n", "test-log", "PASSED");

    common_log_flush(common_log_main());
    // We explicitly free the logger singleton to avoid hanging on Windows
    // related to timing issues of thread startup and DLL teardown
    common_log_free(common_log_main());
    return 0;
}
