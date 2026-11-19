#include "log.h"

#include <cinttypes>
#include <cstdio>

std::string rpc_log_format_size(size_t size) {
    static const char * units[] = { "B", "KiB", "MiB", "GiB", "TiB", "PiB" };

    double value = (double) size;
    size_t unit  = 0;
    while (value >= 1024.0 && unit + 1 < sizeof(units) / sizeof(units[0])) {
        value /= 1024.0;
        unit++;
    }

    char buf[32];
    if (unit == 0) {
        snprintf(buf, sizeof(buf), "%zu B", size);
    } else {
        snprintf(buf, sizeof(buf), "%.2f %s", value, units[unit]);
    }
    return std::string(buf);
}

std::string rpc_log_format_tensor(const ggml_tensor * tensor) {
    char shape[64];
    snprintf(shape, sizeof(shape), "[%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "]", tensor->ne[0], tensor->ne[1],
             tensor->ne[2], tensor->ne[3]);

    char buf[GGML_MAX_NAME + 192];
    snprintf(buf, sizeof(buf), "%s [%s, %s, %s]", ggml_get_name(tensor), ggml_op_name(tensor->op),
             ggml_type_name(tensor->type), shape);
    return std::string(buf);
}
