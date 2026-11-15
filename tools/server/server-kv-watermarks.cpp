#include "server-kv-watermarks.h"

#include "llama.h"
#include "src/llama-memory.h"

#include <algorithm>

void server_kv_watermarks::update(double free_percent, size_t resident, bool completed) {
    if (resident == 0) {
        level = 0;
        admission_held = false;
        return;
    }
    const int pressure = free_percent <= critical ? 2 : free_percent <= low ? 1 : 0;
    if (pressure > level) {
        level = pressure;
        admission_held = true;
    } else if (completed) {
        level = pressure;
    }
    // Recovery follows a completion, not a transient change around the watermark.
    if (completed && free_percent >= low) {
        admission_held = false;
    }
}

size_t server_kv_watermarks::runnable(size_t resident) const {
    return level == 2 ? std::min(size_t(1), resident) : level == 1 ? (resident + 1) / 2 : resident;
}

bool server_kv_watermarks::can_admit(double free_percent, size_t resident) const {
    return resident == 0 || (!admission_held && free_percent >= low);
}

double server_kv_watermark_cells::free_percent() const {
    return total == 0 ? 100.0 : 100.0 * free / total;
}

server_kv_watermark_cells server_kv_watermarks_get_cells(llama_context * ctx) {
    server_kv_watermark_cells result;
    if (ctx) {
        result.supported = llama_memory_get_attention_cells(llama_get_memory(ctx), result.free, result.total);
    }
    return result;
}
