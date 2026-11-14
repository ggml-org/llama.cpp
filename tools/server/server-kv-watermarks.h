#pragma once

#include <cstddef>
#include <cstdint>

struct llama_context;

struct server_kv_watermarks {
    int low = 20;
    int critical = 10;
    int level = 0;
    bool admission_held = false;

    void update(double free_percent, size_t resident, bool completed);
    size_t runnable(size_t resident) const;
    bool can_admit(double free_percent, size_t resident) const;
};

struct server_kv_watermark_cells {
    uint64_t free = 0;
    uint64_t total = 0;
    bool supported = true;

    double free_percent() const;
};

server_kv_watermark_cells server_kv_watermarks_get_cells(llama_context * ctx);
