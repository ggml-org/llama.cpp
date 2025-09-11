#pragma once

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_tensor;

enum ggml_mem_range_type {
    MEM_RANGE_TYPE_SRC = 0,
    MEM_RANGE_TYPE_DST = 1,
};

struct ggml_mem_range_params {
    uint64_t p0; // being
    uint64_t p1; // end

    enum ggml_mem_range_type pt;
};

struct ggml_mem_ranges;

struct ggml_mem_ranges * ggml_mem_ranges_init(int debug);
void ggml_mem_ranges_free(struct ggml_mem_ranges * mrs);

void ggml_mem_ranges_reset(struct ggml_mem_ranges * mrs);

bool ggml_mem_ranges_add(struct ggml_mem_ranges * mrs, struct ggml_mem_range_params mrp);

bool ggml_mem_ranges_add_src(struct ggml_mem_ranges * mrs, const struct ggml_tensor * node);
bool ggml_mem_ranges_add_dst(struct ggml_mem_ranges * mrs, const struct ggml_tensor * node);

// return true if:
// - new src range overlaps with any existing dst range
// - new dst range overlaps with any existing range (src or dst)
bool ggml_mem_ranges_check(const struct ggml_mem_ranges * mrs, struct ggml_mem_range_params mrp);

bool ggml_mem_ranges_check_src(const struct ggml_mem_ranges * mrs, const struct ggml_tensor * node);
bool ggml_mem_ranges_check_dst(const struct ggml_mem_ranges * mrs, const struct ggml_tensor * node);

#ifdef __cplusplus
}
#endif
