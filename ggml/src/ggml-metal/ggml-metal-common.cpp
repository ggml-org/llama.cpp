#include "ggml-metal-common.h"

#include "ggml-impl.h"

#include <vector>

// keep this separate from the public ggml_mem_range_params
struct ggml_mem_range {
    uint64_t p0; // being
    uint64_t p1; // end

    enum ggml_mem_range_type pt;
};

struct ggml_mem_ranges {
    std::vector<struct ggml_mem_range> ranges;

    int debug = 0;
};

struct ggml_mem_ranges * ggml_mem_ranges_init(int debug) {
    auto * res = new struct ggml_mem_ranges;

    res->debug = debug;

    return res;
}

void ggml_mem_ranges_free(struct ggml_mem_ranges * mrs) {
    delete mrs;
}

void ggml_mem_ranges_reset(struct ggml_mem_ranges * mrs) {
    mrs->ranges.clear();
}

bool ggml_mem_ranges_add(struct ggml_mem_ranges * mrs, struct ggml_mem_range_params mrp) {
    mrs->ranges.push_back({
        /*.p0 =*/ mrp.p0,
        /*.p1 =*/ mrp.p1,
        /*.pt =*/ mrp.pt,
    });

    return true;
}

bool ggml_mem_ranges_add_src(struct ggml_mem_ranges * mrs, const struct ggml_tensor * node) {
    GGML_ASSERT(node);

    struct ggml_mem_range_params mrp = {
        /*.p0 =*/ (uint64_t) node->data,
        /*.p1 =*/ (uint64_t) node->data + ggml_nbytes(node),
        /*.pt =*/ MEM_RANGE_TYPE_SRC,
    };

    if (mrs->debug > 2) {
        GGML_LOG_DEBUG("%s: add src range [%lld, %lld)\n", __func__, mrp.p0, mrp.p1);
    }

    return ggml_mem_ranges_add(mrs, mrp);
}

bool ggml_mem_ranges_add_dst(struct ggml_mem_ranges * mrs, const struct ggml_tensor * node) {
    GGML_ASSERT(node);

    struct ggml_mem_range_params mrp = {
        /*.p0 =*/ (uint64_t) node->data,
        /*.p1 =*/ (uint64_t) node->data + ggml_nbytes(node),
        /*.pt =*/ MEM_RANGE_TYPE_DST,
    };

    if (mrs->debug > 2) {
        GGML_LOG_DEBUG("%s: add dst range [%lld, %lld)\n", __func__, mrp.p0, mrp.p1);
    }

    return ggml_mem_ranges_add(mrs, mrp);
}

bool ggml_mem_ranges_check(const struct ggml_mem_ranges * mrs, struct ggml_mem_range_params mrp) {
    for (size_t i = 0; i < mrs->ranges.size(); i++) {
        if (mrp.pt == MEM_RANGE_TYPE_SRC && mrs->ranges[i].pt == MEM_RANGE_TYPE_SRC) {
            continue;
        }

        if (mrp.p0 < mrs->ranges[i].p1 && mrp.p1 > mrs->ranges[i].p0) {
            return true;
        }
    }

    return false;
}

bool ggml_mem_ranges_check_src(const struct ggml_mem_ranges * mrs, const struct ggml_tensor * node) {
    GGML_ASSERT(node);

    struct ggml_mem_range_params mrp = {
        /*.p0 =*/ (uint64_t) node->data,
        /*.p1 =*/ (uint64_t) node->data + ggml_nbytes(node),
        /*.pt =*/ MEM_RANGE_TYPE_SRC,
    };

    const bool res = ggml_mem_ranges_check(mrs, mrp);

    if (res) {
        if (mrs->debug > 2) {
            GGML_LOG_DEBUG("%s: the src range [%lld, %lld) overlaps with a previous dst range\n", __func__, mrp.p0, mrp.p1);
        }
    }

    return res;
}

bool ggml_mem_ranges_check_dst(const struct ggml_mem_ranges * mrs, const struct ggml_tensor * node) {
    GGML_ASSERT(node);

    struct ggml_mem_range_params mrp = {
        /*.p0 =*/ (uint64_t) node->data,
        /*.p1 =*/ (uint64_t) node->data + ggml_nbytes(node),
        /*.pt =*/ MEM_RANGE_TYPE_DST,
    };

    const bool res = ggml_mem_ranges_check(mrs, mrp);

    if (res) {
        if (mrs->debug > 2) {
            GGML_LOG_DEBUG("%s: the dst range [%lld, %lld) overlaps with a previous src range\n", __func__, mrp.p0, mrp.p1);
        }
    }

    return res;
}
