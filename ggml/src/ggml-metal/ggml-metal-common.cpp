#include "ggml-metal-common.h"

#include "ggml-impl.h"

#include <vector>

struct ggml_mem_range {
    uint64_t p0; // begin
    uint64_t p1; // end

    ggml_mem_range_type pt;
};

struct ggml_mem_ranges {
    std::vector<ggml_mem_range> ranges;

    int debug = 0;
};

struct ggml_mem_ranges * ggml_mem_ranges_init(int debug) {
    auto * res = new ggml_mem_ranges;

    res->ranges.reserve(256);
    res->debug = debug;

    return res;
}

void ggml_mem_ranges_free(ggml_mem_ranges * mrs) {
    delete mrs;
}

void ggml_mem_ranges_reset(ggml_mem_ranges * mrs) {
    mrs->ranges.clear();
}

static bool ggml_mem_ranges_add(ggml_mem_ranges * mrs, ggml_mem_range mrp) {
    mrs->ranges.push_back({
        /*.p0 =*/ mrp.p0,
        /*.p1 =*/ mrp.p1,
        /*.pt =*/ mrp.pt,
    });

    return true;
}

bool ggml_mem_ranges_add_src(ggml_mem_ranges * mrs, const ggml_tensor * node) {
    GGML_ASSERT(node);

    node = node->view_src ? node->view_src : node;

    ggml_mem_range mrp;

    if (node->buffer) {
        mrp = {
            /*.p0 =*/ (uint64_t) node->data,
            /*.p1 =*/ (uint64_t) node->data + ggml_nbytes(node),
            /*.pt =*/ MEM_RANGE_TYPE_SRC,
        };
    } else {
        mrp = {
            /*.p0 =*/ (uint64_t) node,
            /*.p1 =*/ (uint64_t) node + 1,
            /*.pt =*/ MEM_RANGE_TYPE_SRC,
        };
    };

    if (mrs->debug > 2) {
        GGML_LOG_DEBUG("%s: add src range [%lld, %lld)\n", __func__, mrp.p0, mrp.p1);
    }

    return ggml_mem_ranges_add(mrs, mrp);
}

bool ggml_mem_ranges_add_dst(ggml_mem_ranges * mrs, const ggml_tensor * node) {
    GGML_ASSERT(node);

    node = node->view_src ? node->view_src : node;

    ggml_mem_range mrp;

    if (node->buffer) {
        mrp = {
            /*.p0 =*/ (uint64_t) node->data,
            /*.p1 =*/ (uint64_t) node->data + ggml_nbytes(node),
            /*.pt =*/ MEM_RANGE_TYPE_DST,
        };
    } else {
        mrp = {
            /*.p0 =*/ (uint64_t) node,
            /*.p1 =*/ (uint64_t) node + 1,
            /*.pt =*/ MEM_RANGE_TYPE_DST,
        };
    };

    if (mrs->debug > 2) {
        GGML_LOG_DEBUG("%s: add dst range [%lld, %lld)\n", __func__, mrp.p0, mrp.p1);
    }

    return ggml_mem_ranges_add(mrs, mrp);
}

static bool ggml_mem_ranges_check(const ggml_mem_ranges * mrs, ggml_mem_range mrp) {
    for (size_t i = 0; i < mrs->ranges.size(); i++) {
        if (mrp.pt == MEM_RANGE_TYPE_SRC && mrs->ranges[i].pt == MEM_RANGE_TYPE_SRC) {
            continue;
        }

        if (mrp.p0 < mrs->ranges[i].p1 && mrp.p1 >= mrs->ranges[i].p0) {
            return true;
        }
    }

    return false;
}

bool ggml_mem_ranges_check_src(const ggml_mem_ranges * mrs, const ggml_tensor * node) {
    GGML_ASSERT(node);

    node = node->view_src ? node->view_src : node;

    ggml_mem_range mrp;

    if (node->buffer) {
        mrp = {
            /*.p0 =*/ (uint64_t) node->data,
            /*.p1 =*/ (uint64_t) node->data + ggml_nbytes(node),
            /*.pt =*/ MEM_RANGE_TYPE_SRC,
        };
    } else {
        mrp = {
            /*.p0 =*/ (uint64_t) node,
            /*.p1 =*/ (uint64_t) node + 1,
            /*.pt =*/ MEM_RANGE_TYPE_SRC,
        };
    };

    const bool res = ggml_mem_ranges_check(mrs, mrp);

    if (res) {
        if (mrs->debug > 2) {
            GGML_LOG_DEBUG("%s: the src range [%lld, %lld) overlaps with a previous dst range\n", __func__, mrp.p0, mrp.p1);
        }
    }

    return res;
}

bool ggml_mem_ranges_check_dst(const ggml_mem_ranges * mrs, const ggml_tensor * node) {
    GGML_ASSERT(node);

    node = node->view_src ? node->view_src : node;

    ggml_mem_range mrp;

    if (node->buffer) {
        mrp = {
            /*.p0 =*/ (uint64_t) node->data,
            /*.p1 =*/ (uint64_t) node->data + ggml_nbytes(node),
            /*.pt =*/ MEM_RANGE_TYPE_DST,
        };
    } else {
        mrp = {
            /*.p0 =*/ (uint64_t) node,
            /*.p1 =*/ (uint64_t) node + 1,
            /*.pt =*/ MEM_RANGE_TYPE_DST,
        };
    }

    const bool res = ggml_mem_ranges_check(mrs, mrp);

    if (res) {
        if (mrs->debug > 2) {
            GGML_LOG_DEBUG("%s: the dst range [%lld, %lld) overlaps with a previous src range\n", __func__, mrp.p0, mrp.p1);
        }
    }

    return res;
}

// TODO: move to ggml.h?
static bool is_empty(ggml_op op) {
    switch (op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
            return true;
        default:
            return false;
    }
}

struct node_info {
    ggml_tensor * node;

    std::vector<const ggml_tensor *> srcs;
    std::vector<ggml_tensor *> fused;

    ggml_op op() const {
        return node->op;
    }

    const ggml_tensor * dst() const {
        return fused.empty() ? node : fused.back();
    }

    bool is_empty() const {
        return ::is_empty(node->op);
    }

    void add_src(const ggml_tensor * src, bool check = false) {
        if (!src) {
            return;
        }

        src = src->view_src ? src->view_src : src;

        if (check) {
            for (const auto * prev : srcs) {
                if (prev == src) {
                    return;
                }
            }
        }

        srcs.push_back(src);
    }

    void add_fused(ggml_tensor * t) {
        fused.push_back(t);
    }
};

static std::vector<node_info> ggml_metal_graph_optimize_reorder(const std::vector<node_info> & nodes) {
    // helper to add node src and dst ranges
    const auto & h_add = [](ggml_mem_ranges * mrs, const node_info & node) {
        for (const auto * src : node.srcs) {
            ggml_mem_ranges_add_src(mrs, src);
        }

        ggml_mem_ranges_add_dst(mrs, node.dst());
    };

    // helper to check if a node ranges overlap with the existing set
    // if they overlap, the node cannot be executed concurrently with the nodes participating in this set
    const auto & h_overlap = [](const ggml_mem_ranges * mrs, const node_info & node) {
        for (const auto * src : node.srcs) {
            if (ggml_mem_ranges_check_src(mrs, src)) {
                return true;
            }
        }

        return ggml_mem_ranges_check_dst(mrs, node.dst());
    };

    // perform reorders only across these types of ops
    // can be expanded when needed
    // IMPORTANT: do not add ops such as GGML_OP_CPY or GGML_OP_SET_ROWS
    //            the dependencies from such ops are not always represented in the graph
    const auto & h_safe = [](ggml_op op) {
        switch (op) {
            case GGML_OP_MUL_MAT:
            case GGML_OP_MUL_MAT_ID:
            case GGML_OP_ROPE:
            case GGML_OP_NORM:
            case GGML_OP_RMS_NORM:
            case GGML_OP_GROUP_NORM:
            case GGML_OP_SUM:
            case GGML_OP_SUM_ROWS:
            case GGML_OP_MEAN:
            case GGML_OP_MUL:
            case GGML_OP_ADD:
            case GGML_OP_SUB:
            case GGML_OP_DIV:
            case GGML_OP_SCALE:
            case GGML_OP_GET_ROWS:
            case GGML_OP_CLAMP:
            case GGML_OP_UNARY:
                return true;
            default:
                return is_empty(op);
        }
    };

    const int n = nodes.size();

    std::vector<node_info> res;
    res.reserve(n);

    std::vector<bool> used(n, false);

    ggml_mem_ranges * mrs0 = ggml_mem_ranges_init(0);
    ggml_mem_ranges * mrs1 = ggml_mem_ranges_init(0);

    for (int i0 = 0; i0 < n; i0++) {
        if (used[i0]) {
            continue;
        }

        const auto & node0 = nodes[i0];

        if (node0.is_empty()) {
            res.push_back(node0);
            continue;
        }

        // the node is not concurrent with the existing concurrent set, so we have to "put a barrier" (i.e reset mrs0)
        // but before we do that, look forward for some other nodes that can be added to the concurrent set mrs0
        if (h_overlap(mrs0, node0)) {
            // this will hold the set of memory ranges from the nodes that haven't been processed yet
            ggml_mem_ranges_reset(mrs1);

            // initialize it with the current node
            h_add(mrs1, node0);

            for (int i1 = i0 + 1; i1 < i0 + 32 && i1 < n; i1++) {
                if (used[i1]) {
                    continue;
                }

                const auto & node1 = nodes[i1];

                if (!h_safe(node1.op())) {
                    break;
                }

                // to add a concurrent node, it has to:
                //   - be empty
                //   - be concurrent with all nodes in the existing concurrent set (mrs0)
                //   - be concurrent with all nodes prior to it that haven't been processed yet (mrs1)
                if ((node1.is_empty() || !h_overlap(mrs0, node1)) && !h_overlap(mrs1, node1)) {
                    if (!node1.is_empty()) {
                        // add the node to the existing concurrent set (i.e. reorder)
                        h_add(mrs0, node1);
                    }
                    res.push_back(node1);
                    used[i1] = true;
                } else {
                    // add the node to the set of nodes that haven't been processed, to prevent invalid reordering
                    h_add(mrs1, node1);
                }
            }

            // finalize the concurrent set and begina new one with the current node
            ggml_mem_ranges_reset(mrs0);
        }

        {
            h_add(mrs0, node0);
            res.push_back(node0);
        }
    }

    ggml_mem_ranges_free(mrs0);
    ggml_mem_ranges_free(mrs1);

    return res;
}

void ggml_metal_graph_optimize(ggml_cgraph * gf) {
    constexpr int MAX_FUSE = 16;

    const int n = gf->n_nodes;

    enum ggml_op ops[MAX_FUSE];

    std::vector<node_info> nodes;
    nodes.reserve(gf->n_nodes);

    // fuse nodes:
    // we don't want to make reorders that break fusing, so we pack all fusable tensors
    //   and perform the reorder over the fused nodes. after the reorder is done, we unfuse
    for (int i = 0; i < n; i++) {
        node_info node = {
            /*.node =*/ gf->nodes[i],
            /*.srcs =*/ {},
            /*.fused =*/ {},
        };

        for (int j = 0; j < GGML_MAX_SRC; j++) {
            node.add_src(gf->nodes[i]->src[j]);
        }

        // fuse only ops that start with these operations
        // can be expanded when needed
        if (node.op() == GGML_OP_ADD ||
            node.op() == GGML_OP_RMS_NORM) {
            ops[0] = node.op();

            int f = i + 1;
            while (f < n && f < i + MAX_FUSE) {
                // conservatively allow fusing only these ops
                // can be expanded when needed
                if (gf->nodes[f]->op != GGML_OP_ADD &&
                    gf->nodes[f]->op != GGML_OP_MUL &&
                    gf->nodes[f]->op != GGML_OP_RMS_NORM) {
                    break;
                }
                ops[f - i] = gf->nodes[f]->op;
                f++;
            }

            f -= i;
            for (; f > 1; f--) {
                if (ggml_can_fuse(gf, i, ops, f)) {
                    break;
                }
            }

            // add the fused tensors into the node info so we can unfuse them later
            for (int k = 1; k < f; k++) {
                ++i;

                // the .dst() becomes the last fused tensor
                node.add_fused(gf->nodes[i]);

                // track all sources of this fused node
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    node.add_src(gf->nodes[i]->src[j], true);
                }
            }
        }

        nodes.push_back(std::move(node));
    }

    // reorder to improve concurrency
    nodes = ggml_metal_graph_optimize_reorder(nodes);

    // unfuse
    {
        int j = 0;
        for (const auto & node : nodes) {
            gf->nodes[j++] = node.node;

            for (auto * fused : node.fused) {
                gf->nodes[j++] = fused;
            }
        }
    }
}
