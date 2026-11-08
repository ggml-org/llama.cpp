#include "grammar.cuh"

struct grammar_program {
    const ggml_grammar_op * ops;
    const ggml_grammar_rule * rules;
    const uint32_t * alts;
    const ggml_grammar_range * ranges;
};

static __device__ bool grammar_charge(uint32_t & work, uint32_t amount) {
    if (amount > work) { return false; }
    work -= amount;
    return true;
}

static __device__ bool grammar_equal(const ggml_grammar_stack & a, const ggml_grammar_stack & b) {
    if (a.size != b.size) { return false; }
    for (uint32_t i = 0; i < a.size; ++i) { if (a.frames[i] != b.frames[i]) { return false; } }
    return true;
}

static __device__ bool grammar_add(ggml_grammar_state & dst, const ggml_grammar_stack & stack) {
    for (uint32_t i = 0; i < dst.count; ++i) { if (grammar_equal(dst.stacks[i], stack)) { return true; } }
    if (dst.count == GGML_GRAMMAR_STACKS) { return false; }
    dst.stacks[dst.count++] = stack;
    return true;
}

static __device__ bool grammar_push(ggml_grammar_stack & stack, uint32_t offset) {
    if (offset == GGML_GRAMMAR_NONE) { return true; }
    if (stack.size == GGML_GRAMMAR_DEPTH) { return false; }
    stack.frames[stack.size++] = offset;
    return true;
}

static __device__ bool grammar_match(grammar_program p, const ggml_grammar_op & op, uint32_t chr) {
    if (op.type == GGML_GRAMMAR_CHAR_ANY) { return true; }
    bool found = false;
    for (uint32_t i = 0; i < op.count; ++i) {
        const auto r = p.ranges[op.value + i];
        found = found || (r.low <= chr && chr <= r.high);
    }
    return found == (op.type == GGML_GRAMMAR_CHAR);
}

static __device__ bool grammar_partial(grammar_program p, const ggml_grammar_op & op, uint32_t value, int remain, bool overlong) {
    if (remain < 0 || (!overlong && remain == 1 && value < 2)) { return false; }
    uint32_t low = value << (remain*6);
    const uint32_t high = low | ((uint32_t(1) << (remain*6)) - 1);
    if (low == 0 && !overlong) {
        if (remain == 2) { low = 1 << 11; }
        if (remain == 3) { low = 1 << 16; }
    }
    if (op.type == GGML_GRAMMAR_CHAR_ANY) { return true; }
    for (uint32_t i = 0; i < op.count; ++i) {
        const auto r = p.ranges[op.value + i];
        if (r.low <= high && low <= r.high) { return op.type == GGML_GRAMMAR_CHAR; }
    }
    return op.type != GGML_GRAMMAR_CHAR;
}

static __device__ bool grammar_advance(grammar_program p, const ggml_grammar_state & current,
        ggml_grammar_worker & worker, uint32_t chr, uint32_t & work) {
    auto & after = worker.next;
    after.count = 0;
    for (uint32_t i = 0; i < current.count; ++i) {
        auto stack = current.stacks[i];
        if (!stack.size) { continue; }
        const auto op = p.ops[stack.frames[stack.size - 1]];
        if (!grammar_charge(work, op.count + 1)) { return false; }
        if (!grammar_match(p, op, chr)) { continue; }
        --stack.size;
        if (!grammar_push(stack, op.next)) { return false; }
        int n_todo = 1;
        worker.todo[0] = stack;
        while (n_todo) {
            stack = worker.todo[--n_todo];
            if (!grammar_charge(work, stack.size + 1)) { return false; }
            if (!stack.size) {
                if (!grammar_add(after, stack)) { return false; }
                continue;
            }
            const auto top = p.ops[stack.frames[stack.size - 1]];
            if (top.type == GGML_GRAMMAR_RULE_REF) {
                --stack.size;
                if (!grammar_push(stack, top.next)) { return false; }
                const auto rule = p.rules[top.value];
                for (uint32_t j = 0; j < rule.count; ++j) {
                    if (!grammar_charge(work, stack.size + 1) || n_todo == GGML_GRAMMAR_STACKS) { return false; }
                    auto branch = stack;
                    const uint32_t alt = p.alts[rule.start + j];
                    if (p.ops[alt].type != GGML_GRAMMAR_END && p.ops[alt].type != GGML_GRAMMAR_ALT && !grammar_push(branch, alt)) { return false; }
                    worker.todo[n_todo++] = branch;
                }
            } else if (top.type == GGML_GRAMMAR_CHAR || top.type == GGML_GRAMMAR_CHAR_NOT || top.type == GGML_GRAMMAR_CHAR_ANY) {
                if (!grammar_add(after, stack)) { return false; }
            } else { return false; }
        }
    }
    return true;
}

static __device__ void grammar_read_state(const uint32_t * key, ggml_grammar_state & state) {
    state.count = key[7];
    state.value = key[5];
    state.remain = int32_t(key[6]);
    state.strict = state.remain > 0;
    state.nul = state.stopped = 0;
    uint32_t offset = 8;
    for (uint32_t i = 0; i < state.count; ++i) {
        auto & stack = state.stacks[i];
        stack.size = key[offset++];
        for (uint32_t j = 0; j < stack.size; ++j) { stack.frames[j] = key[offset++]; }
    }
}

static __device__ uint32_t grammar_hash(const uint32_t * key) {
    uint32_t hash = 2166136261u;
    for (int i = 0; i < GGML_GRAMMAR_KEY_WORDS; ++i) { hash = (hash ^ key[i])*16777619u; }
    return hash;
}

static __global__ void grammar_lookup(const uint32_t * key, uint32_t * cache, uint32_t * mask, uint32_t * status, int words) {
    if (threadIdx.x) { return; }
    const uint32_t hash = grammar_hash(key);
    const int stride = 4 + GGML_GRAMMAR_KEY_WORDS + words;
    for (int i = 0; i < words; ++i) { mask[i] = 0; }
    for (int way = 0; way < GGML_GRAMMAR_WAYS; ++way) {
        const auto * slot = cache + (hash % GGML_GRAMMAR_SETS*GGML_GRAMMAR_WAYS + way)*stride;
        if (!slot[0] || slot[1] != hash) { continue; }
        bool equal = true;
        for (int i = 0; i < GGML_GRAMMAR_KEY_WORDS; ++i) { equal = equal && slot[4 + i] == key[i]; }
        if (!equal) { continue; }
        for (int i = 0; i < words; ++i) { mask[i] = slot[4 + GGML_GRAMMAR_KEY_WORDS + i]; }
        status[0] = slot[2];
        status[1] = 1;
        return;
    }
}

static __global__ void grammar_traverse(const uint32_t * program, const uint32_t * vocab, const uint32_t * key,
        ggml_grammar_worker * scratch, uint32_t * mask, uint32_t * status) {
    if (status[1]) { return; }
    const auto & ph = *reinterpret_cast<const ggml_grammar_program_header *>(program);
    const auto & vh = *reinterpret_cast<const ggml_grammar_vocab_header *>(vocab);
    const grammar_program p = {reinterpret_cast<const ggml_grammar_op *>(program + ph.ops),
        reinterpret_cast<const ggml_grammar_rule *>(program + ph.rules), program + ph.alts,
        reinterpret_cast<const ggml_grammar_range *>(program + ph.ranges)};
    const auto * nodes = reinterpret_cast<const ggml_grammar_node *>(vocab + vh.nodes);
    const auto * terminals = reinterpret_cast<const ggml_grammar_terminal *>(vocab + vh.terminals);
    const auto * tokens = reinterpret_cast<const ggml_grammar_token *>(vocab + vh.tokens);
    auto & worker = scratch[threadIdx.x];
    uint32_t work = status[2];
    grammar_read_state(key, worker.levels[0].state);
    if (threadIdx.x == 0) {
        bool eog = false;
        for (uint32_t i = 0; i < worker.levels[0].state.count; ++i) { eog = eog || !worker.levels[0].state.stacks[i].size; }
        if (eog) {
            for (uint32_t id = 0; id < vh.n_vocab; ++id) {
                if (tokens[id].flags & GGML_GRAMMAR_EOG) { atomicOr(mask + id/32, uint32_t(1) << (id%32)); }
            }
        }
    }
    for (uint32_t root = threadIdx.x; root < 256; root += GGML_GRAMMAR_WORKERS) {
        int depth = 0;
        worker.levels[0].child = vh.roots[root];
        while (depth >= 0) {
            auto & parent = worker.levels[depth];
            if (parent.child == GGML_GRAMMAR_NONE) { --depth; continue; }
            if (depth + 1 == GGML_GRAMMAR_LEVELS || !grammar_charge(work, 1 + parent.state.count)) {
                atomicExch(status, uint32_t(GGML_GRAMMAR_NEED_CPU)); return;
            }
            const auto node = nodes[parent.child];
            parent.child = depth == 0 ? GGML_GRAMMAR_NONE : node.sibling;
            auto & next = worker.levels[depth + 1].state;
            next = parent.state;
            bool reject = false;
            if (!next.nul) {
                const uint32_t byte = node.byte;
                if (!byte) { next.nul = 1; }
                else {
                    if (next.remain > 0) {
                        if (next.strict && (byte >> 6) != 2) { continue; }
                        next.value = (next.value << 6) + (byte & 0x3f);
                        --next.remain;
                    } else {
                        const int length = byte < 128 ? 1 : byte < 192 ? 0 : byte < 224 ? 2 : byte < 240 ? 3 : 4;
                        next.remain = length - 1;
                        if (next.remain < 0) { continue; }
                        next.value = byte & ((1 << (7 - next.remain)) - 1);
                    }
                    if (next.remain == 0) {
                        next.strict = 0;
                        if (!next.value) { next.stopped = 1; }
                        if (!next.stopped) {
                            if (!grammar_advance(p, next, worker, next.value, work)) {
                                atomicExch(status, uint32_t(GGML_GRAMMAR_NEED_CPU)); return;
                            }
                            next.count = worker.next.count;
                            for (uint32_t i = 0; i < next.count; ++i) { next.stacks[i] = worker.next.stacks[i]; }
                            if (!next.count) { continue; }
                        }
                    }
                }
            }
            if (!next.nul && !next.stopped && next.remain > 0 && next.value && next.value <= (UINT32_MAX >> (6*next.remain))) {
                reject = true;
                for (uint32_t i = 0; i < next.count; ++i) {
                    const auto & stack = next.stacks[i];
                    if (stack.size) {
                        const auto op = p.ops[stack.frames[stack.size - 1]];
                        if (op.type == GGML_GRAMMAR_CHAR_NOT || grammar_partial(p, op, next.value, next.remain, true)) { reject = false; break; }
                    }
                }
            }
            if (reject) { continue; }
            if (node.terminal != GGML_GRAMMAR_NONE) {
                bool valid = false;
                for (uint32_t i = 0; i < next.count; ++i) {
                    const auto & stack = next.stacks[i];
                    if (next.remain == 0 || (stack.size && grammar_partial(p, p.ops[stack.frames[stack.size - 1]], next.value, next.remain, false))) {
                        valid = true; break;
                    }
                }
                if (valid) {
                    for (uint32_t t = node.terminal; t != GGML_GRAMMAR_NONE; t = terminals[t].next) {
                        if (!grammar_charge(work, 1)) { atomicExch(status, uint32_t(GGML_GRAMMAR_NEED_CPU)); return; }
                        const uint32_t id = terminals[t].id;
                        atomicOr(mask + id/32, uint32_t(1) << (id%32));
                    }
                }
            }
            worker.levels[++depth].child = node.child;
        }
    }
    atomicAdd(status + 3, status[2] - work);
}

static __global__ void grammar_publish(const uint32_t * key, uint32_t * cache, const uint32_t * mask, uint32_t * status, int words) {
    if (threadIdx.x || status[1] || status[0] == GGML_GRAMMAR_NEED_CPU) { return; }
    bool any = false;
    for (int i = 0; i < words; ++i) { any = any || mask[i] != 0; }
    status[0] = any ? GGML_GRAMMAR_OK : GGML_GRAMMAR_NO_VALID_TOKEN;
    const uint32_t hash = grammar_hash(key);
    const int stride = 4 + GGML_GRAMMAR_KEY_WORDS + words;
    for (int way = 0; way < GGML_GRAMMAR_WAYS; ++way) {
        auto * slot = cache + (hash % GGML_GRAMMAR_SETS*GGML_GRAMMAR_WAYS + way)*stride;
        if (slot[0]) { continue; }
        slot[1] = hash;
        slot[2] = status[0];
        for (int i = 0; i < GGML_GRAMMAR_KEY_WORDS; ++i) { slot[4 + i] = key[i]; }
        for (int i = 0; i < words; ++i) { slot[4 + GGML_GRAMMAR_KEY_WORDS + i] = mask[i]; }
        __threadfence();
        atomicExch(slot, 1u);
        break;
    }
}

bool ggml_cuda_grammar_compute(ggml_backend_cuda_context & ctx, const ggml_tensor * program, const ggml_tensor * vocab,
        const ggml_tensor * state, ggml_tensor * scratch, ggml_tensor * cache, ggml_tensor * mask, ggml_tensor * status) {
    ggml_cuda_set_device(ctx.device);
    const int words = mask->ne[0];
    grammar_lookup<<<1, 1, 0, ctx.stream()>>>(static_cast<const uint32_t *>(state->data), static_cast<uint32_t *>(cache->data),
        static_cast<uint32_t *>(mask->data), static_cast<uint32_t *>(status->data), words);
    grammar_traverse<<<1, GGML_GRAMMAR_WORKERS, 0, ctx.stream()>>>(static_cast<const uint32_t *>(program->data), static_cast<const uint32_t *>(vocab->data),
        static_cast<const uint32_t *>(state->data), static_cast<ggml_grammar_worker *>(scratch->data), static_cast<uint32_t *>(mask->data), static_cast<uint32_t *>(status->data));
    grammar_publish<<<1, 1, 0, ctx.stream()>>>(static_cast<const uint32_t *>(state->data), static_cast<uint32_t *>(cache->data),
        static_cast<const uint32_t *>(mask->data), static_cast<uint32_t *>(status->data), words);
    return cudaGetLastError() == cudaSuccess && cudaStreamSynchronize(ctx.stream()) == cudaSuccess;
}
