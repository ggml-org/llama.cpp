#pragma once

#include "ggml-backend.h"
#include <cstdint>

static constexpr uint32_t GGML_GRAMMAR_NONE = UINT32_MAX;
static constexpr int GGML_GRAMMAR_STACKS = 32;
static constexpr int GGML_GRAMMAR_DEPTH = 16;
static constexpr int GGML_GRAMMAR_LEVELS = 64;
static constexpr int GGML_GRAMMAR_WORKERS = 32;
static constexpr int GGML_GRAMMAR_SETS = 2;
static constexpr int GGML_GRAMMAR_WAYS = 4;
static constexpr int GGML_GRAMMAR_KEY_WORDS = 8 + GGML_GRAMMAR_STACKS*(GGML_GRAMMAR_DEPTH + 1);

enum ggml_grammar_opcode : uint32_t {
    GGML_GRAMMAR_END = 0, GGML_GRAMMAR_ALT = 1, GGML_GRAMMAR_RULE_REF = 2,
    GGML_GRAMMAR_CHAR = 3, GGML_GRAMMAR_CHAR_NOT = 4, GGML_GRAMMAR_CHAR_ANY = 7,
};
enum ggml_grammar_token_flags : uint32_t { GGML_GRAMMAR_EOG = 1, GGML_GRAMMAR_SPECIAL = 2 };
enum ggml_grammar_status { GGML_GRAMMAR_OK, GGML_GRAMMAR_NEED_CPU, GGML_GRAMMAR_NO_VALID_TOKEN };
struct ggml_grammar_program_header { uint32_t id[2], n_ops, n_rules, ops, rules, alts, ranges; };
struct ggml_grammar_op { uint32_t type, value, count, next; };
struct ggml_grammar_rule { uint32_t start, count; };
struct ggml_grammar_range { uint32_t low, high; };
struct ggml_grammar_vocab_header { uint32_t id[2], n_vocab, n_nodes, nodes, terminals, tokens, bytes, roots[256]; };
struct ggml_grammar_node { uint32_t child, sibling, terminal, byte; };
struct ggml_grammar_terminal { uint32_t id, next; };
struct ggml_grammar_token { uint32_t offset, length, flags; };
struct ggml_grammar_stack { uint32_t size, frames[GGML_GRAMMAR_DEPTH]; };
struct ggml_grammar_state {
    uint32_t count, value;
    int32_t remain;
    uint32_t strict, nul, stopped;
    ggml_grammar_stack stacks[GGML_GRAMMAR_STACKS];
};
struct ggml_grammar_level { uint32_t child; ggml_grammar_state state; };
struct ggml_grammar_worker {
    ggml_grammar_level levels[GGML_GRAMMAR_LEVELS];
    ggml_grammar_stack todo[GGML_GRAMMAR_STACKS];
    ggml_grammar_state next;
};
using ggml_backend_grammar_compute = bool (*)(ggml_backend_t, const ggml_tensor *, const ggml_tensor *,
        const ggml_tensor *, ggml_tensor *, ggml_tensor *, ggml_tensor *, ggml_tensor *);
