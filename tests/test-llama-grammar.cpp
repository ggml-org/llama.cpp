#ifdef NDEBUG
#undef NDEBUG
#endif

#include "llama.h"

#include "../src/llama-grammar.h"

#include <cassert>
#include <stdexcept>

int main()
{
    llama_grammar_parser parsed_grammar;

    std::vector<std::pair<std::string, uint32_t>> expected = {
        {"expr", 2},
        {"expr_6", 6},
        {"expr_7", 7},
        {"ident", 8},
        {"ident_10", 10},
        {"num", 9},
        {"num_11", 11},
        {"root", 0},
        {"root_1", 1},
        {"root_5", 5},
        {"term", 4},
        {"ws", 3},
        {"ws_12", 12},
    };

    std::vector<std::vector<llama_grammar_element>> expected_rules = {
        {{LLAMA_GRETYPE_RULE_REF, 5}, {LLAMA_GRETYPE_END, 0}},
        {
            {LLAMA_GRETYPE_RULE_REF, 2},
            {LLAMA_GRETYPE_CHAR, 61},
            {LLAMA_GRETYPE_RULE_REF, 3},
            {LLAMA_GRETYPE_RULE_REF, 4},
            {LLAMA_GRETYPE_CHAR, 10},
            {LLAMA_GRETYPE_END, 0},
        },
        {{LLAMA_GRETYPE_RULE_REF, 4}, {LLAMA_GRETYPE_RULE_REF, 7}, {LLAMA_GRETYPE_END, 0}},
        {{LLAMA_GRETYPE_RULE_REF, 12}, {LLAMA_GRETYPE_END, 0}},
        {
            {LLAMA_GRETYPE_RULE_REF, 8},
            {LLAMA_GRETYPE_ALT, 0},
            {LLAMA_GRETYPE_RULE_REF, 9},
            {LLAMA_GRETYPE_ALT, 0},
            {LLAMA_GRETYPE_CHAR, 40},
            {LLAMA_GRETYPE_RULE_REF, 3},
            {LLAMA_GRETYPE_RULE_REF, 2},
            {LLAMA_GRETYPE_CHAR, 41},
            {LLAMA_GRETYPE_RULE_REF, 3},
            {LLAMA_GRETYPE_END, 0},
        },
        {{LLAMA_GRETYPE_RULE_REF, 1}, {LLAMA_GRETYPE_RULE_REF, 5}, {LLAMA_GRETYPE_ALT, 0}, {LLAMA_GRETYPE_RULE_REF, 1}, {LLAMA_GRETYPE_END, 0}},
        {
            {LLAMA_GRETYPE_CHAR, 45},
            {LLAMA_GRETYPE_CHAR_ALT, 43},
            {LLAMA_GRETYPE_CHAR_ALT, 42},
            {LLAMA_GRETYPE_CHAR_ALT, 47},
            {LLAMA_GRETYPE_RULE_REF, 4},
            {LLAMA_GRETYPE_END, 0},
        },
        {{LLAMA_GRETYPE_RULE_REF, 6}, {LLAMA_GRETYPE_RULE_REF, 7}, {LLAMA_GRETYPE_ALT, 0}, {LLAMA_GRETYPE_END, 0}},
        {
            {LLAMA_GRETYPE_CHAR, 97},
            {LLAMA_GRETYPE_CHAR_RNG_UPPER, 122},
            {LLAMA_GRETYPE_RULE_REF, 10},
            {LLAMA_GRETYPE_RULE_REF, 3},
            {LLAMA_GRETYPE_END, 0},
        },
        {{LLAMA_GRETYPE_RULE_REF, 11}, {LLAMA_GRETYPE_RULE_REF, 3}, {LLAMA_GRETYPE_END, 0}},
        {
            {LLAMA_GRETYPE_CHAR, 97},
            {LLAMA_GRETYPE_CHAR_RNG_UPPER, 122},
            {LLAMA_GRETYPE_CHAR_ALT, 48},
            {LLAMA_GRETYPE_CHAR_RNG_UPPER, 57},
            {LLAMA_GRETYPE_CHAR_ALT, 95},
            {LLAMA_GRETYPE_RULE_REF, 10},
            {LLAMA_GRETYPE_ALT, 0},
            {LLAMA_GRETYPE_END, 0},
        },
        {
            {LLAMA_GRETYPE_CHAR, 48},
            {LLAMA_GRETYPE_CHAR_RNG_UPPER, 57},
            {LLAMA_GRETYPE_RULE_REF, 11},
            {LLAMA_GRETYPE_ALT, 0},
            {LLAMA_GRETYPE_CHAR, 48},
            {LLAMA_GRETYPE_CHAR_RNG_UPPER, 57},
            {LLAMA_GRETYPE_END, 0},
        },
        {
            {LLAMA_GRETYPE_CHAR, 32},
            {LLAMA_GRETYPE_CHAR_ALT, 9},
            {LLAMA_GRETYPE_CHAR_ALT, 10},
            {LLAMA_GRETYPE_RULE_REF, 12},
            {LLAMA_GRETYPE_ALT, 0},
            {LLAMA_GRETYPE_END, 0},
        },
    };

    for (auto pair : expected)
    {
        parsed_grammar.symbol_ids[pair.first] = pair.second;
    }

    for (auto rule : expected_rules)
    {
        parsed_grammar.rules.emplace_back();
        for (auto element : rule)
        {
            parsed_grammar.rules.back().push_back(element);
        }
    }

    std::vector<const llama_grammar_element *> grammar_rules(parsed_grammar.c_rules());

    llama_grammar * grammar = llama_grammar_init_impl(nullptr, grammar_rules.data(), grammar_rules.size(), parsed_grammar.symbol_ids.at("root"));
    if (grammar == nullptr) {
        throw std::runtime_error("Failed to initialize llama_grammar");
    }

    std::vector<std::vector<llama_grammar_element>> expected_stacks = {
        {
            {LLAMA_GRETYPE_CHAR, 61},
            {LLAMA_GRETYPE_RULE_REF, 7},
            {LLAMA_GRETYPE_CHAR, 40},
        },
        {
            {LLAMA_GRETYPE_CHAR, 61},
            {LLAMA_GRETYPE_RULE_REF, 7},
            {LLAMA_GRETYPE_RULE_REF, 3},
            {LLAMA_GRETYPE_CHAR, 48},
        },
        {
            {LLAMA_GRETYPE_CHAR, 61},
            {LLAMA_GRETYPE_RULE_REF, 7},
            {LLAMA_GRETYPE_RULE_REF, 3},
            {LLAMA_GRETYPE_CHAR, 48},
        },
        {
            {LLAMA_GRETYPE_CHAR, 61},
            {LLAMA_GRETYPE_RULE_REF, 7},
            {LLAMA_GRETYPE_CHAR, 97},
        },
        {
            {LLAMA_GRETYPE_RULE_REF, 5},
            {LLAMA_GRETYPE_CHAR, 61},
            {LLAMA_GRETYPE_RULE_REF, 7},
            {LLAMA_GRETYPE_CHAR, 40},
        },
        {
            {LLAMA_GRETYPE_RULE_REF, 5},
            {LLAMA_GRETYPE_CHAR, 61},
            {LLAMA_GRETYPE_RULE_REF, 7},
            {LLAMA_GRETYPE_RULE_REF, 3},
            {LLAMA_GRETYPE_CHAR, 48},
        },
        {
            {LLAMA_GRETYPE_RULE_REF, 5},
            {LLAMA_GRETYPE_CHAR, 61},
            {LLAMA_GRETYPE_RULE_REF, 7},
            {LLAMA_GRETYPE_RULE_REF, 3},
            {LLAMA_GRETYPE_CHAR, 48},
        },
        {
            {LLAMA_GRETYPE_RULE_REF, 5},
            {LLAMA_GRETYPE_CHAR, 61},
            {LLAMA_GRETYPE_RULE_REF, 7},
            {LLAMA_GRETYPE_CHAR, 97},
        }};

    auto index = 0;
    for (const llama_grammar_stack & stack : llama_grammar_get_stacks(grammar))
    {
        // compare stack to expected_stack
        for (uint32_t i = 0; i < stack.size(); i++)
        {
            const llama_grammar_element * element = stack[i];
            const llama_grammar_element & expected_element = expected_stacks[index][i];

            // pretty print error message before asserting
            if (expected_element.type != element->type || expected_element.value != element->value)
            {
                fprintf(stderr, "index: %d\n", index);
                fprintf(stderr, "expected_element: %d, %u\n", expected_element.type, expected_element.value);
                fprintf(stderr, "actual_element: %d, %u\n", element->type, element->value);
                fprintf(stderr, "expected_element != actual_element\n");
            }

            assert(expected_element.type == element->type && expected_element.value == element->value);
        }
        index++;
    }

    std::vector<llama_grammar_candidate> next_candidates;
    next_candidates.resize(23);

    for (size_t i = 0; i < 23; ++i)
    {
        uint32_t *cp = new uint32_t[2]; // dynamically allocate memory for code_point
        cp[0] = 37 + i;
        cp[1] = 0;
        next_candidates[i] = {i, cp, {}, 0, false};
    }

    std::vector<std::vector<std::pair<uint32_t, uint16_t>>> expected_reject = {
        {
            {0, 37},
            {1, 38},
            {2, 39},
            {4, 41},
            {5, 42},
            {6, 43},
            {7, 44},
            {8, 45},
            {9, 46},
            {10, 47},
            {11, 48},
            {12, 49},
            {13, 50},
            {14, 51},
            {15, 52},
            {16, 53},
            {17, 54},
            {18, 55},
            {19, 56},
            {20, 57},
            {21, 58},
            {22, 59},
            {23, 60},
        },
        {
            {0, 37},
            {1, 38},
            {2, 39},
            {3, 40},
            {4, 41},
            {5, 42},
            {6, 43},
            {7, 44},
            {8, 45},
            {9, 46},
            {10, 47},
            {21, 58},
            {22, 59},
            {23, 60},
        },
        {
            {0, 37},
            {1, 38},
            {2, 39},
            {3, 40},
            {4, 41},
            {5, 42},
            {6, 43},
            {7, 44},
            {8, 45},
            {9, 46},
            {10, 47},
            {21, 58},
            {22, 59},
            {23, 60},
        },
        {
            {0, 37},
            {1, 38},
            {2, 39},
            {3, 40},
            {4, 41},
            {5, 42},
            {6, 43},
            {7, 44},
            {8, 45},
            {9, 46},
            {10, 47},
            {11, 48},
            {12, 49},
            {13, 50},
            {14, 51},
            {15, 52},
            {16, 53},
            {17, 54},
            {18, 55},
            {19, 56},
            {20, 57},
            {21, 58},
            {22, 59},
        },
        {
            {0, 37},
            {1, 38},
            {2, 39},
            {4, 41},
            {5, 42},
            {6, 43},
            {7, 44},
            {8, 45},
            {9, 46},
            {10, 47},
            {11, 48},
            {12, 49},
            {13, 50},
            {14, 51},
            {15, 52},
            {16, 53},
            {17, 54},
            {18, 55},
            {19, 56},
            {20, 57},
            {21, 58},
            {22, 59},
            {23, 60},
        },
        {
            {0, 37},
            {1, 38},
            {2, 39},
            {3, 40},
            {4, 41},
            {5, 42},
            {6, 43},
            {7, 44},
            {8, 45},
            {9, 46},
            {10, 47},
            {21, 58},
            {22, 59},
            {23, 60},
        },
        {
            {0, 37},
            {1, 38},
            {2, 39},
            {3, 40},
            {4, 41},
            {5, 42},
            {6, 43},
            {7, 44},
            {8, 45},
            {9, 46},
            {10, 47},
            {21, 58},
            {22, 59},
            {23, 60},
        },
        {
            {0, 37},
            {1, 38},
            {2, 39},
            {3, 40},
            {4, 41},
            {5, 42},
            {6, 43},
            {7, 44},
            {8, 45},
            {9, 46},
            {10, 47},
            {11, 48},
            {12, 49},
            {13, 50},
            {14, 51},
            {15, 52},
            {16, 53},
            {17, 54},
            {18, 55},
            {19, 56},
            {20, 57},
            {21, 58},
            {22, 59},
        },
    };

    std::vector<llama_grammar_candidate> rejects = llama_grammar_reject_candidates_for_stack(llama_grammar_get_rules(grammar), llama_grammar_get_stacks(grammar)[0], next_candidates);

    std::vector<std::vector<llama_grammar_candidate>> all_rejects;

    for (std::size_t count = 0; count < llama_grammar_get_stacks(grammar).size(); ++count)
    {
        rejects = llama_grammar_reject_candidates_for_stack(llama_grammar_get_rules(grammar), llama_grammar_get_stacks(grammar)[count], next_candidates);
        all_rejects.push_back(rejects);
    }

    index = 0;
    for (auto rej : all_rejects)
    {
        for (uint32_t i = 0; i < rej.size(); i++)
        {
            auto element = rej[i];
            auto expected_element = expected_reject[index][i];
            assert(element.index == expected_element.first && *element.code_points == expected_element.second);
        }
        index++;
    }

    for (auto &candidate : next_candidates)
    {
        delete[] candidate.code_points;
        candidate.code_points = nullptr;
    }

    llama_grammar_free_impl(grammar);

    // Test lazy grammar trigger replay rollback on false triggers (e.g. tool markup in code/prose)
    {
        const char * grammar_str =
            "root ::= \"<function=python_repl>\\n<parameter=code>\\n\" [a-z]+ \"\\n</parameter>\\n</function>\"";
        const char * trigger_pattern = "<function=python_repl>";

        llama_grammar * lazy_g = llama_grammar_init_impl(
            nullptr, grammar_str, "root", true, &trigger_pattern, 1, nullptr, 0);
        assert(lazy_g != nullptr);
        assert(lazy_g->awaiting_trigger);

        const size_t initial_stacks = lazy_g->stacks.size();

        // 1. False trigger inside code example: token ">\\" completes pattern match with invalid suffix
        llama_grammar_accept_piece_impl(*lazy_g, 1, "assert xml == '''");
        assert(lazy_g->awaiting_trigger);
        llama_grammar_accept_piece_impl(*lazy_g, 2, "<function=python_repl");
        assert(lazy_g->awaiting_trigger);
        llama_grammar_accept_piece_impl(*lazy_g, 3, ">\\");
        assert(lazy_g->awaiting_trigger);
        assert(lazy_g->stacks.size() == initial_stacks);
        assert(lazy_g->partial_utf8.n_remain == 0);

        // 2. Subsequent valid trigger should still activate the grammar cleanly
        llama_grammar_accept_piece_impl(*lazy_g, 4, "print(1)'''\n");
        assert(lazy_g->awaiting_trigger);
        llama_grammar_accept_piece_impl(*lazy_g, 5, "<function=python_repl");
        assert(lazy_g->awaiting_trigger);
        llama_grammar_accept_piece_impl(*lazy_g, 6, ">\n<parameter=code>\n");
        assert(!lazy_g->awaiting_trigger);
        assert(lazy_g->trigger_buffer.empty());
        assert(lazy_g->trigger_buffer_positions.empty());

        llama_grammar_accept_piece_impl(*lazy_g, 7, "abc\n</parameter>\n</function>");
        bool completed = false;
        for (const auto & s : lazy_g->stacks) {
            if (s.empty()) {
                completed = true;
                break;
            }
        }
        assert(completed);
        llama_grammar_free_impl(lazy_g);
    }

    // Test false trigger followed by valid trigger within the same buffered sequence & UTF-8 safety
    {
        const char * grammar_str = "root ::= \"<function=ok>\\n\" [a-z]+";
        const char * trigger_pattern = "<function=";

        llama_grammar * lazy_g = llama_grammar_init_impl(
            nullptr, grammar_str, "root", true, &trigger_pattern, 1, nullptr, 0);
        assert(lazy_g != nullptr);

        // Multi-byte UTF-8 character in false trigger prefix must not be split
        llama_grammar_accept_piece_impl(*lazy_g, 1, "<function=中");
        assert(lazy_g->awaiting_trigger);

        // False trigger immediately followed by valid trigger in the same token
        llama_grammar_accept_piece_impl(*lazy_g, 2, "<function=bad><function=ok>\n");
        assert(!lazy_g->awaiting_trigger);
        assert(lazy_g->trigger_buffer.empty());

        llama_grammar_accept_piece_impl(*lazy_g, 3, "xyz");
        bool completed = false;
        for (const auto & s : lazy_g->stacks) {
            if (s.empty()) {
                completed = true;
                break;
            }
        }
        assert(completed);
        llama_grammar_free_impl(lazy_g);
    }

    // Test lazy trigger_tokens rollback when trigger token does not match grammar root
    {
        const char * grammar_str = "root ::= \"<tool>\" [a-z]+ \"</tool>\"";
        const llama_token trig_tok = 42;

        llama_grammar * lazy_g = llama_grammar_init_impl(
            nullptr, grammar_str, "root", true, nullptr, 0, &trig_tok, 1);
        assert(lazy_g != nullptr);
        assert(lazy_g->awaiting_trigger);

        llama_grammar_accept_piece_impl(*lazy_g, 1, "hello ");
        assert(lazy_g->awaiting_trigger);

        // Rejected trigger token stays awaiting_trigger without throwing
        llama_grammar_accept_piece_impl(*lazy_g, trig_tok, "<bad>");
        assert(lazy_g->awaiting_trigger);

        // Accepted trigger token activates grammar and clears buffers
        llama_grammar_accept_piece_impl(*lazy_g, trig_tok, "<tool>");
        assert(!lazy_g->awaiting_trigger);
        assert(lazy_g->trigger_buffer.empty());
        assert(lazy_g->trigger_buffer_positions.empty());

        llama_grammar_accept_piece_impl(*lazy_g, 2, "hi</tool>");
        bool completed = false;
        for (const auto & s : lazy_g->stacks) {
            if (s.empty()) {
                completed = true;
                break;
            }
        }
        assert(completed);
        llama_grammar_free_impl(lazy_g);
    }

    return 0;
}
