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
        next_candidates[i] = {i, cp, {}, 0};
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

    // Memoization correctness: verify that repeated calls with equivalent candidate lists
    // (same content, different pointer addresses) produce identical reject sets.
    {
        // digit grammar: only digits 0-9 are valid at the start
        const char * digit_grammar = "root ::= [0-9]+";
        struct llama_grammar * dg = llama_grammar_init_impl(
            nullptr, digit_grammar, "root", false, nullptr, 0, nullptr, 0);
        assert(dg != nullptr);

        const llama_grammar_rules  & drules  = llama_grammar_get_rules(dg);
        const llama_grammar_stacks & dstacks = llama_grammar_get_stacks(dg);
        assert(!dstacks.empty());

        // Build the same 2-candidate list twice using different allocations.
        // Candidate 0: 'a' (97) — should be rejected by [0-9].
        // Candidate 1: '0' (48) — should be accepted.
        auto make_digit_cands = [](std::vector<uint32_t *> & allocs) {
            std::vector<llama_grammar_candidate> cands;
            uint32_t * cpa = new uint32_t[2]; cpa[0] = 97; cpa[1] = 0;
            uint32_t * cpz = new uint32_t[2]; cpz[0] = 48; cpz[1] = 0;
            allocs.push_back(cpa);
            allocs.push_back(cpz);
            llama_partial_utf8 pu{0, 0};
            cands.push_back({0, cpa, pu, 0});
            cands.push_back({1, cpz, pu, 0});
            return cands;
        };

        std::vector<uint32_t *> alloc_a, alloc_b;
        auto cands_a = make_digit_cands(alloc_a);
        auto cands_b = make_digit_cands(alloc_b);

        // The two allocations must be at different addresses (content-based key test).
        assert(cands_a[0].code_points != cands_b[0].code_points);

        const llama_grammar_stack & ds = dstacks[0];

        auto rejects_a = llama_grammar_reject_candidates_for_stack(drules, ds, cands_a);
        auto rejects_b = llama_grammar_reject_candidates_for_stack(drules, ds, cands_b);

        // Both calls must reject exactly the same candidates.
        assert(rejects_a.size() == rejects_b.size());
        for (size_t i = 0; i < rejects_a.size(); i++) {
            assert(rejects_a[i].index == rejects_b[i].index);
        }
        // 'a' (index 0) must be the only reject; '0' (index 1) must pass.
        assert(rejects_a.size() == 1);
        assert(rejects_a[0].index == 0);

        // A third call with the same list must give the same result (cache consistency).
        auto rejects_c = llama_grammar_reject_candidates_for_stack(drules, ds, cands_a);
        assert(rejects_c.size() == rejects_a.size());
        for (size_t i = 0; i < rejects_c.size(); i++) {
            assert(rejects_c[i].index == rejects_a[i].index);
        }

        for (auto * p : alloc_a) { delete[] p; }
        for (auto * p : alloc_b) { delete[] p; }
        llama_grammar_free_impl(dg);
    }

    return 0;
}
