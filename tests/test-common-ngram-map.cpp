// tests/test-common-ngram-map.cpp
//
// Unit tests of common/ngram-map.cpp.
//
#include "log.h"
#include "ngram-map.h"

#include <cassert>
#include <cstdio>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

/**
 * Context Window Evolution:
 *
 * Time | Window Content                             | Action
 * -----|--------------------------------------------|------------------
 * t1   | [U1]                                       | PP: user input
 * t2   | [U1][R1      ]                             | TG: reasoning
 * t3   | [U1][R1      ][A1 ]                        | TG: answer
 * t4   | [U1][A1 ]                                  | deletion of reasoning
 * t5   | [U1][A1 ][U2]                              | PP: user input
 * t6   | [U1][A1 ][U2][R2          ]                | TG: reasoning
 * t7   | [U1][A1 ][U2][R2          ][A2  ]          | TG: answer
 * t8   | [U1][A1 ][U2][A2  ]                        | deletion of reasoning
 * t9   | [U1][A1 ][U2][A2  ][U3]                    | PP: user input
 *                     ^ size_last_begin
 *                               ^ size_begin
 *
 * Legend: U=User, R=Reasoning, A=Assistant
 * U is prompt-processing.
 * R and A are generation (with speculative decoding).
 */

// Helper to convert a string to llama_tokens using characters.
static llama_tokens string_to_tokens(const std::string & str) {
    llama_tokens tokens;
    tokens.reserve(str.size());
    for (unsigned char c : str) {
        tokens.push_back(static_cast<llama_token>(c));
    }
    return tokens;
}

// Helper to convert tokens to string representation.
static std::string tokens_to_string(const llama_tokens & tokens) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < tokens.size(); ++i) {
        char c = (tokens[i] < 32 || tokens[i] > 126) ?  '.' : static_cast<char>(tokens[i]);
        oss << c;
    }
    oss << "]";
    return oss.str();
}


// base class of an adapter of a speculative implementation.
class spec_impl_adapter {
public:
    virtual ~spec_impl_adapter() = default;

    // generate draft
    virtual llama_tokens draft(const llama_tokens & context, llama_token sampled) = 0;

    // send number of accepted tokens of previous draft
    virtual void accept(size_t n_accepted) = 0;

    // marks the begin of a new user prompt
    virtual void begin(llama_tokens & context) = 0;

    // gets details of the current draft state
    virtual std::string get_state(llama_tokens & context) = 0;
};

class spec_test_builder {
    struct action {
        enum type {
            ADD_USER,      // prompt processing
            ADD_REASONING, // generation, reasoning
            END_REASONING, // generation, mark end of reasoning
            ADD_ASSISTANT, // generation, assistant
            EXPECT_DRAFT,  // expect draft (may be accepted, partially accepted or rejected)
            DELETE_RANGE,  // delete a previous reasoning block
            DELETE_ALL,    // delete the entire context window
            DELETE_TAIL,   // delete the tail of the context window
            CALL_BEGIN,    // starts a new generation
            PRINT_STATE    // print current state
        };
        type type;
        std::string id;
        std::string text;
        std::string text_rej;
        size_t      idx_tokens = 0; // index of this action in llama-tokens of context
        std::string to_string() {
            std::string t;
            switch (type) {
                case ADD_USER:      t = "ADD_USER"; break;
                case ADD_REASONING: t = "ADD_REASONING"; break;
                case END_REASONING: t = "END_REASONING"; break;
                case ADD_ASSISTANT: t = "ADD_ASSISTANT"; break;
                case EXPECT_DRAFT:  t = "EXPECT_DRAFT"; break;
                case DELETE_RANGE:  t = "DELETE_RANGE"; break;
                case DELETE_ALL:    t = "DELETE_ALL"; break;
                case DELETE_TAIL:   t = "DELETE_TAIL"; break;
                case CALL_BEGIN:    t = "CALL_BEGIN"; break;
                case PRINT_STATE:   t = "PRINT_STATE"; break;
            }
            return t + "[id='" + id + "' text='"
                + (text.size() > 80 ? text.substr(0, 60) + "[...]" + text.substr(text.size() - 20) : text)
                + "' text_rej='" + text_rej + "' idx=" + std::to_string(idx_tokens) + "]";
        }
    };
    std::vector<action> actions;

public:
    spec_test_builder() = default;

    spec_test_builder & add_user_input(const std::string & text) {
        actions.push_back({action::ADD_USER, "#user#", text, ""});
        return *this;
    }

    spec_test_builder & add_reasoning(const std::string & id, const std::string & text) {
        actions.push_back({action::ADD_REASONING, id, text, ""});
        return *this;
    }

    spec_test_builder & end_reasoning(const std::string & id) {
        actions.push_back({action::END_REASONING, id, "", ""});
        return *this;
    }

    spec_test_builder & add_assistant(const std::string & text) {
        actions.push_back({action::ADD_ASSISTANT, "#gen#", text, ""});
        return *this;
    }

    spec_test_builder & expect_accepted(const std::string & text) {
        actions.push_back({action::EXPECT_DRAFT, "#accept#", text, ""});
        return *this;
    }

    spec_test_builder & expect_partial(const std::string & acc, const std::string & rej) {
        actions.push_back({action::EXPECT_DRAFT, "#partial#", acc, rej});
        return *this;
    }

    spec_test_builder & expect_rejected(const std::string & rej) {
        actions.push_back({action::EXPECT_DRAFT, "#reject#", "", rej});
        return *this;
    }

    spec_test_builder & delete_reasoning(const std::string & id) {
        actions.push_back({action::DELETE_RANGE, id, "", ""});
        return *this;
    }

    spec_test_builder & delete_all() {
        actions.push_back({action::DELETE_ALL, "", "", ""});
        return *this;
    }

    spec_test_builder & delete_tail(const size_t & num) {
        actions.push_back({action::DELETE_TAIL, "#del_tail#", std::to_string(num), ""});
        return *this;
    }

    spec_test_builder & call_begin() {
        actions.push_back({action::CALL_BEGIN, "#begin#", "", ""});
        return *this;
    }

    spec_test_builder & print_state() {
        actions.push_back({action::PRINT_STATE, "#print#", "", ""});
        return *this;
    }

    int run(const std::string & test_name, spec_impl_adapter & impl) {
        // Initialize the map with the token history
        std::cout << "--- " << test_name << ": Calling begin..." << std::endl;
        llama_tokens current_context;
        impl.begin(current_context);
        common_log_flush(common_log_main());

        // Test common_ngram_map_draft in a loop over all tokens
        std::cout << "--- " << test_name << ": Starting draft loop over tokens..." << std::endl;

        size_t mismatch_count = 0;
        llama_tokens draft = {};

        size_t i = 0;
        while (i < actions.size()) {
            action & act = actions[i];
            act.idx_tokens = current_context.size();
            //std::cout << "DBG:   action[" << i << "]: " << act.to_string() << std::endl;
            switch (act.type) {
                case action::ADD_USER:
                    {
                        current_context.insert(current_context.end(), act.text.begin(), act.text.end());
                        break;
                    }
                case action::ADD_REASONING:
                case action::ADD_ASSISTANT:
                    {

                        const size_t len = act.text.size();
                        for (size_t j = 0; j < len; j++) {
                            // call draft method
                            draft = impl.draft(current_context, static_cast<llama_token>(act.text[j]));
                            common_log_flush(common_log_main());
                            if (!draft.empty() && j + 1 < len) {
                                mismatch_count++;
                                std::cout << "INFO:  action[" << i << "]: " << act.to_string() << "@" << j << std::endl;
                                std::cout << "ERROR: unexpected draft tokens at idx " << current_context.size() << ": " << tokens_to_string(draft) << std::endl;
                                break;
                            }
                            current_context.push_back(act.text[j]);
                        }
                        break;
                    }
                case action::END_REASONING:
                    {
                        // nothing to do, the tokens-index has been stored
                        break;
                    }
                case action::EXPECT_DRAFT:
                    {
                        // check draft
                        // The draft should consist of expected and rejected tokens.
                        llama_tokens expected = string_to_tokens(act.text);
                        llama_tokens expected_rej = string_to_tokens(act.text_rej);
                        if (draft.size() != expected.size() + expected_rej.size()) {
                            mismatch_count++;
                            std::cout << "INFO:  action[" << i << "]: " << act.to_string() << ", draft-size=" << draft.size() << ", expected-size=" << expected.size() << ", rejected-size=" << expected_rej.size() << std::endl;
                            std::cout << "ERROR: draft size mismatch, got: draft_size=" << draft.size() << ", '" << tokens_to_string(draft) << "'" << std::endl;
                            break;
                        }
                        // compare draft and tokens of act.text.
                        for (size_t j = 0; j < expected.size(); j++) {
                            if (draft[j] != expected[j]) {
                                mismatch_count++;
                                std::cout << "INFO:  action[" << i << "]: " << act.to_string() << "@" << j << std::endl;
                                std::cout << "ERROR: draft token mismatch at index " << j << ", expected: " << act.text << ", draft: " << tokens_to_string(draft) << std::endl;
                                break;
                            }
                        }
                        // compare tokens of act.text_rej.
                        for (size_t j = 0; j < expected_rej.size(); j++) {
                            if (draft[expected.size() + j] != expected_rej[j]) {
                                mismatch_count++;
                                std::cout << "INFO:  action[" << i << "]: " << act.to_string() << "@" << (expected.size() + j) << std::endl;
                                std::cout << "ERROR: rejected token mismatch at index " << expected.size() + j << ", expected rej: " << act.text_rej << ", draft: " << tokens_to_string(draft) << std::endl;
                                break;
                            }
                        }
                        impl.accept(expected.size());
                        if (expected.size() > 0) {
                            current_context.insert(current_context.end(), expected.begin(), expected.end() - 1);
                            draft = impl.draft(current_context, static_cast<llama_token>(expected.back()));
                            current_context.push_back(expected.back());
                        } else {
                            draft.clear();
                        }

                        break;
                    }
                case action::DELETE_RANGE:
                    {
                        // find first and last (reasoning) action with id, compute offset in token list
                        bool found = false;
                        size_t idx_action_first = 0;
                        size_t idx_action_last = 0;
                        for (size_t j = 0; j < i; j++) {
                            if (actions[j].id == act.id) {
                                if (!found) {
                                    found = true;
                                    idx_action_first = j;
                                }
                                idx_action_last = j;
                            }
                        }
                        if (!found) {
                            GGML_ABORT("missing reasoning block with id: %s", act.id.c_str());
                        }
                        if (idx_action_first == idx_action_last) {
                            GGML_ABORT("delete_range: first and last el should be different, idx_action=%zu, id=%s", idx_action_first, act.id.c_str());
                        }
                        size_t num_tokens = actions[idx_action_last].idx_tokens - actions[idx_action_first].idx_tokens;
                        // std::cout << "DBG:   DELETE_RANGE, idx_action_first=" << idx_action_first << ", idx_action_last=" << idx_action_last << ", num_tokens=" << num_tokens << std::endl;
                        // delete tokens in context window
                        current_context.erase(current_context.begin() + actions[idx_action_first].idx_tokens,
                                              current_context.begin() + actions[idx_action_last ].idx_tokens);
                        // adjust token indices
                        for (size_t j = idx_action_last + 1; j < actions.size(); j++) {
                            actions[j].idx_tokens -= num_tokens;
                        }
                        break;
                    }
                case action::DELETE_ALL:
                    {
                        size_t num_tokens = current_context.size();
                        current_context = llama_tokens();
                        // reserve(14000) was used to reproduce the OOB in test_ngram_map_key_pr23936_oob before the fix.
                        current_context.reserve(14000);
                        for (size_t j = 0; j < actions.size(); j++) {
                            if (actions[j].idx_tokens <= num_tokens) {
                                actions[j].idx_tokens = 0;
                            } else {
                                actions[j].idx_tokens -= num_tokens;
                            }
                        }
                        break;
                    }
                case action::DELETE_TAIL:
                    {
                        size_t size_cc = current_context.size();
                        size_t size_tail = std::stoul(act.text);
                        std::cout << "DBG: DELETE_TAIL: context size " << size_cc << " -> " << (size_cc - size_tail) << std::endl;
                        current_context.erase(current_context.end() - size_tail, current_context.end());
                        break;
                    }
                case action::CALL_BEGIN:
                    {
                        impl.begin(current_context);
                        common_log_flush(common_log_main()); // flush LOG
                        break;
                    }
                case action::PRINT_STATE:
                    {
                        std::cout << "INFO: STATE: " << impl.get_state(current_context) << std::endl;
                        break;
                    }
                default:
                    {
                        GGML_ABORT("unknown action type");
                    }

            }
            i++;
            if (mismatch_count > 0) {
                break;
            }
        }
        return mismatch_count;
    }

};


class ngram_map_simple_adapter : public spec_impl_adapter {

    common_ngram_simple_config config;
public:
    ngram_map_simple_adapter(common_ngram_simple_config cfg) : config(cfg) {}
    ~ngram_map_simple_adapter() override = default;

    llama_tokens draft(const llama_tokens & context, llama_token sampled) override {
        return common_ngram_simple_draft(config, context, sampled);
    }

    void accept(size_t /*n_accepted*/) override {
        // nothing to do
    }

    void begin(llama_tokens & /*context*/) override {
        // nothing to do
    }

    std::string get_state(llama_tokens & /*context*/) override {
        return "no state";
    }
};

class ngram_map_adapter : public spec_impl_adapter {
    common_ngram_map & map;
public:
    ngram_map_adapter(common_ngram_map & m) : map(m) {}
    ~ngram_map_adapter() override = default;

    llama_tokens draft(const llama_tokens & context, llama_token sampled) override {
        llama_tokens draft;
        common_ngram_map_draft(map, context, sampled, draft);
        return draft;
    }

    void accept(size_t n_accepted) override {
        common_ngram_map_accept(map, n_accepted);
    }

    void begin(llama_tokens & context) override {
        common_ngram_map_begin(map, context);
    }

    // prime number used for LCG hash function (32 bit), it is near (sqrt(5) - 1)/2 * 2^32.
    // Source: copy of LOG_FACTOR in common/ngram-map.cpp
    #define LCG_FACTOR 2654435761UL

    // Compute the LCG hash of a n-gram of size len at offset start.
    // Source: copy of common_ngram_map_hash in common/ngram-map.cpp
    static uint32_t common_ngram_map_hash(const llama_tokens & tokens, size_t start, size_t len) {
        uint32_t hash = 0;
        for (size_t i = 0; i < len; ++i) {
            hash = hash * LCG_FACTOR + tokens[start + i];
        }
        return hash;
    }

    std::string get_state(llama_tokens & context) override {
        std::ostringstream oss;

        size_t num_keys = map.keys.size();
        oss << "ngram-map: context.size=" << context.size();
        oss << " context.capacity = " << context.capacity();
        oss << ", size_last_begin=" << map.size_last_begin;
        oss << ", last_draft_created=" << map.last_draft_created;
        oss << ", last_draft_key_idx=" << map.last_draft_key_idx;
        oss << ", last_draft_value_idx=" << map.last_draft_value_idx;
        oss << ", idx_last_check=" << map.idx_last_check;
        oss << ", key_map.size=" << map.key_map.size();
        oss << ", key_map_last_idx=" << map.key_map_last_idx;
        oss << " [ #keys=" << num_keys << " ]";


        if (num_keys > 0) {
            auto format_key = [&](size_t idx) {
                const auto& key = map.keys[idx];
                std::ostringstream koss;
                if (key.key_idx < context.size()) {
                    koss << "KeyIdx " << key.key_idx << " Tokens: [";
                    for (uint16_t i = 0; i < map.size_key && (key.key_idx + i) < context.size(); ++i) {
                        koss << context[key.key_idx + i] << (i == map.size_key - 1 ? "" : ",");
                    }
                    koss << "] (hits=" << key.key_num << ")";
                } else {
                    koss << "KeyIdx " << key.key_idx << " [OOB/Invalid]";
                }
                return koss.str();
            };

            oss << "\n  First 3 keys:";
            for (size_t i = 0; i < std::min(num_keys, (size_t)3); ++i) {
                oss << "\n    [" << i << "] " << format_key(i);
            }

            if (num_keys > 3) {
                oss << "\n  Last 3 keys:";
                for (size_t i = std::max((size_t)0, num_keys - 3); i < num_keys; ++i) {
                    oss << "\n    [" << i << "] " << format_key(i);
                }
            }
        }

        if (!map.key_map.empty()) {
            size_t nonzero_count = 0;
            size_t key_val_min = 0;
            size_t key_val_max = 0;
            uint32_t val_min = UINT32_MAX;
            uint32_t val_max = 0;
            for(size_t i = 0; i < map.key_map.size(); i++) {
                uint32_t val = map.key_map[i];
                if (val == 0) {
                    continue;
                }
                nonzero_count++;
                if (val < val_min) {
                    key_val_min = i;
                    val_min = val;
                }
                if (val > val_max) {
                    key_val_max = i;
                    val_max = val;
                }
            }

            oss << "\n  key_map status: #nonzero-entries=" << nonzero_count;
            if (val_max > 0) {
                oss << ", key_map[" << key_val_min << "]=" << val_min;
                oss << ", key_map[" << key_val_max << "]=" << val_max;
            }
            llama_tokens pattern = {80,65,84,84,69,82,78,45,48,51}; // "PATTERN-03"
            uint32_t hash_key = common_ngram_map_hash(pattern, 0, 10) % map.key_map.size();
            oss << ", key_map[" << hash_key << "]=" << map.key_map[hash_key];
        }

        return oss.str();
    }

};

static int test_ngram_simple() {
    const std::string test_name = "ngram_simple_test";

    // Configuration for n-gram simple (expects size_ngram < size_mgram!)
    common_ngram_simple_config simple_config;
    simple_config.size_ngram = 12;
    simple_config.size_mgram = 16;

    // Create adapter with config
    ngram_map_simple_adapter adapter(simple_config);

    spec_test_builder builder;

    // Test (overview):
    //"# common/ngram-map-k, test 1 ---------------\n" // <- line 0, length: 45
    //"* a1:repeated-part-12345678901234567890:1a--\n" // <- line 1
    //"* a2:repeated-part-12345678901234567890:2a--\n"
    //"* a3:repeated-part-12345678901234567890:3a--\n"
    //"* some reasoning here ... wait, ..., yes!?--\n" // <- line 4, pos 180
    //"* a4:repeated-part-12345678901234567890:4a--\n"
    //"* clear context tail, some reasoning here --\n"
    //"* a5:repeated-part-12345678901234567890:5a--\n"
    //"* a6:XXXXart-1234567890123456789012345--6a--\n";
    // Build test
    builder
        // Line 0: Header
        .add_user_input("# common/ngram-simple, test 1 --------------\n")

        // Line 1: First repeated part - build context
        .add_assistant( "* a1:repeated-part-12345678901234567890:1a--\n")

        // Line 2: Second repeated part with draft expectations
        .add_assistant( "* a2:repeated-pa")
        .expect_accepted(               "rt-1234567890123")
        .expect_partial(                                "4567890:", "1a--\n* a")
        .add_assistant(                                         "2a--\n")

        // Line 3: Third repeated part
        .add_assistant( "* a3:repeated-pa")
        .expect_accepted(               "rt-1234567890123")
        .expect_partial(                                "4567890:", "2a--\n* a")
        .add_assistant(                                         "3a--\n")

        // Line 4: Reasoning section
        .add_reasoning("r1",
                        "* some reasoning here ... wait, ..., yes!?--\n")
        .end_reasoning("r1")

        // Line 5: Fourth repeated part
        .add_assistant( "* a4:repeated-pa")
        .expect_accepted(               "rt-1234567890123")
        .expect_partial(                                "4567890:", "3a--\n* s")
        .add_assistant(                                         "4a--\n")

        // delete reasoning block
        .delete_reasoning("r1")
        .add_reasoning("r2",
                        "* clear context tail, some reasoning here --\n")

        // call begin()
        .call_begin()

        // Line 6: Fifth repeated part after begin()
        .add_assistant( "* a5:repeated-pa")
        .expect_accepted(               "rt-1234567890123")
        .expect_partial(                                "4567890:", "4a--\n* c")
        .expect_rejected(                                       "4a--\n* clear con")
        .add_assistant(                                         "5a--\n")

        // Line 7: Sixth repeated part with different pattern
        .add_assistant( "* a6:XXXXart-12345678")
        .expect_partial(                     "901234567890", ":5a-")
        .add_assistant(                                  "12345--6a--\n");

    // Run the test
    int result = builder.run(test_name, adapter);

    if (result != 0) {
        std::cout << "Test " << test_name << " FAILED" << std::endl;
        return 1;
    }

    std::cout << "Test " << test_name << " completed successfully." << std::endl;
    return 0;
}

static std::string generate_unique_string(const std::string& prefix, size_t start_idx, size_t length) {
    std::ostringstream oss;
    while (oss.str().size() < length) {
        oss << prefix << std::setw(5) << std::setfill('0') << (start_idx);
        start_idx += prefix.size() + 5;
    }
    return oss.str().substr(0, length);
}

static int test_ngram_map_key_pr23936_oob() {
    const std::string test_name = "ngram_map_oob_test";

    const uint16_t n = 10;
    const uint16_t m = 10;
    const bool key_only = true;
    const uint16_t min_hits = 1;

    common_ngram_map map(n, m, key_only, min_hits);
    ngram_map_adapter adapter(map);

    spec_test_builder builder;

    std::string repeated_low_key    = "PATTERN-lo"; // len: 10
    std::string repeated_low_value  = "VALUE__low"; // len: 10

    std::string repeated_high_key   = "PATTERN-hi"; // len: 10
    std::string repeated_high_value = "VALUE_high"; // len: 10

    std::string user_input_0a = generate_unique_string("#User", 0, 500);
    std::string user_input_0b = generate_unique_string("#User", 500, 490);
    builder.add_user_input(user_input_0a);
    builder.add_user_input(repeated_low_key + repeated_low_value);
    builder.add_user_input(user_input_0a);
    builder.call_begin();

    std::string user_input_1 = generate_unique_string("#User", 1000, 13000);
    builder.add_user_input(user_input_1);
    builder.add_user_input(repeated_high_key + repeated_high_value + "%");
    builder.call_begin();

    builder.add_reasoning("r1", "BEGIN_" + repeated_high_key);
    builder.expect_accepted(repeated_high_value);
    std::string reasoning_text = generate_unique_string("#Reas", 3242, 9000);
    builder.add_reasoning("r1", reasoning_text);
    builder.end_reasoning("r1");

    std::string ass_suffix = generate_unique_string("#AssA", 21242,  176);

    builder.add_assistant(ass_suffix);
    // builder.print_state();

    // After this step we delete the whole conversation and start a new one.
    builder.delete_all();
    // Goal is a message similar to:
    // I common_ngram_map_begin: refresh map: idx_last_draft=23242, new begin=13678, #keys_checked=10, #keys_del=0, #values_del=0, #hashes_upd=5
    // Actual:
    //   common_ngram_map_begin: refresh map: idx_last_draft=23242, new begin=13678, #keys_checked=1, #keys_del=0, #values_del=0, #hashes_upd=8501

    builder.add_user_input(user_input_0a);
    builder.add_user_input(repeated_low_key + repeated_low_value);
    builder.add_user_input(user_input_0a);

    std::string user_input_2a = generate_unique_string("#User", 242, 1010);
    std::string user_input_2b = generate_unique_string("#User", 2420, 11638);
    builder.add_user_input(user_input_2a);
    builder.add_user_input(repeated_low_key);
    builder.add_user_input(user_input_2b);
    builder.call_begin();

    builder.add_assistant("Start...");
    // builder.print_state();
    builder.add_assistant(repeated_low_key); // <-- OOB without fix
    builder.expect_accepted(repeated_low_value);
    // builder.print_state();

    int result = builder.run(test_name, adapter);
    return result;
}

static int test_ngram_map_key_only() {
    const std::string test_name = "ngram_map_key_only_test";

    const uint16_t n = 15; // key size
    const uint16_t m = 12; // draft size
    const bool key_only = true;
    const uint16_t min_hits = 1;

    common_ngram_map map(n, m, key_only, min_hits);
    ngram_map_adapter adapter(map);

    spec_test_builder builder;

        //"# common/ngram-map-k, test 1 ---------------\n" // <- line 0, length: 45
        //"* a1:repeated-part-12345678901234567890:1a--\n" // <- line 1
        //"* a2:repeated-part-12345678901234567890:2a--\n"
        //"* a3:repeated-part-12345678901234567890:3a--\n"
        //"* some reasoning here ... wait, ..., yes!?--\n" // <- line 4, pos 180
        //"* a4:repeated-part-12345678901234567890:4a--\n"
        //"* some new prompt here, new question ...  --\n"
        //"* clear context tail, some reasoning here --\n"
        //"* a5:repeated-part-12345678901234567890:5a--\n"
        //"* a6:XXXXart-1234567890123456789012345--6a--\n";

    builder
        .add_user_input("# common/ngram-map-k, test 1 ---------------\n")
        .add_assistant( "* a1:repeated-part-12345678901234567890:1a--\n")
        .add_assistant( "* a2:repeated-part-")
        .expect_accepted(                  "123456789012")
        .expect_partial(                               "34567890:", "1a-")
        .expect_rejected(                                       "1a--\n* a2:re")
        .add_assistant(                                         "2a--\n")
        .add_assistant( "* a3:repeated-part-")
        .expect_accepted(                  "123456789012")
        .expect_accepted(                              "34567890:")
        .add_assistant(                                         "3a--\n")
        .add_reasoning("r1",
                        "* some reasoning here ... wait, ..., yes!?--\n")
        .end_reasoning("r1")
        .add_assistant( "* a4:repeated-part-")
        .expect_accepted(                  "123456789012")
        .expect_accepted(                              "34567890:")
        .add_assistant(                                         "4a--\n")
        .delete_reasoning("r1")
        .add_user_input("* some new prompt here, new question ...  --\n")
        .call_begin()
        .add_reasoning("r2",
                        "* clear context tail, some reasoning here --\n")
        .end_reasoning("r2")
        .add_assistant( "* a5:repeated-part-")
        .expect_accepted(                 "123456789012")
        .expect_accepted(                             "34567890:")
        .add_assistant(                                        "5a--\n")
        .add_assistant( "* a6:XXXXart-12345678901")
        .expect_partial(                        "234567890", ":1a")
        .expect_rejected(                                ":1a--\n* a2:r")
        .add_assistant(                                  "12345")
        .expect_rejected(                                     "67890:1a--\n*")
        .add_assistant(                                       "--6a--\n");

    int result = builder.run(test_name, adapter);

    if (result != 0) {
        std::cout << "Test " << test_name << " FAILED" << std::endl;
        return 1;
    }

    std::cout << "Test " << test_name << " completed successfully." << std::endl;
    return 0;
}

static int test_ngram_map_key_only_reasoning() {
    const std::string test_name = "ngram_map_key_only_reasoning_test";

    const uint16_t n = 12;
    const uint16_t m = 7;
    const bool key_only = true;
    const uint16_t min_hits = 1;

    common_ngram_map map(n, m, key_only, min_hits);
    ngram_map_adapter adapter(map);

    spec_test_builder builder;

        //"# common/ngram-map-k, reasoning\n" // <- line 0, length: 32
        //"* user1: What is 256 * 256? :1u\n" // <- line 1
        //"* reas1: 256 * 256 = 65536  :2r\n" // <- line 2, pos: 64, will be deleted
        //"*   more thinking, 250, 256 :3r\n" // <- line 3, pos: 96, will be deleted
        //"* answ1: 250 * 250 = 62500  :4a\n"
        //"* =...>  256 * 256 = 65536 !:5a\n"
        //"* user2: Again, 256 * 256 is:6u\n"
        //"* reas2: Wait, why again?   :7r\n" // <- line 5, pos 160
        //"* answ2: 256 * 256 = 65536 !:8a\n"
        //"* user3: Again, 256 * 256 is:9u\n"
        //"* reas3: Wait, why again??  :0r\n"
        //"* answ3: 256 * 256 = 65536 !:1a\n";

    builder
        .add_user_input("# common/ngram-map-k, reasoning\n")
        .add_user_input("* user1: What is 256 * 256? :1u\n")
        .add_reasoning("r1",
                        "* reas1: 256 * 256 = 65536  :2r\n"
                        "*   more thinking, 250, 256 :3r\n")
        .end_reasoning("r1")
        .add_assistant( "* answ1: 250 * 250 = 62500  :4a\n")
        .add_assistant( "* =...>  256 * 256 =")
        .expect_accepted(                    " 65536 ")
        .expect_rejected(                           " :2r\n* ")
        .add_assistant(                             "!:5a\n")
        .delete_reasoning("r1")
        .add_user_input("* user2: Again, 256 * 256 is:6u\n")
        .call_begin()
        .add_reasoning("r2",
                        "* reas2: Wait, why again?   :7r\n")
        .end_reasoning("r2")
        .add_assistant( "* answ2: 256 * 256 =")
        .expect_accepted(                   " 65536 ")
        .expect_rejected(                          "!:5a\n* ")
        .add_assistant(                            "!")
        .expect_rejected(                           ":5a\n* u")
        .add_assistant(                             ":")
        .expect_rejected(                            "5a\n* us")
        .add_assistant(                              "8a\n")
        .delete_reasoning("r2")
        .add_user_input("* user3: Again, 256 * 256 is:9u\n")
        .call_begin()
        .add_reasoning("r3",
                        "* reas3: Wait, why again??  :0r\n")
        .end_reasoning("r3")
        .add_assistant( "* answ3: 256 * 256 =")
        .expect_accepted(                   " 65536 ")
        .add_assistant(                            "!:1a\n");

    int result = builder.run(test_name, adapter);

    if (result != 0) {
        std::cout << "Test " << test_name << " FAILED" << std::endl;
        return 1;
    }

    std::cout << "Test " << test_name << " completed successfully." << std::endl;
    return 0;
}

static int test_ngram_map_key_with_values() {
    const std::string test_name = "ngram_map_key_with_values_test";

    const uint16_t n = 10; // key size
    const uint16_t m = 14; // draft/value size
    const bool key_only = false;
    const uint16_t min_hits = 3;

    common_ngram_map map(n, m, key_only, min_hits);
    ngram_map_adapter adapter(map);

    spec_test_builder builder;

        //"# common/ngram-map-k, test 2 --\n" // <- line 0, length: 32
        //"* 01:the-key1:value-1.1----:01-\n" // <- line 1
        //"* 02:the-key2:value-2.1----:02-\n"
        //"* 03:the-key1:value-1.2----:03-\n" // <-- value gets ignored in stat because #key-hits < 3
        //"* 04:the-key2:value-2.1----:04-\n"
        //"* 05:the-key1:value-1.3----:05-\n" // <-- value gets ignored in stat because #key-hits < 3
        //"* 06:the-key2:value-2.1----:06-\n"
        //"* 07:the-key1:value-1.1----:07-\n"
        //"* 08:some reasoning here ...  -\n"
        //"* 09:the-key2:value-2.1----:09-\n" // the-key-2 had three hits, one value only
        //"* 10:the-key1:value-1.2----:10-\n"
        //"* 11:the-key1:value-1.3----:11-\n"
        //"* 12:the-key1:value-1.4----:12-\n"
        //"* 13:the-key1:value-1.5----:13-\n"
        //"* 14:new reasoning here  ...  -\n"
        //"* 15:the-key1:value-1.2----:15-\n"
        //"* 16:the-key1:value-1.2----:16-\n"
        //"* 17:the-key1:value-1.2----:17-\n"
        //"* 18:the-key1:value-1.2----:18-\n";

       // Before the introduction of the fluent API the tests
       // used strings with special instructions, e.g. "{A14:'value-1.1----:'} = accept 14 chars.
       //
       // "# common/ngram-map-k, test 3 --\n" // <- line 0, length: 32
       // "* 01:the-key1:value-1.1----:01-\n" // <- line 1
       // "* 02:the-key2:value-2.1----:02-\n"
       // "* 03:the-key1:value-1.2----:03-\n" // <-- value gets ignored in stat because #key-hits < 3
       // "* 04:the-key2:value-2.1----:04-\n"
       // "* 05:the-key1:value-1.3----:05-\n" // <-- value gets ignored in stat because #key-hits < 3
       // "* 06:the-key2:value-2.1----:06-\n"
       // "* 07:the-key1{A14:'value-1.1----:'}:value-1.1----:07-\n"
       // "* 08:some reasoning here ...  -\n"
       // "* 09:the-key2{A14:'value-2.1----:'}:value-2.1----{R:'02-\n* 03:the-k'}:{R:'2-\n* 03:the-ke'}09-\n" // the-key-2 had three hits, one value only
       // "* 10:the-key1{P08:'value-1.1----:'}:value-1{R:'1----:01-\n* 02'}.2----:10-\n"
       // "* 11:the-key1{A08:'value-1.'}:value-1.3----:11-\n"
       // "* 12:the-key1{A08:'value-1.'}:value-1.4----:12-\n"
       // "* 13:the-key1{A08:'value-1.'}:value-1.5----:13-\n" // value stats: [46/5, 0/0, 0/0, 0/0]
       // "{D256/032}{B}* 14:new reasoning here  ...  -\n"
       // "* 15:the-key1:value-1.2----:15-\n"
       // "* 16:the-key1:value-1.2----:16-\n"
       // "* 17:the-key1:value-1.2----:{R:'0-\n* 11:the-ke'}17-\n" // value stats: [46/3, 110/1, 174/1, 0/0]
       // "* 18:the-key1:value-1.{A04:'----:10-\n* 11:'}2---{R:':10-\n* 11:the-'}-{R:'10-\n* 11:the-k'}:18-\n";

    builder
        .add_user_input("# common/ngram-map-k, test 3 --\n")
        .add_assistant( "* 01:the-key1:value-1.1----:01-\n")
        .add_assistant( "* 02:the-key2:value-2.1----:02-\n")
        .add_assistant( "* 03:the-key1:value-1.2----:03-\n")
        .add_assistant( "* 04:the-key2:value-2.1----:04-\n")
        .add_assistant( "* 05:the-key1:value-1.3----:05-\n")
        .add_assistant( "* 06:the-key2:value-2.1----:06-\n")
        .add_assistant( "* 07:the-key1:")
        .expect_accepted(             "value-1.1----:")
        .add_assistant(                             "07-\n")
        .add_reasoning("r1",
                        "* 08:some reasoning here ...  -\n")
        .end_reasoning("r1")
        .add_assistant( "* 09:the-key2:")
        .expect_accepted(             "value-2.1----:")
        .expect_rejected(                           "02-\n* 03:the-k")
        .add_assistant(                             "0")
        .expect_rejected(                            "2-\n* 03:the-ke")
        .add_assistant(                              "9-\n")
        .add_assistant( "* 10:the-key1:")
        .expect_partial(              "value-1.", "1----:")
        .expect_rejected(                     "1----:01-\n* 02")
        .add_assistant(                       "2----:10-\n")
        .add_assistant( "* 11:the-key1:")
        .expect_accepted(             "value-1.")
        .add_assistant(                       "3----:11-\n")
        .add_assistant( "* 12:the-key1:")
        .expect_accepted(             "value-1.")
        .add_assistant(                       "4----:12-\n")
        .add_assistant( "* 13:the-key1:")
        .expect_accepted(             "value-1.")
        .add_assistant(                       "5----:13-\n")
        .delete_reasoning("r1")
        .call_begin()
        .add_reasoning("r2",
                        "* 14:new reasoning here  ...  -\n")
        .end_reasoning("r2")
        .add_assistant( "* 15:the-key1:value-1.2----:15-\n")
        .add_assistant( "* 16:the-key1:value-1.2----:16-\n")
        .add_assistant( "* 17:the-key1:value-1.2----:1")
        .expect_rejected(                            "0-\n* 11:the-ke")
        .add_assistant(                              "7-\n")
        .add_assistant( "* 18:the-key1:value-1.2") // <--
        .expect_partial(                       "----", ":10-\n* 11:")
        .expect_rejected(                          ":10-\n* 11:the-")
        .add_assistant(                            ":")
        .expect_rejected(                           "10-\n* 11:the-k")
        .add_assistant(                             "18-\n");

    int result = builder.run(test_name, adapter);

    if (result != 0) {
        std::cout << "Test " << test_name << " FAILED" << std::endl;
        return 1;
    }

    std::cout << "Test " << test_name << " completed successfully." << std::endl;
    return 0;
}

int main(int argc, char ** argv) {
    (void)argc;
    (void)argv;

    //common_log_set_verbosity_thold(LOG_LEVEL_DEBUG);

    int num_failed = 0;
    num_failed += test_ngram_simple();
    num_failed += test_ngram_map_key_only();
    num_failed += test_ngram_map_key_only_reasoning();
    num_failed += test_ngram_map_key_with_values();
    // The test of PR #23936 has a duration of 1.6 seconds.
    // num_failed += test_ngram_map_key_pr23936_oob();

    if (num_failed == 0) {
        std::cout << "All tests completed successfully." << std::endl;
    } else if (num_failed == 1) {
        std::cout << "One test failed." << std::endl;
        assert(false);
    } else {
        std::cout << num_failed << " tests failed." << std::endl;
        assert(false);
    }
    return (num_failed == 0) ? 0 : 1;
}

