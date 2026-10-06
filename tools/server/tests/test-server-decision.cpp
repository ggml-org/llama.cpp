// unit tests for the decision route logic that does not need a model
// see the tests of the route itself in tools/server/tests/unit/test_systemone.py

#include "server-common.h"
#include "server-decision.h"
#include "server-queue.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <set>
#include <string>
#include <type_traits>
#include <vector>

// ex_wrapper maps std::invalid_argument to 400 on every route, so a request error must not derive
// from it: the two classes below are what give the decision route its 422 and its 501
static_assert(!std::is_base_of<std::invalid_argument, server_invalid_request>::value);
static_assert(!std::is_base_of<std::invalid_argument, server_status_error>::value);

// both are std::exception, so the arm that answers 500 catches what the ones above do not
static_assert(std::is_base_of<std::exception, server_invalid_request>::value);
static_assert(std::is_base_of<std::exception, server_status_error>::value);

static void test_error_codes_are_http_statuses() {
    for (size_t i = 0; i < ERROR_TYPE_COUNT; i++) {
        const int code = error_type_info((error_type) i).code;
        GGML_ASSERT(code >= 400 && code <= 599);
    }
}

static void test_error_types_are_distinct() {
    std::set<std::string> types;
    for (size_t i = 0; i < ERROR_TYPE_COUNT; i++) {
        const std::string type = error_type_info((error_type) i).type;
        GGML_ASSERT(!type.empty());
        GGML_ASSERT(types.insert(type).second); // a client must be able to tell the kinds apart
    }
    GGML_ASSERT(types.size() == ERROR_TYPE_COUNT);
}

// The kinds a client branches on. The rest of the switch has no client contract of its own and
// is covered by the two property tests above, so there is no verbatim copy of it here.
static void test_client_visible_codes() {
    struct {
        error_type type;
        int code;
    } const expected[] = {
        {ERROR_TYPE_INVALID_REQUEST,           400},
        {ERROR_TYPE_EXCEED_CONTEXT_SIZE,      400},
        {ERROR_TYPE_EXCEED_BATCH_SIZE,        400},
        {ERROR_TYPE_INVALID_REQUEST_SEMANTIC, 422},
        {ERROR_TYPE_NOT_SUPPORTED,            501},
        {ERROR_TYPE_UNAVAILABLE,              503},
        {ERROR_TYPE_RATE_LIMITED,             429},
        {ERROR_TYPE_REQUEST_TOO_LARGE,        413},
    };
    for (const auto & row : expected) {
        GGML_ASSERT(error_type_info(row.type).code == row.code);
    }
}

static void test_format_error_response() {
    for (size_t i = 0; i < ERROR_TYPE_COUNT; i++) {
        const error_type type = (error_type) i;
        const json res = format_error_response("test", type);
        GGML_ASSERT(res.at("code")    == error_type_info(type).code);
        GGML_ASSERT(res.at("message") == "test");
        GGML_ASSERT(res.at("type")    == error_type_info(type).type);
    }
}

// ex_wrapper answers a server_invalid_request with ERROR_TYPE_INVALID_REQUEST_SEMANTIC. 422 must
// belong to that kind alone, or a 422 on the wire could come from somewhere else.
static void test_invalid_request_answers_422() {
    GGML_ASSERT(error_type_info(ERROR_TYPE_INVALID_REQUEST_SEMANTIC).code == 422);
    for (size_t i = 0; i < ERROR_TYPE_COUNT; i++) {
        if (i != ERROR_TYPE_INVALID_REQUEST_SEMANTIC) {
            GGML_ASSERT(error_type_info((error_type) i).code != 422);
        }
    }
}

// ex_wrapper answers a server_status_error with the kind it carries, so every kind must survive
// the round trip and keep the wire shape of the switch
static void test_status_error_carries_its_kind() {
    for (size_t i = 0; i < ERROR_TYPE_COUNT; i++) {
        const error_type type = (error_type) i;
        const server_status_error err(type, "test");
        GGML_ASSERT(err.type == type);
        GGML_ASSERT(std::string(err.what()) == "test");
        GGML_ASSERT(error_type_info(err.type).code == error_type_info(type).code);
    }
}

// 413 reports that the request asks for more work than this server will render. It has to be
// recognizable on its own: a client must not confuse it with the 429 that says the server is busy,
// so no other kind may share its status, and its type string must stay the one the client reads.
// 400 is shared by two kinds on purpose, so this is asserted for this kind and not for the switch
static void test_request_too_large_is_its_own_kind() {
    const server_error_info too_large = error_type_info(ERROR_TYPE_REQUEST_TOO_LARGE);
    GGML_ASSERT(too_large.code == 413);
    GGML_ASSERT(std::string(too_large.type) == "request_too_large_error");

    for (size_t i = 0; i < ERROR_TYPE_COUNT; i++) {
        if (i == ERROR_TYPE_REQUEST_TOO_LARGE) {
            continue;
        }
        const server_error_info other = error_type_info((error_type) i);
        GGML_ASSERT(other.code != too_large.code);
        GGML_ASSERT(std::string(other.type) != too_large.type);
    }
}


// the ceiling Jev puts on the options of one question, and the widest a model here has to answer
static const size_t N_OPTIONS_MAX = 255;

// The traits a model type hands to the seams, one builder per seam so a test states only the traits
// its seam reads. A brace initialization would leave the defaulted fields out and -Wextra would
// report it, so the fields are named. What a test leaves out is the API ceiling and the defaults.

// the parse seam reads the option limit and the two option orders
static decision_model_traits parse_traits(size_t n_options_max   = N_OPTIONS_MAX,
                                          bool   noul_true_first = false,
                                          bool   choice_sorted   = false) {
    decision_model_traits traits;
    traits.n_options_max   = n_options_max;
    traits.noul_true_first = noul_true_first;
    traits.choice_sorted   = choice_sorted;
    return traits;
}

// the template seam reads the per-output labels and the option marker
static decision_model_traits template_traits(const std::vector<std::string> & label_texts = {},
                                             std::string                       text_marker = "") {
    decision_model_traits traits;
    traits.label_texts = label_texts;
    traits.text_marker = text_marker;
    return traits;
}

// a question of one column with the given option keys, the shape every test below builds
static server_decision_question shaped_question(server_decision_question_type type, const std::vector<std::string> & keys) {
    server_decision_question q;
    q.type         = type;
    q.instructions = "probe instructions";
    for (const auto & key : keys) {
        q.options.push_back({key, "description of " + key});
    }
    return q;
}

//
// confidence
//
// The confidence of an answer is read from the answer, so these drive the same path the route does:
// scores that softmax to the distribution under test, then the "confidence" field of the answer.
//

// the values TypeSafe documents, to two decimals
static const double ROUNDING = 0.01;

// the choice formula is (N * p_max - 1) / (N - 1), these are its published examples
static const struct {
    size_t n;
    double p_max;
    double published;
} CHOICE_PUBLISHED[] = {
    { 3, 0.61, 0.42 },
    { 4, 0.40, 0.20 },
    { 5, 0.74, 0.67 },
    { 3, 0.84, 0.76 },
};

// the score formula is the mean distance to the most likely level relative to a uniform
// distribution. TypeSafe publishes no rule above 3 levels; the rows below are what this server
// reports there
static const struct {
    std::vector<double> probs;
    double published;
} SCORE_PUBLISHED[] = {
    { { 0.95, 0.05 },                   0.90 }, // 2 levels
    { { 0.0, 0.57, 0.43 },              0.35 }, // 3 levels
    { { 0.0, 0.74, 0.26 },              0.61 },
    { { 0.0, 0.91, 0.09 },              0.87 },
    { { 0.0, 0.48, 0.52, 0.0 },         0.52 }, // 4 levels
    { { 0.0, 0.0, 0.86, 0.14, 0.0 },    0.89 }, // 5 levels
};

// n probabilities that sum to one, with p_max on the level at mode and the rest spread evenly
static std::vector<double> distribution(size_t n, size_t mode, double p_max) {
    std::vector<double> probs(n, (1.0 - p_max) / (n - 1));
    probs[mode] = p_max;
    return probs;
}

// the one variant openjev evaluates for any question, with the scores that softmax to probs
static server_decision_context openjev_context() {
    server_decision_context ctx;
    ctx.type = COMMON_DECISION_TYPE_OPENJEV;
    return ctx;
}

// softmax(log(p)) is p, so this turns a distribution into the scores that produce it exactly.
// a zero probability has no logarithm, a score far below the peak gives it back as exactly zero
static std::vector<float> scores_of(const std::vector<double> & probs) {
    std::vector<float> scores;
    for (const double p : probs) {
        scores.push_back(p > 0.0 ? (float) std::log(p) : -1e9f);
    }
    return scores;
}

static server_decision_question question_of_columns(server_decision_question_type type, size_t n) {
    server_decision_question q = shaped_question(type, {});
    for (size_t i = 0; i < n; i++) {
        q.options.push_back({ std::to_string(i), "description of level " + std::to_string(i) });
    }
    return q;
}

// the confidence and the probabilities the answer carries for one distribution
static json answer_of(server_decision_question_type type, const std::vector<double> & probs) {
    const server_decision_context ctx = openjev_context();
    const server_decision_question q  = question_of_columns(type, probs.size());
    return ctx.format_answer(q, { scores_of(probs) });
}

static double confidence_of(server_decision_question_type type, const std::vector<double> & probs) {
    return answer_of(type, probs).at("confidence").get<double>();
}

static void test_confidence_reproduces_the_distribution() {
    // the scores have to give back the distribution they were built from, or the confidence below
    // would be measured against a distribution the caller never sees
    for (size_t n = 2; n <= 6; n++) {
        const std::vector<double> probs = distribution(n, 0, 0.7);
        const json answer = answer_of(SERVER_DECISION_QUESTION_CHOICE, probs);
        double sum = 0.0;
        for (size_t i = 0; i < n; i++) {
            sum += answer.at("probabilities").at(std::to_string(i)).get<double>();
        }
        GGML_ASSERT(std::fabs(sum - 1.0) < 1e-9);
        GGML_ASSERT(std::fabs(answer.at("probabilities").at("0").get<double>() - 0.7) < 1e-6);
    }
}

static void test_choice_confidence() {
    for (const auto & row : CHOICE_PUBLISHED) {
        const std::vector<double> probs = distribution(row.n, 0, row.p_max);
        GGML_ASSERT(std::fabs(confidence_of(SERVER_DECISION_QUESTION_CHOICE, probs) - row.published) <= ROUNDING);
    }
    // the published 0.81 for p_max 0.88 comes from unrounded probabilities, this gives 0.82
    GGML_ASSERT(std::fabs(
                    confidence_of(SERVER_DECISION_QUESTION_CHOICE, distribution(3, 0, 0.88)) - 0.82) <= ROUNDING);

    // one option is certain, and a uniform split is not
    GGML_ASSERT(std::fabs(confidence_of(SERVER_DECISION_QUESTION_CHOICE, { 1.0 }) - 1.0) <= ROUNDING);
    GGML_ASSERT(confidence_of(SERVER_DECISION_QUESTION_CHOICE, { 0.5, 0.5 }) <= ROUNDING);
    GGML_ASSERT(std::fabs(confidence_of(SERVER_DECISION_QUESTION_CHOICE, { 1.0, 0.0 }) - 1.0) <= ROUNDING);
}

static void test_score_confidence() {
    for (const auto & row : SCORE_PUBLISHED) {
        GGML_ASSERT(std::fabs(confidence_of(SERVER_DECISION_QUESTION_SCORE, row.probs) - row.published) <= ROUNDING);
    }
    // one option is certain, and a uniform split is not
    GGML_ASSERT(std::fabs(confidence_of(SERVER_DECISION_QUESTION_SCORE, { 1.0 }) - 1.0) <= ROUNDING);
    GGML_ASSERT(confidence_of(SERVER_DECISION_QUESTION_SCORE, { 0.5, 0.5 }) <= ROUNDING);
}

// the two are different measures, not one formula written twice: applied to a score distribution
// the choice formula misses the published value of every row above 3 levels, which is why both exist
static void test_confidence_measures_differ() {
    const std::vector<double> four = { 0.0, 0.48, 0.52, 0.0 };
    const double as_choice = confidence_of(SERVER_DECISION_QUESTION_CHOICE, four);
    GGML_ASSERT(std::fabs(as_choice - 0.36) <= ROUNDING);
    GGML_ASSERT(std::fabs(confidence_of(SERVER_DECISION_QUESTION_SCORE, four) - as_choice) > 0.1);

    const std::vector<double> five = { 0.0, 0.0, 0.86, 0.14, 0.0 };
    const double five_choice  = confidence_of(SERVER_DECISION_QUESTION_CHOICE, five);
    GGML_ASSERT(std::fabs(five_choice - 0.825) <= ROUNDING);
    GGML_ASSERT(std::fabs(confidence_of(SERVER_DECISION_QUESTION_SCORE, five) - five_choice) > 0.01);
}

// they agree on 2 levels, and on 3 levels when the mass sits on the mode and its neighbour, which
// is the shape of every 3-level example TypeSafe publishes
static void test_confidence_equivalence() {
    for (int s = 0; s <= 20; s++) {
        const std::vector<double> two = distribution(2, 0, (double) s / 20.0);
        GGML_ASSERT(std::fabs(confidence_of(SERVER_DECISION_QUESTION_CHOICE, two) -
                              confidence_of(SERVER_DECISION_QUESTION_SCORE, two)) < 1e-9);

        for (size_t mode = 0; mode < 3; mode++) {
            const size_t near = mode == 0 ? 1 : mode - 1;
            std::vector<double> three(3, 0.0);
            three[mode] = 0.5 + 0.5 * s / 20.0;
            three[near] = 1.0 - three[mode];
            GGML_ASSERT(std::fabs(confidence_of(SERVER_DECISION_QUESTION_CHOICE, three) -
                                  confidence_of(SERVER_DECISION_QUESTION_SCORE, three)) < 1e-9);
        }
    }

    // they part when the mode is at an end and the other mass sits two steps away, which is the
    // shape that makes the distance to the mode larger than the distance to the centre
    for (const size_t mode : { (size_t) 0, (size_t) 2 }) {
        std::vector<double> three(3, 0.0);
        three[mode]              = 0.6;
        three[mode == 0 ? 2 : 0] = 0.4;
        GGML_ASSERT(std::fabs(confidence_of(SERVER_DECISION_QUESTION_CHOICE, three) -
                              confidence_of(SERVER_DECISION_QUESTION_SCORE, three)) > 0.1);
    }
}

// both stay in [0, 1] and neither decreases as the distribution gets sharper, for every level count
// and every position of the mode
static void test_confidence_monotonic_in_p_max() {
    for (size_t n = 2; n <= 8; n++) {
        for (size_t mode = 0; mode < n; mode++) {
            double prev_choice = -1.0;
            double prev_score  = -1.0;
            for (int s = 0; s <= 40; s++) {
                const std::vector<double> probs = distribution(n, mode, 1.0 / n + (1.0 - 1.0 / n) * s / 40.0);
                const double choice = confidence_of(SERVER_DECISION_QUESTION_CHOICE, probs);
                const double score  = confidence_of(SERVER_DECISION_QUESTION_SCORE, probs);
                GGML_ASSERT(choice >= 0.0 && choice <= 1.0);
                GGML_ASSERT(score  >= 0.0 && score  <= 1.0);
                GGML_ASSERT(choice >= prev_choice - 1e-12);
                GGML_ASSERT(score  >= prev_score  - 1e-12);
                prev_choice = choice;
                prev_score  = score;
            }
        }
    }
}

//
// variant and output shaping
//

// the number of outputs a question reads is observable only through format_answer: it throws unless
// every variant carries exactly that many scores, so exactly one width answers the question
static size_t outputs_that_answer(common_decision_type type, const server_decision_question & question) {
    server_decision_context ctx;
    ctx.type = type;

    const size_t n_max = std::max(question.options.size(), (size_t) 9) + 1; // lev reads at most nine
    size_t      n_ok   = 0;
    for (size_t n = 1; n <= n_max; n++) {
        try {
            ctx.format_answer(question,
                              std::vector<std::vector<float>>(ctx.n_variants(question),
                                                              std::vector<float>(n, 0.0f)));
        } catch (const std::exception &) {
            continue;
        }
        GGML_ASSERT(n_ok == 0); // two widths cannot answer the same question
        n_ok = n;
    }
    GGML_ASSERT(n_ok > 0);
    return n_ok;
}

static void test_variant_and_output_matrix() {
    // lev is the only type that evaluates two variants, and only for a choice with something to reverse
    server_decision_context lev;
    lev.type = COMMON_DECISION_TYPE_LEV;
    GGML_ASSERT(lev.n_variants(shaped_question(SERVER_DECISION_QUESTION_CHOICE, { "a", "b" })) == 2);
    GGML_ASSERT(lev.n_variants(shaped_question(SERVER_DECISION_QUESTION_CHOICE,
                                               { "a", "b", "c", "d", "e", "f", "g", "h" })) == 2);
    GGML_ASSERT(lev.n_variants(shaped_question(SERVER_DECISION_QUESTION_CHOICE, { "a" })) == 1);
    GGML_ASSERT(lev.n_variants(shaped_question(SERVER_DECISION_QUESTION_SCORE, { "a", "b", "c" })) == 1);
    GGML_ASSERT(lev.n_variants(shaped_question(SERVER_DECISION_QUESTION_NOUL, { "a", "b" })) == 1);

    for (const common_decision_type type : { COMMON_DECISION_TYPE_OPENJEV, COMMON_DECISION_TYPE_KEV,
                                             COMMON_DECISION_TYPE_LAYA, COMMON_DECISION_TYPE_NIMBLE,
                                             COMMON_DECISION_TYPE_CLEF }) {
        server_decision_context ctx;
        ctx.type = type;
        for (const size_t n_options : { (size_t) 1, (size_t) 2, (size_t) 8 }) {
            const server_decision_question q = shaped_question(SERVER_DECISION_QUESTION_CHOICE,
                                                              std::vector<std::string>(n_options, "a"));
            GGML_ASSERT(ctx.n_variants(q) == 1);
            // every other type reads one output per option
            GGML_ASSERT(outputs_that_answer(type, q) == n_options);
        }
    }

    // lev reads a noul on nine ratings, and one output per option for everything else
    GGML_ASSERT(outputs_that_answer(COMMON_DECISION_TYPE_LEV,
                                    shaped_question(SERVER_DECISION_QUESTION_NOUL, { "false", "true" })) == 9);
    GGML_ASSERT(outputs_that_answer(COMMON_DECISION_TYPE_LEV,
                                    shaped_question(SERVER_DECISION_QUESTION_CHOICE, { "a", "b", "c", "d" })) == 4);
    GGML_ASSERT(outputs_that_answer(COMMON_DECISION_TYPE_KEV,
                                    shaped_question(SERVER_DECISION_QUESTION_SCORE, { "a", "b", "c" })) == 3);
}

// The range contract at the seam the route calls, over every model type at once. It sits here rather
// than with the formulas because it needs the per-type output widths: the types disagree on how
// many outputs a noul reads, so no single-type test covers the matrix.
static void test_confidence_is_reported_only_where_it_is_defined() {
    for (const common_decision_type type : { COMMON_DECISION_TYPE_OPENJEV, COMMON_DECISION_TYPE_LEV,
                                              COMMON_DECISION_TYPE_KEV, COMMON_DECISION_TYPE_LAYA,
                                              COMMON_DECISION_TYPE_NIMBLE, COMMON_DECISION_TYPE_CLEF }) {
        server_decision_context ctx;
        ctx.type = type;

        for (const server_decision_question_type qt : { SERVER_DECISION_QUESTION_NOUL,
                                                        SERVER_DECISION_QUESTION_CHOICE,
                                                        SERVER_DECISION_QUESTION_SCORE }) {
            // a noul has its two options whatever its criteria say, the other two read one per level
            const server_decision_question q =
                shaped_question(qt, qt == SERVER_DECISION_QUESTION_NOUL
                                         ? std::vector<std::string>{ "false", "true" }
                                         : std::vector<std::string>{ "0", "1" });

            const std::vector<double>           probs  = distribution(outputs_that_answer(type, q), 0, 0.7);
            const std::vector<std::vector<float>> scores(ctx.n_variants(q), scores_of(probs));
            const json                          answer = ctx.format_answer(q, scores);

            if (qt == SERVER_DECISION_QUESTION_NOUL) {
                GGML_ASSERT(!answer.contains("confidence"));
                continue;
            }
            GGML_ASSERT(answer.contains("confidence"));
            const double confidence = answer.at("confidence").get<double>();
            GGML_ASSERT(confidence >= 0.0 && confidence <= 1.0);
        }
    }
}

static std::vector<std::string> object_keys(const json & val) {
    std::vector<std::string> keys;
    for (const auto & item : val.items()) {
        keys.push_back(item.key());
    }
    return keys;
}

static bool keys_are_sorted(const std::vector<std::string> & keys) {
    return std::is_sorted(keys.begin(), keys.end());
}

static void test_template_input_lev() {
    const std::vector<std::string> labels = {"A", "B", "C"};
    const server_decision_question q = shaped_question(SERVER_DECISION_QUESTION_CHOICE, {"zulu", "alpha", "mike"});

    const json forward  = decision_template_input({ COMMON_DECISION_TYPE_LEV, template_traits(labels), "the state", { q }, q, 0, 0 });
    const json reversed = decision_template_input({ COMMON_DECISION_TYPE_LEV, template_traits(labels), "the state", { q }, q, 1, 0 });

    // variant 1 shows the options in reverse order
    GGML_ASSERT(forward.at("options").at(0).at("key") == "zulu");
    GGML_ASSERT(forward.at("options").at(2).at("key") == "mike");
    GGML_ASSERT(reversed.at("options").at(0).at("key") == "mike");
    GGML_ASSERT(reversed.at("options").at(2).at("key") == "zulu");

    // the label is the raw output label and stays paired with the raw output index, not the display position
    GGML_ASSERT(reversed.at("options").at(0).at("label") == "A");
    GGML_ASSERT(reversed.at("options").at(2).at("label") == "C");

    // lev was trained with sorted keys. `images` is appended after the shaping, as it always was,
    // so it is the one key that does not take part in the sort
    std::vector<std::string> top = object_keys(forward);
    top.erase(std::remove(top.begin(), top.end(), "images"), top.end());
    GGML_ASSERT(keys_are_sorted(top));
    for (const auto & opt : forward.at("options")) {
        GGML_ASSERT(keys_are_sorted(object_keys(opt)));
    }
    GGML_ASSERT(forward.at("images").empty());

    // the key of the question is part of the input, and no key is added or dropped
    GGML_ASSERT(forward.at("id") == q.id);
    GGML_ASSERT(forward.size() == 6);
    GGML_ASSERT(forward.contains("id") && forward.contains("type") && forward.contains("instructions") &&
                forward.contains("state") && forward.contains("options") && forward.contains("images"));
}

static void test_template_input_kev() {
    server_decision_question q = shaped_question(SERVER_DECISION_QUESTION_NOUL, {"false", "true"});
    q.instructions = json{{"what", "do we answer"}};
    q.options[1].description = "<|box_end|>";

    const json inp = decision_template_input({ COMMON_DECISION_TYPE_KEV, template_traits(), json{{"ticket", "<|box_start|>"}}, { q }, q, 0, 0 });

    // state, instructions and descriptions are flattened to text
    GGML_ASSERT(inp.at("state").is_string());
    GGML_ASSERT(inp.at("instructions").is_string());
    GGML_ASSERT(inp.at("options").at(0).at("description").is_string());

    // a special token in the text is escaped so it cannot be read as such
    GGML_ASSERT(inp.at("state").get<std::string>().find("<|box_start|>") == std::string::npos);
    GGML_ASSERT(inp.at("state").get<std::string>().find("<\xC2\xA6" "box_start" "\xC2\xA6>") != std::string::npos);
    GGML_ASSERT(inp.at("options").at(1).at("description").get<std::string>().find("<\xC2\xA6" "box_end" "\xC2\xA6>") != std::string::npos);

    // the key is part of the input, and it is the one the caller wrote
    GGML_ASSERT(inp.contains("id"));
}

static void test_kev_text_golden() {
    const server_decision_question q = shaped_question(SERVER_DECISION_QUESTION_NOUL, {"false", "true"});
    const auto flatten = [&q](const json & state) {
        return decision_template_input({ COMMON_DECISION_TYPE_KEV, template_traits(), state, { q }, q, 0, 0 }).at("state").get<std::string>();
    };

    GGML_ASSERT(flatten(json()).empty());
    GGML_ASSERT(flatten(true)  == "True");
    GGML_ASSERT(flatten(false) == "False");
    GGML_ASSERT(flatten("plain text") == "plain text");
    GGML_ASSERT(flatten(json::array({"a", "b"})) == "- a\n- b");

    json flat = json::object();
    flat["a"] = "x";
    flat["b"] = 1;
    GGML_ASSERT(flatten(flat) == "a: x\nb: 1");

    json nested = json::object();
    nested["a"] = json{{"b", "c"}};
    GGML_ASSERT(flatten(nested) == "a:\n  b: c");

    json list = json::object();
    list["list"] = json::array({json{{"x", "y"}}});
    GGML_ASSERT(flatten(list) == "list:\n  - x: y");
}

static void test_template_input_laya() {
    server_decision_question q = shaped_question(SERVER_DECISION_QUESTION_SCORE, {"0", "1", "2"});
    q.instructions = "which MARK level";
    q.options[0].description = "has MARK inside";
    const std::string marker = "MARK";

    const json inp = decision_template_input({ COMMON_DECISION_TYPE_LAYA, template_traits({}, marker), "state MARK here", { q }, q, 0, 1 });

    // the marker is replaced in every string, at every depth
    GGML_ASSERT(inp.at("state").get<std::string>().find("MARK") == std::string::npos);
    GGML_ASSERT(inp.at("instructions").get<std::string>().find("MARK") == std::string::npos);
    GGML_ASSERT(inp.at("options").at(0).at("description").get<std::string>().find("MARK") == std::string::npos);
    GGML_ASSERT(inp.at("state").get<std::string>().find("state") == 0);

    // one media marker per image, and none when there is no image
    GGML_ASSERT(inp.at("images").size() == 1);
    GGML_ASSERT(inp.at("images").at(0).get<std::string>() == get_media_marker());

    const json no_image = decision_template_input({ COMMON_DECISION_TYPE_LAYA, template_traits({}, marker), "state", { q }, q, 0, 0 });
    GGML_ASSERT(no_image.at("images").empty());
}

// the head window clips options and question to max_head_tokens: over budget the helper reports
// kept below original on both, in budget it reports everything kept and no clipping
static void test_head_clip_reports_kept_below_original() {
    const decision_head_clip over = decision_clip_head({100, 100, 100, 100}, 51, 64, 48);
    GGML_ASSERT(over.n_options_full == 400);
    GGML_ASSERT(over.n_options_kept == 48);
    GGML_ASSERT(over.n_options_kept < over.n_options_full);
    GGML_ASSERT(over.clipped_options);
    GGML_ASSERT(over.n_question_full == 50);
    GGML_ASSERT(over.n_question_kept == 16);
    GGML_ASSERT(over.n_question_kept < over.n_question_full);
    GGML_ASSERT(over.clipped_question);
    // the cap the helper used keeps the same total when applied per option
    size_t resized = 0;
    for (const size_t size : {100, 100, 100, 100}) {
        resized += std::min(size, over.n_option_max);
    }
    GGML_ASSERT(resized == over.n_options_kept);
}

static void test_head_clip_is_silent_in_budget() {
    const decision_head_clip under = decision_clip_head({10, 10}, 20, 64, 48);
    GGML_ASSERT(!under.clipped_options && !under.clipped_question);
    GGML_ASSERT(under.n_options_kept == under.n_options_full);
    GGML_ASSERT(under.n_question_kept == under.n_question_full);
}

// nimble is answered one field of a schema, so its prompt carries the whole request and the key that
// names the field to answer. The listed questions are rendered by the same seam as the current one.
static void test_template_input_nimble() {
    server_decision_question a = shaped_question(SERVER_DECISION_QUESTION_CHOICE, {"x", "y"});
    server_decision_question b = shaped_question(SERVER_DECISION_QUESTION_NOUL, {"false", "true"});
    a.id = "department";
    b.id = "is_urgent";

    const json inp = decision_template_input(
        { COMMON_DECISION_TYPE_NIMBLE, template_traits({ "A", "B" }), "the state", { a, b }, b, 0, 0 });

    GGML_ASSERT(inp.at("id") == "is_urgent");
    GGML_ASSERT(inp.at("instructions") == b.instructions);
    GGML_ASSERT(inp.at("options").size() == 2);
    GGML_ASSERT(inp.at("options").at(0).at("label") == "A");

    GGML_ASSERT(inp.at("questions").size() == 2);
    GGML_ASSERT(inp.at("questions").at(0).at("id") == "department");
    GGML_ASSERT(inp.at("questions").at(1).at("id") == "is_urgent");
    GGML_ASSERT(inp.at("questions").at(0).at("options").size() == 2);
    GGML_ASSERT(inp.at("questions").at(1).at("type") == "noul");
    // a listed question carries no state, so the prompt does not repeat it
    GGML_ASSERT(!inp.at("questions").at(0).contains("state"));

    // every other type is given its own question only
    for (const common_decision_type type : { COMMON_DECISION_TYPE_OPENJEV, COMMON_DECISION_TYPE_LEV,
                                             COMMON_DECISION_TYPE_KEV, COMMON_DECISION_TYPE_LAYA,
                                             COMMON_DECISION_TYPE_CLEF }) {
        const json single = decision_template_input({ type, template_traits(), "the state", { a, b }, b, 0, 0 });
        GGML_ASSERT(!single.contains("questions"));
        GGML_ASSERT(single.at("id") == "is_urgent");
    }
}

static void test_format_answer_lev() {
    server_decision_context lev;
    lev.type = COMMON_DECISION_TYPE_LEV;

    // a lev noul answer is the weighted index over the nine ratings
    const server_decision_question noul = shaped_question(SERVER_DECISION_QUESTION_NOUL, {"false", "true"});
    std::vector<float> low(9, -30.0f);
    low[0] = 9.0f;
    GGML_ASSERT(std::fabs(lev.format_answer(noul, {low}).at("noul").get<double>() - 0.0) < 1e-6);

    std::vector<float> high(9, -30.0f);
    high[8] = 9.0f;
    GGML_ASSERT(std::fabs(lev.format_answer(noul, {high}).at("noul").get<double>() - 1.0) < 1e-6);

    // a lev choice averages the two variants: mass on the first raw index of both lands half on each end
    const server_decision_question choice = shaped_question(SERVER_DECISION_QUESTION_CHOICE, {"a", "b"});
    const json answer = lev.format_answer(choice, {{9.0f, 0.0f}, {9.0f, 0.0f}});
    GGML_ASSERT(std::fabs(answer.at("probabilities").at("a").get<double>() - 0.5) < 1e-9);
    GGML_ASSERT(std::fabs(answer.at("probabilities").at("b").get<double>() - 0.5) < 1e-9);

    // an openjev noul reads the probability of the true option
    server_decision_context openjev;
    openjev.type = COMMON_DECISION_TYPE_OPENJEV;
    const server_decision_question onoul = shaped_question(SERVER_DECISION_QUESTION_NOUL, {"false", "true"});
    const json oanswer = openjev.format_answer(onoul, {{0.0f, 9.0f}});
    GGML_ASSERT(oanswer.at("noul").get<double>() > 0.99);
}

//
// questions
//

static json question(const char * type, const char * instructions = "x") {
    json q = json::object();
    q["type"] = type;
    if (instructions) {
        q["instructions"] = instructions;
    }
    return q;
}

static json request(const json & q) {
    json questions = json::object();
    questions["q"] = q;
    json body = json::object();
    body["state"]     = "the state";
    body["questions"] = questions;
    return body;
}

static json string_array(size_t n) {
    json arr = json::array();
    for (size_t i = 0; i < n; i++) {
        arr.push_back("level");
    }
    return arr;
}

// the message of the server_invalid_request that the parse must raise, empty if it accepted the body
static std::string parse_error(const json & body, size_t n_options_max = N_OPTIONS_MAX, bool noul_true_first = false) {
    try {
        decision_parse_questions(body, parse_traits(n_options_max, noul_true_first));
    } catch (const server_invalid_request & e) {
        return e.what();
    }
    return "";
}

static bool rejects(const json & body, const char * field, size_t n_options_max = N_OPTIONS_MAX) {
    const std::string message = parse_error(body, n_options_max);
    if (message.find(field) == std::string::npos) {
        fprintf(stderr, "expected the message to name '%s', got '%s'\n", field, message.c_str());
        return false;
    }
    return true;
}

// every body of the parametrized table in tools/server/tests/unit/test_systemone.py, and the field
// its message has to name
static void test_questions_are_refused() {
    json no_questions = json::object();
    no_questions["state"] = "the state";
    GGML_ASSERT(rejects(no_questions, "questions"));

    json empty_questions = json::object();
    empty_questions["state"]     = "the state";
    empty_questions["questions"] = json::object();
    GGML_ASSERT(rejects(empty_questions, "questions"));

    GGML_ASSERT(rejects(request(question("unknown")), "questions.q"));
    GGML_ASSERT(rejects(request(question("choice")), "criteria"));

    json empty_criteria = question("choice");
    empty_criteria["criteria"] = json::object();
    GGML_ASSERT(rejects(request(empty_criteria), "criteria"));

    json one_level = question("score");
    one_level["criteria"] = string_array(1);
    GGML_ASSERT(rejects(request(one_level), "2 to 10"));

    json eleven_levels = question("score");
    eleven_levels["criteria"] = string_array(11);
    GGML_ASSERT(rejects(request(eleven_levels), "2 to 10"));

    json bad_noul_criteria = question("noul");
    bad_noul_criteria["criteria"] = "not an object";
    GGML_ASSERT(rejects(request(bad_noul_criteria), "criteria"));
}

// the limits come from the running model, so they are parameters and not constants
static void test_questions_respect_the_option_limit() {
    json criteria = json::object();
    for (int i = 0; i < 3; i++) {
        criteria["option" + std::to_string(i)] = "x";
    }
    json q = question("choice");
    q["criteria"] = criteria;

    GGML_ASSERT(parse_error(request(q), 3).empty());
    // the message must carry the model's own number, not just the words: the count is what a client
    // has to act on, and test_systemone_option_limit_is_the_models_own asserts the same string
    GGML_ASSERT(rejects(request(q), "\"criteria\" has 3 options, at most 2 are supported", 2));

    // a noul question has two options whatever its criteria say
    GGML_ASSERT(parse_error(request(question("noul")), 2).empty());
    GGML_ASSERT(rejects(request(question("noul")), "at most 1 are supported", 1));
}

static void test_questions_are_accepted() {
    json score = question("score");
    score["criteria"] = string_array(2);
    std::vector<server_decision_question> questions = decision_parse_questions(request(score), parse_traits());
    GGML_ASSERT(questions.size() == 1);
    GGML_ASSERT(questions[0].id == "q");
    GGML_ASSERT(questions[0].type == SERVER_DECISION_QUESTION_SCORE);
    GGML_ASSERT(questions[0].options.size() == 2);
    GGML_ASSERT(questions[0].options[0].key == "0");
    GGML_ASSERT(questions[0].options[1].key == "1");

    json ten = question("score");
    ten["criteria"] = string_array(10);
    GGML_ASSERT(decision_parse_questions(request(ten), parse_traits())[0].options.size() == 10);
}

// the instructions domain: a non-empty string, object, or array is accepted; everything else,
// including the falsy values Jinja would branch on, is refused, so no template can fall back.
//
// An empty string is refused on this count alone: the lev template used to substitute the question id
// for it, so a request carrying one rendered the caller's key into the prompt. A GGUF converted
// before that fallback was dropped still ships the old template, so the refusal has to happen here,
// before any render. The id is never in the template input to begin with.
static void test_instructions_domain() {
    const json accepted[] = {
        "is_this_true",
        json{{"what", "x"}},
        json::array({"a", "b"}),
    };
    for (const json & instructions : accepted) {
        json q = json::object();
        q["type"]         = "noul";
        q["instructions"] = instructions;
        GGML_ASSERT(parse_error(request(q)).empty());
    }

    const json refused[] = {
        json(""),
        json::object(),
        json::array(),
        json(0),
        json(1.5),
        json(false),
        json(true),
        json(),
    };
    for (const json & instructions : refused) {
        json q = json::object();
        q["type"]         = "noul";
        q["instructions"] = instructions;
        GGML_ASSERT(rejects(request(q), "instructions"));
    }

    // absent is refused too, and the message names the field
    GGML_ASSERT(rejects(request(question("noul", nullptr)), "instructions"));
}

// the three accepted instructions types survive parsing unchanged and reach a render, so the
// validator cannot be tightened into dropping a legitimate value silently
static void test_instructions_reach_the_template() {
    const json values[] = {
        "is_this_true",
        json{{"what", "x"}},
        json::array({"a", "b"}),
    };
    const common_chat_template tmpl("{{ instructions if instructions is string else instructions | tojson }}", "", "");
    for (const json & value : values) {
        json q = json::object();
        q["type"]         = "noul";
        q["instructions"] = value;
        const std::vector<server_decision_question> questions =
            decision_parse_questions(request(q), parse_traits());
        GGML_ASSERT(questions[0].instructions.dump() == value.dump());

        const std::string rendered = decision_render_template(tmpl, json{{"instructions", questions[0].instructions}});
        GGML_ASSERT(!rendered.empty());
    }
}

// the order of the choice options is a property of the model too: clef was trained with them in the
// order of their keys, every other type keeps the order the caller wrote
static void test_choice_option_order() {
    json q = json::object();
    q["type"]         = "choice";
    q["instructions"] = "which team";
    q["criteria"]     = json{{"shipping", nullptr}, {"billing", nullptr}, {"technical", nullptr}};

    const std::vector<server_decision_question> as_written =
        decision_parse_questions(request(q), parse_traits());
    GGML_ASSERT(as_written[0].options[0].key == "shipping");
    GGML_ASSERT(as_written[0].options[2].key == "technical");

    const std::vector<server_decision_question> sorted =
        decision_parse_questions(request(q), parse_traits(N_OPTIONS_MAX, false, true));
    GGML_ASSERT(sorted[0].options[0].key == "billing");
    GGML_ASSERT(sorted[0].options[1].key == "shipping");
    GGML_ASSERT(sorted[0].options[2].key == "technical");
}

// the order of the noul options is a property of the model, not of the request
static void test_noul_option_order() {
    const std::vector<server_decision_question> false_first =
        decision_parse_questions(request(question("noul")), parse_traits());
    GGML_ASSERT(false_first[0].options[0].key == "false");
    GGML_ASSERT(false_first[0].options[1].key == "true");

    const std::vector<server_decision_question> true_first =
        decision_parse_questions(request(question("noul")), parse_traits(N_OPTIONS_MAX, true));
    GGML_ASSERT(true_first[0].options[0].key == "true");
    GGML_ASSERT(true_first[0].options[1].key == "false");
}

// The limit counts the images of a request over both channels, and it is checked before an image is
// decoded. So a request one over the limit is refused whole, and the images it did carry are left
// exactly as they were accepted: no entry half-decoded, none dropped, none reordered.
static void test_images_over_the_limit_refuse_with_the_stored_count() {
    const char * data_url = "data:image/png;base64,iVBORw0KGgo=";

    json body = json::object();
    body["state"] = "the state";
    body["images"] = json::array();
    for (size_t i = 0; i < DECISION_MAX_IMAGES; i++) {
        body["images"].push_back(data_url);
    }

    // a request at the limit is accepted whole
    std::vector<raw_buffer> accepted;
    GGML_ASSERT(decision_parse_state(body, accepted) == "the state");
    GGML_ASSERT(accepted.size() == DECISION_MAX_IMAGES);

    // one image more is refused, and the refusal names the maximum the client has to stay under
    body["images"].push_back(data_url);
    std::vector<raw_buffer> files;
    std::string              message;
    try {
        decision_parse_state(body, files);
    } catch (const server_invalid_request & e) {
        message = e.what();
    }
    GGML_ASSERT(!message.empty());
    GGML_ASSERT(message.find(std::to_string(DECISION_MAX_IMAGES)) != std::string::npos);

    // and the accepted images survive the refusal untouched
    GGML_ASSERT(files.size() == accepted.size());
    GGML_ASSERT(files == accepted);
}

// the two channels share one count, so the images field cannot be topped up past the limit by images
// hidden in a chat-message state
static void test_the_image_limit_spans_both_channels() {
    const char * data_url = "data:image/png;base64,iVBORw0KGgo=";

    json msg = json::object();
    msg["role"] = "user";
    msg["content"] = json::array({
        { { "type", "image_url" }, { "image_url", json::object({{ "url", data_url }}) } },
    });

    json body = json::object();
    body["images"] = json::array();
    for (size_t i = 0; i < DECISION_MAX_IMAGES - 1; i++) {
        body["images"].push_back(data_url);
    }
    body["state"] = json::array({ msg });

    // one image in the state brings the request to the limit and is accepted
    std::vector<raw_buffer> files;
    decision_parse_state(body, files);
    GGML_ASSERT(files.size() == DECISION_MAX_IMAGES);

    // a second one puts the request over it, and is refused even though the images field is under
    body["state"] = json::array({ msg, msg });
    bool refused = false;
    try {
        std::vector<raw_buffer> over;
        decision_parse_state(body, over);
    } catch (const server_invalid_request &) {
        refused = true;
    }
    GGML_ASSERT(refused);
}

static void test_state_images_are_taken_out() {
    const char * data_url = "data:image/png;base64,iVBORw0KGgo=";

    json body = json::object();
    body["state"]  = "no images here";
    body["images"] = json::array({ data_url });
    std::vector<raw_buffer> files;
    const json plain = decision_parse_state(body, files);
    GGML_ASSERT(plain == "no images here");
    GGML_ASSERT(files.size() == 1);

    // an image part of a chat message is taken out too, and the rest of the state is kept
    json messages = json::array();
    json msg = json::object();
    msg["role"] = "user";
    msg["content"] = json::array({
        { { "type", "image_url" }, { "image_url", json::object({{ "url", data_url }}) } },
        { { "type", "text" }, { "text", "what is this" } },
    });
    messages.push_back(msg);

    json chat = json::object();
    chat["state"] = messages;
    std::vector<raw_buffer> chat_files;
    const json state = decision_parse_state(chat, chat_files);
    GGML_ASSERT(chat_files.size() == 1);
    GGML_ASSERT(state.at(0).at("content").size() == 1);
    GGML_ASSERT(state.at(0).at("content").at(0).at("text") == "what is this");
}

// a body with no state is valid JSON with a missing field, so the semantic refusal (422) has to be
// raised before decision_parse_state reaches body.at("state"), which would be the json error that
// ex_wrapper answers 400
static void test_state_is_required() {
    json no_state = json::object();
    no_state["questions"] = request(question("noul")).at("questions");
    std::vector<raw_buffer> files;

    bool rejected = false;
    try {
        decision_parse_state(no_state, files);
    } catch (const server_invalid_request & e) {
        rejected = true;
        GGML_ASSERT(std::string(e.what()).find("state") != std::string::npos);
    }
    GGML_ASSERT(rejected);
    GGML_ASSERT(files.empty());

    json null_state = json::object();
    null_state["state"]     = json();
    null_state["questions"] = no_state.at("questions");
    rejected = false;
    try {
        decision_parse_state(null_state, files);
    } catch (const server_invalid_request &) {
        rejected = true;
    }
    GGML_ASSERT(rejected);
}

// a data URL that cannot be decoded is valid JSON with a bad value, so the semantic refusal (422)
// must be raised, never the std::invalid_argument that ex_wrapper would map to 400
static void test_state_rejects_malformed_data_url() {
    json body = json::object();
    body["state"]  = "x";
    body["images"] = json::array({ "data:image/png;base64" }); // no comma
    std::vector<raw_buffer> files;

    bool rejected = false;
    try {
        decision_parse_state(body, files);
    } catch (const server_invalid_request & e) {
        rejected = std::string(e.what()).find("base64") != std::string::npos;
    } catch (const std::exception & e) {
        fprintf(stderr, "expected server_invalid_request, got: %s\n", e.what());
    }
    GGML_ASSERT(rejected);
}

//
// grouping
//

// a decision task whose prompt is n_tokens long, starting at token first. Tasks that share a
// prefix are built by giving them the same first and a length that covers it
static server_task decision_task(int id, size_t n_tokens, llama_token first = 0) {
    llama_tokens tokens;
    for (size_t i = 0; i < n_tokens; i++) {
        tokens.push_back(first + (llama_token) i);
    }
    server_task task(SERVER_TASK_TYPE_DECISION);
    task.id    = id;
    task.tokens = server_tokens(tokens, false);
    return task;
}

// a group is its parent followed by its children, so flattening gives the tasks as they were given
static std::vector<int> flattened_ids(const server_decision_tasks & grouped) {
    std::vector<int> ids;
    for (const auto & task : grouped.tasks) {
        ids.push_back(task.id);
        for (const auto & child : task.child_tasks) {
            ids.push_back(child.id);
        }
    }
    return ids;
}

// the number of prompt tokens the batch actually evaluates, read off the grouped tasks: the parent
// pays for its whole prompt, each child only for the tokens past the prefix it inherits
static int32_t tokens_evaluated(const server_decision_tasks & grouped) {
    int32_t n = 0;
    for (const auto & task : grouped.tasks) {
        n += task.n_tokens();
        for (const auto & child : task.child_tasks) {
            n += child.n_tokens() - task.n_tokens_shared;
        }
    }
    return n;
}

// the shared prefix must be discounted exactly once per child that inherits it: what the route
// reports, the sum of the prompt lengths minus n_shared, has to be what the batch evaluates
static void test_grouping_counts_the_prefix_once() {
    std::mt19937 rng(1234);
    for (int round = 0; round < 200; round++) {
        const size_t n_tasks = 1 + rng() % 12;
        const size_t n_slots = 1 + rng() % 5;

        std::vector<server_task> tasks;
        for (size_t i = 0; i < n_tasks; i++) {
            tasks.push_back(decision_task((int) i, 2 + rng() % 8));
        }

        const server_decision_tasks grouped = server_decision_group_tasks(std::move(tasks), n_slots);

        int32_t reported = 0;
        for (const auto & task : grouped.tasks) {
            reported += task.n_tokens();
            for (const auto & child : task.child_tasks) {
                reported += child.n_tokens();
            }
        }

        GGML_ASSERT(reported - grouped.n_shared == tokens_evaluated(grouped));
        // the two never disagree because neither is allowed to go negative
        GGML_ASSERT(grouped.n_shared <= reported);
    }
}

// an empty prompt has no prefix to share: grouping it must yield no discount, never a wrapped size
static void test_grouping_empty_prompt_shares_nothing() {
    std::vector<server_task> tasks;
    tasks.push_back(decision_task(0, 0));
    tasks.push_back(decision_task(1, 0));
    tasks.push_back(decision_task(2, 5));
    const server_decision_tasks grouped = server_decision_group_tasks(std::move(tasks), 3);
    GGML_ASSERT(grouped.n_shared == 0);
    GGML_ASSERT(flattened_ids(grouped).size() == 3);
    GGML_ASSERT(tokens_evaluated(grouped) == 5);
}

// saturating arithmetic used by grouping and the usage envelope: overflow clamps, never wraps
static void test_saturating_arithmetic() {
    const size_t max = std::numeric_limits<size_t>::max();
    GGML_ASSERT(decision_saturating_add(max, 1) == max);
    GGML_ASSERT(decision_saturating_add(max - 1, 2) == max);
    GGML_ASSERT(decision_saturating_add(2, 3) == 5);
    GGML_ASSERT(decision_saturating_mul(max, 2) == max);
    GGML_ASSERT(decision_saturating_mul(max, max) == max);
    GGML_ASSERT(decision_saturating_mul(3, 4) == 12);
    GGML_ASSERT(decision_saturating_mul(0, max) == 0);
}

// the usage envelope bills the prompt sum minus the shared discount, floored at zero:
// a discount past the sum (grouping bug or wrapped counter) must not bill negative
static void test_usage_never_negative() {
    GGML_ASSERT(decision_usage_input_tokens(10, 3) == 7);
    GGML_ASSERT(decision_usage_input_tokens(10, 10) == 0);
    GGML_ASSERT(decision_usage_input_tokens(10, 11) == 0);
    GGML_ASSERT(decision_usage_input_tokens(0, 0) == 0);
    GGML_ASSERT(decision_usage_input_tokens(5, std::numeric_limits<size_t>::max()) == 0);
    GGML_ASSERT(decision_usage_input_tokens(std::numeric_limits<size_t>::max(),
                                            std::numeric_limits<size_t>::max()) == 0);
}

// the budget and the bill disagree by exactly the shared discount: the budget counts every task
// before grouping (work held while building), the envelope bills evaluated tokens (work done)
static void test_budget_sees_pre_group_billing_sees_post_group() {
    std::vector<server_task> tasks;
    for (int i = 0; i < 3; i++) {
        tasks.push_back(decision_task(i, 7)); // identical prompts share all but one token
    }
    int32_t pre_group = 0;
    for (const auto & task : tasks) {
        pre_group += task.n_tokens();
    }
    const server_decision_tasks grouped = server_decision_group_tasks(std::move(tasks), 3);
    GGML_ASSERT(grouped.tasks.size() == 1);
    const size_t billed = decision_usage_input_tokens((size_t) pre_group, (size_t) grouped.n_shared);
    GGML_ASSERT(pre_group - (int32_t) billed == grouped.n_shared);
    GGML_ASSERT(billed == (size_t) tokens_evaluated(grouped));
}

static void test_grouping_preserves_order() {
    for (size_t n_slots = 1; n_slots <= 5; n_slots++) {
        for (size_t n_tasks = 1; n_tasks <= 9; n_tasks++) {
            std::vector<server_task> tasks;
            for (size_t i = 0; i < n_tasks; i++) {
                tasks.push_back(decision_task((int) i, 2 + i % 4));
            }
            const std::vector<int> ids = flattened_ids(server_decision_group_tasks(std::move(tasks), n_slots));
            GGML_ASSERT(ids.size() == n_tasks);
            for (size_t i = 0; i < n_tasks; i++) {
                GGML_ASSERT(ids[i] == (int) i);
            }
        }
    }
}

// The whole grouping policy: a group holds one task per slot the server has, so a group is full
// until the tasks run out. The loops below observe that through the number of parents a batch
// becomes, and they cross-check it against the tasks that survive.
static void test_grouping_respects_n_slots() {
    for (size_t n_slots = 0; n_slots <= 4; n_slots++) {
        for (size_t n_tasks = 1; n_tasks <= 9; n_tasks++) {
            std::vector<server_task> tasks;
            for (size_t i = 0; i < n_tasks; i++) {
                tasks.push_back(decision_task((int) i, 4 + i % 3));
            }

            const server_decision_tasks grouped = server_decision_group_tasks(std::move(tasks), n_slots);

            // the width a group can reach is the slot count, or one task when there are no slots
            const size_t width    = std::max(n_slots, (size_t) 1);
            const size_t expected = (n_tasks + width - 1) / width;
            GGML_ASSERT(grouped.tasks.size() == expected);

            // no task is lost or duplicated, and no group is wider than the slots allow
            GGML_ASSERT(flattened_ids(grouped).size() == n_tasks);
            for (const auto & task : grouped.tasks) {
                GGML_ASSERT(task.child_tasks.size() < width);
                if (task.child_tasks.empty()) {
                    GGML_ASSERT(!task.is_parent());
                    GGML_ASSERT(!task.is_child());
                    GGML_ASSERT(task.n_tokens_shared == 0);
                } else {
                    GGML_ASSERT(task.is_parent());
                    GGML_ASSERT(!task.is_child());
                    GGML_ASSERT(task.n_tokens_shared > 0);
                }
            }
        }
    }

    // a request never groups more tasks than it has, so an absurd slot count must not wrap around
    std::vector<server_task> five;
    for (size_t i = 0; i < 5; i++) {
        five.push_back(decision_task((int) i, 5));
    }
    GGML_ASSERT(server_decision_group_tasks(std::move(five), SIZE_MAX).tasks.size() == 1);
}

// a task always keeps at least one token of its own, so a prompt of one token can never be a child,
// and it holds back every other task of the group it lands in
static void test_grouping_needs_a_token_of_its_own() {
    std::vector<server_task> tasks;
    tasks.push_back(decision_task(0, 6));
    tasks.push_back(decision_task(1, 1));
    tasks.push_back(decision_task(2, 6));
    tasks.push_back(decision_task(3, 1));
    const server_decision_tasks grouped = server_decision_group_tasks(std::move(tasks), 4);
    for (const auto & task : grouped.tasks) {
        for (const auto & child : task.child_tasks) {
            GGML_ASSERT(child.n_tokens() > 1);
        }
    }
    GGML_ASSERT(grouped.n_shared == 0);
    GGML_ASSERT(grouped.tasks.size() == 4);
}

static void test_grouping_degenerate_cases() {
    // one task has nothing to share with
    std::vector<server_task> single;
    single.push_back(decision_task(0, 5));
    GGML_ASSERT(server_decision_group_tasks(std::move(single), 4).n_shared == 0);

    // without slots there is no group to fill, so every task runs on its own
    for (size_t n_slots : { (size_t) 0, (size_t) 1 }) {
        std::vector<server_task> tasks;
        for (size_t i = 0; i < 4; i++) {
            tasks.push_back(decision_task((int) i, 5));
        }
        const server_decision_tasks grouped = server_decision_group_tasks(std::move(tasks), n_slots);
        GGML_ASSERT(grouped.n_shared == 0);
        GGML_ASSERT(grouped.tasks.size() == 4);
    }

    // prompts that share nothing share nothing, whatever the first token is
    std::vector<server_task> disjoint;
    for (size_t i = 0; i < 3; i++) {
        disjoint.push_back(decision_task((int) i, 5, (llama_token) (100 * (i + 1))));
    }
    GGML_ASSERT(server_decision_group_tasks(std::move(disjoint), 3).n_shared == 0);

    // identical prompts share all but their last token
    std::vector<server_task> same;
    for (size_t i = 0; i < 3; i++) {
        same.push_back(decision_task((int) i, 7));
    }
    const server_decision_tasks identical = server_decision_group_tasks(std::move(same), 3);
    GGML_ASSERT(identical.n_shared == 6 * 2);
    GGML_ASSERT(identical.tasks.size() == 1);
    GGML_ASSERT(identical.tasks[0].child_tasks.size() == 2);
}

// a model that cannot continue from another task's prompt is posted as it is, so the route can
// subtract n_shared unconditionally
static void test_shared_prompt_by_model_type() {
    const common_decision_type types[] = {
        COMMON_DECISION_TYPE_NONE,
        COMMON_DECISION_TYPE_OPENJEV,
        COMMON_DECISION_TYPE_LEV,
        COMMON_DECISION_TYPE_KEV,
        COMMON_DECISION_TYPE_LAYA,
        COMMON_DECISION_TYPE_NIMBLE,
        COMMON_DECISION_TYPE_CLEF,
        COMMON_DECISION_TYPE_UNKNOWN,
    };
    for (const common_decision_type type : types) {
        server_decision_context decision;
        decision.type = type;
        const bool expected =
            type == COMMON_DECISION_TYPE_OPENJEV ||
            type == COMMON_DECISION_TYPE_LEV     ||
            type == COMMON_DECISION_TYPE_KEV     ||
            type == COMMON_DECISION_TYPE_NIMBLE;
        GGML_ASSERT(decision.can_share_prompt() == expected);
        // clef answers the whole request from one prompt, so it never groups and never shares
        GGML_ASSERT(decision.is_joint() == (type == COMMON_DECISION_TYPE_CLEF));
    }
}

//
// variant averaging
//

// a scores matrix whose two variants both put all their mass on raw index k. The rest sit far enough
// below the peak that the softmax over them is zero to double precision, so the expected answers are exact.
static const float ONE_HOT_PEAK   =  9.0f;
static const float ONE_HOT_REST   = -30.0f;

static std::vector<std::vector<float>> mass_on_index(size_t n, size_t raw) {
    std::vector<float> s(n, ONE_HOT_REST);
    s[raw] = ONE_HOT_PEAK;
    return {s, s};
}

// the debiasing property, and the whole reason lev runs two variants: a matrix with its mass on one
// raw index and the same matrix reversed must give the same answer, symmetric in the mapped indices.
static void test_averaging_cancels_the_first_listed_preference() {
    for (const size_t n : { (size_t) 2, (size_t) 3, (size_t) 5 }) {
        for (size_t k = 0; k < n; k++) {
            const std::vector<double> forward = decision_average_variants(mass_on_index(n, k), n, 1.0f);
            const std::vector<double> reversed = decision_average_variants(mass_on_index(n, n - 1 - k), n, 1.0f);

            for (size_t i = 0; i < n; i++) {
                GGML_ASSERT(std::abs(forward[i] - reversed[i]) < 1e-6);
            }
            // the mass the model put on the two ends has to be split evenly between them
            GGML_ASSERT(std::abs(forward[0] - forward[n - 1]) < 1e-6);
        }
    }

    // the concrete case: mass on raw index 0 in both variants means the first-listed option of
    // variant 0 and the last-listed of variant 1, so it lands half on each end
    const std::vector<double> probs = decision_average_variants(mass_on_index(3, 0), 3, 1.0f);
    GGML_ASSERT(std::abs(probs[0] - 0.5) < 1e-9);
    GGML_ASSERT(std::abs(probs[1] - 0.0) < 1e-9);
    GGML_ASSERT(std::abs(probs[2] - 0.5) < 1e-9);
}

// the averaging has to be a distribution whatever it is handed
static void test_averaging_sums_to_one() {
    const std::vector<std::vector<std::vector<float>>> inputs = {
        {{0.0f, 0.0f}, {0.0f, 0.0f}},
        {{9.0f, 0.0f}, {9.0f, 0.0f}},
        {{0.0f, 9.0f}, {0.0f, 9.0f}},
        {{-3.0f, 1.0f, 2.5f}, {1.0f, -3.0f, 0.5f}},
        {{1.0f}},
        {{1.0f, 1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 1.0f, 1.0f}},
    };
    for (const auto & scores : inputs) {
        const std::vector<double> probs = decision_average_variants(scores, scores[0].size(), 1.0f);
        double sum = 0.0;
        for (const double p : probs) {
            GGML_ASSERT(p >= 0.0 && p <= 1.0);
            sum += p;
        }
        GGML_ASSERT(std::abs(sum - 1.0) < 1e-9);
    }
}

// equal scores carry no preference, so every option has to come out equal
static void test_averaging_uniform_scores_give_a_uniform_distribution() {
    for (const size_t n : { (size_t) 2, (size_t) 3, (size_t) 7 }) {
        const std::vector<float> flat(n, 0.25f);
        const std::vector<double> probs = decision_average_variants({flat, flat}, n, 1.0f);
        for (const double p : probs) {
            GGML_ASSERT(std::abs(p - 1.0 / n) < 1e-12);
        }
    }
}

// a larger temperature must flatten the answer. A single variant is used for the strict check,
// because averaging can mask the effect when the two variants' peaks map onto each other.
static void test_averaging_honours_the_temperature() {
    const std::vector<std::vector<float>> one_variant = {{6.0f, 0.0f, 0.0f}};
    double previous = 2.0;
    for (const float temperature : { 0.5f, 1.0f, 2.0f, 8.0f }) {
        const std::vector<double> probs = decision_average_variants(one_variant, 3, temperature);
        const double top = *std::max_element(probs.begin(), probs.end());
        GGML_ASSERT(top < previous);
        previous = top;
    }

    // a temperature so high the scores barely matter, so every option is close to its share
    for (const double p : decision_average_variants(one_variant, 3, 1e6f)) {
        GGML_ASSERT(std::abs(p - 1.0 / 3.0) < 1e-3);
    }

    // and the flattening still holds once the variants are averaged
    const std::vector<std::vector<float>> two_variants = {{6.0f, 0.0f, 0.0f}, {0.0f, 6.0f, 0.0f}};
    const auto top_of = [&two_variants](float temperature) {
        const std::vector<double> probs = decision_average_variants(two_variants, 3, temperature);
        return *std::max_element(probs.begin(), probs.end());
    };
    GGML_ASSERT(top_of(8.0f) < top_of(0.5f));
}

// a result whose shape disagrees with the question is a fault of the server, not a silent answer
// a joint head returns one vector for the whole request, and each question reads its own options out
// of it in order. The offset is what makes that readable, and a short vector has to be an error the
// route can report rather than an assertion that takes the server down.
static void test_joint_scores_are_sliced_in_order() {
    const server_decision_question a = shaped_question(SERVER_DECISION_QUESTION_CHOICE, {"x", "y"});
    const server_decision_question b = shaped_question(SERVER_DECISION_QUESTION_NOUL, {"false", "true"});
    const std::vector<float> all = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };

    const std::vector<float> first = decision_joint_scores(all, 0, a);
    GGML_ASSERT(first == std::vector<float>({ 1.0f, 2.0f }));
    const std::vector<float> second = decision_joint_scores(all, 2, b);
    GGML_ASSERT(second == std::vector<float>({ 3.0f, 4.0f }));

    // the slice is exactly as wide as the options, so format_answer reads it like any other variant
    server_decision_context clef;
    clef.type = COMMON_DECISION_TYPE_CLEF;
    GGML_ASSERT(clef.n_variants(a) == 1);
    clef.format_answer(a, { first });
    clef.format_answer(b, { second });

    // one score short of the last option of the request
    bool threw = false;
    try {
        decision_joint_scores({ 1.0f, 2.0f, 3.0f }, 2, b);
    } catch (const std::exception &) {
        threw = true;
    }
    GGML_ASSERT(threw);
}

static void test_averaging_rejects_a_mismatched_result() {
    bool threw = false;
    try {
        decision_average_variants({{0.0f, 0.0f, 0.0f}}, 2, 1.0f);
    } catch (const std::exception &) {
        threw = true;
    }
    GGML_ASSERT(threw);
}

//
// admission control
//

// the depth source counts a parent and each child, and only tasks of the counted type
static void test_queue_depth_counts_decision_tasks() {
    server_queue queue;

    server_task parent(SERVER_TASK_TYPE_DECISION);
    parent.id = queue.get_new_id();
    parent.add_child(parent.id, queue.get_new_id());
    parent.add_child(parent.id, queue.get_new_id());
    queue.post(std::move(parent));
    GGML_ASSERT(queue.queued_count(SERVER_TASK_TYPE_DECISION) == 3);

    // a completion task does not move the decision count
    server_task completion(SERVER_TASK_TYPE_COMPLETION);
    completion.id = queue.get_new_id();
    queue.post(std::move(completion));
    GGML_ASSERT(queue.queued_count(SERVER_TASK_TYPE_DECISION) == 3);
    GGML_ASSERT(queue.queued_count(SERVER_TASK_TYPE_COMPLETION) == 1);

    // a deferred decision task is counted
    server_task deferred(SERVER_TASK_TYPE_DECISION);
    deferred.id = queue.get_new_id();
    queue.defer(std::move(deferred));
    GGML_ASSERT(queue.queued_count(SERVER_TASK_TYPE_DECISION) == 4);

    // a deferred parent counts with its children too
    server_task deferred_parent(SERVER_TASK_TYPE_DECISION);
    deferred_parent.id = queue.get_new_id();
    deferred_parent.add_child(deferred_parent.id, queue.get_new_id());
    queue.defer(std::move(deferred_parent));
    GGML_ASSERT(queue.queued_count(SERVER_TASK_TYPE_DECISION) == 6);
}

// n decision tasks, each with an id of its own
static std::vector<server_task> decision_tasks(server_queue & queue, size_t n) {
    std::vector<server_task> tasks;
    for (size_t i = 0; i < n; i++) {
        server_task task(SERVER_TASK_TYPE_DECISION);
        task.id = queue.get_new_id();
        tasks.push_back(std::move(task));
    }
    return tasks;
}

// The gate reads the depth before the batch and admits the batch whole: an idle server takes a
// request of any size, a saturated one refuses every request. The check and the enqueue share one
// lock, so the depth reported on a refusal is the one the decision was made on.
static void test_try_post_gates_on_depth() {
    server_queue queue;

    // an idle server admits a batch whose own weight is many times the cap
    GGML_ASSERT(queue.try_post(decision_tasks(queue, 10), SERVER_TASK_TYPE_DECISION, 2, false));
    GGML_ASSERT(queue.queued_count(SERVER_TASK_TYPE_DECISION) == 10);

    // at or above the cap, a batch of any weight is refused and changes nothing
    GGML_ASSERT(!queue.try_post(decision_tasks(queue, 1), SERVER_TASK_TYPE_DECISION, 2, false));
    GGML_ASSERT(!queue.try_post(decision_tasks(queue, 100), SERVER_TASK_TYPE_DECISION, 2, false));
    GGML_ASSERT(queue.queued_count(SERVER_TASK_TYPE_DECISION) == 10);

    // a batch that lands exactly on the cap is admitted from below, and locks the queue after
    server_queue idle;
    GGML_ASSERT(idle.try_post(decision_tasks(idle, 2), SERVER_TASK_TYPE_DECISION, 2, false));
    GGML_ASSERT(idle.queued_count(SERVER_TASK_TYPE_DECISION) == 2);
    GGML_ASSERT(!idle.try_post(decision_tasks(idle, 1), SERVER_TASK_TYPE_DECISION, 2, false));

    // 0 is unlimited, it never refuses
    GGML_ASSERT(queue.try_post(decision_tasks(queue, 5), SERVER_TASK_TYPE_DECISION, 0, false));
    GGML_ASSERT(queue.queued_count(SERVER_TASK_TYPE_DECISION) == 15);
}

// the depth is read under the same lock that decides, so on a refusal it is at or above the cap and
// equal to the count the queue reports, and a success leaves it untouched
static void test_try_post_reports_depth_at_refusal() {
    server_queue queue;
    size_t depth = 123;

    GGML_ASSERT(queue.try_post(decision_tasks(queue, 5), SERVER_TASK_TYPE_DECISION, 2, false, &depth));
    GGML_ASSERT(depth == 123);

    GGML_ASSERT(!queue.try_post(decision_tasks(queue, 1), SERVER_TASK_TYPE_DECISION, 5, false, &depth));
    GGML_ASSERT(depth == queue.queued_count(SERVER_TASK_TYPE_DECISION));

    // the out-parameter is optional, the refusal works the same without it
    GGML_ASSERT(!queue.try_post(decision_tasks(queue, 1), SERVER_TASK_TYPE_DECISION, 5, false));
}

// A bounded scan stops once the counted weight reaches the bound, so a saturated check does not
// have to walk a backlog that cannot add to the answer. Tasks of another type carry no weight, so
// they do not bring the scan closer to the bound.
static void test_queued_count_stops_at_cap() {
    server_queue queue;

    // one heavy task in front of a large backlog, so a full scan and a bounded one must differ
    server_task heavy(SERVER_TASK_TYPE_DECISION);
    heavy.id = queue.get_new_id();
    for (int i = 0; i < 100; i++) {
        heavy.add_child(heavy.id, queue.get_new_id());
    }
    queue.post(std::move(heavy));
    for (int i = 0; i < 50; i++) {
        queue.post(decision_tasks(queue, 1));
    }

    const size_t true_count = queue.queued_count(SERVER_TASK_TYPE_DECISION);
    GGML_ASSERT(true_count == 101 + 50);

    // the scan stops inside the heavy task, it does not walk the backlog
    GGML_ASSERT(queue.queued_count(SERVER_TASK_TYPE_DECISION, 5) == 101);

    // a backlog of tasks of another type adds nothing, and the scan cannot skip past it: it walks to
    // the end of the queue before it can report a count at all
    server_queue other;
    for (int i = 0; i < 200; i++) {
        server_task task(SERVER_TASK_TYPE_COMPLETION);
        task.id = other.get_new_id();
        other.post(std::move(task));
    }
    GGML_ASSERT(other.queued_count(SERVER_TASK_TYPE_DECISION) == 0);
    GGML_ASSERT(other.queued_count(SERVER_TASK_TYPE_DECISION, 8) == 0);

    // the bound is where the scan stops, so a bounded answer is never above the bound and never
    // above the true count either: it is the count reached by the time the bound is hit
    other.post(decision_tasks(other, 8));
    GGML_ASSERT(other.queued_count(SERVER_TASK_TYPE_DECISION) == 8);
    GGML_ASSERT(other.queued_count(SERVER_TASK_TYPE_DECISION, 8) == 8);
    GGML_ASSERT(other.queued_count(SERVER_TASK_TYPE_DECISION, 7) == 7);
    GGML_ASSERT(other.queued_count(SERVER_TASK_TYPE_DECISION, 1) == 1);
}

// the cap is per counted type: completion backlog does not consume the decision cap
static void test_try_post_mixed_types() {
    server_queue queue;

    std::vector<server_task> completions;
    for (int i = 0; i < 10; i++) {
        server_task task(SERVER_TASK_TYPE_COMPLETION);
        task.id = queue.get_new_id();
        completions.push_back(std::move(task));
    }
    queue.post(std::move(completions));

    // the decision batch is admitted on the depth of decisions only
    GGML_ASSERT(queue.try_post(decision_tasks(queue, 2), SERVER_TASK_TYPE_DECISION, 2, false));
    GGML_ASSERT(queue.queued_count(SERVER_TASK_TYPE_DECISION) == 2);

    // at the cap, another decision batch is refused, and the completion backlog is untouched
    GGML_ASSERT(!queue.try_post(decision_tasks(queue, 1), SERVER_TASK_TYPE_DECISION, 2, false));
    GGML_ASSERT(queue.queued_count(SERVER_TASK_TYPE_COMPLETION) == 10);
}

// the smallest concrete result, only needed to observe the reader's id registration
struct test_task_result : server_task_result {
    json to_json() override { return json(); }
};

// posting a batch reserves one state and one index per task and child, and registers every id to
// receive a result: a result for a registered id is delivered, the ids are the caller's own
static void test_post_tasks_registers_ids() {
    server_queue queue;
    server_response results;
    server_response_reader reader(queue, results, 0);

    std::vector<server_task> tasks;
    server_task parent(SERVER_TASK_TYPE_DECISION);
    const int parent_id = queue.get_new_id();
    parent.id = parent_id;
    parent.add_child(parent_id, queue.get_new_id());
    parent.add_child(parent_id, queue.get_new_id());
    tasks.push_back(std::move(parent));

    server_task sibling(SERVER_TASK_TYPE_DECISION);
    const int sibling_id = queue.get_new_id();
    sibling.id = sibling_id;
    tasks.push_back(std::move(sibling));

    const std::unordered_set<int> expected = server_task::get_list_id(tasks);
    GGML_ASSERT(expected.size() == 4);

    reader.post_tasks(std::move(tasks), false);

    GGML_ASSERT(reader.id_tasks == expected);
    GGML_ASSERT(reader.states.size() == expected.size());
    GGML_ASSERT(queue.queued_count(SERVER_TASK_TYPE_DECISION) == 4);

    // a result for a registered id is delivered, so the id was on the waiting list before the post
    auto res = std::make_unique<test_task_result>();
    res->id = sibling_id;
    results.send(std::move(res));
    server_task_result_ptr got = results.recv_with_timeout(reader.id_tasks, 0);
    GGML_ASSERT(got != nullptr);
    GGML_ASSERT(got->id == sibling_id);

    // no scheduler runs here, so mark the result received to keep ~reader from cancelling the task
    reader.received_count = reader.id_tasks.size();
}

// a refused reader post leaves no waiting id or state behind: the ids are registered before the
// post, so a refusal has to undo them. The in-place index assignment has already run by then.
static void test_try_post_tasks_cleans_up_on_refusal() {
    server_queue queue;
    server_response results;
    server_response_reader reader(queue, results, 1);

    std::vector<server_task> fill;
    for (int i = 0; i < 2; i++) {
        server_task task(SERVER_TASK_TYPE_DECISION);
        task.id = queue.get_new_id();
        fill.push_back(std::move(task));
    }
    GGML_ASSERT(queue.try_post(std::move(fill), SERVER_TASK_TYPE_DECISION, 2, false));

    std::vector<server_task> refused;
    server_task task(SERVER_TASK_TYPE_DECISION);
    const int refused_id = queue.get_new_id();
    task.id = refused_id;
    task.add_child(refused_id, queue.get_new_id());
    refused.push_back(std::move(task));
    GGML_ASSERT(!reader.try_post_tasks(std::move(refused), SERVER_TASK_TYPE_DECISION, 2, false));
    GGML_ASSERT(reader.id_tasks.empty());
    GGML_ASSERT(reader.states.empty());
    GGML_ASSERT(queue.queued_count(SERVER_TASK_TYPE_DECISION) == 2);
    // the refusal did not move the batch, so the assigned indices are still readable
    GGML_ASSERT(refused[0].index == 0);
    GGML_ASSERT(refused[0].child_tasks[0].index == 1);

    // the refused id was unregistered, so a late result for it is dropped instead of queued
    auto stray = std::make_unique<test_task_result>();
    stray->id = refused_id;
    results.send(std::move(stray));
    GGML_ASSERT(results.recv_with_timeout({ refused_id }, 0) == nullptr);

    // a sufficient cap then succeeds and registers the id
    std::vector<server_task> ok;
    server_task task_ok(SERVER_TASK_TYPE_DECISION);
    const int ok_id = queue.get_new_id();
    task_ok.id = ok_id;
    ok.push_back(std::move(task_ok));
    GGML_ASSERT(reader.try_post_tasks(std::move(ok), SERVER_TASK_TYPE_DECISION, 3, false));
    GGML_ASSERT(reader.id_tasks.size() == 1);
    GGML_ASSERT(reader.id_tasks.count(ok_id) == 1);
    GGML_ASSERT(reader.states.size() == 1);
    GGML_ASSERT(queue.queued_count(SERVER_TASK_TYPE_DECISION) == 3);

    // no scheduler runs here, so mark the result received to keep ~reader from cancelling the task
    reader.received_count = reader.id_tasks.size();
}

// a request of n noul questions, the shape the admission tests use: one task each
static server_decision_request request_of_n_noul(size_t n) {
    server_decision_request req;
    req.state = "the state";
    for (size_t i = 0; i < n; i++) {
        server_decision_question q = shaped_question(SERVER_DECISION_QUESTION_NOUL, {"false", "true"});
        q.id = "q" + std::to_string(i);
        req.questions.push_back(std::move(q));
    }
    return req;
}

// the task bound is sized by the server: one task per token of one slot context. An explicit
// token budget only ever raises it, never lowers it into tiny-budget range; 0 is unlimited.
static void test_task_budget_is_a_bound() {
    GGML_ASSERT(decision_task_budget(-1, 4, 1024) == 4096);
    GGML_ASSERT(decision_task_budget(-1, 2, 512) == 1024);
    // no slot or context still derives one of each, like the token budget
    GGML_ASSERT(decision_task_budget(-1, 0, 0) == 1);
    // a small explicit budget keeps the derived bound, so the token gate still fires first there
    GGML_ASSERT(decision_task_budget(48, 2, 512) == 1024);
    GGML_ASSERT(decision_task_budget(800, 4, 1024) == 4096);
    // a large one raises it at one task per eight tokens
    GGML_ASSERT(decision_task_budget(100000, 4, 1024) == 12500);
    GGML_ASSERT(decision_task_budget(0, 4, 1024) == 0);
}

// the pre-render count: one task for a joint head, else one per variant, with no render involved
static void test_task_count_is_predicted_without_rendering() {
    server_decision_context lev;
    lev.type = COMMON_DECISION_TYPE_LEV;
    server_decision_request req = request_of_n_noul(3);
    req.questions.push_back(shaped_question(SERVER_DECISION_QUESTION_CHOICE, {"a", "b", "c"}));
    req.questions.push_back(shaped_question(SERVER_DECISION_QUESTION_CHOICE, {"a"}));
    // lev runs a choice of several options in 2 variants, everything else in 1
    GGML_ASSERT(lev.count_tasks(req) == 3 + 2 + 1);

    server_decision_context clef;
    clef.type = COMMON_DECISION_TYPE_CLEF;
    GGML_ASSERT(clef.count_tasks(req) == 1);

    GGML_ASSERT(request_of_n_noul(0).questions.empty());
    GGML_ASSERT(lev.count_tasks(request_of_n_noul(0)) == 0);
}

// an over-count request is refused before any render: this context has no template or vocab,
// so reaching fill_task would crash instead of throwing
static void test_over_count_request_is_refused_before_render() {
    server_decision_context ctx;
    ctx.type = COMMON_DECISION_TYPE_OPENJEV;
    const server_decision_request req = request_of_n_noul(10);

    int next = 0;
    const auto next_id = [&next]() { return next++; };
    bool threw = false;
    try {
        ctx.build_tasks(req, next_id, nullptr, mtmd_helper_init_opt{}, 4, 1, 0); // bound is 1 task
    } catch (const server_status_error & e) {
        threw = true;
        GGML_ASSERT(e.type == ERROR_TYPE_REQUEST_TOO_LARGE);
        GGML_ASSERT(std::string(e.what()).find("10") != std::string::npos);
        GGML_ASSERT(std::string(e.what()).find("tasks") != std::string::npos);
    }
    GGML_ASSERT(threw);
    GGML_ASSERT(next == 0); // no task id was handed out, so nothing was built

    // at the bound the same request is counted, not refused by this gate (render would follow)
    GGML_ASSERT(ctx.count_tasks(req) == 10);
    GGML_ASSERT(10 > 1);
}

// a positive cap is itself, a negative cap derives from the slots, and 0 is unlimited
static void test_decision_queue_cap_is_a_bound() {
    GGML_ASSERT(decision_queue_cap(-1, 4) == (size_t) DECISION_QUEUE_CAP_PER_SLOT * 4);
    GGML_ASSERT(decision_queue_cap(-1, 1) == (size_t) DECISION_QUEUE_CAP_PER_SLOT);
    // a server with no slot still derives a cap of one slot's worth
    GGML_ASSERT(decision_queue_cap(-1, 0) == (size_t) DECISION_QUEUE_CAP_PER_SLOT);

    GGML_ASSERT(decision_queue_cap(8, 4) == 8);
    GGML_ASSERT(decision_queue_cap(1, 4) == 1);

    // unlimited is 0 and only 0: it has to differ from every derived or explicit cap
    GGML_ASSERT(decision_queue_cap(0, 4) == 0);
    GGML_ASSERT(decision_queue_cap(0, 0) == 0);
    GGML_ASSERT(decision_queue_cap(0, 1) != decision_queue_cap(-1, 1));
    GGML_ASSERT(decision_queue_cap(0, 1) != decision_queue_cap(1, 1));

    // and 0 is what try_post reads as unlimited, so the two agree
    server_queue queue;
    GGML_ASSERT(queue.try_post(decision_tasks(queue, 50), SERVER_TASK_TYPE_DECISION, decision_queue_cap(0, 1), false));
    GGML_ASSERT(queue.queued_count(SERVER_TASK_TYPE_DECISION) == 50);
}

// the prompt budget resolves its flag the way the queue cap resolves its own: a positive budget is
// itself, a negative one derives from the slots and the slot context, and 0 is unlimited, which
// build_tasks reads as no bound. The derivation saturates rather than wrapping
static void test_decision_prompt_budget_is_a_bound() {
    GGML_ASSERT(decision_prompt_budget(-1, 4, 512) == (size_t) DECISION_QUEUE_CAP_PER_SLOT * 4 * 512);
    GGML_ASSERT(decision_prompt_budget(-1, 1, 512) == (size_t) DECISION_QUEUE_CAP_PER_SLOT * 512);
    // a server with no slot still derives one slot's worth
    GGML_ASSERT(decision_prompt_budget(-1, 0, 512) == (size_t) DECISION_QUEUE_CAP_PER_SLOT * 512);

    GGML_ASSERT(decision_prompt_budget(8, 4, 512) == 8);
    GGML_ASSERT(decision_prompt_budget(1, 4, 512) == 1);

    // unlimited is 0 and only 0: it has to differ from every derived or explicit budget
    GGML_ASSERT(decision_prompt_budget(0, 4, 512) == 0);
    GGML_ASSERT(decision_prompt_budget(0, 0, 0) == 0);
    GGML_ASSERT(decision_prompt_budget(0, 1, 512) != decision_prompt_budget(-1, 1, 512));
    GGML_ASSERT(decision_prompt_budget(0, 1, 512) != decision_prompt_budget(1, 1, 512));

    // a slot context of zero must not derive 0, which build_tasks would read as unlimited: the
    // smallest budget is one slot of one context, the same way no slot still derives one slot's cap
    GGML_ASSERT(decision_prompt_budget(-1, 4, 0) != 0);
    GGML_ASSERT(decision_prompt_budget(-1, 4, 0) == (size_t) DECISION_QUEUE_CAP_PER_SLOT * 4);

    // the derivation saturates rather than wrapping: a wrapped budget would refuse every request
    const int largest = std::numeric_limits<int>::max();
    GGML_ASSERT(decision_prompt_budget(-1, largest, largest) == std::numeric_limits<size_t>::max());

    // an explicit cap is never saturated, whatever the other factors are
    GGML_ASSERT(decision_prompt_budget(largest, largest, largest) == (size_t) largest);
}

int main() {
    test_error_codes_are_http_statuses();
    test_error_types_are_distinct();
    test_client_visible_codes();
    test_format_error_response();
    test_invalid_request_answers_422();
    test_status_error_carries_its_kind();
    test_request_too_large_is_its_own_kind();
    test_confidence_reproduces_the_distribution();
    test_confidence_is_reported_only_where_it_is_defined();
    test_choice_confidence();
    test_score_confidence();
    test_confidence_measures_differ();
    test_confidence_equivalence();
    test_confidence_monotonic_in_p_max();
    test_variant_and_output_matrix();
    test_template_input_lev();
    test_template_input_kev();
    test_kev_text_golden();
    test_template_input_laya();
    test_head_clip_reports_kept_below_original();
    test_head_clip_is_silent_in_budget();
    test_template_input_nimble();
    test_format_answer_lev();
    test_questions_are_refused();
    test_questions_respect_the_option_limit();
    test_questions_are_accepted();
    test_noul_option_order();
    test_choice_option_order();
    test_instructions_domain();
    test_instructions_reach_the_template();
    test_state_is_required();
    test_state_images_are_taken_out();
    test_images_over_the_limit_refuse_with_the_stored_count();
    test_the_image_limit_spans_both_channels();
    test_state_rejects_malformed_data_url();
    test_grouping_counts_the_prefix_once();
    test_budget_sees_pre_group_billing_sees_post_group();
    test_grouping_empty_prompt_shares_nothing();
    test_saturating_arithmetic();
    test_usage_never_negative();
    test_grouping_preserves_order();
    test_grouping_respects_n_slots();
    test_grouping_needs_a_token_of_its_own();
    test_grouping_degenerate_cases();
    test_shared_prompt_by_model_type();
    test_averaging_cancels_the_first_listed_preference();
    test_averaging_sums_to_one();
    test_averaging_uniform_scores_give_a_uniform_distribution();
    test_averaging_honours_the_temperature();
    test_averaging_rejects_a_mismatched_result();
    test_joint_scores_are_sliced_in_order();
    test_queue_depth_counts_decision_tasks();
    test_try_post_gates_on_depth();
    test_try_post_reports_depth_at_refusal();
    test_queued_count_stops_at_cap();
    test_try_post_mixed_types();
    test_post_tasks_registers_ids();
    test_try_post_tasks_cleans_up_on_refusal();
    test_decision_queue_cap_is_a_bound();
    test_decision_prompt_budget_is_a_bound();
    test_task_budget_is_a_bound();
    test_task_count_is_predicted_without_rendering();
    test_over_count_request_is_refused_before_render();
    printf("test-server-decision: OK\n");
    return 0;
}
