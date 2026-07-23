#include "server-task.h"
#include "server-schema.h"
#include "common.h"

#include <cassert>
#include <stdexcept>

static repetition_detection_params params(int min_size, int max_size, int min_count) {
    return { max_size, min_size, min_count };
}

int main() {
    assert(server_check_sequence_repetition({1, 2, 3, 1, 2, 3, 1, 2, 3}, params(2, 3, 3)));
    assert(!server_check_sequence_repetition({1, 2, 3, 1, 2, 3}, params(3, 3, 3)));
    assert(!server_check_sequence_repetition({1, 2, 3, 1, 2, 3, 1, 2, 4}, params(3, 3, 3)));
    assert(!server_check_sequence_repetition({1, 2, 3, 4, 5, 6}, params(1, 3, 2)));
    assert(server_check_sequence_repetition({1, 2, 3, 4, 5, 6, 5, 6, 5, 6}, params(2, 2, 3)));
    assert(!server_check_sequence_repetition({1, 1, 1, 1}, params(1, 0, 4)));

    common_params base;
    const auto parsed = server_schema::eval_llama_cmpl_schema(nullptr, base, 1024, {}, {
        {"repetition_detection", {
            {"min_pattern_size", 3},
            {"max_pattern_size", 64},
            {"min_count", 5},
        }},
    });
    assert(parsed.repetition_detection.min_pattern_size == 3);
    assert(parsed.repetition_detection.max_pattern_size == 64);
    assert(parsed.repetition_detection.min_count == 5);

    bool invalid_rejected = false;
    try {
        server_schema::eval_llama_cmpl_schema(nullptr, base, 1024, {}, {
            {"repetition_detection", {
                {"min_pattern_size", 4},
                {"max_pattern_size", 3},
                {"min_count", 2},
            }},
        });
    } catch (const std::invalid_argument &) {
        invalid_rejected = true;
    }
    assert(invalid_rejected);

    server_task_result_cmpl_final response;
    response.stop = STOP_TYPE_REPETITION;
    response.oaicompat_model = "test";
    response.oaicompat_cmpl_id = "test-id";
    assert(response.to_json_oaicompat_chat().at("choices").at(0).at("finish_reason") == "repetition");
    return 0;
}
