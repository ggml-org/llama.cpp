#include "llama-tp-output-policy.h"

#include <cstdio>
#include <cstdlib>

static void require(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "TP output policy test failure: %s\n", message);
        std::abort();
    }
}

static llama_tp_output_split_mode select(
        bool enabled, bool blocked, bool tensor, bool supported,
        bool vocabulary, bool primary) {
    return llama_tp_output_policy_select({enabled, blocked, tensor, supported, vocabulary, primary});
}

int main() {
    using mode = llama_tp_output_split_mode;

    require(select(true, false, true, true, false, true) == mode::hidden,
            "default primary head must retain hidden/full-logit sharding");
    require(select(true, false, true, true, true, true) == mode::vocabulary,
            "explicit primary vocabulary sharding was not selected");
    require(select(true, false, true, true, true, false) == mode::hidden,
            "auxiliary MTP head must retain hidden/full-logit sharding");
    require(select(false, false, true, true, true, true) == mode::mirrored,
            "disabled output sharding must remain mirrored");
    require(select(true, true, true, true, true, true) == mode::mirrored,
            "shared external head must block output sharding");
    require(select(true, false, false, true, true, true) == mode::mirrored,
            "non-tensor mode must remain mirrored");
    require(select(true, false, true, false, true, true) == mode::mirrored,
            "unsupported architecture must remain mirrored");

    return 0;
}
