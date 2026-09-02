#include "speculative-dflash-controller.h"

#include <cstdio>

static int require(bool condition, const char * label) {
    if (condition) {
        return 0;
    }
    std::fprintf(stderr, "FAILED: %s\n", label);
    return 1;
}

int main() {
    common_speculative_dflash_controller_config config;
    config.mode = common_speculative_dflash_controller_mode::BATCH;

    const auto d1 = common_speculative_dflash_controller_select(config, 1, 4);
    const auto d2 = common_speculative_dflash_controller_select(config, 2, 4);
    const auto d3 = common_speculative_dflash_controller_select(config, 3, 4);
    const auto d4 = common_speculative_dflash_controller_select(config, 4, 4);
    const auto d8 = common_speculative_dflash_controller_select(config, 8, 4);

    int failures = 0;
    failures += require(d1.depth == 4 && !d1.limited_by_batch, "active one keeps width four");
    failures += require(d2.depth == 2 && d2.limited_by_batch, "active two selects width two");
    failures += require(d3.depth == 4 && !d3.limited_by_batch, "active three keeps width four");
    failures += require(d4.depth == 4 && !d4.limited_by_batch, "active four keeps width four");
    failures += require(d8.depth == 4 && !d8.limited_by_batch, "unqualified batches keep fixed width");

    failures += require(common_speculative_dflash_controller_pre_draft_cap(
                    config, 2, 4, false) == 2,
            "batch mode applies the active-two cap");
    failures += require(common_speculative_dflash_controller_pre_draft_cap(
                    config, 2, 4, true) == 4,
            "an explicit request bypasses the cap");
    failures += require(common_speculative_dflash_controller_pre_draft_cap(
                    config, 2, 1, false) == 1,
            "the controller never widens a request");

    config.mode = common_speculative_dflash_controller_mode::TRACE;
    failures += require(common_speculative_dflash_controller_pre_draft_cap(
                    config, 2, 4, false) == 4,
            "trace mode does not change the requested width");
    config.mode = common_speculative_dflash_controller_mode::OFF;
    failures += require(common_speculative_dflash_controller_pre_draft_cap(
                    config, 2, 4, false) == 4,
            "off mode retains fixed width");

    return failures == 0 ? 0 : 1;
}
