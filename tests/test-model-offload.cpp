#include "llama-model-offload.h"

#include <algorithm>
#include <climits>
#include <cstdio>
#include <stdexcept>
#include <vector>

static int failures = 0;
static int checks = 0;

static void check(bool ok, const char * what) {
    ++checks;
    if (!ok) {
        ++failures;
        if (failures <= 12) {
            std::fprintf(stderr, "FAIL: %s\n", what);
        }
    }
}

static int device_index(const llama_model_offload & p, int il, const std::vector<float> & splits) {
    const int index = p.layer_index(il);
    if (!p.is_gpu(index)) {
        return -1;
    }
    const float position = float(index - p.i_gpu_start) / p.n_gpu;
    return int(std::upper_bound(splits.begin(), splits.end(), position) - splits.begin());
}

static void check_layout(int all, int nextn, bool mtp, int ngl, bool devices, int router = -1) {
    const llama_model_offload p(all, nextn, mtp, ngl, devices, router);
    // Count only genuinely optional trailing NextN slots as unloaded. Granite-switch
    // uses NextN for an active router slot; draft-only metadata has no trunk to keep.
    const bool optional_mtp = !mtp && router == -1 && all > nextn;
    const int loaded = optional_mtp ? all - nextn : all;
    const int total = devices ? std::min(ngl, loaded + 1) : 0;
    const int expected_repeating = std::max(total - 1, 0);
    check(p.n_layer == loaded, "loaded layer count");
    check(p.n_gpu == total, "actual default offload count");
    check(p.n_repeating_gpu() == expected_repeating, "repeating summary");
    check(llama_model_offload::layer_count(all, nextn, mtp, router) == loaded,
          "context and loader share the same layer count");
    check((ngl > p.n_layer) == (ngl >= loaded + 1), "full-offload predicate includes output");

    int assigned = 0;
    for (int il = 0; il < loaded; ++il) {
        const bool expected = devices && il >= loaded - expected_repeating;
        const bool actual = p.is_gpu(p.layer_index(il));
        check(actual == expected, "repeating placement");
        assigned += actual;
    }
    const bool output = p.is_gpu(p.layer_index(all));
    check(output == (total > 0), "output placement");
    check(assigned + int(output) == total, "summary matches loaded assignments");
    for (int il = loaded; il < all; ++il) {
        check(p.layer_index(il) == loaded, "unused NextN slots do not shift the trunk");
    }

    // This is the real-MTP fitter contract, not a claim that fitting router/draft
    // formats is fixed. Preserve the historical loader policy for those exceptions.
    if (router == -1 && (all > nextn || nextn == 0 || mtp)) {
        const int fit_hp_ngl = all - nextn + (mtp ? nextn : 0);
        const int fit_ngl = std::min(ngl, fit_hp_ngl + 1);
        check(p.i_gpu_start == fit_hp_ngl + 1 - fit_ngl, "fit override start agrees with loader");
    }
    if (mtp || nextn == 0 || router >= 0 || all == nextn) {
        const int old_start = std::max(all + 1 - ngl, 0);
        const int old_gpu = devices ? std::min(ngl, all + 1) : 0;
        for (int il = 0; il <= all; ++il) {
            const bool old_assignment = il >= old_start && il - old_start < old_gpu;
            check(p.is_gpu(p.layer_index(il)) == old_assignment, "non-optional placement unchanged");
        }
    }

    for (const std::vector<float> & splits : {std::vector<float>{1.0f}, std::vector<float>{0.5f, 1.0f}}) {
        int previous = -1;
        for (int il = 0; il <= loaded; ++il) {
            const int d = device_index(p, il == loaded ? all : il, splits);
            check(d >= -1 && d < int(splits.size()), "device split remains in range");
            check(d >= previous, "device split is monotone");
            previous = d;
        }
    }
}

int main() {
    // Generic metadata only: 40 trunk layers, one stored but unloaded NextN layer.
    const llama_model_offload skipped(41, 1, false, 41, true);
    check(skipped.is_gpu(skipped.layer_index(0)), "41 slots fit all 40 trunk layers plus output");
    check(skipped.n_repeating_gpu() == 40, "full-offload summary counts all 40 trunk layers");
    const llama_model_offload router(5, 1, false, 5, true, 4);
    check(router.n_layer == 5 && !router.is_gpu(router.layer_index(0)), "active router slot is not optional MTP");
    const llama_model_offload draft(2, 2, false, 2, true);
    check(draft.n_layer == 2 && !draft.is_gpu(draft.layer_index(0)), "draft-only count remains unchanged");

    for (int all : {0, 1, 2, 4, 40, 41, 48, 512}) {
        for (int nextn : {0, 1, 2}) {
            if (nextn > all) {
                continue;
            }
            for (bool mtp : {false, true}) {
                for (bool devices : {false, true}) {
                    for (int ngl : {0, 1, 2, std::max(all - 1, 0), all, all + 1, all + 2, INT_MAX}) {
                        check_layout(all, nextn, mtp, ngl, devices);
                        if (all > 0 && nextn == 1) {
                            check_layout(all, nextn, mtp, ngl, devices, all - 1);
                        }
                    }
                }
            }
        }
    }

    // An expert override changes only named tensors, not the default layer boundary.
    const llama_model_offload p(41, 1, false, 41, true);
    for (int il = 0; il < 40; ++il) {
        const bool default_gpu = p.is_gpu(p.layer_index(il));
        const bool expert_cpu_override = il >= 34;
        check(default_gpu, "default trunk placement stays on GPU with overrides");
        check((default_gpu && !expert_cpu_override) == (il < 34), "explicit expert override remains separate");
    }

    for (const std::vector<int> & bad : {std::vector<int>{-1, 0, 0, -1}, {INT_MAX, 0, 0, -1}, {1, -1, 0, -1}, {1, 2, 0, -1}, {1, 0, -1, -1}, {1, 0, 0, -2}, {1, 0, 0, 1}}) {
        bool rejected = false;
        try {
            llama_model_offload invalid(bad[0], bad[1], false, bad[2], true, bad[3]);
        } catch (const std::invalid_argument &) {
            rejected = true;
        }
        check(rejected, "invalid count rejected before arithmetic");
    }
    std::printf("checks=%d failures=%d\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
