// SPDX-License-Identifier: MIT
#pragma once

#include <cmath>

namespace spec_sidecar_dflash {

// Normalize the complete proposal row before sampling it. Returning the same
// normalized row is part of the stochastic sidecar ABI: the target verifier
// must receive the exact q distribution used to select the draft token.
inline int normalize_and_select(float * probs, int count, float sum, double draw) {
    if (probs == nullptr || count <= 0 || !(sum > 0.0f) || !std::isfinite(sum) ||
            !(draw >= 0.0 && draw < 1.0)) {
        return -1;
    }

    for (int i = 0; i < count; ++i) {
        probs[i] /= sum;
    }

    int selected = -1;
    double cumulative = 0.0;
    for (int i = 0; i < count; ++i) {
        if (!(probs[i] > 0.0f) || !std::isfinite(probs[i])) {
            continue;
        }
        selected = i;
        cumulative += probs[i];
        if (draw < cumulative) {
            break;
        }
    }
    return selected;
}

} // namespace spec_sidecar_dflash
