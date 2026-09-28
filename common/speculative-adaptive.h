#pragma once

#include <algorithm>

// Adaptive draft depth controller for MTP speculative decoding (draft-mtp-adaptive).
//
// Hysteresis state machine: the depth climbs one step after N consecutive
// full-accept verifies and drops one step when accumulated miss pressure
// (sum of n_draft - n_accepted per round) reaches a depth-scaled budget.
// The depth stays within [floor, cap]; at the floor no pressure accumulates.
// See climb_threshold / drop_pressure for the per-depth constants.
struct common_speculative_adaptive {
    int n_cur   = 0; // current adaptive draft depth N
    int n_climb = 0; // consecutive verifies that accepted every drafted token
    int n_drop  = 0; // accumulated drop pressure: sum of (n_draft - n_accepted)

    // consecutive full accepts needed to climb one step from depth N; low at the
    // floor and at depth, high in the middle where acceptance is marginal
    static int climb_threshold(int depth) {
        switch (depth) {
            case 1: return 2;
            case 2: return 4;
            case 3: return 10; // hardened 3->4 barrier: keeps prose/reasoning pinned
            case 4: return 6;
            case 5: return 3;
            case 6: return 2;
            default: return 2; // depth >= 7
        }
    }

    // accumulated (n_draft - n_accepted) needed to drop one step from depth N;
    // scaled by depth, with a floor so shallow depths do not collapse too fast
    static int drop_pressure(int depth) {
        return std::max(depth * 5, 20);
    }

    // reset to the floor max(1, n_min_adaptive), bounded by the ceiling n_max;
    // the controller climbs from there once acceptance feedback arrives
    void reset(int n_max, int n_min_adaptive) {
        const int cap   = std::max(1, n_max);
        const int floor = std::max(1, n_min_adaptive);

        n_cur   = std::min(floor, cap);
        n_climb = 0;
        n_drop  = 0;
    }

    // feed one verification result: n_draft is the number of tokens this
    // implementation drafted, n_accepted the number the target accepted
    void update(int n_draft, int n_accepted, int n_max, int n_min_adaptive) {
        if (n_draft <= 0) {
            return;
        }

        const int cap   = std::max(1, n_max);
        const int floor = std::max(1, n_min_adaptive);

        if (n_accepted == n_draft) {
            n_drop = 0;

            // full acceptance: reset the drop pressure, accumulate the climb streak
            if (n_cur < cap && ++n_climb >= climb_threshold(n_cur)) {
                n_cur++;
                n_climb = 0;
            }
        } else {
            n_climb = 0;

            // any miss adds (n_draft - n_accepted) to the drop pressure; drop one
            // step when the accumulated pressure reaches the depth-scaled budget
            if (n_cur > floor) {
                n_drop += n_draft - n_accepted;
                if (n_drop >= drop_pressure(n_cur)) {
                    n_cur--;
                    n_drop = 0;
                }
            }
        }
    }
};
