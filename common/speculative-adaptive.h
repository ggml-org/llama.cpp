#pragma once

#include <algorithm>
#include <cstring>

// Keep the opt-out parser small and testable without exposing a public API.
inline bool common_speculative_adaptive_env_enabled(const char * value) {
    return value == nullptr ||
           (std::strcmp(value, "0") != 0 &&
            std::strcmp(value, "off") != 0 &&
            std::strcmp(value, "false") != 0 &&
            std::strcmp(value, "no") != 0);
}

// Internal acceptance-feedback controller used by the sidecar n-gram stack.
//
// The configured MTP draft width is the safe floor/start point and the selected
// n-gram implementation's own configured width is the ceiling.  The controller
// deliberately climbs a conservative staircase instead of jumping from a small
// MTP batch directly to a wide n-gram batch.  A partial result immediately steps
// back one rung, which prevents repeated oversized verification bursts after the
// first poor round.  A full result shorter than the current rung is not evidence
// that the wider rung is safe and therefore does not promote the controller.
struct common_speculative_adaptive {
    int n_floor   = 0;
    int n_ceiling = 0;
    int n_cur     = 0;
    int n_climb   = 0;

    void reset(int floor, int ceiling) {
        n_floor   = std::max(1, floor);
        n_ceiling = std::max(n_floor, ceiling);
        n_cur     = std::min(n_floor, n_ceiling);
        n_climb   = 0;
    }

    // Require more evidence at the known-safe floor and at the first wider
    // rung, where the target verification cost can change sharply.
    static int climb_threshold(int depth) {
        switch (depth) {
            case 1: return 2;
            case 2: return 4;
            case 3: return 10;
            case 4: return 6;
            case 5: return 4;
            case 6: return 3;
            default: return 2;
        }
    }

    // The widening staircase is 3, 4, 6, 8, 12, 16, 24, 32, 48 for the
    // command-line values used by the validated launcher.  It generalizes to
    // other configured floor/ceiling pairs without adding another setting.
    static int next_depth(int depth, int ceiling) {
        int next = depth + 1;
        if (depth >= 4 && depth < 8) {
            next = depth + 2;
        } else if (depth >= 8 && depth < 16) {
            next = depth + 4;
        } else if (depth >= 16 && depth < 32) {
            next = depth + 8;
        } else if (depth >= 32) {
            next = ceiling;
        }
        return std::min(next, ceiling);
    }

    static int previous_depth(int depth, int floor) {
        int step = 1;
        if (depth > 4 && depth <= 8) {
            step = 2;
        } else if (depth > 8 && depth <= 16) {
            step = 4;
        } else if (depth > 16 && depth <= 32) {
            step = 8;
        } else if (depth > 32) {
            step = 16;
        }
        return std::max(floor, depth - step);
    }

    // Feed the number actually offered to the target and the accepted prefix.
    void update(int n_draft, int n_accepted) {
        if (n_draft <= 0 || n_cur <= 0 || n_ceiling <= 0) {
            return;
        }

        n_accepted = std::clamp(n_accepted, 0, n_draft);

        // A short match or an end-of-context clamp did not exercise the current
        // width.  Do not use it to promote to a wider target batch.
        if (n_draft < n_cur) {
            n_climb = 0;
            return;
        }

        if (n_accepted == n_draft) {
            if (n_cur >= n_ceiling) {
                n_climb = 0;
                return;
            }
            if (++n_climb >= climb_threshold(n_cur)) {
                n_cur = next_depth(n_cur, n_ceiling);
                n_climb = 0;
            }
            return;
        }

        // A partial wide round is the strongest cheap signal that the current
        // rung is too aggressive.  Step back immediately; the next promotion
        // still requires the full hysteresis threshold.
        n_climb = 0;
        if (n_cur > n_floor) {
            n_cur = previous_depth(n_cur, n_floor);
        }
    }
};
