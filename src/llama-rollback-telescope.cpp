// Copyright (c) 2026 Song Wei
// SPDX-License-Identifier: MIT
//
// CPU reference implementation of the forward-telescoping rollback engine.
// See llama-rollback-telescope.h for the design contract.

#include "llama-rollback-telescope.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace llama {

static constexpr uint32_t META_INVALID_POS = 0xFFFFFFFFu;

telescope_rollback::telescope_rollback(const telescope_config & cfg) : cfg_(cfg) {
    const uint32_t ring = std::max(cfg_.ring_cap, 1u);
    const uint32_t sl   = std::max(cfg_.slots, 1u);
    const size_t per_token =
        (size_t) cfg_.L_meta * cfg_.H * (2 * cfg_.d + 2);
    meta_ring_.assign((size_t) ring * per_token, 0.0f);
    meta_pos_.assign(ring, META_INVALID_POS);
    anchor_ring_.assign((size_t) sl * cfg_.L * cfg_.H * cfg_.d * cfg_.d, 0.0f);
    anchor_pos_.assign(sl, META_INVALID_POS);
    anchor_valid_.assign(sl, false);
}

void telescope_rollback::capture(uint32_t pos, const float * k, const float * v,
                                 const float * beta, const float * gate) {
    const uint32_t slot = pos % cfg_.ring_cap;
    const size_t per_token = (size_t) cfg_.L_meta * cfg_.H * (2 * cfg_.d + 2);
    float * dst = meta_ring_.data() + (size_t) slot * per_token;
    // layout per (layer, head): k[d] v[d] beta gamma
    for (uint32_t l = 0; l < cfg_.L_meta; ++l) {
        const float * kl = k + (size_t) l * cfg_.H * cfg_.d;
        const float * vl = v + (size_t) l * cfg_.H * cfg_.d;
        const float * bl = beta + (size_t) l * cfg_.H;
        const float * gl = gate + (size_t) l * cfg_.H;
        for (uint32_t h = 0; h < cfg_.H; ++h) {
            float * dk = dst + (size_t) l * cfg_.H * (2 * cfg_.d + 2) + h * (2 * cfg_.d + 2);
            memcpy(dk, kl + (size_t) h * cfg_.d, cfg_.d * sizeof(float));
            memcpy(dk + cfg_.d, vl + (size_t) h * cfg_.d, cfg_.d * sizeof(float));
            dk[2 * cfg_.d]     = bl[h];
            dk[2 * cfg_.d + 1] = gl[h];
        }
    }
    meta_pos_[slot] = pos;
}

void telescope_rollback::place_anchor(uint32_t pos, const float * S) {
    // find the slot for pos (pos % (window*slots) would be the natural index;
    // we keep it simple: rotate over slots by absolute position)
    const uint32_t slot = (pos / std::max(cfg_.window, 1u)) % cfg_.slots;
    const size_t n = (size_t) cfg_.L * cfg_.H * cfg_.d * cfg_.d;
    memcpy(anchor_ring_.data() + (size_t) slot * n, S, n * sizeof(float));
    anchor_pos_[slot]   = pos;
    anchor_valid_[slot] = true;
}

uint32_t telescope_rollback::locate_anchor(uint32_t p0) const {
    uint32_t best = META_INVALID_POS;
    for (uint32_t s = 0; s < cfg_.slots; ++s) {
        if (anchor_valid_[s] && anchor_pos_[s] <= p0) {
            if (best == META_INVALID_POS || anchor_pos_[s] > best) {
                best = anchor_pos_[s];
            }
        }
    }
    return best;
}

bool telescope_rollback::resident(uint32_t a, uint32_t p0) const {
    if (a == META_INVALID_POS || p0 <= a) return false;
    const uint32_t dist = p0 - a;
    if (dist > cfg_.window) return false;
    // every position in (a, p0] must be present in the metadata ring
    for (uint32_t pos = a + 1; pos <= p0; ++pos) {
        const uint32_t slot = pos % cfg_.ring_cap;
        if (meta_pos_[slot] != pos) return false;
    }
    return true;
}

void telescope_rollback::fwd_telescope_execute(uint32_t a, uint32_t p0, float * S_out) const {
    // load anchor state
    const uint32_t aslot = (a / std::max(cfg_.window, 1u)) % cfg_.slots;
    const size_t n_state = (size_t) cfg_.L * cfg_.H * cfg_.d * cfg_.d;
    const float * Sa = anchor_ring_.data() + (size_t) aslot * n_state;
    memcpy(S_out, Sa, n_state * sizeof(float));

    const uint32_t dist = (p0 > a + 1) ? (p0 - a - 1) : 0;  // replay (a, p0): S_after_{p0-1} per seq_rm semantics
    const size_t per_token = (size_t) cfg_.L_meta * cfg_.H * (2 * cfg_.d + 2);

    // replay (a, p0]: S <- gamma*S - beta*k*(k^T S) + beta*k*v^T
    // layers are independent -> parallelize over layers
    #pragma omp parallel for collapse(1) schedule(static)
    for (int64_t li = 0; li < (int64_t) cfg_.L_meta; ++li) {
        const uint32_t l = (uint32_t) li;
        for (uint32_t h = 0; h < cfg_.H; ++h) {
            float * S_lh = S_out + ((size_t) l * cfg_.H + h) * cfg_.d * cfg_.d;
            for (uint32_t t = 0; t < dist; ++t) {
                const uint32_t slot = (a + 1 + t) % cfg_.ring_cap;
                const float * m = meta_ring_.data() + (size_t) slot * per_token
                                + (size_t) l * cfg_.H * (2 * cfg_.d + 2) + h * (2 * cfg_.d + 2);
                const float * k = m;
                const float * v = m + cfg_.d;
                const float  b  = m[2 * cfg_.d];
                const float  g  = m[2 * cfg_.d + 1]; // gamma multiplier (already exp(gate logit))
                // w = k^T S  (d x d matrix-vector per head)
                for (uint32_t c = 0; c < cfg_.d; ++c) {
                    double w = 0.0;
                    const float * kk = k;
                    const float * Sc = S_lh + c;
                    for (uint32_t r = 0; r < cfg_.d; ++r) {
                        w += (double) kk[r] * (double) Sc[(size_t) r * cfg_.d];
                    }
                    // S[.,c] <- g*S[.,c] - b*k*w + b*k*v[c]
                    const double bv = (double) b * (double) v[c];
                    const double bw = (double) b * w;
                    for (uint32_t r = 0; r < cfg_.d; ++r) {
                        const double kr = (double) kk[r];
                        S_lh[(size_t) r * cfg_.d + c] = (float) ((double) g * (double) S_lh[(size_t) r * cfg_.d + c]
                            - bw * kr + bv * kr);
                    }
                }
            }
        }
    }
}

rollback_result telescope_rollback::rollback(uint32_t p0, float * S_out) {
    rollback_result res;
    const uint32_t a = locate_anchor(p0);
    if (a == META_INVALID_POS || !resident(a, p0)) {
        // explicit fallback: caller performs full recompute
        ++fallback_count_;
        res.fallback_count = fallback_count_;
        return res;
    }
    fwd_telescope_execute(a, p0, S_out);
    res.ok     = true;
    res.anchor = a;
    res.fallback_count = fallback_count_;
    return res;
}

size_t telescope_rollback::resident_bytes() const {
    const size_t per_token = (size_t) cfg_.L_meta * cfg_.H * (2 * cfg_.d + 2);
    const size_t ring = meta_ring_.size() * sizeof(float);
    const size_t anchor = anchor_ring_.size() * sizeof(float);
    return ring + anchor;
}

} // namespace llama
