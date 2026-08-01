#include "tessera-lbfgs.h"

#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>

// ---------------------------------------------------------------------------
// L-BFGS shared state.
//
// The curvature ring buffer (S, Y, rho) is held in an opaque struct so the
// monolithic ts_lbfgs_minimize loop and the single-step ts_lbfgs_step API
// (used by CHAMP-Q, which interleaves a Sinkhorn projection between steps)
// share one implementation of the two-loop recursion and the curvature push.
// Mirrors tools/tessera/_champq_lbfgs.py.
// ---------------------------------------------------------------------------

struct ts_lbfgs_state {
    int64_t n;
    int64_t history;
    int64_t count;  // number of stored pairs (<= history)
    int64_t head;   // ring buffer write position

    std::vector<float> S;      // history * n
    std::vector<float> Y;      // history * n
    std::vector<float> rho;    // history
    std::vector<float> alpha;  // history (scratch)
    std::vector<float> q;      // n (scratch)
    std::vector<float> r;      // n (scratch)

    ts_lbfgs_state(int64_t n_, int64_t history_)
        : n(n_), history(history_), count(0), head(0),
          S(history_ * n_, 0.0f),
          Y(history_ * n_, 0.0f),
          rho(history_, 0.0f),
          alpha(history_, 0.0f),
          q(n_, 0.0f),
          r(n_, 0.0f) {}
};

static float ts_dot(const float * a, const float * b, int64_t n) {
    float sum = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

static float ts_norm(const float * a, int64_t n) {
    return std::sqrt(ts_dot(a, a, n));
}

// two-loop recursion: out = H_k * grad
static void ts_lbfgs_two_loop(ts_lbfgs_state & st, const float * grad, float * out) {
    const int64_t n = st.n;
    const int64_t m = st.count;

    std::memcpy(st.q.data(), grad, n * sizeof(float));

    // first loop (most recent to oldest)
    for (int64_t k = m - 1; k >= 0; k--) {
        int64_t idx = (st.head - m + k + st.history) % st.history;
        const float * s_k = st.S.data() + idx * n;
        const float * y_k = st.Y.data() + idx * n;
        st.alpha[k] = st.rho[idx] * ts_dot(s_k, st.q.data(), n);
        for (int64_t i = 0; i < n; i++) {
            st.q[i] -= st.alpha[k] * y_k[i];
        }
    }

    // initial Hessian scale
    float gamma = 1.0f;
    if (m > 0) {
        int64_t last = (st.head - 1 + st.history) % st.history;
        const float * s_last = st.S.data() + last * n;
        const float * y_last = st.Y.data() + last * n;
        float sy = ts_dot(s_last, y_last, n);
        float yy = ts_dot(y_last, y_last, n);
        gamma = sy / (yy + 1e-12f);
    }

    for (int64_t i = 0; i < n; i++) {
        st.r[i] = gamma * st.q[i];
    }

    // second loop (oldest to most recent)
    for (int64_t k = 0; k < m; k++) {
        int64_t idx = (st.head - m + k + st.history) % st.history;
        const float * s_k = st.S.data() + idx * n;
        const float * y_k = st.Y.data() + idx * n;
        float beta = st.rho[idx] * ts_dot(y_k, st.r.data(), n);
        float coeff = st.alpha[k] - beta;
        for (int64_t i = 0; i < n; i++) {
            st.r[i] += coeff * s_k[i];
        }
    }

    std::memcpy(out, st.r.data(), n * sizeof(float));
}

static void ts_lbfgs_push(ts_lbfgs_state & st, const float * s, const float * y,
                          float curvature_eps) {
    float sy = ts_dot(s, y, st.n);
    if (sy <= curvature_eps) {
        // Curvature condition violated. Drop the whole buffer (stale pairs
        // are worse than none); matches the Python reset-on-violate behavior.
        st.count = 0;
        st.head  = 0;
        return;
    }
    int64_t idx = st.head;
    std::memcpy(st.S.data() + idx * st.n, s, st.n * sizeof(float));
    std::memcpy(st.Y.data() + idx * st.n, y, st.n * sizeof(float));
    st.rho[idx] = 1.0f / sy;
    st.head = (st.head + 1) % st.history;
    if (st.count < st.history) {
        st.count++;
    }
}

// ---------------------------------------------------------------------------
// Monolithic minimize (keeps the original API for test_lbfgs.cpp / search).
// ---------------------------------------------------------------------------

float ts_lbfgs_minimize(float * x, int64_t n,
                        ts_lbfgs_eval_fn eval, void * eval_ctx,
                        const ts_lbfgs_params * params) {
    ts_lbfgs_params defaults = { 100, 10, 1e-5f, 1.0f, false };
    if (!params) {
        params = &defaults;
    }

    const int64_t max_iter  = params->max_iter;
    const float     tol       = params->tol;
    const float     lr_init   = params->lr_init;

    ts_lbfgs_state st(n, params->history);

    std::vector<float> grad(n);
    std::vector<float> grad_new(n);
    std::vector<float> direction(n);
    std::vector<float> x_new(n);
    std::vector<float> s_vec(n);
    std::vector<float> y_vec(n);

    float loss = eval(x, grad.data(), n, eval_ctx);

    for (int64_t iter = 0; iter < max_iter; iter++) {
        float gnorm = ts_norm(grad.data(), n);
        if (gnorm < tol) {
            break;
        }

        if (params->verbose) {
            fprintf(stderr, "  lbfgs iter %4lld/%lld  loss=%.6e  |g|=%.3e\n",
                    (long long)iter, (long long)max_iter, loss, gnorm);
        }

        // search direction d = -H * grad
        ts_lbfgs_two_loop(st, grad.data(), direction.data());
        for (int64_t i = 0; i < n; i++) {
            direction[i] = -direction[i];
        }

        float dg = ts_dot(grad.data(), direction.data(), n);
        if (dg >= 0.0f) {
            // not a descent direction; fall back to steepest descent
            for (int64_t i = 0; i < n; i++) {
                direction[i] = -grad[i];
            }
            dg = ts_dot(grad.data(), direction.data(), n);
        }

        // backtracking Armijo line search
        float alpha = lr_init;
        bool accepted = false;
        float loss_new = loss;
        const float c1        = 1e-4f;
        const int   max_ls    = 25;
        const float ls_shrink = 0.5f;

        for (int ls = 0; ls < max_ls; ls++) {
            for (int64_t i = 0; i < n; i++) {
                x_new[i] = x[i] + alpha * direction[i];
            }
            float cand_loss = eval(x_new.data(), grad_new.data(), n, eval_ctx);
            if (cand_loss <= loss + c1 * alpha * dg) {
                loss_new = cand_loss;
                accepted = true;
                break;
            }
            alpha *= ls_shrink;
        }

        if (!accepted) {
            // take the smallest-alpha candidate as a courtesy
            for (int64_t i = 0; i < n; i++) {
                x_new[i] = x[i] + alpha * direction[i];
            }
            loss_new = eval(x_new.data(), grad_new.data(), n, eval_ctx);
        }

        // curvature pair
        for (int64_t i = 0; i < n; i++) {
            s_vec[i] = x_new[i] - x[i];
            y_vec[i] = grad_new[i] - grad[i];
        }
        ts_lbfgs_push(st, s_vec.data(), y_vec.data(), 1e-12f);

        std::memcpy(x, x_new.data(), n * sizeof(float));
        std::memcpy(grad.data(), grad_new.data(), n * sizeof(float));
        loss = loss_new;

        if (!accepted) {
            break;
        }
    }

    return loss;
}

float ts_pgd_minimize(float * x, int64_t n,
                      ts_lbfgs_eval_fn eval, void * eval_ctx,
                      ts_project_fn project, void * project_ctx,
                      int64_t max_iter, float lr, float tol) {
    std::vector<float> grad(n);

    float loss = eval(x, grad.data(), n, eval_ctx);

    for (int64_t iter = 0; iter < max_iter; iter++) {
        float gnorm = ts_norm(grad.data(), n);
        if (gnorm < tol) {
            break;
        }

        for (int64_t i = 0; i < n; i++) {
            x[i] -= lr * grad[i];
        }

        if (project) {
            project(x, n, project_ctx);
        }

        loss = eval(x, grad.data(), n, eval_ctx);
    }

    return loss;
}

// ---------------------------------------------------------------------------
// Single-step L-BFGS (for CHAMP-Q's project-between-steps driver).
// ---------------------------------------------------------------------------

struct ts_lbfgs_state * ts_lbfgs_state_create(int64_t n, int64_t history) {
    if (n <= 0 || history < 1) {
        return nullptr;
    }
    return new ts_lbfgs_state(n, history);
}

void ts_lbfgs_state_destroy(struct ts_lbfgs_state * st) {
    delete st;
}

void ts_lbfgs_state_reset(struct ts_lbfgs_state * st) {
    if (!st) {
        return;
    }
    st->count = 0;
    st->head  = 0;
}

float ts_lbfgs_step(struct ts_lbfgs_state * st,
                    const float * x, float loss, const float * grad,
                    ts_lbfgs_eval_fn eval, void * eval_ctx,
                    const struct ts_lbfgs_step_params * params,
                    float * x_new, float * grad_new, int * done) {
    if (!st) {
        // No state: degenerate to a no-op. Signal done so the caller stops.
        if (done) *done = 1;
        return loss;
    }

    ts_lbfgs_step_params defaults = { 1e-4f, 25, 0.5f, 1e-12f };
    if (!params) {
        params = &defaults;
    }

    const int64_t n = st->n;
    const float    c1        = params->c1;
    const int      max_ls    = params->max_ls;
    const float    ls_shrink = params->ls_shrink;

    std::vector<float> direction(n);

    // d = -H * grad
    ts_lbfgs_two_loop(*st, grad, direction.data());
    for (int64_t i = 0; i < n; i++) {
        direction[i] = -direction[i];
    }

    float dg = ts_dot(grad, direction.data(), n);
    if (dg >= 0.0f) {
        // not a descent direction (e.g. right after a reset); fall back to
        // the raw negative gradient so the line search still has a direction.
        for (int64_t i = 0; i < n; i++) {
            direction[i] = -grad[i];
        }
        dg = ts_dot(grad, direction.data(), n);
    }

    // backtracking Armijo starting at alpha = 1.0 (the L-BFGS ideal when H
    // approximates the inverse Hessian well). Mirrors Python's LBFGS.step.
    float alpha = 1.0f;
    float loss_new = loss;
    int   accepted = 0;
    for (int ls = 0; ls < max_ls; ls++) {
        for (int64_t i = 0; i < n; i++) {
            x_new[i] = x[i] + alpha * direction[i];
        }
        float cand_loss = eval(x_new, grad_new, n, eval_ctx);
        if (cand_loss <= loss + c1 * alpha * dg) {
            loss_new = cand_loss;
            accepted = 1;
            break;
        }
        alpha *= ls_shrink;
    }
    if (!accepted) {
        // take the smallest-alpha candidate as a courtesy; flag done.
        for (int64_t i = 0; i < n; i++) {
            x_new[i] = x[i] + alpha * direction[i];
        }
        loss_new = eval(x_new, grad_new, n, eval_ctx);
    }

    // curvature pair: s = x_new - x, y = grad_new - grad
    std::vector<float> s_vec(n), y_vec(n);
    for (int64_t i = 0; i < n; i++) {
        s_vec[i] = x_new[i] - x[i];
        y_vec[i] = grad_new[i] - grad[i];
    }
    ts_lbfgs_push(*st, s_vec.data(), y_vec.data(), params->curvature_eps);

    if (done) {
        *done = accepted ? 0 : 1;
    }
    return loss_new;
}
