#pragma once

//
// tessera-lbfgs.h
//
// L-BFGS optimizer and projected gradient descent for CHAMP-Q
// permutation optimization. C-style state struct, no virtual dispatch.
//

#include <cstdint>
#include <cstddef>
#include <functional>

// Callback: compute f(x) and grad(x). grad is pre-allocated (n floats).
// Return the objective value.
typedef float (*ts_lbfgs_eval_fn)(const float * x, float * grad, int64_t n, void * ctx);

// Optional projection callback: project x onto feasible set in-place.
typedef void (*ts_project_fn)(float * x, int64_t n, void * ctx);

struct ts_lbfgs_params {
    int64_t max_iter;       // default 100
    int64_t history;        // L-BFGS memory, default 10
    float     tol;          // convergence tolerance on ||grad||, default 1e-5
    float     lr_init;      // initial step size for line search, default 1.0
    bool      verbose;      // print iteration info
};

// Run L-BFGS. x is (n,) initialized in-place; overwritten with solution.
// Returns final objective value.
float ts_lbfgs_minimize(float * x, int64_t n,
                        ts_lbfgs_eval_fn eval, void * eval_ctx,
                        const ts_lbfgs_params * params);

// Projected gradient descent (simpler alternative for permutation).
// Applies project_fn after each step. Returns final objective.
float ts_pgd_minimize(float * x, int64_t n,
                      ts_lbfgs_eval_fn eval, void * eval_ctx,
                      ts_project_fn project, void * project_ctx,
                      int64_t max_iter, float lr, float tol);

//
// Single-step L-BFGS (mirrors Python's _champq_lbfgs.LBFGS.step).
//
// The monolithic ts_lbfgs_minimize runs the whole loop. CHAMP-Q needs to
// interleave its own Sinkhorn projection between steps and recompute the
// loss/gradient at the projected point, so it drives the loop itself and
// calls ts_lbfgs_step once per iteration. The state struct holds the
// curvature ring buffer across steps.
//
// Usage:
//   ts_lbfgs_state * st = ts_lbfgs_state_create(n, history);
//   float loss = eval(x, grad, n, ctx);
//   for (it in iters) {
//       loss = ts_lbfgs_step(st, x, loss, grad, eval, ctx,
//                            grad_out, x_out, &done);
//       project(x_out);                         // CHAMP-Q Sinkhorn
//       memcpy(x, x_out, ...);
//       loss = eval(x, grad, n, ctx);           // recompute at projection
//       if (done) break;
//   }
//   ts_lbfgs_state_destroy(st);
struct ts_lbfgs_state;

struct ts_lbfgs_step_params {
    float    c1;          // Armijo sufficient-decrease constant, default 1e-4
    int      max_ls;      // max line-search trials, default 25
    float    ls_shrink;   // backtracking shrink factor, default 0.5
    float    curvature_eps; // skip pair when s.y <= eps, default 1e-12
};

// Create state for an (n,) parameter vector with `history` curvature pairs.
struct ts_lbfgs_state * ts_lbfgs_state_create(int64_t n, int64_t history);

void ts_lbfgs_state_destroy(struct ts_lbfgs_state * st);

// Reset the curvature buffer (e.g. after an external projection that
// invalidates the local quadratic model).
void ts_lbfgs_state_reset(struct ts_lbfgs_state * st);

// One L-BFGS step. Reads x (n,) and grad (n,) at the current point plus the
// current loss. Runs the two-loop recursion + backtracking Armijo line
// search, pushes the (s, y) curvature pair, and writes:
//   x_new    : accepted parameter vector (n,)
//   grad_new : gradient at x_new (n,)
// Returns the loss at x_new. `*done` is set to true when the line search
// could not satisfy Armijo (caller should stop). `params` may be null for
// defaults.
float ts_lbfgs_step(struct ts_lbfgs_state * st,
                    const float * x, float loss, const float * grad,
                    ts_lbfgs_eval_fn eval, void * eval_ctx,
                    const struct ts_lbfgs_step_params * params,
                    float * x_new, float * grad_new, int * done);
