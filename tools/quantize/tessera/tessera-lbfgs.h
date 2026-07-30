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
