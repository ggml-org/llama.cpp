#!/usr/bin/env python3
"""Pure-Python L-BFGS optimizer.

Limited-memory BFGS with two-loop recursion and backtracking Armijo
line search. The L-BFGS algorithm itself is in pure Python; matrix
operations inside the supplied loss/gradient closure use numpy (the
L-BFGS state and arithmetic are all numpy vector ops over a flat
parameter vector, so the L-BFGS overhead is negligible compared to
the loss evaluation).

Used by tools/tessera/champq_permute.py to learn a doubly-stochastic
channel permutation that minimises a smoothness proxy. The L-BFGS
search direction is taken in the unconstrained M parameter space;
the doubly-stochastic projection is applied after each accepted step
via Sinkhorn normalisation (see ``champq_permute.sinkhorn_project``).

The interface is intentionally minimal:

    opt = LBFGS(n_params, history=10, c1=1e-4, max_ls=25)
    for it in range(max_iters):
        loss, grad = closure(x)
        x, loss, grad, done = opt.step(x, loss, grad, closure)
        if done:
            break

``closure(x) -> (loss, grad)`` must return a scalar float and a flat
numpy array of length n_params. The returned ``grad`` on iteration
``it+1`` must be the gradient at the new ``x`` (i.e. the caller is
expected to use ``grad_new`` returned by ``step`` on the next call;
the optimizer's curvature update uses the (s, y) pair derived from
``x_new - x`` and ``grad_new - grad``).

A simpler ``projected_gradient_descent`` helper is also exported for
cases where the L-BFGS line search is too expensive (e.g. when the
projection itself dominates the per-step cost). The two interfaces
share the same ``M_new = M - lr * direction`` style update; L-BFGS
just chooses ``direction`` from a low-rank inverse-Hessian
approximation instead of the raw gradient.
"""
from __future__ import annotations

from typing import Callable, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# L-BFGS state
# ---------------------------------------------------------------------------


class LBFGS:
    """Limited-memory BFGS.

    Buffers ``s`` and ``y`` (consecutive step and gradient-difference
    pairs) and ``rho`` (1 / (s . y)) are kept in a ring of length
    ``history`` (default 10, the L-BFGS standard). The two-loop
    recursion forms the search direction; the backtracking Armijo
    line search picks the step length.

    The initial inverse-Hessian scale is the O(L-BFGS) choice
    ``gamma = (s_{k-1} . y_{k-1}) / (y_{k-1} . y_{k-1})``. If no
    history is available (first step), gamma defaults to 1.0.
    """

    def __init__(
        self,
        n_params: int,
        history: int = 10,
        c1: float = 1.0e-4,
        max_ls: int = 25,
        ls_shrink: float = 0.5,
        ls_grow: float = 2.0,
        curvature_eps: float = 1.0e-12,
    ) -> None:
        if n_params <= 0:
            raise ValueError(f"n_params must be positive, got {n_params}")
        if history < 1:
            raise ValueError(f"history must be >= 1, got {history}")
        self.n_params = int(n_params)
        self.history = int(history)
        self.c1 = float(c1)
        self.max_ls = int(max_ls)
        self.ls_shrink = float(ls_shrink)
        self.ls_grow = float(ls_grow)
        self.curvature_eps = float(curvature_eps)

        # Ring buffers of curvature pairs.
        self._S: List[np.ndarray] = []
        self._Y: List[np.ndarray] = []
        self._rho: List[float] = []

    # ------------------------------------------------------------------
    # Two-loop recursion
    # ------------------------------------------------------------------

    def two_loop(self, grad: np.ndarray) -> np.ndarray:
        """Return H_k * grad, the L-BFGS inverse-Hessian-vector product
        applied to ``grad``.
        """
        q = np.asarray(grad, dtype=np.float64).copy()
        m = len(self._S)
        alpha = np.zeros(m, dtype=np.float64)
        for i in range(m - 1, -1, -1):
            alpha[i] = self._rho[i] * np.dot(self._S[i], q)
            q = q - alpha[i] * self._Y[i]
        if m > 0:
            sy = float(np.dot(self._S[-1], self._Y[-1]))
            yy = float(np.dot(self._Y[-1], self._Y[-1]))
            gamma = sy / (yy + self.curvature_eps)
        else:
            gamma = 1.0
        r = gamma * q
        for i in range(m):
            beta = self._rho[i] * np.dot(self._Y[i], r)
            r = r + (alpha[i] - beta) * self._S[i]
        return r

    # ------------------------------------------------------------------
    # Buffer update
    # ------------------------------------------------------------------

    def _append(self, s: np.ndarray, y: np.ndarray) -> None:
        sy = float(np.dot(s, y))
        if sy <= self.curvature_eps:
            # Curvature condition violated (e.g. line search did not
            # make progress); skip this pair. Keeping stale pairs in
            # the buffer is worse than discarding.
            self._S.clear()
            self._Y.clear()
            self._rho.clear()
            return
        self._S.append(s.copy())
        self._Y.append(y.copy())
        self._rho.append(1.0 / sy)
        if len(self._S) > self.history:
            self._S.pop(0)
            self._Y.pop(0)
            self._rho.pop(0)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self,
        x: np.ndarray,
        loss: float,
        grad: np.ndarray,
        closure: Callable[[np.ndarray], Tuple[float, np.ndarray]],
    ) -> Tuple[np.ndarray, float, np.ndarray, bool]:
        """Take one L-BFGS step.

        Args:
            x: current parameter vector (flat, length n_params).
            loss: closure(x).
            grad: closure-derivative at x (flat, length n_params).
            closure: callable mapping a flat vector to (loss, grad).
                Called as needed for the line search.

        Returns:
            (x_new, loss_new, grad_new, done). The caller must use
            ``grad_new`` (not the original ``grad``) on the next call
            to ``step`` so the curvature update uses the right pair.
            ``done`` is True when the line search could not satisfy
            Armijo; the caller should treat it as a non-improvement
            and stop iterating.
        """
        x = np.asarray(x, dtype=np.float64)
        grad = np.asarray(grad, dtype=np.float64)
        if x.shape != (self.n_params,):
            raise ValueError(
                f"x shape {x.shape} does not match n_params {self.n_params}"
            )
        if grad.shape != (self.n_params,):
            raise ValueError(
                f"grad shape {grad.shape} does not match n_params {self.n_params}"
            )

        # L-BFGS search direction: d = -H * grad (a descent direction
        # when the curvature pair buffer is in good shape). We then
        # take x_new = x + alpha * d, the standard convention.
        direction = -self.two_loop(grad)
        dg = float(np.dot(grad, direction))
        if dg >= 0.0:
            # Not a descent direction (e.g. after a reset). Fall back
            # to the raw negative gradient so the line search still
            # has a usable direction.
            direction = -grad
            dg = float(np.dot(grad, direction))

        # Backtracking Armijo with initial step 1.0 (the L-BFGS ideal
        # when H approximates the inverse Hessian well). Shrink by
        # ``ls_shrink`` until the Armijo condition is met.
        alpha = 1.0
        loss_new = float(loss)
        x_new = x.copy()
        grad_new = grad.copy()
        accepted = False
        for _ in range(self.max_ls):
            candidate = x + alpha * direction
            cand_loss, cand_grad = closure(candidate)
            if cand_loss <= float(loss) + self.c1 * alpha * dg:
                x_new = candidate
                loss_new = float(cand_loss)
                # Reuse the gradient evaluation the closure already
                # performed; saves one closure call per accepted step.
                grad_new = cand_grad
                accepted = True
                break
            alpha *= self.ls_shrink
        if not accepted:
            # The line search could not satisfy Armijo. Return the
            # best (smallest-alpha) candidate as a courtesy; the
            # caller may treat this as a non-improvement.
            candidate = x + alpha * direction
            cand_loss, cand_grad = closure(candidate)
            x_new = candidate
            loss_new = float(cand_loss)
            grad_new = cand_grad

        # Curvature update: s = x_new - x, y = grad_new - grad.
        s = x_new - x
        y = grad_new - grad
        self._append(s, y)

        return x_new, loss_new, grad_new, not accepted

    def update(self, s: np.ndarray, y: np.ndarray) -> None:
        """Manually push a curvature pair onto the ring buffer."""
        self._append(np.asarray(s, dtype=np.float64), np.asarray(y, dtype=np.float64))

    def reset(self) -> None:
        self._S.clear()
        self._Y.clear()
        self._rho.clear()


# ---------------------------------------------------------------------------
# Projected gradient descent (simpler fallback)
# ---------------------------------------------------------------------------


def projected_gradient_descent(
    closure: Callable[[np.ndarray], Tuple[float, np.ndarray]],
    x0: np.ndarray,
    n_iters: int,
    lr: float = 1.0e-2,
    project: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    tol: float = 1.0e-6,
    verbose: bool = False,
) -> Tuple[np.ndarray, List[float]]:
    """Vanilla projected gradient descent.

    Useful when the line search inside L-BFGS is too expensive (each
    Armijo trial costs a closure call, which for a Sinkhorn-projected
    loss already pays for the projection). The simpler PGD update
    ``M = project(M - lr * grad)`` is often enough for the
    doubly-stochastic smoothness objective.

    Args:
        closure: x -> (loss, grad).
        x0: starting parameter vector.
        n_iters: number of gradient steps.
        lr: learning rate.
        project: optional projection closure applied after each step.
            Pass None to skip projection (unconstrained descent).
        tol: stop early if the gradient norm drops below this.
        verbose: log loss each iteration to stderr.

    Returns:
        (x_final, history) where ``history[i]`` is the loss at
        iteration ``i`` (length n_iters + 1).
    """
    x = np.asarray(x0, dtype=np.float64).copy()
    history: List[float] = []
    loss, grad = closure(x)
    history.append(float(loss))
    for it in range(int(n_iters)):
        gnorm = float(np.linalg.norm(grad))
        if gnorm < tol:
            break
        x = x - lr * grad
        if project is not None:
            x = project(x)
        loss, grad = closure(x)
        history.append(float(loss))
        if verbose:
            print(
                f"  pgd iter {it:4d}/{n_iters}  loss={loss:.6e}  |g|={gnorm:.3e}",
                flush=True,
            )
    return x, history


# ---------------------------------------------------------------------------
# Self-test (only run when invoked as a script)
# ---------------------------------------------------------------------------


def _self_test() -> None:
    """Verify L-BFGS on a separable quadratic.

    The minimum is at x* = b; the loss is 0.5 * sum_i a_i * (x_i - b_i)^2.
    With ``a = ones``, the Hessian is the identity, so L-BFGS should
    converge in 1 step from any starting x. With ``a != 1`` (diagonal
    Hessian), it should converge in at most a few dozen steps.
    """
    rng = np.random.default_rng(0)
    n = 16

    # Test 1: identity Hessian, x* = b. Should converge in 1 step.
    a = np.ones(n, dtype=np.float64)
    b = rng.normal(size=n)

    def closure(x: np.ndarray) -> Tuple[float, np.ndarray]:
        r = x - b
        loss = 0.5 * float(np.sum(a * r * r))
        grad = a * r
        return loss, grad

    opt = LBFGS(n, history=5, c1=1e-4, max_ls=20)
    x = np.zeros(n, dtype=np.float64)
    loss, grad = closure(x)
    for it in range(5):
        x, loss, grad, done = opt.step(x, loss, grad, closure)
        if done:
            break
    err = float(np.linalg.norm(x - b))
    assert err < 1e-5, f"L-BFGS (identity Hessian) did not converge; |x - x*| = {err:.3e}"

    # Test 2: non-identity diagonal Hessian. Use a moderate condition
    # number so the convergence rate is observable. L-BFGS with
    # history=5 should reach ~1e-3 in a few dozen steps.
    a = rng.uniform(0.5, 2.0, size=n)
    b = rng.normal(size=n)
    opt = LBFGS(n, history=5, c1=1e-4, max_ls=20)
    x = np.zeros(n, dtype=np.float64)
    loss, grad = closure(x)
    for it in range(50):
        x, loss, grad, done = opt.step(x, loss, grad, closure)
        if done:
            break
    err = float(np.linalg.norm(x - b))
    assert err < 1e-2, f"L-BFGS (diagonal Hessian) did not converge; |x - x*| = {err:.3e}"

    # Test 3: projected GD: minimize 0.5 * |x - b|^2 over x with x >= 0.
    # Project by clipping negatives.
    a = np.ones(n, dtype=np.float64)
    b = rng.normal(size=n)

    def project(x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0.0)

    x0 = np.full(n, -1.0, dtype=np.float64)
    x_final, hist = projected_gradient_descent(
        closure, x0, n_iters=200, lr=0.5, project=project, tol=1e-9
    )
    expected = np.maximum(b, 0.0)
    err = float(np.linalg.norm(x_final - expected))
    assert err < 1e-3, f"PGD did not converge; |x - x*| = {err:.3e}"


if __name__ == "__main__":
    _self_test()
    print("OK: _champq_lbfgs self-test passed")
