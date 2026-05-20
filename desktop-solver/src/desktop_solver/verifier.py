"""Verify a computed answer by evaluating the LLM-supplied [VERIFY] expression.

The verify expression is expected to be zero (or boolean True) when the answer
is correct. Anything else — non-zero numeric, False, exception — is treated as
a verification failure and triggers a self-correction pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import sympy as sp


_ZERO_TOL = 1e-9


@dataclass
class VerifyResult:
    verified: bool
    value: Any = None
    reason: str = ""


def verify(expr_code: str, namespace: dict[str, Any]) -> VerifyResult:
    """Evaluate ``expr_code`` in ``namespace`` and check it represents truth/zero.

    The namespace is the one populated by ``executor.execute`` so that variables
    defined in the [CALC] block (e.g. ``x``, ``answer``) are in scope.
    """
    expr_code = expr_code.strip()
    if not expr_code or expr_code == "True":
        return VerifyResult(verified=True, value=True, reason="not applicable")

    try:
        value = eval(compile(expr_code, "<verify>", "eval"), namespace)  # noqa: S307
    except Exception as e:  # noqa: BLE001
        return VerifyResult(verified=False, reason=f"verify raised {type(e).__name__}: {e}")

    if isinstance(value, bool):
        return VerifyResult(verified=value, value=value,
                            reason="" if value else "verify returned False")

    if isinstance(value, (list, tuple)):
        results = [verify_one_value(v) for v in value]
        ok = all(r.verified for r in results)
        return VerifyResult(verified=ok, value=value,
                            reason="" if ok else "one or more substitutions non-zero")

    return verify_one_value(value)


def verify_one_value(value: Any) -> VerifyResult:
    try:
        simplified = sp.simplify(value)
    except Exception as e:  # noqa: BLE001
        return VerifyResult(verified=False, value=value, reason=f"simplify failed: {e}")

    if simplified == 0 or simplified is sp.S.Zero:
        return VerifyResult(verified=True, value=simplified, reason="")

    try:
        numeric = float(sp.N(simplified))
        if abs(numeric) <= _ZERO_TOL:
            return VerifyResult(verified=True, value=simplified, reason="")
        return VerifyResult(verified=False, value=simplified,
                            reason=f"residual {numeric:.3e} (expected 0)")
    except (TypeError, ValueError):
        return VerifyResult(verified=False, value=simplified,
                            reason=f"residual is non-numeric: {simplified}")
