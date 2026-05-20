"""SymPy-backed executor for LLM-emitted math code.

The model emits Python code that uses SymPy. We execute it in a tightly curated
namespace with builtins disabled. The code is expected to assign its final
result to a variable named ``answer``.
"""

from __future__ import annotations

import builtins as _builtins
import math
import signal
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import sympy as sp


# Builtins that are useful to small math snippets and pose no real risk.
_SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "divmod": divmod,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "pow": pow,
    "range": range,
    "reversed": reversed,
    "round": round,
    "set": set,
    "slice": slice,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
    "True": True,
    "False": False,
    "None": None,
    "print": lambda *a, **kw: None,  # swallow LLM debug prints
}


def _build_namespace() -> dict[str, Any]:
    ns: dict[str, Any] = {"__builtins__": _SAFE_BUILTINS, "sp": sp, "sympy": sp, "math": math}
    # Hoist common SymPy names so the LLM does not have to remember the module path.
    names = (
        "Symbol", "symbols", "Eq", "Ne", "Lt", "Le", "Gt", "Ge",
        "solve", "solveset", "linsolve", "nonlinsolve", "roots",
        "simplify", "expand", "factor", "collect", "cancel", "apart", "together",
        "sqrt", "cbrt", "root",
        "sin", "cos", "tan", "cot", "sec", "csc",
        "asin", "acos", "atan", "acot", "asec", "acsc",
        "sinh", "cosh", "tanh", "log", "ln", "exp",
        "Abs", "sign", "floor", "ceiling",
        "diff", "integrate", "limit", "series", "Derivative", "Integral",
        "Sum", "Product", "summation",
        "Matrix", "eye", "zeros", "ones", "diag",
        "Rational", "Float", "Integer", "S", "Mod",
        "pi", "E", "I", "oo", "zoo", "nan",
        "Interval", "FiniteSet", "Union", "Intersection", "Complement",
        "Reals", "Naturals", "Integers", "Rationals",
        "N", "nsimplify", "trigsimp", "radsimp", "ratsimp",
        "binomial", "factorial", "primerange", "isprime", "gcd", "lcm",
    )
    for n in names:
        attr = getattr(sp, n, None)
        if attr is not None:
            ns[n] = attr
    return ns


class _TimeoutError(RuntimeError):
    """Internal timeout signal."""


@contextmanager
def _alarm_timeout(seconds: int):
    """POSIX SIGALRM-based timeout. Falls back to no-op on platforms without SIGALRM
    (e.g. Windows); we still get a bounded run via SymPy's own internal limits."""
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def _handler(signum, frame):  # noqa: ARG001
        raise _TimeoutError(f"execution exceeded {seconds}s")

    previous = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


@dataclass
class ExecutorResult:
    success: bool
    answer: Any = None  # the raw SymPy value, or list, or string
    repr_str: str = ""  # human-readable text form
    latex: str = ""  # SymPy-derived LaTeX (may be overridden by LLM block)
    error: str = ""
    namespace: dict[str, Any] = field(default_factory=dict)  # for verifier reuse


def _to_latex(value: Any) -> str:
    try:
        if isinstance(value, str):
            return rf"\text{{{value}}}"
        if isinstance(value, (list, tuple)):
            return r",\ ".join(_to_latex(v) for v in value)
        return sp.latex(sp.sympify(value))
    except Exception:  # pragma: no cover - defensive
        return ""


def _to_repr(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(_to_repr(v) for v in value)
    if isinstance(value, str):
        return value
    try:
        return sp.sstr(sp.sympify(value))
    except Exception:
        return str(value)


def execute(code: str, *, timeout_seconds: int = 8) -> ExecutorResult:
    """Run the LLM's [CALC] block and return the value bound to `answer`."""
    code = code.strip()
    if not code:
        return ExecutorResult(False, error="empty code")

    namespace = _build_namespace()
    try:
        with _alarm_timeout(timeout_seconds):
            exec(compile(code, "<calc>", "exec"), namespace)  # noqa: S102
    except _TimeoutError as e:
        return ExecutorResult(False, error=str(e), namespace=namespace)
    except SyntaxError as e:
        return ExecutorResult(False, error=f"syntax error: {e.msg} (line {e.lineno})", namespace=namespace)
    except Exception as e:  # noqa: BLE001 — any user-code failure surfaces as result
        return ExecutorResult(False, error=f"{type(e).__name__}: {e}", namespace=namespace)

    if "answer" not in namespace:
        return ExecutorResult(False, error="code did not assign `answer`", namespace=namespace)

    value = namespace["answer"]
    if isinstance(value, str) and value.startswith("ERROR"):
        return ExecutorResult(False, error=value, namespace=namespace)
    if isinstance(value, str) and value == "NOT_MATH":
        return ExecutorResult(False, error="non-mathematical input", namespace=namespace)

    return ExecutorResult(
        success=True,
        answer=value,
        repr_str=_to_repr(value),
        latex=_to_latex(value),
        namespace=namespace,
    )
