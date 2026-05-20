"""Unit tests for the executor, verifier, and tag extractor.

These cover every part of the pipeline that does not require the LLM.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from desktop_solver import tag_extractor
from desktop_solver.executor import execute
from desktop_solver.verifier import verify


# --- tag_extractor ----------------------------------------------------------

def test_extract_all_four_blocks():
    text = """
    [CALC]
    answer = 5
    [/CALC]
    [VERIFY]
    answer - 5
    [/VERIFY]
    [LATEX]
    5
    [/LATEX]
    [STEPS]
    Trivial.
    [/STEPS]
    """
    b = tag_extractor.extract(text)
    assert b.calc == "answer = 5"
    assert b.verify == "answer - 5"
    assert b.latex == "5"
    assert b.steps == "Trivial."
    assert b.complete


def test_missing_blocks_are_empty():
    b = tag_extractor.extract("[CALC]answer = 1[/CALC]")
    assert b.calc == "answer = 1"
    assert b.verify == ""
    assert b.latex == ""
    assert not b.complete


# --- executor: simple cases -------------------------------------------------

def test_linear_equation():
    r = execute("x = Symbol('x'); answer = solve(3*x + 7 - 22, x)[0]")
    assert r.success
    assert r.answer == 5


def test_quadratic_returns_sorted_pair():
    code = "x = Symbol('x'); answer = sorted(solve(x**2 + 5*x + 6, x))"
    r = execute(code)
    assert r.success
    assert r.answer == [-3, -2]


def test_trig_in_radians():
    r = execute("answer = sin(pi / 6)")
    assert r.success
    assert sp.simplify(r.answer - sp.Rational(1, 2)) == 0


def test_definite_integral():
    r = execute("x = Symbol('x'); answer = integrate(2*x + 3, (x, 0, 1))")
    assert r.success
    assert r.answer == 4


def test_derivative_at_point():
    r = execute("x = Symbol('x'); answer = diff(x**2 + 3*x, x).subs(x, 2)")
    assert r.success
    assert r.answer == 7


def test_probability_compound():
    r = execute("answer = Rational(7, 10) ** 5")
    assert r.success
    assert abs(float(r.answer) - 0.16807) < 1e-6


# --- executor: error handling ----------------------------------------------

def test_missing_answer_assignment():
    r = execute("x = 1 + 1")
    assert not r.success
    assert "answer" in r.error


def test_syntax_error():
    r = execute("answer = 1 +")
    assert not r.success
    assert "syntax" in r.error.lower()


def test_runtime_error():
    r = execute("answer = 1 / 0")
    assert not r.success
    assert "zero" in r.error.lower() or "ZeroDivision" in r.error


def test_not_math_marker_is_treated_as_failure():
    r = execute('answer = "NOT_MATH"')
    assert not r.success
    assert "non-math" in r.error.lower()


def test_error_marker_is_treated_as_failure():
    r = execute('answer = "ERROR: missing data"')
    assert not r.success


def test_builtins_are_blocked():
    # `open` is not in our safe builtins.
    r = execute('answer = open("/etc/passwd")')
    assert not r.success
    assert "NameError" in r.error or "open" in r.error


# --- verifier ---------------------------------------------------------------

def test_verify_zero_passes():
    r = execute("x = Symbol('x'); answer = 2; check = x - answer; check_at_2 = check.subs(x, 2)")
    assert r.success
    v = verify("check_at_2", r.namespace)
    assert v.verified


def test_verify_residual_fails():
    r = execute("x = Symbol('x'); answer = 3; check_at_2 = (x - 2).subs(x, answer)")
    # `answer=3` substituted into `x - 2` gives 1, not 0 — should fail.
    v = verify("check_at_2", r.namespace)
    assert not v.verified


def test_verify_true_passes_trivially():
    r = execute("answer = 1")
    v = verify("True", r.namespace)
    assert v.verified


def test_verify_list_of_substitutions():
    code = (
        "x = Symbol('x');"
        "answer = sorted(solve(x**2 - 4, x));"
        "subs = [(x**2 - 4).subs(x, a) for a in answer]"
    )
    r = execute(code)
    assert r.success
    v = verify("subs", r.namespace)
    assert v.verified


# --- formatting -------------------------------------------------------------

def test_latex_for_rational():
    r = execute("answer = Rational(3, 2)")
    assert r.success
    assert "frac" in r.latex
