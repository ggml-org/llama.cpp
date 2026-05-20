"""End-to-end orchestration: LLM → tag extraction → execute → verify → retry."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from . import tag_extractor
from .executor import ExecutorResult, execute
from .llm_engine import LLMEngine
from .verifier import VerifyResult, verify

log = logging.getLogger(__name__)

MAX_RETRIES = 2  # initial attempt + this many corrections


@dataclass
class SolveResult:
    problem: str
    answer_repr: str = ""
    latex: str = ""
    code: str = ""
    steps: str = ""
    verified: bool = False
    verify_reason: str = ""
    raw_llm: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def success(self) -> bool:
        return bool(self.answer_repr) and not self.error


def solve(engine: LLMEngine, problem: str, *,
          max_retries: int = MAX_RETRIES) -> SolveResult:
    """Run a full solve cycle with up to ``max_retries`` self-corrections."""
    result = SolveResult(problem=problem)
    failed_code = ""
    verify_repr = ""
    error_msg = ""

    for attempt in range(max_retries + 1):
        raw = (engine.query(problem) if attempt == 0
               else engine.query_with_error(problem, failed_code, verify_repr, error_msg))
        result.raw_llm.append(raw)
        blocks = tag_extractor.extract(raw)
        log.debug("attempt %d blocks: calc=%r verify=%r", attempt, blocks.calc[:60], blocks.verify[:60])

        if not blocks.calc:
            failed_code = ""
            error_msg = "LLM response missing [CALC] block"
            result.error = error_msg
            continue

        exec_res = execute(blocks.calc)
        if not exec_res.success:
            failed_code = blocks.calc
            error_msg = exec_res.error
            result.error = error_msg
            result.code = blocks.calc
            continue

        ver_res: VerifyResult = verify(blocks.verify, exec_res.namespace)
        result.answer_repr = exec_res.repr_str
        result.latex = blocks.latex or exec_res.latex
        result.code = blocks.calc
        result.steps = blocks.steps
        result.verified = ver_res.verified
        result.verify_reason = ver_res.reason
        result.error = ""

        if ver_res.verified:
            return result

        # Verification failed; feed it back and try again.
        failed_code = blocks.calc
        verify_repr = _safe_repr(ver_res.value)
        error_msg = ver_res.reason or "verification returned non-zero"

    return result


def _safe_repr(value: Any) -> str:
    try:
        s = repr(value)
    except Exception:  # noqa: BLE001
        s = "<unrepresentable>"
    return s[:400]
