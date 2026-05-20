"""llama-cpp-python wrapper for the desktop GCSE solver.

Designed for a workstation with ~8 GB of system RAM. Defaults assume a 7B
math-tuned model quantised to Q4_K_M (~4.4 GB on disk / RSS).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from .prompts import CORRECTION_PROMPT_SUFFIX, SYSTEM_PROMPT

log = logging.getLogger(__name__)


@dataclass
class EngineConfig:
    model_path: str
    n_ctx: int = 4096
    n_threads: Optional[int] = None
    n_gpu_layers: int = 0  # CPU-only by default; user can opt in via CLI
    seed: int = 1
    max_tokens: int = 1024
    temperature: float = 0.0
    top_p: float = 1.0
    repeat_penalty: float = 1.0
    extra: dict = field(default_factory=dict)


class LLMEngine:
    """Stateless wrapper over ``llama_cpp.Llama``.

    The model is loaded eagerly in ``__init__``. ``query`` builds a fresh prompt
    every call — we do not keep KV state across queries because each math problem
    is independent and we want deterministic behaviour.
    """

    def __init__(self, config: EngineConfig) -> None:
        from llama_cpp import Llama  # noqa: PLC0415 — imported lazily so GUI can boot without the model
        self._config = config
        kwargs = dict(
            model_path=config.model_path,
            n_ctx=config.n_ctx,
            n_gpu_layers=config.n_gpu_layers,
            seed=config.seed,
            logits_all=False,
            verbose=False,
        )
        if config.n_threads is not None:
            kwargs["n_threads"] = config.n_threads
        kwargs.update(config.extra)
        log.info("loading model: %s", config.model_path)
        self._llm = Llama(**kwargs)

    def query(self, problem: str) -> str:
        """Single-shot inference for one problem."""
        return self._complete(self._build_prompt(problem))

    def query_with_error(self, problem: str, failed_code: str,
                         verify_result: str, error: str) -> str:
        """Inference with explicit correction context."""
        correction = CORRECTION_PROMPT_SUFFIX.format(
            failed_code=failed_code or "(none)",
            verify_result=verify_result or "(none)",
            error=error or "(none)",
        )
        return self._complete(self._build_prompt(problem) + correction)

    # -- internals -----------------------------------------------------------

    def _build_prompt(self, problem: str) -> str:
        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"Problem: {problem.strip()}\n\n"
            "Response:\n"
        )

    def _complete(self, prompt: str) -> str:
        cfg = self._config
        result = self._llm(
            prompt,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            repeat_penalty=cfg.repeat_penalty,
            stop=["[/STEPS]"],
        )
        text = result["choices"][0]["text"]
        # Re-append the stop tag so [/STEPS] is preserved in the output.
        if "[STEPS]" in text and "[/STEPS]" not in text:
            text += "\n[/STEPS]"
        return text
