"""Extract labelled blocks ([CALC], [VERIFY], [LATEX], [STEPS]) from LLM output."""

from __future__ import annotations

import re
from dataclasses import dataclass


_BLOCK_RE = re.compile(
    r"\[(?P<tag>CALC|VERIFY|LATEX|STEPS)\]\s*(?P<body>.*?)\s*\[/(?P=tag)\]",
    re.DOTALL,
)


@dataclass(frozen=True)
class Blocks:
    calc: str
    verify: str
    latex: str
    steps: str

    @property
    def complete(self) -> bool:
        return bool(self.calc) and bool(self.verify) and bool(self.latex)


def extract(llm_output: str) -> Blocks:
    """Return the four block bodies. Missing blocks come back as empty strings."""
    found: dict[str, str] = {}
    for m in _BLOCK_RE.finditer(llm_output):
        tag = m.group("tag")
        if tag not in found:  # first occurrence wins
            found[tag] = m.group("body").strip()
    return Blocks(
        calc=found.get("CALC", ""),
        verify=found.get("VERIFY", ""),
        latex=found.get("LATEX", ""),
        steps=found.get("STEPS", ""),
    )
