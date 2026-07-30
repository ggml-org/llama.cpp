#!/usr/bin/env python3

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "tools" / "tessera" / "replay-corpus.py"
SPEC = importlib.util.spec_from_file_location("replay_corpus", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_semantic_replay_is_deterministic_and_stratified():
    paragraphs = [
        "Write a Python function that sorts records.",
        "Debug this Rust class and explain the compiler error.",
        "Prove the matrix equation using a theorem.",
        "Calculate the probability and show the derivative.",
        "Imagine a character and write a short story.",
        "Create a poem and rewrite this scene.",
        "Assess the privacy and security risk.",
        "Explain this medical safety policy.",
        "Describe the image and interpret the chart.",
        "What happened in the history of this science?",
        "Compare both approaches and explain the tradeoff.",
        "Return a JSON schema for the API tool.",
    ]
    first = MODULE.select_semantic_replay(paragraphs, 0.5, "receipt")
    second = MODULE.select_semantic_replay(paragraphs, 0.5, "receipt")
    assert first == second
    selected, source_families, selected_families = first
    assert len(selected) == 6
    assert len(selected_families) == 6
    assert set(selected_families).issubset(source_families)


if __name__ == "__main__":
    test_semantic_replay_is_deterministic_and_stratified()
