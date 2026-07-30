#!/usr/bin/env python3

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE = Path(__file__).parents[1] / "tools/tessera/challenge-corpus.py"
SPEC = importlib.util.spec_from_file_location("tessera_challenge", MODULE)
CHALLENGE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = CHALLENGE
SPEC.loader.exec_module(CHALLENGE)


class ChallengeCorpusTest(unittest.TestCase):
    def test_selection_is_deterministic_and_category_balanced(self):
        records = []
        for category in ("code", "reasoning", "zh"):
            for index in range(5):
                records.append({
                    "id": f"{category}-{index}",
                    "category": category,
                    "text": ("{}\n" * index) + category + " <structured>" * index,
                })
        first = CHALLENGE.select(records, 2, 640)
        second = CHALLENGE.select(records, 2, 640)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 6)
        self.assertEqual({record["category"] for record in first}, {"code", "reasoning", "zh"})


if __name__ == "__main__":
    unittest.main()
