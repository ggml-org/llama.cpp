#!/usr/bin/env python3

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any
import requests
from tqdm import tqdm

cache_dir = Path.home() / ".cache" / "huggingface" / "datasets"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["HF_DATASETS_CACHE"] = str(cache_dir)

@dataclass
class EvalState:
    id: str
    tasks: List[str]
    task_states: Dict[str, Dict[str, Any]]
    sampling_config: Dict[str, Any]

@dataclass
class TaskState:
    case_id: str
    prompt: str
    gold: str
    pred: Optional[str] = None
    correct: bool = False
    status: str = "pending"

class AimeDataset:
    def __init__(self, split: str = "train"):
        self.split = split
        self.questions: List[Dict] = []
        self._load_dataset()

    def _load_dataset(self):
        print(f"Loading AIME dataset (split: {self.split})...")
        from datasets import load_dataset
        ds = load_dataset("AI-MO/aimo-validation-aime", split=self.split)
        self.questions = list(ds)
        print(f"AIME dataset loaded: {len(self.questions)} questions")

    def get_question(self, index: int) -> Dict:
        """Get question by index"""
        return self.questions[index]

    def get_answer(self, question: Dict) -> str:
        return str(question["answer"])

class Processor:
    def __init__(
        self,
        server_url: str,
        n_predict: int = 2048,
        threads: int = 32,
        verbose: bool = False
    ):
        self.server_url = server_url
        self.n_predict = n_predict
        self.threads = threads
        self.verbose = verbose
        self.dataset = AimeDataset()
        self.eval_state = EvalState(
            id="aime-2025",
            tasks=["aime"],
            task_states={},
            sampling_config={"temperature": 0, "max_tokens": n_predict}
        )

    def _make_request(self, prompt: str) -> Dict[str, Any]:
        """Make HTTP request to the server"""
        url = f"{self.server_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        data = {
            "model": "llama",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": self.n_predict
        }

        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()

    def _grade_response(self, gold: str, pred: str) -> bool:
        """Grade the response - abstracted for external grader support"""
        try:
            gold_int = int(gold)
            pred_int = int(pred)
            return gold_int == pred_int
        except (ValueError, TypeError):
            return False

    def process(self, n_cases: int = None, seed: int = 42):
        """Process cases and update eval state"""
        if n_cases is None:
            n_cases = len(self.dataset.questions)

        print(f"\nProcessing {n_cases} AIME questions...")
        print(f"Server: {self.server_url}")
        print(f"Threads: {self.threads}")
        print(f"Max tokens: {self.n_predict}")
        print()

        task_states: Dict[str, List[TaskState]] = {task: [] for task in self.eval_state.tasks}
        total = 0
        correct = 0

        for i in tqdm(range(min(n_cases, len(self.dataset.questions))), desc="Processing"):
            question = self.dataset.get_question(i)
            case_id = f"aime_{self.dataset.split}_{question['id']}"
            prompt = question["problem"]
            gold = self.dataset.get_answer(question)

            task_state = TaskState(
                case_id=case_id,
                prompt=prompt,
                gold=gold
            )

            try:
                response = self._make_request(prompt)
                pred = response["choices"][0]["message"]["content"]
                task_state.pred = pred
                task_state.correct = self._grade_response(gold, pred)
                task_state.status = "ok"

                if task_state.correct:
                    correct += 1
            except Exception as e:
                task_state.status = f"error: {str(e)}"

            task_states["aime"].append(task_state)
            total += 1

            if self.verbose:
                print(f"\nCase {i+1}/{total}: {task_state.correct}")
                print(f"  Gold: {gold}")
                if task_state.pred:
                    print(f"  Pred: {task_state.pred}")
                print(f"  Status: {task_state.status}")

        self.eval_state.task_states["aime"] = {
            "total": total,
            "correct": correct,
            "cases": task_states
        }

        print(f"\n{'='*60}")
        print(f"Results: {correct}/{total} correct ({correct/total*100:.1f}%)")
        print(f"{'='*60}")

        return self.eval_state

    def dump_state(self, output_file: Path):
        """Dump eval state to JSON file"""
        with open(output_file, "w") as f:
            json.dump(asdict(self.eval_state), f, indent=2)
        print(f"\nEval state dumped to {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Simplified AIME evaluation tool for llama.cpp"
    )
    parser.add_argument(
        "--server",
        type=str,
        default="http://localhost:8033",
        help="llama-server URL (default: http://localhost:8033)"
    )
    parser.add_argument(
        "--n_cases",
        type=int,
        default=None,
        help="Number of cases to evaluate (default: all)"
    )
    parser.add_argument(
        "--n_predict",
        type=int,
        default=2048,
        help="Max tokens to predict per prompt (default: 2048)"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=32,
        help="Number of threads for parallel requests (default: 32)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed output for each case"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("llama-eval-state.json"),
        help="Output file for eval state (default: llama-eval-state.json)"
    )

    args = parser.parse_args()

    processor = Processor(
        server_url=args.server,
        n_predict=args.n_predict,
        threads=args.threads,
        verbose=args.verbose
    )

    eval_state = processor.process(n_cases=args.n_cases)
    processor.dump_state(args.output)

if __name__ == "__main__":
    main()
