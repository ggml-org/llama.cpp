#!/usr/bin/env python3

import argparse
import json
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import requests
from tqdm import tqdm
import random

cache_dir = Path.home() / ".cache" / "huggingface" / "datasets"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["HF_DATASETS_CACHE"] = str(cache_dir)
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

GRADER_PATTERNS = {
    "aime": r'\boxed{(\d+)}|\b(\d+)\b',
    "gsm8k": r'\b(\d+)\b',
    "mmlu": r'[A-D]',
    "hellaswag": r'[A-D]',
    "arc": r'[A-D]',
    "winogrande": r'[A-D]',
}

TEMPLATE_REGISTRY = {
    "aime": """{question}
Please reason step by step, and put your final answer within \\boxed{{}}.
""",
}

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
    extracted: Optional[str] = None
    correct: bool = False
    status: str = "pending"

def normalize_number(s: str) -> Optional[int]:
    match = re.match(r"\d+", s)  # match digits from the start
    if not match:
        return None
    return int(match.group(0))

class AimeDataset:
    def __init__(self, split: str = "train"):
        self.split = split
        self.questions: List[Dict] = []
        self._load_dataset()

    def _load_dataset(self):
        print(f"Loading AIME dataset (split: {self.split})...")
        from datasets import load_dataset

        cache_path = cache_dir / "AI-MO___aimo-validation-aime" / "default" / "0.0.0"
        if cache_path.exists():
            print(f"Using cached dataset from {cache_path}")
            ds = load_dataset("AI-MO/aimo-validation-aime", split=self.split, cache_dir=str(cache_path))
        else:
            ds = load_dataset("AI-MO/aimo-validation-aime", split=self.split)

        self.questions = []
        for row in ds:
            question = dict(row)
            question["dataset_type"] = "aime"
            self.questions.append(question)

        print(f"AIME dataset loaded: {len(self.questions)} questions")

    def get_question(self, index: int) -> Dict:
        """Get question by index"""
        return self.questions[index]

    def get_answer(self, question: Dict) -> str:
        answer = question["answer"]
        if isinstance(answer, str):
            normalized = normalize_number(answer)
            return str(normalized) if normalized is not None else answer
        return str(answer)

class Grader:
    def __init__(
        self,
        grader_type: str = "regex",
        grader_regex_type: str = "aime",
        grader_script: Optional[str] = None,
        judge_model_name: Optional[str] = None,
        judge_server_url: str = ""
    ):
        self.grader_type = grader_type
        self.grader_regex_type = grader_regex_type
        self.grader_script = grader_script
        self.judge_model_name = judge_model_name
        self.judge_server_url = judge_server_url
        self.pattern = self._get_pattern()

    def _get_pattern(self) -> Optional[str]:
        if self.grader_type == "regex":
            if self.grader_regex_type not in GRADER_PATTERNS:
                raise ValueError(f"Unknown grader regex type: {self.grader_regex_type}")
            return GRADER_PATTERNS[self.grader_regex_type]
        return None

    def _extract_answer_regex(self, pred: str) -> Optional[str]:
        """Extract answer using regex pattern"""
        if not self.pattern:
            return None
        matches = re.findall(self.pattern, pred, re.IGNORECASE)
        if not matches:
            return None

        for match in matches:
            if isinstance(match, tuple):
                match = match[0] if match[0] else match[1]
            extracted = match.strip()
            if extracted:
                return extracted
        return None

    def _grade_regex(self, gold: str, pred: str) -> Tuple[bool, Optional[str]]:
        """Grade using regex pattern matching"""
        extracted = self._extract_answer_regex(pred)
        if extracted is None:
            return False, None
        is_correct = extracted.strip() == gold.strip()
        return is_correct, extracted

    def _grade_cli(self, gold: str, pred: str) -> Tuple[bool, Optional[str]]:
        """Grade using external CLI script"""
        if not self.grader_script:
            raise ValueError("CLI grader requires --grader-script")

        script_path = Path(self.grader_script)
        if not script_path.exists():
            raise FileNotFoundError(f"Grader script not found: {self.grader_script}")

        try:
            result = subprocess.run(
                [str(script_path), "--answer", pred, "--expected", gold],
                capture_output=True,
                text=True,
                timeout=30
            )
            is_correct = result.returncode == 0
            extracted = pred if is_correct else None
            return is_correct, extracted
        except subprocess.TimeoutExpired:
            return False, None
        except Exception as e:
            return False, None

    def _grade_llm(self, gold: str, pred: str, problem: str) -> Tuple[bool, Optional[str]]:
        """Grade using LLM-based extraction"""
        prompt = f"""Extract the answer from this response:

Response: {pred}

Expected answer: {gold}

Please provide only the extracted answer, nothing else."""
        url = f"{self.judge_server_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        data = {
            "model": self.judge_model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 256
        }

        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            extracted = response.json()["choices"][0]["message"]["content"].strip()
            is_correct = extracted.strip().lower() == gold.strip().lower()
            return is_correct, extracted
        except Exception as e:
            return False, None

    def _truncate_response(self, response: str, max_lines: int = 3) -> str:
        """Keep only last N lines of response"""
        lines = response.split('\n')
        return '\n'.join(lines[-max_lines:]) if len(lines) > max_lines else response

    def grade(self, gold: str, pred: str, problem: str = "") -> Tuple[bool, Optional[str]]:
        """Grade the response"""
        if self.grader_type == "regex":
            return self._grade_regex(gold, pred)
        elif self.grader_type == "cli":
            return self._grade_cli(gold, pred)
        elif self.grader_type == "llm":
            return self._grade_llm(gold, pred, problem)
        else:
            raise ValueError(f"Unknown grader type: {self.grader_type}")

class Processor:
    def __init__(
        self,
        server_url: str,
        n_predict: int = 2048,
        threads: int = 32,
        verbose: bool = False,
        grader: Optional[Grader] = None,
        model_name: Optional[str] = None,
        judge_server_url: str = "",
        judge_model_name: Optional[str] = None
    ):
        self.server_url = server_url
        self.n_predict = n_predict
        self.threads = threads
        self.verbose = verbose
        self.model_name = model_name
        self.judge_server_url = judge_server_url if judge_server_url else server_url
        self.judge_model_name = judge_model_name
        self.dataset = AimeDataset()
        self.grader = grader or Grader()
        self.eval_state = EvalState(
            id="aime-2025",
            tasks=["aime"],
            task_states={},
            sampling_config={"temperature": 0, "max_tokens": n_predict}
        )

        # Pass judge configuration to grader if using LLM grader
        if self.grader.grader_type == "llm":
            if self.judge_model_name:
                self.grader.judge_model_name = self.judge_model_name
            if self.judge_server_url:
                self.grader.judge_server_url = self.judge_server_url

    def _make_request(self, prompt: str) -> Dict[str, Any]:
        """Make HTTP request to the server"""
        url = f"{self.server_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        data = {
            "model": self.model_name if self.model_name else "llama",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": self.n_predict
        }

        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()

    def _process_single_case(self, i: int, task_id: str) -> TaskState:
        """Process a single case (thread-safe)"""
        question = self.dataset.get_question(i)
        dataset_id = f"aime_{self.dataset.split}_{question['id']}"
        gold = self.dataset.get_answer(question)

        # Apply template if available
        if question["dataset_type"] in TEMPLATE_REGISTRY:
            prompt = TEMPLATE_REGISTRY[question["dataset_type"]].format(question=question["problem"])
        else:
            prompt = question["problem"]

        task_state = TaskState(
            case_id=task_id,
            prompt=prompt,
            gold=gold
        )

        try:
            response = self._make_request(prompt)
            pred = response["choices"][0]["message"]["content"]
            task_state.pred = pred

            # Truncate response to last 2-3 lines for grading
            pred_truncated = self.grader._truncate_response(pred, max_lines=3)

            # Grade the response
            is_correct, extracted = self.grader.grade(gold, pred_truncated, prompt)
            task_state.correct = is_correct
            task_state.extracted = extracted
            task_state.status = "ok"
        except Exception as e:
            task_state.status = f"error: {str(e)}"

        return task_state

    def process(self, n_cases: int = None, seed: int = 1234):
        """Process cases and update eval state"""
        if n_cases is None:
            n_cases = len(self.dataset.questions)

        print(f"\nProcessing {n_cases} AIME questions...")
        print(f"Server: {self.server_url}")
        print(f"Threads: {self.threads}")
        print(f"Max tokens: {self.n_predict}")
        print(f"Grader: {self.grader.grader_type}", end="")
        if self.grader.grader_type == "llm":
            print(f" (judge server: {self.judge_server_url}, model: {self.judge_model_name})", end="")
        print()
        print()

        dataset_size = len(self.dataset.questions)
        random.seed(seed)

        task_list = []
        for chunk_idx in range((n_cases + dataset_size - 1) // dataset_size):
            chunk_size = min(dataset_size, n_cases - chunk_idx * dataset_size)
            indices = list(range(dataset_size))
            random.shuffle(indices)
            chunk_indices = indices[:chunk_size]

            for i in chunk_indices:
                task_id = f"aime_{self.eval_state.id}_{chunk_idx:03d}_{i:03d}"
                task_list.append((i, task_id))

        # Print task summary table
        print("Tasks:")
        print("  Task ID         Dataset    Prompt (first 40 chars)                        Expected    Status")
        for i, task_id in task_list:
            question = self.dataset.get_question(i)
            prompt = question["problem"]
            gold = self.dataset.get_answer(question)
            truncated_prompt = prompt[:40] + "..." if len(prompt) > 40 else prompt
            print(f"  {task_id:<15} AIME2025   {truncated_prompt:<40}    {gold:<10} pending")
        print()

        task_states: Dict[str, List[TaskState]] = {task: [] for task in self.eval_state.tasks}
        total = 0
        correct = 0

        with ThreadPoolExecutor(max_workers=self.threads) as executor:
            futures = {executor.submit(self._process_single_case, i, task_id): (i, task_id) for i, task_id in task_list}

            for future in as_completed(futures):
                task_state = future.result()
                task_states["aime"].append(task_state)
                total += 1

                if task_state.correct:
                    correct += 1

                # Print task completion status
                extracted_display = task_state.extracted if task_state.extracted else "N/A"
                success_ratio = correct / total if total > 0 else 0.0
                print(f"{total:3}/{n_cases:3}  {task_state.case_id:<15} AIME2025   {task_state.prompt[:40]:<40}    {task_state.gold:<10} {extracted_display:<10} {'✓' if task_state.correct else '✗'}  [{correct:3}/{total:3}, {success_ratio:.3f}]")

                if self.verbose:
                    print(f"\nCase {total}: {task_state.correct}")
                    print(f"  Gold: {task_state.gold}")
                    if task_state.pred:
                        print(f"  Pred: {task_state.pred}")
                    if task_state.extracted:
                        print(f"  Extracted: {task_state.extracted}")
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
        "--seed",
        type=int,
        default=1234,
        help="Random seed for shuffling (default: 1234)"
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
        "--model",
        type=str,
        default=None,
        help="Model name to append as query parameter (e.g., gpt-oss-20b-hf)"
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
    parser.add_argument(
        "--grader-type",
        type=str,
        default="regex",
        choices=["regex", "cli", "llm"],
        help="Grader type: regex, cli, or llm (default: regex)"
    )
    parser.add_argument(
        "--grader-regex-type",
        type=str,
        default="aime",
        choices=list(GRADER_PATTERNS.keys()),
        help="Regex grader type (default: aime)"
    )
    parser.add_argument(
        "--grader-script",
        type=str,
        default=None,
        help="CLI grader script path (required for --grader-type cli)"
    )
    parser.add_argument(
        "--judge-server",
        type=str,
        default="",
        help="Server URL for LLM judge (default: same as main server)"
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Model name for LLM judge (default: same as main model)"
    )

    args = parser.parse_args()

    grader = Grader(
        grader_type=args.grader_type,
        grader_regex_type=args.grader_regex_type,
        grader_script=args.grader_script
    )

    if args.grader_type == "llm" and not args.judge_server:
        print("Warning: Using same server for LLM judge (no --judge-server specified)")

    processor = Processor(
        server_url=args.server,
        n_predict=args.n_predict,
        threads=args.threads,
        verbose=args.verbose,
        grader=grader,
        model_name=args.model,
        judge_server_url=args.judge_server,
        judge_model_name=args.judge_model
    )

    eval_state = processor.process(n_cases=args.n_cases, seed=args.seed)
    processor.dump_state(args.output)

if __name__ == "__main__":
    main()
