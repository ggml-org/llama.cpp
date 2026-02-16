#!/usr/bin/env python3
# type: ignore

import argparse
import json
import os
import re
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict, field
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
    "aime2025": r'\boxed{(\d+)}|\b(\d+)\b',
    "gsm8k": r'\b(\d+)\b',
}

SAMPLE_ANSWERS = {
    "aime": [
        "42",
        "-123",
        "999"
    ],
    "aime2025": [
        "42",
        "-123",
        "999"
    ],
    "gsm8k": [
        "42",
        "-123",
        "999"
    ],
    "gpqa": [
        "A",
        "D",
        "C"
    ],
}

TEMPLATE_REGISTRY = {
    "aime": """{question}
Please reason step by step, and put your final answer within \\boxed{{}}.
""",
    "aime2025": """{question}
Please reason step by step, and put your final answer within \\boxed{{}}.
""",
    "gsm8k": """{question}
Please reason step by step, and put your final numeric answer within \\boxed{{}} without any extra characters.
""",
    "gpqa": """{Question}

(A) {A}
(B) {B}
(C) {C}
(D) {D}

Express your final answer as the corresponding option 'A', 'B', 'C', or 'D'.
""",
}


class BaseDataset(ABC):
    @abstractmethod
    def get_question(self, index: int) -> Dict:
        pass

    @abstractmethod
    def get_answer(self, question: Dict) -> str:
        pass

    @abstractmethod
    def get_prompt(self, question: Dict) -> str:
        pass

    def __len__(self) -> int:
        return len(self.questions)


@dataclass
class TaskState:
    case_id: str
    prompt: str
    gold: str
    pred: Optional[str] = None
    extracted: Optional[str] = None
    grader_log: Dict[str, Any] = field(default_factory=dict)
    correct: bool = False
    status: str = "pending"


class EvalState:
    def __init__(
        self,
        dataset_type: str,
        sampling_config: Dict[str, Any],
        output_file: Path = Path("llama-eval-state.json")
    ):
        self.dataset_type = dataset_type
        self.sampling_config = sampling_config
        self.output_file = output_file
        self.dataset: Optional[BaseDataset] = None
        self.tasks: List[Tuple[int, str]] = []
        self.all_tasks: List[Tuple[int, str]] = []
        self.task_states: Dict[str, Any] = {}
        self.total = 0
        self.correct = 0
        self.processed = 0

    def load_dataset(self, seed: int = 1234):
        if self.dataset_type == "aime":
            self.dataset = AimeDataset()
        elif self.dataset_type == "aime2025":
            self.dataset = Aime2025Dataset()
        elif self.dataset_type == "gsm8k":
            self.dataset = Gsm8kDataset()
        elif self.dataset_type == "gpqa":
            self.dataset = GpqaDataset(variant="diamond", seed=seed)
        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")

    def setup_tasks(self, n_cases: Optional[int] = None, seed: int = 1234):
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")

        if n_cases is None:
            n_cases = len(self.dataset)

        dataset_size = len(self.dataset)
        rng = random.Random(seed)

        self.tasks = []
        for chunk_idx in range((n_cases + dataset_size - 1) // dataset_size):
            chunk_size = min(dataset_size, n_cases - chunk_idx * dataset_size)
            indices = list(range(dataset_size))
            rng.shuffle(indices)
            chunk_indices = indices[:chunk_size]

            for i in chunk_indices:
                task_id = f"{self.dataset_type}_{chunk_idx:03d}_{i:03d}"
                self.tasks.append((i, task_id))

        self.all_tasks = list(self.tasks)

    def get_case(self, index: int) -> Tuple[str, str]:
        if self.dataset is None:
            raise ValueError("Dataset not loaded.")
        question = self.dataset.get_question(index)
        prompt = self.dataset.get_prompt(question)
        gold = self.dataset.get_answer(question)
        return prompt, gold

    def add_result(
        self,
        task_id: str,
        prompt: str,
        gold: str,
        pred: Optional[str],
        extracted: Optional[str],
        grader_log: Dict[str, Any],
        correct: bool,
        status: str
    ):
        if "cases" not in self.task_states:
            self.task_states["cases"] = {}

        self.task_states["cases"][task_id] = {
            "case_id": task_id,
            "prompt": prompt,
            "gold": gold,
            "pred": pred,
            "extracted": extracted,
            "grader_log": grader_log,
            "correct": correct,
            "status": status
        }

        if correct:
            self.correct += 1
        else:
            self.correct = sum(1 for c in self.task_states.get("cases", {}).values() if c.get("correct", False))

    def print_progress(self, task_state: TaskState, total_tasks: int, correct_count: int = 0):
        extracted_display = task_state.extracted if task_state.extracted else "N/A"
        success_ratio = correct_count / self.processed if self.processed > 0 else 0.0
        first_line = task_state.prompt.split('\n')[0]
        truncated_prompt = first_line[:43]
        if len(first_line) > 43:
            truncated_prompt += "..."
        else:
            truncated_prompt = truncated_prompt.ljust(43) + "..."
        print(f"{self.processed:3}/{total_tasks:3}  {task_state.case_id:<20} {self.dataset_type.upper()}   {truncated_prompt:<40}    {task_state.gold:<10} {extracted_display:<10} {'✓' if task_state.correct else '✗'}  [{correct_count:3}/{self.processed:3}, {success_ratio:.3f}]")

    def print_summary(self):
        if self.total == 0:
            print(f"\n{'='*60}")
            print(f"Results: 0/0 correct (0.0%)")
            print(f"{'='*60}")
        else:
            print(f"\n{'='*60}")
            print(f"Results: {self.correct}/{self.total} correct ({self.correct/self.total*100:.1f}%)")
            print(f"{'='*60}")

    def dump(self):
        tasks_to_save = self.all_tasks if self.all_tasks else self.tasks
        all_cases = {}
        for i, task_id in tasks_to_save:
            prompt, gold = self.get_case(i)
            if task_id in self.task_states.get("cases", {}):
                all_cases[task_id] = self.task_states["cases"][task_id]
            else:
                all_cases[task_id] = {
                    "case_id": task_id,
                    "prompt": prompt,
                    "gold": gold,
                    "pred": None,
                    "extracted": None,
                    "grader_log": {},
                    "correct": False,
                    "status": "pending"
                }

        data = {
            "id": self.dataset_type,
            "tasks": [tid for _, tid in tasks_to_save],
            "task_states": {
                "total": self.total,
                "correct": self.correct,
                "cases": all_cases,
            },
            "sampling_config": self.sampling_config
        }
        with open(self.output_file, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "EvalState":
        with open(path, "r") as f:
            data = json.load(f)

        eval_state = cls(
            dataset_type=data["id"],
            sampling_config=data["sampling_config"],
            output_file=path
        )
        eval_state.load_dataset()

        eval_state.tasks = []
        eval_state.all_tasks = []
        for task_id in data.get("tasks", []):
            parts = task_id.rsplit("_", 2)
            if len(parts) >= 3:
                idx = int(parts[-1])
            else:
                idx = 0
            eval_state.tasks.append((idx, task_id))
            eval_state.all_tasks.append((idx, task_id))

        eval_state.task_states = data.get("task_states", {})

        cases = eval_state.task_states.get("cases", {})
        eval_state.total = eval_state.task_states.get("total", 0)
        eval_state.correct = eval_state.task_states.get("correct", 0)

        if eval_state.total == 0:
            eval_state.total = len(cases)
            eval_state.correct = sum(1 for c in cases.values() if c.get("correct", False))

        return eval_state

    def is_complete(self) -> bool:
        if not self.all_tasks:
            return False
        cases = self.task_states.get("cases", {})
        completed = {tid for tid in self.task_states.get("cases", {}).keys() if cases.get(tid, {}).get("status") == "ok"}
        return len(completed) == len(self.all_tasks)

    def get_pending_tasks(self) -> List[Tuple[int, str]]:
        cases = self.task_states.get("cases", {})
        pending = []
        for i, task_id in self.all_tasks:
            if cases.get(task_id, {}).get("status") != "ok":
                pending.append((i, task_id))
        return pending

    def print_all_tasks(self):
        cases = self.task_states.get("cases", {})
        tasks_to_show = self.all_tasks if self.all_tasks else self.tasks
        print()
        print("Tasks:")
        print("  Task ID             Dataset  Prompt (first 40 chars)                        Expected    Extracted    Status")
        for i, task_id in tasks_to_show:
            prompt, gold = self.get_case(i)
            case = cases.get(task_id, {})
            status = case.get("status", "pending")
            extracted = case.get("extracted", "N/A") if status == "ok" else "N/A"
            is_correct = case.get("correct", False) if status == "ok" else False
            symbol = "✓ " if is_correct else ("✗ " if status == "ok" else "")
            first_line = prompt.split('\n')[0]
            truncated_prompt = first_line[:43]
            if len(first_line) > 43:
                truncated_prompt += "..."
            else:
                truncated_prompt = truncated_prompt.ljust(43) + "..."
            print(f"  {task_id:<20} {self.dataset_type.upper()}   {truncated_prompt:<40}    {gold:<10} {extracted:<10} {symbol}{status}")
        print()

    def print_existing_summary(self):
        cases = self.task_states.get("cases", {})
        completed_cases = {tid: c for tid, c in cases.items() if c.get("status") == "ok"}
        correct = sum(1 for c in completed_cases.values() if c.get("correct", False))
        total = len(completed_cases)
        if total == 0:
            print(f"{'='*60}")
            print(f"Results: 0/0 correct (0.0%)")
            print(f"{'='*60}")
        else:
            print(f"{'='*60}")
            print(f"Results: {correct}/{total} correct ({correct/total*100:.1f}%)")
            print(f"{'='*60}")

def normalize_number(s: str) -> Optional[int]:
    match = re.match(r"\d+", s)  # match digits from the start
    if not match:
        return None
    return int(match.group(0))

class AimeDataset(BaseDataset):
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

    def get_prompt(self, question: Dict) -> str:
        """Get formatted prompt for the question"""
        if question["dataset_type"] == "gpqa":
            return TEMPLATE_REGISTRY["gpqa"].format(**question)
        else:
            return TEMPLATE_REGISTRY[question["dataset_type"]].format(
                question=question["problem"] if "problem" in question else question["question"]
            )

class Aime2025Dataset(BaseDataset):
    def __init__(self):
        self.questions: List[Dict] = []
        self._load_dataset()

    def _load_dataset(self):
        print(f"Loading AIME2025 dataset...")
        from datasets import load_dataset

        config_name = "AIME2025-I"
        cache_path = cache_dir / "opencompass___AIME2025" / "default" / "0.0.0"
        if cache_path.exists():
            print(f"Using cached dataset from {cache_path}")
            ds = load_dataset("opencompass/AIME2025", config_name, split="test", cache_dir=str(cache_path))
        else:
            ds = load_dataset("opencompass/AIME2025", config_name, split="test")

        self.questions = []
        for row in ds:
            question = dict(row)
            question["dataset_type"] = "aime2025"
            self.questions.append(question)

        print(f"AIME2025 dataset loaded: {len(self.questions)} questions")

        print(f"Loading AIME2025 dataset (part 2)...")
        config_name_2 = "AIME2025-II"
        cache_path_2 = cache_dir / "opencompass___AIME2025" / "default" / "0.0.0"
        if cache_path_2.exists():
            print(f"Using cached dataset from {cache_path_2}")
            ds_2 = load_dataset("opencompass/AIME2025", config_name_2, split="test", cache_dir=str(cache_path_2))
        else:
            ds_2 = load_dataset("opencompass/AIME2025", config_name_2, split="test")

        for row in ds_2:
            question = dict(row)
            question["dataset_type"] = "aime2025"
            self.questions.append(question)

        print(f"AIME2025 dataset loaded: {len(self.questions)} questions (total)")

    def get_question(self, index: int) -> Dict:
        """Get question by index"""
        return self.questions[index]

    def get_answer(self, question: Dict) -> str:
        answer = question["answer"]
        if isinstance(answer, str):
            normalized = normalize_number(answer)
            return str(normalized) if normalized is not None else answer
        return str(answer)

    def get_prompt(self, question: Dict) -> str:
        """Get formatted prompt for the question"""
        return TEMPLATE_REGISTRY["aime2025"].format(
            question=question["question"]
        )

class Gsm8kDataset(BaseDataset):
    def __init__(self, split: str = "test"):
        self.split = split
        self.questions: List[Dict] = []
        self._load_dataset()

    def _load_dataset(self):
        print(f"Loading GSM8K dataset (split: {self.split})...")
        from datasets import load_dataset

        cache_path = cache_dir / "openai___gsm8k" / "default" / "0.0.0"
        if cache_path.exists():
            print(f"Using cached dataset from {cache_path}")
            ds = load_dataset("openai/gsm8k", "main", split=self.split, cache_dir=str(cache_path))
        else:
            ds = load_dataset("openai/gsm8k", "main", split=self.split)

        self.questions = []
        for row in ds:
            question = dict(row)
            question["dataset_type"] = "gsm8k"

            # Extract numeric answer from the answer field (already has #### prefix)
            gold = question["answer"]
            # Split by #### and take the last part
            parts = gold.split("####")
            if len(parts) > 1:
                gold = parts[-1].strip()
            # Extract the first number from the remaining text
            normalized = normalize_number(gold)
            question["gold"] = str(normalized) if normalized is not None else gold

            self.questions.append(question)

        print(f"GSM8K dataset loaded: {len(self.questions)} questions")

    def get_question(self, index: int) -> Dict:
        """Get question by index"""
        return self.questions[index]

    def get_answer(self, question: Dict) -> str:
        # GSM8K has pre-extracted gold field, AIME uses answer field
        if "gold" in question:
            return question["gold"]
        answer = question["answer"]
        if isinstance(answer, str):
            normalized = normalize_number(answer)
            return str(normalized) if normalized is not None else answer
        return str(answer)

    def get_prompt(self, question: Dict) -> str:
        """Get formatted prompt for the question"""
        return TEMPLATE_REGISTRY[question["dataset_type"]].format(
            question=question["problem"] if "problem" in question else question["question"]
        )

class GpqaDataset(BaseDataset):
    def __init__(self, variant: str = "diamond", seed: int = 1234):
        self.variant = variant
        self.seed = seed
        self.questions: List[Dict] = []
        self._load_dataset()

    def _load_dataset(self):
        print(f"Loading GPQA dataset (variant: {self.variant})...")
        import pandas as pd

        url = f"https://openaipublic.blob.core.windows.net/simple-evals/gpqa_{self.variant}.csv"
        df = pd.read_csv(url)

        rng = random.Random(self.seed)

        self.questions = []
        for _, row in df.iterrows():
            question = row.to_dict()
            question["dataset_type"] = "gpqa"

            # Shuffle the answer options
            correct_answer = question["Correct Answer"]
            incorrect_answers = [
                question["Incorrect Answer 1"],
                question["Incorrect Answer 2"],
                question["Incorrect Answer 3"]
            ]

            # Create list of (answer, is_correct) tuples
            options = [(ans, ans == correct_answer) for ans in incorrect_answers]
            options.append((correct_answer, True))

            # Shuffle the options
            rng.shuffle(options)

            # Extract shuffled answers and determine correct letter
            shuffled_answers = [ans for ans, _ in options]
            correct_letter = chr(ord('A') + options.index((correct_answer, True)))

            # Store shuffled answers and correct letter
            question["shuffled_answers"] = shuffled_answers
            question["correct_letter"] = correct_letter

            self.questions.append(question)

        print(f"GPQA dataset loaded: {len(self.questions)} questions")

    def get_question(self, index: int) -> Dict:
        """Get question by index"""
        return self.questions[index]

    def get_answer(self, question: Dict) -> str:
        # GPQA returns the correct letter (A, B, C, or D)
        return question["correct_letter"]

    def get_prompt(self, question: Dict) -> str:
        """Get formatted prompt for the question"""
        return TEMPLATE_REGISTRY["gpqa"].format(
            Question=question["Question"],
            A=question["shuffled_answers"][0],
            B=question["shuffled_answers"][1],
            C=question["shuffled_answers"][2],
            D=question["shuffled_answers"][3]
        )

class Grader:
    def __init__(
        self,
        grader_type: str = "llm",
        grader_script: Optional[str] = None,
        judge_model_name: Optional[str] = None,
        judge_server_url: str = "",
        dataset_type: str = "aime"
    ):
        self.grader_type = grader_type
        self.grader_script = grader_script
        self.judge_model_name = judge_model_name
        self.judge_server_url = judge_server_url
        self.dataset_type = dataset_type
        self.pattern = self._get_pattern()

    def _get_pattern(self) -> Optional[str]:
        if self.grader_type == "regex":
            return GRADER_PATTERNS.get(self.dataset_type)  # Use dataset_type as key
        return None

    def _extract_answer_regex(self, pred: str) -> Optional[str]:
        """Extract answer using regex pattern"""
        if not self.pattern:
            return None

        # For AIME datasets, prioritize boxed answers
        if self.dataset_type in ["aime", "aime2025"]:
            boxed_pattern = r'\\boxed{([^}]+)}'
            boxed_matches = re.findall(boxed_pattern, pred, re.IGNORECASE)
            if boxed_matches:
                # Return the last boxed answer found (most likely the final answer)
                return boxed_matches[-1].strip()

        # For other datasets, search for numbers from the end of the text
        # This prioritizes numbers that appear later in the response
        matches = re.findall(self.pattern, pred, re.IGNORECASE)
        if not matches:
            return None

        # Process matches from end to start
        for match in reversed(matches):
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
        """Grade using LLM-based extraction with few-shot examples"""
        sample_answers = SAMPLE_ANSWERS.get(self.dataset_type, [])
        sample_examples = "\n".join([
            f"Example {i+1}: {ans}" for i, ans in enumerate(sample_answers)
        ])

        system_prompt = f"""You are an answer extraction system. Your task is to extract the answer from the model's response.

Here are some examples of extracted answers to demonstrate what you are supposed to output:

{sample_examples}

When extracting the answer, provide only the extracted answer itself, nothing else. If there is no clear answer that can be extracted from the response, reply with 'no answer'."""

        user_prompt = f"""Extract the answer from the following response:

"{pred}"

Please provide only the extracted answer, nothing else. If there is no clear answer that can be extracted from the response, reply with 'no answer'."""

        url = f"{self.judge_server_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        data = {
            "model": self.judge_model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0,
        }
        #print(json.dumps(data, indent=2))

        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            extracted = response.json()["choices"][0]["message"]["content"].strip()
            is_correct = extracted.strip().lower() == gold.strip().lower()
            return is_correct, extracted
        except Exception as e:
            return False, None

    def _truncate_response(self, response: str, max_lines: int = 6) -> str:
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
        grader: Grader,
        model_name: Optional[str] = None,
        threads: int = 32
    ):
        self.server_url = server_url
        self.grader = grader
        self.model_name = model_name
        self.threads = threads

    def _make_request(self, eval_state: EvalState, prompt: str) -> Dict[str, Any]:
        url = f"{self.server_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        data = {
            "model": self.model_name if self.model_name else "llama",
            "messages": [{"role": "user", "content": prompt}],
            "n_predict": eval_state.sampling_config.get("n_predict", -1)
        }
        if eval_state.sampling_config.get("temperature") is not None:
            data["temperature"] = eval_state.sampling_config["temperature"]
        if eval_state.sampling_config.get("top_k") is not None:
            data["top_k"] = eval_state.sampling_config["top_k"]
        if eval_state.sampling_config.get("top_p") is not None:
            data["top_p"] = eval_state.sampling_config["top_p"]
        if eval_state.sampling_config.get("min_p") is not None:
            data["min_p"] = eval_state.sampling_config["min_p"]

        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()

    def _process_single_case(self, eval_state: EvalState, i: int, task_id: str) -> TaskState:
        prompt, gold = eval_state.get_case(i)

        task_state = TaskState(
            case_id=task_id,
            prompt=prompt,
            gold=gold
        )

        try:
            response = self._make_request(eval_state, prompt)
            pred = response["choices"][0]["message"]["content"]
            task_state.pred = pred

            pred_truncated = self.grader._truncate_response(pred, max_lines=10)
            is_correct, extracted = self.grader.grade(gold, pred_truncated, prompt)

            grader_log = {
                "pred": pred_truncated,
                "grader_type": self.grader.grader_type
            }
            if self.grader.grader_type == "regex" and self.grader.pattern:
                grader_log["pattern"] = self.grader.pattern

            task_state.correct = is_correct
            task_state.extracted = extracted
            task_state.grader_log = grader_log
            task_state.status = "ok"

            eval_state.add_result(task_id, prompt, gold, pred, extracted, grader_log, is_correct, "ok")

            eval_state.dump()

        except Exception as e:
            task_state.status = f"error: {str(e)}"

        return task_state

    def evaluate(self, eval_state: EvalState, verbose: bool = False, resume: bool = False):
        total_tasks = len(eval_state.tasks)
        eval_state.total = len(eval_state.all_tasks) if eval_state.all_tasks else total_tasks
        eval_state.processed = 0

        print(f"\nProcessing {len(eval_state.tasks)} {eval_state.dataset_type.upper()} tasks ...")
        print(f"Server: {self.server_url} (model: {self.model_name})")
        print(f"Grader: {self.grader.grader_type}")
        print(f"Threads: {self.threads}")
        print(f"Sampling: temp={eval_state.sampling_config.get('temperature', 'skip')}, top-k={eval_state.sampling_config.get('top_k', 'skip')}, top-p={eval_state.sampling_config.get('top_p', 'skip')}, min-p={eval_state.sampling_config.get('min_p', 'skip')}")
        print()

        correct_count = 0

        with ThreadPoolExecutor(max_workers=self.threads) as executor:
            futures = {
                executor.submit(self._process_single_case, eval_state, i, task_id): (i, task_id)
                for i, task_id in eval_state.tasks
            }

            for future in as_completed(futures):
                task_state = future.result()
                eval_state.processed += 1
                if task_state.correct:
                    correct_count += 1
                eval_state.print_progress(task_state, total_tasks, correct_count)

                if verbose:
                    print(f"\nCase {eval_state.processed}: {task_state.correct}")
                    print(f"  Gold: {task_state.gold}")
                    if task_state.pred:
                        print(f"  Pred: {task_state.pred}")
                    if task_state.extracted:
                        print(f"  Extracted: {task_state.extracted}")
                    print(f"  Status: {task_state.status}")

        eval_state.print_summary()
        eval_state.dump()

def main():
    parser = argparse.ArgumentParser(
        description="Simplified evaluation tool for llama.cpp"
    )
    parser.add_argument(
        "--server",
        type=str,
        default="http://localhost:8033",
        help="llama-server URL (default: http://localhost:8033)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="aime",
        choices=["aime", "aime2025", "gsm8k", "gpqa"],
        help="Dataset type (default: aime)"
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
        default=-1,
        help="Max tokens to predict per prompt (default: -1, infinite)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (default: not passed)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top K sampling (default: not passed)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top P sampling (default: not passed)"
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=None,
        help="Min P sampling (default: not passed)"
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
        default="llm",
        choices=["regex", "cli", "llm"],
        help="Grader type: regex, cli, or llm (default: llm)"
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
        default="",
        help="Model name for LLM judge (default: same as main model)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing eval state"
    )

    args = parser.parse_args()

    if args.dataset == "gpqa" and args.grader_type != "llm":
        print("Error: GPQA dataset requires --grader-type llm")
        parser.print_help()
        sys.exit(1)

    if args.output.exists():
        print(f"Loading existing eval state from {args.output}")
        eval_state = EvalState.load(args.output)

        eval_state.print_all_tasks()
        eval_state.print_existing_summary()

        if eval_state.is_complete():
            return

        print()

        if not args.resume:
            print(f"Evaluation incomplete. Run with --resume to continue.")
            return

        pending_tasks = eval_state.get_pending_tasks()
        print(f"Resuming from {len(pending_tasks)} pending tasks")

        existing_cases = eval_state.task_states.get("cases", {})

        eval_state.tasks = pending_tasks
        eval_state.task_states["cases"] = existing_cases

        judge_server_url = args.judge_server if args.judge_server else args.server
        judge_model_name = args.judge_model if args.judge_model else args.model
        grader = Grader(
            grader_type=args.grader_type,
            grader_script=args.grader_script,
            judge_model_name=judge_model_name,
            judge_server_url=judge_server_url,
            dataset_type=eval_state.dataset_type
        )
        resume = True
    else:
        if args.resume:
            print("Error: No existing eval state found to resume")
            sys.exit(1)

        judge_server_url = args.judge_server if args.judge_server else args.server
        judge_model_name = args.judge_model if args.judge_model else args.model

        grader = Grader(
            grader_type=args.grader_type,
            grader_script=args.grader_script,
            judge_model_name=judge_model_name,
            judge_server_url=judge_server_url,
            dataset_type=args.dataset
        )

        if args.grader_type == "llm" and not args.judge_server:
            print("Warning: Using same server for LLM judge (no --judge-server specified)")

        sampling_config = {"n_predict": args.n_predict}
        if args.temperature is not None:
            sampling_config["temperature"] = args.temperature
        if args.top_k is not None:
            sampling_config["top_k"] = args.top_k
        if args.top_p is not None:
            sampling_config["top_p"] = args.top_p
        if args.min_p is not None:
            sampling_config["min_p"] = args.min_p

        eval_state = EvalState(
            dataset_type=args.dataset,
            sampling_config=sampling_config,
            output_file=args.output
        )
        eval_state.load_dataset(seed=args.seed)
        eval_state.setup_tasks(n_cases=args.n_cases, seed=args.seed)
        eval_state.dump()
        resume = False

        eval_state.print_all_tasks()

    processor = Processor(
        server_url=args.server,
        grader=grader,
        model_name=args.model,
        threads=args.threads
    )

    processor.evaluate(eval_state, verbose=args.verbose, resume=resume)
    print(f"\nEval state dumped to {args.output}")

if __name__ == "__main__":
    main()
