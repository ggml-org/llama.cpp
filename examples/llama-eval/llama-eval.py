#!/usr/bin/env python3
# type: ignore

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict, field
from pathlib import Path
from queue import Queue
from typing import Dict, List, Optional, Any, Tuple
import requests
from tqdm import tqdm
import random
from math import sqrt


@dataclass
class ServerConfig:
    url: str
    threads: int
    name: str = ""

def wilson_interval(correct: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score confidence interval for a proportion."""
    if total == 0:
        return (0.0, 1.0)
    p = correct / total
    z2 = z * z / total
    center = (p + z2 / 2) / (1 + z2)
    margin = z * sqrt((p * (1 - p) + z2 / 4) / total) / (1 + z2)
    return (center - margin, center + margin)

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
    "aime": """Solve the following math problem step by step. Put your answer inside \\boxed{{}}.

{question}

Remember to put your answer inside \\boxed{{}}.
""",
    "aime2025": """Solve the following math problem step by step. Put your answer inside \\boxed{{}}.

{question}

Remember to put your answer inside \\boxed{{}}.
""",
    "gsm8k": """{question}
Please reason step by step, and put your final numeric answer within \\boxed{{}} without any extra characters.
""",
    "gpqa": """Answer the following multiple choice question. The last line of your response should be in the following format: 'Answer: A/B/C/D' (e.g. 'Answer: A').

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""",
}


class BaseDataset(ABC):
    @abstractmethod
    def get_question(self, index: int) -> Dict:
        pass

    @abstractmethod
    def get_question_text(self, question: Dict) -> str:
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
    task_id: str
    prompt: str
    expected: str
    question_text: str = ""
    response: Optional[str] = None
    answer: Optional[str] = None
    grader_log: Dict[str, Any] = field(default_factory=dict)
    correct: bool = False
    status: str = "pending"
    tokens: Optional[int] = None
    tps_gen: Optional[float] = None
    t_gen_ms: Optional[float] = None
    reasoning_content: Optional[str] = None
    server_name: Optional[str] = None


class EvalState:
    def __init__(
        self,
        dataset_type: str,
        sampling_config: Dict[str, Any],
        output_file: Path = Path("llama-eval-state.json"),
        model_name: Optional[str] = None
    ):
        self.dataset_type = dataset_type
        self.sampling_config = sampling_config
        self.output_file = output_file
        self.model_name = model_name
        self.dataset: Optional[BaseDataset] = None
        self.tasks: List[Tuple[int, str]] = []
        self.all_tasks: List[Tuple[int, str]] = []
        self.task_states: Dict[str, Any] = {}
        self.total = 0
        self.correct = 0
        self.processed = 0
        self.total_time: float = 0.0
        self._lock = threading.Lock()

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

    def get_case(self, index: int) -> Tuple[str, str, str]:
        if self.dataset is None:
            raise ValueError("Dataset not loaded.")
        question = self.dataset.get_question(index)
        question_text = self.dataset.get_question_text(question)
        prompt = self.dataset.get_prompt(question)
        expected = self.dataset.get_answer(question)
        return question_text, prompt, expected

    def add_result(
        self,
        task_id: str,
        prompt: str,
        expected: str,
        response: Optional[str],
        answer: Optional[str],
        grader_log: Dict[str, Any],
        correct: bool,
        status: str,
        tokens: Optional[int] = None,
        tps_gen: Optional[float] = None,
        t_gen_ms: Optional[float] = None,
        reasoning_content: Optional[str] = None,
        server_name: Optional[str] = None
    ):
        with self._lock:
            if "cases" not in self.task_states:
                self.task_states["cases"] = {}

            self.task_states["cases"][task_id] = {
                "task_id": task_id,
                "prompt": prompt,
                "expected": expected,
                "response": response,
                "answer": answer,
                "grader_log": grader_log,
                "correct": correct,
                "status": status,
                "tokens": tokens,
                "tps_gen": tps_gen,
                "t_gen_ms": t_gen_ms,
                "reasoning_content": reasoning_content,
                "server_name": server_name
            }

            self.correct = sum(1 for c in self.task_states.get("cases", {}).values() if c.get("correct", False))

    def print_progress(self, task_state: TaskState, total_tasks: int, n_correct: int = 0):
        display_answer = task_state.answer if task_state.answer else "N/A"
        display_tokens = str(task_state.tokens) if task_state.tokens is not None else "N/A"
        display_tps = f"{task_state.tps_gen:.1f}" if task_state.tps_gen is not None else "N/A"
        display_t_gen = f"{task_state.t_gen_ms/1000:.1f}" if task_state.t_gen_ms is not None else "N/A"
        display_server = task_state.server_name if task_state.server_name else "N/A"
        success_ratio = n_correct / self.processed if self.processed > 0 else 0.0
        first_line = task_state.question_text.split('\n')[0]
        truncated_question = first_line[:43]
        if len(first_line) > 43:
            truncated_question += "..."
        else:
            truncated_question = truncated_question.ljust(43) + "..."
        print(f"{self.processed:3}/{total_tasks:3}  {task_state.task_id:<20} {self.dataset_type.upper()}   {truncated_question:<40}    {task_state.expected:<10} {display_answer:<10} {display_tokens:<6} {display_tps:<6} {display_t_gen:<8} {'✓' if task_state.correct else '✗'}  [{n_correct:3}/{self.processed:3}, {success_ratio:.3f}]  {display_server}")

    def print_summary(self):
        if self.total == 0:
            print(f"\n{'='*60}")
            print(f"Results: 0/0 correct (0.0%)")
            print(f"{'='*60}")
        else:
            ci_lower, ci_upper = self.accuracy_ci()
            print(f"\n{'='*60}")
            print(f"Results: {self.correct}/{self.total} correct ({self.correct/self.total*100:.1f}%) [{ci_lower*100:.1f}%, {ci_upper*100:.1f}%]")
            print(f"{'='*60}")

    def dump(self):
        with self._lock:
            tasks_to_save = self.all_tasks if self.all_tasks else self.tasks
            all_cases = {}
            for i, task_id in tasks_to_save:
                question_text, prompt, expected = self.get_case(i)
                if task_id in self.task_states.get("cases", {}):
                    all_cases[task_id] = self.task_states["cases"][task_id]
                else:
                    all_cases[task_id] = {
                        "task_id": task_id,
                        "prompt": prompt,
                        "expected": expected,
                        "question_text": question_text,
                        "response": None,
                        "answer": None,
                        "grader_log": {},
                        "correct": False,
                        "status": "pending",
                        "tokens": None,
                        "tps_gen": None,
                        "t_gen_ms": None,
                        "reasoning_content": None,
                        "server_name": None
                    }

            ci_lower, ci_upper = self.accuracy_ci()
            data = {
                "id": self.dataset_type,
                "model_name": self.model_name,
                "tasks": [tid for _, tid in tasks_to_save],
                "task_states": {
                    "total": self.total,
                    "correct": self.correct,
                    "total_time": self.total_time,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "cases": all_cases,
                },
                "sampling_config": self.sampling_config
            }
            with open(self.output_file, "w") as f:
                json.dump(data, f, indent=2)

            self.dump_html(tasks_to_save, all_cases)

    def dump_html(self, tasks_to_save: List[Tuple[int, str]], all_cases: Dict[str, Any]):
        html_file = Path(str(self.output_file) + ".html")

        cases = all_cases
        completed = {tid: c for tid, c in cases.items() if c.get("status") == "ok"}
        n_correct = sum(1 for c in completed.values() if c.get("correct", False))
        n_incorrect = len(completed) - n_correct
        n_pending = len(tasks_to_save) - len(completed)
        accuracy = n_correct / len(completed) * 100 if completed else 0.0
        ci_lower, ci_upper = wilson_interval(n_correct, len(completed)) if completed else (0.0, 1.0)

        sampling_parts = []
        for k, v in self.sampling_config.items():
            if v is not None:
                sampling_parts.append(f"{k}={v}")
        sampling_str = ", ".join(sampling_parts) if sampling_parts else "default"

        rows = []
        for i, task_id in tasks_to_save:
            case = cases.get(task_id, {})
            status = case.get("status", "pending")
            expected = case.get("expected", "")
            answer = case.get("answer", "") if status == "ok" else ""
            is_correct = case.get("correct", False) if status == "ok" else False
            response = case.get("response", "") or ""
            prompt = case.get("prompt", "") or ""
            grader_log = case.get("grader_log", {})

            if status == "ok":
                status_class = "correct" if is_correct else "incorrect"
                status_text = "✓" if is_correct else "✗"
            elif status == "pending":
                status_class = "pending"
                status_text = "–"
            else:
                status_class = "error"
                status_text = "!"

            tokens = case.get("tokens")
            tokens_str = str(tokens) if tokens is not None else ""
            tps_gen = case.get("tps_gen")
            tps_str = f"{tps_gen:.1f}" if tps_gen is not None else ""
            t_gen_ms = case.get("t_gen_ms")
            t_gen_str = f"{t_gen_ms/1000:.1f}" if t_gen_ms is not None else ""
            reasoning_content = case.get("reasoning_content", "") or ""
            server_name = case.get("server_name", "") or ""

            escaped_response = self._escape_html(response)
            escaped_prompt = self._escape_html(prompt)
            escaped_reasoning = self._escape_html(reasoning_content)
            grader_log_str = self._escape_html(json.dumps(grader_log, indent=2))
            escaped_server = self._escape_html(server_name)

            rows.append(f"""<tr class="task-row" onclick="toggleDetails('{task_id}')">
                <td>{task_id}</td>
                <td class="{status_class}">{status_text}</td>
                <td>{self._escape_html(expected)}</td>
                <td>{self._escape_html(answer)}</td>
                <td>{tokens_str}</td>
                <td>{tps_str}</td>
                <td>{t_gen_str}</td>
                <td>{escaped_server}</td>
            </tr>
            <tr id="details-{task_id}" class="details-row">
                <td colspan="8">
                    <div class="details-content">
                        <b>Prompt</b><pre>{escaped_prompt}</pre>
                        <b>Response</b><pre>{escaped_response}</pre>
                        {f'<b>Reasoning</b><pre>{escaped_reasoning}</pre>' if escaped_reasoning else ''}
                        <b>Grader</b><pre>{grader_log_str}</pre>
                    </div>
                </td>
            </tr>""")

        rows_html = "\n".join(rows)

        html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>{self.dataset_type.upper()} Eval</title>
<style>
        body {{ font-family: system-ui, sans-serif; margin: 0; padding: 16px; background: #fff; color: #222; }}
        .bar {{ padding: 8px 0; font-size: 14px; color: #555; }}
        .bar span {{ margin-right: 20px; }}
        .bar b {{ color: #222; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        th {{ text-align: left; padding: 6px 8px; border-bottom: 2px solid #ccc; font-weight: 600; }}
        td {{ padding: 4px 8px; border-bottom: 1px solid #eee; vertical-align: top; }}
        .task-row {{ cursor: pointer; }}
        .task-row:hover {{ background: #f5f5f5; }}
        .correct {{ color: #1a7f37; }}
        .incorrect {{ color: #cf222e; }}
        .pending {{ color: #888; }}
        .error {{ color: #9a6700; }}
        .details-row {{ display: none; }}
        .details-row.open {{ display: table-row; }}
        .details-content {{ padding: 8px 16px; background: #f6f8fa; font-size: 12px; }}
        .details-content b {{ color: #555; }}
        .details-content pre {{ background: #fff; border: 1px solid #e1e4e8; padding: 8px; overflow-x: auto; white-space: pre-wrap; word-wrap: break-word; margin: 4px 0 8px; }}
</style>
</head>
<body>
    <div class="bar">
        <span><b>{self.dataset_type.upper()}</b></span>
        <span>Model: {self.model_name or 'N/A'}</span>
        <span>Accuracy: <b>{accuracy:.1f}%</b> [{ci_lower*100:.1f}%, {ci_upper*100:.1f}%]</span>
        <span>Correct: <span class="correct">{n_correct}</span> / {len(completed)}</span>
        <span>Pending: {n_pending}</span>
        <span>Time: {self.total_time:.1f}s</span>
        <span>Sampling: {sampling_str}</span>
    </div>
    <table>
        <thead>
            <tr>
                <th>ID</th>
                <th></th>
                <th>Gold</th>
                <th>Answer</th>
                <th>Tokens</th>
                <th>T/s</th>
                <th>Gen s</th>
                <th>Server</th>
            </tr>
        </thead>
        <tbody>
            {rows_html}
        </tbody>
    </table>
    <script>
        function toggleDetails(id) {{ document.getElementById('details-'+id).classList.toggle('open'); }}
    </script>
</body>
</html>"""

        with open(html_file, "w") as f:
            f.write(html_content)

    def _escape_html(self, s: str) -> str:
        return (s.replace("&", "&amp;")
                   .replace("<", "&lt;")
                   .replace(">", "&gt;")
                   .replace('"', "&quot;")
                   .replace("'", "&#39;"))

    @classmethod
    def load(cls, path: Path) -> "EvalState":
        with open(path, "r") as f:
            data = json.load(f)

        eval_state = cls(
            dataset_type=data["id"],
            sampling_config=data["sampling_config"],
            output_file=path,
            model_name=data.get("model_name")
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
        eval_state.total_time = eval_state.task_states.get("total_time", 0.0)

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
            status = cases.get(task_id, {}).get("status", "pending")
            if status != "ok":
                pending.append((i, task_id))
        return pending

    def print_all_tasks(self):
        cases = self.task_states.get("cases", {})
        tasks_to_show = self.all_tasks if self.all_tasks else self.tasks
        print()
        print("Tasks:")
        print("  Task ID             Dataset  Prompt (first 40 chars)                        Expected    Answer       Tokens  T/s     Gen s  Status")
        for i, task_id in tasks_to_show:
            question, prompt, expected = self.get_case(i)
            case = cases.get(task_id, {})
            status = case.get("status", "pending")
            answer = case.get("answer", "N/A") if status == "ok" else "N/A"
            tokens = case.get("tokens")
            tokens_str = str(tokens) if tokens is not None else "N/A"
            tps_gen = case.get("tps_gen")
            tps_str = f"{tps_gen:.1f}" if tps_gen is not None else "N/A"
            t_gen_ms = case.get("t_gen_ms")
            t_gen_str = f"{t_gen_ms/1000:.1f}" if t_gen_ms is not None else "N/A"
            server_name = case.get("server_name", "") or ""
            is_correct = case.get("correct", False) if status == "ok" else False
            symbol = "✓ " if is_correct else ("✗ " if status == "ok" else "")
            first_line = question.split('\n')[0]
            question_trunc = first_line[:43]
            if len(first_line) > 43:
                question_trunc += "..."
            else:
                question_trunc = question_trunc.ljust(43) + "..."
            print(f"  {task_id:<20} {self.dataset_type.upper()}   {question_trunc:<40}    {expected:<10} {answer:<10} {tokens_str:<6} {tps_str:<6} {t_gen_str:<8} {symbol}{status}  {server_name}")
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
            ci_lower, ci_upper = self.accuracy_ci()
            print(f"{'='*60}")
            print(f"Results: {correct}/{total} correct ({correct/total*100:.1f}%) [{ci_lower*100:.1f}%, {ci_upper*100:.1f}%]")
            print(f"{'='*60}")

    def accuracy_ci(self) -> Tuple[float, float]:
        """Compute Wilson score confidence interval from completed cases."""
        cases = self.task_states.get("cases", {})
        completed = {tid: c for tid, c in cases.items() if c.get("status") == "ok"}
        correct = sum(1 for c in completed.values() if c.get("correct", False))
        total = len(completed)
        return wilson_interval(correct, total)

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

    def get_question_text(self, question: Dict) -> str:
        """Get question string"""
        return question["problem"] if "problem" in question else question["question"]

    def get_answer(self, question: Dict) -> str:
        answer = question["answer"]
        if isinstance(answer, str):
            normalized = normalize_number(answer)
            return str(normalized) if normalized is not None else answer
        return str(answer)

    def get_prompt(self, question: Dict) -> str:
        """Get formatted prompt for the question"""
        return TEMPLATE_REGISTRY[question["dataset_type"]].format(
            question=self.get_question_text(question),
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

    def get_question_text(self, question: Dict) -> str:
        """Get question string"""
        return question["question"]

    def get_answer(self, question: Dict) -> str:
        answer = question["answer"]
        if isinstance(answer, str):
            normalized = normalize_number(answer)
            return str(normalized) if normalized is not None else answer
        return str(answer)

    def get_prompt(self, question: Dict) -> str:
        """Get formatted prompt for the question"""
        return TEMPLATE_REGISTRY["aime2025"].format(
            question=self.get_question_text(question),
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

    def get_question_text(self, question: Dict) -> str:
        """Get question string"""
        return question["problem"] if "problem" in question else question["question"]

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
            question=self.get_question_text(question),
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

    def get_question_text(self, question: Dict) -> str:
        """Get question string"""
        return question["Question"]

    def get_answer(self, question: Dict) -> str:
        # GPQA returns the correct letter (A, B, C, or D)
        return question["correct_letter"]

    def get_prompt(self, question: Dict) -> str:
        """Get formatted prompt for the question"""
        return TEMPLATE_REGISTRY["gpqa"].format(
            Question=self.get_question_text(question),
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
        grader_model_name: Optional[str] = None,
        grader_server_url: str = "",
        dataset_type: str = "aime"
    ):
        self.grader_type = grader_type
        self.grader_script = grader_script
        self.grader_model_name = grader_model_name
        self.grader_server_url = grader_server_url
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
            answer = match.strip()
            if answer:
                return answer
        return None

    def _grade_regex(self, gold: str, pred: str) -> Tuple[bool, Optional[str]]:
        """Grade using regex pattern matching"""
        answer = self._extract_answer_regex(pred)
        if answer is None:
            return False, None
        is_correct = answer.strip() == gold.strip()
        return is_correct, answer

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
            answer = pred if is_correct else None
            return is_correct, answer
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

        url = f"{self.grader_server_url}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        data = {
            "model": self.grader_model_name,
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
            answer = response.json()["choices"][0]["message"]["content"].strip()
            is_correct = answer.strip().lower() == gold.strip().lower()
            return is_correct, answer
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
        server_configs: List[ServerConfig],
        grader: Grader,
        model_name: Optional[str] = None,
        n_predict: int = -1
    ):
        self.server_configs = server_configs
        self.grader = grader
        self.model_name = model_name
        self.n_predict = n_predict

    @staticmethod
    def _check_server(server_config: ServerConfig) -> List[str]:
        url = f"{server_config.url}/v1/models"
        try:
            response = requests.get(url)
            response.raise_for_status()
            models = [m["id"] for m in response.json().get("data", [])]
            return models
        except Exception as e:
            print(f"Error: Cannot reach server {server_config.name} ({server_config.url}): {e}", file=sys.stderr)
            sys.exit(1)

    def _make_request(
        self, server_config: ServerConfig, eval_state: EvalState, prompt: str
    ) -> Tuple[Dict[str, Any], int, Optional[float], Optional[float], str]:
        url = f"{server_config.url}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        data = {
            "model": self.model_name if self.model_name else "llama",
            "messages": [{"role": "user", "content": prompt}],
            "n_predict": self.n_predict
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
        result = response.json()
        tokens = result.get("usage", {}).get("completion_tokens", 0)
        timings = result.get("timings", {})
        tps_gen = timings.get("predicted_per_second") if timings else None
        t_gen_ms = timings.get("predicted_ms") if timings else None
        finish_reason = result.get("choices", [{}])[0].get("finish_reason", "stop")
        return result, tokens, tps_gen, t_gen_ms, finish_reason

    def _process_single_case(
        self, server_config: ServerConfig, eval_state: EvalState, i: int, task_id: str
    ) -> TaskState:
        question_text, prompt, expected = eval_state.get_case(i)

        task_state = TaskState(
            task_id=task_id,
            prompt=prompt,
            expected=expected,
            question_text=question_text,
            server_name=server_config.name
        )

        try:
            response, tokens, tps_gen, t_gen_ms, finish_reason = self._make_request(server_config, eval_state, prompt)
            result = response["choices"][0]["message"]["content"]
            reasoning_content = response["choices"][0].get("message", {}).get("reasoning_content")
            task_state.response = result
            task_state.tokens = tokens
            task_state.tps_gen = tps_gen
            task_state.t_gen_ms = t_gen_ms
            task_state.reasoning_content = reasoning_content

            if finish_reason != "stop":
                task_state.status = f"error: finish_reason={finish_reason}"
                eval_state.add_result(
                    task_id, prompt, expected, result, None,
                    {"finish_reason": finish_reason}, False, task_state.status,
                    tokens, tps_gen, t_gen_ms, reasoning_content, server_config.name
                )
                eval_state.dump()
                return task_state

            result_truncated = self.grader._truncate_response(result, max_lines=10)
            is_correct, answer = self.grader.grade(expected, result_truncated, prompt)

            grader_log = {
                "pred": result_truncated,
                "grader_type": self.grader.grader_type
            }
            if self.grader.grader_type == "regex" and self.grader.pattern:
                grader_log["pattern"] = self.grader.pattern

            task_state.correct = is_correct
            task_state.answer = answer
            task_state.grader_log = grader_log
            task_state.status = "ok"

            eval_state.add_result(
                task_id, prompt, expected, result, answer,
                grader_log, is_correct, "ok",
                tokens, tps_gen, t_gen_ms, reasoning_content, server_config.name
            )

            eval_state.dump()

        except Exception as e:
            task_state.status = f"error: {str(e)}"

        return task_state

    @staticmethod
    def _worker(
        server_config: ServerConfig,
        processor: "Processor",
        eval_state: EvalState,
        task_queue: Queue,
        results_queue: Queue,
    ):
        """Worker that pulls tasks from a shared queue and sends them to its server."""
        while True:
            task = task_queue.get()
            if task is None:  # sentinel
                task_queue.task_done()
                break
            try:
                i, task_id = task
                result = processor._process_single_case(server_config, eval_state, i, task_id)
                results_queue.put(result)
            finally:
                task_queue.task_done()

    def evaluate(self, eval_state: EvalState, verbose: bool = False, resume: bool = False):
        total_tasks = len(eval_state.tasks)
        eval_state.total = len(eval_state.all_tasks) if eval_state.all_tasks else total_tasks
        eval_state.processed = 0
        start_time = time.time()

        # Check servers and list models
        server_models = [self._check_server(sc) for sc in self.server_configs]

        # Print server info
        print(f"\nProcessing {len(eval_state.tasks)} {eval_state.dataset_type.upper()} tasks ...")
        print(f"Servers ({len(self.server_configs)}):")
        for i, sc in enumerate(self.server_configs):
            models_str = ", ".join(server_models[i]) if server_models[i] else "(none)"
            print(f"  {i+1}. {sc.name} — {sc.url} ({sc.threads} threads) [{models_str}]")
        print(f"Model: {self.model_name}")
        print(f"Grader: {self.grader.grader_type}")
        print(f"Sampling: temp={eval_state.sampling_config.get('temperature', 'skip')}, top-k={eval_state.sampling_config.get('top_k', 'skip')}, top-p={eval_state.sampling_config.get('top_p', 'skip')}, min-p={eval_state.sampling_config.get('min_p', 'skip')}")
        print()

        # Shared task queue: all workers compete for tasks
        task_queue: Queue = Queue()
        for i, task_id in eval_state.tasks:
            task_queue.put((i, task_id))

        # Results queue: workers push completed TaskStates here
        results_queue: Queue = Queue()

        # Total worker threads across all servers
        total_threads = sum(sc.threads for sc in self.server_configs)

        # Add one sentinel per worker so every worker exits cleanly
        for _ in range(total_threads):
            task_queue.put(None)

        # Launch workers: one ThreadPoolExecutor per server
        executors: List[ThreadPoolExecutor] = []
        worker_futures: List[Any] = []
        for server_config in self.server_configs:
            executor = ThreadPoolExecutor(max_workers=server_config.threads)
            executors.append(executor)
            for _ in range(server_config.threads):
                future = executor.submit(
                    self._worker, server_config, self, eval_state,
                    task_queue, results_queue
                )
                worker_futures.append(future)

        # Drain results as they complete
        n_correct = 0
        session_time = 0.0
        completed_count = 0

        while completed_count < total_tasks:
            task_state = results_queue.get()
            eval_state.processed += 1
            completed_count += 1
            if task_state.correct:
                n_correct += 1
            elapsed = time.time() - start_time
            eval_state.total_time += elapsed
            session_time += elapsed
            start_time = time.time()
            eval_state.print_progress(task_state, total_tasks, n_correct)

            if verbose:
                print(f"\nCase {eval_state.processed}: {task_state.correct}")
                print(f"  Expected: {task_state.expected}")
                if task_state.response:
                    print(f"  Response: {task_state.response}")
                if task_state.answer:
                    print(f"  Answer: {task_state.answer}")
                print(f"  Status: {task_state.status}")

        # Wait for all workers to finish and shut down executors
        for future in worker_futures:
            future.result()
        for executor in executors:
            executor.shutdown(wait=True)

        print(f"\nSession time: {session_time:.1f}s | Total accumulated time: {eval_state.total_time:.1f}s")
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
        help="Comma-separated llama-server URLs (default: http://localhost:8033)"
    )
    parser.add_argument(
        "--server-name",
        type=str,
        default="",
        help="Comma-separated display names for servers (default: use URLs)"
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
        type=str,
        default="32",
        help="Comma-separated thread counts per server (default: 32)"
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
        "--grader-server",
        type=str,
        default="",
        help="Server URL for LLM grader (default: same as main server)"
    )
    parser.add_argument(
        "--grader-model",
        type=str,
        default="",
        help="Model name for LLM grader (default: same as main model)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing eval state"
    )

    args = parser.parse_args()

    # Parse server URLs and thread counts
    server_urls = [u.strip() for u in args.server.split(",") if u.strip()]
    thread_counts = [int(t.strip()) for t in args.threads.split(",") if t.strip()]

    if len(server_urls) != len(thread_counts):
        print(f"Error: --server ({len(server_urls)} URLs) and --threads ({len(thread_counts)} values) must have the same count")
        sys.exit(1)

    # Parse server names (optional, defaults to URLs)
    if args.server_name:
        server_names = [n.strip() for n in args.server_name.split(",") if n.strip()]
        if len(server_names) != len(server_urls):
            print(f"Error: --server-name ({len(server_names)} names) and --server ({len(server_urls)} URLs) must have the same count")
            sys.exit(1)
    else:
        server_names = server_urls  # fallback to URLs

    server_configs = [
        ServerConfig(url=url, threads=threads, name=name)
        for url, threads, name in zip(server_urls, thread_counts, server_names)
    ]

    if args.dataset == "gpqa" and args.grader_type != "llm":
        print("Error: GPQA dataset requires --grader-type llm")
        parser.print_help()
        sys.exit(1)

    if args.output.exists():
        print(f"Loading existing eval state from {args.output}")
        eval_state = EvalState.load(args.output)

        # Verify model matches
        if eval_state.model_name is not None and args.model != eval_state.model_name:
            print(f"Error: Model mismatch. State has '{eval_state.model_name}', but --model is '{args.model}'")
            sys.exit(1)

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

        grader_server_url = args.grader_server if args.grader_server else server_configs[0].url
        grader_model_name = args.grader_model if args.grader_model else args.model
        if args.grader_type == "llm" and not grader_model_name:
            print("Error: --grader-type llm requires --grader-model or --model")
            sys.exit(1)
        grader = Grader(
            grader_type=args.grader_type,
            grader_script=args.grader_script,
            grader_model_name=grader_model_name,
            grader_server_url=grader_server_url,
            dataset_type=eval_state.dataset_type
        )
        resume = True
    else:
        if args.resume:
            print("Error: No existing eval state found to resume")
            sys.exit(1)

        grader_server_url = args.grader_server if args.grader_server else server_configs[0].url
        grader_model_name = args.grader_model if args.grader_model else args.model
        if args.grader_type == "llm" and not grader_model_name:
            print("Error: --grader-type llm requires --grader-model or --model")
            sys.exit(1)

        grader = Grader(
            grader_type=args.grader_type,
            grader_script=args.grader_script,
            grader_model_name=grader_model_name,
            grader_server_url=grader_server_url,
            dataset_type=args.dataset
        )

        if args.grader_type == "llm" and not args.grader_server:
            print("Warning: Using same server for LLM grader (no --grader-server specified)")

        sampling_config = {}
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
            output_file=args.output,
            model_name=args.model
        )
        eval_state.load_dataset(seed=args.seed)
        eval_state.setup_tasks(n_cases=args.n_cases, seed=args.seed)
        eval_state.dump()
        resume = False

        eval_state.print_all_tasks()

    processor = Processor(
        server_configs=server_configs,
        grader=grader,
        model_name=args.model,
        n_predict=args.n_predict
    )

    processor.evaluate(eval_state, verbose=args.verbose, resume=resume)
    print(f"\nEval state dumped to {args.output}")

if __name__ == "__main__":
    main()
