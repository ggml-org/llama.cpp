#!/usr/bin/env python3

import argparse
import ast
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

from eval_sandbox import Sandbox, resolve_python


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
    "aime2026": r'\boxed{(\d+)}|\b(\d+)\b',
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
    "aime2026": [
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
    "aime2026": """Solve the following math problem step by step. Put your answer inside \\boxed{{}}.

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
    "humaneval": """Complete the following Python function.

```python
{question}
```

Reply with the complete function in a single ```python code block, including the \
signature and any imports it needs. Do not include tests, example usage, or explanation.
""",
    "classeval": """Complete the following Python class.

```python
{question}
```

Reply with the complete class in a single ```python code block: keep the class name \
and every method signature exactly as given, implement every method body, and include \
any imports the class needs. Do not include tests, example usage, or explanation.
""",
}

# Suites graded by running the generated code instead of matching its text.
# The venv is provisioned from the dataset's own import inventory, so it only
# ever carries what the suite actually needs.
CODE_SUITES = {"humaneval", "classeval", "agentic"}

# Suites where the model drives its own multi-turn tool loop rather than
# answering a single prompt. The Processor hands the whole episode over.
AGENTIC_SUITES = {"agentic"}

# Import name -> pip name, for the cases where they differ. Anything not listed
# is assumed to install under the name it imports as.
PIP_NAMES = {
    "bs4": "beautifulsoup4",
    "PIL": "pillow",
    "docx": "python-docx",
    "fitz": "pymupdf",
    "sklearn": "scikit-learn",
    "yaml": "PyYAML",
    "cv2": "opencv-python-headless",
    "PyPDF2": "pypdf2",
    "Levenshtein": "levenshtein",
    "dateutil": "python-dateutil",
    "attr": "attrs",
    "OpenSSL": "pyOpenSSL",
    "serial": "pyserial",
    "Crypto": "pycryptodome",
}

# Overridden by --dataset-source; lets a suite be pointed at a patched copy
# (local .jsonl or a HuggingFace repo) without touching the loader.
DATASET_SOURCE: Optional[str] = None

# Extra construction options for the agentic suite, set from the CLI.
AGENTIC_OPTS: Dict[str, Any] = {}


class BaseDataset(ABC):
    questions: List[Dict]

    # -- agentic hooks ----------------------------------------------------
    # A suite that drives its own multi-turn tool conversation sets this, and
    # the Processor hands it the whole episode instead of making one request.

    is_agentic: bool = False

    def prepare(self, base_python: Optional[str] = None, force: bool = False) -> None:
        """Provision whatever the suite needs before any task runs.

        Single-shot suites are provisioned from required_packages() instead, so
        this is a no-op for them.
        """

    def run(self, index: int, chat) -> Dict[str, Any]:
        """Drive one episode to completion and return its scored result."""
        raise NotImplementedError(
            f"{type(self).__name__} does not drive its own conversation")

    # -- code-suite hooks -------------------------------------------------
    # Only consulted when the exec grader is active. They let a suite own how
    # its generated code is assembled and scored, which differs a lot: one
    # HumanEval problem is a single function judged by one assert, whereas one
    # ClassEval task is a whole class judged by many independent test methods.

    def required_packages(self) -> List[str]:
        """Third-party pip packages the suite's tests need."""
        return []

    def provision_steps(self) -> List[str]:
        """Python snippets run once at venv build time, with network access."""
        return []

    def preferred_python(self) -> Optional[str]:
        """Interpreter the suite's dependencies need, if not the current one."""
        return None

    def configure_python(self, python: str):
        """Told which interpreter the sandbox will use, before it is built.

        Lets a suite adapt its package set to what that version can actually
        install.
        """

    def default_exec_timeout(self) -> float:
        """Per-task wall-clock limit that suits this suite's workload."""
        return 15.0

    def sandbox_env(self) -> Dict[str, str]:
        """Extra env for every run; '{venv}' expands to the venv path."""
        return {}

    def extract(self, response: str, question: Dict) -> str:
        """Pull the candidate code out of a chat reply."""
        return extract_code(response)

    def build_program(self, question: Dict, completion: str,
                      seed: Optional[int] = None) -> str:
        """Assemble the program whose exit status decides pass/fail."""
        raise NotImplementedError

    def interpret(self, result: Any, question: Dict) -> Tuple[bool, Dict[str, Any]]:
        """Turn an ExecResult into (passed, detail-for-the-log)."""
        return result.passed, {
            "status": result.status,
            "reason": result.summary(),
            "duration": round(result.duration, 3),
        }

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
    chunk_idx: int = 0
    problem_idx: int = 0


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

    @property
    def loaded_dataset(self) -> BaseDataset:
        """The dataset, once loaded. Raises rather than handing back None."""
        if self.dataset is None:
            raise RuntimeError("dataset not loaded; call load_dataset() first")
        return self.dataset

    def load_dataset(self, seed: int = 1234):
        if self.dataset_type == "aime":
            self.dataset = AimeDataset()
        elif self.dataset_type == "aime2025":
            self.dataset = Aime2025Dataset()
        elif self.dataset_type == "aime2026":
            self.dataset = Aime2026Dataset()
        elif self.dataset_type == "gsm8k":
            self.dataset = Gsm8kDataset()
        elif self.dataset_type == "gpqa":
            self.dataset = GpqaDataset(variant="diamond", seed=seed)
        elif self.dataset_type == "humaneval":
            self.dataset = HumanEvalDataset(source=DATASET_SOURCE)
        elif self.dataset_type == "classeval":
            self.dataset = ClassEvalDataset(source=DATASET_SOURCE)
        elif self.dataset_type == "agentic":
            self.dataset = AgenticDataset(source=DATASET_SOURCE, **AGENTIC_OPTS)
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
        server_name: Optional[str] = None,
        chunk_idx: int = 0,
        problem_idx: int = 0,
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
                "server_name": server_name,
                "chunk_idx": chunk_idx,
                "problem_idx": problem_idx,
            }

            self.correct = sum(1 for c in self.task_states.get("cases", {}).values() if c.get("correct", False))

    def _display_pair(self, task_state: TaskState) -> Tuple[str, str]:
        """The (expected, answer) columns, kept narrow for code suites.

        There the gold value is a whole JSON test harness and the answer is a
        whole function, so show the task id and the failure cause instead.
        """
        if self.dataset_type not in CODE_SUITES:
            return (
                task_state.expected,
                task_state.answer if task_state.answer else "N/A",
            )
        try:
            expected = json.loads(task_state.expected)["task_id"]
        except (ValueError, KeyError, TypeError):
            expected = self.dataset_type
        reason = (task_state.grader_log or {}).get("reason", "")
        if task_state.correct:
            reason = "ok"
        return expected, (reason[:28] or "N/A")

    def print_progress(self, task_state: TaskState, total_tasks: int, n_correct: int = 0):
        display_expected, display_answer = self._display_pair(task_state)
        display_tokens = str(task_state.tokens) if task_state.tokens is not None else "N/A"
        display_tps = f"{task_state.tps_gen:.1f}" if task_state.tps_gen is not None else "N/A"
        display_t_gen = f"{task_state.t_gen_ms/1000:.1f}" if task_state.t_gen_ms is not None else "N/A"
        display_server = task_state.server_name if task_state.server_name else "N/A"
        success_ratio = n_correct / self.processed if self.processed > 0 else 0.0
        first_line = next(
            (ln for ln in task_state.question_text.split('\n') if ln.strip()), ""
        )
        truncated_question = first_line[:43]
        if len(first_line) > 43:
            truncated_question += "..."
        else:
            truncated_question = truncated_question.ljust(43) + "..."
        print(f"{self.processed:3}/{total_tasks:3}  {task_state.task_id:<20} {self.dataset_type.upper()}   {truncated_question:<40}    {display_expected:<16} {display_answer:<28} {display_tokens:<6} {display_tps:<6} {display_t_gen:<8} {'✓' if task_state.correct else '✗'}  [{n_correct:3}/{self.processed:3}, {success_ratio:.3f}]  {display_server}")

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
                # Extract chunk_idx from task_id for pending cases
                _parts = task_id.rsplit("_", 2)
                _chunk_idx = int(_parts[-2]) if len(_parts) >= 3 else 0
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
                        "server_name": None,
                        "chunk_idx": _chunk_idx,
                        "problem_idx": i,
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
            answer = case.get("answer") or "" if status == "ok" else ""
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

            answer_class = status_class if status == "ok" else ""
            rows.append(f"""<tr class="task-row" onclick="toggleDetails('{task_id}')">
                <td>{task_id}</td>
                <td class="{status_class}">{status_text}</td>
                <td>{self._escape_html(expected)}</td>
                <td class="{answer_class}">{self._escape_html(answer)}</td>
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

        # ---- per-problem summary table ----
        problem_groups: Dict[int, List[Dict[str, Any]]] = {}
        for _tid, _case in cases.items():
            if _case.get("status") != "ok":
                continue
            _pidx = _case.get("problem_idx")
            if _pidx is None:
                _p_parts = _tid.rsplit("_", 2)
                _pidx = int(_p_parts[-1]) if len(_p_parts) >= 3 else 0
            problem_groups.setdefault(_pidx, []).append(_case)

        summary_rows_html = ""
        if problem_groups:
            def _stat(v, fmt=".1f", avg_fmt=None):
                if not v:
                    return ("–", "–", "–")
                af = fmt if avg_fmt is None else avg_fmt
                return (f"{min(v):{fmt}}", f"{sum(v)/len(v):{af}}", f"{max(v):{fmt}}")

            summary_data = []
            for pidx, g in problem_groups.items():
                runs = len(g)
                n_ok = sum(1 for c in g if c.get("correct", False))
                toks = [c["tokens"] for c in g if c.get("tokens") is not None]
                tps = [c["tps_gen"] for c in g if c.get("tps_gen") is not None]
                tg = [c["t_gen_ms"] / 1000 for c in g if c.get("t_gen_ms") is not None]
                summary_data.append((
                    pidx, runs, n_ok,
                    _stat(toks, "d", ".0f"),
                    _stat(tps),
                    _stat(tg),
                ))

            summary_data.sort(key=lambda r: r[0])  # sort by problem index ascending

            summary_rows_html = "\n".join(
                f"""<tr class="summary-row">
                    <td>{p:03d}</td>
                    <td>{r}</td>
                    <td>{n}/{r}</td>
                    <td>{tk[0]}</td><td>{tk[1]}</td><td>{tk[2]}</td>
                    <td>{tp[0]}</td><td>{tp[1]}</td><td>{tp[2]}</td>
                    <td>{tg[0]}</td><td>{tg[1]}</td><td>{tg[2]}</td>
                </tr>"""
                for p, r, n, tk, tp, tg in summary_data
            )

        html_content = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>{self.dataset_type.upper()} Eval</title>
<style>
        body {{ font-family: system-ui, sans-serif; margin: 0; padding: 16px; background: #fff; color: #222; }}
        .bar {{ padding: 8px 0; font-size: 13px; color: #555; font-family: 'SF Mono', 'Menlo', 'Consolas', monospace; display: grid; grid-template-columns: auto 1fr auto 1fr; gap: 2px 12px; align-items: baseline; }}
        .bar .label {{ color: #888; }}
        .bar .value {{ color: #222; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 13px; font-family: 'SF Mono', 'Menlo', 'Consolas', monospace; }}
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
        .summary-table {{ margin-bottom: 16px; font-size: 13px; width: 100%; }}
        .summary-row {{ background: #fafbfc; }}
        .summary-row:hover {{ background: #f5f5f5; }}
        .summary-table th {{ text-align: right; font-weight: 600; }}
        .summary-table th:first-child {{ text-align: left; }}
        .summary-table th[colspan] {{ text-align: center; }}
        .summary-table td {{ text-align: right; }}
        .summary-table td:first-child {{ text-align: left; }}
        .tabs {{ display: flex; border-bottom: 2px solid #ddd; margin: 12px 0 0; }}
        .tab-btn {{ padding: 6px 16px; border: none; background: none; font-size: 13px; cursor: pointer; color: #555; border-bottom: 2px solid transparent; margin-bottom: -2px; font-weight: 500; }}
        .tab-btn:hover {{ color: #222; }}
        .tab-btn.active {{ color: #222; border-bottom-color: #222; font-weight: 600; }}
        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}
</style>
</head>
<body>
    <div class="bar">
        <div class="label">Dataset</div><div class="value"><b>{self.dataset_type.upper()}</b></div>
        <div class="label">Model</div><div class="value"><b>{self.model_name or 'N/A'}</b></div>
        <div class="label">Accuracy</div><div class="value"><b>{accuracy:.1f}%</b> [{ci_lower*100:.1f}%, {ci_upper*100:.1f}%]</div>
        <div class="label">Correct</div><div class="value"><span class="correct">{n_correct}</span> / {len(completed)}</div>
        <div class="label">Pending</div><div class="value">{n_pending}</div>
        <div class="label">Time</div><div class="value">{self.total_time:.1f}s</div>
        <div class="label">Sampling</div><div class="value">{sampling_str}</div>
    </div>
    <div class="tabs">
        <button class="tab-btn active" data-tab="detailed" onclick="switchTab(this)">Detailed</button>
        <button class="tab-btn" data-tab="summary" onclick="switchTab(this)">Summary</button>
    </div>
    <div id="tab-detailed" class="tab-content active">
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
    </div>
    <div id="tab-summary" class="tab-content">
        <table class="summary-table">
            <thead>
                <tr>
                    <th>Problem</th>
                    <th>Runs</th>
                    <th>Correct</th>
                    <th colspan="3">Tokens</th>
                    <th colspan="3">T/s</th>
                    <th colspan="3">Gen s</th>
                </tr>
                <tr>
                    <th></th>
                    <th></th>
                    <th></th>
                    <th>min</th><th>avg</th><th>max</th>
                    <th>min</th><th>avg</th><th>max</th>
                    <th>min</th><th>avg</th><th>max</th>
                </tr>
            </thead>
            <tbody>
                {summary_rows_html}
            </tbody>
        </table>
    </div>
    <script>
        function toggleDetails(id) {{ document.getElementById('details-'+id).classList.toggle('open'); }}
        function switchTab(btn) {{
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            btn.classList.add('active');
            document.getElementById('tab-'+btn.dataset.tab).classList.add('active');
        }}
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
            answer = case.get("answer") or "N/A" if status == "ok" else "N/A"
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
        self.questions = []
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
        self.questions = []
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

class Aime2026Dataset(BaseDataset):
    def __init__(self):
        self.questions = []
        self._load_dataset()

    def _load_dataset(self):
        print(f"Loading AIME2026 dataset...")
        from datasets import load_dataset

        cache_path = cache_dir / "MathArena___aime_2026" / "default" / "0.0.0"
        if cache_path.exists():
            print(f"Using cached dataset from {cache_path}")
            ds = load_dataset("MathArena/aime_2026", "default", split="train", cache_dir=str(cache_path))
        else:
            ds = load_dataset("MathArena/aime_2026", "default", split="train")

        self.questions = []
        for row in ds:
            question = dict(row)
            question["dataset_type"] = "aime2026"
            self.questions.append(question)

        print(f"AIME2026 dataset loaded: {len(self.questions)} questions")

    def get_question(self, index: int) -> Dict:
        """Get question by index"""
        return self.questions[index]

    def get_question_text(self, question: Dict) -> str:
        """Get question string"""
        return question["problem"]

    def get_answer(self, question: Dict) -> str:
        return str(question["answer"])

    def get_prompt(self, question: Dict) -> str:
        """Get formatted prompt for the question"""
        return TEMPLATE_REGISTRY["aime2026"].format(
            question=self.get_question_text(question),
        )

class Gsm8kDataset(BaseDataset):
    def __init__(self, split: str = "test"):
        self.split = split
        self.questions = []
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
        self.questions = []
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

FENCE_RE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\n(.*?)(?:```|\Z)", re.DOTALL)

RESULT_MARKER = "__LLAMA_EVAL__"

# gensim publishes no wheel past cp313 and its Cython-generated C no longer
# compiles (it still uses PyLongObject.ob_digit, removed in 3.12). ClassEval
# touches exactly two gensim APIs, both pure Python upstream, so on 3.14+ they
# are provided directly rather than losing the two tasks that need them.
GENSIM_SHIM = {
    "__init__.py": (
        '"""Minimal stand-in for the two gensim entry points ClassEval uses."""\n'
        "from . import utils, matutils  # noqa: F401\n"
    ),
    "utils.py": (
        "import re, html\n"
        "RE_ENTITY = re.compile(r'&(#?)([xX]?)(\\w{1,8});')\n"
        "def decode_htmlentities(text):\n"
        "    return RE_ENTITY.sub(lambda m: html.unescape(m.group(0)), text)\n"
    ),
    "matutils.py": (
        "import numpy as np\n"
        "def unitvec(vec, norm='l2', return_norm=False):\n"
        "    v = np.asarray(vec, dtype=float)\n"
        "    n = float(np.sum(np.abs(v))) if norm == 'l1' "
        "else float(np.sqrt(np.sum(v ** 2)))\n"
        "    if n == 0.0:\n"
        "        return (v, 1.0) if return_norm else v\n"
        "    out = v / n\n"
        "    return (out, n) if return_norm else out\n"
    ),
}

# Runs before the suite's test code. Some tasks end with the usual
# `if __name__ == "__main__": unittest.main()`, and unittest.main() exits the
# process -- which would kill the reporting harness appended after it.
UNITTEST_PROLOGUE = '''
import unittest as _ut_pre
_ut_pre.main = lambda *a, **k: None
'''

# Appended to suites whose tasks are judged by many independent unit tests, so
# a partially-correct answer can be scored as such instead of just pass/fail.
# Discovers TestCase subclasses from the module namespace rather than trusting a
# dataset field to name them.
UNITTEST_RUNNER = f'''

if True:
    import json as _json, os as _os, unittest as _ut
    _loader = _ut.TestLoader()
    _suite = _ut.TestSuite()
    for _c in [v for v in list(globals().values())
               if isinstance(v, type) and issubclass(v, _ut.TestCase)
               and v is not _ut.TestCase]:
        _suite.addTests(_loader.loadTestsFromTestCase(_c))
    with open(_os.devnull, "w") as _null:
        _res = _ut.TextTestRunner(stream=_null, verbosity=0).run(_suite)
    _bad = {{t.id() for t, _ in _res.failures}} | {{t.id() for t, _ in _res.errors}}
    _total = _res.testsRun
    print("{RESULT_MARKER}" + _json.dumps({{
        "total": _total,
        "passed": _total - len(_bad),
        "failed": sorted(_bad),
        "first_error": (_res.failures + _res.errors)[0][1].strip().splitlines()[-1]
                       if (_res.failures or _res.errors) else None,
    }}))
'''


def parse_test_report(stdout: str) -> Optional[Dict[str, Any]]:
    """Read the JSON line emitted by UNITTEST_RUNNER, if the program got that far."""
    for line in reversed(stdout.splitlines()):
        if line.startswith(RESULT_MARKER):
            try:
                return json.loads(line[len(RESULT_MARKER):])
            except ValueError:
                return None
    return None


def extract_code(response: str, entry_point: Optional[str] = None,
                 class_name: Optional[str] = None) -> str:
    """Pull the candidate program out of a chat reply.

    Prefers a fenced block that actually defines the thing being asked for,
    since models like to emit a usage example or a test in a second block.
    """
    if not response:
        return ""

    if class_name:
        wanted = rf"^\s*class\s+{re.escape(class_name)}\s*[:\(]"
    elif entry_point:
        wanted = rf"^\s*(?:async\s+)?def\s+{re.escape(entry_point)}\s*\("
    else:
        wanted = None

    blocks = [(lang.lower(), body) for lang, body in FENCE_RE.findall(response)]
    if blocks:
        py = [b for lang, b in blocks if lang in ("python", "py", "python3", "")]
        candidates = py or [b for _, b in blocks]
        if wanted:
            defining = [b for b in candidates if re.search(wanted, b, re.M)]
            if defining:
                return defining[-1].strip("\n")
        return candidates[0].strip("\n")

    # No fence at all: some models just emit bare code.
    return response.strip("\n")


def _import_preamble(prompt: str) -> str:
    """The import lines from the original stub.

    A chat model often returns just the function and drops the imports that the
    stub had above it; replaying them keeps that from being scored as a failure.
    """
    lines = []
    for line in prompt.splitlines():
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")) and not line.startswith((" ", "\t")):
            lines.append(stripped)
    return "\n".join(lines)


def build_program(question: Dict, completion: str, seed: Optional[int] = 0) -> str:
    """Assemble the full program that decides pass/fail for one problem."""
    entry_point = question["entry_point"]
    code = completion

    # If the model returned only a body (no signature), graft it onto the stub.
    if not re.search(rf"^\s*(?:async\s+)?def\s+{re.escape(entry_point)}\s*\(", code, re.M):
        body = completion if completion.startswith((" ", "\t")) else \
            "\n".join("    " + ln for ln in completion.splitlines())
        code = question["prompt"] + "\n" + body

    # HumanEval/38, /50 and /53 draw their test inputs from the *global* unseeded
    # RNG. A correct solution always passes, but a subtly wrong one can flip
    # between runs -- which makes A/B comparisons of two models noisy. Seeding
    # here fixes the inputs without changing what the model is asked to compute.
    prologue = f"import random; random.seed({seed})" if seed is not None else ""

    return "\n\n".join(part for part in [
        _import_preamble(question["prompt"]),
        code,
        question["test"],
        prologue,
        f"check({entry_point})",
    ] if part)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator from the Codex paper (arXiv:2107.03374).

    n samples drawn, c of them correct. Computed as 1 - C(n-c,k)/C(n,k) via a
    product so it stays stable for large n.
    """
    if n - c < k:
        return 1.0
    prod = 1.0
    for i in range(n - c + 1, n + 1):
        prod *= 1.0 - k / i
    return 1.0 - prod


class HumanEvalDataset(BaseDataset):
    """OpenAI HumanEval, 164 hand-written Python problems.

    Every problem in this suite imports only from the standard library, so the
    execution venv needs no pip installs at all -- see `required_packages`.
    """

    HF_REPO = "openai/openai_humaneval"

    def __init__(self, source: Optional[str] = None, split: str = "test"):
        self.source = source or self.HF_REPO
        self.split = split
        self.questions: List[Dict] = []
        self._load_dataset()

    def _load_dataset(self):
        print(f"Loading HumanEval dataset (source: {self.source})...")

        local = Path(self.source)
        if local.exists():
            rows = [json.loads(ln) for ln in local.read_text().splitlines() if ln.strip()]
        else:
            from datasets import load_dataset
            rows = [dict(r) for r in load_dataset(self.source, split=self.split)]

        required = {"task_id", "prompt", "entry_point", "test"}
        for row in rows:
            missing = required - row.keys()
            if missing:
                raise ValueError(
                    f"{self.source}: record {row.get('task_id', '?')} is missing {sorted(missing)}"
                )
            row["dataset_type"] = "humaneval"
            self.questions.append(row)

        print(f"HumanEval dataset loaded: {len(self.questions)} problems")

    def required_packages(self) -> List[str]:
        """Third-party pip packages this suite needs. Derived, not hardcoded."""
        return sorted(third_party_imports(
            "\n".join(q["prompt"] + "\n" + q["test"] for q in self.questions)
        ))

    def extract(self, response: str, question: Dict) -> str:
        return extract_code(response, question.get("entry_point"))

    def build_program(self, question: Dict, completion: str,
                      seed: Optional[int] = None) -> str:
        return build_program(question, completion, seed=seed)

    def get_question(self, index: int) -> Dict:
        return self.questions[index]

    def get_question_text(self, question: Dict) -> str:
        return question["prompt"]

    def get_answer(self, question: Dict) -> str:
        # The "gold answer" for a code suite is the harness that judges it.
        # Carried as JSON so it survives the state file and --resume.
        return json.dumps({
            "task_id": question["task_id"],
            "entry_point": question["entry_point"],
            "prompt": question["prompt"],
            "test": question["test"],
        })

    def get_prompt(self, question: Dict) -> str:
        return TEMPLATE_REGISTRY["humaneval"].format(
            question=self.get_question_text(question).rstrip(),
        )


class ClassEvalDataset(BaseDataset):
    """ClassEval: 100 class-level Python generation tasks.

    A task is a whole class judged by several independent unittest classes, so
    it is scored twice -- class level (every test passes) and method level
    (what fraction of tests passed), matching the upstream protocol.
    """

    HF_REPO = "ilintar/ClassEval"

    def __init__(self, source: Optional[str] = None):
        self.source = source or self.HF_REPO
        self.questions: List[Dict] = []
        self._gensim_shim = False
        self._load_dataset()

    def _load_dataset(self):
        print(f"Loading ClassEval dataset (source: {self.source})...")

        local = Path(self.source)
        if local.exists():
            text = local.read_text()
            rows = (json.loads(text) if local.suffix == ".json"
                    else [json.loads(ln) for ln in text.splitlines() if ln.strip()])
        else:
            from datasets import load_dataset
            rows = [dict(r) for r in load_dataset(self.source, split="test")]

        required = {"task_id", "skeleton", "test", "class_name"}
        for row in rows:
            missing = required - row.keys()
            if missing:
                raise ValueError(
                    f"{self.source}: record {row.get('task_id', '?')} is missing {sorted(missing)}"
                )
            row["dataset_type"] = "classeval"
            # Denominator for the method-level score when the program dies
            # before the runner can report (syntax error, bad import, timeout).
            row["n_tests"] = _count_test_methods(row["test"])
            self.questions.append(row)

        print(f"ClassEval dataset loaded: {len(self.questions)} classes, "
              f"{sum(q['n_tests'] for q in self.questions)} test methods")

    # Dependencies that no import statement mentions, so scanning the source
    # cannot find them. They must be declared or the suite silently loses tests.
    EXTRA_PACKAGES = [
        # ClassEval_44 selects the parser by *string*: BeautifulSoup(html,
        # 'lxml'). Without it bs4 raises FeatureNotFound and that task drops
        # from 23/23 to 7/23. It happens to arrive via python-docx today, but
        # relying on someone else's transitive dependency is not a plan.
        "lxml",
        # Hard requirement of gensim, listed explicitly for the same reason.
        "scipy",
    ]

    def configure_python(self, python: str):
        """gensim stops at cp313, so on 3.14+ swap it for a small local shim."""
        try:
            r = subprocess.run(
                [python, "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
                capture_output=True, text=True, timeout=60,
            )
            major, minor = (int(x) for x in r.stdout.strip().split("."))
        except Exception:
            return
        self._gensim_shim = (major, minor) >= (3, 14)

    def required_packages(self) -> List[str]:
        blob = "\n\n".join(
            "\n\n".join(_as_text(q.get(f)) for f in
                        ("skeleton", "test", "solution_code", "import_statement"))
            for q in self.questions
        )
        found = {PIP_NAMES.get(m, m) for m in third_party_imports(blob)}
        found |= set(self.EXTRA_PACKAGES)
        if self._gensim_shim:
            found.discard("gensim")
        return sorted(found)

    # Exactly what the suite's nltk tasks need, ~33 MB. These are the current
    # names: the dataset asks for 'punkt' and 'averaged_perceptron_tagger',
    # which nltk has since renamed. Seeding the legacy names as well is not
    # harmless -- with a network available nltk treats them as satisfied and
    # then re-fetches the modern ones anyway, tripling the download.
    NLTK_CORPORA = ("punkt_tab", "averaged_perceptron_tagger_eng", "wordnet")

    def provision_steps(self) -> List[str]:
        steps = []

        if any("nltk" in _as_text(q.get("skeleton")) + _as_text(q.get("test"))
               for q in self.questions):
            # Pre-seed the corpora so test time never needs the network.
            steps.append(
                "import os, nltk\n"
                "target = os.path.join(os.environ['SANDBOX_VENV'], 'nltk_data')\n"
                "os.makedirs(target, exist_ok=True)\n"
                f"for corpus in {self.NLTK_CORPORA!r}:\n"
                "    nltk.download(corpus, download_dir=target, quiet=True)\n"
            )

        if self._gensim_shim:
            steps.append(
                "import os, sysconfig\n"
                "pkg = os.path.join(sysconfig.get_paths()['purelib'], 'gensim')\n"
                "os.makedirs(pkg, exist_ok=True)\n"
                f"files = {GENSIM_SHIM!r}\n"
                "for name, body in files.items():\n"
                "    open(os.path.join(pkg, name), 'w').write(body)\n"
                "import gensim.utils, gensim.matutils\n"  # fail loudly if broken
            )

        return steps

    def sandbox_env(self) -> Dict[str, str]:
        # $HOME is a tmpfs under the strongest isolation tier, so the corpora
        # have to be found inside the venv, which stays mounted.
        return {"NLTK_DATA": "{venv}/nltk_data"}

    def preferred_python(self) -> Optional[str]:
        # gensim has no wheel for 3.14 and its generated C no longer compiles
        # there (it still uses PyLongObject.ob_digit). 3.13 is the newest
        # interpreter that takes the whole dependency set.
        return "3.13"

    def default_exec_timeout(self) -> float:
        # These tasks do real work -- the slowest reference solution takes ~6 s
        # (a BFS puzzle solver), and importing pandas or loading an nltk corpus
        # costs seconds on its own.
        return 30.0

    def extract(self, response: str, question: Dict) -> str:
        return extract_code(response, class_name=question.get("class_name"))

    def build_program(self, question: Dict, completion: str,
                      seed: Optional[int] = None) -> str:
        prologue = f"import random; random.seed({seed})" if seed is not None else ""
        return "\n\n".join(part for part in [
            _as_text(question.get("import_statement")),
            UNITTEST_PROLOGUE,
            completion,
            question["test"],
            prologue,
            UNITTEST_RUNNER,
        ] if part)

    def interpret(self, result: Any, question: Dict) -> Tuple[bool, Dict[str, Any]]:
        # The denominator is what the task *should* run, taken from the dataset.
        # Using the observed count would let a class that fails to even import
        # score 0/0 and quietly shrink the total, inflating the method rate.
        expected = question.get("n_tests", 0)

        report = parse_test_report(result.stdout)
        if report is None:
            # never reached the runner -- syntax error, bad import or timeout
            return False, {
                "status": result.status if result.status != "ok" else "no-report",
                "reason": result.summary(),
                "tests_total": expected,
                "tests_passed": 0,
                "duration": round(result.duration, 3),
            }

        total, passed = report.get("total", 0) or expected, report.get("passed", 0)
        ok = total > 0 and passed == total
        return ok, {
            "status": "ok" if ok else "failed",
            "reason": "ok" if ok else (report.get("first_error")
                                       or f"{passed}/{total} tests passed")[:200],
            "tests_total": total,
            "tests_passed": passed,
            "duration": round(result.duration, 3),
        }

    def get_question(self, index: int) -> Dict:
        return self.questions[index]

    def get_question_text(self, question: Dict) -> str:
        return question["skeleton"]

    def get_answer(self, question: Dict) -> str:
        return json.dumps({
            "task_id": question["task_id"],
            "class_name": question["class_name"],
            "import_statement": _as_text(question.get("import_statement")),
            "test": question["test"],
            "n_tests": question["n_tests"],
        })

    def get_prompt(self, question: Dict) -> str:
        return TEMPLATE_REGISTRY["classeval"].format(
            question=self.get_question_text(question).rstrip(),
        )


class AgenticDataset(BaseDataset):
    """Multi-turn tool-driven repair tasks over a small real repository.

    The other code suites answer one prompt. Here the model is handed a tool
    set and drives its own conversation, so the Processor delegates the whole
    episode rather than making a single request.

    Everything language-specific is imported lazily and provisioned per
    language, so selecting the Python tasks never installs a node toolchain --
    and running any other suite installs none of it.
    """

    HF_REPO = "ilintar/SACB"
    is_agentic = True

    def __init__(self, source: Optional[str] = None, lang: Optional[str] = None,
                 max_turns: int = 50, context_limit: Optional[int] = None,
                 max_tokens: int = 2048):
        self.source = source or self.HF_REPO
        self.lang_filter = lang
        self.max_turns = max_turns
        self.context_limit = context_limit
        self.max_tokens = max_tokens
        self._sandboxes: Dict[str, Any] = {}
        self._load_dataset()

    def _load_dataset(self):
        # imported here rather than at module scope so the other suites never
        # pay for it, in import time or in dependencies
        import agentic_eval as ae
        self.ae = ae

        print(f"Loading agentic dataset (source: {self.source})...")
        path = Path(self.source)
        if path.exists():
            rows = [json.loads(line) for line in path.read_text().splitlines()
                    if line.strip()]
        else:
            from datasets import load_dataset
            rows = [dict(r) for r in load_dataset(self.source, split="test")]

        if self.lang_filter:
            rows = [r for r in rows if r.get("lang") == self.lang_filter]
        if not rows:
            raise ValueError(
                f"no agentic tasks in {self.source}"
                + (f" for language {self.lang_filter!r}" if self.lang_filter else ""))

        self.tasks = [ae.AgenticTask.from_record(r) for r in rows]
        self.languages = sorted({t.lang for t in self.tasks})
        print(f"Agentic dataset loaded: {len(self.tasks)} tasks "
              f"({', '.join(self.languages)})")

    # -- BaseDataset surface --------------------------------------------

    def __len__(self) -> int:
        return len(self.tasks)

    def get_question(self, index: int) -> Dict:
        return {"index": index}

    def get_question_text(self, question: Dict) -> str:
        task = self.tasks[question["index"]]
        first = task.instruction.strip().splitlines()[0]
        return f"[{task.task_id}] {first}"

    def get_answer(self, question: Dict) -> str:
        task = self.tasks[question["index"]]
        return json.dumps({"task_id": task.task_id, "index": question["index"]})

    def get_prompt(self, question: Dict) -> str:
        return self.ae.user_prompt(self.tasks[question["index"]])

    def default_exec_timeout(self) -> float:
        return 90.0

    def required_packages(self) -> List[str]:
        return []          # provisioned per language instead, see prepare()

    # -- agentic surface -------------------------------------------------

    def prepare(self, base_python: Optional[str] = None, force: bool = False):
        """Provision one sandbox per language present in the selected tasks."""
        for lang in self.languages:
            sandbox = self.ae.make_sandbox(lang, Sandbox, base_python=base_python)
            sandbox.ensure(force=force)
            self._sandboxes[lang] = sandbox
        return self._sandboxes

    def run(self, index: int, chat) -> Dict[str, Any]:
        task = self.tasks[index]
        return self.ae.run_task(
            task, chat, self._sandboxes[task.lang],
            max_turns=self.max_turns, context_limit=self.context_limit,
            test_timeout=self.default_exec_timeout())


def _as_text(value: Any) -> str:
    """Some dataset fields arrive as a list of lines rather than a blob."""
    if not value:
        return ""
    if isinstance(value, (list, tuple)):
        return "\n".join(str(v) for v in value)
    return str(value)


def _count_test_methods(test_code: str) -> int:
    """How many `def test_*` a task's test suite defines."""
    try:
        tree = ast.parse(test_code)
    except SyntaxError:
        return len(re.findall(r"^\s*def\s+test\w*\s*\(", test_code, re.M))
    return sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test")
    )


def third_party_imports(source: str) -> set:
    """Top-level modules imported by `source` that are not in the stdlib."""
    import ast

    mods = set()
    for chunk in source.split("\n\n"):
        try:
            tree = ast.parse(chunk)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                mods.add(node.module.split(".")[0])

    stdlib = set(sys.stdlib_module_names)
    return {m for m in mods if m and m not in stdlib}


class Grader:
    def __init__(
        self,
        grader_type: str = "llm",
        grader_script: Optional[str] = None,
        grader_model_name: Optional[str] = None,
        grader_server_url: str = "",
        dataset_type: str = "aime",
        sandbox: Optional[Sandbox] = None,
        exec_timeout: float = 15.0,
        exec_seed: Optional[int] = 0,
        dataset: Optional[BaseDataset] = None,
    ):
        self.grader_type = grader_type
        self.grader_script = grader_script
        self.grader_model_name = grader_model_name
        self.grader_server_url = grader_server_url
        self.dataset_type = dataset_type
        self.sandbox = sandbox
        self.exec_timeout = exec_timeout
        self.exec_seed = exec_seed
        self.dataset = dataset
        self.pattern = self._get_pattern()
        # grading runs on the worker pool, so per-run exec detail is per-thread
        self._local = threading.local()

    @property
    def last_exec(self) -> Dict[str, Any]:
        return getattr(self._local, "exec_info", {})

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

    def _grade_exec(self, gold: str, pred: str) -> Tuple[bool, Optional[str]]:
        """Grade by running the generated code against the suite's own tests.

        `gold` is the JSON harness produced by the dataset's get_answer(); the
        dataset owns how the program is assembled and how its result is read.
        """
        if self.sandbox is None or self.dataset is None:
            raise ValueError("exec grader requires a sandbox and a dataset")

        question = json.loads(gold)
        completion = self.dataset.extract(pred, question)
        if not completion.strip():
            self._local.exec_info = {"status": "empty", "reason": "no code in reply"}
            return False, None

        program = self.dataset.build_program(question, completion, seed=self.exec_seed)
        result = self.sandbox.run(program, timeout=self.exec_timeout)
        passed, detail = self.dataset.interpret(result, question)

        self._local.exec_info = detail
        return passed, completion

    def grade(self, gold: str, pred: str, problem: str = "") -> Tuple[bool, Optional[str]]:
        """Grade the response"""
        if self.grader_type == "regex":
            return self._grade_regex(gold, pred)
        elif self.grader_type == "cli":
            return self._grade_cli(gold, pred)
        elif self.grader_type == "llm":
            return self._grade_llm(gold, pred, problem)
        elif self.grader_type == "exec":
            return self._grade_exec(gold, pred)
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

        # Extract chunk_idx from task_id: "{dataset_type}_{chunk_idx:03d}_{index:03d}"
        _parts = task_id.rsplit("_", 2)
        chunk_idx = int(_parts[-2]) if len(_parts) >= 3 else 0
        problem_idx = i

        task_state = TaskState(
            task_id=task_id,
            prompt=prompt,
            expected=expected,
            question_text=question_text,
            server_name=server_config.name,
            chunk_idx=chunk_idx,
            problem_idx=problem_idx,
        )

        if eval_state.loaded_dataset.is_agentic:
            return self._process_agentic_case(
                server_config, eval_state, i, task_id, task_state)

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
                    tokens, tps_gen, t_gen_ms, reasoning_content, server_config.name,
                    chunk_idx, problem_idx,
                )
                eval_state.dump()
                return task_state

            # Text graders only ever look at the tail of a reply, but the exec
            # grader needs the whole thing -- the code block is not at the end.
            if self.grader.grader_type == "exec":
                graded_text = result
            else:
                graded_text = self.grader._truncate_response(result, max_lines=10)
            is_correct, answer = self.grader.grade(expected, graded_text, prompt)

            grader_log = {
                "pred": graded_text,
                "grader_type": self.grader.grader_type
            }
            if self.grader.grader_type == "regex" and self.grader.pattern:
                grader_log["pattern"] = self.grader.pattern
            if self.grader.grader_type == "exec":
                grader_log.update(self.grader.last_exec)

            task_state.correct = is_correct
            task_state.answer = answer
            task_state.grader_log = grader_log
            task_state.status = "ok"

            eval_state.add_result(
                task_id, prompt, expected, result, answer,
                grader_log, is_correct, "ok",
                tokens, tps_gen, t_gen_ms, reasoning_content, server_config.name,
                chunk_idx, problem_idx,
            )

            eval_state.dump()

        except Exception as e:
            task_state.status = f"error: {str(e)}"

        return task_state

    def _agentic_chat(self, server_config: ServerConfig, eval_state: EvalState):
        """A chat callable for agentic_eval: messages + tools in, response out."""
        url = f"{server_config.url}/v1/chat/completions"

        def chat(messages, tools):
            data = {
                "model": self.model_name if self.model_name else "llama",
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto",
            }
            # Cap each turn. A model that falls into a repetition loop would
            # otherwise generate until the context runs out, burning the whole
            # run's wall clock on one turn; capped, it costs a single turn and
            # the episode carries on.
            per_turn = getattr(eval_state.dataset, "max_tokens", 0)
            if self.n_predict and self.n_predict > 0:
                data["max_tokens"] = self.n_predict
            elif per_turn:
                data["max_tokens"] = per_turn
            for key in ("temperature", "top_k", "top_p", "min_p"):
                value = eval_state.sampling_config.get(key)
                if value is not None:
                    data[key] = value
            response = requests.post(url, headers={"Content-Type": "application/json"},
                                     json=data, timeout=1800)
            response.raise_for_status()
            return response.json()

        return chat

    def _process_agentic_case(
        self, server_config: ServerConfig, eval_state: EvalState, i: int,
        task_id: str, task_state: TaskState
    ) -> TaskState:
        """Run one whole tool-driven episode and grade the tree it leaves behind."""
        try:
            result = eval_state.loaded_dataset.run(
                i, self._agentic_chat(server_config, eval_state))
        except Exception as e:
            # Record the failure rather than only marking the in-memory state.
            # Without this the case is persisted as "pending" and the summary
            # quietly counts a smaller denominator, so a run where most
            # episodes crashed reports a high score over the few that survived.
            task_state.status = f"error: {type(e).__name__}: {e}"
            eval_state.add_result(
                task_id, task_state.prompt, task_state.expected,
                task_state.status, None,
                {"error": f"{type(e).__name__}: {e}", "resolved": False},
                False, task_state.status, 0, None, None, None,
                server_config.name, task_state.chunk_idx, i,
            )
            eval_state.dump()
            return task_state

        episode = result.get("episode", {})
        task_state.tokens = episode.get("completion_tokens", 0)
        task_state.correct = bool(result["resolved"])
        task_state.answer = ", ".join(result.get("changed_files", [])) or None
        task_state.grader_log = result
        task_state.status = "ok"
        task_state.response = (
            f"stop={episode.get('stop_reason')} turns={episode.get('turns')} "
            f"calls={episode.get('tool_calls')} edits={episode.get('edits')} "
            f"errors={episode.get('tool_errors')} nudges={episode.get('nudges')} "
            f"peak_ctx={episode.get('peak_context')}\n"
            f"changed: {task_state.answer or '(nothing)'}"
        )

        eval_state.add_result(
            task_id, task_state.prompt, task_state.expected, task_state.response,
            task_state.answer, result, task_state.correct, "ok",
            task_state.tokens, None, None, None, server_config.name,
            task_state.chunk_idx, i,
        )
        eval_state.dump()
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

def preflight_sandbox(sandbox: Sandbox):
    """Catch host policies that would fail problems for non-model reasons.

    A FIPS-enforcing OpenSSL makes hashlib.md5() raise, which would sink
    HumanEval/162 no matter what the model generated.
    """
    probe = sandbox.run("import hashlib; hashlib.md5(b'x').hexdigest()", timeout=15)
    if not probe.passed:
        print(f"Warning: md5 is unavailable in the execution venv "
              f"({probe.summary(120)}). Problems that hash with md5 will fail "
              f"regardless of the model -- this is a host policy, not a model result.")


def print_pass_at_k(eval_state: "EvalState"):
    """Report pass@k plus a breakdown of how the failures failed.

    Each chunk is one full pass over the suite, so running --n_cases at a
    multiple of the dataset size gives n samples per problem for free.
    """
    cases = eval_state.task_states.get("cases", {})
    if not cases:
        return

    by_problem: Dict[int, List[bool]] = {}
    reasons: Dict[str, int] = {}
    for case in cases.values():
        by_problem.setdefault(case.get("problem_idx", 0), []).append(
            bool(case.get("correct", False))
        )
        if not case.get("correct", False):
            status = (case.get("grader_log") or {}).get("status", case.get("status", "?"))
            reasons[status] = reasons.get(status, 0) + 1

    n = min(len(v) for v in by_problem.values())
    n_problems = len(by_problem)

    print(f"\n{'=' * 60}")
    print(f"  {eval_state.dataset_type}: {n_problems} problems, {n} sample(s) each")
    print(f"{'=' * 60}")

    for k in sorted({1, 5, 10, 100, n}):
        if k > n:
            continue
        score = sum(
            pass_at_k(len(v), sum(v), k) for v in by_problem.values()
        ) / n_problems
        print(f"  pass@{k:<4d} {score:.4f}")

    if n == 1:
        solved = sum(1 for v in by_problem.values() if v[0])
        lo, hi = wilson_interval(solved, n_problems)
        print(f"  {solved}/{n_problems} solved   95% CI [{lo:.4f}, {hi:.4f}]")

    # Suites judged by many unit tests per task also get a partial-credit score,
    # which separates "generated nothing usable" from "got most of it right".
    # A case that never reached the grader (truncated generation, HTTP error)
    # still owes its tests to the denominator, so fall back to the count the
    # dataset recorded in the gold blob -- otherwise those tests silently
    # disappear and the rate is computed against a smaller total.
    t_total = t_passed = 0
    for case in cases.values():
        log = case.get("grader_log") or {}
        if "tests_total" in log:
            t_total += log["tests_total"]
        else:
            try:
                t_total += json.loads(case.get("expected") or "{}").get("n_tests", 0)
            except ValueError:
                pass
        t_passed += log.get("tests_passed", 0)
    if t_total:
        print(f"\n  test methods passed  {t_passed}/{t_total}  ({t_passed / t_total:.4f})")

    if reasons:
        print("\n  failures by cause:")
        for status, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"    {status:<12s} {count}")


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
        choices=["aime", "aime2025", "aime2026", "gsm8k", "gpqa", "humaneval",
                 "classeval", "agentic"],
        help="Dataset type (default: aime)"
    )
    parser.add_argument(
        "--dataset-source",
        type=str,
        default=None,
        help="Override the dataset location: a local .jsonl path or a HuggingFace "
             "repo id (e.g. a patched HumanEval). Default: the suite's canonical repo"
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
        default=None,
        choices=["regex", "cli", "llm", "exec"],
        help="Grader type: regex, cli, llm, or exec (default: exec for code suites, "
             "llm otherwise). 'exec' runs the generated code against the suite tests "
             "in a sandboxed venv"
    )
    parser.add_argument(
        "--exec-timeout",
        type=float,
        default=None,
        help="Per-problem wall-clock limit for the exec grader, in seconds "
             "(default: per-suite, 15s for humaneval, 30s for classeval)"
    )
    parser.add_argument(
        "--exec-seed",
        type=int,
        default=0,
        help="Seed the global RNG before running each test, so suites whose tests "
             "draw random inputs score reproducibly (default: 0). Use -1 to leave "
             "the RNG unseeded, matching the stock upstream harness"
    )
    parser.add_argument(
        "--sandbox-python",
        type=str,
        default=None,
        help="Base interpreter used to build the execution venv (default: this one)"
    )
    parser.add_argument(
        "--rebuild-sandbox",
        action="store_true",
        help="Discard and re-provision the cached execution venv"
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
        "--agentic-lang",
        type=str,
        default=None,
        choices=["python", "typescript"],
        help="Restrict the agentic suite to one language. Only that language's "
             "toolchain is then provisioned."
    )
    parser.add_argument(
        "--agentic-max-turns",
        type=int,
        default=50,
        help="Maximum assistant turns per agentic episode (default: 50)"
    )
    parser.add_argument(
        "--agentic-max-tokens",
        type=int,
        default=2048,
        help="Cap on generated tokens per agentic turn (default: 2048). Bounds "
             "a model that falls into a repetition loop."
    )
    parser.add_argument(
        "--agentic-context-limit",
        type=int,
        default=None,
        help="Abandon an agentic episode once its context exceeds this many "
             "tokens (default: no limit)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing eval state"
    )

    args = parser.parse_args()

    global DATASET_SOURCE, AGENTIC_OPTS
    DATASET_SOURCE = args.dataset_source
    AGENTIC_OPTS = {
        "lang": args.agentic_lang,
        "max_turns": args.agentic_max_turns,
        "context_limit": args.agentic_context_limit,
        "max_tokens": args.agentic_max_tokens,
    }

    # Code suites are graded by execution; everything else keeps the llm default.
    if args.grader_type is None:
        args.grader_type = "exec" if args.dataset in CODE_SUITES else "llm"

    if args.grader_type == "exec" and args.dataset not in CODE_SUITES:
        print(f"Error: --grader-type exec is only meaningful for code suites "
              f"({', '.join(sorted(CODE_SUITES))}), not '{args.dataset}'")
        sys.exit(1)

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

    # The execution venv is derived from the dataset, so it is built only once
    # the dataset is known -- true on both the fresh and the resume path.
    if args.grader_type == "exec":
        dataset = eval_state.loaded_dataset

        if dataset.is_agentic:
            # One venv per language in the selected tasks, not one per suite
            dataset.prepare(base_python=args.sandbox_python,
                            force=args.rebuild_sandbox)
        else:
            base_python = args.sandbox_python
            want = dataset.preferred_python()
            if not base_python and want:
                base_python = resolve_python(want)
                if base_python:
                    print(f"[sandbox] using Python {want} for "
                          f"{eval_state.dataset_type} ({base_python})")

            dataset.configure_python(base_python or sys.executable)

            sandbox = Sandbox(
                name=eval_state.dataset_type,
                packages=dataset.required_packages(),
                provision=dataset.provision_steps(),
                env=dataset.sandbox_env(),
                base_python=base_python,
            )
            sandbox.ensure(force=args.rebuild_sandbox)
            preflight_sandbox(sandbox)
            grader.sandbox = sandbox
            grader.dataset = dataset
            grader.exec_timeout = (args.exec_timeout if args.exec_timeout is not None
                                   else dataset.default_exec_timeout())
            grader.exec_seed = None if args.exec_seed < 0 else args.exec_seed

    processor = Processor(
        server_configs=server_configs,
        grader=grader,
        model_name=args.model,
        n_predict=args.n_predict
    )

    processor.evaluate(eval_state, verbose=args.verbose, resume=resume)

    if args.grader_type == "exec":
        print_pass_at_k(eval_state)

    print(f"\nEval state dumped to {args.output}")

if __name__ == "__main__":
    main()
