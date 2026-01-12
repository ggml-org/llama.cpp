#!/usr/bin/env python3

import re
import argparse
import os
from time import time
from typing import Union, Any, Mapping, cast

import datasets
import logging
import requests
from tqdm.contrib.concurrent import thread_map
from typing import Iterator
from abc import ABC, abstractmethod
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("llama-eval")

MATH_TEMPLATE = """
{question}
Do not include any explanation. Put your final answer within \\boxed{{}}.
"""


def format_multiple_choice(prompt: str, choices: list[str]):
    lines = [prompt]

    labels = [chr(ord("A") + i) for i in range(len(choices))]
    for l, c in zip(labels, choices):
        lines.append(f"({l}): {c.strip()}")
    lines.append(
        "Do not include any explanation. Answer with the corresponding option letter only"
    )
    lines.append(", ".join(labels))
    lines.append("Put your final answer within \\boxed{{}}.")

    return "\n".join(lines), labels


def extract_boxed_text(text: str) -> str:
    pattern = r"boxed{(.*?)}|framebox{(.*?)}"
    matches = re.findall(pattern, text, re.DOTALL)
    logger.debug(matches)
    if matches:
        for match in matches[::-1]:
            for group in match:
                if group != "":
                    return group.split(",")[-1].strip()
    logger.warning("Could not extract boxed text. Maybe expand context window")

    return ""


@dataclass(frozen=True)
class Case:
    task: str
    kind: str
    case_id: str
    prompt: str
    gold: str
    meta_data: dict[str, Any]


class TaskSpec(ABC):
    name: str
    kind: str

    @abstractmethod
    def load(self, limit, seed) -> datasets.Dataset:
        pass

    @abstractmethod
    def iter_cases(self, limit: int, seed: int) -> Iterator[Case]:
        pass

    @staticmethod
    @abstractmethod
    def grade(case: Case, response: dict) -> dict[str, Any]:
        pass


class MCTaskSpec(TaskSpec):
    @staticmethod
    def grade(case: Case, response: dict) -> dict[str, Any]:
        logger.debug(f"response {response}")
        result = {
            "task": case.task,
            "case_id": case.case_id,
            "correct": 0,
            "pred": None,
            "gold": case.gold,
            "status": "ok",
        }

        try:
            extracted_answer = extract_boxed_text(response["choices"][0]["text"])
        except Exception as e:
            result["status"] = "error"
            logger.warning("ERROR: extract_boxed_text")

            return result

        if not extracted_answer:
            result["status"] = "invalid"
            logger.warning("INVALID: extract_boxed_text")
            return result

        logger.debug(f"extracted_answer {extracted_answer}")
        logger.debug(f"data['answer'] {case.gold}")
        result["pred"] = extracted_answer
        result["correct"] = 1 if extracted_answer == case.gold else 0

        return result


class MathTaskSpec(TaskSpec):

    @staticmethod
    def grade(case: Case, response: dict) -> dict[str, Any]:
        logger.debug(f"response {response}")
        result = {
            "task": case.task,
            "case_id": case.case_id,
            "correct": 0,
            "gold": case.gold,
            "status": "ok",
            "pred": None,
        }

        try:
            extracted_answer = extract_boxed_text(response["choices"][0]["text"])
        except Exception as e:
            result["status"] = "error"
            return result

        source_answer = case.gold
        try:  # All AIME answers are integers, so we convert the extracted answer to an integer
            extracted_answer = int(extracted_answer)
            source_answer = int(case.gold)
        except (ValueError, TypeError):
            result["status"] = "invalid"
            return result

        logger.debug(f"extracted_answer {extracted_answer}")
        logger.debug(f"data['answer'] {case.gold}")
        result["pred"] = extracted_answer
        result["correct"] = 1 if extracted_answer == source_answer else 0

        return result


class ARC_Task(MCTaskSpec):

    def __init__(self):
        self.name = "arc"
        self.kind = "mc"

    def load(self, limit, seed) -> datasets.Dataset:
        ds = datasets.load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        if limit:
            ds = ds.shuffle(seed=seed)
            ds = ds.select(range(min(limit, len(ds))))
        return ds

    def iter_cases(self, limit: int, seed: int) -> Iterator[Case]:
        ds = self.load(limit, seed)

        for i, doc in enumerate(ds):
            doc = cast(Mapping[str, Any], doc)

            prompt, labels = format_multiple_choice(
                doc["question"], doc["choices"]["text"]
            )
            yield Case(
                task=self.name,
                kind=self.kind,
                case_id=f"ARC-Challenge:{i}",
                prompt=prompt,
                gold=doc["answerKey"],
                meta_data={"labels": labels},
            )


class WinoGrande_Task(MCTaskSpec):

    def __init__(self):
        self.name = "winogrande"
        self.kind = "mc"

    def load(self, limit, seed) -> datasets.Dataset:
        ds = datasets.load_dataset(
            "winogrande", "winogrande_debiased", split="validation"
        )
        if limit:
            ds = ds.shuffle(seed=seed)
            ds = ds.select(range(min(limit, len(ds))))
        return ds

    def iter_cases(self, limit: int, seed: int) -> Iterator[Case]:
        ds = self.load(limit, seed)

        for i, doc in enumerate(ds):
            doc = cast(Mapping[str, Any], doc)

            prompt, labels = format_multiple_choice(
                doc["sentence"], [doc["option1"], doc["option2"]]
            )
            yield Case(
                task=self.name,
                kind=self.kind,
                case_id=f"winogrande:{i}",
                prompt=prompt,
                gold=labels[int(doc["answer"]) - 1],  # winogrande answers are 1 based
                meta_data={"labels": labels},
            )


class MMLU_Task(MCTaskSpec):

    def __init__(self):
        self.name = "mmlu"
        self.kind = "mc"

    def load(self, limit, seed) -> datasets.Dataset:
        ds = datasets.load_dataset("cais/mmlu", "all", split="test")
        if limit:
            ds = ds.shuffle(seed=seed)
            ds = ds.select(range(min(limit, len(ds))))
        return ds

    def iter_cases(self, limit: int, seed: int) -> Iterator[Case]:
        ds = self.load(limit, seed)

        for i, doc in enumerate(ds):
            doc = cast(Mapping[str, Any], doc)

            prompt, labels = format_multiple_choice(doc["question"], doc["choices"])
            yield Case(
                task=self.name,
                kind=self.kind,
                case_id=f"mmlu:{doc['subject']}:{i}",
                prompt=prompt,
                gold=labels[int(doc["answer"])],
                meta_data={"subject": doc["subject"], "labels": labels},
            )


class Hellaswag_Task(MCTaskSpec):

    # Preprocess hellaswag
    @staticmethod
    def preprocess(text: str):
        text = text.strip()
        # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
        text = text.replace(" [title]", ". ")
        text = re.sub("\\[.*?\\]", "", text)
        text = text.replace("  ", " ")
        return text

    @staticmethod
    def hellaswag_process_doc(doc: dict[str, str]):
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        question = Hellaswag_Task.preprocess(doc["activity_label"] + ": " + ctx)
        proc_answers = [Hellaswag_Task.preprocess(answer) for answer in doc["endings"]]
        prompt, labels = format_multiple_choice(question, proc_answers)
        out_doc = {
            "prompt": prompt,
            "gold": labels[int(doc["label"])],
        }
        return out_doc

    def __init__(self):
        self.name = "hellaswag"
        self.kind = "mc"

    def load(self, limit, seed) -> datasets.Dataset:
        ds = datasets.load_dataset("Rowan/hellaswag", split="validation")
        if limit:
            ds = ds.shuffle(seed=seed)
            ds = ds.select(range(min(limit, len(ds))))
        ds = ds.map(Hellaswag_Task.hellaswag_process_doc)

        return ds

    def iter_cases(self, limit: int, seed: int) -> Iterator[Case]:
        ds = self.load(limit, seed)
        for i, doc in enumerate(ds):
            doc = cast(Mapping[str, Any], doc)
            yield Case(
                task=self.name,
                kind=self.kind,
                case_id=f"hellaswag:{i}",
                prompt=doc["prompt"],
                gold=doc["gold"],
                meta_data={},
            )


class Aime_Task(MathTaskSpec):

    def __init__(self):
        self.name = "aime"
        self.kind = "math"

    def load(self, limit, seed) -> datasets.Dataset:
        ds = datasets.load_dataset("AI-MO/aimo-validation-aime", split="train")

        if limit:
            ds = ds.shuffle(seed=seed)
            ds = ds.select(range(min(limit, len(ds))))

        ds = ds.map(
            lambda ex: {
                "prompt": MATH_TEMPLATE.format(
                    question=ex["problem"],
                )
            }
        )
        return ds

    def iter_cases(self, limit: int, seed: int) -> Iterator[Case]:
        ds = self.load(limit, seed)

        for i, doc in enumerate(ds):
            doc = cast(Mapping[str, Any], doc)
            yield Case(
                task=self.name,
                kind=self.kind,
                case_id=f"aime:{i}",
                prompt=doc["prompt"],
                gold=doc["answer"],
                meta_data={},
            )


class Gsm8k_Task(MathTaskSpec):

    def __init__(self):
        self.name = "gsm8k"
        self.kind = "math"

    def load(self, limit, seed) -> datasets.Dataset:
        ds = datasets.load_dataset("openai/gsm8k", "main", split="test")
        if limit:
            ds = ds.shuffle(seed=seed)
            ds = ds.select(range(min(limit, len(ds))))

        ds = ds.map(
            lambda k: {
                "prompt": MATH_TEMPLATE.format(
                    question=k["question"],
                ),
                "gold": k["answer"].split("### ")[-1].rstrip(),
            }
        )
        return ds

    def iter_cases(self, limit: int, seed: int) -> Iterator[Case]:
        ds = self.load(limit, seed)

        for i, doc in enumerate(ds):
            doc = cast(Mapping[str, Any], doc)
            yield Case(
                task=self.name,
                kind=self.kind,
                case_id=f"gsm8k:{i}",
                prompt=doc["prompt"],
                gold=doc["gold"],
                meta_data={},
            )


TASK_DICT: dict[str, type[TaskSpec]] = {
    "mmlu": MMLU_Task,
    "aime": Aime_Task,
    "gsm8k": Gsm8k_Task,
    "hellaswag": Hellaswag_Task,
    "arc": ARC_Task,
    "winogrande": WinoGrande_Task,
}


def build_request(case: Case, n_predict: int) -> dict[str, Any]:
    json_data = {
        "n_predict": n_predict,
        "max_tokens": n_predict,
        "temperature": 0,
        "prompt": case.prompt,
    }
    return json_data


def send_prompt(
    case: Case,
    data: dict,
) -> dict[str, Union[str, int]]:
    ret_err = {
        "task": case.task,
        "case_id": case.case_id,
        "status": "error",
        "correct": 0,
        "gold": case.gold,
        "pred": "",
        "error": "",
    }
    session: requests.Session = data["session"]
    server_address: str = data["server_address"]
    task = TASK_DICT.get(case.task)
    if task is None:
        ret_err["error"] = f"unknown_task: {case.task}"
        return ret_err
    logger.debug(case.prompt)

    json_data = build_request(case, data["n_predict"])
    try:
        response = session.post(f"{server_address}/v1/completions", json=json_data)
        if response.ok:
            res_json = response.json()
        else:
            ret_err["error"] = f"http_response: {response.status_code}"
            logger.warning(ret_err["error"])
            return ret_err
    except Exception as e:
        ret_err["error"] = f"http_exception: {e}"
        logger.warning(ret_err["error"])
        return ret_err
    logger.debug(response.text)
    return TASK_DICT[case.task].grade(case, res_json)


def aggregate_by_task(results: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    tmp = {
        "total": 0,
        "error": 0,
        "invalid": 0,
        "correct": 0,
    }
    agg: dict[str, dict[str, int]] = {}
    for row in results:
        d = agg.get(row["task"], tmp.copy())
        d["total"] += 1
        status = row["status"]
        if status == "ok":
            d["correct"] += row["correct"]
        elif status == "invalid":
            d["invalid"] += 1
        elif status == "error":
            d["error"] += 1

        agg[row["task"]] = d
    return agg


def print_summary(pertask_results: dict[str, dict[str, int]]):
    print("\n=== llama-eval suite summary ===")
    print(
        f"{'Task':<15} {'Acc':>8} {'Correct':>8} {'Total':>8} {'Invalid':>8} {'Error':>8}"
    )
    print("-" * 65)

    suite_total = 0
    suite_correct = 0

    for task in sorted(pertask_results.keys()):
        stats = pertask_results[task]
        total = stats["total"]
        correct = stats["correct"]
        invalid = stats["invalid"]
        error = stats["error"]

        acc = (correct / total) if total > 0 else 0.0

        print(
            f"{task:<15} "
            f"{acc:8.3f} "
            f"{correct:8d} "
            f"{total:8d} "
            f"{invalid:8d} "
            f"{error:8d}"
        )

        suite_total += total
        suite_correct += correct

    # Overall summary
    print("-" * 65)
    suite_acc = (suite_correct / suite_total) if suite_total > 0 else 0.0
    print(
        f"{'ALL':<15} " f"{suite_acc:8.3f} " f"{suite_correct:8d} " f"{suite_total:8d}"
    )


def benchmark(
    path_server: str,
    prompt_source: str,
    n_prompts: int,
    n_predict: int,
    rng_seed: int,
):
    if not path_server.startswith("http://") and not path_server.startswith("https://"):
        logger.error("ERROR: malformed server path")
        return

    if os.environ.get("LLAMA_ARG_N_PARALLEL") is None:
        logger.info("LLAMA_ARG_N_PARALLEL not explicitly set, using 32")
        os.environ["LLAMA_ARG_N_PARALLEL"] = "32"

    parallel: int = int(os.environ.get("LLAMA_ARG_N_PARALLEL"))  # type: ignore

    task_queue: set[TaskSpec] = set()
    for src in prompt_source.split(","):
        if src == "all":
            for v in TASK_DICT.values():
                task_queue.add(v())
            break
        task_queue.add(TASK_DICT[src]())

    session = None
    try:
        server_address: str = path_server

        adapter = requests.adapters.HTTPAdapter(pool_connections=parallel, pool_maxsize=parallel)  # type: ignore
        session = requests.Session()
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        cases: list[Case] = []
        data: list[dict] = []
        for task in task_queue:
            for case in task.iter_cases(n_prompts, rng_seed):
                cases.append(case)
                data.append(
                    {
                        "prompt_source": prompt_source,
                        "session": session,
                        "server_address": server_address,
                        "n_predict": n_predict,
                    }
                )
        logger.info("Starting the benchmark...\n")
        t0 = time()
        results: list[dict[str, Union[str, int]]] = thread_map(
            send_prompt,
            cases,
            data,
            max_workers=parallel,
            chunksize=1,
        )
    finally:
        if session is not None:
            session.close()

    t1 = time()
    logger.info(f"\nllama-eval duration:           {t1-t0:.2f} s")

    pertask_results = aggregate_by_task(results)
    print_summary(pertask_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tool for benchmarking the throughput of the llama.cpp HTTP server. "
        "Results are printed to console and visualized as plots (saved to current working directory). "
        "To pass arguments such as the model path to the server, set the corresponding environment variables (see llama-server --help). "
        "The reported numbers are the speeds as observed by the Python script and may differ from the performance reported by the server, "
        "particularly when the server is fast vs. the network or Python script (e.g. when serving a very small model)."
    )
    parser.add_argument(
        "--path_server",
        type=str,
        default="http://localhost:8033",
        help="llama-server url",
    )
    parser.add_argument(
        "--prompt_source",
        type=str,
        default="mmlu",
        help=f"Eval types supported: all,{list(TASK_DICT.keys())}",
    )
    parser.add_argument(
        "--n_prompts", type=int, default=None, help="Number of prompts to evaluate"
    )
    parser.add_argument(
        "--rng_seed",
        type=int,
        default=42,
        help="Number to see rng (Used to select prompts from datasource)",
    )
    parser.add_argument(
        "--n_predict",
        type=int,
        default=2048,
        help="Max. number of tokens to predict per prompt",
    )
    args = parser.parse_args()
    benchmark(**vars(args))
