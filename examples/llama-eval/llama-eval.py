#!/usr/bin/env python3

import re
import argparse
import json
import os
import random
import subprocess
from time import sleep, time
from typing import Optional, Union

import datasets
import logging
import requests
from tqdm.contrib.concurrent import thread_map
from typing import Iterator
from abc import ABC

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("llama-eval")


MATH_TEMPLATE = """
{question}
Put your final answer within \\boxed{{}}.
"""

MC_FROM_INT = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
}


def format_multiple_choice(prompt: str, choices: list[str]):
    QUERY_TEMPLATE_MULTICHOICE = """
    {question}

    (A) {A}
    (B) {B}
    (C) {C}
    (D) {D}

    Express your final answer as the corresponding option 'A', 'B', 'C', or 'D'. Put your final answer within \\boxed{{}}.

    """.strip()
    A_str = choices[0]
    B_str = choices[1]
    C_str = choices[2]
    D_str = choices[3]
    query = QUERY_TEMPLATE_MULTICHOICE.format(
        question=prompt, A=A_str, B=B_str, C=C_str, D=D_str
    )
    return query


# Preprocess hellaswag
def preprocess(text):
    text = text.strip()
    # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def hellaswag_process_doc(doc):
    ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
    question = preprocess(doc["activity_label"] + ": " + ctx)
    proc_answers = [preprocess(answer) for answer in doc["endings"]]
    prompt = format_multiple_choice(question, proc_answers)
    out_doc = {
        "prompt": prompt,
        "gold": MC_FROM_INT[int(doc["label"])],
    }
    return out_doc


def mmlu_process_doc(doc):
    prompt = format_multiple_choice(doc["question"], doc["choices"])
    out_doc = {
        "prompt": prompt,
        "gold": MC_FROM_INT[int(doc["answer"])],
    }
    return out_doc


def extract_boxed_text(text):
    pattern = r"boxed{(.*?)}|framebox{(.*?)}"
    matches = re.findall(pattern, text, re.DOTALL)
    logger.debug(matches)
    if matches:
        for match in matches[::-1]:
            for group in match:
                if group != "":
                    return group.split(",")[-1].strip()
    logger.warning(
        "Could not extract boxed text. Using last integer. Maybe expand context window"
    )
    pattern = r"\d+"  # get the last integer if no pattern found
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1]

    return ""


def get_prompts_text(
    dataset_name: str, ds: datasets.Dataset
) -> Optional[tuple[list[str], list[str]]]:
    ret = []
    if dataset_name.lower() == "mmlu":
        ds = ds.map(mmlu_process_doc)
        ret = ds["prompt"], ds["gold"]
    elif dataset_name.lower() == "hellaswag":
        ds = ds.map(hellaswag_process_doc)
        ret = ds["prompt"], ds["gold"]
    elif dataset_name.lower() == "aime":
        ds = ds.map(
            lambda k: {
                "prompt": MATH_TEMPLATE.format(
                    question=k["problem"],
                )
            }
        )
        ret = ds["prompt"], ds["answer"]
    elif dataset_name.lower() == "gsm8k":
        ds = ds.map(lambda k: {"prompt": MATH_TEMPLATE.format(question=k["question"])})
        la = []
        for answer in ds["answer"]:
            la.append(answer.split("### ")[-1].rstrip())
        ret = ds["prompt"], la
    else:
        return None

    return ret


def get_dataset(
    dataset_name: str, n_prompts: int, rng_seed: int
) -> Optional[datasets.Dataset]:
    ds = None
    cache_dir = "./build/bin/datasets"
    logger.info(f"Loading {dataset_name.lower()} dataset...")
    if dataset_name.lower() == "mmlu":
        ds = datasets.load_dataset(
            "cais/mmlu", "all", split="test", cache_dir=cache_dir
        )
    elif dataset_name.lower() == "hellaswag":
        ds = datasets.load_dataset(
            "Rowan/hellaswag", split="validation", cache_dir=cache_dir
        )
    elif dataset_name.lower() == "aime":
        ds = datasets.load_dataset(
            "AI-MO/aimo-validation-aime", split="train", cache_dir=cache_dir
        )
    elif dataset_name.lower() == "gsm8k":
        ds = datasets.load_dataset("openai/gsm8k", split="test")
    else:
        return None

    if n_prompts >= 0:
        ds = ds.shuffle(seed=rng_seed)
        ds = ds.select(range(min(n_prompts, len(ds))))
    return ds


def send_prompt(data: dict) -> int:
    session = data["session"]
    server_address: str = data["server_address"]
    prompt: str = data["prompt"]
    logger.info(f"data['external_server'] {data['external_server']}")
    logger.info(f"data['prompt'] {prompt}")
    logger.info(f"data['n_predict'] {data['n_predict']}")

    json_data: dict = {
        "prompt": prompt,
        "max_tokens": data["n_predict"],
        "temperature": 0,
    }
    response = session.post(f"{server_address}/v1/completions", json=json_data)
    res = json.loads(response.text)
    logger.info(f"response {res}")
    extracted_answer = extract_boxed_text(res["choices"][0]["text"])
    source_answer = data["answer"]
    if data["prompt_source"] == "aime" or data["prompt_source"] == "gsm8k":
        try:  # All AIME answers are integers, so we convert the extracted answer to an integer
            extracted_answer = int(extracted_answer)
            source_answer = int(source_answer)
        except (ValueError, TypeError):
            extracted_answer = None
    logger.info(f"extracted_answer {extracted_answer}")
    logger.info(f"data['answer'] {data['answer']}")

    score = 1 if extracted_answer == source_answer else 0

    return score


def get_server(path_server: str, path_log: Optional[str]) -> dict:
    if path_server.startswith("http://") or path_server.startswith("https://"):
        return {"process": None, "address": path_server, "fout": None}
    if os.environ.get("LLAMA_ARG_HOST") is None:
        logger.info("LLAMA_ARG_HOST not explicitly set, using 127.0.0.1")
        os.environ["LLAMA_ARG_HOST"] = "127.0.0.1"
    if os.environ.get("LLAMA_ARG_PORT") is None:
        logger.info("LLAMA_ARG_PORT not explicitly set, using 8080")
        os.environ["LLAMA_ARG_PORT"] = "8080"
    hostname: Optional[str] = os.environ.get("LLAMA_ARG_HOST")
    port: Optional[str] = os.environ.get("LLAMA_ARG_PORT")
    assert hostname is not None
    assert port is not None
    address: str = f"http://{hostname}:{port}"
    logger.info(f"Starting the llama.cpp server under {address}...")

    fout = open(path_log.format(port=port), "w") if path_log is not None else subprocess.DEVNULL
    process = subprocess.Popen([path_server], stdout=fout, stderr=subprocess.STDOUT)

    n_failures: int = 0
    while True:
        try:
            sleep(1.0)
            exit_code = process.poll()
            if exit_code is not None:
                raise RuntimeError(f"llama.cpp server exited unexpectedly with exit code {exit_code}{path_log and f', see {path_log.format(port=port)}' or ''}")
            response = requests.get(f"{address}/health")
            if response.status_code == 200:
                break
        except requests.ConnectionError:
            n_failures += 1
            if n_failures >= 10:
                raise RuntimeError("llama.cpp server is not healthy after 10 seconds")

    return {"process": process, "address": address, "fout": fout}


def benchmark(
    path_server: str,
    path_log: Optional[str],
    prompt_source: str,
    n_prompts: int,
    n_predict: int,
    rng_seed: int,
):
    external_server: bool = path_server.startswith("http://") or path_server.startswith("https://")
    if os.environ.get("LLAMA_ARG_N_PARALLEL") is None:
        logger.info("LLAMA_ARG_N_PARALLEL not explicitly set, using 32")
        os.environ["LLAMA_ARG_N_PARALLEL"] = "32"

    parallel: int = int(os.environ.get("LLAMA_ARG_N_PARALLEL")) # type: ignore
    ds: Union[datasets.Dataset, None] = get_dataset(prompt_source, n_prompts, rng_seed)
    if not ds:
        logger.error("ERROR: get_dataset")
        exit(0)

    res: Union[tuple[list[str], list[str]], None] = get_prompts_text(prompt_source, ds)
    if not res:
        logger.error("ERROR: get_prompts_text")
        exit(0)

    prompts: Union[list[str], list[list[int]]] = res[0]
    answer: Union[list[str], list[list[int]]] = res[1]

    logger.info(prompts)
    logger.info(f"external_server {external_server}")

    server: Optional[dict] = None
    session = None
    try:
        server = get_server(path_server, path_log)
        server_address: str = server["address"]
        assert external_server == (server["process"] is None)

        adapter = requests.adapters.HTTPAdapter(pool_connections=parallel, pool_maxsize=parallel)  # type: ignore
        session = requests.Session()
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        data: list[dict] = []
        for p, a in zip(prompts, answer):
            data.append(
                {
                    "prompt_source": prompt_source,
                    "session": session,
                    "server_address": server_address,
                    "external_server": external_server,
                    "prompt": p,
                    "answer": a,
                    "n_predict": n_predict,
                }
            )

        logger.info("Starting the benchmark...\n")
        t0 = time()
        results: list[int] = thread_map(
            send_prompt, data, max_workers=parallel, chunksize=1
        )
    finally:
        if server is not None and server["process"] is not None:
            server["process"].terminate()
            server["process"].wait()
        if session is not None:
            session.close()

    t1 = time()

    correct: int = sum(results)
    total_questions: int = len(data)
    logger.info(f"llama-eval duration:                {t1-t0:.2f} s")
    logger.info(f"{prompt_source} correct:                {correct}")
    logger.info(f"{prompt_source} total_questions:                {total_questions}")
    logger.info(f"{prompt_source} accuracy:                {correct / total_questions}")


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
        default="llama-server",
        help="Path to the llama.cpp server binary",
    )
    parser.add_argument(
        "--path_log",
        type=str,
        default="server-bench-{port}.log",
        help="Path to the model to use for the benchmark",
    )
    parser.add_argument(
        "--prompt_source",
        type=str,
        default="mmlu",
        help="How to get the prompts for the benchmark, either 'mmlu' for MMLU questions",
    )
    parser.add_argument(
        "--n_prompts", type=int, default=100, help="Number of prompts to evaluate"
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
