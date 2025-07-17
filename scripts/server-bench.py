#!/usr/bin/env python3

import argparse
import json
import os
import random
import subprocess
from time import sleep, time
from typing import Optional, Union

import datasets
import logging
import matplotlib.pyplot as plt
import numpy as np
import requests
from tqdm.contrib.concurrent import thread_map


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("server-bench")


def get_prompts_text(dataset_name: str, n_prompts: int) -> Optional[list[str]]:
    ret = []
    if dataset_name.lower() == "mmlu":
        logger.info("Loading MMLU dataset...")
        ret = datasets.load_dataset("cais/mmlu", "all")["test"]["question"]  # type: ignore
    else:
        return None
    if n_prompts >= 0:
        ret = ret[:n_prompts]
    return ret


def get_prompt_lengths_rng(n_prompts: int, prompt_length_min: int, prompt_length_max: int) -> list[int]:
    assert n_prompts >= 0
    ret: list[int] = []
    for i in range(n_prompts):
        random.seed(13 * i + 0)
        ret.append(random.randint(prompt_length_min, prompt_length_max))
    return ret


def get_prompts_rng(prompt_lengths: list[int]) -> list[list[int]]:
    return [[random.randint(100, 10000) for _ in range(pl)] for pl in prompt_lengths]


def get_server(path_server: str, path_log: Optional[str]) -> dict:
    logger.info("Starting the llama.cpp server...")
    hostname: str = os.environ.get("LLAMA_ARG_HOST", "127.0.0.1")
    port: str = os.environ.get("LLAMA_ARG_PORT", "8080")
    address: str = f"http://{hostname}:{port}"

    fout = open(path_log, "w") if path_log is not None else subprocess.DEVNULL
    process = subprocess.Popen([path_server], stdout=fout, stderr=subprocess.STDOUT)

    n_failures: int = 0
    while True:
        try:
            sleep(1.0)
            exit_code = process.poll()
            if exit_code is not None:
                raise RuntimeError(f"llama.cpp server exited unexpectedly with exit code {exit_code}, see {path_log}")
            response = requests.get(f"{address}/health")
            if response.status_code == 200:
                break
        except requests.ConnectionError:
            n_failures += 1
            if n_failures >= 10:
                raise RuntimeError("llama.cpp server is not healthy after 10 seconds")

    return {"process": process, "address": address, "fout": fout}


def get_prompt_length(data: dict) -> int:
    session = data["session"]
    server_address: str = data["server_address"]

    response = session.post(
        f"{server_address}/apply-template",
        json={"messages": [{"role": "user", "content": data["prompt"], "stream": True}]}
    )
    if response.status_code != 200:
        raise RuntimeError(f"Server returned status code {response.status_code}: {response.text}")
    prompt: str = json.loads(response.text)["prompt"]
    response = session.post(
        f"{server_address}/tokenize",
        json={"content": prompt, "add_special": True}
    )
    if response.status_code != 200:
        raise RuntimeError(f"Server returned status code {response.status_code}: {response.text}")
    tokens: list[str] = json.loads(response.text)["tokens"]
    return len(tokens)


def send_prompt(data: dict) -> tuple[float, list[float]]:
    session = data["session"]
    server_address: str = data["server_address"]

    t_submit = time()
    if data["synthetic_prompt"]:
        json_data: dict = {
            "prompt": data["prompt"], "ignore_eos": True, "cache_prompt": False,
            "seed": data["seed"], "n_predict": data["n_predict"], "stream": True}
        response = session.post(f"{server_address}/completion", json=json_data, stream=True)
    else:
        response = session.post(
            f"{server_address}/apply-template",
            json={"messages": [{"role": "user", "content": data["prompt"], "stream": True}]}
        )
        if response.status_code != 200:
            raise RuntimeError(f"Server returned status code {response.status_code}: {response.text}")
        prompt: str = json.loads(response.text)["prompt"]

        json_data: dict = {"prompt": prompt, "seed": data["seed"], "n_predict": data["n_predict"], "stream": True}
        response = session.post(f"{server_address}/completion", json=json_data, stream=True)

    token_arrival_times: list[float] = []
    for line in response.iter_lines(decode_unicode=False):
        if not line.startswith(b"data: "):
            continue
        token_arrival_times.append(time())
    token_arrival_times = token_arrival_times[:-1]

    if response.status_code != 200:
        raise RuntimeError(f"Server returned status code {response.status_code}: {response.text}")

    return (t_submit, token_arrival_times)


def benchmark(path_server: str, path_log: Optional[str], prompt_source: str, n_prompts: int, n_predict: int, n_predict_min: int):
    if os.environ.get("LLAMA_ARG_N_PARALLEL") is None:
        logger.info("LLAMA_ARG_N_PARALLEL not explicitly set, using 32")
        os.environ["LLAMA_ARG_N_PARALLEL"] = "32"
    if os.environ.get("LLAMA_ARG_N_GPU_LAYERS") is None:
        logger.info("LLAMA_ARG_N_GPU_LAYERS not explicitly set, using 999")
        os.environ["LLAMA_ARG_N_GPU_LAYERS"] = "999"
    if os.environ.get("LLAMA_ARG_FLASH_ATTN") is None:
        logger.info("LLAMA_ARG_FLASH_ATTN not explicitly set, using 'true'")
        os.environ["LLAMA_ARG_FLASH_ATTN"] = "true"

    parallel: int = int(os.environ.get("LLAMA_ARG_N_PARALLEL", 1))
    prompts: Union[None, list[str], list[list[int]]] = get_prompts_text(prompt_source, n_prompts)
    synthetic_prompts: bool = prompts is None
    prompt_n = []

    if synthetic_prompts:
        prompt_source_split: list[str] = prompt_source.split("-")
        assert len(prompt_source_split) == 3
        assert prompt_source_split[0].lower() == "rng"
        prompt_length_min: int = int(prompt_source_split[1])
        prompt_length_max: int = int(prompt_source_split[2])
        logger.info("Generating random prompts...")
        prompt_n = get_prompt_lengths_rng(n_prompts, prompt_length_min, prompt_length_max)
        prompts = get_prompts_rng(prompt_n)
    else:
        n_predict_min = n_predict

    if os.environ.get("LLAMA_ARG_CTX_SIZE") is None:
        context_per_slot: int = int(1.05 * (n_predict + (np.max(prompt_n) if synthetic_prompts else 2048)))
        context_total: int = context_per_slot * parallel
        os.environ["LLAMA_ARG_CTX_SIZE"] = str(context_total)
        logger.info(f"LLAMA_ARG_CTX_SIZE not explicitly set, using {context_total} ({context_per_slot} per slot).")

    server: Optional[dict] = None
    session = None
    try:
        server = get_server(path_server, path_log)
        server_address: str = server["address"]

        adapter = requests.adapters.HTTPAdapter(pool_connections=parallel, pool_maxsize=parallel)  # type: ignore
        session = requests.Session()
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        data: list[dict] = []

        for i, p in enumerate(prompts):
            random.seed(13 * i + 1)
            data.append({
                "session": session, "server_address": server_address, "prompt": p, "synthetic_prompt": synthetic_prompts,
                "n_predict": random.randint(n_predict_min, n_predict), "seed": 13 * i + 2})

        if not synthetic_prompts:
            logger.info("Getting the prompt lengths...")
            prompt_n = [get_prompt_length(d) for d in data]

        logger.info("Starting the benchmark...\n")
        t0 = time()
        results: list[tuple[float, list[float]]] = thread_map(send_prompt, data, max_workers=parallel, chunksize=1)
    finally:
        if server is not None:
            server["process"].terminate()
            server["process"].wait()
        if session is not None:
            session.close()

    prompt_t = []
    token_t = []
    depth_sum: int = 0
    for pn, (t_submit, tat) in zip(prompt_n, results):
        prompt_t.append(tat[0] - t_submit)
        token_t += tat
        n_tokens: int = len(tat)
        depth_sum += n_tokens * pn
        depth_sum += n_tokens * (n_tokens + 1) // 2
    assert len(token_t) > 0
    prompt_n = np.array(prompt_n, dtype=np.int64)
    prompt_t = np.array(prompt_t, dtype=np.float64)
    token_t = np.array(token_t, dtype=np.float64)

    token_t -= t0
    token_t_last = np.max(token_t)

    logger.info("")
    logger.info(f"Benchmark duration:                {token_t_last:.2f} s")
    logger.info(f"Request throughput:                {n_prompts / token_t_last:.2f} requests/s = {n_prompts / (token_t_last/60):.2f} requests/min")
    logger.info(f"Total prompt length:               {np.sum(prompt_n)} tokens")
    logger.info(f"Average prompt length:             {np.mean(prompt_n):.2f} tokens")
    logger.info(f"Average prompt latency:            {1e3 * np.mean(prompt_t):.2f} ms")
    logger.info(f"Average prompt speed:              {np.sum(prompt_n) / np.sum(prompt_t):.2f} tokens/s")
    logger.info(f"Total generated tokens:            {token_t.shape[0]}")
    logger.info(f"Average generation depth:          {depth_sum / token_t.shape[0]:.2f} tokens")
    logger.info(f"Average total generation speed:    {token_t.shape[0] / token_t_last:.2f} tokens/s")
    logger.info(f"Average generation speed per slot: {token_t.shape[0] / (parallel * token_t_last):.2f} tokens/s / slot")
    logger.info("")
    logger.info(
        "The above numbers are the speeds as observed by the Python script and may differ from the performance reported by the server, "
        "particularly when the server is fast vs. the network or Python script (e.g. when serving a very small model).")

    plt.figure()
    plt.scatter(prompt_n, 1e3 * prompt_t, s=10.0, marker=".", alpha=0.25)
    plt.xlim(0, 1.05e0 * np.max(prompt_n))
    plt.ylim(0, 1.05e3 * np.max(prompt_t))
    plt.xlabel("Prompt length [tokens]")
    plt.ylabel("Time to first token [ms]")
    plt.savefig("prompt_time.png", dpi=240)

    bin_max = np.ceil(token_t_last) + 1
    plt.figure()
    plt.hist(token_t, np.arange(0, bin_max))
    plt.xlim(0, bin_max + 1)
    plt.xlabel("Time [s]")
    plt.ylabel("Num. tokens generated per second")
    plt.savefig("gen_rate.png", dpi=240)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tool for benchmarking the throughput of the llama.cpp HTTP server. "
        "Results are printed to console and visualized as plots (saved to current working directory). "
        "To pass arguments such as the model path to the server, set the corresponding environment variables (see llama-server --help).")
    parser.add_argument("--path_server", type=str, default="llama-server", help="Path to the llama.cpp server binary")
    parser.add_argument("--path_log", type=str, default="server-bench.log", help="Path to the model to use for the benchmark")
    parser.add_argument(
        "--prompt_source", type=str, default="rng-1024-2048",
        help="How to get the prompts for the benchmark, either 'mmlu' for MMLU questions or "
        "rng-MIN-MAX for synthetic prompts with random lengths in the interval [MIN, MAX]")
    parser.add_argument("--n_prompts", type=int, default=100, help="Number of prompts to evaluate")
    parser.add_argument("--n_predict", type=int, default=2048, help="Max. number of tokens to predict per prompt")
    parser.add_argument(
        "--n_predict_min", type=int, default=1024,
        help="Min. number of tokens to predict per prompt (supported for synthetic prompts only)")
    args = parser.parse_args()
    benchmark(**vars(args))
