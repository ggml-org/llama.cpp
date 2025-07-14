#!/usr/bin/env python3

import argparse
import json
import subprocess
from time import sleep, time
from typing import Optional

import datasets
import logging
import matplotlib.pyplot as plt
import numpy as np
import requests
from tqdm.contrib.concurrent import thread_map


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("server-bench")


def get_prompts(n_prompts: int) -> list[str]:
    logger.info("Loading MMLU dataset...")
    ret = datasets.load_dataset("cais/mmlu", "all")["test"]["question"]  # type: ignore
    if n_prompts >= 0:
        ret = ret[:n_prompts]
    return ret


def get_server(path_server: str, path_model: str, path_log: Optional[str], port: int, n_gpu_layers: int, parallel: int, ctx_size: int) -> dict:
    logger.info("Starting the llama.cpp server...")
    address = f"http://localhost:{port}"

    popen_args: list[str] = [
        path_server,
        "--flash-attn",
        "--n-gpu-layers", str(n_gpu_layers),
        "--parallel", str(parallel),
        "--ctx-size", str(parallel * ctx_size),
        "--model", path_model,
        "--port", str(port),
        "--swa-full",  # FIXME performance bad otherwise
        # "--attn-streams",
    ]
    fout = open("bench.log", "w") if path_log is not None else subprocess.DEVNULL
    process = subprocess.Popen(popen_args, stdout=fout, stderr=subprocess.STDOUT)

    n_failures: int = 0
    while True:
        try:
            sleep(1.0)
            exit_code = process.poll()
            if exit_code is not None:
                raise RuntimeError(f"llama.cpp server for {path_model} exited unexpectedly with exit code {exit_code}")
            response = requests.get(f"{address}/health")
            if response.status_code == 200:
                break
        except requests.ConnectionError:
            n_failures += 1
            if n_failures >= 10:
                raise RuntimeError(f"llama.cpp server for {path_model} is not healthy after 10 seconds")

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

    response = session.post(
        f"{server_address}/apply-template",
        json={"messages": [{"role": "user", "content": data["prompt"], "stream": True}]}
    )
    if response.status_code != 200:
        raise RuntimeError(f"Server returned status code {response.status_code}: {response.text}")
    prompt: str = json.loads(response.text)["prompt"]

    json_data: dict = {"prompt": prompt, "seed": data["seed"], "n_predict": data["n_predict"], "stream": True}
    response = session.post(f"{server_address}/completion", json=json_data, stream=True)

    last_valid_line: str = ""
    token_arrival_times: list[float] = []
    for line in response.iter_lines(decode_unicode=True):
        if not line.startswith("data: "):
            continue
        last_valid_line = line
        token_arrival_times.append(time())
    token_arrival_times = token_arrival_times[:-1]

    if response.status_code != 200:
        raise RuntimeError(f"Server returned status code {response.status_code}: {response.text}")
    timings: dict = json.loads(last_valid_line[6:])["timings"]

    return (timings["prompt_ms"], token_arrival_times)


def benchmark(path_server: str, path_model: str, path_log: Optional[str], port: int, n_gpu_layers: int, parallel: int, ctx_size: int, n_prompts: int, n_predict: int):
    num_workers: int = parallel + 1
    prompts: list[str] = get_prompts(n_prompts)

    server: Optional[dict] = None
    session = None
    try:
        server = get_server(path_server, path_model, path_log, port, n_gpu_layers, parallel, ctx_size)
        server_address: str = server["address"]

        adapter = requests.adapters.HTTPAdapter(pool_connections=num_workers, pool_maxsize=num_workers)  # type: ignore
        session = requests.Session()
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        data: list[dict] = []
        for i, p in enumerate(prompts):
            data.append({"session": session, "server_address": server_address, "prompt": p, "n_predict": n_predict, "seed": i})

        logger.info("Getting the prompt lengths...")
        prompt_n = [get_prompt_length(d) for d in data]

        logger.info("Starting the benchmark...\n")
        t0 = time()
        results: list[tuple[int, list[float]]] = thread_map(send_prompt, data, max_workers=num_workers, chunksize=1)
    finally:
        if server is not None:
            server["process"].terminate()
            server["process"].wait()
        if session is not None:
            session.close()

    prompt_ms = []
    token_t = []
    depth_sum: int = 0
    for pn, (pms, tat) in zip(prompt_n, results):
        prompt_ms.append(pms)
        token_t += tat
        n_tokens: int = len(tat)
        depth_sum += n_tokens * pn
        depth_sum += n_tokens * (n_tokens + 1) // 2
    prompt_n = np.array(prompt_n, dtype=np.int64)
    prompt_ms = np.array(prompt_ms, dtype=np.float64)
    token_t = np.array(token_t, dtype=np.float64)

    token_t -= t0
    token_t_last = np.max(token_t)

    logger.info("")
    logger.info(f"Benchmark duration:                {token_t_last:.2f} s")
    logger.info(f"Request throughput:                {n_prompts / token_t_last:.2f} requests/s = {n_prompts / (token_t_last/60):.2f} requests/min")
    logger.info(f"Total prompt length:               {np.sum(prompt_n)} tokens")
    logger.info(f"Average prompt length:             {np.mean(prompt_n):.2f} tokens")
    logger.info(f"Average prompt latency:            {np.mean(prompt_ms):.2f} ms")
    logger.info(f"Average prompt speed:              {np.sum(prompt_n) / (1e-3 * np.sum(prompt_ms)):.2f} tokens/s")
    logger.info(f"Total generated tokens:            {token_t.shape[0]}")
    logger.info(f"Average generation depth:          {depth_sum / token_t.shape[0]:.2f} tokens")
    logger.info(f"Average total generation speed:    {token_t.shape[0] / token_t_last:.2f} tokens/s")
    logger.info(f"Average generation speed per slot: {token_t.shape[0] / (parallel * token_t_last):.2f} tokens/s / slot")

    plt.figure()
    plt.scatter(prompt_n, prompt_ms, s=10.0, marker=".", alpha=0.25)
    plt.xlim(0, 1.05 * np.max(prompt_n))
    plt.ylim(0, 1.05 * np.max(prompt_ms))
    plt.title(path_model)
    plt.xlabel("Prompt length [tokens]")
    plt.ylabel("Time to first token [ms]")
    plt.savefig("prompt_time.png", dpi=240)

    bin_max = np.ceil(token_t_last) + 1
    plt.figure()
    plt.hist(token_t, np.arange(0, bin_max))
    plt.xlim(0, bin_max + 1)
    plt.title(path_model)
    plt.xlabel("Time [s]")
    plt.ylabel("Num. tokens generated per second")
    plt.savefig("gen_rate.png", dpi=240)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tool for benchmarking the throughput of the llama.cpp HTTP server. "
        "Results are printed to console and visualized as plots (saved to current working directory).")
    parser.add_argument("--path_server", type=str, default="llama-server", help="Path to the llama.cpp server binary")
    parser.add_argument("--path_model", type=str, required=True, help="Path to the model to use for the benchmark")
    parser.add_argument("--path_log", type=str, default=None, help="Path to the model to use for the benchmark")
    parser.add_argument("--port", type=int, default=18725, help="Port to use for the server during the benchmark")
    parser.add_argument("--n_gpu_layers", type=int, default=999, help="Number of GPU layers for the server")
    parser.add_argument("--parallel", type=int, default=16, help="Number of slots for the server")
    parser.add_argument("--ctx_size", type=int, default=4096, help="Server context size per slot")
    parser.add_argument("--n_prompts", type=int, default=1000, help="Number of prompts to evaluate")
    parser.add_argument("--n_predict", type=int, default=2048, help="Max. number of tokens to predict per prompt")
    args = parser.parse_args()
    benchmark(**vars(args))
