#!/usr/bin/env uv run
'''
    Simplistic tool call benchmarks for llama-server and ollama.

    Essentially runs the tests at server/tools/server/tests/unit/test_tool_call.py N times, at different temperatures and on different backends (current llama-server, baseline llama-server and ollama),
    and plots the results of multiple runs (from same .jsonl file or multiple ones) as a success rate heatmap.

    Simple usage example:

        cmake -B build -DLLAMA_CURL=1 && cmake --build build --config Release -j -t llama-server

        export LLAMA_SERVER_BIN_PATH=$PWD/build/bin/llama-server
        export LLAMA_CACHE=${LLAMA_CACHE:-$HOME/Library/Caches/llama.cpp}

        ./scripts/tool_bench.py run --n 10 --temp -1 --temp 0 --temp 1 --temp 2 --temp 5 --llama-baseline $PWD/buildMaster/bin/llama-server --output qwen14b.jsonl --hf bartowski/Qwen2.5-14B-Instruct-GGUF:Q4_K_L
        ./scripts/tool_bench.py run --n 30 --temp -1 --temp 0 --temp 1 --model "Qwen 2.5 1.5B Q4_K_M"      --output qwen1.5b.jsonl  --hf bartowski/Qwen2.5-1.5B-Instruct-GGUF      --ollama qwen2.5:1.5b-instruct-q4_K_M
        ./scripts/tool_bench.py run --n 30 --temp -1 --temp 0 --temp 1 --model "Qwen 2.5 Coder 7B Q4_K_M"  --output qwenc7b.jsonl   --hf bartowski/Qwen2.5-Coder-7B-Instruct-GGUF  --ollama qwen2.5-coder:7b

        ./scripts/tool_bench.py plot *.jsonl                         # Opens window w/ heatmap
        ./scripts/tool_bench.py plot qwen*.jsonl  --output qwen.png  # Saves heatmap to qwen.png

    (please see ./scripts/tool_bench.sh for a more complete example)
'''
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pytest",
#     "pandas",
#     "matplotlib",
#     "seaborn",
#     "requests",
#     "wget",
#     "typer",
# ]
# ///
from contextlib import contextmanager
from pathlib import Path
import re
from statistics import mean, median
from typing import Annotated, Dict, List, Optional, Tuple
import atexit
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import subprocess
import sys
import time
import typer

sys.path.insert(0, Path(__file__).parent.parent.as_posix())
if True:
    from tools.server.tests.utils import ServerProcess
    from tools.server.tests.unit.test_tool_call import TIMEOUT_SERVER_START, do_test_calc_result, do_test_hello_world, do_test_weather


@contextmanager
def scoped_server(sp: ServerProcess):
    def stop():
        nonlocal sp
        if sp is not None:
            sp.stop()
            sp = None # type: ignore
    atexit.register(stop)
    yield sp
    stop()


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = typer.Typer()


@app.command()
def plot(files: List[Path], output: Optional[Path] = None, test_regex: Optional[str] = None, server_regex: Optional[str] = None):

    lines: List[Dict] = []
    for file in files:
        if not file.exists():
            logger.error(f"File not found: {file}")
            continue

        try:
            with file.open() as f:
                raw_data = f.read()
            logger.info(f"Reading {file} ({len(raw_data)} bytes)")

            for line_num, line in enumerate(raw_data.split('\n'), 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    lines.append(record)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON at {file}:{line_num} - {e}")
        except Exception as e:
            logger.error(f"Error processing {file}: {e}")

    if not lines:
        raise Exception("No valid data was loaded")

    data_dict: Dict[Tuple, float] = {}
    models: List[str] = []
    temps = set()
    tests = set()
    server_names = set()
    total_counts = set()
    for rec in lines:
        try:
            model = rec["model"]
            temp = rec["temp"]
            server_name = rec["server_name"]
            test = rec["test"]
            success = rec["success_ratio"]
            success_count = rec["success_count"]
            failure_count = rec["failure_count"]
            total_count = success_count + failure_count
            total_counts.add(total_count)

            if test_regex and not re.search(test_regex, test):
                continue

            if server_regex and not re.search(server_regex, server_name):
                continue

            data_dict[(model, temp, server_name, test)] = success

            if model not in models:
                models.append(model)
            temps.add(temp)
            tests.add(test)
            server_names.add(server_name)

        except KeyError as e:
            logger.warning(f"Missing required field in record: {e}")

    if len(total_counts) > 1:
        logger.warning(f"Total counts are not consistent: {total_counts}")

    # Sort the collected values
    temps = list(sorted(temps, key=lambda x: x if x is not None else -1))
    tests = list(sorted(tests))
    server_names = list(sorted(server_names))

    logger.info(f"Processed {len(lines)} lines")
    logger.info(f"Found {len(data_dict)} valid data points")
    logger.info(f"Models: {models}")
    logger.info(f"Temperatures: {temps}")
    logger.info(f"Tests: {tests}")
    logger.info(f"Servers: {server_names}")

    matrix: list[list[float]] = []
    index: list[str] = []

    all_cols = [
        (server_name, test)
        for server_name in server_names
        for test in tests
    ]
    for model in models:
        for temp in temps:
            index.append(f"{model} @ {temp}")
            row_vals = [
                data_dict.get((model, temp, server_name, test), np.nan)
                for server_name, test in all_cols
            ]
            matrix.append(row_vals)

    columns: list[str] = [f"{server_name}\n{test}" for server_name, test in all_cols]

    df = pd.DataFrame(matrix, index=np.array(index), columns=np.array(columns))

    plt.figure(figsize=(12, 6))

    sns.heatmap(
        df, annot=True, cmap="RdYlGn", vmin=0.0, vmax=1.0, cbar=True, fmt=".2f", center=0.5, square=True, linewidths=0.5,
        cbar_kws={"label": "Success Ratio"},
    )

    plt.title(f"Tool Call Bench (n = {str(min(total_counts)) if len(total_counts) == 1 else f'{min(total_counts)}-{max(total_counts)}'})\nSuccess Ratios by Server & Test", pad=20)
    plt.xlabel("Server & Test", labelpad=10)
    plt.ylabel("Model @ Temperature", labelpad=10)

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {output}")
    else:
        plt.show()


@app.command()
def run(
    output: Annotated[Path, typer.Option(help="Output JSON file")],
    model: Annotated[Optional[str], typer.Option(help="Name of the model to test (server agnostic)")] = None,
    hf: Annotated[Optional[str], typer.Option(help="GGUF huggingface model repo id (+ optional quant) to test w/ llama-server")] = None,
    chat_template: Annotated[Optional[str], typer.Option(help="Chat template override for llama-server")] = None,
    chat_template_file: Annotated[Optional[str], typer.Option(help="Chat template file override for llama-server")] = None,
    ollama: Annotated[Optional[str], typer.Option(help="Ollama model tag to test")] = None,
    llama_baseline: Annotated[Optional[str], typer.Option(help="llama-server baseline binary path to use as baseline")] = None,
    n: Annotated[int, typer.Option(help="Number of times to run each test")] = 10,
    temp: Annotated[Optional[List[float]], typer.Option(help="Set of temperatures to test")] = None,
    top_p: Annotated[Optional[float], typer.Option(help="top_p")] = None,
    top_k: Annotated[Optional[int], typer.Option(help="top_k")] = None,
    ctk: Annotated[Optional[str], typer.Option(help="ctk")] = None,
    ctv: Annotated[Optional[str], typer.Option(help="ctv")] = None,
    fa: Annotated[Optional[bool], typer.Option(help="fa")] = None,
    seed: Annotated[Optional[int], typer.Option(help="Random seed")] = None,
    port: Annotated[int, typer.Option(help="llama-server port")] = 8084,
    force: Annotated[bool, typer.Option(help="Force overwrite of output file")] = False,
    append: Annotated[bool, typer.Option(help="Append to output file")] = False,

    test_hello_world: Annotated[bool, typer.Option(help="Whether to run the hello world test")] = True,
    test_weather: Annotated[bool, typer.Option(help="Whether to run the weather test")] = True,
    test_calc_result: Annotated[bool, typer.Option(help="Whether to run the calc result test")] = False,
):
    # Check only one of output and append

    n_predict = 512 # High because of DeepSeek R1
    # n_ctx = 8192
    n_ctx = 2048

    if model is None:
        if hf is not None:
            model = hf.split("/")[-1]
        elif ollama is not None:
            model = ollama

    assert force or append or not output.exists(), f"Output file already exists: {output}; use --force to overwrite"

    with output.open('a' if append else 'w') as output_file:

        def run(server: ServerProcess, *, server_name: str, model_id: str, temp: Optional[float] = None, output_kwargs={}, request_kwargs={}):
            request_kwargs = {**request_kwargs}
            if temp is not None:
                request_kwargs['temperature'] = temp
            if top_p is not None:
                request_kwargs['top_p'] = top_p
            if top_k is not None:
                request_kwargs['top_k'] = top_k
            if seed is not None:
                request_kwargs['seed'] = seed

            request_kwargs['cache_prompt'] = False

            tests = {}
            if test_hello_world:
                tests["hello world"] = lambda server: do_test_hello_world(server, **request_kwargs)
            if test_weather:
                tests["weather"] = lambda server: do_test_weather(server, **request_kwargs)
            if test_calc_result:
                tests["calc result"] = lambda server: do_test_calc_result(server, None, 512, **request_kwargs)

            for test_name, test in tests.items():
                success_count = 0
                failure_count = 0
                failures = []
                success_times = []
                failure_times = []
                logger.info(f"Running {test_name} ({server_name}, {model}): ")
                for i in range(n):
                    start_time = time.time()

                    def elapsed():
                        return time.time() - start_time

                    try:
                        test(server)
                        success_times.append(elapsed())
                        success_count += 1
                        logger.info('success')
                    except Exception as e:
                        logger.error(f'failure: {e}')
                        failure_count += 1
                        failure_times.append(elapsed())
                        failures.append(str(e))
                        # import traceback
                        # traceback.print_exc()
                output_file.write(json.dumps({**output_kwargs, **dict(
                    model=model,
                    server_name=server_name,
                    model_id=model_id,
                    test=test_name,
                    temp=t,
                    top_p=top_p,
                    top_k=top_k,
                    ctk=ctk,
                    ctv=ctv,
                    seed=seed,
                    success_ratio=float(success_count) / n,
                    avg_time=mean(success_times + failure_times),
                    median_time=median(success_times + failure_times),
                    success_count=success_count,
                    success_times=success_times,
                    failure_count=failure_count,
                    failure_times=failure_times,
                    failures=list(set(failures)),
                )}) + '\n')
                output_file.flush()

        for t in [None] if temp is None else [t if t >= 0 else None for t in temp]:
            if hf is not None:

                servers: list[Tuple[str, Optional[str]]] = [('llama-server', None)]
                if llama_baseline is not None:
                    servers.append(('llama-server (baseline)', llama_baseline))

                for server_name, server_path in servers:
                    server = ServerProcess()
                    server.n_ctx = n_ctx
                    server.n_slots = 1
                    server.jinja = True
                    server.ctk = ctk
                    server.ctv = ctv
                    server.fa = fa
                    server.n_predict = n_predict
                    server.model_hf_repo = hf
                    server.model_hf_file = None
                    server.chat_template = chat_template
                    server.chat_template_file = chat_template_file
                    server.server_path = server_path
                    if port is not None:
                        server.server_port = port
                    # server.debug = True

                    with scoped_server(server):
                        server.start(timeout_seconds=TIMEOUT_SERVER_START)
                        for ignore_chat_grammar in [False]:
                            run(
                                server,
                                server_name=server_name,
                                model_id=hf,
                                temp=t,
                                output_kwargs=dict(
                                    chat_template=chat_template,
                                    chat_template_file=chat_template_file,
                                ),
                                request_kwargs=dict(
                                    ignore_chat_grammar=ignore_chat_grammar,
                                ),
                            )

            if ollama is not None:
                server = ServerProcess()
                server.server_port = 11434
                server.server_host = "localhost"
                subprocess.check_call(["ollama", "pull", ollama])

                with scoped_server(server):
                    run(
                        server,
                        server_name="ollama",
                        model_id=ollama,
                        temp=t,
                        output_kwargs=dict(
                            chat_template=None,
                            chat_template_file=None,
                        ),
                        request_kwargs=dict(
                            model=ollama,
                            max_tokens=n_predict,
                            num_ctx = n_ctx,
                        ),
                    )


if __name__ == "__main__":
    app()
