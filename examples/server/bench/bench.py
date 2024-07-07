from __future__ import annotations

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from contextlib import closing
from datetime import datetime

import matplotlib
import matplotlib.dates
import matplotlib.pyplot as plt
import requests
from statistics import mean


def main(args_in: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Start server benchmark scenario")
    parser.add_argument("--name", type=str, help="Bench name", required=True)
    parser.add_argument("--runner-label", type=str, help="Runner label", required=True)
    parser.add_argument("--branch", type=str, help="Branch name", default="detached")
    parser.add_argument("--commit", type=str, help="Commit name", default="dirty")
    parser.add_argument("--host", type=str, help="Server listen host", default="0.0.0.0")
    parser.add_argument("--port", type=int, help="Server listen host", default="8080")
    parser.add_argument("--model-path-prefix", type=str, help="Prefix where to store the model files", default="models")
    parser.add_argument("--n-prompts", type=int,
                        help="SERVER_BENCH_N_PROMPTS: total prompts to randomly select in the benchmark", required=True)
    parser.add_argument("--max-prompt-tokens", type=int,
                        help="SERVER_BENCH_MAX_PROMPT_TOKENS: maximum prompt tokens to filter out in the dataset",
                        required=True)
    parser.add_argument("--max-tokens", type=int,
                        help="SERVER_BENCH_MAX_CONTEXT: maximum context size of the completions request to filter out in the dataset: prompt + predicted tokens",
                        required=True)
    parser.add_argument("--hf-repo", type=str, help="Hugging Face model repository", required=True)
    parser.add_argument("--hf-file", type=str, help="Hugging Face model file", required=True)
    parser.add_argument("-ngl", "--n-gpu-layers", type=int, help="layers to the GPU for computation", required=True)
    parser.add_argument("--ctx-size", type=int, help="Set the size of the prompt context", required=True)
    parser.add_argument("--parallel", type=int, help="Set the number of slots for process requests", required=True)
    parser.add_argument("--batch-size", type=int, help="Set the batch size for prompt processing", required=True)
    parser.add_argument("--ubatch-size", type=int, help="physical maximum batch size", required=True)
    parser.add_argument("--scenario", type=str, help="Scenario to run", required=True)
    parser.add_argument("--duration", type=str, help="Bench scenario", required=True)

    args = parser.parse_args(args_in)

    start_time = time.time()

    # Start the server and performance scenario
    try:
        server_process = start_server(args)
    except Exception:
        print("bench: server start error :")
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)

    # start the benchmark
    iterations = 0
    data = {}
    try:
        start_benchmark(args)

        with open("results.github.env", 'w') as github_env:
            # parse output
            with open('k6-results.json', 'r') as bench_results:
                # Load JSON data from file
                data = json.load(bench_results)
                for metric_name in data['metrics']:
                    for metric_metric in data['metrics'][metric_name]:
                        value = data['metrics'][metric_name][metric_metric]
                        if isinstance(value, float) or isinstance(value, int):
                            value = round(value, 2)
                            data['metrics'][metric_name][metric_metric]=value
                            github_env.write(
                                f"{escape_metric_name(metric_name)}_{escape_metric_name(metric_metric)}={value}\n")
                iterations = data['root_group']['checks']['success completion']['passes']

    except Exception:
        print("bench: error :")
        traceback.print_exc(file=sys.stdout)

    # Stop the server
    if server_process:
        try:
            print(f"bench: shutting down server pid={server_process.pid} ...")
            if os.name == 'nt':
                interrupt = signal.CTRL_C_EVENT
            else:
                interrupt = signal.SIGINT
            server_process.send_signal(interrupt)
            server_process.wait(0.5)

        except subprocess.TimeoutExpired:
            print(f"server still alive after 500ms, force-killing pid={server_process.pid} ...")
            server_process.kill()  # SIGKILL
            server_process.wait()

        while is_server_listening(args.host, args.port):
            time.sleep(0.1)

    title = (f"llama.cpp {args.name} on {args.runner_label}\n "
             f"duration={args.duration} {iterations} iterations")
    xlabel = (f"{args.hf_repo}/{args.hf_file}\n"
              f"parallel={args.parallel} ctx-size={args.ctx_size} ngl={args.n_gpu_layers} batch-size={args.batch_size} ubatch-size={args.ubatch_size} pp={args.max_prompt_tokens} pp+tg={args.max_tokens}\n"
              f"branch={args.branch} commit={args.commit}")

    # Prometheus
    end_time = time.time()
    prometheus_metrics = {}
    if is_server_listening("0.0.0.0", 9090):
        metrics = ['prompt_tokens_seconds', 'predicted_tokens_seconds',
                   'kv_cache_usage_ratio', 'requests_processing', 'requests_deferred']

        for metric in metrics:
            resp = requests.get(f"http://localhost:9090/api/v1/query_range",
                                params={'query': 'llamacpp:' + metric, 'start': start_time, 'end': end_time, 'step': 2})

            with open(f"{metric}.json", 'w') as metric_json:
                metric_json.write(resp.text)

            if resp.status_code != 200:
                print(f"bench: unable to extract prometheus metric {metric}: {resp.text}")
            else:
                metric_data = resp.json()
                values = metric_data['data']['result'][0]['values']
                timestamps, metric_values = zip(*values)
                metric_values = [float(value) for value in metric_values]
                prometheus_metrics[metric] = metric_values
                timestamps_dt = [str(datetime.fromtimestamp(int(ts))) for ts in timestamps]
                plt.figure(figsize=(16, 10), dpi=80)
                plt.plot(timestamps_dt, metric_values, label=metric)
                plt.xticks(rotation=0, fontsize=14, horizontalalignment='center', alpha=.7)
                plt.yticks(fontsize=12, alpha=.7)

                ylabel = f"llamacpp:{metric}"
                plt.title(title,
                          fontsize=14, wrap=True)
                plt.grid(axis='both', alpha=.3)
                plt.ylabel(ylabel, fontsize=22)
                plt.xlabel(xlabel, fontsize=14, wrap=True)
                plt.gca().xaxis.set_major_locator(matplotlib.dates.MinuteLocator())
                plt.gca().xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y-%m-%d %H:%M:%S"))
                plt.gcf().autofmt_xdate()

                # Remove borders
                plt.gca().spines["top"].set_alpha(0.0)
                plt.gca().spines["bottom"].set_alpha(0.3)
                plt.gca().spines["right"].set_alpha(0.0)
                plt.gca().spines["left"].set_alpha(0.3)

                # Save the plot as a jpg image
                plt.savefig(f'{metric}.jpg', dpi=60)
                plt.close()

                # Mermaid format in case images upload failed
                with open(f"{metric}.mermaid", 'w') as mermaid_f:
                    mermaid = (
                    f"""---
config:
    xyChart:
        titleFontSize: 12
        width: 900
        height: 600
    themeVariables:
        xyChart:
            titleColor: "#000000"
---
xychart-beta
    title "{title}"
    y-axis "llamacpp:{metric}"
    x-axis "llamacpp:{metric}" {int(min(timestamps))} --> {int(max(timestamps))}
    line [{', '.join([str(round(float(value), 2)) for value in metric_values])}]
                    """)
                    mermaid_f.write(mermaid)

    # 140 chars max for commit status description
    bench_results = {
        "i": iterations,
        "req": {
            "p95": round(data['metrics']["http_req_duration"]["p(95)"], 2),
            "avg": round(data['metrics']["http_req_duration"]["avg"], 2),
        },
        "pp": {
            "p95": round(data['metrics']["llamacpp_prompt_processing_second"]["p(95)"], 2),
            "avg": round(data['metrics']["llamacpp_prompt_processing_second"]["avg"], 2),
            "0": round(mean(prometheus_metrics['prompt_tokens_seconds']), 2),
        },
        "tg": {
            "p95": round(data['metrics']["llamacpp_tokens_second"]["p(95)"], 2),
            "avg": round(data['metrics']["llamacpp_tokens_second"]["avg"], 2),
            "0": round(mean(prometheus_metrics['predicted_tokens_seconds']), 2),
        },
    }
    with open("results.github.env", 'a') as github_env:
        github_env.write(f"BENCH_RESULTS={json.dumps(bench_results, indent=None, separators=(',', ':') )}\n")
        github_env.write(f"BENCH_ITERATIONS={iterations}\n")

        title = title.replace('\n', ' ')
        xlabel = xlabel.replace('\n', ' ')
        github_env.write(f"BENCH_GRAPH_TITLE={title}\n")
        github_env.write(f"BENCH_GRAPH_XLABEL={xlabel}\n")


def start_benchmark(args):
    k6_path = './k6'
    if 'BENCH_K6_BIN_PATH' in os.environ:
        k6_path = os.environ['BENCH_K6_BIN_PATH']
    k6_args = [
        'run', args.scenario,
        '--no-color',
    ]
    k6_args.extend(['--duration', args.duration])
    k6_args.extend(['--iterations', args.n_prompts])
    k6_args.extend(['--vus', args.parallel])
    k6_args.extend(['--summary-export', 'k6-results.json'])
    args = f"SERVER_BENCH_N_PROMPTS={args.n_prompts} SERVER_BENCH_MAX_PROMPT_TOKENS={args.max_prompt_tokens} SERVER_BENCH_MAX_CONTEXT={args.max_tokens} "
    args = args + ' '.join([str(arg) for arg in [k6_path, *k6_args]])
    print(f"bench: starting k6 with: {args}")
    k6_completed = subprocess.run(args, shell=True, stdout=sys.stdout, stderr=sys.stderr)
    if k6_completed.returncode != 0:
        raise Exception("bench: unable to run k6")


def start_server(args):
    server_process = start_server_background(args)

    attempts = 0
    max_attempts = 20
    if 'GITHUB_ACTIONS' in os.environ:
        max_attempts *= 2

    while not is_server_listening(args.host, args.port):
        attempts += 1
        if attempts > max_attempts:
            assert False, "server not started"
        print(f"bench:     waiting for server to start ...")
        time.sleep(0.5)

    print("bench: server started.")
    return server_process


def start_server_background(args):
    # Start the server
    server_path = '../../../build/bin/llama-server'
    if 'LLAMA_SERVER_BIN_PATH' in os.environ:
        server_path = os.environ['LLAMA_SERVER_BIN_PATH']
    server_args = [
        '--host', args.host,
        '--port', args.port,
    ]
    model_file = args.model_path_prefix + os.path.sep + args.hf_file
    model_dir  = os.path.dirname(model_file)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    server_args.extend(['--model', model_file])
    server_args.extend(['--hf-repo', args.hf_repo])
    server_args.extend(['--hf-file', args.hf_file])
    server_args.extend(['--n-gpu-layers', args.n_gpu_layers])
    server_args.extend(['--ctx-size', args.ctx_size])
    server_args.extend(['--parallel', args.parallel])
    server_args.extend(['--batch-size', args.batch_size])
    server_args.extend(['--ubatch-size', args.ubatch_size])
    server_args.extend(['--n-predict', args.max_tokens * 2])
    server_args.extend(['--defrag-thold', "0.1"])
    server_args.append('--cont-batching')
    server_args.append('--metrics')
    server_args.append('--flash-attn')
    server_args.extend(['--log-format', "text"])
    args = [str(arg) for arg in [server_path, *server_args]]
    print(f"bench: starting server with: {' '.join(args)}")
    pkwargs = {
        'stdout': subprocess.PIPE,
        'stderr': subprocess.PIPE
    }
    server_process = subprocess.Popen(
        args,
        **pkwargs)  # pyright: ignore[reportArgumentType, reportCallIssue]

    def server_log(in_stream, out_stream):
        for line in iter(in_stream.readline, b''):
            print(line.decode('utf-8'), end='', file=out_stream)

    thread_stdout = threading.Thread(target=server_log, args=(server_process.stdout, sys.stdout))
    thread_stdout.start()
    thread_stderr = threading.Thread(target=server_log, args=(server_process.stderr, sys.stderr))
    thread_stderr.start()

    return server_process


def is_server_listening(server_fqdn, server_port):
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        result = sock.connect_ex((server_fqdn, server_port))
        _is_server_listening = result == 0
        if _is_server_listening:
            print(f"server is listening on {server_fqdn}:{server_port}...")
        return _is_server_listening


def escape_metric_name(metric_name):
    return re.sub('[^A-Z0-9]', '_', metric_name.upper())


if __name__ == '__main__':
    main()
