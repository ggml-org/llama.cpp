import subprocess
import sys
import os
import re

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from tune import parse_args, main

def run_str(run_options, model_path, binary_path):
        run_opts = " ".join(run_options.values())
        return f"{binary_path} -m {model_path} -p 'Hello, how are you?' -n 1 {run_opts}"

def run_binary(run_options_str):
        process = subprocess.run(run_options_str,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,
                                 shell=True,
                                 check=True,
                                 )
        if process.returncode != 0:
            raise Exception(f"Error running: '{run_options_str}':\n{process.stderr}")

        # Parse timing information from stderr
        stderr_text = process.stderr.decode()

        # Updated regex patterns for llama-simple output
        prompt_eval_time_pattern = r"prompt eval time\s*=\s*([\d.]+)\s*ms"
        eval_time_pattern = r"eval time\s*=\s*([\d.]+)\s*ms"

        prompt_match = re.search(prompt_eval_time_pattern, stderr_text)
        eval_match = re.search(eval_time_pattern, stderr_text)

        if prompt_match and eval_match:
            prompt_eval_time = float(prompt_match.group(1)) / 1000  # Convert to seconds
            eval_time = float(eval_match.group(1)) / 1000  # Convert to seconds
        else:
            # Fallback: look for any timing patterns
            print("Warning: Could not parse timing info, using fallback")
            print("STDERR:", stderr_text)
            return 1000  # High penalty for failed parsing

        print("prompt eval time:", prompt_eval_time)
        print("eval time:", eval_time)

        return eval_time

if __name__ == "__main__":
    args = parse_args(default_bin='./build/bin/llama-cli')
    # Define runtime options to optimize - Core Performance Parameters
    run_options_list = [
        # 1. Batch Processing Parameters (most critical for throughput)
        ("--batch-size", ["--batch-size 31", "--batch-size 64", "--batch-size 128", "--batch-size 256", "--batch-size 512", "--batch-size 1024", "--batch-size 2048"]),
        ("--ubatch-size", ["--ubatch-size 32", "--ubatch-size 64", "--ubatch-size 128", "--ubatch-size 256", "--ubatch-size 512"]),

        # 2. Context and Memory Parameters
        ("--ctx-size", ["-c 512", "-c 1024", "-c 2048", "-c 4096", "-c 8192"]),
        ("--defrag-thold", ["--defrag-thold -1", "--defrag-thold 0.1", "--defrag-thold 0.2", "--defrag-thold 0.5"]),

        # 3. GPU Offloading Parameters (critical for GPU performance)
        # Set range to a value that makes sense for your model
        ("--n-gpu-layers", [f"--n-gpu-layers {i}" for i in range(args.ngl)]),

        # 4. CPU Optimization Parameters
        ("--threads", ["-t 4", "-t 8", "-t 12", "-t 16"]),
        # ("--prio", ["--prio 0", "--prio 1", "--prio 2"]),

        # 5. Memory and Caching Parameters
        # ("--use-mmap", ["", "--no-mmap"]),
        ("--use-mlock", ["--mlock", ""]),
        ("--kv-unified", ["--kv-unified", ""]),

        # 6. Advanced Performance Features
        ("--flash-attn", ["--flash-attn", ""]),
        # ("--no-kv-offload", ["--no-kv-offload", ""]),  # Empty string means don't use the flag

        # Keep seed constant for reproducible results
        ("--seed", ["-s 42"]),
    ]
    main(args, run_str, run_binary, run_options_list)
