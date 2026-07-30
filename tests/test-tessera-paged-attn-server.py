#!/usr/bin/env python3
"""Opt-in end-to-end validation for Tessera's normal server decode graph.

This is intentionally not registered as a default CTest: it needs a local
Gemma/Qwen GGUF and enough unified memory to load it.  CI or a developer runs
it explicitly with --server and --model.  The test is only successful when a
real two-token server completion produces the graph-selection log marker.
"""

import argparse
import json
import os
import pathlib
import subprocess
import tempfile
import time
import urllib.error
import urllib.request


def write_receipt(path, status, detail=""):
    if path is not None:
        pathlib.Path(path).write_text(json.dumps({"status": status, "detail": detail}) + "\n")


def request(url, payload=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST" if data else "GET")
    if data:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=5) as response:
        return response.status, response.read().decode("utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True, type=pathlib.Path)
    parser.add_argument("--model", required=True, type=pathlib.Path)
    parser.add_argument("--port", type=int, default=18097)
    parser.add_argument("--receipt", type=pathlib.Path)
    args = parser.parse_args()

    assert args.server.is_file(), args.server
    assert args.model.is_file(), args.model

    with tempfile.TemporaryDirectory(prefix="tessera-paged-server-") as directory:
        log_path = pathlib.Path(directory) / "server.log"
        env = os.environ.copy()
        env["TESSERA_PAGED_ATTN"] = "1"
        env["TESSERA_PAGED_ATTN_DEBUG"] = "1"
        command = [
            str(args.server), "-m", str(args.model), "--spec-type", "none",
            "--ctx-size", "256", "--cache-type-k", "f16", "--cache-type-v", "f16",
            "--kv-unified", "--parallel", "1", "--n-gpu-layers", "999",
            "--host", "127.0.0.1", "--port", str(args.port), "--no-webui",
            "--log-file", str(log_path),
        ]
        process = subprocess.Popen(command, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    log_tail = log_path.read_text(errors="replace")[-4000:] if log_path.exists() else "<no server log>"
                    raise RuntimeError(
                        f"llama-server exited during startup (exit={process.returncode})\\n{log_tail}")
                try:
                    status, _ = request(f"http://127.0.0.1:{args.port}/health")
                    if status == 200:
                        break
                except urllib.error.URLError:
                    pass
                time.sleep(0.5)
            else:
                raise TimeoutError("llama-server did not become ready")

            status, response = request(
                f"http://127.0.0.1:{args.port}/completion",
                {"prompt": "Hello", "n_predict": 2, "temperature": 0},
            )
            assert status == 200, response
            assert json.loads(response)["tokens_predicted"] == 2, response

            log_text = log_path.read_text(errors="replace")
            assert "TESSERA_PAGED_ATTN_SELECTED:" in log_text, log_text[-4000:]
            write_receipt(args.receipt, "passed")
        except Exception as error:
            log_tail = log_path.read_text(errors="replace")[-4000:] if log_path.exists() else "<no server log>"
            write_receipt(args.receipt, "failed", f"{error}\\nserver log tail:\\n{log_tail}")
            raise
        finally:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=20)


if __name__ == "__main__":
    main()
