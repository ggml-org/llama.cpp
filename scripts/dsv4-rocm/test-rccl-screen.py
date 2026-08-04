#!/usr/bin/env python3
"""Fixtures for the predeclared DSV4 RCCL raw-TG screen."""
import json
import os
import pathlib
import subprocess
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
WRAPPER = HERE / "screen-rccl-tg.sh"
COMPARE = HERE / "compare-rccl-tg.py"
DEPTHS = [16384, 32768, 65536]


def run(command, *, env=None, ok=True):
    result = subprocess.run(command, text=True, capture_output=True, env=env)
    if ok and result.returncode != 0:
        raise RuntimeError(f"command failed: {command}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    if not ok and result.returncode == 0:
        raise RuntimeError(f"command unexpectedly passed: {command}")
    return result


def write_capture(path, non_json_lines):
    non_json_text = "".join(line if line.endswith("\n") else line + "\n" for line in non_json_lines)
    json_text = "{}\n" * len(DEPTHS)
    raw_text = non_json_text + json_text
    (path / "bench.stdout.log").write_text(raw_text)
    (path / "bench.stdout-nonjson.log").write_text(non_json_text)
    (path / "result.jsonl").write_text(json_text)
    (path / "result-completed-at.ns").write_text("1\n2\n3\n")
    (path / "stdout-classification.json").write_text(json.dumps({
        "schema_version": 1,
        "consumer_success": True,
        "total_lines": len(non_json_lines) + len(DEPTHS),
        "total_bytes": len(raw_text.encode()),
        "json_lines": len(DEPTHS),
        "non_json_lines": len(non_json_lines),
        "blank_lines": 0,
        "malformed_json_like_lines": 0,
        "unterminated_final_data": False,
        "max_non_json_lines": 4096,
        "excessive_non_json_output": False,
        "raw_stream_preserved": True,
        "json_completion_timestamps": len(DEPTHS),
    }))


def make_run(root, name, label, medians, digest="a" * 64):
    path = root / name
    path.mkdir()
    algorithm, protocol = {
        "auto": (None, None), "tree-ll": ("Tree", "LL"), "ring-ll": ("Ring", "LL"),
    }[label]
    records = []
    for depth, median in zip(DEPTHS, medians):
        records.append({
            "depth": depth, "median_ts": median, "mad_over_median": 0.01,
            "stable": True, "accepted_repetitions": 5,
        })
    (path / "summary.json").write_text(json.dumps({
        "complete": True, "stable": True, "expected_depths": DEPTHS, "seen_depths": DEPTHS,
        "expected_gen": 32, "expected_raw_repetitions": 6, "discard_first": 1,
        "accepted_repetitions": 5, "records": records,
    }))
    (path / "contract.json").write_text(json.dumps({
        "target_only": True, "speculative_flags": [], "depths": DEPTHS, "n_gen": 32,
        "raw_repetitions": 6, "discard_first": 1, "depth_state_api": "context",
        "model_hash_mode": "full", "stdout_capture": {
            "schema_version": 1, "raw_stream": "bench.stdout.log",
            "non_json_stream": "bench.stdout-nonjson.log",
            "classification": "stdout-classification.json", "max_non_json_lines": 4096,
        }, "communication_candidate": {
            "label": label, "backend": "nccl", "hip_graphs": "1", "runtime_graph_disable": None,
            "algorithm": algorithm, "protocol": protocol,
            "min_channels": None, "max_channels": None, "debug": "INFO", "debug_subsys": "ENV",
        },
    }))
    (path / "status.txt").write_text(
        "process_exit_code=0\nstderr_consumer_exit_code=0\nstdout_consumer_exit_code=0\ntruncated=0\n"
    )
    source_digest = "c" * 64
    (path / "manifest.txt").write_text(
        f"{source_digest}  {path / 'source.patch'}\n"
        f"{source_digest}  {path / 'source-status.txt'}\n"
        f"{source_digest}  {path / 'untracked-files.sha256'}\n"
        f"{digest}  /model/shard.gguf\n"
        "GGML_HIP_GRAPHS:BOOL=ON\n"
    )
    (path / "bench.log").write_text("")
    acknowledgements = []
    if algorithm is not None:
        acknowledgements = [
            f"host:1:2 [0] NCCL INFO NCCL_ALGO set by environment to {algorithm}",
            f"host:1:2 [0] NCCL INFO NCCL_PROTO set by environment to {protocol}",
        ]
    write_capture(path, acknowledgements)
    return path


def main():
    with tempfile.TemporaryDirectory() as temp:
        root = pathlib.Path(temp)
        fake_root = root / "repo"
        fake_script = fake_root / "scripts/dsv4-rocm/run-tg.sh"
        fake_script.parent.mkdir(parents=True)
        fake_script.write_text("""#!/usr/bin/env bash
set -Eeuo pipefail
for name in DSV4_RCCL_CANDIDATE DSV4_TG_DEPTHS DSV4_TG_N_GEN DSV4_TG_REPS DSV4_TG_DISCARD_FIRST DSV4_TG_STDOUT_MAX_NON_JSON_LINES DSV4_HASH_MODE DSV4_TG_PROFILE GGML_HIP_GRAPHS GGML_CUDA_DISABLE_GRAPHS NCCL_ALGO NCCL_PROTO NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS NCCL_DEBUG NCCL_DEBUG_SUBSYS; do
    if declare -p "$name" >/dev/null 2>&1; then printf '%s=%s\\n' "$name" "${!name}"; else printf '%s=<unset>\\n' "$name"; fi
done
printf 'args='; printf '%s ' "$@"; printf '\\n'
""")
        fake_script.chmod(0o755)
        clean_env = {
            k: v for k, v in os.environ.items()
            if not (k.startswith("NCCL_") or k.startswith("RCCL_") or k == "GGML_CUDA_DISABLE_GRAPHS")
        }
        clean_env["DSV4_ROOT_DIR"] = str(fake_root)
        clean_env["DSV4_TG_DEPTHS"] = "1"  # Must be overridden, not inherited.
        tree = run([str(WRAPPER), "tree-ll", "--dry-run"], env=clean_env).stdout
        required = [
            "DSV4_RCCL_CANDIDATE=tree-ll", "DSV4_TG_DEPTHS=16384,32768,65536",
            "DSV4_TG_N_GEN=32", "DSV4_TG_REPS=6", "DSV4_TG_DISCARD_FIRST=1",
            "DSV4_TG_STDOUT_MAX_NON_JSON_LINES=4096", "DSV4_HASH_MODE=full", "DSV4_TG_PROFILE=none", "GGML_HIP_GRAPHS=1",
            "GGML_CUDA_DISABLE_GRAPHS=<unset>", "NCCL_ALGO=Tree", "NCCL_PROTO=LL", "NCCL_MIN_NCHANNELS=<unset>", "NCCL_MAX_NCHANNELS=<unset>",
            "NCCL_DEBUG=INFO", "NCCL_DEBUG_SUBSYS=ENV", "args=--dry-run",
        ]
        if any(item not in tree for item in required):
            raise RuntimeError(f"tree wrapper contract mismatch:\n{tree}")
        auto = run([str(WRAPPER), "auto"], env=clean_env).stdout
        if "NCCL_ALGO=<unset>" not in auto or "NCCL_PROTO=<unset>" not in auto:
            raise RuntimeError(f"auto did not clear forced tuning:\n{auto}")
        dirty_env = dict(clean_env, NCCL_SOCKET_IFNAME="bad")
        rejected = run([str(WRAPPER), "auto"], env=dirty_env, ok=False)
        if "inherited communication/graph environment is forbidden" not in rejected.stderr:
            raise RuntimeError("inherited NCCL variable did not fail closed")
        graph_dirty = dict(clean_env, GGML_CUDA_DISABLE_GRAPHS="1")
        rejected = run([str(WRAPPER), "tree-ll"], env=graph_dirty, ok=False)
        if "GGML_CUDA_DISABLE_GRAPHS" not in rejected.stderr:
            raise RuntimeError("inherited runtime graph disable did not fail closed")

        control = make_run(root, "control", "auto", [20.0, 18.0, 16.0])
        winner = make_run(root, "winner", "tree-ll", [20.1, 18.1, 16.6])
        output = root / "comparison.json"
        passed = run([str(COMPARE), str(control), str(winner), "--json", str(output)]).stdout
        value = json.loads(output.read_text())
        if "PASS TO FULL VALIDATION" not in passed or value["selected_for_full_validation"] is not True:
            raise RuntimeError("positive screen fixture failed")
        loser = make_run(root, "loser", "tree-ll", [20.0, 18.0, 16.4])
        failed_gate = run([str(COMPARE), str(control), str(loser)]).stdout
        if "NO-GO" not in failed_gate:
            raise RuntimeError("sub-3% candidate passed")
        mismatched = make_run(root, "mismatched", "tree-ll", [20.1, 18.1, 16.6], "b" * 64)
        run([str(COMPARE), str(control), str(mismatched)], ok=False)
        noisy = make_run(root, "noisy", "tree-ll", [20.1, 18.1, 16.6])
        write_capture(noisy, [
            "host:1:2 [0] NCCL INFO NCCL_ALGO set by environment to Tree",
            "host:1:2 [0] NCCL INFO NCCL_PROTO set by environment to LL",
            "host:1:1 [0] NCCL INFO AllReduce: 28672 Bytes -> Algo TREE proto LL",
        ])
        run([str(COMPARE), str(control), str(noisy)], ok=False)
        stderr_noisy = make_run(root, "stderr-noisy", "tree-ll", [20.1, 18.1, 16.6])
        (stderr_noisy / "bench.log").write_text(
            "host:1:1 [0] NCCL INFO pre-adjustment threadThreshold:8 nBytes:28672 nc:1\n"
        )
        run([str(COMPARE), str(control), str(stderr_noisy)], ok=False)
        missing_ack = make_run(root, "missing-ack", "tree-ll", [20.1, 18.1, 16.6])
        write_capture(missing_ack, [])
        run([str(COMPARE), str(control), str(missing_ack)], ok=False)
        old_capture = make_run(root, "old-capture", "tree-ll", [20.1, 18.1, 16.6])
        old_contract = json.loads((old_capture / "contract.json").read_text())
        del old_contract["stdout_capture"]
        (old_capture / "contract.json").write_text(json.dumps(old_contract))
        run([str(COMPARE), str(control), str(old_capture)], ok=False)
        old_debug = make_run(root, "old-debug", "tree-ll", [20.1, 18.1, 16.6])
        old_contract = json.loads((old_debug / "contract.json").read_text())
        old_contract["communication_candidate"]["debug_subsys"] = "ENV,TUNING"
        (old_debug / "contract.json").write_text(json.dumps(old_contract))
        run([str(COMPARE), str(control), str(old_debug)], ok=False)
        graphs_off = make_run(root, "graphs-off", "tree-ll", [20.1, 18.1, 16.6])
        manifest = (graphs_off / "manifest.txt").read_text().replace(
            "GGML_HIP_GRAPHS:BOOL=ON", "GGML_HIP_GRAPHS:BOOL=OFF"
        )
        (graphs_off / "manifest.txt").write_text(manifest)
        run([str(COMPARE), str(control), str(graphs_off)], ok=False)
        graphs_wrong_type = make_run(root, "graphs-wrong-type", "tree-ll", [20.1, 18.1, 16.6])
        manifest = (graphs_wrong_type / "manifest.txt").read_text().replace(
            "GGML_HIP_GRAPHS:BOOL=ON", "GGML_HIP_GRAPHS:STRING=ON"
        )
        (graphs_wrong_type / "manifest.txt").write_text(manifest)
        run([str(COMPARE), str(control), str(graphs_wrong_type)], ok=False)
        runtime_disabled = make_run(root, "runtime-disabled", "tree-ll", [20.1, 18.1, 16.6])
        disabled_contract = json.loads((runtime_disabled / "contract.json").read_text())
        disabled_contract["communication_candidate"]["runtime_graph_disable"] = "1"
        (runtime_disabled / "contract.json").write_text(json.dumps(disabled_contract))
        run([str(COMPARE), str(control), str(runtime_disabled)], ok=False)
    print("dsv4 RCCL raw-TG screen fixtures: PASS")


if __name__ == "__main__":
    main()