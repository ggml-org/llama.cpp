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


def make_run(root, name, label, medians, digest="a" * 64):
    path = root / name
    path.mkdir()
    algorithm, protocol = {"auto": (None, None), "tree-ll": ("Tree", "LL")}[label]
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
        "model_hash_mode": "full", "communication_candidate": {
            "label": label, "backend": "nccl", "hip_graphs": "1", "algorithm": algorithm, "protocol": protocol,
            "min_channels": None, "max_channels": None, "debug": "INFO", "debug_subsys": "ENV,TUNING",
        },
    }))
    (path / "status.txt").write_text("process_exit_code=0\ntruncated=0\n")
    (path / "manifest.txt").write_text(f"{digest}  /model/shard.gguf\n")
    return path


def main():
    with tempfile.TemporaryDirectory() as temp:
        root = pathlib.Path(temp)
        fake_root = root / "repo"
        fake_script = fake_root / "scripts/dsv4-rocm/run-tg.sh"
        fake_script.parent.mkdir(parents=True)
        fake_script.write_text("""#!/usr/bin/env bash
set -Eeuo pipefail
for name in DSV4_RCCL_CANDIDATE DSV4_TG_DEPTHS DSV4_TG_N_GEN DSV4_TG_REPS DSV4_TG_DISCARD_FIRST DSV4_HASH_MODE DSV4_TG_PROFILE GGML_HIP_GRAPHS NCCL_ALGO NCCL_PROTO NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS NCCL_DEBUG NCCL_DEBUG_SUBSYS; do
    if declare -p "$name" >/dev/null 2>&1; then printf '%s=%s\\n' "$name" "${!name}"; else printf '%s=<unset>\\n' "$name"; fi
done
printf 'args='; printf '%s ' "$@"; printf '\\n'
""")
        fake_script.chmod(0o755)
        clean_env = {k: v for k, v in os.environ.items() if not (k.startswith("NCCL_") or k.startswith("RCCL_"))}
        clean_env["DSV4_ROOT_DIR"] = str(fake_root)
        clean_env["DSV4_TG_DEPTHS"] = "1"  # Must be overridden, not inherited.
        tree = run([str(WRAPPER), "tree-ll", "--dry-run"], env=clean_env).stdout
        required = [
            "DSV4_RCCL_CANDIDATE=tree-ll", "DSV4_TG_DEPTHS=16384,32768,65536",
            "DSV4_TG_N_GEN=32", "DSV4_TG_REPS=6", "DSV4_TG_DISCARD_FIRST=1",
            "DSV4_HASH_MODE=full", "DSV4_TG_PROFILE=none", "GGML_HIP_GRAPHS=1", "NCCL_ALGO=Tree",
            "NCCL_PROTO=LL", "NCCL_MIN_NCHANNELS=<unset>", "NCCL_MAX_NCHANNELS=<unset>",
            "NCCL_DEBUG=INFO", "NCCL_DEBUG_SUBSYS=ENV,TUNING", "args=--dry-run",
        ]
        if any(item not in tree for item in required):
            raise RuntimeError(f"tree wrapper contract mismatch:\n{tree}")
        auto = run([str(WRAPPER), "auto"], env=clean_env).stdout
        if "NCCL_ALGO=<unset>" not in auto or "NCCL_PROTO=<unset>" not in auto:
            raise RuntimeError(f"auto did not clear forced tuning:\n{auto}")
        dirty_env = dict(clean_env, NCCL_SOCKET_IFNAME="bad")
        rejected = run([str(WRAPPER), "auto"], env=dirty_env, ok=False)
        if "inherited communication environment is forbidden" not in rejected.stderr:
            raise RuntimeError("inherited NCCL variable did not fail closed")

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
    print("dsv4 RCCL raw-TG screen fixtures: PASS")


if __name__ == "__main__":
    main()