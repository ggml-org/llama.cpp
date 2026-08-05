#!/usr/bin/env python3
"""Non-GPU positive/negative fixtures for the short BF16 correctness/TG gates."""

from __future__ import annotations

import copy
import hashlib
import json
import pathlib
import struct
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
CORRECTNESS_COMPARATOR = ROOT / "scripts/dsv4-rocm/compare-bf16-allreduce-equivalence.py"
TG_COMPARATOR = ROOT / "scripts/dsv4-rocm/compare-bf16-tg.py"
COMMIT = "a" * 40


def sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fnv(data: bytes) -> str:
    value = 1469598103934665603
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return f"{value:016x}"


def write_manifest(directory: pathlib.Path, binary: pathlib.Path, common: pathlib.Path, *, graph: str = "ON", bad_dso: bool = False) -> None:
    identities = {
        "source.patch": b"",
        "source-status.txt": b"",
        "untracked-files.sha256": b"",
        "model-identity.txt": b"size=1 mtime=fixed inode=1 path=/model/shard.gguf\n",
        "hardware-identity.txt": b"four fixed V620 devices\n",
    }
    for name, data in identities.items():
        (directory / name).write_bytes(data)
    lines = [f"commit={COMMIT}", f"GGML_HIP_GRAPHS:BOOL={graph}"]
    for name in identities:
        path = directory / name
        lines.append(f"{sha(path)}  {path.resolve()}")
    lines.append(f"{sha(binary)}  {binary.resolve()}")
    for index in range(4):
        path = common / f"libfixture{index}.so"
        digest = ("f" * 64) if bad_dso and index == 0 else sha(path)
        lines.append(f"{digest}  {path.resolve()}")
    cache = common / "CMakeCache.txt"
    lines.append(f"{sha(cache)}  {cache.resolve()}")
    (directory / "manifest.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_correctness(root: pathlib.Path, binary: pathlib.Path, common: pathlib.Path, *, compare: bool = True) -> None:
    root.mkdir()
    write_manifest(root, binary, common)
    (root / "contract.txt").write_text(
        f"git_head={COMMIT}\nbinary={binary.resolve()}\nbinary_sha256={sha(binary)}\n", encoding="utf-8"
    )
    inputs = [11, 12, 13, 14]
    for arm, value in (("control", "0"), ("candidate", "1")):
        directory = root / arm
        directory.mkdir()
        chunks = []
        records = []
        offset = 0
        for step in range(4):
            raw = struct.pack("<3f", float(step), float(step) + 0.1, float(step) - 0.2)
            chunks.append(raw)
            records.append({
                "step": step, "input_token": inputs[step], "argmax_token": 1, "n_vocab": 3,
                "byte_offset": offset, "byte_length": len(raw), "logits_fnv1a64": fnv(raw),
            })
            offset += len(raw)
        (directory / "logits.f32").write_bytes(b"".join(chunks))
        result = {
            "schema_version": 1, "complete": True, "target_only": True, "state_restore_used": False,
            "sampling_used": False, "candidate_value": value, "depth": 2048, "n_gen": 4, "seed": 12345,
            "n_batch": 512, "n_ubatch": 256, "cache_type_k": "f16", "cache_type_v": "f16",
            "flash_attn": "enabled", "logits_file": "logits.f32",
            "float_format": "IEEE-754 binary32 native little-endian", "expected_logits_bytes": offset,
            "depths": [{"depth": 2048, "prefix_fnv1a64": "abcd", "generation_input_tokens": inputs, "records": records}],
        }
        (directory / "result.json").write_text(json.dumps(result), encoding="utf-8")
        audit = {
            "schema_version": 1, "context_id": 0, "candidate_enabled": value == "1", "candidate_topology": True,
            "backend_count": 4, "logical_devices": [0, 1, 2, 3], "allreduce_calls": 500,
            "zero_element_calls": 0, "candidate_eligible_calls": 344,
            "candidate_bf16_calls": 344 if value == "1" else 0,
            "candidate_disabled_fp32_calls": 344 if value == "0" else 0,
            "force_fp32_calls": 4, "force_candidate_conflict_calls": 0,
            "legacy_fp32_calls": 50 if value == "1" else 394, "legacy_bf16_calls": 102, "complete": True,
        }
        (directory / "audit.jsonl").write_text(json.dumps(audit) + "\n", encoding="utf-8")
    if compare:
        completed = subprocess.run([
            str(CORRECTNESS_COMPARATOR), str(root / "control"), str(root / "candidate"),
            "--json", str(root / "comparison.json"),
        ], check=False, stdout=subprocess.DEVNULL)
        assert completed.returncode == 0


def run_correctness_comparator(root: pathlib.Path) -> int:
    completed = subprocess.run([
        str(CORRECTNESS_COMPARATOR), str(root / "control"), str(root / "candidate"),
        "--json", str(root / "comparison.json"),
    ], check=False, stdout=subprocess.DEVNULL)
    return completed.returncode


def mutate_manifest_hash(directory: pathlib.Path, artifact_name: str, mode: str) -> None:
    path = directory / "manifest.txt"
    lines = path.read_text(encoding="utf-8").splitlines()
    matching = [index for index, line in enumerate(lines) if line.endswith(f"/{artifact_name}")]
    assert len(matching) == 1
    index = matching[0]
    if mode == "remove":
        del lines[index]
    elif mode == "corrupt":
        lines[index] = ("0" if lines[index][0] != "0" else "1") + lines[index][1:]
    else:
        raise AssertionError(mode)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def base_summary(multiplier: float) -> dict:
    depths = [0, 2048, 8192]
    records = [{
        "depth": depth, "n_gen": 8, "raw_repetitions": 6, "discarded_first": 1,
        "accepted_repetitions": 5, "stability_limit": 0.03, "contract_errors": [],
        "median_ts": 20.0 * multiplier, "mad_over_median": 0.01, "stable": True,
    } for depth in depths]
    return {
        "complete": True, "stable": True, "truncated": False, "dropped_trailing_partial": False,
        "expected_depths": depths, "seen_depths": depths, "missing_depths": [], "extra_depths": [],
        "duplicate_depths": [], "depth_order_matches": True, "expected_gen": 8,
        "expected_raw_repetitions": 6, "discard_first": 1, "accepted_repetitions": 5,
        "stability_limit": 0.03, "contract_errors": [], "records": records,
    }


def base_contract(value: str) -> dict:
    return {
        "target_only": True, "draft_model_loaded": False, "speculative_flags": [], "mode": "performance",
        "depths": [0, 2048, 8192], "n_gen": 8, "raw_repetitions": 6, "discard_first": 1,
        "accepted_repetitions": 5, "depth_state_api": "context", "profile": "none",
        "model_hash_mode": "metadata", "require_accepted_stack": 1, "allow_busy_gpus": 0,
        "batch": 512, "ubatch": 256, "tensor_split": "1/1/1/1", "cache_type_k": "f16",
        "cache_type_v": "f16", "threads": 12, "load_mode": "mmap",
        "communication_candidate": {
            "backend": "nccl", "hip_graphs": "1", "runtime_graph_disable": None,
            "algorithm": None, "protocol": None, "min_channels": None, "max_channels": None,
            "debug": None, "debug_subsys": None, "bf16_hidden_allreduce": value,
            "bf16_hidden_allreduce_audit": None,
        },
    }


def write_tg_arm(directory: pathlib.Path, value: str, multiplier: float, binary: pathlib.Path, common: pathlib.Path,
                 *, graph: str = "ON", bad_dso: bool = False) -> None:
    directory.mkdir(parents=True)
    write_manifest(directory, binary, common, graph=graph, bad_dso=bad_dso)
    (directory / "summary.json").write_text(json.dumps(base_summary(multiplier)), encoding="utf-8")
    (directory / "contract.json").write_text(json.dumps(base_contract(value)), encoding="utf-8")
    raw_stdout = b"{}\n{}\n{}\n"
    (directory / "bench.stdout.log").write_bytes(raw_stdout)
    (directory / "bench.stdout-nonjson.log").write_bytes(b"")
    classification = {
        "schema_version": 1, "json_lines": 3, "json_completion_timestamps": 3,
        "malformed_json_like_lines": 0, "consumer_success": True, "unterminated_final_data": False,
        "excessive_non_json_output": False, "max_non_json_lines": 4096,
        "raw_stream_preserved": True, "total_bytes": len(raw_stdout),
    }
    (directory / "stdout-classification.json").write_text(json.dumps(classification), encoding="utf-8")
    (directory / "status.txt").write_text(
        "process_exit_code=0\nstderr_consumer_exit_code=0\nstdout_consumer_exit_code=0\n", encoding="utf-8"
    )
    marker = "guarded RDNA2 BF16 hidden AllReduce armed across 4 devices\nusing guarded RDNA2 BF16 hidden AllReduce\n"
    (directory / "bench.log").write_text(marker * 3 if value == "1" else "", encoding="utf-8")
    (directory / "result.jsonl").write_text("{}\n{}\n{}\n", encoding="utf-8")
    (directory / "result-completed-at.ns").write_text("1\n2\n3\n", encoding="utf-8")


def run_tg_case(root: pathlib.Path, correctness: pathlib.Path, binary: pathlib.Path, common: pathlib.Path,
                *, gain: float = 1.05, mutate=None, candidate_graph: str = "ON", bad_dso: bool = False) -> int:
    control = root / "control"
    candidate = root / "candidate"
    write_tg_arm(control, "0", 1.0, binary, common)
    write_tg_arm(candidate, "1", gain, binary, common, graph=candidate_graph, bad_dso=bad_dso)
    if mutate:
        mutate(control, candidate)
    completed = subprocess.run([
        str(TG_COMPARATOR), str(control), str(candidate), "--correctness-dir", str(correctness),
        "--json", str(root / "comparison.json"),
    ], check=False, stdout=subprocess.DEVNULL)
    return completed.returncode


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="dsv4-bf16-fixture-") as temporary:
        root = pathlib.Path(temporary)
        common = root / "common"
        common.mkdir()
        for name in ["correctness-bin", "bench", "CMakeCache.txt"] + [f"libfixture{i}.so" for i in range(4)]:
            (common / name).write_text(name, encoding="utf-8")
        correctness = root / "correctness"
        write_correctness(correctness, common / "correctness-bin", common)

        required_artifacts = [
            "source.patch", "source-status.txt", "untracked-files.sha256",
            "model-identity.txt", "hardware-identity.txt",
        ]
        for artifact in required_artifacts:
            for mode in ("remove", "corrupt"):
                fixture = root / f"correctness-{artifact}-{mode}"
                write_correctness(fixture, common / "correctness-bin", common, compare=False)
                mutate_manifest_hash(fixture, artifact, mode)
                assert run_correctness_comparator(fixture) == 2

        assert run_tg_case(root / "pass", correctness, common / "bench", common) == 0
        assert run_tg_case(root / "no-gain", correctness, common / "bench", common, gain=1.01) == 1

        def bad_stable(_control, candidate):
            path = candidate / "summary.json"
            value = json.loads(path.read_text())
            value["records"][0]["stable"] = "false"
            path.write_text(json.dumps(value))
        assert run_tg_case(root / "bad-stable", correctness, common / "bench", common, mutate=bad_stable) == 2

        def nonfinite(_control, candidate):
            path = candidate / "summary.json"
            text = path.read_text().replace('"median_ts": 21.0', '"median_ts": 1e309', 1)
            path.write_text(text)
        assert run_tg_case(root / "nonfinite", correctness, common / "bench", common, mutate=nonfinite) == 2

        def unstable(_control, candidate):
            path = candidate / "summary.json"
            value = json.loads(path.read_text())
            value["records"][-1]["mad_over_median"] = 0.031
            value["records"][-1]["stable"] = False
            value["stable"] = False
            path.write_text(json.dumps(value))
        assert run_tg_case(root / "unstable", correctness, common / "bench", common, mutate=unstable) == 1

        def busy(_control, candidate):
            path = candidate / "contract.json"
            value = json.loads(path.read_text()); value["allow_busy_gpus"] = 1
            path.write_text(json.dumps(value))
        assert run_tg_case(root / "busy", correctness, common / "bench", common, mutate=busy) == 2
        assert run_tg_case(root / "graph-off", correctness, common / "bench", common, candidate_graph="OFF") == 2
        assert run_tg_case(root / "dso-mismatch", correctness, common / "bench", common, bad_dso=True) == 2

        for artifact in required_artifacts:
            for mode in ("remove", "corrupt"):
                def mutate_required(_control, candidate, artifact=artifact, mode=mode):
                    mutate_manifest_hash(candidate, artifact, mode)
                assert run_tg_case(
                    root / f"tg-{artifact}-{mode}", correctness, common / "bench", common, mutate=mutate_required
                ) == 2

        def missing_cache(control, candidate):
            mutate_manifest_hash(control, "CMakeCache.txt", "remove")
            mutate_manifest_hash(candidate, "CMakeCache.txt", "remove")
        assert run_tg_case(root / "cache-missing", correctness, common / "bench", common, mutate=missing_cache) == 2

        alternate_cache = common / "alternate" / "CMakeCache.txt"
        alternate_cache.parent.mkdir()
        alternate_cache.write_text("CMakeCache.txt", encoding="utf-8")
        def relocated_cache(control, candidate):
            original = str((common / "CMakeCache.txt").resolve())
            replacement = str(alternate_cache.resolve())
            for directory in (control, candidate):
                path = directory / "manifest.txt"
                path.write_text(path.read_text().replace(original, replacement))
        assert run_tg_case(root / "cache-relocated", correctness, common / "bench", common, mutate=relocated_cache) == 2

        def mismatched_cache(control, candidate):
            for directory in (control, candidate):
                path = directory / "manifest.txt"
                lines = path.read_text().splitlines()
                for index, line in enumerate(lines):
                    if line.endswith("/CMakeCache.txt"):
                        lines[index] = "f" * 64 + line[64:]
                path.write_text("\n".join(lines) + "\n")
        assert run_tg_case(root / "cache-mismatched", correctness, common / "bench", common, mutate=mismatched_cache) == 2

        misleading_correctness = root / "correctness-misleading-dso"
        write_correctness(misleading_correctness, common / "correctness-bin", common)
        def replace_dso_paths(directory):
            path = directory / "manifest.txt"
            text = path.read_text()
            for index in range(4):
                original = str((common / f"libfixture{index}.so").resolve())
                text = text.replace(original, original + "-not-a-library")
            path.write_text(text)
        replace_dso_paths(misleading_correctness)
        def misleading_dso(control, candidate):
            replace_dso_paths(control)
            replace_dso_paths(candidate)
        assert run_tg_case(
            root / "misleading-dso-paths", misleading_correctness, common / "bench", common, mutate=misleading_dso
        ) == 2

    print("dsv4 BF16 short-screen fixtures: PASS")


if __name__ == "__main__":
    main()