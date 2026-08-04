#!/usr/bin/env python3
"""Non-GPU fixture tests for DSV4 raw-TG summary and scheduler parsers."""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import tempfile


ROOT = pathlib.Path(__file__).resolve().parent
SUMMARIZE = ROOT / "summarize-tg.py"
PARSE = ROOT / "parse-sched-debug.py"
CAPTURE = ROOT / "capture-tg-stdout.py"
RUN_TG = ROOT / "run-tg.sh"


def run(*args: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise AssertionError(f"command failed {result.returncode}: {' '.join(args)}\n{result.stdout}\n{result.stderr}")
    return result


def record(depth: int, accepted_ns: list[int] | None = None, **changes: object) -> dict[str, object]:
    if accepted_ns is None:
        accepted_ns = [3_200_000_000, 3_210_000_000, 3_190_000_000, 3_205_000_000, 3_195_000_000]
    samples_ns = [4_000_000_000, *accepted_ns]
    value: dict[str, object] = {
        "model_filename": "fixture.gguf",
        "n_prompt": 0,
        "n_gen": 32,
        "n_depth": depth,
        "n_batch": 512,
        "n_ubatch": 256,
        "split_mode": "tensor",
        "tensor_split": "1.00/1.00/1.00/1.00",
        "type_k": "f16",
        "type_v": "f16",
        "flash_attn": 1,
        "samples_ns": samples_ns,
        "samples_ts": [32e9 / ns for ns in samples_ns],
    }
    value.update(changes)
    return value


def write_jsonl(path: pathlib.Path, records: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(item) + "\n" for item in records))


def summarize(tmp: pathlib.Path, records: list[dict[str, object]], expected_depths: str = "0,2048") -> dict[str, object]:
    raw = tmp / "result.jsonl"
    out = tmp / "summary.json"
    tsv = tmp / "summary.tsv"
    write_jsonl(raw, records)
    run(
        sys.executable, str(SUMMARIZE), str(raw), "--json", str(out), "--tsv", str(tsv),
        "--expected-depths", expected_depths, "--expected-gen", "32", "--expected-reps", "6",
        "--discard-first", "1", "--stability-limit", "0.03", "--expected-batch", "512",
        "--expected-ubatch", "256", "--expected-tensor-split", "1/1/1/1",
    )
    assert tsv.read_text().startswith("run_complete\trun_stable\tdepth")
    return json.loads(out.read_text())


def capture_stdout(tmp: pathlib.Path, name: str, payload: str, maximum: int = 4096):
    root = tmp / name
    root.mkdir()
    paths = {
        "result": root / "result.jsonl",
        "completed": root / "result-completed-at.ns",
        "raw": root / "bench.stdout.log",
        "non_json": root / "bench.stdout-nonjson.log",
        "classification": root / "stdout-classification.json",
    }
    process = subprocess.run(
        [
            sys.executable, str(CAPTURE), "--result", str(paths["result"]),
            "--completed-at", str(paths["completed"]), "--raw", str(paths["raw"]),
            "--non-json", str(paths["non_json"]), "--classification", str(paths["classification"]),
            "--max-non-json-lines", str(maximum),
        ],
        input=payload, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return process, paths


def test_stdout_capture(tmp: pathlib.Path) -> None:
    payload = (
        "host:1:2 [0] NCCL INFO NCCL_ALGO set by environment to Tree\n"
        "[2026-08-04 21:00:00] host [0] NCCL WARN fixture warning\n"
        '{"n_depth": 16384}\n'
        "\n"
        '{"n_depth": 32768}\n'
    )
    process, paths = capture_stdout(tmp, "mixed", payload)
    if process.returncode:
        raise AssertionError(f"stdout capture failed:\n{process.stdout}\n{process.stderr}")
    assert paths["raw"].read_text() == payload
    assert paths["result"].read_text() == '{"n_depth": 16384}\n{"n_depth": 32768}\n'
    assert paths["non_json"].read_text() == (
        "host:1:2 [0] NCCL INFO NCCL_ALGO set by environment to Tree\n"
        "[2026-08-04 21:00:00] host [0] NCCL WARN fixture warning\n\n"
    )
    completed = paths["completed"].read_text().splitlines()
    assert len(completed) == 2 and all(item.isdigit() for item in completed)
    classification = json.loads(paths["classification"].read_text())
    assert classification["consumer_success"] is True
    assert classification["json_lines"] == 2 and classification["non_json_lines"] == 2
    assert classification["blank_lines"] == 1 and classification["total_bytes"] == len(payload.encode())

    malformed, malformed_paths = capture_stdout(tmp, "malformed", "noise\n{broken\n")
    assert malformed.returncode != 0
    malformed_value = json.loads(malformed_paths["classification"].read_text())
    assert malformed_value["malformed_json_like_lines"] == 1
    assert malformed_value["consumer_success"] is False

    unterminated, unterminated_paths = capture_stdout(tmp, "unterminated", '{"ok": true}')
    assert unterminated.returncode != 0
    assert json.loads(unterminated_paths["classification"].read_text())["unterminated_final_data"] is True

    noisy_payload = "".join(f"diagnostic {index}\n" for index in range(50))
    noisy, noisy_paths = capture_stdout(tmp, "noisy", noisy_payload, maximum=10)
    assert noisy.returncode != 0
    noisy_value = json.loads(noisy_paths["classification"].read_text())
    assert noisy_value["non_json_lines"] == 50 and noisy_value["excessive_non_json_output"] is True

    failure_root = tmp / "write-failure"
    failure_root.mkdir()
    raw_directory = failure_root / "raw-directory"
    raw_directory.mkdir()
    failed = subprocess.run(
        [
            sys.executable, str(CAPTURE), "--result", str(failure_root / "result"),
            "--completed-at", str(failure_root / "completed"), "--raw", str(raw_directory),
            "--non-json", str(failure_root / "non-json"),
            "--classification", str(failure_root / "classification"),
            "--max-non-json-lines", "10",
        ],
        input="{}\n", text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    assert failed.returncode != 0 and "stdout capture failed" in failed.stderr


def test_run_tg_output_capture(tmp: pathlib.Path) -> None:
    fake_bin = tmp / "fake-bin"
    fake_bin.mkdir()
    model = tmp / "fixture.gguf"
    model.write_bytes(b"fixture model")
    bench = fake_bin / "llama-bench-fixture"
    bench.write_text("""#!/usr/bin/env python3
import json, os, sys
mode = os.environ.get("FAKE_OUTPUT_MODE", "mixed")
print("host:1:2 [0] NCCL INFO NCCL_ALGO set by environment to Tree")
print("host:1:2 [0] NCCL INFO NCCL_PROTO set by environment to LL")
if mode == "noisy":
    for index in range(11):
        print(f"diagnostic {index}")
for benchmark, depth in enumerate((16384, 32768, 65536), 1):
    print(f"llama-bench: benchmark {benchmark}/3: starting", file=sys.stderr, flush=True)
    for repetition in range(1, 7):
        print(f"llama-bench: benchmark {benchmark}/3: depth run {repetition}/6", file=sys.stderr, flush=True)
        print(f"llama-bench: benchmark {benchmark}/3: generation run {repetition}/6", file=sys.stderr, flush=True)
    samples_ns = [4_000_000_000, 3_200_000_000, 3_210_000_000, 3_190_000_000, 3_205_000_000, 3_195_000_000]
    value = {
        "model_filename": "fixture.gguf", "n_prompt": 0, "n_gen": 32, "n_depth": depth,
        "n_batch": 512, "n_ubatch": 256, "split_mode": "tensor",
        "tensor_split": "1.00/1.00/1.00/1.00", "type_k": "f16", "type_v": "f16",
        "flash_attn": 1, "samples_ns": samples_ns,
        "samples_ts": [32e9 / sample for sample in samples_ns],
    }
    print(json.dumps(value))
if mode == "malformed":
    print("{broken")
""")
    bench.chmod(0o755)
    rocm_smi = fake_bin / "rocm-smi"
    rocm_smi.write_text("#!/usr/bin/env bash\necho 'No GPU processes'\n")
    rocm_smi.chmod(0o755)

    base_env = dict(os.environ)
    base_env.update({
        "PATH": f"{fake_bin}:{base_env['PATH']}",
        "HOME": str(tmp / "home"),
        "LLAMA_JOB_DIR": "non-gpu-fixture",
        "DSV4_MODEL": str(model),
        "DSV4_BENCH": str(bench),
        "DSV4_LIBRARY_PATH": str(fake_bin),
        "DSV4_TG_DEPTHS": "16384,32768,65536",
        "DSV4_TG_REPS": "6",
        "DSV4_TG_DISCARD_FIRST": "1",
        "DSV4_TG_SAMPLE_TIMEOUT": "10",
        "DSV4_TG_SETUP_TIMEOUT": "10",
        "DSV4_TERM_GRACE": "1",
        "DSV4_TG_STDOUT_MAX_NON_JSON_LINES": "10",
        "DSV4_HASH_MODE": "full",
        "DSV4_LABEL": "stdout-capture-fixture",
        "DSV4_RCCL_CANDIDATE": "tree-ll",
        "NCCL_ALGO": "Tree",
        "NCCL_PROTO": "LL",
        "NCCL_DEBUG": "INFO",
        "NCCL_DEBUG_SUBSYS": "ENV",
    })

    def invoke(mode: str):
        output_root = tmp / f"runs-{mode}"
        env = dict(base_env, DSV4_TG_OUTPUT_ROOT=str(output_root), FAKE_OUTPUT_MODE=mode)
        process = subprocess.run(
            [str(RUN_TG)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
        )
        runs = list(output_root.iterdir()) if output_root.is_dir() else []
        assert len(runs) == 1, (
            f"expected one fixture artifact for {mode}: {runs}\nstdout:\n{process.stdout}\nstderr:\n{process.stderr}"
        )
        return process, runs[0]

    good, good_run = invoke("mixed")
    if good.returncode:
        raise AssertionError(f"run-tg mixed-output fixture failed:\n{good.stdout}\n{good.stderr}")
    classification = json.loads((good_run / "stdout-classification.json").read_text())
    assert classification["consumer_success"] is True
    assert classification["json_lines"] == 3 and classification["non_json_lines"] == 2
    assert len((good_run / "result.jsonl").read_text().splitlines()) == 3
    assert len((good_run / "result-completed-at.ns").read_text().splitlines()) == 3
    assert "NCCL_ALGO set by environment" in (good_run / "bench.stdout.log").read_text()
    assert "NCCL_ALGO set by environment" in (good_run / "bench.stdout-nonjson.log").read_text()
    assert json.loads((good_run / "summary.json").read_text())["stable"] is True

    malformed, malformed_run = invoke("malformed")
    assert malformed.returncode == 2
    malformed_status = (malformed_run / "status.txt").read_text()
    assert "stdout_consumer_exit_code=2" in malformed_status
    assert json.loads((malformed_run / "stdout-classification.json").read_text())["malformed_json_like_lines"] == 1
    assert not (malformed_run / "summary.json").exists()

    noisy, noisy_run = invoke("noisy")
    assert noisy.returncode == 2
    noisy_value = json.loads((noisy_run / "stdout-classification.json").read_text())
    assert noisy_value["excessive_non_json_output"] is True
    assert "stdout_consumer_exit_code=2" in (noisy_run / "status.txt").read_text()
    assert not (noisy_run / "summary.json").exists()


def test_summary(tmp: pathlib.Path) -> None:
    good = summarize(tmp, [record(0), record(2048)])
    assert good["complete"] is True and good["stable"] is True
    assert good["observational_baseline_accepted"] is True
    assert all(row["accepted_repetitions"] == 5 for row in good["records"])
    assert all(row["mad_over_median"] <= 0.03 for row in good["records"])

    missing = summarize(tmp, [record(0)])
    assert missing["complete"] is False and missing["missing_depths"] == [2048]

    unstable_ns = [1_000_000_000, 2_000_000_000, 3_000_000_000, 4_000_000_000, 5_000_000_000]
    unstable = summarize(tmp, [record(0, unstable_ns), record(2048, unstable_ns)])
    assert unstable["complete"] is True and unstable["stable"] is False

    bad_contract = summarize(tmp, [record(0), record(2048, flash_attn=0)])
    assert bad_contract["complete"] is False
    assert any("flash_attn" in error for error in bad_contract["contract_errors"])


def scheduler_log(cpu_top_k: bool = False) -> str:
    top_backend = "CPU" if cpu_top_k else "Meta("
    return f"""llama-bench: benchmark 1/2: starting
llama-bench: benchmark 1/2: generation run 1/1
## SPLIT #0: CPU # 0 inputs
## SPLIT #1: Meta(ROCm0,ROCm1,ROCm2,ROCm3) # 22 inputs
node #  1 (   MUL_MAT): decode_out           ( 1KiB) [Meta(    2.sup] use=1,c=1:
llama-bench: benchmark 2/2: starting
llama-bench: benchmark 2/2: depth run 1/1
llama-bench: benchmark 2/2: generation run 1/1
## SPLIT #0: CPU # 0 inputs
## SPLIT #1: Meta(ROCm0,ROCm1,ROCm2,ROCm3) # 25 inputs
node # 10 (     TOP_K): lid_top_k-1          ( 2KiB) [{top_backend:<5}    2.sup] use=1,c=1:
node # 11 (      CONT): lid_top_k-1          ( 2KiB) [Meta(    2.sup] use=1,c=1:
node # 12 (  SET_ROWS): selected             ( 2KiB) [Meta(    2.sup] use=1,c=1: lid_top_k-1
node # 13 ( LIGHTNING): lid_score_masked-1   ( 4KiB) [Meta(    2.sup] use=1,c=1:
node # 14 (      CONT): lid_score_copy        ( 4KiB) [Meta(    2.sup] use=1,c=1: lid_score_masked-1
"""


def parse_scheduler(tmp: pathlib.Path, text: str) -> dict[str, object]:
    log = tmp / "bench.log"
    out = tmp / "scheduler.json"
    tsv = tmp / "scheduler.tsv"
    log.write_text(text)
    run(
        sys.executable, str(PARSE), str(log), "--depths", "0,2048",
        "--json", str(out), "--tsv", str(tsv), "--expected-nodes", "1",
    )
    assert tsv.read_text().startswith("run_complete\tresidency_ok\tdepth")
    return json.loads(out.read_text())


def test_scheduler(tmp: pathlib.Path) -> None:
    text = scheduler_log()
    good = parse_scheduler(tmp, text)
    assert good["complete"] is True and good["rocm_residency_ok"] is True
    assert good["require_top_k_from"] == 2048 and good["parse_warnings"] == []
    depth_2048 = next(item for item in good["records"] if item["depth"] == 2048)
    assert depth_2048["top_k_total"] == 1 and depth_2048["lid_total"] == 1
    assert depth_2048["cpu_splits"] == 1 and depth_2048["cpu_split_input_copies"] == 0
    assert depth_2048["total_split_input_copies"] == 25
    assert depth_2048["gpu_meta_split_input_copies"] == 25
    assert depth_2048["marker_structure_ok"] and depth_2048["split_structure_ok"]
    assert depth_2048["op_backend_correlated_to_meta_split"]

    cpu = parse_scheduler(tmp, scheduler_log(cpu_top_k=True))
    assert cpu["complete"] is True and cpu["rocm_residency_ok"] is False
    cpu_2048 = next(item for item in cpu["records"] if item["depth"] == 2048)
    assert cpu_2048["top_k_cpu"] == 1
    assert cpu_2048["op_backend_correlated_to_meta_split"] is False

    missing_nodes = parse_scheduler(tmp, text.replace(
        "node # 10 (     TOP_K): lid_top_k-1          ( 2KiB) [Meta(    2.sup] use=1,c=1:\n", ""))
    assert missing_nodes["complete"] is False and missing_nodes["parse_warnings"]

    bad_split = parse_scheduler(tmp, text.replace(
        "Meta(ROCm0,ROCm1,ROCm2,ROCm3) # 25 inputs",
        "Meta(ROCm0,ROCm1,ROCm2,ROCm3) # 24 inputs"))
    assert bad_split["complete"] is False and bad_split["rocm_residency_ok"] is False

    extra_split = parse_scheduler(tmp, text.replace(
        "## SPLIT #1: Meta(ROCm0,ROCm1,ROCm2,ROCm3) # 25 inputs\n",
        "## SPLIT #1: Meta(ROCm0,ROCm1,ROCm2,ROCm3) # 25 inputs\n## SPLIT #2: Other # 1 inputs\n"))
    assert extra_split["complete"] is False

    bad_marker = parse_scheduler(tmp, text.replace(
        "llama-bench: benchmark 2/2: generation run 1/1",
        "llama-bench: benchmark 2/2: generation run 1/2"))
    assert bad_marker["complete"] is False


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="dsv4-tg-tools-") as name:
        tmp = pathlib.Path(name)
        test_stdout_capture(tmp)
        test_run_tg_output_capture(tmp)
        test_summary(tmp)
        test_scheduler(tmp)
    print("dsv4 raw-TG tool fixtures: PASS")


if __name__ == "__main__":
    main()