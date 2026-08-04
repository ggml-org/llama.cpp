#!/usr/bin/env python3
"""Fail-closed matched raw-TG screen for predeclared DSV4 RCCL candidates."""
import argparse
import json
import os
import re
import tempfile
from pathlib import Path

DEPTHS = [16384, 32768, 65536]
EXPECTED = {
    "auto": (None, None),
    "tree-ll": ("Tree", "LL"),
    "ring-ll": ("Ring", "LL"),
}
SHA_LINE = re.compile(r"^([0-9a-f]{64})  (/.*)$")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_run(path: Path) -> dict:
    require(path.is_dir(), f"missing run directory: {path}")
    summary = json.loads((path / "summary.json").read_text())
    contract = json.loads((path / "contract.json").read_text())
    status = dict(
        line.split("=", 1) for line in (path / "status.txt").read_text().splitlines() if "=" in line
    )
    hashes = {}
    manifest_text = (path / "manifest.txt").read_text()
    for line in manifest_text.splitlines():
        match = SHA_LINE.fullmatch(line)
        if match:
            hashed_path = Path(match.group(2)).resolve(strict=False)
            try:
                relative = hashed_path.relative_to(path)
            except ValueError:
                key = str(hashed_path)
            else:
                key = f"<RUN>/{relative.as_posix()}"
            require(key not in hashes, f"{path}: duplicate normalized manifest hash path {key}")
            hashes[key] = match.group(1)
    require(hashes, f"{path}: no full manifest hashes")
    require(re.search(r"^GGML_HIP_GRAPHS:BOOL=ON$", manifest_text, re.MULTILINE), f"{path}: HIP graphs build option is not BOOL=ON")
    classification = json.loads((path / "stdout-classification.json").read_text())
    require(classification.get("schema_version") == 1, f"{path}: stdout classification schema")
    require(classification.get("consumer_success") is True, f"{path}: stdout classification failed")
    require(classification.get("json_lines") == len(DEPTHS), f"{path}: stdout JSON record count")
    require(classification.get("json_completion_timestamps") == len(DEPTHS), f"{path}: result timestamp count")
    require(classification.get("malformed_json_like_lines") == 0, f"{path}: malformed JSON-like stdout")
    require(classification.get("unterminated_final_data") is False, f"{path}: unterminated stdout")
    require(classification.get("excessive_non_json_output") is False, f"{path}: excessive non-JSON stdout")
    require(classification.get("max_non_json_lines") == 4096, f"{path}: stdout diagnostic limit")
    require(classification.get("raw_stream_preserved") is True, f"{path}: raw stdout not preserved")
    require((path / "bench.stdout.log").stat().st_size == classification.get("total_bytes"), f"{path}: raw stdout size")
    require((path / "bench.stdout-nonjson.log").is_file(), f"{path}: non-JSON stdout log missing")
    require(len((path / "result-completed-at.ns").read_text().splitlines()) == len(DEPTHS), f"{path}: timestamp file count")
    stdout_lines = [line for line in (path / "bench.stdout.log").read_text().splitlines() if line.strip()]
    stdout_lower = "\n".join(stdout_lines).lower()
    diagnostics_lower = stdout_lower + "\n" + (path / "bench.log").read_text().lower()
    tuning_markers = ("allreduce:", "threadthreshold:", "rccl channel tuning")
    require(not any(marker in diagnostics_lower for marker in tuning_markers), f"{path}: in-band RCCL tuning diagnostics")
    require(status.get("process_exit_code") == "0", f"{path}: process did not exit zero")
    require(status.get("stderr_consumer_exit_code") == "0", f"{path}: stderr capture failed")
    require(status.get("stdout_consumer_exit_code") == "0", f"{path}: stdout capture failed")
    require(status.get("truncated") == "0", f"{path}: truncated")
    require(summary["complete"] is True and summary["stable"] is True, f"{path}: incomplete or unstable")
    require(summary["expected_depths"] == DEPTHS and summary["seen_depths"] == DEPTHS, f"{path}: depth contract")
    require(summary["expected_gen"] == 32, f"{path}: not tg32")
    require(summary["expected_raw_repetitions"] == 6, f"{path}: raw repetition count")
    require(summary["discard_first"] == 1 and summary["accepted_repetitions"] == 5, f"{path}: discard/count")
    require(contract["target_only"] is True and contract["speculative_flags"] == [], f"{path}: target-only contract")
    require(contract["depths"] == DEPTHS and contract["n_gen"] == 32, f"{path}: contract depth/tg")
    require(contract["raw_repetitions"] == 6 and contract["discard_first"] == 1, f"{path}: contract repetitions")
    require(contract["depth_state_api"] == "context", f"{path}: wrong depth state API")
    require(contract["model_hash_mode"] == "full", f"{path}: model hashes not full")
    require(contract.get("stdout_capture") == {
        "schema_version": 1,
        "raw_stream": "bench.stdout.log",
        "non_json_stream": "bench.stdout-nonjson.log",
        "classification": "stdout-classification.json",
        "max_non_json_lines": 4096,
    }, f"{path}: stdout capture contract")
    comm = contract["communication_candidate"]
    label = comm["label"]
    require(label in EXPECTED, f"{path}: unknown candidate {label}")
    require(comm["backend"] == "nccl", f"{path}: backend is not nccl")
    require(comm["hip_graphs"] == "1", f"{path}: HIP graph build expectation mismatch")
    require(comm.get("runtime_graph_disable") is None, f"{path}: runtime graphs disabled")
    require((comm["algorithm"], comm["protocol"]) == EXPECTED[label], f"{path}: candidate environment mismatch")
    require(comm["min_channels"] is None and comm["max_channels"] is None, f"{path}: channel forcing is forbidden")
    require(comm["debug"] == "INFO" and comm["debug_subsys"] == "ENV", f"{path}: debug contract mismatch")
    algorithm, protocol = EXPECTED[label]
    control_lines = [line.lower() for line in stdout_lines if "nccl_algo" in line.lower() or "nccl_proto" in line.lower()]
    algorithm_acks = [line for line in control_lines if "nccl_algo set by environment" in line]
    protocol_acks = [line for line in control_lines if "nccl_proto set by environment" in line]
    require(
        not any(token in line for line in control_lines for token in ("invalid", "ignored", "warning", "warn")),
        f"{path}: RCCL rejected or warned on forced controls",
    )
    if algorithm is None:
        require(not algorithm_acks, f"{path}: auto acknowledged forced algorithm")
        require(not protocol_acks, f"{path}: auto acknowledged forced protocol")
    else:
        require(
            any(f"nccl_algo set by environment to {algorithm.lower()}" in line for line in algorithm_acks),
            f"{path}: missing runtime NCCL_ALGO acknowledgement",
        )
        require(
            any(f"nccl_proto set by environment to {protocol.lower()}" in line for line in protocol_acks),
            f"{path}: missing runtime NCCL_PROTO acknowledgement",
        )
        require(len(algorithm_acks) <= 64 and len(protocol_acks) <= 64, f"{path}: excessive RCCL control acknowledgements")
    records = {row["depth"]: row for row in summary["records"]}
    require(sorted(records) == DEPTHS, f"{path}: record depths")
    require(all(row["stable"] and row["accepted_repetitions"] == 5 for row in records.values()), f"{path}: unstable record")
    return {"path": path, "summary": summary, "contract": contract, "hashes": hashes, "records": records, "label": label}


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as out:
            json.dump(value, out, indent=2, sort_keys=True)
            out.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("control", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    control = load_run(args.control.resolve())
    candidate = load_run(args.candidate.resolve())
    require(control["label"] == "auto", "first run must be auto control")
    require(candidate["label"] != "auto", "second run must be a forced candidate")
    require(control["path"] != candidate["path"], "duplicate run directory")
    require(control["hashes"] == candidate["hashes"], "binary/DSO/model hash identity mismatch")

    rows = []
    for depth in DEPTHS:
        base = control["records"][depth]["median_ts"]
        test = candidate["records"][depth]["median_ts"]
        rows.append({
            "depth": depth,
            "control_median_ts": base,
            "candidate_median_ts": test,
            "gain": test / base - 1.0,
            "control_mad_over_median": control["records"][depth]["mad_over_median"],
            "candidate_mad_over_median": candidate["records"][depth]["mad_over_median"],
        })
    by_depth = {row["depth"]: row for row in rows}
    selected_for_full_validation = (
        by_depth[65536]["gain"] >= 0.03
        and by_depth[16384]["gain"] >= -0.02
        and by_depth[32768]["gain"] >= -0.02
    )
    value = {
        "complete": True,
        "screen_only": True,
        "control": str(control["path"]),
        "candidate": str(candidate["path"]),
        "candidate_label": candidate["label"],
        "rows": rows,
        "gate": {"target_depth": 65536, "minimum_target_gain": 0.03, "maximum_shorter_regression": 0.02},
        "selected_for_full_validation": selected_for_full_validation,
        "optimization_accepted": False,
        "reason": "screen pass requires >=3% median TG gain at 64K and no >2% median regression at 16K/32K; five accepted reps select only full validation, never optimization acceptance",
    }
    if args.json:
        atomic_json(args.json, value)
    print("M5.4 DSV4 RCCL RAW-TG SCREEN: " + ("PASS TO FULL VALIDATION" if selected_for_full_validation else "NO-GO"))
    for row in rows:
        print(
            f"depth={row['depth']} control={row['control_median_ts']:.3f} candidate={row['candidate_median_ts']:.3f} "
            f"gain={row['gain']*100:+.3f}% control_mad={row['control_mad_over_median']*100:.3f}% "
            f"candidate_mad={row['candidate_mad_over_median']*100:.3f}%"
        )
    print(f"candidate={candidate['label']} selected_for_full_validation={int(selected_for_full_validation)} optimization_accepted=0")


if __name__ == "__main__":
    main()