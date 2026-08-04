#!/usr/bin/env python3
"""Non-GPU fixtures for selected-region raw-TG profile summarization."""

from __future__ import annotations

import csv
import json
import pathlib
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent
TOOL = ROOT / "summarize-tg-profile.py"


def write_csv(path: pathlib.Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_fixture(root: pathlib.Path, outside: bool = False) -> pathlib.Path:
    run = root / ("outside" if outside else "good")
    rocprof = run / "rocprof"
    rocprof.mkdir(parents=True)
    (run / "contract.json").write_text(json.dumps({
        "profile": "kernel",
        "profile_scope": "accepted-target-generation-selected-regions",
        "profile_skip_repetitions": 1,
        "mode": "performance",
        "depth_state_api": "context",
        "model_hash_mode": "full",
        "require_accepted_stack": 1,
        "accepted_stack": {"mmq_j": 16, "hc_mixes": 1, "lid_subwave": 4},
        "batch": 512,
        "ubatch": 256,
        "tensor_split": "1/1/1/1",
        "cache_type_k": "f16",
        "cache_type_v": "f16",
        "threads": 12,
        "load_mode": "mmap",
        "target_only": True,
        "draft_model_loaded": False,
        "speculative_flags": [],
        "depths": [16384],
        "n_gen": 32,
        "raw_repetitions": 6,
        "discard_first": 1,
        "accepted_repetitions": 5,
    }))
    (run / "summary.json").write_text(json.dumps({
        "complete": True, "stable": True,
        "records": [{"depth": 16384, "stable": True, "mad_over_median": 0.01, "stability_limit": 0.03}],
    }))
    (run / "status.txt").write_text("process_exit_code=0\ntruncated=0\ntimeout_phase=none\nfinished_at_ns=1000000900\n")
    (run / "source-status.txt").write_text("")
    (run / "untracked-files.sha256").write_text("")
    hashes = ["0" * 64, "1" * 64, "2" * 64]
    (run / "manifest.txt").write_text(
        "-- all resolved dependency hashes (local + ROCm/system DSOs) --\n" + "a" * 64 + "  /lib/example.so\n"
        "=== model_files ===\nhash_mode=full\n" +
        "\n".join(f"{digest}  /model/part-{index}.gguf" for index, digest in enumerate(hashes, 1)) +
        "\n=== model_metadata ===\n"
    )
    (run / "result.jsonl").write_text(json.dumps({"samples_ns": [4000, 3200, 3210, 3190, 3205, 3195]}) + "\n")
    (run / "result-completed-at.ns").write_text("1000000850\n")
    (run / "clock-domain.txt").write_text(
        "start_realtime_minus_monotonic_ns=1000000000\n"
        "end_realtime_minus_monotonic_ns=1000000000\n"
        "start_calibration_span_ns=100\nend_calibration_span_ns=100\n"
    )
    phase = ["timestamp_ns\tphase\tbenchmark\trepetition\ttotal_repetitions"]
    stamp = 1000000100
    for rep in range(1, 7):
        phase.append(f"{stamp}\tsetup\t1\t{rep}\t6")
        phase.append(f"{stamp + 10}\tmeasurement\t1\t{rep}\t6")
        stamp += 120
    (run / "phase-events.tsv").write_text("\n".join(phase) + "\n")
    boundaries = ["event\tbenchmark\tdepth\trepetition\ttimestamp_monotonic_ns"]
    for rep in range(2, 7):
        generation_start = 110 + (rep - 1) * 120
        boundaries.append(f"resume_return\t1\t16384\t{rep}\t{generation_start + 2}")
        boundaries.append(f"pause_call\t1\t16384\t{rep}\t{generation_start + 108}")
    (run / "rocprof-selected-regions.tsv").write_text("\n".join(boundaries) + "\n")
    agent_fields = ["Logical_Node_Id", "Agent_Type", "Domain", "Location_Id", "Gpu_Id", "Product_Name"]
    write_csv(rocprof / "dsv4-tg_agent_info.csv", agent_fields, [
        {"Logical_Node_Id": i, "Agent_Type": "GPU", "Domain": 0, "Location_Id": i * 8,
         "Gpu_Id": 100 + i, "Product_Name": "V620"}
        for i in range(1, 5)
    ])
    kernel_fields = [
        "Agent_Id", "Kernel_Name", "Start_Timestamp", "End_Timestamp",
        "Workgroup_Size_X", "Workgroup_Size_Y", "Workgroup_Size_Z",
        "Grid_Size_X", "Grid_Size_Y", "Grid_Size_Z",
    ]
    rows: list[dict[str, object]] = []
    names = [
        "void mul_mat_q<(ggml_type)16, 16, false>(...)",
        "lightning_indexer_kernel_vec",
        "flash_attn_tile_f32",
        "ncclDevKernel_AllReduce",
    ]
    durations = [100, 20, 30, 10]
    for rep in range(2, 7):
        start = (1000000100 + (rep - 1) * 120 + 10) - 1000000000 + 5
        for index, (name, duration) in enumerate(zip(names, durations)):
            event_start = start + index * 20
            rows.append({
                "Agent_Id": f"Agent {index + 1}", "Kernel_Name": name,
                "Start_Timestamp": event_start, "End_Timestamp": event_start + duration,
                "Workgroup_Size_X": 32, "Workgroup_Size_Y": 1, "Workgroup_Size_Z": 1,
                "Grid_Size_X": 128, "Grid_Size_Y": 11, "Grid_Size_Z": 1,
            })
    if outside:
        rows[0]["Start_Timestamp"] = 1
        rows[0]["End_Timestamp"] = 2
    write_csv(rocprof / "dsv4-tg_kernel_trace.csv", kernel_fields, rows)
    api_fields = ["Function", "Start_Timestamp", "End_Timestamp"]
    write_csv(rocprof / "dsv4-tg_hip_api_trace.csv", api_fields, [{
        "Function": "hipGraphLaunch", "Start_Timestamp": 240, "End_Timestamp": 250,
    }])
    write_csv(rocprof / "dsv4-tg_rccl_api_trace.csv", api_fields, [{
        "Function": "ncclAllReduce", "Start_Timestamp": 250, "End_Timestamp": 260,
    }])
    copy_fields = ["Direction", "Start_Timestamp", "End_Timestamp"]
    write_csv(rocprof / "dsv4-tg_memory_copy_trace.csv", copy_fields, [{
        "Direction": "DEVICE_TO_DEVICE", "Start_Timestamp": 260, "End_Timestamp": 270,
    }])
    return run


def expect_bad(run: pathlib.Path, needle: str) -> None:
    result = subprocess.run(
        [sys.executable, str(TOOL), str(run)],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    assert result.returncode == 2, (result.stdout, result.stderr)
    assert needle in result.stderr, result.stderr


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="dsv4-tg-profile-") as name:
        root = pathlib.Path(name)
        good = make_fixture(root / "accepted")
        out = good / "profile.json"
        tsv = good / "profile.tsv"
        result = subprocess.run(
            [sys.executable, str(TOOL), str(good), "--json", str(out), "--tsv", str(tsv)],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        if result.returncode:
            raise AssertionError(f"good fixture failed:\n{result.stdout}\n{result.stderr}")
        value = json.loads(out.read_text())
        assert value["complete"] and value["profiled_repetitions"] == 5
        assert value["evaluated_target_tokens"] == 160
        assert value["kernel_dispatches"] == 20
        assert value["outside_selected_interval_events"]["kernel"] == 0
        assert value["families"][0]["family"] == "routed_expert_iq2_iq3_quant_matmul"
        assert value["profiled_wall_stable"] is True and value["profiled_throughput_eligible"] is False
        assert len(value["per_repetition"]) == 5
        assert value["trace_domain_status"] == {
            "kernel": "present_with_events", "memory_copy": "present_with_events",
            "rccl": "present_with_events", "hip": "present_with_events",
        }
        assert tsv.read_text().startswith("depth\tprofiled_repetitions\ttarget_tokens\tfamily")

        unstable = make_fixture(root / "unstable")
        unstable_summary = json.loads((unstable / "summary.json").read_text())
        unstable_summary["stable"] = False
        unstable_summary["records"][0].update({"stable": False, "mad_over_median": 0.06})
        (unstable / "summary.json").write_text(json.dumps(unstable_summary))
        unstable_out = unstable / "profile.json"
        result = subprocess.run(
            [sys.executable, str(TOOL), str(unstable), "--json", str(unstable_out)],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        assert result.returncode == 0, result.stderr
        unstable_value = json.loads(unstable_out.read_text())
        assert unstable_value["profiled_wall_stable"] is False
        assert unstable_value["csa_decision_eligible"] is False
        assert unstable_value["family_attribution_complete"] is True

        expect_bad(make_fixture(root / "outside", outside=True), "outside accepted generation intervals")

        bad_contract = make_fixture(root / "contract")
        contract = json.loads((bad_contract / "contract.json").read_text())
        contract["n_gen"] = 64
        (bad_contract / "contract.json").write_text(json.dumps(contract))
        expect_bad(bad_contract, "requires tg32")

        too_few = make_fixture(root / "too-few")
        contract = json.loads((too_few / "contract.json").read_text())
        contract["raw_repetitions"] = 5
        contract["accepted_repetitions"] = 4
        (too_few / "contract.json").write_text(json.dumps(contract))
        expect_bad(too_few, "at least 6 raw")

        bad_stack = make_fixture(root / "stack")
        contract = json.loads((bad_stack / "contract.json").read_text())
        contract["accepted_stack"]["mmq_j"] = 8
        (bad_stack / "contract.json").write_text(json.dumps(contract))
        expect_bad(bad_stack, "exact accepted J16")

        bad_status = make_fixture(root / "status")
        (bad_status / "status.txt").write_text(
            "process_exit_code=0\ntruncated=0\ntimeout_phase=measurement\nfinished_at_ns=1000000900\n"
        )
        expect_bad(bad_status, "status is not clean")

        bad_boundary = make_fixture(root / "boundary")
        boundary_text = (bad_boundary / "rocprof-selected-regions.tsv").read_text()
        (bad_boundary / "rocprof-selected-regions.tsv").write_text(boundary_text.replace("pause_call", "resume_return", 1))
        expect_bad(bad_boundary, "not an ordered resume/pause pair")

        missing_hip = make_fixture(root / "missing-hip")
        (missing_hip / "rocprof" / "dsv4-tg_hip_api_trace.csv").unlink()
        expect_bad(missing_hip, "expected one *_hip_api_trace.csv")

        bad_api = make_fixture(root / "bad-api")
        api_path = bad_api / "rocprof" / "dsv4-tg_hip_api_trace.csv"
        write_csv(api_path, ["Function", "Start_Timestamp", "End_Timestamp"], [{
            "Function": "hipGraphLaunch", "Start_Timestamp": 250, "End_Timestamp": 240,
        }])
        expect_bad(bad_api, "ends before it starts")

        bad_drift = make_fixture(root / "drift")
        (bad_drift / "clock-domain.txt").write_text(
            "start_realtime_minus_monotonic_ns=1000000000\n"
            "end_realtime_minus_monotonic_ns=1002000000\n"
            "start_calibration_span_ns=100\nend_calibration_span_ns=100\n"
        )
        expect_bad(bad_drift, "exceeds 1000000")
    print("dsv4 raw-TG profile fixtures: PASS")


if __name__ == "__main__":
    main()