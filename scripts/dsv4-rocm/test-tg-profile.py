#!/usr/bin/env python3
"""Non-GPU fixtures for selected-region raw-TG profile summarization."""

from __future__ import annotations

import csv
import json
import os
import pathlib
import shutil
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


def make_fixture(root: pathlib.Path) -> pathlib.Path:
    run = root / "good"
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
        "deepseek4.block_count=43\n"
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
    signatures = [
        ("void mul_mat_vec_q<(ggml_type)16, 1, false, false>(...)", 16384, 6, 84),
        ("void mul_mat_vec_q<(ggml_type)22, 1, false, false>(...)", 16384, 6, 2),
        ("void mul_mat_vec_q<(ggml_type)18, 1, false, false>(...)", 131072, 6, 41),
        ("void mul_mat_vec_q<(ggml_type)39, 1, false, false>(...)", 131072, 6, 2),
        ("void mul_mat_vec_q<(ggml_type)13, 1, false, false>(...)", 65536, 1, 84),
        ("void mul_mat_vec_q<(ggml_type)14, 1, false, false>(...)", 65536, 1, 2),
        ("void mul_mat_vec_q<(ggml_type)14, 1, true, false>(...)", 131072, 1, 42),
        ("void mul_mat_vec_q<(ggml_type)8, 1, true, false>(...)", 131072, 1, 1),
    ]
    attention_concat_signatures = [
        ("void concat_cont<unsigned short, 2>(unsigned short const*, unsigned short const*, unsigned short*, long, long, long, long, long, long)", 2490368, 21),
        ("void concat_cont<unsigned short, 2>(unsigned short const*, unsigned short const*, unsigned short*, long, long, long, long, long, long)", 393216, 20),
        ("void concat_cont<unsigned short, 0>(unsigned short const*, unsigned short const*, unsigned short*, long, long, long, long, long, long)", 4864, 21),
        ("void concat_cont<unsigned short, 0>(unsigned short const*, unsigned short const*, unsigned short*, long, long, long, long, long, long)", 768, 20),
    ]
    kernel_path = rocprof / "dsv4-tg_kernel_trace.csv"
    with kernel_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=kernel_fields)
        writer.writeheader()
        for rep in range(2, 7):
            start = (1000000100 + (rep - 1) * 120 + 10) - 1000000000 + 5
            for _token in range(32):
                for agent_index in range(1, 5):
                    for signature_index, (kernel_name, grid_x, grid_y, calls_per_token_agent) in enumerate(signatures):
                        for _call in range(calls_per_token_agent):
                            event_start = start + signature_index
                            writer.writerow({
                                "Agent_Id": f"Agent {agent_index}", "Kernel_Name": kernel_name,
                                "Start_Timestamp": event_start, "End_Timestamp": event_start + 1,
                                "Workgroup_Size_X": 32, "Workgroup_Size_Y": 1, "Workgroup_Size_Z": 1,
                                "Grid_Size_X": grid_x, "Grid_Size_Y": grid_y, "Grid_Size_Z": 1,
                            })
                    for concat_index, (kernel_name, grid_x, calls_per_token_agent) in enumerate(attention_concat_signatures):
                        for _call in range(calls_per_token_agent):
                            event_start = start + len(signatures) + concat_index
                            writer.writerow({
                                "Agent_Id": f"Agent {agent_index}", "Kernel_Name": kernel_name,
                                "Start_Timestamp": event_start, "End_Timestamp": event_start + 1,
                                "Workgroup_Size_X": 256, "Workgroup_Size_Y": 1, "Workgroup_Size_Z": 1,
                                "Grid_Size_X": grid_x, "Grid_Size_Y": 1, "Grid_Size_Z": 1,
                            })
            for index, (kernel_name, duration, grid_x, grid_y) in enumerate([
                ("void mul_mat_vec_q<(ggml_type)8, 1, false, false>(...)", 60, 262144, 1),
                ("ncclDevKernel_AllReduce", 10, 512, 1),
            ]):
                event_start = start + 20 + index * 20
                writer.writerow({
                    "Agent_Id": f"Agent {index + 1}", "Kernel_Name": kernel_name,
                    "Start_Timestamp": event_start, "End_Timestamp": event_start + duration,
                    "Workgroup_Size_X": 32, "Workgroup_Size_Y": 1, "Workgroup_Size_Z": 1,
                    "Grid_Size_X": grid_x, "Grid_Size_Y": grid_y, "Grid_Size_Z": 1,
                })
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


def clone_fixture(source: pathlib.Path, destination: pathlib.Path) -> pathlib.Path:
    shutil.copytree(source, destination, copy_function=os.link)
    return destination


def replace_text(path: pathlib.Path, text: str) -> None:
    path.unlink()
    path.write_text(text)


def replace_first(path: pathlib.Path, old: str, new: str) -> None:
    text = path.read_text()
    assert text.count(old) > 0
    path.unlink()
    path.write_text(text.replace(old, new, 1))


def move_first_kernel_outside(path: pathlib.Path) -> None:
    replacement = path.with_suffix(".replacement")
    with path.open(newline="") as source, replacement.open("w", newline="") as destination:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(destination, fieldnames=reader.fieldnames)
        writer.writeheader()
        first = next(reader)
        first["Start_Timestamp"] = "1"
        first["End_Timestamp"] = "2"
        writer.writerow(first)
        writer.writerows(reader)
    path.unlink()
    replacement.rename(path)


def remap_fixture_depth(run: pathlib.Path, depth: int, grids: dict[str, str]) -> None:
    contract_path = run / "contract.json"
    contract = json.loads(contract_path.read_text())
    contract["depths"] = [depth]
    replace_text(contract_path, json.dumps(contract))
    summary_path = run / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["records"][0]["depth"] = depth
    replace_text(summary_path, json.dumps(summary))
    boundaries = run / "rocprof-selected-regions.tsv"
    replace_text(boundaries, boundaries.read_text().replace("\t16384\t", f"\t{depth}\t"))
    kernel_path = run / "rocprof" / "dsv4-tg_kernel_trace.csv"
    replacement = kernel_path.with_suffix(".replacement")
    with kernel_path.open(newline="") as source, replacement.open("w", newline="") as destination:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(destination, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if row["Kernel_Name"].startswith("void concat_cont<unsigned short,"):
                row["Grid_Size_X"] = grids.get(row["Grid_Size_X"], row["Grid_Size_X"])
            writer.writerow(row)
    kernel_path.unlink()
    replacement.rename(kernel_path)


def mutate_first_concat(path: pathlib.Path, mutation: str) -> None:
    replacement = path.with_suffix(".replacement")
    changed = False
    with path.open(newline="") as source, replacement.open("w", newline="") as destination:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(destination, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if not changed and row["Kernel_Name"].startswith("void concat_cont<unsigned short, 2>"):
                if mutation == "agent":
                    assert row["Agent_Id"] == "Agent 1"
                    row["Agent_Id"] = "Agent 2"
                elif mutation == "repetition":
                    row["Start_Timestamp"], row["End_Timestamp"] = "360", "361"
                elif mutation == "near-name":
                    row["Kernel_Name"] = "not_the_kernel_" + row["Kernel_Name"]
                else:
                    raise AssertionError(mutation)
                changed = True
            writer.writerow(row)
    assert changed
    path.unlink()
    replacement.rename(path)


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
        assert value["kernel_dispatches"] == 217610
        assert value["outside_selected_interval_events"]["kernel"] == 0
        assert value["families"][0]["family"] == "routed_expert_quant_matmul"
        assert {row["family"] for row in value["families"]} == {
            "routed_expert_quant_matmul", "shared_expert_quant_matmul",
            "non_moe_quantized_matmul", "communication_nccl",
            "csa_k_concat", "hca_k_concat", "csa_mask_concat", "hca_mask_concat",
        }
        assert value["moe_roles"]["routed_gate_up"]["calls"] == 55040
        assert value["moe_roles"]["routed_down"]["calls"] == 27520
        assert value["moe_roles"]["shared_gate_up"]["calls"] == 55040
        assert value["moe_roles"]["shared_down"]["calls"] == 27520
        assert value["moe_signatures"]["routed_gate_up_type22"]["calls"] == 1280
        assert value["moe_signatures"]["routed_down_type39"]["calls"] == 1280
        assert value["moe_signatures"]["shared_gate_up_type14"]["calls"] == 1280
        assert value["moe_signatures"]["shared_down_type8"]["calls"] == 640
        assert value["attention_concat_roles"]["csa_k_concat"]["calls"] == 13440
        assert value["attention_concat_roles"]["hca_k_concat"]["calls"] == 12800
        assert value["attention_concat_roles"]["csa_mask_concat"]["calls"] == 13440
        assert value["attention_concat_roles"]["hca_mask_concat"]["calls"] == 12800
        assert value["attention_concat_dispatch_contract"] == {
            "applicable": True, "complete": True, "depth": 16384, "gpu_agents": 4, "target_tokens": 160,
            "actual_role_calls": {"csa_k_concat": 13440, "hca_k_concat": 12800, "csa_mask_concat": 13440, "hca_mask_concat": 12800},
            "expected_role_calls": {"csa_k_concat": 13440, "hca_k_concat": 12800, "csa_mask_concat": 13440, "hca_mask_concat": 12800},
            "per_agent_exact": True, "per_repetition_exact": True,
        }
        assert value["moe_dispatch_contract"] == {
            "complete": True, "block_count": 43, "gpu_agents": 4, "target_tokens": 160,
            "actual_role_calls": {"routed_gate_up": 55040, "routed_down": 27520, "shared_gate_up": 55040, "shared_down": 27520},
            "expected_role_calls": {"routed_gate_up": 55040, "routed_down": 27520, "shared_gate_up": 55040, "shared_down": 27520},
            "actual_signature_calls": {
                "routed_gate_up_type16": 53760, "routed_gate_up_type22": 1280,
                "routed_down_type18": 26240, "routed_down_type39": 1280,
                "shared_gate_up_type13": 53760, "shared_gate_up_type14": 1280,
                "shared_down_type14": 26880, "shared_down_type8": 640,
            },
            "expected_signature_calls": {
                "routed_gate_up_type16": 53760, "routed_gate_up_type22": 1280,
                "routed_down_type18": 26240, "routed_down_type39": 1280,
                "shared_gate_up_type13": 53760, "shared_gate_up_type14": 1280,
                "shared_down_type14": 26880, "shared_down_type8": 640,
            },
            "per_agent_exact": True, "per_repetition_exact": True,
        }
        assert value["profiled_wall_stable"] is True and value["profiled_throughput_eligible"] is False
        assert len(value["per_repetition"]) == 5
        assert value["per_repetition"][0]["trace_domains"]["rccl_api"] == {
            "calls": 1, "duration_ns": 10,
            "functions": {"ncclAllReduce": {"calls": 1, "duration_ns": 10}},
        }
        assert value["per_repetition"][1]["trace_domains"]["rccl_api"] == {
            "calls": 0, "duration_ns": 0, "functions": {},
        }
        assert value["trace_domain_status"] == {
            "kernel": "present_with_events", "memory_copy": "present_with_events",
            "rccl": "present_with_events", "hip": "present_with_events",
        }
        assert tsv.read_text().startswith("depth\tprofiled_repetitions\ttarget_tokens\tfamily")

        secondary_grids = clone_fixture(good, root / "secondary-grids")
        replace_first(secondary_grids / "rocprof" / "dsv4-tg_kernel_trace.csv", ",2490368,1,1", ",2359296,1,1")
        replace_first(secondary_grids / "rocprof" / "dsv4-tg_kernel_trace.csv", ",4864,1,1", ",4608,1,1")
        result = subprocess.run(
            [sys.executable, str(TOOL), str(secondary_grids)],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        assert result.returncode == 0, result.stderr

        depth64 = clone_fixture(good, root / "depth64")
        remap_fixture_depth(depth64, 65536, {
            "2490368": "8781824", "393216": "524288", "4864": "17152", "768": "1024",
        })
        replace_first(depth64 / "rocprof" / "dsv4-tg_kernel_trace.csv", ",8781824,1,1", ",8650752,1,1")
        replace_first(depth64 / "rocprof" / "dsv4-tg_kernel_trace.csv", ",17152,1,1", ",16896,1,1")
        depth64_out = depth64 / "profile.json"
        result = subprocess.run(
            [sys.executable, str(TOOL), str(depth64), "--json", str(depth64_out)],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        assert result.returncode == 0, result.stderr
        assert json.loads(depth64_out.read_text())["attention_concat_dispatch_contract"]["complete"] is True

        generic_depth = clone_fixture(good, root / "generic-depth")
        remap_fixture_depth(generic_depth, 32768, {})
        generic_out = generic_depth / "profile.json"
        result = subprocess.run(
            [sys.executable, str(TOOL), str(generic_depth), "--json", str(generic_out)],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        assert result.returncode == 0, result.stderr
        assert json.loads(generic_out.read_text())["attention_concat_dispatch_contract"] == {
            "applicable": False, "complete": False,
        }

        unstable = clone_fixture(good, root / "unstable")
        unstable_summary = json.loads((unstable / "summary.json").read_text())
        unstable_summary["stable"] = False
        unstable_summary["records"][0].update({"stable": False, "mad_over_median": 0.06})
        replace_text(unstable / "summary.json", json.dumps(unstable_summary))
        unstable_out = unstable / "unstable-profile.json"
        result = subprocess.run(
            [sys.executable, str(TOOL), str(unstable), "--json", str(unstable_out)],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        assert result.returncode == 0, result.stderr
        unstable_value = json.loads(unstable_out.read_text())
        assert unstable_value["profiled_wall_stable"] is False
        assert unstable_value["csa_decision_eligible"] is False
        assert unstable_value["family_attribution_complete"] is True

        outside = clone_fixture(good, root / "outside")
        move_first_kernel_outside(outside / "rocprof" / "dsv4-tg_kernel_trace.csv")
        expect_bad(outside, "outside accepted generation intervals")

        bad_contract = clone_fixture(good, root / "contract")
        contract = json.loads((bad_contract / "contract.json").read_text())
        contract["n_gen"] = 64
        replace_text(bad_contract / "contract.json", json.dumps(contract))
        expect_bad(bad_contract, "requires tg32")

        too_few = clone_fixture(good, root / "too-few")
        contract = json.loads((too_few / "contract.json").read_text())
        contract["raw_repetitions"] = 5
        contract["accepted_repetitions"] = 4
        replace_text(too_few / "contract.json", json.dumps(contract))
        expect_bad(too_few, "at least 6 raw")

        bad_stack = clone_fixture(good, root / "stack")
        contract = json.loads((bad_stack / "contract.json").read_text())
        contract["accepted_stack"]["mmq_j"] = 8
        replace_text(bad_stack / "contract.json", json.dumps(contract))
        expect_bad(bad_stack, "exact accepted J16")

        missing_block = clone_fixture(good, root / "missing-block")
        replace_first(missing_block / "manifest.txt", "deepseek4.block_count=43\n", "")
        expect_bad(missing_block, "expected exactly one")

        wrong_block = clone_fixture(good, root / "wrong-block")
        replace_first(wrong_block / "manifest.txt", "deepseek4.block_count=43", "deepseek4.block_count=42")
        expect_bad(wrong_block, "expected exact profile model value 43")

        bad_signature = clone_fixture(good, root / "signature")
        replace_first(
            bad_signature / "rocprof" / "dsv4-tg_kernel_trace.csv",
            "mul_mat_vec_q<(ggml_type)16", "mul_mat_vec_q<(ggml_type)22",
        )
        expect_bad(bad_signature, "signature dispatch contract mismatch")

        bad_concat_shape = clone_fixture(good, root / "concat-shape")
        replace_first(
            bad_concat_shape / "rocprof" / "dsv4-tg_kernel_trace.csv",
            ",2490368,1,1", ",2490369,1,1",
        )
        expect_bad(bad_concat_shape, "unknown exact DSV4 F16 attention concat signature")

        bad_concat_count = clone_fixture(good, root / "concat-count")
        replace_first(
            bad_concat_count / "rocprof" / "dsv4-tg_kernel_trace.csv",
            "concat_cont<unsigned short, 2>", "concat_cont<unsigned int, 2>",
        )
        expect_bad(bad_concat_count, "attention concat dispatch contract mismatch")

        bad_concat_name = clone_fixture(good, root / "concat-name")
        mutate_first_concat(bad_concat_name / "rocprof" / "dsv4-tg_kernel_trace.csv", "near-name")
        expect_bad(bad_concat_name, "attention concat dispatch contract mismatch")

        bad_concat_agent = clone_fixture(good, root / "concat-agent")
        mutate_first_concat(bad_concat_agent / "rocprof" / "dsv4-tg_kernel_trace.csv", "agent")
        expect_bad(bad_concat_agent, "attention concat per-agent dispatch mismatch")

        bad_concat_repetition = clone_fixture(good, root / "concat-repetition")
        mutate_first_concat(bad_concat_repetition / "rocprof" / "dsv4-tg_kernel_trace.csv", "repetition")
        expect_bad(bad_concat_repetition, "attention concat per-repetition dispatch mismatch")

        bad_status = clone_fixture(good, root / "status")
        replace_text(
            bad_status / "status.txt",
            "process_exit_code=0\ntruncated=0\ntimeout_phase=measurement\nfinished_at_ns=1000000900\n",
        )
        expect_bad(bad_status, "status is not clean")

        bad_boundary = clone_fixture(good, root / "boundary")
        replace_first(bad_boundary / "rocprof-selected-regions.tsv", "pause_call", "resume_return")
        expect_bad(bad_boundary, "not an ordered resume/pause pair")

        missing_hip = clone_fixture(good, root / "missing-hip")
        (missing_hip / "rocprof" / "dsv4-tg_hip_api_trace.csv").unlink()
        expect_bad(missing_hip, "expected one *_hip_api_trace.csv")

        bad_api = clone_fixture(good, root / "bad-api")
        api_path = bad_api / "rocprof" / "dsv4-tg_hip_api_trace.csv"
        api_path.unlink()
        write_csv(api_path, ["Function", "Start_Timestamp", "End_Timestamp"], [{
            "Function": "hipGraphLaunch", "Start_Timestamp": 250, "End_Timestamp": 240,
        }])
        expect_bad(bad_api, "ends before it starts")

        bad_drift = clone_fixture(good, root / "drift")
        replace_text(
            bad_drift / "clock-domain.txt",
            "start_realtime_minus_monotonic_ns=1000000000\n"
            "end_realtime_minus_monotonic_ns=1002000000\n"
            "start_calibration_span_ns=100\nend_calibration_span_ns=100\n",
        )
        expect_bad(bad_drift, "exceeds 1000000")
    print("dsv4 raw-TG profile fixtures: PASS")


if __name__ == "__main__":
    main()