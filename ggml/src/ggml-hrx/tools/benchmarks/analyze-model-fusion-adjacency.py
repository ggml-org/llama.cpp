#!/usr/bin/env python3
#
# Analyze HRX command-program adjacency for model-scoped Loom benchmarks.

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TRANSIENT_ORIGIN = "Transient"
READ_ACCESS = "Read"
WRITE_ACCESSES = {"Write", "ReadWrite"}


@dataclass(frozen=True)
class BindingKey:
    value: int
    offset: int
    length: int


@dataclass(frozen=True)
class Producer:
    ordinal: int
    kernel: str
    binding: str
    key: BindingKey


@dataclass(frozen=True)
class Consumer:
    ordinal: int
    kernel: str
    binding: str


def fail(message: str) -> None:
    raise SystemExit(message)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def kernel_symbol(kernel: str) -> str:
    return kernel.split(":")[-1]


def command_shape_key(command: dict[str, Any]) -> str:
    shape_data = {
        "kernel": command.get("kernel"),
        "integer_parameters": command.get("integer_parameters", {}),
        "compile_parameters": command.get("compile_parameters", {}),
        "binding_lengths": [binding.get("length") for binding in command.get("bindings", [])],
    }
    encoded = json.dumps(shape_data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def load_programs(dump_dir: Path) -> list[dict[str, Any]]:
    paths = sorted(dump_dir.glob("program-*/program.json"))
    if not paths:
        fail(f"no program.json files found under {dump_dir}")
    programs = []
    for path in paths:
        program = load_json(path)
        program["path"] = str(path)
        programs.append(program)
    return programs


def load_commands(dump_dir: Path) -> list[dict[str, Any]]:
    commands = []
    for program in load_programs(dump_dir):
        for command in program.get("commands", []):
            command = dict(command)
            command["program"] = {
                "directory": Path(program["path"]).parent.name,
                "dump_id": program.get("dump_id"),
                "shape_hash": program.get("shape_hash"),
                "target": program.get("target"),
            }
            commands.append(command)
    return commands


def shape_key_order(commands: list[dict[str, Any]]) -> list[str]:
    keys = []
    seen = set()
    for command in commands:
        key = command_shape_key(command)
        if key not in seen:
            seen.add(key)
            keys.append(key)
    return keys


def shape_counts(commands: list[dict[str, Any]]) -> Counter[str]:
    return Counter(command_shape_key(command) for command in commands)


def shape_metrics_from_summary(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    by_benchmark = {}
    for kernel in summary.get("kernels", []):
        for shape in kernel.get("shapes", []):
            benchmark = shape.get("benchmark")
            if benchmark:
                by_benchmark[benchmark] = shape
    return by_benchmark


def map_shape_keys_to_dispatches(commands: list[dict[str, Any]],
                                 manifest: dict[str, Any],
                                 summary: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    warnings = []
    keys = shape_key_order(commands)
    dispatches = manifest.get("dispatches", [])
    if len(keys) != len(dispatches):
        warnings.append(f"shape count mismatch: dump has {len(keys)} unique shapes, manifest has {len(dispatches)} dispatches")

    summary_by_benchmark = shape_metrics_from_summary(summary)
    shape_info = {}
    for index, key in enumerate(keys[:len(dispatches)]):
        dispatch = dispatches[index]
        command = next(command for command in commands if command_shape_key(command) == key)
        if command.get("kernel") != dispatch.get("kernel"):
            warnings.append(
                f"shape {index} kernel mismatch: dump has {command.get('kernel')}, manifest has {dispatch.get('kernel')}"
            )
        benchmark = dispatch.get("benchmark")
        metric = summary_by_benchmark.get(benchmark, {})
        shape_info[key] = {
            "benchmark": benchmark,
            "kernel": dispatch.get("kernel"),
            "count": dispatch.get("count", 0),
            "metric_ns": metric.get("metric_ns"),
            "weighted_ns": metric.get("weighted_ns"),
            "state": metric.get("state", "missing_summary"),
            "error": metric.get("error"),
        }
    return shape_info, warnings


def command_metric_ns(command: dict[str, Any], shape_info: dict[str, dict[str, Any]]) -> float:
    info = shape_info.get(command_shape_key(command), {})
    metric = info.get("metric_ns")
    if metric is None:
        return 0.0
    return float(metric)


def binding_key(binding: dict[str, Any]) -> BindingKey:
    return BindingKey(int(binding["value"]), int(binding.get("offset", 0)), int(binding.get("length", 0)))


def read_bindings(command: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        binding
        for binding in command.get("bindings", [])
        if binding.get("origin") == TRANSIENT_ORIGIN and binding.get("access") == READ_ACCESS
    ]


def write_bindings(command: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        binding
        for binding in command.get("bindings", [])
        if binding.get("origin") == TRANSIENT_ORIGIN and binding.get("access") in WRITE_ACCESSES
    ]


def exact_transient_edges(commands: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[Producer, list[Consumer]]]:
    active: dict[BindingKey, Producer] = {}
    edges = []
    fanout: dict[Producer, list[Consumer]] = defaultdict(list)
    for command in sorted(commands, key=lambda item: int(item["ordinal"])):
        consumer_ordinal = int(command["ordinal"])
        for binding in read_bindings(command):
            key = binding_key(binding)
            producer = active.get(key)
            if producer is None:
                continue
            consumer = Consumer(consumer_ordinal, command["kernel"], str(binding.get("name", "")))
            edges.append(
                {
                    "producer_ordinal": producer.ordinal,
                    "producer_kernel": producer.kernel,
                    "producer_binding": producer.binding,
                    "consumer_ordinal": consumer.ordinal,
                    "consumer_kernel": consumer.kernel,
                    "consumer_binding": consumer.binding,
                    "value": key.value,
                    "offset": key.offset,
                    "length": key.length,
                }
            )
            fanout[producer].append(consumer)
        for binding in write_bindings(command):
            key = binding_key(binding)
            active[key] = Producer(consumer_ordinal, command["kernel"], str(binding.get("name", "")), key)
    return edges, fanout


def summarize_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str]] = Counter()
    examples: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        key = (edge["producer_kernel"], edge["consumer_kernel"])
        counts[key] += 1
        if len(examples[key]) < 5:
            examples[key].append(edge)
    return [
        {
            "producer_kernel": producer,
            "consumer_kernel": consumer,
            "edge_count": count,
            "examples": examples[(producer, consumer)],
        }
        for (producer, consumer), count in counts.most_common()
    ]


def summarize_fanout(fanout: dict[Producer, list[Consumer]]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, tuple[str, ...]]] = Counter()
    examples: dict[tuple[str, tuple[str, ...]], list[dict[str, Any]]] = defaultdict(list)
    for producer, consumers in fanout.items():
        consumer_kernels = tuple(sorted(consumer.kernel for consumer in consumers))
        key = (producer.kernel, consumer_kernels)
        counts[key] += 1
        if len(examples[key]) < 5:
            examples[key].append(
                {
                    "producer_ordinal": producer.ordinal,
                    "producer_binding": producer.binding,
                    "consumers": [
                        {
                            "ordinal": consumer.ordinal,
                            "kernel": consumer.kernel,
                            "binding": consumer.binding,
                        }
                        for consumer in consumers
                    ],
                }
            )
    rows = []
    for (producer_kernel, consumer_kernels), count in counts.most_common():
        rows.append(
            {
                "producer_kernel": producer_kernel,
                "consumer_kernels": list(consumer_kernels),
                "producer_count": count,
                "examples": examples[(producer_kernel, consumer_kernels)],
            }
        )
    return rows


def producers_by_exact_key(commands: list[dict[str, Any]]) -> dict[tuple[int, BindingKey], Producer]:
    active: dict[BindingKey, Producer] = {}
    result = {}
    for command in sorted(commands, key=lambda item: int(item["ordinal"])):
        ordinal = int(command["ordinal"])
        for binding in read_bindings(command):
            key = binding_key(binding)
            producer = active.get(key)
            if producer is not None:
                result[(ordinal, key)] = producer
        for binding in write_bindings(command):
            key = binding_key(binding)
            active[key] = Producer(ordinal, command["kernel"], str(binding.get("name", "")), key)
    return result


def producer_for_read(command: dict[str, Any],
                      binding_name: str,
                      read_producers: dict[tuple[int, BindingKey], Producer]) -> Producer | None:
    ordinal = int(command["ordinal"])
    for binding in read_bindings(command):
        if binding.get("name") == binding_name:
            return read_producers.get((ordinal, binding_key(binding)))
    return None


def output_consumers(command: dict[str, Any], fanout: dict[Producer, list[Consumer]], binding_name: str = "output") -> list[Consumer]:
    ordinal = int(command["ordinal"])
    for binding in write_bindings(command):
        if binding.get("name") != binding_name:
            continue
        producer = Producer(ordinal, command["kernel"], str(binding.get("name", "")), binding_key(binding))
        return fanout.get(producer, [])
    return []


def command_by_ordinal(commands: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(command["ordinal"]): command for command in commands}


def compile_parameters(command: dict[str, Any]) -> dict[str, Any]:
    return command.get("compile_parameters", {})


def integer_parameters(command: dict[str, Any]) -> dict[str, Any]:
    return command.get("integer_parameters", {})


def op_name(command: dict[str, Any]) -> str:
    if command["kernel"] != "loom_libs:ggml_binary_f32":
        return ""
    op = str(compile_parameters(command).get("ggml.binary_f32.op", ""))
    names = {
        "0": "add",
        "1": "sub",
        "2": "mul",
        "3": "div",
        "4": "swiglu",
        "5": "geglu",
        "6": "reglu",
        "7": "geglu_erf",
        "8": "geglu_quick",
    }
    return names.get(op, f"op_{op}")


def add_candidate(candidates: list[dict[str, Any]],
                  name: str,
                  kind: str,
                  occurrences: list[dict[str, Any]],
                  current_chain_ns: float,
                  removable_ns: float,
                  notes: list[str]) -> None:
    candidates.append(
        {
            "name": name,
            "kind": kind,
            "occurrence_count": len(occurrences),
            "current_chain_ns": current_chain_ns,
            "removable_standalone_ns": removable_ns,
            "examples": occurrences[:8],
            "notes": notes,
        }
    )


def classify_candidates(commands: list[dict[str, Any]],
                        shape_info: dict[str, dict[str, Any]],
                        fanout: dict[Producer, list[Consumer]]) -> list[dict[str, Any]]:
    by_ordinal = command_by_ordinal(commands)
    read_producers = producers_by_exact_key(commands)
    candidates: list[dict[str, Any]] = []

    attention_occurrences = []
    attention_chain_ns = 0.0
    attention_removable_ns = 0.0
    for command in commands:
        if command["kernel"] != "loom_libs:ggml_flash_attention_decode_split_f32_f16_wmma_next_q8":
            continue
        ordinal = int(command["ordinal"])
        rope = by_ordinal.get(ordinal - 3)
        rope_set_rows = by_ordinal.get(ordinal - 2)
        set_rows = by_ordinal.get(ordinal - 1)
        q_projection = producer_for_read(rope, "input", read_producers) if rope else None
        k_projection = producer_for_read(rope_set_rows, "input", read_producers) if rope_set_rows else None
        v_projection = producer_for_read(set_rows, "rows", read_producers) if set_rows else None
        matched = (
            rope is not None
            and rope_set_rows is not None
            and set_rows is not None
            and rope["kernel"] == "loom_libs:ggml_rope_f32"
            and rope_set_rows["kernel"] == "loom_libs:ggml_rope_set_rows_f32"
            and set_rows["kernel"] == "loom_libs:ggml_set_rows"
            and q_projection is not None
            and k_projection is not None
            and v_projection is not None
            and q_projection.kernel == "loom_libs:ggml_mul_mat_f32_f32_decode_wave64"
            and k_projection.kernel == "loom_libs:ggml_mul_mat_f32_f32_decode_wave64"
            and v_projection.kernel == "loom_libs:ggml_mul_mat_f32_f32_decode_wave64"
        )
        if not matched:
            continue
        occurrence_ns = sum(command_metric_ns(item, shape_info) for item in (rope, rope_set_rows, set_rows))
        attention_chain_ns += occurrence_ns + command_metric_ns(command, shape_info)
        attention_removable_ns += occurrence_ns
        attention_occurrences.append(
            {
                "attention_ordinal": ordinal,
                "q_projection_ordinal": q_projection.ordinal,
                "q_rope_ordinal": int(rope["ordinal"]),
                "k_projection_ordinal": k_projection.ordinal,
                "k_rope_set_rows_ordinal": int(rope_set_rows["ordinal"]),
                "v_projection_ordinal": v_projection.ordinal,
                "v_set_rows_ordinal": int(set_rows["ordinal"]),
                "postprocess_ns": occurrence_ns,
                "head": {
                    "query_heads": compile_parameters(command).get("ggml.flash_attention.query_head_count"),
                    "key_value_heads": compile_parameters(command).get("ggml.flash_attention.key_value_head_count"),
                    "qk_head_size": compile_parameters(command).get("ggml.flash_attention.qk_head_size"),
                    "value_head_size": compile_parameters(command).get("ggml.flash_attention.value_head_size"),
                },
            }
        )
    add_candidate(
        candidates,
        "decode attention projection postprocess",
        "attention_qkv_decode",
        attention_occurrences,
        attention_chain_ns,
        attention_removable_ns,
        [
            "Current decode path uses standalone Q rope, K rope+cache write, and V cache write before split flash attention.",
            "Existing llm_attention_qkv fused projection matchers require token_count > 1, so this looks like decode-route coverage is missing.",
        ],
    )

    swiglu_occurrences = []
    swiglu_chain_ns = 0.0
    swiglu_removable_ns = 0.0
    residual_occurrences = []
    residual_chain_ns = 0.0
    residual_removable_ns = 0.0
    tail_occurrences = []
    tail_chain_ns = 0.0
    tail_removable_ns = 0.0
    for command in commands:
        if command["kernel"] != "loom_libs:ggml_binary_f32":
            continue
        ordinal = int(command["ordinal"])
        current_ns = command_metric_ns(command, shape_info)
        consumers = output_consumers(command, fanout)
        lhs = producer_for_read(command, "lhs", read_producers)
        rhs = producer_for_read(command, "rhs", read_producers)
        op = op_name(command)
        if op == "swiglu":
            down_projection = consumers[0] if len(consumers) == 1 else None
            occurrence = {
                "binary_ordinal": ordinal,
                "lhs_producer": lhs.ordinal if lhs else None,
                "rhs_producer": rhs.ordinal if rhs else None,
                "consumer": down_projection.ordinal if down_projection else None,
                "element_count": integer_parameters(command).get("element_count"),
                "binary_ns": current_ns,
            }
            swiglu_occurrences.append(occurrence)
            swiglu_removable_ns += current_ns
            swiglu_chain_ns += current_ns
            if lhs is not None:
                swiglu_chain_ns += command_metric_ns(by_ordinal[lhs.ordinal], shape_info)
            if rhs is not None:
                swiglu_chain_ns += command_metric_ns(by_ordinal[rhs.ordinal], shape_info)
            if down_projection is not None:
                swiglu_chain_ns += command_metric_ns(by_ordinal[down_projection.ordinal], shape_info)
        elif op == "add":
            consumer_kernels = [consumer.kernel for consumer in consumers]
            occurrence = {
                "binary_ordinal": ordinal,
                "lhs_producer": lhs.ordinal if lhs else None,
                "rhs_producer": rhs.ordinal if rhs else None,
                "consumers": [
                    {
                        "ordinal": consumer.ordinal,
                        "kernel": consumer.kernel,
                        "binding": consumer.binding,
                    }
                    for consumer in consumers
                ],
                "element_count": integer_parameters(command).get("element_count"),
                "binary_ns": current_ns,
            }
            if "loom_libs:ggml_rmsnorm_binary_f32" in consumer_kernels:
                residual_occurrences.append(occurrence)
                residual_removable_ns += current_ns
                residual_chain_ns += current_ns
                for consumer in consumers:
                    if consumer.kernel == "loom_libs:ggml_rmsnorm_binary_f32":
                        residual_chain_ns += command_metric_ns(by_ordinal[consumer.ordinal], shape_info)
            if any(consumer.kernel == "loom_libs:ggml_rmsnorm_binary_f32" for consumer in consumers) and (
                (lhs and lhs.kernel == "hrx:ggml_gather_add_f32") or (rhs and rhs.kernel == "hrx:ggml_gather_add_f32")
            ):
                tail_occurrences.append(occurrence)
                tail_removable_ns += current_ns
                tail_chain_ns += current_ns
                for consumer in consumers:
                    tail_chain_ns += command_metric_ns(by_ordinal[consumer.ordinal], shape_info)

    add_candidate(
        candidates,
        "decode SwiGLU MLP",
        "swiglu_decode",
        swiglu_occurrences,
        swiglu_chain_ns,
        swiglu_removable_ns,
        [
            "Pattern is two decode matmuls feeding binary_f32 op=4, then a down projection.",
            "This is separate from existing binary_swiglu_symmetric_i4_k32 coverage.",
        ],
    )
    add_candidate(
        candidates,
        "residual add next RMS norm",
        "add_next_rmsnorm_f32",
        residual_occurrences,
        residual_chain_ns,
        residual_removable_ns,
        [
            "Pattern is binary_f32 op=0 feeding rmsnorm_binary_f32.",
            "Some add outputs have multiple consumers, so the fused kernel must still publish the residual/add output when it is reused.",
        ],
    )
    add_candidate(
        candidates,
        "final output tail add norm",
        "output_tail",
        tail_occurrences,
        tail_chain_ns,
        tail_removable_ns,
        [
            "Low-count final path involving gather_add, binary add, RMS norm, and output projection.",
        ],
    )

    matmul_add_norm_occurrences = []
    matmul_add_norm_chain_ns = 0.0
    for command in commands:
        if command["kernel"] != "loom_libs:ggml_add_rmsnorm_binary_symmetric_i4_k32":
            continue
        lhs = producer_for_read(command, "lhs", read_producers)
        if lhs is None or lhs.kernel != "loom_libs:ggml_mul_mat_f32_f32_decode_wave64":
            continue
        ordinal = int(command["ordinal"])
        current_ns = command_metric_ns(command, shape_info)
        lhs_ns = command_metric_ns(by_ordinal[lhs.ordinal], shape_info)
        matmul_add_norm_chain_ns += current_ns + lhs_ns
        matmul_add_norm_occurrences.append(
            {
                "add_rmsnorm_ordinal": ordinal,
                "matmul_ordinal": lhs.ordinal,
                "hidden_size": compile_parameters(command).get("ggml.add_rmsnorm_binary_symmetric_i4.hidden_size"),
                "chain_ns": current_ns + lhs_ns,
            }
        )
    add_candidate(
        candidates,
        "decode matmul add RMS norm",
        "matmul_add_next_rmsnorm_decode",
        matmul_add_norm_occurrences,
        matmul_add_norm_chain_ns,
        0.0,
        [
            "Pattern is decode matmul feeding add_rmsnorm_binary_symmetric_i4.",
            "Savings depend on a decode matmul epilogue or fused next-rmsnorm route, so removable standalone time is not estimated here.",
        ],
    )

    total_removable = sum(candidate["removable_standalone_ns"] for candidate in candidates)
    for candidate in candidates:
        candidate["current_chain_ms"] = candidate["current_chain_ns"] / 1_000_000.0
        candidate["removable_standalone_ms"] = candidate["removable_standalone_ns"] / 1_000_000.0
        candidate["removable_share_of_candidates"] = (
            candidate["removable_standalone_ns"] / total_removable * 100.0 if total_removable else 0.0
        )
    return sorted(candidates, key=lambda item: (item["removable_standalone_ns"], item["current_chain_ns"]), reverse=True)


def sequential_triples(commands: list[dict[str, Any]], limit: int = 30) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str, str]] = Counter()
    ordered = sorted(commands, key=lambda item: int(item["ordinal"]))
    for first, second, third in zip(ordered, ordered[1:], ordered[2:], strict=False):
        counts[(first["kernel"], second["kernel"], third["kernel"])] += 1
    return [
        {"kernels": list(kernels), "count": count}
        for kernels, count in counts.most_common(limit)
    ]


def build_report(commands: list[dict[str, Any]],
                 manifest: dict[str, Any],
                 summary: dict[str, Any]) -> dict[str, Any]:
    shape_info, warnings = map_shape_keys_to_dispatches(commands, manifest, summary)
    edges, fanout = exact_transient_edges(commands)
    candidates = classify_candidates(commands, shape_info, fanout)
    total_weighted_ns = float(summary.get("total_weighted_ns", 0.0))
    for candidate in candidates:
        candidate["removable_share_of_total_modelled_kernel_time"] = (
            candidate["removable_standalone_ns"] / total_weighted_ns * 100.0 if total_weighted_ns else 0.0
        )
        candidate["current_chain_share_of_total_modelled_kernel_time"] = (
            candidate["current_chain_ns"] / total_weighted_ns * 100.0 if total_weighted_ns else 0.0
        )
    return {
        "schema": "ggml-hrx-model-fusion-adjacency-v1",
        "model": manifest.get("model"),
        "scenario": manifest.get("scenario"),
        "command_count": len(commands),
        "dispatch_count": manifest.get("dispatch_count"),
        "generated_count": manifest.get("generated_count"),
        "summary_metric": summary.get("metric_description"),
        "total_weighted_ns": total_weighted_ns,
        "warnings": warnings,
        "candidate_fusions": candidates,
        "exact_transient_edge_summary": summarize_edges(edges),
        "fanout_summary": summarize_fanout(fanout),
        "sequential_triples": sequential_triples(commands),
    }


def md_escape(text: Any) -> str:
    return str(text).replace("|", "\\|")


def format_ms(ns: float) -> str:
    return f"{ns / 1_000_000.0:.3f}"


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# HRX Model Fusion Adjacency",
        "",
        f"Model: `{report.get('model')}`",
        f"Scenario: `{report.get('scenario')}`",
        f"Commands: `{report.get('command_count')}`",
        f"Generated dispatch shapes: `{report.get('generated_count')} / {report.get('dispatch_count')}`",
        f"Timing metric: `{report.get('summary_metric')}`",
        f"Total weighted modeled kernel time: `{format_ms(float(report.get('total_weighted_ns', 0.0)))} ms`",
        "",
    ]
    if report.get("warnings"):
        lines.append("## Warnings")
        lines.append("")
        for warning in report["warnings"]:
            lines.append(f"- {warning}")
        lines.append("")

    lines.extend(
        [
            "## Candidate Fusions",
            "",
            "| Candidate | Occurrences | Current chain ms | Removable standalone ms | Current chain share | Removable share | Notes |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for candidate in report["candidate_fusions"]:
        notes = "<br>".join(md_escape(note) for note in candidate.get("notes", []))
        lines.append(
            f"| {md_escape(candidate['name'])} | {candidate['occurrence_count']} | "
            f"{candidate['current_chain_ms']:.3f} | {candidate['removable_standalone_ms']:.3f} | "
            f"{candidate['current_chain_share_of_total_modelled_kernel_time']:.2f}% | "
            f"{candidate['removable_share_of_total_modelled_kernel_time']:.2f}% | {notes} |"
        )

    lines.extend(
        [
            "",
            "## Exact Transient Producer-Consumer Edges",
            "",
            "| Producer | Consumer | Edges | Example ordinals |",
            "| --- | --- | ---: | --- |",
        ]
    )
    for edge in report["exact_transient_edge_summary"][:30]:
        examples = ", ".join(
            f"{example['producer_ordinal']}:{example['producer_binding']}->{example['consumer_ordinal']}:{example['consumer_binding']}"
            for example in edge.get("examples", [])[:3]
        )
        lines.append(
            f"| `{edge['producer_kernel']}` | `{edge['consumer_kernel']}` | {edge['edge_count']} | {md_escape(examples)} |"
        )

    lines.extend(
        [
            "",
            "## Fanout Patterns",
            "",
            "| Producer | Consumers | Produced values |",
            "| --- | --- | ---: |",
        ]
    )
    for row in report["fanout_summary"][:25]:
        consumers = ", ".join(f"`{kernel}`" for kernel in row["consumer_kernels"])
        lines.append(f"| `{row['producer_kernel']}` | {consumers} | {row['producer_count']} |")

    lines.extend(
        [
            "",
            "## Top Sequential Triples",
            "",
            "| Kernels | Count |",
            "| --- | ---: |",
        ]
    )
    for triple in report["sequential_triples"][:20]:
        kernels = " -> ".join(f"`{kernel}`" for kernel in triple["kernels"])
        lines.append(f"| {kernels} | {triple['count']} |")

    lines.extend(
        [
            "",
            "## Candidate Examples",
            "",
        ]
    )
    for candidate in report["candidate_fusions"]:
        lines.append(f"### {candidate['name']}")
        lines.append("")
        if not candidate.get("examples"):
            lines.append("No matching occurrences.")
            lines.append("")
            continue
        lines.append("```json")
        lines.append(json.dumps(candidate["examples"][:3], indent=2, sort_keys=True))
        lines.append("```")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-dir", type=Path, required=True, help="HRX command-program dump directory.")
    parser.add_argument("--scenario-manifest", type=Path, required=True, help="Generated model scenario manifest.")
    parser.add_argument("--summary-json", type=Path, required=True, help="Summarized Loom benchmark JSON.")
    parser.add_argument("--output-json", type=Path, required=True, help="Fusion adjacency JSON output.")
    parser.add_argument("--output-md", type=Path, required=True, help="Fusion adjacency Markdown output.")
    args = parser.parse_args()

    commands = load_commands(args.dump_dir)
    manifest = load_json(args.scenario_manifest)
    summary = load_json(args.summary_json)
    report = build_report(commands, manifest, summary)
    write_json(args.output_json, report)
    write_markdown(args.output_md, report)
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
