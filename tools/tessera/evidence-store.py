#!/usr/bin/env python3
"""Append-only Parquet evidence store for Tessera calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl


SCHEMA = "llama.tessera.evidence.v1"
KINDS = ("observer", "router", "evolution", "shadow", "acceptance", "acceptance_position")


def import_gguf(path: str):
    resolved = str(Path(path).expanduser().resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
    from gguf import GGUFReader
    return GGUFReader


def stable_part_name(kind: str, run_id: str, source: Path) -> str:
    digest = hashlib.sha256(
        f"{kind}\0{run_id}\0{source.resolve()}\0{source.stat().st_size}\0{source.stat().st_mtime_ns}".encode()
    ).hexdigest()[:20]
    return f"part-{digest}.parquet"


def write_part(store: Path, kind: str, run_id: str, source: Path, frame: pl.DataFrame) -> Path:
    destination = store / kind / stable_part_name(kind, run_id, source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination
    temporary = destination.with_suffix(".tmp.parquet")
    frame.write_parquet(
        temporary,
        compression="zstd",
        statistics=True,
        row_group_size=65536,
    )
    temporary.replace(destination)
    return destination


def tensor_groups(reader) -> dict[str, dict[str, np.ndarray]]:
    suffixes = ("in_sum2", "in_sumabs", "in_sum4", "in_maxabs", "counts")
    grouped: dict[str, dict[str, np.ndarray]] = {}
    for tensor in reader.tensors:
        for suffix in suffixes:
            marker = f".{suffix}"
            if tensor.name.endswith(marker):
                grouped.setdefault(tensor.name[:-len(marker)], {})[suffix] = (
                    np.asarray(tensor.data, dtype=np.float32).reshape(-1)
                )
                break
    return grouped


def observer_frames(reader, run_id: str, source: Path, batch_rows: int) -> Iterable[pl.DataFrame]:
    records: list[dict] = []
    for tensor, stats in sorted(tensor_groups(reader).items()):
        if "in_sum2" not in stats or "counts" not in stats:
            continue
        counts = stats["counts"].reshape(-1)
        experts = counts.size
        sum2 = stats["in_sum2"].reshape(experts, -1)
        channels = sum2.shape[1]

        def shaped(key: str, fallback: np.ndarray) -> np.ndarray:
            value = stats.get(key)
            return value.reshape(experts, channels) if value is not None else fallback

        missing = np.full_like(sum2, np.nan)
        sumabs = shaped("in_sumabs", missing)
        sum4 = shaped("in_sum4", missing)
        maxabs = shaped("in_maxabs", missing)
        for expert in range(experts):
            count = float(counts[expert])
            denominator = max(count, 1.0)
            second = sum2[expert] / denominator
            fourth = sum4[expert] / denominator
            rms = np.sqrt(np.maximum(second, 0.0))
            mean_abs = sumabs[expert] / denominator
            kurtosis = fourth / np.maximum(np.square(second), 1e-20)
            tail_ratio = maxabs[expert] / np.maximum(rms, 1e-10)
            for channel in range(channels):
                records.append({
                    "schema": SCHEMA,
                    "run_id": run_id,
                    "source": str(source),
                    "tensor": tensor,
                    "expert": expert,
                    "channel": channel,
                    "count": count,
                    "sum2": float(sum2[expert, channel]),
                    "sumabs": float(sumabs[expert, channel]),
                    "sum4": float(sum4[expert, channel]),
                    "maxabs": float(maxabs[expert, channel]),
                    "rms": float(rms[channel]),
                    "mean_abs": float(mean_abs[channel]),
                    "kurtosis": float(kurtosis[channel]),
                    "tail_ratio": float(tail_ratio[channel]),
                })
                if len(records) >= batch_rows:
                    yield pl.DataFrame(records, infer_schema_length=None)
                    records.clear()
    if records:
        yield pl.DataFrame(records, infer_schema_length=None)


def router_frame(reader, run_id: str, source: Path) -> pl.DataFrame | None:
    grouped = tensor_groups(reader)
    records = []
    for tensor, stats in sorted(grouped.items()):
        if not tensor.endswith(".ffn_moe_router"):
            continue
        match = re.match(r"blk\.(\d+)\.", tensor)
        if match is None or "counts" not in stats or "in_sumabs" not in stats:
            continue
        layer = int(match.group(1))
        observations = int(stats["counts"].reshape(-1)[0])
        probability_sum = stats["in_sumabs"].reshape(-1)
        selected_counts = None
        for suffix in (
            "ffn_gate_up_exps.weight",
            "ffn_gate_exps.weight",
            "ffn_down_exps.weight",
        ):
            routed = grouped.get(f"blk.{layer}.{suffix}")
            if routed is not None and "counts" in routed:
                candidate = routed["counts"].reshape(-1)
                if candidate.size == probability_sum.size:
                    selected_counts = candidate
                    break
        if selected_counts is None:
            selected_counts = np.zeros(probability_sum.size, dtype=np.float32)
        for expert in range(probability_sum.size):
            records.append({
                "schema": SCHEMA,
                "run_id": run_id,
                "source": str(source),
                "layer": layer,
                "expert": expert,
                "observations": observations,
                "selected": int(selected_counts[expert]),
                "probability_sum": float(probability_sum[expert]),
                "confidence_sum": 0.0,
                "margin_sum": 0.0,
                "output_error_sum": 0.0,
                "downstream_divergence_sum": 0.0,
            })
    return pl.DataFrame(records, infer_schema_length=None) if records else None


def ingest_imatrix(args) -> None:
    source = Path(args.imatrix)
    reader = import_gguf(args.gguf_py)(str(source), "r")
    written = 0
    for batch_index, frame in enumerate(observer_frames(reader, args.run_id, source, args.batch_rows)):
        virtual_source = source.with_name(f"{source.name}.batch-{batch_index}")
        # The virtual source does not exist, so derive a deterministic name
        # from the real source and batch number without stable_part_name().
        digest = hashlib.sha256(
            f"observer\0{args.run_id}\0{source.resolve()}\0{source.stat().st_size}\0{batch_index}".encode()
        ).hexdigest()[:20]
        destination = Path(args.store) / "observer" / f"part-{digest}.parquet"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            temporary = destination.with_suffix(".tmp.parquet")
            frame.write_parquet(temporary, compression="zstd", statistics=True, row_group_size=65536)
            temporary.replace(destination)
        written += frame.height
    router = router_frame(reader, args.run_id, source)
    if router is not None:
        write_part(Path(args.store), "router", args.run_id, source, router)
    print(f"observer evidence: {written} rows in {Path(args.store) / 'observer'}")


def evolution_records(checkpoint: dict, run_id: str, source: Path, family: str) -> list[dict]:
    records = []
    for item in checkpoint.get("history", []):
        candidate = item["candidate"]
        score = item["score"]
        records.append({
            "schema": SCHEMA,
            "run_id": run_id,
            "source": str(source),
            "family": family,
            "record_type": "generation_best",
            "generation": int(item["generation"]),
            "cell": None,
            **{key: float(value) for key, value in candidate.items()},
            **{key: float(value) for key, value in score.items()},
        })
    for cell, item in checkpoint.get("archive", {}).items():
        candidate = item["candidate"]
        score = item["score"]
        records.append({
            "schema": SCHEMA,
            "run_id": run_id,
            "source": str(source),
            "family": family,
            "record_type": "map_elite",
            "generation": None,
            "cell": cell,
            **{key: float(value) for key, value in candidate.items()},
            **{key: float(value) for key, value in score.items()},
        })
    return records


def ingest_evolution(args) -> None:
    sources = sorted(Path().glob(args.checkpoint_glob)) if not Path(args.checkpoint_glob).is_absolute() else sorted(
        Path(args.checkpoint_glob).parent.glob(Path(args.checkpoint_glob).name)
    )
    if not sources:
        raise ValueError(f"no checkpoints matched {args.checkpoint_glob}")
    total = 0
    for source in sources:
        checkpoint = json.loads(source.read_text(encoding="utf-8"))
        if checkpoint.get("schema") != "llama.tessera.awq-evolution.v1":
            continue
        family = source.stem.rsplit(".", 1)[-1]
        records = evolution_records(checkpoint, args.run_id, source, family)
        if not records:
            continue
        frame = pl.DataFrame(records, infer_schema_length=None)
        write_part(Path(args.store), "evolution", args.run_id, source, frame)
        total += frame.height
    print(f"evolution evidence: {total} rows in {Path(args.store) / 'evolution'}")


def ingest_shadow(args) -> None:
    source = Path(args.policy)
    policy = json.loads(source.read_text(encoding="utf-8"))
    shadow = policy.get("tessera_shadow_calibration", {})
    if shadow.get("schema") != "llama.tessera.shadow-calibration.v1":
        raise ValueError(f"{source}: no Tessera shadow-calibration receipt")
    records = []
    for item in shadow.get("selected_overrides", []):
        candidate = item.get("refined_candidate", {})
        records.append({
            "schema": SCHEMA,
            "run_id": args.run_id,
            "source": str(source),
            "tensor": item["tensor"],
            "family": item["family"],
            "sample_count": float(item.get("sample_count", 0.0)),
            "coverage_uncertainty": float(item.get("coverage_uncertainty", 1.0)),
            "train_error": float(item["train_error"]),
            "heldout_error": float(item["heldout_error"]),
            "tail_error": float(item["tail_error"]),
            "shadow_error": float(item["shadow_error"]),
            "outlier_fraction": float(candidate.get("outlier_fraction", 0.0)),
            "awq_alpha": float(candidate.get("alpha", 0.0)),
            "awq_clip": float(candidate.get("clip", 1.0)),
        })
    if records:
        write_part(Path(args.store), "shadow", args.run_id, source,
                   pl.DataFrame(records, infer_schema_length=None))
    print(f"shadow evidence: {len(records)} rows in {Path(args.store) / 'shadow'}")


def ingest_acceptance(args) -> None:
    source = Path(args.telemetry)
    events: list[dict] = []
    positions: list[dict] = []
    with source.open("r", encoding="utf-8") as handle:
        for event_index, line in enumerate(handle):
            if not line.strip():
                continue
            event = json.loads(line)
            schema = event.get("schema", "")
            # The unified v3 schema (`llama.spec_calib.v3`) is emitted by the
            # dflash spec_calib path; the legacy `llama.dflash.acceptance.v1`
            # is the v1-compat adapter. MTP has its own schema.
            if schema in ("llama.dflash.acceptance.v1", "llama.spec_calib.v3"):
                draft_type = "dflash"
            elif "mtp" in schema:
                draft_type = "mtp"
            else:
                draft_type = "unknown"
            drafted = int(event["drafted"])
            accepted = int(event["accepted"])
            confidence = [float(value) for value in event.get("confidence", [])]
            events.append({
                "schema": SCHEMA,
                "run_id": args.run_id,
                "source": str(source),
                "event_index": event_index,
                "draft_type": draft_type,
                "drafted": drafted,
                "accepted": accepted,
                "acceptance": accepted / drafted if drafted else 0.0,
                "mean_confidence": sum(confidence) / len(confidence) if confidence else None,
            })
            for position, value in enumerate(confidence):
                positions.append({
                    "schema": SCHEMA,
                    "run_id": args.run_id,
                    "source": str(source),
                    "event_index": event_index,
                    "draft_type": draft_type,
                    "position": position,
                    "confidence": value,
                    "reached": accepted >= position,
                    "accepted": accepted > position,
                })
    if events:
        write_part(Path(args.store), "acceptance", args.run_id, source, pl.DataFrame(events))
    if positions:
        write_part(
            Path(args.store), "acceptance_position", args.run_id, source,
            pl.DataFrame(positions),
        )
    print(
        f"acceptance evidence: {len(events)} events, {len(positions)} positions "
        f"in {Path(args.store)}"
    )


def ingest_router(args) -> None:
    source = Path(args.telemetry)
    records = []
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("schema") != "llama.tessera.moe-router.v1":
                raise ValueError(f"{source}: unsupported router telemetry schema")
            observations = int(event["observations"])
            selected = int(event["selected"])
            if observations < 0 or selected < 0 or selected > observations:
                raise ValueError(f"{source}: invalid router population")
            records.append({
                "schema": SCHEMA,
                "run_id": args.run_id,
                "source": str(source),
                "layer": int(event["layer"]),
                "expert": int(event["expert"]),
                "observations": observations,
                "selected": selected,
                "probability_sum": float(event["probability_sum"]),
                "confidence_sum": float(event["confidence_sum"]),
                "margin_sum": float(event["margin_sum"]),
                "output_error_sum": float(event.get("output_error_sum", 0.0)),
                "downstream_divergence_sum": float(
                    event.get("downstream_divergence_sum", 0.0)
                ),
            })
    if records:
        write_part(
            Path(args.store), "router", args.run_id, source,
            pl.DataFrame(records, infer_schema_length=None),
        )
    print(f"router evidence: {len(records)} expert aggregates in {Path(args.store)}")


def scan_kind(store: Path, kind: str) -> pl.LazyFrame | None:
    files = list((store / kind).glob("*.parquet"))
    return pl.scan_parquet(str(store / kind / "*.parquet")) if files else None


def summarize(args) -> None:
    store = Path(args.store)
    summaries: list[pl.DataFrame] = []
    observer = scan_kind(store, "observer")
    if observer is not None:
        query = observer
        if args.run_id:
            query = query.filter(pl.col("run_id") == args.run_id)
        summaries.append(
            query.group_by("run_id").agg(
                pl.len().alias("observer_channels"),
                pl.col("tensor").n_unique().alias("observer_tensors"),
                pl.col("rms").mean().alias("mean_rms"),
                pl.col("kurtosis").quantile(0.99).alias("kurtosis_p99"),
                pl.col("tail_ratio").quantile(0.99).alias("tail_ratio_p99"),
            ).with_columns(pl.lit("observer").alias("kind")).collect(engine="streaming")
        )
    evolution = scan_kind(store, "evolution")
    if evolution is not None:
        query = evolution.filter(pl.col("record_type") == "generation_best")
        if args.run_id:
            query = query.filter(pl.col("run_id") == args.run_id)
        summaries.append(
            query.group_by(["run_id", "family"]).agg(
                pl.col("fitness").min().alias("best_fitness"),
                pl.col("generation").max().alias("last_generation"),
                pl.col("alpha").sort_by("fitness").first().alias("best_alpha"),
                pl.col("clip").sort_by("fitness").first().alias("best_clip"),
                pl.col("outlier_fraction").sort_by("fitness").first().alias("best_outlier_fraction"),
            ).with_columns(pl.lit("evolution").alias("kind")).collect(engine="streaming")
        )
    acceptance = scan_kind(store, "acceptance")
    if acceptance is not None:
        query = acceptance
        if args.run_id:
            query = query.filter(pl.col("run_id") == args.run_id)
        summaries.append(
            query.group_by(["run_id", "draft_type"]).agg(
                pl.len().alias("events"),
                pl.col("drafted").sum().alias("drafted"),
                pl.col("accepted").sum().alias("accepted"),
                (pl.col("accepted").sum() / pl.col("drafted").sum()).alias("acceptance"),
                pl.col("mean_confidence").mean().alias("mean_confidence"),
            ).with_columns(pl.lit("acceptance").alias("kind")).collect(engine="streaming")
        )
    router = scan_kind(store, "router")
    if router is not None:
        query = router
        if args.run_id:
            query = query.filter(pl.col("run_id") == args.run_id)
        summaries.append(
            query.group_by("run_id").agg(
                pl.col("observations").sum().alias("router_observations"),
                pl.col("layer").n_unique().alias("router_layers"),
                pl.col("expert").n_unique().alias("router_experts"),
                pl.col("selected").min().alias("least_expert_coverage"),
            ).with_columns(pl.lit("router").alias("kind")).collect(engine="streaming")
        )
    if not summaries:
        raise ValueError(f"{store}: no evidence partitions")
    output = pl.concat(summaries, how="diagonal_relaxed")
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        output.write_parquet(destination, compression="zstd", statistics=True)
    print(output)


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--store", required=True)
    parser.add_argument("--run-id", required=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage a Tessera Parquet calibration evidence store")
    subparsers = parser.add_subparsers(dest="command", required=True)
    imatrix = subparsers.add_parser("ingest-imatrix")
    add_common(imatrix)
    imatrix.add_argument("--imatrix", required=True)
    imatrix.add_argument("--gguf-py", default="/Users/user/Developer/GitHub/llama.cpp/gguf-py")
    imatrix.add_argument("--batch-rows", type=int, default=250000)
    imatrix.set_defaults(func=ingest_imatrix)
    evolution = subparsers.add_parser("ingest-evolution")
    add_common(evolution)
    evolution.add_argument("--checkpoint-glob", required=True)
    evolution.set_defaults(func=ingest_evolution)
    shadow = subparsers.add_parser("ingest-shadow")
    add_common(shadow)
    shadow.add_argument("--policy", required=True)
    shadow.set_defaults(func=ingest_shadow)
    acceptance = subparsers.add_parser("ingest-acceptance")
    add_common(acceptance)
    acceptance.add_argument("--telemetry", required=True)
    acceptance.set_defaults(func=ingest_acceptance)
    router = subparsers.add_parser("ingest-router")
    add_common(router)
    router.add_argument("--telemetry", required=True)
    router.set_defaults(func=ingest_router)
    summary = subparsers.add_parser("summarize")
    summary.add_argument("--store", required=True)
    summary.add_argument("--run-id", default=None)
    summary.add_argument("--output", default=None)
    summary.set_defaults(func=summarize)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
