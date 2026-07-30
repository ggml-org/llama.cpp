#!/usr/bin/env python3
"""Publish and bootstrap privacy-safe Tessera evidence on Hugging Face."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import tempfile
from pathlib import Path

import polars as pl
from huggingface_hub import HfApi, hf_hub_download, snapshot_download


SCHEMA = "llama.tessera.evidence.v1"
COMMONS_SCHEMA = "llama.tessera.hf-evidence.v1"
EPOCH_SCHEMA = "llama.tessera.epoch.v1"
MODEL_EPOCH_SCHEMA = "llama.tessera.model-epoch.v1"
DEFAULT_REPO = "juliantorr/tessera-calibration-commons"
CONTRIBUTION_LICENSE_ID = "Tessera-Calibration-Contribution-1.0"
CONTRIBUTION_LICENSE_GRANTEE = "Julian Alejandro Torres Nieto"
CONTRIBUTION_LICENSE_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "TESSERA_CALIBRATION_CONTRIBUTION_TERMS_1.0.md"
)
CONTRIBUTION_LICENSE_REPO_PATH = (
    "licenses/TESSERA_CALIBRATION_CONTRIBUTION_TERMS_1.0.md"
)
PUBLIC_LICENSE_ID = "CC-BY-NC-SA-4.0"
PUBLIC_LICENSE_URI = "https://creativecommons.org/licenses/by-nc-sa/4.0/"
PUBLIC_LICENSE_ATTRIBUTION = (
    "Julian Alejandro Torres Nieto, Tribunus.dev"
)
PUBLIC_LICENSE_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "TESSERA_ARTIFACT_LICENSE_NOTICE.md"
)
PUBLIC_LICENSE_REPO_PATH = "licenses/TESSERA_ARTIFACT_LICENSE_NOTICE_1.0.md"
FORBIDDEN_COLUMN_FRAGMENTS = (
    "prompt", "completion", "generation", "token_id", "embedding", "logit",
    "request", "session", "user", "email", "ip", "filename", "path",
    "timestamp", "media", "image", "audio", "hash", "digest", "source",
)
OBSERVER_COLUMNS = {
    "tensor", "expert", "channel", "count", "sum2", "sumabs", "sum4",
    "maxabs", "rms", "mean_abs", "kurtosis", "tail_ratio",
}
ACCEPTANCE_COLUMNS = {
    "draft_type", "position", "observations", "reached", "accepted",
    "confidence_sum",
}
ROUTER_COLUMNS = {
    "layer", "expert", "observations", "selected", "probability_sum",
    "confidence_sum", "margin_sum", "output_error_sum",
    "downstream_divergence_sum", "frequency", "mean_confidence", "mean_margin",
    "mean_output_error", "mean_downstream_divergence",
}
VOLATILE_CONFIG_KEYS = {
    "_name_or_path",
    "dtype",
    "torch_dtype",
    "transformers_version",
    "quantization_config",
    "id2label",
    "label2id",
}


def structural_config(value):
    if isinstance(value, dict):
        return {
            key: structural_config(child)
            for key, child in sorted(value.items())
            if key not in VOLATILE_CONFIG_KEYS
        }
    if isinstance(value, list):
        return [structural_config(child) for child in value]
    return value


def model_identity(model_dir: Path) -> tuple[str, dict]:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise ValueError(f"{model_dir}: missing config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    identity = {
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures", []),
        "text_config": structural_config(config.get("text_config", config)),
        "vision_config": structural_config(config.get("vision_config")),
        "audio_config": structural_config(config.get("audio_config")),
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:24], identity


def scan(store: Path, kind: str, run_id: str | None) -> pl.LazyFrame | None:
    if not list((store / kind).glob("*.parquet")):
        return None
    frame = pl.scan_parquet(str(store / kind / "*.parquet"))
    return frame.filter(pl.col("run_id") == run_id) if run_id else frame


def audit_columns(frame: pl.DataFrame, allowed: set[str], label: str) -> None:
    unexpected = set(frame.columns) - allowed
    forbidden = {
        column for column in frame.columns
        if any(fragment in column.lower() for fragment in FORBIDDEN_COLUMN_FRAGMENTS)
    }
    if unexpected or forbidden:
        raise ValueError(
            f"{label}: privacy schema rejected columns "
            f"unexpected={sorted(unexpected)} forbidden={sorted(forbidden)}"
        )


def aggregate_observer(
    store: Path,
    run_id: str | None,
    min_tokens: int = 100_000,
    max_tail_ratio: float = 32.0,
) -> pl.DataFrame | None:
    frame = scan(store, "observer", run_id)
    if frame is None:
        return None
    grouped = frame.group_by(["tensor", "expert", "channel"]).agg(
        pl.col("count").sum().alias("count"),
        pl.col("sum2").sum().alias("sum2"),
        pl.col("sumabs").sum().alias("sumabs"),
        pl.col("sum4").sum().alias("sum4"),
        pl.col("maxabs").max().alias("maxabs"),
    )
    result = grouped.filter(pl.col("count") >= min_tokens).with_columns(
        (pl.col("sum2") / pl.col("count").clip(lower_bound=1)).sqrt().alias("rms"),
        (pl.col("sumabs") / pl.col("count").clip(lower_bound=1)).alias("mean_abs"),
        (
            (pl.col("sum4") / pl.col("count").clip(lower_bound=1))
            / (pl.col("sum2") / pl.col("count").clip(lower_bound=1)).pow(2).clip(lower_bound=1e-20)
        ).alias("kurtosis"),
    ).with_columns(
        pl.min_horizontal(
            pl.col("maxabs"),
            pl.col("rms") * max_tail_ratio,
        ).alias("maxabs"),
        pl.col("kurtosis").clip(upper_bound=128.0).alias("kurtosis"),
    ).with_columns(
        (pl.col("maxabs") / pl.col("rms").clip(lower_bound=1e-10))
        .clip(upper_bound=max_tail_ratio)
        .alias("tail_ratio")
    ).collect(engine="streaming")
    audit_columns(result, OBSERVER_COLUMNS, "observer")
    return result


def aggregate_acceptance(
    store: Path,
    run_id: str | None,
    min_observations: int = 128,
) -> pl.DataFrame | None:
    frame = scan(store, "acceptance_position", run_id)
    if frame is None:
        return None
    result = frame.group_by(["draft_type", "position"]).agg(
        pl.len().alias("observations"),
        pl.col("reached").sum().alias("reached"),
        pl.col("accepted").sum().alias("accepted"),
        pl.col("confidence").sum().alias("confidence_sum"),
    ).filter(pl.col("observations") >= min_observations).collect(engine="streaming")
    audit_columns(result, ACCEPTANCE_COLUMNS, "acceptance")
    return result


def aggregate_router(
    store: Path,
    run_id: str | None,
    min_observations: int = 100_000,
    min_expert_selections: int = 128,
) -> pl.DataFrame | None:
    frame = scan(store, "router", run_id)
    if frame is None:
        return None
    result = frame.group_by(["layer", "expert"]).agg(
        pl.col("observations").sum().alias("observations"),
        pl.col("selected").sum().alias("selected"),
        pl.col("probability_sum").sum().alias("probability_sum"),
        pl.col("confidence_sum").sum().alias("confidence_sum"),
        pl.col("margin_sum").sum().alias("margin_sum"),
        pl.col("output_error_sum").sum().alias("output_error_sum"),
        pl.col("downstream_divergence_sum").sum().alias("downstream_divergence_sum"),
    ).filter(
        (pl.col("observations") >= min_observations)
        & (pl.col("selected") >= min_expert_selections)
    ).with_columns(
        (pl.col("selected") / pl.col("observations").clip(lower_bound=1))
        .alias("frequency"),
        (pl.col("confidence_sum") / pl.col("selected").clip(lower_bound=1))
        .alias("mean_confidence"),
        (pl.col("margin_sum") / pl.col("selected").clip(lower_bound=1))
        .alias("mean_margin"),
        (pl.col("output_error_sum") / pl.col("selected").clip(lower_bound=1))
        .alias("mean_output_error"),
        (pl.col("downstream_divergence_sum") / pl.col("selected").clip(lower_bound=1))
        .alias("mean_downstream_divergence"),
    ).collect(engine="streaming")
    audit_columns(result, ROUTER_COLUMNS, "router")
    return result


def require_identity(api: HfApi) -> str:
    try:
        info = api.whoami()
    except Exception as error:
        raise RuntimeError("Hugging Face authentication is required; run `hf auth login`") from error
    name = info.get("name") if isinstance(info, dict) else None
    if not name:
        raise RuntimeError("Hugging Face authentication is required; run `hf auth login`")
    return name


def contribution_license() -> tuple[str, str]:
    if not CONTRIBUTION_LICENSE_PATH.is_file():
        raise RuntimeError(
            f"missing contribution terms at {CONTRIBUTION_LICENSE_PATH}"
        )
    text = CONTRIBUTION_LICENSE_PATH.read_text(encoding="utf-8")
    return text, hashlib.sha256(text.encode()).hexdigest()


def contribution_license_record(accepted_license: str, contributor: str) -> dict:
    if accepted_license != CONTRIBUTION_LICENSE_ID:
        raise ValueError(
            "publishing requires explicit acceptance: "
            f"--accept-contribution-license {CONTRIBUTION_LICENSE_ID}"
        )
    _, digest = contribution_license()
    return {
        "license_id": CONTRIBUTION_LICENSE_ID,
        "license_grantee": CONTRIBUTION_LICENSE_GRANTEE,
        "license_path": CONTRIBUTION_LICENSE_REPO_PATH,
        "license_sha256": digest,
        "assent": "explicit-cli",
        "contributor_identity_provider": "huggingface",
        "contributor_identity": contributor,
        "royalty_obligation": "none",
        "commercial_relicensing": True,
    }


def public_license_record() -> dict:
    if not PUBLIC_LICENSE_PATH.is_file():
        raise RuntimeError(f"missing public license notice at {PUBLIC_LICENSE_PATH}")
    text = PUBLIC_LICENSE_PATH.read_text(encoding="utf-8")
    return {
        "license_id": PUBLIC_LICENSE_ID,
        "license_uri": PUBLIC_LICENSE_URI,
        "attribution": PUBLIC_LICENSE_ATTRIBUTION,
        "notice_path": PUBLIC_LICENSE_REPO_PATH,
        "notice_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "commercial_use": False,
        "share_alike": True,
        "legacy_grants_preserved": True,
    }


def aggregate_coverage(
    observer: pl.DataFrame | None,
    acceptance: pl.DataFrame | None,
) -> tuple[int, int]:
    observer_tokens = int(observer["count"].min()) if observer is not None and not observer.is_empty() else 0
    acceptance_observations = (
        int(acceptance["observations"].min())
        if acceptance is not None and not acceptance.is_empty()
        else 0
    )
    return observer_tokens, acceptance_observations


def epoch_state(
    fingerprint: str,
    manifests: list[dict],
    model_epoch: int = 0,
    observer_tokens_per_epoch: int = 1_000_000,
) -> dict:
    if observer_tokens_per_epoch <= 0:
        raise ValueError("observer_tokens_per_epoch must be positive")
    unique = {
        manifest["aggregate_id"]: manifest
        for manifest in manifests
        if manifest.get("model_fingerprint") == fingerprint and manifest.get("aggregate_id")
    }
    observer_components = {}
    acceptance_components = {}
    for aggregate_id, item in unique.items():
        observer_key = item.get("observer_digest") or aggregate_id
        acceptance_key = item.get("acceptance_digest") or aggregate_id
        observer_components[observer_key] = int(item.get("observer_calibration_tokens", 0))
        acceptance_components[acceptance_key] = int(item.get("acceptance_observations", 0))
    observer_tokens = sum(observer_components.values())
    acceptance_observations = sum(acceptance_components.values())
    epoch = observer_tokens // observer_tokens_per_epoch
    evidence_digest = hashlib.sha256("\n".join(sorted(unique)).encode()).hexdigest()
    return {
        "schema": EPOCH_SCHEMA,
        "model_fingerprint": fingerprint,
        "epoch": epoch,
        "model_epoch": model_epoch,
        "requantization_due": epoch > model_epoch,
        "observer_calibration_tokens": observer_tokens,
        "acceptance_observations": acceptance_observations,
        "observer_tokens_per_epoch": observer_tokens_per_epoch,
        "aggregate_count": len(unique),
        "evidence_digest": evidence_digest,
    }


def remote_epoch_inputs(api: HfApi, repo: str, fingerprint: str) -> tuple[list[dict], dict | None]:
    prefix = f"data/{fingerprint}/"
    files = api.list_repo_files(repo_id=repo, repo_type="dataset")
    manifests = []
    for path in files:
        if path.startswith(f"{prefix}aggregates/") and path.endswith("/manifest.json"):
            local = hf_hub_download(repo_id=repo, repo_type="dataset", filename=path)
            manifest = json.loads(Path(local).read_text(encoding="utf-8"))
            manifest.setdefault("aggregate_id", Path(path).parent.name)
            manifests.append(manifest)
    receipt_path = f"{prefix}model-epoch.json"
    receipt = None
    if receipt_path in files:
        local = hf_hub_download(repo_id=repo, repo_type="dataset", filename=receipt_path)
        receipt = json.loads(Path(local).read_text(encoding="utf-8"))
    return manifests, receipt


def publish_epoch_state(
    api: HfApi,
    repo: str,
    fingerprint: str,
    observer_tokens_per_epoch: int,
    notify: bool,
) -> dict:
    manifests, receipt = remote_epoch_inputs(api, repo, fingerprint)
    model_epoch = int(receipt.get("epoch", 0)) if receipt else 0
    state = epoch_state(fingerprint, manifests, model_epoch, observer_tokens_per_epoch)
    api.upload_file(
        repo_id=repo,
        repo_type="dataset",
        path_or_fileobj=(json.dumps(state, indent=2) + "\n").encode(),
        path_in_repo=f"data/{fingerprint}/epoch.json",
        commit_message=f"Update Tessera epoch {state['epoch']} for {fingerprint}",
    )
    if state["requantization_due"]:
        message = (
            f"Tessera epoch {state['epoch']} is ready; the published GGUF is from "
            f"epoch {state['model_epoch']}. Re-run Tessera quantization."
        )
        print(message)
        if notify and platform.system() == "Darwin":
            subprocess.run(
                [
                    "osascript",
                    "-e",
                    f'display notification {json.dumps(message)} '
                    'with title "Tessera requantization due"',
                ],
                check=False,
            )
    return state


def init_repo(args) -> None:
    api = HfApi()
    require_identity(api)
    url = api.create_repo(
        repo_id=args.repo,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )
    card = f"""---
pretty_name: Tessera Calibration Commons
license: other
task_categories:
- text-generation
tags:
- llama.cpp
- quantization
- calibration
- tessera
---

# Tessera Calibration Commons

This dataset contains architecture-fingerprinted aggregate calibration
statistics produced by the Tessera tooling in llama.cpp. It excludes prompts,
completions, request logs, raw activations, model weights, and credentials.
Each contribution is independently attributable and stored as immutable
Parquet sufficient statistics so clients can merge compatible observations.
Contributions are accepted under the versioned Tessera Calibration Contribution
Terms, which grant Julian Alejandro Torres Nieto a perpetual, irrevocable,
royalty-free right to redistribute and commercially relicense contributions.

New Tessera-published aggregates are made publicly available under CC BY-NC-SA
4.0 with attribution to Julian Alejandro Torres Nieto, Tribunus.dev. The
machine-readable manifest beside each aggregate records its outbound license.
Material previously published under Apache-2.0 retains that license; the new
policy does not revoke or retroactively restrict any prior grant.
"""
    api.upload_file(
        repo_id=args.repo,
        repo_type="dataset",
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        commit_message="Initialize Tessera Calibration Commons",
    )
    terms, _ = contribution_license()
    api.upload_file(
        repo_id=args.repo,
        repo_type="dataset",
        path_or_fileobj=terms.encode(),
        path_in_repo=CONTRIBUTION_LICENSE_REPO_PATH,
        commit_message=f"Add {CONTRIBUTION_LICENSE_ID}",
    )
    public_notice = PUBLIC_LICENSE_PATH.read_text(encoding="utf-8")
    api.upload_file(
        repo_id=args.repo,
        repo_type="dataset",
        path_or_fileobj=public_notice.encode(),
        path_in_repo=PUBLIC_LICENSE_REPO_PATH,
        commit_message=f"Add {PUBLIC_LICENSE_ID} outbound notice",
    )
    print(url)


def publish(args) -> None:
    api = HfApi()
    contributor = require_identity(api)
    license_record = contribution_license_record(
        args.accept_contribution_license, contributor
    )
    fingerprint, identity = model_identity(Path(args.model_dir))
    observer = aggregate_observer(
        Path(args.store), args.run_id, args.min_tokens, args.max_tail_ratio
    )
    acceptance = aggregate_acceptance(
        Path(args.store), args.run_id, args.min_observations
    )
    router = aggregate_router(
        Path(args.store),
        args.run_id,
        args.min_router_observations,
        args.min_expert_selections,
    )
    if (
        (observer is None or observer.is_empty())
        and (acceptance is None or acceptance.is_empty())
        and (router is None or router.is_empty())
    ):
        raise ValueError(
            f"{args.store}: no aggregate passed the minimum-population privacy gates"
        )
    with tempfile.TemporaryDirectory(prefix="tessera-hf-") as directory:
        root = Path(directory)
        if observer is not None and not observer.is_empty():
            observer.write_parquet(root / "observer.parquet", compression="zstd", statistics=True)
        if acceptance is not None and not acceptance.is_empty():
            acceptance.write_parquet(root / "acceptance_position.parquet", compression="zstd", statistics=True)
        if router is not None and not router.is_empty():
            router.write_parquet(
                root / "router.parquet", compression="zstd", statistics=True
            )
        content_hash = hashlib.sha256()
        for artifact in sorted(root.glob("*.parquet")):
            content_hash.update(artifact.name.encode())
            content_hash.update(artifact.read_bytes())
        run_key = content_hash.hexdigest()[:24]
        observer_tokens, acceptance_observations = aggregate_coverage(observer, acceptance)
        observer_digest = (
            hashlib.sha256((root / "observer.parquet").read_bytes()).hexdigest()
            if (root / "observer.parquet").exists()
            else None
        )
        acceptance_digest = (
            hashlib.sha256((root / "acceptance_position.parquet").read_bytes()).hexdigest()
            if (root / "acceptance_position.parquet").exists()
            else None
        )
        router_digest = (
            hashlib.sha256((root / "router.parquet").read_bytes()).hexdigest()
            if (root / "router.parquet").exists()
            else None
        )
        manifest = {
            "schema": COMMONS_SCHEMA,
            "aggregate_id": run_key,
            "model_fingerprint": fingerprint,
            "model_identity": identity,
            "observer_rows": observer.height if observer is not None and not observer.is_empty() else 0,
            "acceptance_rows": acceptance.height if acceptance is not None and not acceptance.is_empty() else 0,
            "router_rows": router.height if router is not None and not router.is_empty() else 0,
            "observer_calibration_tokens": observer_tokens,
            "acceptance_observations": acceptance_observations,
            "observer_digest": observer_digest,
            "acceptance_digest": acceptance_digest,
            "router_digest": router_digest,
            "minimum_observations": args.min_observations,
            "minimum_tokens_per_channel": args.min_tokens,
            "minimum_router_observations": args.min_router_observations,
            "minimum_expert_selections": args.min_expert_selections,
            "maximum_public_tail_ratio": args.max_tail_ratio,
            "privacy": "aggregate-only; no prompts, generations, request logs, or raw activations",
            "contribution_license": license_record,
            "public_license": public_license_record(),
        }
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        api.upload_folder(
            repo_id=args.repo,
            repo_type="dataset",
            folder_path=root,
            path_in_repo=f"data/{fingerprint}/aggregates/{run_key}",
            commit_message=f"Add privacy-gated Tessera aggregate {fingerprint}",
        )
    print(f"published {args.repo}: data/{fingerprint}/aggregates/{run_key}")
    publish_epoch_state(
        api,
        args.repo,
        fingerprint,
        args.observer_tokens_per_epoch,
        args.notify,
    )


def status(args) -> None:
    api = HfApi()
    fingerprint, _ = model_identity(Path(args.model_dir))
    state = publish_epoch_state(
        api,
        args.repo,
        fingerprint,
        args.observer_tokens_per_epoch,
        args.notify,
    )
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        temporary.replace(output)
    print(json.dumps(state, indent=2))


def mark_model(args) -> None:
    api = HfApi()
    require_identity(api)
    fingerprint, _ = model_identity(Path(args.model_dir))
    manifests, _ = remote_epoch_inputs(api, args.repo, fingerprint)
    state = epoch_state(
        fingerprint,
        manifests,
        observer_tokens_per_epoch=args.observer_tokens_per_epoch,
    )
    receipt = {
        "schema": MODEL_EPOCH_SCHEMA,
        "model_fingerprint": fingerprint,
        "epoch": state["epoch"],
        "evidence_digest": state["evidence_digest"],
        "gguf": Path(args.gguf).name,
    }
    encoded = (json.dumps(receipt, indent=2) + "\n").encode()
    api.upload_file(
        repo_id=args.repo,
        repo_type="dataset",
        path_or_fileobj=encoded,
        path_in_repo=f"data/{fingerprint}/model-epoch.json",
        commit_message=f"Record GGUF built from Tessera epoch {state['epoch']}",
    )
    if args.model_repo:
        api.upload_file(
            repo_id=args.model_repo,
            repo_type="model",
            path_or_fileobj=encoded,
            path_in_repo="tessera-epoch.json",
            commit_message=f"Record Tessera epoch {state['epoch']}",
        )
    print(json.dumps(receipt, indent=2))


def pull(args) -> None:
    fingerprint, _ = model_identity(Path(args.model_dir))
    snapshot = Path(snapshot_download(
        repo_id=args.repo,
        repo_type="dataset",
        revision=args.revision,
        allow_patterns=[f"data/{fingerprint}/**"],
    ))
    sources = list((snapshot / "data" / fingerprint).glob("*/*/observer.parquet"))
    destination = Path(args.store) / "observer"
    destination.mkdir(parents=True, exist_ok=True)
    installed = 0
    for source in sources:
        digest = hashlib.sha256(str(source.relative_to(snapshot)).encode()).hexdigest()[:20]
        target = destination / f"part-hf-{digest}.parquet"
        if target.exists():
            continue
        frame = pl.read_parquet(source).with_columns(
            pl.lit(SCHEMA).alias("schema"),
            pl.lit(f"hf:{args.repo}@{args.revision}").alias("run_id"),
            pl.lit(str(source.relative_to(snapshot))).alias("source"),
        )
        temporary = target.with_suffix(".tmp.parquet")
        frame.write_parquet(temporary, compression="zstd", statistics=True)
        temporary.replace(target)
        installed += frame.height
    print(f"bootstrapped {installed} observer rows for model fingerprint {fingerprint}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hugging Face Tessera evidence commons")
    subparsers = parser.add_subparsers(dest="command", required=True)
    initialize = subparsers.add_parser("init")
    initialize.add_argument("--repo", default=DEFAULT_REPO)
    initialize.add_argument("--private", action="store_true")
    initialize.set_defaults(func=init_repo)
    publish_parser = subparsers.add_parser("publish")
    publish_parser.add_argument("--repo", default=DEFAULT_REPO)
    publish_parser.add_argument("--model-dir", required=True)
    publish_parser.add_argument("--store", required=True)
    publish_parser.add_argument("--run-id", default=None)
    publish_parser.add_argument("--min-observations", type=int, default=128)
    publish_parser.add_argument("--min-tokens", type=int, default=100_000)
    publish_parser.add_argument("--max-tail-ratio", type=float, default=32.0)
    publish_parser.add_argument("--min-router-observations", type=int, default=100_000)
    publish_parser.add_argument("--min-expert-selections", type=int, default=128)
    publish_parser.add_argument("--observer-tokens-per-epoch", type=int, default=1_000_000)
    publish_parser.add_argument("--notify", action=argparse.BooleanOptionalAction, default=True)
    publish_parser.add_argument(
        "--accept-contribution-license",
        required=True,
        metavar=CONTRIBUTION_LICENSE_ID,
        help=(
            "Explicitly accept the versioned Tessera calibration contribution "
            "terms and royalty-free commercial grant"
        ),
    )
    publish_parser.set_defaults(func=publish)
    pull_parser = subparsers.add_parser("pull")
    pull_parser.add_argument("--repo", default=DEFAULT_REPO)
    pull_parser.add_argument("--model-dir", required=True)
    pull_parser.add_argument("--store", required=True)
    pull_parser.add_argument("--revision", default="main")
    pull_parser.set_defaults(func=pull)
    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--repo", default=DEFAULT_REPO)
    status_parser.add_argument("--model-dir", required=True)
    status_parser.add_argument("--observer-tokens-per-epoch", type=int, default=1_000_000)
    status_parser.add_argument("--notify", action=argparse.BooleanOptionalAction, default=True)
    status_parser.add_argument("--output", default=None, help="Freeze the current epoch receipt to JSON")
    status_parser.set_defaults(func=status)
    mark_parser = subparsers.add_parser("mark-model")
    mark_parser.add_argument("--repo", default=DEFAULT_REPO)
    mark_parser.add_argument("--model-dir", required=True)
    mark_parser.add_argument("--gguf", required=True)
    mark_parser.add_argument("--model-repo", default=None)
    mark_parser.add_argument("--observer-tokens-per-epoch", type=int, default=1_000_000)
    mark_parser.set_defaults(func=mark_model)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
