#!/usr/bin/env python3
"""Build and publish reproducible BF16 source epochs for Tessera."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file


MANIFEST_SCHEMA = "llama.tessera.source-manifest.v1"
RECEIPT_SCHEMA = "llama.tessera.source-epoch.v1"
DEFAULT_REPO = "juliantorr/tessera-unified-bf16"
TESSERA_PUBLIC_LICENSE = {
    "license": "CC-BY-NC-SA-4.0",
    "license_uri": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
    "attribution": "Julian Alejandro Torres Nieto, Tribunus.dev",
    "commercial_use": False,
    "scope": "Tessera-controlled material only; upstream terms remain applicable",
}
TESSERA_NOTICE = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "TESSERA_ARTIFACT_LICENSE_NOTICE.md"
)
ASSET_NAMES = {
    "config.json",
    "configuration.json",
    "chat_template.jinja",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "LICENSE",
    "LICENSE.txt",
    "NOTICE",
    "NOTICE.txt",
    "README.md",
}


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def tensor_files(component_dir: Path) -> list[Path]:
    return sorted(path for path in component_dir.rglob("*.safetensors") if path.is_file())


def component_inventory(component: dict) -> dict:
    root = Path(component["path"]).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"{root}: component path is not a directory")
    files = tensor_files(root)
    if not files:
        raise ValueError(f"{root}: no safetensors files")
    tensors = 0
    logical_bytes = 0
    file_records = []
    seen = set()
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as reader:
            names = list(reader.keys())
            overlap = seen.intersection(names)
            if overlap:
                raise ValueError(f"{root}: duplicate tensor names: {sorted(overlap)[:3]}")
            seen.update(names)
            tensors += len(names)
            for name in names:
                view = reader.get_slice(name)
                shape = view.get_shape()
                item_size = {
                    "BOOL": 1, "U8": 1, "I8": 1,
                    "U16": 2, "I16": 2, "F16": 2, "BF16": 2,
                    "U32": 4, "I32": 4, "F32": 4,
                    "U64": 8, "I64": 8, "F64": 8,
                }.get(view.get_dtype())
                if item_size is None:
                    raise ValueError(f"{path}:{name}: unsupported dtype {view.get_dtype()}")
                elements = 1
                for extent in shape:
                    elements *= extent
                logical_bytes += elements * item_size
        file_records.append({
            "path": str(path.relative_to(root)),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        })
    return {
        "name": component["name"],
        "namespace": component["namespace"],
        "upstream_repo": component["upstream_repo"],
        "upstream_revision": component["upstream_revision"],
        "license": component["license"],
        "redistribution": bool(component.get("redistribution", False)),
        "files": file_records,
        "tensor_count": tensors,
        "logical_bytes": logical_bytes,
    }


def seal_manifest(manifest: dict) -> dict:
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("unsupported source manifest schema")
    public = {
        "schema": MANIFEST_SCHEMA,
        "epoch": int(manifest["epoch"]),
        "lineage": {
            "parent_source_digest": manifest.get("lineage", {}).get("parent_source_digest"),
            "training_corpus_epoch": manifest.get("lineage", {}).get("training_corpus_epoch"),
            "training_corpus_digest": manifest.get("lineage", {}).get("training_corpus_digest"),
            "telemetry_epoch": manifest.get("lineage", {}).get("telemetry_epoch"),
        },
        "components": [],
        "tessera_distribution_license": TESSERA_PUBLIC_LICENSE,
    }
    namespaces = set()
    for component in manifest["components"]:
        namespace = component["namespace"].strip(".")
        if not namespace or namespace in namespaces:
            raise ValueError(f"duplicate or empty namespace: {namespace!r}")
        namespaces.add(namespace)
        item = dict(component_inventory(component))
        item["namespace"] = namespace
        public["components"].append(item)
    public["components"].sort(key=lambda item: item["namespace"])
    public["source_digest"] = hashlib.sha256(canonical_json(public)).hexdigest()
    return public


def copy_assets(component: dict, destination: Path) -> None:
    root = Path(component["path"]).expanduser().resolve()
    target = destination / "components" / component["namespace"].replace(".", "/")
    for source in sorted(root.rglob("*")):
        if not source.is_file() or source.name not in ASSET_NAMES:
            continue
        output = target / source.relative_to(root)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, output)


def assemble(manifest: dict, output: Path, max_shard_bytes: int) -> dict:
    if max_shard_bytes <= 0:
        raise ValueError("max_shard_bytes must be positive")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"{output}: output directory is not empty")
    sealed = seal_manifest(manifest)
    output.mkdir(parents=True, exist_ok=True)
    (output / "source-manifest.json").write_text(
        json.dumps(sealed, indent=2) + "\n", encoding="utf-8"
    )
    licenses = {
        component["namespace"]: {
            "name": component["name"],
            "license": component["license"],
            "redistribution": component["redistribution"],
            "upstream_repo": component["upstream_repo"],
            "upstream_revision": component["upstream_revision"],
        }
        for component in sealed["components"]
    }
    (output / "LICENSES.json").write_text(
        json.dumps(
            {
                "tessera_distribution": TESSERA_PUBLIC_LICENSE,
                "upstream_components": licenses,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    if not TESSERA_NOTICE.is_file():
        raise RuntimeError(f"missing Tessera artifact notice at {TESSERA_NOTICE}")
    shutil.copyfile(TESSERA_NOTICE, output / "TESSERA_ARTIFACT_LICENSE_NOTICE.md")
    card_license = "other"
    (output / "README.md").write_text(
        "---\n"
        f"license: {card_license}\n"
        "library_name: transformers\n"
        "---\n\n"
        f"# Tessera Unified BF16 Source Epoch {sealed['epoch']}\n\n"
        "This is a reproducible, namespaced BF16 source bundle for Tessera. "
        "Component licenses, immutable upstream revisions, and file hashes are "
        "recorded in `LICENSES.json` and `source-manifest.json`. Modified "
        "Tessera epochs are identified by `tessera-source-epoch.json`. "
        "Tessera-controlled material is distributed under CC BY-NC-SA 4.0 "
        "with attribution to Julian Alejandro Torres Nieto, Tribunus.dev. "
        "Every upstream component remains subject to its recorded license; "
        "see `TESSERA_ARTIFACT_LICENSE_NOTICE.md`.\n",
        encoding="utf-8",
    )
    staged: list[dict[str, Any]] = []
    staged_bytes = 0
    shards: list[tuple[Path, list[str]]] = []
    weight_map: dict[str, str] = {}

    def flush() -> None:
        nonlocal staged, staged_bytes
        if not staged:
            return
        shard_number = len(shards) + 1
        temporary = output / f".model-{shard_number:05d}.safetensors"
        tensors = {item["name"]: item["tensor"] for item in staged}
        save_file(tensors, temporary, metadata={"format": "pt"})
        shards.append((temporary, list(tensors)))
        staged = []
        staged_bytes = 0

    for component in sorted(manifest["components"], key=lambda item: item["namespace"]):
        copy_assets(component, output)
        namespace = component["namespace"].strip(".")
        root = Path(component["path"]).expanduser().resolve()
        for source in tensor_files(root):
            with safe_open(source, framework="pt", device="cpu") as reader:
                for source_name in reader.keys():
                    tensor = reader.get_tensor(source_name)
                    name = f"{namespace}.{source_name}"
                    size = tensor.numel() * tensor.element_size()
                    if staged and staged_bytes + size > max_shard_bytes:
                        flush()
                    staged.append({"name": name, "tensor": tensor})
                    staged_bytes += size
                    if staged_bytes >= max_shard_bytes:
                        flush()
    flush()

    total = len(shards)
    for index, (temporary, names) in enumerate(shards, start=1):
        filename = f"model-{index:05d}-of-{total:05d}.safetensors"
        final = output / filename
        temporary.replace(final)
        for name in names:
            weight_map[name] = filename
    index = {
        "metadata": {
            "total_size": sum(item["logical_bytes"] for item in sealed["components"]),
            "tessera_source_epoch": sealed["epoch"],
            "tessera_source_digest": sealed["source_digest"],
        },
        "weight_map": weight_map,
    }
    (output / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2) + "\n", encoding="utf-8"
    )
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "epoch": sealed["epoch"],
        "source_digest": sealed["source_digest"],
        "tensor_count": len(weight_map),
        "logical_bytes": index["metadata"]["total_size"],
        "lineage": sealed["lineage"],
        "shards": [
            {
                "path": path.name,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path, _ in [
                (output / f"model-{number:05d}-of-{total:05d}.safetensors", names)
                for number, (_, names) in enumerate(shards, start=1)
            ]
        ],
        "components": sealed["components"],
    }
    receipt["artifact_digest"] = hashlib.sha256(canonical_json(receipt)).hexdigest()
    (output / "tessera-source-epoch.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


def validate_bundle(bundle: Path) -> dict:
    receipt_path = bundle / "tessera-source-epoch.json"
    index_path = bundle / "model.safetensors.index.json"
    if not receipt_path.is_file() or not index_path.is_file():
        raise ValueError(f"{bundle}: incomplete Tessera source epoch")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise ValueError(f"{receipt_path}: unsupported schema")
    expected_artifact = receipt.pop("artifact_digest", None)
    actual_artifact = hashlib.sha256(canonical_json(receipt)).hexdigest()
    receipt["artifact_digest"] = expected_artifact
    if expected_artifact != actual_artifact:
        raise ValueError(f"{receipt_path}: artifact digest mismatch")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    names = set()
    expected_shards = {shard["path"] for shard in receipt["shards"]}
    actual_shards = {path.name for path in bundle.glob("model-*-of-*.safetensors")}
    if actual_shards != expected_shards:
        raise ValueError("bundle contains missing or unrecorded safetensor shards")
    for shard in receipt["shards"]:
        path = bundle / shard["path"]
        if not path.is_file() or sha256_file(path) != shard["sha256"]:
            raise ValueError(f"{path}: missing or hash mismatch")
        with safe_open(path, framework="pt", device="cpu") as reader:
            for name in reader.keys():
                if name in names:
                    raise ValueError(f"duplicate output tensor: {name}")
                names.add(name)
    if names != set(index["weight_map"]):
        raise ValueError("safetensors index does not match shard contents")
    return receipt


def require_publishable(receipt: dict) -> None:
    blocked = [
        component["name"]
        for component in receipt["components"]
        if not component.get("redistribution", False)
    ]
    if blocked:
        raise ValueError(
            "public BF16 publication is blocked by unresolved redistribution status: "
            + ", ".join(blocked)
        )


def load_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fetch_manifest(manifest: dict, cache: Path) -> dict:
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("unsupported source manifest schema")
    resolved = {
        "schema": MANIFEST_SCHEMA,
        "epoch": int(manifest["epoch"]),
        "lineage": manifest.get("lineage", {}),
        "components": [],
    }
    cache.mkdir(parents=True, exist_ok=True)
    for component in manifest["components"]:
        revision = component.get("upstream_revision")
        if not revision or revision == "main":
            raise ValueError(
                f"{component['name']}: upstream_revision must be an immutable commit"
            )
        destination = cache / component["namespace"].replace(".", "/")
        snapshot_download(
            repo_id=component["upstream_repo"],
            revision=revision,
            local_dir=destination,
            allow_patterns=[
                "*.safetensors",
                "*.safetensors.index.json",
                "*.json",
                "*.txt",
                "*.jinja",
                "README*",
                "LICENSE*",
                "NOTICE*",
                "speech_tokenizer/**",
            ],
        )
        item = dict(component)
        item["path"] = str(destination.resolve())
        resolved["components"].append(item)
    return resolved


def command_fetch(args) -> None:
    resolved = fetch_manifest(load_manifest(Path(args.manifest)), Path(args.cache))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(resolved, indent=2) + "\n", encoding="utf-8")
    print(output)


def command_seal(args) -> None:
    sealed = seal_manifest(load_manifest(Path(args.manifest)))
    Path(args.output).write_text(json.dumps(sealed, indent=2) + "\n", encoding="utf-8")
    print(sealed["source_digest"])


def command_assemble(args) -> None:
    receipt = assemble(
        load_manifest(Path(args.manifest)),
        Path(args.output),
        int(args.max_shard_size_gib * 1024**3),
    )
    print(json.dumps(receipt, indent=2))


def command_validate(args) -> None:
    print(json.dumps(validate_bundle(Path(args.bundle)), indent=2))


def command_publish(args) -> None:
    bundle = Path(args.bundle)
    receipt = validate_bundle(bundle)
    require_publishable(receipt)
    api = HfApi()
    api.create_repo(
        repo_id=args.repo,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )
    api.upload_folder(
        repo_id=args.repo,
        repo_type="model",
        folder_path=bundle,
        path_in_repo=f"epochs/{receipt['epoch']}",
        commit_message=f"Publish Tessera BF16 source epoch {receipt['epoch']}",
    )
    api.upload_file(
        repo_id=args.repo,
        repo_type="model",
        path_or_fileobj=(json.dumps(receipt, indent=2) + "\n").encode(),
        path_in_repo="tessera-source-epoch.json",
        commit_message=f"Set current Tessera BF16 source epoch to {receipt['epoch']}",
    )
    print(f"https://huggingface.co/{args.repo}/tree/main/epochs/{receipt['epoch']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tessera BF16 source epoch manager")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch")
    fetch.add_argument("--manifest", required=True)
    fetch.add_argument("--cache", required=True)
    fetch.add_argument("--output", required=True)
    fetch.set_defaults(func=command_fetch)
    seal = commands.add_parser("seal")
    seal.add_argument("--manifest", required=True)
    seal.add_argument("--output", required=True)
    seal.set_defaults(func=command_seal)
    build = commands.add_parser("assemble")
    build.add_argument("--manifest", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--max-shard-size-gib", type=float, default=4.0)
    build.set_defaults(func=command_assemble)
    validate = commands.add_parser("validate")
    validate.add_argument("--bundle", required=True)
    validate.set_defaults(func=command_validate)
    publish = commands.add_parser("publish")
    publish.add_argument("--bundle", required=True)
    publish.add_argument("--repo", default=DEFAULT_REPO)
    publish.add_argument("--private", action="store_true")
    publish.set_defaults(func=command_publish)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
