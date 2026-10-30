#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gguf-py"))
from gguf import GGUFReader


def expert_layout(model):
    reader = GGUFReader(model)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    layers = sorted({int(match.group(1)) for name in tensors
                     if (match := re.fullmatch(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight", name))})
    if not layers or layers != list(range(len(layers))):
        raise ValueError("requires contiguous expert layers starting at zero in the first GGUF shard")
    layout = []
    offset = 0
    experts = None
    for layer in layers:
        names = [f"blk.{layer}.ffn_{role}_exps.weight" for role in ("gate", "up", "down")]
        if any(name not in tensors for name in names):
            raise ValueError(f"layer {layer}: requires separate gate, up and down tensors in this shard")
        roles = [tensors[name] for name in names]
        if any(len(tensor.shape) != 3 for tensor in roles):
            raise ValueError(f"layer {layer}: expert tensors must have three dimensions")
        count = int(roles[0].shape[2])
        if count < 1 or any(int(tensor.shape[2]) != count for tensor in roles):
            raise ValueError(f"layer {layer}: inconsistent expert counts")
        if experts is not None and count != experts:
            raise ValueError("all layers must use the same expert count")
        experts = count
        if roles[0].tensor_type != roles[1].tensor_type or tuple(roles[0].shape) != tuple(roles[1].shape):
            raise ValueError(f"layer {layer}: gate and up must have the same shape and type")
        if tuple(roles[2].shape[:2]) != tuple(reversed(roles[0].shape[:2])):
            raise ValueError(f"layer {layer}: incompatible down projection shape")
        if any(tensor.n_bytes % count for tensor in roles):
            raise ValueError(f"layer {layer}: expert byte size is not integral")
        sizes = [tensor.n_bytes // count for tensor in roles]
        blob = sum(sizes)
        if blob <= 0 or blob > 256 * 1024 * 1024:
            raise ValueError(f"layer {layer}: unsupported expert byte size {blob}")
        layout.append({"layer": layer, "offset": offset, "blob": blob, "sizes": sizes,
                       "source_offsets": [tensor.data_offset for tensor in roles],
                       "types": [int(tensor.tensor_type) for tensor in roles]})
        offset += count * blob
    return layout, experts, offset


def index_text(layout):
    header = "# native experts v3: layer gu_type d_type offset blob_bytes gate_off up_off down_off\n"
    return header + "".join(" ".join(str(value) for value in (
        row["layer"], row["types"][0], row["types"][2], row["offset"], row["blob"], *row["source_offsets"]
    )) + "\n" for row in layout)


def read_expert(source, row, expert):
    parts = []
    for offset, size in zip(row["source_offsets"], row["sizes"]):
        source.seek(offset + expert * size)
        data = source.read(size)
        if len(data) != size:
            raise OSError(f"short source read: layer {row['layer']}, expert {expert}")
        parts.append(data)
    return b"".join(parts)


def create_pack(model, output):
    layout, experts, total = expert_layout(model)
    padded = (total + 4095) // 4096 * 4096
    output.mkdir(parents=True, exist_ok=True)
    names = ("experts.bin", "expert-pack.json", "native_experts.txt")
    if any((output / name).exists() for name in names):
        raise FileExistsError("output pack already exists; use --verify or choose an empty directory")
    if shutil.disk_usage(output).free < padded + 1024 * 1024:
        raise OSError(f"not enough free space: need at least {padded / 2**30:.2f} GiB")
    with tempfile.TemporaryDirectory(prefix=".moe-pack-", dir=output) as temporary:
        temporary = Path(temporary)
        digest = hashlib.sha256()
        with model.open("rb") as source, (temporary / "experts.bin").open("xb") as target:
            for row in layout:
                for expert in range(experts):
                    data = read_expert(source, row, expert)
                    target.write(data)
                    digest.update(data)
                print(f"packed layer {row['layer'] + 1}/{len(layout)}", flush=True)
            padding = bytes(padded - total)
            target.write(padding)
            digest.update(padding)
            target.flush()
            os.fsync(target.fileno())
        (temporary / "native_experts.txt").write_text(index_text(layout), encoding="ascii")
        metadata = {"format": "native-experts-v3", "source": model.name, "source_bytes": model.stat().st_size,
                    "layers": len(layout), "experts_per_layer": experts, "bytes": padded,
                    "sha256": digest.hexdigest(), "weights": "unaltered source bytes"}
        (temporary / "expert-pack.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        published = []
        try:
            for name in names:
                destination = output / name
                os.link(temporary / name, destination)
                published.append(destination)
        except OSError:
            for destination in published:
                destination.unlink()
            raise
    return metadata


def verify_pack(model, output, verify_all=False):
    layout, experts, total = expert_layout(model)
    if (output / "experts.bin").stat().st_size != (total + 4095) // 4096 * 4096:
        raise ValueError("pack byte size does not match the model")
    actual = [line.split()[:8] for line in (output / "native_experts.txt").read_text().splitlines()
              if line.strip() and not line.lstrip().startswith("#")]
    expected = [line.split() for line in index_text(layout).splitlines()[1:]]
    if actual != expected:
        raise ValueError("pack index does not match the model")
    selected = range(experts) if verify_all else sorted({0, experts // 2, experts - 1})
    checked = 0
    with model.open("rb") as source, (output / "experts.bin").open("rb") as packed:
        for row in layout:
            for expert in selected:
                packed.seek(row["offset"] + expert * row["blob"])
                if packed.read(row["blob"]) != read_expert(source, row, expert):
                    raise ValueError(f"weight mismatch: layer {row['layer']}, expert {expert}")
                checked += 1
    return {"verified_experts": checked, "total_experts": len(layout) * experts, "all": verify_all}


def main():
    parser = argparse.ArgumentParser(description="Pack separate routed experts without changing quantized bytes.")
    parser.add_argument("--model", type=Path, required=True, help="first GGUF shard, containing all routed experts")
    parser.add_argument("--output", type=Path, required=True, help="pack directory; existing packs are never overwritten")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="inspect layout and disk requirements without writing")
    mode.add_argument("--verify", action="store_true", help="compare three experts per layer against the GGUF")
    mode.add_argument("--verify-all", action="store_true", help="compare every expert against the GGUF")
    args = parser.parse_args()
    try:
        model = args.model.resolve(strict=True)
        if args.dry_run:
            layout, experts, total = expert_layout(model)
            result = {"layers": len(layout), "experts_per_layer": experts, "bytes": (total + 4095) // 4096 * 4096}
        elif args.verify or args.verify_all:
            result = verify_pack(model, args.output, args.verify_all)
        else:
            result = create_pack(model, args.output)
        print(json.dumps(result, indent=2))
    except (OSError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
