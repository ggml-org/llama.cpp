#!/usr/bin/env python3
"""
REFERENCE ONLY - build a small test "slice" of a HF safetensors model without downloading the full weights.

The heuristics below (layer/expert detection, config patching) cover common layouts only. New models often use tensor names, shapes or config keys that this script does not know about. Always run --inspect first, check the plan, and adapt the script to the model.

How it works: each safetensors shard starts with an 8-byte header length + JSON header, which gives the exact byte span of every tensor. We fetch only the headers, then use HTTP range requests to fetch the tensors we keep:
  1. keep only some layers (per numeric "family" in tensor names whose count matches a layer count in config.json), renumbered to 0..n-1
  2. keep only the first E experts (per-expert tensors are dropped, stacked expert tensors and router weights are row-sliced along dim 0)
  3. optionally keep only the first V rows of vocab-sized tensors (off by default, see --vocab-size)
config.json is patched to match. Other small repo files are copied as-is.

Usage:
    python slice_model.py Qwen/Qwen3-0.6B --inspect
    python slice_model.py Qwen/Qwen3-30B-A3B --num-layers 2 --num-experts 8 --dry-run
    python slice_model.py Qwen/Qwen3-30B-A3B --keep-layers 0,3 --output ../models/qwen3moe-slice
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import fnmatch
import json
import logging
import os
import re
import struct
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests
from huggingface_hub import HfApi, get_token, hf_hub_url

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("slice_model")

INDEX_FILE = "model.safetensors.index.json"
CONFIG_FILE = "config.json"

# files not worth copying into a slice
AUX_BLACKLIST = [
    # weights in other formats (the slice replaces them)
    "*.bin", "*.pth", "*.pt", "*.ckpt", "*.h5", "*.msgpack",
    "*.onnx", "*.tflite", "*.mlmodel", "*.mlpackage/*", "*.gguf", "*.nemo",
    "onnx/*", "openvino/*", "coreml/*", "original/*", "metal/*",
    # media / demo assets
    "*.png", "*.jpg", "*.jpeg", "*.gif", "*.webp", "*.avif", "*.svg",
    "*.mp4", "*.webm", "*.mov", "*.wav", "*.mp3", "*.flac", "*.ogg", "*.pdf",
    ".gitattributes",
]
AUX_MAX_SIZE = 50 * 1024 * 1024

LAYER_CONTAINERS = {"layers", "layer", "blocks", "block", "h"}
EXPERT_ROW_KEYWORDS = ("expert", "router", "gate", "score")

LAYER_KEYS = {
    "num_hidden_layers", "num_layers", "n_layer", "n_layers", "num_layer",
    "num_decoder_layers", "encoder_layers", "decoder_layers", "depth",
    "num_encoder_layers", "num_attention_layers", "layer_count",
}
VOCAB_KEYS = {"vocab_size", "padded_vocab_size"}
NEXTN_KEYS = {"num_nextn_predict_layers", "num_mtp_layers"}
TOPK_SUBSTR = ("per_tok", "topk", "top_k", "moe_k")
EXPERT_KEY_EXCLUDE = ("per_tok", "topk", "top_k", "shared", "group", "intermediate", "size", "dim", "dtype")


def parse_size(text: str) -> int:
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([kKmMgGtT]?)[bB]?", text.strip())
    if not m:
        raise argparse.ArgumentTypeError(f"invalid size: {text!r}")
    mult = {"": 1, "k": 1024, "m": 1024**2, "g": 1024**3, "t": 1024**4}[m.group(2).lower()]
    return int(float(m.group(1)) * mult)


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{int(n)}B"
        n /= 1024
    return f"{n}GB"


# --------------------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------------------

_tls = threading.local()


def _session() -> requests.Session:
    s = getattr(_tls, "s", None)
    if s is None:
        s = requests.Session()
        _tls.s = s
    return s


class HubClient:
    def __init__(self, repo_id: str, revision: str, token: str | None):
        self.repo_id = repo_id
        self.revision = revision
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}

    def url(self, filename: str) -> str:
        return hf_hub_url(self.repo_id, filename, revision=self.revision)

    def _request(self, filename: str, headers: dict, stream: bool = False, retries: int = 5):
        last: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                r = _session().get(self.url(filename), headers=headers, stream=stream, timeout=90, allow_redirects=True)
                if r.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"HTTP {r.status_code}", response=r)
                r.raise_for_status()
                return r
            except Exception as e:  # noqa: BLE001
                last = e
                if attempt == retries or "404" in str(e) or "401" in str(e) or "403" in str(e):
                    break
                time.sleep(2.0 * attempt)
        raise RuntimeError(f"{self.repo_id}/{filename}: {last}") from last

    def get_range(self, filename: str, start: int, end_inclusive: int) -> bytes:
        h = dict(self.headers)
        h["Range"] = f"bytes={start}-{end_inclusive}"
        return self._request(filename, h).content

    def get_json(self, filename: str) -> Any:
        return json.loads(self._request(filename, dict(self.headers)).content)

    def download_file(self, filename: str, dest: Path):
        r = self._request(filename, dict(self.headers), stream=True)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                if chunk:
                    f.write(chunk)

    def stream_range_into_fd(self, filename: str, start: int, end_inclusive: int, fd: int, write_offset: int):
        h = dict(self.headers)
        h["Range"] = f"bytes={start}-{end_inclusive}"
        r = self._request(filename, h, stream=True)
        pos = write_offset
        for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
            if chunk:
                os.pwrite(fd, chunk, pos)
                pos += len(chunk)
        if pos - write_offset != end_inclusive - start + 1:
            raise RuntimeError(f"short read: {filename} bytes={start}-{end_inclusive}")


def fetch_shard_header(client: HubClient, filename: str) -> tuple[dict, int]:
    (header_len,) = struct.unpack("<Q", client.get_range(filename, 0, 7))
    header = json.loads(client.get_range(filename, 8, 8 + header_len - 1))
    header.pop("__metadata__", None)
    return header, 8 + header_len


# --------------------------------------------------------------------------------------
# config.json analysis / patching
# --------------------------------------------------------------------------------------

def walk_config(cfg: Any):
    """Yield (container_dict, key, value) for every leaf in nested dicts."""
    if isinstance(cfg, dict):
        for k, v in cfg.items():
            if isinstance(v, dict):
                yield from walk_config(v)
            else:
                yield cfg, k, v


@dataclass
class ConfigInfo:
    cfg: dict | None
    layer_counts: set[int] = field(default_factory=set)
    nextn_counts: set[int] = field(default_factory=set)
    expert_counts: set[int] = field(default_factory=set)
    vocab_sizes: set[int] = field(default_factory=set)
    first_k_dense: int = 0


def analyze_config(cfg: dict | None) -> ConfigInfo:
    info = ConfigInfo(cfg=cfg)
    if cfg is None:
        return info
    for _d, key, val in walk_config(cfg):
        if not isinstance(val, int) or isinstance(val, bool) or val <= 0:
            continue
        if key in LAYER_KEYS:
            info.layer_counts.add(val)
        elif key in NEXTN_KEYS:
            info.nextn_counts.add(val)
        elif key in VOCAB_KEYS:
            info.vocab_sizes.add(val)
        elif key == "first_k_dense_replace":
            info.first_k_dense = val
        elif "expert" in key and val > 1 and not any(x in key for x in EXPERT_KEY_EXCLUDE):
            info.expert_counts.add(val)
    return info


def patch_config(cfg: dict, families: dict[str, Family], new_vocab: int | None, orig_vocabs: set[int]) -> list[str]:
    """Patch counts in-place to match the slice. Returns log lines."""
    log = []
    by_card: dict[int, Family] = {}
    for fam in families.values():
        if fam.kept is not None:
            by_card.setdefault(fam.card, fam)
            by_card.setdefault(fam.regular_total, fam)
    max_new_experts = max((f.new_count for f in families.values() if f.kind == "expert" and f.kept is not None), default=None)

    for d, key, val in list(walk_config(cfg)):
        if isinstance(val, int) and not isinstance(val, bool):
            fam = by_card.get(val)
            if key in LAYER_KEYS and fam is not None and fam.kind == "layer":
                d[key] = fam.new_regular
            elif key == "first_k_dense_replace":
                lfam = next((f for f in families.values() if f.kind == "layer"), None)
                if lfam is None:
                    continue
                d[key] = sum(1 for i in lfam.kept if i < val)
            elif "expert" in key and fam is not None and fam.kind == "expert" and not any(x in key for x in EXPERT_KEY_EXCLUDE):
                d[key] = fam.new_count
            elif "expert" in key and any(x in key for x in TOPK_SUBSTR) and max_new_experts is not None and val > max_new_experts:
                d[key] = max_new_experts
            elif key in VOCAB_KEYS and new_vocab is not None and val in orig_vocabs:
                d[key] = new_vocab
            if d[key] != val:
                log.append(f"{key}: {val} -> {d[key]}")
        elif isinstance(val, list) and val and all(not isinstance(x, (dict, list)) for x in val):
            # per-layer lists, e.g. layer_types
            fam = by_card.get(len(val))
            if fam is not None and fam.kind == "layer" and len(val) in (fam.card, fam.regular_total):
                d[key] = [val[i] for i in fam.kept if i < len(val)]
                log.append(f"{key}: list[{len(val)}] -> list[{len(d[key])}]")
    return log


# --------------------------------------------------------------------------------------
# tensor name analysis
# --------------------------------------------------------------------------------------

@dataclass
class Family:
    """One numeric slot in tensor names, e.g. 'model.layers.#'."""
    prefix: str
    kind: str            # "layer" | "expert" | "other"
    indices: set[int] = field(default_factory=set)
    kept: list[int] | None = None    # None = keep all, no renumbering
    renumber: dict[int, int] = field(default_factory=dict)
    regular_total: int = 0           # layer count excluding MTP tail
    new_regular: int = 0

    @property
    def card(self) -> int:
        return len(self.indices)

    @property
    def new_count(self) -> int:
        return len(self.kept) if self.kept is not None else self.card


def name_slots(name: str):
    """Yield (slot_pos, family_prefix, index) for each numeric path component."""
    tokens = name.split(".")
    norm: list[str] = []
    for i, tok in enumerate(tokens):
        if tok.isdigit():
            yield i, ".".join(norm), int(tok)
            norm.append("#")
        else:
            norm.append(tok)


def build_families(names: list[str], cfg_info: ConfigInfo) -> dict[str, Family]:
    families: dict[str, Family] = {}
    for name in names:
        tokens = name.split(".")
        for i, prefix, idx in name_slots(name):
            container = tokens[i - 1] if i > 0 else ""
            fam = families.get(prefix)
            if fam is None:
                fam = Family(prefix=prefix, kind="expert" if "expert" in container else "other")
                families[prefix] = fam
            fam.indices.add(idx)

    acceptable_layer_cards: dict[int, int] = {}  # card -> regular layer count
    for c in cfg_info.layer_counts:
        acceptable_layer_cards[c] = c
        for m in cfg_info.nextn_counts:
            acceptable_layer_cards[c + m] = c

    for fam in families.values():
        if fam.kind == "expert":
            continue
        container = fam.prefix.rsplit(".", 1)[-1]
        if cfg_info.cfg is not None:
            if fam.card in acceptable_layer_cards:
                fam.kind = "layer"
                fam.regular_total = acceptable_layer_cards[fam.card]
        elif fam.card >= 4 and container in LAYER_CONTAINERS:
            fam.kind = "layer"
            fam.regular_total = fam.card
    return families


def plan_families(families: dict[str, Family], num_layers: int, keep_layers: list[int] | None, num_experts: int, first_k_dense: int):
    for fam in families.values():
        if fam.kind == "layer":
            if keep_layers is not None:
                kept = [i for i in keep_layers if i < fam.regular_total]
            else:
                base = min(num_layers, fam.regular_total)
                kept = list(range(base))
                # keep one MoE layer for models whose first k layers are dense
                if 0 < first_k_dense < fam.regular_total and first_k_dense >= base >= 1:
                    kept = list(range(base - 1)) + [first_k_dense]
            fam.new_regular = len(kept)
            # keep the MTP tail (layer indices beyond the regular count)
            kept += [i for i in sorted(fam.indices) if i >= fam.regular_total]
            fam.kept = sorted(set(kept))
        elif fam.kind == "expert":
            fam.kept = sorted(fam.indices)[:num_experts]
            fam.new_regular = len(fam.kept)
        else:
            fam.kept = None
            continue
        fam.renumber = {old: new for new, old in enumerate(fam.kept)}


def slice_tensor_name(name: str, families: dict[str, Family]) -> str | None:
    """Return the renumbered output name, or None if the tensor is dropped."""
    tokens = name.split(".")
    for i, prefix, idx in name_slots(name):
        fam = families[prefix]
        if fam.kept is None:
            continue
        if idx not in fam.renumber:
            return None
        tokens[i] = str(fam.renumber[idx])
    return ".".join(tokens)


# --------------------------------------------------------------------------------------
# slicing
# --------------------------------------------------------------------------------------

@dataclass
class TensorPlan:
    src_name: str
    out_name: str
    dtype: str
    shape: list[int]
    orig_shape: list[int]
    size: int
    src_file: str
    seg_start: int       # absolute byte range in src_file
    seg_end: int         # inclusive


def pick_weight_files(filenames: list[str]) -> tuple[list[str], bool]:
    """Return (weight files, has_index). Preference: index -> model.safetensors -> any root safetensors."""
    if INDEX_FILE in filenames:
        return [], True
    if "model.safetensors" in filenames:
        return ["model.safetensors"], False
    root_st = [f for f in filenames if f.endswith(".safetensors") and "/" not in f]
    if "consolidated.safetensors" in root_st:
        return ["consolidated.safetensors"], False
    return root_st[:1], False


def row_slice(shape: list[int], size: int, keep_rows: int) -> tuple[list[int], int] | None:
    # only a prefix of dim 0 is a single contiguous byte range
    if len(shape) < 1 or shape[0] <= keep_rows or size % shape[0] != 0:
        return None
    row_bytes = size // shape[0]
    return [keep_rows] + shape[1:], keep_rows * row_bytes


def build_plan(args, families, headers, weight_files, src_file_of, cfg_info) -> list[TensorPlan]:
    plan_families(families, args.num_layers, args.keep_layers, args.num_experts, cfg_info.first_k_dense)
    plans = []
    for fname in weight_files:
        header, data_start = headers[fname]
        for name, meta in header.items():
            out_name = slice_tensor_name(name, families)
            if out_name is None:
                continue
            start, end = meta["data_offsets"]
            shape, size = list(meta["shape"]), end - start
            new_rows = None
            if args.vocab_size and shape and shape[0] > args.vocab_size \
                    and any(shape[0] >= vs and shape[0] - vs <= 1024 for vs in cfg_info.vocab_sizes):
                new_rows = args.vocab_size
            elif shape and shape[0] in cfg_info.expert_counts and shape[0] > args.num_experts \
                    and any(k in name for k in EXPERT_ROW_KEYWORDS):
                new_rows = args.num_experts
            if new_rows is not None:
                sliced = row_slice(shape, size, new_rows)
                if sliced is not None:
                    shape, size = sliced
            plans.append(TensorPlan(name, out_name, meta["dtype"], shape, list(meta["shape"]), size, src_file_of[name],
                                    data_start + start, data_start + start + size - 1))
    return plans


def print_inspect(cfg_info: ConfigInfo, families: dict[str, Family], plans: list[TensorPlan], all_names: list[str]):
    logger.info("=== config analysis ===")
    logger.info(f"layer counts: {sorted(cfg_info.layer_counts)}  nextn: {sorted(cfg_info.nextn_counts)}  "
                f"experts: {sorted(cfg_info.expert_counts)}  vocab: {sorted(cfg_info.vocab_sizes)}  first_k_dense: {cfg_info.first_k_dense}")
    logger.info("\n=== numeric families in tensor names ===")
    for fam in sorted(families.values(), key=lambda f: f.prefix):
        kept = "all" if fam.kept is None else f"{fam.kept[:16]}{'...' if len(fam.kept) > 16 else ''}"
        logger.info(f"{fam.prefix:60s} kind={fam.kind:6s} count={fam.card:4d} kept={kept}")
    logger.info("\n=== kept tensors (first index of each family only) ===")
    seen = set()
    for t in plans:
        key = re.sub(r"\.\d+\.", ".#.", t.out_name)
        if key in seen and t.shape == t.orig_shape:
            continue
        seen.add(key)
        mark = f"  (sliced from {t.orig_shape})" if t.shape != t.orig_shape else ""
        logger.info(f"{t.out_name:70s} {t.dtype:5s} {t.shape}{mark}")
    kept_src = {t.src_name for t in plans}
    dropped = [n for n in all_names if n not in kept_src]
    logger.info(f"\n{len(plans)}/{len(all_names)} tensors kept, {len(dropped)} dropped")


def write_slice(out_dir: Path, client: HubClient, plans: list[TensorPlan], jobs: int):
    header_out: dict[str, Any] = {}
    offset = 0
    for t in plans:
        header_out[t.out_name] = {"dtype": t.dtype, "shape": t.shape, "data_offsets": [offset, offset + t.size]}
        offset += t.size
    header_out["__metadata__"] = {"format": "pt"}
    hbytes = json.dumps(header_out, separators=(",", ":")).encode()
    hbytes += b" " * ((-(len(hbytes) + 8)) % 8)
    data_start = 8 + len(hbytes)

    shard_path = out_dir / "model.safetensors"
    with open(shard_path, "wb") as f:
        f.write(struct.pack("<Q", len(hbytes)))
        f.write(hbytes)
        f.truncate(data_start + offset)
    fd = os.open(shard_path, os.O_WRONLY)
    try:
        def fetch(t: TensorPlan):
            dst = data_start + header_out[t.out_name]["data_offsets"][0]
            client.stream_range_into_fd(t.src_file, t.seg_start, t.seg_end, fd, dst)

        with cf.ThreadPoolExecutor(max_workers=jobs) as ex:
            list(ex.map(fetch, plans))
    finally:
        os.close(fd)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repo", help="HF repo id, e.g. Qwen/Qwen3-0.6B")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--output", type=Path, default=None, help="output dir (default: ./tmp/hf_slices/{user}--{model})")
    ap.add_argument("--inspect", action="store_true", help="print config analysis, tensor families and the plan, write nothing")
    ap.add_argument("--dry-run", action="store_true", help="plan and estimate size only, write nothing")
    ap.add_argument("--num-layers", type=int, default=2, help="keep the first N layers (default: 2)")
    ap.add_argument("--keep-layers", default=None, help="explicit comma-separated layer indices to keep, overrides --num-layers")
    ap.add_argument("--num-experts", type=int, default=8)
    ap.add_argument("--vocab-size", type=int, default=0,
                    help="row-slice vocab-sized tensors to this size (default: 0 = off; tokenizer is NOT sliced, so this breaks conversion for most tokenizers)")
    ap.add_argument("--max-size", type=parse_size, default="1G", help="abort if the sliced weights exceed this (default: 1G)")
    ap.add_argument("--hf-token", default=None, help="HF token (default: HF_TOKEN env var or 'hf auth login' token)")
    ap.add_argument("--jobs", type=int, default=8, help="concurrent HTTP requests")
    args = ap.parse_args()
    args.keep_layers = sorted({int(x) for x in args.keep_layers.split(",")}) if args.keep_layers else None

    token = args.hf_token or get_token()
    api = HfApi(token=token)
    info = api.model_info(args.repo, revision=args.revision, files_metadata=True)
    client = HubClient(args.repo, info.sha or "main", token)
    siblings = {s.rfilename: (s.size or 0) for s in info.siblings or []}
    filenames = sorted(siblings)

    weight_files, has_index = pick_weight_files(filenames)
    weight_map: dict[str, str] = {}
    if has_index:
        weight_map = client.get_json(INDEX_FILE)["weight_map"]
        weight_files = sorted(set(weight_map.values()))
    elif not weight_files:
        logger.error("no .safetensors found")
        return 1

    cfg = client.get_json(CONFIG_FILE) if CONFIG_FILE in filenames else None
    cfg_info = analyze_config(cfg)

    with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        headers = dict(zip(weight_files, ex.map(lambda f: fetch_shard_header(client, f), weight_files)))
    all_names = [n for fname in weight_files for n in headers[fname][0]]
    src_file_of = weight_map if has_index else {n: weight_files[0] for n in all_names}

    families = build_families(all_names, cfg_info)
    plans = build_plan(args, families, headers, weight_files, src_file_of, cfg_info)
    est = sum(t.size for t in plans)

    if args.inspect:
        print_inspect(cfg_info, families, plans, all_names)
    logger.info(f"{args.repo}@{info.sha}: {len(plans)}/{len(all_names)} tensors, {human(est)}")
    if not any(f.kind == "layer" for f in families.values()):
        logger.warning("no layer family detected - check --inspect output and adapt the script")
    if est > args.max_size:
        logger.error(f"slice is {human(est)} > --max-size {human(args.max_size)}")
        return 1
    if args.inspect or args.dry_run:
        return 0

    out_dir = args.output or Path("tmp") / "hf_slices" / args.repo.replace("/", "--")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_slice(out_dir, client, plans, args.jobs)

    skipped_aux = []
    for fname in filenames:
        if fname in weight_files or fname in (INDEX_FILE, CONFIG_FILE) or fname.endswith(".safetensors"):
            continue
        if any(fnmatch.fnmatch(fname, p) or fnmatch.fnmatch(Path(fname).name, p) for p in AUX_BLACKLIST):
            continue
        if siblings[fname] > AUX_MAX_SIZE:
            skipped_aux.append(f"{fname} ({human(siblings[fname])})")
            continue
        client.download_file(fname, out_dir / fname)
    for f in skipped_aux:
        logger.warning(f"skipped large aux file {f}")

    patch_log = []
    if cfg is not None:
        new_vocab = args.vocab_size if args.vocab_size and any(vs > args.vocab_size for vs in cfg_info.vocab_sizes) else None
        patch_log = patch_config(cfg, families, new_vocab, cfg_info.vocab_sizes)
        with open(out_dir / CONFIG_FILE, "w") as f:
            json.dump(cfg, f, indent=2)
            f.write("\n")
    for line in patch_log:
        logger.info(f"config: {line}")

    with open(out_dir / ".slice_meta.json", "w") as f:
        json.dump({
            "repo": args.repo, "revision": info.sha,
            "tensors": len(plans), "total_tensors": len(all_names), "size": est,
            "config_patches": patch_log, "skipped_aux": skipped_aux,
        }, f, indent=2)
    logger.info(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
