#!/usr/bin/env python3
"""
Build small test "slices" of the HF repos declared with @ModelBase.example().

For each example repo, a reduced copy of the model is constructed without ever
downloading the full weights, using HTTP range requests against the safetensors
shards (the 8-byte length + JSON header of each shard tells us the exact byte
span of every tensor):
  1. keep only the first N transformer layers (per numeric "family" in tensor
     names whose cardinality matches a layer count declared in config.json)
  2. keep only the first E experts (per-expert tensors are dropped, stacked
     expert tensors and router weights are row-sliced along dim 0)
  3. keep only the first V rows of vocab-sized tensors (embeddings, lm_head)
config.json is patched to match (layer/expert/vocab counts, per-layer lists are
index-sliced). All other repo files are copied as-is, except a blacklist of
files that are useless for conversion testing (alternate-format weights, media,
demo assets, ...).

Output layout: {output}/{user}--{model}/...

Usage:
    python scripts/test_convert.py --dry-run
    python scripts/test_convert.py --repos "Qwen/*" --max-size 100M
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
logger = logging.getLogger("test_convert")

REPO_ROOT = Path(__file__).resolve().parent.parent

INDEX_FILE = "model.safetensors.index.json"
CONFIG_FILE = "config.json"

# files not worth copying into a conversion-test slice
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

# per-file cap for aux files that are not on the blacklist (e.g. nested
# safetensors weights of a sub-model); bigger files are skipped with a warning
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


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}B"
        n /= 1024
    return f"{n}GB"


def collect_examples() -> dict[str, list[str]]:
    """Map example repo id -> sorted list of model class names using it."""
    sys.path.insert(0, str(REPO_ROOT))
    import conversion
    conversion.load_all_models()
    from conversion.base import ModelBase

    repos: dict[str, set[str]] = {}
    for classes in ModelBase._model_classes.values():
        for modelcls in classes.values():
            if "model_hf_examples" not in modelcls.__dict__:
                continue
            for repo in modelcls.model_hf_examples:
                repos.setdefault(repo, set()).add(modelcls.__name__)
    return {r: sorted(c) for r, c in sorted(repos.items())}


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
        expected = end_inclusive - start + 1
        if pos - write_offset != expected:
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


def patch_config(cfg: dict, families: dict[str, "Family"], new_vocab: int | None, orig_vocabs: set[int]) -> list[str]:
    """Patch counts in-place to match the slice. Returns log lines."""
    log = []
    # map original cardinality -> family (for int patching and list slicing)
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
                log.append(f"{key}: {val} -> {d[key]}")
            elif key == "first_k_dense_replace" and fam is not None and val >= fam.new_regular:
                d[key] = fam.new_regular - 1
                log.append(f"{key}: {val} -> {d[key]}")
            elif "expert" in key and fam is not None and fam.kind == "expert" and not any(x in key for x in EXPERT_KEY_EXCLUDE):
                d[key] = fam.new_count
                log.append(f"{key}: {val} -> {d[key]}")
            elif "expert" in key and any(x in key for x in TOPK_SUBSTR) and max_new_experts is not None and val > max_new_experts:
                d[key] = max_new_experts
                log.append(f"{key}: {val} -> {d[key]}")
            elif key in VOCAB_KEYS and new_vocab is not None and val in orig_vocabs:
                d[key] = new_vocab
                log.append(f"{key}: {val} -> {d[key]}")
        elif isinstance(val, list) and val and all(not isinstance(x, (dict, list)) for x in val):
            fam = by_card.get(len(val))
            if fam is not None and fam.kind == "layer" and len(val) == fam.card:
                d[key] = [val[i] for i in fam.kept]
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
        for i, prefix, idx in name_slots(name):
            tokens = name.split(".")
            container = tokens[i - 1] if i > 0 else ""
            fam = families.get(prefix)
            if fam is None:
                kind = "expert" if "expert" in container else "other"
                fam = Family(prefix=prefix, kind=kind)
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
        container = fam.prefix.rsplit(".", 1)[-1] if "." in fam.prefix else fam.prefix
        if cfg_info.cfg is not None:
            if fam.card in acceptable_layer_cards:
                fam.kind = "layer"
                fam.regular_total = acceptable_layer_cards[fam.card]
        elif fam.card >= 4 and container in LAYER_CONTAINERS:
            fam.kind = "layer"
            fam.regular_total = fam.card
    return families


def plan_families(families: dict[str, Family], num_layers: int, num_experts: int, first_k_dense: int):
    for fam in families.values():
        if fam.kind == "layer":
            base = min(num_layers, fam.regular_total)
            kept = list(range(base))
            # keep one MoE layer for models whose first k layers are dense
            if 0 < first_k_dense < fam.regular_total and first_k_dense >= base and base >= 1:
                kept = list(range(base - 1)) + [first_k_dense]
            # keep the MTP tail (layer indices beyond the regular count)
            kept += [i for i in sorted(fam.indices) if i >= fam.regular_total]
            fam.kept = sorted(kept)
            fam.new_regular = base
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
# per-repo slicing
# --------------------------------------------------------------------------------------

@dataclass
class TensorPlan:
    out_name: str
    dtype: str
    shape: list[int]
    size: int
    src_file: str
    seg_start: int       # absolute byte range in src_file
    seg_end: int         # inclusive


@dataclass
class RepoResult:
    repo: str
    status: str          # OK | DRY | TOO_BIG | NO_WEIGHTS | SKIPPED | ERROR
    detail: str = ""
    est_size: int = 0
    params: str = ""


def pick_weight_files(filenames: list[str]) -> tuple[list[str], bool]:
    """Return (weight files, has_index). Preference: index -> model.safetensors -> any root safetensors."""
    if INDEX_FILE in filenames:
        return [], True
    if "model.safetensors" in filenames:
        return ["model.safetensors"], False
    root_st = [f for f in filenames if f.endswith(".safetensors") and "/" not in f]
    if len(root_st) >= 1:
        # prefer consolidated.safetensors over arbitrary extra files
        if "consolidated.safetensors" in root_st:
            return ["consolidated.safetensors"], False
        return root_st[:1], False
    return [], False


def row_slice(shape: list[int], size: int, keep_rows: int) -> tuple[list[int], int] | None:
    if len(shape) < 1 or shape[0] <= keep_rows or size % shape[0] != 0:
        return None
    row_bytes = size // shape[0]
    return [keep_rows] + shape[1:], keep_rows * row_bytes


def slice_repo(repo: str, args, token: str | None, api: HfApi) -> RepoResult:
    out_dir = Path(args.output) / repo.replace("/", "--")
    meta_path = out_dir / ".slice_meta.json"
    if meta_path.exists() and not args.force and not args.dry_run:
        return RepoResult(repo, "SKIPPED", "already sliced (use --force to redo)")

    info = api.model_info(repo, files_metadata=True)
    client = HubClient(repo, info.sha or "main", token)
    siblings = {s.rfilename: (s.size or 0) for s in info.siblings}
    filenames = sorted(siblings)

    weight_files, has_index = pick_weight_files(filenames)
    weight_map: dict[str, str] = {}
    if has_index:
        index = client.get_json(INDEX_FILE)
        weight_map = index["weight_map"]
        weight_files = sorted(set(weight_map.values()))
    elif not weight_files:
        return RepoResult(repo, "NO_WEIGHTS", "no .safetensors found")

    cfg = client.get_json(CONFIG_FILE) if CONFIG_FILE in filenames else None
    cfg_info = analyze_config(cfg)

    # fetch headers (all shards; names alone are not enough for row slicing)
    headers: dict[str, tuple[dict, int]] = {}
    with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        for fname, hd in zip(weight_files, ex.map(lambda f: fetch_shard_header(client, f), weight_files)):
            headers[fname] = hd
    all_names = [n for fname in weight_files for n in headers[fname][0]]
    src_file_of = weight_map if has_index else {n: weight_files[0] for n in all_names}

    families = build_families(all_names, cfg_info)

    # shrink ladder: try requested params first, then reduce until under --max-size
    attempts = [(args.num_layers, args.num_experts, args.vocab_size)]
    v = args.vocab_size
    while v > 1024:
        v //= 2
        attempts.append((args.num_layers, args.num_experts, v))
    attempts += [(args.num_layers, max(2, args.num_experts // 2), v), (1, 2, v)]

    plans: list[TensorPlan] = []
    est = 0
    used = attempts[-1]
    for num_layers, num_experts, vocab_size in attempts:
        plan_families(families, num_layers, num_experts, cfg_info.first_k_dense)
        plans, est = [], 0
        for fname in weight_files:
            header, data_start = headers[fname]
            for name, meta in header.items():
                out_name = slice_tensor_name(name, families)
                if out_name is None:
                    continue
                start, end = meta["data_offsets"]
                shape, size = list(meta["shape"]), end - start
                new_rows = None
                if any(shape[0] >= vs and shape[0] - vs <= 1024 for vs in cfg_info.vocab_sizes if shape) and shape[0] > vocab_size:
                    new_rows = vocab_size
                elif shape and shape[0] in cfg_info.expert_counts and shape[0] > num_experts \
                        and any(k in name for k in EXPERT_ROW_KEYWORDS):
                    new_rows = num_experts
                if new_rows is not None:
                    sliced = row_slice(shape, size, new_rows)
                    if sliced is not None:
                        shape, size = sliced
                plans.append(TensorPlan(out_name, meta["dtype"], shape, size, src_file_of[name],
                                        data_start + start, data_start + start + size - 1))
                est += size
        used = (num_layers, num_experts, vocab_size)
        if est <= args.max_size:
            break

    params = f"layers={used[0]} experts={used[1]} vocab={used[2]}"
    if est > args.max_size:
        return RepoResult(repo, "TOO_BIG", f"best effort {human(est)} > {human(args.max_size)}", est, params)
    if args.dry_run:
        return RepoResult(repo, "DRY", f"{len(plans)}/{len(all_names)} tensors", est, params)

    out_dir.mkdir(parents=True, exist_ok=True)

    # write the single output shard
    out_shard = weight_files[0] if not has_index and len(weight_files) == 1 else "model.safetensors"
    header_out: dict[str, Any] = {}
    offset = 0
    for t in plans:
        header_out[t.out_name] = {"dtype": t.dtype, "shape": t.shape, "data_offsets": [offset, offset + t.size]}
        offset += t.size
    header_out["__metadata__"] = {"format": "pt"}
    hbytes = json.dumps(header_out, separators=(",", ":")).encode()
    hbytes += b" " * ((-(len(hbytes) + 8)) % 8)
    data_start = 8 + len(hbytes)

    shard_path = out_dir / out_shard
    with open(shard_path, "wb") as f:
        f.write(struct.pack("<Q", len(hbytes)))
        f.write(hbytes)
        f.truncate(data_start + est)
    fd = os.open(shard_path, os.O_WRONLY)
    try:
        offsets = {t.out_name: header_out[t.out_name]["data_offsets"][0] for t in plans}

        def fetch(t: TensorPlan):
            client.stream_range_into_fd(t.src_file, t.seg_start, t.seg_end, fd, data_start + offsets[t.out_name])

        with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
            list(ex.map(fetch, plans))
    finally:
        os.close(fd)

    # aux files
    skipped_aux = []
    for fname in filenames:
        if fname in weight_files or fname == INDEX_FILE or fname.endswith(".safetensors"):
            continue
        if any(fnmatch.fnmatch(fname, p) or fnmatch.fnmatch(Path(fname).name, p) for p in AUX_BLACKLIST):
            continue
        if fname == CONFIG_FILE:
            continue  # patched below
        if siblings[fname] > AUX_MAX_SIZE:
            skipped_aux.append(f"{fname} ({human(siblings[fname])})")
            continue
        client.download_file(fname, out_dir / fname)
    for f in skipped_aux:
        logger.warning(f"  {repo}: skipped large aux file {f}")

    patch_log = []
    if cfg is not None:
        new_vocab = used[2] if any(vs > used[2] for vs in cfg_info.vocab_sizes) else None
        patch_log = patch_config(cfg, families, new_vocab, cfg_info.vocab_sizes)
        with open(out_dir / CONFIG_FILE, "w") as f:
            json.dump(cfg, f, indent=2)
            f.write("\n")

    with open(meta_path, "w") as f:
        json.dump({
            "repo": repo, "revision": info.sha, "params": params,
            "tensors": len(plans), "total_tensors": len(all_names),
            "size": est, "config_patches": patch_log, "skipped_aux": skipped_aux,
        }, f, indent=2)

    return RepoResult(repo, "OK", f"{len(plans)}/{len(all_names)} tensors", est, params)


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=REPO_ROOT / "tmp" / "hf_slices")
    ap.add_argument("--max-size", type=parse_size, default="100M",
                    help="max total size of the sliced safetensors weight (default: 100M)")
    ap.add_argument("--hf-token", default=None,
                    help="HF token (default: HF_TOKEN env var or the token stored by 'hf auth login')")
    ap.add_argument("--dry-run", action="store_true", help="plan and estimate sizes only, write nothing")
    ap.add_argument("--repos", default=None,
                    help="comma-separated globs/substrings to select a subset of example repos")
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--num-experts", type=int, default=8)
    ap.add_argument("--vocab-size", type=int, default=4096)
    ap.add_argument("--jobs", type=int, default=8, help="concurrent HTTP requests per repo")
    ap.add_argument("--force", action="store_true", help="re-slice repos that already have an output dir")
    args = ap.parse_args()

    token = args.hf_token or get_token()
    if not token:
        logger.warning("no HF token found, you may hit rate limits (set HF_TOKEN or run 'hf auth login')")

    examples = collect_examples()
    if args.repos:
        pats = [p.strip() for p in args.repos.split(",") if p.strip()]
        examples = {r: c for r, c in examples.items()
                    if any(fnmatch.fnmatch(r.lower(), p.lower()) or p.lower() in r.lower() for p in pats)}
    logger.info(f"{len(examples)} example repo(s) to process\n")

    api = HfApi(token=token)
    results: list[RepoResult] = []
    for i, (repo, classes) in enumerate(examples.items(), 1):
        logger.info(f"[{i}/{len(examples)}] {repo} ({', '.join(classes)})")
        try:
            res = slice_repo(repo, args, token, api)
        except Exception as e:  # noqa: BLE001
            res = RepoResult(repo, "ERROR", str(e)[:200])
        results.append(res)
        extra = f" [{res.params}]" if res.params else ""
        logger.info(f"    {res.status}: {res.detail} {human(res.est_size) if res.est_size else ''}{extra}")

    logger.info("\n=== summary ===")
    counts: dict[str, int] = {}
    for res in results:
        counts[res.status] = counts.get(res.status, 0) + 1
    for status in ("OK", "DRY", "SKIPPED", "TOO_BIG", "NO_WEIGHTS", "ERROR"):
        if counts.get(status):
            logger.info(f"{status}: {counts[status]}")
    for res in results:
        if res.status in ("TOO_BIG", "NO_WEIGHTS", "ERROR"):
            logger.info(f"  {res.status} {res.repo}: {res.detail}")

    if not args.dry_run:
        args.output.mkdir(parents=True, exist_ok=True)
        with open(args.output / "report.json", "w") as f:
            json.dump([res.__dict__ for res in results], f, indent=2)

    return 1 if counts.get("ERROR") else 0


if __name__ == "__main__":
    sys.exit(main())
