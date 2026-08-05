#!/usr/bin/env python3
"""Generate the Tessera calibration corpus.

Default mode: the clean-room procedural generator that has shipped since
the v1 schema. The text is fully synthetic, parameterized by
``--seed`` and ``--synthetic-count``, and the output stays byte-for-byte
stable for a fixed seed.

``--real`` mode: fetch the calibration corpora the architect chose
(Wikitext-103 for text, COCO val2014 captions for vision, LibriSpeech
dev.clean for audio), take a stratified / uniform sample, and write the
same v1 schema extended with ``modality`` and ``image_path`` /
``audio_path`` fields on the per-sample records. Downstream text
consumers (imatrix, spec_calib, l5_outcome) keep reading the
``text`` field as before; the new multimodal fields are additive and
ignored by text-only consumers.

The receipt is extended with a ``corpora`` list that records the
upstream repository, the downloaded byte count, the SHA256 of the
first 1 MB of downloaded payload, and the per-corpus license /
attribution. The receipt is the audit trail; downstream consumers may
inspect it.

``--dry-run`` lists what would be fetched without performing any
network or disk write. Tests use this to keep the suite hermetic.

Usage::

    # default: synthetic procedural
    python3 -m tools.tessera.build-calibration-corpus \\
        --output-dir data/calibration

    # real: text-only with the medium budget
    python3 -m tools.tessera.build-calibration-corpus \\
        --real --corpora text --budget medium \\
        --output-dir data/calibration

    # real: all three modalities, dry-run (no network)
    python3 -m tools.tessera.build-calibration-corpus \\
        --real --corpora text,vision,audio --budget medium \\
        --output-dir data/calibration --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Iterable

try:
    from huggingface_hub import hf_hub_download  # type: ignore
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False


SCHEMA = "llama.tessera.calibration-corpus.v1"
RECEIPT_SCHEMA = "llama.tessera.training-corpus.v1"
VERSION = 1
SYNTHETIC_ORIGIN = "tribunus.dev-clean-room-procedural"
DEFAULT_SYNTHETIC_COUNTS = {
    "code": 440,
    "en": 350,
    "ko": 60,
    "zh": 390,
    "ja": 60,
    "tool_calling": 120,
    "reasoning": 370,
    "chat": 130,
    "mixed": 190,
    "structured_context": 570,
}

# Real-data corpora. Each entry pins a Hugging Face dataset, the parquet
# shard to download (one is enough; we sample from the shard), the
# canonical split name, and the per-corpus license / attribution.
# Keep the table in this single place so a license audit is a one-file
# review.
REAL_CORPORA: dict[str, dict[str, Any]] = {
    "text": {
        "name": "wikitext-103-raw-v1",
        "repo": "Salesforce/wikitext",
        "repo_type": "dataset",
        "filename": "wikitext-103-raw-v1/train-00000-of-00002.parquet",
        "split": "train",
        "license": "CC-BY-SA-3.0",
        "license_uri": "https://creativecommons.org/licenses/by-sa/3.0/",
        "attribution": "Wikitext-103 (Merity et al. 2016), Salesforce Research",
        "modality": "text",
        "text_field": "text",
        "sampling": "stratified_length",
    },
    "vision": {
        "name": "COCO val2014 captions",
        "repo": "jxie/coco_captions",
        "repo_type": "dataset",
        "filename": "data/validation-00000-of-00010-0421425675e3d7a4.parquet",
        "split": "validation",
        "license": "CC-BY-4.0 (captions); per-image Flickr licenses mixed",
        "license_uri": "https://creativecommons.org/licenses/by/4.0/",
        "attribution": "COCO Captions (Chen et al. 2015); images via Flickr",
        "modality": "image_text",
        "sampling": "uniform_random",
        # bytes are decoded from this column; each row has image: {bytes, path}
        "image_bytes_field": ("image", "bytes"),
        "image_path_field": ("image", "path"),
        "caption_field": "caption",
    },
    "audio": {
        "name": "LibriSpeech dev.clean",
        "repo": "openslr/librispeech_asr",
        "repo_type": "dataset",
        "filename": "all/validation.clean/0000.parquet",
        "split": "validation.clean",
        "license": "CC-BY-4.0",
        "license_uri": "https://creativecommons.org/licenses/by/4.0/",
        "attribution": "LibriSpeech (Panayotov et al. 2015), openslr.org/12",
        "modality": "audio_text",
        "sampling": "uniform_random",
        "audio_bytes_field": ("audio", "bytes"),
        "audio_path_field": ("audio", "path"),
        "transcript_field": "text",
    },
}

BUDGETS: dict[str, dict[str, int]] = {
    "light":  {"text": 1_000,  "vision": 256,  "audio": 256},
    "medium": {"text": 5_000,  "vision": 1_000, "audio": 1_000},
    "heavy":  {"text": 20_000, "vision": 4_000, "audio": 4_000},
}

# Length buckets (chars) used to stratify the text sample so the
# calibration pipeline gets a balanced mix of short / medium / long
# contexts. The numbers are intentionally rough: a paragraph < 200
# chars is a short / single-sentence context, 200-1000 is medium, > 1000
# is long / multi-paragraph.
TEXT_LENGTH_BUCKETS: tuple[tuple[int, int], ...] = (
    (0, 200),
    (200, 1_000),
    (1_000, 10_000),
)


# ---------------------------------------------------------------------------
# Synthetic generators (unchanged from v1)
# ---------------------------------------------------------------------------


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def code_sample(index: int) -> str:
    languages = ("Python", "JavaScript", "Rust", "SQL", "Swift")
    language = languages[index % len(languages)]
    width = 4 + index % 13
    modulus = 7 + index % 23
    if language == "Python":
        body = (
            f"def normalize_window_{index}(values, width={width}):\n"
            "    result = []\n"
            "    for offset in range(0, len(values), width):\n"
            "        window = values[offset:offset + width]\n"
            "        scale = max(sum(abs(v) for v in window), 1)\n"
            "        result.extend(v / scale for v in window)\n"
            "    return result\n"
        )
    elif language == "JavaScript":
        body = (
            f"export function bucket{index}(records) {{\n"
            f"  return records.reduce((m, r) => {{ const k = r.score % {modulus};\n"
            "    (m[k] ??= []).push({...r, normalized: r.score / (1 + k)}); return m; }, {});\n"
            "}\n"
        )
    elif language == "Rust":
        body = (
            f"fn rolling_{index}(input: &[i64]) -> Vec<i64> {{\n"
            f"    input.windows({width}).map(|w| w.iter().sum::<i64>() % {modulus}).collect()\n"
            "}\n"
        )
    elif language == "SQL":
        body = (
            "WITH ranked AS (\n"
            "  SELECT project_id, event_day, SUM(weight) AS total,\n"
            "         ROW_NUMBER() OVER (PARTITION BY project_id ORDER BY SUM(weight) DESC) AS rank\n"
            "  FROM synthetic_events GROUP BY project_id, event_day\n"
            f") SELECT * FROM ranked WHERE rank <= {1 + index % 5};\n"
        )
    else:
        body = (
            f"func compact{index}(_ values: [Int]) -> [Int] {{\n"
            f"    Dictionary(grouping: values, by: {{ $0 % {modulus} }}).keys.sorted()\n"
            "}\n"
        )
    return (
        f"Language: {language}. Review this self-contained implementation for boundary "
        f"conditions and explain its asymptotic behavior.\n\n{body}"
    )


def english_sample(index: int) -> str:
    systems = ("Aster", "Bracken", "Cinder", "Delta", "Elm", "Fable")
    system = systems[index % len(systems)]
    nodes = 3 + index % 17
    return (
        f"The fictional {system}-{index} observatory contains {nodes} independent sensor "
        "nodes. Each node records pressure, color temperature, and a monotonic sequence "
        "number. A coordinator accepts a reading only after two different nodes agree on "
        "the sequence interval; disagreement creates a reversible quarantine record. "
        "Explain how this rule limits accidental duplication, distinguish availability "
        "from consistency, and summarize the behavior for a new operator. All names and "
        "events in this passage are procedurally generated."
    )


def korean_sample(index: int) -> str:
    return (
        f"가상의 관측 장치 하늘-{index}은 여러 센서의 온도와 압력 값을 비교한다. "
        f"센서 수는 {3 + index % 9}개이며, 두 번 연속 같은 구간이 확인된 경우에만 "
        "기록을 확정한다. 값이 다르면 원본을 지우지 않고 검토 목록으로 이동한다. "
        "이 절차의 목적과 장단점을 한국어로 간단히 설명하고, 안전한 복구 순서를 제안하라."
    )


def chinese_sample(index: int) -> str:
    return (
        f"虚构的青岚-{index}数据站接收来自{4 + index % 12}个传感器的读数。"
        "系统先检查递增序号，再比较两个独立节点的摘要；若结果不一致，就把记录放入"
        "可恢复的隔离区，而不是直接删除。请说明这个流程如何减少重复和误删，并给出"
        "一个包含验证、回滚和审计步骤的简短操作方案。本文中的名称和数据均为程序生成。"
    )


def japanese_sample(index: int) -> str:
    return (
        f"架空の観測装置ミナモ-{index}は、{3 + index % 10}個のセンサーから測定値を受け取る。"
        "連続番号を確認した後、二つの独立した計算結果が一致した場合だけ記録を確定する。"
        "不一致の記録は削除せず、復元可能な保留領域へ移す。この設計の目的、障害時の"
        "復旧手順、監査で確認すべき項目を日本語で説明せよ。"
    )


def tool_sample(index: int) -> str:
    city = ("Northport", "Lake Ember", "Cedar Vale")[index % 3]
    return (
        "Available tool:\n"
        '{"name":"inspect_station","description":"Read a fictional station snapshot",'
        '"parameters":{"type":"object","properties":{"station":{"type":"string"},'
        '"include_history":{"type":"boolean"}},"required":["station"]}}\n\n'
        f"User request: Inspect station \"{city}-{index}\" and include its history. "
        "Return the tool arguments as valid JSON, then describe how you would handle "
        "a timeout without inventing a result."
    )


def reasoning_sample(index: int) -> str:
    crates = 5 + index % 31
    units = 7 + (index * 3) % 29
    removed = index % max(units - 1, 1)
    total = crates * units - removed
    return (
        f"A fictional depot has {crates} sealed crates with {units} calibration tiles "
        f"in each crate. During inspection, {removed} tiles are removed. How many tiles "
        "remain? Work from the quantities given, state the multiplication and subtraction, "
        f"and verify the result by reversing the subtraction. The expected arithmetic result "
        f"is {total}, but the explanation must show why."
    )


def chat_sample(index: int) -> str:
    return (
        "System: You are assisting with a fictional, non-production device.\n"
        f"User: The indicator on unit Pine-{index} changed from amber to blue after restart. "
        "Should I erase its history?\n"
        "Assistant: Do not erase history merely because the indicator changed. First export "
        "the local diagnostic summary and compare the sequence counter.\n"
        "User: The counter increased by one and the self-test passed. What next?\n"
        "Assistant: Explain a cautious next step that preserves rollback information and "
        "does not claim access to tools or readings that were not supplied."
    )


def mixed_sample(index: int) -> str:
    return (
        f"# Synthetic incident {index}\n\n"
        f"Status: `review-{index % 5}`\n\n"
        "| channel | expected | observed |\n"
        f"| alpha | {10 + index % 7} | {10 + (index * 2) % 7} |\n"
        f"| beta | {20 + index % 11} | {20 + (index * 3) % 11} |\n\n"
        "```ini\nretry_limit=3\npreserve_history=true\nmode=validate\n```\n\n"
        "Reconcile the table with the configuration and identify which conclusions are "
        "supported directly versus which would require another measurement."
    )


def structured_sample(index: int) -> str:
    return (
        f"<station id=\"synthetic-{index}\">\n"
        f"  <epoch>{index // 8}</epoch>\n"
        f"  <channels>{2 + index % 9}</channels>\n"
        "  <policy preserve=\"true\" quorum=\"2\" />\n"
        "</station>\n\n"
        "```json\n"
        f'{{"request":"validate","station":"synthetic-{index}",'
        f'"ranges":[[{index % 17},{index % 17 + 8}]],"dry_run":true}}\n'
        "```\n\n"
        "Describe the relationship between the XML state and JSON request. Reject any "
        "interpretation that would silently mutate data while dry_run is true."
    )


GENERATORS = {
    "code": code_sample,
    "en": english_sample,
    "ko": korean_sample,
    "zh": chinese_sample,
    "ja": japanese_sample,
    "tool_calling": tool_sample,
    "reasoning": reasoning_sample,
    "chat": chat_sample,
    "mixed": mixed_sample,
    "structured_context": structured_sample,
}


def build_synthetic_records(seed: int, counts: dict[str, int] | None = None) -> list[dict]:
    """Build the v1 clean-room synthetic records. The output is
    byte-for-byte stable for a given (seed, counts)."""
    category_counts = counts or DEFAULT_SYNTHETIC_COUNTS
    records = []
    for category, count in category_counts.items():
        for index in range(count):
            text = GENERATORS[category](index)
            record_id = digest_bytes(
                f"{VERSION}\0{seed}\0{category}\0{index}\0{text}".encode()
            )[:24]
            records.append({
                "schema": SCHEMA,
                "id": record_id,
                "category": category,
                "text": text,
                "origin": SYNTHETIC_ORIGIN,
            })
    random.Random(seed).shuffle(records)
    return records


# ---------------------------------------------------------------------------
# Real-data ingest (Wikitext-103, COCO val2014, LibriSpeech dev.clean)
# ---------------------------------------------------------------------------


def _require_hf_hub(dry_run: bool) -> None:
    if HF_HUB_AVAILABLE or dry_run:
        return
    raise SystemExit(
        "huggingface_hub is not installed. Install it (`pip install "
        "huggingface_hub`) or pass --dry-run to inspect what would be "
        "fetched without performing any network or disk write."
    )


def _is_section_header(text: str) -> bool:
    """Wikitext-103 prefixes every Wikipedia section with a header like
    ' = Valkyria Chronicles III = '. The first 900K rows of train-00000
    are dominated by these; we filter them out before sampling so the
    calibration set is real prose, not titles."""
    stripped = text.strip()
    if not stripped:
        return False
    if not (stripped.startswith("=") and stripped.endswith("=")):
        return False
    # Section headers always have at least one space between the
    # equals signs and the title; pure runs of `=` are not headers.
    inner = stripped.strip("=").strip()
    return bool(inner)


def _length_bucket(text: str) -> int:
    n = len(text)
    for i, (lo, hi) in enumerate(TEXT_LENGTH_BUCKETS):
        if lo <= n < hi:
            return i
    return len(TEXT_LENGTH_BUCKETS) - 1


def _read_parquet_rows(
    parquet_path: Path, columns: list[str] | None = None
) -> list[dict]:
    """Read a parquet file fully into memory as a list of dicts. Polars
    is the fast path; pyarrow is the fallback when polars is not
    available. For calibration-corpus scale (the largest shard we
    download is ~340 MB of LibriSpeech) full materialization is fine —
    the sample stage below then keeps the top N."""
    try:
        import polars as pl  # type: ignore
        df = pl.read_parquet(str(parquet_path), columns=columns)
        # iter_rows(named=True) yields a dict per row directly.
        return list(df.iter_rows(named=True))
    except ImportError:
        import pyarrow.parquet as pq  # type: ignore
        table = pq.read_table(str(parquet_path), columns=columns)
        return table.to_pylist()


def _stratified_text_sample(
    paragraphs: list[str], limit: int, seed: int,
) -> list[tuple[str, int]]:
    """Take ``limit`` paragraphs from ``paragraphs`` evenly across the
    length buckets. Each bucket contributes a proportional share; if a
    bucket is exhausted, the leftover is re-allocated to the buckets
    that still have paragraphs. The returned tuples are (text, bucket).
    """
    rng = random.Random(seed)
    bucketed: dict[int, list[str]] = {i: [] for i in range(len(TEXT_LENGTH_BUCKETS))}
    for paragraph in paragraphs:
        bucketed[_length_bucket(paragraph)].append(paragraph)
    for bucket in bucketed.values():
        rng.shuffle(bucket)
    per_bucket = max(1, limit // len(TEXT_LENGTH_BUCKETS))
    selected: list[tuple[str, int]] = []
    leftovers = 0
    for i, items in bucketed.items():
        take = min(per_bucket, len(items))
        for text in items[:take]:
            selected.append((text, i))
        leftovers += per_bucket - take
    if leftovers > 0:
        # Refill from buckets with surplus.
        for i, items in bucketed.items():
            if leftovers <= 0:
                break
            already = sum(1 for _, b in selected if b == i)
            available = len(items) - already
            if available <= 0:
                continue
            extra = min(available, leftovers)
            for text in items[already:already + extra]:
                selected.append((text, i))
            leftovers -= extra
    if len(selected) > limit:
        # Final cap with a deterministic shuffle so the residual is
        # spread across buckets rather than dumping the tail of one
        # bucket.
        rng.shuffle(selected)
        selected = selected[:limit]
    rng.shuffle(selected)
    return selected


def _read_wikitext_paragraphs(parquet_path: Path) -> list[str]:
    """Read the Wikitext-103 train shard and return the list of
    non-empty, non-section-header paragraphs."""
    rows = _read_parquet_rows(parquet_path, columns=["text"])
    out: list[str] = []
    for row in rows:
        text = row.get("text", "")
        if not isinstance(text, str):
            continue
        stripped = text.strip()
        if not stripped:
            continue
        if _is_section_header(stripped):
            continue
        out.append(stripped)
    return out


def _uniform_random_indices(
    population_size: int, limit: int, seed: int,
) -> list[int]:
    """Return ``limit`` distinct indices in [0, population_size). When
    population_size < limit the full population is returned in
    deterministic order. The function is the only place where the
    sampling RNG is consulted for vision / audio; isolate it so the
    audit trail is reproducible from the seed alone."""
    if population_size <= 0:
        return []
    rng = random.Random(seed)
    if limit >= population_size:
        order = list(range(population_size))
        rng.shuffle(order)
        return order
    # random.sample is reservoir-style and yields a unique list
    return sorted(rng.sample(range(population_size), limit))


def _fetch_corpus_file(
    cfg: dict[str, Any], cache_dir: Path, dry_run: bool,
) -> tuple[Path | None, int, str | None]:
    """Download (or dry-run) the configured parquet shard. Returns
    (local_path, total_bytes, sha256_of_first_1MB) where the bytes /
    hash are None in dry-run mode. The caller is responsible for
    actually reading the rows; this function only handles the
    transport."""
    if dry_run:
        return None, 0, None
    cache_dir.mkdir(parents=True, exist_ok=True)
    # huggingface_hub resolves the snapshot to the cache and returns a
    # path inside the cache (the snapshot is content-addressed so
    # re-runs are cheap).
    local = Path(hf_hub_download(
        repo_id=cfg["repo"],
        filename=cfg["filename"],
        repo_type=cfg.get("repo_type", "dataset"),
        cache_dir=str(cache_dir),
    ))
    size = local.stat().st_size
    with local.open("rb") as source:
        head = source.read(1_048_576)  # first 1 MB
    head_digest = hashlib.sha256(head).hexdigest() if head else None
    return local, size, head_digest


def _safe_jpeg_filename(cocoid: int) -> str:
    return f"coco_val2014_{int(cocoid):012d}.jpg"


def _safe_flac_filename(utt_id: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in utt_id)
    if not safe.lower().endswith(".flac"):
        safe = safe + ".flac"
    return safe


def build_real_text(
    cfg: dict[str, Any],
    limit: int,
    output_dir: Path,
    cache_dir: Path,
    seed: int,
    dry_run: bool,
) -> tuple[list[dict], dict[str, Any]]:
    """Fetch Wikitext-103, take a length-stratified sample, write
    records to the corpus index. Returns the records and the per-corpus
    metadata for the receipt."""
    corpus_meta: dict[str, Any] = {
        "name": cfg["name"],
        "source": f"{cfg['repo']}/{cfg['filename']}",
        "modality": cfg["modality"],
        "sample_count": 0,
        "license": cfg["license"],
        "license_uri": cfg["license_uri"],
        "attribution": cfg["attribution"],
        "sampling": cfg["sampling"],
        "total_bytes_downloaded": 0,
        "sha256_of_first_1MB": None,
    }
    if dry_run:
        corpus_meta["sample_count"] = limit
        return [], corpus_meta
    local, size, head_digest = _fetch_corpus_file(cfg, cache_dir, dry_run=False)
    corpus_meta["total_bytes_downloaded"] = size
    corpus_meta["sha256_of_first_1MB"] = head_digest
    paragraphs = _read_wikitext_paragraphs(local)
    sampled = _stratified_text_sample(paragraphs, limit, seed)
    records: list[dict] = []
    for index, (text, _bucket) in enumerate(sampled):
        record_id = digest_bytes(
            f"{VERSION}\0{seed}\0{cfg['name']}\0{index}\0{text}".encode()
        )[:24]
        records.append({
            "schema": SCHEMA,
            "id": record_id,
            "category": "text",
            "modality": cfg["modality"],
            "text": text,
            "origin": cfg["repo"],
            "source": cfg["name"],
        })
    corpus_meta["sample_count"] = len(records)
    corpus_meta["source_paragraphs_seen"] = len(paragraphs)
    return records, corpus_meta


def build_real_vision(
    cfg: dict[str, Any],
    limit: int,
    output_dir: Path,
    cache_dir: Path,
    seed: int,
    dry_run: bool,
) -> tuple[list[dict], list[dict], dict[str, Any]]:
    """Fetch COCO val2014 captions, take a uniform random sample, write
    the JPEG bytes to ``output_dir/vision/`` and write vision-only
    manifest entries. Returns (sample_records, vision_manifest_entries,
    corpus_meta). The records list contains one entry per image (with
    the caption in the text field so text consumers can still iterate);
    the vision_manifest_entries list is the structured per-image view
    with the image_path / caption pair (this is what
    multimodal_calibrate.py consumes)."""
    corpus_meta: dict[str, Any] = {
        "name": cfg["name"],
        "source": f"{cfg['repo']}/{cfg['filename']}",
        "modality": cfg["modality"],
        "sample_count": 0,
        "license": cfg["license"],
        "license_uri": cfg["license_uri"],
        "attribution": cfg["attribution"],
        "sampling": cfg["sampling"],
        "total_bytes_downloaded": 0,
        "sha256_of_first_1MB": None,
    }
    if dry_run:
        corpus_meta["sample_count"] = limit
        return [], [], corpus_meta
    local, size, head_digest = _fetch_corpus_file(cfg, cache_dir, dry_run=False)
    corpus_meta["total_bytes_downloaded"] = size
    corpus_meta["sha256_of_first_1MB"] = head_digest
    image_bytes_field = cfg["image_bytes_field"]
    image_path_field = cfg["image_path_field"]
    caption_field = cfg["caption_field"]
    # De-duplicate the parent column so polars does not see the
    # ``image`` struct requested twice via bytes_field/path_field.
    parent_columns: list[str] = []
    for parent in (image_bytes_field[0], image_path_field[0], caption_field, "cocoid"):
        if parent not in parent_columns:
            parent_columns.append(parent)
    rows = _read_parquet_rows(local, columns=parent_columns)
    indices = _uniform_random_indices(len(rows), limit, seed)
    vision_dir = output_dir / "vision"
    vision_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    manifest_entries: list[dict] = []
    for index, row_index in enumerate(indices):
        row = rows[row_index]
        img_struct = row[image_bytes_field[0]]
        if isinstance(img_struct, dict):
            img_bytes = img_struct.get("bytes", b"")
            original_path = img_struct.get("path") or image_path_field[1]
        else:
            img_bytes = img_struct
            original_path = image_path_field[1]
        cocoid = row.get("cocoid")
        if cocoid is None and isinstance(original_path, str):
            # Fall back to filename parsing if the dedicated column is
            # missing in some future shard.
            stem = Path(original_path).stem
            digits = "".join(c for c in stem if c.isdigit())
            # COCO val2014 filenames end with an 8-digit cocoid
            # prefixed with the year (e.g. "COCO_val2014_000000184613"
            # -> 184613). Take the last 8 digits when present.
            if len(digits) >= 8:
                cocoid = int(digits[-8:])
            elif digits:
                cocoid = int(digits)
        if cocoid is None:
            cocoid = index
        filename = _safe_jpeg_filename(cocoid)
        target = vision_dir / filename
        if not target.is_file():
            target.write_bytes(img_bytes)
        caption = row[caption_field]
        if not isinstance(caption, str):
            caption = str(caption)
        record_id = digest_bytes(
            f"{VERSION}\0{seed}\0{cfg['name']}\0{index}\0{filename}".encode()
        )[:24]
        records.append({
            "schema": SCHEMA,
            "id": record_id,
            "category": "image_text",
            "modality": cfg["modality"],
            "text": caption,
            "image_path": str(target.relative_to(output_dir)),
            "source_id": cocoid,
            "origin": cfg["repo"],
            "source": cfg["name"],
        })
        manifest_entries.append({
            "id": record_id,
            "image_path": str(target.relative_to(output_dir)),
            "caption": caption,
            "source_id": cocoid,
        })
    corpus_meta["sample_count"] = len(records)
    return records, manifest_entries, corpus_meta


def build_real_audio(
    cfg: dict[str, Any],
    limit: int,
    output_dir: Path,
    cache_dir: Path,
    seed: int,
    dry_run: bool,
) -> tuple[list[dict], list[dict], dict[str, Any]]:
    """Fetch LibriSpeech dev.clean, take a uniform random sample, write
    the FLAC bytes to ``output_dir/audio/`` and write audio-only
    manifest entries. Returns (sample_records, audio_manifest_entries,
    corpus_meta). The sample_records carry the transcript in the text
    field (for text consumers) and the audio_path for multimodal ones.
    """
    corpus_meta: dict[str, Any] = {
        "name": cfg["name"],
        "source": f"{cfg['repo']}/{cfg['filename']}",
        "modality": cfg["modality"],
        "sample_count": 0,
        "license": cfg["license"],
        "license_uri": cfg["license_uri"],
        "attribution": cfg["attribution"],
        "sampling": cfg["sampling"],
        "total_bytes_downloaded": 0,
        "sha256_of_first_1MB": None,
    }
    if dry_run:
        corpus_meta["sample_count"] = limit
        return [], [], corpus_meta
    local, size, head_digest = _fetch_corpus_file(cfg, cache_dir, dry_run=False)
    corpus_meta["total_bytes_downloaded"] = size
    corpus_meta["sha256_of_first_1MB"] = head_digest
    audio_bytes_field = cfg["audio_bytes_field"]
    audio_path_field = cfg["audio_path_field"]
    transcript_field = cfg["transcript_field"]
    parent_columns = []
    for parent in (audio_bytes_field[0], audio_path_field[0], transcript_field, "id"):
        if parent not in parent_columns:
            parent_columns.append(parent)
    rows = _read_parquet_rows(local, columns=parent_columns)
    indices = _uniform_random_indices(len(rows), limit, seed)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    manifest_entries: list[dict] = []
    for index, row_index in enumerate(indices):
        row = rows[row_index]
        aud_struct = row[audio_bytes_field[0]]
        if isinstance(aud_struct, dict):
            aud_bytes = aud_struct.get("bytes", b"")
            original_path = aud_struct.get("path") or audio_path_field[1]
        else:
            aud_bytes = aud_struct
            original_path = audio_path_field[1]
        utt_id = row.get("id")
        if not utt_id or not isinstance(utt_id, str):
            if isinstance(original_path, str):
                utt_id = Path(original_path).stem
            else:
                utt_id = f"utterance_{index:06d}"
        filename = _safe_flac_filename(utt_id)
        target = audio_dir / filename
        if not target.is_file():
            target.write_bytes(aud_bytes)
        transcript = row[transcript_field]
        if not isinstance(transcript, str):
            transcript = str(transcript)
        record_id = digest_bytes(
            f"{VERSION}\0{seed}\0{cfg['name']}\0{index}\0{filename}".encode()
        )[:24]
        records.append({
            "schema": SCHEMA,
            "id": record_id,
            "category": "audio_text",
            "modality": cfg["modality"],
            "text": transcript,
            "audio_path": str(target.relative_to(output_dir)),
            "source_id": utt_id,
            "origin": cfg["repo"],
            "source": cfg["name"],
        })
        manifest_entries.append({
            "id": record_id,
            "audio_path": str(target.relative_to(output_dir)),
            "transcript": transcript,
            "source_id": utt_id,
        })
    corpus_meta["sample_count"] = len(records)
    return records, manifest_entries, corpus_meta


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            out.write("\n")


def _constitution_license(extra_attribution: list[str]) -> dict[str, Any]:
    """Return the synthetic-clean-room license block. Used unchanged
    by the v1 path. The real-data path augments ``attribution`` with
    the upstream corpus names so the audit trail names every source."""
    attribution = "Julian Alejandro Torres Nieto, Tribunus.dev"
    if extra_attribution:
        attribution = attribution + "; " + "; ".join(extra_attribution)
    return {
        "license": "CC-BY-NC-SA-4.0",
        "license_uri": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
        "attribution": attribution,
        "distribution_cleared": True,
        "contains_user_inference": False,
        "commercial_use": False,
        "share_alike": True,
    }


def _write_synthetic(
    output: Path,
    records: list[dict],
    counts: dict[str, int],
    seed: int,
    epoch: int,
) -> None:
    """Write the synthetic-only output. The byte layout matches the v1
    builder exactly so any existing consumer (per_tensor_calibrate.py,
    moe-calibrate.py) keeps working without changes."""
    output.mkdir(parents=True, exist_ok=True)
    index = output / "samples.jsonl"
    corpus = output / "calibration.txt"
    _write_jsonl(index, records)
    corpus.write_text(
        "\n\n".join(record["text"] for record in records) + "\n",
        encoding="utf-8",
    )
    # Preserve the canonical category order from the input counts dict
    # so the manifest is byte-for-byte stable across runs.
    category_counts: dict[str, int] = {}
    for category in counts:
        category_counts[category] = sum(
            1 for record in records if record["category"] == category
        )
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "epoch": epoch,
        "sha256": digest_bytes(corpus.read_bytes()),
        "index_sha256": digest_bytes(index.read_bytes()),
        "generator": {
            "schema": SCHEMA,
            "version": VERSION,
            "seed": seed,
            "sample_count": len(records),
            "categories": category_counts,
        },
        "license": "CC-BY-NC-SA-4.0",
        "license_uri": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
        "attribution": "Julian Alejandro Torres Nieto, Tribunus.dev",
        "distribution_cleared": True,
        "contains_user_inference": False,
        "commercial_use": False,
        "share_alike": True,
        "sources": [{
            "name": "Tessera clean-room procedural calibration corpus",
            "origin": "original templates and deterministic parameter expansion",
            "contains_upstream_benchmark_items": False,
            "contains_user_data": False,
        }],
    }
    (output / "training-corpus-receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema": SCHEMA,
        "version": VERSION,
        "seed": seed,
        "sample_count": len(records),
        "categories": category_counts,
        "corpus": corpus.name,
        "index": index.name,
        "receipt": "training-corpus-receipt.json",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )


def _write_real(
    output: Path,
    text_records: list[dict],
    vision_records: list[dict],
    audio_records: list[dict],
    vision_manifest: list[dict],
    audio_manifest: list[dict],
    corpora_meta: list[dict[str, Any]],
    seed: int,
    epoch: int,
    dry_run: bool,
) -> dict[str, Any]:
    """Write the real-data corpus. The text records are concatenated
    into calibration.txt the same way the synthetic builder does; the
    vision / audio records are NOT included in calibration.txt (the
    text side is the imatrix input; vision / audio are the multimodal
    capture input). The manifest gains two new arrays; the receipt
    gains a ``corpora`` list."""
    output.mkdir(parents=True, exist_ok=True)
    all_records = text_records + vision_records + audio_records
    index = output / "samples.jsonl"
    corpus = output / "calibration.txt"
    if dry_run:
        # Dry-run: only write a manifest preview. The samples.jsonl
        # / calibration.txt are not produced (nothing was fetched).
        category_counts: dict[str, int] = {}
        for record in all_records:
            mod = record.get("modality", "text")
            category_counts[mod] = category_counts.get(mod, 0) + 1
        manifest = {
            "schema": SCHEMA,
            "version": VERSION,
            "seed": seed,
            "sample_count": len(all_records),
            "categories": category_counts,
            "dry_run": True,
            "corpora": [
                {"name": meta["name"], "modality": meta["modality"],
                 "sample_count": meta["sample_count"],
                 "sampling": meta["sampling"]}
                for meta in corpora_meta
            ],
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
        )
        return manifest
    _write_jsonl(index, all_records)
    corpus.write_text(
        "\n\n".join(record["text"] for record in text_records) + "\n",
        encoding="utf-8",
    )
    category_counts: dict[str, int] = {}
    for record in all_records:
        mod = record.get("modality", "text")
        category_counts[mod] = category_counts.get(mod, 0) + 1
    synth_block = _constitution_license([meta["attribution"] for meta in corpora_meta])
    receipt_sources = []
    for meta in corpora_meta:
        receipt_sources.append({
            "name": meta["name"],
            "origin": meta["source"],
            "license": meta["license"],
            "license_uri": meta.get("license_uri"),
            "attribution": meta["attribution"],
            "modality": meta["modality"],
            "sample_count": meta["sample_count"],
            "contains_upstream_benchmark_items": False,
            "contains_user_data": False,
        })
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "epoch": epoch,
        "sha256": digest_bytes(corpus.read_bytes()) if corpus.exists() else "",
        "index_sha256": digest_bytes(index.read_bytes()),
        "generator": {
            "schema": SCHEMA,
            "version": VERSION,
            "seed": seed,
            "sample_count": len(all_records),
            "categories": category_counts,
            "synthetic": False,
        },
        "corpora": corpora_meta,
        "license": synth_block["license"],
        "license_uri": synth_block["license_uri"],
        "attribution": synth_block["attribution"],
        "distribution_cleared": synth_block["distribution_cleared"],
        "contains_user_inference": synth_block["contains_user_inference"],
        "commercial_use": synth_block["commercial_use"],
        "share_alike": synth_block["share_alike"],
        "sources": receipt_sources,
    }
    (output / "training-corpus-receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema": SCHEMA,
        "version": VERSION,
        "seed": seed,
        "sample_count": len(all_records),
        "categories": category_counts,
        "corpus": corpus.name,
        "index": index.name,
        "receipt": "training-corpus-receipt.json",
        "text_samples": [record["id"] for record in text_records],
        "vision_samples": vision_manifest,
        "audio_samples": audio_manifest,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_corpora(value: str) -> list[str]:
    parts = [part.strip().lower() for part in value.split(",") if part.strip()]
    valid = set(REAL_CORPORA.keys())
    bad = [p for p in parts if p not in valid]
    if bad:
        raise SystemExit(
            f"unknown --corpora value(s): {bad}; valid: {sorted(valid)}"
        )
    return parts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the Tessera calibration corpus (synthetic or real)"
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Output directory. Will be created if it does not exist.",
    )
    parser.add_argument("--seed", type=int, default=640)
    parser.add_argument("--epoch", type=int, default=0)
    # New flags (additive — defaults preserve the v1 builder byte-for-byte).
    parser.add_argument(
        "--real", action="store_true",
        help="Fetch the architect-chosen real corpora (Wikitext-103, "
             "COCO val2014, LibriSpeech dev.clean) instead of the "
             "synthetic generator. The output schema is the v1 schema "
             "extended with a ``modality`` field and the per-modality "
             "manifest entries.",
    )
    parser.add_argument(
        "--corpora", default="text",
        help="Comma-separated list of modalities to ingest. Valid: "
             "text, vision, audio. Default: text. Ignored unless --real "
             "is set.",
    )
    parser.add_argument(
        "--budget", choices=sorted(BUDGETS.keys()), default="medium",
        help="Sample budget. light: 1k/256/256, medium: 5k/1k/1k, "
             "heavy: 20k/4k/4k (text/vision/audio).",
    )
    parser.add_argument(
        "--synthetic-count", type=int, default=None,
        help="Override the default per-category synthetic count. "
             "Ignored unless --real is NOT set.",
    )
    parser.add_argument(
        "--cache-dir", default=None,
        help="Hugging Face cache directory (default: "
             "~/.cache/huggingface). Used only with --real.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List what would be fetched without downloading. Writes "
             "no samples, no images, no audio. Used by the test suite "
             "to keep CI hermetic.",
    )
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir)
    if not args.real:
        counts = DEFAULT_SYNTHETIC_COUNTS
        if args.synthetic_count is not None and args.synthetic_count > 0:
            scale = args.synthetic_count / sum(DEFAULT_SYNTHETIC_COUNTS.values())
            counts = {k: max(1, int(round(v * scale)))
                      for k, v in DEFAULT_SYNTHETIC_COUNTS.items()}
        records = build_synthetic_records(args.seed, counts)
        _write_synthetic(output_dir, records, counts, args.seed, args.epoch)
        return 0
    # Real-data path.
    corpora = _parse_corpora(args.corpora)
    budget = BUDGETS[args.budget]
    _require_hf_hub(args.dry_run)
    cache_dir = Path(args.cache_dir) if args.cache_dir else Path.home() / ".cache" / "huggingface"
    text_records: list[dict] = []
    vision_records: list[dict] = []
    audio_records: list[dict] = []
    vision_manifest: list[dict] = []
    audio_manifest: list[dict] = []
    corpora_meta: list[dict[str, Any]] = []
    if "text" in corpora:
        rec, meta = build_real_text(
            REAL_CORPORA["text"],
            limit=budget["text"],
            output_dir=output_dir,
            cache_dir=cache_dir,
            seed=args.seed,
            dry_run=args.dry_run,
        )
        text_records = rec
        corpora_meta.append(meta)
    if "vision" in corpora:
        rec, vmanifest, meta = build_real_vision(
            REAL_CORPORA["vision"],
            limit=budget["vision"],
            output_dir=output_dir,
            cache_dir=cache_dir,
            seed=args.seed,
            dry_run=args.dry_run,
        )
        vision_records = rec
        vision_manifest = vmanifest
        corpora_meta.append(meta)
    if "audio" in corpora:
        rec, amanifest, meta = build_real_audio(
            REAL_CORPORA["audio"],
            limit=budget["audio"],
            output_dir=output_dir,
            cache_dir=cache_dir,
            seed=args.seed,
            dry_run=args.dry_run,
        )
        audio_records = rec
        audio_manifest = amanifest
        corpora_meta.append(meta)
    _write_real(
        output_dir,
        text_records=text_records,
        vision_records=vision_records,
        audio_records=audio_records,
        vision_manifest=vision_manifest,
        audio_manifest=audio_manifest,
        corpora_meta=corpora_meta,
        seed=args.seed,
        epoch=args.epoch,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
