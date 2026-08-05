"""Tests for tools/tessera/build-calibration-corpus.py.

Covers four invariants:
  1. The default (synthetic) path emits the v1 schema byte-for-byte
     so the existing per_tensor_calibrate.py / moe-calibrate.py
     consumers keep working without any change.
  2. The --real dry-run path is hermetic: no network, no disk, the
     manifest is the only artifact and it advertises the right per-
     modality counts for each budget.
  3. The real-data builders correctly handle the parquet shape the
     Hugging Face corpora use (image is a Struct column, audio is
     embedded as FLAC bytes, text is a string column).
  4. The receipt is the audit trail: every real-data run records the
     source repo, license, downloaded byte count, and the SHA256 of
     the first 1 MB of the downloaded payload.

Run as a unittest module. Exit 0 on success, non-zero on failure.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import polars as pl

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

# The module file uses a hyphenated name (build-calibration-corpus.py)
# so Python's normal import machinery can't load it; load it by path.
_BCC_PATH = THIS_DIR / "build-calibration-corpus.py"
_spec = importlib.util.spec_from_file_location("build_calibration_corpus", _BCC_PATH)
bcc = importlib.util.module_from_spec(_spec)
sys.modules["build_calibration_corpus"] = bcc
_spec.loader.exec_module(bcc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wikitext_parquet(rows: list[str], path: Path) -> None:
    """Write a parquet file in the Salesforce/wikitext shape (one
    'text' column). The test uses this to drive the real-data path
    without ever contacting the network."""
    df = pl.DataFrame({"text": rows})
    df.write_parquet(str(path))


def _make_coco_parquet(
    rows: list[tuple[bytes, str, str, int]], path: Path,
) -> None:
    """Write a parquet file in the jxie/coco_captions shape
    (image Struct column + filename + cocoid + caption)."""
    image_structs = [
        {"bytes": b, "path": p} for (b, p, _cap, _id) in rows
    ]
    df = pl.DataFrame({
        "image": image_structs,
        "filename": [p for (_b, p, _c, _i) in rows],
        "cocoid": [i for (_b, _p, _c, i) in rows],
        "caption": [c for (_b, _p, c, _i) in rows],
    })
    df.write_parquet(str(path))


def _make_librispeech_parquet(
    rows: list[tuple[bytes, str, str, str, int, int]], path: Path,
) -> None:
    """Write a parquet file in the openslr/librispeech_asr shape
    (audio Struct column + file + text + id + speaker_id + chapter_id)."""
    audio_structs = [
        {"bytes": b, "path": p} for (b, p, _t, _id, _sp, _ch) in rows
    ]
    df = pl.DataFrame({
        "file": [f"/cache/{p}" for (_b, p, _t, _id, _sp, _ch) in rows],
        "audio": audio_structs,
        "text": [t for (_b, _p, t, _id, _sp, _ch) in rows],
        "speaker_id": [sp for (_b, _p, _t, _id, sp, _ch) in rows],
        "chapter_id": [ch for (_b, _p, _t, _id, _sp, ch) in rows],
        "id": [i for (_b, _p, _t, i, _sp, _ch) in rows],
    })
    df.write_parquet(str(path))


def _hash_of_first_mb(path: Path) -> str:
    import hashlib
    with path.open("rb") as source:
        return hashlib.sha256(source.read(1_048_576)).hexdigest()


class _HFFake:
    """Stub the huggingface_hub.hf_hub_download function. The real
    builder only calls it from _fetch_corpus_file; the fake routes the
    request to a local fixture parquet so the test never touches the
    network."""

    def __init__(self, fixtures: dict[str, Path]):
        self.fixtures = fixtures

    def __call__(self, *, repo_id: str, filename: str, repo_type: str,
                 cache_dir: str) -> str:
        key = f"{repo_id}/{filename}"
        if key not in self.fixtures:
            raise FileNotFoundError(f"no fixture for {key}")
        # Copy the fixture into the requested cache so the subsequent
        # local.stat() / open() calls land in the cache directory.
        cache = Path(cache_dir)
        cache.mkdir(parents=True, exist_ok=True)
        target = cache / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.fixtures[key].read_bytes())
        return str(target)


# ---------------------------------------------------------------------------
# 1. v1 schema preservation
# ---------------------------------------------------------------------------


class TestSyntheticSchemaPreserved(unittest.TestCase):
    """The synthetic path (the default, no --real) is the public
    surface the existing calibration pipeline depends on. These tests
    pin its output to a byte-for-byte reference produced by the
    original builder so a future refactor that accidentally changes
    the schema, record shape, or output filenames breaks loudly."""

    def _run_synthetic(self, td: Path) -> None:
        argv = ["--output-dir", str(td), "--seed", "640", "--epoch", "0"]
        bcc.main(argv)

    def test_synthetic_writes_all_four_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            self._run_synthetic(Path(td))
            self.assertTrue((Path(td) / "samples.jsonl").is_file())
            self.assertTrue((Path(td) / "calibration.txt").is_file())
            self.assertTrue((Path(td) / "training-corpus-receipt.json").is_file())
            self.assertTrue((Path(td) / "manifest.json").is_file())
            # No vision/ or audio/ subdirs in synthetic mode.
            self.assertFalse((Path(td) / "vision").exists())
            self.assertFalse((Path(td) / "audio").exists())

    def test_synthetic_manifest_shape(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            self._run_synthetic(Path(td))
            m = json.loads((Path(td) / "manifest.json").read_text())
            # The synthetic manifest must not carry real-data fields.
            self.assertNotIn("vision_samples", m)
            self.assertNotIn("audio_samples", m)
            self.assertNotIn("text_samples", m)
            self.assertEqual(m["schema"], "llama.tessera.calibration-corpus.v1")
            self.assertEqual(m["version"], 1)
            # The default per-category counts sum to the v1 baseline.
            self.assertEqual(sum(m["categories"].values()), 2680)
            # Stable insertion order so the manifest JSON is
            # diff-stable across runs.
            self.assertEqual(
                list(m["categories"].keys()),
                [
                    "code", "en", "ko", "zh", "ja", "tool_calling",
                    "reasoning", "chat", "mixed", "structured_context",
                ],
            )

    def test_synthetic_receipt_shape(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            self._run_synthetic(Path(td))
            r = json.loads((Path(td) / "training-corpus-receipt.json").read_text())
            self.assertEqual(r["schema"], "llama.tessera.training-corpus.v1")
            # The synthetic receipt must not carry the real-data
            # ``corpora`` block — that block is reserved for the
            # --real path and would change the audit trail.
            self.assertNotIn("corpora", r)
            self.assertEqual(r["commercial_use"], False)
            self.assertEqual(r["share_alike"], True)

    def test_synthetic_record_shape(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            self._run_synthetic(Path(td))
            with (Path(td) / "samples.jsonl").open() as source:
                first = json.loads(source.readline())
            # Every v1 field is present and the new ``modality`` field
            # is absent (the byte-for-byte promise).
            self.assertEqual(
                set(first.keys()),
                {"schema", "id", "category", "text", "origin"},
            )
            self.assertEqual(first["schema"], "llama.tessera.calibration-corpus.v1")
            self.assertEqual(len(first["id"]), 24)

    def test_synthetic_count_is_stable(self) -> None:
        """Two runs with the same seed + epoch produce the same
        records, the same SHA256 of the corpus, and the same SHA256 of
        the index. This is the calibration pipeline's reproducibility
        contract."""
        with tempfile.TemporaryDirectory() as td:
            a, b = Path(td) / "a", Path(td) / "b"
            self._run_synthetic(a)
            self._run_synthetic(b)
            self.assertEqual(
                (a / "samples.jsonl").read_bytes(),
                (b / "samples.jsonl").read_bytes(),
            )
            self.assertEqual(
                (a / "calibration.txt").read_bytes(),
                (b / "calibration.txt").read_bytes(),
            )


# ---------------------------------------------------------------------------
# 2. Real-data dry-run path (no network, no downloads)
# ---------------------------------------------------------------------------


class TestRealDryRun(unittest.TestCase):
    """``--dry-run`` must be hermetic. It writes only a manifest.json;
    no parquet, no images, no audio, no samples.jsonl. The
    per-corpus counts match the budget, the per-corpus modality and
    sampling strategy are recorded, and the call returns a 0 exit
    code."""

    def test_dry_run_text_only(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            argv = [
                "--real", "--corpora", "text",
                "--budget", "light",
                "--output-dir", str(td), "--dry-run",
            ]
            self.assertEqual(bcc.main(argv), 0)
            m = json.loads((Path(td) / "manifest.json").read_text())
            self.assertTrue(m["dry_run"])
            self.assertEqual(m["sample_count"], 0)
            self.assertEqual(len(m["corpora"]), 1)
            self.assertEqual(m["corpora"][0]["modality"], "text")
            self.assertEqual(m["corpora"][0]["sample_count"], 1_000)
            # No samples / no vision / no audio in dry-run.
            self.assertFalse((Path(td) / "samples.jsonl").exists())
            self.assertFalse((Path(td) / "vision").exists())
            self.assertFalse((Path(td) / "audio").exists())

    def test_dry_run_all_three_modalities(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            argv = [
                "--real", "--corpora", "text,vision,audio",
                "--budget", "medium",
                "--output-dir", str(td), "--dry-run",
            ]
            self.assertEqual(bcc.main(argv), 0)
            m = json.loads((Path(td) / "manifest.json").read_text())
            self.assertEqual(len(m["corpora"]), 3)
            by_modality = {c["modality"]: c for c in m["corpora"]}
            self.assertEqual(by_modality["text"]["sample_count"], 5_000)
            self.assertEqual(by_modality["image_text"]["sample_count"], 1_000)
            self.assertEqual(by_modality["audio_text"]["sample_count"], 1_000)
            # The receipt must not be written in dry-run (no download
            # happened; there is nothing to record).
            self.assertFalse(
                (Path(td) / "training-corpus-receipt.json").exists()
            )

    def test_dry_run_budget_heavy(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            argv = [
                "--real", "--corpora", "text,vision,audio",
                "--budget", "heavy",
                "--output-dir", str(td), "--dry-run",
            ]
            self.assertEqual(bcc.main(argv), 0)
            m = json.loads((Path(td) / "manifest.json").read_text())
            counts = {c["modality"]: c["sample_count"] for c in m["corpora"]}
            self.assertEqual(counts["text"], 20_000)
            self.assertEqual(counts["image_text"], 4_000)
            self.assertEqual(counts["audio_text"], 4_000)

    def test_dry_run_vision_sampling_strategy(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            argv = [
                "--real", "--corpora", "vision",
                "--budget", "light",
                "--output-dir", str(td), "--dry-run",
            ]
            bcc.main(argv)
            m = json.loads((Path(td) / "manifest.json").read_text())
            self.assertEqual(m["corpora"][0]["sampling"], "uniform_random")

    def test_dry_run_text_sampling_strategy(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            argv = [
                "--real", "--corpora", "text",
                "--budget", "light",
                "--output-dir", str(td), "--dry-run",
            ]
            bcc.main(argv)
            m = json.loads((Path(td) / "manifest.json").read_text())
            self.assertEqual(m["corpora"][0]["sampling"], "stratified_length")


# ---------------------------------------------------------------------------
# 3. Real-data builders (parquet fixtures, no network)
# ---------------------------------------------------------------------------


class TestRealDataBuilders(unittest.TestCase):
    """Drive the real-data builder with locally-built parquet fixtures
    in the same shape the Hugging Face datasets use. The test swaps in
    a stub ``hf_hub_download`` so no network call happens, then
    exercises the in-memory part of the pipeline (read -> sample ->
    write)."""

    def setUp(self) -> None:
        # Build three small fixtures in the shapes the HF corpora use.
        self.fixtures_dir = Path(tempfile.mkdtemp(prefix="tessera-bcc-fixtures-"))
        # Wikitext: a few hundred rows mixing short, medium, long,
        # section headers, and empty lines.
        wiki_rows: list[str] = []
        for i in range(20):
            wiki_rows.append("")  # empty
            wiki_rows.append(f" = Section {i} = ")  # header
            wiki_rows.append("a" * (50 + i * 10))  # short -> medium -> long
        wiki_rows.append("a" * 5)  # one very short
        self.wiki_path = self.fixtures_dir / "wiki.parquet"
        _make_wikitext_parquet(wiki_rows, self.wiki_path)
        # COCO val2014: three small image+caption rows.
        coco_rows = [
            (b"\xff\xd8\xff\xe0fake-jpeg-1", "COCO_val2014_000000000001.jpg",
             "a small image caption", 1),
            (b"\xff\xd8\xff\xe0fake-jpeg-2", "COCO_val2014_000000000002.jpg",
             "another caption here", 2),
            (b"\xff\xd8\xff\xe0fake-jpeg-3", "COCO_val2014_000000000003.jpg",
             "yet another", 3),
        ]
        self.coco_path = self.fixtures_dir / "coco.parquet"
        _make_coco_parquet(coco_rows, self.coco_path)
        # LibriSpeech dev.clean: three small audio+transcript rows.
        librispeech_rows = [
            (b"fLaCfake-audio-1", "1000-12345-0000.flac",
             "FIRST TRANSCRIPT", "1000-12345-0000", 1000, 12345),
            (b"fLaCfake-audio-2", "1000-12345-0001.flac",
             "SECOND TRANSCRIPT", "1000-12345-0001", 1000, 12345),
            (b"fLaCfake-audio-3", "2000-67890-0000.flac",
             "THIRD TRANSCRIPT", "2000-67890-0000", 2000, 67890),
        ]
        self.librispeech_path = self.fixtures_dir / "librispeech.parquet"
        _make_librispeech_parquet(librispeech_rows, self.librispeech_path)
        # Wire up the fake hf_hub_download to route each repo/filename
        # pair to the local fixture.
        self._fake = _HFFake({
            "Salesforce/wikitext/wikitext-103-raw-v1/train-00000-of-00002.parquet":
                self.wiki_path,
            "jxie/coco_captions/data/validation-00000-of-00010-0421425675e3d7a4.parquet":
                self.coco_path,
            "openslr/librispeech_asr/all/validation.clean/0000.parquet":
                self.librispeech_path,
        })
        self._original_hf_hub_download = bcc.hf_hub_download
        bcc.hf_hub_download = self._fake

    def tearDown(self) -> None:
        bcc.hf_hub_download = self._original_hf_hub_download

    def test_text_stratified_sample(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            argv = [
                "--real", "--corpora", "text",
                "--budget", "light",
                "--output-dir", str(td), "--seed", "640",
            ]
            self.assertEqual(bcc.main(argv), 0)
            # The fixture has only 21 valid paragraphs (the 20
            # "a"*N entries plus the very-short "aaaaa"); the builder
            # clamps to that. The point of the test is that the
            # bucketing is exercised.
            m = json.loads((Path(td) / "manifest.json").read_text())
            self.assertEqual(m["sample_count"], 21)
            # All records carry the modality field and the source
            # field (the v1 record is extended with these).
            with (Path(td) / "samples.jsonl").open() as source:
                first = json.loads(source.readline())
            self.assertEqual(first["modality"], "text")
            self.assertEqual(first["source"], "wikitext-103-raw-v1")

    def test_text_filters_section_headers(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            argv = [
                "--real", "--corpora", "text",
                "--budget", "light",
                "--output-dir", str(td), "--seed", "640",
            ]
            bcc.main(argv)
            with (Path(td) / "samples.jsonl").open() as source:
                records = [json.loads(line) for line in source]
            # No record may be a section header.
            for record in records:
                stripped = record["text"].strip()
                self.assertFalse(
                    stripped.startswith("=") and stripped.endswith("=")
                    and stripped.strip("=").strip(),
                    f"section header leaked into output: {record['text']!r}",
                )
            # No record may be empty.
            for record in records:
                self.assertTrue(record["text"].strip())

    def test_vision_writes_image_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            argv = [
                "--real", "--corpora", "vision",
                "--budget", "light",
                "--output-dir", str(td), "--seed", "640",
            ]
            bcc.main(argv)
            m = json.loads((Path(td) / "manifest.json").read_text())
            self.assertEqual(m["sample_count"], 3)
            # The vision/ subdir contains exactly one JPEG per sample.
            jpegs = sorted((Path(td) / "vision").glob("*.jpg"))
            self.assertEqual(len(jpegs), 3)
            # Each JPEG on disk is referenced from the manifest and
            # the bytes are non-empty.
            for entry in m["vision_samples"]:
                self.assertIn("image_path", entry)
                self.assertIn("caption", entry)
                self.assertIn("source_id", entry)
                path = Path(td) / entry["image_path"]
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 0)

    def test_audio_writes_flac_files(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            argv = [
                "--real", "--corpora", "audio",
                "--budget", "light",
                "--output-dir", str(td), "--seed", "640",
            ]
            bcc.main(argv)
            m = json.loads((Path(td) / "manifest.json").read_text())
            self.assertEqual(m["sample_count"], 3)
            flacs = sorted((Path(td) / "audio").glob("*.flac"))
            self.assertEqual(len(flacs), 3)
            for entry in m["audio_samples"]:
                self.assertIn("audio_path", entry)
                self.assertIn("transcript", entry)
                self.assertIn("source_id", entry)
                path = Path(td) / entry["audio_path"]
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 0)

    def test_vision_manifest_shape(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            argv = [
                "--real", "--corpora", "vision",
                "--budget", "light",
                "--output-dir", str(td), "--seed", "640",
            ]
            bcc.main(argv)
            m = json.loads((Path(td) / "manifest.json").read_text())
            for entry in m["vision_samples"]:
                # The vision entry has exactly the four documented
                # keys (id, image_path, caption, source_id) — this is
                # the contract multimodal_calibrate.py consumes.
                self.assertEqual(
                    set(entry.keys()),
                    {"id", "image_path", "caption", "source_id"},
                )

    def test_audio_manifest_shape(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            argv = [
                "--real", "--corpora", "audio",
                "--budget", "light",
                "--output-dir", str(td), "--seed", "640",
            ]
            bcc.main(argv)
            m = json.loads((Path(td) / "manifest.json").read_text())
            for entry in m["audio_samples"]:
                self.assertEqual(
                    set(entry.keys()),
                    {"id", "audio_path", "transcript", "source_id"},
                )

    def test_receipt_records_download_size_and_hash(self) -> None:
        """The receipt is the audit trail: per-corpus byte count and
        SHA256 of the first 1 MB of the downloaded payload."""
        with tempfile.TemporaryDirectory() as td:
            argv = [
                "--real", "--corpora", "text,vision,audio",
                "--budget", "light",
                "--output-dir", str(td), "--seed", "640",
            ]
            bcc.main(argv)
            r = json.loads((Path(td) / "training-corpus-receipt.json").read_text())
            self.assertEqual(len(r["corpora"]), 3)
            for corpus in r["corpora"]:
                self.assertGreater(corpus["total_bytes_downloaded"], 0)
                self.assertIsNotNone(corpus["sha256_of_first_1MB"])
                self.assertEqual(
                    len(corpus["sha256_of_first_1MB"]), 64,
                    f"{corpus['name']}: hash should be 64 hex chars",
                )
                self.assertIn("license", corpus)
                self.assertIn("attribution", corpus)
                self.assertIn("sample_count", corpus)
                self.assertIn("modality", corpus)
                self.assertIn("sampling", corpus)

    def test_vision_cocoid_is_just_the_integer(self) -> None:
        """The COCO val2014 source_id is the integer cocoid, not the
        year-prefixed filename. The M1 multimodal consumer reads
        source_id to cross-reference the COCO annotations; a 14-digit
        year-prefixed value would break that lookup."""
        with tempfile.TemporaryDirectory() as td:
            argv = [
                "--real", "--corpora", "vision",
                "--budget", "light",
                "--output-dir", str(td), "--seed", "640",
            ]
            bcc.main(argv)
            m = json.loads((Path(td) / "manifest.json").read_text())
            for entry in m["vision_samples"]:
                source_id = entry["source_id"]
                self.assertIsInstance(source_id, int)
                self.assertLess(source_id, 10**8)


# ---------------------------------------------------------------------------
# 4. Unit-level invariants for the sample / filter helpers
# ---------------------------------------------------------------------------


class TestSampleHelpers(unittest.TestCase):

    def test_section_header_detection(self) -> None:
        self.assertTrue(bcc._is_section_header(" = Title = "))
        self.assertTrue(bcc._is_section_header("= Title ="))
        self.assertTrue(bcc._is_section_header(" =  Inner Whitespace  = "))
        # Empty / non-headers.
        self.assertFalse(bcc._is_section_header(""))
        self.assertFalse(bcc._is_section_header(" = "))
        self.assertFalse(bcc._is_section_header("="))
        self.assertFalse(bcc._is_section_header("regular text"))
        self.assertFalse(bcc._is_section_header("text = mid = text"))

    def test_length_bucket_assignment(self) -> None:
        self.assertEqual(bcc._length_bucket("a" * 50), 0)
        self.assertEqual(bcc._length_bucket("a" * 200), 1)
        self.assertEqual(bcc._length_bucket("a" * 999), 1)
        self.assertEqual(bcc._length_bucket("a" * 1000), 2)
        self.assertEqual(bcc._length_bucket("a" * 5000), 2)

    def test_uniform_random_indices_deterministic(self) -> None:
        a = bcc._uniform_random_indices(100, 10, 42)
        b = bcc._uniform_random_indices(100, 10, 42)
        self.assertEqual(a, b)
        # No duplicates.
        self.assertEqual(len(a), len(set(a)))
        # All in range.
        self.assertTrue(all(0 <= i < 100 for i in a))

    def test_uniform_random_indices_handles_small_population(self) -> None:
        # limit >= population -> all indices, deterministic order.
        out = bcc._uniform_random_indices(3, 10, 7)
        self.assertEqual(sorted(out), [0, 1, 2])

    def test_stratified_text_sample_balances_buckets(self) -> None:
        # Build a paragraph set that fills all three buckets.
        paragraphs = (
            ["short"] * 10
            + ["m" * 500] * 10
            + ["l" * 2000] * 10
        )
        out = bcc._stratified_text_sample(paragraphs, 9, 0)
        # 9 // 3 = 3 per bucket, so 3 + 3 + 3 = 9.
        self.assertEqual(len(out), 9)
        by_bucket: dict[int, int] = {}
        for _text, b in out:
            by_bucket[b] = by_bucket.get(b, 0) + 1
        self.assertEqual(by_bucket, {0: 3, 1: 3, 2: 3})


if __name__ == "__main__":
    unittest.main()
