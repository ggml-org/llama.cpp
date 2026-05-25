#!/usr/bin/env python3
"""
Evaluates llama.cpp's DeepSeek-OCR by comparing its output for a test
image to the actual text in part of that image.

Runs the test image through mtmd-cli, calculates CER and chrF for
its output, and holds them against the HF model's scores.
"""

import argparse
import logging
import subprocess
import sys
import unicodedata
from pathlib import Path

logger = logging.getLogger("deepseek-ocr-test")

DEFAULT_IMAGE = "test-1.jpeg"
DEFAULT_EXPECTED_TEXT = "test-1-ground-truth.txt"
RUN_TIMEOUT = 300

# DeepSeek-OCR reference scores on the test image.
# This is the baseline the implementation should keep up with.
HF_REFERENCE_CER = 0.3030
HF_REFERENCE_CHRF = 67.52

CER_TOLERANCE = 0.02
CHRF_TOLERANCE = 2.0

CER_MAX = HF_REFERENCE_CER + CER_TOLERANCE
CHRF_MIN = HF_REFERENCE_CHRF - CHRF_TOLERANCE


def verdict(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def normalize_text(text: str) -> str:
    """NFC-normalize and collapse whitespace, so line-wrap and spacing
    don't count as CER errors."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def locally_align(expected: str, ocr_out: str) -> str:
    """Return the span of `ocr_out` that best matches `expected`.

    The ground truth covers part of the article body.
    But the test image includes half of the newspaper's front page.
    Fuzzy partial-ratio matching picks out
    the body so the unrelated text doesn't disturb CER / chrF.
    """
    from rapidfuzz import fuzz
    alignment = fuzz.partial_ratio_alignment(expected, ocr_out)
    if alignment is None or alignment.dest_end <= alignment.dest_start:
        return ocr_out
    return ocr_out[alignment.dest_start:alignment.dest_end]


def compute_cer(expected: str, ocr_out: str) -> float:
    """Character Error Rate. Lower is better.
    CER: fraction of characters you'd insert/delete/substitute to fix the output; 0 = perfect."""
    import jiwer
    return jiwer.cer(expected, ocr_out)


def compute_chrf(expected: str, ocr_out: str) -> float:
    """chrF score on 0-100. Higher is better.
    chrF: F-score over shared character n-grams; more forgiving of small word/spacing drift than CER.
    """
    from sacrebleu.metrics import CHRF
    return CHRF().sentence_score(ocr_out, [expected]).score


def run_mtmd_cli(model_path, mmproj_path, image_path, bin_path) -> str:
    """Run mtmd-cli on the image and return its output."""
    cmd = [
        str(bin_path),
        "-m", str(model_path),
        "--mmproj", str(mmproj_path),
        "--image", str(image_path),
        "-p", "Free OCR. ",
        "--chat-template", "deepseek-ocr",
        "--temp", "0",
        "--flash-attn", "off",  # match the HF "eager" attention reference
        "--no-warmup",
    ]
    logger.debug(f"  command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=False, timeout=RUN_TIMEOUT)
    except subprocess.TimeoutExpired as e:
        if e.stderr:
            logger.error("llama.cpp stderr:\n%s", e.stderr.decode("utf-8", errors="replace"))
        raise RuntimeError(f"llama-mtmd-cli timed out after {RUN_TIMEOUT}s")

    if result.returncode != 0:
        logger.error("llama.cpp stderr:\n%s", result.stderr.decode("utf-8", errors="replace"))
        raise RuntimeError(f"llama-mtmd-cli failed with code {result.returncode}")

    output = result.stdout.decode("utf-8", errors="replace").strip()
    if not output:
        raise RuntimeError("llama-mtmd-cli produced no output on stdout")
    logger.info(f"  output: {len(output)} chars")
    return output


def read_expected_text(file_path: Path) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def evaluate(expected: str, ocr_out: str) -> bool:
    expected = normalize_text(expected)
    ocr_out = normalize_text(ocr_out)
    aligned = locally_align(expected, ocr_out)

    logger.debug(f"\n--- expected (normalized) ---\n{expected}")
    logger.debug(f"\n--- OCR output (normalized) ---\n{ocr_out}")
    logger.debug(f"\n--- aligned span ---\n{aligned}")

    cer = compute_cer(expected, aligned)
    chrf = compute_chrf(expected, aligned)

    cer_pass = cer <= CER_MAX
    chrf_pass = chrf >= CHRF_MIN
    passed = cer_pass and chrf_pass

    logger.info("")
    logger.info("=" * 60)
    logger.info("Free OCR evaluation:")
    logger.info("=" * 60)
    logger.info(f"  CER               {cer:>7.4f}    (<= {CER_MAX:>7.4f}  -> {verdict(cer_pass)})")
    logger.info(f"  chrF (0-100)      {chrf:>7.2f}    (>= {CHRF_MIN:>7.2f}  -> {verdict(chrf_pass)})")
    logger.info(f"  Expected chars    {len(expected):>7}")
    logger.info(f"  Aligned chars     {len(aligned):>7} (of {len(ocr_out)} OCR chars)")
    logger.info("")
    logger.info(f"  Result: {verdict(passed)}")
    logger.info("=" * 60)
    return passed


def argument_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Compare llama.cpp DeepSeek-OCR output with a ground-truth transcript")
    ap.add_argument("--llama-model", default="gguf_models/deepseek-ai/deepseek-ocr-bf16.gguf",
                    help="Path to llama.cpp GGUF model (relative to repo root or absolute)")
    ap.add_argument("--mmproj", default="gguf_models/deepseek-ai/mmproj-deepseek-ocr-bf16.gguf",
                    help="Path to mmproj GGUF file (relative to repo root or absolute)")
    ap.add_argument("--llama-bin", default="build/bin/llama-mtmd-cli",
                    help="Path to llama-mtmd-cli binary (relative to repo root or absolute)")
    ap.add_argument("--verbose", action="store_true",
                    help="Also log the expected, OCR, and aligned text")
    return ap


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(message)s")


def resolve_path(path: str, base: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else base / p


def main() -> int:
    args = argument_parser().parse_args()
    configure_logging(args.verbose)

    tests_dir = Path(__file__).parent  # tools/mtmd/tests
    mtmd_dir = tests_dir.parent  # tools/mtmd
    repo_root = mtmd_dir.parent.parent  # repo root

    inputs = [
        ("image", resolve_path(DEFAULT_IMAGE, mtmd_dir)),
        ("expected-text", resolve_path(DEFAULT_EXPECTED_TEXT, tests_dir)),
        ("model", resolve_path(args.llama_model, repo_root)),
        ("mmproj", resolve_path(args.mmproj, repo_root)),
        ("binary", resolve_path(args.llama_bin, repo_root)),
    ]
    for label, path in inputs:
        if not path.exists():
            logger.error(f"Error: {label} not found: {path}")
            return 1
    paths = dict(inputs)

    logger.info("=" * 60)
    logger.info("DeepSeek-OCR: llama.cpp vs ground-truth comparison")
    logger.info("=" * 60)
    logger.info(f"HF baselines: CER {HF_REFERENCE_CER:.4f}, chrF {HF_REFERENCE_CHRF:.2f}")
    logger.info(f"Test thresholds: CER <= {CER_MAX:.4f}, chrF >= {CHRF_MIN:.2f}")

    logger.debug("")
    logger.debug("Resolved test inputs:")
    for label, path in inputs:
        logger.debug(f"  {label:<14} {path}")

    logger.info("")
    logger.info("[1/3] Running llama.cpp 'Free OCR'")
    try:
        ocr_out = run_mtmd_cli(paths["model"], paths["mmproj"],
                               paths["image"], paths["binary"])
    except RuntimeError as e:
        logger.error(f"Error: {e}")
        return 1

    logger.info("")
    logger.info("[2/3] Reading expected output")
    expected = read_expected_text(paths["expected-text"])
    logger.info(f"  expected: {len(expected)} chars")

    logger.info("")
    logger.info("[3/3] Computing OCR metrics")
    ok = evaluate(expected, ocr_out)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
