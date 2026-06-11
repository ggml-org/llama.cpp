#!/usr/bin/env python3
"""
Evaluates llama.cpp's DeepSeek-OCR by comparing its output for a test
image to the actual text in part of that image.

Runs each test image through mtmd-cli, calculates CER and chrF for
its output, and holds them against the HF model's scores.
"""

import argparse
import logging
import subprocess
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("deepseek-ocr-test")

RUN_TIMEOUT = 300


@dataclass
class ModelSpec:
    key: str
    label: str
    model_arg: str
    mmproj_arg: str
    model_default: str
    mmproj_default: str


@dataclass
class TestCase:
    model_key: str
    label: str
    image: str
    ground_truth: str
    hf_cer: float
    hf_chrf: float
    cer_tol: float
    chrf_tol: float

    @property
    def cer_max(self) -> float:
        return self.hf_cer + self.cer_tol

    @property
    def chrf_min(self) -> float:
        return self.hf_chrf - self.chrf_tol


MODELS = {
    "v1": ModelSpec(
        key="v1", label="DeepSeek-OCR",
        model_arg="--llama-model", mmproj_arg="--mmproj",
        model_default="gguf_models/deepseek-ai/deepseek-ocr-bf16.gguf",
        mmproj_default="gguf_models/deepseek-ai/mmproj-deepseek-ocr-bf16.gguf",
    ),
    "v2": ModelSpec(
        key="v2", label="DeepSeek-OCR-2",
        model_arg="--llama-model-2", mmproj_arg="--mmproj-2",
        model_default="gguf_models/deepseek-ai/deepseek-ocr-2-bf16.gguf",
        mmproj_default="gguf_models/deepseek-ai/mmproj-deepseek-ocr-2-bf16.gguf",
    ),
}

CASES = [
    TestCase(
        model_key="v1", label="single-view scan",
        image="tools/mtmd/test-1.jpeg",
        ground_truth="tools/mtmd/tests/test-1-ground-truth.txt",
        hf_cer=0.3030, hf_chrf=67.52, cer_tol=0.02, chrf_tol=2.0,
    ),
    TestCase(
        model_key="v2", label="single-view scan",
        image="tools/mtmd/test-1.jpeg",
        ground_truth="tools/mtmd/tests/test-1-ground-truth.txt",
        # 640x488 is below the 768 tiling threshold -- single 1024 global view.
        # hf_cer/hf_chrf are the deepseek-ai repo's own scores (ImageOps.pad);
        # the transformers HF processor is *not* the reference -- its pad_to_square
        # is one pixel off and lands at ~0.69 instead.
        hf_cer=0.7761, hf_chrf=28.70, cer_tol=0.12, chrf_tol=8.0,
    ),
]


def arg_dest(flag: str) -> str:
    return flag.lstrip("-").replace("-", "_")


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
        "-n", "512",  # cap loops on hard images (KV would otherwise fill)
        # HF decodes with no_repeat_ngram_size; llama.cpp's analog is DRY.
        # Default DRY breakers include "\n", so they are cleared below.
        "--dry-multiplier", "0.8",
        "--dry-base", "1.75",
        "--dry-allowed-length", "2",
        "--dry-penalty-last-n", "-1",
        "--dry-sequence-breaker", "none",
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


def evaluate(case: "TestCase", expected: str, ocr_out: str) -> bool:
    expected = normalize_text(expected)
    ocr_out = normalize_text(ocr_out)
    aligned = locally_align(expected, ocr_out)

    logger.debug(f"\n--- expected (normalized) ---\n{expected}")
    logger.debug(f"\n--- OCR output (normalized) ---\n{ocr_out}")
    logger.debug(f"\n--- aligned span ---\n{aligned}")

    cer = compute_cer(expected, aligned)
    chrf = compute_chrf(expected, aligned)

    cer_pass = cer <= case.cer_max
    chrf_pass = chrf >= case.chrf_min
    passed = cer_pass and chrf_pass

    logger.info("")
    logger.info("=" * 60)
    logger.info("Free OCR evaluation:")
    logger.info("=" * 60)
    logger.info(f"  CER               {cer:>7.4f}    (HF {case.hf_cer:.4f}, <= {case.cer_max:>7.4f}  -> {verdict(cer_pass)})")
    logger.info(f"  chrF (0-100)      {chrf:>7.2f}    (HF {case.hf_chrf:.2f}, >= {case.chrf_min:>7.2f}  -> {verdict(chrf_pass)})")
    logger.info(f"  Expected chars    {len(expected):>7}")
    logger.info(f"  Aligned chars     {len(aligned):>7} (of {len(ocr_out)} OCR chars)")
    logger.info("")
    logger.info(f"  Result: {verdict(passed)}")
    logger.info("=" * 60)
    return passed


def argument_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Compare llama.cpp DeepSeek-OCR output with a ground-truth transcript")
    ap.add_argument("--llama-bin", default="build/bin/llama-mtmd-cli",
                    help="Path to llama-mtmd-cli binary (relative to repo root or absolute)")
    for spec in MODELS.values():
        ap.add_argument(spec.model_arg, default=spec.model_default,
                        help=f"Path to the {spec.label} GGUF model (relative to repo root or absolute)")
        ap.add_argument(spec.mmproj_arg, default=spec.mmproj_default,
                        help=f"Path to the {spec.label} mmproj GGUF file (relative to repo root or absolute)")
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

    repo_root = Path(__file__).resolve().parents[3]  # tests -> mtmd -> tools -> repo root
    binary = resolve_path(args.llama_bin, repo_root)

    if not binary.exists():
        logger.error(f"Error: binary not found: {binary}")
        return 1

    logger.info("=" * 60)
    logger.info("DeepSeek-OCR: llama.cpp vs HF parity check")
    logger.info("=" * 60)

    results = {}
    for case in CASES:
        model_spec = MODELS[case.model_key]
        title = f"{model_spec.label} -- {case.label}"

        logger.info("")
        logger.info(f"=== {title} ===")

        model = resolve_path(getattr(args, arg_dest(model_spec.model_arg)), repo_root)
        mmproj = resolve_path(getattr(args, arg_dest(model_spec.mmproj_arg)), repo_root)
        image = resolve_path(case.image, repo_root)
        ground_truth = resolve_path(case.ground_truth, repo_root)

        missing = [(lbl, p) for lbl, p in [("model", model), ("mmproj", mmproj),
                                           ("image", image), ("ground-truth", ground_truth)]
                   if not p.exists()]
        if missing:
            for lbl, p in missing:
                logger.error(f"  Error: {lbl} not found: {p}")
            results[title] = False
            continue

        expected = read_expected_text(ground_truth)
        logger.info(f"  Image: {case.image}")
        logger.info(f"  Expected text: {len(expected)} chars")
        logger.info("  Running llama.cpp 'Free OCR'")
        try:
            ocr_out = run_mtmd_cli(model, mmproj, image, binary)
        except RuntimeError as e:
            logger.error(f"  Error: {e}")
            results[title] = False
            continue

        results[title] = evaluate(case, expected, ocr_out)

    logger.info("")
    logger.info("=== Summary ===")
    for title, ok in results.items():
        logger.info(f"  {title:<48} {verdict(ok)}")
    all_passed = all(results.values())
    logger.info(f"Overall: {verdict(all_passed)}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
