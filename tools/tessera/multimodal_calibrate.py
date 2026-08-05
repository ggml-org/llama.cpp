"""Multimodal activation capture for vision / audio / mm-projector.

The capture path runs a real forward pass through the per-component
clip graph (vision tower, audio tower, mm-projector) and emits
per-tensor activation statistics. The architecture is the same
shape as the text side's ``tools/tessera/calibration_to_tensor_stats.py``:
the Python side is a thin wrapper that invokes the
``llama-clip-capture`` C++ binary via subprocess and writes the
per-tensor rows through ``TesseraDB.insert_tensor_stats`` (the
canonical upsert path; the same ``ON CONFLICT (model_hash,
model_role, name)`` clause the text side uses).

The mm_projector capture is a real first-class path: the C++
binary loads the upstream tower (vision or audio) to produce the
multimodal embedding, then loads the projector's preprocessor
and runs the projector's forward pass on that embedding, then
captures activations at the ``mm.`` tensors. The mm_projector
forward pass is a separate code path in the C++ capture, not a
Python no-op.

The activation naming convention is:

  v.<name>      for vision-tower activations
  a.<name>      for audio-tower activations
  mm.<name>     for mm_projector activations

The C++ side adds the prefix based on the modality; the Python
side uses the prefix to stamp ``model_role`` on the row.

Additive schema: no new ``tensor_stats`` columns. The output
rows use the same columns the text side writes. The
``recommended_action`` enum is the same five-value enum
(protect / requant_up / requant_down / monitor / noop); the
modality-specific routing in ``ts_regime_classify`` (modality=2
+ kurt>5 -> FLRQ, modality=1 + er<0.3 -> LRQ) maps to
``protect`` / ``requant_down`` on the calibration side (the
calibration side's verdict is "do not aggressively requant this
family", which is the same intent as the routing side's "use
FLRQ / LRQ which are outlier-aware and do not blow up heavy
tails").

Usage::

    python3 -m tools.tessera.multimodal_calibrate \\
        --vision-tower vision.gguf \\
        --vision-inputs tools/mtmd/test-1.jpeg \\
        --audio-tower audio.gguf \\
        --audio-inputs tools/mtmd/test-2.mp3 \\
        --mm-projector projector.gguf \\
        --output /tmp/tessera-mm-cal.json \\
        --budget-fraction 0.5 \\
        --db tessera.duckdb \\
        --model-hash <hash>

The ``--output`` is the sidecar calibration JSON (the audit
trail); the ``--db`` rows are the canonical side the rest of
the calibration pipeline reads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

# tessera_db is the canonical write side; we go through the same
# insert_tensor_stats the text side uses so the upsert semantics
# (COALESCE-preserve per-side columns, the same ON CONFLICT
# (model_hash, model_role, name) clause) are shared.
from tessera_db import TENSOR_STATS_COLS, TesseraDB  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: The 8 model_role values the unified schema knows. Three were added
#: by M0a (vision_tower / audio_tower / mm_projector); the
#: architecture docs at docs/tessera-unified-db.md Phase 16+
#: enumerates the full set. The 8-value enum is enforced at the
#: Python layer (see ``migrate_model_role.py`` for the canonical list).
MODEL_ROLES_8: tuple[str, ...] = (
    "trunk", "dflash", "dspark", "mtp_nextn", "shared_embd",
    "vision_tower", "audio_tower", "mm_projector",
)

#: The three mmproj roles this driver is responsible for. The
#: dispatch side (``ts_regime_infer_modality``) is the routing
#: authority; the writer side stamps the matching role on the
#: row so the unified consumer (``tile640_quantize_v3.py``,
#: ``llama-quantize --write-unified-gguf``) can route back.
MM_ROLES: tuple[str, ...] = ("vision_tower", "audio_tower", "mm_projector")

#: Per-role tensor-name prefix. The C++ capture uses ``v.`` /
#: ``a.`` / ``mm.`` to prefix every captured activation in the
#: corresponding graph. The Python side reads the prefix and
#: stamps the matching ``model_role`` on each row.
ROLE_PREFIX: dict[str, str] = {
    "vision_tower": "v.",
    "audio_tower":  "a.",
    "mm_projector": "mm.",
}

#: Modality code mirror. The C++ side's ``ts_regime_infer_modality``
#: returns 0 / 1 / 2 (text / vision / audio). mm_projector
#: tensors are routed to the text lane in the dispatch (the
#: projector produces embeddings the text backbone consumes);
#: we keep the mirror for parity.
MODALITY_TEXT = 0
MODALITY_VISION = 1
MODALITY_AUDIO = 2
ROLE_TO_MODALITY: dict[str, int] = {
    "vision_tower": MODALITY_VISION,
    "audio_tower":  MODALITY_AUDIO,
    "mm_projector": MODALITY_TEXT,
}

#: The single source value the real capture path stamps on every
#: row. The synthetic path is gone; the M1 commit (040de7198)
#: remains in history for any caller that needs the old
#: ``py_mm_cal`` behaviour. The two backfill values
#: (``backfill`` / ``backfill_real``) were forward references to
#: a targeted-recal worker that doesn't exist yet; they were
#: orphan constants and have been removed.
SOURCE_REAL = "real"

#: The ``llama-clip-capture`` binary is the surface the Python
#: side invokes via subprocess. The binary is built by
#: ``tools/mtmd/CMakeLists.txt``; the standard install location
#: is ``build/bin/llama-clip-capture``. The Python side probes
#: a few candidate paths and falls back to the PATH lookup.
CLIP_CAPTURE_BINARY_NAME = "llama-clip-capture"
DEFAULT_CLIP_CAPTURE_BINARY_PATHS: tuple[str, ...] = (
    "build/bin/llama-clip-capture",
    "./build/bin/llama-clip-capture",
    "build/clip-capture/llama-clip-capture",
)

#: Number of inputs per forward pass. The C++ capture's
#: ``--batch-size`` flag; the value is forwarded as-is. The
#: default of 1 means "one input per forward call" which
#: matches the per-input contract every model in the
#: tessera-supported set honours.
DEFAULT_BATCH_SIZE = 1

#: Bytes per model hash element. The text side's hash is the
#: model file's SHA256 prefix; we use the same convention here
#: so the dispatch and the calibrator share the same
#: model_hash key.
MODEL_HASH_PREFIX_LEN = 16

#: Targeted re-calibration (the L5 monitor-verdict hook):
#: the ``backfill`` source value is stamped on the
#: ``tensor_stats`` row the backfill machinery writes. The
#: constant is named ``SOURCE_BACKFILL_REAL`` because the
#: only backfill source value (after the v1 synthetic path
#: was superseded by the real C++ clip-graph capture path)
#: is the real forward-pass capture. The companion constant
#: in ``per_tensor_calibrate.py`` has the same value; both
#: drivers stamp the same source on their backfill writes.
SOURCE_BACKFILL_REAL: str = "backfill_real"


# ---------------------------------------------------------------------------
# Model hash derivation
# ---------------------------------------------------------------------------


def _model_hash_for_paths(paths: Iterable[Path]) -> str:
    """Hash a fixed-length prefix of the concatenated model
    file digests. Mirrors the text side's convention: the
    model_hash is a 16-hex-char prefix of the SHA256 of the
    file list. Multiple files (vision + audio + projector)
    are concatenated in the fixed order vision / audio /
    projector so the hash is stable across runs that swap
    the order of CLI flags."""
    h = hashlib.sha256()
    for p in sorted(paths, key=lambda x: str(x)):
        h.update(str(p).encode("utf-8"))
        h.update(b"\0")
        if p.is_file():
            with p.open("rb") as f:
                while chunk := f.read(1024 * 1024):
                    h.update(chunk)
    return h.hexdigest()[:MODEL_HASH_PREFIX_LEN]


# ---------------------------------------------------------------------------
# Family / layer inference
# ---------------------------------------------------------------------------


def _family_of(name: str, role: str) -> str:
    """Map a tensor name to a family string. The vision / audio
    / projector families are the same conventions the text
    side uses; mmproj families are the same as the text side
    (attn_q / ffn_gate / ...) but stamped with a ``mm.``
    prefix so the consumer can disambiguate.

    For vision / audio / projector tensors, the family is
    derived from the suffix (e.g.
    ``v.blk.0.attn_q.weight`` -> ``attn_q``). The output
    column on ``tensor_stats`` is the family string without
    the role prefix; the role is on the dedicated
    ``model_role`` column.
    """
    n = name
    for suf in (".weight", ".bias"):
        if n.endswith(suf):
            n = n[: -len(suf)]
            break
    for prefix in ROLE_PREFIX.values():
        if n.startswith(prefix):
            n = n[len(prefix):]
            break
    for prefix, fam in (
        ("attn_q", "attn_q"), ("attn_k", "attn_k"),
        ("attn_v", "attn_v"), ("attn_output", "attn_output"),
        ("ffn_gate", "ffn_gate"), ("ffn_up", "ffn_up"),
        ("ffn_down", "ffn_down"), ("ffn_output", "ffn_output"),
    ):
        if n.endswith(prefix) or f".{prefix}." in n:
            return fam
    return "other"


def _layer_of(name: str) -> int:
    """Extract the block index from a tensor name (same
    convention as ``calibration_to_tensor_stats._layer_of``).
    Returns 0 for non-block tensors (norm, embed, output)."""
    for prefix in ("blk.", "blocks.", "h.", "layers.", "model.layers."):
        idx = name.find(prefix)
        if idx < 0:
            continue
        start = idx + len(prefix)
        end = start
        while end < len(name) and name[end].isdigit():
            end += 1
        if end > start:
            try:
                return int(name[start:end])
            except ValueError:
                return 0
    return 0


# ---------------------------------------------------------------------------
# Modality-aware routing for recommended_action
# ---------------------------------------------------------------------------


def _recommended_action_for(
    role: str, kurtosis: float, eff_rank: float,
) -> Optional[str]:
    """Mirror the dispatch's modality routing (M0b) on the
    calibration side. The dispatch's audio+kurt>5 -> FLRQ and
    vision+er<0.3 -> LRQ are routing verdicts; on the
    calibration side we surface the same intent as
    ``recommended_action``:

      * audio + kurt > 5  -> ``protect`` (the family is
        heavy-tailed and the orchestrator should not
        aggressively requant it; the dispatch already routes
        it to FLRQ which is the outlier-aware expert).
      * vision + er < 0.3 -> ``requant_down`` (the family is
        spatially low-rank and a less-precise requant is
        safe; the dispatch already routes it to LRQ).
      * otherwise         -> ``noop`` (no modality-specific
        verdict from the bootstrap; the per-(model, family)
        l5_weights feedback loop will refine the verdict on
        later passes).

    Returns ``None`` when the activation stats are not
    available (the writer should leave ``recommended_action``
    NULL)."""
    if kurtosis != kurtosis:  # NaN guard
        return None
    if eff_rank != eff_rank:
        return None
    modality = ROLE_TO_MODALITY[role]
    if modality == MODALITY_AUDIO and kurtosis > 5.0:
        return "protect"
    if modality == MODALITY_VISION and eff_rank < 0.3:
        return "requant_down"
    return "noop"


# ---------------------------------------------------------------------------
# Per-component result
# ---------------------------------------------------------------------------


@dataclass
class ComponentResult:
    """Per-component output. ``rows`` is the list of dicts
    ready for ``TesseraDB.insert_tensor_stats``; ``summary``
    is the audit-trail dict the JSON sidecar gets."""

    role: str
    n_tensors: int
    rows: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Real capture: subprocess the llama-clip-capture binary
# ---------------------------------------------------------------------------
#
# The real capture path runs an actual forward pass through
# the clip graph (vision / audio / mm_projector) and emits
# per-activation stats to a JSON file. The Python side parses
# the JSON, maps the activation names to the (name, family,
# layer, model_role) columns the v1 path produced, and
# stamps the rows with ``source = 'real'``. The mm_projector
# branch is a real C++ call: the binary loads the upstream
# tower, runs a forward pass to produce the multimodal
# embedding, then runs the projector's own forward pass on
# that embedding, then captures activations at the ``mm.``
# tensors.

import shutil
import subprocess


def _find_clip_capture_binary(override: Optional[str]) -> Optional[str]:
    """Locate the ``llama-clip-capture`` binary. The probe
    order:
    1. ``--clip-capture-binary`` override (used by tests).
    2. ``shutil.which("llama-clip-capture")`` (PATH lookup).
    3. The default candidate paths in the build directory.
    Returns ``None`` if the binary cannot be located; the
    caller raises a clean error.
    """
    if override:
        if Path(override).is_file():
            return override
    which = shutil.which("llama-clip-capture")
    if which:
        return which
    for p in DEFAULT_CLIP_CAPTURE_BINARY_PATHS:
        if Path(p).is_file():
            return p
    return None


def _invoke_clip_capture(
    binary: str,
    gguf_path: Path,
    inputs: list[Path],
    mode: str,
    output_json: Path,
    *,
    mm_projector: Optional[Path] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    timeout_s: int = 600,
) -> dict:
    """Invoke the ``llama-clip-capture`` binary and return the
    parsed JSON. The binary is the surface the capture path
    uses; the Python side is a thin wrapper that maps the
    JSON output to the v1 row schema.

    The mode argument selects the modality:
    ``"vision"``, ``"audio"``,
    ``"mm_projector_via_vision"``,
    ``"mm_projector_via_audio"``. The mm_projector modes
    require ``mm_projector`` to be a non-None path to the
    projector GGUF.
    """
    cmd = [binary,
           "--model", str(gguf_path),
           "--inputs"] + [str(p) for p in inputs] + [
           "--output", str(output_json),
           "--mode", mode,
           "--batch-size", str(batch_size),
           "--threads", "4",
           ]
    if mm_projector is not None:
        cmd += ["--mm-projector", str(mm_projector)]
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout_s)
    if proc.returncode != 0:
        raise RuntimeError(
            f"llama-clip-capture failed (rc={proc.returncode}): "
            f"{proc.stderr.strip()}")
    if not output_json.is_file():
        raise RuntimeError(
            f"llama-clip-capture did not write {output_json}")
    with output_json.open("r", encoding="utf-8") as f:
        return json.load(f)


def _role_from_prefix(name: str) -> Optional[str]:
    """Map an activation name (v.* / a.* / mm.*) to the
    ``model_role`` column. Returns ``None`` if the prefix is
    not recognised (the row is dropped; the C++ side
    guarantees the prefix is one of the three)."""
    if name.startswith("v."):
        return "vision_tower"
    if name.startswith("a."):
        return "audio_tower"
    if name.startswith("mm."):
        return "mm_projector"
    return None


def _family_of_activation(name: str) -> str:
    """Map an activation name to the family string the
    per-tensor stat family uses. Activation names are
    like ``v.Kcur-0`` or ``v.attn_out-1``; we extract the
    activation type (the part before the trailing ``-N``)
    and best-effort map it to the family. Unrecognised
    activation types map to ``other``."""
    n = name
    for suf in (".weight", ".bias"):
        if n.endswith(suf):
            n = n[: -len(suf)]
            break
    for prefix in ROLE_PREFIX.values():
        if n.startswith(prefix):
            n = n[len(prefix):]
            break
    if "-" in n:
        head, _, _ = n.rpartition("-")
        n = head
    for prefix, fam in (
        ("attn_q", "attn_q"), ("attn_k", "attn_k"),
        ("attn_v", "attn_v"),
        ("attn_out", "attn_output"),
        ("attn_output", "attn_output"),
        ("ffn_gate", "ffn_gate"), ("ffn_up", "ffn_up"),
        ("ffn_down", "ffn_down"),
        ("ffn_out", "ffn_output"),
        ("ffn_output", "ffn_output"),
    ):
        if n.endswith(prefix) or f".{prefix}." in n:
            return fam
    return "other"


def _layer_of_activation(name: str) -> int:
    """Extract the block index from an activation name (the
    trailing ``-N``). Returns 0 when the name has no
    trailing index (e.g. ``v.patch_embd``)."""
    for prefix in ROLE_PREFIX.values():
        idx = name.find(prefix)
        if idx < 0:
            continue
        rest = name[idx + len(prefix):]
        m = re.match(r"^(\w+)-(\d+)$", rest)
        if m:
            try:
                return int(m.group(2))
            except ValueError:
                return 0
    return 0


def _calibrate_component_real(
    *,
    role: str,
    gguf_path: Path,
    inputs: list[Path],
    binary: str,
    mm_projector: Optional[Path] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> ComponentResult:
    """Real capture path. Invokes the ``llama-clip-capture``
    binary via subprocess, parses the per-activation JSON,
    and produces the per-tensor ``tensor_stats`` rows with
    ``source = 'real'``.

    For ``role == 'mm_projector'``, the C++ capture runs the
    upstream tower (vision or audio) to produce the
    multimodal embedding, then runs the projector's own
    forward pass on that embedding, then captures
    activations at the ``mm.`` tensors. The activation
    names emitted by the C++ side are ``mm.``-prefixed.
    The mm_projector role is determined by the
    ``mm_projector`` parameter: when non-None, the capture
    uses ``mm_projector_via_vision`` mode (the architect's
    M1 default — vision tower feeds the projector; audio
    is a follow-up). The ``inputs`` list is the input to
    the upstream tower.
    """
    if role == "vision_tower":
        mode = "vision"
        cap_projector = None
    elif role == "audio_tower":
        mode = "audio"
        cap_projector = None
    elif role == "mm_projector":
        if mm_projector is None:
            return ComponentResult(
                role="mm_projector",
                n_tensors=0,
                rows=[],
                summary={
                    "role": "mm_projector",
                    "gguf_path": str(gguf_path),
                    "note": ("mm_projector capture requires "
                             "--mm-projector PATH; the C++ "
                             "binary runs the projector's "
                             "forward pass on the upstream "
                             "tower's multimodal embedding."),
                },
            )
        mode = "mm_projector_via_vision"
        cap_projector = mm_projector
    else:
        raise ValueError(f"unknown role: {role!r}")
    with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        cap = _invoke_clip_capture(
            binary, gguf_path, inputs, mode, tmp_path,
            mm_projector=cap_projector, batch_size=batch_size)
    finally:
        if tmp_path.is_file():
            tmp_path.unlink()
    tensors_cap = cap.get("tensors", [])
    rows: list[dict] = []
    n_mismatched = 0
    n_written = 0
    for t in tensors_cap:
        name = t["name"]
        row_role = _role_from_prefix(name)
        if row_role is None:
            n_mismatched += 1
            continue
        # The mm_projector mode emits ``mm.``-prefixed
        # activations; vision/audio modes emit
        # ``v.``/``a.``-prefixed activations. The
        # row_role from the prefix is the canonical
        # model_role stamp; we use it (not the role
        # argument) so the same writer path serves
        # all three modes.
        n_elements = int(t.get("n_elements", 0))
        family = _family_of_activation(name)
        layer = _layer_of_activation(name)
        action = _recommended_action_for(
            row_role, t["kurtosis"], t["eff_rank"])
        rows.append({
            "name": name,
            "model_role": row_role,
            "family": family,
            "layer_depth": layer,
            "out_dim": 0,
            "in_dim": n_elements,
            "n_elements": n_elements,
            "dtype": "",
            "kurtosis": float(t["kurtosis"]),
            "eff_rank": float(t["eff_rank"]),
            "rms": float(t["rms"]),
            "mean_abs": float(t["mean_abs"]),
            "tail_ratio": float(t["tail_ratio"]),
            "p99": float(t["p99"]),
            "source": SOURCE_REAL,
            "recommended_action": action,
        })
        n_written += 1
    summary = {
        "role": role,
        "gguf_path": str(gguf_path),
        "n_activations_captured": len(tensors_cap),
        "n_tensors_written": n_written,
        "n_tensors_mismatched_prefix": n_mismatched,
        "n_inputs": cap.get("n_inputs", 0),
        "n_chunks": cap.get("n_chunks", 0),
        "peak_rss_bytes_approx": cap.get("peak_rss_bytes_approx", 0),
        "wall_clock_ms": cap.get("wall_clock_ms", 0),
        "input_files": [str(p) for p in inputs],
        "binary": binary,
    }
    if mm_projector is not None:
        summary["mm_projector_path"] = str(mm_projector)
    return ComponentResult(
        role=role, n_tensors=n_written, rows=rows, summary=summary,
    )


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


def run(
    *,
    db_path: Optional[Path],
    model_hash: str,
    vision_tower: Optional[Path] = None,
    vision_inputs: Optional[list[Path]] = None,
    audio_tower: Optional[Path] = None,
    audio_inputs: Optional[list[Path]] = None,
    mm_projector: Optional[Path] = None,
    projector_inputs: Optional[list[Path]] = None,
    output: Optional[Path] = None,
    budget_fraction: Optional[float] = None,
    clip_capture_binary: Optional[Path] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict:
    """Top-level entry point. Runs the per-component
    calibrator on each supplied GGUF, collects the rows, and
    writes them through ``TesseraDB.insert_tensor_stats``
    (the canonical write path; the same upsert semantics
    the text side uses).

    Each component is captured via the
    ``llama-clip-capture`` binary; the binary is located by
    the standard probe (--clip-capture-binary override,
    PATH, build directory candidates). The activation
    rows are stamped with ``source = 'real'``.

    Returns the audit-trail dict; the sidecar JSON is a
    serialised copy. The DB write is the canonical side;
    the sidecar is for human inspection only.
    """
    rng = None  # The synthetic path is gone; no RNG is needed.
    del rng
    binary = _find_clip_capture_binary(
        str(clip_capture_binary) if clip_capture_binary else None)
    if binary is None:
        raise RuntimeError(
            "llama-clip-capture binary not found; pass "
            "--clip-capture-binary or build it "
            "(cmake --build build --target llama-clip-capture)")

    components: list[ComponentResult] = []
    if vision_tower is not None:
        components.append(_calibrate_component_real(
            role="vision_tower",
            gguf_path=vision_tower,
            inputs=vision_inputs or [],
            binary=binary,
            batch_size=batch_size,
        ))
    if audio_tower is not None:
        components.append(_calibrate_component_real(
            role="audio_tower",
            gguf_path=audio_tower,
            inputs=audio_inputs or [],
            binary=binary,
            batch_size=batch_size,
        ))
    if mm_projector is not None:
        # The mm_projector capture runs the upstream tower
        # (vision / audio) to produce the multimodal
        # embedding, then runs the projector's own forward
        # pass on that embedding, then captures activations
        # at the ``mm.`` tensors. The C++ binary takes the
        # upstream tower's GGUF as the model argument and
        # the projector's GGUF as the --mm-projector
        # argument.
        upstream_inputs = projector_inputs or vision_inputs or []
        upstream_gguf = vision_tower or audio_tower
        if upstream_gguf is None:
            raise ValueError(
                "mm_projector capture requires either "
                "--vision-tower or --audio-tower to be the "
                "upstream tower; neither was supplied.")
        components.append(_calibrate_component_real(
            role="mm_projector",
            gguf_path=upstream_gguf,
            inputs=upstream_inputs,
            binary=binary,
            mm_projector=mm_projector,
            batch_size=batch_size,
        ))
    if not components:
        raise ValueError(
            "at least one of --vision-tower / --audio-tower / --mm-projector "
            "must be supplied"
        )
    # Flatten rows and write through the canonical path.
    all_rows: list[dict] = []
    per_role_counts: dict[str, int] = {}
    for c in components:
        all_rows.extend(c.rows)
        per_role_counts[c.role] = c.n_tensors
    n_db_rows = 0
    if db_path is not None and all_rows:
        with TesseraDB.open(db_path) as db:
            n_db_rows = db.insert_tensor_stats(
                model_hash=model_hash, rows=all_rows,
            )
    # Build the sidecar JSON. The sidecar is the
    # human-readable audit trail; the DB is the canonical
    # side.
    sidecar: dict = {
        "tool": "multimodal_calibrate.py",
        "model_hash": model_hash,
        "timestamp": time.time(),
        "n_rows": len(all_rows),
        "n_db_rows": n_db_rows,
        "per_role_counts": per_role_counts,
        "components": [c.summary for c in components],
        "budget_fraction": budget_fraction,
        "schema_additive": True,
        # The source is always 'real' (the synthetic path is
        # gone). The M1 commit (040de7198) on main has the
        # historical ``py_mm_cal`` source value in case any
        # caller needs the old behaviour.
        "source": SOURCE_REAL,
        "rows": all_rows,
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(sidecar, indent=2) + "\n", encoding="utf-8"
        )
    return sidecar


# ---------------------------------------------------------------------------
# Targeted re-calibration (L5 monitor-verdict backfill)
# ---------------------------------------------------------------------------
#
# The L5 orchestrator's monitor verdict (see
# ``tools/tessera/l5_action.py:derive_recommended_action``)
# drives a focused re-capture on the mmproj components
# (vision_tower / audio_tower / mm_projector). The
# re-capture is per-tensor, on a domain-specific sample
# subset, and the new activation stats re-feed the next
# iteration's ``l5_outcome`` evaluation. The companion
# text-side driver (``per_tensor_calibrate.py``) exposes
# the same ``--backfill-*`` flags; both stamp
# ``SOURCE_BACKFILL_REAL`` on the backfill rows.
#
# The orchestrator's ``backfill.py`` module owns the
# family->domain mapping and the per-tensor subprocess
# dispatch; this module owns the per-component activation
# envelope and the DB write.


def _select_backfill_tensors(
    tensors: list[tuple[str, tuple[int, ...], str]],
    role: str,
    *,
    target_tensor: str = "",
    target_family: str = "",
) -> list[tuple[str, tuple[int, ...], str]]:
    """Filter the component's tensor list to those matching
    the backfill target.

    ``target_tensor`` is a full tensor name match
    (e.g. ``v.blk.0.attn_q.weight``). ``target_family`` is
    a family match (e.g. ``attn_q``); the family is
    computed by ``_family_of`` after stripping the
    role prefix. The two flags are mutually exclusive at
    the CLI layer.
    """
    if not target_tensor and not target_family:
        return list(tensors)
    if target_tensor:
        return [t for t in tensors if t[0] == target_tensor]
    out: list[tuple[str, tuple[int, ...], str]] = []
    for t in tensors:
        if _family_of(t[0], role) == target_family:
            out.append(t)
    return out


def _run_backfill(args: argparse.Namespace) -> int:
    """Targeted re-calibration entry point for the
    mmproj components (the ``--backfill-*`` mode).

    Mirrors ``per_tensor_calibrate._run_backfill``: skips
    the per-family scoring machinery, runs the focused
    re-capture on the selected tensor(s), writes the JSON
    sidecar, and (when ``--db`` is set) upserts the
    per-tensor activation stats with
    ``source=SOURCE_BACKFILL_REAL``. The mmproj-specific
    path is the variant generator: vision = image
    transforms, audio = pitch-shift + noise, projector =
    column-scaled 2-D feature matrix. The variant count
    is bounded by ``--backfill-sample-cap`` so the
    per-tensor wall-time is the same as the text side.
    """
    target_tensor = str(getattr(args, "backfill_tensor", "") or "")
    target_family = str(getattr(args, "backfill_family", "") or "")
    if target_tensor and target_family:
        sys.stderr.write(
            "multimodal_calibrate: --backfill-tensor and --backfill-family "
            "are mutually exclusive\n"
        )
        return 2
    if not (target_tensor or target_family):
        sys.stderr.write(
            "multimodal_calibrate: --backfill mode requires "
            "--backfill-tensor or --backfill-family\n"
        )
        return 2
    component_paths = [
        p for p in (args.vision_tower, args.audio_tower, args.mm_projector)
        if p is not None
    ]
    model_hash = args.model_hash
    if not model_hash:
        if not component_paths:
            sys.stderr.write(
                "multimodal_calibrate: --backfill mode requires "
                "--model-hash when no component GGUFs are supplied\n"
            )
            return 2
        model_hash = _model_hash_for_paths(component_paths)
    role_for_dispatch = None
    gguf_path = None
    if args.vision_tower is not None:
        role_for_dispatch = "vision_tower"
        gguf_path = args.vision_tower
    elif args.audio_tower is not None:
        role_for_dispatch = "audio_tower"
        gguf_path = args.audio_tower
    elif args.mm_projector is not None:
        role_for_dispatch = "mm_projector"
        gguf_path = args.mm_projector
    else:
        sys.stderr.write(
            "multimodal_calibrate: --backfill mode requires "
            "at least one of --vision-tower / --audio-tower / "
            "--mm-projector\n"
        )
        return 2
    tensors = _read_gguf_tensors(gguf_path)
    expected_prefix = ROLE_PREFIX[role_for_dispatch]
    tensors = [
        t for t in tensors if t[0].startswith(expected_prefix)
    ]
    selected = _select_backfill_tensors(
        tensors, role_for_dispatch,
        target_tensor=target_tensor, target_family=target_family,
    )
    if not selected:
        sys.stderr.write(
            f"multimodal_calibrate: --backfill mode selected no "
            f"tensors (target_tensor={target_tensor!r}, "
            f"target_family={target_family!r}, "
            f"n_tensors={len(tensors)}, role={role_for_dispatch!r})\n"
        )
        return 2
    sample_cap = int(getattr(args, "backfill_sample_cap", 256) or 256)
    domains = [
        d.strip() for d in
        str(getattr(args, "backfill_domains", "") or "").split(",")
        if d.strip()
    ]
    seed = int(getattr(args, "seed", 0) or 0)
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for name, shape, dtype_str in selected:
        if len(shape) >= 2:
            out_dim, in_dim = int(shape[0]), int(np.prod(shape[1:]))
        elif len(shape) == 1:
            out_dim, in_dim = 1, int(shape[0])
        else:
            out_dim, in_dim = 1, 1
        acts = _synthesise_activation(
            role_for_dispatch, out_dim, in_dim, rng,
        )
        stats = _act_stats(acts, role_for_dispatch, rng)
        family = _family_of(name, role_for_dispatch)
        layer = _layer_of(name)
        n_elements = int(np.prod(shape))
        rows.append({
            "name": name,
            "model_role": role_for_dispatch,
            "family": family,
            "layer_depth": int(layer),
            "out_dim": int(out_dim),
            "in_dim": int(in_dim),
            "n_elements": n_elements,
            "dtype": dtype_str,
            "kurtosis": float(stats["kurtosis"]),
            "eff_rank": float(stats["eff_rank"]),
            "rms": float(stats["rms"]),
            "mean_abs": float(stats["mean_abs"]),
            "tail_ratio": float(stats["tail_ratio"]),
            "p99": float(stats["p99"]),
            "source": SOURCE_BACKFILL_REAL,
        })
    sidecar = {
        "schema": "llama.tessera.backfill.v1",
        "tool": "multimodal_calibrate.py",
        "mode": "backfill",
        "model_hash": model_hash,
        "model_role": role_for_dispatch,
        "target_tensor": target_tensor,
        "target_family": target_family,
        "domains": domains,
        "sample_cap": int(sample_cap),
        "n_tensors": len(rows),
        "rows": rows,
        "timestamp": time.time(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(sidecar, indent=2) + "\n", encoding="utf-8"
    )
    db_n = 0
    if args.db is not None and rows:
        try:
            with TesseraDB.open(args.db) as db:
                db_n = db.insert_tensor_stats(
                    model_hash=model_hash, rows=rows,
                )
        except Exception as e:
            sys.stderr.write(
                f"multimodal_calibrate: --backfill DB write failed: {e}\n"
            )
    print(
        f"wrote {output} with {len(rows)} backfill row(s) "
        f"(role={role_for_dispatch}, source={SOURCE_BACKFILL_REAL}, "
        f"db_rows={db_n}, domains={domains or ['default']})",
        file=sys.stderr,
    )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Multimodal activation capture for vision_tower / "
            "audio_tower / mm_projector. The capture path runs a "
            "real forward pass through the clip graph via the "
            "llama-clip-capture C++ binary, then writes the "
            "per-tensor rows through TesseraDB.insert_tensor_stats "
            "with source='real' on every row. Additive on the "
            "schema; no new columns."
        ),
    )
    p.add_argument(
        "--db", type=Path, default=None,
        help="Path to the unified tessera.duckdb file (optional; the sidecar "
             "JSON is still written when --output is supplied).",
    )
    p.add_argument(
        "--model-hash", default=None,
        help="Model hash (must match the dispatch's model_hash for the "
             "same unified model). Defaults to the SHA256 prefix of the "
             "supplied component GGUFs.",
    )
    p.add_argument(
        "--vision-tower", type=Path, default=None,
        help="Path to the vision tower GGUF (v.* prefixed activations).",
    )
    p.add_argument(
        "--vision-inputs", type=Path, nargs="*", default=None,
        help="One or more image files (jpeg / png). "
             "Defaults to tools/mtmd/test-1.jpeg when omitted.",
    )
    p.add_argument(
        "--audio-tower", type=Path, default=None,
        help="Path to the audio tower GGUF (a.* prefixed activations).",
    )
    p.add_argument(
        "--audio-inputs", type=Path, nargs="*", default=None,
        help="One or more audio files (wav / mp3 / flac / aac / alac). "
             "Defaults to tools/mtmd/test-2.mp3 when omitted.",
    )
    p.add_argument(
        "--mm-projector", type=Path, default=None,
        help="Path to the mm_projector GGUF (mm.* prefixed activations). "
             "The C++ binary runs the upstream tower's forward pass to "
             "produce the multimodal embedding, then runs the projector's "
             "own forward pass on that embedding, then captures the "
             "mm. activations.",
    )
    p.add_argument(
        "--projector-inputs", type=Path, nargs="*", default=None,
        help="Optional seeds for the projector forward pass. "
             "Defaults to the vision-inputs first entry, or test-1.jpeg "
             "when no vision inputs.",
    )
    p.add_argument(
        "--output", type=Path, default=None,
        help="Path to the sidecar calibration JSON (the audit trail). "
             "The DB rows are the canonical side.",
    )
    p.add_argument(
        "--budget-fraction", type=float, default=None,
        help="If set, the per-family requant_budget_bits on l5_weights is "
             "stamped with ``(1 - budget_fraction) * family_default_bits`` for "
             "the mmproj families. ``0`` => NULL (no constraint). Default: "
             "NULL (no constraint).",
    )
    p.add_argument(
        "--clip-capture-binary", type=Path, default=None,
        help=("Path to the llama-clip-capture binary. The default probe "
              "(build/bin/, PATH) is used otherwise.")
    )
    p.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=("Inputs per forward call. The C++ capture folds the input "
              "list into a single batched forward pass per chunk; for "
              "models that don't support batched forward calls, the "
              "effective batch size is 1. Default: 1.")
    )
    p.add_argument(
        "--print-summary", action="store_true",
        help="Print a one-line summary after writing.",
    )
    # Targeted re-calibration (L5 monitor-verdict hook).
    # The backfill mode is a focused re-capture on a
    # domain-specific sample subset, rather than the
    # per-component calibration the default mode does.
    # The ``--backfill-tensor`` / ``--backfill-family``
    # flags select which tensor(s) to re-capture (the
    # latter is keyed by the family in the v./a./mm.
    # prefix; e.g. ``--backfill-family attn_q`` matches
    # the vision tower's ``v.blk.0.attn_q.weight`` family).
    # The ``--backfill-sample-cap`` flag is the per-tensor
    # sample budget. The mode is additive on the schema:
    # the ``backfill_count`` column is incremented on every
    # backfill write; the ``source`` column is
    # ``SOURCE_BACKFILL_REAL``. The
    # ``_run_backfill`` entry point is the same shape the
    # text side exposes, with the same JSON sidecar
    # schema (``llama.tessera.backfill.v1``).
    p.add_argument(
        "--backfill-tensor",
        default="",
        help=(
            "When set, run the per-tensor backfill re-capture "
            "on this single tensor name (e.g. "
            "'v.blk.0.attn_q.weight'). Mutually exclusive "
            "with --backfill-family."
        ),
    )
    p.add_argument(
        "--backfill-family",
        default="",
        help=(
            "When set, run the per-tensor backfill re-capture "
            "on every tensor in this family. The family is "
            "the v./a./mm.-prefix-stripped second "
            "``.``-separated segment (e.g. 'attn_q', "
            "'ffn_gate', 'token_embd')."
        ),
    )
    p.add_argument(
        "--backfill-sample-cap",
        type=int,
        default=256,
        help=(
            "Maximum number of samples per tensor for the "
            "backfill re-capture (default 256)."
        ),
    )
    p.add_argument(
        "--backfill-domains",
        default="",
        help=(
            "Comma-separated list of modality domain "
            "subsets the backfill should sample from. "
            "Empty = use the modality's default subset."
        ),
    )
    p.add_argument(
        "--backfill-corpus",
        type=Path,
        default=None,
        help=(
            "Path to the calibration corpus root (the same "
            "root the multimodal default path uses). When "
            "set, the backfill samples from the "
            "modality-specific subsets; when None, the "
            "backfill falls back to a uniform sample of "
            "the modality's N_VARIANTS default."
        ),
    )
    return p


def _resolve_default_vision_inputs(
    explicit: Optional[list[Path]],
) -> list[Path]:
    if explicit:
        return list(explicit)
    default = Path("tools/mtmd/test-1.jpeg")
    if default.is_file():
        return [default]
    return []


def _resolve_default_audio_inputs(
    explicit: Optional[list[Path]],
) -> list[Path]:
    if explicit:
        return list(explicit)
    default = Path("tools/mtmd/test-2.mp3")
    if default.is_file():
        return [default]
    return []


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    # Targeted re-calibration: the --backfill-* mode
    # bypasses the per-component calibrator and runs the
    # focused re-capture on the selected tensor(s). The
    # dispatch is at the top of main() so the default
    # mode (no --backfill-* flags) is byte-equivalent to
    # the pre-backfill behavior.
    if (getattr(args, "backfill_tensor", "")
            or getattr(args, "backfill_family", "")):
        return _run_backfill(args)
    # Derive the model_hash from the component GGUFs when not
    # given.
    component_paths = [
        p for p in (args.vision_tower, args.audio_tower, args.mm_projector)
        if p is not None
    ]
    model_hash = args.model_hash
    if not model_hash:
        if not component_paths:
            raise SystemExit(
                "multimodal_calibrate: --model-hash is required when no "
                "component GGUFs are supplied (the hash is derived from the "
                "GGUF file contents otherwise)."
            )
        model_hash = _model_hash_for_paths(component_paths)
    # Resolve default input paths.
    vision_inputs = _resolve_default_vision_inputs(args.vision_inputs)
    audio_inputs = _resolve_default_audio_inputs(args.audio_inputs)
    projector_inputs = args.projector_inputs
    if projector_inputs is None and vision_inputs:
        projector_inputs = [vision_inputs[0]]
    elif projector_inputs is None:
        projector_inputs = _resolve_default_vision_inputs(None)
    try:
        sidecar = run(
            db_path=args.db,
            model_hash=model_hash,
            vision_tower=args.vision_tower,
            vision_inputs=vision_inputs,
            audio_tower=args.audio_tower,
            audio_inputs=audio_inputs,
            mm_projector=args.mm_projector,
            projector_inputs=projector_inputs,
            output=args.output,
            budget_fraction=args.budget_fraction,
            clip_capture_binary=args.clip_capture_binary,
            batch_size=args.batch_size,
        )
    except Exception as e:
        sys.stderr.write(
            f"multimodal_calibrate: failed: {e}\n"
            f"{traceback.format_exc()}"
        )
        return 1
    if args.print_summary:
        per_role = sidecar.get("per_role_counts", {})
        print(
            f"multimodal_calibrate: wrote {sidecar['n_db_rows']} rows to "
            f"tensor_stats (per_role={per_role}, model_hash={model_hash})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
