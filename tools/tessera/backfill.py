"""Targeted re-calibration: focused re-capture for L5 monitor-verdict tensors.

When the L5 orchestrator's per-(model, family) feedback loop
classifies a tensor's ``recommended_action`` as ``monitor`` (the
"calibrated for this family; just keep watching" verdict from
``tools/tessera/l5_action.py:derive_recommended_action``), the
orchestrator has a choice of three next-step strategies:

  1. do nothing and rely on the next per-iteration plan
     (the legacy default; the family may stay ``monitor``
     for many iterations and the L5 auto-converge signals
     will eventually trip);
  2. a global 4x sample bump on the next iteration
     (touches every tensor in every family; the
     sidecar JSON's per-family sample count goes up
     uniformly; the monitor verdict is rarely resolved);
  3. the focused re-capture implemented in this module
     (a per-tensor re-capture on a domain-specific sample
     subset, with the new activation stats re-feeding
     the next iteration's ``l5_outcome`` evaluation).

The focused re-capture is cheaper than the global 4x bump
(it only touches the monitor-verdict tensors) and the
domain-specific subset is more diagnostic (the bump samples
uniformly, the backfill samples from the domain the
miscalibrated tensor is most sensitive to). The full
operator-facing spec lives in
``docs/tessera-targeted-recalibration.md`` (the architect's
brief); this module is the implementation.

The module is a thin orchestrator over the two calibration
drivers:

  * ``tools/tessera/per_tensor_calibrate.py`` for the
    text-side / dflash / dspark / mtp_nextn / shared_embd
    roles (the ``--backfill-*`` mode in that driver).
  * ``tools/tessera/multimodal_calibrate.py`` for the
    vision_tower / audio_tower / mm_projector roles (the
    ``--backfill-*`` mode in that driver).

The orchestrator owns the family->domain mapping (a
37-entry table below), the per-tensor subprocess dispatch
(memory isolation between captures), the async dispatch
(``concurrent.futures.ThreadPoolExecutor`` with
``max_workers=2``), and the sidecar writer. The two
calibration drivers own the per-tensor activation
envelope, the JSON sidecar format, and the DB write.

EVOLVE, DON'T VERSION: the orchestrator's hook in
``l5_orchestrator.py`` is the only place the targeted
re-calibration decision is made. The M1 v1 synthetic
forward pass path in ``multimodal_calibrate.py`` is being
deleted by the in-flight clip-capture-v2 worker; this
module only references the v2 path's API surface
(``SOURCE_BACKFILL_REAL`` constant, ``--backfill-*`` flags,
``llama.tessera.backfill.v1`` JSON schema). The orphan
``SOURCE_BACKFILL`` and ``SOURCE_BACKFILL_REAL`` constants
the prior worker added to the M1 drivers are dropped by
the clip-capture worker; this module does not reference
them.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Mapping, Sequence

# tessera_db is the canonical write side; the backfill
# machinery upserts the per-tensor activation stats
# through ``TesseraDB.insert_tensor_stats`` (the same
# upsert path the text side uses). The backfill is also
# idempotent at the SQL level: re-running the same
# backfill pass increments ``backfill_count`` rather than
# duplicating rows.
try:
    from .tessera_db import TesseraDB
    from .tessera_db_buffer import sql_escape
    from .per_tensor_calibrate import SOURCE_BACKFILL_REAL as _TEXT_BACKFILL_SOURCE
    from .multimodal_calibrate import SOURCE_BACKFILL_REAL as _MM_BACKFILL_SOURCE
except ImportError:  # pragma: no cover - script-mode import
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from tessera_db import TesseraDB  # type: ignore[no-redef]
    from tessera_db_buffer import sql_escape  # type: ignore[no-redef]
    from per_tensor_calibrate import (  # type: ignore[no-redef]
        SOURCE_BACKFILL_REAL as _TEXT_BACKFILL_SOURCE,
    )
    from multimodal_calibrate import (  # type: ignore[no-redef]
        SOURCE_BACKFILL_REAL as _MM_BACKFILL_SOURCE,
    )


#: The single source value the backfill machinery writes
#: on the ``source`` column of ``tensor_stats``. Both
#: drivers (``per_tensor_calibrate.py`` /
#: ``multimodal_calibrate.py``) define the same constant
#: with the same value; this module asserts the equality
#: at import time so a future divergence is caught
#: before a write lands. The clip-capture v2 worker
#: re-introduces this constant in both drivers (the v1
#: synthetic path that carried the orphan ``SOURCE_BACKFILL``
#: / ``SOURCE_BACKFILL_REAL`` declarations is being
#: deleted). After v2 lands, this is the only backfill
#: source value.
SOURCE_BACKFILL_REAL: str = _TEXT_BACKFILL_SOURCE
if _MM_BACKFILL_SOURCE != _TEXT_BACKFILL_SOURCE:
    raise ImportError(
        f"backfill: per_tensor_calibrate.SOURCE_BACKFILL_REAL="
        f"{_TEXT_BACKFILL_SOURCE!r} but multimodal_calibrate."
        f"SOURCE_BACKFILL_REAL={_MM_BACKFILL_SOURCE!r}; the two "
        f"drivers must agree on the backfill source value."
    )


#: Sidecar JSON schema. The orchestrator writes one
#: ``llama.tessera.backfill.v1`` per call; the per-tensor
#: rows inside the sidecar are the per-tensor activation
#: envelope (the same column set the canonical
#: ``tensor_stats`` table carries). The ``backfill_count``
#: column is NOT in the sidecar (it lives in the DB only;
#: the sidecar is the per-call audit trail).
SIDECAR_SCHEMA: str = "llama.tessera.backfill.v1"

#: The 8 model_role values the unified schema knows. The
#: backfill dispatch routes by role: the text-side roles
#: (trunk / dflash / dspark / mtp_nextn / shared_embd)
#: use ``per_tensor_calibrate.py``; the mmproj roles
#: (vision_tower / audio_tower / mm_projector) use
#: ``multimodal_calibrate.py``. The dispatch is a simple
#: membership test against ``MM_ROLES``.
MODEL_ROLES: tuple[str, ...] = (
    "trunk", "dflash", "dspark", "mtp_nextn", "shared_embd",
    "vision_tower", "audio_tower", "mm_projector",
)
MM_ROLES: frozenset[str] = frozenset(
    {"vision_tower", "audio_tower", "mm_projector"},
)
TEXT_ROLES: frozenset[str] = frozenset(
    {"trunk", "dflash", "dspark", "mtp_nextn", "shared_embd"},
)

#: The number of concurrent in-flight backfill captures
#: in the orchestrator's ThreadPoolExecutor. The number
#: is small (2) because the per-tensor captures are
#: subprocess-isolated for memory reasons; the
#: ``concurrent.futures`` pool is just an async dispatch
#: layer, not a parallel-execution layer. 2 is the
#: smallest pool size that lets a slow backfill capture
## run alongside the next tensor's setup work without
#: blocking the orchestrator's iteration loop.
DEFAULT_MAX_WORKERS: int = 2

#: The default per-tensor sample cap. The orchestrator
#: exposes this as ``--backfill-sample-cap``; the default
#: is 256, the same default the per-tensor drivers use.
DEFAULT_SAMPLE_CAP: int = 256

#: The default number of backfill rounds per monitor-verdict
#: tensor. The orchestrator exposes this as
#: ``--max-backfill-rounds``; the default is 2, which is
#: the architect's spec (the 2 rounds cover the
#: "near-monitor" -> "monitor" -> "monitor-no-progress"
#: transition the retune's 3-tier lookup expects).
DEFAULT_MAX_BACKFILL_ROUNDS: int = 2


# ---------------------------------------------------------------------------
# Family -> domain mapping (the 37-entry table)
# ---------------------------------------------------------------------------
#
# The mapping is the heart of the focused re-capture. The
# L5 monitor verdict fires when the orchestrator's
# feedback loop classifies a (model, family) as
# "calibrated; just keep watching" -- the per-family
# miscalibration score is below -0.2 (the
# ``MONITOR_MISCALIBRATION_MAX`` threshold) and the
# plan-accepted signal is positive (no sign the plan
# hurt). The focused re-capture is then a
# domain-specific sample of the family: the family's
# distribution of attention / FFN / output / etc. is
# most diagnostic on a subset of the input that
# exercises the family's particular tensor shape.
#
# The 37 entries cover the architectural roles the
# unified schema knows: trunk (8), dflash (3), dspark
# (3), mtp_nextn (3), shared_embd (2), vision_tower (5),
# audio_tower (4), mm_projector (9). The text-side
# family->domain mapping is the corpus's per-domain
# name (code / math / wiki / science); the mmproj
# family->domain mapping is the modality's per-component
# domain (image / audio / token).
#
# The mapping is intentionally authored (not inferred):
# each entry documents the rationale in the comment so
# the next operator can audit / extend it without
# running an offline inference. The rationale field is
# informational; the keys are the canonical
# ``(model_role, family)`` tuples the orchestrator's
# per-(model, family) lookup keys on.
#
# A family without a per-(model_role, family) entry
# falls back to the wildcard ``"*"`` mapping (the
# ``FAMILY_DOMAIN_MAPPING_FALLBACK`` constant). The
# wildcard is the same for all roles: a uniform
# sample of the corpus's full set, with no domain
# bias. The fallback is the contract the prior worker
# documented: "do not invent a generic inference".

FAMILY_DOMAIN_MAPPING: dict[tuple[str, str], list[str]] = {
    # ---- trunk (8) ------------------------------------------------
    # The trunk's attention Q projection is most diagnostic
    # on the "math" + "code" subset: attention heads fire
    # on token transitions, and the math/code subset has
    # the highest per-token symbol entropy.
    ("trunk", "attn_q"):  ["math", "code"],
    # K is the key projection: its distribution is most
    # diagnostic on the "wiki" subset (long-context
    # repetition; K sees the same tokens as Q but on a
    # different head).
    ("trunk", "attn_k"):  ["wiki"],
    # V is the value projection: the "science" subset has
    # the highest per-token value variance.
    ("trunk", "attn_v"):  ["science"],
    # attn_output is the post-attention projection: the
    # "code" subset has the highest attention-output
    # activation envelope (the residual stream after
    # attention sees the most variance in code).
    ("trunk", "attn_output"): ["code"],
    # FFN gate is the gating projection: the "math" subset
    # exercises the most distinct gate patterns (large
    # negative / positive splits).
    ("trunk", "ffn_gate"): ["math"],
    # FFN up is the up-projection: the "science" subset
    # has the highest per-token activation envelope.
    ("trunk", "ffn_up"):   ["science"],
    # FFN down is the down-projection: the "code" subset
    # has the highest per-output variance.
    ("trunk", "ffn_down"): ["code"],
    # token_embd is the embedding lookup: the "wiki" +
    # "science" subset has the broadest vocabulary
    # coverage.
    ("trunk", "token_embd"): ["wiki", "science"],

    # ---- dflash (3) ----------------------------------------------
    # The dflash encoder's attn_q mirrors the trunk's; the
    # domain split is the same.
    ("dflash", "attn_q"): ["math", "code"],
    # FFN gate in the encoder is the same: "math".
    ("dflash", "ffn_gate"): ["math"],
    # token_embd is shared with the trunk in the
    # dflash / shared_embd architecture; the domain
    # split mirrors the trunk's.
    ("dflash", "token_embd"): ["wiki", "science"],

    # ---- dspark (3) ----------------------------------------------
    # dspark is a drafter (autoregressive); its attn_q is
    # most diagnostic on "code" (the drafter is trained
    # on the same data the trunk is, but with the
    # acceptance-rejection signal).
    ("dspark", "attn_q"):  ["code"],
    # FFN up is the same as the trunk's: "science".
    ("dspark", "ffn_up"):   ["science"],
    # token_embd: "wiki" + "science".
    ("dspark", "token_embd"): ["wiki", "science"],

    # ---- mtp_nextn (3) -------------------------------------------
    # MTP-NextN is a multi-token prediction head; its
    # attn_output is the post-attention projection on
    # the predicted-token branch.
    ("mtp_nextn", "attn_output"): ["code"],
    # FFN gate: "math".
    ("mtp_nextn", "ffn_gate"): ["math"],
    # token_embd: "wiki" + "science".
    ("mtp_nextn", "token_embd"): ["wiki", "science"],

    # ---- shared_embd (2) -----------------------------------------
    # The shared embedding lookup (the dflash / dspark
    # share this with the trunk).
    ("shared_embd", "token_embd"): ["wiki", "science"],
    # The shared output projection.
    ("shared_embd", "output"): ["wiki", "code"],

    # ---- vision_tower (5) ----------------------------------------
    # The vision tower's patch embedding is most diagnostic
    # on the "image" subset (the patch grid is the
    # canonical 2-D unfold the projection operates on).
    ("vision_tower", "patch_embd"): ["image"],
    # The position embedding is the per-patch positional
    # bias: the "image" subset has the broadest
    # per-position envelope.
    ("vision_tower", "position_embd"): ["image"],
    # attn_q: the "image" subset has the highest per-patch
    # attention variance.
    ("vision_tower", "attn_q"):  ["image"],
    # attn_v: the "image" subset.
    ("vision_tower", "attn_v"):  ["image"],
    # FFN up: the "image" subset.
    ("vision_tower", "ffn_up"):  ["image"],

    # ---- audio_tower (4) -----------------------------------------
    # The audio tower's patch embedding is most diagnostic
    # on the "audio" subset (the spectrogram patch grid).
    ("audio_tower", "patch_embd"): ["audio"],
    # position_embd: "audio".
    ("audio_tower", "position_embd"): ["audio"],
    # attn_q: "audio" (the per-frame attention envelope).
    ("audio_tower", "attn_q"):  ["audio"],
    # FFN up: "audio".
    ("audio_tower", "ffn_up"):  ["audio"],

    # ---- mm_projector (9) ----------------------------------------
    # The mm_projector's mm_up is the up-projection from
    # the vision / audio tower output to the trunk
    # embedding space.
    ("mm_projector", "mm_up"):   ["image", "audio"],
    # mm_gate is the gating projection on the trunk-
    # side text branch.
    ("mm_projector", "mm_gate"): ["text"],
    # mm_input_projection is the per-patch / per-frame
    # projection from the tower's output to the
    # mm_projector's input.
    ("mm_projector", "mm_input_projection"): ["image", "audio"],
    # attn_q: the "image" + "audio" union (the projector
    # is modality-agnostic on the trunk side).
    ("mm_projector", "attn_q"):  ["image", "audio"],
    # attn_k: same.
    ("mm_projector", "attn_k"):  ["image", "audio"],
    # attn_v: same.
    ("mm_projector", "attn_v"):  ["image", "audio"],
    # attn_output: same.
    ("mm_projector", "attn_output"): ["image", "audio"],
    # FFN gate: "text" (the projector sees the trunk's
    # text-token stream as its post-projection input).
    ("mm_projector", "ffn_gate"): ["text"],
    # FFN up: "text".
    ("mm_projector", "ffn_up"):   ["text"],
}
#: Wildcard fallback for any (model_role, family) not in
#: the table. The fallback is a uniform sample of all
#: domains the corpus / modality knows; the orchestrator
#: uses the fallback only when the family is not in the
#: table (a new model_role / family combination the
#: architect has not yet classified). The fallback's
#: intent: "we don't know what's best, so sample
#: uniformly" -- which is the same effect the global
#: 4x sample bump produces, only on the
#: monitor-verdict tensors rather than every tensor.
FAMILY_DOMAIN_MAPPING_FALLBACK: list[str] = [
    "default", "code", "math", "wiki", "science",
    "image", "audio", "text",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class StatsSnapshot:
    """Per-tensor activation-stats snapshot (the focused
    re-capture's output for one tensor).

    Mirrors the column set the ``tensor_stats`` table
    carries (kurtosis / eff_rank / rms / mean_abs /
    tail_ratio / p99); the orchestrator packages each
    ``StatsSnapshot`` into the ``BackfillResult`` and
    passes it back to the iteration loop. The
    ``source`` field is always ``SOURCE_BACKFILL_REAL``;
    the ``backfill_count`` field is read from the DB
    before the re-capture and used by the orchestrator to
    decide whether to re-trigger the backfill on the
    next iteration.
    """

    tensor_name: str
    model_role: str
    family: str
    layer_depth: int
    kurtosis: float
    eff_rank: float
    rms: float
    mean_abs: float
    tail_ratio: float
    p99: float | None = None
    n_samples: int = 0
    backfill_count: int = 0
    domains: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "tensor": self.tensor_name,
            "model_role": self.model_role,
            "family": self.family,
            "layer_depth": int(self.layer_depth),
            "kurtosis": float(self.kurtosis),
            "eff_rank": float(self.eff_rank),
            "rms": float(self.rms),
            "mean_abs": float(self.mean_abs),
            "tail_ratio": float(self.tail_ratio),
            "p99": (float(self.p99) if self.p99 is not None else None),
            "n_samples": int(self.n_samples),
            "backfill_count": int(self.backfill_count),
            "domains": list(self.domains),
            "source": SOURCE_BACKFILL_REAL,
        }


@dataclasses.dataclass
class BackfillResult:
    """The full output of a single backfill pass.

    The orchestrator consumes ``tensors_processed`` /
    ``samples_consumed`` to log the pass on stderr;
    ``domain_subsets`` is the per-(model_role, family)
    -> [domain, ...] map the backfill used (the audit
    trail for "we sampled from the right domain
    subset"); ``new_stats_summary`` is the per-tensor
    stats the next iteration's ``l5_outcome``
    evaluation reads.
    """

    tensors_processed: int
    samples_consumed: int
    domain_subsets: dict[tuple[str, str], list[str]] = dataclasses.field(
        default_factory=dict,
    )
    new_stats_summary: dict[str, StatsSnapshot] = dataclasses.field(
        default_factory=dict,
    )
    rounds_completed: int = 0
    error_count: int = 0
    error_messages: list[str] = dataclasses.field(default_factory=list)
    wall_time_sec: float = 0.0

    def to_dict(self) -> dict:
        return {
            "tensors_processed": int(self.tensors_processed),
            "samples_consumed": int(self.samples_consumed),
            "domain_subsets": {
                f"{r}.{f}": list(d) for (r, f), d in self.domain_subsets.items()
            },
            "new_stats_summary": {
                k: v.to_dict() for k, v in self.new_stats_summary.items()
            },
            "rounds_completed": int(self.rounds_completed),
            "error_count": int(self.error_count),
            "error_messages": list(self.error_messages),
            "wall_time_sec": float(self.wall_time_sec),
        }


# ---------------------------------------------------------------------------
# Domain-mapping helpers
# ---------------------------------------------------------------------------


def domain_subset_for(role: str, family: str) -> list[str]:
    """Return the domain subset for ``(role, family)``,
    or the wildcard fallback if the tuple is not in the
    table.

    The lookup is a single dict access (O(1)); the
    wildcard fallback is the same set of domains for
    every role. The return is a fresh ``list`` so the
    caller can mutate it without affecting the table.
    """
    key = (str(role), str(family))
    if key in FAMILY_DOMAIN_MAPPING:
        return list(FAMILY_DOMAIN_MAPPING[key])
    return list(FAMILY_DOMAIN_MAPPING_FALLBACK)


def family_from_tensor_name(name: str, role: str) -> str:
    """Derive the family for a tensor name; the
    role-prefixed name forms are normalised first.

    The convention is the same as
    ``l5_orchestrator._tensor_family``: the
    ``.weight`` / ``.bias`` suffix is stripped, the
    ``blk.<i>.`` prefix is dropped, and the family is
    the next ``.``-separated segment. The mmproj
    role-prefixed names (``v.``, ``a.``, ``mm.``) are
    stripped first so a ``v.blk.0.attn_q.weight``
    tensor's family is ``attn_q`` (not
    ``v.attn_q``). The lookup is the canonical
    ``(role, family)`` tuple the
    ``FAMILY_DOMAIN_MAPPING`` table keys on.
    """
    base = str(name)
    for suf in (".weight", ".bias"):
        if base.endswith(suf):
            base = base[: -len(suf)]
            break
    # Strip the role prefix (v. / a. / mm.) so the
    # family derivation is uniform across roles.
    if role == "vision_tower" and base.startswith("v."):
        base = base[2:]
    elif role == "audio_tower" and base.startswith("a."):
        base = base[2:]
    elif role == "mm_projector" and base.startswith("mm."):
        base = base[3:]
    parts = base.split(".")
    if len(parts) >= 3 and parts[0] == "blk":
        return parts[2]
    if len(parts) >= 2:
        return parts[1]
    return base


# ---------------------------------------------------------------------------
# Per-tensor subprocess dispatch
# ---------------------------------------------------------------------------


def _per_tensor_capture_command(
    *,
    python_executable: str,
    role: str,
    tensor_name: str,
    model_hash: str,
    db_path: Path | None,
    corpus_root: Path | None,
    sample_cap: int,
    output_path: Path,
    layers_dir: Path | None,
    component_path: Path | None,
    seed: int,
) -> list[str]:
    """Build the subprocess command for one
    monitor-verdict tensor.

    The text-side roles (trunk / dflash / dspark /
    mtp_nextn / shared_embd) shell out to
    ``per_tensor_calibrate.py --backfill-tensor``; the
    mmproj roles (vision_tower / audio_tower /
    mm_projector) shell out to
    ``multimodal_calibrate.py --backfill-tensor``.
    The ``--backfill-sample-cap`` / ``--backfill-corpus``
    / ``--backfill-db`` / ``--seed`` flags are passed
    through; the ``--output`` flag is a per-tensor
    sidecar the parent reads.

    Memory isolation is the goal of the subprocess
    dispatch: a single in-process capture holds the
    weight + activations + intermediates in memory; the
    next capture would inherit that memory. The
    subprocess-isolated path releases the memory when
    the child exits.
    """
    if role in MM_ROLES:
        # mmproj: the component path is the GGUF
        # (vision / audio / projector). The driver
        # picks the role from the flag; we pass the
        # same --vision-tower flag for vision, etc.
        # The output sidecar is the per-tensor JSON.
        tool = Path(__file__).resolve().parent / "multimodal_calibrate.py"
        cmd = [
            python_executable, str(tool),
            "--output", str(output_path),
            "--backfill-tensor", tensor_name,
            "--backfill-sample-cap", str(int(sample_cap)),
            "--model-hash", str(model_hash),
            "--seed", str(int(seed)),
            "--print-summary",
        ]
        if db_path is not None:
            cmd += ["--db", str(db_path)]
        if corpus_root is not None:
            cmd += ["--backfill-corpus", str(corpus_root)]
        if component_path is not None:
            if role == "vision_tower":
                cmd += ["--vision-tower", str(component_path)]
            elif role == "audio_tower":
                cmd += ["--audio-tower", str(component_path)]
            else:
                cmd += ["--mm-projector", str(component_path)]
        return cmd
    # text-side: the per_tensor_calibrate driver
    # expects a layers directory (or .npz list) of
    # per-tensor bundles. The ``--backfill-tensor``
    # flag selects the single bundle to re-capture.
    tool = Path(__file__).resolve().parent / "per_tensor_calibrate.py"
    cmd = [
        python_executable, str(tool),
        "--fitness", "lrq",
        "--output", str(output_path),
        "--backfill-tensor", tensor_name,
        "--backfill-sample-cap", str(int(sample_cap)),
        "--model-role", role,
        "--model-hash", str(model_hash),
        "--seed", str(int(seed)),
    ]
    if db_path is not None:
        cmd += ["--backfill-db", str(db_path)]
    if corpus_root is not None:
        cmd += ["--backfill-corpus", str(corpus_root)]
    if layers_dir is not None:
        cmd += ["--layers", str(layers_dir)]
    return cmd


def _read_backfill_sidecar(path: Path) -> list[dict]:
    """Read the per-tensor backfill sidecar JSON.

    The driver writes one row per tensor to the
    sidecar; the parent reads the file and converts
    each row to a ``StatsSnapshot``. The function
    tolerates an empty / missing file (the subprocess
    may have failed before writing) and returns an
    empty list in that case; the orchestrator logs the
    failure to ``BackfillResult.error_messages`` and
    continues with the next tensor.
    """
    if not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        sys.stderr.write(
            f"backfill: sidecar read failed for {path}: "
            f"{exc.__class__.__name__}: {str(exc)[:120]}\n"
        )
        return []
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return []
    return [r for r in rows if isinstance(r, dict)]


# ---------------------------------------------------------------------------
# TargetedBackfill
# ---------------------------------------------------------------------------


class TargetedBackfill:
    """The orchestrator's hook target. Owns the
    family->domain mapping, the per-tensor subprocess
    dispatch, the sidecar writer, and the thread pool.

    The orchestrator constructs a single
    ``TargetedBackfill`` instance at startup; the
    ``run_backfill_async`` method returns a
    ``concurrent.futures.Future`` that resolves to a
    ``BackfillResult``. The orchestrator's iteration
    loop waits on the future at the next "apply" step
    (the spec calls for async dispatch + sync at the
    apply step so the next iteration's plan reads the
    re-captured stats).

    The thread pool is ``concurrent.futures.
    ThreadPoolExecutor(max_workers=2)``; the
    per-tensor captures are sequential
    ``subprocess.run`` calls inside the executor
    threads (memory isolation is the goal, not
    in-process concurrency). The thread pool is just
    the async dispatch layer.
    """

    def __init__(
        self,
        *,
        max_workers: int = DEFAULT_MAX_WORKERS,
        max_backfill_rounds: int = DEFAULT_MAX_BACKFILL_ROUNDS,
        sample_cap: int = DEFAULT_SAMPLE_CAP,
        subprocess_timeout_sec: int = 600,
        verbose: bool = False,
    ) -> None:
        self.max_workers = max(1, int(max_workers))
        self.max_backfill_rounds = max(1, int(max_backfill_rounds))
        self.sample_cap = max(1, int(sample_cap))
        self.subprocess_timeout_sec = max(1, int(subprocess_timeout_sec))
        self.verbose = bool(verbose)
        # Thread pool is created lazily on the first
        # ``run_backfill_async`` call so the constructor
        # cost is constant (the orchestrator's startup
        # does not pay for a pool it may not use, e.g.
        # when --no-targeted-recal is set).
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._closed = False
        # Sidecar directory: per-tensor subprocesses
        # write their JSON here; the parent reads the
        # files. The directory is unique to this
        # ``TargetedBackfill`` instance so concurrent
        # orchestrators do not collide.
        self._sidecar_dir = (
            Path(f"/tmp/backfill-sidecars-{os.getpid()}-"
                 f"{int(time.time() * 1e6)}")
        )

    # -- lifecycle ----------------------------------------------------

    def _get_executor(self) -> concurrent.futures.ThreadPoolExecutor:
        if self._executor is None:
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers,
                thread_name_prefix="tessera-backfill",
            )
        return self._executor

    def close(self) -> None:
        """Idempotent shutdown. Flushes the thread pool
        and removes the sidecar directory. The orchestrator
        calls this in its ``__exit__`` / finally block."""
        if self._closed:
            return
        self._closed = True
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        # Best-effort sidecar cleanup. We do not crash
        # on cleanup errors (the sidecar files are
        # best-effort audit trails; the canonical
        # side is the DB).
        try:
            if self._sidecar_dir.is_dir():
                for child in self._sidecar_dir.iterdir():
                    try:
                        child.unlink()
                    except OSError:
                        pass
                self._sidecar_dir.rmdir()
        except OSError:
            pass

    def __enter__(self) -> "TargetedBackfill":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- public API --------------------------------------------------

    def run_backfill_async(
        self,
        *,
        db_path: Path,
        model_hash: str,
        components: Mapping[str, Path | None],
        corpus_root: Path | None,
        monitor_tensors: Sequence[Mapping[str, str]],
    ) -> concurrent.futures.Future[BackfillResult]:
        """Schedule a backfill pass and return a Future.

        ``monitor_tensors`` is the list of dicts the
        orchestrator filters from ``plan.actions``:
        each entry is ``{"name": ..., "model_role": ...,
        "family": ..., "layer_depth": ...}``. The
        backfill iterates the list, dispatches one
        subprocess per tensor, and merges the
        per-tensor sidecars into the
        ``BackfillResult.new_stats_summary`` map.

        The Future is a ``concurrent.futures.Future``
        the orchestrator's iteration loop waits on
        (with a timeout, so a stuck subprocess does
        not block the loop indefinitely). The result
        is always a ``BackfillResult``; even an
        all-error backfill returns one (with
        ``error_count > 0``).
        """
        executor = self._get_executor()
        return executor.submit(
            self._run_backfill_impl,
            db_path=db_path,
            model_hash=model_hash,
            components=dict(components),
            corpus_root=corpus_root,
            monitor_tensors=list(monitor_tensors),
        )

    def run_backfill(
        self,
        *,
        db: TesseraDB,
        model_hash: str,
        components: Mapping[str, Path | None],
        corpus_root: Path | None,
        monitor_tensors: Sequence[Mapping[str, str]],
    ) -> BackfillResult:
        """Synchronous wrapper around ``run_backfill_async``.

        Useful for tests and for the CLI; the
        orchestrator's iteration loop uses the async
        version. ``db`` is the ``TesseraDB`` instance
        the orchestrator already has open; the
        function reads ``backfill_count`` from the DB
        and writes the new rows through the same
        ``TesseraDB`` (the upsert path the orchestrator
        uses for ``tensor_stats``).
        """
        # Read the current backfill_count for every
        # monitor tensor so the gate (``< max_rounds``)
        # is enforced at the call site, not just inside
        # the iteration loop.
        eligible: list[dict] = []
        for entry in monitor_tensors:
            name = str(entry.get("name", ""))
            role = str(entry.get("model_role", "trunk"))
            current = self._read_backfill_count(
                db, model_hash=model_hash, name=name, role=role,
            )
            if current < self.max_backfill_rounds:
                eligible.append(dict(entry))
        if not eligible:
            return BackfillResult(
                tensors_processed=0,
                samples_consumed=0,
                rounds_completed=0,
                wall_time_sec=0.0,
            )
        return self._run_backfill_impl(
            db_path=None,
            model_hash=model_hash,
            components=dict(components),
            corpus_root=corpus_root,
            monitor_tensors=eligible,
            db=db,
        )

    # -- internals ----------------------------------------------------

    def _read_backfill_count(
        self,
        db: TesseraDB,
        *,
        model_hash: str,
        name: str,
        role: str,
    ) -> int:
        """Read the current ``backfill_count`` for
        ``(model_hash, role, name)``; return 0 when
        the row is missing or the column is NULL.

        A NULL ``backfill_count`` means "no backfill
        yet" (the migration's default). The function
        treats NULL as 0 so the ``< max_rounds`` gate
        passes for a fresh row.
        """
        try:
            df = db.query(
                "SELECT backfill_count FROM tensor_stats "
                f"WHERE model_hash = '{sql_escape(model_hash)}' "
                f"AND model_role = '{sql_escape(role)}' "
                f"AND name = '{sql_escape(name)}'"
            )
        except Exception as e:  # pragma: no cover - db safety
            sys.stderr.write(
                f"backfill: read backfill_count failed for "
                f"({model_hash}, {role}, {name}): "
                f"{e.__class__.__name__}: {str(e)[:120]}\n"
            )
            return 0
        if df.is_empty():
            return 0
        v = df["backfill_count"].to_list()[0]
        if v is None:
            return 0
        try:
            return int(v)
        except (TypeError, ValueError):
            return 0

    def _run_backfill_impl(
        self,
        *,
        db_path: Path | None,
        model_hash: str,
        components: Mapping[str, Path | None],
        corpus_root: Path | None,
        monitor_tensors: list[dict],
        db: TesseraDB | None = None,
    ) -> BackfillResult:
        """The actual backfill implementation.

        Subprocess dispatch: for every monitor tensor,
        build the per-tensor command, run it as a
        subprocess (memory isolation), read the
        per-tensor sidecar, accumulate the
        ``StatsSnapshot``, and write the rows to the
        DB (when ``db`` is set). The function is
        single-threaded (the per-tensor captures are
        sequential); the executor wrapper
        (``run_backfill_async``) provides the async
        dispatch.

        The DB write is through ``db.insert_tensor_stats``
        when ``db`` is given; the subprocess
        ``--backfill-db`` flag is set when ``db_path``
        is given and the subprocess does its own
        write. The two paths are mutually exclusive
        at the call site (the orchestrator passes one
        or the other). The implementation here picks
        the subprocess-write path when ``db_path`` is
        given; the in-process path is the test fixture.
        """
        if not monitor_tensors:
            return BackfillResult(
                tensors_processed=0, samples_consumed=0,
            )
        t0 = time.monotonic()
        self._sidecar_dir.mkdir(parents=True, exist_ok=True)
        result = BackfillResult(
            tensors_processed=0,
            samples_consumed=0,
        )
        python_executable = sys.executable
        for i, entry in enumerate(monitor_tensors):
            name = str(entry.get("name", ""))
            role = str(entry.get("model_role", "trunk"))
            if not name:
                continue
            if role not in MODEL_ROLES:
                # Skip unknown roles with a stderr
                # warning. The orchestrator should
                # never emit a monitor verdict for an
                # unknown role; the warning is the
                # safety net.
                sys.stderr.write(
                    f"backfill: skipping tensor {name!r} with "
                    f"unknown model_role {role!r}\n"
                )
                result.error_count += 1
                result.error_messages.append(
                    f"unknown model_role {role!r} for {name!r}"
                )
                continue
            family = family_from_tensor_name(name, role)
            domains = tuple(domain_subset_for(role, family))
            layers_dir = None
            component_path: Path | None = None
            if role in MM_ROLES:
                component_path = components.get(role)
            else:
                layers_dir = components.get(role)
            sidecar_path = (
                self._sidecar_dir / f"backfill-{i:04d}.json"
            )
            cmd = _per_tensor_capture_command(
                python_executable=python_executable,
                role=role,
                tensor_name=name,
                model_hash=model_hash,
                db_path=db_path,
                corpus_root=corpus_root,
                sample_cap=self.sample_cap,
                output_path=sidecar_path,
                layers_dir=(
                    Path(layers_dir) if layers_dir else None
                ),
                component_path=component_path,
                seed=i,
            )
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.subprocess_timeout_sec,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                result.error_count += 1
                result.error_messages.append(
                    f"timeout after {self.subprocess_timeout_sec}s "
                    f"on {name!r}"
                )
                sys.stderr.write(
                    f"backfill: subprocess timeout on {name!r}; "
                    f"stdout={str(exc.stdout or '')[:200]} "
                    f"stderr={str(exc.stderr or '')[:200]}\n"
                )
                continue
            except Exception as exc:  # pragma: no cover - safety
                result.error_count += 1
                result.error_messages.append(
                    f"subprocess error on {name!r}: "
                    f"{exc.__class__.__name__}"
                )
                continue
            if proc.returncode != 0:
                result.error_count += 1
                result.error_messages.append(
                    f"subprocess returned {proc.returncode} on "
                    f"{name!r}; stderr={proc.stderr[:200]}"
                )
                if self.verbose:
                    sys.stderr.write(
                        f"backfill: subprocess on {name!r} returned "
                        f"{proc.returncode}; stderr={proc.stderr[:200]}\n"
                    )
                continue
            rows = _read_backfill_sidecar(sidecar_path)
            if not rows:
                result.error_count += 1
                result.error_messages.append(
                    f"empty sidecar for {name!r}"
                )
                continue
            for row in rows:
                rname = str(row.get("name", name))
                rrole = str(row.get("model_role", role))
                rfamily = str(row.get("family", family))
                rl = int(row.get("layer_depth", 0) or 0)
                snap = StatsSnapshot(
                    tensor_name=rname,
                    model_role=rrole,
                    family=rfamily,
                    layer_depth=rl,
                    kurtosis=float(row.get("kurtosis", 0.0) or 0.0),
                    eff_rank=float(row.get("eff_rank", 0.0) or 0.0),
                    rms=float(row.get("rms", 0.0) or 0.0),
                    mean_abs=float(row.get("mean_abs", 0.0) or 0.0),
                    tail_ratio=float(
                        row.get("tail_ratio", 1.0) or 1.0
                    ),
                    p99=(
                        float(row["p99"])
                        if row.get("p99") is not None else None
                    ),
                    n_samples=int(row.get("n_samples", 0) or 0),
                    backfill_count=int(
                        row.get("backfill_count", 0) or 0
                    ),
                    domains=domains,
                )
                result.new_stats_summary[rname] = snap
                result.samples_consumed += max(
                    0, snap.n_samples if snap.n_samples else self.sample_cap
                )
                result.tensors_processed += 1
            result.domain_subsets[(role, family)] = list(domains)
        # In-process DB write: the orchestrator passes
        # ``db`` (the TesseraDB instance) directly so
        # the subprocess's ``--backfill-db`` flag is
        # not used. The two paths are mutually
        # exclusive; the subprocess path is the
        # production one (the ``--backfill-db``
        # ``--model-hash`` flags are the seam the
        # orchestrator uses).
        if db is not None and result.new_stats_summary:
            db_rows: list[dict] = []
            for snap in result.new_stats_summary.values():
                db_rows.append({
                    "name": snap.tensor_name,
                    "model_role": snap.model_role,
                    "family": snap.family,
                    "layer_depth": int(snap.layer_depth),
                    "kurtosis": float(snap.kurtosis),
                    "eff_rank": float(snap.eff_rank),
                    "rms": float(snap.rms),
                    "mean_abs": float(snap.mean_abs),
                    "tail_ratio": float(snap.tail_ratio),
                    "p99": (float(snap.p99) if snap.p99 is not None else None),
                    "source": SOURCE_BACKFILL_REAL,
                    "recommended_action": "monitor",
                })
            db.insert_tensor_stats(model_hash=model_hash, rows=db_rows)
        result.rounds_completed = 1
        result.wall_time_sec = float(time.monotonic() - t0)
        return result


# ---------------------------------------------------------------------------
# Free function (top-level entry point for the orchestrator)
# ---------------------------------------------------------------------------


def run_backfill(
    *,
    db: TesseraDB,
    model_hash: str,
    components: Mapping[str, Path | None],
    corpus_root: Path,
    max_rounds: int = DEFAULT_MAX_BACKFILL_ROUNDS,
    sample_cap: int = DEFAULT_SAMPLE_CAP,
) -> BackfillResult:
    """The orchestrator's hook. Constructs a
    ``TargetedBackfill`` instance, runs the focused
    re-capture on the monitor-verdict tensors the
    orchestrator flagged, and returns the
    ``BackfillResult``.

    ``components`` is the per-role map: ``{"trunk":
    layers_dir, "vision_tower": vision_gguf, ...}``.
    The role key drives the dispatch (text-side vs
    mmproj); the value is the layers directory (text)
    or the GGUF path (mmproj). A missing / empty
    value skips the role's tensors with a stderr
    warning (the orchestrator's monitor verdict is
    role-agnostic; the backfill is only as good as
    the components the orchestrator passes).

    ``corpus_root`` is the calibration corpus root;
    the backfill samples from the corpus's
    domain-specific subsets (the contract the
    ``build-calibration-corpus.py`` writer
    established). When the corpus is missing, the
    backfill falls back to the per-driver
    synthetic-sample path (the same path the
    default mode uses).
    """
    monitor_tensors: list[dict] = []
    try:
        df = db.query(
            "SELECT name, model_role, family, layer_depth "
            f"FROM tensor_stats WHERE model_hash = '{sql_escape(model_hash)}' "
            "AND recommended_action = 'monitor'"
        )
    except Exception as e:  # pragma: no cover - db safety
        sys.stderr.write(
            f"backfill: monitor query failed: "
            f"{e.__class__.__name__}: {str(e)[:120]}\n"
        )
        df = None
    if df is not None and not df.is_empty():
        for row in df.iter_rows(named=True):
            monitor_tensors.append({
                "name": str(row.get("name", "")),
                "model_role": str(row.get("model_role", "trunk")),
                "family": str(row.get("family", "")),
                "layer_depth": int(row.get("layer_depth", 0) or 0),
            })
    if not monitor_tensors:
        return BackfillResult(tensors_processed=0, samples_consumed=0)
    with TargetedBackfill(
        max_backfill_rounds=max_rounds,
        sample_cap=sample_cap,
    ) as engine:
        return engine.run_backfill(
            db=db,
            model_hash=model_hash,
            components=components,
            corpus_root=corpus_root,
            monitor_tensors=monitor_tensors,
        )


# ---------------------------------------------------------------------------
# Sidecar writer
# ---------------------------------------------------------------------------


def write_sidecar(
    result: BackfillResult,
    path: Path,
    *,
    model_hash: str,
    components: Mapping[str, Path | None] | None = None,
    corpus_root: Path | None = None,
    max_rounds: int = DEFAULT_MAX_BACKFILL_ROUNDS,
    sample_cap: int = DEFAULT_SAMPLE_CAP,
) -> None:
    """Write the backfill result as a
    ``llama.tessera.backfill.v1`` JSON sidecar.

    The sidecar is the per-call audit trail; the DB
    is the canonical side. The sidecar is for human
    inspection (the operator can read the per-tensor
    stats and the per-(role, family) domain subsets
    without querying DuckDB).
    """
    payload = {
        "schema": SIDECAR_SCHEMA,
        "tool": "backfill.py",
        "model_hash": str(model_hash),
        "components": {
            k: (str(v) if v is not None else None)
            for k, v in (components or {}).items()
        },
        "corpus_root": (str(corpus_root) if corpus_root is not None else None),
        "max_backfill_rounds": int(max_rounds),
        "sample_cap": int(sample_cap),
        "source": SOURCE_BACKFILL_REAL,
        "result": result.to_dict(),
        "timestamp": time.time(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Targeted re-calibration: focused re-capture for "
            "L5 monitor-verdict tensors. Reads the "
            "monitor-verdict rows from --db, dispatches one "
            "per-tensor subprocess per row, and writes the "
            "per-tensor activation stats back to the DB with "
            "source=SOURCE_BACKFILL_REAL and "
            "backfill_count incremented."
        ),
    )
    parser.add_argument(
        "--db", type=Path, required=True,
        help="Path to the unified tessera.duckdb file.",
    )
    parser.add_argument(
        "--model-hash", required=True,
        help="Model hash for the monitor-verdict lookup.",
    )
    parser.add_argument(
        "--corpus-root", type=Path, default=None,
        help="Path to the calibration corpus root.",
    )
    parser.add_argument(
        "--max-backfill-rounds",
        type=int,
        default=DEFAULT_MAX_BACKFILL_ROUNDS,
        help=(
            "Maximum number of backfill rounds per "
            "monitor-verdict tensor (default "
            f"{DEFAULT_MAX_BACKFILL_ROUNDS})."
        ),
    )
    parser.add_argument(
        "--backfill-sample-cap",
        type=int,
        default=DEFAULT_SAMPLE_CAP,
        help=(
            "Per-tensor sample cap for the backfill "
            f"re-capture (default {DEFAULT_SAMPLE_CAP})."
        ),
    )
    parser.add_argument(
        "--component",
        action="append",
        default=[],
        metavar="ROLE=PATH",
        help=(
            "Per-role component path. Repeatable. "
            "Format: ROLE=PATH, where ROLE is one of "
            "trunk / dflash / dspark / mtp_nextn / "
            "shared_embd / vision_tower / audio_tower / "
            "mm_projector. Text-side roles take a layers "
            "directory; mmproj roles take a GGUF path."
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help=(
            "Path to the sidecar JSON "
            "(llama.tessera.backfill.v1). Defaults to "
            "stdout when omitted."
        ),
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Log per-tensor progress to stderr.",
    )
    return parser


def _parse_components(raw: list[str]) -> dict[str, Path]:
    """Parse the --component flag's repeatable
    ``ROLE=PATH`` entries into a ``{role: Path}`` map.

    The validation is the same as the
    ``MODEL_ROLES`` membership test; an unknown role
    raises ValueError so the CLI fails fast on a
    typo (the orchestrator never silently ignores an
    unknown role).
    """
    out: dict[str, Path] = {}
    for entry in raw:
        if "=" not in entry:
            raise ValueError(
                f"--component {entry!r}: expected ROLE=PATH"
            )
        role, path_str = entry.split("=", 1)
        role = role.strip()
        if role not in MODEL_ROLES:
            raise ValueError(
                f"--component role {role!r} not in {MODEL_ROLES!r}"
            )
        out[role] = Path(path_str.strip())
    return out


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    components = _parse_components(args.component)
    try:
        with TesseraDB.open(args.db) as db:
            result = run_backfill(
                db=db,
                model_hash=str(args.model_hash),
                components=components,
                corpus_root=(
                    Path(args.corpus_root)
                    if args.corpus_root is not None
                    else Path("/tmp")  # the synthetic fallback
                ),
                max_rounds=int(args.max_backfill_rounds),
                sample_cap=int(args.backfill_sample_cap),
            )
    except Exception as e:
        sys.stderr.write(
            f"backfill: failed: {e.__class__.__name__}: {e}\n"
        )
        return 1
    payload = {
        "schema": SIDECAR_SCHEMA,
        "tool": "backfill.py",
        "model_hash": str(args.model_hash),
        "components": {k: str(v) for k, v in components.items()},
        "corpus_root": str(args.corpus_root) if args.corpus_root else None,
        "max_backfill_rounds": int(args.max_backfill_rounds),
        "sample_cap": int(args.backfill_sample_cap),
        "source": SOURCE_BACKFILL_REAL,
        "result": result.to_dict(),
        "timestamp": time.time(),
    }
    text = json.dumps(payload, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
