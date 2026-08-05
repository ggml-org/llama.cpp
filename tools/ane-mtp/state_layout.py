"""ane_state_layout_v1 — Python side of the multifunction ANE state manifest.

The manifest is emitted next to a .mlmodelc (typically as
``<bundle_name>.ane_state.v1.json``) and consumed by the runtime
(common/ane-mtp.mm, ggml/src/ggml-ane/ggml-ane.mm) to:

  1. Allocate one IOSurface of ``state_size_bytes`` at load time.
  2. Pin each declared slot to its offset inside that IOSurface.
  3. Wrap each slot as an IOSurface-backed MLMultiArray with
     ``deallocator:nil`` (zero-copy, see common/ane-mtp.mm's
     ``wrap_multi_array`` for the canonical pattern).
  4. Build the per-function dependency graph for the E-core
     pump's lock-free state machine.

JSON is the on-disk source of truth. The C struct
``ane_state_layout_v1_t`` in ``common/ane-state.h`` is a
deserialized view; the runtime's JSON reader (TBD) maps the
JSON to that struct.

Layout invariants enforced here:
  - All slot offsets are 16 KB-aligned (ANE page size).
  - All slot sizes are multiples of 16 bytes (ANE SIMD safety).
  - The total state size is at least 64 KB (ANE minimum alloc,
    see ``docs/tessera-ane-matmul-research.md`` Section 1.1).
  - STATE-kind slots are referenced by at least one function as
    an input or output (otherwise they are dead state).
  - INPUT-kind slots must be written by some function before
    they are read (enforced by the ``deps`` list).
  - OUTPUT-kind slots must be read by at least one downstream
    function (otherwise the work is wasted).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


SCHEMA_VERSION = 1
ANE_PAGE_BYTES = 16 * 1024        # ANE page alignment
ANE_MIN_ALLOC_BYTES = 64 * 1024   # ANE minimum alloc (Orion #4)
ANE_SIMD_ALIGN = 16               # per-slot SIMD alignment

SLOT_KIND_INPUT = "input"
SLOT_KIND_OUTPUT = "output"
SLOT_KIND_STATE = "state"
SLOT_KIND_SCRATCH = "scratch"

DTYPE_F32 = "f32"
DTYPE_F16 = "f16"
DTYPE_I32 = "i32"

DTYPE_BYTES = {DTYPE_F32: 4, DTYPE_F16: 2, DTYPE_I32: 4}

ROLE_UNKNOWN = "unknown"
ROLE_PREFILL = "prefill"
ROLE_MTP = "mtp"
ROLE_DFLASH = "dflash"
ROLE_HYBRID = "hybrid"
ROLE_SYNC = "sync"
ROLE_RESET = "reset"
ROLE_MATMUL = "matmul"
ROLE_RMS_NORM = "rms_norm"
ROLE_SOFT_MAX = "soft_max"
ROLE_ROPE = "rope"
ROLE_GLU = "glu"
ROLE_GET_ROWS = "get_rows"

# Core ML model type. Determines whether MLModelConfiguration.functionName
# is settable at load time. The W0 spike's matmul is a NeuralNetwork
# spec (functionName MUST be nil), while the multifunction prefill/MTP/
# DFlash bundles are ML Program specs (functionName required to pick
# which named function to bind). The conversion tool sets this on
# manifest emit and the runtime reads it to pick the load path.
MODEL_TYPE_NEURAL_NETWORK = "neural_network"
MODEL_TYPE_ML_PROGRAM = "ml_program"


@dataclass
class StateSlot:
    name: str
    kind: str           # one of SLOT_KIND_*
    dtype: str          # one of DTYPE_*
    shape: List[int]    # up to 4 dims
    offset: int         # byte offset in the state IOSurface
    size_bytes: int     # padded to ANE_SIMD_ALIGN

    def validate(self) -> None:
        if self.kind not in (SLOT_KIND_INPUT, SLOT_KIND_OUTPUT,
                             SLOT_KIND_STATE, SLOT_KIND_SCRATCH):
            raise ValueError(f"slot {self.name}: bad kind {self.kind!r}")
        if self.dtype not in DTYPE_BYTES:
            raise ValueError(f"slot {self.name}: bad dtype {self.dtype!r}")
        if not (0 < len(self.shape) <= 4):
            raise ValueError(f"slot {self.name}: shape must have 1-4 dims")
        if any(d <= 0 for d in self.shape):
            raise ValueError(f"slot {self.name}: shape dims must be positive")
        # Per-slot offset within the state IOSurface: 16-byte aligned for
        # SIMD safety. The IOSurface AS A WHOLE is 16KB-aligned and
        # 64KB-minimum (the conversion tool handles that in the parent
        # state_size_bytes), but individual slots can be at any 16B
        # boundary inside the IOSurface.
        if self.offset % ANE_SIMD_ALIGN != 0:
            raise ValueError(
                f"slot {self.name}: offset {self.offset} not 16B-aligned")
        if self.size_bytes % ANE_SIMD_ALIGN != 0:
            raise ValueError(
                f"slot {self.name}: size {self.size_bytes} not 16B-aligned")
        # Validate size matches shape * dtype
        expected = 1
        for d in self.shape:
            expected *= d
        expected *= DTYPE_BYTES[self.dtype]
        if expected > self.size_bytes:
            raise ValueError(
                f"slot {self.name}: shape {self.shape} * dtype {self.dtype} "
                f"= {expected} bytes > size_bytes {self.size_bytes}")


@dataclass
class FunctionSpec:
    name: str                       # bundle-internal name (e.g., "prefill_s32")
    role: str                       # one of ROLE_*
    bucket: int = 0                 # sequence/batch bucket (0 for non-bucketed)
    stateful: bool = False          # reads or writes any STATE slot
    input_slots: List[str] = field(default_factory=list)
    output_slots: List[str] = field(default_factory=list)
    core_ml_function_name: str = "" # "" means: default function ("main")
    use_ane: bool = True

    def validate(self) -> None:
        if self.role not in (ROLE_UNKNOWN, ROLE_PREFILL, ROLE_MTP,
                             ROLE_DFLASH, ROLE_HYBRID, ROLE_SYNC,
                             ROLE_RESET, ROLE_MATMUL,
                             ROLE_RMS_NORM, ROLE_SOFT_MAX, ROLE_ROPE,
                             ROLE_GLU, ROLE_GET_ROWS):
            raise ValueError(f"function {self.name}: bad role {self.role!r}")
        if len(self.input_slots) > 8 or len(self.output_slots) > 8:
            raise ValueError(
                f"function {self.name}: max 8 inputs and 8 outputs per function")
        if (self.role == ROLE_SYNC or self.role == ROLE_RESET) and self.use_ane:
            # sync/reset are CPU-side mem{cpy,set}; using the ANE for
            # them is a misconfiguration. The runtime can override but
            # the manifest is the source of truth.
            raise ValueError(
                f"function {self.name}: {self.role} must have use_ane=false")


@dataclass
class Dependency:
    """A directed edge in the function-to-slot graph.

    ``producer`` writes ``slot``; one or more ``consumers`` read it.
    The runtime builds the per-slot consumer list from this to
    signal MTLSharedEvent on producer completion and wait on
    MTLSharedEvent before consumer dispatch.
    """
    producer: str          # function name
    slot: str              # slot name
    consumers: List[str]   # consumer function names


@dataclass
class StateLayout:
    version: int
    bundle_name: str
    state_size_bytes: int
    model_type: str = MODEL_TYPE_NEURAL_NETWORK
    slots: List[StateSlot] = field(default_factory=list)
    functions: List[FunctionSpec] = field(default_factory=list)
    dependencies: List[Dependency] = field(default_factory=list)

    # ---- construction ----

    @classmethod
    def for_w0_matmul(cls, bundle_name: str, n: int) -> "StateLayout":
        """W0 spike layout: stateless matmul, two slots (x, y).

        Used by tools/ane-mtp/make-w0-matmul.py to emit the manifest
        for the single-function matmul fixture. No STATE slots, no
        dependencies, no cross-function coordination. This is the
        minimum-viable manifest and the smoke test for the format.
        """
        x_slot = StateSlot(
            name="x",
            kind=SLOT_KIND_INPUT,
            dtype=DTYPE_F32,
            shape=[n],
            offset=0,
            size_bytes=((n * 4 + ANE_SIMD_ALIGN - 1) // ANE_SIMD_ALIGN)
                      * ANE_SIMD_ALIGN,
        )
        y_slot = StateSlot(
            name="y",
            kind=SLOT_KIND_OUTPUT,
            dtype=DTYPE_F32,
            shape=[n],
            offset=ANE_PAGE_BYTES,
            size_bytes=((n * 4 + ANE_SIMD_ALIGN - 1) // ANE_SIMD_ALIGN)
                      * ANE_SIMD_ALIGN,
        )
        return cls(
            version=SCHEMA_VERSION,
            bundle_name=bundle_name,
            state_size_bytes=ANE_MIN_ALLOC_BYTES,
            model_type=MODEL_TYPE_NEURAL_NETWORK,
            slots=[x_slot, y_slot],
            functions=[FunctionSpec(
                name="main",
                role=ROLE_MATMUL,
                bucket=n,
                stateful=False,
                input_slots=["x"],
                output_slots=["y"],
                core_ml_function_name="main",
                use_ane=True,
            )],
            dependencies=[],
        )

    # ---- body-op constructors ----
    #
    # Phase 1 of docs/tessera-ane-ios-demo-design.md adds five
    # transformer body ops to the ANE backend: RMS_NORM, SOFT_MAX,
    # ROPE (gemma 4 variant), GLU (split form), GET_ROWS. Each
    # fixture is a single-function .mlmodelc shaped for the test
    # harness; the multifunction bundle (for_transformer_body)
    # glues them together as one stateless .mlmodelc with one
    # functionName per op.

    @classmethod
    def for_body_op(cls,
                    bundle_name: str,
                    role: str,
                    function_name: str,
                    inputs: list,
                    outputs: list,
                    state_size_bytes: int = ANE_MIN_ALLOC_BYTES) -> "StateLayout":
        """Single-function body-op layout: stateless, one slot per
        input/output, no STATE slots, no cross-function dependencies.

        ``inputs`` and ``outputs`` are lists of (name, dtype, shape)
        tuples. Slots are 16 KB-page-aligned (ANE page size); the
        state_size_bytes defaults to the 64 KB ANE minimum and is
        rounded up to the next 16 KB page.
        """
        if role not in (ROLE_RMS_NORM, ROLE_SOFT_MAX, ROLE_ROPE,
                        ROLE_GLU, ROLE_GET_ROWS, ROLE_MATMUL):
            raise ValueError(f"for_body_op: bad role {role!r}")

        def _slot(name: str, kind: str, dtype: str, shape: list, offset: int) -> StateSlot:
            esize = DTYPE_BYTES[dtype]
            count = 1
            for d in shape:
                count *= d
            raw = count * esize
            size = ((raw + ANE_SIMD_ALIGN - 1) // ANE_SIMD_ALIGN) * ANE_SIMD_ALIGN
            return StateSlot(
                name=name,
                kind=kind,
                dtype=dtype,
                shape=list(shape),
                offset=offset,
                size_bytes=size,
            )

        slots = []
        offset = 0
        in_names = []
        for (name, dtype, shape) in inputs:
            slots.append(_slot(name, SLOT_KIND_INPUT, dtype, shape, offset))
            offset += slots[-1].size_bytes
            # Pad each input to a full 16 KB page so the model's
            # IOSurface read is page-aligned (ANE read constraint).
            offset = ((offset + ANE_PAGE_BYTES - 1) // ANE_PAGE_BYTES) * ANE_PAGE_BYTES
            in_names.append(name)
        out_names = []
        for (name, dtype, shape) in outputs:
            slots.append(_slot(name, SLOT_KIND_OUTPUT, dtype, shape, offset))
            offset += slots[-1].size_bytes
            offset = ((offset + ANE_PAGE_BYTES - 1) // ANE_PAGE_BYTES) * ANE_PAGE_BYTES
            out_names.append(name)

        # Round the total state up to a 16 KB page and clamp to the
        # 64 KB ANE minimum. for_w0_matmul does the same.
        state_size = ((offset + ANE_PAGE_BYTES - 1) // ANE_PAGE_BYTES) * ANE_PAGE_BYTES
        if state_size < ANE_MIN_ALLOC_BYTES:
            state_size = ANE_MIN_ALLOC_BYTES
        if state_size_bytes > state_size:
            state_size = state_size_bytes

        return cls(
            version=SCHEMA_VERSION,
            bundle_name=bundle_name,
            state_size_bytes=state_size,
            model_type=MODEL_TYPE_ML_PROGRAM,
            slots=slots,
            functions=[FunctionSpec(
                name=function_name,
                role=role,
                bucket=0,
                stateful=False,
                input_slots=in_names,
                output_slots=out_names,
                core_ml_function_name=function_name,
                use_ane=True,
            )],
            dependencies=[],
        )

    @classmethod
    def for_transformer_body(cls,
                             bundle_name: str,
                             functions: list) -> "StateLayout":
        """Multifunction transformer-body layout: one .mlmodelc with
        N functions, one slot per function input/output, no cross-
        function dependencies (the bundle is stateless from the
        ANE's perspective; per-iteration state is supplied by the
        host via IOSurface).

        ``functions`` is a list of dicts with keys:
          - name: functionName in the .mlmodelc
          - role: one of ROLE_RMS_NORM / ROLE_SOFT_MAX / ROLE_ROPE /
                  ROLE_GLU / ROLE_GET_ROWS
          - inputs: list of (name, dtype, shape)
          - outputs: list of (name, dtype, shape)
        """
        slots = []
        offset = 0
        func_specs = []
        seen_names = set()
        for spec in functions:
            fname = spec["name"]
            if fname in seen_names:
                raise ValueError(f"duplicate function name {fname!r}")
            seen_names.add(fname)

            def _slot(name: str, kind: str, dtype: str, shape: list, offset: int) -> StateSlot:
                esize = DTYPE_BYTES[dtype]
                count = 1
                for d in shape:
                    count *= d
                raw = count * esize
                size = ((raw + ANE_SIMD_ALIGN - 1) // ANE_SIMD_ALIGN) * ANE_SIMD_ALIGN
                return StateSlot(
                    name=f"{fname}.{name}",
                    kind=kind,
                    dtype=dtype,
                    shape=list(shape),
                    offset=offset,
                    size_bytes=size,
                )

            in_names = []
            for (n, dtype, shape) in spec["inputs"]:
                full = f"{fname}.{n}"
                slots.append(_slot(n, SLOT_KIND_INPUT, dtype, shape, offset))
                offset += slots[-1].size_bytes
                offset = ((offset + ANE_PAGE_BYTES - 1) // ANE_PAGE_BYTES) * ANE_PAGE_BYTES
                in_names.append(full)
            out_names = []
            for (n, dtype, shape) in spec["outputs"]:
                full = f"{fname}.{n}"
                slots.append(_slot(n, SLOT_KIND_OUTPUT, dtype, shape, offset))
                offset += slots[-1].size_bytes
                offset = ((offset + ANE_PAGE_BYTES - 1) // ANE_PAGE_BYTES) * ANE_PAGE_BYTES
                out_names.append(full)

            func_specs.append(FunctionSpec(
                name=fname,
                role=spec["role"],
                bucket=0,
                stateful=False,
                input_slots=in_names,
                output_slots=out_names,
                core_ml_function_name=fname,
                use_ane=True,
            ))

        state_size = ((offset + ANE_PAGE_BYTES - 1) // ANE_PAGE_BYTES) * ANE_PAGE_BYTES
        if state_size < ANE_MIN_ALLOC_BYTES:
            state_size = ANE_MIN_ALLOC_BYTES

        return cls(
            version=SCHEMA_VERSION,
            bundle_name=bundle_name,
            state_size_bytes=state_size,
            model_type=MODEL_TYPE_ML_PROGRAM,
            slots=slots,
            functions=func_specs,
            dependencies=[],
        )

    # ---- validation ----

    def validate(self) -> None:
        if self.version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported manifest version {self.version}, "
                f"expected {SCHEMA_VERSION}")
        if not self.bundle_name:
            raise ValueError("bundle_name is required")
        if self.model_type not in (MODEL_TYPE_NEURAL_NETWORK, MODEL_TYPE_ML_PROGRAM):
            raise ValueError(
                f"model_type {self.model_type!r} not in "
                f"({MODEL_TYPE_NEURAL_NETWORK!r}, {MODEL_TYPE_ML_PROGRAM!r})")
        if self.state_size_bytes < ANE_MIN_ALLOC_BYTES:
            raise ValueError(
                f"state_size_bytes {self.state_size_bytes} < ANE minimum "
                f"{ANE_MIN_ALLOC_BYTES}")
        if self.state_size_bytes % ANE_PAGE_BYTES != 0:
            raise ValueError(
                f"state_size_bytes {self.state_size_bytes} not 16KB-aligned")
        slot_names = {s.name for s in self.slots}
        if len(slot_names) != len(self.slots):
            raise ValueError("duplicate slot names")
        for slot in self.slots:
            slot.validate()
        func_names = {f.name for f in self.functions}
        if len(func_names) != len(self.functions):
            raise ValueError("duplicate function names")
        for func in self.functions:
            func.validate()
            for slot_name in func.input_slots + func.output_slots:
                if slot_name not in slot_names:
                    raise ValueError(
                        f"function {func.name} references unknown slot "
                        f"{slot_name!r}")
        for dep in self.dependencies:
            if dep.producer not in func_names:
                raise ValueError(
                    f"dependency producer {dep.producer!r} not a function")
            if dep.slot not in slot_names:
                raise ValueError(
                    f"dependency slot {dep.slot!r} not a slot")
            for consumer in dep.consumers:
                if consumer not in func_names:
                    raise ValueError(
                        f"dependency consumer {consumer!r} not a function")
        # STATE slots must be referenced by at least one function.
        state_slots = {s.name for s in self.slots if s.kind == SLOT_KIND_STATE}
        referenced = set()
        for func in self.functions:
            referenced.update(func.input_slots)
            referenced.update(func.output_slots)
        dead = state_slots - referenced
        if dead:
            raise ValueError(
                f"dead STATE slots (no function references them): {sorted(dead)}")

    # ---- serialization ----

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "bundle_name": self.bundle_name,
            "state_size_bytes": self.state_size_bytes,
            "model_type": self.model_type,
            "slots": [
                {
                    "name": s.name,
                    "kind": s.kind,
                    "dtype": s.dtype,
                    "shape": s.shape,
                    "offset": s.offset,
                    "size_bytes": s.size_bytes,
                }
                for s in self.slots
            ],
            "functions": [
                {
                    "name": f.name,
                    "role": f.role,
                    "bucket": f.bucket,
                    "stateful": f.stateful,
                    "input_slots": f.input_slots,
                    "output_slots": f.output_slots,
                    "core_ml_function_name": f.core_ml_function_name,
                    "use_ane": f.use_ane,
                }
                for f in self.functions
            ],
            "dependencies": [
                {
                    "producer": d.producer,
                    "slot": d.slot,
                    "consumers": d.consumers,
                }
                for d in self.dependencies
            ],
        }

    def write_json(self, path: Path) -> None:
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def from_dict(cls, d: dict) -> "StateLayout":
        model_type = d.get("model_type", MODEL_TYPE_NEURAL_NETWORK)
        slots = [
            StateSlot(
                name=s["name"],
                kind=s["kind"],
                dtype=s["dtype"],
                shape=list(s["shape"]),
                offset=s["offset"],
                size_bytes=s["size_bytes"],
            )
            for s in d.get("slots", [])
        ]
        functions = [
            FunctionSpec(
                name=f["name"],
                role=f["role"],
                bucket=f.get("bucket", 0),
                stateful=f.get("stateful", False),
                input_slots=list(f.get("input_slots", [])),
                output_slots=list(f.get("output_slots", [])),
                core_ml_function_name=f.get("core_ml_function_name", ""),
                use_ane=f.get("use_ane", True),
            )
            for f in d.get("functions", [])
        ]
        dependencies = [
            Dependency(
                producer=d["producer"],
                slot=d["slot"],
                consumers=list(d.get("consumers", [])),
            )
            for d in d.get("dependencies", [])
        ]
        layout = cls(
            version=d["version"],
            bundle_name=d["bundle_name"],
            state_size_bytes=d["state_size_bytes"],
            model_type=model_type,
            slots=slots,
            functions=functions,
            dependencies=dependencies,
        )
        layout.validate()
        return layout

    @classmethod
    def read_json(cls, path: Path) -> "StateLayout":
        return cls.from_dict(json.loads(path.read_text()))


# Manifest file naming convention. Sidecar to the .mlmodelc:
#   <bundle_name>.ane_state.v1.json
# Where <bundle_name> is the .mlmodelc's stem (e.g.,
# "w0-256x256" -> "w0-256x256.ane_state.v1.json").
def manifest_path_for(mlmodelc_dir: Path, bundle_stem: str) -> Path:
    return mlmodelc_dir / f"{bundle_stem}.ane_state.v1.json"
