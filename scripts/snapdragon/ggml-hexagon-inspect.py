#!/usr/bin/env python3
"""
ggml-hexagon-inspect.py - Hexagon DSP binary inspection and diagnostic tool.

Inspects Hexagon ELF binaries (libggml-htp-v*.so) for:
  - Register spills (--spills): counts scalar and HVX vector stack spills,
    separating in-loop spills from frame setup/teardown.
  - Soft-float promotions (--promotions): calls to __trunc*/__extend* helpers.
  - Software divides (--swdiv): calls to __hexagon_udivdi3 and related
    integer/float divide and modulo helpers, with source call sites.
  - Function disassembly (--disasm <func>): annotated disassembly showing
    hardware loop bounds, packet boundaries, and spill instructions.
  - Crash address resolution (--addr2line <addr...>): maps hex crash offsets
    to function symbols, offsets, and source lines.
  - CI verification (--strict): fails with non-zero exit if in-loop vector
    spills or DMA worker vector instructions are detected.

Usage:
  # Check spills across all functions or specific operations
  ./scripts/snapdragon/ggml-hexagon-inspect.py --spills
  ./scripts/snapdragon/ggml-hexagon-inspect.py --spills --func "^compute_"
  ./scripts/snapdragon/ggml-hexagon-inspect.py --spills --func "^compute_" --strict

  # Find functions that call software divide helpers
  ./scripts/snapdragon/ggml-hexagon-inspect.py --swdiv
  ./scripts/snapdragon/ggml-hexagon-inspect.py --swdiv --inline --func "^op_cpy$"

  # Disassemble a function with annotated loop and spill markers
  ./scripts/snapdragon/ggml-hexagon-inspect.py --disasm compute_same_shape_div_f32

  # Resolve crash addresses (CLI arguments or piped logcat/FARF logs)
  ./scripts/snapdragon/ggml-hexagon-inspect.py --addr2line 0x51a30 0x5ba54
  adb logcat | ./scripts/snapdragon/ggml-hexagon-inspect.py --addr2line
"""

import argparse
import logging
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Set, Tuple

# Ignore SIGPIPE to handle pipes (e.g. head, grep) gracefully
if hasattr(signal, "SIGPIPE"):
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger("ggml-hexagon-inspect")


class InsnInfo(NamedTuple):
    address: int
    asm_text: str
    is_vec: bool
    is_vspill: bool
    is_sspill: bool
    is_store: bool
    is_load: bool
    in_loop: bool


class LoopStats:
    def __init__(self, loop_type: str, start_addr: int, end_addr: Optional[int] = None, loop_id: int = 0):
        self.loop_id = loop_id
        self.loop_type = loop_type  # "loop0" or "loop1"
        self.start_addr = start_addr
        self.end_addr = end_addr
        self.packet_count = 0
        self.insn_count = 0
        self.vec_insn_count = 0
        self.vspills_st = 0
        self.vspills_ld = 0
        self.sspills_st = 0
        self.sspills_ld = 0

    @property
    def vspills_total(self) -> int:
        return self.vspills_st + self.vspills_ld

    @property
    def sspills_total(self) -> int:
        return self.sspills_st + self.sspills_ld

    @property
    def has_v_roundtrip(self) -> bool:
        return self.vspills_st > 0 and self.vspills_ld > 0

    @property
    def vec_density(self) -> float:
        return (self.vec_insn_count / self.packet_count) if self.packet_count > 0 else 0.0


class FuncStats:
    def __init__(self, name: str, address: int, size: int):
        self.name = name
        self.address = address
        self.size = size
        self.packet_count = 0
        self.insn_count = 0
        self.vec_insn_count = 0
        self.loop_count = 0
        self.vspills_in_loop = 0
        self.vspills_in_loop_st = 0
        self.vspills_in_loop_ld = 0
        self.vspills_total = 0
        self.sspills_in_loop = 0
        self.sspills_in_loop_st = 0
        self.sspills_in_loop_ld = 0
        self.sspills_total = 0
        self.promotions_in_loop = 0
        self.promotions_total = 0
        self.promotion_targets: Dict[str, int] = {}
        self.swdivs_in_loop = 0
        self.swdivs_total = 0
        self.swdiv_sites: List[Tuple[int, str, bool]] = []
        self.calls_in_loop = 0
        self.calls_total = 0
        self.loops: List[LoopStats] = []
        self.insns: List[InsnInfo] = []


class SymbolEntry(NamedTuple):
    address: int
    size: int
    name: str


# Regular expression patterns for Hexagon disassembly parsing
RE_SYMBOL_HEADER = re.compile(r"^([0-9a-fA-F]+)\s+<([^>]+)>:", re.MULTILINE)
RE_INSN_LINE = re.compile(
    r"^\s*([0-9a-fA-F]+):\s+([0-9a-fA-F]{2}(?:\s+[0-9a-fA-F]{2}){3})\s+([0-9a-fA-F]{8})\s*(.*)$"
)
RE_LOOP0_START = re.compile(r"\bloop0\((0x[0-9a-fA-F]+)")
RE_LOOP1_START = re.compile(r"\bloop1\((0x[0-9a-fA-F]+)")
RE_VMEM_BASE = re.compile(r"\bvmemu?\s*\(\s*([a-z0-9]+)\b")
RE_SMEM_BASE = re.compile(r"\bmem[bwhd](?:_locked|_fifo)?\s*\(\s*([a-z0-9]+)\b")
RE_MEM_STORE = re.compile(r"\bv?mem[bwhdu]?(?:_[a-z]+)?\s*\([^)]*\)\s*(\+|-)?=")
RE_ADD_OP = re.compile(r"\b(r[0-9]+)\s*=\s*add\s*\(\s*([^,()]+)\s*,\s*([^,()]+)\s*\)")
RE_ASSIGN_LHS = re.compile(r"^\s*(?:if\s*\([^)]+\)\s*)?(r[0-9]+)(?::(r[0-9]+))?\s*(?:[+\-*/&|^]?=)")
RE_VEC_OP = re.compile(r"\b(v[0-9]+|w[0-9]+|q[0-3]|vmemu?)\b")
RE_PROMOTION_CALL = re.compile(
    r"\b(?:call|jump)\s+(?:0x[0-9a-fA-F]+\s+)?<(__(?:trunc|extend)[a-zA-Z0-9_]+)(?:@plt)?>"
)
RE_SWDIV_CALL = re.compile(
    r"\b(?:call|jump)\s+(?:0x[0-9a-fA-F]+\s+)?<(__hexagon_(?:u?(?:div|mod)[sd]i3|div[sd]f3))(?:@plt)?>"
)
RE_ANY_CALL = re.compile(r"\bcallr?\b")


def is_mem_store(insn: str) -> bool:
    return bool(RE_MEM_STORE.search(insn))


def update_sp_regs(insn: str, sp_regs: Set[str]) -> None:
    # Track registers derived from stack frame (r29/r30)
    m_add = RE_ADD_OP.search(insn)
    if m_add:
        dest = m_add.group(1)
        op1 = m_add.group(2).strip()
        op2 = m_add.group(3).strip()
        if op1 in sp_regs or op2 in sp_regs:
            sp_regs.add(dest)
            return

    m_assign = RE_ASSIGN_LHS.match(insn.strip())
    if m_assign:
        r1 = m_assign.group(1)
        r2 = m_assign.group(2)
        if r1 and r1 not in ("r29", "r30"):
            sp_regs.discard(r1)
        if r2 and r2 not in ("r29", "r30"):
            sp_regs.discard(r2)


def get_repo_root() -> Path:
    # Resolve repository root from script location
    return Path(__file__).resolve().parent.parent.parent


def extract_arch_num(p: Path) -> int:
    # Extract integer architecture version (e.g. v81 -> 81)
    m = re.search(r"-v([0-9]+)\.so$", p.name)
    return int(m.group(1)) if m else 0


def find_default_lib(repo_root: Path, arch_filter: Optional[str] = None) -> Optional[Path]:
    # Search for built Hexagon shared libraries in build and pkg directories
    candidates = []
    search_dirs = [
        repo_root / "build-adb" / "ggml" / "src" / "ggml-hexagon",
        repo_root / "build-android" / "ggml" / "src" / "ggml-hexagon",
        repo_root / "build-ubuntu" / "ggml" / "src" / "ggml-hexagon",
        repo_root / "build-linux" / "ggml" / "src" / "ggml-hexagon",
        repo_root / "pkg-adb" / "llama.cpp" / "lib",
        repo_root / "pkg-android" / "llama.cpp" / "lib",
        repo_root / "pkg-ubuntu" / "llama.cpp" / "lib",
    ]

    arch_needle = None
    if arch_filter:
        arch_needle = arch_filter if arch_filter.startswith("v") else f"v{arch_filter}"

    for d in search_dirs:
        if not d.is_dir():
            continue
        for p in d.glob("libggml-htp-*.so"):
            if arch_needle and arch_needle not in p.name:
                continue
            candidates.append(p)

    if not candidates:
        for p in repo_root.glob("build-*/ggml/src/ggml-hexagon/libggml-htp-*.so"):
            if arch_needle and arch_needle not in p.name:
                continue
            candidates.append(p)

    if not candidates:
        return None

    # Group latest build candidates (within 60s of max mtime) and pick highest arch
    max_mtime = max(p.stat().st_mtime for p in candidates)
    recent = [p for p in candidates if max_mtime - p.stat().st_mtime <= 60]
    recent.sort(key=lambda p: extract_arch_num(p), reverse=True)
    return recent[0]


def translate_container_arg(arg: str, repo_root: Path) -> str:
    # Do not translate non-path command flags
    if arg.startswith("-") and "=" not in arg:
        return arg
    if arg.startswith("--") and "=" in arg:
        flag, val = arg.split("=", 1)
        return f"{flag}={translate_container_arg(val, repo_root)}"
    try:
        p = Path(arg)
        if (p.is_absolute() and p.exists()) or (p.exists() and ("/" in arg or "\\" in arg)):
            resolved = p.resolve()
            if resolved.is_relative_to(repo_root):
                rel = resolved.relative_to(repo_root)
                return f"/workspace/{rel.as_posix()}"
    except Exception:
        pass
    return arg


class HexagonToolchain:
    def __init__(
        self,
        repo_root: Path,
        use_docker: bool = False,
        image_url: str = "ghcr.io/snapdragon-toolchain",
        image_name: str = "arm64-android",
        image_ver: str = "v0.7",
    ):
        self.repo_root = repo_root
        self.image = f"{image_url}/{image_name}:{image_ver}"
        self.docker_bin = shutil.which("docker")
        self.use_docker = use_docker

        if not use_docker:
            self.native_objdump, self.native_addr2line = self._discover_native_tools()
        else:
            self.native_objdump = None
            self.native_addr2line = None

        if not self.native_objdump and not self.native_addr2line:
            self.use_docker = True

    def _discover_native_tools(self) -> Tuple[Optional[str], Optional[str]]:
        # Check system PATH
        objdump = shutil.which("hexagon-llvm-objdump")
        addr2line = shutil.which("hexagon-addr2line") or shutil.which("hexagon-llvm-addr2line")

        # Check HEXAGON_TOOLS_ROOT environment variable
        tools_root = os.environ.get("HEXAGON_TOOLS_ROOT")
        if tools_root:
            bin_dir = Path(tools_root) / "Tools" / "bin"
            objdump_path = bin_dir / "hexagon-llvm-objdump"
            addr2line_path = bin_dir / "hexagon-addr2line"
            if objdump_path.is_file() and not objdump:
                objdump = str(objdump_path)
            if addr2line_path.is_file() and not addr2line:
                addr2line = str(addr2line_path)

        # Check HEXAGON_SDK_ROOT environment variable
        sdk_root = os.environ.get("HEXAGON_SDK_ROOT")
        if sdk_root:
            tools_parent = Path(sdk_root) / "tools" / "HEXAGON_Tools"
            if tools_parent.is_dir():
                for t_dir in tools_parent.iterdir():
                    bin_dir = t_dir / "Tools" / "bin"
                    objdump_path = bin_dir / "hexagon-llvm-objdump"
                    addr2line_path = bin_dir / "hexagon-addr2line"
                    if objdump_path.is_file() and not objdump:
                        objdump = str(objdump_path)
                    if addr2line_path.is_file() and not addr2line:
                        addr2line = str(addr2line_path)

        return objdump, addr2line

    def run_tool(self, tool_name: str, args: List[str], stdin_data: Optional[str] = None) -> str:
        # Execute tool either natively or inside Docker container
        if not self.use_docker:
            tool_path = self.native_objdump if "objdump" in tool_name else self.native_addr2line
            if not tool_path:
                tool_path = shutil.which(tool_name)
            if not tool_path:
                raise RuntimeError(f"Tool {tool_name} not found natively. Use Docker instead.")

            cmd = [tool_path] + args
            res = subprocess.run(cmd, capture_output=True, text=True, input=stdin_data)
            if res.returncode != 0:
                raise RuntimeError(f"Tool {tool_name} failed: {res.stderr.strip()}")
            return res.stdout

        # Running via Docker container
        if not self.docker_bin:
            raise RuntimeError("Docker is required but not installed or found on PATH.")

        container_tools_dir = "/opt/hexagon/6.6.0.0/tools/HEXAGON_Tools/19.0.07/Tools/bin"
        if "objdump" in tool_name:
            container_tool = f"{container_tools_dir}/hexagon-llvm-objdump"
        elif "addr2line" in tool_name:
            container_tool = f"{container_tools_dir}/hexagon-addr2line"
        elif "nm" in tool_name:
            container_tool = f"{container_tools_dir}/llvm-nm"
        else:
            container_tool = f"{container_tools_dir}/{tool_name}"

        # Translate file paths from host to /workspace
        translated_args = [translate_container_arg(arg, self.repo_root) for arg in args]

        docker_cmd = [
            "docker",
            "run",
            "--rm",
            "--platform",
            "linux/amd64",
            "-v",
            f"{self.repo_root}:/workspace",
            "-w",
            "/workspace",
        ]

        if platform.system() != "Windows":
            docker_cmd += ["-u", f"{os.getuid()}:{os.getgid()}"]

        docker_cmd += [self.image, container_tool] + translated_args

        res = subprocess.run(docker_cmd, capture_output=True, text=True, input=stdin_data)
        if res.returncode != 0:
            raise RuntimeError(f"Docker command failed: {res.stderr.strip()}")
        return res.stdout


def parse_symbols(toolchain: HexagonToolchain, lib_path: Path) -> List[SymbolEntry]:
    # Parse function symbols from library using objdump -t
    output = toolchain.run_tool("hexagon-llvm-objdump", ["-t", str(lib_path)])
    sym_re = re.compile(r"^([0-9a-fA-F]+)\s+[lgw! ]+\s+F\s+\.text\s+([0-9a-fA-F]+)\s+(.+)$")

    symbols = []
    for line in output.splitlines():
        m = sym_re.match(line.strip())
        if m:
            addr = int(m.group(1), 16)
            size = int(m.group(2), 16)
            name = m.group(3).strip()
            symbols.append(SymbolEntry(addr, size, name))

    symbols.sort(key=lambda s: s.address)
    return symbols


def find_enclosing_symbol(symbols: List[SymbolEntry], address: int) -> Optional[Tuple[str, int]]:
    # Binary search enclosing function symbol and compute offset
    low = 0
    high = len(symbols) - 1
    best = None

    while low <= high:
        mid = (low + high) // 2
        s = symbols[mid]
        if s.address <= address:
            if address < s.address + s.size:
                return (s.name, address - s.address)
            best = s
            low = mid + 1
        else:
            high = mid - 1

    if best and address < best.address + best.size:
        return (best.name, address - best.address)
    return None


def parse_disassembly(
    disasm_text: str, func_filter: Optional[re.Pattern] = None
) -> List[FuncStats]:
    # Parse disassembly text into structured function statistics
    matches = list(RE_SYMBOL_HEADER.finditer(disasm_text))
    funcs: List[FuncStats] = []

    for i, m in enumerate(matches):
        name = m.group(2)
        if func_filter and not func_filter.search(name):
            continue

        addr = int(m.group(1), 16)
        start_idx = m.end()
        end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(disasm_text)
        chunk = disasm_text[start_idx:end_idx]

        stats = FuncStats(name=name, address=addr, size=0)

        loop0_target: Optional[int] = None
        loop1_target: Optional[int] = None
        loop0_active = False
        loop1_active = False
        current_loop0: Optional[LoopStats] = None
        current_loop1: Optional[LoopStats] = None
        sp_regs: Set[str] = {"r29", "r30"}

        first_addr = None
        last_addr = None

        for raw_line in chunk.splitlines():
            lm = RE_INSN_LINE.match(raw_line)
            if not lm:
                continue

            cur_addr = int(lm.group(1), 16)
            asm_chunk = lm.group(4)

            if first_addr is None:
                first_addr = cur_addr
            last_addr = cur_addr

            # Track packet count
            if "{" in asm_chunk:
                stats.packet_count += 1
                if current_loop0:
                    current_loop0.packet_count += 1
                if current_loop1:
                    current_loop1.packet_count += 1

            # Check loop starts
            m0 = RE_LOOP0_START.search(asm_chunk)
            if m0:
                loop0_target = int(m0.group(1), 16)
                stats.loop_count += 1

            m1 = RE_LOOP1_START.search(asm_chunk)
            if m1:
                loop1_target = int(m1.group(1), 16)
                stats.loop_count += 1

            if loop0_target is not None and cur_addr >= loop0_target:
                loop0_active = True
                if current_loop0 is None:
                    current_loop0 = LoopStats(
                        loop_id=len(stats.loops) + 1,
                        loop_type="loop0",
                        start_addr=loop0_target,
                        end_addr=0,
                    )

            if loop1_target is not None and cur_addr >= loop1_target:
                loop1_active = True
                if current_loop1 is None:
                    current_loop1 = LoopStats(
                        loop_id=len(stats.loops) + 1,
                        loop_type="loop1",
                        start_addr=loop1_target,
                        end_addr=0,
                    )

            in_loop = loop0_active or loop1_active

            # Parse instructions within packet line
            cleaned = re.sub(r"[{}\s]|:endloop[01]", " ", asm_chunk)
            sub_insns = [p.strip() for p in cleaned.split(";") if p.strip()]

            for insn in sub_insns:
                update_sp_regs(insn, sp_regs)

                stats.insn_count += 1
                if current_loop0:
                    current_loop0.insn_count += 1
                if current_loop1:
                    current_loop1.insn_count += 1

                is_vec = bool(RE_VEC_OP.search(insn))
                if is_vec:
                    stats.vec_insn_count += 1
                    if current_loop0:
                        current_loop0.vec_insn_count += 1
                    if current_loop1:
                        current_loop1.vec_insn_count += 1

                vm = RE_VMEM_BASE.search(insn)
                is_vspill = bool(vm and vm.group(1) in sp_regs)

                sm = RE_SMEM_BASE.search(insn)
                is_sspill = bool(sm and sm.group(1) in sp_regs)

                is_store = False
                is_load = False
                if is_vspill or is_sspill:
                    is_store = is_mem_store(insn)
                    is_load = not is_store

                if is_vspill:
                    stats.vspills_total += 1
                    if in_loop:
                        stats.vspills_in_loop += 1
                        if is_store:
                            stats.vspills_in_loop_st += 1
                        else:
                            stats.vspills_in_loop_ld += 1
                    if current_loop0:
                        if is_store:
                            current_loop0.vspills_st += 1
                        else:
                            current_loop0.vspills_ld += 1
                    if current_loop1:
                        if is_store:
                            current_loop1.vspills_st += 1
                        else:
                            current_loop1.vspills_ld += 1
                elif is_sspill:
                    stats.sspills_total += 1
                    if in_loop:
                        stats.sspills_in_loop += 1
                        if is_store:
                            stats.sspills_in_loop_st += 1
                        else:
                            stats.sspills_in_loop_ld += 1
                    if current_loop0:
                        if is_store:
                            current_loop0.sspills_st += 1
                        else:
                            current_loop0.sspills_ld += 1
                    if current_loop1:
                        if is_store:
                            current_loop1.sspills_st += 1
                        else:
                            current_loop1.sspills_ld += 1

                is_call = bool(RE_ANY_CALL.search(insn))
                prom_m = RE_PROMOTION_CALL.search(insn)
                if is_call:
                    stats.calls_total += 1
                    if in_loop:
                        stats.calls_in_loop += 1
                if prom_m:
                    stats.promotions_total += 1
                    ptarget = prom_m.group(1)
                    stats.promotion_targets[ptarget] = stats.promotion_targets.get(ptarget, 0) + 1
                    if in_loop:
                        stats.promotions_in_loop += 1
                swdiv_m = RE_SWDIV_CALL.search(insn)
                if swdiv_m:
                    stats.swdivs_total += 1
                    stats.swdiv_sites.append((cur_addr, swdiv_m.group(1), in_loop))
                    if in_loop:
                        stats.swdivs_in_loop += 1

                stats.insns.append(
                    InsnInfo(
                        address=cur_addr,
                        asm_text=insn,
                        is_vec=is_vec,
                        is_vspill=is_vspill,
                        is_sspill=is_sspill,
                        is_store=is_store,
                        is_load=is_load,
                        in_loop=in_loop,
                    )
                )

            # Check loop ends
            if ":endloop0" in asm_chunk:
                loop0_active = False
                loop0_target = None
                if current_loop0:
                    current_loop0.end_addr = cur_addr
                    stats.loops.append(current_loop0)
                    current_loop0 = None

            if ":endloop1" in asm_chunk:
                loop1_active = False
                loop1_target = None
                if current_loop1:
                    current_loop1.end_addr = cur_addr
                    stats.loops.append(current_loop1)
                    current_loop1 = None

        if current_loop0:
            current_loop0.end_addr = last_addr or 0
            stats.loops.append(current_loop0)
        if current_loop1:
            current_loop1.end_addr = last_addr or 0
            stats.loops.append(current_loop1)

        stats.loops.sort(key=lambda lp: lp.start_addr)
        for idx, loop in enumerate(stats.loops, 1):
            loop.loop_id = idx

        if first_addr is not None and last_addr is not None:
            stats.size = (last_addr - first_addr) + 4

        funcs.append(stats)

    return funcs


def annotate_disasm_line(
    raw_line: str,
    loop0_target: Optional[int],
    loop1_target: Optional[int],
    loop0_active: bool,
    loop1_active: bool,
    use_color: bool = True,
    sp_regs: Optional[Set[str]] = None,
) -> Tuple[str, Optional[int], Optional[int], bool, bool, bool]:
    # Annotate disassembly line with spill and loop tags
    lm = RE_INSN_LINE.match(raw_line)
    if not lm:
        return raw_line, loop0_target, loop1_target, loop0_active, loop1_active, False

    cur_addr = int(lm.group(1), 16)
    asm_chunk = lm.group(4)
    is_event = False

    if sp_regs is None:
        sp_regs = {"r29", "r30"}

    # Check loop starts
    m0 = RE_LOOP0_START.search(asm_chunk)
    if m0:
        loop0_target = int(m0.group(1), 16)
    m1 = RE_LOOP1_START.search(asm_chunk)
    if m1:
        loop1_target = int(m1.group(1), 16)

    if loop0_target is not None and cur_addr >= loop0_target:
        loop0_active = True
    if loop1_target is not None and cur_addr >= loop1_target:
        loop1_active = True

    in_loop = loop0_active or loop1_active

    tags = []
    if m0:
        tags.append("[LOOP0-START]")
        is_event = True
    if m1:
        tags.append("[LOOP1-START]")
        is_event = True

    cleaned = re.sub(r"[{}\s]|:endloop[01]", " ", asm_chunk)
    sub_insns = [p.strip() for p in cleaned.split(";") if p.strip()]

    for insn in sub_insns:
        update_sp_regs(insn, sp_regs)

    for insn in sub_insns:
        vm = RE_VMEM_BASE.search(insn)
        if vm and vm.group(1) in sp_regs:
            base = vm.group(1)
            is_st = is_mem_store(insn)
            op = "STORE" if is_st else "LOAD"
            tgt = f"({base})" if base not in ("r29", "r30") else ""
            if in_loop:
                tag = f"[V-SPILL:{op}{tgt}:IN-LOOP]"
                tags.append(f"\033[1;31m{tag}\033[0m" if use_color else tag)
            else:
                tag = f"[V-SPILL:{op}{tgt}]"
                tags.append(f"\033[1;33m{tag}\033[0m" if use_color else tag)
            is_event = True

        sm = RE_SMEM_BASE.search(insn)
        if sm and sm.group(1) in sp_regs:
            base = sm.group(1)
            is_st = is_mem_store(insn)
            op = "STORE" if is_st else "LOAD"
            tgt = f"({base})" if base not in ("r29", "r30") else ""
            if in_loop:
                tag = f"[S-SPILL:{op}{tgt}:IN-LOOP]"
                tags.append(f"\033[1;35m{tag}\033[0m" if use_color else tag)
            else:
                tag = f"[S-SPILL:{op}{tgt}]"
                tags.append(f"\033[0;35m{tag}\033[0m" if use_color else tag)
            is_event = True

    prom_m = RE_PROMOTION_CALL.search(asm_chunk)
    swdiv_m = RE_SWDIV_CALL.search(asm_chunk)
    if prom_m:
        ptarget = prom_m.group(1)
        if in_loop:
            tag = f"[PROMOTION:{ptarget}:IN-LOOP]"
            tags.append(f"\033[1;31m{tag}\033[0m" if use_color else tag)
        else:
            tag = f"[PROMOTION:{ptarget}]"
            tags.append(f"\033[1;35m{tag}\033[0m" if use_color else tag)
        is_event = True
    elif swdiv_m:
        dtarget = swdiv_m.group(1)
        if in_loop:
            tag = f"[SW-DIV:{dtarget}:IN-LOOP]"
            tags.append(f"\033[1;31m{tag}\033[0m" if use_color else tag)
        else:
            tag = f"[SW-DIV:{dtarget}]"
            tags.append(f"\033[1;35m{tag}\033[0m" if use_color else tag)
        is_event = True
    elif RE_ANY_CALL.search(asm_chunk):
        if in_loop:
            tag = "[CALL:IN-LOOP]"
            tags.append(f"\033[1;31m{tag}\033[0m" if use_color else tag)
            is_event = True
        else:
            tag = "[CALL]"
            tags.append(f"\033[1;36m{tag}\033[0m" if use_color else tag)

    if ":endloop0" in asm_chunk:
        tags.append("[LOOP0-END]")
        loop0_active = False
        loop0_target = None
        is_event = True
    if ":endloop1" in asm_chunk:
        tags.append("[LOOP1-END]")
        loop1_active = False
        loop1_target = None
        is_event = True

    tag_str = " ".join(tags)
    if tag_str:
        annotated = f"{raw_line:<80}  {tag_str}"
    else:
        annotated = raw_line

    return annotated, loop0_target, loop1_target, loop0_active, loop1_active, is_event


def run_spills(
    toolchain: HexagonToolchain,
    lib_path: Path,
    args: argparse.Namespace,
) -> int:
    # Scan and report register spills across binary functions
    logger.info(f"Inspecting library: {lib_path}")
    disasm_text = toolchain.run_tool("hexagon-llvm-objdump", ["-d", str(lib_path)])

    func_re = re.compile(args.func) if args.func else None
    funcs = parse_disassembly(disasm_text, func_re)

    # Filter functions
    reported = []
    for f in funcs:
        has_spills = f.vspills_total > 0 or f.sspills_in_loop > 0 or f.sspills_total > 0
        if args.all or args.func or has_spills:
            reported.append(f)

    # Sort: in-loop vector spills desc, then total vector spills desc, then in-loop scalar spills desc
    reported.sort(
        key=lambda x: (x.vspills_in_loop, x.vspills_total, x.sspills_in_loop, x.sspills_total),
        reverse=True,
    )

    use_color = not args.no_color and sys.stdout.isatty()

    # Print summary table
    col_addr = "Address"
    col_name = "Function"
    col_pkts = "Packets"
    col_insn = "Insns"
    col_vec = "HVX Ops"
    col_vloop = "V-Loop (st/ld)"
    col_vtot = "V-Tot"
    col_sloop = "S-Loop (st/ld)"
    col_stot = "S-Tot"
    col_notes = "Notes"

    name_w = max([40] + [len(f.name) for f in reported])
    hdr = (
        f"{col_addr:<10} | {col_name:<{name_w}} | {col_pkts:>7} | {col_insn:>6} | "
        f"{col_vec:>7} | {col_vloop:>14} | {col_vtot:>5} | {col_sloop:>14} | {col_stot:>5} | {col_notes}"
    )
    sep = "-" * len(hdr)

    logger.info("\n" + sep)
    logger.info(hdr)
    logger.info(re.sub(r"[^|]", "-", hdr))

    tot_vloop = 0
    tot_sloop = 0
    tot_funcs_with_vloop = 0
    strict_violations = []

    dma_re: Optional[re.Pattern[str]] = re.compile(args.dma_pattern) if args.dma_pattern else None

    for f in reported:
        tot_vloop += f.vspills_in_loop
        tot_sloop += f.sspills_in_loop
        if f.vspills_in_loop > 0:
            tot_funcs_with_vloop += 1

        # Check strict criteria
        if args.strict:
            inloop_v = f.vspills_in_loop_st if getattr(args, "strict_stores_only", False) else f.vspills_in_loop
            if inloop_v > args.max_inloop_vspills:
                lbl = "in-loop vector store spills" if getattr(args, "strict_stores_only", False) else "in-loop vector spills"
                strict_violations.append(
                    f"{f.name}: {inloop_v} {lbl} (max allowed: {args.max_inloop_vspills})"
                )
            if dma_re and dma_re.search(f.name):
                if f.vec_insn_count > args.max_dma_vec_ops:
                    strict_violations.append(
                        f"{f.name}: DMA worker contains {f.vec_insn_count} HVX vector ops (max allowed: {args.max_dma_vec_ops})"
                    )

        vloop_detail = f"{f.vspills_in_loop} ({f.vspills_in_loop_st}s,{f.vspills_in_loop_ld}l)" if f.vspills_in_loop > 0 else "0"
        sloop_detail = f"{f.sspills_in_loop} ({f.sspills_in_loop_st}s,{f.sspills_in_loop_ld}l)" if f.sspills_in_loop > 0 else "0"

        notes = ""
        if f.vspills_in_loop_st > 0 and f.vspills_in_loop_ld > 0:
            notes = "\033[1;31m[V-ROUNDTRIP!]\033[0m" if use_color else "[V-ROUNDTRIP!]"
        elif f.vspills_in_loop_st == 0 and f.vspills_in_loop_ld > 0:
            notes = "v-readonly"

        vloop_str = f"{vloop_detail:>14}"
        if f.vspills_in_loop > 0 and use_color:
            if f.vspills_in_loop_st > 0 and f.vspills_in_loop_ld > 0:
                vloop_str = f"\033[1;31m{vloop_str}\033[0m"
            else:
                vloop_str = f"\033[1;33m{vloop_str}\033[0m"

        sloop_str = f"{sloop_detail:>14}"

        logger.info(
            f"0x{f.address:08x} | {f.name:<{name_w}} | {f.packet_count:>7} | {f.insn_count:>6} | "
            f"{f.vec_insn_count:>7} | {vloop_str} | {f.vspills_total:>5} | {sloop_str} | {f.sspills_total:>5} | {notes}"
        )

    logger.info(sep)
    logger.info(
        f"Total functions analyzed: {len(funcs)} | Reported: {len(reported)} | "
        f"Functions with in-loop vector spills: {tot_funcs_with_vloop} | "
        f"Total in-loop vector spills: {tot_vloop} | Total in-loop scalar spills: {tot_sloop}"
    )

    if args.strict:
        logger.info("\n" + "=" * 50)
        if strict_violations:
            if use_color:
                logger.error("\033[1;31mSTRICT CHECK FAILED\033[0m")
            else:
                logger.error("STRICT CHECK FAILED")
            for v in strict_violations:
                logger.error(f"  - {v}")
            logger.info("=" * 50)
            return 1
        else:
            if use_color:
                logger.info("\033[1;32mSTRICT CHECK PASSED: 0 violations\033[0m")
            else:
                logger.info("STRICT CHECK PASSED: 0 violations")
            logger.info("=" * 50)

    return 0


def run_promotions(
    toolchain: HexagonToolchain,
    lib_path: Path,
    args: argparse.Namespace,
) -> int:
    # Scan and report soft-float promotion calls across binary functions
    logger.info(f"Inspecting library: {lib_path}")
    disasm_text = toolchain.run_tool("hexagon-llvm-objdump", ["-d", str(lib_path)])

    func_re = re.compile(args.func) if args.func else None
    funcs = parse_disassembly(disasm_text, func_re)

    reported = []
    for f in funcs:
        if args.all or f.promotions_total > 0:
            reported.append(f)

    # Sort: in-loop promotions desc, then total promotions desc
    reported.sort(
        key=lambda x: (x.promotions_in_loop, x.promotions_total),
        reverse=True,
    )

    use_color = not args.no_color and sys.stdout.isatty()

    col_addr = "Address"
    col_name = "Function"
    col_loop = "Loops"
    col_inloop = "In-Loop"
    col_tot = "Total"
    col_targets = "Promotion Targets"

    name_w = max([40] + [len(f.name) for f in reported])
    hdr = f"{col_addr:<10} | {col_name:<{name_w}} | {col_loop:>5} | {col_inloop:>7} | {col_tot:>5} | {col_targets}"
    sep = "-" * max(len(hdr), 110)

    logger.info("\n" + sep)
    logger.info(hdr)
    logger.info(re.sub(r"[^|]", "-", hdr).ljust(len(sep), "-"))

    tot_inloop = 0
    tot_prom = 0
    tot_funcs_with_prom = 0
    strict_violations = []

    for f in reported:
        tot_inloop += f.promotions_in_loop
        tot_prom += f.promotions_total
        if f.promotions_total > 0:
            tot_funcs_with_prom += 1

        if args.strict:
            max_p = args.max_promotions if args.max_promotions is not None else 0
            if f.promotions_total > max_p:
                strict_violations.append(
                    f"{f.name}: {f.promotions_total} float promotion calls (max allowed: {max_p})"
                )

        inloop_str = f"{f.promotions_in_loop:>7}"
        if f.promotions_in_loop > 0 and use_color:
            inloop_str = f"\033[1;31m{inloop_str}\033[0m"

        targets_str = ", ".join(f"{t}: {c}" for t, c in sorted(f.promotion_targets.items()))
        logger.info(
            f"0x{f.address:08x} | {f.name:<{name_w}} | {f.loop_count:>5} | {inloop_str} | {f.promotions_total:>5} | {targets_str}"
        )

    logger.info(sep)
    logger.info(
        f"Total functions analyzed: {len(funcs)} | Reported: {len(reported)} | "
        f"Functions with float promotions: {tot_funcs_with_prom} | "
        f"Total promotion calls: {tot_prom} | In-loop: {tot_inloop}"
    )

    if args.strict:
        logger.info("\n" + "=" * 50)
        if strict_violations:
            if use_color:
                logger.error("\033[1;31mSTRICT CHECK FAILED\033[0m")
            else:
                logger.error("STRICT CHECK FAILED")
            for v in strict_violations:
                logger.error(f"  - {v}")
            logger.info("=" * 50)
            return 1
        else:
            if use_color:
                logger.info("\033[1;32mSTRICT CHECK PASSED: 0 violations\033[0m")
            else:
                logger.info("STRICT CHECK PASSED: 0 violations")
            logger.info("=" * 50)

    return 0


def run_swdiv(
    toolchain: HexagonToolchain,
    lib_path: Path,
    args: argparse.Namespace,
) -> int:
    # Scan and report software divide/modulo helper calls across binary functions
    logger.info(f"Inspecting library: {lib_path}")
    disasm_text = toolchain.run_tool("hexagon-llvm-objdump", ["-d", str(lib_path)])

    func_re = re.compile(args.func) if args.func else None
    funcs = parse_disassembly(disasm_text, func_re)

    reported = [f for f in funcs if args.all or f.swdivs_total > 0]

    # Sort: in-loop divides desc, then total divides desc
    reported.sort(key=lambda x: (x.swdivs_in_loop, x.swdivs_total), reverse=True)

    use_color = not args.no_color and sys.stdout.isatty()

    # Resolve call sites to source lines, falling back to function offsets without debug info
    site_chains: Dict[int, List[str]] = {}
    sites = [a for f in reported for a, _, _ in f.swdiv_sites]
    if sites:
        # With -i each address prints its inlined frames innermost first, one block per address;
        # the last frame with a known line is the call site in the reported function itself
        raw = toolchain.run_tool("hexagon-addr2line", ["-e", str(lib_path), "-a", "-i"] + [f"0x{a:x}" for a in sites])
        for block in re.split(r"\n\s*\n", raw.strip()):
            addr_line, *frames = block.strip().splitlines()
            locs = [m for m in (re.match(r"^(.*?):(\d+)(?::\d+)?$", fr.strip()) for fr in frames) if m]
            chain = [f"{os.path.basename(m.group(1))}:{m.group(2)}" for m in locs if m.group(1) != "??" and m.group(2) != "0"]
            if chain:
                site_chains[int(addr_line, 16)] = chain
        if not site_chains:
            logger.info("Note: no source line info in library (build with -g); showing call-site offsets instead.")

    col_addr = "Address"
    col_name = "Function"
    col_inloop = "In-Loop"
    col_tot = "Total"
    col_sites = "Call Sites"

    name_w = max([40] + [len(f.name) for f in reported])
    hdr = f"{col_addr:<10} | {col_name:<{name_w}} | {col_inloop:>7} | {col_tot:>5} | {col_sites}"
    sep = "-" * max(len(hdr), 110)

    logger.info("\n" + sep)
    logger.info(hdr)
    logger.info(re.sub(r"[^|]", "-", hdr).ljust(len(sep), "-"))

    tot_inloop = 0
    tot_divs = 0
    tot_funcs_with_divs = 0

    for f in reported:
        tot_inloop += f.swdivs_in_loop
        tot_divs += f.swdivs_total
        if f.swdivs_total > 0:
            tot_funcs_with_divs += 1

        inloop_str = f"{f.swdivs_in_loop:>7}"
        if f.swdivs_in_loop > 0 and use_color:
            inloop_str = f"\033[1;31m{inloop_str}\033[0m"

        # Several calls can share a source line; list each location once
        locs = [site_chains[a][-1] if a in site_chains else f"+0x{a - f.address:x}" for a, _, _ in f.swdiv_sites]
        sites_str = ", ".join(dict.fromkeys(locs))
        logger.info(
            f"0x{f.address:08x} | {f.name:<{name_w}} | {inloop_str} | {f.swdivs_total:>5} | {sites_str}".rstrip()
        )

        # Per-call detail: helper and full inline chain, innermost (the divide itself) first
        if args.inline:
            helper_w = max(len(re.sub(r"^__hexagon_", "", h)) for _, h, _ in f.swdiv_sites)
            for a, helper, in_loop in f.swdiv_sites:
                chain = " <- ".join(site_chains.get(a, ["?"]))
                loop_tag = " [IN-LOOP]" if in_loop else ""
                if in_loop and use_color:
                    loop_tag = f"\033[1;31m{loop_tag}\033[0m"
                off = f"+0x{a - f.address:x}"
                logger.info(f"    {off:<8} {re.sub(r'^__hexagon_', '', helper):<{helper_w}}  {chain}{loop_tag}")

    logger.info(sep)
    logger.info(
        f"Total functions analyzed: {len(funcs)} | Reported: {len(reported)} | "
        f"Functions with sw divides: {tot_funcs_with_divs} | "
        f"Total sw divide calls: {tot_divs} | In-loop: {tot_inloop}"
    )

    return 0


def run_disasm(
    toolchain: HexagonToolchain,
    lib_path: Path,
    args: argparse.Namespace,
) -> int:
    # Disassemble matching function(s) with annotated loop and spill markers
    func_pattern = args.disasm if args.disasm else (args.func or ".*")
    logger.info(f"Inspecting library: {lib_path}")
    logger.info(f"Disassembling functions matching: '{func_pattern}'\n")

    # Disassemble symbol
    disasm_text = toolchain.run_tool(
        "hexagon-llvm-objdump",
        ["-d", f"--disassemble-symbols={func_pattern}", str(lib_path)],
    )

    # If --disassemble-symbols yielded nothing (e.g. pattern was a regex), dump whole binary and filter
    matches = list(RE_SYMBOL_HEADER.finditer(disasm_text))
    if not matches:
        all_disasm = toolchain.run_tool("hexagon-llvm-objdump", ["-d", str(lib_path)])
        pat = re.compile(func_pattern)
        all_matches = list(RE_SYMBOL_HEADER.finditer(all_disasm))
        matched_symbols = [m.group(2) for m in all_matches if pat.search(m.group(2))]
        if not matched_symbols:
            logger.error(f"Error: No symbols found matching '{func_pattern}'.")
            return 1
        # Re-run with symbol list bounded by limit
        sym_limit = args.limit if hasattr(args, "limit") and args.limit and args.limit > 0 else len(matched_symbols)
        sym_arg = ",".join(matched_symbols[:sym_limit])
        disasm_text = toolchain.run_tool(
            "hexagon-llvm-objdump",
            ["-d", f"--disassemble-symbols={sym_arg}", str(lib_path)],
        )
        matches = list(RE_SYMBOL_HEADER.finditer(disasm_text))

    use_color = not args.no_color and sys.stdout.isatty()

    # Parse and log annotated functions
    for i, m in enumerate(matches):
        name = m.group(2)
        addr = int(m.group(1), 16)
        start_idx = m.end()
        end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(disasm_text)
        chunk = disasm_text[start_idx:end_idx]

        # Parse statistics for this function
        func_stats = parse_disassembly(disasm_text[m.start():end_idx])[0]

        # Log header
        hdr_border = "=" * 80
        logger.info(hdr_border)
        logger.info(f"Function: {name}")
        logger.info(f"Address:  0x{addr:08x} - 0x{addr + func_stats.size:08x} ({func_stats.size} bytes)")
        logger.info(f"Packets:  {func_stats.packet_count} | Instructions: {func_stats.insn_count} | Loops: {func_stats.loop_count}")
        vec_pct = (func_stats.vec_insn_count / func_stats.insn_count * 100.0) if func_stats.insn_count else 0.0
        logger.info(f"HVX Ops:  {func_stats.vec_insn_count} ({vec_pct:.1f}% of instructions)")
        vloop_info = f"{func_stats.vspills_in_loop} ({func_stats.vspills_in_loop_st} st, {func_stats.vspills_in_loop_ld} ld)"
        sloop_info = f"{func_stats.sspills_in_loop} ({func_stats.sspills_in_loop_st} st, {func_stats.sspills_in_loop_ld} ld)"
        logger.info(
            f"Spills:   Vector in-loop: {vloop_info} | Vector total: {func_stats.vspills_total} | "
            f"Scalar in-loop: {sloop_info} | Scalar total: {func_stats.sspills_total}"
        )
        logger.info(
            f"Calls:    Total: {func_stats.calls_total} (in-loop: {func_stats.calls_in_loop}) | "
            f"Float promotions: {func_stats.promotions_total} (in-loop: {func_stats.promotions_in_loop}) | "
            f"SW divides: {func_stats.swdivs_total} (in-loop: {func_stats.swdivs_in_loop})"
        )
        logger.info(hdr_border)

        # Print Loop Breakdown Table if function has loops
        if func_stats.loops:
            loop_hdr = (
                f"{'#':<3} | {'Type':<5} | {'Address Range':<23} | {'Packets':>7} | "
                f"{'HVX Ops':>7} | {'Vec/Pkt':>7} | {'V-Spills (st, ld)':>17} | {'S-Spills (st, ld)':>17} | Notes"
            )
            logger.info(f"\nLoops ({len(func_stats.loops)})")
            logger.info("-" * len(loop_hdr))
            logger.info(loop_hdr)
            logger.info(re.sub(r"[^|]", "-", loop_hdr))
            for loop in func_stats.loops:
                vspill_str = f"{loop.vspills_total} ({loop.vspills_st}s,{loop.vspills_ld}l)"
                sspill_str = f"{loop.sspills_total} ({loop.sspills_st}s,{loop.sspills_ld}l)"
                notes = []
                if loop.has_v_roundtrip:
                    notes.append("\033[1;31m[V-ROUNDTRIP!]\033[0m" if use_color else "[V-ROUNDTRIP!]")
                elif loop.vspills_st == 0 and loop.vspills_ld > 0:
                    notes.append("v-readonly")
                if loop.vec_density >= 1.5:
                    notes.append("\033[1;32mdual-hvx\033[0m" if use_color else "dual-hvx")
                notes_str = ", ".join(notes)
                logger.info(
                    f"{loop.loop_id:<3} | {loop.loop_type:<5} | {f'0x{loop.start_addr:08x} - 0x{loop.end_addr:08x}':<23} | "
                    f"{loop.packet_count:>7} | {loop.vec_insn_count:>7} | {loop.vec_density:>7.2f} | "
                    f"{vspill_str:>17} | {sspill_str:>17} | {notes_str}"
                )
            logger.info("-" * len(loop_hdr) + "\n")

        # Parse lines and annotations
        lines = chunk.splitlines()
        annotated_lines = []
        is_event_list = []
        loop0_target = None
        loop1_target = None
        loop0_active = False
        loop1_active = False
        sp_regs = {"r29", "r30"}

        for line in lines:
            ann_line, loop0_target, loop1_target, loop0_active, loop1_active, is_ev = annotate_disasm_line(
                line, loop0_target, loop1_target, loop0_active, loop1_active, use_color, sp_regs
            )
            annotated_lines.append(ann_line)
            is_event_list.append(is_ev)

        # Filter output if --spills-only
        if getattr(args, "spills_only", False):
            ctx = args.context if args.context is not None else 2
            to_show = [False] * len(annotated_lines)
            for idx, ev in enumerate(is_event_list):
                if ev:
                    for j in range(max(0, idx - ctx), min(len(annotated_lines), idx + ctx + 1)):
                        to_show[j] = True

            if not any(to_show):
                logger.info("  (No spills, promotions, sw divides, or in-loop calls detected in this function)\n")
            else:
                in_gap = False
                for idx, show in enumerate(to_show):
                    if show:
                        in_gap = False
                        logger.info(annotated_lines[idx])
                    else:
                        if not in_gap:
                            logger.info("      ...")
                            in_gap = True
                logger.info("")
        else:
            for ann_line in annotated_lines:
                logger.info(ann_line)
            logger.info("")

    return 0


def extract_addresses_from_input(lines: List[str]) -> List[int]:
    # Extract hex program counter addresses from input lines
    re_pc = re.compile(r"\b(?:pc|PC|ip|IP)\s*(?:=|:|\s)\s*0*(?:0x)?([0-9a-fA-F]{3,8})\b")
    re_plus_hex = re.compile(r"\+0x([0-9a-fA-F]{3,8})\b")
    re_hex = re.compile(r"\b0x([0-9a-fA-F]{3,8})\b")
    re_bare_hex = re.compile(r"^\s*0*([0-9a-fA-F]{3,8})\s*$")

    addrs = []
    seen = set()

    for line in lines:
        matched = False
        for m in re_pc.finditer(line):
            val = int(m.group(1), 16)
            if val not in seen:
                seen.add(val)
                addrs.append(val)
            matched = True

        if not matched:
            for m in re_plus_hex.finditer(line):
                val = int(m.group(1), 16)
                if val not in seen:
                    seen.add(val)
                    addrs.append(val)
                matched = True

        if not matched:
            for m in re_hex.finditer(line):
                val = int(m.group(1), 16)
                if val not in seen:
                    seen.add(val)
                    addrs.append(val)
                matched = True

        if not matched:
            m = re_bare_hex.match(line)
            if m:
                val = int(m.group(1), 16)
                if val not in seen:
                    seen.add(val)
                    addrs.append(val)

    return addrs


def run_addr2line(
    toolchain: HexagonToolchain,
    lib_path: Path,
    args: argparse.Namespace,
) -> int:
    # Resolve addresses or crash logs to source locations and symbols
    input_addrs: List[int] = []

    if args.addr2line:
        for arg in args.addr2line:
            if arg == "-":
                continue
            try:
                val = int(arg, 16)
                input_addrs.append(val)
            except ValueError:
                # Treat as text line and search for hex addresses
                input_addrs.extend(extract_addresses_from_input([arg]))

    # Read from stdin if piped or requested via '-'
    if not sys.stdin.isatty() or "-" in (args.addr2line or []):
        stdin_lines = sys.stdin.readlines()
        input_addrs.extend(extract_addresses_from_input(stdin_lines))

    if not input_addrs:
        logger.error("Error: No addresses found to resolve. Provide hex addresses or pipe crash logs to stdin.")
        logger.error("Example: ./scripts/snapdragon/ggml-hexagon-inspect.py --addr2line 0x51a30 0x5ba54")
        return 1

    logger.info(f"Resolving {len(input_addrs)} address(es) against: {lib_path}\n")

    # Load symbol table for symbol + offset fallback
    symbols = parse_symbols(toolchain, lib_path)

    # Format addresses for addr2line tool (prefixed with 0x)
    addr_strs = [f"0x{a:x}" for a in input_addrs]
    tool_args = ["-e", str(lib_path), "-f", "-C", "-p", "-a"] + addr_strs

    raw_output = toolchain.run_tool("hexagon-addr2line", tool_args)

    # Parse output lines
    # Format: 0x51a30: binary_thread_add_id_f32 at /path/file.c:123
    re_out = re.compile(r"^(0x[0-9a-fA-F]+):\s+(.*?)\s+at\s+(.*)$")

    for line in raw_output.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re_out.match(line)
        if m:
            addr_hex = m.group(1)
            addr_val = int(addr_hex, 16)
            func_name = m.group(2)
            src_loc = m.group(3)

            # Check if function name is unknown or generic, look up symbol table
            sym_info = find_enclosing_symbol(symbols, addr_val)
            if sym_info:
                sym_name, sym_offset = sym_info
                sym_display = f"{sym_name}+0x{sym_offset:x}"
            else:
                sym_display = func_name

            logger.info(f"{addr_hex:<12} -> {sym_display:<40} ({src_loc})")
        else:
            logger.info(line)

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Inspect Hexagon DSP binaries for register spills, function disassembly, and crash analysis."
    )

    # Target library
    parser.add_argument(
        "--lib",
        help="Path to Hexagon shared library (e.g. libggml-htp-v81.so). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--arch",
        help="Architecture version filter for auto-detection (e.g. v75, v79, v81).",
    )

    # Modes
    parser.add_argument(
        "--spills",
        action="store_true",
        help="Scan binary and report scalar/vector stack spills table.",
    )
    parser.add_argument(
        "--promotions",
        action="store_true",
        help="Scan binary and report functions with soft-float promotion calls (__trunc*, __extend*).",
    )
    parser.add_argument(
        "--swdiv",
        action="store_true",
        help="Scan binary and report functions with software divide/modulo calls (__hexagon_udivdi3, __hexagon_udivsi3, ...).",
    )
    parser.add_argument(
        "--inline",
        action="store_true",
        help="In --swdiv, list every call under its function with the helper and its inlined source chain.",
    )
    parser.add_argument(
        "--disasm",
        nargs="?",
        const="",
        metavar="FUNC",
        help="Disassemble function symbol or regex pattern with annotated loop and spill markers.",
    )
    parser.add_argument(
        "--spills-only",
        action="store_true",
        help="In --disasm, only display packets containing spills, promotions, sw divides, or in-loop calls, with surrounding context.",
    )
    parser.add_argument(
        "-C",
        "--context",
        type=int,
        default=None,
        metavar="N",
        help="Number of context packets before and after spills in --disasm --spills-only (default: 2).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum symbols to disassemble when using pattern in --disasm (default: 20, 0 for unlimited).",
    )
    parser.add_argument(
        "--addr2line",
        nargs="*",
        metavar="ADDR",
        help="Resolve hex addresses or piped crash traces to symbols and source lines.",
    )

    # Filtering & Display
    parser.add_argument(
        "--func",
        "--fn",
        "-f",
        help="Regex filter for function names in --spills, --promotions, --swdiv, or --disasm.",
    )
    parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="Show all functions in table, even those with 0 spills/promotions/sw divides.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI color output.",
    )

    # Strict check options
    parser.add_argument(
        "--strict",
        action="store_true",
        help="CI mode: exit with non-zero status if violations (in-loop vector spills, DMA worker vector ops) occur.",
    )
    parser.add_argument(
        "--max-inloop-vspills",
        type=int,
        default=0,
        help="Maximum allowed in-loop vector spills in --strict mode (default: 0).",
    )
    parser.add_argument(
        "--strict-stores-only",
        action="store_true",
        help="In --strict mode, only count vector store spills (st > 0) towards violations, ignoring readonly stack loads.",
    )
    parser.add_argument(
        "--max-dma-vec-ops",
        type=int,
        default=0,
        help="Maximum allowed vector instructions in DMA workers in --strict mode (default: 0).",
    )
    parser.add_argument(
        "--max-promotions",
        type=int,
        default=None,
        help="Maximum allowed float promotion calls in --strict mode (default: 0).",
    )
    parser.add_argument(
        "--dma-pattern",
        default=r"^.*_thread(?:_.*)?$",
        help="Regex pattern identifying DMA worker functions (default: '^.*_thread(?:_.*)?$').",
    )

    # Toolchain options
    parser.add_argument(
        "--docker",
        action="store_true",
        help="Force execution inside Docker container.",
    )
    parser.add_argument(
        "--no-docker",
        action="store_true",
        help="Force native execution on host instead of Docker.",
    )
    parser.add_argument(
        "--toolchain-version",
        default="v0.7",
        help="Docker toolchain tag (default: v0.7).",
    )
    parser.add_argument(
        "--toolchain-url",
        default="ghcr.io/snapdragon-toolchain",
        help="Docker toolchain registry (default: ghcr.io/snapdragon-toolchain).",
    )
    parser.add_argument(
        "--image-name",
        default="arm64-android",
        help="Docker toolchain image name (default: arm64-android).",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    repo_root = get_repo_root()

    # Determine target library
    lib_path = None
    if args.lib:
        lib_path = Path(args.lib).resolve()
        if not lib_path.is_file():
            logger.error(f"Error: Specified library '{args.lib}' does not exist.")
            sys.exit(1)
    else:
        lib_path = find_default_lib(repo_root, args.arch)
        if not lib_path:
            logger.error("Error: No Hexagon library found in build-* or pkg-* directories.")
            logger.error("Build the project first via ./scripts/snapdragon/build.py --target adb or specify --lib.")
            sys.exit(1)

    # Initialize toolchain wrapper
    use_docker = args.docker or (not args.no_docker and platform.system() == "Darwin")
    try:
        toolchain = HexagonToolchain(
            repo_root=repo_root,
            use_docker=use_docker,
            image_url=args.toolchain_url,
            image_name=args.image_name,
            image_ver=args.toolchain_version,
        )
    except Exception as e:
        logger.error(f"Error initializing toolchain: {e}")
        sys.exit(1)

    # Dispatch commands
    if args.addr2line is not None:
        sys.exit(run_addr2line(toolchain, lib_path, args))
    elif args.disasm is not None:
        sys.exit(run_disasm(toolchain, lib_path, args))
    elif args.promotions:
        sys.exit(run_promotions(toolchain, lib_path, args))
    elif args.swdiv:
        sys.exit(run_swdiv(toolchain, lib_path, args))
    else:
        # Default action is --spills
        sys.exit(run_spills(toolchain, lib_path, args))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    main()
