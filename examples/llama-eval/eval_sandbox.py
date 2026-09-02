#!/usr/bin/env python3
"""Lightweight sandboxed execution for code-generation evals (HumanEval, ClassEval).

Replaces the usual Docker-per-suite setup with a cached venv plus whatever
process isolation the host can give us without root:

    tier 1  bubblewrap   read-only root, tmpfs /tmp and $HOME, no network
    tier 2  unshare -rn  no network (home still visible)
    tier 3  rlimits only last resort, still capped on cpu/memory/procs/files

Every tier also applies rlimits and runs the candidate program in a throwaway
working directory as its own process group, so a runaway solution is killed
along with anything it spawned.
"""

import json
import os
import re
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import venv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

DEFAULT_CACHE_ROOT = Path(
    os.environ.get("LLAMA_EVAL_CACHE", Path.home() / ".cache" / "llama-eval")
)

# rlimits applied to every candidate program
LIMIT_AS_BYTES = 4 * 1024 * 1024 * 1024  # 4 GiB address space
LIMIT_CPU_SECONDS_SLACK = 5              # cpu limit = wall timeout + slack
LIMIT_NPROC = 256
LIMIT_FSIZE_BYTES = 64 * 1024 * 1024


@dataclass
class ExecResult:
    passed: bool
    status: str           # "ok" | "failed" | "timeout" | "error"
    stdout: str = ""
    stderr: str = ""
    returncode: Optional[int] = None
    duration: float = 0.0

    def summary(self, max_chars: int = 400) -> str:
        """Short human-readable reason, for grader logs."""
        if self.passed:
            return "ok"
        if self.status == "timeout":
            return "timeout"
        tail = (self.stderr or self.stdout).strip()
        if not tail:
            return f"{self.status} (rc={self.returncode})"
        lines = [ln for ln in tail.splitlines() if ln.strip()]
        # the last line of a traceback is the actual exception
        reason = lines[-1] if lines else tail
        return reason[:max_chars]


def resolve_python(spec: Optional[str]) -> Optional[str]:
    """Turn '3.13' or a path into a usable interpreter, or None if unavailable.

    Suites can need an older interpreter than the one running this tool, because
    a dependency has no wheel (or no working source build) for the newest
    CPython. Accepts an explicit path, a bare version, or an executable name.
    """
    if not spec:
        return None

    candidate = Path(spec).expanduser()
    if candidate.exists() and candidate.is_file():
        return str(candidate)

    # For a bare version, ask uv before searching PATH. A `pythonX.Y` on PATH is
    # often a shim into wherever it happened to be installed -- including a
    # scratch directory that will be deleted -- whereas uv reports its own
    # managed install, which is stable.
    uv = shutil.which("uv")
    if uv and re.fullmatch(r"\d+\.\d+(\.\d+)?", spec):
        try:
            r = subprocess.run([uv, "python", "find", spec],
                               capture_output=True, text=True, timeout=120)
            path = r.stdout.strip()
            if r.returncode == 0 and path and Path(path).exists():
                return path
        except Exception:
            pass

    return shutil.which(spec) or shutil.which(f"python{spec}")


def _detect_isolation(probe_python: Optional[str] = None,
                      ro_binds: Sequence[Path] = ()) -> str:
    """Pick the strongest isolation tier this host supports, once.

    The probe must run the *same* interpreter the suite will, with the same
    binds. Probing a bare sys.executable silently downgrades the tier whenever
    this tool is itself run from a venv under $HOME -- which is the usual case,
    since it needs `datasets`. bwrap tmpfs-mounts the home directory, the
    interpreter disappears, the probe fails, and the sandbox falls back to a
    tier that does not hide $HOME at all. Nothing about the results looks wrong
    when that happens, so it has to be got right here.
    """
    py = probe_python or sys.executable
    probe = "import sys; sys.exit(0)"

    if shutil.which("bwrap"):
        try:
            r = subprocess.run(
                _bwrap_argv(Path.cwd(), [py, "-I", "-c", probe], ro_binds=ro_binds),
                capture_output=True, timeout=20,
            )
            if r.returncode == 0:
                return "bwrap"
        except Exception:
            pass

    if shutil.which("unshare"):
        # Prefer the form that keeps the caller's uid. With `unshare -r` the
        # program runs as root inside the namespace, which silently changes
        # behaviour: root bypasses file permission bits, so a test asserting
        # that a write to a chmod 0444 file fails would instead see it succeed.
        for tier, argv in (
            ("unshare", ["unshare", "--user", "--map-current-user", "--net"]),
            ("unshare-root", ["unshare", "-rn"]),
        ):
            try:
                r = subprocess.run(
                    [*argv, py, "-I", "-c", probe],
                    capture_output=True, timeout=20,
                )
                if r.returncode == 0:
                    return tier
            except Exception:
                continue

    return "rlimit"


UNSHARE_ARGV = {
    "unshare": ["unshare", "--user", "--map-current-user", "--net"],
    "unshare-root": ["unshare", "-rn"],
}


def _under(path: Path, parent: str) -> bool:
    return bool(parent) and (str(path) == parent
                             or str(path).startswith(parent.rstrip("/") + os.sep))


def _bwrap_argv(workdir: Path, inner: Sequence[str],
                ro_binds: Sequence[Path] = ()) -> List[str]:
    home = os.path.expanduser("~")
    argv = [
        "bwrap",
        "--ro-bind", "/", "/",
        "--dev", "/dev",
        "--proc", "/proc",
        "--tmpfs", "/tmp",
    ]
    # Hide the real home so candidate code cannot read the user's files. The
    # venv usually lives under ~/.cache, so bind it back afterwards -- bwrap
    # applies these in order, and a later bind wins over the earlier tmpfs.
    if home and home != "/":
        argv += ["--tmpfs", home]
    # Only paths the tmpfs just hid need restoring. Re-binding anything else is
    # not merely redundant: bwrap cannot mount onto a symlink, so a bind whose
    # destination is one (version-manager alias directories usually are) fails
    # outright. Under the tmpfs the destination is recreated as a real
    # directory, which is exactly why the same bind works there.
    for path in ro_binds:
        if _under(path, home):
            argv += ["--ro-bind", str(path), str(path)]
    argv += [
        "--bind", str(workdir), str(workdir),
        "--chdir", str(workdir),
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--",
    ]
    return argv + list(inner)


def _nproc_ceiling() -> int:
    """A process cap that bounds a fork bomb without tripping on existing load.

    RLIMIT_NPROC is enforced per real UID across the whole system, not per
    process tree, so it has to sit above whatever the user is already running.
    """
    try:
        current = len(os.listdir("/proc"))
    except OSError:
        current = 512
    return max(current * 2, 1024)


def _apply_rlimits(timeout: float, with_nproc: bool,
                   as_bytes: Optional[int] = LIMIT_AS_BYTES):
    def _preexec():
        os.setsid()
        cpu = int(timeout) + LIMIT_CPU_SECONDS_SLACK
        limits = [
            (resource.RLIMIT_CPU, (cpu, cpu)),
            (resource.RLIMIT_FSIZE, (LIMIT_FSIZE_BYTES, LIMIT_FSIZE_BYTES)),
            (resource.RLIMIT_CORE, (0, 0)),
        ]
        # RLIMIT_AS caps reserved address space, not resident memory, so it is
        # the wrong instrument for any runtime that reserves a large region up
        # front and commits little of it. Node's TypeScript type-stripping goes
        # through WebAssembly, which reserves multiple gigabytes per instance
        # and dies at 4 GiB with an out-of-memory error that looks like a bug
        # in the code under test. Suites needing such a runtime raise or drop
        # it; the wall-clock timeout and RLIMIT_FSIZE still bound a runaway.
        if as_bytes:
            limits.append((resource.RLIMIT_AS, (as_bytes, as_bytes)))
        # Only in the no-namespace fallback: with bwrap/unshare the PID
        # namespace already contains a fork bomb, and clamping NPROC there
        # makes clone(CLONE_NEWUSER) fail outright with EAGAIN.
        if with_nproc:
            ceiling = _nproc_ceiling()
            limits.append((resource.RLIMIT_NPROC, (ceiling, ceiling)))
        for res, lim in limits:
            try:
                resource.setrlimit(res, lim)
            except (ValueError, OSError):
                pass
    return _preexec


class Sandbox:
    """A cached venv plus an isolated runner for untrusted candidate programs."""

    def __init__(
        self,
        name: str,
        packages: Optional[Sequence[str]] = None,
        cache_root: Optional[Path] = None,
        base_python: Optional[str] = None,
        isolation: Optional[str] = None,
        provision: Optional[Sequence[str]] = None,
        env: Optional[Dict[str, str]] = None,
        address_space_limit: Optional[int] = LIMIT_AS_BYTES,
    ):
        self.name = name
        self.packages = sorted(set(packages or []))
        # Python snippets run once at build time, while the network is still
        # reachable -- this is how a suite that would otherwise download data at
        # test time (nltk corpora, tokenizer models) gets to run offline later.
        self.provision = list(provision or [])
        # Extra environment for every run. Values may contain the literal token
        # {venv}, replaced with the venv path -- data downloaded at provisioning
        # time must live inside the venv, because $HOME is a tmpfs in the
        # strongest isolation tier and anything under it disappears.
        self.env = dict(env or {})
        # None drops the RLIMIT_AS cap entirely -- see _apply_rlimits
        self.address_space_limit = address_space_limit
        self.cache_root = Path(cache_root or DEFAULT_CACHE_ROOT)
        self.base_python = base_python or sys.executable
        self.venv_dir = self.cache_root / f"{name}-venv"
        self._python: Optional[Path] = None
        self._isolation = isolation
        self._base_prefix: List[Path] = []

    # -- provisioning ----------------------------------------------------

    @property
    def python(self) -> Path:
        if self._python is None:
            raise RuntimeError("Sandbox.ensure() must be called first")
        return self._python

    @property
    def isolation(self) -> str:
        if self._isolation is None:
            if self._python:
                py = str(self._python)
                roots = [self.venv_dir, *(self._base_prefix or [])]
            else:
                # probed before the venv exists: still bind the interpreter's
                # own tree, or the probe fails for the wrong reason
                py = sys.executable
                roots = self._detect_base_prefix(Path(py))
            self._isolation = _detect_isolation(py, roots)
        return self._isolation

    def _stamp_path(self) -> Path:
        return self.venv_dir / ".llama-eval-stamp.json"

    def _stamp_matches(self) -> bool:
        try:
            stamp = json.loads(self._stamp_path().read_text())
        except Exception:
            return False
        return (
            stamp.get("packages") == self.packages
            and stamp.get("py") == self._base_version()
            and stamp.get("provision") == self.provision
        )

    def _resolved_env(self) -> Dict[str, str]:
        return {k: v.replace("{venv}", str(self.venv_dir)) for k, v in self.env.items()}

    def _base_version(self) -> str:
        r = subprocess.run(
            [self.base_python, "-c",
             "import sys;print('.'.join(map(str,sys.version_info[:3])))"],
            capture_output=True, text=True,
        )
        return r.stdout.strip()

    def ensure(self, force: bool = False, quiet: bool = False) -> Path:
        """Create/refresh the venv. Idempotent and cheap when already valid."""
        py = self.venv_dir / "bin" / "python"
        if force and self.venv_dir.exists():
            shutil.rmtree(self.venv_dir)

        if py.exists() and self._stamp_matches():
            self._python = py
            self._base_prefix = self._detect_base_prefix(py)
            return py

        if not quiet:
            pkgs = ", ".join(self.packages) if self.packages else "stdlib only"
            print(f"[sandbox] provisioning venv {self.venv_dir} ({pkgs})")

        self.cache_root.mkdir(parents=True, exist_ok=True)
        if self.venv_dir.exists():
            shutil.rmtree(self.venv_dir)

        uv = shutil.which("uv")
        if uv:
            subprocess.run(
                [uv, "venv", "--python", self.base_python, str(self.venv_dir)],
                check=True, capture_output=True,
            )
        elif Path(self.base_python).resolve() == Path(sys.executable).resolve():
            venv.EnvBuilder(with_pip=bool(self.packages), clear=True).create(self.venv_dir)
        else:
            subprocess.run(
                [self.base_python, "-m", "venv", str(self.venv_dir)],
                check=True, capture_output=True,
            )

        if self.packages:
            if uv:
                cmd = [uv, "pip", "install", "--python", str(py), *self.packages]
            else:
                cmd = [str(py), "-m", "pip", "install", "--quiet",
                       "--disable-pip-version-check", *self.packages]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(
                    f"failed to install {len(self.packages)} package(s) into "
                    f"{self.venv_dir}:\n{(r.stderr or r.stdout)[-3000:]}"
                )

        for i, snippet in enumerate(self.provision, 1):
            if not quiet:
                print(f"[sandbox] provisioning step {i}/{len(self.provision)}")
            # deliberately unsandboxed: this is the one point where the suite is
            # allowed network access, so that test time never needs it
            prov_env = dict(os.environ)
            prov_env["SANDBOX_VENV"] = str(self.venv_dir)
            prov_env.update(self._resolved_env())
            r = subprocess.run([str(py), "-c", snippet], env=prov_env,
                               capture_output=True, text=True, timeout=1800)
            if r.returncode != 0:
                raise RuntimeError(
                    f"provisioning step {i} failed:\n{(r.stderr or r.stdout)[-3000:]}"
                )

        self._stamp_path().write_text(json.dumps({
            "packages": self.packages,
            "py": self._base_version(),
            "provision": self.provision,
        }))
        self._python = py
        self._base_prefix = self._detect_base_prefix(py)
        if not quiet:
            print(f"[sandbox] ready ({self.isolation} isolation)")
        if self.isolation == "unshare-root":
            print("Warning: this host's unshare cannot keep the caller's uid, so "
                  "candidate code runs as root inside the namespace. Tests that "
                  "assert a permission denial will not see one. Install "
                  "bubblewrap (bwrap) for correct results.")
        return py

    @staticmethod
    def _detect_base_prefix(py: Path) -> List[Path]:
        """Every directory that must stay visible for the venv to start.

        A venv is only symlinks into its base interpreter, so the whole install
        tree has to survive the tmpfs that hides $HOME -- and binding just the
        resolved tree is not enough. Version-managed interpreters (uv, pyenv)
        are commonly reached through an unversioned alias symlink
        (cpython-3.13-... -> cpython-3.13.14-...), and startup resolves through
        the alias path, so both spellings need to be mounted. Walk the chain
        and keep every install root it passes through.
        """
        roots: List[Path] = []

        def add(path: Path):
            if path.exists() and path not in roots:
                roots.append(path)

        cur, hops = py, 0
        while hops < 16:
            hops += 1
            add(cur.parent.parent)
            if cur.is_symlink():
                target = Path(os.readlink(cur))
                cur = target if target.is_absolute() else (cur.parent / target)
            else:
                break

        try:
            r = subprocess.run([str(py), "-c", "import sys; print(sys.base_prefix)"],
                               capture_output=True, text=True, timeout=60)
            if r.returncode == 0 and r.stdout.strip():
                add(Path(r.stdout.strip()))
        except Exception:
            pass

        # a base prefix of /usr is already covered by the read-only root bind
        return [p for p in roots if str(p) not in ("/", "/usr", "/usr/local")]

    # -- execution -------------------------------------------------------

    def _isolate(self, argv: Sequence[str], workdir: Path,
                 ro_binds: Sequence[Path] = ()) -> List[str]:
        """Wrap a command in the strongest isolation tier this host supports."""
        if self.isolation == "bwrap":
            # the venv, and the base interpreter tree it symlinks into, must
            # stay visible through the tmpfs that hides $HOME
            return _bwrap_argv(
                workdir, argv,
                ro_binds=[self.venv_dir, *(self._base_prefix or []), *ro_binds],
            )
        if self.isolation in UNSHARE_ARGV:
            return [*UNSHARE_ARGV[self.isolation], *argv]
        return list(argv)

    def _base_env(self, workdir: Path) -> Dict[str, str]:
        env = {
            "PATH": "/usr/bin:/bin",
            "HOME": str(workdir),
            "TMPDIR": str(workdir),
            "LC_ALL": "C.UTF-8",
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
        }
        env.update(self._resolved_env())
        return env

    def run_argv(self, argv: Sequence[str], workdir: Path,
                 timeout: float = 15.0, env: Optional[Dict[str, str]] = None,
                 ro_binds: Sequence[Path] = ()) -> ExecResult:
        """Run an arbitrary command in `workdir`, which stays writable.

        Unlike run(), the caller owns the directory and it survives the call.
        That is what a multi-step agent needs: its edits have to persist across
        many tool invocations before anything is graded. Taking an argv rather
        than a source string is what lets a suite drive a linter or a test
        runner for a language other than Python.
        """
        run_env = self._base_env(workdir)
        if env:
            run_env.update(env)

        started = time.monotonic()
        try:
            proc = subprocess.Popen(
                self._isolate(argv, workdir, ro_binds),
                cwd=str(workdir),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                errors="replace",
                env=run_env,
                preexec_fn=_apply_rlimits(timeout,
                                          with_nproc=self.isolation == "rlimit",
                                          as_bytes=self.address_space_limit),
            )
        except Exception as e:  # sandbox itself misbehaved
            return ExecResult(False, "error", "", f"{type(e).__name__}: {e}")

        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_tree(proc)
            out, err = proc.communicate()
            return ExecResult(False, "timeout", out or "", err or "",
                              None, time.monotonic() - started)

        duration = time.monotonic() - started
        if proc.returncode == 0:
            return ExecResult(True, "ok", out, err, 0, duration)
        return ExecResult(False, "failed", out, err, proc.returncode, duration)

    def run(self, code: str, timeout: float = 15.0,
            env: Optional[Dict[str, str]] = None) -> ExecResult:
        """Run `code` to completion in the sandbox. Exit 0 == passed."""
        workdir = Path(tempfile.mkdtemp(prefix=f"{self.name}-", dir="/tmp"))
        try:
            prog = workdir / "candidate.py"
            prog.write_text(code)
            return self.run_argv([str(self.python), "-I", "-B", str(prog)],
                                 workdir, timeout=timeout, env=env)
        except Exception as e:  # sandbox itself misbehaved
            return ExecResult(False, "error", "", f"{type(e).__name__}: {e}")
        finally:
            shutil.rmtree(workdir, ignore_errors=True)


def _kill_tree(proc: subprocess.Popen):
    """Kill the candidate and everything it spawned."""
    for sig in (signal.SIGKILL,):
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except (ProcessLookupError, PermissionError):
            try:
                proc.kill()
            except ProcessLookupError:
                pass
