#!/usr/bin/env python3
"""Multi-turn agentic coding eval: tools over a real working tree.

The existing code suites are single-shot -- one prompt, one reply, execute,
grade. This one is not. The model is given a small but real repository and a
deliberately narrow tool set, and has to find its way to the defect before it
can fix it. That shape is what puts a run in the tens-of-thousands-of-tokens
range without any artificial padding: the context grows because the model
chose to read things.

Three deliberate constraints:

* No shell, and no way to run the tests. The only feedback channel is `lint`.
  A syntax or type error can be driven out mechanically; a logic error has to
  be reasoned about. Suites that let an agent iterate against the test suite
  measure something closer to search than to understanding.
* Two edit tools, one line-addressed and one content-addressed. Which one a
  model reaches for, and whether it keeps line numbers straight after its own
  earlier edit, is itself a signal.
* Tests never touch the working tree. They are held outside it and copied in
  after the agent has stopped, so they can be neither read nor edited.

Everything here is imported lazily by llama-eval.py, and every dependency it
needs is scoped to the language actually under test, so none of it is
installed for anyone running the other suites.
"""

import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Tool output caps. These bound context growth without hiding anything the
# model asked for precisely: a narrowed read is never truncated, only a lazy
# whole-file read of something large.
MAX_READ_LINES = 400
MAX_TOOL_CHARS = 12000
MAX_SEARCH_HITS = 60


# --------------------------------------------------------------------------
# languages
# --------------------------------------------------------------------------

@dataclass
class LangSpec:
    """Everything language-specific: how to lint, how to test, what to install.

    Kept as data so adding a language is a corpus concern rather than a harness
    change, and so a run only ever provisions the toolchain for the language it
    is actually evaluating.
    """
    name: str
    exts: Tuple[str, ...]
    packages: Tuple[str, ...] = ()
    provision: Tuple[str, ...] = ()
    env: Dict[str, str] = field(default_factory=dict)
    # Passed straight to Sandbox. Only raised where a runtime reserves a large
    # virtual address space it never commits.
    address_space_limit: Optional[int] = 4 * 1024 * 1024 * 1024

    def source_files(self, root: Path) -> List[Path]:
        return sorted(p for p in root.rglob("*")
                      if p.is_file() and p.suffix in self.exts)


PYTHON = LangSpec(
    name="python",
    exts=(".py",),
    # ruff catches syntax and obvious defects, mypy the type errors. Both are
    # pure-wheel installs, so the venv builds in seconds and needs no compiler.
    packages=("ruff==0.14.2", "mypy==1.18.2"),
)

TYPESCRIPT = LangSpec(
    name="typescript",
    exts=(".ts",),
    # tsc is the linter and the type checker at once. Tests run on node's own
    # runner against stripped types, so there is no build step, no bundler and
    # no test framework to install.
    # node strips types through a WebAssembly module, and each WASM instance
    # reserves gigabytes of address space up front. At the default 4 GiB cap it
    # aborts with an out-of-memory error that reads like a fault in the code
    # under test rather than a sandbox limit.
    address_space_limit=16 * 1024 * 1024 * 1024,
    provision=(
        "import os, subprocess, shutil, sys\n"
        "npm = shutil.which('npm')\n"
        "if not npm: sys.exit('npm not found on PATH; needed for the typescript suite')\n"
        "dest = os.path.join(os.environ['SANDBOX_VENV'], 'node')\n"
        "os.makedirs(dest, exist_ok=True)\n"
        "r = subprocess.run([npm, 'install', '--silent', '--no-audit', '--no-fund',\n"
        "                    '--prefix', dest, 'typescript@5.9.3'],\n"
        "                   capture_output=True, text=True)\n"
        "sys.exit(0 if r.returncode == 0 else (r.stderr or r.stdout)[-2000:])\n",
    ),
)

LANGS: Dict[str, LangSpec] = {"python": PYTHON, "typescript": TYPESCRIPT}


# --------------------------------------------------------------------------
# workspace
# --------------------------------------------------------------------------

class PathEscape(Exception):
    """A tool was handed a path outside the repository."""


class Workspace:
    """A private, writable copy of one task's repository.

    Every tool path is resolved through here. Confinement is checked on the
    fully-resolved path, so a symlink planted inside the tree cannot be used to
    read or write outside it.
    """

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.escape_attempts = 0

    def resolve(self, rel: str) -> Path:
        if rel is None or not str(rel).strip():
            raise PathEscape("no path given")
        raw = Path(str(rel))
        # An absolute path is honoured when it genuinely points inside the
        # workspace, because models emit them routinely and rejecting them
        # would measure path formatting rather than coding. What it must never
        # do is get silently reinterpreted as relative: stripping the leading
        # slash turns /etc/passwd into <repo>/etc/passwd, which is a different
        # file than the one asked for and fails for a misleading reason.
        base = raw if raw.is_absolute() else (self.root / raw)
        candidate = base.resolve()
        if candidate != self.root and self.root not in candidate.parents:
            self.escape_attempts += 1
            raise PathEscape(f"path {rel!r} is outside the repository")
        return candidate

    def rel(self, path: Path) -> str:
        return str(path.relative_to(self.root))

    def files(self) -> List[Path]:
        return sorted(p for p in self.root.rglob("*")
                      if p.is_file() and "__pycache__" not in p.parts)

    def snapshot(self) -> Dict[str, str]:
        out = {}
        for p in self.files():
            try:
                out[self.rel(p)] = p.read_text()
            except (UnicodeDecodeError, OSError):
                pass
        return out


def _clip(text: str, limit: int = MAX_TOOL_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated, {len(text) - limit} more characters]"


def _numbered(lines: Sequence[str], start: int) -> str:
    width = len(str(start + len(lines) - 1))
    return "\n".join(f"{start + i:>{width}}\t{ln}" for i, ln in enumerate(lines))


# --------------------------------------------------------------------------
# tools
# --------------------------------------------------------------------------

def tool_schemas() -> List[Dict[str, Any]]:
    """OpenAI-format function schemas for the whole tool set."""
    def fn(name, desc, props, required=()):
        return {"type": "function", "function": {
            "name": name, "description": desc,
            "parameters": {"type": "object", "properties": props,
                           "required": list(required)}}}

    return [
        fn("list_files", "List files in the repository.",
           {"path": {"type": "string",
                     "description": "Directory relative to the repository root. "
                                    "Omit for the whole tree."}}),
        fn("read_file",
           "Read a file. Output is prefixed with 1-based line numbers. Give "
           "start_line/end_line to read only part of a large file.",
           {"path": {"type": "string", "description": "File path relative to the repository root."},
            "start_line": {"type": "integer", "description": "First line to read, 1-based, inclusive."},
            "end_line": {"type": "integer", "description": "Last line to read, 1-based, inclusive."}},
           ["path"]),
        fn("search",
           "Search the repository with a regular expression. Returns matching "
           "lines with their file and line number.",
           {"pattern": {"type": "string", "description": "Python regular expression."},
            "path": {"type": "string", "description": "Limit the search to this subdirectory."},
            "glob": {"type": "string", "description": "Limit to files matching this glob, e.g. '*.py'."}},
           ["pattern"]),
        fn("edit_lines",
           "Replace an inclusive 1-based line range with new text. Line numbers "
           "refer to the file as it is now, so re-read it after any edit that "
           "changed the number of lines.",
           {"path": {"type": "string", "description": "File path relative to the repository root."},
            "start_line": {"type": "integer", "description": "First line to replace, 1-based, inclusive."},
            "end_line": {"type": "integer", "description": "Last line to replace, 1-based, inclusive."},
            "new_text": {"type": "string", "description": "Replacement text. Use an empty string to delete the range."}},
           ["path", "start_line", "end_line", "new_text"]),
        fn("edit_replace",
           "Replace an exact snippet of text. The snippet must occur exactly "
           "once in the file unless replace_all is set.",
           {"path": {"type": "string", "description": "File path relative to the repository root."},
            "old_text": {"type": "string", "description": "Exact text to find, including indentation."},
            "new_text": {"type": "string", "description": "Replacement text."},
            "replace_all": {"type": "boolean", "description": "Replace every occurrence instead of requiring exactly one."}},
           ["path", "old_text", "new_text"]),
        fn("write_file",
           "Write a whole file, creating or overwriting it. Prefer the edit "
           "tools for changes to existing files.",
           {"path": {"type": "string", "description": "File path relative to the repository root."},
            "content": {"type": "string", "description": "Full new file contents."}},
           ["path", "content"]),
        fn("lint",
           "Run the language's linter and type checker over the repository and "
           "return the diagnostics. This is the only feedback available; the "
           "test suite cannot be run.",
           {"path": {"type": "string", "description": "Limit to this file or directory."}}),
        fn("finish",
           "Declare the task complete. Call this once the change is made.",
           {"summary": {"type": "string", "description": "Brief description of the change made."}}),
    ]


class ToolBox:
    """Executes tool calls against one workspace. Returns plain text results.

    Failures are reported as ordinary text rather than raised, because how a
    model recovers from a rejected edit is part of what is being measured.
    """

    def __init__(self, ws: Workspace, linter=None):
        self.ws = ws
        self.linter = linter
        self.calls = 0
        self.errors = 0
        self.edits = 0
        self.finished = False
        self.finish_summary: Optional[str] = None
        self.per_tool: Dict[str, int] = {}

    def dispatch(self, name: str, args: Dict[str, Any]) -> str:
        self.calls += 1
        self.per_tool[name] = self.per_tool.get(name, 0) + 1
        handler = getattr(self, f"_t_{name}", None)
        if handler is None:
            result = f"Error: unknown tool {name!r}."
        else:
            try:
                result = handler(args)
            except PathEscape as e:
                result = f"Error: {e}"
            except KeyError as e:
                result = f"Error: missing required argument {e}."
            except Exception as e:
                result = f"Error: {type(e).__name__}: {e}"
        # One counting point for every failure, raised or returned -- a tool
        # that rejects an edit is just as much a failed call as one that
        # throws, and the two must not be tallied differently.
        if result.startswith("Error:"):
            self.errors += 1
        return _clip(result)

    # -- read-only -------------------------------------------------------

    def _t_list_files(self, a):
        base = self.ws.resolve(a.get("path") or ".")
        if not base.exists():
            return f"Error: {a.get('path')!r} does not exist."
        paths = [p for p in self.ws.files() if p == base or base in p.parents]
        if not paths:
            return "(no files)"
        return "\n".join(f"{self.ws.rel(p)}\t{p.stat().st_size} bytes" for p in paths)

    def _t_read_file(self, a):
        path = self.ws.resolve(a["path"])
        if not path.is_file():
            return f"Error: {a['path']!r} is not a file."
        lines = path.read_text().splitlines()
        start = max(1, int(a.get("start_line") or 1))
        end = int(a.get("end_line") or len(lines))
        end = min(end, len(lines))
        if start > len(lines):
            return f"Error: {a['path']!r} has only {len(lines)} lines."
        window = lines[start - 1:end]
        note = ""
        if len(window) > MAX_READ_LINES:
            window = window[:MAX_READ_LINES]
            note = (f"\n... [showing {MAX_READ_LINES} of {end - start + 1} requested "
                    f"lines; re-read with start_line/end_line for the rest]")
        header = f"{a['path']} ({len(lines)} lines total)\n"
        return header + _numbered(window, start) + note

    def _t_search(self, a):
        try:
            rx = re.compile(a["pattern"])
        except re.error as e:
            return f"Error: bad regular expression: {e}"
        base = self.ws.resolve(a.get("path") or ".")
        glob = a.get("glob")
        hits, scanned = [], 0
        for p in self.ws.files():
            if not (p == base or base in p.parents):
                continue
            if glob and not p.match(glob):
                continue
            try:
                text = p.read_text()
            except (UnicodeDecodeError, OSError):
                continue
            scanned += 1
            for i, line in enumerate(text.splitlines(), 1):
                if rx.search(line):
                    hits.append(f"{self.ws.rel(p)}:{i}:{line.strip()[:200]}")
                    if len(hits) >= MAX_SEARCH_HITS:
                        return ("\n".join(hits) +
                                f"\n... [stopped at {MAX_SEARCH_HITS} matches; narrow the pattern]")
        if not hits:
            return f"No matches for {a['pattern']!r} in {scanned} file(s)."
        return "\n".join(hits)

    # -- mutating --------------------------------------------------------

    def _t_edit_lines(self, a):
        path = self.ws.resolve(a["path"])
        if not path.is_file():
            return f"Error: {a['path']!r} is not a file."
        lines = path.read_text().splitlines()
        start, end = int(a["start_line"]), int(a["end_line"])
        if start < 1 or start > len(lines):
            return (f"Error: start_line {start} out of range; "
                    f"{a['path']!r} has {len(lines)} lines.")
        if end < start or end > len(lines):
            return (f"Error: end_line {end} out of range; must be between "
                    f"{start} and {len(lines)}.")
        new = str(a["new_text"]).splitlines()
        path.write_text("\n".join(lines[:start - 1] + new + lines[end:]) + "\n")
        self.edits += 1
        delta = len(new) - (end - start + 1)
        return (f"Replaced lines {start}-{end} of {a['path']} "
                f"({end - start + 1} -> {len(new)} lines, {delta:+d}). "
                f"Later line numbers have shifted." if delta else
                f"Replaced lines {start}-{end} of {a['path']}.")

    def _t_edit_replace(self, a):
        path = self.ws.resolve(a["path"])
        if not path.is_file():
            return f"Error: {a['path']!r} is not a file."
        text = path.read_text()
        old, new = str(a["old_text"]), str(a["new_text"])
        if not old:
            return "Error: old_text must not be empty."
        n = text.count(old)
        if n == 0:
            return (f"Error: old_text not found in {a['path']}. It must match "
                    f"exactly, including indentation.")
        if n > 1 and not a.get("replace_all"):
            return (f"Error: old_text occurs {n} times in {a['path']}. Include "
                    f"more surrounding context to make it unique, or set "
                    f"replace_all.")
        path.write_text(text.replace(old, new))
        self.edits += 1
        return f"Replaced {n} occurrence(s) in {a['path']}."

    def _t_write_file(self, a):
        path = self.ws.resolve(a["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        existed = path.is_file()
        path.write_text(str(a["content"]))
        self.edits += 1
        return f"{'Overwrote' if existed else 'Created'} {a['path']}."

    # -- feedback --------------------------------------------------------

    def _t_lint(self, a):
        if self.linter is None:
            return "Error: no linter is configured for this task."
        target = a.get("path")
        if target:
            self.ws.resolve(target)  # confinement check only
        return self.linter(target)

    def _t_finish(self, a):
        self.finished = True
        self.finish_summary = a.get("summary")
        return "Task marked complete."


# --------------------------------------------------------------------------
# linting and testing
# --------------------------------------------------------------------------

# Emits per-test outcomes rather than a pass/fail exit code, so a partly
# correct change earns partial credit and a regression can be told apart from
# a fix that simply did not go far enough.
PY_TEST_RUNNER = r'''
import json, os, random, sys, unittest
sys.path.insert(0, os.getcwd())
random.seed(0)

class Collect(unittest.TestResult):
    def __init__(self):
        super().__init__()
        self.outcomes = {}
    def addSuccess(self, t): self.outcomes[t.id()] = "pass"
    def addFailure(self, t, e): self.outcomes[t.id()] = "fail"
    def addError(self, t, e): self.outcomes[t.id()] = "error"
    def addSkip(self, t, r): self.outcomes[t.id()] = "skip"
    def addExpectedFailure(self, t, e): self.outcomes[t.id()] = "pass"
    def addUnexpectedSuccess(self, t): self.outcomes[t.id()] = "fail"

def flatten(s):
    for t in s:
        if isinstance(t, unittest.TestSuite):
            yield from flatten(t)
        else:
            yield t

out = {"tests": {}}
try:
    suite = unittest.TestLoader().discover(os.environ.get("TEST_DIR", "tests"),
                                           top_level_dir=".")
    res = Collect()
    for t in flatten(suite):
        tid = t.id()
        try:
            t.run(res)
        except Exception:
            pass
        res.outcomes.setdefault(tid, "error")
    out["tests"] = res.outcomes
except Exception as e:
    out["error"] = f"{type(e).__name__}: {e}"

print("###RESULT###")
print(json.dumps(out))
'''


def _node_roots() -> List[Path]:
    """Directories that must stay visible for node to start under bwrap."""
    node = shutil.which("node")
    if not node:
        return []
    real = Path(node).resolve()
    return [Path(node).parent.parent, real.parent.parent]


class Runner:
    """Runs the linter and the hidden tests for one language, in the sandbox."""

    def __init__(self, sandbox, lang: LangSpec, timeout: float = 90.0):
        self.sandbox = sandbox
        self.lang = lang
        self.timeout = timeout

    def _binds(self) -> List[Path]:
        return _node_roots() if self.lang.name == "typescript" else []

    def _env(self) -> Dict[str, str]:
        if self.lang.name != "typescript":
            return {}
        node = shutil.which("node")
        return {"PATH": f"{Path(node).parent}:/usr/bin:/bin"} if node else {}

    # -- lint ------------------------------------------------------------

    def lint(self, workdir: Path, target: Optional[str] = None) -> str:
        if self.lang.name == "python":
            return self._lint_python(workdir, target)
        return self._lint_typescript(workdir, target)

    def _lint_python(self, workdir: Path, target: Optional[str]) -> str:
        venv = self.sandbox.venv_dir
        tgt = target or "."
        chunks = []
        r = self.sandbox.run_argv(
            [str(venv / "bin" / "ruff"), "check", "--no-cache",
             "--output-format", "concise", tgt],
            workdir, timeout=self.timeout)
        # ruff announces success on stdout; that is not a diagnostic, and
        # passing it through would tell the model it has findings when it does
        # not.
        text = (r.stdout or r.stderr).strip()
        if text and "All checks passed" not in text:
            chunks.append(text)
        r = self.sandbox.run_argv(
            [str(venv / "bin" / "mypy"), "--no-color-output", "--no-error-summary",
             "--cache-dir", "/dev/null", "--ignore-missing-imports", tgt],
            workdir, timeout=self.timeout)
        text = (r.stdout or r.stderr).strip()
        if text and not text.startswith("Success:"):
            chunks.append(text)
        return "\n".join(chunks) if chunks else "No problems found."

    def _lint_typescript(self, workdir: Path, target: Optional[str]) -> str:
        tsc = self.sandbox.venv_dir / "node" / "node_modules" / "typescript" / "bin" / "tsc"
        files = [str(p.relative_to(workdir)) for p in sorted(workdir.rglob("*.ts"))
                 if "node_modules" not in p.parts]
        if target:
            t = (workdir / target).resolve()
            files = [f for f in files if (workdir / f).resolve() == t
                     or t in (workdir / f).resolve().parents]
        if not files:
            return "No TypeScript files to check."
        # Flags are passed explicitly rather than read from a tsconfig in the
        # tree, so the checking strictness cannot be weakened by editing it.
        r = self.sandbox.run_argv(
            ["node", str(tsc), "--noEmit", "--strict", "--target", "es2022",
             "--module", "esnext", "--moduleResolution", "bundler",
             "--allowImportingTsExtensions", *files],
            workdir, timeout=self.timeout, env=self._env(), ro_binds=self._binds())
        out = (r.stdout or "") + (r.stderr or "")
        return out.strip() or "No problems found."

    # -- tests -----------------------------------------------------------

    def test(self, workdir: Path) -> Dict[str, Any]:
        if self.lang.name == "python":
            return self._test_python(workdir)
        return self._test_typescript(workdir)

    def _test_python(self, workdir: Path) -> Dict[str, Any]:
        runner = workdir / "__run_tests.py"
        runner.write_text(PY_TEST_RUNNER)
        r = self.sandbox.run_argv(
            [str(self.sandbox.python), "-I", "-B", str(runner)],
            workdir, timeout=self.timeout, env={"TEST_DIR": "tests"})
        return self._parse_json_result(r)

    def _parse_json_result(self, r) -> Dict[str, Any]:
        if r.status == "timeout":
            return {"tests": {}, "error": "timeout"}
        marker = "###RESULT###"
        if marker not in (r.stdout or ""):
            return {"tests": {},
                    "error": f"no result: {(r.stderr or r.stdout or '')[-400:]}"}
        try:
            return json.loads(r.stdout.split(marker, 1)[1].strip())
        except Exception as e:
            return {"tests": {}, "error": f"unparseable result: {e}"}

    def _test_typescript(self, workdir: Path) -> Dict[str, Any]:
        tests = sorted(p for p in (workdir / "tests").rglob("*.ts")) \
            if (workdir / "tests").is_dir() else []
        if not tests:
            return {"tests": {}, "error": "no test files"}
        r = self.sandbox.run_argv(
            ["node", "--test", "--experimental-strip-types",
             "--test-reporter=tap",
             *[str(p.relative_to(workdir)) for p in tests]],
            workdir, timeout=self.timeout, env=self._env(), ro_binds=self._binds())
        return self._parse_tap(r)

    @staticmethod
    def _parse_tap(r) -> Dict[str, Any]:
        if r.status == "timeout":
            return {"tests": {}, "error": "timeout"}
        text = (r.stdout or "") + "\n" + (r.stderr or "")
        outcomes: Dict[str, str] = {}
        for line in text.splitlines():
            m = re.match(r"\s*(not )?ok\s+\d+\s*-\s*(.+?)\s*$", line)
            if not m:
                continue
            name = m.group(2).strip()
            if name.endswith(" # SKIP") or " # SKIP" in name:
                outcomes[name.split(" # ")[0]] = "skip"
                continue
            # node reports each file as a test too; keep the worst outcome so a
            # failing file cannot be masked by its own passing summary line
            verdict = "fail" if m.group(1) else "pass"
            if outcomes.get(name) != "fail":
                outcomes[name] = verdict
        if not outcomes:
            return {"tests": {},
                    "error": f"no TAP output: {text.strip()[-400:]}"}
        return {"tests": outcomes}


# --------------------------------------------------------------------------
# tasks and episodes
# --------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert {lang} engineer working in an existing repository.

Resolve the user's request by editing files with the tools provided.

Important constraints:
- You cannot run the code, the tests, or any shell command. The only feedback
  available is the `lint` tool, which reports syntax and type problems.
- The repository already has a test suite that will be run against your work
  after you finish, but you cannot see it or run it. Do not create test files.
- Read enough of the code to understand it before editing. A change that only
  makes the symptom go away will usually fail the tests.
- Make the smallest correct change. Do not reformat or refactor unrelated code.

Call `finish` when the change is complete."""


@dataclass
class AgenticTask:
    task_id: str
    lang: str
    category: str            # syntax | type | logic | feature
    difficulty: int
    instruction: str
    files: Dict[str, str]
    tests: Dict[str, str]
    gold: Dict[str, str] = field(default_factory=dict)
    fail_to_pass: List[str] = field(default_factory=list)
    pass_to_pass: List[str] = field(default_factory=list)

    @classmethod
    def from_record(cls, rec: Dict[str, Any]) -> "AgenticTask":
        def as_map(v):
            return v if isinstance(v, dict) else json.loads(v or "{}")
        def as_list(v):
            return v if isinstance(v, list) else json.loads(v or "[]")
        return cls(
            task_id=rec["task_id"], lang=rec["lang"], category=rec["category"],
            difficulty=int(rec.get("difficulty", 3)),
            instruction=rec["instruction"],
            files=as_map(rec["files"]), tests=as_map(rec["tests"]),
            gold=as_map(rec.get("gold")),
            fail_to_pass=as_list(rec.get("fail_to_pass")),
            pass_to_pass=as_list(rec.get("pass_to_pass")),
        )

    @property
    def spec(self) -> LangSpec:
        return LANGS[self.lang]


def materialise(files: Dict[str, str], dest: Path) -> Path:
    """Write a file map out as a real directory tree."""
    dest.mkdir(parents=True, exist_ok=True)
    for rel, content in files.items():
        p = dest / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    return dest


def repo_overview(files: Dict[str, str]) -> str:
    """A compact tree, so a run is not lost merely to not finding the files."""
    lines = []
    for rel in sorted(files):
        n = files[rel].count("\n") + 1
        lines.append(f"  {rel} ({n} lines)")
    return "\n".join(lines)


def user_prompt(task: AgenticTask) -> str:
    return (f"Repository files:\n{repo_overview(task.files)}\n\n"
            f"Task:\n{task.instruction}")


@dataclass
class EpisodeResult:
    stop_reason: str = "unknown"
    turns: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    edits: int = 0
    escape_attempts: int = 0
    finished: bool = False
    bad_tool_json: int = 0
    nudges: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    peak_context: int = 0
    per_tool: Dict[str, int] = field(default_factory=dict)
    error: Optional[str] = None


# Sent when a turn arrives with no tool calls and no `finish`. Models routinely
# narrate their analysis in prose mid-task; treating that as "done" would end
# the episode with no edits and score a reasoning failure that never happened.
# Deliberately neutral: it points at the protocol, never at the defect.
NUDGE = ("You did not call any tool. If your change is complete, call `finish`. "
         "Otherwise continue working using the tools.")


def run_episode(task: AgenticTask, toolbox: ToolBox, chat,
                max_turns: int = 50,
                context_limit: Optional[int] = None,
                max_nudges: int = 3) -> Tuple[EpisodeResult, List[Dict]]:
    """Drive one multi-turn tool-calling episode.

    `chat(messages, tools) -> response dict` is supplied by the caller so this
    module stays free of any HTTP concerns.
    """
    tools = tool_schemas()
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT.format(lang=task.spec.name)},
        {"role": "user", "content": user_prompt(task)},
    ]
    res = EpisodeResult()

    for turn in range(max_turns):
        res.turns = turn + 1
        try:
            reply = chat(messages, tools)
        except Exception as e:
            res.stop_reason, res.error = "request_failed", f"{type(e).__name__}: {e}"
            break

        usage = reply.get("usage") or {}
        pt = int(usage.get("prompt_tokens") or 0)
        ct = int(usage.get("completion_tokens") or 0)
        res.prompt_tokens += pt
        res.completion_tokens += ct
        res.peak_context = max(res.peak_context, pt + ct)

        choices = reply.get("choices") or []
        if not choices:
            res.stop_reason, res.error = "no_choices", json.dumps(reply)[:300]
            break
        msg = choices[0].get("message") or {}
        calls = msg.get("tool_calls") or []

        # Echo the assistant turn back verbatim; the server needs the exact
        # tool_call ids to match the results that follow.
        messages.append({
            "role": "assistant",
            "content": msg.get("content") or "",
            **({"tool_calls": calls} if calls else {}),
        })

        if not calls:
            if toolbox.finished:
                res.stop_reason = "finished"
                break
            if res.nudges >= max_nudges:
                res.stop_reason = "no_tool_calls"
                break
            res.nudges += 1
            messages.append({"role": "user", "content": NUDGE})
            continue

        for call in calls:
            fn = call.get("function") or {}
            name = fn.get("name") or ""
            raw = fn.get("arguments")
            if isinstance(raw, dict):
                args = raw
            else:
                try:
                    args = json.loads(raw or "{}")
                except (json.JSONDecodeError, TypeError):
                    # Small models routinely emit unparseable arguments. That is
                    # a real failure mode worth counting rather than hiding, so
                    # it is reported back like any other tool error.
                    res.bad_tool_json += 1
                    args = None
            output = (toolbox.dispatch(name, args) if args is not None else
                      f"Error: arguments for {name!r} were not valid JSON.")
            messages.append({"role": "tool",
                             "tool_call_id": call.get("id") or f"call_{res.tool_calls}",
                             "content": output})

        if toolbox.finished:
            res.stop_reason = "finished"
            break
        if context_limit and res.peak_context > context_limit:
            res.stop_reason = "context_limit"
            break
    else:
        res.stop_reason = "max_turns"

    res.tool_calls = toolbox.calls
    res.tool_errors = toolbox.errors
    res.edits = toolbox.edits
    res.escape_attempts = toolbox.ws.escape_attempts
    res.finished = toolbox.finished
    res.per_tool = dict(toolbox.per_tool)
    return res, messages


# --------------------------------------------------------------------------
# grading
# --------------------------------------------------------------------------

def run_tests_against(files: Dict[str, str], task: AgenticTask, runner: Runner,
                      workroot: Path, label: str) -> Dict[str, str]:
    """Materialise a tree, drop the hidden tests in, and run them.

    The tests are only ever written into this throwaway copy, never into the
    tree the agent worked in, so there is no window in which they could have
    been read or edited.
    """
    dest = workroot / f"grade-{label}"
    shutil.rmtree(dest, ignore_errors=True)
    materialise(files, dest)
    materialise(task.tests, dest)
    return runner.test(dest).get("tests", {})


def score(task: AgenticTask, outcomes: Dict[str, str]) -> Dict[str, Any]:
    """Resolved means every target test passes and nothing previously passing broke."""
    def passing(tid):
        return outcomes.get(tid) == "pass"

    f2p_ok = [t for t in task.fail_to_pass if passing(t)]
    p2p_ok = [t for t in task.pass_to_pass if passing(t)]
    regressions = [t for t in task.pass_to_pass if not passing(t)]
    resolved = (len(f2p_ok) == len(task.fail_to_pass)
                and not regressions and bool(task.fail_to_pass))
    return {
        "resolved": resolved,
        "f2p_passed": len(f2p_ok), "f2p_total": len(task.fail_to_pass),
        "p2p_passed": len(p2p_ok), "p2p_total": len(task.pass_to_pass),
        "regressions": regressions[:10],
        "n_regressions": len(regressions),
        "partial": (len(f2p_ok) / len(task.fail_to_pass)) if task.fail_to_pass else 0.0,
    }


def make_sandbox(lang_name: str, sandbox_cls, cache_root=None, base_python=None):
    """A sandbox carrying only the toolchain for `lang_name`.

    Keyed per language rather than per suite: running the Python tasks must
    never provision node, and vice versa.
    """
    lang = LANGS[lang_name]
    return sandbox_cls(
        name=f"agentic-{lang_name}",
        packages=list(lang.packages),
        provision=list(lang.provision),
        env=dict(lang.env),
        address_space_limit=lang.address_space_limit,
        cache_root=cache_root,
        base_python=base_python,
    )


def run_task(task: AgenticTask, chat, sandbox, workroot: Optional[Path] = None,
             max_turns: int = 50, context_limit: Optional[int] = None,
             test_timeout: float = 90.0) -> Dict[str, Any]:
    """One full attempt: fresh workspace, agent episode, then hidden tests."""
    root = Path(tempfile.mkdtemp(prefix=f"agentic-{task.task_id}-",
                                 dir=str(workroot) if workroot else None))
    try:
        ws_root = root / "repo"
        materialise(task.files, ws_root)
        ws = Workspace(ws_root)
        runner = Runner(sandbox, task.spec, timeout=test_timeout)
        toolbox = ToolBox(ws, linter=lambda t: runner.lint(ws_root, t))

        episode, messages = run_episode(task, toolbox, chat,
                                        max_turns=max_turns,
                                        context_limit=context_limit)
        final = ws.snapshot()
        changed = sorted(k for k in final
                         if final[k] != task.files.get(k)) + \
                  sorted(k for k in task.files if k not in final)

        outcomes = run_tests_against(final, task, runner, root, "final")
        result = score(task, outcomes)
        result.update({
            "task_id": task.task_id, "lang": task.lang,
            "category": task.category, "difficulty": task.difficulty,
            "changed_files": changed,
            "episode": {k: v for k, v in vars(episode).items()},
        })
        return result
    finally:
        shutil.rmtree(root, ignore_errors=True)
