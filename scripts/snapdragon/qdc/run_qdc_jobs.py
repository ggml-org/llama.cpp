"""Run llama.cpp Hexagon tests in a single QDC job.

Bundles test scripts into one artifact and submits a single QDC job:

  1. run_bench_tests_posix.py — llama-cli and llama-bench on CPU / GPU / NPU
                                (from scripts/snapdragon/qdc/)

Results are written to $GITHUB_STEP_SUMMARY when set (GitHub Actions).

Prerequisites:
  pip install /path/to/qualcomm_device_cloud_sdk*.whl

Platform is inferred from --device:
  android  Appium + pytest (Android phones: SM8750 / SM8650 / SM8850)
  linux    BASH (Linux IoT: QCS9075M)

Required environment variables:
  QDC_API_KEY   API key from QDC UI -> Users -> Settings -> API Keys

Usage:
  python run_qdc_jobs.py \\
      --pkg-dir    pkg-snapdragon/llama.cpp \\
      --model-url  https://.../Llama-3.2-1B-Instruct-Q4_0.gguf \\
      --device     SM8750
"""

from __future__ import annotations

import argparse
import enum
import logging
import os
import re
import shutil
import sys
import tempfile
import time
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from qualcomm_device_cloud_sdk.api import qdc_api
from qualcomm_device_cloud_sdk.logging import configure_logging
from qualcomm_device_cloud_sdk.models import (
    ArtifactType,
    JobMode,
    JobState,
    JobSubmissionParameter,
    JobType,
    TestFramework,
)

# configure_logging only sets up the SDK logger; basicConfig is needed for
# our own log.info to reach stdout.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
configure_logging(level=logging.INFO, handlers=[logging.StreamHandler()])
# Silence per-poll GET/status spam from the SDK and its HTTP client.
logging.getLogger("qualcomm_device_cloud").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

POLL_INTERVAL        = 30
JOB_TIMEOUT          = 3600
LOG_UPLOAD_TIMEOUT   = 600
CAPACITY_TIMEOUT     = 1800
CAPACITY_POLL        = 60
MAX_CONCURRENT_JOBS  = 5
DEFAULT_RETRIES      = 0
RETRY_DELAY          = 300
TERMINAL_STATES     = {JobState.COMPLETED, JobState.CANCELED}
NON_TERMINAL_STATES = {JobState.DISPATCHED, JobState.RUNNING, JobState.SETUP, JobState.SUBMITTED}


class DeviceUnavailableError(Exception):
    """Raised when the QDC device resource is not available (retryable)."""


_SCRIPTS_DIR = Path(__file__).parent
_TESTS_DIR = _SCRIPTS_DIR / "tests"

# --- Shared test assets -------------------------------------------------------
_UTILS = _TESTS_DIR / "utils.py"
_CONFTEST = _TESTS_DIR / "conftest.py"
_PYTEST_LINE_RE = re.compile(
    r"(?:[\w/]+\.py::)?(?:\w+::)?([\w\[\].-]+)\s+(PASSED|FAILED|ERROR|SKIPPED)"
)
_EXCLUDED_LOGS = {
    "qdc_android_whole_host-000.log",
    "qdc_kernel_host-000.log",
    "qdc_LE_whole_host-000.log",
    "qdc_LE_kernel_host-000.log",
    "script.log",
}
_NON_TERMINAL_STATE_VALUES = {s.value for s in NON_TERMINAL_STATES}

# --- Android (Appium + pytest) assets ----------------------------------------
_RUN_BENCH = _TESTS_DIR / "run_bench_tests_posix.py"
_RUN_BACKEND_OPS = _TESTS_DIR / "run_backend_ops_posix.py"
_REQUIREMENTS = _SCRIPTS_DIR / "requirements.txt"
_UPSTREAM_ADB_SCRIPTS = (
    "https://raw.githubusercontent.com/ggml-org/llama.cpp/master/scripts/snapdragon/adb"
)
_ADB_SCRIPT_NAMES = [
    "run-bench.sh",
    "run-cli.sh",
    "run-completion.sh",
    "run-tool.sh",
]

# --- Linux (BASH) assets ------------------------------------------------------
_RUN_LINUX_TEMPLATE = _TESTS_DIR / "linux" / "run_linux.sh"
_LINUX_ENTRY_SCRIPT = "/bin/bash /data/local/tmp/TestContent/run_linux.sh"

# =============================================================================
# Artifact builders (per platform)
# =============================================================================


@dataclass
class JobResult:
    passed: bool
    tests: dict[str, bool] = field(default_factory=dict)
    raw_logs: dict[str, str] = field(default_factory=dict)
    failure_details: dict[str, str] = field(default_factory=dict)


def _write_lf(path: Path, content: str) -> None:
    """Write text with LF line endings (required by /bin/bash on Linux)."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)


def _build_android_artifact(
    pkg_dir: Path,
    stage_dir: Path,
    test_mode: str,
    model_url: str | None,
) -> Path:
    """Android zip (Appium/pytest). Extracted by QDC under /qdc/appium/.

    Zip structure:
      llama_cpp_bundle/            installed package (adb pushed to /data/local/tmp/)
      run-{bench,cli,completion,tool}.sh  upstream adb wrappers (patched)
      tests/
        utils.py                   shared adb helpers
        conftest.py                Appium pytest fixtures
        test_bench_posix.py        bench + cli tests (for --test bench or all)
        test_backend_ops_posix.py  test-backend-ops on HTP0
      requirements.txt
      pytest.ini                   addopts = --junitxml=results.xml
    """
    bundle_dir = stage_dir / "llama_cpp_bundle"
    shutil.copytree(pkg_dir, bundle_dir)

    # Download upstream adb scripts so they land at /qdc/appium/ on the QDC
    # runner. They wrap `adb shell` internally. Patch in `chmod +x bin/* lib/*`
    # right after `cd $basedir` so device binaries are executable.
    for name in _ADB_SCRIPT_NAMES:
        url = f"{_UPSTREAM_ADB_SCRIPTS}/{name}"
        dest = stage_dir / name
        log.info("Downloading %s", url)
        urllib.request.urlretrieve(url, str(dest))
        content = dest.read_text()
        content = content.replace(
            "cd $basedir;",
            "cd $basedir; chmod +x bin/* lib/* 2>/dev/null;",
        )
        dest.write_text(content)
        dest.chmod(0o755)

    tests_dir = stage_dir / "tests"
    tests_dir.mkdir()

    shutil.copy(_UTILS, tests_dir / "utils.py")
    shutil.copy(_CONFTEST, tests_dir / "conftest.py")

    if test_mode in ("bench", "all"):
        assert model_url is not None
        (tests_dir / "test_bench_posix.py").write_text(
            _RUN_BENCH.read_text().replace("<<MODEL_URL>>", model_url)
        )
    if test_mode in ("backend-ops", "all"):
        shutil.copy(_RUN_BACKEND_OPS, tests_dir / "test_backend_ops_posix.py")

    shutil.copy(_REQUIREMENTS, stage_dir / "requirements.txt")
    (stage_dir / "pytest.ini").write_text(
        "[pytest]\naddopts = --junitxml=results.xml\n"
    )

    zip_base = str(stage_dir / "artifact")
    shutil.make_archive(zip_base, "zip", stage_dir)
    return Path(f"{zip_base}.zip")


def _build_linux_artifact(
    pkg_dir: Path,
    stage_dir: Path,
    test_mode: str,
    model_url: str | None,
) -> Path:
    """Linux IoT zip (BASH framework). Extracted by QDC to /data/local/tmp/TestContent/.

    Zip structure:
      run_linux.sh               entry script (placeholder-substituted, LF line endings)
      llama_cpp_bundle/          installed package
    """
    bundle_dir = stage_dir / "llama_cpp_bundle"
    shutil.copytree(pkg_dir, bundle_dir)

    template = _RUN_LINUX_TEMPLATE.read_text(encoding="utf-8")
    rendered = template.replace("{MODEL_URL}", model_url or "").replace(
        "{TEST_MODE}", test_mode
    )
    script_path = stage_dir / "run_linux.sh"
    _write_lf(script_path, rendered)
    script_path.chmod(0o755)

    zip_base = str(stage_dir / "artifact")
    shutil.make_archive(zip_base, "zip", stage_dir)
    return Path(f"{zip_base}.zip")


# =============================================================================
# Platform enum + strategy table
# =============================================================================


class Platform(enum.Enum):
    ANDROID = "android"
    LINUX = "linux"


@dataclass(frozen=True)
class PlatformSpec:
    test_framework: TestFramework
    entry_script: str | None
    build_artifact: Callable[[Path, Path, str, str | None], Path]
    job_name_fmt: str


PLATFORM_SPECS: dict[Platform, PlatformSpec] = {
    Platform.ANDROID: PlatformSpec(
        test_framework=TestFramework.APPIUM,
        entry_script=None,
        build_artifact=_build_android_artifact,
        job_name_fmt="{base}",
    ),
    Platform.LINUX: PlatformSpec(
        test_framework=TestFramework.BASH,
        entry_script=_LINUX_ENTRY_SCRIPT,
        build_artifact=_build_linux_artifact,
        job_name_fmt="{base} (Linux)",
    ),
}

DEVICE_PLATFORM: dict[str, Platform] = {
    "SM8750": Platform.ANDROID,
    "SM8650": Platform.ANDROID,
    "SM8850": Platform.ANDROID,
    "QCS9075M": Platform.LINUX,
}


# =============================================================================
# Shared QDC job plumbing
# =============================================================================


def wait_for_job(client, job_id: str, timeout: int) -> str:
    elapsed = 0
    last_state = None
    consecutive_errors = 0
    max_consecutive_errors = 5
    while elapsed < timeout:
        try:
            raw = qdc_api.get_job_status(client, job_id)
            consecutive_errors = 0
        except Exception as e:
            consecutive_errors += 1
            log.warning(
                "Transient error polling job %s (%d/%d): %s",
                job_id,
                consecutive_errors,
                max_consecutive_errors,
                e,
            )
            if consecutive_errors >= max_consecutive_errors:
                raise
            time.sleep(POLL_INTERVAL)
            elapsed += POLL_INTERVAL
            continue
        try:
            status = JobState(raw)
        except ValueError:
            status = raw
        if status in TERMINAL_STATES:
            return raw.lower()
        if raw != last_state:
            log.info("Job %s: %s", job_id, raw)
            last_state = raw
        time.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL
    # Abort to free the QDC concurrency slot instead of leaking it.
    try:
        qdc_api.abort_job(client, job_id)
        log.warning("Aborted job %s after timeout to free concurrency slot", job_id)
    except Exception as e:
        log.warning("Failed to abort job %s: %s", job_id, e)
    raise TimeoutError(f"Job {job_id} did not finish within {timeout}s")


def wait_for_log_upload(client, job_id: str) -> None:
    elapsed = 0
    while elapsed <= LOG_UPLOAD_TIMEOUT:
        try:
            status = (qdc_api.get_job_log_upload_status(client, job_id) or "").lower()
        except Exception as e:
            log.warning("get_job_log_upload_status failed: %s — will retry", e)
            status = ""
        if status in {"completed", "failed"}:
            return
        log.info("Waiting for log upload (status=%s) ...", status)
        time.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL
    log.warning("Timed out waiting for log upload after %ds", LOG_UPLOAD_TIMEOUT)


def wait_for_capacity(client, max_jobs: int = MAX_CONCURRENT_JOBS) -> None:
    """Block until the user's active (non-terminal) QDC job count is below max_jobs."""
    elapsed = 0
    while elapsed < CAPACITY_TIMEOUT:
        jobs_page = qdc_api.get_jobs_list(client, page_number=0, page_size=50)
        if jobs_page is None:
            log.warning(
                "Could not retrieve job list; proceeding without capacity check"
            )
            return
        items = getattr(jobs_page, "data", []) or []
        active = sum(
            1 for j in items if getattr(j, "state", None) in _NON_TERMINAL_STATE_VALUES
        )
        if active < max_jobs:
            log.info("Active QDC jobs: %d / %d — proceeding", active, max_jobs)
            return
        log.info(
            "Active QDC jobs: %d / %d — waiting %ds ...",
            active,
            max_jobs,
            CAPACITY_POLL,
        )
        time.sleep(CAPACITY_POLL)
        elapsed += CAPACITY_POLL
    raise TimeoutError(
        f"Capacity wait timed out after {CAPACITY_TIMEOUT}s"
    )


# ---------------------------------------------------------------------------
# Log parsing helpers
# ---------------------------------------------------------------------------


def _parse_junit_xml(content: str) -> tuple[dict[str, bool], dict[str, str]]:
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return {}, {}
    results: dict[str, bool] = {}
    failures: dict[str, str] = {}
    for tc in root.iter("testcase"):
        name = tc.get("name", "")
        if classname := tc.get("classname", ""):
            name = f"{classname}.{name}"
        failure_el = tc.find("failure")
        if failure_el is None:
            failure_el = tc.find("error")
        results[name] = failure_el is None
        if failure_el is not None:
            parts = [failure_el.get("message", ""), failure_el.text or ""]
            failures[name] = "\n".join(p for p in parts if p).strip()
    return results, failures


def _parse_pytest_output(content: str) -> dict[str, bool]:
    results: dict[str, bool] = {}
    for m in _PYTEST_LINE_RE.finditer(content):
        results[m.group(1)] = m.group(2) == "PASSED"
    return results


def fetch_logs_and_parse_tests(
    client, job_id: str, max_retries: int = 5, retry_delay: int = 30
) -> tuple[dict[str, bool], dict[str, str], dict[str, str]]:
    """Returns (test_results, raw_logs, failure_details)."""
    log_files = None
    for attempt in range(1, max_retries + 1):
        try:
            log_files = qdc_api.get_job_log_files(client, job_id)
            break
        except Exception as e:
            if attempt < max_retries:
                log.warning(
                    "get_job_log_files failed (attempt %d/%d): %s — retrying in %ds",
                    attempt, max_retries, e, retry_delay,
                )
                time.sleep(retry_delay)
            else:
                log.error(
                    "get_job_log_files failed after %d attempts: %s", max_retries, e
                )
                return {}, {}, {}
    if not log_files:
        log.warning("No log files returned for job %s", job_id)
        return {}, {}, {}

    test_results: dict[str, bool] = {}
    pytest_fallback: dict[str, bool] = {}
    raw_logs: dict[str, str] = {}
    failure_details: dict[str, str] = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        for lf in log_files:
            zip_path = os.path.join(tmpdir, "log.zip")
            log.info("Downloading log file: %s", lf.filename)
            qdc_api.download_job_log_files(client, lf.filename, zip_path)
            try:
                shutil.unpack_archive(zip_path, tmpdir, "zip")
            except Exception as e:
                log.warning("Could not unpack %s as zip: %s", lf.filename, e)

        for root_dir, _, files in os.walk(tmpdir):
            for fname in sorted(files):
                fpath = os.path.join(root_dir, fname)
                content = Path(fpath).read_text(errors="replace")
                if fname.endswith(".xml"):
                    results, failures = _parse_junit_xml(content)
                    test_results.update(results)
                    failure_details.update(failures)
                elif fname.endswith(".log"):
                    if fname in _EXCLUDED_LOGS:
                        continue
                    log.info("--- %s ---\n%s", fname, content)
                    raw_logs[fname] = content
                    pytest_fallback.update(_parse_pytest_output(content))

    return (
        (test_results if test_results else pytest_fallback),
        raw_logs,
        failure_details,
    )


def write_summary(result: JobResult, title: str = "QDC Test Results") -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return

    icon = "✅" if result.passed else "❌"

    lines = [
        f"## {title}\n",
        f"Overall: {icon} {'PASSED' if result.passed else 'FAILED'}\n",
    ]
    reportable = {n: ok for n, ok in result.tests.items() if "test_install" not in n}
    if reportable:
        lines += ["| Test | Result |", "| ---- | ------ |"]
        for name, ok in reportable.items():
            lines.append(f"| `{name}` | {'✅' if ok else '❌'} |")
        passed_n = sum(1 for v in reportable.values() if v)
        failed_n = sum(1 for v in reportable.values() if not v)
        lines += ["", f"**{passed_n} passed, {failed_n} failed**"]
    else:
        lines.append("_No per-test data available._")

    failed_names = [n for n, ok in reportable.items() if not ok]
    if failed_names:
        lines += ["", "### Failures"]
        for name in failed_names:
            detail = result.failure_details.get(name)
            if detail:
                lines += [
                    f"<details><summary><code>{name}</code></summary>",
                    "",
                    "```",
                    detail,
                    "```",
                    "",
                    "</details>",
                ]

    if result.raw_logs:
        lines += ["", "### Raw Logs"]
        for fname, content in sorted(result.raw_logs.items()):
            lines += [
                f"<details><summary>{fname}</summary>",
                "",
                "```",
                content.rstrip(),
                "```",
                "",
                "</details>",
            ]

    with open(summary_path, "a") as f:
        f.write("\n".join(lines) + "\n")


# =============================================================================
# CLI + main
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--pkg-dir", required=True, type=Path,
                   help="Installed llama.cpp package directory (contains bin/ and lib/)")
    p.add_argument("--model-url",
                   help="Direct URL to the GGUF model file (required for --test bench)")
    p.add_argument("--device", required=True,
                   help="QDC chipset name, e.g. SM8750")
    p.add_argument("--test", choices=["bench", "backend-ops", "all"], default="bench",
                   help="Test suite to run (default: bench)")
    p.add_argument("--job-timeout", type=int, default=JOB_TIMEOUT, metavar="SECONDS",
                   help=f"Max seconds to wait for job completion (default: {JOB_TIMEOUT})")
    p.add_argument("--retries", type=int, default=DEFAULT_RETRIES, metavar="N",
                   help="Number of retries when device is unavailable (default: 0)")
    p.add_argument("--retry-delay", type=int, default=RETRY_DELAY, metavar="SECONDS",
                   help=f"Seconds to wait between retries (default: {RETRY_DELAY})")
    args = p.parse_args()
    if args.test in ("bench", "all") and not args.model_url:
        p.error("--model-url is required when --test bench or --test all")
    return args


def _submit_and_run_job(client, args, spec, target_id, artifact_id) -> JobResult:
    """Submit a QDC job and wait for results.

    Raises DeviceUnavailableError for transient device/resource issues that
    are worth retrying. Returns JobResult for definitive outcomes (pass or
    test failure).
    """
    try:
        wait_for_capacity(client)
    except TimeoutError:
        raise DeviceUnavailableError("Capacity wait timed out — device busy")

    job_name = spec.job_name_fmt.format(base="llama.cpp Hexagon tests")

    job_id = qdc_api.submit_job(
        public_api_client=client,
        target_id=target_id,
        job_name=job_name,
        external_job_id=None,
        job_type=JobType.AUTOMATED,
        job_mode=JobMode.APPLICATION,
        timeout=max(1, args.job_timeout // 60),
        test_framework=spec.test_framework,
        entry_script=spec.entry_script,
        job_artifacts=[artifact_id],
        monkey_events=None,
        monkey_session_timeout=None,
        job_parameters=[JobSubmissionParameter.WIFIENABLED],
    )
    if job_id is None:
        raise DeviceUnavailableError("Job submission failed — device may be unavailable")
    log.info("Job submitted: %s  (device=%s)", job_id, args.device)

    try:
        job_status = wait_for_job(client, job_id, timeout=args.job_timeout)
    except TimeoutError as e:
        raise DeviceUnavailableError(str(e))
    log.info("Job %s finished: %s", job_id, job_status)

    wait_for_log_upload(client, job_id)
    tests, raw_logs, failure_details = fetch_logs_and_parse_tests(client, job_id)

    job_ok = job_status == JobState.COMPLETED.value.lower()

    if not job_ok and not tests:
        raise DeviceUnavailableError(
            f"Job did not complete (status={job_status}) and produced no test results"
        )

    passed = job_ok and all(tests.values()) if tests else job_ok
    if spec.test_framework == TestFramework.BASH and not tests:
        log.error("No test results recovered (state=%s). Script likely never ran.", job_status)
        passed = False
    if not passed:
        log.error("Job did not complete successfully or tests failed (status=%s)", job_status)

    return JobResult(passed=passed, tests=tests, raw_logs=raw_logs, failure_details=failure_details)


def main() -> int:
    args = parse_args()

    platform = DEVICE_PLATFORM.get(args.device)
    if platform is None:
        log.error(
            "Unknown device %r. Known: %s",
            args.device, ", ".join(sorted(DEVICE_PLATFORM.keys())),
        )
        return 1
    spec = PLATFORM_SPECS[platform]

    api_key = os.environ.get("QDC_API_KEY")
    if not api_key:
        log.error("QDC_API_KEY environment variable must be set")
        return 1
    if not args.pkg_dir.is_dir():
        log.error("--pkg-dir %s does not exist", args.pkg_dir)
        return 1

    client = qdc_api.get_public_api_client_using_api_key(
        api_key_header=api_key,
        app_name_header="llama-cpp-ci",
        on_behalf_of_header="llama-cpp-ci",
        client_type_header="Python",
    )

    target_id = qdc_api.get_target_id(client, args.device)
    if target_id is None:
        log.error("Could not find QDC target for device %r", args.device)
        return 1

    with tempfile.TemporaryDirectory() as tmpdir:
        log.info("Building %s artifact (test=%s) ...", platform.value, args.test)
        zip_path = spec.build_artifact(
            args.pkg_dir, Path(tmpdir), args.test, args.model_url
        )
        log.info("Uploading artifact (%d MB) ...", zip_path.stat().st_size // 1_000_000)
        artifact_id = qdc_api.upload_file(client, str(zip_path), ArtifactType.TESTSCRIPT)

    if artifact_id is None:
        log.error("Artifact upload failed")
        return 1

    max_attempts = 1 + args.retries
    for attempt in range(1, max_attempts + 1):
        try:
            result = _submit_and_run_job(client, args, spec, target_id, artifact_id)
            break
        except DeviceUnavailableError as e:
            if attempt < max_attempts:
                log.warning(
                    "Attempt %d/%d failed (device unavailable): %s — retrying in %ds",
                    attempt, max_attempts, e, args.retry_delay,
                )
                time.sleep(args.retry_delay)
            else:
                log.error(
                    "Attempt %d/%d failed (device unavailable): %s — no retries left",
                    attempt, max_attempts, e,
                )
                write_summary(
                    JobResult(passed=False, tests={}),
                    title=f"QDC Device Unavailable ({args.device})",
                )
                return 1
    else:
        return 1

    if args.test == "backend-ops":
        title = f"Backend Ops — HTP0 ({args.device})"
    elif args.test == "all":
        title = f"QDC Tests ({args.device})"
    else:
        title = f"QDC Test Results ({args.device})"
    write_summary(result, title=title)

    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
