# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
"""Run llama.cpp Hexagon Android tests in a single QDC Appium job.

Bundles test scripts into one artifact and submits a single QDC job:

  1. run_bench_tests_posix.py — llama-cli and llama-bench on CPU / GPU / NPU
                                (from scripts/snapdragon/qdc/)

Results are written to $GITHUB_STEP_SUMMARY when set (GitHub Actions).

Prerequisites:
  pip install /path/to/qualcomm_device_cloud_sdk*.whl

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
import logging
import os
import re
import shutil
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

from qualcomm_device_cloud_sdk.api import qdc_api
from qualcomm_device_cloud_sdk.logging import configure_logging
from qualcomm_device_cloud_sdk.models import ArtifactType, JobMode, JobState, JobSubmissionParameter, JobType, TestFramework

configure_logging(level=logging.INFO, handlers=[logging.StreamHandler()])
log = logging.getLogger(__name__)

POLL_INTERVAL        = 30
JOB_TIMEOUT          = 3600
LOG_UPLOAD_TIMEOUT   = 600
CAPACITY_TIMEOUT     = 1800
CAPACITY_POLL        = 60
MAX_CONCURRENT_JOBS  = 5
TERMINAL_STATES     = {JobState.COMPLETED, JobState.CANCELED}
NON_TERMINAL_STATES = {JobState.DISPATCHED, JobState.RUNNING, JobState.SETUP, JobState.SUBMITTED}

_SCRIPTS_DIR      = Path(__file__).parent
_QDC_DIR          = _SCRIPTS_DIR / "snapdragon" / "qdc"
_TESTS_DIR        = _QDC_DIR / "tests"
_RUN_BENCH        = _TESTS_DIR / "run_bench_tests_posix.py"
_RUN_BACKEND_OPS  = _TESTS_DIR / "run_backend_ops_posix.py"
_UTILS            = _TESTS_DIR / "utils.py"
_CONFTEST         = _TESTS_DIR / "conftest.py"
_REQUIREMENTS     = _QDC_DIR / "requirements.txt"

_PYTEST_LINE_RE = re.compile(
    r"(?:[\w/]+\.py::)?(?:\w+::)?(\w+)\s+(PASSED|FAILED|ERROR|SKIPPED)"
)
_BACKEND_OPS_LINE_RE = re.compile(
    r"^\s+([A-Z][A-Z_0-9]+)\s.*?(OK|FAIL|PASS)", re.MULTILINE
)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class JobResult:
    passed: bool
    tests: dict[str, bool] = field(default_factory=dict)
    raw_logs: dict[str, str] = field(default_factory=dict)
    op_details: dict[str, bool] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Artifact creation
# ---------------------------------------------------------------------------

def build_artifact_zip(
    pkg_dir: Path,
    stage_dir: Path,
    *,
    test_mode: str = "bench",
    model_url: str | None = None,
) -> Path:
    """Bundle everything into a single QDC artifact zip.

    Zip structure (extracted by QDC to /qdc/appium/ on the runner):
      llama_cpp_bundle/            installed package (adb pushed to /data/local/tmp/)
      tests/
        utils.py                   shared helpers (paths, run_adb_command, …)
        conftest.py                shared pytest fixtures (driver)
        test_bench_posix.py        bench + cli tests (<<MODEL_URL>> substituted)
          AND/OR
        test_backend_ops_posix.py  test-backend-ops -b HTP0
      requirements.txt
    """
    shutil.copytree(pkg_dir, stage_dir / "llama_cpp_bundle")

    tests_dir = stage_dir / "tests"
    tests_dir.mkdir()

    shutil.copy(_UTILS,    tests_dir / "utils.py")
    shutil.copy(_CONFTEST, tests_dir / "conftest.py")

    if test_mode in ("bench", "all"):
        (tests_dir / "test_bench_posix.py").write_text(
            _RUN_BENCH.read_text().replace("<<MODEL_URL>>", model_url)
        )
    if test_mode in ("backend-ops", "all"):
        shutil.copy(_RUN_BACKEND_OPS, tests_dir / "test_backend_ops_posix.py")

    shutil.copy(_REQUIREMENTS, stage_dir / "requirements.txt")

    zip_base = str(stage_dir / "artifact")
    shutil.make_archive(zip_base, "zip", stage_dir)
    return Path(f"{zip_base}.zip")


# ---------------------------------------------------------------------------
# Job lifecycle
# ---------------------------------------------------------------------------

def wait_for_job(client, job_id: str, timeout: int) -> str:
    elapsed = 0
    while elapsed < timeout:
        raw = qdc_api.get_job_status(client, job_id)
        try:
            status = JobState(raw)
        except ValueError:
            status = raw
        if status in TERMINAL_STATES:
            return raw.lower()
        log.info("Job %s: %s", job_id, raw)
        time.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL
    raise TimeoutError(f"Job {job_id} did not finish within {timeout}s")


def wait_for_log_upload(client, job_id: str) -> None:
    elapsed = 0
    while elapsed <= LOG_UPLOAD_TIMEOUT:
        status = (qdc_api.get_job_log_upload_status(client, job_id) or "").lower()
        if status in {"completed", "nologs", "failed"}:
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
            log.warning("Could not retrieve job list; proceeding without capacity check")
            return
        items = getattr(jobs_page, "items", []) or []
        active = sum(1 for j in items if getattr(j, "state", None) in NON_TERMINAL_STATES)
        if active < max_jobs:
            log.info("Active QDC jobs: %d / %d — proceeding", active, max_jobs)
            return
        log.info("Active QDC jobs: %d / %d — waiting %ds ...", active, max_jobs, CAPACITY_POLL)
        time.sleep(CAPACITY_POLL)
        elapsed += CAPACITY_POLL
    log.warning("Capacity wait timed out after %ds; proceeding anyway", CAPACITY_TIMEOUT)


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

def _parse_junit_xml(content: str) -> dict[str, bool]:
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return {}
    results: dict[str, bool] = {}
    for tc in root.iter("testcase"):
        name = tc.get("name", "")
        if classname := tc.get("classname", ""):
            name = f"{classname}.{name}"
        results[name] = tc.find("failure") is None and tc.find("error") is None
    return results


def _parse_pytest_output(content: str) -> dict[str, bool]:
    results: dict[str, bool] = {}
    for m in _PYTEST_LINE_RE.finditer(content):
        results[m.group(1)] = m.group(2) == "PASSED"
    return results


def _parse_backend_ops_output(content: str) -> dict[str, bool]:
    results: dict[str, bool] = {}
    for m in _BACKEND_OPS_LINE_RE.finditer(content):
        op, status = m.group(1), m.group(2)
        if status == "FAIL":
            results[op] = False       # FAIL always wins
        elif op not in results:
            results[op] = True
    return results


def fetch_logs_and_parse_tests(
    client, job_id: str, test_mode: str = "bench"
) -> tuple[dict[str, bool], dict[str, str], dict[str, bool]]:
    """Returns (test_results, raw_logs, op_details)."""
    log_files = qdc_api.get_job_log_files(client, job_id)
    if not log_files:
        log.warning("No log files returned for job %s", job_id)
        return {}, {}, {}

    test_results: dict[str, bool] = {}
    pytest_fallback: dict[str, bool] = {}
    raw_logs: dict[str, str] = {}
    op_details: dict[str, bool] = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        for lf in log_files:
            zip_path = os.path.join(tmpdir, "log.zip")
            qdc_api.download_job_log_files(client, lf.filename, zip_path)
            shutil.unpack_archive(zip_path, tmpdir, "zip")

        for root_dir, _, files in os.walk(tmpdir):
            for fname in sorted(files):
                fpath = os.path.join(root_dir, fname)
                content = Path(fpath).read_text(errors="replace")
                if fname.endswith(".xml"):
                    test_results.update(_parse_junit_xml(content))
                elif fname.endswith(".log"):
                    log.info("--- %s ---", fname)
                    print(content)
                    raw_logs[fname] = content
                    if test_mode in ("backend-ops", "all"):
                        op_details.update(_parse_backend_ops_output(content))
                    pytest_fallback.update(_parse_pytest_output(content))

    return (test_results if test_results else pytest_fallback), raw_logs, op_details


# ---------------------------------------------------------------------------
# GitHub Actions summary
# ---------------------------------------------------------------------------

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

    if result.op_details:
        lines += ["", "### Op Details", "| Op | Result |", "| -- | ------ |"]
        for op, ok in sorted(result.op_details.items()):
            lines.append(f"| `{op}` | {'✅' if ok else '❌'} |")
        passed_ops = sum(1 for v in result.op_details.values() if v)
        failed_ops = sum(1 for v in result.op_details.values() if not v)
        lines += ["", f"**{passed_ops} ops passed, {failed_ops} ops failed**"]

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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--pkg-dir",   required=True, type=Path,
                   help="Installed llama.cpp package directory (contains bin/ and lib/)")
    p.add_argument("--model-url",
                   help="Direct URL to the GGUF model file (required for --test bench)")
    p.add_argument("--device",    required=True,
                   help="QDC chipset name, e.g. SM8750")
    p.add_argument("--test", choices=["bench", "backend-ops", "all"], default="bench",
                   help="Test suite to run (default: bench)")
    p.add_argument("--job-timeout", type=int, default=JOB_TIMEOUT, metavar="SECONDS",
                   help=f"Max seconds to wait for job completion (default: {JOB_TIMEOUT})")
    args = p.parse_args()
    if args.test in ("bench", "all") and not args.model_url:
        p.error("--model-url is required when --test bench or --test all")
    return args


def main() -> int:
    args = parse_args()

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
        log.info("Building artifact ...")
        zip_path = build_artifact_zip(
            args.pkg_dir, Path(tmpdir),
            test_mode=args.test, model_url=args.model_url,
        )
        log.info("Uploading artifact (%d MB) ...", zip_path.stat().st_size // 1_000_000)
        artifact_id = qdc_api.upload_file(client, str(zip_path), ArtifactType.TESTSCRIPT)

    if artifact_id is None:
        log.error("Artifact upload failed")
        return 1

    wait_for_capacity(client)

    job_id = qdc_api.submit_job(
        public_api_client=client,
        target_id=target_id,
        job_name="llama.cpp Hexagon tests",
        external_job_id=None,
        job_type=JobType.AUTOMATED,
        job_mode=JobMode.APPLICATION,
        timeout=max(1, args.job_timeout // 60),
        test_framework=TestFramework.APPIUM,
        entry_script=None,
        job_artifacts=[artifact_id],
        monkey_events=None,
        monkey_session_timeout=None,
        job_parameters=[JobSubmissionParameter.WIFIENABLED],
    )
    if job_id is None:
        log.error("Job submission failed")
        return 1
    log.info("Job submitted: %s  (device=%s)", job_id, args.device)

    try:
        job_status = wait_for_job(client, job_id, timeout=args.job_timeout)
    except TimeoutError as e:
        log.error("%s", e)
        write_summary(JobResult(passed=False, tests={}), title=f"QDC Job Timed Out ({args.device})")
        return 1
    log.info("Job %s finished: %s", job_id, job_status)

    wait_for_log_upload(client, job_id)
    tests, raw_logs, op_details = fetch_logs_and_parse_tests(client, job_id, test_mode=args.test)

    passed = job_status == JobState.COMPLETED.value.lower()
    if not passed:
        log.error("Job did not complete successfully (status=%s)", job_status)

    result = JobResult(passed=passed, tests=tests, raw_logs=raw_logs, op_details=op_details)
    if args.test == "backend-ops":
        title = f"Backend Ops — HTP0 ({args.device})"
    elif args.test == "all":
        title = f"QDC Tests ({args.device})"
    else:
        title = f"QDC Test Results ({args.device})"
    write_summary(result, title=title)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
