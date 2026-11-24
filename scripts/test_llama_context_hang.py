#!/usr/bin/env python3

import argparse
import ctypes
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def write_memory_dump(pid: int, dump_path: Path) -> None:
    if sys.platform != "win32":
        raise OSError("Memory dumps are currently supported only on Windows")

    import msvcrt
    from ctypes import wintypes

    process_query_information = 0x0400
    process_vm_read = 0x0010
    minidump_with_full_memory = 0x00000002

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    dbghelp = ctypes.WinDLL("Dbghelp", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = wintypes.BOOL
    dbghelp.MiniDumpWriteDump.argtypes = [
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    dbghelp.MiniDumpWriteDump.restype = wintypes.BOOL

    process = kernel32.OpenProcess(process_query_information | process_vm_read, False, pid)
    if not process:
        raise ctypes.WinError(ctypes.get_last_error())

    try:
        with dump_path.open("wb") as dump_file:
            file_handle = msvcrt.get_osfhandle(dump_file.fileno())
            success = dbghelp.MiniDumpWriteDump(
                process,
                pid,
                file_handle,
                minidump_with_full_memory,
                None,
                None,
                None,
            )
            if not success:
                raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel32.CloseHandle(process)


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        process.terminate()
    except OSError:
        if process.poll() is None:
            raise
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def find_descendant_process(parent_pid: int, executable_name: str):
    if sys.platform != "win32":
        return None

    class ProcessEntry32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", ctypes.c_ulong),
            ("cntUsage", ctypes.c_ulong),
            ("th32ProcessID", ctypes.c_ulong),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", ctypes.c_ulong),
            ("cntThreads", ctypes.c_ulong),
            ("th32ParentProcessID", ctypes.c_ulong),
            ("pcPriClassBase", ctypes.c_long),
            ("dwFlags", ctypes.c_ulong),
            ("szExeFile", ctypes.c_wchar * 260),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateToolhelp32Snapshot.argtypes = [ctypes.c_ulong, ctypes.c_ulong]
    kernel32.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
    kernel32.Process32FirstW.argtypes = [ctypes.c_void_p, ctypes.POINTER(ProcessEntry32W)]
    kernel32.Process32FirstW.restype = ctypes.c_int
    kernel32.Process32NextW.argtypes = [ctypes.c_void_p, ctypes.POINTER(ProcessEntry32W)]
    kernel32.Process32NextW.restype = ctypes.c_int
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.CloseHandle.restype = ctypes.c_int

    snapshot = kernel32.CreateToolhelp32Snapshot(0x00000002, 0)
    if snapshot == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())

    processes = []
    try:
        entry = ProcessEntry32W()
        entry.dwSize = ctypes.sizeof(entry)
        has_entry = kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
        while has_entry:
            processes.append((entry.th32ProcessID, entry.th32ParentProcessID, entry.szExeFile))
            has_entry = kernel32.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)

    descendant_pids = {parent_pid}
    for _ in range(len(processes)):
        for process_pid, process_parent_pid, process_name in processes:
            if process_parent_pid in descendant_pids and process_pid not in descendant_pids:
                if process_name.casefold() == executable_name.casefold():
                    return process_pid
                descendant_pids.add(process_pid)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run test-llama-context repeatedly and capture hangs."
    )
    parser.add_argument("--runner", choices=("direct", "ctest"), default="direct")
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--runs", required=True, type=int)
    parser.add_argument("--timeout", type=float, default=60, help="Timeout in seconds (default: 60)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("test-llama-context-hang-logs"),
        help="Directory for per-run logs and memory dumps",
    )
    args = parser.parse_args()

    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if args.timeout <= 0:
        parser.error("--timeout must be greater than 0")

    if args.runner == "direct":
        if args.executable is None or args.models is None:
            parser.error("direct runner requires --executable and --models")
        executable = args.executable.resolve()
        models = args.models.resolve()
        if not executable.is_file():
            parser.error(f"test executable not found: {executable}")
        if not models.is_dir():
            parser.error(f"models directory not found: {models}")
        command = [str(executable), "--models", str(models)]
        working_directory = None
    else:
        if args.build_dir is None:
            parser.error("ctest runner requires --build-dir")
        working_directory = args.build_dir.resolve()
        if not (working_directory / "CTestTestfile.cmake").is_file():
            parser.error(f"CTest build directory not found: {working_directory}")
        ctest = shutil.which("ctest")
        if ctest is None:
            parser.error("ctest executable not found in PATH")
        command = [ctest, "-C", "Release", "--output-on-failure", "-L", "main", "-E", "test-opt"]

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    process_env = os.environ.copy()
    for proxy_name in ("http_proxy", "https_proxy"):
        proxy_value = process_env.get(proxy_name) or process_env.get(proxy_name.upper())
        if proxy_value:
            process_env[proxy_name] = proxy_value
            process_env[proxy_name.upper()] = proxy_value

    hang_count = 0
    failure_count = 0

    for run_number in range(1, args.runs + 1):
        log_path = output_dir / f"run-{run_number:03}.log"
        print(f"[{run_number}/{args.runs}] Starting; log: {log_path}", flush=True)

        with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
            log_file.write(f"Command: {subprocess.list2cmdline(command)}{os.linesep}")
            log_file.flush()
            process = subprocess.Popen(
                command,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=working_directory,
                env=process_env,
            )
            started_at = time.monotonic()
            try:
                return_code = process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                elapsed = time.monotonic() - started_at
                hang_count += 1
                dump_path = output_dir / f"run-{run_number:03}.dmp"
                log_file.write(f"\nTimed out after {elapsed:.2f} seconds; PID {process.pid}\n")
                log_file.flush()
                print(
                    f"[{run_number}/{args.runs}] HANG detected after {elapsed:.2f} seconds; "
                    f"creating {dump_path}",
                    flush=True,
                )
                try:
                    dump_pid = process.pid
                    if args.runner == "ctest":
                        dump_pid = find_descendant_process(process.pid, "test-llama-context.exe")
                    if dump_pid is None:
                        message = "test-llama-context process not found; no memory dump captured"
                        log_file.write(f"{message}\n")
                        print(f"[{run_number}/{args.runs}] {message}", flush=True)
                    else:
                        write_memory_dump(dump_pid, dump_path)
                        log_file.write(f"Memory dump for PID {dump_pid}: {dump_path}\n")
                        print(f"[{run_number}/{args.runs}] Memory dump for PID {dump_pid} saved", flush=True)
                except Exception as error:
                    log_file.write(f"Memory dump failed: {error}\n")
                    print(f"[{run_number}/{args.runs}] Memory dump failed: {error}", flush=True)
                finally:
                    stop_process(process)
            else:
                if return_code != 0:
                    failure_count += 1
                log_file.write(f"\nFinished with exit code {return_code}\n")
                log_file.flush()
                print(
                    f"[{run_number}/{args.runs}] Finished in "
                    f"{time.monotonic() - started_at:.2f} seconds "
                    f"with exit code {return_code}",
                    flush=True,
                )

    print(f"Hangs: {hang_count}/{args.runs}; failed runs: {failure_count}/{args.runs}")
    return 1 if hang_count or failure_count else 0


if __name__ == "__main__":
    raise SystemExit(main())