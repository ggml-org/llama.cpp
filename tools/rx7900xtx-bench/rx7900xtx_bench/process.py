from __future__ import annotations

from pathlib import Path
import signal
import subprocess
import time

from .server_client import ServerClient


class ServerProcess:
    def __init__(self, command: list[str], log_path: Path) -> None:
        self.command = command
        self.log_path = log_path
        self.process: subprocess.Popen[bytes] | None = None
        self.log_file = None

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_path.open("wb")
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.DEVNULL,
            stdout=self.log_file,
            stderr=subprocess.STDOUT,
        )

    def wait_ready(self, client: ServerClient, timeout_seconds: float) -> None:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(f"llama-server exited with code {self.process.returncode}; see {self.log_path}")
            if client.health():
                return
            time.sleep(0.5)
        raise TimeoutError(f"llama-server did not become ready within {timeout_seconds} seconds; see {self.log_path}")

    def log_text(self) -> str:
        if self.log_file is not None:
            self.log_file.flush()
        return self.log_path.read_text(encoding="utf-8", errors="replace")

    def wait_mtp(self, timeout_seconds: float) -> None:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            log_text = self.log_text()
            if "failed to initialize speculative decoding" in log_text:
                raise RuntimeError("llama-server reported speculative decoding initialization failure")
            if "creating MTP draft context" in log_text or "adding speculative implementation 'draft-mtp'" in log_text:
                return
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(f"llama-server exited with code {self.process.returncode}; see {self.log_path}")
            time.sleep(0.1)
        raise TimeoutError(f"llama-server did not log draft-mtp initialization; see {self.log_path}")

    def stop(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGINT)
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None
