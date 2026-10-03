from pathlib import Path
import signal
import unittest

from rx7900xtx_bench.process import ServerProcess


class FakeProcess:
    def __init__(self, running: bool) -> None:
        self.running = running
        self.signal: signal.Signals | None = None
        self.wait_calls = 0

    def poll(self):
        return None if self.running else 0

    def send_signal(self, value: signal.Signals) -> None:
        self.signal = value
        self.running = False

    def wait(self, timeout: float | None = None) -> None:
        self.wait_calls += 1


class ProcessTest(unittest.TestCase):
    def test_stop_signals_only_running_owned_process(self):
        server = ServerProcess(["llama-server"], Path("server.log"))
        process = FakeProcess(running=True)
        server.process = process

        server.stop()

        self.assertEqual(process.signal, signal.SIGINT)
        self.assertEqual(process.wait_calls, 1)

    def test_stop_does_not_signal_exited_process(self):
        server = ServerProcess(["llama-server"], Path("server.log"))
        process = FakeProcess(running=False)
        server.process = process

        server.stop()

        self.assertIsNone(process.signal)
        self.assertEqual(process.wait_calls, 0)


if __name__ == "__main__":
    unittest.main()
