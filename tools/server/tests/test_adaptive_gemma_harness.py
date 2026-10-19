"""Host-only contracts for the offline Gemma server qualification command."""

import argparse
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import test_adaptive_gemma as harness


class Commands(unittest.TestCase):
    def test_shared_assistant_types_are_inherited_and_native_controls_match(self):
        args = argparse.Namespace(server=Path("server"), model=Path("model"), assistant=Path("assistant"),
                                  context=4096, batch_size=256, ubatch_size=256, arena_mib=1024)
        for depth in (1, 2, 3):
            for streamed in (False, True):
                command = harness.server_command(args, streamed, depth, 9999)
                self.assertEqual(command[command.index("--spec-draft-n-max")+1], str(depth))
                self.assertEqual(command[command.index("--model-draft")+1], str(args.assistant.resolve()))
                self.assertEqual(command.count("--kv-stream-arena-mib"), int(streamed))
                if streamed:
                    self.assertNotIn("--spec-draft-type-k", command)
                    self.assertNotIn("--spec-draft-type-v", command)
                else:
                    self.assertEqual(command[command.index("--spec-draft-type-k")+1], "q8_0")
                    self.assertEqual(command[command.index("--spec-draft-type-v")+1], "q4_0")
                self.assertNotIn("--no-warmup", command)
                self.assertIn("--offline", command)
                self.assertEqual(command[command.index("--host")+1], "127.0.0.1")

    @unittest.skipUnless(os.name == "posix", "process-group cleanup is POSIX-only")
    def test_owned_server_cleanup_escalates_and_does_not_inherit_uvm(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = argparse.Namespace(server=Path("server"), model=Path("model"), assistant=Path("assistant"),
                                      context=4096, batch_size=256, ubatch_size=256, arena_mib=1024,
                                      output=Path(temporary))
            process = Mock(pid=987654)
            process.wait.side_effect = [subprocess.TimeoutExpired("server", 20), 0]
            with patch.object(harness.subprocess, "Popen", return_value=process) as launch, \
                    patch.object(harness, "wait_for_health"), patch.object(harness.os, "killpg") as kill, \
                    patch.dict(os.environ, {"GGML_CUDA_ENABLE_UNIFIED_MEMORY": "1", "LLAMA_ARG_CTX_SIZE": "65536"}):
                with self.assertRaisesRegex(RuntimeError, "caller failure"):
                    with harness.server(args, True, 3):
                        raise RuntimeError("caller failure")
            self.assertTrue(launch.call_args.kwargs["start_new_session"])
            self.assertNotIn("GGML_CUDA_ENABLE_UNIFIED_MEMORY", launch.call_args.kwargs["env"])
            self.assertNotIn("LLAMA_ARG_CTX_SIZE", launch.call_args.kwargs["env"])
            self.assertEqual(kill.call_args_list[0].args, (process.pid, signal.SIGTERM))
            self.assertEqual(kill.call_args_list[1].args, (process.pid, signal.SIGKILL))
            self.assertEqual(process.wait.call_count, 2)


if __name__ == "__main__":
    unittest.main()
