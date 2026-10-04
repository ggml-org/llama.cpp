#!/usr/bin/env python3
"""Host-only tests for vision qualification launch and cleanup contracts."""

import argparse
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import test_adaptive_vision as harness


class Launch(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.args = argparse.Namespace(server=root/'server', model=root/'model', mmproj=root/'projector',
            output=root, context=8192, batch_size=64, ubatch_size=64, cache_ram_mib=2048,
            arena_mib=1024, mtp_length=3, stock_uvm=False)
        self.commands = []

    def launch(self, diagnostic, timeout=False):
        def popen(command, **kwargs):
            self.commands.append((command, kwargs))
            kwargs['stdout'].write(diagnostic)
            kwargs['stdout'].flush()
            class Process:
                pid = 987654
                waits = 0
                def poll(self):
                    return None
                def wait(self, timeout=None):
                    self.waits += 1
                    if self.waits == 1 and timeout:
                        raise subprocess.TimeoutExpired(command, timeout)
                    return 1
                def terminate(self):
                    pass
                def kill(self):
                    pass
            process = Process()
            if not timeout:
                process.waits = 1
            return process
        return patch.object(harness.subprocess, 'Popen', side_effect=popen)

    def test_launch_sets_matching_mtp_kv_and_arena_disables_inherited_uvm(self):
        with self.launch('vision arena rejects this configuration'), patch.dict(os.environ, {'GGML_CUDA_ENABLE_UNIFIED_MEMORY': '1'}), \
                patch.object(harness.os, 'killpg', create=True):
            with harness.server(self.args, 'arena-reject-parallel', reject=True):
                pass
        command, kwargs = self.commands[0]
        self.assertEqual(command[command.index('--spec-draft-n-max')+1], '3')
        self.assertEqual(command[command.index('--spec-draft-type-k')+1], 'q8_0')
        self.assertEqual(command[command.index('--spec-draft-type-v')+1], 'q4_0')
        self.assertNotIn('GGML_CUDA_ENABLE_UNIFIED_MEMORY', kwargs['env'])

    def test_stock_uvm_is_explicit_and_does_not_enable_arena_uvm(self):
        self.args.stock_uvm = True
        for mode in ('stock', 'arena-reject-parallel'):
            with self.launch('vision arena rejects this configuration'), patch.object(harness.os, 'killpg', create=True):
                with harness.server(self.args, mode, reject=True):
                    pass
        self.assertEqual(self.commands[0][1]['env']['GGML_CUDA_ENABLE_UNIFIED_MEMORY'], '1')
        self.assertNotIn('GGML_CUDA_ENABLE_UNIFIED_MEMORY', self.commands[1][1]['env'])

    def test_mtp_cli_rejection_is_accepted_before_vision_admission(self):
        with self.launch('error: attached MTP KV streaming supports at most 3 draft tokens'), \
                patch.object(harness.os, 'killpg', create=True):
            with harness.server(self.args, 'arena-reject-mtp-length', reject=True):
                pass

    def test_unrelated_vision_oom_is_not_a_configuration_rejection(self):
        with self.launch('CUDA out of memory while loading vision weights'), patch.object(harness.os, 'killpg', create=True):
            with self.assertRaises(AssertionError):
                with harness.server(self.args, 'arena-reject-fit', reject=True):
                    pass

    @unittest.skipUnless(os.name == 'posix', 'process-group cleanup is POSIX-only')
    def test_cleanup_stops_the_whole_instrumented_server_group(self):
        with self.launch('', timeout=True), patch.object(harness.os, 'killpg') as kill, \
                patch.object(harness, 'request', return_value=(200, {})):
            with self.assertRaisesRegex(RuntimeError, 'caller failure'):
                with harness.server(self.args, 'arena'):
                    raise RuntimeError('caller failure')
        self.assertTrue(self.commands[0][1]['start_new_session'])
        self.assertIn((987654, signal.SIGTERM), [call.args for call in kill.call_args_list])
        self.assertIn((987654, signal.SIGKILL), [call.args for call in kill.call_args_list])


if __name__ == '__main__':
    unittest.main()
