"""Host-only contracts for the offline Gemma server qualification command."""

import argparse
import copy
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import test_adaptive_gemma as harness


class Commands(unittest.TestCase):
    def test_all_cache_pairs_match_native_controls_and_streamed_inheritance(self):
        formats=("f32","f16","bf16","q8_0","q5_0","q5_1","q4_0","q4_1")
        for key in formats:
            for value in formats:
                args=argparse.Namespace(server=Path("server"),model=Path("model"),assistant=Path("assistant"),
                    context=4096,batch_size=256,ubatch_size=256,arena_mib=1024,cache_type_k=key,cache_type_v=value)
                for depth in (1,2,3):
                    for streamed in (False,True):
                        command=harness.server_command(args,streamed,depth,9999)
                        self.assertEqual(command[command.index("--cache-type-k")+1],key)
                        self.assertEqual(command[command.index("--cache-type-v")+1],value)
                        if streamed:
                            self.assertNotIn("--spec-draft-type-k",command)
                            self.assertNotIn("--spec-draft-type-v",command)
                        else:
                            self.assertEqual(command[command.index("--spec-draft-type-k")+1],key)
                            self.assertEqual(command[command.index("--spec-draft-type-v")+1],value)

    def test_target_only_does_not_load_an_assistant_or_enable_speculation(self):
        args=argparse.Namespace(server=Path("server"),model=Path("model"),assistant=Path("assistant"),
            context=4096,batch_size=256,ubatch_size=256,arena_mib=1024,cache_type_k="q5_1",cache_type_v="q5_1")
        for streamed in (False,True):
            command=harness.server_command(args,streamed,0,9999)
            for flag in ("--model-draft","--spec-type","--spec-draft-n-max","--spec-draft-n-min",
                    "--spec-draft-p-min","--spec-draft-ngl","--spec-draft-type-k","--spec-draft-type-v"):
                self.assertNotIn(flag,command)
            self.assertEqual(command.count("--kv-stream-arena-mib"),int(streamed))

    def test_cli_accepts_target_and_all_assistant_depths_with_explicit_cache_types(self):
        argv=["qualify","--server","server","--model","model","--assistant","assistant",
            "--output","output","--depths","0,1,2,3","--cache-type-k","q5_1","--cache-type-v","q8_0"]
        with patch.object(harness.sys,"argv",argv),patch.object(harness,"run") as run:
            harness.main()
        args=run.call_args.args[0]
        self.assertEqual(args.depths,[0,1,2,3])
        self.assertEqual((args.cache_type_k,args.cache_type_v),("q5_1","q8_0"))

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


class Results(unittest.TestCase):
    def execute(self,depth,results):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            corpus=directory/"corpus.txt"
            corpus.write_text("Trees grow near streams.",encoding="utf-8")
            args=argparse.Namespace(output=directory,corpus=corpus,depths=[depth],prefill=16,predict=2)
            with patch.object(harness,"server") as server, \
                    patch.object(harness,"request_json",return_value={"tokens":list(range(256))}), \
                    patch.object(harness,"stream_completion",side_effect=results), \
                    patch.object(harness,"cancel_after_first_output"):
                server.return_value.__enter__.return_value="http://unused"
                return harness.run(args)

    def completions(self,drafts):
        result={"tokens":[1,2],"timings":{"predicted_n":2,"draft_n":drafts,"draft_n_accepted":0}}
        return [copy.deepcopy(result) for _ in range(8)]

    def test_target_only_accepts_zero_drafts(self):
        self.assertEqual(len(self.execute(0,self.completions(0))),8)

    def test_target_only_rejects_unexpected_speculation(self):
        with self.assertRaisesRegex(RuntimeError,"target-only"):
            self.execute(0,self.completions(1))

    def test_identical_tokens_do_not_hide_different_draft_or_acceptance_counts(self):
        for key in ("draft_n","draft_n_accepted"):
            results=self.completions(2)
            results[4]["timings"][key]+=1
            with self.assertRaisesRegex(RuntimeError,"draft/acceptance"):
                self.execute(3,results)

    def test_invalid_draft_counters_do_not_pass_qualification(self):
        for generated,accepted in ((-1,0),(1,-1),(1,2),(True,0),(2,0.5)):
            with self.assertRaisesRegex(RuntimeError,"invalid Gemma draft/acceptance"):
                harness.draft_counts({"timings":{"draft_n":generated,"draft_n_accepted":accepted}},3,"fresh")


if __name__ == "__main__":
    unittest.main()
