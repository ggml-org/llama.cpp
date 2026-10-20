"""Host-only contracts for the offline Gemma server qualification command."""

import argparse
import copy
import contextlib
import io
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
            "--output","output","--depths","0,1,2,3","--cache-type-k","q5_1","--cache-type-v","q8_0","--require-cache-hit"]
        with patch.object(harness.sys,"argv",argv),patch.object(harness,"run") as run:
            harness.main()
        args=run.call_args.args[0]
        self.assertEqual(args.depths,[0,1,2,3])
        self.assertEqual((args.cache_type_k,args.cache_type_v),("q5_1","q8_0"))
        self.assertTrue(args.require_cache_hit)

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
    def execute(self,depth,results,require_cache_hit=False):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            corpus=directory/"corpus.txt"
            corpus.write_text("Trees grow near streams.",encoding="utf-8")
            args=argparse.Namespace(output=directory,corpus=corpus,depths=[depth],prefill=16,predict=2,require_cache_hit=require_cache_hit)
            with patch.object(harness,"server") as server, \
                    patch.object(harness,"request_json",return_value={"tokens":list(range(256))}), \
                    patch.object(harness,"stream_completion",side_effect=results), \
                    patch.object(harness,"cancel_after_first_output"):
                server.return_value.__enter__.return_value="http://unused"
                return harness.run(args)

    def completions(self,drafts):
        result={"tokens":[1,2],"timings":{"predicted_n":2,"draft_n":drafts,"draft_n_accepted":0}}
        rows=[copy.deepcopy(result) for _ in range(8)]
        for index,row in enumerate(rows):
            cache,prompt=((0,16),(15,1),(0,64),(0,64))[index%4]
            row["timings"].update(cache_n=cache,prompt_n=prompt)
        return rows

    def test_prompt_accounting_checks_real_reuse_and_reprocessed_suffix(self):
        for cache,prompt in ((15,1),(12,4)):
            self.assertEqual(harness.prompt_counts({"timings":{"cache_n":cache,"prompt_n":prompt}},16,True,"cached"),(cache,prompt))
        self.assertEqual(harness.prompt_counts({"timings":{"cache_n":0,"prompt_n":16}},16,False,"fresh"),(0,16))

    def test_prompt_accounting_rejects_missing_invalid_or_unexpected_cache_counts(self):
        for timing in ({},{"cache_n":True,"prompt_n":15},{"cache_n":0,"prompt_n":16.0},
                {"cache_n":-1,"prompt_n":17},{"cache_n":17,"prompt_n":-1},{"cache_n":0,"prompt_n":15},
                {"cache_n":1,"prompt_n":15}):
            with self.assertRaisesRegex(RuntimeError,"prompt accounting"):
                harness.prompt_counts({"timings":timing},16,False,"fresh")

    def test_repeat_requests_must_actually_reuse_cache_when_required(self):
        results=self.completions(2)
        results[1]["timings"].update(cache_n=0,prompt_n=16)
        with self.assertRaisesRegex(RuntimeError,"cache hit"):
            self.execute(3,results,True)

    def test_actual_cache_hit_accounting_can_pass_all_request_cases(self):
        self.assertEqual(len(self.execute(3,self.completions(2),True)),8)

    def test_same_tokens_do_not_hide_different_checkpoint_replay_boundaries(self):
        results=self.completions(2)
        results[5]["timings"].update(cache_n=14,prompt_n=2)
        with self.assertRaisesRegex(RuntimeError,"prompt accounting mismatch"):
            self.execute(3,results,True)

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


class Lifecycles(unittest.TestCase):
    def test_cli_selects_explicit_stream_only_lifecycle(self):
        argv=["qualify","--server","server","--model","model","--assistant","assistant",
            "--output","output","--lifecycle","--streamed-only"]
        with patch.object(harness.sys,"argv",argv),patch.object(harness,"run_lifecycle") as run:
            harness.main()
        self.assertTrue(run.call_args.args[0].streamed_only)
        self.assertTrue(run.call_args.args[0].lifecycle)

    def test_stream_only_cannot_disable_the_ordinary_native_oracle(self):
        argv=["qualify","--server","server","--model","model","--assistant","assistant",
            "--output","output","--streamed-only"]
        with patch.object(harness.sys,"argv",argv),patch.object(harness,"run") as run, \
                contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            harness.main()
        run.assert_not_called()

    def execute(self,streamed_only=False,damage_restore=False,damage_restart=False,native_cache_difference=False,damage_streamed_cache=False,native_restore_difference=False,damage_streamed_restore=False):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)
            corpus=directory/"corpus.txt"
            corpus.write_text("Trees grow near streams.",encoding="utf-8")
            args=argparse.Namespace(output=directory,corpus=corpus,depths=[3],prefill=128,predict=4,
                context=512,lifecycle=True,streamed_only=streamed_only)
            launches=[]
            actions=[]
            restored=set()
            def server(args,streamed,depth,phase=None):
                launches.append((streamed,phase))
                return contextlib.nullcontext(f"http://{'streamed' if streamed else 'native'}-{phase or 'initial'}")
            def request(url,payload=None,**kwargs):
                if url.endswith("/tokenize"):
                    return {"tokens":list(range(256))}
                action=url.split("action=")[-1]
                actions.append((url,action))
                if action=="erase":
                    return {"n_erased":67}
                streamed="streamed" in url
                save=directory/f"{'streamed' if streamed else 'native'}-mtp3-slots"/"lifecycle.bin"
                save.parent.mkdir(parents=True,exist_ok=True)
                if action=="save":
                    save.write_bytes(b"test")
                    return {"n_saved":131,"n_written":4}
                restored.add(url.split("/slots/")[0])
                return {"n_restored":131,"n_read":3 if damage_restore else 4}
            def completion(url,body,**kwargs):
                length=len(body["prompt"])
                reused=length-1 if body["cache_prompt"] else 0
                tokens=[body["prompt"][0]+i for i in range(body["n_predict"])]
                after_restore=url.split("/completion")[0] in restored
                restored.discard(url.split("/completion")[0])
                if native_cache_difference and body["cache_prompt"]:
                    tokens[0]+=1000
                if damage_streamed_cache and "streamed" in url and body["cache_prompt"]:
                    tokens[0]+=1
                if native_restore_difference and after_restore:
                    reused-=1
                    tokens[0]+=2000
                if damage_streamed_restore and after_restore and "streamed" in url:
                    tokens[0]+=1
                if damage_restart and "restart" in url:
                    tokens[0]+=1
                return {"tokens":tokens,"timings":{"predicted_n":body["n_predict"],"cache_n":reused,
                    "prompt_n":length-reused,"draft_n":2,"draft_n_accepted":1}}
            with patch.object(harness,"server",side_effect=server),patch.object(harness,"request_json",side_effect=request), \
                    patch.object(harness,"stream_completion",side_effect=completion),patch.object(harness,"cancel_after_first_output"), \
                    contextlib.redirect_stdout(io.StringIO()):
                rows=harness.run_lifecycle(args)
                return rows,launches,actions

    def test_lifecycle_covers_low_high_low_save_restore_and_restart(self):
        rows,launches,actions=self.execute()
        self.assertEqual(len(rows),20)
        self.assertEqual(len(launches),4)
        self.assertEqual(sum(action=="save" for _,action in actions),2)
        self.assertEqual(sum(action=="restore" for _,action in actions),4)
        self.assertEqual(sum(action=="erase" for _,action in actions),2)
        self.assertEqual({row["case"] for row in rows},{"low-fresh","low-cached","high-fresh","high-cached",
            "low-after-high","restored-high","cancel-recovery","restarted-restored","restarted-fresh","restarted-low"})

    def test_stream_only_lifecycle_is_not_labelled_native_equivalence(self):
        rows,launches,_=self.execute(streamed_only=True)
        self.assertEqual(len(rows),10)
        self.assertTrue(all(streamed for streamed,_ in launches))
        self.assertTrue(all(row["comparison"]=="streamed-self-reference" for row in rows))

    def test_state_restore_must_consume_the_saved_file(self):
        with self.assertRaisesRegex(RuntimeError,"state restore"):
            self.execute(damage_restore=True)

    def test_restart_must_not_change_continuation_tokens(self):
        with self.assertRaisesRegex(RuntimeError,"token mismatch"):
            self.execute(damage_restart=True)

    def test_cache_replay_matches_its_native_case_not_a_different_prefill_width(self):
        rows,_,_=self.execute(native_cache_difference=True)
        self.assertEqual(len(rows),20)

    def test_native_streamed_cached_case_still_requires_exact_tokens(self):
        with self.assertRaisesRegex(RuntimeError,"native/streamed lifecycle token mismatch"):
            self.execute(damage_streamed_cache=True)

    def test_restored_native_state_is_the_oracle_for_its_own_replay_path(self):
        rows,_,_=self.execute(native_restore_difference=True)
        self.assertEqual(len(rows),20)

    def test_streamed_state_restore_must_match_its_native_counterpart(self):
        with self.assertRaisesRegex(RuntimeError,"native/streamed lifecycle token mismatch"):
            self.execute(damage_streamed_restore=True)

    def test_lifecycle_slot_paths_are_separate_for_native_and_streamed(self):
        args=argparse.Namespace(server=Path("server"),model=Path("model"),assistant=Path("assistant"),
            output=Path("output"),context=4096,batch_size=256,ubatch_size=256,arena_mib=1024,lifecycle=True)
        paths=[]
        for streamed in (False,True):
            command=harness.server_command(args,streamed,3,9999)
            self.assertIn("--slot-save-path",command)
            paths.append(command[command.index("--slot-save-path")+1])
        self.assertNotEqual(*paths)


if __name__ == "__main__":
    unittest.main()
