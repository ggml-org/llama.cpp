import threading
import time
from dataclasses import dataclass

import pytest
from utils import (
    ServerPreset,
    ServerResponse,
    match_regex,
)


@dataclass
class Prompt:
    prompt: str
    content_regex: str
    prompt_n: int
    tokens_cached: int


PROMPTS = {
    1: Prompt(
        prompt="What is the capital of France?",
        content_regex=r"(allgeme)+",
        prompt_n=8,
        tokens_cached=11,
    ),
    2: Prompt(
        prompt="What is the capital of Mexico?",
        content_regex=r"(osumosum)+",
        prompt_n=8,
        tokens_cached=11,
    ),
    # shares no tokens with prompts 1/2 (which share "What is the capital of
    # ...?"), so any KV reused against this prompt that actually belongs to
    # 1/2 (or vice versa) cannot pass as a coincidental prefix match - it can
    # only be genuinely valid or genuinely wrong content
    3: Prompt(
        prompt="Explain how a rainbow forms.",
        content_regex=r"(truck)+",
        prompt_n=7,
        tokens_cached=10,
    ),
}


class ServerTest:
    def __init__(self, tmp_path_) -> None:
        srv = ServerPreset.tinygemma3()
        srv.slot_save_path = str(tmp_path_)
        srv.temperature = 0.0
        # /slots/save|restore reject any multimodal-capable slot outright, and
        # tinygemma3 auto-loads its mmproj otherwise
        srv.no_mmproj = True
        # "created context checkpoint" is logged at trace level (4); the default
        # verbosity (3, info) suppresses it
        srv.log_verbosity = 4
        # so we can query /slots directly to check a slot's live token count
        srv.server_slots = True

        self.srv = srv
        self.tmp_path = tmp_path_
        self.filenames: set[str] = set()
        self.filename: str | None = None

    def start(self) -> None:
        self.srv.start()

    def op_completion(
        self,
        prompt_id: int,
        id_slot: int,
        prompt_n: None | int = None,
        tokens_cached: None | int = None,
    ) -> int:
        p = PROMPTS[prompt_id]
        res = self.srv.make_request(
            "POST",
            "/completion",
            data={
                "prompt": p.prompt,
                "id_slot": id_slot,
                "cache_prompt": True,
            },
        )
        assert res.status_code == 200
        assert match_regex(p.content_regex, res.body["content"])
        expected_prompt_n = p.prompt_n if prompt_n is None else prompt_n
        assert res.body["timings"]["prompt_n"] == expected_prompt_n

        expected_tokens_cached = (
            p.tokens_cached if tokens_cached is None else tokens_cached
        )
        assert res.body["tokens_cached"] == expected_tokens_cached

        return expected_tokens_cached

    def start_busy_completions(
        self, id_slots: list[int], n_predict: int = 200
    ) -> tuple[list[threading.Thread], dict[int, ServerResponse]]:
        """Start a long-running completion on each of id_slots in the background and
        return once they've had a head start, so those slots are genuinely busy by
        the time the caller does whatever it needs to race against them. Returns the
        threads (join them when done) and a dict of id_slot -> response, populated
        once each thread finishes."""
        results = {}
        threads = []
        for id_slot in id_slots:

            def do_completion(id_slot=id_slot):
                results[id_slot] = self.srv.make_request(
                    "POST",
                    "/completion",
                    data={
                        "prompt": PROMPTS[1].prompt,
                        "id_slot": id_slot,
                        "cache_prompt": True,
                        "n_predict": n_predict,
                    },
                )

            t = threading.Thread(target=do_completion)
            t.start()
            threads.append(t)
        time.sleep(0.2)
        return threads, results

    def op_slot_save(
        self, id_slot: int, n_saved: int, filename: str | None = None
    ) -> bytes:
        if filename is None:
            filename = self.filename
        assert filename is not None

        self.filenames.add(filename)
        res = self.srv.make_request(
            "POST",
            f"/slots/{id_slot}?action=save",
            data={
                "filename": filename,
            },
        )
        assert res.status_code == 200
        assert res.body["n_saved"] == n_saved
        saved_file = self.tmp_path / filename
        return saved_file.read_bytes()

    def op_slot_restore(
        self, id_slot: int, n_restored: int, filename: str | None = None
    ) -> None:
        if filename is None:
            filename = self.filename
        assert filename is not None

        res = self.srv.make_request(
            "POST",
            f"/slots/{id_slot}?action=restore",
            data={
                "filename": filename,
            },
        )
        assert res.status_code == 200
        assert res.body["n_restored"] == n_restored

    def op_slot_erase(self, id_slot: int, n_erased: int) -> None:
        res = self.srv.make_request("POST", f"/slots/{id_slot}?action=erase")
        assert res.status_code == 200
        assert res.body["n_erased"] == n_erased

    def slot_n_prompt_tokens(self, id_slot) -> int:
        res = self.srv.make_request("GET", "/slots")
        assert res.status_code == 200
        slot = next(s for s in res.body if s["id"] == id_slot)
        print(f"\nSLOT: {slot}\n\n")
        return slot.get("n_prompt_tokens")

    def assert_filenames(self) -> None:
        in_tmp = set(f.name for f in self.tmp_path.iterdir())
        assert in_tmp == self.filenames


@pytest.fixture
def server(tmp_path):
    return ServerTest(tmp_path)


def assert_checkpoints_are_possible(capfd):
    out, _ = capfd.readouterr()
    assert "created context checkpoint" in out, (
        "no context checkpoint was created - this test requires a model "
        "that actually produces them (e.g. an SWA model like tinygemma3); "
        "a plain model such as tinyllama2 never creates checkpoints, which "
        "would make the save/restore checks below meaningless:\n" + out
    )


#


def test_slots_get_disabled_without_slots_flag(server):
    # our test harness always passes --slots or --no-slots explicitly, never
    # leaving it unset; ServerTest defaults to --slots, so --no-slots
    # otherwise has zero coverage (the server itself defaults to enabled)
    server.srv.server_slots = False
    server.start()

    res = server.srv.make_request("GET", "/slots")
    assert res.status_code == 501
    assert res.body["error"]["message"] == (
        "This server does not support slots endpoint. Start it with `--slots`"
    )


def test_slots_fail_on_no_slot_ignores_value(server):
    # a "falsy"-looking value like "0" still triggers the rejection, same as
    # "1" from the README example - only presence is checked, not the value
    server.start()

    threads, _ = server.start_busy_completions([0, 1])

    res = server.srv.make_request("GET", "/slots?fail_on_no_slot=0")
    assert res.status_code == 503
    assert res.body["error"]["message"] == "no slot available"

    # with the param entirely absent, no-idle-slots is not an error at all
    res = server.srv.make_request("GET", "/slots")
    assert res.status_code == 200

    for t in threads:
        t.join()


def test_slots_save_restore(server, capfd):
    server.start()
    id_slot = 1
    server.filename = "slot1.bin"

    # prompt_n=8 is the tokenized prompt length; n_saved=11 is the full slot
    # state (8 prompt + 3 generated tokens, generation stops early on EOS)
    tokens_cached = server.op_completion(1, id_slot)
    assert_checkpoints_are_possible(capfd)
    assert server.slot_n_prompt_tokens(id_slot) == tokens_cached
    server.op_slot_save(id_slot, tokens_cached)

    # only different part is processed
    server.op_completion(2, id_slot, 5)

    # n_restored matches n_saved above: the full 11-token slot state
    server.op_slot_restore(0, tokens_cached)

    # /slots/save|restore only round-trip raw tokens and KV state.
    server.op_completion(2, 0)

    # same as above - proves slot 1 wasn't corrupted while slot 0 was restored
    server.op_completion(2, id_slot, 5)

    server.assert_filenames()


def test_slots_restore_over_slot_with_own_checkpoints(server, capfd):
    server.start()
    server.filename = "restore_over_own_checkpoints.bin"

    # slot 0 builds up its own checkpoints from a conversation (rainbow) that
    # shares no tokens at all with the France/Mexico prompts used below - so
    # any KV later reused from these checkpoints cannot coincidentally pass
    # as valid content for a different prompt the way France/Mexico's shared
    # "What is the capital of ...?" prefix could
    server.op_completion(3, 0)
    assert_checkpoints_are_possible(capfd)

    # slot 1 gets a genuinely unrelated conversation, saved to a file
    tokens_cached = server.op_completion(1, 1)
    server.op_slot_save(1, tokens_cached)

    # restore slot 1's file into slot 0, which is neither idle nor erased -
    # it still holds its own rainbow checkpoints from above. /slots/restore
    # only ever replaces slot->prompt.tokens (and the underlying KV
    # sequence); it never touches slot->prompt.checkpoints, so those stale,
    # content-mismatched checkpoints survive the restore
    server.op_slot_restore(0, tokens_cached)

    # continuing with the newly-restored France content must still produce a
    # genuine France completion - if the stale rainbow checkpoints get reused
    # by position instead of being invalidated, the KV state fed to the model
    # would belong to a completely unrelated conversation with no shared
    # tokens to fall back on, and the output must visibly corrupt
    server.op_completion(1, 0)

    server.assert_filenames()


def test_slots_id_wraps_out_of_range(server):
    server.start()
    server.filename = "wrap.bin"

    # n_slots=2 (valid ids: 0, 1); get_slot_by_id does `id_slot % slots.size()`,
    # so an out-of-range id silently wraps instead of erroring - pin this
    # intentional-but-surprising behavior so it can't change unnoticed
    tokens_cached = server.op_completion(1, 0)

    # id_slot=2 wraps to slot 0 (2 % 2 == 0), not slot 1 - slot 1 stays idle
    server.op_slot_save(2, tokens_cached)
    assert server.slot_n_prompt_tokens(1) is None

    server.op_slot_erase(2, tokens_cached)
    assert server.slot_n_prompt_tokens(0) == 0
    assert server.slot_n_prompt_tokens(1) is None

    server.op_slot_restore(2, tokens_cached)
    assert server.slot_n_prompt_tokens(0) == tokens_cached
    assert server.slot_n_prompt_tokens(1) is None

    server.assert_filenames()


def test_slots_reject_bad_request(server):
    server.start()

    # pin fs_validate_filename()'s rejection so path traversal via save or
    # restore can't silently start working again
    for bad_filename in ["../evil.bin", "sub/evil.bin", "..", ""]:
        res = server.srv.make_request(
            "POST", "/slots/0?action=save", data={"filename": bad_filename}
        )
        assert res.status_code == 400
        assert res.body["error"]["message"] == "Invalid filename"

        res = server.srv.make_request(
            "POST", "/slots/0?action=restore", data={"filename": bad_filename}
        )
        assert res.status_code == 400
        assert res.body["error"]["message"] == "Invalid filename"

    # a non-numeric id_slot must get a clean 400, not crash or wrap like an
    # out-of-range numeric one does (see test_slots_id_wraps_out_of_range)
    res = server.srv.make_request(
        "POST", "/slots/abc?action=save", data={"filename": "x.bin"}
    )
    assert res.status_code == 400
    assert res.body["error"]["message"] == "Invalid slot ID"

    # unlike /completion, -1 has no documented meaning for a specific slot
    # action - it's rejected instead of silently wrapping via unsigned
    # overflow, same as any other negative id
    res = server.srv.make_request(
        "POST", "/slots/-1?action=save", data={"filename": "x.bin"}
    )
    assert res.status_code == 400
    assert res.body["error"]["message"] == "Invalid slot ID"

    # an action outside save/restore/erase (or a missing one) falls through
    # to the same clean 400, not a 404 or a silent no-op
    res = server.srv.make_request(
        "POST", "/slots/0?action=frobnicate", data={"filename": "x.bin"}
    )
    assert res.status_code == 400
    assert res.body["error"]["message"] == "Invalid action"

    res = server.srv.make_request("POST", "/slots/0", data={"filename": "x.bin"})
    assert res.status_code == 400
    assert res.body["error"]["message"] == "Invalid action"

    # a missing filename key, or one with the wrong JSON type (e.g. a
    # number), must be rejected the same clean way as any other invalid
    # filename - not treated as an empty/absent key differently, and not
    # left to throw an uncaught nlohmann::json exception
    for bad_data in [{}, {"filename": 1}]:
        res = server.srv.make_request("POST", "/slots/0?action=save", data=bad_data)
        assert res.status_code == 400
        assert res.body["error"]["message"] == "Invalid filename"

        # handle_slots_restore has the same filename handling, but is a
        # separate function - pin it separately in case it ever diverges
        res = server.srv.make_request("POST", "/slots/0?action=restore", data=bad_data)
        assert res.status_code == 400
        assert res.body["error"]["message"] == "Invalid filename"

    server.assert_filenames()


def test_slots_post_disabled_without_slot_save_path(server):
    # ServerTest always sets --slot-save-path, so this rejection otherwise
    # has zero coverage; it fires before id_slot is even parsed, so it
    # applies regardless of slot or action
    server.srv.slot_save_path = None
    server.start()

    res = server.srv.make_request(
        "POST", "/slots/0?action=save", data={"filename": "x.bin"}
    )
    assert res.status_code == 501
    assert res.body["error"]["message"] == (
        "This server does not support slots action. Start it with `--slot-save-path`"
    )


def test_slots_idle_slot_behaviors(server):
    server.start()

    # erasing a slot that was never touched is a no-op, not an error
    server.op_slot_erase(1, 0)

    # restoring a file that was never saved is a clean 400, not a crash
    res = server.srv.make_request(
        "POST",
        "/slots/0?action=restore",
        data={
            "filename": "this_file_was_never_saved.bin",
        },
    )
    assert res.status_code == 400

    server.filename = "idle.bin"
    # saving a slot that was never touched is not an error - it just saves
    # an empty prompt
    server.op_slot_save(0, 0)

    # saving an idle slot must not fabricate any prompt state for it
    assert server.slot_n_prompt_tokens(0) is None

    server.assert_filenames()


def test_slots_restore_corrupted_file(server):
    server.start()
    id_slot = 1
    server.filename = "corrupted.bin"

    tokens_cached = server.op_completion(1, id_slot)
    server.op_slot_save(id_slot, tokens_cached)

    # truncate an otherwise-valid save file down to a few bytes - unlike a
    # nonexistent file, this one exists and opens fine, but
    # llama_state_seq_load_file() must still reject it as unreadable/invalid
    # rather than reading past the end of the buffer or restoring garbage
    saved_file = server.tmp_path / server.filename
    saved_file.write_bytes(saved_file.read_bytes()[:16])

    res = server.srv.make_request(
        "POST", "/slots/0?action=restore", data={"filename": server.filename}
    )
    assert res.status_code == 400
    assert res.body["error"]["message"] == (
        "Unable to restore slot, no available space in KV cache or invalid slot save file"
    )
    assert server.slot_n_prompt_tokens(0) is None

    server.assert_filenames()


def test_slots_save_overwrite(server):
    server.start()
    id_slot = 1
    server.filename = "overwrite.bin"

    tokens_cached = server.op_completion(1, id_slot)
    bytes_after_p1 = server.op_slot_save(id_slot, tokens_cached)

    # saving to the same filename again must replace its contents, not
    # append to it or leave the first save's bytes untouched - n_saved and
    # n_written alone can't prove this (both prompts tokenize to the same
    # length), so compare the actual bytes written to disk
    tokens_cached = server.op_completion(2, id_slot, 5)
    bytes_after_p2 = server.op_slot_save(id_slot, tokens_cached)
    assert bytes_after_p1 != bytes_after_p2

    # slot 1 still holds the Mexico tokens, which share a prefix with this
    # France prompt, so this is a partial cache hit, not a full reprocess
    tokens_cached = server.op_completion(1, id_slot, 5)
    bytes_after_p1_again = server.op_slot_save(id_slot, tokens_cached)
    assert bytes_after_p1 == bytes_after_p1_again

    server.assert_filenames()


def test_slots_restore_twice(server):
    server.start()
    id_slot = 1
    server.filename = "slot_restore_twice.bin"

    tokens_cached = server.op_completion(1, id_slot)
    bytes_after = server.op_slot_save(id_slot, tokens_cached)

    # restoring the same file into the same slot twice in a row must be
    # idempotent - neither restore should fail or leave the slot corrupted
    server.op_slot_restore(id_slot, tokens_cached)
    server.op_slot_restore(id_slot, tokens_cached)

    # saving restored should generate same file
    bytes_after_again = server.op_slot_save(id_slot, tokens_cached)
    assert bytes_after == bytes_after_again

    # the slot must still be genuinely usable after the second restore, but
    # as a full reprocess: /slots/restore now clears prompt.checkpoints
    # unconditionally (see test_slots_restore_over_slot_with_own_checkpoints),
    # so even a self-restore of the slot's own file no longer gets a partial
    # cache hit - there's no cheap way to tell "restoring my own still-valid
    # checkpoints" apart from "restoring over stale, content-mismatched ones"
    server.op_completion(1, id_slot)

    server.assert_filenames()


def test_slots_save_erase_restore(server):
    server.start()
    id_slot = 1
    server.filename = "slot_save_erase_restore.bin"

    assert server.slot_n_prompt_tokens(id_slot) is None

    tokens_cached = server.op_completion(1, id_slot)
    assert server.slot_n_prompt_tokens(id_slot) == tokens_cached

    server.op_slot_save(id_slot, tokens_cached)
    assert server.slot_n_prompt_tokens(id_slot) == tokens_cached

    tokens_cached = server.op_completion(2, id_slot, 5)
    assert server.slot_n_prompt_tokens(id_slot) == tokens_cached

    # erasing the source slot after saving must not corrupt the file already
    # written to disk - n_erased reflects what was in the live slot, not the
    # save file's contents
    server.op_slot_erase(id_slot, tokens_cached)
    assert server.slot_n_prompt_tokens(id_slot) == 0

    server.op_slot_erase(id_slot, 0)
    assert server.slot_n_prompt_tokens(id_slot) == 0

    # save/erase/restore are fully independent of each other: the save file
    # has no lasting link to slot 1, so erasing slot 1 - even twice - has no
    # bearing on whether a restore into slot 1 afterward will succeed
    server.op_slot_restore(id_slot, tokens_cached)
    assert server.slot_n_prompt_tokens(id_slot) == tokens_cached

    # the previously saved file must still restore correctly into a
    # different slot even though its source slot has since been erased
    server.op_slot_restore(0, tokens_cached)
    assert server.slot_n_prompt_tokens(0) == tokens_cached

    # unlike test_slot_save_restore (where the still-alive slot 1 keeps its
    # original checkpoint and gets a discount reprocessing the next prompt),
    # here slot 1 was erased first - that also cleared its checkpoints - so
    # its own restore is now on equal footing with the freshly restored slot
    # 0: both must fully reprocess a new prompt, no partial reuse for either

    # no checkpoint survives erase+restore, full reprocess
    tokens_cached = server.op_completion(2, id_slot)
    assert server.slot_n_prompt_tokens(id_slot) == tokens_cached

    tokens_cached = server.op_completion(2, 0)
    # no checkpoint survives restore, full reprocess
    assert server.slot_n_prompt_tokens(0) == tokens_cached

    server.assert_filenames()


def test_slots_erase_after_restore_only(server):
    server.start()
    id_slot = 1
    server.filename = "restore_only.bin"

    tokens_cached = server.op_completion(1, id_slot)

    server.op_slot_save(id_slot, tokens_cached)

    # slot 0 has never processed a task - only ever been restored into
    server.op_slot_restore(0, tokens_cached)
    assert server.slot_n_prompt_tokens(0) == tokens_cached
    server.op_slot_erase(0, tokens_cached)

    # unlike a slot that has processed a real task - which keeps reporting
    # n_prompt_tokens: 0 explicitly after being erased, see
    # test_slot_save_erase_restore - a restore-only slot has no task/task_prev
    # to keep it "known" to /slots once its tokens are cleared: it reverts to
    # reporting nothing at all, indistinguishable from a slot that was never
    # touched. This is accepted as a cosmetic quirk rather than a bug: both
    # states mean the same thing functionally (0 usable tokens, ready for
    # reuse), and fixing it would need a new persistent flag on server_slot
    # purely to serve this one introspection difference. Pinned here so any
    # future change to this behavior is a deliberate, visible decision.
    assert server.slot_n_prompt_tokens(0) is None

    server.assert_filenames()


def test_slots_erase(server, capfd):
    server.start()
    id_slot = 1

    tokens_cached = server.op_completion(1, id_slot)
    assert_checkpoints_are_possible(capfd)
    server.op_slot_erase(id_slot, tokens_cached)

    # re-run the same prompt, it should process all tokens again
    server.op_completion(1, id_slot)


def test_slots_reject_multimodal_slot(server):
    # tinygemma3 auto-loads its mmproj unless no_mmproj is set (see ServerTest
    # above) - flip it back on here to exercise the multimodal rejection
    server.srv.no_mmproj = False
    server.start()

    res = server.srv.make_request(
        "POST", "/slots/0?action=save", data={"filename": "mm.bin"}
    )
    assert res.status_code == 501
    assert "multimodal" in res.body["error"]["message"]

    res = server.srv.make_request(
        "POST", "/slots/0?action=restore", data={"filename": "mm.bin"}
    )
    assert res.status_code == 501
    assert "multimodal" in res.body["error"]["message"]

    res = server.srv.make_request("POST", "/slots/0?action=erase")
    assert res.status_code == 501
    assert "multimodal" in res.body["error"]["message"]


def test_slots_defer_while_slot_processing(server, capfd):
    # exercise the defer-while-busy path directly by racing a save against a
    # long-running completion on the same slot. The defer log line is
    # debug-level.
    server.srv.log_verbosity = 5
    server.start()
    id_slot = 1
    server.filename = "defer.bin"

    hit = False
    save_res = None
    comp_body = None
    for _attempt in range(6):
        threads, results = server.start_busy_completions([id_slot])
        save_res = server.srv.make_request(
            "POST", f"/slots/{id_slot}?action=save", data={"filename": server.filename}
        )
        for t in threads:
            t.join()
        comp_body = results[id_slot].body

        out, _ = capfd.readouterr()
        if "requested slot is unavailable, defer task" in out:
            hit = True
            break

    assert hit, "never observed the slot-busy defer path across 6 attempts"
    assert save_res.status_code == 200
    # the deferred save only ever runs after the completion releases the
    # slot, so it must reflect the full, final post-generation token count,
    # not some stale count from before generation started
    assert save_res.body["n_saved"] == comp_body["tokens_cached"]

    server.filenames.add(server.filename)
    server.assert_filenames()
