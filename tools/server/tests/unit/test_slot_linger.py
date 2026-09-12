import os
import re
import tempfile
import time

import pytest
from utils import *

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.n_slots = 1
    server.n_ctx = 1024
    server.n_predict = 4
    server.temperature = 0.0
    server.cache_ram = 1  # MiB: default, overridden per test
    server.kv_unified = True
    server.debug = True
    fd, server.log_path = tempfile.mkstemp(suffix=".log")
    os.close(fd)
    yield


def test_linger_disabled_with_empty_deferred_queue():
    """No linger when deferred queue is empty: throughput unaffected."""
    server.slot_linger_ms = 2000  # Large value to make any linger visible
    server.n_predict = 8
    server.start()
    log = LogReader(server.log_path)
    log.drain()  # Clear initial logs

    prompt_a = make_prompt(1, "chapterA")

    # Send requests one at a time (no deferred queue)
    start = time.monotonic()
    for _ in range(3):
        res = server.make_request(
            "POST",
            "/completion",
            data={
                "prompt": prompt_a,
                "cache_prompt": True,
            },
        )
        assert res.status_code == 200
    elapsed = time.monotonic() - start

    drained = log.drain()
    # Neither linger marker should appear since deferred queue was always empty
    assert "linger cancelled (hit)" not in drained
    assert "linger expired, falling back to FIFO" not in drained

    # Throughput check: with no deferred queue, linger should not trigger
    # Even with a 2000ms linger setting, total time should be much less than 6s (3 * 2000ms)
    assert elapsed < 3.0, f"Expected fast sequential processing, got {elapsed:.2f}s"


def test_linger_hit_serves_in_window_arrival_first():
    """In-window arrival is served ahead of older queued request."""
    server.slot_linger_ms = 2000  # Long window to ensure test isn't racing
    server.n_predict = 12  # Long enough to queue multiple requests
    server.start()
    log = LogReader(server.log_path)
    log.drain()

    prompt_a = make_prompt(1, "chapterA")
    prompt_b = make_prompt(2, "chapterB")
    prompt_c = make_prompt(3, "chapterC")

    import threading

    blocking_result = None
    old_result = None
    new_result = None
    old_ts = None
    new_ts = None
    blocking_done = threading.Event()

    def blocking_request():
        nonlocal blocking_result
        blocking_result = server.make_request(
            "POST", "/completion", data={"prompt": prompt_a, "cache_prompt": True}
        )
        blocking_done.set()

    def old_request():
        nonlocal old_result, old_ts
        old_result = server.make_request(
            "POST", "/completion", data={"prompt": prompt_b, "cache_prompt": True}
        )
        old_ts = time.monotonic()

    def new_request():
        nonlocal new_result, new_ts
        # Wait for blocking to complete, then send during linger window
        blocking_done.wait()
        new_result = server.make_request(
            "POST", "/completion", data={"prompt": prompt_c, "cache_prompt": True}
        )
        new_ts = time.monotonic()

    # Occupy the slot with a blocking request
    blocking_thread = threading.Thread(target=blocking_request)
    blocking_thread.start()
    time.sleep(0.05)  # Let it start

    # Queue an old request while slot is busy
    old_thread = threading.Thread(target=old_request)
    old_thread.start()
    time.sleep(0.01)  # Ensure it's queued

    # Start new request thread (it will wait for blocking to complete)
    new_thread = threading.Thread(target=new_request)
    new_thread.start()

    # Wait for all to complete
    blocking_thread.join()
    old_thread.join()
    new_thread.join()

    assert blocking_result.status_code == 200
    assert old_result.status_code == 200
    assert new_result.status_code == 200

    drained = log.drain()
    assert "linger cancelled (hit)" in drained, (
        "Expected linger hit marker when new request arrives in window"
    )

    # The new request (arriving in-window) should complete before the old one
    margin = old_ts - new_ts
    print(
        f"\nCompletion timestamps: new={new_ts:.6f}s, old={old_ts:.6f}s, margin={margin:.6f}s"
    )
    assert new_ts < old_ts, (
        f"Expected in-window arrival to complete first, but new={new_ts:.6f}s old={old_ts:.6f}s"
    )


def test_linger_expired_uses_fifo():
    """Post-expiry arrival is NOT preferred: FIFO is honored."""
    server.slot_linger_ms = 100  # Short window that will expire
    server.n_predict = 12
    server.start()
    log = LogReader(server.log_path)
    log.drain()

    prompt_a = make_prompt(1, "chapterA")
    prompt_b = make_prompt(2, "chapterB")
    prompt_c = make_prompt(3, "chapterC")

    import threading

    blocking_result = None
    old_result = None
    late_result = None
    old_ts = None
    late_ts = None
    blocking_done = threading.Event()

    def blocking_request():
        nonlocal blocking_result
        blocking_result = server.make_request(
            "POST", "/completion", data={"prompt": prompt_a, "cache_prompt": True}
        )
        blocking_done.set()

    def old_request():
        nonlocal old_result, old_ts
        old_result = server.make_request(
            "POST", "/completion", data={"prompt": prompt_b, "cache_prompt": True}
        )
        old_ts = time.monotonic()

    def late_request():
        nonlocal late_result, late_ts
        # Wait for blocking to complete, then wait for linger to expire
        blocking_done.wait()
        time.sleep(0.15)  # Wait longer than linger window (100ms)
        late_result = server.make_request(
            "POST", "/completion", data={"prompt": prompt_c, "cache_prompt": True}
        )
        late_ts = time.monotonic()

    # Occupy the slot
    blocking_thread = threading.Thread(target=blocking_request)
    blocking_thread.start()
    time.sleep(0.05)

    # Queue an old request while slot is busy
    old_thread = threading.Thread(target=old_request)
    old_thread.start()
    time.sleep(0.01)

    # Start late request thread (it will wait for linger to expire)
    late_thread = threading.Thread(target=late_request)
    late_thread.start()

    blocking_thread.join()
    old_thread.join()
    late_thread.join()

    assert blocking_result.status_code == 200
    assert old_result.status_code == 200
    assert late_result.status_code == 200

    drained = log.drain()
    assert "linger expired, falling back to FIFO" in drained, (
        "Expected linger expired marker when window times out"
    )

    # FIFO: old request must complete before late request
    assert old_ts < late_ts, (
        f"Expected FIFO order (old first), but old={old_ts:.6f}s late={late_ts:.6f}s"
    )


def test_linger_does_not_drop_concurrent_slot_releases():
    """Two slots have independent linger windows.

    If four tasks are queued with a 1.5s linger window, total time will
    be greater than 1.5s and much less than 3s.
    """
    server.n_slots = 2
    server.n_ctx = 2048  # 1024 per slot; prompts are ~840 tokens
    server.slot_linger_ms = 1500
    server.n_predict = 8
    server.start()
    log = LogReader(server.log_path)
    log.drain()

    prompts = [make_prompt(i, f"chapter{i}") for i in range(4)]

    # slots 0 and 1 take the first two, the other two are deferred. Both slots then release
    # inside the same linger window
    start = time.monotonic()
    results = parallel_function_calls(
        [
            (
                server.make_request,
                ("POST", "/completion", {"prompt": p, "cache_prompt": True}),
            )
            for p in prompts
        ]
    )
    elapsed = time.monotonic() - start

    assert all(r.status_code == 200 for r in results)

    drained = log.drain()
    expired_slots = set(re.findall(r"slot (\d+) linger expired", drained))
    assert len(expired_slots) == 2, (
        f"Expected both slots to expire their own linger, got {expired_slots}"
    )

    # one 1500 ms window, not two
    assert elapsed < 2.6, (
        f"Expected the deferred backlog to drain in a single linger window, took {elapsed:.2f}s"
    )


def test_no_linger_when_a_deferred_task_is_bound_to_the_slot():
    """A deferred task already pinned to the releasing slot must not wait out the window.

    Lingering exists to catch a new arrival that could reuse the slot cache. A task bound to
    this slot by id_slot is already the best use of it, so holding the slot only adds latency.
    """
    server.slot_linger_ms = 3000
    server.n_predict = 16
    server.start()
    log = LogReader(server.log_path)
    log.drain()

    prompt_a = make_prompt(1, "chapterA")
    prompt_b = make_prompt(2, "chapterB")

    import threading

    results = {}

    def occupier():
        results["a"] = server.make_request(
            "POST", "/completion", data={"prompt": prompt_a, "cache_prompt": True}
        )

    start = time.monotonic()
    occupier_thread = threading.Thread(target=occupier)
    occupier_thread.start()
    time.sleep(0.1)  # wait for slot 0 to be occupied

    # arrives while slot 0 is busy, so it is deferred while pinned to slot 0
    results["b"] = server.make_request(
        "POST",
        "/completion",
        data={"prompt": prompt_b, "id_slot": 0, "cache_prompt": True},
    )
    occupier_thread.join()
    elapsed = time.monotonic() - start

    assert results["a"].status_code == 200
    assert results["b"].status_code == 200
    assert results["b"].body["id_slot"] == 0

    drained = log.drain()
    assert "linger expired, falling back to FIFO" not in drained, (
        "Expected no linger window for a slot that already has a task bound to it"
    )
    assert elapsed < 2.0, (
        f"Expected the bound task to run without waiting out the 3000 ms window, took {elapsed:.2f}s"
    )
