#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

import mlx.core as mx
import numpy as np


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from tools.tessera.kv_compaction_mlx import (
    KVCompactionMachine,
    KVCompactionState,
    apply_plan,
    defragment,
    divide_positions,
    make_plan,
    remove,
    rope_shift,
    shift,
)


def state_with_positions(count=8):
    keys = mx.arange(count * 2 * 8, dtype=mx.float32).reshape(count, 2, 8) / 31
    values = mx.arange(count * 2 * 8, dtype=mx.float32).reshape(count, 2, 8) / 17
    return KVCompactionState(
        keys,
        values,
        np.arange(count, dtype=np.int32),
        np.zeros(count, dtype=np.int32),
        np.ones(count, dtype=bool),
        np.zeros(count, dtype=np.int32),
    )


class KVCompactionStateTests(unittest.TestCase):
    def test_remove_shift_and_divide_match_llama_ranges(self):
        state = state_with_positions(6)
        state = remove(state, 0, 1, 2)
        np.testing.assert_array_equal(state.valid, [True, False, True, True, True, True])
        state = shift(state, 0, 3, None, -2)
        np.testing.assert_array_equal(state.positions, [0, 1, 2, 1, 2, 3])
        np.testing.assert_array_equal(state.pending_shifts, [0, 0, 0, -2, -2, -2])
        state = divide_positions(state, 0, 1, 3, 2)
        np.testing.assert_array_equal(state.positions, [0, 1, 1, 0, 1, 3])
        np.testing.assert_array_equal(state.pending_shifts, [0, 0, -1, -3, -3, -2])

    def test_defragment_is_lossless_and_stable(self):
        state = remove(state_with_positions(6), 0, 1, 3)
        compacted = defragment(state)
        np.testing.assert_array_equal(compacted.positions, [0, 3, 4, 5, 1, 2])
        np.testing.assert_array_equal(compacted.valid, [True, True, True, True, False, False])
        expected = np.array(state.keys)[[0, 3, 4, 5]]
        np.testing.assert_array_equal(np.array(compacted.keys)[:4], expected)

    def test_merge_plan_aligns_positions_and_is_differentiable(self):
        state = state_with_positions(8)
        plan = make_plan(
            state,
            target_per_sequence=4,
            keep_prefix=1,
            keep_recent=1,
            strategy="merge",
        )
        np.testing.assert_array_equal(
            plan.destination_positions,
            [0, 3, 6, 7, -1, -1, -1, -1],
        )
        np.testing.assert_array_equal(
            plan.rope_deltas,
            [0, 2, 1, 0, 2, 1, 0, 0],
        )
        compacted = apply_plan(
            state,
            plan,
            n_rot=8,
            frequency_base=10000,
        )
        self.assertEqual(int(np.count_nonzero(compacted.valid)), 4)
        gradient = mx.grad(
            lambda keys: mx.sum(
                apply_plan(
                    KVCompactionState(
                        keys,
                        state.values,
                        state.positions,
                        state.sequence_ids,
                        state.valid,
                        state.pending_shifts,
                    ),
                    plan,
                    n_rot=8,
                    frequency_base=10000,
                ).keys
            )
        )(state.keys)
        self.assertTrue(np.all(np.isfinite(np.array(gradient))))

    def test_eviction_preserves_prefix_and_recent(self):
        state = state_with_positions(10)
        plan = make_plan(state, 5, 2, 2, "evict")
        selected = np.flatnonzero(plan.source_to_destination >= 0)
        np.testing.assert_array_equal(selected, [0, 1, 7, 8, 9])

    def test_rope_shift_round_trip(self):
        keys = state_with_positions(4).keys
        delta = np.array([3, -2, 7, 1], dtype=np.int32)
        shifted = rope_shift(keys, delta, 8, 10000, rope_type="neox")
        restored = rope_shift(shifted, -delta, 8, 10000, rope_type="neox")
        np.testing.assert_allclose(np.array(restored), np.array(keys), atol=2e-6)

    def test_state_machine_tracks_bounded_privacy_safe_transitions(self):
        machine = KVCompactionMachine(
            state_with_positions(8),
            n_rot=8,
            frequency_base=10000,
            history_limit=2,
        )
        machine.seq_rm(0, 1, 2)
        machine.seq_add(0, 4, None, -1)
        compacted, plan = machine.compact(4, 1, 1, "merge")
        self.assertEqual(machine.epoch, 3)
        self.assertEqual(len(machine.history), 2)
        self.assertEqual(machine.history[-1].operation, "compact")
        self.assertEqual(machine.history[-1].valid_after, 4)
        self.assertNotIn("keys", machine.history[-1].details)
        self.assertEqual(int(np.count_nonzero(compacted.valid)), 4)
        self.assertEqual(plan.strategy, "merge")


if __name__ == "__main__":
    unittest.main()
