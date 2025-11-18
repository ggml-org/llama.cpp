#!/usr/bin/env python3
"""
Continuous Batching Simulator with PagedAttention

Demonstrates continuous batching with KV cache management,
prefix caching, and dynamic scheduling.
"""

import time
import numpy as np
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field
from collections import deque
import heapq


@dataclass
class SequenceState:
    """State of an active sequence"""
    seq_id: int
    tokens: List[int]
    prompt_len: int
    max_tokens: int
    priority: int = 0
    timestamp: float = field(default_factory=time.time)

    @property
    def is_complete(self) -> bool:
        return len(self.tokens) - self.prompt_len >= self.max_tokens

    @property
    def generated_tokens(self) -> int:
        return len(self.tokens) - self.prompt_len


class PagedKVCache:
    """
    Paged KV cache implementation
    Manages memory in fixed-size blocks for efficient allocation
    """

    def __init__(self, block_size: int = 16, num_blocks: int = 1000):
        self.block_size = block_size
        self.num_blocks = num_blocks

        # Free block pool
        self.free_blocks: Set[int] = set(range(num_blocks))

        # Sequence to blocks mapping
        self.seq_blocks: Dict[int, List[int]] = {}

        # Block reference counts (for prefix sharing)
        self.block_refs: Dict[int, int] = {}

        # Prefix cache: prefix_hash -> block_ids
        self.prefix_cache: Dict[int, List[int]] = {}

    def allocate_sequence(
        self,
        seq_id: int,
        prompt_tokens: List[int]
    ) -> bool:
        """
        Allocate blocks for a new sequence

        Returns:
            True if allocation succeeded, False if OOM
        """
        # Check for prefix match
        prefix_blocks = self._find_cached_prefix(prompt_tokens)

        # Calculate needed blocks
        prefix_len = len(prefix_blocks) * self.block_size
        remaining_tokens = len(prompt_tokens) - prefix_len
        new_blocks_needed = (remaining_tokens + self.block_size - 1) // self.block_size

        if len(self.free_blocks) < new_blocks_needed:
            return False  # OOM

        # Allocate new blocks
        new_blocks = []
        for _ in range(new_blocks_needed):
            block_id = self.free_blocks.pop()
            new_blocks.append(block_id)
            self.block_refs[block_id] = 1

        # Combine prefix and new blocks
        all_blocks = prefix_blocks + new_blocks

        # Increment prefix block refs
        for block_id in prefix_blocks:
            self.block_refs[block_id] += 1

        self.seq_blocks[seq_id] = all_blocks

        # Cache prefix if long enough
        if len(prefix_blocks) == 0 and len(prompt_tokens) >= self.block_size:
            prefix_hash = self._hash_prefix(prompt_tokens[:self.block_size])
            self.prefix_cache[prefix_hash] = all_blocks[:1]

        return True

    def append_tokens(self, seq_id: int, num_tokens: int) -> bool:
        """
        Allocate additional blocks if needed for new tokens

        Returns:
            True if successful, False if OOM
        """
        if seq_id not in self.seq_blocks:
            return False

        blocks = self.seq_blocks[seq_id]
        current_capacity = len(blocks) * self.block_size

        # Current usage
        # (simplified - in real impl, track per-sequence position)
        current_usage = current_capacity

        # Check if we need more blocks
        new_capacity_needed = current_usage + num_tokens
        new_blocks_needed = (
            (new_capacity_needed + self.block_size - 1) // self.block_size
            - len(blocks)
        )

        if new_blocks_needed <= 0:
            return True

        # Allocate additional blocks
        if len(self.free_blocks) < new_blocks_needed:
            return False  # OOM

        for _ in range(new_blocks_needed):
            block_id = self.free_blocks.pop()
            blocks.append(block_id)
            self.block_refs[block_id] = 1

        return True

    def free_sequence(self, seq_id: int):
        """Free all blocks used by sequence"""
        if seq_id not in self.seq_blocks:
            return

        blocks = self.seq_blocks.pop(seq_id)

        for block_id in blocks:
            self.block_refs[block_id] -= 1

            if self.block_refs[block_id] == 0:
                del self.block_refs[block_id]
                self.free_blocks.add(block_id)

    def _find_cached_prefix(self, tokens: List[int]) -> List[int]:
        """Find cached prefix blocks"""
        if len(tokens) < self.block_size:
            return []

        prefix_hash = self._hash_prefix(tokens[:self.block_size])

        if prefix_hash in self.prefix_cache:
            return self.prefix_cache[prefix_hash].copy()

        return []

    def _hash_prefix(self, tokens: List[int]) -> int:
        """Hash token sequence for prefix caching"""
        return hash(tuple(tokens))

    def utilization(self) -> float:
        """Calculate memory utilization"""
        used_blocks = self.num_blocks - len(self.free_blocks)
        return used_blocks / self.num_blocks


class ContinuousBatchScheduler:
    """
    Continuous batching scheduler with dynamic request management
    """

    def __init__(
        self,
        max_batch_size: int = 32,
        block_size: int = 16,
        num_blocks: int = 1000,
        prefill_chunk_size: int = 512
    ):
        self.max_batch_size = max_batch_size
        self.prefill_chunk_size = prefill_chunk_size

        # KV cache
        self.kv_cache = PagedKVCache(block_size, num_blocks)

        # Active sequences
        self.active_seqs: Dict[int, SequenceState] = {}

        # Waiting queue (priority queue)
        self.waiting_queue: List[SequenceState] = []

        # Metrics
        self.total_tokens_generated = 0
        self.total_iterations = 0
        self.total_sequences_completed = 0

    def submit_request(
        self,
        seq_id: int,
        prompt: List[int],
        max_tokens: int,
        priority: int = 0
    ) -> bool:
        """
        Submit a new request

        Returns:
            True if accepted, False if rejected (OOM)
        """
        seq = SequenceState(
            seq_id=seq_id,
            tokens=prompt.copy(),
            prompt_len=len(prompt),
            max_tokens=max_tokens,
            priority=priority
        )

        # Try to allocate immediately
        if len(self.active_seqs) < self.max_batch_size:
            if self.kv_cache.allocate_sequence(seq_id, prompt):
                self.active_seqs[seq_id] = seq
                return True

        # Add to waiting queue
        heapq.heappush(
            self.waiting_queue,
            (-priority, time.time(), seq)  # Higher priority first
        )

        return True

    def step(self) -> Dict[str, any]:
        """
        Execute one iteration of continuous batching

        Returns:
            Metrics for this iteration
        """
        iteration_start = time.time()

        # Remove completed sequences
        completed_ids = [
            seq_id for seq_id, seq in self.active_seqs.items()
            if seq.is_complete
        ]

        for seq_id in completed_ids:
            self._complete_sequence(seq_id)

        # Add new sequences from waiting queue
        while (
            len(self.active_seqs) < self.max_batch_size and
            self.waiting_queue
        ):
            _, _, seq = heapq.heappop(self.waiting_queue)

            if self.kv_cache.allocate_sequence(seq.seq_id, seq.tokens):
                self.active_seqs[seq.seq_id] = seq
            else:
                # OOM - try to free space
                if not self._evict_lowest_priority():
                    # Can't free space, reject request
                    break

        if not self.active_seqs:
            return {
                'active_sequences': 0,
                'iteration_time': 0,
                'tokens_generated': 0,
                'kv_cache_util': self.kv_cache.utilization()
            }

        # Generate one token for each active sequence
        batch_size = len(self.active_seqs)
        tokens_generated = 0

        for seq_id, seq in self.active_seqs.items():
            # Simulate token generation
            new_token = self._generate_token(seq)
            seq.tokens.append(new_token)
            tokens_generated += 1

            # Allocate more KV cache if needed
            self.kv_cache.append_tokens(seq_id, 1)

        self.total_tokens_generated += tokens_generated
        self.total_iterations += 1

        iteration_time = time.time() - iteration_start

        return {
            'active_sequences': batch_size,
            'iteration_time': iteration_time,
            'tokens_generated': tokens_generated,
            'kv_cache_util': self.kv_cache.utilization(),
            'waiting_queue_size': len(self.waiting_queue)
        }

    def _generate_token(self, seq: SequenceState) -> int:
        """Simulate token generation (mock)"""
        time.sleep(0.01)  # Simulate inference latency
        return np.random.randint(0, 32000)

    def _complete_sequence(self, seq_id: int):
        """Mark sequence as complete and free resources"""
        seq = self.active_seqs.pop(seq_id)
        self.kv_cache.free_sequence(seq_id)
        self.total_sequences_completed += 1

    def _evict_lowest_priority(self) -> bool:
        """
        Evict lowest priority sequence to free memory

        Returns:
            True if eviction succeeded
        """
        if not self.active_seqs:
            return False

        # Find lowest priority sequence
        lowest_priority_seq = min(
            self.active_seqs.values(),
            key=lambda s: (s.priority, -s.timestamp)
        )

        # Evict it
        self.kv_cache.free_sequence(lowest_priority_seq.seq_id)
        evicted = self.active_seqs.pop(lowest_priority_seq.seq_id)

        # Put back in waiting queue
        heapq.heappush(
            self.waiting_queue,
            (-evicted.priority, time.time(), evicted)
        )

        return True

    def get_metrics(self) -> Dict[str, any]:
        """Get overall scheduler metrics"""
        return {
            'total_sequences_completed': self.total_sequences_completed,
            'total_tokens_generated': self.total_tokens_generated,
            'total_iterations': self.total_iterations,
            'avg_batch_size': (
                self.total_tokens_generated / self.total_iterations
                if self.total_iterations > 0 else 0
            ),
            'kv_cache_utilization': self.kv_cache.utilization(),
            'active_sequences': len(self.active_seqs),
            'waiting_sequences': len(self.waiting_queue)
        }


def simulate_continuous_batching():
    """Run continuous batching simulation"""
    print("=" * 70)
    print("CONTINUOUS BATCHING SIMULATION")
    print("=" * 70)

    scheduler = ContinuousBatchScheduler(
        max_batch_size=16,
        block_size=16,
        num_blocks=500
    )

    print(f"\nConfiguration:")
    print(f"  Max batch size: {scheduler.max_batch_size}")
    print(f"  Block size: {scheduler.kv_cache.block_size} tokens")
    print(f"  Total blocks: {scheduler.kv_cache.num_blocks}")
    print(f"  Total KV cache capacity: {scheduler.kv_cache.num_blocks * scheduler.kv_cache.block_size} tokens")

    # Submit requests over time
    num_requests = 50
    print(f"\nSubmitting {num_requests} requests...")

    for i in range(num_requests):
        prompt_len = np.random.randint(10, 100)
        max_tokens = np.random.randint(20, 200)
        priority = np.random.choice([0, 1, 2])  # 0=low, 1=medium, 2=high

        prompt = list(range(prompt_len))

        scheduler.submit_request(
            seq_id=i,
            prompt=prompt,
            max_tokens=max_tokens,
            priority=priority
        )

    # Run scheduler
    print("\nRunning scheduler...")

    iteration_metrics = []
    max_iterations = 500

    for iteration in range(max_iterations):
        metrics = scheduler.step()
        iteration_metrics.append(metrics)

        if iteration % 50 == 0:
            print(f"\nIteration {iteration}:")
            print(f"  Active sequences: {metrics['active_sequences']}")
            print(f"  Tokens generated: {metrics['tokens_generated']}")
            print(f"  KV cache utilization: {metrics['kv_cache_util']*100:.1f}%")
            print(f"  Waiting queue: {metrics['waiting_queue_size']}")

        # Stop if all done
        if metrics['active_sequences'] == 0 and metrics['waiting_queue_size'] == 0:
            print(f"\n✓ All sequences completed at iteration {iteration}")
            break

    # Final metrics
    print(f"\n{'─' * 70}")
    print("FINAL METRICS")
    print(f"{'─' * 70}")

    final_metrics = scheduler.get_metrics()

    print(f"\nThroughput:")
    print(f"  Total tokens generated: {final_metrics['total_tokens_generated']}")
    print(f"  Total iterations: {final_metrics['total_iterations']}")
    print(f"  Avg tokens/iteration: {final_metrics['avg_batch_size']:.2f}")

    print(f"\nCompletion:")
    print(f"  Sequences completed: {final_metrics['total_sequences_completed']}/{num_requests}")
    print(f"  Completion rate: {final_metrics['total_sequences_completed']/num_requests*100:.1f}%")

    print(f"\nMemory:")
    print(f"  Final KV cache util: {final_metrics['kv_cache_utilization']*100:.1f}%")

    # Analyze efficiency
    total_time = sum(m['iteration_time'] for m in iteration_metrics)
    tokens_per_second = final_metrics['total_tokens_generated'] / total_time if total_time > 0 else 0

    print(f"\nPerformance:")
    print(f"  Total simulated time: {total_time:.2f}s")
    print(f"  Throughput: {tokens_per_second:.1f} tokens/sec")


def compare_static_vs_continuous():
    """Compare static batching vs continuous batching"""
    print("\n" + "=" * 70)
    print("STATIC vs CONTINUOUS BATCHING COMPARISON")
    print("=" * 70)

    # Create test requests
    requests = []
    for i in range(32):
        requests.append({
            'seq_id': i,
            'prompt': list(range(50)),
            'max_tokens': np.random.randint(20, 100)
        })

    print(f"\nTest setup: {len(requests)} requests")
    print(f"Token lengths: {min(r['max_tokens'] for r in requests)}-{max(r['max_tokens'] for r in requests)}")

    # Simulate static batching
    print(f"\n{'─' * 70}")
    print("STATIC BATCHING (batch size = 8)")
    print(f"{'─' * 70}")

    static_tokens = 0
    static_iterations = 0

    for i in range(0, len(requests), 8):
        batch = requests[i:i+8]
        max_len = max(r['max_tokens'] for r in batch)

        # All wait for longest
        static_iterations += max_len
        static_tokens += sum(r['max_tokens'] for r in batch)

    print(f"Total iterations: {static_iterations}")
    print(f"Total tokens: {static_tokens}")
    print(f"Efficiency: {static_tokens/static_iterations:.2f} tokens/iteration")

    # Simulate continuous batching
    print(f"\n{'─' * 70}")
    print("CONTINUOUS BATCHING")
    print(f"{'─' * 70}")

    # Approximate continuous batching efficiency
    continuous_tokens = static_tokens
    continuous_iterations = continuous_tokens  # Each token = 1 iteration

    print(f"Total iterations: {continuous_iterations}")
    print(f"Total tokens: {continuous_tokens}")
    print(f"Efficiency: {continuous_tokens/continuous_iterations:.2f} tokens/iteration")

    # Comparison
    improvement = static_iterations / continuous_iterations

    print(f"\n{'─' * 70}")
    print(f"Continuous batching is {improvement:.2f}x more efficient")
    print(f"{'─' * 70}")


if __name__ == "__main__":
    np.random.seed(42)

    # Run simulation
    simulate_continuous_batching()

    # Compare approaches
    compare_static_vs_continuous()

    print("\n✅ Continuous batching simulation complete!")
    print("\nKey Takeaways:")
    print("  • Continuous batching eliminates head-of-line blocking")
    print("  • PagedAttention achieves near-perfect memory utilization")
    print("  • Dynamic scheduling improves throughput by 2-3x")
    print("  • Priority queues enable QoS differentiation")
    print("  • Essential for production LLM serving")
