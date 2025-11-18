#!/usr/bin/env python3
"""
Parallel Batch Processing

Demonstrates static and dynamic batching for improved throughput.
Shows GPU utilization improvements and memory management.
"""

import time
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import defaultdict
import queue
import threading


@dataclass
class Request:
    """Inference request"""
    request_id: int
    prompt: List[int]
    max_tokens: int
    timestamp: float


@dataclass
class BatchConfig:
    """Batch processing configuration"""
    max_batch_size: int = 8
    max_wait_ms: float = 50.0
    context_length: int = 2048


class MockGPUModel:
    """Mock GPU model with realistic batching behavior"""

    def __init__(self, latency_per_token_ms: float = 10.0):
        self.latency_per_token_ms = latency_per_token_ms

    def forward_batch(
        self,
        batch_tokens: List[List[int]],
        batch_size: int
    ) -> List[np.ndarray]:
        """
        Simulate batched forward pass
        Latency is roughly constant for batch (GPU parallelism)
        """
        # Simulate GPU processing time (mostly parallel)
        max_seq_len = max(len(seq) for seq in batch_tokens)
        time.sleep(self.latency_per_token_ms / 1000.0)

        # Generate logits for each sequence
        logits_batch = []
        for _ in range(batch_size):
            logits = np.random.randn(32000)  # Vocab size
            logits_batch.append(logits)

        return logits_batch


class StaticBatchEngine:
    """
    Static batching: Collect B requests, process together,
    return all when slowest completes
    """

    def __init__(self, model: MockGPUModel, config: BatchConfig):
        self.model = model
        self.config = config
        self.metrics = defaultdict(list)

    def process_requests(self, requests: List[Request]) -> Dict[int, List[int]]:
        """
        Process requests in static batches

        Returns:
            results: Dict mapping request_id to generated tokens
        """
        results = {}
        start_time = time.time()

        # Process in batches of batch_size
        for i in range(0, len(requests), self.config.max_batch_size):
            batch = requests[i:i+self.config.max_batch_size]
            batch_results = self._process_batch(batch)
            results.update(batch_results)

        elapsed = time.time() - start_time
        self._record_metrics(requests, elapsed)

        return results

    def _process_batch(self, batch: List[Request]) -> Dict[int, List[int]]:
        """Process a single batch"""
        batch_size = len(batch)
        active_sequences = {req.request_id: req.prompt.copy() for req in batch}
        completed = set()

        # Determine max length (wait for slowest)
        max_length = max(req.max_tokens for req in batch)

        for step in range(max_length):
            # Prepare batch input
            batch_tokens = [
                active_sequences[req.request_id]
                for req in batch
                if req.request_id not in completed
            ]

            if not batch_tokens:
                break

            # Batched forward pass
            logits_batch = self.model.forward_batch(batch_tokens, len(batch_tokens))

            # Sample for each active sequence
            for idx, req in enumerate(batch):
                if req.request_id in completed:
                    continue

                # Sample next token
                logits = logits_batch[idx]
                probs = self._softmax(logits)
                token = np.random.choice(len(probs), p=probs)

                active_sequences[req.request_id].append(token)

                # Check completion
                if len(active_sequences[req.request_id]) - len(req.prompt) >= req.max_tokens:
                    completed.add(req.request_id)

        return active_sequences

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        exp_logits = np.exp(logits - logits.max())
        return exp_logits / exp_logits.sum()

    def _record_metrics(self, requests: List[Request], elapsed: float):
        total_tokens = sum(req.max_tokens for req in requests)
        self.metrics['throughput'].append(total_tokens / elapsed)
        self.metrics['latency'].append(elapsed / len(requests))
        self.metrics['batch_size'].append(len(requests))


class DynamicBatchEngine:
    """
    Dynamic batching: Continuously add/remove requests from batch
    Better GPU utilization and lower latency
    """

    def __init__(self, model: MockGPUModel, config: BatchConfig):
        self.model = model
        self.config = config
        self.request_queue = queue.Queue()
        self.results = {}
        self.active = False
        self.metrics = defaultdict(list)

    def start(self):
        """Start background processing thread"""
        self.active = True
        self.thread = threading.Thread(target=self._processing_loop)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        """Stop processing"""
        self.active = False
        if hasattr(self, 'thread'):
            self.thread.join()

    def submit_request(self, request: Request) -> List[int]:
        """
        Submit request and wait for result

        Returns:
            Generated tokens
        """
        result_event = threading.Event()
        self.request_queue.put((request, result_event))

        # Wait for completion
        result_event.wait()

        return self.results.pop(request.request_id)

    def _processing_loop(self):
        """Main processing loop"""
        active_requests = {}  # request_id -> (Request, Event)
        sequences = {}  # request_id -> tokens

        while self.active:
            # Add new requests from queue
            batch_start_time = time.time()

            while (
                len(active_requests) < self.config.max_batch_size and
                not self.request_queue.empty()
            ):
                try:
                    request, event = self.request_queue.get_nowait()
                    active_requests[request.request_id] = (request, event)
                    sequences[request.request_id] = request.prompt.copy()
                except queue.Empty:
                    break

            # Wait a bit for more requests if batch not full
            if len(active_requests) < self.config.max_batch_size:
                time.sleep(self.config.max_wait_ms / 1000.0)

                # Try to fill batch again
                while (
                    len(active_requests) < self.config.max_batch_size and
                    not self.request_queue.empty()
                ):
                    try:
                        request, event = self.request_queue.get_nowait()
                        active_requests[request.request_id] = (request, event)
                        sequences[request.request_id] = request.prompt.copy()
                    except queue.Empty:
                        break

            if not active_requests:
                time.sleep(0.001)
                continue

            # Process one generation step for all active requests
            batch_tokens = [sequences[req_id] for req_id in active_requests]
            logits_batch = self.model.forward_batch(
                batch_tokens,
                len(active_requests)
            )

            # Sample and update each sequence
            completed_ids = []

            for idx, (req_id, (request, event)) in enumerate(active_requests.items()):
                logits = logits_batch[idx]
                probs = self._softmax(logits)
                token = np.random.choice(len(probs), p=probs)

                sequences[req_id].append(token)

                # Check completion
                if len(sequences[req_id]) - len(request.prompt) >= request.max_tokens:
                    self.results[req_id] = sequences[req_id]
                    event.set()
                    completed_ids.append(req_id)

            # Remove completed requests
            for req_id in completed_ids:
                del active_requests[req_id]
                del sequences[req_id]

            # Record metrics
            batch_time = time.time() - batch_start_time
            self.metrics['batch_size'].append(len(active_requests) + len(completed_ids))
            self.metrics['iteration_time'].append(batch_time)

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        exp_logits = np.exp(logits - logits.max())
        return exp_logits / exp_logits.sum()


def benchmark_static_vs_dynamic():
    """Compare static and dynamic batching"""
    print("=" * 70)
    print("STATIC vs DYNAMIC BATCHING BENCHMARK")
    print("=" * 70)

    model = MockGPUModel(latency_per_token_ms=10.0)

    # Generate test requests with varying lengths
    num_requests = 32
    requests = []
    for i in range(num_requests):
        req = Request(
            request_id=i,
            prompt=[1, 2, 3],
            max_tokens=np.random.randint(20, 100),  # Variable lengths!
            timestamp=time.time()
        )
        requests.append(req)

    print(f"\nTest configuration:")
    print(f"  Number of requests: {num_requests}")
    print(f"  Token lengths: {min(r.max_tokens for r in requests)}-{max(r.max_tokens for r in requests)}")
    print(f"  Avg tokens/request: {np.mean([r.max_tokens for r in requests]):.1f}")

    # Test 1: Static batching
    print(f"\n{'─' * 70}")
    print("STATIC BATCHING")
    print(f"{'─' * 70}")

    config = BatchConfig(max_batch_size=8, max_wait_ms=50.0)
    static_engine = StaticBatchEngine(model, config)

    start_time = time.time()
    static_results = static_engine.process_requests(requests)
    static_elapsed = time.time() - start_time

    total_tokens = sum(len(tokens) - 3 for tokens in static_results.values())  # -3 for prompt

    print(f"\nResults:")
    print(f"  Total time: {static_elapsed:.2f}s")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Throughput: {total_tokens / static_elapsed:.1f} tok/s")
    print(f"  Avg latency: {static_elapsed / num_requests:.2f}s")

    # Calculate waste due to padding
    actual_compute = total_tokens
    max_lengths_per_batch = []
    for i in range(0, num_requests, config.max_batch_size):
        batch = requests[i:i+config.max_batch_size]
        max_len = max(r.max_tokens for r in batch)
        max_lengths_per_batch.append(max_len * len(batch))

    total_padded_compute = sum(max_lengths_per_batch)
    waste_pct = (total_padded_compute - actual_compute) / total_padded_compute * 100

    print(f"  Padding waste: {waste_pct:.1f}%")

    # Test 2: Dynamic batching
    print(f"\n{'─' * 70}")
    print("DYNAMIC BATCHING")
    print(f"{'─' * 70}")

    dynamic_engine = DynamicBatchEngine(model, config)
    dynamic_engine.start()

    # Submit requests with some delay (simulate arrival)
    start_time = time.time()
    threads = []

    def submit_and_wait(req):
        result = dynamic_engine.submit_request(req)
        return result

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
        futures = []
        for req in requests:
            # Submit with slight delay
            time.sleep(0.01)  # Simulate streaming arrivals
            future = executor.submit(submit_and_wait, req)
            futures.append(future)

        # Wait for all
        dynamic_results = [f.result() for f in futures]

    dynamic_elapsed = time.time() - start_time
    dynamic_engine.stop()

    print(f"\nResults:")
    print(f"  Total time: {dynamic_elapsed:.2f}s")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Throughput: {total_tokens / dynamic_elapsed:.1f} tok/s")
    print(f"  Avg latency: {dynamic_elapsed / num_requests:.2f}s")
    print(f"  Avg batch size: {np.mean(dynamic_engine.metrics['batch_size']):.1f}")

    # Comparison
    print(f"\n{'─' * 70}")
    print("COMPARISON")
    print(f"{'─' * 70}")

    throughput_improvement = (total_tokens / dynamic_elapsed) / (total_tokens / static_elapsed)
    latency_improvement = static_elapsed / dynamic_elapsed

    print(f"\nDynamic vs Static:")
    print(f"  Throughput: {throughput_improvement:.2f}x better")
    print(f"  Latency: {latency_improvement:.2f}x better")
    print(f"  Reason: No waiting for slowest request in batch")

    print("\n" + "=" * 70)


def demonstrate_batch_size_optimization():
    """Show optimal batch size selection"""
    print("\n" + "=" * 70)
    print("BATCH SIZE OPTIMIZATION")
    print("=" * 70)

    model = MockGPUModel(latency_per_token_ms=10.0)

    for batch_size in [1, 2, 4, 8, 16, 32]:
        config = BatchConfig(max_batch_size=batch_size)
        engine = StaticBatchEngine(model, config)

        # Create uniform requests
        requests = [
            Request(i, [1, 2, 3], max_tokens=50, timestamp=time.time())
            for i in range(batch_size * 4)  # 4 batches
        ]

        start_time = time.time()
        results = engine.process_requests(requests)
        elapsed = time.time() - start_time

        total_tokens = sum(len(tokens) - 3 for tokens in results.values())

        print(f"\nBatch size = {batch_size}:")
        print(f"  Throughput: {total_tokens / elapsed:.1f} tok/s")
        print(f"  Avg latency: {elapsed / len(requests):.3f}s")


if __name__ == "__main__":
    np.random.seed(42)

    # Run benchmarks
    benchmark_static_vs_dynamic()

    # Show batch size optimization
    demonstrate_batch_size_optimization()

    print("\n✅ Batch processing demo complete!")
    print("\nKey Takeaways:")
    print("  • Static batching: Simple but wastes compute on padding")
    print("  • Dynamic batching: Better utilization, lower latency")
    print("  • Optimal batch size: Balance throughput vs latency")
    print("  • GPU batching provides 5-10x throughput improvement")
