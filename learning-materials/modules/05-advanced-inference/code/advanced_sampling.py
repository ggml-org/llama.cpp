#!/usr/bin/env python3
"""
Advanced Sampling Algorithms

Implements and compares various sampling strategies:
- Temperature, Top-K, Top-P, Min-P
- Mirostat V1 and V2
- Locally Typical Sampling
- Custom sampling pipelines
"""

import numpy as np
from typing import List, Optional, Callable
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class SamplingConfig:
    """Configuration for sampling"""
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.9
    min_p: float = 0.0
    repeat_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


class BaseSampler:
    """Base class for samplers"""

    def sample(self, logits: np.ndarray, context: Optional[List[int]] = None) -> int:
        """Sample a token from logits"""
        raise NotImplementedError

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Numerically stable softmax"""
        logits = logits - logits.max()
        exp_logits = np.exp(logits)
        return exp_logits / exp_logits.sum()


class GreedySampler(BaseSampler):
    """Always pick highest probability token"""

    def sample(self, logits: np.ndarray, context: Optional[List[int]] = None) -> int:
        return np.argmax(logits)


class TemperatureSampler(BaseSampler):
    """Temperature-scaled sampling"""

    def __init__(self, temperature: float = 0.7):
        self.temperature = temperature

    def sample(self, logits: np.ndarray, context: Optional[List[int]] = None) -> int:
        scaled_logits = logits / self.temperature
        probs = self._softmax(scaled_logits)
        return np.random.choice(len(probs), p=probs)


class TopKSampler(BaseSampler):
    """Top-K sampling"""

    def __init__(self, k: int = 40, temperature: float = 1.0):
        self.k = k
        self.temperature = temperature

    def sample(self, logits: np.ndarray, context: Optional[List[int]] = None) -> int:
        # Apply temperature
        scaled_logits = logits / self.temperature

        # Get top-k
        top_k_indices = np.argpartition(scaled_logits, -self.k)[-self.k:]
        top_k_logits = scaled_logits[top_k_indices]

        # Sample from top-k
        probs = self._softmax(top_k_logits)
        local_idx = np.random.choice(len(probs), p=probs)

        return top_k_indices[local_idx]


class TopPSampler(BaseSampler):
    """Top-P (nucleus) sampling"""

    def __init__(self, p: float = 0.9, temperature: float = 1.0):
        self.p = p
        self.temperature = temperature

    def sample(self, logits: np.ndarray, context: Optional[List[int]] = None) -> int:
        # Apply temperature
        scaled_logits = logits / self.temperature
        probs = self._softmax(scaled_logits)

        # Sort by probability
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]

        # Find cumulative probability cutoff
        cumsum = np.cumsum(sorted_probs)
        cutoff_idx = np.searchsorted(cumsum, self.p) + 1

        # Sample from top-p
        top_p_indices = sorted_indices[:cutoff_idx]
        top_p_probs = probs[top_p_indices]
        top_p_probs = top_p_probs / top_p_probs.sum()

        local_idx = np.random.choice(len(top_p_probs), p=top_p_probs)
        return top_p_indices[local_idx]


class MinPSampler(BaseSampler):
    """Min-P sampling (relative threshold)"""

    def __init__(self, min_p: float = 0.05, temperature: float = 1.0):
        self.min_p = min_p
        self.temperature = temperature

    def sample(self, logits: np.ndarray, context: Optional[List[int]] = None) -> int:
        # Apply temperature
        scaled_logits = logits / self.temperature
        probs = self._softmax(scaled_logits)

        # Filter by min_p * max_prob
        max_prob = probs.max()
        threshold = self.min_p * max_prob

        mask = probs >= threshold
        filtered_probs = probs[mask]
        filtered_probs = filtered_probs / filtered_probs.sum()

        filtered_indices = np.where(mask)[0]
        local_idx = np.random.choice(len(filtered_probs), p=filtered_probs)

        return filtered_indices[local_idx]


class MirostatV2Sampler(BaseSampler):
    """
    Mirostat V2: Perplexity-controlled sampling
    Maintains target perplexity via feedback loop
    """

    def __init__(
        self,
        tau: float = 5.0,  # Target perplexity
        eta: float = 0.1,  # Learning rate
        temperature: float = 1.0
    ):
        self.tau = tau
        self.eta = eta
        self.temperature = temperature
        self.mu = 2 * tau  # Initial threshold

    def sample(self, logits: np.ndarray, context: Optional[List[int]] = None) -> int:
        # Apply temperature
        scaled_logits = logits / self.temperature
        probs = self._softmax(scaled_logits)

        # Truncate to tokens with prob > exp(-mu)
        threshold = np.exp(-self.mu)
        mask = probs > threshold

        # Sample from truncated distribution
        filtered_probs = probs[mask]
        filtered_probs = filtered_probs / filtered_probs.sum()

        filtered_indices = np.where(mask)[0]
        local_idx = np.random.choice(len(filtered_probs), p=filtered_probs)
        token = filtered_indices[local_idx]

        # Update mu based on observed surprise
        observed_surprise = -np.log(probs[token])
        error = observed_surprise - self.tau
        self.mu -= self.eta * error

        # Clamp mu to reasonable range
        self.mu = np.clip(self.mu, 0.1, 20.0)

        return token

    def reset(self):
        """Reset mu to initial value"""
        self.mu = 2 * self.tau


class LocallyTypicalSampler(BaseSampler):
    """
    Locally Typical Sampling
    Samples tokens with surprise close to conditional entropy
    """

    def __init__(self, epsilon: float = 0.2, temperature: float = 1.0):
        self.epsilon = epsilon
        self.temperature = temperature

    def sample(self, logits: np.ndarray, context: Optional[List[int]] = None) -> int:
        # Apply temperature
        scaled_logits = logits / self.temperature
        probs = self._softmax(scaled_logits)

        # Compute conditional entropy
        entropy = -np.sum(probs * np.log(probs + 1e-10))

        # Compute surprise for each token
        surprises = -np.log(probs + 1e-10)

        # Keep tokens with surprise close to entropy
        diff = np.abs(surprises - entropy)
        mask = diff < self.epsilon

        if not np.any(mask):
            # Fallback to all tokens if none match
            mask = np.ones_like(mask, dtype=bool)

        # Sample from filtered distribution
        filtered_probs = probs[mask]
        filtered_probs = filtered_probs / filtered_probs.sum()

        filtered_indices = np.where(mask)[0]
        local_idx = np.random.choice(len(filtered_probs), p=filtered_probs)

        return filtered_indices[local_idx]


class RepetitionPenaltySampler(BaseSampler):
    """Apply repetition penalties to logits"""

    def __init__(
        self,
        base_sampler: BaseSampler,
        repeat_penalty: float = 1.1,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0
    ):
        self.base_sampler = base_sampler
        self.repeat_penalty = repeat_penalty
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty

    def sample(self, logits: np.ndarray, context: Optional[List[int]] = None) -> int:
        if context is None or len(context) == 0:
            return self.base_sampler.sample(logits, context)

        # Apply penalties
        penalized_logits = logits.copy()

        # Count token frequencies
        from collections import Counter
        token_counts = Counter(context)

        for token, count in token_counts.items():
            if token < len(penalized_logits):
                # Repeat penalty (scale down if already in logits)
                if penalized_logits[token] > 0:
                    penalized_logits[token] /= self.repeat_penalty
                else:
                    penalized_logits[token] *= self.repeat_penalty

                # Frequency penalty (proportional to count)
                penalized_logits[token] -= self.frequency_penalty * count

                # Presence penalty (binary)
                penalized_logits[token] -= self.presence_penalty

        return self.base_sampler.sample(penalized_logits, context)


class SamplingPipeline:
    """Chain multiple sampling stages"""

    def __init__(self, stages: List[Callable]):
        self.stages = stages

    def sample(self, logits: np.ndarray, context: Optional[List[int]] = None) -> int:
        current_logits = logits.copy()

        for stage in self.stages:
            if isinstance(stage, BaseSampler):
                return stage.sample(current_logits, context)
            else:
                # Stage is a logits transformer
                current_logits = stage(current_logits, context)

        # Fallback: greedy sampling
        return np.argmax(current_logits)


def visualize_sampling_strategies():
    """Compare different sampling strategies visually"""
    print("=" * 70)
    print("SAMPLING STRATEGY COMPARISON")
    print("=" * 70)

    # Create test distribution (realistic model output)
    vocab_size = 100
    logits = np.random.randn(vocab_size)

    # Make some tokens more likely
    logits[0] += 3.0  # "the"
    logits[1] += 2.0  # "a"
    logits[2] += 1.5  # "and"
    logits[3] += 1.0  # "to"

    # Compute base probabilities
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()

    print(f"\nBase distribution (top 10 tokens):")
    top_indices = np.argsort(probs)[::-1][:10]
    for i, idx in enumerate(top_indices):
        print(f"  {i+1}. Token {idx}: {probs[idx]:.4f}")

    # Test different samplers
    samplers = {
        'Greedy': GreedySampler(),
        'Temperature=0.3': TemperatureSampler(0.3),
        'Temperature=0.7': TemperatureSampler(0.7),
        'Temperature=1.5': TemperatureSampler(1.5),
        'Top-K (k=10)': TopKSampler(k=10),
        'Top-P (p=0.9)': TopPSampler(p=0.9),
        'Min-P (p=0.05)': MinPSampler(min_p=0.05),
        'Mirostat V2': MirostatV2Sampler(tau=5.0),
        'Locally Typical': LocallyTypicalSampler(epsilon=0.2),
    }

    print(f"\n{'─' * 70}")
    print("SAMPLING RESULTS (100 samples each)")
    print(f"{'─' * 70}")

    for name, sampler in samplers.items():
        if isinstance(sampler, MirostatV2Sampler):
            sampler.reset()

        # Sample 100 times
        samples = [sampler.sample(logits) for _ in range(100)]

        # Analyze samples
        unique_tokens = len(set(samples))
        most_common = max(set(samples), key=samples.count)
        most_common_pct = samples.count(most_common) / len(samples) * 100

        print(f"\n{name}:")
        print(f"  Unique tokens: {unique_tokens}/100")
        print(f"  Most common: Token {most_common} ({most_common_pct:.1f}%)")
        print(f"  Entropy: {calculate_entropy(samples):.2f} bits")


def calculate_entropy(samples: List[int]) -> float:
    """Calculate empirical entropy of samples"""
    from collections import Counter
    counts = Counter(samples)
    probs = np.array([count / len(samples) for count in counts.values()])
    return -np.sum(probs * np.log2(probs + 1e-10))


def benchmark_mirostat():
    """Demonstrate Mirostat perplexity control"""
    print("\n" + "=" * 70)
    print("MIROSTAT PERPLEXITY CONTROL")
    print("=" * 70)

    vocab_size = 1000
    num_samples = 200

    for target_tau in [3.0, 5.0, 8.0]:
        print(f"\nTarget tau (perplexity) = {target_tau}")

        sampler = MirostatV2Sampler(tau=target_tau, eta=0.1)

        surprises = []
        mus = []

        for _ in range(num_samples):
            # Generate random logits (simulating model)
            logits = np.random.randn(vocab_size)

            # Sample
            token = sampler.sample(logits)

            # Record surprise
            probs = np.exp(logits - logits.max())
            probs /= probs.sum()
            surprise = -np.log(probs[token])

            surprises.append(surprise)
            mus.append(sampler.mu)

        avg_surprise = np.mean(surprises[-50:])  # Last 50 samples (after warmup)

        print(f"  Average surprise (last 50): {avg_surprise:.2f}")
        print(f"  Target surprise: {target_tau:.2f}")
        print(f"  Error: {abs(avg_surprise - target_tau):.2f}")
        print(f"  Final mu: {sampler.mu:.2f}")


def demonstrate_combined_sampling():
    """Show combining multiple sampling techniques"""
    print("\n" + "=" * 70)
    print("COMBINED SAMPLING PIPELINE")
    print("=" * 70)

    vocab_size = 100
    logits = np.random.randn(vocab_size)
    logits[0] += 3.0

    # Simulate context with repeated token
    context = [0, 5, 10, 0, 15, 0]  # Token 0 appears 3 times

    # Pipeline: Repetition penalty → Temperature → Top-P
    base_sampler = TopPSampler(p=0.9, temperature=0.7)
    combined_sampler = RepetitionPenaltySampler(
        base_sampler,
        repeat_penalty=1.1,
        frequency_penalty=0.5,
        presence_penalty=0.2
    )

    print(f"\nContext: {context}")
    print(f"Token 0 appears {context.count(0)} times")

    # Sample without penalty
    print(f"\nWithout repetition penalty:")
    samples_no_penalty = [base_sampler.sample(logits) for _ in range(100)]
    token_0_count_no_penalty = samples_no_penalty.count(0)
    print(f"  Token 0 sampled: {token_0_count_no_penalty}/100 times ({token_0_count_no_penalty}%)")

    # Sample with penalty
    print(f"\nWith repetition penalty:")
    samples_with_penalty = [combined_sampler.sample(logits, context) for _ in range(100)]
    token_0_count_with_penalty = samples_with_penalty.count(0)
    print(f"  Token 0 sampled: {token_0_count_with_penalty}/100 times ({token_0_count_with_penalty}%)")

    reduction = (token_0_count_no_penalty - token_0_count_with_penalty) / token_0_count_no_penalty * 100
    print(f"  Reduction: {reduction:.1f}%")


if __name__ == "__main__":
    np.random.seed(42)

    # Run demonstrations
    visualize_sampling_strategies()
    benchmark_mirostat()
    demonstrate_combined_sampling()

    print("\n✅ Advanced sampling demo complete!")
    print("\nKey Takeaways:")
    print("  • Temperature: Universal randomness control")
    print("  • Top-P: Better than Top-K for adaptive filtering")
    print("  • Min-P: Best for handling flat distributions")
    print("  • Mirostat: Active perplexity control for quality")
    print("  • Combine techniques for optimal results")
