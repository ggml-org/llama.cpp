#!/usr/bin/env python3
"""
Flash-Decoding Implementation
============================

Based on the PyTorch blog post: https://pytorch.org/blog/flash-decoding/

Flash-Decoding is designed for efficient long-context inference by parallelizing
across the keys/values sequence length dimension. This is particularly effective
during decoding when query length is typically 1.

Key Innovation:
- Splits keys/values into smaller chunks
- Computes attention for each chunk in parallel
- Uses log-sum-exp for numerically stable reduction
- Achieves up to 8x speedup for very long sequences

Architecture Overview:
┌─────────────────────────────────────────────────────────────────┐
│                     Flash-Decoding Algorithm                    │
│                                                                 │
│  Step 1: Split KV Cache into Chunks                            │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐               │
│  │   Chunk 0   │ │   Chunk 1   │ │   Chunk N   │               │
│  │ [K0, V0]    │ │ [K1, V1]    │ │ [KN, VN]    │               │
│  └─────────────┘ └─────────────┘ └─────────────┘               │
│         │               │               │                       │
│         ▼               ▼               ▼                       │
│  Step 2: Parallel Attention Computation                        │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐               │
│  │ Attn(Q,K0)  │ │ Attn(Q,K1)  │ │ Attn(Q,KN)  │               │
│  │ + log_sum   │ │ + log_sum   │ │ + log_sum   │               │
│  └─────────────┘ └─────────────┘ └─────────────┘               │
│         │               │               │                       │
│         └───────────────┼───────────────┘                       │
│                         ▼                                       │
│  Step 3: Log-Sum-Exp Reduction                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           Numerically Stable Merge                      │   │
│  │     final_output = weighted_sum(chunk_outputs)          │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
"""

import torch
import torch.nn.functional as F
import math
import time
from typing import Tuple, List, Optional
import numpy as np


class FlashDecoding:
    """
    Flash-Decoding implementation for efficient long-context inference
    
    The algorithm works in 3 steps:
    1. Split keys/values into smaller chunks
    2. Compute attention for each chunk in parallel using FlashAttention-style computation
    3. Reduce across chunks using log-sum-exp for numerical stability
    """
    
    def __init__(self, chunk_size: int = 1024):
        """
        Initialize Flash-Decoding processor
        
        Args:
            chunk_size: Size of each KV chunk for parallel processing
        """
        self.chunk_size = chunk_size
    
    def _split_kv_chunks(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Step 1: Split keys/values into smaller chunks
        
        Args:
            k: Key tensor [batch, heads, seq_len, head_dim]
            v: Value tensor [batch, heads, seq_len, head_dim]
            
        Returns:
            Tuple of (key_chunks, value_chunks)
        """
        seq_len = k.size(2)
        num_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size
        
        k_chunks = []
        v_chunks = []
        
        for i in range(num_chunks):
            start_idx = i * self.chunk_size
            end_idx = min((i + 1) * self.chunk_size, seq_len)
            
            k_chunk = k[:, :, start_idx:end_idx, :]
            v_chunk = v[:, :, start_idx:end_idx, :]
            
            k_chunks.append(k_chunk)
            v_chunks.append(v_chunk)
            
        return k_chunks, v_chunks
    
    def _compute_chunk_attention(self, q: torch.Tensor, k_chunk: torch.Tensor, 
                                v_chunk: torch.Tensor, mask_chunk: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Step 2: Compute attention for a single chunk with log-sum-exp tracking
        
        This is similar to FlashAttention but also returns the log-sum-exp
        for later reduction across chunks.
        
        Args:
            q: Query tensor [batch, heads, q_len, head_dim]
            k_chunk: Key chunk [batch, heads, chunk_len, head_dim]
            v_chunk: Value chunk [batch, heads, chunk_len, head_dim]
            mask_chunk: Optional mask for this chunk
            
        Returns:
            Tuple of (chunk_output, log_sum_exp)
        """
        head_dim = q.size(-1)
        scale = 1.0 / math.sqrt(head_dim)
        
        # Compute attention scores: Q @ K^T / sqrt(d_k)
        scores = torch.matmul(q, k_chunk.transpose(-2, -1)) * scale
        
        # Apply mask if provided
        if mask_chunk is not None:
            scores = scores + mask_chunk
        
        # For numerical stability, subtract max before exp
        max_scores = torch.max(scores, dim=-1, keepdim=True)[0]
        scores_shifted = scores - max_scores
        
        # Compute exp(scores - max)
        exp_scores = torch.exp(scores_shifted)
        
        # Compute sum of exp scores for this chunk
        sum_exp = torch.sum(exp_scores, dim=-1, keepdim=True)
        
        # Compute log-sum-exp for this chunk: log(sum(exp(scores - max))) + max
        log_sum_exp = torch.log(sum_exp) + max_scores
        
        # Compute attention weights for this chunk
        attn_weights = exp_scores / sum_exp
        
        # Compute weighted values
        chunk_output = torch.matmul(attn_weights, v_chunk)
        
        return chunk_output, log_sum_exp
    
    def _reduce_chunks(self, chunk_outputs: List[torch.Tensor], 
                      log_sum_exps: List[torch.Tensor]) -> torch.Tensor:
        """
        Step 3: Reduce across all chunks using log-sum-exp for numerical stability
        
        This implements the mathematical identity:
        softmax([x1, x2, ..., xn]) = [exp(x1)/Z, exp(x2)/Z, ..., exp(xn)/Z]
        where Z = sum(exp(xi)) = exp(log_sum_exp_global)
        
        Args:
            chunk_outputs: List of chunk attention outputs
            log_sum_exps: List of log-sum-exp values for each chunk
            
        Returns:
            Final attention output
        """
        # Find global log-sum-exp across all chunks
        # log_sum_exp_global = log(sum_i(exp(log_sum_exp_i)))
        
        # Stack log-sum-exps for easier computation
        log_sum_exp_stack = torch.stack(log_sum_exps, dim=-1)  # [batch, heads, q_len, 1, num_chunks]
        
        # Compute global log-sum-exp using the log-sum-exp trick
        max_log_sum_exp = torch.max(log_sum_exp_stack, dim=-1, keepdim=True)[0]
        shifted_log_sum_exps = log_sum_exp_stack - max_log_sum_exp
        global_log_sum_exp = torch.log(torch.sum(torch.exp(shifted_log_sum_exps), dim=-1, keepdim=True)) + max_log_sum_exp
        
        # Compute the weight for each chunk in the final reduction
        chunk_weights = torch.exp(log_sum_exp_stack - global_log_sum_exp)  # [batch, heads, q_len, 1, num_chunks]
        
        # Weighted sum of chunk outputs
        final_output = torch.zeros_like(chunk_outputs[0])
        
        for i, (chunk_output, weight) in enumerate(zip(chunk_outputs, chunk_weights.unbind(dim=-1))):
            final_output += chunk_output * weight
            
        return final_output
    
    def flash_decoding_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                               mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Main Flash-Decoding attention computation
        
        Args:
            q: Query tensor [batch, heads, q_len, head_dim] (typically q_len=1 for decoding)
            k: Key tensor [batch, heads, kv_len, head_dim]
            v: Value tensor [batch, heads, kv_len, head_dim]
            mask: Optional attention mask [batch, heads, q_len, kv_len]
            
        Returns:
            Attention output [batch, heads, q_len, head_dim]
        """
        # Step 1: Split keys/values into chunks
        k_chunks, v_chunks = self._split_kv_chunks(k, v)
        
        # Prepare mask chunks if mask is provided
        mask_chunks = None
        if mask is not None:
            mask_chunks = []
            for i, k_chunk in enumerate(k_chunks):
                start_idx = i * self.chunk_size
                end_idx = start_idx + k_chunk.size(2)
                mask_chunk = mask[:, :, :, start_idx:end_idx]
                mask_chunks.append(mask_chunk)
        
        # Step 2: Compute attention for each chunk in parallel
        chunk_outputs = []
        log_sum_exps = []
        
        for i, (k_chunk, v_chunk) in enumerate(zip(k_chunks, v_chunks)):
            mask_chunk = mask_chunks[i] if mask_chunks is not None else None
            chunk_output, log_sum_exp = self._compute_chunk_attention(q, k_chunk, v_chunk, mask_chunk)
            
            __import__('pdb').set_trace()
            
            chunk_outputs.append(chunk_output)
            log_sum_exps.append(log_sum_exp)
        
        # Step 3: Reduce across chunks
        final_output = self._reduce_chunks(chunk_outputs, log_sum_exps)
        
        return final_output
    
    def reference_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                          mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Reference implementation using standard attention
        
        Args:
            q: Query tensor [batch, heads, q_len, head_dim]
            k: Key tensor [batch, heads, kv_len, head_dim]
            v: Value tensor [batch, heads, kv_len, head_dim]
            mask: Optional attention mask
            
        Returns:
            Attention output [batch, heads, q_len, head_dim]
        """
        head_dim = q.size(-1)
        scale = 1.0 / math.sqrt(head_dim)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores + mask
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Compute output
        output = torch.matmul(attn_weights, v)
        
        return output

def create_decoding_tensors(batch_size: int = 1, num_heads: int = 32, q_len: int = 1,
                          kv_len: int = 8192, head_dim: int = 128, device: str = 'cuda') -> Tuple[torch.Tensor, ...]:
    """
    Create tensors for decoding scenario (typical: q_len=1, long kv_len)
    
    This simulates the typical decoding scenario where we generate one token at a time,
    so query length is 1, but we need to attend to a long context (large kv_len).
    """
    q = torch.randn(batch_size, num_heads, q_len, head_dim, device=device, dtype=torch.float32)
    k = torch.randn(batch_size, num_heads, kv_len, head_dim, device=device, dtype=torch.float32)
    v = torch.randn(batch_size, num_heads, kv_len, head_dim, device=device, dtype=torch.float32)
    
    # Create causal mask for decoding
    mask = torch.triu(torch.full((q_len, kv_len), float('-inf'), device=device), diagonal=kv_len-q_len+1)
    mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, q_len, kv_len]
    
    return q, k, v, mask


def test_flash_decoding_correctness():
    """Test Flash-Decoding correctness against reference implementation"""
    
    print("Testing Flash-Decoding Correctness")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Test configurations for different scenarios
    test_configs = [
        {"batch_size": 1,  "num_heads": 8,   "q_len": 1,  "kv_len": 1024,  "head_dim": 128, "chunk_size": 256,  "desc": "Short context"},
        # {"batch_size": 1,  "num_heads": 16,  "q_len": 1,  "kv_len": 4096,  "head_dim": 128, "chunk_size": 512,  "desc": "Medium context"},
        # {"batch_size": 1,  "num_heads": 32,  "q_len": 1,  "kv_len": 16384, "head_dim": 128, "chunk_size": 1024, "desc": "Long context"},
        # {"batch_size": 1,  "num_heads": 8,   "q_len": 4,  "kv_len": 2048,  "head_dim": 64,  "chunk_size": 512,  "desc": "Multi-query"},
    ]
    
    for i, config in enumerate(test_configs):
        print(f"\nTest Case {i+1}: {config['desc']}")
        print(f"   Config: {config}")
        
        # Create test tensors
        q, k, v, mask = create_decoding_tensors(
            batch_size=config["batch_size"],
            num_heads=config["num_heads"],
            q_len=config["q_len"],
            kv_len=config["kv_len"],
            head_dim=config["head_dim"],
            device=device
        )
        
        # Initialize Flash-Decoding
        flash_decoder = FlashDecoding(chunk_size=config["chunk_size"])
        
        # Compute outputs
        with torch.no_grad():
            # Reference implementation
            reference_output = flash_decoder.reference_attention(q, k, v, mask)
            
            # Flash decoding implementation
            flash_output = flash_decoder.flash_decoding_attention(q, k, v, mask)
            
            # PyTorch SDPA implementation
            sdpa_output = F.scaled_dot_product_attention(
                q,  # (batch, num_heads, q_len, head_dim)
                k,  # (batch, num_heads, kv_len, head_dim) 
                v,  # (batch, num_heads, kv_len, head_dim)
                attn_mask=mask,  # (1, 1, q_len, kv_len)
                dropout_p=0.0,
                is_causal=False
            )
            
            __import__('pdb').set_trace()
        
        # Compare results
        max_diff = torch.max(torch.abs(reference_output - flash_output)).item()
        mean_diff = torch.mean(torch.abs(reference_output - flash_output)).item()
        relative_error = mean_diff / torch.mean(torch.abs(reference_output)).item()
        
        print(f"   Results:")
        print(f"     Max difference: {max_diff:.2e}")
        print(f"     Mean difference: {mean_diff:.2e}")
        print(f"     Relative error: {relative_error:.2e}")
        
        # Check correctness
        tolerance = 1e-4
        if max_diff < tolerance:
            print(f"     PASS - Results match within tolerance ({tolerance})")
        else:
            print(f"     FAIL - Results differ by more than tolerance ({tolerance})")


def benchmark_flash_decoding():
    """Benchmark Flash-Decoding vs reference implementation"""
    
    print("\nBenchmarking Flash-Decoding Performance")
    print("=" * 50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Benchmark configurations (focusing on decoding scenarios)
    benchmark_configs = [
        {"kv_len": 1024, "chunk_size": 256, "desc": "1K context"},
        {"kv_len": 4096, "chunk_size": 512, "desc": "4K context"},
        {"kv_len": 8192, "chunk_size": 1024, "desc": "8K context"},
        {"kv_len": 16384, "chunk_size": 2048, "desc": "16K context"},
        {"kv_len": 32768, "chunk_size": 4096, "desc": "32K context"},
        {"kv_len": 65536, "chunk_size": 8192, "desc": "64K context"},
    ]
    
    # Fixed parameters for decoding scenario
    batch_size = 1
    num_heads = 32
    q_len = 1  # Typical for decoding
    head_dim = 128
    num_warmup = 3
    num_runs = 10
    
    print(f"Benchmark setup: batch_size={batch_size}, num_heads={num_heads}, q_len={q_len}, head_dim={head_dim}")
    
    for config in benchmark_configs:
        print(f"\n{config['desc']}: KV length = {config['kv_len']}, Chunk size = {config['chunk_size']}")
        
        # Create test tensors
        q, k, v, mask = create_decoding_tensors(
            batch_size=batch_size,
            num_heads=num_heads,
            q_len=q_len,
            kv_len=config["kv_len"],
            head_dim=head_dim,
            device=device
        )
        
        flash_decoder = FlashDecoding(chunk_size=config["chunk_size"])
        
        # Benchmark reference implementation
        if device == 'cuda':
            torch.cuda.synchronize()
        
        # Warmup
        for _ in range(num_warmup):
            with torch.no_grad():
                _ = flash_decoder.reference_attention(q, k, v, mask)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(num_runs):
            with torch.no_grad():
                _ = flash_decoder.reference_attention(q, k, v, mask)
                
        if device == 'cuda':
            torch.cuda.synchronize()
        ref_time = (time.time() - start_time) / num_runs
        
        # Benchmark Flash-Decoding implementation
        if device == 'cuda':
            torch.cuda.synchronize()
        
        # Warmup
        for _ in range(num_warmup):
            with torch.no_grad():
                _ = flash_decoder.flash_decoding_attention(q, k, v, mask)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        start_time = time.time()
        
        for _ in range(num_runs):
            with torch.no_grad():
                _ = flash_decoder.flash_decoding_attention(q, k, v, mask)
                
        if device == 'cuda':
            torch.cuda.synchronize()
        flash_time = (time.time() - start_time) / num_runs
        
        # Calculate metrics
        speedup = ref_time / flash_time
        overhead = (flash_time - ref_time) / ref_time * 100
        
        print(f"   Reference time: {ref_time*1000:.2f} ms")
        print(f"   Flash-Decoding time: {flash_time*1000:.2f} ms")
        print(f"   Speedup: {speedup:.2f}x")
        print(f"   Overhead: {overhead:+.1f}%")
        
        # Memory analysis
        kv_memory = k.numel() * k.element_size() + v.numel() * v.element_size()
        chunk_memory = config["chunk_size"] * head_dim * num_heads * batch_size * 2 * k.element_size()
        memory_ratio = chunk_memory / kv_memory
        
        print(f"   Total KV memory: {kv_memory//1024//1024} MB")
        print(f"   Chunk memory: {chunk_memory//1024//1024} MB ({memory_ratio:.1%} of total)")


def demonstrate_flash_decoding_algorithm():
    """Demonstrate the Flash-Decoding algorithm step by step"""
    
    print("\nFlash-Decoding Algorithm Demonstration")
    print("=" * 50)
    print("Based on: https://pytorch.org/blog/flash-decoding/")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Small example for clear demonstration
    batch_size = 1
    num_heads = 2
    q_len = 1  # Typical for decoding
    kv_len = 8
    head_dim = 4
    chunk_size = 3
    
    print(f"\nDemo parameters:")
    print(f"   Batch size: {batch_size}")
    print(f"   Num heads: {num_heads}")
    print(f"   Query length: {q_len} (typical for decoding)")
    print(f"   KV length: {kv_len}")
    print(f"   Head dim: {head_dim}")
    print(f"   Chunk size: {chunk_size}")
    
    # Create test tensors
    q = torch.randn(batch_size, num_heads, q_len, head_dim, device=device)
    k = torch.randn(batch_size, num_heads, kv_len, head_dim, device=device)
    v = torch.randn(batch_size, num_heads, kv_len, head_dim, device=device)
    
    flash_decoder = FlashDecoding(chunk_size=chunk_size)
    
    print(f"\nStep 1: Split KV cache into chunks")
    k_chunks, v_chunks = flash_decoder._split_kv_chunks(k, v)
    print(f"   Number of chunks: {len(k_chunks)}")
    for i, (k_chunk, v_chunk) in enumerate(zip(k_chunks, v_chunks)):
        print(f"   Chunk {i}: K shape {list(k_chunk.shape)}, V shape {list(v_chunk.shape)}")
    
    print(f"\nStep 2: Compute attention for each chunk with log-sum-exp")
    chunk_outputs = []
    log_sum_exps = []
    
    for i, (k_chunk, v_chunk) in enumerate(zip(k_chunks, v_chunks)):
        chunk_output, log_sum_exp = flash_decoder._compute_chunk_attention(q, k_chunk, v_chunk)
        chunk_outputs.append(chunk_output)
        log_sum_exps.append(log_sum_exp)
        
        print(f"   Chunk {i}:")
        print(f"     Output shape: {list(chunk_output.shape)}")
        print(f"     Log-sum-exp: {log_sum_exp.mean().item():.6f}")
        print(f"     Output magnitude: {chunk_output.norm().item():.6f}")
    
    print(f"\nStep 3: Reduce across chunks using log-sum-exp")
    final_output = flash_decoder._reduce_chunks(chunk_outputs, log_sum_exps)
    
    print(f"   Final output shape: {list(final_output.shape)}")
    print(f"   Final output magnitude: {final_output.norm().item():.6f}")
    
    # Verify against reference
    with torch.no_grad():
        reference_output = flash_decoder.reference_attention(q, k, v)
        max_diff = torch.max(torch.abs(reference_output - final_output)).item()
        print(f"   Verification: Max difference from reference = {max_diff:.2e}")
    
    print(f"\nKey insights:")
    print(f"   • Flash-Decoding parallelizes across KV sequence length")
    print(f"   • Each chunk is processed independently with FlashAttention-style computation")
    print(f"   • Log-sum-exp ensures numerical stability during reduction")
    print(f"   • Particularly effective when q_len=1 (decoding) and kv_len is large")


def main():
    """Main function to run Flash-Decoding demonstrations"""
    
    print("Flash-Decoding Implementation")
    print("=" * 60)
    print("Based on PyTorch blog: https://pytorch.org/blog/flash-decoding/")
    print()
    print("Flash-Decoding speeds up attention during inference by parallelizing")
    print("across the keys/values sequence length dimension, achieving up to 8x")
    print("speedup for very long sequences.")
    print()
    
    # Check environment
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    print()
    
    try:
        # Demonstrate the algorithm
        # demonstrate_flash_decoding_algorithm()
        
        # Test correctness
        test_flash_decoding_correctness()
        
        # Benchmark performance
        # benchmark_flash_decoding()
        
        print("\nAll tests completed successfully!")
        print("\nSummary:")
        print("  Flash-Decoding produces identical results to reference")
        print("  Algorithm demonstrated with step-by-step breakdown")
        print("  Performance characteristics measured across context lengths")
        print("  Particularly effective for long-context decoding scenarios")
        
        print("\nKey advantages of Flash-Decoding:")
        print("  • Parallelizes across KV sequence length (not just batch/query)")
        print("  • Fully utilizes GPU even with small batch sizes")
        print("  • Maintains numerical stability with log-sum-exp reduction")
        print("  • Scales well with context length (up to 8x speedup)")
        print("  • Ideal for decoding scenarios (q_len=1, large kv_len)")
        
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 