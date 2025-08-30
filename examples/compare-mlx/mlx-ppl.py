# Copyright © 2025 Apple Inc.
# modified: https://github.com/ml-explore/mlx-lm/blob/60320dc2347d45dc3ca08be90e5255fb9424bb09/mlx_lm/perplexity.py
"""
Evaluate perplexity (PPL) of pre-trained MLX models in the same way as llama.cpp's llama-perplexity.
"""

import argparse
import math
import os
import time
import types

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_lm.tuner.datasets import load_dataset
from mlx_lm.tuner.utils import get_total_parameters
from mlx_lm.utils import load


def load_data(
    tokenizer,
    data_path: str,
    num_samples: int,
    sequence_length: int,
):
    """
    Load a Hugging‑Face dataset (via mlx‑lm’s dataset utilities) and convert it
    into a token tensor of shape (N, sequence_length).
    """
    args = types.SimpleNamespace(
        hf_dataset={
            "path": data_path,
            "train_split": "train",
            "valid_split": "train[:1]",
        },
        train=True,
        test=False,
    )
    dataset = load_dataset(args, tokenizer)[0]

    perm = np.random.permutation(len(dataset)).tolist()

    num_tokens = sequence_length * num_samples if num_samples > 0 else float("inf")
    data = []
    i = 0
    while len(data) < num_tokens:
        tokens, _ = dataset.process(dataset[perm[i]])
        i += 1
        data.extend(tokens)

    # Convert to MX array, truncate to a multiple of `sequence_length`
    data = mx.array(data[: (len(data) // sequence_length) * sequence_length])
    data = data.reshape(-1, sequence_length)
    if num_samples > 0:
        data = data[:num_samples]
    return data


def _tokenize_text(tokenizer, text: str):
    """
    Helper that tokenises a string using the MLX‑LM tokenizer.
    Supports the common `encode` method or a callable tokenizer.
    """
    # Most mlx‑lm tokenizers expose an `encode` method.
    if hasattr(tokenizer, "encode"):
        tokens = tokenizer.encode(text)
    elif callable(tokenizer):
        tokens = tokenizer(text)
    else:
        raise AttributeError(
            "Tokenizer does not have an `encode` method nor is it callable."
        )
    # Normalise the output to a Python list of ints.
    if isinstance(tokens, mx.array):
        tokens = tokens.tolist()
    return tokens


# load a raw text file and tokenize it
# generated with gpt-oss-120b
def load_raw_data(
    tokenizer,
    raw_path: str,
    num_samples: int,
    sequence_length: int,
):
    """
    Load a raw text file, tokenize it, and reshape into a (N, sequence_length)
    tensor suitable for perplexity evaluation.
    """
    if not os.path.isfile(raw_path):
        raise FileNotFoundError(f"Raw text file not found: {raw_path}")

    # Read the whole file (UTF‑8).  Users can supply any plain‑text corpus.
    with open(raw_path, "r", encoding="utf-8") as fp:
        raw_text = fp.read()

    # Tokenise the complete text.
    token_list = _tokenize_text(tokenizer, raw_text)

    if len(token_list) == 0:
        raise ValueError("Tokenisation of the raw file produced no tokens.")

    # Convert to MX array (int32 is sufficient for token IDs).
    token_array = mx.array(token_list, dtype=mx.int32)

    # Trim to a length that is an exact multiple of `sequence_length`.
    total_len = (token_array.shape[0] // sequence_length) * sequence_length
    token_array = token_array[:total_len]

    # Reshape into (num_sequences, sequence_length)
    data = token_array.reshape(-1, sequence_length)

    if num_samples > 0:
        data = data[:num_samples]

    #print(f"First 4 samples of the data:")
    #for j in range(min(4, len(data))):
    #    print(f"  Sample {j}: {tokenizer.decode(data[j].tolist())}\n\n-------------------\n\n")

    return data


def eval_ppl(model, tokenizer, data, batch_size=8):
    """
    Evaluate perplexity on a dataset with standard error calculation.

    Args:
        model: The model to evaluate.
        data: Tokenized data tensor (shape: N x L).
        batch_size: Batch size for evaluation.

    Returns:
        tuple: (perplexity, standard_error_of_perplexity)
    """
    all_losses = []

    num_batches = (len(data) + batch_size - 1) // batch_size
    for i, s in enumerate(range(0, len(data), batch_size)):
        batch = data[s : s + batch_size]

        # Set the first token of all samples to the BOS token
        if tokenizer.bos_token_id:
            batch[:, 0] = tokenizer.bos_token_id

        # compute cross entropy only with the second half of the sequence to match llama.cpp behavior
        # ref: https://github.com/ggml-org/llama.cpp/blob/696fccf354e9dbdfbce135bc40b44c9dcc64dda9/tools/perplexity/perplexity.cpp#L527-L541
        #
        #start = 0
        start = batch.shape[1] // 2

        # Forward pass: get logits for all tokens except last
        logits = model(batch[:, :-1]).astype(mx.float32)

        # Calculate cross‑entropy loss with next tokens
        #losses = nn.losses.cross_entropy(logits, batch[:, 1:], reduction="none")
        losses = nn.losses.cross_entropy(logits[:, start:, :], batch[:, start+1:], reduction="none")

        mx.eval(losses)
        # Store individual token losses
        all_losses.append(losses.flatten())

        # Progress indicator
        if (i + 1) % 1 == 0 or (i + 1) == num_batches:
            print(f"  Processed {i + 1}/{num_batches} batches...", end="\r")

    print()  # New line after progress

    # Concatenate all losses into a single array
    all_losses = mx.concatenate(all_losses)

    # Calculate mean loss and perplexity
    mean_loss = all_losses.mean().item()
    ppl = math.exp(mean_loss)

    # Calculate standard error
    std_dev = mx.sqrt(mx.var(all_losses, ddof=1)).item()
    num_tokens = all_losses.size
    standard_error = std_dev / math.sqrt(num_tokens)

    # Delta approximation for standard error of perplexity
    standard_error_ppl = ppl * standard_error

    return ppl, standard_error_ppl


def main():
    parser = argparse.ArgumentParser(description="Evaluate perplexity of MLX models")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model or Hugging Face model ID",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=512,
        help="Sequence length for evaluation",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=256,
        help="Number of samples to use (-1 for all available)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="allenai/tulu-3-sft-mixture",
        help=(
            "A Hugging Face dataset compatible with mlx‑lm. "
            "Ignored if --raw-path is provided."
        ),
    )
    parser.add_argument(
        "--raw-path",
        type=str,
        default=None,
        help=(
            "Path to a local raw‑text file to use for evaluation. "
            "If specified, the script skips loading a HF dataset."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=123, help="Random seed for data sampling"
    )

    args = parser.parse_args()

    # Set random seed (used for HF dataset shuffling)
    mx.random.seed(args.seed)

    # Load model
    print(f"Loading model from {args.model}...")
    model, tokenizer = load(args.model)

    # Count parameters
    total_params = get_total_parameters(model)
    print(f"Model loaded: {total_params/1e6:.1f}M parameters")

    # ----------------------------------------------------------------------
    # Load evaluation data (raw file vs. HF dataset)
    # ----------------------------------------------------------------------
    print("\nLoading dataset...")
    print(f"  Sequence length: {args.sequence_length}")

    if args.raw_path:
        print(f"  Using raw text file: {args.raw_path}")
        data = load_raw_data(
            tokenizer,
            raw_path=args.raw_path,
            num_samples=args.num_samples,
            sequence_length=args.sequence_length,
        )
    else:
        print(f"  Using HF dataset: {args.data_path}")
        data = load_data(
            tokenizer,
            data_path=args.data_path,
            num_samples=args.num_samples,
            sequence_length=args.sequence_length,
        )

    print(f"  Loaded {len(data)} samples")

    # ----------------------------------------------------------------------
    # Evaluate perplexity
    # ----------------------------------------------------------------------
    print(f"\nEvaluating perplexity with batch size {args.batch_size}...")
    start_time = time.time()

    ppl, se = eval_ppl(model, tokenizer, data, batch_size=args.batch_size)

    eval_time = time.time() - start_time
    tokens_evaluated = data.shape[0] * (data.shape[1] - 1)  # B * (L - 1)

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Perplexity: {ppl:.3f} ± {se:.3f}")
    print(f"Evaluation time: {eval_time:.2f} seconds")
    print(f"Peak memory: {mx.get_peak_memory() / 1e9:.2f} GB")
    print(f"Tokens per second: {tokens_evaluated / eval_time:.0f}")

    # Additional statistics
    print(f"\nDataset statistics:")
    print(f"  Total samples: {len(data)}")
    print(f"  Total tokens: {data.size}")

    # ----------------------------------------------------------------------
    # Done
    # ----------------------------------------------------------------------


if __name__ == "__main__":
    main()

