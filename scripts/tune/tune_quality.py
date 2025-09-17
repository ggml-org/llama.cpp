#!/usr/bin/env python3
"""
BERTScore-based translation quality optimization for llama.cpp models.
Uses BERTScore to evaluate translation quality instead of HellaSwag accuracy.
"""
import subprocess
import sys
import os
import re
import json
import hashlib
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter

# Import bert_score for translation quality evaluation
import bert_score

# Import language_tool_python for grammar checking
import language_tool_python

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from tune import parse_args, main

# Configuration
BERTSCORE_MODEL = 'microsoft/deberta-v3-base'

# Translation benchmarks for quality evaluation
# Tiny subset of https://openslr.org/100
TRANSLATION_BENCHMARKS = [
    {
        "prompt": "Translate the following English text to French:\n\nEnglish: As you can see, it does not look like a slam lesson, it is a language lesson, a language which allows to give orders to machines and computers the language of the 21st century: the computer code.\nFrench:",
        "ground_truth": "Comme vous pouvez le constater, il ne s'agit pas d'un cours de slam, il s'agit d'un cours de langue, une langue qui permet de donner des ordres à des machines et à des ordinateurs, la langue du 21e siècle : le code informatique.",
        "tool": "fr-FR"
    },
    {
        "prompt": "Translate the following English text to Spanish:\n\nEnglish: Some years ago, when I was diving in the Lombok Strait, in Indonesia, 98 feet below the water, with that feeling of weightlessness, surrounded by a great biodiversity of reefs, corals, sea turtles, ocean sunfishes and fishes of all colors, I had an intense feeling of connection with nature.\nSpanish:",
        "ground_truth": "Hace unos años, cuando me encontraba buceando en el estrecho de Lombok, en Indonesia, a 30 metros debajo del agua, con esa sensación de ingravidez, rodeado de una gran biodiversidad, de arrecifes, de corales, de tortugas, de peces mola mola y de peces de todos los colores, tuve una intensa sensación de estar conectado con la naturaleza.",
        "tool": "es-ES"
    },
    {
        "prompt": "Translate the following English text to Portuguese:\n\nEnglish: Have you ever stopped to think about clothes for disabled people?\nPortuguese:",
        "ground_truth": "Vocês já pararam pra pensar como é o vestuário das pessoas com deficiência?",
        "tool": "pt-PT"
    }
]

def get_metrics(metrics_filepath: str, ground_truth: str, prediction: str, tool: str) -> Dict[str, Any]:
    """
    Calculate BERTScore and other quality metrics for translation evaluation.
    Caches results to avoid recomputation.
    """
    print(f"Calculating metrics: {metrics_filepath}")

    metrics = {
        'bertscore_model': None,
        'bertscore_P': None,
        'bertscore_R': None,
        'bertscore_F1': None,
        'grammar_errors': None,
        'repetition_score': None,
        'objective_r': None
    }

    # Load cached scores
    try:
        with open(metrics_filepath, 'r', encoding='utf-8') as f:
            metrics.update(json.load(f))
    except FileNotFoundError:
        pass

    # Calculate BERTScore if not cached or model changed
    if (not metrics["bertscore_P"] or not metrics["bertscore_R"] or
        not metrics["bertscore_F1"] or metrics["bertscore_model"] != BERTSCORE_MODEL):
        try:
            metrics["bertscore_model"] = BERTSCORE_MODEL
            score = bert_score.score([prediction], [ground_truth], model_type=BERTSCORE_MODEL)
            metrics["bertscore_P"], metrics["bertscore_R"], metrics["bertscore_F1"] = (
                score[0].item(), score[1].item(), score[2].item()
            )
        except Exception as e:
            print(f"Warning: BERTScore calculation failed: {e}")
            metrics["bertscore_P"] = metrics["bertscore_R"] = metrics["bertscore_F1"] = 0.0

    # Calculate grammar errors if not cached
    if metrics["grammar_errors"] is None:
        metrics["grammar_errors"] = 0.0

    language_tool = language_tool_python.LanguageTool(tool)
    try:
        matches = language_tool.check(prediction)
        metrics["grammar_errors"] = len(matches) / max(len(prediction.split()), 1)
    except Exception as e:
        print(f"Warning: Grammar checking failed: {e}")
        metrics["grammar_errors"] = 0.0

    # Calculate repetition score if not cached
    if metrics["repetition_score"] is None:
        try:
            words = prediction.split()
            if len(words) > 0:
                word_counts = Counter(words)
                repeated_words = sum(count - 1 for count in word_counts.values() if count > 1)
                metrics["repetition_score"] = repeated_words / len(words)
            else:
                metrics["repetition_score"] = 0.0
        except Exception as e:
            print(f"Warning: Repetition calculation failed: {e}")
            metrics["repetition_score"] = 0.0

    # Calculate objective score (we want to minimize this)
    # Higher BERTScore Recall = better translation quality = lower objective value
    # Add penalties for grammar errors and repetitions
    if metrics["bertscore_R"] is not None:
        grammar_penalty = metrics["grammar_errors"] * 0.1  # Small penalty for grammar errors
        repetition_penalty = metrics["repetition_score"] * 0.05  # Small penalty for repetitions
        metrics["objective_r"] = -(metrics["bertscore_R"] - grammar_penalty - repetition_penalty)
    else:
        metrics["objective_r"] = 1.0  # Bad score if BERTScore failed

    # Save metrics to cache
    try:
        with open(metrics_filepath, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Warning: Failed to save metrics: {e}")

    return metrics

def run_binary(run_options_str):
    """Run the binary and evaluate translation quality using BERTScore."""
    try:
        # Parse the command to extract parameters
        parts = run_options_str.split()
        model_path = None
        binary_path = None

        # Find model path and binary path
        for i, part in enumerate(parts):
            if part == "-m" and i + 1 < len(parts):
                model_path = parts[i + 1]
            elif part.endswith("llama-cli") or part.endswith("main"):
                binary_path = part

        if not model_path or not binary_path:
            print("Error: Could not parse model path or binary path from command")
            return 100.0

        # Create output directory for this run
        run_hash = hashlib.md5(run_options_str.encode()).hexdigest()[:8]
        output_dir = f"translation_eval_{run_hash}"
        os.makedirs(output_dir, exist_ok=True)

        all_scores = []

        # Run translation benchmarks
        for i, benchmark in enumerate(TRANSLATION_BENCHMARKS):
            print(f"Running benchmark {i+1}/{len(TRANSLATION_BENCHMARKS)}")

            # Build command for this benchmark - use the base command and add benchmark-specific params
            benchmark_cmd = run_options_str.split()

            # Add benchmark-specific parameters
            benchmark_cmd.extend(["--prompt", benchmark["prompt"]])

            # Run the command
            try:
                process = subprocess.run(benchmark_cmd,
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE,
                                       timeout=120,  # 2 minute timeout per benchmark
                                       check=False)

                if process.returncode != 0:
                    print(f"Warning: Benchmark {i+1} failed with return code {process.returncode}")
                    print(f"STDERR: {process.stderr.decode()}")
                    all_scores.append(1.0)  # Bad score for failed runs
                    continue

                # Extract prediction from output
                output = process.stdout.decode()
                prediction = output.strip()

                # Remove the prompt from prediction if it's included
                if benchmark["prompt"] in prediction:
                    prediction = prediction.split(benchmark["prompt"])[-1].strip()

                # Calculate metrics
                metrics_filepath = os.path.join(output_dir, f"benchmark_{i}_metrics.json")
                metrics = get_metrics(metrics_filepath,
                                    benchmark["ground_truth"], prediction, benchmark["tool"])

                objective_score = metrics.get("objective_r", 1.0)
                all_scores.append(objective_score)

                print(f"Benchmark {i+1} - BERTScore R: {metrics.get('bertscore_R', 0):.4f}, "
                      f"Objective: {objective_score:.4f}")

            except subprocess.TimeoutExpired:
                print(f"Warning: Benchmark {i+1} timed out")
                all_scores.append(1.0)  # Bad score for timeouts
            except Exception as e:
                print(f"Error running benchmark {i+1}: {e}")
                all_scores.append(1.0)  # Bad score for errors

        # Calculate average score across all benchmarks
        if all_scores:
            avg_score = np.mean(all_scores)
            print(f"Average translation quality objective score: {avg_score:.4f}")
            return avg_score
        else:
            print("Warning: No successful benchmarks")
            return 100.0  # Bad score if no benchmarks succeeded

    except Exception as e:
        print(f"Error in run_binary: {e}")
        return 100.0  # Bad score for any other errors

if __name__ == "__main__":
    args = parse_args(default_bin='./build/bin/llama-cli')

    # Define quality-focused sampling parameters for optimization
    run_options_list = [
        # Core Sampling Parameters (Most Critical for Quality)

        # 1. Temperature - Controls randomness vs determinism
        ("--temp", [
            "--temp 0.1",   # Very focused, deterministic
            "--temp 0.3",   # Focused, good for factual tasks
            "--temp 0.5",   # Moderate creativity
            "--temp 0.7",   # Balanced (recommended default)
            "--temp 0.8",   # Good balance
            "--temp 0.9",   # More creative
            "--temp 1.0",   # Creative but coherent
            "--temp 1.2"    # More creative, potentially less coherent
        ]),

        # 2. Top-p (Nucleus Sampling) - Controls diversity while maintaining quality
        ("--top-p", [
            "--top-p 0.5",   # Very focused
            "--top-p 0.7",   # Focused, higher quality
            "--top-p 0.8",   # Good balance
            "--top-p 0.85",  # Balanced
            "--top-p 0.9",   # Good balance (recommended)
            "--top-p 0.95",  # Standard default
            "--top-p 0.98",  # More diverse
            "--top-p 1.0"    # No nucleus filtering
        ]),

        # 3. Top-k - Limits token selection to most probable candidates
        ("--top-k", [
            "--top-k 10",   # Very focused
            "--top-k 20",   # More focused, higher quality
            "--top-k 30",   # Balanced
            "--top-k 40",   # Good balance (default)
            "--top-k 50",   # Balanced, more diverse
            "--top-k 60",   # More diverse
            "--top-k 80",   # Very diverse
            "--top-k 100"   # Most diverse
        ]),

        # 4. Min-p - Filters out low-probability tokens
        ("--min-p", [
            "--min-p 0.01",  # Very permissive
            "--min-p 0.02",  # Permissive
            "--min-p 0.05",  # Good default
            "--min-p 0.08",  # More restrictive
            "--min-p 0.1",   # Restrictive, higher quality
            "--min-p 0.15",  # Very restrictive
            "--min-p 0.2"    # Extremely restrictive
        ]),

        # Repetition Control (Critical for Coherence)

        # 5. Repeat Penalty - Prevents repetitive text
        ("--repeat-penalty", [
            "--repeat-penalty 1.0",   # Disabled
            "--repeat-penalty 1.02",  # Very light penalty
            "--repeat-penalty 1.05",  # Light penalty (recommended)
            "--repeat-penalty 1.1",   # Moderate penalty (recommended)
            "--repeat-penalty 1.15",  # Moderate-strong penalty
            "--repeat-penalty 1.2",   # Strong penalty
            "--repeat-penalty 1.25",  # Very strong penalty
            "--repeat-penalty 1.3"    # Extreme penalty
        ]),

        # 6. Repeat Last N - How far back to look for repetitions
        ("--repeat-last-n", [
            "--repeat-last-n 16",   # Short context
            "--repeat-last-n 32",   # Short-medium context
            "--repeat-last-n 64",   # Balanced default
            "--repeat-last-n 96",   # Medium-large context
            "--repeat-last-n 128",  # Large context
            "--repeat-last-n 192",  # Very large context
            "--repeat-last-n 256"   # Maximum context
        ]),

        # Advanced Quality Parameters

        # 7. Typical-p - Promotes contextually coherent tokens
        ("--typical", [
            "--typical 1.0",   # Disabled
            "--typical 0.95",  # Light filtering
            "--typical 0.9",   # Recommended for quality
            "--typical 0.85",  # Moderate filtering
            "--typical 0.8",   # Strong filtering
            "--typical 0.75",  # Very strong filtering
            "--typical 0.7"    # Extreme filtering
        ]),

        # 8. Mirostat - Adaptive sampling for consistent quality
        ("--mirostat", [
            "--mirostat 0",  # Disabled (default)
            "--mirostat 1",  # Mirostat v1
            "--mirostat 2"   # Mirostat v2 (often better quality)
        ]),

        # Keep seed constant for reproducible results
        ("--seed", ["-s 42"]),
    ]

    def run_str(run_options, model_path, binary_path):
        """Build command string for llama-cli with translation evaluation."""
        if isinstance(run_options, dict):
            run_options = " ".join(run_options.values())
        # Use the main binary for translation evaluation
        return f"{binary_path} -m {model_path} --threads 8 -ngl {args.ngl} {run_options}"

    main(args, run_str, run_binary, run_options_list)
