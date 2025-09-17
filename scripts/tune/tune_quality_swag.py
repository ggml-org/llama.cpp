import subprocess
import sys
import os
import re

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from tune import parse_args, main

def run_binary(run_options_str):
    """Run the binary and parse HellaSwag accuracy score."""
    try:
        process = subprocess.run(run_options_str,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,
                                 shell=True,
                                 check=False,  # Don't raise exception on non-zero exit
                                 timeout=300   # 5 minute timeout
                                 )

        if process.returncode != 0:
            print(f"Warning: Process returned non-zero exit code: {process.returncode}")
            print(f"STDERR: {process.stderr.decode()}")
            return 100.0  # Return bad score for failed runs

        # Parse HellaSwag accuracy from stdout
        stdout_text = process.stdout.decode()
        stderr_text = process.stderr.decode()

        # Look for HellaSwag accuracy patterns in output
        # Pattern for format: "20      75.00000000%    [53.1299%, 88.8138%]"
        accuracy_patterns = [
            r"20\t([\d.]+)%\t\[",
        ]

        accuracy = None
        for pattern in accuracy_patterns:
            match = re.search(pattern, stdout_text, re.IGNORECASE)
            if match:
                accuracy = float(match.group(1))
                # Convert percentage to decimal if needed (values > 1.0 are likely percentages)
                if accuracy > 1.0:
                    accuracy = accuracy / 100.0
                break

        if accuracy is None:
            print("Warning: Could not parse HellaSwag accuracy from output")
            print("STDOUT:", stdout_text[:500])  # Show first 500 chars
            print("STDERR:", stderr_text[:500])
            return 100.0  # Return bad score for unparseable results
        else:
            print(f"HellaSwag accuracy: {accuracy:.4f}")

        # Return negative accuracy since we want to MINIMIZE the objective function
        # (higher accuracy = lower objective value = better)
        return -accuracy

    except subprocess.TimeoutExpired:
        print("Warning: Process timed out")
        return 100.0  # Return bad score for timeouts
    except Exception as e:
        print(f"Error running command: {e}")
        return 100.0  # Return bad score for other errors

if __name__ == "__main__":
    args = parse_args(default_bin='./build/bin/llama-perplexity')

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
        """Build command string for llama-perplexity with hellaswag evaluation."""
        run_opts = " ".join(run_options.values())
        # Use the perplexity command with hellaswag evaluation as specified
        return f"{binary_path} -m {model_path} -f hellaswag_val_full.txt --hellaswag-tasks 20 --hellaswag -ngl {args.ngl} {run_opts}"
    main(args, run_str, run_binary, run_options_list)
