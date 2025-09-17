#!/usr/bin/env python3
"""
Optimize runtime parameters for llama-simple binary using eval time measurements.
Usage: python tune_tps.py --model /path/to/model.gguf
"""
import os
import time
import argparse
from functools import partial

import numpy as np
# pip install scikit-optimize
from skopt import gp_minimize, expected_minimum
from skopt.plots import plot_objective, plot_convergence
from skopt.space import Categorical
import matplotlib.pyplot as plt
import json

BAD_CONFIGURATIONS = []

# Progress tracking global variables
progress_start_time = None
progress_current_call = 0
progress_total_calls = 0
progress_best_score = float('inf')

def display_progress():
    """Display current optimization progress with time estimates."""
    global progress_start_time, progress_current_call, progress_total_calls, progress_best_score

    if progress_start_time is None:
        return

    elapsed_time = time.time() - progress_start_time
    if progress_current_call > -1:
        avg_time_per_call = elapsed_time / progress_current_call
        remaining_calls = progress_total_calls - progress_current_call
        estimated_remaining_time = avg_time_per_call * remaining_calls

        progress_percent = (progress_current_call / progress_total_calls) * 100

        print(f"\n{'='*60}")
        print(f"OPTIMIZATION PROGRESS")
        print(f"{'='*60}")
        print(f"Iteration: {progress_current_call}/{progress_total_calls} ({progress_percent:.1f}%)")
        print(f"Elapsed time: {elapsed_time:.1f}s")
        print(f"Est. remaining time: {estimated_remaining_time:.1f}s")
        print(f"Best metric so far: {progress_best_score:.4f}")
        print(f"{'='*60}\n")

def run_iterations(get_opts_fn, run_binary_fn, run_options, model_path, binary_path="./build/bin/llama-cli", iterations=1):
    """Run llama-siple with specified options and return eval time."""
    try:
        run_options_str = get_opts_fn(run_options, model_path, binary_path)
        print(run_options_str)

        results = []

        # Run the test (can increase iterations for more stable results)
        for _ in range(iterations):
            results.append(run_binary_fn(run_options_str))

        # Return eval time as the objective (we want to minimize this)
        return np.mean(results)

    except Exception as e:
        BAD_CONFIGURATIONS.append(run_options)
        print("ERROR:", e, run_options)
        print("BAD_CONFIGURATIONS:", BAD_CONFIGURATIONS)
        return 1000  # High penalty for failed runs


def optimize_runtime_with_progress(x, get_opts_fn, run_binary_fn, run_options_list, model_path, llama_simple_path):
    """Objective function for optimization with progress tracking."""
    global progress_current_call, progress_best_score

    progress_current_call += 1

    run_options = {
        run_options_list[i][0]: run_options_list[i][1][run_options_list[i][1].index(x[i])]
        for i in range(len(run_options_list))
    }

    result = run_iterations(get_opts_fn, run_binary_fn, run_options, model_path, llama_simple_path)

    # Update best score
    if result < progress_best_score:
        progress_best_score = result

    # Display progress every call
    display_progress()

    return result


def load_cache(cache_filename):
    """Load cached optimization results."""
    try:
        with open(cache_filename, "r") as cache_file:
            cache_data = json.load(cache_file)
            return cache_data["x0"], cache_data["y0"]
    except:
        pass
    return None, None


def save_cache(cache_filename, x0, y0):
    """Save optimization results to cache."""
    # Convert numpy int64 objects to Python int objects
    x0 = [[int(item) if isinstance(item, np.int64) else item for item in sublist] for sublist in x0]
    y0 = [int(item) if isinstance(item, np.int64) else item for item in y0]

    cache_data = {"x0": x0, "y0": y0}
    with open(cache_filename, "w") as cache_file:
        json.dump(cache_data, cache_file)


def plot_iterations(result):
    """Plot optimization iterations."""
    search_space = result.space
    x_iters = result.x_iters
    func_vals = result.func_vals
    search_space_names = [dim.name for dim in search_space]
    opts = search_space_names + ["objective_r"]

    num_params = len(opts) + 1
    fig, axs = plt.subplots(num_params, figsize=(8, num_params * 8), sharex=True)
    iterations = list(range(1, len(x_iters) + 1))

    for i, param in enumerate(opts):
        if param == "objective_r":
            param_values = func_vals
        else:
            param_index = search_space_names.index(param)
            param_values = [x[param_index] for x in x_iters]

        axs[i].scatter(iterations, param_values)
        axs[i].set_xlabel("Iteration")
        axs[i].set_ylabel(param)

    plot_convergence(result, true_minimum=0, ax=axs[-1])
    return axs

def parse_args(default_bin):
    parser = argparse.ArgumentParser(description='Optimize llama-simple runtime parameters')
    parser.add_argument('--model', '-m', required=True, help='Path to the GGUF model file')
    parser.add_argument('--ngl', type=int, required=True, help='Max number of GPU layers')
    parser.add_argument('--llama-binary', default=default_bin,
                       help='Path to llama-simple binary (default: ./build/bin/llama-simple)')
    parser.add_argument('--n-calls', type=int, default=50,
                       help='Number of optimization calls (default: 20)')
    parser.add_argument('--cache', default='cache_simple.json',
                       help='Cache file name (default: cache_simple.json)')
    parser.add_argument('--single-execution', type=str,
                       help='Run single execution with specified options (format: "--param1=value1 --param2=value2")')

    args = parser.parse_args()
    return args

def main(args, get_opts_fn, run_binary_fn, run_options_list):

    # Check if llama-simple binary exists
    if not os.path.exists(args.llama_binary):
        print(f"Error: llama-simple binary not found at {args.llama_binary}")
        print("Please build llama.cpp first or specify correct path with --llama-binary")
        return

    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
        return

    # Handle single execution mode
    if args.single_execution:
        try:
            print("Single execution")
            run_options = args.single_execution
            run_iterations(get_opts_fn, run_binary_fn, run_options, args.model, args.llama_binary)
            return
        except ValueError as e:
            print(f"Error parsing single execution options: {e}")
            return

    # Initialize progress tracking
    global progress_start_time, progress_total_calls
    progress_start_time = time.time()
    progress_total_calls = args.n_calls

    # Create optimization dimensions
    dimensions = [Categorical(opt[1]) for opt in run_options_list]
    for i, opt in enumerate(run_options_list):
        dimensions[i].name = opt[0]

    # Load cache
    x0, y0 = load_cache(args.cache)

    # Create objective function
    objective_function = partial(optimize_runtime_with_progress,
                               get_opts_fn=get_opts_fn,
                               run_binary_fn=run_binary_fn,
                               run_options_list=run_options_list,
                               model_path=args.model,
                               llama_simple_path=args.llama_binary)

    print(f"Starting optimization with {args.n_calls} calls and {args.ngl} gpu layers...")
    print(f"Using model: {args.model}")
    print(f"Cache file: {args.cache}")

    # Run optimization
    result = gp_minimize(objective_function, dimensions,
                        n_calls=args.n_calls,
                        n_initial_points=min(10, args.n_calls),
                        random_state=42,
                        x0=x0, y0=y0,
                        initial_point_generator="lhs")

    # Save results
    save_cache(args.cache, result.x_iters, result.func_vals)

    # Print results
    print(f"\nBest options found: {result.x}")
    print(f"Minimum eval time: {result.fun:.4f} seconds")

    # Convert result.x back to human-readable format - FIX: Find index of value in options list
    best_options = {}
    for i, (name, options) in enumerate(run_options_list):
        # Find the value in result.x[i] and locate its index in the options list
        value = result.x[i]
        if value in options:
            best_options[name] = value
        else:
            # Fallback: use the first option if value not found
            print(f"Warning: Value '{value}' not found in options for {name}, using first option")
            best_options[name] = options[0]

    print("\nBest configuration:")
    for name, value in best_options.items():
        print(f"  {name}: {value}")

    min_x, _ = expected_minimum(result)
    print(f"Expected minimum: {min_x}")

    if BAD_CONFIGURATIONS:
        print(f"\nBAD_CONFIGURATIONS: {len(BAD_CONFIGURATIONS)}")

    # Plot results
    try:
        plot_iterations(result)
        plot_objective(result)
        # Might need PyQt6
        plt.show()
    except Exception as e:
        print(f"Plotting failed: {e}")
