#!/usr/bin/env python3
"""
Batch Processing with llama.cpp

This example demonstrates:
- Processing multiple prompts efficiently
- Parallel processing with threading
- Progress tracking
- CSV input/output
"""

from llama_cpp import Llama
from typing import List, Dict
import csv
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time


class BatchProcessor:
    """Batch processing for llama.cpp."""

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_gpu_layers: int = 35,
        max_workers: int = 4
    ):
        """
        Initialize batch processor.

        Args:
            model_path: Path to GGUF model
            n_ctx: Context window size
            n_gpu_layers: GPU layers to offload
            max_workers: Number of parallel workers
        """
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.max_workers = max_workers

        # Initialize model instances for each worker
        print(f"Initializing {max_workers} model instances...")
        self.models: List[Llama] = []

        for i in range(max_workers):
            llm = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers // max_workers,  # Distribute GPU layers
                verbose=False
            )
            self.models.append(llm)

        print("Models initialized!")

    def process_single(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        model_idx: int = 0
    ) -> Dict[str, any]:
        """
        Process a single prompt.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            model_idx: Model instance index

        Returns:
            Result dictionary
        """
        start_time = time.time()

        try:
            llm = self.models[model_idx % len(self.models)]

            response = llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["\n\n"]
            )

            output = response['choices'][0]['text'].strip()
            elapsed = time.time() - start_time

            return {
                "prompt": prompt,
                "output": output,
                "success": True,
                "elapsed_time": elapsed,
                "tokens": len(llm.tokenize(output.encode('utf-8')))
            }

        except Exception as e:
            return {
                "prompt": prompt,
                "output": None,
                "success": False,
                "error": str(e),
                "elapsed_time": time.time() - start_time
            }

    def process_batch(
        self,
        prompts: List[str],
        max_tokens: int = 256,
        temperature: float = 0.7,
        show_progress: bool = True
    ) -> List[Dict[str, any]]:
        """
        Process multiple prompts in parallel.

        Args:
            prompts: List of prompts
            max_tokens: Maximum tokens per generation
            temperature: Sampling temperature
            show_progress: Show progress bar

        Returns:
            List of results
        """
        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(
                    self.process_single,
                    prompt,
                    max_tokens,
                    temperature,
                    i
                ): i
                for i, prompt in enumerate(prompts)
            }

            # Collect results with progress bar
            if show_progress:
                pbar = tqdm(total=len(prompts), desc="Processing")

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

                if show_progress:
                    pbar.update(1)

            if show_progress:
                pbar.close()

        # Sort results by original order
        results.sort(key=lambda x: prompts.index(x['prompt']))

        return results

    def process_csv(
        self,
        input_file: str,
        output_file: str,
        prompt_column: str = 'prompt',
        max_tokens: int = 256
    ):
        """
        Process prompts from CSV file.

        Args:
            input_file: Input CSV file path
            output_file: Output CSV file path
            prompt_column: Column name containing prompts
            max_tokens: Maximum tokens per generation
        """
        print(f"Reading prompts from {input_file}...")

        # Read input CSV
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if prompt_column not in rows[0]:
            raise ValueError(f"Column '{prompt_column}' not found in CSV")

        prompts = [row[prompt_column] for row in rows]

        print(f"Processing {len(prompts)} prompts...")

        # Process batch
        results = self.process_batch(prompts, max_tokens=max_tokens)

        # Write output CSV
        print(f"Writing results to {output_file}...")

        fieldnames = list(rows[0].keys()) + ['output', 'success', 'elapsed_time']

        with open(output_file, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for row, result in zip(rows, results):
                row['output'] = result['output']
                row['success'] = result['success']
                row['elapsed_time'] = f"{result['elapsed_time']:.2f}s"

                if not result['success']:
                    row['output'] = f"Error: {result.get('error', 'Unknown')}"

                writer.writerow(row)

        print(f"Done! Results saved to {output_file}")

        # Print statistics
        successful = sum(1 for r in results if r['success'])
        total_time = sum(r['elapsed_time'] for r in results)
        avg_time = total_time / len(results)

        print(f"\nStatistics:")
        print(f"  Total prompts: {len(prompts)}")
        print(f"  Successful: {successful}")
        print(f"  Failed: {len(prompts) - successful}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average time per prompt: {avg_time:.2f}s")

    def process_jsonl(
        self,
        input_file: str,
        output_file: str,
        prompt_field: str = 'prompt',
        max_tokens: int = 256
    ):
        """
        Process prompts from JSONL file.

        Args:
            input_file: Input JSONL file path
            output_file: Output JSONL file path
            prompt_field: Field name containing prompts
            max_tokens: Maximum tokens per generation
        """
        print(f"Reading prompts from {input_file}...")

        # Read input JSONL
        rows = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                row = json.loads(line)
                rows.append(row)

        prompts = [row[prompt_field] for row in rows]

        print(f"Processing {len(prompts)} prompts...")

        # Process batch
        results = self.process_batch(prompts, max_tokens=max_tokens)

        # Write output JSONL
        print(f"Writing results to {output_file}...")

        with open(output_file, 'w', encoding='utf-8') as f:
            for row, result in zip(rows, results):
                row['output'] = result['output']
                row['success'] = result['success']
                row['elapsed_time'] = result['elapsed_time']

                if not result['success']:
                    row['error'] = result.get('error', 'Unknown')

                f.write(json.dumps(row) + '\n')

        print(f"Done! Results saved to {output_file}")


def create_sample_csv(output_file: str):
    """Create sample CSV for testing."""
    sample_data = [
        {"id": 1, "prompt": "What is Python?", "category": "programming"},
        {"id": 2, "prompt": "Explain machine learning", "category": "AI"},
        {"id": 3, "prompt": "What is a database?", "category": "data"},
        {"id": 4, "prompt": "Define cloud computing", "category": "infrastructure"},
        {"id": 5, "prompt": "What is Docker?", "category": "devops"},
    ]

    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'prompt', 'category'])
        writer.writeheader()
        writer.writerows(sample_data)

    print(f"Sample CSV created: {output_file}")


def main():
    """Example usage."""
    import argparse

    parser = argparse.ArgumentParser(description='Batch processing with llama.cpp')
    parser.add_argument('model_path', help='Path to GGUF model')
    parser.add_argument('--input', help='Input CSV or JSONL file')
    parser.add_argument('--output', help='Output file')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--max-tokens', type=int, default=256, help='Max tokens')
    parser.add_argument('--create-sample', action='store_true', help='Create sample CSV')

    args = parser.parse_args()

    if args.create_sample:
        create_sample_csv('sample_prompts.csv')
        return

    if not args.input or not args.output:
        print("Error: --input and --output are required")
        print("\nExample usage:")
        print("  # Create sample CSV")
        print("  python 05_batch_processing.py model.gguf --create-sample")
        print("\n  # Process CSV")
        print("  python 05_batch_processing.py model.gguf \\")
        print("    --input sample_prompts.csv \\")
        print("    --output results.csv \\")
        print("    --workers 4 \\")
        print("    --max-tokens 256")
        return

    # Initialize processor
    processor = BatchProcessor(
        model_path=args.model_path,
        max_workers=args.workers
    )

    # Determine file type and process
    input_path = Path(args.input)

    if input_path.suffix == '.csv':
        processor.process_csv(
            input_file=args.input,
            output_file=args.output,
            max_tokens=args.max_tokens
        )
    elif input_path.suffix in ['.jsonl', '.json']:
        processor.process_jsonl(
            input_file=args.input,
            output_file=args.output,
            max_tokens=args.max_tokens
        )
    else:
        print(f"Unsupported file type: {input_path.suffix}")
        print("Supported types: .csv, .jsonl")


if __name__ == '__main__':
    main()
