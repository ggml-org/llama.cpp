#!/usr/bin/env python3

import argparse
import sys
from common import compare_tokens  # type: ignore


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Compare tokens between two models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s pytorch-gemma-3-270m-it llamacpp-gemma-3-270m-it-bf16
        """
    )
    parser.add_argument(
        'original',
        help='Original model name'
    )
    parser.add_argument(
        'converted',
        help='Converted model name'
    )
    parser.add_argument(
        '-s', '--suffix',
        default='',
        help='Type suffix (e.g., "-embeddings")'
    )
    parser.add_argument(
        '-d', '--data-dir',
        default='data',
        help='Directory containing token files (default: data)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Print prompts from both models'
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.verbose:
        from pathlib import Path
        data_dir = Path(args.data_dir)

        prompt1_file = data_dir / f"{args.original}{args.suffix}-prompt.txt"
        prompt2_file = data_dir / f"{args.converted}{args.suffix}-prompt.txt"

        if prompt1_file.exists():
            print(f"\nOriginal model prompt ({args.original}):")
            print(f"  {prompt1_file.read_text().strip()}")

        if prompt2_file.exists():
            print(f"\nConverted model prompt ({args.converted}):")
            print(f"  {prompt2_file.read_text().strip()}")

        print()

    result = compare_tokens(
        args.original,
        args.converted,
        type_suffix=args.suffix,
        output_dir=args.data_dir
    )

    # Enable the script to be used in shell scripts so that they can check
    # the exit code for success/failure.
    sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()
