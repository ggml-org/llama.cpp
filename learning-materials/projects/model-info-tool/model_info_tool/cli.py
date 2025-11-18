#!/usr/bin/env python3
"""
Model Info Tool CLI
Command-line interface for inspecting GGUF models.
"""

import argparse
import sys
from pathlib import Path
from typing import List

from .model_analyzer import ModelAnalyzer
from .exporter import ModelExporter


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_section(title: str):
    """Print a section header."""
    print(f"\n{title}")
    print("-" * 60)


def command_info(args):
    """Display model information."""
    try:
        analyzer = ModelAnalyzer(args.model)

        if args.summary:
            print(analyzer.generate_summary())
            return

        # Basic info
        print_header("MODEL INFORMATION")
        basic = analyzer.get_basic_info()
        print(f"\nName: {basic['name']}")
        print(f"Architecture: {basic['architecture']}")
        print(f"File: {basic['file_name']}")
        print(f"Path: {basic['file_path']}")
        print(f"Size: {basic['file_size_gb']} GB ({basic['file_size_mb']} MB)")
        print(f"Parameters: {basic['parameter_count_formatted']} ({basic['parameter_count']:,})")
        print(f"Tensors: {basic['tensor_count']}")

        # Architecture
        if args.verbose or args.architecture:
            print_section("Architecture Details")
            arch = analyzer.get_architecture_details()
            for key, value in arch.items():
                if key != "architecture":
                    print(f"  {key}: {value}")

        # Tokenizer
        if args.verbose or args.tokenizer:
            print_section("Tokenizer")
            tokenizer = analyzer.get_tokenizer_info()
            for key, value in tokenizer.items():
                if value is not None:
                    print(f"  {key}: {value}")

        # Quantization
        if args.verbose or args.quantization:
            print_section("Quantization")
            quant = analyzer.get_quantization_info()
            print(f"  Predominant Type: {quant['predominant_type']}")
            print(f"  File Type: {quant['file_type']}")
            print(f"  Quantization Version: {quant['quantization_version']}")
            if quant['tensor_types']:
                print(f"\n  Tensor Type Distribution:")
                for qtype, count in sorted(quant['tensor_types'].items()):
                    print(f"    {qtype}: {count}")

        # Memory
        if args.verbose or args.memory:
            print_section("Memory Requirements (Estimated)")
            memory = analyzer.get_memory_requirements()
            print(f"  Model Size: {memory['model_size_gb']} GB")
            print(f"  CPU-only RAM: ~{memory['estimated_ram_cpu_only_gb']} GB")
            print(f"  GPU-full VRAM: ~{memory['estimated_vram_gpu_full_gb']} GB")
            print(f"\n  Note: {memory['note']}")

        # Context
        if args.verbose:
            print_section("Context")
            context = analyzer.get_context_info()
            for key, value in context.items():
                if value is not None:
                    print(f"  {key}: {value}")

        # Tensors
        if args.tensors:
            print_section("Tensor Details")
            tensors = analyzer.get_tensor_details()
            for i, tensor in enumerate(tensors[:10], 1):  # Show first 10
                print(f"\n  {i}. {tensor['name']}")
                print(f"     Type: {tensor['type']}")
                print(f"     Shape: {tensor['shape']}")
                print(f"     Parameters: {tensor['parameters']:,}")
            if len(tensors) > 10:
                print(f"\n  ... and {len(tensors) - 10} more tensors")

        print("\n" + "=" * 60 + "\n")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def command_compare(args):
    """Compare multiple models."""
    try:
        models = args.models
        if len(models) < 2:
            print("Error: Please provide at least 2 models to compare", file=sys.stderr)
            sys.exit(1)

        print_header("MODEL COMPARISON")

        analyzers = []
        for model_path in models:
            try:
                analyzer = ModelAnalyzer(model_path)
                analyzers.append(analyzer)
            except Exception as e:
                print(f"Warning: Could not load {model_path}: {e}", file=sys.stderr)

        if len(analyzers) < 2:
            print("Error: At least 2 valid models required for comparison", file=sys.stderr)
            sys.exit(1)

        # Print comparison table
        print("\n{:<30} {:<15} {:<12} {:<10} {:<15}".format(
            "Model", "Architecture", "Parameters", "Size (GB)", "Quantization"
        ))
        print("-" * 82)

        for analyzer in analyzers:
            basic = analyzer.get_basic_info()
            quant = analyzer.get_quantization_info()
            name = basic['file_name'][:28] if len(basic['file_name']) > 28 else basic['file_name']
            arch = basic['architecture'][:13] if len(basic['architecture']) > 13 else basic['architecture']

            print("{:<30} {:<15} {:<12} {:<10} {:<15}".format(
                name,
                arch,
                basic['parameter_count_formatted'],
                basic['file_size_gb'],
                quant['predominant_type']
            ))

        # Memory comparison
        print_section("Memory Requirements Comparison")
        print("\n{:<30} {:<15} {:<15}".format("Model", "CPU RAM (GB)", "GPU VRAM (GB)"))
        print("-" * 60)

        for analyzer in analyzers:
            basic = analyzer.get_basic_info()
            memory = analyzer.get_memory_requirements()
            name = basic['file_name'][:28] if len(basic['file_name']) > 28 else basic['file_name']

            print("{:<30} {:<15} {:<15}".format(
                name,
                memory['estimated_ram_cpu_only_gb'],
                memory['estimated_vram_gpu_full_gb']
            ))

        print("\n" + "=" * 60 + "\n")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def command_export(args):
    """Export model information."""
    try:
        analyzer = ModelAnalyzer(args.model)
        exporter = ModelExporter(analyzer)

        if args.format == "json":
            output = exporter.to_json(
                output_path=args.output,
                indent=args.indent,
                include_tensors=args.include_tensors
            )
            if not args.output:
                print(output)

        elif args.format == "markdown":
            output = exporter.to_markdown(output_path=args.output)
            if not args.output:
                print(output)

        elif args.format == "csv":
            output = exporter.to_csv_summary(output_path=args.output)
            if not args.output:
                print(output)

        if args.output:
            print(f"\nSuccessfully exported to: {args.output}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def command_list_metadata(args):
    """List all metadata keys and values."""
    try:
        analyzer = ModelAnalyzer(args.model)
        metadata = analyzer.get_all_metadata()

        print_header("ALL METADATA")

        for key in sorted(metadata.keys()):
            value = metadata[key]
            # Truncate long values
            if isinstance(value, (list, dict)):
                value_str = f"{type(value).__name__} with {len(value)} items"
            elif isinstance(value, str) and len(value) > 60:
                value_str = value[:57] + "..."
            else:
                value_str = str(value)

            print(f"{key}: {value_str}")

        print("\n" + "=" * 60 + "\n")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Model Info Tool - Inspect GGUF model files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show model information
  %(prog)s info model.gguf

  # Show verbose information
  %(prog)s info model.gguf -v

  # Show only architecture details
  %(prog)s info model.gguf --architecture

  # Compare multiple models
  %(prog)s compare model1.gguf model2.gguf model3.gguf

  # Export to JSON
  %(prog)s export model.gguf -f json -o model_info.json

  # Export with tensor details
  %(prog)s export model.gguf -f json --include-tensors

  # List all metadata
  %(prog)s metadata model.gguf
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Info command
    info_parser = subparsers.add_parser("info", help="Display model information")
    info_parser.add_argument("model", help="Path to GGUF model file")
    info_parser.add_argument("-v", "--verbose", action="store_true",
                             help="Show all information")
    info_parser.add_argument("-s", "--summary", action="store_true",
                             help="Show summary only")
    info_parser.add_argument("-a", "--architecture", action="store_true",
                             help="Show architecture details")
    info_parser.add_argument("-t", "--tokenizer", action="store_true",
                             help="Show tokenizer information")
    info_parser.add_argument("-q", "--quantization", action="store_true",
                             help="Show quantization information")
    info_parser.add_argument("-m", "--memory", action="store_true",
                             help="Show memory requirements")
    info_parser.add_argument("--tensors", action="store_true",
                             help="Show tensor details")

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare multiple models")
    compare_parser.add_argument("models", nargs="+", help="Paths to GGUF model files")

    # Export command
    export_parser = subparsers.add_parser("export", help="Export model information")
    export_parser.add_argument("model", help="Path to GGUF model file")
    export_parser.add_argument("-f", "--format", choices=["json", "markdown", "csv"],
                               default="json", help="Export format")
    export_parser.add_argument("-o", "--output", help="Output file path")
    export_parser.add_argument("-i", "--indent", type=int, default=2,
                               help="JSON indentation level")
    export_parser.add_argument("--include-tensors", action="store_true",
                               help="Include tensor details in JSON export")

    # Metadata command
    metadata_parser = subparsers.add_parser("metadata", help="List all metadata")
    metadata_parser.add_argument("model", help="Path to GGUF model file")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Route to appropriate command
    if args.command == "info":
        command_info(args)
    elif args.command == "compare":
        command_compare(args)
    elif args.command == "export":
        command_export(args)
    elif args.command == "metadata":
        command_list_metadata(args)


if __name__ == "__main__":
    main()
