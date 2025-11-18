#!/usr/bin/env python3
"""
Format Converter Tool

Batch convert models to multiple quantization formats.
Useful for preparing models for distribution or testing.

Usage:
    python format_converter.py --model model.gguf --formats Q4_K_M Q5_K_M Q8_0
    python format_converter.py --model model.gguf --preset mobile
    python format_converter.py --model model.gguf --all
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Dict
import json
import hashlib


# Format presets for common use cases
PRESETS = {
    "mobile": ["Q4_0", "Q4_K_S", "Q4_K_M"],
    "balanced": ["Q4_K_M", "Q5_K_M", "Q6_K"],
    "quality": ["Q5_K_M", "Q6_K", "Q8_0"],
    "experimental": ["Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L"],
    "all_4bit": ["Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M"],
    "all_5bit": ["Q5_0", "Q5_1", "Q5_K_S", "Q5_K_M"],
    "production": ["Q4_K_M", "Q5_K_M", "Q8_0"],
}

# All available quantization formats
ALL_FORMATS = [
    "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L",
    "Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M",
    "Q5_0", "Q5_1", "Q5_K_S", "Q5_K_M",
    "Q6_K", "Q8_0"
]


class FormatConverter:
    """Batch convert models to multiple formats"""

    def __init__(self, llama_cpp_dir: str = "../../../"):
        self.llama_cpp_dir = Path(llama_cpp_dir)
        self.quantize_tool = self.llama_cpp_dir / "llama-quantize"

        if not self.quantize_tool.exists():
            raise FileNotFoundError(f"llama-quantize not found at {self.quantize_tool}")

        self.conversion_log = []

    def calculate_md5(self, file_path: Path) -> str:
        """Calculate MD5 hash of file"""
        md5 = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                md5.update(chunk)
        return md5.hexdigest()

    def get_file_info(self, file_path: Path) -> Dict:
        """Get file metadata"""
        stat = file_path.stat()
        return {
            'size_bytes': stat.st_size,
            'size_mb': stat.st_size / (1024 * 1024),
            'size_gb': stat.st_size / (1024 * 1024 * 1024),
            'md5': self.calculate_md5(file_path)
        }

    def convert(self, input_model: Path, format: str,
                output_dir: Path = None,
                verify: bool = True,
                importance_matrix: Path = None) -> Dict:
        """Convert model to specified format"""

        # Determine output path
        if output_dir is None:
            output_dir = input_model.parent
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        output_name = f"{input_model.stem}-{format.lower()}.gguf"
        output_path = output_dir / output_name

        print(f"\n{'='*60}")
        print(f"Converting to {format}")
        print(f"{'='*60}")
        print(f"Input:  {input_model}")
        print(f"Output: {output_path}")

        # Check if already exists
        if output_path.exists():
            response = input(f"Output file exists. Overwrite? [y/N]: ")
            if response.lower() != 'y':
                print("⏭️  Skipping")
                return None

        # Build command
        cmd = [str(self.quantize_tool), str(input_model), str(output_path), format]

        # Add importance matrix if specified (for IQ formats)
        if importance_matrix:
            cmd.extend(["--imatrix", str(importance_matrix)])

        # Run conversion
        print(f"\n🔄 Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout
            )

            if result.returncode == 0:
                print("✅ Conversion successful")

                # Get file info
                input_info = self.get_file_info(input_model)
                output_info = self.get_file_info(output_path)

                # Calculate compression ratio
                compression_ratio = input_info['size_bytes'] / output_info['size_bytes']

                conversion_result = {
                    'format': format,
                    'success': True,
                    'input_file': str(input_model),
                    'output_file': str(output_path),
                    'input_size_gb': input_info['size_gb'],
                    'output_size_gb': output_info['size_gb'],
                    'compression_ratio': compression_ratio,
                    'input_md5': input_info['md5'],
                    'output_md5': output_info['md5']
                }

                print(f"  Input size:  {input_info['size_gb']:.2f} GB")
                print(f"  Output size: {output_info['size_gb']:.2f} GB")
                print(f"  Compression: {compression_ratio:.2f}x")

                # Verify if requested
                if verify:
                    if self.verify_model(output_path):
                        print("✅ Verification passed")
                        conversion_result['verified'] = True
                    else:
                        print("⚠️  Verification failed")
                        conversion_result['verified'] = False

                self.conversion_log.append(conversion_result)
                return conversion_result

            else:
                print(f"❌ Conversion failed")
                print(f"Error: {result.stderr}")

                conversion_result = {
                    'format': format,
                    'success': False,
                    'error': result.stderr
                }
                self.conversion_log.append(conversion_result)
                return conversion_result

        except subprocess.TimeoutExpired:
            print("❌ Conversion timeout (30 minutes)")
            return {'format': format, 'success': False, 'error': 'Timeout'}

        except Exception as e:
            print(f"❌ Error: {e}")
            return {'format': format, 'success': False, 'error': str(e)}

    def verify_model(self, model_path: Path) -> bool:
        """Verify model by running a simple test"""
        print(f"🔍 Verifying model...")

        test_tool = self.llama_cpp_dir / "llama-cli"
        if not test_tool.exists():
            print("⚠️  llama-cli not found, skipping verification")
            return True

        cmd = [
            str(test_tool),
            "-m", str(model_path),
            "-p", "Test",
            "-n", "1",
            "--no-display-prompt"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, timeout=60)
            return result.returncode == 0
        except:
            return False

    def batch_convert(self, input_model: Path, formats: List[str],
                     output_dir: Path = None, verify: bool = True) -> List[Dict]:
        """Convert model to multiple formats"""

        print(f"\n{'='*60}")
        print(f"BATCH CONVERSION")
        print(f"{'='*60}")
        print(f"Input model: {input_model}")
        print(f"Formats: {', '.join(formats)}")
        print(f"Total: {len(formats)} conversions")

        results = []
        successful = 0
        failed = 0

        for i, format in enumerate(formats, 1):
            print(f"\n[{i}/{len(formats)}] Converting to {format}...")

            result = self.convert(input_model, format, output_dir, verify)

            if result and result.get('success'):
                successful += 1
                results.append(result)
            else:
                failed += 1

        # Print summary
        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        print(f"Total:      {len(formats)}")
        print(f"Successful: {successful} ✅")
        print(f"Failed:     {failed} ❌")

        return results

    def save_log(self, output_path: str = "conversion_log.json"):
        """Save conversion log to JSON"""
        with open(output_path, 'w') as f:
            json.dump(self.conversion_log, f, indent=2)
        print(f"\n📝 Saved conversion log to {output_path}")

    def generate_readme(self, output_dir: Path, model_name: str):
        """Generate README for converted models"""

        readme_path = output_dir / "README.md"

        with open(readme_path, 'w') as f:
            f.write(f"# {model_name} - Quantized Models\n\n")
            f.write("Available quantization formats:\n\n")

            # Create table
            f.write("| Format | Size (GB) | Use Case |\n")
            f.write("|--------|-----------|----------|\n")

            for log in self.conversion_log:
                if log.get('success'):
                    format = log['format']
                    size = log['output_size_gb']

                    # Determine use case
                    if format in ["Q8_0", "Q6_K"]:
                        use_case = "Maximum quality"
                    elif format in ["Q5_K_M", "Q5_K_S"]:
                        use_case = "Excellent quality"
                    elif format in ["Q4_K_M", "Q4_K_S"]:
                        use_case = "Balanced (recommended)"
                    elif format in ["Q3_K_M", "Q3_K_S"]:
                        use_case = "Small size"
                    else:
                        use_case = "Experimental"

                    f.write(f"| {format} | {size:.2f} | {use_case} |\n")

            f.write("\n## Quick Start\n\n")
            f.write("```bash\n")
            f.write("# Download a quantization\n")
            f.write(f"# Recommended: Q4_K_M (best balance)\n")
            f.write(f"# For quality: Q5_K_M or Q8_0\n")
            f.write(f"# For size: Q4_K_S or Q3_K_M\n")
            f.write("\n")
            f.write("# Run inference\n")
            f.write(f"./llama-cli -m {model_name}-q4_k_m.gguf -p \"Your prompt here\"\n")
            f.write("```\n")

        print(f"\n📄 Generated README at {readme_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch convert GGUF models to multiple quantization formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert to specific formats
  python format_converter.py --model model.gguf --formats Q4_K_M Q5_K_M Q8_0

  # Use a preset
  python format_converter.py --model model.gguf --preset production

  # Convert to all formats
  python format_converter.py --model model.gguf --all

Available presets:
  mobile      : Q4_0, Q4_K_S, Q4_K_M (for mobile devices)
  balanced    : Q4_K_M, Q5_K_M, Q6_K (good quality/size balance)
  quality     : Q5_K_M, Q6_K, Q8_0 (prioritize quality)
  experimental: Q2_K, Q3_K_S, Q3_K_M, Q3_K_L (small sizes)
  production  : Q4_K_M, Q5_K_M, Q8_0 (recommended for production)
        """
    )

    parser.add_argument("--model", required=True, help="Input GGUF model path")
    parser.add_argument("--formats", nargs="+", help="Specific formats to convert to")
    parser.add_argument("--preset", choices=PRESETS.keys(), help="Use a format preset")
    parser.add_argument("--all", action="store_true", help="Convert to all formats")
    parser.add_argument("--output-dir", help="Output directory (default: same as input)")
    parser.add_argument("--no-verify", action="store_true", help="Skip model verification")
    parser.add_argument("--llama-cpp-dir", default="../../../",
                       help="Path to llama.cpp directory")
    parser.add_argument("--log", default="conversion_log.json",
                       help="Conversion log output path")
    parser.add_argument("--generate-readme", action="store_true",
                       help="Generate README.md for converted models")

    args = parser.parse_args()

    # Validate input
    input_model = Path(args.model)
    if not input_model.exists():
        print(f"❌ Error: Model file not found: {input_model}")
        sys.exit(1)

    # Determine formats to convert to
    if args.all:
        formats = ALL_FORMATS
    elif args.preset:
        formats = PRESETS[args.preset]
    elif args.formats:
        formats = args.formats
    else:
        print("❌ Error: Must specify --formats, --preset, or --all")
        sys.exit(1)

    # Create converter
    try:
        converter = FormatConverter(args.llama_cpp_dir)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)

    # Run batch conversion
    results = converter.batch_convert(
        input_model,
        formats,
        Path(args.output_dir) if args.output_dir else None,
        verify=not args.no_verify
    )

    # Save log
    converter.save_log(args.log)

    # Generate README if requested
    if args.generate_readme:
        output_dir = Path(args.output_dir) if args.output_dir else input_model.parent
        converter.generate_readme(output_dir, input_model.stem)

    # Exit with appropriate code
    if all(r.get('success') for r in results):
        print("\n✅ All conversions successful!")
        sys.exit(0)
    else:
        print("\n⚠️  Some conversions failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
