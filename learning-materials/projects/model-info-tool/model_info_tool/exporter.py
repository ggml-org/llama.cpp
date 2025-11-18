"""
Model Information Exporter
Export model information to various formats.
"""

import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
from .model_analyzer import ModelAnalyzer


class ModelExporter:
    """Export model information to files."""

    def __init__(self, analyzer: ModelAnalyzer):
        """
        Initialize exporter.

        Args:
            analyzer: ModelAnalyzer instance
        """
        self.analyzer = analyzer

    def to_json(self, output_path: str = None, indent: int = 2, include_tensors: bool = False) -> str:
        """
        Export model information to JSON.

        Args:
            output_path: Path to save JSON file (optional)
            indent: JSON indentation level
            include_tensors: Include detailed tensor information

        Returns:
            JSON string
        """
        data = {
            "export_info": {
                "timestamp": datetime.now().isoformat(),
                "tool_version": "1.0.0"
            },
            "basic_info": self.analyzer.get_basic_info(),
            "architecture": self.analyzer.get_architecture_details(),
            "tokenizer": self.analyzer.get_tokenizer_info(),
            "quantization": self.analyzer.get_quantization_info(),
            "memory_requirements": self.analyzer.get_memory_requirements(),
            "context": self.analyzer.get_context_info(),
        }

        if include_tensors:
            data["tensors"] = self.analyzer.get_tensor_details()

        json_str = json.dumps(data, indent=indent, default=str)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(json_str)
            print(f"Exported to: {output_path}")

        return json_str

    def to_markdown(self, output_path: str = None) -> str:
        """
        Export model information to Markdown.

        Args:
            output_path: Path to save Markdown file (optional)

        Returns:
            Markdown string
        """
        basic = self.analyzer.get_basic_info()
        arch = self.analyzer.get_architecture_details()
        tokenizer = self.analyzer.get_tokenizer_info()
        quant = self.analyzer.get_quantization_info()
        memory = self.analyzer.get_memory_requirements()
        context = self.analyzer.get_context_info()

        lines = []
        lines.append(f"# {basic['name']}")
        lines.append("")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        lines.append("## Basic Information")
        lines.append("")
        lines.append(f"- **File:** `{basic['file_name']}`")
        lines.append(f"- **Architecture:** {basic['architecture']}")
        lines.append(f"- **Parameters:** {basic['parameter_count_formatted']} ({basic['parameter_count']:,})")
        lines.append(f"- **File Size:** {basic['file_size_gb']} GB ({basic['file_size_mb']} MB)")
        lines.append(f"- **Quantization:** {quant['predominant_type']}")
        lines.append(f"- **Tensor Count:** {basic['tensor_count']}")
        lines.append("")

        if arch:
            lines.append("## Architecture Details")
            lines.append("")
            lines.append("| Property | Value |")
            lines.append("|----------|-------|")
            for key, value in arch.items():
                if key != "architecture":
                    lines.append(f"| {key} | {value} |")
            lines.append("")

        if tokenizer:
            lines.append("## Tokenizer")
            lines.append("")
            lines.append("| Property | Value |")
            lines.append("|----------|-------|")
            for key, value in tokenizer.items():
                if value is not None:
                    lines.append(f"| {key} | {value} |")
            lines.append("")

        if context:
            lines.append("## Context")
            lines.append("")
            lines.append("| Property | Value |")
            lines.append("|----------|-------|")
            for key, value in context.items():
                if value is not None:
                    lines.append(f"| {key} | {value} |")
            lines.append("")

        lines.append("## Memory Requirements (Estimated)")
        lines.append("")
        lines.append(f"- **Model Size:** {memory['model_size_gb']} GB")
        lines.append(f"- **CPU-only RAM:** ~{memory['estimated_ram_cpu_only_gb']} GB")
        lines.append(f"- **GPU-full VRAM:** ~{memory['estimated_vram_gpu_full_gb']} GB")
        lines.append("")
        lines.append(f"> {memory['note']}")
        lines.append("")

        if quant.get('tensor_types'):
            lines.append("## Quantization Distribution")
            lines.append("")
            lines.append("| Type | Count |")
            lines.append("|------|-------|")
            for qtype, count in sorted(quant['tensor_types'].items()):
                lines.append(f"| {qtype} | {count} |")
            lines.append("")

        markdown_str = "\n".join(lines)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(markdown_str)
            print(f"Exported to: {output_path}")

        return markdown_str

    def to_csv_summary(self, output_path: str = None) -> str:
        """
        Export basic model information to CSV format.

        Args:
            output_path: Path to save CSV file (optional)

        Returns:
            CSV string
        """
        basic = self.analyzer.get_basic_info()
        quant = self.analyzer.get_quantization_info()
        memory = self.analyzer.get_memory_requirements()

        headers = [
            "Name",
            "Architecture",
            "Parameters",
            "File_Size_GB",
            "Quantization",
            "Tensor_Count",
            "RAM_CPU_GB",
            "VRAM_GPU_GB"
        ]

        values = [
            basic['name'],
            basic['architecture'],
            basic['parameter_count'],
            basic['file_size_gb'],
            quant['predominant_type'],
            basic['tensor_count'],
            memory['estimated_ram_cpu_only_gb'],
            memory['estimated_vram_gpu_full_gb']
        ]

        csv_lines = [
            ",".join(headers),
            ",".join(map(str, values))
        ]

        csv_str = "\n".join(csv_lines)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(csv_str)
            print(f"Exported to: {output_path}")

        return csv_str

    @staticmethod
    def compare_models_to_json(model_paths: list, output_path: str = None) -> str:
        """
        Compare multiple models and export to JSON.

        Args:
            model_paths: List of paths to GGUF models
            output_path: Path to save JSON file (optional)

        Returns:
            JSON string
        """
        models = []

        for path in model_paths:
            analyzer = ModelAnalyzer(path)
            basic = analyzer.get_basic_info()
            quant = analyzer.get_quantization_info()
            memory = analyzer.get_memory_requirements()

            models.append({
                "file_path": str(path),
                "name": basic['name'],
                "architecture": basic['architecture'],
                "parameters": basic['parameter_count'],
                "parameters_formatted": basic['parameter_count_formatted'],
                "file_size_gb": basic['file_size_gb'],
                "quantization": quant['predominant_type'],
                "tensor_count": basic['tensor_count'],
                "memory_cpu_gb": memory['estimated_ram_cpu_only_gb'],
                "memory_gpu_gb": memory['estimated_vram_gpu_full_gb']
            })

        data = {
            "comparison_info": {
                "timestamp": datetime.now().isoformat(),
                "model_count": len(models)
            },
            "models": models
        }

        json_str = json.dumps(data, indent=2)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(json_str)
            print(f"Comparison exported to: {output_path}")

        return json_str
