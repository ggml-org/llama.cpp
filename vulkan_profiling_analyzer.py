#!/usr/bin/env python3
"""
Vulkan Profiling Results Analyzer

This script parses all Vulkan profiling result inference steps from a log file
and computes comprehensive global statistics.

Usage:
    python vulkan_profiling_analyzer.py [log_file_path]

If no log file path is provided, it defaults to 'log.txt' in the current directory.
"""

import sys
import re
import logging
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import statistics
import argparse
import matplotlib.pyplot as plt
import os
import base64
import io


@dataclass
class GeneralTimingEntry:
    """Represents a single general timing entry."""

    operation: str
    count: int
    avg_time_us: float
    params: Optional[str] = None  # For operations like MUL_MAT_VEC with parameters


@dataclass
class MatMulTimingEntry:
    """Represents a single mat_mul timing entry."""

    operation_type: str
    data_types: str  # e.g., "q4_0 x f32 -> f32"
    matrix_dims: str  # e.g., "[2048x1024] x [2048x1] -> [1024x1]"
    count: int
    avg_time_us: float


@dataclass
class ProfilingSection:
    """Represents one complete profiling section."""

    section_id: int
    general_timings: List[GeneralTimingEntry] = field(default_factory=list)
    mat_mul_timings: List[MatMulTimingEntry] = field(default_factory=list)
    mat_mul_summaries: Dict[str, Tuple[int, float]] = field(
        default_factory=dict
    )  # operation -> (total_ops, avg_time)
    total_operation_types: Optional[int] = None
    total_variations: Optional[int] = None


@dataclass
class ModelInfo:
    """Model information extracted from the log."""

    model_file: str = ""
    model_name: str = ""
    architecture: str = ""
    size_label: str = ""
    context_length: int = 0
    embedding_length: int = 0
    feed_forward_length: int = 0
    layer_count: int = 0
    attention_heads: int = 0
    attention_heads_kv: int = 0
    quantization: str = ""
    file_size: str = ""
    quantized_by: str = ""
    vocab_size: int = 0
    tensor_types: Dict[str, int] = field(default_factory=dict)


@dataclass
class DeviceAllocation:
    """Device allocation information."""

    gpu_device: str = ""
    gpu_layers: int = 0
    cpu_operations: int = 0
    gpu_operations: int = 0
    cpu_operation_types: Dict[str, int] = field(default_factory=dict)


@dataclass
class GlobalStats:
    """Global statistics across all inference steps."""

    total_inference_steps: int
    general_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    mat_mul_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    mat_mul_normalized: Dict[str, Dict[str, float]] = field(
        default_factory=dict
    )  # normalized by tensor size
    operation_frequency: Counter = field(default_factory=Counter)
    model_info: ModelInfo = field(default_factory=ModelInfo)
    device_allocation: DeviceAllocation = field(default_factory=DeviceAllocation)


class VulkanProfilingAnalyzer:
    def __init__(self, log_file_path: str):
        self.log_file_path = log_file_path
        self.inference_steps: List[ProfilingSection] = []

    def parse_log_file(self) -> None:
        """Parse the entire log file and extract all inference steps."""
        with open(self.log_file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # Find all inference steps
        pattern = r"={16}\nVulkan Profiling Results:\n={16}"
        inference_step_contents = re.split(pattern, content)

        # Skip the first section (before first profiling results)
        if len(inference_step_contents) > 1:
            inference_step_contents = inference_step_contents[1:]

        logging.info(f"Found {len(inference_step_contents)} inference steps")

        for i, step_content in enumerate(inference_step_contents):
            self._parse_section(i, step_content)

    def _parse_device_allocation(self) -> DeviceAllocation:
        """Parse device allocation information from the log file."""
        device_alloc = DeviceAllocation()

        with open(self.log_file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # Parse GPU device info
        gpu_match = re.search(r"using device (Vulkan\d+) \(([^)]+)\)", content)
        if gpu_match:
            device_alloc.gpu_device = f"{gpu_match.group(1)} ({gpu_match.group(2)})"

        # Count GPU layers
        device_alloc.gpu_layers = len(
            re.findall(r"assigned to device Vulkan\d+", content)
        )

        # Count CPU/GPU operations by parsing each node
        # Each node represents a single operation, determined by where its output tensor is allocated
        cpu_ops = re.findall(r"node #[^(]*\(\s*([^)]+)\):[^[]*\[\s*CPU\s*\]", content)
        gpu_ops = re.findall(r"node #[^(]*\(\s*([^)]+)\):[^[]*\[Vulka[^]]*\]", content)

        device_alloc.cpu_operations = len(cpu_ops)
        device_alloc.gpu_operations = len(gpu_ops)

        # Parse CPU operation types
        for op_type in cpu_ops:
            op_type = op_type.strip()
            device_alloc.cpu_operation_types[op_type] = (
                device_alloc.cpu_operation_types.get(op_type, 0) + 1
            )

        return device_alloc

    def _parse_model_info(self) -> ModelInfo:
        """Parse model information from the log file."""
        model_info = ModelInfo()

        with open(self.log_file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # Extract model file name
        model_file_match = re.search(
            r"loaded meta data with [^/]+ from ([^\s]+)", content
        )
        if model_file_match:
            model_info.model_file = model_file_match.group(1)

        # Extract model information from key-value pairs
        kv_patterns = {
            "model_name": r"general\.name\s+str\s+=\s+([^\n]+)",
            "architecture": r"general\.architecture\s+str\s+=\s+([^\n]+)",
            "size_label": r"general\.size_label\s+str\s+=\s+([^\n]+)",
            "quantized_by": r"general\.quantized_by\s+str\s+=\s+([^\n]+)",
            "layer_count": r"(\w+)\.block_count\s+u32\s+=\s+(\d+)",
            "context_length": r"(\w+)\.context_length\s+u32\s+=\s+(\d+)",
            "embedding_length": r"(\w+)\.embedding_length\s+u32\s+=\s+(\d+)",
            "feed_forward_length": r"(\w+)\.feed_forward_length\s+u32\s+=\s+(\d+)",
            "attention_heads": r"(\w+)\.attention\.head_count\s+u32\s+=\s+(\d+)",
            "attention_heads_kv": r"(\w+)\.attention\.head_count_kv\s+u32\s+=\s+(\d+)",
        }

        for field_name, pattern in kv_patterns.items():
            match = re.search(pattern, content)
            if match:
                if field_name in [
                    "layer_count",
                    "context_length",
                    "embedding_length",
                    "feed_forward_length",
                    "attention_heads",
                    "attention_heads_kv",
                ]:
                    setattr(
                        model_info,
                        field_name,
                        int(
                            match.group(2)
                            if len(match.groups()) > 1
                            else match.group(1)
                        ),
                    )
                else:
                    setattr(model_info, field_name, match.group(1).strip())

        # Extract quantization info
        quant_match = re.search(r"print_info: file type\s+=\s+(\w+)", content)
        if quant_match:
            model_info.quantization = quant_match.group(1)

        # Extract file size
        size_match = re.search(r"print_info: file size\s+=\s+([^(]+)", content)
        if size_match:
            model_info.file_size = size_match.group(1).strip()

        # Extract vocab size from tokenizer
        vocab_match = re.search(r"tokenizer\.ggml\.tokens\s+arr\[str,(\d+)\]", content)
        if vocab_match:
            model_info.vocab_size = int(vocab_match.group(1))

        # Extract tensor types
        tensor_types = re.findall(
            r"llama_model_loader: - type\s+(\w+):\s+(\d+) tensors", content
        )
        for tensor_type, count in tensor_types:
            model_info.tensor_types[tensor_type] = int(count)

        return model_info

    def _parse_section(self, section_id: int, content: str) -> None:
        """Parse a single profiling section."""
        section = ProfilingSection(section_id=section_id)

        # Parse Legacy Timing Summary
        general_match = re.search(
            r"Legacy Timing Summary:\n-+\n(.*?)\n(?=Enhanced Triplet|$)",
            content,
            re.DOTALL,
        )
        if general_match:
            section.general_timings = self._parse_general_timings(
                general_match.group(1)
            )

        # Parse Enhanced Triplet Timing Analysis
        mat_mul_match = re.search(
            r"Enhanced Triplet Timing Analysis by Operation Type:\n=+\n(.*?)\nOverall Statistics:",
            content,
            re.DOTALL,
        )
        if mat_mul_match:
            section.mat_mul_timings, section.mat_mul_summaries = (
                self._parse_mat_mul_timings(mat_mul_match.group(1))
            )

        # Parse overall statistics
        stats_match = re.search(
            r"Total operation types: (\d+)\nTotal variations: (\d+)", content
        )
        if stats_match:
            section.total_operation_types = int(stats_match.group(1))
            section.total_variations = int(stats_match.group(2))

        self.inference_steps.append(section)

    def _parse_general_timings(self, content: str) -> List[GeneralTimingEntry]:
        """Parse general timing entries."""
        timings = []
        lines = content.strip().split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Handle different formats:
            # MUL_MAT_VEC m=1024 k=2048: 56 x 417.28 us
            # ADD: 56 x 11.48 us

            if ":" in line:
                operation_part, timing_part = line.split(":", 1)
                operation_part = operation_part.strip()
                timing_part = timing_part.strip()

                # Extract operation and parameters
                if " " in operation_part and any(
                    c in operation_part for c in ["=", "x"]
                ):
                    # Has parameters
                    parts = operation_part.split(" ", 1)
                    operation = parts[0]
                    params = parts[1]
                else:
                    operation = operation_part
                    params = None

                # Parse timing: "56 x 417.28 us"
                timing_match = re.match(r"(\d+)\s*x\s*([\d.]+)\s*us", timing_part)
                if timing_match:
                    count = int(timing_match.group(1))
                    avg_time = float(timing_match.group(2))

                    timings.append(
                        GeneralTimingEntry(
                            operation=operation,
                            count=count,
                            avg_time_us=avg_time,
                            params=params,
                        )
                    )

        return timings

    def _parse_mat_mul_timings(
        self, content: str
    ) -> Tuple[List[MatMulTimingEntry], Dict[str, Tuple[int, float]]]:
        """Parse mat_mul timing entries."""
        timings = []
        summaries = {}

        # Split by operation types (sections starting with operation name followed by dashes)
        operation_sections = re.split(r"\n([a-zA-Z_][a-zA-Z0-9_]*):\n-+\n", content)

        if len(operation_sections) > 1:
            # First element is empty or initial content, then alternating operation names and content
            for i in range(1, len(operation_sections), 2):
                if i + 1 < len(operation_sections):
                    operation_type = operation_sections[i]
                    op_content = operation_sections[i + 1]

                    # Parse individual timing entries
                    lines = op_content.strip().split("\n")
                    for line in lines:
                        line = line.strip()
                        if line.startswith("→"):
                            # Summary line: "→ ggml_vk_mul_mat_vec_q_f16 Summary: 197 total ops, 1765.49 us avg"
                            summary_match = re.search(
                                r"(\w+)\s+Summary:\s+(\d+)\s+total ops,\s+([\d.]+)\s+us avg",
                                line,
                            )
                            if summary_match:
                                summary_op = summary_match.group(1)
                                total_ops = int(summary_match.group(2))
                                avg_time = float(summary_match.group(3))
                                summaries[summary_op] = (total_ops, avg_time)
                        else:
                            # Individual entry: "q4_0 x f32 -> f32 | [2048x1024] x [2048x1] -> [1024x1]: 56 ops, 417.28 us avg"
                            entry_match = re.match(
                                r"(.+?)\s*\|\s*(.+?):\s*(\d+)\s+ops,\s*([\d.]+)\s+us avg",
                                line,
                            )
                            if entry_match:
                                data_types = entry_match.group(1).strip()
                                matrix_dims = entry_match.group(2).strip()
                                count = int(entry_match.group(3))
                                avg_time = float(entry_match.group(4))

                                timings.append(
                                    MatMulTimingEntry(
                                        operation_type=operation_type,
                                        data_types=data_types,
                                        matrix_dims=matrix_dims,
                                        count=count,
                                        avg_time_us=avg_time,
                                    )
                                )

        return timings, summaries

    def compute_global_statistics(self) -> GlobalStats:
        """Compute comprehensive global statistics."""
        global_stats = GlobalStats(total_inference_steps=len(self.inference_steps))

        # Parse model information
        global_stats.model_info = self._parse_model_info()

        # Parse device allocation information
        global_stats.device_allocation = self._parse_device_allocation()

        # Aggregate general timing statistics
        general_data = defaultdict(list)  # operation -> list of (count, avg_time_us)

        for section in self.inference_steps:
            for timing in section.general_timings:
                # Filter out MUL_MAT operations from general statistics
                if timing.operation.startswith("MUL_MAT"):
                    continue

                key = (
                    f"{timing.operation}({timing.params})"
                    if timing.params
                    else timing.operation
                )
                general_data[key].append((timing.count, timing.avg_time_us))
                global_stats.operation_frequency[key] += timing.count

        # Compute statistics for each general operation
        for operation, data_points in general_data.items():
            counts = [d[0] for d in data_points]
            times = [d[1] for d in data_points]
            total_ops = sum(counts)

            global_stats.general_stats[operation] = {
                "total_operations": total_ops,
                "total_inference_steps": len(data_points),
                "avg_time_mean": statistics.mean(times),
                "avg_time_median": statistics.median(times),
                "avg_time_min": min(times),
                "avg_time_max": max(times),
                "avg_time_stdev": statistics.stdev(times) if len(times) > 1 else 0.0,
                "count_mean": statistics.mean(counts),
                "count_median": statistics.median(counts),
                "count_min": min(counts),
                "count_max": max(counts),
            }

        # Aggregate mat_mul timing statistics
        mat_mul_data = defaultdict(
            list
        )  # (operation_type, data_types, matrix_dims) -> list of (count, avg_time_us)

        for section in self.inference_steps:
            for timing in section.mat_mul_timings:
                key = (
                    f"{timing.operation_type}|{timing.data_types}|{timing.matrix_dims}"
                )
                mat_mul_data[key].append((timing.count, timing.avg_time_us))

        # Compute statistics for each mat_mul operation
        for key, data_points in mat_mul_data.items():
            operation_type, data_types, matrix_dims = key.split("|", 2)
            counts = [d[0] for d in data_points]
            times = [d[1] for d in data_points]
            total_ops = sum(counts)

            display_key = f"{operation_type} [{data_types}] {matrix_dims}"
            global_stats.mat_mul_stats[display_key] = {
                "total_operations": total_ops,
                "total_inference_steps": len(data_points),
                "avg_time_mean": statistics.mean(times),
                "avg_time_median": statistics.median(times),
                "avg_time_min": min(times),
                "avg_time_max": max(times),
                "avg_time_stdev": statistics.stdev(times) if len(times) > 1 else 0.0,
                "count_mean": statistics.mean(counts),
                "count_median": statistics.median(counts),
                "count_min": min(counts),
                "count_max": max(counts),
            }

        # Calculate normalized MatMul performance (time divided by tensor sizes)
        self._calculate_normalized_mat_mul_performance(global_stats)

        return global_stats

    def _calculate_normalized_mat_mul_performance(
        self, global_stats: GlobalStats
    ) -> None:
        """Calculate MatMul performance normalized by tensor sizes."""
        # Group operations by operation type and tensor types
        grouped_operations = defaultdict(
            list
        )  # (operation_type, tensor_types) -> list of normalized times

        # Process ALL MatMul operations from the timing statistics
        for operation_key, stats in global_stats.mat_mul_stats.items():
            # Parse operation type and tensor types from the key
            # Format: "ggml_vk_mul_mat_vec_q_f16 [q4_0 x f32 -> f32] [2048x6144] x [2048x1] -> [6144x1]"

            # Extract operation type and tensor types
            match = re.match(r"(ggml_vk_mul_mat[^[]*)\s*(\[[^]]+\])", operation_key)
            if match:
                operation_type = match.group(1).strip()
                tensor_types = match.group(2)

                # Extract matrix dimensions
                dims_match = re.search(
                    r"\[(\d+)x(\d+)\]\s*x\s*\[(\d+)x(\d+)\]\s*->\s*\[(\d+)x(\d+)\]",
                    operation_key,
                )
                if dims_match:
                    # Extract input tensor dimensions
                    a = int(dims_match.group(1))  # rows of first matrix
                    b = int(
                        dims_match.group(2)
                    )  # cols of first matrix / rows of second matrix
                    c = int(dims_match.group(4))  # cols of second matrix

                    # Calculate computational volume: a × b × c
                    computational_volume = a * b * c

                    # Normalize average time by computational volume
                    normalized_time = (
                        stats["avg_time_mean"] / computational_volume
                        if computational_volume > 0
                        else 0
                    )

                    # Group key: operation type + tensor types
                    group_key = f"{operation_type} {tensor_types}"
                    grouped_operations[group_key].append(
                        (normalized_time, stats["total_operations"])
                    )

        # Calculate weighted mean normalized time for each group
        for group_key, data_points in grouped_operations.items():
            if data_points:
                # Extract normalized times and weights (total operations)
                normalized_times = [point[0] for point in data_points]
                weights = [point[1] for point in data_points]

                # Calculate weighted mean: sum(value * weight) / sum(weights)
                weighted_sum = sum(
                    norm_time * weight
                    for norm_time, weight in zip(normalized_times, weights)
                )
                total_weight = sum(weights)

                weighted_mean_normalized_time = (
                    weighted_sum / total_weight if total_weight > 0 else 0
                )

                global_stats.mat_mul_normalized[group_key] = {
                    "mean_normalized_time": weighted_mean_normalized_time,
                    "sample_count": len(data_points),
                    "total_operations": total_weight,
                }

    def _create_normalized_time_plot(
        self,
        global_stats: GlobalStats,
        output_dir: str = ".",
        return_base64: bool = False,
    ) -> str:
        """Create a bar plot of normalized times and return the image filename or base64 data."""
        if not global_stats.mat_mul_normalized:
            return ""

        try:
            # Prepare data for plotting
            operations = []
            normalized_times = []

            # Sort by normalized time (most efficient first)
            sorted_data = sorted(
                global_stats.mat_mul_normalized.items(),
                key=lambda x: x[1]["mean_normalized_time"],
            )

            for group_key, stats in sorted_data:
                # Shorten operation names for better readability
                short_name = group_key.replace("ggml_vk_mul_mat_vec_", "vec_").replace(
                    "ggml_vk_mul_mat_", "mat_"
                )
                short_name = (
                    short_name.replace("_q_f16", "")
                    .replace("_f16_f32", "")
                    .replace("_nc_f16_f32", "_nc")
                    .replace("_p021_f16_f32", "_p021")
                )
                operations.append(short_name)
                normalized_times.append(stats["mean_normalized_time"])

            # Create the plot
            plt.figure(figsize=(12, 8))
            bars = plt.bar(
                range(len(operations)), normalized_times, color="steelblue", alpha=0.7
            )

            # Customize the plot
            plt.title(
                "MatMul Operations: Normalized Time Efficiency\n(Lower = More Efficient)",
                fontsize=14,
                fontweight="bold",
            )
            plt.xlabel("Operation Type and Tensor Types", fontsize=12)
            plt.ylabel("Normalized Time (μs per tensor element)", fontsize=12)

            # Set x-axis labels with rotation for readability
            plt.xticks(range(len(operations)), operations, rotation=45, ha="right")

            # Add value labels on top of bars
            for bar, value in zip(bars, normalized_times):
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + value * 0.01,
                    f"{value:.6f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

            # Improve layout
            plt.tight_layout()
            plt.grid(axis="y", alpha=0.3)

            if return_base64:
                # Save to memory buffer and convert to base64
                buffer = io.BytesIO()
                plt.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                plt.close()
                buffer.close()
                return f"data:image/png;base64,{image_base64}"
            else:
                # Save the plot to file
                plot_filename = os.path.join(output_dir, "matmul_normalized_times.png")
                plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
                plt.close()
                return plot_filename

        except ImportError:
            # matplotlib not available
            return ""
        except Exception as e:
            # Any other error in plotting
            logging.warning(f"Could not create plot: {e}")
            return ""

    def generate_report(self, global_stats: GlobalStats) -> str:
        """Generate a comprehensive text report."""
        report = []

        report.append("=" * 80)
        report.append("VULKAN PROFILING GLOBAL ANALYSIS REPORT")
        report.append("=" * 80)
        report.append("")

        # Model Information
        report.append("MODEL INFORMATION")
        report.append("=" * 30)
        report.append(f"Model File: {global_stats.model_info.model_file}")
        report.append(f"Model Name: {global_stats.model_info.model_name}")
        report.append(f"Architecture: {global_stats.model_info.architecture}")
        report.append(f"Model Size: {global_stats.model_info.size_label}")
        if global_stats.model_info.layer_count > 0:
            report.append(f"Layers: {global_stats.model_info.layer_count}")
        if global_stats.model_info.context_length > 0:
            report.append(f"Context Length: {global_stats.model_info.context_length:,}")
        if global_stats.model_info.embedding_length > 0:
            report.append(
                f"Embedding Dimension: {global_stats.model_info.embedding_length:,}"
            )
        if global_stats.model_info.attention_heads > 0:
            if (
                global_stats.model_info.attention_heads_kv > 0
                and global_stats.model_info.attention_heads_kv
                != global_stats.model_info.attention_heads
            ):
                report.append(
                    f"Attention Heads: {global_stats.model_info.attention_heads} ({global_stats.model_info.attention_heads_kv} KV heads)"
                )
            else:
                report.append(
                    f"Attention Heads: {global_stats.model_info.attention_heads}"
                )
        if global_stats.model_info.vocab_size > 0:
            report.append(f"Vocabulary Size: {global_stats.model_info.vocab_size:,}")
        report.append(f"Quantization: {global_stats.model_info.quantization}")
        if global_stats.model_info.file_size:
            report.append(f"File Size: {global_stats.model_info.file_size}")
        if global_stats.model_info.quantized_by:
            report.append(f"Quantized By: {global_stats.model_info.quantized_by}")

        # Show tensor types breakdown
        if global_stats.model_info.tensor_types:
            report.append("Tensor Types:")
            for tensor_type, count in sorted(
                global_stats.model_info.tensor_types.items(),
                key=lambda x: x[1],
                reverse=True,
            ):
                report.append(f"  {tensor_type}: {count:,} tensors")
        report.append("")

        # Device Allocation Information
        report.append("DEVICE ALLOCATION SUMMARY")
        report.append("=" * 40)
        report.append(f"GPU Device: {global_stats.device_allocation.gpu_device}")
        report.append(f"GPU Layers: {global_stats.device_allocation.gpu_layers}")
        report.append(
            f"GPU Operations: {global_stats.device_allocation.gpu_operations:,}"
        )
        report.append(
            f"CPU Operations: {global_stats.device_allocation.cpu_operations:,}"
        )
        total_ops_device = (
            global_stats.device_allocation.gpu_operations
            + global_stats.device_allocation.cpu_operations
        )
        gpu_percentage = (
            (global_stats.device_allocation.gpu_operations / total_ops_device * 100)
            if total_ops_device > 0
            else 0
        )
        cpu_percentage = (
            (global_stats.device_allocation.cpu_operations / total_ops_device * 100)
            if total_ops_device > 0
            else 0
        )
        report.append(
            f"GPU Utilization: {gpu_percentage:.1f}% ({global_stats.device_allocation.gpu_operations:,}/{total_ops_device:,} operations)"
        )
        report.append(
            f"CPU Utilization: {cpu_percentage:.1f}% ({global_stats.device_allocation.cpu_operations:,}/{total_ops_device:,} operations)"
        )

        # Show CPU operation types if any
        if global_stats.device_allocation.cpu_operation_types:
            report.append("CPU Operations by Type:")
            for op_type, count in sorted(
                global_stats.device_allocation.cpu_operation_types.items(),
                key=lambda x: x[1],
                reverse=True,
            ):
                report.append(f"  {op_type}: {count:,} operations")

        report.append("")

        # Inference Steps Summary
        report.append(
            f"Total inference steps analyzed: {global_stats.total_inference_steps}"
        )
        report.append(
            f"Note: Device allocation counts ALL executed operations ({total_ops_device:,}),"
        )
        report.append(
            "      while timing statistics below only count profiled operations."
        )
        report.append("")

        # General Operations Statistics
        report.append("GENERAL OPERATIONS STATISTICS")
        report.append("(Ordered by average execution time - highest first)")
        report.append("=" * 50)
        report.append("")

        # Sort by average time (highest first)
        sorted_general = sorted(
            global_stats.general_stats.items(),
            key=lambda x: x[1]["avg_time_mean"],
            reverse=True,
        )

        for operation, stats in sorted_general:
            report.append(f"Operation: {operation}")
            report.append(f"  Total Operations: {stats['total_operations']:,}")
            report.append(
                f"  Appeared in {stats['total_inference_steps']}/{global_stats.total_inference_steps} inference steps"
            )
            report.append(
                f"  Average Time (μs): {stats['avg_time_mean']:.2f} ± {stats['avg_time_stdev']:.2f}"
            )
            report.append(
                f"    Min: {stats['avg_time_min']:.2f}, Max: {stats['avg_time_max']:.2f}, Median: {stats['avg_time_median']:.2f}"
            )
            report.append(
                f"  Operations per Inference Step: {stats['count_mean']:.1f} ± {stats['count_stdev']:.1f}"
                if stats.get("count_stdev")
                else f"  Operations per Inference Step: {stats['count_mean']:.1f}"
            )
            report.append("")

        # MatMul Timing Statistics
        report.append("MAT_MUL TIMING STATISTICS")
        report.append("(Ordered by average execution time - highest first)")
        report.append("=" * 50)
        report.append("")

        # Sort by average time (highest first)
        sorted_mat_mul = sorted(
            global_stats.mat_mul_stats.items(),
            key=lambda x: x[1]["avg_time_mean"],
            reverse=True,
        )

        for operation, stats in sorted_mat_mul:
            report.append(f"Operation: {operation}")
            report.append(f"  Total Operations: {stats['total_operations']:,}")
            report.append(
                f"  Appeared in {stats['total_inference_steps']}/{global_stats.total_inference_steps} inference steps"
            )
            report.append(
                f"  Average Time (μs): {stats['avg_time_mean']:.2f} ± {stats['avg_time_stdev']:.2f}"
            )
            report.append(
                f"    Min: {stats['avg_time_min']:.2f}, Max: {stats['avg_time_max']:.2f}, Median: {stats['avg_time_median']:.2f}"
            )
            report.append(f"  Operations per Inference Step: {stats['count_mean']:.1f}")
            report.append("")

        # MatMul Tensor Size Normalized Analysis
        if global_stats.mat_mul_normalized:
            report.append("MAT_MUL TENSOR SIZE NORMALIZED ANALYSIS")
            report.append(
                "(Weighted mean normalized time by operation type and tensor types)"
            )
            report.append("=" * 60)
            report.append("")

            # Sort by mean normalized time (most efficient first - lowest time per operation)
            sorted_normalized = sorted(
                global_stats.mat_mul_normalized.items(),
                key=lambda x: x[1]["mean_normalized_time"],
            )

            for group_key, stats in sorted_normalized:
                report.append(f"Operation: {group_key}")
                report.append(
                    f"  Weighted Mean Normalized Time: {stats['mean_normalized_time']:.9f} μs per tensor element"
                )
                report.append(
                    f"  (Based on {stats['total_operations']:,} total operations across {stats['sample_count']} matrix size variations)"
                )
                report.append("")

        # Top operations by total time
        report.append("TOP OPERATIONS BY TOTAL EXECUTION TIME")
        report.append("=" * 50)
        report.append("")

        # Calculate total time for each operation (total_ops * avg_time_mean)
        total_times = []
        for operation, stats in global_stats.general_stats.items():
            total_time = stats["total_operations"] * stats["avg_time_mean"]
            total_times.append(
                (
                    operation,
                    total_time,
                    stats["total_operations"],
                    stats["avg_time_mean"],
                )
            )

        for operation, stats in global_stats.mat_mul_stats.items():
            total_time = stats["total_operations"] * stats["avg_time_mean"]
            total_times.append(
                (
                    operation,
                    total_time,
                    stats["total_operations"],
                    stats["avg_time_mean"],
                )
            )

        total_times.sort(key=lambda x: x[1], reverse=True)

        for i, (operation, total_time, total_ops, avg_time) in enumerate(
            total_times[:20]
        ):
            report.append(f"Operation: {operation}")
            report.append(f"  Total Operations: {total_ops:,}")
            report.append(
                f"  Total Execution Time: {total_time:,.2f} μs ({total_time /1000:.2f} ms)"
            )
            report.append(f"  Average Time (μs): {avg_time:.2f}")
            report.append("")

        # Summary statistics
        report.append("TIMING SUMMARY STATISTICS")
        report.append("(Based on profiled operations only)")
        report.append("=" * 30)
        report.append("")

        total_general_ops = sum(
            stats["total_operations"] for stats in global_stats.general_stats.values()
        )
        total_mat_mul_ops = sum(
            stats["total_operations"] for stats in global_stats.mat_mul_stats.values()
        )
        total_all_ops = total_general_ops + total_mat_mul_ops

        # Calculate percentages
        general_percentage = (
            (total_general_ops / total_all_ops * 100) if total_all_ops > 0 else 0
        )
        mat_mul_percentage = (
            (total_mat_mul_ops / total_all_ops * 100) if total_all_ops > 0 else 0
        )

        # Calculate total execution times
        total_general_time = sum(
            stats["total_operations"] * stats["avg_time_mean"]
            for stats in global_stats.general_stats.values()
        )
        total_mat_mul_time = sum(
            stats["total_operations"] * stats["avg_time_mean"]
            for stats in global_stats.mat_mul_stats.values()
        )
        total_execution_time = total_general_time + total_mat_mul_time

        # Calculate time per inference step
        time_per_inference_step = (
            total_execution_time / global_stats.total_inference_steps
            if global_stats.total_inference_steps > 0
            else 0
        )

        report.append(
            f"Total General Operations: {total_general_ops:,} ({general_percentage:.1f}%)"
        )
        report.append(
            f"Total MatMul Operations: {total_mat_mul_ops:,} ({mat_mul_percentage:.1f}%)"
        )
        report.append(f"Total Profiled Operations: {total_all_ops:,}")
        report.append(
            f"Non-profiled Operations: {total_ops_device - total_all_ops:,} (setup, memory management, etc.)"
        )
        report.append("")
        report.append(
            f"Total Execution Time: {total_execution_time:,.2f} μs ({total_execution_time /1000:.2f} ms)"
        )
        report.append(
            f"  General Operations: {total_general_time:,.2f} μs ({total_general_time /total_execution_time *100:.1f}%)"
        )
        report.append(
            f"  MatMul Operations: {total_mat_mul_time:,.2f} μs ({total_mat_mul_time /total_execution_time *100:.1f}%)"
        )
        report.append(
            f"Average Time per Inference Step: {time_per_inference_step:,.2f} μs ({time_per_inference_step /1000:.2f} ms)"
        )
        report.append("")
        report.append(
            f"Unique General Operation Types: {len(global_stats.general_stats)}"
        )
        report.append(
            f"Unique MatMul Operation Types: {len(global_stats.mat_mul_stats)}"
        )

        return "\n".join(report)

    def generate_markdown_report(
        self, global_stats: GlobalStats, output_dir: str = "."
    ) -> str:
        """Generate a comprehensive markdown report."""
        report = []

        report.append("# Vulkan Profiling Global Analysis Report")
        report.append("")

        # Model Information
        report.append("## Model Information")
        report.append("")
        report.append(f"- **Model File:** {global_stats.model_info.model_file}")
        report.append(f"- **Model Name:** {global_stats.model_info.model_name}")
        report.append(f"- **Architecture:** {global_stats.model_info.architecture}")
        report.append(f"- **Model Size:** {global_stats.model_info.size_label}")
        if global_stats.model_info.layer_count > 0:
            report.append(f"- **Layers:** {global_stats.model_info.layer_count}")
        if global_stats.model_info.context_length > 0:
            report.append(
                f"- **Context Length:** {global_stats.model_info.context_length:,}"
            )
        if global_stats.model_info.embedding_length > 0:
            report.append(
                f"- **Embedding Dimension:** {global_stats.model_info.embedding_length:,}"
            )
        if global_stats.model_info.attention_heads > 0:
            if (
                global_stats.model_info.attention_heads_kv > 0
                and global_stats.model_info.attention_heads_kv
                != global_stats.model_info.attention_heads
            ):
                report.append(
                    f"- **Attention Heads:** {global_stats.model_info.attention_heads} ({global_stats.model_info.attention_heads_kv} KV heads)"
                )
            else:
                report.append(
                    f"- **Attention Heads:** {global_stats.model_info.attention_heads}"
                )
        if global_stats.model_info.vocab_size > 0:
            report.append(
                f"- **Vocabulary Size:** {global_stats.model_info.vocab_size:,}"
            )
        report.append(f"- **Quantization:** {global_stats.model_info.quantization}")
        if global_stats.model_info.file_size:
            report.append(f"- **File Size:** {global_stats.model_info.file_size}")
        if global_stats.model_info.quantized_by:
            report.append(f"- **Quantized By:** {global_stats.model_info.quantized_by}")

        # Show tensor types breakdown
        if global_stats.model_info.tensor_types:
            report.append("")
            report.append("**Tensor Types:**")
            for tensor_type, count in sorted(
                global_stats.model_info.tensor_types.items(),
                key=lambda x: x[1],
                reverse=True,
            ):
                report.append(f"- {tensor_type}: {count:,} tensors")
        report.append("")

        # Device Allocation Information
        total_ops_device = (
            global_stats.device_allocation.gpu_operations
            + global_stats.device_allocation.cpu_operations
        )
        gpu_percentage = (
            (global_stats.device_allocation.gpu_operations / total_ops_device * 100)
            if total_ops_device > 0
            else 0
        )
        cpu_percentage = (
            (global_stats.device_allocation.cpu_operations / total_ops_device * 100)
            if total_ops_device > 0
            else 0
        )

        report.append("## Device Allocation Summary")
        report.append("")
        report.append(f"- **GPU Device:** {global_stats.device_allocation.gpu_device}")
        report.append(f"- **GPU Layers:** {global_stats.device_allocation.gpu_layers}")
        report.append(
            f"- **GPU Operations:** {global_stats.device_allocation.gpu_operations:,} ({gpu_percentage:.1f}%)"
        )
        report.append(
            f"- **CPU Operations:** {global_stats.device_allocation.cpu_operations:,} ({cpu_percentage:.1f}%)"
        )

        # Show CPU operation types if any
        if global_stats.device_allocation.cpu_operation_types:
            report.append("")
            report.append("**CPU Operations by Type:**")
            for op_type, count in sorted(
                global_stats.device_allocation.cpu_operation_types.items(),
                key=lambda x: x[1],
                reverse=True,
            ):
                report.append(f"- {op_type}: {count:,} operations")
        report.append("")

        # General Operations Statistics
        report.append("## General Operations Statistics")
        report.append("*(Ordered by average execution time - highest first)*")
        report.append("")

        # Sort by average time (highest first)
        sorted_general = sorted(
            global_stats.general_stats.items(),
            key=lambda x: x[1]["avg_time_mean"],
            reverse=True,
        )

        for operation, stats in sorted_general:
            report.append(f"### {operation}")
            report.append(f"- **Total Operations:** {stats['total_operations']:,}")
            report.append(
                f"- **Appeared in:** {stats['total_inference_steps']}/{global_stats.total_inference_steps} inference steps"
            )
            report.append(
                f"- **Average Time:** {stats['avg_time_mean']:.2f} ± {stats['avg_time_stdev']:.2f} μs"
            )
            report.append(
                f"- **Min/Max:** {stats['avg_time_min']:.2f} / {stats['avg_time_max']:.2f} μs, Median: {stats['avg_time_median']:.2f} μs"
            )
            report.append(
                f"- **Operations per Inference Step:** {stats['count_mean']:.1f}"
            )
            report.append("")

        # MatMul Timing Statistics
        report.append("## MatMul Timing Statistics")
        report.append("*(Ordered by average execution time - highest first)*")
        report.append("")

        # Sort by average time (highest first)
        sorted_mat_mul = sorted(
            global_stats.mat_mul_stats.items(),
            key=lambda x: x[1]["avg_time_mean"],
            reverse=True,
        )

        for operation, stats in sorted_mat_mul:
            report.append(f"### {operation}")
            report.append(f"- **Total Operations:** {stats['total_operations']:,}")
            report.append(
                f"- **Appeared in:** {stats['total_inference_steps']}/{global_stats.total_inference_steps} inference steps"
            )
            report.append(
                f"- **Average Time:** {stats['avg_time_mean']:.2f} ± {stats['avg_time_stdev']:.2f} μs"
            )
            report.append(
                f"- **Min/Max:** {stats['avg_time_min']:.2f} / {stats['avg_time_max']:.2f} μs, Median: {stats['avg_time_median']:.2f} μs"
            )
            report.append(
                f"- **Operations per Inference Step:** {stats['count_mean']:.1f}"
            )
            report.append("")

        # MatMul Tensor Size Normalized Analysis
        if global_stats.mat_mul_normalized:
            report.append("## MatMul Tensor Size Normalized Analysis")
            report.append(
                "*(Weighted mean normalized time by operation type and tensor types)*"
            )
            report.append("")

            # Sort by mean normalized time (most efficient first - lowest time per operation)
            sorted_normalized = sorted(
                global_stats.mat_mul_normalized.items(),
                key=lambda x: x[1]["mean_normalized_time"],
            )

            for group_key, stats in sorted_normalized:
                report.append(f"### {group_key}")
                report.append(
                    f"- **Weighted Mean Normalized Time:** {stats['mean_normalized_time']:.9f} μs per tensor element"
                )
                report.append(
                    f"- **Based on:** {stats['total_operations']:,} total operations across {stats['sample_count']} matrix size variations"
                )
                report.append("")

            # Create and include plot for markdown report
            plot_filename = self._create_normalized_time_plot(global_stats, output_dir)
            if plot_filename and os.path.exists(plot_filename):
                plot_basename = os.path.basename(plot_filename)
                report.append("### Efficiency Comparison Chart")
                report.append(f"![MatMul Normalized Time Efficiency]({plot_basename})")
                report.append("")
                report.append(
                    "*Chart shows weighted mean normalized time for each operation type and tensor type combination. Lower values indicate higher efficiency.*"
                )
                report.append("")

        # Top operations by total time
        report.append("## Top Operations by Total Execution Time")
        report.append("")

        # Calculate total time for each operation (total_ops * avg_time_mean)
        total_times = []
        for operation, stats in global_stats.general_stats.items():
            total_time = stats["total_operations"] * stats["avg_time_mean"]
            total_times.append(
                (
                    operation,
                    total_time,
                    stats["total_operations"],
                    stats["avg_time_mean"],
                )
            )

        for operation, stats in global_stats.mat_mul_stats.items():
            total_time = stats["total_operations"] * stats["avg_time_mean"]
            total_times.append(
                (
                    operation,
                    total_time,
                    stats["total_operations"],
                    stats["avg_time_mean"],
                )
            )

        total_times.sort(key=lambda x: x[1], reverse=True)

        for i, (operation, total_time, total_ops, avg_time) in enumerate(
            total_times[:20]
        ):
            report.append(f"### {operation}")
            report.append(f"- **Total Operations:** {total_ops:,}")
            report.append(
                f"- **Total Execution Time:** {total_time:,.2f} μs ({total_time /1000:.2f} ms)"
            )
            report.append(f"- **Average Time:** {avg_time:.2f} μs")
            report.append("")

        # Summary statistics
        report.append("## Timing Summary Statistics")
        report.append("*(Based on profiled operations only)*")
        report.append("")

        total_general_ops = sum(
            stats["total_operations"] for stats in global_stats.general_stats.values()
        )
        total_mat_mul_ops = sum(
            stats["total_operations"] for stats in global_stats.mat_mul_stats.values()
        )
        total_all_ops = total_general_ops + total_mat_mul_ops

        # Calculate percentages
        general_percentage = (
            (total_general_ops / total_all_ops * 100) if total_all_ops > 0 else 0
        )
        mat_mul_percentage = (
            (total_mat_mul_ops / total_all_ops * 100) if total_all_ops > 0 else 0
        )

        # Calculate total execution times
        total_general_time = sum(
            stats["total_operations"] * stats["avg_time_mean"]
            for stats in global_stats.general_stats.values()
        )
        total_mat_mul_time = sum(
            stats["total_operations"] * stats["avg_time_mean"]
            for stats in global_stats.mat_mul_stats.values()
        )
        total_execution_time = total_general_time + total_mat_mul_time

        # Calculate time per inference step
        time_per_inference_step = (
            total_execution_time / global_stats.total_inference_steps
            if global_stats.total_inference_steps > 0
            else 0
        )

        report.append(
            f"- **Total General Operations:** {total_general_ops:,} ({general_percentage:.1f}%)"
        )
        report.append(
            f"- **Total MatMul Operations:** {total_mat_mul_ops:,} ({mat_mul_percentage:.1f}%)"
        )
        report.append(f"- **Total Profiled Operations:** {total_all_ops:,}")
        report.append(
            f"- **Non-profiled Operations:** {total_ops_device - total_all_ops:,} (setup, memory management, etc.)"
        )
        report.append("")
        report.append(
            f"- **Total Execution Time:** {total_execution_time:,.2f} μs ({total_execution_time /1000:.2f} ms)"
        )
        report.append(
            f"  - General Operations: {total_general_time:,.2f} μs ({total_general_time /total_execution_time *100:.1f}%)"
        )
        report.append(
            f"  - MatMul Operations: {total_mat_mul_time:,.2f} μs ({total_mat_mul_time /total_execution_time *100:.1f}%)"
        )
        report.append(
            f"- **Average Time per Inference Step:** {time_per_inference_step:,.2f} μs ({time_per_inference_step /1000:.2f} ms)"
        )
        report.append("")
        report.append(
            f"- **Unique General Operation Types:** {len(global_stats.general_stats)}"
        )
        report.append(
            f"- **Unique MatMul Operation Types:** {len(global_stats.mat_mul_stats)}"
        )

        return "\n".join(report)

    def convert_markdown_to_html(
        self, markdown_content: str, html_filename: str, global_stats: GlobalStats
    ) -> bool:
        """Convert markdown content to HTML with base64-embedded images."""
        try:
            # Generate base64-encoded plot
            plot_base64 = self._create_normalized_time_plot(
                global_stats, return_base64=True
            )

            # Add basic CSS styling for better appearance
            css_style = """
            <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 40px auto;
                max-width: 1200px;
                line-height: 1.6;
                color: #333;
            }
            h1 {
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
                text-align: center;
            }
            h2 {
                color: #34495e;
                border-bottom: 2px solid #bdc3c7;
                padding-bottom: 8px;
                margin-top: 25px;
                margin-bottom: 15px;
            }
            h3 {
                color: #7f8c8d;
                margin-top: 15px;
                margin-bottom: 10px;
                padding-left: 10px;
                border-left: 4px solid #3498db;
            }
            code {
                background-color: #f8f9fa;
                padding: 2px 6px;
                border-radius: 4px;
                font-family: 'Courier New', monospace;
            }
            pre {
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 8px;
                overflow-x: auto;
                border: 1px solid #e1e8ed;
            }
            ul, ol { margin: 10px 0; }
            li { margin: 4px 0; }
            strong { color: #2c3e50; font-weight: 600; }
            em { color: #7f8c8d; font-style: italic; }
            img {
                max-width: 100%;
                height: auto;
                margin: 15px auto;
                display: block;
                border: 1px solid #ddd;
                border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            }
            .toc {
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 8px;
                margin: 15px 0;
            }
            </style>
            """

            # Convert markdown-style formatting to basic HTML
            html_content = markdown_content

            # Convert markdown headers
            html_content = re.sub(
                r"^# (.+)$", r"<h1>\1</h1>", html_content, flags=re.MULTILINE
            )
            html_content = re.sub(
                r"^## (.+)$", r"<h2>\1</h2>", html_content, flags=re.MULTILINE
            )
            html_content = re.sub(
                r"^### (.+)$", r"<h3>\1</h3>", html_content, flags=re.MULTILINE
            )

            # Convert bold text
            html_content = re.sub(
                r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", html_content
            )

            # Convert italic text
            html_content = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", html_content)

            # Convert bullet points
            # First handle indented sub-items (for execution time breakdown)
            html_content = re.sub(
                r"^  - (General Operations:.*)$",
                r'<li style="margin-left: 20px;">\1</li>',
                html_content,
                flags=re.MULTILINE,
            )
            html_content = re.sub(
                r"^  - (MatMul Operations:.*)$",
                r'<li style="margin-left: 20px;">\1</li>',
                html_content,
                flags=re.MULTILINE,
            )
            # Then handle regular bullet points
            html_content = re.sub(
                r"^- (.+)$", r"<li>\1</li>", html_content, flags=re.MULTILINE
            )

            # Wrap consecutive <li> elements in <ul> tags
            html_content = re.sub(
                r"(<li>.*?</li>)\s*(?=\n[^<]|\n$)",
                r"<ul>\1</ul>",
                html_content,
                flags=re.DOTALL,
            )
            html_content = re.sub(r"</li>\s*<li>", r"</li><li>", html_content)

            # Convert images - replace external image references with base64 data
            if plot_base64:
                html_content = re.sub(
                    r"!\[([^\]]*)\]\(matmul_normalized_times\.png\)",
                    f'<img src="{plot_base64}" alt="\\1" title="\\1">',
                    html_content,
                )

            # Convert any remaining images (fallback)
            html_content = re.sub(
                r"!\[([^\]]*)\]\(([^)]+)\)",
                r'<img src="\2" alt="\1" title="\1">',
                html_content,
            )

            # Clean up line breaks and spacing
            # Remove empty lines within sections
            html_content = re.sub(r"\n\s*\n", "\n", html_content)

            # Convert remaining line breaks, but be more selective
            # Don't add breaks after HTML tags
            html_content = re.sub(r"\n(?!<)", "<br>\n", html_content)

            # Remove breaks before and after HTML block elements
            html_content = re.sub(r"<br>\s*(<h[1-6])", r"\1", html_content)
            html_content = re.sub(
                r"(<h[1-6][^>]*>.*?</h[1-6]>)<br>", r"\1", html_content
            )
            html_content = re.sub(r"<br>\s*(<ul>|</ul>)", r"\1", html_content)
            html_content = re.sub(r"(<ul>|</ul>)<br>", r"\1", html_content)

            # Clean up excessive breaks
            html_content = re.sub(r"(<br>\s*){3,}", r"<br><br>", html_content)
            html_content = re.sub(r"(<br>\s*){2}(<h[1-6])", r"<br>\2", html_content)

            full_html = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Vulkan Profiling Analysis Report</title>
                {css_style}
            </head>
            <body>
                {html_content}
            </body>
            </html>
            """

            with open(html_filename, "w", encoding="utf-8") as f:
                f.write(full_html)

            return True

        except Exception as e:
            logging.error(f"Error converting markdown to HTML: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Vulkan profiling results from log file"
    )
    parser.add_argument(
        "log_file",
        nargs="?",
        default="log.txt",
        help="Path to log file (default: log.txt)",
    )
    parser.add_argument(
        "--report-output",
        help="Save report to specified file (default: print to stdout)",
    )
    parser.add_argument(
        "--markdown-output", help="Save markdown report to specified file"
    )
    parser.add_argument(
        "--html-output",
        help="Save HTML report to specified file (converted from markdown with embedded images)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s"
    )

    try:
        analyzer = VulkanProfilingAnalyzer(args.log_file)
        logging.info(f"Analyzing log file: {args.log_file}")

        analyzer.parse_log_file()
        global_stats = analyzer.compute_global_statistics()
        report = analyzer.generate_report(global_stats)

        if args.report_output:
            with open(args.report_output, "w") as f:
                f.write(report)
            logging.info(f"Report saved to: {args.report_output}")
        else:
            sys.stdout.write("\n" + report)

        if args.markdown_output:
            markdown_dir = (
                os.path.dirname(os.path.abspath(args.markdown_output))
                if os.path.dirname(args.markdown_output)
                else "."
            )
            markdown_report = analyzer.generate_markdown_report(
                global_stats, markdown_dir
            )
            with open(args.markdown_output, "w") as f:
                f.write(markdown_report)
            logging.info(f"Markdown report saved to: {args.markdown_output}")

        if args.html_output:
            html_dir = (
                os.path.dirname(os.path.abspath(args.html_output))
                if os.path.dirname(args.html_output)
                else "."
            )
            markdown_report = analyzer.generate_markdown_report(global_stats, html_dir)
            success = analyzer.convert_markdown_to_html(
                markdown_report, args.html_output, global_stats
            )
            if success:
                logging.info(
                    f"HTML report saved to: {args.html_output} (self-contained with embedded images)"
                )
            else:
                logging.error("HTML generation failed. See error messages above.")

    except FileNotFoundError:
        logging.error(f"Log file '{args.log_file}' not found.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
