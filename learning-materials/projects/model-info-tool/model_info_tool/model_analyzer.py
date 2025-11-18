"""
Model Analyzer
Analyzes GGUF models and provides detailed information.
"""

from pathlib import Path
from typing import Dict, Any, List
from .gguf_reader import GGUFReader


class ModelAnalyzer:
    """Analyze GGUF models and extract information."""

    # Quantization type names
    QUANT_TYPES = {
        0: "F32",
        1: "F16",
        2: "Q4_0",
        3: "Q4_1",
        6: "Q5_0",
        7: "Q5_1",
        8: "Q8_0",
        9: "Q8_1",
        10: "Q2_K",
        11: "Q3_K_S",
        12: "Q3_K_M",
        13: "Q3_K_L",
        14: "Q4_K_S",
        15: "Q4_K_M",
        16: "Q5_K_S",
        17: "Q5_K_M",
        18: "Q6_K",
        19: "Q8_K",
    }

    def __init__(self, model_path: str):
        """
        Initialize model analyzer.

        Args:
            model_path: Path to the GGUF model file
        """
        self.model_path = Path(model_path)
        self.reader = GGUFReader(str(model_path))
        self.model_data = self.reader.read()

    def get_basic_info(self) -> Dict[str, Any]:
        """Get basic model information."""
        return {
            "name": self.reader.get_model_name(),
            "architecture": self.reader.get_model_architecture(),
            "file_path": str(self.model_path),
            "file_name": self.model_path.name,
            "file_size_mb": round(self.reader.get_file_size_mb(), 2),
            "file_size_gb": round(self.reader.get_file_size_gb(), 3),
            "parameter_count": self.reader.get_parameter_count(),
            "parameter_count_formatted": self._format_parameter_count(
                self.reader.get_parameter_count()
            ),
            "tensor_count": len(self.reader.tensor_info)
        }

    def get_architecture_details(self) -> Dict[str, Any]:
        """Get detailed architecture information."""
        arch = self.reader.get_model_architecture()
        details = {
            "architecture": arch,
            "embedding_length": self.reader.get_metadata_value(f"{arch}.embedding_length"),
            "block_count": self.reader.get_metadata_value(f"{arch}.block_count"),
            "feed_forward_length": self.reader.get_metadata_value(f"{arch}.feed_forward_length"),
            "attention.head_count": self.reader.get_metadata_value(f"{arch}.attention.head_count"),
            "attention.head_count_kv": self.reader.get_metadata_value(f"{arch}.attention.head_count_kv"),
            "attention.layer_norm_rms_epsilon": self.reader.get_metadata_value(f"{arch}.attention.layer_norm_rms_epsilon"),
            "rope.dimension_count": self.reader.get_metadata_value(f"{arch}.rope.dimension_count"),
            "rope.freq_base": self.reader.get_metadata_value(f"{arch}.rope.freq_base"),
        }

        # Remove None values
        return {k: v for k, v in details.items() if v is not None}

    def get_tokenizer_info(self) -> Dict[str, Any]:
        """Get tokenizer information."""
        return {
            "model": self.reader.get_metadata_value("tokenizer.ggml.model"),
            "vocab_size": self.reader.get_metadata_value("tokenizer.ggml.vocab_size"),
            "bos_token_id": self.reader.get_metadata_value("tokenizer.ggml.bos_token_id"),
            "eos_token_id": self.reader.get_metadata_value("tokenizer.ggml.eos_token_id"),
            "unknown_token_id": self.reader.get_metadata_value("tokenizer.ggml.unknown_token_id"),
            "padding_token_id": self.reader.get_metadata_value("tokenizer.ggml.padding_token_id"),
        }

    def get_quantization_info(self) -> Dict[str, Any]:
        """Get quantization information."""
        # Count tensor types
        type_counts = {}
        total_size = 0

        for tensor in self.reader.tensor_info:
            tensor_type = tensor.get("type", 0)
            type_name = self.QUANT_TYPES.get(tensor_type, f"UNKNOWN_{tensor_type}")

            if type_name not in type_counts:
                type_counts[type_name] = 0
            type_counts[type_name] += 1

        return {
            "quantization_version": self.reader.get_metadata_value("general.quantization_version"),
            "file_type": self.reader.get_metadata_value("general.file_type"),
            "tensor_types": type_counts,
            "predominant_type": max(type_counts, key=type_counts.get) if type_counts else "unknown"
        }

    def get_tensor_details(self) -> List[Dict[str, Any]]:
        """Get detailed tensor information."""
        tensors = []
        for tensor in self.reader.tensor_info:
            tensor_type = tensor.get("type", 0)
            dims = tensor.get("dimensions", [])

            # Calculate parameters
            params = 1
            for dim in dims:
                params *= dim

            tensors.append({
                "name": tensor.get("name"),
                "type": self.QUANT_TYPES.get(tensor_type, f"UNKNOWN_{tensor_type}"),
                "dimensions": dims,
                "shape": " x ".join(map(str, dims)),
                "parameters": params,
                "offset": tensor.get("offset")
            })

        return tensors

    def get_memory_requirements(self) -> Dict[str, Any]:
        """Estimate memory requirements for different scenarios."""
        file_size_mb = self.reader.get_file_size_mb()

        # Rough estimates
        return {
            "model_size_mb": round(file_size_mb, 2),
            "model_size_gb": round(file_size_mb / 1024, 3),
            "estimated_ram_cpu_only_mb": round(file_size_mb * 1.2, 2),  # Model + overhead
            "estimated_ram_cpu_only_gb": round(file_size_mb * 1.2 / 1024, 3),
            "estimated_vram_gpu_full_mb": round(file_size_mb * 1.1, 2),  # Slightly less overhead on GPU
            "estimated_vram_gpu_full_gb": round(file_size_mb * 1.1 / 1024, 3),
            "note": "Estimates include model weights and minimal overhead. Actual usage depends on context size and batch size."
        }

    def get_context_info(self) -> Dict[str, Any]:
        """Get context length information."""
        arch = self.reader.get_model_architecture()

        return {
            "context_length": self.reader.get_metadata_value(f"{arch}.context_length"),
            "max_context_length": self.reader.get_metadata_value("tokenizer.ggml.max_context_length"),
        }

    def get_all_metadata(self) -> Dict[str, Any]:
        """Get all metadata from the model."""
        return self.reader.metadata.copy()

    def generate_summary(self) -> str:
        """Generate a human-readable summary of the model."""
        basic = self.get_basic_info()
        arch = self.get_architecture_details()
        quant = self.get_quantization_info()
        memory = self.get_memory_requirements()

        summary = []
        summary.append("=" * 60)
        summary.append("MODEL SUMMARY")
        summary.append("=" * 60)
        summary.append(f"Name: {basic['name']}")
        summary.append(f"Architecture: {basic['architecture']}")
        summary.append(f"Parameters: {basic['parameter_count_formatted']}")
        summary.append(f"File Size: {basic['file_size_gb']} GB ({basic['file_size_mb']} MB)")
        summary.append(f"Quantization: {quant['predominant_type']}")
        summary.append(f"Tensor Count: {basic['tensor_count']}")
        summary.append("")
        summary.append("ARCHITECTURE DETAILS")
        summary.append("-" * 60)
        for key, value in arch.items():
            if key != "architecture":
                summary.append(f"{key}: {value}")
        summary.append("")
        summary.append("MEMORY REQUIREMENTS (ESTIMATED)")
        summary.append("-" * 60)
        summary.append(f"Model Size: {memory['model_size_gb']} GB")
        summary.append(f"CPU-only RAM: ~{memory['estimated_ram_cpu_only_gb']} GB")
        summary.append(f"GPU-full VRAM: ~{memory['estimated_vram_gpu_full_gb']} GB")
        summary.append("")
        summary.append("=" * 60)

        return "\n".join(summary)

    def compare_with(self, other_model_path: str) -> Dict[str, Any]:
        """
        Compare this model with another model.

        Args:
            other_model_path: Path to another GGUF model

        Returns:
            Comparison information
        """
        other = ModelAnalyzer(other_model_path)

        this_info = self.get_basic_info()
        other_info = other.get_basic_info()

        return {
            "model_1": {
                "name": this_info["name"],
                "file": this_info["file_name"],
                "parameters": this_info["parameter_count_formatted"],
                "size_gb": this_info["file_size_gb"],
                "architecture": this_info["architecture"]
            },
            "model_2": {
                "name": other_info["name"],
                "file": other_info["file_name"],
                "parameters": other_info["parameter_count_formatted"],
                "size_gb": other_info["file_size_gb"],
                "architecture": other_info["architecture"]
            },
            "differences": {
                "size_difference_gb": round(
                    other_info["file_size_gb"] - this_info["file_size_gb"], 3
                ),
                "parameter_difference": other_info["parameter_count"] - this_info["parameter_count"],
                "same_architecture": this_info["architecture"] == other_info["architecture"]
            }
        }

    @staticmethod
    def _format_parameter_count(count: int) -> str:
        """Format parameter count in human-readable form."""
        if count >= 1_000_000_000:
            return f"{count / 1_000_000_000:.2f}B"
        elif count >= 1_000_000:
            return f"{count / 1_000_000:.2f}M"
        elif count >= 1_000:
            return f"{count / 1_000:.2f}K"
        else:
            return str(count)
