"""
GGUF File Reader
Reads and parses GGUF (GPT-Generated Unified Format) model files.
"""

import struct
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple
from enum import IntEnum


class GGUFValueType(IntEnum):
    """GGUF metadata value types."""
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


class GGUFReader:
    """Read and parse GGUF model files."""

    GGUF_MAGIC = 0x46554747  # 'GGUF' in ASCII
    GGUF_VERSION = 3

    def __init__(self, file_path: str):
        """
        Initialize GGUF reader.

        Args:
            file_path: Path to the GGUF file
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        self.file_size = self.file_path.stat().st_size
        self.metadata = {}
        self.tensor_info = []
        self.header_info = {}

    def read(self) -> Dict[str, Any]:
        """
        Read and parse the GGUF file.

        Returns:
            Dictionary containing file information
        """
        with open(self.file_path, 'rb') as f:
            # Read header
            magic = struct.unpack('<I', f.read(4))[0]
            if magic != self.GGUF_MAGIC:
                raise ValueError(f"Invalid GGUF file: incorrect magic number {hex(magic)}")

            version = struct.unpack('<I', f.read(4))[0]
            tensor_count = struct.unpack('<Q', f.read(8))[0]
            metadata_kv_count = struct.unpack('<Q', f.read(8))[0]

            self.header_info = {
                "magic": hex(magic),
                "version": version,
                "tensor_count": tensor_count,
                "metadata_kv_count": metadata_kv_count
            }

            # Read metadata key-value pairs
            for _ in range(metadata_kv_count):
                key, value = self._read_metadata_kv(f)
                self.metadata[key] = value

            # Read tensor information
            for _ in range(tensor_count):
                tensor_info = self._read_tensor_info(f)
                self.tensor_info.append(tensor_info)

        return {
            "file_path": str(self.file_path),
            "file_size": self.file_size,
            "header": self.header_info,
            "metadata": self.metadata,
            "tensor_count": len(self.tensor_info),
            "tensors": self.tensor_info
        }

    def _read_string(self, f) -> str:
        """Read a string from the file."""
        length = struct.unpack('<Q', f.read(8))[0]
        string_bytes = f.read(length)
        return string_bytes.decode('utf-8', errors='replace')

    def _read_metadata_kv(self, f) -> Tuple[str, Any]:
        """Read a metadata key-value pair."""
        key = self._read_string(f)
        value_type = GGUFValueType(struct.unpack('<I', f.read(4))[0])
        value = self._read_value(f, value_type)
        return key, value

    def _read_value(self, f, value_type: GGUFValueType) -> Any:
        """Read a value based on its type."""
        if value_type == GGUFValueType.UINT8:
            return struct.unpack('<B', f.read(1))[0]
        elif value_type == GGUFValueType.INT8:
            return struct.unpack('<b', f.read(1))[0]
        elif value_type == GGUFValueType.UINT16:
            return struct.unpack('<H', f.read(2))[0]
        elif value_type == GGUFValueType.INT16:
            return struct.unpack('<h', f.read(2))[0]
        elif value_type == GGUFValueType.UINT32:
            return struct.unpack('<I', f.read(4))[0]
        elif value_type == GGUFValueType.INT32:
            return struct.unpack('<i', f.read(4))[0]
        elif value_type == GGUFValueType.FLOAT32:
            return struct.unpack('<f', f.read(4))[0]
        elif value_type == GGUFValueType.UINT64:
            return struct.unpack('<Q', f.read(8))[0]
        elif value_type == GGUFValueType.INT64:
            return struct.unpack('<q', f.read(8))[0]
        elif value_type == GGUFValueType.FLOAT64:
            return struct.unpack('<d', f.read(8))[0]
        elif value_type == GGUFValueType.BOOL:
            return bool(struct.unpack('<B', f.read(1))[0])
        elif value_type == GGUFValueType.STRING:
            return self._read_string(f)
        elif value_type == GGUFValueType.ARRAY:
            return self._read_array(f)
        else:
            raise ValueError(f"Unknown value type: {value_type}")

    def _read_array(self, f) -> List[Any]:
        """Read an array value."""
        array_type = GGUFValueType(struct.unpack('<I', f.read(4))[0])
        array_length = struct.unpack('<Q', f.read(8))[0]
        return [self._read_value(f, array_type) for _ in range(array_length)]

    def _read_tensor_info(self, f) -> Dict[str, Any]:
        """Read tensor information."""
        name = self._read_string(f)
        n_dims = struct.unpack('<I', f.read(4))[0]
        dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
        tensor_type = struct.unpack('<I', f.read(4))[0]
        offset = struct.unpack('<Q', f.read(8))[0]

        return {
            "name": name,
            "dimensions": dims,
            "type": tensor_type,
            "offset": offset
        }

    def get_metadata_value(self, key: str, default=None) -> Any:
        """Get a specific metadata value."""
        return self.metadata.get(key, default)

    def get_model_architecture(self) -> str:
        """Get the model architecture."""
        return self.get_metadata_value("general.architecture", "unknown")

    def get_model_name(self) -> str:
        """Get the model name."""
        return self.get_metadata_value("general.name", "unknown")

    def get_parameter_count(self) -> int:
        """Estimate total parameter count."""
        total_params = 0
        for tensor in self.tensor_info:
            params = 1
            for dim in tensor["dimensions"]:
                params *= dim
            total_params += params
        return total_params

    def get_file_size_mb(self) -> float:
        """Get file size in megabytes."""
        return self.file_size / (1024 * 1024)

    def get_file_size_gb(self) -> float:
        """Get file size in gigabytes."""
        return self.file_size / (1024 * 1024 * 1024)
