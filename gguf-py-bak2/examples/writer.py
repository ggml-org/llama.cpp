#!/usr/bin/env python3
import sys
from pathlib import Path

import numpy as np

# Necessary to load the local gguf package
sys.path.insert(0, str(Path(__file__).parent.parent))

from gguf import GGUFWriter  # noqa: E402


# Example usage:
def writer_example() -> None:
    # Example usage with a file
    gguf_writer = GGUFWriter("example.gguf", "llama")

    gguf_writer.add_architecture()
    gguf_writer.add_block_count(12)
    gguf_writer.add_uint32("answer", 42)  # Write a 32-bit integer
    gguf_writer.add_float32("answer_in_float", 42.0)  # Write a 32-bit float
    gguf_writer.add_custom_alignment(64)

    tensor1 = np.ones((32,), dtype=np.float32) * 100.0
    tensor2 = np.ones((64,), dtype=np.float32) * 101.0
    tensor3 = np.ones((96,), dtype=np.float32) * 102.0

    gguf_writer.add_tensor("tensor1", tensor1)
    gguf_writer.add_tensor("tensor2", tensor2)
    gguf_writer.add_tensor("tensor3", tensor3)

    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()

    gguf_writer.close()


if __name__ == '__main__':
    writer_example()
