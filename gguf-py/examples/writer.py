#!/usr/bin/env python3
import sys
from pathlib import Path

import numpy as np

# Necessary to load the local gguf package
sys.path.insert(0, str(Path(__file__).parent.parent))

from gguf import GGUFWriter, GGMLQuantizationType  # noqa: E402


# Create a tiny GGUF model for testing SmarterQuant
def create_tiny_model_for_sq_test() -> None:
    # Output file will be in the root directory for easy access by llama-quantize
    gguf_writer = GGUFWriter("../../tiny_model.gguf", "llama") # arch is set here

    # Minimal metadata
    gguf_writer.add_block_count(1) # This should represent layer count for llama arch
    gguf_writer.add_context_length(128) # Dummy
    embedding_length = 512
    head_count = 1
    gguf_writer.add_embedding_length(embedding_length)
    gguf_writer.add_feed_forward_length(1024) # Dummy
    gguf_writer.add_head_count(head_count)
    gguf_writer.add_head_count_kv(1) # Dummy
    gguf_writer.add_rope_dimension_count(embedding_length // head_count)
    gguf_writer.add_layer_norm_rms_eps(1e-5) # Required for llama arch
    gguf_writer.add_file_type(1) # F16 == 1 (GGML_FTYPE_MOSTLY_F16)

    # Tensor to be targeted by SmarterQuant
    # Dimensions: 4 rows, 512 columns.
    # 512 columns = two 256-column blocks.
    tensor_data_sq = np.random.rand(4, 512).astype(np.float32)
    gguf_writer.add_tensor("blk.0.attn_q.weight", tensor_data_sq)

    # Another dummy tensor
    other_tensor_data = np.random.rand(4, 256).astype(np.float32)
    gguf_writer.add_tensor("blk.0.ffn_down.weight", other_tensor_data)

    gguf_writer.add_uint32("answer", 42) # Dummy KV pair

    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()

    gguf_writer.close()
    print("Created ../../tiny_model.gguf")

if __name__ == '__main__':
    create_tiny_model_for_sq_test()
