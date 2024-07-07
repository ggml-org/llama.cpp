#!/usr/bin/env python3
from __future__ import annotations

import uuid
import hashlib

import logging
import argparse
import os
import sys
from pathlib import Path

from tqdm import tqdm

# Necessary to load the local gguf package
if "NO_LOCAL_GGUF" not in os.environ and (Path(__file__).parent.parent.parent / 'gguf-py').exists():
    sys.path.insert(0, str(Path(__file__).parent.parent))

from gguf import GGUFReader  # noqa: E402


logger = logging.getLogger("gguf-hash")

# UUID_NAMESPACE_LLAMA_CPP = uuid.uuid5(uuid.NAMESPACE_URL, 'en.wikipedia.org/wiki/Llama.cpp')
UUID_NAMESPACE_LLAMA_CPP = uuid.UUID('ef001206-dadc-5f6d-a15f-3359e577d4e5')


# For more information about what field.parts and field.data represent,
# please see the comments in the modify_gguf.py example.
def gguf_hash(reader: GGUFReader, filename: str, disable_progress_bar) -> None:
    sha1 = hashlib.sha1()
    uuidv5_sha1 = hashlib.sha1()
    uuidv5_sha1.update(UUID_NAMESPACE_LLAMA_CPP.bytes)

    # Total Weight Calculation For Progress Bar
    total_weights = 0
    for n, tensor in enumerate(reader.tensors, 1):

        # We don't need these
        if tensor.name.endswith((".attention.masked_bias", ".attention.bias", ".rotary_emb.inv_freq")):
            continue

        # Calculate Tensor Volume
        sum_weights_in_tensor = 1
        for dim in tensor.shape:
            sum_weights_in_tensor *= dim
        total_weights += sum_weights_in_tensor

    # Hash Progress Bar
    bar = tqdm(desc="Hashing", total=total_weights, unit="weights", unit_scale=True, disable=disable_progress_bar)

    # Hashing Process
    for n, tensor in enumerate(reader.tensors, 1):

        # We don't need these
        if tensor.name.endswith((".attention.masked_bias", ".attention.bias", ".rotary_emb.inv_freq")):
            continue

        # Progressbar
        sum_weights_in_tensor = 1
        for dim in tensor.shape:
            sum_weights_in_tensor *= dim
        bar.update(sum_weights_in_tensor)

        sha1_layer = hashlib.sha1()
        sha1_layer.update(tensor.data)
        sha1.update(tensor.data)
        uuidv5_sha1.update(tensor.data)
        print("sha1    {0}  {1}:{2}".format(sha1_layer.hexdigest(), filename, tensor.name)) # noqa: NP100

    # Flush Hash Progress Bar
    bar.close()

    # Display Hash Output
    print("sha1    {0}  {1}".format(sha1.hexdigest(), filename)) # noqa: NP100
    print("UUIDv5  {0}  {1}".format(uuid.UUID(bytes=uuidv5_sha1.digest()[:16], version=5), filename)) # noqa: NP100


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump GGUF file metadata")
    parser.add_argument("model",         type=str,            help="GGUF format model filename")
    parser.add_argument("--verbose",     action="store_true", help="increase output verbosity")
    parser.add_argument("--progressbar", action="store_true", help="enable progressbar")
    args = parser.parse_args(None if len(sys.argv) > 1 else ["--help"])
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    reader = GGUFReader(args.model, 'r')
    gguf_hash(reader, args.model, not args.progressbar)


if __name__ == '__main__':
    main()
