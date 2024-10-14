import os

from importlib import import_module


os.environ["NO_LOCAL_GGUF"] = "TRUE"

gguf_convert_endian_entrypoint = import_module("scripts.gguf-convert-endian").main
gguf_dump_entrypoint           = import_module("scripts.gguf-dump").main
gguf_set_metadata_entrypoint   = import_module("scripts.gguf-set-metadata").main
gguf_new_metadata_entrypoint   = import_module("scripts.gguf-new-metadata").main

del import_module, os
