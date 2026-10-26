"""Write a chat template into an existing GGUF (so llama-server needs no
--chat-template-file). Works for BF16X and any other quantized tensors.

    python add_chat_template_gguf.py <src.gguf> <template.jinja> <out.gguf>
"""
import sys

import numpy as np

sys.path.insert(0, r"F:\llama.cpp\gguf-py")
import gguf

src, tmpl_path, out = sys.argv[1:4]
tmpl = open(tmpl_path, encoding="utf-8").read()

r = gguf.GGUFReader(src)
arch = r.fields["general.architecture"].contents()
print("arch:", arch, "| tensors:", len(r.tensors))

SKIP = {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count",
        "general.architecture", "tokenizer.chat_template"}

w = gguf.GGUFWriter(out, arch)
for name, f in r.fields.items():
    if name in SKIP:
        continue
    sub = f.types[1] if len(f.types) > 1 else None
    w.add_key_value(name, f.contents(), f.types[0], sub)
w.add_key_value("tokenizer.chat_template", tmpl, gguf.GGUFValueType.STRING)

for t in r.tensors:
    raw_shape = list(t.data.shape)
    w.add_tensor(t.name, np.asarray(t.data), raw_shape=raw_shape, raw_dtype=t.tensor_type)

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print("written:", out)
