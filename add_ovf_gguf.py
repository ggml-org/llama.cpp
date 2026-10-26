# Add lossless BF16X overflow tables to an existing BF16X GGUF.
#   python add_ovf_gguf.py <source-f16.gguf> <template-bf16x.gguf> <out.gguf>
# The source F16 file supplies the exact weights; the template supplies the
# tensor layout/quantization choices. Every BF16X tensor gets a "<name>.ovf"
# I8 sibling holding the elements the 3-bit exponent delta clamps.
import sys

import numpy as np

sys.path.insert(0, r'F:\llama.cpp\gguf-py')
import gguf

sys.path.insert(0, r'F:\bfloat16x')
import bf16x_gguf_block as B

src_f16, tmpl, out = sys.argv[1:4]

r16 = gguf.GGUFReader(src_f16)
rt = gguf.GGUFReader(tmpl)

arch = rt.fields['general.architecture'].contents()
print('arch:', arch, '| tensors:', len(rt.tensors))

w = gguf.GGUFWriter(out, arch)
for name, f in rt.fields.items():
    if name == 'general.architecture' or name.startswith('GGUF.'):
        continue
    vtype = f.types[0]
    sub = f.types[1] if len(f.types) > 1 else None
    w.add_key_value(name, f.contents(), vtype, sub)

f16 = {t.name: t for t in r16.tensors}
n_ovf = 0
n_bytes = 0
for t in rt.tensors:
    shape = [int(x) for x in t.shape]
    raw_shape = list(t.data.shape)
    w.add_tensor(t.name, np.asarray(t.data), raw_shape=raw_shape, raw_dtype=t.tensor_type)
    if t.tensor_type != gguf.GGMLQuantizationType.BF16X:
        continue
    n_embd = shape[0]
    x = np.asarray(f16[t.name].data, dtype=np.float32).reshape(-1)
    rows = x.size // n_embd
    blob = B.pack_ovf(x.reshape(rows, n_embd))
    w.add_tensor(t.name + '.ovf', np.frombuffer(blob, dtype=np.uint8),
                 raw_shape=[len(blob)], raw_dtype=gguf.GGMLQuantizationType.I8)
    n_ovf += 1
    n_bytes += len(blob)

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print(f'added {n_ovf} overflow tables ({n_bytes/1024/1024:.1f} MiB) -> {out}')
