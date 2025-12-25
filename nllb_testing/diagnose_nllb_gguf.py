#!/usr/bin/env python3
"""
Diagnose NLLB GGUF model file
"""
import gguf

print("=" * 80)
print("NLLB GGUF File Diagnostics")
print("=" * 80)

reader = gguf.GGUFReader('nllb-600m.gguf')

print("\n1. Architecture and Basic Info:")
print("-" * 40)
arch = reader.fields.get('general.architecture')
if arch:
    print(f"Architecture: {bytes(arch.parts[arch.data[0]]).decode('utf-8')}")

for key in ['general.name', 'general.type', 'general.file_type']:
    if key in reader.fields:
        field = reader.fields[key]
        if field.types[0] == gguf.GGUFValueType.STRING:
            val = bytes(field.parts[field.data[0]]).decode('utf-8')
        else:
            val = field.parts[field.data[0]]
        print(f"{key}: {val}")

print("\n2. NLLB-specific Parameters:")
print("-" * 40)
nllb_keys = [k for k in reader.fields.keys() if 'nllb' in k.lower()]
for key in sorted(nllb_keys):
    field = reader.fields[key]
    if len(field.data) > 0:
        val = field.parts[field.data[0]] if len(field.parts) > 0 else field.data[0]
        print(f"{key}: {val}")

print("\n3. Attention and Normalization:")
print("-" * 40)
attn_keys = [k for k in reader.fields.keys() if 'attention' in k.lower() or 'norm' in k.lower()]
for key in sorted(attn_keys):
    field = reader.fields[key]
    if len(field.data) > 0:
        val = field.parts[field.data[0]] if len(field.parts) > 0 else field.data[0]
        print(f"{key}: {val}")

print("\n4. Decoder Parameters:")
print("-" * 40)
dec_keys = [k for k in reader.fields.keys() if 'decoder' in k.lower()]
for key in sorted(dec_keys):
    field = reader.fields[key]
    if len(field.data) > 0:
        val = field.parts[field.data[0]] if len(field.parts) > 0 else field.data[0]
        print(f"{key}: {val}")

print("\n5. Tokenizer Parameters:")
print("-" * 40)
tok_keys = [k for k in reader.fields.keys() if 'tokenizer' in k.lower() and 'tokens' not in k]
for key in sorted(tok_keys):
    field = reader.fields[key]
    if len(field.data) > 0:
        val = field.parts[field.data[0]] if len(field.parts) > 0 else field.data[0]
        if isinstance(val, bytes):
            val = val.decode('utf-8')
        print(f"{key}: {val}")

print("\n6. Sample Tensors (first 10):")
print("-" * 40)
for i, tensor in enumerate(reader.tensors[:10]):
    print(f"{tensor.name}: shape={tensor.shape}, dtype={tensor.tensor_type}")

print("\n7. Tensor Name Patterns:")
print("-" * 40)
encoder_tensors = [t.name for t in reader.tensors if t.name.startswith('enc.')]
decoder_tensors = [t.name for t in reader.tensors if t.name.startswith('dec.')]
other_tensors = [t.name for t in reader.tensors if not t.name.startswith('enc.') and not t.name.startswith('dec.')]

print(f"Encoder tensors: {len(encoder_tensors)}")
print(f"Decoder tensors: {len(decoder_tensors)}")
print(f"Other tensors: {len(other_tensors)}")

if encoder_tensors:
    print(f"\nSample encoder tensors:")
    for t in encoder_tensors[:5]:
        print(f"  {t}")

if decoder_tensors:
    print(f"\nSample decoder tensors:")
    for t in decoder_tensors[:5]:
        print(f"  {t}")

if other_tensors:
    print(f"\nOther tensors:")
    for t in other_tensors:
        print(f"  {t}")

print("\n" + "=" * 80)


