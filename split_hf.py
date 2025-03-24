from safetensors import safe_open
from safetensors.torch import save_file

safetensors_path = "my-models/csm/model.safetensors"

# Open the original SafeTensors file
with safe_open(safetensors_path, framework="pt", device="cpu") as f:
    tensors = {key: f.get_tensor(key) for key in f.keys()}

# Identify tensors belonging to each model
backbone_tensors = {k.replace("backbone.", "model."): v for k, v in tensors.items() if any(x in k for x in ["backbone.", "text_"])}
decoder_tensors = {k.replace("decoder.", "model."): v for k, v in tensors.items() if any(x in k for x in ["decoder.", "audio_", "projection.", "codebook0_head."])}

save_file(backbone_tensors, "backbone.safetensors")
print(f"Saved backbone model with {len(backbone_tensors)} tensors.")

save_file(decoder_tensors, "decoder.safetensors")
print(f"Saved decoder model with {len(decoder_tensors)} tensors.")
