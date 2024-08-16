import argparse
import os
import torch
from transformers import AutoModel, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", help="Path to MiniCPM-V model")
args = ap.parse_args()

# find the model part that includes the the multimodal projector weights
model = AutoModel.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
checkpoint = model.state_dict()

# get a list of mm tensor names
mm_tensors = [k for k, v in checkpoint.items() if k.startswith("resampler")]

# store these tensors in a new dictionary and torch.save them
projector = {name: checkpoint[name].float() for name in mm_tensors}
torch.save(projector, f"{args.model}/minicpmv.projector")

clip_tensors = [k for k, v in checkpoint.items() if k.startswith("vpm")]
if len(clip_tensors) > 0:
    clip = {name.replace("vpm.", ""): checkpoint[name].float() for name in clip_tensors}
    torch.save(clip, f"{args.model}/minicpmv.clip")

    # added tokens should be removed to be able to convert Mistral models
    if os.path.exists(f"{args.model}/added_tokens.json"):
        with open(f"{args.model}/added_tokens.json", "w") as f:
            f.write("{}\n")

config = model.llm.config
config.auto_map = {
    "AutoConfig": "configuration_minicpm.MiniCPMConfig",
    "AutoModel": "modeling_minicpm.MiniCPMModel",
    "AutoModelForCausalLM": "modeling_minicpm.MiniCPMForCausalLM",
    "AutoModelForSeq2SeqLM": "modeling_minicpm.MiniCPMForCausalLM",
    "AutoModelForSequenceClassification": "modeling_minicpm.MiniCPMForSequenceClassification"
}
model.llm.save_pretrained(f"{args.model}/model")
tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
tok.save_pretrained(f"{args.model}/model")

print("Done!")
print(f"Now you can convert {args.model} to a regular LLaMA GGUF file.")
print(f"Also, use {args.model}/minicpmv.projector to prepare a minicpmv-encoder.gguf file.")
