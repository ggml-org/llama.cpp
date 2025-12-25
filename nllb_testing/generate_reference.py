#!/usr/bin/env python3
"""
Generate reference outputs from HuggingFace NLLB model.
This creates ground truth data for numerical verification.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import json
import os

print("=" * 80)
print("NLLB Reference Output Generator")
print("=" * 80)

# Create results directory
os.makedirs("results", exist_ok=True)

# Test sentences
test_sentences = [
    "eng_Latn Hello, how are you?",
    "eng_Latn The quick brown fox jumps over the lazy dog.",
    "eng_Latn Machine learning is transforming the world.",
]

# Target language
target_lang = "fra_Latn"

print("\n1. Loading HuggingFace NLLB model...")
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.eval()

print(f"   Model: {model_name}")
print(f"   Vocab size: {len(tokenizer)}")
print(f"   Model config:")
print(f"     - d_model: {model.config.d_model}")
print(f"     - encoder_layers: {model.config.encoder_layers}")
print(f"     - decoder_layers: {model.config.decoder_layers}")
print(f"     - encoder_attention_heads: {model.config.encoder_attention_heads}")
print(f"     - encoder_ffn_dim: {model.config.encoder_ffn_dim}")

# Save model config
config_data = {
    "model_name": model_name,
    "d_model": model.config.d_model,
    "encoder_layers": model.config.encoder_layers,
    "decoder_layers": model.config.decoder_layers,
    "encoder_attention_heads": model.config.encoder_attention_heads,
    "decoder_attention_heads": model.config.decoder_attention_heads,
    "encoder_ffn_dim": model.config.encoder_ffn_dim,
    "decoder_ffn_dim": model.config.decoder_ffn_dim,
    "max_position_embeddings": model.config.max_position_embeddings,
    "vocab_size": len(tokenizer),
    "bos_token_id": tokenizer.bos_token_id,
    "eos_token_id": tokenizer.eos_token_id,
    "pad_token_id": tokenizer.pad_token_id,
    "decoder_start_token_id": model.config.decoder_start_token_id,
}

with open("results/model_config.json", "w") as f:
    json.dump(config_data, f, indent=2)
print("\n   [OK] Saved model config to results/model_config.json")

print("\n2. Testing Tokenizer...")
tokenizer_data = {}

for i, sentence in enumerate(test_sentences):
    print(f"\n   Test {i+1}: {sentence}")
    
    # Tokenize
    inputs = tokenizer(sentence, return_tensors="pt")
    input_ids = inputs["input_ids"][0].tolist()
    
    print(f"   Token IDs: {input_ids}")
    print(f"   Tokens: {[tokenizer.decode([tid]) for tid in input_ids]}")
    
    tokenizer_data[f"test_{i+1}"] = {
        "sentence": sentence,
        "input_ids": input_ids,
        "tokens": [tokenizer.decode([tid]) for tid in input_ids],
    }

with open("results/tokenizer_reference.json", "w") as f:
    json.dump(tokenizer_data, f, indent=2)
print("\n   [OK] Saved tokenizer reference to results/tokenizer_reference.json")

print("\n3. Generating Encoder Outputs...")
encoder_data = {}

with torch.no_grad():
    for i, sentence in enumerate(test_sentences[:1]):  # Start with one sentence
        print(f"\n   Test {i+1}: {sentence}")
        
        # Tokenize
        inputs = tokenizer(sentence, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        print(f"   Input shape: {input_ids.shape}")
        
        # Get encoder outputs with hidden states
        encoder_outputs = model.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        
        # Save encoder output (last hidden state)
        encoder_output = encoder_outputs.last_hidden_state[0].cpu().numpy()
        print(f"   Encoder output shape: {encoder_output.shape}")
        print(f"   Encoder output stats: min={encoder_output.min():.6f}, max={encoder_output.max():.6f}, mean={encoder_output.mean():.6f}")
        
        # Save layer-by-layer hidden states
        layer_outputs = []
        for layer_idx, hidden_state in enumerate(encoder_outputs.hidden_states):
            layer_output = hidden_state[0].cpu().numpy()
            layer_outputs.append({
                "layer": layer_idx,
                "shape": list(layer_output.shape),
                "mean": float(layer_output.mean()),
                "std": float(layer_output.std()),
                "min": float(layer_output.min()),
                "max": float(layer_output.max()),
            })
            print(f"     Layer {layer_idx}: mean={layer_output.mean():.6f}, std={layer_output.std():.6f}")
        
        encoder_data[f"test_{i+1}"] = {
            "input_ids": input_ids[0].tolist(),
            "encoder_output_shape": list(encoder_output.shape),
            "encoder_output_stats": {
                "mean": float(encoder_output.mean()),
                "std": float(encoder_output.std()),
                "min": float(encoder_output.min()),
                "max": float(encoder_output.max()),
            },
            "layer_outputs": layer_outputs,
        }
        
        # Save full encoder output as numpy array
        np.save(f"results/encoder_output_test_{i+1}.npy", encoder_output)

with open("results/encoder_reference.json", "w") as f:
    json.dump(encoder_data, f, indent=2)
print("\n   [OK] Saved encoder reference to results/encoder_reference.json")

print("\n4. Generating Decoder Outputs...")
decoder_data = {}

with torch.no_grad():
    for i, sentence in enumerate(test_sentences[:1]):  # Start with one sentence
        print(f"\n   Test {i+1}: {sentence}")
        
        # Tokenize source
        inputs = tokenizer(sentence, return_tensors="pt")
        
        # Get encoder outputs
        encoder_outputs = model.model.encoder(**inputs, return_dict=True)
        
        # Prepare decoder input (start with decoder_start_token_id + target language code)
        decoder_start_token_id = model.config.decoder_start_token_id
        target_lang_id = tokenizer.convert_tokens_to_ids(target_lang)
        
        decoder_input_ids = torch.tensor([[decoder_start_token_id, target_lang_id]])
        
        print(f"   Decoder start tokens: {decoder_input_ids[0].tolist()}")
        print(f"   Decoder tokens: {[tokenizer.decode([tid]) for tid in decoder_input_ids[0].tolist()]}")
        
        # Get decoder outputs
        decoder_outputs = model.model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            output_hidden_states=True,
            return_dict=True,
        )
        
        decoder_output = decoder_outputs.last_hidden_state[0].cpu().numpy()
        print(f"   Decoder output shape: {decoder_output.shape}")
        print(f"   Decoder output stats: min={decoder_output.min():.6f}, max={decoder_output.max():.6f}, mean={decoder_output.mean():.6f}")
        
        # Get logits
        lm_logits = model.lm_head(decoder_outputs.last_hidden_state)
        logits = lm_logits[0].cpu().numpy()
        
        print(f"   Logits shape: {logits.shape}")
        print(f"   Top 5 predictions for last token: {torch.topk(lm_logits[0, -1], 5).indices.tolist()}")
        
        decoder_data[f"test_{i+1}"] = {
            "decoder_input_ids": decoder_input_ids[0].tolist(),
            "decoder_output_shape": list(decoder_output.shape),
            "decoder_output_stats": {
                "mean": float(decoder_output.mean()),
                "std": float(decoder_output.std()),
                "min": float(decoder_output.min()),
                "max": float(decoder_output.max()),
            },
            "logits_shape": list(logits.shape),
            "top_5_predictions": torch.topk(lm_logits[0, -1], 5).indices.tolist(),
        }
        
        # Save outputs
        np.save(f"results/decoder_output_test_{i+1}.npy", decoder_output)
        np.save(f"results/decoder_logits_test_{i+1}.npy", logits)

with open("results/decoder_reference.json", "w") as f:
    json.dump(decoder_data, f, indent=2)
print("\n   [OK] Saved decoder reference to results/decoder_reference.json")

print("\n5. Generating Full Translation...")
translation_data = {}

for i, sentence in enumerate(test_sentences):
    print(f"\n   Test {i+1}: {sentence}")
    
    # Translate
    inputs = tokenizer(sentence, return_tensors="pt")
    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_lang),
        max_length=50,
    )
    
    translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    
    print(f"   Translation: {translation}")
    print(f"   Output token IDs: {translated_tokens[0].tolist()}")
    
    translation_data[f"test_{i+1}"] = {
        "source": sentence,
        "target_lang": target_lang,
        "translation": translation,
        "output_token_ids": translated_tokens[0].tolist(),
    }

with open("results/translation_reference.json", "w") as f:
    json.dump(translation_data, f, indent=2)
print("\n   [OK] Saved translation reference to results/translation_reference.json")

print("\n" + "=" * 80)
print("[SUCCESS] Reference generation complete!")
print("=" * 80)
print("\nGenerated files:")
print("  - results/model_config.json")
print("  - results/tokenizer_reference.json")
print("  - results/encoder_reference.json")
print("  - results/encoder_output_test_1.npy")
print("  - results/decoder_reference.json")
print("  - results/decoder_output_test_1.npy")
print("  - results/decoder_logits_test_1.npy")
print("  - results/translation_reference.json")
print("\nNext steps:")
print("  1. Run: python test_1_tokenizer.py")
print("  2. Run: python test_2_encoder.py")
print("  3. Run: python test_3_decoder.py")
print("  4. Run: python test_5_translation.py")
print("=" * 80)

