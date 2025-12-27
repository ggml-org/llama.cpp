#!/usr/bin/env python3
"""
Debug script to understand EXACTLY how HuggingFace NLLB generates translations.
We'll trace every step to replicate in llama.cpp.
"""
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np

def main():
    print("=== Loading NLLB Model ===")
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()
    
    # Test input
    text = "Hello"
    src_lang = "eng_Latn"
    tgt_lang = "fra_Latn"
    
    print(f"\n=== Input ===")
    print(f"Text: {text}")
    print(f"Source: {src_lang} -> Target: {tgt_lang}")
    
    # Step 1: Tokenize input
    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"]
    
    print(f"\n=== Step 1: Tokenization ===")
    print(f"Input IDs: {input_ids.tolist()}")
    print(f"Input tokens: {[tokenizer.decode([t]) for t in input_ids[0]]}")
    
    # Step 2: Encode
    print(f"\n=== Step 2: Encoder ===")
    with torch.no_grad():
        encoder_outputs = model.get_encoder()(input_ids)
    
    print(f"Encoder output shape: {encoder_outputs.last_hidden_state.shape}")
    print(f"Encoder output stats: mean={encoder_outputs.last_hidden_state.mean():.6f}, std={encoder_outputs.last_hidden_state.std():.6f}")
    
    # Step 3: Prepare decoder input
    tgt_lang_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    print(f"\n=== Step 3: Decoder Initialization ===")
    print(f"Target language: {tgt_lang}")
    print(f"Target language ID: {tgt_lang_id}")
    print(f"BOS token ID: {model.config.bos_token_id}")
    print(f"EOS token ID: {model.config.eos_token_id}")
    print(f"Decoder start token ID: {model.config.decoder_start_token_id}")
    print(f"PAD token ID: {model.config.pad_token_id}")
    
    # Step 4: Manual decoding (without generate) to see what happens
    print(f"\n=== Step 4: Manual Greedy Decoding ===")
    
    # Start with decoder_start_token_id (which is EOS for NLLB) + target language
    decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id, tgt_lang_id]])
    print(f"Initial decoder input: {decoder_input_ids.tolist()}")
    print(f"Initial tokens: {[tokenizer.decode([t]) for t in decoder_input_ids[0]]}")
    
    max_length = 20
    generated_tokens = []
    
    for step in range(max_length):
        print(f"\n--- Step {step} ---")
        print(f"Decoder input shape: {decoder_input_ids.shape}")
        print(f"Decoder input IDs: {decoder_input_ids[0].tolist()}")
        
        with torch.no_grad():
            outputs = model(
                input_ids=None,  # Already encoded
                encoder_outputs=encoder_outputs,
                decoder_input_ids=decoder_input_ids,
                use_cache=False  # Disable KV cache for debugging
            )
        
        # Get logits for the last token
        logits = outputs.logits[0, -1, :]
        print(f"Logits shape: {logits.shape}")
        print(f"Logits stats: mean={logits.mean():.6f}, std={logits.std():.6f}, max={logits.max():.6f}")
        
        # Get top-5 predictions
        top_k = 5
        top_logits, top_indices = torch.topk(logits, top_k)
        print(f"Top {top_k} predictions:")
        for i, (idx, logit) in enumerate(zip(top_indices, top_logits)):
            token = tokenizer.decode([idx.item()])
            print(f"  {i+1}. Token {idx.item()}: '{token}' (logit: {logit.item():.4f})")
        
        # Greedy: take the argmax
        next_token = torch.argmax(logits).unsqueeze(0).unsqueeze(0)
        next_token_id = next_token.item()
        next_token_str = tokenizer.decode([next_token_id])
        
        print(f"Selected token: {next_token_id} ('{next_token_str}')")
        
        generated_tokens.append(next_token_id)
        
        # Check for EOS
        if next_token_id == model.config.eos_token_id:
            print("EOS reached!")
            break
        
        # Append to decoder input
        decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)
    
    # Decode full output
    print(f"\n=== Final Result ===")
    print(f"Generated token IDs: {generated_tokens}")
    translation = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print(f"Translation: {translation}")
    
    # Also test with .generate() for comparison
    print(f"\n=== Comparison with .generate() ===")
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    generated_ids = model.generate(
        inputs["input_ids"],
        forced_bos_token_id=forced_bos_token_id,
        max_length=20,
        num_beams=1,  # Greedy
        do_sample=False
    )
    print(f"Generated IDs: {generated_ids[0].tolist()}")
    translation_auto = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"Translation: {translation_auto}")

if __name__ == "__main__":
    main()

