#!/usr/bin/env python3
"""Check what tokens llama.cpp should be getting for the encoder input"""
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
tokenizer.src_lang = "eng_Latn"

text = "Hello"
inputs = tokenizer(text, return_tensors="pt")

print(f"Input text: {text}")
print(f"Token IDs: {inputs['input_ids'][0].tolist()}")
print(f"Tokens: {[tokenizer.decode([t]) for t in inputs['input_ids'][0]]}")
print(f"\nExpected input for llama.cpp:")
print(f"  Token 0: {inputs['input_ids'][0][0].item()} = {tokenizer.decode([inputs['input_ids'][0][0]])}")
print(f"  Token 1: {inputs['input_ids'][0][1].item()} = {tokenizer.decode([inputs['input_ids'][0][1]])}")  
print(f"  Token 2: {inputs['input_ids'][0][2].item()} = {tokenizer.decode([inputs['input_ids'][0][2]])}")

