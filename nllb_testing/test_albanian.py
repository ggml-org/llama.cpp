"""
Test English to Albanian translation with NLLB
Compares llama.cpp output with HuggingFace reference
"""

import sys
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

print("Loading NLLB model...")
model = AutoModelForSeq2SeqLM.from_pretrained('facebook/nllb-200-distilled-600M')
tokenizer = AutoTokenizer.from_pretrained('facebook/nllb-200-distilled-600M')
tokenizer.src_lang = 'eng_Latn'

# Test sentences
test_sentences = [
    "Hello",
    "Thank you",
    "The weather is beautiful today",
    "I would like to order a coffee, please",
    "I am learning Albanian and it is very interesting"
]

print("\n" + "=" * 80)
print("English to Albanian Translation - HuggingFace Reference")
print("=" * 80)

for i, sentence in enumerate(test_sentences, 1):
    print(f"\nTest {i}:")
    print(f"  English: {sentence}")
    
    # Tokenize and translate
    inputs = tokenizer(sentence, return_tensors='pt')
    
    # Generate Albanian translation
    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids('als_Latn'),
        max_length=50,
        num_beams=1  # Greedy decoding
    )
    
    # Decode
    translation = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    print(f"  Albanian: {translation}")

print("\n" + "=" * 80)
print("✅ HuggingFace Reference Generation Complete")
print("=" * 80)
print("\nNow run llama.cpp translations:")
print("  .\\build\\bin\\Release\\nllb-simple.exe nllb-600m.gguf \"eng_Latn <text>\" als_Latn")

