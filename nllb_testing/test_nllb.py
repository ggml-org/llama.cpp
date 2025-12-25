#!/usr/bin/env python3
"""
Test NLLB model loading and translation
"""
import subprocess
import sys

print("=" * 80)
print("NLLB Model Testing")
print("=" * 80)

# Test 1: Model info
print("\nTest 1: Checking model architecture...")
try:
    result = subprocess.run(
        ["./build/bin/Release/llama-cli.exe", "-m", "nllb-600m.gguf", "--version"],
        capture_output=True,
        text=True,
        timeout=10
    )
    print("Version info:")
    print(result.stdout)
    print(result.stderr)
except Exception as e:
    print(f"Error: {e}")

# Test 2: English to French
print("\n" + "=" * 80)
print("Test 2: English to French Translation")
print("=" * 80)
print("Input: 'eng_Latn Hello, how are you? fra_Latn'")
print("Expected output: French translation")
print("\nRunning translation...")

try:
    result = subprocess.run(
        [
            "./build/bin/Release/llama-cli.exe",
            "-m", "nllb-600m.gguf",
            "-p", "eng_Latn Hello, how are you? fra_Latn",
            "-n", "20",
            "-c", "512",
            "--temp", "0.3"
        ],
        capture_output=True,
        text=True,
        timeout=60
    )
    
    print("\n--- Output ---")
    print(result.stdout)
    if result.stderr:
        print("\n--- Errors/Warnings ---")
        print(result.stderr)
    
    print("\n--- Return code ---")
    print(result.returncode)
    
except subprocess.TimeoutExpired:
    print("ERROR: Command timed out after 60 seconds")
except Exception as e:
    print(f"ERROR: {e}")

print("\n" + "=" * 80)
print("Testing complete")
print("=" * 80)

