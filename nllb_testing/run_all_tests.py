"""
Run All NLLB Verification Tests
Executes the complete test suite to verify functional equivalence with HuggingFace
"""

import subprocess
import sys
from pathlib import Path

def run_test(test_file, test_name):
    """Run a single test and return success status"""
    print()
    print("=" * 80)
    print(f"Running: {test_name}")
    print("=" * 80)
    
    try:
        result = subprocess.run(
            [sys.executable, test_file],
            cwd=Path(__file__).parent,
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            print()
            print(f"✅ {test_name} PASSED")
            return True
        else:
            print()
            print(f"❌ {test_name} FAILED (exit code: {result.returncode})")
            return False
            
    except Exception as e:
        print(f"❌ {test_name} ERROR: {e}")
        return False

def main():
    """Run all tests in sequence"""
    print()
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "        NLLB Functional Equivalence Test Suite".center(78) + "║")
    print("║" + "           Verifying llama.cpp vs HuggingFace".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    # Check if reference data exists
    results_dir = Path(__file__).parent / "results"
    if not (results_dir / "tokenizer_reference.json").exists():
        print("❌ ERROR: Reference data not found!")
        print()
        print("Please run first:")
        print("  python generate_reference.py")
        print()
        return 1
    
    # Test suite
    tests = [
        ("test_1_tokenizer.py", "Test 1: Tokenizer Verification"),
        ("test_2_encoder.py", "Test 2: Encoder Verification"),
        ("test_3_decoder.py", "Test 3: Decoder Verification"),
        ("test_4_connection.py", "Test 4: Encoder-Decoder Connection"),
        ("test_5_translation.py", "Test 5: End-to-End Translation"),
    ]
    
    results = []
    for test_file, test_name in tests:
        test_path = Path(__file__).parent / test_file
        success = run_test(test_path, test_name)
        results.append((test_name, success))
    
    # Summary
    print()
    print("=" * 80)
    print("TEST SUITE SUMMARY")
    print("=" * 80)
    print()
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {status}  {test_name}")
    
    print()
    print("-" * 80)
    print(f"  Results: {passed}/{total} tests passed")
    print("-" * 80)
    print()
    
    if passed == total:
        print("╔" + "=" * 78 + "╗")
        print("║" + " " * 78 + "║")
        print("║" + "🎉 ALL TESTS PASSED - FUNCTIONAL EQUIVALENCE VERIFIED! 🎉".center(78) + "║")
        print("║" + " " * 78 + "║")
        print("║" + "llama.cpp NLLB implementation is functionally equivalent".center(78) + "║")
        print("║" + "to HuggingFace reference implementation.".center(78) + "║")
        print("║" + " " * 78 + "║")
        print("╚" + "=" * 78 + "╝")
        print()
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print()
        print("Please review the failed tests above.")
        print()
        return 1

if __name__ == "__main__":
    sys.exit(main())


