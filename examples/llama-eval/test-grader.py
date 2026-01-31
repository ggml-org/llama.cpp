#!/usr/bin/env python3

import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description="Test grader script")
    parser.add_argument("--answer", type=str, required=True, help="Predicted answer")
    parser.add_argument("--expected", type=str, required=True, help="Expected answer")
    args = parser.parse_args()

    pred = args.answer.strip()
    gold = args.expected.strip()

    print(f"Gold: {gold}")
    print(f"Pred: {pred}")

    if pred == gold:
        print("Correct!")
        sys.exit(0)
    else:
        print("Incorrect")
        sys.exit(1)

if __name__ == "__main__":
    main()
