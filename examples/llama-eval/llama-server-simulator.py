#!/usr/bin/env python3

import argparse
import json
import random
import re
import time
import sys
import os
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

import datasets
from flask import Flask, request, jsonify

# Set cache directory for HuggingFace datasets
cache_dir = Path.home() / ".cache" / "huggingface" / "datasets"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["HF_DATASETS_CACHE"] = str(cache_dir)

def dice(s1: str, s2: str) -> float:
    """Calculate Dice coefficient between two strings based on bigram overlap."""
    if not s1 and not s2:
        return 1.0

    def _bigrams(s: str):
        return [s[i : i + 2] for i in range(len(s) - 1)]

    bigrams1 = _bigrams(s1)
    bigrams2 = _bigrams(s2)

    if not bigrams1 and not bigrams2:
        return 1.0

    from collections import Counter

    freq1 = Counter(bigrams1)
    freq2 = Counter(bigrams2)

    intersection = sum(min(freq1[bg], freq2[bg]) for bg in freq1)
    dice_coeff = 2 * intersection / (len(bigrams1) + len(bigrams2))
    return dice_coeff

def debug_log(message: str):
    """Log debug messages to both stdout and a file"""
    print(message, file=sys.stderr)
    with open("/tmp/simulator-debug.log", "a") as f:
        f.write(message + "\n")

app = Flask(__name__)

@dataclass
class EvalState:
    id: str
    tasks: List[str]
    task_states: Dict[str, Dict]
    sampling_config: Dict

def normalize_number(s: str) -> Optional[int]:
    match = re.match(r"\d+", s)  # match digits from the start
    if not match:
        return None
    return int(match.group(0))

class AimeDataset:
    def __init__(self, split: str = "train"):
        self.split = split
        self.questions: List[Dict] = []
        self._load_dataset()

    def _load_dataset(self):
        print(f"Loading AIME dataset (split: {self.split})...")

        cache_path = Path.home() / ".cache" / "huggingface" / "datasets" / "AI-MO___aimo-validation-aime" / "default" / "0.0.0"
        if cache_path.exists():
            print(f"Using cached dataset from {cache_path}")
            ds = datasets.load_dataset("AI-MO/aimo-validation-aime", split=self.split, cache_dir=str(cache_path))
        else:
            ds = datasets.load_dataset("AI-MO/aimo-validation-aime", split=self.split)

        self.questions = list(ds)
        print(f"AIME dataset loaded: {len(self.questions)} questions")

    def find_question(self, request_text: str) -> Optional[Dict]:
        best_match = None
        best_distance = -1
        best_index = -1

        for i, question in enumerate(self.questions):
            question_text = question["problem"]
            request_lower = request_text.lower()
            question_lower = question_text.lower()

            # Exact match
            if question_lower == request_lower:
                debug_log(f"DEBUG: Found exact match at index {i}")
                return question

            # Remove LaTeX formatting for more flexible matching
            question_no_latex = re.sub(r'\$[^$]+\$', '', question_text)
            if question_no_latex.lower() == request_lower:
                debug_log(f"DEBUG: Found match (no LaTeX) at index {i}")
                return question

            # Calculate Levenshtein distance for partial matches
            # Only consider if request is at least 50% of question length
            if len(request_lower) >= len(question_lower) * 0.5:
                distance = dice(question_lower, request_lower)

                if distance > best_distance:
                    best_distance = distance
                    best_match = question
                    best_index = i

        if best_match and best_distance > 0.3:  # Threshold for partial match
            debug_log(f"DEBUG: Found best partial match at index {best_index} with distance {best_distance:.3f}")
            return best_match

        debug_log(f"DEBUG: No matching question found for: {request_text[:100]}...")
        return None

    def get_answer(self, question: Dict) -> str:
        answer = question["answer"]
        if isinstance(answer, str):
            normalized = normalize_number(answer)
            return str(normalized) if normalized is not None else answer
        return str(answer)

class Simulator:
    def __init__(
        self,
        port: int = 8033,
        host: str = "localhost",
        success_rate: float = 0.8,
        dataset_split: str = "train"
    ):
        self.port = port
        self.host = host
        self.success_rate = success_rate
        self.dataset = AimeDataset(dataset_split)
        self.eval_state = EvalState(
            id="aime-2025",
            tasks=["aime"],
            task_states={},
            sampling_config={"temperature": 0, "max_tokens": 2048}
        )

    def _generate_response(
        self,
        question: Dict,
        should_be_correct: bool
    ) -> Dict:
        expected_answer = self.dataset.get_answer(question)

        if should_be_correct:
            response_text = expected_answer
        else:
            response_text = self._generate_wrong_answer(question)

        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "llama",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150
            }
        }

    def _generate_wrong_answer(self, question: Dict) -> str:
        expected_answer = self.dataset.get_answer(question)

        if expected_answer.isdigit():
            wrong_answer = str(int(expected_answer) + 1)
        else:
            wrong_answer = expected_answer + " (wrong)"

        return wrong_answer

    def _process_request(self, request_data: Dict) -> Dict:
        messages = request_data.get("messages", [])
        if not messages:
            return {"error": "No messages in request"}

        request_text = messages[0].get("content", "")
        debug_log(f"DEBUG: Received request with content: {request_text[:150]}...")

        question = self.dataset.find_question(request_text)
        if not question:
            debug_log(f"DEBUG: find_question returned None")
            return {"error": "No matching question found"}

        should_be_correct = random.random() < self.success_rate

        response = self._generate_response(question, should_be_correct)

        task_id = "aime"
        self.eval_state.task_states[task_id] = {
            "correct": should_be_correct,
            "expected": self.dataset.get_answer(question),
            "predicted": response["choices"][0]["message"]["content"]
        }

        return response

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    try:
        request_data = request.get_json()

        if not request_data:
            return jsonify({"error": "Invalid JSON"}), 400

        response = simulator._process_request(request_data)

        return jsonify(response)

    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500

def main():
    parser = argparse.ArgumentParser(
        description="llama-server simulator for testing eval scripts"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8033,
        help="Server port (default: 8033)"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Server host (default: localhost)"
    )
    parser.add_argument(
        "--success-rate",
        type=float,
        default=0.8,
        help="Success rate 0-1 (default: 0.8)"
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train",
        help="AIME dataset split to use (default: train)"
    )

    args = parser.parse_args()

    global simulator
    simulator = Simulator(
        port=args.port,
        host=args.host,
        success_rate=args.success_rate,
        dataset_split=args.dataset_split
    )

    print("\n=== llama-server-simulator ===")
    print(f"Server running on http://{args.host}:{args.port}")
    print(f"Success rate: {args.success_rate}")
    print(f"AIME dataset loaded: {len(simulator.dataset.questions)} questions")
    print("\nPress Ctrl+C to stop\n")

    app.run(host=args.host, port=args.port, debug=False)

if __name__ == "__main__":
    main()
