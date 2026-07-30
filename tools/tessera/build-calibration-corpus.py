#!/usr/bin/env python3
"""Generate the clean-room Tessera balanced calibration corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


SCHEMA = "llama.tessera.calibration-corpus.v1"
RECEIPT_SCHEMA = "llama.tessera.training-corpus.v1"
VERSION = 1
CATEGORY_COUNTS = {
    "code": 440,
    "en": 350,
    "ko": 60,
    "zh": 390,
    "ja": 60,
    "tool_calling": 120,
    "reasoning": 370,
    "chat": 130,
    "mixed": 190,
    "structured_context": 570,
}


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def code_sample(index: int) -> str:
    languages = ("Python", "JavaScript", "Rust", "SQL", "Swift")
    language = languages[index % len(languages)]
    width = 4 + index % 13
    modulus = 7 + index % 23
    if language == "Python":
        body = (
            f"def normalize_window_{index}(values, width={width}):\n"
            "    result = []\n"
            "    for offset in range(0, len(values), width):\n"
            "        window = values[offset:offset + width]\n"
            "        scale = max(sum(abs(v) for v in window), 1)\n"
            "        result.extend(v / scale for v in window)\n"
            "    return result\n"
        )
    elif language == "JavaScript":
        body = (
            f"export function bucket{index}(records) {{\n"
            f"  return records.reduce((m, r) => {{ const k = r.score % {modulus};\n"
            "    (m[k] ??= []).push({...r, normalized: r.score / (1 + k)}); return m; }, {});\n"
            "}\n"
        )
    elif language == "Rust":
        body = (
            f"fn rolling_{index}(input: &[i64]) -> Vec<i64> {{\n"
            f"    input.windows({width}).map(|w| w.iter().sum::<i64>() % {modulus}).collect()\n"
            "}\n"
        )
    elif language == "SQL":
        body = (
            "WITH ranked AS (\n"
            "  SELECT project_id, event_day, SUM(weight) AS total,\n"
            "         ROW_NUMBER() OVER (PARTITION BY project_id ORDER BY SUM(weight) DESC) AS rank\n"
            "  FROM synthetic_events GROUP BY project_id, event_day\n"
            f") SELECT * FROM ranked WHERE rank <= {1 + index % 5};\n"
        )
    else:
        body = (
            f"func compact{index}(_ values: [Int]) -> [Int] {{\n"
            f"    Dictionary(grouping: values, by: {{ $0 % {modulus} }}).keys.sorted()\n"
            "}\n"
        )
    return (
        f"Language: {language}. Review this self-contained implementation for boundary "
        f"conditions and explain its asymptotic behavior.\n\n{body}"
    )


def english_sample(index: int) -> str:
    systems = ("Aster", "Bracken", "Cinder", "Delta", "Elm", "Fable")
    system = systems[index % len(systems)]
    nodes = 3 + index % 17
    return (
        f"The fictional {system}-{index} observatory contains {nodes} independent sensor "
        "nodes. Each node records pressure, color temperature, and a monotonic sequence "
        "number. A coordinator accepts a reading only after two different nodes agree on "
        "the sequence interval; disagreement creates a reversible quarantine record. "
        "Explain how this rule limits accidental duplication, distinguish availability "
        "from consistency, and summarize the behavior for a new operator. All names and "
        "events in this passage are procedurally generated."
    )


def korean_sample(index: int) -> str:
    return (
        f"가상의 관측 장치 하늘-{index}은 여러 센서의 온도와 압력 값을 비교한다. "
        f"센서 수는 {3 + index % 9}개이며, 두 번 연속 같은 구간이 확인된 경우에만 "
        "기록을 확정한다. 값이 다르면 원본을 지우지 않고 검토 목록으로 이동한다. "
        "이 절차의 목적과 장단점을 한국어로 간단히 설명하고, 안전한 복구 순서를 제안하라."
    )


def chinese_sample(index: int) -> str:
    return (
        f"虚构的青岚-{index}数据站接收来自{4 + index % 12}个传感器的读数。"
        "系统先检查递增序号，再比较两个独立节点的摘要；若结果不一致，就把记录放入"
        "可恢复的隔离区，而不是直接删除。请说明这个流程如何减少重复和误删，并给出"
        "一个包含验证、回滚和审计步骤的简短操作方案。本文中的名称和数据均为程序生成。"
    )


def japanese_sample(index: int) -> str:
    return (
        f"架空の観測装置ミナモ-{index}は、{3 + index % 10}個のセンサーから測定値を受け取る。"
        "連続番号を確認した後、二つの独立した計算結果が一致した場合だけ記録を確定する。"
        "不一致の記録は削除せず、復元可能な保留領域へ移す。この設計の目的、障害時の"
        "復旧手順、監査で確認すべき項目を日本語で説明せよ。"
    )


def tool_sample(index: int) -> str:
    city = ("Northport", "Lake Ember", "Cedar Vale")[index % 3]
    return (
        "Available tool:\n"
        '{"name":"inspect_station","description":"Read a fictional station snapshot",'
        '"parameters":{"type":"object","properties":{"station":{"type":"string"},'
        '"include_history":{"type":"boolean"}},"required":["station"]}}\n\n'
        f"User request: Inspect station \"{city}-{index}\" and include its history. "
        "Return the tool arguments as valid JSON, then describe how you would handle "
        "a timeout without inventing a result."
    )


def reasoning_sample(index: int) -> str:
    crates = 5 + index % 31
    units = 7 + (index * 3) % 29
    removed = index % max(units - 1, 1)
    total = crates * units - removed
    return (
        f"A fictional depot has {crates} sealed crates with {units} calibration tiles "
        f"in each crate. During inspection, {removed} tiles are removed. How many tiles "
        "remain? Work from the quantities given, state the multiplication and subtraction, "
        f"and verify the result by reversing the subtraction. The expected arithmetic result "
        f"is {total}, but the explanation must show why."
    )


def chat_sample(index: int) -> str:
    return (
        "System: You are assisting with a fictional, non-production device.\n"
        f"User: The indicator on unit Pine-{index} changed from amber to blue after restart. "
        "Should I erase its history?\n"
        "Assistant: Do not erase history merely because the indicator changed. First export "
        "the local diagnostic summary and compare the sequence counter.\n"
        "User: The counter increased by one and the self-test passed. What next?\n"
        "Assistant: Explain a cautious next step that preserves rollback information and "
        "does not claim access to tools or readings that were not supplied."
    )


def mixed_sample(index: int) -> str:
    return (
        f"# Synthetic incident {index}\n\n"
        f"Status: `review-{index % 5}`\n\n"
        "| channel | expected | observed |\n"
        f"| alpha | {10 + index % 7} | {10 + (index * 2) % 7} |\n"
        f"| beta | {20 + index % 11} | {20 + (index * 3) % 11} |\n\n"
        "```ini\nretry_limit=3\npreserve_history=true\nmode=validate\n```\n\n"
        "Reconcile the table with the configuration and identify which conclusions are "
        "supported directly versus which would require another measurement."
    )


def structured_sample(index: int) -> str:
    return (
        f"<station id=\"synthetic-{index}\">\n"
        f"  <epoch>{index // 8}</epoch>\n"
        f"  <channels>{2 + index % 9}</channels>\n"
        "  <policy preserve=\"true\" quorum=\"2\" />\n"
        "</station>\n\n"
        "```json\n"
        f'{{"request":"validate","station":"synthetic-{index}",'
        f'"ranges":[[{index % 17},{index % 17 + 8}]],"dry_run":true}}\n'
        "```\n\n"
        "Describe the relationship between the XML state and JSON request. Reject any "
        "interpretation that would silently mutate data while dry_run is true."
    )


GENERATORS = {
    "code": code_sample,
    "en": english_sample,
    "ko": korean_sample,
    "zh": chinese_sample,
    "ja": japanese_sample,
    "tool_calling": tool_sample,
    "reasoning": reasoning_sample,
    "chat": chat_sample,
    "mixed": mixed_sample,
    "structured_context": structured_sample,
}


def build_records(seed: int) -> list[dict]:
    records = []
    for category, count in CATEGORY_COUNTS.items():
        for index in range(count):
            text = GENERATORS[category](index)
            record_id = digest_bytes(
                f"{VERSION}\0{seed}\0{category}\0{index}\0{text}".encode()
            )[:24]
            records.append({
                "schema": SCHEMA,
                "id": record_id,
                "category": category,
                "text": text,
                "origin": "tribunus.dev-clean-room-procedural",
            })
    random.Random(seed).shuffle(records)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the clean-room Tessera calibration corpus"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=640)
    parser.add_argument("--epoch", type=int, default=0)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = build_records(args.seed)
    index = output / "samples.jsonl"
    corpus = output / "calibration.txt"
    index.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    corpus.write_text(
        "\n\n".join(record["text"] for record in records) + "\n",
        encoding="utf-8",
    )
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "epoch": args.epoch,
        "sha256": digest_bytes(corpus.read_bytes()),
        "index_sha256": digest_bytes(index.read_bytes()),
        "generator": {
            "schema": SCHEMA,
            "version": VERSION,
            "seed": args.seed,
            "sample_count": len(records),
            "categories": CATEGORY_COUNTS,
        },
        "license": "CC-BY-NC-SA-4.0",
        "license_uri": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
        "attribution": "Julian Alejandro Torres Nieto, Tribunus.dev",
        "distribution_cleared": True,
        "contains_user_inference": False,
        "commercial_use": False,
        "share_alike": True,
        "sources": [{
            "name": "Tessera clean-room procedural calibration corpus",
            "origin": "original templates and deterministic parameter expansion",
            "contains_upstream_benchmark_items": False,
            "contains_user_data": False,
        }],
    }
    (output / "training-corpus-receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema": SCHEMA,
        "version": VERSION,
        "seed": args.seed,
        "sample_count": len(records),
        "categories": CATEGORY_COUNTS,
        "corpus": corpus.name,
        "index": index.name,
        "receipt": "training-corpus-receipt.json",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
