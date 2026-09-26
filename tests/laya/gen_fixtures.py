"""Generate golden fixtures for the laya multilingual model using the PyTorch reference.

Covers choice/score/noul question types, single & multiple options, and Chinese/English input.
Exports the raw marker logits, act logits, and post-processed answers to JSON fixtures.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from laya.agent import Agent
from laya.common import QTYPES, build_sequence, render_options

MODEL_DIR = os.environ.get("LAYA_MODEL_DIR")
OUT_DIR = os.path.join(os.path.dirname(__file__), "golden")


def _require_model_dir() -> str:
    if MODEL_DIR and os.path.isdir(MODEL_DIR):
        return MODEL_DIR
    raise SystemExit("LAYA_MODEL_DIR must point to the laya-multilingual checkpoint directory")


def main():
    agent = Agent(_require_model_dir(), device="cpu")
    agent.model.eval()
    tok = agent.tok
    cfg = agent.cfg
    max_len = cfg.get("max_len", 512)
    head_max_len = cfg.get("head_max_len", 192)

    state_zh = {
        "dialogue": [
            {"role": "user", "content": "请帮我推荐一款适合程序员的高性能笔记本电脑。"},
            {"role": "assistant", "content": "好的，我推荐一款搭载高性能处理器和独立显卡的轻薄笔记本，价格大约在八千元左右。"},
        ]
    }
    state_en = {
        "dialogue": [
            {"role": "user", "content": "Please recommend a high-performance laptop for programmers."},
            {"role": "assistant", "content": "Sure, I recommend a thin-and-light laptop with a high-performance processor and a dedicated GPU, priced around 2000 dollars."},
        ]
    }

    cases = [
        # qtype choice: single option / multi option, zh
        {
            "name": "choice_single_zh",
            "state": state_zh,
            "questions": {
                "q1": {
                    "type": "choice",
                    "instructions": "根据对话，用户是否对推荐的电脑配置满意？",
                    "criteria": {"满意": "用户表示认可", "不满意": "用户明确拒绝", "需要更多信息": "用户没有直接表态"},
                },
            },
        },
        {
            "name": "choice_multi_zh",
            "state": state_zh,
            "questions": {
                "q1": {
                    "type": "choice",
                    "instructions": "对话中提到了下列哪些选购因素？",
                    "criteria": {"处理器性能": "推荐时提到了处理器性能", "便携性": "推荐时提到了便携性", "价格": "推荐时提到了价格"},
                },
            },
        },
        # qtype score
        {
            "name": "score_zh",
            "state": state_zh,
            "questions": {
                "q1": {
                    "type": "score",
                    "instructions": "评估助手回复的相关性评分",
                    "criteria": ["完全不相关", "部分相关", "基本相关", "完全相关"],
                },
            },
        },
        # qtype noul
        {
            "name": "noul_zh",
            "state": state_zh,
            "questions": {
                "q1": {
                    "type": "noul",
                    "instructions": "用户是否明确表达购买意向？",
                },
            },
        },
        # English cases (no brand / entity names)
        {
            "name": "choice_single_en",
            "state": state_en,
            "questions": {
                "q1": {
                    "type": "choice",
                    "instructions": "Which laptop was recommended?",
                    "criteria": {"Model A": "recommended", "Model B": "not recommended", "Model C": "not recommended"},
                },
            },
        },
        {
            "name": "score_en",
            "state": state_en,
            "questions": {
                "q1": {
                    "type": "score",
                    "instructions": "Rate how helpful the assistant's answer is",
                    "criteria": ["not helpful", "somewhat helpful", "helpful", "very helpful"],
                },
            },
        },
        {
            "name": "noul_en",
            "state": state_en,
            "questions": {
                "q1": {
                    "type": "noul",
                    "instructions": "Is the user ready to make a purchase?",
                },
            },
        },
    ]

    os.makedirs(OUT_DIR, exist_ok=True)

    # collect all sequences for batching: build tokenized inputs and run one forward per case
    results = {}
    for case in cases:
        name = case["name"]
        state = case["state"]
        questions = case["questions"]
        items = []
        seqs = {}
        for qid, qdef in questions.items():
            q = agent._to_internal(qdef)
            seq, markers = build_sequence(tok, state, q, max_len, head_max_len)
            opts = render_options(q)
            seqs[qid] = {
                "input_ids": seq,
                "markers": markers,
                "options": opts,
                "qtype": QTYPES[q["t"]],
                "qtype_name": q["t"],
            }
            items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]]})

        from laya.common import collate_items
        batch = collate_items([items], tok.pad_token_id)
        with torch.no_grad():
            logits, act = agent.model(
                batch["input_ids"].to(agent.device),
                batch["attention_mask"].to(agent.device),
                batch["marker_pos"].to(agent.device),
                batch["marker_mask"].to(agent.device),
                batch["qtype"].to(agent.device),
            )
        logits_np = logits.float().cpu().numpy()
        act_np = torch.softmax(act.float(), -1).cpu().numpy()

        answers = agent.system_one(state, questions)

        per_q = {}
        for r, qid in enumerate(questions.keys()):
            info = seqs[qid]
            k = len(info["markers"])
            raw_logits = logits_np[r, :k].tolist()
            per_q[qid] = {
                "qtype": info["qtype_name"],
                "options": info["options"],
                "input_ids": info["input_ids"],
                "marker_pos": info["markers"],
                "raw_logits": raw_logits,
                "act_logits": act_np[r].tolist(),
                "act_probability": float(act_np[r, 0]),
            }
        results[name] = {
            "model": "laya-multilingual",
            "state": state,
            "questions": questions,
            "per_question": per_q,
            "answers": answers["answers"],
        }

    manifest = {}
    for name, data in results.items():
        path = os.path.join(OUT_DIR, f"{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        manifest[name] = os.path.basename(path)
        print(f"wrote {path}")

    with open(os.path.join(OUT_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print("done")


if __name__ == "__main__":
    import torch
    main()
