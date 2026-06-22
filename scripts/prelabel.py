#!/usr/bin/env python3
"""
Optional: pre-label raw posts using Groq API before manual review.

Requires GROQ_API_KEY environment variable.
Outputs data/prelabeled.csv with columns: text, label, notes, prelabeled

You MUST review every label before training. Pre-labeling is a draft only.

Usage:
    set GROQ_API_KEY=your_key
    python scripts/prelabel.py
"""

from __future__ import annotations

import csv
import json
import os
import re
import time
from pathlib import Path

import requests

LABELS = ["analysis", "hot_take", "reaction"]

SYSTEM_PROMPT = """You classify r/nba posts into exactly one label.

Labels:
- analysis: Structured argument with specific, verifiable evidence (stats, film, history, tactics). Evidence would support the claim even without opinion framing.
- hot_take: Bold confident opinion that asserts rather than argues. May include a lone stat but not structured reasoning.
- reaction: Immediate emotional response to a specific event. Little to no argument — venting or excitement in the moment.

Edge rules:
- Single-stat accusatory posts → hot_take unless full structured reasoning
- Venting about refs/a loss with no evidence → reaction
- Debate-worthy claims without evidence → hot_take

Respond with ONLY the label name: analysis, hot_take, or reaction"""


def classify(text: str, api_key: str) -> str:
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Classify this post:\n\n{text[:1500]}"},
            ],
            "temperature": 0,
            "max_tokens": 10,
        },
        timeout=60,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"].strip().lower()
    match = re.search(r"\b(analysis|hot_take|reaction)\b", content)
    if not match:
        raise ValueError(f"Unparseable response: {content!r}")
    return match.group(1)


def main() -> None:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("Set GROQ_API_KEY environment variable")

    root = Path(__file__).resolve().parent.parent
    raw_path = root / "data" / "raw_posts.csv"
    out_path = root / "data" / "prelabeled.csv"

    if not raw_path.exists():
        raise SystemExit(f"Missing {raw_path}. Run collect_reddit.py first.")

    rows = list(csv.DictReader(raw_path.open(encoding="utf-8")))
    print(f"Pre-labeling {len(rows)} examples...")

    out_rows = []
    for i, row in enumerate(rows, 1):
        try:
            label = classify(row["text"], api_key)
        except Exception as exc:
            print(f"  [{i}/{len(rows)}] error: {exc}")
            label = ""
        out_rows.append(
            {
                "text": row["text"],
                "label": label,
                "notes": "prelabeled — needs manual review",
                "prelabeled": "true",
            }
        )
        if i % 10 == 0:
            print(f"  [{i}/{len(rows)}] done")
        time.sleep(0.3)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label", "notes", "prelabeled"])
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Saved to {out_path}. Review every row, then save as data/labeled_dataset.csv")


if __name__ == "__main__":
    main()
