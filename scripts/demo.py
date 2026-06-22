#!/usr/bin/env python3
"""
TakeMeter demo — classify a post with fine-tuned DistilBERT.

Usage:
    python scripts/demo.py
    python scripts/demo.py "The Celtics are frauds. They haven't beaten a healthy team all playoffs."
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "takemeter-model" / "checkpoint-20"
if not MODEL_DIR.exists():
    MODEL_DIR = ROOT / "outputs" / "model"
FALLBACK = "distilbert-base-uncased"

LABELS = ["analysis", "hot_take", "reaction"]


def load_model():
    if (MODEL_DIR / "config.json").exists():
        path = str(MODEL_DIR)
    else:
        # Use last checkpoint subfolder if present
        checkpoints = sorted(MODEL_DIR.glob("checkpoint-*")) if MODEL_DIR.exists() else []
        path = str(checkpoints[-1]) if checkpoints else FALLBACK
        print(f"Note: using {path}")

    tokenizer = AutoTokenizer.from_pretrained(path if Path(path).exists() else FALLBACK)
    if Path(path).exists() and (Path(path) / "config.json").exists():
        model = AutoModelForSequenceClassification.from_pretrained(path)
    else:
        raise FileNotFoundError("Train model first: python scripts/train_and_eval.py")

    model.eval()
    return tokenizer, model


def classify(text: str, tokenizer, model) -> tuple[str, float]:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        idx = int(probs.argmax())
    return LABELS[idx], float(probs[idx])


def main():
    text = sys.argv[1] if len(sys.argv) > 1 else input("Enter r/nba post: ")
    tokenizer, model = load_model()
    label, conf = classify(text, tokenizer, model)
    print(f"\nLabel:      {label}")
    print(f"Confidence: {conf:.1%}")


if __name__ == "__main__":
    main()
