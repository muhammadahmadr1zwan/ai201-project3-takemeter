#!/usr/bin/env python3
"""
Full TakeMeter pipeline: split, fine-tune DistilBERT, evaluate, optional Groq baseline.

Usage:
    pip install transformers datasets scikit-learn torch matplotlib
    python scripts/train_and_eval.py

Optional baseline (requires GROQ_API_KEY):
    python scripts/train_and_eval.py --baseline
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "labeled_dataset.csv"
OUT = ROOT / "outputs"
MODEL_NAME = "distilbert-base-uncased"

LABEL2ID = {"analysis": 0, "hot_take": 1, "reaction": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

BASELINE_PROMPT = """You classify r/nba posts into exactly one label.

Labels:
- analysis: Structured argument with specific, verifiable evidence (stats, film, history, tactics).
- hot_take: Bold confident opinion that asserts rather than argues.
- reaction: Immediate emotional response to a specific event; little to no argument.

Respond with ONLY: analysis, hot_take, or reaction

Post:
{text}"""


def load_data():
    df = pd.read_csv(DATA)
    df = df.dropna(subset=["text", "label"])
    df["label_id"] = df["label"].map(LABEL2ID)
    df = df[df["label_id"].notna()]
    return df


def split_data(df, seed=42):
    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=seed)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["label"], random_state=seed)
    return train_df, val_df, test_df


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def tokenize_df(df, tokenizer, max_length=256):
    ds = Dataset.from_pandas(df[["text", "label_id"]].rename(columns={"label_id": "labels"}))
    return ds.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=max_length), batched=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}


def groq_classify(text: str, api_key: str) -> str:
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": "llama-3.3-70b-versatile",
            "messages": [{"role": "user", "content": BASELINE_PROMPT.format(text=text[:1500])}],
            "temperature": 0,
            "max_tokens": 10,
        },
        timeout=60,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"].strip().lower()
    m = re.search(r"\b(analysis|hot_take|reaction)\b", content)
    return m.group(1) if m else "unknown"


def run_baseline(test_df, api_key: str):
    preds, confidences = [], []
    for i, row in test_df.iterrows():
        try:
            label = groq_classify(row["text"], api_key)
        except Exception:
            label = "unknown"
        preds.append(label)
        if (len(preds)) % 5 == 0:
            print(f"  baseline {len(preds)}/{len(test_df)}")
        time.sleep(0.25)

    valid = [(t, p) for t, p in zip(test_df["label"], preds) if p in LABEL2ID]
    true = [t for t, _ in valid]
    pred = [p for _, p in valid]
    acc = accuracy_score(true, pred) if true else 0
    report = classification_report(true, pred, output_dict=True, zero_division=0)
    return {"accuracy": acc, "report": report, "predictions": preds, "parse_failures": preds.count("unknown")}


def train_model(train_ds, val_ds, test_ds, tokenizer, train_df):
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID
    )

    # Class weights for imbalance
    counts = train_df["label_id"].value_counts().sort_index()
    weights = (len(train_df) / (3 * counts)).values
    weights = torch.tensor(weights, dtype=torch.float)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
            loss = loss_fn(outputs.logits, labels)
            return (loss, outputs) if return_outputs else loss

    args = TrainingArguments(
        output_dir=str(OUT / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=10,
        report_to="none",
        seed=42,
    )

    trainer = WeightedTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(str(OUT / "model"))
    tokenizer.save_pretrained(str(OUT / "model"))
    return trainer


def evaluate_finetuned(trainer, test_ds, test_df):
    outputs = trainer.predict(test_ds)
    preds = np.argmax(outputs.predictions, axis=1)
    probs = torch.softmax(torch.tensor(outputs.predictions), dim=-1).numpy()

    true = test_df["label_id"].values
    acc = accuracy_score(true, preds)
    true_names = [ID2LABEL[t] for t in true]
    pred_names = [ID2LABEL[p] for p in preds]
    report = classification_report(true_names, pred_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(true_names, pred_names, labels=list(LABEL2ID.keys()))

    # Sample predictions with confidence
    samples = []
    for i in range(min(5, len(test_df))):
        idx = test_df.index[i]
        pos = test_df.index.get_loc(idx)
        samples.append(
            {
                "text": test_df.iloc[pos]["text"][:120] + "...",
                "true": test_df.iloc[pos]["label"],
                "predicted": ID2LABEL[preds[pos]],
                "confidence": float(probs[pos][preds[pos]]),
            }
        )

    wrong = []
    for i, (t, p) in enumerate(zip(true_names, pred_names)):
        if t != p:
            wrong.append({"text": test_df.iloc[i]["text"], "true": t, "predicted": p, "confidence": float(probs[i][preds[i]])})

    return {
        "accuracy": acc,
        "report": report,
        "confusion_matrix": cm.tolist(),
        "labels": list(LABEL2ID.keys()),
        "samples": samples,
        "wrong_predictions": wrong[:10],
    }


def plot_confusion(cm, labels, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    OUT.mkdir(exist_ok=True)
    set_seed(42)
    df = load_data()
    print(f"Loaded {len(df)} examples")
    print(df["label"].value_counts())

    train_df, val_df, test_df = split_data(df)
    print(f"Split: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = tokenize_df(train_df, tokenizer)
    val_ds = tokenize_df(val_df, tokenizer)
    test_ds = tokenize_df(test_df, tokenizer)

    results = {"split_sizes": {"train": len(train_df), "val": len(val_df), "test": len(test_df)}}

    if args.baseline:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("Skipping baseline — set GROQ_API_KEY")
        else:
            print("Running Groq baseline...")
            results["baseline"] = run_baseline(test_df.reset_index(drop=True), api_key)

    print("Fine-tuning DistilBERT (CPU — may take 15-30 min)...")
    trainer = train_model(train_ds, val_ds, test_ds, tokenizer, train_df)
    ft = evaluate_finetuned(trainer, test_ds, test_df.reset_index(drop=True))
    results["finetuned"] = {
        "accuracy": ft["accuracy"],
        "report": ft["report"],
        "confusion_matrix": ft["confusion_matrix"],
        "samples": ft["samples"],
        "wrong_predictions": ft["wrong_predictions"],
    }

    plot_confusion(np.array(ft["confusion_matrix"]), ft["labels"], OUT / "confusion_matrix.png")

    with (OUT / "evaluation_results.json").open("w") as f:
        json.dump(results, f, indent=2)

    print(f"\nFine-tuned accuracy: {ft['accuracy']:.3f}")
    if "baseline" in results:
        print(f"Baseline accuracy: {results['baseline']['accuracy']:.3f}")
    print(f"Saved to {OUT / 'evaluation_results.json'}")


if __name__ == "__main__":
    main()
