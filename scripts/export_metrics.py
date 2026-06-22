#!/usr/bin/env python3
"""Export full evaluation metrics for README from saved model."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "labeled_dataset.csv"
MODEL = ROOT / "takemeter-model" / "checkpoint-20"
OUT = ROOT / "outputs" / "evaluation_results.json"

LABEL2ID = {"analysis": 0, "hot_take": 1, "reaction": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def main():
    df = pd.read_csv(DATA)
    df["label_id"] = df["label"].map(LABEL2ID)
    train_df, temp = train_test_split(df, test_size=0.30, random_state=42, stratify=df["label"])
    _, test_df = train_test_split(temp, test_size=0.50, random_state=42, stratify=temp["label"])
    test_df = test_df.reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    model.eval()

    preds, probs_list = [], []
    for text in test_df["text"]:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            logits = model(**enc).logits[0]
            prob = torch.softmax(logits, dim=-1).numpy()
        pid = int(prob.argmax())
        preds.append(pid)
        probs_list.append(prob)

    true = test_df["label_id"].values
    true_names = [ID2LABEL[t] for t in true]
    pred_names = [ID2LABEL[p] for p in preds]
    cm = confusion_matrix(true_names, pred_names, labels=list(LABEL2ID.keys()))
    report = classification_report(true_names, pred_names, output_dict=True, zero_division=0)

    samples = []
    for i in range(min(5, len(test_df))):
        samples.append({
            "text": test_df.iloc[i]["text"][:150],
            "true": test_df.iloc[i]["label"],
            "predicted": pred_names[i],
            "confidence": float(probs_list[i][preds[i]]),
        })

    wrong = []
    for i, (t, p) in enumerate(zip(true_names, pred_names)):
        if t != p:
            wrong.append({
                "text": test_df.iloc[i]["text"],
                "true": t,
                "predicted": p,
                "confidence": float(probs_list[i][preds[i]]),
            })

    existing = json.loads(OUT.read_text()) if OUT.exists() else json.loads((ROOT / "evaluation_results.json").read_text())
    existing["finetuned_detailed"] = {
        "accuracy": float(accuracy_score(true, preds)),
        "report": report,
        "confusion_matrix": cm.tolist(),
        "labels": list(LABEL2ID.keys()),
        "samples": samples,
        "wrong_predictions": wrong[:10],
    }
    OUT.write_text(json.dumps(existing, indent=2))
    print(json.dumps(existing["finetuned_detailed"]["report"], indent=2))
    print("cm", cm.tolist())


if __name__ == "__main__":
    main()
