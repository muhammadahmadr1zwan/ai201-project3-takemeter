#!/usr/bin/env python3
"""Check label distribution in labeled_dataset.csv."""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/labeled_dataset.csv")
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    labels = [r["label"].strip() for r in rows if r.get("label", "").strip()]
    counts = Counter(labels)
    total = len(labels)

    print(f"File: {path}")
    print(f"Total labeled: {total}\n")

    for label, count in counts.most_common():
        pct = 100 * count / total if total else 0
        flag = " ⚠️  OVER 70%" if pct > 70 else ""
        print(f"  {label:12} {count:4}  ({pct:5.1f}%){flag}")

    if total < 200:
        print(f"\n⚠️  Need {200 - total} more examples (minimum 200)")
    else:
        print("\nOK: Meets 200+ requirement")

    if total and max(counts.values()) / total > 0.7:
        print("⚠️  Imbalanced — collect more underrepresented labels")


if __name__ == "__main__":
    main()
