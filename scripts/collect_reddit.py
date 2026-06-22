#!/usr/bin/env python3
"""
Collect public posts and comments from r/nba for TakeMeter annotation.

Primary source: PullPush Reddit archive API (public historical data).
Fallback: Reddit JSON API (may be blocked on some networks).

Outputs data/raw_posts.csv with columns: text, source, post_id, collected_at

Usage:
    python scripts/collect_reddit.py --target 250
"""

from __future__ import annotations

import argparse
import csv
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

SUBREDDIT = "nba"
PULLPUSH_COMMENT = "https://api.pullpush.io/reddit/search/comment"
PULLPUSH_SUBMISSION = "https://api.pullpush.io/reddit/search/submission"
REDDIT_BASE = f"https://old.reddit.com/r/{SUBREDDIT}/"
HEADERS = {"User-Agent": "TakeMeter/1.0 (AI201 educational project)"}
MIN_CHARS = 40
MAX_CHARS = 2000


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(r"http\S+", "", text)
    return text.strip()


def pullpush_search(url: str, params: dict) -> list[dict]:
    resp = requests.get(url, params=params, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    return resp.json().get("data", [])


def collect_from_pullpush(target: int, seen: set[str]) -> list[dict]:
    rows: list[dict] = []
    before = int(datetime.now(timezone.utc).timestamp())

    # Mix submissions and comments for discourse variety
    sources = [
        (PULLPUSH_SUBMISSION, {"subreddit": SUBREDDIT, "size": 100, "sort": "desc", "sort_type": "created_utc"}),
        (PULLPUSH_COMMENT, {"subreddit": SUBREDDIT, "size": 100, "sort": "desc", "sort_type": "created_utc"}),
    ]

    for url, base_params in sources:
        params = dict(base_params)
        label = "submission" if "submission" in url else "comment"

        for _ in range(8):
            if len(rows) >= target:
                break

            batch = pullpush_search(url, params)
            if not batch:
                break

            for item in batch:
                pid = str(item.get("id", ""))
                if not pid or pid in seen:
                    continue
                seen.add(pid)

                if label == "submission":
                    title = item.get("title") or ""
                    selftext = item.get("selftext") or ""
                    if selftext in ("[removed]", "[deleted]"):
                        selftext = ""
                    text = clean_text(f"{title}. {selftext}" if selftext else title)
                    source = "pullpush:submission"
                else:
                    body = item.get("body") or ""
                    if body in ("[removed]", "[deleted]"):
                        continue
                    text = clean_text(body)
                    source = "pullpush:comment"

                if not (MIN_CHARS <= len(text) <= MAX_CHARS):
                    continue

                rows.append(
                    {
                        "text": text,
                        "source": source,
                        "post_id": pid,
                        "collected_at": datetime.now(timezone.utc).isoformat(),
                    }
                )

            before = batch[-1].get("created_utc", before)
            params["before"] = before
            time.sleep(1.0)

    return rows


def collect_from_reddit_direct(target: int, seen: set[str]) -> list[dict]:
    """Fallback if PullPush is unavailable."""
    session = requests.Session()
    rows: list[dict] = []

    for endpoint in ("new.json", "hot.json"):
        try:
            url = f"{REDDIT_BASE}{endpoint}"
            resp = session.get(url, headers=HEADERS, params={"limit": 100, "raw_json": 1}, timeout=30)
            resp.raise_for_status()
            items = resp.json().get("data", {}).get("children", [])
        except requests.RequestException as exc:
            print(f"  Reddit direct failed ({endpoint}): {exc}")
            continue

        for item in items:
            post = item.get("data", {})
            pid = post.get("id")
            if not pid or pid in seen:
                continue
            seen.add(pid)

            title = post.get("title") or ""
            selftext = post.get("selftext") or ""
            text = clean_text(f"{title}. {selftext}" if selftext else title)
            if MIN_CHARS <= len(text) <= MAX_CHARS:
                rows.append(
                    {
                        "text": text,
                        "source": f"reddit:post",
                        "post_id": pid,
                        "collected_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
        time.sleep(1.5)

    return rows[:target]


def dedupe_rows(rows: list[dict]) -> list[dict]:
    unique: dict[str, dict] = {}
    for row in rows:
        key = row["text"].lower()
        if key not in unique:
            unique[key] = row
    return list(unique.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect r/nba posts for TakeMeter")
    parser.add_argument("--target", type=int, default=250, help="Target number of raw examples")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "raw_posts.csv",
    )
    args = parser.parse_args()

    seen: set[str] = set()
    print(f"Collecting up to {args.target} examples from r/{SUBREDDIT}...")

    try:
        rows = collect_from_pullpush(args.target, seen)
        print(f"  PullPush collected: {len(rows)}")
    except requests.RequestException as exc:
        print(f"  PullPush failed: {exc}")
        rows = []

    if len(rows) < args.target // 2:
        print("  Trying Reddit direct fallback...")
        rows.extend(collect_from_reddit_direct(args.target - len(rows), seen))

    final_rows = dedupe_rows(rows)[: args.target]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "source", "post_id", "collected_at"])
        writer.writeheader()
        writer.writerows(final_rows)

    print(f"Saved {len(final_rows)} examples to {args.output}")
    print("Next: label each row -> data/labeled_dataset.csv (columns: text, label, notes)")


if __name__ == "__main__":
    main()
